import { expect, test } from "bun:test";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import {
  AGENT_JOB_TOOLS,
  Agent,
  AgentJobs,
  Mailbox,
  type Message,
  Run,
  Runtime,
  ScriptedProvider,
  SessionLogError,
  type StreamChunk,
  Tools,
  agentTool,
  limits,
  tool,
} from "../src/index.ts";
import { call, tmpDir } from "./helpers.ts";

class Remember {
  async *createMessage(_system: string, messages: Message[]): AsyncGenerator<StreamChunk> {
    yield { type: "text", text: messages.filter((item) => item.role === "user").map((item) => item.content).join(" | ") };
  }
}

const runtime = (context: Record<string, unknown> = {}) => new Runtime(new Run({ context }), new Remember(), new Tools(), limits());
const result = (value: { result: unknown }) => value.result as Record<string, any>;

test("the factory has only model arguments and keeps the child's history", async () => {
  const directory = tmpDir();
  const child = new Agent({ provider: new Remember(), name: "researcher", logDir: directory });
  const ask = agentTool(child, { name: "ask_researcher", description: "Research a topic." });
  expect(Object.keys(ask.spec.parameters.properties as object).sort()).toEqual(["prompt", "sessionId"]);
  expect(ask.spec.parameters.required).toEqual(["prompt"]);
  const parent = runtime();
  const first = result(await ask.invoke({ prompt: "First" }, { runtime: parent }));
  const second = result(await ask.invoke({ prompt: "Second", sessionId: first.sessionId }, { runtime: parent }));
  expect(second.output).toBe("First | Second");
  expect(first.sessionId).toBe(second.sessionId);
  expect(first.runId).not.toBe(second.runId);
  expect(second.continuation).toContain("ask_researcher(");
  const file = readdirSync(directory).find((name) => name.endsWith(".jsonl")) as string;
  const rows = readFileSync(join(directory, file), "utf-8").trim().split("\n").map((line) => JSON.parse(line));
  const headers = rows.filter((row) => row.kind === "run_start").map((row) => row.data);
  expect(headers.length).toBe(2);
  expect(headers.every((row) => row.parentSessionId === parent.run.sessionId && row.parentRunId === parent.run.id)).toBe(true);
  const bad = await ask.invoke({ prompt: "Bad", runtime: {} }, { runtime: parent });
  expect(bad.error).toContain("Invalid arguments");
});

test("a parent ReAct can call a child, and the child's context is explicit", async () => {
  const readSecret = tool({ name: "read_secret", run: (_args, ctx) => ctx.get<string>("secret") });
  const child = new Agent({ provider: new ScriptedProvider([{ toolCalls: [call("read", "read_secret")] }, { text: "Done" }]), tools: new Tools([readSecret]) });
  const ask = agentTool(child, { context: { secret: "child-only" } });
  const parent = new Agent({
    provider: new ScriptedProvider([{ toolCalls: [call("delegate", ask.name, { prompt: "Work" })] }, { text: "Finished delegating" }]),
    tools: new Tools([ask]),
  });
  const run = await parent.run("Start", { context: { secret: "parent-only", runtime: "spoofed" } });
  expect(run.status).toBe("completed");
  const delegated = JSON.parse(run.messages.find((item) => item.role === "tool")?.content as string);
  const checkpoint = child.sessions.load(delegated.sessionId);
  expect(checkpoint?.messages.find((item) => item.role === "tool")?.content).toBe("child-only");
});

for (const outcome of ["completed", "failed", "timeout", "cancelled", "early_cancel"]) {
  test(`background returns immediately, and every outcome wakes the parent (${outcome})`, async () => {
    let entered = false;
    let markEntered!: () => void;
    const enteredPromise = new Promise<void>((resolve) => (markEntered = resolve));
    let release!: () => void;
    const released = new Promise<void>((resolve) => (release = resolve));
    const slow = {
      async *createMessage(): AsyncGenerator<StreamChunk> {
        entered = true;
        markEntered();
        await released;
        if (outcome === "failed") throw new Error("provider failed");
        yield { type: "text", text: "Done" };
      },
    };
    const mailbox = new Mailbox();
    const jobs = new AgentJobs(mailbox);
    try {
      const child = new Agent({ provider: slow, limits: { timeout: outcome === "timeout" ? 0.03 : 2 } });
      const ask = agentTool(child, { execution: "background" });
      const parent = runtime();
      const context = { runtime: parent, values: { agentJobs: jobs } };
      const handle = result(await ask.invoke({ prompt: "Work" }, context));
      expect(handle.status).toBe("running");
      expect(entered).toBe(false); // The tool did not wait for the child.
      if (outcome !== "early_cancel") await enteredPromise;
      if (outcome === "cancelled" || outcome === "early_cancel") {
        const cancelled = await AGENT_JOB_TOOLS.invoke(
          { id: "cancel", name: "cancel_agent_job", arguments: { jobId: handle.jobId }, metadata: {} },
          context,
        );
        expect(result(cancelled).status).toBe("stopping");
      } else if (outcome !== "timeout") {
        release();
      }
      const event = await mailbox.get();
      expect(event.kind).toBe("AgentFinished");
      expect(event.status).toBe(outcome === "early_cancel" ? "cancelled" : outcome);
      expect([event.jobId, event.sessionId]).toEqual([handle.jobId, handle.sessionId]);
      await jobs.waitAll();
      jobs.cancel(handle.jobId, parent.run.sessionId);
      expect(mailbox.size).toBe(0);
      expect(() => jobs.inspect(handle.jobId, "stranger")).toThrow(/another parent/);
      if (outcome === "completed") {
        expect(event.output).toBe("Done");
        expect(event.runId).toBeTruthy();
      }
    } finally {
      release();
      await jobs.close();
    }
  });
}

test("background continuation and supervisor cleanup", async () => {
  const mailbox = new Mailbox();
  const jobs = new AgentJobs(mailbox);
  const parent = runtime();
  const ask = agentTool(new Agent({ provider: new Remember() }), { execution: "background" });
  const context = { runtime: parent, values: { agentJobs: jobs } };
  const first = result(await ask.invoke({ prompt: "A" }, context));
  await jobs.waitAll();
  await mailbox.get();
  await ask.invoke({ prompt: "B", sessionId: first.sessionId }, context);
  expect((await mailbox.get()).output).toBe("A | B");
  await jobs.close();
  const closed = await ask.invoke({ prompt: "C" }, context);
  expect(closed.error).toContain("closed");
});

test("closing the supervisor cancels children and queues their outcome", async () => {
  let markEntered!: () => void;
  const entered = new Promise<void>((resolve) => (markEntered = resolve));
  const slow = {
    async *createMessage(_s: string, _m: Message[], _meta?: unknown, options?: { signal?: AbortSignal }): AsyncGenerator<StreamChunk> {
      markEntered();
      await new Promise((_, reject) => options?.signal?.addEventListener("abort", () => reject(options.signal?.reason)));
      yield { type: "text", text: "unreachable" };
    },
  };
  const mailbox = new Mailbox();
  const jobs = new AgentJobs(mailbox);
  const ask = agentTool(new Agent({ provider: slow }), { execution: "background" });
  await ask.invoke({ prompt: "A" }, { runtime: runtime(), values: { agentJobs: jobs } });
  await entered;
  await jobs.close();
  expect((await mailbox.get()).status).toBe("cancelled");
  expect(mailbox.size).toBe(0);
});

test("the same parent session is rejected, and the mode is validated", async () => {
  const child = new Agent({ provider: new Remember() });
  const parent = runtime();
  const rejected = await agentTool(child).invoke({ prompt: "Work", sessionId: parent.run.sessionId }, { runtime: parent });
  expect(rejected.error).toContain("active parent");
  expect(() => agentTool(child, { execution: "typo" as "wait" })).toThrow(/execution/);
});

test("a delegation logging failure does not launch the child", async () => {
  const mailbox = new Mailbox();
  const jobs = new AgentJobs(mailbox);
  const parent = runtime();
  parent.run.eventSink = () => {
    throw new SessionLogError("disk unavailable");
  };
  const ask = agentTool(new Agent({ provider: new Remember() }), { execution: "background" });
  await expect(ask.invoke({ prompt: "Work" }, { runtime: parent, values: { agentJobs: jobs } })).rejects.toThrow(/disk unavailable/);
  expect(jobs.listJobs(parent.run.sessionId)).toEqual([]);
  await jobs.close();
  expect(mailbox.size).toBe(0);
});
