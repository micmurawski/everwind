import { expect, test } from "bun:test";
import { appendFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { Agent, FileSessionStore, type Message, ReAct, ScriptedProvider, SessionLogError, type StreamChunk, Tools, tool } from "../src/index.ts";
import { call, tmpDir } from "./helpers.ts";

/** Answers with every user message so far, and attaches provider metadata. */
class Remember {
  async *createMessage(_system: string, messages: Message[], _metadata?: unknown, _options?: unknown): AsyncGenerator<StreamChunk> {
    yield { type: "text", text: messages.filter((item) => item.role === "user").map((item) => item.content).join(" | ") };
    yield { type: "metadata", metadata: { minimaxContent: [{ type: "thinking", signature: "opaque" }] } };
  }
}

test("resuming from disk keeps metadata and appends to the log", async () => {
  const directory = tmpDir();
  const agent = new Agent({ provider: new Remember(), sessions: new FileSessionStore(join(directory, "sessions")), logDir: join(directory, "logs") });
  const first = await agent.run("First", { sessionId: "conversation" });
  const original = readFileSync(first.logPath as string, "utf-8");
  first.messages[2].metadata = {}; // A returned Run cannot mutate the saved checkpoint.
  const recreated = new Agent({ provider: new Remember(), sessions: new FileSessionStore(join(directory, "sessions")) });
  const second = await recreated.run("Second", { sessionId: first.sessionId });
  expect([second.status, second.output]).toEqual(["completed", "First | Second"]);
  expect(second.sessionId).toBe(first.sessionId);
  expect(second.id).not.toBe(first.id);
  expect((second.messages[2].metadata.minimaxContent as { signature: string }[])[0].signature).toBe("opaque");
  expect(second.logPath).toBe(first.logPath);
  const text = readFileSync(second.logPath as string, "utf-8");
  expect(text.startsWith(original)).toBe(true);
  const rows = text.trim().split("\n").map((line) => JSON.parse(line));
  expect(new Set(rows.map((row) => row.runId))).toEqual(new Set([first.id, second.id]));
  for (const run of [first, second]) {
    expect(rows.filter((row) => row.runId === run.id).map((row) => row.sequence)).toEqual(run.events.map((_, index) => index));
  }
  expect(() => second.writeEvents(second.logPath as string)).toThrow(/overwrite/);
});

for (const disk of [false, true]) {
  test(`simultaneous continuations are serialized (${disk ? "disk" : "memory"})`, async () => {
    const directory = tmpDir();
    let entered!: () => void;
    const enteredPromise = new Promise<void>((resolve) => (entered = resolve));
    let release!: () => void;
    const released = new Promise<void>((resolve) => (release = resolve));
    class BlockOnce extends Remember {
      override async *createMessage(system: string, messages: Message[]) {
        if (messages.at(-1)?.content === "Second") {
          entered();
          await released;
        }
        yield* super.createMessage(system, messages);
      }
    }
    let agent = new Agent({ provider: new BlockOnce() });
    if (disk) agent = agent.with({ sessions: new FileSessionStore(directory) });
    const first = await agent.run("First");
    const other = disk ? agent.with({ sessions: new FileSessionStore(directory) }) : agent;
    const second = agent.run("Second", { sessionId: first.sessionId });
    await enteredPromise;
    let thirdDone = false;
    const third = other.run("Third", { sessionId: first.sessionId }).then((run) => {
      thirdDone = true;
      return run;
    });
    await Bun.sleep(30);
    expect(thirdDone).toBe(false);
    release();
    const [result2, result3] = await Promise.all([second, third]);
    expect(result2.output).toBe("First | Second");
    expect(result3.output).toBe("First | Second | Third");
  });
}

test("a cancelled tool is checkpointed without re-executing", async () => {
  const effects: string[] = [];
  let entered!: () => void;
  const started = new Promise<void>((resolve) => (entered = resolve));
  const effect = tool({
    name: "effect",
    run: async (_args, ctx) => {
      effects.push("started");
      entered();
      await new Promise((_, reject) => ctx.signal.addEventListener("abort", () => reject(ctx.signal.reason)));
    },
  });
  const agent = new Agent({ provider: new ScriptedProvider([{ toolCalls: [call("effect", "effect")] }]), tools: new Tools([effect]) });
  const controller = new AbortController();
  const pending = agent.run("First", { sessionId: "interrupted", signal: controller.signal });
  await started;
  controller.abort();
  await expect(pending).rejects.toThrow();
  const resumed = await agent.with({ provider: new Remember() }).run("Inspect status", { sessionId: "interrupted" });
  expect(resumed.status).toBe("completed");
  const result = resumed.messages.find((item) => item.role === "tool");
  expect(result?.isError).toBe(true);
  expect(result?.content).toContain("Outcome unknown");
  expect(effects).toEqual(["started"]);
});

test("resume checks configuration, history and log integrity", async () => {
  const directory = tmpDir();
  const agent = new Agent({ provider: new Remember(), logDir: directory });
  const run = await agent.run("First");
  await expect(agent.run("Second", { sessionId: run.sessionId, history: [{ role: "user", content: "Other" }] })).rejects.toThrow(/history/);
  await expect(agent.with({ name: "other" }).run("Second", { sessionId: run.sessionId })).rejects.toThrow(/configuration/);
  await expect(agent.run("Second", { sessionId: run.sessionId, logDir: join(directory, "other") })).rejects.toThrow(/directory/);
  appendFileSync(run.logPath as string, '{"partial":');
  await expect(agent.run("Second", { sessionId: run.sessionId })).rejects.toThrow(SessionLogError);
});

test("a hard-crash checkpoint requires explicit recovery", async () => {
  const agent = new Agent({ provider: new Remember() });
  const run = await agent.run("First");
  const saved = agent.sessions.load(run.sessionId);
  agent.sessions.save({ ...(saved as NonNullable<typeof saved>), status: "running" });
  await expect(agent.run("Continue", { sessionId: run.sessionId })).rejects.toThrow(/before checkpointing/);
});

test("limits reset each invocation, and sessions are independent", async () => {
  const agent = new Agent({ provider: new Remember(), limits: { maxModelCalls: 1 } });
  const first = await agent.run("First");
  first.messages[2].metadata = {};
  const second = await agent.run("Second", { sessionId: first.sessionId });
  const separate = await agent.run("Separate");
  expect([first.modelCalls, second.modelCalls]).toEqual([1, 1]);
  expect(Object.keys(second.messages[2].metadata).length).toBeGreaterThan(0);
  expect(separate.output).toBe("Separate");
  expect([second.state, second.context]).toEqual([{}, {}]);
});

test("an interrupted parallel batch keeps the known results", async () => {
  let completed!: () => void;
  const done = new Promise<void>((resolve) => (completed = resolve));
  const fast = tool({
    name: "done",
    run: async () => {
      completed();
      return "effect succeeded";
    },
  });
  const slow = tool({
    name: "slow",
    run: async (_args, ctx) => {
      await done;
      await new Promise((_, reject) => ctx.signal.addEventListener("abort", () => reject(ctx.signal.reason)));
    },
  });
  const agent = new Agent({
    provider: new ScriptedProvider([{ toolCalls: [call("known", "done"), call("unknown", "slow")] }]),
    tools: new Tools([fast, slow]),
    loop: new ReAct({ toolConcurrency: 2 }),
    limits: { toolTimeout: 0.02 },
  });
  const run = await agent.run("Perform effects");
  expect(run.status).toBe("timeout");
  const resumed = await agent.with({ provider: new Remember() }).run("Inspect", { sessionId: run.sessionId });
  const results = Object.fromEntries(resumed.messages.filter((item) => item.role === "tool").map((item) => [item.toolCallId, item]));
  expect([results.known.content, results.known.isError]).toEqual(["effect succeeded", false]);
  expect(results.unknown.isError).toBe(true);
  expect(results.unknown.content).toContain("Outcome unknown");
});
