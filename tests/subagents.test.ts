import { expect, test } from "bun:test";
import { Agent, AgentJobs, Mailbox, type Message, Run, Runtime, ScriptedProvider, type StreamChunk, Tools, limits, subagentTools } from "../src/index.ts";
import { call } from "./helpers.ts";

/** Answers with every user message so far, so a continued session is visible. */
class Remember {
  constructor(readonly prefix: string) {}

  async *createMessage(_system: string, messages: Message[]): AsyncGenerator<StreamChunk> {
    yield { type: "text", text: `${this.prefix}: ${messages.filter((item) => item.role === "user").map((item) => item.content).join(" | ")}` };
  }
}

const researcher = new Agent({ provider: new Remember("research"), name: "researcher" });
const coder = new Agent({ provider: new Remember("code"), name: "coder" });
const tools = subagentTools(
  { researcher: { agent: researcher, description: "Finds facts" }, coder },
  { defaultType: "coder" },
);

function parent(context: Record<string, unknown> = {}): Runtime {
  return new Runtime(new Run({ context }), new Remember("parent"), new Tools(), limits());
}

const invoke = (runtime: Runtime, name: string, args: Record<string, unknown>) =>
  tools.invoke({ id: "c", name, arguments: args as never, metadata: {} }, { runtime, values: runtime.run.context });

test("the agent tool lists the types and only takes model arguments", () => {
  const spec = tools.get("agent")?.spec.parameters as any;
  expect(Object.keys(spec.properties).sort()).toEqual(["description", "prompt", "runInBackground", "sessionId", "subagentType"]);
  expect(spec.required.sort()).toEqual(["description", "prompt"]);
  expect(spec.properties.subagentType.enum).toEqual(["researcher", "coder"]);
  expect(tools.get("agent")?.description).toContain("- researcher: Finds facts");
  expect(() => subagentTools({})).toThrow("at least one");
  expect(() => subagentTools({ coder }, { defaultType: "nobody" })).toThrow("Unknown default");
});

test("a child of the chosen type answers, and its session can be continued", async () => {
  const runtime = parent();
  const first = (await invoke(runtime, "agent", { description: "Look up", prompt: "Find X", subagentType: "researcher" })).result as any;
  expect(first).toMatchObject({ subagentType: "researcher", status: "completed", output: "research: Find X" });
  expect(first.continuation).toContain(`agent(subagentType="researcher", prompt="...", sessionId="${first.sessionId}")`);
  const second = (await invoke(runtime, "agent", { description: "More", prompt: "And Y", subagentType: "researcher", sessionId: first.sessionId }))
    .result as any;
  expect(second.output).toBe("research: Find X | And Y");
  const fallback = (await invoke(runtime, "agent", { description: "Write", prompt: "Code it" })).result as any;
  expect(fallback).toMatchObject({ subagentType: "coder", output: "code: Code it" });
  expect((await invoke(runtime, "agent", { description: "x", prompt: "x", subagentType: "nobody" })).error).toContain("Invalid arguments");
});

test("background delegation returns a job, and list_agents shows it", async () => {
  const jobs = new AgentJobs(new Mailbox());
  const runtime = parent({ agentJobs: jobs });
  const handle = (await invoke(runtime, "agent", { description: "Bg task", prompt: "Later", runInBackground: true })).result as any;
  expect(handle).toMatchObject({ status: "running", subagentType: "coder", description: "Bg task" });
  expect(await jobs.mailbox.get()).toMatchObject({ kind: "AgentFinished", jobId: handle.jobId, status: "completed", output: "code: Later" });
  const listed = (await invoke(runtime, "list_agents", {})).result as any;
  expect(listed.agents).toEqual([
    { subagentType: "researcher", agent: "researcher", description: "Finds facts", default: false },
    { subagentType: "coder", agent: "coder", description: null, default: true },
  ]);
  expect(listed.jobs.map((job: any) => job.jobId)).toEqual([handle.jobId]);
  expect(((await invoke(parent(), "list_agents", {})).result as any).jobs).toBeUndefined();
  await jobs.close();
});

test("a parent agent delegates through the tool", async () => {
  const boss = new Agent({
    provider: new ScriptedProvider([
      { toolCalls: [call("d", "agent", { description: "Research", prompt: "Facts please", subagentType: "researcher" })] },
      { text: "Delegated" },
    ]),
    tools,
  });
  const run = await boss.run("Get facts");
  expect(run.status).toBe("completed");
  expect(JSON.parse(run.messages[3].content).output).toBe("research: Facts please");
});
