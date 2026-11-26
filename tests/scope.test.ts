import { expect, test } from "bun:test";
import { z } from "zod";
import {
  Agent,
  type JsonObject,
  MemoryStore,
  type Message,
  OlderMessages,
  ReActWithWakeups,
  Remove,
  Run,
  Runtime,
  ScriptedProvider,
  type StreamChunk,
  Tools,
  canSpawn,
  isPinned,
  limits,
  message,
  taskFrame,
  taskScope,
  tool,
} from "../src/index.ts";
import { Crash, call } from "./helpers.ts";

/** A tool the way a task system would write it: hidden at the depth limit, agents narrowed by scope. */
const spawn = tool({
  name: "spawn",
  description: (scope) => `Create a subtask for one of: ${(scope.agents as string[]).join(", ")}.`,
  available: canSpawn,
  input: (scope) => z.object({ agent: z.enum(scope.agents as [string, ...string[]]), prompt: z.string() }),
  run: ({ agent, prompt }, ctx) => `${agent} <- ${prompt} (from ${ctx.scope.task})`,
});
const echo = tool({ name: "echo", effect: "read", input: z.object({ text: z.string() }), run: ({ text }) => text });
const tools = new Tools([spawn, echo]);

const mid = taskScope({ project: "p1", task: "t_a", parent: "t_root", depth: 1, maxDepth: 2, agents: ["researcher", "coder"] });
const leaf = taskScope({ project: "p1", task: "t_a1", parent: "t_a", depth: 2, maxDepth: 2, agents: ["researcher"] });

/** The tool specs the model saw on each request. */
const seenTools = (run: Run) =>
  run.events.filter((event) => event.kind === "model_request").map((event) => (event.data.request as any).tools.map((spec: any) => spec.name));

test("binding a scope hides unavailable tools and narrows scoped schemas", async () => {
  expect(tools.bind(leaf).list().map((item) => item.name)).toEqual(["echo"]);
  const bound = tools.bind(mid).get("spawn");
  expect(bound?.description).toBe("Create a subtask for one of: researcher, coder.");
  expect((bound?.spec.parameters.properties as any).agent.enum).toEqual(["researcher", "coder"]);
  // Unbound, the tool is marked rather than described with a schema it does not have.
  expect(spawn.scoped).toBe(true);
  expect(tools.get("spawn")?.spec.parameters).toEqual({ scoped: true });
  expect(echo.bind(leaf)).toBe(echo);
  // Called directly, a scoped tool binds to the runtime's scope.
  const runtime = new Runtime(new Run({ scope: mid }), new ScriptedProvider([]), new Tools(), limits());
  expect((await spawn.invoke({ agent: "coder", prompt: "fix" }, { runtime })).result).toBe("coder <- fix (from t_a)");
  expect((await spawn.invoke({ agent: "writer", prompt: "x" }, { runtime })).error).toContain("Invalid arguments");
  const leafRuntime = new Runtime(new Run({ scope: leaf }), new ScriptedProvider([]), new Tools(), limits());
  expect((await spawn.invoke({ agent: "researcher", prompt: "x" }, { runtime: leafRuntime })).error).toContain("not available");
});

test("an agent's model sees only the tools its scope allows; a hidden tool is unknown", async () => {
  const agent = new Agent({
    provider: new ScriptedProvider([{ toolCalls: [call("s", "spawn", { agent: "researcher", prompt: "dig" })] }, { text: "done" }]),
    tools,
  });
  const atMid = await agent.run("Plan", { scope: mid });
  expect(seenTools(atMid)[0]).toEqual(["spawn", "echo"]);
  expect(atMid.messages[3].content).toBe("researcher <- dig (from t_a)");
  const atLeaf = await agent.run("Plan", { scope: leaf });
  expect(seenTools(atLeaf)[0]).toEqual(["echo"]);
  expect(atLeaf.messages[3].content).toContain('Unknown tool "spawn"');
  expect(canSpawn({})).toBe(false);
  expect(() => taskScope({ project: "p", task: "t", depth: 3, maxDepth: 2 })).toThrow("beyond maxDepth");
});

test("the frame is appended to the system prompt and fixed for the session", async () => {
  const store = new MemoryStore();
  const agent = new Agent({ provider: new ScriptedProvider([{ text: "one" }, { text: "two" }, { text: "three" }]), store, systemPrompt: "You plan." });
  const frame = taskFrame(mid);
  expect(frame).toContain("task t_a in project p1: a subtask of t_root, at depth 1 of at most 2");
  expect(frame).toContain("researcher, coder");
  expect(taskFrame(leaf)).toContain("You cannot create subtasks");
  const first = await agent.run("Goal: compare providers", { sessionId: "t_a", scope: mid, frame });
  expect(first.messages[0].content).toBe(`You plan.\n\n${frame}`);
  expect(store.load("t_a")).toMatchObject({ systemPrompt: "You plan.", frame, scope: mid });
  // A follow-up keeps the frame and the scope without repeating them.
  const second = await agent.run("Also check fees", { sessionId: "t_a" });
  expect(second.messages[0].content).toBe(`You plan.\n\n${frame}`);
  expect(second.scope).toEqual(mid);
  await expect(agent.run("x", { sessionId: "t_a", frame: "another frame" })).rejects.toThrow("frame is fixed");
  const third = await agent.run("Once more", { sessionId: "t_a", frame });
  expect(third.status).toBe("completed");
});

test("a recovered run keeps its scope and frame, so it binds the same tools", async () => {
  const store = new MemoryStore();
  let crash = true;
  const crashing = tool({
    name: "work",
    effect: "read",
    run: () => {
      if (crash) {
        crash = false;
        throw new Crash();
      }
      return "worked";
    },
  });
  const make = () =>
    new Agent({
      provider: new ScriptedProvider([{ toolCalls: [call("w", "work")] }, { text: "done" }]),
      tools: tools.with(crashing),
      store,
      name: "worker",
      leaseTtl: 60,
    });
  await expect(make().run("Go", { sessionId: "t_a", scope: mid, frame: taskFrame(mid) })).rejects.toThrow(Crash);
  const run = await make().recover("t_a", { force: true });
  expect(run.status).toBe("completed");
  expect(run.scope).toEqual(mid);
  expect(run.messages[0].content).toContain("# Task");
  expect(seenTools(run).at(-1)).toEqual(["spawn", "echo", "work"]);
});

test("pinned messages survive compaction, after its replacement", async () => {
  const runtime = new Runtime(
    new Run({
      messages: [
        message({ role: "system", content: "sys" }),
        message({ role: "user", content: "the task" }),
        message({ role: "assistant", content: "working 1" }),
        message({ role: "user", content: "follow-up: also check fees", metadata: { pinned: true } }),
        message({ role: "assistant", content: "working 2" }),
        message({ role: "user", content: "an ordinary note" }),
        message({ role: "assistant", content: "working 3" }),
        message({ role: "assistant", content: "latest" }),
      ],
    }),
    new ScriptedProvider([]),
    new Tools(),
    limits(),
  );
  const result = await runtime.compact({ selection: new OlderMessages(1), strategy: new Remove() });
  expect(result.applied).toBe(true);
  expect(runtime.run.messages.map((item) => item.content)).toEqual(["sys", "the task", "follow-up: also check fees", "latest"]);
  expect(result.replacement.map((item) => item.content)).toEqual(["follow-up: also check fees"]);
  // Only user messages can be pinned.
  expect(isPinned(message({ role: "assistant", content: "x", metadata: { pinned: true } }))).toBe(false);
});

test("the prompt, say() and inbox messages can be pinned", async () => {
  const store = new MemoryStore();
  class Echo {
    async *createMessage(_system: string, messages: Message[]): AsyncGenerator<StreamChunk> {
      yield { type: "text", text: messages.filter((item) => item.role === "user").map((item) => item.content).join(" | ") };
    }
  }
  const agent = new Agent({ provider: new Echo(), loop: new ReActWithWakeups(), store });
  await store.signal("t_a", "message", { text: "follow-up: add a rollback", pinned: true } as JsonObject);
  const run = await agent.run("Follow-up: write the plan", { sessionId: "t_a", pin: true });
  const users = run.messages.filter((item) => item.role === "user");
  expect(users.map((item) => [item.content, isPinned(item)])).toEqual([
    ["Follow-up: write the plan", true],
    ["follow-up: add a rollback", true],
  ]);
  const runtime = new Runtime(new Run({ messages: [message({ role: "system", content: "s" })] }), new Echo(), new Tools(), limits());
  runtime.say("keep me", { pinned: true });
  expect(isPinned(runtime.run.messages[1])).toBe(true);
  expect(runtime.run.events.at(-1)?.data).toEqual({ content: "keep me", pinned: true });
});
