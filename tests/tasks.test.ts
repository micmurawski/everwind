import { afterEach, expect, test } from "bun:test";
import {
  Agent,
  type DurableStore,
  type Message,
  MemoryStore,
  type ModelResponseInput,
  ReActWithWakeups,
  ScriptedProvider,
  type StreamChunk,
  Supervisor,
  TASK_FINISHED,
  Tools,
  askOperatorTool,
  createTask,
  isPinned,
  project,
  task,
  taskTool,
  tasks,
  waitTool,
} from "../src/index.ts";
import { call } from "./helpers.ts";

/** Answers "done: <its last user message>", with some usage. */
class Echo {
  async *createMessage(_system: string, messages: Message[]): AsyncGenerator<StreamChunk> {
    const last = [...messages].reverse().find((item) => item.role === "user")?.content ?? "";
    yield { type: "text", text: `done: ${last.split("\n").at(-1)}` };
    yield { type: "usage", inputTokens: 100, outputTokens: 10 };
  }
}

const loop = () => new ReActWithWakeups({ minDelay: 0 });
const taskTools = new Tools([taskTool(), waitTool(), askOperatorTool()]);
const agentOf = (store: DurableStore, name: string, provider: Agent["provider"]) =>
  new Agent({ name, provider, tools: taskTools, loop: loop(), store, leaseTtl: 60, limits: { maxModelCalls: 20 } });

let supervisors: Supervisor[] = [];
afterEach(async () => {
  for (const supervisor of supervisors) await supervisor.stop({ freeze: true });
  supervisors = [];
});
function start(store: DurableStore, agents: Agent[]): Supervisor {
  const supervisor = new Supervisor(store, agents, { pollInterval: 0.02 });
  supervisors.push(supervisor);
  supervisor.start();
  return supervisor;
}
async function until(check: () => boolean, ms = 4000) {
  const end = Date.now() + ms;
  while (!check()) {
    if (Date.now() > end) throw new Error("condition not met in time");
    await Bun.sleep(10);
  }
}

const delegating: ModelResponseInput[] = [
  { toolCalls: [call("a", "task", { agent: "researcher", prompt: "Compare EU providers", title: "providers" }), call("b", "task", { agent: "coder", prompt: "Draft the adapter", title: "adapter" })] },
  { toolCalls: [call("w", "wait_for_tasks")] },
  ...Array.from({ length: 6 }, (_, i) => ({ text: `Final plan, after ${i + 1} reports.` })),
];

test("a planner delegates to two agents, waits, and finishes only when both have reported", async () => {
  const store = new MemoryStore();
  const planner = agentOf(store, "planner", new ScriptedProvider(delegating));
  const supervisor = start(store, [planner, agentOf(store, "researcher", new Echo()), agentOf(store, "coder", new Echo())]);
  const created = await supervisor.createProject({ title: "Checkout", goal: "Plan the checkout revamp", agent: "planner", agents: ["researcher", "coder"], maxDepth: 2 });
  const done = await supervisor.idle(created.id);
  expect(done.status).toBe("done");
  expect(done.result).toStartWith("Final plan");
  const all = tasks(store, { project: created.id });
  expect(all.map((item) => [item.depth, item.agent, item.status])).toEqual([[0, "planner", "done"], [1, "researcher", "done"], [1, "coder", "done"]]);
  expect(all[1]).toMatchObject({ title: "providers", result: "done: Compare EU providers", parent: created.rootTask, activations: 1 });
  expect(done.spentTokens).toBe(220); // The two children's usage; the scripted planner reports none.
  // The planner saw both outcomes as pinned messages, and its session is the root task's.
  const root = store.load(created.rootTask);
  const reports = (root?.messages ?? []).filter((item) => item.role === "user" && item.content.startsWith("[task "));
  expect(reports.length).toBe(2);
  expect(reports.every(isPinned)).toBe(true);
  expect(store.inbox(created.rootTask, { kinds: [TASK_FINISHED] }).every((item) => item.consumedBy !== null)).toBe(true);
  // Children run as tasks: their session's scope and frame say where they are.
  const child = store.load(all[1].id);
  expect(child?.frame).toContain(`task ${all[1].id} in project ${created.id}: a subtask of ${created.rootTask}, at depth 1 of at most 2`);
});

test("at the project's maximum depth, the task tools are hidden", async () => {
  const store = new MemoryStore();
  const planner = agentOf(store, "planner", new ScriptedProvider([{ toolCalls: [call("a", "task", { agent: "researcher", prompt: "Look it up" })] }, { toolCalls: [call("w", "wait_for_tasks")] }, { text: "ok" }, { text: "ok" }]));
  const researcher = agentOf(store, "researcher", new Echo());
  const supervisor = start(store, [planner, researcher]);
  const created = await supervisor.createProject({ title: "Shallow", goal: "Go", agent: "planner", agents: ["researcher"], maxDepth: 1 });
  await supervisor.idle(created.id);
  const child = tasks(store, { project: created.id })[1];
  const offered = store
    .events(child.id)
    .filter((stored) => stored.event.kind === "model_request")
    .map((stored) => (stored.event.data.request as any).tools.map((spec: any) => spec.name));
  expect(offered[0]).toEqual(["ask_operator"]);
  expect(store.load(child.id)?.frame).toContain("You cannot create subtasks");
  // The store refuses a deeper task even if asked directly.
  await expect(createTask(store, { id: "t_deeper", parent: child.id, agent: "researcher", title: "x", prompt: "x" })).rejects.toThrow("at most 1");
});

test("a follow-up runs a finished task again, on the same session, with the prompt pinned", async () => {
  const store = new MemoryStore();
  const planner = agentOf(
    store,
    "planner",
    new ScriptedProvider([
      { toolCalls: [call("a", "task", { agent: "coder", prompt: "Write v1" })] },
      { toolCalls: [call("w", "wait_for_tasks")] },
      { toolCalls: [call("f", "task", { id: "__CHILD__", prompt: "Now add tests" })] },
      { toolCalls: [call("w2", "wait_for_tasks")] },
      { text: "Shipped with tests." },
      { text: "Shipped with tests." },
    ]),
  );
  // The follow-up names the child's id, known only once it exists: patch the script when the child appears.
  const script = (planner.provider as ScriptedProvider).responses as any[];
  const coder = agentOf(store, "coder", new Echo());
  const supervisor = start(store, [planner, coder]);
  const created = await supervisor.createProject({ title: "Follow", goal: "Ship it", agent: "planner", agents: ["coder"] });
  await until(() => tasks(store, { project: created.id }).length === 2);
  script[2].toolCalls[0].arguments.id = tasks(store, { project: created.id })[1].id;
  const done = await supervisor.idle(created.id);
  expect(done.status).toBe("done");
  const child = tasks(store, { project: created.id })[1];
  expect(child).toMatchObject({ activations: 2, status: "done", result: "done: Now add tests" });
  const messages = store.load(child.id)?.messages ?? [];
  expect(messages.filter((item) => item.role === "user").map((item) => [item.content, isPinned(item)])).toEqual([
    ["Write v1", false],
    ["Now add tests", true],
  ]);
});

test("fork gives the same agent the conversation, and another agent a transcript", async () => {
  const store = new MemoryStore();
  const script = new ScriptedProvider([
    { text: "", toolCalls: [call("x", "ask_operator", { question: "Region?" })] },
    { toolCalls: [call("a", "task", { agent: "planner", prompt: "Plan EU", context: "fork" }), call("b", "task", { agent: "coder", prompt: "Code EU", context: "fork" })] },
    { toolCalls: [call("w", "wait_for_tasks")] },
    ...Array.from({ length: 4 }, () => ({ text: "Done." })),
  ]);
  // The forked child is a planner too: it answers its own prompt instead of following the root's script.
  const planner = agentOf(store, "planner", {
    createMessage(system: string, messages: Message[], metadata?: any, options?: any) {
      const last = [...messages].reverse().find((item) => item.role === "user")?.content;
      return last === "Plan EU" ? new Echo().createMessage(system, messages) : script.createMessage(system, messages, metadata, options);
    },
  });
  const supervisor = start(store, [planner, agentOf(store, "coder", new Echo())]);
  const created = await supervisor.createProject({ title: "Fork", goal: "Plan by region", agent: "planner", agents: ["planner", "coder"], maxDepth: 1 });
  await until(() => project(store, created.id)?.status === "active" && store.inbox(created.rootTask).length === 0 && tasks(store, { project: created.id }).length === 1 && store.events(created.rootTask).some((s) => s.event.kind === "question"));
  await planner.reply(created.rootTask, "EU");
  await until(() => tasks(store, { project: created.id }).length === 3);
  const [, same, other] = tasks(store, { project: created.id });
  expect(same.history?.map((item) => item.role)).toEqual(["user", "assistant", "tool", "user"]);
  expect(same.history?.[3].content).toBe("[operator] EU");
  expect(other.history?.map((item) => item.role)).toEqual(["user", "assistant"]);
  expect(other.history?.[0].content).toContain("transcript");
  expect(other.history?.[0].content).toContain("[operator] EU");
  await supervisor.idle(created.id);
  // The forked planner's run started from the history, then its own prompt.
  expect(store.load(same.id)?.messages.map((item) => item.role).slice(0, 5)).toEqual(["user", "assistant", "tool", "user", "user"]);
});

test("cancelling a task tells its parent and freezes it; the parent finishes", async () => {
  const store = new MemoryStore();
  const planner = agentOf(store, "planner", new ScriptedProvider(delegating));
  const coder = agentOf(store, "coder", new ScriptedProvider([{ toolCalls: [call("q", "ask_operator", { question: "Which ORM?" })] }, { text: "never" }]));
  const supervisor = start(store, [planner, agentOf(store, "researcher", new Echo()), coder]);
  const created = await supervisor.createProject({ title: "Cancel", goal: "Plan", agent: "planner", agents: ["researcher", "coder"], maxDepth: 1 });
  await until(() => tasks(store, { project: created.id }).some((item) => item.agent === "coder" && store.events(item.id).some((s) => s.event.kind === "question")));
  const coderTask = tasks(store, { project: created.id }).find((item) => item.agent === "coder") as NonNullable<ReturnType<typeof task>>;
  await supervisor.cancel(coderTask.id, "not needed");
  const done = await supervisor.idle(created.id);
  expect(done.status).toBe("done");
  expect(task(store, coderTask.id)).toMatchObject({ status: "cancelled", error: "not needed" });
  expect(store.control(coderTask.id)?.state).toBe("freeze");
  const reports = store.inbox(created.rootTask, { kinds: [TASK_FINISHED] }).map((item) => item.payload.status);
  expect(reports.sort()).toEqual(["cancelled", "done"]);
});

test("a project pauses when its budget is spent; ready tasks wait until it resumes", async () => {
  const store = new MemoryStore();
  const planner = agentOf(store, "planner", new ScriptedProvider(delegating));
  const supervisor = new Supervisor(store, [planner, agentOf(store, "researcher", new Echo()), agentOf(store, "coder", new Echo())], { pollInterval: 0.02, maxActive: 2 });
  supervisors.push(supervisor);
  const created = await supervisor.createProject({ title: "Budget", goal: "Plan", agent: "planner", agents: ["researcher", "coder"], maxDepth: 1, budgetTokens: 100 });
  supervisor.start();
  await until(() => project(store, created.id)?.status === "paused");
  const paused = project(store, created.id);
  expect(paused?.reason).toContain("budget");
  // The root waits for the other child, which is not admitted while paused.
  await Bun.sleep(100);
  expect(tasks(store, { project: created.id }).filter((item) => item.status === "ready").length).toBe(1);
  const { setProjectStatus } = await import("../src/index.ts");
  await setProjectStatus(store, created.id, "active");
  expect((await supervisor.idle(created.id)).status).toBe("done");
});

test("maxTasks bounds a project", async () => {
  const store = new MemoryStore();
  const planner = agentOf(store, "planner", new ScriptedProvider(delegating));
  const supervisor = new Supervisor(store, [planner, agentOf(store, "researcher", new Echo()), agentOf(store, "coder", new Echo())]);
  const created = await supervisor.createProject({ title: "Small", goal: "Plan", agent: "planner", agents: ["researcher"], maxTasks: 2 });
  await createTask(store, { id: "t_one", parent: created.rootTask, agent: "researcher", title: "one", prompt: "one" });
  await expect(createTask(store, { id: "t_two", parent: created.rootTask, agent: "researcher", title: "two", prompt: "two" })).rejects.toThrow("maximum of 2");
  await expect(createTask(store, { id: "t_x", parent: created.rootTask, agent: "coder", title: "x", prompt: "x" })).rejects.toThrow("not one of this project's");
  // Creating it again with the same id returns it unchanged (a retried tool call).
  expect((await createTask(store, { id: "t_one", parent: created.rootTask, agent: "researcher", title: "changed", prompt: "changed" })).title).toBe("one");
  expect(() => supervisor.createProject({ title: "x", goal: "x", agent: "nobody" })).toThrow("no agent");
});

test("a frozen task finished outside the supervisor is picked up by the next one", async () => {
  const store = new MemoryStore();
  const planner = agentOf(store, "planner", new ScriptedProvider(delegating));
  const coder = agentOf(store, "coder", new ScriptedProvider([{ toolCalls: [call("q", "ask_operator", { question: "Which ORM?" })] }, { text: "Used Drizzle" }]));
  const agents = [planner, agentOf(store, "researcher", new Echo()), coder];
  const first = start(store, agents);
  const created = await first.createProject({ title: "Handover", goal: "Plan", agent: "planner", agents: ["researcher", "coder"], maxDepth: 1 });
  await until(() => tasks(store, { project: created.id }).some((item) => item.agent === "coder" && store.events(item.id).some((s) => s.event.kind === "question")));
  await first.stop({ freeze: true });
  supervisors = [];
  const coderTask = tasks(store, { project: created.id }).find((item) => item.agent === "coder") as NonNullable<ReturnType<typeof task>>;
  expect(task(store, coderTask.id)?.status).toBe("running");
  // Someone (the desk) recovers it by hand and answers.
  await coder.reply(coderTask.id, "Drizzle");
  expect((await coder.recover(coderTask.id)).status).toBe("completed");
  // The next supervisor records the outcome and tells the root, which it recovers too.
  await planner.store?.setControl(created.rootTask, "run");
  const second = start(store, agents);
  const done = await second.idle(created.id, 10);
  expect(task(store, coderTask.id)).toMatchObject({ status: "done", result: "Used Drizzle" });
  expect(done.status).toBe("done");
});
