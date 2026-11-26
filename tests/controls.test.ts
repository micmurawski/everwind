import { expect, test } from "bun:test";
import { join } from "node:path";
import {
  Agent,
  type DurableStore,
  isPinned,
  MemoryStore,
  ReAct,
  ReActWithWakeups,
  ScriptedProvider,
  SqliteStore,
  Tools,
  Worker,
  askOperatorTool,
  scheduleWakeupTool,
  tool,
} from "../src/index.ts";
import { call, tmpDir } from "./helpers.ts";

const nodesOf = (events: { kind: string; data: Record<string, unknown> }[], kind: string) =>
  events.filter((event) => event.kind === kind).map((event) => event.data.node);

function worker(store: DurableStore, onWork: (sessionId: string) => unknown, loop = new ReAct()) {
  let calls = 0;
  const work = tool({
    name: "work",
    effect: "read",
    run: async (_args, ctx) => {
      calls += 1;
      await onWork(ctx.runtime.run.sessionId);
      return "worked";
    },
  });
  const agent = new Agent({
    provider: new ScriptedProvider([{ toolCalls: [call("w", "work")] }, { text: "done" }]),
    tools: new Tools([work]),
    loop,
    store,
    name: "worker",
    leaseTtl: 60,
    limits: { timeout: 0.2 },
  });
  return { agent, calls: () => calls };
}

test("hold stops a run at its next step until resumed; holding does not use its timeout", async () => {
  const store = new MemoryStore();
  const { agent } = worker(store, async (sessionId) => {
    await agent.hold(sessionId, "checking the bill");
    setTimeout(() => void agent.resume(sessionId), 300);
  });
  const started = performance.now();
  const run = await agent.run("Go", { sessionId: "s1" });
  expect(run.status).toBe("completed"); // 300 ms held, timeout 200 ms
  expect(performance.now() - started).toBeGreaterThanOrEqual(290);
  expect(nodesOf(run.events, "held")).toEqual(["call_llm"]);
  expect(run.events.find((event) => event.kind === "held")?.data.note).toBe("checking the bill");
  expect(nodesOf(run.events, "resumed")).toEqual(["call_llm"]);
  expect(store.control("s1")?.state).toBe("run");
});

test("a hold set before a run applies at its first step; freezing a held run stops it", async () => {
  const store = new MemoryStore();
  const { agent, calls } = worker(store, () => {});
  await agent.hold("s1");
  const running = agent.run("Go", { sessionId: "s1" });
  await Bun.sleep(50);
  expect(store.runs({ sessionId: "s1" })[0].status).toBe("running");
  expect(calls()).toBe(0);
  await agent.freeze("s1");
  const run = await running;
  expect(run.status).toBe("frozen");
  expect(nodesOf(run.events, "held")).toEqual(["call_llm"]);
  const resumed = await agent.recover("s1");
  expect(resumed.status).toBe("completed");
  expect(calls()).toBe(1);
});

test("freeze stops at the next step, releases the run, and recover() continues from a checkpoint", async () => {
  const store = new MemoryStore();
  const { agent, calls } = worker(store, (sessionId) => agent.freeze(sessionId, "night"));
  const run = await agent.run("Go", { sessionId: "s1" });
  expect(run.status).toBe("frozen");
  expect(run.error).toBeNull();
  const [record] = store.runs({ sessionId: "s1" });
  expect(record).toMatchObject({ status: "running", owner: null, leaseExpiresAt: 0 });
  const logged = store.events("s1").map((stored) => stored.event);
  expect(logged.find((event) => event.kind === "frozen")?.data).toEqual({ node: "call_llm", note: "night" });
  expect(logged.some((event) => event.kind === "checkpoint")).toBe(true);
  // Recovery at startup leaves frozen sessions alone.
  expect(await new Worker(store, [agent]).recoverAll()).toEqual([]);
  // recover() needs no force: the lease was released.
  const resumed = await agent.recover("s1");
  expect(resumed.status).toBe("completed");
  expect(resumed.output).toBe("done");
  expect(resumed.events.find((event) => event.kind === "run_recovered")?.data.fromCheckpoint).toBeGreaterThan(0);
  expect(calls()).toBe(1);
  expect(store.control("s1")?.state).toBe("run");
});

test("a sleeping run freezes at once and later sleeps only what is left", async () => {
  const store = new MemoryStore();
  const agent = new Agent({
    provider: new ScriptedProvider([{ toolCalls: [call("w", "schedule_wakeup", { delaySeconds: 0.4, prompt: "check" })] }, { text: "checked" }]),
    tools: new Tools([scheduleWakeupTool({ minDelay: 0 })]),
    loop: new ReActWithWakeups({ minDelay: 0 }),
    store,
    name: "watcher",
  });
  setTimeout(() => void agent.freeze("s1"), 50);
  const started = performance.now();
  const run = await agent.run("Watch", { sessionId: "s1" });
  expect(run.status).toBe("frozen");
  expect(performance.now() - started).toBeLessThan(300);
  const resumed = await agent.recover("s1");
  expect(resumed.status).toBe("completed");
  expect(performance.now() - started).toBeGreaterThanOrEqual(390);
  expect(resumed.events.some((event) => event.kind === "command_retried" && event.data.command === "suspend")).toBe(true);
  expect(resumed.messages.filter((item) => item.role === "user").map((item) => item.content)).toEqual(["Watch", "check"]);
});

test("ask_operator waits for a reply and adds it as a pinned message", async () => {
  const store = new MemoryStore();
  const agent = new Agent({
    provider: new ScriptedProvider([
      { toolCalls: [call("q", "ask_operator", { question: "EU first, or all regions?", options: ["EU first", "all"] })] },
      { text: "Planning EU first" },
    ]),
    tools: new Tools([askOperatorTool()]),
    loop: new ReActWithWakeups(),
    store,
  });
  setTimeout(() => void agent.reply("s1", "EU first, behind a flag"), 50);
  const run = await agent.run("Plan the migration", { sessionId: "s1" });
  expect(run.status).toBe("completed");
  expect(run.events.find((event) => event.kind === "question")?.data).toEqual({
    question: "EU first, or all regions?",
    options: ["EU first", "all"],
    callId: "q",
  });
  expect(run.events.find((event) => event.kind === "answered")?.data).toEqual({ callId: "q", messageId: 1 });
  const reply = run.messages.find((item) => item.content === "[operator] EU first, behind a flag");
  expect(reply !== undefined && isPinned(reply)).toBe(true);
  expect(store.inbox("s1")[0]).toMatchObject({ kind: "reply", consumedBy: run.id });
});

test("ask_operator with a timeout lets the model decide", async () => {
  const agent = new Agent({
    provider: new ScriptedProvider([{ toolCalls: [call("q", "ask_operator", { question: "Which region?" })] }, { text: "Chose EU" }]),
    tools: new Tools([askOperatorTool()]),
    loop: new ReActWithWakeups({ ask: { timeout: 0.05 } }),
    store: new MemoryStore(),
  });
  const run = await agent.run("Plan", { sessionId: "s1" });
  expect(run.status).toBe("completed");
  expect(run.messages.at(-2)?.content).toContain("No reply from the operator within 0.05 seconds");
});

test("SQLite: another process can hold, resume and freeze a run", async () => {
  const path = join(tmpDir(), "control.db");
  const remote = new SqliteStore(path); // The control plane's own connection.
  const store = new SqliteStore(path);
  const agent = new Agent({
    provider: new ScriptedProvider([{ toolCalls: [call("w", "schedule_wakeup", { delaySeconds: 30, prompt: "later" })] }, { text: "x" }]),
    tools: new Tools([scheduleWakeupTool({ minDelay: 0 })]),
    loop: new ReActWithWakeups({ minDelay: 0 }),
    store,
    name: "watcher",
  });
  setTimeout(() => void remote.setControl("s1", "freeze"), 50);
  const started = performance.now();
  const run = await agent.run("Watch", { sessionId: "s1" });
  expect(run.status).toBe("frozen");
  expect(performance.now() - started).toBeLessThan(1500);
  expect(remote.runs({ sessionId: "s1" })[0]).toMatchObject({ status: "running", owner: null });
  expect(remote.control("s1")?.state).toBe("freeze");
});

test("controls need a durable store", () => {
  const agent = new Agent({ provider: new ScriptedProvider([]) });
  expect(() => agent.hold("s1")).toThrow("durable store");
  expect(() => agent.reply("s1", "x")).toThrow("durable store");
});
