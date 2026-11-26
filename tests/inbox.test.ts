import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import {
  Agent,
  type DurableStore,
  type Message,
  MemoryStore,
  MemoryStoreData,
  ReActWithWakeups,
  type Run,
  ScriptedProvider,
  SqliteStore,
  type StreamChunk,
  Tools,
  scheduleWakeupTool,
  tool,
} from "../src/index.ts";
import { CrashingMemory, call, crashes, tmpDir } from "./helpers.ts";

/** Answers with the user messages seen so far, so a delivered note is visible in the output. */
class Echo {
  async *createMessage(_system: string, messages: Message[]): AsyncGenerator<StreamChunk> {
    yield { type: "text", text: messages.filter((item) => item.role === "user").map((item) => item.content).join(" | ") };
  }
}

const stores: Record<string, () => DurableStore> = {
  memory: () => new MemoryStore(),
  sqlite: () => new SqliteStore(join(tmpDir(), "inbox.db")),
};

describe("inbox storage", () => {
  for (const [name, make] of Object.entries(stores)) {
    test(name, async () => {
      const store = make();
      expect(await store.signal("s1", "message", { text: "a" })).toBe(1);
      expect(await store.signal("s1", "result", { ok: true })).toBe(2);
      expect(await store.signal("s2", "message", { text: "b" })).toBe(1);
      expect(await store.signal("s1", "message", { text: "c" })).toBe(3);
      expect(store.inbox("s1").map((item) => item.id)).toEqual([1, 2, 3]);
      expect(store.inbox("s1", { kinds: ["message"] }).map((item) => item.payload.text)).toEqual(["a", "c"]);
      expect(store.inbox("s1", { after: 1, limit: 1 }).map((item) => item.id)).toEqual([2]);
      expect(store.inbox("s1", { kinds: [] })).toEqual([]);
      expect(store.inbox("s1")[0]).toMatchObject({ sessionId: "s1", kind: "message", consumedBy: null });
      await expect(store.signal("s1", "")).rejects.toThrow("kind");
      await expect(store.signal("bad id!", "message")).rejects.toThrow();
    });
  }
});

test("a message sent while the agent works arrives before its next turn", async () => {
  const store = new MemoryStore();
  let agent: Agent;
  const slow = tool({
    name: "slow",
    effect: "read",
    run: async (_args, ctx) => {
      await agent.send(ctx.runtime.run.sessionId, "use the EU sandbox", { from: "operator" });
      return "done";
    },
  });
  const scripted = new ScriptedProvider([{ toolCalls: [call("s", "slow")] }]);
  const echo = new Echo();
  let turns = 0;
  agent = new Agent({
    provider: {
      createMessage: (...args: Parameters<Echo["createMessage"]>) => (turns++ === 0 ? scripted.createMessage(...args) : echo.createMessage(...args)),
    },
    tools: new Tools([slow]),
    loop: new ReActWithWakeups(),
    store,
  });
  const run = await agent.run("Plan the migration", { sessionId: "s1" });
  expect(run.status).toBe("completed");
  expect(run.output).toBe("Plan the migration | [operator] use the EU sandbox");
  expect(run.messages.map((item) => item.role)).toEqual(["system", "user", "assistant", "tool", "user", "assistant"]);
  expect(store.inbox("s1")).toMatchObject([{ id: 1, consumedBy: run.id }]);
  expect(run.events.filter((event) => event.kind === "received").map((event) => event.data.ids)).toEqual([[1]]);
});

test("messages sent before a run are delivered at its first gate, once", async () => {
  const store = new MemoryStore();
  const agent = new Agent({ provider: new Echo(), loop: new ReActWithWakeups(), store });
  await agent.send("s1", "context from yesterday");
  await store.signal("s1", "audit", { text: "not for the model" });
  const first = await agent.run("Continue", { sessionId: "s1" });
  expect(first.output).toBe("Continue | context from yesterday");
  const second = await agent.run("Again", { sessionId: "s1" });
  expect(second.output).toBe("Continue | context from yesterday | Again");
  // Other kinds stay pending for whoever waits for them.
  expect(store.inbox("s1", { pending: true }).map((item) => item.kind)).toEqual(["audit"]);
  expect(() => new Agent({ provider: new Echo() }).send("s1", "x")).toThrow("durable store");
});

test("a message ends a long sleep early; the wakeup prompt is not added", async () => {
  const store = new MemoryStore();
  const agent = new Agent({
    provider: new ScriptedProvider([
      { toolCalls: [call("w", "schedule_wakeup", { delaySeconds: 3600, prompt: "check CI" })] },
      { text: "Stopped watching" },
    ]),
    tools: new Tools([scheduleWakeupTool()]),
    loop: new ReActWithWakeups(),
    store,
  });
  setTimeout(() => void agent.send("s1", "stop watching, I merged it"), 50);
  const started = performance.now();
  const run = await agent.run("Watch CI", { sessionId: "s1" });
  expect(performance.now() - started).toBeLessThan(1000);
  expect(run.status).toBe("completed");
  expect(run.events.find((event) => event.kind === "woke")?.data).toMatchObject({ by: "signal", messageId: 1, kind: "message" });
  const users = run.messages.filter((item) => item.role === "user").map((item) => item.content);
  expect(users).toEqual(["Watch CI", "stop watching, I merged it"]);
});

test("suspend needs something to wake it, and a run without a store has no inbox", async () => {
  const agent = new Agent({
    provider: new ScriptedProvider([{ toolCalls: [call("w", "schedule_wakeup", { delaySeconds: 0.01, prompt: "again" })] }, { text: "ok" }]),
    tools: new Tools([scheduleWakeupTool({ minDelay: 0 })]),
    loop: new ReActWithWakeups({ minDelay: 0 }),
  });
  const run = await agent.run("Go");
  expect(run.status).toBe("completed");
  expect(run.messages.filter((item) => item.role === "user").map((item) => item.content)).toEqual(["Go", "again"]);
  const bare = new Agent({
    provider: new Echo(),
    loop: { run: async (runtime) => void (await runtime.suspend({ wakeOn: ["message"] })) },
  });
  expect((await bare.run("x")).error).toContain("no inbox");
});

test("a replay returns the messages the run received, not newer ones", async () => {
  const store = new MemoryStore();
  const agent = new Agent({ provider: new Echo(), loop: new ReActWithWakeups(), store });
  await agent.send("s1", "first note");
  const run = await agent.run("Go", { sessionId: "s1" });
  await agent.send("s1", "later note");
  const replay = await agent.replay("s1", run.id);
  expect(replay.run.output).toBe("Go | first note");
  expect(store.inbox("s1", { pending: true }).map((item) => item.payload.text)).toEqual(["later note"]);
});

test("crash at every write: each message is delivered exactly once", async () => {
  const build = (store: DurableStore) =>
    new Agent({ provider: new Echo(), loop: new ReActWithWakeups(), store, name: "worker", leaseTtl: 60 });
  const seed = async (data: MemoryStoreData) => {
    const store = new MemoryStore(data);
    await store.signal("s1", "message", { text: "note A" });
    await store.signal("s1", "message", { text: "note B" });
  };
  const referenceData = new MemoryStoreData();
  await seed(referenceData);
  const counting = new CrashingMemory(referenceData, { crashAt: 1_000_000, after: true });
  const expected = await build(counting).run("Go", { sessionId: "s1" });
  expect(expected.output).toBe("Go | note A | note B");
  let recovered = 0;
  for (let crashAt = 1; crashAt <= counting.written; crashAt++) {
    for (const after of [false, true]) {
      const data = new MemoryStoreData();
      await seed(data);
      await crashes(build(new CrashingMemory(data, { crashAt, after })).run("Go", { sessionId: "s1" }), { mayFinish: true });
      const store = new MemoryStore(data);
      const [record] = store.runs({ sessionId: "s1" });
      if (record === undefined) {
        expect(store.inbox("s1", { pending: true }).length).toBe(2);
        continue;
      }
      let run: Run | null = null;
      if (record.status === "running") {
        recovered += 1;
        run = await build(store).recover("s1", { force: true });
        expect(run.output).toBe(expected.output);
      }
      const consumed = store.inbox("s1").map((item) => item.consumedBy);
      expect(consumed).toEqual([record.runId, record.runId]);
    }
  }
  expect(recovered).toBeGreaterThan(0);
});

test("SQLite: a message written by another process wakes a suspended run", async () => {
  const path = join(tmpDir(), "shared.db");
  const agentStore = new SqliteStore(path);
  const otherProcess = new SqliteStore(path); // A second connection: no in-process listener.
  const agent = new Agent({
    provider: new ScriptedProvider([{ toolCalls: [call("w", "schedule_wakeup", { delaySeconds: 3600, prompt: "later" })] }, { text: "woke" }]),
    tools: new Tools([scheduleWakeupTool()]),
    loop: new ReActWithWakeups(),
    store: agentStore,
  });
  setTimeout(() => void otherProcess.signal("s1", "message", { text: "wake up" }), 50);
  const started = performance.now();
  const run = await agent.run("Wait", { sessionId: "s1" });
  expect(run.status).toBe("completed");
  expect(performance.now() - started).toBeLessThan(2000);
  expect(agentStore.inbox("s1")[0].consumedBy).toBe(run.id);
});
