/** Durable runs: crash at every journal write, then recover and compare. */

import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import {
  Agent,
  INTERRUPTED_CONTENT,
  MemoryStore,
  MemoryStoreData,
  type Message,
  NonDeterminismError,
  PlanThenAct,
  type Run,
  RunLeaseHeld,
  RunOwnershipLost,
  ScriptedProvider,
  SqliteStore,
  Worker,
} from "../src/index.ts";
import {
  CountingMemory,
  Crash,
  CrashingMemory,
  CrashingSqlite,
  REACT_SCRIPT,
  SCENARIOS,
  World,
  build,
  crashes,
  makeTools,
  tmpDir,
} from "./helpers.ts";

async function reference(name: string): Promise<{ run: Run; writes: number }> {
  const store = new CountingMemory();
  const run = await build(name, store, new World()).run("Do the task", { sessionId: "s1" });
  expect(run.status).toBe("completed");
  return { run, writes: store.written };
}

function withoutResults(messages: readonly Message[], ids: Set<string>): Message[] {
  return messages.filter((item) => !(item.role === "tool" && ids.has(item.toolCallId as string)));
}

describe("crash at every write, then recover", () => {
  for (const name of Object.keys(SCENARIOS).sort()) {
    test(name, async () => {
      const { run: expected, writes } = await reference(name);
      if (name === "react_with_summary") expect(expected.events.some((event) => event.kind === "context_compaction")).toBe(true);
      let recovered = 0;
      for (let crashAt = 1; crashAt <= writes; crashAt++) {
        for (const after of [false, true]) {
          const world = new World();
          const data = new MemoryStoreData();
          // Crashing after the very last write is indistinguishable from finishing.
          await crashes(build(name, new CrashingMemory(data, { crashAt, after }), world).run("Do the task", { sessionId: "s1" }), {
            mayFinish: after && crashAt === writes,
          });
          const store = new MemoryStore(data); // A new process over the surviving data.
          const records = store.runs({ sessionId: "s1" });
          if (!records.length) {
            expect(world.unsafe).toEqual([]);
            expect(world.idempotentKeys).toEqual([]);
            continue;
          }
          let messages: Message[];
          let output: string | null;
          if (records[0].status === "running") {
            recovered += 1;
            await expect(build(name, store, world).recover("s1")).rejects.toThrow(RunLeaseHeld);
            const run = await build(name, store, world).recover("s1", { force: true });
            expect(run.id).toBe(records[0].runId);
            expect(run.status).toBe("completed");
            expect(run.usage).toEqual(expected.usage);
            [messages, output] = [run.messages.slice(1), run.output];
          } else {
            [messages, output] = [store.load("s1")?.messages ?? [], store.runs()[0].result?.output as string];
          }
          expect(output).toBe(expected.output);
          const synthetic = messages.filter((item) => item.role === "tool" && item.content === INTERRUPTED_CONTENT);
          const ids = new Set(synthetic.map((item) => item.toolCallId as string));
          expect(withoutResults(messages, ids)).toEqual(withoutResults(expected.messages.slice(1), ids));
          // The unsafe effect happened at most once; without a recorded result it was not repeated.
          expect(world.unsafe.length).toBeLessThanOrEqual(1);
          if (synthetic.length) expect(synthetic.map((item) => item.toolCallId)).toEqual(["c"]);
          // Idempotent retries reuse one key.
          expect(new Set(world.idempotentKeys).size).toBeLessThanOrEqual(1);
          expect(store.runs({ status: "running" })).toEqual([]);
          expect(store.load("s1")?.status).toBe("completed");
        }
      }
      expect(recovered).toBeGreaterThan(writes);
    });
  }
});

test("recovery replays instead of repeating effects (SQLite)", async () => {
  const world = new World();
  const path = join(tmpDir(), "durable.db");
  // Writes: createRun, model(s, c), tools a,b (s), a(c), b(c), model(s, c), send(s), send(c) -> crash after 10.
  const crashing = new CrashingSqlite(path, { crashAt: 10, after: true });
  await crashes(build("react", crashing, world).run("Do the task", { sessionId: "s1" }));
  crashing.close();
  expect([world.unsafe, world.reads, world.idempotentKeys.length]).toEqual([["hello"], 1, 1]);
  const store = new SqliteStore(path);
  const run = await build("react", store, world).recover("s1", { force: true });
  expect([run.status, run.output]).toEqual(["completed", "All done"]);
  expect([world.unsafe, world.reads, world.idempotentKeys.length]).toEqual([["hello"], 1, 1]);
  expect([store.runs()[0].status, store.runs()[0].epoch]).toEqual(["completed", 2]);
  // The completed session continues as a normal conversation.
  const follow = build("react", store, world).with({ provider: new ScriptedProvider([{ text: "Again" }]) });
  const again = await follow.run("And now?", { sessionId: "s1" });
  expect(again.output).toBe("Again");
  expect(again.messages.filter((item) => item.role === "user").at(-1)?.content).toBe("And now?");
  store.close();
});

test("in-flight tools follow their effect kind", async () => {
  const world = new World();
  const data = new MemoryStoreData();
  // createRun, model(s), model(c), batch scheduled (one append) -> crash before the first result.
  await crashes(build("react_parallel", new CrashingMemory(data, { crashAt: 5, after: false }), world).run("Do the task", { sessionId: "s1" }));
  const executed = [world.reads, world.idempotentKeys.length, world.unsafe.length];
  const run = await build("react_parallel", new MemoryStore(data), world).recover("s1", { force: true });
  const results = Object.fromEntries(run.messages.filter((item) => item.role === "tool").map((item) => [item.toolCallId, item]));
  expect(run.status).toBe("completed");
  expect(world.reads).toBe(executed[0] + 1); // read: executed again
  expect(world.idempotentKeys.length).toBe(executed[1] + 1);
  expect(new Set(world.idempotentKeys).size).toBe(1);
  expect(world.unsafe.length).toBe(executed[2]); // unsafe: never repeated
  expect([results.c.isError, results.c.content]).toEqual([true, INTERRUPTED_CONTENT]);
  expect([results.a.isError, results.b.isError]).toEqual([false, false]);
});

test("a stale owner stops without finishing", async () => {
  const world = new World();
  const store = new MemoryStore();
  let entered!: () => void;
  const enteredPromise = new Promise<void>((resolve) => (entered = resolve));
  let release!: () => void;
  const released = new Promise<void>((resolve) => (release = resolve));

  class Slow extends ScriptedProvider {
    override async *createMessage(...args: Parameters<ScriptedProvider["createMessage"]>) {
      entered();
      await released;
      yield* super.createMessage(...args);
    }
  }

  const first = build("react", store, world).with({ provider: new Slow(REACT_SCRIPT) });
  const pending = first.run("Do the task", { sessionId: "s1" });
  await enteredPromise;
  // A second owner (another process) takes over while the first is mid-call.
  const other = new MemoryStore(store.data);
  const record = other.runs({ status: "running" })[0];
  await expect(other.claimRun("s1", record.runId, { ttl: 60 })).rejects.toThrow(RunLeaseHeld);
  expect(await other.claimRun("s1", record.runId, { ttl: 60, force: true })).toBe(2);
  release();
  await expect(pending).rejects.toThrow(RunOwnershipLost);
  expect([store.runs()[0].status, store.runs()[0].epoch]).toEqual(["running", 2]);
  expect(world.reads).toBe(0);
});

test("divergent code raises and keeps the run recoverable", async () => {
  const world = new World();
  const data = new MemoryStoreData();
  await crashes(build("plan_then_act", new CrashingMemory(data, { crashAt: 4, after: true }), world).run("Do the task", { sessionId: "s1" }));
  const store = new MemoryStore(data);
  const changed = build("plan_then_act", store, world).with({ loop: new PlanThenAct({ planningPrompt: "Plan differently" }) });
  await expect(changed.recover("s1", { force: true })).rejects.toThrow(NonDeterminismError);
  expect(store.runs()[0].status).toBe("running");
  const run = await build("plan_then_act", store, world).recover("s1", { force: true });
  expect([run.status, run.output]).toEqual(["completed", "All done"]);
});

test("an interrupted session must be recovered before continuing", async () => {
  const world = new World();
  const data = new MemoryStoreData();
  await crashes(build("react", new CrashingMemory(data, { crashAt: 3, after: true }), world).run("Task", { sessionId: "s1" }));
  const store = new MemoryStore(data);
  await expect(build("react", store, world).run("Next", { sessionId: "s1" })).rejects.toThrow(/recover/);
  await expect(build("react", store, world).with({ name: "other" }).recover("s1", { force: true })).rejects.toThrow(/configuration/);
  await expect(build("react", store, world).recover("missing")).rejects.toThrow(/no interrupted run/);
  await expect(new Agent({ provider: new ScriptedProvider([]) }).recover("s1")).rejects.toThrow(/durable store/);
});

test("two hundred concurrent runs recover after a crash", async () => {
  const world = new World();
  const path = join(tmpDir(), "durable.db");
  const crashing = new CrashingSqlite(path, { crashAt: 1500, after: false });
  const agent = build("react", crashing, world);
  const results = await Promise.allSettled(Array.from({ length: 200 }, (_, index) => agent.run("Do the task", { sessionId: `s${index}` })));
  expect(results.every((result) => result.status === "rejected" && result.reason instanceof Crash)).toBe(true);
  crashing.close();
  const store = new SqliteStore(path);
  const interrupted = store.runs({ status: "running" });
  expect(interrupted.length).toBeGreaterThan(0);
  const recovered = await new Worker(store, [build("react", store, world)]).recoverAll({ force: true });
  expect(recovered.every((run) => !(run instanceof Error) && run.status === "completed" && run.output === "All done")).toBe(true);
  expect(recovered.length).toBe(interrupted.length);
  expect(store.runs({ status: "running" })).toEqual([]);
  // Each run's unsafe tool ran at most once in total.
  expect(world.unsafe.length).toBeLessThanOrEqual(store.runs().length);
  store.close();
});

test("the timeout budget counts active time only", async () => {
  const world = new World();
  const data = new MemoryStoreData();
  await crashes(build("react", new CrashingMemory(data, { crashAt: 3, after: true }), world).run("Task", { sessionId: "s1" }));
  const store = new MemoryStore(data);
  const record = store.runs()[0];
  data.runs.set(`s1\u0000${record.runId}`, { ...record, activeSeconds: 1000 });
  const run = await build("react", store, world).with({ limits: { timeout: 999 } }).recover("s1", { force: true });
  expect(run.status).toBe("timeout");
  expect(store.runs()[0].status).toBe("timeout");
});

test("non-durable agents are unchanged", async () => {
  const agent = new Agent({ provider: new ScriptedProvider(REACT_SCRIPT), tools: makeTools(new World()) });
  const run = await agent.run("Do the task");
  expect(run.status).toBe("completed");
  expect(agent.store).toBeNull();
  expect(agent.sessions.load(run.sessionId)?.status).toBe("completed");
});
