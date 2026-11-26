import { expect, test } from "bun:test";
import {
  Agent,
  type DurableStore,
  Flow,
  MemoryStore,
  MemoryStoreData,
  NonDeterminismError,
  ReActWithWakeups,
  type Run,
  ScriptedProvider,
  Tools,
  node,
  scheduleWakeupTool,
} from "../src/index.ts";
import { CountingMemory, Crash, CrashingMemory, World, build, call, crashes } from "./helpers.ts";

/** first -> second (checkpoint before it) -> END; counts how often each node body runs. */
function counted(options: { crashInSecond: boolean; state?: () => unknown }) {
  const calls = { first: 0, second: 0 };
  let crash = options.crashInSecond;
  const first = node(
    async (runtime) => {
      calls.first += 1;
      await runtime.complete();
      runtime.run.state.note = options.state ? options.state() : { plan: "two steps" };
    },
    { name: "first" },
  );
  const second = node(
    async (runtime) => {
      calls.second += 1;
      if (crash) {
        crash = false;
        throw new Crash(); // The process dies here: the run stays "running".
      }
      const response = await runtime.complete();
      runtime.finish(`${response.text} after ${JSON.stringify(runtime.run.state.note)}`);
    },
    { name: "second", checkpoint: true },
  );
  first.then(second);
  return { calls, loop: new Flow(first) };
}

function agentWith(store: DurableStore, loop: Flow): Agent {
  return new Agent({
    provider: new ScriptedProvider([
      { text: "planned", usage: { inputTokens: 5 } },
      { text: "done", usage: { inputTokens: 7 } },
    ]),
    loop,
    store,
    name: "worker",
    leaseTtl: 60,
  });
}

test("recovery restarts at the checkpoint, without re-running earlier nodes", async () => {
  const store = new MemoryStore();
  const { calls, loop } = counted({ crashInSecond: true });
  const agent = agentWith(store, loop);
  await expect(agent.run("Go", { sessionId: "s1" })).rejects.toThrow(Crash);
  expect(store.runs({ sessionId: "s1" })[0].status).toBe("running");
  const run = await agent.recover("s1", { force: true });
  expect(run.status).toBe("completed");
  expect(run.output).toBe('done after {"plan":"two steps"}');
  expect(calls).toEqual({ first: 1, second: 2 });
  expect(run.usage).toEqual({ inputTokens: 12 });
  expect(run.modelCalls).toBe(2);
  expect(run.messages.map((item) => item.role)).toEqual(["system", "user", "assistant", "assistant"]);
  const recovered = run.events.find((event) => event.kind === "run_recovered");
  expect(recovered?.data.fromCheckpoint).toBe(2);
  // The recovered run's own events start at the checkpoint: turn 1 was not replayed.
  expect(run.events.filter((event) => event.kind === "model_response").map((event) => event.data.turn)).toEqual([2]);
  const logged = store.events("s1").map((stored) => stored.event.kind);
  expect(logged.filter((kind) => kind === "checkpoint").length).toBe(1);
  expect(logged.filter((kind) => kind === "model_response").length).toBe(2);
});

test("state that is not plain JSON skips the checkpoint; recovery replays from the start", async () => {
  const store = new MemoryStore();
  const { calls, loop } = counted({ crashInSecond: true, state: () => new Map([["plan", 1]]) });
  const agent = agentWith(store, loop);
  await expect(agent.run("Go", { sessionId: "s1" })).rejects.toThrow(Crash);
  const run = await agent.recover("s1", { force: true });
  expect(run.status).toBe("completed");
  expect(calls).toEqual({ first: 2, second: 2 });
  const skipped = store.events("s1").find((stored) => stored.event.kind === "checkpoint_skipped");
  expect(skipped?.event.data.reason).toContain("run.state.note is a Map");
  expect(run.events.find((event) => event.kind === "run_recovered")?.data.fromCheckpoint).toBeUndefined();
});

test("replay up to a seq before the checkpoint ignores it; a full replay starts from it", async () => {
  const store = new MemoryStore();
  const { calls, loop } = counted({ crashInSecond: false });
  const agent = agentWith(store, loop);
  const run = await agent.run("Go", { sessionId: "s1" });
  const early = await agent.replay("s1", run.id, { untilSeq: 1 });
  expect(early.reason).toBe("untilSeq");
  expect(early.run.messages.at(-1)?.content).toBe("planned");
  expect(calls.first).toBe(2);
  const full = await agent.replay("s1", run.id);
  expect(full.run.output).toBe(run.output);
  expect(calls.first).toBe(2); // Not re-run: the full replay restored the checkpoint.
});

test("a flow without the checkpoint's node cannot recover, and the run stays recoverable", async () => {
  const store = new MemoryStore();
  const { loop } = counted({ crashInSecond: true });
  await expect(agentWith(store, loop).run("Go", { sessionId: "s1" })).rejects.toThrow(Crash);
  const other = node(async (runtime) => runtime.finish("x"), { name: "other" });
  await expect(agentWith(store, new Flow(other)).recover("s1", { force: true, allowFingerprintChange: true })).rejects.toThrow(
    NonDeterminismError,
  );
  expect(store.runs({ sessionId: "s1" })[0].status).toBe("running");
  const run = await agentWith(store, counted({ crashInSecond: false }).loop).recover("s1", { force: true });
  expect(run.status).toBe("completed");
});

test("compaction writes a checkpoint, and crashes after it recover from there", async () => {
  const reference = new CountingMemory();
  const expected = await build("react_with_summary", reference, new World()).run("Do the task", { sessionId: "s1" });
  expect(expected.events.filter((event) => event.kind === "context_compaction").length).toBe(2);
  expect(expected.events.filter((event) => event.kind === "checkpoint").length).toBe(2);
  let fromCheckpoint = 0;
  for (let crashAt = 1; crashAt <= reference.written; crashAt++) {
    const data = new MemoryStoreData();
    const world = new World();
    await crashes(build("react_with_summary", new CrashingMemory(data, { crashAt, after: true }), world).run("Do the task", { sessionId: "s1" }), {
      mayFinish: true,
    });
    const store = new MemoryStore(data);
    const [record] = store.runs({ sessionId: "s1" });
    if (record?.status !== "running") continue;
    const run = await build("react_with_summary", store, world).recover("s1", { force: true });
    expect(run.output).toBe(expected.output);
    expect(run.usage).toEqual(expected.usage);
    // An unsafe call in flight at the crash may end as "Outcome unknown"; it is never sent twice.
    expect(world.unsafe.length).toBeLessThanOrEqual(1);
    if (run.events.find((event) => event.kind === "run_recovered")?.data.fromCheckpoint) fromCheckpoint += 1;
  }
  expect(fromCheckpoint).toBeGreaterThan(0);
});

test("a run recovered while sleeping restarts at the wakeup node", async () => {
  const make = (store: DurableStore) =>
    new Agent({
      provider: new ScriptedProvider([
        { toolCalls: [call("w", "schedule_wakeup", { delaySeconds: 0.05, prompt: "check again" })] },
        { text: "checked" },
      ]),
      tools: new Tools([scheduleWakeupTool({ minDelay: 0 })]),
      loop: new ReActWithWakeups({ minDelay: 0 }),
      store,
      name: "watcher",
      leaseTtl: 60,
    });
  const reference = new CountingMemory();
  const expected = await make(reference).run("Watch", { sessionId: "s1" });
  expect(expected.events.filter((event) => event.kind === "checkpoint").map((event) => event.data.node)).toEqual(["wakeup"]);
  let resumedSleeps = 0;
  for (let crashAt = 1; crashAt <= reference.written; crashAt++) {
    const data = new MemoryStoreData();
    await crashes(make(new CrashingMemory(data, { crashAt, after: true })).run("Watch", { sessionId: "s1" }), { mayFinish: true });
    const store = new MemoryStore(data);
    const [record] = store.runs({ sessionId: "s1" });
    if (record?.status !== "running") continue;
    const run: Run = await make(store).recover("s1", { force: true });
    expect(run.output).toBe(expected.output);
    expect(run.messages).toEqual(expected.messages);
    if (run.events.some((event) => event.kind === "command_retried" && event.data.command === "suspend")) {
      resumedSleeps += 1;
      expect(run.events.find((event) => event.kind === "run_recovered")?.data.fromCheckpoint).toBeGreaterThan(0);
      expect(run.events.find((event) => event.kind === "step")?.data.node).toBe("wakeup");
    }
  }
  expect(resumedSleeps).toBeGreaterThan(0);
});
