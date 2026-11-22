/** Determinism APIs, configuration checks, inspection replay and forks. */

import { expect, test } from "bun:test";
import {
  Agent,
  ConfigurationChanged,
  type CreateMessageOptions,
  type Loop,
  MemoryStore,
  MemoryStoreData,
  type Message,
  NonDeterminismError,
  PlanThenAct,
  Run,
  Runtime,
  ScriptedProvider,
  Tools,
  VersionUnsupported,
  limits,
  type DurableStore,
} from "../src/index.ts";
import type { CreateMessageMetadata } from "../src/types.ts";
import { CountingMemory, CrashingMemory, REACT_SCRIPT, World, build, call, crashes, makeTools } from "./helpers.ts";

/** Records the last user message of every request it actually receives. */
class Watching extends ScriptedProvider {
  constructor(
    responses: ConstructorParameters<typeof ScriptedProvider>[0],
    readonly seen: string[],
  ) {
    super(responses);
  }

  override async *createMessage(system: string, messages: Message[], metadata?: CreateMessageMetadata, options?: CreateMessageOptions) {
    this.seen.push([...messages].reverse().find((item) => item.role === "user")?.content ?? "");
    yield* super.createMessage(system, messages, metadata, options);
  }
}

class Exploding extends ScriptedProvider {
  constructor() {
    super([]);
  }

  // biome-ignore lint/correctness/useYield: replay must not call the provider
  override async *createMessage(): AsyncGenerator<never> {
    throw new Error("replay must not call the provider");
  }
}

/** A loop whose behaviour depends on time and randomness, through the runtime. */
class Stamped implements Loop {
  async run(runtime: Runtime): Promise<void> {
    const stamp = `${runtime.uuid()} ${runtime.random()} ${runtime.now().toISOString()}`;
    runtime.say(stamp);
    await runtime.complete();
    await runtime.executeTools();
    await runtime.complete();
    runtime.finish(stamp);
  }
}

const STAMP_SCRIPT = [{ toolCalls: [call("c", "send", { text: "hello" })] }, { text: "done" }];

function stampedAgent(store: DurableStore, world: World, seen: string[], loop: Loop = new Stamped()): Agent {
  return new Agent({ provider: new Watching(STAMP_SCRIPT, seen), tools: makeTools(world), loop, name: "worker", store, leaseTtl: 60 });
}

test("side effects return recorded values after a crash", async () => {
  const counting = new CountingMemory();
  await stampedAgent(counting, new World(), []).run("Task", { sessionId: "s1" });
  const writes = counting.written;
  for (let crashAt = 1; crashAt <= writes; crashAt++) {
    for (const after of [false, true]) {
      const world = new World();
      const seen: string[] = [];
      const data = new MemoryStoreData();
      await crashes(stampedAgent(new CrashingMemory(data, { crashAt, after }), world, seen).run("Task", { sessionId: "s1" }), {
        mayFinish: after && crashAt === writes,
      });
      const store = new MemoryStore(data);
      let output: string;
      if (store.runs({ status: "running" }).length) output = (await stampedAgent(store, world, seen).recover("s1", { force: true })).output as string;
      else if (store.runs().length) output = store.runs()[0].result?.output as string;
      else continue;
      // Every model request in every attempt saw the one stamp the run finished with.
      expect(seen.length).toBeGreaterThan(0);
      expect(new Set(seen)).toEqual(new Set([output]));
      const [uuid, number, moment] = output.split(" ");
      expect(uuid.length).toBe(32);
      expect(Number(number) >= 0 && Number(number) < 1).toBe(true);
      expect(Number.isNaN(Date.parse(moment))).toBe(false);
    }
  }
});

test("helpers work without a journal", () => {
  const runtime = new Runtime(new Run(), new ScriptedProvider([]), new Tools(), limits());
  expect(runtime.uuid().length).toBe(32);
  expect(runtime.random()).toBeLessThan(1);
  expect(runtime.now()).toBeInstanceOf(Date);
  expect(runtime.version("change", 0, 3)).toBe(3);
  expect(() => runtime.version("change", 2, 1)).toThrow();
});

/** The same loop before (gate false) and after (gate true) a code change. */
class Versioned implements Loop {
  constructor(
    readonly gate = false,
    readonly minSupported = 0,
  ) {}

  async run(runtime: Runtime): Promise<void> {
    if (this.gate && runtime.version("double-check", this.minSupported, 1) === 1) runtime.say("Double-check your work.");
    await runtime.complete();
    await runtime.executeTools();
    await runtime.complete();
    runtime.finish(runtime.run.messages.at(-1)?.content ?? "");
  }
}

test("version() keeps old journals replayable", async () => {
  const world = new World();
  const seen: string[] = [];
  // Old code crashes after its first model call is recorded.
  let data = new MemoryStoreData();
  await crashes(stampedAgent(new CrashingMemory(data, { crashAt: 3, after: true }), world, seen, new Versioned()).run("Task", { sessionId: "old" }));
  let store = new MemoryStore(data);
  // New code still recovers the old run: version() returns 0, so no extra instruction.
  await expect(stampedAgent(store, world, seen, new Versioned(true, 1)).recover("old", { force: true })).rejects.toThrow(VersionUnsupported);
  expect(store.runs({ sessionId: "old" })[0].status).toBe("running");
  let run = await stampedAgent(store, world, seen, new Versioned(true)).recover("old", { force: true });
  expect(run.status).toBe("completed");
  expect(run.messages.map((item) => item.content)).not.toContain("Double-check your work.");
  // A new run of the new code takes the new path, and its own recovery replays it.
  data = new MemoryStoreData();
  await crashes(stampedAgent(new CrashingMemory(data, { crashAt: 4, after: true }), world, seen, new Versioned(true)).run("Task", { sessionId: "new" }));
  store = new MemoryStore(data);
  run = await stampedAgent(store, world, seen, new Versioned(true, 1)).recover("new", { force: true });
  expect(run.status).toBe("completed");
  expect(run.messages.map((item) => item.content)).toContain("Double-check your work.");
});

test("a changed configuration is named and can be allowed", async () => {
  const world = new World();
  const data = new MemoryStoreData();
  await crashes(build("react", new CrashingMemory(data, { crashAt: 6, after: true }), world).run("T", { sessionId: "s1" }));
  const store = new MemoryStore(data);
  class OtherProvider extends ScriptedProvider {}
  const changed = build("react", store, world).with({ provider: new OtherProvider(REACT_SCRIPT) });
  try {
    await changed.recover("s1", { force: true });
    throw new Error("expected ConfigurationChanged");
  } catch (error) {
    expect(error).toBeInstanceOf(ConfigurationChanged);
    expect((error as ConfigurationChanged).changed).toEqual(["provider"]);
  }
  const run = await changed.recover("s1", { force: true, allowFingerprintChange: true });
  expect([run.status, run.output]).toEqual(["completed", "All done"]);
});

test("replay inspects any step without effects", async () => {
  const world = new World();
  const store = new CountingMemory();
  const source = await build("react", store, world).run("Do the task", { sessionId: "s1" });
  const written = store.written;
  const effects = [world.reads, world.idempotentKeys.length, world.unsafe.length];
  const inspector = build("react", store, world).with({ provider: new Exploding() });

  const start = await inspector.replay("s1", source.id, { untilSeq: 0 });
  expect([start.commands, start.reason]).toEqual([0, "untilSeq"]);
  expect(start.run.messages.map((item) => item.role)).toEqual(["system", "user"]);
  const first = await inspector.replay("s1", source.id, { untilSeq: 1 });
  expect(first.commands).toBe(1);
  expect(first.pendingToolCalls.map((item) => item.id)).toEqual(["a", "b"]);
  // A parallel batch (commands 2 and 3) is replayed whole.
  const split = await inspector.replay("s1", source.id, { untilSeq: 2 });
  expect(split.commands).toBe(1);
  expect(split.run.messages.length).toBe(first.run.messages.length);
  const batch = await inspector.replay("s1", source.id, { untilSeq: 3 });
  expect(batch.commands).toBe(3);
  expect(batch.run.messages.filter((item) => item.role === "tool").map((item) => item.toolCallId)).toEqual(["a", "b"]);

  const full = await inspector.replay("s1", source.id);
  expect([full.reason, full.run.status, full.run.output]).toEqual(["finished", "completed", "All done"]);
  expect(full.run.messages).toEqual(source.messages);
  expect(full.run.usage).toEqual(source.usage);
  expect(store.written).toBe(written);
  expect([world.reads, world.idempotentKeys.length, world.unsafe.length]).toEqual(effects);
  await expect(inspector.with({ loop: new PlanThenAct() }).replay("s1", source.id)).rejects.toThrow(NonDeterminismError);
});

test("replay of an interrupted run stops at the end of the journal", async () => {
  const world = new World();
  const data = new MemoryStoreData();
  await crashes(build("react", new CrashingMemory(data, { crashAt: 6, after: true }), world).run("T", { sessionId: "s1" }));
  const store = new MemoryStore(data);
  const record = store.runs()[0];
  const replayed = await build("react", store, world).with({ provider: new Exploding() }).replay("s1", record.runId);
  expect([replayed.reason, replayed.run.status, replayed.commands]).toEqual(["endOfJournal", "running", 3]);
});

test("a fork reuses the prefix and diverges after it", async () => {
  const world = new World();
  const store = new MemoryStore();
  const source = await build("react", store, world).run("Do the task", { sessionId: "s1" });
  const effects = [world.reads, world.idempotentKeys.length, world.unsafe.length];
  // After the first tool batch (commands 1-3), try a different model and system prompt.
  const other = build("react", store, world).with({
    provider: new ScriptedProvider([{ text: "unused" }, { text: "Forked answer" }]),
    systemPrompt: "Be brief.",
  });
  const fork = await other.fork("s1", source.id, { atSeq: 3, newSessionId: "fork-1" });
  expect([fork.status, fork.output, fork.sessionId, fork.messages[0].content]).toEqual(["completed", "Forked answer", "fork-1", "Be brief."]);
  expect([world.reads, world.idempotentKeys.length, world.unsafe.length]).toEqual(effects); // prefix not re-executed
  expect(fork.messages.filter((item) => item.role === "tool").map((item) => item.toolCallId)).toEqual(["a", "b"]);
  expect(store.runs({ sessionId: "s1" })[0].result?.output).toBe("All done");
  expect(store.runs({ sessionId: "fork-1" })[0].start.forkedFrom).toEqual({ sessionId: "s1", runId: source.id, atSeq: 3 });
  expect(fork.events.some((event) => event.kind === "fork_live")).toBe(true);
  await expect(other.fork("s1", source.id, { atSeq: 3, newSessionId: "fork-1" })).rejects.toThrow(/already exists/);
  // A fork can be inspected like any run.
  const replayed = await other.with({ provider: new Exploding() }).replay("fork-1", fork.id);
  expect(replayed.run.output).toBe("Forked answer");
});

test("a fork is itself recoverable", async () => {
  const world = new World();
  const data = new MemoryStoreData();
  const source = await build("react", new MemoryStore(data), world).run("Do the task", { sessionId: "s1" });
  const script = [{ text: "unused" }, { toolCalls: [call("d", "send", { text: "forked" })] }, { text: "Forked answer" }];
  const forkAgent = (store: DurableStore) => build("react", store, world).with({ provider: new ScriptedProvider(script), systemPrompt: "Brief." });
  // Writes: createRun (with prefix), model(s), model(c) -> crash after the second model call starts.
  await crashes(forkAgent(new CrashingMemory(data, { crashAt: 2, after: true })).fork("s1", source.id, { atSeq: 3, newSessionId: "fork-1" }));
  const run = await forkAgent(new MemoryStore(data)).recover("fork-1", { force: true });
  expect([run.status, run.output]).toEqual(["completed", "Forked answer"]);
  expect(world.unsafe).toEqual(["hello", "forked"]);
});

test("a fork at zero reruns from the input", async () => {
  const world = new World();
  const store = new MemoryStore();
  const source = await build("react", store, world).run("Do the task", { sessionId: "s1" });
  const fork = await build("react", store, world).fork("s1", source.id, { atSeq: 0 });
  expect(fork.output).toBe("All done");
  expect(world.unsafe).toEqual(["hello", "hello"]);
});
