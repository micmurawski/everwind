/** Shared state: conformance suite for every backend, and journaled use from runs. */

import { Database } from "bun:sqlite";
import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { z } from "zod";
import {
  Agent,
  InvalidPatch,
  type Loop,
  MemoryStore,
  MemoryStoreData,
  PatchConflict,
  Run,
  RunOwnershipLost,
  Runtime,
  SCHEMA_VERSION,
  ScriptedProvider,
  SharedWorld,
  SqliteStore,
  StaleRead,
  Tools,
  type DurableStore,
  formatPath,
  limits,
  parsePath,
  tool,
} from "../src/index.ts";
import { CountingMemory, CrashingMemory, World, crashes, makeTools, tmpDir } from "./helpers.ts";

type Reopen = () => DurableStore;

function backends(): [string, () => Reopen][] {
  return [
    ["memory", () => {
      const data = new MemoryStoreData();
      return () => new MemoryStore(data);
    }],
    ["sqlite", () => {
      const path = join(tmpDir(), "durable.db");
      return () => new SqliteStore(path);
    }],
  ];
}

test("paths are JSON Pointers with dotted shorthand", () => {
  expect(parsePath("")).toEqual([]);
  expect(parsePath("/a~1b/c~0d")).toEqual(["a/b", "c~d"]);
  expect(formatPath(["a/b", "c~d"])).toBe("/a~1b/c~0d");
  expect(parsePath("tasks.t42.status")).toEqual(["tasks", "t42", "status"]);
  expect(() => parsePath("a..b")).toThrow(InvalidPatch);
  expect(() => parsePath("a.b~c")).toThrow(InvalidPatch);
});

describe.each(backends())("conformance on %s", (_name, factory) => {
  test("patch operations on objects and arrays", async () => {
    const world = new SharedWorld(factory()());
    await world.patch([
      { op: "add", path: "/tasks", value: { t1: { status: "open" } } },
      { op: "add", path: "/log", value: ["a"] },
    ]);
    await world.patch([
      { op: "add", path: "/log/-", value: "c" },
      { op: "add", path: "/log/1", value: "b" },
      { op: "replace", path: "tasks.t1.status", value: "done" },
      { op: "test", path: "/tasks/t1/status", value: "done" },
    ]);
    expect(world.read("").value).toEqual({ tasks: { t1: { status: "done" } }, log: ["a", "b", "c"] });
    await world.patch([{ op: "remove", path: "/log/0" }, { op: "remove", path: "/tasks/t1" }]);
    expect(world.read("").value).toEqual({ tasks: {}, log: ["b", "c"] });
    const missing = world.read("/tasks/t9");
    expect([missing.exists, missing.value]).toEqual([false, null]);
  });

  test("rejected patches change nothing and take no offset", async () => {
    const world = new SharedWorld(factory()());
    await world.patch([{ op: "add", path: "/a", value: 1 }]);
    for (const ops of [
      [{ op: "replace", path: "/a", value: 2 }, { op: "test", path: "/a", value: 1 }], // test fails
      [{ op: "add", path: "/b", value: 1 }, { op: "replace", path: "/missing", value: 1 }],
      [{ op: "add", path: "/x/y", value: 1 }], // parent missing
      [{ op: "test", path: "/a", value: true }], // JSON equality: 1 is not true
      [{ op: "remove", path: "" }],
    ]) {
      await expect(world.patch(ops)).rejects.toThrow(PatchConflict);
    }
    for (const ops of [
      [],
      [{ op: "move", path: "/a" }],
      [{ op: "add", path: "/a" }],
      [{ op: "add", path: "/a", value: new Set([1, 2]) }],
      Array.from({ length: 101 }, () => ({ op: "add", path: "/a", value: 1 })),
    ]) {
      await expect(world.patch(ops)).rejects.toThrow(InvalidPatch);
    }
    expect(world.offset).toBe(1);
    expect(world.read("").value).toEqual({ a: 1 });
    expect(world.log().map((entry) => entry.offset)).toEqual([1]);
  });

  test("versions detect exactly the writes that affect a read", async () => {
    const world = new SharedWorld(factory()());
    await world.patch([
      { op: "add", path: "/a", value: { b: { x: 1 }, c: 1 } },
      { op: "add", path: "/list", value: [1, 2] },
    ]);
    const staleAfter = async (readPath: string, ops: { op: string; path: string; value?: unknown }[]) => {
      const read = world.read(readPath);
      await world.patch(ops);
      try {
        await world.patch([{ op: "test", path: "", value: world.read("").value }], { basedOn: [read] });
      } catch (error) {
        expect(error).toBeInstanceOf(StaleRead);
        expect((error as StaleRead).paths).toEqual([readPath]);
        return true;
      }
      return false;
    };
    expect(await staleAfter("/a/b", [{ op: "replace", path: "/a/c", value: 2 }])).toBe(false); // sibling
    expect(await staleAfter("/a/b", [{ op: "replace", path: "/a/b/x", value: 2 }])).toBe(true); // child
    expect(await staleAfter("/a", [{ op: "add", path: "/a/b/y", value: 1 }])).toBe(true); // descendant
    expect(await staleAfter("/a/b/x", [{ op: "replace", path: "/a", value: { b: { x: 2 } } }])).toBe(true); // ancestor
    expect(await staleAfter("/a/z", [{ op: "add", path: "/a/z", value: 1 }])).toBe(true); // created after read
    expect(await staleAfter("/new/x", [{ op: "add", path: "/new", value: { x: 1 } }])).toBe(true); // created by ancestor
    expect(await staleAfter("/list/0", [{ op: "add", path: "/list/-", value: 3 }])).toBe(true); // arrays are one unit
    expect(await staleAfter("/a", [{ op: "test", path: "/a/b/x", value: 2 }])).toBe(false); // test writes nothing
  });

  test("op IDs make retries idempotent", async () => {
    const world = new SharedWorld(factory()());
    const first = await world.patch([{ op: "add", path: "/n", value: 1 }], { opId: "once" });
    const again = await world.patch([{ op: "add", path: "/n", value: 99 }], { opId: "once" });
    expect(first).toEqual(again);
    expect([world.read("/n").value, world.offset]).toEqual([1, 1]);
  });

  test("time travel and restart match every offset", async () => {
    const reopen = factory();
    const world = new SharedWorld(reopen(), "default", { snapshotEvery: 3 });
    const states = [world.state];
    await world.patch([{ op: "add", path: "/items", value: [] }]);
    states.push(world.state);
    for (let index = 0; index < 10; index++) {
      await world.patch([
        { op: "add", path: "/items/-", value: index },
        { op: "add", path: `/k${index}`, value: { v: index } },
      ]);
      states.push(world.state);
    }
    states.forEach((state, offset) => {
      const travelled = world.at(offset);
      expect(travelled.offset).toBe(offset);
      expect(travelled.value).toEqual(state.value);
      for (let i = 0; i < 10; i++) expect(travelled.version(`/k${i}/v`)).toBe(state.version(`/k${i}/v`));
    });
    // Loads the latest snapshot plus the log tail.
    const restarted = new SharedWorld(reopen(), "default", { snapshotEvery: 3 });
    expect(restarted.state.toJson()).toEqual(world.state.toJson());
    expect(() => world.at(states.length)).toThrow();
  });

  test("concurrent compare-and-swap through group commit", async () => {
    const world = new SharedWorld(factory()());
    await world.patch([{ op: "add", path: "/n", value: 0 }]);
    let retries = 0;
    const increment = async () => {
      for (;;) {
        const read = world.read("/n");
        try {
          return await world.patch([{ op: "replace", path: "/n", value: (read.value as number) + 1 }], { basedOn: [read] });
        } catch (error) {
          if (!(error instanceof StaleRead)) throw error;
          retries += 1;
        }
      }
    };
    await Promise.all(Array.from({ length: 50 }, increment));
    expect(world.read("/n").value).toBe(50);
    expect(world.log().map((entry) => entry.offset)).toEqual(Array.from({ length: 51 }, (_, i) => i + 1));
    expect(retries).toBeGreaterThan(0); // writers really raced
  });
});

test.each([1, 2])("SQLite version %d files are upgraded", async (version) => {
  const path = join(tmpDir(), "old.db");
  new SqliteStore(path).close();
  const db = new Database(path);
  db.exec("DROP TABLE events");
  if (version === 1) db.exec("DROP TABLE shared_log");
  db.exec(`PRAGMA user_version=${version}`);
  db.close();
  const store = new SqliteStore(path);
  const world = new SharedWorld(store);
  await world.patch([{ op: "add", path: "/ok", value: true }]);
  expect(world.read("/ok").value).toBe(true);
  expect((store.db.query("PRAGMA user_version").get() as { user_version: number }).user_version).toBe(SCHEMA_VERSION);
  store.close();
});

test("world names and agent stores are checked", () => {
  const store = new MemoryStore();
  const world = new SharedWorld(store);
  expect(() => new SharedWorld(store)).toThrow(/already open/);
  expect(() => new Agent({ provider: new ScriptedProvider([]), store: new MemoryStore(), shared: world })).toThrow(/own store/);
  const runtime = new Runtime(new Run(), new ScriptedProvider([]), new Tools(), limits());
  expect(() => runtime.shared).toThrow(/no shared world/);
});

// Runs ------------------------------------------------------------------------------
/** Read a counter, increment it based on that read, then ask the model. */
class Counter implements Loop {
  async run(runtime: Runtime): Promise<void> {
    const read = await runtime.shared.get("/counter");
    let outcome: string;
    try {
      await runtime.shared.patch([{ op: "replace", path: "/counter", value: (read.value as number) + 1 }], { basedOn: [read] });
      outcome = `set ${(read.value as number) + 1}`;
    } catch (error) {
      if (!(error instanceof StaleRead)) throw error;
      outcome = "stale";
    }
    runtime.say(outcome);
    await runtime.complete();
    runtime.finish(outcome);
  }
}

function counterAgent(store: DurableStore, world: SharedWorld, loop: Loop = new Counter()): Agent {
  return new Agent({ provider: new ScriptedProvider([{ text: "ok" }]), loop, name: "counter", store, shared: world, leaseTtl: 60 });
}

async function seeded(store: DurableStore): Promise<SharedWorld> {
  const world = new SharedWorld(store);
  if (!world.read("/counter").exists) await world.patch([{ op: "add", path: "/counter", value: 0 }]);
  return world;
}

test("shared writes from a run happen exactly once across crashes", async () => {
  const counting = new CountingMemory();
  const seededWorld = await seeded(counting);
  const before = counting.written;
  await counterAgent(counting, seededWorld).run("Go", { sessionId: "s1" });
  const writes = counting.written - before + 1;
  for (let crashAt = 2; crashAt <= writes; crashAt++) {
    for (const after of [false, true]) {
      const data = new MemoryStoreData();
      const crashing = new CrashingMemory(data, { crashAt, after });
      await crashes(counterAgent(crashing, await seeded(crashing)).run("Go", { sessionId: "s1" }), {
        mayFinish: after && crashAt === writes,
      });
      const store = new MemoryStore(data);
      const world = new SharedWorld(store);
      if (!store.runs().length) {
        expect(world.read("/counter").value).toBe(0); // The run never started.
        continue;
      }
      if (store.runs({ status: "running" }).length) {
        const recovered = await counterAgent(store, world).recover("s1", { force: true });
        expect([recovered.status, recovered.output]).toEqual(["completed", "set 1"]);
      }
      // Exactly one increment, by this run, however the crash fell.
      expect(store.runs()[0].status).toBe("completed");
      expect(world.read("/counter").value).toBe(1);
      const authors = world.log().flatMap((entry) => (entry.author ? [entry.author] : []));
      expect(authors.length).toBe(1);
      expect(authors[0].session).toBe("s1");
    }
  }
});

test("a stale read is journaled and replays", async () => {
  const data = new MemoryStoreData();
  const crashing = new CrashingMemory(data, { crashAt: 3, after: true }); // seed, createRun, read: then crash
  await crashes(counterAgent(crashing, await seeded(crashing)).run("Go", { sessionId: "s1" }));
  const store = new MemoryStore(data);
  const world = new SharedWorld(store);
  expect(world.read("/counter").value).toBe(0);
  await world.patch([{ op: "replace", path: "/counter", value: 10 }]); // another writer, meanwhile
  const recovered = await counterAgent(store, world).recover("s1", { force: true });
  expect(recovered.output).toBe("stale");
  expect(world.read("/counter").value).toBe(10);
  const replayed = await counterAgent(store, world).replay("s1", recovered.id);
  expect(replayed.run.output).toBe("stale");
});

test("replay returns what the run read, not the current value", async () => {
  const store = new MemoryStore();
  const world = await seeded(store);
  const source = await counterAgent(store, world).run("Go", { sessionId: "s1" });
  await world.patch([{ op: "replace", path: "/counter", value: 500 }]);
  const replayed = await counterAgent(store, world).replay("s1", source.id);
  expect(replayed.run.output).toBe("set 1");
  expect(world.read("/counter").value).toBe(500);
  const read = store.journal("s1", source.id).find((entry) => entry.kind === "shared_read")?.payload.read as { offset: number };
  expect(world.at(read.offset).read("/counter").value).toBe(0); // the world as the run saw it
});

test("a stale owner cannot write shared state", async () => {
  const store = new MemoryStore();
  const world = await seeded(store);
  let entered!: () => void;
  const enteredPromise = new Promise<void>((resolve) => (entered = resolve));
  let release!: () => void;
  const released = new Promise<void>((resolve) => (release = resolve));

  class Paused implements Loop {
    async run(runtime: Runtime): Promise<void> {
      const read = await runtime.shared.get("/counter");
      await runtime.journal?.flush(); // Nothing queued: the commit itself must refuse.
      entered();
      await released;
      await runtime.shared.patch([{ op: "replace", path: "/counter", value: 1 }], { basedOn: [read] });
    }
  }

  const pending = counterAgent(store, world, new Paused()).run("Go", { sessionId: "s1" });
  await enteredPromise;
  const record = store.runs({ status: "running" })[0];
  await store.claimRun("s1", record.runId, { ttl: 60, force: true }); // another process took over
  release();
  await expect(pending).rejects.toThrow(RunOwnershipLost);
  expect([world.read("/counter").value, world.offset]).toEqual([0, 1]);
});

test("idempotent tools write once when re-executed", async () => {
  const effects = new World();
  const record = tool({
    name: "record",
    description: "Append a note to the shared log.",
    effect: "idempotent",
    input: z.object({ note: z.string() }),
    run: async ({ note }, ctx) => {
      await ctx.shared.patch([{ op: "add", path: "/notes/-", value: note }]);
      effects.reads += 1;
      return "noted";
    },
  });
  const agent = (store: DurableStore, world: SharedWorld) =>
    new Agent({
      provider: new ScriptedProvider([{ toolCalls: [{ id: "n", name: "record", arguments: { note: "hi" } }] }, { text: "done" }]),
      tools: Tools.of(record, makeTools(effects)),
      name: "noter",
      store,
      shared: world,
      leaseTtl: 60,
    });
  const data = new MemoryStoreData();
  // seed, createRun, model(s), model(c), tool(s), shared write -> crash before the tool's result
  const crashing = new CrashingMemory(data, { crashAt: 6, after: true });
  const first = new SharedWorld(crashing);
  await first.patch([{ op: "add", path: "/notes", value: [] }]);
  await crashes(agent(crashing, first).run("Note it", { sessionId: "s1" }));
  expect(first.read("/notes").value).toEqual(["hi"]);
  const store = new MemoryStore(data);
  const world = new SharedWorld(store);
  const recovered = await agent(store, world).recover("s1", { force: true });
  expect(recovered.status).toBe("completed");
  expect(effects.reads).toBe(2); // the tool ran again ...
  expect(world.read("/notes").value).toEqual(["hi"]); // ... but its write was deduplicated
});
