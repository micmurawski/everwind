/** The session log of a durable agent is stored in the journal's store. */

import { describe, expect, test } from "bun:test";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { Agent, type LoggedEvent, MemoryStore, MemoryStoreData, ReAct, ScriptedProvider, SqliteStore, Tools, type DurableStore, tool } from "../src/index.ts";
import { CountingMemory, CrashingMemory, SCENARIOS, World, build, call, crashes, makeTools, tmpDir } from "./helpers.ts";

const ATTEMPT_KINDS = new Set(["run_recovered", "command_retried", "fork_live"]);

const logged = (store: DurableStore, sessionId = "s1") => store.events(sessionId);

/** Invariants of any stored run log. */
function checkLog(events: LoggedEvent[]): void {
  const byRun = new Map<string, LoggedEvent[]>();
  for (const stored of events) byRun.set(stored.event.runId, [...(byRun.get(stored.event.runId) ?? []), stored]);
  for (const runEvents of byRun.values()) {
    expect(runEvents.map((stored) => stored.event.sequence)).toEqual(runEvents.map((_, index) => index));
    const kinds = runEvents.map((stored) => stored.event.kind);
    expect(kinds).not.toContain("model_chunk");
    expect(kinds.filter((kind) => kind === "run_start").length).toBe(1);
    expect(kinds[0]).toBe("run_start");
    const ends = kinds.filter((kind) => kind === "run_end").length;
    expect(ends).toBeLessThanOrEqual(1);
    if (ends) expect(kinds.at(-1)).toBe("run_end");
  }
  const positions = events.map((stored) => stored.position);
  expect(positions).toEqual([...positions].sort((a, b) => a - b));
  expect(new Set(positions).size).toBe(positions.length);
}

function keyCounts(events: LoggedEvent[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const stored of events) {
    if (!ATTEMPT_KINDS.has(stored.event.kind)) counts.set(stored.key, (counts.get(stored.key) ?? 0) + 1);
  }
  return counts;
}

describe("a recovered log has every event once", () => {
  for (const name of Object.keys(SCENARIOS).sort()) {
    test(name, async () => {
      const counting = new CountingMemory();
      await build(name, counting, new World()).run("Do the task", { sessionId: "s1" });
      const writes = counting.written;
      const expected = keyCounts(logged(counting));
      let recoveredPoints = 0;
      for (let crashAt = 1; crashAt <= writes; crashAt++) {
        for (const after of [false, true]) {
          const world = new World();
          const data = new MemoryStoreData();
          await crashes(build(name, new CrashingMemory(data, { crashAt, after }), world).run("Do the task", { sessionId: "s1" }), {
            mayFinish: after && crashAt === writes,
          });
          const store = new MemoryStore(data);
          if (!store.runs().length) {
            expect(logged(store)).toEqual([]);
            continue;
          }
          const recovering = store.runs()[0].status === "running";
          if (recovering) {
            await build(name, store, world).recover("s1", { force: true });
            recoveredPoints += 1;
          }
          const events = logged(store);
          checkLog(events);
          // The same events as an uninterrupted run, each once: replayed ones are not logged
          // twice, and ones the crash lost before they were written are logged on recovery.
          expect(keyCounts(events)).toEqual(expected);
          expect(events.filter((stored) => stored.event.kind === "run_recovered").length).toBe(recovering ? 1 : 0);
        }
      }
      expect(recoveredPoints).toBeGreaterThan(writes); // Most crash points really went through recovery.
    });
  }
});

test("retried commands are logged per attempt", async () => {
  const world = new World();
  const data = new MemoryStoreData();
  // Write 4 schedules the first tool batch (lookup + upsert); crash before either finishes.
  await crashes(build("react", new CrashingMemory(data, { crashAt: 4, after: true }), world).run("T", { sessionId: "s1" }));
  const store = new MemoryStore(data);
  await build("react", store, world).recover("s1", { force: true });
  const retried = logged(store).filter((stored) => stored.event.kind === "command_retried").map((stored) => stored.event.data);
  expect(retried.map((item) => [item.callId, item.effect])).toEqual([["a", "read"], ["b", "idempotent"]]);
});

test("inspection replay writes nothing, and a fork logs its prefix", async () => {
  const world = new World();
  const store = new MemoryStore();
  const source = await build("react", store, world).run("Do the task", { sessionId: "s1" });
  const before = logged(store);
  const agent = build("react", store, world);
  const replayed = await agent.replay("s1", source.id);
  expect(replayed.run.events.length).toBeGreaterThan(0);
  expect(logged(store)).toEqual(before);
  const fork = await agent
    .with({ provider: new ScriptedProvider([{ text: "x" }, { text: "Forked" }]) })
    .fork("s1", source.id, { atSeq: 3, newSessionId: "fork-1" });
  expect(logged(store)).toEqual(before); // The source session's log is untouched.
  const events = logged(store, "fork-1");
  checkLog(events);
  const kinds = events.map((stored) => stored.event.kind);
  // The copied prefix is in the fork's own log, then the marker, then live events.
  expect(kinds.indexOf("fork_live")).toBeGreaterThan(kinds.indexOf("tool_result"));
  expect(kinds.filter((kind) => kind === "tool_result").length).toBe(2);
  expect(kinds.at(-1)).toBe("run_end");
  expect(events.every((stored) => stored.event.runId === fork.id)).toBe(true);
});

test("events of a long tool call are stored before it returns", async () => {
  const store = new MemoryStore();
  const seen: string[] = [];
  const slow = tool({
    name: "slow",
    description: "Take a while.",
    effect: "read",
    run: async () => {
      await Bun.sleep(250);
      seen.push(...logged(store).map((stored) => stored.event.kind));
      return "done";
    },
  });
  const agent = new Agent({
    provider: new ScriptedProvider([{ toolCalls: [call("a", "slow")] }, { text: "ok" }]),
    tools: new Tools([slow]),
    loop: new ReAct(),
    store,
    leaseTtl: 0.15,
  });
  await agent.run("T", { sessionId: "s1" });
  checkLog(logged(store));
  expect(seen.at(-1)).toBe("tool_start"); // Flushed with the heartbeat, while the tool was running.
});

test("the SQLite log survives reopening and exports JSONL", async () => {
  const directory = tmpDir();
  const path = join(directory, "store.db");
  let store = new SqliteStore(path);
  await build("react", store, new World()).run("Do the task", { sessionId: "s1" });
  store.close();
  store = new SqliteStore(path);
  const size = logged(store).length;
  await build("react", store, new World()).run("Next", { sessionId: "s1" });
  const events = logged(store);
  checkLog(events);
  expect(new Set(events.map((stored) => stored.event.runId)).size).toBe(2);
  const tail = store.events("s1", { after: events[size - 1].position });
  expect(tail.every((stored) => stored.event.runId === events.at(-1)?.event.runId)).toBe(true);
  expect(store.events("s1", { limit: 2 })).toEqual(events.slice(0, 2));
  const exported = store.exportLog("s1", join(directory, "out", "s1.jsonl"));
  const rows = readFileSync(exported, "utf-8").trim().split("\n").map((line) => JSON.parse(line));
  expect(rows.map((row) => row.kind)).toEqual(events.map((stored) => stored.event.kind));
  expect(new Set(rows.map((row) => row.schemaVersion))).toEqual(new Set([1]));
  expect(new Set(rows.map((row) => row.sessionId))).toEqual(new Set(["s1"]));
  store.close();
});

test("a store replaces logDir", async () => {
  const directory = tmpDir();
  expect(() => new Agent({ provider: new ScriptedProvider([]), store: new MemoryStore(), logDir: directory })).toThrow(/store.events/);
  const agent = new Agent({ provider: new ScriptedProvider([{ text: "x" }]), tools: makeTools(new World()), store: new MemoryStore() });
  await expect(agent.run("T", { logDir: directory })).rejects.toThrow(/store.events/);
  expect(readdirSync(directory)).toEqual([]);
});

