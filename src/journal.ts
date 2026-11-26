/**
 * Durable run journal: run ownership, recorded commands, the event log and a group committer.
 *
 * A run's model and tool commands are journaled at the Runtime boundary. Recovery
 * re-executes the loop and returns recorded results instead of repeating effects.
 * The run's events are stored with the journal: they are the session log.
 * See docs/durable-state.md, sections 4, 5 and 11.
 */

import { Database } from "bun:sqlite";
import { mkdirSync, writeFileSync } from "node:fs";
import { hostname } from "node:os";
import { dirname, resolve } from "node:path";
import { JournalError, NonDeterminismError, ReplayExhausted, RunLeaseHeld, RunOwnershipLost } from "./errors.ts";
import { eventLine, validateSessionId } from "./sessionLog.ts";
import { type Session, type SessionStore, SessionSchema } from "./sessions.ts";
import type { Event } from "./types.ts";
import { type JsonObject, KeyedMutex, clone, hexId, sha256, stableStringify } from "./util.ts";
import type { LogEntry, SharedState, SharedWorld } from "./shared.ts";

/** 1: runs, journal, sessions. 2: shared-state tables. 3: the event log. */
export const SCHEMA_VERSION = 5;
export const INTERRUPTED_CONTENT = "Execution interrupted. Outcome unknown; inspect external state before retrying.";
/** Live-only events: never stored. A model_response holds everything its chunks said. */
export const EPHEMERAL_EVENTS: ReadonlySet<string> = new Set(["model_chunk"]);

export type Phase = "scheduled" | "completed" | "marker";
export type CommandKind = "model" | "tool" | "side_effect" | "version" | "shared_read" | "shared_write" | "suspend" | "receive" | "checkpoint";

export interface JournalEntry {
  readonly seq: number;
  /** A marker (a version() decision) is found by its key, not by its position. */
  readonly phase: Phase;
  readonly kind: CommandKind;
  readonly key: string;
  readonly payload: JsonObject;
}

export interface RunRecord {
  readonly sessionId: string;
  readonly runId: string;
  readonly agent: string;
  readonly status: string;
  readonly owner: string | null;
  readonly epoch: number;
  readonly leaseExpiresAt: number;
  readonly activeSeconds: number;
  readonly start: JsonObject;
  readonly result: JsonObject | null;
}

/**
 * A message in a session's durable inbox: a note from an operator, a child's
 * result, a follow-up. `id` increases per session in arrival order, whatever
 * the storage layout; journals record ids, never positions.
 */
export interface InboxMessage {
  readonly id: number;
  readonly sessionId: string;
  readonly kind: string;
  readonly payload: JsonObject;
  /** ISO 8601, UTC. */
  readonly at: string;
  /** The run that consumed it, or null while it is pending. */
  readonly consumedBy: string | null;
}

/**
 * What an operator wants a session's runs to do: keep going, hold at the next
 * step, or freeze (stop and release the run, leaving it recoverable). Not a
 * message: a run reads the current state at each step, and it is never
 * journaled, because it never changes which commands a run issues.
 */
export type ControlState = "run" | "hold" | "freeze";

export interface ControlRecord {
  readonly sessionId: string;
  readonly state: ControlState;
  readonly note: string | null;
  /** ISO 8601, UTC. */
  readonly at: string;
}

/** A row of the board (projects and tasks; see board.ts): an id, indexed columns and a JSON body. */
export interface BoardRow {
  readonly id: string;
  readonly body: JsonObject;
}

export interface InboxQuery {
  /** Only these kinds; all kinds when omitted. */
  kinds?: readonly string[];
  /** Only messages no run has consumed. */
  pending?: boolean;
  /** Only ids above this one. */
  after?: number;
  limit?: number;
}

/** A stored event. `position` orders all events of a store; `event.sequence` counts within the run. */
export interface LoggedEvent {
  readonly position: number;
  readonly key: string;
  readonly event: Event;
}

export type KeyedEvent = readonly [key: string, event: Event];

/** Stable hash identifying a command, used to detect divergent replays. */
export function commandKey(kind: string, value: unknown): string {
  return `${kind}:${sha256(stableStringify(value)).slice(0, 32)}`;
}

export function runRecord(
  init: Pick<RunRecord, "sessionId" | "runId" | "agent"> & Partial<RunRecord>,
): RunRecord {
  return {
    status: "running",
    owner: null,
    epoch: 1,
    leaseExpiresAt: 0,
    activeSeconds: 0,
    start: {},
    result: null,
    ...init,
  };
}

// Operations submitted to the committer. Each is applied atomically on its own.
export type Op =
  | { type: "createRun"; record: RunRecord; entries: readonly JournalEntry[] }
  | { type: "claimRun"; sessionId: string; runId: string; owner: string; leaseExpiresAt: number; now: number; force: boolean }
  | {
      type: "append";
      sessionId: string;
      runId: string;
      epoch: number;
      entries: readonly JournalEntry[];
      events: readonly KeyedEvent[];
      /** Inbox ids consumed by these entries, in the same transaction. */
      consume?: readonly number[];
    }
  | { type: "signal"; sessionId: string; kind: string; payload: JsonObject; at: string }
  | { type: "control"; record: ControlRecord }
  | { type: "release"; sessionId: string; runId: string; epoch: number }
  | { type: "heartbeat"; sessionId: string; runId: string; epoch: number; leaseExpiresAt: number; activeSeconds: number }
  | {
      type: "finish";
      sessionId: string;
      runId: string;
      epoch: number;
      status: string;
      result: JsonObject;
      activeSeconds: number;
      session: Session | null;
      events: readonly KeyedEvent[];
    }
  | { type: "saveSession"; session: Session }
  | { type: "custom"; applyIn(tx: Tx): unknown };

/** Primitive reads and writes one backend offers inside a write batch. */
export interface Tx {
  getRun(sessionId: string, runId: string): RunRecord | null;
  putRun(record: RunRecord): void;
  insertEntries(sessionId: string, runId: string, epoch: number, entries: readonly JournalEntry[]): void;
  putSession(session: Session): void;
  /** Events with a key already stored for the run are ignored: a replay re-emits them. */
  insertEvents(sessionId: string, runId: string, events: readonly KeyedEvent[]): void;
  // Shared worlds (see shared.ts).
  getHead(world: string): number;
  setHead(world: string, head: number): void;
  insertLog(world: string, entry: LogEntry): void;
  findOp(world: string, opId: string): LogEntry | null;
  readLog(world: string, after: number): LogEntry[];
  putSnapshot(world: string, offset: number, body: JsonObject): void;
  // Inboxes.
  /** Append a message; returns its id, one above the session's highest. */
  insertSignal(sessionId: string, kind: string, payload: JsonObject, at: string): number;
  /** Mark a pending message consumed by a run; a consumed or unknown one is an error. */
  consumeSignal(sessionId: string, id: number, runId: string): void;
  putControl(record: ControlRecord): void;
  // The board.
  getProject(id: string): JsonObject | null;
  putProject(id: string, status: string, body: JsonObject): void;
  getTask(id: string): JsonObject | null;
  putTask(id: string, project: string, parent: string | null, status: string, body: JsonObject): void;
}

export function owned(tx: Tx, sessionId: string, runId: string, epoch: number): RunRecord {
  const record = tx.getRun(sessionId, runId);
  if (record === null) throw new RunOwnershipLost(`Run ${runId} does not exist`);
  if (record.status !== "running" || record.epoch !== epoch) {
    throw new RunOwnershipLost(
      `Run ${runId} is ${record.status} at epoch ${record.epoch}; this owner holds epoch ${epoch}`,
    );
  }
  return record;
}

function apply(tx: Tx, op: Op): unknown {
  switch (op.type) {
    case "createRun": {
      if (tx.getRun(op.record.sessionId, op.record.runId) !== null) {
        throw new JournalError(`Run ${op.record.runId} already exists`);
      }
      tx.putRun(op.record);
      // A fork starts with its copied prefix, atomically.
      if (op.entries.length) tx.insertEntries(op.record.sessionId, op.record.runId, op.record.epoch, op.entries);
      return op.record.epoch;
    }
    case "claimRun": {
      const record = tx.getRun(op.sessionId, op.runId);
      if (record === null) throw new Error(`Run ${op.runId} does not exist`);
      if (record.status !== "running") throw new Error(`Run ${op.runId} is ${record.status}, not running`);
      if (!op.force && record.owner !== op.owner && record.leaseExpiresAt > op.now) {
        throw new RunLeaseHeld(
          `Run ${op.runId} is leased by ${record.owner} for ${(record.leaseExpiresAt - op.now).toFixed(1)}s more`,
        );
      }
      const claimed = { ...record, owner: op.owner, epoch: record.epoch + 1, leaseExpiresAt: op.leaseExpiresAt };
      tx.putRun(claimed);
      return claimed.epoch;
    }
    case "append":
      owned(tx, op.sessionId, op.runId, op.epoch);
      tx.insertEntries(op.sessionId, op.runId, op.epoch, op.entries);
      for (const id of op.consume ?? []) tx.consumeSignal(op.sessionId, id, op.runId);
      if (op.events.length) tx.insertEvents(op.sessionId, op.runId, op.events);
      return null;
    case "signal":
      return tx.insertSignal(op.sessionId, op.kind, op.payload, op.at);
    case "control":
      tx.putControl(op.record);
      return null;
    case "release": {
      // Stop owning a run without finishing it: a frozen run waits for recover().
      const record = owned(tx, op.sessionId, op.runId, op.epoch);
      tx.putRun({ ...record, owner: null, leaseExpiresAt: 0 });
      return null;
    }
    case "heartbeat": {
      const record = owned(tx, op.sessionId, op.runId, op.epoch);
      tx.putRun({ ...record, leaseExpiresAt: op.leaseExpiresAt, activeSeconds: op.activeSeconds });
      return null;
    }
    case "finish": {
      const record = owned(tx, op.sessionId, op.runId, op.epoch);
      if (op.events.length) tx.insertEvents(op.sessionId, op.runId, op.events);
      tx.putRun({
        ...record,
        status: op.status,
        result: op.result,
        activeSeconds: op.activeSeconds,
        owner: null,
        leaseExpiresAt: 0,
      });
      // Same transaction: the checkpoint cannot lag the run.
      if (op.session !== null) tx.putSession(op.session);
      return null;
    }
    case "saveSession":
      tx.putSession(op.session);
      return null;
    case "custom":
      // Operations defined by other modules, e.g. shared-state commits.
      return op.applyIn(tx);
  }
}

type Waiter = { op: Op; resolve: (value: unknown) => void; reject: (error: unknown) => void };

/**
 * Batches concurrent submissions into one write; each op gets its own result.
 * Ops are written in the order they were queued. Submissions that arrive while
 * a batch is written join the next batch.
 */
export class Committer {
  private queue: Waiter[] = [];
  private draining = false;

  constructor(
    private readonly write: (ops: Op[]) => (unknown | Error)[],
    private readonly maxBatch = 512,
  ) {}

  submit<T = unknown>(op: Op): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.queue.push({ op, resolve: resolve as (value: unknown) => void, reject });
      if (!this.draining) {
        this.draining = true;
        queueMicrotask(() => void this.drain());
      }
    });
  }

  private async drain(): Promise<void> {
    try {
      while (this.queue.length) {
        await Promise.resolve(); // Let concurrently scheduled submitters join this batch.
        const batch = this.queue.splice(0, this.maxBatch);
        let results: unknown[];
        try {
          results = this.write(batch.map((waiter) => waiter.op));
        } catch (error) {
          for (const waiter of batch) waiter.reject(error); // A failed batch fails each waiting submitter.
          continue;
        }
        batch.forEach((waiter, index) => {
          const result = results[index];
          if (result instanceof OpFailure) waiter.reject(result.error);
          else waiter.resolve(result);
        });
      }
    } finally {
      this.draining = false;
    }
  }
}

/** A failed op inside an otherwise committed batch. */
export class OpFailure {
  constructor(readonly error: unknown) {}
}

/**
 * Journal, run ownership, the event log and session checkpoints behind one group committer.
 *
 * Also implements SessionStore, so one store holds everything an Agent persists.
 * Session locks are in-process: run every agent of a store in one process.
 */
export abstract class DurableStore implements SessionStore {
  readonly ownerId = `${hostname()}:${process.pid}:${hexId().slice(0, 8)}`;
  /** @internal */
  readonly committer = new Committer((ops) => this.writeBatch(ops));
  private readonly mutex = new KeyedMutex();
  /** @internal Open SharedWorlds by name. */
  readonly worlds = new Map<string, SharedWorld>();
  /** @internal World states written in the current batch, not yet committed. */
  staged = new Map<string, SharedState>();

  /** @internal After a successful commit: make staged world states visible to readers. */
  protected publish(): void {
    const staged = this.staged;
    this.staged = new Map();
    for (const [name, state] of staged) this.worlds.get(name)?.publish(state);
  }

  protected discard(): void {
    this.staged = new Map();
  }

  // Backend hooks --------------------------------------------------------
  /** Apply ops in order, each atomically; return a result or an OpFailure per op. */
  protected abstract writeBatch(ops: Op[]): unknown[];
  abstract getRun(sessionId: string, runId: string): RunRecord | null;
  abstract runs(filter?: { status?: string; sessionId?: string }): RunRecord[];
  abstract journal(sessionId: string, runId: string): JournalEntry[];
  abstract load(sessionId: string): Session | null;
  /** The session's stored events in commit order; `after` a position, to follow the log. */
  abstract events(
    sessionId: string,
    options?: { runId?: string; after?: number; limit?: number },
  ): LoggedEvent[];
  abstract eventKeys(sessionId: string, runId: string): Set<string>;
  /** @internal */
  abstract sharedHead(world: string): number;
  /** @internal */
  abstract sharedLog(world: string, after: number, until: number | null): LogEntry[];
  /** @internal */
  abstract sharedSnapshot(world: string, atOrBefore: number | null): JsonObject | null;
  /** A session's inbox messages in id order. */
  abstract inbox(sessionId: string, query?: InboxQuery): InboxMessage[];
  /** A session's control state, or null when none was ever set (it then runs). */
  abstract control(sessionId: string): ControlRecord | null;
  /** @internal Board reads; use board.ts. */
  abstract boardProjects(): JsonObject[];
  /** @internal */
  abstract boardTasks(filter: { project?: string; parent?: string; status?: string }): JsonObject[];
  /** Seconds between inbox checks while a run waits, for writers in other processes; null: only this process writes. */
  protected readonly pollInterval: number | null = null;
  private readonly listeners = new Map<string, Set<() => void>>();

  // Inboxes --------------------------------------------------------------
  /**
   * Put a message in a session's inbox and wake a run of that session waiting
   * for its kind. Returns the message id.
   */
  async signal(sessionId: string, kind: string, payload: JsonObject = {}): Promise<number> {
    validateSessionId(sessionId);
    if (typeof kind !== "string" || !kind) throw new Error("A signal needs a nonempty kind");
    const id = await this.committer.submit<number>({ type: "signal", sessionId, kind, payload: clone(payload), at: new Date().toISOString() });
    for (const wake of [...(this.listeners.get(sessionId) ?? [])]) wake();
    return id;
  }

  /** Set what a session's runs should do; a run waiting on the session notices at once. */
  async setControl(sessionId: string, state: ControlState, note: string | null = null): Promise<void> {
    validateSessionId(sessionId);
    if (!["run", "hold", "freeze"].includes(state)) throw new Error(`Unknown control state ${JSON.stringify(state)}`);
    await this.committer.submit({ type: "control", record: { sessionId, state, note, at: new Date().toISOString() } });
    for (const wake of [...(this.listeners.get(sessionId) ?? [])]) wake();
  }

  /** @internal Wake runs of this session that wait for its inbox or control state (after a custom write). */
  notify(sessionId: string): void {
    for (const wake of [...(this.listeners.get(sessionId) ?? [])]) wake();
  }

  /** @internal Release a run's lease without finishing it (a frozen run). */
  async releaseRun(sessionId: string, runId: string, epoch: number): Promise<void> {
    await this.committer.submit({ type: "release", sessionId, runId, epoch });
  }

  /**
   * The oldest pending message of one of `kinds`, as soon as there is one; null
   * once `deadline` (epoch milliseconds) passes, or as soon as `interrupt()`
   * returns true (checked whenever the session's inbox or control state
   * changes). Empty `kinds` waits for the deadline or the interrupt only.
   */
  async waitForSignal(
    sessionId: string,
    kinds: readonly string[],
    options: { deadline: number | null; signal?: AbortSignal; interrupt?: () => boolean },
  ): Promise<InboxMessage | null> {
    if (!kinds.length && options.deadline === null && options.interrupt === undefined) {
      throw new Error("A wait needs signal kinds, a deadline or an interrupt");
    }
    for (;;) {
      options.signal?.throwIfAborted();
      if (options.interrupt?.()) return null;
      if (kinds.length) {
        const [first] = this.inbox(sessionId, { kinds, pending: true, limit: 1 });
        if (first !== undefined) return first;
      }
      const left = options.deadline === null ? Number.POSITIVE_INFINITY : options.deadline - Date.now();
      if (left <= 0) return null;
      const watching = kinds.length > 0 || options.interrupt !== undefined;
      const poll = watching && this.pollInterval !== null ? this.pollInterval * 1000 : Number.POSITIVE_INFINITY;
      await new Promise<void>((resolveWait, rejectWait) => {
        let listeners = this.listeners.get(sessionId);
        if (listeners === undefined) this.listeners.set(sessionId, (listeners = new Set()));
        const set = listeners;
        const done = () => {
          set.delete(done);
          if (!set.size) this.listeners.delete(sessionId);
          if (timer !== null) clearTimeout(timer);
          options.signal?.removeEventListener("abort", aborted);
        };
        const wake = () => {
          done();
          resolveWait();
        };
        const aborted = () => {
          done();
          rejectWait(options.signal?.reason);
        };
        const wait = Math.min(left, poll, 2 ** 31 - 1);
        const timer = Number.isFinite(wait) ? setTimeout(wake, wait) : null;
        if (watching) set.add(wake);
        options.signal?.addEventListener("abort", aborted, { once: true });
      });
    }
  }

  /** Write the session's log as JSONL, in the session log format; overwrites `path`. */
  exportLog(sessionId: string, path: string): string {
    const target = resolve(path);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, this.events(sessionId).map((stored) => eventLine(stored.event)).join(""), "utf-8");
    return target;
  }

  // SessionStore ---------------------------------------------------------
  withLock<T>(sessionId: string, fn: () => Promise<T>): Promise<T> {
    validateSessionId(sessionId);
    return this.mutex.run(sessionId, fn);
  }

  save(session: Session): void {
    const [result] = this.writeBatch([{ type: "saveSession", session }]);
    if (result instanceof OpFailure) throw result.error;
  }

  async asave(session: Session): Promise<void> {
    await this.committer.submit({ type: "saveSession", session });
  }

  // Runs and journal -----------------------------------------------------
  async createRun(record: RunRecord, entries: readonly JournalEntry[] = []): Promise<void> {
    await this.committer.submit({ type: "createRun", record, entries });
  }

  /** Take ownership of a running run; returns the new epoch. */
  claimRun(sessionId: string, runId: string, options: { ttl: number; force?: boolean }): Promise<number> {
    const now = Date.now() / 1000;
    return this.committer.submit({
      type: "claimRun",
      sessionId,
      runId,
      owner: this.ownerId,
      leaseExpiresAt: now + options.ttl,
      now,
      force: options.force ?? false,
    });
  }

  async append(
    sessionId: string,
    runId: string,
    epoch: number,
    entries: readonly JournalEntry[],
    events: readonly KeyedEvent[] = [],
    consume: readonly number[] = [],
  ): Promise<void> {
    await this.committer.submit({ type: "append", sessionId, runId, epoch, entries, events, consume });
  }

  async heartbeat(sessionId: string, runId: string, epoch: number, options: { ttl: number; activeSeconds: number }): Promise<void> {
    await this.committer.submit({
      type: "heartbeat",
      sessionId,
      runId,
      epoch,
      leaseExpiresAt: Date.now() / 1000 + options.ttl,
      activeSeconds: options.activeSeconds,
    });
  }

  /** Record the run's outcome and, atomically with it, the session checkpoint and last events. */
  async finishRun(
    sessionId: string,
    runId: string,
    epoch: number,
    outcome: { status: string; result: JsonObject; activeSeconds: number; session?: Session | null; events?: readonly KeyedEvent[] },
  ): Promise<void> {
    await this.committer.submit({
      type: "finish",
      sessionId,
      runId,
      epoch,
      status: outcome.status,
      result: outcome.result,
      activeSeconds: outcome.activeSeconds,
      session: outcome.session ?? null,
      events: outcome.events ?? [],
    });
  }
}

// Memory ------------------------------------------------------------------
/** Backing data of a MemoryStore. Share it between stores to simulate a restart. */
export class MemoryStoreData {
  runs = new Map<string, RunRecord>();
  entries = new Map<string, Map<string, JournalEntry>>();
  sessions = new Map<string, Session>();
  heads = new Map<string, number>();
  /** world -> entries; offset i at index i - 1. */
  logs = new Map<string, LogEntry[]>();
  opIds = new Map<string, number>();
  snapshots = new Map<string, Map<number, JsonObject>>();
  /** position i at index i - 1. */
  events: LoggedEvent[] = [];
  eventKeys = new Map<string, Set<string>>();
  /** sessionId -> messages; id i at index i - 1. */
  inbox = new Map<string, InboxMessage[]>();
  controls = new Map<string, ControlRecord>();
  projects = new Map<string, JsonObject>();
  /** In creation order. */
  tasks = new Map<string, JsonObject>();
}

const runKey = (sessionId: string, runId: string) => `${sessionId}\u0000${runId}`;
const entryKey = (entry: JournalEntry) => `${entry.seq}:${entry.phase}`;

/** Stages one op's changes; commit() applies them, so a failed op changes nothing. */
class MemoryTx implements Tx {
  private runs = new Map<string, RunRecord>();
  private entries: [string, JournalEntry][] = [];
  private sessions = new Map<string, Session>();
  private heads = new Map<string, number>();
  private logs: [string, LogEntry][] = [];
  private snapshots: [string, number, JsonObject][] = [];
  private events: [string, string, string, Event][] = [];
  private signals: InboxMessage[] = [];
  private consumed: [string, number, string][] = [];
  private controls: ControlRecord[] = [];
  private projects = new Map<string, JsonObject>();
  private tasks = new Map<string, JsonObject>();

  constructor(private readonly data: MemoryStoreData) {}

  getProject(id: string): JsonObject | null {
    const found = this.projects.get(id) ?? this.data.projects.get(id);
    return found === undefined ? null : clone(found);
  }

  putProject(id: string, _status: string, body: JsonObject): void {
    this.projects.set(id, clone(body));
  }

  getTask(id: string): JsonObject | null {
    const found = this.tasks.get(id) ?? this.data.tasks.get(id);
    return found === undefined ? null : clone(found);
  }

  putTask(id: string, _project: string, _parent: string | null, _status: string, body: JsonObject): void {
    this.tasks.set(id, clone(body));
  }

  putControl(record: ControlRecord): void {
    this.controls.push(clone(record));
  }

  insertSignal(sessionId: string, kind: string, payload: JsonObject, at: string): number {
    const id = (this.data.inbox.get(sessionId)?.length ?? 0) + this.signals.filter((item) => item.sessionId === sessionId).length + 1;
    this.signals.push({ id, sessionId, kind, payload: clone(payload), at, consumedBy: null });
    return id;
  }

  consumeSignal(sessionId: string, id: number, runId: string): void {
    const message = this.data.inbox.get(sessionId)?.[id - 1];
    if (message === undefined) throw new JournalError(`Inbox message ${id} of session ${sessionId} does not exist`);
    if (message.consumedBy !== null || this.consumed.some(([s, i]) => s === sessionId && i === id)) {
      throw new JournalError(`Inbox message ${id} of session ${sessionId} was already consumed`);
    }
    this.consumed.push([sessionId, id, runId]);
  }

  getRun(sessionId: string, runId: string): RunRecord | null {
    const key = runKey(sessionId, runId);
    return this.runs.get(key) ?? this.data.runs.get(key) ?? null;
  }

  putRun(record: RunRecord): void {
    this.runs.set(runKey(record.sessionId, record.runId), clone(record));
  }

  insertEntries(sessionId: string, runId: string, _epoch: number, entries: readonly JournalEntry[]): void {
    const key = runKey(sessionId, runId);
    const existing = this.data.entries.get(key) ?? new Map();
    const seen = new Set<string>();
    for (const entry of entries) {
      const position = entryKey(entry);
      if (existing.has(position) || seen.has(position)) {
        throw new JournalError(`Journal entry ${position} already exists for run ${runId}`);
      }
      seen.add(position);
      this.entries.push([key, clone(entry)]);
    }
  }

  putSession(session: Session): void {
    this.sessions.set(session.sessionId, clone(session));
  }

  insertEvents(sessionId: string, runId: string, events: readonly KeyedEvent[]): void {
    for (const [key, event] of events) this.events.push([sessionId, runId, key, event]);
  }

  getHead(world: string): number {
    return this.heads.get(world) ?? this.data.heads.get(world) ?? 0;
  }

  setHead(world: string, head: number): void {
    this.heads.set(world, head);
  }

  insertLog(world: string, entry: LogEntry): void {
    if (this.data.opIds.has(`${world}\u0000${entry.opId}`) || (this.data.logs.get(world)?.length ?? 0) >= entry.offset) {
      throw new JournalError(`Shared log entry ${entry.offset} or op ${entry.opId} already exists`);
    }
    this.logs.push([world, clone(entry)]);
  }

  findOp(world: string, opId: string): LogEntry | null {
    const offset = this.data.opIds.get(`${world}\u0000${opId}`);
    return offset === undefined ? null : clone((this.data.logs.get(world) as LogEntry[])[offset - 1]);
  }

  readLog(world: string, after: number): LogEntry[] {
    return clone((this.data.logs.get(world) ?? []).slice(after));
  }

  putSnapshot(world: string, offset: number, body: JsonObject): void {
    this.snapshots.push([world, offset, clone(body)]);
  }

  commit(): void {
    for (const [id, body] of this.projects) this.data.projects.set(id, body);
    for (const [id, body] of this.tasks) this.data.tasks.set(id, body);
    for (const record of this.controls) this.data.controls.set(record.sessionId, record);
    for (const message of this.signals) {
      let list = this.data.inbox.get(message.sessionId);
      if (list === undefined) this.data.inbox.set(message.sessionId, (list = []));
      list.push(message);
    }
    for (const [sessionId, id, runId] of this.consumed) {
      const list = this.data.inbox.get(sessionId) as InboxMessage[];
      list[id - 1] = { ...list[id - 1], consumedBy: runId };
    }
    for (const [key, record] of this.runs) this.data.runs.set(key, record);
    for (const [key, entry] of this.entries) {
      let map = this.data.entries.get(key);
      if (map === undefined) this.data.entries.set(key, (map = new Map()));
      map.set(entryKey(entry), entry);
    }
    for (const [key, value] of this.sessions) this.data.sessions.set(key, value);
    for (const [world, entry] of this.logs) {
      let log = this.data.logs.get(world);
      if (log === undefined) this.data.logs.set(world, (log = []));
      log.push(entry);
      this.data.opIds.set(`${world}\u0000${entry.opId}`, entry.offset);
    }
    for (const [world, head] of this.heads) this.data.heads.set(world, head);
    for (const [world, offset, body] of this.snapshots) {
      let map = this.data.snapshots.get(world);
      if (map === undefined) this.data.snapshots.set(world, (map = new Map()));
      map.set(offset, body);
    }
    for (const [sessionId, runId, key, event] of this.events) {
      const id = runKey(sessionId, runId);
      let keys = this.data.eventKeys.get(id);
      if (keys === undefined) this.data.eventKeys.set(id, (keys = new Set()));
      if (keys.has(key)) continue;
      keys.add(key);
      this.data.events.push({
        position: this.data.events.length + 1,
        key,
        event: { ...clone(event), sequence: keys.size - 1 },
      });
    }
  }
}

function selectInbox(messages: readonly InboxMessage[], query: InboxQuery): InboxMessage[] {
  const kinds = query.kinds === undefined ? null : new Set(query.kinds);
  const selected = messages.filter(
    (item) =>
      item.id > (query.after ?? 0) && (kinds === null || kinds.has(item.kind)) && (!query.pending || item.consumedBy === null),
  );
  return query.limit === undefined ? selected : selected.slice(0, query.limit);
}

const PHASE_ORDER: Record<Phase, number> = { scheduled: 0, completed: 1, marker: 2 };

function sortEntries(entries: JournalEntry[]): JournalEntry[] {
  return entries.sort((a, b) => {
    const am = a.phase === "marker" ? 1 : 0;
    const bm = b.phase === "marker" ? 1 : 0;
    return am - bm || a.seq - b.seq || PHASE_ORDER[a.phase] - PHASE_ORDER[b.phase];
  });
}

/** In-process durable store for tests and experiments. Survives nothing but itself. */
export class MemoryStore extends DurableStore {
  readonly data: MemoryStoreData;

  constructor(data?: MemoryStoreData) {
    super();
    this.data = data ?? new MemoryStoreData();
  }

  protected writeBatch(ops: Op[]): unknown[] {
    return ops.map((op) => {
      const tx = new MemoryTx(this.data);
      try {
        const result = apply(tx, op);
        tx.commit();
        this.publish();
        return result;
      } catch (error) {
        this.discard();
        return new OpFailure(error);
      }
    });
  }

  getRun(sessionId: string, runId: string): RunRecord | null {
    return clone(this.data.runs.get(runKey(sessionId, runId)) ?? null);
  }

  runs({ status, sessionId }: { status?: string; sessionId?: string } = {}): RunRecord[] {
    return clone(
      [...this.data.runs.values()].filter(
        (record) => (status === undefined || record.status === status) && (sessionId === undefined || record.sessionId === sessionId),
      ),
    );
  }

  journal(sessionId: string, runId: string): JournalEntry[] {
    return sortEntries(clone([...(this.data.entries.get(runKey(sessionId, runId))?.values() ?? [])]));
  }

  load(sessionId: string): Session | null {
    validateSessionId(sessionId);
    return clone(this.data.sessions.get(sessionId) ?? null);
  }

  events(sessionId: string, { runId, after = 0, limit }: { runId?: string; after?: number; limit?: number } = {}): LoggedEvent[] {
    const found = this.data.events
      .slice(after)
      .filter((stored) => stored.event.sessionId === sessionId && (runId === undefined || stored.event.runId === runId));
    return clone(limit === undefined ? found : found.slice(0, limit));
  }

  inbox(sessionId: string, query: InboxQuery = {}): InboxMessage[] {
    return clone(selectInbox(this.data.inbox.get(sessionId) ?? [], query));
  }

  control(sessionId: string): ControlRecord | null {
    const record = this.data.controls.get(sessionId);
    return record === undefined ? null : clone(record);
  }

  boardProjects(): JsonObject[] {
    return clone([...this.data.projects.values()]);
  }

  boardTasks(filter: { project?: string; parent?: string; status?: string }): JsonObject[] {
    return clone(
      [...this.data.tasks.values()].filter(
        (task) =>
          (filter.project === undefined || task.project === filter.project) &&
          (filter.parent === undefined || task.parent === filter.parent) &&
          (filter.status === undefined || task.status === filter.status),
      ),
    );
  }

  eventKeys(sessionId: string, runId: string): Set<string> {
    return new Set(this.data.eventKeys.get(runKey(sessionId, runId)) ?? []);
  }

  sharedHead(world: string): number {
    return this.data.heads.get(world) ?? 0;
  }

  sharedLog(world: string, after: number, until: number | null): LogEntry[] {
    return clone((this.data.logs.get(world) ?? []).slice(after, until ?? undefined));
  }

  sharedSnapshot(world: string, atOrBefore: number | null): JsonObject | null {
    const snapshots = this.data.snapshots.get(world);
    if (snapshots === undefined) return null;
    const offsets = [...snapshots.keys()].filter((offset) => atOrBefore === null || offset <= atOrBefore);
    return offsets.length ? clone(snapshots.get(Math.max(...offsets)) as JsonObject) : null;
  }
}

// SQLite ------------------------------------------------------------------
const SCHEMA = `
CREATE TABLE IF NOT EXISTS runs (
  session_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  status TEXT NOT NULL,
  owner TEXT,
  epoch INTEGER NOT NULL,
  lease_expires_at REAL NOT NULL,
  active_seconds REAL NOT NULL,
  start TEXT NOT NULL,
  result TEXT,
  updated_at REAL NOT NULL,
  PRIMARY KEY (session_id, run_id)
);
CREATE INDEX IF NOT EXISTS runs_by_status ON runs (status, session_id);
CREATE TABLE IF NOT EXISTS journal (
  session_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  phase TEXT NOT NULL,
  kind TEXT NOT NULL,
  key TEXT NOT NULL,
  payload TEXT NOT NULL,
  epoch INTEGER NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (session_id, run_id, seq, phase)
);
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shared_head (
  world TEXT PRIMARY KEY,
  head INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS shared_log (
  world TEXT NOT NULL,
  offset INTEGER NOT NULL,
  op_id TEXT NOT NULL,
  entry TEXT NOT NULL,
  PRIMARY KEY (world, offset)
);
CREATE UNIQUE INDEX IF NOT EXISTS shared_log_op ON shared_log (world, op_id);
CREATE TABLE IF NOT EXISTS shared_snapshots (
  world TEXT NOT NULL,
  offset INTEGER NOT NULL,
  body TEXT NOT NULL,
  PRIMARY KEY (world, offset)
);
CREATE TABLE IF NOT EXISTS events (
  position INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  key TEXT NOT NULL,
  sequence INTEGER NOT NULL,
  kind TEXT NOT NULL,
  body TEXT NOT NULL,
  UNIQUE (session_id, run_id, key)
);
CREATE INDEX IF NOT EXISTS events_by_session ON events (session_id, position);
CREATE TABLE IF NOT EXISTS inbox (
  session_id TEXT NOT NULL,
  id INTEGER NOT NULL,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  at TEXT NOT NULL,
  consumed_by TEXT,
  PRIMARY KEY (session_id, id)
);
CREATE INDEX IF NOT EXISTS inbox_pending ON inbox (session_id, consumed_by, id);
CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  project TEXT NOT NULL,
  parent TEXT,
  status TEXT NOT NULL,
  created INTEGER NOT NULL,
  body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tasks_by_project ON tasks (project, created);
CREATE INDEX IF NOT EXISTS tasks_by_parent ON tasks (parent, created);
CREATE INDEX IF NOT EXISTS tasks_by_status ON tasks (status, created);
CREATE TABLE IF NOT EXISTS controls (
  session_id TEXT PRIMARY KEY,
  state TEXT NOT NULL,
  note TEXT,
  at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_run ON events (session_id, run_id, sequence);
`;

const RUN_COLUMNS = "session_id, run_id, agent, status, owner, epoch, lease_expires_at, active_seconds, start, result";

type RunRow = {
  session_id: string;
  run_id: string;
  agent: string;
  status: string;
  owner: string | null;
  epoch: number;
  lease_expires_at: number;
  active_seconds: number;
  start: string;
  result: string | null;
};

function recordFromRow(row: RunRow): RunRecord {
  return {
    sessionId: row.session_id,
    runId: row.run_id,
    agent: row.agent,
    status: row.status,
    owner: row.owner,
    epoch: row.epoch,
    leaseExpiresAt: row.lease_expires_at,
    activeSeconds: row.active_seconds,
    start: JSON.parse(row.start),
    result: row.result === null ? null : JSON.parse(row.result),
  };
}

function isConstraint(error: unknown): boolean {
  return error instanceof Error && /constraint/i.test(error.message);
}

type InboxRow = { session_id: string; id: number; kind: string; payload: string; at: string; consumed_by: string | null };

function messageFromRow(row: InboxRow): InboxMessage {
  return { id: row.id, sessionId: row.session_id, kind: row.kind, payload: JSON.parse(row.payload), at: row.at, consumedBy: row.consumed_by };
}

class SqliteTx implements Tx {
  constructor(private readonly db: Database) {}

  insertSignal(sessionId: string, kind: string, payload: JsonObject, at: string): number {
    const { next } = this.db.query("SELECT COALESCE(MAX(id), 0) + 1 AS next FROM inbox WHERE session_id = ?").get(sessionId) as { next: number };
    this.db.query("INSERT INTO inbox (session_id, id, kind, payload, at) VALUES (?, ?, ?, ?, ?)").run(sessionId, next, kind, JSON.stringify(payload), at);
    return next;
  }

  getProject(id: string): JsonObject | null {
    const row = this.db.query("SELECT body FROM projects WHERE id = ?").get(id) as { body: string } | null;
    return row === null ? null : JSON.parse(row.body);
  }

  putProject(id: string, status: string, body: JsonObject): void {
    this.db.query("INSERT OR REPLACE INTO projects (id, status, body) VALUES (?, ?, ?)").run(id, status, JSON.stringify(body));
  }

  getTask(id: string): JsonObject | null {
    const row = this.db.query("SELECT body FROM tasks WHERE id = ?").get(id) as { body: string } | null;
    return row === null ? null : JSON.parse(row.body);
  }

  putTask(id: string, project: string, parent: string | null, status: string, body: JsonObject): void {
    // Keeps its creation position on update, so tasks list in the order they were created.
    this.db
      .query(
        "INSERT INTO tasks (id, project, parent, status, created, body) VALUES (?, ?, ?, ?, (SELECT COALESCE(MAX(created), 0) + 1 FROM tasks), ?) " +
          "ON CONFLICT(id) DO UPDATE SET project = excluded.project, parent = excluded.parent, status = excluded.status, body = excluded.body",
      )
      .run(id, project, parent, status, JSON.stringify(body));
  }

  putControl(record: ControlRecord): void {
    this.db
      .query("INSERT OR REPLACE INTO controls (session_id, state, note, at) VALUES (?, ?, ?, ?)")
      .run(record.sessionId, record.state, record.note, record.at);
  }

  consumeSignal(sessionId: string, id: number, runId: string): void {
    this.db.query("UPDATE inbox SET consumed_by = ? WHERE session_id = ? AND id = ? AND consumed_by IS NULL").run(runId, sessionId, id);
    // Older Bun versions return nothing from run(): ask SQLite for the count.
    const { changed } = this.db.query("SELECT changes() AS changed").get() as { changed: number };
    if (changed !== 1) throw new JournalError(`Inbox message ${id} of session ${sessionId} does not exist or was already consumed`);
  }

  getRun(sessionId: string, runId: string): RunRecord | null {
    const row = this.db
      .query(`SELECT ${RUN_COLUMNS} FROM runs WHERE session_id = ? AND run_id = ?`)
      .get(sessionId, runId) as RunRow | null;
    return row ? recordFromRow(row) : null;
  }

  putRun(record: RunRecord): void {
    this.db
      .query(`INSERT OR REPLACE INTO runs (${RUN_COLUMNS}, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
      .run(
        record.sessionId,
        record.runId,
        record.agent,
        record.status,
        record.owner,
        record.epoch,
        record.leaseExpiresAt,
        record.activeSeconds,
        JSON.stringify(record.start),
        record.result === null ? null : JSON.stringify(record.result),
        Date.now() / 1000,
      );
  }

  insertEntries(sessionId: string, runId: string, epoch: number, entries: readonly JournalEntry[]): void {
    const insert = this.db.query(
      "INSERT INTO journal (session_id, run_id, seq, phase, kind, key, payload, epoch, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    );
    const now = Date.now() / 1000;
    try {
      for (const entry of entries) {
        insert.run(sessionId, runId, entry.seq, entry.phase, entry.kind, entry.key, JSON.stringify(entry.payload), epoch, now);
      }
    } catch (error) {
      if (isConstraint(error)) throw new JournalError(`Journal entry already exists for run ${runId}: ${(error as Error).message}`);
      throw error;
    }
  }

  putSession(session: Session): void {
    this.db.query("INSERT OR REPLACE INTO sessions (session_id, body) VALUES (?, ?)").run(session.sessionId, JSON.stringify(session));
  }

  insertEvents(sessionId: string, runId: string, events: readonly KeyedEvent[]): void {
    let { next } = this.db
      .query("SELECT COALESCE(MAX(sequence) + 1, 0) AS next FROM events WHERE session_id = ? AND run_id = ?")
      .get(sessionId, runId) as { next: number };
    const insert = this.db.query(
      "INSERT OR IGNORE INTO events (session_id, run_id, key, sequence, kind, body) VALUES (?, ?, ?, ?, ?, ?)",
    );
    const changes = this.db.query("SELECT changes() AS n");
    for (const [key, event] of events) {
      insert.run(sessionId, runId, key, next, event.kind, JSON.stringify({ ...event, sequence: next }));
      next += (changes.get() as { n: number }).n;
    }
  }

  getHead(world: string): number {
    const row = this.db.query("SELECT head FROM shared_head WHERE world = ?").get(world) as { head: number } | null;
    return row?.head ?? 0;
  }

  setHead(world: string, head: number): void {
    this.db.query("INSERT OR REPLACE INTO shared_head (world, head) VALUES (?, ?)").run(world, head);
  }

  insertLog(world: string, entry: LogEntry): void {
    try {
      this.db
        .query("INSERT INTO shared_log (world, offset, op_id, entry) VALUES (?, ?, ?, ?)")
        .run(world, entry.offset, entry.opId, JSON.stringify(entry));
    } catch (error) {
      if (isConstraint(error)) throw new JournalError(`Shared log entry ${entry.offset} or op ${entry.opId} already exists`);
      throw error;
    }
  }

  findOp(world: string, opId: string): LogEntry | null {
    const row = this.db.query("SELECT entry FROM shared_log WHERE world = ? AND op_id = ?").get(world, opId) as
      | { entry: string }
      | null;
    return row ? JSON.parse(row.entry) : null;
  }

  readLog(world: string, after: number): LogEntry[] {
    const rows = this.db
      .query("SELECT entry FROM shared_log WHERE world = ? AND offset > ? ORDER BY offset")
      .all(world, after) as { entry: string }[];
    return rows.map((row) => JSON.parse(row.entry));
  }

  putSnapshot(world: string, offset: number, body: JsonObject): void {
    this.db
      .query("INSERT OR REPLACE INTO shared_snapshots (world, offset, body) VALUES (?, ?, ?)")
      .run(world, offset, JSON.stringify(body));
  }
}

/**
 * Single-file durable store (WAL, synchronous=FULL) for one process on one host.
 *
 * Writes are batched by the group committer: one transaction per batch, one
 * savepoint per operation.
 */
export class SqliteStore extends DurableStore {
  readonly path: string;
  /** Other processes may write to the file: a waiting run also checks its inbox this often. */
  protected override readonly pollInterval: number | null = 0.25;
  /** @internal */
  readonly db: Database;

  constructor(path: string) {
    super();
    this.path = resolve(path);
    mkdirSync(dirname(this.path), { recursive: true });
    this.db = new Database(this.path, { create: true });
    this.db.exec("PRAGMA journal_mode=WAL");
    this.db.exec("PRAGMA synchronous=FULL");
    this.db.exec("PRAGMA busy_timeout=5000");
    const { user_version: version } = this.db.query("PRAGMA user_version").get() as { user_version: number };
    if (![0, 1, 2, 3, 4, SCHEMA_VERSION].includes(version)) {
      throw new JournalError(`Unsupported journal schema version ${version} in ${this.path}`);
    }
    // One statement per call: older Bun versions run only the first statement of a string.
    for (const statement of SCHEMA.split(";")) if (statement.trim()) this.db.exec(statement);
    this.db.exec(`PRAGMA user_version=${SCHEMA_VERSION}`);
  }

  close(): void {
    this.db.close();
  }

  protected writeBatch(ops: Op[]): unknown[] {
    const tx = new SqliteTx(this.db);
    const results: unknown[] = [];
    try {
      this.db.exec("BEGIN IMMEDIATE");
      for (const op of ops) {
        this.db.exec("SAVEPOINT op");
        try {
          results.push(apply(tx, op));
          this.db.exec("RELEASE op");
        } catch (error) {
          this.db.exec("ROLLBACK TO op");
          this.db.exec("RELEASE op");
          results.push(new OpFailure(error));
        }
      }
      this.db.exec("COMMIT");
    } catch (error) {
      this.discard();
      if (this.db.inTransaction) this.db.exec("ROLLBACK");
      throw new JournalError(`Cannot write journal ${this.path}: ${(error as Error).message}`, { cause: error });
    }
    this.publish();
    return results;
  }

  getRun(sessionId: string, runId: string): RunRecord | null {
    return new SqliteTx(this.db).getRun(sessionId, runId);
  }

  runs({ status, sessionId }: { status?: string; sessionId?: string } = {}): RunRecord[] {
    const clauses: string[] = [];
    const parameters: string[] = [];
    if (status !== undefined) {
      clauses.push("status = ?");
      parameters.push(status);
    }
    if (sessionId !== undefined) {
      clauses.push("session_id = ?");
      parameters.push(sessionId);
    }
    const where = clauses.length ? ` WHERE ${clauses.join(" AND ")}` : "";
    return (this.db.query(`SELECT ${RUN_COLUMNS} FROM runs${where}`).all(...parameters) as RunRow[]).map(recordFromRow);
  }

  journal(sessionId: string, runId: string): JournalEntry[] {
    const rows = this.db
      .query("SELECT seq, phase, kind, key, payload FROM journal WHERE session_id = ? AND run_id = ?")
      .all(sessionId, runId) as { seq: number; phase: Phase; kind: CommandKind; key: string; payload: string }[];
    return sortEntries(rows.map((row) => ({ ...row, payload: JSON.parse(row.payload) })));
  }

  load(sessionId: string): Session | null {
    validateSessionId(sessionId);
    const row = this.db.query("SELECT body FROM sessions WHERE session_id = ?").get(sessionId) as { body: string } | null;
    if (row === null) return null;
    const loaded = SessionSchema.parse(JSON.parse(row.body));
    if (loaded.sessionId !== sessionId) throw new Error("Session checkpoint ID does not match its key");
    return loaded;
  }

  events(sessionId: string, { runId, after = 0, limit }: { runId?: string; after?: number; limit?: number } = {}): LoggedEvent[] {
    let sql = "SELECT position, key, body FROM events WHERE session_id = ? AND position > ?";
    const parameters: (string | number)[] = [sessionId, after];
    if (runId !== undefined) {
      sql += " AND run_id = ?";
      parameters.push(runId);
    }
    sql += " ORDER BY position";
    if (limit !== undefined) {
      sql += " LIMIT ?";
      parameters.push(limit);
    }
    return (this.db.query(sql).all(...parameters) as { position: number; key: string; body: string }[]).map((row) => ({
      position: row.position,
      key: row.key,
      event: JSON.parse(row.body),
    }));
  }

  boardProjects(): JsonObject[] {
    return (this.db.query("SELECT body FROM projects ORDER BY rowid").all() as { body: string }[]).map((row) => JSON.parse(row.body));
  }

  boardTasks(filter: { project?: string; parent?: string; status?: string }): JsonObject[] {
    const clauses: string[] = [];
    const params: string[] = [];
    for (const name of ["project", "parent", "status"] as const) {
      if (filter[name] === undefined) continue;
      clauses.push(`${name} = ?`);
      params.push(filter[name] as string);
    }
    const where = clauses.length ? ` WHERE ${clauses.join(" AND ")}` : "";
    return (this.db.query(`SELECT body FROM tasks${where} ORDER BY created`).all(...params) as { body: string }[]).map((row) => JSON.parse(row.body));
  }

  control(sessionId: string): ControlRecord | null {
    const row = this.db.query("SELECT session_id, state, note, at FROM controls WHERE session_id = ?").get(sessionId) as {
      session_id: string;
      state: ControlState;
      note: string | null;
      at: string;
    } | null;
    return row === null ? null : { sessionId: row.session_id, state: row.state, note: row.note, at: row.at };
  }

  inbox(sessionId: string, query: InboxQuery = {}): InboxMessage[] {
    const clauses = ["session_id = ?", "id > ?"];
    const params: (string | number)[] = [sessionId, query.after ?? 0];
    if (query.kinds !== undefined) {
      if (!query.kinds.length) return [];
      clauses.push(`kind IN (${query.kinds.map(() => "?").join(", ")})`);
      params.push(...query.kinds);
    }
    if (query.pending) clauses.push("consumed_by IS NULL");
    const limit = query.limit === undefined ? "" : ` LIMIT ${Math.max(0, Math.floor(query.limit))}`;
    const rows = this.db.query(`SELECT * FROM inbox WHERE ${clauses.join(" AND ")} ORDER BY id${limit}`).all(...params) as InboxRow[];
    return rows.map(messageFromRow);
  }

  eventKeys(sessionId: string, runId: string): Set<string> {
    const rows = this.db.query("SELECT key FROM events WHERE session_id = ? AND run_id = ?").all(sessionId, runId) as {
      key: string;
    }[];
    return new Set(rows.map((row) => row.key));
  }

  sharedHead(world: string): number {
    return new SqliteTx(this.db).getHead(world);
  }

  sharedLog(world: string, after: number, until: number | null): LogEntry[] {
    const rows = this.db
      .query("SELECT entry FROM shared_log WHERE world = ? AND offset > ? AND offset <= ? ORDER BY offset")
      .all(world, after, until ?? Number.MAX_SAFE_INTEGER) as { entry: string }[];
    return rows.map((row) => JSON.parse(row.entry));
  }

  sharedSnapshot(world: string, atOrBefore: number | null): JsonObject | null {
    const row = this.db
      .query("SELECT body FROM shared_snapshots WHERE world = ? AND offset <= ? ORDER BY offset DESC LIMIT 1")
      .get(world, atOrBefore ?? Number.MAX_SAFE_INTEGER) as { body: string } | null;
    return row ? JSON.parse(row.body) : null;
  }
}

// Per-run view used by Runtime ------------------------------------------------
export interface RunJournalOptions {
  readonly?: boolean;
  untilSeq?: number | null;
  /** Up to this seq (a fork's copied prefix), recorded results match by position and kind, not key. */
  lenientUntil?: number;
  eventKeys?: Iterable<string>;
}

/**
 * One owner's view of a run's journal: recorded entries plus appends at its epoch.
 *
 * `readonly` (inspection) never writes; reaching an unrecorded command, or one
 * past `untilSeq`, raises ReplayExhausted.
 */
export class RunJournal {
  readonly readonly: boolean;
  readonly untilSeq: number | null;
  readonly lenientUntil: number;
  readonly markers = new Map<string, number>();
  readonly lastRecordedSeq: number;
  private readonly recordedEntries = new Map<string, JournalEntry>();
  private nextMarker: number;
  private queued: Promise<void>[] = [];
  /** Events not yet stored; each write carries them. Keys already stored are skipped. */
  private bufferedEvents: KeyedEvent[] = [];
  private readonly knownEventKeys: Set<string>;

  constructor(
    readonly store: DurableStore | null,
    readonly sessionId: string,
    readonly runId: string,
    readonly epoch: number,
    entries: Iterable<JournalEntry> = [],
    options: RunJournalOptions = {},
  ) {
    this.readonly = options.readonly ?? false;
    this.untilSeq = options.untilSeq ?? null;
    this.lenientUntil = options.lenientUntil ?? 0;
    this.knownEventKeys = new Set(options.eventKeys ?? []);
    let lastMarker = 0;
    let lastSeq = 0;
    for (const entry of entries) {
      if (entry.phase === "marker") {
        this.markers.set(entry.key, entry.payload.version as number);
        lastMarker = Math.max(lastMarker, entry.seq);
      } else {
        this.recordedEntries.set(`${entry.seq}:${entry.phase}`, entry);
        lastSeq = Math.max(lastSeq, entry.seq);
      }
    }
    this.nextMarker = lastMarker + 1;
    this.lastRecordedSeq = lastSeq;
  }

  /** The latest usable checkpoint a replay may start from (at or before untilSeq), or null. */
  latestCheckpoint(): JournalEntry | null {
    let latest: JournalEntry | null = null;
    for (const entry of this.recordedEntries.values()) {
      if (entry.kind !== "checkpoint" || entry.phase !== "completed" || entry.payload.skipped) continue;
      if (this.untilSeq !== null && entry.seq > this.untilSeq) continue;
      if (latest === null || entry.seq > latest.seq) latest = entry;
    }
    return latest;
  }

  recorded(seq: number): [JournalEntry | null, JournalEntry | null] {
    return [this.recordedEntries.get(`${seq}:scheduled`) ?? null, this.recordedEntries.get(`${seq}:completed`) ?? null];
  }

  check(seq: number, kind: CommandKind, key: string): [JournalEntry | null, JournalEntry | null] {
    if (this.untilSeq !== null && seq > this.untilSeq) throw new ReplayExhausted(seq, `untilSeq=${this.untilSeq}`);
    const [scheduled, completed] = this.recorded(seq);
    const lenient = seq <= this.lenientUntil;
    for (const entry of [scheduled, completed]) {
      if (entry !== null && (entry.kind !== kind || (entry.key !== key && !lenient))) {
        throw new NonDeterminismError(seq, `${entry.kind} ${entry.key}`, `${kind} ${key}`);
      }
    }
    return [scheduled, completed];
  }

  recordEvent(key: string, event: Event): void {
    if (this.readonly || this.knownEventKeys.has(key)) return;
    this.knownEventKeys.add(key);
    this.bufferedEvents.push([key, event]);
  }

  takeEvents(): KeyedEvent[] {
    const events = this.bufferedEvents;
    this.bufferedEvents = [];
    return events;
  }

  /** Queue buffered events on their own, e.g. while a long tool call runs. */
  flushEvents(): void {
    if (this.bufferedEvents.length) this.enqueue();
  }

  async append(...entries: JournalEntry[]): Promise<void> {
    await this.appendConsuming([], ...entries);
  }

  /** Append entries and, in the same transaction, mark these inbox messages consumed by this run. */
  async appendConsuming(consume: readonly number[], ...entries: JournalEntry[]): Promise<void> {
    if (this.readonly) throw new ReplayExhausted(entries[0]?.seq ?? 0, "inspection replay does not write");
    await this.flush();
    await (this.store as DurableStore).append(this.sessionId, this.runId, this.epoch, entries, this.takeEvents(), consume);
  }

  /** Queue entries without waiting; a later append or flush() reports failures. */
  enqueue(...entries: JournalEntry[]): void {
    if (this.readonly) return;
    const pending = (this.store as DurableStore).append(this.sessionId, this.runId, this.epoch, entries, this.takeEvents());
    pending.catch(() => {}); // Reported by flush().
    this.queued.push(pending);
  }

  async flush(): Promise<void> {
    const queued = this.queued;
    this.queued = [];
    for (const pending of queued) await pending;
  }

  marker(changeId: string, version: number, atSeq: number): void {
    this.markers.set(changeId, version);
    // atSeq: commands issued before the decision, so a fork copies only its prefix's markers.
    const entry: JournalEntry = {
      seq: this.nextMarker,
      phase: "marker",
      kind: "version",
      key: changeId,
      payload: { version, atSeq },
    };
    this.nextMarker += 1;
    this.enqueue(entry);
  }
}
