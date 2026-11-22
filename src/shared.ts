/**
 * Shared state: one JSON tree per world, many agents, an append-only log.
 *
 * Paths are JSON Pointers (RFC 6901); writes are JSON Patch operations (RFC 6902).
 * Every committed entry gets a gap-free offset, so the world can be rebuilt at any
 * offset. Each path has a version (the offset of the last write affecting it), so
 * a write can require that what it read has not changed (`basedOn`).
 * See docs/durable-state.md, sections 6.1, 6.2 and 7.
 */

import { JournalError, TourbillonError } from "./errors.ts";
import { type DurableStore, type JournalEntry, type KeyedEvent, type Tx, commandKey, owned } from "./journal.ts";
import type { Runtime } from "./runtime.ts";
import { type Json, type JsonObject, hexId, stableStringify } from "./util.ts";

export const MAX_OPS = 100;
export const MAX_ENTRY_BYTES = 256 * 1024;
const MISSING = Symbol("missing");
type Found = Json | typeof MISSING;

export class SharedError extends TourbillonError {}

/** The patch is malformed. Raised before anything is journaled or committed. */
export class InvalidPatch extends SharedError {}

export interface RejectionResult {
  status: "rejected";
  error: string;
  message: string;
  paths: string[];
}
export type CommitResult = { status: "committed"; offset: number } | RejectionResult;

/** The store rejected a well-formed write; the rejection is journaled and replays. */
export class Rejected extends SharedError {
  static readonly code: string = "rejected";

  constructor(
    message: string,
    readonly paths: readonly string[] = [],
  ) {
    super(message);
  }

  toResult(): RejectionResult {
    return {
      status: "rejected",
      error: (this.constructor as typeof Rejected).code,
      message: this.message,
      paths: [...this.paths],
    };
  }
}

/** A path did not exist where the patch needed it, or a `test` op failed. */
export class PatchConflict extends Rejected {
  static override readonly code: string = "conflict";
}

/** A `basedOn` read is no longer current: something it covered was written since. */
export class StaleRead extends Rejected {
  static override readonly code: string = "stale";
}

const REJECTIONS: Record<string, typeof Rejected> = { conflict: PatchConflict, stale: StaleRead };

export interface Committed {
  readonly offset: number;
}

export function raiseFor(result: CommitResult): Committed {
  if (result.status === "committed") return { offset: result.offset };
  throw new REJECTIONS[result.error](result.message, result.paths);
}

// Paths ------------------------------------------------------------------------
/** JSON Pointer ("/a/b", "" for the root) or dotted shorthand ("a.b") to segments. */
export function parsePath(path: string): string[] {
  if (typeof path !== "string") throw new InvalidPatch("A path must be a string");
  if (path === "") return [];
  if (path.startsWith("/")) {
    return path
      .slice(1)
      .split("/")
      .map((part) => part.replaceAll("~1", "/").replaceAll("~0", "~"));
  }
  const parts = path.split(".");
  if (parts.some((part) => !part || part.includes("/") || part.includes("~"))) {
    throw new InvalidPatch(
      `Invalid path ${JSON.stringify(path)}: use a JSON Pointer such as '/a/b' for keys with '.', '/' or '~'`,
    );
  }
  return parts;
}

export function formatPath(segments: Iterable<string>): string {
  let text = "";
  for (const segment of segments) text += `/${segment.replaceAll("~", "~0").replaceAll("/", "~1")}`;
  return text;
}

export function canonicalPath(path: string): string {
  return formatPath(parsePath(path));
}

const isObject = (value: unknown): value is JsonObject =>
  value !== null && typeof value === "object" && !Array.isArray(value);

function arrayIndex(segment: string, length: number, append: boolean): number {
  if (append && segment === "-") return length;
  if (!/^\d+$/.test(segment) || (segment !== "0" && segment.startsWith("0"))) {
    throw new PatchConflict(`Invalid array index ${JSON.stringify(segment)}`);
  }
  const index = Number(segment);
  if (index > length || (index === length && !append)) throw new PatchConflict(`Array index ${index} out of range`);
  return index;
}

function child(value: Found, segment: string): Found {
  if (isObject(value)) return Object.hasOwn(value, segment) ? value[segment] : MISSING;
  if (Array.isArray(value)) {
    try {
      return value[arrayIndex(segment, value.length, false)];
    } catch {
      return MISSING;
    }
  }
  return MISSING;
}

function lookup(value: Json, segments: readonly string[]): Found {
  let current: Found = value;
  for (const segment of segments) {
    current = child(current, segment);
    if (current === MISSING) return MISSING;
  }
  return current;
}

/** A private, validated JSON copy of a value. */
function jsonCopy(value: unknown): Json {
  const check = (item: unknown): void => {
    if (typeof item === "number" && !Number.isFinite(item)) throw new InvalidPatch("Value is not JSON: non-finite number");
    if (item === undefined || typeof item === "function" || typeof item === "symbol" || typeof item === "bigint") {
      throw new InvalidPatch(`Value is not JSON: ${typeof item}`);
    }
    if (Array.isArray(item)) item.forEach(check);
    else if (item !== null && typeof item === "object") {
      const prototype = Object.getPrototypeOf(item);
      if (prototype !== Object.prototype && prototype !== null) throw new InvalidPatch("Value is not JSON: not a plain object");
      Object.values(item).forEach(check);
    }
  };
  check(value);
  return JSON.parse(JSON.stringify(value));
}

/** JSON equality: 1 and true differ, key order does not matter. */
function same(left: Json, right: Json): boolean {
  return stableStringify(left) === stableStringify(right);
}

// Patch ops ------------------------------------------------------------------------
export type PatchOp =
  | { op: "add" | "replace" | "test"; path: string; value: Json }
  | { op: "remove"; path: string };
export type PatchInput = { op: string; path?: string; value?: unknown; [key: string]: unknown };

const OPS = new Set(["add", "remove", "replace", "test"]);

/** Validate a patch and return it with canonical paths and private JSON values. */
export function normalizeOps(ops: Iterable<PatchInput>): PatchOp[] {
  if (!ops || typeof (ops as Iterable<unknown>)[Symbol.iterator] !== "function") {
    throw new InvalidPatch("A patch is a list of operations");
  }
  const normalized: PatchOp[] = [];
  for (const op of ops) {
    if (!isObject(op) || !OPS.has(op.op as string)) {
      throw new InvalidPatch(`Each operation needs 'op' in ${[...OPS].sort().join(", ")}: ${JSON.stringify(op)}`);
    }
    const extra = Object.keys(op).filter((key) => !["op", "path", "value"].includes(key));
    if (extra.length) throw new InvalidPatch(`Unsupported operation fields ${extra.sort().join(", ")}`);
    if (!("path" in op)) throw new InvalidPatch(`Operation '${op.op}' needs a path`);
    const path = canonicalPath(op.path as string);
    if (op.op === "remove") {
      if ("value" in op) throw new InvalidPatch("'remove' takes no value");
      normalized.push({ op: "remove", path });
    } else if (!("value" in op)) {
      throw new InvalidPatch(`'${op.op}' needs a value`);
    } else {
      normalized.push({ op: op.op as "add" | "replace" | "test", path, value: jsonCopy(op.value) });
    }
  }
  if (!normalized.length) throw new InvalidPatch("A patch needs at least one operation");
  if (normalized.length > MAX_OPS) throw new InvalidPatch(`A patch has at most ${MAX_OPS} operations`);
  if (JSON.stringify(normalized).length > MAX_ENTRY_BYTES) {
    throw new InvalidPatch(`A patch is at most ${MAX_ENTRY_BYTES} bytes; store large values elsewhere by reference`);
  }
  return normalized;
}

/** Copy the containers along `segments` and apply `change` to the last one. */
function setParent(value: Json, segments: readonly string[], change: (container: Json) => Json, path: string): Json {
  if (!segments.length) return change(value);
  const [head, ...rest] = segments;
  if (isObject(value)) {
    if (!Object.hasOwn(value, head)) throw new PatchConflict(`Path ${path} does not exist`, [path]);
    return { ...value, [head]: setParent(value[head], rest, change, path) };
  }
  if (Array.isArray(value)) {
    const index = arrayIndex(head, value.length, false);
    const copy = [...value];
    copy[index] = setParent(value[index], rest, change, path);
    return copy;
  }
  throw new PatchConflict(`Path ${path} does not exist`, [path]);
}

function applyOp(value: Json, segments: readonly string[], op: PatchOp): Json {
  const { path } = op;
  if (!segments.length) {
    if (op.op === "remove") throw new PatchConflict("The root cannot be removed", [path]);
    return (op as { value: Json }).value;
  }
  const parent = segments.slice(0, -1);
  const last = segments[segments.length - 1];
  const change = (container: Json): Json => {
    if (isObject(container)) {
      if (op.op === "add") return { ...container, [last]: op.value };
      if (!Object.hasOwn(container, last)) throw new PatchConflict(`Path ${path} does not exist`, [path]);
      const copy = { ...container };
      if (op.op === "remove") delete copy[last];
      else copy[last] = (op as { value: Json }).value;
      return copy;
    }
    if (Array.isArray(container)) {
      const index = arrayIndex(last, container.length, op.op === "add");
      const copy = [...container];
      if (op.op === "add") copy.splice(index, 0, op.value);
      else if (op.op === "remove") copy.splice(index, 1);
      else copy[index] = (op as { value: Json }).value;
      return copy;
    }
    throw new PatchConflict(`Parent of ${path} is not an object or array`, [path]);
  };
  return setParent(value, parent, change, path);
}

// Versions ------------------------------------------------------------------------
class VNode {
  constructor(
    /** Last write at exactly this path. */
    readonly own = 0,
    /** Last write at or below this path. */
    readonly sub = 0,
    readonly kids: ReadonlyMap<string, VNode> = new Map(),
  ) {}

  toJson(): JsonObject {
    const body: JsonObject = { o: this.own, s: this.sub };
    if (this.kids.size) body.k = Object.fromEntries([...this.kids].map(([key, node]) => [key, node.toJson()]));
    return body;
  }

  static fromJson(body: JsonObject): VNode {
    const kids = (body.k ?? {}) as Record<string, JsonObject>;
    return new VNode(
      body.o as number,
      body.s as number,
      new Map(Object.entries(kids).map(([key, kid]) => [key, VNode.fromJson(kid)])),
    );
  }
}

/** Versions stop at arrays: an index is not a stable identity, so a whole array is one unit. */
function versionTarget(value: Found, segments: readonly string[]): string[] {
  let current = value;
  for (let depth = 0; depth < segments.length; depth++) {
    if (Array.isArray(current)) return segments.slice(0, depth);
    current = child(current, segments[depth]);
  }
  return [...segments];
}

function bump(node: VNode, target: readonly string[], offset: number): VNode {
  // The value was replaced: older versions below no longer apply.
  if (!target.length) return new VNode(offset, offset);
  const kids = new Map(node.kids);
  kids.set(target[0], bump(node.kids.get(target[0]) ?? new VNode(), target.slice(1), offset));
  return new VNode(node.own, offset, kids);
}

// State ------------------------------------------------------------------------
/** A read: the value at `path` and the version a later write can be `basedOn`. */
export interface Versioned {
  readonly path: string;
  readonly value: Json;
  readonly version: number;
  readonly offset: number;
  readonly exists: boolean;
}

export interface Basis {
  readonly path: string;
  readonly version: number;
}

export interface LogEntry {
  readonly offset: number;
  readonly opId: string;
  readonly ops: PatchOp[];
  readonly basedOn: Basis[];
  readonly author: JsonObject | null;
  readonly committedAt: number;
}

/** An immutable world state at `offset`. Writes return a new state (path copying). */
export class SharedState {
  constructor(
    readonly value: Json = {},
    /** @internal */
    readonly versions: VNode = new VNode(),
    readonly offset = 0,
    readonly committedAt = 0,
  ) {}

  version(path: string): number {
    const segments = parsePath(path);
    let value: Found = this.value;
    let node = this.versions;
    let latest = node.own;
    for (const segment of segments) {
      if (Array.isArray(value)) break;
      const next = node.kids.get(segment);
      if (next === undefined) return latest; // Never written below here: only ancestor replacements matter.
      node = next;
      value = child(value, segment);
      latest = Math.max(latest, node.own);
    }
    return Math.max(latest, node.sub);
  }

  read(path: string): Versioned {
    const canonical = canonicalPath(path);
    const value = lookup(this.value, parsePath(canonical));
    const exists = value !== MISSING;
    return {
      path: canonical,
      value: exists ? structuredClone(value) : null,
      version: this.version(canonical),
      offset: this.offset,
      exists,
    };
  }

  stale(basedOn: Iterable<Basis>): string[] {
    return [...basedOn].filter((item) => this.version(item.path) !== item.version).map((item) => item.path);
  }

  /** Apply a normalized patch atomically; raises PatchConflict and changes nothing. */
  apply(ops: readonly PatchOp[], offset: number, committedAt: number): SharedState {
    let value = this.value;
    let versions = this.versions;
    for (const op of ops) {
      const segments = parsePath(op.path);
      if (op.op === "test") {
        const current = lookup(value, segments);
        if (current === MISSING || !same(current, op.value)) throw new PatchConflict(`Test failed at ${op.path}`, [op.path]);
        continue;
      }
      const target = versionTarget(value, segments);
      value = applyOp(value, segments, op);
      versions = bump(versions, target, offset);
    }
    return new SharedState(value, versions, offset, committedAt);
  }

  applyEntry(entry: LogEntry): SharedState {
    if (entry.offset !== this.offset + 1) {
      throw new JournalError(`Shared log gap: state at ${this.offset}, entry ${entry.offset}`);
    }
    return this.apply(entry.ops, entry.offset, entry.committedAt);
  }

  toJson(): JsonObject {
    return { value: this.value, versions: this.versions.toJson(), offset: this.offset, committedAt: this.committedAt };
  }

  static fromJson(body: JsonObject): SharedState {
    return new SharedState(
      body.value as Json,
      VNode.fromJson(body.versions as JsonObject),
      body.offset as number,
      body.committedAt as number,
    );
  }
}

// Commit operation (runs inside the store's write batch) -------------------------------
interface JournalTarget {
  sessionId: string;
  runId: string;
  epoch: number;
  seq: number;
  key: string;
  /** The run's buffered events, stored with the write. */
  events: readonly KeyedEvent[];
}

function sharedCommit(
  world: SharedWorld,
  opId: string,
  ops: readonly PatchOp[],
  basedOn: readonly Basis[],
  author: JsonObject | null,
  target: JournalTarget | null,
) {
  return {
    type: "custom" as const,
    applyIn(tx: Tx): CommitResult {
      if (target !== null) {
        owned(tx, target.sessionId, target.runId, target.epoch); // A stale owner cannot write shared state.
        if (target.events.length) tx.insertEvents(target.sessionId, target.runId, target.events);
      }
      const state = world.working(tx);
      const existing = tx.findOp(world.name, opId);
      let next: SharedState | null = null;
      let result: CommitResult;
      if (existing !== null) {
        result = { status: "committed", offset: existing.offset }; // Deduplicated retry.
      } else {
        try {
          const stale = state.stale(basedOn);
          if (stale.length) throw new StaleRead(`Changed since read: ${stale.join(", ")}`, stale);
          const committedAt = Math.max(Date.now() / 1000, state.committedAt);
          next = state.apply(ops, state.offset + 1, committedAt);
          const entry: LogEntry = {
            offset: next.offset,
            opId,
            ops: [...ops],
            basedOn: [...basedOn],
            author,
            committedAt,
          };
          tx.insertLog(world.name, entry);
          tx.setHead(world.name, entry.offset);
          if (entry.offset % world.snapshotEvery === 0) tx.putSnapshot(world.name, entry.offset, next.toJson());
          result = { status: "committed", offset: entry.offset };
        } catch (error) {
          if (!(error instanceof Rejected)) throw error;
          next = null;
          result = error.toResult();
        }
      }
      if (target !== null) {
        // The run's journal records the outcome in the same transaction as the log entry.
        const entry: JournalEntry = {
          seq: target.seq,
          phase: "completed",
          kind: "shared_write",
          key: target.key,
          payload: { result: result as unknown as JsonObject },
        };
        tx.insertEntries(target.sessionId, target.runId, target.epoch, [entry]);
      }
      if (next !== null) world.stage(next); // Last: nothing after this can fail and roll the op back.
      return result;
    },
  };
}

// World ------------------------------------------------------------------------
/**
 * A named JSON tree in a DurableStore, cached in memory for reads.
 *
 * Writes go through the store's committer and are validated in commit order
 * against the cached state. Run one writer process per store; other processes
 * may `refresh()` to follow the log.
 */
export class SharedWorld {
  readonly snapshotEvery: number;
  private published: SharedState;

  constructor(
    readonly store: DurableStore,
    readonly name = "default",
    options: { snapshotEvery?: number } = {},
  ) {
    if (!name || typeof name !== "string") throw new Error("A world needs a nonempty name");
    this.snapshotEvery = options.snapshotEvery ?? 1000;
    if (!(this.snapshotEvery >= 1)) throw new Error("snapshotEvery must be positive");
    if (store.worlds.has(name)) throw new Error(`World ${JSON.stringify(name)} is already open on this store`);
    this.published = this.load(null);
    store.worlds.set(name, this);
  }

  // Reading ---------------------------------------------------------------
  get offset(): number {
    return this.published.offset;
  }

  get state(): SharedState {
    return this.published;
  }

  read(path = ""): Versioned {
    return this.published.read(path);
  }

  async get(path = ""): Promise<Versioned> {
    return this.read(path);
  }

  /** The world as it was right after the entry at `offset` (time travel). */
  at(offset: number): SharedState {
    const head = this.store.sharedHead(this.name);
    if (!(offset >= 0 && offset <= head)) throw new Error(`Offset ${offset} is outside 0..${head}`);
    return this.load(offset);
  }

  log({ after = 0, until = null }: { after?: number; until?: number | null } = {}): LogEntry[] {
    return this.store.sharedLog(this.name, after, until);
  }

  /** Catch up with entries committed by another process. */
  refresh(): void {
    let state = this.published;
    for (const entry of this.store.sharedLog(this.name, state.offset, null)) state = state.applyEntry(entry);
    this.published = state;
  }

  // Writing ---------------------------------------------------------------
  /** Commit a patch outside any run. `opId` makes a retry idempotent. */
  async patch(ops: Iterable<PatchInput>, options: { basedOn?: Iterable<Versioned | Basis>; opId?: string } = {}): Promise<Committed> {
    const result = await this.commit(normalizeOps(ops), normalizeBasis(options.basedOn ?? []), options.opId ?? hexId(), null, null);
    return raiseFor(result);
  }

  /** @internal */
  commit(
    ops: readonly PatchOp[],
    basedOn: readonly Basis[],
    opId: string,
    author: JsonObject | null,
    target: JournalTarget | null,
  ): Promise<CommitResult> {
    return this.store.committer.submit<CommitResult>(sharedCommit(this, opId, ops, basedOn, author, target));
  }

  // Store hooks -------------------------------------------------------------
  private load(offset: number | null): SharedState {
    const snapshot = this.store.sharedSnapshot(this.name, offset);
    let state = snapshot !== null ? SharedState.fromJson(snapshot) : new SharedState();
    for (const entry of this.store.sharedLog(this.name, state.offset, offset)) state = state.applyEntry(entry);
    return state;
  }

  /** @internal State at the store's head, including earlier ops of the current batch. */
  working(tx: Tx): SharedState {
    let state = this.store.staged.get(this.name) ?? this.published;
    const head = tx.getHead(this.name);
    if (state.offset < head) {
      // Another process wrote; catch up inside this transaction.
      for (const entry of tx.readLog(this.name, state.offset)) state = state.applyEntry(entry);
    }
    if (state.offset !== head) {
      throw new JournalError(`World ${JSON.stringify(this.name)} cache at ${state.offset} but store head is ${head}`);
    }
    return state;
  }

  /** @internal */
  stage(state: SharedState): void {
    this.store.staged.set(this.name, state);
  }

  /** @internal */
  publish(state: SharedState): void {
    this.published = state;
  }
}

export function normalizeBasis(basedOn: Iterable<Versioned | Basis>): Basis[] {
  const basis: Basis[] = [];
  for (const item of basedOn) {
    const keys = isObject(item) ? Object.keys(item) : [];
    const isRead = keys.includes("exists") && keys.includes("offset");
    if (!isObject(item) || (!isRead && (keys.length !== 2 || !("path" in item) || !("version" in item)))) {
      throw new InvalidPatch("basedOn takes reads (Versioned) or { path, version } objects");
    }
    const { path, version } = item as unknown as Basis;
    if (!Number.isInteger(version) || version < 0) throw new InvalidPatch("A basedOn version is a nonnegative integer");
    basis.push({ path: canonicalPath(path), version });
  }
  return basis;
}

// Run-bound access ------------------------------------------------------------------
/** `runtime.shared`: reads and writes journaled as commands of the run. */
export class SharedView {
  constructor(
    private readonly runtime: Runtime,
    readonly world: SharedWorld,
  ) {}

  async get(path = ""): Promise<Versioned> {
    const runtime = this.runtime;
    runtime.ensureActive();
    const canonical = canonicalPath(path);
    const seq = runtime.nextSeq();
    const journal = runtime.journal;
    const key = commandKey("shared_read", { world: this.world.name, path: canonical });
    if (journal !== null) {
      const [, completed] = journal.check(seq, "shared_read", key);
      if (completed !== null) {
        runtime.applied = seq;
        return completed.payload.read as unknown as Versioned;
      }
      runtime.goLive();
    }
    const read = this.world.read(canonical);
    if (journal !== null) {
      // Queued without waiting: ordered writes make it durable before any later effect.
      journal.enqueue({ seq, phase: "completed", kind: "shared_read", key, payload: { read: read as unknown as JsonObject } });
    }
    runtime.applied = seq;
    return read;
  }

  /** Commit a patch; raises StaleRead or PatchConflict (recorded, so a replay raises them too). */
  async patch(ops: Iterable<PatchInput>, options: { basedOn?: Iterable<Versioned | Basis> } = {}): Promise<Committed> {
    const runtime = this.runtime;
    runtime.ensureActive();
    const normalized = normalizeOps(ops);
    const basis = normalizeBasis(options.basedOn ?? []);
    const seq = runtime.nextSeq();
    const journal = runtime.journal;
    const key = commandKey("shared_write", { world: this.world.name, ops: normalized, basedOn: basis });
    let target: JournalTarget | null = null;
    if (journal !== null) {
      const [, completed] = journal.check(seq, "shared_write", key);
      if (completed !== null) {
        runtime.applied = seq;
        return raiseFor(completed.payload.result as unknown as CommitResult);
      }
      runtime.goLive();
      await journal.flush();
      target = { sessionId: journal.sessionId, runId: journal.runId, epoch: journal.epoch, seq, key, events: journal.takeEvents() };
    }
    const run = runtime.run;
    const result = await this.world.commit(
      normalized,
      basis,
      `${run.sessionId}:${run.id}:${seq}`,
      { session: run.sessionId, run: run.id, seq },
      target,
    );
    runtime.applied = seq;
    return raiseFor(result);
  }
}

/**
 * Shared access given to a tool call (`ctx.shared`).
 *
 * The tool call itself is journaled, so its reads are not. Writes get op IDs
 * derived from the call's idempotency key, so re-executing an idempotent tool
 * after a crash does not apply its writes twice, as long as it issues them in
 * the same order.
 */
export class ToolShared {
  private count = 0;

  constructor(
    readonly world: SharedWorld,
    private readonly opPrefix: string,
    private readonly author: JsonObject,
  ) {}

  async get(path = ""): Promise<Versioned> {
    return this.world.read(path);
  }

  async patch(ops: Iterable<PatchInput>, options: { basedOn?: Iterable<Versioned | Basis> } = {}): Promise<Committed> {
    this.count += 1;
    const result = await this.world.commit(
      normalizeOps(ops),
      normalizeBasis(options.basedOn ?? []),
      `${this.opPrefix}:${this.count}`,
      this.author,
      null,
    );
    return raiseFor(result);
  }
}
