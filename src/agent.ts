/** Agent configuration, plain and durable execution, recovery, replay and forks. */

import { resolve } from "node:path";
import {
  Cancelled,
  ConfigurationChanged,
  DeadlineExceeded,
  Frozen,
  LimitReached,
  ReplayExhausted,
  SessionLogError,
  NonDeterminismError,
  describeError,
  isStopError,
} from "./errors.ts";
import { validateHistory } from "./history.ts";
import {
  type DurableStore,
  INTERRUPTED_CONTENT,
  type JournalEntry,
  RunJournal,
  type RunRecord,
  runRecord,
} from "./journal.ts";
import { MESSAGE } from "./inbox.ts";
import { REPLY } from "./operator.ts";
import { type Loop, ReAct } from "./loops.ts";
import { type Provider, providerModel, providerName } from "./providers/base.ts";
import { Runtime } from "./runtime.ts";
import { SessionLog, validateSessionId } from "./sessionLog.ts";
import { MemorySessionStore, type Session, type SessionStore, session as makeSession } from "./sessions.ts";
import type { SharedWorld } from "./shared.ts";
import { Tools } from "./tools.ts";
import { type FlowSource, buildMermaid, renderGraph } from "./viz.ts";
import {
  type Limits,
  type LimitsInput,
  LimitsSchema,
  type Message,
  type MessageInput,
  Run,
  type ToolCall,
  message,
} from "./types.ts";
import { type JsonObject, abortable, childController, clone, hexId, sha256, sleep, stableStringify } from "./util.ts";

const LOG_IN_STORE =
  "An agent with a store keeps its session log in the store; read it with store.events(sessionId) " +
  "or store.exportLog(sessionId, path) instead of logDir";

/** Close interrupted calls without repeating effects whose outcome is unknown. */
function checkpointMessages(run: Run): Message[] {
  const messages = clone(run.messages.slice(1));
  const pending = new Map<string, ToolCall>();
  for (const item of messages) {
    if (item.role === "assistant") for (const call of item.toolCalls) pending.set(call.id, call);
    else if (item.role === "tool") pending.delete(item.toolCallId as string);
  }
  // A parallel batch may have returned some results before a sibling failed.
  const observed = new Map<string, MessageInput>();
  for (const event of run.events) {
    if (event.kind === "model_response" && event.data.conversation !== false) observed.clear();
    else if (event.kind === "tool_result") {
      const result = event.data.message as unknown as MessageInput;
      observed.set(result.toolCallId as string, result);
    }
  }
  for (const callId of pending.keys()) {
    const known = observed.get(callId);
    messages.push(known ? message(known) : message({ role: "tool", toolCallId: callId, isError: true, content: INTERRUPTED_CONTENT }));
  }
  validateHistory(messages);
  return messages;
}

/** Hash per configuration part a journal was recorded with, so a mismatch can name the part. */
function fingerprint(agent: Agent): Record<string, string> {
  let flow: unknown = null;
  const toFlow = (agent.loop as { toFlow?: () => { start: string; edgeList(): unknown } }).toFlow;
  if (typeof toFlow === "function") {
    try {
      const compiled = toFlow.call(agent.loop);
      flow = { start: compiled.start, edges: compiled.edgeList() };
    } catch {
      flow = "unavailable"; // A fingerprint must never fail the run.
    }
  }
  const body: Record<string, unknown> = {
    agent: agent.name,
    systemPrompt: agent.systemPrompt,
    loop: agent.loop.constructor?.name ?? "Loop",
    flow,
    tools: agent.tools.specs,
    provider: providerName(agent.provider),
    model: providerModel(agent.provider),
  };
  return Object.fromEntries(Object.entries(body).map(([name, value]) => [name, sha256(stableStringify(value)).slice(0, 16)]));
}

function changedParts(recorded: JsonObject, current: Record<string, string>): string[] {
  const parts = recorded.fingerprint;
  if (parts === null || typeof parts !== "object" || Array.isArray(parts)) return []; // Recorded before fingerprints.
  const names = new Set([...Object.keys(parts), ...Object.keys(current)]);
  return [...names].filter((name) => parts[name] !== current[name]).sort();
}

function lenientUntil(start: JsonObject): number {
  const forkedFrom = start.forkedFrom as JsonObject | undefined;
  return forkedFrom && typeof forkedFrom === "object" ? ((forkedFrom.atSeq as number) ?? 0) : 0;
}

interface Recovery {
  epoch: number;
  entries: readonly JournalEntry[];
  activeSeconds: number;
  lenientUntil: number;
  liveEvent: [string, JsonObject];
  eventKeys: Iterable<string>;
}

/**
 * A run rebuilt from its journal without performing any effect.
 *
 * `run` holds the messages, state, usage and events as of `commands` replayed
 * commands. `reason` says why replay stopped: `finished` (the loop ended),
 * `untilSeq` (the requested step), or `endOfJournal` (the next command was never
 * recorded, e.g. the run was interrupted there).
 */
export interface Replay {
  readonly run: Run;
  readonly commands: number;
  readonly reason: "finished" | "untilSeq" | "endOfJournal";
  readonly pendingToolCalls: readonly ToolCall[];
}

export interface AgentOptions {
  provider: Provider;
  tools?: Tools;
  loop?: Loop;
  systemPrompt?: string;
  limits?: LimitsInput;
  name?: string;
  /** A JSONL session log directory, for agents without a store. */
  logDir?: string | null;
  sessions?: SessionStore;
  /** Makes runs durable: journaled, recoverable, replayable. */
  store?: DurableStore | null;
  /** Seconds a run's lease lasts without a heartbeat. */
  leaseTtl?: number;
  shared?: SharedWorld | null;
}

export interface RunOptions {
  context?: Record<string, unknown>;
  /**
   * What this run is for, as plain JSON (a task's id, depth, allowed agents…).
   * Tools bind to it (Tool `available`, scoped `input`), and it is recorded, so
   * recovery and replay bind the same tools. A continuation without one reuses
   * the session's latest scope.
   */
  scope?: JsonObject;
  /**
   * Text a supervisor appends to the agent's system prompt for this session,
   * such as a task frame. It is fixed for the session: set it on the first run;
   * a continuation may omit it or must repeat it exactly.
   */
  frame?: string;
  /** Pin the prompt: compaction never removes it (see pinned messages). */
  pin?: boolean;
  /** The new run's id (default: a fresh one), so a caller can record it before the run starts. */
  runId?: string;
  history?: readonly MessageInput[];
  logDir?: string;
  sessionId?: string;
  parentSessionId?: string | null;
  parentRunId?: string | null;
  jobId?: string | null;
  /** Abort to cancel the run; it then rejects with Cancelled after recording the outcome. */
  signal?: AbortSignal;
}

/**
 * An agent configuration. Runs are durable when `store` is a DurableStore.
 *
 * With a store, sessions are checkpointed in it too unless `sessions` is given,
 * and the session log is kept in it with the journal: read it with
 * `store.events(sessionId)` or `store.exportLog(sessionId, path)`. `logDir`
 * (a JSONL file per session) is for agents without a store.
 */
export class Agent {
  readonly provider: Provider;
  readonly tools: Tools;
  readonly loop: Loop;
  readonly systemPrompt: string;
  readonly limits: Limits;
  readonly name: string;
  readonly logDir: string | null;
  readonly sessions: SessionStore;
  readonly store: DurableStore | null;
  readonly leaseTtl: number;
  readonly shared: SharedWorld | null;
  private readonly options: AgentOptions;
  /** run id -> the session frame its system prompt was built with. */
  private readonly frames = new Map<string, string | null>();

  constructor(options: AgentOptions) {
    this.options = options;
    this.provider = options.provider;
    this.tools = options.tools ?? new Tools();
    this.loop = options.loop ?? new ReAct();
    this.systemPrompt = options.systemPrompt ?? "You are a helpful assistant.";
    this.limits = LimitsSchema.parse(options.limits ?? {});
    this.name = options.name ?? "agent";
    this.logDir = options.logDir ?? null;
    this.store = options.store ?? null;
    this.sessions = options.sessions ?? this.store ?? new MemorySessionStore();
    this.leaseTtl = options.leaseTtl ?? 30;
    this.shared = options.shared ?? null;
    if (!(this.leaseTtl > 0)) throw new Error("leaseTtl must be positive");
    if (this.store !== null && this.logDir !== null) throw new Error(LOG_IN_STORE);
    // One store: a shared write and the run's journal row must commit together.
    if (this.shared !== null && this.store !== null && this.shared.store !== this.store) {
      throw new Error("Agent.shared must be a world in the agent's own store");
    }
  }

  /** A copy with some options replaced (the session store is kept unless given). */
  with(changes: Partial<AgentOptions>): Agent {
    return new Agent({ ...this.options, sessions: this.sessions, ...changes });
  }

  /** Mermaid for a Flow loop, or a loop exposing toFlow(). */
  flowGraph(options: { direction?: string } = {}): string {
    return buildMermaid(this.loop as unknown as FlowSource, options);
  }

  /** Save a PNG or SVG locally using Graphviz. */
  renderGraph(filename: string, options: { direction?: string; dot?: string } = {}): Promise<string> {
    return renderGraph(this.loop as unknown as FlowSource, filename, options);
  }

  /**
   * Start a conversation, or continue a known session with a fresh run ID.
   *
   * A supplied, unknown ID creates a named session. Context and loop state are
   * invocation-local; only conversation (including provider metadata) resumes.
   */
  async run(prompt: string, options: RunOptions = {}): Promise<Run> {
    const history = (options.history ?? []).map((item) => message(item));
    validateHistory(history);
    if (this.store !== null && options.logDir !== undefined) throw new Error(LOG_IN_STORE);
    const sessionId = options.sessionId ?? hexId();
    validateSessionId(sessionId);
    if (sessionId === options.parentSessionId) throw new Error("A child cannot run in its active parent session");
    return this.sessions.withLock(sessionId, async () => {
      if (this.store !== null) {
        const interrupted = this.store.runs({ sessionId, status: "running" });
        if (interrupted.length) {
          throw new Error(`Session has an interrupted run ${interrupted[0].runId}; call Agent.recover(sessionId) first`);
        }
      }
      const previous = this.sessions.load(sessionId);
      if (previous !== null) {
        if (history.length) throw new Error("Cannot combine history with an existing session");
        if (previous.agent !== this.name || previous.systemPrompt !== this.systemPrompt) {
          throw new Error("Session belongs to a different agent configuration");
        }
        if (options.frame !== undefined && options.frame !== (previous.frame ?? undefined)) {
          throw new Error("A session's frame is fixed: omit it, or pass the same frame, to continue");
        }
        // With a store, interrupted runs are found in the journal, not the checkpoint.
        if (previous.status === "running" && this.store === null) {
          throw new Error("Session was interrupted before checkpointing; inspect it before recovery");
        }
      }
      const frame = options.frame ?? previous?.frame ?? null;
      const run = new Run({
        id: options.runId,
        sessionId,
        parentSessionId: options.parentSessionId ?? null,
        parentRunId: options.parentRunId ?? null,
        jobId: options.jobId ?? null,
        context: { ...(options.context ?? {}) },
        scope: options.scope ?? (previous?.scope as JsonObject | null) ?? {},
        messages: [
          message({ role: "system", content: this.promptWith(frame) }),
          ...clone(previous?.messages ?? history),
          message({ role: "user", content: prompt, metadata: options.pin ? { pinned: true } : {} }),
        ],
      });
      this.frames.set(run.id, frame);
      let directory = options.logDir ?? this.logDir;
      if (previous?.logPath) {
        const previousDirectory = resolve(previous.logPath, "..");
        if (directory !== null && resolve(directory) !== previousDirectory) {
          throw new Error("A resumed session must keep its original log directory");
        }
        directory = previousDirectory;
      }
      return this.execute(run, { directory, resumeLog: Boolean(previous?.logPath), signal: options.signal });
    });
  }

  /**
   * Send a message to a session ("send now"): its agent reads it before the
   * next model turn, and a suspended run of the session wakes for it. Needs a
   * durable store; the message waits in the inbox if no run is active.
   */
  send(sessionId: string, text: string, options: { from?: string; kind?: string } = {}): Promise<number> {
    if (this.store === null) throw new Error("send() requires an Agent with a durable store");
    if (typeof text !== "string" || !text) throw new Error("A message needs text");
    const payload: JsonObject = { text, ...(options.from ? { from: options.from } : {}) };
    return this.store.signal(sessionId, options.kind ?? MESSAGE, payload);
  }

  /** Answer the question the session's agent asked with ask_operator. */
  reply(sessionId: string, text: string): Promise<number> {
    if (typeof text !== "string" || !text) throw new Error("A reply needs text");
    return this.requireStore("reply").signal(sessionId, REPLY, { text });
  }

  /** Hold the session's runs at their next step (the timeout clock stops); resume() lets them go on. */
  hold(sessionId: string, note?: string): Promise<void> {
    return this.requireStore("hold").setControl(sessionId, "hold", note ?? null);
  }

  /** Let a held session go on. A frozen one is resumed with recover() instead. */
  resume(sessionId: string): Promise<void> {
    return this.requireStore("resume").setControl(sessionId, "run");
  }

  /**
   * Freeze the session's run: it stops at its next step (or at once while
   * suspended), writes a checkpoint and releases its lease. agent.run() then
   * returns the run with status "frozen"; recover() continues it later.
   */
  freeze(sessionId: string, note?: string): Promise<void> {
    return this.requireStore("freeze").setControl(sessionId, "freeze", note ?? null);
  }

  private requireStore(action: string): DurableStore {
    if (this.store === null) throw new Error(`${action}() requires an Agent with a durable store`);
    return this.store;
  }

  /**
   * Resume the session's interrupted run from its journal.
   *
   * Recorded model calls and tool results are replayed without repeating them.
   * Supply `context` again: live resources are never journaled. `force` takes
   * over a run whose previous owner still holds an unexpired lease; use it only
   * when that owner is known to be gone.
   *
   * A configuration that differs from the recorded one (loop, flow, tools,
   * provider, model) raises ConfigurationChanged, naming the parts. With
   * `allowFingerprintChange` recovery proceeds; commands that no longer match
   * the journal still raise NonDeterminismError.
   */
  async recover(
    sessionId: string,
    options: { context?: Record<string, unknown>; force?: boolean; allowFingerprintChange?: boolean; signal?: AbortSignal } = {},
  ): Promise<Run> {
    const store = this.store;
    if (store === null) throw new Error("recover() requires an Agent with a durable store");
    validateSessionId(sessionId);
    return this.sessions.withLock(sessionId, async () => {
      const interrupted = store.runs({ sessionId, status: "running" });
      if (!interrupted.length) throw new Error(`Session ${sessionId} has no interrupted run`);
      const record = interrupted[0];
      const start = record.start;
      // Recovering is how a frozen session resumes.
      if (store.control(sessionId)?.state === "freeze") await store.setControl(sessionId, "run");
      if (record.agent !== this.name || start.systemPrompt !== this.promptWith((start.frame as string | null) ?? null)) {
        throw new Error("Run belongs to a different agent configuration");
      }
      const changed = changedParts(start, fingerprint(this));
      if (changed.length && !options.allowFingerprintChange) throw new ConfigurationChanged(changed);
      const epoch = await store.claimRun(sessionId, record.runId, { ttl: this.leaseTtl, force: options.force });
      const run = this.runFromRecord(record, options.context);
      return this.execute(run, {
        directory: null,
        resumeLog: true,
        signal: options.signal,
        recovery: {
          epoch,
          entries: store.journal(sessionId, record.runId),
          activeSeconds: record.activeSeconds,
          lenientUntil: lenientUntil(start),
          liveEvent: ["run_recovered", {}],
          eventKeys: store.eventKeys(sessionId, record.runId),
        },
      });
    });
  }

  /**
   * Rebuild a run's state from its journal, optionally only up to command `untilSeq`.
   *
   * Nothing is written and no provider or tool is called. Use it to inspect a
   * finished or interrupted run at any step. Code that diverges from the journal
   * raises NonDeterminismError. A parallel tool batch is replayed whole:
   * `untilSeq` inside a batch stops before the batch.
   */
  async replay(
    sessionId: string,
    runId: string,
    options: { untilSeq?: number; context?: Record<string, unknown> } = {},
  ): Promise<Replay> {
    const store = this.store;
    if (store === null) throw new Error("replay() requires an Agent with a durable store");
    if (options.untilSeq !== undefined && options.untilSeq < 0) throw new Error("untilSeq must be nonnegative");
    const record = store.getRun(sessionId, runId);
    if (record === null) throw new Error(`Run ${runId} not found in session ${sessionId}`);
    const run = this.runFromRecord(record, options.context);
    this.frames.delete(run.id); // Only execute() reads it; a replay writes no record.
    const journal = new RunJournal(null, sessionId, runId, record.epoch, store.journal(sessionId, runId), {
      readonly: true,
      untilSeq: options.untilSeq ?? null,
      lenientUntil: lenientUntil(record.start),
    });
    const runtime = new Runtime(run, this.provider, this.tools.bind(run.scope), this.limits, { journal, shared: this.shared });
    let reason: Replay["reason"] = "finished";
    try {
      await this.loop.run(runtime);
    } catch (error) {
      if (error instanceof ReplayExhausted) reason = error.reason.startsWith("untilSeq") ? "untilSeq" : "endOfJournal";
      else if (error instanceof LimitReached) [run.status, run.error] = ["limit", error.message];
      else if (error instanceof NonDeterminismError) throw error;
      // A replayed failure is part of the inspected state.
      else [run.status, run.error] = ["failed", describeError(error)];
    }
    return { run, commands: runtime.seq, reason, pendingToolCalls: clone(runtime.pending) };
  }

  /**
   * Start a new run that reuses a recorded run's first `atSeq` commands, then runs live.
   *
   * Call it on an agent with the configuration to try, e.g.
   * `agent.with({ provider: other }).fork(...)`. Recorded results of the prefix
   * are reused by position, so a changed system prompt or tool set does not
   * invalidate them. The fork gets its own session (`newSessionId` or a fresh ID)
   * and its own journal and log, so it can itself be recovered or forked.
   */
  async fork(
    sessionId: string,
    runId: string,
    options: { atSeq: number; context?: Record<string, unknown>; newSessionId?: string; signal?: AbortSignal },
  ): Promise<Run> {
    const store = this.store;
    if (store === null) throw new Error("fork() requires an Agent with a durable store");
    const { atSeq } = options;
    if (!(atSeq >= 0)) throw new Error("atSeq must be nonnegative");
    const source = store.getRun(sessionId, runId);
    if (source === null) throw new Error(`Run ${runId} not found in session ${sessionId}`);
    const newSessionId = options.newSessionId ?? hexId();
    validateSessionId(newSessionId);
    const entries = store
      .journal(sessionId, runId)
      .filter((entry) => (entry.phase === "marker" ? ((entry.payload.atSeq as number) ?? 0) <= atSeq : entry.seq <= atSeq));
    const forkedFrom = { sessionId, runId, atSeq };
    return this.sessions.withLock(newSessionId, async () => {
      if (this.sessions.load(newSessionId) !== null || store.runs({ sessionId: newSessionId }).length) {
        throw new Error(`Session ${newSessionId} already exists`);
      }
      const recorded = (source.start.messages as unknown as MessageInput[]).map((item) => message(item));
      const frame = (source.start.frame as string | null | undefined) ?? null;
      if (recorded.length && recorded[0].role === "system") recorded[0] = message({ role: "system", content: this.promptWith(frame) });
      const run = new Run({
        sessionId: newSessionId,
        context: { ...(options.context ?? {}) },
        scope: (source.start.scope as JsonObject | undefined) ?? {},
        messages: recorded,
      });
      this.frames.set(run.id, frame);
      await store.createRun(
        runRecord({
          sessionId: run.sessionId,
          runId: run.id,
          agent: this.name,
          owner: store.ownerId,
          epoch: 1,
          leaseExpiresAt: Date.now() / 1000 + this.leaseTtl,
          start: { ...this.startRecord(run), forkedFrom },
        }),
        entries,
      );
      return this.execute(run, {
        directory: null,
        resumeLog: false,
        signal: options.signal,
        forkedFrom,
        recovery: { epoch: 1, entries, activeSeconds: 0, lenientUntil: atSeq, liveEvent: ["fork_live", { forkedFrom }], eventKeys: [] },
      });
    });
  }

  /** The system prompt with a session's frame appended. */
  private promptWith(frame: string | null): string {
    return frame ? `${this.systemPrompt}\n\n${frame}` : this.systemPrompt;
  }

  private frameOf(run: Run): string | null {
    return this.frames.get(run.id) ?? null;
  }

  private startRecord(run: Run): JsonObject {
    return clone({
      systemPrompt: this.promptWith(this.frameOf(run)),
      frame: this.frameOf(run),
      scope: run.scope,
      messages: run.messages,
      limits: this.limits,
      loop: this.loop.constructor?.name ?? "Loop",
      fingerprint: fingerprint(this),
      parentSessionId: run.parentSessionId,
      parentRunId: run.parentRunId,
      jobId: run.jobId,
    }) as unknown as JsonObject;
  }

  private runFromRecord(record: RunRecord, context?: Record<string, unknown>): Run {
    const start = record.start;
    const run = new Run({
      id: record.runId,
      sessionId: record.sessionId,
      parentSessionId: (start.parentSessionId as string | null) ?? null,
      parentRunId: (start.parentRunId as string | null) ?? null,
      jobId: (start.jobId as string | null) ?? null,
      context: { ...(context ?? {}) },
      scope: (start.scope as JsonObject | undefined) ?? {},
      messages: (start.messages as unknown as MessageInput[]).map((item) => message(item)),
    });
    this.frames.set(run.id, (start.frame as string | null | undefined) ?? null);
    return run;
  }

  private session(run: Run, status: string): Session {
    return makeSession({
      sessionId: run.sessionId,
      agent: this.name,
      systemPrompt: this.systemPrompt,
      frame: this.frameOf(run),
      scope: run.scope,
      messages: checkpointMessages(run),
      status,
      logPath: run.logPath,
    });
  }

  private async checkpoint(run: Run, status: string): Promise<void> {
    const value = this.session(run, status);
    if (this.sessions.asave) await this.sessions.asave(value);
    else this.sessions.save(value);
  }

  private async execute(
    run: Run,
    options: {
      directory: string | null;
      resumeLog: boolean;
      signal?: AbortSignal;
      recovery?: Recovery;
      forkedFrom?: JsonObject;
    },
  ): Promise<Run> {
    const { recovery } = options;
    const store = this.store;
    let journal: RunJournal | null = null;
    let runtime: Runtime | null = null;
    let sessionLog: SessionLog | null = null;
    let stopped = false; // The journal must not be finished or checkpointed by this owner.
    let frozen = false;
    let rethrow: unknown = null;
    if (options.directory !== null) {
      // Opened before anything else: a log that cannot be created fails the run before any effect.
      sessionLog = new SessionLog(options.directory, run.sessionId, options.resumeLog).open();
      run.logPath = sessionLog.path;
      const log = sessionLog;
      run.eventSink = (event) => log.write(event);
    }
    const activeBefore = recovery?.activeSeconds ?? 0;
    const started = performance.now();
    // Time spent in a durable timer (runtime.sleep) is not active time.
    let pausedMs = 0;
    let pausedAt: number | null = null;
    const active = () =>
      activeBefore + (performance.now() - started - pausedMs - (pausedAt === null ? 0 : performance.now() - pausedAt)) / 1000;
    const controller = childController();
    const onCancel = () => controller.abort(new Cancelled("Run cancelled by caller"));
    if (options.signal?.aborted) onCancel();
    options.signal?.addEventListener("abort", onCancel, { once: true });
    // The timeout budget counts active execution only, not downtime between crash and recovery.
    const remaining = this.limits.timeout - activeBefore;
    const expire = () => controller.abort(new DeadlineExceeded("Run deadline exceeded"));
    if (remaining <= 0) expire();
    let deadline = setTimeout(expire, Math.max(remaining, 0) * 1000);
    const clock = {
      pause: () => {
        if (pausedAt !== null) return;
        pausedAt = performance.now();
        clearTimeout(deadline);
      },
      resume: () => {
        if (pausedAt === null) return;
        pausedMs += performance.now() - pausedAt;
        pausedAt = null;
        const left = this.limits.timeout - active();
        if (left <= 0) expire();
        else deadline = setTimeout(expire, left * 1000);
      },
    };
    const heartbeat = new AbortController();

    try {
      if (store !== null) {
        if (recovery === undefined) {
          await store.createRun(
            runRecord({
              sessionId: run.sessionId,
              runId: run.id,
              agent: this.name,
              owner: store.ownerId,
              epoch: 1,
              leaseExpiresAt: Date.now() / 1000 + this.leaseTtl,
              start: this.startRecord(run),
            }),
          );
          journal = new RunJournal(store, run.sessionId, run.id, 1);
        } else {
          journal = new RunJournal(store, run.sessionId, run.id, recovery.epoch, recovery.entries, {
            lenientUntil: recovery.lenientUntil,
            eventKeys: recovery.eventKeys,
          });
        }
        void this.heartbeat(store, journal, active, heartbeat.signal);
      } else {
        // Without a journal, a "running" checkpoint is how a crash is detected later.
        await this.checkpoint(run, "running");
      }
      runtime = new Runtime(run, this.provider, this.tools.bind(run.scope), this.limits, {
        journal,
        shared: this.shared,
        liveEvent: recovery?.liveEvent ?? null,
        signal: controller.signal,
        clock,
      });
      run.emit("run_start", {
        agent: this.name,
        loop: this.loop.constructor?.name ?? "Loop",
        cwd: process.cwd(),
        provider: providerName(this.provider),
        model: providerModel(this.provider),
        limits: this.limits,
        messages: run.messages,
        resumed: options.resumeLog,
        parentSessionId: run.parentSessionId,
        parentRunId: run.parentRunId,
        jobId: run.jobId,
        ...(options.forkedFrom ? { forkedFrom: options.forkedFrom } : {}),
      });
      const looping = this.loop.run(runtime);
      looping.catch(() => {}); // After a deadline the loop may still settle; its outcome is decided here.
      await abortable(looping, controller.signal);
      if (run.status === "running") throw new Error("Loop ended without runtime.finish(output)");
    } catch (error) {
      if (error instanceof Frozen) {
        stopped = true;
        frozen = true;
        [run.status, run.error] = ["frozen", null];
      } else if (isStopError(error)) {
        stopped = true;
        [run.status, run.error] = ["failed", describeError(error)];
        rethrow = error;
      } else if (error instanceof LimitReached) {
        [run.status, run.error] = ["limit", error.message];
      } else if (error instanceof DeadlineExceeded) {
        [run.status, run.error] = ["timeout", "Run or tool deadline exceeded; an in-flight effect may be incomplete"];
      } else if (error instanceof Cancelled) {
        [run.status, run.error] = ["cancelled", "Run cancelled by caller"];
        rethrow = error;
      } else if (error instanceof SessionLogError) {
        [run.status, run.error] = ["failed", error.message];
        rethrow = error; // Logging was requested: never silently run without it.
      } else {
        [run.status, run.error] = ["failed", describeError(error)];
      }
    } finally {
      clearTimeout(deadline);
      options.signal?.removeEventListener("abort", onCancel);
      heartbeat.abort();
      // Stop the loop's remaining work at its next runtime call.
      if (!controller.signal.aborted) controller.abort(new Cancelled("Run finished"));
    }

    try {
      if (run.status !== "completed") run.output = null;
      if (stopped) {
        run.eventSink = null;
        // Frozen: hand the run back without finishing it, so recover() needs no force.
        if (frozen && journal !== null) await (store as DurableStore).releaseRun(run.sessionId, run.id, journal.epoch);
      } else {
        try {
          runtime?.goLive();
          run.emit("run_end", { status: run.status, output: run.output, error: run.error, usage: run.usage });
        } finally {
          run.eventSink = null;
          if (journal === null) {
            await this.checkpoint(run, run.status);
          } else {
            await journal.flush(); // Queued side effects precede the final status.
            const outcome = {
              status: run.status,
              result: { output: run.output, error: run.error, usage: run.usage } as JsonObject,
              activeSeconds: active(),
            };
            if (this.sessions === store) {
              await (store as DurableStore).finishRun(run.sessionId, run.id, journal.epoch, {
                ...outcome,
                session: this.session(run, run.status),
                events: journal.takeEvents(),
              });
            } else {
              await (store as DurableStore).finishRun(run.sessionId, run.id, journal.epoch, { ...outcome, events: journal.takeEvents() });
              await this.checkpoint(run, run.status);
            }
          }
        }
      }
    } finally {
      sessionLog?.close();
      this.frames.delete(run.id);
    }
    if (rethrow !== null) throw rethrow;
    return run;
  }

  private async heartbeat(store: DurableStore, journal: RunJournal, active: () => number, signal: AbortSignal): Promise<void> {
    while (!signal.aborted) {
      try {
        await sleep(this.leaseTtl / 3, signal);
      } catch {
        return;
      }
      // Events of a long command (e.g. tool_start) reach the log without waiting for its result.
      journal.flushEvents();
      try {
        await store.heartbeat(journal.sessionId, journal.runId, journal.epoch, { ttl: this.leaseTtl, activeSeconds: active() });
      } catch {
        return; // The run's own next append reports the failure.
      }
    }
  }
}

/**
 * Recovers every interrupted run in a store, for example on process start.
 *
 * `context` gives each recovered run its live resources: an object shared by all
 * runs, or a function receiving the run's RunRecord.
 */
export class Worker {
  readonly agents: Map<string, Agent>;

  constructor(
    readonly store: DurableStore,
    agents: Iterable<Agent>,
    readonly options: {
      context?: Record<string, unknown> | ((record: RunRecord) => Record<string, unknown>);
      concurrency?: number;
    } = {},
  ) {
    const list = [...agents];
    this.agents = new Map(list.map((agent) => [agent.name, agent]));
    if (this.agents.size !== list.length) throw new Error("Agent names must be unique");
    if (list.some((agent) => agent.store !== store)) throw new Error("Every agent must use this worker's store");
    if (!((options.concurrency ?? 50) >= 1)) throw new Error("concurrency must be positive");
  }

  /**
   * Recover all running runs; results are in store order, failures as errors.
   *
   * On a single host, `force: true` at startup is safe: no other process can own
   * these runs, so there is no need to wait for their leases to expire.
   */
  async recoverAll(
    options: { force?: boolean; allowFingerprintChange?: boolean; includeFrozen?: boolean } = {},
  ): Promise<(Run | Error)[]> {
    const { includeFrozen = false, ...recoverOptions } = options;
    // A frozen session stays frozen until someone recovers it on purpose.
    const records = this.store
      .runs({ status: "running" })
      .filter((record) => includeFrozen || this.store.control(record.sessionId)?.state !== "freeze");
    const limit = this.options.concurrency ?? 50;
    const results: (Run | Error)[] = new Array(records.length);
    let next = 0;
    const worker = async () => {
      while (next < records.length) {
        const index = next++;
        const record = records[index];
        try {
          const agent = this.agents.get(record.agent);
          if (agent === undefined) throw new Error(`No agent named ${JSON.stringify(record.agent)} for run ${record.runId}`);
          const context = typeof this.options.context === "function" ? this.options.context(record) : this.options.context;
          results[index] = await agent.recover(record.sessionId, { context, ...recoverOptions });
        } catch (error) {
          results[index] = error instanceof Error ? error : new Error(String(error));
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(limit, records.length) }, worker));
    return results;
  }
}
