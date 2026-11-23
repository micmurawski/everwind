/** The common execution boundary used by every loop variation. */

import { AsyncLocalStorage } from "node:async_hooks";
import {
  type CompactionResult,
  type CompactionStrategy,
  MessageMarker,
  MessageRange,
  OlderHistory,
  type Selection,
  OlderMessages,
  type TokenCounter,
  isSummary,
} from "./contextOps.ts";
import { DeadlineExceeded, LimitReached, ReplayExhausted, SessionLogError, VersionUnsupported, isStopError } from "./errors.ts";
import { validateConversation } from "./history.ts";
import {
  EPHEMERAL_EVENTS,
  INTERRUPTED_CONTENT,
  type JournalEntry,
  type RunJournal,
  commandKey,
} from "./journal.ts";
import { type Provider, providerModel, providerName } from "./providers/base.ts";
import { collectResponse } from "./providers/streaming.ts";
import { SharedView, type SharedWorld, ToolShared } from "./shared.ts";
import type { Tools } from "./tools.ts";
import {
  type Event,
  type Limits,
  type Message,
  type ModelRequest,
  type ModelResponse,
  type Run,
  type StreamChunk,
  type ToolCall,
  message,
  modelResponse,
} from "./types.ts";
import { type JsonObject, abortable, childController, clone, hexId } from "./util.ts";

export type { Loop } from "./loops.ts";
export type { Provider } from "./providers/base.ts";

/** The tool call whose execution emits an event, so concurrent calls' events get distinct keys. */
const currentCall = new AsyncLocalStorage<string>();
/** Events that happen once per attempt (owner epoch), not once per position in the run. */
const ATTEMPT_EVENTS = new Set(["run_recovered", "fork_live", "command_retried"]);

function modelKey(request: ModelRequest): string {
  // runId is excluded so that a fork can match its parent's recorded prefix.
  const { runId: _, ...rest } = request;
  return commandKey("model", rest);
}

function toolKey(call: ToolCall): string {
  return commandKey("tool", { id: call.id, name: call.name, arguments: call.arguments });
}

export interface RuntimeOptions {
  journal?: RunJournal | null;
  /** Emitted once where a recovered run or a fork starts executing live (null: a fresh run). */
  liveEvent?: [string, JsonObject] | null;
  shared?: SharedWorld | null;
  /** Aborted when the run times out or is cancelled. */
  signal?: AbortSignal;
}

export class Runtime {
  readonly journal: RunJournal | null;
  readonly signal: AbortSignal;
  /** Replaying: upcoming commands were recorded earlier and are not repeated. */
  replaying: boolean;
  /** @internal Last command whose result is reflected in the run's state. */
  applied = 0;
  /** @internal Tool calls the last model turn requested, not yet executed. */
  pending: ToolCall[] = [];
  private seqCounter = 0;
  private liveEvent: [string, JsonObject] | null;
  private readonly world: SharedWorld | null;
  private sharedView: SharedView | null = null;
  private busy: string | null = null;
  private modelInflight = false;
  private readonly markerOwner = {};
  private contextVersion = 0;
  private compactionToken: object | null = null;
  private readonly eventOrdinals = new Map<string, number>();

  constructor(
    readonly run: Run,
    readonly provider: Provider,
    readonly tools: Tools,
    readonly limits: Limits,
    options: RuntimeOptions = {},
  ) {
    this.journal = options.journal ?? null;
    this.signal = options.signal ?? new AbortController().signal;
    this.replaying = this.journal !== null && this.journal.lastRecordedSeq > 0;
    this.liveEvent = options.liveEvent ?? null;
    this.world = options.shared ?? null;
    if (this.journal !== null && !this.journal.readonly) {
      // The journal is the session log: events are stored with its writes.
      run.eventSink = (event) => this.recordEvent(event);
    }
  }

  /** The agent's shared world; reads and writes are journaled commands of this run. */
  get shared(): SharedView {
    if (this.world === null) throw new Error("This agent has no shared world; pass new Agent({ shared: new SharedWorld(store) })");
    this.sharedView ??= new SharedView(this, this.world);
    return this.sharedView;
  }

  /** Number of journaled commands whose results are reflected in the run's state. */
  get seq(): number {
    return this.applied;
  }

  /** @internal */
  nextSeq(): number {
    this.seqCounter += 1;
    return this.seqCounter;
  }

  /** @internal Leave replay before the first command that was not recorded. */
  goLive(): void {
    if (this.journal?.readonly) throw new ReplayExhausted(this.seqCounter, "end of the recorded journal");
    this.replaying = false;
    if (this.liveEvent !== null && this.journal !== null) {
      const [kind, data] = this.liveEvent;
      this.liveEvent = null;
      this.run.emit(kind, { ...data, epoch: this.journal.epoch, recordedCommands: this.journal.lastRecordedSeq });
    }
  }

  /**
   * Buffer an event for the journal under a key that a replay reproduces.
   *
   * The key is the event's position in the run: the command number at which it
   * was emitted, its kind, the tool call it belongs to and an ordinal. A recovered
   * run re-emits its replayed events with the same keys, and the store keeps the
   * first copy, so the log has each event once, including ones a crash lost
   * before they were written.
   */
  private recordEvent(event: Event): void {
    if (EPHEMERAL_EVENTS.has(event.kind) || this.journal === null) return;
    const data = event.data as Record<string, JsonObject>;
    let owner: string;
    if (event.kind === "tool_start") owner = String(data.call.id);
    else if (event.kind === "tool_result") owner = String(data.message.toolCallId);
    else owner = currentCall.getStore() ?? "";
    if (ATTEMPT_EVENTS.has(event.kind)) owner = `${owner}@${this.journal.epoch}`;
    const base = `${this.seqCounter}:${event.kind}:${owner}`;
    const ordinal = this.eventOrdinals.get(base) ?? 0;
    this.eventOrdinals.set(base, ordinal + 1);
    this.journal.recordEvent(`${base}:${ordinal}`, event);
  }

  private sideEffect<T>(name: string, produce: () => T): T {
    this.ensureActive();
    const seq = this.nextSeq();
    if (this.journal === null) {
      this.applied = seq;
      return produce();
    }
    const key = commandKey("side_effect", name);
    const [, completed] = this.journal.check(seq, "side_effect", key);
    if (completed !== null) {
      this.applied = seq;
      return completed.payload.value as T;
    }
    this.goLive();
    const value = produce();
    // Queued without waiting: ordered writes make it durable before any later effect.
    this.journal.enqueue({ seq, phase: "completed", kind: "side_effect", key, payload: { value: value as JsonObject } });
    this.applied = seq;
    return value;
  }

  /** Current UTC time, recorded: a replay returns the time of the original run. */
  now(): Date {
    return new Date(this.sideEffect("now", () => new Date().toISOString()));
  }

  /** A random hex UUID, recorded so that replays return the same one. */
  uuid(): string {
    return this.sideEffect("uuid", hexId);
  }

  /** A float in [0, 1), recorded so that replays return the same one. */
  random(): number {
    return this.sideEffect("random", Math.random);
  }

  /**
   * Choose a code path for a change so that old journals still replay.
   *
   * A new execution records and returns `maxVersion`. A replay returns the
   * recorded version. A replay of history from before the change existed returns
   * 0, the code as it was before any version() call. Raise `minSupported` once
   * journals with older versions no longer need to replay; they then fail with
   * VersionUnsupported and stay recoverable. Markers use no command number, so
   * adding a version() call never shifts later commands.
   */
  version(changeId: string, minSupported = 0, maxVersion = 1): number {
    this.ensureActive();
    if (typeof changeId !== "string" || !changeId) throw new Error("changeId must be a nonempty string");
    if (!(minSupported >= 0 && minSupported <= maxVersion)) throw new Error("Versions need 0 <= minSupported <= maxVersion");
    if (this.journal === null) return maxVersion;
    let recorded = this.journal.markers.get(changeId);
    if (recorded === undefined) {
      // Recorded commands still ahead: this point was executed by code without the change.
      recorded = this.seqCounter < this.journal.lastRecordedSeq ? 0 : maxVersion;
      if (!(minSupported <= recorded && recorded <= maxVersion)) {
        throw new VersionUnsupported(changeId, recorded, minSupported, maxVersion);
      }
      this.journal.marker(changeId, recorded, this.seqCounter);
    } else if (!(minSupported <= recorded && recorded <= maxVersion)) {
      throw new VersionUnsupported(changeId, recorded, minSupported, maxVersion);
    }
    return recorded;
  }

  /** @internal */
  ensureActive(): void {
    if (this.signal.aborted) throw this.signal.reason;
    if (this.run.status !== "running") throw new Error("Run has already finished");
  }

  private ensureIdle(): void {
    this.ensureActive();
    if (this.busy !== null) throw new Error(`Conversation is busy with ${this.busy}`);
  }

  private ensureContextReady(): void {
    this.ensureIdle();
    if (this.pending.length) throw new Error("Resolve pending tool calls before changing conversation context");
  }

  step(name: string, options: { metadata?: Record<string, unknown> } = {}): void {
    this.ensureActive();
    if (this.run.steps >= this.limits.maxSteps) throw new LimitReached("Graph step limit reached");
    this.run.steps += 1;
    this.run.emit("step", { node: name, step: this.run.steps, metadata: options.metadata ?? {} });
  }

  say(content: string): void {
    this.ensureIdle();
    if (this.pending.length) throw new Error("Resolve pending tool calls before adding an instruction");
    this.run.messages.push(message({ role: "user", content }));
    this.run.emit("instruction", { content });
  }

  async complete(options: { allowTools?: boolean; purpose?: string } = {}): Promise<ModelResponse> {
    this.ensureContextReady();
    this.busy = "a model turn";
    try {
      const response = await this.request([...this.run.messages], {
        allowTools: options.allowTools ?? true,
        purpose: options.purpose ?? "act",
        conversation: true,
      });
      this.pending = clone(response.toolCalls);
      this.run.messages.push(
        message({
          role: "assistant",
          content: response.text,
          toolCalls: response.toolCalls,
          metadata: { ...response.metadata, ...(response.reasoning ? { reasoning: response.reasoning } : {}) },
        }),
      );
      return response;
    } finally {
      this.busy = null;
    }
  }

  /** Account for a model call without implicitly editing the conversation. */
  private async request(
    messages: Message[],
    options: { allowTools: boolean; purpose: string; conversation: boolean; provider?: Provider },
  ): Promise<ModelResponse> {
    this.ensureActive();
    if (this.modelInflight) throw new Error("Concurrent model requests on one runtime are not supported");
    if (this.run.modelCalls >= this.limits.maxModelCalls) throw new LimitReached("Model call limit reached");
    this.run.modelCalls += 1;
    const request: ModelRequest = {
      runId: this.run.id,
      messages: clone(messages),
      tools: options.allowTools ? this.tools.specs : [],
      turn: this.run.modelCalls,
      purpose: options.purpose,
    };
    const seq = this.nextSeq();
    const provider = options.provider ?? this.provider;
    this.run.emit("model_request", {
      request,
      conversation: options.conversation,
      provider: providerName(provider),
      model: providerModel(provider),
    });
    let key = "";
    if (this.journal !== null) {
      key = modelKey(request);
      const [scheduled, completed] = this.journal.check(seq, "model", key);
      if (completed !== null) {
        const response = modelResponse(completed.payload.response as never);
        for (const [name, value] of Object.entries(response.usage)) this.run.usage[name] = (this.run.usage[name] ?? 0) + value;
        this.run.emit("model_response", {
          response,
          turn: request.turn,
          purpose: options.purpose,
          conversation: options.conversation,
        });
        this.applied = seq;
        return response;
      }
      this.goLive();
      if (scheduled === null) {
        await this.journal.append({ seq, phase: "scheduled", kind: "model", key, payload: { purpose: options.purpose, turn: request.turn } });
      } else {
        this.run.emit("command_retried", { seq, command: "model" });
      }
    }
    const onChunk = (chunk: StreamChunk) => {
      // Usage already received remains charged if a later chunk fails or the call is cancelled.
      if (chunk.type === "usage") {
        for (const [name, value] of Object.entries(chunk)) {
          if (name !== "type") this.run.usage[name] = (this.run.usage[name] ?? 0) + (value as number);
        }
      }
      this.run.emit("model_chunk", { turn: request.turn, chunk });
    };
    this.modelInflight = true;
    let response: ModelResponse;
    try {
      response = await collectResponse(provider, request, { onChunk, signal: this.signal });
    } finally {
      this.modelInflight = false;
    }
    this.ensureActive();
    if (response.toolCalls.length && !options.allowTools) throw new Error("Provider returned tool calls when tools were disabled");
    const ids = response.toolCalls.map((call) => call.id);
    if (new Set(ids).size !== ids.length) throw new Error("Provider returned duplicate tool call IDs");
    if (this.journal !== null) {
      await this.journal.append({ seq, phase: "completed", kind: "model", key, payload: { response: response as unknown as JsonObject } });
    }
    this.run.emit("model_response", {
      response,
      turn: request.turn,
      purpose: options.purpose,
      conversation: options.conversation,
    });
    this.applied = seq;
    return response;
  }

  async executeTools(options: { concurrency?: number } = {}): Promise<void> {
    this.ensureIdle();
    this.busy = "tool execution";
    try {
      await this.runTools(options.concurrency ?? 1);
    } finally {
      this.busy = null;
    }
  }

  private async runTools(concurrency: number): Promise<void> {
    this.ensureActive();
    if (!Number.isInteger(concurrency) || concurrency < 1) throw new Error("Tool concurrency must be positive");
    const calls = this.pending;
    if (!calls.length) throw new Error("No pending tool calls to execute");
    if (this.run.toolCalls + calls.length > this.limits.maxToolCalls) {
      throw new LimitReached("Tool call limit reached; batch was not executed");
    }
    this.run.toolCalls += calls.length;
    // Sequence numbers follow the provider's order, never completion order.
    const seqs = calls.map(() => this.nextSeq());
    const keys = calls.map(toolKey);
    const known = new Map<number, Message>();
    if (this.journal !== null) {
      const newRows: JournalEntry[] = [];
      const interrupted: number[] = [];
      const retried: number[] = [];
      calls.forEach((call, index) => {
        const [scheduled, completed] = (this.journal as RunJournal).check(seqs[index], "tool", keys[index]);
        if (completed !== null) known.set(index, message(completed.payload.message as never));
        else if (scheduled === null) {
          newRows.push({
            seq: seqs[index],
            phase: "scheduled",
            kind: "tool",
            key: keys[index],
            payload: { callId: call.id, name: call.name, effect: this.tools.effect(call.name) },
          });
        } else if (this.tools.effect(call.name) === "unsafe") {
          // Started before a crash with no recorded result: do not repeat the effect.
          interrupted.push(index);
        } else {
          retried.push(index);
        }
      });
      for (const index of [...known.keys()].sort((a, b) => a - b)) {
        this.run.emit("tool_start", { call: calls[index] });
        this.run.emit("tool_result", { message: known.get(index) });
      }
      if (known.size < calls.length) this.goLive();
      if (newRows.length) await this.journal.append(...newRows);
      for (const index of retried) {
        const call = calls[index];
        this.run.emit("command_retried", { seq: seqs[index], command: "tool", callId: call.id, effect: this.tools.effect(call.name) });
      }
      for (const index of interrupted) {
        // The crashed attempt started it; its tool_start may not have reached the log.
        this.run.emit("tool_start", { call: calls[index] });
        const result = message({ role: "tool", toolCallId: calls[index].id, isError: true, content: INTERRUPTED_CONTENT });
        await this.journal.append({
          seq: seqs[index],
          phase: "completed",
          kind: "tool",
          key: keys[index],
          payload: { message: result as unknown as JsonObject, synthetic: true },
        });
        this.run.emit("tool_result", { message: result, synthetic: true });
        known.set(index, result);
      }
    }

    const batch = childController(this.signal);
    const execute = (index: number): Promise<Message> => currentCall.run(calls[index].id, () => this.executeCall(calls[index], seqs[index], keys[index], batch.signal));

    if (concurrency === 1) {
      for (let index = 0; index < calls.length; index++) {
        this.run.messages.push(known.get(index) ?? (await execute(index)));
      }
    } else {
      // Like a task group: the first failure aborts the siblings; results keep the provider's order.
      const results = new Map<number, Message>();
      const queue = calls.map((_, index) => index).filter((index) => !known.has(index));
      let failure: { error: unknown } | null = null;
      const worker = async () => {
        while (queue.length && failure === null) {
          const index = queue.shift() as number;
          try {
            results.set(index, await abortable(execute(index), batch.signal));
          } catch (error) {
            if (failure === null) {
              failure = { error };
              batch.abort(error);
            }
          }
        }
      };
      await Promise.all(Array.from({ length: Math.min(concurrency, queue.length) }, worker));
      if (failure !== null) throw (failure as { error: unknown }).error;
      for (let index = 0; index < calls.length; index++) {
        this.run.messages.push(known.get(index) ?? (results.get(index) as Message));
      }
    }
    this.pending = [];
    this.applied = seqs[seqs.length - 1];
  }

  private async executeCall(call: ToolCall, seq: number, key: string, parent: AbortSignal): Promise<Message> {
    this.run.emit("tool_start", { call });
    const idempotencyKey = `${this.run.sessionId}:${this.run.id}:${seq}:${call.id}`;
    const controller = childController(parent);
    // A timed-out effect has an unknown outcome: stop the run, do not retry it.
    const timer = setTimeout(
      () => controller.abort(new DeadlineExceeded(`Tool ${call.name} exceeded its ${this.limits.toolTimeout}s deadline`)),
      this.limits.toolTimeout * 1000,
    );
    let result;
    try {
      result = await abortable(
        this.tools.invoke(call, {
          values: this.run.context,
          runtime: this,
          idempotencyKey,
          shared:
            this.world === null
              ? undefined
              : new ToolShared(this.world, idempotencyKey, { session: this.run.sessionId, run: this.run.id, seq, callId: call.id }),
          signal: controller.signal,
        }),
        controller.signal,
      );
    } finally {
      clearTimeout(timer);
    }
    this.ensureActive();
    const output = message({ role: "tool", toolCallId: call.id, content: result.content(), isError: result.error !== null });
    if (this.journal !== null) {
      await this.journal.append({ seq, phase: "completed", kind: "tool", key, payload: { message: output as unknown as JsonObject } });
    }
    this.run.emit("tool_result", { message: output });
    return output;
  }

  finish(output: string): void {
    this.ensureIdle();
    if (this.pending.length) throw new Error("Resolve pending tool calls before finishing");
    this.run.output = output;
    this.run.status = "completed";
  }

  /**
   * Mark a complete conversation boundary for a later compaction.
   *
   * Markers belong to this invocation. Any applied compaction invalidates
   * existing markers; appending messages does not invalidate them.
   */
  markMessages(): MessageMarker {
    this.ensureContextReady();
    validateConversation(this.run.messages);
    return new MessageMarker(this.markerOwner, this.contextVersion, clone(this.run.messages));
  }

  private markerIndex(marker: MessageMarker): number {
    if (!(marker instanceof MessageMarker) || marker.owner !== this.markerOwner) {
      throw new Error("Message marker belongs to another runtime");
    }
    if (marker.version !== this.contextVersion) throw new Error("Stale message marker: conversation was compacted");
    const index = marker.prefix.length;
    if (JSON.stringify(this.run.messages.slice(0, index)) !== JSON.stringify(marker.prefix)) {
      throw new Error("Stale message marker: earlier messages changed");
    }
    return index;
  }

  private selectMessages(selection: Selection): [number, number] {
    const messages = this.run.messages;
    let start: number;
    let stop: number;
    if (selection instanceof MessageRange) {
      start = this.markerIndex(selection.start);
      stop = selection.stop === null ? messages.length : this.markerIndex(selection.stop);
    } else if (selection instanceof OlderHistory) {
      const users = messages.flatMap((item, index) => (item.role === "user" ? [index] : []));
      if (users.length <= selection.keepRecentTurns) return [messages.length, messages.length];
      start = users[0] + 1;
      stop = users[users.length - selection.keepRecentTurns];
    } else if (selection instanceof OlderMessages) {
      const first = messages.length && messages[0].role === "system" ? 1 : 0;
      start = Math.min(first + 1, messages.length);
      stop = Math.max(start, messages.length - selection.keepRecentMessages);
      // A tool-result tail must retain the assistant and every sibling result,
      // including parallel calls returned in a different order.
      while (stop > start && messages[stop].role === "tool") stop -= 1;
    } else {
      throw new TypeError("selection must be a MessageRange, OlderHistory or OlderMessages");
    }
    if (start > stop) throw new Error("Compaction range starts after its stop marker");
    if (messages.slice(start, stop).some((item) => item.role === "system")) throw new Error("Compaction cannot replace system messages");
    // Both boundaries must close every tool call, including parallel calls.
    validateConversation(messages.slice(0, start));
    validateConversation(messages.slice(0, stop));
    return [start, stop];
  }

  compactSince(marker: MessageMarker, options: { strategy: CompactionStrategy; tokenCounter?: TokenCounter }): Promise<CompactionResult> {
    return this.compact({ selection: new MessageRange(marker), ...options });
  }

  /**
   * Prepare, validate and atomically replace a conversation region.
   *
   * With an explicit tokenCounter, apply only a strict reduction in the full
   * conversation's token count. Failures propagate without a fallback. Events
   * and model usage are retained even when a proposal is not applied.
   */
  async compact(options: { selection: Selection; strategy: CompactionStrategy; tokenCounter?: TokenCounter }): Promise<CompactionResult> {
    const { selection, strategy, tokenCounter } = options;
    this.ensureContextReady();
    validateConversation(this.run.messages);
    const [start, stop] = this.selectMessages(selection);
    const name = strategy.constructor.name;
    const snapshot = clone(this.run.messages);
    const source = snapshot.slice(start, stop);
    let skip: string | null = source.length ? null : "empty_selection";
    if (selection instanceof OlderMessages) {
      if (snapshot.slice(stop).some(isSummary)) skip = "recent_summary";
      else if (source.length < 2 && !source.some(isSummary)) skip = "not_enough_messages";
    }
    const result = (applied: boolean, reason: string, extra: Partial<CompactionResult> = {}): CompactionResult => ({
      applied,
      reason,
      strategy: name,
      start,
      stop,
      replacement: [],
      beforeTokens: null,
      afterTokens: null,
      ...extra,
    });
    if (skip !== null) {
      this.run.emit("context_compaction_skipped", { strategy: name, start, stop, reason: skip });
      return result(false, skip);
    }

    this.busy = "compaction";
    const token = {};
    this.compactionToken = token;
    const generateSummary = async (prompt: string, summary: { provider?: Provider; startIndex?: number } = {}) => {
      const startIndex = summary.startIndex ?? 0;
      if (this.compactionToken !== token) throw new Error("Summary generator is only valid during its compaction");
      if (typeof prompt !== "string" || !prompt.trim()) throw new Error("Summarization prompt must be nonempty");
      if (!Number.isInteger(startIndex) || !(startIndex >= 0 && startIndex < source.length)) {
        throw new Error("Summary startIndex must select a nonempty source suffix");
      }
      // Encode tool calls and results as transcript data rather than native
      // protocol messages, so a different summarizer provider can read them.
      const transcript = JSON.stringify(source.slice(startIndex));
      return this.request(
        [message({ role: "system", content: prompt }), message({ role: "user", content: `Conversation region:\n${transcript}` })],
        { allowTools: false, purpose: "compact", conversation: false, provider: summary.provider },
      );
    };
    const count = async (messages: Message[]): Promise<number | null> => {
      if (tokenCounter === undefined) return null;
      const value = await tokenCounter(clone(messages));
      if (!Number.isInteger(value) || value < 0) throw new Error("Token counter must return a nonnegative integer");
      return value;
    };

    try {
      const beforeTokens = await count(snapshot);
      const proposal = await strategy.compact({ messages: clone(source), generateSummary });
      if (proposal === null || typeof proposal !== "object" || !Array.isArray(proposal.messages)) {
        throw new TypeError("Compaction strategy must return a CompactionProposal");
      }
      const replacement = proposal.messages.map((item) => message(item));
      if (replacement.some((item) => item.role === "system")) throw new Error("Compaction cannot insert system messages");
      const candidate = [...snapshot.slice(0, start), ...replacement, ...snapshot.slice(stop)];
      validateConversation(candidate);
      const afterTokens = await count(candidate);
      this.ensureActive();
      if (this.pending.length || this.modelInflight || JSON.stringify(this.run.messages) !== JSON.stringify(snapshot)) {
        throw new Error("Conversation changed during compaction; proposal was not applied");
      }
      let reason = "applied";
      if (JSON.stringify(replacement) === JSON.stringify(source)) reason = "unchanged";
      else if (beforeTokens !== null && (afterTokens as number) >= beforeTokens) reason = "not_smaller";
      const outcome = result(reason === "applied", reason, { replacement: clone(replacement), beforeTokens, afterTokens });
      if (!outcome.applied) {
        this.run.emit("context_compaction_skipped", { strategy: name, start, stop, reason, beforeTokens, afterTokens });
        return outcome;
      }
      // A requested log failure must not mutate the live conversation.
      this.run.emit("context_compaction", {
        strategy: name,
        start,
        stop,
        removedMessages: source,
        replacementMessages: replacement,
        beforeTokens,
        afterTokens,
      });
      this.run.messages.splice(0, this.run.messages.length, ...candidate);
      this.contextVersion += 1;
      return outcome;
    } catch (error) {
      if (!(error instanceof SessionLogError) && !isStopError(error)) {
        this.run.emit("context_compaction_failed", {
          strategy: name,
          start,
          stop,
          error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
        });
      }
      throw error;
    } finally {
      this.busy = null;
      this.compactionToken = null;
    }
  }
}
