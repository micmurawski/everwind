/** Provider-independent messages and serializable experiment records. */

import { z } from "zod";
import { SessionLogError } from "./errors.ts";
import { eventLine } from "./sessionLog.ts";
import { type Json, type JsonObject, clone, hexId } from "./util.ts";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const json: z.ZodType<Json> = z.lazy(() =>
  z.union([z.null(), z.boolean(), z.number(), z.string(), z.array(json), z.record(z.string(), json)]),
);
const jsonObject = z.record(z.string(), json);

export const ToolCallSchema = z.strictObject({
  id: z.string().min(1),
  name: z.string().min(1),
  arguments: jsonObject.default({}),
  metadata: jsonObject.default({}),
});
export type ToolCall = z.output<typeof ToolCallSchema>;
export type ToolCallInput = z.input<typeof ToolCallSchema>;

export const MessageSchema = z
  .strictObject({
    role: z.enum(["system", "user", "assistant", "tool"]),
    content: z.string().default(""),
    toolCalls: z.array(ToolCallSchema).default([]),
    toolCallId: z.string().nullable().default(null),
    isError: z.boolean().default(false),
    metadata: jsonObject.default({}),
  })
  .superRefine((message, context) => {
    if (message.toolCalls.length && message.role !== "assistant") {
      context.addIssue({ code: "custom", message: "Only assistant messages may contain tool calls" });
    }
    if ((message.role === "tool") !== Boolean(message.toolCallId)) {
      context.addIssue({ code: "custom", message: "A tool message requires toolCallId; other roles must omit it" });
    }
  });
export type Message = z.output<typeof MessageSchema>;
export type MessageInput = z.input<typeof MessageSchema>;

/** Validate and complete a message; the result is a private copy. */
export function message(input: MessageInput): Message {
  return MessageSchema.parse(clone(input));
}

export function toolCall(input: ToolCallInput): ToolCall {
  return ToolCallSchema.parse(clone(input));
}

export interface ToolSpec {
  readonly name: string;
  readonly description: string;
  readonly parameters: Record<string, unknown>;
}

export interface CreateMessageMetadata {
  runId?: string;
  taskId?: string;
  turn?: number;
  purpose?: string;
}

export type TextChunk = { type: "text"; text: string };
export type ReasoningChunk = { type: "reasoning"; text: string };
export type ToolUseChunk = { type: "tool_use"; id: string; name: string; input: JsonObject; metadata?: JsonObject };
export type UsageChunk = {
  type: "usage";
  inputTokens?: number;
  outputTokens?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  reasoningTokens?: number;
  totalCost?: number;
};
export type MetadataChunk = { type: "metadata"; metadata: JsonObject };
export type GroundingChunk = { type: "grounding"; sources: Json[] };
export type StreamChunk = TextChunk | ReasoningChunk | ToolUseChunk | UsageChunk | MetadataChunk | GroundingChunk;

export interface ModelRequest {
  readonly runId: string;
  readonly messages: readonly Message[];
  readonly tools: readonly ToolSpec[];
  readonly turn: number;
  readonly purpose: string;
}

export const ModelResponseSchema = z.strictObject({
  text: z.string().default(""),
  reasoning: z.string().default(""),
  toolCalls: z.array(ToolCallSchema).default([]),
  usage: z.record(z.string(), z.number()).default({}),
  metadata: jsonObject.default({}),
});
export type ModelResponse = z.output<typeof ModelResponseSchema>;
export type ModelResponseInput = z.input<typeof ModelResponseSchema>;

export function modelResponse(input: ModelResponseInput = {}): ModelResponse {
  return ModelResponseSchema.parse(clone(input));
}

const positive = z.number().positive();
export const LimitsSchema = z.strictObject({
  maxModelCalls: z.number().int().positive().default(20),
  maxToolCalls: z.number().int().nonnegative().default(100),
  maxSteps: z.number().int().positive().default(100),
  /** Seconds of active execution for the whole run. */
  timeout: positive.default(300),
  /** Seconds for one tool call. */
  toolTimeout: positive.default(60),
});
export type Limits = z.output<typeof LimitsSchema>;
export type LimitsInput = z.input<typeof LimitsSchema>;

export function limits(input: LimitsInput = {}): Limits {
  return LimitsSchema.parse(input);
}

export interface Event {
  readonly runId: string;
  readonly sessionId: string | null;
  readonly sequence: number;
  readonly kind: string;
  readonly data: JsonObject;
  /** ISO 8601, UTC. */
  readonly timestamp: string;
}

/** "frozen": an operator froze it; its stored record stays running, and recover() continues it. */
export type RunStatus = "running" | "completed" | "limit" | "timeout" | "failed" | "cancelled" | "frozen";

export interface RunInit {
  id?: string;
  sessionId?: string;
  parentSessionId?: string | null;
  parentRunId?: string | null;
  jobId?: string | null;
  messages?: Message[];
  context?: Record<string, unknown>;
  scope?: JsonObject;
}

/** Mutable state owned by exactly one execution; context is never logged. */
export class Run {
  readonly id: string;
  readonly sessionId: string;
  readonly parentSessionId: string | null;
  readonly parentRunId: string | null;
  readonly jobId: string | null;
  messages: Message[];
  readonly context: Record<string, unknown>;
  /** What the run is for, as plain JSON: recorded, so recovery binds the same tools. */
  readonly scope: JsonObject;
  state: Record<string, unknown> = {};
  readonly events: Event[] = [];
  usage: Record<string, number> = {};
  modelCalls = 0;
  toolCalls = 0;
  steps = 0;
  status: RunStatus = "running";
  output: string | null = null;
  error: string | null = null;
  logPath: string | null = null;
  /** @internal Receives each event as it is emitted (a session log, or the journal). */
  eventSink: ((event: Event) => void) | null = null;

  constructor(init: RunInit = {}) {
    this.id = init.id ?? hexId();
    this.sessionId = init.sessionId ?? hexId();
    this.parentSessionId = init.parentSessionId ?? null;
    this.parentRunId = init.parentRunId ?? null;
    this.jobId = init.jobId ?? null;
    this.messages = init.messages ?? [];
    this.context = init.context ?? {};
    this.scope = clone(init.scope ?? {});
  }

  emit(kind: string, data: Record<string, unknown> = {}): void {
    const event: Event = {
      runId: this.id,
      sessionId: this.sessionId,
      sequence: this.events.length,
      kind,
      data: JSON.parse(JSON.stringify(data)),
      timestamp: new Date().toISOString(),
    };
    this.events.push(event);
    if (this.eventSink !== null) {
      try {
        this.eventSink(clone(event));
      } catch (error) {
        if (error instanceof SessionLogError) this.eventSink = null;
        throw error;
      }
    }
  }

  /** Write a run's events as JSONL (overwrites the given file). */
  writeEvents(path: string): void {
    const target = resolve(path);
    if (this.logPath !== null && target === this.logPath) {
      throw new Error("Cannot overwrite the session log with a single-run export");
    }
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, this.events.map(eventLine).join(""), "utf-8");
  }
}
