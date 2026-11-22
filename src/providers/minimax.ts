/**
 * MiniMax streaming: https://platform.minimax.io/docs/api-reference/text-anthropic-api
 *
 * Full assistant content is retained for tool turns, including thinking and signatures.
 * The `@anthropic-ai/sdk` package is loaded only when creating a client.
 */

import type { CreateMessageMetadata, Message, StreamChunk, ToolSpec } from "../types.ts";
import { toolCall } from "../types.ts";
import type { JsonObject } from "../util.ts";
import { ApiHandler, type CreateMessageOptions } from "./base.ts";
import { closeStream } from "./openai.ts";

export const BASE_URL = "https://api.minimax.io/anthropic";
export const CONTENT_KEY = "minimaxContent";
const RESERVED = ["model", "messages", "system", "stream", "tools"];

type Block = Record<string, unknown>;

function formatMessages(messages: Message[]): { role: string; content: Block[] }[] {
  const formatted: { role: string; content: Block[] }[] = [];
  for (const message of messages) {
    if (message.role === "system") throw new Error("Pass system messages through systemPrompt");
    const role = message.role === "tool" ? "user" : message.role;
    let content: Block[];
    if (message.role === "tool") {
      content = [{ type: "tool_result", tool_use_id: message.toolCallId, content: message.content, is_error: message.isError }];
    } else if (message.role === "assistant" && CONTENT_KEY in message.metadata) {
      const stored = structuredClone(message.metadata[CONTENT_KEY]);
      if (!Array.isArray(stored) || !stored.every((block) => block !== null && typeof block === "object" && !Array.isArray(block))) {
        throw new TypeError(`${CONTENT_KEY} must be a list of content blocks`);
      }
      content = stored as Block[];
    } else {
      content = message.content ? [{ type: "text", text: message.content }] : [];
      for (const call of message.toolCalls) content.push({ type: "tool_use", id: call.id, name: call.name, input: call.arguments });
    }
    const last = formatted[formatted.length - 1];
    if (last && role === "user" && last.role === "user") last.content.push(...content);
    else formatted.push({ role, content });
  }
  return formatted;
}

function withoutNulls(block: Block): JsonObject {
  return Object.fromEntries(Object.entries(block).filter(([, value]) => value !== null && value !== undefined)) as JsonObject;
}

type MessagesClient = { messages: { stream(request: Record<string, unknown>): AsyncIterable<any> & { finalMessage(): Promise<any> } } };

export interface MiniMaxOptions {
  apiKey?: string;
  baseUrl?: string;
  maxTokens?: number;
  client?: MessagesClient;
  /** Creates the owned client; defaults to the Anthropic SDK. */
  clientFactory?: (options: { apiKey: string; baseURL: string; maxRetries: number }) => MessagesClient;
  [option: string]: unknown;
}

export class MiniMax extends ApiHandler {
  readonly options: Record<string, unknown>;
  private client: MessagesClient | null;
  private readonly ownsClient: boolean;
  private readonly createClient: (() => Promise<MessagesClient>) | null = null;

  constructor(
    readonly model: string,
    { apiKey, baseUrl = BASE_URL, maxTokens = 4096, client, clientFactory, ...options }: MiniMaxOptions = {},
  ) {
    super();
    const reserved = RESERVED.filter((key) => key in options);
    if (reserved.length) throw new Error(`Reserved model options: ${reserved.sort().join(", ")}`);
    if (!Number.isInteger(maxTokens) || maxTokens <= 0) throw new Error("maxTokens must be a positive integer");
    this.options = { max_tokens: maxTokens, ...options };
    this.client = client ?? null;
    this.ownsClient = client === undefined;
    if (client === undefined) {
      const key = apiKey ?? process.env.MINIMAX_API_KEY;
      if (!key) throw new Error("Pass apiKey or set MINIMAX_API_KEY");
      const settings = { apiKey: key, baseURL: baseUrl, maxRetries: 0 };
      if (clientFactory) this.client = clientFactory(settings);
      else {
        this.createClient = async () => {
          try {
            const { default: Anthropic } = await import("@anthropic-ai/sdk" as string);
            return new Anthropic(settings) as MessagesClient;
          } catch (error) {
            throw new Error("Install @anthropic-ai/sdk to use MiniMax", { cause: error });
          }
        };
      }
    }
  }

  async *createMessage(
    systemPrompt: string,
    messages: Message[],
    _metadata?: CreateMessageMetadata,
    { tools: specs = [], signal: _signal, ...kwargs }: CreateMessageOptions = {},
  ): AsyncGenerator<StreamChunk> {
    const options: Record<string, unknown> = { ...this.options, ...kwargs };
    if (RESERVED.some((key) => key in options)) throw new Error("model, messages, system and stream are controlled by the adapter");
    if (specs.length) {
      options.tools = specs.map((spec: ToolSpec) => ({ name: spec.name, description: spec.description, input_schema: spec.parameters }));
    } else {
      delete options.tool_choice;
    }
    if (this.client === null && this.createClient !== null) this.client = await this.createClient();
    const client = this.client as MessagesClient;
    // Runtime metadata is local bookkeeping, not an Anthropic metadata payload.
    const stream = client.messages.stream({ model: this.model, system: systemPrompt, messages: formatMessages(messages), ...options });
    try {
      let stopped = false;
      const toolJson = new Map<number, string>();
      for await (const event of stream) {
        if (event.type === "content_block_start") {
          const block = event.content_block;
          if (block.type === "text" && block.text) yield { type: "text", text: block.text };
          else if (block.type === "thinking" && block.thinking) yield { type: "reasoning", text: block.thinking };
        } else if (event.type === "content_block_delta") {
          const delta = event.delta;
          if (delta.type === "text_delta") yield { type: "text", text: delta.text };
          else if (delta.type === "thinking_delta") yield { type: "reasoning", text: delta.thinking };
          else if (delta.type === "input_json_delta") toolJson.set(event.index, (toolJson.get(event.index) ?? "") + delta.partial_json);
        } else if (event.type === "message_stop") {
          stopped = true;
        } else if (event.type === "error") {
          throw new Error(`MiniMax stream error: ${JSON.stringify(event.error)}`);
        }
        // Other SDK convenience events are ignored, to avoid duplicating raw deltas.
      }
      if (!stopped) throw new Error("Incomplete MiniMax stream: missing message_stop");
      const response = await stream.finalMessage();
      if (!["end_turn", "tool_use", "stop_sequence"].includes(response.stop_reason)) {
        throw new Error(`Incomplete MiniMax response: ${response.stop_reason}`);
      }
      const content = (response.content as Block[]).map(withoutNulls);
      for (const [index, block] of content.entries()) {
        if (block.type !== "tool_use") continue;
        const args = toolJson.has(index) ? JSON.parse(toolJson.get(index) as string) : block.input;
        if (args === null || typeof args !== "object" || Array.isArray(args)) throw new TypeError("Tool arguments must be a JSON object");
        const call = toolCall({ id: block.id as string, name: block.name as string, arguments: args });
        yield { type: "tool_use", id: call.id, name: call.name, input: call.arguments };
      }
      const usage: Record<string, number> = {};
      for (const [source, target] of [
        ["input_tokens", "inputTokens"],
        ["output_tokens", "outputTokens"],
        ["cache_read_input_tokens", "cacheReadTokens"],
        ["cache_creation_input_tokens", "cacheWriteTokens"],
      ] as const) {
        const value = response.usage?.[source];
        if (typeof value === "number") usage[target] = value;
      }
      yield { type: "usage", ...usage };
      yield { type: "metadata", metadata: { model: response.model, stopReason: response.stop_reason, [CONTENT_KEY]: content } };
    } finally {
      await closeStream(stream);
    }
  }

  async close(): Promise<void> {
    if (this.ownsClient && this.client !== null) await (this.client as { close?: () => Promise<void> }).close?.();
  }
}
