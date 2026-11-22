/** OpenAI-compatible streaming chat; the `openai` SDK is an optional peer dependency. */

import type { CreateMessageMetadata, Message, StreamChunk, ToolSpec } from "../types.ts";
import { toolCall } from "../types.ts";
import { ApiHandler, type CreateMessageOptions } from "./base.ts";

const RESERVED = ["model", "messages", "tools", "stream", "n"];

/** Release a provider SDK stream, whichever closing method it offers. */
export async function closeStream(stream: unknown): Promise<void> {
  const target = stream as {
    close?: () => unknown;
    abort?: () => unknown;
    controller?: { abort?: () => unknown };
    return?: () => unknown;
  };
  if (typeof target?.close === "function") await target.close();
  else if (typeof target?.abort === "function") await target.abort();
  else if (typeof target?.controller?.abort === "function") target.controller.abort();
  else if (typeof target?.return === "function") await target.return();
}

type ChatClient = { chat: { completions: { create(request: Record<string, unknown>): Promise<AsyncIterable<any>> } } };

export interface OpenAIChatOptions {
  client?: ChatClient;
  apiKey?: string;
  baseUrl?: string;
  /** Request options, e.g. temperature, sent with every request. */
  [option: string]: unknown;
}

export class OpenAIChat extends ApiHandler {
  readonly options: Record<string, unknown>;
  private client: ChatClient | null;
  private readonly ownsClient: boolean;
  private readonly apiKey: string | undefined;
  private readonly baseUrl: string | undefined;

  constructor(
    readonly model: string,
    { client, apiKey, baseUrl, ...options }: OpenAIChatOptions = {},
  ) {
    super();
    const reserved = RESERVED.filter((key) => key in options);
    if (reserved.length) throw new Error(`Reserved model options: ${reserved.sort().join(", ")}`);
    this.options = options;
    this.client = client ?? null;
    this.ownsClient = client === undefined;
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  private async getClient(): Promise<ChatClient> {
    if (this.client === null) {
      let OpenAI: new (options: Record<string, unknown>) => ChatClient;
      try {
        OpenAI = (await import("openai" as string)).default;
      } catch (error) {
        throw new Error("Install the openai package to use OpenAIChat", { cause: error });
      }
      this.client = new OpenAI({ apiKey: this.apiKey, baseURL: this.baseUrl, maxRetries: 0 });
    }
    return this.client;
  }

  async *createMessage(
    systemPrompt: string,
    messages: Message[],
    _metadata?: CreateMessageMetadata,
    { tools: specs = [], signal: _signal, ...kwargs }: CreateMessageOptions = {},
  ): AsyncGenerator<StreamChunk> {
    const formatted: Record<string, unknown>[] = systemPrompt ? [{ role: "system", content: systemPrompt }] : [];
    for (const message of messages) {
      if (message.role === "system") throw new Error("Pass system messages through systemPrompt");
      const item: Record<string, unknown> = { role: message.role, content: message.content };
      if (message.toolCallId) item.tool_call_id = message.toolCallId;
      if (message.toolCalls.length) {
        item.tool_calls = message.toolCalls.map((call) => ({
          id: call.id,
          type: "function",
          function: { name: call.name, arguments: JSON.stringify(call.arguments) },
        }));
      }
      formatted.push(item);
    }
    const options: Record<string, unknown> = { ...this.options, ...kwargs };
    if (["model", "messages", "stream", "n"].some((key) => key in options)) {
      throw new Error("model, messages, stream and n are controlled by the adapter");
    }
    if (specs.length) {
      options.tools = specs.map((spec: ToolSpec) => ({
        type: "function",
        function: { name: spec.name, description: spec.description, parameters: spec.parameters },
      }));
    } else {
      delete options.tool_choice;
      delete options.parallel_tool_calls;
    }
    options.stream_options = { ...((options.stream_options as object) ?? {}), include_usage: true };
    const client = await this.getClient();
    const stream = await client.chat.completions.create({ model: this.model, messages: formatted, stream: true, ...options });
    const buffers = new Map<number, { id: string; name: string; arguments: string }>();
    let finishReason: string | null = null;
    let usage: { prompt_tokens?: number; completion_tokens?: number } | null = null;
    let model = this.model;
    try {
      for await (const chunk of stream) {
        model = chunk.model || model;
        if (chunk.usage) usage = chunk.usage;
        if (!chunk.choices?.length) continue;
        const choice = chunk.choices[0];
        if (choice.finish_reason) finishReason = choice.finish_reason;
        const delta = choice.delta ?? {};
        if (delta.content) yield { type: "text", text: delta.content };
        const reasoning = delta.reasoning || delta.reasoning_content;
        if (reasoning) yield { type: "reasoning", text: reasoning };
        for (const call of delta.tool_calls ?? []) {
          let buffer = buffers.get(call.index);
          if (buffer === undefined) buffers.set(call.index, (buffer = { id: "", name: "", arguments: "" }));
          if (call.id) buffer.id = call.id;
          if (call.function) {
            buffer.name += call.function.name ?? "";
            buffer.arguments += call.function.arguments ?? "";
          }
        }
      }
      if (finishReason !== "stop" && finishReason !== "tool_calls") throw new Error(`Incomplete model response: ${finishReason}`);
      for (const index of [...buffers.keys()].sort((a, b) => a - b)) {
        const buffer = buffers.get(index) as { id: string; name: string; arguments: string };
        const args = JSON.parse(buffer.arguments || "{}");
        if (args === null || typeof args !== "object" || Array.isArray(args)) throw new TypeError("Tool arguments must be a JSON object");
        const call = toolCall({ id: buffer.id, name: buffer.name, arguments: args });
        yield { type: "tool_use", id: call.id, name: call.name, input: call.arguments };
      }
      if (usage !== null) {
        yield { type: "usage", inputTokens: usage.prompt_tokens ?? 0, outputTokens: usage.completion_tokens ?? 0 };
      }
      yield { type: "metadata", metadata: { finishReason, model } };
    } finally {
      await closeStream(stream);
    }
  }

  async close(): Promise<void> {
    if (this.ownsClient && this.client !== null) await (this.client as { close?: () => Promise<void> }).close?.();
  }
}
