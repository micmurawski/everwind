/** Shared stream collection for runtime turns and optional complete() calls. */

import { abortable } from "../util.ts";
import {
  type Message,
  type ModelRequest,
  type ModelResponse,
  type StreamChunk,
  type ToolCall,
  modelResponse,
  toolCall,
} from "../types.ts";
import type { Provider } from "./base.ts";
import type { JsonObject } from "../util.ts";

/** Emit normalized chunks from a scripted or full response. */
export function* responseChunks(response: ModelResponse): Generator<StreamChunk> {
  if (response.reasoning) yield { type: "reasoning", text: response.reasoning };
  if (response.text) yield { type: "text", text: response.text };
  for (const call of response.toolCalls) {
    yield { type: "tool_use", id: call.id, name: call.name, input: call.arguments, metadata: call.metadata };
  }
  if (Object.keys(response.usage).length) yield { type: "usage", ...response.usage };
  if (Object.keys(response.metadata).length) yield { type: "metadata", metadata: response.metadata };
}

export interface CollectOptions {
  onChunk?: (chunk: StreamChunk) => void;
  signal?: AbortSignal;
}

export async function collectResponse(
  provider: Provider,
  request: ModelRequest,
  options: CollectOptions = {},
): Promise<ModelResponse> {
  const systemPrompt = request.messages
    .filter((message) => message.role === "system")
    .map((message) => message.content)
    .join("\n\n");
  const messages: Message[] = structuredClone(request.messages.filter((message) => message.role !== "system"));
  const stream = provider.createMessage(
    systemPrompt,
    messages,
    { runId: request.runId, taskId: request.runId, turn: request.turn, purpose: request.purpose },
    { tools: structuredClone(request.tools), signal: options.signal },
  );
  const iterator = stream[Symbol.asyncIterator]();
  const text: string[] = [];
  const reasoning: string[] = [];
  const calls: ToolCall[] = [];
  const usage: Record<string, number> = {};
  const metadata: JsonObject = {};
  let finished = false;
  try {
    for (;;) {
      const next = await abortable(iterator.next(), options.signal);
      if (next.done) {
        finished = true;
        break;
      }
      const chunk = next.value;
      switch (chunk.type) {
        case "text":
        case "reasoning":
          if (typeof chunk.text !== "string") throw new TypeError(`${chunk.type} chunk requires string text`);
          (chunk.type === "text" ? text : reasoning).push(chunk.text);
          break;
        case "tool_use":
          calls.push(toolCall({ id: chunk.id, name: chunk.name, arguments: chunk.input, metadata: chunk.metadata ?? {} }));
          break;
        case "usage":
          // Usage chunks are increments, not repeated cumulative snapshots.
          for (const [key, value] of Object.entries(chunk)) {
            if (key === "type") continue;
            if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`Invalid usage counter: ${key}`);
            usage[key] = (usage[key] ?? 0) + value;
          }
          break;
        case "metadata":
          Object.assign(metadata, chunk.metadata);
          break;
        case "grounding": {
          const sources = (metadata.sources as JsonObject[] | undefined) ?? [];
          metadata.sources = [...sources, ...(chunk.sources as JsonObject[])];
          break;
        }
        default:
          throw new Error(`Unknown stream chunk type: ${JSON.stringify((chunk as { type: unknown }).type)}`);
      }
      options.onChunk?.(chunk);
    }
  } finally {
    if (!finished) {
      // Close the provider's stream; do not wait for a generator stuck in an await.
      iterator.return?.()?.catch?.(() => {});
    }
  }
  return modelResponse({ text: text.join(""), reasoning: reasoning.join(""), toolCalls: calls, usage, metadata });
}
