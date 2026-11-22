/** The unified provider contract; implementations are async generators. */

import type { CreateMessageMetadata, Message, ModelRequest, ModelResponse, StreamChunk, ToolSpec } from "../types.ts";

export interface CreateMessageOptions {
  tools?: readonly ToolSpec[];
  /** Aborted when the run is cancelled or times out; stop streaming and release resources. */
  signal?: AbortSignal;
  [option: string]: unknown;
}

export interface Provider {
  /** Yield normalized chunks; a tool_use chunk contains a complete tool call. */
  createMessage(
    systemPrompt: string,
    messages: Message[],
    metadata?: CreateMessageMetadata,
    options?: CreateMessageOptions,
  ): AsyncIterable<StreamChunk>;
  readonly model?: string;
}

export abstract class ApiHandler implements Provider {
  abstract createMessage(
    systemPrompt: string,
    messages: Message[],
    metadata?: CreateMessageMetadata,
    options?: CreateMessageOptions,
  ): AsyncIterable<StreamChunk>;

  /** Convenience collector; createMessage is the sole provider primitive. */
  async complete(request: ModelRequest, signal?: AbortSignal): Promise<ModelResponse> {
    const { collectResponse } = await import("./streaming.ts");
    return collectResponse(this, request, { signal });
  }
}

export function providerName(provider: Provider): string {
  return provider.constructor?.name && provider.constructor.name !== "Object" ? provider.constructor.name : "Provider";
}

export function providerModel(provider: Provider): string | null {
  return typeof provider.model === "string" ? provider.model : null;
}
