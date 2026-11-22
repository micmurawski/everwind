import { type ModelResponse, type ModelResponseInput, type Message, type CreateMessageMetadata, type StreamChunk, modelResponse } from "../types.ts";
import { ApiHandler, type CreateMessageOptions } from "./base.ts";
import { responseChunks } from "./streaming.ts";

/** Per-turn script; metadata.turn selects a response without a shared cursor. */
export class ScriptedProvider extends ApiHandler {
  readonly responses: readonly ModelResponse[];

  constructor(responses: readonly ModelResponseInput[]) {
    super();
    this.responses = responses.map((response) => modelResponse(response));
  }

  async *createMessage(
    _systemPrompt: string,
    _messages: Message[],
    metadata?: CreateMessageMetadata,
    _options?: CreateMessageOptions,
  ): AsyncGenerator<StreamChunk> {
    const turn = metadata?.turn ?? 1;
    if (!(turn >= 1 && turn <= this.responses.length)) throw new Error(`Script exhausted at model turn ${turn}`);
    yield* responseChunks(structuredClone(this.responses[turn - 1]));
  }
}
