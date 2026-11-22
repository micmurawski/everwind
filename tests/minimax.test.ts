import { expect, test } from "bun:test";
import { z } from "zod";
import { Agent, MiniMax, type ModelRequest, Tools, message, tool } from "../src/index.ts";

const block = (data: Record<string, unknown>) => data;
const delta = (type: string, index: number, data: Record<string, unknown>) => ({ type: "content_block_delta", index, delta: { type, ...data } });

class Stream {
  closed = false;
  readonly events: unknown[];
  readonly response: Record<string, unknown>;

  constructor(content: unknown[], events: unknown[] = [], { stopReason = "end_turn" as string | null, stopped = true } = {}) {
    this.events = [...events, ...(stopped ? [{ type: "message_stop" }] : [])];
    this.response = {
      content,
      stop_reason: stopReason,
      model: "MiniMax-test",
      usage: { input_tokens: 12, output_tokens: 5, cache_read_input_tokens: 3, cache_creation_input_tokens: 2 },
    };
  }

  async *[Symbol.asyncIterator]() {
    for (const event of this.events) yield event;
  }

  async finalMessage() {
    return this.response;
  }

  abort() {
    this.closed = true;
  }
}

class Client {
  readonly requests: Record<string, any>[] = [];
  closed = false;
  readonly messages: { stream: (request: Record<string, unknown>) => Stream };
  private readonly queue: Stream[];

  constructor(...streams: Stream[]) {
    this.queue = streams;
    this.messages = { stream: (request) => (this.requests.push(request), this.queue.shift() as Stream) };
  }

  async close() {
    this.closed = true;
  }
}

const request = (text: string): ModelRequest => ({ runId: "", messages: [message({ role: "user", content: text })], tools: [], turn: 1, purpose: "act" });

test("streams text and reasoning once, and accounts final usage", async () => {
  const stream = new Stream(
    [block({ type: "thinking", thinking: "Check inputs", signature: "opaque" }), block({ type: "text", text: "Hello chef" })],
    [
      delta("thinking_delta", 0, { thinking: "Check inputs" }),
      delta("text_delta", 1, { text: "Hello " }),
      { type: "text", text: "Hello " }, // An SDK convenience event must not duplicate text.
      delta("text_delta", 1, { text: "chef" }),
      { type: "message_delta", usage: { output_tokens: 5 } },
    ],
  );
  const client = new Client(stream);
  const provider = new MiniMax("MiniMax-test", { client, temperature: 0.7 });
  const response = await provider.complete(request("Hi"));
  expect([response.text, response.reasoning]).toEqual(["Hello chef", "Check inputs"]);
  expect(response.usage).toEqual({ inputTokens: 12, outputTokens: 5, cacheReadTokens: 3, cacheWriteTokens: 2 });
  expect((response.metadata.minimaxContent as { signature: string }[])[0].signature).toBe("opaque");
  expect([client.requests[0].max_tokens, client.requests[0].temperature]).toEqual([4096, 0.7]);
  expect(stream.closed).toBe(true);
  await provider.close();
  expect(client.closed).toBe(false); // Injected clients remain caller-owned.
});

test("a full tool round trip keeps thinking and groups parallel results", async () => {
  const add = tool({ name: "add", input: z.object({ a: z.number().int(), b: z.number().int() }), run: ({ a, b }) => a + b });
  const original = [
    block({ type: "thinking", thinking: "Calculate both sums", signature: "keep-this" }),
    block({ type: "text", text: "Calculating" }),
    block({ type: "tool_use", id: "a", name: "add", input: { a: 1, b: 2 } }),
    block({ type: "tool_use", id: "b", name: "add", input: { a: 3, b: 4 } }),
  ];
  const first = new Stream(
    original,
    [
      delta("thinking_delta", 0, { thinking: "Calculate both sums" }),
      delta("text_delta", 1, { text: "Calculating" }),
      delta("input_json_delta", 2, { partial_json: '{"a":1,' }),
      delta("input_json_delta", 3, { partial_json: '{"a":3,"b":4}' }),
      delta("input_json_delta", 2, { partial_json: '"b":2}' }),
    ],
    { stopReason: "tool_use" },
  );
  const second = new Stream([block({ type: "text", text: "3 and 7" })], [delta("text_delta", 0, { text: "3 and 7" })]);
  const client = new Client(first, second);
  const run = await new Agent({ provider: new MiniMax("MiniMax-test", { client }), tools: new Tools([add]) }).run("Add");
  expect([run.status, run.output, run.toolCalls]).toEqual(["completed", "3 and 7", 2]);
  expect("input_schema" in client.requests[0].tools[0]).toBe(true);
  const messages = client.requests[1].messages;
  expect(messages[1].role).toBe("assistant");
  expect(messages[1].content).toEqual(original);
  expect(messages[2].content.map((item: { tool_use_id: string }) => item.tool_use_id)).toEqual(["a", "b"]);
  expect(messages[2].content.map((item: { content: string }) => item.content)).toEqual(["3", "7"]);
  expect([first.closed, second.closed]).toEqual([true, true]);
});

for (const [stopReason, stopped] of [["max_tokens", true], [null, true], ["end_turn", false]] as const) {
  test(`a truncated stream fails and closes (${stopReason}, ${stopped})`, async () => {
    const stream = new Stream([block({ type: "text", text: "Partial" })], [delta("text_delta", 0, { text: "Partial" })], { stopReason, stopped });
    const run = await new Agent({ provider: new MiniMax("test", { client: new Client(stream) }) }).run("Task");
    expect(run.status).toBe("failed");
    expect(run.error).toContain("Incomplete MiniMax");
    expect([run.output, stream.closed]).toEqual([null, true]);
  });
}

test("incomplete tool JSON is not executed", async () => {
  const stream = new Stream([block({ type: "tool_use", id: "a", name: "action", input: { a: 1 } })], [delta("input_json_delta", 0, { partial_json: '{"a":1' })], {
    stopReason: "tool_use",
  });
  const run = await new Agent({ provider: new MiniMax("test", { client: new Client(stream) }) }).run("Task");
  expect([run.status, run.toolCalls, stream.closed]).toEqual(["failed", 0, true]);
});

test("a direct stream is incremental and closes early", async () => {
  const stream = new Stream([block({ type: "text", text: "One two" })], [delta("text_delta", 0, { text: "One " }), delta("text_delta", 0, { text: "two" })]);
  const chunks = new MiniMax("test", { client: new Client(stream) }).createMessage("System", [message({ role: "user", content: "Hi" })]);
  expect((await chunks.next()).value).toEqual({ type: "text", text: "One " });
  expect(stream.closed).toBe(false);
  await chunks.return(undefined);
  expect(stream.closed).toBe(true);
});

test("no tools, and request options", async () => {
  const stream = new Stream([block({ type: "text", text: "Plan" })], [delta("text_delta", 0, { text: "Plan" })]);
  const client = new Client(stream);
  const provider = new MiniMax("test", { client, tool_choice: { type: "auto" } });
  for await (const _ of provider.createMessage("System", [message({ role: "user", content: "Plan" })], undefined, {
    tools: [],
    max_tokens: 8192,
    temperature: 0.5,
  })) {
    // Consume the stream.
  }
  const sent = client.requests[0];
  expect([sent.max_tokens, sent.temperature]).toEqual([8192, 0.5]);
  expect("tools" in sent || "tool_choice" in sent).toBe(false);
});

test("an owned client uses the MiniMax key, endpoint and no retries", async () => {
  const constructed: unknown[] = [];
  const client = new Client();
  const previous = process.env.MINIMAX_API_KEY;
  process.env.MINIMAX_API_KEY = "test-only";
  try {
    const provider = new MiniMax("test", { clientFactory: (options) => (constructed.push(options), client) });
    expect(constructed).toEqual([{ apiKey: "test-only", baseURL: "https://api.minimax.io/anthropic", maxRetries: 0 }]);
    await provider.close();
    expect(client.closed).toBe(true);
  } finally {
    if (previous === undefined) delete process.env.MINIMAX_API_KEY;
    else process.env.MINIMAX_API_KEY = previous;
  }
});

test("a missing key has a clear error", () => {
  const previous = process.env.MINIMAX_API_KEY;
  delete process.env.MINIMAX_API_KEY;
  try {
    expect(() => new MiniMax("test")).toThrow(/MINIMAX_API_KEY/);
  } finally {
    if (previous !== undefined) process.env.MINIMAX_API_KEY = previous;
  }
});
