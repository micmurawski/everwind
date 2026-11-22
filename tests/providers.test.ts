import { expect, test } from "bun:test";
import { z } from "zod";
import { Agent, type Message, type ModelRequest, OpenAIChat, type StreamChunk, Tools, message, tool } from "../src/index.ts";

class FakeStream {
  closed = false;
  constructor(readonly chunks: unknown[]) {}
  async *[Symbol.asyncIterator]() {
    for (const chunk of this.chunks) yield chunk;
  }
  async close() {
    this.closed = true;
  }
}

const delta = ({ text, calls, finish, reasoning }: { text?: string; calls?: unknown[]; finish?: string; reasoning?: string } = {}) => ({
  choices: [{ finish_reason: finish ?? null, delta: { content: text ?? null, tool_calls: calls ?? null, reasoning_content: reasoning ?? null } }],
  usage: null,
  model: "test-model",
});
const toolDelta = (index: number, { id, name, args }: { id?: string; name?: string; args?: string } = {}) => ({
  index,
  id: id ?? null,
  function: { name: name ?? null, arguments: args ?? null },
});
const clientFor = (stream: FakeStream, captured: Record<string, unknown>[] = []) => ({
  chat: { completions: { create: async (request: Record<string, unknown>) => (captured.push(request), stream) } },
});
const request = (messages: Message[] = [], tools: ModelRequest["tools"] = []): ModelRequest => ({ runId: "", messages, tools, turn: 1, purpose: "act" });

test("OpenAI chat round trip with an injected client", async () => {
  const captured: Record<string, unknown>[] = [];
  const stream = new FakeStream([
    delta({ text: "Looking ", calls: [toolDelta(0, { id: "c1", name: "look", args: '{"key":' })] }),
    delta({ text: "up", calls: [toolDelta(0, { name: "up", args: '"a"}' })], finish: "tool_calls" }),
    { choices: [], usage: { prompt_tokens: 10, completion_tokens: 2 } },
  ]);
  const provider = new OpenAIChat("test-model", { client: clientFor(stream, captured), temperature: 0.1 });
  const response = await provider.complete(
    request(
      [
        message({ role: "system", content: "Be useful" }),
        message({ role: "assistant", toolCalls: [{ id: "old", name: "lookup", arguments: { key: "b" } }] }),
        message({ role: "tool", content: "old result", toolCallId: "old" }),
      ],
      [{ name: "lookup", description: "Lookup", parameters: { type: "object" } }],
    ),
  );
  expect(response.toolCalls[0].arguments).toEqual({ key: "a" });
  expect(response.text).toBe("Looking up");
  expect(stream.closed).toBe(true);
  expect(response.usage).toEqual({ inputTokens: 10, outputTokens: 2 });
  const sent = captured[0] as Record<string, any>;
  expect(sent.messages[1].tool_calls[0].id).toBe("old");
  expect(sent.messages[2].tool_call_id).toBe("old");
  expect(sent.tools[0].function.name).toBe("lookup");
  expect([sent.temperature, sent.stream, sent.stream_options.include_usage]).toEqual([0.1, true, true]);
  await provider.close(); // Does not close a caller-owned client.
});

test("OpenAI planning omits tools, and truncation fails", async () => {
  const stream = new FakeStream([delta({ finish: "length" })]);
  const captured: Record<string, unknown>[] = [];
  const provider = new OpenAIChat("test", { client: clientFor(stream, captured) });
  await expect(provider.complete(request())).rejects.toThrow(/Incomplete/);
  expect("tools" in captured[0]).toBe(false);
  expect(stream.closed).toBe(true);
});

test("a direct stream yields text before tool assembly and can be closed", async () => {
  const stream = new FakeStream([
    delta({ text: "First token", calls: [toolDelta(0, { id: "c1", name: "lookup", args: '{"key":' })] }),
    delta({ calls: [toolDelta(0, { args: '"value"}' })], finish: "tool_calls" }),
  ]);
  const provider = new OpenAIChat("test", { client: clientFor(stream) });
  const chunks = provider.createMessage("System", [message({ role: "user", content: "Task" })]);
  expect((await chunks.next()).value).toEqual({ type: "text", text: "First token" });
  expect(stream.closed).toBe(false);
  await chunks.return(undefined);
  expect(stream.closed).toBe(true);
});

test("interleaved tool fragments keep their order, with reasoning", async () => {
  const stream = new FakeStream([
    delta({ reasoning: "Check ", calls: [toolDelta(1, { id: "b", name: "second", args: '{"b":' })] }),
    delta({ reasoning: "inputs", calls: [toolDelta(0, { id: "a", name: "first", args: '{"a":1}' })] }),
    delta({ calls: [toolDelta(1, { args: "2}" })], finish: "tool_calls" }),
  ]);
  const response = await new OpenAIChat("test", { client: clientFor(stream) }).complete(request());
  expect(response.toolCalls.map((item) => item.id)).toEqual(["a", "b"]);
  expect(response.toolCalls[1].arguments).toEqual({ b: 2 });
  expect(response.reasoning).toBe("Check inputs");
});

for (const [fragment, finish] of [['{"a":', "tool_calls"], ["{}", undefined]] as const) {
  test(`a malformed or unfinished OpenAI stream fails (${fragment})`, async () => {
    const stream = new FakeStream([delta({ calls: [toolDelta(0, { id: "c1", name: "tool", args: fragment })], finish })]);
    await expect(new OpenAIChat("test", { client: clientFor(stream) }).complete(request())).rejects.toThrow();
    expect(stream.closed).toBe(true);
  });
}

test("the runtime consumes createMessage only and records each chunk", async () => {
  const provider = {
    async *createMessage(system: string, messages: Message[], metadata?: { turn?: number; runId?: string }) {
      expect(system).toBe("Chef");
      expect(messages.map((item) => item.role)).toEqual(["user"]);
      expect(metadata?.turn).toBe(1);
      expect(metadata?.runId).toBeTruthy();
      yield { type: "reasoning", text: "Checking" } as StreamChunk;
      yield { type: "text", text: "Dinner " } as StreamChunk;
      yield { type: "text", text: "ready" } as StreamChunk;
      yield { type: "usage", inputTokens: 3, outputTokens: 2 } as StreamChunk;
      yield { type: "usage", outputTokens: 1 } as StreamChunk;
    },
  };
  const run = await new Agent({ provider, systemPrompt: "Chef" }).run("Cook");
  expect([run.status, run.output]).toEqual(["completed", "Dinner ready"]);
  expect(run.usage).toEqual({ inputTokens: 3, outputTokens: 3 });
  expect(run.messages.at(-1)?.metadata.reasoning).toBe("Checking");
  expect(run.events.filter((event) => event.kind === "model_chunk").length).toBe(5);
});

test("a failed stream never executes a partial tool turn", async () => {
  let invocations = 0;
  let closed = 0;
  const action = tool({ name: "action", input: z.object({}), run: () => void (invocations += 1) });
  const provider = {
    async *createMessage() {
      try {
        yield { type: "tool_use", id: "c1", name: "action", input: {} } as StreamChunk;
        throw new Error("stream interrupted");
      } finally {
        closed += 1;
      }
    },
  };
  const run = await new Agent({ provider, tools: new Tools([action]) }).run("Task");
  expect(run.status).toBe("failed");
  expect(run.error).toContain("stream interrupted");
  expect([invocations, closed]).toEqual([0, 1]);
  expect(run.messages.some((item) => item.role === "assistant")).toBe(false);
});

test("cancelling a turn closes the provider generator", async () => {
  let closed = 0;
  const provider = {
    async *createMessage(_s: string, _m: Message[], _meta?: unknown, options?: { signal?: AbortSignal }) {
      try {
        yield { type: "text", text: "Starting" } as StreamChunk;
        await new Promise((_, reject) => options?.signal?.addEventListener("abort", () => reject(options.signal?.reason)));
      } finally {
        closed += 1;
      }
    },
  };
  const controller = new AbortController();
  const pending = new Agent({ provider }).run("Task", { signal: controller.signal });
  await Bun.sleep(2);
  controller.abort();
  await expect(pending).rejects.toThrow();
  await Bun.sleep(1);
  expect(closed).toBe(1);
});
