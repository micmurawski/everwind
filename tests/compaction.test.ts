import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  Agent,
  type CompactionProposal,
  type CompactionRequest,
  type CompactionStrategy,
  FileSessionStore,
  LimitReached,
  type LimitsInput,
  type Loop,
  type Message,
  MessageRange,
  OlderHistory,
  type Provider,
  ReAct,
  Remove,
  Replace,
  Run,
  Runtime,
  ScriptedProvider,
  SessionLogError,
  type StreamChunk,
  Summarize,
  Tools,
  limits,
  message,
  validateConversation,
} from "../src/index.ts";
import { tmpDir } from "./helpers.ts";

function runtime(provider: Provider = new ScriptedProvider([]), limitOptions: LimitsInput = {}, signal?: AbortSignal): Runtime {
  const run = new Run({ messages: [message({ role: "system", content: "System" }), message({ role: "user", content: "Original task" })] });
  return new Runtime(run, provider, new Tools(), limits(limitOptions), { signal });
}

const snapshot = (rt: Runtime) => structuredClone(rt.run.messages);

function toolTurn(): Message[] {
  return [
    message({ role: "assistant", content: "Checking", toolCalls: [{ id: "a", name: "check" }, { id: "b", name: "check" }] }),
    message({ role: "tool", toolCallId: "b", content: "B verified" }),
    message({ role: "tool", toolCallId: "a", content: "A failed", isError: true }),
    message({ role: "assistant", content: "Use B; A failed" }),
  ];
}

const proposal = (messages: Message[]): CompactionProposal => ({ messages });

for (const strategy of [new Remove(), new Replace("Final plan")]) {
  test(`${strategy.constructor.name} only touches the selected region`, async () => {
    const rt = runtime();
    const before = snapshot(rt);
    const marker = rt.markMessages();
    rt.say("Plan");
    rt.run.messages.push(...toolTurn());
    const stop = rt.markMessages();
    rt.say("Latest user request");
    const messages = rt.run.messages;
    const result = await rt.compact({ selection: new MessageRange(marker, stop), strategy });
    expect([result.applied, result.start, result.stop]).toEqual([true, 2, 7]);
    expect(rt.run.messages).toBe(messages);
    expect(rt.run.messages.slice(0, 2)).toEqual(before);
    expect(rt.run.messages.at(-1)?.content).toBe("Latest user request");
    expect([rt.run.modelCalls, rt.run.toolCalls]).toEqual([0, 0]);
    expect(rt.run.messages.length).toBe(strategy instanceof Remove ? 3 : 4);
    const event = rt.run.events.at(-1);
    expect(event?.kind).toBe("context_compaction");
    expect((event?.data.removedMessages as unknown[]).length).toBe(5);
    if (result.replacement.length) {
      (result.replacement[0] as { metadata: object }).metadata = {};
      expect(Object.keys(rt.run.messages[2].metadata).length).toBeGreaterThan(0); // The result is detached.
    }
    validateConversation(rt.run.messages);
  });
}

test("a summary is isolated and accounted, and provider metadata is not replayed", async () => {
  class SummaryProvider {
    readonly model = "summary-model";
    async *createMessage(system: string, messages: Message[], metadata?: { purpose?: string; turn?: number }, options?: { tools?: readonly unknown[] }) {
      expect(system).toBe("Summarize verified facts.");
      expect([metadata?.purpose, metadata?.turn, options?.tools]).toEqual(["compact", 1, []]);
      expect(messages.map((item) => item.role)).toEqual(["user"]);
      const transcript = JSON.parse(messages[0].content.split("\n").slice(1).join("\n"));
      expect(transcript.map((item: Message) => item.role)).toEqual(["assistant", "tool", "tool", "assistant"]);
      expect(transcript[2].isError).toBe(true);
      yield { type: "usage", inputTokens: 10 } as StreamChunk;
      yield { type: "text", text: " B verified; A failed. " } as StreamChunk;
      yield { type: "reasoning", text: "internal" } as StreamChunk;
      yield { type: "usage", outputTokens: 4 } as StreamChunk;
      yield { type: "metadata", metadata: { minimaxContent: [{ type: "thinking", signature: "opaque" }] } } as StreamChunk;
    }
  }
  const rt = runtime();
  const marker = rt.markMessages();
  rt.run.messages.push(...toolTurn());
  const result = await rt.compactSince(marker, { strategy: new Summarize({ prompt: "Summarize verified facts.", provider: new SummaryProvider() }) });
  expect(result.applied).toBe(true);
  expect(rt.run.messages.at(-1)?.content).toBe("B verified; A failed.");
  expect([rt.run.messages.length, rt.pending.length]).toEqual([3, 0]);
  expect(rt.run.messages.at(-1)?.metadata).toEqual({ compaction: { kind: "summary" } });
  expect(rt.run.modelCalls).toBe(1);
  expect(rt.run.usage).toEqual({ inputTokens: 10, outputTokens: 4 });
  const response = rt.run.events.find((event) => event.kind === "model_response")?.data as Record<string, any>;
  expect([response.conversation, response.purpose]).toEqual([false, "compact"]);
  expect(response.response.metadata.minimaxContent[0].signature).toBe("opaque");
  const requested = rt.run.events.find((event) => event.kind === "model_request")?.data;
  expect([requested?.provider, requested?.model]).toEqual(["SummaryProvider", "summary-model"]);
  expect(rt.run.messages.some((item) => item.content === "Summarize verified facts.")).toBe(false);
});

test("summarization uses the shared budget and the default provider", async () => {
  const rt = runtime(new ScriptedProvider([{ text: "summary" }]), { maxModelCalls: 1 });
  let marker = rt.markMessages();
  rt.say("Long draft");
  expect((await rt.compactSince(marker, { strategy: new Summarize() })).applied).toBe(true);
  await expect(rt.complete()).rejects.toThrow(LimitReached);
  marker = rt.markMessages();
  rt.say("More draft");
  const original = snapshot(rt);
  await expect(rt.compactSince(marker, { strategy: new Summarize() })).rejects.toThrow(LimitReached);
  expect(rt.run.messages).toEqual(original);
  expect(rt.run.modelCalls).toBe(1);
  expect((await rt.compactSince(marker, { strategy: new Remove() })).applied).toBe(true);
});

for (const kind of ["empty", "tool", "error"]) {
  test(`a failed summary (${kind}) never removes history and keeps received usage`, async () => {
    const broken = {
      async *createMessage() {
        yield { type: "usage", inputTokens: 3 } as StreamChunk;
        if (kind === "tool") yield { type: "tool_use", id: "c", name: "effect", input: {} } as StreamChunk;
        else if (kind === "error") {
          yield { type: "text", text: "partial" } as StreamChunk;
          throw new Error("stream failed");
        } else yield { type: "text", text: " " } as StreamChunk;
      },
    };
    const rt = runtime(broken);
    const marker = rt.markMessages();
    rt.say("Keep this");
    const original = snapshot(rt);
    await expect(rt.compactSince(marker, { strategy: new Summarize() })).rejects.toThrow();
    expect(rt.run.messages).toEqual(original);
    expect(rt.pending).toEqual([]);
    expect([rt.run.modelCalls, rt.run.usage]).toEqual([1, { inputTokens: 3 }]);
    expect(rt.run.events.at(-1)?.kind).toBe("context_compaction_failed");
    rt.say("Runtime is unlocked");
  });
}

test("cancellation closes the stream and keeps history and usage", async () => {
  let entered!: () => void;
  const started = new Promise<void>((resolve) => (entered = resolve));
  let closed = false;
  const blocked = {
    async *createMessage(_s: string, _m: Message[], _meta?: unknown, options?: { signal?: AbortSignal }) {
      try {
        yield { type: "usage", inputTokens: 7 } as StreamChunk;
        entered();
        await new Promise((_, reject) => options?.signal?.addEventListener("abort", () => reject(options.signal?.reason)));
      } finally {
        closed = true;
      }
    },
  };
  const controller = new AbortController();
  const rt = runtime(blocked, {}, controller.signal);
  const marker = rt.markMessages();
  rt.say("Keep original");
  const original = snapshot(rt);
  const pending = rt.compactSince(marker, { strategy: new Summarize() });
  await started;
  expect(() => rt.say("Concurrent instruction")).toThrow(/busy/);
  await expect(rt.complete()).rejects.toThrow(/busy/);
  await expect(rt.compactSince(marker, { strategy: new Remove() })).rejects.toThrow(/busy/);
  controller.abort(new Error("cancelled"));
  await expect(pending).rejects.toThrow(/cancelled/);
  await Bun.sleep(1);
  expect(closed).toBe(true);
  expect(rt.run.messages).toEqual(original);
  expect(rt.run.usage).toEqual({ inputTokens: 7 });
});

test("markers reject foreign, stale and changed prefixes", async () => {
  const rt = runtime();
  const marker = rt.markMessages();
  rt.say("Draft");
  await expect(runtime().compactSince(marker, { strategy: new Remove() })).rejects.toThrow(/another runtime/);
  rt.run.messages[0].metadata.changed = true;
  await expect(rt.compactSince(marker, { strategy: new Remove() })).rejects.toThrow(/earlier messages/);
  delete rt.run.messages[0].metadata.changed;
  const stop = rt.markMessages();
  await expect(rt.compact({ selection: new MessageRange(stop, marker), strategy: new Remove() })).rejects.toThrow(/starts after/);
  await rt.compactSince(marker, { strategy: new Replace("Summary") });
  await expect(rt.compactSince(marker, { strategy: new Remove() })).rejects.toThrow(/Stale/);
  await expect(rt.compactSince(stop, { strategy: new Remove() })).rejects.toThrow(/Stale/);
});

test("pending calls and in-flight requests block marking and compaction", async () => {
  let rt = runtime(new ScriptedProvider([{ toolCalls: [{ id: "c", name: "check" }] }]));
  let marker = rt.markMessages();
  await rt.complete();
  expect(() => rt.markMessages()).toThrow(/pending/);
  await expect(rt.compactSince(marker, { strategy: new Remove() })).rejects.toThrow(/pending/);
  await rt.executeTools();
  expect((await rt.compactSince(marker, { strategy: new Remove() })).applied).toBe(true);

  let entered!: () => void;
  const started = new Promise<void>((resolve) => (entered = resolve));
  const slow = {
    async *createMessage(_s: string, _m: Message[], _meta?: unknown, options?: { signal?: AbortSignal }) {
      entered();
      await new Promise((_, reject) => options?.signal?.addEventListener("abort", () => reject(options.signal?.reason)));
      yield { type: "text", text: "unreachable" } as StreamChunk;
    },
  };
  const controller = new AbortController();
  rt = runtime(slow, {}, controller.signal);
  marker = rt.markMessages();
  const pending = rt.complete();
  await started;
  expect(() => rt.markMessages()).toThrow(/busy/);
  await expect(rt.compactSince(marker, { strategy: new Remove() })).rejects.toThrow(/busy/);
  controller.abort(new Error("cancelled"));
  await expect(pending).rejects.toThrow(/cancelled/);
});

test("OlderHistory keeps the first user message and recent complete turns", async () => {
  const rt = runtime();
  rt.run.messages.push(...toolTurn());
  rt.say("Second task");
  rt.run.messages.push(...toolTurn());
  rt.say("Third task");
  const original = snapshot(rt);
  const result = await rt.compact({ selection: new OlderHistory(2), strategy: new Replace("Earlier results") });
  expect([result.applied, result.start, result.stop]).toEqual([true, 2, 6]);
  expect(rt.run.messages).toEqual([...original.slice(0, 2), ...result.replacement, ...original.slice(6)]);
  validateConversation(rt.run.messages);
  expect((await rt.compact({ selection: new OlderHistory(10), strategy: new Remove() })).applied).toBe(false);
});

test("custom proposals cannot break tool history or change the system prompt", async () => {
  class Custom implements CompactionStrategy {
    constructor(readonly messages: Message[]) {}
    async compact(request: CompactionRequest) {
      (request.messages[0] as Message).metadata.mutated = true;
      return proposal(this.messages);
    }
  }
  const rt = runtime();
  const marker = rt.markMessages();
  rt.say("Draft");
  const original = snapshot(rt);
  for (const messages of [toolTurn().slice(0, 1), toolTurn().slice(1), [message({ role: "system", content: "Changed" })]]) {
    await expect(rt.compactSince(marker, { strategy: new Custom(messages) })).rejects.toThrow();
    expect(rt.run.messages).toEqual(original);
  }
  expect((await rt.compactSince(marker, { strategy: new Custom(toolTurn()) })).applied).toBe(true);
  validateConversation(rt.run.messages);
});

test("the token counter rejects growth without refunding the summary call", async () => {
  const rt = runtime(new ScriptedProvider([{ text: "much longer summary", usage: { outputTokens: 4 } }]));
  const marker = rt.markMessages();
  rt.say("tiny");
  const original = snapshot(rt);
  const counted: string[][] = [];
  const characters = async (messages: readonly Message[]) => {
    counted.push(messages.map((item) => item.content));
    return messages.reduce((total, item) => total + item.content.length, 0);
  };
  let result = await rt.compactSince(marker, { strategy: new Summarize(), tokenCounter: characters });
  expect([result.applied, result.reason]).toEqual([false, "not_smaller"]);
  expect(result.afterTokens as number).toBeGreaterThan(result.beforeTokens as number);
  expect(rt.run.messages).toEqual(original);
  expect([rt.run.modelCalls, rt.run.usage]).toEqual([1, { outputTokens: 4 }]);
  expect(counted.every((messages) => messages[0] === "System" && messages[1] === "Original task")).toBe(true);
  result = await rt.compactSince(marker, { strategy: new Remove(), tokenCounter: characters });
  expect(result.applied).toBe(true);
  expect(result.afterTokens as number).toBeLessThan(result.beforeTokens as number);
});

test("empty and unchanged regions do not invalidate markers or call models", async () => {
  const identity: CompactionStrategy = { compact: async (request) => proposal([...request.messages]) };
  const rt = runtime();
  const marker = rt.markMessages();
  let result = await rt.compactSince(marker, { strategy: new Summarize() });
  expect([result.reason, rt.run.modelCalls]).toEqual(["empty_selection", 0]);
  rt.say("Draft");
  result = await rt.compactSince(marker, { strategy: identity });
  expect([result.applied, result.reason]).toEqual([false, "unchanged"]);
  expect((await rt.compactSince(marker, { strategy: new Remove() })).applied).toBe(true);
});

test("direct conversation mutation during a strategy rejects the commit", async () => {
  const rt = runtime();
  const marker = rt.markMessages();
  rt.say("Original draft");
  const misbehaving: CompactionStrategy = {
    compact: async () => {
      rt.run.messages.push(message({ role: "user", content: "Concurrent direct append" }));
      return proposal([message({ role: "assistant", content: "Summary" })]);
    },
  };
  await expect(rt.compactSince(marker, { strategy: misbehaving })).rejects.toThrow(/changed during/);
  expect(rt.run.messages.slice(-2).map((item) => item.content)).toEqual(["Original draft", "Concurrent direct append"]);
});

test("a log failure does not apply the proposal", async () => {
  const rt = runtime();
  const marker = rt.markMessages();
  rt.say("Draft");
  const original = snapshot(rt);
  rt.run.eventSink = () => {
    throw new SessionLogError("disk failed");
  };
  await expect(rt.compactSince(marker, { strategy: new Remove() })).rejects.toThrow(SessionLogError);
  expect(rt.run.messages).toEqual(original);
  rt.run.eventSink = null;
  expect((await rt.compactSince(marker, { strategy: new Remove() })).applied).toBe(true);
});

test("checkpoints contain the handoff and logs keep the removed messages", async () => {
  const directory = tmpDir();
  const planning: Loop = {
    async run(rt) {
      const marker = rt.markMessages();
      rt.say("Generate a draft");
      await rt.complete({ allowTools: false, purpose: "plan" });
      await rt.compactSince(marker, { strategy: new Replace("Approved handoff") });
      const response = await rt.complete();
      rt.finish(response.text);
    },
  };
  const proceed = {
    async *createMessage(_s: string, messages: Message[]) {
      expect(messages.some((item) => item.content === "Approved handoff")).toBe(true);
      expect(messages.some((item) => ["Generate a draft", "Rejected draft"].includes(item.content))).toBe(false);
      yield { type: "text", text: "Continued" } as StreamChunk;
    },
  };
  const agent = new Agent({
    provider: new ScriptedProvider([{ text: "Rejected draft" }, { text: "Executed" }]),
    loop: planning,
    sessions: new FileSessionStore(join(directory, "sessions")),
    logDir: join(directory, "logs"),
  });
  const first = await agent.run("Task");
  expect([first.status, first.output]).toEqual(["completed", "Executed"]);
  expect(agent.sessions.load(first.sessionId)?.messages.map((item) => item.content)).toEqual(["Task", "Approved handoff", "Executed"]);
  expect(readFileSync(first.logPath as string, "utf-8")).toContain("Rejected draft");
  // Default ReAct on continuation, sharing the checkpoint store.
  const second = await agent.with({ provider: proceed, loop: new ReAct() }).run("Continue", { sessionId: first.sessionId });
  expect([second.status, second.output]).toEqual(["completed", "Continued"]);
});

test("a failed summary during a run keeps the checkpoint", async () => {
  const planning: Loop = {
    async run(rt) {
      const marker = rt.markMessages();
      rt.say("Draft must survive");
      await rt.compactSince(marker, { strategy: new Summarize() });
    },
  };
  const agent = new Agent({ provider: new ScriptedProvider([{ text: "" }]), loop: planning, sessions: new FileSessionStore(tmpDir()) });
  const run = await agent.run("Task");
  expect(run.status).toBe("failed");
  expect(agent.sessions.load(run.sessionId)?.messages.at(-1)?.content).toBe("Draft must survive");
});

for (const value of [0, -1, 1.5]) {
  test(`an invalid recent turn count is rejected (${value})`, () => {
    expect(() => new OlderHistory(value)).toThrow();
  });
}

test("an empty replacement or summary prompt is rejected", () => {
  expect(() => new Replace(" ")).toThrow();
  expect(() => new Summarize({ prompt: " " })).toThrow();
});

test("the summary generator expires, and a custom strategy can use it", async () => {
  const rt = runtime(new ScriptedProvider([{ text: "Facts" }]));
  const marker = rt.markMessages();
  rt.say("Some facts");
  const saved: CompactionRequest["generateSummary"][] = [];
  const custom: CompactionStrategy = {
    compact: async (request) => {
      saved.push(request.generateSummary);
      const response = await request.generateSummary("Extract facts");
      return proposal([message({ role: "assistant", content: response.text })]);
    },
  };
  await rt.compactSince(marker, { strategy: custom });
  expect([rt.run.messages.at(-1)?.content, rt.run.modelCalls]).toEqual(["Facts", 1]);
  await expect(saved[0]("Late request")).rejects.toThrow(/only valid/);
  expect(rt.run.modelCalls).toBe(1);
});

for (const bad of [-1, 2.5, null]) {
  test(`invalid token counts leave context unchanged (${bad})`, async () => {
    const rt = runtime();
    const marker = rt.markMessages();
    rt.say("Original");
    const original = snapshot(rt);
    await expect(rt.compactSince(marker, { strategy: new Remove(), tokenCounter: async () => bad as number })).rejects.toThrow(/nonnegative integer/);
    expect(rt.run.messages).toEqual(original);
  });
}
