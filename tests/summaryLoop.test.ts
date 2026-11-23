import { expect, test } from "bun:test";
import { z } from "zod";
import {
  Agent,
  type Message,
  type Provider,
  ReActWithSummary,
  Run,
  Runtime,
  SUMMARY_PROMPT,
  ScriptedProvider,
  Summarize,
  OlderMessages,
  Tools,
  limits,
  message,
  tool,
  validateConversation,
} from "../src/index.ts";
import { call, countTokens } from "./helpers.ts";

const makeRuntime = (messages: Message[], provider: Provider = new ScriptedProvider([{ text: "Summary" }])) =>
  new Runtime(new Run({ messages }), provider, new Tools(), limits());

const conversation = (): Message[] => [
  message({ role: "system", content: "System" }),
  message({ role: "user", content: "Original task" }),
  message({ role: "assistant", content: "Old investigation. ".repeat(30) }),
  message({ role: "user", content: "Old follow-up. ".repeat(30) }),
  message({ role: "assistant", content: "Recent reasoning" }),
  message({ role: "user", content: "Recent request" }),
  message({ role: "assistant", content: "Recent answer" }),
];

const requests = (rt: { run: Run }) => rt.run.events.filter((event) => event.kind === "model_request").map((event) => event.data.request as any);
const transcript = (request: any) => JSON.parse(request.messages[1].content.split("\n").slice(1).join("\n"));

test("the summary window keeps the first and recent messages and rolls forward", async () => {
  const rt = makeRuntime(conversation(), new ScriptedProvider([{ text: "First summary" }, { text: "Second summary" }]));
  const original = [...rt.run.messages];
  const first = await rt.compact({ selection: new OlderMessages(), strategy: new Summarize(), tokenCounter: countTokens });
  expect(first.applied).toBe(true);
  expect(first.afterTokens as number).toBeLessThan(first.beforeTokens as number);
  expect(rt.run.messages.slice(0, 2)).toEqual(original.slice(0, 2));
  expect(rt.run.messages.slice(3)).toEqual(original.slice(-3));
  expect(rt.run.messages[2].content).toBe("First summary");
  // Once enough new history accumulates, the preceding summary is input to the next one.
  rt.run.messages.push(...Array.from({ length: 3 }, () => message({ role: "user", content: "More work. ".repeat(30) })));
  const second = await rt.compact({ selection: new OlderMessages(), strategy: new Summarize(), tokenCounter: countTokens });
  expect(second.applied).toBe(true);
  expect(rt.run.messages[2].content).toBe("Second summary");
  const sent = requests(rt);
  expect(sent[0].messages[0].content).toBe(SUMMARY_PROMPT);
  const second_transcript = transcript(sent[1]);
  expect(second_transcript[0].content).toBe("First summary");
  expect(second_transcript.some((item: Message) => item.content.includes("Old investigation"))).toBe(false);
  validateConversation(rt.run.messages);
});

for (const metadata of [{ compaction: { kind: "summary" } }, { isSummary: true }]) {
  test(`the latest summary supersedes earlier messages (${JSON.stringify(metadata)})`, async () => {
    const messages = conversation();
    messages.splice(3, 1, message({ role: "assistant", content: "Last summary", metadata }), message({ role: "user", content: "New findings. ".repeat(30) }));
    const rt = makeRuntime(messages);
    const result = await rt.compact({ selection: new OlderMessages(), strategy: new Summarize({ prompt: "Custom instructions" }), tokenCounter: countTokens });
    expect(result.applied).toBe(true);
    const request = requests(rt)[0];
    expect(request.messages[0].content).toBe("Custom instructions");
    expect(transcript(request).map((item: Message) => item.content)).toEqual(["Last summary", "New findings. ".repeat(30)]);
    expect(rt.run.messages.length).toBe(6);
  });
}

test("the summary window expands the tail for parallel tool results", async () => {
  const messages = [
    ...conversation().slice(0, 4),
    message({ role: "assistant", toolCalls: [{ id: "a", name: "check" }, { id: "b", name: "check" }] }),
    message({ role: "tool", toolCallId: "b", content: "B" }),
    message({ role: "tool", toolCallId: "a", content: "A" }),
    message({ role: "assistant", content: "Latest answer" }),
  ];
  const rt = makeRuntime(messages);
  const result = await rt.compact({ selection: new OlderMessages(), strategy: new Summarize(), tokenCounter: countTokens });
  expect([result.applied, result.stop]).toEqual([true, 4]);
  expect(rt.run.messages.slice(3)).toEqual(messages.slice(-4));
  validateConversation(rt.run.messages);
});

for (let size = 0; size < 7; size++) {
  test(`a short history (${size}) is not summarized`, async () => {
    const rt = makeRuntime(conversation().slice(0, size));
    const original = [...rt.run.messages];
    const result = await rt.compact({ selection: new OlderMessages(), strategy: new Summarize(), tokenCounter: countTokens });
    expect([result.applied, result.reason]).toEqual([false, "not_enough_messages"]);
    expect(rt.run.messages).toEqual(original);
    expect(rt.run.modelCalls).toBe(0);
  });
}

test("a recent summary or a non-shrinking summary leaves history unchanged", async () => {
  let rt = makeRuntime(conversation());
  rt.run.messages[rt.run.messages.length - 2].metadata.compaction = { kind: "summary" };
  let original = [...rt.run.messages];
  const skipped = await rt.compact({ selection: new OlderMessages(), strategy: new Summarize(), tokenCounter: countTokens });
  expect([skipped.reason, rt.run.modelCalls]).toEqual(["recent_summary", 0]);
  expect(rt.run.messages).toEqual(original);
  rt = makeRuntime(conversation(), new ScriptedProvider([{ text: "Huge summary".repeat(500) }]));
  original = [...rt.run.messages];
  const rejected = await rt.compact({ selection: new OlderMessages(), strategy: new Summarize(), tokenCounter: countTokens });
  expect([rejected.reason, rt.run.modelCalls]).toEqual(["not_smaller", 1]);
  expect(rt.run.messages).toEqual(original);
});

const lookup = tool({
  name: "lookup",
  description: "Return verbose evidence for the selected key.",
  input: z.object({ key: z.string() }),
  run: ({ key }) => key.repeat(300),
});

test("ReAct summarizes between complete tool batches and checkpoints", async () => {
  const agent = new Agent({
    systemPrompt: "System",
    provider: new ScriptedProvider([
      { toolCalls: [call("a", "lookup", { key: "a" })] },
      { toolCalls: [call("b", "lookup", { key: "b" })] },
      { toolCalls: [call("c", "lookup", { key: "c" })] },
      { text: "Lookup a verified." },
      { text: "Done" },
    ]),
    tools: new Tools([lookup]),
    loop: new ReActWithSummary({
      contextWindow: 1000,
      maxOutputTokens: 100,
      thresholdPercent: 50,
      tokenCounter: countTokens,
      summary: new Summarize({ prompt: "Keep evidence" }),
    }),
  });
  const run = await agent.run("Investigate");
  expect([run.status, run.output, run.modelCalls, run.toolCalls]).toEqual(["completed", "Done", 5, 3]);
  const sent = requests({ run });
  expect(sent.map((request: any) => request.purpose)).toEqual(["act", "act", "act", "compact", "act"]);
  expect(sent[3].messages[0].content).toBe("Keep evidence");
  expect(sent[4].messages.some((item: Message) => item.content === "Lookup a verified.")).toBe(true);
  expect(sent[4].messages.some((item: Message) => item.toolCallId === "a")).toBe(false);
  expect(run.events.some((event) => event.kind === "tool_result" && (event.data.message as any).toolCallId === "a")).toBe(true);
  validateConversation(run.messages);
  expect(agent.sessions.load(run.sessionId)?.messages).toEqual(run.messages.slice(1));
  const flow = (agent.loop as ReActWithSummary).toFlow();
  expect([flow.start, flow.edges.tools.default, flow.edges.call_llm.answer]).toEqual(["summarize", "summarize", "finish"]);
});

for (const [threshold, reserve, expected] of [
  [100, 100, 1],
  [50, 100, 2],
  [100, 400, 2],
]) {
  test(`a summary triggers on the percentage or the output reserve (${threshold}, ${reserve})`, async () => {
    const rt = makeRuntime(conversation(), new ScriptedProvider([{ text: "Small" }, { text: "Done" }]));
    // 650 measured units crosses 50%, and also the 500-unit allowed input with a
    // 400-unit output reserve; it does not cross the 800-unit limit.
    const counter = async (messages: readonly Message[]) => (messages.length === 7 ? 650 : 100);
    await new ReActWithSummary({ contextWindow: 1000, maxOutputTokens: reserve, thresholdPercent: threshold, tokenCounter: counter }).run(rt);
    expect([rt.run.modelCalls, rt.run.status]).toEqual([expected, "completed"]);
  });
}

test("the summary model call obeys the shared run limit", async () => {
  const agent = new Agent({
    provider: new ScriptedProvider([{ text: "Small" }]),
    limits: { maxModelCalls: 1 },
    loop: new ReActWithSummary({ contextWindow: 1000, maxOutputTokens: 100, tokenCounter: countTokens, thresholdPercent: 50 }),
  });
  const run = await agent.run("Continue", { history: conversation().slice(1) });
  expect([run.status, run.modelCalls]).toEqual(["limit", 1]);
  expect(run.events.some((event) => event.kind === "context_compaction")).toBe(true);
});

for (const options of [
  { contextWindow: 0 },
  { maxOutputTokens: 900 },
  { maxOutputTokens: -1 },
  { keepRecentMessages: 0 },
  { thresholdPercent: 4 },
  { thresholdPercent: 101 },
  { thresholdPercent: Number.NaN },
  { tokenCounter: null },
  { toolConcurrency: 0 },
]) {
  test(`the summary loop validates its configuration (${JSON.stringify(options)})`, () => {
    expect(() => new ReActWithSummary({ contextWindow: 1000, maxOutputTokens: 100, tokenCounter: countTokens, ...(options as object) })).toThrow();
  });
}
