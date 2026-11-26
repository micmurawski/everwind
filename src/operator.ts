/** Asking the operator: the model asks with a tool, and a loop node waits durably for the reply. */

import { z } from "zod";
import type { Step } from "./flow.ts";
import type { Runtime } from "./runtime.ts";
import { type Tool, tool } from "./tools.ts";

/** The inbox kind of an operator's answer (agent.reply). */
export const REPLY = "reply";

export interface QuestionOptions {
  /** Name of the asking tool. Default "ask_operator". */
  name?: string;
  /** Seconds to wait for a reply before the model must decide itself; null (default) waits forever. */
  timeout?: number | null;
}

export interface Question {
  question: string;
  options: string[];
  callId: string;
}

const QuestionInput = z.object({
  question: z.string().min(1).describe("One clear question; include what you know and what depends on the answer"),
  options: z.array(z.string().min(1)).max(6).optional().describe("Suggested answers, if the choice is among a few"),
});

/**
 * The tool a model calls to ask the operator a question. It only records the
 * question; the loop's ask node (answerStep) waits for the reply.
 */
export function askOperatorTool(options: QuestionOptions = {}): Tool {
  return tool({
    name: options.name ?? "ask_operator",
    description:
      "Ask the operator (a person) a question when a decision depends on something you cannot find out or should not " +
      "decide alone. After this turn's tools run, you wait; the reply arrives as the next message. Ask one question at a time.",
    effect: "read",
    input: QuestionInput,
    run: () => ({ asked: true, note: "The operator's reply will arrive as the next message." }),
  });
}

/**
 * The question the last model turn asked, if any (the last successful call
 * wins). Read from the conversation, so a replay finds the same question.
 */
export function questionRequest(runtime: Runtime, options: QuestionOptions = {}): Question | null {
  const name = options.name ?? "ask_operator";
  const messages = runtime.run.messages;
  let index = messages.length - 1;
  while (index >= 0 && messages[index].role === "tool") index -= 1;
  const assistant = messages[index];
  if (assistant === undefined || assistant.role !== "assistant") return null;
  const failed = new Set(messages.slice(index + 1).filter((item) => item.isError).map((item) => item.toolCallId));
  const calls = assistant.toolCalls.filter((call) => call.name === name && !failed.has(call.id));
  if (!calls.length) return null;
  const call = calls[calls.length - 1];
  const args = QuestionInput.strict().safeParse(call.arguments);
  if (!args.success) return null;
  return { question: args.data.question, options: args.data.options ?? [], callId: call.id };
}

/**
 * A loop node to put after tool execution: if the model asked a question,
 * emit a `question` event, suspend until a reply message (or the timeout),
 * and add the reply as a pinned user message. Without a question it does
 * nothing. While it waits, the run keeps its lease and its timeout clock stops;
 * freezing it is cheap, since the loop checkpoints before this node.
 */
export function answerStep(options: QuestionOptions = {}): Step {
  const timeout = options.timeout ?? null;
  if (timeout !== null && !(timeout >= 0)) throw new Error("timeout must be nonnegative");
  return async function ask(runtime: Runtime): Promise<void> {
    const question = questionRequest(runtime, options);
    if (question === null) return;
    runtime.run.emit("question", { ...question });
    await runtime.suspend({ seconds: timeout ?? undefined, wakeOn: [REPLY], reason: "waiting for the operator" });
    const [reply] = await runtime.receive({ kinds: [REPLY], limit: 1 });
    const text = reply?.payload.text;
    if (typeof text === "string" && text) {
      runtime.run.emit("answered", { callId: question.callId, messageId: reply.id });
      runtime.say(`[operator] ${text}`, { pinned: true });
    } else {
      runtime.say(`No reply from the operator within ${timeout} seconds. Decide yourself, and say what you assumed.`);
    }
  };
}
