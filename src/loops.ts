/** Loop recipes built from the same model and tool execution primitives. */

import { OlderMessages, Summarize, type TokenCounter } from "./contextOps.ts";
import { END, Flow, Node, node } from "./flow.ts";
import type { Runtime } from "./runtime.ts";
import { type GateOptions, gateStep } from "./inbox.ts";
import { type QuestionOptions, answerStep, questionRequest } from "./operator.ts";
import { readTaskScope } from "./scope.ts";
import { settleStep, waitRequest, waitStep } from "./tasks.ts";
import { type WakeupOptions, wakeupRequest, wakeupStep } from "./wakeup.ts";

export interface Loop {
  run(runtime: Runtime): Promise<void>;
}

export async function modelStep(runtime: Runtime): Promise<string> {
  const response = await runtime.complete();
  return response.toolCalls.length ? "tools" : "answer";
}

export async function toolsStep(runtime: Runtime): Promise<void> {
  await runtime.executeTools();
}

export function finishStep(runtime: Runtime): void {
  runtime.finish(runtime.run.messages[runtime.run.messages.length - 1].content);
}

/**
 * afterTools: the first and last node of a chain run between tools and the next model turn.
 * beforeFinish: a node between an answer and finish; its "done" finishes, its "continue" goes back to the entry.
 */
function reactNodes(toolConcurrency: number, beforeModel?: Node, afterTools?: [Node, Node], beforeFinish?: Node): Node {
  if (!Number.isInteger(toolConcurrency) || toolConcurrency < 1) throw new Error("Tool concurrency must be positive");
  const callLlm = node(modelStep, { name: "call_llm" });
  const finish = node(finishStep, { name: "finish" });
  const tools = node((runtime) => runtime.executeTools({ concurrency: toolConcurrency }), { name: "tools" });
  callLlm.on("tools", tools).on("answer", beforeFinish ?? finish);
  const entry = beforeModel ?? callLlm;
  if (beforeModel !== undefined) beforeModel.then(callLlm);
  if (afterTools === undefined) tools.then(entry);
  else {
    tools.then(afterTools[0]);
    afterTools[1].then(entry);
  }
  if (beforeFinish !== undefined) beforeFinish.on("done", finish).on("continue", entry);
  finish.then(END);
  return entry;
}

export function reactFlow({ toolConcurrency = 1 }: { toolConcurrency?: number } = {}): Flow {
  return new Flow(reactNodes(toolConcurrency));
}

export class ReAct implements Loop {
  readonly toolConcurrency: number;

  constructor({ toolConcurrency = 1 }: { toolConcurrency?: number } = {}) {
    this.toolConcurrency = toolConcurrency;
    reactNodes(toolConcurrency);
  }

  toFlow(): Flow {
    return reactFlow({ toolConcurrency: this.toolConcurrency });
  }

  run(runtime: Runtime): Promise<void> {
    return this.toFlow().run(runtime);
  }
}

export interface ReActWithSummaryOptions {
  contextWindow: number;
  tokenCounter: TokenCounter;
  summary?: Summarize;
  thresholdPercent?: number;
  maxOutputTokens?: number;
  keepRecentMessages?: number;
  toolConcurrency?: number;
}

/**
 * ReAct that summarizes older messages (OlderMessages) before a model turn once a token threshold is hit.
 *
 * Supply a counter for the execution model, including request overhead as
 * needed. Summarize at thresholdPercent, or within the 10% context buffer plus
 * the output reserve. A summary is applied only when the full context shrinks.
 * Skips leave history intact; errors propagate, with no removal fallback.
 */
export class ReActWithSummary implements Loop {
  readonly contextWindow: number;
  readonly tokenCounter: TokenCounter;
  readonly summary: Summarize;
  readonly thresholdPercent: number;
  readonly maxOutputTokens: number;
  readonly keepRecentMessages: number;
  readonly toolConcurrency: number;

  constructor(options: ReActWithSummaryOptions) {
    this.contextWindow = options.contextWindow;
    this.tokenCounter = options.tokenCounter;
    this.summary = options.summary ?? new Summarize();
    this.thresholdPercent = options.thresholdPercent ?? 80;
    this.maxOutputTokens = options.maxOutputTokens ?? 8192;
    this.keepRecentMessages = options.keepRecentMessages ?? 3;
    this.toolConcurrency = options.toolConcurrency ?? 1;
    for (const name of ["contextWindow", "maxOutputTokens", "toolConcurrency"] as const) {
      if (!Number.isInteger(this[name]) || this[name] < 1) throw new Error(`${name} must be a positive integer`);
    }
    if (typeof this.thresholdPercent !== "number" || !(this.thresholdPercent >= 5 && this.thresholdPercent <= 100)) {
      throw new Error("thresholdPercent must be between 5 and 100");
    }
    if (this.maxOutputTokens >= this.contextWindow * 0.9) {
      throw new Error("Output reserve and 10% buffer must leave room for conversation");
    }
    if (typeof this.tokenCounter !== "function") throw new TypeError("tokenCounter must be an async function");
    new OlderMessages(this.keepRecentMessages);
  }

  toFlow(): Flow {
    const summarize = node(
      async (runtime) => {
        const tokens = await this.tokenCounter(structuredClone(runtime.run.messages));
        if (!Number.isInteger(tokens) || tokens < 0) throw new Error("Token counter must return a nonnegative integer");
        const allowed = this.contextWindow * 0.9 - this.maxOutputTokens;
        if (tokens * 100 >= this.contextWindow * this.thresholdPercent || tokens > allowed) {
          await runtime.compact({
            selection: new OlderMessages(this.keepRecentMessages),
            strategy: this.summary,
            tokenCounter: this.tokenCounter,
          });
        }
      },
      { name: "summarize" },
    );
    return new Flow(reactNodes(this.toolConcurrency, summarize));
  }

  run(runtime: Runtime): Promise<void> {
    return this.toFlow().run(runtime);
  }
}

export class PlanThenAct implements Loop {
  readonly planningPrompt: string;
  readonly executionPrompt: string;
  readonly toolConcurrency: number;

  constructor(options: { planningPrompt?: string; executionPrompt?: string; toolConcurrency?: number } = {}) {
    this.planningPrompt = options.planningPrompt ?? "Write a concise action plan for the task. Do not call tools yet.";
    this.executionPrompt = options.executionPrompt ?? "Execute the plan using the available tools, then give the result.";
    this.toolConcurrency = options.toolConcurrency ?? 1;
  }

  toFlow(): Flow {
    const plan = node(
      async (runtime) => {
        runtime.say(this.planningPrompt);
        const response = await runtime.complete({ allowTools: false, purpose: "plan" });
        runtime.run.state.plan = response.text;
        runtime.say(this.executionPrompt);
      },
      { name: "plan" },
    );
    plan.then(reactNodes(this.toolConcurrency));
    return new Flow(plan);
  }

  run(runtime: Runtime): Promise<void> {
    return this.toFlow().run(runtime);
  }
}

/**
 * ReAct whose model can pace itself, ask you questions, and be messaged while it works.
 *
 * gate -> call_llm -> tools -> ask -> wakeup -> gate, or call_llm -> finish on an answer.
 * The `gate` delivers inbox messages (agent.send) before each model turn. The
 * `ask` node waits for your reply (agent.reply) when the model called
 * ask_operator (see askOperatorTool). The `wakeup` node suspends for as long as
 * the model asked with schedule_wakeup (see scheduleWakeupTool), then gives it
 * the wakeup prompt; a message ends the sleep early. Waiting is durable and does
 * not count toward limits.timeout; give the agent enough maxSteps and
 * maxModelCalls for its wakeups.
 */
export class ReActWithWakeups implements Loop {
  readonly toolConcurrency: number;
  readonly wakeup: WakeupOptions;
  readonly gate: GateOptions;
  readonly ask: QuestionOptions;

  constructor(options: WakeupOptions & { toolConcurrency?: number; gate?: GateOptions; ask?: QuestionOptions } = {}) {
    const { toolConcurrency = 1, gate = {}, ask = {}, ...wakeup } = options;
    this.toolConcurrency = toolConcurrency;
    this.wakeup = wakeup;
    this.gate = gate;
    this.ask = ask;
    this.toFlow();
  }

  /**
   * ask -> wakeup -> wait, each checkpointed before it waits, so a run recovered (or frozen) while waiting
   * restarts there. The checkpoint decisions read only the conversation, so replays make the same ones.
   */
  private waits(): [Node, Node] {
    const ask = node(answerStep(this.ask), { name: "ask", checkpoint: (runtime) => questionRequest(runtime, this.ask) !== null });
    const wakeup = node(wakeupStep(this.wakeup), { name: "wakeup", checkpoint: (runtime) => wakeupRequest(runtime, this.wakeup) !== null });
    const wait = node(waitStep(), { name: "wait", checkpoint: (runtime) => waitRequest(runtime) !== null });
    ask.then(wakeup).then(wait);
    return [ask, wait];
  }

  toFlow(): Flow {
    return new Flow(
      reactNodes(
        this.toolConcurrency,
        node(gateStep(this.gate), { name: "gate" }),
        this.waits(),
        // A task's answer finishes it only once its children are done; a checkpoint before, as settling can wait long.
        node(settleStep(), { name: "settle", checkpoint: (runtime) => readTaskScope(runtime.run.scope) !== null }),
      ),
    );
  }

  run(runtime: Runtime): Promise<void> {
    return this.toFlow().run(runtime);
  }
}
