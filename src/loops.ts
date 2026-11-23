/** Loop recipes built from the same model and tool execution primitives. */

import { OlderMessages, Summarize, type TokenCounter } from "./contextOps.ts";
import { END, Flow, Node, node } from "./flow.ts";
import type { Runtime } from "./runtime.ts";

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

function reactNodes(toolConcurrency: number, beforeModel?: Node): Node {
  if (!Number.isInteger(toolConcurrency) || toolConcurrency < 1) throw new Error("Tool concurrency must be positive");
  const callLlm = node(modelStep, { name: "call_llm" });
  const finish = node(finishStep, { name: "finish" });
  const tools = node((runtime) => runtime.executeTools({ concurrency: toolConcurrency }), { name: "tools" });
  callLlm.on("tools", tools).on("answer", finish);
  const entry = beforeModel ?? callLlm;
  if (beforeModel !== undefined) beforeModel.then(callLlm);
  tools.then(entry);
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
