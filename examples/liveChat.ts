/** A live OpenAI or MiniMax run; install the matching SDK and set its API key. */

import { parseArgs } from "node:util";
import { z } from "zod";
import { Agent, FileSessionStore, MiniMax, OpenAIChat, type Provider, ReAct, Tools, tool } from "../src/index.ts";

const multiply = tool({
  name: "multiply",
  description: "Multiply two integers.",
  effect: "read",
  input: z.object({ a: z.number().int(), b: z.number().int() }),
  run: ({ a, b }) => a * b,
});

export interface ChatArgs {
  provider: "openai" | "minimax";
  model: string;
  baseUrl?: string;
  logDir?: string;
  events?: string;
  sessionDir: string;
  sessionId?: string;
  prompt: string;
}

const quote = (value: string) => (/^[\w./-]+$/.test(value) ? value : `'${value.replaceAll("'", "'\\''")}'`);

export async function main(args: ChatArgs, makeProvider?: (args: ChatArgs) => Provider & { close?(): Promise<void> }): Promise<number> {
  const provider =
    makeProvider?.(args) ??
    (args.provider === "minimax" ? new MiniMax(args.model, { baseUrl: args.baseUrl }) : new OpenAIChat(args.model, { baseUrl: args.baseUrl }));
  try {
    const run = await new Agent({
      provider,
      tools: new Tools([multiply]),
      loop: new ReAct(),
      logDir: args.logDir,
      sessions: new FileSessionStore(args.sessionDir),
    }).run(args.prompt, { sessionId: args.sessionId });
    console.log(run.status === "completed" ? run.output : `${run.status}: ${run.error}`);
    if (run.logPath) console.log(`Session log: ${run.logPath}`);
    if (args.events) run.writeEvents(args.events);
    const command = ["bun", "examples/liveChat.ts", "--provider", args.provider, "--model", args.model, "--session-dir", args.sessionDir, "--session-id", run.sessionId];
    if (args.baseUrl) command.push("--base-url", args.baseUrl);
    console.log(`To continue: ${[...command, "Your next prompt"].map(quote).join(" ")}`);
    return run.status === "completed" ? 0 : 1;
  } finally {
    await provider.close?.();
  }
}

if (import.meta.main) {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      provider: { type: "string", default: "openai" },
      model: { type: "string" },
      "base-url": { type: "string" },
      events: { type: "string" },
      "log-dir": { type: "string" },
      "session-dir": { type: "string", default: "artifacts/sessions" },
      "session-id": { type: "string" },
    },
  });
  if (!values.model) throw new Error("--model is required");
  process.exitCode = await main({
    provider: values.provider === "minimax" ? "minimax" : "openai",
    model: values.model,
    baseUrl: values["base-url"],
    events: values.events,
    logDir: values["log-dir"],
    sessionDir: values["session-dir"] as string,
    sessionId: values["session-id"],
    prompt: positionals[0] ?? "Use multiply to compute 6 times 7.",
  });
}
