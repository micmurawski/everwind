/** Offline ReAct with rolling summaries; no external service or API key needed. */

import { z } from "zod";
import { Agent, type Message, ReActWithSummary, ScriptedProvider, Summarize, Tools, tool } from "../src/index.ts";

const inspectComponent = tool({
  name: "inspect_component",
  description: "Return a verbose, simulated diagnostic report.",
  effect: "read",
  input: z.object({ component: z.string() }),
  run: ({ component }) => `${component}: healthy. `.repeat(30),
});

/** Character-based units make the offline trigger deterministic. Use your model's token counter for real runs. */
async function demonstrationCounter(messages: readonly Message[]): Promise<number> {
  return messages.reduce((total, item) => total + item.content.length + 1, 0);
}

export function makeAgent(): Agent {
  const inspect = (id: string, component: string) => ({ toolCalls: [{ id, name: "inspect_component", arguments: { component } }] });
  return new Agent({
    systemPrompt: "Inspect components and report the evidence.",
    tools: new Tools([inspectComponent]),
    provider: new ScriptedProvider([
      inspect("a", "A"),
      inspect("b", "B"),
      inspect("c", "C"),
      { text: "Component A was inspected and is healthy. No changes were made." },
      { text: "A, B, and C are healthy. All three inspections completed; no changes were made." },
    ]),
    loop: new ReActWithSummary({
      contextWindow: 2000,
      maxOutputTokens: 200,
      thresholdPercent: 40,
      tokenCounter: demonstrationCounter,
      summary: new Summarize({ prompt: "Preserve inspected components, findings, and any actions already taken." }),
    }),
  });
}

export async function main(): Promise<void> {
  const agent = makeAgent();
  const run = await agent.run("Inspect A, B, and C, then report their health.");
  if (run.status !== "completed") throw new Error(run.error ?? run.status);
  console.log(run.output);
  for (const event of run.events.filter((item) => item.kind === "context_compaction")) {
    console.log(`Summary reduced context from ${event.data.beforeTokens} to ${event.data.afterTokens} test units.`);
  }
  console.log(`Model calls: ${run.modelCalls}; tool calls: ${run.toolCalls}`);
  console.log(agent.flowGraph());
}

if (import.meta.main) await main();
