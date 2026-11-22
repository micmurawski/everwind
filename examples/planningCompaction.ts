/**
 * Offline plan → review → test assumptions → final plan → compact → ReAct.
 *
 * Run with --summarize to make a separate, budgeted summarization call instead
 * of retaining the final plan directly. No external services or effects needed.
 */

import { parseArgs } from "node:util";
import { z } from "zod";
import {
  Agent,
  END,
  Flow,
  type MessageMarker,
  type ModelResponseInput,
  Replace,
  ScriptedProvider,
  Summarize,
  Tools,
  finishStep,
  modelStep,
  node,
  tool,
  toolsStep,
} from "../src/index.ts";

const checkCapacity = tool({
  name: "check_capacity",
  description: "Check whether all items fit in one batch.",
  effect: "read",
  input: z.object({ items: z.number().int(), workers: z.number().int() }),
  run: ({ items, workers }) => ({ fits: items <= workers, items, workers }),
});

const processBatch = tool({
  name: "process_batch",
  description: "Simulate processing a batch of at most three items.",
  input: z.object({ items: z.array(z.string()) }),
  run: ({ items }) => {
    if (items.length > 3) throw new Error("A batch may contain at most three items");
    return { processed: items };
  },
});

const FINAL_PLAN =
  "Process four items in two batches: [A, B, C], then [D]. " +
  "check_capacity verified that four items do not fit with three workers. " +
  "No items have been processed yet.";

export function planningFlow({ summarize = false } = {}): Flow {
  const begin = node((runtime) => void (runtime.run.state.planningStart = runtime.markMessages()), { name: "begin" });
  const plan = node(
    async (runtime) => {
      runtime.say("Plan how to process the four items.");
      await runtime.complete({ allowTools: false, purpose: "plan" });
    },
    { name: "plan" },
  );
  const review = node(
    async (runtime) => {
      runtime.say("Review the plan and identify assumptions to test.");
      await runtime.complete({ allowTools: false, purpose: "review" });
      runtime.say("Test the capacity assumption with check_capacity, then explain the finding.");
    },
    { name: "review" },
  );
  const testAssumptions = node(async (runtime) => ((await runtime.complete({ purpose: "test_assumptions" })).toolCalls.length ? "tools" : "done"), {
    name: "test_assumptions",
  });
  const assumptionTools = node(toolsStep, { name: "assumption_tools" });
  const finalPlan = node(
    async (runtime) => {
      runtime.say("Finalize the plan, preserving evidence, uncertainties, and actions already taken.");
      runtime.run.state.finalPlan = (await runtime.complete({ allowTools: false, purpose: "final_plan" })).text;
    },
    { name: "final_plan" },
  );
  const compact = node(
    async (runtime) => {
      const strategy = summarize ? new Summarize() : new Replace(runtime.run.state.finalPlan as string);
      const marker = runtime.run.state.planningStart as MessageMarker;
      delete runtime.run.state.planningStart;
      const result = await runtime.compactSince(marker, { strategy });
      runtime.run.emit("planning_handoff", { applied: result.applied, strategy: result.strategy });
      runtime.say("Execute the final plan using process_batch, then report the result.");
    },
    { name: "compact" },
  );
  const callLlm = node(modelStep, { name: "call_llm" });
  const executeTools = node(toolsStep, { name: "execute_tools" });
  const respond = node(finishStep, { name: "respond" });

  begin.then(plan).then(review).then(testAssumptions);
  testAssumptions.on("tools", assumptionTools).on("done", finalPlan);
  assumptionTools.then(testAssumptions);
  finalPlan.then(compact).then(callLlm);
  callLlm.on("tools", executeTools).on("answer", respond);
  executeTools.then(callLlm);
  respond.then(END);
  return new Flow(begin);
}

export function makeAgent({ summarize = false } = {}): Agent {
  const responses: ModelResponseInput[] = [
    { text: "Draft: process A, B, C, and D in one batch." },
    { text: "The draft assumes four items fit with three workers. Check this." },
    { toolCalls: [{ id: "capacity", name: "check_capacity", arguments: { items: 4, workers: 3 } }] },
    { text: "Capacity check failed: four items need at least two batches." },
    { text: FINAL_PLAN },
  ];
  if (summarize) responses.push({ text: FINAL_PLAN });
  responses.push(
    { toolCalls: [{ id: "batch1", name: "process_batch", arguments: { items: ["A", "B", "C"] } }] },
    { toolCalls: [{ id: "batch2", name: "process_batch", arguments: { items: ["D"] } }] },
    { text: "Processed all four items in two batches." },
  );
  return new Agent({ provider: new ScriptedProvider(responses), tools: new Tools([checkCapacity, processBatch]), loop: planningFlow({ summarize }) });
}

export async function main(summarize = false): Promise<void> {
  const run = await makeAgent({ summarize }).run("Process A, B, C, D with three workers.");
  console.log(`${run.status}: ${run.output}`);
  console.log(`Model calls: ${run.modelCalls}; tool calls: ${run.toolCalls}`);
  console.log("\nConversation retained for continuation:");
  for (const item of run.messages) console.log(`  ${item.role}: ${item.content}`);
  const event = run.events.find((item) => item.kind === "context_compaction");
  console.log(`\nFull trace retains ${(event?.data.removedMessages as unknown[]).length} compacted planning messages.`);
}

if (import.meta.main) await main(parseArgs({ options: { summarize: { type: "boolean", default: false } } }).values.summarize);
