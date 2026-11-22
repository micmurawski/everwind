/** Run two loop recipes without credentials: bun examples/compareLoops.ts */

import { z } from "zod";
import { Agent, END, Flow, ScriptedProvider, Tools, finishStep, modelStep, node, tool, toolsStep } from "../src/index.ts";

const multiply = tool({
  name: "multiply",
  description: "Multiply two integers.",
  effect: "read",
  input: z.object({ a: z.number().int(), b: z.number().int() }),
  run: ({ a, b }) => a * b,
});

export async function main(): Promise<void> {
  const action = { toolCalls: [{ id: "multiply-1", name: "multiply", arguments: { a: 6, b: 7 } }] };
  const answer = { text: "42" };
  for (const [label, planning, script] of [
    ["ReAct", false, [action, answer]],
    ["PlanThenAct", true, [{ text: "Multiply 6 by 7, then report the result." }, action, answer]],
  ] as const) {
    // Fresh graph handles per variant; Flow snapshots their wiring.
    const callLlm = node(modelStep, { name: "call_llm" });
    const tools = node(toolsStep, { name: "tools" });
    const finish = node(finishStep, { name: "finish" });
    callLlm.on("tools", tools).on("answer", finish);
    tools.then(callLlm);
    finish.then(END);
    const plan = node(
      async (runtime) => {
        runtime.say("Write a concise action plan. Do not call tools yet.");
        const response = await runtime.complete({ allowTools: false, purpose: "plan" });
        runtime.run.state.plan = response.text;
        runtime.say("Execute the plan, then give the result.");
      },
      { name: "plan" },
    );
    if (planning) plan.then(callLlm);
    const agent = new Agent({ provider: new ScriptedProvider([...script]), tools: new Tools([multiply]), loop: new Flow(planning ? plan : callLlm) });
    const run = await agent.run("What is 6 times 7?");
    if (run.status !== "completed") throw new Error(run.error ?? run.status);
    console.log(`${label.padEnd(12)} answer=${run.output} modelCalls=${run.modelCalls} toolCalls=${run.toolCalls}`);
  }
}

if (import.meta.main) await main();
