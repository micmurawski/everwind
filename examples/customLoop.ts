/** Insert a review step before accepting an answer, with one revision at most. */

import { Agent, END, Flow, ScriptedProvider, finishStep, modelStep, node, toolsStep } from "../src/index.ts";

const review = node(
  (runtime) => {
    if (runtime.run.state.reviewed) return "accept";
    runtime.run.state.reviewed = true;
    runtime.say("Check your proposed answer for errors. Return a corrected final answer if necessary.");
    return "revise";
  },
  { name: "review", metadata: { description: "Review the first proposed answer once" } },
);
const callLlm = node(modelStep, { name: "call_llm" });
const tools = node(toolsStep, { name: "tools" });
const finish = node(finishStep, { name: "finish" });

callLlm.on("tools", tools).on("answer", review);
tools.then(callLlm);
review.on("revise", callLlm).on("accept", finish);
finish.then(END);

export const loop = new Flow(callLlm);

export async function main(): Promise<void> {
  const agent = new Agent({ loop, provider: new ScriptedProvider([{ text: "The result is 41." }, { text: "The corrected result is 42." }]) });
  const run = await agent.run("What is 6 times 7?");
  if (run.status !== "completed") throw new Error(run.error ?? run.status);
  console.log(run.output);
  console.log(run.events.filter((event) => event.kind === "step").map((event) => event.data.node).join(" -> "));
}

if (import.meta.main) await main();
