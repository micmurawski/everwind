/** Offline DSL example: a chef delegates, wakes on completion, then resumes the child. */

import { parseArgs } from "node:util";
import {
  AGENT_JOB_TOOLS,
  Agent,
  AgentJobs,
  END,
  Flow,
  Mailbox,
  type Message,
  type Run,
  type StreamChunk,
  agentTool,
  finishStep,
  modelStep,
  node,
  toolsStep,
} from "../src/index.ts";

class SousChef {
  async *createMessage(_system: string, messages: Message[]): AsyncGenerator<StreamChunk> {
    const requests = messages.filter((item) => item.role === "user").map((item) => item.content);
    let text = "Menu: pasta with tomato sauce.";
    if (requests.length > 1) text += ` Updated after '${requests.at(-1)}': use gluten-free pasta.`;
    yield { type: "text", text };
  }
}

/** A deterministic policy demonstrating delegation, not a live model. */
class HeadChef {
  async *createMessage(_system: string, messages: Message[], metadata?: { turn?: number }): AsyncGenerator<StreamChunk> {
    const last = messages[messages.length - 1];
    if (last.role === "tool") {
      const handle = JSON.parse(last.content);
      yield { type: "text", text: `Sous chef working: ${handle.jobId}. Listening for updates.` };
      return;
    }
    const event = last.content.startsWith("{") ? JSON.parse(last.content) : null;
    let input: Record<string, string>;
    if (event === null) input = { prompt: "Plan a pasta dinner." };
    else if (messages.filter((item) => item.role === "user" && item.content.includes('"AgentFinished"')).length === 1) {
      input = { prompt: "Make that dinner gluten-free.", sessionId: event.sessionId };
    } else {
      yield { type: "text", text: `Dinner plan ready: ${event.output}` };
      return;
    }
    yield { type: "tool_use", id: `delegate-${metadata?.turn}`, name: "ask_sous_chef", input };
  }
}

function childLoop(): Flow {
  const callLlm = node(modelStep, { name: "call_llm" });
  const tools = node(toolsStep, { name: "tools" });
  const finish = node(finishStep, { name: "finish" });
  callLlm.on("tools", tools).on("answer", finish);
  tools.then(callLlm);
  finish.then(END);
  return new Flow(callLlm);
}

function chefLoop(mailbox: Mailbox, report: (line: string) => void): Flow {
  const callLlm = node(modelStep, { name: "call_llm" });
  const tools = node(toolsStep, { name: "tools" });
  const inbox = node(
    async (runtime) => {
      const event = await mailbox.get();
      runtime.run.emit("mailbox_event", { event });
      report(`INBOX ${JSON.stringify(event)}`);
      runtime.run.state.finished = ((runtime.run.state.finished as number) ?? 0) + 1;
      runtime.say(JSON.stringify(event));
    },
    { name: "inbox" },
  );
  const respond = node(
    (runtime) => {
      const last = runtime.run.messages[runtime.run.messages.length - 1].content;
      report(`CHEF ${last}`);
      if (runtime.run.state.finished === 2) {
        runtime.finish(last);
        return "closed";
      }
      return "listen";
    },
    { name: "respond" },
  );
  callLlm.on("tools", tools).on("answer", respond);
  tools.then(callLlm);
  respond.on("listen", inbox).on("closed", END);
  inbox.then(callLlm);
  return new Flow(callLlm);
}

export async function demo({ report = console.log, logDir }: { report?: (line: string) => void; logDir?: string } = {}): Promise<Run> {
  const mailbox = new Mailbox();
  const child = new Agent({ name: "sous_chef", provider: new SousChef(), loop: childLoop(), logDir });
  const askSousChef = agentTool(child, { name: "ask_sous_chef", execution: "background" });
  const jobs = new AgentJobs(mailbox);
  try {
    const chef = new Agent({ name: "chef", provider: new HeadChef(), logDir, tools: AGENT_JOB_TOOLS.with(askSousChef), loop: chefLoop(mailbox, report) });
    return await chef.run("Plan dinner with your sous chef.", { context: { agentJobs: jobs } });
  } finally {
    await jobs.close();
  }
}

if (import.meta.main) {
  const result = await demo({ logDir: parseArgs({ options: { "log-dir": { type: "string" } } }).values["log-dir"] });
  if (result.status !== "completed") throw new Error(result.error ?? result.status);
}
