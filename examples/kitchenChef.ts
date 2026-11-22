/**
 * Offline event-driven ReAct demo: a chef, a mailbox, and real background jobs.
 *
 * Run with: bun examples/kitchenChef.ts
 * The provider is a deterministic demo policy, not an LLM. Cooking jobs are tiny
 * Bun subprocesses, so cancel and kill really terminate work. No shell is involved.
 */

import type { Subprocess } from "bun";
import { constants } from "node:os";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { z } from "zod";
import {
  Agent,
  ApiHandler,
  END,
  Flow,
  Mailbox,
  type Message,
  type ModelResponse,
  type Run,
  type StreamChunk,
  Tools,
  type CreateMessageMetadata,
  modelResponse,
  modelStep,
  node,
  responseChunks,
  tool,
  toolsStep,
} from "../src/index.ts";

export const ORDER = "Please cook pasta, sauce, and toast.";
export const STATUS = "How is dinner coming along?";
export const STOP = "Kill the toast job; we don't need toast anymore.";
const COOK = fileURLToPath(new URL("./cook.ts", import.meta.url));

type Event = Record<string, any>;

class Job {
  status = "running";
  requestedStop: string | null = null;
  output: string[] = [];
  watcher: Promise<void> | null = null;

  constructor(
    readonly id: string,
    readonly dish: string,
    readonly process: Subprocess<"ignore", "pipe", "inherit">,
  ) {}

  get exitCode(): number | null {
    if (this.process.exitCode !== null) return this.process.exitCode;
    const signal = this.process.signalCode;
    return signal ? -(constants.signals[signal as keyof typeof constants.signals] ?? 1) : null;
  }

  snapshot(): Event {
    return { jobId: this.id, dish: this.dish, status: this.status, requestedStop: this.requestedStop, exitCode: this.exitCode, output: [...this.output] };
  }
}

/** Owns processes; its single watcher per job publishes terminal events once. */
export class Kitchen {
  readonly jobs = new Map<string, Job>();
  readonly started = new Mailbox<string>();
  private closed = false;

  constructor(readonly mailbox: Mailbox) {}

  start(dish: string, seconds: number, { fail = false } = {}): Event {
    if (this.closed) throw new Error("Kitchen is closed");
    if (!Number.isFinite(seconds) || !(seconds > 0 && seconds <= 60)) throw new Error("Cooking duration must be between 0 and 60 seconds");
    const child = Bun.spawn([process.execPath, COOK, dish, String(seconds), String(fail)], { stdout: "pipe", stderr: "inherit" });
    const job = new Job(`job-${this.jobs.size + 1}`, dish, child);
    this.jobs.set(job.id, job);
    job.watcher = this.watch(job);
    this.started.put(job.id);
    return job.snapshot(); // Return a handle immediately, not the cooked dish.
  }

  private async watch(job: Job): Promise<void> {
    const decoder = new TextDecoder();
    let buffer = "";
    for await (const chunk of job.process.stdout) {
      buffer += decoder.decode(chunk, { stream: true });
      let index: number;
      while ((index = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, index);
        buffer = buffer.slice(index + 1);
        job.output = [...job.output, line].slice(-20);
        this.mailbox.put({ kind: "JobProgress", jobId: job.id, text: line });
      }
    }
    const code = await job.process.exited;
    job.status = code === 0 && job.process.signalCode === null ? "succeeded" : (job.requestedStop ?? "failed");
    // Success, failure, cancellation and killing all wake the chef.
    this.mailbox.put({ kind: "JobFinished", ...job.snapshot() });
  }

  inspect(jobId: string): Event {
    const job = this.jobs.get(jobId);
    if (job === undefined) throw new Error(`Unknown job ${jobId}`);
    return job.snapshot();
  }

  listJobs(): Event[] {
    return [...this.jobs.values()].map((job) => job.snapshot());
  }

  stop(jobId: string, { force }: { force: boolean }): Event {
    const job = this.jobs.get(jobId);
    if (job === undefined) throw new Error(`Unknown job ${jobId}`);
    if (job.process.exitCode === null && job.process.signalCode === null && job.requestedStop === null) {
      job.process.kill(force ? constants.signals.SIGKILL : constants.signals.SIGTERM); // Numbers: older Bun ignores names.
      job.requestedStop = force ? "killed" : "cancelled";
      job.status = "stopping";
    }
    return job.snapshot(); // An acknowledgement; the watcher confirms the exit.
  }

  async waitAll(): Promise<void> {
    await Promise.all([...this.jobs.values()].map((job) => job.watcher));
  }

  async close(): Promise<void> {
    this.closed = true;
    for (const job of this.jobs.values()) if (job.process.exitCode === null && job.process.signalCode === null) this.stop(job.id, { force: true });
    await this.waitAll();
  }
}

const kitchenOf = (ctx: { get<T>(key: string): T }) => ctx.get<Kitchen>("kitchen");
const jobId = z.object({ jobId: z.string() });

export const KITCHEN_TOOLS = new Tools([
  tool({
    name: "start_job",
    description: "Start cooking in the background; return a job handle immediately.",
    input: z.object({ dish: z.string(), seconds: z.number() }),
    run: ({ dish, seconds }, ctx) => ({ action: "start", ...kitchenOf(ctx).start(dish, seconds) }),
  }),
  tool({ name: "list_jobs", description: "List cooking jobs, their statuses, and recent output.", effect: "read", run: (_args, ctx) => ({ action: "list", jobs: kitchenOf(ctx).listJobs() }) }),
  tool({ name: "inspect_job", description: "Inspect one cooking job and its latest output.", effect: "read", input: jobId, run: ({ jobId }, ctx) => ({ action: "inspect", ...kitchenOf(ctx).inspect(jobId) }) }),
  tool({ name: "cancel_job", description: "Request graceful termination; JobFinished will confirm the outcome.", input: jobId, run: ({ jobId }, ctx) => ({ action: "cancel", ...kitchenOf(ctx).stop(jobId, { force: false }) }) }),
  tool({ name: "kill_job", description: "Force termination; JobFinished will confirm the outcome.", input: jobId, run: ({ jobId }, ctx) => ({ action: "kill", ...kitchenOf(ctx).stop(jobId, { force: true }) }) }),
  tool({
    name: "plate_dish",
    description: "Plate a dish after its cooking job has succeeded.",
    input: jobId,
    run: ({ jobId }, ctx) => {
      const job = kitchenOf(ctx).inspect(jobId);
      if (job.status !== "succeeded") throw new Error("Only successfully cooked dishes can be plated");
      return { action: "plate", jobId, dish: job.dish };
    },
  }),
]);

/** A deterministic policy driven by observations, not a turn-number script. */
class DemoChef extends ApiHandler {
  async *createMessage(_system: string, messages: Message[], metadata?: CreateMessageMetadata): AsyncGenerator<StreamChunk> {
    yield* responseChunks(this.respond(messages, metadata?.turn ?? 1));
  }

  private respond(messages: Message[], turn: number): ModelResponse {
    const event = JSON.parse(
      [...messages].reverse().find((item) => item.role === "user" && item.content.startsWith('{"kind":'))?.content as string,
    ) as Event;
    const calls = (...actions: [string, Record<string, unknown>][]) =>
      modelResponse({ toolCalls: actions.map(([name, args], index) => ({ id: `turn-${turn}-${index}`, name, arguments: args as never })) });
    const say = (text: string) => modelResponse({ text });
    const last = messages[messages.length - 1];
    if (last.role === "tool") {
      if (last.isError) return say(`Kitchen problem: ${last.content}`);
      const result = JSON.parse(last.content);
      switch (result.action) {
        case "list": {
          if (event.text === STOP) return calls(["kill_job", { jobId: result.jobs.find((job: Event) => job.dish === "toast").jobId }]);
          return say(`Dinner status: ${result.jobs.map((job: Event) => `${job.dish}=${job.status}`).join(", ")}.`);
        }
        case "start":
          return say("Cooking jobs started. I can take questions while they cook.");
        case "kill":
        case "cancel":
          return say(`Stop requested for ${result.dish}; waiting for confirmation.`);
        case "plate":
          return say(`Plated ${result.dish} after its completion event woke me up.`);
        default:
          return say(JSON.stringify(result));
      }
    }
    if (event.kind === "UserMessage") {
      if (event.text === ORDER) {
        return calls(["start_job", { dish: "pasta", seconds: 0.5 }], ["start_job", { dish: "sauce", seconds: 0.8 }], ["start_job", { dish: "toast", seconds: 30 }]);
      }
      return calls(["list_jobs", {}]);
    }
    if (event.kind === "JobFinished") {
      if (event.status === "succeeded") return calls(["plate_dish", { jobId: event.jobId }]);
      return say(`Observed ${event.dish} ${event.status} (exit ${event.exitCode}).`);
    }
    return say(`Progress noted: ${event.text}.`);
  }
}

export function chefLoop(mailbox: Mailbox, kitchen: Kitchen, report: (line: string) => void): Flow {
  const inbox = node(
    async (runtime) => {
      const event = await mailbox.get(); // Sleeps without polling; queued events wake it.
      runtime.run.emit("mailbox_event", { event });
      if (event.kind === "Shutdown") {
        runtime.run.state.jobs = kitchen.listJobs();
        runtime.finish("Kitchen service closed; all job outcomes were observed.");
        return "closed";
      }
      report(`INBOX ${JSON.stringify(event)}`);
      // The start_job call already got its tool result. Completion is a new observation.
      runtime.say(JSON.stringify(event));
      return "message";
    },
    { name: "inbox" },
  );
  const callLlm = node(modelStep, { name: "call_llm" });
  const tools = node(toolsStep, { name: "tools" });
  // Replying does not finish the agent: it returns to listening.
  const respond = node(
    (runtime) => {
      report(`CHEF  ${runtime.run.messages[runtime.run.messages.length - 1].content}`);
    },
    { name: "respond" },
  );
  inbox.on("message", callLlm).on("closed", END);
  callLlm.on("tools", tools).on("answer", respond);
  tools.then(callLlm);
  respond.then(inbox);
  return new Flow(inbox);
}

export async function demo({ report = console.log, logDir }: { report?: (line: string) => void; logDir?: string } = {}): Promise<Run> {
  const mailbox = new Mailbox();
  const kitchen = new Kitchen(mailbox);
  mailbox.put({ kind: "UserMessage", text: ORDER });
  const customer = (async () => {
    // Wait for the three handles, not job completion. No timing-based polling.
    for (let i = 0; i < 3; i++) await kitchen.started.get();
    mailbox.put({ kind: "UserMessage", text: STATUS });
    mailbox.put({ kind: "UserMessage", text: STOP });
    await kitchen.waitAll();
    // Watchers enqueue every final event before returning, so FIFO places this
    // shutdown after those events have been handled by the chef.
    mailbox.put({ kind: "Shutdown" });
  })();
  try {
    const agent = new Agent({
      logDir,
      name: "kitchen-chef",
      provider: new DemoChef(),
      tools: KITCHEN_TOOLS,
      loop: chefLoop(mailbox, kitchen, report),
      limits: { maxModelCalls: 60, maxSteps: 250, timeout: 10 },
      systemPrompt: "Manage kitchen jobs. Their completion events arrive as new mailbox observations.",
    });
    const result = await agent.run("Open kitchen service.", { context: { kitchen } });
    report(`SERVICE ${result.status}: ${result.output ?? result.error}`);
    return result;
  } finally {
    await kitchen.close();
    await customer.catch(() => {});
  }
}

if (import.meta.main) {
  const result = await demo({ logDir: parseArgs({ options: { "log-dir": { type: "string" } } }).values["log-dir"] });
  if (result.logPath) console.log(`Session log: ${result.logPath}`);
  if (result.status !== "completed") process.exitCode = 1;
}
