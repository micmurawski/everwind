/** Expose agents as tools, with optional supervised background execution. */

import { z } from "zod";
import type { Agent } from "./agent.ts";
import { describeError } from "./errors.ts";
import type { Runtime } from "./runtime.ts";
import { validateSessionId } from "./sessionLog.ts";
import { type Tool, Tools, tool } from "./tools.ts";
import type { Run } from "./types.ts";
import { hexId } from "./util.ts";

function continuation(name: string, sessionId: string): string {
  return `To continue, call ${name}(prompt="...", sessionId="${sessionId}").`;
}

function runResult(run: Run, name: string): Record<string, unknown> {
  return {
    sessionId: run.sessionId,
    runId: run.id,
    status: run.status,
    output: run.output,
    error: run.error,
    continuation: continuation(name, run.sessionId),
  };
}

/** An unbounded async queue. */
export class Mailbox<T = Record<string, unknown>> {
  private items: T[] = [];
  private waiters: ((item: T) => void)[] = [];

  put(item: T): void {
    const waiter = this.waiters.shift();
    if (waiter) waiter(item);
    else this.items.push(item);
  }

  get(): Promise<T> {
    if (this.items.length) return Promise.resolve(this.items.shift() as T);
    return new Promise((resolve) => this.waiters.push(resolve));
  }

  get size(): number {
    return this.items.length;
  }
}

class AgentJob {
  status = "running";
  result: Record<string, unknown> | null = null;
  promise: Promise<void> | null = null;
  readonly controller = new AbortController();

  constructor(
    readonly id: string,
    readonly sessionId: string,
    readonly parentSessionId: string,
    readonly parentRunId: string,
    readonly toolName: string,
  ) {}

  snapshot(): Record<string, unknown> {
    return {
      jobId: this.id,
      sessionId: this.sessionId,
      parentSessionId: this.parentSessionId,
      parentRunId: this.parentRunId,
      status: this.status,
      continuation: continuation(this.toolName, this.sessionId),
      ...(this.result ?? {}),
    };
  }
}

/**
 * One parent's supervisor. Terminal events wake its unbounded mailbox.
 *
 * Call close() (or use `await using`) so shutdown cancels and awaits children.
 * Cancellation is cooperative: it aborts the child's signal.
 */
export class AgentJobs {
  private readonly jobs = new Map<string, AgentJob>();
  private parentSessionId: string | null = null;
  private closed = false;

  constructor(readonly mailbox: Mailbox) {}

  start(
    agent: Agent,
    prompt: string,
    options: { sessionId?: string | null; runtime: Runtime; toolName: string; context: Record<string, unknown> },
  ): Record<string, unknown> {
    if (this.closed) throw new Error("AgentJobs is closed");
    const parent = options.runtime.run;
    if (this.parentSessionId !== null && this.parentSessionId !== parent.sessionId) {
      throw new Error("Use a separate AgentJobs mailbox for each parent session");
    }
    this.parentSessionId = parent.sessionId;
    const sessionId = options.sessionId ?? hexId();
    validateSessionId(sessionId);
    if (sessionId === parent.sessionId) throw new Error("A child cannot run in its active parent session");
    const job = new AgentJob(hexId(), sessionId, parent.sessionId, parent.id, options.toolName);
    // If logging fails, do not launch an unacknowledged child.
    parent.emit("agent_job_started", job.snapshot());
    this.jobs.set(job.id, job);
    job.promise = agent
      .run(prompt, {
        sessionId,
        context: options.context,
        parentSessionId: parent.sessionId,
        parentRunId: parent.id,
        jobId: job.id,
        signal: job.controller.signal,
      })
      .then(
        (run) => {
          job.result = runResult(run, job.toolName);
          job.status = run.status;
        },
        (error) => {
          if (job.controller.signal.aborted) {
            job.status = "cancelled";
            job.result = { output: null, error: "Child cancelled by supervisor" };
          } else {
            job.status = "failed";
            job.result = { output: null, error: describeError(error) };
          }
        },
      )
      .then(() => this.mailbox.put({ kind: "AgentFinished", ...job.snapshot() }));
    return job.snapshot();
  }

  private owned(jobId: string, parentSessionId: string): AgentJob {
    const job = this.jobs.get(jobId);
    if (job === undefined) throw new Error(`Unknown job ${jobId}`);
    if (job.parentSessionId !== parentSessionId) throw new Error("Job belongs to another parent session");
    return job;
  }

  inspect(jobId: string, parentSessionId: string): Record<string, unknown> {
    return this.owned(jobId, parentSessionId).snapshot();
  }

  listJobs(parentSessionId: string): Record<string, unknown>[] {
    return [...this.jobs.values()].filter((job) => job.parentSessionId === parentSessionId).map((job) => job.snapshot());
  }

  cancel(jobId: string, parentSessionId: string): Record<string, unknown> {
    const job = this.owned(jobId, parentSessionId);
    if (job.result === null && job.status !== "stopping") {
      job.status = "stopping";
      job.controller.abort();
    }
    return job.snapshot();
  }

  async waitAll(): Promise<void> {
    await Promise.allSettled([...this.jobs.values()].map((job) => job.promise));
  }

  async close(): Promise<void> {
    this.closed = true;
    for (const job of this.jobs.values()) this.cancel(job.id, job.parentSessionId);
    await this.waitAll();
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.close();
  }
}

/**
 * Bind an agent configuration; the model supplies only prompt and sessionId.
 *
 * Child resources are explicit via `context`, never inherited from the parent.
 * Background mode requires AgentJobs in the parent's `agentJobs` context value.
 */
export function agentTool(
  agent: Agent,
  options: { name?: string; description?: string; execution?: "wait" | "background"; context?: Record<string, unknown> } = {},
): Tool {
  const execution = options.execution ?? "wait";
  if (execution !== "wait" && execution !== "background") throw new Error("execution must be 'wait' or 'background'");
  const name = options.name ?? `ask_${agent.name}`;
  const childContext = { ...(options.context ?? {}) };
  const description =
    (options.description ?? `Ask ${agent.name} to perform a task.`) +
    " Omit sessionId to start a conversation; supply it to continue. " +
    (execution === "wait"
      ? "Returns the answer and sessionId."
      : "Returns a job handle immediately; AgentFinished arrives in the mailbox when it ends.");
  return tool({
    name,
    description,
    input: z.object({ prompt: z.string(), sessionId: z.string().nullable().optional() }),
    async run({ prompt, sessionId }, ctx) {
      const runtime = ctx.runtime;
      if (execution === "background") {
        return ctx.get<AgentJobs>("agentJobs").start(agent, prompt, { sessionId, runtime, toolName: name, context: childContext });
      }
      const run = await agent.run(prompt, {
        sessionId: sessionId ?? undefined,
        context: childContext,
        parentSessionId: runtime.run.sessionId,
        parentRunId: runtime.run.id,
        signal: ctx.signal,
      });
      return runResult(run, name);
    },
  });
}

export const listAgentJobs = tool({
  name: "list_agent_jobs",
  description: "List this parent's delegated agent jobs and their results.",
  run: (_args, ctx) => ctx.get<AgentJobs>("agentJobs").listJobs(ctx.runtime.run.sessionId),
});

export const inspectAgentJob = tool({
  name: "inspect_agent_job",
  description: "Inspect a delegated agent job's status and result.",
  input: z.object({ jobId: z.string() }),
  run: ({ jobId }, ctx) => ctx.get<AgentJobs>("agentJobs").inspect(jobId, ctx.runtime.run.sessionId),
});

export const cancelAgentJob = tool({
  name: "cancel_agent_job",
  description: "Request cancellation; AgentFinished will confirm the terminal outcome.",
  input: z.object({ jobId: z.string() }),
  run: ({ jobId }, ctx) => ctx.get<AgentJobs>("agentJobs").cancel(jobId, ctx.runtime.run.sessionId),
});

export const AGENT_JOB_TOOLS = new Tools([listAgentJobs, inspectAgentJob, cancelAgentJob]);
