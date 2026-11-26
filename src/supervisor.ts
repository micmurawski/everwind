/**
 * The supervisor runs projects: it admits ready tasks, runs each activation
 * (agent.run on the task's session, with its scope and frame), and records
 * outcomes on the board. Everything it decides comes from the store, so a
 * restarted supervisor (or one next to the desk) picks up where another left off.
 */

import type { Agent } from "./agent.ts";
import {
  type ProjectInput,
  type ProjectRecord,
  type TaskRecord,
  cancelTask,
  createProject,
  finishActivation,
  project as boardProject,
  projects as boardProjects,
  revertActivation,
  startActivation,
  tasks as boardTasks,
} from "./board.ts";
import { isStopError } from "./errors.ts";
import type { DurableStore, RunRecord } from "./journal.ts";
import { taskFrame, taskScope } from "./scope.ts";
import type { Run } from "./types.ts";
import { type JsonObject, hexId, sleep } from "./util.ts";

export interface SupervisorOptions {
  /** Most activations running at once, over all projects. Default 8. */
  maxActive?: number;
  /** Most activations at once per agent definition. */
  perAgent?: Record<string, number>;
  /** Live resources for a task's run (its tools' context). */
  context?: Record<string, unknown> | ((task: TaskRecord) => Record<string, unknown>);
  /** Seconds between scheduling passes. Default 0.25. */
  pollInterval?: number;
}

/** Tokens a run used (input and output), from its usage. */
export function tokensOf(usage: Record<string, number> | undefined | null): number {
  let total = 0;
  for (const [name, value] of Object.entries(usage ?? {})) if (name.endsWith("Tokens") && name !== "reasoningTokens") total += value;
  return total;
}

export class Supervisor {
  readonly agents: ReadonlyMap<string, Agent>;
  private readonly inflight = new Map<string, Promise<void>>();
  private looping: Promise<void> | null = null;
  private running = false;
  private firstPass = true;
  private readonly maxActive: number;
  private readonly poll: number;

  constructor(
    readonly store: DurableStore,
    agents: readonly Agent[],
    private readonly options: SupervisorOptions = {},
  ) {
    this.agents = new Map(agents.map((agent) => [agent.name, agent]));
    for (const agent of agents) if (agent.store !== store) throw new Error(`Agent ${agent.name} must use the supervisor's store`);
    this.maxActive = options.maxActive ?? 8;
    this.poll = (options.pollInterval ?? 0.25) * 1000;
  }

  /** Create a project; its root task starts at the next pass. */
  createProject(input: ProjectInput): Promise<ProjectRecord> {
    for (const name of [input.agent, ...(input.agents ?? [])]) {
      if (!this.agents.has(name)) throw new Error(`The supervisor has no agent named ${JSON.stringify(name)}`);
    }
    return createProject(this.store, input);
  }

  /** Cancel a task and its subtree; running ones are frozen (they stay recoverable, but are not resumed). */
  async cancel(taskId: string, reason?: string): Promise<void> {
    for (const id of await cancelTask(this.store, taskId, reason)) await this.store.setControl(id, "freeze", "task cancelled");
  }

  /** Start scheduling. The first pass also recovers runs of tasks whose process died. */
  start(): void {
    if (this.running) return;
    this.running = true;
    this.looping = (async () => {
      while (this.running) {
        try {
          await this.tick();
        } catch {
          // A failed pass is retried at the next one.
        }
        await sleep(this.poll / 1000);
      }
    })();
  }

  /**
   * Stop scheduling and wait for the activations in flight. With `freeze`,
   * freeze them first: they stop at their next step and stay recoverable.
   */
  async stop(options: { freeze?: boolean } = {}): Promise<void> {
    this.running = false;
    await this.looping;
    if (options.freeze) for (const id of this.inflight.keys()) await this.store.setControl(id, "freeze", "supervisor stopped");
    await Promise.allSettled([...this.inflight.values()]);
  }

  /** Wait until no task of the project is ready or running (or `timeout` seconds pass). */
  async idle(projectId: string, timeout = 30): Promise<ProjectRecord> {
    const end = Date.now() + timeout * 1000;
    for (;;) {
      const open = boardTasks(this.store, { project: projectId }).filter((item) => item.status === "ready" || item.status === "running");
      const found = boardProject(this.store, projectId);
      if (found === null) throw new Error(`No project ${projectId}`);
      if (!open.length || found.status !== "active" || Date.now() > end) return found;
      await sleep(0.05);
    }
  }

  /** One scheduling pass: reconcile running tasks with their runs, then admit ready ones. */
  async tick(): Promise<void> {
    const force = this.firstPass;
    this.firstPass = false;
    for (const item of boardTasks(this.store, { status: "running" })) {
      if (!this.inflight.has(item.id)) await this.reconcile(item, force);
    }
    const active = new Set(boardProjects(this.store).filter((item) => item.status === "active").map((item) => item.id));
    for (const item of boardTasks(this.store, { status: "ready" })) {
      if (this.inflight.size >= this.maxActive) break;
      if (!active.has(item.project) || this.inflight.has(item.id)) continue;
      const agent = this.agents.get(item.agent);
      if (agent === undefined) continue; // Another supervisor may run this agent.
      const limit = this.options.perAgent?.[item.agent];
      if (limit !== undefined && [...this.inflight.keys()].filter((id) => this.agentOf(id) === item.agent).length >= limit) continue;
      await this.activate(item, agent);
    }
  }

  private agentOf(id: string): string | undefined {
    return boardTasks(this.store).find((item) => item.id === id)?.agent;
  }

  private contextFor(item: TaskRecord): Record<string, unknown> {
    return typeof this.options.context === "function" ? this.options.context(item) : { ...(this.options.context ?? {}) };
  }

  private async activate(item: TaskRecord, agent: Agent): Promise<void> {
    const runId = hexId();
    const started = await startActivation(this.store, item.id, runId);
    if (started === null) return;
    const owner = boardProject(this.store, item.project) as ProjectRecord;
    const scope = taskScope({
      project: owner.id,
      task: item.id,
      parent: item.parent,
      depth: item.depth,
      maxDepth: owner.maxDepth,
      agents: owner.agents,
      agent: item.agent,
    });
    const run = agent.run(started.prompt, {
      sessionId: item.id,
      runId,
      scope,
      // The frame is fixed for the session: set by the first activation, reused after.
      frame: started.first ? taskFrame(scope) : undefined,
      pin: !started.first,
      history: started.first && started.task.history ? started.task.history : undefined,
      context: this.contextFor(item),
      parentSessionId: item.parent ?? undefined,
    });
    this.track(item.id, run);
  }

  private track(id: string, run: Promise<Run>): void {
    const done = run
      .then((finished) => this.record(id, finished.id, finished.status, finished.output, finished.error, finished.usage))
      .catch((error) => {
        // A stop error (lost ownership, a journal failure) leaves the run as it is; reconcile looks at it later.
        // Anything else (a recover() refused, an invalid session) does too: the next pass decides from the store.
        void isStopError(error);
      })
      .finally(() => this.inflight.delete(id));
    this.inflight.set(id, done);
  }

  private async record(id: string, runId: string, status: string, output: string | null, error: string | null, usage: Record<string, number> | null): Promise<void> {
    // Running (lost ownership) or frozen: not an outcome; reconcile looks again.
    if (status === "running" || status === "frozen") return;
    await finishActivation(this.store, { id, runId, status, output, error, tokens: tokensOf(usage) });
  }

  /** A running task without an activation in this process: look at its run in the store. */
  private async reconcile(item: TaskRecord, force: boolean): Promise<void> {
    const record: RunRecord | null = item.lastRun === null ? null : this.store.getRun(item.id, item.lastRun);
    if (record === null) {
      await revertActivation(this.store, item.id); // Its process died before the run started.
      return;
    }
    if (record.status !== "running") {
      const result = (record.result ?? {}) as JsonObject;
      await this.record(item.id, record.runId, record.status, (result.output as string | null) ?? null, (result.error as string | null) ?? null, (result.usage as Record<string, number>) ?? null);
      return;
    }
    if (this.store.control(item.id)?.state === "freeze") return; // Frozen on purpose: someone recovers it.
    const agent = this.agents.get(item.agent);
    const expired = record.owner === null || record.leaseExpiresAt * 1000 < Date.now();
    if (agent === undefined || !(expired || force)) return; // Live in another process.
    this.track(item.id, agent.recover(item.id, { context: this.contextFor(item), force: true }));
  }
}
