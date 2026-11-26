/**
 * The board: projects and their task trees, in the durable store.
 *
 * A project is one goal; its root task is that goal. A task is a session (its
 * id is the session id) and each activation is one run. A task finishes when
 * its agent finishes; its parent then gets a `task_finished` inbox message.
 * Every write here is one atomic store operation.
 */

import { validateSessionId } from "./sessionLog.ts";
import type { DurableStore, Tx } from "./journal.ts";
import type { Message } from "./types.ts";
import { type JsonObject, clone, sha256 } from "./util.ts";

/** The inbox kind of a child task's outcome, delivered to its parent. */
export const TASK_FINISHED = "task_finished";

export type ProjectStatus = "active" | "paused" | "done" | "failed" | "cancelled";
export type TaskStatus = "ready" | "running" | "done" | "failed" | "cancelled";

export interface ProjectRecord {
  id: string;
  title: string;
  goal: string;
  status: ProjectStatus;
  /** Why the project is paused (an operator, or the budget), or null. */
  reason: string | null;
  rootTask: string;
  /** Agent definitions its tasks may give subtasks to. */
  agents: string[];
  maxDepth: number;
  maxTasks: number;
  /** Tasks created so far, the root included: the maxTasks guardrail. */
  taskCount: number;
  budgetTokens: number | null;
  spentTokens: number;
  result: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface FollowUp {
  /** Deduplicates a follow-up whose tool call is retried after a crash. */
  key: string;
  text: string;
  from: string | null;
}

export interface TaskRecord {
  id: string;
  project: string;
  parent: string | null;
  depth: number;
  agent: string;
  title: string;
  /** The task: the first activation's prompt. */
  prompt: string;
  /** "fork": the first activation starts from `history`, the creator's conversation. */
  context: "fresh" | "fork";
  history: Message[] | null;
  status: TaskStatus;
  /** Prompts for the next activations, oldest first. */
  followUps: FollowUp[];
  /** Keys of follow-ups already accepted (the latest 200). */
  followUpKeys: string[];
  activations: number;
  spentTokens: number;
  result: string | null;
  error: string | null;
  /** The prompt of the activation in progress (put back if its run never starts). */
  activePrompt: string | null;
  /** The run of the latest activation, and the last run whose outcome was recorded. */
  lastRun: string | null;
  finishedRun: string | null;
  createdBy: string | null;
  createdAt: string;
  updatedAt: string;
}

const now = () => new Date().toISOString();
const asProject = (body: JsonObject | null) => body as unknown as ProjectRecord | null;
const asTask = (body: JsonObject | null) => body as unknown as TaskRecord | null;

function putProject(tx: Tx, project: ProjectRecord): void {
  tx.putProject(project.id, project.status, project as unknown as JsonObject);
}

function putTask(tx: Tx, task: TaskRecord): void {
  tx.putTask(task.id, task.project, task.parent, task.status, task as unknown as JsonObject);
}

function write<T>(store: DurableStore, applyIn: (tx: Tx) => T): Promise<T> {
  return store.committer.submit<T>({ type: "custom", applyIn });
}

/** A task id from a stable key (an idempotency key): retried creation finds the same task. */
export function taskId(key: string): string {
  return `t_${sha256(key).slice(0, 12)}`;
}

// Reads ----------------------------------------------------------------------
export function projects(store: DurableStore): ProjectRecord[] {
  return store.boardProjects() as unknown as ProjectRecord[];
}

export function project(store: DurableStore, id: string): ProjectRecord | null {
  return projects(store).find((item) => item.id === id) ?? null;
}

export function tasks(store: DurableStore, filter: { project?: string; parent?: string; status?: TaskStatus } = {}): TaskRecord[] {
  return store.boardTasks(filter) as unknown as TaskRecord[];
}

export function task(store: DurableStore, id: string): TaskRecord | null {
  return tasks(store).find((item) => item.id === id) ?? null;
}

/** The task and all its descendants, parents before children. */
export function subtree(store: DurableStore, id: string): TaskRecord[] {
  const all = tasks(store);
  const out: TaskRecord[] = [];
  const queue = all.filter((item) => item.id === id);
  while (queue.length) {
    const next = queue.shift() as TaskRecord;
    out.push(next);
    queue.push(...all.filter((item) => item.parent === next.id));
  }
  return out;
}

// Writes ---------------------------------------------------------------------
export interface ProjectInput {
  title: string;
  goal: string;
  /** The root task's agent. */
  agent: string;
  /** Agents tasks may give subtasks to. Default: [agent]. */
  agents?: string[];
  maxDepth?: number;
  maxTasks?: number;
  budgetTokens?: number | null;
  id?: string;
}

/** Create a project and its root task (ready to run), atomically. */
export async function createProject(store: DurableStore, input: ProjectInput): Promise<ProjectRecord> {
  const id = input.id ?? `p_${sha256(`${input.title}\u0000${now()}\u0000${Math.random()}`).slice(0, 10)}`;
  validateSessionId(id);
  if (!input.title?.trim() || !input.goal?.trim() || !input.agent?.trim()) throw new Error("A project needs a title, a goal and an agent");
  const maxDepth = input.maxDepth ?? 3;
  const maxTasks = input.maxTasks ?? 100;
  if (!(Number.isInteger(maxDepth) && maxDepth >= 0) || !(Number.isInteger(maxTasks) && maxTasks >= 1)) throw new Error("Need maxDepth >= 0 and maxTasks >= 1");
  const at = now();
  const root: TaskRecord = {
    id: `${id}-root`,
    project: id,
    parent: null,
    depth: 0,
    agent: input.agent,
    title: input.title,
    prompt: input.goal,
    context: "fresh",
    history: null,
    status: "ready",
    followUps: [],
    followUpKeys: [],
    activations: 0,
    spentTokens: 0,
    result: null,
    error: null,
    activePrompt: null,
    lastRun: null,
    finishedRun: null,
    createdBy: null,
    createdAt: at,
    updatedAt: at,
  };
  const record: ProjectRecord = {
    id,
    title: input.title,
    goal: input.goal,
    status: "active",
    reason: null,
    rootTask: root.id,
    agents: [...new Set(input.agents ?? [input.agent])],
    maxDepth,
    maxTasks,
    taskCount: 1,
    budgetTokens: input.budgetTokens ?? null,
    spentTokens: 0,
    result: null,
    createdAt: at,
    updatedAt: at,
  };
  return write(store, (tx) => {
    if (tx.getProject(id) !== null) throw new Error(`Project ${id} already exists`);
    putProject(tx, record);
    putTask(tx, root);
    return clone(record);
  });
}

export interface TaskInput {
  /** From the creating tool call's idempotency key, so a retried call finds the same task. */
  id: string;
  parent: string;
  agent: string;
  title: string;
  prompt: string;
  context?: "fresh" | "fork";
  history?: Message[] | null;
}

/**
 * Create a child task, ready to run; an existing task with this id is returned
 * unchanged. Rejects one that would break the project's guardrails.
 */
export async function createTask(store: DurableStore, input: TaskInput): Promise<TaskRecord> {
  validateSessionId(input.id);
  return write(store, (tx) => {
    const existing = asTask(tx.getTask(input.id));
    if (existing !== null) return clone(existing);
    const parent = asTask(tx.getTask(input.parent));
    if (parent === null) throw new Error(`No task ${input.parent}`);
    const owner = asProject(tx.getProject(parent.project)) as ProjectRecord;
    if (parent.depth + 1 > owner.maxDepth) throw new Error(`Task ${parent.id} is at depth ${parent.depth}; the project allows at most ${owner.maxDepth}`);
    if (!owner.agents.includes(input.agent)) throw new Error(`Agent ${input.agent} is not one of this project's: ${owner.agents.join(", ")}`);
    if (owner.status !== "active") throw new Error(`Project ${owner.id} is ${owner.status}`);
    if (owner.taskCount >= owner.maxTasks) throw new Error(`Project ${owner.id} already has its maximum of ${owner.maxTasks} tasks`);
    const at = now();
    const created: TaskRecord = {
      id: input.id,
      project: owner.id,
      parent: parent.id,
      depth: parent.depth + 1,
      agent: input.agent,
      title: input.title,
      prompt: input.prompt,
      context: input.context ?? "fresh",
      history: input.history ?? null,
      status: "ready",
      followUps: [],
      followUpKeys: [],
      activations: 0,
      spentTokens: 0,
      result: null,
      error: null,
      activePrompt: null,
      lastRun: null,
      finishedRun: null,
      createdBy: parent.id,
      createdAt: at,
      updatedAt: at,
    };
    putTask(tx, created);
    putProject(tx, { ...owner, taskCount: owner.taskCount + 1, updatedAt: at });
    return clone(created);
  });
}

/**
 * Send a follow-up to a task. A running task gets it as a pinned inbox message
 * before its next model turn; any other task (finished, failed, ready) gets a
 * new activation with it as the prompt. A repeated `key` does nothing.
 * Returns how it was delivered.
 */
export async function followUp(store: DurableStore, id: string, input: FollowUp): Promise<"message" | "activation" | "duplicate"> {
  const outcome = await write(store, (tx) => {
    const found = asTask(tx.getTask(id));
    if (found === null) throw new Error(`No task ${id}`);
    if (found.status === "cancelled") throw new Error(`Task ${id} was cancelled`);
    if (found.followUpKeys.includes(input.key)) return "duplicate" as const;
    const followUpKeys = [...found.followUpKeys, input.key].slice(-200);
    if (found.status === "running") {
      tx.insertSignal(id, "message", { text: input.text, ...(input.from ? { from: input.from } : {}), pinned: true }, now());
      putTask(tx, { ...found, followUpKeys, updatedAt: now() });
      return "message" as const;
    }
    putTask(tx, { ...found, status: "ready", followUps: [...found.followUps, input], followUpKeys, updatedAt: now() });
    return "activation" as const;
  });
  if (outcome === "message") store.notify(id);
  return outcome;
}

/** Mark a task running for a new activation and take its prompt: the task's own first, then follow-ups in order. */
export async function startActivation(store: DurableStore, id: string, runId: string): Promise<{ task: TaskRecord; prompt: string; first: boolean } | null> {
  return write(store, (tx) => {
    const found = asTask(tx.getTask(id));
    if (found === null || found.status !== "ready") return null;
    const first = found.activations === 0;
    const [next, ...rest] = found.followUps;
    if (!first && next === undefined) return null;
    const prompt = first ? found.prompt : (next as FollowUp).text;
    const updated: TaskRecord = {
      ...found,
      status: "running",
      activations: found.activations + 1,
      followUps: first ? found.followUps : rest,
      activePrompt: prompt,
      lastRun: runId,
      updatedAt: now(),
    };
    putTask(tx, updated);
    return { task: clone(updated), prompt, first };
  });
}

/**
 * Record an activation's outcome once per run, atomically with telling the
 * parent (a pinned `task_finished` message) and the project's accounting: its
 * spent tokens, its status when the root task ends, and a pause when the budget
 * is spent. Pending follow-ups make the task ready again instead.
 */
export async function finishActivation(
  store: DurableStore,
  input: { id: string; runId: string; status: string; output: string | null; error: string | null; tokens: number },
): Promise<TaskRecord | null> {
  const outcome = await write(store, (tx) => {
    const found = asTask(tx.getTask(input.id));
    if (found === null || found.finishedRun === input.runId || found.status === "cancelled") return null;
    const at = now();
    const status: TaskStatus = input.status === "completed" ? "done" : input.status === "cancelled" ? "cancelled" : "failed";
    const next: TaskStatus = found.followUps.length && status !== "cancelled" ? "ready" : status;
    const updated: TaskRecord = {
      ...found,
      status: next,
      result: status === "done" ? input.output : found.result,
      error: status === "done" ? null : (input.error ?? input.status),
      spentTokens: found.spentTokens + input.tokens,
      finishedRun: input.runId,
      updatedAt: at,
    };
    putTask(tx, updated);
    const owner = asProject(tx.getProject(found.project)) as ProjectRecord;
    const spent = owner.spentTokens + input.tokens;
    let projectUpdate: ProjectRecord = { ...owner, spentTokens: spent, updatedAt: at };
    if (owner.rootTask === found.id && next !== "ready") {
      projectUpdate = { ...projectUpdate, status: status === "done" ? "done" : status === "cancelled" ? "cancelled" : "failed", result: input.output ?? input.error };
    } else if (owner.status === "active" && owner.budgetTokens !== null && spent >= owner.budgetTokens) {
      projectUpdate = { ...projectUpdate, status: "paused", reason: `budget: ${spent} of ${owner.budgetTokens} tokens spent` };
    }
    putProject(tx, projectUpdate);
    if (found.parent !== null) {
      const body = status === "done" ? (input.output ?? "") : `${input.status}: ${input.error ?? ""}`;
      tx.insertSignal(
        found.parent,
        TASK_FINISHED,
        { text: `[task ${found.id} · ${found.title} · ${status}]\n${body}`, taskId: found.id, status, pinned: true },
        at,
      );
    }
    return clone(updated);
  });
  if (outcome?.parent) store.notify(outcome.parent);
  return outcome;
}

/**
 * Cancel a task and its open descendants; returns the ids that were running (to
 * freeze). The cancelled task's parent is told, so it does not wait for it.
 */
export async function cancelTask(store: DurableStore, id: string, reason = "cancelled by the operator"): Promise<string[]> {
  const ids = subtree(store, id).map((item) => item.id);
  const top = task(store, id);
  const running = await write(store, (tx) => {
    const running: string[] = [];
    if (top !== null && top.parent !== null && !["done", "failed", "cancelled"].includes(top.status)) {
      tx.insertSignal(top.parent, TASK_FINISHED, { text: `[task ${top.id} · ${top.title} · cancelled]\n${reason}`, taskId: top.id, status: "cancelled", pinned: true }, now());
    }
    for (const taskIdValue of ids) {
      const found = asTask(tx.getTask(taskIdValue));
      if (found === null || ["done", "failed", "cancelled"].includes(found.status)) continue;
      if (found.status === "running") running.push(found.id);
      putTask(tx, { ...found, status: "cancelled", error: reason, updatedAt: now() });
      const owner = asProject(tx.getProject(found.project)) as ProjectRecord;
      if (owner.rootTask === found.id) putProject(tx, { ...owner, status: "cancelled", updatedAt: now() });
    }
    return running;
  });
  if (top?.parent) store.notify(top.parent);
  return running;
}

/** Put a task whose activation never got a run (its process died first) back to ready, its prompt first again. */
export async function revertActivation(store: DurableStore, id: string): Promise<void> {
  await write(store, (tx) => {
    const found = asTask(tx.getTask(id));
    if (found === null || found.status !== "running") return;
    const first = found.activations === 1;
    const followUps = first || found.activePrompt === null ? found.followUps : [{ key: `revert:${found.lastRun}`, text: found.activePrompt, from: null }, ...found.followUps];
    putTask(tx, { ...found, status: "ready", activations: found.activations - 1, followUps, lastRun: found.finishedRun, updatedAt: now() });
  });
}

/** Pause a project (no new activations; running ones go on) or let it go on. */
export async function setProjectStatus(store: DurableStore, id: string, status: "active" | "paused", reason: string | null = null): Promise<ProjectRecord> {
  return write(store, (tx) => {
    const found = asProject(tx.getProject(id));
    if (found === null) throw new Error(`No project ${id}`);
    if (!["active", "paused"].includes(found.status)) throw new Error(`Project ${id} is ${found.status}`);
    const updated = { ...found, status, reason: status === "paused" ? (reason ?? "paused by the operator") : null, updatedAt: now() };
    putProject(tx, updated);
    return clone(updated);
  });
}
