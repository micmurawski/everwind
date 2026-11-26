/** A run's task scope: what a supervisor tells a task's agent, and what its tools bind to. */

import { z } from "zod";
import type { JsonObject } from "./util.ts";

/**
 * The scope of one task's runs. Everything here is stable for the task's
 * lifetime, so its frame (taskFrame) can sit in the system prompt and stay
 * cacheable. What changes during a task (budget left, children's results)
 * belongs in messages, not here.
 */
export const TaskScopeSchema = z.strictObject({
  project: z.string().min(1),
  task: z.string().min(1),
  parent: z.string().min(1).nullable().default(null),
  /** 0 for a project's root task. */
  depth: z.number().int().nonnegative(),
  /** The deepest a task may be; a task at this depth cannot create subtasks. */
  maxDepth: z.number().int().nonnegative(),
  /** Agent definitions this task may give subtasks to. */
  agents: z.array(z.string().min(1)).default([]),
  /** The agent running this task, so a fork knows whether it stays with the same agent. */
  agent: z.string().min(1).nullable().default(null),
});
export type TaskScope = z.output<typeof TaskScopeSchema>;
export type TaskScopeInput = z.input<typeof TaskScopeSchema>;

/** A validated task scope, as the JSON a run carries. */
export function taskScope(input: TaskScopeInput): JsonObject {
  const scope = TaskScopeSchema.parse(input);
  if (scope.depth > scope.maxDepth) throw new Error(`Task depth ${scope.depth} is beyond maxDepth ${scope.maxDepth}`);
  return scope as unknown as JsonObject;
}

/** The task scope inside a run's scope, or null when the run is not a task. */
export function readTaskScope(scope: JsonObject): TaskScope | null {
  const parsed = TaskScopeSchema.safeParse(scope);
  return parsed.success ? parsed.data : null;
}

/**
 * Whether a run with this scope may create subtasks: it is a task, below
 * maxDepth, with at least one agent to give them to. Use it as a spawning
 * tool's `available`, so the model never sees the tool at the limit.
 */
export function canSpawn(scope: JsonObject): boolean {
  const task = readTaskScope(scope);
  return task !== null && task.depth < task.maxDepth && task.agents.length > 0;
}

/**
 * The text a supervisor appends to a task's system prompt (agent.run's
 * `frame`). It holds only facts the supervisor controls; the task itself (its
 * goal and instructions) is the run's prompt, a user message.
 */
export function taskFrame(scope: JsonObject): string {
  const task = readTaskScope(scope);
  if (task === null) throw new Error("taskFrame needs a task scope");
  const where = task.parent === null ? "the root task" : `a subtask of ${task.parent}`;
  const lines = [
    "# Task",
    `You are working on task ${task.task} in project ${task.project}: ${where}, at depth ${task.depth} of at most ${task.maxDepth}.`,
    "Your final answer is the task's result. It goes to whoever created the task, so make it complete and self-contained.",
    "Messages can arrive while you work: notes from the operator, and follow-ups on this task. Follow-ups stay in your context.",
    canSpawn(scope)
      ? `You can give subtasks to these agents: ${task.agents.join(", ")}.`
      : "You cannot create subtasks: do the work yourself.",
  ];
  return lines.join("\n");
}
