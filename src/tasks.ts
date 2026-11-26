/**
 * Model-facing tasks: `task` creates a child task (or follows one up),
 * `wait_for_tasks` waits for children, and the `settle` node keeps a task from
 * finishing while its children are still out. Children run asynchronously
 * (a Supervisor runs them); each one's outcome arrives as a `task_finished`
 * message that the gate delivers.
 */

import { z } from "zod";
import { TASK_FINISHED, createTask, followUp, task as boardTask, taskId, tasks as boardTasks } from "./board.ts";
import type { Step } from "./flow.ts";
import { MESSAGE } from "./inbox.ts";
import type { Runtime } from "./runtime.ts";
import { canSpawn, readTaskScope } from "./scope.ts";
import { type Tool, tool } from "./tools.ts";
import { type Message, message } from "./types.ts";

/** What wakes a task waiting for its children: their outcomes, and messages from you. */
const WAKE = [TASK_FINISHED, MESSAGE];

/**
 * The conversation a forked child starts from. The same agent gets the
 * messages themselves; another agent gets them as a transcript, since their
 * tool calls name tools it may not have, and provider metadata may not fit it.
 */
function forkHistory(messages: readonly Message[], sameAgent: boolean): Message[] {
  const conversation = messages.filter((item) => item.role !== "system");
  // Drop the turn that is creating the task: its tool calls have no results yet.
  let end = conversation.length;
  while (end > 0 && conversation[end - 1].role === "tool") end -= 1;
  if (end > 0 && conversation[end - 1].role === "assistant") end -= 1;
  const prefix = conversation.slice(0, end);
  if (!prefix.length) return [];
  if (sameAgent) return prefix.map((item) => message(item));
  const transcript = prefix.map((item) => ({ role: item.role, content: item.content, toolCalls: item.toolCalls.map((call) => ({ name: call.name, arguments: call.arguments })) }));
  return [
    message({ role: "user", content: `The conversation of the task that created you, as a transcript:\n${JSON.stringify(transcript)}` }),
    message({ role: "assistant", content: "I have read it." }),
  ];
}

/**
 * `task`: create a child task for one of the agents this task may use, or
 * follow up a child created earlier (`id`). Hidden from a task at the
 * project's maximum depth. The child runs in the background.
 */
export function taskTool(): Tool {
  return tool({
    name: "task",
    available: canSpawn,
    description: (scope) =>
      `Give a subtask to another agent (${(scope.agents as string[]).join(", ")}). It runs in the background and its final answer ` +
      "arrives later as a message; call wait_for_tasks to wait for it. Write a complete, self-contained prompt: with " +
      'context "fresh" (default) the agent sees only your prompt; "fork" also gives it your conversation so far. ' +
      "To follow up a task you created, pass its id and the new prompt: a running task gets it as a message, a finished one runs again.",
    effect: "idempotent",
    input: (scope) =>
      z.object({
        prompt: z.string().min(1).describe("The task, or the follow-up"),
        agent: z.enum(scope.agents as [string, ...string[]]).optional().describe("Who does a new task (not used for a follow-up)"),
        title: z.string().min(1).max(120).optional().describe("A short title for the board"),
        context: z.enum(["fresh", "fork"]).optional().describe("fresh (default): only the prompt; fork: also your conversation"),
        id: z.string().optional().describe("A task you created earlier, to follow up"),
      }),
    async run({ prompt, agent, title, context, id }, ctx) {
      const scope = readTaskScope(ctx.scope);
      const store = ctx.runtime.store;
      if (scope === null || store === null) throw new Error("task needs a task scope and a durable store");
      if (id !== undefined) {
        const child = boardTask(store, id);
        if (child === null || child.parent !== scope.task) throw new Error(`${id} is not a task you created`);
        const how = await followUp(store, id, { key: ctx.idempotencyKey, text: prompt, from: scope.task });
        return {
          taskId: id,
          delivered: how === "activation" ? "as a new activation; its answer arrives as a message" : how === "message" ? "as a message to the running task" : "already delivered",
        };
      }
      if (agent === undefined) throw new Error("Name the agent for a new task");
      const mode = context ?? "fresh";
      const created = await createTask(store, {
        id: taskId(ctx.idempotencyKey),
        parent: scope.task,
        agent,
        title: title ?? prompt.slice(0, 80),
        prompt,
        context: mode,
        history: mode === "fork" ? forkHistory(ctx.runtime.run.messages, agent === scope.agent) : null,
      });
      return { taskId: created.id, status: created.status, note: "It runs in the background; its result arrives as a message. Call wait_for_tasks to wait." };
    },
  });
}

const WaitInput = z.object({ ids: z.array(z.string()).optional().describe("Tasks to wait for; all your unfinished tasks when omitted") });

/** `wait_for_tasks`: records the wait; the loop's `wait` node waits until one of them reports. */
export function waitTool(): Tool {
  return tool({
    name: "wait_for_tasks",
    available: canSpawn,
    description: "Wait until a task you created finishes (any of `ids`, or any of your unfinished tasks). Its result then arrives as a message.",
    effect: "read",
    input: WaitInput,
    run: ({ ids }) => ({ waiting: ids ?? "all your unfinished tasks" }),
  });
}

/** The task ids the last model turn asked to wait for ([] = all), or null when it did not ask. */
export function waitRequest(runtime: Runtime): string[] | null {
  const messages = runtime.run.messages;
  let index = messages.length - 1;
  while (index >= 0 && messages[index].role === "tool") index -= 1;
  const assistant = messages[index];
  if (assistant?.role !== "assistant") return null;
  const failed = new Set(messages.slice(index + 1).filter((item) => item.isError).map((item) => item.toolCallId));
  const calls = assistant.toolCalls.filter((call) => call.name === "wait_for_tasks" && !failed.has(call.id));
  if (!calls.length) return null;
  const parsed = WaitInput.safeParse(calls[calls.length - 1].arguments);
  return parsed.success ? (parsed.data.ids ?? []) : null;
}

/**
 * This task's children that have not finished, and whether outcomes wait in
 * the inbox. Journaled (runtime.record), so a replay decides the same way.
 */
export function outstanding(runtime: Runtime, ids: readonly string[] = []): { running: string[]; delivered: boolean } {
  const scope = readTaskScope(runtime.run.scope);
  const store = runtime.store;
  if (scope === null || store === null) return { running: [], delivered: false };
  return runtime.record("outstanding", () => {
    const children = boardTasks(store, { parent: scope.task }).filter((child) => !ids.length || ids.includes(child.id));
    const running = children.filter((child) => child.status === "ready" || child.status === "running").map((child) => child.id);
    const delivered = store.inbox(runtime.run.sessionId, { kinds: [TASK_FINISHED], pending: true, limit: 1 }).length > 0;
    return { running, delivered };
  });
}

/** A loop node after tools: if the model called wait_for_tasks, wait until a child reports (or you write). */
export function waitStep(): Step {
  return async function wait(runtime: Runtime): Promise<void> {
    const ids = waitRequest(runtime);
    if (ids === null) return;
    const { running, delivered } = outstanding(runtime, ids);
    if (delivered || !running.length) return; // An outcome is already there for the gate.
    await runtime.suspend({ wakeOn: WAKE, reason: `waiting for ${running.join(", ")}` });
  };
}

/**
 * A loop node between an answer and finishing: a task whose children are still
 * out does not finish. It waits for them (or a message from you), then returns
 * "continue" so the gate delivers their outcomes and the model goes on; with
 * nothing outstanding it returns "done".
 */
export function settleStep(): Step {
  return async function settle(runtime: Runtime): Promise<string> {
    const { running, delivered } = outstanding(runtime);
    if (delivered) return "continue";
    if (!running.length) return "done";
    runtime.run.emit("settling", { waitingFor: running });
    await runtime.suspend({ wakeOn: WAKE, reason: `waiting for ${running.join(", ")}` });
    return "continue";
  };
}
