/** Delivering a session's inbox to its agent: the gate node before each model turn. */

import { TASK_FINISHED } from "./board.ts";
import type { Step } from "./flow.ts";
import type { InboxMessage } from "./journal.ts";
import type { Runtime } from "./runtime.ts";

/** The kind of a message for the agent to read, such as an operator's "send now" note. */
export const MESSAGE = "message";

/** `payload.text`, prefixed with `[from]` when the payload names a sender; null for no text. */
export function formatMessage(message: InboxMessage): string | null {
  const text = message.payload.text;
  if (typeof text !== "string" || !text) return null;
  const from = message.payload.from;
  return typeof from === "string" && from ? `[${from}] ${text}` : text;
}

export interface GateOptions {
  /** Kinds delivered as user messages. Default ["message", "task_finished"]: your messages and child tasks' outcomes. */
  kinds?: readonly string[];
  /** The text of a message for the agent, or null to consume it silently. */
  format?: (message: InboxMessage) => string | null;
}

/**
 * A loop node to put before a model turn: take the session's pending messages
 * and add each as a user message, so the next turn sees them. This is how a
 * message sent while the agent works arrives at the next turn ("send now").
 * A run without a durable store has no inbox; the gate then does nothing.
 */
export function gateStep(options: GateOptions = {}): Step {
  const kinds = [...(options.kinds ?? [MESSAGE, TASK_FINISHED])];
  const format = options.format ?? formatMessage;
  return async function gate(runtime: Runtime): Promise<void> {
    for (const message of await runtime.receive({ kinds })) {
      const text = format(message);
      if (text !== null) runtime.say(text, { pinned: message.payload.pinned === true });
    }
  };
}
