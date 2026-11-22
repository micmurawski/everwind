/** Validation shared by session continuation and context compaction. */

import { type Message, MessageSchema } from "./types.ts";

export function validateHistory(history: readonly Message[]): void {
  const pending = new Set<string>();
  for (const item of history) {
    // Revalidate at this persistence boundary: callers may have built the object by hand.
    const message = MessageSchema.parse(item);
    if (message.role === "system") {
      throw new Error("Set systemPrompt on Agent; history must not include system messages");
    }
    if (message.role === "tool") {
      if (!pending.has(message.toolCallId as string)) throw new Error("History contains an unmatched tool result");
      pending.delete(message.toolCallId as string);
    } else {
      if (pending.size) throw new Error("History contains unresolved tool calls");
      const ids = message.toolCalls.map((call) => call.id);
      if (new Set(ids).size !== ids.length) throw new Error("History contains duplicate tool call IDs");
      for (const id of ids) pending.add(id);
    }
  }
  if (pending.size) throw new Error("History contains unresolved tool calls");
}

/** The system message, when present, must be first; all tool groups must close. */
export function validateConversation(messages: readonly Message[]): void {
  const offset = messages.length && messages[0].role === "system" ? 1 : 0;
  validateHistory(messages.slice(offset));
}
