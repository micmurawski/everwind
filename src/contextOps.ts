/** Explicit context selection and compaction strategies; Runtime owns the commit. */

import type { Provider } from "./providers/base.ts";
import { type Message, type ModelResponse, message } from "./types.ts";

export const SUMMARY_PROMPT = `Your task is to create a detailed summary of the conversation so far, paying close attention to the user's
     explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions
     that would be essential for continuing with the conversation and supporting any continuing tasks.

Your summary should be structured as follows:
Context: The context to continue the conversation with. If applicable based on the current task, this should
     include:
  1. Previous Conversation: High level details about what was discussed throughout the entire conversation
     with the user. This should be written to allow someone to be able to follow the general overarching
     conversation flow.
  2. Current Work: Describe in detail what was being worked on prior to this request to summarize the
     conversation. Pay special attention to the more recent messages in the conversation.
  3. Key Technical Concepts: List all important technical concepts, technologies, coding conventions, and
     frameworks discussed, which might be relevant for continuing with this work.
  4. Relevant Files and Code: If applicable, enumerate specific files and code sections examined, modified, or
     created for the task continuation. Pay special attention to the most recent messages and changes.
  5. Problem Solving: Document problems solved thus far and any ongoing troubleshooting efforts.
  6. Pending Tasks and Next Steps: Outline all pending tasks that you have explicitly been asked to work on,
     as well as list the next steps you will take for all outstanding work, if applicable. Include code
     snippets where they add clarity. For any next steps, include direct quotes from the most recent
     conversation showing exactly what task you were working on and where you left off. This should be
     verbatim to ensure there's no information loss in context between tasks.

Example summary structure:
1. Previous Conversation:
  [Detailed description]
2. Current Work:
  [Detailed description]
3. Key Technical Concepts:
  - [Concept 1]
  - [Concept 2]
  - [...]
4. Relevant Files and Code:
  - [File Name 1]
    - [Summary of why this file is important]
    - [Summary of the changes made to this file, if any]
    - [Important Code Snippet]
  - [File Name 2]
    - [Important Code Snippet]
  - [...]
5. Problem Solving:
  [Detailed description]
6. Pending Tasks and Next Steps:
  - [Task 1 details & next steps]
  - [Task 2 details & next steps]
  - [...]

Output only the summary of the conversation so far, without any additional commentary or explanation.
`;

/** Opaque, invocation-local boundary returned by Runtime.markMessages(). */
export class MessageMarker {
  /** @internal */
  constructor(
    readonly owner: object,
    readonly version: number,
    readonly prefix: readonly Message[],
  ) {}
}

/** Messages between markers; omit stop to include everything after start. */
export class MessageRange {
  constructor(
    readonly start: MessageMarker,
    readonly stop: MessageMarker | null = null,
  ) {}
}

function positiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 1) throw new Error(`${name} must be a positive integer`);
}

/**
 * Preserve the first user message and the most recent user-initiated turns.
 *
 * A turn starts at a user message and includes all following assistant/tool
 * exchanges up to the next user message. Runtime.say() also starts such a turn.
 */
export class OlderHistory {
  constructor(readonly keepRecentTurns = 3) {
    positiveInteger(keepRecentTurns, "keepRecentTurns");
  }
}

/**
 * Select everything between the first message and the last N messages.
 *
 * Counts messages, not user turns, so it also works for a long tool-using run
 * that has a single user message (where OlderHistory finds nothing). The system
 * prompt and the first message are kept. The recent tail is extended backwards
 * so a tool call is never separated from its results. Compaction skips when the
 * tail already contains a summary, or when fewer than 2 messages would be
 * replaced; an earlier summary inside the selection is folded into the new one.
 */
export class OlderMessages {
  constructor(readonly keepRecentMessages = 3) {
    positiveInteger(keepRecentMessages, "keepRecentMessages");
  }
}

export type Selection = MessageRange | OlderHistory | OlderMessages;

/**
 * A pinned message is never removed by compaction: a task's follow-up, a rule
 * that must stay in view. Only user messages can be pinned. When a compacted
 * region holds pinned messages, they are kept, in order, after its replacement.
 */
export function isPinned(item: Message): boolean {
  return item.role === "user" && item.metadata.pinned === true;
}

export function isSummary(item: Message): boolean {
  const compaction = item.metadata.compaction;
  return (
    (compaction !== null && typeof compaction === "object" && !Array.isArray(compaction) && compaction.kind === "summary") ||
    item.metadata.isSummary === true
  );
}

export type SummaryGenerator = (
  prompt: string,
  options?: { provider?: Provider; startIndex?: number },
) => Promise<ModelResponse>;

export type TokenCounter = (messages: readonly Message[]) => Promise<number>;

/**
 * Detached source messages and a budgeted, tool-free summary request function.
 *
 * Await generateSummary inside compact(); it is valid only for this operation.
 * The callback summarizes the original selected snapshot, not later mutations.
 * startIndex optionally limits its input to a suffix of that snapshot.
 */
export interface CompactionRequest {
  readonly messages: readonly Message[];
  readonly generateSummary: SummaryGenerator;
}

export interface CompactionProposal {
  readonly messages: readonly Message[];
}

export interface CompactionStrategy {
  compact(request: CompactionRequest): Promise<CompactionProposal>;
}

/** Discard the selected region without a model call. */
export class Remove implements CompactionStrategy {
  async compact(): Promise<CompactionProposal> {
    return { messages: [] };
  }
}

/** Retain an already generated handoff without another model call. */
export class Replace implements CompactionStrategy {
  constructor(readonly text: string) {
    if (typeof text !== "string" || !text.trim()) {
      throw new Error("Replacement text must be nonempty; use Remove() to discard messages");
    }
  }

  async compact(): Promise<CompactionProposal> {
    return {
      messages: [message({ role: "assistant", content: this.text, metadata: { compaction: { kind: "replacement" } } })],
    };
  }
}

/**
 * Summarize from the latest existing summary, using a configurable prompt.
 *
 * Older messages before that summary are replaced too, but are not sent again
 * to the summarizer. The provider defaults to the run's provider.
 */
export class Summarize implements CompactionStrategy {
  readonly prompt: string;
  readonly provider: Provider | undefined;

  constructor(options: { prompt?: string; provider?: Provider } = {}) {
    this.prompt = options.prompt ?? SUMMARY_PROMPT;
    this.provider = options.provider;
    if (typeof this.prompt !== "string" || !this.prompt.trim()) throw new Error("Summarization prompt must be nonempty");
  }

  async compact(request: CompactionRequest): Promise<CompactionProposal> {
    let start = 0;
    for (let i = request.messages.length - 1; i >= 0; i--) {
      if (isSummary(request.messages[i])) {
        start = i;
        break;
      }
    }
    const response = await request.generateSummary(this.prompt, { provider: this.provider, startIndex: start });
    const text = response.text.trim();
    if (!text) throw new Error("Failed to generate a nonempty summary");
    // Provider-specific signatures or thinking blocks belong to the summary
    // request, not the main conversation, and must not be replayed there.
    return { messages: [message({ role: "assistant", content: text, metadata: { compaction: { kind: "summary" } } })] };
  }
}

export interface CompactionResult {
  readonly applied: boolean;
  readonly reason: string;
  readonly strategy: string;
  readonly start: number;
  readonly stop: number;
  readonly replacement: readonly Message[];
  readonly beforeTokens: number | null;
  readonly afterTokens: number | null;
}
