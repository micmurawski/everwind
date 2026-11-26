/** Self-paced agents: the model asks to be woken later, and a loop node sleeps durably until then. */

import { z } from "zod";
import type { Step } from "./flow.ts";
import { MESSAGE } from "./inbox.ts";
import type { Runtime } from "./runtime.ts";
import { type Tool, tool } from "./tools.ts";

export interface WakeupOptions {
  /** Name of the scheduling tool. Default "schedule_wakeup". */
  name?: string;
  /** Bounds a requested delay is clamped to, in seconds. Default 60 and 3600. */
  minDelay?: number;
  maxDelay?: number;
  /** Inbox kinds that end the sleep early. Default ["message"]: a note from you wakes the agent. */
  wakeOn?: readonly string[];
}

export interface WakeupRequest {
  delaySeconds: number;
  prompt: string;
  reason: string;
  noop: boolean;
}

function bounds(options: WakeupOptions): [number, number] {
  const min = options.minDelay ?? 60;
  const max = options.maxDelay ?? 3600;
  if (!(min >= 0 && min <= max && Number.isFinite(max))) throw new Error("Need 0 <= minDelay <= maxDelay");
  return [min, max];
}

const WakeupInput = z.object({
  delaySeconds: z.number().optional().describe("Seconds from now to wake up; clamped to the allowed range"),
  prompt: z.string().optional().describe("The instruction you receive on waking; pass the same one each time to repeat a task"),
  reason: z.string().optional().describe("One short sentence on why this delay"),
  noop: z.boolean().optional().describe("True when nothing changed since the last wakeup"),
  stop: z.boolean().optional().describe("End the loop: no further wakeups"),
});
type WakeupArgs = z.output<typeof WakeupInput>;

function parse(args: WakeupArgs, [min, max]: [number, number]): WakeupRequest | null {
  if (args.stop) return null;
  if (args.delaySeconds === undefined || !args.prompt) throw new Error("delaySeconds and prompt are required unless stop is true");
  return {
    delaySeconds: Math.min(Math.max(args.delaySeconds, min), max),
    prompt: args.prompt,
    reason: args.reason ?? "",
    noop: args.noop ?? false,
  };
}

/**
 * The tool a model calls to be woken later. It only records the request:
 * the wakeup node of the loop (wakeupStep) does the sleeping.
 */
export function scheduleWakeupTool(options: WakeupOptions = {}): Tool {
  const range = bounds(options);
  return tool({
    name: options.name ?? "schedule_wakeup",
    description:
      `Schedule when to resume this task: after this turn's tools run, you sleep for delaySeconds ` +
      `(clamped to ${range[0]}-${range[1]}) and then receive prompt as a new instruction. ` +
      "Match the delay to what you are waiting for. To end the loop, pass stop: true, or give your final answer " +
      "without calling this tool.",
    effect: "read",
    input: WakeupInput,
    run(args) {
      const request = parse(args, range);
      if (request === null) return { scheduled: false, stopped: true };
      return { scheduled: true, delaySeconds: request.delaySeconds, reason: request.reason };
    },
  });
}

/**
 * The wakeup the last model turn requested, if any (the last call wins).
 *
 * It is read from the conversation, not from tool-side state, so a replay
 * after a crash finds the same request without running the tool again.
 */
export function wakeupRequest(runtime: Runtime, options: WakeupOptions = {}): WakeupRequest | null {
  const name = options.name ?? "schedule_wakeup";
  const range = bounds(options);
  const messages = runtime.run.messages;
  let index = messages.length - 1;
  while (index >= 0 && messages[index].role === "tool") index -= 1;
  const assistant = messages[index];
  if (assistant === undefined || assistant.role !== "assistant") return null;
  const failed = new Set(messages.slice(index + 1).filter((item) => item.isError).map((item) => item.toolCallId));
  const calls = assistant.toolCalls.filter((call) => call.name === name && !failed.has(call.id));
  if (!calls.length) return null;
  const args = WakeupInput.strict().safeParse(calls[calls.length - 1].arguments);
  if (!args.success) return null;
  try {
    return parse(args.data, range);
  } catch {
    return null;
  }
}

/**
 * A loop node to put after tool execution: if the model scheduled a wakeup,
 * suspend durably until then and add the wakeup prompt as a user instruction.
 * A message of a `wakeOn` kind ends the sleep early; the prompt is then not
 * added, and the gate delivers the message instead. Without a request the node
 * does nothing. Either way it returns the default action.
 */
export function wakeupStep(options: WakeupOptions = {}): Step {
  bounds(options);
  const wakeOn = [...(options.wakeOn ?? [MESSAGE])];
  return async function wakeup(runtime: Runtime): Promise<void> {
    const request = wakeupRequest(runtime, options);
    if (request === null) return;
    runtime.run.emit("wakeup_scheduled", { ...request });
    const wake = await runtime.suspend({ seconds: request.delaySeconds, wakeOn, reason: request.reason });
    if (wake.by === "timer") runtime.say(request.prompt);
  };
}
