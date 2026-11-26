/** Durable-run test harness: simulated crashes at every journal write, and a world of effects. */

import { expect } from "bun:test";
import { z } from "zod";
import {
  Agent,
  JournalError,
  type Loop,
  MemoryStore,
  MemoryStoreData,
  type ModelResponse,
  PlanThenAct,
  ReAct,
  ReActWithSummary,
  ReActWithWakeups,
  ScriptedProvider,
  SqliteStore,
  type DurableStore,
  type Message,
  type ModelResponseInput,
  type ToolCallInput,
  type Tool,
  Tools,
  scheduleWakeupTool,
  tool,
} from "../src/index.ts";
import type { Op } from "../src/journal.ts";

/** Simulated process death: nothing after it reaches the store, and the run is left as it is. */
export class Crash extends JournalError {
  constructor() {
    super("simulated crash");
  }
}

/** Await until the simulated crash. */
export async function crashes(promise: Promise<unknown>, { mayFinish = false } = {}): Promise<void> {
  try {
    await promise;
  } catch (error) {
    if (error instanceof Crash) return;
    throw error;
  }
  expect(mayFinish).toBe(true); // expected a simulated crash
}

interface Crashing {
  limit: number;
  written: number;
  dead: boolean;
}

/** Writes the first `crashAt - 1` operations (or `crashAt` with after), then dies. */
function crashingWrite(store: Crashing, ops: Op[], write: (ops: Op[]) => unknown[]): unknown[] {
  if (store.dead) throw new Crash();
  const allowed = store.limit - store.written;
  if (ops.length <= allowed) {
    store.written += ops.length;
    return write(ops);
  }
  write(ops.slice(0, allowed));
  store.written += allowed;
  store.dead = true;
  throw new Crash();
}

export class CrashingMemory extends MemoryStore implements Crashing {
  limit: number;
  written = 0;
  dead = false;

  constructor(data: MemoryStoreData, { crashAt, after }: { crashAt: number; after: boolean }) {
    super(data);
    this.limit = after ? crashAt : crashAt - 1;
  }

  protected override writeBatch(ops: Op[]): unknown[] {
    return crashingWrite(this, ops, (batch) => super.writeBatch(batch));
  }
}

export class CrashingSqlite extends SqliteStore implements Crashing {
  limit: number;
  written = 0;
  dead = false;

  constructor(path: string, { crashAt, after }: { crashAt: number; after: boolean }) {
    super(path);
    this.limit = after ? crashAt : crashAt - 1;
  }

  protected override writeBatch(ops: Op[]): unknown[] {
    return crashingWrite(this, ops, (batch) => super.writeBatch(batch));
  }
}

/** Counts committed operations of an uninterrupted reference run. */
export class CountingMemory extends MemoryStore {
  written = 0;

  protected override writeBatch(ops: Op[]): unknown[] {
    this.written += ops.length;
    return super.writeBatch(ops);
  }
}

/** External state that survives crashes: what tools really did. */
export class World {
  unsafe: string[] = [];
  idempotentKeys: string[] = [];
  reads = 0;
}

export function makeTools(world: World): Tools {
  const lookup = tool({
    name: "lookup",
    description: "Read a record.",
    effect: "read",
    input: z.object({ key: z.string() }),
    run: ({ key }) => {
      world.reads += 1;
      return `record ${key}`;
    },
  });
  const upsert = tool({
    name: "upsert",
    description: "Write a record with an idempotency key.",
    effect: "idempotent",
    input: z.object({ key: z.string() }),
    run: ({ key }, ctx) => {
      world.idempotentKeys.push(ctx.idempotencyKey);
      return `stored ${key}`;
    },
  });
  const send = tool({
    name: "send",
    description: "Send a message (unsafe: never repeated).",
    input: z.object({ text: z.string() }),
    run: ({ text }) => {
      world.unsafe.push(text);
      return "sent";
    },
  });
  return new Tools([lookup, upsert, send]);
}

export function call(id: string, name: string, args: Record<string, unknown> = {}): ToolCallInput {
  return { id, name, arguments: args as ToolCallInput["arguments"] };
}

export const REACT_SCRIPT: ModelResponseInput[] = [
  { toolCalls: [call("a", "lookup", { key: "x" }), call("b", "upsert", { key: "y" })], usage: { inputTokens: 3 } },
  { toolCalls: [call("c", "send", { text: "hello" })], usage: { inputTokens: 5 } },
  { text: "All done", usage: { inputTokens: 7 } },
];

export async function countTokens(messages: readonly Message[]): Promise<number> {
  return messages.reduce((total, item) => total + item.content.length + 1, 0);
}

export const SCENARIOS: Record<string, () => { loop: Loop; script: ModelResponseInput[]; tools?: Tool[] }> = {
  react: () => ({ loop: new ReAct(), script: REACT_SCRIPT }),
  react_parallel: () => ({
    loop: new ReAct({ toolConcurrency: 3 }),
    script: [
      { toolCalls: [call("a", "lookup", { key: "x" }), call("b", "upsert", { key: "y" }), call("c", "send", { text: "hello" })] },
      { text: "All done" },
    ],
  }),
  plan_then_act: () => ({ loop: new PlanThenAct(), script: [{ text: "Plan: lookup, send" }, ...REACT_SCRIPT] }),
  react_with_summary: () => ({
    loop: new ReActWithSummary({
      contextWindow: 400,
      tokenCounter: countTokens,
      maxOutputTokens: 10,
      thresholdPercent: 40,
      keepRecentMessages: 2,
    }),
    script: [
      { text: "Looking ".repeat(20), toolCalls: [call("a", "lookup", { key: "x" })] },
      { text: "Writing ".repeat(20), toolCalls: [call("b", "upsert", { key: "y" })] },
      { text: "Summary of earlier work" },
      { toolCalls: [call("c", "send", { text: "hello" })] },
      { text: "Second summary" },
      { text: "All done" },
    ],
  }),
  react_with_wakeups: () => ({
    loop: new ReActWithWakeups({ minDelay: 0 }),
    tools: [scheduleWakeupTool({ minDelay: 0 })],
    script: [
      { toolCalls: [call("a", "lookup", { key: "x" }), call("w", "schedule_wakeup", { delaySeconds: 0.02, prompt: "Check again" })] },
      { toolCalls: [call("c", "send", { text: "hello" })] },
      { text: "All done" },
    ],
  }),
};

export function build(name: string, store: DurableStore, world: World): Agent {
  const spec = SCENARIOS[name]();
  return new Agent({
    provider: new ScriptedProvider(spec.script),
    tools: makeTools(world).with(...(spec.tools ?? [])),
    loop: spec.loop,
    name: "worker",
    store,
    leaseTtl: 60,
  });
}

export function tmpDir(): string {
  const { mkdtempSync } = require("node:fs") as typeof import("node:fs");
  const { tmpdir } = require("node:os") as typeof import("node:os");
  const { join } = require("node:path") as typeof import("node:path");
  return mkdtempSync(join(tmpdir(), "everwind-test-"));
}

export type { ModelResponse };
