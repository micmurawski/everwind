import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import {
  Agent,
  Cancelled,
  END,
  Flow,
  type Loop,
  type Message,
  PlanThenAct,
  ReAct,
  type Runtime,
  ScriptedProvider,
  type StreamChunk,
  Tools,
  tool,
} from "../src/index.ts";
import { call, tmpDir } from "./helpers.ts";

const add = tool({
  name: "add",
  description: "Add two integers.",
  input: z.object({ a: z.number().int(), b: z.number().int() }),
  run: ({ a, b }) => a + b,
});

const c1 = (args: Record<string, unknown>, name = "add", id = "c1") => call(id, name, args);

/** Never resolves until the signal aborts. */
const forever = (signal?: AbortSignal) =>
  new Promise<never>((_, reject) => signal?.addEventListener("abort", () => reject(signal.reason), { once: true }));

test("ReAct records the tool turn, usage and events", async () => {
  const agent = new Agent({
    provider: new ScriptedProvider([
      { toolCalls: [c1({ a: 2, b: 3 })], usage: { inputTokens: 10 } },
      { text: "5", usage: { inputTokens: 15, outputTokens: 2 } },
    ]),
    tools: new Tools([add]),
  });
  const run = await agent.run("Add 2 and 3");
  expect([run.status, run.output, run.modelCalls, run.toolCalls]).toEqual(["completed", "5", 2, 1]);
  expect(run.usage).toEqual({ inputTokens: 25, outputTokens: 2 });
  expect(run.messages.map((item) => item.role)).toEqual(["system", "user", "assistant", "tool", "assistant"]);
  expect([run.messages[3].toolCallId, run.messages[3].content]).toEqual(["c1", "5"]);
  const path = join(tmpDir(), "events.jsonl");
  run.writeEvents(path);
  const events = readFileSync(path, "utf-8").trim().split("\n").map((line) => JSON.parse(line));
  expect(events.map((event) => event.sequence)).toEqual(events.map((_, index) => index));
  expect(events.at(-1).data.status).toBe("completed");
  expect(new Set(events.map((event) => event.runId))).toEqual(new Set([run.id]));
});

test("PlanThenAct disables tools for the plan", async () => {
  const provider = {
    async *createMessage(_system: string, messages: Message[], metadata?: { turn?: number; purpose?: string }, options?: { tools?: readonly { name: string }[] }) {
      if (metadata?.turn === 1) {
        expect([metadata.purpose, options?.tools]).toEqual(["plan", []]);
        yield { type: "text", text: "Look up the answer." } as StreamChunk;
      } else {
        expect(options?.tools?.[0].name).toBe("add");
        expect(messages.some((item) => item.content === "Look up the answer.")).toBe(true);
        yield { type: "text", text: "Done" } as StreamChunk;
      }
    },
  };
  const run = await new Agent({ provider, tools: new Tools([add]), loop: new PlanThenAct() }).run("Task");
  expect([run.status, run.state.plan, run.modelCalls]).toEqual(["completed", "Look up the answer.", 2]);
});

test("context is not model input, and nullable values are accepted", async () => {
  const observed: unknown[] = [];
  const lookup = tool({
    name: "lookup",
    input: z.object({ value: z.string().nullable() }),
    run: ({ value }, ctx) => {
      observed.push([value, ctx.get("cwd")]);
      return value;
    },
  });
  expect(Object.keys(lookup.spec.parameters.properties as object)).toEqual(["value"]);
  const agent = new Agent({
    tools: new Tools([lookup]),
    provider: new ScriptedProvider([
      { toolCalls: [c1({ value: null, cwd: "untrusted" }, "lookup")] },
      { toolCalls: [c1({ value: null }, "lookup", "c2")] },
      { text: "Done" },
    ]),
  });
  const run = await agent.run("Task", { context: { cwd: "trusted", secret: "do-not-log" } });
  expect(run.status).toBe("completed");
  expect(observed).toEqual([[null, "trusted"]]);
  expect(run.messages[3].isError).toBe(true);
  expect(run.messages[5].content).toBe("null");
  expect(JSON.stringify(run.events)).not.toContain("do-not-log");
});

for (const args of [{ a: "2", b: 3 }, { a: 1 }, { a: 1, b: 2, extra: 3 }]) {
  test(`invalid arguments never execute: ${JSON.stringify(args)}`, async () => {
    const result = await add.invoke(args);
    expect(result.error).toContain("Invalid arguments");
  });
}

test("a missing context value fails the run", async () => {
  const lookup = tool({ name: "lookup", run: (_args, ctx) => ctx.get<string>("cwd") });
  const run = await new Agent({ provider: new ScriptedProvider([{ toolCalls: [c1({}, "lookup")] }]), tools: new Tools([lookup]) }).run("Task");
  expect(run.status).toBe("failed");
  expect(run.error).toContain("injected context 'cwd'");
});

test("a tool exception is an observation, without retry", async () => {
  let attempts = 0;
  const broken = tool({
    name: "broken",
    run: () => {
      attempts += 1;
      throw new Error("Something broke");
    },
  });
  const run = await new Agent({
    provider: new ScriptedProvider([{ toolCalls: [c1({}, "broken")] }, { text: "Reported the failure" }]),
    tools: new Tools([broken]),
  }).run("Task");
  expect([run.status, attempts, run.messages[3].isError]).toEqual(["completed", 1, true]);
  expect(run.messages[3].content).toContain("Something broke");
});

test("an unknown tool is an observation", async () => {
  const run = await new Agent({ provider: new ScriptedProvider([{ toolCalls: [c1({}, "missing")] }, { text: "No tools" }]) }).run("Task");
  expect(run.status).toBe("completed");
  expect(run.messages[3].isError).toBe(true);
  expect(run.messages[3].content).toContain("Unknown tool");
});

test("concurrent runs have independent state and script positions", async () => {
  const echo = tool({
    name: "echo",
    run: async (_args, ctx) => {
      await Promise.resolve();
      return ctx.get<string>("value");
    },
  });
  const agent = new Agent({ provider: new ScriptedProvider([{ toolCalls: [c1({}, "echo")] }, { text: "Done" }]), tools: new Tools([echo]) });
  const [first, second] = await Promise.all([agent.run("A", { context: { value: "a" } }), agent.run("B", { context: { value: "b" } })]);
  expect([first.status, second.status]).toEqual(["completed", "completed"]);
  expect(first.id).not.toBe(second.id);
  expect([first.messages[3].content, second.messages[3].content]).toEqual(["a", "b"]);
  first.state.changed = true;
  expect(second.state).toEqual({});
});

test("parallel tools are bounded and results keep call order", async () => {
  let active = 0;
  let peak = 0;
  const echo = tool({
    name: "echo",
    input: z.object({ value: z.number().int() }),
    run: async ({ value }) => {
      active += 1;
      peak = Math.max(peak, active);
      await Bun.sleep(5 * (4 - value));
      active -= 1;
      return value;
    },
  });
  const run = await new Agent({
    provider: new ScriptedProvider([{ toolCalls: [0, 1, 2, 3].map((i) => call(String(i), "echo", { value: i })) }, { text: "Done" }]),
    tools: new Tools([echo]),
    loop: new ReAct({ toolConcurrency: 2 }),
  }).run("Task");
  expect([run.status, peak]).toEqual(["completed", 2]);
  expect(run.messages.filter((item) => item.role === "tool").map((item) => item.content)).toEqual(["0", "1", "2", "3"]);
});

test("the model limit applies", async () => {
  const run = await new Agent({
    provider: new ScriptedProvider([{ toolCalls: [c1({ a: 1, b: 2 })] }, { text: "Must not be reached" }]),
    tools: new Tools([add]),
    limits: { maxModelCalls: 1 },
  }).run("Task");
  expect([run.status, run.modelCalls]).toEqual(["limit", 1]);
});

test("the tool budget refuses the entire batch", async () => {
  const run = await new Agent({
    provider: new ScriptedProvider([{ toolCalls: [c1({ a: 1, b: 2 }), call("c2", "add", { a: 2, b: 3 })] }]),
    tools: new Tools([add]),
    limits: { maxToolCalls: 1 },
  }).run("Task");
  expect([run.status, run.toolCalls]).toEqual(["limit", 0]);
  expect(run.events.some((event) => event.kind === "tool_start")).toBe(false);
});

test("a bad transition fails explicitly, and a cycle hits the step limit", async () => {
  const bad = new Flow({ start: "route", nodes: { route: () => "typo" }, edges: { route: { done: END } } });
  const failed = await new Agent({ provider: new ScriptedProvider([]), loop: bad }).run("Task");
  expect(failed.status).toBe("failed");
  expect(failed.error).toContain("No transition");
  const cycle = new Flow({ start: "repeat", nodes: { repeat: () => "next" }, edges: { repeat: { next: "repeat" } } });
  const limited = await new Agent({ provider: new ScriptedProvider([]), loop: cycle, limits: { maxSteps: 3 } }).run("Task");
  expect([limited.status, limited.steps]).toEqual(["limit", 3]);
});

for (const concurrency of [1, 2]) {
  test(`a tool timeout aborts the call and never retries (concurrency ${concurrency})`, async () => {
    let attempts = 0;
    let aborted = 0;
    const waitForever = tool({
      name: "wait_forever",
      run: async (_args, ctx) => {
        attempts += 1;
        try {
          await forever(ctx.signal);
        } finally {
          aborted += 1;
        }
      },
    });
    const run = await new Agent({
      provider: new ScriptedProvider([{ toolCalls: [c1({}, "wait_forever")] }]),
      tools: new Tools([waitForever]),
      loop: new ReAct({ toolConcurrency: concurrency }),
      limits: { toolTimeout: 0.01 },
    }).run("Task");
    expect(run.status).toBe("timeout");
    expect([attempts, aborted]).toEqual([1, 1]);
  });
}

test("the run deadline and caller cancellation", async () => {
  const provider = {
    async *createMessage(_s: string, _m: Message[], _meta?: unknown, options?: { signal?: AbortSignal }) {
      yield { type: "text", text: "Starting" } as StreamChunk;
      await forever(options?.signal);
    },
  };
  const timedOut = await new Agent({ provider, limits: { timeout: 0.01 } }).run("Task");
  expect(timedOut.status).toBe("timeout");
  const controller = new AbortController();
  const pending = new Agent({ provider }).run("Task", { signal: controller.signal });
  await Bun.sleep(5);
  controller.abort();
  await expect(pending).rejects.toThrow(Cancelled);
});

test("a provider failure is not success", async () => {
  const provider = {
    async *createMessage() {
      yield { type: "text", text: "Starting" } as StreamChunk;
      throw new Error("offline");
    },
  };
  const run = await new Agent({ provider }).run("Task");
  expect([run.status, run.output]).toEqual(["failed", null]);
  expect(run.error).toContain("offline");
});

test("duplicate tool call IDs are rejected before tools run", async () => {
  const run = await new Agent({
    provider: new ScriptedProvider([{ toolCalls: [c1({ a: 1, b: 2 }), c1({ a: 3, b: 4 })] }]),
    tools: new Tools([add]),
  }).run("Task");
  expect(run.status).toBe("failed");
  expect(run.error).toContain("duplicate");
  expect(run.toolCalls).toBe(0);
});

test("an incomplete loop does not report success", async () => {
  const flow = new Flow({ start: "empty", nodes: { empty: () => "done" }, edges: { empty: { done: END } } });
  const run = await new Agent({ provider: new ScriptedProvider([]), loop: flow }).run("Task");
  expect(run.status).toBe("failed");
  expect(run.error).toContain("without runtime.finish");
});

for (const operation of ["complete", "finish", "say"]) {
  test(`a custom loop cannot skip tool results (${operation})`, async () => {
    const loop: Loop = {
      async run(runtime: Runtime) {
        await runtime.complete();
        if (operation === "complete") await runtime.complete();
        else if (operation === "finish") runtime.finish("Skipped");
        else runtime.say("Skipped");
      },
    };
    const run = await new Agent({ provider: new ScriptedProvider([{ toolCalls: [c1({ a: 1, b: 2 })] }]), tools: new Tools([add]), loop }).run("Task");
    expect(run.status).toBe("failed");
    expect(run.error).toContain("pending tool calls");
  });
}

test("history is copied, and incomplete history is rejected", async () => {
  const history = [{ role: "assistant" as const, content: "Before", metadata: { nested: [1] } }];
  const run = await new Agent({ provider: new ScriptedProvider([{ text: "After" }]) }).run("Task", { history });
  (run.messages[1].metadata.nested as number[]).push(2);
  expect(history[0].metadata).toEqual({ nested: [1] });
  await expect(
    new Agent({ provider: new ScriptedProvider([]) }).run("Task", { history: [{ role: "assistant", toolCalls: [c1({ a: 1, b: 2 })] }] }),
  ).rejects.toThrow(/unresolved/);
});

test("tool composition is ordered and checks duplicates", () => {
  const lookup = tool({ name: "lookup", tags: ["read"], run: () => "ok" });
  const combined = Tools.of(add, lookup, add);
  expect(combined.specs.map((spec) => spec.name)).toEqual(["add", "lookup"]);
  expect(combined.select("read").specs.map((spec) => spec.name)).toEqual(["lookup"]);
  expect(() => combined.with(tool({ name: "add", run: () => 1 }))).toThrow(/Duplicate/);
});
