import { expect, test } from "bun:test";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { Agent, Cancelled, type Message, ScriptedProvider, SessionLog, SessionLogError, type StreamChunk, Tools, tool } from "../src/index.ts";
import { call, tmpDir } from "./helpers.ts";

const records = (path: string) => readFileSync(path, "utf-8").trim().split("\n").map((line) => JSON.parse(line));

test("the session log contains metadata, conversation, tool events and outcome", async () => {
  const directory = tmpDir();
  const add = tool({ name: "add", input: z.object({ a: z.number().int(), b: z.number().int() }), run: ({ a, b }) => a + b });
  const provider = new ScriptedProvider([{ toolCalls: [call("call-1", "add", { a: 2, b: 3 })] }, { text: "5", usage: { outputTokens: 2 } }]);
  const run = await new Agent({ provider, tools: new Tools([add]), logDir: join(directory, "nested"), name: "chef" }).run("Add 2 and 3\nPlease", {
    history: [{ role: "user", content: "Earlier" }],
    context: { credential: "not-automatically-logged" },
    sessionId: "dinner-01",
  });
  expect(run.status).toBe("completed");
  expect(run.sessionId).toBe("dinner-01");
  expect(run.id).not.toBe(run.sessionId);
  expect(run.logPath).toBe(join(directory, "nested", "everwind-dinner-01.jsonl"));
  const data = records(run.logPath as string);
  expect(data.length).toBe(run.events.length);
  expect(data.map((row) => row.sequence)).toEqual(data.map((_, index) => index));
  expect(data.every((row) => row.sessionId === run.sessionId && row.runId === run.id && row.schemaVersion === 1 && row.timestamp)).toBe(true);
  const header = data[0];
  expect([header.kind, header.data.agent, header.data.provider]).toEqual(["run_start", "chef", "ScriptedProvider"]);
  expect(header.data.cwd).toBeTruthy();
  expect(header.data.messages.at(-1).content).toBe("Add 2 and 3\nPlease");
  expect(header.data.messages[1].content).toBe("Earlier");
  const kinds = new Set(data.map((row) => row.kind));
  for (const kind of ["model_request", "model_chunk", "model_response", "tool_start", "tool_result"]) expect(kinds.has(kind)).toBe(true);
  expect([data.at(-1).kind, data.at(-1).data.status]).toEqual(["run_end", "completed"]);
  expect(data.at(-1).data.usage).toEqual({ outputTokens: 2 });
  expect(readFileSync(run.logPath as string, "utf-8")).not.toContain("not-automatically-logged");
  const exported = join(directory, "snapshot.jsonl");
  run.writeEvents(exported);
  expect(records(exported)).toEqual(data);
});

test("chunks are written before the turn finishes, and cancellation is logged", async () => {
  const directory = tmpDir();
  let ready!: () => void;
  const started = new Promise<void>((resolve) => (ready = resolve));
  const provider = {
    async *createMessage(_s: string, _m: Message[], _meta?: unknown, options?: { signal?: AbortSignal }) {
      yield { type: "text", text: "First token" } as StreamChunk;
      ready();
      await new Promise((_, reject) => options?.signal?.addEventListener("abort", () => reject(options.signal?.reason)));
    },
  };
  const controller = new AbortController();
  const pending = new Agent({ provider, logDir: directory }).run("Task", { sessionId: "live", signal: controller.signal });
  await started;
  await Bun.sleep(1);
  const path = join(directory, "everwind-live.jsonl");
  const before = records(path);
  expect(before.at(-1).kind).toBe("model_chunk");
  expect(before.at(-1).data.chunk.text).toBe("First token");
  expect(before.some((row) => row.kind === "run_end")).toBe(false);
  controller.abort();
  await expect(pending).rejects.toThrow(Cancelled);
  expect(records(path).at(-1).data.status).toBe("cancelled");
});

for (const outcome of ["failed", "timeout", "limit"] as const) {
  test(`${outcome} runs have final records`, async () => {
    const directory = tmpDir();
    const broken = {
      async *createMessage(_s: string, _m: Message[], _meta?: unknown, options?: { signal?: AbortSignal }) {
        yield { type: "text", text: "Partial" } as StreamChunk;
        if (outcome === "failed") throw new Error("offline");
        await new Promise((_, reject) => options?.signal?.addEventListener("abort", () => reject(options.signal?.reason)));
      },
    };
    const agent =
      outcome === "limit"
        ? new Agent({ provider: new ScriptedProvider([{ toolCalls: [call("a", "unknown")] }]), limits: { maxToolCalls: 0 }, logDir: directory })
        : new Agent({ provider: broken, limits: { timeout: 0.01 }, logDir: directory });
    const run = await agent.run("Task");
    expect(run.status).toBe(outcome);
    expect(records(run.logPath as string).at(-1).data.status).toBe(outcome);
  });
}

test("concurrent sessions have separate files, and a run can override the directory", async () => {
  const directory = tmpDir();
  const agent = new Agent({ provider: new ScriptedProvider([{ text: "Done" }]), logDir: join(directory, "default") });
  const [first, second] = await Promise.all([agent.run("A"), agent.run("B", { logDir: join(directory, "override") })]);
  expect(first.id).not.toBe(second.id);
  expect(join(first.logPath as string, "..")).not.toBe(join(second.logPath as string, ".."));
  for (const run of [first, second]) {
    expect(new Set(records(run.logPath as string).map((row) => row.sessionId))).toEqual(new Set([run.sessionId]));
    expect((run.logPath as string).endsWith(`everwind-${run.sessionId}.jsonl`)).toBe(true);
  }
});

test("logging is opt-in", async () => {
  const run = await new Agent({ provider: new ScriptedProvider([{ text: "Done" }]) }).run("Task");
  expect(run.logPath).toBeNull();
});

test("an existing session log is not overwritten", async () => {
  const directory = tmpDir();
  const path = join(directory, "everwind-existing.jsonl");
  writeFileSync(path, "original\n");
  await expect(new Agent({ provider: new ScriptedProvider([]), logDir: directory }).run("Task", { sessionId: "existing" })).rejects.toThrow(/Cannot create/);
  expect(readFileSync(path, "utf-8")).toBe("original\n");
});

for (const sessionId of ["../escape", "a/b", "a\\b", "", ".", "a".repeat(129)]) {
  test(`an invalid session ID does not create a file: ${JSON.stringify(sessionId).slice(0, 20)}`, async () => {
    const directory = tmpDir();
    await expect(new Agent({ provider: new ScriptedProvider([]), logDir: directory }).run("Task", { sessionId })).rejects.toThrow(/sessionId/);
    expect(readdirSync(directory)).toEqual([]);
  });
}

test("a log creation failure happens before the provider call", async () => {
  const directory = tmpDir();
  const file = join(directory, "file");
  writeFileSync(file, "not a directory");
  await expect(new Agent({ provider: new ScriptedProvider([]), logDir: file }).run("Task")).rejects.toThrow(/Cannot create/);
});

test("a write failure is visible and the file is closed", async () => {
  const directory = tmpDir();
  const original = SessionLog.prototype.write;
  const logs: SessionLog[] = [];
  SessionLog.prototype.write = function (this: SessionLog, event) {
    if (event.kind === "model_chunk") {
      logs.push(this);
      throw new SessionLogError("disk unavailable");
    }
    original.call(this, event);
  };
  try {
    await expect(new Agent({ provider: new ScriptedProvider([{ text: "Done" }]), logDir: directory }).run("Task")).rejects.toThrow(/disk unavailable/);
  } finally {
    SessionLog.prototype.write = original;
  }
  expect(logs.length).toBe(1);
  expect(logs[0].closed).toBe(true);
});
