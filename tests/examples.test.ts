/** The examples run offline and keep demonstrating what they claim. */

import { expect, test } from "bun:test";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import type { Message, StreamChunk } from "../src/index.ts";
import { DockerPythonSandbox, WORKER_PATH } from "../src/python/index.ts";
import { demo as delegatingDemo } from "../examples/delegatingChef.ts";
import { Kitchen, demo as kitchenDemo } from "../examples/kitchenChef.ts";
import { main as liveChat } from "../examples/liveChat.ts";
import { makeAgent as planningAgent } from "../examples/planningCompaction.ts";
import { demo as pythonDemo } from "../examples/pythonRlm.ts";
import { Mailbox } from "../src/index.ts";
import { tmpDir } from "./helpers.ts";

test("the chef handles messages and is woken by every job finishing", async () => {
  const transcript: string[] = [];
  const run = await kitchenDemo({ report: (line) => transcript.push(line) });
  expect(run.error).toBeNull();
  expect(run.status).toBe("completed");
  const jobs = Object.fromEntries((run.state.jobs as { dish: string; status: string; exitCode: number }[]).map((job) => [job.dish, job]));
  expect([jobs.pasta.status, jobs.sauce.status, jobs.toast.status]).toEqual(["succeeded", "succeeded", "killed"]);
  expect(jobs.toast.exitCode).not.toBe(0);
  const observations = run.events.filter((event) => event.kind === "mailbox_event").map((event) => event.data.event as Record<string, unknown>);
  const finished = observations.filter((event) => event.kind === "JobFinished");
  expect(finished.length).toBe(3);
  expect(new Set(finished.map((event) => event.jobId)).size).toBe(3);
  expect(observations.at(-1)?.kind).toBe("Shutdown");
  for (const phrase of ["Dinner status:", "Plated pasta after its completion event", "Plated sauce after its completion event", "Observed toast killed"]) {
    expect(transcript.some((line) => line.includes(phrase))).toBe(true);
  }
  // Completion triggers a new model turn even though no further user message is sent.
  run.events.forEach((event, index) => {
    if (event.kind === "mailbox_event" && (event.data.event as { kind: string }).kind === "JobFinished") {
      expect(run.events.slice(index + 1).some((later) => later.kind === "model_request")).toBe(true);
    }
  });
});

for (const fail of [false, true]) {
  test(`a completion is queued even when the agent is not listening (fail=${fail})`, async () => {
    const mailbox = new Mailbox();
    const kitchen = new Kitchen(mailbox);
    try {
      const handle = kitchen.start("soup", 0.01, { fail });
      expect(handle.status).toBe("running");
      await kitchen.waitAll();
      const events: Record<string, unknown>[] = [];
      while (mailbox.size) events.push(await mailbox.get());
      const finished = events.filter((event) => event.kind === "JobFinished");
      expect(finished.length).toBe(1);
      expect([finished[0].status, finished[0].jobId]).toEqual([fail ? "failed" : "succeeded", handle.jobId]);
    } finally {
      await kitchen.close();
    }
  });
}

for (const [force, expected] of [[false, "cancelled"], [true, "killed"]] as const) {
  test(`a stop is acknowledged, then confirmed by one completion event (${expected})`, async () => {
    const mailbox = new Mailbox();
    const kitchen = new Kitchen(mailbox);
    try {
      const handle = kitchen.start("slow roast", 30);
      expect(kitchen.stop(handle.jobId, { force }).status).toBe("stopping");
      await kitchen.waitAll();
      expect(kitchen.inspect(handle.jobId).status).toBe(expected);
      kitchen.stop(handle.jobId, { force }); // A repeated stop emits no other outcome.
      const events: Record<string, unknown>[] = [];
      while (mailbox.size) events.push(await mailbox.get());
      const finished = events.filter((event) => event.kind === "JobFinished");
      expect(finished.length).toBe(1);
      expect(finished[0].status).toBe(expected);
    } finally {
      await kitchen.close();
    }
  });
}

test("cleanup reaps background processes", async () => {
  const kitchen = new Kitchen(new Mailbox());
  kitchen.start("slow roast", 30);
  await kitchen.close();
  expect([...kitchen.jobs.values()].every((job) => job.process.exitCode !== null || job.process.signalCode !== null)).toBe(true);
  expect(() => kitchen.start("another dish", 1)).toThrow(/closed/);
});

test("the delegating chef wakes and continues the same child session", async () => {
  const directory = tmpDir();
  const run = await delegatingDemo({ report: () => {}, logDir: directory });
  expect(run.status).toBe("completed");
  expect(run.output).toContain("gluten-free");
  const events = run.events.filter((event) => event.kind === "mailbox_event").map((event) => event.data.event as Record<string, unknown>);
  expect(events.length).toBe(2);
  expect(events[0].sessionId).toBe(events[1].sessionId);
  expect(events[0].runId).not.toBe(events[1].runId);
  expect(events[0].jobId).not.toBe(events[1].jobId);
  expect(readdirSync(directory).filter((name) => name.endsWith(".jsonl")).length).toBe(2); // Parent and child.
});

for (const summarize of [false, true]) {
  test(`the planning example executes after compaction (summarize=${summarize})`, async () => {
    const run = await planningAgent({ summarize }).run("Process A, B, C, D with three workers.");
    expect([run.status, run.output]).toEqual(["completed", "Processed all four items in two batches."]);
    expect([run.modelCalls, run.toolCalls]).toEqual([summarize ? 9 : 8, 3]);
    expect(run.messages.some((item) => item.content.includes("Draft:"))).toBe(false);
    const event = run.events.find((item) => item.kind === "context_compaction");
    expect(event?.data.strategy).toBe(summarize ? "Summarize" : "Replace");
    expect((event?.data.removedMessages as Message[]).some((item) => item.content.includes("Draft:"))).toBe(true);
    expect("planningStart" in run.state).toBe(false);
  });
}

test("the live chat CLI prints a working continuation command", async () => {
  const directory = join(tmpDir(), "sessions with spaces");
  const offline = () => ({
    async *createMessage(_system: string, messages: Message[]): AsyncGenerator<StreamChunk> {
      yield { type: "text", text: messages.filter((item) => item.role === "user").map((item) => item.content).join(" | ") };
    },
  });
  const lines: string[] = [];
  const log = console.log;
  console.log = (line: string) => lines.push(String(line));
  try {
    const args = { provider: "openai" as const, model: "offline", sessionDir: directory, prompt: "First", sessionId: undefined as string | undefined };
    expect(await liveChat(args, offline)).toBe(0);
    const command = lines.find((line) => line.startsWith("To continue: ")) as string;
    expect(command).toContain(`'${directory}'`);
    args.sessionId = command.match(/--session-id (\S+)/)?.[1];
    args.prompt = "Second";
    await liveChat(args, offline);
    expect(lines.some((line) => line.includes("First | Second"))).toBe(true);
  } finally {
    console.log = log;
  }
});

class LocalWorker extends DockerPythonSandbox {
  protected override async createContainer(): Promise<void> {}
  protected override attachCommand(): string[] {
    return [Bun.which("python3") as string, "-I", "-u", WORKER_PATH];
  }
  protected override async removeContainer(): Promise<void> {}
}

test.skipIf(Bun.which("python3") === null)("the Python example uses persistent variables and exports an artifact", async () => {
  const directory = tmpDir();
  const run = await pythonDemo({ sandbox: (options) => new LocalWorker({ data: options.data, setup: `import os\nos.chdir(${JSON.stringify(directory)})` }) });
  expect(run.status).toBe("completed");
  expect(JSON.parse(readFileSync(join(directory, "summary.json"), "utf-8"))).toEqual({ matches: 2, document_ids: ["a", "c"] });
  expect(run.output).toContain('"matches": 2');
});
