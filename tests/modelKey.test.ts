import { expect, test } from "bun:test";
import { z } from "zod";
import {
  Agent,
  type JsonObject,
  MemoryStore,
  MemoryStoreData,
  NonDeterminismError,
  ScriptedProvider,
  type Tool,
  Tools,
  commandKey,
  tool,
} from "../src/index.ts";
import { Crash, call } from "./helpers.ts";

/** read_notes crashes the process once, after the first model call is journaled. */
function crashingTools(variant: "original" | "edited") {
  let crash = variant === "original";
  const notes = tool({
    name: "read_notes",
    description: variant === "original" ? "Read the notes." : "Read the project notes, newest first.",
    effect: "read",
    input: variant === "original" ? z.object({}) : z.object({ limit: z.number().int().optional() }),
    run: () => {
      if (crash) {
        crash = false;
        throw new Crash();
      }
      return "notes";
    },
  });
  const lookup = tool({ name: "lookup", effect: "read", input: z.object({ key: z.string() }), run: ({ key }) => key });
  return { notes, lookup };
}

function agent(store: MemoryStore, tools: Tool[]): Agent {
  return new Agent({
    provider: new ScriptedProvider([{ toolCalls: [call("n", "read_notes")] }, { text: "done" }]),
    tools: new Tools(tools),
    store,
    name: "worker",
    leaseTtl: 60,
  });
}

async function crashedRun(): Promise<MemoryStoreData> {
  const data = new MemoryStoreData();
  const { notes, lookup } = crashingTools("original");
  await expect(agent(new MemoryStore(data), [notes, lookup]).run("Go", { sessionId: "s1" })).rejects.toThrow(Crash);
  expect(new MemoryStore(data).runs()[0].status).toBe("running");
  return data;
}

test("a changed tool description or schema does not break the replay of recorded model calls", async () => {
  const data = await crashedRun();
  const { notes, lookup } = crashingTools("edited");
  // Order does not matter either: keys use the sorted names.
  const run = await agent(new MemoryStore(data), [lookup, notes]).recover("s1", { force: true, allowFingerprintChange: true });
  expect(run.status).toBe("completed");
  expect(run.output).toBe("done");
  // The live call after recovery offers the edited spec; the recorded one kept its own.
  const specs = run.events.filter((event) => event.kind === "model_request").map((event) => (event.data.request as any).tools);
  expect(specs.at(-1).find((spec: any) => spec.name === "read_notes").description).toBe("Read the project notes, newest first.");
});

test("an added or removed tool still diverges", async () => {
  const data = await crashedRun();
  const { notes } = crashingTools("edited");
  await expect(agent(new MemoryStore(data), [notes]).recover("s1", { force: true, allowFingerprintChange: true })).rejects.toThrow(
    NonDeterminismError,
  );
  expect(new MemoryStore(data).runs()[0].status).toBe("running");
});

test("journals keyed on full tool specs (before this change) still recover", async () => {
  const data = await crashedRun();
  // Rewrite the recorded model rows to the old key: full specs, as earlier versions wrote them.
  const store = new MemoryStore(data);
  const request = store.events("s1").find((stored) => stored.event.kind === "model_request")?.event.data.request as JsonObject;
  const { runId: _, ...rest } = request;
  const legacy = commandKey("model", rest);
  let rewritten = 0;
  for (const entries of data.entries.values()) {
    for (const [position, entry] of entries) {
      if (entry.kind !== "model") continue;
      entries.set(position, { ...entry, key: legacy });
      rewritten += 1;
    }
  }
  expect(rewritten).toBe(2); // scheduled and completed
  // The original specs, so the legacy key matches; this read_notes does not crash.
  const { notes, lookup } = crashingTools("original");
  const calm = tool({ name: "read_notes", description: notes.description, effect: "read", run: () => "notes" });
  const run = await agent(new MemoryStore(data), [calm, lookup]).recover("s1", { force: true });
  expect(run.status).toBe("completed");
  expect(run.output).toBe("done");
});
