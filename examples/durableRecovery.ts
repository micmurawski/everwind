/**
 * Durable runs: journal every step in SQLite, crash, then recover without repeating effects.
 *
 * Run with: bun examples/durableRecovery.ts
 * The "crash" is simulated by stopping the store mid-run; the second phase is
 * what a restarted process does: open the same file and recover every run.
 */

import { rmSync } from "node:fs";
import { z } from "zod";
import { Agent, JournalError, ScriptedProvider, SharedWorld, SqliteStore, Tools, Worker, tool, type DurableStore } from "../src/index.ts";
import type { Op } from "../src/journal.ts";

const PATH = "artifacts/durable-example.db";
const sent: string[] = [];

const lookup = tool({ name: "lookup", description: "Read a record.", effect: "read", input: z.object({ key: z.string() }), run: ({ key }) => `record ${key}` });
const sendEmail = tool({
  name: "send_email",
  description: "Send an email (unsafe: never repeated after a crash).",
  input: z.object({ to: z.string() }),
  run: ({ to }) => {
    sent.push(to);
    return "sent";
  },
});

const script = [
  { toolCalls: [{ id: "a", name: "lookup", arguments: { key: "customer-7" } }] },
  { toolCalls: [{ id: "b", name: "send_email", arguments: { to: "customer-7" } }] },
  { text: "Emailed customer-7." },
];

function agent(store: DurableStore): Agent {
  return new Agent({ name: "mailer", provider: new ScriptedProvider(script), tools: new Tools([lookup, sendEmail]), store, shared: new SharedWorld(store) });
}

/** A store that dies after a number of writes, like a process killed mid-run. */
class Dying extends SqliteStore {
  constructor(path: string, private remaining: number) {
    super(path);
  }
  protected override writeBatch(ops: Op[]): unknown[] {
    if (ops.length > this.remaining) throw new JournalError("process killed");
    this.remaining -= ops.length;
    return super.writeBatch(ops);
  }
}

rmSync(PATH, { force: true });
const dying = new Dying(PATH, 9);
await agent(dying)
  .run("Email customer-7 their record.", { sessionId: "task-42" })
  .catch((error) => console.log(`First process: ${error.message}`));
dying.close();
console.log(`Emails sent before the crash: ${sent.length}`);

const store = new SqliteStore(PATH);
const [run] = await new Worker(store, [agent(store)]).recoverAll({ force: true });
if (run instanceof Error) throw run;
console.log(`Recovered: ${run.status}, "${run.output}"; emails sent in total: ${sent.length}`);
for (const stored of store.events("task-42")) console.log(`  ${String(stored.position).padStart(3)} ${stored.event.kind}`);
store.close();
