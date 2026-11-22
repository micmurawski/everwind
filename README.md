# tourbillon (tbln)

Small, explicit agent loops for TypeScript on [Bun](https://bun.sh), with durable runs
that survive crashes, replay deterministically and share state between agents.

- **Loops are code.** ReAct, plan-then-act and summarizing loops are short flows of
  nodes over one `Runtime`. Write your own with the same primitives.
- **Tools are typed.** Model input is a zod schema; everything else (the runtime, an
  idempotency key, application resources) comes from the call's context and never
  from the model.
- **Runs are durable.** With a store, every model call and tool call is journaled.
  After a crash, recovery replays the journal instead of repeating effects, like
  Temporal workflows.
- **State is shared.** Agents read and write one JSON tree with versions, optimistic
  writes and time travel.

No service is required: a run needs a provider, and durability needs one SQLite file.

```sh
bun install
bun test            # the full suite, offline
bun run typecheck
```

## A first agent

```ts
import { z } from "zod";
import { Agent, OpenAIChat, Tools, tool } from "tourbillon";

const multiply = tool({
  name: "multiply",
  description: "Multiply two integers.",
  effect: "read",
  input: z.object({ a: z.number().int(), b: z.number().int() }),
  run: ({ a, b }) => a * b,
});

const agent = new Agent({ provider: new OpenAIChat("gpt-4.1-mini"), tools: new Tools([multiply]) });
const run = await agent.run("What is 6 times 7?");
console.log(run.status, run.output, run.usage);
```

`run.messages`, `run.events`, `run.usage` and `run.state` describe what happened.
Offline, use `ScriptedProvider` with a list of responses.

Providers: `OpenAIChat` (any OpenAI-compatible endpoint; install `openai`), `MiniMax`
(install `@anthropic-ai/sdk`, set `MINIMAX_API_KEY`) and `ScriptedProvider`. A provider
is any object with an async generator `createMessage(systemPrompt, messages, metadata,
options)` that yields normalized chunks; honour `options.signal` so cancellation and
deadlines can stop a stream.

## Tools and context

```ts
const lookup = tool({
  name: "lookup",
  description: "Look up a customer.",
  effect: "read",                      // read | idempotent | unsafe (default)
  input: z.object({ id: z.string() }), // unknown keys and wrong types are rejected
  run: async ({ id }, ctx) => {
    const db = ctx.get<Database>("db"); // from agent.run(prompt, { context: { db } })
    return db.find(id, { signal: ctx.signal });
  },
});
```

- A thrown error becomes an error result the model can read. A missing context value
  fails the run instead.
- `ctx.signal` aborts when the tool's `limits.toolTimeout` passes or the run stops.
- `ctx.idempotencyKey` is stable across crash recovery; pass it to idempotent APIs.
- `ctx.runtime` and `ctx.shared` give runtime-aware tools access to the run.

## Loops and flows

```ts
import { END, Flow, finishStep, modelStep, node, toolsStep } from "tourbillon";

const callLlm = node(modelStep, { name: "call_llm" });
const tools = node(toolsStep, { name: "tools" });
const finish = node(finishStep, { name: "finish" });
callLlm.on("tools", tools).on("answer", finish); // a named action routes to a node
tools.then(callLlm);                              // the default action
finish.then(END);

const agent = new Agent({ provider, loop: new Flow(callLlm) });
console.log(agent.flowGraph()); // Mermaid; flow.toDot() and flow.render("flow.svg") too
```

A node returns an action string, or nothing for `"default"`. `ReAct`, `PlanThenAct` and
`ReActWithSummary` are ready-made loops. A loop is anything with
`run(runtime): Promise<void>`; the runtime offers `complete()`, `executeTools()`,
`say()`, `finish()`, `compact()`, and the budgets in `limits`.

## Context compaction

`runtime.compact({ selection, strategy, tokenCounter })` replaces a region of the
conversation. Selections are `MessageRange` (between `markMessages()` markers),
`OlderHistory` and `SummaryHistory`; strategies are `Remove`, `Replace(text)`,
`Summarize` (a budgeted, tool-free model call) or your own. A proposal that breaks tool
history, adds a system message or doesn't shrink the context (with a counter) is not
applied. `ReActWithSummary` summarizes before each model turn once a threshold is hit.

## Sessions and logs

`agent.run(prompt, { sessionId })` continues a conversation. Without a store, sessions
are kept in memory, or in `FileSessionStore(directory)`; `logDir` writes one JSONL log
per session (`tbln-<session>.jsonl`). Concurrent continuations of one session are
serialized. An interrupted tool call is closed with an "Outcome unknown" result rather
than repeated.

## Durable runs and crash recovery

```ts
import { Agent, SqliteStore, Worker } from "tourbillon";

const store = new SqliteStore("artifacts/durable.db"); // WAL, one file, one process
const agent = new Agent({ provider, tools, name: "worker", store });
await agent.run("Do the task", { sessionId: "task-42" });

// After a crash, in a new process:
await new Worker(store, [agent]).recoverAll({ force: true }); // or agent.recover("task-42")
```

Recovery re-runs the loop and returns recorded results instead of calling the provider
or tools again. A tool call that started but has no recorded result follows its
`effect`: `read` runs again, `idempotent` runs again with the same `idempotencyKey`,
and `unsafe` is not repeated: it gets an "Outcome unknown" result.

- Each run has an owner and an epoch; a process that lost ownership stops at its next
  write. Code that issues different commands than the journal raises
  `NonDeterminismError`, and the run stays recoverable. Loop code must be
  deterministic: use `runtime.now()`, `runtime.uuid()` and `runtime.random()`, and
  `runtime.version("change-id")` to change a loop while old journals still replay.
- `agent.replay(sessionId, runId, { untilSeq })` rebuilds any step without effects.
  `agent.with({ provider }).fork(sessionId, runId, { atSeq })` reuses a prefix and runs
  live after it.
- A changed configuration raises `ConfigurationChanged`, naming the parts.
- The timeout budget counts active time only.

With a store, the session log lives in the store too, written with the journal:
`store.events(sessionId, { after })` follows it, and `store.exportLog(sessionId, path)`
writes the JSONL format. Each event is logged once, even across crashes; recovery adds
`run_recovered`, and a re-run step adds `command_retried`.

## Shared state

```ts
const world = new SharedWorld(store);
const agent = new Agent({ ..., store, shared: world });

// in loop code:
const plan = await runtime.shared.get("/plan");
await runtime.shared.patch([{ op: "replace", path: "/plan/status", value: "done" }], { basedOn: [plan] });

world.at(offset); // the whole world as of any earlier offset
```

Paths are JSON Pointers (or `a.b.c`), writes are JSON Patch. Each write commits once,
in the same transaction as its journal row; a `basedOn` read that changed since raises
`StaleRead`. Tools get a deduplicating `ctx.shared`.

See [docs/durable-state.md](docs/durable-state.md) for the design and the next phases
(locks, task trees, entities and refs, signals and timers).

## Agents as tools

`agentTool(child)` lets a model delegate to another agent and continue its session.
With `execution: "background"`, it returns a job handle at once; `AgentJobs` reports
each outcome to a `Mailbox`, and `AGENT_JOB_TOOLS` list, inspect and cancel jobs.

## Python execution

`tourbillon/python` runs model-written Python in a locked-down Docker container
(no network, read-only root, explicit mounts): `pythonTools(new PythonSessions())`
gives each session its own persistent interpreter.

## Examples

All run offline with `bun examples/<name>.ts`, except `liveChat.ts` (needs an API key)
and `pythonRlm.ts` (needs Docker):

| Example | Shows |
|---|---|
| `customLoop.ts` | a review step inserted into ReAct |
| `compareLoops.ts` | ReAct and plan-then-act on one task |
| `reactWithSummary.ts` | rolling summaries |
| `planningCompaction.ts` | plan, review, test, then compact the planning away |
| `delegatingChef.ts` | background delegation and continuing a child session |
| `kitchenChef.ts` | an event-driven agent supervising real subprocesses |
| `durableRecovery.ts` | a crash mid-run and recovery without repeating an email |
| `liveChat.ts`, `printGraph.ts`, `pythonRlm.ts` | a live provider, graph export, Python |

Coming from the Python version (mickro-agent)? See [docs/porting.md](docs/porting.md).
