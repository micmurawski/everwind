# everwind

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

## Install

```sh
bun add everwind zod
bun add openai               # optional: OpenAIChat
bun add @anthropic-ai/sdk    # optional: MiniMax
```

The package is published as `everwind`. It needs
Bun: the SQLite store uses `bun:sqlite`.

## Development

```sh
bun install
bun test            # the full suite, offline
bun run typecheck
bun run build       # dist/: JS and type declarations
```

### Releasing

Bump `version` in `package.json`, merge to `main`, then publish a GitHub release
tagged `v<version>` (for example `v0.1.0`). The `Publish to npm` workflow checks that
the tag matches, runs the tests, builds and publishes with provenance. It can also be
started by hand as a dry run. It publishes with npm trusted publishing (configure
`micmurawski/everwind`, workflow `publish.yml`, environment `npm` on npmjs.com), or
with an `NPM_TOKEN` secret in the `npm` environment: a granular token for all packages
with "bypass two-factor authentication" enabled.

## A first agent

```ts
import { z } from "zod";
import { Agent, OpenAIChat, Tools, tool } from "everwind";

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
import { END, Flow, finishStep, modelStep, node, toolsStep } from "everwind";

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
`OlderHistory` and `OlderMessages`; strategies are `Remove`, `Replace(text)`,
`Summarize` (a budgeted, tool-free model call) or your own. A proposal that breaks tool
history, adds a system message or doesn't shrink the context (with a counter) is not
applied. `ReActWithSummary` summarizes before each model turn once a threshold is hit.

## Sessions and logs

`agent.run(prompt, { sessionId })` continues a conversation. Without a store, sessions
are kept in memory, or in `FileSessionStore(directory)`; `logDir` writes one JSONL log
per session (`everwind-<session>.jsonl`). Concurrent continuations of one session are
serialized. An interrupted tool call is closed with an "Outcome unknown" result rather
than repeated.

## Durable runs and crash recovery

```ts
import { Agent, SqliteStore, Worker } from "everwind";

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
- **Checkpoints** keep recovery short. A `Flow` writes one between nodes after an
  applied compaction, and before any node created with `node(fn, { checkpoint })`
  (`true` or a predicate). `ReActWithWakeups` writes one before each sleep. A checkpoint
  holds the conversation, `run.state`, usage and counters, and the node to enter next.
  Recovery and replay restart there and replay only later commands, so a long-lived
  agent does not replay its whole history. `run.state` must be plain JSON at that
  point; otherwise the checkpoint is skipped (logged as `checkpoint_skipped`) and
  recovery replays from the start. Custom loops that are not a `Flow` always replay
  from the start.
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

## Workspace, sub-agent and wakeup tools

Ready-made tools for coding-style agents:

```ts
import { Agent, BackgroundShells, ReActWithWakeups, Workspace, scheduleWakeupTool, subagentTools, workspaceTools } from "everwind";

const workspace = new Workspace({ root: "./project" });  // file paths are confined to root
const tools = workspaceTools(workspace)                    // bash, read_file, write_file, edit_file
  .with(subagentTools({ researcher: { agent: researcher, description: "Finds facts" }, coder }))
  .with(scheduleWakeupTool());                             // agent, list_agents, schedule_wakeup
const agent = new Agent({ provider, tools, loop: new ReActWithWakeups(), limits: { maxSteps: 500 } });
await agent.run("Watch CI and fix failures", { context: { shells: new BackgroundShells() } });
```

- `bash` runs a command in its own process group, with a timeout that kills its
  children. `cd` carries over between calls of a session. With `runInBackground`, a
  `ShellFinished` message arrives in the `shells` context value's mailbox.
  `shellJobTools()` adds `list_shells` and `kill_shell`.
- `read_file` numbers lines like `cat -n`. `write_file` and `edit_file` refuse to
  change a file the session has not read, or one that changed since it was read.
- `agent` runs a task in a child agent of the type the model picks, and returns a
  `sessionId` the model can pass back to continue it. `runInBackground` needs
  `AgentJobs` as the `agentJobs` context value. `list_agents` lists the types and the
  background jobs.
- `schedule_wakeup` only records the request. In `ReActWithWakeups`, a `wakeup` node
  after the tools step sleeps with `runtime.sleep(seconds)` and then gives the model
  the wakeup prompt. The run ends when the model answers without calling a tool.
  `wakeupStep()` adds the same node to your own flow.

`runtime.sleep` is a durable timer. The wake-up time is journaled before the run
waits. A run recovered during its sleep waits only for the time that is left, and
sleeping does not count toward `limits.timeout`.

## Inbox: messaging a running agent

With a store, every session has a durable inbox. Send to it from anywhere, even
another process using the same SQLite file:

```ts
await agent.send("task-42", "Use the EU sandbox", { from: "operator" }); // kind "message"
await store.signal("task-42", "child_finished", { taskId: "t_7", output: "..." });
store.inbox("task-42", { pending: true });                                // read without consuming
```

Loop code reads it through two journaled runtime calls:

- `runtime.receive({ kinds })` takes the pending messages, oldest first. The journal
  records which messages it took, in the same transaction that marks them consumed, so
  each message is delivered once, across crashes, and a replay gets the same ones.
- `runtime.suspend({ seconds, wakeOn: kinds })` waits until the time passes or a
  message of those kinds is pending, and returns what woke it. It does not consume
  the message. `runtime.sleep(seconds)` is `suspend` with only a timer.

`gateStep()` is a node that adds pending `"message"` messages as user messages.
`ReActWithWakeups` runs it before every model turn (`gate -> call_llm`), so a message
sent while the agent works arrives at its next turn, and a message ends a
`schedule_wakeup` sleep early. Message ids increase per session; journals record ids,
never storage positions.

## Projects and tasks

A project is one goal; its root task is that goal. Agents split work into child tasks,
so the work forms a tree. A task is a session, each activation is a run, and a task
finishes when its agent finishes.

```ts
import { Supervisor, taskTool, waitTool, askOperatorTool, ReActWithWakeups, Tools } from "everwind";

const tools = new Tools([taskTool(), waitTool(), askOperatorTool()]);
const lead = new Agent({ name: "lead", provider, tools, loop: new ReActWithWakeups(), store });
// … researcher, coder with the same store

const supervisor = new Supervisor(store, [lead, researcher, coder], { maxActive: 8, perAgent: { coder: 2 } });
supervisor.start();
await supervisor.createProject({ title: "Checkout", goal: "…", agent: "lead", agents: ["researcher", "coder"], maxDepth: 3, budgetTokens: 2_000_000 });
```

- **`task({ agent, prompt, title?, context? })`** creates a child task and returns its id
  at once; the child runs in the background. `context: "fork"` starts it from the
  creator's conversation (the messages themselves for the same agent, a transcript for
  another). **`task({ id, prompt })`** follows a child up: a running one gets it as a
  pinned message, a finished one runs again on its session with it as a pinned prompt.
  Creation and follow-ups use the tool call's idempotency key, so a recovered run never
  duplicates them.
- **Outcomes** arrive as pinned `task_finished` inbox messages, delivered by the gate.
  **`wait_for_tasks`** waits for children; the **`settle`** node keeps a task from
  finishing while its children are out (it waits, then lets the model see their
  results). Both are in `ReActWithWakeups`: `call_llm -answer-> settle -> finish`.
- **Scope and guardrails**: each activation runs with a task scope and frame
  (`taskFrame`), so `task` is hidden at the project's `maxDepth` and offers only the
  project's agents. The store also refuses tasks past `maxDepth`, `maxTasks` or with
  other agents.
- **The supervisor** admits ready tasks of active projects (`maxActive`, `perAgent`),
  runs each activation, and records its outcome on the board atomically with the
  parent's message and the project's token count; a spent budget pauses the project.
  Every pass reconciles running tasks with their runs in the store, so a crash, a
  freeze, or a run recovered from the desk is picked up by any supervisor.
  `supervisor.cancel(taskId)` cancels a subtree (running tasks are frozen, the parent is
  told). The board's reads and writes are in `board.ts` (`projects`, `tasks`,
  `followUp`, `setProjectStatus`, …).

## Control desk

`everwind/desk` serves a small web UI over a store: every session's live state, its
event log as it happens, and the controls above.

```ts
import { serveDesk } from "everwind/desk";

const desk = serveDesk({ store, agents: [planner, coder] }); // http://127.0.0.1:7777/
```

Try it offline with `bun examples/desk.ts`: a project whose lead delegates to an analyst
and an engineer (who asks you a question), plus standalone sessions: a planner waiting
for your answer, a watcher sleeping, one held and one frozen.

- **Projects** (the home page): the projects, a new-project dialog, and the selected
  project's task tree with each task's live status (running, needs you, held, sleeping,
  done, failed), tokens and result. A task's panel shows its prompt, result, children and
  queued follow-ups; you can answer its question, message it while it runs or follow it
  up when it is done, hold, freeze, recover or cancel it. Projects can stop admitting
  tasks and resume.
- **Sessions**: the latest run of each session, with a derived status (running, sleeping,
  held, frozen, needs you, stale, failed, done), the current node, calls and tokens.
  Filter by status, project, task, agent or text, show only what needs attention, and hold, resume,
  freeze or message several sessions at once.
- **Session**: the event log, streamed (server-sent events polling the store), with
  replay to or fork at any event; the loop's flow as a graph (laid out like thearc's flow
  visualizer: loops drawn as back edges, the current node in the session's status colour,
  visit and edge counts; drag, pan and zoom); the conversation, an
  open question with its suggested answers, a composer for messages, the inbox and run
  details.
- **Agents**: each registered definition (provider, loop and its flow graph, limits, tools with
  their effects, system prompt) and its live and finished sessions.

The desk reads and writes only the store, so it can run in its own process: controls
are session state and inbox messages that the agents' processes pick up. Recover,
replay and fork run in the desk's process and need the agent in `agents` (and
`context` for tools that need resources). It binds to 127.0.0.1; writes need a JSON
body and an `X-Everwind: 1` header, so other sites cannot post to it, and `token`
requires a secret (open `/?token=…` once). The look is thearc's theme
(`theme.css`, `theme.js`, Apache-2.0).

## Operator controls: hold, freeze, questions

Every durable run can be steered from outside, even from another process using the
same SQLite file:

```ts
await agent.hold("task-42", "checking the bill"); // stops at its next step; the timeout clock stops
await agent.resume("task-42");
await agent.freeze("task-42");                    // stops at its next step, or at once while it sleeps
const run = await agent.recover("task-42");       // a frozen run continues from its checkpoint
await agent.reply("task-42", "EU first");         // answers ask_operator
```

- **Hold and freeze are session control state** (`store.setControl`, `store.control`),
  checked by a `Flow` before each node. They are never journaled: they change *when* a
  run proceeds, not which commands it issues, so replays are unaffected. A held session
  keeps its lease; a hold survives a crash and recovery.
- **Freeze** writes a checkpoint, releases the lease without finishing the run, and
  `agent.run()` returns the run with status `"frozen"`. The stored run stays `running`;
  `Worker.recoverAll()` skips frozen sessions (`includeFrozen` to include them), and
  `agent.recover()` resumes one without `force`. A run frozen while sleeping later
  sleeps only what is left.
- **Questions**: `askOperatorTool()` lets the model ask; `ReActWithWakeups` then waits
  in its `ask` node (a `question` event, a durable wait, a checkpoint before it) for
  `agent.reply()`, and adds the reply as a pinned user message. With
  `new ReActWithWakeups({ ask: { timeout } })`, the model decides itself after the timeout.

## Scope, frame and pinned messages

A supervisor tells an agent what a run is for in three places, each with its own job:

```ts
import { canSpawn, taskFrame, taskScope, tool } from "everwind";

const scope = taskScope({ project: "p1", task: "t_a", parent: "t_root", depth: 1, maxDepth: 2, agents: ["researcher"] });
const spawn = tool({
  name: "spawn",
  available: canSpawn,                                   // hidden at maxDepth
  input: (scope) => z.object({ agent: z.enum(scope.agents as [string]), prompt: z.string() }),
  run: (args, ctx) => /* ctx.scope is the run's scope */ ...,
});
await agent.run("Goal: compare providers…", { sessionId: "t_a", scope, frame: taskFrame(scope) });
await agent.run("Follow-up: also check fees", { sessionId: "t_a", pin: true });
```

- **Scope** (`scope`): plain JSON, recorded with the run. Tools bind to it once per run:
  `available(scope)` hides a tool, and `input` or `description` can be functions of it.
  Recovery and replay bind the same tools. A continuation without a scope reuses the
  session's latest one.
- **Frame** (`frame`): supervisor text appended to the agent's system prompt, fixed for
  the session. `taskFrame(scope)` writes one with facts only the supervisor controls
  (task, project, depth, who receives the result, whether subtasks are possible). The
  task's own goal stays the prompt, a user message: text another model wrote never gets
  system-prompt authority.
- **Pinned messages**: user messages with `metadata.pinned` that compaction never
  removes. When a compacted region contains them, they are kept after its replacement.
  Pin with `agent.run(prompt, { pin: true })`, `runtime.say(text, { pinned: true })`, or
  an inbox message whose payload has `pinned: true`. The first conversation message
  (a task's prompt) is never compacted by `OlderMessages` or `OlderHistory` anyway.

## Python execution

`everwind/python` runs model-written Python in a locked-down Docker container
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
| `desk.ts` | the control desk over live, held, frozen and waiting sessions |
| `liveChat.ts`, `printGraph.ts`, `pythonRlm.ts` | a live provider, graph export, Python |

Coming from the Python version (mickro-agent)? See [docs/porting.md](docs/porting.md).
