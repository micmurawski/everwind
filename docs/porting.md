# Porting from mickro-agent (Python)

tourbillon is the TypeScript and Bun port of mickro-agent. Behaviour is the same, and
the test suite was ported test by test; the API follows TypeScript conventions.

| Python | TypeScript |
|---|---|
| `snake_case` names and fields | `camelCase` (`sessionId`, `toolCalls`, `isError`, `inputTokens`) |
| pydantic records (`Message(role=...)`) | plain objects, validated by `message({...})`, `modelResponse({...})` |
| `@tool def f(a: int, ctx: Hidden[T])` | `tool({ name, input: z.object({...}), run: (args, ctx) => ... })` |
| `Hidden[T]` context values | `ctx.get<T>("key")`, `ctx.runtime`, `ctx.idempotencyKey`, `ctx.shared`, `ctx.signal` |
| `a \| b` (tools) | `Tools.of(a, b)`, `tools.with(c)` |
| `a >> b`, `a - "x" >> b` | `a.then(b)`, `a.on("x", b)` |
| `Flow(nodes=..., edges={(a, x): b})` | `new Flow({ start, nodes, edges: { a: { x: "b" } } })` |
| `dataclasses.replace(agent, x=...)` | `agent.with({ x })` |
| `agent.get_flow_graph()` | `agent.flowGraph()` |
| `task.cancel()` | `agent.run(prompt, { signal })`; the run rejects with `Cancelled` |
| `asyncio.timeout` / `TimeoutError` | `DeadlineExceeded`; tools and providers receive an `AbortSignal` |
| `asyncio.Queue` mailbox | `Mailbox` |
| `async with AgentJobs(...)` | `await jobs.close()` |
| `Limits(timeout=...)` | `limits: { timeout }` (seconds, as before) |
| `mikro-agent-<id>.jsonl`, `schema_version: 2` | `tbln-<id>.jsonl`, `schemaVersion: 1` |
| `MiniMax` content in `metadata["minimax_content"]` | `metadata.minimaxContent` |

## Behaviour that differs

- **Cancellation is cooperative.** JavaScript cannot interrupt a promise. The run stops
  waiting at once and every later runtime call fails, but a provider or tool keeps
  running until it observes its `signal`. Built-in providers, the Python sandbox and
  the agent tools all honour it.
- **Concurrent tool batches** abort the remaining calls through the batch signal when
  one fails, instead of cancelling tasks.
- **A background child agent** starts on the next turn of the event loop, so the tool
  returns its handle first.
- **The SQLite schema is the same** (version 3), but JSON bodies use camelCase, so the
  two implementations don't read each other's files.
- **Tools always receive a context object.** The model cannot supply context values:
  any argument outside the input schema is rejected.
