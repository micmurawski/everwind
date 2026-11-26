/**
 * The control desk: a small HTTP server over a durable store. It shows every
 * session's runs and events and steers them (hold, resume, freeze, send,
 * reply, recover) through the same store, so it can run in its own process
 * next to the processes running the agents.
 */

import { readFileSync } from "node:fs";
import { extname } from "node:path";
import type { Agent } from "../agent.ts";
import type { Flow } from "../flow.ts";
import { INTERRUPTED_CONTENT, type ControlState, type DurableStore, type LoggedEvent, type RunRecord } from "../journal.ts";
import { type TaskRecord, cancelTask, createProject, followUp, project as boardProject, projects as boardProjects, setProjectStatus, tasks as boardTasks } from "../board.ts";
import { MESSAGE } from "../inbox.ts";
import { REPLY } from "../operator.ts";
import { providerModel, providerName } from "../providers/base.ts";
import { validateSessionId } from "../sessionLog.ts";
import type { Event } from "../types.ts";
import { type JsonObject, hexId } from "../util.ts";

export interface DeskOptions {
  store: DurableStore;
  /** Agent definitions: shown on the agents page, and needed to recover or replay their runs. */
  agents?: readonly Agent[];
  /** Live resources for runs the desk recovers, as Worker takes them. */
  context?: Record<string, unknown> | ((record: RunRecord) => Record<string, unknown>);
  /** Default 7777; 0 picks a free port. */
  port?: number;
  /** Default 127.0.0.1. Bind elsewhere only behind your own access control. */
  hostname?: string;
  /** When set, every request needs it: open `/?token=…` once to get a cookie. */
  token?: string;
  /** Seconds between store checks for live event streams. Default 0.5. */
  pollInterval?: number;
}

export interface Desk {
  readonly url: string;
  stop(): void;
}

/** What the fleet table shows for a session: its latest run, with live state derived from events. */
export interface SessionSummary {
  sessionId: string;
  runId: string;
  agent: string;
  /** running, held, frozen, sleeping, ask (waiting for a reply), stale (lease expired), or the run's final status. */
  status: string;
  control: ControlState;
  title: string;
  node: string | null;
  question: string | null;
  flag: string | null;
  model: string | null;
  startedAt: string | null;
  lastEventAt: string | null;
  modelCalls: number;
  toolCalls: number;
  tokens: number;
  parentSessionId: string | null;
  runs: number;
  /** For a task's session: its project and task, from the board (null otherwise). */
  project: string | null;
  projectTitle: string | null;
  task: string | null;
  taskTitle: string | null;
}

const UI = new URL("./ui/", import.meta.url);
const TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
};
const PAGES: Record<string, string> = { "/": "projects.html", "/sessions": "index.html", "/session": "session.html", "/agents": "agents.html" };

function lastUserText(record: RunRecord): string {
  const messages = (record.start.messages as { role: string; content: string }[] | undefined) ?? [];
  const user = [...messages].reverse().find((item) => item.role === "user");
  return user?.content ?? "";
}

/** A session's latest run and what it is doing, from its record, control state and events. */
export function summarize(store: DurableStore, records: RunRecord[], now = Date.now()): SessionSummary {
  const timed = records.map((record) => {
    const events = store.events(record.sessionId, { runId: record.runId }).map((stored) => stored.event);
    return { record, events, started: events[0]?.timestamp ?? "" };
  });
  timed.sort((a, b) => (a.started < b.started ? -1 : a.started > b.started ? 1 : 0));
  const { record, events } = timed[timed.length - 1];
  let node: string | null = null;
  let question: string | null = null;
  let sleeping = false;
  let held = false;
  let interrupted = false;
  let model: string | null = null;
  let modelCalls = 0;
  let toolCalls = 0;
  let tokens = 0;
  for (const event of events) {
    const data = event.data as Record<string, any>;
    if (event.kind === "run_start") model = data.model ?? null;
    else if (event.kind === "step") node = data.node;
    else if (event.kind === "question") question = data.question;
    else if (event.kind === "answered") question = null;
    else if (event.kind === "suspended") sleeping = true;
    else if (event.kind === "woke") sleeping = false;
    else if (event.kind === "held") held = true;
    else if (event.kind === "resumed") held = false;
    else if (event.kind === "model_response" && data.conversation !== false) {
      modelCalls += 1;
      for (const [name, value] of Object.entries((data.response?.usage ?? {}) as Record<string, number>)) {
        if (name.endsWith("Tokens") && name !== "reasoningTokens") tokens += value;
      }
    } else if (event.kind === "tool_result") {
      toolCalls += 1;
      if (data.message?.content === INTERRUPTED_CONTENT) interrupted = true;
    }
  }
  const control = store.control(record.sessionId)?.state ?? "run";
  let status = record.status;
  if (record.status === "running") {
    if (control === "freeze" && record.owner === null) status = "frozen";
    else if (record.owner !== null && record.leaseExpiresAt * 1000 < now) status = "stale";
    else if (question !== null) status = "ask";
    else if (held || control === "hold") status = "held";
    else if (sleeping) status = "sleeping";
  }
  let flag: string | null = null;
  if (status === "ask") flag = `question: ${question}`;
  else if (status === "stale") flag = "lease expired: the process running it is gone; recover it";
  else if (interrupted) flag = "outcome unknown: an unsafe tool was interrupted by a crash";
  else if (["failed", "timeout", "limit"].includes(status)) flag = `${status}: ${(record.result?.error as string) ?? ""}`;
  return {
    sessionId: record.sessionId,
    runId: record.runId,
    agent: record.agent,
    status,
    control,
    title: lastUserText(record).slice(0, 200),
    node,
    question,
    flag,
    model,
    startedAt: events[0]?.timestamp ?? null,
    lastEventAt: events[events.length - 1]?.timestamp ?? null,
    modelCalls,
    toolCalls,
    tokens,
    parentSessionId: (record.start.parentSessionId as string | null) ?? null,
    runs: records.length,
    project: ((record.start.scope as JsonObject | undefined)?.project as string | undefined) ?? null,
    projectTitle: null,
    task: ((record.start.scope as JsonObject | undefined)?.task as string | undefined) ?? null,
    taskTitle: null,
  };
}

function sessions(store: DurableStore): SessionSummary[] {
  const bySession = new Map<string, RunRecord[]>();
  for (const record of store.runs()) {
    const list = bySession.get(record.sessionId) ?? [];
    list.push(record);
    bySession.set(record.sessionId, list);
  }
  const now = Date.now();
  const projectTitles = new Map(boardProjects(store).map((item) => [item.id, item.title]));
  const taskTitles = new Map(boardTasks(store).map((item) => [item.id, item.title]));
  return [...bySession.values()]
    .map((records) => {
      const summary = summarize(store, records, now);
      summary.projectTitle = summary.project === null ? null : (projectTitles.get(summary.project) ?? null);
      summary.taskTitle = summary.task === null ? null : (taskTitles.get(summary.task) ?? null);
      return summary;
    })
    .sort((a, b) => ((b.lastEventAt ?? "") < (a.lastEventAt ?? "") ? -1 : (b.lastEventAt ?? "") > (a.lastEventAt ?? "") ? 1 : 0));
}

/** An agent definition as the agents page shows it. */
function describeAgent(agent: Agent): JsonObject {
  let flow: JsonObject | null = null;
  const toFlow = (agent.loop as { toFlow?: () => Flow }).toFlow;
  if (typeof toFlow === "function") {
    try {
      const compiled = toFlow.call(agent.loop);
      // Per node: whether it may write a checkpoint before it (it can wait there), and its metadata.
      const nodes = Object.fromEntries(
        Object.keys(compiled.nodes).map((name) => [
          name,
          { checkpoint: (compiled.checkpoints[name] ?? false) !== false, metadata: (compiled.nodeMetadata[name] ?? {}) as JsonObject },
        ]),
      );
      flow = { start: compiled.start, edges: compiled.edgeList(), nodes };
    } catch {
      flow = null;
    }
  }
  return {
    name: agent.name,
    systemPrompt: agent.systemPrompt,
    loop: agent.loop.constructor?.name ?? "Loop",
    flow,
    provider: providerName(agent.provider),
    model: providerModel(agent.provider),
    limits: agent.limits as unknown as JsonObject,
    tools: agent.tools.list().map((item) => ({
      name: item.name,
      description: item.description,
      effect: item.effect,
      scoped: item.scoped,
      parameters: item.spec.parameters as JsonObject,
    })),
  };
}

class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });

/** Start the desk. Open its `url` in a browser. */
export function serveDesk(options: DeskOptions): Desk {
  const { store } = options;
  const agents = new Map((options.agents ?? []).map((agent) => [agent.name, agent]));
  const poll = (options.pollInterval ?? 0.5) * 1000;
  const token = options.token ?? null;
  const recovering = new Set<string>();

  const agentFor = (record: RunRecord): Agent => {
    const agent = agents.get(record.agent);
    if (agent === undefined) throw new HttpError(409, `The desk has no agent named ${JSON.stringify(record.agent)}; pass it in agents`);
    return agent;
  };
  const latest = (sessionId: string): RunRecord => {
    const records = store.runs({ sessionId });
    if (!records.length) throw new HttpError(404, `No session ${sessionId}`);
    return records.find((record) => record.runId === summarize(store, records).runId) as RunRecord;
  };
  const body = async (request: Request): Promise<Record<string, any>> => {
    // Cross-site forms cannot send this header or a JSON content type: no CSRF.
    if (request.headers.get("x-everwind") !== "1" || !request.headers.get("content-type")?.includes("application/json")) {
      throw new HttpError(403, "Send JSON with the X-Everwind: 1 header");
    }
    try {
      return (await request.json()) as Record<string, any>;
    } catch {
      throw new HttpError(400, "Invalid JSON");
    }
  };
  const text = (value: unknown, name: string): string => {
    if (typeof value !== "string" || !value.trim()) throw new HttpError(400, `${name} must be nonempty text`);
    return value;
  };
  const setControl = async (sessionId: string, state: unknown, note: unknown) => {
    if (state !== "run" && state !== "hold" && state !== "freeze") throw new HttpError(400, "state must be run, hold or freeze");
    try {
      validateSessionId(sessionId);
    } catch (error) {
      throw new HttpError(400, (error as Error).message);
    }
    await store.setControl(sessionId, state, typeof note === "string" && note ? note : null);
  };

  const stream = (request: Request, sessionId: string, after: number): Response => {
    let timer: ReturnType<typeof setInterval> | null = null;
    let lastState = "";
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        const send = (chunk: string) => controller.enqueue(encoder.encode(chunk));
        const tick = () => {
          try {
            const events = store.events(sessionId, { after });
            for (const stored of events) {
              send(`id: ${stored.position}\nevent: log\ndata: ${JSON.stringify(stored)}\n\n`);
              after = stored.position;
            }
            const records = store.runs({ sessionId });
            const state = records.length ? JSON.stringify(summarize(store, records)) : "null";
            if (state !== lastState) {
              lastState = state;
              send(`event: summary\ndata: ${state}\n\n`);
            }
            if (!events.length) send(": keep-alive\n\n");
          } catch (error) {
            send(`event: failure\ndata: ${JSON.stringify(String(error))}\n\n`);
          }
        };
        tick();
        timer = setInterval(tick, poll);
        request.signal.addEventListener("abort", () => {
          if (timer !== null) clearInterval(timer);
          try {
            controller.close();
          } catch {}
        });
      },
      cancel() {
        if (timer !== null) clearInterval(timer);
      },
    });
    return new Response(body, { headers: { "content-type": "text/event-stream", "cache-control": "no-store", connection: "keep-alive" } });
  };

  /** A task as the board shows it: its record, and for a running one, what its session is doing. */
  const boardTask = (item: TaskRecord) => {
    const records = store.runs({ sessionId: item.id });
    const summary = records.length ? summarize(store, records) : null;
    const status = item.status === "running" && summary !== null ? summary.status : item.status;
    const { history: _history, ...rest } = item;
    return { ...rest, forked: item.history !== null, display: status, summary };
  };
  const projectView = (id: string) => {
    const found = boardProject(store, id);
    if (found === null) throw new HttpError(404, `No project ${id}`);
    const list = boardTasks(store, { project: id }).map(boardTask);
    const counts: Record<string, number> = {};
    for (const item of list) counts[item.display] = (counts[item.display] ?? 0) + 1;
    return { ...found, counts, tasks: list };
  };

  const projectsApi = async (request: Request, path: string[]): Promise<Response> => {
    const method = request.method;
    if (path.length === 1 && method === "GET") {
      return json(boardProjects(store).map((item) => {
        const { tasks: _tasks, ...view } = projectView(item.id);
        return view;
      }));
    }
    if (path.length === 1 && method === "POST") {
      const input = await body(request);
      const optionalNumber = (value: unknown) => (value === undefined || value === null || value === "" ? undefined : Number(value));
      try {
        const created = await createProject(store, {
          title: text(input.title, "title"),
          goal: text(input.goal, "goal"),
          agent: text(input.agent, "agent"),
          agents: Array.isArray(input.agents) && input.agents.length ? input.agents.map(String) : undefined,
          maxDepth: optionalNumber(input.maxDepth),
          maxTasks: optionalNumber(input.maxTasks),
          budgetTokens: optionalNumber(input.budgetTokens) ?? null,
        });
        return json(created, 201);
      } catch (error) {
        if (error instanceof HttpError) throw error;
        throw new HttpError(400, (error as Error).message);
      }
    }
    const id = decodeURIComponent(path[1] ?? "");
    if (path.length === 2 && method === "GET") return json(projectView(id));
    if (path[2] === "status" && method === "POST") {
      const input = await body(request);
      if (input.status !== "active" && input.status !== "paused") throw new HttpError(400, "status must be active or paused");
      try {
        return json(await setProjectStatus(store, id, input.status, typeof input.reason === "string" ? input.reason : null));
      } catch (error) {
        throw new HttpError(409, (error as Error).message);
      }
    }
    throw new HttpError(404, "Unknown endpoint");
  };

  const tasksApi = async (request: Request, path: string[]): Promise<Response> => {
    const id = decodeURIComponent(path[1] ?? "");
    const found = boardTasks(store).find((item) => item.id === id);
    if (found === undefined) throw new HttpError(404, `No task ${id}`);
    if (path[2] === "cancel" && request.method === "POST") {
      const input = await body(request);
      const running = await cancelTask(store, id, typeof input.reason === "string" && input.reason ? input.reason : "cancelled by the operator");
      for (const sessionId of running) await store.setControl(sessionId, "freeze", "task cancelled");
      return json({ ok: true, frozen: running });
    }
    if (path[2] === "followup" && request.method === "POST") {
      const input = await body(request);
      try {
        const how = await followUp(store, id, { key: `desk:${hexId()}`, text: text(input.text, "text"), from: "operator" });
        return json({ delivered: how });
      } catch (error) {
        throw new HttpError(409, (error as Error).message);
      }
    }
    throw new HttpError(404, "Unknown endpoint");
  };

  const api = async (request: Request, path: string[], url: URL): Promise<Response> => {
    const method = request.method;
    if (path[0] === "projects") return projectsApi(request, path);
    if (path[0] === "tasks") return tasksApi(request, path);
    if (path[0] === "overview" && method === "GET") {
      return json({ sessions: sessions(store), agents: [...agents.keys()], now: new Date().toISOString() });
    }
    if (path[0] === "agents" && method === "GET") {
      const all = sessions(store);
      return json([...agents.values()].map((agent) => ({ ...describeAgent(agent), sessions: all.filter((item) => item.agent === agent.name) })));
    }
    if (path[0] === "control" && method === "POST") {
      const input = await body(request);
      const ids = Array.isArray(input.sessions) ? (input.sessions as unknown[]) : [];
      if (!ids.length) throw new HttpError(400, "sessions must be a nonempty list");
      for (const id of ids) await setControl(text(id, "session"), input.state, input.note);
      return json({ ok: true, sessions: ids.length });
    }
    if (path[0] !== "sessions" || path[1] === undefined) throw new HttpError(404, "Unknown endpoint");
    const sessionId = decodeURIComponent(path[1]);
    try {
      validateSessionId(sessionId);
    } catch (error) {
      throw new HttpError(400, (error as Error).message);
    }
    const action = path[2] ?? "";
    if (action === "" && method === "GET") {
      const records = store.runs({ sessionId });
      if (!records.length) throw new HttpError(404, `No session ${sessionId}`);
      const summary = summarize(store, records);
      const record = records.find((item) => item.runId === summary.runId) as RunRecord;
      const agent = agents.get(record.agent);
      return json({
        summary,
        runs: records.map((item) => ({ runId: item.runId, status: item.status, owner: item.owner, epoch: item.epoch, result: item.result })),
        control: store.control(sessionId),
        inbox: store.inbox(sessionId),
        systemPrompt: record.start.systemPrompt ?? null,
        scope: record.start.scope ?? null,
        agent: agent === undefined ? null : describeAgent(agent),
        recovering: recovering.has(sessionId),
      });
    }
    if (action === "events" && method === "GET") {
      const after = Number(url.searchParams.get("after") ?? 0) || 0;
      const runId = url.searchParams.get("runId") ?? undefined;
      return json(store.events(sessionId, { after, runId, limit: 2000 }));
    }
    if (action === "stream" && method === "GET") {
      const after = Number(url.searchParams.get("after") ?? request.headers.get("last-event-id") ?? 0) || 0;
      return stream(request, sessionId, after);
    }
    if (action === "control" && method === "POST") {
      const input = await body(request);
      await setControl(sessionId, input.state, input.note);
      return json({ ok: true });
    }
    if (action === "send" && method === "POST") {
      const input = await body(request);
      const payload: JsonObject = { text: text(input.text, "text"), from: typeof input.from === "string" && input.from ? input.from : "operator" };
      if (input.pinned === true) payload.pinned = true;
      return json({ id: await store.signal(sessionId, MESSAGE, payload) });
    }
    if (action === "reply" && method === "POST") {
      const input = await body(request);
      return json({ id: await store.signal(sessionId, REPLY, { text: text(input.text, "text") }) });
    }
    if (action === "recover" && method === "POST") {
      await body(request);
      const record = latest(sessionId);
      if (record.status !== "running") throw new HttpError(409, `The latest run is ${record.status}, not recoverable`);
      if (record.owner !== null && record.leaseExpiresAt * 1000 > Date.now()) throw new HttpError(409, "The run is live in another process");
      if (recovering.has(sessionId)) throw new HttpError(409, "Already recovering");
      const agent = agentFor(record);
      const context = typeof options.context === "function" ? options.context(record) : options.context;
      recovering.add(sessionId);
      // The desk runs the recovered run itself; its events stream like any other run's.
      void agent
        .recover(sessionId, { context, force: true })
        .catch(() => {})
        .finally(() => recovering.delete(sessionId));
      return json({ ok: true }, 202);
    }
    if (action === "replay" && method === "GET") {
      const runId = url.searchParams.get("runId") ?? latest(sessionId).runId;
      const record = store.getRun(sessionId, runId);
      if (record === null) throw new HttpError(404, `No run ${runId}`);
      const untilSeq = url.searchParams.has("untilSeq") ? Number(url.searchParams.get("untilSeq")) : undefined;
      const replay = await agentFor(record).replay(sessionId, runId, { untilSeq });
      return json({ reason: replay.reason, commands: replay.commands, messages: replay.run.messages, pendingToolCalls: replay.pendingToolCalls });
    }
    if (action === "fork" && method === "POST") {
      const input = await body(request);
      const runId = typeof input.runId === "string" ? input.runId : latest(sessionId).runId;
      const record = store.getRun(sessionId, runId);
      if (record === null) throw new HttpError(404, `No run ${runId}`);
      if (!(typeof input.atSeq === "number" && input.atSeq >= 0)) throw new HttpError(400, "atSeq must be a nonnegative number");
      const newSessionId = `${sessionId}-fork-${hexId().slice(0, 6)}`;
      const context = typeof options.context === "function" ? options.context(record) : options.context;
      void agentFor(record).fork(sessionId, runId, { atSeq: input.atSeq, newSessionId, context }).catch(() => {});
      return json({ sessionId: newSessionId }, 202);
    }
    throw new HttpError(404, "Unknown endpoint");
  };

  const authorized = (request: Request, url: URL): Response | null => {
    if (token === null) return null;
    const cookie = /(?:^|;\s*)everwind_token=([^;]+)/.exec(request.headers.get("cookie") ?? "")?.[1];
    if (cookie === token || request.headers.get("x-everwind-token") === token) return null;
    if (url.searchParams.get("token") === token && request.method === "GET") {
      return new Response(null, {
        status: 302,
        headers: { location: url.pathname, "set-cookie": `everwind_token=${token}; HttpOnly; SameSite=Strict; Path=/` },
      });
    }
    return new Response("Open the desk with ?token=…", { status: 401 });
  };

  const server = Bun.serve({
    port: options.port ?? 7777,
    hostname: options.hostname ?? "127.0.0.1",
    async fetch(request) {
      const url = new URL(request.url);
      const denied = authorized(request, url);
      if (denied !== null) return denied;
      try {
        if (url.pathname.startsWith("/api/")) return await api(request, url.pathname.slice(5).split("/").filter(Boolean), url);
        const file = PAGES[url.pathname] ?? (/^\/[a-z]+\.(css|js|svg)$/.test(url.pathname) ? url.pathname.slice(1) : null);
        if (file === null || request.method !== "GET") return new Response("Not found", { status: 404 });
        const content = readFileSync(new URL(file, UI));
        return new Response(content, { headers: { "content-type": TYPES[extname(file)] ?? "application/octet-stream", "cache-control": "no-store" } });
      } catch (error) {
        if (error instanceof HttpError) return json({ error: error.message }, error.status);
        if ((error as NodeJS.ErrnoException)?.code === "ENOENT") return new Response("Not found", { status: 404 });
        return json({ error: error instanceof Error ? error.message : String(error) }, 500);
      }
    },
  });
  return { url: `http://${server.hostname}:${server.port}/`, stop: () => server.stop(true) };
}

export type { Event, LoggedEvent };
