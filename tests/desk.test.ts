import { afterAll, beforeAll, expect, test } from "bun:test";
import { join } from "node:path";
import { type Run, SqliteStore } from "../src/index.ts";
import { type Desk, type SessionSummary, serveDesk } from "../src/desk/index.ts";
import { agents, projectAgents, seed, startProjects } from "../examples/desk.ts";
import type { Supervisor } from "../src/index.ts";
// @ts-expect-error The desk's browser module is plain JavaScript.
import { planFlow } from "../src/desk/ui/flow.js";
import { PlanThenAct, ReActWithWakeups } from "../src/index.ts";
import { tmpDir } from "./helpers.ts";

let store: SqliteStore;
let desk: Desk;
let live: Promise<Run>[];
let supervisor: Supervisor;

const post = (path: string, body: unknown, headers: Record<string, string> = { "x-everwind": "1", "content-type": "application/json" }) =>
  fetch(new URL(`api/${path}`, desk.url), { method: "POST", headers, body: JSON.stringify(body) });
const get = async (path: string) => {
  const response = await fetch(new URL(`api/${path}`, desk.url));
  return { status: response.status, body: (await response.json()) as any };
};
const statuses = async () => {
  const { body } = await get("overview");
  return Object.fromEntries((body.sessions as SessionSummary[]).map((s) => [s.sessionId, s]));
};
async function until(check: () => Promise<boolean>, ms = 3000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (await check()) return;
    await Bun.sleep(25);
  }
  throw new Error("condition not met in time");
}

beforeAll(async () => {
  store = new SqliteStore(join(tmpDir(), "desk.db"));
  desk = serveDesk({ store, agents: [...agents(store), ...projectAgents(store)], port: 0, pollInterval: 0.05 });
  live = await seed(store);
  supervisor = await startProjects(store);
  await until(async () => (await statuses())["migration-plan"]?.status === "ask");
});

afterAll(async () => {
  for (const session of ["watch-ci-412", "migration-plan", "summarize-interviews"]) await store.setControl(session, "freeze");
  await Promise.allSettled(live);
  await supervisor.stop({ freeze: true });
  desk.stop();
});

test("the overview derives each session's status and flags what needs attention", async () => {
  const s = await statuses();
  expect(s["cart-tests"]).toMatchObject({ status: "completed", agent: "coder", flag: null, modelCalls: 2, toolCalls: 1 });
  expect(s["cart-tests"].tokens).toBe(4000);
  expect(s["flaky-api"]).toMatchObject({ status: "limit" });
  expect(s["flaky-api"].flag).toContain("limit:");
  expect(s["migration-plan"]).toMatchObject({ status: "ask", question: "Migrate EU accounts first, or all regions at once?", node: "ask" });
  expect(s["migration-plan"].flag).toContain("question:");
  expect(s["watch-ci-412"]).toMatchObject({ status: "sleeping", node: "wakeup" });
  expect(s["summarize-interviews"]).toMatchObject({ status: "held", control: "hold" });
  expect(s["nightly-build"]).toMatchObject({ status: "frozen", control: "freeze" });
  expect(s["cart-tests"].title).toBe("Run the cart tests and report");
});

test("pages and assets are served; nothing else is", async () => {
  for (const [path, type] of [["/", "text/html"], ["/sessions", "text/html"], ["/session?id=x", "text/html"], ["/agents", "text/html"], ["/desk.js", "text/javascript"], ["/flow.js", "text/javascript"], ["/theme.css", "text/css"]]) {
    const response = await fetch(new URL(path, desk.url));
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain(type);
  }
  for (const path of ["/nope", "/../package.json", "/server.ts", "/api/nope"]) expect((await fetch(new URL(path, desk.url))).status).toBe(404);
  expect((await get("sessions/bad%20id!")).status).toBe(400);
  expect((await get("sessions/unknown")).status).toBe(404);
});

test("writes need the desk header and JSON, so other sites cannot post to it", async () => {
  expect((await post("sessions/watch-ci-412/send", { text: "x" }, { "content-type": "text/plain" })).status).toBe(403);
  expect((await post("sessions/watch-ci-412/send", { text: "" })).status).toBe(400);
  expect((await post("sessions/watch-ci-412/control", { state: "pause" })).status).toBe(400);
  expect((await post("control", { sessions: [], state: "hold" })).status).toBe(400);
});

test("a reply answers the planner's question, and the session finishes", async () => {
  expect((await post("sessions/migration-plan/reply", { text: "EU first" })).status).toBe(200);
  await until(async () => (await statuses())["migration-plan"].status === "completed");
  const { body } = await get("sessions/migration-plan");
  expect(body.inbox).toMatchObject([{ kind: "reply", payload: { text: "EU first" } }]);
  const events = (await get("sessions/migration-plan/events")).body as any[];
  expect(events.some((stored) => stored.event.kind === "instruction" && stored.event.data.content === "[operator] EU first")).toBe(true);
});

test("resume lets a held session go on; bulk control sets several at once", async () => {
  expect((await post("control", { sessions: ["summarize-interviews"], state: "run" })).status).toBe(200);
  await until(async () => (await statuses())["summarize-interviews"].status === "completed");
  expect((await post("control", { sessions: ["cart-tests", "flaky-api"], state: "hold", note: "audit" })).status).toBe(200);
  expect(store.control("flaky-api")).toMatchObject({ state: "hold", note: "audit" });
});

test("the desk recovers a frozen session itself", async () => {
  expect((await post("sessions/cart-tests/recover", {})).status).toBe(409); // Finished: nothing to recover.
  expect((await post("sessions/nightly-build/recover", {})).status).toBe(202);
  await until(async () => (await statuses())["nightly-build"].status === "completed", 5000);
  const events = (await get("sessions/nightly-build/events")).body as any[];
  expect(events.some((stored) => stored.event.kind === "run_recovered")).toBe(true);
});

test("the stream sends the session's events and its summary", async () => {
  const controller = new AbortController();
  const response = await fetch(new URL("api/sessions/cart-tests/stream", desk.url), { signal: controller.signal });
  expect(response.headers.get("content-type")).toContain("text/event-stream");
  const reader = (response.body as ReadableStream<Uint8Array>).getReader();
  let text = "";
  while (!text.includes("event: summary")) text += new TextDecoder().decode((await reader.read()).value);
  controller.abort();
  expect(text).toContain("event: log");
  expect(text).toContain('"kind":"run_start"');
});

test("replay rebuilds a run up to a command; fork starts a new session from it", async () => {
  // In ReActWithWakeups, seq 1 is the gate's inbox read and seq 2 the first model call.
  const replay = await get("sessions/cart-tests/replay?untilSeq=2");
  expect(replay.status).toBe(200);
  expect(replay.body.reason).toBe("untilSeq");
  expect(replay.body.messages.at(-1).toolCalls[0].name).toBe("run_tests");
  const fork = await post("sessions/cart-tests/fork", { atSeq: 2 });
  expect(fork.status).toBe(202);
  const { sessionId } = (await fork.json()) as { sessionId: string };
  expect(sessionId).toStartWith("cart-tests-fork-");
  await until(async () => (await statuses())[sessionId]?.status === "completed");
});

test("agents lists definitions with their tools and sessions", async () => {
  const { body } = await get("agents");
  const planner = (body as any[]).find((agent) => agent.name === "planner");
  expect(planner.loop).toBe("ReActWithWakeups");
  expect(planner.flow.start).toBe("gate");
  expect(planner.tools.map((t: any) => [t.name, t.effect])).toEqual([["run_tests", "read"], ["ask_operator", "read"], ["schedule_wakeup", "read"]]);
  expect(planner.sessions.map((s: SessionSummary) => s.sessionId).sort()).toEqual(["migration-plan", "summarize-interviews"]);
});

test("with a token, the desk refuses requests without it and sets a cookie from ?token=", async () => {
  const locked = serveDesk({ store, port: 0, token: "s3cret" });
  try {
    expect((await fetch(new URL("api/overview", locked.url))).status).toBe(401);
    const login = await fetch(new URL("/?token=s3cret", locked.url), { redirect: "manual" });
    expect(login.status).toBe(302);
    const cookie = (login.headers.get("set-cookie") ?? "").split(";")[0];
    expect(cookie).toBe("everwind_token=s3cret");
    expect((await fetch(new URL("api/overview", locked.url), { headers: { cookie } })).status).toBe(200);
    expect((await fetch(new URL("api/overview", locked.url), { headers: { "x-everwind-token": "s3cret" } })).status).toBe(200);
  } finally {
    locked.stop();
  }
});

test("the flow layout puts a loop's main path on the first row and marks its back edges", () => {
  const edges = (loop: { toFlow(): { start: string; edgeList(): [string, string, string][] } }) => {
    const flow = loop.toFlow();
    return { start: flow.start, edges: flow.edgeList() };
  };
  const wakeups = planFlow(edges(new ReActWithWakeups()));
  expect(wakeups.pos).toEqual({
    gate: [0, 0],
    call_llm: [1, 0],
    tools: [2, 0],
    settle: [2, 1],
    ask: [3, 0],
    finish: [3, 1],
    wakeup: [4, 0],
    __end__: [4, 1],
    wait: [5, 0],
  });
  expect(wakeups.back).toEqual(["settle→gate", "wait→gate"]);
  const planned = planFlow(edges(new PlanThenAct()));
  expect(planned.pos.plan).toEqual([0, 0]);
  expect(planned.back).toEqual(["tools→call_llm"]);
});

test("the desk's browser scripts use no undefined names", () => {
  // Plain JavaScript in the pages: TypeScript's checker finds a name used without being defined (TS2304, TS2552).
  const { mkdirSync, readFileSync, writeFileSync, copyFileSync } = require("node:fs") as typeof import("node:fs");
  const dir = join(tmpDir(), "ui");
  mkdirSync(dir);
  for (const name of ["desk.js", "flow.js"]) copyFileSync(join("src/desk/ui", name), join(dir, name));
  for (const page of ["projects", "index", "session", "agents"]) {
    const html = readFileSync(join("src/desk/ui", `${page}.html`), "utf-8");
    const script = /<script type="module">([\s\S]*?)<\/script>/.exec(html)?.[1] ?? "";
    writeFileSync(join(dir, `${page}.js`), script.replaceAll('"/desk.js"', '"./desk.js"').replaceAll('"/flow.js"', '"./flow.js"'));
  }
  const files = ["desk.js", "flow.js", "projects.js", "index.js", "session.js", "agents.js"].map((name) => join(dir, name));
  const tsc = Bun.spawnSync(["node_modules/.bin/tsc", "--allowJs", "--checkJs", "--noEmit", "--target", "es2022", "--module", "es2022", "--moduleResolution", "bundler", "--lib", "es2022,dom", ...files]);
  const undefinedNames = tsc.stdout.toString().split("\n").filter((line) => /TS2304|TS2552/.test(line));
  expect(undefinedNames).toEqual([]);
});

// Projects and tasks -----------------------------------------------------------

const projectView = async (id = "checkout-revamp") => (await get(`projects/${id}`)).body as any;
const taskByAgent = (view: any, agent: string) => view.tasks.find((t: any) => t.agent === agent);

test("the board shows the project's task tree with each task's live status", async () => {
  await until(async () => taskByAgent(await projectView(), "engineer")?.display === "ask" && taskByAgent(await projectView(), "analyst")?.display === "done", 6000);
  const list = (await get("projects")).body as any[];
  const listed = list.find((p) => p.id === "checkout-revamp");
  expect(listed).toMatchObject({ status: "active", taskCount: 3, agents: ["analyst", "engineer"] });
  expect(listed.tasks).toBeUndefined();
  const view = await projectView();
  expect(view.tasks.map((t: any) => [t.depth, t.agent])).toEqual([[0, "lead"], [1, "analyst"], [1, "engineer"]]);
  expect(view.tasks[0].display).toBe("sleeping"); // The lead waits for its tasks.
  const engineer = taskByAgent(view, "engineer");
  expect(engineer.summary.question).toBe("Use the provider's official SDK, or plain HTTP calls?");
  expect(engineer.history).toBeUndefined();
  expect(taskByAgent(view, "analyst").result).toStartWith('Report on "Compare EU payment providers');
  expect(view.counts).toMatchObject({ done: 1, ask: 1, sleeping: 1 });
});

test("answering the engineer lets the lead finish, and the project is done", async () => {
  const engineer = taskByAgent(await projectView(), "engineer");
  expect((await post(`sessions/${engineer.id}/reply`, { text: "official SDK" })).status).toBe(200);
  await until(async () => (await projectView()).status === "done", 6000);
  const view = await projectView();
  expect(view.result).toContain("Checkout revamp plan");
  expect(taskByAgent(view, "engineer").result).toContain("official SDK");
  expect(view.spentTokens).toBeGreaterThan(0);
});

test("a follow-up from the desk runs a finished task again", async () => {
  const analyst = taskByAgent(await projectView(), "analyst");
  // The project is done, but a task can still be followed up; the supervisor runs it only in an active project.
  const response = await post(`tasks/${analyst.id}/followup`, { text: "Add Mollie to the comparison" });
  expect(((await response.json()) as any).delivered).toBe("activation");
  expect(taskByAgent(await projectView(), "analyst")).toMatchObject({ status: "ready", followUps: [{ text: "Add Mollie to the comparison", from: "operator" }] });
  expect((await post("tasks/nope/followup", { text: "x" })).status).toBe(404);
});

test("projects can be created, paused, resumed and cancelled from the desk", async () => {
  const response = await post("projects", { title: "Docs", goal: "Write the docs", agent: "lead", agents: ["analyst"], maxDepth: "1", budgetTokens: "" });
  expect(response.status).toBe(201);
  const created = (await response.json()) as any;
  expect(created).toMatchObject({ title: "Docs", agents: ["analyst"], maxDepth: 1, budgetTokens: null, status: "active" });
  expect((await post(`projects/${created.id}/status`, { status: "paused" })).status).toBe(200);
  expect((await projectView(created.id)).status).toBe("paused");
  expect((await post(`projects/${created.id}/status`, { status: "active" })).status).toBe(200);
  expect((await post(`tasks/${created.rootTask}/cancel`, { reason: "not now" })).status).toBe(200);
  const cancelled = await projectView(created.id);
  expect(cancelled.status).toBe("cancelled");
  expect(cancelled.tasks[0]).toMatchObject({ status: "cancelled", error: "not now" });
  expect((await post(`projects/${created.id}/status`, { status: "active" })).status).toBe(409);
  expect((await post("projects", { title: "x", goal: "", agent: "lead" })).status).toBe(400);
  expect((await post(`projects/${created.id}/status`, { status: "done" })).status).toBe(400);
  expect((await get("projects/nope")).status).toBe(404);
});

test("task sessions carry their project and task, for the sessions filters", async () => {
  const sessions = (await get("overview")).body.sessions as SessionSummary[];
  const inProject = sessions.filter((s) => s.project === "checkout-revamp");
  expect(inProject.map((s) => s.agent).sort()).toEqual(["analyst", "engineer", "lead"]);
  expect(inProject.every((s) => s.projectTitle === "Checkout revamp" && s.task === s.sessionId)).toBe(true);
  expect(inProject.find((s) => s.agent === "analyst")?.taskTitle).toBe("Compare EU payment providers");
  const standalone = sessions.find((s) => s.sessionId === "cart-tests");
  expect(standalone).toMatchObject({ project: null, projectTitle: null, task: null, taskTitle: null });
});
