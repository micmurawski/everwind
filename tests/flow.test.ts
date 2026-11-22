import { expect, test } from "bun:test";
import { Agent, END, Flow, type Runtime, ScriptedProvider, node } from "../src/index.ts";

const run = (flow: Flow, prompt = "Task") => new Agent({ provider: new ScriptedProvider([]), loop: flow }).run(prompt);

test("a default chain supports sync and async nodes and an explicit end", async () => {
  const first = node((runtime) => void (runtime.run.state.visited = ["first"]), { name: "first" });
  const second = node(
    async (runtime) => {
      await Promise.resolve();
      (runtime.run.state.visited as string[]).push("second");
      return "default";
    },
    { name: "second" },
  );
  const finish = node((runtime) => runtime.finish((runtime.run.state.visited as string[]).join(",")), { name: "finish" });
  first.then(second).then(finish).then(END);
  const result = await run(new Flow(first));
  expect([result.status, result.output]).toEqual(["completed", "first,second"]);
});

test("named branches and a cycle use run state", async () => {
  const route = node((runtime) => (((runtime.run.state.count as number) ?? 0) < 2 ? "again" : "done"), { name: "route" });
  const increment = node((runtime) => void (runtime.run.state.count = ((runtime.run.state.count as number) ?? 0) + 1), { name: "increment" });
  const finish = node((runtime) => runtime.finish(String(runtime.run.state.count)), { name: "finish" });
  route.on("again", increment).on("done", finish);
  increment.then(route);
  // A leaf ends on its default action without an explicit .then(END).
  const flow = new Flow(route);
  expect((await run(flow)).output).toBe("2");
  expect((await run(flow)).output).toBe("2");
});

test("default is not a fallback for unrecognized actions", async () => {
  const start = node(() => "typo", { name: "start" });
  start.then(node((runtime) => runtime.finish("Must not be reached"), { name: "finish" }));
  const result = await run(new Flow(start));
  expect(result.status).toBe("failed");
  expect(result.error).toContain("No transition");
});

test("a flow snapshots edges and metadata", async () => {
  const start = node(() => "later", { name: "start", metadata: { phase: { name: "original" } } });
  const finish = node((runtime) => runtime.finish("Changed"), { name: "finish" });
  const original = new Flow(start);
  start.on("later", finish);
  (start.metadata.phase as { name: string }).name = "modified";
  const changed = new Flow(start);
  const [before, after] = [await run(original), await run(changed)];
  expect([before.status, after.output]).toEqual(["failed", "Changed"]);
  expect(before.events.find((event) => event.kind === "step")?.data.metadata).toEqual({ phase: { name: "original" } });
});

test("duplicate edges, names and bad targets fail during definition", () => {
  const start = node(() => {}, { name: "start" });
  const finish = node((runtime) => runtime.finish("Done"), { name: "finish" });
  start.then(finish);
  expect(() => start.then(END)).toThrow(/Duplicate transition/);
  expect(() => finish.then("typo" as typeof END)).toThrow(/target/);
  expect(() => finish.on(123 as unknown as string, END)).toThrow(/action/);
  expect(() => node(() => {}, { name: END })).toThrow(/reserved/);
  finish.then(node(() => {}, { name: "start" }));
  expect(() => new Flow(start)).toThrow(/Duplicate node name/);
});

test("one compiled flow can run concurrently", async () => {
  const remember = node(
    async (runtime) => {
      runtime.run.state.task = runtime.run.messages.at(-1)?.content;
      await Promise.resolve();
    },
    { name: "remember" },
  );
  remember.then(node((runtime) => runtime.finish(runtime.run.state.task as string), { name: "finish" }));
  const agent = new Agent({ provider: new ScriptedProvider([]), loop: new Flow(remember) });
  const results = await Promise.all([agent.run("A"), agent.run("B")]);
  expect(results.map((result) => result.output)).toEqual(["A", "B"]);
});

test("a non-action return fails with a clear error", async () => {
  const invalid = node(() => ({ answer: 42 }) as unknown as string, { name: "invalid" });
  const result = await run(new Flow(invalid));
  expect(result.status).toBe("failed");
  expect(result.error).toContain("string action or nothing");
});

test("explicit mappings build a flow", async () => {
  const flow = new Flow({ start: "only", nodes: { only: (runtime: Runtime) => runtime.finish("ok") }, edges: { only: { default: END } } });
  expect((await run(flow)).output).toBe("ok");
});
