import { expect, test } from "bun:test";
import { chmodSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Agent, END, Flow, PlanThenAct, ReAct, ScriptedProvider, buildDot, buildMermaid, node, renderGraph } from "../src/index.ts";
import { tmpDir } from "./helpers.ts";

const forbidden = () => {
  throw new Error("Export must not execute any node");
};

test("graph export shows cycles, branches, default edges and the actual end", () => {
  const model = node(forbidden, { name: "model" });
  const tools = node(forbidden, { name: "tools" });
  model.on("tools", tools).on("answer", END);
  tools.then(model);
  const flow = new Flow(model);
  expect(flow.toMermaid()).toBe(
    'flowchart LR\n    start_node(["START"])\n    n0["model"]\n    n1["tools"]\n' +
      '    end_node(["END"])\n    start_node --> n0\n    n0 -->|"answer"| end_node\n' +
      '    n0 -->|"tools"| n1\n    n1 -.-> n0\n',
  );
  expect(flow.toDot()).toContain('n0 -> n1 [label="tools"]');
  expect(flow.toDot()).toContain("n1 -> n0 [style=dashed]");
  expect(flow.toDot({ direction: "TB" }).startsWith("digraph flow {\n    rankdir=TB;")).toBe(true);
  const cycle = node(forbidden, { name: "cycle" });
  cycle.then(cycle);
  expect(new Flow(cycle).toMermaid()).not.toContain("end_node");
});

test("export uses the compiled snapshot and escapes labels", () => {
  const name = 'END" ]\n<script> &#quot; | \\unicode:ą';
  const action = 'a"|b\n%% comment';
  const start = node(() => {}, { name });
  const flow = new Flow(start);
  start.on(action, start);
  expect(flow.toMermaid()).not.toContain(action);
  const changed = new Flow(start);
  const mermaid = changed.toMermaid();
  expect(mermaid).not.toContain("<script>");
  expect(mermaid).not.toContain("%% comment");
  for (const part of ["#34;", "#124;", "<br/>", 'n0["', "ą"]) expect(mermaid).toContain(part);
  expect(changed.toDot()).toContain(`label=${JSON.stringify(name)}`);
  expect(changed.toDot()).toContain(`label=${JSON.stringify(action)}`);
});

for (const loop of [new ReAct(), new PlanThenAct()]) {
  test(`the agent graph matches the recipe and runtime transitions (${loop.constructor.name})`, async () => {
    const agent = new Agent({ provider: new ScriptedProvider([{ text: "Plan" }, { text: "Answer" }]), loop });
    const flow = loop.toFlow();
    expect(agent.flowGraph()).toBe(buildMermaid(flow));
    expect(agent.flowGraph({ direction: "TB" }).startsWith("flowchart TB")).toBe(true);
    const run = await agent.run("Task");
    expect(run.status).toBe("completed");
    for (const event of run.events.filter((item) => item.kind === "transition")) {
      expect(flow.edges[event.data.node as string][event.data.action as string]).toBe(event.data.target as string);
    }
    expect("plan" in flow.nodes).toBe(loop instanceof PlanThenAct);
  });
}

test("export rejects uninspectable loops and invalid options", async () => {
  const agent = new Agent({ provider: new ScriptedProvider([]), loop: { run: async (runtime) => runtime.finish("done") } });
  expect(() => agent.flowGraph()).toThrow(/toFlow/);
  expect(() => buildDot({ toFlow: () => "not a flow" as never })).toThrow(/must return a Flow/);
  expect(() => buildMermaid(new ReAct(), { direction: "injected" })).toThrow(/direction/);
  await expect(renderGraph(new ReAct(), join(tmpDir(), "graph.html"))).rejects.toThrow(/png or .svg/);
});

function fakeDot(directory: string, body: string): string {
  const path = join(directory, "fake-dot");
  writeFileSync(path, `#!/bin/sh\n${body}\n`);
  chmodSync(path, 0o755);
  return path;
}

test("render calls local Graphviz", async () => {
  const directory = tmpDir();
  const dot = fakeDot(directory, `echo "$@" >> "${directory}/calls"\ncat > "${directory}/input"\nprintf 'rendered graph'`);
  const agent = new Agent({ provider: new ScriptedProvider([]) });
  const png = await agent.renderGraph(join(directory, "nested", "flow.png"), { dot });
  expect(readFileSync(png, "utf-8")).toBe("rendered graph");
  await new ReAct().toFlow().render(join(directory, "flow.svg"), { direction: "BT", dot });
  expect(readFileSync(join(directory, "calls"), "utf-8").trim().split("\n")).toEqual(["-Tpng", "-Tsvg"]);
  expect(readFileSync(join(directory, "input"), "utf-8")).toContain("rankdir=BT");
});

test("a failed render does not overwrite an existing file", async () => {
  const directory = tmpDir();
  const path = join(directory, "graph.svg");
  writeFileSync(path, "original");
  await expect(renderGraph(new ReAct(), path, { dot: fakeDot(directory, "echo bad graph >&2; exit 1") })).rejects.toThrow(/bad graph/);
  await expect(renderGraph(new ReAct(), path, { dot: join(directory, "missing-dot") })).rejects.toThrow(/Graphviz/);
  expect(readFileSync(path, "utf-8")).toBe("original");
});

test.skipIf(Bun.which("dot") === null)("real Graphviz renders PNG and SVG", async () => {
  const directory = tmpDir();
  const png = await renderGraph(new PlanThenAct(), join(directory, "plan.png"));
  const svg = await renderGraph(new ReAct(), join(directory, "react.svg"));
  expect(readFileSync(png).subarray(0, 8)).toEqual(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  expect(readFileSync(svg, "utf-8")).toContain("call_llm");
});
