/** Export compiled flow graphs without running nodes or contacting a service. */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, extname } from "node:path";
import { END, Flow } from "./flow.ts";

export interface FlowSource {
  toFlow(): Flow;
}

function asFlow(source: Flow | FlowSource): Flow {
  if (source instanceof Flow) return source;
  if (typeof source?.toFlow !== "function") {
    throw new TypeError("Graph export requires a Flow or a loop implementing toFlow(): Flow");
  }
  const result = source.toFlow();
  if (!(result instanceof Flow)) throw new TypeError("toFlow() must return a Flow");
  return result;
}

function checkDirection(direction: string): string {
  if (!["LR", "RL", "TB", "BT"].includes(direction)) throw new Error("direction must be LR, RL, TB, or BT");
  return direction;
}

function mermaidLabel(text: string): string {
  // User labels never become Mermaid identifiers, markup, or graph statements.
  return [...text]
    .map((char) => (char === "\n" ? "<br/>" : /[\p{L}\p{N} _\-.]/u.test(char) ? char : `#${char.codePointAt(0)};`))
    .join("");
}

function ids(flow: Flow): [string[], Record<string, string>] {
  const names = Object.keys(flow.nodes).sort();
  const map: Record<string, string> = Object.fromEntries(names.map((name, index) => [name, `n${index}`]));
  map[END] = "end_node";
  return [names, map];
}

export function buildMermaid(source: Flow | FlowSource, { direction = "LR" }: { direction?: string } = {}): string {
  const flow = asFlow(source);
  const [names, id] = ids(flow);
  const edges = flow.edgeList();
  const lines = [`flowchart ${checkDirection(direction)}`, '    start_node(["START"])'];
  lines.push(...names.map((name) => `    ${id[name]}["${mermaidLabel(name)}"]`));
  if (edges.some(([, , target]) => target === END)) lines.push('    end_node(["END"])');
  lines.push(`    start_node --> ${id[flow.start]}`);
  for (const [name, action, target] of edges) {
    const edge = action === "default" ? "-.->" : `-->|"${mermaidLabel(action)}"|`;
    lines.push(`    ${id[name]} ${edge} ${id[target]}`);
  }
  return `${lines.join("\n")}\n`;
}

export function buildDot(source: Flow | FlowSource, { direction = "LR" }: { direction?: string } = {}): string {
  const flow = asFlow(source);
  const [names, id] = ids(flow);
  const edges = flow.edgeList();
  const lines = ["digraph flow {", `    rankdir=${checkDirection(direction)};`, '    start_node [label="START", shape=oval];'];
  lines.push(...names.map((name) => `    ${id[name]} [label=${JSON.stringify(name)}, shape=box];`));
  if (edges.some(([, , target]) => target === END)) lines.push('    end_node [label="END", shape=oval];');
  lines.push(`    start_node -> ${id[flow.start]};`);
  for (const [name, action, target] of edges) {
    const attrs = action === "default" ? "style=dashed" : `label=${JSON.stringify(action)}`;
    lines.push(`    ${id[name]} -> ${id[target]} [${attrs}];`);
  }
  return [...lines, "}", ""].join("\n");
}

/**
 * Render locally using Graphviz's `dot` executable; supports PNG and SVG.
 *
 * Mermaid and DOT exports need no Graphviz. No graph is sent to a remote
 * rendering service. The output file is written only after rendering succeeds.
 */
export async function renderGraph(
  source: Flow | FlowSource,
  filename: string,
  options: { direction?: string; format?: "png" | "svg"; timeout?: number; dot?: string } = {},
): Promise<string> {
  const format = options.format ?? extname(filename).slice(1).toLowerCase();
  if (format !== "png" && format !== "svg") throw new Error("Use a .png or .svg filename, or format: 'png' | 'svg'");
  const diagram = buildDot(source, { direction: options.direction });
  let child: ReturnType<typeof Bun.spawn>;
  try {
    child = Bun.spawn([options.dot ?? "dot", `-T${format}`], { stdin: "pipe", stdout: "pipe", stderr: "pipe" });
  } catch (error) {
    throw new Error(
      "Image export requires Graphviz's 'dot' executable on PATH. Mermaid/DOT text export needs no extra dependencies.",
      { cause: error },
    );
  }
  const stdin = child.stdin as import("bun").FileSink;
  stdin.write(diagram);
  stdin.end();
  const timer = setTimeout(() => child.kill(), (options.timeout ?? 30) * 1000);
  const [output, errors, code] = await Promise.all([
    new Response(child.stdout as ReadableStream).arrayBuffer(),
    new Response(child.stderr as ReadableStream).text(),
    child.exited,
  ]);
  clearTimeout(timer);
  if (code !== 0) throw new Error(`Graphviz rendering failed: ${errors}`);
  mkdirSync(dirname(filename), { recursive: true });
  writeFileSync(filename, new Uint8Array(output));
  return filename;
}
