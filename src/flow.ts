/** Compose nodes with .then() and .on(action, ...), then snapshot them into a runnable flow. */

import { NonDeterminismError } from "./errors.ts";
import type { Runtime } from "./runtime.ts";
import { buildDot, buildMermaid, renderGraph } from "./viz.ts";

export const END = "__end__";
export type Action = string | null | undefined | void;
export type Step = (runtime: Runtime) => Action | Promise<Action>;
/** Whether to write a checkpoint before entering a node; see NodeOptions.checkpoint. */
export type CheckpointPolicy = boolean | ((runtime: Runtime) => boolean);

export interface NodeOptions {
  name?: string;
  metadata?: Record<string, unknown>;
  /**
   * Write a checkpoint before entering this node (when the predicate holds),
   * so a recovered run restarts here instead of replaying from the start. Use
   * it before a node that may wait a long time. An applied compaction also
   * requests one at the next node boundary.
   */
  checkpoint?: CheckpointPolicy;
}
type Target = Node | typeof END;

/** A graph-building handle. Execution state lives on Runtime, never on this node. */
export class Node {
  readonly name: string;
  readonly metadata: Record<string, unknown>;
  readonly checkpoint: CheckpointPolicy;
  /** @internal */
  readonly successors = new Map<string, Target>();

  constructor(
    readonly fn: Step,
    options: NodeOptions = {},
  ) {
    if (typeof fn !== "function") throw new TypeError("A node requires a function");
    this.name = options.name ?? fn.name;
    if (!this.name || this.name === END) throw new Error("Node names must be nonempty; __end__ is reserved");
    this.metadata = structuredClone(options.metadata ?? {});
    this.checkpoint = options.checkpoint ?? false;
  }

  /** Route `action` to `target`; returns this node, so routes can be chained. */
  on(action: string, target: Target): this {
    if (typeof action !== "string" || !action) throw new TypeError("A transition action must be a nonempty string");
    if (!(target instanceof Node) && target !== END) throw new TypeError("A transition target must be a Node or END");
    if (this.successors.has(action)) {
      throw new Error(`Duplicate transition from ${JSON.stringify(this.name)} for ${JSON.stringify(action)}`);
    }
    this.successors.set(action, target);
    return this;
  }

  /** Route the default action to `target`; returns the target, so a pipeline can be chained. */
  then<T extends Target>(target: T): T {
    this.on("default", target);
    return target;
  }
}

/** Wrap a function as a node: `node(fn)` or `node(fn, { name })`. */
export function node(fn: Step, options: NodeOptions = {}): Node {
  return new Node(fn, options);
}

export type Edges = Readonly<Record<string, Readonly<Record<string, string>>>>;

export interface FlowMappings {
  start: string;
  nodes: Readonly<Record<string, Step>>;
  edges: Edges;
  nodeMetadata?: Readonly<Record<string, Record<string, unknown>>>;
  checkpoints?: Readonly<Record<string, CheckpointPolicy>>;
}

function compile(start: Node): FlowMappings {
  const nodes: Record<string, Step> = {};
  const edges: Record<string, Record<string, string>> = {};
  const metadata: Record<string, Record<string, unknown>> = {};
  const checkpoints: Record<string, CheckpointPolicy> = {};
  const names = new Map<string, Node>();
  const pending = [start];
  const visited = new Set<Node>();
  while (pending.length) {
    const current = pending.pop() as Node;
    if (visited.has(current)) continue;
    visited.add(current);
    const existing = names.get(current.name);
    if (existing !== undefined && existing !== current) {
      throw new Error(`Duplicate node name: ${JSON.stringify(current.name)}; give each node a distinct name`);
    }
    names.set(current.name, current);
    nodes[current.name] = current.fn;
    metadata[current.name] = structuredClone(current.metadata);
    if (current.checkpoint !== false) checkpoints[current.name] = current.checkpoint;
    // A leaf ends on the default action. Named actions still need explicit routes.
    const successors = current.successors.size ? current.successors : new Map([["default", END as Target]]);
    edges[current.name] = {};
    for (const [action, target] of successors) {
      edges[current.name][action] = target instanceof Node ? target.name : target;
      if (target instanceof Node) pending.push(target);
    }
  }
  return { start: start.name, nodes, edges, nodeMetadata: metadata, checkpoints };
}

export class Flow {
  readonly start: string;
  readonly nodes: Readonly<Record<string, Step>>;
  readonly edges: Edges;
  readonly nodeMetadata: Readonly<Record<string, Record<string, unknown>>>;
  readonly checkpoints: Readonly<Record<string, CheckpointPolicy>>;

  constructor(start: Node | FlowMappings) {
    const mappings = start instanceof Node ? compile(start) : start;
    this.start = mappings.start;
    this.nodes = Object.freeze({ ...mappings.nodes });
    this.edges = Object.freeze(
      Object.fromEntries(Object.entries(mappings.edges).map(([key, value]) => [key, Object.freeze({ ...value })])),
    );
    this.nodeMetadata = Object.freeze({ ...(mappings.nodeMetadata ?? {}) });
    this.checkpoints = Object.freeze({ ...(mappings.checkpoints ?? {}) });
    if (!(this.start in this.nodes) || END in this.nodes) throw new Error("Flow needs a valid start node; __end__ is reserved");
    for (const [source, actions] of Object.entries(this.edges)) {
      for (const target of Object.values(actions)) {
        if (!(source in this.nodes) || (target !== END && !(target in this.nodes))) {
          throw new Error(`Invalid edge: ${JSON.stringify(source)} -> ${JSON.stringify(target)}`);
        }
      }
    }
  }

  /** Every edge as [source, action, target], sorted. */
  edgeList(): [string, string, string][] {
    const list: [string, string, string][] = [];
    for (const [source, actions] of Object.entries(this.edges)) {
      for (const [action, target] of Object.entries(actions)) list.push([source, action, target]);
    }
    return list.sort((a, b) => (a[0] === b[0] ? (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0) : a[0] < b[0] ? -1 : 1));
  }

  toFlow(): Flow {
    return this;
  }

  toMermaid(options: { direction?: string } = {}): string {
    return buildMermaid(this, options);
  }

  toDot(options: { direction?: string } = {}): string {
    return buildDot(this, options);
  }

  /** Render a PNG or SVG with the local Graphviz `dot`. */
  render(filename: string, options: { direction?: string; dot?: string } = {}): Promise<string> {
    return renderGraph(this, filename, options);
  }

  /**
   * Run from the start node or, when recovering, from the node of the latest
   * checkpoint in the journal; write a checkpoint before entering a node that
   * asks for one, or after an applied compaction.
   */
  async run(runtime: Runtime): Promise<void> {
    let current = runtime.resumePoint() ?? this.start;
    if (!(current in this.nodes)) {
      throw new NonDeterminismError(runtime.seq, `a checkpoint before node ${JSON.stringify(current)}`, "a flow without that node");
    }
    while (current !== END) {
      await runtime.controlPoint(current);
      runtime.step(current, { metadata: this.nodeMetadata[current] });
      let action = await this.nodes[current](runtime);
      if (action === null || action === undefined) action = "default";
      if (typeof action !== "string") throw new TypeError(`Node ${JSON.stringify(current)} must return a string action or nothing`);
      const target = this.edges[current]?.[action];
      if (target === undefined) {
        throw new Error(`No transition from ${JSON.stringify(current)} for action ${JSON.stringify(action)}`);
      }
      runtime.run.emit("transition", { node: current, action, target });
      current = target;
      if (current !== END) {
        const policy = this.checkpoints[current] ?? false;
        if (runtime.checkpointRequested || (typeof policy === "function" ? policy(runtime) : policy)) await runtime.checkpoint(current);
      }
    }
  }
}
