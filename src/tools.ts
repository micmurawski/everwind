/** Typed tools with a strict boundary between model input and injected context. */

import { z } from "zod";
import { MissingContext, describeError, isFatalToolError } from "./errors.ts";
import type { Runtime } from "./runtime.ts";
import type { ToolShared } from "./shared.ts";
import type { ToolCall, ToolSpec } from "./types.ts";
import { type JsonObject, clone } from "./util.ts";

/**
 * How recovery treats a call that was started but has no recorded result:
 * read and idempotent calls run again (idempotent ones with the same key);
 * unsafe calls are not repeated and get an "outcome unknown" error result.
 */
export type Effect = "read" | "idempotent" | "unsafe";
export const EFFECTS: ReadonlySet<Effect> = new Set(["read", "idempotent", "unsafe"]);

export class ToolResult {
  constructor(
    readonly result: unknown = null,
    readonly error: string | null = null,
  ) {}

  content(): string {
    if (this.error !== null) return this.error;
    if (typeof this.result === "string") return this.result;
    const text = JSON.stringify(this.result === undefined ? null : this.result);
    if (text === undefined) throw new TypeError("Tool result is not JSON-serializable");
    return text;
  }
}

/** What the runtime passes to a tool next to its model-supplied arguments. */
export interface ToolContextInit {
  /** The run's application context (`agent.run(prompt, { context })`). Never logged. */
  values?: Readonly<Record<string, unknown>>;
  runtime?: Runtime;
  /** Stable across crash recovery: pass it to idempotent external APIs. */
  idempotencyKey?: string;
  /** Deduplicating access to the agent's shared world. */
  shared?: ToolShared;
  /** Aborted when the tool times out or the run stops. */
  signal?: AbortSignal;
}

export class ToolContext {
  readonly values: Readonly<Record<string, unknown>>;
  readonly signal: AbortSignal;
  private readonly init: ToolContextInit;

  constructor(
    readonly toolName: string,
    init: ToolContextInit,
  ) {
    this.init = init;
    this.values = init.values ?? {};
    this.signal = init.signal ?? new AbortController().signal;
  }

  get runtime(): Runtime {
    return this.require("runtime", this.init.runtime);
  }

  get idempotencyKey(): string {
    return this.require("idempotencyKey", this.init.idempotencyKey);
  }

  get shared(): ToolShared {
    return this.require("shared", this.init.shared);
  }

  /** The run's scope (agent.run(prompt, { scope })); empty outside a run. */
  get scope(): JsonObject {
    return this.init.runtime?.run.scope ?? {};
  }

  /** An application context value; a missing one fails the run, not just the call. */
  get<T>(key: string): T {
    return this.require(key, this.values[key] as T | undefined, key in this.values);
  }

  has(key: string): boolean {
    return key in this.values;
  }

  private require<T>(key: string, value: T | undefined, present = value !== undefined): T {
    if (!present) throw new MissingContext(`Tool ${this.toolName} requires injected context '${key}'`);
    return value as T;
  }
}

type Shape = z.ZodObject<z.ZodRawShape>;

export interface ToolDefinition<S extends Shape> {
  name: string;
  description?: string | ((scope: JsonObject) => string);
  /**
   * Model-supplied arguments. Unknown keys and type mismatches are rejected.
   * A function builds them from the run's scope, e.g. an enum of the agents
   * this task may start.
   */
  input?: S | ((scope: JsonObject) => S);
  /** Whether the model sees this tool in a run with this scope. Default: always. */
  available?: (scope: JsonObject) => boolean;
  tags?: Iterable<string>;
  effect?: Effect;
  run(args: z.output<S>, ctx: ToolContext): unknown;
}

export class Tool<S extends Shape = Shape> {
  readonly name: string;
  readonly description: string;
  readonly tags: ReadonlySet<string>;
  readonly effect: Effect;
  readonly input: z.ZodObject<z.ZodRawShape>;
  readonly spec: ToolSpec;
  /** Its description or input depends on the run's scope: bind() it before use. */
  readonly scoped: boolean;
  private readonly fn: ToolDefinition<S>["run"];
  private readonly definition: ToolDefinition<S>;

  constructor(definition: ToolDefinition<S>, scope: JsonObject | null = null) {
    const effect = definition.effect ?? "unsafe";
    if (!EFFECTS.has(effect)) throw new Error(`effect must be one of ${[...EFFECTS].sort().join(", ")}`);
    if (!definition.name) throw new Error("A tool needs a name");
    this.definition = definition;
    this.name = definition.name;
    this.scoped = scope === null && (typeof definition.input === "function" || typeof definition.description === "function");
    const description = typeof definition.description === "function" ? (scope === null ? "" : definition.description(scope)) : definition.description;
    this.description = description ?? "";
    this.tags = new Set(definition.tags ?? []);
    this.effect = effect;
    const input = typeof definition.input === "function" ? (scope === null ? z.object({}) : definition.input(scope)) : definition.input;
    this.input = (input ?? z.object({})).strict();
    this.fn = definition.run;
    const { $schema: _, ...parameters } = z.toJSONSchema(this.input) as Record<string, unknown>;
    // An unbound scoped tool has no real schema; its spec marks it, so fingerprints still see the tool.
    this.spec = { name: this.name, description: this.description, parameters: this.scoped ? { scoped: true } : parameters };
  }

  /** This tool as a run with `scope` sees it, or null when it is not available there. */
  bind(scope: JsonObject): Tool<S> | null {
    if (this.definition.available !== undefined && !this.definition.available(scope)) return null;
    return this.scoped ? new Tool(this.definition, scope) : this;
  }

  async invoke(args: Record<string, unknown>, context: ToolContextInit = {}): Promise<ToolResult> {
    if (this.scoped) {
      // Called outside a runtime, e.g. directly in a test: bind to the context's scope first.
      const bound = this.bind(context.runtime?.run.scope ?? {});
      if (bound === null) return new ToolResult(null, `Tool ${this.name} is not available in this scope`);
      return bound.invoke(args, context);
    }
    const parsed = this.input.safeParse(args);
    if (!parsed.success) {
      return new ToolResult(null, `Invalid arguments for ${this.name}: ${z.prettifyError(parsed.error)}`);
    }
    // Input errors are recoverable by the model. Missing runtime dependencies are not.
    try {
      const value = await this.fn(parsed.data as z.output<S>, new ToolContext(this.name, context));
      const result = value instanceof ToolResult ? value : new ToolResult(value);
      result.content(); // Check serialization before handing output to a provider.
      return result;
    } catch (error) {
      if (isFatalToolError(error)) throw error;
      // Tool failures become observations; a timeout or cancellation is raised by the caller.
      return new ToolResult(null, describeError(error));
    }
  }
}

/** Define a tool: `tool({ name, description, input: z.object({...}), effect, run })`. */
export function tool<S extends Shape = z.ZodObject<{}>>(definition: ToolDefinition<S>): Tool<S> {
  return new Tool(definition);
}

export class Tools {
  private readonly registry: ReadonlyMap<string, Tool>;

  constructor(tools: Iterable<Tool> = []) {
    const registry = new Map<string, Tool>();
    for (const item of tools) {
      const existing = registry.get(item.name);
      if (existing !== undefined && existing !== item) throw new Error(`Duplicate tool name: ${item.name}`);
      registry.set(item.name, item);
    }
    this.registry = registry;
  }

  static of(...items: (Tool | Tools)[]): Tools {
    return new Tools(items.flatMap((item) => (item instanceof Tools ? item.list() : [item])));
  }

  list(): Tool[] {
    return [...this.registry.values()];
  }

  get specs(): ToolSpec[] {
    return this.list().map((item) => clone(item.spec));
  }

  with(...items: (Tool | Tools)[]): Tools {
    return Tools.of(this, ...items);
  }

  get(name: string): Tool | undefined {
    return this.registry.get(name);
  }

  /** Effect kind of a tool; an unknown tool performs nothing, so it is a read. */
  effect(name: string): Effect {
    return this.registry.get(name)?.effect ?? "read";
  }

  /** The tools a run with this scope sees: unavailable ones left out, scoped ones bound. */
  bind(scope: JsonObject): Tools {
    return new Tools(this.list().flatMap((item) => item.bind(scope) ?? []));
  }

  select(...tags: string[]): Tools {
    return new Tools(this.list().filter((item) => tags.every((tag) => item.tags.has(tag))));
  }

  async invoke(call: ToolCall, context: ToolContextInit = {}): Promise<ToolResult> {
    const item = this.registry.get(call.name);
    if (item === undefined) {
      return new ToolResult(
        null,
        `Unknown tool ${JSON.stringify(call.name)}; available: ${[...this.registry.keys()].join(", ")}`,
      );
    }
    return item.invoke(call.arguments, context);
  }
}
