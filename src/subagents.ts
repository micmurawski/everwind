/** One `agent` tool that delegates to a named agent type, and `list_agents` to discover them. */

import { z } from "zod";
import { Agent } from "./agent.ts";
import { AgentJobs, type Continuation, runResult } from "./agentTools.ts";
import { type Tool, Tools, tool } from "./tools.ts";

export interface SubagentType {
  agent: Agent;
  /** When the model should pick this type. */
  description?: string;
  /** The child's application context; it never inherits the parent's. */
  context?: Record<string, unknown>;
}

export interface SubagentOptions {
  /** Name of the delegating tool. Default "agent". */
  name?: string;
  /** Name of the listing tool. Default "list_agents". */
  listName?: string;
  /** The type used when the model names none. Default: the first type. */
  defaultType?: string;
}

/**
 * `agent` runs a task in a child agent of the chosen type, and `list_agents`
 * lists the types and this parent's background jobs.
 *
 * `runInBackground` needs AgentJobs as the parent's `agentJobs` context value;
 * an AgentFinished message arrives in its mailbox when the child ends. Pass a
 * returned sessionId back to continue that child's conversation.
 */
export function subagentTools(types: Record<string, Agent | SubagentType>, options: SubagentOptions = {}): Tools {
  const entries = Object.entries(types).map(([type, value]) => [type, value instanceof Agent ? { agent: value } : value] as const);
  if (!entries.length) throw new Error("subagentTools needs at least one agent type");
  const registry = new Map<string, SubagentType>(entries);
  const names = [...registry.keys()] as [string, ...string[]];
  const defaultType = options.defaultType ?? names[0];
  if (!registry.has(defaultType)) throw new Error(`Unknown default agent type ${defaultType}`);
  const name = options.name ?? "agent";
  const listName = options.listName ?? "list_agents";
  const hint =
    (type: string): Continuation =>
    (sessionId) =>
      `To continue, call ${name}(subagentType="${type}", prompt="...", sessionId="${sessionId}").`;
  const catalogue = entries.map(([type, entry]) => `- ${type}: ${entry.description ?? `the ${entry.agent.name} agent`}`).join("\n");

  const delegate: Tool = tool({
    name,
    description:
      "Hand a task to a sub-agent, which works on it alone and returns its answer. Give it a complete, " +
      "self-contained prompt: it does not see this conversation. Omit sessionId to start a new " +
      "conversation; pass one to continue it. With runInBackground, a job handle returns at once and " +
      `AgentFinished arrives when it ends.\nAgent types (default ${defaultType}):\n${catalogue}`,
    input: z.object({
      description: z.string().min(1).describe("A short (3-5 word) description of the task"),
      prompt: z.string().min(1).describe("The task for the agent"),
      subagentType: z.enum(names).optional().describe("Which agent type to use"),
      runInBackground: z.boolean().optional().describe("Return a job handle at once instead of waiting"),
      sessionId: z.string().nullable().optional().describe("A sessionId returned earlier, to continue that conversation"),
    }),
    async run({ description, prompt, subagentType, runInBackground, sessionId }, ctx) {
      const type = subagentType ?? defaultType;
      const entry = registry.get(type) as SubagentType;
      const runtime = ctx.runtime;
      const context = { ...(entry.context ?? {}) };
      if (runInBackground) {
        return ctx.get<AgentJobs>("agentJobs").start(entry.agent, prompt, {
          sessionId,
          runtime,
          toolName: name,
          context,
          continuation: hint(type),
          details: { subagentType: type, description },
        });
      }
      const run = await entry.agent.run(prompt, {
        sessionId: sessionId ?? undefined,
        context,
        parentSessionId: runtime.run.sessionId,
        parentRunId: runtime.run.id,
        signal: ctx.signal,
      });
      return { subagentType: type, description, ...runResult(run, hint(type)) };
    },
  });

  const list: Tool = tool({
    name: listName,
    description: `List the agent types ${name} can start, and this conversation's background agent jobs.`,
    effect: "read",
    run(_args, ctx) {
      const agents = entries.map(([type, entry]) => ({
        subagentType: type,
        agent: entry.agent.name,
        description: entry.description ?? null,
        default: type === defaultType,
      }));
      if (!ctx.has("agentJobs")) return { agents };
      return { agents, jobs: ctx.get<AgentJobs>("agentJobs").listJobs(ctx.runtime.run.sessionId) };
    },
  });

  return new Tools([delegate, list]);
}
