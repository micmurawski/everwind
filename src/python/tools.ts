/** Session-scoped Python tools, independent of providers and data domains. */

import { z } from "zod";
import { validateSessionId } from "../sessionLog.ts";
import { Tools, tool } from "../tools.ts";
import { DockerPythonSandbox, type PythonSandbox } from "./sandbox.ts";

/**
 * Own one lazily created sandbox per agent session; no global registry.
 *
 * Keep this owner alive across Agent.run continuations to retain interpreter
 * state. Session stores persist messages, not running Python interpreters.
 */
export class PythonSessions {
  private readonly sandboxes = new Map<string, PythonSandbox>();
  private closed = false;

  constructor(readonly factory: (sessionId: string) => PythonSandbox = () => new DockerPythonSandbox()) {}

  get(sessionId: string): PythonSandbox {
    validateSessionId(sessionId);
    if (this.closed) throw new Error("PythonSessions is closed");
    let sandbox = this.sandboxes.get(sessionId);
    if (sandbox === undefined) this.sandboxes.set(sessionId, (sandbox = this.factory(sessionId)));
    return sandbox;
  }

  async close(): Promise<void> {
    this.closed = true;
    const results = await Promise.allSettled([...this.sandboxes.values()].map((sandbox) => sandbox.close()));
    const errors = results.flatMap((result) => (result.status === "rejected" ? [result.reason] : []));
    if (errors.length) throw new AggregateError(errors, "Failed to close Python sandboxes");
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.close();
  }
}

/** Expose execution, history and reset; container configuration stays application-owned. */
export function pythonTools(sessions: PythonSessions, options: { description?: string } = {}): Tools {
  let description =
    "Execute Python with persistent variables in this session's container.\n\n" +
    "Use print() for output; the last expression's repr is also returned. Read stdout, stderr, result, and error. " +
    "Errors may leave partial state changes. If stateLost is true, reset_python is required. /workspace is the " +
    "working directory; input supplied by the application is available as `data`.";
  if (options.description) description += `\n${options.description}`;
  const executePython = tool({
    name: "execute_python",
    description,
    input: z.object({ code: z.string() }),
    run: async ({ code }, ctx) => {
      const sandbox = sessions.get(ctx.runtime.run.sessionId);
      return sandbox instanceof DockerPythonSandbox ? sandbox.execute(code, ctx.signal) : sandbox.execute(code);
    },
  });
  const pythonHistory = tool({
    name: "python_history",
    description: "Read recent Python code and results for this session, oldest first.",
    input: z.object({ limit: z.number().int().default(10) }),
    effect: "read",
    run: ({ limit }, ctx) => sessions.get(ctx.runtime.run.sessionId).history(limit),
  });
  const resetPython = tool({
    name: "reset_python",
    description:
      "Discard this session's interpreter, temporary files and history. The next execution creates a fresh " +
      "container and reruns application setup. Files on explicitly writable host mounts survive reset.",
    run: async (_args, ctx) => {
      await sessions.get(ctx.runtime.run.sessionId).reset();
      return { status: "reset", sessionId: ctx.runtime.run.sessionId };
    },
  });
  return new Tools([executePython, pythonHistory, resetPython]);
}
