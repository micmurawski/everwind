/** File and shell tools over one directory: read_file, write_file, edit_file and bash. */

import { type ChildProcess, spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";
import { z } from "zod";
import { Mailbox } from "./agentTools.ts";
import { type Tool, type ToolContext, ToolResult, Tools, tool } from "./tools.ts";
import { hexId } from "./util.ts";

export interface WorkspaceOptions {
  /** The directory the tools work in. Default: the process's working directory. */
  root?: string;
  /** Refuse file paths outside root (symlinks resolved). Default true. Bash is not confined. */
  confine?: boolean;
  /** Longest tool output returned to the model; longer output keeps its head and tail. Default 30000. */
  maxOutputChars?: number;
  /** Default and maximum bash timeouts, in milliseconds. Default 120000 and 600000. */
  bashTimeout?: number;
  maxBashTimeout?: number;
  /** The shell bash commands run in. Default "bash". */
  shell?: string;
  /** Extra environment variables for commands. */
  env?: Record<string, string>;
}

const DEFAULT_READ_LINES = 2000;
const MAX_LINE_CHARS = 2000;

/** The real path of `path`, or of its nearest existing ancestor joined with the rest. */
function realish(path: string): string {
  try {
    return realpathSync(path);
  } catch {
    const parent = dirname(path);
    return parent === path ? path : join(realish(parent), basename(path));
  }
}

function inside(root: string, path: string): boolean {
  const rel = relative(root, path);
  return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
}

/** The session a tool call belongs to; calls made outside a run share one. */
function sessionOf(ctx: ToolContext): string {
  try {
    return ctx.runtime.run.sessionId;
  } catch {
    return "";
  }
}

/** Keep the head and tail of long output. */
export function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  const half = Math.floor(max / 2);
  return `${text.slice(0, half)}\n\n... [${text.length - max} characters truncated] ...\n\n${text.slice(text.length - half)}`;
}

/**
 * A directory the file and shell tools work in, with per-session state:
 * the shell's working directory, and which files the session has read.
 *
 * A file must be read before it is overwritten or edited, and must not have
 * changed since. This state is in memory: after a crash, recovery replays
 * earlier reads without running them, so the model reads a file again before
 * changing it.
 */
export class Workspace {
  readonly root: string;
  readonly confine: boolean;
  readonly maxOutputChars: number;
  readonly bashTimeout: number;
  readonly maxBashTimeout: number;
  readonly shell: string;
  readonly env: Record<string, string>;
  private readonly cwds = new Map<string, string>();
  private readonly reads = new Map<string, Map<string, number>>();

  constructor(options: WorkspaceOptions = {}) {
    this.root = realpathSync(resolve(options.root ?? process.cwd()));
    if (!statSync(this.root).isDirectory()) throw new Error(`Workspace root ${this.root} is not a directory`);
    this.confine = options.confine ?? true;
    this.maxOutputChars = options.maxOutputChars ?? 30000;
    this.bashTimeout = options.bashTimeout ?? 120000;
    this.maxBashTimeout = options.maxBashTimeout ?? 600000;
    this.shell = options.shell ?? "bash";
    this.env = { ...(options.env ?? {}) };
    if (!(this.bashTimeout > 0 && this.bashTimeout <= this.maxBashTimeout)) {
      throw new Error("Need 0 < bashTimeout <= maxBashTimeout");
    }
  }

  /** The shell's working directory for a session (starts at root). */
  cwd(sessionId = ""): string {
    return this.cwds.get(sessionId) ?? this.root;
  }

  /** @internal */
  setCwd(sessionId: string, cwd: string): boolean {
    if (this.confine && !inside(this.root, realish(cwd))) {
      this.cwds.delete(sessionId);
      return false;
    }
    this.cwds.set(sessionId, cwd);
    return true;
  }

  /** An absolute path: relative paths start at the session's working directory. */
  resolve(path: string, sessionId = ""): string {
    const absolute = resolve(this.cwd(sessionId), path);
    if (this.confine && !inside(this.root, realish(absolute))) {
      throw new Error(`${path} is outside the workspace ${this.root}`);
    }
    return absolute;
  }

  /** @internal Remember a file's modification time as this session saw it. */
  markRead(sessionId: string, path: string): void {
    let files = this.reads.get(sessionId);
    if (files === undefined) this.reads.set(sessionId, (files = new Map()));
    files.set(path, statSync(path).mtimeMs);
  }

  /** @internal Throw unless the session read the existing file and it has not changed since. */
  checkFresh(sessionId: string, path: string): void {
    const seen = this.reads.get(sessionId)?.get(path);
    if (seen === undefined) throw new Error(`Read ${path} with read_file before changing it`);
    if (statSync(path).mtimeMs !== seen) throw new Error(`${path} changed since it was read; read it again`);
  }
}

function exists(path: string): boolean {
  try {
    statSync(path);
    return true;
  } catch {
    return false;
  }
}

/** Read a text file, numbered like `cat -n`. */
export function readTool(workspace: Workspace, options: { name?: string } = {}): Tool {
  return tool({
    name: options.name ?? "read_file",
    description:
      "Read a text file. Returns lines numbered like cat -n, up to 2000 from offset. " +
      "Relative paths start at the shell's working directory. Read a file before you edit or overwrite it.",
    effect: "read",
    input: z.object({
      filePath: z.string().min(1).describe("Path of the file"),
      offset: z.number().int().min(1).optional().describe("First line to read, counted from 1"),
      limit: z.number().int().min(1).optional().describe("Number of lines to read (default 2000)"),
    }),
    run({ filePath, offset, limit }, ctx) {
      const session = sessionOf(ctx);
      const path = workspace.resolve(filePath, session);
      if (!exists(path)) throw new Error(`${path} does not exist`);
      if (statSync(path).isDirectory()) throw new Error(`${path} is a directory; list it with bash`);
      const bytes = readFileSync(path);
      if (bytes.subarray(0, 8000).includes(0)) throw new Error(`${path} is a binary file`);
      workspace.markRead(session, path);
      const text = bytes.toString("utf-8");
      if (text === "") return `${path} is empty.`;
      const lines = text.split("\n");
      if (lines[lines.length - 1] === "") lines.pop();
      const start = (offset ?? 1) - 1;
      if (start >= lines.length) throw new Error(`${path} has ${lines.length} lines; offset ${offset} is past the end`);
      const stop = Math.min(start + (limit ?? DEFAULT_READ_LINES), lines.length);
      const body = lines
        .slice(start, stop)
        .map((line, index) => {
          const shown = line.length > MAX_LINE_CHARS ? `${line.slice(0, MAX_LINE_CHARS)}... [line truncated]` : line;
          return `${String(start + index + 1).padStart(6)}\t${shown}`;
        })
        .join("\n");
      const more = stop < lines.length ? `\n... ${lines.length - stop} more lines; continue with offset=${stop + 1}` : "";
      return truncate(body, workspace.maxOutputChars) + more;
    },
  });
}

/** Create or overwrite a file. An existing file must have been read first. */
export function writeTool(workspace: Workspace, options: { name?: string } = {}): Tool {
  return tool({
    name: options.name ?? "write_file",
    description:
      "Write a file, creating parent directories. Overwrites an existing file, which you must have read " +
      "with read_file first. Prefer edit_file to change part of a file.",
    // Writing the same content twice leaves the same file.
    effect: "idempotent",
    input: z.object({
      filePath: z.string().min(1).describe("Path of the file"),
      content: z.string().describe("The whole new content"),
    }),
    run({ filePath, content }, ctx) {
      const session = sessionOf(ctx);
      const path = workspace.resolve(filePath, session);
      const existed = exists(path);
      if (existed) {
        if (statSync(path).isDirectory()) throw new Error(`${path} is a directory`);
        workspace.checkFresh(session, path);
      }
      mkdirSync(dirname(path), { recursive: true });
      writeFileSync(path, content, "utf-8");
      workspace.markRead(session, path);
      return `${existed ? "Updated" : "Created"} ${path} (${Buffer.byteLength(content)} bytes)`;
    },
  });
}

/** Replace an exact string in a file that was read first. */
export function editTool(workspace: Workspace, options: { name?: string } = {}): Tool {
  return tool({
    name: options.name ?? "edit_file",
    description:
      "Replace exact text in a file you have read with read_file. oldString must occur exactly once, " +
      "with its indentation, unless replaceAll is true. Include surrounding lines to make it unique.",
    // A repeated edit could apply twice or fail: after a crash its outcome is unknown.
    effect: "unsafe",
    input: z.object({
      filePath: z.string().min(1).describe("Path of the file"),
      oldString: z.string().min(1).describe("The exact text to replace"),
      newString: z.string().describe("The replacement text"),
      replaceAll: z.boolean().optional().describe("Replace every occurrence (default false)"),
    }),
    run({ filePath, oldString, newString, replaceAll }, ctx) {
      const session = sessionOf(ctx);
      const path = workspace.resolve(filePath, session);
      if (!exists(path)) throw new Error(`${path} does not exist; create it with write_file`);
      workspace.checkFresh(session, path);
      if (oldString === newString) throw new Error("oldString and newString are the same");
      const text = readFileSync(path, "utf-8");
      const parts = text.split(oldString);
      const count = parts.length - 1;
      if (count === 0) throw new Error(`oldString was not found in ${path}`);
      if (count > 1 && !replaceAll) {
        throw new Error(`oldString occurs ${count} times in ${path}; add context to make it unique, or set replaceAll`);
      }
      // split/join: no `$` patterns in the replacement.
      const updated = replaceAll ? parts.join(newString) : parts[0] + newString + parts.slice(1).join(oldString);
      writeFileSync(path, updated, "utf-8");
      workspace.markRead(session, path);
      return `Edited ${path}: ${count === 1 ? "1 replacement" : `${count} replacements`}`;
    },
  });
}

interface ShellOutcome {
  output: string;
  exitCode: number | null;
  signal: string | null;
  timedOut: boolean;
  /** The shell's working directory when it exited, if it could be recorded. */
  cwd: string | null;
}

/** Run one command in its own process group, so a timeout stops its children too. */
function startShell(
  workspace: Workspace,
  command: string,
  cwd: string,
  options: { timeoutMs: number | null; signal?: AbortSignal },
): { done: Promise<ShellOutcome>; kill: () => void } {
  const scratch = mkdtempSync(join(tmpdir(), "everwind-bash-"));
  const cwdFile = join(scratch, "cwd");
  // The EXIT trap records the final directory and keeps the command's exit status.
  const script = `trap 'builtin pwd -P > "$EVERWIND_CWD_FILE"' EXIT\n${command}`;
  const child: ChildProcess = spawn(workspace.shell, ["-c", script], {
    cwd,
    env: { ...process.env, ...workspace.env, EVERWIND_CWD_FILE: cwdFile },
    stdio: ["ignore", "pipe", "pipe"],
    detached: true,
  });
  const chunks: string[] = [];
  let size = 0;
  const keep = (data: Buffer) => {
    // Bounded: enough for the head and tail the model sees.
    if (size > workspace.maxOutputChars * 4) return;
    const text = data.toString("utf-8");
    chunks.push(text);
    size += text.length;
  };
  child.stdout?.on("data", keep);
  child.stderr?.on("data", keep);
  let timedOut = false;
  let killTimer: ReturnType<typeof setTimeout> | null = null;
  const kill = () => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    const group = (signal: NodeJS.Signals) => {
      try {
        process.kill(-(child.pid as number), signal);
      } catch {
        child.kill(signal);
      }
    };
    group("SIGTERM");
    killTimer ??= setTimeout(() => group("SIGKILL"), 2000);
  };
  const timer =
    options.timeoutMs === null
      ? null
      : setTimeout(() => {
          timedOut = true;
          kill();
        }, options.timeoutMs);
  options.signal?.addEventListener("abort", kill, { once: true });
  const done = new Promise<ShellOutcome>((resolveDone, reject) => {
    child.on("error", (error) => {
      if (timer !== null) clearTimeout(timer);
      rmSync(scratch, { recursive: true, force: true });
      reject(error);
    });
    child.on("close", (exitCode, signal) => {
      if (timer !== null) clearTimeout(timer);
      if (killTimer !== null) clearTimeout(killTimer);
      options.signal?.removeEventListener("abort", kill);
      let finalCwd: string | null = null;
      try {
        finalCwd = readFileSync(cwdFile, "utf-8").trim() || null;
      } catch {}
      rmSync(scratch, { recursive: true, force: true });
      resolveDone({ output: chunks.join(""), exitCode, signal, timedOut, cwd: finalCwd });
    });
  });
  return { done, kill };
}

function describeOutcome(outcome: ShellOutcome, workspace: Workspace, timeoutMs: number | null): string {
  let text = truncate(outcome.output.replace(/\s+$/, ""), workspace.maxOutputChars);
  if (outcome.timedOut) text += `\n[Timed out after ${timeoutMs} ms; the command was killed]`;
  else if (outcome.exitCode !== 0) text += `\n[Exit code ${outcome.exitCode ?? outcome.signal}]`;
  return text || "(no output)";
}

class BackgroundShell {
  status: "running" | "completed" | "failed" | "killed" = "running";
  exitCode: number | null = null;
  output: string | null = null;

  constructor(
    readonly id: string,
    readonly sessionId: string,
    readonly command: string,
    readonly description: string | null,
    readonly kill: () => void,
    readonly done: Promise<unknown>,
  ) {}

  snapshot(): Record<string, unknown> {
    return {
      shellId: this.id,
      command: this.command,
      description: this.description,
      status: this.status,
      exitCode: this.exitCode,
      ...(this.output === null ? {} : { output: this.output }),
    };
  }
}

/**
 * Commands started with runInBackground. Each one's end is put in the mailbox
 * as `{ kind: "ShellFinished", ... }`. Pass it as the `shells` context value,
 * and call close() (or `await using`) to kill what is still running.
 */
export class BackgroundShells {
  private readonly shells = new Map<string, BackgroundShell>();
  private closed = false;

  constructor(readonly mailbox: Mailbox = new Mailbox()) {}

  start(workspace: Workspace, command: string, options: { sessionId: string; description?: string | null }): Record<string, unknown> {
    if (this.closed) throw new Error("BackgroundShells is closed");
    const id = hexId();
    const { done, kill } = startShell(workspace, command, workspace.cwd(options.sessionId), { timeoutMs: null });
    let shell: BackgroundShell;
    const settled = done.then(
      (outcome) => {
        shell.exitCode = outcome.exitCode;
        shell.output = describeOutcome(outcome, workspace, null);
        if (shell.status !== "killed") shell.status = outcome.exitCode === 0 ? "completed" : "failed";
      },
      (error) => {
        shell.status = "failed";
        shell.output = String(error);
      },
    );
    shell = new BackgroundShell(id, options.sessionId, command, options.description ?? null, kill, settled);
    this.shells.set(id, shell);
    void settled.then(() => this.mailbox.put({ kind: "ShellFinished", sessionId: options.sessionId, ...shell.snapshot() }));
    return shell.snapshot();
  }

  list(sessionId: string): Record<string, unknown>[] {
    return [...this.shells.values()].filter((shell) => shell.sessionId === sessionId).map((shell) => shell.snapshot());
  }

  kill(shellId: string, sessionId: string): Record<string, unknown> {
    const shell = this.shells.get(shellId);
    if (shell === undefined || shell.sessionId !== sessionId) throw new Error(`Unknown shell ${shellId}`);
    if (shell.status === "running") {
      shell.status = "killed";
      shell.kill();
    }
    return shell.snapshot();
  }

  async close(): Promise<void> {
    this.closed = true;
    for (const shell of this.shells.values()) if (shell.status === "running") this.kill(shell.id, shell.sessionId);
    await Promise.allSettled([...this.shells.values()].map((shell) => shell.done));
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.close();
  }
}

/**
 * Run a shell command in the workspace. The working directory persists between
 * calls of one session; environment variables and shell state do not.
 */
export function bashTool(workspace: Workspace, options: { name?: string } = {}): Tool {
  return tool({
    name: options.name ?? "bash",
    description:
      `Run a ${workspace.shell} command and return its combined output and exit code. ` +
      "`cd` carries over to the next call; variables and other shell state do not. " +
      `Timeout in milliseconds (default ${workspace.bashTimeout}, max ${workspace.maxBashTimeout}). ` +
      "With runInBackground the command runs detached and a ShellFinished message arrives when it exits.",
    effect: "unsafe",
    input: z.object({
      command: z.string().min(1).describe("The command to run"),
      description: z.string().optional().describe("What the command does, in a few words"),
      timeout: z.number().int().min(1).max(workspace.maxBashTimeout).optional().describe("Timeout in milliseconds"),
      runInBackground: z.boolean().optional().describe("Run detached; you are notified when it exits"),
    }),
    async run({ command, description, timeout, runInBackground }, ctx) {
      const session = sessionOf(ctx);
      if (runInBackground) {
        return ctx.get<BackgroundShells>("shells").start(workspace, command, { sessionId: session, description });
      }
      const timeoutMs = timeout ?? workspace.bashTimeout;
      const outcome = await startShell(workspace, command, workspace.cwd(session), { timeoutMs, signal: ctx.signal }).done;
      let text = describeOutcome(outcome, workspace, timeoutMs);
      if (outcome.cwd !== null && outcome.cwd !== workspace.cwd(session) && !workspace.setCwd(session, outcome.cwd)) {
        text += `\n[Working directory ${outcome.cwd} is outside the workspace; reset to ${workspace.root}]`;
      }
      return outcome.exitCode === 0 && !outcome.timedOut ? text : new ToolResult(null, text);
    },
  });
}

/** List and kill this session's background commands. Needs the `shells` context value. */
export function shellJobTools(): Tools {
  return new Tools([
    tool({
      name: "list_shells",
      description: "List background commands started with bash runInBackground, with their status and output.",
      effect: "read",
      run: (_args, ctx) => ctx.get<BackgroundShells>("shells").list(sessionOf(ctx)),
    }),
    tool({
      name: "kill_shell",
      description: "Kill a background command; ShellFinished confirms it.",
      input: z.object({ shellId: z.string() }),
      run: ({ shellId }, ctx) => ctx.get<BackgroundShells>("shells").kill(shellId, sessionOf(ctx)),
    }),
  ]);
}

/** bash, read_file, write_file and edit_file over one workspace. */
export function workspaceTools(workspace: Workspace): Tools {
  return new Tools([bashTool(workspace), readTool(workspace), writeTool(workspace), editTool(workspace)]);
}
