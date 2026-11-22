/** Persistent Python execution in an explicitly configured Docker container. */

import type { Subprocess } from "bun";
import { realpathSync } from "node:fs";
import { posix } from "node:path";
import { fileURLToPath } from "node:url";
import { KeyedMutex, hexId } from "../util.ts";

export const WORKER_PATH = fileURLToPath(new URL("./worker.py", import.meta.url));
const WORKER_TARGET = "/opt/tbln_worker.py";

export interface PythonResult {
  stdout: string;
  stderr: string;
  result: string | null;
  error: string | null;
  truncated: boolean;
  stateLost: boolean;
  /** Seconds. */
  duration: number;
}

export interface Execution {
  code: string;
  result: PythonResult;
}

function pythonResult(fields: Partial<PythonResult>): PythonResult {
  return { stdout: "", stderr: "", result: null, error: null, truncated: false, stateLost: false, duration: 0, ...fields };
}

export interface PythonSandbox {
  execute(code: string): Promise<PythonResult>;
  reset(): Promise<void>;
  close(): Promise<void>;
  history(limit?: number): Execution[];
}

export interface Mount {
  source: string;
  target: string;
  /** Defaults to true. */
  readOnly?: boolean;
}

export interface DockerPythonSandboxOptions {
  image?: string;
  mounts?: Mount[];
  env?: Record<string, string>;
  setup?: string;
  data?: unknown;
  network?: string;
  memory?: string;
  cpus?: number;
  pidsLimit?: number;
  /** Seconds per snippet. */
  timeout?: number;
  startupTimeout?: number;
  maxOutputBytes?: number;
  maxCodeBytes?: number;
  historyLimit?: number;
  docker?: string;
}

class Timeout extends Error {}

function withTimeout<T>(promise: Promise<T>, seconds: number, onTimeout: () => void): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  return Promise.race([
    promise,
    new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        onTimeout();
        reject(new Timeout("Execution deadline exceeded"));
      }, seconds * 1000);
    }),
  ]).finally(() => clearTimeout(timer));
}

/** Reads newline-terminated lines from a byte stream. */
class LineReader {
  private buffer = "";
  private readonly decoder = new TextDecoder();
  private readonly reader: { read(): Promise<{ value?: Uint8Array; done: boolean }> };

  constructor(stream: ReadableStream<Uint8Array>) {
    this.reader = stream.getReader();
  }

  /** The next line without its newline, or null at the end of the stream. */
  async readLine(): Promise<string | null> {
    for (;;) {
      const index = this.buffer.indexOf("\n");
      if (index >= 0) {
        const line = this.buffer.slice(0, index);
        this.buffer = this.buffer.slice(index + 1);
        return line;
      }
      const { value, done } = await this.reader.read();
      if (done) return null;
      this.buffer += this.decoder.decode(value, { stream: true });
    }
  }
}

/**
 * One interpreter, serialized snippets, bounded output, explicit lifecycle.
 *
 * The Docker CLI and a running local daemon are required. Images must already
 * exist; no SDK, port, automatic pip install, or implicit host execution is used.
 */
export class DockerPythonSandbox implements PythonSandbox {
  readonly image: string;
  readonly mounts: readonly Mount[];
  readonly env: Readonly<Record<string, string>>;
  readonly setup: string;
  readonly network: string;
  readonly memory: string;
  readonly cpus: number;
  readonly pidsLimit: number;
  readonly timeout: number;
  readonly startupTimeout: number;
  readonly maxOutputBytes: number;
  readonly maxCodeBytes: number;
  readonly historyLimit: number;
  readonly docker: string;
  private readonly dataJson: string;
  private readonly executions: Execution[] = [];
  private readonly mutex = new KeyedMutex();
  /** @internal */
  process: Subprocess<"pipe", "pipe", "pipe"> | null = null;
  private lines: LineReader | null = null;
  private diagnostics = "";
  private stderrTask: Promise<void> | null = null;
  private mayExist = false;
  protected name: string | null = null;
  private closed = false;
  private lost = false;

  constructor(options: DockerPythonSandboxOptions = {}) {
    this.image = options.image ?? "python:3.12-slim";
    this.mounts = options.mounts ?? [];
    this.env = { ...(options.env ?? {}) };
    this.setup = options.setup ?? "";
    this.network = options.network ?? "none";
    this.memory = options.memory ?? "512m";
    this.cpus = options.cpus ?? 1;
    this.pidsLimit = options.pidsLimit ?? 64;
    this.timeout = options.timeout ?? 30;
    this.startupTimeout = options.startupTimeout ?? 30;
    this.maxOutputBytes = options.maxOutputBytes ?? 32_768;
    this.maxCodeBytes = options.maxCodeBytes ?? 262_144;
    this.historyLimit = options.historyLimit ?? 100;
    this.docker = options.docker ?? "docker";
    for (const [name, value] of [["timeout", this.timeout], ["startupTimeout", this.startupTimeout], ["cpus", this.cpus]] as const) {
      if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be finite and positive`);
    }
    if (Math.min(this.pidsLimit, this.maxOutputBytes, this.maxCodeBytes, this.historyLimit) < 1) {
      throw new Error("Resource and history limits must be positive");
    }
    this.dataJson = JSON.stringify(options.data ?? null);
  }

  /** @internal The `docker create` command line. */
  createCommand(): string[] {
    const command = [
      this.docker, "create", "--rm", "-i", "--pull=never",
      "--name", this.name as string,
      "--network", this.network,
      "--memory", this.memory,
      "--cpus", String(this.cpus),
      "--pids-limit", String(this.pidsLimit),
      "--cap-drop=ALL",
      "--security-opt=no-new-privileges",
      "--read-only",
      "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
      "--workdir", "/workspace",
      "--env", "PYTHONDONTWRITEBYTECODE=1",
      "--mount", `type=bind,source=${realpathSync(WORKER_PATH)},target=${WORKER_TARGET},readonly`,
    ];
    const targets = new Set<string>();
    for (const mount of this.mounts) {
      const source = realpathSync(mount.source);
      const target = posix.normalize(mount.target);
      if (
        !posix.isAbsolute(mount.target) ||
        mount.target.split("/").includes("..") ||
        ["/", "/opt", WORKER_TARGET, "/tmp"].includes(target) ||
        targets.has(target) ||
        source.includes(",") ||
        target.includes(",")
      ) {
        throw new Error(`Unsupported or duplicate mount target: ${mount.target}`);
      }
      targets.add(target);
      command.push("--mount", `type=bind,source=${source},target=${target}${mount.readOnly === false ? "" : ",readonly"}`);
    }
    if (!targets.has("/workspace")) command.push("--tmpfs", "/workspace:rw,nosuid,nodev,size=128m,mode=1777");
    for (const [key, value] of Object.entries(this.env)) {
      if (!key || key.includes("=") || key.includes("\0") || value.includes("\0")) throw new Error("Invalid container environment entry");
      command.push("--env", `${key}=${value}`);
    }
    command.push("--entrypoint", "python", this.image, "-I", "-u", WORKER_TARGET);
    return command;
  }

  protected async createContainer(): Promise<void> {
    const child = Bun.spawn(this.createCommand(), { stdout: "pipe", stderr: "pipe" });
    this.mayExist = true;
    const errors = new Response(child.stderr).text();
    const code = await withTimeout(child.exited, this.startupTimeout, () => child.kill());
    if (code !== 0) {
      this.mayExist = false;
      throw new Error(`Cannot create Python container: ${await errors}`);
    }
  }

  protected attachCommand(): string[] {
    return [this.docker, "start", "--attach", "--interactive", this.name as string];
  }

  protected async removeContainer(): Promise<void> {
    if (this.name === null || !this.mayExist) return;
    const child = Bun.spawn([this.docker, "rm", "--force", this.name], { stdout: "ignore", stderr: "pipe" });
    const errors = new Response(child.stderr).text();
    const code = await withTimeout(child.exited, 10, () => child.kill());
    const text = await errors;
    if (code !== 0 && !text.includes("No such container")) {
      throw new Error(`Could not remove Python container ${this.name}: ${text}`);
    }
    this.mayExist = false;
  }

  private async start(): Promise<void> {
    this.name = `tbln-python-${hexId()}`;
    this.diagnostics = "";
    await this.createContainer();
    const child = Bun.spawn(this.attachCommand(), { stdin: "pipe", stdout: "pipe", stderr: "pipe" });
    this.process = child;
    this.lines = new LineReader(child.stdout);
    this.stderrTask = (async () => {
      const decoder = new TextDecoder();
      for await (const chunk of child.stderr) this.diagnostics = (this.diagnostics + decoder.decode(chunk)).slice(-8192);
    })().catch(() => {});
    const ready = await this.lines.readLine();
    if (ready === null || JSON.stringify(JSON.parse(ready)) !== JSON.stringify({ ready: true })) {
      throw new Error("Python container did not become ready; check Docker and the local image");
    }
    const initial = `import json as _json\ndata = _json.loads(${JSON.stringify(this.dataJson)})\ndel _json\n${this.setup}`;
    const result = await this.exchange(initial);
    if (result.error) throw new Error(`Python initialization failed: ${result.error}`);
    this.lost = false;
  }

  private async exchange(code: string): Promise<PythonResult> {
    const child = this.process as Subprocess<"pipe", "pipe", "pipe">;
    const id = hexId();
    child.stdin.write(`${JSON.stringify({ id, code, max_output_bytes: this.maxOutputBytes })}\n`);
    await child.stdin.flush();
    const line = await (this.lines as LineReader).readLine();
    if (line === null) throw new Error("Python interpreter exited before returning a result");
    const response = JSON.parse(line);
    if (response.id !== id) throw new Error("Python response did not match the request");
    return pythonResult({
      stdout: response.stdout,
      stderr: response.stderr,
      result: response.result,
      error: response.error,
      truncated: response.truncated,
    });
  }

  private async stop(): Promise<void> {
    try {
      await this.removeContainer();
    } finally {
      if (this.process !== null) {
        if (this.process.exitCode === null) this.process.kill();
        await this.process.exited;
        this.process = null;
        this.lines = null;
      }
      if (this.stderrTask !== null) {
        await this.stderrTask;
        this.stderrTask = null;
      }
    }
    this.name = null;
  }

  private remember(code: string, result: PythonResult): void {
    this.executions.push({ code, result });
    if (this.executions.length > this.historyLimit) this.executions.shift();
  }

  /** Run a snippet. Aborting `signal` stops it and loses the interpreter's state. */
  execute(code: string, signal?: AbortSignal): Promise<PythonResult> {
    if (new TextEncoder().encode(code).length > this.maxCodeBytes) {
      return Promise.reject(new Error(`Code exceeds ${this.maxCodeBytes} bytes`));
    }
    return this.mutex.run("execute", async () => {
      if (this.closed) throw new Error("Python sandbox is closed");
      if (this.lost) return pythonResult({ error: "Python state was lost. Call reset_python before executing again.", stateLost: true });
      const started = performance.now();
      let result: PythonResult;
      const kill = () => this.process?.kill();
      signal?.addEventListener("abort", kill, { once: true });
      try {
        if (this.process === null) await withTimeout(this.start(), this.startupTimeout, kill);
        if (signal?.aborted) throw signal.reason;
        result = await withTimeout(this.exchange(code), this.timeout, kill);
        if (signal?.aborted) throw signal.reason;
      } catch (error) {
        const cancelled = signal?.aborted === true;
        this.lost = true;
        await this.stop();
        if (cancelled) {
          this.remember(code, pythonResult({ error: "Execution cancelled; Python state lost.", stateLost: true }));
          throw signal?.reason ?? error;
        }
        const reason = error instanceof Timeout ? "Execution deadline exceeded" : `${(error as Error).name}: ${(error as Error).message}`;
        result = pythonResult({ error: `${reason}. Python state lost. ${this.diagnostics}`.trimEnd(), stateLost: true });
      } finally {
        signal?.removeEventListener("abort", kill);
      }
      result = { ...result, duration: (performance.now() - started) / 1000 };
      this.remember(code, result);
      return result;
    });
  }

  history(limit = 10): Execution[] {
    if (limit < 0) throw new Error("History limit cannot be negative");
    return limit ? structuredClone(this.executions.slice(-limit)) : [];
  }

  reset(): Promise<void> {
    return this.mutex.run("execute", async () => {
      if (this.closed) throw new Error("Python sandbox is closed");
      await this.stop();
      this.lost = false;
      this.executions.length = 0;
    });
  }

  close(): Promise<void> {
    return this.mutex.run("execute", async () => {
      this.closed = true;
      await this.stop();
    });
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.close();
  }
}
