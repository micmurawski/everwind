import { describe, expect, test } from "bun:test";
import { Agent, Run, Runtime, ScriptedProvider, Tools, limits } from "../src/index.ts";
import { DockerPythonSandbox, type DockerPythonSandboxOptions, PythonSessions, WORKER_PATH, pythonTools } from "../src/python/index.ts";
import { call, tmpDir } from "./helpers.ts";

const PYTHON = Bun.which("python3");

/** Test-only transport: run trusted test snippets in a local process. Not a public fallback for Docker. */
class LocalWorker extends DockerPythonSandbox {
  protected override async createContainer(): Promise<void> {}
  protected override attachCommand(): string[] {
    return [PYTHON as string, "-I", "-u", WORKER_PATH];
  }
  protected override async removeContainer(): Promise<void> {}
}

const local = (options: DockerPythonSandboxOptions = {}) => new LocalWorker(options);

describe.skipIf(PYTHON === null)("Python sandbox", () => {
  test("state, expressions, stdout, stderr, native output and exceptions", async () => {
    const sandbox = local({ data: { values: [1, 2, 3] }, setup: "import math" });
    try {
      const first = await sandbox.execute("total = sum(data['values'])\nprint('hello')\ntotal");
      expect([first.stdout, first.result, first.error]).toEqual(["hello\n", "6", null]);
      const native = await sandbox.execute(
        "import os, subprocess, sys\nos.write(1, b'native\\n')\n" +
          "subprocess.run([sys.executable, '-c', \"print('child')\"], check=True)\n" +
          "print('warning', file=sys.stderr)\nmath.sqrt(total * 6)",
      );
      expect([native.stdout, native.stderr, native.result]).toEqual(["native\nchild\n", "warning\n", "6.0"]);
      const broken = await sandbox.execute("total += 1\nraise ValueError('oops')");
      expect(broken.error).toContain("ValueError: oops");
      expect(broken.stateLost).toBe(false);
      expect((await sandbox.execute("total")).result).toBe("7");
      expect((await sandbox.execute("raise SystemExit(4)")).error).toContain("SystemExit");
      expect((await sandbox.execute("invalid ! syntax")).error).toContain("SyntaxError");
      expect((await sandbox.execute("input('prompt')")).error).toContain("EOFError");
      expect((await sandbox.execute("total")).result).toBe("7");
    } finally {
      await sandbox.close();
    }
  });

  test("output and history are bounded without breaking the protocol", async () => {
    const sandbox = local({ maxOutputBytes: 100, historyLimit: 2 });
    try {
      const result = await sandbox.execute("import os\nos.write(1, b'x' * 1000000)\n'v' * 10000");
      expect([result.stdout.length, result.result?.length, result.truncated]).toEqual([100, 100, true]);
      expect((await sandbox.execute("21 * 2")).result).toBe("42");
      await sandbox.execute("True");
      expect(sandbox.history(100).map((entry) => entry.code)).toEqual(["21 * 2", "True"]);
      expect(sandbox.history(0)).toEqual([]);
      expect(() => sandbox.history(-1)).toThrow(/negative/);
    } finally {
      await sandbox.close();
    }
  });

  for (const stop of ["timeout", "cancel", "exit"]) {
    test(`stopping execution (${stop}) loses state and requires an explicit reset`, async () => {
      const sandbox = local({ timeout: 0.2, setup: "seed = 10" });
      try {
        expect((await sandbox.execute("seed")).result).toBe("10");
        const process = sandbox.process;
        if (stop === "cancel") {
          const controller = new AbortController();
          const pending = sandbox.execute("while True: pass", controller.signal);
          await Bun.sleep(50);
          controller.abort(new Error("cancelled"));
          await expect(pending).rejects.toThrow(/cancelled/);
        } else {
          const result = await sandbox.execute(stop === "exit" ? "import os; os._exit(2)" : "while True: pass");
          expect(result.stateLost).toBe(true);
          expect(result.error).toBeTruthy();
        }
        expect(process?.exitCode !== null || process?.signalCode !== null).toBe(true);
        const blocked = await sandbox.execute("seed");
        expect(blocked.stateLost).toBe(true);
        expect(blocked.error).toContain("reset_python");
        await sandbox.reset();
        expect(sandbox.history()).toEqual([]);
        expect((await sandbox.execute("seed")).result).toBe("10");
        expect(sandbox.process?.pid).not.toBe(process?.pid);
      } finally {
        await sandbox.close();
      }
    });
  }

  test("concurrent snippets are serialized, and sessions are isolated", async () => {
    const sessions = new PythonSessions(() => local());
    const [first, second] = [sessions.get("first"), sessions.get("second")];
    expect(sessions.get("first")).toBe(first);
    await first.execute("value = 0");
    const results = await Promise.all(
      Array.from({ length: 5 }, () => first.execute("import time\nold = value\ntime.sleep(.01)\nvalue = old + 1\nvalue")),
    );
    expect(results.map((result) => result.result)).toEqual(["1", "2", "3", "4", "5"]);
    expect((await second.execute("value")).error).toContain("NameError");
    const processes = [(first as LocalWorker).process, (second as LocalWorker).process];
    await sessions.close();
    for (const process of processes) await process?.exited;
    expect(() => sessions.get("third")).toThrow(/closed/);
  });

  test("agent tools keep state across continuations and hide configuration", async () => {
    const sessions = new PythonSessions(() => local({ data: [10, 20] }));
    try {
      const tools = pythonTools(sessions, { description: "data contains the input numbers." });
      const specs = Object.fromEntries(tools.specs.map((spec) => [spec.name, spec]));
      expect(Object.keys(specs.execute_python.parameters.properties as object)).toEqual(["code"]);
      expect(specs.execute_python.description).toContain("input numbers");
      const code = "count = globals().get('count', 0) + 1\n{'total': sum(data), 'visits': count}";
      const agent = new Agent({ provider: new ScriptedProvider([{ toolCalls: [call("python", "execute_python", { code })] }, { text: "Calculated" }]), tools });
      const first = await agent.run("Calculate");
      const second = await agent.run("Again", { sessionId: first.sessionId });
      expect([first.status, second.status]).toEqual(["completed", "completed"]);
      const results = second.messages.filter((item) => item.role === "tool").map((item) => JSON.parse(item.content));
      expect(results.map((item) => item.result)).toEqual(["{'total': 30, 'visits': 1}", "{'total': 30, 'visits': 2}"]);
    } finally {
      await sessions.close();
    }
  });

  test("a setup error and missing Docker are reported without a host fallback", async () => {
    const broken = local({ setup: "raise RuntimeError('bad setup')" });
    const result = await broken.execute("print('never')");
    expect(result.error).toContain("bad setup");
    expect(result.stateLost).toBe(true);
    expect(broken.process).toBeNull();
    await broken.close();
    const missing = new DockerPythonSandbox({ docker: "/does-not-exist/tbln-docker" });
    const failed = await missing.execute("print('never')");
    expect(failed.error).toMatch(/ENOENT|No such file|not found/i);
    expect(failed.stateLost).toBe(true);
    await missing.close();
  });

  test("cancelling during creation waits for creation, then removes the container", async () => {
    const actions: string[] = [];
    let markEntered!: () => void;
    const entered = new Promise<void>((resolve) => (markEntered = resolve));
    let release!: () => void;
    const released = new Promise<void>((resolve) => (release = resolve));
    class Delayed extends LocalWorker {
      protected override async createContainer(): Promise<void> {
        markEntered();
        await released;
        actions.push("created");
      }
      protected override async removeContainer(): Promise<void> {
        actions.push("removed");
      }
    }
    const sandbox = new Delayed();
    const controller = new AbortController();
    let done = false;
    const pending = sandbox.execute("1", controller.signal).finally(() => (done = true));
    await entered;
    controller.abort(new Error("cancelled"));
    await Bun.sleep(5);
    expect(done).toBe(false);
    release();
    await expect(pending).rejects.toThrow(/cancelled/);
    expect(actions).toEqual(["created", "removed"]);
    expect(sandbox.process).toBeNull();
  });

  test("history, reset, and the outer tool deadline", async () => {
    const sessions = new PythonSessions(() => local({ timeout: 10 }));
    try {
      const tools = pythonTools(sessions);
      const run = new Run();
      const context = { runtime: new Runtime(run, new ScriptedProvider([]), new Tools(), limits()) };
      await sessions.get(run.sessionId).execute("value = 42");
      let process = (sessions.get(run.sessionId) as LocalWorker).process;
      const history = await tools.invoke({ id: "history", name: "python_history", arguments: {}, metadata: {} }, context);
      expect((history.result as { code: string }[])[0].code).toBe("value = 42");
      const reset = await tools.invoke({ id: "reset", name: "reset_python", arguments: {}, metadata: {} }, context);
      expect((reset.result as { status: string }).status).toBe("reset");
      await process?.exited;
      expect(sessions.get(run.sessionId).history()).toEqual([]);
      // Runtime cancellation must actually stop execution, even with a longer sandbox deadline.
      const agent = new Agent({
        provider: new ScriptedProvider([{ toolCalls: [call("loop", "execute_python", { code: "while True: pass" })] }]),
        tools,
        limits: { toolTimeout: 0.1 },
      });
      await sessions.get("deadline").execute("1"); // Warm up, so this checks execution rather than startup.
      process = (sessions.get("deadline") as LocalWorker).process;
      const result = await agent.run("Run", { sessionId: "deadline" });
      expect(result.status).toBe("timeout");
      await process?.exited;
      expect((await sessions.get("deadline").execute("1")).stateLost).toBe(true);
    } finally {
      await sessions.close();
    }
  });
});

test("the Docker command uses explicit mounts and publishes no ports", () => {
  const directory = tmpDir();
  const sandbox = new DockerPythonSandbox({
    mounts: [{ source: directory, target: "/inputs" }, { source: directory, target: "/workspace", readOnly: false }],
    env: { EXPLICIT: "configured" },
  });
  (sandbox as unknown as { name: string }).name = "test-owned-container";
  const command = sandbox.createCommand();
  const real = require("node:fs").realpathSync(directory);
  expect(command.slice(0, 2)).toEqual(["docker", "create"]);
  for (const flag of ["--pull=never", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "EXPLICIT=configured"]) {
    expect(command).toContain(flag);
  }
  expect(command[command.indexOf("--network") + 1]).toBe("none");
  expect(command.some((arg) => ["-p", "--publish", "--privileged"].includes(arg))).toBe(false);
  expect(command).toContain(`type=bind,source=${real},target=/inputs,readonly`);
  expect(command).toContain(`type=bind,source=${real},target=/workspace`);
  expect(command.some((arg) => arg.includes("/workspace:rw"))).toBe(false);
  for (const target of ["relative", "/", "/opt/tbln_worker.py", "/some/../where"]) {
    const bad = new DockerPythonSandbox({ mounts: [{ source: directory, target }] });
    expect(() => bad.createCommand()).toThrow(/mount/);
  }
});

test.skipIf(process.env.TBLN_TEST_DOCKER !== "1")("real Docker: state, mounts, timeout and cleanup", async () => {
  const directory = tmpDir();
  require("node:fs").writeFileSync(`${directory}/input.txt`, "hello");
  const sandbox = new DockerPythonSandbox({ mounts: [{ source: directory, target: "/inputs" }], timeout: 0.5 });
  try {
    const first = await sandbox.execute("from pathlib import Path\nx = 40\nPath('/inputs/input.txt').read_text()");
    expect(first.result).toBe("'hello'");
    expect((await sandbox.execute("x + 2")).result).toBe("42");
    const denied = await sandbox.execute("Path('/inputs/input.txt').write_text('changed')");
    expect(denied.error).toBeTruthy();
    expect((await sandbox.execute("while True: pass")).stateLost).toBe(true);
  } finally {
    await sandbox.close();
  }
});
