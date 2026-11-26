import { expect, test } from "bun:test";
import { mkdirSync, readFileSync, symlinkSync, utimesSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Agent, BackgroundShells, Mailbox, Run, Runtime, ScriptedProvider, Tools, Workspace, limits, shellJobTools, workspaceTools } from "../src/index.ts";
import { call, tmpDir } from "./helpers.ts";

function setup(options: ConstructorParameters<typeof Workspace>[0] = {}) {
  const root = tmpDir();
  const workspace = new Workspace({ root, ...options });
  const tools = workspaceTools(workspace);
  const invoke = (name: string, args: Record<string, unknown>, context: Record<string, unknown> = {}, sessionId = "s") => {
    const runtime = new Runtime(new Run({ sessionId, context }), new ScriptedProvider([]), new Tools(), limits());
    return tools.invoke({ id: "c", name, arguments: args as never, metadata: {} }, { runtime, values: context });
  };
  return { root, workspace, invoke };
}

test("read_file numbers lines and pages with offset and limit", async () => {
  const { root, invoke } = setup();
  writeFileSync(join(root, "a.txt"), "one\ntwo\nthree\n");
  expect((await invoke("read_file", { filePath: "a.txt" })).result).toBe("     1\tone\n     2\ttwo\n     3\tthree");
  const page = (await invoke("read_file", { filePath: join(root, "a.txt"), offset: 2, limit: 1 })).result as string;
  expect(page).toBe("     2\ttwo\n... 1 more lines; continue with offset=3");
  expect((await invoke("read_file", { filePath: "a.txt", offset: 9 })).error).toContain("past the end");
  expect((await invoke("read_file", { filePath: "missing.txt" })).error).toContain("does not exist");
  expect((await invoke("read_file", { filePath: "." })).error).toContain("is a directory");
  writeFileSync(join(root, "bin"), Buffer.from([1, 0, 2]));
  expect((await invoke("read_file", { filePath: "bin" })).error).toContain("binary");
  writeFileSync(join(root, "empty"), "");
  expect((await invoke("read_file", { filePath: "empty" })).result).toContain("is empty");
});

test("paths outside the workspace are refused, including through symlinks", async () => {
  const { root, invoke } = setup();
  const outside = tmpDir();
  writeFileSync(join(outside, "secret"), "x");
  symlinkSync(outside, join(root, "link"));
  expect((await invoke("read_file", { filePath: join(outside, "secret") })).error).toContain("outside the workspace");
  expect((await invoke("read_file", { filePath: "../x" })).error).toContain("outside the workspace");
  expect((await invoke("write_file", { filePath: "link/new", content: "x" })).error).toContain("outside the workspace");
  const open = setup({ confine: false });
  expect((await open.invoke("read_file", { filePath: join(outside, "secret") })).result).toBe("     1\tx");
});

test("write_file creates files, and overwrites only what was read and not changed since", async () => {
  const { root, invoke } = setup();
  expect((await invoke("write_file", { filePath: "dir/new.txt", content: "hi" })).result).toContain("Created");
  expect(readFileSync(join(root, "dir/new.txt"), "utf-8")).toBe("hi");
  // Writing it counts as having read it.
  expect((await invoke("write_file", { filePath: "dir/new.txt", content: "hello" })).result).toContain("Updated");
  writeFileSync(join(root, "old.txt"), "old");
  expect((await invoke("write_file", { filePath: "old.txt", content: "x" })).error).toContain("Read");
  // Another session's read does not count.
  await invoke("read_file", { filePath: "old.txt" }, {}, "other");
  expect((await invoke("write_file", { filePath: "old.txt", content: "x" })).error).toContain("Read");
  await invoke("read_file", { filePath: "old.txt" });
  utimesSync(join(root, "old.txt"), new Date(), new Date(Date.now() + 5000));
  expect((await invoke("write_file", { filePath: "old.txt", content: "x" })).error).toContain("changed since");
  await invoke("read_file", { filePath: "old.txt" });
  expect((await invoke("write_file", { filePath: "old.txt", content: "x" })).error).toBeNull();
});

test("edit_file replaces a unique string, or every occurrence with replaceAll", async () => {
  const { root, invoke } = setup();
  const path = join(root, "code.ts");
  writeFileSync(path, "let a = 1;\nlet b = 1;\nconst price = '$1';\n");
  expect((await invoke("edit_file", { filePath: "code.ts", oldString: "a", newString: "z" })).error).toContain("Read");
  await invoke("read_file", { filePath: "code.ts" });
  expect((await invoke("edit_file", { filePath: "code.ts", oldString: "= 1", newString: "= 2" })).error).toContain("occurs 2 times");
  expect((await invoke("edit_file", { filePath: "code.ts", oldString: "nope", newString: "x" })).error).toContain("not found");
  expect((await invoke("edit_file", { filePath: "code.ts", oldString: "a", newString: "a" })).error).toContain("the same");
  expect((await invoke("edit_file", { filePath: "code.ts", oldString: "let a = 1", newString: "let a = $&" })).result).toContain("1 replacement");
  expect((await invoke("edit_file", { filePath: "code.ts", oldString: "let", newString: "var", replaceAll: true })).result).toContain(
    "2 replacements",
  );
  expect(readFileSync(path, "utf-8")).toBe("var a = $&;\nvar b = 1;\nconst price = '$1';\n");
});

test("bash returns output and exit code, and keeps the working directory per session", async () => {
  const { root, workspace, invoke } = setup();
  mkdirSync(join(root, "sub"));
  const ok = await invoke("bash", { command: "echo out; echo err >&2" });
  expect(ok.result).toBe("out\nerr");
  const failed = await invoke("bash", { command: "echo nope; exit 3" });
  expect(failed.error).toBe("nope\n[Exit code 3]");
  await invoke("bash", { command: "cd sub" });
  expect((await invoke("bash", { command: "pwd -P" })).result).toBe(join(workspace.root, "sub"));
  expect((await invoke("bash", { command: "basename $(pwd)" }, {}, "other")).result).not.toBe("sub");
  // Relative file paths follow the shell's directory.
  await invoke("write_file", { filePath: "here.txt", content: "x" });
  expect(readFileSync(join(root, "sub/here.txt"), "utf-8")).toBe("x");
  // Leaving the workspace resets the directory.
  const left = await invoke("bash", { command: "cd /" });
  expect(left.result).toContain("outside the workspace; reset to");
  expect((await invoke("bash", { command: "pwd -P" })).result).toBe(workspace.root);
});

test("bash kills a command that times out, with its children", async () => {
  const { invoke } = setup();
  const started = performance.now();
  const result = await invoke("bash", { command: "sleep 5 & sleep 5; echo never", timeout: 200 });
  expect(result.error).toContain("Timed out after 200 ms");
  expect(performance.now() - started).toBeLessThan(3000);
  expect((await invoke("bash", { command: "true", timeout: 700000 })).error).toContain("Invalid arguments");
});

test("bash truncates long output in the middle", async () => {
  const { invoke } = setup({ maxOutputChars: 100 });
  const text = (await invoke("bash", { command: "seq 1 1000" })).result as string;
  expect(text.startsWith("1\n2\n")).toBe(true);
  expect(text.endsWith("999\n1000")).toBe(true);
  expect(text).toContain("characters truncated");
});

test("background commands report to the mailbox and can be listed and killed", async () => {
  const { workspace, invoke } = setup();
  const shells = new BackgroundShells(new Mailbox());
  // A missing context value fails the run, not just the call.
  await expect(invoke("bash", { command: "echo x", runInBackground: true })).rejects.toThrow("shells");
  const handle = (await invoke("bash", { command: "echo done", runInBackground: true, description: "say done" }, { shells })).result as any;
  expect(handle.status).toBe("running");
  const finished = await shells.mailbox.get();
  expect(finished).toMatchObject({ kind: "ShellFinished", shellId: handle.shellId, status: "completed", exitCode: 0, output: "done" });

  const jobs = shellJobTools();
  const slow = (await invoke("bash", { command: "sleep 30", runInBackground: true }, { shells })).result as any;
  const runtime = new Runtime(new Run({ sessionId: "s", context: { shells } }), new ScriptedProvider([]), new Tools(), limits());
  const context = { runtime, values: { shells } };
  const listed = (await jobs.invoke({ id: "l", name: "list_shells", arguments: {}, metadata: {} }, context)).result as any[];
  expect(listed.map((item) => item.shellId)).toEqual([handle.shellId, slow.shellId]);
  await jobs.invoke({ id: "k", name: "kill_shell", arguments: { shellId: slow.shellId }, metadata: {} }, context);
  expect(await shells.mailbox.get()).toMatchObject({ shellId: slow.shellId, status: "killed" });
  await shells.close();
  expect(workspace.cwd("s")).toBe(workspace.root);
});

test("an agent can read, edit and check a file", async () => {
  const root = tmpDir();
  writeFileSync(join(root, "greet.txt"), "Hello, world\n");
  const agent = new Agent({
    provider: new ScriptedProvider([
      { toolCalls: [call("r", "read_file", { filePath: "greet.txt" })] },
      { toolCalls: [call("e", "edit_file", { filePath: "greet.txt", oldString: "world", newString: "everwind" })] },
      { toolCalls: [call("b", "bash", { command: "cat greet.txt" })] },
      { text: "Done" },
    ]),
    tools: workspaceTools(new Workspace({ root })),
  });
  const run = await agent.run("Greet everwind");
  expect(run.status).toBe("completed");
  expect(run.messages.filter((item) => item.role === "tool").map((item) => item.content)[2]).toBe("Hello, everwind");
});
