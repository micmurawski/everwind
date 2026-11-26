import { expect, test } from "bun:test";
import {
  Agent,
  MemoryStore,
  MemoryStoreData,
  ReActWithWakeups,
  Run,
  Runtime,
  ScriptedProvider,
  Tools,
  limits,
  message,
  scheduleWakeupTool,
  wakeupRequest,
} from "../src/index.ts";
import { CountingMemory, CrashingMemory, World, build, call, crashes, makeTools } from "./helpers.ts";

function runtimeWith(...messages: Parameters<typeof message>[0][]): Runtime {
  return new Runtime(new Run({ messages: messages.map(message) }), new ScriptedProvider([]), new Tools(), limits());
}

test("the tool clamps the delay and needs a prompt unless stopping", async () => {
  const wakeup = scheduleWakeupTool();
  expect((await wakeup.invoke({ delaySeconds: 5, prompt: "again" })).result).toEqual({ scheduled: true, delaySeconds: 60, reason: "" });
  expect((await wakeup.invoke({ delaySeconds: 99999, prompt: "again" })).result).toMatchObject({ delaySeconds: 3600 });
  expect((await wakeup.invoke({ stop: true })).result).toEqual({ scheduled: false, stopped: true });
  expect((await wakeup.invoke({ delaySeconds: 60 })).error).toContain("required unless stop");
  expect(() => scheduleWakeupTool({ minDelay: 10, maxDelay: 5 })).toThrow("minDelay");
});

test("the request is read from the last model turn; the last valid call wins", () => {
  const assistant = (calls: ReturnType<typeof call>[]) => ({ role: "assistant" as const, toolCalls: calls });
  const result = (id: string, isError = false) => ({ role: "tool" as const, toolCallId: id, content: "ok", isError });
  const turn = [
    { role: "user" as const, content: "go" },
    assistant([
      call("a", "schedule_wakeup", { delaySeconds: 120, prompt: "first" }),
      call("b", "schedule_wakeup", { delaySeconds: 300, prompt: "second", reason: "slow CI" }),
    ]),
    result("a"),
    result("b"),
  ];
  expect(wakeupRequest(runtimeWith(...turn))).toEqual({ delaySeconds: 300, prompt: "second", reason: "slow CI", noop: false });
  // A failed call is ignored.
  expect(wakeupRequest(runtimeWith(...turn.slice(0, 3), result("b", true)))?.prompt).toBe("first");
  // Stop ends the wakeups, and an earlier turn's request does not count.
  expect(wakeupRequest(runtimeWith(turn[0], assistant([call("s", "schedule_wakeup", { stop: true })]), result("s")))).toBeNull();
  expect(wakeupRequest(runtimeWith(...turn, { role: "user", content: "next" }))).toBeNull();
});

test("the wakeup node sleeps, then gives the prompt; sleeping does not use the run's timeout", async () => {
  const agent = new Agent({
    provider: new ScriptedProvider([
      { toolCalls: [call("w", "schedule_wakeup", { delaySeconds: 0.3, prompt: "Check the build" })] },
      { text: "Build is green" },
    ]),
    tools: new Tools([scheduleWakeupTool({ minDelay: 0 })]),
    loop: new ReActWithWakeups({ minDelay: 0 }),
    limits: { timeout: 0.2 },
  });
  const run = await agent.run("Watch the build");
  expect(run.status).toBe("completed");
  expect(run.output).toBe("Build is green");
  expect(run.messages.map((item) => item.role)).toEqual(["system", "user", "assistant", "tool", "user", "assistant"]);
  expect(run.messages[4].content).toBe("Check the build");
  const kinds = run.events.map((event) => event.kind);
  expect(kinds).toContain("suspended");
  expect(run.events.find((event) => event.kind === "woke")?.data.by).toBe("timer");
  const nodes = run.events.filter((event) => event.kind === "step").map((event) => event.data.node);
  expect(nodes).toEqual(["gate", "call_llm", "tools", "ask", "wakeup", "wait", "gate", "call_llm", "settle", "finish"]);
});

test("cancelling a sleeping run stops it at once", async () => {
  const agent = new Agent({
    provider: new ScriptedProvider([{ toolCalls: [call("w", "schedule_wakeup", { delaySeconds: 3600, prompt: "later" })] }]),
    tools: new Tools([scheduleWakeupTool()]),
    loop: new ReActWithWakeups(),
  });
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 50);
  const started = performance.now();
  await expect(agent.run("Wait", { signal: controller.signal })).rejects.toThrow("cancelled");
  expect(performance.now() - started).toBeLessThan(1000);
});

test("a run recovered during its sleep keeps the original wake-up time", async () => {
  const reference = new CountingMemory();
  await build("react_with_wakeups", reference, new World()).run("Do the task", { sessionId: "s1" });
  let resumed = 0;
  for (let crashAt = 1; crashAt <= reference.written; crashAt++) {
    const data = new MemoryStoreData();
    const world = new World();
    await crashes(build("react_with_wakeups", new CrashingMemory(data, { crashAt, after: true }), world).run("Do the task", { sessionId: "s1" }), {
      mayFinish: true,
    });
    const store = new MemoryStore(data);
    const records = store.runs({ sessionId: "s1" });
    if (!records.length || records[0].status !== "running") continue;
    const run = await build("react_with_wakeups", store, world).recover("s1", { force: true });
    expect(run.status).toBe("completed");
    expect(world.unsafe).toEqual(["hello"]);
    const events = [...store.events("s1")].map((entry) => entry.event);
    const again = events.find((event) => event.kind === "command_retried" && event.data.command === "suspend");
    if (again !== undefined) {
      resumed += 1;
      const started = events.find((event) => event.kind === "suspended");
      expect(again?.data.fireAt).toBe(started?.data.fireAt as string);
    }
  }
  expect(resumed).toBeGreaterThan(0);
});

test("the shared tools still describe the wakeup scenario", () => {
  expect(makeTools(new World()).with(scheduleWakeupTool()).get("schedule_wakeup")?.effect).toBe("read");
});
