/**
 * A generic RLM-style Python loop over application-supplied documents.
 *
 * Requires Docker and a local Python image. The demonstration provider is
 * scripted; replace it with any streaming provider for a live experiment.
 */

import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { Agent, type Message, ReAct, type Run, type StreamChunk } from "../src/index.ts";
import { DockerPythonSandbox, type DockerPythonSandboxOptions, type Mount, PythonSessions, type PythonSandbox, pythonTools } from "../src/python/index.ts";

export const DOCUMENTS = [
  { id: "a", text: "Python supports asynchronous programming." },
  { id: "b", text: "Containers provide separate execution environments." },
  { id: "c", text: "Python tools can inspect a large document collection." },
];

class DemoAnalyst {
  async *createMessage(_system: string, messages: Message[], metadata?: { turn?: number }): AsyncGenerator<StreamChunk> {
    const turn = metadata?.turn ?? 1;
    const last = messages[messages.length - 1];
    let observation: { error: string | null; stdout: string } | null = null;
    if (last.role === "tool") {
      observation = JSON.parse(last.content);
      if (observation?.error) {
        yield { type: "text", text: `Python execution failed: ${observation.error}` };
        return;
      }
    }
    let code: string;
    if (turn === 1) code = "selected = [doc for doc in data if 'python' in doc['text'].lower()]\nlen(selected)";
    else if (turn === 2) {
      code =
        "import json\nfrom pathlib import Path\n" +
        "summary = {'matches': len(selected), 'document_ids': [doc['id'] for doc in selected]}\n" +
        "Path('summary.json').write_text(json.dumps(summary))\nprint(json.dumps(summary))";
    } else {
      yield { type: "text", text: `Analysis: ${observation?.stdout.trim()}. Saved summary.json.` };
      return;
    }
    yield { type: "tool_use", id: `python-${turn}`, name: "execute_python", input: { code } };
  }
}

export async function demo(
  options: { image?: string; workspace?: string; logDir?: string; sandbox?: (options: DockerPythonSandboxOptions) => PythonSandbox } = {},
): Promise<Run> {
  const mounts: Mount[] = [];
  if (options.workspace) {
    const workspace = resolve(options.workspace);
    mkdirSync(workspace, { recursive: true });
    mounts.push({ source: workspace, target: "/workspace", readOnly: false });
  }
  const make = options.sandbox ?? ((settings) => new DockerPythonSandbox(settings));
  const sessions = new PythonSessions(() => make({ image: options.image ?? "python:3.12-slim", data: DOCUMENTS, mounts }));
  try {
    const agent = new Agent({
      name: "python-analyst",
      provider: new DemoAnalyst(),
      loop: new ReAct(),
      tools: pythonTools(sessions, { description: "data is a list of documents with id and text fields." }),
      systemPrompt: "Inspect data using Python, retain useful variables, and report concise results.",
      limits: { toolTimeout: 90 },
      logDir: options.logDir,
    });
    return await agent.run("Find documents about Python and save their IDs in summary.json.");
  } finally {
    await sessions.close();
  }
}

if (import.meta.main) {
  const { values } = parseArgs({ options: { image: { type: "string" }, workspace: { type: "string" }, "log-dir": { type: "string" } } });
  const result = await demo({ image: values.image, workspace: values.workspace, logDir: values["log-dir"] });
  console.log(result.output ?? result.error);
  if (result.status !== "completed") process.exitCode = 1;
}
