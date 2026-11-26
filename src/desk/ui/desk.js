// Shared helpers for the desk pages: API calls, status labels, small DOM builders. No framework, no build step.

export const STATUS = {
  running: "● running",
  sleeping: "☾ sleeping",
  held: "❚❚ held",
  frozen: "■ frozen",
  ask: "? needs you",
  stale: "! stale",
  completed: "✓ done",
  failed: "! failed",
  timeout: "! timeout",
  limit: "! limit",
  cancelled: "✕ cancelled",
  ready: "◌ ready",
  done: "✓ done",
  active: "● active",
  paused: "❚❚ paused",
};

/** Status groups for the fleet filters. */
export const GROUPS = {
  running: ["running"],
  waiting: ["sleeping", "held", "frozen"],
  ask: ["ask"],
  failed: ["failed", "timeout", "limit", "stale"],
  done: ["completed", "cancelled"],
};

export async function api(path, options = {}) {
  const init = { ...options, headers: { ...(options.headers ?? {}) } };
  if (options.body !== undefined) {
    init.method = init.method ?? "POST";
    init.body = JSON.stringify(options.body);
    init.headers["content-type"] = "application/json";
    init.headers["x-everwind"] = "1";
  }
  const response = await fetch(`/api/${path}`, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error ?? `${response.status} ${response.statusText}`);
  return payload;
}

/** Create an element: el("td", { class: "num" }, "text", child, …). */
export function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value === undefined || value === null || value === false) continue;
    if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export const chip = (status) => el("span", { class: `chip ${status}` }, STATUS[status] ?? status);

export function ago(iso) {
  if (!iso) return "—";
  const seconds = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

export const kilo = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(n));

let toastTimer = null;
export function toast(text, bad = false) {
  document.querySelector(".toast")?.remove();
  const node = el("div", { class: `toast${bad ? " bad" : ""}`, role: "status" }, text);
  document.body.append(node);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.remove(), 4000);
}

/** Run an action, reporting its failure as a toast. */
export async function act(fn, done) {
  try {
    const result = await fn();
    if (done) toast(done);
    return result;
  } catch (error) {
    toast(error.message, true);
    return null;
  }
}

/** One line of the event log: [group class, summary]. */
export function describe(event) {
  const d = event.data ?? {};
  const trim = (text, n = 140) => (text && text.length > n ? `${text.slice(0, n)}…` : text ?? "");
  switch (event.kind) {
    case "run_start": return ["k-sys", `${d.agent} · ${d.model ?? d.provider ?? ""} · ${d.loop}`];
    case "step": return ["k-sys", `→ ${d.node}`];
    case "transition": return ["k-sys", `${d.node} —${d.action}→ ${d.target}`];
    case "model_request": return ["k-model", `turn ${d.request?.turn} · ${d.request?.messages?.length} messages · ${d.request?.tools?.length} tools${d.conversation === false ? " · summary" : ""}`];
    case "model_response": {
      const calls = (d.response?.toolCalls ?? []).map((call) => `${call.name}(…)`).join(", ");
      return ["k-model", `turn ${d.turn}${calls ? ` → ${calls}` : ""} ${trim(d.response?.text)}`];
    }
    case "tool_start": return ["k-tool", `${d.call?.name} ${trim(JSON.stringify(d.call?.arguments ?? {}), 100)}`];
    case "tool_result": return [d.message?.isError ? "k-bad" : "k-tool", `${d.message?.isError ? "! " : "✓ "}${trim(d.message?.content)}`];
    case "instruction": return ["k-op", `${d.pinned ? "[pinned] " : ""}${trim(d.content)}`];
    case "received": return ["k-op", `inbox ${d.ids?.join(", ")}`];
    case "question": return ["k-op", `? ${trim(d.question)}`];
    case "answered": return ["k-op", "answered"];
    case "held": case "resumed": case "frozen": return ["k-op", `${event.kind} before ${d.node ?? "(sleeping)"}${d.note ? ` · ${d.note}` : ""}`];
    case "suspended": return ["k-ctx", `waiting${d.seconds != null ? ` ${d.seconds}s` : ""}${d.wakeOn?.length ? ` or ${d.wakeOn.join(", ")}` : ""}${d.reason ? ` · ${d.reason}` : ""}`];
    case "woke": return ["k-ctx", `woke by ${d.by}${d.kind ? ` (${d.kind})` : ""}`];
    case "checkpoint": return ["k-ctx", `checkpoint before ${d.node} · ${d.messages} messages`];
    case "context_compaction": return ["k-ctx", `compaction · ${d.strategy} · ${d.removedMessages?.length} → ${d.replacementMessages?.length} messages`];
    case "run_end": return [d.status === "completed" ? "k-sys" : "k-bad", `${d.status}${d.error ? ` · ${d.error}` : ""}${d.output ? ` · ${trim(d.output)}` : ""}`];
    case "run_recovered": return ["k-op", `recovered (epoch ${d.epoch}${d.fromCheckpoint ? `, from checkpoint #${d.fromCheckpoint}` : ""})`];
    case "command_retried": return ["k-ctx", `retried ${d.command} #${d.seq}`];
    default: return ["k-sys", trim(JSON.stringify(d), 140)];
  }
}

/** Paint the theme toggle (thearc's theme.js does the switching). */
export function header(stats) {
  const box = document.querySelector(".stats");
  if (!box) return;
  box.replaceChildren(...stats.map(([cls, label, value]) => el("div", { class: `stat ${cls}` }, el("div", { class: "eyebrow" }, label), el("b", {}, value))));
}
