// The flow graph, ported from thearc's flow visualizer (honed/flow: viz.py's layout and SVG, viz.js's interaction).
// Layout: a DFS marks back edges (loops); the forward edges give each node a column at its longest-path depth, and
// nodes sharing a column stack in discovery order. Cards: shadow, status bar, kind / name / sub line. Forward edges are
// curves from a card's right to the next card's left; back edges loop above the graph (between top-row nodes) or below
// it, dashed. Pan, wheel zoom, draggable cards (positions kept per graph in localStorage) and a metadata tooltip.

const NW = 176, NH = 64, SH = 4, CW = 248, RH = 104, PAD = 40, CHARS = 21;
const END = "__end__";
const SVG = "http://www.w3.org/2000/svg";

function svgEl(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Nodes in discovery order from the start, END last if any edge reaches it; edges as {from, action, to}. */
function graphOf(flow) {
  const edges = flow.edges.map(([from, action, to]) => ({ from, action, to }));
  const ids = [flow.start];
  for (let i = 0; i < ids.length; i++) {
    for (const edge of edges) if (edge.from === ids[i] && edge.to !== END && !ids.includes(edge.to)) ids.push(edge.to);
  }
  for (const name of Object.keys(flow.nodes ?? {})) if (!ids.includes(name)) ids.push(name);
  if (edges.some((edge) => edge.to === END)) ids.push(END);
  return { ids, edges };
}

function layout({ ids, edges }) {
  const out = new Map(ids.map((id) => [id, []]));
  for (const edge of edges) out.get(edge.from)?.push(edge.to);
  const color = new Map(ids.map((id) => [id, 0]));
  const back = new Set();
  const dfs = (u) => {
    color.set(u, 1);
    for (const v of out.get(u) ?? []) {
      if (!color.has(v)) continue;
      if (color.get(v) === 1) back.add(`${u}\u0000${v}`);
      else if (color.get(v) === 0) dfs(v);
    }
    color.set(u, 2);
  };
  for (const id of ids) if (color.get(id) === 0) dfs(id);
  const isBack = (u, v) => back.has(`${u}\u0000${v}`);
  const targets = new Set(edges.filter((e) => !isBack(e.from, e.to)).map((e) => e.to));
  const roots = ids.filter((id) => !targets.has(id));
  const col = new Map((roots.length ? roots : [ids[0]]).map((id) => [id, 0]));
  const queue = [...col.keys()];
  const seen = new Set(queue);
  while (queue.length) {
    const u = queue.shift();
    for (const v of out.get(u) ?? []) {
      if (isBack(u, v)) continue;
      col.set(v, Math.max(col.get(v) ?? 0, col.get(u) + 1));
      if (!seen.has(v)) { seen.add(v); queue.push(v); }
    }
  }
  const maxCol = Math.max(0, ...col.values());
  for (const id of ids) if (!col.has(id)) col.set(id, maxCol + 1);
  // Within a column, the node with the longest forward path ahead goes on top (ties: discovery order), so a loop's
  // main path stays on the first row and exits such as finish sit below it.
  const ahead = new Map();
  const reach = (u) => {
    if (ahead.has(u)) return ahead.get(u);
    ahead.set(u, 0);
    let best = 0;
    for (const v of out.get(u) ?? []) if (!isBack(u, v) && col.has(v)) best = Math.max(best, 1 + reach(v));
    ahead.set(u, best);
    return best;
  };
  const rows = new Map();
  const byCol = new Map();
  for (const id of ids) byCol.set(col.get(id), [...(byCol.get(col.get(id)) ?? []), id]);
  for (const members of byCol.values()) {
    members.sort((a, b) => reach(b) - reach(a) || ids.indexOf(a) - ids.indexOf(b)).forEach((id, row) => rows.set(id, row));
  }
  return { pos: new Map(ids.map((id) => [id, [col.get(id), rows.get(id)]])), isBack };
}

/** The layout alone, for tests: each node's [column, row], and the back edges as "from→to". */
export function planFlow(flow) {
  const graph = graphOf(flow);
  const { pos, isBack } = layout(graph);
  return {
    pos: Object.fromEntries(pos),
    back: graph.edges.filter((e) => isBack(e.from, e.to)).map((e) => `${e.from}→${e.to}`),
  };
}

function hash(text) {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (Math.imul(31, h) + text.charCodeAt(i)) | 0;
  return (h >>> 0).toString(16);
}

/**
 * Draw `flow` ({start, edges: [from, action, to][], nodes: {name: {checkpoint, metadata}}}) into `box`.
 * Returns update(state): state.nodes[name] = {status, sub}, state.edges["from\0action\0to"] = count.
 */
export function mountFlow(box, flow) {
  const graph = graphOf(flow);
  const { pos, isBack } = layout(graph);
  const maxCol = Math.max(...[...pos.values()].map(([c]) => c));
  const maxRow = Math.max(...[...pos.values()].map(([, r]) => r));
  const backEdges = graph.edges.filter((e) => isBack(e.from, e.to));
  const over = new Set(backEdges.filter((e) => pos.get(e.from)[1] === 0 && pos.get(e.to)[1] === 0));
  const nOver = over.size, nUnder = backEdges.length - over.size;
  const top = PAD + (nOver ? nOver * 40 + 20 : 0);
  const W = PAD + maxCol * CW + NW + PAD;
  const H = top + maxRow * RH + NH + PAD + (nUnder ? nUnder * 40 + 30 : 0);
  const key = hash(JSON.stringify(flow.edges));
  const positionsKey = `everwind-desk-flow-positions:${key}`;
  const xy = (id) => { const [c, r] = pos.get(id); return [PAD + c * CW, top + r * RH]; };

  const svg = svgEl("svg", { class: "flow-svg", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet", role: "img", "aria-label": `Flow graph starting at ${flow.start}` });
  const defs = svgEl("defs");
  const dots = svgEl("pattern", { id: `dots-${key}`, width: 16, height: 16, patternUnits: "userSpaceOnUse" });
  dots.append(svgEl("rect", { class: "grid-dot", x: 0, y: 0, width: 1.5, height: 1.5 }));
  const marker = (id, cls) => {
    const m = svgEl("marker", { id: `${id}-${key}`, markerWidth: 10, markerHeight: 7, refX: 9, refY: 3.5, orient: "auto" });
    m.append(svgEl("polygon", { class: cls, points: "0 0,10 3.5,0 7" }));
    return m;
  };
  defs.append(dots, marker("arr", "arrow"), marker("arr-bk", "arrow-bk"));
  const background = svgEl("rect", { x: -5000, y: -5000, width: 10000, height: 10000, fill: `url(#dots-${key})` });
  const viewport = svgEl("g");
  svg.append(defs, background, viewport);

  // Edges first, so cards stay above connectors.
  const edgeParts = [];
  const labels = new Map();
  let drawnOver = 0, drawnUnder = 0;
  for (const edge of graph.edges) {
    const back = isBack(edge.from, edge.to);
    const anchors = back ? (over.has(edge) ? ["t", "t"] : ["b", "b"]) : ["r", "l"];
    const drop = !back ? 0 : over.has(edge) ? top - 30 - drawnOver++ * 40 : top + maxRow * RH + NH + 30 + drawnUnder++ * 40;
    const path = svgEl("path", { class: back ? "edge back" : "edge", "marker-end": `url(#${back ? "arr-bk" : "arr"}-${key})` });
    const label = svgEl("text", { class: back ? "edge-label back" : "edge-label", "text-anchor": "middle" });
    viewport.append(path, label);
    const part = { edge, path, label, back, drop, anchors, action: edge.action === "default" ? "" : edge.action };
    edgeParts.push(part);
    labels.set(`${edge.from}\u0000${edge.action}\u0000${edge.to}`, part);
  }

  const cards = new Map();
  for (const id of graph.ids) {
    const [x, y] = xy(id);
    const isEnd = id === END;
    const info = flow.nodes?.[id] ?? {};
    const kind = isEnd ? "end" : id === flow.start ? "start" : info.checkpoint ? "may wait" : "node";
    const g = svgEl("g", { class: "node pending", transform: `translate(${x},${y})` });
    g.dataset.id = id;
    g.dataset.x = x;
    g.dataset.y = y;
    const title = svgEl("title", {}, isEnd ? "END" : id);
    const sub = svgEl("text", { class: "n-sub", x: 14, y: 53 }, "");
    g.append(title,
      svgEl("rect", { class: "n-shadow", x: SH, y: SH, width: NW, height: NH }),
      svgEl("rect", { class: "n-card", x: 0, y: 0, width: NW, height: NH }),
      svgEl("rect", { class: "n-bar", x: 0, y: 0, width: 4, height: NH }),
      svgEl("text", { class: "n-kind", x: 14, y: 18 }, kind),
      svgEl("text", { class: "n-label", x: 14, y: 36 }, (isEnd ? "END" : id).length > CHARS ? `${id.slice(0, CHARS - 1)}…` : isEnd ? "END" : id),
      sub);
    viewport.append(g);
    const metadata = info.metadata && Object.keys(info.metadata).length ? JSON.stringify(info.metadata, null, 2) : null;
    cards.set(id, { g, sub, title, metadata });
  }

  const anchor = (id, a) => {
    const g = cards.get(id).g;
    const x = Number(g.dataset.x), y = Number(g.dataset.y);
    if (a === "r") return [x + NW, y + NH / 2];
    if (a === "l") return [x, y + NH / 2];
    if (a === "t") return [x + NW / 2, y];
    return [x + NW / 2, y + NH];
  };
  const drawEdge = (part) => {
    const [sx, sy] = anchor(part.edge.from, part.anchors[0]);
    const [tx, ty] = anchor(part.edge.to, part.anchors[1]);
    if (part.back) {
      part.path.setAttribute("d", `M ${sx},${sy} C ${sx},${part.drop} ${tx},${part.drop} ${tx},${ty}`);
      part.label.setAttribute("x", (sx + tx) / 2);
      part.label.setAttribute("y", 0.125 * (sy + ty) + 0.75 * part.drop - 4);
    } else {
      const off = Math.max(30, Math.abs(tx - sx) * 0.5);
      part.path.setAttribute("d", `M ${sx},${sy} C ${sx + off},${sy} ${tx - off},${ty} ${tx},${ty}`);
      part.label.setAttribute("x", (sx + tx) / 2);
      part.label.setAttribute("y", (sy + ty) / 2 - 6);
    }
  };
  const place = (id, x, y) => {
    const g = cards.get(id).g;
    g.dataset.x = x;
    g.dataset.y = y;
    g.setAttribute("transform", `translate(${x},${y})`);
  };
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(positionsKey)) ?? {}; } catch {}
  for (const [id, p] of Object.entries(saved)) if (cards.has(id)) place(id, p.x, p.y);
  edgeParts.forEach(drawEdge);

  // Frame: a box sized to the graph, with a zoom bar.
  const frame = document.createElement("div");
  frame.className = "graph-box";
  const bar = document.createElement("div");
  bar.className = "zoom-bar";
  for (const [label, text, fn] of [["Zoom in", "+", () => zoom(1.25)], ["Zoom out", "−", () => zoom(0.8)], ["Reset layout", "⟲", reset]]) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "zoom-btn";
    b.textContent = text;
    b.setAttribute("aria-label", label);
    b.title = label;
    b.onclick = fn;
    bar.append(b);
  }
  frame.append(svg, bar);
  box.replaceChildren(frame);
  frame.style.height = `${Math.min(Math.max(H * Math.min(1, frame.clientWidth / W) + 2, 200), Math.max(300, innerHeight * 0.6))}px`;

  let view = { x: 0, y: 0, scale: 1 };
  const apply = () => {
    const t = `translate(${view.x} ${view.y}) scale(${view.scale})`;
    viewport.setAttribute("transform", t);
    background.setAttribute("transform", t);
  };
  const toBox = (cx, cy) => {
    const b = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
    return { x: (cx - b.left) * vb.width / b.width, y: (cy - b.top) * vb.height / b.height };
  };
  const delta = (dx, dy) => {
    const b = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
    return { x: dx * vb.width / b.width / view.scale, y: dy * vb.height / b.height / view.scale };
  };
  function zoomAt(point, factor) {
    const old = view.scale;
    view.scale = Math.min(Math.max(old * factor, 0.35), 4);
    const r = view.scale / old;
    view.x = point.x - (point.x - view.x) * r;
    view.y = point.y - (point.y - view.y) * r;
    apply();
  }
  function zoom(factor) { zoomAt({ x: W / 2, y: H / 2 }, factor); }
  function reset() {
    view = { x: 0, y: 0, scale: 1 };
    apply();
    for (const id of graph.ids) { const [x, y] = xy(id); place(id, x, y); }
    edgeParts.forEach(drawEdge);
    saved = {};
    try { localStorage.removeItem(positionsKey); } catch {}
  }

  let pan = null, drag = null;
  const tip = document.createElement("div");
  tip.className = "viz-tooltip";
  document.body.append(tip);
  for (const [id, card] of cards) {
    card.g.addEventListener("mouseenter", (e) => {
      if (!card.metadata) return;
      tip.textContent = card.metadata;
      tip.style.display = "block";
      tip.style.left = `${e.clientX + 14}px`;
      tip.style.top = `${e.clientY + 14}px`;
    });
    card.g.addEventListener("mousemove", (e) => { tip.style.left = `${e.clientX + 14}px`; tip.style.top = `${e.clientY + 14}px`; });
    card.g.addEventListener("mouseleave", () => { tip.style.display = "none"; });
    card.g.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation();
      drag = { id, cx: e.clientX, cy: e.clientY, x: Number(card.g.dataset.x), y: Number(card.g.dataset.y) };
      card.g.classList.add("dragging");
    });
  }
  svg.addEventListener("mousedown", (e) => {
    if (e.button !== 0 || e.target.closest(".node")) return;
    pan = { cx: e.clientX, cy: e.clientY, x: view.x, y: view.y };
    svg.classList.add("panning");
  });
  addEventListener("mousemove", (e) => {
    if (drag) {
      const d = delta(e.clientX - drag.cx, e.clientY - drag.cy);
      place(drag.id, drag.x + d.x, drag.y + d.y);
      edgeParts.filter((p) => p.edge.from === drag.id || p.edge.to === drag.id).forEach(drawEdge);
    } else if (pan) {
      const d = delta(e.clientX - pan.cx, e.clientY - pan.cy);
      view.x = pan.x + d.x * view.scale;
      view.y = pan.y + d.y * view.scale;
      apply();
    }
  });
  addEventListener("mouseup", () => {
    if (drag) {
      const g = cards.get(drag.id).g;
      g.classList.remove("dragging");
      saved[drag.id] = { x: Number(g.dataset.x), y: Number(g.dataset.y) };
      try { localStorage.setItem(positionsKey, JSON.stringify(saved)); } catch {}
      drag = null;
    }
    pan = null;
    svg.classList.remove("panning");
  });
  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomAt(toBox(e.clientX, e.clientY), e.deltaY < 0 ? 1.12 : 1 / 1.12);
  }, { passive: false });
  apply();

  /** Paint live state: each node's status class and sub line, and edge counts on labels. */
  function update(state = {}) {
    for (const [id, card] of cards) {
      const s = state.nodes?.[id] ?? {};
      const status = s.status ?? "pending";
      card.g.setAttribute("class", `node ${status}${card.g.classList.contains("dragging") ? " dragging" : ""}`);
      card.sub.textContent = s.sub ?? (status === "pending" ? "" : status);
      card.title.textContent = `${id === END ? "END" : id} — ${s.sub ?? status}`;
    }
    for (const [k, part] of labels) {
      const n = state.edges?.[k];
      part.label.textContent = [part.action, n ? `×${n}` : ""].filter(Boolean).join(" ");
      part.path.classList.toggle("taken", Boolean(n));
    }
  }
  update();
  return { update };
}
