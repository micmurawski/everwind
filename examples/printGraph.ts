/** Print a loop as Mermaid or DOT, optionally render it to a local PNG or SVG. */

import { parseArgs } from "node:util";
import { PlanThenAct, ReAct } from "../src/index.ts";

const { values } = parseArgs({
  options: {
    loop: { type: "string", default: "react" },
    format: { type: "string", default: "mermaid" },
    direction: { type: "string", default: "LR" },
    output: { type: "string" },
  },
});
const flow = (values.loop === "plan" ? new PlanThenAct() : new ReAct()).toFlow();
const options = { direction: values.direction };
process.stdout.write(values.format === "dot" ? flow.toDot(options) : flow.toMermaid(options));
if (values.output) await flow.render(values.output, options); // Requires Graphviz.
