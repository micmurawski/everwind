/** Build dist/: JS and declarations with tsc, then fix declaration imports and copy assets. */

import { copyFileSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

rmSync("dist", { recursive: true, force: true });
const tsc = Bun.spawnSync(["bunx", "tsc", "-p", "tsconfig.build.json"], { stdout: "inherit", stderr: "inherit" });
if (tsc.exitCode !== 0) process.exit(tsc.exitCode ?? 1);

// tsc rewrites ".ts" specifiers in JS output only; declarations must point at the shipped files too.
function walk(directory: string): string[] {
  return readdirSync(directory).flatMap((name) => {
    const path = join(directory, name);
    return statSync(path).isDirectory() ? walk(path) : [path];
  });
}
for (const file of walk("dist").filter((path) => path.endsWith(".d.ts"))) {
  const text = readFileSync(file, "utf-8");
  writeFileSync(file, text.replace(/(from\s+["']\.{1,2}\/[^"']+)\.ts(["'])/g, "$1.js$2").replace(/(import\(["']\.{1,2}\/[^"']+)\.ts(["']\))/g, "$1.js$2"));
}
copyFileSync("src/python/worker.py", "dist/python/worker.py");
console.log("built dist/");
