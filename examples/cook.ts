/** A tiny cooking job for kitchenChef.ts: bun examples/cook.ts <dish> <seconds> <fail> */

const [dish, seconds, fail] = [process.argv[2], Number(process.argv[3]), process.argv[4] === "true"];
console.log(`${dish}: preparing`);
await Bun.sleep((seconds / 2) * 1000);
console.log(`${dish}: cooking`);
await Bun.sleep((seconds / 2) * 1000);
console.log(`${dish}: ${fail ? "burnt!" : "ready!"}`);
process.exit(fail ? 1 : 0);
export {};
