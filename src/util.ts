/** Small runtime helpers: JSON values, IDs, hashing, locks and abortable waits. */

import { createHash } from "node:crypto";

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export type JsonObject = { [key: string]: Json };

export function hexId(): string {
  return crypto.randomUUID().replaceAll("-", "");
}

/** A private deep copy of a JSON-compatible value. */
export function clone<T>(value: T): T {
  return value === undefined ? value : structuredClone(value);
}

/** JSON with sorted object keys, so equal values have equal text. */
export function stableStringify(value: unknown): string {
  return JSON.stringify(sortKeys(value));
}

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (value !== null && typeof value === "object") {
    if (value instanceof Date) return value.toISOString();
    const sorted: Record<string, unknown> = {};
    for (const key of Object.keys(value).sort()) {
      const item = (value as Record<string, unknown>)[key];
      if (item !== undefined) sorted[key] = sortKeys(item);
    }
    return sorted;
  }
  return value;
}

export function sha256(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

/** A FIFO mutex per key, for exclusive access within one process. */
export class KeyedMutex {
  private tails = new Map<string, Promise<void>>();

  async run<T>(key: string, fn: () => Promise<T>): Promise<T> {
    const previous = this.tails.get(key) ?? Promise.resolve();
    let release!: () => void;
    const current = new Promise<void>((resolve) => (release = resolve));
    const tail = previous.then(() => current);
    this.tails.set(key, tail);
    await previous;
    try {
      return await fn();
    } finally {
      release();
      if (this.tails.get(key) === tail) this.tails.delete(key);
    }
  }
}

export function sleep(seconds: number, signal?: AbortSignal): Promise<void> {
  return abortable(new Promise((resolve) => setTimeout(resolve, seconds * 1000)), signal);
}

/** Settle with `promise`, or reject with the signal's reason as soon as it aborts. */
export function abortable<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (signal === undefined) return promise;
  if (signal.aborted) {
    promise.catch(() => {});
    return Promise.reject(signal.reason);
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", onAbort);
        reject(error);
      },
    );
  });
}

/** An AbortController that also aborts when any parent signal does. */
export function childController(...parents: (AbortSignal | undefined)[]): AbortController {
  const controller = new AbortController();
  for (const parent of parents) {
    if (parent === undefined) continue;
    if (parent.aborted) {
      controller.abort(parent.reason);
      break;
    }
    parent.addEventListener("abort", () => controller.abort(parent.reason), { once: true });
  }
  return controller;
}

export function isPositiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 1;
}
