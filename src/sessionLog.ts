/** Append-only, per-session JSONL logging for agents without a durable store. */

import { closeSync, existsSync, fsyncSync, mkdirSync, openSync, readFileSync, writeSync } from "node:fs";
import { resolve } from "node:path";
import { SessionLogError } from "./errors.ts";
import type { Event } from "./types.ts";

export const LOG_SCHEMA_VERSION = 1;

export function validateSessionId(sessionId: unknown): asserts sessionId is string {
  if (typeof sessionId !== "string" || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(sessionId)) {
    throw new Error(
      "sessionId must be 1–128 letters, digits, underscores or hyphens, starting with a letter/digit",
    );
  }
}

export function logFileName(sessionId: string): string {
  return `everwind-${sessionId}.jsonl`;
}

export function eventLine(event: Event): string {
  return `${JSON.stringify({ schemaVersion: LOG_SCHEMA_VERSION, ...event, sessionId: event.sessionId ?? event.runId })}\n`;
}

/** Per-session JSONL log of an agent without a durable store; a store keeps its own log. */
export class SessionLog {
  readonly path: string;
  private fd: number | null = null;

  constructor(
    directory: string,
    readonly sessionId: string,
    readonly resume = false,
  ) {
    validateSessionId(sessionId);
    this.path = resolve(directory, logFileName(sessionId));
  }

  open(): this {
    try {
      mkdirSync(resolve(this.path, ".."), { recursive: true });
      if (this.resume) {
        // Appending is authorized only by a matching, locked checkpoint.
        const text = existsSync(this.path) ? readFileSync(this.path, "utf-8") : "";
        if (text && !text.endsWith("\n")) throw new Error("Incomplete or mismatched session log");
        for (const line of text.split("\n")) {
          if (!line) continue;
          const record = JSON.parse(line);
          if (record?.sessionId !== this.sessionId) throw new Error("Incomplete or mismatched session log");
        }
      }
      this.fd = openSync(this.path, this.resume ? "a" : "wx");
    } catch (error) {
      throw new SessionLogError(`Cannot create session log ${this.path}: ${(error as Error).message}`, {
        cause: error,
      });
    }
    return this;
  }

  write(event: Event): void {
    if (this.fd === null) throw new SessionLogError("Session log is not open");
    try {
      writeSync(this.fd, eventLine(event));
    } catch (error) {
      throw new SessionLogError(`Cannot write session log ${this.path}: ${(error as Error).message}`, {
        cause: error,
      });
    }
  }

  get closed(): boolean {
    return this.fd === null;
  }

  close(): void {
    if (this.fd === null) return;
    const fd = this.fd;
    this.fd = null;
    try {
      fsyncSync(fd);
      closeSync(fd);
    } catch (error) {
      throw new SessionLogError(`Cannot close session log ${this.path}: ${(error as Error).message}`, {
        cause: error,
      });
    }
  }
}
