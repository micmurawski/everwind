/** Conversation checkpoints and exclusive access for resumable sessions. */

import { closeSync, fsyncSync, mkdirSync, openSync, readFileSync, renameSync, rmSync, writeSync } from "node:fs";
import { resolve } from "node:path";
import { z } from "zod";
import { validateSessionId } from "./sessionLog.ts";
import { MessageSchema } from "./types.ts";
import { KeyedMutex, clone, sleep } from "./util.ts";

export const SessionSchema = z.strictObject({
  sessionId: z.string(),
  agent: z.string(),
  systemPrompt: z.string(),
  /** Supervisor text appended to the system prompt for this session (a task frame); fixed once set. */
  frame: z.string().nullable().default(null),
  /** The scope of the session's latest run; a continuation without one reuses it. */
  scope: z.record(z.string(), z.unknown()).nullable().default(null),
  messages: z.array(MessageSchema).default([]),
  logPath: z.string().nullable().default(null),
  status: z.string().default("completed"),
});
export type Session = z.output<typeof SessionSchema>;
export type SessionInput = z.input<typeof SessionSchema>;

export function session(input: SessionInput): Session {
  return SessionSchema.parse(clone(input));
}

/** load/save are called while holding the session's lock for the entire agent invocation. */
export interface SessionStore {
  withLock<T>(sessionId: string, fn: () => Promise<T>): Promise<T>;
  load(sessionId: string): Session | null;
  save(session: Session): void;
  /** Optional: an asynchronous save, preferred when present. */
  asave?(session: Session): Promise<void>;
}

export class MemorySessionStore implements SessionStore {
  private sessions = new Map<string, Session>();
  private mutex = new KeyedMutex();

  withLock<T>(sessionId: string, fn: () => Promise<T>): Promise<T> {
    validateSessionId(sessionId);
    return this.mutex.run(sessionId, fn);
  }

  load(sessionId: string): Session | null {
    const found = this.sessions.get(sessionId);
    return found ? clone(found) : null;
  }

  save(value: Session): void {
    this.sessions.set(value.sessionId, clone(value));
  }
}

function processAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

/**
 * Atomic JSON checkpoints with lock files (local filesystem).
 *
 * A lock file holds its owner's PID; a lock whose process is gone is taken over.
 * Waiting never blocks the event loop. Checkpoints contain conversation only.
 */
export class FileSessionStore implements SessionStore {
  readonly directory: string;
  private mutex = new KeyedMutex();

  constructor(directory: string) {
    this.directory = resolve(directory);
  }

  private path(sessionId: string): string {
    validateSessionId(sessionId);
    return resolve(this.directory, `${sessionId}.json`);
  }

  withLock<T>(sessionId: string, fn: () => Promise<T>): Promise<T> {
    const lockPath = this.path(sessionId).replace(/\.json$/, ".lock");
    // In-process waiters queue on the mutex; other processes are excluded by the file.
    return this.mutex.run(sessionId, async () => {
      mkdirSync(this.directory, { recursive: true });
      for (;;) {
        try {
          const fd = openSync(lockPath, "wx");
          writeSync(fd, String(process.pid));
          closeSync(fd);
          break;
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
          let owner = Number.NaN;
          try {
            owner = Number(readFileSync(lockPath, "utf-8"));
          } catch {}
          if (Number.isInteger(owner) && owner > 0 && !processAlive(owner)) {
            rmSync(lockPath, { force: true });
            continue;
          }
          await sleep(0.02);
        }
      }
      try {
        return await fn();
      } finally {
        rmSync(lockPath, { force: true });
      }
    });
  }

  load(sessionId: string): Session | null {
    let text: string;
    try {
      text = readFileSync(this.path(sessionId), "utf-8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw error;
    }
    const loaded = SessionSchema.parse(JSON.parse(text));
    if (loaded.sessionId !== sessionId) throw new Error("Session checkpoint ID does not match its filename");
    return loaded;
  }

  save(value: Session): void {
    const path = this.path(value.sessionId);
    const temporary = `${path}.tmp`;
    mkdirSync(this.directory, { recursive: true });
    try {
      const fd = openSync(temporary, "w");
      try {
        writeSync(fd, JSON.stringify(value));
        fsyncSync(fd);
      } finally {
        closeSync(fd);
      }
      renameSync(temporary, path);
    } finally {
      rmSync(temporary, { force: true });
    }
  }
}
