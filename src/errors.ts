/** Error classes shared by every module. Each sets `name` to its class name. */

export class EverwindError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** The requested session log could not be opened or written. */
export class SessionLogError extends EverwindError {}

/** The journal could not be written. The run must stop without further effects. */
export class JournalError extends EverwindError {}

/** Another owner claimed this run (a newer epoch), or the run is no longer running. */
export class RunOwnershipLost extends JournalError {}

/** The run's current owner still holds a valid lease; pass `force: true` to take over. */
export class RunLeaseHeld extends EverwindError {}

/** Replay produced a different command than the journal recorded at this position. */
export class NonDeterminismError extends EverwindError {
  constructor(
    readonly seq: number,
    readonly expected: string,
    readonly actual: string,
    message?: string,
  ) {
    super(message ?? `Command ${seq}: journal recorded ${expected}, code issued ${actual}`);
  }
}

/** The journal recorded a version of a change that this code no longer supports. */
export class VersionUnsupported extends NonDeterminismError {
  constructor(
    readonly changeId: string,
    recorded: number,
    minSupported: number,
    maxVersion: number,
  ) {
    super(
      0,
      String(recorded),
      `${minSupported}..${maxVersion}`,
      `Change ${JSON.stringify(changeId)}: journal recorded version ${recorded}, code supports ${minSupported}..${maxVersion}`,
    );
  }
}

/** The agent's configuration differs from the one the run was recorded with. */
export class ConfigurationChanged extends EverwindError {
  constructor(readonly changed: readonly string[]) {
    super(
      `Configuration changed since the run was recorded: ${changed.join(", ")}. ` +
        "Pass allowFingerprintChange: true if the recorded commands still replay.",
    );
  }
}

/** Inspection replay reached a command it must not perform. */
export class ReplayExhausted extends EverwindError {
  constructor(
    readonly seq: number,
    readonly reason: string,
  ) {
    super(`Replay stopped before command ${seq}: ${reason}`);
  }
}

/** A run budget (model calls, tool calls, steps) was exhausted. */
export class LimitReached extends EverwindError {}

/** A run or tool deadline passed. */
export class DeadlineExceeded extends EverwindError {}

/** The caller cancelled the run. */
export class Cancelled extends EverwindError {}

/** A tool needed a context value the run was not given. */
export class MissingContext extends EverwindError {}

/** An operator froze the run: it stopped at a step without finishing and stays recoverable. */
export class Frozen extends EverwindError {}

/**
 * Errors that must never become a tool observation or a failed run: the journal
 * is in an unknown state (or an operator froze the run), so the run stops as it
 * is and stays recoverable.
 */
export function isStopError(error: unknown): boolean {
  return error instanceof JournalError || error instanceof NonDeterminismError || error instanceof Frozen;
}

/** Errors a tool must not swallow: stop errors, a failed session log, missing context. */
export function isFatalToolError(error: unknown): boolean {
  return isStopError(error) || error instanceof SessionLogError || error instanceof MissingContext;
}

export function describeError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  return `Error: ${String(error)}`;
}
