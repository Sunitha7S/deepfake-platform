"""SQLite-backed job queue. Durable, and shared between processes on one host.

The API and the worker are separate processes (``uvicorn backend.app:create_app``
and ``python -m backend.worker_main``). ``InMemoryQueue`` cannot connect them —
each process gets its own dictionary, so the API enqueues into one and the
worker polls the other forever. ``RedisQueue`` is the multi-host answer and is
still a skeleton. This backend is the single-host answer, built on the standard
library alone: one file, no server, no new dependency.

**Atomicity.** Every operation is one ``BEGIN IMMEDIATE`` transaction. That
takes SQLite's write lock *before* the first read, so the read-decide-write
sequences ``RedisQueue``'s docstring says must be Lua scripts — leasing, and
the compare-and-act on ``lease_id`` in ack/nack/extend — are serialised across
every process holding the file. Two workers cannot both see one job at the top
of the ready set, and a worker whose lease was reaped cannot ack the job a
second worker now holds. A deferred ``BEGIN`` would not be enough: two readers
can both pass the check, and the loser's upgrade to a write lock fails with
``SQLITE_BUSY`` mid-decision.

**Semantics.** Identical to ``InMemoryQueue``, rule for rule — ordering is
``(priority.rank, available_at, sequence)``, a lapsed lease counts as a failed
attempt and backs off through the same ``RetryPolicy``, a non-retryable nack
dead-letters at once. Both run the same conformance suite
(``tests/queue_conformance.py``); a divergence is a failing check, not a
documentation note.

**Timestamps** are integer microseconds since the epoch. ``available_at`` is an
ordering key, and an integer orders exactly where ISO text with a variable
offset suffix, or a float, would not be guaranteed to.

**Locking and waiting.** ``busy_timeout_s`` (30 s) is how long an operation
waits for another process's write lock before raising
``sqlite3.OperationalError: database is locked``. Every transaction here is a
handful of indexed statements, so the lock is held for milliseconds; the
timeout only matters under pathological contention or a stalled disk. WAL mode
is set once per file and persists. Read-only calls (``get``, ``dead_letters``)
also take the write lock — simpler, and cheap at this scale. All timestamps,
lease expiry included, come from the clock of the process doing the write, so
every process sharing a file must share a clock: one host, by construction.

**Not for multiple hosts.** SQLite's locking is only as good as the filesystem's;
over NFS or SMB it is not trustworthy. One host, local disk — the same
deployment shape ``LocalStorageBackend`` already assumes.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from jobqueue.base import DEFAULT_LEASE_SECONDS, BaseQueueBackend
from jobqueue.errors import JobNotFound, JobNotLeased, LeaseNotHeld
from jobqueue.policy import RetryPolicy
from jobqueue.types import (
    JobEnvelope,
    JobState,
    Lease,
    LeasedJob,
    NackOutcome,
    NackResult,
    Priority,
    QueueHealth,
    QueueStats,
)

T = TypeVar("T")

SCHEMA_VERSION = 1
"""Stored in ``PRAGMA user_version``. A file written by a different layout is
refused rather than read: misreading a column is silent job corruption."""

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECOND = timedelta(microseconds=1)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    sequence          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id            TEXT    NOT NULL UNIQUE,
    queue             TEXT    NOT NULL,
    payload           TEXT    NOT NULL,
    state             TEXT    NOT NULL,
    priority          TEXT    NOT NULL,
    priority_rank     INTEGER NOT NULL,
    attempts          INTEGER NOT NULL,
    max_attempts      INTEGER NOT NULL,
    enqueued_at       INTEGER NOT NULL,
    available_at      INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL,
    idempotency_key   TEXT,
    last_error        TEXT,
    lease_id          TEXT,
    worker_id         TEXT,
    lease_acquired_at INTEGER,
    lease_expires_at  INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS jobs_idempotency
    ON jobs (queue, idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS jobs_ready
    ON jobs (queue, state, priority_rank, available_at, sequence);
CREATE INDEX IF NOT EXISTS jobs_leases
    ON jobs (lease_expires_at) WHERE lease_id IS NOT NULL;
"""

_CLEAR_LEASE = (
    "lease_id = NULL, worker_id = NULL, lease_acquired_at = NULL, lease_expires_at = NULL"
)


def _to_us(at: datetime) -> int:
    return (at - _EPOCH) // _MICROSECOND


def _from_us(us: int) -> datetime:
    return _EPOCH + timedelta(microseconds=us)


def _is_busy(exc: sqlite3.OperationalError) -> bool:
    name = getattr(exc, "sqlite_errorname", "") or ""
    if name.startswith(("SQLITE_BUSY", "SQLITE_LOCKED")):
        return True
    text = str(exc).lower()
    return "database is locked" in text or "database is busy" in text


def _envelope(row: sqlite3.Row) -> JobEnvelope:
    return JobEnvelope(
        job_id=row["job_id"],
        queue=row["queue"],
        payload=json.loads(row["payload"]),
        state=JobState(row["state"]),
        priority=Priority(row["priority"]),
        sequence=row["sequence"],
        attempts=row["attempts"],
        max_attempts=row["max_attempts"],
        enqueued_at=_from_us(row["enqueued_at"]),
        available_at=_from_us(row["available_at"]),
        updated_at=_from_us(row["updated_at"]),
        idempotency_key=row["idempotency_key"],
        last_error=row["last_error"],
    )


def _lease(row: sqlite3.Row) -> Lease:
    return Lease(
        job_id=row["job_id"],
        lease_id=row["lease_id"],
        worker_id=row["worker_id"],
        acquired_at=_from_us(row["lease_acquired_at"]),
        expires_at=_from_us(row["lease_expires_at"]),
    )


class SQLiteQueue(BaseQueueBackend):
    """``QueueBackend`` over one SQLite file, safe across processes on one host."""

    name = "sqlite"

    def __init__(
        self,
        path: str | Path,
        *,
        policy: RetryPolicy | None = None,
        busy_timeout_s: float = 30.0,
    ) -> None:
        super().__init__(policy)
        # Absolute now, against the working directory this process started in.
        # A relative path left relative is resolved again by every process that
        # opens it: an API and a worker launched from different directories
        # would each get their own file and silently never meet — the exact
        # failure this backend exists to remove. Both processes log this value
        # at startup so a mismatch is visible.
        self.path = Path(path).absolute()
        self.busy_timeout_s = busy_timeout_s
        # Opening is lazy: a misconfigured path must surface as an unhealthy
        # /health at boot, not as an exception out of create_app().
        self._ready = False
        self._init_lock = threading.Lock()

    def __repr__(self) -> str:
        # The path verbatim, not repr()'d: on Windows repr doubles every
        # backslash, and an operator comparing two processes' startup lines
        # should be comparing the same text.
        return f"SQLiteQueue(path={self.path})"

    # ---- connection and transaction plumbing -----------------------------

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None: the module must not open transactions behind
        # our back; every one is an explicit BEGIN IMMEDIATE below.
        conn = sqlite3.connect(self.path, timeout=self.busy_timeout_s, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Switching a fresh file to WAL needs an exclusive lock, and SQLite
            # does NOT run the busy handler for a journal-mode change: when the
            # API and a worker open a new file at the same moment, the loser
            # gets "database is locked" immediately, whatever busy_timeout_s
            # says (measured: 44 of 600 concurrent first-opens failed before
            # this loop). So first-open contention is retried here, with
            # backoff, inside the same time budget every other operation gets.
            deadline = time.monotonic() + self.busy_timeout_s
            delay = 0.005
            while True:
                try:
                    self._initialise()
                    break
                except sqlite3.OperationalError as exc:
                    if not _is_busy(exc) or time.monotonic() >= deadline:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, 0.25)
            self._ready = True

    def _initialise(self) -> None:
        conn = self._connect()
        try:
            # WAL lets readers proceed while one writer holds the lock. The mode
            # is persistent in the file, so only the first opener ever writes it.
            if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN IMMEDIATE")
            try:
                found = conn.execute("PRAGMA user_version").fetchone()[0]
                if found not in (0, SCHEMA_VERSION):
                    raise RuntimeError(
                        f"{self.path} has queue schema version {found}; this "
                        f"build reads version {SCHEMA_VERSION}. Refusing to guess."
                    )
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def _transact(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``fn`` inside one write-locked transaction. Blocking."""
        self._ensure_schema()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(conn)
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            return result
        finally:
            conn.close()

    async def _run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        # A busy wait on the file lock must not stall the event loop that is
        # also heartbeating leases and serving requests.
        return await asyncio.to_thread(self._transact, fn)

    @staticmethod
    def _row(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return row

    @staticmethod
    def _promote_due(conn: sqlite3.Connection, now_us: int) -> None:
        """DELAYED jobs whose time has come become READY."""
        conn.execute(
            "UPDATE jobs SET state = ?, updated_at = ? WHERE state = ? AND available_at <= ?",
            (JobState.READY.value, now_us, JobState.DELAYED.value, now_us),
        )

    def _reap(self, conn: sqlite3.Connection, now: datetime) -> int:
        """Return jobs whose lease lapsed. Counts as a failed attempt.

        Same rule as ``InMemoryQueue._reap_sync``: back off into DELAYED, never
        straight to READY, so a job that reliably kills its worker is not handed
        to the next one at once; dead-letter when the budget is spent.
        """
        now_us = _to_us(now)
        expired = conn.execute(
            "SELECT * FROM jobs WHERE lease_id IS NOT NULL AND lease_expires_at <= ?",
            (now_us,),
        ).fetchall()
        for row in expired:
            attempts = row["attempts"] + 1
            error = f"lease expired; worker {row['worker_id']} did not report back"
            if self.policy.should_retry(
                attempts=attempts, retryable=True, max_attempts=row["max_attempts"]
            ):
                retry_at = self.policy.next_attempt_at(now, attempts=attempts)
                conn.execute(
                    f"UPDATE jobs SET state = ?, attempts = ?, available_at = ?, "
                    f"updated_at = ?, last_error = ?, {_CLEAR_LEASE} WHERE job_id = ?",
                    (JobState.DELAYED.value, attempts, _to_us(retry_at), now_us, error,
                     row["job_id"]),
                )
            else:
                conn.execute(
                    f"UPDATE jobs SET state = ?, attempts = ?, updated_at = ?, "
                    f"last_error = ?, {_CLEAR_LEASE} WHERE job_id = ?",
                    (JobState.DEAD.value, attempts, now_us, error, row["job_id"]),
                )
        return len(expired)

    def _held_lease(self, conn: sqlite3.Connection, job_id: str, lease_id: str) -> sqlite3.Row:
        """The compare half of compare-and-act, inside the caller's transaction."""
        row = self._row(conn, job_id)
        if row["lease_id"] is None:
            raise JobNotLeased(job_id)
        self.assert_lease_current(job_id, row["lease_id"], lease_id)
        return row

    # ---- producing -------------------------------------------------------

    async def enqueue(
        self,
        queue: str,
        payload: dict[str, Any],
        *,
        priority: Priority = Priority.NORMAL,
        delay_s: float = 0.0,
        max_attempts: int | None = None,
        idempotency_key: str | None = None,
        job_id: str | None = None,
    ) -> JobEnvelope:
        # Serialised before the transaction: a payload that is not JSON fails
        # at the call site, not as a corrupt row a worker later chokes on.
        encoded = json.dumps(payload, allow_nan=False)
        new_id = job_id or self.new_job_id()
        attempts_budget = max_attempts or self.policy.max_attempts

        def op(conn: sqlite3.Connection) -> JobEnvelope:
            if idempotency_key is not None:
                existing = conn.execute(
                    "SELECT * FROM jobs WHERE queue = ? AND idempotency_key = ?",
                    (queue, idempotency_key),
                ).fetchone()
                if existing is not None:
                    return _envelope(existing)
            if job_id is not None and conn.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone():
                # A collision, not a dedup request — returning the other job
                # would hand back someone else's analysis.
                raise ValueError(f"job_id {job_id!r} is already in use")

            now = self.now()
            now_us = _to_us(now)
            state = JobState.DELAYED if delay_s > 0 else JobState.READY
            conn.execute(
                "INSERT INTO jobs (job_id, queue, payload, state, priority, priority_rank, "
                "attempts, max_attempts, enqueued_at, available_at, updated_at, "
                "idempotency_key) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)",
                (new_id, queue, encoded, state.value, priority.value, priority.rank,
                 attempts_budget, now_us, _to_us(now + timedelta(seconds=delay_s)), now_us,
                 idempotency_key),
            )
            return _envelope(self._row(conn, new_id))

        return await self._run(op)

    # ---- consuming -------------------------------------------------------

    async def reap_expired_leases(self) -> int:
        return await self._run(lambda conn: self._reap(conn, self.now()))

    async def lease(
        self,
        queue: str,
        *,
        worker_id: str,
        count: int = 1,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> list[LeasedJob]:
        self.lease_deadline(lease_seconds)  # reject a non-positive duration up front

        def op(conn: sqlite3.Connection) -> list[LeasedJob]:
            now = self.now()
            now_us = _to_us(now)
            # Reaping here keeps at-least-once true with no supervisor running.
            self._reap(conn, now)
            self._promote_due(conn, now_us)
            rows = conn.execute(
                "SELECT job_id FROM jobs WHERE queue = ? AND state = ? AND available_at <= ? "
                "ORDER BY priority_rank, available_at, sequence LIMIT ?",
                (queue, JobState.READY.value, now_us, max(0, count)),
            ).fetchall()
            leased: list[LeasedJob] = []
            for (candidate,) in rows:
                lease_id = self.new_lease_id()
                expires_us = _to_us(self.lease_deadline(lease_seconds))
                conn.execute(
                    "UPDATE jobs SET state = ?, updated_at = ?, lease_id = ?, worker_id = ?, "
                    "lease_acquired_at = ?, lease_expires_at = ? WHERE job_id = ?",
                    (JobState.LEASED.value, now_us, lease_id, worker_id, now_us, expires_us,
                     candidate),
                )
                row = self._row(conn, candidate)
                leased.append(LeasedJob(envelope=_envelope(row), lease=_lease(row)))
            return leased

        return await self._run(op)

    async def extend_lease(
        self, job_id: str, lease_id: str, *, lease_seconds: float = DEFAULT_LEASE_SECONDS
    ) -> LeasedJob:
        self.lease_deadline(lease_seconds)

        def op(conn: sqlite3.Connection) -> LeasedJob:
            row = self._row(conn, job_id)
            self.assert_lease_current(job_id, row["lease_id"], lease_id)
            conn.execute(
                "UPDATE jobs SET lease_expires_at = ? WHERE job_id = ?",
                (_to_us(self.lease_deadline(lease_seconds)), job_id),
            )
            row = self._row(conn, job_id)
            return LeasedJob(envelope=_envelope(row), lease=_lease(row))

        return await self._run(op)

    async def ack(self, job_id: str, lease_id: str) -> JobEnvelope:
        def op(conn: sqlite3.Connection) -> JobEnvelope:
            row = self._held_lease(conn, job_id, lease_id)
            conn.execute(
                f"UPDATE jobs SET state = ?, attempts = ?, updated_at = ?, {_CLEAR_LEASE} "
                "WHERE job_id = ?",
                (JobState.DONE.value, row["attempts"] + 1, _to_us(self.now()), job_id),
            )
            return _envelope(self._row(conn, job_id))

        return await self._run(op)

    async def nack(
        self, job_id: str, lease_id: str, *, error: str, retryable: bool = True
    ) -> NackResult:
        def op(conn: sqlite3.Connection) -> NackResult:
            row = self._held_lease(conn, job_id, lease_id)
            now = self.now()
            now_us = _to_us(now)
            attempts = row["attempts"] + 1
            if self.policy.should_retry(
                attempts=attempts, retryable=retryable, max_attempts=row["max_attempts"]
            ):
                retry_at = self.policy.next_attempt_at(now, attempts=attempts)
                conn.execute(
                    f"UPDATE jobs SET state = ?, attempts = ?, available_at = ?, "
                    f"updated_at = ?, last_error = ?, {_CLEAR_LEASE} WHERE job_id = ?",
                    (JobState.DELAYED.value, attempts, _to_us(retry_at), now_us, error, job_id),
                )
                return NackResult(
                    outcome=NackOutcome.RETRY_SCHEDULED,
                    envelope=_envelope(self._row(conn, job_id)),
                    retry_at=retry_at,
                )
            conn.execute(
                f"UPDATE jobs SET state = ?, attempts = ?, updated_at = ?, last_error = ?, "
                f"{_CLEAR_LEASE} WHERE job_id = ?",
                (JobState.DEAD.value, attempts, now_us, error, job_id),
            )
            return NackResult(
                outcome=NackOutcome.DEAD_LETTERED,
                envelope=_envelope(self._row(conn, job_id)),
                retry_at=None,
            )

        return await self._run(op)

    # ---- inspection ------------------------------------------------------

    async def get(self, job_id: str) -> JobEnvelope:
        return await self._run(lambda conn: _envelope(self._row(conn, job_id)))

    async def dead_letters(self, queue: str, *, limit: int = 100) -> list[JobEnvelope]:
        def op(conn: sqlite3.Connection) -> list[JobEnvelope]:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE queue = ? AND state = ? "
                "ORDER BY updated_at, sequence LIMIT ?",
                (queue, JobState.DEAD.value, max(0, limit)),
            ).fetchall()
            return [_envelope(r) for r in rows]

        return await self._run(op)

    async def requeue_dead_letter(self, job_id: str) -> JobEnvelope:
        def op(conn: sqlite3.Connection) -> JobEnvelope:
            row = self._row(conn, job_id)
            if row["state"] != JobState.DEAD.value:
                raise LeaseNotHeld(job_id, "-", f"job is {row['state']}, not dead")
            now_us = _to_us(self.now())
            conn.execute(
                "UPDATE jobs SET state = ?, attempts = 0, available_at = ?, updated_at = ? "
                "WHERE job_id = ?",
                (JobState.READY.value, now_us, now_us, job_id),
            )
            return _envelope(self._row(conn, job_id))

        return await self._run(op)

    async def stats(self, queue: str) -> QueueStats:
        def op(conn: sqlite3.Connection) -> QueueStats:
            now = self.now()
            now_us = _to_us(now)
            self._promote_due(conn, now_us)
            counts = dict(
                conn.execute(
                    "SELECT state, COUNT(*) FROM jobs WHERE queue = ? GROUP BY state", (queue,)
                ).fetchall()
            )
            ready, oldest = conn.execute(
                "SELECT COUNT(*), MIN(enqueued_at) FROM jobs "
                "WHERE queue = ? AND state = ? AND available_at <= ?",
                (queue, JobState.READY.value, now_us),
            ).fetchone()
            return QueueStats(
                queue=queue,
                ready=ready,
                delayed=counts.get(JobState.DELAYED.value, 0),
                leased=counts.get(JobState.LEASED.value, 0),
                dead=counts.get(JobState.DEAD.value, 0),
                oldest_ready_age_seconds=(
                    max(0.0, (now - _from_us(oldest)).total_seconds())
                    if oldest is not None
                    else None
                ),
            )

        return await self._run(op)

    async def purge(self, queue: str) -> int:
        return await self._run(
            lambda conn: conn.execute("DELETE FROM jobs WHERE queue = ?", (queue,)).rowcount
        )

    async def health(self) -> QueueHealth:
        started = time.perf_counter()
        try:
            await self._run(lambda conn: conn.execute("SELECT 1").fetchone())
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return QueueHealth(
                ok=False, backend=self.name, detail=f"{type(exc).__name__}: {exc}"
            )
        return QueueHealth(
            ok=True, backend=self.name, latency_ms=(time.perf_counter() - started) * 1000
        )
