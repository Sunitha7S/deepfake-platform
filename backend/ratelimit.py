"""Per-client request limits for the endpoints that create work.

**What is limited.** ``POST /uploads`` and ``POST /analyze`` — the two calls
that allocate storage and queue analysis. They share one budget per client:
``rate_limit_per_minute`` requests (60), so a full submission (negotiate, then
analyze) costs two. Reads — ``/jobs``, ``/events``, ``/results``, ``/health`` —
and the byte ``PUT`` are not limited: a client polling ``/jobs`` once a second
would otherwise exhaust the budget on its own status checks.

**Who a client is.** An anonymous caller is its **direct peer address** — the
TCP connection's remote host as the ASGI server reports it. ``X-Forwarded-For``,
``Forwarded`` and ``X-Real-IP`` are never read: no trusted proxy is configured
anywhere in this repository, and without one those headers are whatever the
client chose to send, so honouring them would let anyone pick a fresh budget
per request. Behind a reverse proxy every caller therefore shares the proxy's
budget. Supporting that deployment means configuring the proxy's address as
trusted first — deliberately not done here. An authenticated principal (none
exist yet) keys on its own ``rate_limit_key`` instead.

The ASGI server must not rewrite the peer either: uvicorn's proxy-headers
handling (on by default, trusting 127.0.0.1) replaces ``scope["client"]`` with
the ``X-Forwarded-For`` value before this code runs, and nothing here can
recover the original. Run it with ``--no-proxy-headers`` (the README command
does).

**Window.** Fixed one-minute windows aligned to the epoch: request *n* in the
window ending at *T* is allowed while ``n <= limit``, and a refused request is
told ``Retry-After: ceil(T - now)`` seconds (1 to 60), after which a fresh
window starts. The known cost of fixed windows: a client can spend one budget
at the end of a window and the next at the start of the following one, up to
twice the limit across a boundary. Accepted for its simplicity — one integer
per client per minute, updated in a single statement.

**Where the count lives.**

* ``memory`` — this process only. With several API processes each keeps its
  own count, so the effective limit is the limit times the process count. For
  tests and single-process development.
* ``sqlite`` — one file shared by every API process on the host. Each request is
  one ``BEGIN IMMEDIATE`` transaction around an upsert that increments and
  returns the count, so concurrent processes cannot both read the same value
  and both pass. Same single-host, local-disk constraint as the SQLite queue;
  **multi-host limiting is not supported** (there is no shared backend for it).

A limiter that cannot reach its store raises, and the request fails as
``internal_error``: an outage of the limiter is loud, never a silent switch to
unlimited.
"""

from __future__ import annotations

import asyncio
import math
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastapi import Request

WINDOW_S = 60
"""``rate_limit_per_minute`` is per minute; the window is not configurable."""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    count: int
    """Requests counted in this window, this one included."""
    retry_after_s: int
    """Seconds until the window resets (1-60). Meaningful when refused."""


class RateLimiter(Protocol):
    name: str
    limit: int

    async def hit(self, key: str) -> Decision:
        """Count one request against ``key`` and say whether it may proceed."""
        ...


def _window(now: float) -> tuple[int, int]:
    """(window start, seconds until it ends, rounded up, at least 1)."""
    start = int(now // WINDOW_S) * WINDOW_S
    remaining = math.ceil(start + WINDOW_S - now)
    return start, max(1, min(WINDOW_S, remaining))


class InMemoryRateLimiter:
    """Fixed-window counter inside one process."""

    name = "memory"

    def __init__(self, limit: int, *, clock: Callable[[], float] = time.time) -> None:
        self.limit = limit
        self._clock = clock
        self._counts: dict[tuple[str, int], int] = {}
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"InMemoryRateLimiter(limit={self.limit}/min)"

    async def hit(self, key: str) -> Decision:
        start, retry_after = _window(self._clock())
        with self._lock:
            # Earlier windows can never be counted against again.
            for stale in [k for k in self._counts if k[1] < start]:
                del self._counts[stale]
            count = self._counts.get((key, start), 0) + 1
            self._counts[(key, start)] = count
        return Decision(count <= self.limit, count, retry_after)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS rate_hits (
    client_key   TEXT    NOT NULL,
    window_start INTEGER NOT NULL,
    count        INTEGER NOT NULL,
    PRIMARY KEY (client_key, window_start)
);
CREATE INDEX IF NOT EXISTS rate_hits_window ON rate_hits (window_start);
"""


class SQLiteRateLimiter:
    """Fixed-window counter in one SQLite file, shared across processes on a host."""

    name = "sqlite"

    def __init__(
        self,
        path: str | Path,
        limit: int,
        *,
        clock: Callable[[], float] = time.time,
        busy_timeout_s: float = 30.0,
    ) -> None:
        self.limit = limit
        # Absolute now, for the reason SQLiteQueue gives: two processes started
        # from different directories must not each count in their own file.
        self.path = Path(path).absolute()
        self._clock = clock
        self.busy_timeout_s = busy_timeout_s
        self._ready = False
        self._init_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"SQLiteRateLimiter(path={self.path}, limit={self.limit}/min)"

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None: every transaction is the explicit BEGIN below.
        return sqlite3.connect(self.path, timeout=self.busy_timeout_s, isolation_level=None)

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # A journal-mode change skips SQLite's busy handler, so concurrent
            # first opens are retried here (the same fix as SQLiteQueue's).
            deadline = time.monotonic() + self.busy_timeout_s
            delay = 0.005
            while True:
                try:
                    conn = self._connect()
                    try:
                        if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                            conn.execute("PRAGMA journal_mode=WAL")
                        conn.executescript(_SCHEMA)
                    finally:
                        conn.close()
                    break
                except sqlite3.OperationalError as exc:
                    busy = "locked" in str(exc).lower() or "busy" in str(exc).lower()
                    if not busy or time.monotonic() >= deadline:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, 0.25)
            self._ready = True

    def _hit(self, key: str) -> Decision:
        self._ensure_schema()
        start, retry_after = _window(self._clock())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM rate_hits WHERE window_start < ?", (start,))
                (count,) = conn.execute(
                    "INSERT INTO rate_hits (client_key, window_start, count) VALUES (?, ?, 1) "
                    "ON CONFLICT (client_key, window_start) DO UPDATE SET count = count + 1 "
                    "RETURNING count",
                    (key, start),
                ).fetchone()
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()
        return Decision(count <= self.limit, count, retry_after)

    async def hit(self, key: str) -> Decision:
        # Waiting on another process's write lock must not stall the event loop.
        return await asyncio.to_thread(self._hit, key)


def build_rate_limiter(kind: str, *, limit: int, path: Path | None) -> RateLimiter:
    if kind == "memory":
        return InMemoryRateLimiter(limit)
    if kind == "sqlite":
        if path is None:
            raise ValueError("rate_limit_path is required when rate_limit_kind='sqlite'")
        return SQLiteRateLimiter(path, limit)
    raise ValueError(f"unknown rate_limit_kind {kind!r}")


def client_key(request: Request, principal_key: str, *, anonymous: bool) -> str:
    """The budget a request counts against.

    Anonymous: the direct peer address only — forwarding headers are ignored on
    purpose (see the module docstring). ``unknown`` when the server reports no
    peer (some ASGI transports); all such requests then share one budget.
    """
    if not anonymous:
        return principal_key
    host = request.client.host if request.client else None
    return f"ip:{host or 'unknown'}"
