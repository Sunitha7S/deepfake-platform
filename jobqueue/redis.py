"""Redis-backed job queue. Skeleton.

Satisfies ``QueueBackend`` structurally and raises ``NotImplementedError`` from
every operation, each naming the Redis structure and commands that will fill it
in. The same treatment ``S3CompatibleStorageBackend`` gets, for the same reason:
a skeleton that documents its own implementation is a design note that cannot
drift from the interface, because the interface is right there.

Planned key layout, per queue ``q``::

    dfp:q:{q}:ready      ZSET   score = (priority.rank, available_at, sequence)
    dfp:q:{q}:delayed    ZSET   score = available_at epoch
    dfp:q:{q}:leased     ZSET   score = lease expiry epoch
    dfp:q:{q}:dead       LIST
    dfp:job:{job_id}     HASH   the serialised envelope
    dfp:lease:{job_id}   STRING lease_id, with a TTL
    dfp:idem:{q}:{key}   STRING job_id, for idempotent enqueue

Two things must be Lua scripts rather than command sequences, and both are the
kind of bug that only appears under load:

**Leasing.** Pop from ``ready``, write ``leased``, and set ``dfp:lease:*`` have
to be one atomic step. Done as three round trips, two workers can both observe
the same job at the top of ``ready`` and both proceed, and the job is processed
twice.

**Ack and nack.** Each must check the presented ``lease_id`` against
``dfp:lease:{job_id}`` and act only if it still matches, in one step. A
check-then-act across two round trips lets a worker whose lease expired between
the two calls acknowledge a job a second worker is midway through — the exact
failure ``LeaseNotHeld`` exists to prevent, reintroduced by the transport.

Lease expiry is recovered by ``reap_expired_leases`` scanning the ``leased``
ZSET by score. Redis key TTLs are deliberately *not* the mechanism: a key that
simply vanishes tells nobody the job needs requeueing.
"""

from __future__ import annotations

from typing import Any

from jobqueue.base import DEFAULT_LEASE_SECONDS, BaseQueueBackend
from jobqueue.policy import RetryPolicy
from jobqueue.types import (
    JobEnvelope,
    LeasedJob,
    NackResult,
    Priority,
    QueueHealth,
    QueueStats,
)

_TODO = "RedisQueue is a skeleton; "


class RedisQueue(BaseQueueBackend):
    """``QueueBackend`` over Redis. Not yet implemented."""

    name = "redis"

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        namespace: str = "dfp",
        policy: RetryPolicy | None = None,
        client: Any = None,
    ) -> None:
        super().__init__(policy)
        self.url = url
        self.namespace = namespace
        self._client = client

    def __repr__(self) -> str:
        # The URL can carry a password. Render the host only.
        safe = self.url.split("@")[-1] if "@" in self.url else self.url
        return f"RedisQueue(url=…@{safe}, namespace={self.namespace!r})"

    # ---- key helpers (usable now; they are pure string building) ---------

    def _k_ready(self, queue: str) -> str:
        return f"{self.namespace}:q:{queue}:ready"

    def _k_delayed(self, queue: str) -> str:
        return f"{self.namespace}:q:{queue}:delayed"

    def _k_leased(self, queue: str) -> str:
        return f"{self.namespace}:q:{queue}:leased"

    def _k_dead(self, queue: str) -> str:
        return f"{self.namespace}:q:{queue}:dead"

    def _k_job(self, job_id: str) -> str:
        return f"{self.namespace}:job:{job_id}"

    def _k_lease(self, job_id: str) -> str:
        return f"{self.namespace}:lease:{job_id}"

    def _k_idem(self, queue: str, key: str) -> str:
        return f"{self.namespace}:idem:{queue}:{key}"

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
        raise NotImplementedError(
            _TODO + "enqueue will SET NX the idempotency key, HSET the envelope at "
            "dfp:job:{id}, then ZADD to ready or delayed. The sequence comes from "
            "INCR on a per-queue counter so FIFO among equal priority survives "
            "across processes. A caller-supplied job_id must be written with "
            "HSETNX so a collision fails loudly rather than overwriting another "
            "job's envelope."
        )

    # ---- consuming -------------------------------------------------------

    async def lease(
        self,
        queue: str,
        *,
        worker_id: str,
        count: int = 1,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> list[LeasedJob]:
        raise NotImplementedError(
            _TODO + "lease must be a single Lua script: ZRANGEBYSCORE ready, ZREM, "
            "ZADD leased with the expiry as score, SET dfp:lease:{id}. Splitting "
            "these into separate round trips lets two workers claim one job."
        )

    async def extend_lease(
        self, job_id: str, lease_id: str, *, lease_seconds: float = DEFAULT_LEASE_SECONDS
    ) -> LeasedJob:
        raise NotImplementedError(
            _TODO + "extend_lease will be a Lua script comparing dfp:lease:{id} to "
            "the presented lease_id before ZADD-ing the new expiry."
        )

    async def ack(self, job_id: str, lease_id: str) -> JobEnvelope:
        raise NotImplementedError(
            _TODO + "ack will be a Lua compare-and-delete on dfp:lease:{id}, then "
            "ZREM from leased and DEL the job hash."
        )

    async def nack(
        self, job_id: str, lease_id: str, *, error: str, retryable: bool = True
    ) -> NackResult:
        raise NotImplementedError(
            _TODO + "nack will compare-and-clear the lease, then either ZADD to "
            "delayed at policy.next_attempt_at() or LPUSH to dead — decided by "
            "the same RetryPolicy the in-memory backend uses, so both behave "
            "identically."
        )

    async def reap_expired_leases(self) -> int:
        raise NotImplementedError(
            _TODO + "reap will ZRANGEBYSCORE leased for scores <= now, then run the "
            "nack path with retryable=True for each. Key TTLs are not used: a key "
            "that vanishes silently tells nobody the job needs requeueing."
        )

    # ---- inspection ------------------------------------------------------

    async def get(self, job_id: str) -> JobEnvelope:
        raise NotImplementedError(_TODO + "get will HGETALL dfp:job:{id}.")

    async def dead_letters(self, queue: str, *, limit: int = 100) -> list[JobEnvelope]:
        raise NotImplementedError(_TODO + "dead_letters will LRANGE the dead list.")

    async def requeue_dead_letter(self, job_id: str) -> JobEnvelope:
        raise NotImplementedError(
            _TODO + "requeue will LREM from dead and ZADD back to ready with "
            "attempts reset to 0."
        )

    async def stats(self, queue: str) -> QueueStats:
        raise NotImplementedError(
            _TODO + "stats will ZCARD each set. oldest_ready_age_seconds needs "
            "ZRANGE ready 0 0 WITHSCORES and the envelope's enqueued_at."
        )

    async def purge(self, queue: str) -> int:
        raise NotImplementedError(
            _TODO + "purge will SCAN and DEL by prefix. Never KEYS — it blocks the "
            "server for the duration on a large keyspace."
        )

    # ---- operational -----------------------------------------------------

    async def health(self) -> QueueHealth:
        """Never raises, even as a skeleton.

        Reporting ``ok=False`` is correct behaviour rather than a shortcut: a
        health check exists to describe condition, and a backend that cannot
        serve is exactly what that means. Wiring this in prematurely therefore
        fails readiness instead of crashing the process.
        """
        return QueueHealth(
            ok=False, backend=self.name, detail="RedisQueue is not implemented"
        )
