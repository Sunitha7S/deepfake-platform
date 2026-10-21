"""In-process job queue. Fully working.

Ordering is ``(priority.rank, available_at, sequence)``. The sequence is what
makes FIFO among equal-priority jobs a guarantee rather than an accident:
``enqueued_at`` has microsecond resolution and two jobs enqueued in the same
tick collide, leaving the tie to dict insertion order — which happens to be
correct in CPython today and is not a property to build on.

Lease expiry counts as a failed attempt. A worker that died is not owed an
unlimited number of retries: without that, a job that reliably kills its worker
cycles forever, and the queue looks busy while completing nothing. Expiry runs
the same retry policy as an explicit nack, so a poison job dead-letters on
schedule.

Not durable and not shared between processes. Fine for tests and for a single
process that hosts both the API and its workers; useless the moment the API and
``backend.worker_main`` run separately, which is why ``SQLiteQueue`` (one host)
and ``RedisQueue`` (many hosts, still a skeleton) exist behind the same protocol.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from datetime import datetime, timedelta
from typing import Any

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


class InMemoryQueue(BaseQueueBackend):
    """``QueueBackend`` backed by dictionaries and an asyncio lock."""

    name = "memory"

    def __init__(self, policy: RetryPolicy | None = None) -> None:
        super().__init__(policy)
        self._jobs: dict[str, JobEnvelope] = {}
        self._leases: dict[str, Lease] = {}
        self._by_idempotency: dict[tuple[str, str], str] = {}
        self._sequence = itertools.count(1)
        self._lock = asyncio.Lock()

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
        async with self._lock:
            if idempotency_key is not None:
                existing_id = self._by_idempotency.get((queue, idempotency_key))
                if existing_id is not None and existing_id in self._jobs:
                    # Returning the original rather than a new job is the whole
                    # point: the same viral clip submitted by fifty people is one
                    # unit of work, not fifty.
                    return self._jobs[existing_id]

            if job_id is not None and job_id in self._jobs:
                # A caller-supplied id that already exists is a collision, not a
                # dedup request — silently returning the other job would hand
                # back someone else's analysis.
                raise ValueError(f"job_id {job_id!r} is already in use")

            now = self.now()
            envelope = JobEnvelope(
                job_id=job_id or self.new_job_id(),
                queue=queue,
                payload=payload,
                state=JobState.DELAYED if delay_s > 0 else JobState.READY,
                priority=priority,
                sequence=next(self._sequence),
                attempts=0,
                max_attempts=max_attempts or self.policy.max_attempts,
                enqueued_at=now,
                available_at=now + timedelta(seconds=delay_s),
                updated_at=now,
                idempotency_key=idempotency_key,
            )
            self._jobs[envelope.job_id] = envelope
            if idempotency_key is not None:
                self._by_idempotency[(queue, idempotency_key)] = envelope.job_id
            return envelope

    # ---- consuming -------------------------------------------------------

    def _promote_due_sync(self, now: datetime) -> None:
        """DELAYED jobs whose time has come become READY."""
        for job_id, job in self._jobs.items():
            if job.state is JobState.DELAYED and now >= job.available_at:
                self._jobs[job_id] = job.model_copy(
                    update={"state": JobState.READY, "updated_at": now}
                )

    def _reap_sync(self, now: datetime) -> int:
        """Return jobs whose lease lapsed. Counts as a failed attempt."""
        reaped = 0
        for job_id, lease in list(self._leases.items()):
            if not lease.is_expired(now):
                continue
            del self._leases[job_id]
            job = self._jobs.get(job_id)
            if job is None:
                continue
            attempts = job.attempts + 1
            error = f"lease expired; worker {lease.worker_id} did not report back"
            if self.policy.should_retry(
                attempts=attempts, retryable=True, max_attempts=job.max_attempts
            ):
                # DELAYED, not READY. Backoff applies to a lapsed lease exactly
                # as it does to an explicit nack — a job that reliably kills its
                # worker must not be handed straight back to the next one.
                #
                # The state matters beyond tidiness: READY with a future
                # available_at is counted by stats as neither ready (not due)
                # nor delayed (wrong state), so the work would be invisible in
                # backlog and an autoscaler would see an idle queue with jobs
                # pending.
                self._jobs[job_id] = job.model_copy(
                    update={
                        "state": JobState.DELAYED,
                        "attempts": attempts,
                        "available_at": self.policy.next_attempt_at(now, attempts=attempts),
                        "updated_at": now,
                        "last_error": error,
                    }
                )
            else:
                self._jobs[job_id] = job.model_copy(
                    update={
                        "state": JobState.DEAD,
                        "attempts": attempts,
                        "updated_at": now,
                        "last_error": error,
                    }
                )
            reaped += 1
        return reaped

    async def reap_expired_leases(self) -> int:
        async with self._lock:
            return self._reap_sync(self.now())

    async def lease(
        self,
        queue: str,
        *,
        worker_id: str,
        count: int = 1,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> list[LeasedJob]:
        async with self._lock:
            now = self.now()
            # Reaping opportunistically here means at-least-once holds even with
            # no supervisor loop running. A queue that only recovers lost work
            # when a separate process remembers to ask is one deployment mistake
            # away from losing it.
            self._reap_sync(now)
            self._promote_due_sync(now)

            candidates = sorted(
                (j for j in self._jobs.values() if j.queue == queue and j.is_available(now)),
                key=lambda j: (j.priority.rank, j.available_at, j.sequence),
            )[: max(0, count)]

            leased: list[LeasedJob] = []
            for job in candidates:
                lease = Lease(
                    job_id=job.job_id,
                    lease_id=self.new_lease_id(),
                    worker_id=worker_id,
                    acquired_at=now,
                    expires_at=self.lease_deadline(lease_seconds),
                )
                updated = job.model_copy(update={"state": JobState.LEASED, "updated_at": now})
                self._jobs[job.job_id] = updated
                self._leases[job.job_id] = lease
                leased.append(LeasedJob(envelope=updated, lease=lease))
            return leased

    async def extend_lease(
        self, job_id: str, lease_id: str, *, lease_seconds: float = DEFAULT_LEASE_SECONDS
    ) -> LeasedJob:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(job_id)
            current = self._leases.get(job_id)
            self.assert_lease_current(job_id, current.lease_id if current else None, lease_id)
            assert current is not None
            renewed = current.model_copy(update={"expires_at": self.lease_deadline(lease_seconds)})
            self._leases[job_id] = renewed
            return LeasedJob(envelope=job, lease=renewed)

    async def ack(self, job_id: str, lease_id: str) -> JobEnvelope:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(job_id)
            current = self._leases.get(job_id)
            if current is None:
                raise JobNotLeased(job_id)
            self.assert_lease_current(job_id, current.lease_id, lease_id)

            now = self.now()
            done = job.model_copy(
                update={"state": JobState.DONE, "attempts": job.attempts + 1, "updated_at": now}
            )
            self._jobs[job_id] = done
            del self._leases[job_id]
            return done

    async def nack(
        self, job_id: str, lease_id: str, *, error: str, retryable: bool = True
    ) -> NackResult:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(job_id)
            current = self._leases.get(job_id)
            if current is None:
                raise JobNotLeased(job_id)
            self.assert_lease_current(job_id, current.lease_id, lease_id)

            now = self.now()
            attempts = job.attempts + 1
            del self._leases[job_id]

            if self.policy.should_retry(
                attempts=attempts, retryable=retryable, max_attempts=job.max_attempts
            ):
                retry_at = self.policy.next_attempt_at(now, attempts=attempts)
                updated = job.model_copy(
                    update={
                        "state": JobState.DELAYED,
                        "attempts": attempts,
                        "available_at": retry_at,
                        "updated_at": now,
                        "last_error": error,
                    }
                )
                self._jobs[job_id] = updated
                return NackResult(
                    outcome=NackOutcome.RETRY_SCHEDULED, envelope=updated, retry_at=retry_at
                )

            updated = job.model_copy(
                update={
                    "state": JobState.DEAD,
                    "attempts": attempts,
                    "updated_at": now,
                    "last_error": error,
                }
            )
            self._jobs[job_id] = updated
            return NackResult(outcome=NackOutcome.DEAD_LETTERED, envelope=updated, retry_at=None)

    # ---- inspection ------------------------------------------------------

    async def get(self, job_id: str) -> JobEnvelope:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(job_id)
            return job

    async def dead_letters(self, queue: str, *, limit: int = 100) -> list[JobEnvelope]:
        async with self._lock:
            dead = [
                j for j in self._jobs.values() if j.queue == queue and j.state is JobState.DEAD
            ]
            dead.sort(key=lambda j: j.updated_at)
            return dead[:limit]

    async def requeue_dead_letter(self, job_id: str) -> JobEnvelope:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(job_id)
            if job.state is not JobState.DEAD:
                raise LeaseNotHeld(job_id, "-", f"job is {job.state.value}, not dead")
            now = self.now()
            revived = job.model_copy(
                update={
                    "state": JobState.READY,
                    "attempts": 0,
                    "available_at": now,
                    "updated_at": now,
                }
            )
            self._jobs[job_id] = revived
            return revived

    async def stats(self, queue: str) -> QueueStats:
        async with self._lock:
            now = self.now()
            self._promote_due_sync(now)
            jobs = [j for j in self._jobs.values() if j.queue == queue]
            ready = [j for j in jobs if j.state is JobState.READY and now >= j.available_at]
            oldest = min((j.enqueued_at for j in ready), default=None)
            return QueueStats(
                queue=queue,
                ready=len(ready),
                delayed=sum(1 for j in jobs if j.state is JobState.DELAYED),
                leased=sum(1 for j in jobs if j.state is JobState.LEASED),
                dead=sum(1 for j in jobs if j.state is JobState.DEAD),
                oldest_ready_age_seconds=(now - oldest).total_seconds() if oldest else None,
            )

    async def purge(self, queue: str) -> int:
        async with self._lock:
            doomed = [jid for jid, j in self._jobs.items() if j.queue == queue]
            for jid in doomed:
                job = self._jobs.pop(jid)
                self._leases.pop(jid, None)
                if job.idempotency_key is not None:
                    self._by_idempotency.pop((queue, job.idempotency_key), None)
            return len(doomed)

    async def health(self) -> QueueHealth:
        started = time.perf_counter()
        try:
            async with self._lock:
                pass
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return QueueHealth(ok=False, backend=self.name, detail=type(exc).__name__)
        return QueueHealth(
            ok=True, backend=self.name, latency_ms=(time.perf_counter() - started) * 1000
        )
