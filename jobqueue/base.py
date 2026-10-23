"""The ``QueueBackend`` protocol and shared leasing rules.

Delivery is **at-least-once**. A worker leases a job, does the work, and acks;
if it dies, the lease expires and the job returns. The alternative — deleting on
dequeue — is at-most-once and loses work whenever a process dies, which on a
spot instance is routine rather than exceptional.

At-least-once means a job can run twice. That is a real cost and it is paid
deliberately: the alternative is losing jobs silently. Consumers must therefore
be idempotent, which for this platform they naturally are — analysis is a pure
function of the input bytes and the model version, and the result is written
under a content-addressed key. Running it twice wastes compute and changes
nothing else.

``extend_lease`` exists for video. A 60-second job under a 30-second lease would
be reclaimed and re-run while the first worker is still going, forever, and the
queue would look busy while completing nothing. Long consumers heartbeat.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from jobqueue.errors import LeaseNotHeld
from jobqueue.policy import RetryPolicy
from jobqueue.types import (
    JobEnvelope,
    LeasedJob,
    NackResult,
    Priority,
    QueueHealth,
    QueueStats,
)

DEFAULT_LEASE_SECONDS = 60


@runtime_checkable
class QueueBackend(Protocol):
    """Everything the platform needs from a job queue.

    Every method is async: Redis is async, and a synchronous protocol now would
    force every call site to change when it arrives.
    """

    name: str

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
        """Add work. Returns the existing job if ``idempotency_key`` is already present.

        ``job_id`` lets the caller supply the handle instead of having one
        generated. The API passes the contract's ``JobSpec.job_id`` so that one
        identifier addresses a job in the queue, in the result store and in the
        URL a user holds. Two ids for one job is not a naming inconvenience: it
        silently breaks any lookup that guesses wrong, and the only symptom is
        work being redone.

        Note the interaction with idempotency: when a duplicate is detected the
        *original* envelope is returned, so a caller supplying a fresh
        ``job_id`` must use the id it gets back rather than the one it sent.
        """
        ...

    # ---- consuming -------------------------------------------------------

    async def lease(
        self,
        queue: str,
        *,
        worker_id: str,
        count: int = 1,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> list[LeasedJob]:
        """Claim up to ``count`` available jobs. Returns [] rather than blocking.

        Non-blocking by design: a worker loop that polls is trivial to reason
        about and to shut down, where a blocking pop has to be interrupted.
        """
        ...

    async def extend_lease(
        self, job_id: str, lease_id: str, *, lease_seconds: float = DEFAULT_LEASE_SECONDS
    ) -> LeasedJob:
        """Heartbeat. Raises ``LeaseNotHeld`` if the lease is stale."""
        ...

    async def ack(self, job_id: str, lease_id: str) -> JobEnvelope:
        """Mark complete. Raises ``LeaseNotHeld`` if the lease is stale."""
        ...

    async def nack(
        self, job_id: str, lease_id: str, *, error: str, retryable: bool = True
    ) -> NackResult:
        """Report failure. Retries with backoff, or dead-letters.

        ``retryable`` comes from the caller because only the caller can tell a
        transient failure from a permanent one.
        """
        ...

    # ---- inspection and maintenance --------------------------------------

    async def get(self, job_id: str) -> JobEnvelope:
        """Raises ``JobNotFound``."""
        ...

    async def reap_expired_leases(self) -> int:
        """Return jobs whose lease lapsed to READY. Returns how many.

        The mechanism that makes at-least-once true. Must be called
        periodically — a supervisor loop, or opportunistically inside ``lease``.
        """
        ...

    async def dead_letters(self, queue: str, *, limit: int = 100) -> list[JobEnvelope]:
        """Jobs that exhausted their attempts or failed permanently."""
        ...

    async def requeue_dead_letter(self, job_id: str) -> JobEnvelope:
        """Return one dead job to READY with its attempt count reset.

        The operator's undo. A dead-letter queue nobody can drain is a list of
        problems, not a recovery mechanism.
        """
        ...

    async def stats(self, queue: str) -> QueueStats: ...

    async def purge(self, queue: str) -> int:
        """Remove every job. Returns the count. Tests and operator recovery."""
        ...

    async def health(self) -> QueueHealth:
        """Probe the backend. Must never raise."""
        ...


class BaseQueueBackend:
    """Leasing rules shared by every backend.

    Not required by the protocol — implementations conform structurally — but
    it keeps lease identity and expiry from being re-derived, slightly
    differently, in each backend.
    """

    name: str = "base"

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        *,
        queue_policies: Mapping[str, RetryPolicy] | None = None,
    ) -> None:
        self.policy = policy or RetryPolicy()
        self.queue_policies: dict[str, RetryPolicy] = dict(queue_policies or {})

    def policy_for_queue(self, queue: str) -> RetryPolicy:
        """The retry policy that governs jobs on ``queue``.

        A per-queue entry wins; ``policy`` is the fallback for any queue without
        one. Every retry decision -- the default attempt budget at enqueue, the
        backoff after a nack, the backoff after a lapsed lease -- goes through
        here, so a backend cannot apply the per-queue budget in one place and
        the fallback in another.
        """
        return self.queue_policies.get(queue, self.policy)

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def new_lease_id() -> str:
        return f"l_{secrets.token_urlsafe(12)}"

    @staticmethod
    def new_job_id() -> str:
        return f"j_{secrets.token_urlsafe(12)}"

    def lease_deadline(self, seconds: float) -> datetime:
        """Compute a lease expiry.

        Rejects a non-positive duration rather than clamping it. Silently
        turning 0 into 1 hides the caller's mistake and produces a lease whose
        length nobody asked for; a lease that is already expired at issue is a
        bug worth surfacing at the call site.

        Sub-second leases are permitted. They are meaningless in production but
        let a conformance suite exercise expiry without sleeping for a second
        per check.
        """
        if seconds <= 0:
            raise ValueError(f"lease_seconds must be positive, got {seconds}")
        return self.now() + timedelta(seconds=seconds)

    def assert_lease_current(self, job_id: str, held: str | None, presented: str) -> None:
        """Reject a stale worker before it can affect another's job."""
        if held is None:
            raise LeaseNotHeld(job_id, presented, "job is not leased")
        if held != presented:
            raise LeaseNotHeld(job_id, presented, "job has been re-leased to another worker")
