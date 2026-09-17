"""Queue value types.

Same independence rule as ``storage``: standard library and pydantic only. No
FastAPI, no workers, no models, not ``contracts``. A queue moves opaque
payloads; that they happen to describe deepfake analysis is the caller's
business.

The central design choice is **leasing rather than dequeue**. A plain
``pop()`` hands a job to a worker and forgets it, so a worker that dies
mid-video takes the job with it. A lease is a time-limited claim: if the worker
does not acknowledge or renew before it expires, the job returns to the queue
and someone else takes it. That is the difference between at-most-once and
at-least-once delivery, and for a 60-second video job on a spot instance it is
not a theoretical distinction.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobState(StrEnum):
    """Where a job sits in the queue.

    Distinct from the *application's* job state (queued/running/succeeded).
    This describes delivery, not work: a job can be ``READY`` here while the
    application considers it queued, and ``DEAD`` here while the application
    reports failed. Conflating the two is how a retry silently resurrects a job
    the application already told a user had finished.
    """

    READY = "ready"
    DELAYED = "delayed"
    LEASED = "leased"
    DONE = "done"
    DEAD = "dead"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.DONE, JobState.DEAD)


class Priority(StrEnum):
    """Coarse ordering. Three bands, not an integer.

    An integer priority invites callers to invent 7, then 8, then 9 to jump the
    queue, and the numbers stop meaning anything. Three named bands force the
    question "is this genuinely more urgent than everything at HIGH".
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {"high": 0, "normal": 1, "low": 2}[self.value]


class JobEnvelope(Base):
    """One unit of work, with everything the queue needs and nothing it does not.

    ``payload`` is an opaque dict. The queue never inspects it — that is what
    keeps this package reusable and free of a dependency on the detection
    contracts.
    """

    job_id: str
    queue: str = Field(min_length=1, description="Logical queue name, e.g. 'image', 'video'.")
    payload: dict[str, Any] = Field(description="Opaque to the queue.")
    state: JobState = JobState.READY
    priority: Priority = Priority.NORMAL
    sequence: int = Field(
        default=0,
        ge=0,
        description=(
            "Monotonic enqueue counter, assigned by the backend. Exists to make "
            "FIFO among equal-priority jobs a total order: two jobs enqueued in "
            "the same microsecond have identical enqueued_at, so timestamps "
            "alone leave the tie to be broken arbitrarily by dict ordering."
        ),
    )

    attempts: int = Field(default=0, ge=0, description="Completed attempts, not including the current one.")
    max_attempts: int = Field(default=3, ge=1)

    enqueued_at: datetime
    available_at: datetime = Field(description="Not leasable before this. Drives delay and backoff.")
    updated_at: datetime

    idempotency_key: str | None = Field(
        default=None,
        description=(
            "Enqueueing the same key twice returns the original job rather than "
            "duplicating work. The same viral clip arriving from fifty people is "
            "one unit of work."
        ),
    )
    last_error: str | None = Field(default=None, description="Why the previous attempt failed.")

    @property
    def attempts_remaining(self) -> int:
        return max(0, self.max_attempts - self.attempts)

    def is_available(self, at: datetime) -> bool:
        return self.state is JobState.READY and at >= self.available_at


class Lease(Base):
    """A time-limited claim on a job.

    ``lease_id`` is not decorative. Without it, a worker whose lease expired —
    but which is still running, unaware — could acknowledge a job that a second
    worker has since taken and is midway through. Every ack, nack and extend is
    checked against the current lease id, so a stale worker's call is rejected
    rather than corrupting another's work.
    """

    job_id: str
    lease_id: str
    worker_id: str
    acquired_at: datetime
    expires_at: datetime

    def is_expired(self, at: datetime) -> bool:
        return at >= self.expires_at

    def remaining(self, at: datetime) -> timedelta:
        return max(timedelta(0), self.expires_at - at)


class LeasedJob(Base):
    """What a worker receives: the work, and its claim on it."""

    envelope: JobEnvelope
    lease: Lease

    @property
    def job_id(self) -> str:
        return self.envelope.job_id

    @property
    def payload(self) -> dict[str, Any]:
        return self.envelope.payload


class NackOutcome(StrEnum):
    """What happened to a job the worker could not complete."""

    RETRY_SCHEDULED = "retry_scheduled"
    DEAD_LETTERED = "dead_lettered"


class NackResult(Base):
    outcome: NackOutcome
    envelope: JobEnvelope
    retry_at: datetime | None = Field(
        default=None, description="When it becomes leasable again. None if dead-lettered."
    )


class QueueStats(Base):
    """Depth by state. What an autoscaler and a dashboard both read."""

    queue: str
    ready: int = Field(default=0, ge=0)
    delayed: int = Field(default=0, ge=0, description="Waiting on backoff or an explicit delay.")
    leased: int = Field(default=0, ge=0, description="Claimed by a worker right now.")
    dead: int = Field(default=0, ge=0)
    oldest_ready_age_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Age of the longest-waiting ready job, or null when none are ready. "
            "A better autoscaling signal than depth: ten jobs enqueued a second "
            "ago is healthy, one job waiting five minutes is not. Belongs here "
            "rather than in QueueHealth because it describes the workload, not "
            "the backend's condition — a queue can be perfectly healthy and "
            "hopelessly behind."
        ),
    )

    @property
    def backlog(self) -> int:
        return self.ready + self.delayed


class QueueHealth(Base):
    """Mirrors ``StorageHealth`` and ``StoreHealth`` deliberately — an operator
    should not have to learn three vocabularies for the same question."""

    ok: bool
    backend: str
    latency_ms: float | None = Field(default=None, ge=0.0)
    detail: str | None = None
