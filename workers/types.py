"""Worker value types and the failure taxonomy.

The worker layer is the composition root: it is the first package permitted to
depend on ``contracts``, ``storage`` and ``jobqueue`` together, because
producing an ``EvidenceBundle`` from bytes in object storage in response to a
queued job is precisely the act of joining those three.

The failure taxonomy below is the heart of the layer. Everything the runner does
after a handler returns or raises follows from which of these it is.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from contracts.job import JobSpec


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# --------------------------------------------------------------------------
# Failure taxonomy
# --------------------------------------------------------------------------


class HandlerError(Exception):
    """Base for failures a handler reports deliberately.

    ``retryable`` is the only thing the runner needs from an exception, and it
    is a property of the *cause*, not of the caller's mood: a corrupt file is
    permanently corrupt, and a storage timeout is not permanently timed out.
    """

    retryable: bool = True

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        self.detail = detail
        super().__init__(message)


class TransientError(HandlerError):
    """Worth another attempt: storage blip, timeout, resource exhaustion."""

    retryable = True


class PermanentError(HandlerError):
    """Will fail identically next time: corrupt container, unsupported codec.

    Dead-lettered immediately. Retrying it produces the same failure three times
    and consumes capacity that healthy jobs need.
    """

    retryable = False


class LeaseLost(Exception):
    """The lease was reclaimed while this worker was still running the job.

    Neither an ack nor a nack may follow. The job now belongs to another worker
    and is very likely in flight there; acking would mark *their* work complete
    and throw away the real result, while nacking would schedule a retry of a
    job already being retried.

    The correct response is to abandon the work silently and let the other
    worker finish.
    """

    def __init__(self, job_id: str, lease_id: str) -> None:
        self.job_id = job_id
        self.lease_id = lease_id
        super().__init__(f"lease {lease_id!r} on job {job_id!r} was lost mid-execution")


def classify(exc: BaseException) -> bool:
    """Return whether ``exc`` should be retried.

    An unclassified exception is treated as **transient**. An unexpected crash
    is more often a blip or a bug in an edge case than a permanent property of
    the input, and ``max_attempts`` bounds the cost of being wrong. The opposite
    default would dead-letter real work on the first unhandled ``KeyError``.
    """
    if isinstance(exc, HandlerError):
        return exc.retryable
    return True


# --------------------------------------------------------------------------
# Handler interface types
# --------------------------------------------------------------------------


class JobRequest(Base):
    """Everything a handler gets. Payload in, result out.

    Deliberately contains no lease, no queue reference and no storage key: a
    handler that cannot reach the queue cannot accidentally couple itself to it.
    """

    spec: JobSpec
    media: bytes = Field(description="The bytes to analyse, already fetched and verified.")

    @property
    def job_id(self) -> str:
        return str(self.spec.job_id)

    def __len__(self) -> int:
        return len(self.media)


class ExecutionOutcome(StrEnum):
    """What the runner did with one leased job."""

    COMPLETED = "completed"
    ALREADY_DONE = "already_done"
    RETRY_SCHEDULED = "retry_scheduled"
    DEAD_LETTERED = "dead_lettered"
    LEASE_LOST = "lease_lost"
    ABANDONED_SHUTDOWN = "abandoned_shutdown"


class ExecutionReport(Base):
    """The result of processing one job. Returned for tests and metrics."""

    job_id: str
    outcome: ExecutionOutcome
    attempts: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    heartbeats: int = Field(default=0, ge=0)
    error: str | None = None
    verdict: str | None = Field(
        default=None, description="Set when a bundle was produced, for at-a-glance logs."
    )


class ShutdownReason(StrEnum):
    STOP_REQUESTED = "stop_requested"
    IDLE_LIMIT = "idle_limit"
    JOB_LIMIT = "job_limit"


class WorkerReport(Base):
    """Summary of a worker's whole life. What a supervisor logs on exit."""

    worker_id: str
    queue: str
    reason: ShutdownReason
    started_at: datetime
    stopped_at: datetime
    leased: int = Field(default=0, ge=0)
    completed: int = Field(default=0, ge=0)
    already_done: int = Field(default=0, ge=0)
    retried: int = Field(default=0, ge=0)
    dead_lettered: int = Field(default=0, ge=0)
    lease_lost: int = Field(default=0, ge=0)
    abandoned: int = Field(
        default=0, ge=0, description="In-flight at shutdown and returned to the queue."
    )

    @property
    def processed(self) -> int:
        return self.completed + self.already_done
