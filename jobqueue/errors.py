"""Queue-domain exceptions.

No HTTP status, no ``ErrorCode``. The layer that speaks HTTP maps these; putting
the mapping here would make the queue depend on a web vocabulary it should be
able to outlive.
"""

from __future__ import annotations


class QueueError(Exception):
    """Base for everything this package raises."""


class JobNotFound(QueueError):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"no job {job_id!r}")


class LeaseNotHeld(QueueError):
    """The caller's lease is not the current one.

    Raised when a worker whose lease expired tries to ack, nack or extend a job
    that has since been re-leased to someone else. Rejecting is essential: the
    stale worker would otherwise mark another worker's in-progress job complete,
    and the second result would be silently discarded.
    """

    def __init__(self, job_id: str, lease_id: str, reason: str = "lease is stale or expired") -> None:
        self.job_id = job_id
        self.lease_id = lease_id
        super().__init__(f"lease {lease_id!r} on job {job_id!r}: {reason}")


class JobNotLeased(QueueError):
    """Acked or nacked a job nobody currently holds."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"job {job_id!r} is not currently leased")


class QueueUnavailable(QueueError):
    def __init__(self, backend: str, detail: str) -> None:
        self.backend = backend
        self.detail = detail
        super().__init__(f"{backend} unavailable: {detail}")


class UnknownQueue(QueueError):
    def __init__(self, queue: str) -> None:
        self.queue = queue
        super().__init__(f"queue {queue!r} is not registered")
