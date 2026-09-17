"""Job queue subsystem.

Independent by construction: standard library and pydantic only. No FastAPI, no
storage, no workers, no models, and not ``contracts``. A queue moves opaque
payloads, and keeping it that way is what makes the whole leasing lifecycle
testable with nothing else running.

**On the package name.** This is ``jobqueue`` and not ``queue`` because a
top-level ``queue`` package shadows the standard library module of that name.
The failure is not subtle once triggered but is easy to create: stdlib ``queue``
is already in ``sys.modules`` before user code runs, so ``from queue.types
import ...`` raises ``'queue' is not a package`` and the package is simply
unreachable. ``concurrent.futures.thread`` imports stdlib ``queue``, and
``asyncio.to_thread`` goes through it, so the blast radius includes anything
touching a thread pool. Same class of trap as a top-level ``datasets``
directory shadowing the HuggingFace package.

Delivery is at-least-once via leasing. See ``base.py`` for why that trade is
made deliberately, and why consumers here are naturally idempotent.

Usage::

    from jobqueue import InMemoryQueue, Priority, RetryPolicy

    q = InMemoryQueue(policy=RetryPolicy(max_attempts=3))
    await q.enqueue("image", {"sha256": "..."}, idempotency_key="...")
    leased = await q.lease("image", worker_id="w1", lease_seconds=30)
    await q.ack(leased[0].job_id, leased[0].lease.lease_id)
"""

from jobqueue.base import DEFAULT_LEASE_SECONDS, BaseQueueBackend, QueueBackend
from jobqueue.errors import (
    JobNotFound,
    JobNotLeased,
    LeaseNotHeld,
    QueueError,
    QueueUnavailable,
    UnknownQueue,
)
from jobqueue.policy import DEFAULT_POLICY, NO_RETRY, VIDEO_POLICY, RetryPolicy
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

__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "DEFAULT_POLICY",
    "NO_RETRY",
    "VIDEO_POLICY",
    "BaseQueueBackend",
    "JobEnvelope",
    "JobNotFound",
    "JobNotLeased",
    "JobState",
    "Lease",
    "LeaseNotHeld",
    "LeasedJob",
    "NackOutcome",
    "NackResult",
    "Priority",
    "QueueBackend",
    "QueueError",
    "QueueHealth",
    "QueueStats",
    "QueueUnavailable",
    "RetryPolicy",
    "UnknownQueue",
]
