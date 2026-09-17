"""Worker layer — the composition root.

The first package permitted to depend on ``contracts``, ``storage`` and
``jobqueue`` together, because turning a queued job into an ``EvidenceBundle``
from bytes in object storage is exactly the act of joining those three. No
adapter layer sits between them; one would exist only to preserve a purity that
buys nothing here.

Handlers remain queue-agnostic regardless: a ``JobHandler`` receives a
``JobRequest`` and returns a bundle. It never sees a lease, a lease id or a
queue, and it reports liveness by calling ``progress`` — the runner is what
turns that into a heartbeat.

Still no ML. Handlers arrive in the next batch with stub scoring that produces
real, contract-valid bundles.
"""

from workers.io import InMemoryResultSink, StorageMediaFetcher, StorageResultSink
from workers.protocol import JobHandler, MediaFetcher, ProgressReporter, ResultSink
from workers.runner import WorkerRunner, run_with_shutdown
from workers.types import (
    ExecutionOutcome,
    ExecutionReport,
    HandlerError,
    JobRequest,
    LeaseLost,
    PermanentError,
    ShutdownReason,
    TransientError,
    WorkerReport,
    classify,
)

__all__ = [
    "ExecutionOutcome",
    "ExecutionReport",
    "HandlerError",
    "InMemoryResultSink",
    "JobHandler",
    "JobRequest",
    "LeaseLost",
    "MediaFetcher",
    "PermanentError",
    "ProgressReporter",
    "ResultSink",
    "ShutdownReason",
    "StorageMediaFetcher",
    "StorageResultSink",
    "TransientError",
    "WorkerReport",
    "WorkerRunner",
    "classify",
    "run_with_shutdown",
]
