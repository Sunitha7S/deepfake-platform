"""Worker assembly.

The one place that wires a queue, a storage backend and a handler into a
runnable worker. Everything else depends on protocols, so swapping the in-memory
queue for Redis or the local store for S3 changes this module and nothing else.

Lease durations differ per queue by design. An image job finishes in seconds and
a 30-second lease is generous; a video job runs for minutes, so its lease starts
longer and the heartbeat extends it. Giving both the same lease would either
have video jobs reclaimed mid-flight or leave a dead image worker's job stalled
for minutes.
"""

from __future__ import annotations

from contracts.enums import MediaKind
from jobqueue.base import QueueBackend
from jobqueue.registry import IMAGE_QUEUE, VIDEO_QUEUE
from storage.base import StorageBackend
from workers.image import ImageHandler
from workers.video import VideoHandler
from workers.io import StorageMediaFetcher, StorageResultSink
from workers.protocol import JobHandler, MediaFetcher, ResultSink
from workers.runner import WorkerRunner

LEASE_SECONDS: dict[str, float] = {
    IMAGE_QUEUE: 30.0,
    VIDEO_QUEUE: 120.0,
}

DRAIN_TIMEOUT_S: dict[str, float] = {
    # How long shutdown waits for in-flight work before returning it to the
    # queue. Roughly a job's expected duration: waiting much longer delays a
    # deploy, waiting much less throws away nearly-finished work.
    IMAGE_QUEUE: 10.0,
    VIDEO_QUEUE: 60.0,
}


def handler_for(media_kind: MediaKind | str) -> JobHandler:
    """The handler for a media kind, backed by the inference layer.

    Both delegate to ``inference`` and contain no scoring logic of their own.
    Swapping stub weights for real ones is a change inside ``inference`` and is
    invisible here.
    """
    kind = MediaKind(media_kind)
    return ImageHandler() if kind is MediaKind.IMAGE else VideoHandler()


def queue_for(media_kind: MediaKind | str) -> str:
    return IMAGE_QUEUE if MediaKind(media_kind) is MediaKind.IMAGE else VIDEO_QUEUE


def build_worker(
    media_kind: MediaKind | str,
    *,
    queue: QueueBackend,
    storage: StorageBackend,
    worker_id: str | None = None,
    handler: JobHandler | None = None,
    fetcher: MediaFetcher | None = None,
    sink: ResultSink | None = None,
) -> WorkerRunner:
    """Assemble a worker for one media kind.

    ``handler``, ``fetcher`` and ``sink`` are injectable so a test can substitute
    one without rebuilding the wiring — and so a real model can replace the stub
    without touching this function.
    """
    kind = MediaKind(media_kind)
    queue_name = queue_for(kind)
    return WorkerRunner(
        queue=queue,
        queue_name=queue_name,
        handler=handler or handler_for(kind),
        fetcher=fetcher or StorageMediaFetcher(storage),
        sink=sink or StorageResultSink(storage),
        worker_id=worker_id,
        lease_seconds=LEASE_SECONDS[queue_name],
        drain_timeout_s=DRAIN_TIMEOUT_S[queue_name],
    )


def build_all_workers(
    *, queue: QueueBackend, storage: StorageBackend
) -> dict[str, WorkerRunner]:
    """One worker per queue. What a supervisor process starts."""
    return {
        queue_for(kind): build_worker(kind, queue=queue, storage=storage)
        for kind in (MediaKind.IMAGE, MediaKind.VIDEO)
    }
