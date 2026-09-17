"""Worker layer tests.

Three groups:

* handler conformance, run against both stub handlers;
* runner behaviour — classification, ack/nack, progress, shutdown;
* **crash recovery and concurrency**, which are the reason the lease and the
  idempotent sink exist at all.

Runs standalone and under pytest, like every other suite here.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile

import numpy as np
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.enums import MediaKind, Verdict  # noqa: E402
from contracts.evidence import EvidenceBundle  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from jobqueue.memory import InMemoryQueue  # noqa: E402
from jobqueue.registry import IMAGE_QUEUE, VIDEO_QUEUE  # noqa: E402
from jobqueue.types import JobState  # noqa: E402
from storage import HmacUrlSigner  # noqa: E402
from storage.local import LocalStorageBackend  # noqa: E402
from tests.stub_handlers import (  # noqa: E402
    PROGRESS_BATCH,
    VIDEO_FRAMES_SAMPLED,
    StubImageHandler,
    StubVideoHandler,
)
from workers.io import InMemoryResultSink, StorageMediaFetcher, StorageResultSink  # noqa: E402
from workers.registry import build_all_workers, build_worker  # noqa: E402
from workers.runner import WorkerRunner  # noqa: E402
from workers.types import (  # noqa: E402
    ExecutionOutcome,
    JobRequest,
    PermanentError,
    ShutdownReason,
    TransientError,
    classify,
)
from tests.worker_conformance import RecordingProgress, run_conformance  # noqa: E402

SECRET = "worker-test-signing-secret-0123456789"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _spec(data: bytes, *, kind: MediaKind = MediaKind.IMAGE, job_id: UUID | None = None) -> JobSpec:
    return JobSpec(
        job_id=job_id or uuid4(),
        media_kind=kind,
        storage_key=f"uploads/2026/08/30/{hashlib.sha256(data).hexdigest()[:16]}",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="image/jpeg" if kind is MediaKind.IMAGE else "video/mp4",
        model_version="stub-0",
        idempotency_key=f"{hashlib.sha256(data).hexdigest()}:stub-0",
        submitted_at=datetime.now(UTC),
    )


def _request(data: bytes, *, kind: MediaKind = MediaKind.IMAGE) -> JobRequest:
    return JobRequest(spec=_spec(data, kind=kind), media=data)


class DictFetcher:
    """Serves bytes by storage key. Stands in for object storage."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.calls = 0

    def add(self, spec: JobSpec, data: bytes) -> None:
        self.objects[spec.storage_key] = data

    async def fetch(self, spec: JobSpec) -> bytes:
        self.calls += 1
        try:
            return self.objects[spec.storage_key]
        except KeyError as exc:
            raise PermanentError("missing object") from exc


class CountingHandler:
    """Wraps a handler and counts invocations, for crash-recovery assertions."""

    media_kind = MediaKind.IMAGE.value
    name = "counting"

    def __init__(self, inner=None) -> None:
        self.inner = inner or StubImageHandler()
        self.calls = 0

    async def handle(self, request, progress):
        self.calls += 1
        return await self.inner.handle(request, progress)


class HangingHandler:
    """Never finishes. Simulates a worker wedged mid-job."""

    media_kind = MediaKind.IMAGE.value
    name = "hanging"

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def handle(self, request, progress):
        self.started.set()
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


class ExplodingHandler:
    media_kind = MediaKind.IMAGE.value
    name = "exploding"

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def handle(self, request, progress):
        raise self.exc


async def _enqueue(queue, spec: JobSpec, name: str = IMAGE_QUEUE):
    return await queue.enqueue(name, spec.model_dump(mode="json"))


def _runner(queue, handler, fetcher, sink, **kw) -> WorkerRunner:
    kw.setdefault("lease_seconds", 5.0)
    kw.setdefault("poll_interval_s", 0.02)
    return WorkerRunner(
        queue=queue,
        queue_name=IMAGE_QUEUE,
        handler=handler,
        fetcher=fetcher,
        sink=sink,
        **kw,
    )


# --------------------------------------------------------------------------
# Handler conformance
# --------------------------------------------------------------------------


def test_image_handler_passes_conformance():
    failures = asyncio.run(
        run_conformance(StubImageHandler, lambda d: _request(d, kind=MediaKind.IMAGE))
    )
    assert failures == [], "image handler:\n  " + "\n  ".join(failures)


def test_video_handler_passes_conformance():
    failures = asyncio.run(
        run_conformance(StubVideoHandler, lambda d: _request(d, kind=MediaKind.VIDEO))
    )
    assert failures == [], "video handler:\n  " + "\n  ".join(failures)


def test_conformance_catches_a_nondeterministic_handler():
    """The suite must have teeth: determinism is what the recovery tests rest on."""
    import random

    class Flaky(StubImageHandler):
        async def handle(self, request, progress):
            bundle = await super().handle(request, progress)
            if bundle.p_fake is None:
                return bundle
            jitter = min(0.99, max(0.01, bundle.p_fake + random.random() * 0.1))
            return bundle.model_copy(
                update={
                    "p_fake": jitter,
                    "verdict": Verdict.MANIPULATED
                    if jitter >= bundle.threshold.value
                    else Verdict.AUTHENTIC,
                    "interval": (max(0.0, jitter - 0.1), min(1.0, jitter + 0.1)),
                }
            )

    failures = asyncio.run(run_conformance(Flaky, lambda d: _request(d)))
    assert any("same_bytes" in f for f in failures), failures


def test_conformance_catches_per_frame_progress():
    """Requirement: video handlers batch progress rather than calling per frame."""

    class Chatty(StubVideoHandler):
        def __init__(self):
            super().__init__(batch=1)

    failures = asyncio.run(
        run_conformance(Chatty, lambda d: _request(d, kind=MediaKind.VIDEO))
    )
    assert any("batched" in f for f in failures), failures


def test_video_progress_is_batched():
    request = _request(b"video payload", kind=MediaKind.VIDEO)
    progress = RecordingProgress()
    bundle = asyncio.run(StubVideoHandler().handle(request, progress))
    assert bundle.video.frames_sampled == VIDEO_FRAMES_SAMPLED
    expected = VIDEO_FRAMES_SAMPLED // PROGRESS_BATCH + 2  # sampling + temporal
    assert len(progress) == expected, f"{len(progress)} calls, expected {expected}"
    assert len(progress) < VIDEO_FRAMES_SAMPLED


def test_handlers_reject_the_wrong_media_kind():
    """A video routed to the image queue is a permanent error, not a retry."""
    try:
        asyncio.run(
            StubImageHandler().handle(
                _request(b"x", kind=MediaKind.VIDEO), RecordingProgress()
            )
        )
    except PermanentError:
        return
    raise AssertionError("expected PermanentError")


def test_handlers_reject_empty_media():
    """A spec declaring bytes that did not arrive is permanently wrong.

    ``JobSpec.size_bytes`` has a floor of 1, so the contract already makes a
    zero-byte *declaration* unrepresentable. The reachable case is a spec that
    claims bytes and media that is empty, which is what this builds.
    """
    request = JobRequest(spec=_spec(b"x"), media=b"")
    try:
        asyncio.run(StubImageHandler().handle(request, RecordingProgress()))
    except PermanentError:
        return
    raise AssertionError("expected PermanentError")


def test_stub_model_version_is_labelled_as_a_stub():
    """Nothing downstream may mistake a stub score for a measurement."""
    bundle = asyncio.run(StubImageHandler().handle(_request(b"label"), RecordingProgress()))
    assert "stub" in bundle.model.version


# --------------------------------------------------------------------------
# Runner: the happy path and classification
# --------------------------------------------------------------------------


def test_successful_job_persists_then_acks():
    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(b"happy path")
        fetch.add(spec, b"happy path")
        await _enqueue(q, spec)
        report = await _runner(q, StubImageHandler(), fetch, sink).run(
            max_jobs=1, idle_timeout_s=0.1
        )
        job = next(iter(q._jobs.values()))
        return report, sink, job

    report, sink, job = asyncio.run(scenario())
    assert report.completed == 1
    assert len(sink) == 1
    assert job.state is JobState.DONE


def test_permanent_error_dead_letters_without_retrying():
    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(b"corrupt")
        fetch.add(spec, b"corrupt")
        await _enqueue(q, spec)
        handler = ExplodingHandler(PermanentError("unreadable container"))
        report = await _runner(q, handler, fetch, sink).run(max_jobs=1, idle_timeout_s=0.1)
        return report, await q.dead_letters(IMAGE_QUEUE)

    report, dead = asyncio.run(scenario())
    assert report.dead_lettered == 1
    assert len(dead) == 1
    assert "unreadable container" in (dead[0].last_error or "")


def test_transient_error_schedules_a_retry():
    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(b"blip")
        fetch.add(spec, b"blip")
        await _enqueue(q, spec)
        handler = ExplodingHandler(TransientError("storage hiccup"))
        report = await _runner(q, handler, fetch, sink).run(max_jobs=1, idle_timeout_s=0.1)
        return report, next(iter(q._jobs.values()))

    report, job = asyncio.run(scenario())
    assert report.retried == 1
    assert job.state is JobState.DELAYED
    assert job.attempts == 1


def test_unclassified_exception_is_treated_as_transient():
    """A bug in an edge case should not dead-letter real work on first sight."""
    assert classify(KeyError("boom")) is True
    assert classify(PermanentError("no")) is False
    assert classify(TransientError("yes")) is True


def test_malformed_payload_dead_letters():
    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        await q.enqueue(IMAGE_QUEUE, {"not": "a job spec"})
        report = await _runner(q, StubImageHandler(), fetch, sink).run(
            max_jobs=1, idle_timeout_s=0.1
        )
        return report, await q.dead_letters(IMAGE_QUEUE)

    report, dead = asyncio.run(scenario())
    assert report.dead_lettered == 1
    assert len(dead) == 1


def test_insufficient_evidence_is_acked_as_success():
    """Requirement 4: an unjudgeable input is a completed job, not a failure."""

    class AlwaysAbstains(StubImageHandler):
        async def handle(self, request, progress):
            bundle = await super().handle(request, progress)
            from contracts.enums import AbstainReason
            from contracts.common import QualityReport
            from contracts.enums import ResolutionClass

            return bundle.model_copy(
                update={
                    "verdict": Verdict.INSUFFICIENT_EVIDENCE,
                    "p_fake": None,
                    "interval": None,
                    "abstain_reason": AbstainReason.FACE_TOO_SMALL,
                    "quality": QualityReport(
                        face_px=20,
                        resolution_class=ResolutionClass.TINY,
                        gate_passed=False,
                        gate_reason=AbstainReason.FACE_TOO_SMALL,
                    ),
                }
            )

    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(b"tiny face")
        fetch.add(spec, b"tiny face")
        await _enqueue(q, spec)
        report = await _runner(q, AlwaysAbstains(), fetch, sink).run(
            max_jobs=1, idle_timeout_s=0.1
        )
        stored = await sink.find(str(spec.job_id))
        return report, stored, next(iter(q._jobs.values()))

    report, stored, job = asyncio.run(scenario())
    assert report.completed == 1, "abstention must complete, not fail"
    assert job.state is JobState.DONE
    assert stored is not None and stored.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert await_none(stored.p_fake)


def await_none(value) -> bool:
    return value is None


# --------------------------------------------------------------------------
# Crash recovery
# --------------------------------------------------------------------------


def test_worker_death_mid_job_returns_the_work():
    """A wedged worker must not take the job with it."""

    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(b"interrupted")
        fetch.add(spec, b"interrupted")
        await _enqueue(q, spec)

        hanging = HangingHandler()
        doomed = _runner(q, hanging, fetch, sink, lease_seconds=0.1, worker_id="doomed")
        task = asyncio.create_task(doomed.run(max_jobs=1, idle_timeout_s=1.0))
        await asyncio.wait_for(hanging.started.wait(), timeout=2)

        # The process dies: no ack, no nack, nothing tidied up.
        task.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await task

        await asyncio.sleep(0.2)
        await q.reap_expired_leases()

        job = next(iter(q._jobs.values()))
        return job, len(sink)

    job, results = asyncio.run(scenario())
    assert job.state is not JobState.LEASED, "the job is still held by a dead worker"
    assert job.attempts == 1, "lease expiry counts as a failed attempt"
    assert results == 0, "nothing should have been persisted"


def test_second_worker_completes_what_the_first_abandoned():
    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(b"handover")
        fetch.add(spec, b"handover")
        await _enqueue(q, spec)

        hanging = HangingHandler()
        doomed = _runner(q, hanging, fetch, sink, lease_seconds=0.1, worker_id="doomed")
        task = asyncio.create_task(doomed.run(max_jobs=1, idle_timeout_s=1.0))
        await asyncio.wait_for(hanging.started.wait(), timeout=2)
        task.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await task

        await asyncio.sleep(0.2)
        await q.reap_expired_leases()
        # Retry backoff applies; make it available for the successor.
        job = next(iter(q._jobs.values()))
        q._jobs[job.job_id] = job.model_copy(
            update={"available_at": job.enqueued_at, "state": JobState.READY}
        )

        survivor = _runner(q, StubImageHandler(), fetch, sink, worker_id="survivor")
        report = await survivor.run(max_jobs=1, idle_timeout_s=0.3)
        return report, sink, next(iter(q._jobs.values()))

    report, sink, job = asyncio.run(scenario())
    assert report.completed == 1
    assert len(sink) == 1
    assert job.state is JobState.DONE


def test_crash_after_persist_before_ack_does_not_rerun_the_handler():
    """The expensive half already succeeded; redoing it wastes GPU minutes."""

    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(b"persisted already")
        fetch.add(spec, b"persisted already")

        # Simulate the first attempt: result written, process died before ack.
        first = await StubImageHandler().handle(
            JobRequest(spec=spec, media=b"persisted already"), RecordingProgress()
        )
        await sink.publish(first)

        await _enqueue(q, spec)
        counting = CountingHandler()
        report = await _runner(q, counting, fetch, sink).run(max_jobs=1, idle_timeout_s=0.2)
        return report, counting, sink, first, next(iter(q._jobs.values()))

    report, counting, sink, first, job = asyncio.run(scenario())
    assert report.already_done == 1
    assert counting.calls == 0, "the handler was re-run despite an existing result"
    assert len(sink) == 1
    assert job.state is JobState.DONE
    stored = asyncio.run(sink.find(str(first.job_id)))
    assert stored == first, "the original result must survive untouched"


def test_result_sink_is_idempotent():
    """Requirement 3: one final result per job, first writer wins."""

    async def scenario():
        sink = InMemoryResultSink()
        spec = _spec(b"idempotent")
        first = await StubImageHandler().handle(
            JobRequest(spec=spec, media=b"idempotent"), RecordingProgress()
        )
        second = first.model_copy(update={"disagreement": 0.999})
        a, was_a = await sink.publish(first)
        b, was_b = await sink.publish(second)
        return a, was_a, b, was_b, len(sink)

    a, was_a, b, was_b, count = asyncio.run(scenario())
    assert was_a is False and was_b is True
    assert b == a, "a second publish must return the original, not overwrite it"
    assert count == 1


def test_storage_result_sink_is_idempotent_and_durable():
    """Same contract over real storage, and it survives a fresh sink object."""

    async def scenario():
        root = Path(tempfile.mkdtemp(prefix="dfp-worker-"))
        storage = LocalStorageBackend(root, signer=HmacUrlSigner(SECRET))
        sink = StorageResultSink(storage)
        spec = _spec(b"durable")
        bundle = await StubImageHandler().handle(
            JobRequest(spec=spec, media=b"durable"), RecordingProgress()
        )
        _, was_new = await sink.publish(bundle)
        _, was_dup = await sink.publish(bundle.model_copy(update={"disagreement": 0.5}))

        reborn = StorageResultSink(LocalStorageBackend(root, signer=HmacUrlSigner(SECRET)))
        found = await reborn.find(str(spec.job_id))
        return was_new, was_dup, found, bundle

    was_new, was_dup, found, bundle = asyncio.run(scenario())
    assert was_new is False
    assert was_dup is True
    assert found is not None, "the result did not survive a new sink instance"
    assert found.p_fake == bundle.p_fake
    assert found.disagreement == bundle.disagreement, "the duplicate overwrote the original"


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


def test_two_workers_cannot_both_process_the_same_job():
    """Requirement 1. One job, two workers racing: exactly one may complete."""

    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(b"contended")
        fetch.add(spec, b"contended")
        await _enqueue(q, spec)

        handlers = [CountingHandler(), CountingHandler()]
        workers = [
            _runner(q, h, fetch, sink, worker_id=f"w{i}")
            for i, h in enumerate(handlers)
        ]
        reports = await asyncio.gather(
            *(w.run(max_jobs=1, idle_timeout_s=0.3) for w in workers)
        )
        return reports, handlers, sink, next(iter(q._jobs.values()))

    reports, handlers, sink, job = asyncio.run(scenario())
    completed = sum(r.completed for r in reports)
    leased = sum(r.leased for r in reports)
    assert leased == 1, f"the job was leased {leased} times; it must be leased once"
    assert completed == 1, f"{completed} workers completed the same job"
    assert sum(h.calls for h in handlers) == 1, "the handler ran more than once"
    assert len(sink) == 1, "two results were stored for one job"
    assert job.state is JobState.DONE


def test_many_workers_split_the_backlog_without_duplication():
    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        for i in range(12):
            data = f"job-{i}".encode()
            spec = _spec(data)
            fetch.add(spec, data)
            await _enqueue(q, spec)

        handlers = [CountingHandler() for _ in range(4)]
        workers = [
            _runner(q, h, fetch, sink, worker_id=f"w{i}") for i, h in enumerate(handlers)
        ]
        reports = await asyncio.gather(
            *(w.run(idle_timeout_s=0.25) for w in workers)
        )
        return reports, handlers, sink

    reports, handlers, sink = asyncio.run(scenario())
    assert sum(r.completed for r in reports) == 12
    assert sum(h.calls for h in handlers) == 12, "a job was processed twice"
    assert len(sink) == 12


def test_stale_worker_cannot_ack_a_stolen_job():
    """The lease-ownership rule, exercised end to end.

    The first worker's lease is reclaimed and re-leased while it is still
    running. It must abandon the job rather than ack it — acking would mark the
    second worker's in-flight work complete and discard the real result.
    """

    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(b"stolen")
        fetch.add(spec, b"stolen")
        await _enqueue(q, spec)

        [first] = await q.lease(IMAGE_QUEUE, worker_id="slow", lease_seconds=0.1)
        await asyncio.sleep(0.15)
        await q.reap_expired_leases()

        job = next(iter(q._jobs.values()))
        q._jobs[job.job_id] = job.model_copy(
            update={"available_at": job.enqueued_at, "state": JobState.READY}
        )
        [second] = await q.lease(IMAGE_QUEUE, worker_id="fast", lease_seconds=30)

        runner = _runner(q, StubImageHandler(), fetch, sink, worker_id="slow")
        report = await runner.process(first)
        return report, second, next(iter(q._jobs.values()))

    report, second, job = asyncio.run(scenario())
    assert report.outcome is ExecutionOutcome.LEASE_LOST
    assert job.state is JobState.LEASED, "the second worker's lease was disturbed"


# --------------------------------------------------------------------------
# Shutdown
# --------------------------------------------------------------------------


def test_stop_between_jobs_exits_promptly():
    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        runner = _runner(q, StubImageHandler(), fetch, sink, poll_interval_s=5.0)
        task = asyncio.create_task(runner.run())
        await asyncio.sleep(0.05)
        runner.request_stop()
        return await asyncio.wait_for(task, timeout=1.0)

    report = asyncio.run(scenario())
    assert report.reason is ShutdownReason.STOP_REQUESTED


def test_shutdown_drains_in_flight_work_then_returns_it():
    """Requirement 4: work still running past the drain window goes back to the
    queue explicitly, rather than being stranded until its lease lapses."""

    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(b"draining")
        fetch.add(spec, b"draining")
        await _enqueue(q, spec)

        hanging = HangingHandler()
        runner = _runner(
            q, hanging, fetch, sink, lease_seconds=30, drain_timeout_s=0.1
        )
        task = asyncio.create_task(runner.run(max_jobs=1))
        await asyncio.wait_for(hanging.started.wait(), timeout=2)
        runner.request_stop()
        report = await asyncio.wait_for(task, timeout=3)
        return report, next(iter(q._jobs.values()))

    report, job = asyncio.run(scenario())
    assert report.abandoned == 1
    assert job.state is not JobState.LEASED, "abandoned work was not returned"
    assert job.attempts == 1


def test_finished_work_is_not_abandoned_by_a_late_stop():
    """A generous drain window must let nearly-complete work finish."""

    async def scenario():
        q, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(b"nearly done")
        fetch.add(spec, b"nearly done")
        await _enqueue(q, spec)
        runner = _runner(q, StubImageHandler(), fetch, sink, drain_timeout_s=5.0)
        runner.request_stop()
        return await runner.process(
            (await q.lease(IMAGE_QUEUE, worker_id="w", lease_seconds=30))[0]
        )

    report = asyncio.run(scenario())
    assert report.outcome is ExecutionOutcome.COMPLETED


# --------------------------------------------------------------------------
# Registry and wiring
# --------------------------------------------------------------------------


def test_registry_builds_one_worker_per_queue():
    root = Path(tempfile.mkdtemp(prefix="dfp-reg-"))
    workers = build_all_workers(
        queue=InMemoryQueue(), storage=LocalStorageBackend(root, signer=HmacUrlSigner(SECRET))
    )
    assert set(workers) == {IMAGE_QUEUE, VIDEO_QUEUE}
    assert workers[VIDEO_QUEUE].lease_seconds > workers[IMAGE_QUEUE].lease_seconds


def test_video_lease_is_longer_than_image_lease():
    """A video job runs for minutes; the same lease as an image would be
    reclaimed mid-flight."""
    root = Path(tempfile.mkdtemp(prefix="dfp-lease-"))
    storage = LocalStorageBackend(root, signer=HmacUrlSigner(SECRET))
    image = build_worker(MediaKind.IMAGE, queue=InMemoryQueue(), storage=storage)
    video = build_worker(MediaKind.VIDEO, queue=InMemoryQueue(), storage=storage)
    assert video.lease_seconds >= 4 * image.lease_seconds
    assert video.drain_timeout_s > image.drain_timeout_s


def test_end_to_end_over_real_storage():
    """Upload through storage, enqueue, run a worker, read the result back."""

    async def scenario():
        root = Path(tempfile.mkdtemp(prefix="dfp-e2e-"))
        storage = LocalStorageBackend(root, signer=HmacUrlSigner(SECRET))
        queue = InMemoryQueue()

        from storage.types import NewUpload

        # A decodable image: this test builds a worker from the registry, which
        # now wires the real inference-backed handler rather than a stub.
        import cv2

        data = cv2.imencode(
            ".jpg", np.random.default_rng(5).integers(0, 255, (256, 256, 3), dtype=np.uint8)
        )[1].tobytes()
        session, signed = await storage.create_upload(
            NewUpload(
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                content_type="image/jpeg",
            )
        )
        await storage.accept_upload(session.upload_id, data, signature=signed.signature)
        await storage.complete_upload(session.upload_id)

        spec = JobSpec(
            job_id=uuid4(),
            media_kind=MediaKind.IMAGE,
            storage_key=session.key,
            sha256=session.sha256,
            size_bytes=session.size_bytes,
            declared_mime="image/jpeg",
            model_version="stub-0",
            idempotency_key=f"{session.sha256}:stub-0",
            submitted_at=datetime.now(UTC),
        )
        await queue.enqueue(IMAGE_QUEUE, spec.model_dump(mode="json"))

        worker = build_worker(MediaKind.IMAGE, queue=queue, storage=storage)
        worker.poll_interval_s = 0.02
        report = await worker.run(max_jobs=1, idle_timeout_s=0.3)

        stored = await StorageResultSink(storage).find(str(spec.job_id))
        return report, stored

    report, stored = asyncio.run(scenario())
    assert report.completed == 1
    assert stored is not None
    EvidenceBundle.model_validate(stored.model_dump(mode="json"))


def test_fetcher_rejects_a_checksum_mismatch_permanently():
    """Bytes that are not what the job says they are will not become so on retry."""

    async def scenario():
        root = Path(tempfile.mkdtemp(prefix="dfp-mismatch-"))
        storage = LocalStorageBackend(root, signer=HmacUrlSigner(SECRET))
        await storage.put_object("uploads/x", b"actual bytes")
        spec = _spec(b"declared bytes")
        spec = spec.model_copy(update={"storage_key": "uploads/x"})
        try:
            await StorageMediaFetcher(storage).fetch(spec)
        except PermanentError:
            return True
        return False

    assert asyncio.run(scenario())


def test_fetcher_reports_a_missing_object_permanently():
    async def scenario():
        root = Path(tempfile.mkdtemp(prefix="dfp-missing-"))
        storage = LocalStorageBackend(root, signer=HmacUrlSigner(SECRET))
        try:
            await StorageMediaFetcher(storage).fetch(_spec(b"gone"))
        except PermanentError:
            return True
        return False

    assert asyncio.run(scenario())


# --------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------


def _run() -> int:
    tests = sorted(
        (n, o)
        for n, o in globals().items()
        if n.startswith("test_") and callable(o)
    )
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
