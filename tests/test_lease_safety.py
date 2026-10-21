"""Lease safety under CPU-bound inference.

The runner renews its lease from an ``asyncio`` task:

    heartbeat()  ->  await asyncio.sleep(lease * 1/3)  ->  extend_lease(...)

and it runs the handler as another task on the same loop. That arrangement is
correct only while the handler *yields*. An ``async def`` that does synchronous
CPU work — cv2 decode, Haar detection, a model forward — never yields, so the
sleep cannot fire. The heartbeat is not merely delayed; for the whole duration of
the analysis it does not exist.

The consequence is specific and bad. A worker that is alive and working loses a
lease it is actively holding, another worker picks the job up and redoes it, and
the first worker's ack is refused. Nothing errors on the happy path, so the only
external symptom is duplicated compute.

SCALED DURATIONS
----------------
The shipped lease is 60 s and renews at 20 s. Sleeping through that in a test
would make the suite unusable, so these tests scale time down and keep the
*ratios* the brief cares about. ``LEASE`` below is 0.6 s, so:

    inference / lease     shipped equivalent      constant
    ------------------    ------------------      --------
        0.17x                    10 s             WORK_10S
        0.50x                    30 s             WORK_30S
        1.00x                    60 s             WORK_60S
        2.00x                   120 s             WORK_120S

Ratios are what determine whether the heartbeat can keep up, so a test that
holds them fixed tests the same property the production numbers do.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.enums import MediaKind  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from inference import ImagePipeline, VideoPipeline  # noqa: E402
from jobqueue.errors import LeaseNotHeld  # noqa: E402
from jobqueue.memory import InMemoryQueue  # noqa: E402
from jobqueue.registry import IMAGE_QUEUE  # noqa: E402
from tests.detector_fixtures import haar_selection  # noqa: E402
from tests.stub_handlers import StubImageHandler  # noqa: E402
from workers.io import InMemoryResultSink  # noqa: E402
from workers.runner import WorkerRunner  # noqa: E402
from workers.types import ExecutionOutcome  # noqa: E402

LEASE = 0.6
"""Scaled stand-in for the shipped 60 s lease. Renewal lands at LEASE/3."""

WORK_10S = LEASE * 10 / 60
WORK_30S = LEASE * 30 / 60
WORK_60S = LEASE * 60 / 60
WORK_120S = LEASE * 120 / 60

RATIOS = [("10s", WORK_10S), ("30s", WORK_30S), ("60s", WORK_60S), ("120s", WORK_120S)]

DATA = b"lease-safety-fixture"


# --------------------------------------------------------------------------
# Handlers: the two ways to spend CPU time
# --------------------------------------------------------------------------


async def _silent_progress(fraction: float, *, stage=None) -> None:
    """Swallows progress so only the heartbeat can renew a lease.

    The runner's own ``progress`` callback renews opportunistically, which is
    correct in production and ruinous in a test of the heartbeat: a handler that
    reports progress after a long block renews the lease on the way out and
    hides the starvation entirely.
    """
    return None


class BlockingHandler:
    """Holds the event loop, as an un-offloaded pipeline does.

    ``time.sleep`` rather than ``asyncio.sleep`` on purpose: it reproduces a
    CPU-bound call that never yields, which is what cv2 and a model forward are.
    """

    media_kind = MediaKind.IMAGE.value
    name = "blocking"

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.calls = 0

    async def handle(self, request, progress):
        self.calls += 1
        time.sleep(self.seconds)
        return await StubImageHandler().handle(request, _silent_progress)


class ThreadedHandler:
    """Spends the same CPU time off the loop, as the corrected pipelines do."""

    media_kind = MediaKind.IMAGE.value
    name = "threaded"

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.calls = 0

    async def handle(self, request, progress):
        self.calls += 1
        await asyncio.to_thread(time.sleep, self.seconds)
        return await StubImageHandler().handle(request, _silent_progress)


class AuditedQueue(InMemoryQueue):
    """Records whether each lease renewal arrived before its own deadline.

    This is the measurable form of the safety property. Whether a *reaper*
    happens to reclaim a lapsed lease first is a timing detail of the
    deployment; whether the renewal was even capable of arriving in time is a
    property of the code, and it is the one worth asserting.
    """

    def __init__(self) -> None:
        super().__init__()
        self.renewals: list[bool] = []  # True when the renewal arrived late

    async def extend_lease(self, job_id, lease_id, *, lease_seconds=None):
        prior = self._leases.get(job_id)
        if prior is not None:
            self.renewals.append(self.now() > prior.expires_at)
        kwargs = {} if lease_seconds is None else {"lease_seconds": lease_seconds}
        return await super().extend_lease(job_id, lease_id, **kwargs)

    @property
    def late_renewals(self) -> int:
        return sum(self.renewals)


class LeaseStealingQueue(InMemoryQueue):
    """Another worker reclaims the job at a chosen moment.

    ``extend_lease`` and ``ack`` in the real backend check lease *identity*, not
    expiry — a lapsed lease is honoured until something reaps it. This double
    makes the reclaim happen deterministically rather than by racing a reaper on
    an event loop the handler may be holding.
    """

    def __init__(self, *, steal_on_renewal: int | None = None, steal_on_ack: bool = False):
        super().__init__()
        self.steal_on_renewal = steal_on_renewal
        self.steal_on_ack = steal_on_ack
        self.renewal_count = 0
        self.stolen = False

    async def extend_lease(self, job_id, lease_id, *, lease_seconds=None):
        self.renewal_count += 1
        if self.steal_on_renewal is not None and self.renewal_count >= self.steal_on_renewal:
            self.stolen = True
            raise LeaseNotHeld(job_id, lease_id, "job has been re-leased to another worker")
        kwargs = {} if lease_seconds is None else {"lease_seconds": lease_seconds}
        return await super().extend_lease(job_id, lease_id, **kwargs)

    async def ack(self, job_id, lease_id):
        if self.steal_on_ack:
            self.stolen = True
            raise LeaseNotHeld(job_id, lease_id, "job has been re-leased to another worker")
        return await super().ack(job_id, lease_id)


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def _spec(job_id: UUID | None = None) -> JobSpec:
    digest = hashlib.sha256(DATA).hexdigest()
    return JobSpec(
        job_id=job_id or uuid4(),
        media_kind=MediaKind.IMAGE,
        storage_key="uploads/lease",
        sha256=digest,
        size_bytes=len(DATA),
        declared_mime="image/jpeg",
        model_version="stub-0",
        idempotency_key=f"{digest}:stub-0",
        submitted_at=datetime.now(UTC),
    )


class _Fetcher:
    async def fetch(self, spec: JobSpec) -> bytes:
        return DATA


def _runner(queue, handler, sink, *, worker_id: str) -> WorkerRunner:
    return WorkerRunner(
        queue=queue,
        queue_name=IMAGE_QUEUE,
        handler=handler,
        fetcher=_Fetcher(),
        sink=sink,
        worker_id=worker_id,
        lease_seconds=LEASE,
        poll_interval_s=0.01,
    )


async def _enqueue(queue, spec: JobSpec):
    return await queue.enqueue(
        IMAGE_QUEUE, spec.model_dump(mode="json"), job_id=str(spec.job_id)
    )


async def _process_one(handler, sink=None, queue=None):
    """Lease and process a single job. Returns (report, queue, sink)."""
    queue = queue or InMemoryQueue()
    sink = sink or InMemoryResultSink()
    spec = _spec()
    await _enqueue(queue, spec)
    runner = _runner(queue, handler, sink, worker_id="w1")
    leased = await queue.lease(IMAGE_QUEUE, worker_id="w1", count=1, lease_seconds=LEASE)
    return await runner.process(leased[0]), queue, sink


# --------------------------------------------------------------------------
# The defect
# --------------------------------------------------------------------------


def test_blocking_inference_starves_the_lease_heartbeat():
    """The root cause, pinned as a permanent regression guard.

    The heartbeat is scheduled and never runs, because the loop it lives on is
    held by the handler. Zero renewals during the whole of a job that should
    have renewed twice — not "a few, late".
    """

    async def scenario():
        queue = AuditedQueue()
        report, _, _ = await _process_one(BlockingHandler(WORK_120S), queue=queue)
        return report, queue

    report, queue = asyncio.run(scenario())
    assert report.heartbeats == 0, (
        f"expected a starved heartbeat, got {report.heartbeats} renewals; "
        "the blocking double no longer reproduces the defect"
    )
    assert queue.renewals == [] or queue.late_renewals == len(queue.renewals), (
        "a blocked worker managed a renewal that was not late"
    )


def test_blocking_inference_cannot_renew_before_its_lease_expires():
    """The safety property, stated so it does not depend on reaper timing.

    A renewal that arrives after the deadline has already passed is not a
    renewal; it is a worker asking to keep something it has lost. Whether a
    reaper got there first is a deployment detail — that the renewal *could not*
    arrive in time is a property of the code.
    """

    async def scenario(handler):
        queue = AuditedQueue()
        await _process_one(handler, queue=queue)
        return queue

    blocked = asyncio.run(scenario(BlockingHandler(WORK_120S)))
    offloaded = asyncio.run(scenario(ThreadedHandler(WORK_120S)))

    assert blocked.late_renewals == len(blocked.renewals), (
        "the blocking handler renewed its lease on time, which it cannot do"
    )
    assert offloaded.renewals, "the offloaded handler never renewed at all"
    assert offloaded.late_renewals == 0, (
        f"{offloaded.late_renewals} of {len(offloaded.renewals)} renewals "
        "arrived after the lease had already expired"
    )


def test_an_expired_lease_is_honoured_until_something_reaps_it():
    """Documents why the defect's severity depends on fleet activity.

    ``extend_lease`` and ``ack`` check lease *identity*, never expiry; a lapsed
    lease is only realised when ``_reap_sync`` runs, and that happens inside
    ``lease()`` and ``reap_expired_leases()``. So a blocked worker on an idle
    single-worker deployment gets away with it, and the same worker on a busy
    queue loses the job the moment any peer polls. That is a wide severity range
    for one defect, and it is worth knowing which end a deployment sits at.
    """

    async def scenario():
        queue = InMemoryQueue()
        spec = _spec()
        await _enqueue(queue, spec)
        leased = await queue.lease(
            IMAGE_QUEUE, worker_id="w1", count=1, lease_seconds=0.05
        )
        job_id, lease_id = str(spec.job_id), leased[0].lease.lease_id
        await asyncio.sleep(0.12)  # well past the deadline

        # Nobody has reaped, so the lapsed lease is still accepted for renewal.
        # This is what lets a blocked worker on a quiet queue get away with it.
        renewed = await queue.extend_lease(job_id, lease_id, lease_seconds=0.05)
        accepted_after_expiry = renewed is not None

        # Let it lapse again; now a peer reaps and the job really is gone.
        await asyncio.sleep(0.12)
        reaped = await queue.reap_expired_leases()
        return accepted_after_expiry, reaped

    accepted_after_expiry, reaped = asyncio.run(scenario())
    assert accepted_after_expiry, (
        "extend_lease rejected an expired lease; if expiry is now enforced at "
        "renewal, the severity note in this docstring needs revisiting"
    )
    assert reaped == 1, "an expired lease was never reclaimed even by an explicit reap"


def test_real_pipelines_do_not_hold_the_event_loop():
    """The fix, measured where it matters: on the shipped pipelines.

    A ticker task samples the loop every 5 ms while a real analysis runs. Before
    the offload it recorded *zero* ticks for both pipelines — the loop was held
    from the first line of the analysis to the last, so the heartbeat could not
    have fired at any lease length.
    """

    async def measure(coro) -> tuple[int, float]:
        gaps: list[float] = []

        async def ticker() -> None:
            last = time.perf_counter()
            while True:
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        task = asyncio.create_task(ticker())
        await coro
        task.cancel()
        return len(gaps), (max(gaps) if gaps else float("inf"))

    image = cv2.imencode(
        ".jpg", np.random.default_rng(1).integers(0, 255, (900, 900, 3), dtype=np.uint8)
    )[1].tobytes()

    pipeline = ImagePipeline(face_selection=haar_selection())  # event-loop behaviour under test
    ticks, worst = asyncio.run(measure(pipeline.analyze_image(_img_spec(image), image)))
    assert ticks > 0, "the image pipeline held the event loop for the whole analysis"
    assert worst < LEASE / 3, (
        f"image analysis blocked the loop for {worst:.3f}s, longer than a "
        f"{LEASE:.1f}s lease's renewal interval"
    )

    video = _clip()
    ticks, worst = asyncio.run(
        measure(VideoPipeline().analyze_video(_vid_spec(video), video))
    )
    assert ticks > 0, "the video pipeline held the event loop for the whole analysis"
    assert worst < LEASE / 3, (
        f"video analysis blocked the loop for {worst:.3f}s, longer than a "
        f"{LEASE:.1f}s lease's renewal interval"
    )


# --------------------------------------------------------------------------
# The fix, across the required durations
# --------------------------------------------------------------------------


def test_offloaded_inference_keeps_its_lease_at_every_duration():
    """10 s, 30 s, 60 s and 120 s equivalents all complete without losing a lease."""

    async def scenario(seconds: float):
        report, _, sink = await _process_one(ThreadedHandler(seconds))
        return report, sink

    for label, seconds in RATIOS:
        report, sink = asyncio.run(scenario(seconds))
        assert report.outcome is ExecutionOutcome.COMPLETED, (
            f"{label}-equivalent inference ended as {report.outcome.value}"
        )
        assert len(sink) == 1, f"{label}: expected exactly one stored result"


def test_renewals_scale_with_the_length_of_the_work():
    """Work past the renewal interval must actually renew, not merely survive."""

    async def scenario(seconds: float):
        report, _, _ = await _process_one(ThreadedHandler(seconds))
        return report

    short = asyncio.run(scenario(WORK_10S))
    long = asyncio.run(scenario(WORK_120S))
    assert long.heartbeats > short.heartbeats, (
        f"a {WORK_120S}s job renewed {long.heartbeats} times and a {WORK_10S}s "
        f"job renewed {short.heartbeats}; the heartbeat is not tracking the work"
    )
    assert long.heartbeats >= 2, (
        "work lasting twice the lease must renew at least twice to survive it"
    )


def test_the_job_is_never_re_leased_while_a_healthy_worker_holds_it():
    """Lease-expiration pressure: a reaper runs throughout and must find nothing."""

    async def scenario():
        queue = InMemoryQueue()
        sink = InMemoryResultSink()
        spec = _spec()
        await _enqueue(queue, spec)
        runner = _runner(queue, ThreadedHandler(WORK_120S), sink, worker_id="w1")
        leased = await queue.lease(
            IMAGE_QUEUE, worker_id="w1", count=1, lease_seconds=LEASE
        )

        reaped = 0
        stop = asyncio.Event()

        async def reaper() -> None:
            nonlocal reaped
            while not stop.is_set():
                reaped += await queue.reap_expired_leases()
                await asyncio.sleep(0.02)

        task = asyncio.create_task(reaper())
        report = await runner.process(leased[0])
        stop.set()
        await task
        return report, reaped, sink

    report, reaped, sink = asyncio.run(scenario())
    assert reaped == 0, (
        f"the reaper reclaimed {reaped} lease(s) from a worker that was alive "
        "and renewing"
    )
    assert report.outcome is ExecutionOutcome.COMPLETED
    assert len(sink) == 1


# --------------------------------------------------------------------------
# No duplicate completion, no stolen acknowledgement
# --------------------------------------------------------------------------


def test_a_worker_whose_lease_is_stolen_mid_flight_does_not_acknowledge():
    """The stolen-ack case.

    Acking here would mark *another* worker's in-flight job complete and throw
    away the result it is about to produce; nacking would schedule a retry of a
    job already being retried. The runner must do neither.
    """

    async def scenario():
        queue = LeaseStealingQueue(steal_on_renewal=1)
        sink = InMemoryResultSink()
        spec = _spec()
        await _enqueue(queue, spec)

        acked: list[str] = []
        nacked: list[str] = []
        real_ack, real_nack = queue.ack, queue.nack

        async def watched_ack(job_id, lease_id):
            acked.append(lease_id)
            return await real_ack(job_id, lease_id)

        async def watched_nack(job_id, lease_id, **kw):
            nacked.append(lease_id)
            return await real_nack(job_id, lease_id, **kw)

        queue.ack, queue.nack = watched_ack, watched_nack

        runner = _runner(queue, ThreadedHandler(WORK_120S), sink, worker_id="w1")
        leased = await queue.lease(
            IMAGE_QUEUE, worker_id="w1", count=1, lease_seconds=LEASE
        )
        stolen = leased[0].lease.lease_id
        report = await runner.process(leased[0])
        return report, acked, nacked, stolen, queue

    report, acked, nacked, stolen, queue = asyncio.run(scenario())
    assert queue.stolen, "the stealing double never fired; the test proves nothing"
    assert report.outcome is ExecutionOutcome.LEASE_LOST
    assert stolen not in acked, "a worker acked with a lease it no longer held"
    assert stolen not in nacked, "a worker nacked with a lease it no longer held"


def test_a_result_persisted_before_the_ack_is_never_lost():
    """Losing the lease *after* persisting must not discard the work.

    The runner persists and then acks, in that order. A lease stolen between the
    two costs the acknowledgement, not the result — and the next worker to lease
    the job finds the stored bundle and acks immediately instead of redoing the
    analysis.
    """

    async def scenario():
        queue = LeaseStealingQueue(steal_on_ack=True)
        sink = InMemoryResultSink()
        spec = _spec()
        await _enqueue(queue, spec)

        handler = ThreadedHandler(WORK_10S)
        r1 = _runner(queue, handler, sink, worker_id="w1")
        leased = await queue.lease(
            IMAGE_QUEUE, worker_id="w1", count=1, lease_seconds=LEASE
        )
        first = await r1.process(leased[0])

        # The result survived the failed ack.
        assert len(sink) == 1, "a stolen ack discarded a persisted result"

        # The job returns to the queue and a second worker picks it up.
        queue.steal_on_ack = False
        await queue.reap_expired_leases()
        leased2 = await queue.lease(
            IMAGE_QUEUE, worker_id="w2", count=1, lease_seconds=LEASE
        )
        before = handler.calls
        r2 = _runner(queue, handler, sink, worker_id="w2")
        second = await r2.process(leased2[0]) if leased2 else None
        return first, second, handler.calls - before, sink

    first, second, extra_calls, sink = asyncio.run(scenario())
    assert first.outcome is ExecutionOutcome.LEASE_LOST
    assert first.verdict is not None, "the report lost the verdict it had persisted"
    if second is not None:
        assert second.outcome is ExecutionOutcome.ALREADY_DONE, (
            f"crash recovery reported {second.outcome.value} instead of finding "
            "the stored result"
        )
        assert extra_calls == 0, "crash recovery re-ran an analysis already on disk"
    assert len(sink) == 1, (
        f"{len(sink)} results stored for one job; duplicate delivery must still "
        "commit exactly one bundle"
    )


def test_duplicate_execution_commits_exactly_one_bundle():
    """At-least-once means the job can genuinely run twice. It must store once.

    Two workers race on the same job id with a shared sink. First writer wins;
    the loser accepts the stored bundle rather than overwriting it.
    """

    async def scenario():
        sink = InMemoryResultSink()
        spec = _spec()
        results = []
        for worker in ("w1", "w2"):
            queue = InMemoryQueue()
            await _enqueue(queue, spec)
            runner = _runner(queue, ThreadedHandler(WORK_10S), sink, worker_id=worker)
            leased = await queue.lease(
                IMAGE_QUEUE, worker_id=worker, count=1, lease_seconds=LEASE
            )
            results.append(await runner.process(leased[0]))
        return results, sink

    results, sink = asyncio.run(scenario())
    assert results[0].outcome is ExecutionOutcome.COMPLETED
    assert results[1].outcome is ExecutionOutcome.ALREADY_DONE, (
        "the second execution of the same job did not detect the stored result"
    )
    assert len(sink) == 1


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _img_spec(data: bytes) -> JobSpec:
    return JobSpec(
        job_id=uuid4(),
        media_kind=MediaKind.IMAGE,
        storage_key="u",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="image/jpeg",
        model_version="stub-0",
        idempotency_key="k",
        submitted_at=datetime.now(UTC),
    )


def _vid_spec(data: bytes) -> JobSpec:
    return JobSpec(
        job_id=uuid4(),
        media_kind=MediaKind.VIDEO,
        storage_key="u",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="video/mp4",
        model_version="stub-0",
        idempotency_key="k",
        submitted_at=datetime.now(UTC),
    )


def _clip(frames: int = 48, size: int = 192, fps: int = 25) -> bytes:
    path = Path(tempfile.mkdtemp()) / "lease.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    rng = np.random.default_rng(3)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _run() -> int:
    tests = sorted((n, o) for n, o in globals().items() if n.startswith("test_") and callable(o))
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
