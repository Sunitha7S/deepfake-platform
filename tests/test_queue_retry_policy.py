"""Per-queue retry policies are applied, by both working queue backends.

``jobqueue.registry.QUEUE_DEFAULTS`` has always specified different budgets for
the two queues — image: 3 attempts, 2 s base backoff, 30 s ceiling; video:
2 attempts, 10 s base, 120 s ceiling — because a video retry costs GPU-minutes.
Until this suite, nothing applied them: ``build_queue`` handed each backend one
default ``RetryPolicy()``, so a video job got three attempts at a 2 s backoff
and an image job's backoff could grow to 300 s. Nothing failed; the asymmetry
was simply decorative.

Every scenario here runs against ``InMemoryQueue`` and ``SQLiteQueue`` alike,
on a **fake clock**: the backend's ``now()`` returns a value the test controls,
so backoff is asserted to the microsecond and lease expiry happens by moving
the clock, never by sleeping. The schedules are checked with jitter disabled
(copies of the shipped policies, otherwise identical); the shipped, jittered
policies are checked for bounds.

    python tests/test_queue_retry_policy.py
"""

from __future__ import annotations

import asyncio
import itertools
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobqueue.memory import InMemoryQueue  # noqa: E402
from jobqueue.policy import RetryPolicy  # noqa: E402
from jobqueue.registry import (  # noqa: E402
    IMAGE_QUEUE,
    QUEUE_DEFAULTS,
    VIDEO_QUEUE,
    QueueConfig,
    build_queue,
)
from jobqueue.sqlite import SQLiteQueue  # noqa: E402
from jobqueue.types import JobState, NackOutcome  # noqa: E402

T0 = datetime(2026, 1, 1, tzinfo=UTC)
_n = itertools.count()

EXACT = {q: p.model_copy(update={"jitter": False}) for q, p in QUEUE_DEFAULTS.items()}
"""The shipped policies with jitter off, so a delay is a number, not a range."""


def _sqlite_path() -> Path:
    return Path(tempfile.mkdtemp(prefix="dfp-retry-")) / f"q{next(_n)}.sqlite3"


def _clocked(cls: type) -> type:
    """A subclass whose clock is ``self.clock`` — set and advanced by the test."""

    class Clocked(cls):
        clock: datetime = T0

        def now(self) -> datetime:  # BaseQueueBackend.now is a staticmethod
            return self.clock

    Clocked.__name__ = f"Clocked{cls.__name__}"
    return Clocked


BACKENDS: dict[str, Callable[..., object]] = {
    "memory": lambda **kw: _clocked(InMemoryQueue)(**kw),
    "sqlite": lambda **kw: _clocked(SQLiteQueue)(_sqlite_path(), **kw),
}


def _each_backend(scenario: Callable[[Callable[..., object]], object]) -> None:
    """Run ``scenario`` once per backend; report which one failed."""
    for name, make in BACKENDS.items():
        try:
            asyncio.run(scenario(make))
        except AssertionError as exc:
            raise AssertionError(f"[{name}] {exc}") from exc


async def _fail(q, queue: str, *, retryable: bool = True, error: str = "boom"):
    """Lease the one available job on ``queue`` and nack it."""
    [held] = await q.lease(queue, worker_id="w")
    return await q.nack(held.job_id, held.lease.lease_id, error=error, retryable=retryable)


# --------------------------------------------------------------------------
# Selection and wiring
# --------------------------------------------------------------------------


def test_build_queue_wires_the_configured_policy_for_each_queue():
    """The production path. Before the fix both lookups returned RetryPolicy()."""
    for config in (
        QueueConfig(kind="memory"),
        QueueConfig(kind="sqlite", path=_sqlite_path()),
    ):
        q = build_queue(config)
        assert q.policy_for_queue(IMAGE_QUEUE) == QUEUE_DEFAULTS[IMAGE_QUEUE], config.kind
        assert q.policy_for_queue(VIDEO_QUEUE) == QUEUE_DEFAULTS[VIDEO_QUEUE], config.kind
        assert q.policy_for_queue("unlisted") == config.policy, "fallback must stay the fallback"


def test_redis_skeleton_accepts_the_same_policies():
    q = build_queue(QueueConfig(kind="redis", url="redis://localhost:6379/0"))
    assert q.policy_for_queue(VIDEO_QUEUE) == QUEUE_DEFAULTS[VIDEO_QUEUE]


def test_the_shipped_policies_are_the_documented_ones():
    """Pins the numbers this fix exists to enforce, so a silent edit is loud."""
    image, video = QUEUE_DEFAULTS[IMAGE_QUEUE], QUEUE_DEFAULTS[VIDEO_QUEUE]
    assert (image.max_attempts, image.base_delay_s, image.max_delay_s) == (3, 2.0, 30.0)
    assert (video.max_attempts, video.base_delay_s, video.max_delay_s) == (2, 10.0, 120.0)


def test_queue_policies_can_be_overridden_per_config():
    custom = RetryPolicy(max_attempts=5, base_delay_s=1.0, max_delay_s=5.0)
    q = build_queue(QueueConfig(kind="memory", queue_policies={VIDEO_QUEUE: custom}))
    assert q.policy_for_queue(VIDEO_QUEUE) == custom
    assert q.policy_for_queue(IMAGE_QUEUE) == QueueConfig().policy, (
        "an explicit mapping replaces the defaults; it does not merge with them"
    )


def test_a_backend_built_without_queue_policies_behaves_as_before():
    """Direct construction (as the existing suites do) keeps one policy for all."""
    single = RetryPolicy(max_attempts=4)
    for q in (InMemoryQueue(policy=single), SQLiteQueue(_sqlite_path(), policy=single)):
        assert q.policy_for_queue(VIDEO_QUEUE) is single
        assert q.policy_for_queue(IMAGE_QUEUE) is single


# --------------------------------------------------------------------------
# Attempt budgets
# --------------------------------------------------------------------------


def test_enqueue_takes_its_attempt_budget_from_the_job_s_queue():
    async def scenario(make):
        q = make(queue_policies=QUEUE_DEFAULTS)
        image = await q.enqueue(IMAGE_QUEUE, {})
        video = await q.enqueue(VIDEO_QUEUE, {})
        other = await q.enqueue("unlisted", {})
        assert (image.max_attempts, video.max_attempts) == (3, 2)
        assert other.max_attempts == q.policy.max_attempts
        explicit = await q.enqueue(VIDEO_QUEUE, {}, max_attempts=5)
        assert explicit.max_attempts == 5, "an explicit max_attempts still wins"

    _each_backend(scenario)


def test_a_video_job_dead_letters_after_two_attempts():
    """The defect, end to end: under the old wiring this job got a third try."""

    async def scenario(make):
        q = make(queue_policies=EXACT)
        job = await q.enqueue(VIDEO_QUEUE, {"clip": 1})

        first = await _fail(q, VIDEO_QUEUE, error="gpu oom")
        assert first.outcome is NackOutcome.RETRY_SCHEDULED
        assert first.envelope.attempts == 1

        q.clock = first.retry_at
        second = await _fail(q, VIDEO_QUEUE, error="gpu oom again")
        assert second.outcome is NackOutcome.DEAD_LETTERED, "video got a third attempt"
        assert second.retry_at is None

        final = await q.get(job.job_id)
        assert (final.state, final.attempts, final.last_error) == (
            JobState.DEAD, 2, "gpu oom again"
        )
        assert [d.job_id for d in await q.dead_letters(VIDEO_QUEUE)] == [job.job_id]
        assert await q.lease(VIDEO_QUEUE, worker_id="w") == []

    _each_backend(scenario)


def test_an_image_job_gets_three_attempts_then_dead_letters():
    async def scenario(make):
        q = make(queue_policies=EXACT)
        job = await q.enqueue(IMAGE_QUEUE, {})
        outcomes = []
        for _ in range(3):
            result = await _fail(q, IMAGE_QUEUE)
            outcomes.append(result.outcome)
            if result.retry_at is not None:
                q.clock = result.retry_at
        assert outcomes == [
            NackOutcome.RETRY_SCHEDULED,
            NackOutcome.RETRY_SCHEDULED,
            NackOutcome.DEAD_LETTERED,
        ]
        final = await q.get(job.job_id)
        assert (final.state, final.attempts) == (JobState.DEAD, 3)

    _each_backend(scenario)


def test_a_permanent_failure_dead_letters_at_once_on_either_queue():
    async def scenario(make):
        q = make(queue_policies=EXACT)
        for queue in (IMAGE_QUEUE, VIDEO_QUEUE):
            await q.enqueue(queue, {})
            result = await _fail(q, queue, retryable=False, error="corrupt")
            assert result.outcome is NackOutcome.DEAD_LETTERED, queue
            assert result.envelope.attempts == 1, f"{queue}: a permanent failure was retried"

    _each_backend(scenario)


# --------------------------------------------------------------------------
# Backoff
# --------------------------------------------------------------------------


def test_video_backoff_is_ten_seconds():
    async def scenario(make):
        q = make(queue_policies=EXACT)
        await q.enqueue(VIDEO_QUEUE, {})
        result = await _fail(q, VIDEO_QUEUE)
        assert result.retry_at - T0 == timedelta(seconds=10), result.retry_at - T0
        assert result.envelope.available_at == result.retry_at

    _each_backend(scenario)


def test_image_backoff_follows_its_schedule_and_stops_at_thirty_seconds():
    """2 s, 6 s, 18 s, then the 30 s ceiling (the default 300 s cap would give
    54 s and 162 s). A generous explicit budget lets the schedule reach it."""

    async def scenario(make):
        q = make(queue_policies=EXACT)
        await q.enqueue(IMAGE_QUEUE, {}, max_attempts=6)
        delays = []
        for _ in range(5):
            before = q.clock
            result = await _fail(q, IMAGE_QUEUE)
            assert result.outcome is NackOutcome.RETRY_SCHEDULED
            delays.append((result.retry_at - before).total_seconds())
            q.clock = result.retry_at
        assert delays == [2.0, 6.0, 18.0, 30.0, 30.0], delays

    _each_backend(scenario)


def test_a_backed_off_job_is_not_leasable_until_its_retry_time():
    async def scenario(make):
        q = make(queue_policies=EXACT)
        await q.enqueue(VIDEO_QUEUE, {})
        result = await _fail(q, VIDEO_QUEUE)
        q.clock = result.retry_at - timedelta(microseconds=1)
        assert await q.lease(VIDEO_QUEUE, worker_id="w") == [], "leased before its retry time"
        assert (await q.stats(VIDEO_QUEUE)).delayed == 1
        q.clock = result.retry_at
        assert len(await q.lease(VIDEO_QUEUE, worker_id="w")) == 1

    _each_backend(scenario)


def test_shipped_jittered_backoff_stays_within_each_queue_s_ceiling():
    """The real policies jitter (full jitter: uniform in [0, cap]), so assert
    bounds over many draws rather than values."""

    async def scenario(make):
        q = make(queue_policies=QUEUE_DEFAULTS)
        for _ in range(25):
            await q.enqueue(VIDEO_QUEUE, {})
            await q.enqueue(IMAGE_QUEUE, {}, max_attempts=10)
        video_delays, image_tail = [], []
        for _ in range(25):
            r = await _fail(q, VIDEO_QUEUE)
            video_delays.append((r.retry_at - q.clock).total_seconds())
        for _ in range(25):
            r = await _fail(q, IMAGE_QUEUE)
            image_tail.append((r.retry_at - q.clock).total_seconds())
        assert all(0.0 <= d <= 10.0 for d in video_delays), max(video_delays)
        assert all(0.0 <= d <= 2.0 for d in image_tail), max(image_tail)
        assert len(set(video_delays)) > 1, "the shipped video policy lost its jitter"

    _each_backend(scenario)


# --------------------------------------------------------------------------
# Lease expiry uses the same per-queue policy
# --------------------------------------------------------------------------


def test_a_lapsed_video_lease_backs_off_ten_seconds_then_dead_letters():
    """A worker that dies mid-job is a failed attempt under the job's own queue
    policy — not the fallback. Expiry happens by moving the clock."""

    async def scenario(make):
        q = make(queue_policies=EXACT)
        job = await q.enqueue(VIDEO_QUEUE, {})

        await q.lease(VIDEO_QUEUE, worker_id="doomed", lease_seconds=60)
        q.clock = T0 + timedelta(seconds=60)
        assert await q.reap_expired_leases() == 1
        backed_off = await q.get(job.job_id)
        assert backed_off.state is JobState.DELAYED and backed_off.attempts == 1
        assert backed_off.available_at - q.clock == timedelta(seconds=10)

        q.clock = backed_off.available_at
        await q.lease(VIDEO_QUEUE, worker_id="doomed-again", lease_seconds=60)
        q.clock += timedelta(seconds=60)
        assert await q.reap_expired_leases() == 1
        dead = await q.get(job.job_id)
        assert (dead.state, dead.attempts) == (JobState.DEAD, 2), "a third lease was allowed"
        assert "did not report back" in (dead.last_error or "")

    _each_backend(scenario)


def test_a_lease_is_not_reaped_one_microsecond_early():
    async def scenario(make):
        q = make(queue_policies=EXACT)
        await q.enqueue(IMAGE_QUEUE, {})
        await q.lease(IMAGE_QUEUE, worker_id="w", lease_seconds=30)
        q.clock = T0 + timedelta(seconds=30) - timedelta(microseconds=1)
        assert await q.reap_expired_leases() == 0
        q.clock += timedelta(microseconds=1)
        assert await q.reap_expired_leases() == 1

    _each_backend(scenario)


def test_requeuing_a_dead_video_job_restores_its_full_budget():
    async def scenario(make):
        q = make(queue_policies=EXACT)
        job = await q.enqueue(VIDEO_QUEUE, {})
        await _fail(q, VIDEO_QUEUE, retryable=False)
        revived = await q.requeue_dead_letter(job.job_id)
        assert (revived.state, revived.attempts, revived.max_attempts) == (JobState.READY, 0, 2)
        first = await _fail(q, VIDEO_QUEUE)
        assert first.outcome is NackOutcome.RETRY_SCHEDULED

    _each_backend(scenario)


# --------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------


def _run() -> int:
    tests = sorted(
        (n, o) for n, o in globals().items() if n.startswith("test_") and callable(o)
    )
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:400]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
