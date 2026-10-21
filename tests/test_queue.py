"""Queue subsystem tests.

Two layers, mirroring the storage suite:

* the reusable conformance suite, run against ``InMemoryQueue``;
* backend-specific and policy tests for behaviour the shared contract cannot
  express.

Note the imports: no FastAPI, no ``storage``, no ``backend``, no ``contracts``.
Independence is checked here rather than merely claimed.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jobqueue.base import QueueBackend  # noqa: E402
from jobqueue.errors import JobNotFound, LeaseNotHeld  # noqa: E402
from jobqueue.memory import InMemoryQueue  # noqa: E402
from jobqueue.policy import RetryPolicy  # noqa: E402
from jobqueue.redis import RedisQueue  # noqa: E402
from jobqueue.registry import (  # noqa: E402
    IMAGE_QUEUE,
    QUEUE_NAMES,
    VIDEO_QUEUE,
    QueueConfig,
    build_queue,
    policy_for,
)
from jobqueue.types import JobState, NackOutcome, Priority  # noqa: E402
from tests.queue_conformance import CHECKS, run_conformance  # noqa: E402


def _make() -> InMemoryQueue:
    return InMemoryQueue()


# --------------------------------------------------------------------------
# Conformance
# --------------------------------------------------------------------------


def test_in_memory_queue_passes_the_conformance_suite():
    failures = asyncio.run(run_conformance(_make))
    assert failures == [], "conformance failures:\n  " + "\n  ".join(failures)


def test_conformance_suite_is_not_vacuous():
    assert len(CHECKS) >= 30


def test_conformance_suite_catches_a_broken_backend():
    """Prove the suite has teeth with a backend that loses FIFO order."""

    class ReversesOrder(InMemoryQueue):
        async def lease(self, queue, *, worker_id, count=1, lease_seconds=60):
            leased = await super().lease(
                queue, worker_id=worker_id, count=count, lease_seconds=lease_seconds
            )
            return list(reversed(leased))

    failures = asyncio.run(run_conformance(ReversesOrder))
    assert failures, "the suite passed a backend that reorders jobs"
    assert any("fifo" in f for f in failures), failures


def test_conformance_suite_catches_double_delivery():
    """The at-least-once guarantee must be enforced, not assumed."""

    class LeasesTwice(InMemoryQueue):
        async def lease(self, queue, *, worker_id, count=1, lease_seconds=60):
            leased = await super().lease(
                queue, worker_id=worker_id, count=count, lease_seconds=lease_seconds
            )
            if not leased:
                # Hand back an already-leased job: the classic double-delivery bug.
                held = [j for j in self._jobs.values() if j.state is JobState.LEASED]
                if held:
                    from jobqueue.types import Lease, LeasedJob

                    job = held[0]
                    return [
                        LeasedJob(
                            envelope=job,
                            lease=Lease(
                                job_id=job.job_id,
                                lease_id="duplicate",
                                worker_id=worker_id,
                                acquired_at=datetime.now(UTC),
                                expires_at=self.lease_deadline(60),
                            ),
                        )
                    ]
            return leased

    failures = asyncio.run(run_conformance(LeasesTwice))
    assert any("leased_twice" in f for f in failures), failures


def test_redis_skeleton_satisfies_the_protocol():
    assert isinstance(RedisQueue(url="redis://localhost:6379/0"), QueueBackend)


def test_redis_health_reports_unhealthy_rather_than_raising():
    health = asyncio.run(RedisQueue().health())
    assert health.ok is False
    assert "not implemented" in (health.detail or "")


def test_redis_operations_explain_what_is_missing():
    q = RedisQueue()
    try:
        asyncio.run(q.lease("image", worker_id="w1"))
    except NotImplementedError as exc:
        assert "Lua" in str(exc), "the skeleton should name the atomicity requirement"
        return
    raise AssertionError("expected NotImplementedError")


def test_redis_repr_never_leaks_a_password():
    q = RedisQueue(url="redis://user:hunter2@cache.internal:6379/0")
    assert "hunter2" not in repr(q)


def test_redis_key_layout_is_namespaced():
    q = RedisQueue(namespace="dfp")
    assert q._k_ready("image") == "dfp:q:image:ready"
    assert q._k_lease("j1") == "dfp:lease:j1"


# --------------------------------------------------------------------------
# FIFO, in depth
# --------------------------------------------------------------------------


def test_fifo_survives_identical_timestamps():
    """The reason ``sequence`` exists.

    Jobs are given a literally identical ``enqueued_at``, so any implementation
    relying on timestamp comparison alone has nothing to order by.
    """
    q = _make()

    async def scenario():
        frozen = datetime.now(UTC)
        ids = []
        for i in range(10):
            job = await q.enqueue(IMAGE_QUEUE, {"i": i})
            # Force a tie on both ordering keys except the sequence.
            q._jobs[job.job_id] = job.model_copy(
                update={"enqueued_at": frozen, "available_at": frozen}
            )
            ids.append(job.job_id)
        leased = await q.lease(IMAGE_QUEUE, worker_id="w", count=10)
        return ids, [j.job_id for j in leased]

    enqueued, delivered = asyncio.run(scenario())
    assert delivered == enqueued, "FIFO broke when timestamps collided"


def test_sequence_is_monotonic_across_queues():
    q = _make()

    async def scenario():
        a = await q.enqueue(IMAGE_QUEUE, {})
        b = await q.enqueue(VIDEO_QUEUE, {})
        c = await q.enqueue(IMAGE_QUEUE, {})
        return a.sequence, b.sequence, c.sequence

    a, b, c = asyncio.run(scenario())
    assert a < b < c


def test_requeued_job_does_not_jump_the_queue():
    """A retried job becomes available later, so newer work is not starved."""
    q = _make()

    async def scenario():
        first = await q.enqueue(IMAGE_QUEUE, {"which": "first"})
        [leased] = await q.lease(IMAGE_QUEUE, worker_id="w1")
        await q.nack(leased.job_id, leased.lease.lease_id, error="blip", retryable=True)
        await q.enqueue(IMAGE_QUEUE, {"which": "second"})
        got = await q.lease(IMAGE_QUEUE, worker_id="w2", count=5)
        return [j.payload["which"] for j in got]

    assert asyncio.run(scenario()) == ["second"], "the backed-off job should not be ready yet"


# --------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------


def test_backoff_grows_and_is_capped():
    policy = RetryPolicy(base_delay_s=2.0, max_delay_s=30.0, jitter=False)
    delays = [policy.delay_for(n).total_seconds() for n in range(1, 8)]
    assert delays[0] == 2.0
    assert delays == sorted(delays), "backoff must be non-decreasing"
    assert max(delays) <= 30.0, "backoff must respect its ceiling"


def test_jitter_spreads_retries():
    """Without jitter, a failure that hits many jobs retries them in lockstep."""
    policy = RetryPolicy(base_delay_s=10.0, jitter=True)
    import random

    values = {policy.delay_for(3, rng=random.Random(seed)).total_seconds() for seed in range(20)}
    assert len(values) > 1, "jitter produced identical delays"


def test_non_retryable_never_retries_regardless_of_budget():
    policy = RetryPolicy(max_attempts=10)
    assert policy.should_retry(attempts=1, retryable=True) is True
    assert policy.should_retry(attempts=1, retryable=False) is False


def test_exhausted_budget_stops_retrying():
    policy = RetryPolicy(max_attempts=3)
    assert policy.should_retry(attempts=2, retryable=True) is True
    assert policy.should_retry(attempts=3, retryable=True) is False


def test_policy_rejects_a_ceiling_below_the_floor():
    try:
        RetryPolicy(base_delay_s=30.0, max_delay_s=5.0)
    except ValueError:
        return
    raise AssertionError("max_delay_s below base_delay_s should be rejected")


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_registry_builds_an_in_memory_queue():
    q = build_queue(QueueConfig(kind="memory"))
    assert isinstance(q, QueueBackend)
    assert q.name == "memory"


def test_registry_builds_a_redis_queue():
    q = build_queue(QueueConfig(kind="redis", url="redis://localhost:6379/0"))
    assert q.name == "redis"


def test_registry_requires_a_url_for_redis():
    try:
        QueueConfig(kind="redis")
    except ValueError as exc:
        assert "url is required" in str(exc)
        return
    raise AssertionError("QueueConfig accepted redis without a url")


def test_queue_config_never_prints_its_url():
    config = QueueConfig(kind="redis", url="redis://user:hunter2@host:6379/0")
    assert "hunter2" not in repr(config)
    assert "hunter2" not in str(config)


def test_image_and_video_queues_have_different_retry_budgets():
    """Video work is minutes of GPU time; retrying it hard starves healthy jobs."""
    image = policy_for(IMAGE_QUEUE)
    video = policy_for(VIDEO_QUEUE)
    assert video.max_attempts < image.max_attempts
    assert video.base_delay_s > image.base_delay_s


def test_queue_names_are_declared_once():
    assert QUEUE_NAMES == (IMAGE_QUEUE, VIDEO_QUEUE)


# --------------------------------------------------------------------------
# Independence
# --------------------------------------------------------------------------


def test_queue_package_imports_nothing_it_should_not():
    import jobqueue
    import jobqueue.base
    import jobqueue.memory
    import jobqueue.redis
    import jobqueue.registry
    import jobqueue.sqlite

    forbidden = ("fastapi", "starlette", "import torch", "from backend", "from contracts", "from storage")
    for module in (
        jobqueue, jobqueue.base, jobqueue.memory, jobqueue.redis, jobqueue.registry, jobqueue.sqlite
    ):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{module.__name__} references {token}"


def test_package_name_does_not_shadow_the_stdlib():
    """``queue`` is a stdlib module. A top-level package of that name is
    unreachable, because stdlib ``queue`` is already in sys.modules before user
    code runs — and ``concurrent.futures.thread`` imports it, so anything using
    a thread pool breaks too."""
    import queue as stdlib_queue

    assert hasattr(stdlib_queue, "Queue"), "stdlib queue was shadowed"
    assert "jobqueue" in sys.modules


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
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
