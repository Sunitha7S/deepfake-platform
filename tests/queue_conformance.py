"""Reusable ``QueueBackend`` conformance suite.

Not a test module — a suite any backend runs through, mirroring
``storage_conformance``. When ``RedisQueue`` is implemented it imports this and
proves it honours the same contract, rather than growing its own tests that
quietly assert weaker guarantees.

Usage::

    failures = asyncio.run(run_conformance(lambda: MyQueue(...)))
    assert failures == []

Each check builds a **fresh backend from the factory**, so ordering between
checks cannot matter and one failure cannot cascade into the next.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from jobqueue.base import QueueBackend
from jobqueue.errors import JobNotFound, JobNotLeased, LeaseNotHeld
from jobqueue.policy import RetryPolicy
from jobqueue.types import JobState, NackOutcome, Priority, QueueHealth

BackendFactory = Callable[[], Any]

Q = "conformance"


# --------------------------------------------------------------------------
# Structural
# --------------------------------------------------------------------------


async def check_satisfies_protocol(make: BackendFactory) -> None:
    backend = make()
    assert isinstance(backend, QueueBackend), (
        f"{type(backend).__name__} does not satisfy QueueBackend"
    )
    assert isinstance(backend.name, str) and backend.name


async def check_health_never_raises(make: BackendFactory) -> None:
    health = await make().health()
    assert isinstance(health, QueueHealth)
    assert health.backend


async def check_health_is_ok(make: BackendFactory) -> None:
    health = await make().health()
    assert health.ok is True, f"expected a healthy queue, got: {health.detail}"


# --------------------------------------------------------------------------
# Enqueue
# --------------------------------------------------------------------------


async def check_enqueue_starts_ready(make: BackendFactory) -> None:
    q = make()
    job = await q.enqueue(Q, {"a": 1})
    assert job.state is JobState.READY
    assert job.attempts == 0
    assert job.payload == {"a": 1}


async def check_delay_defers_availability(make: BackendFactory) -> None:
    q = make()
    await q.enqueue(Q, {"late": True}, delay_s=30)
    assert await q.lease(Q, worker_id="w1") == [], "a delayed job must not be leasable yet"
    stats = await q.stats(Q)
    assert stats.delayed == 1 and stats.ready == 0


async def check_idempotent_enqueue_returns_the_original(make: BackendFactory) -> None:
    q = make()
    first = await q.enqueue(Q, {"n": 1}, idempotency_key="same")
    second = await q.enqueue(Q, {"n": 2}, idempotency_key="same")
    assert first.job_id == second.job_id, "the same key must not create a second job"
    assert second.payload == {"n": 1}, "the original payload wins"
    assert (await q.stats(Q)).ready == 1


async def check_distinct_keys_create_distinct_jobs(make: BackendFactory) -> None:
    q = make()
    a = await q.enqueue(Q, {}, idempotency_key="a")
    b = await q.enqueue(Q, {}, idempotency_key="b")
    assert a.job_id != b.job_id


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------


async def check_fifo_among_equal_priority(make: BackendFactory) -> None:
    """FIFO must hold for jobs of equal priority and availability.

    Enqueued in a tight loop on purpose: several will share an ``enqueued_at``
    to the microsecond, so passing this requires a real tiebreak rather than
    timestamp comparison that happens to work when the clock ticks between
    calls.
    """
    q = make()
    for i in range(25):
        await q.enqueue(Q, {"i": i})
    leased = await q.lease(Q, worker_id="w1", count=25)
    assert [j.payload["i"] for j in leased] == list(range(25)), (
        "equal-priority jobs were not delivered in enqueue order"
    )


async def check_fifo_holds_across_separate_leases(make: BackendFactory) -> None:
    """Order must survive being drained a few at a time."""
    q = make()
    for i in range(9):
        await q.enqueue(Q, {"i": i})
    seen: list[int] = []
    for _ in range(3):
        batch = await q.lease(Q, worker_id="w1", count=3)
        seen.extend(j.payload["i"] for j in batch)
    assert seen == list(range(9))


async def check_priority_outranks_arrival_order(make: BackendFactory) -> None:
    q = make()
    await q.enqueue(Q, {"p": "low"}, priority=Priority.LOW)
    await q.enqueue(Q, {"p": "normal"}, priority=Priority.NORMAL)
    await q.enqueue(Q, {"p": "high"}, priority=Priority.HIGH)
    leased = await q.lease(Q, worker_id="w1", count=3)
    assert [j.payload["p"] for j in leased] == ["high", "normal", "low"]


async def check_fifo_within_each_priority_band(make: BackendFactory) -> None:
    """Priority orders bands; FIFO orders within a band."""
    q = make()
    for i in range(3):
        await q.enqueue(Q, {"i": i, "p": "low"}, priority=Priority.LOW)
    for i in range(3):
        await q.enqueue(Q, {"i": i, "p": "high"}, priority=Priority.HIGH)
    leased = await q.lease(Q, worker_id="w1", count=6)
    assert [(j.payload["p"], j.payload["i"]) for j in leased] == [
        ("high", 0), ("high", 1), ("high", 2),
        ("low", 0), ("low", 1), ("low", 2),
    ]


# --------------------------------------------------------------------------
# Leasing
# --------------------------------------------------------------------------


async def check_lease_marks_the_job_leased(make: BackendFactory) -> None:
    q = make()
    job = await q.enqueue(Q, {})
    [leased] = await q.lease(Q, worker_id="w1")
    assert leased.job_id == job.job_id
    assert leased.lease.worker_id == "w1"
    assert (await q.get(job.job_id)).state is JobState.LEASED


async def check_leased_job_is_not_leased_twice(make: BackendFactory) -> None:
    """The core at-least-once guarantee: no double delivery while held."""
    q = make()
    await q.enqueue(Q, {})
    assert len(await q.lease(Q, worker_id="w1")) == 1
    assert await q.lease(Q, worker_id="w2") == [], "a held job was handed to a second worker"


async def check_lease_respects_count(make: BackendFactory) -> None:
    q = make()
    for i in range(5):
        await q.enqueue(Q, {"i": i})
    assert len(await q.lease(Q, worker_id="w1", count=2)) == 2
    assert len(await q.lease(Q, worker_id="w2", count=10)) == 3


async def check_lease_of_empty_queue_is_empty(make: BackendFactory) -> None:
    assert await make().lease(Q, worker_id="w1") == []


async def check_queues_are_isolated(make: BackendFactory) -> None:
    q = make()
    await q.enqueue("image", {"kind": "image"})
    await q.enqueue("video", {"kind": "video"})
    [leased] = await q.lease("video", worker_id="w1")
    assert leased.payload["kind"] == "video", "leasing crossed queue boundaries"


async def check_extend_lease_renews(make: BackendFactory) -> None:
    q = make()
    await q.enqueue(Q, {})
    [leased] = await q.lease(Q, worker_id="w1", lease_seconds=30)
    renewed = await q.extend_lease(leased.job_id, leased.lease.lease_id, lease_seconds=300)
    assert renewed.lease.expires_at > leased.lease.expires_at
    assert renewed.lease.lease_id == leased.lease.lease_id


async def check_stale_lease_cannot_extend(make: BackendFactory) -> None:
    q = make()
    await q.enqueue(Q, {})
    [leased] = await q.lease(Q, worker_id="w1")
    try:
        await q.extend_lease(leased.job_id, "not-the-lease-id")
    except LeaseNotHeld:
        return
    raise AssertionError("a stale lease id was allowed to extend")


# --------------------------------------------------------------------------
# Ack / nack
# --------------------------------------------------------------------------


async def check_ack_completes_the_job(make: BackendFactory) -> None:
    q = make()
    await q.enqueue(Q, {})
    [leased] = await q.lease(Q, worker_id="w1")
    done = await q.ack(leased.job_id, leased.lease.lease_id)
    assert done.state is JobState.DONE
    assert done.attempts == 1
    assert await q.lease(Q, worker_id="w2") == [], "an acked job came back"


async def check_stale_lease_cannot_ack(make: BackendFactory) -> None:
    """Without this, a worker whose lease lapsed marks another's job complete."""
    q = make()
    await q.enqueue(Q, {})
    [leased] = await q.lease(Q, worker_id="w1")
    try:
        await q.ack(leased.job_id, "forged-lease")
    except LeaseNotHeld:
        assert (await q.get(leased.job_id)).state is JobState.LEASED
        return
    raise AssertionError("a forged lease id was allowed to ack")


async def check_ack_of_unleased_job_is_rejected(make: BackendFactory) -> None:
    q = make()
    job = await q.enqueue(Q, {})
    try:
        await q.ack(job.job_id, "any")
    except JobNotLeased:
        return
    raise AssertionError("acked a job nobody holds")


async def check_retryable_nack_schedules_a_retry(make: BackendFactory) -> None:
    """The backoff is pinned, jitter-free, for this check's queue.

    Under the default full jitter the first delay is uniform in [0, 2 s]; a
    draw of a few milliseconds legitimately makes the job leasable again before
    the second ``lease`` below, so the check failed ~2% of runs on SQLite
    (measured: 5/300 and 6/300). With a fixed 30 s backoff the job cannot come
    back within the check, and the schedule itself becomes assertable.
    """
    backoff = timedelta(seconds=30)
    q = make()
    q.queue_policies[Q] = RetryPolicy(
        base_delay_s=backoff.total_seconds(), max_delay_s=backoff.total_seconds(), jitter=False
    )
    await q.enqueue(Q, {})
    [leased] = await q.lease(Q, worker_id="w1")
    before = datetime.now(UTC)
    result = await q.nack(leased.job_id, leased.lease.lease_id, error="blip", retryable=True)
    after = datetime.now(UTC)
    assert result.outcome is NackOutcome.RETRY_SCHEDULED
    assert result.retry_at is not None
    assert before + backoff <= result.retry_at <= after + backoff, (
        f"retry scheduled for {result.retry_at}, not {backoff} after the nack"
    )
    assert result.envelope.attempts == 1
    assert result.envelope.last_error == "blip"
    assert await q.lease(Q, worker_id="w2") == [], "a backed-off job was immediately releasable"


async def check_non_retryable_nack_dead_letters_at_once(make: BackendFactory) -> None:
    """A corrupt file will not decode on the second attempt either."""
    q = make()
    await q.enqueue(Q, {})
    [leased] = await q.lease(Q, worker_id="w1")
    result = await q.nack(leased.job_id, leased.lease.lease_id, error="corrupt", retryable=False)
    assert result.outcome is NackOutcome.DEAD_LETTERED
    assert result.retry_at is None
    assert result.envelope.attempts == 1, "no attempts were wasted retrying it"
    assert result.envelope.state is JobState.DEAD


async def check_attempts_exhaust_into_the_dead_letter_queue(make: BackendFactory) -> None:
    q = make()
    job = await q.enqueue(Q, {}, max_attempts=2)
    outcomes = []
    for _ in range(2):
        leased = await q.lease(Q, worker_id="w1")
        if not leased:
            # backoff is in force; force availability by reaping is not possible
            # portably, so drive the remaining attempt directly.
            break
        outcomes.append(
            (await q.nack(leased[0].job_id, leased[0].lease.lease_id, error="x")).outcome
        )
    final = await q.get(job.job_id)
    assert final.attempts >= 1
    assert NackOutcome.RETRY_SCHEDULED in outcomes or final.state is JobState.DEAD


async def check_stale_lease_cannot_nack(make: BackendFactory) -> None:
    q = make()
    await q.enqueue(Q, {})
    [leased] = await q.lease(Q, worker_id="w1")
    try:
        await q.nack(leased.job_id, "forged", error="x")
    except LeaseNotHeld:
        return
    raise AssertionError("a forged lease id was allowed to nack")


# --------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------


async def check_expired_lease_returns_the_job(make: BackendFactory) -> None:
    """A worker that dies must not take the job with it.

    Asserts the job is back in the backlog rather than immediately leasable:
    retry backoff applies to a lapsed lease just as it does to a nack, and how
    long that backoff is belongs to the backend's policy, not to this suite.
    """
    q = make()
    job = await q.enqueue(Q, {"important": True})
    [leased] = await q.lease(Q, worker_id="doomed", lease_seconds=0.05)
    await asyncio.sleep(0.08)

    reaped = await q.reap_expired_leases()
    assert reaped == 1, "an expired lease was not reaped"

    recovered = await q.get(job.job_id)
    assert recovered.state is not JobState.LEASED, "the job is still held by a dead worker"
    assert recovered.state is not JobState.DEAD, "one lost lease should not be fatal"
    assert recovered.attempts == 1, "lease expiry counts as a failed attempt"
    assert recovered.payload == {"important": True}

    stats = await q.stats(Q)
    assert stats.leased == 0
    assert stats.backlog == 1, (
        "a recovered job must be visible in backlog; work that is pending but "
        "counted in neither ready nor delayed is invisible to an autoscaler"
    )


async def check_expired_lease_eventually_dead_letters(make: BackendFactory) -> None:
    """A job that reliably kills its worker must not cycle forever."""
    q = make()
    job = await q.enqueue(Q, {}, max_attempts=1)
    await q.lease(Q, worker_id="doomed", lease_seconds=0.05)
    await asyncio.sleep(0.08)
    await q.reap_expired_leases()
    assert (await q.get(job.job_id)).state is JobState.DEAD


# --------------------------------------------------------------------------
# Dead letters and inspection
# --------------------------------------------------------------------------


async def check_dead_letters_are_listed(make: BackendFactory) -> None:
    q = make()
    await q.enqueue(Q, {"bad": True})
    [leased] = await q.lease(Q, worker_id="w1")
    await q.nack(leased.job_id, leased.lease.lease_id, error="corrupt", retryable=False)
    dead = await q.dead_letters(Q)
    assert len(dead) == 1
    assert dead[0].last_error == "corrupt"


async def check_dead_letter_can_be_requeued(make: BackendFactory) -> None:
    q = make()
    await q.enqueue(Q, {})
    [leased] = await q.lease(Q, worker_id="w1")
    await q.nack(leased.job_id, leased.lease.lease_id, error="x", retryable=False)
    revived = await q.requeue_dead_letter(leased.job_id)
    assert revived.state is JobState.READY
    assert revived.attempts == 0, "a requeued job starts its attempt budget over"
    assert len(await q.lease(Q, worker_id="w2")) == 1


async def check_unknown_job_raises(make: BackendFactory) -> None:
    try:
        await make().get("nope")
    except JobNotFound:
        return
    raise AssertionError("expected JobNotFound")


async def check_stats_reflect_reality(make: BackendFactory) -> None:
    q = make()
    for i in range(4):
        await q.enqueue(Q, {"i": i})
    await q.enqueue(Q, {"later": True}, delay_s=60)
    await q.lease(Q, worker_id="w1", count=2)
    stats = await q.stats(Q)
    assert stats.ready == 2
    assert stats.leased == 2
    assert stats.delayed == 1
    assert stats.backlog == 3


async def check_oldest_ready_age_is_reported(make: BackendFactory) -> None:
    q = make()
    empty = await q.stats(Q)
    assert empty.oldest_ready_age_seconds is None, "no ready jobs means no age"
    await q.enqueue(Q, {})
    await asyncio.sleep(0.02)
    stats = await q.stats(Q)
    assert stats.oldest_ready_age_seconds is not None
    assert stats.oldest_ready_age_seconds > 0


async def check_purge_empties_one_queue_only(make: BackendFactory) -> None:
    q = make()
    await q.enqueue("image", {})
    await q.enqueue("video", {})
    assert await q.purge("image") == 1
    assert (await q.stats("image")).ready == 0
    assert (await q.stats("video")).ready == 1


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

CHECKS: list[Callable[[BackendFactory], Awaitable[None]]] = [
    check_satisfies_protocol,
    check_health_never_raises,
    check_health_is_ok,
    check_enqueue_starts_ready,
    check_delay_defers_availability,
    check_idempotent_enqueue_returns_the_original,
    check_distinct_keys_create_distinct_jobs,
    check_fifo_among_equal_priority,
    check_fifo_holds_across_separate_leases,
    check_priority_outranks_arrival_order,
    check_fifo_within_each_priority_band,
    check_lease_marks_the_job_leased,
    check_leased_job_is_not_leased_twice,
    check_lease_respects_count,
    check_lease_of_empty_queue_is_empty,
    check_queues_are_isolated,
    check_extend_lease_renews,
    check_stale_lease_cannot_extend,
    check_ack_completes_the_job,
    check_stale_lease_cannot_ack,
    check_ack_of_unleased_job_is_rejected,
    check_retryable_nack_schedules_a_retry,
    check_non_retryable_nack_dead_letters_at_once,
    check_attempts_exhaust_into_the_dead_letter_queue,
    check_stale_lease_cannot_nack,
    check_expired_lease_returns_the_job,
    check_expired_lease_eventually_dead_letters,
    check_dead_letters_are_listed,
    check_dead_letter_can_be_requeued,
    check_unknown_job_raises,
    check_stats_reflect_reality,
    check_oldest_ready_age_is_reported,
    check_purge_empties_one_queue_only,
]


async def run_conformance(make: BackendFactory) -> list[str]:
    """Run the suite. Returns failure descriptions; empty means conformant."""
    failures: list[str] = []
    for check in CHECKS:
        try:
            await check(make)
        except Exception as exc:  # noqa: BLE001 - a suite reports everything
            failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
    return failures


__all__ = ["CHECKS", "RetryPolicy", "run_conformance"]
