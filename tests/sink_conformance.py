"""Reusable ``ResultSink`` conformance suite.

Not a test module — a suite any sink runs through, mirroring the storage and
queue suites. A future Redis- or Postgres-backed sink imports this and proves it
honours the same contract rather than growing its own tests that quietly assert
weaker guarantees.

The contract being enforced is the authority rule from audit item D-3:

    missing  -> present            allowed
    present  -> different present  forbidden
    present  -> overwrite          forbidden

A committed result is immutable. ``put_if_absent`` decides who commits, and
losing is an ordinary outcome rather than an exception — at-least-once delivery
makes duplicate execution expected, so the losing path is the common path and
must not look like an error.

Usage::

    failures = asyncio.run(run_conformance(lambda: MySink()))
    assert failures == []
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from contracts import fixtures as fx
from contracts.evidence import EvidenceBundle

SinkFactory = Callable[[], Any]


def _bundle(job_id, *, disagreement: float | None = None) -> EvidenceBundle:
    update: dict[str, Any] = {"job_id": job_id}
    if disagreement is not None:
        update["disagreement"] = disagreement
    return fx.image_manipulated().model_copy(update=update)


# --------------------------------------------------------------------------
# Basic behaviour
# --------------------------------------------------------------------------


async def check_find_returns_none_when_absent(make: SinkFactory) -> None:
    assert await make().find(str(uuid4())) is None


async def check_first_write_wins_the_key(make: SinkFactory) -> None:
    sink = make()
    job_id = str(uuid4())
    assert await sink.put_if_absent(job_id, _bundle(job_id)) is True


async def check_second_write_loses(make: SinkFactory) -> None:
    sink = make()
    job_id = str(uuid4())
    await sink.put_if_absent(job_id, _bundle(job_id))
    assert await sink.put_if_absent(job_id, _bundle(job_id)) is False


async def check_losing_returns_false_rather_than_raising(make: SinkFactory) -> None:
    """The common path under at-least-once delivery must not be the error path."""
    sink = make()
    job_id = str(uuid4())
    await sink.put_if_absent(job_id, _bundle(job_id))
    try:
        result = await sink.put_if_absent(job_id, _bundle(job_id))
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"contention raised {type(exc).__name__}") from exc
    assert result is False


async def check_committed_result_is_readable(make: SinkFactory) -> None:
    sink = make()
    job_id = str(uuid4())
    bundle = _bundle(job_id)
    await sink.put_if_absent(job_id, bundle)
    stored = await sink.find(job_id)
    assert stored is not None
    assert stored.p_fake == bundle.p_fake
    EvidenceBundle.model_validate(stored.model_dump(mode="json"))


# --------------------------------------------------------------------------
# Immutability
# --------------------------------------------------------------------------


async def check_a_losing_write_does_not_change_the_stored_result(
    make: SinkFactory,
) -> None:
    """``present -> different present`` must be impossible.

    Uses genuinely different payloads: with identical ones an overwriting sink
    would pass, since the result looks the same either way.
    """
    sink = make()
    job_id = str(uuid4())
    first = _bundle(job_id, disagreement=0.111)
    second = _bundle(job_id, disagreement=0.999)

    assert await sink.put_if_absent(job_id, first) is True
    assert await sink.put_if_absent(job_id, second) is False

    stored = await sink.find(job_id)
    assert stored.disagreement == first.disagreement, (
        f"the second writer overwrote the first: stored {stored.disagreement}"
    )


async def check_repeated_losing_writes_never_take_hold(make: SinkFactory) -> None:
    sink = make()
    job_id = str(uuid4())
    await sink.put_if_absent(job_id, _bundle(job_id, disagreement=0.5))
    for i in range(10):
        assert await sink.put_if_absent(job_id, _bundle(job_id, disagreement=i / 10)) is False
    assert (await sink.find(job_id)).disagreement == 0.5


async def check_distinct_jobs_do_not_collide(make: SinkFactory) -> None:
    sink = make()
    a, b = str(uuid4()), str(uuid4())
    assert await sink.put_if_absent(a, _bundle(a, disagreement=0.1)) is True
    assert await sink.put_if_absent(b, _bundle(b, disagreement=0.9)) is True
    assert (await sink.find(a)).disagreement == 0.1
    assert (await sink.find(b)).disagreement == 0.9


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


async def check_two_concurrent_writers_produce_one_winner(make: SinkFactory) -> None:
    sink = make()
    job_id = str(uuid4())
    outcomes = await asyncio.gather(
        sink.put_if_absent(job_id, _bundle(job_id, disagreement=0.2)),
        sink.put_if_absent(job_id, _bundle(job_id, disagreement=0.8)),
    )
    assert sum(outcomes) == 1, f"expected exactly one winner, got {outcomes}"


async def check_many_concurrent_writers_produce_one_winner(make: SinkFactory) -> None:
    """The stress case. A check-then-write sink passes with two racers often
    enough to look correct and fails reliably at scale."""
    sink = make()
    job_id = str(uuid4())
    n = 24
    outcomes = await asyncio.gather(
        *(sink.put_if_absent(job_id, _bundle(job_id, disagreement=i / n)) for i in range(n))
    )
    assert sum(outcomes) == 1, f"{sum(outcomes)} of {n} writers claimed the key"

    stored = await sink.find(job_id)
    assert stored is not None
    winner = outcomes.index(True)
    assert abs(stored.disagreement - winner / n) < 1e-12, (
        "the stored result is not the one the winning writer submitted"
    )


async def check_concurrent_writes_to_distinct_jobs_all_succeed(
    make: SinkFactory,
) -> None:
    """Exclusivity must be per key, not a global lock that serialises everything
    into a single winner."""
    sink = make()
    ids = [str(uuid4()) for _ in range(12)]
    outcomes = await asyncio.gather(*(sink.put_if_absent(j, _bundle(j)) for j in ids))
    assert all(outcomes), f"only {sum(outcomes)} of {len(ids)} distinct jobs committed"


# --------------------------------------------------------------------------
# publish() built on put_if_absent
# --------------------------------------------------------------------------


async def check_publish_reports_a_fresh_write(make: SinkFactory) -> None:
    sink = make()
    job_id = uuid4()
    _, already = await sink.publish(_bundle(job_id))
    assert already is False


async def check_publish_reports_a_duplicate_and_returns_the_original(
    make: SinkFactory,
) -> None:
    sink = make()
    job_id = uuid4()
    first = _bundle(job_id, disagreement=0.123)
    await sink.publish(first)
    returned, already = await sink.publish(_bundle(job_id, disagreement=0.876))
    assert already is True
    assert returned.disagreement == first.disagreement, (
        "publish returned the losing bundle rather than the committed one"
    )


async def check_publish_and_put_if_absent_agree(make: SinkFactory) -> None:
    """``publish`` must be built on ``put_if_absent``, not beside it.

    Two independent write paths mean the atomicity guarantee holds only for
    whichever one a caller happens to use.
    """
    sink = make()
    job_id = uuid4()
    await sink.publish(_bundle(job_id, disagreement=0.4))
    assert await sink.put_if_absent(str(job_id), _bundle(job_id, disagreement=0.6)) is False
    assert (await sink.find(str(job_id))).disagreement == 0.4


# --------------------------------------------------------------------------
# Content index
# --------------------------------------------------------------------------


async def check_content_lookup_finds_a_committed_result(make: SinkFactory) -> None:
    sink = make()
    job_id = uuid4()
    bundle = _bundle(job_id)
    await sink.put_if_absent(str(job_id), bundle)
    found = await sink.find_by_content(bundle.media.sha256, bundle.model.version)
    assert found is not None
    assert found.job_id == job_id


async def check_content_lookup_is_scoped_to_the_model_version(
    make: SinkFactory,
) -> None:
    """A new model must not serve a verdict computed by the old one."""
    sink = make()
    job_id = uuid4()
    bundle = _bundle(job_id)
    await sink.put_if_absent(str(job_id), bundle)
    assert await sink.find_by_content(bundle.media.sha256, "some-other-version") is None


async def check_content_index_points_at_the_winner(make: SinkFactory) -> None:
    """A loser must not repoint the cache at its own discarded bundle."""
    sink = make()
    first_id, second_id = uuid4(), uuid4()
    first = _bundle(first_id, disagreement=0.15)
    second = _bundle(second_id, disagreement=0.85)
    assert first.media.sha256 == second.media.sha256, "fixture must share content"

    await sink.put_if_absent(str(first_id), first)
    await sink.put_if_absent(str(second_id), second)

    found = await sink.find_by_content(first.media.sha256, first.model.version)
    assert found is not None
    assert found.job_id == first_id, "the content index moved to a later writer"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

CHECKS: list[Callable[[SinkFactory], Awaitable[None]]] = [
    check_find_returns_none_when_absent,
    check_first_write_wins_the_key,
    check_second_write_loses,
    check_losing_returns_false_rather_than_raising,
    check_committed_result_is_readable,
    check_a_losing_write_does_not_change_the_stored_result,
    check_repeated_losing_writes_never_take_hold,
    check_distinct_jobs_do_not_collide,
    check_two_concurrent_writers_produce_one_winner,
    check_many_concurrent_writers_produce_one_winner,
    check_concurrent_writes_to_distinct_jobs_all_succeed,
    check_publish_reports_a_fresh_write,
    check_publish_reports_a_duplicate_and_returns_the_original,
    check_publish_and_put_if_absent_agree,
    check_content_lookup_finds_a_committed_result,
    check_content_lookup_is_scoped_to_the_model_version,
    check_content_index_points_at_the_winner,
]


async def run_conformance(make: SinkFactory) -> list[str]:
    """Run the suite. Returns failure descriptions; empty means conformant."""
    failures: list[str] = []
    for check in CHECKS:
        try:
            await check(make)
        except Exception as exc:  # noqa: BLE001 - a suite reports everything
            failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
    return failures
