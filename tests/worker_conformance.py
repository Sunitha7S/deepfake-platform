"""Reusable ``JobHandler`` conformance suite.

Not a test module — a suite any handler runs through. When the stub handlers are
replaced by real models, they import this and prove they still honour the
contract, rather than growing their own tests that quietly assert less.

The checks encode the properties the rest of the platform relies on:

* the returned bundle is contract-valid and carries the *job's* id;
* the same bytes produce the same verdict, so idempotency and crash recovery
  are testable;
* ``insufficient_evidence`` is returned as a value, never raised as an error;
* branch P never contributes negative evidence;
* progress is reported, and for long handlers reported in batches.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from contracts.enums import (
    AbstainReason,
    BranchId,
    BranchStatus,
    JobStage,
    MediaKind,
    Verdict,
)
from contracts.evidence import EvidenceBundle
from workers.protocol import JobHandler
from workers.types import JobRequest

HandlerFactory = Callable[[], Any]
RequestFactory = Callable[[bytes], JobRequest]


class RecordingProgress:
    """Captures progress calls so a suite can assert on them."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, JobStage | None]] = []

    async def __call__(self, fraction: float, *, stage: JobStage | None = None) -> None:
        self.calls.append((fraction, stage))

    def __len__(self) -> int:
        return len(self.calls)


async def _run(
    make: HandlerFactory, request: JobRequest
) -> tuple[EvidenceBundle, RecordingProgress]:
    handler = make()
    progress = RecordingProgress()
    bundle = await handler.handle(request, progress)
    return bundle, progress


# --------------------------------------------------------------------------
# Structural
# --------------------------------------------------------------------------


async def check_satisfies_protocol(make: HandlerFactory, req: RequestFactory) -> None:
    handler = make()
    assert isinstance(handler, JobHandler), (
        f"{type(handler).__name__} does not satisfy JobHandler"
    )
    assert handler.name and handler.media_kind


async def check_returns_a_valid_bundle(make: HandlerFactory, req: RequestFactory) -> None:
    bundle, _ = await _run(make, req(b"conformance payload"))
    EvidenceBundle.model_validate(bundle.model_dump(mode="json"))


async def check_bundle_carries_the_job_id(make: HandlerFactory, req: RequestFactory) -> None:
    """A bundle stamped with the wrong id is filed against the wrong job."""
    request = req(b"identity check")
    bundle, _ = await _run(make, request)
    assert bundle.job_id == request.spec.job_id


async def check_bundle_matches_the_media_kind(make: HandlerFactory, req: RequestFactory) -> None:
    request = req(b"kind check")
    bundle, _ = await _run(make, request)
    assert bundle.media.kind is request.spec.media_kind
    if bundle.media.kind is MediaKind.VIDEO:
        assert bundle.video is not None, "a video bundle must carry its frame analysis"
    else:
        assert bundle.video is None


async def check_media_hash_is_carried_through(make: HandlerFactory, req: RequestFactory) -> None:
    request = req(b"hash check")
    bundle, _ = await _run(make, request)
    assert bundle.media.sha256 == request.spec.sha256


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


async def check_same_bytes_produce_the_same_verdict(
    make: HandlerFactory, req: RequestFactory
) -> None:
    """Crash recovery compares results across attempts; a handler that drifted
    would make those tests flap for reasons unrelated to what they check."""
    payload = b"determinism payload"
    first, _ = await _run(make, req(payload))
    second, _ = await _run(make, req(payload))
    assert first.verdict is second.verdict
    assert first.p_fake == second.p_fake
    assert [b.logit for b in first.branches] == [b.logit for b in second.branches]


async def check_different_bytes_can_differ(make: HandlerFactory, req: RequestFactory) -> None:
    """A handler returning one hardcoded answer would pass every other check."""
    scores = set()
    for i in range(40):
        bundle, _ = await _run(make, req(f"variation-{i}".encode()))
        scores.add(bundle.p_fake)
    assert len(scores) > 1, "the handler returns the same score for every input"


# --------------------------------------------------------------------------
# Abstention
# --------------------------------------------------------------------------


async def check_insufficient_evidence_is_returned_not_raised(
    make: HandlerFactory, req: RequestFactory
) -> None:
    """The whole point of the three-state verdict.

    Sweeps inputs until an abstention appears. If the handler never abstains
    that is acceptable; if it does, it must be a returned value.

    There are **two** kinds of abstention and they differ in exactly one field.
    A gate refusal — no face, face too small, quality below the floor — means
    the input was never scored, and ``gate_passed`` is False. A band abstention
    means the input passed the gate, was scored, and the score landed too close
    to the decision boundary to call; there ``gate_passed`` is True and the
    branch logits are published. Asserting ``gate_passed is False`` for every
    abstention conflated the two, and only passed because the stub handlers had
    no band logic at all and so could never produce the second kind.
    """
    for i in range(200):
        bundle, _ = await _run(make, req(f"abstain-probe-{i}".encode()))
        if bundle.verdict is Verdict.INSUFFICIENT_EVIDENCE:
            assert bundle.p_fake is None, "an abstention must not carry a probability"
            assert bundle.abstain_reason is not None
            if bundle.abstain_reason is AbstainReason.SCORE_IN_ABSTENTION_BAND:
                assert bundle.quality.gate_passed is True, (
                    "a band abstention means the gate passed and the score was "
                    "ambiguous; reporting a gate failure would tell the user the "
                    "wrong thing about their file"
                )
                assert any(b.status is BranchStatus.SCORED for b in bundle.branches), (
                    "branch scores stay visible when only the headline number is withheld"
                )
            else:
                assert bundle.quality.gate_passed is False, (
                    f"abstention {bundle.abstain_reason} is a gate refusal and "
                    "must report gate_passed=False"
                )
            return


async def check_decided_verdicts_carry_a_probability(
    make: HandlerFactory, req: RequestFactory
) -> None:
    for i in range(20):
        bundle, _ = await _run(make, req(f"decided-{i}".encode()))
        if bundle.verdict is not Verdict.INSUFFICIENT_EVIDENCE:
            assert bundle.p_fake is not None
            assert bundle.interval is not None
            lo, hi = bundle.interval
            assert lo <= bundle.p_fake <= hi


# --------------------------------------------------------------------------
# Branch discipline
# --------------------------------------------------------------------------


async def check_provenance_never_argues_for_fake(
    make: HandlerFactory, req: RequestFactory
) -> None:
    """The one-sided rule, enforced at the handler as well as in the contract."""
    for i in range(30):
        bundle, _ = await _run(make, req(f"provenance-{i}".encode()))
        p = bundle.branch(BranchId.P)
        if p is not None and p.status is BranchStatus.SCORED:
            assert p.logit is not None and p.logit >= 0.0, (
                "branch P emitted negative evidence; absence of provenance must "
                "contribute exactly zero"
            )


async def check_branch_c_is_never_emitted(make: HandlerFactory, req: RequestFactory) -> None:
    bundle, _ = await _run(make, req(b"branches"))
    assert {b.id for b in bundle.branches} <= {BranchId.A, BranchId.B, BranchId.P}


# --------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------


async def check_progress_is_reported(make: HandlerFactory, req: RequestFactory) -> None:
    """A handler that never reports progress will have its lease reclaimed."""
    _, progress = await _run(make, req(b"progress"))
    assert len(progress) >= 1, "the handler reported no progress at all"


async def check_progress_is_monotonic_and_bounded(
    make: HandlerFactory, req: RequestFactory
) -> None:
    _, progress = await _run(make, req(b"monotonic"))
    fractions = [f for f, _ in progress.calls]
    assert fractions == sorted(fractions), "progress went backwards"
    assert all(0.0 <= f <= 1.0 for f in fractions)


async def check_progress_is_batched_not_per_unit(
    make: HandlerFactory, req: RequestFactory
) -> None:
    """Progress is a liveness signal, not telemetry.

    A handler reporting once per frame would renew the lease dozens of times per
    job — on a Redis queue that is dozens of round trips for a signal that needs
    to fire a handful of times.
    """
    bundle, progress = await _run(make, req(b"batching"))
    units = bundle.video.frames_sampled if bundle.video else 1
    if units <= 4:
        return
    assert len(progress) < units, (
        f"progress called {len(progress)} times for {units} frames; batch them"
    )
    assert len(progress) <= max(4, units // 4 + 4), (
        f"progress called {len(progress)} times for {units} frames; the batch looks too small"
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

CHECKS: list[Callable[[HandlerFactory, RequestFactory], Awaitable[None]]] = [
    check_satisfies_protocol,
    check_returns_a_valid_bundle,
    check_bundle_carries_the_job_id,
    check_bundle_matches_the_media_kind,
    check_media_hash_is_carried_through,
    check_same_bytes_produce_the_same_verdict,
    check_different_bytes_can_differ,
    check_insufficient_evidence_is_returned_not_raised,
    check_decided_verdicts_carry_a_probability,
    check_provenance_never_argues_for_fake,
    check_branch_c_is_never_emitted,
    check_progress_is_reported,
    check_progress_is_monotonic_and_bounded,
    check_progress_is_batched_not_per_unit,
]


async def run_conformance(make: HandlerFactory, request_factory: RequestFactory) -> list[str]:
    """Run the suite. Returns failure descriptions; empty means conformant."""
    failures: list[str] = []
    for check in CHECKS:
        try:
            await check(make, request_factory)
        except Exception as exc:  # noqa: BLE001 - a suite reports everything
            failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
    return failures
