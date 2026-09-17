"""Fusion semantics: the one-sided provenance rule, proved arithmetically.

``contracts/branch.py`` refuses a negative branch-P logit and
``ProvenanceReport`` bounds ``authenticity_credit`` to [0, 1]. Both guards are
correct and both were satisfied by an implementation that still let provenance
push a file toward "manipulated" — because the violation happened in the
*arithmetic* rather than in a field, and a field validator cannot see arithmetic.

This suite covers that gap. It asserts the properties directly, and then asserts
that the pre-Batch-0 implementations — reproduced verbatim below as
``_legacy_fuse`` and ``_legacy_disagreement`` — **fail** them. A property test
that has never rejected anything is a hypothesis; the legacy doubles are what
turn these into checks.

The two failure modes being pinned:

    F1  branch P's credit was averaged into log-odds of *fake*, so a valid C2PA
        signature raised p_fake. Cryptographic proof of authenticity made a file
        look more manipulated.

    F2  a *zero* credit still entered the renormalised mean and shrank the fused
        logit toward zero, so the mere absence of provenance — the state of
        almost every real upload — pushed confident scores toward the abstention
        band.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.common import Timing  # noqa: E402
from contracts.enums import AbstainReason, BranchId, MediaKind, Verdict  # noqa: E402
from contracts.evidence import EvidenceBundle  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from inference.calibration import inverse_sigmoid, sigmoid  # noqa: E402
from inference.models import ModelSpec  # noqa: E402
from inference.postprocess import (  # noqa: E402
    BranchOutcome,
    QualityFacts,
    authenticity_credit,
    build_image_bundle,
    calibrated_probability,
    disagreement_of,
    fuse,
    manipulation_evidence,
)

TIMING = Timing(queued_ms=0, preprocess_ms=1, inference_ms=1, total_ms=2)

WEIGHTS = {BranchId.A: 0.72, BranchId.B: 0.24, BranchId.P: 0.04}

# A spec whose provenance coefficient is large enough for its effect to be
# legible in a test. The shipped default of 0.04 is deliberately conservative
# and makes a correct effect too small to see; the *sign* of the effect is what
# these tests are about, and it must not depend on the magnitude.
VISIBLE_P = ModelSpec(
    name="visible-p",
    version="test-0",
    backend="stub",
    branch_weights={BranchId.A: 0.72, BranchId.B: 0.24, BranchId.P: 1.5},
)


# --------------------------------------------------------------------------
# The pre-Batch-0 implementations, kept as doubles
# --------------------------------------------------------------------------


def _legacy_fuse(
    outcomes: dict[BranchId, BranchOutcome], weights: dict[BranchId, float]
) -> float | None:
    """Fusion exactly as it stood before Batch 0. Averages branch P in."""
    total_weight = 0.0
    total = 0.0
    for branch, outcome in outcomes.items():
        if not outcome.scored:
            continue
        w = weights.get(branch, 0.0)
        if w <= 0.0:
            continue
        total += w * float(outcome.logit)
        total_weight += w
    if total_weight <= 0.0:
        return None
    return total / total_weight


def _legacy_disagreement(outcomes: dict[BranchId, BranchOutcome]) -> float:
    """Disagreement as it stood before Batch 0. Includes branch P."""
    ps = [float(sigmoid(o.logit)) for o in outcomes.values() if o.scored]
    return float(max(ps) - min(ps)) if len(ps) > 1 else 0.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _job(kind: MediaKind = MediaKind.IMAGE) -> JobSpec:
    return JobSpec(
        job_id=uuid4(),
        media_kind=kind,
        storage_key="uploads/x",
        sha256="b" * 64,
        size_bytes=1024,
        declared_mime="image/jpeg",
        model_version="test-0",
        idempotency_key="k",
        submitted_at=datetime.now(UTC),
    )


def _evidence(l_a: float | None = None, l_b: float | None = None) -> dict:
    out: dict[BranchId, BranchOutcome] = {}
    if l_a is not None:
        out[BranchId.A] = BranchOutcome(logit=l_a)
    if l_b is not None:
        out[BranchId.B] = BranchOutcome(logit=l_b)
    return out


def _with_credit(base: dict, credit: float) -> dict:
    return {**base, BranchId.P: BranchOutcome(logit=credit)}


def _bundle(outcomes: dict, spec: ModelSpec = VISIBLE_P) -> EvidenceBundle:
    return build_image_bundle(
        job=_job(),
        model=spec,
        size_bytes=1024,
        width=1920,
        height=1080,
        quality=QualityFacts(width=1920, height=1080, face_px=400),
        outcomes=outcomes,
        timing=TIMING,
    )


# --------------------------------------------------------------------------
# Property 1 — authenticity evidence can never increase fake probability
# --------------------------------------------------------------------------


def test_provenance_can_never_increase_the_fused_logit():
    """The unconditional guarantee, swept across the whole plausible domain.

    Not "usually lower" and not "lower for realistic values" — never higher, for
    every combination. IEEE-754 subtraction is correctly rounded and therefore
    monotone, so ``z - w*credit <= z`` holds for all finite z and all
    non-negative ``w*credit``. That is what makes this assertable as an equality
    of guarantees rather than as a tolerance.
    """
    for l_a in (-12.0, -4.0, -1.0, -0.001, 0.0, 0.001, 1.0, 4.0, 12.0):
        for l_b in (None, -8.0, 0.0, 8.0):
            base = _evidence(l_a, l_b)
            without = fuse(base, WEIGHTS)
            for credit in (0.0, 1e-9, 0.001, 0.25, 0.5, 1.0, 10.0):
                with_p = fuse(_with_credit(base, credit), WEIGHTS)
                assert with_p <= without, (
                    f"provenance raised the fused logit: L_A={l_a} L_B={l_b} "
                    f"credit={credit} -> {with_p} > {without}"
                )


def test_provenance_can_never_increase_the_calibrated_probability():
    """The same guarantee at the level users actually see."""
    for l_a in (-6.0, -2.0, 0.0, 2.0, 6.0):
        base = _evidence(l_a, l_a * 0.5)
        p_without = calibrated_probability(fuse(base, WEIGHTS), VISIBLE_P)
        for credit in (0.0, 0.1, 0.5, 1.0):
            p_with = calibrated_probability(
                fuse(_with_credit(base, credit), WEIGHTS), VISIBLE_P
            )
            assert p_with <= p_without, (
                f"provenance raised p_fake: L_A={l_a} credit={credit} "
                f"-> {p_with} > {p_without}"
            )


def test_legacy_fusion_raised_p_fake_on_valid_provenance():
    """F1, pinned. The old implementation fails the property above.

    This is the failure mode in its plainest form: a file carrying a verified
    C2PA signature scored *more* manipulated than the identical file with no
    signature at all.
    """
    base = _evidence(l_a=-2.0, l_b=-1.5)
    legacy_without = _legacy_fuse(base, WEIGHTS)
    legacy_with = _legacy_fuse(_with_credit(base, 1.0), WEIGHTS)
    assert legacy_with > legacy_without, (
        "the legacy double no longer reproduces F1; the regression it guards "
        "has stopped being demonstrable and this suite has gone vacuous"
    )
    # And the corrected implementation moves it the other way on the same input.
    assert fuse(_with_credit(base, 1.0), WEIGHTS) < fuse(base, WEIGHTS)


# --------------------------------------------------------------------------
# Property 2 — absent provenance is exactly neutral
# --------------------------------------------------------------------------


def test_absent_provenance_is_bit_identical_to_a_zero_credit():
    """Exact float equality, not a tolerance.

    "Contributes exactly zero" is a claim about identity. Asserting it within
    1e-12 would permit a drift that accumulates differently at different score
    magnitudes, which is precisely how a rule like this decays.
    """
    for l_a in (-9.0, -3.0, 0.0, 3.0, 9.0):
        for l_b in (None, -2.0, 5.0):
            base = _evidence(l_a, l_b)
            assert fuse(base, WEIGHTS) == fuse(_with_credit(base, 0.0), WEIGHTS)


def test_absent_provenance_does_not_dilute_a_confident_score():
    """F2, stated as the number a user would see.

    An authentic photograph with no metadata — the commonest upload there is —
    must score exactly as it would if the provenance branch did not exist.
    """
    base = _evidence(l_a=-3.0)
    p_no_branch = calibrated_probability(fuse(base, WEIGHTS), VISIBLE_P)
    p_zero_credit = calibrated_probability(
        fuse(_with_credit(base, 0.0), WEIGHTS), VISIBLE_P
    )
    assert p_no_branch == p_zero_credit
    assert abs(p_no_branch - float(sigmoid(-3.0))) < 1e-12, (
        "a lone branch-A logit of -3.0 must survive fusion untouched"
    )


def test_legacy_fusion_diluted_confident_scores_with_absent_provenance():
    """F2, pinned. Absence of metadata moved the score toward fake."""
    base = _evidence(l_a=-3.0)
    legacy_without = _legacy_fuse(base, WEIGHTS)
    legacy_with = _legacy_fuse(_with_credit(base, 0.0), WEIGHTS)
    assert legacy_with > legacy_without, (
        "the legacy double no longer reproduces F2; this suite has gone vacuous"
    )
    p_legacy = float(sigmoid(legacy_with))
    p_correct = float(sigmoid(fuse(_with_credit(base, 0.0), WEIGHTS)))
    assert p_legacy > p_correct, (
        f"legacy dilution raised p_fake from {p_correct} to {p_legacy} on an "
        "authentic image whose only offence was carrying no metadata"
    )


# --------------------------------------------------------------------------
# Property 3 — conflicting signals
# --------------------------------------------------------------------------


def test_conflicting_signals_move_the_score_toward_authentic():
    """Strong manipulation evidence plus a valid signature.

    The score must fall, because the signature is real evidence — but it is
    fused, not overriding.
    """
    base = _evidence(l_a=3.0, l_b=2.5)
    p_unsigned = calibrated_probability(fuse(base, WEIGHTS), VISIBLE_P)
    p_signed = calibrated_probability(fuse(_with_credit(base, 1.0), WEIGHTS), VISIBLE_P)
    assert p_signed < p_unsigned
    assert p_signed > 0.5, (
        "a bounded credit must not overturn strong manipulation evidence; "
        "provenance is a contribution, not a veto"
    )


def test_provenance_cannot_veto_overwhelming_manipulation_evidence():
    """The bound that stops a stolen or replayed manifest from laundering a fake.

    C2PA proves a manifest was signed, not that the pixels are untouched. A
    credit large enough to flip a confident detection would make the signature a
    bypass, so the verdict must survive the maximum credit the contract permits.
    """
    strong = _evidence(l_a=8.0, l_b=7.0)
    bundle = _bundle(_with_credit(strong, 1.0))
    assert bundle.verdict is Verdict.MANIPULATED, (
        "maximum authenticity credit overturned overwhelming spatial evidence"
    )


def test_branches_disagreeing_is_reported_not_averaged_away():
    outcomes = _with_credit(_evidence(l_a=4.0, l_b=-4.0), 0.0)
    bundle = _bundle(outcomes)
    assert bundle.disagreement > 0.9, (
        "two branches at opposite extremes must surface as disagreement"
    )


# --------------------------------------------------------------------------
# Property 4 — provenance alone is not a verdict
# --------------------------------------------------------------------------


def test_provenance_only_produces_no_fused_score():
    """Nothing looked at the pixels, so there is nothing to modulate."""
    assert fuse({BranchId.P: BranchOutcome(logit=1.0)}, WEIGHTS) is None
    assert fuse({BranchId.P: BranchOutcome(logit=0.0)}, WEIGHTS) is None
    assert manipulation_evidence({BranchId.P: BranchOutcome(logit=1.0)}, WEIGHTS) is None


def test_provenance_only_bundle_abstains_with_a_null_probability():
    bundle = _bundle(
        {
            BranchId.A: BranchOutcome(abstain_reason=AbstainReason.NO_FACE_DETECTED),
            BranchId.B: BranchOutcome(abstain_reason=AbstainReason.NO_FACE_DETECTED),
            BranchId.P: BranchOutcome(logit=1.0),
        }
    )
    assert bundle.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert bundle.p_fake is None
    assert bundle.interval is None
    p_branch = bundle.branch(BranchId.P)
    assert p_branch.logit == 1.0, "the credit stays visible even though it decided nothing"


def test_legacy_fusion_invented_a_verdict_from_provenance_alone():
    """The old code returned 0.0 — a fifty-fifty manipulation claim about media
    no branch had examined."""
    legacy = _legacy_fuse({BranchId.P: BranchOutcome(logit=0.0)}, WEIGHTS)
    assert legacy == 0.0, "the legacy double no longer reproduces the defect"
    assert float(sigmoid(legacy)) == 0.5
    assert fuse({BranchId.P: BranchOutcome(logit=0.0)}, WEIGHTS) is None


def test_spec_rejects_weights_that_can_never_produce_evidence():
    try:
        ModelSpec(name="p-only", version="v", branch_weights={BranchId.P: 1.0})
    except ValueError as exc:
        assert "manipulation branch" in str(exc)
        return
    raise AssertionError("a provenance-only weighting was accepted")


# --------------------------------------------------------------------------
# Property 5 — branch P stays off the probability scale
# --------------------------------------------------------------------------


def test_disagreement_ignores_branch_p():
    agreeing = _evidence(l_a=-3.0, l_b=-2.5)
    assert disagreement_of(_with_credit(agreeing, 0.0)) == disagreement_of(agreeing)
    assert disagreement_of(_with_credit(agreeing, 1.0)) == disagreement_of(agreeing)


def test_legacy_disagreement_was_inflated_by_an_empty_provenance_branch():
    """The measured consequence: a ~6x over-wide interval on every upload."""
    agreeing = _evidence(l_a=-3.0, l_b=-2.5)
    real = disagreement_of(agreeing)
    legacy = _legacy_disagreement(_with_credit(agreeing, 0.0))
    assert real < 0.05, f"two agreeing branches should barely disagree, got {real}"
    assert legacy > 0.4, (
        "the legacy double no longer reproduces the inflation; suite is vacuous"
    )
    assert legacy > real * 10


def test_branch_p_publishes_no_probability():
    bundle = _bundle(_with_credit(_evidence(l_a=1.0, l_b=1.0), 0.0))
    p_branch = bundle.branch(BranchId.P)
    assert p_branch.logit == 0.0
    assert p_branch.p is None, (
        "branch P published a probability; sigmoid(credit) renders 'no "
        "provenance' as a confident-looking 50% manipulated"
    )
    for bid in (BranchId.A, BranchId.B):
        assert bundle.branch(bid).p is not None, "manipulation branches keep their p"


def test_authenticity_credit_is_floored_at_zero_at_the_arithmetic():
    assert authenticity_credit({BranchId.P: BranchOutcome(logit=-5.0)}) == 0.0
    assert authenticity_credit({}) == 0.0
    assert authenticity_credit(
        {BranchId.P: BranchOutcome(abstain_reason=AbstainReason.NO_PROVENANCE_DATA)}
    ) == 0.0
    assert authenticity_credit({BranchId.P: BranchOutcome(logit=0.75)}) == 0.75


def test_a_negative_credit_cannot_reach_the_fused_score():
    """Defence in depth: the clamp lives at the arithmetic, not only at the field.

    A backend emitting a negative branch-P value would otherwise be adding
    positive evidence of manipulation through a double negative.
    """
    base = _evidence(l_a=0.0)
    assert fuse(_with_credit(base, -5.0), WEIGHTS) == fuse(base, WEIGHTS)


# --------------------------------------------------------------------------
# Numerical demonstration
# --------------------------------------------------------------------------


def demonstrate() -> None:
    """Print the before/after table. Called by the report, not by the suite."""
    rows = [
        ("authentic photo, no metadata", -3.0, None, 0.0),
        ("authentic photo, valid C2PA", -3.0, None, 1.0),
        ("borderline, no metadata", 0.5, 0.3, 0.0),
        ("borderline, valid C2PA", 0.5, 0.3, 1.0),
        ("clear fake, no metadata", 3.0, 2.5, 0.0),
        ("clear fake, stolen C2PA", 3.0, 2.5, 1.0),
    ]
    print(f"{'scenario':<32} {'p_fake before':>14} {'p_fake after':>13} {'delta':>9}")
    print("-" * 72)
    for label, l_a, l_b, credit in rows:
        outcomes = _with_credit(_evidence(l_a, l_b), credit)
        before = float(sigmoid(_legacy_fuse(outcomes, WEIGHTS)))
        after = float(sigmoid(fuse(outcomes, WEIGHTS)))
        print(f"{label:<32} {before:>14.6f} {after:>13.6f} {after - before:>+9.6f}")


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
