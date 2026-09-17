"""Postprocess tests: fusion, thresholding, abstention, bundle invariants.

``postprocess`` is the only module that constructs an ``EvidenceBundle``, so it
is the only place every contract invariant can be enforced — and therefore the
only place worth testing them at. The tests below assert both directions: that
valid inputs produce valid bundles, and that the contract rejects the
contradictions this module is built to make unrepresentable.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.common import ThresholdInfo, Timing  # noqa: E402
from contracts.enums import (  # noqa: E402
    AbstainReason,
    BranchId,
    BranchStatus,
    MediaKind,
    Verdict,
)
from contracts.evidence import EvidenceBundle  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from contracts.video import FrameScore  # noqa: E402
from inference.calibration import inverse_sigmoid, sigmoid  # noqa: E402
from inference.models import CalibrationSpec, ModelSpec  # noqa: E402
from inference.postprocess import (  # noqa: E402
    BranchOutcome,
    QualityFacts,
    build_image_bundle,
    build_video_analysis,
    build_video_bundle,
    calibrated_probability,
    confidence_interval,
    decide,
    disagreement_of,
    fuse,
    hysteresis_segments,
    topk_mean,
)

MODEL = ModelSpec(name="t", version="test-0", backend="stub")
TIMING = Timing(queued_ms=0, preprocess_ms=1, inference_ms=1, total_ms=2)


def _job(kind: MediaKind = MediaKind.IMAGE) -> JobSpec:
    return JobSpec(
        job_id=uuid4(),
        media_kind=kind,
        storage_key="uploads/x",
        sha256="a" * 64,
        size_bytes=1024,
        declared_mime="image/jpeg" if kind is MediaKind.IMAGE else "video/mp4",
        model_version="test-0",
        idempotency_key="k",
        submitted_at=datetime.now(UTC),
    )


def _outcomes(p_a: float = 0.9, p_b: float | None = None) -> dict[BranchId, BranchOutcome]:
    out = {
        BranchId.A: BranchOutcome(logit=float(inverse_sigmoid(p_a))),
        BranchId.P: BranchOutcome(logit=0.0),
    }
    if p_b is not None:
        out[BranchId.B] = BranchOutcome(logit=float(inverse_sigmoid(p_b)))
    return out


def _quality(**over) -> QualityFacts:
    base = {"width": 1920, "height": 1080, "face_px": 400}
    base.update(over)
    return QualityFacts(**base)


def _image(**over) -> EvidenceBundle:
    kwargs = {
        "job": _job(),
        "model": MODEL,
        "size_bytes": 1024,
        "width": 1920,
        "height": 1080,
        "quality": _quality(),
        "outcomes": _outcomes(),
        "timing": TIMING,
    }
    kwargs.update(over)
    return build_image_bundle(**kwargs)


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def test_fusion_is_a_weighted_mean():
    outcomes = {BranchId.A: BranchOutcome(logit=2.0), BranchId.B: BranchOutcome(logit=0.0)}
    weights = {BranchId.A: 0.75, BranchId.B: 0.25}
    assert abs(fuse(outcomes, weights) - 1.5) < 1e-12


def test_fusion_renormalises_over_scored_branches():
    """An abstaining branch must not drag the fused logit toward zero.

    A missing branch should reduce confidence — which interval widening
    handles — not act as a vote for 'authentic'.
    """
    weights = {BranchId.A: 0.72, BranchId.B: 0.24, BranchId.P: 0.04}
    only_a = {
        BranchId.A: BranchOutcome(logit=3.0),
        BranchId.B: BranchOutcome(abstain_reason=AbstainReason.NO_FACE_DETECTED),
    }
    assert abs(fuse(only_a, weights) - 3.0) < 1e-12, "abstention diluted the score"


def test_fusion_returns_none_when_nothing_scored():
    outcomes = {BranchId.A: BranchOutcome(abstain_reason=AbstainReason.BRANCH_ERROR)}
    assert fuse(outcomes, {BranchId.A: 1.0}) is None


def test_fusion_ignores_zero_weighted_branches():
    outcomes = {BranchId.A: BranchOutcome(logit=5.0), BranchId.B: BranchOutcome(logit=-5.0)}
    assert abs(fuse(outcomes, {BranchId.A: 1.0, BranchId.B: 0.0}) - 5.0) < 1e-12


def test_disagreement_is_zero_for_a_single_branch():
    assert disagreement_of({BranchId.A: BranchOutcome(logit=2.0)}) == 0.0


def test_disagreement_grows_with_spread():
    close = disagreement_of(
        {BranchId.A: BranchOutcome(logit=1.0), BranchId.B: BranchOutcome(logit=1.1)}
    )
    far = disagreement_of(
        {BranchId.A: BranchOutcome(logit=-4.0), BranchId.B: BranchOutcome(logit=4.0)}
    )
    assert far > close


# --------------------------------------------------------------------------
# Calibration and thresholds
# --------------------------------------------------------------------------


def test_calibrated_probability_is_bounded():
    for logit in (-1000.0, -5.0, 0.0, 5.0, 1000.0):
        p = calibrated_probability(logit, MODEL)
        assert 0.0 <= p <= 1.0, f"logit {logit} produced {p}"


def test_calibration_temperature_is_applied():
    hot = ModelSpec(
        name="hot", version="v", calibration=CalibrationSpec(temperature=5.0, n_samples=500)
    )
    assert calibrated_probability(3.0, hot) < calibrated_probability(3.0, MODEL)


BAND = ThresholdInfo.at_margin(value=0.62, margin=0.4299, target_fpr=0.02)


def test_decide_at_and_around_the_band_edges():
    """The decision boundary is band_hi, not value. See ThresholdInfo."""
    assert decide(BAND.band_hi, BAND)[0] is Verdict.MANIPULATED, "the edge decides"
    assert decide(BAND.band_hi - 1e-9, BAND)[0] is Verdict.INSUFFICIENT_EVIDENCE
    assert decide(BAND.value, BAND)[0] is Verdict.INSUFFICIENT_EVIDENCE
    assert decide(BAND.band_lo, BAND)[0] is Verdict.INSUFFICIENT_EVIDENCE
    assert decide(BAND.band_lo - 1e-9, BAND)[0] is Verdict.AUTHENTIC
    assert decide(0.0, BAND)[0] is Verdict.AUTHENTIC
    assert decide(1.0, BAND)[0] is Verdict.MANIPULATED


def test_decide_partitions_the_unit_interval():
    """Every probability lands in exactly one region — no gap, no overlap."""
    seen = {}
    for i in range(2001):
        p = i / 2000.0
        verdict, reason = decide(p, BAND)
        seen[p] = verdict
        assert (reason is not None) == (verdict is Verdict.INSUFFICIENT_EVIDENCE)
    verdicts = list(seen.values())
    # Monotone: authentic, then abstention, then manipulated. Never back again.
    order = [Verdict.AUTHENTIC, Verdict.INSUFFICIENT_EVIDENCE, Verdict.MANIPULATED]
    ranks = [order.index(v) for v in verdicts]
    assert ranks == sorted(ranks), "the decision regions are not contiguous"
    assert set(verdicts) == set(order), "some region is unreachable"


def test_band_abstention_carries_its_reason():
    verdict, reason = decide(BAND.value, BAND)
    assert verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert reason is AbstainReason.SCORE_IN_ABSTENTION_BAND


def test_interval_brackets_the_estimate_and_stays_bounded():
    for p in (0.0, 0.01, 0.5, 0.99, 1.0):
        lo, hi = confidence_interval(p, disagreement=0.4)
        assert 0.0 <= lo <= p <= hi <= 1.0


def test_interval_widens_with_disagreement():
    narrow = confidence_interval(0.5, 0.0)
    wide = confidence_interval(0.5, 0.6)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


# --------------------------------------------------------------------------
# Bundle invariants
# --------------------------------------------------------------------------


def test_bundle_validates_through_contracts():
    bundle = _image()
    EvidenceBundle.model_validate(bundle.model_dump(mode="json"))


def test_verdict_is_derivable_from_probability_and_threshold():
    for p_a in (0.05, 0.3, 0.5, 0.7, 0.99):
        bundle = _image(outcomes=_outcomes(p_a=p_a))
        if bundle.p_fake is None:
            assert bundle.verdict is Verdict.INSUFFICIENT_EVIDENCE
            continue
        expected = (
            Verdict.MANIPULATED
            if bundle.p_fake >= bundle.threshold.value
            else Verdict.AUTHENTIC
        )
        assert bundle.verdict is expected


def test_p_fake_is_null_exactly_when_abstaining():
    decided = _image(outcomes=_outcomes(p_a=0.99))
    assert decided.verdict is not Verdict.INSUFFICIENT_EVIDENCE
    assert decided.p_fake is not None

    gated = _image(
        quality=_quality(face_px=10),
        forced_abstention=AbstainReason.FACE_TOO_SMALL,
        outcomes={BranchId.A: BranchOutcome(abstain_reason=AbstainReason.FACE_TOO_SMALL)},
    )
    assert gated.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert gated.p_fake is None and gated.interval is None


def test_band_abstention_discards_the_number_it_computed():
    """Publishing a probability beside 'insufficient evidence' invites a reader
    to use it anyway."""
    bundle = _image(outcomes=_outcomes(p_a=0.62, p_b=0.62))
    assert bundle.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert bundle.p_fake is None
    assert bundle.abstain_reason is AbstainReason.SCORE_IN_ABSTENTION_BAND
    assert any(b.status is BranchStatus.SCORED for b in bundle.branches), (
        "branch scores stay visible even when the headline number does not"
    )
    assert bundle.quality.gate_passed is True, "the gate passed; the score was ambiguous"


def test_branch_ids_are_unique_and_complete():
    ids = [b.id for b in _image().branches]
    assert len(ids) == len(set(ids))
    assert set(ids) == {BranchId.A, BranchId.B, BranchId.P}


def test_provenance_logit_is_clamped_non_negative():
    """The one-sided rule, enforced at construction rather than trusted."""
    bundle = _image(
        outcomes={
            BranchId.A: BranchOutcome(logit=1.0),
            BranchId.P: BranchOutcome(logit=-5.0),
        }
    )
    p_branch = next(b for b in bundle.branches if b.id is BranchId.P)
    assert p_branch.logit == 0.0, "a negative provenance logit survived"


def test_model_spec_rejects_a_negative_provenance_weight():
    """The other half of the rule: a negative weight on a non-negative logit
    reintroduces exactly the shortcut."""
    try:
        ModelSpec(name="bad", version="v", branch_weights={BranchId.A: 1.0, BranchId.P: -0.1})
    except ValueError as exc:
        assert "non-negative" in str(exc)
        return
    raise AssertionError("a negative branch-P weight was accepted")


def test_missing_branch_is_reported_unavailable_not_omitted():
    bundle = _image(outcomes={BranchId.A: BranchOutcome(logit=2.0)})
    b_branch = next(b for b in bundle.branches if b.id is BranchId.B)
    assert b_branch.status is BranchStatus.UNAVAILABLE
    assert b_branch.logit is None


def test_image_bundle_has_no_video_block():
    assert _image().video is None


def test_video_bundle_carries_its_analysis():
    scores = [FrameScore(frame_idx=i, t_ms=i * 40, p=0.1) for i in range(10)]
    bundle = build_video_bundle(
        job=_job(MediaKind.VIDEO),
        model=MODEL,
        size_bytes=2048,
        width=640,
        height=480,
        duration_s=0.4,
        frame_count=10,
        fps=25.0,
        quality=_quality(width=640, height=480),
        outcomes=_outcomes(p_a=0.1),
        video=build_video_analysis(
            frames_sampled=10,
            frames_scored=10,
            frame_scores=scores,
            threshold=MODEL.threshold,
        ),
        timing=TIMING,
    )
    EvidenceBundle.model_validate(bundle.model_dump(mode="json"))
    assert bundle.video is not None
    assert bundle.media.kind is MediaKind.VIDEO
    assert bundle.media.duration_s == 0.4


def test_gate_failure_forces_abstention_and_records_the_reason():
    bundle = _image(
        quality=_quality(gate_passed=False, gate_reason=AbstainReason.NO_FACE_DETECTED),
        outcomes={BranchId.A: BranchOutcome(abstain_reason=AbstainReason.NO_FACE_DETECTED)},
        forced_abstention=AbstainReason.NO_FACE_DETECTED,
    )
    assert bundle.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert bundle.quality.gate_passed is False
    assert bundle.quality.gate_reason is AbstainReason.NO_FACE_DETECTED
    EvidenceBundle.model_validate(bundle.model_dump(mode="json"))


def test_no_scored_branches_abstains_rather_than_guessing():
    bundle = _image(
        outcomes={BranchId.A: BranchOutcome(abstain_reason=AbstainReason.BRANCH_ERROR)}
    )
    assert bundle.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert bundle.p_fake is None


# --------------------------------------------------------------------------
# The contract rejects what this module makes unrepresentable
# --------------------------------------------------------------------------


def test_contract_rejects_a_verdict_contradicting_its_probability():
    payload = _image(outcomes=_outcomes(p_a=0.99)).model_dump(mode="json")
    payload["verdict"] = "authentic"  # p_fake stays high
    try:
        EvidenceBundle.model_validate(payload)
    except Exception as exc:
        assert "contradicts" in str(exc)
        return
    raise AssertionError("a contradictory verdict validated")


def test_contract_rejects_a_probability_on_an_abstention():
    payload = _image(
        outcomes={BranchId.A: BranchOutcome(abstain_reason=AbstainReason.BRANCH_ERROR)}
    ).model_dump(mode="json")
    payload["p_fake"] = 0.5
    try:
        EvidenceBundle.model_validate(payload)
    except Exception as exc:
        assert "must be null" in str(exc)
        return
    raise AssertionError("an abstention carrying a probability validated")


def test_contract_rejects_a_video_block_on_an_image():
    payload = _image().model_dump(mode="json")
    payload["video"] = {
        "frames_sampled": 1,
        "frames_scored": 1,
        "sampling": "uniform",
        "pooling": "topk_mean",
        "frame_scores": [],
        "segments": [],
        "top_frames": [],
    }
    try:
        EvidenceBundle.model_validate(payload)
    except Exception as exc:
        assert "must be null when media.kind is image" in str(exc)
        return
    raise AssertionError("an image bundle carrying video analysis validated")


# --------------------------------------------------------------------------
# Temporal aggregation
# --------------------------------------------------------------------------


def test_topk_mean_preserves_a_short_burst():
    """The reason pooling is not a plain mean.

    Six high frames in forty-eight: the mean is ~0.16 and reads as authentic;
    top-k keeps the burst visible.
    """
    import numpy as np

    scores = np.full(48, 0.05)
    scores[20:26] = 0.95
    assert float(np.mean(scores)) < 0.2
    assert topk_mean(scores) > 0.5


def test_topk_mean_rejects_an_empty_curve():
    import numpy as np

    try:
        topk_mean(np.array([]))
    except ValueError:
        return
    raise AssertionError("pooling an empty curve was accepted")


def test_hysteresis_merges_a_wobbling_run():
    """A single cut shatters a boundary-hugging curve into noise."""
    scores = [0.1, 0.7, 0.5, 0.7, 0.1]
    frames = [FrameScore(frame_idx=i, t_ms=i * 40, p=p) for i, p in enumerate(scores)]
    segments = hysteresis_segments(frames, enter=0.62, exit_=0.42)
    assert len(segments) == 1, f"expected one merged run, got {len(segments)}"
    assert segments[0].frame_count == 3


def test_hysteresis_finds_nothing_in_a_quiet_curve():
    frames = [FrameScore(frame_idx=i, t_ms=i * 40, p=0.05) for i in range(20)]
    assert hysteresis_segments(frames, enter=0.62, exit_=0.42) == []


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
