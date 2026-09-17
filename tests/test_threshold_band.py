"""F11 + R8 + R11 — the abstention band cannot drift from its threshold, and
cannot fail to exist.

TWO DEFECTS, ONE TYPE
---------------------
**F11 (Batch 1).** Three numbers were independently settable: ``value``,
``band_lo`` and ``band_hi``. ``decide`` tested ``p >= value`` first and the band
second, so any disagreement resolved silently in favour of whichever was checked
first, and two configurations were representable that should never have been:

    band_hi > value    the band is truncated — scores between ``value`` and
                       ``band_hi`` are decided as manipulated even though they
                       sit inside the region the operator marked as too close
                       to call.
    band_hi < value    a dead zone opens between them — scores just under the
                       decision boundary come back "authentic" while the band
                       claims to cover them.

**R11 (Batch 6, fixed here).** The Batch 1 fix derived the band from ``value``
and a *probability* width, and refused any band running outside [0, 1]. The
refusal was correct and the geometry was not: a width chosen independently of
the threshold has no valid placement once the threshold moves within half a
width of an endpoint. A 2% false-positive budget on 22 negatives put the
threshold at 0.9734, where no 0.20-wide band fits, and calibration raised.

Batch 7 makes the band symmetric in **log-odds** instead. Every finite margin
around every threshold strictly inside (0, 1) is representable, so R11 is gone
structurally rather than handled — while the derivation check that closed F11
survives unweakened, because log-odds symmetry is still an equality on a single
derived parameter.

Each requirement below is checked against a *named double* — the geometry that
shipped before — to prove the test can actually fail.
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.common import (  # noqa: E402
    BAND_EDGE_FLOOR,
    DEFAULT_LOG_ODDS_MARGIN,
    LOG_ODDS_BAND_TOLERANCE,
    ThresholdInfo,
    Timing,
    expit,
    logit,
)
from contracts.enums import AbstainReason, BranchId, MediaKind, Verdict  # noqa: E402
from contracts.evidence import EvidenceBundle  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from evaluation.band import (  # noqa: E402
    clipped_fixed_width_band,
    fixed_width_band,
)
from inference.calibration import abstention_band  # noqa: E402
from inference.models import ModelSpec  # noqa: E402
from inference.postprocess import (  # noqa: E402
    BranchOutcome,
    QualityFacts,
    build_image_bundle,
    decide,
)

TIMING = Timing(queued_ms=0, preprocess_ms=1, inference_ms=1, total_ms=2)

R11_THRESHOLD = 0.9734218951277995
"""The threshold from the audit's R11 note, to the digit.

Produced by a 2% false-positive budget over 22 negatives, where the threshold is
simply the largest negative score. Used verbatim so the fixed case and the
recorded failure are the same case.
"""

R11_LEGACY_WIDTH = 0.20
"""The band width shipped alongside it. The other half of the impossible pair."""


# --------------------------------------------------------------------------
# The named doubles
# --------------------------------------------------------------------------


class LegacyFixedWidthBand:
    """The pre-Batch-7 geometry, preserved so its failures can be measured.

    Not a description of the old code — it *is* the old arithmetic, kept in
    ``evaluation.band`` and wrapped here. Every "the new implementation prevents
    this" claim in this file is a pair: the double does the bad thing, and the
    shipped type does not.
    """

    name = "fixed-width, symmetric in probability"

    @staticmethod
    def band(threshold: float, width: float) -> tuple[float, float]:
        return fixed_width_band(threshold, width)

    @staticmethod
    def fits(threshold: float, width: float) -> bool:
        half = width / 2.0
        return threshold - half >= 0.0 and threshold + half <= 1.0


class LegacyClippedBand:
    """The obvious "fix" for R11, and the reason it is not one.

    Clamping keeps construction total at the cost of the property the band
    exists to have. Measured below: at the R11 threshold the clamped band's
    upper edge is exactly 1.0, so no score is ever published as manipulated.
    """

    name = "fixed-width, clipped into [0, 1]"

    @staticmethod
    def band(threshold: float, width: float) -> tuple[float, float]:
        return clipped_fixed_width_band(threshold, width)


def _legacy_threshold_payload(value: float = 0.62, width: float = 0.20) -> dict:
    """The pre-Batch-1 wire shape: band below the threshold, ``band_hi == value``."""
    return {
        "value": value,
        "target_fpr": 0.02,
        "band_lo": max(0.0, value - width),
        "band_hi": value,
    }


def _p_symmetric_payload(value: float = 0.62, width: float = 0.20) -> dict:
    """The pre-Batch-7 wire shape: centred, but in probability."""
    half = width / 2.0
    return {
        "value": value,
        "target_fpr": 0.02,
        "band_lo": value - half,
        "band_hi": value + half,
    }


def _job() -> JobSpec:
    return JobSpec(
        job_id=uuid4(),
        media_kind=MediaKind.IMAGE,
        storage_key="uploads/x",
        sha256="d" * 64,
        size_bytes=1024,
        declared_mime="image/jpeg",
        model_version="test-0",
        idempotency_key="k",
        submitted_at=datetime.now(UTC),
    )


# --------------------------------------------------------------------------
# Requirement 1 — every valid configuration brackets its threshold
# --------------------------------------------------------------------------


def test_every_constructible_band_brackets_its_threshold():
    """Swept, not spot-checked. The property must hold for the whole domain."""
    checked = 0
    for value in (0.001, 0.05, 0.2, 0.35, 0.5, 0.62, 0.8, 0.95, 0.999):
        for margin in (0.04, 0.2, 0.4055, 1.0, 2.5):
            t = ThresholdInfo.at_margin(value=value, margin=margin, target_fpr=0.02)
            assert t.band_lo < t.value < t.band_hi, (
                f"band [{t.band_lo}, {t.band_hi}] does not bracket {t.value}"
            )
            checked += 1
    assert checked == 45, f"only {checked} configurations were constructible"


def test_the_band_is_centred_in_log_odds_and_recovers_its_margin():
    t = ThresholdInfo.at_margin(value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02)
    centre = logit(t.value)
    assert abs((centre - logit(t.band_lo)) - (logit(t.band_hi) - centre)) < 1e-12
    assert abs(t.margin - DEFAULT_LOG_ODDS_MARGIN) < 1e-12
    assert abs(t.odds_factor - 1.5) < 1e-12


def test_the_band_is_deliberately_not_centred_in_probability():
    """The change, stated as a property rather than as a comment.

    Above 0.5 the lower half is the wider one in probability, because equal
    evidence buys more probability where there is more room for it.
    """
    t = ThresholdInfo.at_margin(value=0.90, margin=0.5, target_fpr=0.02)
    below = t.value - t.band_lo
    above = t.band_hi - t.value
    assert below > above, "the probability-space asymmetry has vanished"
    assert abs(below - above) > 0.01, "the asymmetry is too small to be the real one"


def test_a_zero_or_negative_margin_is_refused():
    for margin in (0.0, -0.1, float("inf"), float("nan")):
        try:
            ThresholdInfo.at_margin(value=0.62, margin=margin, target_fpr=0.02)
        except ValueError as exc:
            assert "margin must be" in str(exc)
            continue
        raise AssertionError(f"a margin of {margin} was accepted")


def test_a_threshold_at_an_endpoint_is_refused():
    """Not a band failure. A saturated fit, named as one."""
    for value in (0.0, 1.0):
        try:
            ThresholdInfo.at_margin(value=value, margin=0.4, target_fpr=0.02)
        except ValueError as exc:
            assert "strictly inside (0, 1)" in str(exc)
            continue
        raise AssertionError(f"a threshold of {value} was accepted")


# --------------------------------------------------------------------------
# Requirement 2 — R11: no (threshold, band) pair is jointly impossible
# --------------------------------------------------------------------------


def test_the_legacy_double_reproduces_the_r11_failure():
    """Requirement 9. The recorded failure, against the real old arithmetic."""
    assert not LegacyFixedWidthBand.fits(R11_THRESHOLD, R11_LEGACY_WIDTH)
    try:
        LegacyFixedWidthBand.band(R11_THRESHOLD, R11_LEGACY_WIDTH)
    except ValueError as exc:
        assert "outside [0, 1]" in str(exc)
        assert str(R11_THRESHOLD) in str(exc), (
            "the double no longer reproduces the audit's exact case"
        )
        return
    raise AssertionError("the legacy geometry accepted the impossible pair")


def test_the_new_implementation_survives_the_r11_case():
    """The same threshold, the shipped default margin, no exception."""
    t = ThresholdInfo.at_margin(
        value=R11_THRESHOLD, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
    )
    assert t.band_lo < R11_THRESHOLD < t.band_hi
    assert 0.0 < t.band_lo and t.band_hi < 1.0
    # And it is a usable operating point, not a degenerate one: there is real
    # room above the upper edge for a score to be called manipulated.
    assert t.band_hi < 0.99


def test_no_threshold_and_margin_pair_is_jointly_impossible():
    """The totality claim, swept over five decades of threshold on both sides.

    Every pair here is representable — 36 of 36. Against the same thresholds the
    legacy geometry could not place its shipped width at all for most of them,
    which is what "jointly unrepresentable" meant.
    """
    thresholds = [expit(z) for z in (-12, -8, -4, -1, 0, 1, 4, 8, 12)]
    margins = (0.01, 0.1, 0.4055, 1.0)
    built = 0
    for value in thresholds:
        for margin in margins:
            t = ThresholdInfo.at_margin(value=value, margin=margin, target_fpr=0.02)
            assert t.band_lo < value < t.band_hi
            built += 1
    assert built == 36

    legacy_impossible = [
        v for v in thresholds if not LegacyFixedWidthBand.fits(v, R11_LEGACY_WIDTH)
    ]
    assert len(legacy_impossible) >= 6, (
        f"only {len(legacy_impossible)} of these thresholds refused the shipped "
        "legacy width; the sweep no longer covers the failing regime"
    )


def test_the_shipped_parameter_is_feasible_over_essentially_the_whole_range():
    """The honest statement of the fix, as a measured comparison.

    Both geometries have a feasible threshold range for a *fixed* parameter —
    that is unavoidable, since a band needs room. What changed is its size. The
    shipped width of 0.20 was infeasible for a fifth of the unit interval,
    including every threshold a tight false-positive budget produces. The
    shipped margin is infeasible only within about a millionth of an endpoint,
    where the operating point is degenerate for reasons that have nothing to do
    with the band.
    """

    def legacy_feasible(value: float) -> bool:
        return LegacyFixedWidthBand.fits(value, R11_LEGACY_WIDTH)

    def new_feasible(value: float) -> bool:
        try:
            ThresholdInfo.at_margin(
                value=value, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
            )
        except ValueError:
            return False
        return True

    # 1e-4 rather than 1e-5: the two fractions below are 0.80 and 1.00, and a
    # ten-times finer grid moves neither while costing 180000 extra pydantic
    # validations. The exact limit is checked analytically at the end.
    grid = [i / 10000.0 for i in range(1, 10000)]
    legacy_bad = [v for v in grid if not legacy_feasible(v)]
    new_bad = [v for v in grid if not new_feasible(v)]

    assert len(legacy_bad) / len(grid) > 0.19, (
        "the legacy width is no longer infeasible over a fifth of the range; "
        "the comparison has lost its subject"
    )
    assert new_bad == [], (
        f"the shipped margin is infeasible somewhere on a 1e-4 grid: {new_bad[:5]}"
    )
    # And the true limit, found exactly rather than by grid search.
    assert ThresholdInfo.max_margin(1.5e-6) > DEFAULT_LOG_ODDS_MARGIN
    assert ThresholdInfo.max_margin(1.0e-6) < DEFAULT_LOG_ODDS_MARGIN


def test_clipping_would_have_destroyed_the_manipulated_region():
    """Requirement 9's second half: why the obvious fix is worse than the bug.

    Clipping is total, so it never raises — and at the R11 threshold it produces
    a band whose upper edge is 1.0. Since ``decide`` publishes MANIPULATED only
    for ``p >= band_hi``, the detector's positive class becomes unreachable and
    every published field still validates as a number.
    """
    lo, hi = LegacyClippedBand.band(R11_THRESHOLD, R11_LEGACY_WIDTH)
    assert hi == 1.0, "the clipped double no longer demonstrates the collapse"
    assert lo < R11_THRESHOLD

    decided_manipulated = [p / 1000.0 for p in range(1001) if p / 1000.0 >= hi]
    assert decided_manipulated == [1.0], (
        "only a score of exactly 1.0 survives, which no calibrated model emits"
    )

    # The shipped geometry keeps the region reachable at the same threshold.
    t = ThresholdInfo.at_margin(
        value=R11_THRESHOLD, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
    )
    still_reachable = [p / 1000.0 for p in range(1001) if p / 1000.0 >= t.band_hi]
    assert len(still_reachable) > 10


def test_the_clipped_band_is_refused_by_the_contract():
    """Totality is not enough — an off-centre band must still be unrepresentable."""
    lo, hi = LegacyClippedBand.band(R11_THRESHOLD, R11_LEGACY_WIDTH)
    try:
        ThresholdInfo(value=R11_THRESHOLD, target_fpr=0.02, band_lo=lo, band_hi=hi)
    except ValueError as exc:
        assert "endpoint" in str(exc) or "log-odds" in str(exc)
        return
    raise AssertionError("a clipped band was accepted")


# --------------------------------------------------------------------------
# Requirement 3 — moving the threshold moves the band
# --------------------------------------------------------------------------


def test_moving_the_threshold_moves_the_whole_band():
    margin = DEFAULT_LOG_ODDS_MARGIN
    previous = None
    for value in (0.3, 0.4, 0.5, 0.6, 0.7):
        t = ThresholdInfo.at_margin(value=value, margin=margin, target_fpr=0.02)
        assert abs(t.margin - margin) < 1e-12, "the margin drifted with the threshold"
        if previous is not None:
            assert t.band_lo > previous.band_lo
            assert t.band_hi > previous.band_hi
        previous = t


def test_the_band_cannot_be_left_behind_when_the_threshold_moves():
    """The failure the type exists to prevent: an operating point raised in
    configuration while the band edges stay where they were."""
    good = ThresholdInfo.at_margin(value=0.62, margin=0.4055, target_fpr=0.02)
    try:
        ThresholdInfo(
            value=0.75,  # moved
            target_fpr=good.target_fpr,
            band_lo=good.band_lo,  # left behind
            band_hi=good.band_hi,
        )
    except ValueError as exc:
        assert "band_hi" in str(exc) or "log-odds" in str(exc)
        return
    raise AssertionError("a threshold moved away from its band was accepted")


def test_the_margin_is_constant_evidence_across_the_range():
    """What the new parameter buys, stated as the property it was chosen for.

    One margin means one odds ratio everywhere. The old width did not: 0.20 of
    probability is 0.80 nats at a threshold of 0.5 and 2.35 nats at 0.9.
    """
    margin = 0.4055
    for value in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        t = ThresholdInfo.at_margin(value=value, margin=margin, target_fpr=0.02)
        odds_hi = t.band_hi / (1 - t.band_hi)
        odds_at = value / (1 - value)
        assert abs(odds_hi / odds_at - math.exp(margin)) < 1e-9

    evidence_of_a_fixed_width = []
    for value in (0.5, 0.85):
        _, hi = fixed_width_band(value, 0.20)
        evidence_of_a_fixed_width.append(logit(hi) - logit(value))
    assert evidence_of_a_fixed_width[1] > 2.5 * evidence_of_a_fixed_width[0], (
        "a fixed probability width no longer varies in evidence; the argument "
        "for the change has stopped holding"
    )
    # Pushed a little further and the fixed width stops being a band at all —
    # ``logit(1.0)`` is where the old geometry's upper edge ends up at 0.90.
    assert fixed_width_band(0.90, 0.20)[1] == 1.0


# --------------------------------------------------------------------------
# Requirement 4 — no gap, no overlap
# --------------------------------------------------------------------------


def test_the_decision_regions_partition_the_unit_interval():
    """Every probability gets exactly one verdict, and the regions are contiguous."""
    t = ThresholdInfo.at_margin(value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02)
    order = [Verdict.AUTHENTIC, Verdict.INSUFFICIENT_EVIDENCE, Verdict.MANIPULATED]
    ranks = []
    for i in range(4001):
        p = i / 4000.0
        verdict, reason = decide(p, t)
        assert verdict in order
        assert (reason is AbstainReason.SCORE_IN_ABSTENTION_BAND) == (
            verdict is Verdict.INSUFFICIENT_EVIDENCE
        ), f"reason and verdict disagree at p={p}"
        ranks.append(order.index(verdict))
    assert ranks == sorted(ranks), "the regions are not contiguous — there is a gap"
    assert set(ranks) == {0, 1, 2}, "some region is unreachable"


def test_the_partition_holds_at_an_extreme_threshold_too():
    """The R11 regime, checked for the property the old geometry could not reach."""
    t = ThresholdInfo.at_margin(value=R11_THRESHOLD, margin=1.0, target_fpr=0.02)
    seen = set()
    for i in range(20001):
        p = i / 20000.0
        verdict, _ = decide(p, t)
        seen.add(verdict)
    assert seen == {Verdict.AUTHENTIC, Verdict.INSUFFICIENT_EVIDENCE, Verdict.MANIPULATED}


def test_no_probability_is_both_decided_and_inside_the_band():
    """The headline symptom, checked directly."""
    t = ThresholdInfo.at_margin(value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02)
    for i in range(4001):
        p = i / 4000.0
        verdict, _ = decide(p, t)
        inside = t.band_lo <= p < t.band_hi
        decided = verdict is not Verdict.INSUFFICIENT_EVIDENCE
        assert inside != decided, (
            f"p={p} is {'inside' if inside else 'outside'} the band and "
            f"{'decided' if decided else 'undecided'} — the two must be exclusive"
        )


def test_the_legacy_configuration_produced_exactly_that_contradiction():
    """The F11 bug, reproduced against the old shape to prove the check bites."""
    legacy = _legacy_threshold_payload()
    lo, hi, value = legacy["band_lo"], legacy["band_hi"], legacy["value"]

    def legacy_decide(p: float) -> Verdict:
        if p >= value:
            return Verdict.MANIPULATED
        if lo <= p < hi:
            return Verdict.INSUFFICIENT_EVIDENCE
        return Verdict.AUTHENTIC

    assert hi == value, "the legacy double no longer reproduces the coincidence"
    assert legacy_decide(value) is Verdict.MANIPULATED
    try:
        ThresholdInfo(**legacy)
    except ValueError:
        return
    raise AssertionError("the legacy threshold shape was accepted")


# --------------------------------------------------------------------------
# Requirement 5 — inconsistent configurations fail validation
# --------------------------------------------------------------------------


def test_inconsistent_configurations_are_rejected():
    base = {"value": 0.62, "target_fpr": 0.02}
    cases = {
        "band ends at the threshold": {"band_lo": 0.42, "band_hi": 0.62},
        "band starts at the threshold": {"band_lo": 0.62, "band_hi": 0.82},
        "band entirely below": {"band_lo": 0.30, "band_hi": 0.50},
        "band entirely above": {"band_lo": 0.70, "band_hi": 0.90},
        "off-centre": {"band_lo": 0.60, "band_hi": 0.90},
        "inverted": {"band_lo": 0.72, "band_hi": 0.52},
        "edge at zero": {"band_lo": 0.0, "band_hi": 0.9999},
        "edge at one": {"band_lo": 0.0001, "band_hi": 1.0},
    }
    for name, band in cases.items():
        try:
            ThresholdInfo(**base, **band)
        except ValueError:
            continue
        raise AssertionError(f"an inconsistent configuration was accepted: {name}")


def test_a_probability_symmetric_band_is_now_rejected():
    """The Batch 7 boundary, in both directions.

    The band that shipped before is refused, and the message says why rather
    than reporting a generic inconsistency — this is the one rejection an
    engineer will meet while upgrading.
    """
    try:
        ThresholdInfo(**_p_symmetric_payload())
    except ValueError as exc:
        assert "log-odds" in str(exc)
        assert "at_margin" in str(exc)
        return
    raise AssertionError("a probability-symmetric band was accepted")


def test_only_the_midpoint_is_symmetric_in_both_spaces():
    """A sanity check on the previous test: the two geometries do coincide at
    0.5, so a test that passed there would prove nothing."""
    t = ThresholdInfo.at_margin(value=0.5, margin=0.4, target_fpr=0.02)
    assert abs((t.value - t.band_lo) - (t.band_hi - t.value)) < 1e-12


def test_a_stored_bundle_with_an_inconsistent_band_is_rejected_on_read():
    """Validation must bite on deserialisation, not only on construction."""
    good = ThresholdInfo.at_margin(value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02)
    bundle = build_image_bundle(
        job=_job(),
        model=ModelSpec(name="t", version="v", backend="stub", threshold=good),
        size_bytes=1024,
        width=1920,
        height=1080,
        quality=QualityFacts(width=1920, height=1080, face_px=400),
        outcomes={
            BranchId.A: BranchOutcome(logit=4.0),
            BranchId.B: BranchOutcome(logit=3.5),
            BranchId.P: BranchOutcome(logit=0.0),
        },
        timing=TIMING,
    )
    payload = bundle.model_dump(mode="json")
    payload["threshold"]["band_hi"] = payload["threshold"]["value"]  # the legacy shape
    try:
        EvidenceBundle.model_validate(payload)
    except Exception as exc:
        assert "band_hi" in str(exc) or "log-odds" in str(exc)
        return
    raise AssertionError("a stored bundle carrying an inconsistent band was read back")


def test_a_verdict_inside_the_band_cannot_be_published():
    """The strengthened bundle invariant, unchanged by the new geometry."""
    good = ThresholdInfo.at_margin(value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02)
    bundle = build_image_bundle(
        job=_job(),
        model=ModelSpec(name="t", version="v", backend="stub", threshold=good),
        size_bytes=1024,
        width=1920,
        height=1080,
        quality=QualityFacts(width=1920, height=1080, face_px=400),
        outcomes={
            BranchId.A: BranchOutcome(logit=6.0),
            BranchId.B: BranchOutcome(logit=6.0),
            BranchId.P: BranchOutcome(logit=0.0),
        },
        timing=TIMING,
    )
    payload = bundle.model_dump(mode="json")
    payload["p_fake"] = 0.65  # above the centre, inside the band
    payload["interval"] = [0.60, 0.70]
    try:
        EvidenceBundle.model_validate(payload)
    except Exception as exc:
        assert "abstention band" in str(exc)
        return
    raise AssertionError("a decided verdict inside the abstention band validated")


# --------------------------------------------------------------------------
# Requirement 6 — the numerical limits, measured rather than assumed
# --------------------------------------------------------------------------


def test_the_symmetry_tolerance_holds_across_the_whole_admissible_domain():
    """The tolerance is a measured figure. This is the measurement, as a test.

    Constructs a band, re-reads its edges as stored, and re-derives the two half
    margins — which is exactly what the validator does on a bundle arriving from
    storage. If the round-trip error ever exceeds the tolerance, a legitimately
    constructed operating point becomes unreadable, so the number has to be
    checked and not just chosen.
    """
    worst = 0.0
    checked = 0
    for k in range(0, 121):
        value = expit(-13.5 + k * 27.0 / 120)
        for margin in (1e-3, 0.01, 0.1, 0.4055, 1.0, 3.0, 8.0):
            try:
                t = ThresholdInfo.at_margin(value=value, margin=margin, target_fpr=0.02)
            except ValueError:
                continue  # an edge would reach the floor; refused, by design
            centre = logit(t.value)
            above = logit(t.band_hi) - centre
            below = centre - logit(t.band_lo)
            worst = max(worst, abs(above - below) / max(above, below, 1.0))
            checked += 1
    assert checked > 500, f"only {checked} pairs were exercised"
    assert worst < LOG_ODDS_BAND_TOLERANCE, (
        f"worst round-trip asymmetry {worst:.3e} is not inside the tolerance "
        f"{LOG_ODDS_BAND_TOLERANCE:.3e}"
    )
    assert worst > LOG_ODDS_BAND_TOLERANCE / 1000, (
        f"worst round-trip asymmetry {worst:.3e} is more than a thousand times "
        f"inside the tolerance; the tolerance is looser than it needs to be"
    )


def test_a_margin_below_float_resolution_is_named_as_such():
    """The one remaining construction failure, and it is not a domain error."""
    value = 1.0 - 1e-9
    try:
        ThresholdInfo.at_margin(value=value, margin=1e-12, target_fpr=0.02)
    except ValueError as exc:
        assert "float64 resolution" in str(exc) or "endpoint" in str(exc)
        return
    raise AssertionError("a sub-ULP margin produced a band")


def test_a_margin_that_pushes_an_edge_to_the_floor_is_refused_with_the_limit():
    t = 0.9
    limit = ThresholdInfo.max_margin(t)
    ThresholdInfo.at_margin(value=t, margin=limit * 0.999, target_fpr=0.02)
    try:
        ThresholdInfo.at_margin(value=t, margin=limit * 1.001, target_fpr=0.02)
    except ValueError as exc:
        assert "unreachable" in str(exc)
        assert f"{limit:.4f}" in str(exc), "the message does not report the limit"
        return
    raise AssertionError("a margin past the edge floor was accepted")


def test_the_edge_floor_is_what_bounds_the_usable_margin():
    for value in (0.5, 0.62, R11_THRESHOLD, 0.001):
        limit = ThresholdInfo.max_margin(value)
        t = ThresholdInfo.at_margin(value=value, margin=limit, target_fpr=0.02)
        assert t.band_lo >= BAND_EDGE_FLOOR * (1 - 1e-9)
        assert t.band_hi <= 1.0 - BAND_EDGE_FLOOR * (1 - 1e-9)


# --------------------------------------------------------------------------
# The operating point reaches the cache key
# --------------------------------------------------------------------------


def test_moving_the_operating_point_changes_the_model_version():
    """Batch 1's invariant, re-checked on the new parameter.

    Weights, fusion coefficients and the quality gate all reached the composite
    version; the threshold and its band did not, so moving the decision boundary
    changed every verdict while leaving the result cache untouched.
    """
    from dataclasses import replace

    base = ModelSpec(name="t", version="v", backend="stub")
    moved = replace(
        base,
        threshold=ThresholdInfo.at_margin(value=0.80, margin=0.4055, target_fpr=0.02),
    )
    widened = replace(
        base,
        threshold=ThresholdInfo.at_margin(value=0.62, margin=0.62, target_fpr=0.02),
    )
    assert base.effective_version != moved.effective_version, (
        "moving the threshold did not change the model version"
    )
    assert base.effective_version != widened.effective_version, (
        "widening the abstention band did not change the model version"
    )
    assert moved.effective_version != widened.effective_version


def test_the_geometry_change_moved_the_shipped_operating_point():
    """Deliberate, and it must be visible.

    The shipped band moved from [0.5200, 0.7200) to [0.5210, 0.7099). Verdicts
    for scores in the two slivers between those edges change, so the calibration
    fingerprint has to move too — a geometry change that left the cache key
    alone would serve old verdicts under the new rule. This asserts the size of
    the move as well as the fact of it: small enough to review, not zero.
    """
    shipped = ModelSpec(name="t", version="v", backend="stub").threshold
    old_lo, old_hi = 0.52, 0.72
    assert abs(shipped.band_lo - old_lo) < 0.01
    assert abs(shipped.band_hi - old_hi) < 0.011
    assert (shipped.band_lo, shipped.band_hi) != (old_lo, old_hi)


def test_calibration_helper_agrees_with_the_contract():
    """``abstention_band`` and ``ThresholdInfo.at_margin`` must not disagree."""
    lo, hi = abstention_band(0.62, margin=DEFAULT_LOG_ODDS_MARGIN)
    t = ThresholdInfo.at_margin(value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02)
    assert lo == t.band_lo and hi == t.band_hi, (
        "the helper has grown its own copy of the derivation"
    )


def test_the_helper_survives_the_r11_threshold_too():
    lo, hi = abstention_band(R11_THRESHOLD)
    assert 0.0 < lo < R11_THRESHOLD < hi < 1.0


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
