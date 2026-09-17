"""R8 — the band is derived from the data, not carried as a constant.

WHAT R8 WAS
-----------
``band_width = 0.20`` was picked, never derived. It was the largest single
unjustified number in the system, and it was unjustifiable in principle as well
as in practice: a probability width means a different amount of evidence at
every threshold it is used at, so there is no value of it that is right
everywhere (see ``test_threshold_band.py``).

WHAT CLOSES IT
--------------
Two things, and the second is the one that matters.

*The parameter changed.* A log-odds margin is constant evidence, so a single
number can be right everywhere. That is a precondition for deriving one.

*The derivation moved into the calibration pass.* ``derive_operating_point``
takes the calibrated scores and the product constraints and returns threshold
and margin together. ``fit_calibration(margin=DERIVE)`` — the default — routes
through it. There is no longer an argument anywhere that sets the band without
the threshold in hand, which is what made R11 possible.

These tests are organised by requirement 8 of the batch brief: thresholds near 0
and near 1, tiny and large calibration sets, impossible bands, symmetric and
asymmetric bands, and abstention correctness.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.common import (  # noqa: E402
    DEFAULT_LOG_ODDS_MARGIN,
    ThresholdInfo,
    expit,
    logit,
)
from contracts.enums import Verdict  # noqa: E402
from evaluation.band import (  # noqa: E402
    InfeasibleOperatingPoint,
    admissible_threshold,
    clipped_fixed_width_band,
    derive_operating_point,
    fixed_width_band,
    margin_for_probability_floor,
    margin_grid,
    proportional_band,
    quantile_band,
    wilson_margin,
)
from evaluation.calibration_protocol import DERIVE, fit_calibration  # noqa: E402
from evaluation.dataset import MediaItem  # noqa: E402
from evaluation.protocol import FFPP_CELEBDF, DatasetRole, select  # noqa: E402
from evaluation.risk import operating_point, select_margin  # noqa: E402
from evaluation.split_lock import create_lock  # noqa: E402
from evaluation.thresholds import at_target_fpr  # noqa: E402
from inference.postprocess import decide  # noqa: E402


def scores(
    n: int = 4000,
    sep: float = 2.0,
    seed: int = 11,
    prevalence: float = 0.5,
    sigma: float = 1.6,
):
    """Calibrated-looking probabilities with both classes present.

    ``sigma`` is the knob that moves the *band*, and ``sep`` the one that moves
    the threshold. Worth separating: the margin is set by how heavy the negative
    tail is above the threshold, which is a property of the spread, not of how
    far apart the two class means are.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(np.float64)
    z = rng.normal(0.0, sigma, n) + sep * (2.0 * y - 1.0)
    return y, np.asarray([expit(v) for v in z], dtype=np.float64)


def tiny_saturated(n_neg: int = 22, seed: int = 3):
    """The R11 fixture's shape: few negatives, a detector that separates well.

    A 2% budget over this many negatives puts the threshold at the largest
    negative score, which is what drove it to 0.9734 in the audit.
    """
    rng = np.random.default_rng(seed)
    neg = np.asarray([expit(v) for v in rng.normal(-1.0, 1.8, n_neg)])
    pos = np.asarray([expit(v) for v in rng.normal(3.5, 1.2, 60)])
    y = np.r_[np.zeros(n_neg), np.ones(pos.size)]
    return y, np.r_[neg, pos]


# --------------------------------------------------------------------------
# 1. Thresholds near 0 and near 1
# --------------------------------------------------------------------------


def test_a_threshold_near_one_yields_a_band_where_the_old_geometry_could_not():
    y, p = tiny_saturated()
    choice = at_target_fpr(y, p, target_fpr=0.02)
    assert choice.value > 0.90, (
        f"the fixture no longer produces an extreme threshold ({choice.value:.4f})"
    )

    # The old geometry, on this exact threshold.
    try:
        fixed_width_band(choice.value, 0.20)
    except ValueError as exc:
        assert "outside [0, 1]" in str(exc)
    else:
        raise AssertionError("the legacy width fitted; the regime has moved")

    # The new one, with no special-casing.
    derived = derive_operating_point(
        y, p, target_fpr=0.02, max_selective_fpr=0.02, min_coverage=0.5
    )
    assert derived.ok, derived.describe()
    info = derived.threshold_info(target_fpr=0.02)
    assert 0.0 < info.band_lo < info.value < info.band_hi < 1.0


def test_a_threshold_near_zero_yields_a_band_too():
    """The mirror case. Reached by a TPR-style budget rather than an FPR one."""
    y, p = scores(sep=1.0)
    for value in (1e-4, 1e-3, 0.01, 0.05):
        info = ThresholdInfo.at_margin(
            value=value, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
        )
        point = operating_point(y, p, info)
        assert 0.0 < info.band_lo < value < info.band_hi
        assert point.n_decided + point.n_abstained == point.n


def test_the_band_shrinks_in_probability_as_the_threshold_saturates():
    """The behaviour near the endpoints, stated as the property it should have.

    A constant evidence margin buys less probability where there is less room.
    That is the correct behaviour and it is worth pinning, because the opposite
    — a band that keeps its probability width and runs out of room — is exactly
    what R11 was.
    """
    widths = [
        ThresholdInfo.at_margin(
            value=v, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
        ).band_width
        for v in (0.5, 0.8, 0.95, 0.99, 0.999)
    ]
    assert widths == sorted(widths, reverse=True), (
        f"the band did not narrow monotonically toward the endpoint: {widths}"
    )
    assert widths[0] > 40 * widths[-1]


def test_the_margin_limit_degrades_gracefully_rather_than_cliff_edging():
    """``max_margin`` falls off linearly in log-odds, so there is no sharp edge.

    Under the old geometry the feasible width collapsed to zero *linearly in
    probability*, so a threshold moving from 0.90 to 0.95 halved it. Here the
    limit is 13.8155 - |logit(t)|, which still has 6.9 of headroom at a
    threshold of 0.999 — seventeen times the shipped margin.
    """
    for value, floor in ((0.9, 11.0), (0.99, 9.2), (0.999, 6.9)):
        limit = ThresholdInfo.max_margin(value)
        assert limit > floor, f"max_margin({value}) = {limit:.3f}, below {floor}"
        assert limit > 15 * DEFAULT_LOG_ODDS_MARGIN


# --------------------------------------------------------------------------
# 2. Tiny and large calibration sets
# --------------------------------------------------------------------------


def test_a_tiny_calibration_set_derives_a_band_and_says_it_is_unreliable():
    """The failure moves from an exception to a warning that names the cause.

    This is the whole shape of the R11 fix in one test: the band is available,
    and the thing that was actually wrong — an extreme quantile of 22 negatives
    — is reported instead of being hidden behind a band error.
    """
    y, p = tiny_saturated()
    derived = derive_operating_point(
        y, p, target_fpr=0.02, max_selective_fpr=0.02, min_coverage=0.5
    )
    assert derived.ok
    text = " ".join(derived.warnings)
    assert "negatives" in text
    assert "resampling" in text or "stable" in text
    assert "bootstrap_threshold_ci" in text


def test_a_large_calibration_set_derives_a_stable_operating_point():
    y, p = scores(n=8000, sep=2.2)
    a = derive_operating_point(
        y, p, target_fpr=0.05, max_selective_fpr=0.02, min_coverage=0.7
    )
    b = derive_operating_point(
        y[::-1].copy(), p[::-1].copy(), target_fpr=0.05, max_selective_fpr=0.02,
        min_coverage=0.7,
    )
    assert a.ok and b.ok
    assert a.threshold == b.threshold, "the threshold moved under reordering"
    assert a.margin == b.margin, "the margin moved under reordering"


def test_the_derived_margin_responds_to_the_data_not_to_a_constant():
    """R8's actual claim: the number comes from the scores.

    A detector whose scores are tightly concentrated needs less abstention to
    reach the same selective error, so it should be handed a narrower band. If
    the derivation returned the same margin across these, it would be a constant
    wearing a function's clothes.

    Measured on this fixture: 0.99, 1.60, 2.36, 3.36 for spreads of 1.0 to 3.5 —
    a factor of 3.4, from a procedure with no constant in it.
    """
    margins = []
    for sigma in (1.0, 1.6, 2.5, 3.5):
        derived = derive_operating_point(
            *scores(sigma=sigma), target_fpr=0.05, max_selective_fpr=0.01,
            min_coverage=0.4,
        )
        assert derived.ok and derived.binding, derived.describe()
        margins.append(derived.margin)
    assert margins == sorted(margins), (
        f"the margin did not grow with the score spread: {margins}"
    )
    assert margins[-1] > 3 * margins[0], (
        f"the margin barely responded to the data: {margins}"
    )


def test_the_grid_is_geometric_and_bounded_by_what_is_representable():
    grid = margin_grid(0.9734218951277995)
    assert grid.size > 50
    ratios = grid[1:] / grid[:-1]
    assert np.allclose(ratios, ratios[0]), "the grid is not geometric"
    assert grid[-1] <= ThresholdInfo.max_margin(0.9734218951277995) + 1e-12
    for m in grid:
        ThresholdInfo.at_margin(value=0.9734218951277995, margin=float(m), target_fpr=0.02)


# --------------------------------------------------------------------------
# 3. Impossible bands
# --------------------------------------------------------------------------


def test_impossible_constraints_refuse_with_a_census_rather_than_widening():
    y, p = scores(sep=0.15)
    derived = derive_operating_point(
        y, p, target_fpr=0.02, max_selective_fpr=0.0001, min_coverage=0.99
    )
    assert not derived.ok
    text = derived.describe()
    assert "no margin satisfies" in text
    assert "rejected" in text
    assert sum(derived.rejected.values()) > 0
    try:
        derived.threshold_info(target_fpr=0.02)
    except InfeasibleOperatingPoint as exc:
        assert "no margin satisfies" in str(exc)
        return
    raise AssertionError("threshold_info() returned a point that does not exist")


def test_a_sentinel_threshold_is_reported_as_a_fitting_problem():
    """``at_target_fpr`` returns ``max(score) + 1`` when nothing meets the budget.

    That value used to reach ``ThresholdInfo`` and fail there with a message
    about bands, which sent the reader to the wrong place.
    """
    for bad in (1.9, -0.5, float("inf"), float("nan")):
        try:
            admissible_threshold(bad)
        except InfeasibleOperatingPoint as exc:
            assert "band" not in str(exc).lower()
            continue
        raise AssertionError(f"{bad} was accepted as a threshold")


def test_a_saturated_threshold_names_saturation():
    try:
        admissible_threshold(1.0)
    except InfeasibleOperatingPoint as exc:
        assert "saturated" in str(exc)
        return
    raise AssertionError("a threshold of exactly 1.0 was accepted")


def test_a_determinism_floor_too_close_to_an_endpoint_is_refused_honestly():
    """The one remaining joint impossibility, and it is not the band's fault.

    A floor of 0.05 either side of a threshold sitting 0.03 from an endpoint is
    unachievable at any width in any geometry — probability simply runs out. The
    refusal says so, and says which knob has to move.
    """
    try:
        margin_for_probability_floor(0.97, 0.05)
    except InfeasibleOperatingPoint as exc:
        assert "0.03" in str(exc)
        assert "acquiring negatives" in str(exc)
        return
    raise AssertionError("an unachievable floor was converted to a margin")


def test_the_floor_conversion_is_met_on_both_sides():
    """Not the average of the two sides — the weaker one."""
    for value in (0.2, 0.5, 0.62, 0.9):
        for floor in (0.001, 0.01, 0.05):
            margin = margin_for_probability_floor(value, floor)
            info = ThresholdInfo.at_margin(value=value, margin=margin, target_fpr=0.02)
            assert info.value - info.band_lo >= floor - 1e-12
            assert info.band_hi - info.value >= floor - 1e-12


# --------------------------------------------------------------------------
# 4. Symmetric and asymmetric bands
# --------------------------------------------------------------------------


def test_the_band_is_symmetric_in_log_odds_and_asymmetric_in_probability():
    for value in (0.2, 0.4, 0.62, 0.85):
        info = ThresholdInfo.at_margin(value=value, margin=0.6, target_fpr=0.02)
        centre = logit(value)
        assert abs((centre - logit(info.band_lo)) - (logit(info.band_hi) - centre)) < 1e-9
        if abs(value - 0.5) > 1e-9:
            assert abs((value - info.band_lo) - (info.band_hi - value)) > 1e-6


def test_the_asymmetry_always_leans_toward_the_middle():
    """The wider half in probability is the one facing 0.5.

    Worth pinning as a direction rather than a magnitude: it is the sanity check
    that the geometry is not inverted, and an inverted band would still validate
    (log-odds symmetry is preserved under swapping the two sides only at 0.5).
    """
    below_half = ThresholdInfo.at_margin(value=0.25, margin=0.6, target_fpr=0.02)
    assert (below_half.band_hi - 0.25) > (0.25 - below_half.band_lo)
    above_half = ThresholdInfo.at_margin(value=0.75, margin=0.6, target_fpr=0.02)
    assert (0.75 - above_half.band_lo) > (above_half.band_hi - 0.75)


def test_a_proportional_band_would_have_been_symmetric_and_meaningless():
    """The runner-up design, kept measurable.

    ``proportional_band`` is total and probability-symmetric, so it would have
    preserved the old validator. Its parameter is the problem: the same fraction
    produces bands whose evidence content differs by orders of magnitude, which
    is the defect being fixed, merely relabelled.
    """
    evidence = []
    for value in (0.5, 0.9, 0.99):
        lo, hi = proportional_band(value, 0.5)
        evidence.append(logit(hi) - logit(value))
    assert max(evidence) / min(evidence) > 1.5, (
        "the proportional family no longer varies in evidence; the argument "
        "against it has stopped holding"
    )
    # And it does agree with the log-odds band to first order near an endpoint,
    # which is why it was a serious candidate: fraction = sinh(margin).
    margin = 0.3
    lo_p, hi_p = proportional_band(0.999, math.sinh(margin))
    info = ThresholdInfo.at_margin(value=0.999, margin=margin, target_fpr=0.02)
    assert abs((hi_p - lo_p) - info.band_width) / info.band_width < 0.02


def test_a_quantile_band_brackets_the_threshold_but_is_not_derived_from_it():
    """Approach C, and the reason it is not the policy.

    It brackets and it is total, so it passes the shallow checks. What it cannot
    do is be a function of the threshold: change the corpus and the band moves
    while the threshold stands still, so a stored operating point is no longer
    reproducible from its own published fields.
    """
    y, p = scores(seed=1)
    lo_a, hi_a = quantile_band(p, 0.5, abstention_rate=0.2)
    assert lo_a < 0.5 < hi_a

    _, p_other = scores(seed=2, sep=3.0)
    lo_b, hi_b = quantile_band(p_other, 0.5, abstention_rate=0.2)
    assert (lo_a, hi_a) != (lo_b, hi_b), "the corpus dependence has vanished"

    # Neither is expressible as a ThresholdInfo, because neither is centred.
    for lo, hi in ((lo_a, hi_a), (lo_b, hi_b)):
        try:
            ThresholdInfo(value=0.5, target_fpr=0.02, band_lo=lo, band_hi=hi)
        except ValueError:
            continue
        raise AssertionError("a quantile band was accepted by the contract")


def test_a_confidence_interval_band_is_a_floor_not_a_policy():
    """Approach F. It shrinks with the sample size and says nothing about risk.

    Both properties are demonstrated rather than argued: quadrupling n halves
    the margin, and the scores never enter the calculation at all — so a model
    that got worse overnight would be handed the same band as long as the
    holdout stayed the same size.
    """
    small = wilson_margin(0.62, 200)
    large = wilson_margin(0.62, 800)
    assert small > large
    assert 1.9 < small / large < 2.2, f"the sqrt(n) scaling is off: {small / large:.3f}"
    # The margin is well below the shipped one at any realistic holdout size,
    # which is what makes it a floor rather than a choice.
    assert wilson_margin(0.62, 2000) < DEFAULT_LOG_ODDS_MARGIN


# --------------------------------------------------------------------------
# 5. Abstention correctness
# --------------------------------------------------------------------------


def test_the_derived_point_abstains_exactly_where_it_says_it_does():
    y, p = scores(sep=2.0)
    derived = derive_operating_point(
        y, p, target_fpr=0.05, max_selective_fpr=0.01, min_coverage=0.6
    )
    info = derived.threshold_info(target_fpr=0.05)
    point = derived.chosen

    abstained = 0
    for score in p:
        verdict, _ = decide(float(score), info)
        inside = info.band_lo <= score < info.band_hi
        assert inside == (verdict is Verdict.INSUFFICIENT_EVIDENCE)
        abstained += inside
    assert abstained == point.n_abstained, (
        f"the reported abstention count ({point.n_abstained}) does not match "
        f"the decisions the published band produces ({abstained})"
    )


def test_the_reported_selective_rates_hold_at_the_chosen_point():
    y, p = scores(sep=2.0)
    derived = derive_operating_point(
        y, p, target_fpr=0.05, max_selective_fpr=0.05, min_coverage=0.6
    )
    point = derived.chosen
    assert point.selective_fpr <= 0.05
    assert point.coverage >= 0.6
    assert point.tp + point.fp + point.tn + point.fn == point.n_decided
    assert point.n_decided + point.n_abstained == point.n


def test_widening_past_the_chosen_margin_only_costs_coverage():
    """Why "narrowest survivor" is the rule and not an arbitrary tie-break."""
    y, p = scores(sep=2.0)
    derived = derive_operating_point(
        y, p, target_fpr=0.05, max_selective_fpr=0.01, min_coverage=0.6
    )
    chosen = derived.chosen
    wider = operating_point(
        y, p,
        ThresholdInfo.at_margin(
            value=derived.threshold, margin=derived.margin * 2.0, target_fpr=0.05
        ),
    )
    assert wider.coverage < chosen.coverage
    assert wider.selective_fpr <= chosen.selective_fpr + 1e-12


def test_the_selective_fpr_ceiling_defaults_to_the_budget_not_to_something_looser():
    """A looser ceiling would be satisfied by construction and constrain nothing.

    Because the decision boundary is ``band_hi`` rather than ``value``, widening
    the band can only lower the achieved false-positive rate. So any ceiling
    above the budget the threshold was fitted to is met by the narrowest
    candidate on the grid and never rejects anything.
    """
    y, p = scores(sep=1.6)
    tight = derive_operating_point(
        y, p, target_fpr=0.02, max_selective_fpr=0.005, min_coverage=0.5
    )
    assert tight.binding, "a tighter ceiling than the budget must bind"
    loose = derive_operating_point(
        y, p, target_fpr=0.02, max_selective_fpr=0.9, min_coverage=0.5
    )
    assert loose.rejected["selective_fpr_too_high"] == 0, (
        "a ceiling of 0.9 rejected something; the monotonicity argument is wrong"
    )
    assert loose.margin <= tight.margin


def test_the_clipped_alternative_reports_zero_recall_at_the_r11_threshold():
    """Approach B, measured rather than asserted. Requirement 9's counterpart."""
    y, p = tiny_saturated()
    threshold = at_target_fpr(y, p, target_fpr=0.02).value
    lo, hi = clipped_fixed_width_band(threshold, 0.20)
    assert hi == 1.0
    decided_manipulated = int(np.sum(p >= hi))
    assert decided_manipulated == 0, (
        "the clipped band still decides something; the collapse has moved"
    )
    # The shipped geometry, same threshold, same scores.
    info = ThresholdInfo.at_margin(
        value=threshold, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
    )
    # One, on this fixture — and that is the honest number. A threshold of
    # 0.9944 detects almost nothing whatever band is drawn around it, which is
    # the point: the band was never what was broken here. What matters is that
    # the manipulated region is *reachable* rather than provably empty.
    assert int(np.sum(p >= info.band_hi)) >= 1


# --------------------------------------------------------------------------
# 6. The knob is gone: threshold and band come from one call
# --------------------------------------------------------------------------


def corpus(n_per_dataset: int = 1600) -> list[MediaItem]:
    items: list[MediaItem] = []
    for dataset in ("ffpp", "celebdf"):
        for i in range(n_per_dataset):
            items.append(
                MediaItem(
                    f"{dataset}_m{i}", f"{dataset}_s{i}", dataset, i % 2,
                    subject_id=f"{dataset}_p{i}",
                )
            )
    return items


def calibration_selection():
    items = corpus()
    lock = create_lock(items)
    return select(
        FFPP_CELEBDF, DatasetRole.CALIBRATION, items, lock.assignments,
        lock_hash=lock.content_hash(),
    )


def logits_for(selection, *, sep: float = 1.4, seed: int = 0):
    y = np.array([i.label for i in selection], dtype=np.float64)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 2.0, y.size) + sep * (2.0 * y - 1.0), y


def test_fit_calibration_derives_the_band_by_default():
    selection = calibration_selection()
    z, y = logits_for(selection)
    result = fit_calibration(selection, z, y, target_fpr=0.1, min_coverage=0.5)
    assert result.witness.margin is not None
    assert result.derivation is not None and result.derivation.ok
    assert result.witness.margin == result.derivation.margin
    info = result.threshold_info()
    assert info.band_lo < info.value < info.band_hi


def test_the_derived_margin_is_not_the_shipped_placeholder():
    """If it were, nothing would have been derived."""
    selection = calibration_selection()
    z, y = logits_for(selection)
    result = fit_calibration(selection, z, y, target_fpr=0.1, min_coverage=0.5)
    assert result.witness.margin != DEFAULT_LOG_ODDS_MARGIN


def test_an_explicit_margin_is_still_accepted_for_reproducing_a_point():
    selection = calibration_selection()
    z, y = logits_for(selection)
    result = fit_calibration(selection, z, y, target_fpr=0.1, margin=0.35)
    assert result.witness.margin == 0.35
    assert result.derivation is None


def test_an_infeasible_derivation_refuses_the_whole_calibration():
    """A calibration that cannot produce a usable operating point is not a
    calibration with a missing band — it is a failed calibration."""
    selection = calibration_selection()
    z, y = logits_for(selection, sep=0.05)
    try:
        fit_calibration(
            selection, z, y, target_fpr=0.02, margin=DERIVE,
            max_selective_fpr=0.0001, min_coverage=0.995,
        )
    except InfeasibleOperatingPoint as exc:
        assert "no operating point" in str(exc)
        assert "rejected" in str(exc)
        return
    raise AssertionError("an infeasible operating point produced a witness")


def test_the_witness_records_the_margin_and_refuses_a_1_0_0_payload():
    selection = calibration_selection()
    z, y = logits_for(selection)
    payload = fit_calibration(
        selection, z, y, target_fpr=0.1, margin=0.35
    ).witness.to_dict()
    assert "margin" in payload and "band_width" not in payload

    from evaluation.calibration_protocol import CalibrationWitness

    legacy = dict(payload)
    legacy["band_width"] = legacy.pop("margin")
    try:
        CalibrationWitness.from_dict(legacy)
    except ValueError as exc:
        assert "1.0.0 calibration witness" in str(exc)
        assert "Refit" in str(exc)
        return
    raise AssertionError("a 1.0.0 witness payload was read by this build")


def test_select_margin_and_derive_operating_point_agree_on_a_shared_threshold():
    """Two entry points, one rule. They differ only in who chose the threshold."""
    y, p = scores(sep=2.0)
    derived = derive_operating_point(
        y, p, target_fpr=0.05, max_selective_fpr=0.05, min_coverage=0.7
    )
    selected = select_margin(
        y, p, threshold=derived.threshold, target_fpr=0.05,
        max_selective_fpr=0.05, min_coverage=0.7,
    )
    assert selected.ok
    assert abs(selected.require().margin - derived.margin) < 1e-9


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
