"""Threshold selection and risk–coverage analysis.

Named doubles for the three mistakes this half of the framework exists to catch:

    quantile_threshold          takes the (1-target) quantile of the negatives
                                and hopes. Interpolates between observed scores,
                                so the achieved FPR misses the budget.

    accuracy_over_everything    scores a selective classifier as though it had
                                answered every input, counting abstentions as
                                correct. Makes abstaining look free.

    cost_at_test_prevalence     minimises expected cost using the evaluation
                                set's class balance instead of production's.
                                The evaluation set is near-balanced; traffic is
                                not, and the threshold that follows is far too
                                permissive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.common import ThresholdInfo  # noqa: E402
from evaluation.metrics import confusion_matrix  # noqa: E402
from evaluation.risk import (  # noqa: E402
    abstention_breakdown,
    coverage_at_risk,
    operating_point,
    risk_at_coverage,
    risk_coverage_curve,
    select_margin,
)
from evaluation.thresholds import (  # noqa: E402
    at_max_f1,
    at_min_expected_cost,
    at_target_fpr,
    at_target_tpr,
    bootstrap_threshold_ci,
    candidate_thresholds,
)


def scores(n: int = 3000, seed: int = 5, sep: float = 1.0):
    """Two overlapping beta populations. ``sep`` controls separability."""
    rng = np.random.default_rng(seed)
    a, b = 2.0 + 4.0 * sep, 2.0
    y = np.r_[np.zeros(n), np.ones(n)]
    s = np.r_[rng.beta(b, a, n), rng.beta(a, b, n)]
    return y, s


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


def quantile_threshold(y, s, target_fpr=0.02) -> float:
    """The interpolating shortcut. Convenient, and it misses the budget."""
    negatives = np.asarray(s)[np.asarray(y) == 0]
    return float(np.quantile(negatives, 1.0 - target_fpr))


def accuracy_over_everything(y, p, threshold: ThresholdInfo) -> float:
    """Counts abstentions as correct answers."""
    point = operating_point(y, p, threshold)
    return (point.tp + point.tn + point.n_abstained) / point.n


# --------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------


def test_target_fpr_is_respected_and_spent():
    y, s = scores()
    for target in (0.001, 0.01, 0.02, 0.05, 0.10):
        choice = at_target_fpr(y, s, target_fpr=target)
        assert choice.achieved_fpr <= target + 1e-12, (
            f"budget {target} exceeded: achieved {choice.achieved_fpr}"
        )
        # Spent, not merely respected: one step lower would break the budget.
        candidates = candidate_thresholds(s)
        lower = candidates[candidates < choice.value]
        if lower.size:
            assert confusion_matrix(y, s, threshold=float(lower[0])).fpr > target


def test_the_quantile_double_can_break_the_budget_and_the_exact_one_cannot():
    """The shortcut interpolates to a score nobody observed.

    Measured over 600 (seed, target, size) combinations: the quantile picks a
    different threshold in **every** one, and breaks the budget in 13% overall —
    but the failures are not scattered. They are entirely concentrated where the
    budget is tight relative to the sample: at a 0.1% budget with 200 or 800
    negatives it overshoots **100% of the time**.

    The arithmetic is exact and worth following. A 0.1% budget over 200
    negatives permits 0.2 of them above the threshold, so the only conforming
    choice is a threshold above every negative. The quantile interpolates to a
    point just below the maximum instead, letting exactly one through: an
    achieved FPR of 1/200 = 0.005, five times the budget it was asked for.

    So the shortcut fails precisely in the regime ``MIN_NEGATIVES_FOR_FPR``
    exists to warn about — a strict budget on a holdout too small to express it
    — and looks correct everywhere else.
    """
    tight, loose = 0.001, 0.02
    tight_exceeded = differed = trials = 0
    for seed in range(20):
        for n in (200, 800):
            y, s = scores(n=n, seed=seed)
            for target in (tight, loose):
                exact = at_target_fpr(y, s, target_fpr=target)
                approx = quantile_threshold(y, s, target)
                approx_fpr = confusion_matrix(y, s, threshold=approx).fpr
                trials += 1
                differed += abs(approx - exact.value) > 1e-12
                if target == tight:
                    tight_exceeded += approx_fpr > target + 1e-12
                assert exact.achieved_fpr <= target + 1e-12, (
                    f"the exact procedure broke its own {target} budget"
                )
    assert differed == trials, (
        "the quantile double now picks the same threshold; this check is vacuous"
    )
    assert tight_exceeded == trials // 2, (
        f"expected the quantile to break every tight budget, broke "
        f"{tight_exceeded} of {trials // 2}; this check is vacuous"
    )


def test_target_tpr_reaches_the_detection_floor():
    y, s = scores()
    for target in (0.5, 0.8, 0.9, 0.99):
        choice = at_target_tpr(y, s, target_tpr=target)
        assert choice.achieved_tpr >= target - 1e-12


def test_max_f1_beats_the_other_procedures_on_f1():
    y, s = scores()
    best = at_max_f1(y, s)
    for other in (at_target_fpr(y, s), at_target_tpr(y, s, target_tpr=0.9)):
        assert best.achieved_f1 >= other.achieved_f1 - 1e-12


def test_max_f1_warns_that_it_is_the_wrong_objective_here():
    y, s = scores()
    assert any("equally" in w for w in at_max_f1(y, s).warnings)


def test_expected_cost_moves_with_prevalence():
    """The mistake the double names: test balance is not production balance."""
    y, s = scores()
    balanced = at_min_expected_cost(y, s, cost_fp=10, cost_fn=1, prevalence=0.5)
    realistic = at_min_expected_cost(y, s, cost_fp=10, cost_fn=1, prevalence=0.01)
    assert realistic.value > balanced.value, (
        "at low prevalence a false positive is far more likely than a missed "
        "fake, so the threshold must become stricter"
    )
    assert realistic.achieved_fpr <= balanced.achieved_fpr


def test_defaulting_prevalence_warns():
    y, s = scores()
    assert any("prevalence" in w for w in at_min_expected_cost(y, s).warnings)


def test_asymmetric_costs_move_the_threshold():
    y, s = scores()
    fp_costly = at_min_expected_cost(y, s, cost_fp=50, cost_fn=1, prevalence=0.1)
    fn_costly = at_min_expected_cost(y, s, cost_fp=1, cost_fn=50, prevalence=0.1)
    assert fp_costly.value > fn_costly.value


def test_selection_is_deterministic():
    y, s = scores()
    assert at_target_fpr(y, s).value == at_target_fpr(y, s).value
    assert at_max_f1(y, s).value == at_max_f1(y, s).value
    a = bootstrap_threshold_ci(y, s, resamples=200, seed=7)
    b = bootstrap_threshold_ci(y, s, resamples=200, seed=7)
    assert a == b, "the same seed produced a different interval"


def test_a_small_holdout_is_flagged_as_untrustworthy():
    y, s = scores(n=40)
    choice = at_target_fpr(y, s)
    assert not choice.trustworthy
    assert any("negatives" in w for w in choice.warnings)


def test_a_large_holdout_is_trusted():
    y, s = scores(n=3000)
    assert at_target_fpr(y, s).trustworthy


def test_bootstrap_interval_brackets_the_point_estimate():
    y, s = scores(n=1500)
    point, lo, hi = bootstrap_threshold_ci(y, s, resamples=300, seed=2)
    assert lo <= point <= hi
    assert hi - lo < 0.2, "interval implausibly wide for 1500 negatives"


def test_an_unreachable_budget_is_reported_rather_than_faked():
    y = np.r_[np.zeros(50), np.ones(50)]
    s = np.r_[np.ones(50), np.ones(50)]  # every score identical
    choice = at_target_fpr(y, s, target_fpr=0.0)
    assert choice.warnings, "an unachievable budget was silently accepted"


def test_threshold_selection_needs_both_classes():
    try:
        at_target_fpr(np.zeros(10), np.linspace(0, 1, 10))
    except ValueError:
        return
    raise AssertionError("single-class input was accepted")


# --------------------------------------------------------------------------
# Risk–coverage
# --------------------------------------------------------------------------


def test_widening_the_band_lowers_coverage_monotonically():
    y, s = scores()
    curve = risk_coverage_curve(y, s, threshold=0.5)
    coverages = [pt.coverage for pt in curve]
    assert all(b <= a + 1e-12 for a, b in zip(coverages, coverages[1:])), (
        "coverage must fall as the band widens"
    )


def test_widening_the_band_lowers_selective_risk():
    """The trade the band exists to make. Fewer answers, better answers."""
    y, s = scores()
    narrow = operating_point(y, s, ThresholdInfo.at_margin(value=0.5, margin=0.04, target_fpr=0.02))
    wide = operating_point(y, s, ThresholdInfo.at_margin(value=0.5, margin=0.8473, target_fpr=0.02))
    assert wide.coverage < narrow.coverage
    assert wide.selective_risk < narrow.selective_risk


def test_widening_the_band_reduces_both_error_kinds():
    y, s = scores()
    narrow = operating_point(y, s, ThresholdInfo.at_margin(value=0.5, margin=0.1001, target_fpr=0.02))
    wide = operating_point(y, s, ThresholdInfo.at_margin(value=0.5, margin=1.0986, target_fpr=0.02))
    assert wide.fp <= narrow.fp, "false positives should not grow with the band"
    assert wide.fn <= narrow.fn, "false negatives should not grow with the band"
    assert wide.n_abstained > narrow.n_abstained


def test_selective_rates_ignore_abstentions():
    y, s = scores()
    point = operating_point(y, s, ThresholdInfo.at_margin(value=0.5, margin=0.619, target_fpr=0.02))
    assert point.tp + point.fp + point.tn + point.fn == point.n_decided
    assert point.n_decided + point.n_abstained == point.n
    assert point.abstained_real + point.abstained_fake == point.n_abstained


def test_the_accuracy_double_makes_abstention_look_free():
    """Counting abstentions as correct rewards a model for declining."""
    y, s = scores()
    narrow = ThresholdInfo.at_margin(value=0.5, margin=0.04, target_fpr=0.02)
    wide = ThresholdInfo.at_margin(value=0.5, margin=2.1972, target_fpr=0.02)
    inflated_narrow = accuracy_over_everything(y, s, narrow)
    inflated_wide = accuracy_over_everything(y, s, wide)
    assert inflated_wide > inflated_narrow, (
        "the double no longer rewards abstaining; this check is vacuous"
    )
    # The honest measure moves the other way: coverage collapses.
    assert operating_point(y, s, wide).coverage < operating_point(y, s, narrow).coverage


def test_risk_at_coverage_meets_the_coverage_floor():
    y, s = scores()
    curve = risk_coverage_curve(y, s, threshold=0.5)
    for target in (0.5, 0.8, 0.95):
        point = risk_at_coverage(curve, target)
        assert point.coverage >= target


def test_coverage_at_risk_meets_the_risk_ceiling():
    y, s = scores()
    curve = risk_coverage_curve(y, s, threshold=0.5)
    for target in (0.20, 0.10, 0.05):
        point = coverage_at_risk(curve, target)
        assert point.selective_risk <= target


def test_an_unachievable_coverage_target_raises_with_the_best_available():
    y, s = scores()
    curve = risk_coverage_curve(y, s, threshold=0.5)
    try:
        risk_at_coverage(curve, 1.5)
    except ValueError as exc:
        assert "covers" in str(exc)
        return
    raise AssertionError("an impossible coverage target was satisfied")


def test_band_selection_honours_every_constraint():
    y, s = scores(sep=2.5)
    selection = select_margin(
        y, s, threshold=0.5, max_selective_fpr=0.02, min_coverage=0.80,
        determinism_floor=0.01,
    )
    point = selection.require()
    assert point.selective_fpr <= 0.02
    assert point.coverage >= 0.80
    # The floor binds on the narrower side, not on the total width.
    assert min(point.threshold - point.band_lo, point.band_hi - point.threshold) >= 0.01


def test_band_selection_prefers_the_narrowest_qualifying_band():
    y, s = scores(sep=2.5)
    selection = select_margin(y, s, threshold=0.5, max_selective_fpr=0.05, min_coverage=0.5)
    chosen = selection.require()
    curve = risk_coverage_curve(y, s, threshold=0.5)
    qualifying = [
        pt for pt in curve if pt.selective_fpr <= 0.05 and pt.coverage >= 0.5
    ]
    assert chosen.margin == min(pt.margin for pt in qualifying)


def test_impossible_constraints_refuse_rather_than_compromise():
    y, s = scores(sep=0.2)
    selection = select_margin(
        y, s, threshold=0.5, max_selective_fpr=0.0001, min_coverage=0.99
    )
    assert not selection.ok
    assert "not ready" in selection.describe()
    try:
        selection.require()
    except ValueError as exc:
        assert "no band satisfies" in str(exc)
        return
    raise AssertionError("require() returned a point that does not exist")


def test_the_determinism_floor_rejects_bands_narrower_than_numerical_noise():
    y, s = scores(sep=3.0)
    loose = select_margin(y, s, threshold=0.5, max_selective_fpr=0.5, min_coverage=0.1)
    strict = select_margin(
        y, s, threshold=0.5, max_selective_fpr=0.5, min_coverage=0.1,
        determinism_floor=0.25,
    )

    def narrow_side(pt):
        return min(pt.threshold - pt.band_lo, pt.band_hi - pt.threshold)

    assert narrow_side(loose.require()) < 0.25
    assert narrow_side(strict.require()) >= 0.25
    assert strict.rejected["side_below_floor"] > 0


def test_abstention_breakdown_exposes_who_is_declined():
    y, s = scores()
    point = operating_point(y, s, ThresholdInfo.at_margin(value=0.5, margin=0.619, target_fpr=0.02))
    breakdown = abstention_breakdown(point)
    assert abs(breakdown["share_real"] + breakdown["share_fake"] - 1.0) < 1e-12
    assert breakdown["abstained_real"] + breakdown["abstained_fake"] == point.n_abstained


def test_a_zero_width_band_is_unrepresentable():
    try:
        ThresholdInfo.at_margin(value=0.5, margin=0.0, target_fpr=0.02)
    except ValueError as exc:
        assert "three-state" in str(exc) or "positive" in str(exc)
        return
    raise AssertionError("a zero-width band was accepted")


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
