"""Metrics and calibration, checked against hand-computed values.

Every metric is verified against a number worked out by hand in the test, not
against another implementation. Checking one implementation against another
proves they agree, which is a weaker claim than either being right.

Named doubles for the two mistakes that matter here:

    argsort_auc          breaks ties by array order instead of averaging ranks.
                         Inflates AUC on any saturated detector — which is every
                         trained one.

    trapezoid_pr_auc     interpolates linearly between PR points. Precision is
                         not linear in recall between thresholds, so this
                         reports an area no operating point achieves. Note the
                         direction of the error is *not* fixed — see the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.calibration import (  # noqa: E402
    brier_score,
    evaluate_temperature,
    expected_calibration_error,
    fit_and_report,
    holdout_temperature_check,
    maximum_calibration_error,
    reliability_diagram,
)
from evaluation.metrics import (  # noqa: E402
    average_precision,
    confusion_matrix,
    precision_recall_curve,
    positive_rate,
    roc_auc,
    roc_curve,
    summary,
)
from inference.calibration import inverse_sigmoid  # noqa: E402


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


def argsort_auc(y_true, y_score) -> float:
    """AUC with ties broken by array order. The flattering mistake."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)  # no tie averaging
    n_pos, n_neg = float(np.sum(y == 1)), float(np.sum(y == 0))
    return float((np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def trapezoid_pr_auc(y_true, y_score) -> float:
    """PR area by the trapezoid rule — an interpolation the data does not support.

    Errs in whichever direction the sawtooth happens to run; see the test that
    pins it. Wrong either way, because it is not average precision.
    """
    curve = precision_recall_curve(y_true, y_score)
    rec = np.r_[0.0, curve.recall]
    prec = np.r_[curve.precision[0], curve.precision]
    return float(np.trapezoid(prec, rec)) if hasattr(np, "trapezoid") else float(
        np.trapz(prec, rec)
    )


# --------------------------------------------------------------------------
# Hand-computed metrics
# --------------------------------------------------------------------------

# Four items, scores .9 .8 .7 .6, labels 1 0 1 0.
#   ROC AUC: pairs (pos, neg) where pos outranks neg = (.9>.8),(.9>.6),(.7>.6)
#            = 3 of 4 -> 0.75
#   PR points at thresholds .9/.8/.7/.6 -> precision 1, 1/2, 2/3, 1/2
#                                          recall    1/2, 1/2, 1, 1
#   AP = (0.5-0)*1 + (0.5-0.5)*0.5 + (1-0.5)*(2/3) + 0 = 0.5 + 1/3 = 0.833333…
HAND_Y = [1, 0, 1, 0]
HAND_S = [0.9, 0.8, 0.7, 0.6]


def test_roc_auc_matches_hand_computation():
    assert abs(roc_auc(HAND_Y, HAND_S) - 0.75) < 1e-12


def test_average_precision_matches_hand_computation():
    assert abs(average_precision(HAND_Y, HAND_S) - (0.5 + 1.0 / 3.0)) < 1e-12


def test_precision_recall_curve_points_match_hand_computation():
    curve = precision_recall_curve(HAND_Y, HAND_S)
    assert list(np.round(curve.thresholds, 6)) == [0.9, 0.8, 0.7, 0.6]
    assert list(np.round(curve.precision, 6)) == [1.0, 0.5, round(2 / 3, 6), 0.5]
    assert list(np.round(curve.recall, 6)) == [0.5, 0.5, 1.0, 1.0]


def test_perfect_separation_scores_one():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_inverted_ranking_scores_zero():
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_all_scores_tied_is_chance():
    """A constant predictor ranks nothing. AUC 0.5, AP equal to prevalence."""
    y = [0, 1, 0, 1]
    assert roc_auc(y, [0.5] * 4) == 0.5
    assert abs(average_precision(y, [0.5] * 4) - 0.5) < 1e-12
    assert average_precision([0, 0, 0, 1], [0.5] * 4) == 0.25


def test_average_precision_of_a_random_ranker_approaches_prevalence():
    rng = np.random.default_rng(4)
    y = (rng.random(4000) < 0.2).astype(float)
    s = rng.random(4000)
    assert abs(average_precision(y, s) - positive_rate(y)) < 0.03


def test_tie_handling_is_the_difference_and_the_double_proves_it():
    """Half the scores tied: mean ranks give 0.5, array order flatters."""
    y = [0, 1, 0, 1, 0, 1]
    s = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    assert roc_auc(y, s) == 0.5
    assert argsort_auc(y, s) > 0.5, (
        "the tie-ignoring double no longer inflates; this check is vacuous"
    )


def test_trapezoid_pr_double_disagrees_with_average_precision():
    """The trapezoid rule is not average precision, and the gap is material.

    Worth stating precisely, because the usual warning ("trapezoid
    overestimates AP") is only true of the *interpolated* PR curve. On the raw
    sawtooth the direction flips: admitting a positive jumps precision up at the
    new recall, and the step rule credits the post-jump value while the
    trapezoid averages across the jump — so here it consistently *under*states.
    The defect is not the sign. It is that the number depends on an
    interpolation the data does not support, so it is not comparable with
    anyone else's AP in either direction.
    """
    rng = np.random.default_rng(3)
    y = (rng.random(60) < 0.3).astype(float)
    s = rng.random(60)
    step, trapz = average_precision(y, s), trapezoid_pr_auc(y, s)
    assert abs(trapz - step) > 0.005, (
        "the trapezoid double now agrees with AP; this check is vacuous"
    )
    # On the hand-computed fixture the exact values are known.
    assert abs(average_precision(HAND_Y, HAND_S) - (0.5 + 1 / 3)) < 1e-12
    assert abs(trapezoid_pr_auc(HAND_Y, HAND_S) - (0.5 + 1 / 3)) > 1e-6


def test_roc_curve_starts_at_the_origin_and_reaches_one():
    curve = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert curve.fpr[0] == 0.0 and curve.tpr[0] == 0.0
    assert curve.fpr[-1] == 1.0 and curve.tpr[-1] == 1.0
    assert np.all(np.diff(curve.fpr) >= 0) and np.all(np.diff(curve.tpr) >= 0)


def test_confusion_matrix_counts_at_the_stated_threshold():
    cm = confusion_matrix([0, 0, 1, 1], [0.1, 0.7, 0.4, 0.9], threshold=0.5)
    assert (cm.tn, cm.fp, cm.fn, cm.tp) == (1, 1, 1, 1)
    assert cm.fpr == 0.5 and cm.tpr == 0.5
    assert cm.threshold == 0.5


def test_summary_carries_pr_auc_and_prevalence():
    out = summary([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], threshold=0.5)
    assert out["roc_auc"] == 1.0
    assert out["pr_auc"] == 1.0
    assert out["positive_rate"] == 0.5


def test_metrics_reject_malformed_input():
    for bad in (
        lambda: roc_auc([0, 1], [0.1]),
        lambda: roc_auc([], []),
        lambda: roc_auc([0, 2], [0.1, 0.2]),
        lambda: roc_auc([0, 0], [0.1, 0.2]),
        lambda: average_precision([0, 0], [0.1, 0.2]),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("malformed input was accepted")


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def perfectly_calibrated(per_bin: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Bins whose empirical rate equals their predicted probability exactly."""
    y: list[float] = []
    p: list[float] = []
    for centre in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        positives = round(centre * per_bin)
        y += [1.0] * positives + [0.0] * (per_bin - positives)
        p += [centre] * per_bin
    return np.array(y), np.array(p)


def test_perfect_calibration_scores_zero_to_floating_point():
    """Exactly-calibrated bins score zero, up to float noise.

    Not ``== 0.0``: the mean of a hundred copies of 0.05 is 0.04999999999999999,
    so the gap is ~1e-16 rather than nil. Asserting exact equality here would be
    asserting a property of IEEE-754 rather than of the metric.
    """
    y, p = perfectly_calibrated()
    assert expected_calibration_error(y, p) < 1e-12
    assert maximum_calibration_error(y, p) < 1e-12


def test_brier_matches_hand_computation():
    # ((0.9-1)^2 + (0.2-0)^2) / 2 = (0.01 + 0.04) / 2 = 0.025
    assert abs(brier_score([1, 0], [0.9, 0.2]) - 0.025) < 1e-12
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert brier_score([1, 0], [0.0, 1.0]) == 1.0


def test_overconfidence_is_penalised():
    y, p = perfectly_calibrated()
    stretched = np.clip((p - 0.5) * 3 + 0.5, 0.001, 0.999)
    assert expected_calibration_error(y, stretched) > 0.1
    assert brier_score(y, stretched) > brier_score(y, p)


def test_underconfidence_is_penalised_too():
    y, p = perfectly_calibrated()
    squashed = (p - 0.5) * 0.2 + 0.5
    assert expected_calibration_error(y, squashed) > 0.1


def test_reliability_bins_cover_every_prediction():
    y, p = perfectly_calibrated()
    diagram = reliability_diagram(y, p, n_bins=10)
    assert sum(b.count for b in diagram.bins) == y.size


def test_a_probability_of_exactly_one_lands_in_the_last_bin():
    """Half-open bins would drop it, and ECE would silently ignore it."""
    diagram = reliability_diagram([1, 0], [1.0, 0.0], n_bins=10)
    assert sum(b.count for b in diagram.bins) == 2


def test_quantile_bins_are_steadier_on_concentrated_predictions():
    """The ECE trap: equal-width bins report almost entirely on one bin."""
    rng = np.random.default_rng(0)
    p = np.clip(rng.normal(0.97, 0.005, 4000), 0.0, 1.0)
    y = (rng.random(4000) < p).astype(float)
    uniform = reliability_diagram(y, p, n_bins=10, strategy="uniform")
    quantile = reliability_diagram(y, p, n_bins=10, strategy="quantile")
    assert len(uniform.populated) <= 2, "fixture is not concentrated enough"
    assert len(quantile.populated) >= 5, "quantile bins failed to spread the mass"


def test_temperature_scaling_improves_miscalibrated_data():
    y, p = perfectly_calibrated()
    stretched = np.clip((p - 0.5) * 3 + 0.5, 0.001, 0.999)
    logits = inverse_sigmoid(stretched)
    report = fit_and_report(logits, y)
    assert report.nll_improved, report.describe()
    assert report.ece_improved, report.describe()
    assert report.ece_after < 0.05
    assert report.temperature > 1.0, "an overconfident model needs softening"


def test_temperature_scaling_never_changes_the_ranking():
    """Monotone transform: AUC must be bit-identical, not merely close."""
    rng = np.random.default_rng(2)
    logits = rng.normal(0, 3, 2000)
    y = (rng.random(2000) < 1 / (1 + np.exp(-logits))).astype(float)
    for t in (0.5, 1.0, 2.0, 7.5):
        report = evaluate_temperature(logits, y, t)
        assert report.ranking_preserved, report.describe()
        assert report.auc_before == report.auc_after


def test_identity_temperature_changes_nothing():
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 2, 500)
    y = (rng.random(500) < 0.5).astype(float)
    report = evaluate_temperature(logits, y, 1.0)
    assert report.nll_before == report.nll_after
    assert report.ece_before == report.ece_after
    assert report.brier_before == report.brier_after


def test_a_holdout_check_catches_a_temperature_fitted_to_noise():
    """The gate the in-sample refusals cannot provide.

    ``fit_and_validate`` computes all three of its refusals on the data the
    temperature was fitted to, so a fit to sampling noise passes them. This one
    reports NLL on data the fit never saw.
    """
    rng = np.random.default_rng(9)
    logits = rng.normal(0, 2, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-logits))).astype(float)
    t, before, after = holdout_temperature_check(logits, y, seed=1)
    assert 0.5 < t < 2.0, f"already-calibrated data should need T near 1, got {t}"
    assert after <= before + 0.02, "held-out NLL worsened materially"


def test_calibration_rejects_probabilities_outside_the_unit_interval():
    try:
        expected_calibration_error([0, 1], [0.5, 1.5])
    except ValueError:
        return
    raise AssertionError("a probability above 1 was accepted")


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
