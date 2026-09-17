"""Calibration tests.

Calibration is the only thing that makes a probability mean what it says, and
it is easy to get subtly wrong in ways no downstream test would notice — a
temperature that silently shifts scores, an overflow that collapses distinct
values into ties. These are mostly property tests for that reason: they assert
invariants that must hold for every input rather than checking a handful of
points.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.calibration import (  # noqa: E402
    MIN_FIT_SAMPLES,
    CalibrationFit,
    abstention_band,
    fit_and_validate,
    fit_temperature,
    inverse_sigmoid,
    negative_log_likelihood,
    sigmoid,
    temperature_scale,
    threshold_at_fpr,
)


def _miscalibrated(n: int = 4000, true_t: float = 2.5, seed: int = 0):
    """Logits deliberately over-confident by a known factor."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 3.0, n)
    labels = (rng.random(n) < sigmoid(logits / true_t)).astype(np.float64)
    return logits, labels


# --------------------------------------------------------------------------
# sigmoid / inverse
# --------------------------------------------------------------------------


def test_sigmoid_known_points():
    assert sigmoid(0.0) == 0.5
    assert abs(float(sigmoid(1.0)) - 0.7310585786) < 1e-9
    assert abs(float(sigmoid(-1.0)) - 0.2689414214) < 1e-9


def test_sigmoid_is_bounded_for_every_input():
    x = np.concatenate([np.linspace(-1000, 1000, 5000), [np.inf, -np.inf]])
    x = x[np.isfinite(x)]
    p = np.asarray(sigmoid(x))
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_sigmoid_does_not_overflow_at_extremes():
    """The naive 1/(1+exp(-x)) overflows below about -710."""
    with np.errstate(over="raise", invalid="raise"):
        assert float(sigmoid(-800.0)) == 0.0
        assert float(sigmoid(800.0)) == 1.0
        assert float(sigmoid(-1e6)) == 0.0


def test_sigmoid_is_monotonic():
    x = np.linspace(-40, 40, 2001)
    p = np.asarray(sigmoid(x))
    assert np.all(np.diff(p) >= 0.0)


def test_inverse_sigmoid_round_trips():
    for value in (-8.0, -2.5, -0.1, 0.0, 0.1, 2.5, 8.0):
        assert abs(float(inverse_sigmoid(sigmoid(value))) - value) < 1e-6


def test_inverse_sigmoid_clamps_rather_than_diverging():
    """p exactly 0 or 1 has infinite logit; clamping keeps it finite."""
    assert math.isfinite(float(inverse_sigmoid(0.0)))
    assert math.isfinite(float(inverse_sigmoid(1.0)))


# --------------------------------------------------------------------------
# temperature scaling
# --------------------------------------------------------------------------


def test_temperature_one_is_exact_identity():
    """Not approximately — exactly. An uncalibrated model must behave as it did,
    not slightly differently."""
    for value in (-12.0, -1.0, 0.0, 0.75, 12.0):
        assert float(temperature_scale(value, 1.0)) == float(sigmoid(value))


def test_temperature_scaling_stays_bounded():
    x = np.linspace(-500, 500, 4001)
    for t in (0.1, 0.5, 1.0, 3.0, 20.0):
        p = np.asarray(temperature_scale(x, t))
        assert np.all(p >= 0.0) and np.all(p <= 1.0), f"T={t} escaped [0,1]"


def test_temperature_scaling_preserves_order():
    """Calibration is monotone, which is why it changes no ranking metric.

    Anyone reporting a calibration-driven AUC improvement has measured
    something else.
    """
    x = np.linspace(-20, 20, 1001)
    for t in (0.25, 1.0, 4.0):
        p = np.asarray(temperature_scale(x, t))
        assert np.all(np.diff(p) >= 0.0), f"T={t} reordered scores"


def test_high_temperature_softens_and_low_sharpens():
    strong = 3.0
    assert float(temperature_scale(strong, 5.0)) < float(temperature_scale(strong, 1.0))
    assert float(temperature_scale(strong, 0.3)) > float(temperature_scale(strong, 1.0))


def test_non_positive_temperature_is_rejected():
    for bad in (0.0, -1.0):
        try:
            temperature_scale(1.0, bad)
        except ValueError:
            continue
        raise AssertionError(f"temperature {bad} was accepted")


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def test_fitting_recovers_a_known_temperature():
    logits, labels = _miscalibrated(true_t=2.5)
    assert abs(fit_temperature(logits, labels) - 2.5) < 0.25


def test_fitting_recovers_a_second_known_temperature():
    """One recovery could be luck; two with different values is the property."""
    logits, labels = _miscalibrated(true_t=0.6, seed=7)
    assert abs(fit_temperature(logits, labels) - 0.6) < 0.15


def test_fitting_improves_nll():
    logits, labels = _miscalibrated()
    t = fit_temperature(logits, labels)
    before = negative_log_likelihood(logits, labels, 1.0)
    after = negative_log_likelihood(logits, labels, t)
    assert after < before, f"NLL got worse: {before:.4f} -> {after:.4f}"


def test_fitting_leaves_already_calibrated_scores_alone():
    logits, labels = _miscalibrated(true_t=1.0, seed=11)
    assert abs(fit_temperature(logits, labels) - 1.0) < 0.2


def test_fitting_refuses_a_single_class():
    """NLL is minimised by a degenerate extreme when only one label is present —
    a fit to an accident of the sample, not to the model."""
    for labels in (np.zeros(500), np.ones(500)):
        try:
            fit_temperature(np.random.default_rng(0).normal(size=500), labels)
        except ValueError as exc:
            assert "both classes" in str(exc)
            continue
        raise AssertionError("single-class fit was accepted")


def test_fitting_rejects_empty_and_mismatched_input():
    for logits, labels in (
        (np.array([]), np.array([])),
        (np.zeros(5), np.zeros(4)),
    ):
        try:
            fit_temperature(logits, labels)
        except ValueError:
            continue
        raise AssertionError("invalid input was accepted")


# --------------------------------------------------------------------------
# fit validation
# --------------------------------------------------------------------------


def test_fit_is_accepted_on_good_data():
    fit = fit_and_validate(*_miscalibrated())
    assert isinstance(fit, CalibrationFit)
    assert fit.accepted is True
    assert fit.improvement > 0


def test_degenerate_model_is_refused():
    """A near-constant predictor has nothing to calibrate; fitting one fits noise."""
    logits = np.full(1000, 0.5) + np.random.default_rng(0).normal(0, 0.001, 1000)
    labels = (np.arange(1000) % 2).astype(np.float64)
    fit = fit_and_validate(logits, labels)
    assert fit.accepted is False
    assert fit.temperature == 1.0
    assert "degenerate" in (fit.reason or "")


def test_small_sample_is_refused_but_reported():
    logits, labels = _miscalibrated(n=50)
    fit = fit_and_validate(logits, labels)
    assert fit.accepted is False
    assert str(MIN_FIT_SAMPLES) in (fit.reason or "")
    assert fit.temperature != 1.0, "the fit is still reported, just not trusted"


# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------


def test_threshold_at_fpr_uses_negatives_only():
    """It is a false-positive budget: the quantile of authentic scores above
    which we accept being wrong."""
    scores = np.concatenate([np.linspace(0.0, 1.0, 100), np.full(100, 0.99)])
    labels = np.concatenate([np.zeros(100), np.ones(100)])
    t = threshold_at_fpr(scores, labels, target_fpr=0.05)
    fpr = float(np.mean(scores[labels == 0] >= t))
    assert fpr <= 0.06, f"observed FPR {fpr}"


def test_tighter_fpr_gives_a_higher_threshold():
    scores = np.concatenate([np.linspace(0, 1, 200), np.linspace(0.5, 1, 200)])
    labels = np.concatenate([np.zeros(200), np.ones(200)])
    assert threshold_at_fpr(scores, labels, 0.01) >= threshold_at_fpr(scores, labels, 0.20)


def test_threshold_needs_negatives():
    try:
        threshold_at_fpr(np.ones(10), np.ones(10))
    except ValueError as exc:
        assert "negative" in str(exc)
        return
    raise AssertionError("a threshold was produced with no negatives")


def test_abstention_band_is_centred_on_the_threshold():
    """The band brackets the operating point, so the regions partition [0, 1].

    It used to sit entirely below the threshold, which left ``band_hi`` and
    ``value`` free to be the same number — and therefore free to drift apart
    without anything noticing.

    Since Batch 7 "centred" means centred in log-odds, so the two sides are not
    equal in probability. The property that survives is the one that mattered:
    the band brackets its threshold, and the parameter is recoverable from the
    edges.
    """
    lo, hi = abstention_band(0.62, margin=0.4055)
    assert lo < 0.62 < hi, "the band must bracket its threshold"
    below = inverse_sigmoid(0.62) - inverse_sigmoid(lo)
    above = inverse_sigmoid(hi) - inverse_sigmoid(0.62)
    assert abs(below - above) < 1e-9, "the band must be centred in log-odds"
    assert abs(above - 0.4055) < 1e-9, "the band must have the requested margin"


def test_abstention_band_no_longer_runs_off_the_scale():
    """R11's closure at the helper.

    Every one of these threshold/width pairs was impossible under the old
    geometry — the whole point of the change is that a band is now available at
    each of them. The margins here are the ones producing comparable widths.
    """
    for threshold, margin in ((0.1, 1.2), (0.9, 1.0), (0.5, 3.0)):
        lo, hi = abstention_band(threshold, margin=margin)
        assert 0.0 < lo < threshold < hi < 1.0, (
            f"a margin of {margin} around {threshold} did not produce a band"
        )


def test_abstention_band_rejects_a_zero_margin():
    try:
        abstention_band(0.62, margin=0.0)
    except ValueError as exc:
        assert "margin must be" in str(exc)
        return
    raise AssertionError("a zero-width abstention band was accepted")


def test_abstention_band_refuses_a_saturated_threshold():
    """What is left of "the band does not fit": a threshold with no odds."""
    for threshold in (0.0, 1.0):
        try:
            abstention_band(threshold)
        except ValueError as exc:
            assert "strictly inside (0, 1)" in str(exc)
            continue
        raise AssertionError(f"a threshold of {threshold} was accepted")


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
