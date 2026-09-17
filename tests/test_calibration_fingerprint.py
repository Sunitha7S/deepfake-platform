"""R1: a calibration's fingerprint must not depend on the order of its data.

THE DEFECT, RESTATED
--------------------
A calibration set is assembled from a split, and a split is a set. Nothing
promises an order, and re-running an evaluation can legitimately produce the
same samples in a different sequence.

Under the old implementation that changed the fitted temperature by about 1e-8 —
floating-point addition is not associative, ``np.mean`` visits terms in array
order, and ``fit_temperature`` compares two NLLs that are equal to within that
noise at each of 200 ternary-search steps, so the comparison flips and the
bracket lands elsewhere. The temperature is then fingerprinted at full float
precision, so 1e-8 produced a completely different ``effective_model_version``:
the result cache missed everything and two runs of one calibration reported as
two models.

WHAT THE TESTS ESTABLISH, IN ORDER
----------------------------------
1. ``test_the_old_summation_is_still_order_sensitive`` reproduces the *cause*
   with a local ``np.mean`` implementation, so the premise is demonstrated
   rather than asserted. If this ever stops failing, numpy has changed and the
   fix may be reconsidered — but not before.
2. The NLL, the temperature, and the fingerprint are each order-invariant now.
3. Genuinely different calibrations still fingerprint differently — the fix must
   not have been bought by making the detector blind.
4. The resolution of the quantiser is pinned in both directions, so the trade it
   makes is a decision on the record rather than an accident.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import STUB_ENGINE_VERSION  # noqa: E402
from contracts.common import DEFAULT_LOG_ODDS_MARGIN, ThresholdInfo  # noqa: E402
from contracts.version import (  # noqa: E402
    FINGERPRINT_DECIMALS,
    fingerprint,
    quantize,
)
from inference.calibration import (  # noqa: E402
    EPS,
    fit_temperature,
    negative_log_likelihood,
    temperature_scale,
)
from inference.models import CalibrationSpec, ModelSpec  # noqa: E402


def calibration_data(n: int = 5000, seed: int = 7):
    """Separable-but-overlapping logits, deterministic and both classes present."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(np.float64)
    z = rng.normal(0.0, 3.0, n) + 2.0 * (2.0 * y - 1.0)
    return z, y


def shuffled(z, y, seed: int = 99):
    idx = np.random.default_rng(seed).permutation(z.size)
    return z[idx], y[idx]


# --------------------------------------------------------------------------
# The premise: the old summation really was order-sensitive
# --------------------------------------------------------------------------


def legacy_nll(logits, labels, temperature: float = 1.0) -> float:
    """The pre-Batch-4 implementation: ``np.mean`` over the per-sample terms."""
    p = np.asarray(temperature_scale(logits, temperature), dtype=np.float64)
    p = np.clip(p, EPS, 1.0 - EPS)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def legacy_fit(logits, labels, *, lo=0.05, hi=20.0, iterations=200) -> float:
    """The pre-Batch-4 fit, differing only in which sum it uses."""
    a, b = math.log(lo), math.log(hi)
    for _ in range(iterations):
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        if legacy_nll(logits, labels, math.exp(m1)) < legacy_nll(
            logits, labels, math.exp(m2)
        ):
            b = m2
        else:
            a = m1
    return float(math.exp((a + b) / 2.0))


SWEEP = (0.3, 0.5, 0.8, 1.0, 1.3, 1.7, 2.0, 2.5, 3.0, 4.0, 6.0)


def test_the_old_summation_is_still_order_sensitive():
    """The premise of the whole fix, demonstrated rather than assumed.

    A sweep rather than a single temperature, because that is the actual shape
    of the defect: ``np.mean`` is not *guaranteed* order-invariant, not
    reliably-wrong. At most temperatures the pairwise sums happen to agree
    exactly; at a few they do not (measured: 3 of 11 on this fixture). One
    disagreement anywhere in the sweep is enough, because ``fit_temperature``
    evaluates the objective at hundreds of temperatures and only needs the
    comparison to flip once.
    """
    z, y = calibration_data()
    zs, ys = shuffled(z, y)
    divergent = [t for t in SWEEP if legacy_nll(z, y, t) != legacy_nll(zs, ys, t)]
    assert divergent, (
        "np.mean agreed at every temperature in the sweep; the motivation for "
        "math.fsum should be re-examined before this test is deleted"
    )


def test_the_old_fit_produced_a_different_temperature_and_fingerprint():
    """The measured consequence: ~1e-8 in T, a wholly different fingerprint."""
    z, y = calibration_data()
    zs, ys = shuffled(z, y)
    t_a, t_b = legacy_fit(z, y), legacy_fit(zs, ys)
    assert t_a != t_b, "the legacy fit no longer drifts; premise broken"
    assert abs(t_a - t_b) < 1e-5, (
        f"the legacy drift was {abs(t_a - t_b):.2e}, far larger than the ~1e-8 "
        "float noise this fix addresses — something else is wrong"
    )
    assert fingerprint({"t": t_a}) != fingerprint({"t": t_b}), (
        "a sub-1e-5 difference must produce different raw fingerprints; that is "
        "exactly why the fix is upstream of the fingerprint"
    )


# --------------------------------------------------------------------------
# The fix: order invariance, layer by layer
# --------------------------------------------------------------------------


def test_the_nll_is_order_invariant():
    z, y = calibration_data()
    zs, ys = shuffled(z, y)
    for t in SWEEP:
        assert negative_log_likelihood(z, y, t) == negative_log_likelihood(zs, ys, t), (
            f"NLL moved under reordering at T={t}"
        )


def test_the_fitted_temperature_is_bit_identical_under_reordering():
    """Equality, not ``isclose``. Anything short of bit-identical re-opens R1."""
    z, y = calibration_data()
    for seed in (1, 42, 99):
        zs, ys = shuffled(z, y, seed=seed)
        assert fit_temperature(z, y) == fit_temperature(zs, ys), (
            f"the temperature moved under permutation seed {seed}"
        )


def test_the_calibration_fingerprint_is_order_invariant():
    """The property R1 actually asked for, at the level it is consumed."""
    z, y = calibration_data()
    zs, ys = shuffled(z, y)

    def spec_for(logits, labels) -> ModelSpec:
        t = fit_temperature(logits, labels)
        return ModelSpec(
            name="m",
            version="w-1",
            gate_fingerprint="g0",
            calibration=CalibrationSpec(
                temperature=t, fitted_on="ffpp-test", n_samples=int(labels.size)
            ),
        )

    assert (
        spec_for(z, y).calibration_fingerprint == spec_for(zs, ys).calibration_fingerprint
    )
    assert spec_for(z, y).effective_version == spec_for(zs, ys).effective_version


def test_reversing_the_calibration_set_changes_nothing():
    """Reversal is the ordering a real pipeline is most likely to produce."""
    z, y = calibration_data(n=2000)
    assert fit_temperature(z, y) == fit_temperature(z[::-1].copy(), y[::-1].copy())


def test_calibration_correctness_is_preserved():
    """Determinism must not have been bought with a worse fit."""
    z, y = calibration_data()
    t = fit_temperature(z, y)
    assert 0.05 < t < 20.0
    assert negative_log_likelihood(z, y, t) <= negative_log_likelihood(z, y, 1.0), (
        "the fitted temperature does not improve NLL on its own fitting set"
    )
    # The exactly-rounded sum should agree with the naive one to well within
    # float tolerance — this is a more accurate sum, not a different objective.
    assert abs(negative_log_likelihood(z, y, 1.3) - legacy_nll(z, y, 1.3)) < 1e-12


def test_the_fit_still_refuses_a_single_class_set():
    z, _ = calibration_data(n=400)
    try:
        fit_temperature(z, np.ones(z.size))
    except ValueError as exc:
        assert "both classes" in str(exc)
        return
    raise AssertionError("a single-class calibration set was accepted")


def test_the_nll_refuses_an_empty_set():
    try:
        negative_log_likelihood(np.array([]), np.array([]), 1.0)
    except ValueError as exc:
        assert "empty" in str(exc)
        return
    raise AssertionError("NLL over nothing returned a number")


# --------------------------------------------------------------------------
# The detector is still a detector
# --------------------------------------------------------------------------


def base_spec(**overrides) -> ModelSpec:
    kwargs = {
        "name": "m",
        "version": "w-1",
        "gate_fingerprint": "g0",
        "calibration": CalibrationSpec(
            temperature=1.4, fitted_on="ffpp-test", n_samples=2000
        ),
        "threshold": ThresholdInfo.at_margin(value=0.62, margin=0.4299, target_fpr=0.02),
    }
    kwargs.update(overrides)
    return ModelSpec(**kwargs)


def test_a_genuinely_different_temperature_still_differs():
    a = base_spec()
    b = base_spec(
        calibration=CalibrationSpec(temperature=1.5, fitted_on="ffpp-test", n_samples=2000)
    )
    assert a.calibration_fingerprint != b.calibration_fingerprint
    assert a.effective_version != b.effective_version


def test_a_genuinely_different_threshold_still_differs():
    a = base_spec()
    b = base_spec(
        threshold=ThresholdInfo.at_margin(value=0.70, margin=0.481, target_fpr=0.02)
    )
    assert a.calibration_fingerprint != b.calibration_fingerprint


def test_a_different_band_width_still_differs():
    a = base_spec()
    b = base_spec(
        threshold=ThresholdInfo.at_margin(value=0.62, margin=0.6555, target_fpr=0.02)
    )
    assert a.calibration_fingerprint != b.calibration_fingerprint


def test_the_same_temperature_for_different_reasons_still_differs():
    """T=1.0 uncalibrated and T=1.0 fitted are not interchangeable."""
    a = base_spec(calibration=CalibrationSpec(temperature=1.0))
    b = base_spec(
        calibration=CalibrationSpec(temperature=1.0, fitted_on="ffpp-test", n_samples=2000)
    )
    assert a.calibration_fingerprint != b.calibration_fingerprint


def test_a_decision_relevant_temperature_difference_survives_quantisation():
    """1e-4 in T is the smallest difference anyone could argue about. It survives."""
    a = base_spec(calibration=CalibrationSpec(temperature=1.4000))
    b = base_spec(calibration=CalibrationSpec(temperature=1.4001))
    assert a.calibration_fingerprint != b.calibration_fingerprint


# --------------------------------------------------------------------------
# The quantiser, and the trade it makes
# --------------------------------------------------------------------------


def test_quantise_folds_noise_below_its_resolution():
    assert quantize(1.4000000001) == quantize(1.4)
    assert quantize(2.1937994756748207) == quantize(2.193799452081738), (
        "the two temperatures measured from the original defect still differ "
        "after quantisation"
    )


def test_quantise_preserves_decision_relevant_differences():
    assert quantize(0.6200) != quantize(0.6201)
    assert quantize(1.4) != quantize(1.5)


def test_quantise_resolution_is_pinned_in_both_directions():
    """The trade, on the record: 1e-6 is kept, 1e-8 is not."""
    assert FINGERPRINT_DECIMALS == 6
    assert quantize(1.0 + 1e-6) != quantize(1.0)
    assert quantize(1.0 + 1e-8) == quantize(1.0)


def test_quantise_does_not_split_positive_and_negative_zero():
    assert repr(quantize(-0.0)) == repr(quantize(0.0))
    assert fingerprint({"v": quantize(-1e-9)}) == fingerprint({"v": quantize(0.0)})


def test_quantise_refuses_a_non_finite_fit():
    """A failed fit must not be publishable as a version string."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        try:
            quantize(bad)
        except ValueError as exc:
            assert "non-finite" in str(exc)
        else:
            raise AssertionError(f"{bad!r} was accepted as a fingerprintable value")


def test_the_general_fingerprint_still_distinguishes_neighbouring_floats():
    """The invariant this fix was routed around, re-asserted from here.

    ``_canonical`` is untouched: it preserves floats exactly, because it hashes
    exact values — branch weights, media ids — as well as fitted ones, and a
    detector that rounds everything is a detector with a blind spot. Rounding
    happens at the call site, on fitted quantities only.
    """
    assert fingerprint({"t": 1.0}) != fingerprint({"t": 1.0000000000000002})


# --------------------------------------------------------------------------
# No churn for the shipped deployment
# --------------------------------------------------------------------------


def test_the_configured_defaults_are_unaffected_by_quantisation():
    """0.62, 0.02 and 1.0 all round to themselves.

    Worth pinning: had any *configured* default landed on a value like
    ``0.6199999999999999``, quantisation would have silently invalidated every
    cached result in production.

    The band edges are deliberately not in this list any more. Since Batch 7
    they are computed — ``expit(logit(0.62) ± ln 1.5)`` — rather than typed, so
    they are irrational-looking by nature and quantisation is doing exactly the
    job it exists for. The next test is what pins their stability.
    """
    spec = ModelSpec(name="stub", version="stub-0")
    for value in (
        spec.calibration.temperature,
        spec.threshold.value,
        spec.threshold.target_fpr,
    ):
        assert quantize(value) == value, f"{value!r} moved under quantisation"


def test_the_derived_band_edges_quantise_stably():
    """The property that actually matters for the cache, now that the edges are
    computed: quantisation must be idempotent and must not move the fingerprint.

    A recomputed edge can differ from a stored one in the last bit or two —
    different numpy build, different libm — and that is what ``quantize`` is
    for. What would be a defect is an edge sitting astride a rounding boundary,
    where the two representations quantise to *different* values and the cache
    key splits. Checked by perturbing each edge by a few ULPs and requiring the
    fingerprint to survive.
    """
    spec = ModelSpec(name="stub", version="stub-0")
    baseline = spec.calibration_fingerprint

    for edge in (spec.threshold.band_lo, spec.threshold.band_hi):
        q = quantize(edge)
        assert quantize(q) == q, f"quantisation is not idempotent at {edge!r}"
        assert abs(q - edge) < 1e-6

    perturbed = replace(
        spec,
        threshold=ThresholdInfo.at_margin(
            value=0.62,
            # A margin one part in 1e12 away: far below anything that moves a
            # verdict, and enough to move the raw edges in their last bits.
            margin=DEFAULT_LOG_ODDS_MARGIN * (1 + 1e-12),
            target_fpr=0.02,
        ),
    )
    assert perturbed.threshold.band_lo != spec.threshold.band_lo, (
        "the perturbation did not move the raw edges; this check is vacuous"
    )
    assert perturbed.calibration_fingerprint == baseline, (
        "a sub-picodigit change in the margin split the cache key; the shipped "
        "band edges sit astride a quantisation boundary and the default should "
        "be nudged"
    )


def test_the_shipped_stub_version_is_current():
    spec = ModelSpec(name="stub", version="stub-0")
    assert STUB_ENGINE_VERSION.startswith(spec.effective_version), (
        "the configured stub version no longer matches the composed one; "
        "update backend.config.STUB_ENGINE_VERSION"
    )


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
