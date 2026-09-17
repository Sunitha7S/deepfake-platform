"""Probability calibration.

A model's raw output is a ranking, not a probability. Cross-entropy training
produces systematic overconfidence, so a score of 0.98 does not mean "wrong one
time in fifty" — it means "very high in the ranking". Temperature scaling fixes
the scale without touching the ranking.

Two properties follow from that, and both matter downstream:

**Calibration changes no AUC.** It is a monotone transform, so every ranking
metric is identical before and after. Anyone reporting a calibration-driven AUC
improvement has measured something else.

**Calibration is what makes a threshold meaningful.** Choosing an operating
point at 2% false-positive rate is only possible if the numbers being
thresholded mean what they say.

Applied to the **fused** logit, never per-branch. Calibrating each branch and
then fusing recalibrates nothing, because fusion changes the distribution the
temperature was fitted against.

Pure numpy. No scipy, no sklearn — the temperature fit is a one-dimensional
convex minimisation and a ternary search solves it in a few dozen iterations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from contracts.common import DEFAULT_LOG_ODDS_MARGIN, ThresholdInfo

EPS = 1e-12
LOGIT_CLIP = 30.0
"""Beyond ±30 a float64 sigmoid saturates to exactly 0 or 1, and every distinct
score collapses into a tie. Clipping preserves ordering where the arithmetic
would otherwise destroy it."""


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic.

    The naive ``1/(1+exp(-x))`` overflows for x below about -710. Splitting on
    the sign keeps both branches in range.
    """
    arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(arr)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    e = np.exp(arr[~pos])
    out[~pos] = e / (1.0 + e)
    return float(out) if np.isscalar(x) or out.ndim == 0 else out


def inverse_sigmoid(p: np.ndarray | float) -> np.ndarray | float:
    """Logit. Clamps away from 0 and 1 so the result stays finite."""
    arr = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    out = np.log(arr / (1.0 - arr))
    return float(out) if np.isscalar(p) or out.ndim == 0 else out


logit = inverse_sigmoid


def temperature_scale(logits: np.ndarray | float, temperature: float) -> np.ndarray | float:
    """Divide by T, then squash.

    T > 1 softens (less confident), T < 1 sharpens. T = 1 is the identity, which
    is the correct default when no calibration has been fitted — an uncalibrated
    model should behave exactly as it did, not slightly differently.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    return sigmoid(np.clip(scaled, -LOGIT_CLIP, LOGIT_CLIP))


def negative_log_likelihood(
    logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0
) -> float:
    """Mean NLL. The objective the temperature fit minimises.

    NLL rather than ECE on purpose. ECE is binned and therefore discontinuous:
    a model whose predictions concentrate can sit inside one bin before
    calibration and straddle two after, moving ECE sharply with no real change
    in honesty. NLL is bin-free and a proper scoring rule, so when the two
    disagree it is the one to trust.

    THE SUM IS ``math.fsum``, NOT ``np.mean``, AND THAT IS LOAD-BEARING
    ------------------------------------------------------------------
    Floating-point addition is not associative, and numpy's pairwise summation
    visits the terms in array order. So ``np.mean`` over a calibration set
    returns a slightly different number when the same samples arrive in a
    different order — which is an ordinary thing to happen, since a calibration
    set is assembled from a split and a split is a set.

    That difference is around 1e-16 per evaluation and would be invisible, were
    it not for what sits downstream. ``fit_temperature`` compares two NLLs at
    each of 200 ternary-search steps; near convergence the two are equal to
    within that noise, so the comparison flips, the bracket lands elsewhere, and
    the fitted temperature differs at about 1e-8. That number is then
    fingerprinted with full float precision, so an 1e-8 difference produces a
    *completely different* ``effective_model_version``: the result cache misses
    everything, and two runs of the same calibration report as two models.

    ``math.fsum`` is exactly rounded — it returns the correctly rounded sum of
    the inputs regardless of the order they arrive in — so the fit becomes
    bit-identical under permutation. It is not a trade of accuracy for
    determinism; the exactly rounded sum is also the more accurate one.

    It costs roughly 20x a vectorised mean (measured: 400 evaluations over
    100 000 samples, 0.02 s to 2.9 s). That is paid once per offline fit and
    never on the serving path, which calls ``temperature_scale`` and never this.
    """
    p = np.asarray(temperature_scale(logits, temperature), dtype=np.float64)
    p = np.clip(p, EPS, 1.0 - EPS)
    y = np.asarray(labels, dtype=np.float64)
    if y.size == 0:
        raise ValueError("cannot compute NLL on an empty set")
    terms = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    return float(-math.fsum(terms) / y.size)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    lo: float = 0.05,
    hi: float = 20.0,
    iterations: int = 200,
) -> float:
    """Find T minimising NLL, by ternary search on log T.

    NLL in T is unimodal, so ternary search converges without derivatives and
    without scipy. Searching in log space spends its resolution evenly across
    the range rather than crowding it near the upper bound.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if logits.size != labels.size:
        raise ValueError("logits and labels must be the same length")
    if logits.size == 0:
        raise ValueError("cannot fit a temperature on an empty set")
    if len(np.unique(labels)) < 2:
        # One class present: NLL is minimised by pushing T toward a degenerate
        # extreme, which is a fit to an accident of the sample rather than to
        # the model. Refusing beats returning a confident wrong number.
        raise ValueError("temperature fitting needs both classes present")

    a, b = math.log(lo), math.log(hi)
    for _ in range(iterations):
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        if negative_log_likelihood(logits, labels, math.exp(m1)) < negative_log_likelihood(
            logits, labels, math.exp(m2)
        ):
            b = m2
        else:
            a = m1
    return float(math.exp((a + b) / 2.0))


@dataclass(frozen=True)
class CalibrationFit:
    """The outcome of a fit, including whether it should be used.

    ``accepted`` is not decoration. A fit can improve NLL while being unsafe to
    apply — too few samples, or a degenerate model whose logits barely vary —
    and shipping it would replace honest uncertainty with a number nobody
    should trust.
    """

    temperature: float
    nll_before: float
    nll_after: float
    n_samples: int
    accepted: bool
    reason: str | None = None

    @property
    def improvement(self) -> float:
        return self.nll_before - self.nll_after


MIN_FIT_SAMPLES = 200
MIN_LOGIT_STD = 0.05


def fit_and_validate(logits: np.ndarray, labels: np.ndarray) -> CalibrationFit:
    """Fit a temperature and decide whether it is fit to use."""
    logits = np.asarray(logits, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    before = negative_log_likelihood(logits, labels, 1.0)

    std = float(np.std(logits))
    if std < MIN_LOGIT_STD:
        return CalibrationFit(
            temperature=1.0,
            nll_before=before,
            nll_after=before,
            n_samples=logits.size,
            accepted=False,
            reason=(
                f"model is degenerate (logit std {std:.4f} < {MIN_LOGIT_STD}); "
                "calibrating a constant predictor fits noise"
            ),
        )

    t = fit_temperature(logits, labels)
    after = negative_log_likelihood(logits, labels, t)

    if after > before:
        return CalibrationFit(
            temperature=1.0,
            nll_before=before,
            nll_after=before,
            n_samples=logits.size,
            accepted=False,
            reason="fitted temperature worsened NLL; keeping the identity",
        )
    if logits.size < MIN_FIT_SAMPLES:
        return CalibrationFit(
            temperature=t,
            nll_before=before,
            nll_after=after,
            n_samples=logits.size,
            accepted=False,
            reason=f"only {logits.size} samples; {MIN_FIT_SAMPLES} needed to trust a fit",
        )
    return CalibrationFit(
        temperature=t,
        nll_before=before,
        nll_after=after,
        n_samples=logits.size,
        accepted=True,
    )


def threshold_at_fpr(
    scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.02
) -> float:
    """Lowest threshold whose false-positive rate does not exceed ``target_fpr``.

    Chosen on the negatives alone, which is what makes it a *false-positive*
    budget: it is the quantile of authentic scores above which we are willing to
    be wrong. Calling a real person's photograph fake is the costlier error, and
    this is where that judgement is encoded.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    negatives = scores[labels == 0]
    if negatives.size == 0:
        raise ValueError("no negative samples; cannot set a false-positive budget")
    return float(np.quantile(negatives, 1.0 - target_fpr))


def abstention_band(
    threshold: float, *, margin: float = DEFAULT_LOG_ODDS_MARGIN
) -> tuple[float, float]:
    """The interval around the threshold where the model declines to decide.

    Centred on the threshold **in log-odds**, so the decision regions partition
    [0, 1] with the band bracketing the operating point: below ``lo`` authentic,
    at or above ``hi`` manipulated, between them insufficient evidence.

    The cost is explicit and is paid in detections. Scores in
    ``[threshold, hi)`` are calibrated as manipulated and become abstentions
    instead, so the achieved false positive rate is bounded above by the
    threshold's ``target_fpr`` rather than equal to it. ``margin`` should
    therefore come from a measured risk–coverage sweep on the calibration
    holdout — ``evaluation.band.derive_operating_point`` — rather than being
    assumed. The default is a placeholder, and is documented as one.

    Delegates to :meth:`ThresholdInfo.at_margin` rather than recomputing the
    edges, because two implementations of one geometry is how the contract and
    its helper drifted apart the first time. There is one derivation in this
    system and this function calls it.
    """
    info = ThresholdInfo.at_margin(value=threshold, margin=margin, target_fpr=0.0)
    return info.band_lo, info.band_hi
