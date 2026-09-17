"""Threshold selection: four procedures, one record of what was chosen and why.

Every function here returns a :class:`ThresholdChoice` rather than a bare float.
A threshold without the data it was fitted on, the rule that produced it and the
operating point it achieves is a number nobody can reproduce or argue with — and
it is half of the decision this whole system exists to make.

THE ASYMMETRY THAT DRIVES ALL OF THIS
-------------------------------------
Calling a real person's photograph a forgery is the costlier error. So the
default procedure fixes a **false-positive budget** and accepts whatever
detection rate follows, rather than maximising a symmetric score. That is why
``at_target_fpr`` is the one wired into the pipeline and the other three are
provided for comparison and for callers whose economics genuinely differ.

If the achieved TPR at the chosen budget is too low, the answer is not to widen
the budget. The answer is that the model is not ready.

EMPIRICAL, NOT INTERPOLATED
---------------------------
``inference.calibration.threshold_at_fpr`` takes the ``1 - target_fpr`` quantile
of the negatives, which interpolates between observed scores. That is fine for a
first cut and can miss the target on the actual sample. The procedures here
sweep the observed scores instead and report the operating point that will in
fact be achieved on this data, which is what a release gate needs to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evaluation.metrics import confusion_matrix

MIN_NEGATIVES_FOR_FPR = 500
"""Below this a false-positive budget is not measurable.

At a 2% budget the threshold is the 98th percentile of the authentic class. A
stable extreme quantile needs roughly ten samples beyond it, so 500 negatives is
the floor — at 500 exactly ten sit above the threshold — and 2000 is the number
to work with.
"""


@dataclass(frozen=True)
class ThresholdChoice:
    """A chosen operating point and everything needed to reproduce or audit it."""

    value: float
    method: str
    params: dict[str, float]
    achieved_fpr: float
    achieved_tpr: float
    achieved_precision: float
    achieved_f1: float
    expected_cost: float
    n_positive: int
    n_negative: int
    warnings: list[str] = field(default_factory=list)

    @property
    def achieved_fnr(self) -> float:
        return 1.0 - self.achieved_tpr

    @property
    def trustworthy(self) -> bool:
        return not self.warnings

    def describe(self) -> str:
        head = (
            f"{self.method} -> threshold {self.value:.6f}\n"
            f"  achieved  FPR {self.achieved_fpr:.4f}  TPR {self.achieved_tpr:.4f}  "
            f"precision {self.achieved_precision:.4f}  F1 {self.achieved_f1:.4f}\n"
            f"  on        {self.n_positive} positive / {self.n_negative} negative"
        )
        if self.warnings:
            head += "\n" + "\n".join(f"  warning  {w}" for w in self.warnings)
        return head


def _prepare(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if y.size != s.size:
        raise ValueError(f"length mismatch: {y.size} vs {s.size}")
    if y.size == 0:
        raise ValueError("empty input")
    bad = set(np.unique(y)) - {0.0, 1.0}
    if bad:
        raise ValueError(f"y_true must be 0/1, found {sorted(bad)}")
    if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        raise ValueError("threshold selection needs both classes present")
    return y, s


def candidate_thresholds(y_score: np.ndarray) -> np.ndarray:
    """Every threshold that produces a distinct partition, highest first.

    The observed scores plus one above the maximum. Sweeping a fixed grid
    instead would miss the exact operating points a saturated detector actually
    offers and report a threshold that no score ever crosses.
    """
    s = np.asarray(y_score, dtype=np.float64).ravel()
    return np.r_[np.unique(s)[::-1], -np.inf]


def _point(y: np.ndarray, s: np.ndarray, threshold: float) -> tuple[float, float, float, float]:
    cm = confusion_matrix(y, s, threshold=threshold)
    prec = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) else 0.0
    f1v = 2 * prec * cm.tpr / (prec + cm.tpr) if (prec + cm.tpr) else 0.0
    return cm.fpr, cm.tpr, float(prec), float(f1v)


def _finish(
    y: np.ndarray,
    s: np.ndarray,
    threshold: float,
    method: str,
    params: dict[str, float],
    *,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    prevalence: float | None = None,
    warnings: list[str] | None = None,
) -> ThresholdChoice:
    fpr, tpr, prec, f1v = _point(y, s, threshold)
    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    pi = float(np.mean(y == 1)) if prevalence is None else float(prevalence)
    cost = cost_fp * fpr * (1.0 - pi) + cost_fn * (1.0 - tpr) * pi

    notes = list(warnings or [])
    if n_neg < MIN_NEGATIVES_FOR_FPR:
        notes.append(
            f"only {n_neg} negatives; a false-positive budget needs at least "
            f"{MIN_NEGATIVES_FOR_FPR} for the threshold to be stable"
        )
    return ThresholdChoice(
        value=float(threshold),
        method=method,
        params=dict(params),
        achieved_fpr=fpr,
        achieved_tpr=tpr,
        achieved_precision=prec,
        achieved_f1=f1v,
        expected_cost=float(cost),
        n_positive=n_pos,
        n_negative=n_neg,
        warnings=notes,
    )


# --------------------------------------------------------------------------
# The four procedures
# --------------------------------------------------------------------------


def at_target_fpr(
    y_true: np.ndarray, y_score: np.ndarray, *, target_fpr: float = 0.02
) -> ThresholdChoice:
    """Lowest threshold whose false-positive rate stays within the budget.

    Lowest, not any — among all thresholds meeting the budget this is the one
    with the highest detection rate, so the budget is spent rather than merely
    respected.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must lie in [0, 1], got {target_fpr}")
    y, s = _prepare(y_true, y_score)

    chosen, warnings = None, []
    for t in candidate_thresholds(s):
        if confusion_matrix(y, s, threshold=float(t)).fpr <= target_fpr:
            chosen = float(t)
        else:
            break
    if chosen is None:
        # Even the strictest threshold exceeds the budget: every negative
        # scores at or above the top of the range.
        chosen = float(np.max(s)) + 1.0
        warnings.append(
            f"no threshold achieves FPR <= {target_fpr}; the strictest possible "
            "operating point still exceeds the budget, so nothing is detectable "
            "within it"
        )
    return _finish(y, s, chosen, "at_target_fpr", {"target_fpr": target_fpr}, warnings=warnings)


def at_target_tpr(
    y_true: np.ndarray, y_score: np.ndarray, *, target_tpr: float = 0.90
) -> ThresholdChoice:
    """Highest threshold that still detects at least ``target_tpr`` of fakes.

    The mirror of ``at_target_fpr``: fixes a detection floor and minimises false
    positives underneath it. Useful when a regulator specifies recall; it is not
    this product's default, because a detection floor with no cost ceiling can
    quietly accept an unbounded false-positive rate.
    """
    if not 0.0 <= target_tpr <= 1.0:
        raise ValueError(f"target_tpr must lie in [0, 1], got {target_tpr}")
    y, s = _prepare(y_true, y_score)

    chosen, warnings = None, []
    for t in candidate_thresholds(s):
        if confusion_matrix(y, s, threshold=float(t)).tpr >= target_tpr:
            chosen = float(t)
            break
    if chosen is None:
        chosen = float(np.min(s))
        warnings.append(
            f"no threshold reaches TPR >= {target_tpr}; even calling everything "
            "manipulated falls short"
        )
    return _finish(y, s, chosen, "at_target_tpr", {"target_tpr": target_tpr}, warnings=warnings)


def at_max_f1(y_true: np.ndarray, y_score: np.ndarray) -> ThresholdChoice:
    """Threshold maximising F1. Ties break toward the stricter threshold.

    Reported because it is asked for, and worth distrusting here: F1 weights a
    false positive and a false negative equally, which is exactly the symmetry
    this product rejects. Use it to compare detectors, never to ship one.
    """
    y, s = _prepare(y_true, y_score)
    best_t, best_f1 = None, -1.0
    for t in candidate_thresholds(s):
        _, _, _, f1v = _point(y, s, float(t))
        if f1v > best_f1:  # strict: the first (highest) threshold wins ties
            best_t, best_f1 = float(t), f1v
    return _finish(
        y,
        s,
        best_t,
        "at_max_f1",
        {},
        warnings=[
            "F1 weights false positives and false negatives equally; this "
            "product does not. Compare with at_target_fpr before shipping."
        ],
    )


def at_min_expected_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    cost_fp: float = 10.0,
    cost_fn: float = 1.0,
    prevalence: float | None = None,
) -> ThresholdChoice:
    """Threshold minimising ``cost_fp·FPR·(1−π) + cost_fn·FNR·π``.

    ``prevalence`` is the fraction of genuinely manipulated media **in
    production**, and defaults to the rate in ``y_true`` with a warning, because
    those two numbers are almost never the same. Evaluation sets are built
    near-balanced; real upload traffic is overwhelmingly authentic. Minimising
    cost at 50% prevalence and deploying the result into 1% prevalence produces
    a threshold far too permissive, and the error is invisible — every metric on
    the evaluation set looks right.

    The default costs encode the product's asymmetry: a false accusation is ten
    times the cost of a missed fake.
    """
    if cost_fp <= 0 or cost_fn <= 0:
        raise ValueError("costs must be positive")
    y, s = _prepare(y_true, y_score)

    warnings: list[str] = []
    if prevalence is None:
        prevalence = float(np.mean(y == 1))
        warnings.append(
            f"prevalence defaulted to the evaluation set's {prevalence:.3f}; "
            "production prevalence is usually far lower and changes the "
            "cost-optimal threshold. Pass it explicitly."
        )
    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"prevalence must lie strictly in (0, 1), got {prevalence}")

    best_t, best_cost = None, float("inf")
    for t in candidate_thresholds(s):
        fpr, tpr, _, _ = _point(y, s, float(t))
        cost = cost_fp * fpr * (1.0 - prevalence) + cost_fn * (1.0 - tpr) * prevalence
        if cost < best_cost:
            best_t, best_cost = float(t), cost
    return _finish(
        y,
        s,
        best_t,
        "at_min_expected_cost",
        {"cost_fp": cost_fp, "cost_fn": cost_fn, "prevalence": prevalence},
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        prevalence=prevalence,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------


def bootstrap_threshold_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_fpr: float = 0.02,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(threshold, lo, hi)`` — a percentile CI over resampled negatives.

    Resamples the **negatives only**, because the threshold is a quantile of the
    negative class and resampling the positives would add variance that does not
    affect it. A wide interval here means the holdout is too small to ship a
    threshold from, whatever the point estimate says.
    """
    y, s = _prepare(y_true, y_score)
    point = at_target_fpr(y, s, target_fpr=target_fpr).value

    negatives = s[y == 0]
    positives = s[y == 1]
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        sample_neg = rng.choice(negatives, size=negatives.size, replace=True)
        ys = np.r_[np.zeros(sample_neg.size), np.ones(positives.size)]
        ss = np.r_[sample_neg, positives]
        draws[i] = at_target_fpr(ys, ss, target_fpr=target_fpr).value
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(point), float(lo), float(hi)
