"""Evaluation metrics. Builds on ``inference.metrics`` rather than replacing it.

The serving package already carries ROC AUC with correct tie handling, the
confusion matrix, and the four thresholded metrics — they are re-exported here
so callers have one import and there is still one implementation. What this
module adds is the offline half: precision–recall area, and the counting
primitives that risk–coverage analysis is built from.

TIES ARE THE WHOLE PROBLEM
--------------------------
Every metric here treats a run of equal scores as one decision point rather than
as an arbitrary ordering. A trained detector saturates — thousands of items at
0.999 — and a metric that silently breaks those ties by array order reports a
flattering number that no other implementation reproduces. ``roc_auc`` handles
this with mean ranks; ``average_precision`` handles it by stepping over unique
thresholds, which is also why it is not computed as a trapezoid over a sampled
curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inference.metrics import (
    ConfusionMatrix,
    accuracy,
    confusion_matrix,
    equal_error_rate,
    f1,
    precision,
    recall,
    roc_auc,
)

__all__ = [
    "ConfusionMatrix",
    "PrecisionRecallCurve",
    "RocCurve",
    "accuracy",
    "average_precision",
    "confusion_matrix",
    "equal_error_rate",
    "f1",
    "precision",
    "precision_recall_curve",
    "pr_auc",
    "recall",
    "roc_auc",
    "roc_curve",
]


def _validate(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if y.size != s.size:
        raise ValueError(f"length mismatch: {y.size} vs {s.size}")
    if y.size == 0:
        raise ValueError("empty input")
    bad = set(np.unique(y)) - {0.0, 1.0}
    if bad:
        raise ValueError(f"y_true must be 0/1, found {sorted(bad)}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    return y, s


@dataclass(frozen=True)
class PrecisionRecallCurve:
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray

    def __len__(self) -> int:
        return int(self.thresholds.size)


@dataclass(frozen=True)
class RocCurve:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray

    def __len__(self) -> int:
        return int(self.thresholds.size)


def precision_recall_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> PrecisionRecallCurve:
    """Precision and recall at every distinct score, highest threshold first.

    One point per *unique* score, not per item: items sharing a score are either
    all predicted positive or all predicted negative, so they cannot be
    separated by any threshold and must not appear as separate operating points.
    """
    y, s = _validate(y_true, y_score)
    n_pos = float(np.sum(y == 1))
    if n_pos == 0:
        raise ValueError("precision_recall_curve needs at least one positive")

    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]

    # Cumulative counts at each position, then keep only the last index of each
    # tied run — the point at which that whole run has been included.
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, s_sorted.size - 1]

    tp, fp = tp[idx], fp[idx]
    prec = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    rec = tp / n_pos
    return PrecisionRecallCurve(precision=prec, recall=rec, thresholds=s_sorted[idx])


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision–recall curve, by the step (rectangle) rule.

    Step rather than trapezoid, deliberately. Interpolating linearly between two
    adjacent PR points assumes an operating point that does not exist — precision
    is not linear in recall between thresholds — and the trapezoid rule
    consequently reports an area slightly better than any achievable operating
    point. The step rule is the definition of average precision and is what
    every other implementation worth comparing against uses.
    """
    curve = precision_recall_curve(y_true, y_score)
    # Recall starts at 0 before any item is admitted.
    recall_deltas = np.diff(np.r_[0.0, curve.recall])
    return float(np.sum(recall_deltas * curve.precision))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Alias for :func:`average_precision`, under the name people search for."""
    return average_precision(y_true, y_score)


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> RocCurve:
    """FPR/TPR at every distinct score, highest threshold first."""
    y, s = _validate(y_true, y_score)
    n_pos, n_neg = float(np.sum(y == 1)), float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_curve needs both classes present")

    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, s_sorted.size - 1]
    return RocCurve(
        fpr=np.r_[0.0, fp[idx] / n_neg],
        tpr=np.r_[0.0, tp[idx] / n_pos],
        thresholds=np.r_[np.inf, s_sorted[idx]],
    )


def positive_rate(y_true: np.ndarray) -> float:
    """Prevalence. The baseline any PR number must be read against.

    Average precision for a random ranker equals the positive rate, so an AP of
    0.8 is excellent at 10% prevalence and poor at 90%. Reporting AP without it
    is reporting half a number.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    if y.size == 0:
        raise ValueError("empty input")
    return float(np.mean(y == 1))


def summary(
    y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5
) -> dict[str, float | int]:
    """Every metric at one operating point, with the threshold recorded."""
    from inference.metrics import summary as base_summary

    out = dict(base_summary(y_true, y_score, threshold=threshold))
    out["pr_auc"] = average_precision(y_true, y_score)
    out["positive_rate"] = positive_rate(y_true)
    return out
