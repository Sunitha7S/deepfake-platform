"""Evaluation metrics. Pure numpy, no sklearn.

Not a reimplementation for its own sake. Three reasons this is hand-written:

* sklearn is a heavy dependency for six functions, and this package's whole
  point is a short dependency list;
* ``roc_auc`` needs correct tie handling, and writing it makes the tie
  behaviour explicit rather than inherited;
* the thresholded metrics need to state which threshold they used, and an
  API that silently assumes 0.5 is how a detector gets evaluated at an
  operating point nobody chose.

Every function takes ``y_true`` as 0/1 labels and either hard predictions or
continuous scores, and every function that thresholds says so in its signature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate(y_true: np.ndarray, y_other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_other, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    if a.size == 0:
        raise ValueError("empty input")
    bad = set(np.unique(a)) - {0.0, 1.0}
    if bad:
        raise ValueError(f"y_true must be 0/1, found {sorted(bad)}")
    return a, b


@dataclass(frozen=True)
class ConfusionMatrix:
    """Counts at one operating point.

    Carries the threshold it was computed at. A confusion matrix without its
    threshold is a number without units — it cannot be compared with another
    one, and it cannot be reproduced.
    """

    tn: int
    fp: int
    fn: int
    tp: int
    threshold: float

    @property
    def total(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    @property
    def fpr(self) -> float:
        """False-positive rate — real content called fake.

        The costly error for this product: telling someone their genuine
        photograph is a forgery.
        """
        denom = self.fp + self.tn
        return float(self.fp / denom) if denom else 0.0

    @property
    def tpr(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom else 0.0

    def as_array(self) -> np.ndarray:
        """``[[tn, fp], [fn, tp]]`` — sklearn's layout, for comparability."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=np.int64)


def confusion_matrix(
    y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5
) -> ConfusionMatrix:
    """Counts at ``threshold``. Prediction is positive when ``score >= threshold``."""
    y, s = _validate(y_true, y_score)
    pred = (s >= threshold).astype(np.float64)
    return ConfusionMatrix(
        tn=int(np.sum((y == 0) & (pred == 0))),
        fp=int(np.sum((y == 0) & (pred == 1))),
        fn=int(np.sum((y == 1) & (pred == 0))),
        tp=int(np.sum((y == 1) & (pred == 1))),
        threshold=float(threshold),
    )


def accuracy(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5) -> float:
    """Fraction correct.

    Reported because it is asked for, and worth distrusting: on an imbalanced
    set a detector that answers "authentic" to everything scores well. Read it
    beside ``roc_auc`` and the confusion matrix, never alone.
    """
    cm = confusion_matrix(y_true, y_score, threshold=threshold)
    return float((cm.tp + cm.tn) / cm.total)


def precision(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5) -> float:
    """Of everything called fake, how much was. Zero predictions → 0.0."""
    cm = confusion_matrix(y_true, y_score, threshold=threshold)
    denom = cm.tp + cm.fp
    return float(cm.tp / denom) if denom else 0.0


def recall(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5) -> float:
    """Of everything that was fake, how much was caught. No positives → 0.0."""
    cm = confusion_matrix(y_true, y_score, threshold=threshold)
    denom = cm.tp + cm.fn
    return float(cm.tp / denom) if denom else 0.0


def f1(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(y_true, y_score, threshold=threshold)
    r = recall(y_true, y_score, threshold=threshold)
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve, by the rank (Mann–Whitney U) identity.

    Threshold-free, which is why it is the primary metric here: it measures
    ranking quality without committing to an operating point.

    Ties are given average ranks, which is the whole reason for computing it
    this way rather than by trapezoid over a sampled curve. A detector that
    saturates — and any well-trained one does — produces many identical scores,
    and treating those as arbitrary orderings inflates AUC by a small,
    flattering, entirely fictional amount.
    """
    y, s = _validate(y_true, y_score)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs both classes present")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    sorted_scores = s[order]

    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # Ranks are 1-based; a tied run shares the mean of the ranks it spans.
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1

    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """(EER, threshold) — the point where false positives and false negatives meet.

    A single summary of the trade-off, useful for comparing detectors. It is not
    the operating point to ship: this product weights false positives more
    heavily than EER's symmetry assumes.
    """
    y, s = _validate(y_true, y_score)
    thresholds = np.unique(s)
    best = (1.0, float(thresholds[0]))
    for t in thresholds:
        cm = confusion_matrix(y, s, threshold=float(t))
        gap = abs(cm.fpr - (1.0 - cm.tpr))
        if gap < best[0]:
            best = (gap, float(t))
    cm = confusion_matrix(y, s, threshold=best[1])
    return float((cm.fpr + (1.0 - cm.tpr)) / 2.0), best[1]


def summary(
    y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5
) -> dict[str, float | int]:
    """Every metric at once, with the threshold recorded alongside."""
    cm = confusion_matrix(y_true, y_score, threshold=threshold)
    eer, eer_threshold = equal_error_rate(y_true, y_score)
    return {
        "n": cm.total,
        "threshold": float(threshold),
        "accuracy": accuracy(y_true, y_score, threshold=threshold),
        "precision": precision(y_true, y_score, threshold=threshold),
        "recall": recall(y_true, y_score, threshold=threshold),
        "f1": f1(y_true, y_score, threshold=threshold),
        "roc_auc": roc_auc(y_true, y_score),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "fpr": cm.fpr,
        "tpr": cm.tpr,
        "tn": cm.tn,
        "fp": cm.fp,
        "fn": cm.fn,
        "tp": cm.tp,
    }
