"""Calibration measurement: reliability, ECE, MCE, Brier, temperature evaluation.

``inference.calibration`` owns the things the serving path needs — sigmoid,
temperature scaling, the NLL fit. This module owns the things you need to decide
whether that fit was any good, which is offline work and does not belong in a
worker.

WHAT CALIBRATION IS, AND WHAT IT IS NOT
---------------------------------------
A model's raw output is a ranking. Cross-entropy training produces systematic
overconfidence, so 0.98 means "very high in the ranking", not "wrong one time in
fifty". Calibration fixes the scale without touching the order — which has two
consequences that this module exists to keep visible:

* **AUC cannot change.** Temperature scaling is monotone, so every ranking
  metric is identical before and after. A report claiming a calibration-driven
  AUC improvement has measured something else.
* **Only calibration makes a threshold mean anything.** "2% false positives" is
  a claim about probabilities, and it is empty unless the numbers being
  thresholded are probabilities.

THE ECE TRAP
------------
ECE is binned and therefore discontinuous. A model whose predictions concentrate
can sit inside one bin before calibration and straddle two after, moving ECE
sharply with no real change in honesty — this project has already been bitten by
exactly that, watching ECE jump 0.02 to 0.40 while NLL improved.

Two defences, both provided here. ``strategy="quantile"`` puts equal *counts* in
each bin rather than equal widths, which is far more stable for concentrated
predictions. And every report carries NLL and Brier alongside ECE: both are
bin-free proper scoring rules, and **when they disagree with ECE, they are the
ones to believe.**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from inference.calibration import (
    fit_temperature,
    negative_log_likelihood,
    sigmoid,
    temperature_scale,
)

BinStrategy = Literal["uniform", "quantile"]


def _validate(y_true: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    probs = np.asarray(p, dtype=np.float64).ravel()
    if y.size != probs.size:
        raise ValueError(f"length mismatch: {y.size} vs {probs.size}")
    if y.size == 0:
        raise ValueError("empty input")
    bad = set(np.unique(y)) - {0.0, 1.0}
    if bad:
        raise ValueError(f"y_true must be 0/1, found {sorted(bad)}")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    return y, probs


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilityBin:
    """One row of a reliability diagram."""

    lo: float
    hi: float
    count: int
    mean_predicted: float
    empirical_rate: float

    @property
    def gap(self) -> float:
        """Signed. Positive means the model claimed more than it delivered."""
        return self.mean_predicted - self.empirical_rate


@dataclass(frozen=True)
class ReliabilityDiagram:
    bins: list[ReliabilityBin]
    strategy: BinStrategy
    n: int

    @property
    def populated(self) -> list[ReliabilityBin]:
        return [b for b in self.bins if b.count > 0]

    def ece(self) -> float:
        """Count-weighted mean absolute gap."""
        if self.n == 0:
            return 0.0
        return float(
            sum(b.count * abs(b.gap) for b in self.populated) / self.n
        )

    def mce(self) -> float:
        """Worst bin. Zero when nothing is populated."""
        gaps = [abs(b.gap) for b in self.populated]
        return float(max(gaps)) if gaps else 0.0

    def to_table(self) -> str:
        header = f"{'range':>14} {'n':>7} {'predicted':>10} {'actual':>8} {'gap':>8}"
        rows = [header, "-" * len(header)]
        for b in self.populated:
            rows.append(
                f"[{b.lo:.3f},{b.hi:.3f}) {b.count:>7} {b.mean_predicted:>10.4f} "
                f"{b.empirical_rate:>8.4f} {b.gap:>+8.4f}"
            )
        rows.append(f"\nECE {self.ece():.5f}   MCE {self.mce():.5f}   n {self.n}")
        return "\n".join(rows)


def reliability_diagram(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> ReliabilityDiagram:
    """Bin predictions and compare claimed probability with observed frequency.

    ``uniform`` bins are equal-width and are what most papers report.
    ``quantile`` bins hold equal counts and are the honest choice when
    predictions concentrate: with equal-width bins a saturated model puts 95% of
    its mass in one bin, and ECE then reports almost entirely on that bin while
    the empty ones contribute nothing.
    """
    y, probs = _validate(y_true, p)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:  # every prediction identical
            edges = np.array([probs[0], probs[0] + 1e-12])
    else:
        raise ValueError(f"unknown binning strategy {strategy!r}")

    bins: list[ReliabilityBin] = []
    for i in range(edges.size - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # Half-open, except the last bin which must contain p == 1.0. Without
        # that, a perfectly confident correct prediction falls outside every bin
        # and silently stops contributing to ECE.
        last = i == edges.size - 2
        mask = (probs >= lo) & (probs <= hi) if last else (probs >= lo) & (probs < hi)
        count = int(np.sum(mask))
        bins.append(
            ReliabilityBin(
                lo=lo,
                hi=hi,
                count=count,
                mean_predicted=float(np.mean(probs[mask])) if count else 0.0,
                empirical_rate=float(np.mean(y[mask])) if count else 0.0,
            )
        )
    return ReliabilityDiagram(bins=bins, strategy=strategy, n=int(y.size))


def expected_calibration_error(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> float:
    return reliability_diagram(y_true, p, n_bins=n_bins, strategy=strategy).ece()


def maximum_calibration_error(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> float:
    return reliability_diagram(y_true, p, n_bins=n_bins, strategy=strategy).mce()


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probabilities. Lower is better; 0 is perfect.

    Bin-free and a proper scoring rule, so it cannot be gamed by a model that
    hedges. It rewards calibration and discrimination together, which is why it
    is reported next to ECE rather than instead of it.
    """
    y, probs = _validate(y_true, p)
    return float(np.mean((probs - y) ** 2))


# --------------------------------------------------------------------------
# Temperature evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationReport:
    """Before and after, on every measure that matters."""

    temperature: float
    n: int
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    mce_before: float
    mce_after: float
    brier_before: float
    brier_after: float
    auc_before: float
    auc_after: float

    @property
    def nll_improved(self) -> bool:
        return self.nll_after < self.nll_before

    @property
    def ece_improved(self) -> bool:
        return self.ece_after < self.ece_before

    @property
    def ranking_preserved(self) -> bool:
        """Temperature scaling is monotone, so AUC must be unchanged.

        A tolerance of 1e-9 covers the floating-point path through
        ``exp``/``log``; anything larger means the transform was not monotone
        and something other than calibration happened.
        """
        return abs(self.auc_after - self.auc_before) < 1e-9

    def describe(self) -> str:
        return (
            f"T={self.temperature:.4f} on n={self.n}\n"
            f"  NLL   {self.nll_before:.5f} -> {self.nll_after:.5f}\n"
            f"  ECE   {self.ece_before:.5f} -> {self.ece_after:.5f}\n"
            f"  MCE   {self.mce_before:.5f} -> {self.mce_after:.5f}\n"
            f"  Brier {self.brier_before:.5f} -> {self.brier_after:.5f}\n"
            f"  AUC   {self.auc_before:.5f} -> {self.auc_after:.5f}"
            f"   {'(preserved)' if self.ranking_preserved else '(CHANGED - not a monotone transform)'}"
        )


def evaluate_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> CalibrationReport:
    """Measure what applying ``temperature`` does, on every axis at once.

    Reports AUC on both sides deliberately. It must not move, and checking that
    is the cheapest available guard against a "calibration" step that quietly
    reorders predictions.
    """
    from evaluation.metrics import roc_auc

    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    before = np.asarray(sigmoid(z), dtype=np.float64)
    after = np.asarray(temperature_scale(z, temperature), dtype=np.float64)

    diag_before = reliability_diagram(y, before, n_bins=n_bins, strategy=strategy)
    diag_after = reliability_diagram(y, after, n_bins=n_bins, strategy=strategy)

    return CalibrationReport(
        temperature=float(temperature),
        n=int(y.size),
        nll_before=negative_log_likelihood(z, y, 1.0),
        nll_after=negative_log_likelihood(z, y, temperature),
        ece_before=diag_before.ece(),
        ece_after=diag_after.ece(),
        mce_before=diag_before.mce(),
        mce_after=diag_after.mce(),
        brier_before=brier_score(y, before),
        brier_after=brier_score(y, after),
        auc_before=roc_auc(y, before),
        auc_after=roc_auc(y, after),
    )


def fit_and_report(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> CalibrationReport:
    """Fit a temperature on these logits and report what it did."""
    t = fit_temperature(np.asarray(logits), np.asarray(labels))
    return evaluate_temperature(logits, labels, t, n_bins=n_bins, strategy=strategy)


def holdout_temperature_check(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    fraction: float = 0.8,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Fit on part, measure on the rest. Returns ``(T, nll_before, nll_after)``.

    The acceptance gate that the existing ``fit_and_validate`` does not have:
    its three refusals — degenerate logits, worsened NLL, too few samples — are
    all computed on the same data the temperature was fitted to, so a
    temperature fitted to sampling noise passes all three. This one cannot,
    because the number it reports comes from data the fit never saw.
    """
    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if z.size != y.size:
        raise ValueError("logits and labels must be the same length")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    order = rng.permutation(z.size)
    cut = int(z.size * fraction)
    fit_idx, held_idx = order[:cut], order[cut:]
    if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[held_idx])) < 2:
        raise ValueError("both folds need both classes present")

    t = fit_temperature(z[fit_idx], y[fit_idx])
    return (
        float(t),
        negative_log_likelihood(z[held_idx], y[held_idx], 1.0),
        negative_log_likelihood(z[held_idx], y[held_idx], t),
    )
