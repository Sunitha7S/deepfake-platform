"""Risk–coverage analysis: what the abstention band actually costs and buys.

A three-state verdict is a *selective* classifier. It answers on some inputs and
declines on others, so it cannot be summarised by accuracy alone — a detector
that abstains on everything has no errors and no value. The two numbers that
describe it together are:

    coverage         the fraction of inputs that receive a decision
    selective risk   the error rate *among the inputs that were decided*

Sweeping the band traces one against the other. That curve is how the band
should be chosen, and choosing it any other way — a round number, a guess, "0.20
looks reasonable" — is picking an abstention rate without knowing what it buys.

THE SWEEP IS OVER MARGINS, NOT WIDTHS
-------------------------------------
Batch 7 moved the band's parameter from a probability width to a log-odds
margin (see :class:`~contracts.common.ThresholdInfo`). The consequence here is
that the sweep no longer has to skip candidates: every margin is representable
at every admissible threshold, so a curve has the same shape near 0.97 as it
does near 0.5, and ``risk_at_coverage`` cannot silently be answering over a
truncated domain because the wide end ran off [0, 1].

``band_width`` survives on :class:`OperatingPoint` as a reported quantity — the
probability width the chosen margin produced — because a risk-coverage report is
read in probability. It is an output of the sweep, never its input.

WHY THE BAND'S PRICE IS PAID IN DETECTIONS
------------------------------------------
The band straddles the calibrated threshold, so widening it withdraws verdicts
from *both* sides: scores just below ``value`` stop being called authentic and
scores just above it stop being called manipulated. Because a detector's scores
concentrate near the boundary, a small widening removes a disproportionate
number of decisions. The curve makes that visible; intuition does not.

THE SECOND JOB OF THE BAND
--------------------------
Real inference is not bit-deterministic across device classes, so ``p_fake``
moves slightly between runs. A band at least an order of magnitude wider than
that jitter makes it impossible for numerical noise to flip a published verdict:
a score that could wobble across a boundary is inside the band, and therefore
was never going to be published as a decision. ``select_margin`` takes that
floor as an argument so the guarantee is chosen rather than hoped for — and
because the floor is stated in probability while the band is parameterised in
log-odds, the conversion lives in one place,
``evaluation.band.margin_for_probability_floor``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.common import ThresholdInfo


def _prepare(y_true: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


@dataclass(frozen=True)
class OperatingPoint:
    """One band setting, fully described.

    Every rate here is *selective* — computed over the decided subset only.
    Mixing abstentions into a false-positive rate would let a model improve its
    apparent FPR by declining to answer, which is the one thing a selective
    metric must not permit.
    """

    threshold: float
    margin: float
    """The band's parameter: half-width in log-odds. What the sweep varies."""
    band_width: float
    """``band_hi - band_lo``. Derived from ``margin``; reported, never set."""
    band_lo: float
    band_hi: float
    n: int
    n_decided: int
    n_abstained: int
    tp: int
    fp: int
    tn: int
    fn: int
    abstained_real: int
    abstained_fake: int

    @property
    def coverage(self) -> float:
        return self.n_decided / self.n if self.n else 0.0

    @property
    def abstention_rate(self) -> float:
        return 1.0 - self.coverage

    @property
    def selective_risk(self) -> float:
        """Error rate among the decided. Zero coverage has no defined risk."""
        return (self.fp + self.fn) / self.n_decided if self.n_decided else 0.0

    @property
    def selective_fpr(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    @property
    def selective_tpr(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def selective_fnr(self) -> float:
        return 1.0 - self.selective_tpr

    def describe(self) -> str:
        return (
            f"margin {self.margin:.4f} (width {self.band_width:.3f})  "
            f"[{self.band_lo:.3f}, {self.band_hi:.3f})  "
            f"coverage {self.coverage:.3f}  risk {self.selective_risk:.4f}  "
            f"FPR {self.selective_fpr:.4f}  FNR {self.selective_fnr:.4f}  "
            f"abstained {self.n_abstained} ({self.abstained_real} real / "
            f"{self.abstained_fake} fake)"
        )


def operating_point(
    y_true: np.ndarray, p: np.ndarray, threshold: ThresholdInfo
) -> OperatingPoint:
    """Evaluate one published :class:`ThresholdInfo` against labelled scores.

    Takes the contract type rather than loose floats so that what is measured
    here is exactly what the pipeline will decide with — including the guarantee
    that the band is centred, which ``ThresholdInfo`` enforces at construction.
    """
    y, probs = _prepare(y_true, p)
    lo, hi = threshold.band_lo, threshold.band_hi

    abstain = (probs >= lo) & (probs < hi)
    manipulated = probs >= hi
    authentic = probs < lo

    return OperatingPoint(
        threshold=threshold.value,
        margin=threshold.margin,
        band_width=threshold.band_width,
        band_lo=lo,
        band_hi=hi,
        n=int(y.size),
        n_decided=int(np.sum(~abstain)),
        n_abstained=int(np.sum(abstain)),
        tp=int(np.sum(manipulated & (y == 1))),
        fp=int(np.sum(manipulated & (y == 0))),
        tn=int(np.sum(authentic & (y == 0))),
        fn=int(np.sum(authentic & (y == 1))),
        abstained_real=int(np.sum(abstain & (y == 0))),
        abstained_fake=int(np.sum(abstain & (y == 1))),
    )


def risk_coverage_curve(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    threshold: float,
    margins: np.ndarray | list[float] | None = None,
    target_fpr: float = 0.02,
) -> list[OperatingPoint]:
    """Sweep the margin, widest coverage first.

    Every margin on the default grid is representable at ``threshold``, so
    nothing is skipped and the curve spans the whole usable range. That is the
    Batch 7 difference: the old width sweep silently dropped its wide end
    whenever the threshold sat near an endpoint, so a curve fitted at 0.97
    covered a fraction of the domain a curve fitted at 0.5 did, without saying
    so anywhere.

    A caller-supplied ``margins`` array is still filtered — an unrepresentable
    candidate is skipped rather than raised on — because a hand-built sweep is
    exploratory by nature and should not have to know the limit in advance.
    """
    y, probs = _prepare(y_true, p)
    if margins is None:
        from evaluation.band import margin_grid

        margins = margin_grid(threshold)

    points: list[OperatingPoint] = []
    for m in np.asarray(margins, dtype=np.float64).ravel():
        try:
            info = ThresholdInfo.at_margin(
                value=threshold, margin=float(m), target_fpr=target_fpr
            )
        except ValueError:
            continue
        points.append(operating_point(y, probs, info))
    return points


def risk_at_coverage(
    curve: list[OperatingPoint], target_coverage: float
) -> OperatingPoint:
    """The tightest band that still decides at least ``target_coverage``.

    "At least": coverage is a floor the caller is willing to pay for, so among
    the points that meet it the one with the lowest risk is returned.
    """
    eligible = [pt for pt in curve if pt.coverage >= target_coverage]
    if not eligible:
        raise ValueError(
            f"no band achieves coverage >= {target_coverage}; the widest point "
            f"on this curve covers {max(pt.coverage for pt in curve):.3f}"
        )
    return min(eligible, key=lambda pt: (pt.selective_risk, -pt.coverage))


def coverage_at_risk(curve: list[OperatingPoint], target_risk: float) -> OperatingPoint:
    """The widest coverage whose selective risk stays within ``target_risk``."""
    eligible = [pt for pt in curve if pt.selective_risk <= target_risk]
    if not eligible:
        raise ValueError(
            f"no band achieves selective risk <= {target_risk}; the best point "
            f"on this curve reaches {min(pt.selective_risk for pt in curve):.4f}"
        )
    return max(eligible, key=lambda pt: (pt.coverage, -pt.selective_risk))


@dataclass(frozen=True)
class BandSelection:
    """A chosen margin, and the constraints it had to satisfy."""

    chosen: OperatingPoint | None
    considered: int
    max_selective_fpr: float
    min_coverage: float
    determinism_floor: float
    rejected: dict[str, int]

    @property
    def margin(self) -> float | None:
        return self.chosen.margin if self.chosen is not None else None

    @property
    def ok(self) -> bool:
        return self.chosen is not None

    def require(self) -> OperatingPoint:
        """The chosen point, or a loud failure explaining why there isn't one.

        ``chosen`` is optional because "no band works" is a real and important
        answer, but an optional field invites a caller to reach straight through
        it and get ``AttributeError: 'NoneType'`` — which reports a Python
        problem instead of the release decision the analysis actually made. This
        is the accessor for callers that need a point and want the diagnosis
        when there isn't one.
        """
        if self.chosen is None:
            raise ValueError(self.describe())
        return self.chosen

    def describe(self) -> str:
        if self.chosen is None:
            return (
                "no band satisfies the constraints "
                f"(selective FPR <= {self.max_selective_fpr}, coverage >= "
                f"{self.min_coverage}, both band sides >= "
                f"{self.determinism_floor}). "
                f"Rejected: {self.rejected}. The model is not ready at this "
                "operating point — widening the band further only trades away "
                "more coverage."
            )
        return f"selected {self.chosen.describe()}"


def select_margin(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    threshold: float,
    max_selective_fpr: float = 0.01,
    min_coverage: float = 0.85,
    determinism_floor: float = 0.0,
    target_fpr: float = 0.02,
    margins: np.ndarray | list[float] | None = None,
) -> BandSelection:
    """Choose the narrowest margin meeting every constraint. Narrowest = most coverage.

    Three constraints, and a refusal rather than a compromise when they cannot
    all be met:

    * ``max_selective_fpr`` — the product's asymmetry, applied to the decided
      subset.
    * ``min_coverage`` — an abstention rate the product can live with.
    * ``determinism_floor`` — the numerical guarantee, in **probability**. Pass
      ten times the measured run-to-run variation in ``p_fake`` for the target
      device class, and a verdict can never be flipped by floating-point noise.

    The floor is applied to ``min(value - band_lo, band_hi - value)`` and not to
    ``band_width``. Those were the same test when the band was symmetric in
    probability; they are not now, and the weaker of the two sides is what the
    guarantee has to hold on. Checking the total width instead would pass a band
    whose narrow side is half the jitter it is supposed to swallow.

    Returning ``None`` is a real answer. If no band satisfies the constraints,
    the model does not meet them, and the fix is a better model rather than a
    looser rule.

    This chooses a band for a threshold already in hand. To choose **both** —
    which is what a calibration run does, and where the two used to be set
    independently — use ``evaluation.band.derive_operating_point``.
    """
    curve = risk_coverage_curve(
        y_true, p, threshold=threshold, margins=margins, target_fpr=target_fpr
    )
    rejected = {"side_below_floor": 0, "fpr_too_high": 0, "coverage_too_low": 0}
    eligible: list[OperatingPoint] = []
    for pt in curve:
        narrow_side = min(pt.threshold - pt.band_lo, pt.band_hi - pt.threshold)
        if narrow_side < determinism_floor:
            rejected["side_below_floor"] += 1
            continue
        if pt.selective_fpr > max_selective_fpr:
            rejected["fpr_too_high"] += 1
            continue
        if pt.coverage < min_coverage:
            rejected["coverage_too_low"] += 1
            continue
        eligible.append(pt)

    chosen = min(eligible, key=lambda pt: pt.margin) if eligible else None
    return BandSelection(
        chosen=chosen,
        considered=len(curve),
        max_selective_fpr=max_selective_fpr,
        min_coverage=min_coverage,
        determinism_floor=determinism_floor,
        rejected=rejected,
    )


def abstention_breakdown(point: OperatingPoint) -> dict[str, float]:
    """Who is being declined. An abstention rate alone hides the asymmetry.

    A band that abstains almost entirely on fakes is protecting the false
    positive rate by declining the hard positives, which reads as caution and
    behaves as a detection failure.
    """
    total = point.n_abstained
    return {
        "abstention_rate": point.abstention_rate,
        "abstained_real": point.abstained_real,
        "abstained_fake": point.abstained_fake,
        "share_real": point.abstained_real / total if total else 0.0,
        "share_fake": point.abstained_fake / total if total else 0.0,
    }
