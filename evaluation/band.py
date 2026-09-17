"""The operating-point derivation policy: threshold and band, chosen together.

WHAT THIS MODULE EXISTS TO PREVENT
----------------------------------
Until Batch 7 the two halves of the operating point were set in different
places. The threshold was *fitted* — ``at_target_fpr`` on the calibration split
— and the band width was *configured*, a constant carried in a dataclass
default. Nothing reconciled them, and nothing had to: they were separate
arguments to a constructor that happened to accept both.

Their product could be impossible. A 2% false-positive budget on 22 negatives
puts the threshold at the largest negative score — 0.9734 was the observed value
— and no symmetric 0.20-wide band fits around it. That was R11, and it surfaced
as a ``ValueError`` at calibration time with no route forward, because the
caller had set one of the two knobs and the fit had set the other.

Batch 7 removes the second knob from the caller's hands. :func:`derive_operating
_point` takes the calibration scores and the *constraints* — a false-positive
budget, a selective-error ceiling, a coverage floor, a determinism floor — and
produces the threshold and the margin from them, together, in one pass. There is
no argument to this function that sets the band directly.

WHY A REFUSAL IS A RESULT
-------------------------
The function can decline. If no margin satisfies the constraints, that is a
statement about the model rather than a problem to be routed around, and it
comes back as a :class:`OperatingPointDerivation` with ``chosen=None`` and a
census of why each candidate was rejected. Widening the band until something
passes is the failure mode this reports on; it is not a fallback it takes.

WHY THE MARGIN GRID IS GEOMETRIC
--------------------------------
The margin is a log-odds quantity, so equal *ratios* are the equal steps. A
linear grid from 0 to 3 spends most of its resolution on bands nobody would
ship and almost none near the interesting region below 1. The grid here is
geometric between the determinism floor and the widest representable margin,
which puts roughly the same number of candidates in every octave.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from contracts.common import ThresholdInfo, logit
from evaluation.risk import OperatingPoint, operating_point
from evaluation.thresholds import MIN_NEGATIVES_FOR_FPR, ThresholdChoice, at_target_fpr

DEFAULT_MARGIN_GRID_POINTS = 96
"""Candidates on the geometric margin grid. Enough that adjacent candidates
differ by well under a percent of a margin over the whole usable range, which is
finer than any score distribution can resolve."""


class InfeasibleOperatingPoint(ValueError):
    """No (threshold, margin) pair satisfies the constraints as stated.

    A distinct type because the caller's response differs from every other
    ``ValueError`` here: this one is a release decision — acquire more data,
    loosen a constraint deliberately, or ship nothing — and not a bug to fix at
    the call site.
    """


# --------------------------------------------------------------------------
# Admissibility of a fitted threshold
# --------------------------------------------------------------------------


def admissible_threshold(value: float) -> float:
    """The fitted threshold, checked for the ways a fit can leave the domain.

    ``at_target_fpr`` is a sweep over observed scores plus sentinels, and two of
    its outputs are not probabilities:

    * ``max(score) + 1`` when no threshold meets the budget — deliberately
      out of range so it cannot be mistaken for an operating point.
    * an observed score of exactly 0.0 or 1.0, from a saturated detector.

    Both used to reach ``ThresholdInfo`` and fail there with a message about
    bands. They fail here instead, with a message about the fit, because that is
    what is actually wrong.
    """
    if not math.isfinite(value):
        raise InfeasibleOperatingPoint(
            f"the fitted threshold is {value}; no operating point exists. This "
            "is the sentinel a threshold search returns when nothing meets its "
            "budget, not a score."
        )
    if value > 1.0 or value < 0.0:
        raise InfeasibleOperatingPoint(
            f"the fitted threshold is {value}, outside [0, 1]. A threshold "
            "search returns a value above the maximum score when no operating "
            "point meets its budget: nothing is detectable within it, and the "
            "budget is what has to move."
        )
    if value in (0.0, 1.0):
        raise InfeasibleOperatingPoint(
            f"the fitted threshold is exactly {value}: the detector is "
            "saturated, so the calibrated scores carry no odds at the boundary "
            "and no band can bracket it. Fit the threshold on logits, or fix "
            "the saturation."
        )
    return float(value)


def margin_for_probability_floor(threshold: float, floor: float) -> float:
    """Smallest margin whose band is at least ``floor`` wide on **both** sides.

    The determinism floor is stated in probability, because that is where the
    run-to-run jitter in ``p_fake`` is measured, while the band is parameterised
    in log-odds. This is the conversion, and it takes the larger of the two
    sides rather than an average: a floor met on one side only leaves the other
    side flippable by noise, which is the entire failure the floor exists to
    exclude.

        margin >= logit(t + f) - logit(t)      upper side
        margin >= logit(t) - logit(t - f)      lower side

    Raises when ``floor >= min(t, 1 - t)``. That limit is not an artefact of
    log-odds — a probability-space band has exactly the same ceiling, and it is
    the honest statement that a threshold 0.03 from an endpoint cannot support a
    0.05 guarantee on both sides of itself at any width.
    """
    t = admissible_threshold(threshold)
    if floor <= 0.0:
        return 0.0
    # Tested against the actual float sums rather than against ``min(t, 1-t)``:
    # at t = 0.99 and floor = 0.01 the analytic comparison passes (1 - t comes
    # back as 0.010000000000000009) and ``t + floor`` is then exactly 1.0, where
    # the logit does not exist. Found by tests/band_study.py.
    if t + floor >= 1.0 or t - floor <= 0.0:
        raise InfeasibleOperatingPoint(
            f"a determinism floor of {floor} cannot be met at a threshold of "
            f"{t}: the nearer endpoint is only {min(t, 1.0 - t):.6g} away, so no "
            "band of any width has that much room on both sides. Either the "
            "threshold has to move away from the endpoint — usually by "
            "acquiring negatives, since a tight budget on a small sample is "
            "what drives it there — or the jitter has to come down."
        )
    centre = logit(t)
    return max(logit(t + floor) - centre, centre - logit(t - floor))


def margin_grid(
    threshold: float,
    *,
    floor_margin: float = 0.0,
    points: int = DEFAULT_MARGIN_GRID_POINTS,
) -> np.ndarray:
    """Geometric grid of candidate margins, narrowest first.

    Bounded above by ``ThresholdInfo.max_margin(threshold)`` so every candidate
    is representable and the sweep never has to catch an exception to find its
    own limit.
    """
    t = admissible_threshold(threshold)
    hi = ThresholdInfo.max_margin(t)
    lo = max(floor_margin, 1e-4)
    if lo >= hi:
        return np.empty(0, dtype=np.float64)
    return np.exp(np.linspace(math.log(lo), math.log(hi), points))


# --------------------------------------------------------------------------
# The derivation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatingPointDerivation:
    """The chosen operating point, the constraints, and every rejection.

    Carries enough to reconstruct the decision without the scores: which
    threshold the budget produced, which margins were considered, why each
    unusable one was unusable, and what the survivor achieves.
    """

    threshold_choice: ThresholdChoice
    chosen: OperatingPoint | None
    margin: float | None
    considered: int
    floor_margin: float
    max_selective_fpr: float
    min_coverage: float
    determinism_floor: float
    rejected: dict[str, int]
    warnings: list[str]
    binding: bool = True
    """Whether any constraint actually selected the margin.

    False when the narrowest candidate on the grid already satisfied everything,
    which means the band is buying nothing beyond the determinism floor — the
    ceiling is looser than the bare threshold already achieves. That is a real
    and reportable answer ("this model does not need a band here"), and it is
    also the most likely symptom of a ceiling set carelessly, so it is surfaced
    rather than returned as a confident 0.0001.
    """

    @property
    def ok(self) -> bool:
        return self.chosen is not None

    @property
    def threshold(self) -> float:
        return self.threshold_choice.value

    def threshold_info(self, *, target_fpr: float) -> ThresholdInfo:
        """The publishable operating point, or a loud refusal.

        Built through :meth:`ThresholdInfo.at_margin` like every other path, so
        a derivation cannot mint a band the contract would not accept.
        """
        if self.margin is None:
            raise InfeasibleOperatingPoint(self.describe())
        return ThresholdInfo.at_margin(
            value=self.threshold, margin=self.margin, target_fpr=target_fpr
        )

    def describe(self) -> str:
        head = (
            f"threshold {self.threshold:.6f} "
            f"({self.threshold_choice.method}, achieved FPR "
            f"{self.threshold_choice.achieved_fpr:.4f}, TPR "
            f"{self.threshold_choice.achieved_tpr:.4f})"
        )
        if self.chosen is None:
            body = (
                f"  no margin satisfies the constraints (selective FPR <= "
                f"{self.max_selective_fpr}, coverage >= {self.min_coverage}, "
                f"jitter floor {self.determinism_floor} => margin >= "
                f"{self.floor_margin:.4f}). Considered {self.considered}; "
                f"rejected {self.rejected}. Widening further only trades away "
                "more coverage — this operating point is not available on this "
                "model."
            )
        else:
            note = "" if self.binding else "  [no constraint binds]"
            body = (
                f"  margin {self.margin:.4f} (odds x{math.exp(self.margin):.3f})"
                f"{note}  {self.chosen.describe()}"
            )
        lines = [head, body]
        lines += [f"  warning  {w}" for w in self.warnings]
        return "\n".join(lines)


def derive_operating_point(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    target_fpr: float,
    max_selective_fpr: float,
    min_coverage: float = 0.85,
    determinism_floor: float = 0.0,
    margins: np.ndarray | list[float] | None = None,
) -> OperatingPointDerivation:
    """Derive threshold **and** margin from one set of calibrated scores.

    The policy, in order:

    1. **Threshold** from the false-positive budget — ``at_target_fpr``, the
       lowest threshold whose FPR stays inside ``target_fpr``, so the budget is
       spent rather than merely respected.
    2. **Admissibility** — the fit must have produced a real probability
       strictly inside (0, 1). A sentinel or a saturated endpoint stops here,
       named as a fitting problem.
    3. **Floor** — convert ``determinism_floor`` (probability) into the smallest
       margin that clears it on both sides.
    4. **Sweep** — geometric grid from that floor to the widest representable
       margin.
    5. **Constrain** — keep candidates whose *selective* FPR is within
       ``max_selective_fpr`` and whose coverage is at least ``min_coverage``.
    6. **Choose the narrowest survivor.** Narrowest is most coverage: the band
       is a cost, and the rule is to pay the least of it that meets the
       constraints.
    7. **Refuse** if there is no survivor.

    WHY ``max_selective_fpr`` HAS NO DEFAULT
    ----------------------------------------
    It is the product decision the band exists to implement, and R8 was the
    consequence of nobody having made it. A default here would be a new
    unjustified constant standing exactly where the old one stood.

    It also has to be *tighter than the bare threshold already achieves*, or it
    buys nothing. Widening the band can only lower the false-positive rate,
    because the decision boundary is ``band_hi`` and it moves away from the
    negatives — so a ceiling at or above ``target_fpr`` is satisfied by the
    narrowest candidate on the grid and selects nothing. That case is not an
    error; it is reported as ``binding=False`` with a warning, because "this
    model does not need a band at this operating point" is a legitimate finding
    and an unremarked margin of 1e-4 is not.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    probs = np.asarray(p, dtype=np.float64).ravel()

    choice = at_target_fpr(y, probs, target_fpr=target_fpr)
    threshold = admissible_threshold(choice.value)

    warnings = list(choice.warnings)
    n_neg = int(np.sum(y == 0))
    nearest_edge = min(threshold, 1.0 - threshold)
    if nearest_edge < 0.05 and n_neg < MIN_NEGATIVES_FOR_FPR:
        # The R11 regime, named while it is still only a warning. The band no
        # longer fails here, but a threshold this close to an endpoint on this
        # few negatives is a quantile of nothing, and the operating point it
        # produces will move on the next resample.
        warnings.append(
            f"the threshold sits {nearest_edge:.4f} from an endpoint on only "
            f"{n_neg} negatives; a band now fits, but the threshold itself is "
            "an extreme quantile of a small sample and will move on resampling. "
            "Check bootstrap_threshold_ci before shipping this."
        )

    floor_margin = margin_for_probability_floor(threshold, determinism_floor)
    candidates = (
        np.asarray(margins, dtype=np.float64).ravel()
        if margins is not None
        else margin_grid(threshold, floor_margin=floor_margin)
    )

    rejected = {
        "below_determinism_floor": 0,
        "not_representable": 0,
        "selective_fpr_too_high": 0,
        "coverage_too_low": 0,
    }
    eligible: list[tuple[float, OperatingPoint]] = []
    considered = 0
    for raw in candidates:
        margin = float(raw)
        if margin < floor_margin:
            rejected["below_determinism_floor"] += 1
            continue
        try:
            info = ThresholdInfo.at_margin(
                value=threshold, margin=margin, target_fpr=target_fpr
            )
        except ValueError:
            # Only reachable for a caller-supplied grid; margin_grid is bounded.
            rejected["not_representable"] += 1
            continue
        considered += 1
        point = operating_point(y, probs, info)
        if point.selective_fpr > max_selective_fpr:
            rejected["selective_fpr_too_high"] += 1
            continue
        if point.coverage < min_coverage:
            rejected["coverage_too_low"] += 1
            continue
        eligible.append((margin, point))

    best = min(eligible, key=lambda pair: pair[0]) if eligible else None

    # Did anything actually choose this margin? If the narrowest candidate the
    # grid offered survived every constraint, the answer is no.
    binding = True
    if best is not None:
        narrowest_offered = min(
            (float(m) for m in candidates if float(m) >= floor_margin),
            default=None,
        )
        binding = narrowest_offered is None or best[0] > narrowest_offered * (1 + 1e-12)
        if not binding:
            warnings.append(
                f"no constraint binds: the narrowest margin on the grid "
                f"({best[0]:.6f}) already meets a selective-FPR ceiling of "
                f"{max_selective_fpr} and a coverage floor of {min_coverage}, so "
                "the band is buying nothing beyond the determinism floor. Either "
                "this model does not need a band at this operating point, or the "
                f"ceiling is looser than the {choice.achieved_fpr:.4f} the bare "
                "threshold already achieves — in which case tighten it to the "
                "rate the product actually requires."
            )

    return OperatingPointDerivation(
        threshold_choice=choice,
        chosen=best[1] if best else None,
        margin=best[0] if best else None,
        considered=considered,
        binding=binding,
        floor_margin=floor_margin,
        max_selective_fpr=max_selective_fpr,
        min_coverage=min_coverage,
        determinism_floor=determinism_floor,
        rejected=rejected,
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# The alternatives, kept honest
# --------------------------------------------------------------------------


def fixed_width_band(threshold: float, width: float) -> tuple[float, float]:
    """Approach A, the pre-Batch-7 geometry. **Partial** — this is the bug.

    Retained as a named implementation rather than deleted, so the failure it
    exhibits can be reproduced in a test against the real arithmetic instead of
    being asserted from memory. It is not called by anything that ships.

    Raises for every threshold within ``width/2`` of an endpoint, which is the
    whole of R11.
    """
    half = width / 2.0
    if width <= 0.0:
        raise ValueError(f"band width must be positive, got {width}")
    if threshold - half < 0.0 or threshold + half > 1.0:
        raise ValueError(
            f"a band of width {width} around {threshold} runs outside [0, 1]; "
            "narrow the band or move the threshold"
        )
    return threshold - half, threshold + half


def clipped_fixed_width_band(threshold: float, width: float) -> tuple[float, float]:
    """Approach B: A, with the edges clamped into [0, 1]. **Total, and worse.**

    Also not called by anything that ships. It exists because "just clamp it" is
    the first thing anyone proposes on hitting R11, and the reason it is wrong
    is quantitative rather than obvious: at a threshold of 0.9734 a 0.20-wide
    band clamps to [0.8734, 1.0], and since the manipulated region is
    ``p >= band_hi``, *no input is ever called manipulated again*. The detector
    silently becomes a two-state authentic/abstain classifier with a TPR of
    zero, and every published field still looks well-formed.

    ``tests/band_study.py`` measures that collapse rather than asserting it.
    """
    half = width / 2.0
    if width <= 0.0:
        raise ValueError(f"band width must be positive, got {width}")
    return max(0.0, threshold - half), min(1.0, threshold + half)


def quantile_band(
    p: np.ndarray, threshold: float, *, abstention_rate: float
) -> tuple[float, float]:
    """Approach C: edges at score quantiles bracketing the threshold.

    Places ``abstention_rate`` of the observed scores inside the band, split
    evenly by count either side of ``threshold``. Total by construction — a
    quantile of the data is always in range — and it controls coverage directly,
    which is its one genuine advantage.

    Its cost is that the band is defined by the *evaluation distribution* rather
    than by the model: the same model against a corpus with a different mix of
    easy and hard media gets a different band, so the coverage guarantee does
    not transfer to production, and the edges are not a function of the
    threshold at all. It is provided for the comparison in the design report and
    is not the shipped policy.
    """
    scores = np.asarray(p, dtype=np.float64).ravel()
    if not 0.0 < abstention_rate < 1.0:
        raise ValueError(f"abstention_rate must lie in (0, 1), got {abstention_rate}")
    below = scores[scores < threshold]
    above = scores[scores >= threshold]
    half = abstention_rate / 2.0
    lo = (
        float(np.quantile(below, max(0.0, 1.0 - half * scores.size / max(below.size, 1))))
        if below.size
        else max(0.0, threshold - 1e-6)
    )
    hi = (
        float(np.quantile(above, min(1.0, half * scores.size / max(above.size, 1))))
        if above.size
        else min(1.0, threshold + 1e-6)
    )
    return min(lo, threshold - 1e-9), max(hi, threshold + 1e-9)


def proportional_band(threshold: float, fraction: float) -> tuple[float, float]:
    """Approach E(ii): a fixed fraction of the widest band that would fit.

    ``width = fraction * 2 * min(t, 1 - t)``. Total, and — unlike the shipped
    design — symmetric in probability, so it would have preserved the Batch 1
    validator unchanged. It was the strongest competitor and it lost on meaning:
    ``fraction`` is "how far toward certainty", which is not a quantity anyone
    can reason about, and the family has a kink in its derivative at t = 0.5.

    Worth recording that it agrees with the shipped log-odds band to first order
    as t approaches an endpoint: a margin of d gives a width of
    ``2(1-t)sinh(d)`` there, against ``2(1-t)fraction`` here, so
    ``fraction = sinh(d)``. The two differ in how they *split* that width, not
    in how much of it there is.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must lie in (0, 1), got {fraction}")
    half = fraction * min(threshold, 1.0 - threshold)
    return threshold - half, threshold + half


def wilson_margin(threshold: float, n: int, *, z: float = 1.96) -> float:
    """Approach F: a margin sized by the sampling error of the threshold itself.

    The threshold is an extreme quantile of the negative scores, so it carries
    binomial sampling error. A Wald interval on that quantile has a half-width
    of roughly ``z * sqrt(t(1-t)/n)`` in probability; converting it through the
    same both-sides rule as the determinism floor gives a margin.

    Its guarantee is the appealing one — "abstain wherever the threshold's own
    uncertainty could have put the boundary" — and its failure mode is that it
    is a statement about the *calibration sample size* and nothing else. It
    shrinks as the holdout grows even if the model gets worse, and it is silent
    about selective risk, which is the thing the band is actually for. Reported
    in the design study as a floor worth comparing against, not as the policy.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    t = admissible_threshold(threshold)
    half = z * math.sqrt(t * (1.0 - t) / n)
    half = min(half, min(t, 1.0 - t) * (1.0 - 1e-9))
    if half <= 0.0:
        raise InfeasibleOperatingPoint("the sampling interval has no width")
    return margin_for_probability_floor(t, half)


__all__ = [
    "DEFAULT_MARGIN_GRID_POINTS",
    "InfeasibleOperatingPoint",
    "OperatingPointDerivation",
    "admissible_threshold",
    "clipped_fixed_width_band",
    "derive_operating_point",
    "fixed_width_band",
    "margin_for_probability_floor",
    "margin_grid",
    "proportional_band",
    "quantile_band",
    "wilson_margin",
]
