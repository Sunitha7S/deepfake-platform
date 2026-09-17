"""The six band designs, measured side by side. Standalone, like ``import_audit``.

    python tests/band_study.py

Produces the tables in ``docs/THRESHOLD_BAND_DESIGN.md``. Kept in the repository
rather than in a notebook because every number in that document is an assertion
about this code, and an assertion nobody can re-run is a claim.

WHAT IS BEING COMPARED
----------------------
Six ways of turning a fitted threshold into an abstention band:

    A  fixed width, symmetric in probability          (what shipped)
    B  fixed width, clipped into [0, 1]               (the obvious "fix")
    C  quantile band at a target abstention rate
    D  risk-coverage derived                          (the selection procedure)
    E  log-odds margin, symmetric in evidence         (shipped after Batch 7)
    E' proportional: a fixed fraction of the widest fitting band
    F  confidence-interval derived from the threshold's own sampling error

D is not a geometry — it is the rule for *choosing* a parameter, and it needs
one of the others to parameterise. It is run here over E, which is what the
shipped policy does.

THE FIXTURE
-----------
Three regimes, all with calibrated-looking scores:

    central     a threshold near 0.5      — where every design agrees
    shifted     a threshold near 0.85     — where A starts to bind
    saturated   a threshold near 0.97     — the R11 regime, where A does not exist

The saturated regime is the one that matters. It is not exotic: a tight
false-positive budget on a small calibration split produces it by construction.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.common import (  # noqa: E402
    DEFAULT_LOG_ODDS_MARGIN,
    ThresholdInfo,
    expit,
    logit,
)
from evaluation.band import (  # noqa: E402
    InfeasibleOperatingPoint,
    clipped_fixed_width_band,
    derive_operating_point,
    fixed_width_band,
    margin_for_probability_floor,
    proportional_band,
    quantile_band,
    wilson_margin,
)
from evaluation.thresholds import at_target_fpr  # noqa: E402

SHIPPED_WIDTH = 0.20
"""The band width this system carried from Batch 1 to Batch 6."""


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def regime(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """``(labels, calibrated probabilities, target_fpr)`` for one regime."""
    rng = np.random.default_rng(4)
    if name == "central":
        n, sigma, sep, target = 4000, 1.6, 2.0, 0.20
    elif name == "shifted":
        n, sigma, sep, target = 4000, 1.6, 2.0, 0.05
    elif name == "saturated":
        n, sigma, sep, target = 4000, 2.6, 2.0, 0.01
    else:  # pragma: no cover - programmer error
        raise ValueError(name)
    y = (rng.random(n) < 0.5).astype(np.float64)
    z = rng.normal(0.0, sigma, n) + sep * (2.0 * y - 1.0)
    return y, np.asarray([expit(v) for v in z], dtype=np.float64), target


def small_calibration_split(n_neg: int = 22, seed: int = 3):
    """The R11 fixture as recorded in the audit: a 2% budget over 22 negatives."""
    rng = np.random.default_rng(seed)
    neg = np.asarray([expit(v) for v in rng.normal(-1.0, 1.8, n_neg)])
    pos = np.asarray([expit(v) for v in rng.normal(3.5, 1.2, 60)])
    return np.r_[np.zeros(n_neg), np.ones(pos.size)], np.r_[neg, pos]


# --------------------------------------------------------------------------
# Scoring a band, whatever produced it
# --------------------------------------------------------------------------


def evaluate(y: np.ndarray, p: np.ndarray, lo: float, hi: float) -> dict:
    """Selective rates for an arbitrary (lo, hi) pair, contract or not.

    Deliberately does not go through ``operating_point``, which takes a
    ``ThresholdInfo`` and therefore refuses four of the six designs. The point
    of the study is to score the designs the contract rejects, so the decision
    rule is reimplemented here — once, in eight lines, matching ``decide``.
    """
    abstain = (p >= lo) & (p < hi)
    manipulated = p >= hi
    authentic = p < lo
    n = int(y.size)
    n_dec = int(np.sum(~abstain))
    tp = int(np.sum(manipulated & (y == 1)))
    fp = int(np.sum(manipulated & (y == 0)))
    tn = int(np.sum(authentic & (y == 0)))
    fn = int(np.sum(authentic & (y == 1)))
    return {
        "lo": lo,
        "hi": hi,
        "width": hi - lo,
        "coverage": n_dec / n if n else 0.0,
        "risk": (fp + fn) / n_dec if n_dec else float("nan"),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "tpr": tp / (tp + fn) if (tp + fn) else float("nan"),
        "n_manipulated": tp + fp,
    }


def _row(label: str, result: dict | str) -> str:
    if isinstance(result, str):
        return f"  {label:<34s} {result}"
    return (
        f"  {label:<34s} [{result['lo']:.4f}, {result['hi']:.4f})  "
        f"w {result['width']:.4f}  cov {result['coverage']:.3f}  "
        f"risk {result['risk']:.4f}  FPR {result['fpr']:.4f}  "
        f"TPR {result['tpr']:.4f}  decided-fake {result['n_manipulated']:>4d}"
    )


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------


def compare(y: np.ndarray, p: np.ndarray, target_fpr: float, *, label: str) -> None:
    choice = at_target_fpr(y, p, target_fpr=target_fpr)
    t = choice.value
    print(f"\n=== {label} ===")
    print(
        f"  threshold {t:.6f}  (budget {target_fpr:.0%}, achieved FPR "
        f"{choice.achieved_fpr:.4f}, bare TPR {choice.achieved_tpr:.4f}, "
        f"{choice.n_negative} negatives)"
    )
    print(f"  logit(threshold) {logit(t):+.4f}   odds {t / (1 - t):.3f}")
    print()

    # A — fixed width, symmetric in probability.
    try:
        print(_row("A  fixed width 0.20", evaluate(y, p, *fixed_width_band(t, SHIPPED_WIDTH))))
    except ValueError as exc:
        print(_row("A  fixed width 0.20", f"DOES NOT EXIST - {exc}"))

    # B — the same, clipped.
    lo, hi = clipped_fixed_width_band(t, SHIPPED_WIDTH)
    res = evaluate(y, p, lo, hi)
    note = "  <- manipulated region empty" if res["n_manipulated"] == 0 else ""
    print(_row("B  fixed width 0.20, clipped", res) + note)

    # C — quantile band at a fixed abstention rate.
    print(_row("C  quantile band, 20% abstain", evaluate(y, p, *quantile_band(p, t, abstention_rate=0.20))))

    # E' — proportional.
    print(_row("E' proportional, fraction 0.40", evaluate(y, p, *proportional_band(t, 0.40))))

    # F — confidence-interval derived.
    try:
        m = wilson_margin(t, int(np.sum(y == 0)))
        info = ThresholdInfo.at_margin(value=t, margin=m, target_fpr=target_fpr)
        print(_row(f"F  Wilson margin {m:.4f}", evaluate(y, p, info.band_lo, info.band_hi)))
    except (ValueError, InfeasibleOperatingPoint) as exc:
        print(_row("F  Wilson margin", f"DOES NOT EXIST - {exc}"))

    # E — the shipped geometry at the shipped placeholder margin.
    info = ThresholdInfo.at_margin(
        value=t, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=target_fpr
    )
    print(_row(f"E  log-odds margin {DEFAULT_LOG_ODDS_MARGIN:.4f}", evaluate(y, p, info.band_lo, info.band_hi)))

    # D over E — the shipped policy: derive the margin from the scores.
    for ceiling in (target_fpr / 2.0, target_fpr / 5.0):
        derived = derive_operating_point(
            y, p, target_fpr=target_fpr, max_selective_fpr=ceiling, min_coverage=0.40
        )
        if not derived.ok:
            print(_row(f"D  derived, sFPR<={ceiling:.4f}", "REFUSED - constraints unmet"))
            continue
        info = derived.threshold_info(target_fpr=target_fpr)
        bind = "" if derived.binding else "  <- no constraint binds"
        print(
            _row(
                f"D  derived, sFPR<={ceiling:.4f} (m={derived.margin:.4g})",
                evaluate(y, p, info.band_lo, info.band_hi),
            )
            + bind
        )


def totality_table() -> None:
    """For each design, the set of thresholds at which it produces a band."""
    print("\n=== feasible threshold range, shipped parameter ===")
    grid = [i / 100000.0 for i in range(1, 100000)]

    def frac(ok) -> float:
        return sum(1 for v in grid if ok(v)) / len(grid)

    def a_ok(v: float) -> bool:
        try:
            fixed_width_band(v, SHIPPED_WIDTH)
        except ValueError:
            return False
        return True

    def e_ok(v: float) -> bool:
        try:
            ThresholdInfo.at_margin(
                value=v, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
            )
        except ValueError:
            return False
        return True

    def b_ok(v: float) -> bool:
        lo, hi = clipped_fixed_width_band(v, SHIPPED_WIDTH)
        return lo < v < hi

    def b_useful(v: float) -> bool:
        lo, hi = clipped_fixed_width_band(v, SHIPPED_WIDTH)
        return lo < v < hi and hi < 1.0 and lo > 0.0

    print(f"  A  fixed width 0.20                constructible on {frac(a_ok):.5f} of (0, 1)")
    print(f"  B  clipped                         constructible on {frac(b_ok):.5f} of (0, 1)")
    print(f"  B  clipped, both regions reachable                  {frac(b_useful):.5f} of (0, 1)")
    print(f"  E  log-odds margin {DEFAULT_LOG_ODDS_MARGIN:.4f}         constructible on {frac(e_ok):.5f} of (0, 1)")
    print()
    print("  E's exact limit, from max_margin (no grid search):")
    for v in (1.0e-6, 1.5e-6, 0.5, 0.9999985, 0.9999990):
        try:
            lim = ThresholdInfo.max_margin(v)
        except ValueError:
            lim = float("nan")
        ok = lim > DEFAULT_LOG_ODDS_MARGIN
        print(f"    threshold {v:<12.7g} max margin {lim:8.4f}   shipped margin fits: {ok}")


def evidence_table() -> None:
    """What a fixed probability width is worth, in evidence, across the range."""
    print("\n=== a fixed 0.20 width, measured in log-odds ===")
    print("  threshold   band                    evidence below   evidence above")
    for v in (0.15, 0.30, 0.50, 0.70, 0.85, 0.89):
        lo, hi = fixed_width_band(v, SHIPPED_WIDTH)
        below = logit(v) - logit(lo) if lo > 0 else float("inf")
        above = logit(hi) - logit(v) if hi < 1 else float("inf")
        print(f"  {v:<11.2f} [{lo:.4f}, {hi:.4f})        {below:>8.4f}         {above:>8.4f}")
    print("  0.90        [0.8000, 1.0000)          0.8109              inf")
    print("\n=== a fixed log-odds margin of ln(1.5), measured in probability ===")
    print("  threshold   band                    width below      width above")
    for v in (0.15, 0.30, 0.50, 0.70, 0.85, 0.90, 0.99):
        info = ThresholdInfo.at_margin(
            value=v, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
        )
        print(
            f"  {v:<11.2f} [{info.band_lo:.4f}, {info.band_hi:.4f})        "
            f"{v - info.band_lo:>8.4f}         {info.band_hi - v:>8.4f}"
        )


def determinism_table() -> None:
    """The floor conversion: what margin a given jitter in p_fake demands."""
    print("\n=== margin required to swallow a given jitter in p_fake ===")
    print("  jitter    t=0.50    t=0.62    t=0.90    t=0.97    t=0.99")
    for floor in (1e-4, 1e-3, 5e-3, 1e-2, 5e-2):
        cells = []
        for t in (0.50, 0.62, 0.90, 0.97, 0.99):
            try:
                cells.append(f"{margin_for_probability_floor(t, floor):8.4f}")
            except InfeasibleOperatingPoint:
                cells.append("       -")
        print(f"  {floor:<8.4g} {'  '.join(cells)}")
    print("  '-' means no band of any width has that much room on both sides.")


def main() -> int:
    print(__doc__.strip())
    for name, label in (
        ("central", "central regime (threshold near 0.5)"),
        ("shifted", "shifted regime (threshold near 0.85)"),
        ("saturated", "saturated regime (threshold near 0.97) - the R11 case"),
    ):
        y, p, target = regime(name)
        compare(y, p, target, label=label)

    y, p = small_calibration_split()
    compare(y, p, 0.02, label="the audit's R11 fixture (2% budget, 22 negatives)")

    totality_table()
    evidence_table()
    determinism_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
