"""Dependency-free SVG for the three plots a calibration review actually needs.

WHY NOT MATPLOTLIB
------------------
Because these figures go into a reproducibility argument, and matplotlib output
is not reproducible enough to be evidence. Its SVG carries a creation
timestamp and a version comment; font resolution differs per machine, so text
metrics — and therefore layout — differ per machine; and minor releases change
default styling. Two runs of the same evaluation on two boxes produce two
different files, which means a snapshot test cannot exist and a figure in a
report cannot be traced back to the run that made it.

The plots here are built by string concatenation from numbers that are already
computed. Same inputs, same bytes, on any machine, with no dependency at all —
which makes a snapshot test the natural way to test them, and a snapshot test on
a chart is worth having: it catches the axis that silently stopped being drawn.

DETERMINISM RULES, ENFORCED BY TESTS
------------------------------------
* every float goes through :func:`_f`, fixed precision, with ``-0.00`` folded to
  ``0.00`` — the sign of a rounded-away negative is the classic source of a
  diff between two runs that computed the same thing;
* no timestamps, no random or address-derived ids, no ``set`` iteration;
* text is escaped, never interpolated raw;
* the font is a stack of generic families, so a missing font changes rendering
  on screen but never changes the bytes.

WHAT THESE PLOTS ARE FOR
------------------------
A reliability diagram is the only view that shows *where* a model is
miscalibrated rather than by how much on average. ECE is one number and hides
direction: a model overconfident at the top and underconfident at the bottom can
report a small ECE while being wrong about every decision near the threshold.
The histogram underneath it is not decoration — a reliability curve computed
from a bin holding four samples is noise, and the only way to see that is to
plot the counts next to it.
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts.common import ThresholdInfo
from evaluation.calibration import ReliabilityDiagram

# A palette that survives greyscale printing and the common colour-vision
# deficiencies: the pairs differ in lightness as well as hue, so nothing here
# depends on telling red from green.
INK = "#1b1f24"
MUTED = "#8a9199"
GRID = "#e3e6e9"
REFERENCE = "#b0b6bc"
REAL = "#2f6f9f"
FAKE = "#c25a30"
GAP_OVER = "#c25a30"
GAP_UNDER = "#2f6f9f"
BAND = "#f0d97d"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def _f(value: float, places: int = 2) -> str:
    """Fixed-precision, sign-stable float formatting.

    ``-0.00`` becomes ``0.00``. Without this, a coordinate that rounds to zero
    from below renders with a minus sign and a byte-identical computation on two
    machines produces two different files.
    """
    text = f"{value:.{places}f}"
    if float(text) == 0.0:
        return f"{0.0:.{places}f}"
    return text


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


@dataclass(frozen=True)
class Frame:
    """The plot rectangle inside the canvas, and the data range it maps."""

    width: int
    height: int
    left: int = 52
    right: int = 16
    top: int = 60
    bottom: int = 48
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    @property
    def plot_w(self) -> float:
        return self.width - self.left - self.right

    @property
    def plot_h(self) -> float:
        return self.height - self.top - self.bottom

    def x(self, value: float) -> float:
        span = self.x_max - self.x_min or 1.0
        return self.left + (value - self.x_min) / span * self.plot_w

    def y(self, value: float) -> float:
        """Note the flip: SVG's y grows downward and a chart's does not."""
        span = self.y_max - self.y_min or 1.0
        return self.top + self.plot_h - (value - self.y_min) / span * self.plot_h


def _header(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_esc(title)}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="16" y="22" font-family="{FONT}" font-size="13" '
        f'font-weight="600" fill="{INK}">{_esc(title)}</text>',
    ]
    if subtitle:
        parts.append(
            f'<text x="16" y="36" font-family="{FONT}" font-size="10" '
            f'fill="{MUTED}">{_esc(subtitle)}</text>'
        )
    return parts


def _axes(
    frame: Frame,
    *,
    x_label: str,
    y_label: str,
    ticks: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    y_ticks: tuple[float, ...] | None = None,
    y_format: int = 2,
) -> list[str]:
    out: list[str] = []
    for t in y_ticks if y_ticks is not None else ticks:
        y = frame.y(t)
        out.append(
            f'<line x1="{_f(frame.x(frame.x_min))}" y1="{_f(y)}" '
            f'x2="{_f(frame.x(frame.x_max))}" y2="{_f(y)}" stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{frame.left - 8}" y="{_f(y + 3.5)}" text-anchor="end" '
            f'font-family="{FONT}" font-size="9" fill="{MUTED}">{_f(t, y_format)}</text>'
        )
    for t in ticks:
        x = frame.x(t)
        out.append(
            f'<text x="{_f(x)}" y="{_f(frame.y(frame.y_min) + 14)}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="9" fill="{MUTED}">{_f(t)}</text>'
        )
    out.append(
        f'<rect x="{frame.left}" y="{frame.top}" width="{_f(frame.plot_w)}" '
        f'height="{_f(frame.plot_h)}" fill="none" stroke="{MUTED}" stroke-width="1"/>'
    )
    out.append(
        f'<text x="{_f(frame.left + frame.plot_w / 2)}" '
        f'y="{frame.height - 10}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="10" fill="{INK}">{_esc(x_label)}</text>'
    )
    out.append(
        f'<text x="14" y="{_f(frame.top + frame.plot_h / 2)}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="10" fill="{INK}" '
        f'transform="rotate(-90 14 {_f(frame.top + frame.plot_h / 2)})">'
        f"{_esc(y_label)}</text>"
    )
    return out


def _legend(x: float, y: float, entries: list[tuple[str, str]]) -> list[str]:
    """Swatch + label, laid out left to right from ``x``."""
    out: list[str] = []
    cursor = x
    for colour, label in entries:
        out.append(
            f'<rect x="{_f(cursor)}" y="{_f(y - 7)}" width="9" height="9" '
            f'fill="{colour}" opacity="0.85"/>'
        )
        out.append(
            f'<text x="{_f(cursor + 13)}" y="{_f(y)}" font-family="{FONT}" '
            f'font-size="9" fill="{INK}">{_esc(label)}</text>'
        )
        cursor += 22 + 5.6 * len(label)
    return out


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------


def reliability_svg(
    diagram: ReliabilityDiagram,
    *,
    title: str = "Reliability",
    width: int = 440,
    height: int = 420,
    show_counts: bool = True,
) -> str:
    """Claimed probability against observed frequency, with the gaps drawn.

    The diagonal is perfect calibration. A bar is drawn from the diagonal to the
    observed rate, so its *height* is the error and its *direction* is the sign
    of it — above the line the model delivered more than it claimed, below it
    claimed more than it delivered. Colour encodes the same thing redundantly,
    which is what keeps the figure readable in greyscale.

    Empty bins are skipped rather than drawn at zero. A bin nobody landed in has
    no observed rate, and plotting one at 0.0 invents a catastrophic
    miscalibration out of an absence of data — the single most misleading thing
    a reliability diagram can do.
    """
    populated = diagram.populated
    frame = Frame(width=width, height=height, bottom=68 if show_counts else 48)
    subtitle = (
        f"n={diagram.n}  ECE {_f(diagram.ece(), 4)}  MCE {_f(diagram.mce(), 4)}  "
        f"{diagram.strategy} bins"
    )
    parts = _header(width, height, title, subtitle)
    parts += _axes(frame, x_label="mean predicted probability", y_label="observed frequency")

    parts.append(
        f'<line x1="{_f(frame.x(0.0))}" y1="{_f(frame.y(0.0))}" '
        f'x2="{_f(frame.x(1.0))}" y2="{_f(frame.y(1.0))}" stroke="{REFERENCE}" '
        f'stroke-width="1" stroke-dasharray="4 3"/>'
    )

    for b in populated:
        x = frame.x(b.mean_predicted)
        y_pred = frame.y(b.mean_predicted)
        y_obs = frame.y(b.empirical_rate)
        colour = GAP_OVER if b.gap > 0 else GAP_UNDER
        if abs(b.gap) > 1e-12:
            parts.append(
                f'<line x1="{_f(x)}" y1="{_f(y_pred)}" x2="{_f(x)}" y2="{_f(y_obs)}" '
                f'stroke="{colour}" stroke-width="6" opacity="0.35"/>'
            )
        parts.append(
            f'<circle cx="{_f(x)}" cy="{_f(y_obs)}" r="3.5" fill="{colour}"/>'
        )

    if len(populated) > 1:
        points = " ".join(
            f"{_f(frame.x(b.mean_predicted))},{_f(frame.y(b.empirical_rate))}"
            for b in populated
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{INK}" '
            f'stroke-width="1.2" opacity="0.55"/>'
        )

    if show_counts:
        # A count strip under the axis, not a second panel. It answers exactly
        # one question — "is that point made of ten samples or ten thousand" —
        # and giving it its own axis would imply it deserves more attention than
        # the curve it qualifies.
        strip_top = height - 34
        strip_h = 16.0
        biggest = max((b.count for b in populated), default=0) or 1
        parts.append(
            f'<text x="{frame.left}" y="{_f(strip_top - 4)}" font-family="{FONT}" '
            f'font-size="9" fill="{MUTED}">samples per bin (max {biggest})</text>'
        )
        for b in diagram.bins:
            x0, x1 = frame.x(b.lo), frame.x(b.hi)
            w = max(x1 - x0 - 1.0, 0.6)
            h = strip_h * (b.count / biggest)
            parts.append(
                f'<rect x="{_f(x0)}" y="{_f(strip_top + strip_h - h)}" '
                f'width="{_f(w)}" height="{_f(h)}" fill="{MUTED}" opacity="0.55"/>'
            )

    parts += _legend(
        frame.left,
        52.0,
        [(GAP_OVER, "overconfident"), (GAP_UNDER, "underconfident")],
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def calibration_comparison_svg(
    before: ReliabilityDiagram,
    after: ReliabilityDiagram,
    *,
    title: str = "Calibration: before and after",
    width: int = 440,
    height: int = 400,
) -> str:
    """Two reliability curves on one pair of axes.

    The plot that makes a temperature fit arguable rather than assertable: the
    "after" curve should sit closer to the diagonal *along its whole length*.
    A fit that improves the middle and worsens the tails moves ECE down and is
    the wrong trade for a detector, whose decisions all live in one tail — and
    that is visible here and in no single number.
    """
    frame = Frame(width=width, height=height)
    parts = _header(
        width,
        height,
        title,
        f"ECE {_f(before.ece(), 4)} -> {_f(after.ece(), 4)}   "
        f"MCE {_f(before.mce(), 4)} -> {_f(after.mce(), 4)}   n={after.n}",
    )
    parts += _axes(frame, x_label="mean predicted probability", y_label="observed frequency")
    parts.append(
        f'<line x1="{_f(frame.x(0.0))}" y1="{_f(frame.y(0.0))}" '
        f'x2="{_f(frame.x(1.0))}" y2="{_f(frame.y(1.0))}" stroke="{REFERENCE}" '
        f'stroke-width="1" stroke-dasharray="4 3"/>'
    )

    for diagram, colour, dash in ((before, MUTED, "5 3"), (after, REAL, "")):
        populated = diagram.populated
        if len(populated) > 1:
            points = " ".join(
                f"{_f(frame.x(b.mean_predicted))},{_f(frame.y(b.empirical_rate))}"
                for b in populated
            )
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{colour}" '
                f'stroke-width="1.6"{dash_attr}/>'
            )
        for b in populated:
            parts.append(
                f'<circle cx="{_f(frame.x(b.mean_predicted))}" '
                f'cy="{_f(frame.y(b.empirical_rate))}" r="2.8" fill="{colour}"/>'
            )

    parts += _legend(frame.left, 52.0, [(MUTED, "before"), (REAL, "after")])
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# Confidence histogram
# --------------------------------------------------------------------------


def confidence_histogram_svg(
    scores,
    labels=None,
    *,
    threshold: ThresholdInfo | None = None,
    n_bins: int = 20,
    title: str = "Score distribution",
    width: int = 460,
    height: int = 300,
) -> str:
    """Where the scores actually land, with the abstention band drawn over them.

    Split by class when ``labels`` are given, because the overlap between the two
    distributions *is* the model's discrimination and a pooled histogram hides
    it entirely. Counts are drawn as outlines rather than filled blocks so the
    overlapping region stays readable — the region that matters most is exactly
    the one a filled chart obscures.

    Passing ``threshold`` shades the band. Seeing how much of each class sits
    inside it is the only intuitive account of what widening the band costs:
    those are the cases the system will decline to answer.
    """
    values = [float(s) for s in scores]
    if not values:
        raise ValueError("cannot plot an empty score list")
    label_list = [int(y) for y in labels] if labels is not None else None
    if label_list is not None and len(label_list) != len(values):
        raise ValueError(
            f"scores and labels must be the same length, got {len(values)} and "
            f"{len(label_list)}"
        )
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    edges = [i / n_bins for i in range(n_bins + 1)]
    series: list[tuple[str, str, list[int]]] = []
    if label_list is None:
        series.append(("all", INK, _bin_counts(values, n_bins)))
    else:
        real = [v for v, y in zip(values, label_list) if y == 0]
        fake = [v for v, y in zip(values, label_list) if y == 1]
        series.append(("authentic", REAL, _bin_counts(real, n_bins)))
        series.append(("manipulated", FAKE, _bin_counts(fake, n_bins)))

    biggest = max((max(counts) for _, _, counts in series), default=0) or 1
    frame = Frame(width=width, height=height, y_max=float(biggest))
    subtitle = f"n={len(values)}  {n_bins} bins"
    if threshold is not None:
        subtitle += (
            f"  band [{_f(threshold.band_lo, 3)}, {_f(threshold.band_hi, 3)})"
        )
    parts = _header(width, height, title, subtitle)

    if threshold is not None:
        x0, x1 = frame.x(threshold.band_lo), frame.x(threshold.band_hi)
        parts.append(
            f'<rect x="{_f(x0)}" y="{frame.top}" width="{_f(x1 - x0)}" '
            f'height="{_f(frame.plot_h)}" fill="{BAND}" opacity="0.35"/>'
        )

    step = max(1, biggest // 4)
    y_ticks = tuple(float(v) for v in range(0, biggest + 1, step))
    parts += _axes(
        frame,
        x_label="p(manipulated)",
        y_label="count",
        y_ticks=y_ticks,
        y_format=0,
    )

    for _, colour, counts in series:
        # A step outline: horizontal run across each bin, vertical connector
        # between neighbours. Built explicitly rather than with a polyline over
        # bin centres, which would slope between bins and imply a continuity the
        # data does not have.
        points: list[str] = [f"{_f(frame.x(edges[0]))},{_f(frame.y(0.0))}"]
        for i, count in enumerate(counts):
            y = frame.y(float(count))
            points.append(f"{_f(frame.x(edges[i]))},{_f(y)}")
            points.append(f"{_f(frame.x(edges[i + 1]))},{_f(y)}")
        points.append(f"{_f(frame.x(edges[-1]))},{_f(frame.y(0.0))}")
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="{colour}" '
            f'fill-opacity="0.18" stroke="{colour}" stroke-width="1.4"/>'
        )

    if threshold is not None:
        x = frame.x(threshold.value)
        parts.append(
            f'<line x1="{_f(x)}" y1="{frame.top}" x2="{_f(x)}" '
            f'y2="{_f(frame.top + frame.plot_h)}" stroke="{INK}" stroke-width="1" '
            f'stroke-dasharray="3 2"/>'
        )

    parts += _legend(frame.left, 52.0, [(c, n) for n, c, _ in series])
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _bin_counts(values: list[float], n_bins: int) -> list[int]:
    """Counts per equal-width bin over [0, 1]; the last bin includes 1.0.

    The same half-open convention as ``reliability_diagram``, and for the same
    reason: without the closed right edge on the final bin a score of exactly
    1.0 falls out of the histogram, and a saturated model is precisely the one
    that produces those.
    """
    counts = [0] * n_bins
    for v in values:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"score {v} lies outside [0, 1]")
        index = min(int(v * n_bins), n_bins - 1)
        counts[index] += 1
    return counts
