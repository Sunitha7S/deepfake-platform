"""Deterministic SVG rendering, pinned by snapshot.

A chart is the one artefact in this repository whose defects are invisible to
ordinary assertions. ``assert "<svg" in out`` passes for a blank canvas, for a
plot with no axis, and for one where every point landed on top of another. So
the real check here is a byte-for-byte snapshot: the committed files under
``tests/snapshots/`` are the expected rendering, and any change to them shows up
as a reviewable diff rather than as a chart nobody looked at.

    UPDATE_SNAPSHOTS=1 python tests/test_evaluation_svg.py

rewrites them. Do that deliberately and read the diff — a snapshot updated
without being read is a snapshot that is no longer testing anything.

The determinism tests around the snapshots are what make snapshotting viable at
all: they assert the properties (fixed precision, no timestamps, no set
iteration, stable ordering) that would otherwise make the committed files fail
on somebody else's machine for no reason and train everyone to regenerate them
on sight.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.common import DEFAULT_LOG_ODDS_MARGIN, ThresholdInfo  # noqa: E402
from evaluation.calibration import reliability_diagram  # noqa: E402
from evaluation.svg import (  # noqa: E402
    _f,
    calibration_comparison_svg,
    confidence_histogram_svg,
    reliability_svg,
)

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
UPDATE = os.environ.get("UPDATE_SNAPSHOTS") == "1"


# --------------------------------------------------------------------------
# Fixtures: deterministic, and never numpy.random without a seed
# --------------------------------------------------------------------------


def overconfident(n: int = 600) -> tuple[list[float], list[int]]:
    """Scores from a model that claims more than it delivers.

    Built from an integer sweep rather than a sampler so the fixture is
    identical on every machine and every numpy version — a snapshot test whose
    input depends on a random number generator's implementation is a snapshot
    test that fails on upgrade.
    """
    scores: list[float] = []
    labels: list[int] = []
    for i in range(n):
        u = (i + 0.5) / n
        z = math.log(u / (1.0 - u))
        p = 1.0 / (1.0 + math.exp(-z * 2.2))  # sharpened: overconfident
        scores.append(p)
        # The true rate is the unsharpened probability, so the model's claim
        # exceeds reality everywhere above 0.5 and falls short below it.
        labels.append(1 if ((i * 7919) % 1000) / 1000.0 < u else 0)
    return scores, labels


def two_class_scores() -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    for i in range(400):
        u = (i + 0.5) / 400
        z = math.log(u / (1.0 - u))
        label = i % 2
        p = 1.0 / (1.0 + math.exp(-(z + (1.6 if label else -1.6))))
        scores.append(p)
        labels.append(label)
    return scores, labels


def check_snapshot(name: str, content: str) -> None:
    path = SNAPSHOT_DIR / name
    if UPDATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if not UPDATE:
            raise AssertionError(
                f"snapshot {name} did not exist and has been written; review it "
                "and re-run"
            )
        return
    expected = path.read_text(encoding="utf-8")
    if expected != content:
        for i, (a, b) in enumerate(zip(expected, content)):
            if a != b:
                raise AssertionError(
                    f"{name} differs from its snapshot at character {i}:\n"
                    f"  expected …{expected[max(0, i - 60):i + 60]!r}\n"
                    f"  actual   …{content[max(0, i - 60):i + 60]!r}"
                )
        raise AssertionError(
            f"{name} differs in length from its snapshot "
            f"({len(expected)} vs {len(content)} characters)"
        )


# --------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------


def test_reliability_snapshot():
    scores, labels = overconfident()
    diagram = reliability_diagram(labels, scores, n_bins=10)
    check_snapshot("reliability.svg", reliability_svg(diagram))


def test_reliability_quantile_bins_snapshot():
    scores, labels = overconfident()
    diagram = reliability_diagram(labels, scores, n_bins=8, strategy="quantile")
    check_snapshot(
        "reliability_quantile.svg",
        reliability_svg(diagram, title="Reliability (quantile bins)"),
    )


def test_calibration_comparison_snapshot():
    scores, labels = overconfident()
    before = reliability_diagram(labels, scores, n_bins=10)
    softened = [1.0 / (1.0 + math.exp(-math.log(s / (1.0 - s)) / 2.2)) for s in scores]
    after = reliability_diagram(labels, softened, n_bins=10)
    check_snapshot("calibration_comparison.svg", calibration_comparison_svg(before, after))


def test_confidence_histogram_snapshot():
    scores, labels = two_class_scores()
    threshold = ThresholdInfo.at_margin(
        value=0.5, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
    )
    check_snapshot(
        "confidence_histogram.svg",
        confidence_histogram_svg(scores, labels, threshold=threshold),
    )


def test_unlabelled_histogram_snapshot():
    scores, _ = two_class_scores()
    check_snapshot(
        "confidence_histogram_pooled.svg",
        confidence_histogram_svg(scores, title="Pooled scores"),
    )


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_rendering_twice_gives_identical_bytes():
    scores, labels = overconfident()
    diagram = reliability_diagram(labels, scores, n_bins=10)
    assert reliability_svg(diagram) == reliability_svg(diagram)
    assert confidence_histogram_svg(scores, labels) == confidence_histogram_svg(
        scores, labels
    )


def test_an_equivalent_diagram_renders_identically():
    """Same numbers computed twice from scratch, not the same object twice."""
    scores, labels = overconfident()
    a = reliability_svg(reliability_diagram(labels, scores, n_bins=10))
    b = reliability_svg(reliability_diagram(list(labels), list(scores), n_bins=10))
    assert a == b


def test_no_timestamp_or_generator_comment():
    """The reason this module exists instead of matplotlib."""
    scores, labels = overconfident()
    out = reliability_svg(reliability_diagram(labels, scores, n_bins=10))
    lowered = out.lower()
    for token in ("date", "created", "generated", "version", "matplotlib", "<!--"):
        assert token not in lowered, f"output contains {token!r}"


def test_negative_zero_is_folded():
    """``-0.00`` and ``0.00`` are the same coordinate and must be the same bytes."""
    assert _f(-0.0001) == _f(0.0001) == "0.00"
    assert _f(-0.4) == "-0.40"


def test_output_is_wellformed_enough_to_parse():
    import xml.etree.ElementTree as ET

    scores, labels = two_class_scores()
    threshold = ThresholdInfo.at_margin(
        value=0.5, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
    )
    for svg in (
        reliability_svg(reliability_diagram(labels, scores, n_bins=10)),
        confidence_histogram_svg(scores, labels, threshold=threshold),
    ):
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")


def test_titles_are_escaped():
    scores, _ = two_class_scores()
    out = confidence_histogram_svg(scores, title='Model "A" <B> & C')
    assert "&quot;" in out and "&lt;B&gt;" in out and "&amp;" in out
    assert "<B>" not in out


# --------------------------------------------------------------------------
# The plots say what they claim to say
# --------------------------------------------------------------------------


def test_empty_bins_are_not_drawn_as_zero():
    """The most misleading thing a reliability diagram can do.

    A bin nobody landed in has no observed rate; plotting it at 0.0 invents a
    catastrophic miscalibration out of missing data.
    """
    # Every score sits in the top decile, so nine of ten uniform bins are empty.
    scores = [0.90 + i * 0.001 for i in range(80)]
    labels = [i % 2 for i in range(80)]
    diagram = reliability_diagram(labels, scores, n_bins=10)
    assert len(diagram.populated) == 1, "fixture no longer has empty bins"
    out = reliability_svg(diagram)
    assert out.count("<circle") == 1, (
        "an empty bin was plotted; it would read as a total calibration failure"
    )


def test_the_abstention_band_is_drawn_when_a_threshold_is_given():
    scores, labels = two_class_scores()
    narrow = ThresholdInfo.at_margin(value=0.5, margin=0.04, target_fpr=0.02)
    wide = ThresholdInfo.at_margin(value=0.5, margin=0.8473, target_fpr=0.02)
    assert "#f0d97d" not in confidence_histogram_svg(scores, labels)
    assert "#f0d97d" in confidence_histogram_svg(scores, labels, threshold=narrow)
    # A wider band must actually be wider on the page — the trade the figure is
    # drawn to make legible.
    def band_width_px(svg: str) -> float:
        marker = svg.split('fill="#f0d97d"')[0].rsplit("<rect", 1)[1]
        return float(marker.split('width="')[1].split('"')[0])

    assert band_width_px(confidence_histogram_svg(scores, labels, threshold=wide)) > (
        band_width_px(confidence_histogram_svg(scores, labels, threshold=narrow))
    )


def test_both_classes_are_drawn_when_labels_are_given():
    scores, labels = two_class_scores()
    out = confidence_histogram_svg(scores, labels)
    assert "authentic" in out and "manipulated" in out
    assert out.count("<polyline") == 2


def test_a_score_of_exactly_one_still_lands_in_the_histogram():
    """Saturated models produce them, and a dropped sample is a silent undercount."""
    from evaluation.svg import _bin_counts

    assert _bin_counts([1.0, 1.0, 0.5], 4) == [0, 0, 1, 2], (
        "the closed right edge on the final bin is gone; p == 1.0 falls out of "
        "the histogram entirely"
    )
    out = confidence_histogram_svg([1.0, 1.0, 0.5], n_bins=4)
    assert "n=3" in out


def test_a_score_outside_the_unit_interval_is_refused():
    try:
        confidence_histogram_svg([0.5, 1.4])
    except ValueError as exc:
        assert "outside [0, 1]" in str(exc)
        return
    raise AssertionError("a score above 1 was plotted")


def test_mismatched_labels_are_refused():
    try:
        confidence_histogram_svg([0.1, 0.2], [1])
    except ValueError as exc:
        assert "same length" in str(exc)
        return
    raise AssertionError("a short label list was accepted")


def test_an_empty_score_list_is_refused():
    try:
        confidence_histogram_svg([])
    except ValueError as exc:
        assert "empty" in str(exc)
        return
    raise AssertionError("an empty histogram was rendered")


def test_the_ece_in_the_subtitle_matches_the_diagram():
    scores, labels = overconfident()
    diagram = reliability_diagram(labels, scores, n_bins=10)
    out = reliability_svg(diagram)
    assert f"ECE {diagram.ece():.4f}" in out


def test_the_comparison_shows_both_curves():
    scores, labels = overconfident()
    before = reliability_diagram(labels, scores, n_bins=10)
    softened = [1.0 / (1.0 + math.exp(-math.log(s / (1.0 - s)) / 2.2)) for s in scores]
    after = reliability_diagram(labels, softened, n_bins=10)
    out = calibration_comparison_svg(before, after)
    assert out.count("<polyline") == 2
    assert "before" in out and "after" in out


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
