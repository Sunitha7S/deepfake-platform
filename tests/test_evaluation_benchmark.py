"""Benchmark harness, regression gating, and effective model versioning.

Named doubles for the three mistakes:

    blend_predictor       ``skill * label + (1 - skill) * noise``. The obvious
                          stub, and it never overlaps for skill >= 0.5, so every
                          setting is a perfect ranker and no regression can be
                          manufactured. A gate that can never fire looks exactly
                          like a gate that passes.

    unpaired_bootstrap    resamples the two models independently. Ignores that
                          they share a test set, so it reports an interval far
                          wider than the evidence and calls real improvements
                          insignificant.

    overall_only_gate     checks aggregate AUC alone. Passes a model that
                          improved on average by collapsing on a slice.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.common import ThresholdInfo  # noqa: E402
from evaluation.benchmark import (  # noqa: E402
    BenchmarkRun,
    StubPredictor,
    append_history,
    dataset_fingerprint,
    paired_bootstrap,
    read_history,
    regression_gate,
    run_benchmark,
)
from evaluation.dataset import MediaItem  # noqa: E402
from evaluation.metrics import roc_auc  # noqa: E402
from inference.models import CalibrationSpec, ModelSpec  # noqa: E402

THRESHOLD = ThresholdInfo.at_margin(value=0.62, margin=0.4299, target_fpr=0.02)


def corpus(n: int = 600) -> list[MediaItem]:
    return [
        MediaItem(f"m{i}", f"s{i // 4}", "ffpp", i % 2, subject_id=f"p{i // 8}")
        for i in range(n)
    ]


def labels(items: list[MediaItem]) -> np.ndarray:
    return np.array([i.label for i in items], dtype=np.float64)


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


class BlendPredictor:
    """The stub that cannot regress. ``skill*label + (1-skill)*noise``."""

    def __init__(self, *, name: str = "blend", version: str = "b-0", skill: float = 0.8):
        self.name, self.version, self.skill = name, version, skill

    def predict(self, items: list[MediaItem]) -> np.ndarray:
        import hashlib

        out = []
        for item in items:
            digest = hashlib.sha256(f"{self.version}:{item.media_id}".encode()).hexdigest()
            noise = int(digest[:12], 16) / float(0x1000000000000)
            out.append(self.skill * item.label + (1.0 - self.skill) * noise)
        return np.array(out, dtype=np.float64)


def unpaired_bootstrap(y, a, b, *, resamples=400, seed=0) -> tuple[float, float]:
    """Resamples each model independently. Loses the pairing, widens the CI."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    deltas = []
    for _ in range(resamples):
        ia = rng.integers(0, y.size, y.size)
        ib = rng.integers(0, y.size, y.size)  # a *different* resample
        if len(np.unique(y[ia])) < 2 or len(np.unique(y[ib])) < 2:
            continue
        deltas.append(roc_auc(y[ib], np.asarray(b)[ib]) - roc_auc(y[ia], np.asarray(a)[ia]))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return float(lo), float(hi)


def overall_only_gate(incumbent: BenchmarkRun, challenger: BenchmarkRun) -> bool:
    """Aggregate AUC only. Blind to a slice collapse."""
    return challenger.roc_auc >= incumbent.roc_auc - 0.005


# --------------------------------------------------------------------------
# The stub must actually be tunable
# --------------------------------------------------------------------------


def test_stub_skill_controls_separability():
    items = corpus()
    y = labels(items)
    aucs = [
        roc_auc(y, StubPredictor(version=f"v{s}", skill=s).predict(items))
        for s in (0.0, 0.15, 0.3, 0.5)
    ]
    assert aucs == sorted(aucs), f"AUC must rise with skill, got {aucs}"
    assert aucs[0] < 0.60, "skill 0 should be near chance"
    assert aucs[-1] > 0.95, "skill 0.5 should be strong"


def test_the_blend_double_is_perfect_at_every_useful_skill():
    """Why the obvious stub is unusable: no overlap, so AUC is pinned at 1.0."""
    items = corpus()
    y = labels(items)
    for skill in (0.5, 0.6, 0.8, 0.95):
        assert roc_auc(y, BlendPredictor(version=f"b{skill}", skill=skill).predict(items)) == 1.0, (
            "the blend double no longer separates perfectly; this check is vacuous"
        )
    # The corrected stub does not have this property.
    assert roc_auc(y, StubPredictor(version="s", skill=0.2).predict(items)) < 1.0


def test_stub_predictions_are_deterministic():
    items = corpus(100)
    a = StubPredictor(version="v1", skill=0.4).predict(items)
    b = StubPredictor(version="v1", skill=0.4).predict(items)
    assert np.array_equal(a, b)


def test_a_different_stub_version_scores_differently():
    items = corpus(100)
    a = StubPredictor(version="v1", skill=0.4).predict(items)
    b = StubPredictor(version="v2", skill=0.4).predict(items)
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


def test_a_run_records_the_version_and_the_dataset():
    items = corpus()
    run = run_benchmark(
        StubPredictor(version="v1", skill=0.3),
        items,
        threshold=THRESHOLD,
        effective_model_version="w-1+c.abc.g.g0",
    )
    assert run.effective_model_version == "w-1+c.abc.g.g0"
    assert run.dataset_fingerprint == dataset_fingerprint(items)
    assert run.n == len(items)
    assert run.threshold == THRESHOLD.value
    assert run.band_width == THRESHOLD.band_width


def test_a_run_round_trips_through_history():
    items = corpus(200)
    run = run_benchmark(
        StubPredictor(version="v1", skill=0.3), items,
        threshold=THRESHOLD, effective_model_version="w-1",
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "history.jsonl"
        append_history(run, path)
        append_history(run, path)
        rows = read_history(path)
    assert len(rows) == 2, "history must be append-only"
    assert rows[0]["effective_model_version"] == "w-1"
    assert rows[0]["roc_auc"] == run.roc_auc


def test_slices_are_measured_separately():
    items = corpus()
    hard = [i for i in items if int(i.media_id[1:]) % 3 == 0]
    run = run_benchmark(
        StubPredictor(version="v1", skill=0.25), items,
        threshold=THRESHOLD, effective_model_version="w-1",
        slice_by={"hard": hard},
    )
    assert len(run.slices) == 1
    assert run.slices[0].name == "hard"
    assert run.slices[0].n == len(hard)
    assert run.worst_slice is not None


def test_a_single_class_slice_is_skipped_rather_than_faked():
    items = corpus(200)
    only_fake = [i for i in items if i.label == 1]
    run = run_benchmark(
        StubPredictor(version="v1", skill=0.3), items,
        threshold=THRESHOLD, effective_model_version="w-1",
        slice_by={"fakes_only": only_fake},
    )
    assert run.slices == [], "a slice with one class has no defined AUC"


def test_a_benchmark_needs_both_classes():
    items = [MediaItem(f"m{i}", f"s{i}", "d", 1) for i in range(10)]
    try:
        run_benchmark(StubPredictor(), items, threshold=THRESHOLD, effective_model_version="w")
    except ValueError:
        return
    raise AssertionError("a single-class benchmark was accepted")


# --------------------------------------------------------------------------
# Paired comparison
# --------------------------------------------------------------------------


def test_paired_bootstrap_detects_a_real_improvement():
    items = corpus(800)
    y = labels(items)
    weak = StubPredictor(version="v1", skill=0.10).predict(items)
    strong = StubPredictor(version="v2", skill=0.30).predict(items)
    result = paired_bootstrap(y, weak, strong, resamples=600, seed=1)
    assert result.delta > 0
    assert result.significant, result.describe()
    assert result.win_rate > 0.9


def test_paired_bootstrap_calls_a_tie_a_tie():
    items = corpus(800)
    y = labels(items)
    a = StubPredictor(version="same", skill=0.2).predict(items)
    result = paired_bootstrap(y, a, a.copy(), resamples=400, seed=1)
    assert result.delta == 0.0
    assert not result.significant


def test_the_unpaired_double_reports_a_wider_interval():
    """Losing the pairing attributes shared test-set variance to the difference."""
    items = corpus(600)
    y = labels(items)
    a = StubPredictor(version="v1", skill=0.12).predict(items)
    b = StubPredictor(version="v2", skill=0.20).predict(items)
    paired = paired_bootstrap(y, a, b, resamples=500, seed=2)
    lo, hi = unpaired_bootstrap(y, a, b, resamples=500, seed=2)
    assert (hi - lo) > (paired.ci_hi - paired.ci_lo), (
        "the unpaired double no longer widens the interval; this check is vacuous"
    )


def test_paired_bootstrap_is_reproducible():
    items = corpus(300)
    y = labels(items)
    a = StubPredictor(version="v1", skill=0.15).predict(items)
    b = StubPredictor(version="v2", skill=0.25).predict(items)
    first = paired_bootstrap(y, a, b, resamples=200, seed=9)
    second = paired_bootstrap(y, a, b, resamples=200, seed=9)
    assert first == second


# --------------------------------------------------------------------------
# Regression gate
# --------------------------------------------------------------------------


def _run_pair(skill_a: float, skill_b: float, *, slices: bool = True):
    items = corpus()
    hard = [i for i in items if int(i.media_id[1:]) % 3 == 0]
    slice_by = {"hard": hard} if slices else None
    inc = run_benchmark(
        StubPredictor(name="inc", version="v1", skill=skill_a), items,
        threshold=THRESHOLD, effective_model_version="w-1+c.a.g.g0", slice_by=slice_by,
    )
    cha = run_benchmark(
        StubPredictor(name="cha", version="v2", skill=skill_b), items,
        threshold=THRESHOLD, effective_model_version="w-2+c.a.g.g0", slice_by=slice_by,
    )
    return inc, cha


def test_the_gate_passes_a_genuine_improvement():
    inc, cha = _run_pair(0.20, 0.30)
    assert regression_gate(inc, cha).passed


def test_the_gate_fails_an_overall_regression():
    inc, cha = _run_pair(0.30, 0.10)
    result = regression_gate(inc, cha)
    assert not result.passed
    assert any("overall AUC" in f for f in result.failures), result.describe()


def test_the_gate_refuses_to_compare_different_datasets():
    inc, _ = _run_pair(0.3, 0.3)
    other = run_benchmark(
        StubPredictor(version="v9", skill=0.3), corpus(120),
        threshold=THRESHOLD, effective_model_version="w-9",
    )
    result = regression_gate(inc, other)
    assert not result.passed
    assert any("different datasets" in f for f in result.failures)


def test_the_gate_notices_two_runs_claiming_the_same_version():
    items = corpus(200)
    args = dict(threshold=THRESHOLD, effective_model_version="same-version")
    a = run_benchmark(StubPredictor(version="v1", skill=0.3), items, **args)
    b = run_benchmark(StubPredictor(version="v2", skill=0.2), items, **args)
    result = regression_gate(a, b)
    assert any("same effective model version" in n for n in result.notes), (
        "two materially different models reported one version and nothing said so"
    )


def test_the_overall_only_double_misses_a_slice_collapse():
    """A model can improve on average *while* failing where it already struggled.

    The challenger is better than the incumbent on 87% of the corpus and at
    chance on the remaining 13%. Measured: overall AUC rises 0.932 -> 0.978
    while the hard slice falls 0.934 -> 0.576. A gate that reads only the
    aggregate sees an improvement and ships it.
    """
    items = corpus(900)
    # A slice spanning both labels — every other item is a fake, so slicing on
    # the raw index would select one class and have no defined AUC at all.
    hard = [i for i in items if (int(i.media_id[1:]) // 2) % 8 == 0]
    hard_ids = {i.media_id for i in hard}
    assert {i.label for i in hard} == {0, 1}, "the slice must contain both classes"

    class SliceCollapsing:
        """Strong overall, at chance on the hard slice."""

        name, version = "collapsing", "v2"

        def predict(self, items_):
            good = StubPredictor(version="good", skill=0.60).predict(items_)
            bad = StubPredictor(version="bad", skill=0.0).predict(items_)
            return np.array(
                [b if it.media_id in hard_ids else g for it, g, b in zip(items_, good, bad)]
            )

    inc = run_benchmark(
        StubPredictor(name="inc", version="v1", skill=0.30), items,
        threshold=THRESHOLD, effective_model_version="w-1", slice_by={"hard": hard},
    )
    cha = run_benchmark(
        SliceCollapsing(), items,
        threshold=THRESHOLD, effective_model_version="w-2", slice_by={"hard": hard},
    )
    assert cha.roc_auc > inc.roc_auc, "the fixture must improve on aggregate"
    assert cha.slices[0].roc_auc < inc.slices[0].roc_auc - 0.2, (
        "the fixture must genuinely collapse the slice"
    )
    assert overall_only_gate(inc, cha), (
        "the single-scope double no longer passes this model; check is vacuous"
    )
    result = regression_gate(inc, cha)
    assert not result.passed, "the two-scope gate missed a slice collapse"
    assert any("slice 'hard'" in f for f in result.failures), result.describe()


def test_the_gate_fails_detection_bought_with_false_positives():
    items = corpus(800)
    inc = run_benchmark(
        StubPredictor(name="inc", version="v1", skill=0.35), items,
        threshold=THRESHOLD, effective_model_version="w-1",
    )
    cha = run_benchmark(
        StubPredictor(name="cha", version="v2", skill=0.05), items,
        threshold=THRESHOLD, effective_model_version="w-2",
    )
    result = regression_gate(inc, cha)
    assert not result.passed
    assert any("selective FPR rose" in f for f in result.failures), result.describe()


# --------------------------------------------------------------------------
# Effective model version — every score-affecting knob must move it
# --------------------------------------------------------------------------


BASE = ModelSpec(name="m", version="w-1", backend="stub", gate_fingerprint="g0")


def test_moving_the_threshold_changes_the_effective_version():
    """The requirement stated plainly. A new boundary is a new model."""
    stricter = replace(
        BASE, threshold=ThresholdInfo.at_margin(value=0.80, margin=0.6239, target_fpr=0.005)
    )
    assert BASE.effective_version != stricter.effective_version


def test_widening_the_band_changes_the_effective_version():
    wider = replace(
        BASE, threshold=ThresholdInfo.at_margin(value=0.62, margin=0.8957, target_fpr=0.02)
    )
    assert BASE.effective_version != wider.effective_version


def test_every_score_affecting_knob_produces_a_distinct_version():
    from contracts.enums import BranchId

    variants = {
        "baseline": BASE,
        "weights": replace(BASE, version="w-2"),
        "branch": replace(BASE, branch_versions={BranchId.A: "a-2"}),
        "fusion": replace(BASE, branch_weights={BranchId.A: 0.5, BranchId.B: 0.5, BranchId.P: 0.04}),
        "calibration": replace(BASE, calibration=CalibrationSpec(temperature=1.7, n_samples=900)),
        "threshold": replace(
            BASE, threshold=ThresholdInfo.at_margin(value=0.7, margin=0.481, target_fpr=0.02)
        ),
        "band": replace(
            BASE, threshold=ThresholdInfo.at_margin(value=0.62, margin=0.6555, target_fpr=0.02)
        ),
        "gate": replace(BASE, gate_fingerprint="g1"),
    }
    versions = {name: spec.effective_version for name, spec in variants.items()}
    assert len(set(versions.values())) == len(versions), (
        "two configurations that score differently share a version: "
        f"{ {v: [k for k in versions if versions[k] == v] for v in set(versions.values())} }"
    )


def test_an_identical_configuration_keeps_its_version():
    twin = ModelSpec(name="m", version="w-1", backend="stub", gate_fingerprint="g0")
    assert BASE.effective_version == twin.effective_version
    assert len({BASE.effective_version for _ in range(5)}) == 1


# --------------------------------------------------------------------------
# Layering
# --------------------------------------------------------------------------


def test_the_serving_path_does_not_import_the_evaluation_package():
    """One-way dependency: evaluation may use inference, never the reverse."""
    import subprocess

    root = Path(__file__).resolve().parent.parent
    script = (
        f"import sys; sys.path.insert(0, r'{root}');"
        "import inference, backend.service, workers;"
        "bad=[m for m in sys.modules if m.split('.')[0]=='evaluation'];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip() == "", (
        f"the serving path pulled in evaluation modules: {out.stdout.strip()}"
    )


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
