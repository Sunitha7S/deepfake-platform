"""R2 and R3: a slice cannot vanish, and a run knows where it came from.

R2 — SLICE COMPLETENESS
-----------------------
The old ``regression_gate`` iterated the challenger's slices and looked each one
up in the incumbent. A slice present in the baseline and absent from the
challenger was therefore never examined — no failure, not even a note. The
two-scope gate silently became a one-scope gate, and the interesting part is
that nothing anywhere was broken: ``run_benchmark`` dropped unmeasurable slices
on purpose, and the gate iterated the data it had.

``overall_and_challenger_only_gate`` below reproduces the old logic exactly, so
every test here is measured against the thing it replaced rather than against an
assertion that the new code does what it says.

R3 — REPRODUCIBILITY METADATA
-----------------------------
A run recorded what the numbers were and, of what is needed to obtain them
again, only the model version and a fingerprint of the evaluated items. The
tests here pin that a run now carries its protocol and split lock, that a
history row from before this batch still loads, and that a 1.0.0 baseline is not
mistaken for a run which measured no slices.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.common import ThresholdInfo  # noqa: E402
from evaluation.benchmark import (  # noqa: E402
    RUN_SCHEMA_VERSION,
    BenchmarkReport,
    BenchmarkRun,
    StubPredictor,
    append_history,
    read_runs,
    regression_gate,
    run_benchmark,
)
from evaluation.dataset import MediaItem  # noqa: E402
from evaluation.protocol import (  # noqa: E402
    FFPP_CELEBDF,
    REQUIRED_REPORT_FIELDS,
    DatasetRole,
    select,
)
from evaluation.split_lock import create_lock  # noqa: E402

THRESHOLD = ThresholdInfo.at_margin(value=0.5, margin=0.2007, target_fpr=0.02)


def corpus(n: int = 400, *, dataset: str = "ffpp") -> list[MediaItem]:
    return [
        MediaItem(f"{dataset}_m{i}", f"{dataset}_s{i}", dataset, i % 2, subject_id=f"p{i}")
        for i in range(n)
    ]


def pick(items, role, lock, protocol=FFPP_CELEBDF):
    """Obtain items for a role the only way a run may record a protocol.

    Since R5 a protocol cannot be attached to a hand-assembled list, so every
    test here that wants a protocol on its run goes through selection — which
    is the point: the tests exercise the compliant path because it is now the
    only path.
    """
    return select(
        protocol, role, items, lock.assignments, lock_hash=lock.content_hash()
    )


def calibrate(items, lock, *, min_samples: int = 20):
    """A real protocol-fitted calibration witness for the report fixtures.

    Since Batch 6 a report cannot claim compliance from a caller-supplied
    fingerprint string, so these fixtures fit for real. ``min_samples`` is
    lowered because these suites are about *report* completeness — the
    production floor of 200 has its own test in ``test_calibration_witness``,
    and dragging a 1400-item corpus through every report test to satisfy it
    would slow the suite to prove something already proved elsewhere.
    """
    from evaluation.calibration_protocol import fit_calibration

    selection = pick(items, DatasetRole.CALIBRATION, lock)
    y = np.array([i.label for i in selection], dtype=np.float64)
    assert len(set(y.tolist())) == 2, "the calibration split needs both classes"
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 2.0, y.size) + 1.5 * (2.0 * y - 1.0)
    # A 20% budget and an explicit margin, not the shipped 2%/ln(1.5). On a
    # calibration split this small a 2% budget puts the threshold at the largest
    # negative score — around 0.97 — which is a fine band but a badly determined
    # threshold, and these are tests about reports rather than about fits.
    #
    # Before Batch 7 this fixture could not have used the shipped band at any
    # budget: no 0.20-wide band fits around 0.97 (R11). The compensation here is
    # now for the threshold's stability alone.
    return fit_calibration(
        selection, z, y, target_fpr=0.2, margin=0.2007, min_samples=min_samples
    ).witness


def tmp_path(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix="dfp-bench-")) / name


def bench(items, *, version: str, skill: float = 0.4, slice_by=None, **kw) -> BenchmarkRun:
    return run_benchmark(
        StubPredictor(name=f"p{version}", version=version, skill=skill),
        items,
        threshold=THRESHOLD,
        effective_model_version=version,
        slice_by=slice_by,
        **kw,
    )


# --------------------------------------------------------------------------
# R2 — the old gate, reproduced, so the fix is measured against it
# --------------------------------------------------------------------------


def overall_and_challenger_only_gate(
    incumbent: BenchmarkRun, challenger: BenchmarkRun, *, max_slice_auc_drop: float = 0.02
) -> bool:
    """The pre-Batch-4 slice comparison: iterate the challenger, look up the incumbent."""
    by_name = {s.name: s for s in incumbent.slices}
    for s in challenger.slices:
        before = by_name.get(s.name)
        if before is None:
            continue
        if before.roc_auc - s.roc_auc > max_slice_auc_drop:
            return False
    return True


def test_the_old_gate_misses_a_vanished_slice_and_the_new_one_does_not():
    """The headline R2 test. The double must pass where the real gate fails."""
    items = corpus(400)
    hard = [i for i in items if int(i.media_id.rsplit("m", 1)[1]) % 4 < 2]

    inc = bench(items, version="v1", skill=0.5, slice_by={"hard": hard})
    cha = bench(items, version="v2", skill=0.5, slice_by={})  # the slice is simply gone

    assert inc.slices and not cha.slices, "fixture: the slice must vanish"
    assert overall_and_challenger_only_gate(inc, cha), (
        "the old logic no longer passes a vanished slice; this double is broken"
    )

    result = regression_gate(inc, cha)
    assert not result.passed, "a slice with a baseline vanished and the gate passed"
    assert any("missing from the challenger" in f for f in result.failures), (
        result.describe()
    )
    assert any("'hard'" in f for f in result.failures), result.describe()


def test_a_slice_that_went_single_class_is_named_as_such():
    """The realistic way a slice disappears: an authentic-only acquisition."""
    items = corpus(400)
    hard = [i for i in items if int(i.media_id.rsplit("m", 1)[1]) % 4 < 2]
    real_only = [i for i in items if i.label == 0][:40]

    inc = bench(items, version="v1", slice_by={"hard": hard})
    cha = bench(items, version="v2", slice_by={"hard": real_only})

    assert cha.single_class_slices == ("hard",), (
        f"fixture: expected a single-class slice, got {cha.single_class_slices}"
    )
    result = regression_gate(inc, cha)
    assert not result.passed
    assert any("single-class" in f for f in result.failures), result.describe()


def test_an_empty_slice_is_named_as_such():
    items = corpus(300)
    hard = [i for i in items if int(i.media_id.rsplit("m", 1)[1]) % 4 < 2]
    elsewhere = corpus(20, dataset="celebdf")  # no overlap with `items`

    inc = bench(items, version="v1", slice_by={"hard": hard})
    cha = bench(items, version="v2", slice_by={"hard": elsewhere})

    assert cha.empty_slices == ("hard",), f"fixture: got {cha.empty_slices}"
    result = regression_gate(inc, cha)
    assert not result.passed
    assert any("matched no evaluated items" in f for f in result.failures), (
        result.describe()
    )


def test_a_challenger_only_slice_is_noted_and_does_not_fail():
    """A new slice has no baseline. That is information, not a regression."""
    items = corpus(400)
    hard = [i for i in items if int(i.media_id.rsplit("m", 1)[1]) % 4 < 2]

    inc = bench(items, version="v1", skill=0.4)
    cha = bench(items, version="v2", skill=0.4, slice_by={"hard": hard})

    result = regression_gate(inc, cha)
    assert result.passed, result.describe()
    assert any("new and has no baseline" in n for n in result.notes), result.describe()


def test_a_slice_unmeasured_in_both_runs_is_reported_as_outside_the_gate():
    """Not a regression — nothing changed — but not coverage either."""
    items = corpus(300)
    real_only = [i for i in items if i.label == 0][:30]

    inc = bench(items, version="v1", slice_by={"lowres": real_only})
    cha = bench(items, version="v2", slice_by={"lowres": real_only})

    assert inc.single_class_slices == cha.single_class_slices == ("lowres",)
    result = regression_gate(inc, cha)
    assert result.passed, "nothing regressed"
    assert any("outside this gate entirely" in n for n in result.notes), result.describe()


def test_a_renamed_slice_fails_rather_than_passing_quietly():
    """Re-bucketing `heavy` to `c40` removes a scope. It must be loud."""
    items = corpus(400)
    hard = [i for i in items if int(i.media_id.rsplit("m", 1)[1]) % 4 < 2]

    inc = bench(items, version="v1", slice_by={"heavy": hard})
    cha = bench(items, version="v2", slice_by={"c40": hard})

    result = regression_gate(inc, cha)
    assert not result.passed
    assert any("'heavy'" in f and "renamed" in f for f in result.failures), (
        result.describe()
    )
    assert any("'c40'" in n for n in result.notes), result.describe()


def test_a_genuine_slice_collapse_is_still_caught():
    """The Batch 2 behaviour must survive the rewrite."""
    items = corpus(600)
    hard = [i for i in items if int(i.media_id.rsplit("m", 1)[1]) % 4 < 2]
    hard_ids = {i.media_id for i in hard}

    class SliceCollapsing:
        name, version = "collapsing", "v2"

        def predict(self, items_):
            good = StubPredictor(version="good", skill=0.60).predict(items_)
            bad = StubPredictor(version="bad", skill=0.0).predict(items_)
            return np.array(
                [b if it.media_id in hard_ids else g for it, g, b in zip(items_, good, bad)]
            )

    inc = bench(items, version="v1", skill=0.30, slice_by={"hard": hard})
    cha = run_benchmark(
        SliceCollapsing(), items, threshold=THRESHOLD,
        effective_model_version="v2", slice_by={"hard": hard},
    )
    result = regression_gate(inc, cha)
    assert not result.passed
    assert any("AUC fell" in f for f in result.failures), result.describe()


def test_slice_accounting_appears_in_the_run_description():
    items = corpus(300)
    real_only = [i for i in items if i.label == 0][:30]
    run = bench(items, version="v1", slice_by={"lowres": real_only})
    text = run.describe()
    assert "NOT MEASURED" in text and "lowres" in text


def test_accounted_slice_names_covers_measured_and_unmeasured():
    items = corpus(400)
    hard = [i for i in items if int(i.media_id.rsplit("m", 1)[1]) % 4 < 2]
    real_only = [i for i in items if i.label == 0][:30]
    run = bench(items, version="v1", slice_by={"hard": hard, "lowres": real_only})
    assert run.measured_slice_names == {"hard"}
    assert run.accounted_slice_names == {"hard", "lowres"}


# --------------------------------------------------------------------------
# R3 — a run knows its provenance
# --------------------------------------------------------------------------


def test_a_run_records_the_protocol_and_the_split_lock():
    items = corpus(200, dataset="ffpp") + corpus(200, dataset="celebdf")
    lock = create_lock(items)
    selection = pick(items, DatasetRole.PRIMARY_BENCHMARK, lock)
    witness = calibrate(items, lock)
    run = bench(selection, version="v1", split_lock=lock, calibration=witness)
    assert run.protocol == FFPP_CELEBDF.name
    assert run.protocol_role == DatasetRole.PRIMARY_BENCHMARK.value
    assert run.selection_witness == selection.witness
    assert run.split_lock_hash == lock.content_hash()
    assert run.lock_generation == 0
    assert run.calibration_fingerprint == witness.witness, (
        "the calibration fingerprint must be derived from the witness, not "
        "supplied alongside it"
    )
    assert run.calibration["role"] == DatasetRole.CALIBRATION.value
    assert run.schema_version == RUN_SCHEMA_VERSION


def test_a_bare_lock_hash_is_still_accepted_without_a_protocol():
    """A caller holding only the hash should not have to rebuild a lock."""
    run = bench(corpus(200), version="v1", split_lock="sha256:abcd")
    assert run.split_lock_hash == "sha256:abcd"
    assert run.lock_generation is None
    assert run.protocol is None


def test_unrecorded_metadata_is_none_and_never_a_plausible_default():
    run = bench(corpus(200), version="v1")
    assert run.protocol is None
    assert run.split_lock_hash is None
    assert run.lock_generation is None, (
        "an unrecorded lock generation must not default to 0; that is a lie "
        "that validates"
    )
    assert "(unrecorded)" in run.describe()


def test_a_lock_generation_is_carried_through_growth():
    from evaluation.split_lock import extend_lock

    base = corpus(200)
    grown = base + corpus(60, dataset="ffpp2")
    lock = extend_lock(create_lock(base), grown)
    run = bench(grown, version="v1", split_lock=lock)
    assert run.lock_generation == 1
    assert run.split_lock_hash == lock.content_hash()


# --------------------------------------------------------------------------
# R3 — reports assemble from run artifacts alone
# --------------------------------------------------------------------------


def paired_runs(
    *, primary_version: str = "v1", in_domain_version: str = "v1"
) -> tuple[BenchmarkRun, BenchmarkRun]:
    """A compliant pair: both runs obtained through ``select()``."""
    items = corpus(300, dataset="ffpp") + corpus(300, dataset="celebdf")
    lock = create_lock(items)
    primary_sel = pick(items, DatasetRole.PRIMARY_BENCHMARK, lock)
    in_domain_sel = pick(items, DatasetRole.CALIBRATION, lock)
    hard = [i for i in primary_sel if int(i.media_id.rsplit("m", 1)[1]) % 4 < 2]
    shared = {"split_lock": lock, "calibration": calibrate(items, lock)}
    primary = bench(
        primary_sel, version=primary_version, slice_by={"hard": hard}, **shared
    )
    in_domain = bench(in_domain_sel, version=in_domain_version, **shared)
    return primary, in_domain


def test_a_report_satisfies_every_required_field():
    primary, in_domain = paired_runs()
    report = BenchmarkReport(
        primary=primary, in_domain=in_domain, protocol=FFPP_CELEBDF.name
    )
    payload = report.to_payload()
    missing = [f for f in REQUIRED_REPORT_FIELDS if payload.get(f) in (None, "")]
    assert not missing, f"report is missing {missing}"
    assert not report.validate(), report.describe()


def test_a_report_from_a_run_with_no_lock_is_incomplete():
    """Selection without a lock hash: compliant on protocol, unreproducible."""
    items = corpus(200, dataset="ffpp") + corpus(200, dataset="celebdf")
    lock = create_lock(items)

    def unlocked(role):
        return select(FFPP_CELEBDF, role, items, lock.assignments)  # no lock_hash

    report = BenchmarkReport(
        primary=bench(unlocked(DatasetRole.PRIMARY_BENCHMARK), version="v1"),
        in_domain=bench(unlocked(DatasetRole.CALIBRATION), version="v1"),
        protocol=FFPP_CELEBDF.name,
    )
    problems = report.validate()
    assert any("split_lock_hash" in p for p in problems), report.describe()


def test_a_selection_carries_its_lock_into_the_run():
    """The lock need not be passed twice; the selection already binds it."""
    items = corpus(200, dataset="ffpp") + corpus(200, dataset="celebdf")
    lock = create_lock(items)
    run = bench(pick(items, DatasetRole.PRIMARY_BENCHMARK, lock), version="v1")
    assert run.split_lock_hash == lock.content_hash()


def test_a_report_pairing_two_different_models_is_refused():
    """Quoting one model's cross-dataset number beside another's in-domain one."""
    primary, in_domain = paired_runs(primary_version="v1", in_domain_version="v2")
    report = BenchmarkReport(
        primary=primary, in_domain=in_domain, protocol=FFPP_CELEBDF.name
    )
    assert any("different models" in p for p in report.validate()), report.describe()


def test_a_report_whose_two_runs_share_a_corpus_is_refused():
    items = corpus(200, dataset="ffpp") + corpus(200, dataset="celebdf")
    lock = create_lock(items)
    selection = pick(items, DatasetRole.PRIMARY_BENCHMARK, lock)
    run = bench(selection, version="v1", split_lock=lock, calibration=calibrate(items, lock))
    report = BenchmarkReport(primary=run, in_domain=run, protocol=FFPP_CELEBDF.name)
    assert any("not cross-dataset" in p for p in report.validate()), report.describe()


def test_threshold_dependent_numbers_come_from_the_primary_run():
    """The in-domain split is the calibration set; its coverage is best-case."""
    primary, in_domain = paired_runs()
    payload = BenchmarkReport(
        primary=primary, in_domain=in_domain, protocol=FFPP_CELEBDF.name
    ).to_payload()
    assert payload["coverage"] == primary.coverage
    assert payload["selective_fpr"] == primary.selective_fpr
    assert payload["n"] == primary.n
    assert payload["in_domain"] == in_domain.roc_auc


def test_a_report_names_the_worst_slice_and_the_unmeasured_ones():
    items = corpus(300, dataset="ffpp") + corpus(300, dataset="celebdf")
    lock = create_lock(items)
    primary_sel = pick(items, DatasetRole.PRIMARY_BENCHMARK, lock)
    hard = [i for i in primary_sel if int(i.media_id.rsplit("m", 1)[1]) % 4 < 2]
    real_only = [i for i in primary_sel if i.label == 0][:30]
    shared = {"split_lock": lock, "calibration": calibrate(items, lock)}
    report = BenchmarkReport(
        primary=bench(
            primary_sel,
            version="v1",
            slice_by={"hard": hard, "lowres": real_only},
            **shared,
        ),
        in_domain=bench(
            pick(items, DatasetRole.CALIBRATION, lock), version="v1", **shared
        ),
        protocol=FFPP_CELEBDF.name,
    )
    payload = report.to_payload()
    assert payload["worst_slice"] == "hard"
    assert payload["unmeasured_slices"] == ["lowres"]
    assert "lowres" in report.describe()


# --------------------------------------------------------------------------
# R3 — history migration
# --------------------------------------------------------------------------


def legacy_row(run: BenchmarkRun) -> dict:
    """A 1.0.0 history row: the Batch 2 fields and nothing else."""
    row = json.loads(run.to_json())
    for added in (
        "schema_version",
        "protocol",
        "split_lock_hash",
        "lock_generation",
        "calibration_fingerprint",
        "empty_slices",
        "single_class_slices",
    ):
        row.pop(added, None)
    return row


def test_a_pre_batch4_history_row_still_loads():
    run = bench(corpus(200), version="v1")
    restored = BenchmarkRun.from_dict(legacy_row(run))
    assert restored.schema_version == "1.0.0"
    assert restored.roc_auc == run.roc_auc
    assert restored.protocol is None
    assert restored.empty_slices == ()


def test_a_legacy_baseline_is_not_mistaken_for_one_with_no_slices():
    """A 1.0.0 row records no slice accounting. That is silence, not absence.

    The gate must still fail on a vanished slice — the baseline measured it —
    but it cannot claim to know *why*, because the challenger's accounting is
    what names the cause and a legacy row has none.
    """
    items = corpus(400)
    hard = [i for i in items if int(i.media_id.rsplit("m", 1)[1]) % 4 < 2]
    inc = BenchmarkRun.from_dict(legacy_row(bench(items, version="v1", slice_by={"hard": hard})))
    cha = BenchmarkRun.from_dict(legacy_row(bench(items, version="v2")))

    assert inc.slices and not cha.slices
    result = regression_gate(inc, cha)
    assert not result.passed
    assert any("did not measure it at all" in f for f in result.failures), (
        result.describe()
    )


def test_a_run_round_trips_through_json():
    items = corpus(200, dataset="ffpp") + corpus(200, dataset="celebdf")
    lock = create_lock(items)
    selection = pick(items, DatasetRole.PRIMARY_BENCHMARK, lock)
    hard = [i for i in selection if int(i.media_id.rsplit("m", 1)[1]) % 4 < 2]
    real_only = [i for i in selection if i.label == 0][:20]
    run = bench(
        selection,
        version="v1",
        split_lock=lock,
        calibration=calibrate(items, lock),
        slice_by={"hard": hard, "lowres": real_only},
    )
    assert BenchmarkRun.from_dict(json.loads(run.to_json())) == run


def test_read_runs_parses_a_history_file():
    path = tmp_path("history.jsonl")
    items = corpus(200, dataset="ffpp") + corpus(200, dataset="celebdf")
    lock = create_lock(items)
    run = bench(pick(items, DatasetRole.PRIMARY_BENCHMARK, lock), version="v1")
    append_history(run, path)
    append_history(run, path)
    runs = read_runs(path)
    assert len(runs) == 2
    assert all(r.protocol == FFPP_CELEBDF.name for r in runs)


def test_a_row_from_a_newer_schema_is_refused():
    """Guessing at an unknown field is how a benchmark reads the wrong number."""
    row = json.loads(bench(corpus(200), version="v1").to_json())
    row["some_future_field"] = 1
    try:
        BenchmarkRun.from_dict(row)
    except ValueError as exc:
        assert "some_future_field" in str(exc)
        return
    raise AssertionError("a row from a newer schema was read anyway")


def test_a_row_missing_required_fields_is_refused():
    row = json.loads(bench(corpus(200), version="v1").to_json())
    del row["roc_auc"]
    try:
        BenchmarkRun.from_dict(row)
    except ValueError as exc:
        assert "roc_auc" in str(exc)
        return
    raise AssertionError("a row with no AUC was read as a run")


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
