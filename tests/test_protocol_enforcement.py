"""R5: a run must show it followed the protocol it names, not merely name one.

THE DEFECT
----------
``run_benchmark(predictor, items, protocol=...)`` took the items and the
protocol as unrelated arguments. Nothing connected them, so a run could score
the **training set** and record ``protocol="ffpp-train/celebdf-primary/v1"``
beside it, and every downstream consumer — the report, the history, the
regression gate — would treat that as a cross-dataset benchmark.

``legacy_run_benchmark`` below reproduces that signature exactly.
``test_the_legacy_path_would_label_the_training_set_a_benchmark`` builds the
false claim with it and asserts it succeeds, because a test that only shows the
new code working proves nothing about what the new code prevents.

WHAT ENFORCEMENT CAN AND CANNOT BE
----------------------------------
Not unforgeability. A caller inside the process can compute a witness and
construct anything; claiming otherwise would be exactly the kind of false
assurance this finding is about.

What it is: the recorded protocol now comes **only** from a ``RoleSelection``,
the witness is re-derived from the items actually scored, and
``verify_run_selection`` re-derives the whole claim from artifacts alone — the
protocol, the corpus index and the lock — so an auditor who was not there can
check it. Accidental false claims are impossible; deliberate ones are
detectable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.common import ThresholdInfo  # noqa: E402
from evaluation.benchmark import (  # noqa: E402
    BenchmarkRun,
    StubPredictor,
    dataset_fingerprint,
    run_benchmark,
)
from evaluation.dataset import MediaItem  # noqa: E402
from evaluation.metrics import average_precision, roc_auc  # noqa: E402
from evaluation.protocol import (  # noqa: E402
    FFPP_CELEBDF,
    DatasetRole,
    EvaluationProtocol,
    ProtocolViolation,
    RoleAssignment,
    RoleSelection,
    Split,
    select,
    verify_run_selection,
)
from evaluation.risk import operating_point  # noqa: E402
from evaluation.split_lock import create_lock  # noqa: E402

THRESHOLD = ThresholdInfo.at_margin(value=0.5, margin=0.2007, target_fpr=0.02)


def mixed_corpus(n: int = 200) -> list[MediaItem]:
    items: list[MediaItem] = []
    for dataset in ("ffpp", "celebdf"):
        for i in range(n):
            items.append(
                MediaItem(
                    f"{dataset}_m{i}", f"{dataset}_s{i}", dataset, i % 2,
                    subject_id=f"{dataset}_p{i}",
                )
            )
    return items


def setup():
    items = mixed_corpus()
    lock = create_lock(items)
    return items, lock


def pick(items, lock, role, protocol=FFPP_CELEBDF):
    return select(
        protocol, role, items, lock.assignments, lock_hash=lock.content_hash()
    )


# --------------------------------------------------------------------------
# The legacy path, reproduced
# --------------------------------------------------------------------------


def legacy_run_benchmark(predictor, items, *, threshold, effective_model_version, protocol):
    """The pre-Batch-5 signature: items and protocol, unrelated.

    Trimmed to what the demonstration needs — no slices, no lock — but the
    coupling (or absence of it) is exactly as it was.
    """
    y = np.array([i.label for i in items], dtype=np.float64)
    scores = np.asarray(predictor.predict(items), dtype=np.float64)
    point = operating_point(y, scores, threshold)
    return BenchmarkRun(
        predictor_name=predictor.name,
        effective_model_version=effective_model_version,
        dataset_fingerprint=dataset_fingerprint(items),
        n=len(items),
        roc_auc=roc_auc(y, scores),
        pr_auc=average_precision(y, scores),
        threshold=threshold.value,
        band_width=threshold.band_width,
        coverage=point.coverage,
        selective_risk=point.selective_risk,
        selective_fpr=point.selective_fpr,
        selective_fnr=point.selective_fnr,
        protocol=getattr(protocol, "name", protocol),
    )


def test_the_legacy_path_would_label_the_training_set_a_benchmark():
    """The premise, and it is worth reading twice: this succeeds."""
    items, lock = setup()
    training = pick(items, lock, DatasetRole.TRAIN).as_list()
    assert training, "fixture: the training role must select something"

    run = legacy_run_benchmark(
        StubPredictor(version="v1"),
        training,
        threshold=THRESHOLD,
        effective_model_version="v1",
        protocol=FFPP_CELEBDF,
    )
    assert run.protocol == FFPP_CELEBDF.name, (
        "the legacy path no longer accepts an unrelated protocol; premise broken"
    )
    assert all(i.dataset_id == "ffpp" for i in training), (
        "these are training items, recorded under a protocol whose benchmark is "
        "celebdf — and nothing objected"
    )
    assert run.selection_witness is None, "the legacy run carries no evidence"


# --------------------------------------------------------------------------
# The compliant path
# --------------------------------------------------------------------------


def test_the_compliant_path_succeeds_and_records_its_evidence():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    run = run_benchmark(
        StubPredictor(version="v1"),
        selection,
        threshold=THRESHOLD,
        effective_model_version="v1",
    )
    assert run.protocol == FFPP_CELEBDF.name
    assert run.protocol_role == DatasetRole.PRIMARY_BENCHMARK.value
    assert run.selection_witness == selection.witness
    assert run.split_lock_hash == lock.content_hash()


def test_the_recorded_protocol_is_verifiable_from_artifacts_alone():
    """The property that makes the witness worth carrying."""
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    run = run_benchmark(
        StubPredictor(version="v1"), selection,
        threshold=THRESHOLD, effective_model_version="v1",
    )
    problems = verify_run_selection(
        run, FFPP_CELEBDF, DatasetRole.PRIMARY_BENCHMARK, items,
        lock.assignments, lock_hash=lock.content_hash(),
    )
    assert not problems, problems


def test_a_selection_is_list_like_so_existing_callers_keep_working():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    assert len(selection) == len(selection.as_list())
    assert all(i.dataset_id == "celebdf" for i in selection)
    assert selection[0] in selection.as_list()
    assert bool(selection)


# --------------------------------------------------------------------------
# The bypass path is rejected
# --------------------------------------------------------------------------


def test_a_protocol_beside_a_hand_assembled_list_is_refused():
    """The exact call the legacy path allowed."""
    items, lock = setup()
    training = pick(items, lock, DatasetRole.TRAIN).as_list()
    try:
        run_benchmark(
            StubPredictor(version="v1"), training,
            threshold=THRESHOLD, effective_model_version="v1", protocol=FFPP_CELEBDF,
        )
    except ProtocolViolation as exc:
        assert "Recording is not enforcement" in str(exc)
        return
    raise AssertionError("the training set was labelled a benchmark")


def test_a_bare_protocol_string_is_refused_too():
    """Closing the obvious way around the type check."""
    items, _ = setup()
    try:
        run_benchmark(
            StubPredictor(version="v1"), items,
            threshold=THRESHOLD, effective_model_version="v1", protocol="anything/v1",
        )
    except ProtocolViolation:
        return
    raise AssertionError("a protocol name string was recorded without evidence")


def test_passing_both_a_selection_and_a_protocol_is_refused():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    try:
        run_benchmark(
            StubPredictor(version="v1"), selection,
            threshold=THRESHOLD, effective_model_version="v1", protocol=FFPP_CELEBDF,
        )
    except ProtocolViolation as exc:
        assert "not both" in str(exc)
        return
    raise AssertionError("an ambiguous protocol claim was accepted")


def test_an_unprotocolled_run_is_still_allowed():
    """Not every evaluation is a protocol run; those simply record no protocol."""
    items, _ = setup()
    run = run_benchmark(
        StubPredictor(version="v1"), items,
        threshold=THRESHOLD, effective_model_version="v1",
    )
    assert run.protocol is None and run.selection_witness is None


# --------------------------------------------------------------------------
# Forged metadata is rejected
# --------------------------------------------------------------------------


def test_a_selection_with_a_forged_witness_is_refused_at_construction():
    """The realistic failure — a hand-built selection — cannot even be made."""
    try:
        RoleSelection(
            protocol_name=FFPP_CELEBDF.name,
            role=DatasetRole.PRIMARY_BENCHMARK,
            media_ids=("a", "b"),
            lock_hash=None,
            witness="deadbeef",
        )
    except ProtocolViolation as exc:
        assert "assembled by hand" in str(exc)
        return
    raise AssertionError("a selection with an invented witness was constructed")


def test_a_selection_whose_items_contradict_its_ids_is_refused():
    items, lock = setup()
    good = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    try:
        RoleSelection(
            protocol_name=good.protocol_name,
            role=good.role,
            media_ids=good.media_ids,
            lock_hash=good.lock_hash,
            witness=good.witness,
            items=tuple(items[:3]),  # not the attested set
        )
    except ProtocolViolation as exc:
        assert "do not match the media ids" in str(exc)
        return
    raise AssertionError("a selection attesting to items it does not hold was built")


def test_a_run_whose_witness_names_a_different_corpus_fails_verification():
    """A deliberately forged but self-consistent claim, caught after the fact."""
    items, lock = setup()
    training = pick(items, lock, DatasetRole.TRAIN)
    # Self-consistent, and a lie: training items attested as the benchmark role.
    forged_ids = training.media_ids
    forged = RoleSelection(
        protocol_name=FFPP_CELEBDF.name,
        role=DatasetRole.PRIMARY_BENCHMARK,
        media_ids=forged_ids,
        lock_hash=lock.content_hash(),
        witness=RoleSelection.compute_witness(
            FFPP_CELEBDF.name, DatasetRole.PRIMARY_BENCHMARK, forged_ids,
            lock.content_hash(),
        ),
        items=training.items,
    )
    run = run_benchmark(
        StubPredictor(version="v1"), forged,
        threshold=THRESHOLD, effective_model_version="v1",
    )
    assert run.protocol == FFPP_CELEBDF.name, "the forgery is accepted in-process"

    problems = verify_run_selection(
        run, FFPP_CELEBDF, DatasetRole.PRIMARY_BENCHMARK, items,
        lock.assignments, lock_hash=lock.content_hash(),
    )
    assert problems, "the forged run passed artifact verification"
    assert any("different set of items" in p for p in problems), problems


def test_altering_the_items_after_selection_is_caught():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    tampered = RoleSelection(
        protocol_name=selection.protocol_name,
        role=selection.role,
        media_ids=selection.media_ids,
        lock_hash=selection.lock_hash,
        witness=selection.witness,
    )
    # An items-free selection passes construction; run_benchmark refuses it as
    # empty rather than scoring an unattested set.
    try:
        run_benchmark(
            StubPredictor(version="v1"), tampered,
            threshold=THRESHOLD, effective_model_version="v1",
        )
    except ValueError as exc:
        assert "empty" in str(exc)
        return
    raise AssertionError("a selection carrying no items produced a run")


def test_a_lock_disagreement_between_selection_and_run_is_refused():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    try:
        run_benchmark(
            StubPredictor(version="v1"), selection,
            threshold=THRESHOLD, effective_model_version="v1",
            split_lock="sha256:" + "0" * 64,
        )
    except ProtocolViolation as exc:
        assert "one of the two is wrong" in str(exc)
        return
    raise AssertionError("a run recorded a lock its selection contradicts")


# --------------------------------------------------------------------------
# An invalid protocol cannot select
# --------------------------------------------------------------------------


def test_an_invalid_protocol_selects_nothing():
    """Validation at the point of use, not only at the point of writing."""
    items, lock = setup()
    self_benchmarking = EvaluationProtocol(
        name="bad/v1",
        assignments=(
            RoleAssignment("ffpp", DatasetRole.TRAIN, (Split.TRAIN,)),
            RoleAssignment("ffpp", DatasetRole.VALIDATION, (Split.VAL,)),
            RoleAssignment("ffpp", DatasetRole.CALIBRATION, (Split.TEST,)),
            RoleAssignment("ffpp", DatasetRole.PRIMARY_BENCHMARK, tuple(Split)),
        ),
    )
    try:
        select(
            self_benchmarking, DatasetRole.PRIMARY_BENCHMARK, items, lock.assignments
        )
    except ProtocolViolation as exc:
        assert "is invalid and cannot select" in str(exc)
        return
    raise AssertionError("an invalid protocol produced a selection")


# --------------------------------------------------------------------------
# Historical reports still load
# --------------------------------------------------------------------------


def test_a_historical_run_without_a_witness_still_loads():
    """Backward compatibility: enforcement is forward-looking, not retroactive."""
    items, lock = setup()
    legacy = legacy_run_benchmark(
        StubPredictor(version="v1"),
        pick(items, lock, DatasetRole.TRAIN).as_list(),
        threshold=THRESHOLD, effective_model_version="v1", protocol=FFPP_CELEBDF,
    )
    import json

    restored = BenchmarkRun.from_dict(json.loads(legacy.to_json()))
    assert restored.protocol == FFPP_CELEBDF.name
    assert restored.selection_witness is None


def test_verification_says_unverifiable_rather_than_verified_for_a_historical_run():
    """The distinction that matters: no evidence is not the same as evidence."""
    items, lock = setup()
    legacy = legacy_run_benchmark(
        StubPredictor(version="v1"),
        pick(items, lock, DatasetRole.TRAIN).as_list(),
        threshold=THRESHOLD, effective_model_version="v1", protocol=FFPP_CELEBDF,
    )
    problems = verify_run_selection(
        legacy, FFPP_CELEBDF, DatasetRole.PRIMARY_BENCHMARK, items, lock.assignments
    )
    assert any("no selection witness" in p for p in problems), problems


def test_verification_reports_a_protocol_name_mismatch():
    items, lock = setup()
    other = FFPP_CELEBDF.with_secondary("dfd-2026")
    selection = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK, protocol=other)
    run = run_benchmark(
        StubPredictor(version="v1"), selection,
        threshold=THRESHOLD, effective_model_version="v1",
    )
    # `with_secondary` keeps the name, so force a genuine mismatch by verifying
    # against a protocol with a different name.
    renamed = EvaluationProtocol(name="other/v9", assignments=FFPP_CELEBDF.assignments)
    problems = verify_run_selection(
        run, renamed, DatasetRole.PRIMARY_BENCHMARK, items,
        lock.assignments, lock_hash=lock.content_hash(),
    )
    assert any("claims protocol" in p for p in problems), problems


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
