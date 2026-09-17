"""The cross-dataset protocol, and the slice metadata it reports against.

Two subjects in one suite because they are one idea: the protocol says *which*
corpus a number came from, and the slice metadata says *which part of it*. Both
exist so that a reported AUC can be traced back to a population, and a test that
only checked one of them would leave the other free to lie.

As in the dataset suite, every rule is paired with the wrong protocol it exists
to reject. A validator that has never rejected anything is a hypothesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.dataset import (  # noqa: E402
    Compression,
    MediaItem,
    ResolutionBucket,
    Split,
    build_components,
    group_by_slice,
    slice_counts,
)
from evaluation.protocol import (  # noqa: E402
    FFPP_CELEBDF,
    REQUIRED_REPORT_FIELDS,
    DatasetRole,
    EvaluationProtocol,
    RoleAssignment,
    check_report,
    select,
    validate_protocol,
)


def roles(**overrides) -> EvaluationProtocol:
    """A valid protocol, with named parts replaceable to build the broken ones."""
    base = {
        DatasetRole.TRAIN: ("ffpp", (Split.TRAIN,)),
        DatasetRole.VALIDATION: ("ffpp", (Split.VAL,)),
        DatasetRole.CALIBRATION: ("ffpp", (Split.TEST,)),
        DatasetRole.PRIMARY_BENCHMARK: ("celebdf", tuple(Split)),
    }
    base.update({DatasetRole[k.upper()]: v for k, v in overrides.items()})
    return EvaluationProtocol(
        name="test",
        assignments=tuple(
            RoleAssignment(dataset_id=d, role=r, splits=s) for r, (d, s) in base.items()
        ),
    )


# --------------------------------------------------------------------------
# The shipped protocol
# --------------------------------------------------------------------------


def test_the_shipped_protocol_is_valid():
    report = validate_protocol(FFPP_CELEBDF)
    assert report.valid, report.describe()
    assert not report.warnings, report.describe()


def test_the_shipped_protocol_benchmarks_a_corpus_it_never_fits_on():
    fitted = FFPP_CELEBDF.dataset_ids(
        DatasetRole.TRAIN, DatasetRole.VALIDATION, DatasetRole.CALIBRATION
    )
    benchmarked = FFPP_CELEBDF.dataset_ids(
        DatasetRole.PRIMARY_BENCHMARK, DatasetRole.SECONDARY_BENCHMARK
    )
    assert not (fitted & benchmarked)


def test_the_shipped_protocol_calibrates_off_the_checkpoint_selection_split():
    """Calibration must not reuse the split that chose the checkpoint."""
    cal = FFPP_CELEBDF.one(DatasetRole.CALIBRATION)
    val = FFPP_CELEBDF.one(DatasetRole.VALIDATION)
    assert not (set(cal.splits) & set(val.splits))


def test_report_truthiness_reads_as_is_it_valid():
    assert validate_protocol(FFPP_CELEBDF)
    assert not validate_protocol(roles(primary_benchmark=("ffpp", tuple(Split))))


# --------------------------------------------------------------------------
# The protocols that must be rejected
# --------------------------------------------------------------------------


def test_benchmarking_the_training_corpus_is_rejected():
    report = validate_protocol(roles(primary_benchmark=("ffpp", tuple(Split))))
    assert not report.valid
    assert any("different corpus" in p for p in report.problems), report.describe()


def test_calibrating_on_the_benchmark_is_rejected():
    """The common one: fit a temperature on Celeb-DF, then report Celeb-DF."""
    report = validate_protocol(roles(calibration=("celebdf", (Split.VAL,))))
    assert not report.valid
    assert any("contribute to nothing" in p for p in report.problems), report.describe()


def test_calibrating_on_the_training_split_is_rejected():
    report = validate_protocol(roles(calibration=("ffpp", (Split.TRAIN,))))
    assert not report.valid
    assert any("saturated" in p for p in report.problems), report.describe()


def test_validating_on_the_training_split_is_rejected():
    report = validate_protocol(roles(validation=("ffpp", (Split.TRAIN,))))
    assert not report.valid
    assert any("overfitted checkpoint" in p for p in report.problems), report.describe()


def test_a_missing_role_is_rejected():
    partial = EvaluationProtocol(
        name="partial",
        assignments=(
            RoleAssignment("ffpp", DatasetRole.TRAIN, (Split.TRAIN,)),
            RoleAssignment("celebdf", DatasetRole.PRIMARY_BENCHMARK, tuple(Split)),
        ),
    )
    report = validate_protocol(partial)
    assert not report.valid
    assert any("validation" in p for p in report.problems)
    assert any("calibration" in p for p in report.problems)


def test_two_primary_benchmarks_are_rejected():
    doubled = EvaluationProtocol(
        name="two-headlines",
        assignments=roles().assignments
        + (RoleAssignment("dfdc", DatasetRole.PRIMARY_BENCHMARK, tuple(Split)),),
    )
    report = validate_protocol(doubled)
    assert not report.valid
    assert any("exactly one is allowed" in p for p in report.problems)


def test_sharing_a_split_between_calibration_and_validation_warns():
    report = validate_protocol(roles(calibration=("ffpp", (Split.VAL,))))
    assert report.valid, "this is a warning, not a rejection"
    assert any("optimistic" in w for w in report.warnings), report.describe()


def test_a_partially_used_primary_benchmark_warns():
    report = validate_protocol(roles(primary_benchmark=("celebdf", (Split.TEST,))))
    assert report.valid
    assert any("held-out corpus" in w for w in report.warnings), report.describe()


def test_a_role_naming_no_splits_is_refused_at_construction():
    try:
        RoleAssignment("ffpp", DatasetRole.TRAIN, ())
    except ValueError as exc:
        assert "names no splits" in str(exc)
        return
    raise AssertionError("a role using nothing was accepted")


# --------------------------------------------------------------------------
# Future datasets
# --------------------------------------------------------------------------


def test_a_future_dataset_can_be_added_as_a_secondary_benchmark():
    extended = FFPP_CELEBDF.with_secondary("dfd-2026", note="quarterly holdout")
    assert validate_protocol(extended).valid
    assert "dfd-2026" in extended.dataset_ids(DatasetRole.SECONDARY_BENCHMARK)


def test_adding_a_dataset_that_is_already_fitted_on_is_rejected():
    """The check has to survive extension, not only initial construction."""
    report = validate_protocol(FFPP_CELEBDF.with_secondary("ffpp"))
    assert not report.valid
    assert any("ffpp" in p for p in report.problems)


def test_extension_leaves_the_original_untouched():
    before = len(FFPP_CELEBDF.assignments)
    FFPP_CELEBDF.with_secondary("dfd-2026")
    assert len(FFPP_CELEBDF.assignments) == before


# --------------------------------------------------------------------------
# Selecting the items for a role
# --------------------------------------------------------------------------


def mixed_corpus() -> list[MediaItem]:
    items: list[MediaItem] = []
    for dataset in ("ffpp", "celebdf"):
        for i in range(9):
            items.append(
                MediaItem(f"{dataset}_{i}", f"src{i}", dataset, i % 2, subject_id=f"p{i}")
            )
    return items


def assignment_for(items: list[MediaItem]) -> dict[str, Split]:
    order = (Split.TRAIN, Split.VAL, Split.TEST)
    return {item.media_id: order[i % 3] for i, item in enumerate(items)}


def test_select_returns_only_the_right_dataset_and_split():
    items = mixed_corpus()
    assignment = assignment_for(items)
    chosen = select(FFPP_CELEBDF, DatasetRole.TRAIN, items, assignment)
    assert chosen, "the training role selected nothing"
    for item in chosen:
        assert item.dataset_id == "ffpp"
        assert assignment[item.media_id] is Split.TRAIN


def test_select_takes_the_whole_benchmark_corpus():
    items = mixed_corpus()
    chosen = select(
        FFPP_CELEBDF, DatasetRole.PRIMARY_BENCHMARK, items, assignment_for(items)
    )
    assert {i.media_id for i in chosen} == {
        i.media_id for i in items if i.dataset_id == "celebdf"
    }


def test_select_never_returns_the_same_item_to_train_and_to_the_benchmark():
    items = mixed_corpus()
    assignment = assignment_for(items)
    train = {i.media_id for i in select(FFPP_CELEBDF, DatasetRole.TRAIN, items, assignment)}
    bench = {
        i.media_id
        for i in select(FFPP_CELEBDF, DatasetRole.PRIMARY_BENCHMARK, items, assignment)
    }
    assert not (train & bench)


def test_unassigned_items_are_excluded_rather_than_defaulted():
    items = mixed_corpus()
    assignment = assignment_for(items)
    orphan = MediaItem("orphan", "srcX", "ffpp", 1)
    chosen = select(FFPP_CELEBDF, DatasetRole.TRAIN, items + [orphan], assignment)
    assert "orphan" not in {i.media_id for i in chosen}


# --------------------------------------------------------------------------
# Reporting conventions
# --------------------------------------------------------------------------


def complete_report() -> dict:
    return {f: "x" for f in REQUIRED_REPORT_FIELDS} | {"n": 1200, "coverage": 0.87}


def test_a_complete_report_passes():
    assert check_report(complete_report()) == []


def test_a_cross_dataset_number_without_the_in_domain_one_is_incomplete():
    payload = complete_report()
    del payload["in_domain"]
    assert "in_domain" in check_report(payload)


def test_an_auc_without_coverage_is_incomplete():
    """A three-state verdict has no accuracy; coverage is half the number."""
    payload = complete_report()
    payload["coverage"] = None
    assert "coverage" in check_report(payload)


def test_a_report_without_the_split_lock_is_incomplete():
    payload = complete_report()
    del payload["split_lock_hash"]
    assert "split_lock_hash" in check_report(payload)


def test_the_bare_weights_version_is_not_enough():
    payload = complete_report()
    payload["effective_model_version"] = ""
    assert "effective_model_version" in check_report(payload)


# --------------------------------------------------------------------------
# Slice metadata
# --------------------------------------------------------------------------


def test_slice_fields_are_optional_and_positional_construction_still_works():
    """Backward compatibility, asserted rather than assumed."""
    item = MediaItem("m1", "s1", "ffpp", 0)
    assert item.compression is None
    assert item.resolution is None
    assert item.generator is None
    assert item.slice_values() == {"dataset": "ffpp"}


def test_strings_are_coerced_to_their_enums():
    """A CSV gives strings; a field holding "c40" would slice as its own regime."""
    item = MediaItem("m1", "s1", "ffpp", 1, compression="heavy", resolution="hd")
    assert item.compression is Compression.HEAVY
    assert item.resolution is ResolutionBucket.HD


def test_an_unknown_enum_value_is_refused_with_the_alternatives():
    try:
        MediaItem("m1", "s1", "ffpp", 1, compression="c40")
    except ValueError as exc:
        assert "c40" in str(exc) and "heavy" in str(exc)
        return
    raise AssertionError("a corpus-native quality label was accepted as normalised")


def test_ffpp_qualities_map_onto_the_normalised_scale():
    assert Compression.from_ffpp("raw") is Compression.NONE
    assert Compression.from_ffpp("c23") is Compression.LIGHT
    assert Compression.from_ffpp("C40") is Compression.HEAVY
    assert Compression.from_crf(23) is Compression.LIGHT
    assert Compression.from_crf(40) is Compression.HEAVY


def test_resolution_buckets_key_on_the_short_side():
    """2560x1080 has no more facial detail than 1920x1080, and must not rank higher."""
    assert ResolutionBucket.from_frame(1920, 1080) is ResolutionBucket.FHD
    assert ResolutionBucket.from_frame(2560, 1080) is ResolutionBucket.FHD
    assert ResolutionBucket.from_frame(640, 360) is ResolutionBucket.LOW
    assert ResolutionBucket.from_frame(3840, 2160) is ResolutionBucket.UHD


def test_an_empty_generator_is_refused():
    try:
        MediaItem("m1", "s1", "ffpp", 1, generator="   ")
    except ValueError as exc:
        assert "non-empty" in str(exc)
        return
    raise AssertionError("a blank generator was accepted and would form a nameless slice")


def test_serialisation_omits_unset_fields_so_old_indices_fingerprint_the_same():
    plain = MediaItem("m1", "s1", "ffpp", 0)
    assert plain.to_dict() == {
        "media_id": "m1",
        "source_id": "s1",
        "dataset_id": "ffpp",
        "label": 0,
    }


def test_serialisation_round_trips_every_field():
    item = MediaItem(
        "m1",
        "s1",
        "ffpp",
        1,
        subject_id="p1",
        compression=Compression.HEAVY,
        resolution=ResolutionBucket.LOW,
        generator="FaceSwap",
    )
    assert MediaItem.from_dict(item.to_dict()) == item


def test_a_misspelled_field_is_refused_rather_than_dropped():
    """A dropped field slices into nothing and takes its dimension off the gate."""
    payload = {
        "media_id": "m1",
        "source_id": "s1",
        "dataset_id": "ffpp",
        "label": 1,
        "resolution_bucket": "hd",
    }
    try:
        MediaItem.from_dict(payload)
    except ValueError as exc:
        assert "resolution_bucket" in str(exc)
        return
    raise AssertionError("an unknown field was silently ignored")


def test_slice_metadata_does_not_affect_identity_components():
    """Compression is a slice, not an identity. c23 and c40 stay together."""
    items = [
        MediaItem("a", "shot1", "ffpp", 1, subject_id="p1", compression=Compression.LIGHT),
        MediaItem("b", "shot1", "ffpp", 1, subject_id="p1", compression=Compression.HEAVY),
    ]
    assert len(build_components(items)) == 1


def test_grouping_by_a_dimension_buckets_and_sorts():
    items = [
        MediaItem(f"m{i}", f"s{i}", "ffpp", i % 2, compression=c)
        for i, c in enumerate([Compression.HEAVY, Compression.LIGHT, Compression.HEAVY])
    ]
    grouped = group_by_slice(items, "compression")
    assert list(grouped) == ["heavy", "light"], "slice keys must come back sorted"
    assert len(grouped["heavy"]) == 2


def test_unrecorded_items_are_not_a_slice_by_default():
    items = [
        MediaItem("m0", "s0", "ffpp", 0, compression=Compression.LIGHT),
        MediaItem("m1", "s1", "ffpp", 1),
    ]
    assert set(group_by_slice(items, "compression")) == {"light"}
    assert set(group_by_slice(items, "compression", include_unrecorded=True)) == {
        "light",
        "unrecorded",
    }


def test_slice_counts_always_show_the_unrecorded_remainder():
    """A dimension recorded on 4% of the corpus must be visibly useless."""
    items = [MediaItem(f"m{i}", f"s{i}", "ffpp", i % 2) for i in range(24)]
    items.append(MediaItem("rec", "sx", "ffpp", 1, generator="sdxl-1.0"))
    counts = slice_counts(items, "generator")
    assert counts["unrecorded"]["total"] == 24
    assert counts["sdxl-1.0"]["total"] == 1


def test_dataset_is_available_as_a_slice_dimension():
    items = mixed_corpus()
    grouped = group_by_slice(items, "dataset")
    assert set(grouped) == {"celebdf", "ffpp"}


def test_an_unknown_slice_dimension_is_refused():
    try:
        MediaItem("m", "s", "ffpp", 0).slice_value("codec")
    except ValueError as exc:
        assert "codec" in str(exc)
        return
    raise AssertionError("an unknown slice dimension returned a value")


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
