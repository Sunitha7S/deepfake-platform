"""Dataset identity and leakage prevention.

Every check here is paired with a named double that reproduces the mistake it
exists to catch. There is no pre-Batch-2 implementation to compare against —
this framework is new — so the "legacy" implementations are the *wrong ways*
people actually split deepfake corpora, written out and asserted to fail. A
property test that has never rejected anything is a hypothesis.

The two doubles:

    naive_row_split       assigns item by item, ignoring identity entirely.
                          The classic. Puts c23 in train and c40 of the same
                          footage in test.

    source_only_split     groups by source but ignores subject. Looks careful,
                          still puts the same actor on both sides — and that
                          residual leak is worth 0.057 AUC to a model that keys
                          on the face rather than on the footage.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.dataset import (  # noqa: E402
    IdentitySplitter,
    MediaItem,
    Split,
    assess_split,
    build_components,
    class_balance,
    detect_leakage,
)


# --------------------------------------------------------------------------
# Corpora
# --------------------------------------------------------------------------


def ffpp_shaped(
    actors: int = 8, videos_per_actor: int = 2, methods: int = 4
) -> list[MediaItem]:
    """FF++ in miniature: actors → source videos → real + manipulated, ×2 quality."""
    items: list[MediaItem] = []
    for actor in range(actors):
        for video in range(videos_per_actor):
            source = f"v{actor}_{video}"
            for quality in ("c23", "c40"):
                items.append(
                    MediaItem(
                        media_id=f"{source}_real_{quality}",
                        source_id=source,
                        dataset_id="ffpp",
                        label=0,
                        subject_id=f"p{actor}",
                    )
                )
                for method in range(methods):
                    items.append(
                        MediaItem(
                            media_id=f"{source}_fake{method}_{quality}",
                            source_id=source,
                            dataset_id="ffpp",
                            label=1,
                            subject_id=f"p{actor}",
                        )
                    )
    return items


def anonymous_corpus(sources: int = 40) -> list[MediaItem]:
    """A corpus with no subject labels — source identity is all there is."""
    items: list[MediaItem] = []
    for s in range(sources):
        for variant in range(3):
            items.append(
                MediaItem(
                    media_id=f"s{s}_v{variant}",
                    source_id=f"s{s}",
                    dataset_id="wild",
                    label=variant % 2,
                )
            )
    return items


# --------------------------------------------------------------------------
# The wrong ways, written out
# --------------------------------------------------------------------------


def _bucket(key: str, salt: str) -> Split:
    """Deterministic 70/15/15 assignment from a hash. The usual random split."""
    draw = int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if draw < 0.70:
        return Split.TRAIN
    return Split.VAL if draw < 0.85 else Split.TEST


def naive_row_split(items: list[MediaItem]) -> dict[str, Split]:
    """Random split over rows. Identity is not consulted at all.

    Hash-based rather than round-robin on purpose: a round-robin aliases against
    any regular corpus ordering — over a corpus laid out as one real followed by
    four fakes, a 5-cycle sends only fakes to test — and the resulting
    single-class fold is an artefact of the double, not the mistake being
    modelled. This is what a real ``train_test_split`` does.
    """
    return {item.media_id: _bucket(item.media_id, "row") for item in items}


def source_only_split(items: list[MediaItem]) -> dict[str, Split]:
    """Groups by source. Careful-looking, and still leaks the actor."""
    return {item.media_id: _bucket(item.source_id, "src") for item in items}


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------


def test_components_merge_everything_sharing_a_subject():
    """Transitivity is the point: real, fakes, both qualities, both videos."""
    items = ffpp_shaped(actors=3, videos_per_actor=2, methods=4)
    components = build_components(items)
    assert len(components) == 3, (
        f"expected one component per actor, got {len(components)}"
    )
    assert sum(c.size for c in components) == len(items)
    # Each component must hold exactly one actor. The media id encodes it as
    # the "v<actor>_<video>" prefix.
    by_id = {i.media_id: i for i in items}
    for component in components:
        actors = {by_id[m].subject_id for m in component.media_ids}
        assert len(actors) == 1, f"component {component.key} mixes actors {actors}"
        # …and every file of that actor, across both videos and both qualities.
        assert component.size == 2 * 2 * (1 + 4)


def test_components_fall_back_to_source_without_subjects():
    items = anonymous_corpus(sources=10)
    components = build_components(items)
    assert len(components) == 10, "each source should stand alone"
    assert all(c.size == 3 for c in components)


def test_a_manipulated_item_stays_with_the_original_it_derives_from():
    items = [
        MediaItem("orig", "shot1", "ffpp", 0),
        MediaItem("swapped", "shot1", "ffpp", 1),
    ]
    components = build_components(items)
    assert len(components) == 1
    assert set(components[0].media_ids) == {"orig", "swapped"}


def test_subject_scope_controls_cross_dataset_merging():
    """Two corpora numbering their subjects from zero are not the same people."""
    items = [
        MediaItem("a", "s1", "ffpp", 0, subject_id="0"),
        MediaItem("b", "s2", "celebdf", 1, subject_id="0"),
    ]
    assert len(build_components(items, subject_scope="dataset")) == 2
    assert len(build_components(items, subject_scope="global")) == 1


def test_component_ordering_is_deterministic():
    items = ffpp_shaped(actors=5)
    assert [c.key for c in build_components(items)] == [
        c.key for c in build_components(items)
    ]


# --------------------------------------------------------------------------
# Splits produced here never leak
# --------------------------------------------------------------------------


def test_identity_split_is_leak_free():
    items = ffpp_shaped(actors=30)
    assignment = IdentitySplitter().split(items)
    report = detect_leakage(items, assignment)
    assert report.clean, report.describe()


def test_identity_split_is_reproducible():
    items = ffpp_shaped(actors=20)
    a = IdentitySplitter(seed="abc").split(items)
    b = IdentitySplitter(seed="abc").split(items)
    assert a.by_media == b.by_media


def test_a_different_seed_can_produce_a_different_split():
    items = ffpp_shaped(actors=20)
    a = IdentitySplitter(seed="one").split(items)
    b = IdentitySplitter(seed="two").split(items)
    assert a.by_media != b.by_media, "the seed had no effect"
    assert detect_leakage(items, b).clean, "a reseeded split must still be clean"


def test_split_ratios_are_approximately_honoured():
    items = ffpp_shaped(actors=60)
    assignment = IdentitySplitter().split(items)
    for split, target in ((Split.TRAIN, 0.7), (Split.VAL, 0.15), (Split.TEST, 0.15)):
        assert abs(assignment.achieved[split] - target) < 0.08, (
            f"{split.value} achieved {assignment.achieved[split]:.3f} against {target}"
        )


def test_every_item_is_assigned_exactly_once():
    items = ffpp_shaped(actors=12)
    assignment = IdentitySplitter().split(items)
    assert set(assignment.by_media) == {i.media_id for i in items}
    total = sum(len(assignment.media_in(s)) for s in Split)
    assert total == len(items)


def test_duplicate_media_ids_are_refused_at_the_door():
    items = [
        MediaItem("same", "s1", "d", 0),
        MediaItem("same", "s2", "d", 1),
    ]
    try:
        IdentitySplitter().split(items)
    except ValueError as exc:
        assert "duplicate media_id" in str(exc)
        return
    raise AssertionError("a repeated media_id was accepted")


# --------------------------------------------------------------------------
# Detection catches the wrong ways
# --------------------------------------------------------------------------


def test_detection_catches_a_naive_row_split():
    items = ffpp_shaped(actors=10)
    report = detect_leakage(items, naive_row_split(items))
    assert report, "a row-wise split was reported as clean"
    assert report.by_kind("source"), "the shared source was not reported"
    assert report.by_kind("subject"), "the shared subject was not reported"


def test_detection_catches_a_source_only_split():
    """The subtle one. Grouping by source still leaves the actor on both sides."""
    items = ffpp_shaped(actors=10, videos_per_actor=2)
    report = detect_leakage(items, source_only_split(items))
    assert report.by_kind("subject"), (
        "grouping by source alone was reported as clean; the same actor appears "
        "in two source videos and therefore in two splits"
    )
    assert not report.by_kind("source"), (
        "no source should span splits under a source-grouped assignment — if one "
        "does, this double is broken rather than the detector"
    )


def test_detection_is_clean_on_a_correct_split():
    items = ffpp_shaped(actors=15)
    assignment = IdentitySplitter().split(items)
    assert not detect_leakage(items, assignment)


def test_findings_name_the_identity_the_splits_and_the_files():
    items = [
        MediaItem("a_c23", "shot1", "ffpp", 0, subject_id="p1"),
        MediaItem("a_c40", "shot1", "ffpp", 0, subject_id="p1"),
    ]
    report = detect_leakage(items, {"a_c23": Split.TRAIN, "a_c40": Split.TEST})
    text = report.describe()
    assert "shot1" in text and "p1" in text
    assert "train" in text and "test" in text
    assert "a_c23" in text or "a_c40" in text


def test_report_truthiness_reads_as_is_there_leakage():
    """``if detect_leakage(...)`` must mean 'if there is leakage'."""
    items = ffpp_shaped(actors=8)
    assert not detect_leakage(items, IdentitySplitter().split(items))
    assert detect_leakage(items, naive_row_split(items))


def test_unassigned_items_are_not_treated_as_leakage():
    items = ffpp_shaped(actors=4)
    partial = {items[0].media_id: Split.TRAIN}
    assert not detect_leakage(items, partial), (
        "items with no split assignment are out of scope, not leaks"
    )


# --------------------------------------------------------------------------
# A clean split can still be unusable
# --------------------------------------------------------------------------


def test_a_tiny_corpus_produces_an_empty_fold_and_says_so():
    """Three actors cannot make three balanced folds. That must be loud."""
    items = ffpp_shaped(actors=3)
    assignment = IdentitySplitter().split(items)
    assert detect_leakage(items, assignment).clean, "should still be leak-free"
    viability = assess_split(items, assignment)
    assert not viability.usable
    assert any("empty" in p for p in viability.problems), viability.describe()


def test_a_large_corpus_is_usable():
    items = ffpp_shaped(actors=80)
    assignment = IdentitySplitter().split(items)
    viability = assess_split(items, assignment)
    assert viability.usable, viability.describe()


def test_a_single_class_fold_is_reported_as_a_problem():
    items = [MediaItem(f"m{i}", f"s{i}", "d", 1) for i in range(30)]
    items += [MediaItem(f"r{i}", f"t{i}", "d", 0) for i in range(30)]
    assignment = IdentitySplitter().split(items)
    # Force a degenerate fold by hand to exercise the check.
    forced = dict(assignment.by_media)
    for item in items:
        forced[item.media_id] = Split.TRAIN if item.label == 1 else Split.TEST
    from evaluation.dataset import SplitAssignment

    degenerate = SplitAssignment(
        by_media=forced,
        components=assignment.components,
        achieved={Split.TRAIN: 0.5, Split.VAL: 0.0, Split.TEST: 0.5},
        counts={Split.TRAIN: 30, Split.VAL: 0, Split.TEST: 30},
        seed="x",
        ratios=assignment.ratios,
    )
    viability = assess_split(items, degenerate)
    assert any("only" in p for p in viability.problems), viability.describe()


def test_class_balance_counts_both_classes_per_split():
    items = ffpp_shaped(actors=20)
    assignment = IdentitySplitter().split(items)
    balance = class_balance(items, assignment)
    assert sum(b["total"] for b in balance.values()) == len(items)
    for split, counts in balance.items():
        if counts["total"]:
            assert counts["real"] + counts["fake"] == counts["total"]


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
