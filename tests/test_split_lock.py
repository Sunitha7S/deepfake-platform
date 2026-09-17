"""The split lock: freeze a benchmark and prove it stayed frozen.

The test that gives this module its reason to exist is
``test_growth_moves_items_when_the_split_is_regenerated`` — it asserts the
*failure*. Without a demonstration that an unlocked split really does reassign
items under growth, every other test here is checking a property nobody needed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.dataset import (  # noqa: E402
    IdentitySplitter,
    MediaItem,
    Split,
    detect_leakage,
)
from evaluation.split_lock import (  # noqa: E402
    LOCK_VERSION,
    MIGRATIONS,
    LockCorrupted,
    LockHashMismatch,
    LockMergeConflict,
    LockVersionMismatch,
    SplitLock,
    SplitLockError,
    apply_migrations,
    check_regeneration,
    create_lock,
    extend_lock,
    verify_lock,
)


# --------------------------------------------------------------------------
# Corpora
# --------------------------------------------------------------------------


def corpus(actors: int = 30, *, first: int = 0, methods: int = 4) -> list[MediaItem]:
    """FF++ in miniature. ``first`` lets a caller build a disjoint extension."""
    items: list[MediaItem] = []
    for actor in range(first, first + actors):
        for video in range(2):
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


def tmp_path(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix="dfp-lock-")) / name


# --------------------------------------------------------------------------
# The failure the lock exists to prevent
# --------------------------------------------------------------------------


def test_growth_moves_items_when_the_split_is_regenerated():
    """Unlocked, a grown corpus reassigns items that were already placed.

    This is the whole justification for the module. If this ever stops failing,
    the splitter has become stable under growth and the lock is redundant.
    """
    base = corpus(actors=20)
    grown = base + corpus(actors=20, first=20)

    before = IdentitySplitter().split(base).by_media
    after = IdentitySplitter().split(grown).by_media

    moved = [m for m, s in before.items() if after[m] is not s]
    assert moved, (
        "growth did not move a single item; the greedy splitter has become "
        "stable and this test's premise no longer holds"
    )


def test_the_lock_does_not_move_them():
    base = corpus(actors=20)
    grown = base + corpus(actors=20, first=20)

    lock = create_lock(base)
    extended = extend_lock(lock, grown)

    for media_id, split in lock.assignments.items():
        assert extended.assignments[media_id] is split, (
            f"{media_id} moved from {split.value} to "
            f"{extended.assignments[media_id].value} under growth"
        )


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_identical_corpus_produces_an_identical_lock():
    items = corpus(actors=25)
    a, b = create_lock(items), create_lock(items)
    assert a.to_json() == b.to_json(), "two locks over one corpus differ byte for byte"
    assert a.content_hash() == b.content_hash()


def test_corpus_ordering_does_not_change_the_lock():
    """A different index ordering is not a different corpus."""
    items = corpus(actors=18)
    shuffled = list(reversed(items))
    assert create_lock(items).to_json() == create_lock(shuffled).to_json()


def test_a_different_seed_gives_a_different_lock():
    items = corpus(actors=20)
    a = create_lock(items, seed="one")
    b = create_lock(items, seed="two")
    assert a.assignments != b.assignments, "the seed had no effect on the lock"


def test_the_lock_carries_no_timestamp():
    """A timestamp would make the determinism test above unwritable."""
    text = create_lock(corpus(actors=6)).to_json()
    lowered = text.lower()
    for word in ("created", "timestamp", "generated_at", "date"):
        assert word not in lowered, f"the lock body contains {word!r}"


def test_round_trip_through_json_preserves_everything():
    lock = create_lock(corpus(actors=15), seed="rt")
    back = SplitLock.from_json(lock.to_json())
    assert back.assignments == lock.assignments
    assert back.seed == lock.seed
    assert back.ratios == lock.ratios
    assert back.generation == lock.generation
    assert back.content_hash() == lock.content_hash()


def test_save_and_load_round_trip():
    path = tmp_path("split.lock.json")
    lock = create_lock(corpus(actors=10))
    lock.save(path)
    assert SplitLock.load(path).content_hash() == lock.content_hash()


# --------------------------------------------------------------------------
# Growth
# --------------------------------------------------------------------------


def test_growth_assigns_the_new_items():
    base = corpus(actors=15)
    grown = base + corpus(actors=10, first=15)
    extended = extend_lock(create_lock(base), grown)
    assert set(extended.assignments) == {i.media_id for i in grown}


def test_a_new_file_of_a_locked_identity_joins_that_identity_split():
    """The rule that keeps a grown corpus leak-free.

    A newly encoded copy of footage already in test must go to test — not to
    whichever fold is emptiest, which is what a quota-driven placement would do
    and which would put the same source on both sides.
    """
    base = corpus(actors=12)
    lock = create_lock(base)
    target = base[0]
    newcomer = MediaItem(
        media_id=f"{target.source_id}_real_c50",
        source_id=target.source_id,
        dataset_id=target.dataset_id,
        label=0,
        subject_id=target.subject_id,
    )
    extended = extend_lock(lock, base + [newcomer])
    assert extended.assignments[newcomer.media_id] is lock.assignments[target.media_id]


def test_a_grown_lock_is_still_leak_free():
    base = corpus(actors=20)
    grown = base + corpus(actors=20, first=20)
    extended = extend_lock(create_lock(base), grown)
    report = detect_leakage(grown, extended.assignments)
    assert report.clean, report.describe()


def test_growth_tracks_the_ratios():
    base = corpus(actors=20)
    grown = base + corpus(actors=40, first=20)
    extended = extend_lock(create_lock(base), grown)
    for split, target in ((Split.TRAIN, 0.7), (Split.VAL, 0.15), (Split.TEST, 0.15)):
        achieved = extended.achieved()[split]
        assert abs(achieved - target) < 0.1, (
            f"{split.value} drifted to {achieved:.3f} against {target}"
        )


def test_generation_and_parent_hash_chain():
    base = corpus(actors=10)
    g0 = create_lock(base)
    g1 = extend_lock(g0, base + corpus(actors=5, first=10))
    g2 = extend_lock(g1, base + corpus(actors=10, first=10))
    assert (g0.generation, g1.generation, g2.generation) == (0, 1, 2)
    assert g1.parent_hash == g0.content_hash()
    assert g2.parent_hash == g1.content_hash()
    assert g0.parent_hash is None


def test_extending_with_nothing_new_is_a_no_op_apart_from_the_generation():
    base = corpus(actors=12)
    g0 = create_lock(base)
    g1 = extend_lock(g0, base)
    assert g1.assignments == g0.assignments
    assert g1.generation == 1


def test_a_shrinking_corpus_is_refused():
    base = corpus(actors=12)
    lock = create_lock(base)
    try:
        extend_lock(lock, base[:-40])
    except SplitLockError as exc:
        assert "missing from the corpus" in str(exc)
        return
    raise AssertionError("dropping locked media was accepted")


# --------------------------------------------------------------------------
# The conflict the lock cannot solve
# --------------------------------------------------------------------------


def bridging_setup() -> tuple[list[MediaItem], MediaItem, SplitLock]:
    """Two components frozen into *different* splits, and an item merging them.

    The lock is built by hand rather than by ``create_lock``. A twelve-item
    corpus of two components does not straddle the folds under a 70/15/15 quota
    — both land in train — and a fixture that has to grow to fifty actors before
    it exhibits the conflict tests the quota rather than the conflict.
    """
    left = [
        MediaItem(f"L{i}", "srcL", "ffpp", i % 2, subject_id="pA") for i in range(6)
    ]
    right = [
        MediaItem(f"R{i}", "srcR", "ffpp", i % 2, subject_id="pB") for i in range(6)
    ]
    # Records the same person in both source videos: pA and pB were the same
    # actor all along, and the corrected metadata merges the components.
    bridge = MediaItem("BRIDGE", "srcL", "ffpp", 0, subject_id="pB")
    lock = SplitLock(
        assignments=(
            {i.media_id: Split.TRAIN for i in left}
            | {i.media_id: Split.TEST for i in right}
        ),
        seed="bridge",
        ratios={Split.TRAIN: 0.7, Split.VAL: 0.15, Split.TEST: 0.15},
    )
    return left + right, bridge, lock


def test_a_bridging_item_is_refused_by_default():
    items, bridge, lock = bridging_setup()
    assert len({lock.of("L0"), lock.of("R0")}) == 2, "premise: the two are split apart"
    try:
        extend_lock(lock, items + [bridge])
    except LockMergeConflict as exc:
        assert "BRIDGE" in str(exc)
        assert len(exc.conflicts) == 1
        return
    raise AssertionError("a bridging item was silently absorbed")


def test_quarantine_keeps_the_freeze_and_names_the_casualty():
    items, bridge, lock = bridging_setup()
    extended = extend_lock(lock, items + [bridge], on_conflict="quarantine")

    assert "BRIDGE" in extended.quarantined
    assert "BRIDGE" not in extended.assignments
    for media_id, split in lock.assignments.items():
        assert extended.assignments[media_id] is split, "the freeze was broken"


def test_quarantine_leaves_leakage_that_verification_keeps_reporting():
    """The honest consequence: freezing wins, and the cost stays visible."""
    items, bridge, frozen = bridging_setup()
    lock = extend_lock(frozen, items + [bridge], on_conflict="quarantine")
    audit = verify_lock(lock, items + [bridge])
    assert not audit.clean
    assert audit.conflicts, "the spanning identity stopped being reported"
    assert audit.quarantined_present == ("BRIDGE",)


def test_a_quarantined_item_is_released_once_the_identity_is_corrected():
    items, bridge, frozen = bridging_setup()
    quarantined = extend_lock(
        frozen, items + [bridge], on_conflict="quarantine"
    )
    # The bridge is re-indexed as its own subject: no longer a bridge.
    fixed = MediaItem("BRIDGE", "srcL", "ffpp", 0, subject_id="pA")
    released = extend_lock(quarantined, items + [fixed])
    assert "BRIDGE" not in released.quarantined
    assert released.assignments["BRIDGE"] is released.assignments["L0"]


# --------------------------------------------------------------------------
# Integrity
# --------------------------------------------------------------------------


def test_a_hand_edited_assignment_is_caught():
    lock = create_lock(corpus(actors=10))
    envelope = json.loads(lock.to_json())
    media_id = sorted(envelope["body"]["assignments"])[0]
    envelope["body"]["assignments"][media_id] = "train"  # may or may not differ
    envelope["body"]["assignments"][media_id] = (
        "test" if envelope["body"]["assignments"][media_id] != "test" else "train"
    )
    try:
        SplitLock.from_json(json.dumps(envelope))
    except LockHashMismatch as exc:
        assert "content hash mismatch" in str(exc)
        return
    raise AssertionError("an edited assignment passed the integrity check")


def test_a_truncated_file_is_caught():
    lock = create_lock(corpus(actors=8))
    truncated = lock.to_json()[: len(lock.to_json()) // 2]
    try:
        SplitLock.from_json(truncated)
    except LockCorrupted as exc:
        assert "not valid JSON" in str(exc)
        return
    raise AssertionError("a truncated lock was loaded")


def test_a_missing_body_field_is_caught():
    envelope = json.loads(create_lock(corpus(actors=6)).to_json())
    del envelope["body"]["seed"]
    try:
        SplitLock.from_json(json.dumps(envelope))
    except LockCorrupted as exc:
        assert "seed" in str(exc)
        return
    raise AssertionError("a lock missing its seed was loaded")


def test_a_bogus_split_name_is_caught():
    envelope = json.loads(create_lock(corpus(actors=6)).to_json())
    media_id = sorted(envelope["body"]["assignments"])[0]
    envelope["body"]["assignments"][media_id] = "holdout"
    try:
        SplitLock.from_json(json.dumps(envelope))
    except LockCorrupted as exc:
        assert "holdout" in str(exc)
        return
    raise AssertionError("an unknown split name was accepted")


def test_a_foreign_json_file_is_not_mistaken_for_a_lock():
    try:
        SplitLock.from_json('{"hello": "world"}')
    except LockCorrupted as exc:
        assert "not a split lock" in str(exc)
        return
    raise AssertionError("an arbitrary JSON object was read as a lock")


def test_an_assigned_item_may_not_also_be_quarantined():
    try:
        SplitLock(
            assignments={"a": Split.TRAIN},
            seed="s",
            ratios={Split.TRAIN: 1.0},
            quarantined=("a",),
        )
    except LockCorrupted as exc:
        assert "quarantined" in str(exc)
        return
    raise AssertionError("a media id was both assigned and quarantined")


def test_pretty_printing_does_not_change_the_hash():
    """The seal covers the canonical body, not the file's formatting."""
    lock = create_lock(corpus(actors=6))
    envelope = json.loads(lock.to_json())
    compact = json.dumps(envelope, separators=(",", ":"), sort_keys=True)
    assert SplitLock.from_json(compact).content_hash() == lock.content_hash()


# --------------------------------------------------------------------------
# Versioning and migration
# --------------------------------------------------------------------------


def test_an_unknown_lock_version_is_refused():
    envelope = json.loads(create_lock(corpus(actors=6)).to_json())
    envelope["body"]["lock_version"] = "9.9.9"
    try:
        SplitLock.from_json(json.dumps(envelope))
    except LockVersionMismatch as exc:
        assert "9.9.9" in str(exc)
        return
    raise AssertionError("a lock from an unknown format version was read")


def test_a_version_mismatch_is_reported_as_such_and_not_as_a_hash_failure():
    """Both are wrong at once; the reader must be sent to the right cause."""
    envelope = json.loads(create_lock(corpus(actors=6)).to_json())
    envelope["body"]["lock_version"] = "0.9.0"
    envelope["content_hash"] = "sha256:" + "0" * 64
    try:
        SplitLock.from_json(json.dumps(envelope))
    except LockVersionMismatch:
        return
    except LockHashMismatch:
        raise AssertionError(
            "reported a hash mismatch for what is a version problem"
        ) from None
    raise AssertionError("neither failure was raised")


def test_the_current_version_needs_no_migration():
    body = create_lock(corpus(actors=5)).body()
    assert apply_migrations(dict(body)) == body


def test_a_registered_migration_is_applied_and_the_lock_loads():
    """The mechanism, exercised with a fabricated predecessor version."""
    lock = create_lock(corpus(actors=8))
    old_body = dict(lock.body())
    old_body["lock_version"] = "0.9.0"
    old_body.pop("quarantined")  # pretend the field was added in 1.0.0

    def upgrade(body: dict) -> dict:
        out = dict(body)
        out["quarantined"] = []
        out["lock_version"] = LOCK_VERSION
        return out

    MIGRATIONS["0.9.0"] = upgrade
    try:
        migrated = apply_migrations(old_body)
        assert migrated["lock_version"] == LOCK_VERSION
        rebuilt = SplitLock._from_body(migrated)
        assert rebuilt.assignments == lock.assignments, (
            "a migration changed an assignment; migrations may add fields, "
            "never move items"
        )
    finally:
        MIGRATIONS.pop("0.9.0", None)


def test_a_body_with_no_version_is_corrupt_not_unversioned():
    body = create_lock(corpus(actors=5)).body()
    del body["lock_version"]
    try:
        apply_migrations(body)
    except LockCorrupted:
        return
    raise AssertionError("a body with no lock_version was migrated anyway")


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def test_verification_is_clean_on_a_matching_corpus():
    items = corpus(actors=15)
    audit = verify_lock(create_lock(items), items)
    assert audit.clean, audit.describe()


def test_verification_reports_unassigned_items():
    base = corpus(actors=10)
    grown = base + corpus(actors=3, first=10)
    audit = verify_lock(create_lock(base), grown)
    assert not audit.clean
    assert len(audit.unassigned) == len(grown) - len(base)
    assert "extend_lock" in audit.describe()


def test_verification_reports_a_shrunken_corpus():
    items = corpus(actors=10)
    audit = verify_lock(create_lock(items), items[:-20])
    assert audit.absent, audit.describe()


def test_verification_catches_leakage_introduced_by_a_hand_edit():
    """A lock edited to look plausible still fails the identity check."""
    items = corpus(actors=12)
    lock = create_lock(items)
    victim = items[0]
    sibling = next(
        i for i in items[1:] if i.source_id == victim.source_id
    )
    tampered = dict(lock.assignments)
    other = next(s for s in Split if s is not tampered[victim.media_id])
    tampered[sibling.media_id] = other
    edited = SplitLock(
        assignments=tampered, seed=lock.seed, ratios=lock.ratios
    )
    audit = verify_lock(edited, items)
    assert audit.leakage.findings, "a source spanning two splits was not reported"


def test_regeneration_check_is_identical_on_an_unchanged_corpus():
    items = corpus(actors=14)
    check = check_regeneration(create_lock(items), items)
    assert check.identical, check.describe()


def test_regeneration_check_names_what_growth_would_have_moved():
    base = corpus(actors=15)
    grown = base + corpus(actors=15, first=15)
    lock = extend_lock(create_lock(base), grown)
    check = check_regeneration(lock, grown)
    assert not check.identical, (
        "regeneration reproduced a grown corpus's lock exactly; the premise of "
        "the lock is that it would not"
    )
    assert "expected" in check.describe()


def test_a_lock_can_be_viewed_as_a_split_assignment():
    items = corpus(actors=20)
    lock = create_lock(items)
    assignment = lock.as_assignment(items)
    assert assignment.by_media == lock.assignments
    assert assignment.seed == lock.seed
    assert detect_leakage(items, assignment).clean


def test_asking_for_an_unlocked_media_id_fails_loudly():
    lock = create_lock(corpus(actors=6))
    try:
        lock.of("never-seen")
    except KeyError as exc:
        assert "extend_lock" in str(exc)
        return
    raise AssertionError("an unlocked media id returned a split")


def test_corpus_fingerprint_ignores_the_split_but_not_the_membership():
    items = corpus(actors=12)
    a = create_lock(items, seed="one")
    b = create_lock(items, seed="two")
    assert a.assignments != b.assignments
    assert a.corpus_fingerprint() == b.corpus_fingerprint()
    grown = create_lock(items + corpus(actors=2, first=12))
    assert grown.corpus_fingerprint() != a.corpus_fingerprint()


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
