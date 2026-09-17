"""R4: a lock must seal the identity behind its assignments, not just the assignments.

THE DEFECT, MEASURED
--------------------
A 1.0.0 lock sealed ``assignments`` and ``corpus_fingerprint``. Neither covers
``source_id`` or ``subject_id`` — the fields that decide which identity
component an item joins, and therefore whether the split leaks at all.

Reproduced on a 24-item corpus by relabelling one subject::

    components        6  ->  5
    corpus_fingerprint      unchanged
    content_hash            unchanged

The file remains valid, sealed, and byte-perfect while attesting to a split that
no longer exists. ``legacy_body`` below is the real 1.0.0 body shape, and
``test_the_legacy_format_cannot_detect_a_relabelled_actor`` asserts that it
cannot see this — so every check here is measured against the format it replaced
rather than against a claim.

TWO SEALS, TWO VERDICTS
-----------------------
``IDENTITY_FIELDS`` decide membership; ``CONTENT_FIELDS`` decide meaning.
Compression and resolution are in the second group and cannot leak a split — but
they define the slices the two-scope gate compares, so a silent re-bucketing
removes a scope from it. The audit reports the two independently because they
send the reader to different problems.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.dataset import (  # noqa: E402
    Compression,
    MediaItem,
    ResolutionBucket,
    build_components,
)
from evaluation.split_lock import (  # noqa: E402
    CONTENT_FIELDS,
    IDENTITY_FIELDS,
    LOCK_VERSION,
    SEALING_UNSEALED,
    SEALING_V1,
    LockCorrupted,
    LockHashMismatch,
    LockIdentityDrift,
    SplitLock,
    content_digest,
    create_lock,
    detect_drift,
    extend_lock,
    hash_body,
    identity_digest,
    verify_lock,
)


def corpus(n: int = 24, *, first: int = 0) -> list[MediaItem]:
    return [
        MediaItem(f"m{i}", f"s{i // 2}", "ffpp", i % 2, subject_id=f"p{i // 4}")
        for i in range(first, first + n)
    ]


def relabel(items, media_id: str, **changes) -> list[MediaItem]:
    """Return the corpus with one item's metadata edited. The tamper."""
    out = []
    for i in items:
        if i.media_id == media_id:
            fields = {
                "source_id": i.source_id,
                "dataset_id": i.dataset_id,
                "label": i.label,
                "subject_id": i.subject_id,
                "compression": i.compression,
                "resolution": i.resolution,
                "generator": i.generator,
            }
            fields.update(changes)
            out.append(MediaItem(media_id=i.media_id, **fields))
        else:
            out.append(i)
    return out


def tmp_path(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix="dfp-seal-")) / name


# --------------------------------------------------------------------------
# The legacy format, and what it could not see
# --------------------------------------------------------------------------


def legacy_body(lock: SplitLock) -> dict:
    """A genuine 1.0.0 body: the sealed fields as they stood before R4."""
    return {
        "assignments": {m: s.value for m, s in sorted(lock.assignments.items())},
        "corpus_fingerprint": lock.corpus_fingerprint(),
        "generation": lock.generation,
        "lock_version": "1.0.0",
        "parent_hash": lock.parent_hash,
        "quarantined": sorted(lock.quarantined),
        "ratios": {s.value: r for s, r in sorted(lock.ratios.items())},
        "seed": lock.seed,
        "subject_scope": lock.subject_scope,
    }


def legacy_file(lock: SplitLock) -> str:
    body = legacy_body(lock)
    return json.dumps(
        {"format": "dfp-split-lock", "content_hash": hash_body(body), "body": body},
        indent=2,
        sort_keys=True,
    ) + "\n"


# The three tampers, chosen from a measured sweep over this corpus. Each leaves
# every 1.0.0 check clean; the first two change which identity components exist.
#
#   subject merge, SAME split   p3 -> p1 via m12   components 6 -> 5, no leakage
#   subject split               p5 -> p99 via m20  components 6 -> 6
#   compression re-bucket       m5                 membership untouched
#
# A cross-split merge is deliberately NOT the premise here. 1.0.0 *does* catch
# that one — `detect_leakage` sees a subject spanning two splits — and building
# the argument on it would overstate the defect. The real finding is narrower and
# worse: 1.0.0's detection is **incidental**. It fires when the merged identities
# happen to straddle folds and is silent when they do not, so whether a
# relabelled corpus is caught depends on where the greedy splitter put them.
INVISIBLE_TO_1_0_0 = (
    ("same-split subject merge", "m12", {"subject_id": "p1"}),
    ("subject split", "m20", {"subject_id": "p99"}),
    ("compression re-bucket", "m5", {"compression": Compression.HEAVY}),
)


def test_the_legacy_checks_are_clean_on_every_one_of_these():
    """The premise: what a 1.0.0 lock verified cannot see any of them."""
    items = corpus()
    lock = create_lock(items)
    for name, victim, change in INVISIBLE_TO_1_0_0:
        tampered = relabel(items, victim, **change)
        audit = verify_lock(lock, tampered)
        # `assignments_intact` is exactly the 1.0.0 verdict: coverage, leakage,
        # merge conflicts. Nothing else was sealed or checked.
        assert audit.assignments_intact, (
            f"{name} is caught by the pre-R4 checks; it does not demonstrate the gap"
        )
        assert audit.leakage.clean and not audit.conflicts


def test_a_same_split_merge_changes_the_components_and_1_0_0_seals_nothing():
    """The sharpest case: two actors become one, and the file is still valid."""
    items = corpus()
    lock = create_lock(items)
    tampered = relabel(items, "m12", subject_id="p1")

    assert len(build_components(items)) == 6
    assert len(build_components(tampered)) == 5, "the fixture must merge components"

    body = legacy_body(lock)
    assert hash_body(body) == hash_body(legacy_body(lock)), "sanity: hashing is stable"
    # Nothing a 1.0.0 body records mentions subject_id at all.
    serialised = json.dumps(body)
    assert "p1" not in serialised and "subject" not in serialised.replace(
        "subject_scope", ""
    ), "a 1.0.0 body would have to record subject identity for this to be sealable"


def test_the_sealed_format_detects_all_three():
    """The same scenarios, now caught, named, and classified."""
    items = corpus()
    lock = create_lock(items)
    for name, victim, change in INVISIBLE_TO_1_0_0:
        drift = detect_drift(lock, relabel(items, victim, **change))
        assert drift, f"{name} went undetected"
        assert [d.media_id for d in drift] == [victim], name
        expected = "content" if "compression" in name else "identity"
        assert drift[0].kind == expected, f"{name} classified as {drift[0].kind}"


# --------------------------------------------------------------------------
# What determines membership — traced, not assumed
# --------------------------------------------------------------------------


def test_the_identity_fields_are_exactly_what_build_components_reads():
    """Change any one of them and the digest moves; change nothing else and it does not."""
    base = MediaItem("m", "s1", "ffpp", 0, subject_id="p1")
    assert IDENTITY_FIELDS == ("dataset_id", "source_id", "subject_id")
    for field, value in (
        ("dataset_id", "celebdf"),
        ("source_id", "s2"),
        ("subject_id", "p2"),
    ):
        changed = relabel([base], "m", **{field: value})[0]
        assert identity_digest(changed) != identity_digest(base), (
            f"{field} does not move the identity digest"
        )


def test_content_fields_do_not_move_the_identity_digest():
    """Compression and resolution cannot leak a split. That is why they are separate."""
    base = MediaItem("m", "s1", "ffpp", 0, subject_id="p1")
    for field, value in (
        ("compression", Compression.HEAVY),
        ("resolution", ResolutionBucket.LOW),
        ("generator", "FaceSwap"),
        ("label", 1),
    ):
        changed = relabel([base], "m", **{field: value})[0]
        assert identity_digest(changed) == identity_digest(base), (
            f"{field} moved the identity digest; it does not enter build_components"
        )
        assert content_digest(changed) != content_digest(base), (
            f"{field} did not move the content digest"
        )
    assert set(CONTENT_FIELDS) == {"label", "compression", "resolution", "generator"}


def test_a_compression_relabel_is_reported_as_content_not_identity():
    """Sound split, different meaning — and the audit must say which."""
    items = corpus()
    lock = create_lock(items)
    tampered = relabel(items, "m5", compression=Compression.HEAVY)

    audit = verify_lock(lock, tampered)
    assert audit.assignments_intact, "membership is untouched by a compression change"
    assert not audit.identity_intact
    assert not audit.identity_drift, "compression is not an identity field"
    assert [d.media_id for d in audit.content_drift] == ["m5"]
    assert "mean something else" in audit.describe()


def test_a_flipped_label_is_caught():
    """The starkest content change: every metric moves, no component does."""
    items = corpus()
    lock = create_lock(items)
    audit = verify_lock(lock, relabel(items, "m3", label=1 - items[3].label))
    assert audit.content_drift and not audit.identity_drift


# --------------------------------------------------------------------------
# Independent verification of the two integrities
# --------------------------------------------------------------------------


def test_assignment_integrity_and_identity_integrity_are_separate_verdicts():
    """A same-split merge: assignments perfect, identity account false."""
    items = corpus()
    lock = create_lock(items)
    audit = verify_lock(lock, relabel(items, "m12", subject_id="p1"))
    assert audit.assignments_intact, (
        "the lock still covers the corpus exactly; that verdict must not be "
        "polluted by the identity failure"
    )
    assert not audit.identity_intact
    assert not audit.clean


def test_a_cross_split_merge_fails_both_verdicts():
    """When 1.0.0 *did* catch it, R4 must not have made the report quieter."""
    items = corpus()
    audit = verify_lock(create_lock(items), relabel(items, "m20", subject_id="p0"))
    assert not audit.assignments_intact, "the leakage check should still fire"
    assert not audit.identity_intact
    assert audit.leakage.findings and audit.identity_drift


def test_a_clean_corpus_passes_both():
    items = corpus()
    audit = verify_lock(create_lock(items), items)
    assert audit.assignments_intact and audit.identity_intact and audit.clean
    assert "identity seal intact" in audit.describe()


def test_an_unsealed_lock_reports_unknown_rather_than_clean():
    """"No drift detected" and "no drift detectable" are different states."""
    items = corpus()
    unsealed = SplitLock(
        assignments=create_lock(items).assignments,
        seed="dfp-v1",
        ratios=create_lock(items).ratios,
    )
    assert unsealed.sealing_version == SEALING_UNSEALED
    audit = verify_lock(unsealed, items)
    assert audit.assignments_intact
    assert not audit.identity_intact, (
        "an unsealed lock reported an intact identity it has no evidence for"
    )
    assert not audit.sealed
    assert "carries no identity seal" in audit.describe()


def test_growth_is_not_drift():
    """New items have no seal to violate."""
    base = corpus(24)
    grown = base + corpus(12, first=24)
    assert detect_drift(create_lock(base), grown) == []


# --------------------------------------------------------------------------
# Re-sealing must not launder a change
# --------------------------------------------------------------------------


def test_extending_over_a_tampered_corpus_is_refused():
    """The trap: extend_lock re-seals, so it would make the tamper permanent."""
    items = corpus()
    lock = create_lock(items)
    tampered = relabel(items, "m20", subject_id="p0")
    try:
        extend_lock(lock, tampered + corpus(8, first=24))
    except LockIdentityDrift as exc:
        assert "m20" in str(exc)
        assert "would re-seal these" in str(exc)
        return
    raise AssertionError("extending over a relabelled corpus was accepted")


def test_extending_over_an_untampered_corpus_re_seals_correctly():
    base = corpus(24)
    grown = base + corpus(12, first=24)
    extended = extend_lock(create_lock(base), grown)
    assert set(extended.identity_digests) == set(extended.assignments)
    assert detect_drift(extended, grown) == []
    assert extended.sealing_version == SEALING_V1


def test_a_content_change_also_blocks_an_extension():
    items = corpus()
    lock = create_lock(items)
    tampered = relabel(items, "m5", compression=Compression.HEAVY)
    try:
        extend_lock(lock, tampered)
    except LockIdentityDrift as exc:
        assert "m5" in str(exc) and "content" in str(exc)
        return
    raise AssertionError("a re-bucketed slice was silently re-sealed")


# --------------------------------------------------------------------------
# Format versioning and migration
# --------------------------------------------------------------------------


def test_a_1_0_0_lock_still_loads():
    """Backward compatibility: existing files must keep working."""
    items = corpus()
    text = legacy_file(create_lock(items))
    loaded = SplitLock.from_json(text)
    assert loaded.lock_version == LOCK_VERSION == "1.1.0"
    assert loaded.assignments == create_lock(items).assignments, (
        "the migration changed an assignment"
    )


def test_a_migrated_lock_is_unsealed_and_says_so():
    """The migration must not invent digests for a corpus it never saw."""
    items = corpus()
    loaded = SplitLock.from_json(legacy_file(create_lock(items)))
    assert loaded.sealing_version == SEALING_UNSEALED
    assert not loaded.is_sealed
    assert loaded.identity_digests == {}
    # And crucially: it cannot be relied on to catch a tamper it never sealed.
    assert detect_drift(loaded, relabel(items, "m20", subject_id="p0")) == []
    assert not verify_lock(loaded, items).identity_intact


def test_a_migrated_lock_is_sealed_by_the_next_extension():
    """The migration path out of unsealed, against the corpus as it stands."""
    items = corpus()
    loaded = SplitLock.from_json(legacy_file(create_lock(items)))
    extended = extend_lock(loaded, items + corpus(8, first=24))
    assert extended.is_sealed
    assert extended.sealing_version == SEALING_V1
    assert verify_lock(extended, items + corpus(8, first=24)).identity_intact


def test_the_integrity_check_runs_against_the_bytes_as_written():
    """A 1.0.0 file must not fail as tampered merely because it was migrated."""
    text = legacy_file(create_lock(corpus()))
    SplitLock.from_json(text)  # must not raise

    broken = json.loads(text)
    broken["body"]["seed"] = "somebody-edited-this"
    try:
        SplitLock.from_json(json.dumps(broken))
    except LockHashMismatch:
        return
    raise AssertionError("an edited 1.0.0 body passed the integrity check")


def test_a_lock_claiming_a_seal_it_does_not_carry_is_refused():
    """The one lie a sealed format must not accept."""
    envelope = json.loads(create_lock(corpus()).to_json())
    envelope["body"]["identity_digests"] = {}
    envelope["body"]["content_digests"] = {}
    envelope["content_hash"] = hash_body(envelope["body"])  # correctly resealed
    try:
        SplitLock.from_json(json.dumps(envelope))
    except LockCorrupted as exc:
        assert "claims sealing_version" in str(exc)
        return
    raise AssertionError("a lock claiming identity-v1 with no digests was accepted")


def test_a_partially_sealed_lock_is_refused():
    envelope = json.loads(create_lock(corpus()).to_json())
    dropped = sorted(envelope["body"]["identity_digests"])[0]
    del envelope["body"]["identity_digests"][dropped]
    envelope["content_hash"] = hash_body(envelope["body"])
    try:
        SplitLock.from_json(json.dumps(envelope))
    except LockCorrupted as exc:
        assert "unsealed assignments" in str(exc)
        return
    raise AssertionError("a partially sealed lock was accepted")


def test_a_tampered_digest_map_is_caught_by_the_content_hash():
    """Editing a digest to match a relabelled item breaks the outer seal."""
    lock = create_lock(corpus())
    envelope = json.loads(lock.to_json())
    victim = sorted(envelope["body"]["identity_digests"])[0]
    envelope["body"]["identity_digests"][victim] = "deadbeef"
    try:
        SplitLock.from_json(json.dumps(envelope))
    except LockHashMismatch:
        return
    raise AssertionError("an edited identity digest passed the integrity check")


def test_sealing_survives_a_save_and_load_round_trip():
    items = corpus()
    path = tmp_path("split.lock.json")
    create_lock(items).save(path)
    loaded = SplitLock.load(path)
    assert loaded.is_sealed
    assert verify_lock(loaded, items).clean
    assert not verify_lock(loaded, relabel(items, "m20", subject_id="p0")).clean


def test_stale_generation_is_visible_through_the_chain():
    """A run pinned to generation 0 must be distinguishable from one at 1."""
    base = corpus(24)
    g0 = create_lock(base)
    g1 = extend_lock(g0, base + corpus(8, first=24))
    assert (g0.generation, g1.generation) == (0, 1)
    assert g1.parent_hash == g0.content_hash()
    assert g0.content_hash() != g1.content_hash()
    # The stale lock no longer covers the grown corpus, and says so precisely.
    stale = verify_lock(g0, base + corpus(8, first=24))
    assert stale.unassigned and not stale.clean
    assert "extend_lock" in stale.describe()


def test_a_sealed_lock_is_deterministic():
    items = corpus()
    assert create_lock(items).to_json() == create_lock(items).to_json()
    assert create_lock(items).to_json() == create_lock(list(reversed(items))).to_json()


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
