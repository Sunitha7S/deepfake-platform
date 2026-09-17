"""R21 — ``faces[]`` had no uniqueness or consistency invariant.

THE DEFECT, AS A SENTENCE
------------------------
``EvidenceBundle`` carried ``_branches_unique`` from Batch 0 and, from Batch 13,
``_top_frames_name_a_published_track``. It carried nothing about ``faces[]``
itself. So two of these validated:

* two ``FaceTrack``s at the same ``track_id`` with contradictory ``p_fake`` — a
  client cannot tell them apart, and ``_top_frames_name_a_published_track``
  silently deduplicates the ids it resolves against;
* a decided headline ``p_fake`` equal to no published face's score — ``verdict``
  = ``manipulated`` at 0.913 beside faces scored 0.90 and 0.10.

Both are the class of drift ``_verdict_follows_threshold`` and the
``ThresholdInfo`` band derivation were written to make unrepresentable, left
open on the one field an R16 per-face design will populate.

WHY A DOUBLE RATHER THAN A GIT CHECKOUT
--------------------------------------
``LegacyEvidenceContract`` is not a description of the old contract; it is the
tuple of ``EvidenceBundle`` model-validators that existed before Batch 16,
looked up by name on the live class, minus the two this batch adds.
``accepts(bundle)`` runs exactly those. Every fail-before test is a pair: the
malformed bundle satisfies the entire prior contract and the shipped contract
refuses it. ``test_the_legacy_contract_double_is_complete`` fails if a validator
is added or renamed without updating the double, so it cannot quietly fall
behind.

WHAT IS AND IS NOT CLAIMED
-------------------------
These invariants stop a bundle contradicting its own evidence. They do **not**
make one face's verdict a correct summary of a group photograph (R16), compute
the non-selected faces' evidence, or give the video path a face stage (R18).
The headline rule is deliberately vacuous where the platform legitimately
publishes a per-face score without a headline (in-band abstention) or a headline
without a face (video). Those limits are asserted here, not left to prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import contracts.fixtures as fx  # noqa: E402
from contracts.common import BoundingBox, FaceTrack  # noqa: E402
from contracts.enums import AbstainReason, MediaKind, Verdict  # noqa: E402
from contracts.evidence import EvidenceBundle  # noqa: E402

# --------------------------------------------------------------------------
# The named double
# --------------------------------------------------------------------------

_PRE_B16_INVARIANTS = (
    "_verdict_and_probability_agree",
    "_interval_brackets_estimate",
    "_branches_unique",
    "_video_block_matches_media_kind",
    "_top_frames_name_a_published_track",
    "_gate_failure_implies_abstention",
    "_verdict_follows_threshold",
)

_B16_INVARIANTS = (
    "_faces_are_uniquely_identified",
    "_headline_is_the_score_of_a_published_face",
)


class LegacyEvidenceContract:
    """The ``EvidenceBundle`` model-validators as of Batch 15, by name.

    Not a paraphrase — the bound methods on the current class, minus the two
    Batch 16 adds. ``accepts()`` answers "would yesterday's contract have
    validated this bundle?"
    """

    name = "EvidenceBundle validators, pre-Batch-16 (no faces[] invariant)"
    invariants = _PRE_B16_INVARIANTS

    @classmethod
    def accepts(cls, bundle: EvidenceBundle) -> bool:
        for method_name in cls.invariants:
            getattr(EvidenceBundle, method_name)(bundle)
        return True


# --------------------------------------------------------------------------
# Helpers — build an otherwise-valid bundle with a chosen faces[] list
# --------------------------------------------------------------------------


def _base() -> EvidenceBundle:
    """A shipped, fully valid manipulated-image bundle. Headline == faces[0]."""
    return fx.image_manipulated()


def _with_faces(faces: list[FaceTrack], *, base: EvidenceBundle | None = None) -> EvidenceBundle:
    """``base`` with ``faces`` swapped in, skipping bundle-level validation.

    ``model_construct`` leaves every nested object (each ``FaceTrack``,
    ``ThresholdInfo``, the branch rows) a real validated instance and skips only
    the ``EvidenceBundle`` cross-field checks — which is exactly the set the
    legacy double and the shipped contract are being compared on.
    """
    base = base or _base()
    fields = {name: getattr(base, name) for name in EvidenceBundle.model_fields}
    fields["faces"] = faces
    return EvidenceBundle.model_construct(**fields)


def _validate(bundle: EvidenceBundle) -> None:
    """Re-run the full shipped contract on a constructed bundle."""
    EvidenceBundle.model_validate(bundle.model_dump())


def _refused(bundle: EvidenceBundle, needle: str) -> bool:
    try:
        _validate(bundle)
    except ValueError as exc:
        return needle in str(exc)
    return False


# --------------------------------------------------------------------------
# 0. The double is faithful and complete
# --------------------------------------------------------------------------


def test_the_legacy_contract_double_is_complete():
    """Every EvidenceBundle validator is either pre-B16 or one of the two new ones.

    If this fails, a validator was added or renamed and the fail-before pairs
    below no longer prove "satisfied the whole prior contract".
    """
    live = set(EvidenceBundle.__pydantic_decorators__.model_validators)
    assert live == set(_PRE_B16_INVARIANTS) | set(_B16_INVARIANTS), (
        f"validator set drifted: {sorted(live)}"
    )


def test_the_legacy_contract_accepts_every_shipped_fixture():
    """The double must accept what actually ships, or it is not 'the prior contract'."""
    for name, bundle in fx.all_bundles().items():
        assert LegacyEvidenceContract.accepts(bundle), f"legacy rejected fixture {name}"


# --------------------------------------------------------------------------
# 1. Uniqueness — fail before, pass after
# --------------------------------------------------------------------------


def test_duplicate_track_ids_satisfied_the_prior_contract():
    """**Fails after the fix.** Two tracks at id 0, contradictory scores."""
    f0 = _base().faces[0]
    bad = _with_faces([f0, f0.model_copy(update={"p_fake": 0.1})])
    assert LegacyEvidenceContract.accepts(bad), (
        "a bundle with two track_id=0 faces did not satisfy the pre-B16 contract"
    )


def test_duplicate_track_ids_are_now_refused():
    f0 = _base().faces[0]
    bad = _with_faces([f0, f0.model_copy(update={"p_fake": 0.1})])
    assert _refused(bad, "duplicate track_id in faces[]")


def test_three_faces_with_a_repeated_id_are_refused():
    f0 = _base().faces[0]
    bad = _with_faces(
        [
            f0,
            f0.model_copy(update={"track_id": 1}),
            f0.model_copy(update={"track_id": 1, "p_fake": 0.2}),
        ]
    )
    assert _refused(bad, "duplicate track_id in faces[]")


def test_distinct_ids_are_accepted():
    """Uniqueness must not forbid the multi-face case R16 will produce."""
    headline = _base().p_fake
    assert headline is not None
    f0 = _base().faces[0]
    ok = _base().model_copy(
        update={
            "faces": [
                f0,
                f0.model_copy(update={"track_id": 1, "p_fake": 0.1}),
                f0.model_copy(update={"track_id": 2, "p_fake": 0.2}),
            ]
        }
    )
    _validate(ok)  # headline still equals faces[0].p_fake — allowed


# --------------------------------------------------------------------------
# 2. Headline consistency — fail before, pass after
# --------------------------------------------------------------------------


def test_a_headline_matching_no_face_satisfied_the_prior_contract():
    """**Fails after the fix.** verdict=manipulated at 0.913, faces scored 0.80 / 0.10."""
    f0 = _base().faces[0]
    bad = _with_faces(
        [
            f0.model_copy(update={"p_fake": 0.80}),
            f0.model_copy(update={"track_id": 1, "p_fake": 0.10}),
        ]
    )
    assert LegacyEvidenceContract.accepts(bad), (
        "a bundle whose headline matched no face did not satisfy the pre-B16 contract"
    )


def test_a_headline_matching_no_face_is_now_refused():
    f0 = _base().faces[0]
    bad = _with_faces(
        [
            f0.model_copy(update={"p_fake": 0.80}),
            f0.model_copy(update={"track_id": 1, "p_fake": 0.10}),
        ]
    )
    assert _refused(bad, "is the score of no published face")


def test_the_headline_may_equal_any_published_face_not_only_the_first():
    """R16 Option 2 publishes N faces and defines the headline as one of them.

    The rule is 'equal to *a* scored face', not 'equal to faces[0]', so a design
    that ranks the second-listed face as the headline is representable.
    """
    headline = _base().p_fake
    assert headline is not None
    f0 = _base().faces[0]
    ok = _base().model_copy(
        update={
            "faces": [
                f0.model_copy(update={"p_fake": 0.10}),
                f0.model_copy(update={"track_id": 1, "p_fake": headline}),
            ]
        }
    )
    _validate(ok)


# --------------------------------------------------------------------------
# 3. The rule is vacuous exactly where the platform legitimately diverges
# --------------------------------------------------------------------------


def test_an_abstention_may_publish_a_scored_face():
    """In-band abstention: bundle p_fake is None, the face's is not.

    This is the image pipeline publishing branch-score visibility on a verdict it
    declines to decide. The headline rule must not touch it.
    """
    base = fx.image_manipulated()
    f0 = base.faces[0]
    abstained = _with_faces(
        [f0.model_copy(update={"p_fake": 0.63})],
        base=base.model_copy(
            update={
                "verdict": Verdict.INSUFFICIENT_EVIDENCE,
                "p_fake": None,
                "interval": None,
                "abstain_reason": AbstainReason.SCORE_IN_ABSTENTION_BAND,
            }
        ),
    )
    LegacyEvidenceContract.accepts(abstained)
    _validate(abstained)


def test_a_decided_video_bundle_with_no_published_face_is_accepted():
    """R18: the video path publishes faces=[] and still decides. Unchanged."""
    base = fx.video_manipulated()
    video = base.video.model_copy(
        update={
            "top_frames": [f.model_copy(update={"track_id": None}) for f in base.video.top_frames]
        }
    )
    bundle = base.model_copy(update={"faces": [], "video": video})
    _validate(bundle)


def test_the_rule_is_vacuous_when_no_published_face_scored():
    """Carve-out boundary, asserted so it is a decision and not an accident.

    An abstained face (``status`` set, ``p_fake`` null) is not a headline
    candidate, and the rule only bites when *some* published face scored. So a
    decided bundle whose only face abstained is **accepted** — the same
    treatment ``faces=[]`` gets. Tightening this would need to tell that shape
    apart from the video path's legitimate ``faces=[]``, which is R18/R16 work,
    not R21. Recorded as a guarantee not gained.
    """
    f_abstain = FaceTrack(
        track_id=0,
        bbox=BoundingBox(x=1, y=1, width=10, height=10),
        frames_seen=1,
        status=AbstainReason.FACE_TOO_SMALL,
    )
    _validate(_with_faces([f_abstain]))


# --------------------------------------------------------------------------
# 4. Everything that ships still validates
# --------------------------------------------------------------------------


def test_every_shipped_fixture_still_validates():
    for name, bundle in fx.all_bundles().items():
        _validate(bundle)


def test_the_single_face_pipeline_shape_validates():
    """faces[0].p_fake computed from the same fused outcomes as the headline."""
    for factory in (fx.image_manipulated, fx.image_authentic, fx.image_authentic_signed):
        bundle = factory()
        assert bundle.p_fake == bundle.faces[0].p_fake
        _validate(bundle)


# --------------------------------------------------------------------------
# 5. Quantify the defect
# --------------------------------------------------------------------------


def test_quantify_before_and_after():
    """The window the prior contract left open, measured.

    For a decided bundle publishing one scored face, the prior contract accepted
    the face's score anywhere in [0, 1] — a gap of the full probability range
    against a headline it was meant to describe. And it accepted any number of
    faces sharing one id.
    """
    f0 = _base().faces[0]
    headline = _base().p_fake
    assert headline is not None

    # Sweep the published face's score across the range, and include the exact
    # headline value so there is one legitimate candidate.
    swept = [i / 20 for i in range(21)] + [headline]
    legacy_accepts = shipped_accepts = 0
    for score in swept:
        cand = _with_faces([f0.model_copy(update={"p_fake": score})])
        if LegacyEvidenceContract.accepts(cand):
            legacy_accepts += 1
        try:
            _validate(cand)
            shipped_accepts += 1
        except ValueError:
            pass

    # The prior contract accepted the face's score anywhere in [0, 1] — a gap of
    # the full probability range against a headline it was meant to describe.
    assert legacy_accepts == len(swept)
    # The shipped contract accepts exactly the one score that equals the headline.
    assert shipped_accepts == 1

    # Duplicate-id count: the prior contract accepted an unbounded pile; the
    # shipped one accepts none.
    pile = _with_faces([f0.model_copy(update={"p_fake": headline})] * 8)
    assert LegacyEvidenceContract.accepts(pile)
    assert _refused(pile, "duplicate track_id in faces[]")


# --------------------------------------------------------------------------
# 6. The referential-integrity invariant still behaves
# --------------------------------------------------------------------------


def test_top_frames_still_resolve_against_unique_faces():
    """Batch 13's invariant and Batch 16's compose without conflict."""
    bundle = fx.video_manipulated()
    published = {f.track_id for f in bundle.faces}
    for frame in bundle.video.top_frames:
        assert frame.track_id is None or frame.track_id in published
    _validate(bundle)


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
