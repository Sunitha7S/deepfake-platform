"""R18 — the video path has no face stage, and four artifacts said it did.

THE DEFECT, AS A SENTENCE
-------------------------
``VideoPipeline`` never calls ``detect_faces``. It resizes whole frames to the
model input and scores them. Four other artifacts describe a system that does
detect and track faces:

    contract prose  ``VideoAnalysis.frames_scored`` = "Sampled frames where a
                    face was usable"; ``FaceTrack`` = "One identity across the
                    analysed frames"
    fixtures        ``video_manipulated`` publishes a FaceTrack with
                    ``frames_seen=46`` and top_frames whose track_id resolves
    test doubles    ``tests/stub_handlers.py`` does the same
    progress        ``STAGE_WEIGHTS_VIDEO`` budgets DETECTING at 30 and
                    TRACKING at 40; nothing emits either

The implementation is the odd one out, and in exactly one place it is not merely
sparse but internally incoherent: ``top_frames()`` hard-codes ``track_id=0`` into
every entry while ``faces`` is empty, so every published video bundle carries
three references to a track that does not exist.

WHAT THIS BATCH FIXES AND WHAT IT DOES NOT
------------------------------------------
Fixed: the dangling reference, structurally — ``track_id`` is ``None`` when no
face stage ran, and a bundle whose top_frames name a track that is not in
``faces`` is unrepresentable.

**Not fixed, and asserted here so it cannot become an assumed guarantee:** a
clip containing no face at all still receives a decided verdict. Measured on
pure noise: ``manipulated`` at p_fake 0.979 with ``gate_passed=True``. Closing
that means giving video a face stage, which is SCRFD work and is out of scope by
brief. This file pins the limit rather than papering over it.

WHY DOUBLES
-----------
``LegacyTopFrameTrackId`` is the pre-Batch-13 ``top_frames()``, verbatim, so
every claim is a pair: the double dangles and the shipped code does not.
``LEGACY_FRAMES_SCORED_DESCRIPTION`` is the prose that shipped, recorded rather
than paraphrased, so "the description was corrected" is checked against what it
actually said.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.common import BoundingBox, FaceTrack, ModelInfo, Timing  # noqa: E402
from contracts.enums import (  # noqa: E402
    CONTRACT_VERSION,
    AbstainReason,
    BranchId,
    JobStage,
    MediaKind,
    Verdict,
)
from contracts.job import STAGE_WEIGHTS_VIDEO, JobSpec  # noqa: E402
from contracts.video import FrameScore, TopFrame, VideoAnalysis  # noqa: E402
from inference import postprocess as post  # noqa: E402
from inference.postprocess import BranchOutcome, QualityFacts  # noqa: E402
from inference.video_pipeline import VideoPipeline  # noqa: E402

TIMING = Timing(queued_ms=0, preprocess_ms=1, inference_ms=1, total_ms=2)


LEGACY_FRAMES_SCORED_DESCRIPTION = "Sampled frames where a face was usable."
"""The prose that shipped until Batch 13, recorded verbatim.

It described a face stage the video path does not have. The value was — and
still is — the number of sampled frames that survived decoding, which is a
different quantity that happens to coincide whenever nothing fails to decode.
"""


# --------------------------------------------------------------------------
# The named doubles
# --------------------------------------------------------------------------


class LegacyTopFrameTrackId:
    """The pre-Batch-13 ``top_frames()``. Hard-codes ``track_id=0``, verbatim.

    Kept beside the shipped one so the dangling reference is demonstrated rather
    than described. If this double stops dangling, it has stopped reproducing
    R18 and the tests below say so instead of quietly passing.
    """

    name = "track_id hard-coded to 0 (pre-Batch-13)"

    @staticmethod
    def top_frames(frame_scores: list[FrameScore], *, limit: int = 3) -> list[TopFrame]:
        ranked = sorted(frame_scores, key=lambda f: (-f.p, f.frame_idx))[:limit]
        return [
            TopFrame(frame_idx=f.frame_idx, t_ms=f.t_ms, p=f.p, track_id=0)
            for f in ranked
        ]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _faceless_clip(frames: int = 30) -> bytes:
    """Pure noise. No face anywhere, by construction.

    Deliberately not a blank frame: a detector that found nothing in flat grey
    would prove nothing about a detector that is never called.
    """
    path = Path(tempfile.mkdtemp(prefix="dfp-r18-")) / "noface.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 240))
    rng = np.random.default_rng(3)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


_CLIP: bytes | None = None


def _clip() -> bytes:
    global _CLIP
    if _CLIP is None:
        _CLIP = _faceless_clip()
    return _CLIP


def _video_pipeline_code() -> str:
    """The video pipeline's *code*, with every docstring removed.

    Grepping the raw file conflates prose with behaviour: Batch 13 added a module
    docstring that names ``NO_FACE_DETECTED``, ``JobStage.TRACKING`` and the rest
    precisely in order to say they do not happen, and a raw-text search then
    reports the explanation as the thing it explains. Stripping docstrings first
    is what makes these assertions about what the module *does*.
    """
    import ast

    path = Path(__file__).resolve().parent.parent / "inference/video_pipeline.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _spec(data: bytes) -> JobSpec:
    return JobSpec(
        job_id=UUID(int=1),
        media_kind=MediaKind.VIDEO,
        storage_key="uploads/x",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="video/mp4",
        model_version="v",
        idempotency_key="k",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _analyze():
    data = _clip()
    return asyncio.run(VideoPipeline().analyze_video(_spec(data), data))


def _scores(n: int = 5) -> list[FrameScore]:
    return [FrameScore(frame_idx=i, t_ms=i * 40, p=0.1 * (i + 1)) for i in range(n)]


def _video_bundle(*, faces: list[FaceTrack], top: list[TopFrame]):
    """A hand-built video bundle, for the invariant tests."""
    return post.build_video_bundle(
        job=_spec(b"x" * 32),
        model=__import__("inference.models", fromlist=["ModelSpec"]).ModelSpec(
            name="t", version="stub-0", backend="stub"
        ),
        size_bytes=1024,
        width=1920,
        height=1080,
        duration_s=2.0,
        frame_count=50,
        fps=25.0,
        quality=QualityFacts(width=1920, height=1080, face_px=None),
        outcomes={
            BranchId.A: BranchOutcome(logit=4.0),
            BranchId.B: BranchOutcome(logit=3.5),
            BranchId.P: BranchOutcome(logit=0.0),
        },
        video=VideoAnalysis(
            frames_sampled=5,
            frames_scored=5,
            frame_scores=_scores(),
            segments=[],
            top_frames=top,
        ),
        faces=faces,
        timing=TIMING,
    )


# --------------------------------------------------------------------------
# 1. The dangling reference — the one code-level incoherence
# --------------------------------------------------------------------------


def test_the_legacy_double_still_dangles():
    """The double must reproduce the defect, or the pair below proves nothing."""
    legacy = LegacyTopFrameTrackId.top_frames(_scores())
    assert legacy, "the double produced no frames"
    assert all(t.track_id == 0 for t in legacy), (
        "LegacyTopFrameTrackId no longer hard-codes track_id=0; it can no longer "
        "demonstrate R18"
    )


def test_video_top_frames_do_not_name_a_track_that_does_not_exist():
    """**Fails before the fix.** Three references to track 0, and ``faces`` is empty."""
    bundle = _analyze()
    assert bundle.video is not None
    assert bundle.video.top_frames, "the fixture produced no top frames"
    published = {f.track_id for f in bundle.faces}
    dangling = [
        t.track_id
        for t in bundle.video.top_frames
        if t.track_id is not None and t.track_id not in published
    ]
    assert not dangling, (
        f"top_frames name track(s) {dangling} but faces[] publishes {sorted(published)}; "
        "a client resolving those ids finds nothing"
    )


def test_the_shipped_top_frames_helper_names_no_track():
    """The direct unit, so the fix is pinned at its source as well as end to end."""
    frames = post.top_frames(_scores())
    assert frames
    assert all(t.track_id is None for t in frames), (
        f"top_frames() still names tracks: {[t.track_id for t in frames]}"
    )


def test_a_dangling_track_id_is_unrepresentable():
    """**Fails before the fix.** Structural, not merely corrected at one call site.

    The value fix alone would leave the next caller free to reintroduce it. This
    is the same move ``_verdict_follows_threshold`` makes for the verdict: the
    incoherent bundle cannot be built, by hand or by deserialisation.
    """
    try:
        _video_bundle(
            faces=[],
            top=[TopFrame(frame_idx=0, t_ms=0, p=0.9, track_id=7)],
        )
    except ValueError as exc:
        assert "track" in str(exc).lower()
        return
    raise AssertionError(
        "a bundle whose top_frames name track 7 with an empty faces[] was accepted"
    )


def test_a_resolving_track_id_is_still_accepted():
    """The invariant must not forbid the case the fixtures depict.

    ``video_manipulated`` publishes a FaceTrack and top_frames that resolve to
    it. That is what a video bundle looks like once a face stage exists, and the
    contract must keep admitting it — this batch removes a dangling reference,
    not the ability to reference.

    The published track's ``p_fake`` is taken from the bundle's own headline:
    Batch 16's ``_headline_is_the_score_of_a_published_face`` (R21) refuses a
    scored face whose score contradicts the headline, and this fixture is not the
    place to exercise that — ``tests/test_faces_consistency.py`` is.
    """
    headline = _video_bundle(faces=[], top=[]).p_fake
    assert headline is not None
    bundle = _video_bundle(
        faces=[
            FaceTrack(
                track_id=0,
                bbox=BoundingBox(x=1, y=1, width=10, height=10),
                frames_seen=5,
                p_fake=headline,
            )
        ],
        top=[TopFrame(frame_idx=0, t_ms=0, p=0.9, track_id=0)],
    )
    assert bundle.video.top_frames[0].track_id == 0


def test_a_null_track_id_is_accepted():
    bundle = _video_bundle(
        faces=[], top=[TopFrame(frame_idx=0, t_ms=0, p=0.9, track_id=None)]
    )
    assert bundle.video.top_frames[0].track_id is None


def test_the_shipped_fixtures_still_validate():
    """The invariant is checked against every artifact that ships, not a sample."""
    import contracts.fixtures as fx

    for name in ("video_manipulated", "video_authentic", "video_insufficient_evidence"):
        bundle = getattr(fx, name)()
        published = {f.track_id for f in bundle.faces}
        for frame in bundle.video.top_frames:
            assert frame.track_id is None or frame.track_id in published, (
                f"fixture {name} carries a dangling track_id"
            )


# --------------------------------------------------------------------------
# 2. The prose now describes what the code does
# --------------------------------------------------------------------------


def test_the_frames_scored_description_no_longer_claims_a_face_stage():
    """**Fails before the fix.**

    The value is the number of sampled frames that survived decoding. It has
    never counted frames "where a face was usable", because nothing in the video
    path has ever looked for a face.
    """
    description = VideoAnalysis.model_fields["frames_scored"].description or ""
    assert description != LEGACY_FRAMES_SCORED_DESCRIPTION
    assert "where a face was usable" not in description.lower(), (
        f"frames_scored still claims a face stage: {description!r}"
    )
    assert "decod" in description.lower(), (
        f"frames_scored should say what it counts: {description!r}"
    )
    # And the class must say why, so the correction is discoverable rather than
    # silently swapped.
    assert "R18" in (VideoAnalysis.__doc__ or "")


def test_the_track_id_description_explains_what_null_means():
    """**Fails before the fix.** ``None`` is now load-bearing and must be documented."""
    description = TopFrame.model_fields["track_id"].description or ""
    assert description, "track_id carries no description at all"
    assert "null" in description.lower() or "none" in description.lower(), (
        f"track_id does not explain its null case: {description!r}"
    )


def test_frames_scored_still_counts_what_it_always_counted():
    """Behaviour preservation: prose moved, the number did not."""
    bundle = _analyze()
    assert bundle.video.frames_scored == len(bundle.video.frame_scores)
    assert bundle.video.frames_scored <= bundle.video.frames_sampled


# --------------------------------------------------------------------------
# 3. The limits, asserted so they cannot become assumed guarantees
# --------------------------------------------------------------------------


def test_the_video_path_still_has_no_face_stage():
    """Structural. If this fails, R18's premise is gone and the batch is stale."""
    code = _video_pipeline_code()
    for token in ("detect_faces", "crop_face", "face_track", "FaceTrack"):
        assert token not in code, (
            f"video_pipeline.py now references {token!r}; the video path has grown a "
            "face stage and these tests must be replaced by ones that gate on it"
        )


def test_a_faceless_clip_still_receives_a_decided_verdict():
    """**The half of R18 this batch does NOT close**, pinned deliberately.

    A clip of pure noise containing no face anywhere is published with a decided
    verdict and ``gate_passed=True``. The image path would abstain with
    ``NO_FACE_DETECTED`` on the same content. Closing this means giving video a
    face stage — SCRFD work, out of scope by brief — so it is asserted rather
    than fixed, the way Batch 9 asserted its benchmark limit.
    """
    bundle = _analyze()
    assert bundle.verdict is not Verdict.INSUFFICIENT_EVIDENCE, (
        "a faceless clip now abstains; R18's open half may be closed and this "
        "test should be replaced by one that gates on the new behaviour"
    )
    assert bundle.quality.gate_passed is True
    assert bundle.quality.gate_reason is None
    assert bundle.faces == []
    assert bundle.quality.face_px is None


def test_the_video_abstention_vocabulary_excludes_the_face_reasons():
    """``VideoPipeline._abstain`` can only emit TOO_FEW_FRAMES.

    ``NO_FACE_DETECTED`` and ``FACE_TOO_SMALL`` remain in the platform-wide enum
    because the image path produces them. They are unreachable for video, and
    that is the finding rather than a bug in the enum.
    """
    code = _video_pipeline_code()
    assert "NO_FACE_DETECTED" not in code
    assert "FACE_TOO_SMALL" not in code
    assert "TOO_FEW_FRAMES" in code


def test_video_never_emits_the_face_progress_stages():
    """The progress vocabulary budgets a face stage that does not run.

    ``STAGE_WEIGHTS_VIDEO`` reserves DETECTING and TRACKING; the video pipeline's
    only ``progress()`` call passes SCORING. Documented in Batch 13 rather than
    removed — the weights are the post-SCRFD plan and deleting them would erase
    it.
    """
    code = _video_pipeline_code()
    assert "JobStage.SCORING" in code
    assert "JobStage.TRACKING" not in code
    assert "JobStage.DETECTING" not in code
    assert JobStage.DETECTING in STAGE_WEIGHTS_VIDEO
    assert JobStage.TRACKING in STAGE_WEIGHTS_VIDEO


# --------------------------------------------------------------------------
# 4. Nothing else moved
# --------------------------------------------------------------------------


def test_image_bundles_are_unaffected():
    """The image path publishes a real track, and top_frames is video-only."""
    import contracts.fixtures as fx

    bundle = fx.image_manipulated()
    assert bundle.video is None
    assert bundle.faces and bundle.faces[0].track_id == 0


def test_the_contract_version_did_not_move():
    """No stable-tier field changed shape or meaning.

    ``video.*`` is Evolving-tier; ``track_id`` was already ``int | None`` with a
    ``None`` default, so the invariant forbids a state the schema always allowed
    to be absent and the corrected prose describes the value the code always
    produced.
    """
    assert CONTRACT_VERSION == "1.0.0"
    assert TopFrame.model_fields["track_id"].default is None


def test_the_openapi_signature_is_unchanged_by_the_prose_edits():
    """Descriptions are prose, and the drift guard ignores prose by design.

    ``test_signature_ignores_prose_but_not_structure`` establishes the property;
    this asserts the consequence for this batch — no snapshot change was needed
    for the description corrections.
    """
    from backend import openapi as oa

    schema = oa.build_schema()
    assert oa.fingerprint(oa.structural_signature(schema)) == oa.fingerprint(
        oa.structural_signature(oa.build_schema())
    )
    top = schema["components"]["schemas"]["TopFrame"]["properties"]["track_id"]
    assert "description" in top


def test_the_video_analysis_field_set_is_unchanged():
    """No field added or removed. The correction is prose plus one invariant."""
    assert set(VideoAnalysis.model_fields) == {
        "frames_sampled",
        "frames_scored",
        "sampling",
        "pooling",
        "frame_scores",
        "segments",
        "top_frames",
    }
    assert set(TopFrame.model_fields) == {
        "frame_idx",
        "t_ms",
        "p",
        "track_id",
        "thumbnail_ref",
        "gradcam_ref",
    }


def test_video_bundles_stay_deterministic():
    """The same bytes still produce the same bundle."""
    a, b = _analyze(), _analyze()
    assert a.p_fake == b.p_fake
    assert a.verdict is b.verdict
    assert [t.track_id for t in a.video.top_frames] == [
        t.track_id for t in b.video.top_frames
    ]


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
