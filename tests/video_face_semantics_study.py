"""R18 -- the video path has no face stage, and four artifacts said it did.

    python tests/video_face_semantics_study.py

Produces every number in ``docs/VIDEO_FACE_SEMANTICS.md``. Same pattern as
``decode_study.py``, ``band_study.py``, ``inference_determinism_study.py``,
``multi_face_study.py`` and ``face_selection_study.py``.

THE QUESTION
------------
Batch 11's architecture review found that ``VideoPipeline`` never calls
``detect_faces``, and recorded R18. This study asks the follow-up the brief
poses: **do the contract, the fixtures, the doubles, the implementation and the
tests agree about that** — and where they disagree, which disagreements are
correctness defects and which are prose?

The answer is that five artifacts describe the system and **the implementation is
the odd one out**, but only one of the disagreements is a code-level incoherence.
That distinction is the whole of what Batch 13 could fix without building a face
stage, which is SCRFD work and out of scope.

WHAT THIS STUDY MEASURES RATHER THAN ASSERTS
--------------------------------------------
Stage 3 executes the fail-before evidence: it builds the exact bundle the
platform published yesterday, using ``LegacyTopFrameTrackId`` — the pre-Batch-13
``top_frames()``, verbatim — and shows the contract now refuses it. A test that
only exercises the fix cannot tell you the fix was needed.
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

from contracts.common import BoundingBox, FaceTrack, Timing  # noqa: E402
from contracts.enums import BranchId, JobStage, MediaKind  # noqa: E402
from contracts.job import STAGE_WEIGHTS_VIDEO, JobSpec  # noqa: E402
from contracts.video import FrameScore, TopFrame, VideoAnalysis  # noqa: E402
from inference import postprocess as post  # noqa: E402
from inference.models import ModelSpec  # noqa: E402
from inference.postprocess import BranchOutcome, QualityFacts  # noqa: E402
from inference.video_pipeline import VideoPipeline  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dfp-r18-study-"))
TIMING = Timing(queued_ms=0, preprocess_ms=1, inference_ms=1, total_ms=2)
SPEC = ModelSpec(name="t", version="stub-0", backend="stub")


def head(n: int, title: str, question: str) -> None:
    print(f"\n{'=' * 76}\n{n}. {title}\n   {question}\n{'=' * 76}")


# --------------------------------------------------------------------------
# The named double
# --------------------------------------------------------------------------


class LegacyTopFrameTrackId:
    """The pre-Batch-13 ``top_frames()``. Hard-codes ``track_id=0``, verbatim."""

    name = "track_id hard-coded to 0 (pre-Batch-13)"

    @staticmethod
    def top_frames(frame_scores: list[FrameScore], *, limit: int = 3) -> list[TopFrame]:
        ranked = sorted(frame_scores, key=lambda f: (-f.p, f.frame_idx))[:limit]
        return [
            TopFrame(frame_idx=f.frame_idx, t_ms=f.t_ms, p=f.p, track_id=0)
            for f in ranked
        ]


LEGACY_FRAMES_SCORED_DESCRIPTION = "Sampled frames where a face was usable."
"""The prose that shipped until Batch 13, recorded verbatim."""


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def faceless_clip(frames: int = 30) -> bytes:
    """Pure noise. No face anywhere, by construction."""
    path = TMP / "noface.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 240))
    rng = np.random.default_rng(3)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def spec_for(data: bytes) -> JobSpec:
    return JobSpec(
        job_id=UUID(int=1),
        media_kind=MediaKind.VIDEO,
        storage_key="u",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="video/mp4",
        model_version="v",
        idempotency_key="k",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def scores(n: int = 5) -> list[FrameScore]:
    return [FrameScore(frame_idx=i, t_ms=i * 40, p=0.1 * (i + 1)) for i in range(n)]


def build(*, faces: list[FaceTrack], top: list[TopFrame]):
    return post.build_video_bundle(
        job=spec_for(b"x" * 32),
        model=SPEC,
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
            frames_sampled=5, frames_scored=5, frame_scores=scores(),
            segments=[], top_frames=top,
        ),
        faces=faces,
        timing=TIMING,
    )


_BUNDLE = None


def live():
    global _BUNDLE
    if _BUNDLE is None:
        data = faceless_clip()
        _BUNDLE = asyncio.run(VideoPipeline().analyze_video(spec_for(data), data))
    return _BUNDLE


# --------------------------------------------------------------------------
# 1 — trace every face-related field on the video path
# --------------------------------------------------------------------------


def s1_trace() -> None:
    head(1, "Every video-path use of a face-related field",
         "what does the pipeline actually emit?")
    b = live()
    rows = [
        ("faces[]", repr(b.faces), "never populated -- no detector is called"),
        ("FaceTrack", "none constructed", "video_pipeline.py references the type 0 times"),
        ("track_id (faces)", "n/a", "no track exists to carry one"),
        ("top_frames[].track_id", str([t.track_id for t in b.video.top_frames]),
         "was [0,0,0] and dangling; now null -- FIXED in B13"),
        ("frames_scored", f"{b.video.frames_scored} of {b.video.frames_sampled} sampled",
         "frames that decoded; never a face count"),
        ("quality.face_px", repr(b.quality.face_px), "null -- nothing measured a face"),
        ("gate_passed", repr(b.quality.gate_passed), "always true; no face gate exists"),
        ("gate_reason", repr(b.quality.gate_reason), "null; only TOO_FEW_FRAMES reachable"),
        ("NO_FACE_DETECTED", "unreachable", "VideoPipeline._abstain cannot emit it"),
        ("FACE_TOO_SMALL", "unreachable", "same"),
    ]
    print(f"  {'field':<24} {'video value':<26} note")
    print(f"  {'-'*24} {'-'*26} {'-'*44}")
    for name, value, note in rows:
        print(f"  {name:<24} {value:<26} {note}")

    print("\n  Progress vocabulary:")
    print(f"    STAGE_WEIGHTS_VIDEO budgets DETECTING={STAGE_WEIGHTS_VIDEO[JobStage.DETECTING]}"
          f" and TRACKING={STAGE_WEIGHTS_VIDEO[JobStage.TRACKING]}")
    print("    the pipeline's only progress() call passes JobStage.SCORING")


# --------------------------------------------------------------------------
# 2 — do the five voices agree?
# --------------------------------------------------------------------------


def s2_voices() -> None:
    head(2, "Do contract, fixtures, doubles, implementation and tests agree?",
         "on detection, scoring, tracking, and what frames_scored means")
    import contracts.fixtures as fx

    fixture = fx.video_manipulated()
    b = live()

    print(f"  {'artifact':<26} {'detects?':<10} {'tracks?':<10} frames_scored means")
    print(f"  {'-'*26} {'-'*10} {'-'*10} {'-'*34}")
    print(f"  {'contract prose (pre-B13)':<26} {'implied':<10} {'implied':<10} "
          f"'where a face was usable'")
    print(f"  {'fixtures':<26} {'yes':<10} {'yes':<10} "
          f"{fixture.video.frames_scored} scored, track frames_seen="
          f"{fixture.faces[0].frames_seen}")
    print(f"  {'stub_handlers double':<26} {'yes':<10} {'yes':<10} = frames sampled")
    print(f"  {'IMPLEMENTATION':<26} {'NO':<10} {'NO':<10} "
          f"{b.video.frames_scored} frames that decoded")
    print(f"  {'tests (pre-B13)':<26} {'silent':<10} {'silent':<10} "
          f"only monotonicity asserted")

    print("\n  The implementation is the odd one out on every column.")
    print("  Four artifacts describe a face stage; one runs no face code at all.")
    print()
    print("  But only ONE of those disagreements is a code-level incoherence:")
    print("    fixtures/doubles depicting a target      -> intent, not a defect")
    print("    prose describing a stage that is absent  -> documentation defect")
    print("    top_frames naming a non-existent track   -> CORRECTNESS defect")


# --------------------------------------------------------------------------
# 3 — fail before, pass after (executed)
# --------------------------------------------------------------------------


def s3_fail_before() -> None:
    head(3, "The correctness defect, reproduced",
         "build yesterday's bundle and ask the contract to accept it")
    print(f"  double: {LegacyTopFrameTrackId.name}\n")

    legacy_top = LegacyTopFrameTrackId.top_frames(scores())
    print(f"  legacy top_frames  -> track_ids {[t.track_id for t in legacy_top]}")
    shipped_top = post.top_frames(scores())
    print(f"  shipped top_frames -> track_ids {[t.track_id for t in shipped_top]}")

    print("\n  A: the bundle the platform published yesterday (legacy ids, faces=[]):")
    try:
        build(faces=[], top=legacy_top)
        print("     ACCEPTED  <- the defect is still present")
    except ValueError as exc:
        # pydantic wraps the message; surface the line that names the defect.
        reason = next(
            (ln.strip() for ln in str(exc).splitlines() if "track" in ln.lower()),
            str(exc).splitlines()[0],
        )
        print(f"     REFUSED   {reason[:150]}")
        print("     ^ this is the fail-before evidence: that exact bundle validated")
        print("       before Batch 13 and cannot be constructed now.")

    print("\n  B: today's bundle (null ids, faces=[]):")
    try:
        b = build(faces=[], top=shipped_top)
        print(f"     ACCEPTED  track_ids={[t.track_id for t in b.video.top_frames]}")
    except ValueError as exc:
        print(f"     REFUSED   {exc}")

    print("\n  C: the post-SCRFD shape the fixtures depict (ids that resolve):")
    # p_fake from the bundle's own headline — Batch 16's R21 invariant refuses a
    # scored face that contradicts it, and C is about referencing, not that.
    headline = build(faces=[], top=[]).p_fake
    track = FaceTrack(track_id=0, bbox=BoundingBox(x=1, y=1, width=10, height=10),
                      frames_seen=5, p_fake=headline)
    try:
        b = build(faces=[track], top=legacy_top)
        print(f"     ACCEPTED  track_ids={[t.track_id for t in b.video.top_frames]}"
              " resolve to faces[0]")
    except ValueError as exc:
        print(f"     REFUSED   {exc}")

    print("\n  The invariant forbids dangling, not referencing. A face stage that")
    print("  lands later publishes tracks and names them, and C is what it looks like.")

    print("\n  Prose, before and after:")
    print(f"    before: {LEGACY_FRAMES_SCORED_DESCRIPTION!r}")
    after = VideoAnalysis.model_fields["frames_scored"].description
    print(f"    after : {after!r}")


# --------------------------------------------------------------------------
# 4 — downstream consumers
# --------------------------------------------------------------------------


def s4_consumers() -> None:
    head(4, "Downstream consumers", "who relies on the current behaviour?")
    rows = [
        ("frames_scored", "published only", "no computation reads it",
         "tests/test_timing_contract.py asserts it grows with n_frames"),
        ("top_frames[].track_id", "published only", "no computation reads it",
         "nothing in-repo resolves it; a UI would"),
        ("faces[]", "published only", "no video code writes it",
         "EvidenceBundle validators now read it for referential integrity"),
        ("quality.face_px", "published only", "null for every video bundle",
         "QualityReport allows null; no consumer"),
        ("STAGE_WEIGHTS_VIDEO", "progress percent", "maps stage -> percent",
         "DETECTING/TRACKING entries are dead but harmless"),
    ]
    print(f"  {'field':<24} {'role':<16} {'computation':<28} consumer")
    print(f"  {'-'*24} {'-'*16} {'-'*28} {'-'*40}")
    for name, role, comp, consumer in rows:
        print(f"  {name:<24} {role:<16} {comp:<28} {consumer}")
    print()
    print("  Nothing computes anything from these fields. That is why the correction")
    print("  is cheap: the blast radius of changing track_id from 0 to null is the")
    print("  published payload only, and no in-repo consumer resolved it.")


# --------------------------------------------------------------------------
# 5 — what Batch 13 does NOT fix
# --------------------------------------------------------------------------


def s5_not_fixed() -> None:
    head(5, "What is NOT fixed", "the half of R18 that needs a face stage")
    b = live()
    print("  A clip of PURE NOISE -- no face anywhere, by construction:")
    print(f"    verdict        : {b.verdict.value}")
    print(f"    p_fake         : {b.p_fake}")
    print(f"    gate_passed    : {b.quality.gate_passed}")
    print(f"    gate_reason    : {b.quality.gate_reason}")
    print(f"    abstain_reason : {b.abstain_reason}")
    print()
    print("  The image path abstains with NO_FACE_DETECTED on equivalent content.")
    print("  The video path decides. Closing that gap means adding detection to the")
    print("  video pipeline, which is the SCRFD migration:")
    print("    - a Haar-based stage would inherit R13's 310x crop instability, on")
    print("      every one of 48 frames rather than once;")
    print("    - R15's threading requirement transfers to whatever replaces it;")
    print("    - and per-frame tracking is a component the platform does not have.")
    print()
    print("  So Batch 13 corrected the CLAIMS and left the CAPABILITY alone.")
    print("  test_a_faceless_clip_still_receives_a_decided_verdict pins the limit so")
    print("  it cannot decay into an assumed guarantee.")


# --------------------------------------------------------------------------


def main() -> int:
    print(__doc__)
    s1_trace()
    s2_voices()
    s3_fail_before()
    s4_consumers()
    s5_not_fixed()
    print(f"\n{'=' * 76}\nartifacts in {TMP}\n{'=' * 76}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
