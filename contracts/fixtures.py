"""Deterministic EvidenceBundle fixtures.

This module is why the whole web application can be built before a single model
exists. It emits realistic, contract-valid bundles covering every verdict, every
branch status and both media kinds — with no torch, no weights and no GPU.

Everything here is deterministic: fixed UUIDs, fixed timestamps, curves from a
closed-form function. Fixtures are used as golden files in tests and are exported
verbatim to ``frontend/src/fixtures/`` so the UI and the backend cannot drift.

Usage::

    python -m contracts.fixtures            # write JSON to ./fixtures_out
    python -m contracts.fixtures --stdout   # print one bundle
"""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from contracts.branch import BRANCH_NAMES, BranchResult
from contracts.common import (
    DEFAULT_LOG_ODDS_MARGIN,
    Artifact,
    BoundingBox,
    FaceTrack,
    MediaInfo,
    ModelInfo,
    QualityReport,
    ThresholdInfo,
    Timing,
)
from contracts.enums import (
    AbstainReason,
    ArtifactKind,
    BranchId,
    BranchStatus,
    JobStage,
    JobState,
    MediaKind,
    ResolutionClass,
    Verdict,
)
from contracts.errors import ApiError, ErrorCode, ErrorResponse
from contracts.evidence import EvidenceBundle
from contracts.job import JobProgress, JobStatus
from contracts.provenance import (
    EMPTY_PROVENANCE,
    C2paReport,
    ExifReport,
    ProvenanceFinding,
    ProvenanceReport,
)
from contracts.video import FrameScore, SuspectSegment, TopFrame, VideoAnalysis

FIXED_TIME = datetime(2026, 8, 23, 14, 30, 0, tzinfo=UTC)
MODEL_VERSION = "1.0.0-dev"

# The shipped operating point: threshold tuned at 2% FPR, with an abstention
# band either side of it where the model declines rather than guesses.
THRESHOLD = ThresholdInfo.at_margin(
    value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
)

MODEL = ModelInfo(
    version=MODEL_VERSION,
    branch_versions={"A": "convnextv2t-sbi-1.0", "B": "dinov2b-probe-1.0", "P": "c2pa-1.0"},
    calibration_temp=1.37,
    git_sha="0000000000000000000000000000000000000000",
)


def _uuid(n: int) -> UUID:
    return UUID(f"{n:032x}")


def _sha(n: int) -> str:
    return f"{n:064x}"


def _branch(
    bid: BranchId,
    *,
    logit: float | None = None,
    p: float | None = None,
    weight: float | None = None,
    reason: AbstainReason | None = None,
) -> BranchResult:
    # Branch P is always built with ``p=None``: its logit is a one-sided
    # authenticity credit, not log-odds, so it has no probability reading.
    # Publishing σ(credit) rendered the commonest case — no provenance at all —
    # as a confident-looking "50% manipulated", which is the one-sided rule
    # leaking into the UI. Fixtures are the frontend's specification, so they
    # must show the shape the engine actually emits.
    scored = logit is not None
    return BranchResult(
        id=bid,
        name=BRANCH_NAMES[bid],
        status=BranchStatus.SCORED if scored else BranchStatus.ABSTAINED,
        logit=logit,
        p=p,
        weight=weight if scored else None,
        abstain_reason=None if scored else reason,
        version=MODEL.branch_versions.get(bid.value),
    )


def _timing(total: int) -> Timing:
    return Timing(
        queued_ms=int(total * 0.05),
        preprocess_ms=int(total * 0.25),
        inference_ms=int(total * 0.55),
        explain_ms=int(total * 0.12),
        total_ms=total,
    )


# --------------------------------------------------------------------------
# Image fixtures
# --------------------------------------------------------------------------


def image_manipulated() -> EvidenceBundle:
    """The headline case: a face swap, both learned branches agree, artifacts present."""
    return EvidenceBundle(
        job_id=_uuid(1),
        created_at=FIXED_TIME,
        media=MediaInfo(
            kind=MediaKind.IMAGE,
            sha256=_sha(1),
            size_bytes=2_184_233,
            mime="image/jpeg",
            width=1920,
            height=1080,
        ),
        verdict=Verdict.MANIPULATED,
        p_fake=0.9134827362,
        interval=(0.8712, 0.9455),
        threshold=THRESHOLD,
        quality=QualityReport(
            jpeg_q_est=88.0,
            face_px=412,
            blur_score=0.12,
            resolution_class=ResolutionClass.HIGH,
            gate_passed=True,
        ),
        branches=[
            _branch(BranchId.A, logit=2.3581093, p=0.9136, weight=0.72),
            _branch(BranchId.B, logit=1.1204418, p=0.7540, weight=0.24),
            _branch(BranchId.P, logit=0.0, weight=0.04),
        ],
        disagreement=0.1596,
        faces=[
            FaceTrack(
                track_id=0,
                bbox=BoundingBox(x=742, y=196, width=436, height=436),
                landmarks=[(812.0, 340.0), (1010.0, 338.0), (912.0, 448.0), (836.0, 534.0), (992.0, 532.0)],
                frames_seen=1,
                p_fake=0.9134827362,
            )
        ],
        provenance=EMPTY_PROVENANCE,
        artifacts=[
            Artifact(
                kind=ArtifactKind.GRADCAM,
                ref="/artifacts/00000001/gradcam.png",
                mime="image/png",
                width=1920,
                height=1080,
                caption="Regions that most influenced this score",
            ),
            Artifact(
                kind=ArtifactKind.SPECTRUM,
                ref="/artifacts/00000001/spectrum.svg",
                mime="image/svg+xml",
                caption="Frequency profile compared with the authentic reference band",
            ),
        ],
        model=MODEL,
        timing=_timing(1420),
    )


def image_authentic() -> EvidenceBundle:
    """A real photograph with no metadata — the common authentic case.

    Note branch P scores 0.0 and still carries weight. Absence of provenance
    contributes exactly nothing; it does not drag the score upward.
    """
    return EvidenceBundle(
        job_id=_uuid(2),
        created_at=FIXED_TIME,
        media=MediaInfo(
            kind=MediaKind.IMAGE,
            sha256=_sha(2),
            size_bytes=987_412,
            mime="image/jpeg",
            width=1440,
            height=1440,
        ),
        verdict=Verdict.AUTHENTIC,
        p_fake=0.0731844029,
        interval=(0.0412, 0.1198),
        threshold=THRESHOLD,
        quality=QualityReport(
            jpeg_q_est=76.0,
            face_px=298,
            blur_score=0.19,
            resolution_class=ResolutionClass.HIGH,
            gate_passed=True,
        ),
        branches=[
            _branch(BranchId.A, logit=-2.5391847, p=0.0731, weight=0.72),
            _branch(BranchId.B, logit=-1.8830291, p=0.1320, weight=0.24),
            _branch(BranchId.P, logit=0.0, weight=0.04),
        ],
        disagreement=0.0589,
        faces=[
            FaceTrack(
                track_id=0,
                bbox=BoundingBox(x=560, y=402, width=310, height=310),
                landmarks=[(618.0, 498.0), (760.0, 496.0), (690.0, 570.0), (634.0, 640.0), (748.0, 638.0)],
                frames_seen=1,
                p_fake=0.0731844029,
            )
        ],
        provenance=EMPTY_PROVENANCE,
        artifacts=[
            Artifact(
                kind=ArtifactKind.GRADCAM,
                ref="/artifacts/00000002/gradcam.png",
                mime="image/png",
                width=1440,
                height=1440,
                caption="Regions that most influenced this score",
            )
        ],
        model=MODEL,
        timing=_timing(1190),
    )


def image_authentic_signed() -> EvidenceBundle:
    """A camera-captured image carrying a valid C2PA manifest.

    The only case where branch P contributes a positive logit.
    """
    prov = ProvenanceReport(
        c2pa=C2paReport(
            present=True,
            signature_valid=True,
            signer="Leica Camera AG",
            claim_generator="Leica M11-P",
            captured_at=datetime(2026, 7, 4, 9, 12, 44, tzinfo=UTC),
            edits_declared=["c2pa.color_adjustments"],
        ),
        exif=ExifReport(
            present=True,
            coherent=True,
            camera_make="Leica",
            camera_model="M11-P",
            software="Adobe Lightroom 14.2",
        ),
        authenticity_credit=0.92,
        findings=[
            ProvenanceFinding(
                label="Valid content credentials",
                detail="Signed by Leica Camera AG at capture. The signature verifies against the file's current contents.",
                supports_authenticity=True,
            ),
            ProvenanceFinding(
                label="Declared edits",
                detail="The manifest declares colour adjustments only. No content-altering edits are recorded.",
                supports_authenticity=True,
            ),
        ],
    )
    return EvidenceBundle(
        job_id=_uuid(3),
        created_at=FIXED_TIME,
        media=MediaInfo(
            kind=MediaKind.IMAGE,
            sha256=_sha(3),
            size_bytes=14_882_301,
            mime="image/jpeg",
            width=3840,
            height=2560,
        ),
        verdict=Verdict.AUTHENTIC,
        p_fake=0.0142093811,
        interval=(0.0051, 0.0388),
        threshold=THRESHOLD,
        quality=QualityReport(
            jpeg_q_est=97.0,
            face_px=884,
            blur_score=0.04,
            resolution_class=ResolutionClass.HIGH,
            gate_passed=True,
        ),
        branches=[
            _branch(BranchId.A, logit=-3.8812094, p=0.0202, weight=0.68),
            _branch(BranchId.B, logit=-3.1049183, p=0.0429, weight=0.22),
            _branch(BranchId.P, logit=1.4400000, weight=0.10),
        ],
        disagreement=0.0227,
        faces=[
            FaceTrack(
                track_id=0,
                bbox=BoundingBox(x=1480, y=610, width=920, height=920),
                frames_seen=1,
                p_fake=0.0142093811,
            )
        ],
        provenance=prov,
        artifacts=[],
        model=MODEL,
        timing=_timing(2380),
    )


def image_no_face() -> EvidenceBundle:
    """A landscape. Branch A abstains; the semantic branch carries the verdict alone.

    **R22 — this depicts the frozen-architecture target, not current behaviour.**
    Branch B's target is whole-image scoring, so a faceless image is still
    scoreable by B and the gate lets it through. The shipped pipeline does not do
    this: ``ModelBackend.score`` is single-input, branch B is fed the same face
    crop as branch A, and ``ImagePipeline._gate`` abstains the whole bundle with
    ``NO_FACE_DETECTED`` — ``test_inference::test_no_face_path_abstains``. This
    fixture is the shape a whole-image branch B would publish, kept as the
    frontend's specification exactly as the video fixtures depict the post-SCRFD
    face stage. ``docs/BRANCH_B_SEMANTICS.md``.
    """
    return EvidenceBundle(
        job_id=_uuid(4),
        created_at=FIXED_TIME,
        media=MediaInfo(
            kind=MediaKind.IMAGE,
            sha256=_sha(4),
            size_bytes=1_204_882,
            mime="image/png",
            width=1024,
            height=1024,
        ),
        verdict=Verdict.MANIPULATED,
        p_fake=0.8419203847,
        interval=(0.7112, 0.9210),
        threshold=THRESHOLD,
        quality=QualityReport(
            jpeg_q_est=None,
            face_px=0,
            blur_score=0.08,
            resolution_class=ResolutionClass.STANDARD,
            gate_passed=True,
        ),
        branches=[
            _branch(BranchId.A, reason=AbstainReason.NO_FACE_DETECTED),
            _branch(BranchId.B, logit=1.6690283, p=0.8414, weight=0.94),
            _branch(BranchId.P, logit=0.0, weight=0.06),
        ],
        disagreement=0.0,
        faces=[],
        provenance=EMPTY_PROVENANCE,
        artifacts=[
            Artifact(
                kind=ArtifactKind.SPECTRUM,
                ref="/artifacts/00000004/spectrum.svg",
                mime="image/svg+xml",
                caption="Frequency profile compared with the authentic reference band",
            )
        ],
        model=MODEL,
        timing=_timing(880),
    )


def image_insufficient_evidence() -> EvidenceBundle:
    """A tiny, heavily recompressed face. The gate declines before any model votes.

    This is a *successful* job. p_fake is null, and the UI must present it as an
    answer, not a failure.
    """
    return EvidenceBundle(
        job_id=_uuid(5),
        created_at=FIXED_TIME,
        media=MediaInfo(
            kind=MediaKind.IMAGE,
            sha256=_sha(5),
            size_bytes=18_442,
            mime="image/jpeg",
            width=320,
            height=240,
        ),
        verdict=Verdict.INSUFFICIENT_EVIDENCE,
        p_fake=None,
        interval=None,
        threshold=THRESHOLD,
        abstain_reason=AbstainReason.FACE_TOO_SMALL,
        quality=QualityReport(
            jpeg_q_est=34.0,
            face_px=41,
            blur_score=0.71,
            resolution_class=ResolutionClass.TINY,
            gate_passed=False,
            gate_reason=AbstainReason.FACE_TOO_SMALL,
        ),
        branches=[
            _branch(BranchId.A, reason=AbstainReason.FACE_TOO_SMALL),
            _branch(BranchId.B, reason=AbstainReason.QUALITY_BELOW_FLOOR),
            _branch(BranchId.P, reason=AbstainReason.NO_PROVENANCE_DATA),
        ],
        disagreement=0.0,
        faces=[
            FaceTrack(
                track_id=0,
                bbox=BoundingBox(x=138, y=76, width=41, height=41),
                frames_seen=1,
                status=AbstainReason.FACE_TOO_SMALL,
            )
        ],
        provenance=EMPTY_PROVENANCE,
        artifacts=[],
        model=MODEL,
        timing=_timing(210),
    )


# --------------------------------------------------------------------------
# Video fixtures
# --------------------------------------------------------------------------


def _curve(n: int, *, peak_at: float, width: float, base: float, peak: float) -> list[float]:
    """Deterministic bell-shaped probability curve. No randomness anywhere."""
    out = []
    for i in range(n):
        x = i / max(n - 1, 1)
        bump = math.exp(-(((x - peak_at) / width) ** 2))
        out.append(round(base + (peak - base) * bump, 10))
    return out


def video_manipulated() -> EvidenceBundle:
    """A 12-second clip with a manipulated segment in the middle third.

    The curve is the point: a short manipulated run inside a longer authentic
    clip is exactly what mean pooling destroys and top-k pooling preserves.
    """
    n = 48
    fps = 25.0
    duration = 12.0
    probs = _curve(n, peak_at=0.55, width=0.13, base=0.06, peak=0.94)
    step_ms = int(duration * 1000 / n)

    frame_scores = [
        FrameScore(frame_idx=i * 6, t_ms=i * step_ms, p=probs[i]) for i in range(n)
    ]
    hot = [f for f in frame_scores if f.p >= 0.62]
    segment = SuspectSegment(
        start_ms=hot[0].t_ms,
        end_ms=hot[-1].t_ms,
        peak_p=max(f.p for f in hot),
        mean_p=round(sum(f.p for f in hot) / len(hot), 10),
        frame_count=len(hot),
    )
    top = sorted(frame_scores, key=lambda f: -f.p)[:3]

    return EvidenceBundle(
        job_id=_uuid(6),
        created_at=FIXED_TIME,
        media=MediaInfo(
            kind=MediaKind.VIDEO,
            sha256=_sha(6),
            size_bytes=41_882_004,
            mime="video/mp4",
            width=1920,
            height=1080,
            duration_s=duration,
            frame_count=300,
            fps=fps,
        ),
        verdict=Verdict.MANIPULATED,
        p_fake=0.8871029384,
        interval=(0.8018, 0.9344),
        threshold=THRESHOLD,
        quality=QualityReport(
            jpeg_q_est=71.0,
            face_px=356,
            blur_score=0.22,
            resolution_class=ResolutionClass.HIGH,
            gate_passed=True,
        ),
        branches=[
            _branch(BranchId.A, logit=2.0672913, p=0.8877, weight=0.78),
            _branch(BranchId.B, logit=0.7710244, p=0.6838, weight=0.18),
            _branch(BranchId.P, logit=0.0, weight=0.04),
        ],
        disagreement=0.2039,
        faces=[
            FaceTrack(
                track_id=0,
                bbox=BoundingBox(x=690, y=180, width=380, height=380),
                frames_seen=46,
                p_fake=0.8871029384,
            )
        ],
        video=VideoAnalysis(
            frames_sampled=n,
            frames_scored=46,
            sampling="uniform",
            pooling="topk_mean",
            frame_scores=frame_scores,
            segments=[segment],
            top_frames=[
                TopFrame(
                    frame_idx=f.frame_idx,
                    t_ms=f.t_ms,
                    p=f.p,
                    track_id=0,
                    thumbnail_ref=f"/artifacts/00000006/frame_{f.frame_idx}.jpg",
                    gradcam_ref=f"/artifacts/00000006/gradcam_{f.frame_idx}.png",
                )
                for f in top
            ],
        ),
        provenance=EMPTY_PROVENANCE,
        artifacts=[
            Artifact(
                kind=ArtifactKind.FRAME_CURVE,
                ref="/artifacts/00000006/curve.svg",
                mime="image/svg+xml",
                caption="Per-frame manipulation probability across the clip",
            )
        ],
        model=MODEL,
        timing=_timing(8940),
    )


def video_authentic() -> EvidenceBundle:
    """A real clip. Flat low curve, no segments."""
    n = 32
    duration = 8.0
    probs = _curve(n, peak_at=0.5, width=0.30, base=0.04, peak=0.19)
    step_ms = int(duration * 1000 / n)
    frame_scores = [FrameScore(frame_idx=i * 6, t_ms=i * step_ms, p=probs[i]) for i in range(n)]

    return EvidenceBundle(
        job_id=_uuid(7),
        created_at=FIXED_TIME,
        media=MediaInfo(
            kind=MediaKind.VIDEO,
            sha256=_sha(7),
            size_bytes=12_004_882,
            mime="video/mp4",
            width=1280,
            height=720,
            duration_s=duration,
            frame_count=192,
            fps=24.0,
        ),
        verdict=Verdict.AUTHENTIC,
        p_fake=0.1108293847,
        interval=(0.0704, 0.1712),
        threshold=THRESHOLD,
        quality=QualityReport(
            jpeg_q_est=64.0,
            face_px=214,
            blur_score=0.31,
            resolution_class=ResolutionClass.STANDARD,
            gate_passed=True,
        ),
        branches=[
            _branch(BranchId.A, logit=-2.0801934, p=0.1110, weight=0.78),
            _branch(BranchId.B, logit=-1.5520481, p=0.1747, weight=0.18),
            _branch(BranchId.P, logit=0.0, weight=0.04),
        ],
        disagreement=0.0637,
        faces=[
            FaceTrack(
                track_id=0,
                bbox=BoundingBox(x=430, y=120, width=248, height=248),
                frames_seen=31,
                p_fake=0.1108293847,
            )
        ],
        video=VideoAnalysis(
            frames_sampled=n,
            frames_scored=31,
            frame_scores=frame_scores,
            segments=[],
            top_frames=[],
        ),
        provenance=EMPTY_PROVENANCE,
        artifacts=[
            Artifact(
                kind=ArtifactKind.FRAME_CURVE,
                ref="/artifacts/00000007/curve.svg",
                mime="image/svg+xml",
                caption="Per-frame manipulation probability across the clip",
            )
        ],
        model=MODEL,
        timing=_timing(6210),
    )


def video_insufficient_evidence() -> EvidenceBundle:
    """A clip where the face was usable in too few sampled frames."""
    return EvidenceBundle(
        job_id=_uuid(8),
        created_at=FIXED_TIME,
        media=MediaInfo(
            kind=MediaKind.VIDEO,
            sha256=_sha(8),
            size_bytes=3_118_920,
            mime="video/webm",
            width=640,
            height=480,
            duration_s=4.0,
            frame_count=100,
            fps=25.0,
        ),
        verdict=Verdict.INSUFFICIENT_EVIDENCE,
        p_fake=None,
        interval=None,
        threshold=THRESHOLD,
        abstain_reason=AbstainReason.TOO_FEW_FRAMES,
        quality=QualityReport(
            jpeg_q_est=48.0,
            face_px=96,
            blur_score=0.58,
            resolution_class=ResolutionClass.LOW,
            gate_passed=False,
            gate_reason=AbstainReason.TOO_FEW_FRAMES,
        ),
        branches=[
            _branch(BranchId.A, reason=AbstainReason.TOO_FEW_FRAMES),
            _branch(BranchId.B, reason=AbstainReason.QUALITY_BELOW_FLOOR),
            _branch(BranchId.P, reason=AbstainReason.NO_PROVENANCE_DATA),
        ],
        faces=[],
        video=VideoAnalysis(
            frames_sampled=32,
            frames_scored=3,
            frame_scores=[],
            segments=[],
            top_frames=[],
        ),
        provenance=EMPTY_PROVENANCE,
        artifacts=[],
        model=MODEL,
        timing=_timing(3040),
    )


# --------------------------------------------------------------------------
# Job status and error fixtures
# --------------------------------------------------------------------------


def job_running() -> JobStatus:
    return JobStatus(
        job_id=_uuid(6),
        state=JobState.RUNNING,
        progress=JobProgress(stage=JobStage.SCORING, percent=75, message="Scoring 48 frames"),
        submitted_at=FIXED_TIME,
        started_at=FIXED_TIME,
        attempt=1,
    )


def job_succeeded() -> JobStatus:
    return JobStatus(
        job_id=_uuid(6),
        state=JobState.SUCCEEDED,
        progress=JobProgress(stage=JobStage.FINALIZING, percent=100),
        submitted_at=FIXED_TIME,
        started_at=FIXED_TIME,
        finished_at=FIXED_TIME,
        attempt=1,
        result_url="/api/v1/jobs/00000000-0000-0000-0000-000000000006/result",
    )


def job_failed() -> JobStatus:
    return JobStatus(
        job_id=_uuid(9),
        state=JobState.FAILED,
        submitted_at=FIXED_TIME,
        started_at=FIXED_TIME,
        finished_at=FIXED_TIME,
        attempt=1,
        error=ApiError.of(ErrorCode.DECODE_FAILED, detail="moov atom not found"),
    )


def error_too_large() -> ErrorResponse:
    return ErrorResponse(error=ApiError.of(ErrorCode.FILE_TOO_LARGE, field="size_bytes"))


# --------------------------------------------------------------------------
# Registry and CLI
# --------------------------------------------------------------------------

BUNDLES = {
    "image_manipulated": image_manipulated,
    "image_authentic": image_authentic,
    "image_authentic_signed": image_authentic_signed,
    "image_no_face": image_no_face,
    "image_insufficient_evidence": image_insufficient_evidence,
    "video_manipulated": video_manipulated,
    "video_authentic": video_authentic,
    "video_insufficient_evidence": video_insufficient_evidence,
}

OTHERS = {
    "job_running": job_running,
    "job_succeeded": job_succeeded,
    "job_failed": job_failed,
    "error_too_large": error_too_large,
}

ALL = {**BUNDLES, **OTHERS}


def all_bundles() -> dict[str, EvidenceBundle]:
    return {name: fn() for name, fn in BUNDLES.items()}


def write_all(out_dir: Path) -> list[Path]:
    """Emit every fixture as formatted JSON. Consumed by the frontend build."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in ALL.items():
        path = out_dir / f"{name}.json"
        path.write_text(fn().model_dump_json(indent=2), encoding="utf-8")
        written.append(path)
    index = out_dir / "index.json"
    index.write_text(json.dumps(sorted(ALL), indent=2), encoding="utf-8")
    written.append(index)
    return written


def main(argv: list[str]) -> int:
    if "--stdout" in argv:
        name = next((a for a in argv[1:] if not a.startswith("-")), "image_manipulated")
        if name not in ALL:
            print(f"unknown fixture {name!r}; choose from {sorted(ALL)}", file=sys.stderr)
            return 2
        print(ALL[name]().model_dump_json(indent=2))
        return 0
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path("fixtures_out")
    written = write_all(out)
    print(f"wrote {len(written)} files to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
