"""Deterministic stub handlers. **Test doubles — not production code.**

These lived in ``workers/handlers.py`` until inference was wired in. They were
moved here rather than deleted because worker *orchestration* tests — leasing,
retries, crash recovery, shutdown — need a handler that is cheap, deterministic
and free of media decoding. What they must not need is a real scorer, and what
``workers/`` must not contain is scoring logic.

Real analysis now lives in ``inference`` and reaches workers through
``workers/image.py`` and ``workers/video.py``. A regression test asserts that no
scoring logic returns to the worker package.
"""


from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime

from contracts.branch import BRANCH_NAMES, BranchResult
from contracts.common import (
    DEFAULT_LOG_ODDS_MARGIN,
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
    BranchId,
    BranchStatus,
    JobStage,
    MediaKind,
    ResolutionClass,
    Verdict,
)
from contracts.evidence import EvidenceBundle
from contracts.provenance import EMPTY_PROVENANCE
from contracts.video import FrameScore, SuspectSegment, TopFrame, VideoAnalysis
from inference.postprocess import decide
from workers.protocol import ProgressReporter
from workers.types import JobRequest, PermanentError

STUB_MODEL_VERSION = "stub-0"

THRESHOLD = ThresholdInfo.at_margin(
    value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
)

MIN_FACE_PX = 64
"""Below this the quality gate declines. Mirrors the real gate's intent: a face
too small to carry forensic detail is not judged, it is abstained on."""

VIDEO_FRAMES_SAMPLED = 48
PROGRESS_BATCH = 8
"""Frames per progress call.

Deliberately batched. Calling ``progress`` per frame would renew the lease 48
times for a job that needs it perhaps twice, turning a liveness signal into
queue traffic — and on a Redis backend, 48 round trips per job at any real
throughput is a measurable load for no benefit.
"""


# --------------------------------------------------------------------------
# Deterministic pseudo-scoring
# --------------------------------------------------------------------------


def _digest(data: bytes, salt: str = "") -> str:
    return hashlib.sha256(salt.encode() + data).hexdigest()


def _unit(data: bytes, salt: str = "") -> float:
    """A stable float in [0, 1) from the content. Not random — reproducible."""
    return int(_digest(data, salt)[:12], 16) / float(0x1000000000000)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _resolution_class(width: int, height: int) -> ResolutionClass:
    short = min(width, height)
    if short < 256:
        return ResolutionClass.TINY
    if short < 480:
        return ResolutionClass.LOW
    if short < 1080:
        return ResolutionClass.STANDARD
    return ResolutionClass.HIGH


def _derive_dimensions(data: bytes) -> tuple[int, int]:
    widths = (320, 640, 1280, 1920, 3840)
    w = widths[int(_unit(data, "w") * len(widths)) % len(widths)]
    return w, int(w * 9 / 16)


def _derive_face_px(data: bytes, width: int) -> int:
    """Deterministic face size. Occasionally below the gate, so the abstention
    path is reachable from real input rather than only from a crafted test."""
    return int(40 + _unit(data, "face") * max(64, width // 3))


def _branches(p_fake: float, *, face_present: bool) -> list[BranchResult]:
    """Three branches. P is always non-negative — the one-sided provenance rule.

    Absence of provenance contributes exactly zero here, as everywhere: a stub
    that emitted a negative P logit would train the rest of the system to
    tolerate one.

    ``face_present=False`` reweights branch B to 0.94 and keeps it scored — the
    whole-image branch B that R22 records as the frozen-architecture target, not
    the shipped behaviour. Every call site passes ``face_present=True``; the
    branch is dead today and kept as the target shape. ``docs/BRANCH_B_SEMANTICS.md``.
    """
    out: list[BranchResult] = []
    if face_present:
        out.append(
            BranchResult(
                id=BranchId.A,
                name=BRANCH_NAMES[BranchId.A],
                status=BranchStatus.SCORED,
                logit=_logit(p_fake),
                p=p_fake,
                weight=0.72,
                version=STUB_MODEL_VERSION,
            )
        )
    else:
        out.append(
            BranchResult(
                id=BranchId.A,
                name=BRANCH_NAMES[BranchId.A],
                status=BranchStatus.ABSTAINED,
                abstain_reason=AbstainReason.NO_FACE_DETECTED,
                version=STUB_MODEL_VERSION,
            )
        )

    semantic = min(0.99, max(0.01, p_fake * 0.85 + 0.05))
    out.append(
        BranchResult(
            id=BranchId.B,
            name=BRANCH_NAMES[BranchId.B],
            status=BranchStatus.SCORED,
            logit=_logit(semantic),
            p=semantic,
            weight=0.24 if face_present else 0.94,
            version=STUB_MODEL_VERSION,
        )
    )
    out.append(
        BranchResult(
            id=BranchId.P,
            name=BRANCH_NAMES[BranchId.P],
            status=BranchStatus.SCORED,
            logit=0.0,
            p=0.5,
            weight=0.04,
            version=STUB_MODEL_VERSION,
        )
    )
    return out


def _model_info() -> ModelInfo:
    return ModelInfo(
        version=STUB_MODEL_VERSION,
        branch_versions={b.value: STUB_MODEL_VERSION for b in BranchId},
        calibration_temp=1.0,
    )


def _abstained_branches(reason: AbstainReason) -> list[BranchResult]:
    return [
        BranchResult(
            id=bid,
            name=BRANCH_NAMES[bid],
            status=BranchStatus.ABSTAINED,
            abstain_reason=reason
            if bid is not BranchId.P
            else AbstainReason.NO_PROVENANCE_DATA,
            version=STUB_MODEL_VERSION,
        )
        for bid in BranchId
    ]


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------


class StubImageHandler:
    """Image analysis. Deterministic, no model."""

    media_kind = MediaKind.IMAGE.value
    name = "image-stub"

    async def handle(self, request: JobRequest, progress: ProgressReporter) -> EvidenceBundle:
        if not request.media:
            raise PermanentError("The uploaded file is empty.")
        if request.spec.media_kind is not MediaKind.IMAGE:
            raise PermanentError(
                f"This handler analyses images; the job declares {request.spec.media_kind.value}."
            )

        await progress(0.25, stage=JobStage.DETECTING)

        data = request.media
        width, height = _derive_dimensions(data)
        face_px = _derive_face_px(data, width)
        started = datetime.now(UTC)

        media = MediaInfo(
            kind=MediaKind.IMAGE,
            sha256=request.spec.sha256,
            size_bytes=len(data),
            mime=request.spec.declared_mime,
            width=width,
            height=height,
        )

        # The quality gate runs before any scoring, exactly as the real pipeline
        # will. A face too small to carry forensic detail produces an answer of
        # "insufficient evidence" — a success, not a failure.
        if face_px < MIN_FACE_PX:
            return EvidenceBundle(
                job_id=request.spec.job_id,
                created_at=started,
                media=media,
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                p_fake=None,
                interval=None,
                threshold=THRESHOLD,
                abstain_reason=AbstainReason.FACE_TOO_SMALL,
                quality=QualityReport(
                    face_px=face_px,
                    resolution_class=_resolution_class(width, height),
                    gate_passed=False,
                    gate_reason=AbstainReason.FACE_TOO_SMALL,
                ),
                branches=_abstained_branches(AbstainReason.FACE_TOO_SMALL),
                faces=[
                    FaceTrack(
                        track_id=0,
                        bbox=BoundingBox(x=0, y=0, width=face_px, height=face_px),
                        frames_seen=1,
                        status=AbstainReason.FACE_TOO_SMALL,
                    )
                ],
                provenance=EMPTY_PROVENANCE,
                model=_model_info(),
                timing=Timing(
                    queued_ms=0, preprocess_ms=5, inference_ms=0, total_ms=5
                ),
            )

        await progress(0.7, stage=JobStage.SCORING)

        p_fake = _unit(data, "score")
        # decide() is the single source of truth for the verdict. This module
        # used to re-implement it as `p_fake >= THRESHOLD.value`, which is a
        # fourth copy of the rule and one that never knew the abstention band
        # existed — so the stub reported "manipulated" for scores the real
        # pipeline declines to decide.
        verdict, band_reason = decide(p_fake, THRESHOLD)
        abstained = verdict is Verdict.INSUFFICIENT_EVIDENCE
        margin = 0.04 + _unit(data, "band") * 0.05

        return EvidenceBundle(
            job_id=request.spec.job_id,
            created_at=started,
            media=media,
            verdict=verdict,
            p_fake=None if abstained else p_fake,
            interval=None if abstained else (max(0.0, p_fake - margin), min(1.0, p_fake + margin)),
            abstain_reason=band_reason,
            threshold=THRESHOLD,
            quality=QualityReport(
                jpeg_q_est=60.0 + _unit(data, "q") * 40.0,
                face_px=face_px,
                blur_score=_unit(data, "blur") * 0.5,
                resolution_class=_resolution_class(width, height),
                gate_passed=True,
            ),
            branches=_branches(p_fake, face_present=True),
            disagreement=abs(p_fake - (p_fake * 0.85 + 0.05)),
            faces=[
                FaceTrack(
                    track_id=0,
                    bbox=BoundingBox(
                        x=width // 4, y=height // 5, width=face_px, height=face_px
                    ),
                    frames_seen=1,
                    p_fake=p_fake,
                )
            ],
            provenance=EMPTY_PROVENANCE,
            model=_model_info(),
            timing=Timing(queued_ms=0, preprocess_ms=20, inference_ms=60, total_ms=80),
        )


class StubVideoHandler:
    """Video analysis. Deterministic, no model.

    Progress is reported once per ``PROGRESS_BATCH`` frames rather than per
    frame — see the constant's note.
    """

    media_kind = MediaKind.VIDEO.value
    name = "video-stub"

    def __init__(self, frames: int = VIDEO_FRAMES_SAMPLED, batch: int = PROGRESS_BATCH) -> None:
        self.frames = frames
        self.batch = batch

    async def handle(self, request: JobRequest, progress: ProgressReporter) -> EvidenceBundle:
        if not request.media:
            raise PermanentError("The uploaded file is empty.")
        if request.spec.media_kind is not MediaKind.VIDEO:
            raise PermanentError(
                f"This handler analyses video; the job declares {request.spec.media_kind.value}."
            )

        data = request.media
        started = datetime.now(UTC)
        width, height = _derive_dimensions(data)
        face_px = _derive_face_px(data, width)
        duration_s = 4.0 + _unit(data, "dur") * 26.0
        fps = 25.0

        await progress(0.1, stage=JobStage.SAMPLING)

        peak = _unit(data, "score")
        centre = 0.25 + _unit(data, "centre") * 0.5
        spread = 0.10 + _unit(data, "spread") * 0.12
        step_ms = int(duration_s * 1000 / self.frames)

        frame_scores: list[FrameScore] = []
        for i in range(self.frames):
            x = i / max(self.frames - 1, 1)
            bump = math.exp(-(((x - centre) / spread) ** 2))
            p = round(0.04 + (peak - 0.04) * bump, 10) if peak > 0.04 else 0.04
            frame_scores.append(
                FrameScore(frame_idx=i * 6, t_ms=i * step_ms, p=min(max(p, 0.0), 1.0))
            )
            # One progress call per batch, not per frame.
            if (i + 1) % self.batch == 0:
                await progress(
                    0.1 + 0.8 * (i + 1) / self.frames, stage=JobStage.SCORING
                )

        await progress(0.95, stage=JobStage.TEMPORAL)

        # Top-k mean pooling — the v1 baseline. Mean over all frames would dilute
        # a short manipulated segment into nothing.
        k = max(1, self.frames // 8)
        top_k = sorted((f.p for f in frame_scores), reverse=True)[:k]
        p_fake = sum(top_k) / len(top_k)

        hot = [f for f in frame_scores if f.p >= THRESHOLD.band_hi]
        segments: list[SuspectSegment] = []
        if hot:
            segments.append(
                SuspectSegment(
                    start_ms=hot[0].t_ms,
                    end_ms=hot[-1].t_ms,
                    peak_p=max(f.p for f in hot),
                    mean_p=round(sum(f.p for f in hot) / len(hot), 10),
                    frame_count=len(hot),
                )
            )

        top_frames = [
            TopFrame(frame_idx=f.frame_idx, t_ms=f.t_ms, p=f.p, track_id=0)
            for f in sorted(frame_scores, key=lambda f: -f.p)[:3]
        ]

        verdict, band_reason = decide(p_fake, THRESHOLD)
        abstained = verdict is Verdict.INSUFFICIENT_EVIDENCE
        margin = 0.05 + _unit(data, "band") * 0.05

        return EvidenceBundle(
            job_id=request.spec.job_id,
            created_at=started,
            media=MediaInfo(
                kind=MediaKind.VIDEO,
                sha256=request.spec.sha256,
                size_bytes=len(data),
                mime=request.spec.declared_mime,
                width=width,
                height=height,
                duration_s=duration_s,
                frame_count=int(duration_s * fps),
                fps=fps,
            ),
            verdict=verdict,
            p_fake=None if abstained else p_fake,
            interval=None if abstained else (max(0.0, p_fake - margin), min(1.0, p_fake + margin)),
            abstain_reason=band_reason,
            threshold=THRESHOLD,
            quality=QualityReport(
                jpeg_q_est=55.0 + _unit(data, "q") * 40.0,
                face_px=face_px,
                blur_score=_unit(data, "blur") * 0.5,
                resolution_class=_resolution_class(width, height),
                gate_passed=True,
            ),
            branches=_branches(p_fake, face_present=True),
            disagreement=abs(p_fake - (p_fake * 0.85 + 0.05)),
            faces=[
                FaceTrack(
                    track_id=0,
                    bbox=BoundingBox(
                        x=width // 4, y=height // 5, width=face_px, height=face_px
                    ),
                    frames_seen=self.frames,
                    p_fake=p_fake,
                )
            ],
            video=VideoAnalysis(
                frames_sampled=self.frames,
                frames_scored=self.frames,
                sampling="uniform",
                pooling="topk_mean",
                frame_scores=frame_scores,
                segments=segments,
                top_frames=top_frames,
            ),
            provenance=EMPTY_PROVENANCE,
            model=_model_info(),
            timing=Timing(queued_ms=0, preprocess_ms=200, inference_ms=900, total_ms=1100),
        )
