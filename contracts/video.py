"""Video-specific payload: the per-frame curve, suspect segments, top frames.

These are *measurements* rather than attributions, which is why the Results page
ranks them above the Grad-CAM panel in trustworthiness.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from contracts.common import Base


class FrameScore(Base):
    """One point on the timeline curve."""

    frame_idx: int = Field(ge=0)
    t_ms: int = Field(ge=0)
    p: float = Field(ge=0.0, le=1.0)


class SuspectSegment(Base):
    """A contiguous run above the hysteresis threshold.

    Produced by hysteresis rather than a single cut so that a curve hovering at
    the boundary does not shatter into dozens of one-frame segments.
    """

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    peak_p: float = Field(ge=0.0, le=1.0)
    mean_p: float = Field(ge=0.0, le=1.0)
    frame_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> SuspectSegment:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")
        if self.mean_p > self.peak_p:
            raise ValueError("mean_p cannot exceed peak_p")
        return self


class TopFrame(Base):
    """A frame surfaced as a thumbnail, with its own attribution artifact."""

    frame_idx: int = Field(ge=0)
    t_ms: int = Field(ge=0)
    p: float = Field(ge=0.0, le=1.0)
    track_id: int | None = Field(
        default=None,
        ge=0,
        description=(
            "The faces[] track this frame's score is attributed to. **Null means "
            "no face was attributed** — which is every frame the current video "
            "path produces, because it scores whole frames and runs no face "
            "detection. A non-null value must name a track the bundle actually "
            "publishes; EvidenceBundle refuses a dangling reference."
        ),
    )
    thumbnail_ref: str | None = None
    gradcam_ref: str | None = None


class VideoAnalysis(Base):
    """Present iff ``media.kind == video``.

    **The current video path runs no face detection.** It samples frames,
    resizes them whole to the model input, and scores them; it does not detect,
    crop, track or attribute faces. So a video bundle publishes an empty
    ``faces[]`` and a null ``quality.face_px``, and every ``top_frames[].track_id``
    is null. That is R18, and the frozen architecture's face stage for video is
    SCRFD work that has not landed — see ``docs/VIDEO_FACE_SEMANTICS.md``.

    The fixtures in ``contracts/fixtures.py`` deliberately depict the *target*
    shape, with a populated ``FaceTrack`` and resolving track ids, because they
    are what the frontend builds against. They are not a description of what the
    pipeline emits today.
    """

    frames_sampled: int = Field(ge=1)
    frames_scored: int = Field(
        ge=0,
        description=(
            "Sampled frames that decoded and were scored. Never exceeds "
            "frames_sampled; lower when part of a damaged clip failed to decode. "
            "See the class docstring for what this field used to claim and why "
            "that was wrong (R18)."
        ),
    )
    sampling: str = Field(
        default="uniform",
        description=(
            "How frames were chosen — the ``sampling_rule`` of the "
            "``VideoSelectionPolicy`` in force. Recorded so a result can be "
            "reproduced. The shipped value is plain uniform stride; nothing "
            "adaptive runs."
        ),
    )
    pooling: str = Field(
        default="topk_mean",
        description=(
            "Aggregation over frames. 'topk_mean' is the v1 baseline; 'attention' "
            "once the temporal transformer beats it by a paired-bootstrap margin."
        ),
    )
    frame_scores: list[FrameScore] = Field(default_factory=list)
    segments: list[SuspectSegment] = Field(default_factory=list)
    top_frames: list[TopFrame] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> VideoAnalysis:
        if self.frames_scored > self.frames_sampled:
            raise ValueError("frames_scored cannot exceed frames_sampled")
        idxs = [f.frame_idx for f in self.frame_scores]
        if idxs != sorted(idxs):
            raise ValueError("frame_scores must be ordered by frame_idx")
        if len(set(idxs)) != len(idxs):
            raise ValueError("frame_scores contains duplicate frame_idx")
        return self
