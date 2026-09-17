"""Job lifecycle: what the worker is asked to do, and what it reports back."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, model_validator

from contracts.common import Base
from contracts.enums import JobStage, JobState, MediaKind
from contracts.errors import ApiError

# Stage weights used to turn a stage into a percentage. Rough, but honest —
# far better than an animated bar that means nothing.
STAGE_WEIGHTS_IMAGE: dict[JobStage, int] = {
    JobStage.VALIDATING: 5,
    JobStage.DECODING: 15,
    JobStage.DETECTING: 35,
    JobStage.SCORING: 70,
    JobStage.FUSING: 80,
    JobStage.EXPLAINING: 95,
    JobStage.FINALIZING: 100,
}

STAGE_WEIGHTS_VIDEO: dict[JobStage, int] = {
    # R18: DETECTING and TRACKING are budgeted here and emitted by nothing. The
    # video pipeline's only progress() call passes SCORING, because it runs no
    # face stage. Kept rather than deleted — these weights are the post-SCRFD
    # plan, and removing them would erase it — but a reader must not infer from
    # this table that the stages run. See docs/VIDEO_FACE_SEMANTICS.md.
    JobStage.VALIDATING: 2,
    JobStage.DECODING: 8,
    JobStage.SAMPLING: 15,
    JobStage.DETECTING: 30,
    JobStage.TRACKING: 40,
    JobStage.SCORING: 75,
    JobStage.TEMPORAL: 85,
    JobStage.FUSING: 90,
    JobStage.EXPLAINING: 97,
    JobStage.FINALIZING: 100,
}


class AnalysisOptions(Base):
    """Per-request knobs. All optional; defaults are the shipped configuration."""

    explain: bool = Field(default=True, description="Generate Grad-CAM and spectrum artifacts.")
    max_frames: int | None = Field(
        default=None, ge=8, le=256, description="Override the video frame budget."
    )


class JobSpec(Base):
    """What the API hands the queue. Immutable once enqueued.

    ``idempotency_key`` is ``sha256 + model_version``: the same file analysed by
    the same model is the same job, so a duplicate enqueue returns the original
    rather than repeating the work.
    """

    job_id: UUID
    media_kind: MediaKind
    storage_key: str = Field(description="Object-storage key of the uploaded file.")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)
    declared_mime: str = Field(description="Client-declared. Never trusted; re-sniffed by the worker.")
    model_version: str
    idempotency_key: str
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)
    submitted_at: datetime
    principal_id: str = Field(
        default="anonymous",
        description="Future-ready. Today always 'anonymous'; rate limits key on it.",
    )


class JobProgress(Base):
    stage: JobStage
    percent: int = Field(ge=0, le=100)
    message: str | None = None


class JobStatus(Base):
    """Polled by ``GET /jobs/{id}`` and pushed over SSE. Same shape both ways."""

    job_id: UUID
    state: JobState
    progress: JobProgress | None = None
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempt: int = Field(default=1, ge=1)
    error: ApiError | None = None
    result_url: str | None = Field(
        default=None, description="Set once state is succeeded."
    )

    @model_validator(mode="after")
    def _terminal_state_consistency(self) -> JobStatus:
        if self.state is JobState.FAILED and self.error is None:
            raise ValueError("a failed job must carry an error")
        if self.state is JobState.SUCCEEDED and self.error is not None:
            raise ValueError("a succeeded job must not carry an error")
        if self.state.is_terminal and self.finished_at is None:
            raise ValueError(f"state {self.state} requires finished_at")
        if not self.state.is_terminal and self.finished_at is not None:
            raise ValueError("finished_at must be null while the job is not terminal")
        return self


class ProgressEvent(Base):
    """One SSE frame on ``GET /events/{job_id}``.

    Sent as ``event: <event>`` with this object JSON-encoded as ``data:``.
    """

    event: str = Field(description="One of: state, progress, done, error, heartbeat.")
    job_id: UUID
    status: JobStatus
    at: datetime
