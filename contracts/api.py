"""HTTP request and response schemas.

Upload flow is direct-to-storage, frozen at architecture review: the browser
hashes the file, asks for a presigned URL, PUTs the bytes straight to object
storage, then asks for analysis. **File bytes never pass through the API.** That
removes the worst scaling property of a Python ASGI service and makes a cache hit
cost zero bytes of upload.

    1. POST /api/v1/uploads   → presigned URL, or an immediate cached result
    2. PUT  <presigned url>   → bytes go to storage, API not involved
    3. POST /api/v1/analyze   → 202 with a job id
    4. GET  /api/v1/events/…  → SSE progress
    5. GET  /api/v1/jobs/…/result → EvidenceBundle
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, model_validator

from contracts.common import Base
from contracts.enums import JobState, MediaKind, UploadMethod
from contracts.evidence import EvidenceBundle
from contracts.job import AnalysisOptions

# Caps enforced at request time, before a presigned URL is ever issued.
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_VIDEO_DURATION_S = 120

ALLOWED_IMAGE_MIME = ("image/jpeg", "image/png", "image/webp")
ALLOWED_VIDEO_MIME = ("video/mp4", "video/quicktime", "video/webm")


# --------------------------------------------------------------------------
# 1 · Upload negotiation
# --------------------------------------------------------------------------


class CreateUploadRequest(Base):
    """Asks for somewhere to put a file. Carries no bytes.

    ``sha256`` is computed in the browser before upload. That is what allows the
    server to answer with a finished result and skip the transfer entirely — on a
    200 MB video that is the whole perceived latency.
    """

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)
    content_type: str = Field(description="Client-declared; re-sniffed from magic bytes later.")
    media_kind: MediaKind
    filename_hint: str | None = Field(
        default=None,
        max_length=255,
        description="Display only. Never used to build a storage path, never echoed back.",
    )

    @model_validator(mode="after")
    def _within_caps(self) -> CreateUploadRequest:
        if self.media_kind is MediaKind.IMAGE:
            if self.content_type not in ALLOWED_IMAGE_MIME:
                raise ValueError(f"content_type {self.content_type} not allowed for images")
            if self.size_bytes > MAX_IMAGE_BYTES:
                raise ValueError("image exceeds 25 MB")
        else:
            if self.content_type not in ALLOWED_VIDEO_MIME:
                raise ValueError(f"content_type {self.content_type} not allowed for videos")
            if self.size_bytes > MAX_VIDEO_BYTES:
                raise ValueError("video exceeds 200 MB")
        return self


class PresignedUpload(Base):
    url: str
    method: UploadMethod = UploadMethod.PUT
    headers: dict[str, str] = Field(
        default_factory=dict, description="Must be sent verbatim with the PUT."
    )
    expires_at: datetime


class CreateUploadResponse(Base):
    """Either somewhere to upload, or the answer already.

    Exactly one of ``upload`` / ``cached_result`` is set. A cache hit short-
    circuits the entire pipeline.
    """

    upload_id: str
    cache_hit: bool
    upload: PresignedUpload | None = None
    cached_result: EvidenceBundle | None = None

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> CreateUploadResponse:
        if self.cache_hit:
            if self.cached_result is None or self.upload is not None:
                raise ValueError("cache_hit requires cached_result and forbids upload")
        else:
            if self.upload is None or self.cached_result is not None:
                raise ValueError("a cache miss requires upload and forbids cached_result")
        return self


# --------------------------------------------------------------------------
# 2 · Analysis
# --------------------------------------------------------------------------


class AnalyzeRequest(Base):
    """Submitted after the bytes have landed in storage."""

    upload_id: str
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class AnalyzeAcceptedResponse(Base):
    """202. The job is queued; nothing has been analysed yet."""

    job_id: UUID
    state: JobState = JobState.QUEUED
    status_url: str
    events_url: str
    result_url: str
    queue_position: int | None = Field(default=None, ge=0)
    estimated_wait_s: int | None = Field(default=None, ge=0)


# --------------------------------------------------------------------------
# 3 · Health
# --------------------------------------------------------------------------


class BranchHealth(Base):
    id: str
    loaded: bool
    version: str | None = None


class HealthResponse(Base):
    """Readiness probe. Reports whether models are actually resident, not just
    whether the process is up — a broken deploy must fail readiness and take no
    traffic."""

    status: str = Field(description="ok | degraded | starting")
    model_version: str | None = None
    branches: list[BranchHealth] = Field(default_factory=list)
    device: str = Field(default="cpu")
    queue_depth: dict[str, int] = Field(default_factory=dict)
    uptime_s: int = Field(ge=0)
