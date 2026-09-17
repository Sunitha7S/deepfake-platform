"""Canonical example payloads, one per public model.

Examples are **constructed as real model instances**, never written as loose
dicts. Two consequences, both deliberate:

* An invalid example is an ImportError, not a stale line in the API docs. The
  classic rot — documentation showing a payload the server would reject — cannot
  happen here.
* Response examples reuse the fixtures verbatim, so the documented payload and
  the payload the frontend develops against are the same bytes.

``schema.py`` injects these into the generated JSON Schema at export time rather
than pinning them into ``model_config``. That keeps the model modules free of a
dependency on fixtures, which would otherwise be circular.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from contracts import fixtures as fx
from contracts.api import (
    AnalyzeAcceptedResponse,
    AnalyzeRequest,
    BranchHealth,
    CreateUploadRequest,
    CreateUploadResponse,
    HealthResponse,
    PresignedUpload,
)
from contracts.enums import JobStage, JobState, MediaKind, UploadMethod
from contracts.errors import ApiError, ErrorCode, ErrorResponse
from contracts.evidence import EvidenceBundle
from contracts.job import AnalysisOptions, JobProgress, JobSpec, JobStatus, ProgressEvent

_EXPIRY = datetime(2026, 8, 23, 14, 45, 0, tzinfo=UTC)
_SHA = "3b1f8c2e9a7d4b6055e1c8f3a2d90b47e6c5148a9f0b3d2e7c614a58d093f2be"


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


def create_upload_request() -> CreateUploadRequest:
    return CreateUploadRequest(
        sha256=_SHA,
        size_bytes=2_184_233,
        content_type="image/jpeg",
        media_kind=MediaKind.IMAGE,
        filename_hint="portrait.jpg",
    )


def create_upload_request_video() -> CreateUploadRequest:
    return CreateUploadRequest(
        sha256=_SHA,
        size_bytes=41_882_004,
        content_type="video/mp4",
        media_kind=MediaKind.VIDEO,
        filename_hint="clip.mp4",
    )


def analyze_request() -> AnalyzeRequest:
    return AnalyzeRequest(
        upload_id="u_01J8ZQ4M7X2K9V3P",
        options=AnalysisOptions(explain=True, max_frames=None),
    )


def analysis_options() -> AnalysisOptions:
    return AnalysisOptions(explain=True, max_frames=48)


# --------------------------------------------------------------------------
# Upload negotiation responses
# --------------------------------------------------------------------------


def presigned_upload() -> PresignedUpload:
    return PresignedUpload(
        url="https://storage.example.com/uploads/u_01J8ZQ4M7X2K9V3P?X-Signature=abc123",
        method=UploadMethod.PUT,
        headers={"Content-Type": "image/jpeg"},
        expires_at=_EXPIRY,
    )


def create_upload_response_miss() -> CreateUploadResponse:
    """Cache miss: the client uploads to the returned URL."""
    return CreateUploadResponse(
        upload_id="u_01J8ZQ4M7X2K9V3P",
        cache_hit=False,
        upload=presigned_upload(),
    )


def create_upload_response_hit() -> CreateUploadResponse:
    """Cache hit: the answer arrives before a single byte is uploaded."""
    return CreateUploadResponse(
        upload_id="u_01J8ZQ4M7X2K9V3P",
        cache_hit=True,
        cached_result=fx.image_manipulated(),
    )


def analyze_accepted_response() -> AnalyzeAcceptedResponse:
    jid = "00000000-0000-0000-0000-000000000006"
    return AnalyzeAcceptedResponse(
        job_id=UUID(jid),
        state=JobState.QUEUED,
        status_url=f"/api/v1/jobs/{jid}",
        events_url=f"/api/v1/events/{jid}",
        result_url=f"/api/v1/jobs/{jid}/result",
        queue_position=3,
        estimated_wait_s=12,
    )


# --------------------------------------------------------------------------
# Job lifecycle
# --------------------------------------------------------------------------


def job_spec() -> JobSpec:
    return JobSpec(
        job_id=UUID("00000000-0000-0000-0000-000000000006"),
        media_kind=MediaKind.VIDEO,
        storage_key="uploads/2026/08/23/u_01J8ZQ4M7X2K9V3P",
        sha256=_SHA,
        size_bytes=41_882_004,
        declared_mime="video/mp4",
        model_version="1.0.0-dev",
        idempotency_key=f"{_SHA}:1.0.0-dev",
        submitted_at=fx.FIXED_TIME,
        principal_id="anonymous",
    )


def job_progress() -> JobProgress:
    return JobProgress(stage=JobStage.SCORING, percent=75, message="Scoring 48 frames")


def progress_event() -> ProgressEvent:
    return ProgressEvent(
        event="progress",
        job_id=UUID("00000000-0000-0000-0000-000000000006"),
        status=fx.job_running(),
        at=fx.FIXED_TIME,
    )


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


def branch_health() -> BranchHealth:
    return BranchHealth(id="A", loaded=True, version="convnextv2t-sbi-1.0")


def health_response() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version="1.0.0-dev",
        branches=[
            BranchHealth(id="A", loaded=True, version="convnextv2t-sbi-1.0"),
            BranchHealth(id="B", loaded=True, version="dinov2b-probe-1.0"),
            BranchHealth(id="P", loaded=True, version="c2pa-1.0"),
        ],
        device="cpu",
        queue_depth={"image": 0, "video": 2},
        uptime_s=84_210,
    )


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def api_error() -> ApiError:
    return ApiError.of(ErrorCode.FILE_TOO_LARGE, field="size_bytes")


def error_response_deterministic() -> ErrorResponse:
    """A failure that will never succeed on retry."""
    return ErrorResponse(error=ApiError.of(ErrorCode.DECODE_FAILED, detail="moov atom not found"))


def error_response_transient() -> ErrorResponse:
    """A failure worth retrying — note ``retryable: true``."""
    return ErrorResponse(error=ApiError.of(ErrorCode.MODEL_UNAVAILABLE, retry_after_s=5))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

EXAMPLES: dict[type[BaseModel], list[tuple[str, Any]]] = {
    # requests
    CreateUploadRequest: [
        ("image", create_upload_request),
        ("video", create_upload_request_video),
    ],
    AnalyzeRequest: [("default", analyze_request)],
    AnalysisOptions: [("default", analysis_options)],
    # upload negotiation
    PresignedUpload: [("default", presigned_upload)],
    CreateUploadResponse: [
        ("cache_miss", create_upload_response_miss),
        ("cache_hit", create_upload_response_hit),
    ],
    AnalyzeAcceptedResponse: [("default", analyze_accepted_response)],
    # results — reuse the fixtures verbatim
    EvidenceBundle: [
        ("image_manipulated", fx.image_manipulated),
        ("image_authentic", fx.image_authentic),
        ("image_authentic_signed", fx.image_authentic_signed),
        ("image_no_face", fx.image_no_face),
        ("image_insufficient_evidence", fx.image_insufficient_evidence),
        ("video_manipulated", fx.video_manipulated),
        ("video_authentic", fx.video_authentic),
        ("video_insufficient_evidence", fx.video_insufficient_evidence),
    ],
    # job lifecycle
    JobSpec: [("default", job_spec)],
    JobProgress: [("default", job_progress)],
    JobStatus: [
        ("running", fx.job_running),
        ("succeeded", fx.job_succeeded),
        ("failed", fx.job_failed),
    ],
    ProgressEvent: [("default", progress_event)],
    # health
    BranchHealth: [("default", branch_health)],
    HealthResponse: [("default", health_response)],
    # errors
    ApiError: [("default", api_error)],
    ErrorResponse: [
        ("deterministic", error_response_deterministic),
        ("transient", error_response_transient),
    ],
}
"""Model → [(label, builder)]. ``schema.py`` and the tests both read this."""


def examples_for(model: type[BaseModel]) -> list[dict[str, Any]]:
    """JSON-mode payloads for one model. Empty list when none are registered."""
    return [fn().model_dump(mode="json") for _, fn in EXAMPLES.get(model, [])]


def labelled_examples_for(model: type[BaseModel]) -> dict[str, dict[str, Any]]:
    """Named payloads, for OpenAPI's ``examples`` map and for the docs pages."""
    return {label: fn().model_dump(mode="json") for label, fn in EXAMPLES.get(model, [])}


def all_examples() -> dict[str, dict[str, dict[str, Any]]]:
    """Every example, keyed by model name then label."""
    return {model.__name__: labelled_examples_for(model) for model in EXAMPLES}
