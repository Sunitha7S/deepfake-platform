"""Contract definitions shared by every layer of the deepfake detection platform.

This package is the keystone: it is built before anything that uses it, so the
frontend, backend and workers can be developed in parallel against fixtures while
the ML path is still blocked on datasets and GPUs.

It depends on pydantic and nothing else. No torch, no numpy, no I/O.
"""

from contracts.api import (
    AnalyzeAcceptedResponse,
    AnalyzeRequest,
    BranchHealth,
    CreateUploadRequest,
    CreateUploadResponse,
    HealthResponse,
    PresignedUpload,
)
from contracts.branch import BRANCH_DESCRIPTIONS, BRANCH_NAMES, BranchResult
from contracts.common import (
    Artifact,
    Base,
    BoundingBox,
    FaceTrack,
    MediaInfo,
    ModelInfo,
    QualityReport,
    ThresholdInfo,
    Timing,
)
from contracts.enums import (
    CONTRACT_VERSION,
    DETERMINISTIC_ERRORS,
    TRANSIENT_ERRORS,
    AbstainReason,
    ArtifactKind,
    BranchId,
    BranchStatus,
    ErrorCode,
    JobStage,
    JobState,
    MediaKind,
    ResolutionClass,
    UploadMethod,
    Verdict,
    is_retryable,
)
from contracts.errors import DEFAULT_MESSAGES, HTTP_STATUS, ApiError, ErrorResponse
from contracts.evidence import EvidenceBundle
from contracts.job import (
    STAGE_WEIGHTS_IMAGE,
    STAGE_WEIGHTS_VIDEO,
    AnalysisOptions,
    JobProgress,
    JobSpec,
    JobStatus,
    ProgressEvent,
)
from contracts.provenance import (
    EMPTY_PROVENANCE,
    C2paReport,
    ExifReport,
    ProvenanceFinding,
    ProvenanceReport,
)
from contracts.video import FrameScore, SuspectSegment, TopFrame, VideoAnalysis

__all__ = [
    "CONTRACT_VERSION",
    "AbstainReason",
    "AnalysisOptions",
    "AnalyzeAcceptedResponse",
    "AnalyzeRequest",
    "ApiError",
    "Artifact",
    "ArtifactKind",
    "BRANCH_DESCRIPTIONS",
    "BRANCH_NAMES",
    "Base",
    "BoundingBox",
    "BranchHealth",
    "BranchId",
    "BranchResult",
    "BranchStatus",
    "C2paReport",
    "CreateUploadRequest",
    "CreateUploadResponse",
    "DEFAULT_MESSAGES",
    "DETERMINISTIC_ERRORS",
    "EMPTY_PROVENANCE",
    "ErrorCode",
    "ErrorResponse",
    "EvidenceBundle",
    "ExifReport",
    "FaceTrack",
    "FrameScore",
    "HTTP_STATUS",
    "HealthResponse",
    "JobProgress",
    "JobSpec",
    "JobStage",
    "JobState",
    "JobStatus",
    "MediaInfo",
    "MediaKind",
    "ModelInfo",
    "PresignedUpload",
    "ProgressEvent",
    "ProvenanceFinding",
    "ProvenanceReport",
    "QualityReport",
    "ResolutionClass",
    "STAGE_WEIGHTS_IMAGE",
    "STAGE_WEIGHTS_VIDEO",
    "SuspectSegment",
    "TRANSIENT_ERRORS",
    "ThresholdInfo",
    "Timing",
    "TopFrame",
    "UploadMethod",
    "Verdict",
    "VideoAnalysis",
    "is_retryable",
]
