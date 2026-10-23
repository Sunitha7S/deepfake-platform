"""Closed vocabularies shared by every layer of the platform.

Every string that crosses a process boundary — API, queue, storage, browser —
comes from an enum in this file. Nothing downstream should ever compare against
a bare string literal.

Frozen architecture note: BranchId has no ``C``. The frequency branch was folded
into Branch A as additional input channels (SRM/Bayar residuals), so it is no
longer a separate scorer and must not reappear as one.
"""

from __future__ import annotations

from enum import StrEnum

CONTRACT_VERSION = "1.0.0"
"""Semantic version of the EvidenceBundle contract.

Bump MAJOR when a *stable-tier* field changes shape or meaning. Stable-tier
fields carry compatibility guarantees; evolving-tier fields may change within a
MAJOR version. See ``EvidenceBundle`` for which fields sit in which tier.
"""


class MediaKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"


class Verdict(StrEnum):
    """The three-state output. Never collapse this to a boolean.

    ``INSUFFICIENT_EVIDENCE`` is a successful result, not a failure. It is
    returned when the quality gate declines the input or when no branch was able
    to score it.
    """

    AUTHENTIC = "authentic"
    MANIPULATED = "manipulated"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class BranchId(StrEnum):
    """Evidence sources. ``C`` is intentionally absent — see module docstring."""

    A = "A"  # spatial    — ConvNeXt V2-T over RGB + SRM/Bayar residual channels
    B = "B"  # semantic   — DINOv2-B/14 frozen + probe; whole-image is the target, scores A's tensor today (R22)
    P = "P"  # provenance  — C2PA + EXIF, not learned, one-sided
    # Branch input scope: see contracts/branch.py BRANCH_DESCRIPTIONS. Today
    # every branch scores one tensor — the face crop (image) or the whole frame
    # (video) — because ModelBackend.score is single-input. R22.


class BranchStatus(StrEnum):
    SCORED = "scored"
    ABSTAINED = "abstained"
    UNAVAILABLE = "unavailable"


class AbstainReason(StrEnum):
    """Why a branch declined to vote, or why the whole bundle abstained.

    These are *outcomes*, not errors. They never map to an HTTP error status and
    never render as a failure in the UI. Contrast with ``ErrorCode``.
    """

    NO_FACE_DETECTED = "no_face_detected"
    FACE_TOO_SMALL = "face_too_small"
    QUALITY_BELOW_FLOOR = "quality_below_floor"
    EXCESSIVE_COMPRESSION = "excessive_compression"
    TOO_FEW_FRAMES = "too_few_frames"
    NO_PROVENANCE_DATA = "no_provenance_data"
    BRANCH_NOT_APPLICABLE = "branch_not_applicable"
    BRANCH_ERROR = "branch_error"
    SCORE_IN_ABSTENTION_BAND = "score_in_abstention_band"


class ResolutionClass(StrEnum):
    TINY = "tiny"  # < 256 px on the short edge
    LOW = "low"  # < 480
    STANDARD = "standard"  # < 1080
    HIGH = "high"  # >= 1080


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)


class JobStage(StrEnum):
    """Pipeline stages, reported as real progress rather than a fake animation.

    Image jobs pass through a subset; video jobs use all of them.
    """

    VALIDATING = "validating"
    DECODING = "decoding"
    SAMPLING = "sampling"  # video only
    DETECTING = "detecting"
    TRACKING = "tracking"  # video only
    SCORING = "scoring"
    TEMPORAL = "temporal"  # video only
    FUSING = "fusing"
    EXPLAINING = "explaining"
    FINALIZING = "finalizing"


class ArtifactKind(StrEnum):
    GRADCAM = "gradcam"
    SPECTRUM = "spectrum"
    FRAME_CURVE = "frame_curve"
    THUMBNAIL = "thumbnail"


class UploadMethod(StrEnum):
    PUT = "PUT"
    POST = "POST"


class ErrorCode(StrEnum):
    """Genuine failures. Everything here means the request could not be served.

    An input the model cannot judge is NOT an error — it is a successful job
    whose verdict is ``insufficient_evidence`` with an ``AbstainReason``.
    """

    # validation — rejected before any work
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    FILE_TOO_LARGE = "file_too_large"
    FILE_EMPTY = "file_empty"
    DURATION_TOO_LONG = "duration_too_long"
    RESOLUTION_TOO_LARGE = "resolution_too_large"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    INVALID_REQUEST = "invalid_request"

    # upload lifecycle
    UPLOAD_NOT_FOUND = "upload_not_found"
    UPLOAD_EXPIRED = "upload_expired"
    UPLOAD_INCOMPLETE = "upload_incomplete"

    # processing — deterministic, never retried
    DECODE_FAILED = "decode_failed"
    UNSUPPORTED_CODEC = "unsupported_codec"

    # processing — transient, retried
    DECODE_TIMEOUT = "decode_timeout"
    MODEL_UNAVAILABLE = "model_unavailable"
    RESOURCE_EXHAUSTED = "resource_exhausted"

    # lookup
    JOB_NOT_FOUND = "job_not_found"
    RESULT_NOT_READY = "result_not_ready"

    # platform
    RATE_LIMITED = "rate_limited"
    INTERNAL_ERROR = "internal_error"


DETERMINISTIC_ERRORS: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.UNSUPPORTED_MEDIA_TYPE,
        ErrorCode.FILE_TOO_LARGE,
        ErrorCode.FILE_EMPTY,
        ErrorCode.DURATION_TOO_LONG,
        ErrorCode.RESOLUTION_TOO_LARGE,
        ErrorCode.CHECKSUM_MISMATCH,
        ErrorCode.INVALID_REQUEST,
        ErrorCode.DECODE_FAILED,
        ErrorCode.UNSUPPORTED_CODEC,
        ErrorCode.UPLOAD_NOT_FOUND,
        ErrorCode.UPLOAD_EXPIRED,
    }
)
"""Retrying these produces the same failure. The worker must not re-enqueue them."""

TRANSIENT_ERRORS: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.DECODE_TIMEOUT,
        ErrorCode.MODEL_UNAVAILABLE,
        ErrorCode.RESOURCE_EXHAUSTED,
        ErrorCode.RATE_LIMITED,
        ErrorCode.INTERNAL_ERROR,
    }
)
"""The same request can succeed later: retry with backoff, honouring
``retry_after_s`` where it is set (``rate_limited`` always sets it)."""


def is_retryable(code: ErrorCode) -> bool:
    """Whether a client retrying the same request could succeed.

    The single source of ``ApiError.retryable`` on the wire (via
    ``ApiError.of``). Worker re-delivery does not read it: the runner decides
    from the handler's exception (``workers.types.classify``).
    """
    return code in TRANSIENT_ERRORS
