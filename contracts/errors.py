"""Error taxonomy.

Every failure the API can return is one of these, with a message written for a
human. There is no bare 500 and no stack trace on the wire.

What is NOT here: inputs the model cannot judge. "No face detected" and "quality
below floor" are successful jobs whose verdict is ``insufficient_evidence``. They
carry an ``AbstainReason``, never an ``ErrorCode``, and the UI renders them as
answers rather than failures.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from contracts.common import Base
from contracts.enums import ErrorCode, is_retryable

HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: 415,
    ErrorCode.FILE_TOO_LARGE: 413,
    ErrorCode.FILE_EMPTY: 400,
    ErrorCode.DURATION_TOO_LONG: 400,
    ErrorCode.RESOLUTION_TOO_LARGE: 400,
    ErrorCode.CHECKSUM_MISMATCH: 400,
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.UPLOAD_NOT_FOUND: 404,
    ErrorCode.UPLOAD_EXPIRED: 410,
    ErrorCode.UPLOAD_INCOMPLETE: 409,
    ErrorCode.DECODE_FAILED: 422,
    ErrorCode.UNSUPPORTED_CODEC: 415,
    ErrorCode.DECODE_TIMEOUT: 504,
    ErrorCode.MODEL_UNAVAILABLE: 503,
    ErrorCode.RESOURCE_EXHAUSTED: 503,
    ErrorCode.JOB_NOT_FOUND: 404,
    ErrorCode.RESULT_NOT_READY: 409,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.INTERNAL_ERROR: 500,
}

DEFAULT_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: "That file type isn't supported. Use JPEG, PNG, WebP, MP4, MOV or WebM.",
    ErrorCode.FILE_TOO_LARGE: "That file is larger than the limit. Images can be up to 25 MB and videos up to 200 MB.",
    ErrorCode.FILE_EMPTY: "That file is empty.",
    ErrorCode.DURATION_TOO_LONG: "That video is longer than the 120-second limit. Trim it and try again.",
    ErrorCode.RESOLUTION_TOO_LARGE: "That file exceeds the 4K resolution limit.",
    ErrorCode.CHECKSUM_MISMATCH: "The uploaded file doesn't match the checksum sent with it. Try uploading again.",
    ErrorCode.INVALID_REQUEST: "The request was malformed.",
    ErrorCode.UPLOAD_NOT_FOUND: "That upload doesn't exist. Start a new one.",
    ErrorCode.UPLOAD_EXPIRED: "That upload link expired. Request a new one and upload again.",
    ErrorCode.UPLOAD_INCOMPLETE: "The file hasn't finished uploading yet.",
    ErrorCode.DECODE_FAILED: "That file couldn't be read. It may be corrupt or incompletely uploaded.",
    ErrorCode.UNSUPPORTED_CODEC: "That video uses a codec we can't read. Re-encode it as H.264 MP4.",
    ErrorCode.DECODE_TIMEOUT: "Reading that file took too long. Try a shorter or smaller version.",
    ErrorCode.MODEL_UNAVAILABLE: "The analysis service is starting up. Try again in a moment.",
    ErrorCode.RESOURCE_EXHAUSTED: "The service is at capacity. Try again shortly.",
    ErrorCode.JOB_NOT_FOUND: "That analysis doesn't exist or has expired.",
    ErrorCode.RESULT_NOT_READY: "That analysis is still running.",
    ErrorCode.RATE_LIMITED: "Too many requests. Wait a moment before trying again.",
    ErrorCode.INTERNAL_ERROR: "Something went wrong on our side. The analysis wasn't completed.",
}


class ApiError(Base):
    """One failure, described well enough for the UI to render a sentence."""

    code: ErrorCode
    message: str = Field(description="User-facing. Plain language, no jargon, no apology.")
    detail: str | None = Field(default=None, description="Operator-facing. Never shown to users.")
    field: str | None = Field(default=None, description="Request field at fault, for 400s.")
    retryable: bool = Field(description="Whether retrying the same request could succeed.")
    job_id: UUID | None = None
    retry_after_s: int | None = Field(default=None, ge=0)

    @classmethod
    def of(
        cls,
        code: ErrorCode,
        *,
        detail: str | None = None,
        field: str | None = None,
        job_id: UUID | None = None,
        retry_after_s: int | None = None,
        message: str | None = None,
    ) -> ApiError:
        """Build an error with the canonical message and retry flag for its code."""
        return cls(
            code=code,
            message=message or DEFAULT_MESSAGES[code],
            detail=detail,
            field=field,
            retryable=is_retryable(code),
            job_id=job_id,
            retry_after_s=retry_after_s,
        )

    @property
    def http_status(self) -> int:
        return HTTP_STATUS[self.code]


class ErrorResponse(Base):
    """The single error envelope. Every non-2xx response body is this shape."""

    error: ApiError


# --------------------------------------------------------------------------
# Documentation examples
# --------------------------------------------------------------------------

EXAMPLE_CONTEXT: dict[ErrorCode, dict[str, object]] = {
    ErrorCode.INVALID_REQUEST: {"field": "sha256"},
    ErrorCode.FILE_TOO_LARGE: {"field": "size_bytes"},
    ErrorCode.CHECKSUM_MISMATCH: {"field": "body"},
    ErrorCode.FILE_EMPTY: {"field": "body"},
    ErrorCode.RESULT_NOT_READY: {"retry_after_s": 2},
    ErrorCode.MODEL_UNAVAILABLE: {"retry_after_s": 5},
    ErrorCode.RESOURCE_EXHAUSTED: {"retry_after_s": 30},
    ErrorCode.RATE_LIMITED: {"retry_after_s": 60},
    ErrorCode.DECODE_FAILED: {"detail": "moov atom not found"},
}
"""Per-code extras that make an example representative rather than skeletal.

A documented payload showing ``field: null`` on a validation error teaches an
integrator the wrong shape for the case they will actually hit.
"""


def example_for(code: ErrorCode, **over: object) -> dict[str, object]:
    """The canonical documented payload for one error code.

    Built through ``ApiError.of`` rather than hand-written, so a documented
    example is by construction a payload the server could actually emit — the
    message, HTTP status and ``retryable`` flag all come from the same place the
    real response does. An example that drifts from the implementation is worse
    than none, because an integrator codes against it.
    """
    context: dict[str, object] = dict(EXAMPLE_CONTEXT.get(code, {}))
    context.update(over)
    return ErrorResponse(error=ApiError.of(code, **context)).model_dump(mode="json")  # type: ignore[arg-type]


def all_examples() -> dict[str, dict[str, object]]:
    """Every code's example, keyed by code value. Used by the schema tests."""
    return {code.value: example_for(code) for code in ErrorCode}
