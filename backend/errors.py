"""Exception handling.

Requirement: *every* response validates against the S0 contracts. Meeting that
takes more than raising tidy errors from route handlers, because three failure
paths bypass route code entirely and FastAPI's defaults for all three emit a
body shape that is **not** ``ErrorResponse``:

* ``RequestValidationError`` — a malformed body. FastAPI's default is
  ``{"detail": [...]}``. That is the single most common error a frontend
  developer will hit, so leaving it unmapped would break integration on day one.
* ``StarletteHTTPException`` — 404 on an unknown path, 405 on a wrong method.
* Anything uncaught — a 500 with a stack trace, which must never reach a client.

Each is mapped here, so there is exactly one error shape on the wire.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.logging import get_logger
from contracts.api import (
    ALLOWED_IMAGE_MIME,
    ALLOWED_VIDEO_MIME,
    MAX_IMAGE_BYTES,
    MAX_VIDEO_BYTES,
)
from contracts.enums import ErrorCode
from contracts.errors import (
    DEFAULT_MESSAGES,
    HTTP_STATUS,
    ApiError,
    ErrorResponse,
    example_for,
)

log = get_logger(__name__)

_STATUS_TO_CODE: dict[int, ErrorCode] = {
    404: ErrorCode.JOB_NOT_FOUND,
    405: ErrorCode.INVALID_REQUEST,
    409: ErrorCode.RESULT_NOT_READY,
    413: ErrorCode.FILE_TOO_LARGE,
    415: ErrorCode.UNSUPPORTED_MEDIA_TYPE,
    429: ErrorCode.RATE_LIMITED,
    503: ErrorCode.MODEL_UNAVAILABLE,
}


class AppError(Exception):
    """The one exception route handlers raise.

    Carries a contract ``ApiError``, so the HTTP status, the user-facing message
    and the retryable flag all derive from the code rather than being chosen at
    the call site.
    """

    def __init__(
        self,
        code: ErrorCode,
        *,
        message: str | None = None,
        detail: str | None = None,
        field: str | None = None,
        job_id: UUID | None = None,
        retry_after_s: int | None = None,
    ) -> None:
        self.api_error = ApiError.of(
            code,
            message=message,
            detail=detail,
            field=field,
            job_id=job_id,
            retry_after_s=retry_after_s,
        )
        super().__init__(self.api_error.message)

    @property
    def status_code(self) -> int:
        return self.api_error.http_status


def error_response(err: ApiError) -> JSONResponse:
    """Render an ``ApiError`` as the single wire shape."""
    headers: dict[str, str] = {}
    if err.retry_after_s is not None:
        headers["Retry-After"] = str(err.retry_after_s)
    return JSONResponse(
        status_code=err.http_status,
        content=ErrorResponse(error=err).model_dump(mode="json"),
        headers=headers or None,
    )


def _first_field(errors: list[dict[str, Any]]) -> str | None:
    """Pick a human-meaningful field name out of pydantic's ``loc`` tuple."""
    if not errors:
        return None
    loc = [str(p) for p in errors[0].get("loc", ()) if p not in ("body", "query", "path")]
    return ".".join(loc) or None


def _refine_upload_error(errors: list[dict[str, Any]]) -> ApiError | None:
    """Recover the specific error code from a contract model-validator failure.

    ``CreateUploadRequest`` enforces the size caps and the MIME allowlist inside
    a pydantic validator, which surfaces as a generic ``RequestValidationError``.
    Left alone, every oversized upload would answer 400 ``invalid_request`` and
    the 413 ``file_too_large`` and 415 ``unsupported_media_type`` codes — each
    with a message written to tell the user exactly what to do — would be
    unreachable.

    Rather than pattern-match the validator's prose, which would break silently
    the next time someone rewords it, this re-derives the cause from the raw
    input against the same constants the contract used. The contracts package is
    not modified; the constants are simply read from it.

    Returns None when the failure is not one of these two, leaving the generic
    handler to answer.
    """
    if not errors:
        return None
    err = errors[0]
    if err.get("type") != "value_error":
        return None

    payload = err.get("input")
    if not isinstance(payload, dict):
        return None
    kind = payload.get("media_kind")
    if kind not in ("image", "video"):
        return None

    is_image = kind == "image"
    allowed = ALLOWED_IMAGE_MIME if is_image else ALLOWED_VIDEO_MIME
    cap = MAX_IMAGE_BYTES if is_image else MAX_VIDEO_BYTES

    content_type = payload.get("content_type")
    if isinstance(content_type, str) and content_type not in allowed:
        return ApiError.of(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            field="content_type",
            detail=f"{content_type!r} is not permitted for media_kind={kind}",
        )

    size = payload.get("size_bytes")
    if isinstance(size, int) and size > cap:
        return ApiError.of(
            ErrorCode.FILE_TOO_LARGE,
            field="size_bytes",
            detail=f"{size} bytes exceeds the {cap} byte cap for {kind}",
        )
    return None


def _pascal(code: ErrorCode) -> str:
    return "".join(part.capitalize() for part in code.value.split("_"))


def component_name_for(codes: tuple[ErrorCode, ...]) -> str:
    """Stable name for the shared component a response is hoisted into.

    Derived from the codes rather than the route, so two operations that can
    fail the same way share one component instead of duplicating the schema —
    which is the whole point of hoisting them.
    """
    return "".join(_pascal(code) for code in codes)


def error_responses(*codes: ErrorCode) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` for the errors an operation can actually produce.

    Routes declare the codes they raise; everything else — status, description,
    schema reference and example — is derived. Hand-written response blocks are
    how documentation drifts from behaviour, and this is the only way to declare
    an error in this codebase.

    Codes sharing a status are grouped into one response with a named example
    each, because HTTP gives an operation one body shape per status and pretending
    otherwise produces a document no generator can use.
    """
    grouped: dict[int, list[ErrorCode]] = {}
    for code in codes:
        grouped.setdefault(HTTP_STATUS[code], []).append(code)

    out: dict[int | str, dict[str, Any]] = {}
    for status, group in sorted(grouped.items()):
        ordered = tuple(group)
        out[status] = {
            "model": ErrorResponse,
            "description": " ".join(DEFAULT_MESSAGES[c] for c in ordered),
            "content": {
                "application/json": {
                    "examples": {
                        c.value: {
                            "summary": c.value,
                            "description": DEFAULT_MESSAGES[c],
                            "value": example_for(c),
                        }
                        for c in ordered
                    }
                }
            },
            # Read by the schema post-processor to hoist this into
            # components.responses. Stripped before the document is emitted.
            "x-error-codes": [c.value for c in ordered],
            "x-component-name": component_name_for(ordered),
        }
    return out


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        log.warning(
            "request failed",
            extra={"code": exc.api_error.code.value, "status": exc.status_code},
        )
        return error_response(exc.api_error)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()

        refined = _refine_upload_error(errors)
        if refined is not None:
            log.info("request rejected", extra={"code": refined.code.value})
            return error_response(refined)

        field = _first_field(errors)
        detail = errors[0].get("msg") if errors else None
        log.info("request rejected", extra={"code": "invalid_request", "field": field})
        return error_response(
            ApiError.of(
                ErrorCode.INVALID_REQUEST,
                message=(
                    f"{DEFAULT_MESSAGES[ErrorCode.INVALID_REQUEST]} Check '{field}'."
                    if field
                    else DEFAULT_MESSAGES[ErrorCode.INVALID_REQUEST]
                ),
                detail=detail,
                field=field,
            )
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_TO_CODE.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        # Preserve the intended status even where the code maps elsewhere: a 405
        # must stay a 405 rather than becoming INVALID_REQUEST's 400.
        err = ApiError.of(code, detail=str(exc.detail) if exc.detail else None)
        payload = ErrorResponse(error=err).model_dump(mode="json")
        status = exc.status_code if exc.status_code in HTTP_STATUS.values() else err.http_status
        return JSONResponse(status_code=status, content=payload)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # The traceback goes to the log; the client gets a sentence. A stack
        # trace on the wire is an information leak and useless to the caller.
        log.exception("unhandled exception", extra={"code": "internal_error"})
        return error_response(
            ApiError.of(ErrorCode.INTERNAL_ERROR, detail=type(exc).__name__)
        )
