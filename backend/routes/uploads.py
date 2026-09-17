"""Upload negotiation, and the local-dev receiver.

``POST /uploads`` carries no file bytes. The client hashes locally, asks for
somewhere to put the file, and only then uploads. A file already analysed by the
current model is answered here without being uploaded at all — on a 200 MB video
that is the entire perceived latency.

``PUT /uploads/{upload_id}`` is the server half of the signed URL. An object
store receives bytes through its own presigned endpoint and this process never
sees them; a local filesystem has no such endpoint, so this route exists to make
the flow work end to end in development. It returns 400 with an explanation on
backends that upload directly, rather than pretending to accept the bytes.

Both routes are thin: every decision lives in ``AnalysisService``.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, Response, status

from backend.deps import PrincipalDep, ServiceDep, SettingsDep
from backend.errors import AppError, error_responses
from backend.logging import get_logger
from contracts.api import CreateUploadRequest, CreateUploadResponse
from contracts.enums import ErrorCode

router = APIRouter(tags=["upload"])
log = get_logger(__name__)


@router.post(
    "/uploads",
    response_model=CreateUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Negotiate an upload, or receive a cached result",
    responses={
        200: {
            "description": (
                "Either a URL to upload to (`cache_hit: false`) or a finished "
                "analysis (`cache_hit: true`). Exactly one is populated."
            )
        },
        **error_responses(
            ErrorCode.INVALID_REQUEST,
            ErrorCode.FILE_TOO_LARGE,
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            ErrorCode.INTERNAL_ERROR,
        ),
    },
)
async def create_upload(
    body: CreateUploadRequest,
    service: ServiceDep,
    settings: SettingsDep,
    principal: PrincipalDep,
) -> CreateUploadResponse:
    if settings.mock_latency_ms:
        await asyncio.sleep(settings.mock_latency_ms / 1000)
    return await service.create_upload(body)


@router.put(
    "/uploads/{upload_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Upload file bytes (local deployments only)",
    responses={
        204: {"description": "Bytes received and verified. Now POST /analyze."},
        **error_responses(
            ErrorCode.INVALID_REQUEST,
            ErrorCode.FILE_EMPTY,
            ErrorCode.CHECKSUM_MISMATCH,
            ErrorCode.UPLOAD_NOT_FOUND,
            ErrorCode.UPLOAD_EXPIRED,
            ErrorCode.FILE_TOO_LARGE,
            ErrorCode.INTERNAL_ERROR,
        ),
    },
)
async def receive_upload(
    upload_id: str,
    request: Request,
    service: ServiceDep,
    settings: SettingsDep,
    principal: PrincipalDep,
    signature: str | None = None,
) -> Response:
    body = await request.body()
    if not body:
        raise AppError(ErrorCode.FILE_EMPTY, field="body")

    # A second cap here is not redundant with the one on POST /uploads. That one
    # checks a *declared* size, which a client controls; this one checks bytes
    # that actually arrived. Storage verifies the checksum and exact length
    # afterwards, but refusing an obviously oversized body first avoids buffering
    # something enormous only to reject it.
    if len(body) > max(settings.max_image_bytes, settings.max_video_bytes):
        raise AppError(ErrorCode.FILE_TOO_LARGE, field="body")

    await service.receive_bytes(upload_id, body, signature=signature)
    log.info(
        "upload received",
        extra={
            "upload_id": upload_id,
            "size_bytes": len(body),
            "principal": principal.rate_limit_key,
        },
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
