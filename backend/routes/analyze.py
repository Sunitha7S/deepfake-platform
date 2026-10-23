"""``POST /analyze`` — turn a completed upload into a queued job.

Returns 202 with a job id and the three URLs a client needs to follow it. No
analysis has happened at this point and none happens on this request thread:
video work takes minutes, and holding an HTTP connection open for that exhausts
the connection pool under trivial load, times out at every proxy in the path,
and gives the user a spinner with no progress.

**Idempotency is visible in the response.** Submitting content already queued or
analysed under the same model returns the *original* job's id rather than
creating a second one. A client must therefore use the id it receives rather
than assuming its own — which is why the id is in the body and not merely
implied by a 202.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, status

from backend.deps import PrincipalDep, RateLimited, ServiceDep, SettingsDep
from backend.errors import error_responses
from contracts.enums import ErrorCode
from backend.logging import get_logger
from contracts.api import AnalyzeAcceptedResponse, AnalyzeRequest

router = APIRouter(tags=["analysis"])
log = get_logger(__name__)


@router.post(
    "/analyze",
    response_model=AnalyzeAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue an uploaded file for analysis",
    dependencies=[RateLimited],
    responses={
        202: {
            "description": (
                "Queued. Poll `status_url`, or fetch `result_url` once the job "
                "succeeds. If the same content was already submitted under this "
                "model version, `job_id` is that earlier job's id."
            )
        },
        **error_responses(
            ErrorCode.INVALID_REQUEST,
            ErrorCode.UPLOAD_NOT_FOUND,
            ErrorCode.UPLOAD_INCOMPLETE,
            ErrorCode.RATE_LIMITED,
            ErrorCode.INTERNAL_ERROR,
        ),
    },
)
async def analyze(
    body: AnalyzeRequest,
    service: ServiceDep,
    settings: SettingsDep,
    principal: PrincipalDep,
) -> AnalyzeAcceptedResponse:
    if settings.mock_latency_ms:
        await asyncio.sleep(settings.mock_latency_ms / 1000)

    accepted = await service.submit(body.upload_id, body.options)
    log.info(
        "analysis queued",
        extra={
            "job_id": str(accepted.job_id),
            "upload_id": body.upload_id,
            "queue_position": accepted.queue_position,
            "principal": principal.rate_limit_key,
        },
    )
    return accepted
