"""``GET /jobs/{job_id}`` — job status, derived rather than stored.

The polling fallback for clients without SSE, and the same ``JobStatus`` shape
the event stream (``GET /events/{job_id}``) carries. Building both on one model lets a client
implement polling first and switch to SSE without touching its rendering.

Status is computed per request from the queue envelope and the presence of a
result. There is no status record to read, and deliberately so: a stored copy
would be a fourth authority that nothing keeps in sync, and its failure is
silent — the queue dead-letters a job, the record still reads ``running``, and a
user watches a spinner forever.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, Path

from backend.deps import PrincipalDep, ServiceDep, SettingsDep
from backend.errors import error_responses
from contracts.enums import ErrorCode
from backend.logging import get_logger
from contracts.job import JobStatus

router = APIRouter(tags=["jobs"])
log = get_logger(__name__)


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    summary="Poll job status",
    responses={
        200: {
            "description": (
                "Current status. `result_url` is present only when a result "
                "actually exists — a job can succeed and have its result expire."
            )
        },
        **error_responses(
            ErrorCode.INVALID_REQUEST,
            ErrorCode.JOB_NOT_FOUND,
            ErrorCode.INTERNAL_ERROR,
        ),
    },
)
async def get_job(
    service: ServiceDep,
    settings: SettingsDep,
    principal: PrincipalDep,
    job_id: UUID = Path(description="Job identifier returned by POST /analyze."),
) -> JobStatus:
    if settings.mock_latency_ms:
        await asyncio.sleep(settings.mock_latency_ms / 1000)
    return await service.job_status(job_id)
