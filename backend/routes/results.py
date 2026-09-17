"""``GET /results/{job_id}`` — the finished EvidenceBundle.

Four outcomes, and keeping them distinct is what lets a client behave sensibly:

* **200** — the bundle. This includes ``verdict: insufficient_evidence``, which
  is a *result*, not a failure. A client that treats 200 as "answer" and
  non-200 as "no answer" is then always right.
* **409 result_not_ready** — the job exists and has not finished. Keep polling.
* **the job's own error** — the job failed, so no result will ever exist.
  Surfacing the original failure rather than a generic 404 is the difference
  between a user learning their file was corrupt and a user retrying a file that
  can never work.
* **404** — unknown, or the result has expired.

A stored result is served immediately whatever the queue says. The runner
persists before it acks, and lease expiry counts as an attempt, so a bundle can
exist while the envelope reads ``LEASED`` or even ``DEAD``. The envelope answers
"has this delivery been acknowledged", not "does an answer exist" — and only the
second question matters here.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, Path

from backend.deps import PrincipalDep, ServiceDep, SettingsDep
from backend.errors import error_responses
from contracts.enums import ErrorCode
from backend.logging import get_logger
from contracts.evidence import EvidenceBundle

router = APIRouter(tags=["results"])
log = get_logger(__name__)


@router.get(
    "/results/{job_id}",
    response_model=EvidenceBundle,
    summary="Fetch a completed analysis",
    responses={
        200: {
            "description": (
                "The analysis. `verdict` may be `insufficient_evidence`, which is "
                "a successful outcome with a null `p_fake` — render it as an "
                "answer, not an error."
            )
        },
        **error_responses(
            ErrorCode.INVALID_REQUEST,
            ErrorCode.JOB_NOT_FOUND,
            ErrorCode.RESULT_NOT_READY,
            ErrorCode.DECODE_FAILED,
            ErrorCode.INTERNAL_ERROR,
        ),
    },
)
async def get_result(
    service: ServiceDep,
    settings: SettingsDep,
    principal: PrincipalDep,
    job_id: UUID = Path(description="Job identifier returned by POST /analyze."),
) -> EvidenceBundle:
    if settings.mock_latency_ms:
        await asyncio.sleep(settings.mock_latency_ms / 1000)
    return await service.result(job_id)
