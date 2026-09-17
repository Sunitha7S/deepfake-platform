"""``GET /health`` — readiness, not liveness.

Answers "should this instance receive traffic", which is a stronger claim than
"is the process running". It probes storage and the queue for real rather than
returning a constant, so a deployment with an unreachable dependency fails
readiness and takes no requests instead of failing every one.

Two constraints shape the response. ``HealthResponse`` is a frozen contract with
no storage or queue fields, so their condition drives the ``status`` value and
the log line while depths go in ``queue_depth`` — the detail an operator needs is
in the logs, not invented into a schema clients already depend on. And there is
no state store to consult: readiness is derived from the same two dependencies
everything else uses.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response, status

from backend.deps import ServiceDep, SettingsDep
from backend.logging import get_logger
from contracts.api import BranchHealth, HealthResponse
from contracts.branch import BRANCH_NAMES
from contracts.enums import BranchId

router = APIRouter(tags=["system"])
log = get_logger(__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Readiness probe",
    responses={
        200: {"description": "Ready to serve traffic. `status` is `ok`."},
        503: {
            # Deliberately NOT an ErrorResponse. A failing readiness probe still
            # answers the question it was asked — it reports condition, and the
            # detail an operator needs is in `status` and `queue_depth`. Wrapping
            # it in an error envelope would make a load balancer's health check
            # look like a request failure.
            "model": HealthResponse,
            "description": (
                "A dependency is unavailable; do not route traffic here. The body "
                "is still a HealthResponse, with `status` set to `degraded` or "
                "`starting` — never an error envelope."
            ),
        },
    },
)
async def health(
    request: Request,
    response: Response,
    settings: SettingsDep,
    service: ServiceDep,
) -> HealthResponse:
    ready, depths, problems = await service.health()

    # No weights are loaded yet, so a branch is "loaded" exactly when the
    # dependencies behind it are reachable. When real models arrive this becomes
    # a registry query; the response shape does not change, only what fills it.
    suffix = "stub" if settings.stub_scoring else settings.model_version
    branches = [
        BranchHealth(
            id=bid.value,
            loaded=ready,
            version=f"{BRANCH_NAMES[bid].lower()}-{suffix}",
        )
        for bid in BranchId
    ]

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        log.warning("health check not ready", extra={"problems": problems})

    started_at = getattr(request.app.state, "started_at", None)
    uptime = int(time.monotonic() - started_at) if started_at else 0

    return HealthResponse(
        status="ok" if ready else "degraded",
        model_version=settings.model_version,
        branches=branches,
        device="cpu",
        queue_depth=depths,
        uptime_s=uptime,
    )
