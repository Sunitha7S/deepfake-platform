"""``GET /events/{job_id}`` — job progress as Server-Sent Events.

Every ``202`` from ``POST /analyze`` has always advertised this URL in
``events_url``; until this route existed, following it was a 404. It carries
exactly the ``JobStatus`` that ``GET /jobs/{job_id}`` returns, wrapped in the
contract's ``ProgressEvent``, so a client that renders polling results renders
these frames unchanged.

Wire format, per ``ProgressEvent``'s own docstring::

    event: <state|progress|heartbeat|done|error>
    data: <ProgressEvent as JSON>

The status is re-derived from the queue and the result store on a poll
interval — the same derivation, with no new authority to keep in sync. Progress
is therefore as coarse as ``/jobs`` reports it: state transitions, not
per-frame percentages, until workers publish finer progress somewhere the API
can read.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Path, Request
from fastapi.responses import StreamingResponse

from backend.deps import PrincipalDep, ServiceDep, SettingsDep
from backend.errors import error_responses
from contracts.enums import ErrorCode
from contracts.job import ProgressEvent

router = APIRouter(tags=["jobs"])


class EventSourceResponse(StreamingResponse):
    """A streaming response FastAPI documents as ``text/event-stream``."""

    media_type = "text/event-stream"


@router.get(
    "/events/{job_id}",
    response_class=EventSourceResponse,
    summary="Stream job progress (Server-Sent Events)",
    responses={
        200: {
            "model": ProgressEvent,
            "description": (
                "A `text/event-stream`. Each frame is `event: <name>` followed by "
                "`data:` holding one `ProgressEvent`. The first frame is `state`; "
                "the last is `done` (succeeded) or `error` (failed). `progress` "
                "marks a change within a state; `heartbeat` repeats the current "
                "status during silence. The server closes the stream after the "
                "final frame, or after a maximum duration — reconnect, or poll "
                "`GET /jobs/{job_id}`, if the stream ends without one."
            ),
        },
        **error_responses(
            ErrorCode.INVALID_REQUEST,
            ErrorCode.JOB_NOT_FOUND,
            ErrorCode.INTERNAL_ERROR,
        ),
    },
)
async def stream_job_events(
    request: Request,
    service: ServiceDep,
    settings: SettingsDep,
    principal: PrincipalDep,
    job_id: UUID = Path(description="Job identifier returned by POST /analyze."),
) -> EventSourceResponse:
    if settings.mock_latency_ms:
        await asyncio.sleep(settings.mock_latency_ms / 1000)
    # Resolved before the response starts: an unknown job is a 404 with the
    # standard error envelope, not a 200 stream that closes with nothing in it.
    first = await service.job_status(job_id)

    async def frames() -> AsyncIterator[str]:
        async for event in service.job_events(job_id, first, request.is_disconnected):
            yield f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"

    return EventSourceResponse(
        frames(),
        headers={
            # Intermediaries must neither cache nor buffer a live stream; nginx
            # buffers by default and would deliver every frame at the end.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
