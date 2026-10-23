"""``GET /events/{job_id}`` — the progress stream every ``202`` advertises.

Before this route existed, the ``events_url`` in every ``POST /analyze``
response led to a 404. These tests hold the route to the contract's
``ProgressEvent`` and to its own documented behaviour:

* HTTP: the advertised URL is served, as ``text/event-stream``, with the
  standard error envelope for an unknown or malformed id; a job that finishes
  while the stream is open is followed from ``state`` to ``done``.
* Stream logic (``AnalysisService.job_events``, driven with scripted statuses):
  frame ordering, ``progress`` vs ``state``, heartbeats, the maximum duration,
  client disconnect, and a job that vanishes mid-stream.

Timing knobs are shrunk through ``Settings`` so nothing waits on the
production 1 s poll / 15 s heartbeat.

    python tests/test_events.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import create_app  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.errors import AppError  # noqa: E402
from contracts import fixtures as fx  # noqa: E402
from contracts.enums import ErrorCode, JobStage, JobState  # noqa: E402
from contracts.job import JobProgress, ProgressEvent  # noqa: E402
from jobqueue.sqlite import SQLiteQueue  # noqa: E402
from tests.test_backend import Exploding, drain, submit  # noqa: E402
from workers.types import PermanentError  # noqa: E402

FAST = {"events_poll_interval_s": 0.02, "events_heartbeat_s": 5.0, "events_max_duration_s": 20.0}


@contextmanager
def platform(**over):
    """App on real storage and a SQLite queue (safe to drive from two threads)."""
    work = Path(tempfile.mkdtemp(prefix="dfp-events-"))
    settings = Settings(
        log_json=False,
        log_level="CRITICAL",
        storage_root=work / "storage",
        **{**FAST, **over},
    )
    queue = SQLiteQueue(work / "queue.sqlite3")
    app = create_app(settings, queue=queue)
    with TestClient(app) as client:
        yield client, app, queue


def parse_sse(text: str) -> list[tuple[str, ProgressEvent]]:
    """Split a text/event-stream body into (event name, validated payload)."""
    frames = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        assert set(fields) == {"event", "data"}, f"unexpected frame fields: {block!r}"
        payload = ProgressEvent.model_validate(json.loads(fields["data"]))
        assert payload.event == fields["event"], "event line and payload disagree"
        frames.append((fields["event"], payload))
    return frames


def events(client, job_id: str, prefix: str = ""):
    r = client.get(f"{prefix}/events/{job_id}")
    return r, (parse_sse(r.text) if r.status_code == 200 else None)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def test_the_advertised_events_url_is_served():
    """The defect: this URL was in every 202 and answered 404."""
    with platform() as (c, app, q):
        negotiated = c.post("/uploads", json=_body()).json()
        c.put(negotiated["upload"]["url"], content=_image())
        accepted = c.post("/analyze", json={"upload_id": negotiated["upload_id"]}).json()
        drain(app, q)
        r = c.get(accepted["events_url"])
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = parse_sse(r.text)
    assert [name for name, _ in frames] == ["state", "done"]
    assert frames[-1][1].status.state is JobState.SUCCEEDED
    assert str(frames[-1][1].job_id) == accepted["job_id"]


def test_a_failed_job_ends_with_an_error_frame_carrying_the_error():
    with platform() as (c, app, q):
        job_id = submit(c)
        drain(app, q, handler=Exploding(PermanentError("moov atom not found")))
        r, frames = events(c, job_id)
    assert [name for name, _ in frames] == ["state", "error"]
    final = frames[-1][1].status
    assert final.state is JobState.FAILED
    assert final.error is not None and "moov atom" in (final.error.detail or "")


def test_a_job_finishing_while_the_stream_is_open_is_followed_to_done():
    """Queued when the stream opens; a worker on another thread completes it."""
    with platform() as (c, app, q):
        job_id = submit(c)

        def work_later():
            time.sleep(0.4)
            drain(app, q)

        worker = threading.Thread(target=work_later)
        worker.start()
        started = time.monotonic()
        r, frames = events(c, job_id)
        worker.join()
    names = [name for name, _ in frames]
    assert names[0] == "state" and frames[0][1].status.state is JobState.QUEUED, names
    assert names[-1] == "done" and frames[-1][1].status.state is JobState.SUCCEEDED, names
    assert "done" not in names[:-1], "frames after the final frame"
    assert time.monotonic() - started < 15, "the stream did not close at done"
    stamps = [e.at for _, e in frames]
    assert stamps == sorted(stamps)


def test_stream_headers_forbid_caching_and_proxy_buffering():
    with platform() as (c, app, q):
        job_id = submit(c)
        drain(app, q)
        r, _ = events(c, job_id)
    assert r.headers["cache-control"] == "no-cache"
    assert r.headers["x-accel-buffering"] == "no"


def test_an_unknown_job_is_a_404_envelope_not_an_empty_stream():
    with platform() as (c, app, q):
        r = c.get("/events/99999999-9999-9999-9999-999999999999")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == ErrorCode.JOB_NOT_FOUND.value
    assert not r.headers["content-type"].startswith("text/event-stream")


def test_a_malformed_job_id_is_a_400():
    with platform() as (c, app, q):
        r = c.get("/events/not-a-uuid")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == ErrorCode.INVALID_REQUEST.value


def test_the_stream_follows_the_api_prefix():
    with platform(api_prefix="/api/v1") as (c, app, q):
        negotiated = c.post("/api/v1/uploads", json=_body()).json()
        c.put(negotiated["upload"]["url"], content=_image())
        accepted = c.post("/api/v1/analyze", json={"upload_id": negotiated["upload_id"]}).json()
        assert accepted["events_url"].startswith("/api/v1/events/")
        drain(app, q)
        r = c.get(accepted["events_url"])
        assert c.get(f"/events/{accepted['job_id']}").status_code == 404, "unprefixed path served"
    assert r.status_code == 200


def test_an_idle_job_gets_heartbeats_then_the_stream_closes_at_the_maximum_duration():
    """No worker: the job stays queued. The client must still see life on the
    connection, and the server must not hold it forever."""
    with platform(events_heartbeat_s=0.1, events_max_duration_s=0.6) as (c, app, q):
        job_id = submit(c)
        started = time.monotonic()
        r, frames = events(c, job_id)
        elapsed = time.monotonic() - started
    names = [name for name, _ in frames]
    assert names[0] == "state"
    assert names.count("heartbeat") >= 3, names
    assert set(names[1:]) == {"heartbeat"}, names
    assert all(e.status.state is JobState.QUEUED for _, e in frames)
    assert elapsed < 5, f"stream outlived its maximum duration ({elapsed:.1f}s)"


# --------------------------------------------------------------------------
# Stream logic, with scripted statuses
# --------------------------------------------------------------------------


def _scripted(service, statuses):
    """Make ``service.job_status`` return ``statuses`` in order (then repeat the
    last). An exception instance in the list is raised instead."""
    remaining = list(statuses)

    async def job_status(job_id):
        item = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if isinstance(item, BaseException):
            raise item
        return item

    service.job_status = job_status


async def _collect(service, first, *, disconnect_after: int | None = None):
    calls = {"n": 0}

    async def is_disconnected():
        calls["n"] += 1
        return disconnect_after is not None and calls["n"] > disconnect_after

    return [
        e async for e in service.job_events(first.job_id, first, is_disconnected)
    ]


def _running(stage: JobStage, percent: int):
    progress = JobProgress(stage=stage, percent=percent)
    return fx.job_running().model_copy(update={"progress": progress})


def _queued():
    return fx.job_running().model_copy(
        update={"state": JobState.QUEUED, "started_at": None,
                "progress": JobProgress(stage=JobStage.VALIDATING, percent=0)}
    )


def _service(**over):
    settings = Settings(log_json=False, log_level="CRITICAL",
                        storage_root=Path(tempfile.mkdtemp(prefix="dfp-events-")),
                        **{**FAST, **over})
    return create_app(settings).state.service


def test_frames_mark_state_changes_progress_changes_and_the_end():
    service = _service()
    _scripted(service, [
        _running(JobStage.DECODING, 10),
        _running(JobStage.DECODING, 10),   # unchanged: no frame
        _running(JobStage.SCORING, 60),    # same state, new progress
        fx.job_succeeded(),
    ])
    frames = asyncio.run(_collect(service, _queued()))
    assert [(f.event, f.status.state) for f in frames] == [
        ("state", JobState.QUEUED),
        ("state", JobState.RUNNING),
        ("progress", JobState.RUNNING),
        ("done", JobState.SUCCEEDED),
    ]
    assert frames[2].status.progress.stage is JobStage.SCORING


def test_an_already_finished_job_gets_state_then_its_final_frame_and_nothing_else():
    service = _service()
    _scripted(service, [AssertionError("must not poll a finished job")])
    for first, last in ((fx.job_succeeded(), "done"), (fx.job_failed(), "error")):
        frames = asyncio.run(_collect(service, first))
        assert [f.event for f in frames] == ["state", last]


def test_a_cancelled_job_ends_with_an_error_frame():
    service = _service()
    cancelled = fx.job_failed().model_copy(update={"state": JobState.CANCELLED, "error": None})
    _scripted(service, [cancelled])
    frames = asyncio.run(_collect(service, _queued()))
    assert [f.event for f in frames] == ["state", "error"]


def test_the_stream_stops_polling_when_the_client_disconnects():
    service = _service()
    _scripted(service, [_queued()])
    frames = asyncio.run(_collect(service, _queued(), disconnect_after=2))
    assert [f.event for f in frames] == ["state"], "frames sent to a departed client"


def test_a_job_that_vanishes_mid_stream_ends_the_stream_quietly():
    service = _service()
    _scripted(service, [_queued(), AppError(ErrorCode.JOB_NOT_FOUND)])
    frames = asyncio.run(_collect(service, _queued()))
    assert [f.event for f in frames] == ["state"]


def test_heartbeats_repeat_the_current_status_only_after_silence():
    service = _service(events_heartbeat_s=0.1, events_max_duration_s=0.45)
    _scripted(service, [_queued()])
    frames = asyncio.run(_collect(service, _queued()))
    names = [f.event for f in frames]
    assert names[0] == "state" and 2 <= names.count("heartbeat") <= 5, names
    gaps = [(b.at - a.at).total_seconds() for a, b in zip(frames, frames[1:], strict=False)]
    assert min(gaps) >= 0.08, f"heartbeat sent before the silence threshold: {gaps}"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _image() -> bytes:
    from tests.test_backend import IMAGE

    return IMAGE


def _body() -> dict:
    import hashlib

    data = _image()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
            "content_type": "image/jpeg", "media_kind": "image"}


# --------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------


def _run() -> int:
    tests = sorted(
        (n, o) for n, o in globals().items() if n.startswith("test_") and callable(o)
    )
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:400]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
