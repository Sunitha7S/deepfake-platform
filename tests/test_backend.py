"""Backend API tests against the real platform.

No fixtures, no job store. Every test builds an app over a temporary
``LocalStorageBackend`` and an ``InMemoryQueue``, and drives a real
``WorkerRunner`` inline when a job needs to finish. The only thing still standing
in for the finished system is scoring, which is deterministic stub analysis.

Emphasis is on the properties a client depends on and that are cheap to break:
the error envelope is uniform on every path, ``insufficient_evidence`` arrives as
a 200, a persisted result outranks whatever the queue says, and ``/health`` does
not move when the API prefix changes.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import sys
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import create_app  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.deps import ANONYMOUS, Principal  # noqa: E402
from backend.service import AnalysisService  # noqa: E402
from contracts.enums import ErrorCode, MediaKind, Verdict  # noqa: E402
from contracts.errors import ErrorResponse  # noqa: E402
from contracts.evidence import EvidenceBundle  # noqa: E402
from contracts.job import JobStatus  # noqa: E402
from jobqueue.memory import InMemoryQueue  # noqa: E402
from jobqueue.registry import IMAGE_QUEUE  # noqa: E402
from jobqueue.types import JobState as QueueState  # noqa: E402
from storage.types import StorageHealth  # noqa: E402
from tests.stub_handlers import StubImageHandler  # noqa: E402
from workers.registry import build_worker  # noqa: E402
from workers.types import PermanentError, TransientError  # noqa: E402

def _real_jpeg(size: int = 320) -> bytes:
    """A decodable image.

    Wiring real inference made this necessary: the worker now runs the actual
    decode path, so a placeholder byte string raises UnreadableMedia and
    dead-letters instead of producing a bundle. The face gate may still abstain
    — which is a valid 200 result — so these tests only need the bytes to
    decode, not to contain a face.
    """
    import cv2
    import numpy as np

    rng = np.random.default_rng(11)
    return cv2.imencode(".jpg", rng.integers(0, 255, (size, size, 3), dtype=np.uint8))[1].tobytes()


IMAGE = _real_jpeg()
IMAGE_SHA = hashlib.sha256(IMAGE).hexdigest()
UNKNOWN_JOB = "99999999-9999-9999-9999-999999999999"


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


@contextmanager
def platform(**over):
    """An app on real storage and a real queue. Yields (client, app, queue)."""
    # model_version is deliberately NOT pinned here. Pinning it to the same
    # literal the handlers happened to emit is what hid the read/write key
    # mismatch in the result cache: the harness agreed with itself while the
    # shipped default disagreed with the engines, so every cache test passed
    # against a configuration nobody deploys. The harness now runs the real
    # default, and workers.engine_registry.verify_version_alignment keeps it
    # honest.
    settings = Settings(
        log_json=False,
        log_level="CRITICAL",
        storage_root=Path(tempfile.mkdtemp(prefix="dfp-api-")),
        **over,
    )
    queue = InMemoryQueue()
    app = create_app(settings, queue=queue)
    with TestClient(app) as client:
        yield client, app, queue


def _upload_body(data: bytes = IMAGE, **over) -> dict:
    body = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "content_type": "image/jpeg",
        "media_kind": "image",
    }
    body.update(over)
    return body


def negotiate(client, data: bytes = IMAGE, **over) -> dict:
    r = client.post("/uploads", json=_upload_body(data, **over))
    assert r.status_code == 200, r.text
    return r.json()


def put_bytes(client, negotiated: dict, data: bytes = IMAGE):
    return client.put(negotiated["upload"]["url"], content=data)


def submit(client, data: bytes = IMAGE, **over) -> str:
    """Upload and queue a file. Returns the job id."""
    negotiated = negotiate(client, data, **over)
    assert put_bytes(client, negotiated, data).status_code == 204
    r = client.post("/analyze", json={"upload_id": negotiated["upload_id"]})
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def drain(app, queue, *, handler=None, kind=MediaKind.IMAGE, jobs: int = 1):
    """Run a real worker until ``jobs`` are processed."""
    worker = build_worker(kind, queue=queue, storage=app.state.storage, handler=handler)
    worker.poll_interval_s = 0.02
    return asyncio.run(worker.run(max_jobs=jobs, idle_timeout_s=0.3))


class Exploding:
    media_kind = MediaKind.IMAGE.value
    name = "exploding"

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def handle(self, request, progress):
        raise self.exc


# --------------------------------------------------------------------------
# POST /uploads
# --------------------------------------------------------------------------


def test_upload_negotiation_returns_a_url_and_no_result():
    with platform() as (c, _, _):
        body = negotiate(c)
    assert body["cache_hit"] is False
    assert body["cached_result"] is None
    assert body["upload"]["method"] == "PUT"
    assert body["upload"]["url"].startswith("/uploads/")


def test_upload_ids_are_unique_and_ignore_the_filename():
    with platform() as (c, _, _):
        a = negotiate(c, filename_hint="../../etc/passwd")
        b = negotiate(c, filename_hint="../../etc/passwd")
    assert a["upload_id"] != b["upload_id"]
    assert "passwd" not in json.dumps(a)


def test_oversized_upload_is_413():
    with platform() as (c, _, _):
        r = c.post("/uploads", json=_upload_body(size_bytes=99_999_999))
    assert r.status_code == 413
    assert r.json()["error"]["code"] == ErrorCode.FILE_TOO_LARGE.value


def test_disallowed_content_type_is_415():
    with platform() as (c, _, _):
        r = c.post("/uploads", json=_upload_body(content_type="image/gif"))
    assert r.status_code == 415
    assert r.json()["error"]["code"] == ErrorCode.UNSUPPORTED_MEDIA_TYPE.value


def test_malformed_upload_request_is_400_with_the_field():
    with platform() as (c, _, _):
        r = c.post("/uploads", json=_upload_body(sha256="not-a-hash"))
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == ErrorCode.INVALID_REQUEST.value
    assert err["field"] == "sha256"
    assert err["retryable"] is False


def test_cache_hit_short_circuits_the_upload_entirely():
    """The designed feature: a file already analysed is answered without transfer."""
    with platform() as (c, app, q):
        job_id = submit(c)
        drain(app, q)
        assert c.get(f"/results/{job_id}").status_code == 200

        again = negotiate(c)
    assert again["cache_hit"] is True
    assert again["upload"] is None
    bundle = EvidenceBundle.model_validate(again["cached_result"])
    assert str(bundle.job_id) == job_id


def test_cache_is_scoped_to_the_model_version():
    """A new model must not serve a verdict computed by the old one."""
    with platform() as (c, app, q):
        submit(c)
        drain(app, q)
        app.state.settings.model_version = "stub-1"
        again = negotiate(c)
    assert again["cache_hit"] is False, "a different model version reused a cached result"


# --------------------------------------------------------------------------
# PUT /uploads/{upload_id}
# --------------------------------------------------------------------------


def test_receiving_bytes_completes_the_upload():
    with platform() as (c, app, _):
        negotiated = negotiate(c)
        r = put_bytes(c, negotiated)
        session = asyncio.run(app.state.storage.get_session(negotiated["upload_id"]))
    assert r.status_code == 204
    assert session.state.has_bytes
    assert session.verified_sha256 == IMAGE_SHA


def test_upload_with_a_forged_signature_is_rejected():
    with platform() as (c, app, _):
        negotiated = negotiate(c)
        url = negotiated["upload"]["url"].split("?")[0] + "?signature=forged.token"
        r = c.put(url, content=IMAGE)
        session = asyncio.run(app.state.storage.get_session(negotiated["upload_id"]))
    assert r.status_code == 400
    assert session.state.value == "created", "a rejected signature advanced the session"


def test_upload_to_an_unknown_id_is_404():
    with platform() as (c, _, _):
        r = c.put("/uploads/u_does_not_exist?signature=x", content=IMAGE)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == ErrorCode.UPLOAD_NOT_FOUND.value


def test_empty_upload_body_is_rejected():
    with platform() as (c, _, _):
        negotiated = negotiate(c)
        r = c.put(negotiated["upload"]["url"], content=b"")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == ErrorCode.FILE_EMPTY.value


def test_upload_body_that_disagrees_with_the_declaration_is_rejected():
    """Storage verifies the checksum; a swapped body must not be accepted."""
    with platform() as (c, _, _):
        negotiated = negotiate(c)
        r = c.put(negotiated["upload"]["url"], content=b"different bytes entirely")
    assert r.status_code in (400, 413)
    assert r.json()["error"]["retryable"] is False


# --------------------------------------------------------------------------
# POST /analyze
# --------------------------------------------------------------------------


def test_analyze_queues_a_job_and_returns_its_urls():
    with platform() as (c, _, q):
        negotiated = negotiate(c)
        put_bytes(c, negotiated)
        r = c.post("/analyze", json={"upload_id": negotiated["upload_id"]})
        body = r.json()
        stats = asyncio.run(q.stats(IMAGE_QUEUE))
    assert r.status_code == 202
    assert body["state"] == "queued"
    assert body["status_url"] == f"/jobs/{body['job_id']}"
    assert body["result_url"] == f"/results/{body['job_id']}"
    assert stats.ready == 1


def test_analyze_before_uploading_is_409():
    with platform() as (c, _, _):
        negotiated = negotiate(c)
        r = c.post("/analyze", json={"upload_id": negotiated["upload_id"]})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == ErrorCode.UPLOAD_INCOMPLETE.value


def test_analyze_with_an_unknown_upload_is_404():
    with platform() as (c, _, _):
        r = c.post("/analyze", json={"upload_id": "u_nope"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == ErrorCode.UPLOAD_NOT_FOUND.value


def test_resubmitting_the_same_content_returns_the_original_job():
    """Idempotency is visible to the client: it must use the id it receives."""
    with platform() as (c, _, q):
        first = submit(c)
        n2 = negotiate(c)
        put_bytes(c, n2)
        second = c.post("/analyze", json={"upload_id": n2["upload_id"]}).json()["job_id"]
        stats = asyncio.run(q.stats(IMAGE_QUEUE))
    assert second == first, "a duplicate submission created a second job"
    assert stats.ready == 1


def test_video_content_routes_to_the_video_queue():
    video = b"video payload" * 64
    with platform() as (c, _, q):
        negotiated = negotiate(c, video, content_type="video/mp4", media_kind="video")
        put_bytes(c, negotiated, video)
        c.post("/analyze", json={"upload_id": negotiated["upload_id"]})
        image_depth = asyncio.run(q.stats("image")).ready
        video_depth = asyncio.run(q.stats("video")).ready
    assert (image_depth, video_depth) == (0, 1)


# --------------------------------------------------------------------------
# GET /jobs/{job_id}
# --------------------------------------------------------------------------


def test_job_status_is_queued_before_a_worker_runs():
    with platform() as (c, _, _):
        job_id = submit(c)
        r = c.get(f"/jobs/{job_id}")
    assert r.status_code == 200
    status = JobStatus.model_validate(r.json())
    assert status.state.value == "queued"
    assert status.result_url is None, "no result exists yet"


def test_job_status_is_succeeded_after_a_worker_runs():
    with platform() as (c, app, q):
        job_id = submit(c)
        drain(app, q)
        r = c.get(f"/jobs/{job_id}")
    body = r.json()
    assert body["state"] == "succeeded"
    assert body["result_url"] == f"/results/{job_id}"
    assert body["finished_at"] is not None


def test_job_status_is_failed_after_a_permanent_error():
    with platform() as (c, app, q):
        job_id = submit(c)
        drain(app, q, handler=Exploding(PermanentError("unreadable container")))
        r = c.get(f"/jobs/{job_id}")
    body = r.json()
    assert body["state"] == "failed"
    assert body["error"]["retryable"] is False
    assert body["result_url"] is None


def test_job_status_stays_queued_while_a_retry_is_backing_off():
    with platform() as (c, app, q):
        job_id = submit(c)
        drain(app, q, handler=Exploding(TransientError("storage blip")))
        r = c.get(f"/jobs/{job_id}")
        envelope = asyncio.run(q.get(job_id))
    assert envelope.state is QueueState.DELAYED
    assert r.json()["state"] == "queued", "a backing-off retry is still queued work"


def test_unknown_job_is_404():
    with platform() as (c, _, _):
        r = c.get(f"/jobs/{UNKNOWN_JOB}")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == ErrorCode.JOB_NOT_FOUND.value


def test_malformed_job_id_is_400_not_404():
    with platform() as (c, _, _):
        r = c.get("/jobs/not-a-uuid")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == ErrorCode.INVALID_REQUEST.value


def test_job_urls_follow_the_api_prefix():
    with platform(api_prefix="api/v1") as (c, _, _):
        negotiated = c.post("/api/v1/uploads", json=_upload_body()).json()
        c.put(negotiated["upload"]["url"], content=IMAGE)
        body = c.post(
            "/api/v1/analyze", json={"upload_id": negotiated["upload_id"]}
        ).json()
    assert body["status_url"].startswith("/api/v1/jobs/")
    assert body["result_url"].startswith("/api/v1/results/")


# --------------------------------------------------------------------------
# GET /results/{job_id}
# --------------------------------------------------------------------------


def test_result_is_409_while_the_job_is_queued():
    with platform() as (c, _, _):
        job_id = submit(c)
        r = c.get(f"/results/{job_id}")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == ErrorCode.RESULT_NOT_READY.value
    assert r.headers.get("Retry-After") == "2"


def test_result_is_a_valid_bundle_after_the_worker_runs():
    with platform() as (c, app, q):
        job_id = submit(c)
        drain(app, q)
        r = c.get(f"/results/{job_id}")
    assert r.status_code == 200
    bundle = EvidenceBundle.model_validate(r.json())
    assert str(bundle.job_id) == job_id
    assert bundle.media.sha256 == IMAGE_SHA
    assert "stub" in bundle.model.version


def test_result_of_a_failed_job_surfaces_its_own_error():
    """A generic 404 would send a user to retry a file that can never work."""
    with platform() as (c, app, q):
        job_id = submit(c)
        drain(app, q, handler=Exploding(PermanentError("moov atom not found")))
        r = c.get(f"/results/{job_id}")
    err = r.json()["error"]
    assert r.status_code >= 400
    assert err["retryable"] is False
    assert "moov atom" in (err["detail"] or "")


def test_unknown_result_is_404():
    with platform() as (c, _, _):
        r = c.get(f"/results/{UNKNOWN_JOB}")
    assert r.status_code == 404


def test_insufficient_evidence_is_returned_as_a_200():
    """The three-state verdict, end to end. A 200 with a null p_fake."""

    class Abstains(StubImageHandler):
        async def handle(self, request, progress):
            from contracts.common import QualityReport
            from contracts.enums import AbstainReason, ResolutionClass

            bundle = await super().handle(request, progress)
            return bundle.model_copy(
                update={
                    "verdict": Verdict.INSUFFICIENT_EVIDENCE,
                    "p_fake": None,
                    "interval": None,
                    "abstain_reason": AbstainReason.FACE_TOO_SMALL,
                    "quality": QualityReport(
                        face_px=20,
                        resolution_class=ResolutionClass.TINY,
                        gate_passed=False,
                        gate_reason=AbstainReason.FACE_TOO_SMALL,
                    ),
                }
            )

    with platform() as (c, app, q):
        job_id = submit(c)
        drain(app, q, handler=Abstains())
        result = c.get(f"/results/{job_id}")
        status = c.get(f"/jobs/{job_id}")

    assert result.status_code == 200, "an abstention must not be an error"
    body = result.json()
    assert body["verdict"] == "insufficient_evidence"
    assert body["p_fake"] is None
    assert "error" not in body
    assert status.json()["state"] == "succeeded"


# --------------------------------------------------------------------------
# Authority model: a persisted result outranks every queue state
# --------------------------------------------------------------------------


def test_result_outranks_a_leased_envelope():
    """The worker persists before it acks, so this window is real."""
    with platform() as (c, app, q):
        job_id = submit(c)
        bundle = asyncio.run(
            StubImageHandler().handle(
                _request_for(job_id), _NullProgress()
            )
        )
        asyncio.run(app.state.sink.publish(bundle))
        asyncio.run(q.lease(IMAGE_QUEUE, worker_id="slow", lease_seconds=60))

        envelope = asyncio.run(q.get(job_id))
        status = c.get(f"/jobs/{job_id}").json()
        result = c.get(f"/results/{job_id}")

    assert envelope.state is QueueState.LEASED
    assert status["state"] == "succeeded", "a stored result was overruled by the queue"
    assert result.status_code == 200


def test_result_outranks_a_dead_lettered_envelope():
    """Reachable: lease expiry counts as an attempt, so a job with one attempt
    that persists and then dies is reaped straight to the DLQ with a good
    bundle on disk."""
    with platform() as (c, app, q):
        job_id = submit(c)
        bundle = asyncio.run(
            StubImageHandler().handle(_request_for(job_id), _NullProgress())
        )
        asyncio.run(app.state.sink.publish(bundle))

        async def kill_it():
            await q.lease(IMAGE_QUEUE, worker_id="doomed", lease_seconds=0.05)
            await asyncio.sleep(0.08)
            envelope = await q.get(job_id)
            await q._lock.acquire()
            q._jobs[job_id] = envelope.model_copy(update={"max_attempts": 1})
            q._lock.release()
            await q.reap_expired_leases()
            return await q.get(job_id)

        envelope = asyncio.run(kill_it())
        status = c.get(f"/jobs/{job_id}").json()
        result = c.get(f"/results/{job_id}")
        dead = asyncio.run(q.dead_letters(IMAGE_QUEUE))

    assert envelope.state is QueueState.DEAD
    assert status["state"] == "succeeded", "a stored result was overruled by a dead letter"
    assert result.status_code == 200
    assert len(dead) == 1, "the dead letter must remain visible to operators"


def _request_for(job_id: str):
    from contracts.job import JobSpec
    from workers.types import JobRequest

    return JobRequest(
        spec=JobSpec(
            job_id=job_id,
            media_kind=MediaKind.IMAGE,
            storage_key="uploads/x",
            sha256=IMAGE_SHA,
            size_bytes=len(IMAGE),
            declared_mime="image/jpeg",
            model_version="stub-0",
            idempotency_key=f"{IMAGE_SHA}:stub-0",
            submitted_at=datetime.now(UTC),
        ),
        media=IMAGE,
    )


class _NullProgress:
    async def __call__(self, fraction, *, stage=None):
        return None


# --------------------------------------------------------------------------
# GET /health
# --------------------------------------------------------------------------


def test_health_is_ok_and_lists_every_branch():
    with platform() as (c, _, _):
        r = c.get("/health")
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "ok"
    assert [b["id"] for b in body["branches"]] == ["A", "B", "P"]
    assert set(body["queue_depth"]) == {"image", "video"}


def test_health_reports_queue_depth():
    with platform() as (c, _, _):
        submit(c)
        body = c.get("/health").json()
    assert body["queue_depth"]["image"] == 1


def test_health_is_503_when_storage_is_unreachable():
    with platform() as (c, app, _):
        original = app.state.storage.health

        async def broken() -> StorageHealth:
            return StorageHealth(ok=False, backend="local", detail="disk offline")

        app.state.storage.health = broken
        r = c.get("/health")
        app.state.storage.health = original

    assert r.status_code == 503
    assert r.json()["status"] == "degraded"
    assert all(b["loaded"] is False for b in r.json()["branches"])


def test_health_stays_at_the_root_when_the_api_is_prefixed():
    """A probe URL configured in a load balancer must not move with a setting."""
    with platform(api_prefix="api/v1") as (c, _, _):
        assert c.get("/health").status_code == 200
        assert c.get("/api/v1/health").status_code == 404
        assert c.post("/api/v1/uploads", json=_upload_body()).status_code == 200
        assert c.post("/uploads", json=_upload_body()).status_code == 404


# --------------------------------------------------------------------------
# AnalysisService, without HTTP
# --------------------------------------------------------------------------


def test_service_holds_no_state_of_its_own():
    """The fourth-source-of-truth check, made mechanical.

    Every attribute must be one of the four collaborators. A cache or a status
    map appearing here is the drift this architecture exists to prevent.
    """
    with platform() as (_, app, _):
        service: AnalysisService = app.state.service
        assert set(vars(service)) == {"settings", "storage", "queue", "sink"}


def test_service_derives_status_without_a_store():
    with platform() as (c, app, q):
        job_id = submit(c)
        service: AnalysisService = app.state.service
        before = asyncio.run(service.job_status(uuid_of(job_id)))
        drain(app, q)
        after = asyncio.run(service.job_status(uuid_of(job_id)))
    assert before.state.value == "queued"
    assert after.state.value == "succeeded"


def uuid_of(job_id: str):
    from uuid import UUID

    return UUID(job_id)


# --------------------------------------------------------------------------
# Cross-cutting
# --------------------------------------------------------------------------


def test_every_error_path_uses_the_same_envelope():
    with platform() as (c, _, _):
        responses = [
            c.get("/nonexistent-path"),
            c.get(f"/jobs/{UNKNOWN_JOB}"),
            c.get("/jobs/not-a-uuid"),
            c.get(f"/results/{UNKNOWN_JOB}"),
            c.post("/uploads", json={"nonsense": True}),
            c.post("/uploads", json=_upload_body(size_bytes=99_999_999)),
            c.post("/analyze", json={"upload_id": "u_nope"}),
            c.request("DELETE", "/health"),
        ]
        for r in responses:
            assert r.status_code >= 400, r.request.url
            ErrorResponse.model_validate(r.json())


def test_request_id_is_echoed_and_honoured():
    with platform() as (c, _, _):
        assert c.get("/health").headers["X-Request-ID"]
        supplied = c.get("/health", headers={"X-Request-ID": "trace-me-123"})
        assert supplied.headers["X-Request-ID"] == "trace-me-123"


def test_cors_allows_put_for_the_upload_receiver():
    """Without PUT in the allowlist a browser preflight kills the upload."""
    with platform() as (c, _, _):
        r = c.options(
            "/uploads/u_x",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "PUT",
            },
        )
    assert "PUT" in r.headers.get("access-control-allow-methods", "")


def test_principal_is_anonymous_but_shaped_for_auth():
    assert ANONYMOUS.is_anonymous
    assert ANONYMOUS.rate_limit_key == "anonymous:anonymous"
    user = Principal(id="u_42", kind="user", scopes=("analyze",))
    assert not user.is_anonymous
    assert user.rate_limit_key == "user:u_42"


def test_logs_carry_correlation_and_never_personal_data():
    from backend.logging import configure_logging, get_logger, request_context

    buf = io.StringIO()
    configure_logging(level="INFO", json_output=True)
    logging.getLogger().handlers[0].stream = buf
    log = get_logger("test")
    with request_context("rid-1"):
        log.info(
            "upload negotiated",
            extra={
                "sha256": "abc123",
                "filename_hint": "my_passport_scan.jpg",
                "camera_model": "iPhone 15 Pro",
                "client_ip": "203.0.113.7",
            },
        )
    record = json.loads(buf.getvalue().strip())
    assert record["request_id"] == "rid-1"
    assert record["sha256"] == "abc123", "a content hash is how an operator correlates a report"
    for field in ("filename_hint", "camera_model", "client_ip"):
        assert record[field] == "[redacted]", f"{field} leaked into the log"


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_end_to_end_upload_analyze_worker_status_result_cache():
    """The whole platform, through the API only.

    upload → analyze → worker → job status → result → cache hit.
    """
    with platform() as (c, app, q):
        # 1. negotiate
        negotiated = negotiate(c)
        assert negotiated["cache_hit"] is False

        # 2. upload the bytes
        assert put_bytes(c, negotiated).status_code == 204

        # 3. queue it
        accepted = c.post(
            "/analyze", json={"upload_id": negotiated["upload_id"]}
        ).json()
        job_id = accepted["job_id"]

        # 4. not ready yet
        assert c.get(f"/jobs/{job_id}").json()["state"] == "queued"
        assert c.get(f"/results/{job_id}").status_code == 409

        # 5. a real worker does the work
        report = drain(app, q)
        assert report.completed == 1

        # 6. status flips, result is fetchable and valid
        status = c.get(f"/jobs/{job_id}").json()
        assert status["state"] == "succeeded"
        result = c.get(status["result_url"])
        assert result.status_code == 200
        bundle = EvidenceBundle.model_validate(result.json())
        assert str(bundle.job_id) == job_id

        # 7. the same content is now answered without an upload at all
        repeat = negotiate(c)
        assert repeat["cache_hit"] is True
        assert repeat["upload"] is None
        assert repeat["cached_result"]["verdict"] == bundle.verdict.value


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
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
