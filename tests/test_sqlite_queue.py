"""``SQLiteQueue`` — the queue the API and a separate worker process can share.

Before this backend, ``python -m backend.worker_main`` next to
``uvicorn backend.app:create_app`` could not complete a single job: the only
working queue was in-process, so each process had its own, and every
submission stayed ``queued`` forever with no error anywhere. The last test in
this file runs exactly that deployment — the API in this process, the worker
as a real child process — and requires a finished result.

Layers:

* the shared conformance suite (same 33 checks ``InMemoryQueue`` passes);
* properties only a durable, multi-process backend has: survives a restart,
  no double delivery across racing processes, a reaped worker cannot ack;
* integrity: schema-version refusal, non-JSON payloads rejected before write;
* wiring: registry, settings, the production guard, the worker boot guard.

    python tests/test_sqlite_queue.py
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import Settings  # noqa: E402
from backend.worker_main import require_shared_queue  # noqa: E402
from jobqueue.base import QueueBackend  # noqa: E402
from jobqueue.errors import LeaseNotHeld  # noqa: E402
from jobqueue.registry import SHARED_KINDS, QueueConfig, build_queue  # noqa: E402
from jobqueue.sqlite import SCHEMA_VERSION, SQLiteQueue  # noqa: E402
from jobqueue.types import JobState  # noqa: E402
from tests.queue_conformance import CHECKS, run_conformance  # noqa: E402

Q = "image"
_counter = itertools.count()


def _db() -> Path:
    """A fresh database path in a fresh directory (the file does not exist yet)."""
    return Path(tempfile.mkdtemp(prefix="dfp-sqliteq-")) / f"q{next(_counter)}.sqlite3"


# --------------------------------------------------------------------------
# Conformance
# --------------------------------------------------------------------------


def test_sqlite_queue_passes_the_conformance_suite():
    failures = asyncio.run(run_conformance(lambda: SQLiteQueue(_db())))
    assert failures == [], "conformance failures:\n  " + "\n  ".join(failures)
    assert len(CHECKS) >= 30


def test_sqlite_queue_satisfies_the_protocol():
    assert isinstance(SQLiteQueue(_db()), QueueBackend)


# --------------------------------------------------------------------------
# Durability and sharing
# --------------------------------------------------------------------------


def test_jobs_survive_a_restart():
    """A second instance on the same file is a restarted process: it must see
    the job, its payload and its place in line."""
    path = _db()

    async def scenario():
        first = SQLiteQueue(path)
        a = await first.enqueue(Q, {"n": 1}, idempotency_key="k1")
        b = await first.enqueue(Q, {"n": 2})
        restarted = SQLiteQueue(path)
        again = await restarted.enqueue(Q, {"n": 99}, idempotency_key="k1")
        leased = await restarted.lease(Q, worker_id="w", count=5)
        return a, b, again, leased

    a, b, again, leased = asyncio.run(scenario())
    assert again.job_id == a.job_id, "idempotency did not survive a restart"
    assert [j.job_id for j in leased] == [a.job_id, b.job_id], "FIFO lost across restart"
    assert leased[0].payload == {"n": 1}


def test_a_lease_taken_by_one_instance_is_acked_through_another():
    path = _db()

    async def scenario():
        producer, consumer = SQLiteQueue(path), SQLiteQueue(path)
        job = await producer.enqueue(Q, {})
        [held] = await consumer.lease(Q, worker_id="w1")
        done = await producer.ack(held.job_id, held.lease.lease_id)
        return job, done, await consumer.get(job.job_id)

    job, done, seen = asyncio.run(scenario())
    assert done.state is JobState.DONE
    assert seen.state is JobState.DONE and seen.attempts == 1


def test_a_reaped_worker_cannot_ack_the_job_another_worker_now_holds():
    """The compare-and-act on ``lease_id`` must hold across instances, not just
    inside one object's lock."""
    path = _db()

    async def scenario():
        slow, fast = SQLiteQueue(path), SQLiteQueue(path)
        await slow.enqueue(Q, {}, max_attempts=3)
        [stale] = await slow.lease(Q, worker_id="slow", lease_seconds=0.05)
        await asyncio.sleep(0.08)
        assert await fast.reap_expired_leases() == 1
        # Make the backed-off job due now, as time passing would.
        with sqlite3.connect(path) as conn:
            conn.execute("UPDATE jobs SET available_at = 0")
        [fresh] = await fast.lease(Q, worker_id="fast")
        try:
            await slow.ack(stale.job_id, stale.lease.lease_id)
        except LeaseNotHeld:
            pass
        else:
            raise AssertionError("a stale worker acked a job re-leased to another")
        return fresh, await fast.get(fresh.job_id)

    fresh, current = asyncio.run(scenario())
    assert current.state is JobState.LEASED, "the stale ack disturbed the new holder"
    assert fresh.envelope.attempts == 1, "the lapsed lease was not counted as an attempt"


_DRAIN = textwrap.dedent(
    """
    import asyncio, json, sys
    sys.path.insert(0, {root!r})
    from jobqueue.sqlite import SQLiteQueue

    async def main():
        q = SQLiteQueue({path!r})
        got = []
        while True:
            batch = await q.lease("image", worker_id={name!r}, count=3)
            if not batch:
                break
            for job in batch:
                got.append(job.job_id)
                await q.ack(job.job_id, job.lease.lease_id)
        print(json.dumps(got))

    asyncio.run(main())
    """
)


def test_racing_processes_never_receive_the_same_job():
    """The core guarantee, measured across real OS processes: every job is
    delivered, and none is delivered twice while the others are draining."""
    path = _db()
    total = 150

    async def fill():
        q = SQLiteQueue(path)
        return [(await q.enqueue(Q, {"i": i})).job_id for i in range(total)]

    enqueued = asyncio.run(fill())
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _DRAIN.format(root=str(ROOT), path=str(path), name=f"p{n}")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for n in range(4)
    ]
    delivered: list[str] = []
    per_process: list[int] = []
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
        ids = json.loads(out)
        per_process.append(len(ids))
        delivered.extend(ids)

    assert len(delivered) == len(set(delivered)), "a job was delivered to two processes"
    assert sorted(delivered) == sorted(enqueued), "a job was lost"
    stats = asyncio.run(SQLiteQueue(path).stats(Q))
    assert stats.ready == stats.leased == stats.delayed == 0


def test_concurrent_instances_in_one_process_do_not_double_lease():
    path = _db()

    async def scenario():
        await asyncio.gather(*(SQLiteQueue(path).enqueue(Q, {"i": i}) for i in range(40)))
        workers = [SQLiteQueue(path) for _ in range(6)]
        rounds = await asyncio.gather(
            *(w.lease(Q, worker_id=f"w{i}", count=10) for i, w in enumerate(workers))
        )
        return [j.job_id for batch in rounds for j in batch]

    leased = asyncio.run(scenario())
    assert len(leased) == 40
    assert len(set(leased)) == 40


def test_simultaneous_first_opens_of_a_new_file_all_succeed():
    """The API and a worker booting together against a file that does not exist
    yet. Switching to WAL is not covered by SQLite's busy handler, so without
    the retry in ``_ensure_schema`` some openers failed at once with
    "database is locked" (44 of 600 here, before the fix)."""

    async def one(path: Path, i: int) -> str | None:
        try:
            await SQLiteQueue(path).enqueue(Q, {"i": i})
        except Exception as exc:  # noqa: BLE001 - collected and reported below
            return f"{type(exc).__name__}: {exc}"
        return None

    async def scenario():
        errors, counts = [], []
        for _ in range(10):
            path = _db()
            results = await asyncio.gather(*(one(path, i) for i in range(40)))
            errors.extend(r for r in results if r)
            counts.append((await SQLiteQueue(path).stats(Q)).ready)
        return errors, counts

    errors, counts = asyncio.run(scenario())
    assert errors == [], f"{len(errors)} first-opens failed, e.g. {errors[:2]}"
    assert counts == [40] * 10, counts


# --------------------------------------------------------------------------
# Integrity
# --------------------------------------------------------------------------


def test_a_database_from_another_schema_version_is_refused():
    path = _db()
    asyncio.run(SQLiteQueue(path).enqueue(Q, {}))
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    try:
        asyncio.run(SQLiteQueue(path).get("anything"))
    except RuntimeError as exc:
        assert "schema version" in str(exc)
    else:
        raise AssertionError("opened a queue file written by a different layout")
    health = asyncio.run(SQLiteQueue(path).health())
    assert health.ok is False and "schema version" in (health.detail or "")


def test_a_non_json_payload_is_rejected_before_anything_is_written():
    path = _db()
    q = SQLiteQueue(path)
    try:
        asyncio.run(q.enqueue(Q, {"when": object()}))
    except TypeError:
        pass
    else:
        raise AssertionError("an unserialisable payload was accepted")
    assert asyncio.run(q.stats(Q)).backlog == 0


def test_timestamps_round_trip_to_the_microsecond():
    q = SQLiteQueue(_db())
    job = asyncio.run(q.enqueue(Q, {}, delay_s=12.345678))
    again = asyncio.run(q.get(job.job_id))
    assert again.enqueued_at == job.enqueued_at
    assert again.available_at == job.available_at
    assert again.enqueued_at.tzinfo is not None


def test_an_unusable_path_is_unhealthy_rather_than_fatal():
    """``create_app`` builds the queue at boot; a bad path must show up as a
    failing readiness probe, not as an exception out of the factory."""
    blocker = Path(tempfile.mkdtemp(prefix="dfp-sqliteq-")) / "a-file"
    blocker.write_text("not a directory")
    q = SQLiteQueue(blocker / "queue.sqlite3")  # parent is a file
    health = asyncio.run(q.health())
    assert health.ok is False
    assert health.backend == "sqlite"


def test_lease_rejects_a_non_positive_duration():
    q = SQLiteQueue(_db())
    asyncio.run(q.enqueue(Q, {}))
    try:
        asyncio.run(q.lease(Q, worker_id="w", lease_seconds=0))
    except ValueError:
        pass
    else:
        raise AssertionError("a zero-length lease was issued")
    assert asyncio.run(q.stats(Q)).ready == 1, "a rejected lease still claimed the job"


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_registry_builds_a_sqlite_queue():
    path = _db()
    q = build_queue(QueueConfig(kind="sqlite", path=path))
    assert isinstance(q, SQLiteQueue)
    assert q.name == "sqlite" and q.path == path.absolute()


def test_a_relative_path_is_pinned_to_the_starting_directory():
    """Resolved once, at construction — not re-resolved per connection, and so
    not dependent on a later chdir."""
    q = SQLiteQueue("var/queue.sqlite3")
    assert q.path.is_absolute()
    assert q.path == (Path.cwd() / "var" / "queue.sqlite3")
    assert str(q.path) in repr(q), "the startup log must show which file was opened"


def test_registry_requires_a_path_for_sqlite():
    try:
        QueueConfig(kind="sqlite")
    except ValueError as exc:
        assert "path is required" in str(exc)
    else:
        raise AssertionError("QueueConfig accepted sqlite without a path")


def test_settings_map_the_queue_path_through():
    path = _db()
    config = Settings(queue_kind="sqlite", queue_path=path).queue_config()
    assert config.kind == "sqlite" and config.path == path
    assert Settings().queue_config().path is None, "memory must not carry a path"


def test_production_refuses_an_in_process_queue():
    secret = "a-real-secret-for-this-test-only"
    try:
        Settings(environment="prod", signing_secret=secret)
    except ValueError as exc:
        assert "queue_kind='memory'" in str(exc)
    else:
        raise AssertionError("production accepted a queue the worker cannot reach")
    ok = Settings(
        environment="prod", signing_secret=secret, queue_kind="sqlite", rate_limit_kind="sqlite"
    )
    assert ok.queue_kind == "sqlite"


def test_worker_refuses_to_boot_on_an_unshared_queue():
    try:
        require_shared_queue("memory")
    except RuntimeError as exc:
        assert "DFP_QUEUE_KIND=sqlite" in str(exc), "the error must name the fix"
    else:
        raise AssertionError("a worker started on a queue the API cannot reach")
    for kind in SHARED_KINDS:
        require_shared_queue(kind)
    assert "memory" not in SHARED_KINDS


def test_sqlite_module_stays_inside_the_queue_package_boundary():
    import jobqueue.sqlite

    source = Path(jobqueue.sqlite.__file__).read_text(encoding="utf-8")
    for token in ("fastapi", "starlette", "import torch", "from backend", "from contracts",
                  "from storage", "import redis"):
        assert token not in source, f"jobqueue.sqlite references {token}"


# --------------------------------------------------------------------------
# The deployment this exists for
# --------------------------------------------------------------------------


def test_api_and_a_separate_worker_process_complete_a_job():
    """API here, ``python -m backend.worker_main`` as a child process, one
    SQLite file and one storage root between them.

    Both jobs are submitted *before* the worker exists, so the worker must find
    work that was already persisted when it booted. One is a real image and
    must succeed; one is a truncated JPEG, which uploads fine (checksum and
    size match) but cannot decode, and must end ``failed`` with its error
    readable through the API — not sit in ``queued`` or vanish.
    """
    from fastapi.testclient import TestClient

    from backend.app import create_app
    from contracts.evidence import EvidenceBundle
    from tests.test_backend import IMAGE

    corrupt = IMAGE[:400]
    workdir = Path(tempfile.mkdtemp(prefix="dfp-2proc-"))
    queue_path, storage_root = workdir / "queue.sqlite3", workdir / "storage"
    settings = Settings(
        log_json=False,
        log_level="CRITICAL",
        queue_kind="sqlite",
        queue_path=queue_path,
        storage_root=storage_root,
    )
    env = {
        **os.environ,
        "DFP_QUEUE_KIND": "sqlite",
        "DFP_QUEUE_PATH": str(queue_path),
        "DFP_STORAGE_ROOT": str(storage_root),
        "DFP_LOG_LEVEL": "INFO",
    }

    def submit(c, data: bytes) -> str:
        body = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "content_type": "image/jpeg",
            "media_kind": "image",
        }
        negotiated = c.post("/uploads", json=body).json()
        assert c.put(negotiated["upload"]["url"], content=data).status_code == 204
        accepted = c.post("/analyze", json={"upload_id": negotiated["upload_id"]})
        assert accepted.status_code == 202, accepted.text
        return accepted.json()["job_id"]

    log_path = workdir / "worker.log"

    def worker_log() -> str:
        return log_path.read_text(encoding="utf-8", errors="replace")

    with open(log_path, "w", encoding="utf-8") as log, TestClient(create_app(settings)) as c:
        good, bad = submit(c, IMAGE), submit(c, corrupt)
        assert c.get(f"/jobs/{good}").json()["state"] == "queued"

        worker = subprocess.Popen(
            [sys.executable, "-m", "backend.worker_main"],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 180
            states: dict[str, str] = {}
            while time.monotonic() < deadline:
                assert worker.poll() is None, "the worker exited:\n" + worker_log()
                states = {j: c.get(f"/jobs/{j}").json()["state"] for j in (good, bad)}
                if all(s in ("succeeded", "failed") for s in states.values()):
                    break
                time.sleep(0.25)
        finally:
            worker.terminate()
            worker.wait(timeout=30)

        assert states.get(good) == "succeeded", f"{states}; worker log:\n" + worker_log()
        result = c.get(f"/results/{good}")
        assert result.status_code == 200
        assert str(EvidenceBundle.model_validate(result.json()).job_id) == good

        assert states.get(bad) == "failed", f"{states}; worker log:\n" + worker_log()
        failed = c.get(f"/results/{bad}")
        assert failed.status_code >= 400
        assert failed.json()["error"]["retryable"] is False

        dead = asyncio.run(SQLiteQueue(queue_path).dead_letters("image"))
        assert [d.job_id for d in dead] == [bad], "the corrupt job is not in the DLQ"
        assert dead[0].attempts == 1, "a permanent failure must not be retried"

    # JSON logs escape backslashes, so accept either spelling of the path.
    logged = worker_log()
    assert str(queue_path) in logged or json.dumps(str(queue_path))[1:-1] in logged, (
        "the worker did not log which queue file it opened"
    )


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
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:600]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
