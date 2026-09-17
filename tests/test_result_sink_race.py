"""Result-sink race elimination (audit item D-3).

Before this, the sink read then wrote. Two workers could both observe a job's
key as free and both commit, with the *second* winning — the inverse of the
first-writer-wins contract the protocol advertised. With deterministic stubs
that was wasted work; with real models it is two different answers for one job
and no way to tell which was served.

The tests below prove the race is closed at three levels: the storage
primitive, the sink built on it, and a worker fleet racing end to end.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts import fixtures as fx  # noqa: E402
from contracts.enums import MediaKind  # noqa: E402
from contracts.evidence import EvidenceBundle  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from jobqueue.memory import InMemoryQueue  # noqa: E402
from jobqueue.registry import IMAGE_QUEUE  # noqa: E402
from jobqueue.types import JobState  # noqa: E402
from storage import HmacUrlSigner  # noqa: E402
from storage.local import LocalStorageBackend  # noqa: E402
from tests.sink_conformance import CHECKS, run_conformance  # noqa: E402
from workers.io import InMemoryResultSink, StorageResultSink  # noqa: E402
from workers.runner import WorkerRunner  # noqa: E402
from workers.types import JobRequest, PermanentError  # noqa: E402

SECRET = "sink-race-signing-secret-0123456789"


def _storage() -> LocalStorageBackend:
    return LocalStorageBackend(
        Path(tempfile.mkdtemp(prefix="dfp-sink-")), signer=HmacUrlSigner(SECRET)
    )


def _storage_sink() -> StorageResultSink:
    return StorageResultSink(_storage())


def _bundle(job_id, **over) -> EvidenceBundle:
    return fx.image_manipulated().model_copy(update={"job_id": job_id, **over})


# --------------------------------------------------------------------------
# Conformance
# --------------------------------------------------------------------------


def test_in_memory_sink_passes_conformance():
    failures = asyncio.run(run_conformance(InMemoryResultSink))
    assert failures == [], "in-memory sink:\n  " + "\n  ".join(failures)


def test_storage_sink_passes_conformance():
    failures = asyncio.run(run_conformance(_storage_sink))
    assert failures == [], "storage sink:\n  " + "\n  ".join(failures)


def test_conformance_suite_is_not_vacuous():
    assert len(CHECKS) >= 15


def test_conformance_suite_catches_a_check_then_write_sink():
    """Prove the suite has teeth by giving it the bug it was written for.

    This is the exact implementation D-3 replaced: read, then write. It passes
    every sequential test and fails the concurrent ones.
    """

    class RacySink(InMemoryResultSink):
        async def put_if_absent(self, job_id, bundle):
            if await self.find(job_id) is not None:
                return False
            await asyncio.sleep(0)  # the window a real backend's I/O opens
            self._results[job_id] = bundle
            self._by_content.setdefault(
                (bundle.media.sha256, bundle.model.version), job_id
            )
            return True

    failures = asyncio.run(run_conformance(RacySink))
    assert failures, "the suite accepted a check-then-write sink"
    assert any("concurrent" in f for f in failures), failures


def test_conformance_suite_catches_an_overwriting_sink():
    class OverwritingSink(InMemoryResultSink):
        async def put_if_absent(self, job_id, bundle):
            self._results[job_id] = bundle
            self._by_content[(bundle.media.sha256, bundle.model.version)] = job_id
            return True

    failures = asyncio.run(run_conformance(OverwritingSink))
    assert any("losing_write_does_not_change" in f for f in failures), failures


# --------------------------------------------------------------------------
# The storage primitive
# --------------------------------------------------------------------------


def test_put_object_if_absent_is_exclusive():
    storage = _storage()

    async def scenario():
        first = await storage.put_object_if_absent("results/x.json", b"first")
        second = await storage.put_object_if_absent("results/x.json", b"second")
        obj = await storage.get_object("results/x.json")
        return first, second, obj.data

    first, second, data = asyncio.run(scenario())
    assert (first, second) == (True, False)
    assert data == b"first", "the second writer overwrote the first"


def test_put_object_if_absent_never_publishes_a_partial_file():
    """A reader sees the whole object or no object.

    The obvious alternative — O_EXCL then write — claims the key with an empty
    file and fills it afterwards, so a concurrent read gets truncated JSON. For
    a result store that is worse than losing, because half a bundle parses as
    corrupt rather than as absent.
    """
    storage = _storage()
    payload = b'{"big":"' + b"x" * 200_000 + b'"}'

    async def scenario():
        reads: list[int | None] = []

        async def reader():
            for _ in range(60):
                try:
                    obj = await storage.get_object("results/big.json")
                    reads.append(len(obj.data))
                except Exception:
                    reads.append(None)
                await asyncio.sleep(0)

        await asyncio.gather(
            storage.put_object_if_absent("results/big.json", payload), reader()
        )
        return reads

    reads = asyncio.run(scenario())
    seen = {n for n in reads if n is not None}
    assert seen <= {len(payload)}, f"a partial read was observed: {sorted(seen)}"


def test_put_object_if_absent_leaves_no_temporary_files():
    storage = _storage()

    async def scenario():
        await storage.put_object_if_absent("results/a.json", b"a")
        await storage.put_object_if_absent("results/a.json", b"b")

    asyncio.run(scenario())
    leftovers = list(storage.objects_dir.rglob("*.tmp"))
    assert not leftovers, f"temporary files leaked: {leftovers}"


# --------------------------------------------------------------------------
# Different outputs — the case real models make possible
# --------------------------------------------------------------------------


def test_only_the_first_of_two_different_bundles_survives():
    for factory in (InMemoryResultSink, _storage_sink):
        sink = factory()

        async def scenario(sink=sink):
            job_id = uuid4()
            a = _bundle(job_id, disagreement=0.111)
            b = _bundle(job_id, disagreement=0.999)
            won = await asyncio.gather(
                sink.put_if_absent(str(job_id), a), sink.put_if_absent(str(job_id), b)
            )
            return won, await sink.find(str(job_id)), a, b

        won, stored, a, b = asyncio.run(scenario())
        assert sum(won) == 1, f"{factory}: {sum(won)} winners"
        winner = a if won[0] else b
        assert stored.disagreement == winner.disagreement, (
            f"{factory}: stored bundle is not the winner's"
        )


def test_committed_result_is_immutable_under_sustained_contention():
    sink = InMemoryResultSink()

    async def scenario():
        job_id = uuid4()
        await sink.put_if_absent(str(job_id), _bundle(job_id, disagreement=0.5))
        await asyncio.gather(
            *(
                sink.put_if_absent(str(job_id), _bundle(job_id, disagreement=i / 50))
                for i in range(50)
            )
        )
        return await sink.find(str(job_id))

    assert asyncio.run(scenario()).disagreement == 0.5


# --------------------------------------------------------------------------
# Worker behaviour
# --------------------------------------------------------------------------


def _image() -> bytes:
    rng = np.random.default_rng(3)
    return cv2.imencode(".jpg", rng.integers(0, 255, (240, 240, 3), dtype=np.uint8))[
        1
    ].tobytes()


IMAGE = _image()


def _spec(data: bytes, job_id: UUID | None = None) -> JobSpec:
    return JobSpec(
        job_id=job_id or uuid4(),
        media_kind=MediaKind.IMAGE,
        storage_key=f"uploads/{hashlib.sha256(data).hexdigest()[:16]}",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="image/jpeg",
        model_version="stub-0",
        idempotency_key="k",
        submitted_at=datetime.now(UTC),
    )


class DictFetcher:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def add(self, spec: JobSpec, data: bytes) -> None:
        self.objects[spec.storage_key] = data

    async def fetch(self, spec: JobSpec) -> bytes:
        try:
            return self.objects[spec.storage_key]
        except KeyError as exc:
            raise PermanentError("missing object") from exc


class CountingHandler:
    media_kind = MediaKind.IMAGE.value
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    async def handle(self, request: JobRequest, progress):
        self.calls += 1
        return _bundle(request.spec.job_id)


def test_a_losing_worker_acks_without_error():
    """Requirement: treat as complete, ack the lease, rewrite nothing, no error."""

    async def scenario():
        queue, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(IMAGE)
        fetch.add(spec, IMAGE)

        # A previous attempt already committed a different bundle.
        committed = _bundle(spec.job_id, disagreement=0.42)
        await sink.put_if_absent(str(spec.job_id), committed)

        await queue.enqueue(
            IMAGE_QUEUE, spec.model_dump(mode="json"), job_id=str(spec.job_id)
        )
        handler = CountingHandler()
        report = await WorkerRunner(
            queue=queue,
            queue_name=IMAGE_QUEUE,
            handler=handler,
            fetcher=fetch,
            sink=sink,
            lease_seconds=10.0,
            poll_interval_s=0.02,
        ).run(max_jobs=1, idle_timeout_s=0.3)

        return report, handler, await queue.get(str(spec.job_id)), await sink.find(
            str(spec.job_id)
        )

    report, handler, envelope, stored = asyncio.run(scenario())
    assert report.dead_lettered == 0 and report.retried == 0, "contention surfaced as failure"
    assert report.already_done == 1
    assert envelope.state is JobState.DONE, "the lease was not acked"
    assert handler.calls == 0, "inference re-ran despite an existing result"
    assert stored.disagreement == 0.42, "the committed result was rewritten"


def test_crash_recovery_still_skips_inference():
    """Existing behaviour preserved: a result found after restart short-circuits."""

    async def scenario():
        queue, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(IMAGE)
        fetch.add(spec, IMAGE)
        await sink.put_if_absent(str(spec.job_id), _bundle(spec.job_id))
        await queue.enqueue(
            IMAGE_QUEUE, spec.model_dump(mode="json"), job_id=str(spec.job_id)
        )
        handler = CountingHandler()
        report = await WorkerRunner(
            queue=queue,
            queue_name=IMAGE_QUEUE,
            handler=handler,
            fetcher=fetch,
            sink=sink,
            poll_interval_s=0.02,
        ).run(max_jobs=1, idle_timeout_s=0.3)
        return report, handler

    report, handler = asyncio.run(scenario())
    assert report.already_done == 1
    assert handler.calls == 0


def test_racing_worker_fleet_commits_exactly_one_result():
    """The end-to-end case. Six workers, one job, one committed bundle."""

    async def scenario():
        storage = _storage()
        queue, sink, fetch = InMemoryQueue(), StorageResultSink(storage), DictFetcher()
        spec = _spec(IMAGE)
        fetch.add(spec, IMAGE)
        await queue.enqueue(
            IMAGE_QUEUE, spec.model_dump(mode="json"), job_id=str(spec.job_id)
        )

        handlers = [CountingHandler() for _ in range(6)]
        workers = [
            WorkerRunner(
                queue=queue,
                queue_name=IMAGE_QUEUE,
                handler=h,
                fetcher=fetch,
                sink=sink,
                worker_id=f"w{i}",
                lease_seconds=10.0,
                poll_interval_s=0.02,
            )
            for i, h in enumerate(handlers)
        ]
        reports = await asyncio.gather(*(w.run(idle_timeout_s=0.3) for w in workers))
        results = list((storage.objects_dir / "results").glob("*.json"))
        return reports, handlers, results, await sink.find(str(spec.job_id))

    reports, handlers, results, stored = asyncio.run(scenario())
    assert sum(r.leased for r in reports) == 1, "the job was leased more than once"
    assert sum(h.calls for h in handlers) == 1, "inference ran more than once"
    assert len(results) == 1, f"{len(results)} result objects were written"
    assert stored is not None
    EvidenceBundle.model_validate(stored.model_dump(mode="json"))


def _run() -> int:
    tests = sorted((n, o) for n, o in globals().items() if n.startswith("test_") and callable(o))
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
