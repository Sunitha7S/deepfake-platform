"""Worker ↔ inference integration.

Proves the seam: a worker processes a job by calling the inference engine, and
every existing worker behaviour — leasing, retries, persistence, heartbeats,
progress batching — continues unchanged.

The regression tests at the bottom matter as much as the integration ones. The
failure this batch is guarding against is not "inference breaks" but "workers
quietly go back to scoring for themselves", which no functional test would
notice because the output would still be a valid bundle.
"""

from __future__ import annotations

import ast
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

from contracts.enums import JobStage, MediaKind, Verdict  # noqa: E402
from contracts.evidence import EvidenceBundle  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from inference.protocol import ModelUnavailable, UnreadableMedia, UnsupportedMedia  # noqa: E402
from jobqueue.memory import InMemoryQueue  # noqa: E402
from jobqueue.registry import IMAGE_QUEUE, VIDEO_QUEUE  # noqa: E402
from jobqueue.types import JobState  # noqa: E402
from storage import HmacUrlSigner  # noqa: E402
from storage.local import LocalStorageBackend  # noqa: E402
from workers.engine_registry import (  # noqa: E402
    get_image_engine,
    get_video_engine,
    loaded_engines,
    shutdown_engines,
)
from workers.image import ImageHandler  # noqa: E402
from workers.inference_adapter import to_handler_error, translated  # noqa: E402
from workers.io import InMemoryResultSink  # noqa: E402
from workers.registry import build_worker, handler_for  # noqa: E402
from workers.runner import WorkerRunner  # noqa: E402
from workers.types import ExecutionOutcome, JobRequest, PermanentError, TransientError  # noqa: E402
from workers.video import VideoHandler  # noqa: E402

SECRET = "integration-signing-secret-0123456789"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _face_image(seed: int = 1, size: int = 400) -> bytes:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.rectangle(img, (120, 120), (280, 300), (220, 200, 180), -1)
    cv2.circle(img, (170, 190), 14, (40, 40, 40), -1)
    cv2.circle(img, (235, 190), 14, (40, 40, 40), -1)
    cv2.ellipse(img, (202, 255), (40, 18), 0, 0, 180, (60, 40, 40), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _gate_passing_image() -> bytes:
    """An image the face gate accepts, so the model path is actually exercised.

    A fixture that silently abstains makes downstream assertions pass for the
    wrong reason, so this fails loudly rather than returning something unusable.
    """
    from inference.preprocess import decode_image, detect_faces

    for seed in range(24):
        data = _face_image(seed)
        if detect_faces(decode_image(data)):
            return data
    raise AssertionError("no synthetic face was detectable; rebuild the fixtures")


IMAGE = _gate_passing_image()


def _clip(frames: int = 60, size: int = 128, seed: int = 1) -> bytes:
    path = Path(tempfile.mkdtemp()) / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (size, size))
    rng = np.random.default_rng(seed)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


VIDEO = _clip()


def _spec(data: bytes, kind: MediaKind = MediaKind.IMAGE, job_id: UUID | None = None) -> JobSpec:
    return JobSpec(
        job_id=job_id or uuid4(),
        media_kind=kind,
        storage_key=f"uploads/{hashlib.sha256(data).hexdigest()[:16]}",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="image/jpeg" if kind is MediaKind.IMAGE else "video/mp4",
        model_version="stub-0",
        idempotency_key=f"{hashlib.sha256(data).hexdigest()}:stub-0",
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


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[float, JobStage | None]] = []

    async def __call__(self, fraction: float, *, stage: JobStage | None = None) -> None:
        self.calls.append((fraction, stage))

    def __len__(self) -> int:
        return len(self.calls)


def _runner(queue, handler, fetcher, sink, *, name=IMAGE_QUEUE, **kw) -> WorkerRunner:
    kw.setdefault("lease_seconds", 10.0)
    kw.setdefault("poll_interval_s", 0.02)
    return WorkerRunner(
        queue=queue, queue_name=name, handler=handler, fetcher=fetcher, sink=sink, **kw
    )


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_enqueue_worker_inference_sink_produces_a_valid_bundle():
    async def scenario():
        shutdown_engines()
        queue, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(IMAGE)
        fetch.add(spec, IMAGE)
        await queue.enqueue(IMAGE_QUEUE, spec.model_dump(mode="json"), job_id=str(spec.job_id))

        report = await _runner(queue, ImageHandler(), fetch, sink).run(
            max_jobs=1, idle_timeout_s=0.4
        )
        stored = await sink.find(str(spec.job_id))
        envelope = await queue.get(str(spec.job_id))
        return report, stored, envelope

    report, stored, envelope = asyncio.run(scenario())
    assert report.completed == 1
    assert envelope.state is JobState.DONE
    assert stored is not None
    EvidenceBundle.model_validate(stored.model_dump(mode="json"))
    assert stored.media.sha256 == hashlib.sha256(IMAGE).hexdigest()


def test_video_job_runs_end_to_end():
    async def scenario():
        shutdown_engines()
        queue, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(VIDEO, MediaKind.VIDEO)
        fetch.add(spec, VIDEO)
        await queue.enqueue(VIDEO_QUEUE, spec.model_dump(mode="json"), job_id=str(spec.job_id))
        report = await _runner(
            queue, VideoHandler(), fetch, sink, name=VIDEO_QUEUE
        ).run(max_jobs=1, idle_timeout_s=0.4)
        return report, await sink.find(str(spec.job_id))

    report, stored = asyncio.run(scenario())
    assert report.completed == 1
    EvidenceBundle.model_validate(stored.model_dump(mode="json"))
    assert stored.video is not None
    assert stored.media.kind is MediaKind.VIDEO


def test_end_to_end_over_real_storage():
    async def scenario():
        shutdown_engines()
        root = Path(tempfile.mkdtemp(prefix="dfp-int-"))
        storage = LocalStorageBackend(root, signer=HmacUrlSigner(SECRET))
        queue = InMemoryQueue()

        from storage.types import NewUpload

        session, signed = await storage.create_upload(
            NewUpload(
                sha256=hashlib.sha256(IMAGE).hexdigest(),
                size_bytes=len(IMAGE),
                content_type="image/jpeg",
            )
        )
        await storage.accept_upload(session.upload_id, IMAGE, signature=signed.signature)
        await storage.complete_upload(session.upload_id)

        spec = _spec(IMAGE).model_copy(update={"storage_key": session.key})
        await queue.enqueue(IMAGE_QUEUE, spec.model_dump(mode="json"), job_id=str(spec.job_id))

        worker = build_worker(MediaKind.IMAGE, queue=queue, storage=storage)
        worker.poll_interval_s = 0.02
        report = await worker.run(max_jobs=1, idle_timeout_s=0.4)

        from workers.io import StorageResultSink

        return report, await StorageResultSink(storage).find(str(spec.job_id))

    report, stored = asyncio.run(scenario())
    assert report.completed == 1
    assert stored is not None
    EvidenceBundle.model_validate(stored.model_dump(mode="json"))


def test_registry_wires_the_inference_backed_handlers():
    assert isinstance(handler_for(MediaKind.IMAGE), ImageHandler)
    assert isinstance(handler_for(MediaKind.VIDEO), VideoHandler)


# --------------------------------------------------------------------------
# Error mapping
# --------------------------------------------------------------------------


def test_unreadable_media_maps_to_permanent():
    mapped = to_handler_error(UnreadableMedia("bad bytes", detail="d"))
    assert isinstance(mapped, PermanentError)
    assert mapped.retryable is False


def test_unsupported_media_maps_to_permanent():
    assert isinstance(to_handler_error(UnsupportedMedia("too big")), PermanentError)


def test_model_unavailable_maps_to_transient():
    """The deployment is broken, not the input. Dead-lettering here would throw
    away a queue's worth of good jobs during a bad deploy."""
    mapped = to_handler_error(ModelUnavailable("weights missing"))
    assert isinstance(mapped, TransientError)
    assert mapped.retryable is True


def test_validation_error_maps_to_permanent():
    from pydantic import ValidationError

    try:
        EvidenceBundle.model_validate({"nonsense": True})
    except ValidationError as exc:
        mapped = to_handler_error(exc)
        assert isinstance(mapped, PermanentError)
        assert "contract violation" in (mapped.detail or "")
        return
    raise AssertionError("expected a ValidationError")


def test_unrecognised_exception_is_not_translated():
    """Returning None lets the runner apply its own default — transient — rather
    than dead-lettering real work on the first unhandled KeyError."""
    assert to_handler_error(KeyError("boom")) is None
    assert to_handler_error(RuntimeError("boom")) is None


def test_translated_context_manager_reraises_mapped_errors():
    try:
        with translated():
            raise UnreadableMedia("nope")
    except PermanentError:
        pass
    else:
        raise AssertionError("UnreadableMedia was not translated")

    try:
        with translated():
            raise KeyError("passthrough")
    except KeyError:
        return
    raise AssertionError("an unrecognised error was swallowed")


def test_corrupt_media_dead_letters_through_the_worker():
    async def scenario():
        shutdown_engines()
        queue, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        junk = b"definitely not an image"
        spec = _spec(junk)
        fetch.add(spec, junk)
        await queue.enqueue(IMAGE_QUEUE, spec.model_dump(mode="json"), job_id=str(spec.job_id))
        report = await _runner(queue, ImageHandler(), fetch, sink).run(
            max_jobs=1, idle_timeout_s=0.4
        )
        return report, await queue.dead_letters(IMAGE_QUEUE)

    report, dead = asyncio.run(scenario())
    assert report.dead_lettered == 1, "corrupt media should not be retried"
    assert len(dead) == 1


def test_missing_model_retries_through_the_worker():
    class Unavailable:
        media_kind = MediaKind.IMAGE.value
        name = "unavailable"

        async def handle(self, request, progress):
            with translated():
                raise ModelUnavailable("weights not loaded")

    async def scenario():
        queue, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        spec = _spec(IMAGE)
        fetch.add(spec, IMAGE)
        await queue.enqueue(IMAGE_QUEUE, spec.model_dump(mode="json"), job_id=str(spec.job_id))
        report = await _runner(queue, Unavailable(), fetch, sink).run(
            max_jobs=1, idle_timeout_s=0.4
        )
        return report, await queue.get(str(spec.job_id))

    report, envelope = asyncio.run(scenario())
    assert report.retried == 1, "a missing model should be retried, not dead-lettered"
    assert envelope.state is JobState.DELAYED


# --------------------------------------------------------------------------
# Engine lifecycle
# --------------------------------------------------------------------------


def test_engines_are_lazy():
    shutdown_engines()
    assert loaded_engines() == {"image": False, "video": False}
    get_image_engine()
    assert loaded_engines() == {"image": True, "video": False}, "the video engine was built eagerly"
    shutdown_engines()


def test_engines_are_singletons():
    shutdown_engines()
    assert get_image_engine() is get_image_engine()
    assert get_video_engine() is get_video_engine()
    assert get_image_engine() is not get_video_engine()
    shutdown_engines()


def test_shutdown_clears_the_cache_and_unloads_models():
    from inference import default_registry

    shutdown_engines()
    engine = get_image_engine()
    asyncio.run(engine.warmup())
    assert default_registry().get_loaded_models(), "warmup loaded nothing"

    unloaded = shutdown_engines()
    assert unloaded >= 1
    assert loaded_engines() == {"image": False, "video": False}
    assert default_registry().get_loaded_models() == {}
    assert get_image_engine() is not engine, "a stale engine survived shutdown"
    shutdown_engines()


def test_shutdown_is_idempotent():
    shutdown_engines()
    assert shutdown_engines() == 0


def test_engines_are_reused_across_jobs():
    async def scenario():
        shutdown_engines()
        queue, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        for i in range(3):
            data = _face_image(i + 40)
            spec = _spec(data)
            fetch.add(spec, data)
            await queue.enqueue(
                IMAGE_QUEUE, spec.model_dump(mode="json"), job_id=str(spec.job_id)
            )
        first = get_image_engine()
        await _runner(queue, ImageHandler(), fetch, sink).run(max_jobs=3, idle_timeout_s=0.4)
        return first is get_image_engine()

    assert asyncio.run(scenario()), "the engine was rebuilt between jobs"
    shutdown_engines()


# --------------------------------------------------------------------------
# Progress forwarding
# --------------------------------------------------------------------------


def test_video_progress_is_forwarded_exactly_as_inference_produced_it():
    """No extra events, no dropped events.

    The batching guarantee belongs to the inference pipeline. Comparing the
    worker's view against a direct engine call proves nothing sits in between
    adding or swallowing calls.
    """
    shutdown_engines()
    spec = _spec(VIDEO, MediaKind.VIDEO)

    direct = Recorder()
    asyncio.run(get_video_engine().analyze_video(spec, VIDEO, direct))

    through_handler = Recorder()
    asyncio.run(VideoHandler().handle(JobRequest(spec=spec, media=VIDEO), through_handler))

    assert through_handler.calls == direct.calls, "progress was altered in transit"
    assert len(through_handler) == 6, f"expected 6 batched calls, got {len(through_handler)}"
    assert all(stage is JobStage.SCORING for _, stage in through_handler.calls)
    shutdown_engines()


def test_image_progress_is_forwarded():
    shutdown_engines()
    recorder = Recorder()
    asyncio.run(ImageHandler().handle(JobRequest(spec=_spec(IMAGE), media=IMAGE), recorder))
    assert len(recorder) >= 1, "the handler reported no progress at all"
    fractions = [f for f, _ in recorder.calls]
    assert fractions == sorted(fractions)
    shutdown_engines()


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_same_bytes_processed_twice_produce_identical_bundles():
    shutdown_engines()
    job = uuid4()
    spec = _spec(IMAGE, job_id=job)
    handler = ImageHandler()
    a = asyncio.run(handler.handle(JobRequest(spec=spec, media=IMAGE), Recorder()))
    b = asyncio.run(handler.handle(JobRequest(spec=spec, media=IMAGE), Recorder()))
    assert a.verdict is b.verdict
    assert a.p_fake == b.p_fake
    assert [x.logit for x in a.branches] == [x.logit for x in b.branches]
    shutdown_engines()


def test_determinism_survives_an_engine_restart():
    """A worker restart must not change the answer, or crash recovery cannot
    compare results across attempts."""
    shutdown_engines()
    spec = _spec(IMAGE, job_id=uuid4())
    first = asyncio.run(ImageHandler().handle(JobRequest(spec=spec, media=IMAGE), Recorder()))
    shutdown_engines()
    second = asyncio.run(ImageHandler().handle(JobRequest(spec=spec, media=IMAGE), Recorder()))
    assert first.p_fake == second.p_fake
    shutdown_engines()


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


def test_concurrent_workers_share_one_engine():
    async def scenario():
        shutdown_engines()
        queue, sink, fetch = InMemoryQueue(), InMemoryResultSink(), DictFetcher()
        for i in range(8):
            data = _face_image(i + 60)
            spec = _spec(data)
            fetch.add(spec, data)
            await queue.enqueue(
                IMAGE_QUEUE, spec.model_dump(mode="json"), job_id=str(spec.job_id)
            )

        workers = [
            _runner(queue, ImageHandler(), fetch, sink, worker_id=f"w{i}") for i in range(4)
        ]
        reports = await asyncio.gather(*(w.run(idle_timeout_s=0.3) for w in workers))
        return reports, sink

    reports, sink = asyncio.run(scenario())
    assert sum(r.completed for r in reports) == 8
    assert len(sink) == 8

    from inference import default_registry

    loaded = default_registry().get_loaded_models()
    assert len(loaded) <= 1, f"more than one model resident: {sorted(loaded)}"
    assert get_image_engine() is get_image_engine()
    shutdown_engines()


def test_engine_construction_is_thread_safe():
    """Double-checked locking under real thread contention.

    ``asyncio.to_thread`` means two threads genuinely can reach the accessor at
    once; without the lock each would build an engine and one would be silently
    orphaned with its weights still resident.
    """
    import threading

    shutdown_engines()
    seen: list[object] = []
    barrier = threading.Barrier(8)

    def grab() -> None:
        barrier.wait()
        seen.append(get_image_engine())

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == 8
    assert len({id(x) for x in seen}) == 1, "concurrent callers built different engines"
    shutdown_engines()


# --------------------------------------------------------------------------
# Boundaries and regression protection
# --------------------------------------------------------------------------


def _imports_of(package: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted(Path(package).rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        mods: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            )
            mods.update(n.split(".")[0] for n in names)
        found[str(path)] = mods
    return found


def test_workers_import_inference():
    """The seam exists at all."""
    all_mods = set().union(*_imports_of("workers").values())
    assert "inference" in all_mods


def test_inference_imports_nothing_from_workers_or_infrastructure():
    forbidden = {"workers", "jobqueue", "storage", "backend", "fastapi", "starlette"}
    violations = [
        f"{path} -> {sorted(mods & forbidden)}"
        for path, mods in _imports_of("inference").items()
        if mods & forbidden
    ]
    assert not violations, "inference reached upward:\n  " + "\n  ".join(violations)


def test_dependency_graph_stays_acyclic():
    packages = ["contracts", "storage", "jobqueue", "inference", "workers", "backend"]
    edges = {
        pkg: (set().union(*_imports_of(pkg).values()) & set(packages)) - {pkg}
        for pkg in packages
    }
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(node: str, trail: list[str]) -> None:
        if node in visiting:
            raise AssertionError("cycle: " + " -> ".join([*trail, node]))
        if node in done:
            return
        visiting.add(node)
        for dep in sorted(edges[node]):
            visit(dep, [*trail, node])
        visiting.discard(node)
        done.add(node)

    for pkg in packages:
        visit(pkg, [])


def test_workers_contain_no_scoring_logic():
    """Regression guard.

    The failure being prevented is not "inference breaks" — it is workers
    quietly growing their own scorer again, which no functional test would
    catch because the output would still be a valid bundle. Scoring means
    turning content into a number: hashing media, building logits, fusing
    branches.
    """
    markers = ("hashlib.sha256(", "np.log(", "_unit(", "inverse_sigmoid", "branch_weights")
    offenders = []
    for path in sorted(Path("workers").rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker in text:
                offenders.append(f"{path} contains {marker!r}")
    assert not offenders, "scoring logic returned to workers:\n  " + "\n  ".join(offenders)


def test_the_old_stub_handler_module_is_gone_from_workers():
    assert not Path("workers/handlers.py").exists(), (
        "workers/handlers.py is back; stub scoring belongs in the test tree"
    )
    assert Path("tests/stub_handlers.py").exists(), "the relocated test double is missing"


def test_handlers_are_thin_delegations():
    """A handler that grew a decode step or a gate is a handler drifting back
    toward doing the work itself."""
    for path in ("workers/image.py", "workers/video.py"):
        text = Path(path).read_text(encoding="utf-8")
        assert "analyze_image" in text or "analyze_video" in text
        assert "cv2" not in text, f"{path} decodes media itself"
        assert "numpy" not in text and "np." not in text, f"{path} manipulates arrays"


def test_no_module_scope_torch_import_in_workers():
    offenders = []
    for path in sorted(Path("workers").rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            )
            for name in names:
                if name.split(".")[0] in ("torch", "onnxruntime"):
                    offenders.append(f"{path}:{node.lineno}")
    assert not offenders, f"module-scope heavy imports: {offenders}"


def test_importing_workers_does_not_load_torch():
    """Measured in a clean interpreter, not against this process.

    An in-process assertion here is order-dependent and therefore worthless:
    another suite legitimately exercises the torch backend's lazy-import path,
    which leaves torch resident. The property being guaranteed is that
    *importing* the worker and inference layers does not pull it in — and a
    subprocess is the only honest way to observe that.
    """
    import subprocess

    here = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0,'.'); "
            "import workers.image, workers.video, workers.engine_registry; "
            "print('torch' in sys.modules, 'onnxruntime' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=here,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip() == "False False", f"heavy imports leaked: {out.stdout!r}"


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
    shutdown_engines()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
