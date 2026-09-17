"""Pipeline, lifecycle, boundary and conformance tests for the inference layer.

Three groups worth distinguishing:

* **Pipelines** — determinism, the abstention paths, progress batching, parity.
* **Guards** — dependency boundaries and the torch-import guarantee. These fail
  loudly the moment someone adds a convenient import, which is the only thing
  keeping the layering honest between audits.
* **Conformance doubles** — deliberately broken implementations, proving the
  suite detects the failures it claims to. A test that has never caught
  anything is a hypothesis, not a check.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.enums import BranchId, JobStage, MediaKind, Verdict  # noqa: E402
from contracts.evidence import EvidenceBundle  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from contracts.version import is_composed  # noqa: E402
from inference import (  # noqa: E402
    ImagePipeline,
    ModelUnavailable,
    UnreadableMedia,
    UnsupportedMedia,
    VideoPipeline,
)
from inference.models import (  # noqa: E402
    ModelRegistry,
    ModelSpec,
    StubBackend,
    default_registry,
)

FORBIDDEN = ("fastapi", "starlette", "backend", "workers", "storage", "jobqueue")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def face_image(seed: int = 1, size: int = 400) -> bytes:
    """A crude face-like image, deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.rectangle(img, (120, 120), (280, 300), (220, 200, 180), -1)
    cv2.circle(img, (170, 190), 14, (40, 40, 40), -1)
    cv2.circle(img, (235, 190), 14, (40, 40, 40), -1)
    cv2.ellipse(img, (202, 255), (40, 18), 0, 0, 180, (60, 40, 40), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def flat_image(value: int = 128, size: int = 300) -> bytes:
    return cv2.imencode(".jpg", np.full((size, size, 3), value, np.uint8))[1].tobytes()


def clip(frames: int = 60, size: int = 128, fps: int = 25, seed: int = 1) -> bytes:
    path = Path(tempfile.mkdtemp()) / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    rng = np.random.default_rng(seed)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def spec(data: bytes, kind: MediaKind = MediaKind.IMAGE, job_id: UUID | None = None) -> JobSpec:
    return JobSpec(
        job_id=job_id or uuid4(),
        media_kind=kind,
        storage_key="uploads/x",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="image/jpeg" if kind is MediaKind.IMAGE else "video/mp4",
        model_version="stub-0",
        idempotency_key="k",
        submitted_at=datetime.now(UTC),
    )


def _first_gate_passing_seed(limit: int = 24) -> bytes:
    """An image the face gate actually accepts.

    Haar on synthetic drawings is unreliable — most seeds produce no detection,
    and a fixture that silently abstains makes every downstream assertion pass
    vacuously. This searches for one that works and fails loudly if none does,
    so a change in OpenCV's cascade surfaces as a broken fixture rather than as
    a suite that stopped testing anything.
    """
    from inference.preprocess import decode_image, detect_faces

    for seed in range(limit):
        data = face_image(seed)
        if detect_faces(decode_image(data)):
            return data
    raise AssertionError(
        f"no synthetic face was detectable in {limit} seeds; the image fixtures "
        "need rebuilding before the gate-passing tests mean anything"
    )


GATE_PASSING_IMAGE = _first_gate_passing_seed()


class Recorder:
    """Captures progress calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, JobStage | None]] = []

    async def __call__(self, fraction: float, *, stage: JobStage | None = None) -> None:
        self.calls.append((fraction, stage))

    def __len__(self) -> int:
        return len(self.calls)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_identical_bytes_produce_identical_image_bundles():
    data = face_image(1)
    job = uuid4()
    engine = ImagePipeline()
    a = asyncio.run(engine.analyze_image(spec(data, job_id=job), data))
    b = asyncio.run(engine.analyze_image(spec(data, job_id=job), data))
    assert a.verdict is b.verdict
    assert a.p_fake == b.p_fake
    assert [x.logit for x in a.branches] == [x.logit for x in b.branches]


def test_different_bytes_produce_different_scores():
    engine = ImagePipeline()
    scores = set()
    for seed in range(6):
        data = face_image(seed)
        scores.add(asyncio.run(engine.analyze_image(spec(data), data)).p_fake)
    assert len(scores) > 1, "every input scored identically"


def test_video_pipeline_is_deterministic_across_runs():
    data = clip()
    job = uuid4()
    engine = VideoPipeline()
    a = asyncio.run(engine.analyze_video(spec(data, MediaKind.VIDEO, job), data))
    b = asyncio.run(engine.analyze_video(spec(data, MediaKind.VIDEO, job), data))
    assert a.p_fake == b.p_fake
    assert [f.p for f in a.video.frame_scores] == [f.p for f in b.video.frame_scores]
    assert len(a.video.segments) == len(b.video.segments)


def test_scoring_does_not_depend_on_process_state_or_clock():
    """Run the same analysis in a fresh interpreter and compare.

    Wall clock, PYTHONHASHSEED and any accumulated process state all differ
    between the two, so a match rules out all three at once.
    """
    data = face_image(3)
    digest = hashlib.sha256(data).hexdigest()
    script = (
        "import asyncio,sys,hashlib;sys.path.insert(0,'.');"
        "from datetime import datetime,UTC;from uuid import UUID;"
        "from inference import ImagePipeline;from contracts.job import JobSpec;"
        "from contracts.enums import MediaKind;"
        f"d=open({str(Path(tempfile.mkdtemp()) / 'x.jpg')!r},'rb').read();"
        "s=JobSpec(job_id=UUID(int=7),media_kind=MediaKind.IMAGE,storage_key='u',"
        f"sha256={digest!r},size_bytes=len(d),declared_mime='image/jpeg',"
        "model_version='stub-0',idempotency_key='k',submitted_at=datetime.now(UTC));"
        "b=asyncio.run(ImagePipeline().analyze_image(s,d));print(b.p_fake)"
    )
    path = Path(script.split("open(")[1].split(",'rb'")[0].strip("'"))
    path.write_bytes(data)
    here = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=here, timeout=180
    )
    assert out.returncode == 0, out.stderr[-400:]
    local = asyncio.run(ImagePipeline().analyze_image(spec(data), data)).p_fake
    assert out.stdout.strip() == str(local), "score changed between processes"


# --------------------------------------------------------------------------
# Image pipeline paths
# --------------------------------------------------------------------------


def test_image_bundle_validates_and_carries_the_job_id():
    data = face_image(2)
    job = uuid4()
    bundle = asyncio.run(ImagePipeline().analyze_image(spec(data, job_id=job), data))
    EvidenceBundle.model_validate(bundle.model_dump(mode="json"))
    assert bundle.job_id == job
    assert bundle.media.kind is MediaKind.IMAGE


def test_no_face_path_abstains():
    data = flat_image()
    bundle = asyncio.run(ImagePipeline().analyze_image(spec(data), data))
    assert bundle.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert bundle.p_fake is None
    assert bundle.quality.gate_passed is False
    EvidenceBundle.model_validate(bundle.model_dump(mode="json"))


def test_low_quality_path_abstains_before_the_model_runs():
    data = flat_image(size=40)
    bundle = asyncio.run(ImagePipeline().analyze_image(spec(data), data))
    assert bundle.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert bundle.timing.inference_ms == 0, "the model ran despite a failed gate"


def test_wrong_media_kind_is_rejected():
    data = face_image()
    try:
        asyncio.run(ImagePipeline().analyze_image(spec(data, MediaKind.VIDEO), data))
    except UnsupportedMedia:
        return
    raise AssertionError("an image pipeline accepted a video job")


def test_malformed_image_raises_unreadable():
    data = b"definitely not an image"
    try:
        asyncio.run(ImagePipeline().analyze_image(spec(data), data))
    except UnreadableMedia:
        return
    raise AssertionError("malformed bytes were analysed")


# --------------------------------------------------------------------------
# Video pipeline paths
# --------------------------------------------------------------------------


def test_progress_fires_once_per_batch_never_per_frame():
    data = clip()
    recorder = Recorder()
    engine = VideoPipeline(n_frames=48, batch_size=8)
    asyncio.run(engine.analyze_video(spec(data, MediaKind.VIDEO), data, recorder))
    assert len(recorder) == 6, f"expected 6 calls for 48 frames, got {len(recorder)}"
    fractions = [f for f, _ in recorder.calls]
    assert fractions == sorted(fractions)
    assert abs(fractions[-1] - 1.0) < 1e-9
    assert all(stage is JobStage.SCORING for _, stage in recorder.calls)


def test_progress_scales_with_the_batch_size():
    data = clip()
    recorder = Recorder()
    asyncio.run(
        VideoPipeline(n_frames=48, batch_size=16).analyze_video(
            spec(data, MediaKind.VIDEO), data, recorder
        )
    )
    assert len(recorder) == 3


def test_video_without_a_progress_callback_still_works():
    data = clip()
    bundle = asyncio.run(VideoPipeline().analyze_video(spec(data, MediaKind.VIDEO), data))
    EvidenceBundle.model_validate(bundle.model_dump(mode="json"))


def test_long_video_is_sampled_not_fully_decoded():
    data = clip(frames=400, size=96)
    engine = VideoPipeline(n_frames=48)
    bundle = asyncio.run(engine.analyze_video(spec(data, MediaKind.VIDEO), data))
    assert bundle.media.frame_count == 400
    assert bundle.video.frames_sampled <= 48, "the whole clip was decoded"


def test_short_video_abstains_on_too_few_frames():
    data = clip(frames=4, size=96)
    bundle = asyncio.run(
        VideoPipeline(min_usable_frames=8).analyze_video(spec(data, MediaKind.VIDEO), data)
    )
    assert bundle.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert bundle.video is not None, "a video bundle keeps its analysis block even when abstaining"
    EvidenceBundle.model_validate(bundle.model_dump(mode="json"))


def test_malformed_video_raises_unreadable():
    data = b"\x00\x01\x02 not a container"
    try:
        asyncio.run(VideoPipeline().analyze_video(spec(data, MediaKind.VIDEO), data))
    except UnreadableMedia:
        return
    raise AssertionError("malformed bytes were analysed as video")


def test_image_and_video_pipelines_agree_on_contract_shape():
    """Parity: both produce bundles the same consumer can read."""
    img_data, vid_data = face_image(4), clip()
    a = asyncio.run(ImagePipeline().analyze_image(spec(img_data), img_data))
    b = asyncio.run(
        VideoPipeline().analyze_video(spec(vid_data, MediaKind.VIDEO), vid_data)
    )
    image_engine, video_engine = ImagePipeline(), VideoPipeline()
    for bundle, engine in ((a, image_engine), (b, video_engine)):
        EvidenceBundle.model_validate(bundle.model_dump(mode="json"))
        assert {x.id for x in bundle.branches} == {BranchId.A, BranchId.B, BranchId.P}
        # The published identity is the *effective* version — weights,
        # calibration and gate composed — not the bare weights string. Asserting
        # the literal "stub-0" here encoded the defect that let a recalibration
        # reuse results computed under the previous temperature.
        assert bundle.model.version == engine.model_version
        assert is_composed(bundle.model.version), (
            "a bare version cannot express a calibration change and so cannot key one"
        )
        assert "stub" in bundle.model.version
    assert a.model.version == b.model.version, (
        "both pipelines serve the same weights behind the same gate and must "
        "agree on identity; disagreeing would split the result cache in two"
    )
    assert a.video is None and b.video is not None


# --------------------------------------------------------------------------
# Model lifecycle
# --------------------------------------------------------------------------


def test_load_is_idempotent():
    registry = ModelRegistry()
    registry.register(ModelSpec(name="m", version="v", backend="stub"))
    first = registry.load_model("m")
    assert registry.load_model("m") is first


def test_unload_semantics_and_tracking():
    registry = ModelRegistry()
    registry.register(ModelSpec(name="m", version="v", backend="stub"))
    assert registry.get_loaded_models() == {}
    registry.load_model("m")
    assert set(registry.get_loaded_models()) == {"m"}
    assert registry.is_loaded("m") is True
    assert registry.unload_model("m") is True
    assert registry.unload_model("m") is False, "a second unload should report nothing to do"
    assert registry.get_loaded_models() == {}


def test_unknown_model_raises_model_unavailable():
    try:
        ModelRegistry().load_model("nope")
    except ModelUnavailable as exc:
        assert "registered" in (exc.detail or "")
        return
    raise AssertionError("an unregistered model loaded")


def test_torch_backend_refuses_rather_than_falling_back():
    """A deployment quietly serving hash-derived scores under a real model
    version is worse than one that refuses to start.

    Accepts either legitimate refusal: `torch` importable (the backend is a
    declared stub, not implemented yet) or `torch` not installed (the `torch`
    extra is deliberately excluded from `dev` — pyproject.toml, so it does
    not have to be present here). Either way the registry must refuse and
    never fall back to the stub silently; the test does not care which of the
    two true reasons it refuses for.
    """
    registry = ModelRegistry()
    registry.register(
        ModelSpec(name="real", version="v1", backend="torch", weights_uri="file://w.pt")
    )
    try:
        registry.load_model("real")
    except ModelUnavailable as exc:
        message = str(exc)
        if "not installed" in message:
            return
        assert "not implemented" in message
        assert "stub" in (exc.detail or "").lower()
        return
    raise AssertionError("the torch backend silently produced a model")


def test_onnx_backend_refuses_rather_than_falling_back():
    registry = ModelRegistry()
    registry.register(
        ModelSpec(name="o", version="v1", backend="onnx", weights_uri="file://w.onnx")
    )
    try:
        registry.load_model("o")
    except ModelUnavailable:
        return
    raise AssertionError("the onnx backend silently produced a model")


def test_stub_backend_scores_without_torch():
    """Asserts the *delta*, not the absolute state.

    Another test in this file exercises the torch backend's lazy-import path,
    which legitimately leaves torch resident. Asserting absence outright would
    make this pass or fail on alphabetical test order — what matters is that
    stub scoring imports nothing new.
    """
    before = "torch" in sys.modules
    backend = StubBackend(ModelSpec(name="s", version="v"))
    out = backend.score(np.zeros((2, 8, 8, 3), np.float32))
    assert set(out) == {BranchId.A, BranchId.B, BranchId.P}
    assert all(v.shape == (2,) for v in out.values())
    assert ("torch" in sys.modules) == before, "stub scoring imported torch"


def test_stub_backend_rejects_a_wrong_shape():
    backend = StubBackend(ModelSpec(name="s", version="v"))
    try:
        backend.score(np.zeros((8, 8, 3), np.float32))
    except ValueError:
        return
    raise AssertionError("an unbatched array was accepted")


def test_stub_provenance_logit_is_never_negative():
    backend = StubBackend(ModelSpec(name="s", version="v"))
    rng = np.random.default_rng(0)
    out = backend.score(rng.random((16, 8, 8, 3)).astype(np.float32))
    assert bool(np.all(out[BranchId.P] >= 0.0))


def test_default_registry_exposes_the_stub_specs():
    assert {"stub-image", "stub-video"} <= set(default_registry().specs())


# --------------------------------------------------------------------------
# Guards: dependencies and torch
# --------------------------------------------------------------------------


def test_inference_imports_nothing_forbidden():
    violations = []
    for path in sorted(Path("inference").rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            )
            for name in names:
                if name.split(".")[0] in FORBIDDEN:
                    violations.append(f"{path}:{node.lineno} imports {name}")
    assert not violations, "forbidden imports:\n  " + "\n  ".join(violations)


def test_no_module_scope_torch_or_onnx_import():
    """Lazy loading is a property of *where* the import sits, not whether one
    exists. A module-scope import defeats it even if nothing calls the code."""
    offenders = []
    for path in sorted(Path("inference").rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in tree.body:  # module scope only
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            )
            for name in names:
                if name.split(".")[0] in ("torch", "onnxruntime"):
                    offenders.append(f"{path}:{node.lineno} -> {name}")
    assert not offenders, "module-scope heavy imports:\n  " + "\n  ".join(offenders)


def test_importing_inference_does_not_load_torch():
    """Measured in a clean interpreter: importing this package must not pay the
    15-second torch import that every consumer would otherwise inherit."""
    here = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0,'.'); import inference; "
            "print('torch' in sys.modules, 'onnxruntime' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=here,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip() == "False False", f"heavy imports leaked: {out.stdout!r}"


def test_pipelines_run_without_a_gpu():
    """Same delta-based check as the stub backend, for the same reason."""
    before = "torch" in sys.modules
    data = face_image(9)
    asyncio.run(ImagePipeline().analyze_image(spec(data), data))
    assert ("torch" in sys.modules) == before, "the stub path imported torch"


# --------------------------------------------------------------------------
# Conformance: prove the suite catches failures
# --------------------------------------------------------------------------


def test_suite_catches_nondeterministic_scoring():
    """A backend that varies between calls must fail the determinism check."""

    class Drifting(StubBackend):
        """Alternates between two decided extremes on successive calls.

        Deliberately far from the abstention band in both directions: a subtler
        drift can land one call inside the band, which abstains, discards the
        probability, and makes two different scores look identical — the double
        would then go undetected for a reason unrelated to determinism.
        """

        def __init__(self, spec_: ModelSpec) -> None:
            super().__init__(spec_)
            self._n = 0

        def score(self, batch: np.ndarray) -> dict[BranchId, np.ndarray]:
            self._n += 1
            magnitude = 8.0 if self._n % 2 else -8.0
            n = batch.shape[0]
            return {
                BranchId.A: np.full(n, magnitude),
                BranchId.B: np.full(n, magnitude),
                BranchId.P: np.zeros(n),
            }

    registry = ModelRegistry()
    registry.register(ModelSpec(name="drift", version="v", backend="stub"))
    engine = ImagePipeline(model_name="drift", registry=registry)
    loaded = registry.load_model("drift")
    loaded.backend = Drifting(registry.get_spec("drift"))

    data = GATE_PASSING_IMAGE
    job = uuid4()
    a = asyncio.run(engine.analyze_image(spec(data, job_id=job), data))
    b = asyncio.run(engine.analyze_image(spec(data, job_id=job), data))
    assert a.p_fake is not None, "fixture abstained; the backend was never exercised"
    assert a.p_fake != b.p_fake, "the drifting backend went undetected"


def test_suite_catches_per_frame_progress():
    """batch_size=1 is the mistake this check exists to catch."""
    data = clip()
    recorder = Recorder()
    asyncio.run(
        VideoPipeline(n_frames=48, batch_size=1).analyze_video(
            spec(data, MediaKind.VIDEO), data, recorder
        )
    )
    assert len(recorder) > 6, "a per-frame reporter looked like a batched one"


def test_suite_catches_out_of_bounds_probability():
    """Extreme logits must saturate, not escape [0, 1]."""

    class Exploding(StubBackend):
        def score(self, batch: np.ndarray) -> dict[BranchId, np.ndarray]:
            n = batch.shape[0]
            return {
                BranchId.A: np.full(n, 1e9),
                BranchId.B: np.full(n, 1e9),
                BranchId.P: np.zeros(n),
            }

    registry = ModelRegistry()
    registry.register(ModelSpec(name="boom", version="v", backend="stub"))
    engine = ImagePipeline(model_name="boom", registry=registry)
    loaded = registry.load_model("boom")
    loaded.backend = Exploding(registry.get_spec("boom"))

    data = GATE_PASSING_IMAGE
    bundle = asyncio.run(engine.analyze_image(spec(data), data))
    assert bundle.p_fake is not None, "fixture abstained; the backend was never exercised"
    assert 0.0 <= bundle.p_fake <= 1.0, f"probability escaped bounds: {bundle.p_fake}"
    EvidenceBundle.model_validate(bundle.model_dump(mode="json"))


def test_suite_catches_an_invalid_verdict_derivation():
    """Tamper with a valid bundle; the contract must reject it."""
    data = face_image(7)
    bundle = asyncio.run(ImagePipeline().analyze_image(spec(data), data))
    if bundle.p_fake is None:
        return  # abstained; nothing to contradict
    payload = bundle.model_dump(mode="json")
    payload["verdict"] = (
        "authentic" if bundle.verdict is Verdict.MANIPULATED else "manipulated"
    )
    try:
        EvidenceBundle.model_validate(payload)
    except Exception:
        return
    raise AssertionError("a flipped verdict validated against the contract")


def test_suite_catches_a_negative_provenance_logit():
    payload_source = ModelSpec(name="x", version="v")
    try:
        ModelSpec(
            name="x", version="v", branch_weights={BranchId.A: 1.0, BranchId.P: -1.0}
        )
    except ValueError:
        assert payload_source.branch_weights[BranchId.P] >= 0
        return
    raise AssertionError("a negative provenance weight was accepted")


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
