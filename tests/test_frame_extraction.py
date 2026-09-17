"""F12 and F13 — bounded memory and one decode pass.

Both defects live in ``extract_frames`` and both are measured here against
``legacy_extract``, a verbatim copy of the pre-Batch-1 implementation kept in
this file so the comparison stays honest and stays runnable.

    F13  the old loop called ``capture.set(CAP_PROP_POS_FRAMES, i)`` before each
         wanted frame. On long-GOP video every seek decodes forward from the
         preceding keyframe, so sampling 48 frames re-decoded much of the clip
         dozens of times — 67.1 s for a 60-frame 2160p clip, against 14.1 s for
         one sequential pass.

    F12  it then accumulated full-resolution frames in a list and called
         ``np.stack``, holding the list and its copy at once, and the caller
         resized afterwards. Peak RSS growth 2522 MB for a single 4K job whose
         useful output is 21 MB.

The correctness bar is the strict one the brief set: **identical bundles**. A
faster, smaller implementation that changed a single score would be a
regression, so the scoring path is compared value by value rather than
approximately.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.enums import BranchId, MediaKind  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from inference import VideoPipeline  # noqa: E402
from inference import preprocess as pre  # noqa: E402

TARGET = 384


# --------------------------------------------------------------------------
# The pre-Batch-1 implementation, kept as a double
# --------------------------------------------------------------------------


def legacy_extract(
    data: bytes,
    *,
    mime: str = "video/mp4",
    n_frames: int = 48,
    caps: pre.MediaCaps = pre.DEFAULT_CAPS,
):
    """Seek per frame, accumulate a list, stack it. Verbatim."""
    with pre.temp_media_file(data, suffix=pre.suffix_for(mime)) as path:
        meta = pre.probe_video(path, caps=caps)
        wanted = pre.sample_frame_indices(meta.frame_count, n_frames)
        capture = cv2.VideoCapture(str(path))
        collected: list[np.ndarray] = []
        kept: list[int] = []
        try:
            for index in wanted:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                collected.append(pre.to_rgb(frame))
                kept.append(index)
        finally:
            capture.release()
    if not collected:
        raise RuntimeError("no frames decoded")
    return meta, np.stack(collected), kept


def legacy_prepare(frames: np.ndarray, size: int = TARGET) -> np.ndarray:
    """What ``VideoPipeline._prepare`` did on top of the extraction."""
    return np.stack([pre.resize(frames[i], size) for i in range(frames.shape[0])]).astype(
        np.float32
    )


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------


_REAL_CAPTURE = cv2.VideoCapture


class _CountingCapture:
    """Delegating proxy. Subclassing the cv2 binding segfaults on this build."""

    counts: dict[str, int] = {}

    def __init__(self, *args, **kwargs):
        self._cap = _REAL_CAPTURE(*args, **kwargs)

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            _CountingCapture.counts["seek"] = _CountingCapture.counts.get("seek", 0) + 1
        return self._cap.set(prop, value)

    def read(self, *a):
        _CountingCapture.counts["read"] = _CountingCapture.counts.get("read", 0) + 1
        return self._cap.read(*a)

    def grab(self):
        _CountingCapture.counts["grab"] = _CountingCapture.counts.get("grab", 0) + 1
        return self._cap.grab()

    def retrieve(self, *a):
        _CountingCapture.counts["retrieve"] = (
            _CountingCapture.counts.get("retrieve", 0) + 1
        )
        return self._cap.retrieve(*a)

    def __getattr__(self, name):
        return getattr(self._cap, name)


class counting_decodes:
    """Context manager yielding a dict of cv2 call counts."""

    def __enter__(self) -> dict[str, int]:
        _CountingCapture.counts = {}
        cv2.VideoCapture = _CountingCapture
        return _CountingCapture.counts

    def __exit__(self, *a):
        cv2.VideoCapture = _REAL_CAPTURE


def _clip(frames: int = 60, size: int = 240, fps: int = 25, seed: int = 13) -> bytes:
    path = Path(tempfile.mkdtemp()) / "extract.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    for i in range(frames):
        writer.write(np.roll(base, i * 5, axis=1))
    writer.release()
    return path.read_bytes()


CLIP = _clip()


def _spec(data: bytes) -> JobSpec:
    return JobSpec(
        job_id=UUID(int=99),
        media_kind=MediaKind.VIDEO,
        storage_key="u",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="video/mp4",
        model_version="v",
        idempotency_key="k",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


# --------------------------------------------------------------------------
# F13 — decode operations
# --------------------------------------------------------------------------


def test_extraction_performs_no_seeks():
    with counting_decodes() as counts:
        pre.extract_frames(CLIP, n_frames=48, target_size=TARGET)
    assert counts.get("seek", 0) == 0, (
        f"{counts['seek']} seeks remain; extraction is meant to be one pass"
    )
    assert counts.get("grab", 0) > 0, "nothing was decoded"


def test_the_legacy_path_seeks_once_per_wanted_frame():
    """The double must still reproduce the defect, or the comparison is empty."""
    with counting_decodes() as counts:
        legacy_extract(CLIP, n_frames=48)
    assert counts.get("seek", 0) == 48, (
        f"expected 48 seeks from the legacy implementation, saw {counts.get('seek', 0)}"
    )


def test_decode_work_drops_measurably():
    with counting_decodes() as legacy_counts:
        legacy_extract(CLIP, n_frames=48)
    legacy_ops = dict(legacy_counts)
    with counting_decodes() as new_counts:
        pre.extract_frames(CLIP, n_frames=48, target_size=TARGET)
    new_ops = dict(new_counts)

    assert legacy_ops.get("seek", 0) > 0 and new_ops.get("seek", 0) == 0
    # Full colour-converting reads happen only for frames we keep.
    assert new_ops.get("retrieve", 0) <= 48
    assert new_ops.get("retrieve", 0) == 48, (
        "one retrieve per retained frame is the whole point of grab/retrieve"
    )


def test_only_wanted_frames_pay_for_colour_conversion():
    """``grab`` advances the stream; ``retrieve`` is the expensive half."""
    with counting_decodes() as counts:
        _, _, indices, _ = pre.extract_frames(CLIP, n_frames=12, target_size=TARGET)
    assert counts["retrieve"] == len(indices)
    assert counts["grab"] > counts["retrieve"], (
        "frames between samples must be grabbed but not retrieved"
    )


def test_sequential_extraction_is_faster_than_seeking():
    started = time.perf_counter()
    legacy_extract(CLIP, n_frames=48)
    legacy_s = time.perf_counter() - started

    started = time.perf_counter()
    pre.extract_frames(CLIP, n_frames=48, target_size=TARGET)
    new_s = time.perf_counter() - started

    assert new_s < legacy_s, (
        f"one pass took {new_s:.3f}s against {legacy_s:.3f}s of seeking"
    )


# --------------------------------------------------------------------------
# F13 — semantics preserved
# --------------------------------------------------------------------------


def test_selected_frame_indices_are_unchanged():
    for n in (8, 16, 48):
        _, _, legacy_indices = legacy_extract(CLIP, n_frames=n)
        _, _, indices, _ = pre.extract_frames(CLIP, n_frames=n, target_size=TARGET)
        assert indices == legacy_indices, f"frame selection changed at n_frames={n}"


def test_extraction_is_deterministic():
    a = pre.extract_frames(CLIP, n_frames=48, target_size=TARGET)
    b = pre.extract_frames(CLIP, n_frames=48, target_size=TARGET)
    assert a.indices == b.indices
    assert a.timestamps_ms == b.timestamps_ms
    assert np.array_equal(a.frames, b.frames)


def test_full_resolution_extraction_still_works():
    """``target_size`` is optional; without it the old shape is preserved."""
    meta, frames, indices, stamps = pre.extract_frames(CLIP, n_frames=16)
    assert frames.shape[1:] == (meta.height, meta.width, 3)
    assert len(indices) == len(stamps) == frames.shape[0]


# --------------------------------------------------------------------------
# F12 — memory
# --------------------------------------------------------------------------


def test_no_intermediate_stack_is_built():
    """The structural check, independent of any RSS measurement.

    The old peak came from holding a list of full-resolution frames *and* the
    array ``np.stack`` copied it into. Writing into one pre-allocated buffer
    means ``np.stack`` is never reached — so forbidding it proves the allocation
    shape rather than sampling its consequences.
    """
    real_stack = np.stack

    def forbidden(*a, **k):
        raise AssertionError("extract_frames called np.stack; it must pre-allocate")

    np.stack = forbidden
    try:
        _, frames, _, _ = pre.extract_frames(CLIP, n_frames=48, target_size=TARGET)
    finally:
        np.stack = real_stack
    assert frames.shape[0] > 0

    # And the legacy implementation would have tripped it.
    np.stack = forbidden
    try:
        legacy_extract(CLIP, n_frames=48)
    except AssertionError:
        return
    finally:
        np.stack = real_stack
    raise AssertionError("the legacy double no longer stacks; the check is vacuous")


def test_the_returned_payload_is_bounded_by_the_target_size():
    """Deterministic memory bound, no sampling required."""
    for n in (8, 48):
        _, frames, _, _ = pre.extract_frames(CLIP, n_frames=n, target_size=TARGET)
        assert frames.shape[1:] == (TARGET, TARGET, 3)
        assert frames.nbytes <= n * TARGET * TARGET * 3
        assert frames.dtype == np.uint8, (
            "frames must stay uint8; widening to float32 here quadruples the "
            "resident clip for no benefit"
        )


def test_peak_memory_growth_is_bounded_on_a_large_clip():
    """The end-to-end measurement, with a wide margin so it does not flake.

    Legacy peak on a 2160p clip was 2522 MB. The bound here is deliberately far
    above what the streaming path needs (measured 379 MB, most of it cv2's own
    4K decode buffers) and far below what accumulating full-resolution frames
    costs, so it distinguishes the two without depending on allocator detail.
    """
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is present on this box
        return

    proc = psutil.Process()
    big = _clip(frames=40, size=1440, seed=17)

    peak = {"v": 0.0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak["v"] = max(peak["v"], proc.memory_info().rss / 1e6)
            time.sleep(0.002)

    gc.collect()
    base = proc.memory_info().rss / 1e6
    peak["v"] = base
    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    try:
        _, frames, _, _ = pre.extract_frames(big, n_frames=48, target_size=TARGET)
    finally:
        stop.set()
        watcher.join()

    growth = peak["v"] - base
    full_res_cost = 48 * 1440 * 1440 * 3 / 1e6
    assert growth < full_res_cost / 2, (
        f"peak grew {growth:.0f} MB; holding every frame at source resolution "
        f"would cost {full_res_cost:.0f} MB, so this looks like the old shape"
    )
    assert frames.nbytes / 1e6 < 25


# --------------------------------------------------------------------------
# Identical output
# --------------------------------------------------------------------------


def _legacy_bundle_scores(data: bytes):
    """Score a clip through the legacy extraction, using today's fusion.

    Isolates the extraction change: everything downstream is the shipped code,
    so any difference in the numbers is attributable to decoding alone.
    """
    from inference import postprocess as post
    from inference.models import default_registry

    _, frames, kept = legacy_extract(data, n_frames=48)
    prepared = legacy_prepare(frames)
    loaded = default_registry().load_model("stub-video")
    per_branch: dict[BranchId, list[float]] = {}
    per_frame: list[float] = []
    for batch in pre.batch_clips(prepared, 8):
        logits = loaded.backend.score(batch)
        for branch, values in logits.items():
            per_branch.setdefault(branch, []).extend(float(v) for v in values)
        for j in range(batch.shape[0]):
            outcomes = {
                b: post.BranchOutcome(logit=float(v[j])) for b, v in logits.items()
            }
            fused = post.fuse(outcomes, loaded.spec.branch_weights)
            per_frame.append(0.0 if fused is None else fused)
    return kept, per_frame, per_branch


def test_scores_are_identical_to_the_legacy_extraction():
    """Bit-for-bit, not approximately."""
    from inference.calibration import sigmoid

    kept, legacy_frame_logits, legacy_branches = _legacy_bundle_scores(CLIP)
    bundle = asyncio.run(VideoPipeline().analyze_video(_spec(CLIP), CLIP))

    assert [f.frame_idx for f in bundle.video.frame_scores] == kept
    legacy_ps = [float(min(1.0, max(0.0, sigmoid(x)))) for x in legacy_frame_logits]
    assert [f.p for f in bundle.video.frame_scores] == legacy_ps, (
        "per-frame probabilities changed"
    )

    from inference.postprocess import pool_at, topk_indices

    selected = topk_indices(np.asarray(legacy_frame_logits, dtype=np.float64))
    for branch in (BranchId.A, BranchId.B, BranchId.P):
        expected = pool_at(legacy_branches[branch], selected)
        assert bundle.branch(branch).logit == expected, (
            f"branch {branch.value} logit changed: {bundle.branch(branch).logit} "
            f"vs {expected}"
        )


def test_the_bundle_is_unchanged_run_to_run():
    a = asyncio.run(VideoPipeline().analyze_video(_spec(CLIP), CLIP))
    b = asyncio.run(VideoPipeline().analyze_video(_spec(CLIP), CLIP))
    assert a.verdict is b.verdict
    assert a.p_fake == b.p_fake
    assert [x.logit for x in a.branches] == [x.logit for x in b.branches]
    assert [(f.frame_idx, f.t_ms, f.p) for f in a.video.frame_scores] == [
        (f.frame_idx, f.t_ms, f.p) for f in b.video.frame_scores
    ]


def test_progress_callback_count_is_unchanged():
    """Batching is a contract with the worker's lease; it must survive this."""

    class Recorder:
        def __init__(self):
            self.calls = []

        async def __call__(self, fraction, *, stage=None):
            self.calls.append((fraction, stage))

    for batch_size, expected in ((8, 6), (16, 3), (48, 1)):
        rec = Recorder()
        asyncio.run(
            VideoPipeline(batch_size=batch_size).analyze_video(
                _spec(CLIP), CLIP, rec
            )
        )
        assert len(rec.calls) == expected, (
            f"batch_size={batch_size} fired {len(rec.calls)} progress calls, "
            f"expected {expected}"
        )
        assert rec.calls[-1][0] == 1.0, "progress must finish at 1.0"
        fractions = [c[0] for c in rec.calls]
        assert fractions == sorted(fractions), "progress went backwards"


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
