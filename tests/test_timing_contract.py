"""F6 — published timing means what its field names say.

TWO DEFECTS, ONE THEME
----------------------
**The Timing block was derived by subtraction.** The image path measured
``preprocess_ms`` as elapsed-so-far after decoding, then ``inference_ms`` as
elapsed-so-far minus that. Face detection ran between the two samples, so its
cost landed in neither: a measured 1268 ms of a 1431 ms gate-refused job — 89% —
appeared in no field, while ``total_ms`` reported the truth. The video path used
the same subtraction in the other direction, so ``inference_ms`` absorbed fusion
and bundle construction along with the model calls. Two pipelines, two
conventions, neither matching its own field names.

**FrameScore.t_ms was inferred from how many frames survived.** It was
``i * duration / len(survivors)`` — the retained frames re-spread evenly across
the whole clip — while ``frame_idx`` kept the true source index. On a healthy
clip the two already disagreed; when a frame failed to decode the curve's
timestamps drifted further, and every suspect-segment boundary pointed at a
moment the analysis never looked at.

Both are now measured rather than inferred. Timings are asserted as
*relationships* — non-negative, bounded by the total, responsive to real work —
rather than as millisecond values, which would make the suite a flake generator.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.common import Timing  # noqa: E402
from contracts.enums import MediaKind  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from inference import ImagePipeline, VideoPipeline  # noqa: E402
from inference.preprocess import extract_frames  # noqa: E402
from tests.detector_fixtures import haar_selection  # noqa: E402

_HAAR = haar_selection()  # the timing contract, not the detector, is under test
from inference.timing import (  # noqa: E402
    INFERENCE,
    PREPROCESS,
    PhaseTimer,
    queued_ms_since,
)


def _clip(frames: int = 60, size: int = 192, fps: int = 25, seed: int = 11) -> bytes:
    path = Path(tempfile.mkdtemp()) / "timing.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    rng = np.random.default_rng(seed)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _image(size: int = 600) -> bytes:
    rng = np.random.default_rng(3)
    return cv2.imencode(".jpg", rng.integers(0, 255, (size, size, 3), dtype=np.uint8))[
        1
    ].tobytes()


def _spec(data: bytes, kind: MediaKind = MediaKind.IMAGE, *, waited_s: float = 0.0):
    return JobSpec(
        job_id=uuid4(),
        media_kind=kind,
        storage_key="u",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="image/jpeg" if kind is MediaKind.IMAGE else "video/mp4",
        model_version="v",
        idempotency_key="k",
        submitted_at=datetime.now(UTC) - timedelta(seconds=waited_s),
    )


IMAGE = _image()
CLIP = _clip()


# --------------------------------------------------------------------------
# The contract, as invariants
# --------------------------------------------------------------------------


def _bundles():
    yield "image", asyncio.run(ImagePipeline(face_selection=_HAAR).analyze_image(_spec(IMAGE), IMAGE))
    yield "video", asyncio.run(
        VideoPipeline().analyze_video(_spec(CLIP, MediaKind.VIDEO), CLIP)
    )


def test_every_timing_field_is_non_negative():
    for name, bundle in _bundles():
        t = bundle.timing
        for field in ("queued_ms", "preprocess_ms", "inference_ms", "explain_ms", "total_ms"):
            assert getattr(t, field) >= 0, f"{name}.{field} is negative"


def test_phases_never_exceed_the_total():
    """No double counting. The parts are parts of the whole."""
    for name, bundle in _bundles():
        t = bundle.timing
        parts = t.preprocess_ms + t.inference_ms + t.explain_ms
        assert parts <= t.total_ms, (
            f"{name}: phases sum to {parts}ms inside a {t.total_ms}ms job — "
            "an interval has been counted twice"
        )


def test_the_work_is_actually_accounted_for():
    """The other half: parts must not vanish either.

    Loose on purpose — fusion and bundle construction are real and unmeasured.
    The threshold is set well below what a correct implementation achieves
    (measured above 99%) and well above what the subtraction-based one did on
    the gate-refused path (11%).
    """
    for name, bundle in _bundles():
        t = bundle.timing
        parts = t.preprocess_ms + t.inference_ms + t.explain_ms
        if t.total_ms < 5:
            continue  # too short to apportion meaningfully
        share = parts / t.total_ms
        assert share > 0.5, (
            f"{name}: only {share:.0%} of a {t.total_ms}ms job is accounted for "
            "by its phases; work is being measured into no field at all"
        )


def test_a_gate_refusal_still_reports_the_work_it_did():
    """The path the old accounting lost almost entirely.

    Decoding and face detection both run before the gate declines, and both are
    preprocessing. Reporting zero for them told an operator the job was free.
    """
    tiny = cv2.imencode(".jpg", np.zeros((80, 80, 3), np.uint8))[1].tobytes()
    bundle = asyncio.run(ImagePipeline(face_selection=_HAAR).analyze_image(_spec(tiny), tiny))
    assert bundle.verdict.value == "insufficient_evidence"
    assert bundle.timing.preprocess_ms > 0, (
        "a gate-refused job reported no preprocessing, but it decoded and "
        "ran detection to reach the gate"
    )
    assert bundle.timing.inference_ms == 0, "no model ran; inference must be zero"


def test_inference_is_only_the_model_call():
    """Video's inference_ms used to absorb everything that was not decode."""
    bundle = asyncio.run(
        VideoPipeline().analyze_video(_spec(CLIP, MediaKind.VIDEO), CLIP)
    )
    t = bundle.timing
    assert t.preprocess_ms > 0, "frame extraction must be reported"
    assert t.inference_ms > 0, "scoring must be reported"
    assert t.preprocess_ms + t.inference_ms <= t.total_ms


def test_timing_tracks_real_work():
    """More work must report more time. Ordering, not absolute values."""
    small = asyncio.run(
        VideoPipeline(n_frames=8).analyze_video(_spec(CLIP, MediaKind.VIDEO), CLIP)
    )
    large = asyncio.run(
        VideoPipeline(n_frames=48).analyze_video(_spec(CLIP, MediaKind.VIDEO), CLIP)
    )
    assert large.timing.inference_ms >= small.timing.inference_ms, (
        "scoring six times as many frames did not report more inference time"
    )
    assert large.video.frames_scored > small.video.frames_scored


def test_queued_time_is_sampled_when_analysis_starts():
    """Sampling it at the end folds the whole analysis into the queue wait."""
    waited = 1.2
    bundle = asyncio.run(
        ImagePipeline(face_selection=_HAAR).analyze_image(_spec(IMAGE, waited_s=waited), IMAGE)
    )
    reported = bundle.timing.queued_ms
    assert 1100 <= reported <= 1400, (
        f"queued_ms was {reported}ms for a {waited}s wait; sampling it after the "
        "analysis would add total_ms to it"
    )
    # The tell: sampled at the end, the reported wait would exceed the true one
    # by roughly the whole analysis.
    assert reported < int(waited * 1000) + bundle.timing.total_ms, (
        "queued_ms looks like it includes the analysis duration"
    )


def test_queue_wait_is_outside_the_total():
    """``total_ms`` covers analysis. A long wait must not inflate it."""
    quick = asyncio.run(ImagePipeline(face_selection=_HAAR).analyze_image(_spec(IMAGE, waited_s=0.0), IMAGE))
    stale = asyncio.run(ImagePipeline(face_selection=_HAAR).analyze_image(_spec(IMAGE, waited_s=5.0), IMAGE))
    assert stale.timing.queued_ms > quick.timing.queued_ms + 4000
    assert stale.timing.total_ms < 4000, (
        "a five-second queue wait leaked into total_ms"
    )


# --------------------------------------------------------------------------
# PhaseTimer itself
# --------------------------------------------------------------------------


def test_total_is_monotonic():
    timer = PhaseTimer()
    readings = [timer.total_ms for _ in range(5)]
    time.sleep(0.01)
    readings.append(timer.total_ms)
    assert readings == sorted(readings), "total_ms went backwards"
    assert readings[-1] >= readings[0]


def test_repeated_phases_accumulate_rather_than_overwrite():
    """The video path scores in batches; each batch adds to the same phase."""
    timer = PhaseTimer()
    for _ in range(3):
        with timer.phase(INFERENCE):
            time.sleep(0.01)
    assert timer.seconds(INFERENCE) >= 0.028, (
        f"three 10ms phases accumulated to {timer.seconds(INFERENCE):.4f}s; "
        "a later phase overwrote an earlier one"
    )


def test_a_raising_phase_still_records_its_time():
    timer = PhaseTimer()
    try:
        with timer.phase(PREPROCESS):
            time.sleep(0.01)
            raise RuntimeError("decode failed")
    except RuntimeError:
        pass
    assert timer.seconds(PREPROCESS) >= 0.008, "time spent before a failure was lost"


def test_phases_floor_and_total_ceils_so_the_sum_always_fits():
    """The rounding rule that makes the contract assertable without a tolerance."""
    for _ in range(40):
        timer = PhaseTimer()
        with timer.phase(PREPROCESS):
            time.sleep(0.001)
        with timer.phase(INFERENCE):
            time.sleep(0.001)
        t = timer.to_timing()
        assert t.preprocess_ms + t.inference_ms + t.explain_ms <= t.total_ms


def test_the_contract_rejects_double_counted_phases():
    """A legacy-style timer that attributes one interval twice must not validate."""
    try:
        Timing(queued_ms=0, preprocess_ms=80, inference_ms=80, explain_ms=0, total_ms=100)
    except ValueError as exc:
        assert "counted twice" in str(exc)
        return
    raise AssertionError("a Timing whose phases exceed its total validated")


def test_queued_ms_since_clamps_clock_skew():
    future = datetime.now(UTC) + timedelta(seconds=30)
    assert queued_ms_since(future) == 0


# --------------------------------------------------------------------------
# FrameScore.t_ms
# --------------------------------------------------------------------------


def test_frame_timestamps_match_the_container():
    """Each retained frame reports its own presentation time."""
    meta, frames, indices, stamps = extract_frames(
        CLIP, mime="video/mp4", n_frames=48, target_size=384
    )
    assert len(stamps) == len(indices) == frames.shape[0]
    for index, stamp in zip(indices, stamps, strict=True):
        expected = round(index * 1000.0 / meta.fps)
        assert abs(stamp - expected) <= 1, (
            f"frame {index} reported t_ms={stamp}, container says {expected}"
        )


def test_the_curve_timestamps_agree_with_its_frame_indices():
    bundle = asyncio.run(
        VideoPipeline().analyze_video(_spec(CLIP, MediaKind.VIDEO), CLIP)
    )
    fps = bundle.media.fps
    for frame in bundle.video.frame_scores:
        expected = round(frame.frame_idx * 1000.0 / fps)
        assert abs(frame.t_ms - expected) <= 1, (
            f"frame_idx {frame.frame_idx} carries t_ms {frame.t_ms}, but that "
            f"frame sits at {expected}ms"
        )


def test_timestamps_are_non_decreasing():
    bundle = asyncio.run(
        VideoPipeline().analyze_video(_spec(CLIP, MediaKind.VIDEO), CLIP)
    )
    stamps = [f.t_ms for f in bundle.video.frame_scores]
    assert stamps == sorted(stamps), "the frame curve runs backwards in time"


def test_timestamps_stay_inside_the_clip():
    bundle = asyncio.run(
        VideoPipeline().analyze_video(_spec(CLIP, MediaKind.VIDEO), CLIP)
    )
    duration_ms = bundle.media.duration_s * 1000.0
    for frame in bundle.video.frame_scores:
        assert 0 <= frame.t_ms <= duration_ms + 1, (
            f"t_ms {frame.t_ms} falls outside a {duration_ms:.0f}ms clip"
        )


def test_the_legacy_formula_disagreed_with_its_own_frame_indices():
    """The defect, reproduced. Timestamps were a function of the survivor count.

    Not a rounding difference: the legacy value for a frame is derived from its
    *position in the surviving list*, so it moves whenever a different number of
    frames decodes, even though the frame itself has not moved.
    """
    meta, frames, indices, stamps = extract_frames(
        CLIP, mime="video/mp4", n_frames=48, target_size=384
    )
    step_ms = int(meta.duration_s * 1000 / max(1, len(stamps)))
    legacy = [i * step_ms for i in range(len(stamps))]

    truth = [round(idx * 1000.0 / meta.fps) for idx in indices]
    assert stamps == [int(t) for t in truth], "the corrected timestamps are not the truth"
    assert legacy != truth, (
        "the legacy formula no longer disagrees with the container; this test "
        "has stopped demonstrating the defect"
    )
    worst = max(abs(a - b) for a, b in zip(legacy, truth, strict=True))
    assert worst > 5, f"worst legacy drift was only {worst}ms on a healthy clip"

    # And the drift grows when fewer frames survive, because the survivors are
    # re-spread across the whole duration regardless of where they were.
    fewer = len(stamps) // 2
    legacy_step = int(meta.duration_s * 1000 / fewer)
    drift = abs((fewer - 1) * legacy_step - truth[fewer - 1])
    assert drift > worst, (
        "the legacy formula's error is meant to grow as frames drop out"
    )


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
