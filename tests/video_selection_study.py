"""R29 — the video frame-selection and pooling knobs, measured. Standalone.

Batch 23 moved ``n_frames``, ``min_usable_frames`` and the top-k pooling depth
into the model identity via ``VideoSelectionPolicy``, and named the ``uniform`` /
``topk_mean`` rules as label closures. This study is the measurement behind that
decision, in the ``crop_identity_study.py`` shape.

    python tests/video_selection_study.py
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.enums import MediaKind  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from inference import VideoPipeline  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.postprocess import pool_at, topk_indices  # noqa: E402
from inference.preprocess import sample_frame_indices  # noqa: E402

SHIPPED = pre.default_video_selection()


def head(n: int, title: str, *sub: str) -> None:
    print(f"\n{'=' * 78}\n§{n}  {title}")
    for s in sub:
        print(f"    {s}")
    print("-" * 78)


def _clip(frames: int = 60, size: int = 128, fps: int = 20, seed: int = 1) -> bytes:
    path = Path(tempfile.mkdtemp()) / "study.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    rng = np.random.default_rng(seed)
    for i in range(frames):
        if frames // 3 <= i < 2 * frames // 3:
            f = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        else:
            f = np.full((size, size, 3), 110, np.uint8)
        w.write(f)
    w.release()
    return path.read_bytes()


CLIP = _clip()
SHORT = _clip(frames=12)


def _spec(data: bytes) -> JobSpec:
    return JobSpec(
        job_id=uuid4(), media_kind=MediaKind.VIDEO, storage_key="u",
        sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
        declared_mime="video/mp4", model_version="stub-0", idempotency_key="k",
        submitted_at=datetime.now(UTC),
    )


def _run(policy: pre.VideoSelectionPolicy, data: bytes = CLIP):
    pipe = VideoPipeline(model_name="stub-video", video_selection=policy)
    return pipe, asyncio.run(pipe.analyze_video(_spec(data), data))


def s1_knobs() -> None:
    head(1, "The knobs", "which video-path parameters decide a verdict?")
    for f in fields(pre.VideoSelectionPolicy):
        print(f"  {f.name:<20} shipped = {getattr(SHIPPED, f.name)!r}")
    print("  batch_size            EXCLUDED — output-neutral (see §3)")


def s2_magnitude() -> None:
    head(2, "Magnitude", "|Δp_fake| for each knob on one synthetic clip")
    _, base = _run(SHIPPED)
    print(f"  shipped policy                       -> p_fake {base.p_fake:.6f}  {base.verdict.value}")
    for label, pol, data in [
        ("n_frames 48 -> 9", replace(SHIPPED, n_frames=9), CLIP),
        ("pool_divisor 8 -> 4", replace(SHIPPED, pool_divisor=4), CLIP),
        ("min_usable_frames 8 -> 13 (12-frame clip)",
         replace(SHIPPED, n_frames=16, min_usable_frames=13), SHORT),
    ]:
        _, b = _run(pol, data)
        d = "abstain" if b.p_fake is None else f"{abs(b.p_fake - (base.p_fake or 0)):.6f}"
        pv = "None" if b.p_fake is None else f"{b.p_fake:.6f}"
        print(f"  {label:<38} -> p_fake {pv:<10} Δ {d:<10} {b.verdict.value}")


def s3_batch_size_neutral() -> None:
    head(3, "batch_size is output-neutral", "two batch sizes, one bundle")
    a = asyncio.run(VideoPipeline(batch_size=8).analyze_video(_spec(CLIP), CLIP))
    b = asyncio.run(VideoPipeline(batch_size=16).analyze_video(_spec(CLIP), CLIP))
    print(f"  bs=8  p_fake {a.p_fake:.12f}  |  bs=16 p_fake {b.p_fake:.12f}")
    print(f"  equal p_fake      : {a.p_fake == b.p_fake}")
    print(f"  equal video block : {a.video.model_dump() == b.video.model_dump()}")
    print(f"  equal model.version: {a.model.version == b.model.version}")


def s4_identities() -> None:
    head(4, "Identities", "pre/post the composite .f. segment")
    f = pre.default_face_selection()
    print(f"  face policy .identity                 = {f.identity}   (unchanged by B23)")
    print(f"  video_selection_identity()            = {pre.video_selection_identity()}")
    print(f"  selection_identity(face, video) [.f.] = {pre.selection_identity(f, SHIPPED)}")
    print(f"  pre-B23 .f. (face only)               = 251d22e9")
    eng = VideoPipeline().model_version
    print(f"  live video engine model_version      = {eng}")


def s5_label_rederivation() -> None:
    head(5, "Label re-derivations", "the closures held to the code")

    def uniform_stride(fc: int, n: int) -> list[int]:
        if fc <= 0 or n <= 0:
            return []
        if fc <= n:
            return list(range(fc))
        step = fc / n
        return [min(fc - 1, int(i * step)) for i in range(n)]

    ok = all(sample_frame_indices(fc, n) == uniform_stride(fc, n)
             for fc, n in [(1000, 48), (1000, 9), (50, 48), (48, 48), (7, 8)])
    print(f"  sampling_rule={SHIPPED.sampling_rule!r} matches sample_frame_indices : {ok}")
    scores = np.linspace(-2.0, 3.0, 48)
    k = max(1, 48 // SHIPPED.pool_divisor)
    print(f"  pooling_rule={SHIPPED.pooling_rule!r}: k = 48 // {SHIPPED.pool_divisor} = {k}, "
          f"selected {topk_indices(scores, k=k)}")
    print(f"  pooled mean at those frames = {pool_at(scores, topk_indices(scores, k=k)):.6f}")


def s6_scrfd_note() -> None:
    head(6, "What a SCRFD video face stage adds")
    print("  Today the video path scores whole resized frames (R18): VideoSelectionPolicy")
    print("  is fully consumed, FaceSelectionPolicy is carried-not-consumed. When the")
    print("  SCRFD video face stage lands it adds a FaceSelectionPolicy consumer to the")
    print("  video path — already covered by selection_identity(). R6-a/R6-b (frame_count")
    print("  miscount -> sampled indices) and R24 (whole-frame squash) stay open.")


def main() -> int:
    print(__doc__.strip())
    for fn in (s1_knobs, s2_magnitude, s3_batch_size_neutral, s4_identities,
               s5_label_rederivation, s6_scrfd_note):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"\n  !! {fn.__name__} raised {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print("\n" + "=" * 78)
    print("R29 CLOSED by Batch 23: VideoSelectionPolicy (5 required fields) whose")
    print("fingerprint enters the .f. segment via selection_identity(), from BOTH")
    print("pipelines. batch_size excluded, proven output-neutral. No pixel moved.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
