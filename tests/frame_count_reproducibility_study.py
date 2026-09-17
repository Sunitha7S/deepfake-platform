"""R6 residual — ``frame_count`` reproducibility. Six experiments, measured. Standalone.

    python tests/frame_count_reproducibility_study.py

Produces every number in ``docs/FRAME_COUNT_REPRODUCIBILITY.md``. Same pattern as
``decode_study.py``: a claim nobody can re-run is not evidence.

THE QUESTION
------------
``MediaInfo.frame_count`` is a decoder-reported integer that ``sample_frame_indices``
uses to choose which frames are scored. Batch 8 measured the amplification (E2:
a one-frame error moves 60% of the sampled indices) but did not reproduce the
trigger, and left the residual as *"Unresolved — no validation of frame_count
exists"* (``DECODE_DETERMINISM.md`` §3). This study asks the sharper questions:

* S1  What exactly is ``frame_count``, and what is derived from it?
* S2  The amplifier, re-measured — and is *this* measurement portable?
* S3  Can one build be made to miscount? (attempted with real containers)
* S4  Does ``decode_identity`` — the Batch 9 cache-key segment and the Batch 11
      benchmark field — actually cover the library that produces ``frame_count``?
* S5  Fail-before: two demuxers that disagree by one frame, one identity string.
* S6  Which artifacts consume ``frame_count``?

WHAT MAKES S4/S5 MEASURABLE WITHOUT A SECOND MACHINE
---------------------------------------------------
``CAP_PROP_FRAME_COUNT`` is read by libavformat (the demuxer), not libavcodec
(the frame decoder). ``getBuildInformation`` lists them as separate libraries
with separate versions. ``decode_env`` parses only the avcodec line. So the
"does the identity cover it" question is answerable by inspection, and the
"what happens if it disagrees" question is answerable with a demuxer double —
a monkeypatch of ``probe_video`` that shaves one frame, standing in for a
libavformat build that estimates ``nb_frames`` differently.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.enums import MediaKind  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.decode_env import _build_facts, decode_environment, decode_identity  # noqa: E402
from inference.models import ModelRegistry, ModelSpec  # noqa: E402
from inference.video_pipeline import VideoPipeline  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dfp-framecount-study-"))
DEFAULT_N = 48


def head(n: int, title: str, hypothesis: str = "", failure: str = "") -> None:
    print(f"\n{'=' * 74}\nS{n}. {title}\n{'-' * 74}")
    if hypothesis:
        print(f"  hypothesis      : {hypothesis}")
    if failure:
        print(f"  expected effect : {failure}")
    print()


def clip(path: Path, *, frames=60, size=128, fps=25, fourcc="mp4v", seed=0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (size, size))
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    for i in range(frames):
        writer.write(np.roll(base, i * 7, axis=1))
    writer.release()
    return path


def _spec(data: bytes) -> JobSpec:
    return JobSpec(
        job_id=UUID(int=6),
        media_kind=MediaKind.VIDEO,
        storage_key="uploads/x",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="video/mp4",
        model_version="stub-0",
        idempotency_key="k",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


# --------------------------------------------------------------------------
# S1 — what frame_count is
# --------------------------------------------------------------------------


def s1_what_it_is() -> None:
    head(1, "What frame_count is, and what is derived from it")
    path = clip(TMP / "s1.mp4", frames=60)
    meta = pre.probe_video(path)
    cap = cv2.VideoCapture(str(path))
    raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    print(f"  VideoMeta.frame_count            = {meta.frame_count}")
    print(f"  raw cv2.CAP_PROP_FRAME_COUNT     = {raw}     (identical: {meta.frame_count == raw})")
    print(f"  VideoMeta.duration_s             = {meta.duration_s}")
    print(f"  frame_count / effective_fps      = {meta.frame_count / (raw_fps or 25.0)}")
    print("    -> duration_s is COMPUTED from frame_count (probe_video:822). An error")
    print("       in frame_count lands in BOTH MediaInfo.frame_count and MediaInfo.duration_s.")
    print(f"\n  sample_frame_indices(frame_count={meta.frame_count}, n={DEFAULT_N}) picks the frames")
    idx = pre.sample_frame_indices(meta.frame_count, DEFAULT_N)
    print(f"    first 8: {idx[:8]} ...  last: {idx[-1]}")
    print("    -> this is the ONLY consumer of frame_count that changes a score.")


# --------------------------------------------------------------------------
# S2 — the amplifier, and whether this measurement is portable
# --------------------------------------------------------------------------


def s2_amplifier() -> None:
    head(
        2,
        "The amplifier (E2 re-measured) — and this measurement is decoder-INDEPENDENT",
        "sample_frame_indices is uniform over the declared count",
        "an off-by-k frame_count moves most sampled indices",
    )
    for n_budget in sorted({16, DEFAULT_N}):
        base = pre.sample_frame_indices(1000, n_budget)
        print(f"  n_frames={n_budget}, reference frame_count=1000:")
        for delta in (-10, -5, -2, -1, 1, 2, 5, 10):
            other = pre.sample_frame_indices(1000 + delta, n_budget)
            differ = sum(1 for a, b in zip(base, other) if a != b)
            print(f"    frame_count={1000 + delta:5d} ({delta:+d}): "
                  f"{differ:3d}/{len(base)} indices differ ({100 * differ / len(base):.0f}%)")
    print("\n  sample_frame_indices is pure integer arithmetic — no cv2, no decode.")
    print("  So THIS number is portable and pinnable, unlike a real cross-decoder")
    print("  frame_count divergence, which would be as build-specific as the bug.")


# --------------------------------------------------------------------------
# S3 — can one build be made to miscount?
# --------------------------------------------------------------------------


def s3_attempt_real_divergence() -> None:
    head(
        3,
        "Attempt: make one build report a wrong frame_count with a real container",
        "a damaged index / trailing garbage / patched mvhd changes CAP_PROP_FRAME_COUNT",
        "reproduce E1's 'unconfirmed trigger' as confirmed",
    )
    raw = clip(TMP / "s3_ref.mp4", frames=60).read_bytes()

    def count(data: bytes, api=cv2.CAP_ANY) -> str:
        p = TMP / "s3_probe.mp4"
        p.write_bytes(data)
        c = cv2.VideoCapture(str(p), api)
        if not c.isOpened():
            c.release()
            return "refused"
        n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        grabbed = 0
        while c.grab():
            grabbed += 1
        c.release()
        return f"declared={n:3d} decoded={grabbed:3d}"

    ref = count(raw)
    print(f"  reference                         : {ref}")
    print(f"  + 4 KiB trailing zeros            : {count(raw + b'\\x00' * 4096)}")
    print(f"  + 4 KiB trailing 0xFF            : {count(raw + b'\\xff' * 4096)}")
    print(f"  truncated to 99% (keeps moov?)   : {count(raw[: int(len(raw) * 0.99)])}")
    print(f"  truncated to 90%                 : {count(raw[: int(len(raw) * 0.90)])}")

    # Patch the 32-bit big-endian value right after every 'stsz' sample-count-ish
    # marker is too fragile to do portably; instead flip the last mdat byte, a
    # realistic 'one corrupt frame' case.
    patched = bytearray(raw)
    patched[-1] ^= 0xFF
    print(f"  last byte flipped                : {count(bytes(patched))}")

    backends = {
        cv2.videoio_registry.getBackendName(b): b
        for b in cv2.videoio_registry.getBackends()
    }
    for name in ("FFMPEG", "MSMF"):
        if name in backends:
            print(f"  reference via {name:7s}            : {count(raw, backends[name])}")

    print("\n  VERDICT: on this one build, an intact container is counted exactly and a")
    print("  damaged one is either counted the same or refused outright. Consistent")
    print("  with E1. The divergence regime — VFR media, a demuxer that estimates")
    print("  nb_frames from duration*rate, a partially-rebuilt index — needs media")
    print("  this box cannot author and a second libavformat build. S5 uses a double.")


# --------------------------------------------------------------------------
# S4 — does decode_identity cover the demuxer?
# --------------------------------------------------------------------------


def s4_identity_gap() -> None:
    head(
        4,
        "Does decode_identity cover the library that produces frame_count?",
        "CAP_PROP_FRAME_COUNT is a libavformat quantity; decode_identity covers libavcodec",
        "two builds sharing avcodec but not avformat share an identity",
    )
    info = cv2.getBuildInformation()
    lines = {
        k: next((l.strip() for l in info.splitlines() if l.strip().startswith(k)), "(absent)")
        for k in ("avcodec:", "avformat:", "avutil:", "swscale:")
    }
    for k, v in lines.items():
        print(f"  {v}")
    ffmpeg, baseline, dispatch = _build_facts()
    print(f"\n  decode_env._build_facts() extracted ffmpeg_version = {ffmpeg!r}")
    print(f"    (this is the avcodec version; avformat is never parsed)")

    env = decode_environment()
    print(f"\n  DecodeEnvironment.identity        = {env.identity}")
    print(f"  fields feeding it: opencv, backend, ffmpeg(=avcodec), system, machine")
    print(f"  'avformat' appears in the identity inputs : "
          f"{'avformat' in str(env.to_dict()).lower()}")
    print("\n  -> The demuxer version is structurally absent from the cache-key")
    print("     segment (Batch 9) and the BenchmarkRun.decode_identity field (Batch 11).")
    print("     This is R27.")


# --------------------------------------------------------------------------
# S5 — fail-before: one identity, two frame counts, two verdicts
# --------------------------------------------------------------------------


class MiscountingDemuxer:
    """Stands in for a libavformat build that reports one frame fewer.

    Wraps the real ``probe_video`` and subtracts 1 from ``frame_count`` — the
    exact E2 witness. Everything else (fps, size, the pixels) is untouched: this
    is a *metadata* divergence, the regime E1 could not author on one build.
    """

    name = "probe_video with frame_count -> frame_count - 1 (a different demuxer)"

    def __init__(self, real):
        self._real = real

    def __call__(self, path, *, caps=pre.DEFAULT_CAPS):
        m = self._real(path, caps=caps)
        return pre.VideoMeta(m.fps, m.frame_count - 1, m.duration_s, m.width, m.height)


def _run_video(data: bytes) -> tuple:
    reg = ModelRegistry()
    reg.register(ModelSpec(name="v", version="stub-0", backend="stub", input_size=384))
    eng = VideoPipeline(model_name="v", registry=reg, n_frames=16)
    bundle = asyncio.run(eng.analyze_video(_spec(data), data))
    return bundle, eng


def s5_fail_before() -> None:
    head(
        5,
        "Fail-before: two demuxers disagree by one frame, one identity string",
        "",
        "different sampled frames, different verdict, IDENTICAL model_version + cache key",
    )
    data = clip(TMP / "s5.mp4", frames=200, seed=5).read_bytes()

    real_probe = pre.probe_video
    b_true, eng_true = _run_video(data)
    id_true = decode_identity()

    pre.probe_video = MiscountingDemuxer(real_probe)  # type: ignore[assignment]
    try:
        b_off, eng_off = _run_video(data)
        id_off = decode_identity()
    finally:
        pre.probe_video = real_probe  # type: ignore[assignment]

    idx_true = [f.frame_idx for f in b_true.video.frame_scores]
    idx_off = [f.frame_idx for f in b_off.video.frame_scores]
    moved = sum(1 for a, b in zip(idx_true, idx_off) if a != b)

    print(f"  true demuxer  : frame_count={b_true.media.frame_count}  "
          f"p_fake={b_true.p_fake}  verdict={b_true.verdict.value}")
    print(f"  -1  demuxer   : frame_count={b_off.media.frame_count}  "
          f"p_fake={b_off.p_fake}  verdict={b_off.verdict.value}")
    print(f"\n  sampled frame indices moved      : {moved}/{len(idx_true)}")
    print(f"  p_fake changed                   : {b_true.p_fake != b_off.p_fake}")
    print(f"  verdict changed                  : {b_true.verdict != b_off.verdict}")
    print(f"  MediaInfo.duration_s changed     : {b_true.media.duration_s != b_off.media.duration_s}")
    print(f"\n  decode_identity() true / -1       : {id_true} / {id_off}   "
          f"(identical: {id_true == id_off})")
    print(f"  bundle.model.version true         : {b_true.model.version}")
    print(f"  bundle.model.version -1           : {b_off.model.version}")
    print(f"  model.version identical           : {b_true.model.version == b_off.model.version}")
    print(f"  sha256 identical (same bytes)     : {b_true.media.sha256 == b_off.media.sha256}")
    print(f"  content cache key identical       : "
          f"{(b_true.media.sha256, b_true.model.version) == (b_off.media.sha256, b_off.model.version)}")
    print("\n  -> A cache populated by the first deployment would serve its verdict to")
    print("     the second, which scores a different clip. Same cache key. Same")
    print("     'would this deployment produce the same number?' string. That is the")
    print("     residual, executed.")


# --------------------------------------------------------------------------
# S6 — consumers
# --------------------------------------------------------------------------


def s6_consumers() -> None:
    head(6, "Which artifacts consume frame_count")
    root = Path(__file__).resolve().parent.parent
    consumers: dict[str, list[str]] = {}
    for sub in ("inference", "contracts", "backend", "workers", "evaluation", "jobqueue", "storage"):
        d = root / sub
        if not d.exists():
            continue
        for p in d.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if "frame_count" in line and not line.strip().startswith("#"):
                    consumers.setdefault(str(p.relative_to(root)), []).append(f"{i}: {line.strip()[:90]}")
    for f, hits in sorted(consumers.items()):
        print(f"  {f}")
        for h in hits:
            print(f"      {h}")
    print("\n  Absent from: the content cache key (workers/io.py content_key_for =")
    print("  sha256 + model_version), the idempotency key, evaluation/ (the harness")
    print("  never decodes), CalibrationWitness, SplitLock, regression_gate, and")
    print("  DecodeEnvironment (no per-file fields). frame_count reaches the wire")
    print("  (MediaInfo.frame_count, MediaInfo.duration_s) and sample_frame_indices.")
    print("\n  Batch 23 (R29) note: n_frames — the sampling *budget*, distinct from")
    print("  frame_count the *measurement* — IS now in the model identity via")
    print("  VideoSelectionPolicy. That does not touch R6-a/R6-b: a demuxer that")
    print("  miscounts frame_count still shifts every uniformly-sampled index under")
    print("  one identity string. R6's fix is timestamp-fraction sampling plus a")
    print("  decodable-count cross-check (§9 of docs/FRAME_COUNT_REPRODUCIBILITY.md).")


def main() -> int:
    print("R6 RESIDUAL — frame_count reproducibility")
    print(f"decode env: {decode_environment().describe()}")
    for fn in (s1_what_it_is, s2_amplifier, s3_attempt_real_divergence,
               s4_identity_gap, s5_fail_before, s6_consumers):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"\n  !! {fn.__name__} raised {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{'=' * 74}\nartifacts in {TMP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
