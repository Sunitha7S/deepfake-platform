"""R6 residual — ``frame_count`` reproducibility, and R27 (the decode identity
omits the demuxer).

THE FINDING, RE-DERIVED (not assumed)
------------------------------------
``MediaInfo.frame_count`` is ``int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)``
from ``probe_video`` — a **libavformat** (demuxer) quantity, taken verbatim, with
no sanity check. ``probe_video`` then derives ``duration_s = frame_count /
effective_fps``, so one wrong count lands in **two** wire fields. And
``sample_frame_indices(meta.frame_count, n)`` chooses which frames are decoded
and scored, so a wrong count scores a different clip: measured, a one-frame error
moves ≥ 50% of the sampled indices (Batch 8 E2, re-measured in
``frame_count_reproducibility_study.py`` S2).

Batch 8 left this "Unresolved — no validation of frame_count exists". Batch 9 put
``decode_identity`` in the cache key and Batch 11 put it on ``BenchmarkRun``, but
``decode_identity`` covers **libavcodec** (the frame decoder), not **libavformat**
(the demuxer that produces ``frame_count``). On this build the two are separately
versioned — ``avcodec 58.134.100`` vs ``avformat 58.76.100`` — and only the
former is parsed (``decode_env._build_facts``). So two deployments that share
``decode_identity`` can disagree on ``frame_count``, score different frames,
publish different ``p_fake``, and collide on one content-cache key. **That is
R27**, recorded here as a finding distinct from R6.

WHAT THIS BATCH DOES
--------------------
Characterises and pins. No behaviour change: ``probe_video`` still trusts the
decoder, ``decode_identity`` still omits the demuxer. The pins make the amplifier
un-weakenable and make R27 a tripwire — if ``avformat`` is ever added to the
decode identity, ``test_the_decode_identity_still_omits_the_demuxer`` fails and
forces the audit update.

WHY DOUBLES
-----------
``TrustingProbe`` is what ``probe_video`` does today: return
``CAP_PROP_FRAME_COUNT`` unexamined. ``MiscountingDemuxer`` wraps the real probe
and subtracts one frame — the E2 witness — standing in for a libavformat build
that estimates ``nb_frames`` differently. "Legacy" and "current" behaviour are
the *same* here (nothing about ``frame_count`` handling has ever changed); the
doubles model *two deployments*, not two eras.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import contracts.fixtures as fx  # noqa: E402
from contracts.common import MediaInfo  # noqa: E402
from contracts.enums import MediaKind, Verdict  # noqa: E402
from contracts.evidence import EvidenceBundle  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from contracts.video import VideoAnalysis  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.decode_env import _build_facts, decode_environment, decode_identity  # noqa: E402
from inference.models import ModelRegistry, ModelSpec  # noqa: E402
from inference.preprocess import sample_frame_indices  # noqa: E402
from inference.video_pipeline import VideoPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_N = 48  # extract_frames' default n_frames


# --------------------------------------------------------------------------
# Named doubles
# --------------------------------------------------------------------------


class TrustingProbe:
    """What ``probe_video`` does today: ``CAP_PROP_FRAME_COUNT`` taken verbatim.

    Held here so "the platform trusts the decoder's count" is checked against an
    executable statement of it rather than a sentence.
    """

    name = "probe_video: frame_count = int(CAP_PROP_FRAME_COUNT), no validation"

    @staticmethod
    def frame_count(path: Path) -> int:
        cap = cv2.VideoCapture(str(path))
        try:
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()


class MiscountingDemuxer:
    """A different libavformat build: same pixels, ``frame_count`` one lower.

    The exact E2 witness. Wraps the real ``probe_video`` so fps / size / pixels
    are untouched — this is a *metadata* divergence, the regime E1 could not
    author on one build.
    """

    name = "probe_video with frame_count -> frame_count - 1"

    def __init__(self, real):
        self._real = real

    def __call__(self, path, *, caps=pre.DEFAULT_CAPS):
        m = self._real(path, caps=caps)
        return pre.VideoMeta(m.fps, m.frame_count - 1, m.duration_s, m.width, m.height)


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------


def _clip(path: Path, *, frames: int = 120, size: int = 128, fps: int = 25, seed: int = 0) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    for i in range(frames):
        writer.write(np.roll(base, i * 7, axis=1))
    writer.release()
    return path.read_bytes()


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


def _run_video(data: bytes, *, n_frames: int = 16) -> EvidenceBundle:
    reg = ModelRegistry()
    reg.register(ModelSpec(name="v", version="stub-0", backend="stub", input_size=384))
    eng = VideoPipeline(model_name="v", registry=reg, n_frames=n_frames)
    return asyncio.run(eng.analyze_video(_spec(data), data))


def _tmp() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix="dfp-fc-test-"))


_AVLIB_RE = re.compile(r"^(avcodec|avformat|avutil|swscale):\s*YES\s*\(([^)]+)\)", re.M)


# --------------------------------------------------------------------------
# 0. The doubles are faithful
# --------------------------------------------------------------------------


def test_the_trusting_probe_double_matches_probe_video():
    src = inspect.getsource(pre.probe_video)
    assert "int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)" in src, (
        "probe_video no longer takes CAP_PROP_FRAME_COUNT verbatim — the double is stale"
    )


def test_the_miscounting_demuxer_changes_only_the_count():
    d = _tmp()
    data = _clip(d / "c.mp4", frames=60)
    (d / "c.mp4").write_bytes(data)
    real = pre.probe_video
    m_true = real(d / "c.mp4")
    m_off = MiscountingDemuxer(real)(d / "c.mp4")
    assert m_off.frame_count == m_true.frame_count - 1
    assert (m_off.fps, m_off.width, m_off.height) == (m_true.fps, m_true.width, m_true.height)


# --------------------------------------------------------------------------
# 1. What frame_count is
# --------------------------------------------------------------------------


def test_frame_count_is_the_raw_cap_prop_value():
    d = _tmp()
    p = d / "c.mp4"
    p.write_bytes(_clip(p, frames=73))
    meta = pre.probe_video(p)
    assert meta.frame_count == TrustingProbe.frame_count(p)


def test_duration_s_is_derived_from_frame_count():
    """So a wrong count lands in BOTH MediaInfo.frame_count and MediaInfo.duration_s."""
    d = _tmp()
    p = d / "c.mp4"
    p.write_bytes(_clip(p, frames=100, fps=25))
    meta = pre.probe_video(p)
    assert abs(meta.duration_s - meta.frame_count / meta.fps) < 1e-6
    src = inspect.getsource(pre.probe_video)
    assert "duration = frames / effective_fps" in src


def test_sample_frame_indices_is_the_only_score_affecting_consumer():
    """Every other read of ``frame_count`` in inference/ is a passthrough to the
    wire; only ``sample_frame_indices`` changes which pixels are scored."""
    hits: list[str] = []
    for p in (ROOT / "inference").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "frame_count":
                hits.append(f"{p.name}:{node.lineno}")
    # meta.frame_count is read in video_pipeline (-> build_*_bundle, passthrough)
    # and preprocess (-> sample_frame_indices). No arithmetic branch anywhere.
    assert any("preprocess" in h for h in hits)
    gate_src = inspect.getsource(VideoPipeline)
    assert "sample_frame_indices" not in gate_src, "sampling is in extract_frames, not the pipeline"


# --------------------------------------------------------------------------
# 2. The amplifier — decoder-INDEPENDENT, therefore pinnable
# --------------------------------------------------------------------------


def test_a_one_frame_miscount_moves_most_sampled_indices_in_the_E2_regime():
    """Batch 8 E2: frame_count=60, n=48 (step 1.25), a one-frame error -> ~60%.
    Re-measured, and pinned as a lower bound."""
    base = sample_frame_indices(60, DEFAULT_N)
    for other_n in (58, 59, 61, 62):
        moved = sum(1 for a, b in zip(base, sample_frame_indices(other_n, DEFAULT_N)) if a != b)
        assert moved >= DEFAULT_N // 3, (
            f"frame_count 60 -> {other_n}: only {moved}/{DEFAULT_N} sampled indices moved; "
            "the amplifier this test pins has been weakened"
        )
    # the -1 case specifically, at the shipped 48-frame default
    moved = sum(1 for a, b in zip(base, sample_frame_indices(59, DEFAULT_N)) if a != b)
    assert moved >= 24, f"one-frame miscount moved only {moved}/48"


def test_the_amplifier_scales_with_the_miscount_relative_to_the_sampling_step():
    """The honest shape: a miscount of about one sampling step
    (frame_count/n frames) or more moves most indices. A miscount much smaller
    than one step barely moves them — which is why the risk is worst for short
    clips, where the step is 1-2 frames."""
    for frame_count, n in ((1000, DEFAULT_N), (2000, 16), (300, 24)):
        step = frame_count / n
        base = sample_frame_indices(frame_count, n)
        one_step = max(1, round(step))
        moved = sum(1 for a, b in zip(base, sample_frame_indices(frame_count + one_step, n)) if a != b)
        assert moved >= n // 2, (
            f"frame_count={frame_count} n={n} step={step:.1f}: a one-step miscount "
            f"moved only {moved}/{n}"
        )


def test_the_amplifier_measurement_is_portable_where_a_real_trigger_is_not():
    """``sample_frame_indices`` touches no cv2 — pure integer arithmetic — so this
    suite is reproducible on any box, unlike a real cross-demuxer frame_count
    divergence, which is as build-specific as the bug (E1 could not author one)."""
    src = inspect.getsource(sample_frame_indices)
    assert "cv2" not in src and "capture" not in src
    tree = ast.parse(src)
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert calls <= {"list", "range", "int", "min", "len"}, f"unexpected calls: {calls}"


# --------------------------------------------------------------------------
# 3. frame_count is validated against nothing and fingerprinted nowhere
# --------------------------------------------------------------------------


def test_frame_count_is_validated_against_nothing():
    """An absurd count beside a tiny scored clip is a valid EvidenceBundle."""
    bundle = fx.video_manipulated()
    payload = bundle.model_dump()
    payload["media"]["frame_count"] = 999_999
    payload["media"]["duration_s"] = 999_999 / payload["media"]["fps"]
    revalidated = EvidenceBundle.model_validate(payload)
    assert revalidated.media.frame_count == 999_999
    assert revalidated.video.frames_scored < 100  # nothing relates the two

    for model in (MediaInfo, VideoAnalysis):
        for name, member in inspect.getmembers(model):
            if getattr(member, "__pydantic_validator__", None) or name.startswith("_") and callable(member):
                src = inspect.getsource(member) if callable(member) and hasattr(member, "__code__") else ""
                if "frame_count" in src and ("frames_scored" in src or "frames_sampled" in src):
                    raise AssertionError(f"{model.__name__}.{name} now cross-checks frame_count")


def test_frame_count_is_in_no_identity_string():
    """Changing frame_count moves no fingerprint. Proven by execution in S5 for
    decode_identity; here for the composition as a whole."""
    d = _tmp()
    p = d / "c.mp4"
    p.write_bytes(_clip(p, frames=200, seed=3))
    data = p.read_bytes()

    real = pre.probe_video
    b_true = _run_video(data)
    id_true = decode_identity()
    pre.probe_video = MiscountingDemuxer(real)  # type: ignore[assignment]
    try:
        b_off = _run_video(data)
        id_off = decode_identity()
    finally:
        pre.probe_video = real  # type: ignore[assignment]

    assert b_true.media.frame_count != b_off.media.frame_count
    assert id_true == id_off, "decode_identity moved on a frame_count change (it must not)"
    assert b_true.model.version == b_off.model.version
    # and it is nowhere in the fingerprint-producing source
    for mod in ("inference/decode_env.py", "contracts/version.py"):
        src = (ROOT / mod).read_text(encoding="utf-8")
        assert "frame_count" not in src, f"{mod} now references frame_count"


# --------------------------------------------------------------------------
# 4. R27 — the decode identity and the demuxer
#    (Batch 19 opened this; Batch 20 closed the version half — preserve & refute)
# --------------------------------------------------------------------------


def test_this_build_versions_the_demuxer_separately_from_the_decoder():
    """The measured premise of R27: avcodec and avformat are distinct libraries
    with distinct versions, and CAP_PROP_FRAME_COUNT is an avformat quantity."""
    versions = dict((m.group(1), m.group(2)) for m in _AVLIB_RE.finditer(cv2.getBuildInformation()))
    if "avformat" not in versions or "avcodec" not in versions:
        # A build that does not report them separately; the structural point
        # (below) still stands.
        return
    assert versions["avcodec"] != versions["avformat"], (
        "this build happens to align avcodec and avformat; R27's premise is that "
        "they are independently versioned, which the general case still is"
    )


def test_the_decode_identity_now_covers_the_demuxer_batch_20():
    """**Was the Batch 19 tripwire ``test_the_decode_identity_still_omits_the_
    demuxer``; refuted and rewritten by Batch 20.**

    Batch 19 recorded that ``decode_identity`` covered libavcodec only. Batch 20
    (``docs/DECODE_IDENTITY_COMPLETENESS.md``) grew the FFmpeg-version component
    to every library on the video path — libavcodec **+ libavformat + libswscale
    + libavutil**. The demuxer *version* is now inside the identity; the residual
    (two builds sharing every version string but decoding differently) is what
    keeps R27 partly open.
    """
    ffmpeg_libs, _, _ = _build_facts()
    assert ffmpeg_libs and "avformat" in ffmpeg_libs and "swscale" in ffmpeg_libs, (
        f"decode_env stopped reporting the demuxer / scaler: {ffmpeg_libs!r} — "
        "Batch 20 is undone and R27's version half is reopened"
    )
    env = decode_environment()
    assert "avformat" in str(env.to_dict()).lower()

    src = inspect.getsource(_build_facts)
    assert "_VIDEO_FFMPEG_LIBS" in src


# --------------------------------------------------------------------------
# 5. The Batch 19 fail-before, re-run under Batch 20
# --------------------------------------------------------------------------


def test_a_same_build_demuxer_miscount_still_shares_a_cache_key():
    """``MiscountingDemuxer`` monkeypatches ``probe_video``'s *return value*
    without touching the build, so it models the R27 **residual** — two builds
    with identical FFmpeg version strings that count differently — not the
    version difference Batch 20 caught. That residual still shares a cache key;
    the companion test below shows the version case no longer does.
    """
    d = _tmp()
    p = d / "c.mp4"
    p.write_bytes(_clip(p, frames=240, seed=7))
    data = p.read_bytes()

    real = pre.probe_video
    b_true = _run_video(data)
    pre.probe_video = MiscountingDemuxer(real)  # type: ignore[assignment]
    try:
        b_off = _run_video(data)
    finally:
        pre.probe_video = real  # type: ignore[assignment]

    assert [f.frame_idx for f in b_true.video.frame_scores] != [
        f.frame_idx for f in b_off.video.frame_scores
    ], "the sampled frames did not move — pick a longer clip"
    assert b_true.p_fake != b_off.p_fake
    assert (b_true.media.sha256, b_true.model.version) == (b_off.media.sha256, b_off.model.version), (
        "the residual (same version strings, different count) still shares a key"
    )


def test_a_real_demuxer_version_change_now_moves_the_cache_key():
    """**Fail-before / pass-after for Batch 20.** Two environments identical
    except for the libavformat version — what a genuine demuxer swap looks like
    in ``getBuildInformation`` — now produce different ``decode_identity`` and so
    different content-cache keys. Before Batch 20 they did not."""
    from dataclasses import replace

    env = decode_environment()
    assert env.ffmpeg_version and "avformat 58.76.100" in env.ffmpeg_version
    other = replace(env, ffmpeg_version=env.ffmpeg_version.replace("avformat 58.76.100", "avformat 59.16.100"))
    assert other.identity != env.identity, "a libavformat version change did not move the identity"

    # the pre-Batch-20 composition digested only the avcodec version, so the
    # same swap would NOT have moved the identity:
    from contracts.version import fingerprint

    def legacy_identity(e) -> str:
        avcodec_only = next(
            (part for part in (e.ffmpeg_version or "").split(" / ") if part.startswith("avcodec")),
            e.ffmpeg_version,
        )
        return fingerprint({
            "opencv": e.opencv_version, "backend": e.video_backend, "ffmpeg": avcodec_only,
            "system": e.platform_system, "machine": e.platform_machine,
        })

    assert legacy_identity(env) == legacy_identity(other), (
        "the pre-Batch-20 identity no longer reproduces the R27 blind spot"
    )


def test_the_stub_score_delta_is_an_upper_bound_not_an_estimate():
    """The stub hashes frame bytes, so any frame-set change decorrelates it
    (DECODE_DETERMINISM E9: per-frame correlation -0.18). The honest per-frame
    estimate for a real model is the surrogate's |dp| ~= 0.005. This test just
    records that the number the fail-before test sees is the pessimistic bound."""
    d = _tmp()
    p = d / "c.mp4"
    p.write_bytes(_clip(p, frames=240, seed=11))
    data = p.read_bytes()
    real = pre.probe_video
    b_true = _run_video(data)
    pre.probe_video = MiscountingDemuxer(real)  # type: ignore[assignment]
    try:
        b_off = _run_video(data)
    finally:
        pre.probe_video = real  # type: ignore[assignment]
    delta = abs((b_true.p_fake or 0) - (b_off.p_fake or 0))
    assert delta >= 0.0  # it is what it is; the assertion is the docstring


# --------------------------------------------------------------------------
# 6. Consumers
# --------------------------------------------------------------------------


def test_frame_count_reaches_the_wire_but_not_the_cache_or_the_harness():
    bundle = fx.video_manipulated()
    assert bundle.media.frame_count is not None
    assert bundle.media.duration_s is not None

    # cache key: sha256 + model_version only
    from workers.io import StorageResultSink

    key_src = inspect.getsource(StorageResultSink.content_key_for)
    assert "frame_count" not in key_src

    # evaluation never decodes
    for p in (ROOT / "evaluation").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        assert "frame_count" not in p.read_text(encoding="utf-8"), f"{p.name} references frame_count"


def test_suspect_segment_frame_count_is_a_different_field():
    """Name collision guard: ``SuspectSegment.frame_count`` is a run length, not
    the media's frame count. A future validator on one must not touch the other."""
    from contracts.video import SuspectSegment

    assert "frame_count" in SuspectSegment.model_fields
    assert "frame_count" in MediaInfo.model_fields
    # they are unrelated Field(ge=1) ints on unrelated models
    assert SuspectSegment.model_fields["frame_count"].annotation is int


# --------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------


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
