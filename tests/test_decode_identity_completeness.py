"""R27 — decode identity completeness for the video decode path (Batch 20).

THE FINDING, RE-DERIVED (not assumed)
------------------------------------
``DecodeEnvironment.identity`` — the Batch 9 cache-key segment and the Batch 11
``BenchmarkRun`` / ``CalibrationWitness`` field — digested, for FFmpeg, only the
**libavcodec** version. The video decode path links four FFmpeg libraries, and
``getBuildInformation()`` reports each separately (measured here:
``avcodec 58.134.100``, ``avformat 58.76.100``, ``avutil 56.70.100``,
``swscale 5.9.100``):

* **libavformat** (demuxer) produces ``CAP_PROP_FRAME_COUNT`` / ``FPS`` /
  ``POS_MSEC`` and therefore decides *which frames are sampled* (R27, Batch 19 —
  executed: two demuxers off by one frame moved 11/16 sampled indices);
* **libswscale** converts the decoded YUV frame to BGR in
  ``VideoCapture.retrieve()`` (Batch 8 E4 measured a chroma-upsampling
  difference at that stage);
* **libavutil** carries the pixfmt descriptors and the rational arithmetic
  behind timestamps.

So two deployments differing in any of those three — a routine distro-vs-prebuilt
difference — shared a decode identity, and could disagree on the sampled frames
and the verdict while colliding on one content-cache key, one benchmark
identity, and one calibration identity.

WHAT THIS BATCH DOES
--------------------
Grows the FFmpeg component of ``decode_identity`` from libavcodec alone to
**every FFmpeg library on the video path** (:data:`_VIDEO_FFMPEG_LIBS` —
avcodec + avformat + avutil + swscale; ``avresample`` excluded, it is FFmpeg's
audio resampler). One string field, one place in the digest — the minimal
expansion. ``STUB_ENGINE_VERSION``'s ``.d.`` segment moves ``c20a064a`` ->
``a97ccb68`` (a one-time cache generation, precedented by Batch 9). **No**
``CONTRACT_VERSION`` / ``RUN_SCHEMA_VERSION`` / ``CALIBRATION_WITNESS_VERSION`` /
OpenAPI change.

WHY DOUBLES
-----------
This box has one FFmpeg. ``LegacyAvcodecOnlyIdentity`` reproduces the pre-Batch-20
composition (bare libavcodec version) verbatim, so "the identity grew" is checked
against what it was. ``mismatched_ffmpeg`` builds two ``DecodeEnvironment``s that
differ only in one FFmpeg library's version — what a genuine cross-build
difference looks like in ``getBuildInformation`` — since the real libraries
cannot be swapped.
"""

from __future__ import annotations

import inspect
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from contracts.enums import CONTRACT_VERSION  # noqa: E402
from contracts.version import (  # noqa: E402
    DECODE_MARKER,
    compose_runtime_version,
    fingerprint,
    model_identity_of,
)
from evaluation.benchmark import (  # noqa: E402
    RUN_SCHEMA_VERSION,
    BenchmarkRun,
    StubPredictor,
    regression_gate,
    run_benchmark,
)
from evaluation.calibration_protocol import (  # noqa: E402
    CALIBRATION_WITNESS_VERSION,
    CalibrationWitness,
)
from evaluation.dataset import MediaItem  # noqa: E402
from inference import decode_env as de  # noqa: E402
from inference.decode_env import (  # noqa: E402
    _VIDEO_FFMPEG_LIBS,
    _build_facts,
    decode_environment,
    decode_fingerprint,
    decode_identity,
)
from contracts.common import DEFAULT_LOG_ODDS_MARGIN, ThresholdInfo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# S8 Batch 22 (R25) moved the MODEL half — weights ``w7f45fada`` -> ``wacdd9240``
# and face-selection ``19559387`` -> ``251d22e9``. S8 Batch 23 (R29) moved the
# ``.f.`` segment again — ``251d22e9`` -> ``e629642c`` — when it became the
# composite of the face and video selection policies. Batch 20 (this suite's
# subject) still touched only the ``.d.`` decode segment; this is the collateral
# update so ``model_identity_of`` is checked against the current literal.
MODEL_ID = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c"
TH = ThresholdInfo.at_margin(value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02)


# --------------------------------------------------------------------------
# Named doubles
# --------------------------------------------------------------------------


class LegacyAvcodecOnlyIdentity:
    """``DecodeEnvironment.identity`` as of Batch 9-19: FFmpeg = the bare
    libavcodec version, nothing else."""

    name = "decode identity, pre-Batch-20 (bare libavcodec version only)"

    @staticmethod
    def avcodec_of(env) -> str | None:
        for part in (env.ffmpeg_version or "").split(" / "):
            if part.startswith("avcodec "):
                return part[len("avcodec "):]
        return env.ffmpeg_version or None

    @classmethod
    def identity_of(cls, env) -> str:
        return fingerprint({
            "opencv": env.opencv_version,
            "backend": env.video_backend,
            "ffmpeg": cls.avcodec_of(env),
            "system": env.platform_system,
            "machine": env.platform_machine,
        })


def mismatched_ffmpeg(lib: str, old: str, new: str):
    """Two ``DecodeEnvironment``s identical except ``<lib>`` version old -> new."""
    env = decode_environment()
    assert env.ffmpeg_version and f"{lib} {old}" in env.ffmpeg_version, (
        f"this build does not report {lib} {old}; adjust the fixture"
    )
    other = replace(env, ffmpeg_version=env.ffmpeg_version.replace(f"{lib} {old}", f"{lib} {new}"))
    return env, other


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _items(n: int = 40) -> list[MediaItem]:
    return [
        MediaItem(media_id=f"m{i:04d}", label=i % 2, dataset_id="d",
                  source_id=f"s{i // 4}", subject_id=f"p{i // 8}")
        for i in range(n)
    ]


_ITEMS = _items()


def _bench(decode: str | None) -> BenchmarkRun:
    return run_benchmark(
        StubPredictor(version="stub-0", skill=0.3),
        _ITEMS,
        threshold=TH,
        effective_model_version=MODEL_ID,
        decode_identity=decode,
    )


def _witness_fields(**over) -> dict:
    base = dict(
        protocol_name="ffpp-train/celebdf-primary/v1", role="calibration",
        corpus_fingerprint="cafe1234", n_samples=400, temperature=1.234567,
        nll_before=0.69, nll_after=0.61, threshold=0.62, margin=DEFAULT_LOG_ODDS_MARGIN,
        target_fpr=0.02, selection_witness="sel12345", split_lock_hash="lock1234",
    )
    base.update(over)
    return base


def _witness(**over) -> CalibrationWitness:
    f = _witness_fields(**over)
    return CalibrationWitness(**f, witness=CalibrationWitness.compute_witness(**f))


# --------------------------------------------------------------------------
# 0. The legacy double is faithful
# --------------------------------------------------------------------------


def test_the_legacy_double_reproduces_the_pre_batch_20_identity():
    """It must equal ``c20a064a`` on this build — the value Batch 9/19 recorded —
    or "the identity grew" is measured against the wrong baseline."""
    env = decode_environment()
    legacy = LegacyAvcodecOnlyIdentity.identity_of(env)
    assert legacy != env.identity, "the identity did not change — Batch 20 is a no-op"
    # sanity: the legacy value is a pure function of the avcodec version + the
    # unchanged fields, so on this specific build it is the historical constant.
    if env.platform_system == "Windows" and env.opencv_version == "4.8.1":
        assert legacy == "c20a064a", f"legacy double drifted: {legacy}"


# --------------------------------------------------------------------------
# 1. What decode_identity now covers
# --------------------------------------------------------------------------


def test_decode_identity_now_covers_every_ffmpeg_library_on_the_video_path():
    ffmpeg_libs, _, _ = _build_facts()
    assert ffmpeg_libs is not None
    for lib in _VIDEO_FFMPEG_LIBS:
        assert lib in ffmpeg_libs, f"{lib} missing from the FFmpeg component: {ffmpeg_libs!r}"
    # and the identity digests that string
    env = decode_environment()
    assert env.ffmpeg_version == ffmpeg_libs
    # every lib version is inside the identity's preimage
    src = inspect.getsource(de.DecodeEnvironment.identity.fget)
    assert "self.ffmpeg_version" in src


def test_the_video_ffmpeg_libs_list_is_exactly_the_four_on_the_path():
    assert _VIDEO_FFMPEG_LIBS == ("avcodec", "avformat", "avutil", "swscale")


def test_avresample_is_excluded_because_it_is_audio_only():
    assert "avresample" not in _VIDEO_FFMPEG_LIBS
    ffmpeg_libs, _, _ = _build_facts()
    assert "avresample" not in (ffmpeg_libs or "")
    # no audio decode API anywhere in inference/
    hits = []
    for p in (ROOT / "inference").rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            for kw in ("swr_", "swresample", "CAP_PROP_AUDIO", "audio_stream"):
                if kw in code:
                    hits.append(f"{p.name}:{i}")
    assert hits == [], f"audio decode API in use: {hits}"


def test_swscale_is_linked_on_the_video_path():
    """libswscale is not incidental: OpenCV's FFMPEG capture converts YUV->BGR
    with it in retrieve()."""
    info = cv2.getBuildInformation()
    sws = next((l.strip() for l in info.splitlines() if l.strip().startswith("swscale:")), "")
    assert sws.startswith("swscale:") and "YES" in sws, f"swscale not linked: {sws!r}"
    assert "swscale" in (_build_facts()[0] or ""), "swscale is not captured in the identity"


# --------------------------------------------------------------------------
# 2. The reproducibility hole — fail-before / pass-after, all four protections
# --------------------------------------------------------------------------


def _four_protections_separate(id_a: str, id_b: str) -> dict[str, bool]:
    rt_a = compose_runtime_version(model=MODEL_ID, decode=id_a)
    rt_b = compose_runtime_version(model=MODEL_ID, decode=id_b)
    return {
        "content_cache_key": rt_a != rt_b,      # workers/io: sha256 + runtime version
        "deployment_identity": rt_a != rt_b,    # /health, verify_version_alignment
        "benchmark_identity": id_a != id_b,     # regression_gate._decode_comparability
        "calibration_identity": id_a != id_b,   # CalibrationWitness digest
    }


def test_a_libavformat_difference_now_separates_all_four_protections():
    """**The headline. Fail-before / pass-after.**

    Two deployments identical except libavformat 58.76.100 vs 59.16.100 — the
    demuxer that decides which frames are sampled.
    """
    env, other = mismatched_ffmpeg("avformat", "58.76.100", "59.16.100")

    after = _four_protections_separate(env.identity, other.identity)
    assert all(after.values()), f"a protection is still blind after Batch 20: {after}"

    before = _four_protections_separate(
        LegacyAvcodecOnlyIdentity.identity_of(env),
        LegacyAvcodecOnlyIdentity.identity_of(other),
    )
    assert not any(before.values()), (
        f"the pre-Batch-20 identity is not blind to the demuxer — the fixture is wrong: {before}"
    )


def test_a_libswscale_difference_now_moves_the_identity():
    env, other = mismatched_ffmpeg("swscale", "5.9.100", "6.1.100")
    assert other.identity != env.identity
    assert LegacyAvcodecOnlyIdentity.identity_of(env) == LegacyAvcodecOnlyIdentity.identity_of(other)


def test_a_libavutil_difference_now_moves_the_identity():
    env, other = mismatched_ffmpeg("avutil", "56.70.100", "57.28.100")
    assert other.identity != env.identity


def test_the_benchmark_gate_refuses_across_a_demuxer_difference():
    """``regression_gate`` now refuses a comparison whose two sides carry the
    finer decode identities that differ on libavformat."""
    env, other = mismatched_ffmpeg("avformat", "58.76.100", "59.16.100")
    a = _bench(env.identity)
    b = _bench(other.identity)
    gate = regression_gate(a, b)
    assert not gate.passed
    assert any("decode" in f.lower() for f in gate.failures), gate.failures


def test_the_calibration_witness_binds_the_finer_identity():
    env, other = mismatched_ffmpeg("avformat", "58.76.100", "59.16.100")
    wa = _witness(decode_identity=env.identity)
    wb = _witness(decode_identity=other.identity)
    assert wa.witness != wb.witness, "two demuxers produced one witness digest"


# --------------------------------------------------------------------------
# 3. The CPU fields stay out — Batch 9's decision preserved
# --------------------------------------------------------------------------


def test_the_cpu_fields_are_still_excluded_from_the_identity():
    env = decode_environment()
    other = replace(env, cpu_baseline="SSE SSE2", cpu_dispatch="")
    if (env.cpu_baseline, env.cpu_dispatch) != (other.cpu_baseline, other.cpu_dispatch):
        assert other.identity == env.identity, "a CPU difference fragmented the cache"
        assert other.digest != env.digest, "the forensic digest must still see it"


def test_the_forensic_digest_is_still_strictly_finer_than_the_identity():
    env = decode_environment()
    assert env.identity != env.digest
    assert env.digest == decode_fingerprint()
    payload = env.to_dict()
    assert payload["identity"] == env.identity and payload["digest"] == env.digest


# --------------------------------------------------------------------------
# 4. Migration — what moved and what did not
# --------------------------------------------------------------------------


def test_the_decode_segment_moved_and_the_literal_was_updated():
    from backend.config import STUB_ENGINE_VERSION
    from workers.engine_registry import verify_version_alignment

    assert DECODE_MARKER in STUB_ENGINE_VERSION
    assert STUB_ENGINE_VERSION.endswith(f"{DECODE_MARKER}{decode_identity()}")
    assert STUB_ENGINE_VERSION.endswith(".d.a97ccb68"), STUB_ENGINE_VERSION
    verify_version_alignment(STUB_ENGINE_VERSION)  # raises if the literal is stale


def test_the_model_identity_did_not_move_in_batch_20():
    """Batch 20 touched only the ``.d.`` segment. ``MODEL_ID`` was later moved by
    Batch 22 (R25); this checks ``model_identity_of`` strips ``.d.`` cleanly from
    the current literal and recovers exactly ``MODEL_ID``."""
    from backend.config import STUB_ENGINE_VERSION

    assert model_identity_of(STUB_ENGINE_VERSION) == MODEL_ID, (
        "model_identity_of did not recover the recorded MODEL_ID from the literal"
    )


def test_no_contract_schema_or_witness_version_bump():
    assert CONTRACT_VERSION == "1.0.0"
    assert RUN_SCHEMA_VERSION == "1.2.0"
    assert CALIBRATION_WITNESS_VERSION == "1.3.0"


def test_the_openapi_snapshot_does_not_carry_the_decode_segment():
    snap = (ROOT / "backend/openapi.snapshot.json").read_text(encoding="utf-8")
    assert "c20a064a" not in snap and "a97ccb68" not in snap
    assert ".d." not in snap  # the runtime version is a value, not structure


def test_an_old_benchmark_row_still_loads_and_a_cross_boundary_comparison_refuses():
    """Benchmark incompatibility handling: none required. An old row keeps its
    coarse ``decode_identity`` string; a new row has the finer one; the gate
    refuses to compare them (correct — the identity scheme changed)."""
    old = _bench("c20a064a")          # a run made just before Batch 20
    payload = json.loads(old.to_json())
    assert payload["decode_identity"] == "c20a064a"
    reloaded = BenchmarkRun.from_dict(payload)
    assert reloaded.decode_identity == "c20a064a"

    new = _bench(decode_identity())    # the same environment, post-Batch-20
    gate = regression_gate(reloaded, new)
    assert not gate.passed
    assert any("decode" in f.lower() for f in gate.failures)


def test_an_old_calibration_witness_still_verifies_after_the_change():
    """The witness digest is over the ``decode_identity`` *string it stored*, not
    over a recomputation, so changing what ``decode_identity()`` returns does not
    invalidate a stored witness. Hence no ``CALIBRATION_WITNESS_VERSION`` bump."""
    w = _witness(decode_identity="c20a064a")
    payload = w.to_dict()
    reloaded = CalibrationWitness.from_dict(payload)  # __post_init__ raises on a digest mismatch
    assert reloaded.decode_identity == "c20a064a"
    assert reloaded.witness == w.witness


def test_a_pre_batch_11_witness_and_run_are_unaffected():
    """decode_identity=None -> "not recorded" -> a note, never a pass. Unchanged."""
    a = _bench(None)
    b = _bench(None)
    gate = regression_gate(a, b)
    assert gate.passed  # None vs None is a note, not a refusal
    assert any("decode" in n.lower() for n in gate.notes)


# --------------------------------------------------------------------------
# 5. Residuals — stated, not fixed
# --------------------------------------------------------------------------


def test_the_wheel_patch_version_gap_is_a_known_residual():
    """``cv2.__version__`` drops the opencv-python wheel's 4th component. Capturing
    the FFmpeg libs directly moots this for the *decode* path; OpenCV's own
    imgproc changing between wheel patches without an FFmpeg change is the
    residual (low-risk: B8 E6 measured resize/cvtColor dispatch-independent)."""
    import importlib.metadata as md

    try:
        wheel = md.version("opencv-python")
    except md.PackageNotFoundError:
        return  # distro/conda build — no wheel metadata, which is its own argument
    assert wheel.startswith(cv2.__version__ + "."), (wheel, cv2.__version__)
    assert wheel != cv2.__version__, "the wheel's 4th digit is not dropped on this box"


def test_same_version_different_build_is_still_a_blind_spot():
    """Two FFmpeg builds reporting identical version strings but compiled
    differently (a distro patch, a different chroma-filter default) still share
    an identity. This is the R27 residual the amplifier double models; it is not
    closable by a version string."""
    env = decode_environment()
    twin = replace(env)  # identical fields
    assert twin.identity == env.identity  # by construction — the residual


def test_batch_20_did_not_touch_the_frame_count_amplifier_or_validation():
    """R6-a (uniform sampling) and R6-b (no validation) are untouched — B20 is an
    identity change only."""
    from inference.preprocess import probe_video, sample_frame_indices

    assert "int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)" in inspect.getsource(probe_video)
    base = sample_frame_indices(60, 48)
    assert sum(1 for a, b in zip(base, sample_frame_indices(59, 48)) if a != b) >= 24


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
