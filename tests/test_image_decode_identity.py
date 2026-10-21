"""R30 + D8/D9 — image decode library completeness for `decode_identity` (Batch 26).

THE FINDING, RE-DERIVED (not assumed)
--------------------------------------
`DecodeEnvironment.identity` (Batch 9's cache-key segment; Batch 11's
`BenchmarkRun` / `CalibrationWitness` field) covered OpenCV, the video backend
and — since Batch 20 (R27) — every FFmpeg library the video path links. The
*image* decode path's actual codecs (libjpeg-turbo, libpng, libwebp) were
proxied only by `cv2.__version__`, which drops the wheel's 4th digit (D9) and
says nothing at all about a system-linked OpenCV's codec libraries. This is
the image analogue of R27, tracked since the Batch 23 repro audit and
formally re-derived, ranked, and designed in `docs/R30_BUNDLE_DESIGN.md` (and
its inputs, `docs/B24_DESIGN.md` / `docs/B24_ADVERSARIAL_REVIEW.md` /
`docs/B24_R2_DESIGN.md`).

WHAT THIS BATCH DOES
--------------------
Adds `image_codec_version` (libjpeg-turbo / libpng / libwebp, verbatim,
`identity`-affecting) and `image_codec_simd` (libjpeg-turbo's SIMD flag,
`digest`-only — measured performance-only, `cpu_dispatch` precedent) to
`DecodeEnvironment`, via a new sibling function `_image_codec_facts()` that
leaves `_build_facts()` and its five other call sites untouched. zlib is
deliberately absent everywhere (PNG's `inflate()` is RFC-1951-deterministic;
measured identical between stock zlib and zlib-ng — R30 design §1/§2).
`STUB_ENGINE_VERSION`'s `.d.` segment moves `a97ccb68` -> `c9d4025c` (one
cache generation, the B20/B23 precedent). No `CONTRACT_VERSION` /
`RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` / `LOCK_VERSION` /
OpenAPI change.

WHY DOUBLES
-----------
This box has one OpenCV build. `LegacyOpencvProxyIdentity` reproduces the
pre-Batch-26 five-key identity verbatim, so "the identity grew" is checked
against what it was — the `LegacyAvcodecOnlyIdentity` pattern from
`tests/test_decode_identity_completeness.py`. `mismatched_image_codec`
builds two `DecodeEnvironment`s differing only in one codec library's
version string — a genuine cross-build difference, since the real libraries
cannot be swapped on one machine.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.api import ALLOWED_IMAGE_MIME  # noqa: E402
from contracts.enums import CONTRACT_VERSION, MediaKind  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from contracts.version import (  # noqa: E402
    DECODE_MARKER,
    compose_runtime_version,
    fingerprint,
)
from evaluation.benchmark import (  # noqa: E402
    RUN_SCHEMA_VERSION,
    StubPredictor,
    regression_gate,
    run_benchmark,
)
from evaluation.calibration_protocol import (  # noqa: E402
    CALIBRATION_WITNESS_VERSION,
    CalibrationWitness,
)
from evaluation.dataset import MediaItem  # noqa: E402
from contracts.common import DEFAULT_LOG_ODDS_MARGIN, ThresholdInfo  # noqa: E402
from inference import image_pipeline as ip  # noqa: E402
from inference import video_pipeline as vp  # noqa: E402
from inference.decode_env import (  # noqa: E402
    _IMAGE_CODEC_LIBS,
    _build_facts,
    _image_codec_facts,
    _parse_image_codecs,
    decode_environment,
    decode_fingerprint,
    decode_identity,
)
from inference.image_pipeline import ImagePipeline  # noqa: E402
from inference.models import ModelRegistry, ModelSpec  # noqa: E402
from inference.video_pipeline import VideoPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# Unaffected by Batch 26 — only `.d.` moved there. See docs/R30_BUNDLE_DESIGN.md
# §5. SCRFD Integration Program, Phase 2 later moved `.f.` (`e629642c` ->
# `f8244ad0`, docs/SCRFD_PHASE2_SPEC_RESOLUTION.md Decision 7) — this is that
# collateral update.
MODEL_ID = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.f8244ad0"
TH = ThresholdInfo.at_margin(value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02)


# --------------------------------------------------------------------------
# Named doubles
# --------------------------------------------------------------------------


class LegacyOpencvProxyIdentity:
    """`DecodeEnvironment.identity` as of Batch 9-25: 5 keys, no image codec."""

    name = "decode identity, pre-Batch-26 (opencv_version proxies the codecs)"

    @staticmethod
    def identity_of(env) -> str:
        return fingerprint(
            {
                "opencv": env.opencv_version,
                "backend": env.video_backend,
                "ffmpeg": env.ffmpeg_version,
                "system": env.platform_system,
                "machine": env.platform_machine,
            }
        )


def mismatched_image_codec(lib: str, old: str, new: str):
    """Two `DecodeEnvironment`s identical except `<lib>` version old -> new."""
    env = decode_environment()
    assert env.image_codec_version and f"{lib} {old}" in env.image_codec_version, (
        f"this build does not report {lib} {old}; adjust the fixture"
    )
    other = replace(
        env,
        image_codec_version=env.image_codec_version.replace(f"{lib} {old}", f"{lib} {new}"),
    )
    return env, other


def _four_protections_separate(id_a: str, id_b: str) -> dict[str, bool]:
    rt_a = compose_runtime_version(model=MODEL_ID, decode=id_a)
    rt_b = compose_runtime_version(model=MODEL_ID, decode=id_b)
    return {
        "content_cache_key": rt_a != rt_b,
        "deployment_identity": rt_a != rt_b,
        "benchmark_identity": id_a != id_b,
        "calibration_identity": id_a != id_b,
    }


def _items(n: int = 40) -> list[MediaItem]:
    return [
        MediaItem(media_id=f"m{i:04d}", label=i % 2, dataset_id="d",
                  source_id=f"s{i // 4}", subject_id=f"p{i // 8}")
        for i in range(n)
    ]


_ITEMS = _items()


def _bench(decode: str | None):
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


def _spec(data: bytes, kind: MediaKind) -> JobSpec:
    return JobSpec(
        job_id=UUID(int=26),
        media_kind=kind,
        storage_key="uploads/x",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="image/png" if kind is MediaKind.IMAGE else "video/mp4",
        model_version="stub-0",
        idempotency_key="k",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _canvas(w: int = 480, h: int = 480) -> np.ndarray:
    rng = np.random.default_rng(26)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def _analyse_image(data: bytes) -> object:
    registry = ModelRegistry()
    registry.register(ModelSpec(name="i", version="stub-0", backend="stub", input_size=384))
    pipeline = ImagePipeline(model_name="i", registry=registry)
    return asyncio.run(pipeline.analyze_image(_spec(data, MediaKind.IMAGE), data))


def _noise_clip(frames: int = 16, size: int = 160) -> bytes:
    import tempfile

    path = Path(tempfile.mkdtemp(prefix="dfp-r30-")) / "c.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (size, size))
    rng = np.random.default_rng(26)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _analyse_video(clip: bytes) -> object:
    registry = ModelRegistry()
    registry.register(ModelSpec(name="v", version="stub-0", backend="stub", input_size=384))
    # No n_frames override: the deprecated legacy-scalar shim rebuilds the
    # VideoSelectionPolicy via replace(), which — for any field not equal to
    # the shipped default — produces a different .f. composite. Using the
    # pipeline's un-overridden default keeps this test isolated to the .d.
    # axis, which is the only thing this batch touches.
    pipeline = VideoPipeline(model_name="v", registry=registry)
    return asyncio.run(pipeline.analyze_video(_spec(clip, MediaKind.VIDEO), clip))


# --------------------------------------------------------------------------
# 0. The double and the fixture are faithful
# --------------------------------------------------------------------------


def test_the_legacy_double_reproduces_the_pre_batch_26_identity():
    """It must equal `a97ccb68` on this build — the value shipped through
    Batch 25 — or "the identity grew" is measured against the wrong baseline."""
    env = decode_environment()
    legacy = LegacyOpencvProxyIdentity.identity_of(env)
    assert legacy != env.identity, "the identity did not change — this batch is a no-op"
    if env.platform_system == "Windows" and env.opencv_version == "4.8.1":
        assert legacy == "a97ccb68", f"legacy double drifted: {legacy}"


def test_this_build_reports_every_expected_codec():
    env = decode_environment()
    assert env.image_codec_version is not None
    for lib in _IMAGE_CODEC_LIBS:
        assert lib in env.image_codec_version, (
            f"{lib} missing from image_codec_version: {env.image_codec_version!r} "
            "— adjust fixtures for this build"
        )


# --------------------------------------------------------------------------
# 1. Parser — pure function, defensive, verbatim capture (C2, ratified in
#    docs/R30_BUNDLE_DESIGN.md §2.2)
# --------------------------------------------------------------------------


def test_the_parser_captures_every_library_verbatim():
    info = (
        "  Media I/O: \n"
        "    ZLib:                        build (ver 1.2.13)\n"
        "    JPEG:                        build-libjpeg-turbo (ver 2.1.3-62)\n"
        "      SIMD Support Request:      YES\n"
        "      SIMD Support:              NO\n"
        "    WEBP:                        build (ver encoder: 0x020f)\n"
        "    PNG:                         build (ver 1.6.37)\n"
        "    TIFF:                        build (ver 42 - 4.2.0)\n"
    )
    codecs, simd = _parse_image_codecs(info)
    assert codecs == "JPEG 2.1.3-62 / PNG 1.6.37 / WEBP encoder: 0x020f"
    assert simd == "NO"


def test_the_parser_captures_a_multi_token_ver_value_verbatim():
    """A WebP line with both a decoder and an encoder token — no component
    parsing, the whole parenthesised value is one opaque string (R30 §2.1)."""
    info = "    WEBP:                        build (ver decoder: 0x0209, encoder: 0x020f)\n"
    codecs, _ = _parse_image_codecs(info)
    assert codecs == "WEBP decoder: 0x0209, encoder: 0x020f"


def test_the_parser_is_defensive_on_an_empty_or_absent_block():
    codecs, simd = _parse_image_codecs("")
    assert codecs is None
    assert simd == "unknown"

    codecs, simd = _parse_image_codecs("Some other build info with no Media I/O block at all\n")
    assert codecs is None
    assert simd == "unknown"


def test_the_parser_tolerates_a_missing_simd_line():
    info = "    JPEG:                        build-libjpeg-turbo (ver 2.1.3-62)\n"
    codecs, simd = _parse_image_codecs(info)
    assert codecs == "JPEG 2.1.3-62"
    assert simd == "unknown"


def test_the_parser_is_not_cached_but_the_facts_function_is():
    """PA7: `_parse_image_codecs` is driven directly by the tests above with no
    `lru_cache` in the way; `_image_codec_facts` is the cached, `cv2`-reading
    wrapper — the `_build_facts` pattern."""
    assert not hasattr(_parse_image_codecs, "cache_clear")
    assert hasattr(_image_codec_facts, "cache_clear")


def test_the_image_codec_libs_are_exactly_the_codecs_behind_allowed_image_mime():
    """PA2. If `ALLOWED_IMAGE_MIME` ever widens (e.g. TIFF), this fails — a
    deliberate tripwire, not a coincidence."""
    assert _IMAGE_CODEC_LIBS == ("JPEG", "PNG", "WEBP")
    assert ALLOWED_IMAGE_MIME == ("image/jpeg", "image/png", "image/webp")


# --------------------------------------------------------------------------
# 2. `_build_facts` is untouched (the sibling-function decision, R30 §3.1)
# --------------------------------------------------------------------------


def test_build_facts_is_byte_for_byte_unchanged():
    """The reason a sibling function was chosen over editing `_build_facts`:
    `tests/test_frame_count_reproducibility.py::test_the_decode_identity_now_
    covers_the_demuxer_batch_20` pins `"_VIDEO_FFMPEG_LIBS" in
    inspect.getsource(_build_facts)`. This batch must not be the one that
    breaks that pin."""
    src = inspect.getsource(_build_facts)
    assert "_VIDEO_FFMPEG_LIBS" in src
    assert "image_codec" not in src, (
        "_build_facts should not know about image codecs — that is "
        "_image_codec_facts's job (R30 §3.1)"
    )


# --------------------------------------------------------------------------
# 3. Identity membership — pass-after matrix
# --------------------------------------------------------------------------


def test_a_libjpeg_turbo_difference_now_moves_the_identity():
    env, other = mismatched_image_codec("JPEG", "2.1.3-62", "3.1.4")
    assert other.identity != env.identity
    assert LegacyOpencvProxyIdentity.identity_of(env) == LegacyOpencvProxyIdentity.identity_of(other)


def test_a_libpng_difference_now_moves_the_identity():
    env, other = mismatched_image_codec("PNG", "1.6.37", "1.6.40")
    assert other.identity != env.identity
    assert LegacyOpencvProxyIdentity.identity_of(env) == LegacyOpencvProxyIdentity.identity_of(other)


def test_a_libwebp_difference_now_moves_the_identity():
    env, other = mismatched_image_codec("WEBP", "encoder: 0x020f", "encoder: 0x0210")
    assert other.identity != env.identity
    assert LegacyOpencvProxyIdentity.identity_of(env) == LegacyOpencvProxyIdentity.identity_of(other)


def test_the_simd_flag_is_in_the_digest_not_the_identity():
    env = decode_environment()
    other = replace(env, image_codec_simd="YES" if env.image_codec_simd != "YES" else "NO")
    assert other.identity == env.identity, "SIMD flag leaked into the cache key"
    assert other.digest != env.digest, "the forensic digest must still see it"


def test_zlib_is_recorded_nowhere():
    """zlib is excluded from both `identity` and `digest` (R30 §2/§3.1) —
    unlike the SIMD flag, it has no field at all. (The property docstrings
    discuss *why* zlib is excluded in prose, so this checks the dataclass's
    actual field/dict surface, not the source text.)"""
    from dataclasses import fields

    env = decode_environment()
    assert "zlib" not in env.to_dict()
    assert not any("zlib" in f.name.lower() for f in fields(env))


# --------------------------------------------------------------------------
# 4. The headline: four protections, fail-before / pass-after
# --------------------------------------------------------------------------


def test_a_libjpeg_turbo_difference_now_separates_all_four_protections():
    """**Fail-before / pass-after**, the B20 pattern applied to the image path."""
    env, other = mismatched_image_codec("JPEG", "2.1.3-62", "3.1.4")

    after = _four_protections_separate(env.identity, other.identity)
    assert all(after.values()), f"a protection is still blind after Batch 26: {after}"

    before = _four_protections_separate(
        LegacyOpencvProxyIdentity.identity_of(env),
        LegacyOpencvProxyIdentity.identity_of(other),
    )
    assert not any(before.values()), (
        f"the pre-Batch-26 identity is not blind to the codec change — the fixture is wrong: {before}"
    )


def test_the_benchmark_gate_refuses_across_an_image_codec_difference():
    env, other = mismatched_image_codec("JPEG", "2.1.3-62", "3.1.4")
    a = _bench(env.identity)
    b = _bench(other.identity)
    gate = regression_gate(a, b)
    assert not gate.passed
    assert any("decode" in f.lower() for f in gate.failures), gate.failures


def test_the_calibration_witness_binds_the_finer_identity():
    env, other = mismatched_image_codec("JPEG", "2.1.3-62", "3.1.4")
    wa = _witness(decode_identity=env.identity)
    wb = _witness(decode_identity=other.identity)
    assert wa.witness != wb.witness, "two image codec versions produced one witness digest"


# --------------------------------------------------------------------------
# 5. `decode_identity()` cannot reach a pixel — structural guarantee
# --------------------------------------------------------------------------


def test_decode_identity_is_consumed_only_to_compose_the_version_string():
    """The whole reason this batch cannot move a pixel: `decode_identity()`'s
    only call site in each pipeline is inside `compose_runtime_version`, never
    passed to `decode`, `crop_face`, `resize`, or the backend's `score`."""
    for module in (ip, vp):
        src = inspect.getsource(module)
        count = src.count("decode_identity()")
        assert count == 1, (
            f"{module.__name__} calls decode_identity() {count} times; expected "
            "exactly one, inside compose_runtime_version"
        )
        call_line = next(line for line in src.splitlines() if "decode_identity()" in line)
        assert "compose_runtime_version" in call_line or "decode=decode_identity()" in call_line


def test_the_shipped_image_bundle_did_not_move():
    """A synthetic (faceless, random-noise) image through the real pipeline:
    only `bundle.model.version`'s `.d.` tail should differ from any pre-batch
    run — nothing upstream of `compose_runtime_version` reads `decode_identity`."""
    data = _encode_png(_canvas())
    bundle = _analyse_image(data)
    assert bundle.model.version.endswith(f"{DECODE_MARKER}{decode_identity()}")
    assert bundle.model.version.startswith(MODEL_ID + DECODE_MARKER) or MODEL_ID in bundle.model.version


def test_the_shipped_video_bundle_did_not_move():
    """Same proof, video path."""
    clip = _noise_clip()
    bundle = _analyse_video(clip)
    assert bundle.model.version.endswith(f"{DECODE_MARKER}{decode_identity()}")
    assert MODEL_ID in bundle.model.version


def test_the_decoded_pixels_are_byte_identical():
    """OI3: this batch adds a fingerprint key, touches no pixel. `pre.decode`
    (the actual image-bytes-to-array function) does not import `decode_env` —
    grepped, not assumed."""
    from inference import preprocess as pre

    assert "decode_env" not in inspect.getsource(pre.decode_image)
    img = _canvas()
    data = _encode_png(img)
    decoded = pre.decode_image(data)
    again = pre.decode_image(data)
    assert np.array_equal(decoded, again)
    assert hashlib.sha256(decoded.tobytes()).hexdigest() == hashlib.sha256(again.tobytes()).hexdigest()


# --------------------------------------------------------------------------
# 6. Migration — moved exactly what it should, nothing else
# --------------------------------------------------------------------------


def test_the_decode_segment_moved_and_the_literal_was_updated():
    from backend.config import STUB_ENGINE_VERSION
    from workers.engine_registry import verify_version_alignment

    assert DECODE_MARKER in STUB_ENGINE_VERSION
    assert STUB_ENGINE_VERSION.endswith(f"{DECODE_MARKER}{decode_identity()}")
    assert STUB_ENGINE_VERSION.endswith(".d.c9d4025c"), STUB_ENGINE_VERSION
    verify_version_alignment(STUB_ENGINE_VERSION)  # raises if the literal is stale


def test_no_contract_schema_witness_or_lock_bump():
    assert CONTRACT_VERSION == "1.0.0"
    assert RUN_SCHEMA_VERSION == "1.2.0"
    assert CALIBRATION_WITNESS_VERSION == "1.3.0"

    from evaluation.split_lock import LOCK_VERSION

    assert LOCK_VERSION == "1.1.0"


def test_the_openapi_snapshot_schema_out_and_fixtures_out_are_decode_independent():
    snap = (ROOT / "backend/openapi.snapshot.json").read_text(encoding="utf-8")
    assert "a97ccb68" not in snap and "c9d4025c" not in snap and ".d." not in snap

    schema_files = list((ROOT / "schema_out").glob("*.json"))
    assert len(schema_files) >= 10
    for f in schema_files:
        assert "stub-0" not in f.read_text(encoding="utf-8")

    fixtures_files = list((ROOT / "fixtures_out").glob("*.json"))
    assert len(fixtures_files) >= 1
    for f in fixtures_files:
        assert "a97ccb68" not in f.read_text(encoding="utf-8")
        assert "c9d4025c" not in f.read_text(encoding="utf-8")


def test_an_old_benchmark_row_and_witness_still_load_and_a_cross_boundary_comparison_refuses():
    """B20/B23 precedent: no schema bump means the *shape* never breaks; a
    cross-`.d.` comparison is refused (conservative false incomparability,
    not a correctness gain — R30 design §6.3/§7.3)."""
    old = _bench("a97ccb68")
    new = _bench(decode_identity())
    gate = regression_gate(old, new)
    assert not gate.passed
    assert any("decode" in f.lower() for f in gate.failures), gate.failures

    w_old = _witness(decode_identity="a97ccb68")
    w_new = _witness(decode_identity=decode_identity())
    # both still construct and verify against their own recorded witness —
    # loading an old row never raises.
    assert w_old.witness and w_new.witness
    assert w_old.witness != w_new.witness


def test_the_video_decode_path_is_unchanged():
    """Non-movement proof: the FFmpeg component of the identity, and the
    demuxer/scaler protections B20 (R27) built, are untouched by this batch."""
    from inference.decode_env import _VIDEO_FFMPEG_LIBS

    env = decode_environment()
    assert _VIDEO_FFMPEG_LIBS == ("avcodec", "avformat", "avutil", "swscale")
    assert env.ffmpeg_version and "avformat" in env.ffmpeg_version and "swscale" in env.ffmpeg_version


def test_a_decode_environment_round_trips_with_the_new_fields():
    env = decode_environment()
    payload = env.to_dict()
    assert payload["image_codec_version"] == env.image_codec_version
    assert payload["image_codec_simd"] == env.image_codec_simd
    rebuilt_fields = {k: v for k, v in payload.items() if k not in ("identity", "digest")}
    from inference.decode_env import DecodeEnvironment

    rebuilt = DecodeEnvironment(**rebuilt_fields)
    assert rebuilt.identity == env.identity
    assert rebuilt.digest == env.digest
    assert rebuilt.differences(env) == []


def test_the_forensic_fields_now_include_the_image_codec():
    env = decode_environment()
    payload = env.to_dict()
    for field in ("image_codec_version", "image_codec_simd"):
        assert field in payload, f"{field} is not recoverable for forensics"
    assert decode_fingerprint() == env.digest


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
