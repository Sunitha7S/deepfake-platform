"""R27 — is `decode_identity()` complete for the video decode path? Standalone.

    python tests/decode_identity_completeness_study.py

Produces every number in `docs/DECODE_IDENTITY_COMPLETENESS.md`.

THE QUESTION
------------
Batch 9 put a decode-environment digest in the result-cache key; Batch 11 put it
on `BenchmarkRun` and inside `CalibrationWitness`. Batch 19 found one hole:
`CAP_PROP_FRAME_COUNT` is a libavformat quantity and the digest captured only
libavcodec. This study re-derives the completeness question from scratch:

* E1  Enumerate every component of the current video decode path and what it
      can change (pixels / counts / timestamps / ordering / seeking / resize /
      colour / metadata).
* E2  What `getBuildInformation()` actually exposes.
* E3  What `decode_identity()` digests — before and after this batch.
* E4  The reproducibility hole across all four protections (cache key, benchmark
      identity, calibration identity, deployment identity), fail-before /
      pass-after.
* E5  Is libavformat sufficient? libswscale on the pixel path.
* E6  avresample — excluded, and why (audio only).
* E7  Cache-fragmentation impact of the expansion.
* E8  The opencv-python wheel-version gap (`cv2.__version__` drops the 4th digit).

WHERE REAL ENVIRONMENTS CANNOT BE REPRODUCED
-------------------------------------------
This box has one OpenCV build and one FFmpeg. A genuine "different libavformat"
cannot be produced, so E4's cross-build case uses `replace()` on a
`DecodeEnvironment` to construct the two, and `LegacyAvcodecOnlyIdentity` to show
what the pre-batch composition did with them.
"""

from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from contracts.version import compose_runtime_version, fingerprint  # noqa: E402
from inference import decode_env as de  # noqa: E402
from inference.decode_env import _build_facts, decode_environment, decode_identity  # noqa: E402


def head(n: int, title: str) -> None:
    print(f"\n{'=' * 76}\nE{n}. {title}\n{'-' * 76}")


class LegacyAvcodecOnlyIdentity:
    """`DecodeEnvironment.identity` as of Batch 9-19: FFmpeg = avcodec only."""

    name = "decode identity, pre-Batch-20 (avcodec version only)"

    @staticmethod
    def ffmpeg_of(env) -> str | None:
        """The pre-Batch-20 value: the bare libavcodec version, no ``avcodec `` prefix."""
        s = env.ffmpeg_version or ""
        for part in s.split(" / "):
            if part.startswith("avcodec "):
                return part[len("avcodec "):]
        return s or None

    @classmethod
    def identity_of(cls, env) -> str:
        return fingerprint({
            "opencv": env.opencv_version,
            "backend": env.video_backend,
            "ffmpeg": cls.ffmpeg_of(env),
            "system": env.platform_system,
            "machine": env.platform_machine,
        })


# --------------------------------------------------------------------------
# E1 — the video decode path, component by component
# --------------------------------------------------------------------------


def e1_component_enumeration() -> None:
    head(1, "Every component on the current video decode path")
    rows = [
        ("temp_media_file (spool bytes to disk)", "nothing - byte-exact copy", "n/a"),
        ("resolve_video_backend -> VideoCapture(path, FFMPEG)", "which demuxer+decoder", "video_backend [in]"),
        ("libavformat (demuxer)", "frame_count, fps, POS_MSEC, packet order, seeking, rotation/SAR", "was MISSING; B20 [in]"),
        ("libavcodec (frame decoder)", "decoded YUV pixels, coded width/height", "ffmpeg_version [in]"),
        ("libswscale (YUV->BGR in retrieve())", "the BGR pixels OpenCV returns; chroma upsampling", "was MISSING; B20 [in]"),
        ("libavutil (pixfmt descriptors, rational ts math)", "timestamp rounding, pixfmt", "was MISSING; B20 [in]"),
        ("OpenCV cap_ffmpeg wrapper (cap_ffmpeg_impl.hpp)", "POS_MSEC derivation, EOF handling, retrieve", "opencv_version [in]"),
        ("sample_frame_indices(frame_count, n)", "which indices are scored", "pure arithmetic - not an env component"),
        ("cv2.cvtColor BGR->RGB (to_rgb)", "channel order", "opencv_version [in] (CPU-independent, B8 E6)"),
        ("cv2.resize INTER_AREA/INTER_LINEAR", "the final square frame", "opencv_version [in] (CPU-independent, B8 E6)"),
        ("Intel IPP (used by resize/cvtColor on x86)", "resize/cvt kernels", "via opencv_version (wheel-pinned); not measured"),
        ("np.ndarray.astype(float32)", "widening", "deterministic - elementwise"),
        ("OS / architecture", "which prebuilt FFmpeg, which kernels", "platform_system/machine [in]"),
        ("CPU baseline / dispatch (SIMD)", "opencv kernels (B8 E6: not a pixel); swscale kernels NOT measured", "digest only, not identity"),
        ("DFP_VIDEO_BACKEND / OPENCV_VIDEOIO_PRIORITY_*", "backend choice", "neutralised - backend is named (B8)"),
    ]
    print(f"  {'component':52s} {'affects':46s} in decode_identity?")
    for c, a, i in rows:
        print(f"  {c:52s} {a:46s} {i}")


# --------------------------------------------------------------------------
# E2 — what getBuildInformation exposes
# --------------------------------------------------------------------------


def e2_build_information() -> None:
    head(2, "What getBuildInformation() exposes for the video path")
    info = cv2.getBuildInformation()
    for key in ("avcodec:", "avformat:", "avutil:", "swscale:", "avresample:",
                "FFMPEG:", "Media Foundation:", "Intel IPP:", "OpenCL:"):
        line = next((l.strip() for l in info.splitlines() if l.strip().startswith(key)), "(absent)")
        print(f"  {line}")
    print(f"\n  cv2.__version__                = {cv2.__version__!r}")
    try:
        import importlib.metadata as md
        for pkg in ("opencv-python", "opencv-python-headless", "opencv-contrib-python"):
            try:
                print(f"  importlib.metadata['{pkg}']  = {md.version(pkg)!r}")
            except md.PackageNotFoundError:
                pass
    except Exception as exc:  # noqa: BLE001
        print(f"  (wheel version unavailable: {exc})")


# --------------------------------------------------------------------------
# E3 — what decode_identity digests, before and after
# --------------------------------------------------------------------------


def e3_identity_membership() -> None:
    head(3, "decode_identity() membership — before vs after this batch")
    env = decode_environment()
    ffmpeg_libs, _, _ = _build_facts()
    print(f"  _build_facts() ffmpeg component (AFTER)  : {ffmpeg_libs!r}")
    print(f"  avcodec-only component (BEFORE)          : {LegacyAvcodecOnlyIdentity.ffmpeg_of(env)!r}")
    print(f"\n  decode_identity()  BEFORE (avcodec only) : {LegacyAvcodecOnlyIdentity.identity_of(env)}")
    print(f"  decode_identity()  AFTER  (all libs)     : {env.identity}")
    print(f"  digest (forensic, AFTER)                 : {env.digest}")
    print(f"\n  runtime version AFTER: {compose_runtime_version(model='stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387', decode=env.identity)}")


# --------------------------------------------------------------------------
# E4 — the hole across all four protections
# --------------------------------------------------------------------------


def _four_protections(id_a: str, id_b: str) -> dict[str, bool]:
    """True = the protection SEPARATES the two environments (good)."""
    MODEL = "stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387"
    rt_a = compose_runtime_version(model=MODEL, decode=id_a)
    rt_b = compose_runtime_version(model=MODEL, decode=id_b)
    return {
        "content cache key (sha256 + runtime version)": rt_a != rt_b,
        "deployment identity (/health, alignment)": rt_a != rt_b,
        "benchmark identity (regression_gate refuses)": id_a != id_b,
        "calibration identity (witness digest differs)": id_a != id_b,
    }


def e4_hole_across_protections() -> None:
    head(4, "The hole across all four protections — fail-before / pass-after")
    env = decode_environment()
    # "a different libavformat" — same everything else
    assert env.ffmpeg_version and "avformat" in env.ffmpeg_version
    other = replace(
        env,
        ffmpeg_version=env.ffmpeg_version.replace("avformat 58.76.100", "avformat 59.16.100"),
    )

    print("  Scenario: two deployments, identical except libavformat 58.76.100 vs 59.16.100")
    print("  (a routine distro-vs-prebuilt difference). The demuxers disagree on")
    print("  frame_count for a VFR clip -> different frames scored -> different p_fake.\n")

    print("  BEFORE Batch 20 (decode_identity = avcodec only):")
    before = _four_protections(
        LegacyAvcodecOnlyIdentity.identity_of(env), LegacyAvcodecOnlyIdentity.identity_of(other)
    )
    for k, sep in before.items():
        print(f"    {'SEPARATED' if sep else 'BLIND    '}  {k}")

    print("\n  AFTER Batch 20 (decode_identity = every FFmpeg lib on the video path):")
    after = _four_protections(env.identity, other.identity)
    for k, sep in after.items():
        print(f"    {'SEPARATED' if sep else 'BLIND    '}  {k}")

    n_blind_before = sum(1 for v in before.values() if not v)
    n_blind_after = sum(1 for v in after.values() if not v)
    print(f"\n  protections blind: {n_blind_before}/4 -> {n_blind_after}/4")


# --------------------------------------------------------------------------
# E5 — libswscale on the pixel path
# --------------------------------------------------------------------------


def e5_swscale_on_the_path() -> None:
    head(5, "Is libavformat sufficient? libswscale on the video pixel path")
    info = cv2.getBuildInformation()
    sws = next((l.strip() for l in info.splitlines() if l.strip().startswith("swscale:")), "(absent)")
    print(f"  {sws}   <- linked under Video I/O -> FFMPEG")
    print("  OpenCV's CvCapture_FFMPEG::retrieveFrame converts the decoded frame")
    print("  (native pixfmt, YUV420P for H.264/mp4v) to BGR24 via sws_scale.")
    print("  So the bytes retrieve() hands back pass through libswscale.")
    print()
    print("  Evidence it matters (decode_study.py E3/E4, already measured):")
    print("    FFMPEG vs MSMF: a near-identity colour matrix + per-channel bias ~-1.2")
    print("    AND a non-affine residual mean 2.01 -> 'chroma upsampling differs too'.")
    print("    Chroma upsampling is exactly libswscale's job. A swscale version that")
    print("    changed its default chroma filter would move decoded pixels the same way.")
    print()
    print("  VERDICT: libavformat is necessary (frame selection) but NOT sufficient.")
    print("  libswscale is on the pixel path and belongs with it. libavutil is the")
    print("  shared base (pixfmt descriptors, rational ts math) and moves on the same")
    print("  FFmpeg release; included at the cost of one more parsed line.")


# --------------------------------------------------------------------------
# E6 — avresample excluded
# --------------------------------------------------------------------------


def e6_avresample_excluded() -> None:
    head(6, "avresample — excluded, and why")
    root = Path(__file__).resolve().parent.parent
    audio_hits = []
    for p in (root / "inference").rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            for kw in ("swr_", "swresample", "CAP_PROP_AUDIO", ".read_audio", "audio_stream"):
                if kw in code:
                    audio_hits.append(f"{p.name}:{i}:{kw}")
    print(f"  audio-API usage anywhere in inference/ : {audio_hits or 'none'}")
    print("  libavresample is FFmpeg's audio sample-rate converter, deprecated in")
    print("  FFmpeg 4 and removed in FFmpeg 5. This codebase decodes video frames")
    print("  only. It is deliberately NOT in _VIDEO_FFMPEG_LIBS.")


# --------------------------------------------------------------------------
# E7 — fragmentation impact
# --------------------------------------------------------------------------


def e7_fragmentation() -> None:
    head(7, "Cache-fragmentation cost of the expansion")
    env = decode_environment()
    print("  Homogeneous fleet (one FFmpeg build): the 4 lib versions move together")
    print("  on an FFmpeg upgrade, so the number of cache namespaces is unchanged:")
    print(f"    1 namespace before, 1 after   (identity now {env.identity})")
    print()
    print("  The expansion fragments MORE in exactly one case: a Frankenbuild that")
    print("  mixes FFmpeg library versions (distro libavcodec + vendored libswscale,")
    print("  say). That is the case we WANT to separate, because those decode")
    print("  differently. So there is no fragmentation cost that is not a correctness")
    print("  gain — unlike keying on the CPU fields, which E6 measured as pure loss.")


# --------------------------------------------------------------------------
# E8 — the wheel-version gap
# --------------------------------------------------------------------------


def e8_wheel_gap() -> None:
    head(8, "The opencv-python wheel-version gap (a residual, not closed here)")
    print(f"  cv2.__version__ = {cv2.__version__!r}")
    try:
        import importlib.metadata as md
        wheel = md.version("opencv-python")
        print(f"  opencv-python wheel = {wheel!r}")
        if wheel.rsplit(".", 1)[0] != cv2.__version__ and not wheel.startswith(cv2.__version__ + "."):
            print("  -> versions unrelated")
        else:
            dropped = wheel[len(cv2.__version__):].lstrip(".")
            print(f"  -> cv2.__version__ drops the wheel's 4th component ({dropped!r}).")
            print("     Two opencv-python builds 4.8.1.78 and 4.8.1.86 both report")
            print("     opencv_version='4.8.1'. If they bundle different FFmpeg, the new")
            print("     ffmpeg_version field catches it directly. If they differ only in")
            print("     OpenCV's own imgproc (resize/cvtColor) between wheel patches,")
            print("     that is NOT caught — a residual, low-risk (E6: those ops are")
            print("     dispatch-independent and imgproc is stable across patch releases).")
    except Exception as exc:  # noqa: BLE001
        print(f"  (wheel metadata unavailable: {exc} — common for distro/conda builds,")
        print("   which is itself the reason to fingerprint the FFmpeg libs directly)")


def main() -> int:
    print("R27 — DECODE IDENTITY COMPLETENESS")
    print(f"env: {decode_environment().describe()}")
    for fn in (e1_component_enumeration, e2_build_information, e3_identity_membership,
               e4_hole_across_protections, e5_swscale_on_the_path, e6_avresample_excluded,
               e7_fragmentation, e8_wheel_gap):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"\n  !! {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
