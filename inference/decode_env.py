"""Identity of the decoder that produced the pixels.

WHY THIS EXISTS
---------------
Batch 8 measured it: the same bytes, decoded by two backends available in one
OpenCV build on one machine, produce **no identical frames**. Mean absolute
difference 2.2/255 on photographic content, 12.6 on noise, 85-96% of pixels
changed. The two differ by a near-identity colour matrix with a per-channel bias
of about -1.2 plus a non-affine residual, which is a levels/chroma
interpretation difference, not float noise.

Nothing in the platform recorded which decoder ran. ``effective_model_version``
composes weights, calibration and the quality gate; the decoder is upstream of
all three and appears in none of them. Two deployments could therefore produce
different verdicts for the same file, same model, same calibration, and agree
byte-for-byte on every version string in the response — measured, and the reason
this module exists.

WHAT IT DOES AND DOES NOT CLAIM
-------------------------------
This makes the decoder **detectable**, not identical. A fingerprint cannot make
two libavcodec builds agree on a pixel; it can only ensure that when they
disagree, the two results are distinguishable rather than silently pooled. That
is the honest guarantee, and it is the same shape as the one the split lock
makes about corpora.

**Still deliberately NOT in ``calibration_fingerprint``.** That fingerprint
identifies a *calibration*, and the decoder is not one. Folding it in would make
"the temperature changed" and "the worker was rebuilt" the same event, which is
the conflation Batch 9 exists to avoid. The decoder gets its own segment of the
runtime version instead — ``contracts.version.DECODE_MARKER``.

**Batch 8 said this could not reach the API's cache key. That was wrong.**
The reasoning was: ``backend`` must stay free of ``inference`` (Batch 0
invariant 3), ``cv2.getBuildInformation`` is unavailable in the API process, and
so a fingerprint only the worker can compute cannot be part of a key the API
must build.

Every premise is true and the conclusion does not follow, because the API does
not compute ``Settings.model_version`` either — it reads a configured literal
that ``workers.engine_registry.verify_version_alignment`` checks against what the
engines actually report, failing startup on a mismatch. That mechanism was built
in Batch 0 for exactly this shape of problem, and the decode identity travels by
it. Batch 9 does what Batch 8 said was impossible, using machinery Batch 8 had
already described two paragraphs earlier.

TWO DIGESTS, AND THE DIFFERENCE MATTERS
---------------------------------------
:attr:`DecodeEnvironment.identity` is the cache key segment; :attr:`~
DecodeEnvironment.digest` is the forensic record. ``identity`` excludes the CPU
fields because Batch 8 measured them not to change a decoded pixel, and keying a
mixed-hardware fleet on them would cost hit rate for no correctness. Each
property's docstring states its own membership and why.
"""

from __future__ import annotations

import platform
import re
import sys
from dataclasses import dataclass
from functools import lru_cache

import cv2

from contracts.version import fingerprint

VIDEO_BACKEND_ENV = "DFP_VIDEO_BACKEND"
"""Environment variable naming the video backend to use, e.g. ``FFMPEG``.

Exists so a deployment can *state* its decoder rather than inherit whichever one
OpenCV's priority table happens to rank first. Unset means
:data:`PREFERRED_VIDEO_BACKENDS`.
"""

PREFERRED_VIDEO_BACKENDS: tuple[str, ...] = ("FFMPEG",)
"""Backends to try, in order, before falling back to OpenCV's own choice.

FFMPEG first because it is the one backend present on every platform OpenCV
ships for, so pinning it is the choice most likely to produce the same pixels on
a Linux worker and a Windows laptop. It is **not** first because it is more
correct than MSMF — measured, neither is; they simply disagree.

Why pin at all: ``cv2.VideoCapture(path)`` consults a priority table that
``OPENCV_VIDEOIO_PRIORITY_<BACKEND>`` overrides at runtime. Measured on one
machine, one build, one file: setting ``OPENCV_VIDEOIO_PRIORITY_MSMF=9999``
switches the decoder and changes every decoded frame. No rebuild, no code
change, no record anywhere. Naming the backend removes an environment variable
from the set of things that can silently change a verdict.
"""


def _backend_id(name: str) -> int | None:
    """Numeric id for a backend name, or ``None`` if this build lacks it."""
    for backend in cv2.videoio_registry.getBackends():
        if cv2.videoio_registry.getBackendName(backend) == name:
            return int(backend)
    return None


@lru_cache(maxsize=1)
def resolve_video_backend() -> tuple[int, str, str]:
    """``(api_preference, resolved_name, how)`` for opening video.

    ``how`` is one of ``env``, ``preferred`` or ``fallback`` and is recorded
    rather than inferred, because "we asked for FFMPEG and got it" and "FFMPEG
    was missing so OpenCV chose" are different facts about a run and the pixels
    may differ between them.

    Cached: the answer cannot change within a process, and this is called once
    per media file.
    """
    import os

    requested = os.environ.get(VIDEO_BACKEND_ENV, "").strip().upper()
    if requested:
        found = _backend_id(requested)
        if found is not None:
            return found, requested, "env"
        # A named backend that this build does not have is a configuration
        # error worth surfacing, but not worth refusing to decode over — the
        # fallback is recorded and the fingerprint changes, which is the point.
        return cv2.CAP_ANY, _default_backend_name(), "fallback"

    for name in PREFERRED_VIDEO_BACKENDS:
        found = _backend_id(name)
        if found is not None:
            return found, name, "preferred"
    return cv2.CAP_ANY, _default_backend_name(), "fallback"


def _default_backend_name() -> str:
    """What ``CAP_ANY`` resolves to, without opening a file.

    Best effort: the registry's stream backends are listed in priority order, so
    the first is what ``CAP_ANY`` would pick. Reported as ``UNKNOWN`` rather
    than guessed if the registry is empty.
    """
    try:
        backends = cv2.videoio_registry.getStreamBackends()
    except Exception:  # pragma: no cover - registry unavailable
        return "UNKNOWN"
    if not backends:  # pragma: no cover - no video support at all
        return "UNKNOWN"
    return cv2.videoio_registry.getBackendName(backends[0])


def open_video(path) -> cv2.VideoCapture:
    """Open a video through the resolved backend.

    The single place a ``VideoCapture`` is constructed in this package. Two call
    sites used to construct their own with no ``apiPreference``, which meant the
    probe and the decode could in principle resolve differently — and it meant
    the decoder in use was whatever the environment said, unrecorded.
    """
    api, _, _ = resolve_video_backend()
    return cv2.VideoCapture(str(path), api)


@dataclass(frozen=True)
class DecodeEnvironment:
    """Everything about this process that can change a decoded pixel.

    Fields are chosen by what was *measured* to matter, plus what is cheap to
    record and would be wanted at the moment a discrepancy shows up. The CPU
    baseline is here despite the SIMD hypothesis failing to reproduce (see the
    design report): a negative result on one build is not a guarantee for every
    build, and the field costs nothing.
    """

    opencv_version: str
    video_backend: str
    backend_source: str
    ffmpeg_version: str | None
    platform_system: str
    platform_machine: str
    python_version: str
    cpu_baseline: str
    cpu_dispatch: str
    image_codec_version: str | None
    image_codec_simd: str

    @property
    def identity(self) -> str:
        """**The cache-affecting digest.** Equal identities, same decoded pixels.

        This is the segment that goes into the runtime version and therefore
        into the result cache key, so its membership is a deliberate trade
        rather than "everything we know".

        IN, because each was measured or is structurally certain to change
        pixels:

        * ``opencv_version`` — the decode implementation itself.
        * ``video_backend`` — Batch 8 E3: FFMPEG and MSMF share *zero* identical
          frames on the same file, in every codec tested.
        * ``ffmpeg_version`` — **every FFmpeg library the video path links**
          (:data:`_VIDEO_FFMPEG_LIBS`): libavcodec decodes the frame,
          **libavformat** produces ``CAP_PROP_FRAME_COUNT`` and so decides which
          frames are sampled (R27, Batch 19/20 — measured: two demuxers off by
          one frame moved 11/16 sampled indices while this string was
          avcodec-only and identical), **libswscale** converts YUV to BGR in
          ``retrieve()`` (Batch 8 E4 measured a chroma-upsampling difference at
          that stage), and libavutil carries the pixfmt and timestamp
          arithmetic. Before Batch 20 this was libavcodec alone.
        * ``platform_system`` / ``platform_machine`` — different OS builds and
          different architectures compile different kernels, and an
          x86-vs-ARM difference cannot be measured from one machine. Included on
          the conservative side: for a *cache* key a false miss costs compute
          and a false hit costs a wrong verdict, so the two errors are not
          symmetric and the tie breaks toward missing.
        * ``image_codec_version`` — **R30.** libjpeg-turbo / libpng / libwebp,
          the image-decode analogue of ``ffmpeg_version``. Measured (R30
          design, `docs/R30_BUNDLE_DESIGN.md` §1.4/§2, and the earlier
          cross-build study in `docs/B24_R2_DESIGN.md` §1) to be pixel-neutral
          for every accepted MIME type across a major libjpeg-turbo version
          gap, a libpng change, and stock zlib vs. zlib-ng — included anyway
          as the coarsest reliable signal, on the same over-sensitive-by-design
          reasoning as ``ffmpeg_version``: a false miss costs compute, a false
          hit costs a wrong verdict.

        OUT, because including them would fragment the cache on things measured
        *not* to change a pixel:

        * ``cpu_baseline`` / ``cpu_dispatch`` — Batch 8 E6 ran ``resize``,
          ``to_rgb`` and ``imdecode`` under ``OPENCV_CPU_DISABLE`` in a
          subprocess and got byte-identical output. Two machines differing only
          in AVX support decode identically, so keying on that would halve the
          hit rate of a mixed fleet for nothing. Still recorded in
          :attr:`digest` for forensics, and pinned by a test, so if a future
          build does diverge the negative result fails loudly instead of
          silently becoming wrong.
        * ``image_codec_simd`` — **R30.** libjpeg-turbo's SIMD flag. OpenCV's
          JPEG loader uses ``JDCT_ISLOW`` (accurate integer IDCT), which
          libjpeg-turbo holds bit-identical between its C and SIMD paths;
          measured via a cross-build check (SIMD-off OpenCV vs. SIMD-on
          Pillow, three JPEGs, identical). Performance-only, like
          ``cpu_dispatch`` — same treatment, same reasoning.
        * ``backend_source`` — how the backend was chosen. Provenance; the same
          backend decodes the same whether it was named or inherited.
        * ``python_version`` — does not touch a decoded pixel.

        zlib is not recorded at all: its only role on the image path is
        ``inflate()`` for PNG, which is RFC-1951-deterministic — measured
        identical between stock zlib and zlib-ng — so it carries no pixel risk
        and no forensic value the libpng version does not already cover.
        """
        return fingerprint(
            {
                "opencv": self.opencv_version,
                "backend": self.video_backend,
                "ffmpeg": self.ffmpeg_version,
                "system": self.platform_system,
                "machine": self.platform_machine,
                "image_codec": self.image_codec_version,
            }
        )

    @property
    def digest(self) -> str:
        """Short hash of the **whole** environment. The forensic digest.

        Strictly finer than :attr:`identity`: it adds the CPU fields and (R30)
        ``image_codec_simd``, so two environments can share an identity and
        differ here. That is the point — this is what a run record carries so
        a discrepancy can be reconstructed, and it must not be used as a cache
        key or a fleet of mixed CPUs will never hit.

        Equal digests mean equal pixels. The converse is not claimed: two
        environments with different digests may well decode identically. This is
        a change *detector*, and over-sensitivity is the safe direction for a
        detector — it reports "not comparable" where the truth is "probably
        comparable", never the reverse.
        """
        return fingerprint(
            {
                "identity": self.identity,
                "cpu_baseline": self.cpu_baseline,
                "cpu_dispatch": self.cpu_dispatch,
                "image_codec_simd": self.image_codec_simd,
            }
        )

    def to_dict(self) -> dict:
        return {
            "opencv_version": self.opencv_version,
            "video_backend": self.video_backend,
            "backend_source": self.backend_source,
            "ffmpeg_version": self.ffmpeg_version,
            "platform_system": self.platform_system,
            "platform_machine": self.platform_machine,
            "python_version": self.python_version,
            "cpu_baseline": self.cpu_baseline,
            "cpu_dispatch": self.cpu_dispatch,
            "image_codec_version": self.image_codec_version,
            "image_codec_simd": self.image_codec_simd,
            "identity": self.identity,
            "digest": self.digest,
        }

    def describe(self) -> str:
        return (
            f"decode {self.identity}/{self.digest}: OpenCV {self.opencv_version} via "
            f"{self.video_backend} ({self.backend_source}), "
            f"ffmpeg {self.ffmpeg_version or 'unreported'}, "
            f"image codecs {self.image_codec_version or 'unreported'} "
            f"(SIMD {self.image_codec_simd}), "
            f"{self.platform_system}/{self.platform_machine}, "
            f"CPU {self.cpu_baseline} + [{self.cpu_dispatch}]"
        )

    def differences(self, other: DecodeEnvironment) -> list[str]:
        """Field-by-field, so a mismatch says *what* changed.

        ``backend_source`` and ``python_version`` are compared but excluded from
        the digest: how the backend was chosen is provenance rather than
        behaviour, and the Python version does not touch a decoded pixel. They
        are reported here because an auditor looking at two runs wants them.
        """
        out = []
        for field in (
            "opencv_version",
            "video_backend",
            "backend_source",
            "ffmpeg_version",
            "platform_system",
            "platform_machine",
            "python_version",
            "cpu_baseline",
            "cpu_dispatch",
            "image_codec_version",
            "image_codec_simd",
        ):
            mine, theirs = getattr(self, field), getattr(other, field)
            if mine != theirs:
                out.append(f"{field}: {mine!r} != {theirs!r}")
        return out


_FFMPEG_RE = re.compile(r"FFMPEG:\s*YES\s*\(prebuilt binaries\)|FFMPEG:\s*YES")
_AVCODEC_RE = re.compile(r"avcodec:\s*YES\s*\(([^)]+)\)")

# The FFmpeg libraries OpenCV's video path actually links, in the order
# ``getBuildInformation`` lists them under ``Video I/O -> FFMPEG``.  R27 (Batch
# 19/20): the pre-Batch-20 identity captured only ``avcodec`` — the frame
# decoder — and omitted:
#   * ``avformat`` — the demuxer, which produces ``CAP_PROP_FRAME_COUNT`` /
#     ``FPS`` / ``POS_MSEC`` and therefore *which frames are sampled*;
#   * ``swscale`` — the YUV->BGR conversion in ``VideoCapture.retrieve()``
#     (Batch 8 E4 measured a chroma-upsampling difference at that stage);
#   * ``avutil`` — pixel-format descriptors and the rational arithmetic behind
#     timestamps; shared base, moves on the same release.
# ``avresample`` is deliberately absent: it is FFmpeg's *audio* resampler,
# deprecated in FFmpeg 4 and removed in 5, and this codebase decodes no audio.
_VIDEO_FFMPEG_LIBS: tuple[str, ...] = ("avcodec", "avformat", "avutil", "swscale")
_AVLIB_RE = re.compile(
    r"^\s*(" + "|".join(_VIDEO_FFMPEG_LIBS) + r"):\s*YES\s*\(([^)]+)\)", re.M
)

# R30 (`docs/R30_BUNDLE_DESIGN.md`): the image-decode analogue of the FFmpeg
# block above. ``getBuildInformation()`` lists these under "Media I/O", each as
# ``<LIB>:  build[-variant] (ver <value>)``. Captured verbatim inside the
# parentheses — no component parsing (a WebP "encoder: 0x020f" or a
# hypothetical "decoder: 0x0209, encoder: 0x020f" is stored as one opaque
# string) — because the format varies across wheels/distros and the value is
# a change-detector input, not something this codebase interprets.
#
# zlib is deliberately absent: PNG's ``inflate()`` step is RFC-1951-
# deterministic, measured identical between stock zlib and zlib-ng (R30 §2),
# so it carries no pixel risk. TIFF / JPEG 2000 / OpenEXR are absent because
# they are outside ``ALLOWED_IMAGE_MIME`` — unreachable, not merely unlisted.
_IMAGE_CODEC_LIBS: tuple[str, ...] = ("JPEG", "PNG", "WEBP")
_IMAGE_CODEC_RE = re.compile(
    r"^\s*(" + "|".join(_IMAGE_CODEC_LIBS) + r"):\s*\S+\s*\(ver\s+([^)]+)\)", re.M
)
# libjpeg-turbo's SIMD flag, nested under the JPEG line. Digest-only (R30 §2):
# OpenCV's JDCT_ISLOW is bit-exact across C and SIMD, so this is a
# performance fact, not a pixel one — the `cpu_dispatch` precedent.
_SIMD_SUPPORT_RE = re.compile(r"^\s*SIMD Support:\s*(\S+)", re.M)


def _parse_image_codecs(info: str) -> tuple[str | None, str]:
    """``(image_codec_version, image_codec_simd)`` from a build-info string.

    A pure function, deliberately not ``lru_cache``d, so a test can drive it
    directly with a synthetic string (R30's PA7 — the C2 parser-robustness
    item ratified in `docs/R30_BUNDLE_DESIGN.md` §2.2). Defensive like
    :func:`_build_facts`: a library missing from the string is simply absent
    from the joined result rather than raising, since a missing diagnostic
    must not be able to stop a decode.
    """
    found = {lib: ver.strip() for lib, ver in _IMAGE_CODEC_RE.findall(info)}
    codecs = (
        " / ".join(f"{lib} {found[lib]}" for lib in _IMAGE_CODEC_LIBS if lib in found)
        or None
    )
    simd_match = _SIMD_SUPPORT_RE.search(info)
    simd = simd_match.group(1).strip() if simd_match else "unknown"
    return codecs, simd


@lru_cache(maxsize=1)
def _build_facts() -> tuple[str | None, str, str]:
    """``(ffmpeg_libs, cpu_baseline, cpu_dispatch)`` from the OpenCV build.

    ``ffmpeg_libs`` names **every FFmpeg library on the video decode path**, not
    only libavcodec — see :data:`_VIDEO_FFMPEG_LIBS` and R27. It is a single
    ``"lib x.y.z / lib x.y.z / ..."`` string so the existing single
    ``ffmpeg_version`` field and its place in :attr:`DecodeEnvironment.identity`
    are unchanged; only the value gained the demuxer, the scaler and avutil.

    Parsed from ``getBuildInformation`` rather than from a dedicated API because
    OpenCV exposes none. Parsing is defensive: every field falls back to a
    marker string rather than raising, since a missing diagnostic must not be
    able to stop a decode.
    """
    try:
        info = cv2.getBuildInformation()
    except Exception:  # pragma: no cover - no build info available
        return None, "unknown", "unknown"

    found = {lib: ver.strip() for lib, ver in _AVLIB_RE.findall(info)}
    if found:
        # Stable order regardless of how the build string is laid out.
        ffmpeg = " / ".join(f"{lib} {found[lib]}" for lib in _VIDEO_FFMPEG_LIBS if lib in found)
    elif _AVCODEC_RE.search(info):
        # avcodec present but the others unparsed (an OpenCV that lists only it).
        ffmpeg = f"avcodec {_AVCODEC_RE.search(info).group(1).strip()}"
    elif _FFMPEG_RE.search(info):
        ffmpeg = "present, versions unreported"
    else:
        ffmpeg = None

    baseline = "unknown"
    dispatch = "unknown"
    for line in info.splitlines():
        stripped = line.strip()
        if stripped.startswith("Baseline:"):
            baseline = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Dispatched code generation:"):
            dispatch = stripped.split(":", 1)[1].strip()
    return ffmpeg, baseline, dispatch


@lru_cache(maxsize=1)
def _image_codec_facts() -> tuple[str | None, str]:
    """Cached wrapper around :func:`_parse_image_codecs` — the ``_build_facts``
    pattern, kept as a **separate** cache and a **separate** function (R30).

    Not folded into :func:`_build_facts`: that function has 5 call sites
    outside its own definition (`decode_environment`, two studies, and two
    test files), one of which asserts
    ``"_VIDEO_FFMPEG_LIBS" in inspect.getsource(_build_facts)``. A sibling
    function leaves every one of those five untouched by construction.
    """
    try:
        info = cv2.getBuildInformation()
    except Exception:  # pragma: no cover - no build info available
        return None, "unknown"
    return _parse_image_codecs(info)


def decode_environment() -> DecodeEnvironment:
    """Capture the current decode environment. Cheap enough to call per job."""
    ffmpeg, baseline, dispatch = _build_facts()
    image_codec_version, image_codec_simd = _image_codec_facts()
    _, backend, source = resolve_video_backend()
    return DecodeEnvironment(
        opencv_version=cv2.__version__,
        video_backend=backend,
        backend_source=source,
        ffmpeg_version=ffmpeg,
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        python_version=".".join(str(v) for v in sys.version_info[:3]),
        cpu_baseline=baseline,
        cpu_dispatch=dispatch,
        image_codec_version=image_codec_version,
        image_codec_simd=image_codec_simd,
    )


def decode_fingerprint() -> str:
    """Forensic digest of the decode environment. The one-liner for a run record.

    Not the cache key — see :func:`decode_identity`, and
    :attr:`DecodeEnvironment.identity` for why the two differ.
    """
    return decode_environment().digest


def decode_identity() -> str:
    """**The cache-affecting** decode fingerprint, for the runtime version.

    Coarser than :func:`decode_fingerprint` on purpose: it excludes the CPU
    fields, which Batch 8 measured not to change a decoded pixel and which would
    otherwise fragment the result cache across a fleet of mixed hardware for no
    correctness gain.
    """
    return decode_environment().identity
