"""Media decoding and preparation. Bytes in, numpy arrays out.

Three rules shape this module.

**Caps are enforced before allocation, not after.** A decompression bomb is a
small file declaring enormous dimensions, so a byte-length check catches
nothing. ``probe_image_header`` reads width and height out of the container
header — JPEG SOF, PNG IHDR, WebP VP8/VP8L/VP8X, BMP — and the pixel cap is
applied there, before ``cv2.imdecode`` allocates. Residual risk is stated
honestly in that function's docstring: a header that lies still allocates, and
containing that needs a subprocess with a memory ceiling, which is the caller's
job because this package receives bytes and cannot sandbox itself.

**Malformed input raises, it never returns something empty.** ``cv2.imdecode``
answers ``None`` for garbage; that becomes ``UnreadableMedia`` rather than an
empty array that fails confusingly three functions later.

**cv2 speaks BGR; everything downstream speaks RGB.** The conversion happens
once, here, at the decode boundary. A channel-order bug is invisible in tests
that use synthetic grey images and catastrophic on real faces.

**Nothing stateful in this module is shared between threads.** Both pipelines
run their blocking work through ``asyncio.to_thread``, so every function here is
reached from an arbitrary pool thread. The face cascade is the one object with
mutable per-call state, and it is held in a ``threading.local`` for the reason
given at :data:`_cascade_local` — a shared one silently returns wrong answers
rather than raising, which is the worst failure mode available.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import struct
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from contracts.version import fingerprint
from inference.decode_env import open_video
from inference.protocol import UnreadableMedia, UnsupportedMedia

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class MediaCaps:
    """Limits applied before decode. Configurable because deployments differ."""

    max_bytes: int = 200 * 1024 * 1024
    max_pixels: int = 4096 * 4096
    max_duration_s: float = 120.0
    max_frames: int = 216_000
    min_short_edge: int = 64
    min_face_px: int = 64

    def __post_init__(self) -> None:
        if self.max_pixels <= 0 or self.max_bytes <= 0:
            raise ValueError("caps must be positive")

    @property
    def gate_fingerprint(self) -> str:
        """Identity of this gate, for the composed model version.

        The caps decide which inputs get a score at all. Raising
        ``min_face_px`` turns a set of verdicts into abstentions without
        touching a single weight, so a cache that cannot see the caps will
        happily serve the verdicts the new floor was meant to withdraw.
        """
        return fingerprint(
            {
                "max_bytes": self.max_bytes,
                "max_pixels": self.max_pixels,
                "max_duration_s": self.max_duration_s,
                "max_frames": self.max_frames,
                "min_short_edge": self.min_short_edge,
                "min_face_px": self.min_face_px,
            }
        )


DEFAULT_CAPS = MediaCaps()


# --------------------------------------------------------------------------
# Image header probing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageHeader:
    width: int
    height: int
    format: str

    @property
    def pixels(self) -> int:
        return self.width * self.height


def probe_image_header(data: bytes) -> ImageHeader:
    """Read dimensions from the container header without decoding pixels.

    Exists so the pixel cap can be enforced *before* ``cv2.imdecode`` allocates.
    Checking dimensions after decode is checking after the damage.

    Limits worth stating: a deliberately corrupt header that understates its
    dimensions will still pass this and then allocate on decode. Fully
    containing that requires decoding in a subprocess under a memory ceiling —
    a caller-side measure, since this package is handed bytes and has no
    process boundary of its own to work with.
    """
    if len(data) < 12:
        raise UnreadableMedia("The file is too short to be an image.")

    # PNG: signature then IHDR at a fixed offset.
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        if len(data) < 24 or data[12:16] != b"IHDR":
            raise UnreadableMedia("PNG header is malformed.")
        w, h = struct.unpack(">II", data[16:24])
        return ImageHeader(w, h, "png")

    # GIF: little-endian dimensions immediately after the signature.
    if data[:6] in (b"GIF87a", b"GIF89a"):
        w, h = struct.unpack("<HH", data[6:10])
        return ImageHeader(w, h, "gif")

    # BMP: DIB header, dimensions at offset 18. Height may be negative
    # (top-down rows), which is a storage direction, not a size.
    if data[:2] == b"BM" and len(data) >= 26:
        w, h = struct.unpack("<ii", data[18:26])
        return ImageHeader(abs(w), abs(h), "bmp")

    # WebP: three sub-formats, three different places to look.
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8X" and len(data) >= 30:
            w = int.from_bytes(data[24:27], "little") + 1
            h = int.from_bytes(data[27:30], "little") + 1
            return ImageHeader(w, h, "webp")
        if chunk == b"VP8 " and len(data) >= 30:
            w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
            return ImageHeader(w, h, "webp")
        if chunk == b"VP8L" and len(data) >= 25:
            bits = int.from_bytes(data[21:25], "little")
            return ImageHeader((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1, "webp")
        raise UnreadableMedia("WebP header is malformed.")

    # JPEG: walk the marker chain to a start-of-frame.
    if data[:2] == b"\xff\xd8":
        i = 2
        n = len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            # SOF0..SOF15, excluding the non-frame markers in that range.
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return ImageHeader(w, h, "jpeg")
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            if i + 4 > n:
                break
            segment = struct.unpack(">H", data[i + 2 : i + 4])[0]
            if segment < 2:
                break
            i += 2 + segment
        raise UnreadableMedia("JPEG has no readable frame header.")

    raise UnreadableMedia("Unrecognised image format.")


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------


def decode_image(data: bytes, *, caps: MediaCaps = DEFAULT_CAPS) -> np.ndarray:
    """Decode to an ``(H, W, 3)`` uint8 RGB array. Raises on anything malformed."""
    if not data:
        raise UnreadableMedia("The file is empty.")
    if len(data) > caps.max_bytes:
        raise UnsupportedMedia(
            f"The file is larger than the {caps.max_bytes} byte limit."
        )

    header = probe_image_header(data)
    if header.pixels > caps.max_pixels:
        raise UnsupportedMedia(
            f"The image is {header.width}×{header.height}, beyond the "
            f"{caps.max_pixels}-pixel limit.",
            detail="rejected from the header, before decoding",
        )
    if header.width <= 0 or header.height <= 0:
        raise UnreadableMedia("The image header declares zero dimensions.")

    buffer = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise UnreadableMedia(
            "The file could not be decoded as an image.",
            detail=f"header claimed {header.format} {header.width}×{header.height}",
        )
    return to_rgb(bgr)


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    """cv2 hands back BGR. Convert once, here, at the boundary."""
    if bgr.ndim == 2:
        return cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    if bgr.shape[2] == 4:
        return cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize(image: np.ndarray, size: int) -> np.ndarray:
    """Square resize to ``size``×``size``.

    ``INTER_AREA`` when shrinking: it averages over the source region rather
    than sampling it, which matters here because forensic signal lives in
    high-frequency texture and point sampling aliases it into noise.
    """
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interp)


def normalize(
    image: np.ndarray,
    *,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """uint8 RGB → float32, scaled to [0,1] then standardised."""
    arr = image.astype(np.float32) / 255.0
    return (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)


FACE_SIZE_METRIC = "short-edge"
"""**The** face-size metric. One quantity, read by ranking, gating and reporting.

**R19.** Until Batch 14 there were two. ``face_order_key`` ranked boxes by
*area* and ``ImagePipeline._gate`` then tested the winner's *short edge* against
``MediaCaps.min_face_px`` — so the platform could select a box on one measure and
refuse it on another. That admits a state in which a file containing a 200 px
face is published as ``insufficient_evidence / face_too_small`` because a wider,
flatter box outranked it:

    900x50   area 45000  short edge  50   <- selected, then refused
    200x200  area 40000  short edge 200   <- never scored, never published

Two metrics ranking and gating the same decision is reachable exactly when they
are not monotonically related, and area and short edge are not. One metric makes
the state unrepresentable: if ranking maximises the gated quantity, then
``faces[0]`` failing the gate means every face fails it. See
``docs/FACE_SIZE_POLICY.md`` for the proof and ``tests/face_size_policy_study.py``
for the measurements.

**Short edge rather than area**, because the short edge is what decides whether
facial detail is *resolvable* — a 900x50 box has a large area and no vertical
detail — and because ``QualityReport.face_px`` already documents itself as a
short edge. Area survives as the ranking tiebreak, where it orders boxes that
resolve equally well.

A *name*, like :data:`ORDERING_RULE`, and held to the behaviour by
``test_the_face_size_metric_name_matches_the_metric`` rather than by intention.
"""


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def size_px(self) -> int:
        """How big this face is, for every decision that asks. See
        :data:`FACE_SIZE_METRIC`.

        Ranking (:func:`face_order_key`), gating (``ImagePipeline._gate``
        against ``MediaCaps.min_face_px``) and reporting
        (``QualityReport.face_px``) all read *this* property, so they cannot
        disagree about which face is the biggest. That they could is R19.

        Equal to :attr:`short_edge` — the metric is the short edge. The two
        names are not redundant: ``short_edge`` is a fact about a rectangle and
        ``size_px`` is the platform's choice of which fact to use. A future
        migration changes the second without touching the first.
        """
        return min(self.width, self.height)

    @property
    def short_edge(self) -> int:
        return min(self.width, self.height)

    @property
    def long_edge(self) -> int:
        """Named because :func:`crop_face` scales the crop by it — the third
        metric in the system, and the one R19's write-up does not mention.

        Not a size for gating or ranking. It is here so that the crop geometry's
        dependence on it is greppable rather than buried in a ``max()``. R23."""
        return max(self.width, self.height)

    @property
    def area(self) -> int:
        return self.width * self.height


CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

_cascade_local = threading.local()
"""One cascade per thread. **R15 — the shared one produced wrong answers.**

``cv2.CascadeClassifier`` carries mutable per-detection state: the feature
evaluator and its scaled-image buffers hang off the classifier, not off the
call. Two threads inside ``detectMultiScale`` on one instance therefore overwrite
each other's working set. It does not raise, and it does not corrupt memory
visibly — it returns a *plausible wrong answer*, which is the worst of the
available failure modes because every consumer treats it as a fact.

Measured in Batch 10, one process, one build, one runtime identity, identical
bytes (``tests/inference_determinism_study.py``, stage 11):

* eight threads detecting on the **same** image: 9 of 320 calls (2.8%) returned
  an empty box list — the face lost outright;
* through the real ``analyze_image``: 24 of 160 concurrent analyses published
  ``INSUFFICIENT_EVIDENCE / no_face_detected`` where all 40 sequential runs of
  the same bytes published ``AUTHENTIC`` at p_fake 0.0329;
* two threads on **different** images: a box belonging to neither, i.e. a
  detection fabricated from another job's media.

A ``threading.local`` rather than a lock around ``detectMultiScale``. Both fix it
— both measured at 320/320 correct — and they differ in what they cost. The lock
serialises the single most expensive step of every concurrent image job (24 ms
per call on this box); the local pays a one-off 45–73 ms load and 0.41 MB of
resident memory per pool thread, and the pool is bounded and reused, so that is
paid once per thread for the life of the process rather than once per job.

This is prevention, not detection, and it had to be: the two answers come from
one process, one decode identity and one model version, so no fingerprint
anywhere in the platform can tell them apart. See
``test_runtime_identity_cannot_detect_the_cascade_race``.
"""


def _face_cascade() -> cv2.CascadeClassifier:
    """This thread's Haar cascade, loaded on its first use.

    Placeholder detector. The frozen architecture specifies SCRFD with 5-point
    landmark alignment; Haar is frontal-only and misses profiles, occlusion and
    small faces. It is here so the no-face and face-too-small paths are
    exercisable end to end before the model stage, and it must be replaced
    rather than tuned.

    Per thread rather than per process — see :data:`_cascade_local`. The
    replacement detector inherits the requirement instead of retiring it:
    whatever SCRFD runtime lands must either be documented thread-safe or be
    held exactly this way, and ``test_the_cascade_is_not_shared_between_threads``
    fails if the sharing comes back.
    """
    cascade = getattr(_cascade_local, "cascade", None)
    if cascade is None:
        cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if cascade.empty():  # pragma: no cover - packaging failure
            raise UnreadableMedia(
                "Face cascade could not be loaded.", detail=CASCADE_PATH
            )
        _cascade_local.cascade = cascade
    return cascade


@lru_cache(maxsize=1)
def cascade_digest() -> str:
    """Content hash of the cascade XML this build actually loads.

    The detector's *asset* identity, as distinct from its algorithm. OpenCV ships
    the cascade as a data file and is free to reship a re-trained one under the
    same filename in a patch release; ``cv2.__version__`` would move, but nothing
    in this platform's model identity reads ``cv2.__version__`` — the decode
    identity does, and the decode identity is not part of a *model* comparison by
    design. So the asset is hashed here, where it belongs to the selection policy.

    Measured cost on this box: 46.8 ms for 930 127 bytes, paid once per process.
    That is the same order as the 45–73 ms the cascade load itself already costs
    per thread, and it is paid once rather than per thread.

    Falls back to a marker string rather than raising: a missing diagnostic must
    not be able to stop an analysis, and a policy that says it could not read its
    own asset is more honest than one that pretends the asset is absent.
    """
    try:
        with open(CASCADE_PATH, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:16]
    except OSError:  # pragma: no cover - packaging failure
        return "unreadable"


ORDERING_RULE = "short-edge-desc,area-desc,y-asc,x-asc,width-asc,height-asc"
"""Name of the rule :func:`face_order_key` implements. Goes into the identity.

**Batch 14 put ``short-edge-desc`` in front of ``area-desc``** so that ranking
maximises the quantity the gate tests. See :data:`FACE_SIZE_METRIC` for why, and
``docs/FACE_SIZE_POLICY.md`` §5 for the proof that this changes no output under
any detector whose boxes are within two pixels of square — which the shipped one
is, measured.

A *name*, not a hash of the function's bytecode. Bytecode moves with every
Python release and with every comment edit, which would invalidate the result
cache on changes that cannot move a box. The name is what a reviewer changes
deliberately when the rule changes, and ``test_the_ordering_rule_name_matches_
the_ordering_rule`` fails if the two drift apart — so the name is held to the
behaviour by a test rather than by good intentions.
"""


@dataclass(frozen=True)
class FaceSelectionPolicy:
    """Everything that decides **which face is scored and what pixels it shows**.

    **R17.** Before Batch 12 every field below was a hard-coded constant or a
    default argument, and none of them reached any identity string. Measured in
    ``tests/face_selection_study.py`` on one image, one model, one decoder:

        crop_margin 1.3 -> p_fake 0.013562  authentic              (shipped)
        crop_margin 1.2 -> p_fake 0.659734  insufficient_evidence
        crop_margin 1.0 -> p_fake 0.889291  manipulated

    Three different published verdicts under one byte-identical
    ``model_version``. The result cache keys on that string, so it would serve
    any of the three to a deployment running either of the others.

    **Every field is required — there are no defaults on this class.** That is
    deliberate and it is the structural half of the fix: hidden defaults were the
    defect, so a policy cannot be constructed without stating every knob, and
    :func:`default_face_selection` is the one place the shipped values live.

    **The policy is the parameter.** ``detect_faces`` and ``crop_face`` take this
    object rather than loose ``min_px`` / ``margin`` keywords, so the values that
    drive the behaviour and the values that drive the fingerprint are read from
    the same object. A caller cannot change one without changing the other, which
    is a stronger guarantee than hashing a set of constants some other code
    happens to use.
    """

    detector: str
    """Algorithm identity, e.g. ``opencv-haar-frontalface-default``. Moves when
    the detector is replaced — which is what the SCRFD migration will do."""

    detector_asset: str
    """Content digest of the detector's weights or cascade. See
    :func:`cascade_digest` for why the asset is separate from the algorithm."""

    scale_factor: float
    """``detectMultiScale`` pyramid step. Measured: 1.1 finds the fixture face,
    1.05, 1.2 and 1.3 all lose it outright and publish
    ``INSUFFICIENT_EVIDENCE / no_face_detected``."""

    min_neighbors: int
    """``detectMultiScale`` neighbour threshold. Measured: 3 and 5 agree, 8 and
    12 lose the face."""

    detect_min_px: int
    """``minSize`` passed to the detector — **not** ``MediaCaps.min_face_px``.

    These are two different floors and only one of them was ever fingerprinted.
    32 here decides which boxes *exist*; 64 in the caps decides which of them is
    large enough to score. A box between the two is detected, becomes
    ``faces[0]``, and is then gated as ``FACE_TOO_SMALL`` — so the pair jointly
    decides the abstention, and covering one without the other was a half
    fingerprint."""

    ordering: str
    """Name of the total order over boxes. See :data:`ORDERING_RULE`."""

    crop_margin: float
    """Expansion factor around the box. The largest single verdict-mover
    measured, and it was a default argument."""

    crop_shape: str
    """How the square crop is derived from the box. ``square-longest-edge`` is
    ``max(width, height) * margin``, centred on the box."""

    interpolation_rule: str
    """Name of the rule :func:`resize` uses to pick its cv2 interpolation.

    **R25, Batch 22 — a label closure, weaker than a threaded parameter.**
    ``resize`` is a bare module function and does not receive this policy, so
    this string is *not* read by any behaviour: it is a name held to
    ``resize``'s branch by ``test_the_interpolation_rule_name_matches_resize``,
    exactly as :data:`ORDERING_RULE` and :attr:`crop_shape` are. A code edit to
    ``resize`` that changes the interpolation without changing this string
    re-creates R25 one level down, and only that test catches it.

    The shipped value ``area-if-either-axis-exceeds-target-else-linear`` names
    the *current* behaviour verbatim, including the ``or`` that Batch 22 recorded
    as **R28** (an ``INTER_AREA`` low-pass applied to an axis being enlarged when
    a crop straddles the input size). B22 does not fix that predicate — doing so
    moves shipped pixels — it only records what the predicate does. R28's fix
    will change this string and move the ``.f.`` segment again, and rides the
    next batch that already moves it."""

    crop_clip_rule: str
    """Name of the rule :func:`crop_face` uses where the square window leaves the
    image.

    **R25, Batch 22 — a label closure, weaker than a threaded parameter.**
    ``crop_face`` does take this policy, so this *could* be threaded, but with a
    single shipped value (``intersect-image`` — the window is intersected with
    the image, which is why a border-flush box yields a non-square crop, R23) it
    branches on nothing yet. Held to ``crop_face``'s slice by
    ``test_the_clip_rule_name_matches_crop_face``. When R23 lands, the value
    becomes the chosen fill mode (``pad-<fill>``) and the ``.f.`` segment moves
    once more — the one accepted double-spend, scoped to this field."""

    @property
    def identity(self) -> str:
        """Fingerprint of this policy. The ``.f.`` segment of the model identity.

        Every field is in it. Unlike the decode identity, nothing here is
        excluded on the strength of a measurement: each of these was either
        measured to change a verdict, or is structurally certain to once the
        detector is replaced, and none of them has the cache-fragmentation cost
        that justified excluding CPU fields from the decode identity — a fleet
        runs one selection policy because it runs one build.
        """
        return fingerprint(
            {
                "detector": self.detector,
                "detector_asset": self.detector_asset,
                "scale_factor": self.scale_factor,
                "min_neighbors": self.min_neighbors,
                "detect_min_px": self.detect_min_px,
                "ordering": self.ordering,
                "crop_margin": self.crop_margin,
                "crop_shape": self.crop_shape,
                "interpolation_rule": self.interpolation_rule,
                "crop_clip_rule": self.crop_clip_rule,
            }
        )

    def to_dict(self) -> dict:
        """Forensic record. Every field plus the identity it produces."""
        return {
            "detector": self.detector,
            "detector_asset": self.detector_asset,
            "scale_factor": self.scale_factor,
            "min_neighbors": self.min_neighbors,
            "detect_min_px": self.detect_min_px,
            "ordering": self.ordering,
            "crop_margin": self.crop_margin,
            "crop_shape": self.crop_shape,
            "interpolation_rule": self.interpolation_rule,
            "crop_clip_rule": self.crop_clip_rule,
            "identity": self.identity,
        }

    def differences(self, other: "FaceSelectionPolicy") -> list[str]:
        """Field by field, so a mismatch says *what* changed."""
        out = []
        for field in (
            "detector",
            "detector_asset",
            "scale_factor",
            "min_neighbors",
            "detect_min_px",
            "ordering",
            "crop_margin",
            "crop_shape",
            "interpolation_rule",
            "crop_clip_rule",
        ):
            mine, theirs = getattr(self, field), getattr(other, field)
            if mine != theirs:
                out.append(f"{field}: {mine!r} != {theirs!r}")
        return out


@lru_cache(maxsize=1)
def default_face_selection() -> FaceSelectionPolicy:
    """The shipped policy. **The single place these numbers live.**

    Cached because it reads the cascade file. Called once per pipeline
    construction, so the 46.8 ms asset hash is paid once per process.

    The values are exactly what the platform did before Batch 12 — this batch
    moved them into an object and put that object's fingerprint into the version
    string. Nothing about the behaviour changed, and
    ``test_the_default_policy_reproduces_the_pre_batch_12_behaviour`` proves it
    against the constants verbatim.

    **Batch 22 (R25)** added ``interpolation_rule`` and ``crop_clip_rule``. Both
    name the *current* shipped behaviour verbatim — no pixels moved — and both
    are label closures held to ``resize`` / ``crop_face`` by re-derivation tests.
    """
    return FaceSelectionPolicy(
        detector="opencv-haar-frontalface-default",
        detector_asset=cascade_digest(),
        scale_factor=1.1,
        min_neighbors=5,
        detect_min_px=32,
        ordering=ORDERING_RULE,
        crop_margin=1.3,
        crop_shape="square-longest-edge",
        interpolation_rule="area-if-either-axis-exceeds-target-else-linear",
        crop_clip_rule="intersect-image",
    )


def face_selection_identity() -> str:
    """Fingerprint of the shipped face-selection policy. The one-liner.

    **Face policy only.** Since Batch 23 the ``.f.`` segment of the model
    identity is the *combined* selection identity (:func:`selection_identity`) —
    face selection plus, for video, frame selection and pooling. This function
    stays face-only for callers that genuinely want just that fingerprint.
    """
    return default_face_selection().identity


@dataclass(frozen=True)
class VideoSelectionPolicy:
    """Everything that decides **which frames are scored and how they are pooled**.

    **R29.** Before Batch 23 ``VideoPipeline.__init__`` took ``n_frames``,
    ``batch_size`` and ``min_usable_frames`` as bare scalar defaults, and the
    ``topk`` pooling depth was the literal ``arr.size // 8`` inside
    :func:`~inference.postprocess.topk_indices`. None of the four reached any
    identity string, so two deployments that disagreed on them produced a
    byte-identical ``model_version`` — the whole result-cache key — while
    sampling a different frame set and able to publish a different verdict. This
    is the video analogue of R17 (face selection) and R25 (crop geometry), and
    the fix is the same: one frozen dataclass, every field required, one factory
    function, its fingerprint folded into the model identity.

    **The policy is the parameter.** ``VideoPipeline`` reads ``n_frames``,
    ``min_usable_frames`` and ``pool_divisor`` from this object and passes them
    to ``sample_frame_indices`` / the abstention gate / ``topk_indices``, so the
    values that drive the behaviour and the values that drive the fingerprint are
    read from the same object.

    ``batch_size`` is **deliberately excluded** — it is output-neutral. The
    uint8→float32 widening is elementwise and ``topk_indices`` / ``pool_at`` run
    on the full ``scores`` array after every batch, so only progress-event
    fractions depend on it. This is the ``cpu_baseline`` / ``decode_env`` pattern:
    a knob that cannot move a published number stays out of the identity.
    ``test_batch_size_is_output_neutral`` holds that.

    Neutrality is **backend-dependent** (R32). It holds only while all three of:
    (a) ``batch_clips`` yields contiguous non-overlapping slices with **no
    padding** — it slices ``frames[start : start + batch_size]``, so the final
    batch is short, never padded; (b) the model backend's ``score()`` is
    **row-independent** — ``StubBackend.score`` iterates ``for i in range(N)``;
    and (c) ``per_frame`` reconstruction is **order- and value-preserving** —
    ``topk_indices`` / ``pool_at`` run on the full ``scores`` array after every
    batch. A static-shape backend (**ONNX** / **TensorRT**) pads the final batch
    and breaks (a); a cross-frame backend (temporal **attention** — the
    ``pooling_rule="attention"`` future) breaks (b). Either re-opens the
    exclusion and forces ``batch_size`` into this policy — a further ``.f.`` move
    and cache generation. ``test_the_batch_size_exclusion_note_is_complete``
    (PA11) holds this paragraph to naming (a)/(b)/(c) and the trigger.
    """

    n_frames: int
    """How many frames are uniformly sampled. **Functional** — passed to
    :func:`sample_frame_indices`, it decides which frames decode and (via
    ``arr.size``) the pool depth. Shipped value 48."""

    min_usable_frames: int
    """The ``TOO_FEW_FRAMES`` abstention floor. **Functional** — the pipeline
    reads it and abstains when fewer frames survive decoding. A verdict→abstention
    gate, the same shape as ``min_face_px`` (R19). Shipped value 8."""

    pool_divisor: int
    """The ``topk`` pooling depth divisor: ``k = max(1, n_scores // divisor)``.
    **Functional** — the pipeline computes ``k`` from this and passes it
    explicitly to :func:`~inference.postprocess.topk_indices`; a different
    divisor pools a different frame set and returns a different mean. Shipped
    value 8 (the historic ``arr.size // 8``)."""

    sampling_rule: str
    """Name of the stride rule :func:`sample_frame_indices` implements.

    **A label closure**, weaker than a threaded parameter — ``sample_frame_indices``
    is a bare module function and does not receive this policy. The string is held
    to its behaviour by ``test_the_sampling_rule_name_matches_sample_frame_indices``
    (a re-derivation test), exactly as :data:`ORDERING_RULE` is held to
    ``face_order_key``. It is *also* the value published in
    ``VideoAnalysis.sampling``, so the wire value and the identity input are one
    field. Shipped value ``"uniform"``."""

    pooling_rule: str
    """Name of the aggregation family. ``"topk_mean"`` now; ``"attention"`` when
    the temporal transformer beats it by a paired-bootstrap margin. ``pool_divisor``
    parameterises the family; this names it.

    **A label closure**, held to the pipeline by
    ``test_the_pooling_rule_and_divisor_match_the_pipeline`` and published as
    ``VideoAnalysis.pooling``. Shipped value ``"topk_mean"``."""

    def __post_init__(self) -> None:
        if self.n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {self.n_frames}")
        if not 1 <= self.min_usable_frames <= self.n_frames:
            raise ValueError(
                f"min_usable_frames must be in [1, n_frames={self.n_frames}], "
                f"got {self.min_usable_frames}"
            )
        if self.pool_divisor < 1:
            raise ValueError(f"pool_divisor must be >= 1, got {self.pool_divisor}")

    @property
    def identity(self) -> str:
        """Fingerprint of this policy. Folded into the ``.f.`` segment of the
        model identity by :func:`selection_identity`.

        Every field is in it. Each was either measured to change a verdict
        (``video_selection_study.py``) or is the parameter that drives the
        behaviour the label names.
        """
        return fingerprint(
            {
                "n_frames": self.n_frames,
                "min_usable_frames": self.min_usable_frames,
                "pool_divisor": self.pool_divisor,
                "sampling_rule": self.sampling_rule,
                "pooling_rule": self.pooling_rule,
            }
        )

    def to_dict(self) -> dict:
        """Forensic record. Every field plus the identity it produces."""
        return {
            "n_frames": self.n_frames,
            "min_usable_frames": self.min_usable_frames,
            "pool_divisor": self.pool_divisor,
            "sampling_rule": self.sampling_rule,
            "pooling_rule": self.pooling_rule,
            "identity": self.identity,
        }

    def differences(self, other: "VideoSelectionPolicy") -> list[str]:
        """Field by field, so a mismatch says *what* changed."""
        out = []
        for field in (
            "n_frames",
            "min_usable_frames",
            "pool_divisor",
            "sampling_rule",
            "pooling_rule",
        ):
            mine, theirs = getattr(self, field), getattr(other, field)
            if mine != theirs:
                out.append(f"{field}: {mine!r} != {theirs!r}")
        return out


@lru_cache(maxsize=1)
def default_video_selection() -> VideoSelectionPolicy:
    """The shipped video-selection policy. **The single place these numbers live.**

    The values are exactly what the platform did before Batch 23 — ``n_frames``
    and ``min_usable_frames`` were ``VideoPipeline`` defaults, ``pool_divisor``
    was the ``arr.size // 8`` literal in ``topk_indices``, and the two labels
    were hard-coded in ``build_video_analysis`` (``"uniform"``) and
    ``VideoAnalysis`` (``"topk_mean"``). Nothing about the behaviour changed;
    Batch 23 moved them into an object and put that object's fingerprint into the
    version string.
    """
    return VideoSelectionPolicy(
        n_frames=48,
        min_usable_frames=8,
        pool_divisor=8,
        sampling_rule="uniform",
        pooling_rule="topk_mean",
    )


def video_selection_identity() -> str:
    """Fingerprint of the shipped video-selection policy. The one-liner."""
    return default_video_selection().identity


def selection_identity(
    face: FaceSelectionPolicy, video: VideoSelectionPolicy
) -> str:
    """The combined selection identity — the ``.f.`` segment of the model identity.

    **Batch 23 broadened the ``.f.`` segment's meaning.** It named the
    face-selection policy alone (Batch 12, R17); it now names the *selection*
    policy: face selection for the image path, face selection plus frame
    selection and pooling for the video path. Both pipelines call this and pass
    the result to ``ModelSpec.with_selection(...)``, so the single
    ``STUB_ENGINE_VERSION`` literal is preserved and ``contracts/version.py``'s
    parser is untouched — ``.f.`` is still one opaque 8-hex fingerprint.

    The cost is that the *image* cache namespace regenerates for a video-only
    policy addition (a false miss — compute, not correctness). This is precedented
    exactly by Batch 20, whose ``.d.`` demuxer change regenerated the image
    namespace although image decoding uses no demuxer, and is settled by
    ``docs/RUN_IDENTITY.md`` §3 (a cache key breaks ties toward missing).

    This is a hash of two hashes and is **not injective** over the ``(face,
    video)`` pair — two distinct policy pairs could in principle collide on the
    8-hex ``.f.`` value. That is acceptable for exactly the reason every
    ``fingerprint()`` in the platform is (``contracts/version.py`` FINGERPRINT_LENGTH):
    it is a 32-bit *change detector* across the handful of policies a deployment
    ever holds, not a security boundary, and nothing here defends against a
    deliberately constructed collision. The composition adds no collision risk
    beyond what ``FaceSelectionPolicy.identity`` and ``VideoSelectionPolicy.identity``
    already carry individually.
    """
    return fingerprint({"face": face.identity, "video": video.identity})


def face_order_key(box: FaceBox) -> tuple[int, int, int, int, int, int]:
    """Total order on faces: largest first, then top-most, then left-most.

    **"Largest" means :attr:`FaceBox.size_px`, and that is R19's fix.** Until
    Batch 14 the leading component was ``-area`` while the gate this ordering
    feeds tested ``short_edge``, so the platform could select a box on one
    measure and refuse it on another — publishing ``FACE_TOO_SMALL`` for a file
    whose second face cleared every floor. Ranking on the gated quantity makes
    that unrepresentable: if ``faces[0]`` fails the floor then every box does,
    because ``faces[0]`` maximises exactly what the floor tests.

    Area is the *secondary* key, where it does real work: among boxes that
    resolve equally well, prefer the one covering more of the image. Under any
    detector whose boxes are within two pixels of square this key is
    **byte-identical** to the pre-Batch-14 ``(-area, y, x, w, h)`` — proved
    exhaustively in ``test_the_two_orderings_agree_below_the_threshold``, because
    ``se1 > se2`` forces ``area1 > area2`` once ``|w - h| <= 2``. The two
    orderings diverge exactly at the aspect deviation that makes R19 reachable,
    three pixels, and not before.

    **Every component is a property of the box itself**, so the ordering is a
    function of the image rather than of the detector's output order. That is
    the whole point, and it is the same defect ``topk_indices`` fixed for frame
    ranking in Batch 0 — an index tiebreak there, a geometric one here.

    Before this, ``sorted(boxes, key=lambda b: -b.area)`` decided ties by
    position in the detector's output, because Python's sort is stable. Two
    faces of equal area therefore resolved to whichever the cascade happened to
    emit first, and ``faces[0]`` chooses the crop that the entire verdict is
    computed from. Measured on a 400x800 fixture with two 200x200 regions:

        A first -> crop 6c9a2dfa35, p_fake 0.4830  (authentic)
        B first -> crop f04cb2ef50, p_fake 0.5637  (insufficient evidence)

    Same image, same model, different verdict, decided by an ordering nothing
    specified. The tie is exact-area only, so it is rare with Haar and it did
    not fire on any shipped fixture — but ``minSize`` quantisation makes equal
    areas entirely reachable, and the frozen architecture replaces this detector
    with SCRFD, whose box order comes out of NMS over float scores and is an
    implementation detail by construction.

    Width and height are in the key after ``y`` and ``x`` so that the order is
    *total*: two boxes with the same area and origin but different aspect
    (100x200 and 200x100 both have area 20000) would otherwise still tie.
    """
    return (-box.size_px, -box.area, box.y, box.x, box.width, box.height)


def detect_faces(
    image: np.ndarray, *, policy: FaceSelectionPolicy | None = None
) -> list[FaceBox]:
    """Detect frontal faces. Largest first. Empty list is a normal answer.

    Ordered by :func:`face_order_key`, which is a total order on the box
    geometry — so ``faces[0]``, and therefore the crop the model sees, cannot
    depend on the order the detector emitted its boxes in.

    **Every detector parameter comes from ``policy``, and nothing else.** Batch
    12 replaced a ``min_px`` keyword and two hard-coded literals with this
    object, because the literals were verdict-affecting and unfingerprinted
    (R17): ``scaleFactor`` 1.05, 1.2 or 1.3 loses the fixture face outright and
    turns an ``AUTHENTIC`` verdict into ``INSUFFICIENT_EVIDENCE``. A caller
    wanting different parameters passes a different policy, and that policy has a
    different :attr:`~FaceSelectionPolicy.identity`, so the change cannot reach
    the pixels without reaching the version string.
    """
    policy = policy or default_face_selection()
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    found = _face_cascade().detectMultiScale(
        grey,
        scaleFactor=policy.scale_factor,
        minNeighbors=policy.min_neighbors,
        minSize=(policy.detect_min_px, policy.detect_min_px),
    )
    boxes = [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in found]
    return sorted(boxes, key=face_order_key)


def crop_face(
    image: np.ndarray, box: FaceBox, *, policy: FaceSelectionPolicy | None = None
) -> np.ndarray:
    """A square *window* around a face, expanded by the policy's margin — then
    **clipped to the image, which does not always leave it square**.

    A clipped crop is distorted by the square resize downstream. The three cases
    below say exactly when that happens.

    The intent is square, because the model takes a square input and a square
    crop keeps facial proportions intact where a stretched one would not. The
    implementation does not always achieve it, and Batch 15 measured where:

    * **Away from every border** the window fits and the crop is square. The
      model sees the face centred, at ``1/margin`` of the crop's width.
    * **At the left or top border** ``x0``/``y0`` clamp to 0, the crop keeps its
      full ``side`` and the *face slides off centre* — a 200 px box flush left
      sits 30 px from where it should be. Framing changes; shape does not.
    * **At the right or bottom border** ``x1``/``y1`` truncate, so the crop is
      **not square** and :func:`prepare_image` stretches it onto the square
      input. A 200 px box flush with the right edge of a 480 px image yields a
      230x260 crop — a **13% anisotropic stretch** of exactly the facial geometry
      the paragraph above says this function protects.

    Truncation fires when the gap between the box edge and the frame is less
    than ``ceil(side/2) - ceil(s/2)``, about 15% of the box's own width at the
    shipped margin — an off-centre subject, or any face wider than ``1/margin``
    of the frame. **This needs no free-aspect detector: square Haar boxes reach
    it today.** It is R23, it is measured in ``tests/crop_geometry_study.py`` and
    pinned in ``tests/test_crop_geometry.py``, and it is deliberately *not fixed*
    — Batch 15 is an investigation batch and changing the crop moves every
    verdict. ``docs/CROP_GEOMETRY.md`` has the alternatives and the argument for
    doing it with the SCRFD migration rather than before it.

    A second consequence, latent rather than live: ``side`` comes from the
    **long** edge, so a free-aspect box produces a crop that is mostly not face —
    59% of the model's input at aspect 1.0, 44% at 0.75, 3.3% at 900x50.

    **The margin comes from ``policy``.** It was a default argument of 1.3 until
    Batch 12, and it is the single largest verdict-mover R17 measured: 1.3, 1.2
    and 1.0 publish ``authentic``, ``insufficient_evidence`` and ``manipulated``
    for the same image under the same model. A number that decides that must be
    in the version string, and now is.
    """
    policy = policy or default_face_selection()
    h, w = image.shape[:2]
    side = int(max(box.width, box.height) * policy.crop_margin)
    cx, cy = box.x + box.width // 2, box.y + box.height // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        raise UnreadableMedia("Face crop fell outside the image bounds.")
    return crop


def prepare_image(
    image: np.ndarray, *, size: int, normalise: bool = False
) -> np.ndarray:
    """Resize to the model's input, as a single-item batch ``(1, S, S, 3)``.

    Left un-normalised by default: the stub backend hashes raw pixels, and
    normalising first would make its output depend on the constants rather than
    on the image. Real backends normalise inside ``score``.
    """
    resized = resize(image, size)
    arr = normalize(resized) if normalise else resized.astype(np.float32)
    return arr[np.newaxis, ...]


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int

    @property
    def short_edge(self) -> int:
        return min(self.width, self.height)


_SUFFIX_FOR_MIME = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
}


@contextlib.contextmanager
def temp_media_file(data: bytes, *, suffix: str = ".mp4") -> Iterator[Path]:
    """Spool bytes to a temporary file and yield its path.

    ``cv2.VideoCapture`` has no bytes interface; it opens a path. Hence the
    spool, and hence ``delete=False``: on Windows a still-open
    ``NamedTemporaryFile`` cannot be reopened by another handle, so the file is
    written, closed, handed to cv2 by path, and unlinked in ``finally`` —
    including when decoding raises, which is precisely when a leaked 200 MB
    temp file would go unnoticed.
    """
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        handle.write(data)
        handle.flush()
        handle.close()
        yield Path(handle.name)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(handle.name)


def suffix_for(mime: str) -> str:
    return _SUFFIX_FOR_MIME.get(mime, ".mp4")


def probe_video(path: Path, *, caps: MediaCaps = DEFAULT_CAPS) -> VideoMeta:
    """Read metadata without decoding frames, and apply the caps to it.

    Opened through :func:`~inference.decode_env.open_video` so the backend is
    the one this deployment names, not the one OpenCV's priority table happens
    to rank first. Measured in Batch 8: ``OPENCV_VIDEOIO_PRIORITY_MSMF=9999``
    switches the decoder with no rebuild and changes every decoded frame, and
    ``frame_count`` read here selects which frames are sampled at all.
    """
    capture = open_video(path)
    try:
        if not capture.isOpened():
            raise UnreadableMedia("The file could not be opened as a video.")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    if width <= 0 or height <= 0 or frames <= 0:
        raise UnreadableMedia(
            "The video reports no usable frames.",
            detail=f"fps={fps} frames={frames} {width}×{height}",
        )
    if width * height > caps.max_pixels:
        raise UnsupportedMedia(f"Video resolution {width}×{height} exceeds the limit.")
    if frames > caps.max_frames:
        raise UnsupportedMedia(f"Video has {frames} frames, beyond the limit.")

    # A container can report 0 fps. Treating that as 25 keeps duration finite
    # rather than dividing by zero, and it is flagged by the caller through the
    # frame count rather than silently trusted.
    effective_fps = fps if fps > 0 else 25.0
    duration = frames / effective_fps
    if duration > caps.max_duration_s:
        raise UnsupportedMedia(
            f"The video is {duration:.1f}s, beyond the {caps.max_duration_s}s limit."
        )
    return VideoMeta(effective_fps, frames, duration, width, height)


def sample_frame_indices(frame_count: int, n: int) -> list[int]:
    """Uniformly spaced indices. Deterministic — no randomness anywhere here.

    Determinism is not a nicety: crash recovery compares results across
    attempts, and a sampler that picked different frames on a retry would
    produce a different bundle for the same input.
    """
    if frame_count <= 0 or n <= 0:
        return []
    if frame_count <= n:
        return list(range(frame_count))
    step = frame_count / n
    return [min(frame_count - 1, int(i * step)) for i in range(n)]


class ExtractedFrames(NamedTuple):
    """What one decode pass yields.

    ``timestamps_ms`` is per *retained* frame and is read from the container,
    not inferred from how many frames survived. That distinction is the whole
    point: a clip whose frames partly fail to decode must still report where the
    survivors actually were.
    """

    meta: VideoMeta
    frames: np.ndarray
    indices: list[int]
    timestamps_ms: list[int]


def _position_ms(capture: cv2.VideoCapture, index: int, fps: float) -> int:
    """Presentation time of the frame just grabbed.

    Prefers the container's own timestamp, which is correct for variable frame
    rate as well as constant. Falls back to ``index / fps`` when the container
    does not report one — deterministic either way, and never derived from how
    many frames happened to survive decoding.
    """
    position = capture.get(cv2.CAP_PROP_POS_MSEC)
    if position and position > 0:
        return int(round(position))
    return int(round(index * 1000.0 / fps)) if fps > 0 else 0


def extract_frames(
    data: bytes,
    *,
    mime: str = "video/mp4",
    n_frames: int = 48,
    caps: MediaCaps = DEFAULT_CAPS,
    target_size: int | None = None,
) -> ExtractedFrames:
    """Decode ``n_frames`` uniformly sampled RGB frames in one sequential pass.

    **One pass, no seeking.** The previous implementation called
    ``capture.set(CAP_PROP_POS_FRAMES, i)`` before each wanted frame. On
    long-GOP H.264 every such seek decodes forward from the preceding keyframe,
    so sampling 48 frames re-decoded large parts of the clip dozens of times —
    measured at 33.5 s for a 60-frame 4K clip. Walking the stream once with
    ``grab()`` and paying for ``retrieve()`` only on wanted frames does the same
    work once. Seeking is also unreliable on variable-frame-rate containers,
    which meant the returned ``indices`` did not always name the frames actually
    decoded.

    **Bounded memory.** With ``target_size`` set, frames are resized as they are
    decoded and written straight into one pre-allocated buffer. Nothing
    accumulates a list of full-resolution frames and nothing calls ``np.stack``,
    which previously held the list and its copy simultaneously — 2.4 GB peak for
    a single 4K job whose useful output is 21 MB. Only one source frame is
    resident at a time.

    ``frames`` is ``(N, H, W, 3)`` uint8, at ``target_size`` when given and at
    source resolution otherwise. N may be smaller than requested when frames
    fail to decode; whether what survived is enough to judge is the caller's
    decision, not this function's.
    """
    if not data:
        raise UnreadableMedia("The file is empty.")
    if len(data) > caps.max_bytes:
        raise UnsupportedMedia(f"The file is larger than the {caps.max_bytes} byte limit.")

    with temp_media_file(data, suffix=suffix_for(mime)) as path:
        meta = probe_video(path, caps=caps)
        wanted = sample_frame_indices(meta.frame_count, n_frames)
        if not wanted:
            raise UnreadableMedia("The video reports no frames to sample.")
        pending = set(wanted)

        height = target_size or meta.height
        width = target_size or meta.width
        # Allocated once, at the size actually needed. The old code grew a list
        # of full-resolution frames and then copied it; this is the single
        # change that takes a 4K job from gigabytes to megabytes.
        out = np.empty((len(wanted), height, width, 3), dtype=np.uint8)

        capture = open_video(path)
        kept: list[int] = []
        stamps: list[int] = []
        written = 0
        try:
            if not capture.isOpened():
                raise UnreadableMedia("The video could not be reopened for reading.")
            index = 0
            remaining = len(wanted)
            while remaining > 0:
                # grab() advances the stream without the colour conversion and
                # copy that read() performs. Frames we do not want cost only
                # the decode the codec requires anyway.
                if not capture.grab():
                    break
                if index in pending:
                    remaining -= 1
                    stamp = _position_ms(capture, index, meta.fps)
                    ok, frame = capture.retrieve()
                    if ok and frame is not None:
                        rgb = to_rgb(frame)
                        out[written] = resize(rgb, target_size) if target_size else rgb
                        kept.append(index)
                        stamps.append(stamp)
                        written += 1
                    # A single unreadable frame is normal in a damaged file and
                    # is not fatal; too few surviving frames is, and that
                    # judgement belongs to the pipeline.
                index += 1
        finally:
            capture.release()

    if written == 0:
        raise UnreadableMedia("No frames could be decoded from the video.")
    # A view, not a copy: the tail of the buffer is simply not returned.
    return ExtractedFrames(meta, out[:written], kept, stamps)


def batch_clips(frames: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    """Yield contiguous batches. The unit of both inference and progress."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, frames.shape[0], batch_size):
        yield frames[start : start + batch_size]
