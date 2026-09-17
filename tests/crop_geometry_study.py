"""Crop geometry: from the detector's box to the model's input tensor.
Standalone, read-only, changes nothing.

    python tests/crop_geometry_study.py

Produces every number in ``docs/CROP_GEOMETRY.md``. Same pattern as
``decode_study.py``, ``band_study.py``, ``face_selection_study.py``,
``multi_face_study.py`` and ``face_size_policy_study.py``.

THE QUESTION
------------
R23, opened by Batch 14: *``crop_face`` computes ``side = long_edge * margin``,
clips to image bounds, and can produce a crop where a free-aspect face occupies
a small fraction of the crop; the result is then resized to a square by
``prepare_image``, potentially introducing geometric distortion despite the
docstring claiming distortion is avoided.*

Is that a correctness defect, a reproducibility defect, a documentation defect,
or some combination? This study measures rather than argues, and the answer
turns out to be *all three* — and to differ from the finding as recorded in two
important ways:

  1. **The distortion has nothing to do with free aspect.** It comes from
     *clipping*, and a perfectly square Haar box near a frame border produces it
     today. Stage 3.
  2. **Occupancy collapse and distortion are two different defects** with two
     different activation conditions. Free aspect drives occupancy; border
     proximity drives distortion. Conflating them hides that one of the two is
     already live. Stage 3.

And the largest geometry defect in the platform is not in the crop path at all.
Stage 6.

WHY THERE ARE DOUBLES
---------------------
Haar detects the synthetic fixture only when it is alone on a 480x480 canvas —
trim twenty pixels and the face is lost (stage 5, and R13 again). So the image
scenarios script the detector with boxes of exactly the shape Haar emits:
*square*. That is the point rather than a limitation, because it is what shows
the defect needs no free-aspect box.

    ScriptedDetector   replaces  inference.preprocess.detect_faces
    LegacyClipToBounds replaces  nothing -- it IS the shipped crop_face, verbatim
    PadToSquareCrop    a corrected crop, for the fail-before/after pairing
    LetterboxResize    an aspect-preserving resize, for the same pairing on video

Everything else is shipped code.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from inference import preprocess as pre  # noqa: E402
from inference.models import ModelSpec  # noqa: E402
from inference.preprocess import FaceBox  # noqa: E402

POLICY = pre.default_face_selection()
MARGIN = POLICY.crop_margin
INPUT = ModelSpec(name="stub-image", version="stub-0", backend="stub").input_size


def head(n: int, title: str, question: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n   {question}\n{'=' * 78}")


# --------------------------------------------------------------------------
# Geometry vocabulary — the numbers this study reports
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """What one crop did to one face, in five numbers."""

    crop_w: int
    crop_h: int
    side: int
    box: FaceBox
    img_w: int
    img_h: int

    @property
    def anisotropy(self) -> float:
        """How much ``prepare_image``'s square resize stretches one axis.

        The crop is resized to ``size x size`` regardless of its shape, so the
        two axes are scaled by ``size/crop_w`` and ``size/crop_h``. Their ratio
        is the distortion: 1.0 is none, 1.13 means one axis is stretched 13%
        relative to the other. A circle in the source becomes an ellipse of
        exactly this eccentricity in the model's input.
        """
        return max(self.crop_w / self.crop_h, self.crop_h / self.crop_w)

    @property
    def occupancy(self) -> float:
        """Fraction of the model's input that is the detected face box."""
        return (self.box.width * self.box.height) / float(self.crop_w * self.crop_h)

    @property
    def truncated(self) -> bool:
        return self.crop_w < self.side or self.crop_h < self.side

    @property
    def centring_error(self) -> tuple[int, int]:
        """How far the face centre sits from the crop centre, in source pixels.

        Zero when the expanded window fits. Non-zero when the window was pushed
        off the left or top edge — the crop stays square and the face moves.
        """
        cx = self.box.x + self.box.width // 2
        cy = self.box.y + self.box.height // 2
        x0 = max(0, cx - self.side // 2)
        y0 = max(0, cy - self.side // 2)
        return (cx - x0) - self.crop_w // 2, (cy - y0) - self.crop_h // 2


def measure(img_w: int, img_h: int, box: FaceBox, margin: float = MARGIN) -> Geometry:
    """The shipped ``crop_face`` arithmetic, on dimensions alone."""
    side = int(max(box.width, box.height) * margin)
    cx, cy = box.x + box.width // 2, box.y + box.height // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(img_w, x0 + side)
    y1 = min(img_h, y0 + side)
    return Geometry(x1 - x0, y1 - y0, side, box, img_w, img_h)


# --------------------------------------------------------------------------
# The named doubles
# --------------------------------------------------------------------------


class LegacyClipToBounds:
    """The shipped ``crop_face``, verbatim. Intersects the window with the image.

    Kept beside the corrected double so the defect is demonstrated rather than
    described. Two behaviours follow from the one line ``x1 = min(w, x0 + side)``
    and they are *different at different borders*:

    * **left / top** — ``x0`` is clamped to 0, the window keeps its full width,
      and the face slides off centre. No distortion, wrong framing.
    * **right / bottom** — ``x0`` is not clamped, ``x1`` is truncated, and the
      crop is no longer square. Correct framing, **anisotropic distortion**.

    ``prepare_image`` then resizes whatever it is handed to ``size x size``, so
    the second case reaches the model as a stretched image.
    """

    name = "square window intersected with the image (shipped)"

    @staticmethod
    def crop(image: np.ndarray, box: FaceBox, margin: float = MARGIN) -> np.ndarray:
        h, w = image.shape[:2]
        side = int(max(box.width, box.height) * margin)
        cx, cy = box.x + box.width // 2, box.y + box.height // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
        return image[y0:y1, x0:x1]


class PadToSquareCrop:
    """A corrected crop: compute the window, then *fill* it rather than shrink it.

    The window is the same square ``crop_face`` intends. Where it falls outside
    the image the missing pixels are filled by edge replication instead of the
    window being cut down. Two properties follow, and they are the two the
    shipped code loses:

    * the crop is **always** ``side x side``, so ``prepare_image``'s square
      resize is isotropic — anisotropy 1.0, always;
    * the face centre is **always** at the crop centre, so the model sees the
      same framing wherever in the frame the face happens to be.

    Edge replication rather than a constant fill because a constant introduces a
    hard synthetic edge, and hard edges are exactly the high-frequency signal a
    manipulation detector reads. This double exists to prove the failure is
    eliminable at all; *which* fill a shipped version should use is a modelling
    question this study does not answer (§9).

    **Not shipped.** Batch 15 is an investigation batch.
    """

    name = "square window padded by edge replication (corrected)"

    @staticmethod
    def crop(image: np.ndarray, box: FaceBox, margin: float = MARGIN) -> np.ndarray:
        h, w = image.shape[:2]
        side = int(max(box.width, box.height) * margin)
        cx, cy = box.x + box.width // 2, box.y + box.height // 2
        x0 = cx - side // 2
        y0 = cy - side // 2
        x1, y1 = x0 + side, y0 + side
        pad_l, pad_t = max(0, -x0), max(0, -y0)
        pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
        inner = image[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
        if pad_l or pad_r or pad_t or pad_b:
            inner = cv2.copyMakeBorder(
                inner, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
            )
        return inner


class LetterboxResize:
    """An aspect-preserving resize to a square, for the video-path pairing.

    Scales by the *same* factor on both axes and pads the short dimension. A
    circle stays a circle. The shipped ``resize`` scales each axis
    independently, so a 16:9 frame reaches the model squashed by 1.778x.

    **Not shipped.** Listed as alternative E in ``docs/CROP_GEOMETRY.md`` §9.
    """

    name = "scale by one factor, pad the short axis (corrected)"

    @staticmethod
    def resize(image: np.ndarray, size: int) -> np.ndarray:
        h, w = image.shape[:2]
        scale = size / max(h, w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        small = cv2.resize(image, (nw, nh), interpolation=interp)
        top, left = (size - nh) // 2, (size - nw) // 2
        return cv2.copyMakeBorder(
            small, top, size - nh - top, left, size - nw - left, cv2.BORDER_REPLICATE
        )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def face_tile(size: int = 480) -> np.ndarray:
    """The determinism study's synthetic face, verbatim. The one Haar can see."""
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (175, 160, 150), -1)
    cv2.circle(img, (200, 210), 16, (60, 60, 60), -1)
    cv2.circle(img, (280, 210), 16, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 300), (18, 30), 0, 0, 360, (150, 130, 120), -1)
    cv2.ellipse(img, (240, 340), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    return cv2.GaussianBlur(img, (5, 5), 0)


def stub_p(crop: np.ndarray, size: int = INPUT) -> float:
    """``StubBackend``'s spatial branch, reproduced so a crop change is visible.

    The stub hashes its input, so the *direction* of any change here is
    arithmetic rather than judgement — stated in Batch 12 and repeated. What it
    establishes is the part that matters: the model is handed different pixels.
    """
    batch = pre.prepare_image(crop, size=size)
    digest = hashlib.sha256(b"spatial" + batch[0].tobytes()).hexdigest()
    return int(digest[:12], 16) / float(0x1000000000000)


# --------------------------------------------------------------------------
# 1 — the geometry flow
# --------------------------------------------------------------------------


def s1_flow() -> None:
    head(1, "The geometry flow", "where is face geometry created, changed and published?")

    root = Path(__file__).resolve().parent.parent
    sites = [
        ("created", "preprocess.detect_faces", "cv2 detectMultiScale -> list[FaceBox], ints"),
        ("ordered", "preprocess.face_order_key", "-size_px, -area, y, x, w, h  (B14)"),
        ("expanded", "preprocess.crop_face", "side = long_edge * crop_margin"),
        ("translated", "preprocess.crop_face", "x0 = max(0, cx - side//2)   <- clamp"),
        ("clipped", "preprocess.crop_face", "x1 = min(w, x0 + side)      <- truncate"),
        ("resized", "preprocess.prepare_image -> resize", "cv2.resize(crop, (size, size))"),
        ("normalised", "preprocess.prepare_image", "off by default; astype(float32) only"),
        ("batched", "preprocess.prepare_image", "arr[np.newaxis] -> (1, S, S, 3)"),
        ("scored", "backend.score", "the model never sees the source image"),
        ("published", "postprocess.face_track", "BoundingBox(max(0,x), max(0,y), ...)"),
        ("video: resized", "preprocess.extract_frames", "resize(whole frame, target_size)"),
    ]
    print(f"  {'stage':<16} {'site':<40} what it does")
    for stage, site, what in sites:
        print(f"  {stage:<16} {site:<40} {what}")

    print("\n  Line numbers, from source:")
    src = (root / "inference/preprocess.py").read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(src, 1):
        s = line.strip()
        if s.startswith(("def crop_face", "def prepare_image", "def resize", "def detect_faces")):
            print(f"    inference/preprocess.py:{i}  {s}")
        if "x1 = min(w, x0 + side)" in s or "side = int(max(" in s or "resized = resize(" in s:
            print(f"    inference/preprocess.py:{i}  {s}")
        if "out[written] = resize(" in s:
            print(f"    inference/preprocess.py:{i}  {s}")

    print("\n  THE FLOW, image path:")
    print("    bytes -> decode_image -> detect_faces -> [face_order_key] -> faces[0]")
    print("          -> crop_face   : expand by margin on the LONG edge,")
    print("                           translate (clamp at 0), clip (truncate at w/h)")
    print("          -> prepare_image: cv2.resize to (S, S), independent per-axis scale")
    print("          -> backend.score")
    print("    and separately: faces[0] -> face_track -> EvidenceBundle.faces[0].bbox")
    print()
    print("  THE FLOW, video path:")
    print("    bytes -> extract_frames(target_size=input_size)")
    print("          -> resize(WHOLE FRAME, S) -> backend.score")
    print("    No detection, no crop, no box. Stage 6.")
    print()
    print("  **Nothing publishes the region the model actually scored.** The bundle")
    print("  carries the detector's box; the crop is expanded, possibly translated,")
    print("  possibly truncated and always resized after that box is recorded.")


# --------------------------------------------------------------------------
# 2 — the assumptions
# --------------------------------------------------------------------------


def s2_assumptions() -> None:
    head(2, "Geometry assumptions", "what is assumed, where, and is anything holding it?")

    rows = [
        ("detector boxes are square-ish", "implicit in crop_face + gate", "no", "B14 only", "no"),
        ("crop is square", "crop_face docstring", "yes -- and it is FALSE", "no", "no"),
        ("aspect distortion is avoided", "crop_face docstring", "yes -- and it is FALSE", "no", "no"),
        ("margin expands the LONG edge", "crop_face body", "since B14 (long_edge)", "no", "YES crop_margin"),
        ("crop shape name", "policy.crop_shape", "yes", "no", "YES crop_shape"),
        ("clipping/centring rule", "crop_face body", "no", "no", "NO"),
        ("padding: none, ever", "crop_face body", "no", "no", "NO"),
        ("model input is square", "prepare_image, ModelSpec.input_size", "partly", "no", "NO input_size"),
        ("resize is per-axis independent", "preprocess.resize", "no", "no", "NO"),
        ("interpolation choice", "preprocess.resize", "partly (why AREA)", "no", "NO"),
        ("video frames may be squashed", "extract_frames", "no", "no", "NO"),
        ("published bbox == scored region", "face_track", "no", "no", "n/a"),
    ]
    print(f"  {'assumption':<32} {'encoded in':<38} {'documented':<22} {'tested':<10} fingerprinted")
    for a, b, c, d, e in rows:
        print(f"  {a:<32} {b:<38} {c:<22} {d:<10} {e}")

    print("\n  Two of these are documented and false. From crop_face's docstring:")
    doc = pre.crop_face.__doc__ or ""
    for line in doc.splitlines():
        if "Square because" in line or "aspect distortion" in line or "stretches" in line:
            ascii_line = line.strip().encode("ascii", "replace").decode("ascii")
            print(f"    > {ascii_line}")
    print("\n  Stage 3 measures the crop that sentence describes.")

    print("\n  And one test is named for a property it does not check:")
    print("    tests/test_preprocess.py::test_crop_face_is_square_and_clipped_to_bounds")
    img = np.zeros((100, 100, 3), np.uint8)
    corner = pre.crop_face(img, FaceBox(90, 90, 40, 40))
    edge = pre.crop_face(img, FaceBox(90, 40, 40, 40))
    print(f"      its own fixture, FaceBox(90,90,40,40) -> {corner.shape[1]}x{corner.shape[0]} "
          f"square={corner.shape[0] == corner.shape[1]}")
    print(f"      the SAME box moved off the corner, (90,40) -> {edge.shape[1]}x{edge.shape[0]} "
          f"square={edge.shape[0] == edge.shape[1]}  "
          f"aniso={max(edge.shape[1]/edge.shape[0], edge.shape[0]/edge.shape[1]):.4f}")
    print("      The fixture sits in the one corner where both axes clip by the same")
    print("      amount, so the crop is square by coincidence. The body asserts only")
    print("      `crop.size > 0` and that the crop fits the image; the word 'square'")
    print("      appears in the name and nowhere in the assertions.")


# --------------------------------------------------------------------------
# 3 — does distortion actually occur?
# --------------------------------------------------------------------------


def s3_distortion() -> None:
    head(3, "Does distortion occur", "measured, over box shapes and positions")

    print("  (a) SQUARE boxes -- the only shape the shipped detector emits --")
    print("      swept across a 480x480 image. side = 200 * 1.3 = 260.\n")
    print(f"      {'position':<26} {'crop':>10} {'aniso':>8} {'stretch':>9} {'occupancy':>10} {'off-centre':>12}")
    cases = [
        ("centred", FaceBox(140, 140, 200, 200)),
        ("flush LEFT", FaceBox(0, 140, 200, 200)),
        ("flush TOP", FaceBox(140, 0, 200, 200)),
        ("flush RIGHT", FaceBox(280, 140, 200, 200)),
        ("flush BOTTOM", FaceBox(140, 280, 200, 200)),
        ("bottom-right corner", FaceBox(280, 280, 200, 200)),
        ("20px from right edge", FaceBox(260, 140, 200, 200)),
        ("40px from right edge", FaceBox(240, 140, 200, 200)),
        ("face fills the frame", FaceBox(40, 40, 400, 400)),
    ]
    for label, box in cases:
        g = measure(480, 480, box)
        dx, dy = g.centring_error
        print(f"      {label:<26} {f'{g.crop_w}x{g.crop_h}':>10} {g.anisotropy:>8.4f} "
              f"{(g.anisotropy - 1) * 100:>8.1f}% {g.occupancy:>9.1%} {f'({dx},{dy})':>12}")

    print("\n      Read the LEFT and RIGHT rows together. Same face, same distance from")
    print("      the frame edge, two structurally different treatments:")
    print("        left  -> crop stays square, face slides 30px off centre  (framing error)")
    print("        right -> crop truncated to 230x260, face stays centred   (DISTORTION)")
    print("      The clamp at 0 and the truncate at w are not each other's mirror.")

    print("\n  (b) The exact activation condition, brute-forced.")
    print("      Truncation on the right occurs iff")
    print("          gap  <  ceil(side/2) - ceil(s/2)      where gap = W - (box right edge)")
    print(f"      For a square box of side s at margin {MARGIN} that is about 0.15*s.")
    mismatches = 0
    rng = np.random.default_rng(23)
    for _ in range(50_000):
        W, H = int(rng.integers(200, 900)), int(rng.integers(200, 900))
        s = int(rng.integers(64, min(W, H)))
        x, y = int(rng.integers(0, W - s + 1)), int(rng.integers(0, H - s + 1))
        g = measure(W, H, FaceBox(x, y, s, s))
        gap = W - (x + s)
        clamped = (x + s // 2 - g.side // 2) < 0
        thresh = math.ceil(g.side / 2) - math.ceil(s / 2)
        predicted = (g.side > W) if clamped else (gap < thresh)
        if (g.crop_w < g.side) != predicted:
            mismatches += 1
    print(f"      50 000 random square boxes: {mismatches} mismatches")
    print("      So: a face whose box edge is within ~15% of its own width of the frame")
    print("      border is scored on a stretched crop. That is an ordinary photograph,")
    print("      not a pathological one, and it needs no free-aspect detector.")

    print("\n  (c) FREE-ASPECT boxes with room to spare -- no clipping possible.")
    print(f"      {'box':<20} {'crop':>12} {'aniso':>8} {'occupancy':>10}")
    for label, box in [
        ("square 200x200", FaceBox(500, 500, 200, 200)),
        ("near-square 200x203", FaceBox(500, 500, 200, 203)),
        ("portrait 150x200", FaceBox(500, 500, 150, 200)),
        ("landscape 200x150", FaceBox(500, 500, 200, 150)),
        ("SCRFD-ish 150x200", FaceBox(500, 500, 150, 200)),
        ("extreme 900x50", FaceBox(500, 500, 900, 50)),
    ]:
        g = measure(2000, 2000, box)
        print(f"      {label:<20} {f'{g.crop_w}x{g.crop_h}':>12} {g.anisotropy:>8.4f} {g.occupancy:>9.1%}")
    print("\n      **Free aspect produces NO distortion.** The window is square by")
    print("      construction, so with room to spare the crop is square. What free")
    print("      aspect destroys is OCCUPANCY -- 59% of the model's input is face for a")
    print("      square box and 3.3% for a 900x50 one.")
    print()
    print("      So R23 as recorded conflates two defects with different triggers:")
    print("        occupancy collapse -- caused by free aspect     -- LATENT (needs SCRFD)")
    print("        aspect distortion  -- caused by border clipping -- LIVE TODAY")

    print("\n  (d) Both at once: a free-aspect box near a border.")
    for label, (W, H, box) in [
        ("900x50 face, room", (2000, 2000, FaceBox(500, 500, 900, 50))),
        ("900x50 face, at edge", (1100, 600, FaceBox(150, 250, 900, 50))),
    ]:
        g = measure(W, H, box)
        print(f"      {label:<24} crop={g.crop_w}x{g.crop_h} aniso={g.anisotropy:.4f} occ={g.occupancy:.1%}")


# --------------------------------------------------------------------------
# 4 — the fail-before / after pairing
# --------------------------------------------------------------------------


def s4_pairing() -> None:
    head(4, "Legacy against corrected", "does the correction eliminate the failure?")

    print(f"  legacy    : {LegacyClipToBounds.name}")
    print(f"  corrected : {PadToSquareCrop.name}\n")

    img = np.full((480, 480, 3), 128, np.uint8)
    tile = cv2.resize(face_tile(), (200, 200), interpolation=cv2.INTER_AREA)

    print(f"  {'position':<22} {'legacy crop':>13} {'aniso':>8} | {'corrected':>13} {'aniso':>8}")
    worst_legacy = 1.0
    for label, (x, y) in [
        ("centred", (140, 140)),
        ("flush left", (0, 140)),
        ("flush right", (280, 140)),
        ("flush bottom", (140, 280)),
        ("bottom-right", (280, 280)),
    ]:
        canvas = img.copy()
        canvas[y : y + 200, x : x + 200] = tile
        box = FaceBox(x, y, 200, 200)
        a = LegacyClipToBounds.crop(canvas, box)
        b = PadToSquareCrop.crop(canvas, box)
        an_a = max(a.shape[1] / a.shape[0], a.shape[0] / a.shape[1])
        an_b = max(b.shape[1] / b.shape[0], b.shape[0] / b.shape[1])
        worst_legacy = max(worst_legacy, an_a)
        print(f"  {label:<22} {f'{a.shape[1]}x{a.shape[0]}':>13} {an_a:>8.4f} | "
              f"{f'{b.shape[1]}x{b.shape[0]}':>13} {an_b:>8.4f}")

    print(f"\n  legacy worst anisotropy    : {worst_legacy:.4f}  ({(worst_legacy-1)*100:.1f}% stretch)")
    print("  corrected worst anisotropy : 1.0000  (by construction)")
    print()
    print("  WHY the correction eliminates it, in one sentence: the legacy code")
    print("  computes a window and then *intersects* it with the image, so the image")
    print("  bounds change the crop's SHAPE; the corrected code computes the same")
    print("  window and then *fills* it, so the bounds change only its CONTENT.")
    print("  Shape is what prepare_image's square resize is sensitive to.")

    print("\n  The same face, framed two ways, reaches the model as different pixels:")
    a_canvas = np.full((480, 480, 3), 128, np.uint8)
    a_canvas[140:340, 140:340] = tile
    b_canvas = np.full((480, 480, 3), 128, np.uint8)
    b_canvas[140:340, 280:480] = tile
    pa = stub_p(LegacyClipToBounds.crop(a_canvas, FaceBox(140, 140, 200, 200)))
    pb = stub_p(LegacyClipToBounds.crop(b_canvas, FaceBox(280, 140, 200, 200)))
    print(f"    identical face pixels, centred   -> stub p_fake {pa:.6f}")
    print(f"    identical face pixels, at edge   -> stub p_fake {pb:.6f}")
    print("    (StubBackend hashes its input, so the DIRECTION is arithmetic. What is")
    print("     established is that the model is handed a different image.)")


# --------------------------------------------------------------------------
# 5 — why the image scenarios are scripted
# --------------------------------------------------------------------------


def s5_haar() -> None:
    head(5, "Why scripted boxes", "can real Haar reach the border case?")

    tile = face_tile()
    print(f"  {'canvas':<28} {'detection':<22} {'crop':>12} {'aniso':>8}")
    for label, img in [
        ("480x480, the fixture", tile),
        ("460x480, trimmed 20px", tile[:, :460]),
        ("440x480, trimmed 40px", tile[:, :440]),
        ("400x480, trimmed 80px", tile[:, :400]),
    ]:
        found = pre.detect_faces(img, policy=POLICY)
        if not found:
            print(f"  {label:<28} {'no detection':<22} {'-':>12} {'-':>8}")
            continue
        b = found[0]
        crop = pre.crop_face(img, b, policy=POLICY)
        an = max(crop.shape[1] / crop.shape[0], crop.shape[0] / crop.shape[1])
        print(f"  {label:<28} {f'{b.width}x{b.height}@({b.x},{b.y})':<22} "
              f"{f'{crop.shape[1]}x{crop.shape[0]}':>12} {an:>8.4f}")

    print("\n  Haar loses the face the moment the canvas is trimmed -- R13's fixture")
    print("  fragility, measured again. So the border cases are scripted, with boxes")
    print("  of exactly the shape Haar emits: SQUARE. That is the point rather than a")
    print("  limitation, because it is what shows the defect needs no free aspect.")
    print()
    print("  Reachability without a fixture: Haar requires the whole frontal face")
    print("  inside the frame, so the box is always fully inside. Truncation then needs")
    print("  the box edge within ~15% of the box's own width of the frame border -- an")
    print("  off-centre subject, or any face wider than 1/1.3 = 77% of the frame.")
    print("  Neither is unusual in real photographs.")


# --------------------------------------------------------------------------
# 6 — the video path
# --------------------------------------------------------------------------


def s6_video() -> None:
    head(6, "The video path", "the largest geometry defect is not in the crop path")

    print("  extract_frames(..., target_size=self._spec.input_size) resizes each WHOLE")
    print("  FRAME to size x size. No detection, no crop, no aspect preservation.\n")
    print(f"  {'source':<26} {'model input':<14} {'anisotropy':>11}  effect")
    for label, (w, h) in [
        ("1920x1080  16:9", (1920, 1080)),
        ("1280x720   16:9", (1280, 720)),
        ("640x480     4:3", (640, 480)),
        ("1080x1920  9:16", (1080, 1920)),
        ("512x512     1:1", (512, 512)),
    ]:
        an = max(w / h, h / w)
        effect = "unchanged" if an == 1 else f"squashed {an:.3f}x on the long axis"
        print(f"  {label:<26} {f'{INPUT}x{INPUT}':<14} {an:>11.4f}  {effect}")

    print("\n  Measured on a test pattern rather than asserted:")
    pattern = np.zeros((1080, 1920, 3), np.uint8)
    cv2.circle(pattern, (960, 540), 400, (255, 255, 255), -1)
    out = pre.resize(pattern, INPUT)
    ys, xs = np.where(out[:, :, 0] > 127)
    w_px, h_px = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    print(f"    a CIRCLE of radius 400 in a 1920x1080 frame reaches the model as")
    print(f"    a {w_px} x {h_px} ELLIPSE -- axis ratio {w_px / h_px:.4f}, not 1.0")

    lb = LetterboxResize.resize(pattern, INPUT)
    ys, xs = np.where(lb[:, :, 0] > 127)
    print(f"    through {LetterboxResize.name}: "
          f"{xs.max()-xs.min()+1} x {ys.max()-ys.min()+1}, ratio "
          f"{(xs.max()-xs.min()+1)/(ys.max()-ys.min()+1):.4f}")

    print("\n  This is live on EVERY video bundle the platform has ever published, it")
    print("  is 1.778x on the commonest source aspect, and it dwarfs anything in the")
    print("  image path. It is not R23 -- R23 is about crop_face -- and it is recorded")
    print("  separately as R24.")


# --------------------------------------------------------------------------
# 7 — reproducibility
# --------------------------------------------------------------------------


def s7_reproducibility() -> None:
    head(7, "Reproducibility", "can two deployments agree on every identity and differ?")

    print("  *** SUPERSEDED BY BATCH 22 (R25 CLOSED). ***")
    print("  This stage measured the OPEN state: three geometry knobs outside every")
    print("  identity string. Batch 22 folded input_size into weights_identity and")
    print("  added interpolation_rule + crop_clip_rule to FaceSelectionPolicy, so the")
    print("  identities below now DIFFER (the 'identical:' lines print False) and the")
    print("  ANSWER is no longer 'yes'. See docs/CROP_IDENTITY.md and")
    print("  tests/test_crop_identity.py. The measurements are kept as the evidence")
    print("  Batch 22 was built against.\n")

    caps = pre.DEFAULT_CAPS
    base = (
        ModelSpec(name="stub-image", version="stub-0", backend="stub")
        .with_gate(caps.gate_fingerprint)
        .with_face_selection(POLICY.identity)
    )
    print("  Fingerprinted geometry knobs (Batch 12, R17):")
    print(f"    crop_margin = {POLICY.crop_margin}   -> in the .f. segment")
    print(f"    crop_shape  = {POLICY.crop_shape!r} -> in the .f. segment")
    print(f"    face-selection identity: {POLICY.identity}")
    print()
    print("  UNfingerprinted geometry knobs, each of which moves every scored pixel:\n")

    # 1. input_size
    a = ModelSpec(name="stub-image", version="stub-0", backend="stub", input_size=384)
    b = ModelSpec(name="stub-image", version="stub-0", backend="stub", input_size=224)
    a = a.with_gate(caps.gate_fingerprint).with_face_selection(POLICY.identity)
    b = b.with_gate(caps.gate_fingerprint).with_face_selection(POLICY.identity)
    print(f"    1. ModelSpec.input_size   384 -> {a.effective_version}")
    print(f"                              224 -> {b.effective_version}")
    print(f"       identical: {a.effective_version == b.effective_version}")
    crop = np.full((260, 260, 3), 100, np.uint8)
    crop[::3] = 200
    print(f"       and the model input differs: {pre.prepare_image(crop, size=384).shape} "
          f"vs {pre.prepare_image(crop, size=224).shape}")

    # 2. interpolation policy
    print("\n    2. the interpolation rule in preprocess.resize")
    src = np.random.default_rng(1).integers(0, 255, (300, 500, 3), dtype=np.uint8)
    area = cv2.resize(src, (384, 384), interpolation=cv2.INTER_AREA)
    linear = cv2.resize(src, (384, 384), interpolation=cv2.INTER_LINEAR)
    print(f"       a 500x300 crop -> 384: rule picks INTER_AREA (w > size)")
    print(f"       INTER_AREA vs INTER_LINEAR: mean |d| = "
          f"{float(np.abs(area.astype(int) - linear.astype(int)).mean()):.3f}/255, "
          f"identical = {np.array_equal(area, linear)}")
    print("       Neither the rule nor the choice is in any fingerprint. Note also")
    print("       that the rule is `h > size or w > size`, so a crop wider than the")
    print("       input but SHORTER than it is area-filtered on the axis being")
    print("       UPSAMPLED.")

    # 3. the clipping rule
    print("\n    3. the clipping / centring rule inside crop_face")
    img = np.full((480, 480, 3), 128, np.uint8)
    box = FaceBox(280, 140, 200, 200)
    leg = LegacyClipToBounds.crop(img, box)
    pad = PadToSquareCrop.crop(img, box)
    print(f"       shipped (clip)  -> {leg.shape[1]}x{leg.shape[0]}")
    print(f"       padded          -> {pad.shape[1]}x{pad.shape[0]}")
    print(f"       both are honestly describable as crop_shape={POLICY.crop_shape!r},")
    print(f"       so both produce face-selection identity {POLICY.identity}")

    print("\n  ANSWER (as of Batch 15): **yes.** Two deployments agreeing on weights,")
    print("  calibration, gate, face-selection and decode identity could hand the model")
    print("  different pixels and publish different verdicts, through any of the three")
    print("  knobs above. The shape of the defect was exactly R17's: verdict-affecting")
    print("  geometry sitting in a default argument, a module-level function and a")
    print("  dataclass field nobody thinks of as config. Recorded as R25 —")
    print("  **CLOSED in Batch 22**: input_size -> weights_identity, interpolation_rule")
    print("  and crop_clip_rule -> FaceSelectionPolicy.identity. R28 (the resize `or`")
    print("  predicate) deferred.")


# --------------------------------------------------------------------------
# 8 — SCRFD activation
# --------------------------------------------------------------------------


def s8_scrfd() -> None:
    head(8, "SCRFD migration", "what is latent, what is live, what would notice?")

    print(f"  {'behaviour':<40} {'today (Haar)':<22} under SCRFD")
    rows = [
        ("crop truncation at a border", "LIVE (square boxes)", "live, unchanged"),
        ("anisotropic stretch from truncation", "LIVE", "live, larger"),
        ("face off-centre at left/top border", "LIVE", "live, unchanged"),
        ("occupancy collapse from free aspect", "latent (aspect ~1)", "ACTIVE immediately"),
        ("crop mostly background", "no (59% face)", "ACTIVE (44% at 0.75 aspect)"),
        ("negative box coords clamped by face_track", "unreachable", "ACTIVE (regressed boxes)"),
        ("published bbox != scored region", "LIVE", "live, and now also clamped"),
        ("video whole-frame squash (R24)", "LIVE", "unchanged -- no face stage"),
    ]
    for a, b, c in rows:
        print(f"  {a:<40} {b:<22} {c}")

    print("\n  Occupancy at the aspect ratios a real face detector produces:")
    print(f"    {'w/h':<8} {'box':<14} {'crop':<14} occupancy")
    for ratio in (1.00, 0.90, 0.80, 0.75, 0.65):
        h = 200
        w = int(h * ratio)
        g = measure(2000, 2000, FaceBox(500, 500, w, h))
        print(f"    {ratio:<8.2f} {f'{w}x{h}':<14} {f'{g.crop_w}x{g.crop_h}':<14} {g.occupancy:.1%}")
    print("    A 0.75-aspect box -- ordinary for SCRFD -- puts 44% of the model's")
    print("    input on the face against 59% for a square one. Same margin, same")
    print("    fingerprint, a quarter less face.")

    print("\n  Which existing tests would notice? Measured by grep:")
    root = Path(__file__).resolve().parent.parent
    tokens = ("aniso", "crop.shape", "crop_face", "occupancy", "letterbox", "aspect")
    for path in sorted((root / "tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        hits = {t: text.count(t) for t in tokens if text.count(t)}
        if hits:
            print(f"    {path.name:<38} {hits}")
    print("\n  What those hits actually assert:")
    print("    test_preprocess           `crop.size > 0` and the crop fits the image.")
    print("                              Named for squareness; does not check it (stage 2).")
    print("    test_face_size_policy     Batch 14's R23 tripwire -- PINS today's behaviour,")
    print("                              including that a clipped crop is NOT square.")
    print("    test_face_selection_ident Crop BYTES, to prove the margin reaches the model.")
    print("    test_inference_determinism Crop bytes again, for repeat-invariance.")
    print()
    print("  So: nothing asserts that the resize is isotropic, nothing asserts occupancy,")
    print("  nothing exercises a border position with a non-corner box, and the one test")
    print("  named for crop shape checks everything except crop shape. Every assertion")
    print("  that touches geometry is a byte comparison or a tripwire pinning the status")
    print("  quo. **Activation under SCRFD would be silent.**")


# --------------------------------------------------------------------------


def main() -> int:
    print(__doc__)
    s1_flow()
    s2_assumptions()
    s3_distortion()
    s4_pairing()
    s5_haar()
    s6_video()
    s7_reproducibility()
    s8_scrfd()
    print(f"\n{'=' * 78}\ndone\n{'=' * 78}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
