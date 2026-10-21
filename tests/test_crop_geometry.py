"""R23 — crop geometry, and what it does to the pixels the model is shown.

WHAT THIS BATCH IS
------------------
An **investigation**. Batch 15 changes no geometry. It measures what the shipped
code does, pins it, and proves against named doubles that the failures are
eliminable — so the SCRFD migration starts from a defensible baseline rather
than from a docstring.

WHAT R23 TURNED OUT TO BE
-------------------------
All three of the categories the brief offered, and the recorded finding split in
two along the way:

* **correctness, LIVE** — ``crop_face`` intersects its square window with the
  image, so a face near the right or bottom border yields a non-square crop that
  ``prepare_image`` then stretches to square. A 200x200 box flush with the right
  edge of a 480x480 image is stretched **13.0%**. This needs no free-aspect box:
  square Haar boxes produce it today.
* **correctness, LATENT** — a free-aspect box collapses *occupancy*: 59% of the
  model's input is face at aspect 1.0, 44% at 0.75, 3.3% at 900x50. This is the
  half of R23 that SCRFD arms, and it is a different defect with a different
  trigger from the one above.
* **reproducibility** — three geometry knobs move every scored pixel and sit in
  no identity string: ``ModelSpec.input_size``, the interpolation rule in
  ``resize``, and the clipping/centring rule inside ``crop_face``. R17's shape,
  recorded as **R25**.
* **documentation** — ``crop_face``'s docstring says it avoids the aspect
  distortion that stretches facial geometry. It does not. And
  ``test_crop_face_is_square_and_clipped_to_bounds`` is named for a property its
  body never checks, on the one fixture where the property holds by coincidence.

Two further findings recorded rather than fixed: **R24**, the video path resizes
whole frames to a square (1.778x squash on 16:9, live on every video bundle),
and **R26**, the published ``bbox`` is the detector's box rather than the region
the model actually scored.

WHY DOUBLES
-----------
``LegacyClipToBounds`` is the shipped ``crop_face``, verbatim, so the failure is
executed rather than described. ``PadToSquareCrop`` and ``LetterboxResize`` are
*corrected* geometry — not shipped — so "this is fixable, and here is why the
fix works" is a measurement rather than a promise. ``ScriptedDetector`` supplies
boxes because Haar loses the fixture the moment the canvas is trimmed (R13), and
the boxes it supplies are **square**, which is the point.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from inference import preprocess as pre  # noqa: E402
from inference.models import ModelSpec  # noqa: E402
from inference.preprocess import FaceBox  # noqa: E402

POLICY = pre.default_face_selection()
MARGIN = POLICY.crop_margin
INPUT = ModelSpec(name="x", version="v", backend="stub").input_size


# --------------------------------------------------------------------------
# The named doubles
# --------------------------------------------------------------------------


class LegacyClipToBounds:
    """The shipped ``crop_face``, verbatim: window ∩ image.

    Kept here so the defect is demonstrated against something that cannot
    silently start passing. ``test_the_legacy_double_matches_the_shipped_crop``
    pins it to the real function on a sweep of positions; if the shipped code is
    ever corrected, that test fails and says the double has stopped reproducing
    the behaviour it exists to reproduce.
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
    """Corrected crop: compute the window, then *fill* it instead of shrinking it.

    Same window as ``crop_face`` intends. Where it leaves the image the missing
    pixels are replicated from the edge rather than the window being cut down.

    The correction works because it changes *what the image bounds affect*: in
    the shipped code they change the crop's **shape**, and shape is exactly what
    ``prepare_image``'s square resize is sensitive to; here they change only its
    **content**. Two invariants follow and both are asserted below — the crop is
    always ``side x side``, and the face is always centred in it.

    **Not shipped.** Edge replication is one choice among several (§9 of
    ``docs/CROP_GEOMETRY.md``); which one a real detector should use is a
    modelling question this batch does not answer.
    """

    name = "square window padded by edge replication (corrected)"

    @staticmethod
    def crop(image: np.ndarray, box: FaceBox, margin: float = MARGIN) -> np.ndarray:
        h, w = image.shape[:2]
        side = int(max(box.width, box.height) * margin)
        cx, cy = box.x + box.width // 2, box.y + box.height // 2
        x0, y0 = cx - side // 2, cy - side // 2
        x1, y1 = x0 + side, y0 + side
        pad_l, pad_t = max(0, -x0), max(0, -y0)
        pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
        inner = image[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
        if pad_l or pad_r or pad_t or pad_b:
            inner = cv2.copyMakeBorder(inner, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        return inner


class LetterboxResize:
    """Corrected resize: one scale factor for both axes, pad the short one.

    The shipped ``resize`` scales each axis independently onto a square, so a
    16:9 video frame reaches the model squashed by 1.778x. **Not shipped.**
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
        return cv2.copyMakeBorder(small, top, size - nh - top, left, size - nw - left,
                                  cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _anisotropy(arr: np.ndarray) -> float:
    h, w = arr.shape[:2]
    return max(w / h, h / w)


def _blank(w: int = 480, h: int = 480) -> np.ndarray:
    return np.full((h, w, 3), 128, np.uint8)


def _face_tile(size: int = 480) -> np.ndarray:
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (175, 160, 150), -1)
    cv2.circle(img, (200, 210), 16, (60, 60, 60), -1)
    cv2.circle(img, (280, 210), 16, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 300), (18, 30), 0, 0, 360, (150, 130, 120), -1)
    cv2.ellipse(img, (240, 340), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _positions() -> list[tuple[str, FaceBox]]:
    return [
        ("centred", FaceBox(140, 140, 200, 200)),
        ("flush left", FaceBox(0, 140, 200, 200)),
        ("flush top", FaceBox(140, 0, 200, 200)),
        ("flush right", FaceBox(280, 140, 200, 200)),
        ("flush bottom", FaceBox(140, 280, 200, 200)),
        ("bottom-right corner", FaceBox(280, 280, 200, 200)),
        ("20px from the right", FaceBox(260, 140, 200, 200)),
    ]


# --------------------------------------------------------------------------
# 1. The doubles are faithful
# --------------------------------------------------------------------------


def test_the_legacy_double_matches_the_shipped_crop():
    """Everything below is a pairing, so the double must be the shipped code."""
    img = _blank()
    for label, box in _positions():
        a = pre.crop_face(img, box, policy=POLICY)
        b = LegacyClipToBounds.crop(img, box)
        assert a.shape == b.shape, f"{label}: double {b.shape} != shipped {a.shape}"
        assert np.array_equal(a, b), f"{label}: double and shipped disagree on pixels"


def test_the_legacy_double_still_reproduces_the_defect():
    """If clipping is ever corrected upstream, say so rather than passing quietly."""
    worst = max(_anisotropy(LegacyClipToBounds.crop(_blank(), b)) for _, b in _positions())
    assert worst > 1.0, (
        "the shipped crop no longer distorts on any swept position; R23's "
        "fail-before evidence has gone stale and these tests need rewriting"
    )


# --------------------------------------------------------------------------
# 2. The defect: distortion is live, and needs no free-aspect box
# --------------------------------------------------------------------------


def test_a_square_box_at_the_right_border_produces_a_non_square_crop():
    """The headline. Square box in, rectangle out, stretched to square after.

    This is the case the Batch 11 review called latent and SCRFD-armed. It is
    neither: the box is 200x200, exactly the shape Haar emits.
    """
    crop = pre.crop_face(_blank(), FaceBox(280, 140, 200, 200), policy=POLICY)
    assert crop.shape[:2] == (260, 230), f"expected a 230x260 crop, got {crop.shape[:2]}"
    assert _anisotropy(crop) > 1.13, (
        f"anisotropy {_anisotropy(crop):.4f}; the crop is stretched "
        f"{(_anisotropy(crop) - 1) * 100:.1f}% by prepare_image's square resize"
    )


def test_the_square_resize_turns_a_circle_into_an_ellipse():
    """Distortion measured on content, not on shape alone.

    A circle drawn inside the crop region reaches the model as an ellipse of
    exactly the crop's anisotropy. This is what "13% stretch" means to a model
    that reads facial geometry.
    """
    img = np.zeros((480, 480, 3), np.uint8)
    cv2.circle(img, (380, 240), 90, (255, 255, 255), -1)
    crop = pre.crop_face(img, FaceBox(280, 140, 200, 200), policy=POLICY)
    prepared = pre.prepare_image(crop, size=INPUT)[0]
    mask = prepared[:, :, 0] > 200
    ys, xs = np.where(mask)
    w_px = xs.max() - xs.min() + 1
    h_px = ys.max() - ys.min() + 1
    ratio = max(w_px / h_px, h_px / w_px)
    assert ratio > 1.10, (
        f"the circle came through as {w_px}x{h_px} (ratio {ratio:.4f}); if this "
        "is 1.0 the crop stopped being distorted and the finding is stale"
    )


def test_free_aspect_alone_does_not_distort():
    """The correction to R23 as recorded: free aspect collapses occupancy, not shape.

    With room to spare the window is square, so the crop is square, so the resize
    is isotropic — for every aspect ratio, including 900x50.
    """
    big = np.full((2000, 2000, 3), 128, np.uint8)
    for w, h in ((200, 200), (150, 200), (200, 150), (900, 50)):
        crop = pre.crop_face(big, FaceBox(500, 500, w, h), policy=POLICY)
        assert _anisotropy(crop) == 1.0, (
            f"a {w}x{h} box with room to spare produced a {crop.shape[1]}x"
            f"{crop.shape[0]} crop; free aspect was not supposed to distort"
        )


def test_free_aspect_collapses_occupancy():
    """The half of R23 that SCRFD does arm, quantified."""
    big = np.full((2000, 2000, 3), 128, np.uint8)
    occupancies = {}
    for label, (w, h) in (("square", (200, 200)), ("scrfd-ish", (150, 200)), ("extreme", (900, 50))):
        crop = pre.crop_face(big, FaceBox(500, 500, w, h), policy=POLICY)
        occupancies[label] = (w * h) / float(crop.shape[0] * crop.shape[1])
    assert occupancies["square"] > 0.59
    assert occupancies["scrfd-ish"] < 0.45, (
        f"a 0.75-aspect box puts {occupancies['scrfd-ish']:.1%} of the model input "
        "on the face"
    )
    assert occupancies["extreme"] < 0.04


def test_the_left_and_right_borders_are_handled_differently():
    """The asymmetry: a clamp on one side, a truncation on the other.

    The same face at the same distance from opposite borders is treated in two
    structurally different ways — one loses framing, the other loses shape.
    Neither is the other's mirror, and nothing documents either.
    """
    img = _blank()
    left = pre.crop_face(img, FaceBox(0, 140, 200, 200), policy=POLICY)
    right = pre.crop_face(img, FaceBox(280, 140, 200, 200), policy=POLICY)

    assert _anisotropy(left) == 1.0, "the left border was supposed to keep the crop square"
    assert _anisotropy(right) > 1.0, "the right border was supposed to truncate"

    # Left: square crop, face off centre. Right: truncated crop, face centred.
    side = int(200 * MARGIN)
    left_cx = (0 + 100) - max(0, 100 - side // 2)
    assert left_cx != left.shape[1] // 2, "the face should be off centre at the left border"
    right_cx = (280 + 100) - max(0, 380 - side // 2)
    assert abs(right_cx - side // 2) <= 1, "the face should stay centred at the right border"


def test_the_truncation_condition_is_exact():
    """When does it fire? A closed form, brute-forced over 20 000 random cases.

    Truncation on the right occurs iff the gap between the box's right edge and
    the frame is smaller than ``ceil(side/2) - ceil(s/2)`` — about 15% of the
    box's own width at the shipped margin. Ordinary framing, not a pathology.
    """
    rng = np.random.default_rng(23)
    mismatches = 0
    for _ in range(20_000):
        W, H = int(rng.integers(200, 900)), int(rng.integers(200, 900))
        s = int(rng.integers(64, min(W, H)))
        x, y = int(rng.integers(0, W - s + 1)), int(rng.integers(0, H - s + 1))
        crop = pre.crop_face(np.zeros((H, W, 3), np.uint8), FaceBox(x, y, s, s), policy=POLICY)
        side = int(s * MARGIN)
        clamped = (x + s // 2 - side // 2) < 0
        predicted = (side > W) if clamped else (
            (W - (x + s)) < math.ceil(side / 2) - math.ceil(s / 2)
        )
        if (crop.shape[1] < side) != predicted:
            mismatches += 1
    assert mismatches == 0, f"{mismatches} cases the closed form did not predict"


# --------------------------------------------------------------------------
# 3. The correction, and why it works
# --------------------------------------------------------------------------


def test_the_corrected_crop_is_always_square():
    """Fail-before/after, on the same sweep. The legacy double fails this."""
    img = _blank()
    for label, box in _positions():
        corrected = PadToSquareCrop.crop(img, box)
        assert _anisotropy(corrected) == 1.0, f"{label}: corrected crop {corrected.shape[:2]}"
        side = int(max(box.width, box.height) * MARGIN)
        assert corrected.shape[:2] == (side, side), f"{label}: not the intended window"

    failures = [
        label for label, box in _positions()
        if _anisotropy(LegacyClipToBounds.crop(img, box)) > 1.0
    ]
    assert failures, "the legacy double distorted on no swept position"


def test_the_corrected_crop_always_centres_the_face():
    """The second invariant the shipped code loses, at the left/top border."""
    img = _blank()
    for label, box in _positions():
        corrected = PadToSquareCrop.crop(img, box)
        side = int(max(box.width, box.height) * MARGIN)
        cx = box.x + box.width // 2
        offset_in_crop = cx - (cx - side // 2)
        assert abs(offset_in_crop - corrected.shape[1] // 2) <= 1, (
            f"{label}: the face is not centred in the corrected crop"
        )


def test_why_the_correction_works_bounds_change_content_not_shape():
    """The mechanism, asserted rather than asserted-about.

    Legacy: image bounds enter the crop's SHAPE (via ``min``/``max`` on the
    slice), and shape is what the square resize is sensitive to.
    Corrected: bounds enter only the crop's CONTENT (via padding), and the shape
    is fixed before the image is consulted at all.
    """
    box = FaceBox(280, 140, 200, 200)
    side = int(200 * MARGIN)
    shapes_legacy = {
        LegacyClipToBounds.crop(_blank(w, 480), box).shape[:2] for w in (480, 520, 560, 600)
    }
    shapes_corrected = {
        PadToSquareCrop.crop(_blank(w, 480), box).shape[:2] for w in (480, 520, 560, 600)
    }
    assert len(shapes_legacy) > 1, (
        f"legacy crop shape should depend on image width; got {shapes_legacy}"
    )
    assert shapes_corrected == {(side, side)}, (
        f"corrected crop shape must not depend on image width; got {shapes_corrected}"
    )


def test_the_correction_changes_the_pixels_the_model_sees():
    """A correction that changed nothing would not be a correction."""
    img = _blank()
    tile = cv2.resize(_face_tile(), (200, 200), interpolation=cv2.INTER_AREA)
    img[140:340, 280:480] = tile
    box = FaceBox(280, 140, 200, 200)

    legacy = pre.prepare_image(LegacyClipToBounds.crop(img, box), size=INPUT)
    corrected = pre.prepare_image(PadToSquareCrop.crop(img, box), size=INPUT)
    assert legacy.shape == corrected.shape == (1, INPUT, INPUT, 3)
    assert not np.array_equal(legacy, corrected), (
        "the corrected crop reaches the model as identical pixels; the pairing "
        "proves nothing"
    )


# --------------------------------------------------------------------------
# 4. R24 — the video path squashes whole frames
# --------------------------------------------------------------------------


def test_the_video_path_resizes_whole_frames_to_a_square():
    """**R24.** Live on every video bundle, and larger than anything in §2.

    ``extract_frames(target_size=input_size)`` calls ``resize`` on the whole
    frame, and ``resize`` scales each axis independently onto a square. A 16:9
    frame is squashed 1.778x. Pinned, not fixed — a video face stage is R18/SCRFD
    work and out of scope by brief.
    """
    source = (
        Path(__file__).resolve().parent.parent / "inference/video_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "target_size=self._spec.input_size" in source, (
        "the video pipeline stopped resizing whole frames; R24 needs re-measuring"
    )

    pattern = np.zeros((1080, 1920, 3), np.uint8)
    cv2.circle(pattern, (960, 540), 400, (255, 255, 255), -1)
    out = pre.resize(pattern, INPUT)
    ys, xs = np.where(out[:, :, 0] > 127)
    w_px, h_px = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    assert h_px > w_px * 1.6, (
        f"a circle came through {w_px}x{h_px}; R24 claims a 1.778x squash on 16:9"
    )


def test_the_letterbox_double_preserves_the_circle():
    """The pairing for R24: one scale factor, and the circle survives."""
    pattern = np.zeros((1080, 1920, 3), np.uint8)
    cv2.circle(pattern, (960, 540), 400, (255, 255, 255), -1)
    out = LetterboxResize.resize(pattern, INPUT)
    ys, xs = np.where(out[:, :, 0] > 127)
    w_px, h_px = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    assert abs(w_px - h_px) <= 2, f"letterbox produced {w_px}x{h_px}, expected a circle"


# --------------------------------------------------------------------------
# 5. R25 — CLOSED in Batch 22. Identity coverage now lives in
#    ``tests/test_crop_identity.py``.
# --------------------------------------------------------------------------
#
# Batch 15 opened R25 with four tripwires here that asserted the *open* state
# (input_size / interpolation rule / clip rule outside every identity string).
# Batch 22 folded ``input_size`` into ``weights_identity`` and added
# ``interpolation_rule`` and ``crop_clip_rule`` to ``FaceSelectionPolicy``, so
# those tripwires are superseded:
#
#   test_input_size_is_in_no_identity_string
#       -> test_crop_identity.py::test_input_size_moves_the_weights_identity
#   test_the_interpolation_policy_is_in_no_identity_string   (REPLACED — its only
#       identity signal was a fragile substring check that a field named
#       ``interpolation_rule`` does not trip)
#       -> test_crop_identity.py::test_the_interpolation_rule_moves_the_face_selection_identity
#   test_the_clipping_rule_is_in_no_identity_string          (REPLACED — it had
#       no identity assertion at all, only a pixel-shape comparison unchanged by
#       the fix)
#       -> test_crop_identity.py::test_the_clip_rule_moves_the_face_selection_identity
#   test_two_deployments_can_agree_on_every_identity_and_differ
#       -> test_crop_identity.py::test_two_deployments_that_differ_on_geometry_now_get_different_identities
#
# The geometry defects themselves (R23 border distortion, R24 video squash, R26
# published-box) are still open and still pinned by the sections above/below.


# --------------------------------------------------------------------------
# 6. Documentation defects
# --------------------------------------------------------------------------


def test_the_crop_docstring_no_longer_claims_distortion_is_avoided():
    """The one thing Batch 15 corrected. Prose only; no behaviour changed.

    The pre-Batch-15 sentence, recorded verbatim so "the docstring was corrected"
    is checked against what it actually said.
    """
    legacy_claim = "avoids the aspect distortion that stretches facial geometry"
    doc = pre.crop_face.__doc__ or ""
    assert legacy_claim not in doc, (
        "crop_face still claims to avoid aspect distortion; it does not — a "
        "200x200 box at the right border is stretched 13%"
    )
    assert "clip" in doc.lower() and "distort" in doc.lower(), (
        "the corrected docstring should say what actually happens at a border"
    )


def test_the_existing_square_named_test_is_renamed_or_true():
    """``test_crop_face_is_square_and_clipped_to_bounds`` claimed what it never checked.

    Its fixture — ``FaceBox(90, 90, 40, 40)`` in a 100x100 image — sits in the one
    corner where both axes clip by the same amount, so the crop is square by
    coincidence. Move the same box off the corner and it is 16x52.
    """
    import test_preprocess

    assert not hasattr(test_preprocess, "test_crop_face_is_square_and_clipped_to_bounds"), (
        "the test named for squareness is back; either it asserts squareness "
        "(which fails at a border) or it should not claim it"
    )

    img = np.zeros((100, 100, 3), np.uint8)
    corner = pre.crop_face(img, FaceBox(90, 90, 40, 40), policy=POLICY)
    edge = pre.crop_face(img, FaceBox(90, 40, 40, 40), policy=POLICY)
    assert corner.shape[0] == corner.shape[1], "the old fixture was square by coincidence"
    assert edge.shape[:2] == (52, 16), (
        f"the same box off the corner should be 16x52, got {edge.shape[:2]}"
    )
    assert _anisotropy(edge) > 3.0


# --------------------------------------------------------------------------
# 7. R26 and the limits this batch does not close
# --------------------------------------------------------------------------


def test_the_published_bbox_is_not_the_scored_region():
    """**R26.** The bundle records the detector's box; the model saw a different one.

    Nothing in ``EvidenceBundle`` carries the crop window, so a client drawing
    ``faces[0].bbox`` draws a region that is neither what was scored nor, at a
    border, the same shape as what was scored.
    """
    from inference import postprocess as post

    box = FaceBox(280, 140, 200, 200)
    track = post.face_track((box.x, box.y, box.width, box.height), p_fake=0.5)
    assert (track.bbox.width, track.bbox.height) == (200, 200)

    crop = pre.crop_face(_blank(), box, policy=POLICY)
    assert crop.shape[:2] != (200, 200), "the scored region and the published box coincide"

    fields = set(type(track).model_fields)
    assert not {"crop", "crop_bbox", "scored_region"} & fields, (
        "a scored-region field appeared on FaceTrack; R26 is closed and this "
        "test should be replaced"
    )


def test_face_track_clamps_coordinates_that_crop_face_does_not():
    """R26's second half, latent: the publisher clamps, the cropper does not.

    ``face_track`` forces ``x, y >= 0`` because ``BoundingBox`` requires it, while
    ``crop_face`` centres on the raw box. Unreachable under Haar, which cannot
    emit a box outside the image; SCRFD regresses box corners and can.
    """
    from inference import postprocess as post

    track = post.face_track((-20, -10, 200, 200), p_fake=0.5)
    assert (track.bbox.x, track.bbox.y) == (0, 0), "the clamp is gone; re-measure R26"
    assert FaceBox(-20, -10, 200, 200).x == -20, "FaceBox does not clamp, and should not"


def test_batch_22_moved_the_identity_not_the_pixels():
    """R25 (Batch 22): the crop-geometry knobs entered the identity and **no
    shipped pixel moved**.

    Batch 15 was an investigation that changed nothing; its assertion here was
    ``POLICY.identity == "19559387"``. Batch 22 folded ``input_size`` into
    ``weights_identity`` and added ``interpolation_rule`` / ``crop_clip_rule`` to
    the policy, so the identity moved on purpose — while ``crop_margin``,
    ``crop_shape`` and every crop/resize output stayed exactly where they were.
    """
    from backend.config import STUB_ENGINE_VERSION
    from contracts.enums import CONTRACT_VERSION
    from contracts.version import model_identity_of

    # Geometry parameters that decide pixels are untouched.
    assert POLICY.crop_margin == 1.3
    assert POLICY.crop_shape == "square-longest-edge"

    # The identity moved to the Batch 22 values, then again in the SCRFD
    # Integration Program Phase 2 (confidence_threshold/nms_iou_threshold
    # added, both None under Haar — docs/SCRFD_PHASE2_SPEC_RESOLUTION.md
    # Decision 7).
    assert POLICY.identity == "b0591be3", "the face-selection identity is not the Phase-2 value"
    # The *face* policy identity did not move in B23; the ``.f.`` segment did,
    # because B23 (R29) made it the composite of the face and video selection
    # policies. See test_video_selection_identity.py.
    assert model_identity_of(STUB_ENGINE_VERSION) == (
        "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.f8244ad0"
    ), "the model identity is not the current (Phase-2) value"
    assert ".d." in STUB_ENGINE_VERSION

    # Output-identical: the shipped crop and resize produce the same bytes as the
    # verbatim pre-B22 crop and a plain square resize.
    img = _blank()
    img[140:340, 280:480] = cv2.resize(_face_tile(), (200, 200), interpolation=cv2.INTER_AREA)
    box = FaceBox(280, 140, 200, 200)
    assert np.array_equal(
        pre.crop_face(img, box, policy=POLICY), LegacyClipToBounds.crop(img, box)
    ), "crop_face output moved"
    src = np.random.default_rng(0).integers(0, 255, (300, 500, 3), dtype=np.uint8)
    interp = cv2.INTER_AREA if (300 > INPUT or 500 > INPUT) else cv2.INTER_LINEAR
    assert np.array_equal(
        pre.resize(src, INPUT), cv2.resize(src, (INPUT, INPUT), interpolation=interp)
    ), "resize output moved"

    assert CONTRACT_VERSION == "1.0.0"


# --------------------------------------------------------------------------


def _run() -> int:
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failed += 1
                print(f"  FAIL {name}: {exc}")
            except Exception as exc:  # pragma: no cover - surfaces real breakage
                failed += 1
                print(f"  ERR  {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
