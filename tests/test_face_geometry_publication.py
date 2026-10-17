"""R26 Tier 2 (option a) — the `face_track()` clamp, guarded not fixed (Batch 27).

THE FINDING, RE-DERIVED (not assumed)
--------------------------------------
`inference/postprocess.py:face_track()` publishes `FaceTrack.bbox` as
``BoundingBox(x=max(0, x), y=max(0, y), ...)`` — a defensive clamp that
exists because `BoundingBox.x`/`.y` are contract-constrained `ge=0`
(`contracts/common.py`). `inference/preprocess.py:crop_face()` computes an
entirely different quantity from the *centre* of a margin-expanded window,
clamped at a different point. Under a detector that ever regresses a box
whose origin is negative (a face partially outside the frame — SCRFD can do
this; `cv2.CascadeClassifier.detectMultiScale`'s own docstring says
detections "may be partially outside the original image"), the two
diverge, and `face_track()`'s clamp would silently misreport the detector's
true output rather than raise.

`docs/CROP_GEOMETRY.md` §3.2 already established, by a 50,000-random-box
brute force (Batch 15), that **Haar cannot detect a face partly outside the
frame — the box is always fully inside**. §8's SCRFD-migration table names
this exact row: *"negative box coords clamped by face_track | unreachable
(Haar) | ACTIVE (regressed boxes) under SCRFD."*

WHAT THIS BATCH DOES
---------------------
Nothing to the clamp itself — see `docs/R16_R26_SPEC_RESOLUTION.md`
Decision 1: closing this for real (loosening `BoundingBox`'s constraint,
"option b") is deferred until a real detector exists to decide what the
published value should become. This suite only **guards** the current,
already-true fact — Haar cannot trigger the divergence — so a future
regression is caught loudly instead of silently, and **names** the gap with
a permanent, deliberately-not-fixed demonstration (the same "measured and
rejected" pattern R28's `interpolation_rule` uses).

WHY THIS ONE FIXTURE, AND WHY ONLY ONE CONFIGURATION
-----------------------------------------------------
`_face_tile()` is the determinism study's synthetic face, reused verbatim
from `tests/test_face_selection_identity.py::_face_image` — chosen because
it is already proven ("detects reliably") rather than a fresh fixture that
might not trigger Haar at all. **Measured for this batch**: it is far more
fragile than "trimming loses it" (`docs/CROP_GEOMETRY.md` §5) suggests —
padding the canvas by as little as 10px on every side (480x480 -> 500x500,
face repositioned to stay centred) **also** loses the detection entirely,
and the module docstring above already warns the fixture is scale-sensitive
enough that a `scaleFactor` of 1.05 vs. the shipped default "loses the
fixture face outright." Constructing genuinely edge-adjacent variants of
this fixture that still detect is not achievable at a cost proportionate to
this batch (R16_R26_SPEC_RESOLUTION.md's own risk register named this exact
possibility: "PA1's fixture set is not representative enough... this
batch's claim rests primarily on `docs/CROP_GEOMETRY.md`'s own prior
50,000-box brute force, not on PA1 alone — PA1 is corroborating, not the
sole evidence"). PA1/PA2 below therefore exercise the **one** proven,
reliable configuration — a genuine, real `detect_faces()` call, not a
scripted double — as a live sanity check that the current code path really
does satisfy the invariant `docs/CROP_GEOMETRY.md` §3.2 already established
exhaustively, not as an independent re-proof of the underlying claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from inference import postprocess as post  # noqa: E402
from inference import preprocess as pre  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FINDING = "R26"


# --------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------


def _face_tile(size: int = 480) -> np.ndarray:
    """The determinism study's synthetic face, verbatim (`test_face_selection_
    identity.py::_face_image`) — it detects reliably. Not re-derived here;
    reused so this suite's detections are measured against a fixture already
    proven to trigger Haar, not a fresh one that might not."""
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (175, 160, 150), -1)
    cv2.circle(img, (200, 210), 16, (60, 60, 60), -1)
    cv2.circle(img, (280, 210), 16, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 300), (18, 30), 0, 0, 360, (150, 130, 120), -1)
    cv2.ellipse(img, (240, 340), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _reliable_detections() -> list:
    """The one proven-reliable configuration's real detections (§ module
    docstring — the fixture does not survive repositioning or padding)."""
    return pre.detect_faces(_face_tile())


# --------------------------------------------------------------------------
# 0. Permanent demonstration — the gap this batch names but does not fix
# --------------------------------------------------------------------------


def test_a_regressed_box_would_be_silently_misreported():
    """**Not a fail-before/pass-after pair — a permanent record of an
    accepted, deferred gap**, the `R28`/`interpolation_rule` pattern.

    A detector that ever regresses ``x = -20`` (a face whose left edge is
    outside the frame — unreachable under Haar, per this module's docstring)
    is published by ``face_track`` as ``x = 0``: a different number than the
    detector actually returned, with nothing on the wire to say so. This
    batch deliberately does not fix it (`docs/R16_R26_SPEC_RESOLUTION.md`
    Decision 1, option b, deferred) — this test exists so the gap has a
    concrete example instead of only a prose description, and so a future
    session "fixing" the clamp without reading this file breaks a named
    test rather than an unnoticed comment.
    """
    regressed = (-20, 5, 100, 100)
    track = post.face_track(regressed, p_fake=0.5)
    assert track.bbox.x == 0, "the clamp's behaviour moved — update this test, not just the code"
    assert track.bbox.x != regressed[0], (
        "face_track no longer diverges from its input — R26 Tier 2 option (b) "
        "may have shipped; this test (and its docstring) need updating, not deleting"
    )
    assert track.bbox.y == regressed[1], "y was already non-negative and must pass through unchanged"


# --------------------------------------------------------------------------
# 1. Pass-after — the current, already-true guarantee, guarded
# --------------------------------------------------------------------------


def test_haar_backed_detection_never_produces_a_negative_origin_box():
    """PA1. Live re-measurement, on this box, corroborating (not
    substituting for) `docs/CROP_GEOMETRY.md` §3.2's 50,000-box brute force.
    Uses the real, Haar-backed `detect_faces()`, not a scripted double."""
    boxes = _reliable_detections()
    assert boxes, "the fixture stopped detecting; this test proves nothing — fix the fixture"
    for box in boxes:
        assert box.x >= 0 and box.y >= 0, (
            f"Haar returned a negative-origin box {box} — R26 Tier 2's "
            "premise (docs/CROP_GEOMETRY.md §3.2) no longer holds; this is "
            "a real, live finding, not a test bug"
        )


def test_face_track_clamp_is_a_no_op_for_every_box_detect_faces_can_produce():
    """PA2. The structural "no-op" claim asserted directly: every box the
    real detector returns survives `face_track()` unchanged."""
    boxes = _reliable_detections()
    assert boxes, "the fixture stopped detecting; this test proves nothing — fix the fixture"
    for box in boxes:
        track = post.face_track((box.x, box.y, box.width, box.height), p_fake=0.5)
        assert track.bbox.x == box.x and track.bbox.y == box.y, (
            f"face_track's clamp changed a Haar-detected box {box} -> "
            f"{track.bbox} — it was supposed to be a no-op under this "
            "detector (docs/CROP_GEOMETRY.md §3.2)"
        )


def test_the_clamp_gap_is_tracked_in_the_reproducibility_audit():
    """PA3. Doc-linkage, the `test_r29_is_tracked_in_the_reproducibility_audit`
    pattern (`tests/test_video_selection_identity.py`)."""
    audit = (ROOT / "REPRODUCIBILITY_AUDIT.md").read_text(encoding="utf-8")
    low = audit.lower()
    assert f"| {FINDING} |" in audit, "R26 has no row in the summary table"
    assert "batch 27" in low
    # The clamp itself must not be claimed fixed — only guarded. Tier
    # 2-option-(b) and Tier 3 stay open; the row's own prose must say so.
    assert "option (b)" in low or "tier 2" in low or "not fixed" in low or "deferred" in low, (
        "the audit row no longer distinguishes the guarded gap from a real "
        "fix — re-check docs/R16_R26_SPEC_RESOLUTION.md Decision 1 is still "
        "reflected accurately"
    )


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
