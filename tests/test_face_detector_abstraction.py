"""SCRFD Integration Program, Phase 1 — the detector abstraction layer.

WHAT THIS BATCH DOES
---------------------
Introduces `FaceDetector` (a `Protocol`, structural parallel of
`inference.models.ModelBackend`) and `HaarDetector` (its one, shipped
implementation) in `inference/preprocess.py`. `detect_faces()` becomes a
thin, **hard-wired** call to `HaarDetector` plus the existing
`face_order_key` sort — an extraction, not a reimplementation. See
`docs/SCRFD_PROGRAM_DESIGN.md` and
`docs/SCRFD_PHASE1_ABSTRACTION_VERIFICATION.md`.

WHAT THIS BATCH DOES NOT DO
----------------------------
No `SCRFDDetector`, no dispatch on `policy.detector`, no identity movement
(`FaceSelectionPolicy.identity`, `cascade_digest()`, `STUB_ENGINE_VERSION`
are all pinned unmoved by this suite), no contract/schema change. This is
the "zero behaviour change" phase; the tests below prove that, they do not
merely assert it.

WHY THESE FIVE TESTS
---------------------
`docs/SCRFD_PHASE1_ABSTRACTION_VERIFICATION.md` §1.4 found that three
module-level objects — `_face_cascade`, `_cascade_local`, `cascade_digest`
— are directly inspected by other suites (`test_inference_determinism.py`)
and must stay untouched, in place, by name; and that five test/study files
monkey-patch `inference.preprocess.detect_faces` by module-attribute
reassignment, which only keeps working if `detect_faces` stays a plain,
directly reassignable module function. Tests A–E below are exactly the
claims that verification made — proved here, not merely carried over.
"""

from __future__ import annotations

import inspect
import sys
import threading as _threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from inference import preprocess as pre  # noqa: E402
from inference.image_pipeline import _detect_and_measure  # noqa: E402

# Baseline identity values, read live immediately before Phase 1's edit
# (docs/SCRFD_PHASE1_ABSTRACTION_VERIFICATION.md's own baseline capture).
# This suite's own scope is Phase 1 only — Phase 1 itself moved neither value,
# which is what test_default_face_selection_identity_is_unmoved below proves.
#
# _BASELINE_CASCADE_DIGEST is still correct: nothing about the Haar cascade
# asset changed in Phase 2 either.
#
# _BASELINE_FACE_SELECTION_IDENTITY is **not** a permanent invariant, and
# SCRFD Integration Program Phase 2 (docs/SCRFD_PHASE2_SPEC_RESOLUTION.md
# Decision 7) is the batch that legitimately moves it next — FaceSelectionPolicy
# gained confidence_threshold/nms_iou_threshold, a structural identity move
# unrelated to Phase 1's own scope. Updated here to the post-Phase-2 value
# (251d22e9 -> b0591be3) rather than left green-but-false; this test's own
# purpose (proving Phase 1 moved nothing) is unaffected, since Phase 1's own
# diff is still zero regardless of what a later, independent batch does.
_BASELINE_CASCADE_DIGEST = "0f7d4527844eb514"
_BASELINE_FACE_SELECTION_IDENTITY = "b0591be3"


def _face_tile(size: int = 480) -> np.ndarray:
    """The determinism study's synthetic face, verbatim
    (`test_face_selection_identity.py::_face_image`,
    `test_face_geometry_publication.py::_face_tile`) — proven to detect
    reliably under the shipped Haar configuration. Reused rather than
    re-derived so this suite's detections are measured against a fixture
    already known to trigger the cascade."""
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (175, 160, 150), -1)
    cv2.circle(img, (200, 210), 16, (60, 60, 60), -1)
    cv2.circle(img, (280, 210), 16, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 300), (18, 30), 0, 0, 360, (150, 130, 120), -1)
    cv2.ellipse(img, (240, 340), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _noise_tile(size: int = 128, seed: int = 0) -> np.ndarray:
    """A faceless fixture — the empty-list path through the same call chain."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# A — HaarDetector actually satisfies the protocol it is shipped against
# --------------------------------------------------------------------------


def test_haar_detector_satisfies_the_face_detector_protocol():
    """The abstraction was declared *and* implemented against — not just
    named. `FaceDetector` is `@runtime_checkable`, so this is a real
    structural check, not a naming convention."""
    assert isinstance(pre.HaarDetector(), pre.FaceDetector), (
        "HaarDetector no longer structurally satisfies FaceDetector — check "
        "the detect() method's name and arity"
    )


# --------------------------------------------------------------------------
# B — the extraction changed nothing: HaarDetector + the shared sort ==
#     detect_faces, on both a face and a faceless fixture
# --------------------------------------------------------------------------


def test_haar_detector_and_detect_faces_agree_on_a_detected_face():
    policy = pre.default_face_selection()
    image = _face_tile()
    via_detector = sorted(
        pre.HaarDetector().detect(image, policy=policy), key=pre.face_order_key
    )
    via_dispatcher = pre.detect_faces(image, policy=policy)
    assert via_detector == via_dispatcher, (
        f"HaarDetector+sort disagrees with detect_faces: {via_detector!r} != "
        f"{via_dispatcher!r} — the extraction moved behaviour"
    )
    assert via_dispatcher, "the fixture stopped detecting; adjust it, not this test"


def test_haar_detector_and_detect_faces_agree_on_a_faceless_image():
    policy = pre.default_face_selection()
    image = _noise_tile()
    via_detector = sorted(
        pre.HaarDetector().detect(image, policy=policy), key=pre.face_order_key
    )
    via_dispatcher = pre.detect_faces(image, policy=policy)
    assert via_detector == via_dispatcher == [], (
        "empty-list agreement broke on a faceless fixture: "
        f"{via_detector!r} vs {via_dispatcher!r}"
    )


def test_haar_detector_reads_scale_factor_from_the_policy_not_a_default():
    """Regression pin for the exact R17 property this extraction must not
    lose: a policy's ``scale_factor`` still reaches ``detectMultiScale``,
    not a value baked into ``HaarDetector``. 1.05 is documented elsewhere
    (``inference/preprocess.py``'s own module docstring history, R17) to
    lose the fixture face outright — the same live behaviour this test
    checks survived the refactor."""
    from dataclasses import replace

    shipped = pre.default_face_selection()
    fragile = replace(shipped, scale_factor=1.05)
    image = _face_tile()
    assert pre.HaarDetector().detect(image, policy=shipped), (
        "the shipped policy stopped detecting the fixture face"
    )
    assert pre.HaarDetector().detect(image, policy=fragile) == [], (
        "HaarDetector ignored policy.scale_factor — it should have lost the "
        "face at 1.05, exactly as detect_faces always has"
    )


# --------------------------------------------------------------------------
# C — no identity movement
# --------------------------------------------------------------------------


def test_default_face_selection_identity_is_unmoved():
    assert pre.cascade_digest() == _BASELINE_CASCADE_DIGEST, (
        f"cascade_digest() moved to {pre.cascade_digest()!r} — this phase "
        "must not touch the cascade asset or its hash"
    )
    assert pre.default_face_selection().identity == _BASELINE_FACE_SELECTION_IDENTITY, (
        f"FaceSelectionPolicy.identity moved to "
        f"{pre.default_face_selection().identity!r} — this phase must not "
        "move .f., per requirement 4"
    )


# --------------------------------------------------------------------------
# D — detect_faces stays a plain, directly reassignable module function,
#     and the production call path (ImagePipeline's _detect_and_measure)
#     still resolves it by attribute lookup at call time, not a bound
#     reference captured earlier
# --------------------------------------------------------------------------


def test_detect_faces_stays_monkeypatchable_at_module_level():
    original = pre.detect_faces
    sentinel_box = pre.FaceBox(1, 2, 3, 4)

    def _scripted(image, *, policy=None):  # noqa: ARG001
        return [sentinel_box]

    pre.detect_faces = _scripted  # type: ignore[assignment]
    try:
        faces, _blur = _detect_and_measure(_noise_tile(), pre.default_face_selection())
        assert faces == [sentinel_box], (
            "ImagePipeline's call path did not see the monkey-patched "
            "detect_faces — the abstraction broke the pattern "
            "test_branch_b_semantics.py / test_headline_and_face_semantics.py "
            "/ test_face_size_policy.py / test_multi_face_study.py / "
            "test_branch_b_semantics_study.py all depend on"
        )
    finally:
        pre.detect_faces = original  # type: ignore[assignment]

    # And the patch is fully reverted: production behaviour is back to real
    # detection, not the sentinel.
    faces_after, _ = _detect_and_measure(_face_tile(), pre.default_face_selection())
    assert faces_after != [sentinel_box], "the monkey-patch was not fully reverted"


# --------------------------------------------------------------------------
# E — HaarDetector is an extraction, not a reimplementation: it must call
#     the existing _face_cascade(), not construct its own CascadeClassifier
# --------------------------------------------------------------------------


def test_haar_detector_calls_the_shared_face_cascade_not_its_own():
    src = inspect.getsource(pre.HaarDetector.detect)
    assert "_face_cascade()" in src, (
        "HaarDetector.detect no longer calls the shared _face_cascade() — "
        "if it loads its own CascadeClassifier, R15's per-thread cache and "
        "thread-safety fix no longer cover this code path"
    )
    assert "CascadeClassifier(" not in src, (
        "HaarDetector.detect constructs its own CascadeClassifier instead "
        "of reusing _face_cascade() — a reimplementation, not an extraction"
    )


def test_face_cascade_and_cascade_local_are_untouched():
    """The three objects `docs/SCRFD_PHASE1_ABSTRACTION_VERIFICATION.md`
    §1.4 named as load-bearing for other suites are still exactly what
    they were: module-level, and `_cascade_local` is still a
    `threading.local` (the same property `test_inference_determinism.py`
    asserts independently)."""
    assert callable(pre._face_cascade)
    assert callable(pre.cascade_digest)
    assert isinstance(pre._cascade_local, _threading.local)


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
