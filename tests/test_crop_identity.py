"""R25 — crop-geometry inputs are now inside the model identity. Batch 22.

The pass-after suite for `docs/CROP_IDENTITY_B22_PLAN.md`. It replaces the four
Batch-15 tripwires in `tests/test_crop_geometry.py` §5, which asserted the *open*
state (the review found two of them did not actually fail after the fix — one had
no identity assertion, the other a fragile substring check).

Three knobs, three homes:

    input_size          -> ModelSpec.weights_identity   (functionally threaded)
    interpolation_rule   -> FaceSelectionPolicy.identity (label pinned by a test)
    crop_clip_rule       -> FaceSelectionPolicy.identity (label pinned by a test)

R28 (the `resize` `or`-predicate that area-filters an enlarged axis) is
DEFERRED — fixing it moves shipped pixels. B22 names the rule with the current
behaviour and pins that; the predicate fix rides a future `.f.`-moving batch.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.config import STUB_ENGINE_VERSION  # noqa: E402
from contracts.enums import CONTRACT_VERSION  # noqa: E402
from contracts.version import (  # noqa: E402
    compose_model_version,
    compose_runtime_version,
    model_identity_of,
    parse_model_version,
)
from evaluation.benchmark import BenchmarkReport, regression_gate  # noqa: E402
from evaluation.calibration_protocol import CALIBRATION_WITNESS_VERSION  # noqa: E402
from evaluation.benchmark import RUN_SCHEMA_VERSION  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.models import STUB_IMAGE, ModelSpec  # noqa: E402
from inference.preprocess import FaceBox  # noqa: E402

POLICY = pre.default_face_selection()
CAPS = pre.DEFAULT_CAPS
INPUT = STUB_IMAGE.input_size

# The values Batch 22 regenerated. If the tree moves again these move with it and
# `test_the_migration_moved_only_what_it_should` is where the "moved once" check
# lives.
B22_POLICY_IDENTITY = "251d22e9"  # the *face* policy identity — unchanged by B23
B22_MODEL_IDENTITY = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9"
B22_STUB_ENGINE_VERSION = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9.d.a97ccb68"
PRE_B22_MODEL_IDENTITY = "stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387"

# S8 Batch 23 (R29): the ``.f.`` segment became the composite of the face and
# video selection policies. The face policy identity (``251d22e9``) is unchanged;
# the ``.f.`` fingerprint is now ``e629642c`` and the model identity / engine
# literal moved with it. Nothing else in the version string moved.
B23_SELECTION_IDENTITY = "e629642c"
B23_MODEL_IDENTITY = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c"
B23_STUB_ENGINE_VERSION = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.a97ccb68"

# S8 Batch 26 (R30): the ``.d.`` segment moved (image codec libraries joined the
# decode identity). Kept separate from ``B23_STUB_ENGINE_VERSION`` above so the
# "total across all eras" parser test keeps exercising the B23-shaped string
# rather than silently retargeting to whatever the current literal is.
B26_STUB_ENGINE_VERSION = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.c9d4025c"


def _spec(input_size: int) -> ModelSpec:
    return (
        ModelSpec(name="stub-image", version="stub-0", backend="stub", input_size=input_size)
        .with_gate(CAPS.gate_fingerprint)
        .with_face_selection(POLICY.identity)
    )


def _face_tile(size: int = 480) -> np.ndarray:
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (175, 160, 150), -1)
    cv2.circle(img, (200, 210), 16, (60, 60, 60), -1)
    cv2.circle(img, (280, 210), 16, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 340), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    return cv2.GaussianBlur(img, (5, 5), 0)


# --------------------------------------------------------------------------
# 1. Positive identity-movement tests (replace the Batch-15 tripwires)
# --------------------------------------------------------------------------


def test_input_size_moves_the_weights_identity():
    """**R25 knob 1**, replaces `test_input_size_is_in_no_identity_string`.

    Two specs differing only in `input_size` must now differ in
    `weights_identity` and therefore in `effective_version`; nothing else moves.
    """
    a = ModelSpec(name="m", version="stub-0", backend="stub", input_size=384)
    b = ModelSpec(name="m", version="stub-0", backend="stub", input_size=224)
    assert a.weights_identity != b.weights_identity
    assert a.effective_version != b.effective_version

    pa, pb = parse_model_version(a.effective_version), parse_model_version(b.effective_version)
    assert pa.weights != pb.weights
    assert (pa.calibration, pa.gate, pa.face_selection) == (
        pb.calibration, pb.gate, pb.face_selection
    ), "input_size moved a segment other than weights"


def test_the_interpolation_rule_moves_the_face_selection_identity():
    """**R25 knob 2**, replaces `test_the_interpolation_policy_is_in_no_identity_string`.

    Two policies differing only in `interpolation_rule` produce different
    `.identity`. The old test's substring check would have stayed green.
    """
    other = replace(POLICY, interpolation_rule="linear-always")
    assert other.interpolation_rule != POLICY.interpolation_rule
    assert other.identity != POLICY.identity
    assert other.differences(POLICY) == [
        f"interpolation_rule: 'linear-always' != {POLICY.interpolation_rule!r}"
    ]


def test_the_clip_rule_moves_the_face_selection_identity():
    """**R25 knob 3**, replaces `test_the_clipping_rule_is_in_no_identity_string`.

    Two policies differing only in `crop_clip_rule` produce different
    `.identity`. The old test asserted only a pixel-shape difference, which the
    fix does not touch.
    """
    other = replace(POLICY, crop_clip_rule="pad-replicate")
    assert other.crop_clip_rule != POLICY.crop_clip_rule
    assert other.identity != POLICY.identity


def test_two_deployments_that_differ_on_geometry_now_get_different_identities():
    """Replaces `test_two_deployments_can_agree_on_every_identity_and_differ`.

    The Batch-15 statement of R25 was "same everything, different pixels". After
    Batch 22 the identities differ for every one of the three knobs.
    """
    assert _spec(384).effective_version != _spec(224).effective_version

    base = _spec(384)
    interp = base.with_face_selection(replace(POLICY, interpolation_rule="linear-always").identity)
    clip = base.with_face_selection(replace(POLICY, crop_clip_rule="pad-replicate").identity)
    assert base.effective_version != interp.effective_version
    assert base.effective_version != clip.effective_version


# --------------------------------------------------------------------------
# 2. Re-derivation protection — the label closures held to the behaviour
# --------------------------------------------------------------------------


def _name_says_area(h: int, w: int, size: int) -> bool:
    """What `"area-if-either-axis-exceeds-target-else-linear"` means, independently."""
    return h > size or w > size


def test_the_interpolation_rule_name_matches_resize():
    """`resize`'s branch must be exactly what the policy's name describes,
    including the mixed case `h > size xor w > size` that R28 objects to.

    Fails if `resize` changes its predicate (e.g. R28's future `or -> and`)
    without the policy string changing with it.
    """
    assert POLICY.interpolation_rule == "area-if-either-axis-exceeds-target-else-linear"
    rng = np.random.default_rng(22)
    size = INPUT
    for h, w in [
        (300, 500), (500, 300), (200, 200), (400, 400),
        (size, size), (size + 1, size - 1), (size - 1, size + 1), (600, 100),
    ]:
        src = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        want = cv2.INTER_AREA if _name_says_area(h, w, size) else cv2.INTER_LINEAR
        assert np.array_equal(
            pre.resize(src, size), cv2.resize(src, (size, size), interpolation=want)
        ), f"resize disagrees with its named rule at (h={h}, w={w}, size={size})"


def test_the_clip_rule_name_matches_crop_face():
    """`crop_face` must intersect its window with the image — nothing else —
    for the name `"intersect-image"` to be honest.

    Fails if `crop_face` starts padding (R23's future fix) without the string
    changing.
    """
    assert POLICY.crop_clip_rule == "intersect-image"
    img = np.full((480, 480, 3), 128, np.uint8)
    img[120:340, 260:480] = cv2.resize(_face_tile(), (220, 220), interpolation=cv2.INTER_AREA)
    for box in [
        FaceBox(140, 140, 200, 200),   # interior
        FaceBox(280, 140, 200, 200),   # flush right -> non-square under intersect
        FaceBox(0, 0, 200, 200),       # flush top-left
        FaceBox(300, 300, 160, 160),   # bottom-right corner
    ]:
        h, w = img.shape[:2]
        side = int(max(box.width, box.height) * POLICY.crop_margin)
        cx, cy = box.x + box.width // 2, box.y + box.height // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        expected = img[y0 : min(h, y0 + side), x0 : min(w, x0 + side)]
        assert np.array_equal(pre.crop_face(img, box, policy=POLICY), expected), (
            f"crop_face is not a plain intersect-with-image at box {box}"
        )


# --------------------------------------------------------------------------
# 3. Output-identical — the batch's headline guarantee, asserted
# --------------------------------------------------------------------------


def test_the_shipped_pixels_did_not_move():
    """R25 records geometry in the identity; it changes no geometry.

    `crop_face` and `resize` under the shipped policy produce byte-identical
    output to a verbatim reimplementation of the pre-B22 behaviour.
    """
    img = np.full((480, 480, 3), 128, np.uint8)
    img[100:360, 100:360] = cv2.resize(_face_tile(), (260, 260), interpolation=cv2.INTER_AREA)

    def legacy_crop(image, box, margin=POLICY.crop_margin):
        h, w = image.shape[:2]
        side = int(max(box.width, box.height) * margin)
        cx, cy = box.x + box.width // 2, box.y + box.height // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        return image[y0 : min(h, y0 + side), x0 : min(w, x0 + side)]

    def legacy_resize(image, size):
        h, w = image.shape[:2]
        interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
        return cv2.resize(image, (size, size), interpolation=interp)

    for box in [FaceBox(120, 120, 200, 200), FaceBox(300, 120, 200, 200), FaceBox(0, 0, 200, 200)]:
        assert np.array_equal(
            pre.crop_face(img, box, policy=POLICY), legacy_crop(img, box)
        ), f"crop_face output moved at {box}"

    for shape in [(260, 260), (230, 260), (500, 300), (400, 400), (100, 700)]:
        src = np.random.default_rng(sum(shape)).integers(0, 255, (*shape, 3), dtype=np.uint8)
        assert np.array_equal(pre.resize(src, INPUT), legacy_resize(src, INPUT)), (
            f"resize output moved at {shape}"
        )

    # And prepare_image, which composes resize.
    crop = pre.crop_face(img, FaceBox(120, 120, 200, 200), policy=POLICY)
    assert np.array_equal(
        pre.prepare_image(crop, size=INPUT)[0],
        legacy_resize(crop, INPUT).astype(np.float32),
    )


# --------------------------------------------------------------------------
# 4. Migration — moved exactly what it should, nothing else
# --------------------------------------------------------------------------


def test_the_migration_moved_only_what_it_should():
    assert CONTRACT_VERSION == "1.0.0"
    assert RUN_SCHEMA_VERSION == "1.2.0"
    assert CALIBRATION_WITNESS_VERSION == "1.3.0"

    # OpenAPI + generated schema: no version literal, no structural change.
    root = Path(__file__).resolve().parent.parent
    snap = (root / "backend/openapi.snapshot.json").read_text(encoding="utf-8")
    assert (
        "wacdd9240" not in snap
        and "251d22e9" not in snap
        and "e629642c" not in snap
        and "w7f45fada" not in snap
    )
    schema_files = list((root / "schema_out").glob("*.json"))
    assert len(schema_files) >= 10, (
        f"schema_out has only {len(schema_files)} files; the no-version-literal "
        "check below would pass vacuously"
    )
    for f in schema_files:
        assert "stub-0" not in f.read_text(encoding="utf-8")

    # The *face* policy identity did not move; the ``.f.`` segment became the
    # face+video composite in B23, and the model identity / engine literal moved
    # with it. B26 (R30) then moved only the ``.d.`` tail.
    assert POLICY.identity == B22_POLICY_IDENTITY
    assert model_identity_of(STUB_ENGINE_VERSION) == B23_MODEL_IDENTITY
    assert STUB_ENGINE_VERSION == B26_STUB_ENGINE_VERSION

    # weights, calibration and gate segments did not move.
    parts = parse_model_version(STUB_ENGINE_VERSION)
    assert parts.weights == "stub-0-wacdd9240"
    assert parts.calibration == "739c3fec"
    assert parts.gate == "ae232cba"
    assert parts.face_selection == B23_SELECTION_IDENTITY
    assert STUB_ENGINE_VERSION.endswith(".d.c9d4025c")  # R30, Batch 26


def test_parse_model_version_is_total_across_all_eras():
    eras = [
        "stub-0-w7f45fada+c.739c3fec",                                  # pre-B9
        "stub-0-w7f45fada+c.739c3fec.g.ae232cba",                       # pre-B12
        "stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.d662e22d",            # B12
        "stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387.d.c20a064a",  # B14-B19
        "stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387.d.a97ccb68",  # B20-B21
        B22_STUB_ENGINE_VERSION,                                         # B22
        B23_STUB_ENGINE_VERSION,                                         # B23 (composite .f.)
        B26_STUB_ENGINE_VERSION,                                         # B26 (R30, .d. moved)
    ]
    for v in eras:
        parts = parse_model_version(v)
        assert parts.compose() == model_identity_of(v)
        assert model_identity_of(compose_runtime_version(model=parts.compose(), decode="deadbeef")) == parts.compose()


def test_old_artifacts_still_load():
    """A stored EvidenceBundle whose `model.version` is a pre-B22 runtime string
    still validates — the field is an opaque `str` and B22 added no constraint.
    """
    import contracts.fixtures as fx
    from contracts.evidence import EvidenceBundle

    payload = fx.image_manipulated().model_dump(mode="json")
    payload["model"]["version"] = PRE_B22_MODEL_IDENTITY + ".d.c20a064a"
    restored = EvidenceBundle.model_validate(payload)
    assert restored.model.version == PRE_B22_MODEL_IDENTITY + ".d.c20a064a"


# --------------------------------------------------------------------------
# 5. Alignment + benchmark behaviour across the boundary
# --------------------------------------------------------------------------


def test_alignment_fails_on_an_input_size_mismatch():
    """A worker whose `input_size` differs from the configured literal fails at
    startup rather than sharing a cache namespace.
    """
    from workers.engine_registry import verify_version_alignment

    verify_version_alignment(STUB_ENGINE_VERSION)  # the real one passes

    stale_spec = (
        ModelSpec(name="stub-image", version="stub-0", backend="stub", input_size=224)
        .with_gate(CAPS.gate_fingerprint)
        .with_face_selection(POLICY.identity)
    )
    stale = compose_runtime_version(
        model=stale_spec.effective_version, decode=STUB_ENGINE_VERSION.split(".d.")[1]
    )
    assert stale != STUB_ENGINE_VERSION
    try:
        verify_version_alignment(stale)
    except RuntimeError as exc:
        assert "does not match the engines" in str(exc)
        return
    raise AssertionError("a mismatched input_size passed the alignment check")


def test_the_benchmark_gate_still_compares_across_the_boundary():
    """`regression_gate` does not short-circuit on `effective_model_version`
    inequality (only `dataset_fingerprint` and `decode_identity` do), so a
    pre/post-B22 pair is still compared numerically — and the numbers are
    identical under the shipped detector. `BenchmarkReport.validate()` does
    refuse the mismatched pair, which is the intended tightening.
    """
    from test_benchmark_reproducibility import bench, corpus

    items = corpus(400)
    pre_run = bench(items, version=PRE_B22_MODEL_IDENTITY)
    post_run = bench(items, version=B22_MODEL_IDENTITY)

    result = regression_gate(pre_run, post_run)
    assert result.passed, f"identical stub numbers should pass: {result.failures}"
    assert not any("different datasets" in f for f in result.failures)
    # The "same effective model version" note must NOT be present — the versions differ.
    assert not any("same effective model version" in n for n in result.notes)

    report = BenchmarkReport(primary=post_run, in_domain=pre_run, protocol="p/v1")
    problems = report.validate()
    assert any("different models" in p for p in problems)


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
