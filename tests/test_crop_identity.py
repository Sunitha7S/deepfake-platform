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

# SCRFD Integration Program, Phase 2 (docs/SCRFD_PHASE2_SPEC_RESOLUTION.md
# Decision 7): FaceSelectionPolicy gained confidence_threshold/
# nms_iou_threshold (both None under Haar). The *face* policy identity moved
# this time (251d22e9 -> b0591be3) — unlike B23, this is a structural move
# (a new key in the fingerprinted dict), not a value-relabelling — and the
# composite .f. segment moved with it. These are the *current* values, unlike
# the B22/B23/B26 constants above, which stay frozen for cross-era parsing.
PHASE2_POLICY_IDENTITY = "b0591be3"
PHASE2_SELECTION_IDENTITY = "f8244ad0"
PHASE2_MODEL_IDENTITY = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.f8244ad0"
PHASE2_STUB_ENGINE_VERSION = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.f8244ad0.d.c9d4025c"


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


def test_a_real_crop_produces_the_mixed_case_resize_predicate_objects_to():
    """R28 reachability (`docs/R23_R28_DESIGN.md` §3.2/§7, D1). Not a scripted
    array — the real, shipped `crop_face()`, fed a large, near-border
    `FaceBox` under the shipped policy. Reproduces the finding from first
    principles: a crop whose two dimensions straddle `input_size` is
    genuinely reachable through the pipeline's own code, not only through a
    hand-built array like `test_the_interpolation_rule_name_matches_resize`
    above uses. Proves nothing is fixed — see D2/D3 and
    `docs/R23_R28_DESIGN.md` §3.4 for why this stays unfixed.
    """
    img = np.full((1200, 1200, 3), 128, np.uint8)
    box = FaceBox(x=1000, y=400, width=350, height=350)  # large, near the right edge
    crop = pre.crop_face(img, box, policy=POLICY)
    h, w = crop.shape[:2]
    assert (h > INPUT) != (w > INPUT), (
        f"crop {w}x{h} does not straddle input_size={INPUT}; adjust the fixture — "
        "this test needs the mixed case to exist, not merely a stretched crop"
    )


def test_the_mixed_case_crop_gets_area_interpolation_on_its_upsampled_axis():
    """R28 reachability, D2. The label (`interpolation_rule`) and the live,
    reachable behaviour must agree on D1's own real output, not only on an
    isolated scripted array — closing the gap
    `test_the_interpolation_rule_name_matches_resize` leaves: that test
    proves the *name* matches `resize`'s predicate; this one proves the
    predicate actually fires on the pipeline's own real crop, and that the
    axis being enlarged (upsampled) is the one that incorrectly inherits
    `INTER_AREA` — R28's exact objection.
    """
    img = np.full((1200, 1200, 3), 128, np.uint8)
    box = FaceBox(x=1000, y=400, width=350, height=350)
    crop = pre.crop_face(img, box, policy=POLICY)
    h, w = crop.shape[:2]
    assert h > INPUT or w > INPUT, "the fixture stopped triggering the AREA branch"
    assert min(h, w) < INPUT < max(h, w), (
        "the fixture no longer straddles input_size on both sides"
    )
    resized = pre.resize(crop, INPUT)
    expected = cv2.resize(crop, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    assert np.array_equal(resized, expected), (
        "resize() took a different branch than the mixed-case predicate "
        "predicts for a real crop_face() output — R28's premise no longer "
        "holds, or the predicate changed without this test noticing"
    )


def test_r28_is_documented_as_live_not_hypothetical():
    """R28 reachability, D3. Doc-linkage, the
    `test_r29_is_tracked_in_the_reproducibility_audit` /
    `test_the_clamp_gap_is_tracked_in_the_reproducibility_audit` pattern
    (`tests/test_video_selection_identity.py`, `tests/
    test_face_geometry_publication.py`) — but scoped to the **R28 row
    itself**, not the whole document: `crop_face` and `real`/`reachable`
    both appear dozens of times elsewhere in this file's history sections
    (measured — `grep -c crop_face REPRODUCIBILITY_AUDIT.md` finds 15+ hits
    outside the R28 row alone), so a whole-document substring check would
    pass regardless of what the R28 row itself says.
    """
    import re

    audit = (Path(__file__).resolve().parent.parent / "REPRODUCIBILITY_AUDIT.md").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^\| R28 \|.*$", audit, re.MULTILINE)
    assert match, "R28 has no row in the summary table"
    row = match.group(0).lower()
    assert "crop_face" in row, (
        "the R28 row no longer names crop_face — re-check docs/R23_R28_DESIGN.md "
        "§3.2 is still reflected"
    )
    assert "real" in row or "reachable" in row, (
        "the R28 row no longer documents the defect as reachable through real "
        "pipeline output, not only a scripted example"
    )


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

    # The *face* policy identity did not move between B22 and B23; the ``.f.``
    # segment became the face+video composite in B23. SCRFD Integration
    # Program Phase 2 then moved both: FaceSelectionPolicy gained two fields.
    assert POLICY.identity == PHASE2_POLICY_IDENTITY
    assert model_identity_of(STUB_ENGINE_VERSION) == PHASE2_MODEL_IDENTITY
    assert STUB_ENGINE_VERSION == PHASE2_STUB_ENGINE_VERSION

    # weights, calibration and gate segments did not move.
    parts = parse_model_version(STUB_ENGINE_VERSION)
    assert parts.weights == "stub-0-wacdd9240"
    assert parts.calibration == "739c3fec"
    assert parts.gate == "ae232cba"
    assert parts.face_selection == PHASE2_SELECTION_IDENTITY
    assert STUB_ENGINE_VERSION.endswith(".d.c9d4025c")  # R30, Batch 26 (.d. unmoved by Phase 2)


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
