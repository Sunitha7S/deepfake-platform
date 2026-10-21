"""R22 — branch B is declared "whole image" and scored on the face crop.

THE FINDING, RE-DERIVED
----------------------
``contracts/enums.py`` (comment), ``contracts/branch.py`` (``BRANCH_DESCRIPTIONS``)
and ``README.md`` declare branch B as *"DINOv2-B/14 ... whole image"*, against
branch A's *"aligned face crop"*. The ``ModelBackend.score`` interface takes one
tensor and returns one logit vector per branch, and both pipelines call it once:

* **image path** — ``_crop_and_score`` scores ``prepare_image(crop_face(...))``,
  so branch B sees **the face crop**, not the whole image;
* **video path** — ``_score_batch`` scores ``resize(whole frame)``, so branch B
  sees the whole frame *and so does branch A*, whose "aligned face crop" claim is
  the one violated there.

Nothing on the wire carries the claim: ``BranchResult`` has no description field,
``BRANCH_DESCRIPTIONS`` has no functional consumer, and "whole image" appears in
no schema and no fingerprint. But the claim has a shipped, executable
consequence: ``contracts/fixtures.py::image_no_face`` — a **published OpenAPI
example** — depicts a faceless image where branch A abstains, the gate passes,
branch B scores at weight 0.94 and the verdict is ``manipulated``. The real
pipeline abstains that same input entirely. Both
``test_contracts::test_no_face_image_abstains_on_branch_a_only`` and
``test_inference::test_no_face_path_abstains`` pass today.

WHAT THIS BATCH DOES
--------------------
Closes R22 as a **contract clarification**, the shape Batch 13 used for R18's
false-claim half:

* the claims now state what the code does — branch B scores the same prepared
  tensor as branch A — and name whole-image scoring as the frozen-architecture
  target it is;
* ``image_no_face`` is annotated as depicting that target, not current behaviour,
  with the pre-Batch-17 docstring preserved verbatim in ``LegacyBranchBClaim``;
* the current pipeline behaviour (full abstention on a faceless image, branch B
  fed the crop) is pinned so a future whole-image B cannot land silently.

No behaviour changes. No field, schema, fingerprint or ``CONTRACT_VERSION`` moves.

WHY DOUBLES
-----------
``TensorCaptureBackend`` records the exact tensor each ``score()`` call receives.
``ScriptedDetector`` places a face in a corner of a large image so the crop and
the whole image are provably different pixels. ``LegacyBranchBClaim`` holds the
description strings as of Batch 16, verbatim, so "the claim was corrected" is
checked against what it said.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.branch import BRANCH_DESCRIPTIONS  # noqa: E402
import contracts.fixtures as fx  # noqa: E402
from contracts.enums import AbstainReason, BranchId, BranchStatus, MediaKind, Verdict  # noqa: E402
from contracts.evidence import EvidenceBundle  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.image_pipeline import ImagePipeline  # noqa: E402
from inference.models import ModelBackend, ModelRegistry, ModelSpec  # noqa: E402
from inference.postprocess import BranchOutcome, fuse, manipulation_evidence  # noqa: E402
from inference.video_pipeline import VideoPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Named doubles
# --------------------------------------------------------------------------


class LegacyBranchBClaim:
    """The branch descriptions and the enums comment as of Batch 16, verbatim."""

    name = "branch descriptions pre-Batch-17"
    A = "ConvNeXt V2-T over the aligned face crop, with SRM/Bayar residual channels."
    B = "DINOv2-B/14 frozen features with a trained probe, over the whole image."
    ENUM_B_CLAIM = "whole image"
    FIXTURE_DOCSTRING = "A landscape. Branch A abstains; the semantic branch carries the verdict alone."


class TensorCaptureBackend:
    """Records every ``batch`` handed to ``score()``. Returns a fixed logit."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.calls: list[np.ndarray] = []

    def score(self, batch: np.ndarray) -> dict[BranchId, np.ndarray]:
        self.calls.append(np.array(batch, copy=True))
        n = batch.shape[0]
        return {
            BranchId.A: np.full(n, 1.0),
            BranchId.B: np.full(n, 1.0),
            BranchId.P: np.zeros(n),
        }

    def close(self) -> None:
        return None


class ScriptedDetector:
    """Replaces ``preprocess.detect_faces``. Returns a fixed box list."""

    def __init__(self, boxes: list[pre.FaceBox]) -> None:
        self.boxes = boxes

    def __call__(self, image, *, policy=None, min_px=None):
        return sorted(self.boxes, key=pre.face_order_key)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _spec(data: bytes, kind: MediaKind = MediaKind.IMAGE) -> JobSpec:
    return JobSpec(
        job_id=UUID(int=22),
        media_kind=kind,
        storage_key="uploads/x",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="image/png" if kind is MediaKind.IMAGE else "video/mp4",
        model_version="stub-0",
        idempotency_key="k",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _corner_face_png(w: int = 900, h: int = 700, box=(60, 60, 180, 180)) -> tuple[bytes, pre.FaceBox]:
    rng = np.random.default_rng(22)
    img = (rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) // 6
    x, y, bw, bh = box
    img[y : y + bh, x : x + bw] = 230
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes(), pre.FaceBox(x=x, y=y, width=bw, height=bh)


def _faceless_png(w: int = 1024, h: int = 768) -> bytes:
    rng = np.random.default_rng(4)
    ok, buf = cv2.imencode(".png", rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    return buf.tobytes()


def _noise_clip(frames: int = 20, size: int = 160) -> bytes:
    import tempfile

    path = Path(tempfile.mkdtemp(prefix="dfp-r22-")) / "c.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (size, size))
    rng = np.random.default_rng(7)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _run_image_capturing(image_bytes: bytes, boxes: list[pre.FaceBox]):
    reg = ModelRegistry()
    reg.register(ModelSpec(name="cap", version="stub-0", backend="stub", input_size=384))
    eng = ImagePipeline(model_name="cap", registry=reg)
    loaded = reg.load_model("cap")
    cap = TensorCaptureBackend(reg.get_spec("cap"))
    loaded.backend = cap
    original = pre.detect_faces
    pre.detect_faces = ScriptedDetector(boxes)  # type: ignore[assignment]
    try:
        bundle = asyncio.run(eng.analyze_image(_spec(image_bytes), image_bytes))
    finally:
        pre.detect_faces = original  # type: ignore[assignment]
    return bundle, cap


def _run_video_capturing(clip: bytes):
    reg = ModelRegistry()
    reg.register(ModelSpec(name="capv", version="stub-0", backend="stub", input_size=384))
    eng = VideoPipeline(model_name="capv", registry=reg, n_frames=16)
    loaded = reg.load_model("capv")
    cap = TensorCaptureBackend(reg.get_spec("capv"))
    loaded.backend = cap
    bundle = asyncio.run(eng.analyze_video(_spec(clip, MediaKind.VIDEO), clip))
    return bundle, cap


# --------------------------------------------------------------------------
# 0. The doubles are faithful
# --------------------------------------------------------------------------


def test_the_legacy_claim_double_still_says_whole_image():
    assert "whole image" in LegacyBranchBClaim.B
    assert "aligned face crop" in LegacyBranchBClaim.A


# --------------------------------------------------------------------------
# 1. What branch B actually consumes — measured
# --------------------------------------------------------------------------


def test_branch_b_scores_the_face_crop_in_the_image_path():
    """Not the whole image. The tensor B is scored from is the crop, byte for byte."""
    data, box = _corner_face_png()
    bundle, cap = _run_image_capturing(data, [box])
    assert len(cap.calls) == 1, f"expected one score() call, got {len(cap.calls)}"
    seen = cap.calls[0]

    img = pre.decode_image(data)
    as_crop = pre.prepare_image(pre.crop_face(img, box, policy=pre.default_face_selection()), size=384)
    as_whole = pre.prepare_image(img, size=384)

    assert _digest(seen) == _digest(as_crop), "branch B was not scored from the face crop"
    assert _digest(seen) != _digest(as_whole), (
        "the crop and the whole image are the same pixels — the fixture is degenerate"
    )


def test_branch_a_and_b_come_from_one_backend_call():
    """One tensor produces every branch. There is no second input for B."""
    data, box = _corner_face_png()
    bundle, cap = _run_image_capturing(data, [box])
    assert len(cap.calls) == 1
    a, b = bundle.branch(BranchId.A), bundle.branch(BranchId.B)
    assert a.status is BranchStatus.SCORED and b.status is BranchStatus.SCORED


def test_the_model_backend_interface_cannot_split_the_input():
    """``score`` takes exactly one array; the call sites pass exactly one arg."""
    sig = inspect.signature(ModelBackend.score)
    array_params = [p for p in sig.parameters.values() if p.name != "self"]
    assert len(array_params) == 1, f"score() takes {len(array_params)} array params, not 1: {sig}"

    for mod in ("inference/image_pipeline.py", "inference/video_pipeline.py"):
        tree = ast.parse((ROOT / mod).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "score"
            ):
                assert len(node.args) == 1, f"{mod}: backend.score() called with {len(node.args)} args"


def test_branch_a_scores_the_whole_frame_in_the_video_path():
    """R22's mirror: video feeds branch A the whole frame, not an aligned crop."""
    clip = _noise_clip()
    meta, frames, _idx, _stamps = pre.extract_frames(
        clip, mime="video/mp4", n_frames=16, caps=pre.DEFAULT_CAPS, target_size=384
    )
    bundle, cap = _run_video_capturing(clip)
    seen0 = cap.calls[0][0]
    whole = pre.resize(frames[0], 384).astype(np.float32)
    assert _digest(seen0) == _digest(whole), "video branch A was not scored from the whole frame"
    assert bundle.branch(BranchId.A).status is BranchStatus.SCORED


# --------------------------------------------------------------------------
# 2. The shipped contradiction — both states pass today
# --------------------------------------------------------------------------


def test_the_fixture_and_the_pipeline_disagree_about_a_faceless_image():
    """**The fail-before evidence, executed.**

    ``image_no_face`` (a published example) and the real pipeline give opposite
    answers for a faceless image. Both are checked in by tests that pass. This
    batch does not make them agree — a whole-image B is a capability, not a
    clarification — it makes the fixture *say* it depicts a target.
    """
    fixture = fx.image_no_face()
    pipeline = asyncio.run(ImagePipeline().analyze_image(_spec(_faceless_png()), _faceless_png()))

    # The fixture: branch B carries a decided verdict with no face.
    assert fixture.branch(BranchId.A).status is BranchStatus.ABSTAINED
    assert fixture.branch(BranchId.A).abstain_reason is AbstainReason.NO_FACE_DETECTED
    assert fixture.branch(BranchId.B).status is BranchStatus.SCORED
    assert fixture.quality.gate_passed is True
    assert fixture.verdict is Verdict.MANIPULATED

    # The pipeline: everything abstains.
    assert pipeline.branch(BranchId.B).status is BranchStatus.ABSTAINED
    assert pipeline.quality.gate_passed is False
    assert pipeline.verdict is Verdict.INSUFFICIENT_EVIDENCE

    # They disagree on all three.
    assert fixture.branch(BranchId.B).status is not pipeline.branch(BranchId.B).status
    assert fixture.quality.gate_passed is not pipeline.quality.gate_passed
    assert fixture.verdict is not pipeline.verdict


def test_the_pipeline_never_scores_branch_b_without_a_face():
    """The current, load-bearing behaviour. A future whole-image B changes this."""
    for maker in (_faceless_png, lambda: _faceless_png(40, 40)):
        data = maker()
        bundle = asyncio.run(ImagePipeline().analyze_image(_spec(data), data))
        assert bundle.verdict is Verdict.INSUFFICIENT_EVIDENCE
        assert all(br.status is not BranchStatus.SCORED for br in bundle.branches), (
            "a branch scored on an input with no usable face — R22 may have moved"
        )


# --------------------------------------------------------------------------
# 3. The claim carries nothing on the wire
# --------------------------------------------------------------------------


def test_branch_descriptions_have_no_functional_consumer():
    """If this changes, ``BRANCH_DESCRIPTIONS`` became load-bearing and the
    clarification below must be re-evaluated as a contract change.

    A *functional* consumer reads the dict: subscripts it, iterates it, or hands
    it somewhere. A comment or a docstring naming it is not one.
    """
    consumers = []
    allowed = {"branch.py", "__init__.py", "test_branch_b_semantics.py", "branch_b_semantics_study.py"}
    for p in ROOT.rglob("*.py"):
        if "__pycache__" in p.parts or p.name in allowed:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "BRANCH_DESCRIPTIONS":
                consumers.append(f"{p.relative_to(ROOT)}:{node.lineno}")
    assert consumers == [], f"BRANCH_DESCRIPTIONS now has functional consumers: {consumers}"


def test_the_whole_image_claim_is_not_on_the_wire():
    branch_result_fields = set(
        __import__("contracts.branch", fromlist=["BranchResult"]).BranchResult.model_fields
    )
    assert "description" not in branch_result_fields, "BranchResult gained a description field"

    # The branch-B claim, specifically: "DINOv2-B/14 ... over the whole image".
    # NOT a bare "whole image" — that also matches FaceTrack's unrelated R16 prose
    # ("the whole image's verdict"), which is how this assertion was red at the
    # Batch 17 baseline (see REPRODUCIBILITY_AUDIT.md, Batch 18).
    for name in ("schema_out/EvidenceBundle.schema.json", "backend/openapi.snapshot.json"):
        f = ROOT / name
        if f.exists():
            text = f.read_text(encoding="utf-8")
            assert "over the whole image" not in text, f"{name} carries the branch-B claim"
            assert "DINOv2" not in text, f"{name} carries the branch-B claim"


# --------------------------------------------------------------------------
# 4. The clarification — fail before, pass after
# --------------------------------------------------------------------------


def test_the_branch_b_description_states_what_it_scores():
    """**Fails before the fix.** The bare 'over the whole image' claim is corrected."""
    desc = BRANCH_DESCRIPTIONS[BranchId.B]
    assert desc != LegacyBranchBClaim.B, "BRANCH_DESCRIPTIONS[B] is unchanged"
    assert "over the whole image" not in desc.lower(), (
        f"branch B still claims to score the whole image as a bare fact: {desc!r}"
    )
    assert "R22" in desc, f"the description does not point at the finding: {desc!r}"
    assert "crop" in desc.lower() and "branch a" in desc.lower(), (
        f"the description does not say it scores what branch A scores: {desc!r}"
    )


def test_the_branch_a_description_notes_the_video_mismatch():
    """**Fails before the fix.** Branch A's 'aligned face crop' is also violated (video)."""
    desc = BRANCH_DESCRIPTIONS[BranchId.A]
    assert desc != LegacyBranchBClaim.A
    assert "video" in desc.lower(), f"branch A does not note the video path: {desc!r}"


def test_the_enums_comment_no_longer_states_whole_image_as_a_bare_fact():
    """**Fails before the fix.** ``enums.py``'s BranchId.B comment is corrected."""
    src = (ROOT / "contracts/enums.py").read_text(encoding="utf-8")
    line = next(ln for ln in src.splitlines() if ln.strip().startswith('B = "B"'))
    assert "R22" in line, f"the BranchId.B comment does not point at R22: {line!r}"
    assert "whole image" not in line or "target" in line.lower(), (
        f"the BranchId.B comment still asserts whole-image scoring flatly: {line!r}"
    )


def test_the_readme_branch_table_is_corrected():
    """**Fails before the fix.**"""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    b_row = next(ln for ln in readme.splitlines() if "Branch B" in ln and "semantic" in ln.lower())
    assert "R22" in b_row or "target" in b_row.lower(), (
        f"the README branch B row still states whole-image scoring flatly: {b_row!r}"
    )


def test_the_image_no_face_fixture_is_annotated_as_a_target():
    """**Fails before the fix.** The fixture depicts a capability the pipeline lacks.

    Preserved as ``LegacyBranchBClaim.FIXTURE_DOCSTRING`` so the correction is
    checked against what it said, not paraphrased.
    """
    doc = fx.image_no_face.__doc__ or ""
    assert doc.strip() != LegacyBranchBClaim.FIXTURE_DOCSTRING, "the fixture docstring is unchanged"
    assert "R22" in doc, f"the fixture does not point at R22: {doc!r}"
    assert "target" in doc.lower() or "does not" in doc.lower(), (
        f"the fixture does not say it depicts a target rather than current behaviour: {doc!r}"
    )


# --------------------------------------------------------------------------
# 5. R16 interaction — the shape, pinned
# --------------------------------------------------------------------------


def test_fuse_takes_a_single_branch_set():
    """R16 under 'B is crop-scoped' = call fuse() N times, unchanged.

    Under 'B is whole-image' fuse() would need a new signature (one shared B
    pooled with each face-level A). Pinning the signature makes that a visible
    decision rather than a silent refactor.
    """
    for fn in (fuse, manipulation_evidence):
        params = [p for p in inspect.signature(fn).parameters]
        assert params[0] == "outcomes", f"{fn.__name__} signature changed: {params}"

    # crop-scoped R16: fuse works per face with no change
    w = {BranchId.A: 0.72, BranchId.B: 0.24, BranchId.P: 0.04}
    for a_logit in (3.0, -1.0):
        z = fuse(
            {
                BranchId.A: BranchOutcome(logit=a_logit),
                BranchId.B: BranchOutcome(logit=0.5),
                BranchId.P: BranchOutcome(logit=0.0),
            },
            w,
        )
        assert z is not None


# --------------------------------------------------------------------------
# 6. Nothing moved
# --------------------------------------------------------------------------

# S8 Batch 22 (R25): weights ``w7f45fada`` -> ``wacdd9240`` (input_size folded in)
# and face-selection ``19559387`` -> ``251d22e9`` (interpolation_rule + crop_clip_rule
# added). Branch B semantics were not touched; this is the collateral pin update.
# S8 Batch 23 (R29): the ``.f.`` segment became the composite of the face and
# video selection policies (``251d22e9`` -> ``e629642c``). Branch B semantics
# were not touched; this is the collateral pin update.
# SCRFD Integration Program, Phase 2: FaceSelectionPolicy gained
# confidence_threshold/nms_iou_threshold (``e629642c`` -> ``f8244ad0``,
# docs/SCRFD_PHASE2_SPEC_RESOLUTION.md Decision 7). Branch B semantics were
# not touched; this is the collateral pin update.
_EXPECTED_EFFECTIVE = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.f8244ad0"


def test_no_identity_string_moves():
    reg = ModelRegistry()
    reg.register(ModelSpec(name="i", version="stub-0", backend="stub"))
    eng = ImagePipeline(model_name="i", registry=reg)
    assert eng.model_identity == _EXPECTED_EFFECTIVE, (
        f"effective_version moved: {eng.model_identity}"
    )


def test_every_shipped_fixture_still_validates():
    for name, bundle in fx.all_bundles().items():
        EvidenceBundle.model_validate(bundle.model_dump())


def test_image_no_face_is_still_a_published_example_and_validates():
    """The fixture stays a documented EvidenceBundle example — annotated, not removed."""
    import contracts.examples as ex

    examples = ex.all_examples().get("EvidenceBundle", {})
    assert "image_no_face" in examples, "image_no_face is no longer a documented example"
    EvidenceBundle.model_validate(examples["image_no_face"])


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
