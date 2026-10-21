"""R20 — ``QualityReport.face_px`` does not describe the value it carries.

THE FINDING, RE-DERIVED (not assumed)
-------------------------------------
``contracts/common.py`` declares::

    face_px: int | None = Field(default=None, ge=0,
                                description="Short edge of the largest face.")

The value actually published, ``inference/image_pipeline.py``::

    face_px = faces[0].size_px if faces else 0        # image path
    face_px = None                                    # video path (no face stage)

``faces[0]`` is the box that minimises :func:`inference.preprocess.face_order_key`,
whose leading component is ``-box.size_px``; ``FaceBox.size_px == min(w, h) ==
short_edge`` (Batch 14, R19). So the published figure is *the short edge of the
face whose short edge is the largest* — which, because the ranking metric and the
reported metric are now the same quantity, is also *the largest face size
present* (``test_face_size_policy::test_the_published_face_px_is_the_largest_face_
size_present``). Plus two cases the sentence does not mention: **0** on a faceless
image, **null** on every video bundle.

WHY THIS IS DOCUMENTATION-ONLY TODAY (and was not always)
--------------------------------------------------------
The Batch 11 multi-face review (``docs/MULTI_FACE_SEMANTICS.md`` §10, R20) recorded
the value as *"faces[0].short_edge, the short edge of the largest-**area** face"* —
a genuine semantic mismatch, because pre-Batch-14 ``face_order_key`` ranked on
**area** while the report took the **short edge**. Batch 14 (R19) collapsed
ranking, gating and reporting onto one metric (``size_px``), which removed that
mismatch. That Batch 11 reading of the *value* is now false, and
``test_the_batch_11_reading_of_the_value_is_refuted_by_execution`` proves it
against the running pipeline rather than paraphrasing it.

What remains is prose: the word "largest" is unqualified (area? short edge?), and
the 0 / null cases are undocumented. Correcting it changes a ``Field`` description
string and nothing else — no field, constraint, behaviour, fingerprint, schema
version or cache key.

WHY DOUBLES
-----------
``LegacyFacePxClaim`` holds the pre-Batch-18 description string verbatim, plus the
pre-Batch-14 area-ranked reading of the value as an executable method, so "the
prose was corrected" and "the Batch 11 reading is refuted" are both checked
against what was actually said.
``ScriptedDetector`` stands in for the detector stage so a box list of a chosen
shape reaches the *shipped* ``face_order_key`` and the *shipped* gate.
``StubBackend`` (the real one) scores whatever tensor it is handed; this suite
never needs a luminance oracle because it reads back ``quality.face_px``, not a
verdict.

This suite is Batch 18. R22 was closed independently by Batch 17
(``docs/BRANCH_B_SEMANTICS.md``); §6 here re-verifies its load-bearing
invariants from scratch, because the brief asked for R20 and R22 together.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import contracts.fixtures as fx  # noqa: E402
from contracts.common import QualityReport  # noqa: E402
from contracts.enums import CONTRACT_VERSION, AbstainReason, MediaKind, Verdict  # noqa: E402
from contracts.evidence import EvidenceBundle  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from contracts.schema import verify_examples  # noqa: E402
from evaluation.benchmark import RUN_SCHEMA_VERSION  # noqa: E402
from evaluation.calibration_protocol import CALIBRATION_WITNESS_VERSION  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.image_pipeline import ImagePipeline  # noqa: E402
from inference.models import LoadedModel, ModelRegistry, ModelSpec  # noqa: E402
from inference.postprocess import QualityFacts  # noqa: E402
from inference.preprocess import FaceBox, face_order_key  # noqa: E402
from inference.video_pipeline import VideoPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# The exact runtime identity Batch 12/14/16/17 pinned. A description string feeds
# none of it; every test in §5 that asserts this is proving that, not assuming it.
# S8 Batch 22 (R25) moved it — weights ``w7f45fada`` -> ``wacdd9240`` and
# face-selection ``19559387`` -> ``251d22e9`` — because the crop-geometry knobs
# entered the identity. S8 Batch 23 (R29) moved it again — ``251d22e9`` ->
# ``e629642c`` — because the ``.f.`` segment became the composite of the face
# and video selection policies. SCRFD Integration Program, Phase 2 moved it
# again — ``e629642c`` -> ``f8244ad0`` — FaceSelectionPolicy gained
# confidence_threshold/nms_iou_threshold (docs/SCRFD_PHASE2_SPEC_RESOLUTION.md
# Decision 7). R20/R22 remain prose-only; this is the collateral update.
_EXPECTED_EFFECTIVE = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.f8244ad0"


# --------------------------------------------------------------------------
# Named doubles
# --------------------------------------------------------------------------


class LegacyFacePxClaim:
    """``QualityReport.face_px`` as documented and as read, before Batch 18.

    ``DESCRIPTION`` is the ``Field(description=...)`` string verbatim, the one
    ``docs/MULTI_FACE_SEMANTICS.md`` §10 quotes. ``legacy_value`` is the
    pre-Batch-14 computation the Batch 11 review described: the short edge of the
    **area**-largest box. Keeping it executable is what lets
    ``test_the_batch_11_reading_of_the_value_is_refuted_by_execution`` demonstrate
    the mismatch rather than restate it.
    """

    name = "QualityReport.face_px prose + value, pre-Batch-18"
    DESCRIPTION = "Short edge of the largest face."
    BATCH_11_VALUE_CLAIM = "the short edge of the largest-area face"

    @staticmethod
    def legacy_key(box: FaceBox) -> tuple[int, int, int, int, int]:
        # (-area, y, x, width, height) — face_order_key as of Batch 10-13.
        return (-box.area, box.y, box.x, box.width, box.height)

    @classmethod
    def legacy_value(cls, boxes: list[FaceBox]) -> int | None:
        if not boxes:
            return 0
        return sorted(boxes, key=cls.legacy_key)[0].short_edge


class ScriptedDetector:
    """Replaces ``preprocess.detect_faces``. Returns a fixed list, shipped order."""

    name = "scripted box list, ordered by the shipped face_order_key"

    def __init__(self, boxes: list[FaceBox]) -> None:
        self.boxes = list(boxes)

    def __call__(self, image, *, policy=None, min_px=None):
        return sorted(self.boxes, key=face_order_key)


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

# Two boxes whose area order and short-edge order disagree. Not reachable from
# Haar (|w-h| <= 1 measured, Batch 14) but exactly the shape SCRFD NMS returns.
THIN = FaceBox(x=100, y=100, width=900, height=50)   # area 45000, size_px  50
GOOD = FaceBox(x=100, y=300, width=200, height=200)  # area 40000, size_px 200


def _canvas(w: int = 1200, h: int = 600) -> np.ndarray:
    return np.full((h, w, 3), 128, np.uint8)


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def _spec(data: bytes, kind: MediaKind = MediaKind.IMAGE) -> JobSpec:
    return JobSpec(
        job_id=UUID(int=20),
        media_kind=kind,
        storage_key="uploads/x",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="image/png" if kind is MediaKind.IMAGE else "video/mp4",
        model_version="stub-0",
        idempotency_key="k",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _analyse_image(boxes: list[FaceBox]) -> EvidenceBundle:
    """Scripted boxes, the real ``StubBackend``, real ``analyze_image`` between."""
    registry = ModelRegistry()
    registry.register(ModelSpec(name="s", version="stub-0", backend="stub", input_size=384))
    pipeline = ImagePipeline(model_name="s", registry=registry)
    loaded = registry.load_model("s")
    registry._loaded["s"] = LoadedModel(
        spec=pipeline._spec, backend=loaded.backend, loaded_at=datetime.now(UTC), load_ms=0
    )
    data = _encode(_canvas())
    original = pre.detect_faces
    pre.detect_faces = ScriptedDetector(boxes)  # type: ignore[assignment]
    try:
        return asyncio.run(pipeline.analyze_image(_spec(data), data))
    finally:
        pre.detect_faces = original  # type: ignore[assignment]


def _noise_clip(frames: int = 20, size: int = 160) -> bytes:
    import tempfile

    path = Path(tempfile.mkdtemp(prefix="dfp-r20-")) / "c.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (size, size))
    rng = np.random.default_rng(7)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _analyse_video() -> EvidenceBundle:
    registry = ModelRegistry()
    registry.register(ModelSpec(name="v", version="stub-0", backend="stub", input_size=384))
    pipeline = VideoPipeline(model_name="v", registry=registry, n_frames=16)
    clip = _noise_clip()
    return asyncio.run(pipeline.analyze_video(_spec(clip, MediaKind.VIDEO), clip))


def _face_px_description() -> str:
    return QualityReport.model_fields["face_px"].description or ""


# --------------------------------------------------------------------------
# 0. The doubles are faithful
# --------------------------------------------------------------------------


def test_the_legacy_description_double_is_the_string_the_review_quoted():
    """``LegacyFacePxClaim.DESCRIPTION`` must be the string on record, or every
    fail-before / refutation test below is checking against a paraphrase."""
    doc = (ROOT / "docs/MULTI_FACE_SEMANTICS.md").read_text(encoding="utf-8")
    assert LegacyFacePxClaim.DESCRIPTION in doc, (
        "docs/MULTI_FACE_SEMANTICS.md no longer quotes "
        f"{LegacyFacePxClaim.DESCRIPTION!r}; the double may be stale"
    )


def test_the_legacy_area_reading_and_the_shipped_metric_disagree_on_the_witness():
    """The premise of the refutation: THIN/GOOD is a real divergence witness."""
    assert LegacyFacePxClaim.legacy_value([THIN, GOOD]) == THIN.short_edge == 50
    assert sorted([THIN, GOOD], key=face_order_key)[0].size_px == GOOD.size_px == 200


# --------------------------------------------------------------------------
# 1. Current behaviour and the values actually published
# --------------------------------------------------------------------------


def test_face_px_field_shape_is_unchanged():
    """The Batch 18 change is prose. The field itself must not move."""
    f = QualityReport.model_fields["face_px"]
    assert f.annotation == (int | None), f.annotation
    assert f.default is None
    md = {type(m).__name__: m for m in f.metadata}
    assert "Ge" in md and md["Ge"].ge == 0, f"ge=0 constraint changed: {f.metadata}"


def test_the_published_face_px_is_faces0_size_px():
    """The value is ``faces[0].size_px`` — the selected face's short edge."""
    bundle = _analyse_image([THIN, GOOD])
    selected = sorted([THIN, GOOD], key=face_order_key)[0]
    assert bundle.quality.face_px == selected.size_px == 200, (
        f"face_px={bundle.quality.face_px}, faces[0].size_px={selected.size_px}"
    )
    assert (bundle.faces[0].bbox.width, bundle.faces[0].bbox.height) == (GOOD.width, GOOD.height)


def test_the_published_face_px_is_the_largest_short_edge_present():
    """Because ranking and reporting read one metric, it is also the max."""
    bundle = _analyse_image([THIN, GOOD])
    assert bundle.quality.face_px == max(b.size_px for b in (THIN, GOOD)) == 200


def test_a_faceless_image_publishes_face_px_zero_and_a_failed_gate():
    """The 0 sentinel. It only ever co-occurs with ``gate_passed=False``."""
    bundle = _analyse_image([])
    assert bundle.quality.face_px == 0
    assert bundle.quality.gate_passed is False
    assert bundle.quality.gate_reason is AbstainReason.NO_FACE_DETECTED
    assert bundle.verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_every_real_video_bundle_publishes_face_px_null():
    """The video path constructs ``QualityFacts(..., face_px=None)`` unconditionally."""
    src = inspect.getsource(VideoPipeline)
    assert "face_px=None" in src, "video_pipeline no longer hard-codes face_px=None"
    bundle = _analyse_video()
    assert bundle.quality.face_px is None
    assert bundle.media.kind is MediaKind.VIDEO


def test_the_video_fixtures_depict_a_populated_face_px_as_a_target():
    """Fixtures carry integers (post-SCRFD target); the engine carries null. This
    is the R18 split, recorded here so the Batch 18 prose does not claim the
    fixtures are wrong."""
    populated = [
        name for name, b in fx.all_bundles().items()
        if b.media.kind is MediaKind.VIDEO and b.quality.face_px is not None
    ]
    assert populated, "expected at least one video fixture depicting the target"


# --------------------------------------------------------------------------
# 2. The wording mismatch, executed
# --------------------------------------------------------------------------


def test_the_batch_11_reading_of_the_value_is_refuted_by_execution():
    """Preserve-and-refute. ``docs/MULTI_FACE_SEMANTICS.md`` §10 says the value is
    'the short edge of the largest-area face'. Run it: it is not, since Batch 14."""
    bundle = _analyse_image([THIN, GOOD])
    legacy = LegacyFacePxClaim.legacy_value([THIN, GOOD])
    assert legacy == 50
    assert bundle.quality.face_px == 200
    assert bundle.quality.face_px != legacy, (
        "the Batch 11 reading happens to match again — the refutation is void"
    )


def test_the_two_readings_agree_for_haar_shaped_boxes():
    """Why R20 was documentation-only and never a live defect: for boxes within a
    pixel of square, area order and short-edge order pick the same box, so the old
    description and the value never disagreed under the shipped detector."""
    rng = np.random.default_rng(20)
    checked = 0
    for _ in range(20_000):
        n = int(rng.integers(1, 5))
        boxes = []
        for _i in range(n):
            s = int(rng.integers(1, 400))
            delta = int(rng.integers(-1, 2))  # |w-h| <= 1, like Haar
            boxes.append(FaceBox(0, 0, s, max(1, s + delta)))
        by_area = sorted(boxes, key=LegacyFacePxClaim.legacy_key)[0].short_edge
        by_metric = sorted(boxes, key=face_order_key)[0].size_px
        assert by_area == by_metric
        checked += 1
    assert checked == 20_000


def test_the_bare_word_largest_is_ambiguous_before_the_fix():
    """Documents the defect: the legacy string names no metric for 'largest'."""
    legacy = LegacyFacePxClaim.DESCRIPTION
    assert "largest" in legacy
    assert "short edge" not in legacy.split("largest")[1], (
        "the legacy string already disambiguates 'largest' — re-check the finding"
    )
    for token in ("size_px", "min(", "area", "0", "video", "null"):
        assert token not in legacy, f"legacy string unexpectedly mentions {token!r}"


# --------------------------------------------------------------------------
# 3. The correction is true of the value — fail before, pass after
# --------------------------------------------------------------------------


def test_the_corrected_description_names_the_ranking_metric():
    """**Fails before the fix.** 'largest' must be tied to ``size_px`` / min(w,h)."""
    desc = _face_px_description()
    assert desc != LegacyFacePxClaim.DESCRIPTION, "the description is unchanged"
    low = desc.lower()
    assert "size_px" in low or "min(width" in low or "min(w" in low, (
        f"the corrected description still does not say which metric 'largest' uses: {desc!r}"
    )
    assert "short edge" in low, f"the corrected description dropped 'short edge': {desc!r}"


def test_the_corrected_description_names_the_zero_and_null_cases():
    """**Fails before the fix.** The 0 sentinel and the video null are documented."""
    desc = _face_px_description()
    low = desc.lower()
    assert "0" in desc, f"the 0 (no face) case is undocumented: {desc!r}"
    assert "video" in low or "r18" in low, f"the video-null case is undocumented: {desc!r}"


def test_the_corrected_description_points_at_the_policy_doc():
    """**Fails before the fix.** A reader can get from the field to the full rule."""
    desc = _face_px_description()
    assert "FACE_SIZE_POLICY" in desc or "R20" in desc, (
        f"the description does not reference the finding or its doc: {desc!r}"
    )


def test_the_corrected_description_is_true_of_every_published_value():
    """The one that matters: the sentence must hold for each case the engine emits."""
    desc = _face_px_description().lower()
    # image, faces present
    b1 = _analyse_image([THIN, GOOD])
    assert b1.quality.face_px == 200 and "short edge" in desc
    # image, no face -> 0
    b2 = _analyse_image([])
    assert b2.quality.face_px == 0 and "0" in _face_px_description()
    # video -> null
    b3 = _analyse_video()
    assert b3.quality.face_px is None and ("video" in desc or "r18" in desc)


# --------------------------------------------------------------------------
# 4. face_px is reporting-only — correcting the prose cannot change a decision
# --------------------------------------------------------------------------


def test_the_gate_does_not_read_face_px():
    """``_gate`` tests ``faces[0].size_px``, never ``quality.face_px``. So the
    published figure feeds no decision and its wording is pure documentation.
    Checked against the AST (the docstring names ``face_px`` in prose)."""
    tree = ast.parse(inspect.getsource(ImagePipeline._gate).strip())
    reads = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == "face_px"
    ]
    assert reads == [], f"_gate reads .face_px at lines {[n.lineno for n in reads]}"
    assert "size_px" in ast.dump(tree), "the gate no longer tests size_px"


def test_nothing_computes_from_face_px():
    """AST sweep of inference/ + contracts/: every use of ``face_px`` is an
    assignment or a plain attribute copy — never a conditional test or an operand."""
    offenders: list[str] = []
    for sub in ("inference", "contracts"):
        for p in (ROOT / sub).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                # face_px as an operand of a comparison or binop
                if isinstance(node, ast.Compare):
                    names = [n for n in ast.walk(node) if isinstance(n, ast.Name) and n.id == "face_px"]
                    attrs = [
                        a for a in ast.walk(node)
                        if isinstance(a, ast.Attribute) and a.attr == "face_px"
                    ]
                    if names or attrs:
                        offenders.append(f"{p.relative_to(ROOT)}:{node.lineno} compare")
                if isinstance(node, ast.BinOp):
                    attrs = [
                        a for a in ast.walk(node)
                        if isinstance(a, ast.Attribute) and a.attr == "face_px"
                    ]
                    if attrs:
                        offenders.append(f"{p.relative_to(ROOT)}:{node.lineno} binop")
    assert offenders == [], f"face_px is read into a computation: {offenders}"


# --------------------------------------------------------------------------
# 5. No identity string, version, schema or cache key moves
# --------------------------------------------------------------------------


def test_correcting_the_description_moves_no_runtime_identity():
    reg = ModelRegistry()
    reg.register(ModelSpec(name="i", version="stub-0", backend="stub"))
    eng = ImagePipeline(model_name="i", registry=reg)
    assert eng.model_identity == _EXPECTED_EFFECTIVE, f"effective_version moved: {eng.model_identity}"
    # model_version = model_identity + '.d.<decode>' — still composed, still carries decode
    assert eng.model_version.startswith(_EXPECTED_EFFECTIVE + ".d."), eng.model_version


def test_contract_version_is_unchanged():
    assert CONTRACT_VERSION == "1.0.0"
    assert EvidenceBundle.model_fields["schema_version"].default == "1.0.0"


def test_evaluation_versions_are_unchanged():
    assert RUN_SCHEMA_VERSION == "1.2.0"
    assert CALIBRATION_WITNESS_VERSION == "1.3.0"


def test_no_evaluation_artifact_reads_quality_or_face_px():
    """Proof that RUN_SCHEMA_VERSION / CALIBRATION_WITNESS_VERSION are independent:
    ``face_px`` and ``QualityReport`` appear nowhere under evaluation/."""
    hits: list[str] = []
    for p in (ROOT / "evaluation").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        for token in ("face_px", "QualityReport", "QualityFacts"):
            if token in text:
                hits.append(f"{p.relative_to(ROOT)}:{token}")
    assert hits == [], f"evaluation/ references quality reporting: {hits}"


def test_the_description_is_not_in_the_openapi_structural_signature():
    """``test_openapi`` strips prose, so a reworded description cannot trip drift.
    Prove it: neither string appears in the signature the snapshot compares."""
    from backend import openapi as oa

    sig = oa.structural_signature(oa.build_schema())
    blob = str(sig)
    assert LegacyFacePxClaim.DESCRIPTION not in blob
    assert _face_px_description() not in blob
    assert "face_px" in blob, "sanity: the field itself is still in the signature"


def test_the_committed_openapi_snapshot_does_not_carry_the_prose():
    snap = (ROOT / "backend/openapi.snapshot.json").read_text(encoding="utf-8")
    assert LegacyFacePxClaim.DESCRIPTION not in snap
    assert "Short edge" not in snap


def test_schema_out_is_regenerable_and_carries_the_current_prose():
    """The generated schema files under schema_out/ embed the description. No test
    compares them to a fresh render (staleness is unguarded — Batch 13 noted the
    same). If they were regenerated this batch, they carry the new string; if not,
    they are stale in the same harmless way they already were."""
    from contracts import schema as sch

    fresh = sch.schema_for(EvidenceBundle)
    rendered = fresh["$defs"]["QualityReport"]["properties"]["face_px"].get("description", "")
    assert rendered == _face_px_description(), (
        "schema_for() and the live model disagree about face_px's description"
    )


def test_no_fixture_or_example_payload_changes():
    """Fixtures carry integer ``face_px`` values, never the prose. Untouched."""
    for name, bundle in fx.all_bundles().items():
        EvidenceBundle.model_validate(bundle.model_dump())
    assert verify_examples() == []


# --------------------------------------------------------------------------
# 6. R22 re-verification (closed by Batch 17; the brief asked for R20 + R22)
# --------------------------------------------------------------------------


def test_r22_the_backend_interface_is_still_single_tensor():
    """B17's P1, re-derived: one tensor in, every branch out — so branch B cannot
    be handed a different input than branch A."""
    from inference.models import ModelBackend

    params = [p for p in inspect.signature(ModelBackend.score).parameters.values() if p.name != "self"]
    assert len(params) == 1, f"ModelBackend.score takes {len(params)} params: {params}"
    for mod in ("inference/image_pipeline.py", "inference/video_pipeline.py"):
        tree = ast.parse((ROOT / mod).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "score"
            ):
                assert len(node.args) == 1


def test_r22_the_whole_image_claim_reaches_no_wire_field():
    """B17's P5, re-derived from the committed artifacts."""
    for name in ("backend/openapi.snapshot.json", "schema_out/EvidenceBundle.schema.json"):
        f = ROOT / name
        if f.exists():
            text = f.read_text(encoding="utf-8")
            assert "over the whole image" not in text
            assert "DINOv2" not in text


def test_r22_and_r20_share_the_same_unmoved_identity():
    """Both findings are prose; neither R20 nor R22 moved ``effective_version``.

    The constant itself moved in Batch 22 (R25, crop-geometry knobs into the
    identity) — an unrelated change — so this pins the current value and that
    R20/R22 did not contribute to it.
    """
    assert _EXPECTED_EFFECTIVE == "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.f8244ad0"


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
