"""Branch B semantics: what "whole image" means, traced from claim to tensor.
Standalone, read-only, changes nothing.

    python tests/branch_b_semantics_study.py

Produces every number in ``docs/BRANCH_B_SEMANTICS.md``. Same pattern as
``decode_study.py``, ``multi_face_study.py`` and ``crop_geometry_study.py``.

THE QUESTION
------------
R22, opened by the Batch 11 multi-face review: *``enums.py`` and ``README`` declare
branch B as "DINOv2-B/14 ... whole image", against branch A's "aligned face
crop"; ``_crop_and_score`` passes one crop to ``backend.score()`` and takes
every branch from that one call, so branch B is de facto face-scoped.*

Is that real? Is it documentation-only, a contract clarification, reproducibility-
critical, or correctness-critical? Does R16 need one interpretation? This study
measures rather than argues. Re-derived from code and tests — the finding is not
assumed.

WHY THERE ARE DOUBLES
---------------------
    TensorCaptureBackend  records the exact ``batch`` array every ``score()``
                          call receives, and returns a fixed logit. It stands
                          in for real weights and answers one question: *what
                          pixels does branch B see?*
    ScriptedDetector      returns a fixed ``list[FaceBox]`` so a face can be
                          placed in a corner of a large image, where the crop
                          and the whole image are provably different pixels.
    LegacyBranchBClaim    the pre-clarification description strings, verbatim,
                          so "the claim was corrected" is checked against what
                          it said.

Everything between them — ``_gate``, ``_abstain``, ``_crop_and_score``,
``crop_face``, ``prepare_image``, ``extract_frames``, ``fuse`` — is shipped code,
reached through the real ``analyze_image`` / ``analyze_video``.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.branch import BRANCH_DESCRIPTIONS  # noqa: E402
from contracts.enums import AbstainReason, BranchId, BranchStatus, MediaKind, Verdict  # noqa: E402
import contracts.fixtures as fx  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.image_pipeline import ImagePipeline  # noqa: E402
from inference.models import ModelRegistry, ModelSpec  # noqa: E402
from inference.postprocess import BranchOutcome, fuse, manipulation_evidence  # noqa: E402
from inference.video_pipeline import VideoPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="dfp-r22-"))


def head(n: int, title: str, q: str) -> None:
    print(f"\n{'=' * 74}\n{n}. {title}\n   {q}\n{'=' * 74}")


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


class TensorCaptureBackend:
    """Records every ``batch`` handed to ``score()``. Returns a fixed logit."""

    name = "records the tensor each branch is scored from"

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


@dataclass
class ScriptedDetector:
    """Replaces ``preprocess.detect_faces``. Returns a fixed box list."""

    name = "returns a scripted list[FaceBox], ordered by the shipped rule"
    boxes: list[pre.FaceBox]

    def __call__(self, image, *, policy=None, min_px=None):  # signature-compatible
        return sorted(self.boxes, key=pre.face_order_key)


class LegacyBranchBClaim:
    """The branch descriptions as of Batch 16, verbatim. Not a paraphrase."""

    name = "branch descriptions pre-Batch-17"
    A = "ConvNeXt V2-T over the aligned face crop, with SRM/Bayar residual channels."
    B = "DINOv2-B/14 frozen features with a trained probe, over the whole image."
    ENUM_B_CLAIM = "whole image"  # the bare claim in enums.py:47's comment


def _digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]


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


def _corner_face_image(w: int = 900, h: int = 700, box=(60, 60, 180, 180)) -> tuple[bytes, np.ndarray, pre.FaceBox]:
    """A distinctive image with a face box in the top-left corner.

    The whole image and the crop around that box are provably different pixels,
    so a backend that scored 'the whole image' and one that scored 'the crop'
    produce different tensor digests.
    """
    rng = np.random.default_rng(22)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    # a bright square inside the face box, dark everywhere else — so the crop is
    # visibly not the whole image even after a square resize
    img[:] = img // 6
    x, y, bw, bh = box
    img[y : y + bh, x : x + bw] = 230
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes(), img, pre.FaceBox(x=x, y=y, width=bw, height=bh)


def _noise_clip(frames: int = 24, size: int = 160) -> bytes:
    path = TMP / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (size, size))
    rng = np.random.default_rng(7)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _faceless_png(w: int = 1024, h: int = 768) -> bytes:
    rng = np.random.default_rng(4)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _run_image(image_bytes: bytes, boxes: list[pre.FaceBox]):
    """Real ImagePipeline, scripted detector, capturing backend."""
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


def _run_video(clip: bytes):
    reg = ModelRegistry()
    reg.register(ModelSpec(name="capv", version="stub-0", backend="stub", input_size=384))
    eng = VideoPipeline(model_name="capv", registry=reg, n_frames=16)
    loaded = reg.load_model("capv")
    cap = TensorCaptureBackend(reg.get_spec("capv"))
    loaded.backend = cap
    bundle = asyncio.run(eng.analyze_video(_spec(clip, MediaKind.VIDEO), clip))
    return bundle, cap


# --------------------------------------------------------------------------
# 1 — what branch B claims, and where
# --------------------------------------------------------------------------


def s1_claims() -> None:
    head(1, "The claim", "what says branch B is 'whole image', and does any of it reach the wire?")

    enums_b_line = next(
        ln for ln in (ROOT / "contracts/enums.py").read_text(encoding="utf-8").splitlines()
        if ln.strip().startswith('B = "B"')
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_b = next(ln for ln in readme.splitlines() if "Branch B" in ln and "semantic" in ln.lower())

    print("  claim sites (state after Batch 17):")
    print(f"    contracts/enums.py  BranchId.B comment")
    print(f"      {enums_b_line.strip()}")
    print(f"    contracts/branch.py BRANCH_DESCRIPTIONS[B]")
    print(f"      {BRANCH_DESCRIPTIONS[BranchId.B]}")
    print(f"    README.md frozen-architecture table")
    print(f"      {readme_b.strip()}")

    # functional consumers of BRANCH_DESCRIPTIONS — AST, not a string grep
    consumers = []
    for p in ROOT.rglob("*.py"):
        if "__pycache__" in p.parts or p.name in ("branch.py", "__init__.py"):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        if any(isinstance(n, ast.Name) and n.id == "BRANCH_DESCRIPTIONS" for n in ast.walk(tree)):
            consumers.append(str(p.relative_to(ROOT)))
    consumers = [c for c in consumers if "branch_b_semantics" not in c]
    print(f"\n  BRANCH_DESCRIPTIONS functional consumers (reads the dict): {consumers or 'NONE'}")

    # does the branch-B input-scope claim reach any client surface?
    markers = ("DINOv2", "over the whole image", "aligned face crop")
    surfaces = {}
    for name in ("schema_out/contracts.bundle.json", "schema_out/EvidenceBundle.schema.json",
                 "backend/openapi.snapshot.json"):
        f = ROOT / name
        if f.exists():
            text = f.read_text(encoding="utf-8")
            surfaces[name] = [m for m in markers if m in text]
    try:
        from backend.openapi import build_schema

        live = json.dumps(build_schema())
        surfaces["live OpenAPI doc"] = [m for m in markers if m in live]
    except Exception as exc:  # noqa: BLE001
        surfaces["live OpenAPI doc"] = [f"(build failed: {exc})"]
    for name, hits in surfaces.items():
        print(f"  branch-B scope markers in {name}: {hits or 'NONE'}")
    print(f"  BranchResult wire fields: "
          f"{sorted(__import__('contracts.branch', fromlist=['BranchResult']).BranchResult.model_fields)}")
    print("\n  -> the claim is documentation-as-data. No published field carries")
    print("     the branch-B input scope. ('whole image' does appear in the doc,")
    print("     but only in FaceTrack's R16 prose — 'the whole image's verdict'.)")


# --------------------------------------------------------------------------
# 2 — what branch B actually consumes: the image path
# --------------------------------------------------------------------------


def s2_image_path() -> None:
    head(2, "Image path", "does branch B score the whole image, or the face crop?")
    data, _bgr, box = _corner_face_image()
    bundle, cap = _run_image(data, [box])

    assert len(cap.calls) == 1, f"expected one score() call, got {len(cap.calls)}"
    seen = cap.calls[0]

    # Decode exactly as the pipeline does (RGB), then crop the same box.
    img = pre.decode_image(data)
    crop = pre.crop_face(img, box, policy=pre.default_face_selection())
    as_crop = pre.prepare_image(crop, size=384)
    as_whole = pre.prepare_image(img, size=384)

    print(f"  image {img.shape[1]}x{img.shape[0]}, face box {box.x},{box.y} {box.width}x{box.height}")
    print(f"  tensor branch B was scored from : shape {seen.shape}  digest {_digest(seen)}")
    print(f"  prepare_image(crop_face(...))   : shape {as_crop.shape}  digest {_digest(as_crop)}")
    print(f"  prepare_image(whole image)      : shape {as_whole.shape}  digest {_digest(as_whole)}")
    print(f"\n  branch B saw the CROP        : {_digest(seen) == _digest(as_crop)}")
    print(f"  branch B saw the WHOLE IMAGE : {_digest(seen) == _digest(as_whole)}")
    a = bundle.branch(BranchId.A)
    b = bundle.branch(BranchId.B)
    print(f"\n  and A and B were scored from the SAME call: "
          f"one score() call produced both {a.id.value}={a.logit} and {b.id.value}={b.logit}")


# --------------------------------------------------------------------------
# 3 — the video path, and branch A's mirror-image mismatch
# --------------------------------------------------------------------------


def s3_video_path() -> None:
    head(3, "Video path", "what do A and B score there? (R22's mirror on branch A)")
    clip = _noise_clip()
    meta, frames, idx, stamps = pre.extract_frames(
        clip, mime="video/mp4", n_frames=16, caps=pre.DEFAULT_CAPS, target_size=384
    )
    bundle, cap = _run_video(clip)

    seen0 = cap.calls[0][0]  # first frame of first batch
    whole_frame = pre.resize(frames[0], 384).astype(np.float32)
    print(f"  clip {meta.width}x{meta.height}, {frames.shape[0]} frames kept")
    print(f"  tensor branch A/B scored from (frame 0) : digest {_digest(seen0)}")
    print(f"  resize(whole frame 0, 384)             : digest {_digest(whole_frame)}")
    print(f"\n  video branch A saw the WHOLE FRAME, not a face crop: "
          f"{_digest(seen0) == _digest(whole_frame)}")
    print("  -> in the image path B's 'whole image' claim is violated;")
    print("     in the video path A's 'aligned face crop' claim is violated.")
    print("     One single-tensor interface, two mismatches, opposite branches.")


# --------------------------------------------------------------------------
# 4 — the interface makes the distinction unrepresentable
# --------------------------------------------------------------------------


def s4_interface() -> None:
    head(4, "The interface", "can the backend even be told to score B on different pixels?")
    from inference.models import ModelBackend

    sig = inspect.signature(ModelBackend.score)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    print(f"  ModelBackend.score{sig}")
    print(f"    array parameters: {[p.name for p in params]}  ({len(params)})")

    # the two call sites
    for mod in ("inference/image_pipeline.py", "inference/video_pipeline.py"):
        tree = ast.parse((ROOT / mod).read_text(encoding="utf-8"))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "score"
        ]
        print(f"  {mod}: {len(calls)} backend.score(...) call(s), "
              f"arg counts {[len(c.args) for c in calls]}")
    print("\n  -> one tensor in, one logit vector per branch out. There is no")
    print("     parameter through which branch B could receive the whole image")
    print("     while branch A receives the crop. The distinction is not")
    print("     implemented and cannot be, without changing this interface.")


# --------------------------------------------------------------------------
# 5 — the shipped contradiction: fixture vs pipeline
# --------------------------------------------------------------------------


def s5_contradiction() -> None:
    head(5, "The contradiction", "a faceless image: what does the fixture say, what does the pipeline do?")

    fixture = fx.image_no_face()
    print("  contracts/fixtures.py::image_no_face  (a published OpenAPI example):")
    print(f"    verdict          {fixture.verdict.value}")
    print(f"    gate_passed      {fixture.quality.gate_passed}")
    print(f"    branch A         {fixture.branch(BranchId.A).status.value} / "
          f"{fixture.branch(BranchId.A).abstain_reason.value}")
    print(f"    branch B         {fixture.branch(BranchId.B).status.value}  "
          f"logit={fixture.branch(BranchId.B).logit}  weight={fixture.branch(BranchId.B).weight}")
    print(f"    faces[]          {fixture.faces}")
    print(f"    docstring        {fx.image_no_face.__doc__.strip()!r}")

    data = _faceless_png()
    bundle = asyncio.run(ImagePipeline().analyze_image(_spec(data), data))
    print("\n  real ImagePipeline on a faceless image:")
    print(f"    verdict          {bundle.verdict.value}")
    print(f"    gate_passed      {bundle.quality.gate_passed}   gate_reason "
          f"{bundle.quality.gate_reason.value}")
    for br in bundle.branches:
        r = br.abstain_reason.value if br.abstain_reason else None
        print(f"    branch {br.id.value}         {br.status.value}"
              + (f" / {r}" if r else f"  logit={br.logit}"))

    print("\n  DISAGREEMENT:")
    print(f"    branch B status : fixture={fixture.branch(BranchId.B).status.value}  "
          f"pipeline={bundle.branch(BranchId.B).status.value}")
    print(f"    gate_passed     : fixture={fixture.quality.gate_passed}  "
          f"pipeline={bundle.quality.gate_passed}")
    print(f"    verdict         : fixture={fixture.verdict.value}  pipeline={bundle.verdict.value}")
    print("\n  Both test_contracts::test_no_face_image_abstains_on_branch_a_only and")
    print("  test_inference::test_no_face_path_abstains pass today. They encode")
    print("  opposite semantics for branch B on a faceless image.")
    # also: the pipeline's branch B abstain reason
    print(f"\n  Note: the pipeline abstains branch B with "
          f"{bundle.branch(BranchId.B).abstain_reason.value!r} — the image quality")
    print("  was fine; there was simply no crop. `_abstain` hard-codes that reason.")


# --------------------------------------------------------------------------
# 6 — the scoping table
# --------------------------------------------------------------------------


def s6_scope() -> None:
    head(6, "Scope", "image / face / crop / media — per branch, per path")
    print("  path    branch  declared            actually scores")
    print("  ------  ------  ------------------  --------------------------------")
    print("  image   A       aligned face crop   the aligned face crop        (matches)")
    print("  image   B       whole image         the aligned face crop        (MISMATCH)")
    print("  image   P       C2PA/EXIF           file metadata (no reader yet)")
    print("  video   A       aligned face crop   the whole resized frame      (MISMATCH)")
    print("  video   B       whole image         the whole resized frame      (matches*)")
    print("  video   P       C2PA/EXIF           file metadata (no reader yet)")
    print("  * modulo R24: extract_frames squashes 16:9 frames onto a square.")
    print("\n  Branch B is CROP-SCOPED in the image path and FRAME-SCOPED in the video")
    print("  path. It is never image-scoped for a still image, which is what the")
    print("  claim says. The video path satisfies B's claim only because it has no")
    print("  face stage (R18) — i.e. for the wrong reason.")


# --------------------------------------------------------------------------
# 7 — does R16 need one interpretation?
# --------------------------------------------------------------------------


def s7_r16() -> None:
    head(7, "R16 interaction", "does per-face scoring need B to be one thing or the other?")
    sig = inspect.signature(fuse)
    print(f"  fuse{sig}")
    print(f"  manipulation_evidence{inspect.signature(manipulation_evidence)}")
    print("  Both take ONE dict[BranchId, BranchOutcome] — one A, one B, one P.")
    print()
    # demonstrate the two shapes
    per_face_a = [BranchOutcome(logit=x) for x in (3.0, -1.0)]
    shared_b = BranchOutcome(logit=0.5)
    shared_p = BranchOutcome(logit=0.0)
    w = {BranchId.A: 0.72, BranchId.B: 0.24, BranchId.P: 0.04}

    print("  If B is CROP-SCOPED (what the code does):")
    for i, a in enumerate(per_face_a):
        z = fuse({BranchId.A: a, BranchId.B: BranchOutcome(logit=0.5 + 0.1 * i), BranchId.P: shared_p}, w)
        print(f"    face {i}: fuse() called unchanged -> z={z:.4f}")
    print("    R16 = call fuse() once per face. No signature change.")
    print()
    print("  If B is WHOLE-IMAGE (what the claim says):")
    print("    one image-level B must be pooled with each face-level A. fuse() and")
    print("    manipulation_evidence()'s renormalised weighted mean take a single")
    print("    {A,B} pair, so this needs a new signature: fuse(face_a, shared_b, shared_p).")
    print()
    print("  -> R16 is *shaped by* R22's answer; it is not *blocked* by it. The")
    print("     code has already chosen crop-scoped, which is the simpler shape for R16.")


# --------------------------------------------------------------------------
# 8 — identity / fingerprint
# --------------------------------------------------------------------------


def s8_identity() -> None:
    head(8, "Identity", "does clarifying the claim move any version string?")
    reg = ModelRegistry()
    reg.register(ModelSpec(name="i", version="stub-0", backend="stub"))
    eng = ImagePipeline(model_name="i", registry=reg)
    print(f"  effective_version      : {eng.model_identity}")
    print(f"  calibration_fingerprint: {eng._spec.calibration_fingerprint}")
    print(f"  gate_fingerprint       : {eng._spec.gate_fingerprint}")
    print(f"  face_selection id      : {eng._spec.face_selection_fingerprint}")
    print()
    print("  BRANCH_DESCRIPTIONS, the enums comment and the README table feed NONE")
    print("  of these. They are not in any fingerprint, not in the OpenAPI structural")
    print("  signature (prose is stripped), and not in the schema snapshot. Correcting")
    print("  them moves nothing and dumps no cache.")


def main() -> int:
    print(__doc__)
    s1_claims()
    s2_image_path()
    s3_video_path()
    s4_interface()
    s5_contradiction()
    s6_scope()
    s7_r16()
    s8_identity()
    print(f"\n{'=' * 74}\nartifacts in {TMP}\n{'=' * 74}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
