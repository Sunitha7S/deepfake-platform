"""Multi-face semantics: where the verdict comes from when a file has more
than one face. Standalone, read-only, changes nothing.

    python tests/multi_face_study.py

Produces every number in ``docs/MULTI_FACE_SEMANTICS.md``. Same pattern as
``decode_study.py``, ``band_study.py`` and ``inference_determinism_study.py``.

THE QUESTION
------------
Batch 10 closed R14 — face *selection* no longer depends on the order the
detector emitted its boxes in — and recorded R16 in the same breath: the
platform still publishes one face's verdict as the media's. R14 made the choice
reproducible. It did not make it right, and it did not write it down anywhere a
client can read.

This study asks what the shipped code actually does, stage by stage:

    Where does selection happen?  Which face wins?  Is that deterministic?
    What is published when the faces disagree?  Does the contract say?

WHY THERE ARE DOUBLES
---------------------
Haar cannot reliably find two synthetic faces in one canvas — measured in stage
0 below, and it is R13 showing up as a fixture constraint exactly as it did in
``shifted_face_image``. Building the multi-face cases out of real detections
would therefore measure the placeholder detector's eyesight rather than the
platform's semantics.

So the *detector* is replaced by a named double that returns a scripted box
list, and the *backend* by a named double that scores a crop from its own
pixels. Everything between them — ``_gate``, ``_abstain``, ``_crop_and_score``,
``fuse``, ``calibrated_probability``, ``decide``, ``build_image_bundle`` — is
the shipped code, reached through the real ``analyze_image`` on a real event
loop. The doubles stand in for the two ends the platform does not have yet
(SCRFD, and real weights); they do not stand in for anything under test.

Both are named, both are printed in the output, and stage 0 states what each
one replaces so a reader can tell a measurement from a construction.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.enums import BranchId, MediaKind  # noqa: E402
from contracts.evidence import EvidenceBundle  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.image_pipeline import ImagePipeline  # noqa: E402
from inference.models import ModelRegistry, ModelSpec  # noqa: E402
from inference.preprocess import FaceBox, face_order_key  # noqa: E402
from inference.video_pipeline import VideoPipeline  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dfp-multiface-study-"))


def head(n: int, title: str, question: str) -> None:
    print(f"\n{'=' * 74}\n{n}. {title}\n   {question}\n{'=' * 74}")


# --------------------------------------------------------------------------
# The named doubles
# --------------------------------------------------------------------------


class ScriptedDetector:
    """Stands in for the detector stage. Returns a fixed box list.

    Replaces ``inference.preprocess.detect_faces`` and nothing else. The
    pipeline's whole contract with the detector is ``list[FaceBox]``, and this
    double honours it exactly, including passing the list through the shipped
    ``face_order_key`` — so the ordering under test is the real one, and only
    the *detection* is scripted.

    It exists because Haar cannot see the fixtures (stage 0), and because SCRFD
    will return boxes this shape from NMS over float scores. Scripting the boxes
    is the only way to measure what the platform does with two of them today.
    """

    name = "scripted box list, ordered by the shipped face_order_key"

    def __init__(self, boxes: list[FaceBox]) -> None:
        self.boxes = list(boxes)
        self.calls = 0

    def __call__(self, image: np.ndarray, *, policy=None) -> list[FaceBox]:
        # Signature tracks the shipped ``detect_faces``, which took ``min_px``
        # when this study was written and takes a ``FaceSelectionPolicy`` since
        # Batch 12. The policy is accepted and ignored: this double scripts the
        # boxes, which is upstream of every parameter in it.
        self.calls += 1
        return sorted(self.boxes, key=face_order_key)


class RegionOracleBackend:
    """Stands in for real weights. Scores a crop from its own mean luminance.

    A crop whose mean is below ``dark_max`` is authentic evidence; above
    ``bright_min`` it is manipulated evidence. Both branches A and B get the same
    logit, so the fused value is that logit exactly (P contributes zero) and the
    published probability is ``sigmoid(logit / T)`` with T = 1.

    This is what lets a scenario have a *ground truth*: the study paints one
    region dark and one bright and then reads which one the platform published.
    A hash-derived stub cannot do that — its scores are meaningless by design.
    """

    name = "logit from crop mean luminance: dark = authentic, bright = manipulated"

    AUTHENTIC_LOGIT = -3.0
    MANIPULATED_LOGIT = +3.0

    def __init__(self, spec: ModelSpec, *, dark_max: int = 96, bright_min: int = 160) -> None:
        self.spec = spec
        self.dark_max = dark_max
        self.bright_min = bright_min
        self.seen: list[float] = []

    def score(self, batch: np.ndarray) -> dict[BranchId, np.ndarray]:
        out: dict[BranchId, list[float]] = {b: [] for b in self.spec.branch_weights}
        for i in range(batch.shape[0]):
            mean = float(batch[i].mean())
            self.seen.append(mean)
            if mean <= self.dark_max:
                logit = self.AUTHENTIC_LOGIT
            elif mean >= self.bright_min:
                logit = self.MANIPULATED_LOGIT
            else:
                logit = 0.0
            for branch in (BranchId.A, BranchId.B):
                if branch in out:
                    out[branch].append(logit)
            if BranchId.P in out:
                out[BranchId.P].append(0.0)
        return {b: np.asarray(v, dtype=np.float64) for b, v in out.items()}

    def close(self) -> None:
        return None


class LegacyAreaOnlyOrder:
    """The pre-Batch-10 ordering, kept for the stage 3 comparison.

    Verbatim from ``inference_determinism_study.py``: ``sorted(boxes, key=lambda
    b: -b.area)``. Python's sort is stable, so equal areas resolve to detector
    output order. Reproduced here so this study can show that R14's fix holds
    while R16 stands independently of it.
    """

    name = "area only, stable sort (pre-Batch-10, R14)"

    @staticmethod
    def order(boxes: list[FaceBox]) -> list[FaceBox]:
        return sorted(boxes, key=lambda b: -b.area)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


DARK = 40
BRIGHT = 220


def face_tile(size: int = 480) -> np.ndarray:
    """The determinism study's synthetic face, verbatim. Used only in stage 0."""
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (175, 160, 150), -1)
    cv2.circle(img, (200, 210), 16, (60, 60, 60), -1)
    cv2.circle(img, (280, 210), 16, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 300), (18, 30), 0, 0, 360, (150, 130, 120), -1)
    cv2.ellipse(img, (240, 340), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    return cv2.GaussianBlur(img, (5, 5), 0)


def painted(regions: list[tuple[FaceBox, int]], *, size: tuple[int, int] = (600, 1200)) -> np.ndarray:
    """A canvas with each box filled at a stated luminance.

    Mid-grey background so an accidental crop overlap is visible as a mid score
    rather than silently reading as one of the two answers.
    """
    img = np.full((size[0], size[1], 3), 128, np.uint8)
    for box, level in regions:
        img[box.y : box.y + box.height, box.x : box.x + box.width] = level
    return img


def encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:  # pragma: no cover - encoding a synthetic array does not fail
        raise RuntimeError("failed to encode fixture")
    return buf.tobytes()


def spec_for(data: bytes, kind: MediaKind, mime: str) -> JobSpec:
    return JobSpec(
        job_id=UUID(int=1),
        media_kind=kind,
        storage_key="u",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime=mime,
        model_version="v",
        idempotency_key="k",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@dataclass
class Rig:
    """One analysis, with both doubles installed and then removed."""

    pipeline: ImagePipeline
    detector: ScriptedDetector
    backend: RegionOracleBackend


def run_image(boxes: list[FaceBox], regions: list[tuple[FaceBox, int]]) -> tuple[EvidenceBundle, Rig]:
    """Analyse a painted canvas with a scripted detector. Real pipeline throughout."""
    registry = ModelRegistry()
    spec = ModelSpec(name="oracle-image", version="oracle-0", backend="stub", input_size=384)
    registry.register(spec)
    pipeline = ImagePipeline(model_name="oracle-image", registry=registry)
    backend = RegionOracleBackend(pipeline._spec)
    # Pre-seat the loaded model so the registry never builds a StubBackend.
    from inference.models import LoadedModel

    registry._loaded["oracle-image"] = LoadedModel(
        spec=pipeline._spec, backend=backend, loaded_at=datetime.now(UTC), load_ms=0
    )

    detector = ScriptedDetector(boxes)
    original = pre.detect_faces
    pre.detect_faces = detector  # type: ignore[assignment]
    try:
        data = encode(painted(regions))
        bundle = asyncio.run(pipeline.analyze_image(spec_for(data, MediaKind.IMAGE, "image/png"), data))
    finally:
        pre.detect_faces = original  # type: ignore[assignment]
    return bundle, Rig(pipeline, detector, backend)


def describe(bundle: EvidenceBundle) -> str:
    p = "None" if bundle.p_fake is None else f"{bundle.p_fake:.6f}"
    reason = bundle.abstain_reason.value if bundle.abstain_reason else "-"
    return f"verdict={bundle.verdict.value:<22} p_fake={p:<10} reason={reason}"


# --------------------------------------------------------------------------
# 0 — why the doubles exist
# --------------------------------------------------------------------------


def s0_why_doubles() -> None:
    head(0, "Why this study uses doubles", "can Haar find two faces on one canvas?")
    a = face_tile()
    for label, scale in (("two full-size tiles", 1.0), ("one full, one half", 0.5), ("one full, one quarter", 0.25)):
        canvas = np.full((480, 960, 3), 200, np.uint8)
        canvas[:, :480] = a
        b = a
        if scale != 1.0:
            n = int(480 * scale)
            b = np.full((480, 480, 3), 200, np.uint8)
            b[:n, :n] = cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA)
        canvas[:, 480:] = b
        found = pre.detect_faces(canvas)
        print(f"  {label:<24} -> {len(found)} box(es)  {[(f.x, f.y, f.width, f.height) for f in found]}")
    print()
    print("  Haar finds neither face once a second tile is on the canvas. That is R13's")
    print("  fixture fragility, not a semantics result, so the multi-face cases below")
    print("  script the detector instead:")
    print(f"    detector double : {ScriptedDetector.name}")
    print(f"    backend double  : {RegionOracleBackend.name}")
    print("  Everything between them is the shipped pipeline.")


# --------------------------------------------------------------------------
# 1 — where selection happens
# --------------------------------------------------------------------------


def s1_where() -> None:
    head(1, "Where selection happens", "which lines choose the face?")
    root = Path(__file__).resolve().parent.parent
    for rel in ("inference/image_pipeline.py", "inference/video_pipeline.py", "inference/postprocess.py"):
        path = root / rel
        hits = [
            (i, line.rstrip())
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if "faces[0]" in line or "detect_faces" in line
        ]
        print(f"  {rel}: {len(hits)} selection site(s)")
        for i, line in hits:
            print(f"    {rel}:{i}  {line.strip()}")
    print()
    print("  Selection is not a function. It is five subscripts of one list in one file,")
    print("  and the video pipeline has no selection site at all.")


# --------------------------------------------------------------------------
# 2 — which face wins
# --------------------------------------------------------------------------


def s2_which() -> None:
    head(2, "Which face wins", "largest? highest confidence? which 'largest'?")
    print(f"  FaceBox fields: {list(FaceBox.__dataclass_fields__)}")
    print("  -> there is no confidence field, so 'highest-confidence face' is not")
    print("     representable in the shipped type at all.\n")

    print(f"  face_order_key = {face_order_key.__doc__.splitlines()[0]}")
    print("  = (-area, y, x, width, height)\n")

    # area-largest and short-edge-largest are different faces.
    wide = FaceBox(x=100, y=100, width=400, height=100)   # area 40000, short edge 100
    square = FaceBox(x=700, y=100, width=190, height=190)  # area 36100, short edge 190
    ordered = sorted([wide, square], key=face_order_key)
    print("  Two boxes:")
    print(f"    wide   {wide}  area={wide.area}  short_edge={wide.short_edge}")
    print(f"    square {square}  area={square.area}  short_edge={square.short_edge}")
    print(f"  faces[0] by face_order_key = {ordered[0]}  (area wins)")
    print(f"  largest short edge         = {max([wide, square], key=lambda b: b.short_edge)}")
    print()
    print("  These are different faces. QualityReport.face_px is documented as")
    print("  'Short edge of the largest face' and is filled from faces[0].short_edge,")
    print(f"  which here reports {ordered[0].short_edge} while the largest short edge present is")
    print(f"  {max(b.short_edge for b in (wide, square))}.")


# --------------------------------------------------------------------------
# 3 — determinism of the ordering
# --------------------------------------------------------------------------


def s3_determinism() -> None:
    head(3, "Is the ordering deterministic", "does emission order reach faces[0]?")
    boxes = [
        FaceBox(x=10, y=10, width=200, height=200),
        FaceBox(x=400, y=10, width=200, height=200),
        FaceBox(x=10, y=400, width=200, height=200),
    ]
    shipped = {tuple(sorted(p, key=face_order_key))[0] for p in itertools.permutations(boxes)}
    legacy = {tuple(LegacyAreaOnlyOrder.order(list(p)))[0] for p in itertools.permutations(boxes)}
    print(f"  three equal-area boxes, all {len(list(itertools.permutations(boxes)))} emission orders")
    print(f"    shipped face_order_key : {len(shipped)} distinct faces[0]  {sorted(b.x for b in shipped)}")
    print(f"    {LegacyAreaOnlyOrder.name}: {len(legacy)} distinct faces[0]  {sorted(b.x for b in legacy)}")

    # Total order: no two distinct boxes share a key.
    rng = np.random.default_rng(11)
    sample = [
        FaceBox(*(int(v) for v in rng.integers(0, 300, 2)), *(int(v) for v in rng.integers(1, 300, 2)))
        for _ in range(4000)
    ]
    keys = {face_order_key(b) for b in sample}
    print(f"\n  {len(sample)} random boxes -> {len(keys)} distinct keys, {len(set(sample))} distinct boxes")
    print("  -> the key is total: identical keys only for identical boxes.")
    print("\n  Ordering is deterministic. R14 is closed and stays closed. That is a")
    print("  statement about reproducibility, not about correctness -- the policy")
    print("  being reproduced is still 'the biggest face speaks for the file'.")


# --------------------------------------------------------------------------
# 4 — what is published
# --------------------------------------------------------------------------


def s4_scenarios() -> None:
    head(4, "What verdict is published", "four scenarios, through the real pipeline")

    big = FaceBox(x=100, y=100, width=300, height=300)
    small = FaceBox(x=700, y=150, width=200, height=200)
    tiny = FaceBox(x=700, y=150, width=40, height=40)

    cases = [
        (
            "one authentic (big) + one manipulated (small)",
            [big, small],
            [(big, DARK), (small, BRIGHT)],
        ),
        (
            "one manipulated (big) + one authentic (small)  [same pixels, sizes swapped]",
            [big, small],
            [(big, BRIGHT), (small, DARK)],
        ),
        ("two manipulated faces", [big, small], [(big, BRIGHT), (small, BRIGHT)]),
        ("two authentic faces", [big, small], [(big, DARK), (small, DARK)]),
        ("no face detected", [], [(big, BRIGHT)]),
        ("one tiny face only (40 px, floor is 64)", [tiny], [(tiny, BRIGHT)]),
        (
            "one big manipulated face + one tiny face",
            [big, tiny],
            [(big, BRIGHT), (tiny, DARK)],
        ),
        (
            "one tiny face + one big face, tiny listed first by the detector",
            [tiny, big],
            [(big, DARK), (tiny, BRIGHT)],
        ),
    ]

    for label, boxes, regions in cases:
        bundle, rig = run_image(boxes, regions)
        published = [(f.track_id, f.bbox.width, f.p_fake, f.status.value if f.status else None) for f in bundle.faces]
        print(f"\n  {label}")
        print(f"    detected {len(boxes)} face(s), scored {len(rig.backend.seen)} crop(s)")
        print(f"    {describe(bundle)}")
        print(f"    faces[] published : {published}")
        print(f"    quality.face_px   : {bundle.quality.face_px}   gate_passed={bundle.quality.gate_passed}")

    print("\n  Read the first two rows together: identical face content, sizes swapped,")
    print("  opposite published verdicts. The verdict is a property of which face is")
    print("  bigger, not of what the file contains.")


# --------------------------------------------------------------------------
# 5 — the gate reads the selected face, not the best one
# --------------------------------------------------------------------------


def s5_gate() -> None:
    head(5, "The gate reads faces[0]", "can a scorable face be gated away by a bigger one?")

    # Area-largest is a thin box below the 64 px short-edge floor; a smaller,
    # squarer box comfortably clears it.
    thin = FaceBox(x=100, y=100, width=900, height=50)   # area 45000, short edge 50
    good = FaceBox(x=100, y=300, width=200, height=200)  # area 40000, short edge 200
    print(f"    thin {thin}  area={thin.area}  short_edge={thin.short_edge}  (below the 64 px floor)")
    print(f"    good {good}  area={good.area}  short_edge={good.short_edge}  (clears it)")
    print(f"    face_order_key picks: {sorted([thin, good], key=face_order_key)[0]}\n")

    bundle, rig = run_image([thin, good], [(thin, BRIGHT), (good, BRIGHT)])
    print(f"    {describe(bundle)}")
    print(f"    crops scored: {len(rig.backend.seen)}")
    print(f"    faces[] published: {[(f.bbox.width, f.bbox.height, f.status.value if f.status else None) for f in bundle.faces]}")
    print()
    print("    A face 200 px across was present, passed every floor, and was never")
    print("    scored or published. The gate consulted one box and the file abstained.")
    print()
    print("    Reachable today? Haar's frontal cascade returns square boxes, so area")
    print("    order and short-edge order coincide and this cannot fire. SCRFD returns")
    print("    NMS boxes with free aspect. The defect is latent, and the migration is")
    print("    what arms it.")


# --------------------------------------------------------------------------
# 6 — face-scoped or image-scoped
# --------------------------------------------------------------------------


def s6_scoping() -> None:
    head(6, "Scope of each published field", "what is face-scoped and what is image-scoped?")
    big = FaceBox(x=100, y=100, width=300, height=300)
    small = FaceBox(x=700, y=150, width=200, height=200)
    bundle, _ = run_image([big, small], [(big, BRIGHT), (small, DARK)])

    face = bundle.faces[0]
    print(f"  bundle.p_fake        = {bundle.p_fake!r}")
    print(f"  faces[0].p_fake      = {face.p_fake!r}")
    print(f"  identical            = {bundle.p_fake == face.p_fake}")
    print(f"  bundle.interval      = {bundle.interval}")
    print(f"  bundle.disagreement  = {bundle.disagreement}")
    print(f"  bundle.quality.face_px = {bundle.quality.face_px}  (= faces[0].short_edge)")
    print(f"  branches             = {[(b.id.value, b.logit, b.p) for b in bundle.branches]}")
    print(f"  provenance findings  = c2pa.present={bundle.provenance.c2pa.present}  (file-scoped by nature)")
    print()
    print("  Every published number is computed from one crop and labelled as the")
    print("  file's. The one field with per-face shape -- faces[].p_fake -- carries a")
    print("  copy of the image-level figure, so a client cannot tell the two apart.")
    print()
    print("  Second face's evidence: never computed. It is not abstained, not")
    print("  recorded, not counted. It is absent.")


# --------------------------------------------------------------------------
# 7 — video
# --------------------------------------------------------------------------


def s7_video() -> None:
    head(7, "Video", "what happens when faces appear across frames?")
    root = Path(__file__).resolve().parent.parent
    src = (root / "inference/video_pipeline.py").read_text(encoding="utf-8")
    for token in ("detect_faces", "faces", "FaceTrack", "crop_face", "face_track", "TRACKING"):
        print(f"    '{token}' in video_pipeline.py: {src.count(token)} occurrence(s)")
    print()

    path = TMP / "two_faces.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (960, 480))
    tile = face_tile()
    for _ in range(30):
        canvas = np.full((480, 960, 3), 200, np.uint8)
        canvas[:, :480] = tile
        canvas[:, 480:] = tile
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()
    data = path.read_bytes()
    bundle = asyncio.run(VideoPipeline().analyze_video(spec_for(data, MediaKind.VIDEO, "video/mp4"), data))

    print(f"    {describe(bundle)}")
    print(f"    faces[]              : {bundle.faces}")
    print(f"    quality.face_px      : {bundle.quality.face_px}")
    print(f"    quality.gate_reason  : {bundle.quality.gate_reason}")
    print(f"    frames_sampled       : {bundle.video.frames_sampled}")
    print(f"    frames_scored        : {bundle.video.frames_scored}"
          f"   (contract: 'Sampled frames where a face was usable')")
    print(f"    top_frames track_ids : {[t.track_id for t in bundle.video.top_frames]}")
    print()
    print("    The video path never detects a face. It scores whole resized frames.")
    print("    So:")
    print("      - no face gate applies; NO_FACE_DETECTED and FACE_TOO_SMALL are")
    print("        unreachable for video, and a clip with no face in it at all still")
    print("        receives a decided verdict;")
    print("      - faces[] is empty, and top_frames[].track_id = 0 names a track that")
    print("        is not in the bundle;")
    print("      - frames_scored counts decoded frames, not frames where a face was")
    print("        usable, which is what the field says it counts.")
    print()
    print("    Multi-face video is therefore not mis-attributed -- it is un-attributed.")
    print("    Every face in the frame is averaged into one number by the backbone.")


# --------------------------------------------------------------------------
# 8 — the selection policy is outside every identity string
# --------------------------------------------------------------------------


def s8_identity() -> None:
    head(8, "Identity", "does the version string move when the selection policy moves?")
    caps = pre.MediaCaps()
    spec = ModelSpec(name="x", version="v0", backend="stub").with_gate(caps.gate_fingerprint)
    print(f"  gate fingerprint inputs: max_bytes, max_pixels, max_duration_s, max_frames,")
    print(f"                           min_short_edge, min_face_px")
    print(f"  effective_version      : {spec.effective_version}")

    raised = pre.MediaCaps(min_face_px=96)
    print(f"  min_face_px 64 -> 96   : {ModelSpec(name='x', version='v0', backend='stub').with_gate(raised.gate_fingerprint).effective_version}")
    print("  -> the gate is fingerprinted, on purpose: it changes which inputs get a")
    print("     score at all, so a cache that cannot see it serves withdrawn verdicts.\n")

    print("  Now the two policies that decide *which face* and *what crop*:")
    policy = pre.default_face_selection()
    print(f"    ordering rule       : {policy.ordering}")
    print(f"    crop margin         : {policy.crop_margin}")
    print(f"    face-selection id   : {policy.identity}")
    print(f"    in the version      : "
          f"{ModelSpec(name='x', version='v0', backend='stub')
             .with_gate(caps.gate_fingerprint)
             .with_face_selection(policy.identity).effective_version}")
    print()
    print("  **CLOSED IN BATCH 12 (R17).** When this study was written both were")
    print("  unfingerprinted -- a module-level function and a default argument -- so")
    print("  changing either moved p_fake while effective_version stayed byte-identical.")
    print("  They now compose into the `.f.` segment above. See")
    print("  docs/FACE_SELECTION_IDENTITY.md and tests/face_selection_study.py.")
    print()
    print("  What this does NOT close: R16 is untouched. The policy is now named and")
    print("  reproducible; it still publishes one face's verdict as the media's.")


# --------------------------------------------------------------------------


def main() -> int:
    print(__doc__)
    s0_why_doubles()
    s1_where()
    s2_which()
    s3_determinism()
    s4_scenarios()
    s5_gate()
    s6_scoping()
    s7_video()
    s8_identity()
    print(f"\n{'=' * 74}\nartifacts in {TMP}\n{'=' * 74}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
