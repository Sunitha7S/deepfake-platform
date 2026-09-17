"""Face size: one box, four metrics, and the three decisions that read them.
Standalone, read-only, changes nothing.

    python tests/face_size_policy_study.py

Produces every number in ``docs/FACE_SIZE_POLICY.md``. Same pattern as
``decode_study.py``, ``band_study.py``, ``face_selection_study.py`` and
``multi_face_study.py``.

THE QUESTION
------------
R19, raised by the Batch 11 review: *the gate ranks faces on area and tests
short edge*. ``face_order_key`` puts the largest-**area** box first;
``ImagePipeline._gate`` then tests that box's **short edge** against
``min_face_px``. Two metrics, one decision, and the review recorded the defect
as unreachable "because Haar returns square boxes" and armed by SCRFD.

This study asks four things the review did not:

    Which metrics exist, and which decision reads each one?
    Which of the four disagreement cases can actually be constructed?
    *How far* from square must a box be before the disagreement fires?
    Does Haar really return square boxes?

The last two turn out to have exact answers, and they are not the ones the
review assumed.

WHY THERE ARE DOUBLES
---------------------
Haar cannot see a two-face canvas — measured in ``multi_face_study.py`` stage 0
and unchanged here — so multi-face scenarios are built with a scripted detector.
Everything between the detector and the bundle is shipped code, reached through
the real ``analyze_image`` on a real event loop.

    ScriptedDetector       replaces  inference.preprocess.detect_faces
    RegionOracleBackend    replaces  real weights
    LegacyAreaOrder        replaces  nothing -- it *is* the pre-Batch-14
                                     ``face_order_key``, kept verbatim so the
                                     defect is demonstrated rather than described

Stage 0 restates what each one stands for, so a reader can tell a measurement
from a construction.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import sys
from dataclasses import dataclass, replace
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

FLOOR = pre.DEFAULT_CAPS.min_face_px  # 64


def head(n: int, title: str, question: str) -> None:
    print(f"\n{'=' * 74}\n{n}. {title}\n   {question}\n{'=' * 74}")


# --------------------------------------------------------------------------
# The named doubles
# --------------------------------------------------------------------------


class LegacyAreaOrder:
    """The pre-Batch-14 ``face_order_key``, verbatim.

    ``(-box.area, box.y, box.x, box.width, box.height)`` — the rule Batch 10
    introduced to close R14 and Batch 12 fingerprinted as ``area-desc,y-asc,
    x-asc,width-asc,height-asc``. It ranks on area; the gate it feeds tests
    short edge. That pairing is R19, and this double is what makes the pairing
    executable after the shipped code stops implementing it.
    """

    name = "area-desc,y-asc,x-asc,width-asc,height-asc (pre-Batch-14)"
    rule = "area-desc,y-asc,x-asc,width-asc,height-asc"

    @staticmethod
    def key(box: FaceBox) -> tuple[int, int, int, int, int]:
        return (-box.area, box.y, box.x, box.width, box.height)

    @classmethod
    def order(cls, boxes: list[FaceBox]) -> list[FaceBox]:
        return sorted(boxes, key=cls.key)

    @classmethod
    def decision(cls, boxes: list[FaceBox], floor: int = FLOOR) -> tuple[FaceBox | None, bool]:
        """The whole pre-Batch-14 policy: rank by area, gate on short edge.

        Returns ``(selected, scored)``. This is the composition R19 names, in
        four lines, so the report can talk about it without talking about the
        pipeline.
        """
        if not boxes:
            return None, False
        chosen = cls.order(boxes)[0]
        return chosen, chosen.short_edge >= floor


class ScriptedDetector:
    """Stands in for the detector stage. Returns a fixed box list.

    Replaces ``inference.preprocess.detect_faces`` and nothing else. The
    pipeline's whole contract with the detector is ``list[FaceBox]``, and this
    double honours it exactly, including passing the list through the shipped
    ordering — so the ordering under test is the real one and only the
    *detection* is scripted.

    Same double as ``multi_face_study.ScriptedDetector``. It exists because Haar
    cannot see a two-face canvas, and because SCRFD will return boxes of exactly
    this shape from NMS.
    """

    name = "scripted box list, ordered by the shipped face_order_key"

    def __init__(self, boxes: list[FaceBox]) -> None:
        self.boxes = list(boxes)
        self.calls = 0

    def __call__(self, image: np.ndarray, *, policy=None) -> list[FaceBox]:
        self.calls += 1
        return sorted(self.boxes, key=face_order_key)


class RegionOracleBackend:
    """Stands in for real weights. Scores a crop from its own mean luminance.

    Dark crop ⇒ authentic evidence (−3.0), bright ⇒ manipulated (+3.0), both
    branches. This is what gives a scenario a *ground truth*: the study paints
    one region dark and one bright and reads back which one the platform
    published. The shipped ``StubBackend`` hashes pixels and cannot.

    Same double as ``multi_face_study.RegionOracleBackend``.
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


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


DARK = 40
BRIGHT = 220


def painted(regions: list[tuple[FaceBox, int]], *, size: tuple[int, int] = (600, 1200)) -> np.ndarray:
    img = np.full((size[0], size[1], 3), 128, np.uint8)
    for box, level in regions:
        img[box.y : box.y + box.height, box.x : box.x + box.width] = level
    return img


def encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:  # pragma: no cover - encoding a synthetic array does not fail
        raise RuntimeError("failed to encode fixture")
    return buf.tobytes()


def spec_for(data: bytes) -> JobSpec:
    return JobSpec(
        job_id=UUID(int=1),
        media_kind=MediaKind.IMAGE,
        storage_key="u",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="image/png",
        model_version="v",
        idempotency_key="k",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@dataclass
class Rig:
    pipeline: ImagePipeline
    detector: ScriptedDetector
    backend: RegionOracleBackend


def run_image(boxes: list[FaceBox], regions: list[tuple[FaceBox, int]]) -> tuple[EvidenceBundle, Rig]:
    """Analyse a painted canvas with a scripted detector. Real pipeline throughout."""
    from inference.models import LoadedModel

    registry = ModelRegistry()
    spec = ModelSpec(name="oracle-image", version="oracle-0", backend="stub", input_size=384)
    registry.register(spec)
    pipeline = ImagePipeline(model_name="oracle-image", registry=registry)
    backend = RegionOracleBackend(pipeline._spec)
    registry._loaded["oracle-image"] = LoadedModel(
        spec=pipeline._spec, backend=backend, loaded_at=datetime.now(UTC), load_ms=0
    )

    detector = ScriptedDetector(boxes)
    original = pre.detect_faces
    pre.detect_faces = detector  # type: ignore[assignment]
    try:
        data = encode(painted(regions))
        bundle = asyncio.run(pipeline.analyze_image(spec_for(data), data))
    finally:
        pre.detect_faces = original  # type: ignore[assignment]
    return bundle, Rig(pipeline, detector, backend)


def describe(bundle: EvidenceBundle) -> str:
    p = "None" if bundle.p_fake is None else f"{bundle.p_fake:.6f}"
    reason = bundle.abstain_reason.value if bundle.abstain_reason else "-"
    return f"verdict={bundle.verdict.value:<22} p_fake={p:<10} reason={reason}"


def box_str(b: FaceBox) -> str:
    return f"{b.width}x{b.height}@({b.x},{b.y}) area={b.area:<7} short={b.short_edge:<4} long={max(b.width, b.height)}"


# --------------------------------------------------------------------------
# 0 — the doubles
# --------------------------------------------------------------------------


def s0_doubles() -> None:
    head(0, "The doubles", "what stands in for what?")
    print(f"  detector double : {ScriptedDetector.name}")
    print(f"  backend double  : {RegionOracleBackend.name}")
    print(f"  legacy order    : {LegacyAreaOrder.name}")
    print()
    print("  Haar cannot find two faces on one canvas (multi_face_study stage 0), so")
    print("  the multi-face scenarios script the detector. Everything between the")
    print("  detector and the bundle -- _gate, _abstain, _crop_and_score, crop_face,")
    print("  prepare_image, fuse, calibrated_probability, decide, build_image_bundle --")
    print("  is the shipped pipeline on a real event loop.")
    print()
    print(f"  shipped ordering rule : {pre.ORDERING_RULE}")
    print(f"  legacy ordering rule  : {LegacyAreaOrder.rule}")
    print(f"  shipped gate floor    : min_face_px = {FLOOR}")
    print(f"  detection floor       : detect_min_px = {pre.default_face_selection().detect_min_px}")


# --------------------------------------------------------------------------
# 1 — every face-size concept
# --------------------------------------------------------------------------


def s1_concepts() -> None:
    head(1, "Every face-size concept", "how many quantities describe 'how big is this face'?")

    b = FaceBox(x=100, y=100, width=900, height=50)
    print(f"  One box: {b}")
    print(f"    width          {b.width}")
    print(f"    height         {b.height}")
    print(f"    area           {b.area}      = width * height")
    print(f"    short edge     {b.short_edge}         = min(width, height)")
    print(f"    long edge      {max(b.width, b.height)}        = max(width, height)")
    print(f"    size_px        {getattr(b, 'size_px', '(absent before Batch 14)')}")
    print()

    root = Path(__file__).resolve().parent.parent
    tokens = (
        "short_edge",
        "size_px",
        ".area",
        "min_face_px",
        "min_short_edge",
        "detect_min_px",
        "face_px",
    )
    print("  Where each name is read, outside tests:")
    for rel in (
        "inference/preprocess.py",
        "inference/image_pipeline.py",
        "inference/video_pipeline.py",
        "inference/postprocess.py",
        "contracts/common.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        counts = {t: text.count(t) for t in tokens if text.count(t)}
        print(f"    {rel:<34} {counts}")
    print()
    print("  Two names collide and mean different things:")
    print(f"    MediaCaps.min_short_edge = {pre.DEFAULT_CAPS.min_short_edge}  -> the *image's* short edge")
    print(f"    MediaCaps.min_face_px    = {pre.DEFAULT_CAPS.min_face_px}  -> the *face's* short edge")
    print("  Same number, same shape of comparison, different subject. Nothing in the")
    print("  names says so.")


# --------------------------------------------------------------------------
# 2 — which decision reads which metric
# --------------------------------------------------------------------------


def s2_decisions() -> None:
    head(2, "Which decision reads which metric", "the mapping, from source")

    rows = [
        ("detector floor (minSize)", "short edge, per axis", "policy.detect_min_px", "preprocess.detect_faces"),
        ("image resolution gate", "image short edge", "caps.min_short_edge", "image_pipeline._gate"),
        ("face selection / ranking", "SEE BELOW", "-", "preprocess.face_order_key"),
        ("face size gate", "short edge", "caps.min_face_px", "image_pipeline._gate"),
        ("published quality.face_px", "short edge", "-", "image_pipeline.analyze_image"),
        ("crop side length", "LONG edge", "policy.crop_margin", "preprocess.crop_face"),
    ]
    print(f"  {'decision':<28} {'metric':<22} {'threshold':<22} site")
    for name, metric, thresh, site in rows:
        print(f"  {name:<28} {metric:<22} {thresh:<22} {site}")

    probe = FaceBox(0, 0, 200, 100)
    ranking_metric = "area" if face_order_key(probe)[0] == -probe.area else "short edge"
    print(f"\n  ranking metric, read off face_order_key on {probe}: {ranking_metric}")
    print(f"    face_order_key = {face_order_key(probe)}")
    print(f"    legacy key     = {LegacyAreaOrder.key(probe)}")
    print()
    print("  So a single box is measured three different ways on its way through one")
    print("  analysis: ranked one way, gated another, cropped a third. Under square")
    print("  boxes all three coincide, which is why nothing has ever noticed.")
    print()
    print("  The crop row is the one the R19 write-up does not mention. crop_face")
    print("  computes  side = max(width, height) * crop_margin  -- the LONG edge --")
    print("  so the gate approves a face on its short edge and the crop is scaled by")
    print("  its long edge. Stage 8.")


# --------------------------------------------------------------------------
# 3 — the four cases
# --------------------------------------------------------------------------


def _scorable(b: FaceBox, floor: int = FLOOR) -> bool:
    return b.short_edge >= floor


def s3_cases() -> None:
    head(3, "The four cases", "which of them can actually be constructed?")

    # (a) area-largest != short-edge-largest
    wide = FaceBox(x=100, y=100, width=400, height=100)
    square = FaceBox(x=700, y=100, width=190, height=190)
    print("  (a) area-largest != short-edge-largest")
    print(f"        {box_str(wide)}")
    print(f"        {box_str(square)}")
    print(f"      area-largest       : {LegacyAreaOrder.order([wide, square])[0]}")
    print(f"      short-edge-largest : {max([wide, square], key=lambda b: b.short_edge)}")
    print("      CONSTRUCTIBLE.")

    # (b) area-largest scorable, short-edge-largest not -- provably empty
    print("\n  (b) area-largest is scorable but short-edge-largest is not")
    print("      PROVABLY EMPTY, and the proof is one line:")
    print("        the gate is  short_edge >= floor.")
    print("        short-edge-largest maximises exactly the gated quantity.")
    print("        so if it fails, every box fails -- including the area-largest one.")
    rng = np.random.default_rng(19)
    searched = witnesses = 0
    for _ in range(200_000):
        n = int(rng.integers(2, 5))
        boxes = [
            FaceBox(0, 0, int(rng.integers(1, 400)), int(rng.integers(1, 400)))
            for _ in range(n)
        ]
        searched += 1
        by_area = max(boxes, key=lambda b: b.area)
        by_short = max(boxes, key=lambda b: b.short_edge)
        if _scorable(by_area) and not _scorable(by_short):
            witnesses += 1
    print(f"      random search: {searched} box sets, {witnesses} witnesses")
    print("      -> the case is not rare. It is impossible.")

    # (c) short-edge-largest scorable, area-largest not
    print("\n  (c) short-edge-largest is scorable but area-largest is not  == R19")
    thin = FaceBox(x=100, y=100, width=900, height=50)
    good = FaceBox(x=100, y=300, width=200, height=200)
    print(f"        {box_str(thin)}   below the {FLOOR}px floor")
    print(f"        {box_str(good)}   clears it")
    sel, scored = LegacyAreaOrder.decision([thin, good])
    print(f"      legacy policy selects {sel} -> scored={scored}")
    bundle, rig = run_image([thin, good], [(thin, BRIGHT), (good, BRIGHT)])
    print(f"      through the real pipeline: {describe(bundle)}")
    print(f"        crops scored     : {len(rig.backend.seen)}")
    print(f"        faces[] published: {[(f.bbox.width, f.bbox.height, f.status.value if f.status else None) for f in bundle.faces]}")
    print(f"        quality.face_px  : {bundle.quality.face_px}")
    print("      CONSTRUCTIBLE.")

    # (d) ranking and gating disagree, measured
    print("\n  (d) ranking and gating disagree -- how often, over random geometry?")
    rng = np.random.default_rng(1914)
    trials = 200_000
    diverge = pathological = 0
    for _ in range(trials):
        n = int(rng.integers(2, 5))
        boxes = [
            FaceBox(0, 0, int(rng.integers(1, 400)), int(rng.integers(1, 400)))
            for _ in range(n)
        ]
        by_area = LegacyAreaOrder.order(boxes)[0]
        by_short = sorted(boxes, key=face_order_key)[0]
        if by_area != by_short:
            diverge += 1
        if not _scorable(by_area) and any(_scorable(b) for b in boxes):
            pathological += 1
    print(f"      {trials} random box sets, free aspect, edges 1..400")
    print(f"        area-first != shipped-first        : {diverge} ({100*diverge/trials:.1f}%)")
    print(f"        legacy abstains on a scorable file : {pathological} ({100*pathological/trials:.1f}%)")
    print("      CONSTRUCTIBLE, and common once aspect is free.")


# --------------------------------------------------------------------------
# 4 — contradiction, or two policies?
# --------------------------------------------------------------------------


def s4_contradiction() -> None:
    head(4, "Contradiction or two policies", "is the pairing defective, or merely plural?")

    print("  The claim to test: 'rank by area' and 'gate by short edge' are two")
    print("  independent, individually defensible policies. Are they compatible?")
    print()
    print("  THEOREM (gate completeness). Let r be the ranking metric and g the gate")
    print("  metric. The state")
    print("      faces[0] fails the gate  AND  some other face would pass")
    print("  is reachable iff r and g are not monotonically related.")
    print("      If r = g up to a monotone map:  r(b1) > r(b2)  =>  g(b1) >= g(b2),")
    print("      so g(b1) < floor forces g(b2) < floor for every b2. No witness.")
    print()
    print("  area and short edge are NOT monotonically related:")
    for a, b in [
        (FaceBox(0, 0, 900, 50), FaceBox(0, 0, 200, 200)),
        (FaceBox(0, 0, 400, 100), FaceBox(0, 0, 190, 190)),
    ]:
        print(f"      area({a.width}x{a.height})={a.area} > area({b.width}x{b.height})={b.area}"
              f"   but short={a.short_edge} < short={b.short_edge}")
    print()
    print("  So the pairing admits a state in which the platform publishes")
    print("  FACE_TOO_SMALL for a file containing a face that clears every floor.")
    print("  That is not a plural policy. It is a false statement about the file,")
    print("  emitted by a gate whose stated purpose (image_pipeline.py, first")
    print("  paragraph) is to route *low-information* input to honest abstention.")
    print()

    print("  HOW FAR FROM SQUARE? The exact activation threshold.")
    print("    Write d = max |width - height| the detector can emit, F = min_face_px.")
    print("    A witness needs  se1 < F <= se2  and  area1 > area2, so")
    print("        area1 <= (F-1)(F-1+d)      area2 >= F*F")
    print("        (F-1)(F-1+d) > F^2   <=>   d > 2 + 1/(F-1)")
    print(f"    At F = {FLOOR}:  d > {2 + 1 / (FLOOR - 1):.4f}, i.e. **d >= 3 pixels**.")
    for F in (32, 64, 128, 512):
        print(f"      floor {F:<5} -> d > {2 + 1/(F-1):.4f}  -> d >= 3")
    print("    The threshold is 3 pixels at every floor. Verified by brute force:")

    # brute force the minimal witness
    best = None
    for d in range(0, 6):
        for se1 in range(1, FLOOR):
            for se2 in range(FLOOR, FLOOR + 40):
                a1 = se1 * (se1 + d)
                a2 = se2 * se2
                if a1 > a2:
                    if best is None or d < best[0]:
                        best = (d, se1, se1 + d, se2)
                    break
            if best and best[0] == d:
                break
        if best:
            break
    d, w1, l1, se2 = best
    print(f"      minimal witness: {l1}x{w1} (area {w1*l1}, short {w1}) beats "
          f"{se2}x{se2} (area {se2*se2}, short {se2}) on area")
    print(f"      required aspect deviation: |w-h| = {d} px, aspect ratio {l1/w1:.4f}")

    print("\n  The same threshold governs the ORDERING. area-desc and short-edge-desc")
    print("  disagree iff  se1 > se2 and area1 < area2, which needs d > 2 + 1/se2,")
    print("  i.e. d >= 3 as well. So:")
    print("      d <= 2 : the two orderings are identical AND R19 is unreachable")
    print("      d >= 3 : the two orderings differ AND R19 is reachable")
    print("  Changing the ranking metric is a no-op exactly where the defect cannot")
    print("  fire, and takes effect exactly where it can. Verified exhaustively:")

    same = differ = 0
    for w1 in range(30, 90):
        for h1 in range(max(1, w1 - 2), w1 + 3):
            for w2 in range(30, 90, 3):
                for h2 in range(max(1, w2 - 2), w2 + 3):
                    a, b = FaceBox(0, 0, w1, h1), FaceBox(0, 0, w2, h2)
                    if sorted([a, b], key=face_order_key)[0] == LegacyAreaOrder.order([a, b])[0]:
                        same += 1
                    else:
                        differ += 1
    print(f"      all box pairs with |w-h| <= 2, edges 30..90: {same} agree, {differ} differ")


# --------------------------------------------------------------------------
# 5 — is it reachable under Haar?
# --------------------------------------------------------------------------


def s5_reachability() -> None:
    head(5, "Reachability under Haar", "does the shipped detector really return square boxes?")

    cascade = cv2.CascadeClassifier(pre.CASCADE_PATH)
    print(f"  cascade window size: {cascade.getOriginalWindowSize()}  (square)")
    print("  Candidates are that window scaled, so the review's reasoning was that")
    print("  every box is square and area order and short-edge order coincide.")
    print()

    rng = np.random.default_rng(2026)
    shapes: list[tuple[int, int]] = []
    for _ in range(150):
        img = cv2.GaussianBlur(rng.integers(0, 255, (300, 400), dtype=np.uint8), (5, 5), 0)
        for (_x, _y, w, h) in cascade.detectMultiScale(
            img, scaleFactor=1.1, minNeighbors=0, minSize=(32, 32)
        ):
            shapes.append((int(w), int(h)))
    deltas = sorted({abs(w - h) for w, h in shapes})
    nonsquare = [(w, h) for w, h in shapes if w != h]
    worst = max((max(w / h, h / w) for w, h in shapes), default=1.0)
    print(f"  {len(shapes)} raw candidates (grouping disabled), noise images:")
    print(f"    non-square      : {len(nonsquare)} ({100*len(nonsquare)/max(len(shapes),1):.2f}%)")
    print(f"    |w-h| observed  : {deltas}")
    print(f"    worst aspect    : {worst:.4f}   examples {nonsquare[:4]}")
    print()
    print("  **The review's premise is false as stated.** Haar does not return square")
    print("  boxes; it returns boxes within ONE pixel of square. The per-axis rounding")
    print("  in OpenCV's scale pipeline is not contractually square, and neither is")
    print("  groupRectangles' averaging of a cluster.")
    print()
    print(f"  But stage 4 gives the threshold: d >= 3. Measured d <= 1. So R19 is")
    print("  unreachable today -- for a reason two pixels wide, resting on an OpenCV")
    print("  implementation detail this platform does not control, rather than on the")
    print("  structural guarantee the review believed it had.")
    print()
    print("  SCRFD regresses box corners as free floats. A 100 px face at the aspect")
    print("  ratios face detectors actually produce (w/h around 0.75-0.8) has")
    print("  |w-h| around 25 px -- eight times the threshold. It arms immediately.")


# --------------------------------------------------------------------------
# 6 — alternatives
# --------------------------------------------------------------------------


def s6_alternatives() -> None:
    head(6, "Alternatives", "six policies, measured on the same box sets")

    rng = np.random.default_rng(77)
    sets = []
    for _ in range(50_000):
        n = int(rng.integers(2, 5))
        sets.append([
            FaceBox(0, 0, int(rng.integers(1, 400)), int(rng.integers(1, 400)))
            for _ in range(n)
        ])

    def evaluate(pick) -> tuple[int, int]:
        """(false abstentions, sets where any face was scorable)."""
        bad = live = 0
        for boxes in sets:
            any_ok = any(_scorable(b) for b in boxes)
            if any_ok:
                live += 1
                if not _scorable(pick(boxes)):
                    bad += 1
        return bad, live

    policies = {
        "rank by area, gate short edge (legacy)": lambda bs: LegacyAreaOrder.order(bs)[0],
        "rank by short edge (shipped B14)": lambda bs: sorted(bs, key=face_order_key)[0],
        "rank by scoreability then area": lambda bs: sorted(
            bs, key=lambda b: (not _scorable(b), -b.area, b.y, b.x, b.width, b.height)
        )[0],
        "rank by long edge": lambda bs: sorted(
            bs, key=lambda b: (-max(b.width, b.height), b.y, b.x)
        )[0],
    }
    print(f"  {len(sets)} random box sets, free aspect, edges 1..400, floor {FLOOR}")
    print(f"  {'policy':<40} {'false abstentions':>18}")
    for name, pick in policies.items():
        bad, live = evaluate(pick)
        print(f"  {name:<40} {bad:>7} / {live:<8} ({100*bad/max(live,1):.2f}%)")
    print()
    print("  'False abstention' = the file contained a face clearing every floor and")
    print("  the policy still published FACE_TOO_SMALL.")
    print()
    print("  Two policies score zero. They differ in what they cost:")
    print("    rank by short edge          -- one metric, ordering stays a function of")
    print("                                   geometry alone, and face_px's documented")
    print("                                   meaning becomes true (R20's defect")
    print("                                   condition disappears).")
    print("    rank by scoreability, area  -- also zero, but the ordering now depends on")
    print("                                   MediaCaps.min_face_px, a *deployment* knob")
    print("                                   in a *different* fingerprint. Two")
    print("                                   fingerprints would jointly determine which")
    print("                                   face is scored, and the platform would")
    print("                                   still carry two face-size metrics.")


# --------------------------------------------------------------------------
# 7 — migration impact
# --------------------------------------------------------------------------


def s7_migration() -> None:
    head(7, "Migration impact", "what moves, and what does not?")

    shipped = pre.default_face_selection()
    legacy = replace(shipped, ordering=LegacyAreaOrder.rule)
    caps = pre.DEFAULT_CAPS

    def version(policy: pre.FaceSelectionPolicy) -> str:
        return (
            ModelSpec(name="stub-image", version="stub-0", backend="stub")
            .with_gate(caps.gate_fingerprint)
            .with_face_selection(policy.identity)
            .effective_version
        )

    print(f"  ordering rule   before : {LegacyAreaOrder.rule}")
    print(f"                  after  : {shipped.ordering}")
    print(f"  face-selection  before : {legacy.identity}")
    print(f"                  after  : {shipped.identity}")
    print(f"  effective_ver   before : {version(legacy)}")
    print(f"                  after  : {version(shipped)}")
    print(f"  moved                  : {version(legacy) != version(shipped)}")
    print(f"  string length          : {len(version(legacy))} -> {len(version(shipped))}")
    print()

    from contracts.version import is_face_selection_pinned, parse_model_version

    for label, v in (("before", version(legacy)), ("after", version(shipped))):
        parts = parse_model_version(v)
        print(f"  parse({label:<6}) weights={parts.weights} calibration={parts.calibration} "
              f"gate={parts.gate} face_selection={parts.face_selection} "
              f"pinned={is_face_selection_pinned(v)}")
    print()
    print(f"  gate fingerprint       : {caps.gate_fingerprint}  (unchanged -- the floor")
    print("                           did not move, only what it is compared against)")
    print()
    print("  Artifact by artifact:")
    print("    result cache      one-time full miss, both media kinds. Namespaces do")
    print("                      not overlap, so nothing stale is served.")
    print("    idempotency keys  derived per submission; in-flight duplicates re-enqueue.")
    print("    benchmark runs    readable, RUN_SCHEMA_VERSION unchanged. A pre- and a")
    print("                      post-B14 run compare as different models -- which is")
    print("                      conservative rather than correct here, because under")
    print("                      Haar the two produce identical numbers. Cost: one")
    print("                      refused comparison across the boundary.")
    print("    calibration       untouched. A selection change is not a calibration.")
    print("    stored bundles    readable. ModelInfo.version is a str.")
    print("    split locks       untouched.")
    print("    OpenAPI / schema  unchanged; no field added, removed or retyped.")
    print("    CONTRACT_VERSION  unchanged.")


# --------------------------------------------------------------------------
# 8 — the residual: crop geometry reads the long edge
# --------------------------------------------------------------------------


def s8_crop() -> None:
    head(8, "The residual", "the crop is scaled by the long edge, and clipped")

    policy = pre.default_face_selection()
    img = np.full((600, 1200, 3), 128, np.uint8)
    print(f"  crop_shape = {policy.crop_shape!r}, crop_margin = {policy.crop_margin}")
    print(f"  side = max(width, height) * margin -- the LONG edge\n")
    print(f"  {'box':<22} {'side':>6} {'crop shape':>14} {'square?':>8} {'face fraction':>14}")
    for b in (
        FaceBox(100, 100, 200, 200),
        FaceBox(100, 100, 260, 200),
        FaceBox(100, 100, 400, 100),
        FaceBox(100, 100, 900, 50),
    ):
        crop = pre.crop_face(img, b, policy=policy)
        side = int(max(b.width, b.height) * policy.crop_margin)
        h, w = crop.shape[:2]
        fraction = (b.width * b.height) / float(h * w)
        print(f"  {b.width}x{b.height}@({b.x},{b.y}){'':<4} {side:>6} {f'{w}x{h}':>14} "
              f"{str(w == h):>8} {fraction:>13.1%}")
    print()
    print("  Two things this shows, neither of them R19 and neither fixed here:")
    print("    1. A free-aspect box is cropped at its LONG edge, so the fraction of the")
    print("       crop that is face falls as aspect diverges -- 59% of the model's")
    print("       input for a square box, 6% for a 900x50 one.")
    print("    2. The crop is clipped to the image, so it is not always square, and")
    print("       prepare_image then resizes a rectangle to a square -- the aspect")
    print("       distortion crop_face's own docstring says it exists to avoid.")
    print()
    print("  Both are unreachable under a detector whose boxes are within a pixel of")
    print("  square, both are armed by the same migration that arms R19, and both are")
    print("  outside this batch's brief. Recorded as R23.")


# --------------------------------------------------------------------------


def main() -> int:
    print(__doc__)
    s0_doubles()
    s1_concepts()
    s2_decisions()
    s3_cases()
    s4_contradiction()
    s5_reachability()
    s6_alternatives()
    s7_migration()
    s8_crop()
    print(f"\n{'=' * 74}\ndone\n{'=' * 74}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
