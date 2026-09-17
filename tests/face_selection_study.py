"""R17 — face selection is verdict-affecting and was in no identity string.

    python tests/face_selection_study.py

Produces every number in ``docs/FACE_SELECTION_IDENTITY.md``. Same pattern as
``decode_study.py``, ``band_study.py``, ``inference_determinism_study.py`` and
``multi_face_study.py``.

THE QUESTION
------------
Batch 9 put the decoder into the runtime identity because Batch 8 measured two
decoders publishing different verdicts under one byte-identical version string.
R17, raised by the Batch 12 architecture review, says the same sentence about a
different set of knobs:

    Can two deployments select a different face, crop it differently, and publish
    different verdicts, while agreeing on every identity string the platform has?

The answer is **yes**, by six mechanisms, and this study measures each. Five of
the six were hard-coded literals or default arguments — the kind of value nobody
thinks of as configuration precisely because it never appears in a config file.

WHY THE STUB BACKEND IS ENOUGH, AND WHAT IT CANNOT SHOW
--------------------------------------------------------
``StubBackend`` derives its logits from a hash of the crop's bytes. So the
*direction* of every score change below is meaningless — it is arithmetic, not
judgement, and this study never claims otherwise.

What the stub does establish is the part that matters for an identity argument:
the crop bytes change, and a scorer that is a function of its input therefore
changes its answer. The hard measurement is the crop digest; the verdict flip is
the consequence made visible. A real model would move by a different amount in a
different direction, and it would still move.

THE DOUBLES
-----------
``LegacyHardcodedSelection`` is the pre-Batch-12 ``detect_faces`` and
``crop_face``, literals and all, kept verbatim. ``SelectionBlindVersion`` is the
pre-Batch-12 ``compose_model_version``, three lines, kept next to the new one.
Each stage that claims a fix compares the two against the same fixture.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.enums import MediaKind  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from contracts.version import (  # noqa: E402
    CALIBRATION_MARKER,
    GATE_MARKER,
    compose_model_version,
    compose_runtime_version,
    face_selection_identity_of,
    is_face_selection_pinned,
    model_identity_of,
)
from inference import postprocess as post  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.image_pipeline import ImagePipeline  # noqa: E402
from inference.models import ModelSpec, StubBackend  # noqa: E402
from inference.preprocess import FaceBox, FaceSelectionPolicy, face_order_key  # noqa: E402
from inference.video_pipeline import VideoPipeline  # noqa: E402
from workers.io import InMemoryResultSink  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dfp-faceselect-study-"))
SHIPPED = pre.default_face_selection()


def head(n: int, title: str, question: str) -> None:
    print(f"\n{'=' * 76}\n{n}. {title}\n   {question}\n{'=' * 76}")


# --------------------------------------------------------------------------
# The named doubles
# --------------------------------------------------------------------------


class LegacyHardcodedSelection:
    """The pre-Batch-12 detector and cropper. Literals baked in, verbatim.

    ``detect_faces`` carried ``scaleFactor=1.1, minNeighbors=5`` inline and a
    ``min_px: int = 32`` keyword; ``crop_face`` carried ``margin: float = 1.3``.
    None of the four reached any identity string, which is R17.

    Kept as a double rather than described so stage 6 can prove the shipped
    default policy reproduces this behaviour *exactly* — a refactor that quietly
    changed a number while adding a fingerprint for it would be worse than the
    defect.
    """

    name = "hard-coded literals + default arguments (pre-Batch-12)"

    @staticmethod
    def detect(image: np.ndarray, *, min_px: int = 32) -> list[FaceBox]:
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        found = pre._face_cascade().detectMultiScale(
            grey, scaleFactor=1.1, minNeighbors=5, minSize=(min_px, min_px)
        )
        boxes = [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in found]
        return sorted(boxes, key=face_order_key)

    @staticmethod
    def crop(image: np.ndarray, box: FaceBox, *, margin: float = 1.3) -> np.ndarray:
        h, w = image.shape[:2]
        side = int(max(box.width, box.height) * margin)
        cx, cy = box.x + box.width // 2, box.y + box.height // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
        return image[y0:y1, x0:x1]


class SelectionBlindVersion:
    """The pre-Batch-12 ``compose_model_version``. No face-selection segment.

    Verbatim in the only respect that matters: it composes weights, calibration
    and gate, so two deployments running different selection policies produce the
    same string and share a cache namespace.
    """

    name = "weights + calibration + gate (pre-Batch-12)"

    @staticmethod
    def compose(*, weights: str, calibration: str, gate: str | None = None) -> str:
        composed = f"{weights}{CALIBRATION_MARKER}{calibration}"
        if gate is not None:
            composed = f"{composed}{GATE_MARKER}{gate}"
        return composed


class SelectionAwareVersion:
    """The shipped composition. Delegates, so the study cannot drift from it."""

    name = "weights + calibration + gate + face selection (Batch 12)"

    @staticmethod
    def compose(*, weights: str, calibration: str, gate: str | None, face: str) -> str:
        return compose_model_version(
            weights=weights, calibration=calibration, gate=gate, face_selection=face
        )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def face_image(size: int = 480) -> np.ndarray:
    """The determinism study's synthetic face, verbatim — it detects reliably."""
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (175, 160, 150), -1)
    cv2.circle(img, (200, 210), 16, (60, 60, 60), -1)
    cv2.circle(img, (280, 210), 16, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 300), (18, 30), 0, 0, 360, (150, 130, 120), -1)
    cv2.ellipse(img, (240, 340), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    return cv2.GaussianBlur(img, (5, 5), 0)


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:10]


def encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:  # pragma: no cover
        raise RuntimeError("failed to encode fixture")
    return buf.tobytes()


def spec_for(data: bytes, kind: MediaKind = MediaKind.IMAGE, mime: str = "image/png") -> JobSpec:
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


def score_under(img: np.ndarray, policy: FaceSelectionPolicy):
    """Detect, crop and score one image under one policy. No pipeline, no gates.

    Returns ``(boxes, crop_digest, p_fake, verdict)`` — or ``None`` for the last
    three when the detector found nothing, which is itself an outcome.
    """
    spec = ModelSpec(name="s", version="stub-0", backend="stub", input_size=384)
    boxes = pre.detect_faces(img, policy=policy)
    if not boxes:
        return boxes, None, None, "no_face_detected"
    crop = pre.crop_face(img, boxes[0], policy=policy)
    batch = pre.prepare_image(crop, size=384)
    logits = StubBackend(spec).score(batch)
    outcomes = {b: post.BranchOutcome(logit=float(v[0])) for b, v in logits.items()}
    p = post.calibrated_probability(post.fuse(outcomes, spec.branch_weights), spec)
    verdict, _ = post.decide(p, spec.threshold)
    return boxes, sha(crop), p, verdict.value


# --------------------------------------------------------------------------
# 1 — enumerate the variables
# --------------------------------------------------------------------------


def s1_enumerate() -> None:
    head(1, "Every variable that influences image face selection",
         "what are they, where do they live, and were they fingerprinted?")
    rows = [
        ("detector implementation", "opencv-haar-frontalface-default",
         "preprocess.CASCADE_PATH", "no"),
        ("detector asset (cascade XML)", pre.cascade_digest(),
         "cv2.data.haarcascades, on disk", "no"),
        ("scaleFactor", "1.1", "literal inside detect_faces", "no"),
        ("minNeighbors", "5", "literal inside detect_faces", "no"),
        ("detection floor (minSize)", "32", "detect_faces(min_px=32) default arg", "no"),
        ("ordering rule", pre.ORDERING_RULE, "face_order_key", "no"),
        ("crop margin", "1.3", "crop_face(margin=1.3) default arg", "no"),
        ("crop shape", "square-longest-edge", "crop_face body", "no"),
        ("gate floor (min_face_px)", str(pre.DEFAULT_CAPS.min_face_px),
         "MediaCaps", "YES — gate fingerprint"),
        ("image floor (min_short_edge)", str(pre.DEFAULT_CAPS.min_short_edge),
         "MediaCaps", "YES — gate fingerprint"),
    ]
    print(f"  {'variable':<30} {'shipped value':<40} {'fingerprinted before B12'}")
    print(f"  {'-' * 30} {'-' * 40} {'-' * 24}")
    for name, value, _where, fp in rows:
        print(f"  {name:<30} {value:<40} {fp}")
    print()
    print("  Two floors, not one. `detect_min_px`=32 decides which boxes EXIST;")
    print("  `min_face_px`=64 decides which of them is big enough to score. Only the")
    print("  second was ever in a version string, so the pair that jointly decides an")
    print("  abstention was half covered.")
    print()
    print(f"  Shipped policy identity: {SHIPPED.identity}")


# --------------------------------------------------------------------------
# 2 — sensitivity
# --------------------------------------------------------------------------


def s2_sensitivity() -> None:
    head(2, "Which variables reach the verdict",
         "vary one knob at a time, hold everything else")
    img = face_image()
    base_boxes, base_crop, base_p, base_v = score_under(img, SHIPPED)
    print(f"  baseline: box={tuple(base_boxes[0].__dict__.values()) if base_boxes else None} "
          f"crop={base_crop} p={base_p:.6f} {base_v}\n")

    def row(label: str, policy: FaceSelectionPolicy, shipped: bool = False) -> None:
        boxes, crop, p, verdict = score_under(img, policy)
        mark = "   <- shipped" if shipped else ""
        p_txt = "     -    " if p is None else f"{p:.6f}"
        print(f"    {label:<22} n={len(boxes)}  crop={crop or '   none   '}  "
              f"p={p_txt}  {verdict}{mark}")

    print("  --- scale_factor ---")
    for v in (1.05, 1.1, 1.2, 1.3):
        row(f"scale_factor {v}", replace(SHIPPED, scale_factor=v), v == 1.1)

    print("\n  --- min_neighbors ---")
    for v in (3, 5, 8, 12):
        row(f"min_neighbors {v}", replace(SHIPPED, min_neighbors=v), v == 5)

    print("\n  --- detect_min_px ---")
    for v in (32, 64, 260, 320):
        row(f"detect_min_px {v}", replace(SHIPPED, detect_min_px=v), v == 32)

    print("\n  --- crop_margin ---")
    for v in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0):
        row(f"crop_margin {v}", replace(SHIPPED, crop_margin=v), v == 1.3)

    print("\n  --- ordering rule (needs two faces; scripted boxes) ---")
    wide, square = FaceBox(100, 100, 400, 100), FaceBox(700, 100, 190, 190)
    for name, key in (
        ("area-desc (shipped)", lambda b: (-b.area, b.y, b.x, b.width, b.height)),
        ("short-edge-desc", lambda b: (-b.short_edge, b.y, b.x, b.width, b.height)),
        ("top-left-first", lambda b: (b.y, b.x, -b.area)),
    ):
        chosen = sorted([wide, square], key=key)[0]
        print(f"    {name:<22} faces[0] = ({chosen.x},{chosen.y},{chosen.width},{chosen.height})")

    print("\n  Every one of these is a different published answer for the same bytes,")
    print("  the same weights, the same calibration, the same gate and the same decoder.")


# --------------------------------------------------------------------------
# 3 — the identity question
# --------------------------------------------------------------------------


def s3_identity() -> None:
    head(3, "Do two such runs share an identity?",
         "runtime, benchmark, calibration and decode identity, before and after")
    img = face_image()
    tight = replace(SHIPPED, crop_margin=1.2)

    _, crop_a, p_a, v_a = score_under(img, SHIPPED)
    _, crop_b, p_b, v_b = score_under(img, tight)

    pipe_a = ImagePipeline(face_selection=SHIPPED)
    pipe_b = ImagePipeline(face_selection=tight)

    weights = pipe_a._spec.weights_identity
    cal = pipe_a._spec.calibration_fingerprint
    gate = pipe_a._spec.gate_fingerprint

    legacy_a = SelectionBlindVersion.compose(weights=weights, calibration=cal, gate=gate)
    legacy_b = SelectionBlindVersion.compose(weights=weights, calibration=cal, gate=gate)

    print(f"  margin 1.3 : crop={crop_a} p={p_a:.6f} {v_a}")
    print(f"  margin 1.2 : crop={crop_b} p={p_b:.6f} {v_b}")
    print(f"  same crop?  {crop_a == crop_b}      same verdict?  {v_a == v_b}\n")

    print(f"  BEFORE — {SelectionBlindVersion.name}")
    print(f"    margin 1.3 -> {legacy_a}")
    print(f"    margin 1.2 -> {legacy_b}")
    print(f"    identical: {legacy_a == legacy_b}   <- R17\n")

    print(f"  AFTER  — {SelectionAwareVersion.name}")
    print(f"    margin 1.3 -> {pipe_a.model_identity}")
    print(f"    margin 1.2 -> {pipe_b.model_identity}")
    print(f"    identical: {pipe_a.model_identity == pipe_b.model_identity}\n")

    print("  Which identities the new segment reaches, and which it does not:")
    print(f"    model identity    : {pipe_a.model_identity != pipe_b.model_identity}"
          "   (weights+cal+gate+face)")
    print(f"    runtime identity  : {pipe_a.model_version != pipe_b.model_version}"
          "   (model + decode; inherited)")
    print(f"    benchmark identity: {pipe_a.model_identity != pipe_b.model_identity}"
          "   (BenchmarkRun.effective_model_version carries the model identity)")
    print("    calibration fp    : False   (unchanged by design — a crop margin is")
    print("                                 not a calibration; see the report §5)")
    print(f"    decode identity   : False   (unchanged by design — selection is code,")
    print("                                 not environment)")


# --------------------------------------------------------------------------
# 4 — the cache proof
# --------------------------------------------------------------------------


def s4_cache() -> None:
    head(4, "The consequence", "does the cache serve one policy's verdict to another?")
    img = face_image()
    data = encode(img)

    async def run(policy):
        pipe = ImagePipeline(face_selection=policy)
        bundle = await pipe.analyze_image(spec_for(data), data)
        return pipe, bundle

    pipe_wide, wide = asyncio.run(run(SHIPPED))
    pipe_tight, tight = asyncio.run(run(replace(SHIPPED, crop_margin=1.2)))

    print(f"  margin 1.3 -> {wide.verdict.value:<22} p_fake={wide.p_fake}")
    print(f"  margin 1.2 -> {tight.verdict.value:<22} p_fake={tight.p_fake}\n")

    legacy_key = model_identity_of  # pre-B12: the model identity had no .f. segment
    blind_wide = SelectionBlindVersion.compose(
        weights=pipe_wide._spec.weights_identity,
        calibration=pipe_wide._spec.calibration_fingerprint,
        gate=pipe_wide._spec.gate_fingerprint,
    )
    blind_tight = SelectionBlindVersion.compose(
        weights=pipe_tight._spec.weights_identity,
        calibration=pipe_tight._spec.calibration_fingerprint,
        gate=pipe_tight._spec.gate_fingerprint,
    )
    store: dict[tuple[str, str], object] = {(wide.media.sha256, blind_wide): wide}
    hit = store.get((wide.media.sha256, blind_tight))
    print(f"  BEFORE, {SelectionBlindVersion.name}:")
    print(f"    a margin-1.2 deployment looks up its key -> "
          f"{'HIT: ' + hit.verdict.value if hit is not None else 'miss'}")
    if hit is not None:
        print(f"    it is served p_fake={hit.p_fake}, computed under margin 1.3,")
        print(f"    when its own crop gives {tight.p_fake}")

    async def real_sink():
        sink = InMemoryResultSink()
        await sink.put_if_absent(str(wide.job_id), wide)
        same = await sink.find_by_content(wide.media.sha256, pipe_wide.model_version)
        cross = await sink.find_by_content(wide.media.sha256, pipe_tight.model_version)
        return same, cross

    same, cross = asyncio.run(real_sink())
    print(f"\n  AFTER, against the real InMemoryResultSink:")
    print(f"    same policy  -> {'hit' if same is not None else 'MISS'}")
    print(f"    other policy -> {'HIT (defect)' if cross is not None else 'miss'}")
    print(f"    {legacy_key(pipe_wide.model_version) != legacy_key(pipe_tight.model_version)}"
          "  the two model identities differ, so the namespaces do not overlap")


# --------------------------------------------------------------------------
# 5 — alternatives
# --------------------------------------------------------------------------


def s5_alternatives() -> None:
    head(5, "Alternatives", "which identity should have carried this?")
    tight = replace(SHIPPED, crop_margin=1.2)
    pipe_a, pipe_b = ImagePipeline(face_selection=SHIPPED), ImagePipeline(face_selection=tight)

    checks = [
        ("cache key (result reuse)", pipe_a.model_version != pipe_b.model_version),
        ("idempotency key", pipe_a.model_version != pipe_b.model_version),
        ("/health + alignment", pipe_a.model_version != pipe_b.model_version),
        ("benchmark comparison", pipe_a.model_identity != pipe_b.model_identity),
        ("forensic reconstruction", bool(SHIPPED.to_dict())),
    ]
    print("  What the shipped choice — a segment of the MODEL identity — covers:")
    for label, ok in checks:
        print(f"    {label:<28} {'covered' if ok else 'NOT covered'}")

    print("\n  What each rejected alternative would have covered:")
    table = [
        ("runtime identity only", "cache, health", "benchmark comparison — a policy "
         "change would read as environment noise"),
        ("benchmark identity only", "regression gates", "the cache — the live defect"),
        ("calibration identity only", "cache and benchmark, incidentally",
         "meaning: a crop margin is not a calibration"),
        ("dedicated face-selection identity", "everything, by inheritance",
         "nothing — this is what shipped"),
        ("full detector fingerprint", "everything, plus the cascade asset",
         "nothing — the asset digest IS included; the label is a subset of the above"),
        ("policy version only (a hand-set string)", "nothing reliably",
         "the number it names — a human must remember to bump it"),
    ]
    for name, covers, misses in table:
        print(f"    {name}")
        print(f"       covers : {covers}")
        print(f"       misses : {misses}")

    print("\n  Note the asset digest is not decoration:")
    print(f"    cascade sha256[:16] = {pre.cascade_digest()}")
    other = replace(SHIPPED, detector_asset="0" * 16)
    print(f"    a reshipped cascade under the same filename moves the identity: "
          f"{SHIPPED.identity != other.identity}")


# --------------------------------------------------------------------------
# 6 — the fix changes identity, not behaviour
# --------------------------------------------------------------------------


def s6_behaviour_preserved() -> None:
    head(6, "Behaviour preservation",
         "does the default policy reproduce the pre-Batch-12 code exactly?")
    img = face_image()

    legacy_boxes = LegacyHardcodedSelection.detect(img)
    shipped_boxes = pre.detect_faces(img, policy=SHIPPED)
    print(f"  {LegacyHardcodedSelection.name}")
    print(f"    legacy  boxes: {[(b.x, b.y, b.width, b.height) for b in legacy_boxes]}")
    print(f"    shipped boxes: {[(b.x, b.y, b.width, b.height) for b in shipped_boxes]}")
    print(f"    identical: {legacy_boxes == shipped_boxes}")

    legacy_crop = LegacyHardcodedSelection.crop(img, legacy_boxes[0])
    shipped_crop = pre.crop_face(img, shipped_boxes[0], policy=SHIPPED)
    print(f"\n    legacy  crop: {sha(legacy_crop)}  {legacy_crop.shape}")
    print(f"    shipped crop: {sha(shipped_crop)}  {shipped_crop.shape}")
    print(f"    byte-identical: {sha(legacy_crop) == sha(shipped_crop)}")

    print("\n  The values did not move. Only the version string did — which is the")
    print("  whole of R17's fix: the same pixels, now under a name that says which")
    print("  policy produced them.")


# --------------------------------------------------------------------------
# 7 — migration impact
# --------------------------------------------------------------------------


def s7_migration() -> None:
    head(7, "Migration impact", "what invalidates, and what still reads?")
    from backend.config import STUB_ENGINE_VERSION

    old = "stub-0-w7f45fada+c.739c3fec.g.ae232cba.d.c20a064a"
    print(f"  STUB_ENGINE_VERSION before: {old}")
    print(f"  STUB_ENGINE_VERSION after : {STUB_ENGINE_VERSION}")
    print(f"  length {len(old)} -> {len(STUB_ENGINE_VERSION)} characters\n")

    print("  Backward compatibility — every pre-Batch-12 string still parses:")
    from contracts.version import parse_model_version

    parts = parse_model_version(old)
    print(f"    parse('{old[:34]}...')")
    print(f"      weights        = {parts.weights}")
    print(f"      calibration    = {parts.calibration}")
    print(f"      gate           = {parts.gate}")
    print(f"      face_selection = {parts.face_selection}   <- None, not an error")
    print(f"    is_face_selection_pinned(old) = {is_face_selection_pinned(old)}")
    print(f"    is_face_selection_pinned(new) = {is_face_selection_pinned(STUB_ENGINE_VERSION)}")
    print(f"    face_selection_identity_of(new) = {face_selection_identity_of(STUB_ENGINE_VERSION)}")
    print("      ^ Batch 23 (R29): the .f. segment now carries face AND video")
    print("        selection, combined by selection_identity(). For a video-era")
    print("        string face_selection_identity_of() returns that composite, not")
    print("        the face policy alone — face_selection_identity() stays face-only.")

    print("\n  Artifact-by-artifact:")
    for artifact, impact in [
        ("result cache", "one-time full miss; namespaces do not overlap, nothing stale is served"),
        ("job idempotency keys", "in-flight duplicates re-enqueue once; keys are per-submission"),
        ("benchmark reports", "readable; effective_model_version gains a segment, so a "
                              "pre-B12 run and a post-B12 run compare as different models — correct"),
        ("calibration witnesses", "unchanged; no schema bump, no stored witness re-digests"),
        ("stored bundles", "readable; ModelInfo.version is still a str, no schema change"),
        ("OpenAPI snapshot", "unchanged — the field is a string and got longer"),
        ("CONTRACT_VERSION", "unchanged at 1.0.0 — no contract field changed shape or meaning"),
    ]:
        print(f"    {artifact:<24} {impact}")


# --------------------------------------------------------------------------
# 8 — the video engine carries what it does not consume
# --------------------------------------------------------------------------


def s8_video() -> None:
    head(8, "The video engine", "it detects no faces — why does it carry the segment?")
    image = ImagePipeline()
    video = VideoPipeline()
    print(f"  image engine: {image.model_version}")
    print(f"  video engine: {video.model_version}")
    print(f"  equal: {image.model_version == video.model_version}\n")

    src = (Path(__file__).resolve().parent.parent / "inference/video_pipeline.py").read_text(
        encoding="utf-8"
    )
    print(f"  'detect_faces' in video_pipeline.py: {src.count('detect_faces')} occurrence(s)")
    print(f"  'crop_face'    in video_pipeline.py: {src.count('crop_face')} occurrence(s)\n")

    print("  So the video engine's verdicts cannot move when the policy moves — and it")
    print("  carries the segment anyway, because Settings.model_version is ONE string")
    print("  that keys the cache for both media kinds and verify_version_alignment")
    print("  requires every engine to report it. An image-only segment would fail")
    print("  startup or force a per-kind configured version, which is a serving change.")
    print()
    print("  Cost: the video cache namespace changes when a policy video does not use")
    print("  changes. That is a false miss -- compute, not correctness -- and RUN_IDENTITY")
    print("  section 3 already settles which way a cache key breaks a tie.")


# --------------------------------------------------------------------------


def main() -> int:
    print(__doc__)
    s1_enumerate()
    s2_sensitivity()
    s3_identity()
    s4_cache()
    s5_alternatives()
    s6_behaviour_preserved()
    s7_migration()
    s8_video()
    print(f"\n{'=' * 76}\nartifacts in {TMP}\n{'=' * 76}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
