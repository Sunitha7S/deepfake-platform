"""R19 — the platform measured one face two ways and let the two disagree.

THE DEFECT, AS A SENTENCE
------------------------
``face_order_key`` ranked boxes by **area**; ``ImagePipeline._gate`` then tested
the winner's **short edge** against ``MediaCaps.min_face_px``. Area and short
edge are not monotonically related, so the platform could select a box on one
measure and refuse it on the other:

    900x50   area 45000  short edge  50   <- selected by area, refused by the gate
    200x200  area 40000  short edge 200   <- never scored, never published

and publish ``insufficient_evidence / face_too_small`` for a file containing a
200 px face. Not "the wrong face was picked" — a false statement about the file,
from a gate whose stated purpose is to route *low-information* input to honest
abstention.

WHAT BATCH 14 CHANGED
---------------------
One quantity, :attr:`FaceBox.size_px`, named by ``FACE_SIZE_METRIC``, read by
the ranking, the gate and ``quality.face_px``. Ranking now maximises the gated
quantity, which makes the state unrepresentable rather than merely unlikely: if
``faces[0]`` fails the floor, every box does.

WHY THIS CHANGES NO PUBLISHED NUMBER TODAY
------------------------------------------
The two orderings are identical for every pair of boxes within **two pixels** of
square, and diverge from three — the same threshold at which R19 itself becomes
reachable. Haar's boxes are within one pixel of square, measured over 1 025 of
them. So the fix is a no-op exactly where the defect cannot fire and takes effect
exactly where it can. Both halves are proved below by exhaustive enumeration
rather than asserted.

WHY DOUBLES
-----------
``LegacyAreaOrder`` is the pre-Batch-14 ``face_order_key``, verbatim, so every
claim is a pair: the double abstains and the shipped code does not.
``ScriptedDetector`` and ``RegionOracleBackend`` are the ``multi_face_study``
doubles — Haar cannot see a two-face canvas, and a hash-derived stub has no
ground truth. Everything between them is the shipped pipeline.

WHAT THIS FILE DOES NOT CLAIM
-----------------------------
R16, R20, R21, R22 and R18's capability half are untouched, and §7 pins them so
the closure cannot be read as wider than it is. §7 also pins **R23**, found while
tracing: ``crop_face`` scales the crop by the box's *long* edge and clips it, so
a free-aspect box yields a crop that is mostly background and not square.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.enums import CONTRACT_VERSION, AbstainReason, BranchId, MediaKind, Verdict  # noqa: E402
from contracts.evidence import EvidenceBundle  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.image_pipeline import ImagePipeline  # noqa: E402
from inference.models import ModelRegistry, ModelSpec  # noqa: E402
from inference.preprocess import FaceBox, face_order_key  # noqa: E402
from tests.detector_fixtures import haar_selection  # noqa: E402

FLOOR = pre.DEFAULT_CAPS.min_face_px
_HAAR = haar_selection()  # gate/ordering mechanics and Haar's own boxes are under test


# --------------------------------------------------------------------------
# The named doubles
# --------------------------------------------------------------------------


class LegacyAreaOrder:
    """The pre-Batch-14 ``face_order_key`` and the gate it fed, verbatim.

    ``(-area, y, x, width, height)`` — the rule Batch 10 introduced to close R14
    and Batch 12 fingerprinted as ``area-desc,y-asc,x-asc,width-asc,height-asc``.
    Ranking on area while the gate tests the short edge *is* R19, and keeping the
    composition here is what lets the tests below demonstrate the defect instead
    of describing it. If this double stops abstaining on the witness, it has
    stopped reproducing R19 and ``test_the_legacy_double_still_reproduces_r19``
    says so rather than passing quietly.
    """

    name = "rank by area, gate on short edge (pre-Batch-14)"
    rule = "area-desc,y-asc,x-asc,width-asc,height-asc"

    @staticmethod
    def key(box: FaceBox) -> tuple[int, int, int, int, int]:
        return (-box.area, box.y, box.x, box.width, box.height)

    @classmethod
    def order(cls, boxes: list[FaceBox]) -> list[FaceBox]:
        return sorted(boxes, key=cls.key)

    @classmethod
    def scores_the_file(cls, boxes: list[FaceBox], floor: int = FLOOR) -> bool:
        if not boxes:
            return False
        return cls.order(boxes)[0].short_edge >= floor


class ScriptedDetector:
    """Stands in for the detector stage. Returns a fixed box list.

    Replaces ``inference.preprocess.detect_faces`` and nothing else — the
    pipeline's whole contract with the detector is ``list[FaceBox]``, and the
    list is passed through the *shipped* ordering, so what is scripted is the
    detection and what is under test is the order. Haar cannot find two faces on
    one canvas (``multi_face_study`` stage 0), and SCRFD will return boxes of
    exactly this shape from NMS.
    """

    name = "scripted box list, ordered by the shipped face_order_key"

    def __init__(self, boxes: list[FaceBox]) -> None:
        self.boxes = list(boxes)

    def __call__(self, image: np.ndarray, *, policy=None) -> list[FaceBox]:
        return sorted(self.boxes, key=face_order_key)


class RegionOracleBackend:
    """Stands in for real weights. Logit from the crop's own mean luminance.

    Dark ⇒ authentic (−3.0), bright ⇒ manipulated (+3.0), both branches. This is
    what gives a scenario a ground truth: paint one region dark and one bright,
    then read back which one the platform published. ``StubBackend`` hashes
    pixels and cannot.
    """

    name = "logit from crop mean luminance: dark = authentic, bright = manipulated"

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.seen: list[float] = []

    def score(self, batch: np.ndarray) -> dict[BranchId, np.ndarray]:
        out: dict[BranchId, list[float]] = {b: [] for b in self.spec.branch_weights}
        for i in range(batch.shape[0]):
            mean = float(batch[i].mean())
            self.seen.append(mean)
            logit = -3.0 if mean <= 96 else (3.0 if mean >= 160 else 0.0)
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


THIN = FaceBox(x=100, y=100, width=900, height=50)   # area 45000, size_px  50
GOOD = FaceBox(x=100, y=300, width=200, height=200)  # area 40000, size_px 200

DARK, BRIGHT = 40, 220


def _painted(regions: list[tuple[FaceBox, int]]) -> np.ndarray:
    img = np.full((600, 1200, 3), 128, np.uint8)
    for box, level in regions:
        img[box.y : box.y + box.height, box.x : box.x + box.width] = level
    return img


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def _spec(data: bytes) -> JobSpec:
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


def _analyse(
    boxes: list[FaceBox], regions: list[tuple[FaceBox, int]]
) -> tuple[EvidenceBundle, RegionOracleBackend]:
    """Scripted boxes, oracle backend, real ``analyze_image`` in between."""
    from inference.models import LoadedModel

    registry = ModelRegistry()
    registry.register(ModelSpec(name="oracle", version="oracle-0", backend="stub", input_size=384))
    pipeline = ImagePipeline(model_name="oracle", registry=registry, face_selection=_HAAR)
    backend = RegionOracleBackend(pipeline._spec)
    registry._loaded["oracle"] = LoadedModel(
        spec=pipeline._spec, backend=backend, loaded_at=datetime.now(UTC), load_ms=0
    )

    original = pre.detect_faces
    pre.detect_faces = ScriptedDetector(boxes)  # type: ignore[assignment]
    try:
        data = _encode(_painted(regions))
        bundle = asyncio.run(pipeline.analyze_image(_spec(data), data))
    finally:
        pre.detect_faces = original  # type: ignore[assignment]
    return bundle, backend


def _face_image(size: int = 480) -> np.ndarray:
    """The determinism study's synthetic face, verbatim. Real Haar sees this one."""
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (175, 160, 150), -1)
    cv2.circle(img, (200, 210), 16, (60, 60, 60), -1)
    cv2.circle(img, (280, 210), 16, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 300), (18, 30), 0, 0, 360, (150, 130, 120), -1)
    cv2.ellipse(img, (240, 340), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _random_boxes(rng, n: int, hi: int = 400) -> list[FaceBox]:
    return [
        FaceBox(0, 0, int(rng.integers(1, hi)), int(rng.integers(1, hi))) for _ in range(n)
    ]


# --------------------------------------------------------------------------
# 1. The defect, reproduced and closed
# --------------------------------------------------------------------------


def test_the_legacy_double_still_reproduces_r19():
    """The double must actually fail, or every pairing below is vacuous."""
    assert LegacyAreaOrder.order([THIN, GOOD])[0] == THIN, (
        "the legacy key no longer prefers the thin box; it has stopped "
        "reproducing R19 and the tests that pair against it prove nothing"
    )
    assert not LegacyAreaOrder.scores_the_file([THIN, GOOD]), (
        "the legacy composition no longer abstains on the witness"
    )
    assert GOOD.short_edge >= FLOOR, "the witness must contain a genuinely scorable face"


def test_the_shipped_pipeline_scores_the_file_the_legacy_policy_refused():
    """Fail-before-fix, through the real ``analyze_image``.

    Same two boxes, same canvas. Before Batch 14 this published
    ``insufficient_evidence / face_too_small`` with zero crops scored, while a
    200 px face sat in the image untouched.
    """
    bundle, backend = _analyse([THIN, GOOD], [(THIN, BRIGHT), (GOOD, BRIGHT)])

    assert bundle.verdict is not Verdict.INSUFFICIENT_EVIDENCE, (
        f"the file abstained ({bundle.abstain_reason}) although it contains a "
        f"{GOOD.width}x{GOOD.height} face clearing the {FLOOR}px floor — R19"
    )
    assert bundle.quality.gate_reason is not AbstainReason.FACE_TOO_SMALL
    assert len(backend.seen) == 1, "exactly one crop is scored; R16 is not in scope here"
    assert (bundle.faces[0].bbox.width, bundle.faces[0].bbox.height) == (
        GOOD.width,
        GOOD.height,
    ), "the published box is not the face the gate approved"


def test_a_scorable_face_is_never_gated_away_by_another_one():
    """The property, over random geometry. Fail-before-fix.

    Not "rarely" and not "not in our fixtures": if any box in the list clears
    the floor then ``faces[0]`` clears it, for every list.
    """
    rng = np.random.default_rng(19)
    for _ in range(20_000):
        boxes = _random_boxes(rng, int(rng.integers(2, 5)))
        if not any(b.size_px >= FLOOR for b in boxes):
            continue
        chosen = sorted(boxes, key=face_order_key)[0]
        assert chosen.size_px >= FLOOR, (
            f"faces[0]={chosen} fails the {FLOOR}px floor although "
            f"{[b for b in boxes if b.size_px >= FLOOR][0]} clears it"
        )


def test_the_legacy_policy_fails_that_same_property():
    """The pairing. Same generator, same seed, and the double must lose."""
    rng = np.random.default_rng(19)
    failures = 0
    for _ in range(20_000):
        boxes = _random_boxes(rng, int(rng.integers(2, 5)))
        if not any(b.short_edge >= FLOOR for b in boxes):
            continue
        if not LegacyAreaOrder.scores_the_file(boxes):
            failures += 1
    assert failures > 0, (
        "the legacy composition no longer produces a false abstention on this "
        "generator; the fail-before evidence has gone stale"
    )


# --------------------------------------------------------------------------
# 2. One metric, and the theorem that makes it enough
# --------------------------------------------------------------------------


def test_the_face_size_metric_name_matches_the_metric():
    """``FACE_SIZE_METRIC`` is a name, so something must hold it to behaviour.

    Same contract ``ORDERING_RULE`` has had since Batch 12, for the same reason:
    a name in an identity string that drifts from the code is worse than no name.
    """
    assert pre.FACE_SIZE_METRIC == "short-edge"
    rng = np.random.default_rng(3)
    for _ in range(2_000):
        b = _random_boxes(rng, 1)[0]
        assert b.size_px == min(b.width, b.height), (
            f"FACE_SIZE_METRIC names {pre.FACE_SIZE_METRIC!r} but size_px is "
            f"{b.size_px} for {b}"
        )


def test_the_ranking_and_the_gate_read_the_same_quantity():
    """The structural half of the fix, asserted rather than trusted.

    The ordering's leading component must be the negated gate metric. If either
    side is edited to use area, long edge or anything else, this fails — which is
    the only thing standing between the platform and R19 recurring, because the
    gate's *metric* is in no fingerprint (§7).
    """
    rng = np.random.default_rng(5)
    for _ in range(2_000):
        b = _random_boxes(rng, 1)[0]
        assert face_order_key(b)[0] == -b.size_px, (
            f"face_order_key ranks on {-face_order_key(b)[0]} but the gate tests "
            f"size_px={b.size_px} for {b}; R19 is back"
        )

    gate_source = (
        Path(__file__).resolve().parent.parent / "inference/image_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "faces[0].size_px < self._caps.min_face_px" in gate_source, (
        "the face gate no longer compares faces[0].size_px against min_face_px"
    )


def test_ranking_on_the_gated_quantity_makes_the_defect_unrepresentable():
    """The theorem, stated as code.

    ``faces[0]`` maximises ``size_px``. The gate tests ``size_px``. So
    ``faces[0]`` failing the floor implies every box fails it, for every floor —
    not just the shipped 64.
    """
    rng = np.random.default_rng(7)
    for floor in (1, 16, 32, 64, 128, 399):
        for _ in range(2_000):
            boxes = _random_boxes(rng, int(rng.integers(1, 6)))
            chosen = sorted(boxes, key=face_order_key)[0]
            if chosen.size_px < floor:
                assert all(b.size_px < floor for b in boxes), (
                    f"floor={floor}: faces[0]={chosen} was refused while "
                    f"{[b for b in boxes if b.size_px >= floor]} would have passed"
                )


def test_the_opposite_disagreement_is_impossible():
    """"area-largest is scorable but short-edge-largest is not" has no witness.

    Asked by the brief, and the answer is a proof rather than an example: the
    gate is ``short_edge >= floor`` and the short-edge-largest box maximises
    exactly that, so if it fails then every box fails — the area-largest one
    included. This holds for the *legacy* ordering too, so it is a fact about the
    gate rather than about the fix, and it is why only one of the two directions
    was ever a defect.
    """
    rng = np.random.default_rng(11)
    for _ in range(50_000):
        boxes = _random_boxes(rng, int(rng.integers(2, 5)))
        by_area = max(boxes, key=lambda b: b.area)
        by_short = max(boxes, key=lambda b: b.short_edge)
        if by_area.short_edge >= FLOOR:
            assert by_short.short_edge >= FLOOR, (
                f"impossible witness found: {by_area} scorable, {by_short} not"
            )


# --------------------------------------------------------------------------
# 3. The three-pixel threshold
# --------------------------------------------------------------------------


def _pathology_exists(d: int, floor: int) -> bool:
    """Is there a box pair with |w-h| <= d where area-ranking refuses a scorable file?"""
    for se1 in range(1, floor):
        for delta1 in range(0, d + 1):
            for se2 in range(floor, floor + 4 * d + 8):
                for delta2 in range(0, d + 1):
                    if se1 * (se1 + delta1) > se2 * (se2 + delta2):
                        return True
    return False


def test_the_activation_threshold_is_three_pixels():
    """R19 needs a box three pixels from square. Not "free aspect" — three pixels.

    A witness needs ``se1 < F <= se2`` and ``area1 > area2``, so at most
    ``(F-1)(F-1+d)`` must exceed at least ``F*F``, i.e. ``d > 2 + 1/(F-1)``. The
    threshold is 3 at every floor. Brute-forced here rather than trusted to the
    algebra.
    """
    for floor in (16, 32, 64, 128):
        assert not _pathology_exists(2, floor), f"floor {floor}: a witness exists at d=2"
        assert _pathology_exists(3, floor), f"floor {floor}: no witness at d=3"

    # The minimal witness at the shipped floor, named.
    thin, square = FaceBox(0, 0, 66, 63), FaceBox(0, 0, 64, 64)
    assert thin.area > square.area and thin.short_edge < FLOOR <= square.short_edge
    assert LegacyAreaOrder.order([thin, square])[0] == thin
    assert sorted([thin, square], key=face_order_key)[0] == square


def test_the_two_orderings_agree_below_the_threshold():
    """Behaviour preservation, exhaustively.

    For every pair of boxes within two pixels of square, the shipped ordering and
    the pre-Batch-14 one put the same box first. ``se1 > se2`` forces
    ``area1 >= se1^2 >= (se2+1)^2 > se2(se2+2) >= area2``, so the leading
    components cannot disagree; equal short edges fall through to area, which is
    the legacy key's own leading component.
    """
    pairs = 0
    for w1 in range(1, 70):
        for h1 in range(max(1, w1 - 2), min(70, w1 + 3)):
            for w2 in range(1, 70, 3):
                for h2 in range(max(1, w2 - 2), min(70, w2 + 3)):
                    a, b = FaceBox(0, 0, w1, h1), FaceBox(5, 5, w2, h2)
                    pairs += 1
                    assert (
                        sorted([a, b], key=face_order_key)[0]
                        == LegacyAreaOrder.order([a, b])[0]
                    ), f"the orderings disagree on near-square boxes {a} / {b}"
    assert pairs > 10_000, "the enumeration shrank; it is no longer exhaustive enough to mean much"


def test_the_two_orderings_diverge_at_the_threshold():
    """And the fix is not vacuous: at three pixels they differ. Fail-before-fix."""
    a, b = FaceBox(0, 0, 66, 63), FaceBox(0, 0, 64, 64)
    assert LegacyAreaOrder.order([a, b])[0] != sorted([a, b], key=face_order_key)[0], (
        "the shipped ordering still agrees with the legacy one at |w-h|=3; the "
        "ranking metric did not actually change"
    )


def test_square_boxes_order_identically():
    """The case that covers every box the shipped detector has ever emitted."""
    boxes = [FaceBox(x, y, s, s) for s, x, y in ((50, 0, 0), (200, 10, 10), (200, 300, 5), (64, 7, 9))]
    for perm in itertools.permutations(boxes):
        assert sorted(perm, key=face_order_key) == LegacyAreaOrder.order(list(perm))


def test_the_ordering_is_still_a_total_order():
    """R14 must survive R19's fix: no two distinct boxes may share a key."""
    rng = np.random.default_rng(13)
    sample = [FaceBox(int(rng.integers(0, 300)), int(rng.integers(0, 300)),
                      int(rng.integers(1, 300)), int(rng.integers(1, 300)))
              for _ in range(5_000)]
    keys = {face_order_key(b) for b in sample}
    assert len(keys) == len(set(sample)), "two distinct boxes share an ordering key"

    # The pair an area-plus-origin key misses, and the pair a short-edge-only key
    # would miss.
    for a, b in (
        (FaceBox(0, 0, 100, 200), FaceBox(0, 0, 200, 100)),
        (FaceBox(0, 0, 100, 100), FaceBox(0, 0, 100, 400)),
    ):
        assert face_order_key(a) != face_order_key(b)


# --------------------------------------------------------------------------
# 4. Behaviour under the shipped detector
# --------------------------------------------------------------------------


def test_the_shipped_detector_emits_boxes_within_the_safe_margin():
    """The measurement the whole "no behaviour change" claim rests on.

    The Batch 11 review said Haar "returns square boxes". It does not — per-axis
    rounding in OpenCV's scale pipeline emits boxes one pixel off square, rarely.
    It stays inside the two-pixel margin, which is the claim that matters, and
    this test fails loudly if a future OpenCV widens it: at that point the two
    orderings can diverge under Haar and the behaviour-preservation argument in
    ``docs/FACE_SIZE_POLICY.md`` needs re-running, not patching.
    """
    cascade = cv2.CascadeClassifier(pre.CASCADE_PATH)
    assert cascade.getOriginalWindowSize() == (24, 24)

    rng = np.random.default_rng(2026)
    worst = 0
    seen = 0
    for _ in range(40):
        img = cv2.GaussianBlur(rng.integers(0, 255, (300, 400), dtype=np.uint8), (5, 5), 0)
        for (_x, _y, w, h) in cascade.detectMultiScale(
            img, scaleFactor=1.1, minNeighbors=0, minSize=(32, 32)
        ):
            seen += 1
            worst = max(worst, abs(int(w) - int(h)))
    assert seen > 50, "too few candidates to say anything; the fixture stopped detecting"
    assert worst <= 2, (
        f"Haar emitted a box {worst} px from square. The three-pixel threshold is "
        "reachable under the shipped detector, so R19 is no longer latent and the "
        "identical-ordering argument no longer holds"
    )


def test_the_real_detection_path_is_unchanged():
    """End to end on the fixture real Haar can see: same box, same crop bytes."""
    img = _face_image()
    found = pre.detect_faces(img, policy=_HAAR)
    assert found, "the fixture must detect at least one face"
    assert found == sorted(found, key=face_order_key)
    assert found == LegacyAreaOrder.order(list(found)), (
        "the shipped and legacy orderings disagree on real Haar output"
    )
    assert found[0].width == found[0].height, "the shipped fixture's box is square"

    crop = pre.crop_face(img, found[0])
    assert crop.shape[0] > 0 and crop.shape[1] > 0


def test_the_published_face_px_is_the_largest_face_size_present():
    """``quality.face_px`` now describes the largest face by the one metric.

    Fail-before-fix: under area ranking this reported 50 for a file whose largest
    face measured 200.
    """
    bundle, _ = _analyse([THIN, GOOD], [(THIN, BRIGHT), (GOOD, BRIGHT)])
    assert bundle.quality.face_px == max(THIN.size_px, GOOD.size_px) == 200, (
        f"face_px={bundle.quality.face_px}, but the largest face measures "
        f"{max(THIN.size_px, GOOD.size_px)}"
    )


def test_a_file_whose_every_face_is_too_small_still_abstains():
    """The fix must not have turned the gate off."""
    a = FaceBox(x=100, y=100, width=40, height=40)
    b = FaceBox(x=400, y=100, width=50, height=50)
    bundle, backend = _analyse([a, b], [(a, BRIGHT), (b, BRIGHT)])
    assert bundle.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert bundle.quality.gate_reason is AbstainReason.FACE_TOO_SMALL
    assert backend.seen == [], "a refused file must not reach the model"


# --------------------------------------------------------------------------
# 5. Identity and migration
# --------------------------------------------------------------------------


def _legacy_policy() -> pre.FaceSelectionPolicy:
    return replace(pre.default_face_selection(), ordering=LegacyAreaOrder.rule)


def test_the_face_selection_identity_moved():
    """The policy changed, so the fingerprint must have. Fail-before-fix."""
    shipped = pre.default_face_selection()
    assert shipped.ordering == "short-edge-desc,area-desc,y-asc,x-asc,width-asc,height-asc"
    assert shipped.identity != _legacy_policy().identity, (
        "the ordering rule changed and the face-selection fingerprint did not; a "
        "cache keyed on it would serve results computed under the other policy"
    )
    assert shipped.differences(_legacy_policy()) == [
        f"ordering: {shipped.ordering!r} != {LegacyAreaOrder.rule!r}"
    ], "exactly one field moved, and it is the ordering rule"


def test_the_engine_version_literal_was_updated():
    """The configured literal must name the policy this build actually runs.

    S8 Batch 23 (R29): the ``.f.`` segment is the *combined* selection identity
    (face + video), so the literal is checked against
    ``selection_identity(default_face_selection(), default_video_selection())``,
    not ``face_selection_identity()`` alone.
    """
    from backend.config import STUB_ENGINE_VERSION
    from contracts.version import face_selection_identity_of

    combined = pre.selection_identity(
        pre.default_face_selection(), pre.default_video_selection()
    )
    spec = (
        ModelSpec(name="stub-image", version="stub-0", backend="stub")
        .with_gate(pre.DEFAULT_CAPS.gate_fingerprint)
        .with_face_selection(combined)
    )
    assert STUB_ENGINE_VERSION.startswith(spec.effective_version), (
        f"the selection policy moved; update backend.config."
        f"STUB_ENGINE_VERSION to start with {spec.effective_version}"
    )
    assert face_selection_identity_of(STUB_ENGINE_VERSION) == combined


def test_the_gate_fingerprint_did_not_move():
    """The floor did not change — only what it is compared against.

    Recorded because it is the one thing a reader would assume moved: this is a
    change to the *gate's* honesty, and the gate fingerprint is untouched. The
    metric lives with the selection policy, where the ordering rule names it.
    """
    assert pre.DEFAULT_CAPS.gate_fingerprint == "ae232cba"
    assert pre.DEFAULT_CAPS.min_face_px == 64
    assert pre.DEFAULT_CAPS.min_short_edge == 64


def test_old_version_strings_still_parse():
    """Backward compatibility: a pre-Batch-14 string is readable, not an error."""
    from contracts.version import is_face_selection_pinned, parse_model_version

    old = "stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.d662e22d.d.c20a064a"
    parts = parse_model_version(old)
    assert parts.weights == "stub-0-w7f45fada"
    assert parts.gate == "ae232cba"
    assert parts.face_selection == "d662e22d"
    assert is_face_selection_pinned(old)

    pre_b12 = "stub-0-w7f45fada+c.739c3fec.g.ae232cba.d.c20a064a"
    assert parse_model_version(pre_b12).face_selection is None
    assert not is_face_selection_pinned(pre_b12)


def test_the_contract_version_did_not_move():
    """No field added, removed or retyped. Nothing on the wire changed shape."""
    assert CONTRACT_VERSION == "1.0.0"


# --------------------------------------------------------------------------
# 6. What this does NOT close
# --------------------------------------------------------------------------


def test_r16_is_untouched():
    """One face still speaks for the file. The *right* face now; still one face."""
    big = FaceBox(x=100, y=100, width=300, height=300)
    small = FaceBox(x=700, y=150, width=200, height=200)
    bundle, backend = _analyse([big, small], [(big, DARK), (small, BRIGHT)])
    assert len(backend.seen) == 1, "R16 was implemented; this batch's brief forbade it"
    assert len(bundle.faces) == 1
    assert bundle.verdict is Verdict.AUTHENTIC, (
        "the manipulated smaller face is still invisible to the published verdict"
    )


def test_r21_is_untouched():
    """``faces[]`` still has no uniqueness or consistency invariant."""
    validators = set(EvidenceBundle.__pydantic_decorators__.model_validators)
    assert not any("faces_unique" in name or "faces_agree" in name for name in validators), (
        "a faces[] consistency invariant appeared; that is R21 and it was out of scope"
    )


def test_the_video_path_is_untouched():
    """No face stage was added, and none was implied."""
    source = (
        Path(__file__).resolve().parent.parent / "inference/video_pipeline.py"
    ).read_text(encoding="utf-8")
    assert source.count("detect_faces") == 0
    assert source.count("size_px") == 0


def test_the_crop_is_still_scaled_by_the_long_edge():
    """**R23**, found while tracing and deliberately not fixed.

    ``crop_face`` computes ``side = max(width, height) * margin``. So the gate
    approves a face on its short edge and the crop is scaled by its long edge —
    a third metric on the same box. For a free-aspect box the face occupies a
    vanishing fraction of what the model is shown.

    Pinned rather than fixed: it is unreachable under a detector whose boxes are
    within a pixel of square, it is armed by the same migration that arms R19,
    and it is outside this batch's brief. If it is fixed, this test fails and
    asks to be replaced — the pattern Batch 9 established and Batches 11 and 13
    honoured.
    """
    img = np.full((600, 1200, 3), 128, np.uint8)
    policy = pre.default_face_selection()
    assert policy.crop_shape == "square-longest-edge"

    square_crop = pre.crop_face(img, FaceBox(100, 100, 200, 200), policy=policy)
    assert square_crop.shape[0] == square_crop.shape[1] == int(200 * policy.crop_margin)

    thin_crop = pre.crop_face(img, THIN, policy=policy)
    face_fraction = (THIN.width * THIN.height) / float(thin_crop.shape[0] * thin_crop.shape[1])
    assert face_fraction < 0.10, (
        "the crop geometry changed; R23's measurement is stale and this test "
        "should be replaced rather than adjusted"
    )
    assert thin_crop.shape[0] != thin_crop.shape[1], (
        "a clipped crop is not square, so prepare_image resizes a rectangle to a "
        "square — the aspect distortion crop_face's docstring says it avoids"
    )


def test_the_gate_metric_is_in_no_fingerprint():
    """The residual limit, asserted so §7 of the report cannot overclaim.

    ``MediaCaps.gate_fingerprint`` covers the floor, not the quantity compared
    against it. Editing ``_gate`` to test ``area`` would move no identity string
    anywhere in the platform — only
    ``test_the_ranking_and_the_gate_read_the_same_quantity`` stands in the way.
    Same shape of guarantee ``ORDERING_RULE`` has: a name held to behaviour by a
    test rather than by a hash.
    """
    caps = pre.MediaCaps()
    before = caps.gate_fingerprint
    fingerprint_inputs = pre.MediaCaps.gate_fingerprint.fget.__doc__ or ""
    assert "min_face_px" in fingerprint_inputs
    assert caps.gate_fingerprint == before
    assert "size_px" not in str(pre.default_face_selection().to_dict()), (
        "the metric is named through ORDERING_RULE, not as a separate field"
    )
    assert pre.FACE_SIZE_METRIC in pre.ORDERING_RULE, (
        "the ordering rule no longer names the face-size metric, so nothing in "
        "any identity string records which quantity ranks and gates"
    )


# --------------------------------------------------------------------------


def _run() -> int:
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failed += 1
                print(f"  FAIL {name}: {exc}")
            except Exception as exc:  # pragma: no cover - surfaces real breakage
                failed += 1
                print(f"  ERR  {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
