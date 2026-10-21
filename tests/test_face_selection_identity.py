"""R17 — the face-selection policy was outside every identity string.

THE DEFECT, AS A SENTENCE
-------------------------
The result cache is keyed on ``(sha256, model_version)``. Until Batch 12
``model_version`` composed weights, calibration, gate and decode — and none of
those four move when the detector's ``scaleFactor`` changes, when the ordering
rule that picks ``faces[0]`` changes, or when the crop margin changes from 1.3 to
1.2. Measured in ``tests/face_selection_study.py`` on one image, one model, one
decoder:

    crop_margin 1.3 -> p_fake 0.013562  authentic              (shipped)
    crop_margin 1.2 -> p_fake 0.659734  insufficient_evidence
    crop_margin 1.0 -> p_fake 0.889291  manipulated

Three published verdicts, one byte-identical version string. So the cache could
store a bundle cropped at 1.3 and serve it to a deployment cropping at 1.2, and
nothing anywhere would record that it had. This file proves it against the real
cache implementation, using a named double for the old composition.

WHY DOUBLES RATHER THAN A GIT CHECKOUT
--------------------------------------
``SelectionBlindVersion`` is the pre-Batch-12 ``compose_model_version``, three
lines, kept next to the new one. ``LegacyHardcodedSelection`` is the pre-Batch-12
``detect_faces`` and ``crop_face`` with their literals and default arguments
baked in, verbatim. Every claim below is a pair: the double does the wrong thing
and the shipped code does not, measured against the same fixture. A test that
only exercises the fix cannot tell you the fix was needed — and the second double
carries a duty the first does not, because a refactor that quietly *changed* one
of those numbers while adding a fingerprint for it would be worse than the defect
it fixed.

WHAT IS AND IS NOT CLAIMED
--------------------------
Putting the policy in the version makes two policies **distinguishable**. It does
not make either of them correct, it does not choose the right face (R16), and it
does not give the video path a face stage (R18). The video engine carries the
segment without consuming it, deliberately, and that is asserted here rather than
left to the report — an untested limit drifts into an assumed guarantee.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.enums import CONTRACT_VERSION, BranchId, MediaKind  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from contracts.version import (  # noqa: E402
    CALIBRATION_MARKER,
    DECODE_MARKER,
    FACE_SELECTION_MARKER,
    GATE_MARKER,
    compose_model_version,
    compose_runtime_version,
    face_selection_identity_of,
    is_face_selection_pinned,
    model_identity_of,
    parse_model_version,
)
from inference import preprocess as pre  # noqa: E402
from inference.image_pipeline import ImagePipeline  # noqa: E402
from inference.models import ModelSpec  # noqa: E402
from inference.preprocess import FaceBox, face_order_key  # noqa: E402
from inference.video_pipeline import VideoPipeline  # noqa: E402
from workers.io import InMemoryResultSink  # noqa: E402

SHIPPED = pre.default_face_selection()
TIGHT = replace(SHIPPED, crop_margin=1.2)
"""The shipped policy and the one measured to publish a different verdict.

1.2 rather than an arbitrary value: the study measured it moving this fixture
from ``authentic`` at 0.013562 to ``insufficient_evidence`` at 0.659734, so the
fixture and the finding are the same case.
"""

PRE_BATCH_12 = "stub-0-w7f45fada+c.739c3fec.g.ae232cba.d.c20a064a"
"""A real version string from before this batch. Used verbatim for the migration
tests, so "an old string still parses" is asserted against one that actually
shipped rather than one constructed to parse."""


# --------------------------------------------------------------------------
# The named doubles
# --------------------------------------------------------------------------


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


class LegacyHardcodedSelection:
    """The pre-Batch-12 ``detect_faces`` and ``crop_face``. Literals baked in.

    ``scaleFactor=1.1`` and ``minNeighbors=5`` were inline; ``min_px`` and
    ``margin`` were default arguments of 32 and 1.3. Kept verbatim so
    ``test_the_default_policy_reproduces_the_pre_batch_12_behaviour`` can prove
    the refactor moved the version string and nothing else.
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


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _face_image(size: int = 480) -> np.ndarray:
    """The determinism study's synthetic face, verbatim — it detects reliably."""
    img = np.full((size, size, 3), 200, np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (175, 160, 150), -1)
    cv2.circle(img, (200, 210), 16, (60, 60, 60), -1)
    cv2.circle(img, (280, 210), 16, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 300), (18, 30), 0, 0, 360, (150, 130, 120), -1)
    cv2.ellipse(img, (240, 340), (45, 20), 0, 0, 180, (120, 90, 90), 6)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def _job(data: bytes) -> JobSpec:
    return JobSpec(
        job_id=UUID(int=1),
        media_kind=MediaKind.IMAGE,
        storage_key="uploads/x",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="image/png",
        model_version="v",
        idempotency_key="k",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _analyze(policy):
    """One real analysis through the real pipeline under one policy."""
    data = _encode(_face_image())
    pipeline = ImagePipeline(face_selection=policy)
    bundle = asyncio.run(pipeline.analyze_image(_job(data), data))
    return pipeline, bundle


# --------------------------------------------------------------------------
# 1. The proof: can a cache entry cross selection policies?
# --------------------------------------------------------------------------


def test_the_fixture_actually_disagrees():
    """Every claim below rests on this. Asserted first, so a fixture that stopped
    reproducing R17 fails here rather than making the other tests vacuously pass."""
    _, wide = _analyze(SHIPPED)
    _, tight = _analyze(TIGHT)
    assert wide.verdict is not tight.verdict, (
        f"the crop margin no longer changes the verdict ({wide.verdict} both "
        "times); this fixture can no longer demonstrate R17"
    )


def test_the_selection_blind_version_serves_one_crop_policy_to_another():
    """R17, proved rather than argued.

    Two bundles, same bytes, same weights, same calibration, same gate, same
    decoder — different crop margins and different verdicts. Under the legacy
    composition they collide, so the second deployment reads the first's answer.
    """
    pipe_wide, wide = _analyze(SHIPPED)
    pipe_tight, tight = _analyze(TIGHT)

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
    assert blind_wide == blind_tight, (
        "the legacy double no longer collides; it can no longer demonstrate R17"
    )

    store: dict[tuple[str, str], object] = {(wide.media.sha256, blind_wide): wide}
    hit = store.get((wide.media.sha256, blind_tight))
    assert hit is wide, "the legacy key stopped colliding"
    assert hit.verdict is not tight.verdict, (
        f"a deployment cropping at 1.2 was served verdict {hit.verdict}, computed "
        f"at margin 1.3, when its own crop gives {tight.verdict}"
    )


def test_the_selection_aware_version_does_not():
    pipe_wide, _ = _analyze(SHIPPED)
    pipe_tight, _ = _analyze(TIGHT)
    assert pipe_wide.model_identity != pipe_tight.model_identity
    assert pipe_wide.model_version != pipe_tight.model_version


def test_the_real_sink_isolates_two_selection_policies():
    """The same proof against ``InMemoryResultSink``, not a dict.

    The sink keys ``_by_content`` on ``bundle.model.version``, so this exercises
    the production key derivation rather than a restatement of it.
    """
    pipe_wide, wide = _analyze(SHIPPED)
    pipe_tight, _ = _analyze(TIGHT)

    async def scenario():
        sink = InMemoryResultSink()
        await sink.put_if_absent(str(wide.job_id), wide)
        same = await sink.find_by_content(wide.media.sha256, pipe_wide.model_version)
        cross = await sink.find_by_content(wide.media.sha256, pipe_tight.model_version)
        legacy = await sink.find_by_content(wide.media.sha256, PRE_BATCH_12)
        return same, cross, legacy

    same, cross, legacy = asyncio.run(scenario())
    assert same is not None, "a same-policy lookup must still hit"
    assert cross is None, "a bundle cropped at margin 1.3 was served to a 1.2 lookup"
    assert legacy is None, (
        "a lookup by a pre-Batch-12 version hit a face-pinned entry; the two "
        "namespaces must not overlap"
    )


# --------------------------------------------------------------------------
# 2. Every variable reaches the identity
# --------------------------------------------------------------------------


def test_every_policy_field_moves_the_identity():
    """No field is decoration. Each one is varied and the fingerprint must move.

    Iterates the dataclass rather than a hand-written list, so a field added
    later without being put into ``identity`` fails here instead of silently
    joining the set of things nothing records.
    """
    alternatives = {
        "detector": "scrfd-10g",
        "detector_asset": "0" * 16,
        "scale_factor": 1.2,
        "min_neighbors": 8,
        "detect_min_px": 64,
        "ordering": "short-edge-desc",
        "crop_margin": 1.4,
        "crop_shape": "tight",
        # S8 Batch 22 (R25): two label closures over resize / crop_face geometry.
        "interpolation_rule": "linear-always",
        "crop_clip_rule": "pad-replicate",
        # SCRFD Integration Program, Phase 2 (docs/SCRFD_PHASE2_SPEC_RESOLUTION.md
        # Decision 6): None under Haar, so any real value must move the identity.
        "confidence_threshold": 0.5,
        "nms_iou_threshold": 0.4,
    }
    names = {f.name for f in fields(pre.FaceSelectionPolicy)}
    assert names == set(alternatives), (
        f"FaceSelectionPolicy fields {sorted(names)} do not match the variations "
        f"{sorted(alternatives)}; a new knob was added without a test that it "
        "reaches the identity"
    )
    for field, value in alternatives.items():
        other = replace(SHIPPED, **{field: value})
        assert other.identity != SHIPPED.identity, (
            f"changing {field} from {getattr(SHIPPED, field)!r} to {value!r} left "
            "the face-selection identity unchanged"
        )


def test_the_policy_has_no_defaults():
    """The structural half of the fix: hidden defaults were the defect.

    A policy cannot be constructed without stating every knob, so there is
    exactly one place the shipped numbers live and it is
    ``default_face_selection``.
    """
    import dataclasses

    for f in fields(pre.FaceSelectionPolicy):
        assert f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING, (
            f"FaceSelectionPolicy.{f.name} has a default; the whole point of R17 "
            "is that a knob nobody has to state is a knob nobody fingerprints"
        )


def test_the_ordering_rule_name_matches_the_ordering_rule():
    """The tripwire ``ORDERING_RULE``'s docstring promises.

    The identity carries a *name* rather than a hash of the function, because
    bytecode moves on comment edits. That is only sound if the name is held to
    the behaviour, which is what this does: it re-derives the order the name
    describes and compares it against ``face_order_key`` on boxes chosen to
    separate every component of the key.
    """
    assert pre.ORDERING_RULE == "short-edge-desc,area-desc,y-asc,x-asc,width-asc,height-asc"

    boxes = [
        FaceBox(10, 10, 200, 200),   # area 40000
        FaceBox(400, 10, 200, 200),  # same area, larger x
        FaceBox(10, 400, 200, 200),  # same area, larger y
        FaceBox(0, 0, 100, 400),     # same area, different aspect
        FaceBox(0, 0, 400, 100),     # same area and origin, aspect swapped
        FaceBox(5, 5, 300, 300),     # larger area
        FaceBox(9, 9, 210, 500),     # largest area of all, but a smaller short edge
    ]
    from_name = sorted(
        boxes, key=lambda b: (-min(b.width, b.height), -b.area, b.y, b.x, b.width, b.height)
    )
    from_code = sorted(boxes, key=face_order_key)
    assert from_name == from_code, (
        f"ORDERING_RULE names {pre.ORDERING_RULE!r} but face_order_key does "
        "something else; the identity is naming a rule the code no longer follows"
    )


def test_the_cascade_asset_is_hashed_not_assumed():
    """A re-trained cascade shipped under the same filename must move the identity.

    OpenCV owns this file. ``cv2.__version__`` would move on such a reship, but
    no part of the *model* identity reads it — that lives in the decode identity,
    which a model comparison deliberately strips.
    """
    digest = pre.cascade_digest()
    assert digest and digest != "unreadable", "the cascade asset could not be read"
    assert len(digest) == 16
    assert pre.cascade_digest() is digest or pre.cascade_digest() == digest
    assert replace(SHIPPED, detector_asset="0" * 16).identity != SHIPPED.identity


def test_both_face_floors_are_now_fingerprinted():
    """Two floors decide an abstention jointly; only one was ever covered.

    ``detect_min_px`` decides which boxes exist, ``MediaCaps.min_face_px``
    decides which of them is large enough to score. They live in different
    fingerprints on purpose — one is code, the other is deployment configuration
    — but neither may be missing.
    """
    assert replace(SHIPPED, detect_min_px=64).identity != SHIPPED.identity
    stricter = pre.MediaCaps(min_face_px=128)
    assert stricter.gate_fingerprint != pre.DEFAULT_CAPS.gate_fingerprint
    assert SHIPPED.detect_min_px != pre.DEFAULT_CAPS.min_face_px, (
        "the two floors have converged; the distinction this test protects is gone"
    )


def test_the_identity_is_stable_across_calls_and_equal_policies():
    """The other half of the requirement: no spurious invalidation.

    A fingerprint that changed on every process start would dump the cache on
    every deploy, which is as broken as one that never changes.
    """
    assert len({SHIPPED.identity for _ in range(5)}) == 1
    twin = replace(SHIPPED)
    assert twin.identity == SHIPPED.identity


# --------------------------------------------------------------------------
# 3. Behaviour is preserved; only the name moved
# --------------------------------------------------------------------------


def test_the_default_policy_reproduces_the_pre_batch_12_behaviour():
    """The duty the second double carries.

    Adding a fingerprint is worthless if the refactor that added it also moved
    one of the numbers. Boxes and crop bytes are compared against the literals
    verbatim.
    """
    img = _face_image()
    legacy_boxes = LegacyHardcodedSelection.detect(img)
    shipped_boxes = pre.detect_faces(img, policy=SHIPPED)
    assert legacy_boxes == shipped_boxes, (
        f"detection moved: {legacy_boxes} != {shipped_boxes}"
    )
    assert legacy_boxes, "the fixture stopped detecting; this test proves nothing"

    legacy_crop = LegacyHardcodedSelection.crop(img, legacy_boxes[0])
    shipped_crop = pre.crop_face(img, shipped_boxes[0], policy=SHIPPED)
    assert _sha(legacy_crop) == _sha(shipped_crop), (
        "the crop moved; R17's fix was supposed to change the version string and "
        "nothing else"
    )
    assert legacy_crop.shape == shipped_crop.shape


def test_omitting_the_policy_uses_the_shipped_one():
    """Callers that pass no policy must behave exactly as before.

    ``detect_faces(img)`` and ``crop_face(img, box)`` are what the studies and
    several suites call. They must not silently acquire different parameters.
    """
    img = _face_image()
    assert pre.detect_faces(img) == pre.detect_faces(img, policy=SHIPPED)
    box = pre.detect_faces(img)[0]
    assert _sha(pre.crop_face(img, box)) == _sha(pre.crop_face(img, box, policy=SHIPPED))


def test_a_crop_margin_change_moves_the_published_probability():
    """The measurement the whole batch rests on, as an assertion."""
    _, wide = _analyze(SHIPPED)
    _, tight = _analyze(TIGHT)
    assert (wide.p_fake, wide.verdict) != (tight.p_fake, tight.verdict)


# --------------------------------------------------------------------------
# 4. Composition and parsing
# --------------------------------------------------------------------------


def test_a_pre_batch_12_version_parses_with_no_face_segment():
    """Migration: every version written before Batch 12 still reads.

    ``face_selection=None`` is not an error. It means "this deployment makes no
    claim about which face it chose", which is exactly what an old string means,
    and is distinguishable from a claim.
    """
    parts = parse_model_version(PRE_BATCH_12)
    assert parts.weights == "stub-0-w7f45fada"
    assert parts.calibration == "739c3fec"
    assert parts.gate == "ae232cba"
    assert parts.face_selection is None
    assert not is_face_selection_pinned(PRE_BATCH_12)
    assert face_selection_identity_of(PRE_BATCH_12) is None


def test_a_batch_12_version_round_trips():
    version = compose_model_version(
        weights="stub-0-w7f45fada",
        calibration="739c3fec",
        gate="ae232cba",
        face_selection="d662e22d",
    )
    parts = parse_model_version(version)
    assert parts.face_selection == "d662e22d"
    assert parts.compose() == version
    assert is_face_selection_pinned(version)
    assert face_selection_identity_of(version) == "d662e22d"


def test_composing_without_a_face_selection_returns_the_version_unchanged():
    """No sentinel, for the reason ``compose_runtime_version`` has none: a
    placeholder would make "makes no claim" and "pinned to nothing" the same
    string, and the scheme rests on telling them apart."""
    bare = compose_model_version(weights="w", calibration="c", gate="g")
    assert FACE_SELECTION_MARKER not in bare
    assert bare == compose_model_version(
        weights="w", calibration="c", gate="g", face_selection=None
    )


def test_the_decode_segment_stays_last():
    """RUN_IDENTITY §5.1 rests on the model identity being a *prefix* of the
    runtime identity. Inserting a segment must not break that."""
    model = compose_model_version(
        weights="w", calibration="c", gate="g", face_selection="f"
    )
    runtime = compose_runtime_version(model=model, decode="d")
    assert runtime.startswith(model + DECODE_MARKER)
    assert model_identity_of(runtime) == model
    assert face_selection_identity_of(runtime) == "f"


def test_a_weights_version_may_not_contain_the_face_marker():
    try:
        compose_model_version(
            weights=f"stub{FACE_SELECTION_MARKER}0", calibration="abcd1234"
        )
    except ValueError as exc:
        assert FACE_SELECTION_MARKER in str(exc)
        return
    raise AssertionError("a weights version carrying the face marker was accepted")


def test_a_face_fingerprint_containing_a_marker_is_refused():
    for bad in (f"a{DECODE_MARKER}b", f"a{GATE_MARKER}b", ""):
        try:
            compose_model_version(
                weights="w", calibration="c", gate="g", face_selection=bad
            )
        except ValueError:
            continue
        raise AssertionError(f"face fingerprint {bad!r} was accepted")


def test_a_trailing_face_marker_with_no_fingerprint_is_refused():
    try:
        parse_model_version("w" + CALIBRATION_MARKER + "c" + FACE_SELECTION_MARKER)
    except ValueError as exc:
        assert "no fingerprint" in str(exc)
        return
    raise AssertionError("a dangling face marker parsed")


def test_face_selection_survives_a_version_with_no_gate():
    """The segments are independent. A spec with no gate but a policy is
    representable, and must not parse its policy back as a gate."""
    version = compose_model_version(weights="w", calibration="c", face_selection="f")
    parts = parse_model_version(version)
    assert parts.gate is None
    assert parts.face_selection == "f"


# --------------------------------------------------------------------------
# 5. The segment reaches the consumers that matter
# --------------------------------------------------------------------------


def test_the_pipelines_publish_a_face_pinned_version():
    # S8 Batch 23 (R29): the ``.f.`` segment is now the composite selection
    # identity (face + video), so ``face_selection_identity_of`` of a live
    # version returns ``selection_identity(...)``, not the face policy alone.
    expected = pre.selection_identity(SHIPPED, pre.default_video_selection())
    for pipeline in (ImagePipeline(model_name="stub-image"), VideoPipeline(model_name="stub-video")):
        version = pipeline.model_version
        assert is_face_selection_pinned(version), f"{version} carries no face segment"
        assert face_selection_identity_of(version) == expected
        assert model_identity_of(version) == pipeline.model_identity


def test_the_bundle_carries_the_face_segment():
    """The write key and the read key must be the same string."""
    pipeline, bundle = _analyze(SHIPPED)
    assert bundle.model.version == pipeline.model_version
    assert is_face_selection_pinned(bundle.model.version)


def test_the_configured_literal_matches_the_engines():
    """``verify_version_alignment`` on the value that actually ships."""
    from backend.config import STUB_ENGINE_VERSION
    from workers.engine_registry import verify_version_alignment

    assert is_face_selection_pinned(STUB_ENGINE_VERSION)
    verify_version_alignment(STUB_ENGINE_VERSION)


def test_alignment_fails_when_the_configured_policy_is_stale():
    """The failure a deployment running a different policy must produce.

    A worker whose selection policy differs from the configured one fails at
    startup rather than sharing a cache namespace with it — the same protection
    Batch 9 gave the decoder.
    """
    from backend.config import STUB_ENGINE_VERSION
    from workers.engine_registry import verify_version_alignment

    parts = parse_model_version(STUB_ENGINE_VERSION)
    stale = compose_runtime_version(
        model=compose_model_version(
            weights=parts.weights,
            calibration=parts.calibration,
            gate=parts.gate,
            face_selection="00000000",
        ),
        decode="c20a064a",
    )
    try:
        verify_version_alignment(stale)
    except RuntimeError as exc:
        assert "does not match the engines" in str(exc)
        return
    raise AssertionError("a stale face-selection identity passed the alignment check")


def test_the_benchmark_identity_inherits_the_segment():
    """``BenchmarkRun.effective_model_version`` records the *model* identity, and
    a selection-policy change is a model change: the same weights cropping
    differently score differently, so a comparison across the two is not a
    like-for-like model comparison and must not read as one."""
    wide = ImagePipeline(face_selection=SHIPPED).model_identity
    tight = ImagePipeline(face_selection=TIGHT).model_identity
    assert wide != tight
    assert DECODE_MARKER not in wide, (
        "the model identity has grown a decode segment; a benchmark comparison "
        "would answer 'the model changed' for a worker rebuild"
    )
    assert face_selection_identity_of(wide) != face_selection_identity_of(tight)


# --------------------------------------------------------------------------
# 6. The limits, asserted so they cannot become assumed guarantees
# --------------------------------------------------------------------------


def test_the_video_pipeline_carries_the_segment_without_consuming_it():
    """Deliberate, and both halves matter.

    *Carries* — because ``Settings.model_version`` is one string keying the cache
    for both media kinds, and ``verify_version_alignment`` requires every engine
    to report it. *Does not consume* — because the video path detects no faces at
    all (R18), so no selection policy can move a video verdict today.

    The cost is a video cache namespace that changes when a policy it does not
    use changes: a false miss, which is the safe direction for a cache key.
    """
    src = (Path(__file__).resolve().parent.parent / "inference/video_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "detect_faces" not in src and "crop_face" not in src, (
        "the video pipeline has grown a face stage; this test's premise is gone "
        "and R18 may be closable — replace it with one that gates on the behaviour"
    )
    image = ImagePipeline(model_name="stub-image").model_version
    video = VideoPipeline(model_name="stub-video").model_version
    assert image == video, (
        "the two engines report different versions; the API holds one configured "
        "string for both media kinds and alignment would fail at startup"
    )
    assert is_face_selection_pinned(video)


def test_a_selection_change_does_not_move_the_calibration_fingerprint():
    """By design. A crop margin is not a calibration, and folding it into that
    fingerprint would make "the temperature changed" and "the detector changed"
    the same event — the conflation Batch 9 rejected for the decoder."""
    wide = ImagePipeline(face_selection=SHIPPED)._spec
    tight = ImagePipeline(face_selection=TIGHT)._spec
    assert wide.calibration_fingerprint == tight.calibration_fingerprint
    assert wide.weights_identity == tight.weights_identity
    assert wide.gate_fingerprint == tight.gate_fingerprint
    assert wide.effective_version != tight.effective_version


def test_a_selection_change_does_not_move_the_decode_identity():
    """Also by design. Selection is code this deployment ships; the decoder is
    environment it inherits. A version string that conflated them could not
    answer "which of the two moved"."""
    from contracts.version import decode_identity_of

    wide = ImagePipeline(face_selection=SHIPPED).model_version
    tight = ImagePipeline(face_selection=TIGHT).model_version
    assert decode_identity_of(wide) == decode_identity_of(tight)
    assert model_identity_of(wide) != model_identity_of(tight)


def test_a_spec_built_outside_a_pipeline_is_unpinned_not_wrong():
    """The default path for a caller constructing a spec directly.

    It publishes a version making no claim about face selection, rather than
    quietly asserting a policy it never applied.
    """
    spec = ModelSpec(name="t", version="stub-0", backend="stub")
    assert spec.face_selection_fingerprint is None
    assert not is_face_selection_pinned(spec.effective_version)
    bound = spec.with_face_selection(SHIPPED.identity)
    assert is_face_selection_pinned(bound.effective_version)
    assert bound.with_face_selection(SHIPPED.identity) is bound


def test_the_contract_version_did_not_move():
    """No contract field changed shape or meaning. The version string is a ``str``
    and got eleven characters longer."""
    from contracts.common import ModelInfo

    assert CONTRACT_VERSION == "1.0.0"
    assert set(ModelInfo.model_fields) == {
        "version",
        "branch_versions",
        "calibration_temp",
        "git_sha",
    }
    assert ModelInfo.model_fields["version"].annotation is str


def test_identity_does_not_make_two_policies_agree():
    """The honest limit, and the same shape as Batch 9's.

    Fingerprinting the policy makes two policies distinguishable. It does not
    make either correct, it does not pick the right face (R16), and it does not
    reduce the spread the study measured.
    """
    img = _face_image()
    box = pre.detect_faces(img, policy=SHIPPED)[0]
    wide = _sha(pre.crop_face(img, box, policy=SHIPPED))
    tight = _sha(pre.crop_face(img, box, policy=TIGHT))
    assert wide != tight, "the two policies still produce different pixels"
    assert SHIPPED.differences(TIGHT) == ["crop_margin: 1.3 != 1.2"], (
        "differences() must name what changed, for the operator reading two runs"
    )


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
