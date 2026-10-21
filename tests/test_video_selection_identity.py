"""R29 + N4 — video frame-selection and pooling are now inside the model identity.

**This suite replaces ``tests/test_video_identity_gap.py``**, which asserted the
*open* state in the B19/B20 tripwire style: two ``VideoPipeline`` instances that
differ on ``n_frames`` / ``min_usable_frames`` / the pooling depth produced a
byte-identical ``model_version`` — the whole result-cache key — while sampling a
different frame set and able to publish a different verdict. Batch 23 closed that
gap the way Batch 12 (R17) and Batch 22 (R25) closed the image-path analogues:

    n_frames          -> VideoSelectionPolicy  (functional — sample_frame_indices)
    min_usable_frames -> VideoSelectionPolicy  (functional — the abstention gate)
    pool_divisor      -> VideoSelectionPolicy  (functional — topk_indices' k)
    sampling_rule     -> VideoSelectionPolicy  (label — held to sample_frame_indices,
                                                also the VideoAnalysis.sampling wire value)
    pooling_rule      -> VideoSelectionPolicy  (label — also VideoAnalysis.pooling)
    batch_size        -> excluded, proven output-neutral

``selection_identity(face, video)`` folds the video policy into the ``.f.``
segment, which **both** pipelines now carry, so the single ``STUB_ENGINE_VERSION``
literal and ``contracts/version.py``'s parser are untouched (``.f.`` is still one
opaque 8-hex fingerprint).

N4: the stale ``VideoAnalysis.sampling`` default ``"uniform+adaptive"`` is
corrected to ``"uniform"`` — nothing adaptive runs — in the contract, the
fixtures and the schema.

    python tests/test_video_selection_identity.py
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import sys
import tempfile
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.config import STUB_ENGINE_VERSION  # noqa: E402
from contracts.enums import CONTRACT_VERSION, MediaKind, Verdict  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from contracts.version import (  # noqa: E402
    FACE_SELECTION_MARKER,
    compose_runtime_version,
    face_selection_identity_of,
    is_face_selection_pinned,
    is_selection_pinned,
    model_identity_of,
    parse_model_version,
    selection_identity_of,
)
from evaluation.benchmark import RUN_SCHEMA_VERSION  # noqa: E402
from evaluation.calibration_protocol import CALIBRATION_WITNESS_VERSION  # noqa: E402
from inference import ImagePipeline, VideoPipeline  # noqa: E402
from inference import postprocess as post  # noqa: E402
from inference import preprocess as pre  # noqa: E402
from inference.decode_env import decode_identity  # noqa: E402
from inference.models import ModelSpec  # noqa: E402
from inference.postprocess import pool_at, topk_indices  # noqa: E402
from inference.preprocess import sample_frame_indices  # noqa: E402

FINDING = "R29"

# The value Batch 23 regenerated: the composite ``.f.`` segment, and the model
# identity / engine literal it produces. The *face* policy identity (251d22e9) is
# unchanged; only the segment that now combines it with the video policy moved.
B23_SELECTION_IDENTITY = "e629642c"
B23_MODEL_IDENTITY = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c"
B23_STUB_ENGINE_VERSION = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.a97ccb68"

# S8 Batch 26 (R30): the ``.d.`` segment moved (image codec libraries joined the
# decode identity); ``.f.`` / the model identity are unchanged since B23. Kept as
# a separate constant, not a mutation of ``B23_STUB_ENGINE_VERSION`` above, so
# tests that specifically exercise "the B23 string shape"
# (``test_parse_model_version_is_total_across_the_b23_string``) keep testing that
# shape rather than silently retargeting to whatever the current literal is.
B26_STUB_ENGINE_VERSION = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.c9d4025c"

# SCRFD Integration Program, Phase 2 (docs/SCRFD_PHASE2_SPEC_RESOLUTION.md
# Decision 7): FaceSelectionPolicy gained confidence_threshold/
# nms_iou_threshold (both None under Haar). Unlike B23, the *face* policy
# identity itself moves this time (251d22e9 -> b0591be3) — a structural move,
# a new key in the fingerprinted dict, not a value relabelling — and the
# composite .f. segment moves with it. These are the *current* values, unlike
# the B22/B23/B26 constants above, which stay frozen for cross-era parsing.
PHASE2_FACE_POLICY_IDENTITY = "b0591be3"
PHASE2_SELECTION_IDENTITY = "f8244ad0"
PHASE2_MODEL_IDENTITY = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.f8244ad0"
PHASE2_STUB_ENGINE_VERSION = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.f8244ad0.d.c9d4025c"

# A fixed run of the shipped policy through the synthetic clip below. If pooling
# or sampling logic moves, this moves with it — the output-identical guarantee,
# pinned.
SHIPPED_P_FAKE = 0.9236678556413206

_FRAMES = 60
_SIZE = 128


def _clip(frames: int = _FRAMES, size: int = _SIZE, fps: int = 20, seed: int = 1) -> bytes:
    """Middle third is high-frequency noise (reads as manipulated to the stub),
    outer thirds flat. Sampling more or fewer frames re-weights that segment."""
    path = Path(tempfile.mkdtemp()) / "r29.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    rng = np.random.default_rng(seed)
    for i in range(frames):
        if frames // 3 <= i < 2 * frames // 3:
            frame = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        else:
            frame = np.full((size, size, 3), 110, np.uint8)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


CLIP = _clip()
SHIPPED = pre.default_video_selection()


def _spec(data: bytes = CLIP) -> JobSpec:
    return JobSpec(
        job_id=uuid4(),
        media_kind=MediaKind.VIDEO,
        storage_key="u",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        declared_mime="video/mp4",
        model_version="stub-0",
        idempotency_key="k",
        submitted_at=datetime.now(UTC),
    )


def _analyse(video_selection: pre.VideoSelectionPolicy):
    pipe = VideoPipeline(model_name="stub-video", video_selection=video_selection)
    bundle = asyncio.run(pipe.analyze_video(_spec(), CLIP))
    return pipe, bundle


# --------------------------------------------------------------------------
# 1. The three functional knobs move the identity (the gap, closed)
# --------------------------------------------------------------------------


def test_r29_n_frames_moves_the_model_version():
    """Was ``test_r29_n_frames_changes_p_fake_under_one_model_version`` — it
    asserted the two versions were **equal**. Now they must differ."""
    dense, b_dense = _analyse(replace(SHIPPED, n_frames=48))
    sparse, b_sparse = _analyse(replace(SHIPPED, n_frames=9))

    assert dense.model_version != sparse.model_version, "n_frames does not move the cache key"
    assert dense.model_identity != sparse.model_identity
    assert dense._spec.effective_version != sparse._spec.effective_version
    # and the pipelines still looked at different frames / published different numbers
    assert b_dense.p_fake is not None and b_sparse.p_fake is not None
    assert b_dense.p_fake != b_sparse.p_fake


def test_r29_the_sampled_frame_set_still_depends_on_n_frames():
    """The mechanism — ``sample_frame_indices`` is a pure function of a count —
    is unchanged; that count is now in the identity."""
    assert sample_frame_indices(1000, 48) != sample_frame_indices(1000, 9)


def test_r29_min_usable_frames_moves_the_identity():
    """Was ``..._flips_the_verdict_under_one_model_version``. A verdict→abstention
    gate, the ``min_face_px`` (R19) shape — now fingerprinted.

    A 12-frame clip decodes 12 frames under ``n_frames=16``; a floor of 8 decides,
    a floor of 13 abstains. (Batch 23 also centralised ``1 <= min_usable_frames
    <= n_frames`` into ``VideoSelectionPolicy.__post_init__``.)
    """
    short = _clip(frames=12)
    spec = _spec(short)

    def run(floor: int):
        pol = replace(SHIPPED, n_frames=16, min_usable_frames=floor)
        pipe = VideoPipeline(model_name="stub-video", video_selection=pol)
        return pipe, asyncio.run(pipe.analyze_video(spec, short))

    decided, b_decided = run(8)
    abstained, b_abstained = run(13)

    assert decided.model_version != abstained.model_version
    assert b_decided.p_fake is not None
    assert b_abstained.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert b_abstained.p_fake is None


def test_r29_min_usable_frames_is_in_the_video_selection_identity():
    """Was ``..._is_not_a_media_cap`` (asserting it reached nothing)."""
    assert replace(SHIPPED, min_usable_frames=9).identity != SHIPPED.identity
    assert "min_usable_frames" in SHIPPED.to_dict()


def test_r29_pool_divisor_is_bound_by_the_policy():
    """Was ``test_r29_pooling_depth_k_is_an_unbound_constant``. ``topk_indices``'s
    ``k`` now comes from ``VideoSelectionPolicy.pool_divisor``: two divisors give
    two identities and two pooled means."""
    shallow = replace(SHIPPED, pool_divisor=8)
    deep = replace(SHIPPED, pool_divisor=4)
    assert shallow.identity != deep.identity

    a, b_a = _analyse(shallow)
    b, b_b = _analyse(deep)
    assert a.model_version != b.model_version
    assert b_a.p_fake != b_b.p_fake

    scores = np.linspace(-2.0, 3.0, 48)
    assert pool_at(scores, topk_indices(scores, k=max(1, 48 // 8))) != pool_at(
        scores, topk_indices(scores, k=max(1, 48 // 4))
    )


def test_every_policy_field_moves_the_identity():
    """No field is decoration. Iterates the dataclass so a field added later
    without being put into ``identity`` fails here."""
    alternatives = {
        "n_frames": 24,
        "min_usable_frames": 4,
        "pool_divisor": 4,
        "sampling_rule": "uniform-stride",
        "pooling_rule": "attention",
    }
    names = {f.name for f in fields(pre.VideoSelectionPolicy)}
    assert names == set(alternatives), (
        f"VideoSelectionPolicy fields {sorted(names)} do not match the variations "
        f"{sorted(alternatives)}; a new knob was added without a test that it "
        "reaches the identity"
    )
    for field, value in alternatives.items():
        other = replace(SHIPPED, **{field: value})
        assert other.identity != SHIPPED.identity, f"changing {field} left the identity unchanged"
        assert other.differences(SHIPPED)[0].startswith(f"{field}: ")


def test_the_video_selection_policy_has_no_defaults():
    """The structural half of the fix — hidden defaults were the defect."""
    import dataclasses

    for f in fields(pre.VideoSelectionPolicy):
        assert f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING, (
            f"VideoSelectionPolicy.{f.name} has a default"
        )


def test_the_policy_validates_its_invariants():
    with _expect(ValueError):
        pre.VideoSelectionPolicy(0, 0, 1, "uniform", "topk_mean")
    with _expect(ValueError):
        pre.VideoSelectionPolicy(8, 9, 1, "uniform", "topk_mean")  # min_usable > n_frames
    with _expect(ValueError):
        pre.VideoSelectionPolicy(8, 4, 0, "uniform", "topk_mean")  # pool_divisor < 1


# --------------------------------------------------------------------------
# 2. The label closures are held to the behaviour
# --------------------------------------------------------------------------


def test_the_sampling_rule_name_matches_sample_frame_indices():
    """``"uniform"`` names ``sample_frame_indices``'s stride rule. Re-implement it
    independently and compare index lists across several (frame_count, n) pairs."""
    assert SHIPPED.sampling_rule == "uniform"

    def uniform_stride(frame_count: int, n: int) -> list[int]:
        if frame_count <= 0 or n <= 0:
            return []
        if frame_count <= n:
            return list(range(frame_count))
        step = frame_count / n
        return [min(frame_count - 1, int(i * step)) for i in range(n)]

    for fc, n in [(1000, 48), (1000, 9), (50, 48), (48, 48), (7, 8), (1, 4)]:
        assert sample_frame_indices(fc, n) == uniform_stride(fc, n), (
            f"sample_frame_indices disagrees with the named 'uniform' rule at ({fc}, {n})"
        )


def test_the_pooling_rule_and_divisor_match_the_pipeline():
    """The path is ``topk_indices(k=…)`` -> ``pool_at``; the published
    ``VideoAnalysis.pooling`` equals ``policy.pooling_rule``; and the selected set
    has ``max(1, n_scores // pool_divisor)`` frames."""
    assert SHIPPED.pooling_rule == "topk_mean"
    _, bundle = _analyse(SHIPPED)
    assert bundle.video.pooling == SHIPPED.pooling_rule
    assert bundle.video.sampling == SHIPPED.sampling_rule

    n_scores = bundle.video.frames_scored
    scores = np.linspace(-1.0, 2.0, n_scores)
    k = max(1, n_scores // SHIPPED.pool_divisor)
    assert len(topk_indices(scores, k=k)) == k


def test_batch_size_is_output_neutral():
    """The excluded knob. Two batch sizes -> the same verdict, probability, curve
    and version. Only progress-event fractions depend on it."""
    a = asyncio.run(VideoPipeline(batch_size=8).analyze_video(_spec(), CLIP))
    b = asyncio.run(VideoPipeline(batch_size=16).analyze_video(_spec(), CLIP))
    assert a.p_fake == b.p_fake
    assert a.verdict == b.verdict
    assert a.model.version == b.model.version
    assert a.video.model_dump() == b.video.model_dump()


# --------------------------------------------------------------------------
# 3. Output-identical + the migration moved only what it should
# --------------------------------------------------------------------------


def test_the_shipped_video_bundle_did_not_move():
    """The shipped policy through a fixed clip: the version is the regenerated
    literal, ``sampling`` is the N4-corrected value, and ``p_fake`` is the pinned
    number. Every field names current behaviour verbatim."""
    _, bundle = _analyse(SHIPPED)
    assert bundle.model.version == STUB_ENGINE_VERSION == PHASE2_STUB_ENGINE_VERSION
    assert bundle.video.sampling == "uniform"  # N4: was "uniform+adaptive" in fixtures
    assert bundle.video.pooling == "topk_mean"
    assert bundle.p_fake == SHIPPED_P_FAKE, (
        f"the shipped video bundle's p_fake moved to {bundle.p_fake}; B23 must be "
        "output-identical — a pooling/sampling change slipped in"
    )


def test_the_selection_identity_is_the_composite():
    face = pre.default_face_selection()
    combined = pre.selection_identity(face, SHIPPED)
    assert combined == PHASE2_SELECTION_IDENTITY
    assert combined != face.identity, "the .f. segment is still just the face policy"
    assert combined != SHIPPED.identity
    assert parse_model_version(STUB_ENGINE_VERSION).face_selection == combined


def test_batch_23_moved_only_the_selection_identity():
    assert CONTRACT_VERSION == "1.0.0"
    assert RUN_SCHEMA_VERSION == "1.2.0"
    assert CALIBRATION_WITNESS_VERSION == "1.3.0"

    root = Path(__file__).resolve().parent.parent
    snap = (root / "backend/openapi.snapshot.json").read_text(encoding="utf-8")
    assert PHASE2_SELECTION_IDENTITY not in snap and "wacdd9240" not in snap
    schema_files = list((root / "schema_out").glob("*.json"))
    assert len(schema_files) >= 10
    for f in schema_files:
        assert "stub-0" not in f.read_text(encoding="utf-8")

    # weights / calibration / gate / decode did not move; only .f. did.
    parts = parse_model_version(STUB_ENGINE_VERSION)
    assert parts.weights == "stub-0-wacdd9240"
    assert parts.calibration == "739c3fec"
    assert parts.gate == "ae232cba"
    assert parts.face_selection == PHASE2_SELECTION_IDENTITY
    assert STUB_ENGINE_VERSION.endswith(".d.c9d4025c")  # R30, Batch 26 (.d. unmoved by Phase 2)
    assert model_identity_of(STUB_ENGINE_VERSION) == PHASE2_MODEL_IDENTITY
    # the face policy identity moved again in SCRFD Integration Program Phase 2
    # (see test_the_face_policy_identity_is_unchanged's own updated docstring)
    assert pre.default_face_selection().identity == PHASE2_FACE_POLICY_IDENTITY


def test_parse_model_version_is_total_across_the_b23_string():
    """``.f.`` is still one fingerprint — no new parser era."""
    parts = parse_model_version(B23_STUB_ENGINE_VERSION)
    assert parts.compose() == B23_MODEL_IDENTITY
    rt = compose_runtime_version(model=parts.compose(), decode="deadbeef")
    assert model_identity_of(rt) == parts.compose()


# --------------------------------------------------------------------------
# 4. Both pipelines carry it; alignment holds
# --------------------------------------------------------------------------


def test_image_and_video_engines_report_the_same_version():
    image = ImagePipeline(model_name="stub-image").model_version
    video = VideoPipeline(model_name="stub-video").model_version
    assert image == video == STUB_ENGINE_VERSION


def test_the_image_pipeline_carries_the_video_segment_without_consuming_it():
    """The mirror of ``test_the_video_pipeline_carries_the_segment_without_consuming_it``.

    *Carries* — because one configured string keys the cache for both media
    kinds. *Does not consume* — the image path samples no frames and pools
    nothing, so no ``VideoSelectionPolicy`` field can move an image verdict.
    """
    src = (Path(__file__).resolve().parent.parent / "inference/image_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "sample_frame_indices" not in src and "topk_indices" not in src, (
        "the image pipeline has grown frame-selection / pooling; this test's "
        "premise is gone"
    )
    # PA15 — the carry is documented as PERMANENT, distinguishing it from
    # VideoPipeline's temporary carry of FaceSelectionPolicy (which SCRFD makes
    # live). If a future stage makes VideoSelectionPolicy live on the image path
    # this word must be removed deliberately, and this test is where that lands.
    assert "PERMANENT" in src and "NEVER" in src, (
        "the image_pipeline 'carried, not consumed' comment no longer states the "
        "carry is permanent"
    )
    base = ImagePipeline(model_name="stub-image").model_version
    other = ImagePipeline(
        model_name="stub-image", video_selection=replace(SHIPPED, n_frames=9)
    ).model_version
    assert base != other, "a video-policy change did not move the image identity"


def test_the_legacy_scalar_shim_keeps_the_identity_honest():
    """``VideoPipeline(n_frames=…)`` still works (deprecated) and the identity
    reflects the value actually used, reconstructed via ``replace()``."""
    assert VideoPipeline(n_frames=9).model_version != VideoPipeline().model_version
    assert (
        VideoPipeline(n_frames=9).model_version
        == VideoPipeline(video_selection=replace(SHIPPED, n_frames=9)).model_version
    )
    assert VideoPipeline().model_version == STUB_ENGINE_VERSION


def test_alignment_accepts_the_shipped_configuration_and_rejects_a_stale_one():
    from workers.engine_registry import verify_version_alignment

    verify_version_alignment(STUB_ENGINE_VERSION)  # the real one passes

    stale_face_only = compose_runtime_version(
        model=B23_MODEL_IDENTITY.replace(".f.e629642c", ".f.251d22e9"),
        decode=decode_identity(),
    )
    try:
        verify_version_alignment(stale_face_only)
    except RuntimeError as exc:
        assert "does not match the engines" in str(exc)
        return
    raise AssertionError("a face-only .f. segment passed alignment after B23")


def test_r29_is_tracked_in_the_reproducibility_audit():
    audit = (Path(__file__).resolve().parent.parent / "REPRODUCIBILITY_AUDIT.md").read_text(
        encoding="utf-8"
    )
    low = audit.lower()
    assert f"| {FINDING} |" in audit, "R29 has no row in the summary table"
    assert "closed" in low and "batch 23" in low
    assert "videoselectionpolicy" in low or "video selection policy" in low


# --------------------------------------------------------------------------
# 5. R3 additions — PA12–PA15, R33 alias tripwires, R35, MG5
# --------------------------------------------------------------------------


def test_the_face_policy_identity_is_unchanged():
    """PA12 — B23 moved the *combined* ``.f.`` segment, not the face policy; the
    face policy fingerprint stayed ``251d22e9`` through B23 and B26.

    **This is scoped to those batches, not a permanent invariant.** SCRFD
    Integration Program, Phase 2 (``docs/SCRFD_PHASE2_SPEC_RESOLUTION.md``
    Decision 7) is the first batch to move the face policy identity again —
    ``FaceSelectionPolicy`` gained two real fields (``confidence_threshold``/
    ``nms_iou_threshold``), not a relabelling, so PA12's original claim
    ("B23 did not touch the face policy") still holds for B23 specifically;
    it was never a claim that nothing ever would again. Updated here rather
    than left green-but-false.
    """
    assert pre.default_face_selection().identity == PHASE2_FACE_POLICY_IDENTITY
    assert pre.default_face_selection().identity != PHASE2_SELECTION_IDENTITY


def test_image_and_video_engines_agree_when_only_one_policy_is_overridden():
    """PA13 — symmetric defaulting. Overriding just the face policy on both
    pipelines: each defaults the *other* (video) policy to its factory, so both
    still compute ``selection_identity(X, default_video_selection())`` and report
    the same ``model_version`` — a single configured literal keeps satisfying
    both engines."""
    x = replace(pre.default_face_selection(), crop_margin=1.4)
    assert x.identity != pre.default_face_selection().identity
    img = ImagePipeline(model_name="stub-image", face_selection=x).model_version
    vid = VideoPipeline(model_name="stub-video", face_selection=x).model_version
    assert img == vid
    assert img != STUB_ENGINE_VERSION  # and it did move off the shipped literal


def test_build_video_analysis_defaults_match_the_shipped_policy():
    """PA14 (R35) — ``build_video_analysis`` keeps ``sampling`` / ``pooling``
    kwarg defaults for study/test callers. Hold them equal to
    ``default_video_selection()`` so the fallback cannot become a stale second
    source of truth for the ``VideoAnalysis`` wire value."""
    sig = inspect.signature(post.build_video_analysis)
    shipped = pre.default_video_selection()
    assert sig.parameters["sampling"].default == shipped.sampling_rule
    assert sig.parameters["pooling"].default == shipped.pooling_rule


def test_selection_identity_of_returns_the_composite_and_is_the_honest_name():
    """PA10 / R33 — ``selection_identity_of`` is the honest-named reader of the
    ``.f.`` segment; it returns the composite for a B23+ string, equals
    ``face_selection_identity_of`` for every string, and is NOT
    ``default_face_selection().identity``."""
    combined = pre.selection_identity(
        pre.default_face_selection(), pre.default_video_selection()
    )
    assert selection_identity_of(STUB_ENGINE_VERSION) == combined == PHASE2_SELECTION_IDENTITY
    assert selection_identity_of(STUB_ENGINE_VERSION) != pre.default_face_selection().identity
    # the two names are exactly equivalent, always
    assert selection_identity_of(STUB_ENGINE_VERSION) == face_selection_identity_of(
        STUB_ENGINE_VERSION
    )
    assert is_selection_pinned(STUB_ENGINE_VERSION) is is_face_selection_pinned(
        STUB_ENGINE_VERSION
    ) is True
    # marker-free composite: no parser era added
    assert FACE_SELECTION_MARKER not in combined
    assert all(m not in combined for m in ("+c.", ".g.", ".d."))


def test_the_f_marker_and_public_readers_document_both_policies():
    """R33 tripwire — the module docstring, the two public reader functions and
    the ``FACE_SELECTION_MARKER`` source docstring all name the video half. Fails
    if the docs drift back to face-only.

    (``FACE_SELECTION_MARKER`` is a bare ``str`` constant, so its trailing
    docstring is not reachable as ``.__doc__`` — it is checked in source.)
    """
    import contracts.version as ver

    def _names_video(text: str) -> bool:
        # The specific B23 concept, not the bare word "video": the ``.f.`` segment
        # now names *frame selection and pooling* for the video path. A drift that
        # kept an incidental "video" but dropped the B23 content fails here.
        low = " ".join((text or "").split()).lower()
        return "frame selection" in low and "pooling" in low

    assert _names_video(ver.__doc__), "the contracts.version module docstring dropped the video half"
    assert _names_video(selection_identity_of.__doc__)
    assert _names_video(face_selection_identity_of.__doc__)

    src = (
        Path(ver.__file__).read_text(encoding="utf-8")
        if hasattr(ver, "__file__")
        else ""
    )
    marker, _, after = src.partition('FACE_SELECTION_MARKER = ".f."')
    assert after, "FACE_SELECTION_MARKER assignment not found"
    marker_docstring = after.split('"""')[1] if '"""' in after else ""
    assert _names_video(marker_docstring), (
        "the FACE_SELECTION_MARKER source docstring no longer names the video half"
    )

    # the alias carries the explicit "never compare to face identity" banner
    assert "never" in (face_selection_identity_of.__doc__ or "").lower()
    assert "default_face_selection().identity" in (face_selection_identity_of.__doc__ or "")


def test_model_spec_with_selection_is_the_honest_alias():
    """R33 — ``ModelSpec.with_selection`` delegates to ``with_face_selection``;
    both docstrings name the composite."""
    base = ModelSpec(name="stub-image", version="stub-0", backend="stub")
    fp = "abcd1234"
    assert (
        base.with_selection(fp).face_selection_fingerprint
        == base.with_face_selection(fp).face_selection_fingerprint
        == fp
    )
    assert "selection" in (ModelSpec.with_selection.__doc__ or "").lower()
    for doc in (ModelSpec.with_selection.__doc__, ModelSpec.with_face_selection.__doc__):
        assert "batch 23" in (doc or "").lower()


def test_the_batch_size_exclusion_note_is_complete():
    """PA11 (R32) — the ``VideoSelectionPolicy`` docstring names the **full**
    ``batch_size``-neutrality precondition, not only the elementwise-widen
    reason: (a) ``batch_clips`` does not pad, (b) the backend ``score()`` is
    row-independent, (c) ``scores`` reconstruction is order/value preserving —
    and the re-opening trigger (a static-shape ONNX / TensorRT backend, or a
    cross-frame ``pooling_rule="attention"`` backend). Fails if the note drifts
    back to face-value neutrality. (Dropped between B23-R2 and B23-R3; restored
    by the post-audit cleanup batch — `docs/B23_FRESH_INDEPENDENT_AUDIT_2.md`
    D1/D2.)"""
    doc = " ".join((pre.VideoSelectionPolicy.__doc__ or "").split()).lower()
    assert "batch_clips" in doc and "no padding" in doc, "(a) batch_clips no-pad not named"
    assert "row-independent" in doc, "(b) row-independent score() not named"
    assert "order- and value-preserving" in doc, "(c) order/value-preserving reconstruction not named"
    assert "onnx" in doc and "tensorrt" in doc, "static-shape re-opening trigger not named"
    assert "attention" in doc, "cross-frame (attention) re-opening trigger not named"


def test_the_video_analysis_wire_defaults_match_the_shipped_policy():
    """R36 (`docs/B23_FRESH_INDEPENDENT_AUDIT_2.md` E1) — the **authoritative**
    wire defaults. ``contracts.video.VideoAnalysis`` cannot import
    ``inference`` (``contracts`` is a leaf package — a ``contracts -> inference``
    edge would cycle with ``inference -> contracts``), so its ``Field(default=)``
    values are hand-copied literals. Hold every copy — the model-field default,
    and the generated fixture files — equal to ``default_video_selection()`` so a
    labels change cannot leave the published schema / fixtures stale (the N4
    defect class, un-guarded for ``pooling`` before this test).

    PA14 guards only ``build_video_analysis``'s kwarg default; this guards the
    default that an omitted field actually serialises to and that the
    OpenAPI / JSON-schema publishes."""
    from contracts.video import VideoAnalysis

    shipped = pre.default_video_selection()
    assert VideoAnalysis.model_fields["sampling"].default == shipped.sampling_rule
    assert VideoAnalysis.model_fields["pooling"].default == shipped.pooling_rule

    fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures_out"
    for name in ("video_authentic", "video_manipulated", "video_insufficient_evidence"):
        data = json.loads((fixtures_dir / f"{name}.json").read_text(encoding="utf-8"))
        assert data["video"]["sampling"] == shipped.sampling_rule, name
        assert data["video"]["pooling"] == shipped.pooling_rule, name


def test_n4_default_is_uniform_in_the_generated_schema():
    """MG5 — the regenerated JSON schema carries the policy's labels as the
    ``VideoAnalysis`` defaults, scoped to the actual
    ``$defs/VideoAnalysis/properties`` node (not a whole-file substring), so a
    future ``python -m contracts.schema`` cannot silently revert N4 or drift
    ``pooling``."""
    raw = (
        Path(__file__).resolve().parent.parent / "schema_out" / "EvidenceBundle.schema.json"
    ).read_text(encoding="utf-8")
    schema = json.loads(raw)
    props = schema["$defs"]["VideoAnalysis"]["properties"]
    shipped = pre.default_video_selection()
    assert props["sampling"]["default"] == shipped.sampling_rule == "uniform"
    assert props["pooling"]["default"] == shipped.pooling_rule == "topk_mean"
    assert "uniform+adaptive" not in raw


def test_settings_gained_a_wired_skip_version_alignment_field_in_or1_b():
    """R3 §1.8 predicted this reversal by name: through B23 the worker gate was
    env-only (``$DFP_SKIP_VERSION_ALIGNMENT``); a ``Settings.skip_version_
    alignment`` the worker could not read would have been dead config. **OR1-b
    (Batch 25)** wires it: `backend/worker_main.py`, the composition root, may
    import both `backend` and `workers`, and passes the field's value into
    `verify_engine_alignment_at_startup(..., skip_version_alignment=...)`
    directly — checked with `or` alongside the environment variable, which a
    bare `workers`-only deployment still uses on its own."""
    from backend.config import Settings

    assert "skip_version_alignment" in Settings.model_fields
    assert Settings.model_fields["skip_version_alignment"].default is False


# --------------------------------------------------------------------------
# helpers + standalone runner
# --------------------------------------------------------------------------


class _expect:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        assert et is not None and issubclass(et, self.exc), f"expected {self.exc.__name__}"
        return True


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
