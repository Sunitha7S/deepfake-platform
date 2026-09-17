"""Video branch provenance: every published branch traces to a measurement.

The video pipeline used to fuse each frame, pool the fused curve into one
number, and then reconstruct three branch results from that single number:

    BranchId.A: pooled
    BranchId.B: pooled * 0.85 + 0.05      <- invented
    BranchId.P: 0.0                       <- invented

The backend had already produced a real branch-B logit for every frame. It was
computed, discarded, and replaced with an affine function of branch A. Two
things followed, both visible to a user:

  * ``disagreement`` became a deterministic function of branch A alone, so the
    "two branches agree" signal on the Results page carried no information;
  * ``interval`` is derived from ``disagreement``, so the published uncertainty
    band was a restatement of the same invented correlation.

This suite pins the correction: branches are pooled independently from their own
per-frame scores, over one shared frame selection.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts.enums import BranchId, BranchStatus, MediaKind  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from inference import VideoPipeline  # noqa: E402
from inference.models import STUB_VIDEO, StubBackend  # noqa: E402
from inference.postprocess import pool_at, topk_indices  # noqa: E402
from inference.preprocess import DEFAULT_CAPS, batch_clips, extract_frames, resize  # noqa: E402

LEGACY_B_SLOPE = 0.85
LEGACY_B_INTERCEPT = 0.05


def _clip(frames: int = 48, size: int = 160, fps: int = 25, seed: int = 5) -> bytes:
    path = Path(tempfile.mkdtemp()) / "branches.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    rng = np.random.default_rng(seed)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _spec(data: bytes) -> JobSpec:
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


CLIP = _clip()


def _analyse(data: bytes = CLIP):
    return asyncio.run(VideoPipeline().analyze_video(_spec(data), data))


# --------------------------------------------------------------------------
# Branch B is measured, not invented
# --------------------------------------------------------------------------


def test_branch_b_is_not_an_affine_function_of_branch_a():
    """The fabrication, pinned. This is the exact formula that was shipped."""
    bundle = _analyse()
    a, b = bundle.branch(BranchId.A), bundle.branch(BranchId.B)
    assert a.status is BranchStatus.SCORED and b.status is BranchStatus.SCORED
    fabricated = a.logit * LEGACY_B_SLOPE + LEGACY_B_INTERCEPT
    assert abs(b.logit - fabricated) > 1e-9, (
        f"branch B ({b.logit}) still matches the invented formula "
        f"A*{LEGACY_B_SLOPE}+{LEGACY_B_INTERCEPT} = {fabricated}"
    )


def test_branch_b_matches_the_backend_scores_it_pools():
    """Trace the published branch B back to the numbers the backend produced.

    Recomputing the pool independently here means the assertion is about the
    *provenance* of the value, not merely about it being different from branch
    A — a fabrication with a different constant would pass the previous test and
    fail this one.
    """
    meta, frames, _, _ = extract_frames(
        CLIP, mime="video/mp4", n_frames=48, caps=DEFAULT_CAPS
    )
    prepared = np.stack([resize(frames[i], STUB_VIDEO.input_size) for i in range(frames.shape[0])])
    prepared = prepared.astype(np.float32)

    backend = StubBackend(STUB_VIDEO)
    per_branch: dict[BranchId, list[float]] = {}
    for batch in batch_clips(prepared, 8):
        for branch, values in backend.score(batch).items():
            per_branch.setdefault(branch, []).extend(float(v) for v in values)

    bundle = _analyse()
    curve = [f.p for f in bundle.video.frame_scores]
    assert len(curve) == len(per_branch[BranchId.A]), "frame count drifted"

    # The pipeline ranks on the fused curve; reproduce that selection.
    from inference.calibration import inverse_sigmoid

    selected = topk_indices(np.asarray([inverse_sigmoid(p) for p in curve]))
    for branch in (BranchId.A, BranchId.B):
        expected = pool_at(per_branch[branch], selected)
        published = bundle.branch(branch).logit
        assert abs(published - expected) < 1e-6, (
            f"branch {branch.value} published {published} but pooling its own "
            f"per-frame scores over the selected frames gives {expected}"
        )


def test_every_branch_pools_over_the_same_frames():
    """Independent selection per branch would make disagreement meaningless.

    If branch A pooled its own top frames and branch B pooled a different set,
    the two would be describing different moments of the clip and their spread
    would measure frame selection rather than the media.
    """
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.3, 0.7])
    selected = topk_indices(scores, k=3)
    assert selected == sorted(selected), "selection must be index-ordered"
    assert len(selected) == 3
    # Same selection reused for a completely different value series.
    other = np.array([9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0])
    assert pool_at(other, selected) == float(np.mean(other[selected]))


def test_topk_selection_is_deterministic_under_ties():
    """A clip of identical frames must pool the same way on every run."""
    flat = np.full(16, 0.5)
    assert topk_indices(flat, k=4) == [0, 1, 2, 3]
    assert topk_indices(flat, k=4) == topk_indices(flat, k=4)


# --------------------------------------------------------------------------
# Consequences for what a user sees
# --------------------------------------------------------------------------


def test_video_disagreement_is_not_a_restatement_of_branch_a():
    """Disagreement must vary independently of the headline score.

    Under the fabrication the spread between A and B was a fixed affine gap, so
    two clips with the same branch-A score always reported the same
    disagreement — and therefore the same interval width — no matter what the
    footage looked like.
    """
    bundles = [_analyse(_clip(seed=s)) for s in (1, 2, 3, 4)]
    scored = [b for b in bundles if b.p_fake is not None]
    assert len(scored) >= 2, "need at least two decided clips to compare"

    pairs = {
        (round(b.branch(BranchId.A).logit, 9), round(b.disagreement, 9)) for b in scored
    }
    ratios = {round(b.disagreement / max(1e-9, abs(b.branch(BranchId.A).logit)), 6) for b in scored}
    assert len(pairs) >= 2
    assert len(ratios) > 1, (
        "disagreement is a fixed function of branch A across every clip, which "
        "is the signature of a fabricated second branch"
    )


def test_video_branches_are_all_present_and_scored_from_measurements():
    bundle = _analyse()
    assert {b.id for b in bundle.branches} == {BranchId.A, BranchId.B, BranchId.P}
    for branch in (BranchId.A, BranchId.B):
        row = bundle.branch(branch)
        assert row.status is BranchStatus.SCORED
        assert row.logit is not None and np.isfinite(row.logit)
    provenance = bundle.branch(BranchId.P)
    assert provenance.logit == 0.0, "no provenance reader exists yet; zero is the honest value"
    assert provenance.p is None, "a one-sided credit has no probability reading"


def test_video_bundle_stays_deterministic_after_the_change():
    a, b = _analyse(), _analyse()
    assert a.p_fake == b.p_fake
    assert [x.logit for x in a.branches] == [x.logit for x in b.branches]
    assert [f.p for f in a.video.frame_scores] == [f.p for f in b.video.frame_scores]


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
