"""Branch fusion, calibration, thresholding, and bundle construction.

**The only module in the package that constructs an ``EvidenceBundle``.** Every
contract invariant is enforced here, once, rather than being re-derived by each
pipeline — two pipelines each building bundles is two places for the same
invariant to be got subtly differently wrong.

The structural point: ``verdict`` is *computed* from ``p_fake`` and the
threshold, never accepted as an argument. That makes "the verdict must be
derivable from the probability" impossible to violate rather than merely
checked. The same applies to branch P's logit, which is clamped at zero on the
way in — the one-sided provenance rule enforced at the point of construction
instead of trusted to every call site.

BRANCH SEMANTICS
----------------
The three branches do not speak the same language, and fusing them as though
they did was the defect this module was corrected for. Their units:

    Branch A  signed log-odds of MANIPULATION from spatial evidence.
              Range ℝ. Positive means "looks manipulated"; negative means
              "looks authentic"; zero means "no preference".

    Branch B  signed log-odds of MANIPULATION from semantic evidence.
              Same units, same sign convention, different evidence.

    Branch P  UNSIGNED magnitude of evidence FOR AUTHENTICITY — an
              "authenticity credit". Range [0, ∞), bounded below at zero by
              ``BranchResult`` and by ``ProvenanceReport.authenticity_credit``.
              Zero means "no information", which is the overwhelmingly common
              case because platforms strip metadata. It does NOT mean "50%
              likely fake".

A and B are commensurable and may be averaged with one another. **P is not on
that scale at all** and averaging it in is a category error with two visible
consequences, both of which were present before this correction:

  1. A positive credit — a valid C2PA signature — raised the fused log-odds of
     manipulation, so cryptographic proof of authenticity made a file look
     *more* fake.
  2. A zero credit still entered the renormalised mean and shrank the fused
     logit toward zero, so the mere *absence* of provenance pushed confident
     scores toward the abstention band. That is the one-sided rule being
     violated through arithmetic rather than through a negative logit, which is
     the one route the structural guard in ``contracts/branch.py`` cannot see.

FUSION, CORRECTED
-----------------
Manipulation evidence is pooled among the branches that measure it, and
provenance is applied afterwards as a one-sided subtraction:

    z_evidence = Σ w_i · L_i / Σ w_i      over scored i ∈ {A, B}
    z          = z_evidence − w_P · credit_P            w_P ≥ 0, credit_P ≥ 0

Three properties follow, and each is pinned by a test:

  * ``credit_P == 0`` reproduces ``z_evidence`` exactly — bit-identical to a
    run in which branch P is absent entirely.
  * ``credit_P > 0`` can only lower ``z``. Correctly-rounded IEEE-754
    subtraction is monotone, so ``z ≤ z_evidence`` holds unconditionally for
    every finite input rather than merely for realistic ones.
  * No manipulation branch scored means ``None``, and the caller abstains.
    Provenance *modulates* manipulation evidence; it never *constitutes* it. A
    valid signature says nothing about the pixels when nothing has looked at
    them, and returning σ(−w_P·credit) there would be a manipulation
    probability derived from no examination of the media.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from contracts.branch import BRANCH_NAMES, BranchResult
from contracts.common import (
    Artifact,
    BoundingBox,
    FaceTrack,
    MediaInfo,
    ModelInfo,
    QualityReport,
    ThresholdInfo,
    Timing,
)
from contracts.enums import (
    AbstainReason,
    BranchId,
    BranchStatus,
    MediaKind,
    ResolutionClass,
    Verdict,
)
from contracts.evidence import EvidenceBundle
from contracts.job import JobSpec
from contracts.provenance import EMPTY_PROVENANCE
from contracts.video import FrameScore, SuspectSegment, TopFrame, VideoAnalysis
from inference.calibration import sigmoid, temperature_scale
from inference.models import ModelSpec


@dataclass(frozen=True)
class BranchOutcome:
    """One branch's contribution, before fusion."""

    logit: float | None = None
    abstain_reason: AbstainReason | None = None

    @property
    def scored(self) -> bool:
        return self.logit is not None


@dataclass
class QualityFacts:
    """What the gates measured. Drives both the gate decision and the report."""

    width: int
    height: int
    face_px: int | None = None
    jpeg_q_est: float | None = None
    blur_score: float | None = None
    gate_passed: bool = True
    gate_reason: AbstainReason | None = None

    @property
    def resolution_class(self) -> ResolutionClass:
        short = min(self.width, self.height)
        if short < 256:
            return ResolutionClass.TINY
        if short < 480:
            return ResolutionClass.LOW
        if short < 1080:
            return ResolutionClass.STANDARD
        return ResolutionClass.HIGH

    def to_report(self) -> QualityReport:
        return QualityReport(
            jpeg_q_est=self.jpeg_q_est,
            face_px=self.face_px,
            blur_score=self.blur_score,
            resolution_class=self.resolution_class,
            gate_passed=self.gate_passed,
            gate_reason=self.gate_reason,
        )


# --------------------------------------------------------------------------
# Fusion and calibration
# --------------------------------------------------------------------------


MANIPULATION_BRANCHES: tuple[BranchId, ...] = (BranchId.A, BranchId.B)
"""Branches whose logits are signed log-odds of manipulation.

Only these are commensurable and only these may be pooled with one another.
Branch P is deliberately absent: it reports an unsigned authenticity credit on a
different scale, and ``fuse`` applies it separately. Iterating this tuple rather
than ``outcomes`` is what makes including P by accident impossible.
"""


def manipulation_evidence(
    outcomes: dict[BranchId, BranchOutcome], weights: dict[BranchId, float]
) -> float | None:
    """Weighted mean of the scored manipulation branches. None if none scored.

    Renormalised over the branches that actually scored, so an abstaining
    branch does not silently drag the fused logit toward zero. A missing branch
    should reduce confidence, which the interval widening handles — it should
    not act as a vote for "authentic".
    """
    total_weight = 0.0
    total = 0.0
    for branch in MANIPULATION_BRANCHES:
        outcome = outcomes.get(branch)
        if outcome is None or not outcome.scored:
            continue
        w = weights.get(branch, 0.0)
        if w <= 0.0:
            continue
        total += w * float(outcome.logit)
        total_weight += w
    if total_weight <= 0.0:
        return None
    return total / total_weight


def authenticity_credit(outcomes: dict[BranchId, BranchOutcome]) -> float:
    """Branch P's credit, floored at zero. Zero when P did not score.

    Clamped here as well as at ``BranchResult`` construction because this is the
    value that reaches the arithmetic. Validating the published field but fusing
    the raw one would leave the guarantee resting on the two staying in step.
    """
    outcome = outcomes.get(BranchId.P)
    if outcome is None or not outcome.scored:
        return 0.0
    return max(0.0, float(outcome.logit))


def fuse(
    outcomes: dict[BranchId, BranchOutcome], weights: dict[BranchId, float]
) -> float | None:
    """Fused log-odds of manipulation. None when no manipulation branch scored.

    Manipulation evidence is pooled first; provenance is then subtracted as a
    one-sided credit. See the module docstring for why the two cannot be
    averaged together.

    The subtraction — rather than an averaged-in term — is what makes the
    one-sided rule hold arithmetically instead of only at the field validator:
    ``w_P ≥ 0`` and ``credit ≥ 0`` give ``z ≤ z_evidence`` for every finite
    input, so provenance can lower a fake score and can never raise one.
    """
    evidence = manipulation_evidence(outcomes, weights)
    if evidence is None:
        # Provenance modulates manipulation evidence; it does not constitute it.
        # With nothing to modulate there is no honest number to return.
        return None
    weight_p = weights.get(BranchId.P, 0.0)
    if weight_p <= 0.0:
        return evidence
    return evidence - weight_p * authenticity_credit(outcomes)


def calibrated_probability(fused_logit: float, spec: ModelSpec) -> float:
    """Apply the model's temperature and squash. Bounded [0,1] structurally."""
    p = float(temperature_scale(fused_logit, spec.calibration.temperature))
    return min(1.0, max(0.0, p))


def disagreement_of(outcomes: dict[BranchId, BranchOutcome]) -> float:
    """Spread across the scored *manipulation* branches, as probabilities.

    Surfaced rather than collapsed: branches contradicting each other is
    information, and averaging it away is the cheapest way to lose it.

    Branch P is excluded for the same reason it is excluded from the mean — it
    is not on this scale. Including it treated a zero credit as σ(0) = 0.5 and
    made every ordinary upload look like a branch shouting "50%" at two branches
    agreeing on 0.05: measured spread 0.45 where the real disagreement was 0.03,
    and an interval roughly six times wider than the evidence warranted. Branch
    P disagreeing with branch A is not a meaningful quantity, because they are
    not answering the same question.
    """
    ps = [
        float(sigmoid(outcomes[b].logit))
        for b in MANIPULATION_BRANCHES
        if b in outcomes and outcomes[b].scored
    ]
    return float(max(ps) - min(ps)) if len(ps) > 1 else 0.0


def confidence_interval(p: float, disagreement: float) -> tuple[float, float]:
    """Interval around ``p``, widened by branch disagreement.

    A crude uncertainty estimate and labelled as such — it is not a posterior.
    It exists so a UI can show that two branches disagreeing means less
    certainty than two agreeing, which is true and useful even when the width
    is not rigorous.
    """
    half = 0.03 + 0.5 * disagreement
    return max(0.0, p - half), min(1.0, p + half)


def decide(p: float, threshold: ThresholdInfo) -> tuple[Verdict, AbstainReason | None]:
    """Map a calibrated probability to a verdict.

    Three ordered comparisons against the band's two edges, which is what makes
    the regions a partition of [0, 1] rather than three conditions that happen
    not to overlap. The previous form tested ``p >= value`` first and the band
    second, so the band's edges were consulted only for scores the threshold had
    already declined, and any disagreement between them resolved silently in
    favour of whichever was checked first.

    ``band_hi`` is the decision boundary, not ``value``. See ``ThresholdInfo``
    for what that costs and why.
    """
    if p >= threshold.band_hi:
        return Verdict.MANIPULATED, None
    if p >= threshold.band_lo:
        return Verdict.INSUFFICIENT_EVIDENCE, AbstainReason.SCORE_IN_ABSTENTION_BAND
    return Verdict.AUTHENTIC, None


# --------------------------------------------------------------------------
# Bundle construction
# --------------------------------------------------------------------------


def _branch_results(
    outcomes: dict[BranchId, BranchOutcome], spec: ModelSpec
) -> list[BranchResult]:
    """Build branch rows, enforcing the one-sided provenance rule at the source."""
    rows: list[BranchResult] = []
    for branch in BranchId:  # iterate the enum so ids are unique by construction
        outcome = outcomes.get(branch)
        if outcome is None:
            rows.append(
                BranchResult(
                    id=branch,
                    name=BRANCH_NAMES[branch],
                    status=BranchStatus.UNAVAILABLE,
                    abstain_reason=AbstainReason.BRANCH_NOT_APPLICABLE,
                    version=spec.version,
                )
            )
            continue
        if not outcome.scored:
            rows.append(
                BranchResult(
                    id=branch,
                    name=BRANCH_NAMES[branch],
                    status=BranchStatus.ABSTAINED,
                    abstain_reason=outcome.abstain_reason or AbstainReason.BRANCH_ERROR,
                    version=spec.version,
                )
            )
            continue
        logit = float(outcome.logit)
        if branch is BranchId.P:
            # Clamped, not asserted. Absence of provenance contributes exactly
            # zero; a negative value here is the shortcut the rule exists to
            # prevent, and making it unrepresentable beats catching it later.
            logit = max(0.0, logit)
        rows.append(
            BranchResult(
                id=branch,
                name=BRANCH_NAMES[branch],
                status=BranchStatus.SCORED,
                logit=logit,
                # Branch P has no probability interpretation, so it publishes
                # none. σ(credit) would render "no provenance" — by far the
                # commonest case — as a confident-looking 0.5, which a Results
                # page shows as "Provenance: 50% manipulated". That is the
                # one-sided rule leaking into the UI: a branch that holds no
                # information must not display as a branch that is undecided.
                p=None if branch is BranchId.P else float(sigmoid(logit)),
                weight=spec.branch_weights.get(branch),
                version=spec.version,
            )
        )
    return rows


def _media_info(
    spec: JobSpec,
    *,
    size_bytes: int,
    width: int,
    height: int,
    kind: MediaKind,
    duration_s: float | None = None,
    frame_count: int | None = None,
    fps: float | None = None,
) -> MediaInfo:
    return MediaInfo(
        kind=kind,
        sha256=spec.sha256,
        size_bytes=size_bytes,
        mime=spec.declared_mime,
        width=width,
        height=height,
        duration_s=duration_s,
        frame_count=frame_count,
        fps=fps,
    )


def _model_info(spec: ModelSpec, runtime_version: str | None = None) -> ModelInfo:
    """Publish the *runtime* version, never the bare weights version.

    This value is the result cache's write key. Publishing ``spec.version`` here
    meant a recalibration — a new temperature reinterpreting every logit —
    produced an identical key, so the cache went on serving probabilities
    computed under the temperature that had just been replaced.

    Batch 9 extends the same argument one step. ``spec.effective_version`` names
    the model; it does not name the decoder, and Batch 8 measured a decoder
    change moving ``p_fake`` far enough to flip 1.3% of decisions. So the value
    published here is the *runtime* version — model identity plus a decode
    segment — and the cache key inherits it.

    ``runtime_version=None`` falls back to the model identity, which is what a
    caller constructing a bundle outside a pipeline gets. That is a version
    making no claim about its decode environment rather than a wrong one:
    ``contracts.version.is_environment_pinned`` tells the two apart.
    """
    return ModelInfo(
        version=runtime_version or spec.effective_version,
        branch_versions={b.value: spec.version_of(b) for b in BranchId},
        calibration_temp=spec.calibration.temperature,
    )


def _build(
    *,
    job: JobSpec,
    model: ModelSpec,
    media: MediaInfo,
    quality: QualityFacts,
    outcomes: dict[BranchId, BranchOutcome],
    faces: list[FaceTrack],
    video: VideoAnalysis | None,
    artifacts: list[Artifact],
    timing: Timing,
    forced_abstention: AbstainReason | None,
    runtime_version: str | None = None,
) -> EvidenceBundle:
    """The single construction point. Every invariant is settled here."""
    branches = _branch_results(outcomes, model)
    disagreement = disagreement_of(outcomes)

    fused = fuse(outcomes, model.branch_weights)
    if forced_abstention is not None or fused is None:
        reason = forced_abstention or AbstainReason.BRANCH_ERROR
        verdict, p_fake, interval = Verdict.INSUFFICIENT_EVIDENCE, None, None
    else:
        p = calibrated_probability(fused, model)
        verdict, band_reason = decide(p, model.threshold)
        if verdict is Verdict.INSUFFICIENT_EVIDENCE:
            # The number exists and is deliberately discarded. Publishing a
            # probability beside "insufficient evidence" invites a reader to use
            # it anyway, which is the opposite of what abstention means. Branch
            # scores stay visible — informative without a headline figure that
            # overstates what is known.
            reason, p_fake, interval = band_reason, None, None
        else:
            reason, p_fake = None, p
            interval = confidence_interval(p, disagreement)

    return EvidenceBundle(
        job_id=job.job_id,
        created_at=datetime.now(UTC),
        media=media,
        verdict=verdict,
        p_fake=p_fake,
        interval=interval,
        threshold=model.threshold,
        abstain_reason=reason,
        quality=quality.to_report(),
        branches=branches,
        disagreement=disagreement,
        faces=faces,
        video=video,
        provenance=EMPTY_PROVENANCE,
        artifacts=artifacts,
        model=_model_info(model, runtime_version),
        timing=timing,
    )


def build_image_bundle(
    *,
    job: JobSpec,
    model: ModelSpec,
    size_bytes: int,
    width: int,
    height: int,
    quality: QualityFacts,
    outcomes: dict[BranchId, BranchOutcome],
    faces: list[FaceTrack] | None = None,
    artifacts: list[Artifact] | None = None,
    timing: Timing,
    forced_abstention: AbstainReason | None = None,
    runtime_version: str | None = None,
) -> EvidenceBundle:
    """Build an image bundle. ``video`` is always null — the contract requires it."""
    return _build(
        job=job,
        model=model,
        media=_media_info(
            job, size_bytes=size_bytes, width=width, height=height, kind=MediaKind.IMAGE
        ),
        quality=quality,
        outcomes=outcomes,
        faces=faces or [],
        video=None,
        artifacts=artifacts or [],
        timing=timing,
        forced_abstention=forced_abstention,
        runtime_version=runtime_version,
    )


def build_video_bundle(
    *,
    job: JobSpec,
    model: ModelSpec,
    size_bytes: int,
    width: int,
    height: int,
    duration_s: float,
    frame_count: int,
    fps: float,
    quality: QualityFacts,
    outcomes: dict[BranchId, BranchOutcome],
    video: VideoAnalysis,
    faces: list[FaceTrack] | None = None,
    artifacts: list[Artifact] | None = None,
    timing: Timing,
    forced_abstention: AbstainReason | None = None,
    runtime_version: str | None = None,
) -> EvidenceBundle:
    """Build a video bundle. ``video`` is mandatory — the contract requires it."""
    return _build(
        job=job,
        model=model,
        media=_media_info(
            job,
            size_bytes=size_bytes,
            width=width,
            height=height,
            kind=MediaKind.VIDEO,
            duration_s=duration_s,
            frame_count=frame_count,
            fps=fps,
        ),
        quality=quality,
        outcomes=outcomes,
        faces=faces or [],
        video=video,
        artifacts=artifacts or [],
        timing=timing,
        forced_abstention=forced_abstention,
        runtime_version=runtime_version,
    )


# --------------------------------------------------------------------------
# Temporal aggregation
# --------------------------------------------------------------------------


def topk_indices(scores: np.ndarray, k: int | None = None) -> list[int]:
    """Which frames the top-k pool selects. Deterministic, ties broken by index.

    Exposed separately from ``topk_mean`` so that **every branch pools over the
    same frames**. Selecting frames independently per branch would let branch A
    describe one moment of a clip and branch B another, and their "disagreement"
    would then be a statement about frame selection rather than about the media.

    ``k`` defaults to ``arr.size // 8`` for study and direct callers; since
    Batch 23 (R29) ``VideoPipeline`` always passes an explicit ``k`` derived from
    ``VideoSelectionPolicy.pool_divisor``, so the pooling depth is in the model
    identity rather than an unbound constant here.
    """
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("no frame scores to pool")
    k = min(max(1, k or arr.size // 8), arr.size)
    # Descending by score, ascending by index. np.argsort alone leaves ties in
    # an unspecified order, which would make a clip of identical frames pool
    # differently between runs — a determinism hole that only opens on the
    # inputs least likely to be tested.
    order = np.lexsort((np.arange(arr.size), -arr))
    return sorted(int(i) for i in order[:k])


def pool_at(values: np.ndarray | list[float], indices: list[int]) -> float:
    """Mean of ``values`` at ``indices``. The pooling primitive."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if not indices:
        raise ValueError("no frames selected to pool over")
    return float(np.mean(arr[list(indices)]))


def topk_mean(scores: np.ndarray, k: int | None = None) -> float:
    """Mean of the highest ``k`` frame scores. The v1 pooling baseline.

    Mean over all frames would dilute a six-frame manipulation inside a
    forty-eight-frame clip into nothing, which is the failure the whole video
    path exists to avoid. ``k`` defaults to an eighth of the frames.
    """
    return pool_at(scores, topk_indices(scores, k))


def hysteresis_segments(
    frame_scores: list[FrameScore], *, enter: float, exit_: float
) -> list[SuspectSegment]:
    """Contiguous runs above ``enter``, ending when the score drops below ``exit_``.

    Two thresholds rather than one so a curve hovering at the boundary does not
    shatter into dozens of one-frame segments — which is what a single cut
    produces, and which reads to a user as noise rather than a finding.
    """
    segments: list[SuspectSegment] = []
    run: list[FrameScore] = []

    def flush() -> None:
        if not run:
            return
        ps = [f.p for f in run]
        segments.append(
            SuspectSegment(
                start_ms=run[0].t_ms,
                end_ms=run[-1].t_ms,
                peak_p=max(ps),
                mean_p=float(sum(ps) / len(ps)),
                frame_count=len(run),
            )
        )
        run.clear()

    for frame in frame_scores:
        if not run:
            if frame.p >= enter:
                run.append(frame)
        elif frame.p >= exit_:
            run.append(frame)
        else:
            flush()
    flush()
    return segments


def top_frames(
    frame_scores: list[FrameScore], *, limit: int = 3, track_id: int | None = None
) -> list[TopFrame]:
    """The highest-scoring frames, ranked. ``track_id`` names a face or is null.

    **R18.** This hard-coded ``track_id=0`` until Batch 13, while the only caller
    is the video path and the video path publishes no ``FaceTrack`` at all — so
    every video bundle carried three references to a track that did not exist. A
    client resolving them against ``faces[]`` found nothing.

    ``None`` is the honest value and it is now the default: it says *no face was
    attributed to this frame*, which is exactly true of a pipeline that scores
    whole frames and never detects. The parameter exists so a future face stage
    can pass a real id without touching this function, and
    ``EvidenceBundle._top_frames_name_a_published_track`` makes the dangling case
    unrepresentable rather than merely fixed here.
    """
    ranked = sorted(frame_scores, key=lambda f: (-f.p, f.frame_idx))[:limit]
    return [
        TopFrame(frame_idx=f.frame_idx, t_ms=f.t_ms, p=f.p, track_id=track_id)
        for f in ranked
    ]


def build_video_analysis(
    *,
    frames_sampled: int,
    frames_scored: int,
    frame_scores: list[FrameScore],
    threshold: ThresholdInfo,
    sampling: str = "uniform",
    pooling: str = "topk_mean",
) -> VideoAnalysis:
    # ``sampling`` / ``pooling`` are sourced from the ``VideoSelectionPolicy`` in
    # force by ``VideoPipeline`` (R29 / N4) — the wire value is the policy field.
    # The defaults name the shipped values for study / test callers that do not
    # thread a policy; they are held equal to ``default_video_selection()`` by
    # ``test_build_video_analysis_defaults_match_the_shipped_policy`` (R35), so
    # this fallback cannot drift into a stale second source of truth.
    return VideoAnalysis(
        frames_sampled=frames_sampled,
        frames_scored=frames_scored,
        sampling=sampling,
        pooling=pooling,
        frame_scores=frame_scores,
        # Enter on the decision boundary, leave at the bottom of the abstention
        # band. Entering at ``threshold.value`` would start a suspect segment in
        # the middle of the band — marking frames as suspect that the
        # bundle-level logic would report as insufficient evidence.
        segments=hysteresis_segments(
            frame_scores, enter=threshold.band_hi, exit_=threshold.band_lo
        ),
        top_frames=top_frames(frame_scores),
    )


def face_track(box_xywh: tuple[int, int, int, int], *, p_fake: float | None,
               frames_seen: int = 1,
               status: AbstainReason | None = None) -> FaceTrack:
    x, y, w, h = box_xywh
    return FaceTrack(
        track_id=0,
        bbox=BoundingBox(x=max(0, x), y=max(0, y), width=max(1, w), height=max(1, h)),
        frames_seen=frames_seen,
        p_fake=p_fake,
        status=status,
    )
