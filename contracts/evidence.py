"""The EvidenceBundle — the keystone contract.

Everything the platform produces converges on this object, and every consumer
reads it. It is defined before any model exists so that the frontend, backend and
workers can be built in parallel against fixtures.

STABILITY TIERS
---------------
Stable — compatibility guaranteed within a MAJOR contract version. Breaking one
of these bumps ``CONTRACT_VERSION`` MAJOR and requires a client migration:

    schema_version, job_id, media, verdict, p_fake, interval, threshold,
    branches[].{id,status,logit,p}, quality.gate_passed, model.version

Evolving — may gain or reshape fields within a MAJOR version. Clients must
tolerate additions and must not assume presence:

    video.*, provenance.findings, artifacts[], faces[], timing, disagreement,
    abstain_reason detail
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, model_validator

from contracts.branch import BranchResult
from contracts.common import Artifact, Base, FaceTrack, MediaInfo, ModelInfo, QualityReport, ThresholdInfo, Timing
from contracts.enums import CONTRACT_VERSION, AbstainReason, BranchId, BranchStatus, MediaKind, Verdict
from contracts.provenance import ProvenanceReport
from contracts.video import VideoAnalysis


class EvidenceBundle(Base):
    """The single result object. Cached under ``sha256 + model.version``."""

    schema_version: str = Field(default=CONTRACT_VERSION)
    job_id: UUID
    created_at: datetime

    media: MediaInfo

    verdict: Verdict = Field(
        ...,
        description=(
            "The three-state result for the media's single published face — the "
            "largest one detected (face_order_key selects it; see FaceTrack's own "
            "note). A multi-subject photograph's verdict describes one person in "
            "it; the platform does not currently score or publish evidence for "
            "any other face present (R16, open)."
        ),
    )
    p_fake: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated probability of manipulation. Null iff the verdict is "
            "insufficient_evidence. Stored at full precision — never round before "
            "persisting. Describes the same single selected face verdict does, "
            "not an aggregate over every face in the media."
        ),
    )
    interval: tuple[float, float] | None = Field(
        default=None, description="Uncertainty band around p_fake. Widens with branch disagreement."
    )
    threshold: ThresholdInfo
    abstain_reason: AbstainReason | None = Field(
        default=None, description="Required iff verdict is insufficient_evidence."
    )

    quality: QualityReport
    branches: list[BranchResult] = Field(min_length=1)
    disagreement: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Spread across scored branches. Surfaced to the user rather than "
            "collapsed — contradiction between branches is information."
        ),
    )

    faces: list[FaceTrack] = Field(default_factory=list)
    video: VideoAnalysis | None = None
    provenance: ProvenanceReport
    artifacts: list[Artifact] = Field(default_factory=list)

    model: ModelInfo
    timing: Timing

    # ---- invariants ------------------------------------------------------

    @model_validator(mode="after")
    def _verdict_and_probability_agree(self) -> EvidenceBundle:
        abstained = self.verdict is Verdict.INSUFFICIENT_EVIDENCE
        if abstained:
            if self.p_fake is not None:
                raise ValueError("p_fake must be null when verdict is insufficient_evidence")
            if self.abstain_reason is None:
                raise ValueError("abstain_reason is required when verdict is insufficient_evidence")
        else:
            if self.p_fake is None:
                raise ValueError(f"p_fake is required when verdict is {self.verdict}")
            if self.abstain_reason is not None:
                raise ValueError("abstain_reason must be null unless the verdict abstains")
        return self

    @model_validator(mode="after")
    def _interval_brackets_estimate(self) -> EvidenceBundle:
        if self.interval is None:
            return self
        lo, hi = self.interval
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError("interval must satisfy 0 <= lo <= hi <= 1")
        if self.p_fake is not None and not (lo <= self.p_fake <= hi):
            raise ValueError("interval must bracket p_fake")
        return self

    @model_validator(mode="after")
    def _branches_unique(self) -> EvidenceBundle:
        ids = [b.id for b in self.branches]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate branch id in branches[]")
        return self

    @model_validator(mode="after")
    def _video_block_matches_media_kind(self) -> EvidenceBundle:
        if self.media.kind is MediaKind.VIDEO and self.video is None:
            raise ValueError("video analysis is required when media.kind is video")
        if self.media.kind is MediaKind.IMAGE and self.video is not None:
            raise ValueError("video analysis must be null when media.kind is image")
        return self

    @model_validator(mode="after")
    def _top_frames_name_a_published_track(self) -> EvidenceBundle:
        """A frame may not point at a face the bundle does not publish.

        **R18.** ``top_frames()`` hard-coded ``track_id=0`` while the video path
        publishes no ``FaceTrack`` at all, so every video bundle carried three
        references to a track that did not exist. Fixing the value at that one
        call site would leave the next caller free to reintroduce it; this makes
        the incoherent bundle unrepresentable — by hand, from a factory, and on
        the way back in from storage or the wire.

        ``None`` stays legal and is the common case: it means no face was
        attributed to this frame, which is true of every video bundle the
        platform produces today. What is refused is a bundle that *names* a
        track and then does not publish it.

        Scope: this checks *referential integrity* — a named track exists. Whether
        ``faces[]`` is internally consistent (unique ids; a headline that agrees
        with a published face) is R21, closed separately by
        ``_faces_are_uniquely_identified`` and
        ``_headline_is_the_score_of_a_published_face`` below.
        """
        if self.video is None:
            return self
        published = {face.track_id for face in self.faces}
        dangling = sorted(
            {
                frame.track_id
                for frame in self.video.top_frames
                if frame.track_id is not None and frame.track_id not in published
            }
        )
        if dangling:
            raise ValueError(
                f"video.top_frames name track(s) {dangling} but faces[] publishes "
                f"{sorted(published)}; a client resolving those ids finds nothing. "
                "Use track_id=None when no face was attributed to the frame."
            )
        return self

    @model_validator(mode="after")
    def _faces_are_uniquely_identified(self) -> EvidenceBundle:
        """No two ``FaceTrack``s may share a ``track_id``.

        **R21.** ``branches[]`` had ``_branches_unique`` since Batch 0; ``faces[]``
        had nothing. A bundle carrying two tracks both at ``track_id=0`` with
        contradictory ``p_fake`` validated, and ``_top_frames_name_a_published_track``
        silently deduplicates the ids it resolves against, so a client could not
        tell the two apart either. This is the same invariant ``_branches_unique``
        is, on the field R16 will populate.

        Harmless today — the pipeline publishes at most one track — and a
        precondition for anything that publishes more than one.
        """
        ids = [face.track_id for face in self.faces]
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"duplicate track_id in faces[]: {sorted(ids)}. Each FaceTrack is "
                "a distinct identity; two entries sharing an id cannot be resolved "
                "apart, and video.top_frames names faces by that id."
            )
        return self

    @model_validator(mode="after")
    def _headline_is_the_score_of_a_published_face(self) -> EvidenceBundle:
        """A decided ``p_fake`` must be the score of a face the bundle publishes.

        **R21.** ``faces[]`` was the one evidence list with no tie to the
        headline. A bundle claiming ``manipulated`` at 0.913 beside faces scored
        0.90 and 0.10 — a figure matching neither — was a valid ``EvidenceBundle``.
        Every other evidence-to-headline path is pinned: ``_verdict_follows_
        threshold`` binds the verdict to ``p_fake``, ``_branches_unique`` and the
        ``ThresholdInfo`` band derivation bind the rest. This closes the last gap,
        on the field a per-face design will populate.

        Deliberately narrow, so it forbids drift without asserting a policy the
        product has not chosen:

        * **Only when the bundle carries a decided ``p_fake``.** An abstention
          publishes no headline probability to contradict. The image path also
          legitimately publishes a per-face ``p_fake`` on an *in-band* abstention
          (bundle ``p_fake`` is ``None``, the face's is not) — branch-score
          visibility, not drift — and that stays representable.
        * **Only against faces that themselves scored.** An abstained face
          (``status`` set, ``p_fake`` null) is not a candidate.
        * **Vacuous when no published face scored.** The video path publishes
          ``faces=[]`` with a decided verdict today (R18); that is unchanged.

        The rule — *if any published face scored, the headline equals one of them
        exactly* — holds under today's single-face pipeline by construction
        (``faces[0].p_fake`` is computed from the same fused outcomes as the
        headline) and under an R16 per-face design, where the headline is defined
        as one designated face's score. A future *averaging* aggregate would not
        satisfy it; that is a deliberate meaning change to a Stable field and
        should trip this check rather than pass it unnoticed.
        """
        if self.p_fake is None:
            return self
        scored = [face.p_fake for face in self.faces if face.p_fake is not None]
        if scored and self.p_fake not in scored:
            raise ValueError(
                f"p_fake={self.p_fake} is the score of no published face "
                f"({scored}); the headline probability must be the score of a "
                "face carried in faces[]. A bundle whose headline contradicts its "
                "own evidence is the drift this check exists to prevent."
            )
        return self

    @model_validator(mode="after")
    def _gate_failure_implies_abstention(self) -> EvidenceBundle:
        if not self.quality.gate_passed and self.verdict is not Verdict.INSUFFICIENT_EVIDENCE:
            raise ValueError("a failed quality gate must produce verdict=insufficient_evidence")
        return self

    @model_validator(mode="after")
    def _verdict_follows_threshold(self) -> EvidenceBundle:
        """A published verdict must be re-derivable from p_fake and threshold.

        Without this a UI could show 'authentic' next to a probability above the
        decision boundary, and nobody would notice until a user did.
        """
        if self.p_fake is None:
            return self
        # Checked against the band's edges, which are where the decision is
        # actually made. Comparing against ``threshold.value`` alone accepted a
        # published verdict for any score on the correct side of the centre,
        # including scores inside the abstention band that the engine would
        # never have decided — so a hand-built or mis-generated bundle could
        # claim a verdict the pipeline could not have produced.
        if self.p_fake >= self.threshold.band_hi:
            expected = Verdict.MANIPULATED
        elif self.p_fake < self.threshold.band_lo:
            expected = Verdict.AUTHENTIC
        else:
            raise ValueError(
                f"p_fake={self.p_fake} lies inside the abstention band "
                f"[{self.threshold.band_lo}, {self.threshold.band_hi}) and so "
                f"cannot carry the decided verdict {self.verdict}"
            )
        if self.verdict is not expected:
            raise ValueError(
                f"verdict {self.verdict} contradicts p_fake={self.p_fake} "
                f"against threshold {self.threshold.value} "
                f"(band [{self.threshold.band_lo}, {self.threshold.band_hi}))"
            )
        return self

    # ---- convenience -----------------------------------------------------

    @property
    def scored_branches(self) -> list[BranchResult]:
        return [b for b in self.branches if b.status is BranchStatus.SCORED]

    def branch(self, bid: BranchId) -> BranchResult | None:
        return next((b for b in self.branches if b.id is bid), None)

    @property
    def is_abstention(self) -> bool:
        return self.verdict is Verdict.INSUFFICIENT_EVIDENCE
