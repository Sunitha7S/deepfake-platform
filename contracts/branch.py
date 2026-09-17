"""Per-branch results.

Abstention is a first-class status, not an exception. A branch that declines to
vote returns a ``BranchResult`` like any other, and fusion handles the gap.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from contracts.common import Base
from contracts.enums import AbstainReason, BranchId, BranchStatus

BRANCH_NAMES: dict[BranchId, str] = {
    BranchId.A: "Spatial",
    BranchId.B: "Semantic",
    BranchId.P: "Provenance",
}

BRANCH_DESCRIPTIONS: dict[BranchId, str] = {
    # R22. These strings describe *what each branch scores*, and they used to
    # describe the frozen-architecture target rather than the shipped code. The
    # ``ModelBackend.score`` interface takes one prepared tensor and returns one
    # logit vector per branch, and both pipelines call it once — so today every
    # branch scores the *same* tensor: the aligned face crop in the image path,
    # the whole resized frame in the video path (which has no face stage, R18).
    # Whole-image scoring for branch B, and face-crop scoring for branch A on
    # video, both need a dual-input backend that does not exist yet.
    #
    # This dict has no functional consumer — it is documentation, exported for a
    # future UI and pinned dead by ``test_branch_descriptions_have_no_functional_
    # consumer``. If it ever gains one, the "no contract change" basis for the
    # Batch 17 wording change no longer holds.
    BranchId.A: (
        "ConvNeXt V2-T with SRM/Bayar residual channels. Frozen-architecture "
        "target: the aligned face crop. Today it scores the crop in the image "
        "path and the whole resized frame in the video path (no face stage, R18)."
    ),
    BranchId.B: (
        "DINOv2-B/14 frozen features with a trained probe. Frozen-architecture "
        "target: the whole image, for scene-level context a face crop cannot "
        "show. Today it scores the same prepared tensor as branch A — the "
        "aligned face crop in the image path, the whole resized frame in the "
        "video path (R22)."
    ),
    BranchId.P: "C2PA signature and EXIF coherence. Not a learned model.",
}


class BranchResult(Base):
    """One evidence source's contribution.

    ``logit`` is stored at full precision and must never be rounded before
    persistence. Rounding collapses distinct saturated scores into ties, which
    silently perturbs AUC and breaks the monotonic relationship between logit
    and p_fake that calibration depends on.
    """

    id: BranchId
    name: str
    status: BranchStatus
    logit: float | None = Field(
        default=None,
        description=(
            "Raw pre-calibration logit. Full precision, never rounded. "
            "For branch P this is a one-sided authenticity credit and MUST be >= 0 "
            "— absence of provenance contributes exactly zero, never positive "
            "evidence of manipulation."
        ),
    )
    p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Branch-local probability, for display only."
    )
    weight: float | None = Field(
        default=None, description="Fusion coefficient applied. Null when the branch did not score."
    )
    abstain_reason: AbstainReason | None = None
    version: str | None = None

    @model_validator(mode="after")
    def _status_consistency(self) -> BranchResult:
        if self.status is BranchStatus.SCORED:
            if self.logit is None:
                raise ValueError(f"branch {self.id}: logit is required when status=scored")
            if self.abstain_reason is not None:
                raise ValueError(f"branch {self.id}: abstain_reason must be null when scored")
        else:
            if self.logit is not None or self.p is not None:
                raise ValueError(f"branch {self.id}: logit/p must be null when not scored")
            if self.abstain_reason is None:
                raise ValueError(f"branch {self.id}: abstain_reason is required when not scored")
        return self

    @model_validator(mode="after")
    def _provenance_is_one_sided(self) -> BranchResult:
        """Structural guard for the frozen one-sided-provenance decision.

        A fitted fusion would otherwise learn 'no metadata implies fake', because
        that correlation is real in every training corpus and false in
        production, where platforms strip metadata from nearly everything. Making
        a negative value unrepresentable is cheaper than remembering not to
        produce one.
        """
        if self.id is BranchId.P and self.logit is not None and self.logit < 0.0:
            raise ValueError(
                "branch P logit must be >= 0: provenance may only support authenticity, "
                "never contribute toward a fake verdict"
            )
        return self
