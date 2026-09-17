"""Branch P payload: C2PA content credentials and EXIF coherence.

The whole module exists under one rule, frozen at architecture review:

    A valid signature is evidence of authenticity.
    The absence of provenance data is evidence of nothing.

``authenticity_credit`` is therefore bounded to [0, 1] rather than [-1, 1], so a
metadata-stripped photograph — which describes almost every real upload — cannot
be pushed toward a fake verdict by this branch.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from contracts.common import Base


class C2paReport(Base):
    present: bool
    signature_valid: bool | None = Field(
        default=None, description="Null when no manifest was present."
    )
    signer: str | None = None
    claim_generator: str | None = None
    captured_at: datetime | None = None
    edits_declared: list[str] = Field(
        default_factory=list, description="Declared edit actions from the manifest, if any."
    )

    @model_validator(mode="after")
    def _validity_requires_presence(self) -> C2paReport:
        if not self.present and self.signature_valid is not None:
            raise ValueError("signature_valid must be null when no manifest is present")
        if self.present and self.signature_valid is None:
            raise ValueError("signature_valid is required when a manifest is present")
        return self


class ExifReport(Base):
    present: bool
    coherent: bool | None = Field(
        default=None,
        description=(
            "Whether the declared camera, sensor dimensions and image dimensions "
            "agree. Null when no EXIF is present."
        ),
    )
    camera_make: str | None = None
    camera_model: str | None = None
    software: str | None = None
    anomalies: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coherence_requires_presence(self) -> ExifReport:
        if not self.present and self.coherent is not None:
            raise ValueError("coherent must be null when no EXIF is present")
        return self


class ProvenanceFinding(Base):
    """One human-readable line for the Results page. Rendered verbatim."""

    label: str
    detail: str
    supports_authenticity: bool = Field(
        description=(
            "True only for positive evidence. A finding that merely notes missing "
            "data sets this False and still contributes zero to the score."
        )
    )


class ProvenanceReport(Base):
    c2pa: C2paReport
    exif: ExifReport
    authenticity_credit: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "The one-sided feature handed to fusion. 0.0 means 'no information', "
            "which is the correct and common case. Bounded below at zero by "
            "construction so absence can never imply manipulation."
        ),
    )
    findings: list[ProvenanceFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def _credit_requires_positive_evidence(self) -> ProvenanceReport:
        """Credit above zero must be backed by something verifiable.

        Coherent EXIF alone is weak evidence and is allowed to earn a small
        credit; a valid C2PA signature is strong. Neither present means zero, and
        that is enforced here rather than trusted to the caller.
        """
        has_valid_signature = self.c2pa.present and bool(self.c2pa.signature_valid)
        has_coherent_exif = self.exif.present and bool(self.exif.coherent)
        if self.authenticity_credit > 0.0 and not (has_valid_signature or has_coherent_exif):
            raise ValueError(
                "authenticity_credit must be 0.0 without a valid C2PA signature "
                "or coherent EXIF"
            )
        if self.authenticity_credit > 0.3 and not has_valid_signature:
            raise ValueError(
                "authenticity_credit above 0.3 requires a valid C2PA signature; "
                "EXIF coherence alone is weak evidence"
            )
        return self

    @property
    def has_information(self) -> bool:
        return self.c2pa.present or self.exif.present


EMPTY_PROVENANCE = ProvenanceReport(
    c2pa=C2paReport(present=False),
    exif=ExifReport(present=False),
    authenticity_credit=0.0,
    findings=[
        ProvenanceFinding(
            label="No content credentials",
            detail=(
                "This file carries no C2PA manifest and no EXIF metadata. That is "
                "normal for images shared through social platforms, which strip "
                "metadata, and it neither supports nor undermines authenticity."
            ),
            supports_authenticity=False,
        )
    ],
)
"""The overwhelmingly common case. Note the wording: explicitly not suspicious."""
