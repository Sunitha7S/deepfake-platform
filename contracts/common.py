"""Shared value objects: media description, quality gate, faces, artifacts."""

from __future__ import annotations

import math
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.enums import AbstainReason, ArtifactKind, MediaKind, ResolutionClass

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class Base(BaseModel):
    """Contract base. ``extra="forbid"`` turns a typo into a test failure.

    ``protected_namespaces=()`` is deliberate: fields like ``model_version`` are
    the natural names here, and pydantic's default guard against the ``model_``
    prefix would otherwise force awkward renames across the whole contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=False,
        protected_namespaces=(),
    )


class MediaInfo(Base):
    """What was analysed. Identifies the input without retaining the user's filename."""

    kind: MediaKind
    sha256: str = Field(pattern=SHA256_PATTERN, description="Content hash; half the cache key.")
    size_bytes: int = Field(ge=1)
    mime: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    duration_s: float | None = Field(default=None, ge=0.0)
    frame_count: int | None = Field(default=None, ge=1)
    fps: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _video_fields_match_kind(self) -> MediaInfo:
        video_only = (self.duration_s, self.frame_count, self.fps)
        if self.kind is MediaKind.IMAGE and any(v is not None for v in video_only):
            raise ValueError("duration_s/frame_count/fps must be null for kind=image")
        if self.kind is MediaKind.VIDEO and any(v is None for v in video_only):
            raise ValueError("duration_s/frame_count/fps are required for kind=video")
        return self


class QualityReport(Base):
    """Input quality, and the gate's decision.

    This is what routes low-information inputs to abstention *before* any model
    votes, and what the fusion layer uses as an interaction term.
    """

    jpeg_q_est: float | None = Field(default=None, ge=0.0, le=100.0)
    face_px: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Short edge — min(width, height), in source pixels — of the largest "
            "detected face, where face size is FaceBox.size_px (equal to the short "
            "edge): the one metric selection, the gate and this figure all read "
            "(see docs/FACE_SIZE_POLICY.md, R20). 0 when an image has no detected "
            "face; null on every video bundle, whose path runs no face stage (R18)."
        ),
    )
    blur_score: float | None = Field(default=None, ge=0.0)
    resolution_class: ResolutionClass
    gate_passed: bool
    gate_reason: AbstainReason | None = None

    @model_validator(mode="after")
    def _reason_iff_failed(self) -> QualityReport:
        if self.gate_passed and self.gate_reason is not None:
            raise ValueError("gate_reason must be null when gate_passed is true")
        if not self.gate_passed and self.gate_reason is None:
            raise ValueError("gate_reason is required when gate_passed is false")
        return self


class BoundingBox(Base):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class FaceTrack(Base):
    """One identity across the analysed frames.

    **What actually publishes these today, stated plainly because three other
    artifacts imply otherwise:**

    * **Images** yield exactly one track — the largest detected face, whose
      verdict is published as the whole image's. That is a policy the product
      chose for single-subject media, not a property of images; it is R16 and it
      is open.
    * **Video yields none at all.** ``VideoPipeline`` runs no face detection, so
      every video bundle carries ``faces=[]`` and a null ``quality.face_px``.
      That is R18. The fixtures and the stub handlers depict a populated track
      because they describe the post-SCRFD target the frontend builds against.

    ``frames_seen`` is therefore always 1 in practice; a value above 1 belongs to
    the tracked video path that does not exist yet.
    """

    track_id: int = Field(ge=0)
    bbox: BoundingBox = Field(description="Representative box, in source-image coordinates.")
    landmarks: list[tuple[float, float]] = Field(
        default_factory=list, description="5-point, source coordinates. Empty if unavailable."
    )
    frames_seen: int = Field(ge=1)
    p_fake: float | None = Field(default=None, ge=0.0, le=1.0)
    status: AbstainReason | None = Field(
        default=None, description="Set when this track abstained; null when it scored."
    )

    @model_validator(mode="after")
    def _score_xor_abstain(self) -> FaceTrack:
        if (self.p_fake is None) == (self.status is None):
            raise ValueError("exactly one of p_fake / status must be set on a face track")
        return self


class Artifact(Base):
    """A rendered explanation, stored out of band and referenced by URL.

    Never inlined as base64 — a bundle must stay small enough to cache cheaply.
    """

    kind: ArtifactKind
    ref: str = Field(description="Signed, expiring URL or storage key.")
    mime: str
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    expires_at: datetime | None = None
    caption: str = Field(
        description=(
            "User-facing caption. For GRADCAM this MUST read as attribution "
            "('regions that most influenced this score'), never as localisation."
        )
    )


def logit(p: float) -> float:
    """Log-odds of ``p``. The scale the band is defined on; see ThresholdInfo."""
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"logit is undefined at {p}; probabilities must lie strictly inside "
            "(0, 1)"
        )
    return math.log(p / (1.0 - p))


def expit(z: float) -> float:
    """Inverse of :func:`logit`, evaluated on the stable side of zero."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


LOG_ODDS_BAND_TOLERANCE = 1e-8
"""Relative slack allowed when re-deriving the band's margin from its edges.

The edges are stored as probabilities and the symmetry they must satisfy lives
in log-odds, so validation has to invert ``expit`` — and that inversion is
ill-conditioned near 0 and 1, where a one-ULP error in an edge becomes a large
error in its log-odds. Measured over 16134 (threshold, margin) pairs spanning
the whole admissible domain, the worst round-trip asymmetry is 2.54e-10 relative
to the margin. This is that figure with a factor of forty in hand, and it is
still seven orders of magnitude below the narrowest margin any operating point
would use.

The check is *relative* to the margin rather than absolute, because a margin is
unbounded above: an absolute tolerance would be vacuous for a wide band and
unsatisfiable for the edge cases the conditioning analysis above covers.
"""

BAND_EDGE_FLOOR = 1e-6
"""How far a band edge must stay from 0 and 1.

Two independent reasons, which happen to agree on the number:

*Meaning.* An edge within 1e-6 of an endpoint makes the region beyond it
unreachable — a ``band_hi`` of 1 − 1e-7 means no score is ever published as
manipulated. That is a broken operating point, not a strict one, and it should
fail loudly at construction rather than quietly at every request.

*Numerics.* The log-odds round trip degrades as an edge approaches an endpoint
(see :data:`LOG_ODDS_BAND_TOLERANCE`). 1e-6 is where it is still comfortably
inside the tolerance; at 1e-9 the worst case is 30× worse.

Note this is a floor on the *edges*, not on ``value``, and it is what bounds the
usable margin at a given threshold: ``13.8155 - |logit(value)|``. At the R11
threshold of 0.9734 that is a margin of 10.2, which is nine times wider than
anything a product would ship.
"""


DEFAULT_LOG_ODDS_MARGIN = math.log(1.5)
"""The shipped margin, pending a risk-coverage sweep on real scores.

Still a placeholder, and said so plainly — but a placeholder with three
properties the old ``band_width = 0.20`` did not have.

*It is stated in a unit that means something.* ln(1.5) is "abstain unless the
odds are at least 1.5x past the threshold odds", which is a sentence an operator
can agree or disagree with. 0.20 was not.

*It is valid at every threshold.* There is no threshold at which shipping this
default raises, which is the R11 closure.

*It changes as little as possible.* At the shipped threshold of 0.62 it produces
[0.5210, 0.7099), against the [0.5200, 0.7200) that shipped before — inside a
hundredth on both edges. The operating point moves, and moving it is correct:
the geometry changed, so every verdict near the boundary could change, and the
calibration fingerprint moves with it. It was chosen so that the move is small
enough to be reviewable rather than so that it is zero.

Derive the real value with ``evaluation.band.derive_operating_point``, which
takes the constraints that actually determine it: a selective-FPR ceiling, a
coverage floor, and the run-to-run jitter the band has to swallow.
"""


class ThresholdInfo(Base):
    """The operating point. Published so a verdict can be re-derived from p_fake.

    THE BAND IS DERIVED, NEVER CONFIGURED
    -------------------------------------
    ``band_lo`` and ``band_hi`` are outputs of ``value`` and a single width
    parameter, not independent settings. They stay in the payload because
    clients read them and the response schema is frozen, but every construction
    is checked against the derivation, so a band that disagrees with its
    threshold cannot be built — not by a factory, not by hand, and not by
    deserialising a stored bundle.

    Before this, the three were free to drift apart. ``decide`` compared against
    ``value`` first and the band second, so a configuration with
    ``band_hi > value`` silently truncated the band, and one with
    ``band_hi < value`` opened a dead zone in which scores just under the
    decision boundary were reported as authentic. Both were representable and
    neither was detectable from the published fields.

    THE BAND STRADDLES THE THRESHOLD
    --------------------------------
    ``band_lo < value < band_hi``, so the three regions partition [0, 1] with no
    gap and no overlap:

        p <  band_lo   authentic
        p <  band_hi   insufficient evidence
        otherwise      manipulated

    Note what this means for calibration: the boundary a score is actually
    decided at is ``band_hi``, not ``value``. ``value`` is the calibrated centre
    — the point ``threshold_at_fpr`` selects — and ``target_fpr`` is the false
    positive rate *at that centre*. Because the manipulated region starts above
    it, the achieved false positive rate is bounded above by ``target_fpr``
    rather than equal to it, and the cost is paid in detections: scores in
    ``[value, band_hi)`` are calibrated as manipulated and are reported as
    insufficient evidence instead.

    THE BAND IS SYMMETRIC IN LOG-ODDS, NOT IN PROBABILITY
    -----------------------------------------------------
    This is the Batch 7 change, and it is the whole of it. The edges are

        band_lo = expit(logit(value) - margin)
        band_hi = expit(logit(value) + margin)

    so the band is still *centred on its threshold* and still derived from one
    number — the centring has been moved to the scale where it means something.
    ``margin`` is an evidence margin in log-odds: a margin of ln(1.5) says
    "abstain unless the odds are at least 1.5x past the threshold odds".

    Two things forced this, one mathematical and one semantic.

    *Mathematical.* A band that is symmetric in probability with a width chosen
    independently of the threshold **cannot exist** for every threshold: a width
    of w needs ``w/2 <= min(value, 1 - value)``, so any fixed w fails once the
    threshold moves within w/2 of an endpoint. That is not a hypothetical. A 2%
    false-positive budget fitted on 22 negatives put the threshold at 0.9734,
    where no 0.20-wide symmetric band fits, and calibration raised (R11 in the
    audit). Clamping the band to [0, 1] is worse than raising: at that threshold
    the clamped band is [0.8734, 1.0] and nothing is ever called manipulated
    again. Log-odds symmetry has no such constraint — every finite margin around
    every threshold strictly inside (0, 1) yields a valid band — so the failure
    is structurally gone rather than handled.

    *Semantic.* A fixed probability width is not a fixed amount of evidence.
    Widening from 0.50 to 0.60 is 0.41 nats; from 0.90 to 1.00 is unbounded. The
    old 0.20-wide band therefore meant something different at every threshold it
    was used at, which is exactly why no value of it could be justified (R8).

    ``band_width`` is retained as a derived, published quantity — it is what the
    risk-coverage reports read — but it is an output now, never an input.
    """

    value: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated centre of the operating point. Always strictly inside "
            "(0, 1), since a threshold at an endpoint has no log-odds and no "
            "band can bracket it — but the bound stays [0, 1] so the published "
            "JSON Schema is unchanged; the model validator below is what "
            "rejects the endpoints."
        ),
    )
    target_fpr: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "False positive rate the centre was tuned for. The achieved rate at "
            "the decision boundary (band_hi) is bounded above by this."
        ),
    )
    band_lo: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Derived: expit(logit(value) - margin). Below this, authentic. Not a "
            "setting — the band is centred on the threshold in log-odds."
        ),
    )
    band_hi: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Derived: expit(logit(value) + margin). At or above this, "
            "manipulated. This, not value, is the decision boundary."
        ),
    )

    @classmethod
    def at_margin(
        cls, *, value: float, margin: float, target_fpr: float
    ) -> ThresholdInfo:
        """Build an operating point from a threshold and an evidence margin.

        The only intended constructor, and it is **total**: every threshold
        strictly inside (0, 1) admits every margin that keeps the edges off the
        endpoints, so there is no such thing as a (threshold, margin) pair this
        cannot represent. That is the property the old ``centred(value,
        band_width, ...)`` did not have and could not be given.

        Three refusals remain, and none of them is "these two numbers are
        jointly impossible":

        * ``value`` at 0 or 1 — a degenerate operating point with no odds.
        * a margin so wide that an edge reaches :data:`BAND_EDGE_FLOOR` — the
          region past that edge would be unreachable.
        * a margin so narrow that the edges are not distinguishable from the
          threshold in float64 — below the resolution of the arithmetic, which
          is a different failure from being outside the domain, and is reported
          as such.
        """
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"the threshold must lie strictly inside (0, 1), got {value}. A "
                "threshold at an endpoint decides every input the same way and "
                "has no log-odds; if a fit produced this, the detector is "
                "saturated and the fit is the thing to look at."
            )
        if not margin > 0.0 or not math.isfinite(margin):
            raise ValueError(
                f"margin must be a positive, finite number of log-odds, got "
                f"{margin}. A zero margin removes abstention, and this system's "
                "verdict is three-state by design — 'insufficient evidence' is a "
                "result, not a gap."
            )
        centre = logit(value)
        band_lo = expit(centre - margin)
        band_hi = expit(centre + margin)

        if band_lo < BAND_EDGE_FLOOR or band_hi > 1.0 - BAND_EDGE_FLOOR:
            raise ValueError(
                f"a margin of {margin} around {value} puts a band edge within "
                f"{BAND_EDGE_FLOOR} of an endpoint ([{band_lo}, {band_hi})); the "
                "region beyond that edge would be unreachable, so this operating "
                f"point decides nothing. The widest usable margin here is "
                f"{cls.max_margin(value):.4f}."
            )
        if not band_lo < value < band_hi:
            raise ValueError(
                f"a margin of {margin} is below float64 resolution at a threshold "
                f"of {value}: the edges came back as [{band_lo}, {band_hi}), "
                "which does not bracket it. This is an arithmetic limit, not a "
                "domain error — widen the margin."
            )
        return cls(
            value=value,
            target_fpr=target_fpr,
            band_lo=band_lo,
            band_hi=band_hi,
        )

    @staticmethod
    def max_margin(value: float) -> float:
        """The widest margin :meth:`at_margin` will accept at ``value``.

        ``logit(1 - BAND_EDGE_FLOOR) - |logit(value)|``: the nearer endpoint
        binds. Reported by the failure message above, and used by the derivation
        policy to bound its sweep instead of discovering the limit by exception.
        """
        if not 0.0 < value < 1.0:
            raise ValueError(f"the threshold must lie strictly inside (0, 1), got {value}")
        return logit(1.0 - BAND_EDGE_FLOOR) - abs(logit(value))

    @property
    def margin(self) -> float:
        """The evidence margin, recovered from the edges. In log-odds.

        The mean of the two half-margins rather than either one, so the recovered
        value does not depend on which edge happens to carry the rounding. The
        validator has already established that they agree.
        """
        centre = logit(self.value)
        return ((logit(self.band_hi) - centre) + (centre - logit(self.band_lo))) / 2.0

    @property
    def odds_factor(self) -> float:
        """``exp(margin)`` — the band stated the way an operator reads it.

        "Abstain unless the odds are at least this many times past the threshold
        odds." A factor of 1.5 and a margin of 0.4055 are the same setting.
        """
        return math.exp(self.margin)

    @property
    def band_width(self) -> float:
        """``band_hi - band_lo``. Derived and published; never an input.

        Still the right number for a risk-coverage report, which is measured in
        probability because that is where the scores are. It is simply no longer
        the number anybody sets.
        """
        return self.band_hi - self.band_lo

    @model_validator(mode="after")
    def _band_is_derived_from_the_threshold(self) -> ThresholdInfo:
        """Reject any band that is not centred, in log-odds, on its threshold.

        Checked here rather than only in ``at_margin`` because a bundle arrives
        from storage and from the wire as well as from a factory, and a stored
        result carrying an inconsistent band would otherwise be read back as
        though it were coherent.

        Note this is still an *equality* check on a single derived parameter, of
        exactly the strength the probability-space version had. The Batch 1
        invariant is not weakened by Batch 7; it is restated on the scale where
        it can hold everywhere.
        """
        if not self.band_lo < self.value:
            raise ValueError(
                f"band_lo ({self.band_lo}) must be strictly below the threshold "
                f"({self.value}); the abstention band brackets the decision point"
            )
        if not self.value < self.band_hi:
            raise ValueError(
                f"band_hi ({self.band_hi}) must be strictly above the threshold "
                f"({self.value}); a band ending at the threshold leaves scores "
                "just above it decided with no abstention region, which is the "
                "drift this check exists to prevent"
            )
        if self.band_lo < BAND_EDGE_FLOOR or self.band_hi > 1.0 - BAND_EDGE_FLOOR:
            raise ValueError(
                f"a band edge sits within {BAND_EDGE_FLOOR} of an endpoint "
                f"([{self.band_lo}, {self.band_hi})); the region beyond it is "
                "unreachable, so one of the three verdicts can never be published"
            )
        centre = logit(self.value)
        above = logit(self.band_hi) - centre
        below = centre - logit(self.band_lo)
        if abs(above - below) > LOG_ODDS_BAND_TOLERANCE * max(above, below, 1.0):
            raise ValueError(
                f"the band is not centred on its threshold in log-odds: {below} "
                f"below, {above} above. band_lo and band_hi are derived from "
                "value and a margin; build one with ThresholdInfo.at_margin(). "
                "(A band symmetric in probability is not symmetric here, and "
                "that is deliberate — see the class docstring.)"
            )
        return self


class ModelInfo(Base):
    """Traceability. Half the cache key, and the answer to 'which model said this'."""

    version: str = Field(description="Release version of the whole ensemble.")
    branch_versions: dict[str, str] = Field(default_factory=dict)
    calibration_temp: float | None = Field(default=None, gt=0.0)
    git_sha: str | None = None


class Timing(Base):
    """Where the time went. Every field is wall clock, in milliseconds.

    ``queued_ms`` is time before analysis started and is therefore *outside*
    ``total_ms``; the three work phases are inside it. The remainder —
    ``total_ms`` less the phases — is fusion, calibration and bundle
    construction, which are too small to be worth their own field but must not
    be silently folded into one that means something else.
    """

    queued_ms: int = Field(ge=0, description="Submission to start of analysis. Outside total_ms.")
    preprocess_ms: int = Field(
        ge=0, description="Decode, sampling, face detection, tensor preparation."
    )
    inference_ms: int = Field(ge=0, description="Model forward passes only.")
    explain_ms: int = Field(default=0, ge=0, description="Artifact generation.")
    total_ms: int = Field(ge=0, description="Wall clock for the whole analysis call.")

    @model_validator(mode="after")
    def _phases_fit_inside_the_total(self) -> Timing:
        """The phases are parts of the total, so they cannot exceed it.

        Enforceable without tolerance because ``PhaseTimer`` floors each phase
        and ceils the total. A violation means a phase was counted twice, or
        derived by subtraction from a different clock — both of which happened
        before the timing contract was written down.
        """
        parts = self.preprocess_ms + self.inference_ms + self.explain_ms
        if parts > self.total_ms:
            raise ValueError(
                f"timing phases sum to {parts}ms but total_ms is {self.total_ms}ms; "
                "a phase has been counted twice or measured against another clock"
            )
        return self
