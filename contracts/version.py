"""Model version composition — the single authority for "which model said this".

A version string in this platform is not decoration. It is half of the result
cache key, it is half of the job idempotency key, and it is the only record of
what produced a stored verdict. Anything that can change a score and is *not* in
this string is a way to serve a stale answer under a name that looks current.

WHY THIS LIVES IN ``contracts``
-------------------------------
Both ends need it and neither may import the other. ``inference`` composes the
version because the model owns its own identity; ``backend`` reads it to build
the cache key, and must stay free of numpy, cv2 and torch — importing
``inference`` into the API process would drag all three in for the sake of one
string. So the composition rule sits where every cross-boundary string in this
platform already sits: in ``contracts``, with no dependencies beyond stdlib.

WHAT GOES IN
------------
Three fingerprints, because three things independently change a score and have
genuinely different lifecycles:

    weights       changes when a branch is retrained. Rare, deliberate, obvious.
    calibration   changes whenever the validation distribution moves. Frequent,
                  small, and stored *outside* the checkpoint by design — which
                  makes it exactly the change least likely to be noticed. A
                  temperature swap silently reinterprets every logit the model
                  produces; a cache that cannot see it serves probabilities
                  computed under the old one.
    gate          changes when the quality floors move. Raising the minimum face
                  size does not alter a single score, but it changes which
                  inputs get a score at all, turning verdicts into abstentions.

    face          the *selection segment*. It changes when the selection policy
                  changes. Introduced in Batch 12 (R17) carrying the
                  face-selection policy alone: the detector and its asset, its
                  detection parameters, the ordering rule that picks ``faces[0]``,
                  and the crop geometry the model is shown. Batch 12 measured a
                  crop margin of 1.2 against 1.3 flipping a published verdict
                  while every other segment stayed byte-identical. Batch 23 (R29)
                  broadened it to the *combined* selection identity — face
                  selection for the image path, and face selection **plus frame
                  selection and pooling** for the video path (``n_frames``,
                  ``min_usable_frames``, the top-k pool depth, the sampling and
                  pooling rules). Still one opaque fingerprint; the parser is
                  unchanged. The gate decides *whether* an input is scored; this
                  decides *which* face/frames are scored and *what pixels* the
                  model sees.

    decode        changes when the *decode environment* changes: a different
                  video backend, OpenCV build or libavcodec. Added in Batch 9.
                  It is not part of the model and is kept in its own segment for
                  exactly that reason — see below.

The format is readable on purpose. An operator reading a log line can see which
*segment* moved without decoding anything:

    stub-0+c.1f4a9c2b.g.7e30d5a1.f.4b1d9e07.d.6f2a91cc
    └────┘  └───────┘  └───────┘  └───────┘  └───────┘
    weights calibration  gate    selection   decode

The ``.f.`` **selection** segment aggregates every selection policy: face
selection for the image path, and face selection plus frame selection and
pooling for the video path (Batch 23, R29). When ``.f.`` moves, *which* policy
changed is recovered from ``FaceSelectionPolicy.differences()`` /
``VideoSelectionPolicy.differences()`` or from ``verify_version_alignment``'s
per-engine detail — not from the log line alone. This is the one place the
"without decoding anything" property is deliberately traded away, to keep
``.f.`` a single opaque fingerprint and the parser (``parse_model_version``,
``_MODEL_MARKERS``) unchanged. The alternative — a separate ``.v.`` segment —
was rejected because it adds a permanent parser era for a legibility gain that
``differences()`` already provides off the log line.

WHY ``face`` IS ITS OWN SEGMENT AND NOT PART OF ``gate``
--------------------------------------------------------
``MediaCaps.gate_fingerprint`` already covers ``min_face_px``, so folding the
selection policy in there was the smaller diff. It was rejected for the reason
Batch 9 rejected folding the decoder into the calibration fingerprint: it would
make "we raised the face floor" and "we replaced the detector" the same event.
They have different owners and different lifecycles — the caps are deployment
configuration passed into a pipeline, the selection policy is shipped code plus
an OpenCV asset — and the SCRFD migration moves one without touching the other.
An operator diffing two versions has to be able to see which.

TWO IDENTITIES, NOT ONE
-----------------------
Batch 9 splits what used to be a single string into two, because two different
questions were being answered by one answer:

    model identity    weights + calibration + gate.  "Which model is this?"
    runtime identity  model identity + decode.       "Would this deployment
                                                      produce the same number?"

Only the second is a valid cache key. Batch 8 measured why: two video backends
in one OpenCV build decode the same file to *zero* identical frames, shifting
``p_fake`` enough to change 1.3% of decisions — while every field of the model
identity stayed byte-identical. A cache keyed on model identity alone can
therefore return a result produced under a different decoder, which it did.

Only the *first* belongs in a benchmark comparison that asks "did the model get
worse", which is why the two are separate strings rather than one longer one.
``model_identity_of`` recovers the first from the second, so nothing that needs
the narrower question has to reconstruct it.

WHY THIS CAN LIVE IN ``contracts`` WHEN THE DECODER CANNOT
----------------------------------------------------------
Composing the runtime version is string concatenation and belongs here with
every other cross-boundary string. *Computing* the decode fingerprint needs
``cv2.getBuildInformation``, which the API process must never import.

Batch 8 concluded from that that the decoder could not reach a key the API
builds. **That was wrong, and the mechanism that disproves it was already in the
codebase**: ``Settings.model_version`` is not computed by the API either. It is a
configured literal that ``workers.engine_registry.verify_version_alignment``
checks against what the engines actually report, failing startup on a mismatch.
The decode fingerprint reaches the API by exactly that route.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

FINGERPRINT_LENGTH = 8
"""Eight hex characters — 32 bits.

Long enough that an accidental collision between the handful of calibrations a
deployment ever holds is not a practical concern, short enough that the composed
version stays readable in a log line. This is a change *detector*, not a
security boundary: it answers "is this the same calibration as before", and
nothing here defends against someone constructing a collision on purpose.
"""

CALIBRATION_MARKER = "+c."
GATE_MARKER = ".g."
FACE_SELECTION_MARKER = ".f."
"""Separator for the selection segment. Last *within the model identity*.

Named ``.f.`` for the face-selection policy it carried when introduced (Batch 12,
R17); since Batch 23 (R29) it carries the combined selection identity — face
selection, plus frame selection and pooling for video. Still one opaque
fingerprint; the parser is unchanged. The letter is kept: ``.s.`` was rejected
because ``selection`` is already taken in this codebase by *dataset* selection
(``CalibrationWitness.selection_witness`` names the corpus a temperature was
fitted on), and two things called "selection" in one identity vocabulary is how
an auditor reads the wrong one.

Placed after :data:`GATE_MARKER` so that a version written before Batch 12
parses with ``face_selection=None``: the same backward-compatible shape the
decode segment has, and for the same reason.
"""

DECODE_MARKER = ".d."
"""Separator for the decode-environment segment. Always last.

Last so that :func:`model_identity_of` is a single ``partition`` — the model
identity is a *prefix* of the runtime identity, which makes "strip the
environment" a substring operation rather than a reassembly, and makes a runtime
version readable as a model version with something appended.
"""

_MODEL_MARKERS = (CALIBRATION_MARKER, GATE_MARKER, FACE_SELECTION_MARKER, DECODE_MARKER)

FINGERPRINT_DECIMALS = 6
"""Decimal places a fitted quantity is rounded to before it is fingerprinted.

Every quantity this is applied to — a temperature, a threshold, a band edge, a
false-positive budget — lives in a small bounded range, so absolute decimal
places are the right unit and no relative-precision machinery is needed.

The number is chosen from both ends. A difference of 1e-6 in any of them moves
``p_fake`` by well under 1e-6, which is four orders of magnitude below anything
that changes a verdict, so rounding here cannot merge two calibrations that
would decide differently. And 1e-6 is comfortably *coarser* than the ~1e-8
float noise that motivated it, so the noise is rounded away rather than
preserved at higher resolution.
"""


def quantize(value: float, decimals: int = FINGERPRINT_DECIMALS) -> float:
    """Round a fitted float to fingerprint resolution.

    WHY THIS IS NOT ``_canonical``'S JOB
    ------------------------------------
    ``_canonical`` deliberately preserves floats exactly, via ``repr``, and a
    test pins that: ``1.0`` and ``1.0000000000000002`` are different
    calibrations and must fingerprint differently. That rule is right for the
    general case — the fingerprint is a change *detector*, and a detector that
    rounds is a detector with a blind spot — and it applies to every fingerprint
    in the platform, including ones over exact values like branch weights and
    media ids, where rounding would be pure loss.

    So the rounding happens here, at the call site, on the specific quantities
    that are *fitted* rather than configured. Those are the only ones that carry
    float noise, and they are the only ones where two bit-different values are
    known to mean the same thing.

    WHAT THIS DOES AND DOES NOT GUARANTEE
    -------------------------------------
    It does **not** make a fingerprint stable on its own, and treating it as
    though it does is the trap. Two values astride a rounding boundary —
    ``0.6199999996`` and ``0.6200000004`` — still round apart. Quantisation
    narrows the window in which noise matters; it does not close it.

    The actual stability guarantee comes from upstream: ``fit_temperature``
    returns a bit-identical result for any ordering of the same calibration set,
    because ``negative_log_likelihood`` sums with ``math.fsum``. This function is
    defence in depth for the noise sources that fix does not cover — a different
    numpy build, float32 inputs, a re-derived threshold — where being *nearly*
    always stable is a real improvement over never being stable.
    """
    if value != value or value in (float("inf"), float("-inf")):
        # NaN would canonicalise to the string "nan" and silently unify every
        # failed fit into one fingerprint; an infinity would do the same.
        raise ValueError(
            f"cannot fingerprint a non-finite value ({value!r}); a fit that "
            "produced one has failed and must not be published as a version"
        )
    rounded = round(float(value), decimals)
    # -0.0 and 0.0 are the same number and must not be two fingerprints;
    # ``repr`` renders them differently.
    return 0.0 if rounded == 0.0 else rounded


def _canonical(value: Any) -> Any:
    """Reduce a value to something JSON can serialise deterministically.

    Floats go through ``repr`` rather than being left to the JSON encoder: a
    temperature of ``1.0000000000000002`` and one of ``1.0`` are different
    calibrations and must fingerprint differently, and round-tripping through a
    shortened float representation is exactly how that distinction gets lost.
    """
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def fingerprint(parts: Mapping[str, Any]) -> str:
    """Stable short hash of a named set of values.

    Stable across processes and runs — no ``hash()``, which is randomised per
    interpreter by PYTHONHASHSEED and would give a different cache key on every
    restart. Keys are sorted so that the caller's dict ordering cannot change
    the result.
    """
    canonical = json.dumps(
        _canonical(dict(parts)), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


@dataclass(frozen=True)
class ModelVersionParts:
    """A composed version, taken back apart.

    ``face_selection`` carries the combined selection fingerprint — face
    selection, plus frame selection and pooling for video (Batch 23, R29). It is
    ``None`` for a version written before Batch 12, which means the version makes
    no claim about which face/frames were chosen or how they were cropped.
    :func:`is_selection_pinned` (or its alias :func:`is_face_selection_pinned`)
    is how a caller asks.
    """

    weights: str
    calibration: str
    gate: str | None = None
    face_selection: str | None = None

    def compose(self) -> str:
        return compose_model_version(
            weights=self.weights,
            calibration=self.calibration,
            gate=self.gate,
            face_selection=self.face_selection,
        )


def compose_model_version(
    *,
    weights: str,
    calibration: str,
    gate: str | None = None,
    face_selection: str | None = None,
) -> str:
    """Build the effective version. This is the only function that may.

    ``weights`` is the branch-weights release identity; ``calibration``, ``gate``
    and ``face_selection`` are fingerprints from :func:`fingerprint`. The
    ``face_selection`` keyword keeps its name for compatibility; since Batch 23
    (R29) the value it carries is the *combined* selection fingerprint (face
    selection, plus frame selection and pooling for video).

    ``face_selection=None`` returns a version with no such segment rather than
    inventing a placeholder — same rule as ``compose_runtime_version``, and for
    the same reason: a deployment that makes no claim about its selection policy
    must stay distinguishable from one pinned to some particular policy.
    """
    if not weights:
        raise ValueError("weights version must not be empty")
    if DECODE_MARKER in weights:
        raise ValueError(
            f"weights version {weights!r} must not contain {DECODE_MARKER!r}; "
            "that segment names the decode environment and would be parsed back "
            "as one"
        )
    if FACE_SELECTION_MARKER in weights:
        raise ValueError(
            f"weights version {weights!r} must not contain "
            f"{FACE_SELECTION_MARKER!r}; that segment names the selection policy "
            "(the combined selection fingerprint since Batch 23) and would be "
            "parsed back as one"
        )
    if CALIBRATION_MARKER in weights or GATE_MARKER in weights:
        # A weights identity containing a marker would parse back into the wrong
        # components, and the failure would look like a spurious cache miss
        # rather than like a malformed version.
        raise ValueError(
            f"weights version {weights!r} must not contain {CALIBRATION_MARKER!r} "
            f"or {GATE_MARKER!r}"
        )
    composed = f"{weights}{CALIBRATION_MARKER}{calibration}"
    if gate is not None:
        composed = f"{composed}{GATE_MARKER}{gate}"
    if face_selection is not None:
        if not face_selection or any(m in face_selection for m in _MODEL_MARKERS):
            raise ValueError(
                f"invalid selection fingerprint {face_selection!r}: it must "
                "be non-empty and must not contain a segment marker"
            )
        composed = f"{composed}{FACE_SELECTION_MARKER}{face_selection}"
    return composed


@dataclass(frozen=True)
class RuntimeVersionParts:
    """A runtime identity, taken apart into its two halves.

    ``decode`` is ``None`` for a model identity that carries no environment
    segment. That is not an error and must not be treated as one: a
    configuration written before Batch 9, or a deployment that deliberately
    declines to key its cache on the decoder, is representable and readable —
    it simply makes no claim about the decode environment. ``is_environment_
    pinned`` is how a caller asks whether a claim was made.
    """

    model: str
    decode: str | None = None

    @property
    def is_environment_pinned(self) -> bool:
        return self.decode is not None

    def compose(self) -> str:
        return compose_runtime_version(model=self.model, decode=self.decode)


def compose_runtime_version(*, model: str, decode: str | None) -> str:
    """Append a decode-environment segment to a model identity.

    ``decode=None`` returns the model identity unchanged rather than inventing a
    placeholder. An unpinned deployment must be *distinguishable* from one
    pinned to some particular environment, and a sentinel like ``"none"`` would
    make them the same string.
    """
    if not model:
        raise ValueError("model version must not be empty")
    if DECODE_MARKER in model:
        raise ValueError(
            f"{model!r} already carries a decode segment; composing again would "
            "produce a version with two, and the second would win on parse"
        )
    if decode is None:
        return model
    if not decode or DECODE_MARKER in decode:
        raise ValueError(f"invalid decode fingerprint {decode!r}")
    return f"{model}{DECODE_MARKER}{decode}"


def parse_runtime_version(version: str) -> RuntimeVersionParts:
    """Split a runtime identity into model and decode halves.

    Total on any non-empty string: a version with no decode segment parses to
    ``decode=None``. That is deliberate and is what makes this readable against
    versions written before Batch 9 without a migration.
    """
    if not version:
        raise ValueError("version must not be empty")
    model, marker, decode = version.partition(DECODE_MARKER)
    if not marker:
        return RuntimeVersionParts(model=version, decode=None)
    if not decode:
        raise ValueError(
            f"{version!r} ends in {DECODE_MARKER!r} with no fingerprint after it"
        )
    return RuntimeVersionParts(model=model, decode=decode)


def model_identity_of(version: str) -> str:
    """The model half of a runtime identity. Idempotent on a model identity.

    The function a benchmark comparison uses: "did the *model* change" must not
    be answered yes because a worker was rebuilt with a different libavcodec.
    """
    return parse_runtime_version(version).model


def decode_identity_of(version: str) -> str | None:
    """The decode half, or ``None`` if the version makes no environment claim."""
    return parse_runtime_version(version).decode


def is_environment_pinned(version: str) -> bool:
    """Whether ``version`` names a decode environment.

    The check a deployment runs against its own configuration, and the exact
    analogue of :func:`is_composed`: a version without this segment means the
    result cache cannot tell two decoders apart, which Batch 8 measured to be a
    real source of wrong answers rather than a theoretical one.
    """
    return DECODE_MARKER in version


def parse_model_version(version: str) -> ModelVersionParts:
    """Take a composed version apart. Raises ``ValueError`` on anything else.

    Refuses to guess at an unrecognised string. A version that does not parse
    means something wrote one by hand, and silently treating it as a bare
    weights identity would let an uncalibrated deployment masquerade as a
    calibrated one.
    """
    # A runtime identity is a model identity with a segment appended, so it
    # parses here too rather than being refused: every caller of this function
    # wants the model components, and making each one remember to strip the
    # environment first is how a version string acquires two parsers that
    # disagree.
    version = model_identity_of(version)
    if CALIBRATION_MARKER not in version:
        raise ValueError(
            f"{version!r} is not a composed model version "
            f"(no {CALIBRATION_MARKER!r} segment)"
        )
    weights, _, rest = version.partition(CALIBRATION_MARKER)
    # Face selection is stripped first because it is the *last* segment of the
    # model identity, exactly as the decode segment is stripped before this
    # function runs. Peeling from the right keeps each partition unambiguous.
    face_selection: str | None = None
    if FACE_SELECTION_MARKER in rest:
        rest, _, face_selection = rest.partition(FACE_SELECTION_MARKER)
        if not face_selection:
            raise ValueError(
                f"{version!r} ends in {FACE_SELECTION_MARKER!r} with no "
                "fingerprint after it"
            )
    if GATE_MARKER in rest:
        calibration, _, gate = rest.partition(GATE_MARKER)
        return ModelVersionParts(
            weights=weights,
            calibration=calibration,
            gate=gate,
            face_selection=face_selection,
        )
    return ModelVersionParts(
        weights=weights, calibration=rest, gate=None, face_selection=face_selection
    )


def selection_identity_of(version: str) -> str | None:
    """The selection-policy fingerprint carried in the ``.f.`` segment, or
    ``None`` if the version claims none.

    Since Batch 23 (R29) the ``.f.`` segment names the *combined* selection
    policy: face selection for the image path, and face selection plus frame
    selection and pooling for the video path. This is the honest name for what
    ``.f.`` carries. Total on any composed version, pre- or post-Batch-23.

    For a version written before Batch 23 the value genuinely *is* the
    face-selection identity; :func:`face_selection_identity_of` is the historical
    alias and returns the same value.
    """
    return parse_model_version(version).face_selection


def is_selection_pinned(version: str) -> bool:
    """Whether ``version`` names a selection policy (face, or face + frame
    selection and pooling for video).

    The exact analogue of :func:`is_environment_pinned` and :func:`is_composed`.
    A version without this segment means the result cache cannot tell two
    selection policies apart — which Batch 12 measured to flip a published
    verdict, not merely to be untidy.
    """
    return FACE_SELECTION_MARKER in version


def face_selection_identity_of(version: str) -> str | None:
    """SINCE BATCH 23 (R29) THIS RETURNS THE COMBINED SELECTION IDENTITY (face
    selection + video frame selection + pooling), NOT a
    ``FaceSelectionPolicy.identity``.

    For a Batch-23+ version, compare the result to
    ``selection_identity(default_face_selection(), default_video_selection())``,
    NEVER to ``default_face_selection().identity``. For a pre-Batch-23 version
    the value is the face-selection identity and the two comparisons coincide.
    New code should call :func:`selection_identity_of`.

    Kept as a permanent alias — this name is correct for every version written
    before Batch 23 and returns the right value under a broadened name after it.
    """
    return selection_identity_of(version)


def is_face_selection_pinned(version: str) -> bool:
    """Alias of :func:`is_selection_pinned`. Named for the pre-Batch-23 era when
    ``.f.`` carried only the face policy; both names test for the presence of the
    ``.f.`` segment."""
    return is_selection_pinned(version)


def is_composed(version: str) -> bool:
    """Whether ``version`` carries calibration identity.

    The check a deployment runs against its own configuration. A bare weights
    string here means the cache cannot tell two calibrations apart.
    """
    return CALIBRATION_MARKER in version
