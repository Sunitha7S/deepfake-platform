"""Calibration that can prove which corpus it was fitted on.

THE GAP THIS CLOSES
-------------------
Batch 5 bound *benchmark* runs to their protocol: a run records a protocol only
when its items came through ``protocol.select()``. Calibration was left open,
and it is the more dangerous half.

``fit_temperature(logits, labels)`` and ``at_target_fpr(y, s)`` take bare arrays.
Where those arrays came from is not an argument, is not recorded, and cannot be
recovered afterwards. The only account of provenance was
``CalibrationSpec.fitted_on`` — a free-text string, written by the same caller,
never checked against anything. So a temperature fitted on Celeb-DF and a
threshold chosen on Celeb-DF could be stamped ``fitted_on="ffpp-test"`` and
carried into a Celeb-DF benchmark report that passed every check in the system.

That is worse than the benchmark case because the damage is invisible in the
numbers. A threshold chosen on the corpus it is then measured on reports the
best false-positive rate that corpus can produce; it is not a prediction of
anything, and it looks like a very good result.

WHAT A WITNESS IS
-----------------
The same shape as ``RoleSelection``'s: a record of what was done, plus a digest
binding the parts together so the record cannot be edited piecemeal. It names
the protocol, the role, the corpus, the sample count and every fitted output,
and it carries the selection witness of the corpus it consumed — so a
calibration witness transitively attests to the split lock as well.

THE ROLE CHECK IS THE STRUCTURAL PART
-------------------------------------
:func:`fit_calibration` accepts a ``RoleSelection`` and **refuses any role but
CALIBRATION**. Fitting on a benchmark selection is not a warning, not a flag on
the output — it raises. That is what makes "benchmark data masquerading as
calibration data" unrepresentable rather than merely detectable, and it is the
one guarantee here that does not depend on anybody running a verifier.

THE TIMESTAMP IS RECORDED BUT NOT DIGESTED
------------------------------------------
``created_at`` is real provenance and worth having. It is deliberately excluded
from the witness digest, for the same reason the split lock carries no timestamp
at all: a digest over the clock makes "identical inputs produce identical
witnesses" unwritable as a test, and that property is the cheapest
reproducibility check there is — refit, compare, expect silence. So the digest
covers the reproducible content and the timestamp sits beside it.

FITTED FLOATS ARE QUANTISED FIRST
---------------------------------
Every float entering the digest goes through ``contracts.version.quantize``,
exactly as ``calibration_fingerprint`` does. Without it the witness inherits the
R1 defect one level up: two runs of the same fit would produce two witnesses,
and a verifier would report tampering where there was float noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from contracts.common import ThresholdInfo
from contracts.version import fingerprint, quantize
from evaluation.benchmark import dataset_fingerprint
from evaluation.protocol import DatasetRole, ProtocolViolation, RoleSelection
from evaluation.thresholds import ThresholdChoice, at_target_fpr
from inference.calibration import (
    fit_temperature,
    negative_log_likelihood,
    temperature_scale,
)

CALIBRATION_WITNESS_VERSION = "1.3.0"
"""Schema of a serialised witness, on its own version axis.

Independent of ``CONTRACT_VERSION`` (frozen, and this is not a serving
contract), of the lock format, and of ``RUN_SCHEMA_VERSION``. Four artefacts
with four lifecycles; coupling them would mean bumping a public contract to add
a diagnostic field.

**1.0.0 -> 1.1.0 (Batch 7).** ``band_width`` became ``margin``: the band's
parameter is now a log-odds half-width rather than a probability width, so the
recorded number means something different and keeping the name would have made
two incompatible witnesses indistinguishable. A 1.0.0 payload is refused by
:meth:`CalibrationWitness.from_dict` with a message naming the change rather
than the generic unknown-field error, and it is refused rather than migrated
because the old number cannot be converted without the threshold's *fitted*
value and a decision about which side to preserve — a migration that silently
picks one would be inventing an operating point.

**1.1.0 -> 1.2.0 (Batch 11).** Adds ``decode_identity``, **inside the digest**.
It has to be inside: a provenance field recorded beside the digest could be
edited without invalidating anything, and ``created_at`` is outside for a reason
that does not apply here — a timestamp is not an input to the fitted numbers,
and the decode environment is.

Unlike 1.0.0, a 1.1.0 payload **still loads and still verifies**. The digest is
computed against the payload's *own* schema version (see
:meth:`CalibrationWitness.compute_witness`), so an old witness is checked under
the old rule rather than being grandfathered past the check. Without that,
bumping this constant alone would have recomputed every stored digest and made
every stored witness unreadable — measured, and with an error message accusing
the operator of hand-editing a file that was fine.

**1.2.0 -> 1.3.0 (R31).** Adds ``pre_calibration_identity``, **inside the
digest**, same reasoning as ``decode_identity``. Closes R31: a temperature
fitted for one model (weights + gate + selection) applied to another with
byte-identical or ``quantize``-converged fitted numbers produced an
indistinguishable witness, because nothing about the model's own
configuration — as opposed to the corpus and decode environment already
covered — reached the digest. ``ModelSpec.pre_calibration_identity``
(``inference/models.py``) names *weights + gate + selection only*, computed
before calibration is fitted so there is no dependency cycle with
:attr:`~inference.models.ModelSpec.calibration_fingerprint` (which
deliberately excludes the witness). A 1.2.0 payload still loads and still
verifies, by the identical schema-version-as-parameter mechanism the 1.1.0 ->
1.2.0 bump established. See ``docs/R31_DESIGN.md`` and
``docs/R31_SPEC_RESOLUTION.md``.
"""

SCHEMA_VERSIONS_WITH_DECODE: frozenset[str] = frozenset({"1.2.0", "1.3.0"})
"""Schema versions whose digest covers ``decode_identity``.

An explicit set rather than a version comparison. Adding 1.3.0 should be a
deliberate line here — a future schema that *removes* the field would silently
start covering it again under a ``>=`` test, and the digest rule is the last
place to want a clever predicate. 1.3.0 (R31) still covers ``decode_identity``
— it adds ``pre_calibration_identity`` alongside it, not instead of it.
"""

SCHEMA_VERSIONS_WITH_PRE_CALIBRATION: frozenset[str] = frozenset({"1.3.0"})
"""Schema versions whose digest covers ``pre_calibration_identity``.

Same rationale as :data:`SCHEMA_VERSIONS_WITH_DECODE`, on its own axis: 1.2.0
witnesses predate this field and must not silently start being checked against
it. See R31 (``docs/R31_DESIGN.md``).
"""

CALIBRATION_ROLE = DatasetRole.CALIBRATION
"""The only role a calibration may be fitted on. See the module docstring."""

MIN_WITNESSED_SAMPLES = 200
"""Below this a fit is noise, and a witness would attest to noise.

The same floor ``inference.calibration.fit_and_validate`` applies, enforced here
as a refusal rather than an ``accepted=False`` flag: that function may return an
unaccepted fit because it is a measurement tool, but a *witness* is a claim of
compliance and there is no such thing as a compliant fit on 40 samples.
"""


@dataclass(frozen=True)
class CalibrationWitness:
    """Evidence that a calibration was fitted on the corpus it claims.

    Every field except ``created_at`` and ``schema_version`` is inside the
    digest. Constructing one with a digest that does not match its own fields
    raises, so a witness cannot be edited in place — the realistic failure,
    where somebody adjusts a threshold in a JSON file and leaves the rest.
    """

    protocol_name: str
    role: str
    corpus_fingerprint: str
    n_samples: int
    temperature: float
    nll_before: float
    nll_after: float
    threshold: float | None
    margin: float | None
    target_fpr: float | None
    selection_witness: str | None
    split_lock_hash: str | None
    witness: str
    created_at: str | None = None
    """ISO-8601 UTC. Provenance, deliberately outside the digest."""
    schema_version: str = CALIBRATION_WITNESS_VERSION
    decode_identity: str | None = None
    """The decode environment the fitted logits were produced under. **Batch 11.**

    ``inference.decode_env.decode_identity()``, from the process that ran the
    model — passed to :func:`fit_calibration`, never detected here, for the same
    reason ``BenchmarkRun.decode_identity`` is passed: this process may not be
    the one that produced the logits.

    **What it closes.** A temperature fitted under one decoder and applied under
    another is silently miscalibrated: ``T`` was fitted to map FFMPEG-decoded
    logits to probabilities and is scaling MSMF-decoded ones. Batch 9 stopped the
    result *cache* crossing decoders; nothing stopped the calibration.

    **Why the digest needed it even though fitted values usually differ.** A
    decoder change moves the logits, so it moves the temperature, so it moves the
    digest — detection mostly worked already, by accident of what the digest
    covers. The blind spot is convergence: two fits landing on the same values
    after ``quantize`` to 6 decimals produce byte-identical witnesses for two
    environments, which was measured. Provenance that works except when the
    numbers agree is not provenance.

    ``None`` means *not recorded* — every witness written before 1.2.0, and any
    fit whose caller declined to name its environment.
    """
    pre_calibration_identity: str | None = None
    """The model configuration the fitted logits were produced by, **excluding
    calibration itself**. **R31.**

    ``ModelSpec.pre_calibration_identity`` (``inference/models.py``) — weights
    + gate + selection, composed the same way :attr:`~inference.models.
    ModelSpec.effective_version` is, minus the calibration segment. Passed to
    :func:`fit_calibration`, never detected here, for the identical reason
    ``decode_identity`` is passed: this function is handed arrays and has no
    way to know which ``ModelSpec`` produced them.

    **What it closes.** Nothing about the model's own weights, quality gate or
    face/frame-selection policy previously reached this digest at all — not
    even incidentally, the way ``input_size`` moved the fitted temperature
    before Batch 22 folded it into ``weights_identity``. Two models differing
    in gate or selection but scoring a fixed calibration corpus identically
    (the common case, not a contrived one — see ``docs/R31_DESIGN.md`` §2)
    produced byte-identical witnesses.

    **No circularity.** ``ModelSpec.calibration_fingerprint`` deliberately
    excludes ``CalibrationSpec.witness`` (``inference/models.py``), so this
    field is computed *before* calibration exists and calibration's own
    contribution to ``effective_version`` never reads it back.

    ``None`` means *not recorded* — every witness written before 1.3.0, and
    any fit whose caller declined to name its model configuration.
    """

    def __post_init__(self) -> None:
        expected = self.compute_witness(
            protocol_name=self.protocol_name,
            role=self.role,
            corpus_fingerprint=self.corpus_fingerprint,
            n_samples=self.n_samples,
            temperature=self.temperature,
            nll_before=self.nll_before,
            nll_after=self.nll_after,
            threshold=self.threshold,
            margin=self.margin,
            target_fpr=self.target_fpr,
            selection_witness=self.selection_witness,
            split_lock_hash=self.split_lock_hash,
            decode_identity=self.decode_identity,
            pre_calibration_identity=self.pre_calibration_identity,
            schema_version=self.schema_version,
        )
        if self.witness != expected:
            raise ProtocolViolation(
                f"calibration witness {self.witness!r} does not match its own "
                f"fields (expected {expected!r}); this witness was assembled or "
                "edited by hand rather than produced by fit_calibration()"
            )
        if self.role != CALIBRATION_ROLE.value:
            raise ProtocolViolation(
                f"a calibration witness claims role {self.role!r}; only "
                f"{CALIBRATION_ROLE.value!r} may be fitted on. A witness naming "
                "a benchmark role is precisely the laundering this type exists "
                "to prevent."
            )

    @staticmethod
    def compute_witness(
        *,
        protocol_name: str,
        role: str,
        corpus_fingerprint: str,
        n_samples: int,
        temperature: float,
        nll_before: float,
        nll_after: float,
        threshold: float | None,
        margin: float | None,
        target_fpr: float | None,
        selection_witness: str | None,
        split_lock_hash: str | None,
        decode_identity: str | None = None,
        pre_calibration_identity: str | None = None,
        schema_version: str = CALIBRATION_WITNESS_VERSION,
    ) -> str:
        """The binding digest. Floats quantised; ``created_at`` excluded.

        **``schema_version`` is a parameter, and that is the migration.** The
        digest has always covered the schema version, so changing the constant
        changes the digest of *unchanged fields* — which meant that bumping to
        1.2.0 would recompute every stored witness's digest, find it did not
        match the recorded one, and raise ``ProtocolViolation`` telling the
        operator their file had been hand-edited. Measured: ``092895e2`` under
        1.1.0 becomes ``54adb0e2`` under 1.2.0 for identical fields.

        Computing against the payload's *own* version instead means a 1.1.0
        witness is verified under the 1.1.0 rule — no ``decode`` key, original
        digest, still valid — and a 1.2.0 witness under the 1.2.0 rule. Old
        artifacts stay **verifiable**, not merely tolerated: this is a migration
        rather than an amnesty.

        ``decode_identity`` enters the digest only for schemas in
        :data:`SCHEMA_VERSIONS_WITH_DECODE`. Adding the key unconditionally with
        a ``None`` value would still change every 1.1.0 digest, which is the
        whole failure this avoids.

        ``pre_calibration_identity`` enters the digest only for schemas in
        :data:`SCHEMA_VERSIONS_WITH_PRE_CALIBRATION`, for the identical reason
        (R31). It is a separate frozenset from ``SCHEMA_VERSIONS_WITH_DECODE``
        on purpose — the two axes were added in different batches and a
        future schema could in principle carry one without the other.
        """
        parts = {
            "protocol": protocol_name,
            "role": role,
            "corpus_fingerprint": corpus_fingerprint,
            "n_samples": n_samples,
            "temperature": quantize(temperature),
            "nll_before": quantize(nll_before),
            "nll_after": quantize(nll_after),
            "threshold": None if threshold is None else quantize(threshold),
            "margin": None if margin is None else quantize(margin),
            "target_fpr": None if target_fpr is None else quantize(target_fpr),
            "selection_witness": selection_witness,
            "split_lock_hash": split_lock_hash,
            "schema": schema_version,
        }
        if schema_version in SCHEMA_VERSIONS_WITH_DECODE:
            parts["decode_identity"] = decode_identity
        elif decode_identity is not None:
            # A pre-1.2.0 schema cannot carry this and must not silently drop it:
            # the resulting witness would attest to an environment it does not
            # record, which is the exact species of false claim this field exists
            # to remove.
            raise ProtocolViolation(
                f"schema {schema_version} has no place for a decode identity, but "
                f"{decode_identity!r} was supplied. A witness that records an "
                "environment outside its own digest is not evidence of it."
            )
        if schema_version in SCHEMA_VERSIONS_WITH_PRE_CALIBRATION:
            parts["pre_calibration_identity"] = pre_calibration_identity
        elif pre_calibration_identity is not None:
            # R31, same reasoning as the decode_identity guard above: a
            # pre-1.3.0 schema has no digest slot for this, and silently
            # dropping it would produce a witness that attests to a model
            # configuration it does not record.
            raise ProtocolViolation(
                f"schema {schema_version} has no place for a pre-calibration "
                f"identity, but {pre_calibration_identity!r} was supplied. A "
                "witness that records a model configuration outside its own "
                "digest is not evidence of it."
            )
        return fingerprint(parts)

    # -- derived artefacts -------------------------------------------------

    def threshold_info(self) -> ThresholdInfo:
        """Rebuild the published operating point from the witness.

        Through ``ThresholdInfo.centred``, never by setting the band edges, so a
        witness cannot reconstitute a band that the contract would refuse. The
        Batch 1 invariant holds on this path like every other.
        """
        if self.threshold is None or self.margin is None:
            raise ValueError(
                "this witness records no threshold; it attests to a temperature "
                "fit only"
            )
        return ThresholdInfo.at_margin(
            value=self.threshold,
            margin=self.margin,
            target_fpr=self.target_fpr if self.target_fpr is not None else 0.02,
        )

    def calibration_spec(self):
        """A :class:`~inference.models.CalibrationSpec` carrying this witness.

        The bridge into the model identity chain. The spec built here is
        distinguishable from a hand-built one by ``is_protocol_compliant``.
        """
        from inference.models import CalibrationSpec

        return CalibrationSpec(
            temperature=self.temperature,
            fitted_on=f"{self.protocol_name}/{self.role}",
            n_samples=self.n_samples,
            nll_before=self.nll_before,
            nll_after=self.nll_after,
            witness=self.witness,
        )

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "protocol_name": self.protocol_name,
            "role": self.role,
            "corpus_fingerprint": self.corpus_fingerprint,
            "n_samples": self.n_samples,
            "temperature": self.temperature,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "threshold": self.threshold,
            "margin": self.margin,
            "target_fpr": self.target_fpr,
            "selection_witness": self.selection_witness,
            "split_lock_hash": self.split_lock_hash,
            "witness": self.witness,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "decode_identity": self.decode_identity,
            "pre_calibration_identity": self.pre_calibration_identity,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> CalibrationWitness:
        """Rebuild, re-verifying the digest on the way in.

        Unknown fields are refused rather than ignored: a payload from a newer
        schema may carry something that changes the meaning of what is here, and
        a witness read under a guess is not evidence.
        """
        known = set(cls.__dataclass_fields__)
        if "band_width" in payload and "margin" not in payload:
            raise ValueError(
                "this is a 1.0.0 calibration witness: it records 'band_width', a "
                "probability width, where "
                f"{CALIBRATION_WITNESS_VERSION} records 'margin', a log-odds "
                "half-width. The two are not interconvertible without deciding "
                "which edge of the old band to preserve, and picking one would "
                "invent an operating point that was never calibrated. Refit."
            )
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(
                f"unknown calibration witness field(s) {unknown}; a witness from "
                f"a newer schema than {CALIBRATION_WITNESS_VERSION} cannot be "
                "read by this build"
            )
        # ``decode_identity`` and ``pre_calibration_identity`` join the optional
        # set: a 1.1.0 payload does not carry either and must still load. Their
        # absence becomes None, which reads as "not recorded" and never as "no
        # environment" / "no model configuration".
        optional = {
            "created_at",
            "schema_version",
            "decode_identity",
            "pre_calibration_identity",
        }
        missing = sorted(known - optional - set(payload))
        if missing:
            raise ValueError(f"calibration witness payload is missing {missing}")
        if (
            payload.get("decode_identity") is not None
            and payload.get("schema_version", "1.1.0") not in SCHEMA_VERSIONS_WITH_DECODE
        ):
            raise ValueError(
                f"this payload records decode_identity "
                f"{payload['decode_identity']!r} but declares schema "
                f"{payload.get('schema_version')!r}, which does not cover it in "
                "the digest. The field would be unprotected, so the witness does "
                "not attest to the environment it names."
            )
        if (
            payload.get("pre_calibration_identity") is not None
            and payload.get("schema_version", "1.1.0")
            not in SCHEMA_VERSIONS_WITH_PRE_CALIBRATION
        ):
            raise ValueError(
                f"this payload records pre_calibration_identity "
                f"{payload['pre_calibration_identity']!r} but declares schema "
                f"{payload.get('schema_version')!r}, which does not cover it in "
                "the digest. The field would be unprotected, so the witness does "
                "not attest to the model configuration it names."
            )
        return cls(**payload)

    def describe(self) -> str:
        lines = [
            f"calibration under {self.protocol_name!r} / {self.role}",
            f"  corpus     {self.corpus_fingerprint}  n={self.n_samples}",
            f"  temperature {self.temperature:.6f}  "
            f"NLL {self.nll_before:.5f} -> {self.nll_after:.5f}",
        ]
        if self.threshold is not None:
            lines.append(
                f"  threshold  {self.threshold:.6f}  margin {self.margin:.4f} "
                f"(odds x{math.exp(self.margin):.3f})  target FPR {self.target_fpr}"
            )
        lines.append(
            f"  selection  {self.selection_witness or '(none)'}  lock "
            f"{self.split_lock_hash or '(none)'}"
        )
        lines.append(f"  decode     {self.decode_identity or '(unrecorded)'}")
        lines.append(
            f"  model cfg  {self.pre_calibration_identity or '(unrecorded)'}"
        )
        lines.append(f"  witness    {self.witness}  at {self.created_at or '(unstamped)'}")
        return "\n".join(lines)


DERIVE = "derive"
"""Sentinel for ``margin=``: derive the band from these scores, do not take one.

The recommended setting, and the one the design report argues for. Passing a
number is still allowed — reproducing a previous operating point exactly is a
real need — but it is now the exception rather than the only option.
"""


@dataclass(frozen=True)
class CalibrationResult:
    """What a compliant fit produces: the numbers, and the evidence for them."""

    witness: CalibrationWitness
    threshold_choice: ThresholdChoice | None
    derivation: object | None = None
    """The :class:`~evaluation.band.OperatingPointDerivation`, when ``margin``
    was derived rather than supplied. Untyped here only to keep ``band`` out of
    this module's import cycle; it is always that type or ``None``."""

    @property
    def temperature(self) -> float:
        return self.witness.temperature

    def threshold_info(self) -> ThresholdInfo:
        return self.witness.threshold_info()

    def describe(self) -> str:
        text = self.witness.describe()
        if self.threshold_choice is not None:
            text += "\n" + "\n".join(
                f"  {line}" for line in self.threshold_choice.describe().splitlines()
            )
        if self.derivation is not None:
            text += "\n" + "\n".join(
                f"  {line}" for line in self.derivation.describe().splitlines()
            )
        return text


def fit_calibration(
    selection: RoleSelection,
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    target_fpr: float = 0.02,
    margin: float | str | None = DERIVE,
    min_coverage: float = 0.85,
    max_selective_fpr: float | None = None,
    determinism_floor: float = 0.0,
    require_binding: bool = False,
    split_lock_hash: str | None = None,
    created_at: str | None = None,
    min_samples: int = MIN_WITNESSED_SAMPLES,
    decode_identity: str | None = None,
    pre_calibration_identity: str | None = None,
) -> CalibrationResult:
    """Fit a temperature and an operating point, through the protocol.

    **The only path that produces a calibration witness.** ``fit_temperature``
    and ``at_target_fpr`` remain available and unchanged for internal and
    exploratory use; they simply cannot produce evidence, which is the
    distinction that makes compliance mean something.

    ``logits`` and ``labels`` are still arrays — this function does not run the
    model — but they must line up with ``selection``, and that is checked: the
    lengths must match, so a caller cannot pass a calibration selection
    alongside logits computed over the benchmark corpus without the mismatch
    surfacing. The remaining exposure is a caller who passes arrays of the right
    length from the wrong corpus, which is stated in the audit rather than
    papered over.

    THE BAND IS NOW DERIVED HERE, NOT SUPPLIED
    ------------------------------------------
    ``margin`` defaults to :data:`DERIVE`, which hands the calibrated scores to
    ``evaluation.band.derive_operating_point`` and takes threshold and margin
    from the same pass over the same data. That is the R8/R11 closure: before
    Batch 7 the threshold was fitted here and the width arrived as a default
    argument from somewhere else, and the two could be jointly impossible — a
    2% budget on 22 negatives put the threshold at 0.9734, where no 0.20-wide
    band fits, and this function raised with no route forward.

    A float is still accepted, and is now a log-odds margin rather than a
    probability width. It is the right thing for reproducing a previously
    calibrated operating point exactly; it is the wrong thing for a new one, and
    ``min_coverage`` / ``max_selective_fpr`` / ``determinism_floor`` are ignored
    when it is used, because nothing is being chosen.

    ``margin=None`` fits a temperature only, and the witness then records no
    threshold. That is a real use — recalibrating without moving the operating
    point — and it is better represented than faked with a placeholder band.

    ``decode_identity`` names the environment the ``logits`` were produced
    under. Pass ``inference.decode_env.decode_identity()`` from the process that
    ran the model; it is not detected here, because this function is handed
    arrays and has no way to know whether they were computed in this process, on
    another machine, or last month. Omitting it leaves the witness recording
    ``None``, which a verifier reports as unanswerable rather than as agreement.

    ``pre_calibration_identity`` names the model configuration — weights, gate
    and selection, excluding calibration — the ``logits`` were produced by.
    **R31.** Pass ``ModelSpec.pre_calibration_identity`` from the spec that
    produced them; not detected here for the identical reason
    ``decode_identity`` is not. Omitting it leaves the witness recording
    ``None``.
    """
    if not isinstance(selection, RoleSelection):
        raise ProtocolViolation(
            "fit_calibration requires a RoleSelection from protocol.select(); "
            "bare arrays cannot produce a witness. Use "
            "inference.calibration.fit_temperature for an unwitnessed fit."
        )
    if selection.role is not CALIBRATION_ROLE:
        raise ProtocolViolation(
            f"cannot fit a calibration on a {selection.role.value!r} selection. "
            f"Only {CALIBRATION_ROLE.value!r} may be fitted on: a threshold "
            "chosen on the corpus it is later measured on reports the best "
            "false-positive rate that corpus can produce, which is not a "
            "prediction of anything."
        )

    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if z.size != y.size:
        raise ValueError(f"logits and labels differ in length: {z.size} vs {y.size}")
    if z.size != len(selection):
        raise ProtocolViolation(
            f"{z.size} logits were supplied for a selection of "
            f"{len(selection)} item(s). The arrays do not correspond to the "
            "corpus this calibration would attest to."
        )
    if z.size < min_samples:
        raise ProtocolViolation(
            f"{z.size} samples is below the {min_samples} needed to trust a fit; "
            "a witness over this would attest to noise"
        )

    temperature = fit_temperature(z, y)
    nll_before = negative_log_likelihood(z, y, 1.0)
    nll_after = negative_log_likelihood(z, y, temperature)

    threshold_value: float | None = None
    margin_value: float | None = None
    choice: ThresholdChoice | None = None
    derivation = None
    if margin is not None:
        # Imported here rather than at module scope: ``band`` imports ``risk``
        # which imports the contract, and this module is what ``band`` is
        # ultimately called from. A local import keeps the package's import
        # graph acyclic without splitting the policy across two files.
        from evaluation.band import InfeasibleOperatingPoint, derive_operating_point

        probabilities = np.asarray(temperature_scale(z, temperature), dtype=np.float64)

        if margin is DERIVE or margin == DERIVE:
            derivation = derive_operating_point(
                y,
                probabilities,
                target_fpr=target_fpr,
                # Defaulting the ceiling to the budget is the *loosest* setting
                # that is not incoherent, and it is reported as non-binding
                # rather than presented as a choice. A caller who means to buy
                # something with the band passes a tighter number.
                max_selective_fpr=(
                    target_fpr if max_selective_fpr is None else max_selective_fpr
                ),
                min_coverage=min_coverage,
                determinism_floor=determinism_floor,
            )
            choice = derivation.threshold_choice
            if not derivation.ok:
                raise InfeasibleOperatingPoint(
                    "no operating point on this calibration set satisfies the "
                    "constraints.\n" + derivation.describe()
                )
            if require_binding and not derivation.binding:
                raise InfeasibleOperatingPoint(
                    "the band was not chosen by anything.\n" + derivation.describe()
                )
            threshold_value = derivation.threshold
            margin_value = derivation.margin
        else:
            choice = at_target_fpr(y, probabilities, target_fpr=target_fpr)
            threshold_value = choice.value
            margin_value = float(margin)

        # Build it once here so an unusable operating point fails at calibration
        # time rather than at the first request that needs it. Unlike the Batch 6
        # version of this block, a *jointly impossible* threshold and band is no
        # longer among the things that can go wrong: every margin is
        # representable at every admissible threshold. What remains is a threshold
        # the fit should not have produced — a sentinel, or a saturated endpoint —
        # and that is now reported as a problem with the fit.
        ThresholdInfo.at_margin(
            value=threshold_value, margin=margin_value, target_fpr=target_fpr
        )

    corpus_fingerprint = dataset_fingerprint(selection.as_list())
    witness_digest = CalibrationWitness.compute_witness(
        protocol_name=selection.protocol_name,
        role=selection.role.value,
        corpus_fingerprint=corpus_fingerprint,
        n_samples=int(z.size),
        temperature=temperature,
        nll_before=nll_before,
        nll_after=nll_after,
        threshold=threshold_value,
        margin=margin_value,
        target_fpr=target_fpr if margin_value is not None else None,
        selection_witness=selection.witness,
        split_lock_hash=split_lock_hash or selection.lock_hash,
        decode_identity=decode_identity,
        pre_calibration_identity=pre_calibration_identity,
    )
    return CalibrationResult(
        witness=CalibrationWitness(
            protocol_name=selection.protocol_name,
            role=selection.role.value,
            corpus_fingerprint=corpus_fingerprint,
            n_samples=int(z.size),
            temperature=temperature,
            nll_before=nll_before,
            nll_after=nll_after,
            threshold=threshold_value,
            margin=margin_value,
            target_fpr=target_fpr if margin_value is not None else None,
            selection_witness=selection.witness,
            split_lock_hash=split_lock_hash or selection.lock_hash,
            witness=witness_digest,
            decode_identity=decode_identity,
            pre_calibration_identity=pre_calibration_identity,
            created_at=created_at
            if created_at is not None
            else datetime.now(timezone.utc).isoformat(),
        ),
        threshold_choice=choice,
        derivation=derivation,
    )


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify_calibration_witness(
    witness: CalibrationWitness | dict | None,
    *,
    protocol=None,
    selection: RoleSelection | None = None,
    items=None,
    split_lock_hash: str | None = None,
    decode_identity: str | None = None,
    pre_calibration_identity: str | None = None,
) -> list[str]:
    """Every way ``witness`` fails to corroborate its claim. Empty means sound.

    Answers the four questions a months-later audit asks, and answers them from
    artifacts alone:

    * **Was this produced through a compliant path?** A ``None`` witness means
      no — reported as "no witness", never as "verified". The absence of
      evidence is the finding.
    * **Which corpus role was used?** From the witness, and cross-checked
      against the role the type permits.
    * **Does the recorded corpus fingerprint match the supplied corpus?** Only
      when a corpus is supplied; otherwise the question is reported as
      unanswered rather than assumed.
    * **Is the witness internally consistent?** Re-derived, so a hand-edited
      threshold or sample count is caught.

    Returns a list rather than raising: an auditor wants everything that is
    wrong, not the first thing.
    """
    problems: list[str] = []
    if witness is None:
        return [
            "no calibration witness; this calibration was fitted on bare arrays "
            "and its corpus cannot be established"
        ]
    if isinstance(witness, dict):
        try:
            witness = CalibrationWitness.from_dict(witness)
        except (ValueError, ProtocolViolation) as exc:
            return [f"calibration witness does not load: {exc}"]

    # Internal consistency. Reconstructing through the constructor already
    # re-derives the digest, so reaching here means it held; recompute anyway so
    # a witness handed in as an object (never round-tripped) is checked too.
    expected = CalibrationWitness.compute_witness(
        protocol_name=witness.protocol_name,
        role=witness.role,
        corpus_fingerprint=witness.corpus_fingerprint,
        n_samples=witness.n_samples,
        temperature=witness.temperature,
        nll_before=witness.nll_before,
        nll_after=witness.nll_after,
        threshold=witness.threshold,
        margin=witness.margin,
        target_fpr=witness.target_fpr,
        selection_witness=witness.selection_witness,
        split_lock_hash=witness.split_lock_hash,
        decode_identity=witness.decode_identity,
        pre_calibration_identity=witness.pre_calibration_identity,
        # Under the witness's OWN schema, so a 1.1.0 witness is checked by the
        # rule it was written under. Verifying an old artifact against a new
        # rule reports tampering where there is none.
        schema_version=witness.schema_version,
    )
    if expected != witness.witness:
        problems.append(
            f"calibration witness is internally inconsistent: recorded "
            f"{witness.witness}, fields hash to {expected}"
        )

    if witness.role != CALIBRATION_ROLE.value:
        problems.append(
            f"the calibration claims role {witness.role!r}, not "
            f"{CALIBRATION_ROLE.value!r}"
        )

    if protocol is not None:
        name = getattr(protocol, "name", protocol)
        if witness.protocol_name != name:
            problems.append(
                f"the calibration was fitted under protocol "
                f"{witness.protocol_name!r}, not {name!r}"
            )

    if selection is not None:
        if witness.selection_witness != selection.witness:
            problems.append(
                "the calibration does not correspond to this selection "
                f"({witness.selection_witness} against {selection.witness})"
            )
        if selection.role is not CALIBRATION_ROLE:
            problems.append(
                f"the supplied selection is for {selection.role.value!r}, which "
                "no calibration may be fitted on"
            )

    if items is not None:
        actual = dataset_fingerprint(list(items))
        if actual != witness.corpus_fingerprint:
            problems.append(
                f"corpus fingerprint mismatch: the witness attests to "
                f"{witness.corpus_fingerprint}, the supplied corpus hashes to "
                f"{actual}"
            )

    if split_lock_hash is not None and witness.split_lock_hash != split_lock_hash:
        problems.append(
            f"the calibration was resolved under lock {witness.split_lock_hash}, "
            f"not {split_lock_hash}"
        )

    if decode_identity is not None:
        if witness.decode_identity is None:
            problems.append(
                "this calibration records no decode environment, so it cannot be "
                f"confirmed to match {decode_identity}; a temperature fitted "
                "under a different decoder scales logits it was not fitted on"
            )
        elif witness.decode_identity != decode_identity:
            problems.append(
                f"the calibration was fitted under decode environment "
                f"{witness.decode_identity}, not {decode_identity}; the "
                "temperature maps logits from a different decoder's pixels"
            )

    if pre_calibration_identity is not None:
        if witness.pre_calibration_identity is None:
            problems.append(
                "this calibration records no model configuration, so it cannot "
                f"be confirmed to match {pre_calibration_identity}; a "
                "temperature fitted for a different weights/gate/selection "
                "configuration scales logits it was not fitted on"
            )
        elif witness.pre_calibration_identity != pre_calibration_identity:
            problems.append(
                f"the calibration was fitted under model configuration "
                f"{witness.pre_calibration_identity}, not "
                f"{pre_calibration_identity}; the temperature maps logits from "
                "a different weights/gate/selection configuration"
            )

    return problems


def assert_not_benchmark_corpus(
    witness: CalibrationWitness | dict | None, benchmark_fingerprint: str
) -> list[str]:
    """Refuse a calibration fitted on the corpus it is being benchmarked on.

    The role check in :func:`fit_calibration` already makes this
    unrepresentable on the compliant path. This is the report-level
    backstop, which catches the case the role check cannot see: a protocol
    whose calibration role and benchmark role name the *same corpus*, which
    ``validate_protocol`` rejects as a document but which a hand-built pair of
    runs could still present.
    """
    if witness is None:
        return []
    corpus = (
        witness["corpus_fingerprint"]
        if isinstance(witness, dict)
        else witness.corpus_fingerprint
    )
    if corpus == benchmark_fingerprint:
        return [
            "the calibration was fitted on the very items being benchmarked "
            f"(both hash to {corpus}); the reported false-positive rate is the "
            "best this corpus can produce and predicts nothing"
        ]
    return []
