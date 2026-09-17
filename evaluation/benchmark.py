"""Benchmark harness: run model versions, compare them honestly, gate releases.

Three things this exists to prevent.

**Comparing on different data.** Every run records the dataset fingerprint and
the split seed. Two runs whose fingerprints differ are not comparable and the
comparison refuses rather than producing a number.

**Comparing without accounting for the shared test set.** A raw AUC delta
between two models on one test set is not evidence: both saw the same items, so
their errors are correlated, and the difference has a much narrower sampling
distribution than either score does alone. The **paired bootstrap** resamples
items — not models — and reports how often the challenger wins. That is the only
comparison here that produces a decision.

**Comparing on the average.** A model can raise overall AUC while collapsing on
a slice that matters — low resolution, heavy compression, one generator. The
gate is therefore two-scope: overall *and* worst slice, both must hold.

No real models are required. ``StubPredictor`` produces deterministic scores from
a hash of the item id, so the whole harness is exercisable — including the
regression gate and its failure modes — before any weights exist.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from contracts.common import ThresholdInfo
from contracts.version import fingerprint
from evaluation.dataset import MediaItem
from evaluation.metrics import average_precision, roc_auc
from evaluation.risk import operating_point


@runtime_checkable
class Predictor(Protocol):
    """Anything that turns items into scores in [0, 1]."""

    name: str
    version: str

    def predict(self, items: list[MediaItem]) -> np.ndarray: ...


class StubPredictor:
    """Deterministic scores from a hash of the media id. No model, no weights.

    Not a model and never pretends to be — ``version`` says ``stub``. Its
    purpose is to make the harness testable: the same item always scores the
    same, ``skill`` moves it smoothly from useless to perfect, and a regression
    can therefore be manufactured on demand.
    """

    def __init__(self, *, name: str = "stub", version: str = "stub-0", skill: float = 0.8):
        if not 0.0 <= skill <= 1.0:
            raise ValueError("skill must lie in [0, 1]")
        self.name = name
        self.version = version
        self.skill = skill

    MAX_MARGIN = 6.0
    """Logit separation at ``skill == 1``. Chosen so the top of the range is
    near-perfect without saturating every score to exactly 0 or 1."""

    def _noise(self, media_id: str) -> float:
        """Content-derived uniform in (0, 1). Stable across processes."""
        digest = hashlib.sha256(f"{self.version}:{media_id}".encode()).hexdigest()
        return (int(digest[:12], 16) + 0.5) / float(0x1000000000000)

    def predict(self, items: list[MediaItem]) -> np.ndarray:
        """Scores whose separability actually tracks ``skill``.

        The obvious construction — ``skill * label + (1 - skill) * noise`` — is
        wrong, and wrong in a way that quietly disables the harness. Its two
        classes occupy ``[skill, 1]`` and ``[0, 1 - skill]``, which do not
        overlap for any ``skill >= 0.5``: every setting from 0.5 upward is a
        *perfect* ranker with AUC exactly 1.0, so no regression can be
        manufactured and a gate that never fires looks like a gate that passes.

        Instead the label sets a logit margin and content-derived logistic noise
        is added on top, so the classes genuinely overlap and AUC moves smoothly
        from 0.5 at ``skill=0`` to near 1.0 at ``skill=1``.
        """
        margin = self.MAX_MARGIN * self.skill
        out = np.empty(len(items), dtype=np.float64)
        for i, item in enumerate(items):
            u = self._noise(item.media_id)
            noise = float(np.log(u / (1.0 - u)))  # uniform -> logistic
            z = margin * (2.0 * item.label - 1.0) + noise
            out[i] = 1.0 / (1.0 + np.exp(-z))
        return out


def dataset_fingerprint(items: list[MediaItem]) -> str:
    """Identity of the evaluation set. Two runs may only be compared under it."""
    return fingerprint(
        {"items": sorted(f"{i.media_id}:{i.label}" for i in items), "n": len(items)}
    )


RUN_SCHEMA_VERSION = "1.2.0"
"""Shape of a serialised :class:`BenchmarkRun`.

1.0.0 is the Batch 2 shape — everything up to ``slices``. 1.1.0 adds the
reproducibility metadata and the two slice-accounting lists. **1.2.0 (Batch 11)
adds** :attr:`BenchmarkRun.decode_identity`. A history file written before a
given version lacks the corresponding fields, and
:meth:`BenchmarkRun.from_dict` fills them with the values that say "not
recorded" rather than with plausible-looking defaults; see the note there.

**Why a bump for an optional field.** Nothing forces ``decode_identity``, so an
old row loads unchanged. The version still has to move because ``from_dict``
refuses unknown fields: a 1.2.0 row read by a 1.1.0 build must fail loudly
rather than be read minus the field that says which decoder produced it. Same
reasoning as the 1.0.0 → 1.1.0 bump in Batch 4.
"""


@dataclass(frozen=True)
class SliceResult:
    name: str
    n: int
    roc_auc: float
    pr_auc: float
    coverage: float
    selective_risk: float
    selective_fpr: float


@dataclass(frozen=True)
class BenchmarkRun:
    """One model version measured on one dataset. Reproducible by construction.

    The metadata block below is what makes "reproducible" more than an
    aspiration. Before it, a run recorded *what the numbers were* and, of the
    things needed to obtain them again, only the model version and a fingerprint
    of the evaluated items. Which split produced those items and which protocol
    said they were a benchmark at all were nowhere, so a comparison that failed
    on differing fingerprints could report that the two runs were incomparable
    and never say why — when the answer, that the lock had advanced a
    generation, was one field away.

    Every metadata field is optional and defaults to ``None``, which reads as
    *not recorded* and never as a default value. A run that does not know its
    split lock must say so; inventing ``generation=0`` would be a lie that
    validates.
    """

    predictor_name: str
    effective_model_version: str
    dataset_fingerprint: str
    n: int
    roc_auc: float
    pr_auc: float
    threshold: float
    band_width: float
    """``band_hi - band_lo`` of the operating point this run decided with.

    Still the probability width after Batch 7, not the log-odds margin that now
    parameterises the band. Deliberate: a risk-coverage report is read in
    probability, the field's meaning is unchanged, and adding ``margin`` beside
    it would bump ``RUN_SCHEMA_VERSION`` and invalidate every stored row to
    record something recoverable from ``threshold`` and this. The margin is
    carried by the calibration witness, which is where provenance belongs.
    """
    coverage: float
    selective_risk: float
    selective_fpr: float
    selective_fnr: float
    slices: list[SliceResult] = field(default_factory=list)

    # -- reproducibility metadata (R3) -------------------------------------
    schema_version: str = RUN_SCHEMA_VERSION
    protocol: str | None = None
    """Name of the :class:`~evaluation.protocol.EvaluationProtocol` in force.

    Populated **only** from a ``RoleSelection``; ``run_benchmark`` refuses to
    record one for a hand-assembled item list. A non-``None`` value here is
    therefore evidence, not a label.
    """
    protocol_role: str | None = None
    """Which role of that protocol these items played."""
    selection_witness: str | None = None
    """Digest binding protocol, role, media ids and lock.

    Re-derivable from artifacts by
    :func:`~evaluation.protocol.verify_run_selection`, which is what lets
    somebody who was not there audit the claim. ``None`` on a historical run —
    and that is why the audit reports "no witness" rather than "verified".
    """
    split_lock_hash: str | None = None
    lock_generation: int | None = None
    calibration: dict | None = None
    """The serialised
    :class:`~evaluation.calibration_protocol.CalibrationWitness`, or ``None``.

    The whole witness rather than its digest, because a report has to stand on
    its own months later: an auditor needs the corpus fingerprint and the role
    to check that the model was not calibrated on the corpus it is benchmarked
    on, and a bare digest can only be compared against a witness they would have
    to find first.

    ``None`` means the calibration was fitted on bare arrays and its corpus
    cannot be established. That is a reportable state, not a gap to fill with a
    plausible default.
    """
    calibration_fingerprint: str | None = None
    """The calibration half of ``effective_model_version``, carried separately.

    Redundant with the composed version and worth the duplication: it is the
    component that changes most often and least visibly, and having it as its
    own field means a history can be grouped by calibration without every reader
    re-implementing ``parse_model_version``.
    """
    decode_identity: str | None = None
    """The decode environment that produced this run's scores. **R12, Batch 11.**

    ``inference.decode_env.decode_identity()`` — OpenCV version, video backend,
    libavcodec, OS and machine. Not the forensic digest: Batch 8 measured the CPU
    fields not to change a decoded pixel, so gating on them would make two
    machines in one fleet permanently incomparable for no correctness gain.

    **Why this is a separate field and not part of ``effective_model_version``.**
    That string is the *model* identity on purpose (Batch 9): "did the model get
    worse" must not be answered yes by a worker rebuild. Folding the decoder in
    would make every model-versus-model comparison also a decoder comparison and
    would stop the "same effective model version" note firing when it should.
    Kept apart, the runtime identity is still recoverable —
    ``compose_runtime_version(model=effective_model_version,
    decode=decode_identity)`` — so the decomposed pair carries strictly more
    information than the composed string would.

    **Why it is passed in rather than detected.** Same reason
    ``effective_model_version`` is: the harness process is not necessarily the
    process that produced the scores. Offline scoring, cached logits, a scoring
    job on one box and a report assembled on another are all ordinary, and in
    every one of them this process's own environment is the wrong answer.
    Recording it automatically would be a false claim, which is worse than none.

    ``None`` means *not recorded*, and :func:`regression_gate` reports that as a
    question it cannot answer rather than as agreement.
    """

    # -- slice accounting (R2) ---------------------------------------------
    empty_slices: tuple[str, ...] = ()
    """Named in ``slice_by``, matched no evaluated items."""
    single_class_slices: tuple[str, ...] = ()
    """Matched items, but all of one label — so no AUC is defined.

    Recorded rather than dropped. A slice that quietly becomes single-class is
    how a two-scope gate becomes a one-scope gate, and the acquisition that did
    it — a batch of authentic-only low-resolution media, say — is an ordinary
    event that nothing else would report.
    """

    @property
    def worst_slice(self) -> SliceResult | None:
        return min(self.slices, key=lambda s: s.roc_auc) if self.slices else None

    @property
    def measured_slice_names(self) -> set[str]:
        return {s.name for s in self.slices}

    @property
    def accounted_slice_names(self) -> set[str]:
        """Every slice this run knows about, measured or not."""
        return (
            self.measured_slice_names
            | set(self.empty_slices)
            | set(self.single_class_slices)
        )

    def describe(self) -> str:
        lines = [
            f"{self.predictor_name}  [{self.effective_model_version}]",
            f"  n={self.n}  AUC {self.roc_auc:.4f}  PR-AUC {self.pr_auc:.4f}",
            f"  at threshold {self.threshold:.3f} (band {self.band_width:.3f}): "
            f"coverage {self.coverage:.3f}  risk {self.selective_risk:.4f}  "
            f"FPR {self.selective_fpr:.4f}  FNR {self.selective_fnr:.4f}",
        ]
        provenance = [
            f"protocol={self.protocol or '(unrecorded)'}",
            f"lock={self.split_lock_hash or '(unrecorded)'}",
            f"gen={self.lock_generation if self.lock_generation is not None else '-'}",
            f"data={self.dataset_fingerprint}",
            f"decode={self.decode_identity or '(unrecorded)'}",
        ]
        lines.append("  " + "  ".join(provenance))
        for s in sorted(self.slices, key=lambda x: x.roc_auc):
            lines.append(f"    slice {s.name:<18} n={s.n:<6} AUC {s.roc_auc:.4f}")
        for name in sorted(self.single_class_slices):
            lines.append(f"    slice {name:<18} NOT MEASURED (single class)")
        for name in sorted(self.empty_slices):
            lines.append(f"    slice {name:<18} NOT MEASURED (empty)")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict) -> BenchmarkRun:
        """Rebuild a run from a history row, including a pre-1.1.0 one.

        Missing metadata becomes ``None`` and missing slice accounting becomes
        empty, because that is what a 1.0.0 row actually tells us: not that
        there were no empty slices, but that nobody looked. The distinction
        matters to :func:`regression_gate`, which must not report a slice as
        "vanished" when comparing against a baseline that never recorded slice
        accounting in the first place.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(
                f"unknown BenchmarkRun field(s) {unknown}; a history row from a "
                f"newer schema than {RUN_SCHEMA_VERSION} cannot be read by this "
                "build"
            )
        missing = sorted(
            {
                "predictor_name",
                "effective_model_version",
                "dataset_fingerprint",
                "n",
                "roc_auc",
            }
            - set(payload)
        )
        if missing:
            raise ValueError(f"history row is missing required field(s) {missing}")
        data = dict(payload)
        data["slices"] = [SliceResult(**s) for s in data.get("slices", [])]
        data["empty_slices"] = tuple(data.get("empty_slices") or ())
        data["single_class_slices"] = tuple(data.get("single_class_slices") or ())
        data.setdefault("schema_version", "1.0.0")
        return cls(**data)


def run_benchmark(
    predictor: Predictor,
    items: list[MediaItem],
    *,
    threshold: ThresholdInfo,
    effective_model_version: str,
    slice_by: dict[str, list[MediaItem]] | None = None,
    protocol: object | None = None,
    split_lock: object | None = None,
    calibration: object | None = None,
    calibration_fingerprint: str | None = None,
    decode_identity: str | None = None,
) -> BenchmarkRun:
    """Score ``items`` and record everything needed to compare this run later.

    ``effective_model_version`` is passed in rather than read off the predictor
    because it is the *deployment's* identity — weights, calibration, threshold
    and gate composed — and only the caller holds all four.

    ``items`` is either a plain list — an unprotocolled run, which records no
    protocol — or a :class:`~evaluation.protocol.RoleSelection` returned by
    ``protocol.select()``. **Those are the only two options, and passing
    ``protocol=`` beside a plain list is refused.** That combination was the R5
    defect: the two arguments were unrelated, so a run could score the training
    set and record a benchmark protocol next to it, and nothing was checking.

    ``split_lock`` accepts a :class:`~evaluation.split_lock.SplitLock` or a bare
    hash string, duck-typed so a caller holding only the hash need not
    reconstruct a lock. When a selection was resolved under a lock, the two must
    agree, and a disagreement is refused rather than recorded.

    Slices that cannot be measured are **named in the result** rather than
    dropped. That was the R2 defect: a slice which quietly became single-class
    disappeared from the run, and the two-scope regression gate then had nothing
    to compare and said nothing about it.

    ``decode_identity`` is the environment that produced the *scores* — pass
    ``inference.decode_env.decode_identity()`` from the process that ran the
    model. It is an argument rather than something this function detects, for
    the reason given on the field: this process may not be that process, and a
    confidently wrong environment is worse than an absent one. Omitting it is
    allowed and makes every later comparison of this run carry a note.
    """
    from evaluation.protocol import ProtocolViolation, RoleSelection

    selection = items if isinstance(items, RoleSelection) else None
    if selection is not None:
        if protocol is not None:
            raise ProtocolViolation(
                "pass either a RoleSelection or a protocol, not both; the "
                "selection already names the protocol it was made under "
                f"({selection.protocol_name!r})"
            )
        items = selection.as_list()
    elif protocol is not None:
        raise ProtocolViolation(
            "a protocol cannot be recorded for a hand-assembled item list. "
            "Recording is not enforcement: these two arguments used to be "
            "unrelated, so a run could score the training set and label itself "
            "a benchmark. Obtain the items through "
            "protocol.select(protocol, role, items, assignment) and pass the "
            "RoleSelection as `items`."
        )

    if not items:
        raise ValueError("cannot benchmark an empty item list")
    y = np.array([i.label for i in items], dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise ValueError("benchmark needs both classes present")

    scores = np.asarray(predictor.predict(items), dtype=np.float64)
    if scores.size != len(items):
        raise ValueError(
            f"predictor returned {scores.size} scores for {len(items)} items"
        )

    point = operating_point(y, scores, threshold)
    slices: list[SliceResult] = []
    empty: list[str] = []
    single_class: list[str] = []
    for name, subset in (slice_by or {}).items():
        ids = {i.media_id for i in subset}
        mask = np.array([i.media_id in ids for i in items])
        if mask.sum() == 0:
            empty.append(name)
            continue
        sy, ss = y[mask], scores[mask]
        if len(np.unique(sy)) < 2:
            single_class.append(name)  # no AUC is defined, but the slice exists
            continue
        sp = operating_point(sy, ss, threshold)
        slices.append(
            SliceResult(
                name=name,
                n=int(mask.sum()),
                roc_auc=roc_auc(sy, ss),
                pr_auc=average_precision(sy, ss),
                coverage=sp.coverage,
                selective_risk=sp.selective_risk,
                selective_fpr=sp.selective_fpr,
            )
        )

    lock_hash = (
        split_lock.content_hash()
        if hasattr(split_lock, "content_hash")
        else split_lock
    )

    # Calibration metadata is *consumed*, not trusted. When a witness is given
    # the fingerprint is derived from it, so a run cannot pair a witness with an
    # unrelated fingerprint the caller happened to type.
    calibration_payload = None
    if calibration is not None:
        calibration_payload = (
            calibration.to_dict() if hasattr(calibration, "to_dict") else calibration
        )
        if not isinstance(calibration_payload, dict):
            raise ProtocolViolation(
                "calibration must be a CalibrationWitness or its serialised "
                f"dict, got {type(calibration).__name__}"
            )
        derived = calibration_payload.get("witness")
        if calibration_fingerprint is not None and calibration_fingerprint != derived:
            raise ProtocolViolation(
                f"the run records calibration fingerprint "
                f"{calibration_fingerprint!r} beside a witness attesting to "
                f"{derived!r}; one of the two is wrong and guessing which would "
                "defeat the point of recording either"
            )
        calibration_fingerprint = derived

    if selection is not None:
        # Re-derive the witness from the items actually scored. The selection
        # could have been built by hand; this is what makes the recorded
        # protocol a statement about *these* items rather than a label.
        scored_ids = tuple(sorted(i.media_id for i in items))
        expected = selection.compute_witness(
            selection.protocol_name, selection.role, scored_ids, selection.lock_hash
        )
        if expected != selection.witness:
            raise ProtocolViolation(
                "the scored items do not match the selection's witness; the "
                "item list was altered after selection"
            )
        if (
            selection.lock_hash is not None
            and lock_hash is not None
            and selection.lock_hash != lock_hash
        ):
            raise ProtocolViolation(
                f"the selection was resolved under lock {selection.lock_hash} "
                f"but the run records {lock_hash}; one of the two is wrong and "
                "guessing which would defeat the point of recording either"
            )
        lock_hash = lock_hash or selection.lock_hash

    return BenchmarkRun(
        predictor_name=predictor.name,
        effective_model_version=effective_model_version,
        dataset_fingerprint=dataset_fingerprint(items),
        n=len(items),
        roc_auc=roc_auc(y, scores),
        pr_auc=average_precision(y, scores),
        threshold=threshold.value,
        band_width=threshold.band_width,
        coverage=point.coverage,
        selective_risk=point.selective_risk,
        selective_fpr=point.selective_fpr,
        selective_fnr=point.selective_fnr,
        slices=slices,
        protocol=selection.protocol_name if selection else None,
        protocol_role=selection.role.value if selection else None,
        selection_witness=selection.witness if selection else None,
        split_lock_hash=lock_hash,
        lock_generation=getattr(split_lock, "generation", None),
        calibration=calibration_payload,
        calibration_fingerprint=calibration_fingerprint,
        decode_identity=decode_identity,
        empty_slices=tuple(sorted(empty)),
        single_class_slices=tuple(sorted(single_class)),
    )


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedComparison:
    """Challenger against incumbent, on the same items, with the same resamples."""

    metric: str
    incumbent: float
    challenger: float
    delta: float
    win_rate: float
    ci_lo: float
    ci_hi: float
    resamples: int

    @property
    def significant(self) -> bool:
        """True when the interval on the difference excludes zero."""
        return self.ci_lo > 0.0 or self.ci_hi < 0.0

    def describe(self) -> str:
        verdict = (
            "challenger wins"
            if self.significant and self.delta > 0
            else "incumbent holds"
            if self.significant
            else "no significant difference"
        )
        return (
            f"{self.metric}: {self.incumbent:.4f} -> {self.challenger:.4f} "
            f"(delta {self.delta:+.4f}, 95% CI [{self.ci_lo:+.4f}, {self.ci_hi:+.4f}], "
            f"win rate {self.win_rate:.3f}) — {verdict}"
        )


def paired_bootstrap(
    y_true: np.ndarray,
    incumbent_scores: np.ndarray,
    challenger_scores: np.ndarray,
    *,
    metric: str = "roc_auc",
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedComparison:
    """Compare two scorers on one test set, resampling **items**, not models.

    Both models are evaluated on the *same* resampled indices every draw. That
    is what makes it paired, and it is the entire point: the two scorers share
    the test set, so their errors are correlated, and an unpaired comparison
    would attribute that shared variance to the difference between them and
    report a far wider interval than the evidence supports.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    a = np.asarray(incumbent_scores, dtype=np.float64).ravel()
    b = np.asarray(challenger_scores, dtype=np.float64).ravel()
    if not (y.size == a.size == b.size):
        raise ValueError("labels and both score arrays must be the same length")

    fn = {"roc_auc": roc_auc, "pr_auc": average_precision}[metric]
    base_a, base_b = fn(y, a), fn(y, b)

    rng = np.random.default_rng(seed)
    deltas = np.empty(resamples, dtype=np.float64)
    wins = 0
    drawn = 0
    for _ in range(resamples):
        idx = rng.integers(0, y.size, y.size)
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue  # a degenerate resample has no defined AUC
        da = fn(ys, b[idx]) - fn(ys, a[idx])
        deltas[drawn] = da
        wins += da > 0
        drawn += 1
    if drawn == 0:
        raise ValueError("every bootstrap resample was single-class")

    deltas = deltas[:drawn]
    lo, hi = np.quantile(deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
    return PairedComparison(
        metric=metric,
        incumbent=base_a,
        challenger=base_b,
        delta=base_b - base_a,
        win_rate=wins / drawn,
        ci_lo=float(lo),
        ci_hi=float(hi),
        resamples=drawn,
    )


# --------------------------------------------------------------------------
# Regression gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        lines = [f"regression gate: {head}"]
        lines += [f"  FAIL  {f}" for f in self.failures]
        lines += [f"  note  {n}" for n in self.notes]
        return "\n".join(lines)


def _slice_failures(
    incumbent: BenchmarkRun,
    challenger: BenchmarkRun,
    max_slice_auc_drop: float,
    notes: list[str],
) -> list[str]:
    """Compare slices over the **union** of both runs' slice names.

    Iterating the challenger's slices alone — which is what this did before —
    means a slice that exists in the baseline and not in the challenger is never
    examined. It does not fail the gate and does not even produce a note: the
    two-scope gate silently becomes a one-scope gate, and the model that
    collapsed on heavy compression ships because the check that would have
    caught it did not run.

    Two ordinary things make a slice vanish, and the run now records which:

    * it became **single-class** — an acquisition of authentic-only low-resolution
      media is enough — so no AUC is defined for it;
    * it became **empty**, or was renamed, or was re-bucketed.

    A vanished slice is a failure, not a note. The gate cannot know whether the
    slice was retired on purpose, and the safe reading of "the check I relied on
    last time is missing" is that the release does not go out until somebody
    says why.
    """
    failures: list[str] = []
    before_by_name = {s.name: s for s in incumbent.slices}
    after_by_name = {s.name: s for s in challenger.slices}

    for name in sorted(before_by_name.keys() | after_by_name.keys()):
        before, after = before_by_name.get(name), after_by_name.get(name)

        if before is not None and after is not None:
            drop = before.roc_auc - after.roc_auc
            if drop > max_slice_auc_drop:
                failures.append(
                    f"slice {name!r} AUC fell {drop:.4f} "
                    f"({before.roc_auc:.4f} -> {after.roc_auc:.4f}), beyond the "
                    f"{max_slice_auc_drop} allowance"
                )
            continue

        if before is None:
            notes.append(f"slice {name!r} is new and has no baseline")
            continue

        # Present in the baseline, absent from the challenger. Name the cause
        # when the run recorded one — "it went single-class" is a different
        # conversation from "it was renamed", and guessing wastes the expensive
        # kind of afternoon.
        if name in challenger.single_class_slices:
            why = (
                "it is now single-class, so no AUC is defined for it; the "
                "acquisition that made it single-class removed a scope from "
                "this gate"
            )
        elif name in challenger.empty_slices:
            why = "it matched no evaluated items"
        else:
            why = (
                "the challenger did not measure it at all; it was renamed, "
                "re-bucketed, or dropped from slice_by"
            )
        failures.append(
            f"slice {name!r} has a baseline (AUC {before.roc_auc:.4f}) but is "
            f"missing from the challenger: {why}"
        )

    # Accounted-for-but-unmeasured slices that BOTH runs share are not a
    # regression — nothing changed — but they are a hole in the gate and the
    # report should say so rather than reading as full coverage.
    shared_unmeasured = (
        incumbent.accounted_slice_names - incumbent.measured_slice_names
    ) & (challenger.accounted_slice_names - challenger.measured_slice_names)
    for name in sorted(shared_unmeasured):
        notes.append(
            f"slice {name!r} is unmeasured in both runs, so it is outside this "
            "gate entirely"
        )

    return failures


def _decode_comparability(
    incumbent: BenchmarkRun, challenger: BenchmarkRun, notes: list[str]
) -> str | None:
    """Refuse a cross-decoder comparison. Returns the failure, or ``None``.

    **Three states, not two**, which is R4's "unsealed is a third state, not a
    clean one" applied to a different artifact. Two recorded and equal is a pass;
    two recorded and different is a refusal; either unrecorded is a *note*,
    because no evidence is not evidence of sameness and a gate that treated it as
    one would launder every historical row into a guarantee it never made.

    **Why a refusal rather than a tolerance.** Batch 8 measured the decoder
    difference as a systematic levels shift, not noise, and a systematic shift is
    close to rank-preserving — so AUC barely moves while the *threshold*-dependent
    metrics do. Measured on a synthetic model of that shift at Batch 8's own
    magnitude, a decoder change alone moves selective FPR by 24.6% of this gate's
    allowance while moving AUC by 0.0%; at four times that magnitude the gate
    fails and reports "detection was bought with false accusations" for a model
    that never changed.

    That is the worst available combination: the metric that survives a decoder
    change is the one this gate leans on hardest, and the metric that moves is
    the one it attributes to the model. Neither a tighter tolerance nor a looser
    one helps, because the quantity being spent is not the model's.

    Short-circuits for the same reason a dataset mismatch does: every subsequent
    comparison inherits the contamination, so continuing would produce a list of
    findings about a delta that is not the model's.
    """
    mine, theirs = incumbent.decode_identity, challenger.decode_identity
    if mine is None or theirs is None:
        which = (
            "neither run records"
            if mine is None and theirs is None
            else "the incumbent does not record"
            if mine is None
            else "the challenger does not record"
        )
        notes.append(
            f"{which} a decode identity, so this comparison cannot rule out that "
            "the two runs were scored under different decoders; Batch 8 measured "
            "that alone moving selective FPR by a quarter of this gate's budget"
        )
        return None
    if mine == theirs:
        return None
    return (
        f"runs were scored under different decode environments ({mine} vs "
        f"{theirs}); they are not comparable. A decoder change shifts the "
        "threshold-dependent metrics while leaving AUC almost untouched, so the "
        "difference would be charged to the model. Re-run one side under the "
        "other's environment, or compare each against its own baseline."
    )


def regression_gate(
    incumbent: BenchmarkRun,
    challenger: BenchmarkRun,
    *,
    max_auc_drop: float = 0.005,
    max_slice_auc_drop: float = 0.02,
    max_fpr_increase: float = 0.005,
) -> GateResult:
    """Two-scope gate: overall and worst slice, plus the false-positive budget.

    A model may not improve on average by getting worse where it already
    struggled, and it may not buy detection with false positives. Both are
    ordinary outcomes of retraining and neither shows up in a single number.

    Slices are compared over the **union** of both runs' names, so a slice
    cannot leave the gate by disappearing. See :func:`_slice_failures`.
    """
    failures: list[str] = []
    notes: list[str] = []

    if incumbent.dataset_fingerprint != challenger.dataset_fingerprint:
        failures.append(
            "runs were measured on different datasets "
            f"({incumbent.dataset_fingerprint} vs {challenger.dataset_fingerprint}); "
            "they are not comparable"
        )
        return GateResult(passed=False, failures=failures)

    decode_problem = _decode_comparability(incumbent, challenger, notes)
    if decode_problem is not None:
        failures.append(decode_problem)
        return GateResult(passed=False, failures=failures, notes=notes)

    if incumbent.effective_model_version == challenger.effective_model_version:
        notes.append(
            "both runs report the same effective model version; if the models "
            "really differ, something that changes a score is missing from the "
            "version composition"
        )

    drop = incumbent.roc_auc - challenger.roc_auc
    if drop > max_auc_drop:
        failures.append(
            f"overall AUC fell {drop:.4f} ({incumbent.roc_auc:.4f} -> "
            f"{challenger.roc_auc:.4f}), beyond the {max_auc_drop} allowance"
        )

    failures.extend(_slice_failures(incumbent, challenger, max_slice_auc_drop, notes))

    fpr_rise = challenger.selective_fpr - incumbent.selective_fpr
    if fpr_rise > max_fpr_increase:
        failures.append(
            f"selective FPR rose {fpr_rise:.4f} ({incumbent.selective_fpr:.4f} -> "
            f"{challenger.selective_fpr:.4f}); detection was bought with false "
            "accusations"
        )

    return GateResult(passed=not failures, failures=failures, notes=notes)


def append_history(run: BenchmarkRun, path: Path) -> None:
    """Append one run to a JSONL history. Append-only on purpose.

    A history a process can rewrite is a history nobody can trust; this is the
    record a regression is measured against.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(run.to_json() + "\n")


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_runs(path: Path) -> list[BenchmarkRun]:
    """Read a history as typed runs, tolerating rows from before this batch.

    Additive rather than a change to ``read_history``, whose dict return is what
    a caller wants for ad-hoc inspection of a file that may hold several schema
    versions. This one is for callers that want to *compare* rows, where a typed
    object with explicit "not recorded" fields is what keeps a 1.0.0 baseline
    from being mistaken for a run that measured no slices.
    """
    return [BenchmarkRun.from_dict(row) for row in read_history(path)]


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkReport:
    """A publishable result: the cross-dataset number and the in-domain one.

    Two runs, not one, because ``REQUIRED_REPORT_FIELDS`` demands
    ``primary_benchmark`` and ``in_domain`` **together** and a single run can
    only ever hold one of them. Quoting either alone is the standard way to
    mislead — in-domain alone sells a solved problem as an achievement,
    cross-dataset alone hides that the model may simply be weak everywhere — so
    the pairing is structural here rather than a convention someone remembers.

    :meth:`validate` refuses two more things that produce a report which passes
    every field check and still says nothing: a pair of runs from *different
    models*, and a pair from the *same corpus*.
    """

    primary: BenchmarkRun
    in_domain: BenchmarkRun
    protocol: str

    def to_payload(self) -> dict:
        """The report, assembled entirely from run artifacts.

        Threshold, coverage and the selective rates come from the **primary
        benchmark** run and never from the in-domain one. Under this platform's
        protocol the in-domain split is the calibration set, so its
        threshold-dependent numbers are best-case by construction — the
        threshold was chosen on it. Its AUC is untouched by that (temperature
        scaling is monotone), which is why the in-domain contribution to a
        report is a ranking number and nothing else.
        """
        worst = self.primary.worst_slice
        calibration = self.primary.calibration or {}
        return {
            "protocol": self.protocol,
            "effective_model_version": self.primary.effective_model_version,
            "selection_witness": self.primary.selection_witness,
            "calibration_witness": self.primary.calibration_fingerprint,
            "calibration_role": calibration.get("role"),
            "calibration_corpus_fingerprint": calibration.get("corpus_fingerprint"),
            "calibration_n_samples": calibration.get("n_samples"),
            "calibration_fingerprint": self.primary.calibration_fingerprint,
            "split_lock_hash": self.primary.split_lock_hash,
            "lock_generation": self.primary.lock_generation,
            "dataset_fingerprint": self.primary.dataset_fingerprint,
            # Deliberately NOT added to REQUIRED_REPORT_FIELDS: every historical
            # report would become incomplete overnight for a field nothing could
            # back-fill, and check_report treats None as missing. Published so a
            # reader can see it, not demanded so a reader can trust it.
            "decode_identity": self.primary.decode_identity,
            "calibration_decode_identity": calibration.get("decode_identity"),
            "primary_benchmark": self.primary.roc_auc,
            "in_domain": self.in_domain.roc_auc,
            "threshold": self.primary.threshold,
            "band_width": self.primary.band_width,
            "coverage": self.primary.coverage,
            "selective_risk": self.primary.selective_risk,
            "selective_fpr": self.primary.selective_fpr,
            "worst_slice": worst.name if worst else None,
            "worst_slice_auc": worst.roc_auc if worst else None,
            "unmeasured_slices": sorted(
                set(self.primary.empty_slices) | set(self.primary.single_class_slices)
            ),
            "n": self.primary.n,
        }

    def validate(self) -> list[str]:
        """Everything wrong with this report. Empty means publishable."""
        from evaluation.protocol import check_report

        problems = [
            f"{field} is not recorded" for field in check_report(self.to_payload())
        ]
        if (
            self.primary.effective_model_version
            != self.in_domain.effective_model_version
        ):
            problems.append(
                "the two runs are of different models "
                f"({self.primary.effective_model_version} vs "
                f"{self.in_domain.effective_model_version}); a report pairs one "
                "model's cross-dataset and in-domain numbers, and pairing two "
                "models' best results is not a measurement of either"
            )
        if self.primary.dataset_fingerprint == self.in_domain.dataset_fingerprint:
            problems.append(
                "the primary benchmark and the in-domain run were measured on "
                "the same items, so the cross-dataset number is not "
                "cross-dataset"
            )

        # Calibration provenance, checked from the run's own artifacts rather
        # than from anything the caller says now.
        from evaluation.calibration_protocol import (
            assert_not_benchmark_corpus,
            verify_calibration_witness,
        )

        # decode_identity= closes R31's D1 wiring: the field has existed on
        # BenchmarkRun since Batch 11, but this call never used it to
        # cross-check the attached calibration witness. See
        # docs/R31_SPEC_RESOLUTION.md Decision 2 for why only this half ships
        # here — a model-configuration cross-check would need a new
        # BenchmarkRun field and its own RUN_SCHEMA_VERSION bump, deferred.
        problems += verify_calibration_witness(
            self.primary.calibration,
            protocol=self.protocol,
            decode_identity=self.primary.decode_identity,
        )
        problems += assert_not_benchmark_corpus(
            self.primary.calibration, self.primary.dataset_fingerprint
        )
        if self.primary.calibration is not None:
            lock = self.primary.calibration.get("split_lock_hash")
            if (
                lock is not None
                and self.primary.split_lock_hash is not None
                and lock != self.primary.split_lock_hash
            ):
                problems.append(
                    f"the calibration was resolved under split lock {lock} but "
                    f"the benchmark ran under {self.primary.split_lock_hash}; "
                    "the two are not the same frozen corpus"
                )
            # The same shape of check, one axis over. A calibration fitted under
            # one decoder and reported against a benchmark run under another is
            # a temperature mapping logits it was not fitted on — and unlike the
            # lock case, nothing upstream would have refused it.
            fitted_under = self.primary.calibration.get("decode_identity")
            if (
                fitted_under is not None
                and self.primary.decode_identity is not None
                and fitted_under != self.primary.decode_identity
            ):
                problems.append(
                    f"the calibration was fitted under decode environment "
                    f"{fitted_under} but the benchmark ran under "
                    f"{self.primary.decode_identity}; the temperature was fitted "
                    "on logits from a different decoder's pixels"
                )

        # Both runs in a report must come from one environment, for the same
        # reason regression_gate refuses a cross-decoder comparison: the pairing
        # is presented as one model measured two ways, not as two environments.
        if (
            self.primary.decode_identity is not None
            and self.in_domain.decode_identity is not None
            and self.primary.decode_identity != self.in_domain.decode_identity
        ):
            problems.append(
                f"the primary benchmark ran under decode environment "
                f"{self.primary.decode_identity} and the in-domain run under "
                f"{self.in_domain.decode_identity}; a report pairs one "
                "deployment's two numbers, not two deployments'"
            )
        return problems

    def describe(self) -> str:
        problems = self.validate()
        payload = self.to_payload()
        lines = [
            f"report under protocol {self.protocol!r}"
            + ("" if not problems else f"  — {len(problems)} PROBLEM(S)"),
            f"  model      {payload['effective_model_version']}",
            f"  split lock {payload['split_lock_hash'] or '(unrecorded)'}"
            f"  gen {payload['lock_generation'] if payload['lock_generation'] is not None else '-'}",
            f"  decode     {payload['decode_identity'] or '(unrecorded)'}"
            f"  calibrated under {payload['calibration_decode_identity'] or '(unrecorded)'}",
            f"  primary    AUC {payload['primary_benchmark']:.4f}  n={payload['n']}",
            f"  in-domain  AUC {payload['in_domain']:.4f}",
            f"  at {payload['threshold']:.3f} (band {payload['band_width']:.3f}): "
            f"coverage {payload['coverage']:.3f}  risk {payload['selective_risk']:.4f}  "
            f"FPR {payload['selective_fpr']:.4f}",
        ]
        if payload["worst_slice"]:
            lines.append(
                f"  worst slice {payload['worst_slice']} "
                f"AUC {payload['worst_slice_auc']:.4f}"
            )
        if payload["unmeasured_slices"]:
            lines.append(f"  unmeasured slices: {payload['unmeasured_slices']}")
        lines += [f"  PROBLEM  {p}" for p in problems]
        return "\n".join(lines)
