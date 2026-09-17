"""The evaluation protocol: which corpus plays which role, and why it matters.

THE NUMBER EVERYONE REPORTS IS THE ONE THAT MEANS LEAST
-------------------------------------------------------
FF++ in-dataset AUC above 0.99 is a solved problem and has been for years. It
survives in papers because it is easy, and it predicts nothing: a detector that
has learned FaceSwap's blending seam gets 0.99 on FF++ and falls to chance on a
diffusion-generated face it has never seen. The only measurement that carries
information about production is one taken on a corpus that had **no** influence
on the model — not on its weights, not on its temperature, not on its threshold.

Keeping those two apart is not something a naming convention can enforce, which
is why it is a validated object here rather than a paragraph in a README. Five
roles, and every one of them is a way for information to leak from a benchmark
back into the model if it is assigned to the wrong corpus:

    TRAIN                fits the weights.
    VALIDATION           picks the checkpoint, the learning rate, the
                         architecture. Sees the model thousands of times and is
                         thoroughly used up by the end of a project; it is not a
                         benchmark and must never be reported as one.
    CALIBRATION          fits the temperature and selects the threshold. Touches
                         no weights, and is *still* a way to overfit an
                         operating point: a threshold chosen on the benchmark
                         reports the best false-positive rate that corpus can
                         produce rather than the one production will see.
    PRIMARY_BENCHMARK    the headline. Cross-dataset, held out entirely,
                         measured rarely.
    SECONDARY_BENCHMARK  the others. Reported alongside, never instead.

THE RULE THAT DOES THE WORK
---------------------------
**A benchmark dataset contributes to nothing.** Not a training split, not a
validation split, not a calibration set, not a slice used to pick a band width.
Once any part of a corpus has influenced any fitted quantity, the rest of that
corpus is no longer measuring generalisation to unseen data — it is measuring
generalisation to unseen *files* from a distribution the model has already been
tuned against, and those are different numbers by tens of AUC points.

This is stricter than the usual "don't train on test" and deliberately so. The
usual rule is satisfied by fitting a temperature on Celeb-DF and then reporting
Celeb-DF AUC, which happens constantly and is precisely the failure this module
refuses to represent.

WHY CALIBRATION GETS ITS OWN ROLE
---------------------------------
It could have been folded into validation, and folding it in is a mistake worth
naming. The validation split is used to select checkpoints, so its logits are
optimistic — the checkpoint chosen is the one that happened to do best on them.
A temperature fitted there is fitted to that optimism and comes out close to
1.0, reporting a well-calibrated model that is overconfident everywhere else.
Calibration wants data that selected nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from contracts.version import fingerprint
from evaluation.dataset import MediaItem, Split


class DatasetRole(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    PRIMARY_BENCHMARK = "primary_benchmark"
    SECONDARY_BENCHMARK = "secondary_benchmark"


FITTING_ROLES = (
    DatasetRole.TRAIN,
    DatasetRole.VALIDATION,
    DatasetRole.CALIBRATION,
)
"""Roles in which data influences a number the model will later use.

The distinction that the whole protocol turns on. Weights, checkpoint choice and
temperature are all fitted quantities; a corpus that touched any of them cannot
also be a benchmark for it.
"""

BENCHMARK_ROLES = (
    DatasetRole.PRIMARY_BENCHMARK,
    DatasetRole.SECONDARY_BENCHMARK,
)

SINGLETON_ROLES = (
    DatasetRole.TRAIN,
    DatasetRole.VALIDATION,
    DatasetRole.CALIBRATION,
    DatasetRole.PRIMARY_BENCHMARK,
)
"""Roles that take exactly one assignment.

One headline benchmark, on purpose. Two primaries means whichever one looks
better gets quoted, and the choice is made after the numbers are in.
"""


@dataclass(frozen=True)
class RoleAssignment:
    """One corpus, in one role, using named splits of itself."""

    dataset_id: str
    role: DatasetRole
    splits: tuple[Split, ...]
    note: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        if not self.splits:
            raise ValueError(
                f"{self.dataset_id}/{self.role.value} names no splits; a role "
                "that uses nothing should be left out of the protocol instead"
            )
        if len(set(self.splits)) != len(self.splits):
            raise ValueError(f"{self.dataset_id}/{self.role.value} repeats a split")

    def describe(self) -> str:
        splits = "+".join(s.value for s in self.splits)
        note = f"  — {self.note}" if self.note else ""
        return f"{self.role.value:<20} {self.dataset_id} [{splits}]{note}"


@dataclass(frozen=True)
class ProtocolReport:
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.problems

    def describe(self) -> str:
        if self.valid and not self.warnings:
            return "protocol is valid"
        lines = [f"  PROBLEM  {p}" for p in self.problems]
        lines += [f"  warning  {w}" for w in self.warnings]
        return "\n".join(lines)

    def __bool__(self) -> bool:
        """Truthy when the protocol is valid. Reads as ``if validate(p):``."""
        return self.valid


@dataclass(frozen=True)
class EvaluationProtocol:
    """Which corpus does what. Validated, not merely documented."""

    name: str
    assignments: tuple[RoleAssignment, ...]

    def by_role(self, role: DatasetRole) -> list[RoleAssignment]:
        return [a for a in self.assignments if a.role is role]

    def one(self, role: DatasetRole) -> RoleAssignment:
        found = self.by_role(role)
        if len(found) != 1:
            raise ValueError(
                f"expected exactly one {role.value} assignment, found {len(found)}"
            )
        return found[0]

    def dataset_ids(self, *roles: DatasetRole) -> set[str]:
        wanted = set(roles) if roles else set(DatasetRole)
        return {a.dataset_id for a in self.assignments if a.role in wanted}

    def with_secondary(
        self, dataset_id: str, *, splits: tuple[Split, ...] = tuple(Split), note: str = ""
    ) -> EvaluationProtocol:
        """Add a held-out corpus as an additional benchmark.

        The intended way to admit a future dataset: the returned protocol is
        validated by the caller exactly as the original was, so a new corpus
        that overlaps a fitting role is refused on the same terms as an old one.
        Adding it as *secondary* rather than promoting it is also deliberate —
        the primary is the number that gates releases, and swapping it changes
        the meaning of the whole history.
        """
        return EvaluationProtocol(
            name=self.name,
            assignments=self.assignments
            + (
                RoleAssignment(
                    dataset_id=dataset_id,
                    role=DatasetRole.SECONDARY_BENCHMARK,
                    splits=tuple(splits),
                    note=note,
                ),
            ),
        )

    def describe(self) -> str:
        order = {r: i for i, r in enumerate(DatasetRole)}
        lines = [f"protocol {self.name!r}"]
        for a in sorted(self.assignments, key=lambda x: (order[x.role], x.dataset_id)):
            lines.append("  " + a.describe())
        return "\n".join(lines)


def validate_protocol(protocol: EvaluationProtocol) -> ProtocolReport:
    """Check a protocol against the rules in the module docstring."""
    problems: list[str] = []
    warnings: list[str] = []

    for role in SINGLETON_ROLES:
        found = protocol.by_role(role)
        if len(found) == 0:
            problems.append(
                f"no {role.value} dataset. Every one of {[r.value for r in SINGLETON_ROLES]} "
                "must be assigned; an unassigned role does not disappear, it just "
                "gets filled implicitly by whatever data is at hand."
            )
        elif len(found) > 1:
            problems.append(
                f"{len(found)} datasets claim the {role.value} role "
                f"({sorted(a.dataset_id for a in found)}); exactly one is allowed"
            )

    fitting = protocol.dataset_ids(*FITTING_ROLES)
    benchmark = protocol.dataset_ids(*BENCHMARK_ROLES)

    overlap = sorted(fitting & benchmark)
    for dataset_id in overlap:
        roles = sorted(
            a.role.value for a in protocol.assignments if a.dataset_id == dataset_id
        )
        problems.append(
            f"{dataset_id!r} is both fitted on and benchmarked on (roles: {roles}). "
            "A benchmark corpus must contribute to nothing — not weights, not "
            "checkpoint selection, not the temperature, not the threshold."
        )

    primary = protocol.by_role(DatasetRole.PRIMARY_BENCHMARK)
    train = protocol.by_role(DatasetRole.TRAIN)
    if primary and train and primary[0].dataset_id == train[0].dataset_id:
        problems.append(
            f"the primary benchmark and the training set are both "
            f"{train[0].dataset_id!r}. An in-dataset headline number measures "
            "memorisation of one corpus's artefacts; the primary benchmark must "
            "be a different corpus."
        )

    # Split-level checks inside a single corpus.
    for role, forbidden, why in (
        (
            DatasetRole.VALIDATION,
            Split.TRAIN,
            "the validation set may not use the training split; selecting a "
            "checkpoint on data the weights were fitted to selects the most "
            "overfitted checkpoint",
        ),
        (
            DatasetRole.CALIBRATION,
            Split.TRAIN,
            "the calibration set may not use the training split; logits on "
            "trained-on data are saturated and the temperature fitted to them "
            "is close to 1.0 no matter how overconfident the model is",
        ),
    ):
        for a in protocol.by_role(role):
            if forbidden in a.splits:
                problems.append(f"{a.dataset_id}/{role.value}: {why}")

    calibration = protocol.by_role(DatasetRole.CALIBRATION)
    validation = protocol.by_role(DatasetRole.VALIDATION)
    if calibration and validation:
        cal, val = calibration[0], validation[0]
        if cal.dataset_id == val.dataset_id and set(cal.splits) & set(val.splits):
            shared = sorted(s.value for s in set(cal.splits) & set(val.splits))
            warnings.append(
                f"calibration and validation share {cal.dataset_id} {shared}. "
                "The temperature is then fitted to logits from the split that "
                "chose the checkpoint, which is optimistic; prefer a split that "
                "selected nothing."
            )

    for a in protocol.by_role(DatasetRole.PRIMARY_BENCHMARK):
        if set(a.splits) != set(Split):
            warnings.append(
                f"the primary benchmark uses only {sorted(s.value for s in a.splits)} "
                f"of {a.dataset_id}. A wholly held-out corpus has no reason to "
                "withhold part of itself, and a partial benchmark is a smaller "
                "sample than it looks."
            )

    seen: set[tuple[str, DatasetRole]] = set()
    for a in protocol.assignments:
        key = (a.dataset_id, a.role)
        if key in seen:
            problems.append(f"{a.dataset_id!r} is assigned to {a.role.value} twice")
        seen.add(key)

    return ProtocolReport(problems=problems, warnings=warnings)


class ProtocolViolation(Exception):
    """A run claims a protocol it cannot show it followed."""


@dataclass(frozen=True)
class RoleSelection:
    """Items chosen for one role, carrying evidence of how they were chosen.

    WHY A TYPE RATHER THAN A LIST
    -----------------------------
    Because reporting is not enforcement, and until this existed the two were
    indistinguishable. ``run_benchmark`` accepted ``items`` and ``protocol`` as
    unrelated arguments, so a run could score the *training* set and record
    ``protocol="ffpp-train/celebdf-primary/v1"`` beside it. Nothing was lying;
    nothing was checking either. The protocol was a label on the run, not a
    property of it.

    A selection is the evidence: it names the protocol, the role, the exact
    media ids, and the split lock they were resolved under, and carries a
    ``witness`` digest binding the four together. ``run_benchmark`` records a
    protocol **only** from one of these, and re-derives the witness from the
    items it actually scored before recording anything.

    WHAT THE WITNESS DOES AND DOES NOT GUARANTEE
    --------------------------------------------
    It does not make forgery impossible. Nothing in a Python process can — a
    determined caller can compute a witness with :meth:`compute_witness` and
    construct whatever they like, and claiming otherwise would be the same
    species of false assurance R5 exists to remove.

    What it does is make the claim **verifiable from artifacts**.
    :func:`verify_run_selection` re-derives the selection from the protocol, the
    corpus index and the lock, and compares. A forged or mistaken run fails that
    check, and the check needs nothing that was in the original process. It also
    ends the realistic failure — someone passing ``protocol=`` out of habit —
    because :meth:`__post_init__` rejects a selection whose witness does not
    match its own fields, so a false claim has to be deliberate rather than
    accidental.
    """

    protocol_name: str
    role: DatasetRole
    media_ids: tuple[str, ...]
    lock_hash: str | None
    witness: str
    items: tuple[MediaItem, ...] = ()

    def __post_init__(self) -> None:
        expected = self.compute_witness(
            self.protocol_name, self.role, self.media_ids, self.lock_hash
        )
        if self.witness != expected:
            raise ProtocolViolation(
                f"selection witness {self.witness!r} does not match its own "
                f"fields (expected {expected!r}); this selection was assembled "
                "by hand rather than by select()"
            )
        selected = tuple(sorted(i.media_id for i in self.items))
        if self.items and selected != self.media_ids:
            raise ProtocolViolation(
                "the selection's items do not match the media ids it attests "
                f"to ({len(selected)} item(s) against {len(self.media_ids)} id(s))"
            )

    @staticmethod
    def compute_witness(
        protocol_name: str,
        role: DatasetRole,
        media_ids: tuple[str, ...],
        lock_hash: str | None,
    ) -> str:
        """The binding digest. Sorted ids, so ordering cannot change it."""
        return fingerprint(
            {
                "protocol": protocol_name,
                "role": role.value,
                "media_ids": sorted(media_ids),
                "lock_hash": lock_hash,
            }
        )

    # List-like, so every caller written against the old ``select`` — which
    # returned a plain list — keeps working unchanged.
    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def as_list(self) -> list[MediaItem]:
        return list(self.items)

    def describe(self) -> str:
        return (
            f"{self.role.value} under {self.protocol_name!r}: "
            f"{len(self.media_ids)} item(s), lock "
            f"{self.lock_hash or '(none)'}, witness {self.witness}"
        )


def select(
    protocol: EvaluationProtocol,
    role: DatasetRole,
    items: list[MediaItem],
    assignment: dict[str, Split],
    *,
    lock_hash: str | None = None,
) -> RoleSelection:
    """The items that play ``role``, plus the evidence that they do.

    Takes the assignment as a plain ``media_id -> Split`` mapping so a
    :class:`~evaluation.split_lock.SplitLock`, a ``SplitAssignment.by_media`` or
    a hand-built dict all work without this module importing any of them. Pass
    ``lock_hash`` to bind the selection to a specific frozen split — without it
    the witness attests to the protocol and the items but not to *which* split
    produced them.

    Items with no assignment are excluded rather than defaulted. There is no
    safe default: guessing train poisons the weights and guessing test poisons
    the benchmark.

    **An invalid protocol selects nothing.** Validation runs here, not only
    where a protocol is written down, because a protocol that benchmarks its own
    training corpus is not a document problem — it is a selection that must
    never happen, and the only place to stop it is the moment it is acted on.
    """
    report = validate_protocol(protocol)
    if not report.valid:
        raise ProtocolViolation(
            f"protocol {protocol.name!r} is invalid and cannot select items:\n"
            + report.describe()
        )
    wanted = protocol.by_role(role)
    if not wanted:
        raise ValueError(f"protocol {protocol.name!r} assigns nothing to {role.value}")
    allowed = {(a.dataset_id, s) for a in wanted for s in a.splits}
    chosen = tuple(
        i
        for i in items
        if i.media_id in assignment and (i.dataset_id, assignment[i.media_id]) in allowed
    )
    media_ids = tuple(sorted(i.media_id for i in chosen))
    return RoleSelection(
        protocol_name=protocol.name,
        role=role,
        media_ids=media_ids,
        lock_hash=lock_hash,
        witness=RoleSelection.compute_witness(protocol.name, role, media_ids, lock_hash),
        items=chosen,
    )


def verify_run_selection(
    run: object,
    protocol: EvaluationProtocol,
    role: DatasetRole,
    items: list[MediaItem],
    assignment: dict[str, Split],
    *,
    lock_hash: str | None = None,
) -> list[str]:
    """Re-derive what ``run`` claims and report every disagreement.

    The after-the-fact check, and the one that makes the witness worth having.
    It needs only artifacts — the protocol, the corpus index, the lock — so a
    report can be audited months later by someone who was not there, which is
    the situation the whole reproducibility effort is for.

    Returns a list so a caller learns everything that is wrong at once. Empty
    means the run's protocol claim is corroborated.
    """
    problems: list[str] = []
    claimed_protocol = getattr(run, "protocol", None)
    claimed_witness = getattr(run, "selection_witness", None)

    if claimed_protocol is None:
        problems.append("the run records no protocol, so there is nothing to verify")
    elif claimed_protocol != protocol.name:
        problems.append(
            f"the run claims protocol {claimed_protocol!r}, not {protocol.name!r}"
        )

    if claimed_witness is None:
        problems.append(
            "the run carries no selection witness; it predates protocol "
            "enforcement or bypassed select()"
        )
        return problems

    expected = select(protocol, role, items, assignment, lock_hash=lock_hash)
    if claimed_witness != expected.witness:
        problems.append(
            f"selection witness {claimed_witness} does not match the "
            f"{role.value} selection re-derived from this protocol and corpus "
            f"({expected.witness}); the run scored a different set of items "
            "than the protocol prescribes"
        )
    return problems


# --------------------------------------------------------------------------
# Reporting conventions
# --------------------------------------------------------------------------

REQUIRED_REPORT_FIELDS: tuple[str, ...] = (
    "protocol",
    "effective_model_version",
    "selection_witness",
    "calibration_witness",
    "split_lock_hash",
    "dataset_fingerprint",
    "primary_benchmark",
    "in_domain",
    "threshold",
    "band_width",
    "coverage",
    "selective_risk",
    "selective_fpr",
    "worst_slice",
    "n",
)
"""Every field that must accompany a reported number.

Not a style preference. Each of these has, at some point in this project's
design review, been the thing whose absence made a number unreproducible or
misleading:

    protocol / split_lock_hash / dataset_fingerprint
        what was measured and on which frozen split. Without these a number
        cannot be compared with any other number.
    selection_witness / calibration_witness
        that the protocol was *followed*, not merely named — the items came
        through ``select()``, and the temperature and threshold were fitted on
        the calibration role rather than on the corpus being reported. A
        protocol name alone is a label; these two are the evidence, and a report
        without them cannot be audited by anyone who was not in the room.
    effective_model_version
        weights + calibration + gate. The bare weights version hides a
        recalibration, and a recalibration moves every probability.
    primary_benchmark **and** in_domain, together
        quoting either alone is the standard way to mislead. In-domain alone
        sells a solved problem as an achievement; cross-dataset alone hides that
        the model may simply be weak everywhere.
    threshold / band_width / coverage / selective_risk / selective_fpr
        a three-state verdict has no accuracy. Coverage is how often it answered
        at all, and a system can improve every rate by answering less.
    worst_slice
        the average is not the product. Low resolution and heavy compression are
        where a detector fails and where users need it.
    n
        an AUC without a sample size is not a measurement.
"""


def check_report(payload: dict) -> list[str]:
    """Which required fields a result payload is missing. Empty means complete.

    A list rather than a raise: the caller is usually assembling a report and
    wants to know everything that is absent, not the first thing.
    """
    return [f for f in REQUIRED_REPORT_FIELDS if payload.get(f) in (None, "")]


# --------------------------------------------------------------------------
# The protocol this project actually uses
# --------------------------------------------------------------------------

FFPP_CELEBDF = EvaluationProtocol(
    name="ffpp-train/celebdf-primary/v1",
    assignments=(
        RoleAssignment(
            dataset_id="ffpp",
            role=DatasetRole.TRAIN,
            splits=(Split.TRAIN,),
            note="FF++ c23 and c40 together; both qualities of one source share "
            "an identity component and never straddle the split",
        ),
        RoleAssignment(
            dataset_id="ffpp",
            role=DatasetRole.VALIDATION,
            splits=(Split.VAL,),
            note="checkpoint and hyper-parameter selection only; used up by the "
            "end of training and never reported",
        ),
        RoleAssignment(
            dataset_id="ffpp",
            role=DatasetRole.CALIBRATION,
            splits=(Split.TEST,),
            note="temperature and threshold. FF++ test rather than val because "
            "val selected the checkpoint; this split selected nothing. It is "
            "spent as a benchmark by being used here, which is intended — the "
            "benchmark is Celeb-DF.",
        ),
        RoleAssignment(
            dataset_id="celebdf",
            role=DatasetRole.PRIMARY_BENCHMARK,
            splits=tuple(Split),
            note="the headline. Entirely held out; target AUC 0.90-0.93.",
        ),
        RoleAssignment(
            dataset_id="dfdc",
            role=DatasetRole.SECONDARY_BENCHMARK,
            splits=tuple(Split),
            note="harder and noisier; target AUC 0.75-0.80",
        ),
    ),
)
"""The frozen protocol for this platform.

The interesting choice is calibration on FF++ **test** rather than FF++ val: it
buys a temperature fitted to logits from a split that selected no checkpoint,
where val's logits are optimistic because val chose the checkpoint.

What that costs is narrower than it first looks, and the distinction is worth
keeping straight. Temperature scaling is monotone, so FF++ test **AUC** is
untouched by having been calibrated on and remains a legitimate in-domain
ranking number — which is what ``in_domain`` in a report means. What is spent is
every *threshold-dependent* number on that split: coverage, selective risk and
selective FPR measured there are the best case, because the threshold was chosen
to make them so. Report those from the benchmark corpora only.
"""
