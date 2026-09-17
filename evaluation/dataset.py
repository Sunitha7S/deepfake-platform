"""Dataset identity, and splits that cannot leak.

THE FOUR IDENTITIES
-------------------
A deepfake corpus is not a bag of files. Every item carries four identities and
confusing any two of them is how a benchmark comes to report a number that
production will not reproduce:

    dataset_id   which corpus it came from. FF++ and Celeb-DF number their
                 subjects from zero independently, so a subject id means
                 nothing without it.

    source_id    the *original* media. Every manipulated video in FF++ is
                 derived from a real one; c23 and c40 are the same footage at
                 two compression levels. All of them share a source.

    media_id     this particular file. The derived identity — one row in the
                 index, one thing the model scores.

    subject_id   the person depicted, where the corpus records it. One actor
                 appears in many source videos, so subject identity spans
                 sources and is the coarsest grouping of the four.

WHY LEAKAGE IS PREVENTED BY CONSTRUCTION RATHER THAN CHECKED FOR
----------------------------------------------------------------
Splitting a corpus row by row puts frame 12 of a video in train and frame 88 of
the same video in test. The model then scores test items it has effectively
memorised, and reported AUC climbs toward 1.0 while cross-dataset performance
does not move at all. The failure is silent, flattering, and survives every
downstream check — which is exactly why it must be made unrepresentable rather
than detected afterwards.

So the unit of assignment is not the item. Items are linked into **identity
components** by union-find — two items are linked if they share a source, or if
they share a subject — and a whole component goes to exactly one split. The
transitivity matters and is the reason for union-find rather than a groupby: a
manipulated clip and its original share a source; two originals of the same
actor share a subject; so all four end up in one component and move together.

``detect_leakage`` exists for splits this module did not produce — a lock file
from an earlier run, a published split, a hand edit — and reports what leaked,
between which splits, and with which ids, because "leakage detected" on its own
tells nobody what to fix.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


SubjectScope = Literal["dataset", "global"]


# --------------------------------------------------------------------------
# Slice metadata
# --------------------------------------------------------------------------
#
# A headline AUC is an average, and an average hides the only failures that
# matter in production. The three axes below are the ones a deepfake detector
# actually collapses along, so they are first-class fields rather than a free
# ``dict`` of tags: a typo'd key in a tag bag produces an empty slice, and an
# empty slice is silently excluded from the two-scope regression gate — the
# model looks fine precisely because the check did not run.


class Compression(StrEnum):
    """How much lossy re-encoding the file has been through.

    Normalised rather than corpus-native ("c23", "crf28", "quality 75") because
    the point of a slice is to compare *across* corpora: FF++ c40 and a
    WhatsApp re-upload are the same regime, and a slice keyed on the raw label
    would never put them in the same bucket. The mapping from a corpus's own
    vocabulary lives in the ``from_*`` constructors, where it can be read.

    ``UNKNOWN`` is a real state and is not the same as ``None``. ``None`` means
    nobody recorded it; ``UNKNOWN`` means someone looked and could not tell.
    """

    NONE = "none"
    """Original or visually lossless. FF++ raw/c0."""
    LIGHT = "light"
    """Ordinary delivery quality. FF++ c23, most platform uploads."""
    HEAVY = "heavy"
    """Aggressive re-encoding, where residual-based branches degrade first.
    FF++ c40, repeated re-shares."""
    UNKNOWN = "unknown"

    @classmethod
    def from_ffpp(cls, quality: str) -> Compression:
        """Map FF++'s own quality labels onto the normalised scale."""
        table = {
            "raw": cls.NONE,
            "c0": cls.NONE,
            "c23": cls.LIGHT,
            "c40": cls.HEAVY,
        }
        key = quality.strip().lower()
        if key not in table:
            raise ValueError(
                f"unknown FF++ quality {quality!r}; expected one of "
                f"{sorted(table)}"
            )
        return table[key]

    @classmethod
    def from_crf(cls, crf: int) -> Compression:
        """Map an x264/x265 CRF onto the normalised scale.

        Boundaries chosen to line up with FF++: c23 is CRF 23 and c40 is CRF
        40, so ``<= 17`` is visually lossless, ``<= 30`` is the light regime and
        anything above is the heavy one.
        """
        if crf < 0:
            raise ValueError(f"crf must be non-negative, got {crf}")
        if crf <= 17:
            return cls.NONE
        if crf <= 30:
            return cls.LIGHT
        return cls.HEAVY


class ResolutionBucket(StrEnum):
    """Coarse resolution class, keyed on the **short side** of the frame.

    Short side rather than pixel count or width: a face crop's usable detail is
    set by the smaller dimension, and a 2560x1080 ultrawide has no more facial
    detail than 1920x1080 while having 33% more pixels. Bucketing on area would
    call it the better sample.
    """

    LOW = "low"
    """Below 480. Face crops here are usually upsampled, and every
    residual-domain feature the model relies on is already gone."""
    SD = "sd"
    HD = "hd"
    FHD = "fhd"
    UHD = "uhd"
    UNKNOWN = "unknown"

    @classmethod
    def from_short_side(cls, pixels: int) -> ResolutionBucket:
        if pixels <= 0:
            raise ValueError(f"short side must be positive, got {pixels}")
        if pixels < 480:
            return cls.LOW
        if pixels < 720:
            return cls.SD
        if pixels < 1080:
            return cls.HD
        if pixels < 1440:
            return cls.FHD
        return cls.UHD

    @classmethod
    def from_frame(cls, width: int, height: int) -> ResolutionBucket:
        return cls.from_short_side(min(width, height))


SLICE_DIMENSIONS = ("dataset", "compression", "resolution", "generator")
"""Every axis ``MediaItem.slice_value`` understands.

``dataset`` is included even though it is not one of the optional fields: which
corpus an item came from is the most important slice of all in a cross-dataset
protocol, and a caller asking to slice by it should not have to special-case it.
"""


def _coerce(enum_cls, value, field_name: str):
    """Accept the enum, the enum's value, or None. Refuse anything else loudly."""
    if value is None or isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            raise ValueError(
                f"{field_name}={value!r} is not a known "
                f"{enum_cls.__name__}; expected one of "
                f"{[m.value for m in enum_cls]} or None"
            ) from None
    raise ValueError(
        f"{field_name} must be a {enum_cls.__name__}, one of its string "
        f"values, or None; got {type(value).__name__}"
    )


@dataclass(frozen=True)
class MediaItem:
    """One scoreable file, with everything needed to place it safely.

    The first five fields are identity and are required. The three after them
    are **slice metadata**: optional, defaulted, and appended rather than
    inserted, so every existing positional construction
    (``MediaItem("a", "s1", "ffpp", 0)``) keeps working unchanged. Serialisation
    omits them when unset, so an index written before they existed and one
    written after fingerprint identically.
    """

    media_id: str
    source_id: str
    dataset_id: str
    label: int
    subject_id: str | None = None
    compression: Compression | None = None
    resolution: ResolutionBucket | None = None
    generator: str | None = None
    """Which manipulation produced it — "Deepfakes", "FaceSwap", "sdxl-1.0" —
    or ``None`` for authentic media.

    Deliberately a free string and not an enum. Generators appear faster than
    this file can be edited, and an enum here would mean a new one either
    crashes the index or gets recorded as ``OTHER``, which is the same as not
    recording it. The cost is that spelling is the caller's problem;
    ``slice_counts`` exists partly so that a typo shows up as a slice of one.
    """

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label!r}")
        for name in ("media_id", "source_id", "dataset_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")

        # Coerce strings to their enums at the door. Callers read these values
        # out of a CSV or a JSON index, where everything is a string, and a
        # frozen dataclass holding the string "c40" in a field annotated
        # ``Compression`` would compare unequal to ``Compression.HEAVY`` and
        # split into two slices that look like two regimes.
        object.__setattr__(self, "compression", _coerce(Compression, self.compression, "compression"))
        object.__setattr__(self, "resolution", _coerce(ResolutionBucket, self.resolution, "resolution"))
        if self.generator is not None:
            generator = self.generator.strip()
            if not generator:
                raise ValueError(
                    "generator must be a non-empty string or None; an empty "
                    "string would form a slice with no name"
                )
            object.__setattr__(self, "generator", generator)

    def source_key(self) -> str:
        """Sources are always dataset-scoped. Two corpora may reuse a filename."""
        return f"{self.dataset_id}::src::{self.source_id}"

    def subject_key(self, scope: SubjectScope = "dataset") -> str | None:
        """Subject key, or None when the corpus records no subject.

        Scope is explicit because it is a real modelling decision. Corpora
        number their subjects independently, so ``dataset`` scope is the safe
        default; ``global`` is correct only when the caller has resolved
        identities across corpora and can promise that ``subject_id`` means the
        same person everywhere. Getting this wrong in the ``global`` direction
        merges strangers into one component and needlessly shrinks the split;
        getting it wrong the other way lets the same actor appear on both sides
        of a cross-dataset evaluation.
        """
        if self.subject_id is None:
            return None
        if scope == "global":
            return f"subj::{self.subject_id}"
        return f"{self.dataset_id}::subj::{self.subject_id}"

    # -- slice metadata ----------------------------------------------------

    def slice_value(self, dimension: str) -> str | None:
        """This item's value on one slice axis, or ``None`` if unrecorded."""
        if dimension not in SLICE_DIMENSIONS:
            raise ValueError(
                f"unknown slice dimension {dimension!r}; expected one of "
                f"{list(SLICE_DIMENSIONS)}"
            )
        if dimension == "dataset":
            return self.dataset_id
        value = getattr(self, dimension)
        return str(value) if value is not None else None

    def slice_values(self) -> dict[str, str]:
        """Every recorded axis. Unrecorded axes are absent, not ``"None"``."""
        return {
            d: v
            for d in SLICE_DIMENSIONS
            if (v := self.slice_value(d)) is not None
        }

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Plain JSON-safe mapping.

        Unset optional fields are **omitted, not written as null**. That is what
        makes this backward compatible in the direction that matters: an index
        built before slice metadata existed and the same index round-tripped
        through this method produce byte-identical JSON, so no dataset
        fingerprint and no split-lock hash moves when the fields land.
        """
        out: dict[str, object] = {
            "media_id": self.media_id,
            "source_id": self.source_id,
            "dataset_id": self.dataset_id,
            "label": self.label,
        }
        for name in ("subject_id", "compression", "resolution", "generator"):
            value = getattr(self, name)
            if value is not None:
                out[name] = str(value)
        return out

    @classmethod
    def from_dict(cls, payload: dict) -> MediaItem:
        """Rebuild from :meth:`to_dict`.

        Unknown keys are refused rather than ignored. A misspelled
        ``"resolution_bucket"`` silently dropped would produce an item that
        looks fine, slices into nothing, and takes its dimension out of the
        regression gate without any error being raised anywhere.
        """
        known = {
            "media_id",
            "source_id",
            "dataset_id",
            "label",
            "subject_id",
            "compression",
            "resolution",
            "generator",
        }
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(
                f"unknown MediaItem field(s) {unknown}; known fields are "
                f"{sorted(known)}"
            )
        missing = sorted({"media_id", "source_id", "dataset_id", "label"} - set(payload))
        if missing:
            raise ValueError(f"MediaItem payload is missing {missing}")
        return cls(
            media_id=payload["media_id"],
            source_id=payload["source_id"],
            dataset_id=payload["dataset_id"],
            label=int(payload["label"]),
            subject_id=payload.get("subject_id"),
            compression=payload.get("compression"),
            resolution=payload.get("resolution"),
            generator=payload.get("generator"),
        )


# --------------------------------------------------------------------------
# Union-find over identity
# --------------------------------------------------------------------------


class _UnionFind:
    """Path-compressed union-find keyed by arbitrary strings."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def add(self, key: str) -> None:
        if key not in self._parent:
            self._parent[key] = key
            self._rank[key] = 0

    def find(self, key: str) -> str:
        self.add(key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:  # path compression
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


@dataclass(frozen=True)
class IdentityComponent:
    """One indivisible group. Everything here goes to the same split."""

    key: str
    media_ids: tuple[str, ...]
    n_real: int
    n_fake: int

    @property
    def size(self) -> int:
        return len(self.media_ids)


def build_components(
    items: list[MediaItem], *, subject_scope: SubjectScope = "dataset"
) -> list[IdentityComponent]:
    """Group items into indivisible identity components.

    Linked by shared source and by shared subject, transitively. Components come
    back in a deterministic order — by descending size, then by key — so that
    everything downstream is reproducible without a sort of its own.
    """
    uf = _UnionFind()
    for item in items:
        uf.add(item.media_id)
        uf.union(item.media_id, item.source_key())
        subject = item.subject_key(subject_scope)
        if subject is not None:
            uf.union(item.media_id, subject)

    grouped: dict[str, list[MediaItem]] = defaultdict(list)
    for item in items:
        grouped[uf.find(item.media_id)].append(item)

    components = []
    for members in grouped.values():
        ids = tuple(sorted(m.media_id for m in members))
        components.append(
            IdentityComponent(
                # Named by the smallest member id rather than by the union-find
                # root, which is an implementation detail and would change if
                # the union order changed.
                key=ids[0],
                media_ids=ids,
                n_real=sum(1 for m in members if m.label == 0),
                n_fake=sum(1 for m in members if m.label == 1),
            )
        )
    return sorted(components, key=lambda c: (-c.size, c.key))


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------


def tiebreak_key(seed: str, key: str) -> str:
    """Stable, seeded ordering for equal-sized components.

    ``hashlib`` rather than ``hash()``: Python randomises string hashing per
    interpreter, so a seeded split built on ``hash()`` would produce a different
    test set on every process start while looking reproducible.

    Module-level rather than a method because the split lock has to place new
    components in exactly the order the original split would have used. Two
    implementations of one ordering rule is how a lock and its splitter drift
    apart, and the drift would be invisible until a regenerated lock disagreed
    with the frozen one.
    """
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


@dataclass(frozen=True)
class SplitAssignment:
    """A split, plus the evidence that it is sound."""

    by_media: dict[str, Split]
    components: list[IdentityComponent]
    achieved: dict[Split, float]
    counts: dict[Split, int]
    seed: str
    ratios: dict[Split, float]
    """What was asked for, kept beside what was achieved. A split that missed
    its target is not a bug, but nobody can tell without both numbers."""

    def of(self, media_id: str) -> Split:
        return self.by_media[media_id]

    def media_in(self, split: Split) -> list[str]:
        return sorted(m for m, s in self.by_media.items() if s is split)

    @property
    def n_components(self) -> int:
        return len(self.components)


class IdentitySplitter:
    """Assigns whole identity components to splits.

    Greedy largest-first: components are placed in descending size order, each
    into whichever split is furthest below its quota. Deterministic given the
    same items and seed, and it tracks the requested ratios closely even when a
    few components are much larger than the rest — which is the normal shape of
    a deepfake corpus, where one source video yields dozens of derived files.

    The trade worth naming: this is **not stable under growth**. Adding items
    can move existing components between splits, so a frozen test set must be
    written to a lock file and read back, never re-derived. That is what
    ``SplitAssignment`` is for and why it records its seed.
    """

    def __init__(
        self,
        *,
        ratios: dict[Split, float] | None = None,
        seed: str = "dfp-v1",
        subject_scope: SubjectScope = "dataset",
    ) -> None:
        self.ratios = ratios or {Split.TRAIN: 0.7, Split.VAL: 0.15, Split.TEST: 0.15}
        total = sum(self.ratios.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        if any(v < 0 for v in self.ratios.values()):
            raise ValueError("split ratios must be non-negative")
        self.seed = seed
        self.subject_scope = subject_scope

    def _tiebreak(self, key: str) -> str:
        return tiebreak_key(self.seed, key)

    def split(self, items: list[MediaItem]) -> SplitAssignment:
        if not items:
            raise ValueError("cannot split an empty item list")
        seen = Counter(i.media_id for i in items)
        duplicates = [m for m, n in seen.items() if n > 1]
        if duplicates:
            raise ValueError(
                f"duplicate media_id in the input: {sorted(duplicates)[:5]}. "
                "Each file must appear once; a repeated id would be assigned "
                "twice and could land in two splits."
            )

        components = build_components(items, subject_scope=self.subject_scope)
        ordered = sorted(components, key=lambda c: (-c.size, self._tiebreak(c.key)))

        n_total = len(items)
        quota = {s: r * n_total for s, r in self.ratios.items()}
        placed = {s: 0 for s in self.ratios}
        by_media: dict[str, Split] = {}

        for component in ordered:
            # Whichever split is furthest below its quota takes the next
            # component. Ties break on the split name so the result does not
            # depend on dict ordering.
            target = min(
                self.ratios,
                key=lambda s: (placed[s] - quota[s], s.value),
            )
            for media_id in component.media_ids:
                by_media[media_id] = target
            placed[target] += component.size

        return SplitAssignment(
            by_media=by_media,
            components=components,
            achieved={s: placed[s] / n_total for s in self.ratios},
            counts=dict(placed),
            seed=self.seed,
            ratios=dict(self.ratios),
        )


# --------------------------------------------------------------------------
# Detection, for splits this module did not produce
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakageFinding:
    """One identity that spans splits, with enough detail to act on it."""

    kind: Literal["media", "source", "subject"]
    key: str
    splits: tuple[Split, ...]
    media_ids: tuple[str, ...]

    def describe(self) -> str:
        where = " and ".join(s.value for s in self.splits)
        examples = ", ".join(self.media_ids[:3])
        more = f" (+{len(self.media_ids) - 3} more)" if len(self.media_ids) > 3 else ""
        return (
            f"{self.kind} {self.key!r} appears in {where}: {examples}{more}"
        )


@dataclass(frozen=True)
class LeakageReport:
    findings: list[LeakageFinding] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    def by_kind(self, kind: str) -> list[LeakageFinding]:
        return [f for f in self.findings if f.kind == kind]

    def describe(self) -> str:
        if self.clean:
            return "no leakage: every identity is confined to one split"
        lines = [f"{len(self.findings)} identity/identities span more than one split:"]
        lines += [f"  - {f.describe()}" for f in self.findings[:20]]
        if len(self.findings) > 20:
            lines.append(f"  … and {len(self.findings) - 20} more")
        return "\n".join(lines)

    def __bool__(self) -> bool:
        """Truthy when there IS leakage.

        Deliberately not "truthy when clean". ``if detect_leakage(...):`` should
        read as "if there is leakage", which is the check a caller wants to
        write, and the inversion would make every such guard silently backwards.
        """
        return bool(self.findings)


def detect_leakage(
    items: list[MediaItem],
    assignment: dict[str, Split] | SplitAssignment,
    *,
    subject_scope: SubjectScope = "dataset",
) -> LeakageReport:
    """Find every identity that appears in more than one split.

    Reports all three kinds separately rather than stopping at the first,
    because they have different causes and different fixes: a repeated
    ``media_id`` is an index bug, a spanning source is a row-wise split, and a
    spanning subject is a split that grouped by file rather than by person.
    """
    by_media = (
        assignment.by_media if isinstance(assignment, SplitAssignment) else assignment
    )

    buckets: dict[tuple[str, str], dict[Split, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in items:
        split = by_media.get(item.media_id)
        if split is None:
            continue  # not assigned; out of scope for a leakage question
        buckets[("media", item.media_id)][split].append(item.media_id)
        buckets[("source", item.source_key())][split].append(item.media_id)
        subject = item.subject_key(subject_scope)
        if subject is not None:
            buckets[("subject", subject)][split].append(item.media_id)

    findings: list[LeakageFinding] = []
    for (kind, key), per_split in sorted(buckets.items()):
        if len(per_split) < 2:
            continue
        media_ids = sorted({m for ms in per_split.values() for m in ms})
        findings.append(
            LeakageFinding(
                kind=kind,  # type: ignore[arg-type]
                key=key,
                splits=tuple(sorted(per_split, key=lambda s: s.value)),
                media_ids=tuple(media_ids),
            )
        )
    # Coarsest first: a spanning subject usually explains the spanning sources
    # underneath it, and fixing it fixes them.
    order = {"subject": 0, "source": 1, "media": 2}
    findings.sort(key=lambda f: (order[f.kind], f.key))
    return LeakageReport(findings=findings)


def class_balance(items: list[MediaItem], assignment: SplitAssignment) -> dict[Split, dict[str, int]]:
    """Real/fake counts per split. A split can be leak-free and still useless."""
    out: dict[Split, dict[str, int]] = {
        s: {"real": 0, "fake": 0, "total": 0} for s in Split
    }
    for item in items:
        split = assignment.by_media.get(item.media_id)
        if split is None:
            continue
        out[split]["fake" if item.label == 1 else "real"] += 1
        out[split]["total"] += 1
    return out


@dataclass(frozen=True)
class SplitViability:
    """Whether a leak-free split is also a *usable* one.

    Leakage and viability are different failures and only one of them is loud.
    A corpus with three actors has three indivisible components, so a three-way
    split cannot balance: one fold ends up empty. That split is perfectly
    leak-free and completely useless — calibration would fit a temperature on
    nothing and report success. This is the check that says so.
    """

    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.problems

    def describe(self) -> str:
        if self.usable and not self.warnings:
            return "split is usable: every fold is populated and has both classes"
        lines = []
        lines += [f"  PROBLEM  {p}" for p in self.problems]
        lines += [f"  warning  {w}" for w in self.warnings]
        return "\n".join(lines)


MIN_CALIBRATION_NEGATIVES = 500
"""Below this, an extreme quantile is not a threshold — it is an accident.

``threshold_at_fpr`` at a 2% budget is the 98th percentile of the authentic
class. A stable extreme quantile needs roughly ten samples beyond it, so 500
negatives is the floor and 2000 is the working number.
"""


def assess_split(
    items: list[MediaItem],
    assignment: SplitAssignment,
    *,
    min_calibration_negatives: int = MIN_CALIBRATION_NEGATIVES,
) -> SplitViability:
    """Report the ways a split can be sound and still not fit for purpose."""
    balance = class_balance(items, assignment)
    problems: list[str] = []
    warnings: list[str] = []

    for split in Split:
        counts = balance[split]
        if counts["total"] == 0:
            problems.append(
                f"{split.value} is empty. The corpus has "
                f"{assignment.n_components} identity component(s) and they "
                "cannot be divided further without leaking — add subjects or "
                "sources, or use fewer folds."
            )
            continue
        if counts["real"] == 0 or counts["fake"] == 0:
            present = "fake" if counts["real"] == 0 else "real"
            problems.append(
                f"{split.value} contains only {present} items, so every "
                "threshold-free metric on it is undefined."
            )

    test_negatives = balance[Split.TEST]["real"]
    if 0 < test_negatives < min_calibration_negatives:
        warnings.append(
            f"test has {test_negatives} authentic items; a false-positive "
            f"budget needs at least {min_calibration_negatives} for the "
            "quantile to be stable."
        )

    largest = assignment.components[0] if assignment.components else None
    if largest is not None and len(items) and largest.size / len(items) > 0.4:
        warnings.append(
            f"the largest identity component holds {largest.size} of "
            f"{len(items)} items ({largest.size / len(items):.0%}); ratios "
            "cannot be met closely when one block dominates."
        )

    for split, achieved in assignment.achieved.items():
        target = assignment.ratios.get(split, 0.0)
        if target and abs(achieved - target) > 0.1:
            warnings.append(
                f"{split.value} achieved {achieved:.0%} against a {target:.0%} "
                "target; indivisible components forced the drift."
            )

    return SplitViability(problems=problems, warnings=warnings)


# --------------------------------------------------------------------------
# Slicing
# --------------------------------------------------------------------------


def group_by_slice(
    items: list[MediaItem], dimension: str, *, include_unrecorded: bool = False
) -> dict[str, list[MediaItem]]:
    """Bucket items by one slice axis, ready for ``run_benchmark(slice_by=...)``.

    Items with nothing recorded on that axis are dropped by default. They are
    not a slice — "resolution unknown" is a gap in the index, and reporting it
    as a cohort invites a reader to draw a conclusion about a group whose only
    shared property is that nobody measured it. ``include_unrecorded`` puts them
    in an explicit ``"unrecorded"`` bucket when the gap itself is the subject.

    Keys come back sorted so the caller's slice ordering — and therefore the
    ordering of a rendered report — does not depend on corpus ordering.
    """
    buckets: dict[str, list[MediaItem]] = defaultdict(list)
    for item in items:
        value = item.slice_value(dimension)
        if value is None:
            if not include_unrecorded:
                continue
            value = "unrecorded"
        buckets[value].append(item)
    return {k: buckets[k] for k in sorted(buckets)}


def slice_counts(items: list[MediaItem], dimension: str) -> dict[str, dict[str, int]]:
    """Per-slice real/fake/total counts, including the unrecorded remainder.

    The remainder is always shown here — this is the coverage report for the
    index itself, and a dimension recorded on 4% of the corpus should be
    visibly useless rather than quietly producing a confident slice AUC.
    """
    out: dict[str, dict[str, int]] = {}
    for name, group in group_by_slice(
        items, dimension, include_unrecorded=True
    ).items():
        out[name] = {
            "real": sum(1 for i in group if i.label == 0),
            "fake": sum(1 for i in group if i.label == 1),
            "total": len(group),
        }
    return out
