"""Frozen split assignments: the file that stops a benchmark from drifting.

WHY A LOCK FILE EXISTS AT ALL
-----------------------------
``IdentitySplitter`` is deterministic — same items, same seed, same split — and
that is not the same thing as stable. It places whole identity components
largest-first into whichever fold is furthest below quota, so **adding items
moves existing ones**. Acquire 200 more Celeb-DF clips and a component that was
in test last month can be in train today. Nothing errors. Every test still
passes. The only symptom is that this quarter's benchmark number is not
comparable with last quarter's, and there is no artefact anywhere that says so.

That is benchmark drift, and it is the failure this module exists to make
impossible. The split is computed **once**, written to disk, and read back
forever after. Regeneration is a fallback for a lost file, not the normal path.

WHAT THE LOCK GUARANTEES
------------------------
1. **Old assignments never move.** Growth may add, never reassign. This is the
   whole contract and every other property is subordinate to it.
2. **New items inherit their component's split.** A new c40 encode of a video
   already in test goes to test — not to whichever fold is emptiest. Placing it
   by quota would put the same footage on both sides and leak.
3. **The file is self-verifying.** A content hash over the canonical body
   catches truncation, a hand edit, and a merge conflict marker.
4. **Versioned.** The format carries its own version and a generation counter,
   and refuses to load anything it does not understand rather than guessing.

THE ONE CASE THE LOCK CANNOT SOLVE
----------------------------------
A new item can *bridge* two frozen components — an actor who was only ever in
FF++'s ``v3`` clips turns up in a ``v9`` clip, and ``v3`` is in train while
``v9`` is in test. Union-find now says all of it is one identity. The two
guarantees then contradict each other: honouring the freeze keeps that identity
in two splits (leakage), and removing the leakage moves frozen items.

There is no resolution inside the lock, so it does not invent one. The default
is to **refuse** — ``on_conflict="error"`` — and hand the caller the identities
involved, because the decision is theirs and it is a decision about the corpus,
not about a file format. ``"quarantine"`` is the escape hatch for a corpus that
must keep growing: the bridging items are left unassigned and named, the frozen
ones stay put, and ``verify_lock`` reports the resulting leakage every time it
runs so the compromise cannot be forgotten.

WHAT IS DELIBERATELY *NOT* IN THE FILE
--------------------------------------
Timestamps. A ``created_at`` field would make two locks built from the same
corpus differ byte for byte, which destroys the cheapest possible reproducibility
check — regenerate, diff, expect silence — and turns "identical corpus produces
an identical lock" into an assertion nobody can write. Provenance lives in
``generation`` and ``parent_hash``, which chain without ever varying for the same
inputs. Put the timestamp in the commit, where it belongs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from contracts.version import fingerprint
from evaluation.dataset import (
    IdentityComponent,
    LeakageReport,
    MediaItem,
    Split,
    SplitAssignment,
    SubjectScope,
    build_components,
    detect_leakage,
    tiebreak_key,
)

LOCK_FORMAT = "dfp-split-lock"
"""Format marker. Present so a JSON file that is not a lock is rejected as
'not a lock' rather than as 'a corrupt lock'."""

LOCK_VERSION = "1.1.0"
"""Version of the *lock format*, independent of ``CONTRACT_VERSION``.

Deliberately its own axis. The lock is an offline artefact that no serving
component reads, and coupling it to the frozen API contract version would mean
either bumping a public contract to add a lock field or never being able to
evolve the lock at all.

1.0.0 sealed assignments only. 1.1.0 adds identity and content sealing; see
:data:`SEALING_V1`.
"""

SUPPORTED_LOCK_VERSIONS = ("1.1.0",)
"""What a *constructed* lock may claim. Older files are migrated on load."""

SEALING_UNSEALED = "unsealed"
"""A lock that seals its assignments and nothing about the items behind them.

Every 1.0.0 lock, and any 1.1.0 lock built without items. Loads, verifies its
assignments, and cannot answer the identity question at all — which it says,
rather than reporting a clean audit it has no basis for.
"""

SEALING_V1 = "identity-v1"
"""Per-item digests over the identity fields and the content fields."""

IDENTITY_FIELDS = ("dataset_id", "source_id", "subject_id")
"""**Exactly** what determines which component an item lands in.

Traced from ``build_components``: it unions ``media_id`` with ``source_key()``
(``dataset_id::src::source_id``) and ``subject_key(scope)``
(``dataset_id::subj::subject_id``). Nothing else enters the union-find. So these
three fields plus ``media_id`` and ``subject_scope`` are the complete input to
split membership, and a change to any of them can silently redefine the split
while leaving every assignment where it was.

Measured on a 24-item corpus: relabelling one subject merged two components,
6 -> 5, with ``corpus_fingerprint`` and ``content_hash`` both unchanged.
"""

CONTENT_FIELDS = ("label", "compression", "resolution", "generator")
"""What determines the *meaning* of a split without touching its membership.

Sealed separately, and the separation is the point. Compression and resolution
do not enter the union-find at all — relabelling them cannot leak a split — but
they define the slices the two-scope regression gate compares, so a silent
re-bucketing takes a scope out of the gate. ``label`` is here for the starkest
version: flipping labels changes every metric ever computed and moves no
component.

An identity failure means the split may leak. A content failure means the split
is sound and the numbers mean something else. Reporting them as one verdict
would send the reader to the wrong problem.
"""

ConflictPolicy = Literal["error", "quarantine"]


# --------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------


class SplitLockError(Exception):
    """Base for everything this module refuses to do."""


class LockCorrupted(SplitLockError):
    """The file is not a well-formed lock: bad JSON, missing or malformed fields."""


class LockHashMismatch(SplitLockError):
    """The body is well-formed but does not hash to its recorded digest.

    Distinct from :class:`LockCorrupted` on purpose. Malformed means somebody's
    tooling wrote it wrong; a hash mismatch on a *valid* body means the content
    was edited after it was sealed, which is a different conversation.
    """


class LockVersionMismatch(SplitLockError):
    """The lock was written by a format version this build cannot read."""


class LockIdentityDrift(SplitLockError):
    """A sealed item's identity or content metadata changed under the lock.

    Raised by :func:`extend_lock`, which would otherwise **launder** the change:
    it recomputes digests over the corpus it is given, so re-sealing a tampered
    index would produce a perfectly valid lock attesting to the tampered
    identity. The seal has to be checked before it is renewed.
    """


class LockMergeConflict(SplitLockError):
    """A new item bridges two frozen components; freeze and leak-freedom collide."""

    def __init__(self, conflicts: list[MergeConflict]):
        self.conflicts = conflicts
        detail = "\n".join(f"  - {c.describe()}" for c in conflicts[:10])
        more = f"\n  … and {len(conflicts) - 10} more" if len(conflicts) > 10 else ""
        super().__init__(
            f"{len(conflicts)} new item(s) bridge identity components that are "
            f"frozen into different splits:\n{detail}{more}\n"
            "Honouring the lock would leave one identity in two splits. Either "
            "drop the bridging items, correct the identity metadata that links "
            "them, or extend with on_conflict='quarantine' to set them aside "
            "and keep the frozen assignments."
        )


# --------------------------------------------------------------------------
# The lock
# --------------------------------------------------------------------------


def identity_digest(item: MediaItem) -> str:
    """Short digest of the fields that decide which component ``item`` joins."""
    return fingerprint({f: getattr(item, f) for f in IDENTITY_FIELDS})


def content_digest(item: MediaItem) -> str:
    """Short digest of the fields that decide what a measurement over ``item`` means."""
    return fingerprint(
        {f: (str(v) if (v := getattr(item, f)) is not None else None) for f in CONTENT_FIELDS}
    )


def seal(items: list[MediaItem], media_ids: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Identity and content digests for the covered items, keyed by media id.

    Per item rather than one aggregate over the corpus. An aggregate is cheaper
    and answers only "something changed"; this codebase's whole posture is that
    a detector which cannot name the offender makes somebody grep for it.
    Per-item digests also survive growth without ambiguity — a corpus-wide hash
    changes when items are *added*, which is legitimate, so it could never
    distinguish growth from tampering.
    """
    identity: dict[str, str] = {}
    content: dict[str, str] = {}
    for item in items:
        if item.media_id not in media_ids:
            continue
        identity[item.media_id] = identity_digest(item)
        content[item.media_id] = content_digest(item)
    return identity, content


@dataclass(frozen=True)
class MetadataDrift:
    """One item whose sealed metadata no longer matches the corpus."""

    media_id: str
    kind: Literal["identity", "content"]
    sealed: str
    actual: str

    def describe(self) -> str:
        consequence = (
            "the split may no longer be leak-free"
            if self.kind == "identity"
            else "slices and metrics measured under this lock mean something else"
        )
        return (
            f"{self.media_id!r} {self.kind} digest {self.sealed} -> {self.actual}; "
            f"{consequence}"
        )


@dataclass(frozen=True)
class MergeConflict:
    """One identity component whose frozen members disagree about their split."""

    component_key: str
    splits: tuple[Split, ...]
    frozen_media_ids: tuple[str, ...]
    new_media_ids: tuple[str, ...]

    def describe(self) -> str:
        where = " and ".join(s.value for s in self.splits)
        new = ", ".join(self.new_media_ids[:3]) or "(none)"
        return (
            f"component {self.component_key!r} spans {where}; "
            f"new item(s) {new} link them"
        )


@dataclass(frozen=True)
class SplitLock:
    """A frozen assignment of media to splits, plus how it was produced.

    Immutable by design: :func:`extend_lock` returns a *new* lock with the
    generation advanced and ``parent_hash`` pointing at the one it grew from.
    In-place mutation of a lock is how the frozen set stops being frozen.
    """

    assignments: dict[str, Split]
    seed: str
    ratios: dict[Split, float]
    subject_scope: SubjectScope = "dataset"
    generation: int = 0
    parent_hash: str | None = None
    quarantined: tuple[str, ...] = ()
    lock_version: str = LOCK_VERSION
    identity_digests: dict[str, str] = field(default_factory=dict)
    """Per assigned media id, a digest over :data:`IDENTITY_FIELDS`.

    Empty means unsealed — a 1.0.0 lock, or one built without items. Not the
    same as "sealed and everything matched"; :class:`LockAudit` keeps the
    distinction, because a lock that cannot check its identity must not report a
    clean identity.
    """
    content_digests: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lock_version not in SUPPORTED_LOCK_VERSIONS:
            raise LockVersionMismatch(
                f"lock_version {self.lock_version!r} is not supported by this "
                f"build (supports {list(SUPPORTED_LOCK_VERSIONS)})"
            )
        if self.generation < 0:
            raise LockCorrupted(f"generation must be >= 0, got {self.generation}")
        overlap = sorted(set(self.quarantined) & set(self.assignments))
        if overlap:
            raise LockCorrupted(
                f"media {overlap[:5]} are both assigned and quarantined; a "
                "quarantined item is by definition unassigned"
            )
        for name, digests in (
            ("identity_digests", self.identity_digests),
            ("content_digests", self.content_digests),
        ):
            if digests and set(digests) != set(self.assignments):
                extra = sorted(set(digests) - set(self.assignments))[:3]
                absent = sorted(set(self.assignments) - set(digests))[:3]
                raise LockCorrupted(
                    f"{name} must cover exactly the assigned media; "
                    f"unassigned digests {extra}, unsealed assignments {absent}. "
                    "A partially sealed lock reports a clean identity for the "
                    "items it happens to cover, which is worse than no seal."
                )

    @property
    def sealing_version(self) -> str:
        """Derived, never stored independently.

        Serialised into the body so the file states its sealing explicitly, and
        checked against this on load — but computed here so the file cannot
        claim a seal it does not carry.
        """
        return SEALING_V1 if self.identity_digests else SEALING_UNSEALED

    @property
    def is_sealed(self) -> bool:
        return self.sealing_version != SEALING_UNSEALED

    # -- reading -----------------------------------------------------------

    def of(self, media_id: str) -> Split:
        try:
            return self.assignments[media_id]
        except KeyError:
            raise KeyError(
                f"{media_id!r} is not in the lock. Locked corpora do not assign "
                "on demand — run extend_lock and commit the result."
            ) from None

    def media_in(self, split: Split) -> list[str]:
        return sorted(m for m, s in self.assignments.items() if s is split)

    def counts(self) -> dict[Split, int]:
        return {s: sum(1 for v in self.assignments.values() if v is s) for s in Split}

    def achieved(self) -> dict[Split, float]:
        n = len(self.assignments)
        return {s: (c / n if n else 0.0) for s, c in self.counts().items()}

    @property
    def n(self) -> int:
        return len(self.assignments)

    def as_assignment(self, items: list[MediaItem]) -> SplitAssignment:
        """View the lock as a :class:`SplitAssignment` over ``items``.

        So everything already built against ``SplitAssignment`` — ``assess_split``,
        ``class_balance``, ``detect_leakage`` — works on a locked split without
        knowing the lock exists. Components are recomputed from ``items`` rather
        than stored, so this also surfaces the case where the corpus's identity
        metadata has been edited underneath a lock that no longer matches it.
        """
        known = {i.media_id for i in items}
        missing = sorted(set(self.assignments) - known)
        if missing:
            raise SplitLockError(
                f"{len(missing)} locked media are absent from the item list "
                f"(e.g. {missing[:5]}); a lock may only be viewed against the "
                "corpus it covers"
            )
        components = build_components(items, subject_scope=self.subject_scope)
        by_media = {m: s for m, s in self.assignments.items()}
        n = len(by_media) or 1
        counts = self.counts()
        return SplitAssignment(
            by_media=by_media,
            components=components,
            achieved={s: counts[s] / n for s in Split},
            counts=dict(counts),
            seed=self.seed,
            ratios=dict(self.ratios),
        )

    # -- serialisation -----------------------------------------------------

    def body(self) -> dict:
        """The hashed payload. Key order here is irrelevant — dumping sorts."""
        return {
            "assignments": {m: s.value for m, s in sorted(self.assignments.items())},
            "content_digests": dict(sorted(self.content_digests.items())),
            "corpus_fingerprint": self.corpus_fingerprint(),
            "generation": self.generation,
            "identity_digests": dict(sorted(self.identity_digests.items())),
            "lock_version": self.lock_version,
            "parent_hash": self.parent_hash,
            "quarantined": sorted(self.quarantined),
            "ratios": {s.value: r for s, r in sorted(self.ratios.items())},
            "seed": self.seed,
            "sealing_version": self.sealing_version,
            "subject_scope": self.subject_scope,
        }

    def corpus_fingerprint(self) -> str:
        """Short identity of *what is covered*, ignoring how it was split.

        Two locks over the same corpus share this even when their assignments
        differ, which is exactly the comparison a caller wants before asking
        whether two benchmark runs are on the same data.
        """
        return fingerprint(
            {
                "media": sorted(self.assignments),
                "quarantined": sorted(self.quarantined),
            }
        )

    def content_hash(self) -> str:
        """Full SHA-256 over the canonical body, ``sha256:`` prefixed.

        Full digest rather than the eight-character ``fingerprint`` used for
        model versions: this one is an integrity seal on a file that a human may
        hand-edit, not a change detector that has to stay readable in a log line.

        Note this covers the *assignments and the digests*, not the corpus. A
        matching content hash says the file has not been edited; it says nothing
        about whether the index behind it still means the same thing. That is
        what the identity seal is for, and keeping the two separate is what lets
        :class:`LockAudit` report them independently.
        """
        return hash_body(self.body())

    def to_json(self) -> str:
        """Canonical, stable, and diffable.

        Indented and newline-terminated because this file is committed and read
        by people; the *hash* is computed over the compact canonical form, so
        the pretty-printing cannot change it.
        """
        envelope = {
            "format": LOCK_FORMAT,
            "content_hash": self.content_hash(),
            "body": self.body(),
        }
        return json.dumps(envelope, indent=2, sort_keys=True) + "\n"

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, text: str) -> SplitLock:
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LockCorrupted(f"lock is not valid JSON: {exc}") from exc
        return cls.from_envelope(envelope)

    @classmethod
    def from_envelope(cls, envelope: object) -> SplitLock:
        if not isinstance(envelope, dict):
            raise LockCorrupted(f"lock must be a JSON object, got {type(envelope).__name__}")
        if envelope.get("format") != LOCK_FORMAT:
            raise LockCorrupted(
                f"not a split lock: expected format {LOCK_FORMAT!r}, found "
                f"{envelope.get('format')!r}"
            )
        for key in ("content_hash", "body"):
            if key not in envelope:
                raise LockCorrupted(f"lock envelope is missing {key!r}")
        body = envelope["body"]
        if not isinstance(body, dict):
            raise LockCorrupted("lock body must be a JSON object")

        # Three checks, in this order, and the order is the whole of the
        # diagnosis. A hand-edited file usually fails more than one of them, and
        # which failure the reader is shown decides whether they spend the
        # afternoon on the right problem.
        #
        # 1. Is the version one we can read at all? A body written by a newer
        #    format may not even have the fields the other checks read, and
        #    reporting a hash mismatch for what is really a version problem
        #    sends the reader in the wrong direction.
        _check_readable_version(body)

        # 2. Structure, on the migrated body. "assignments contains 'holdout'"
        #    is a more actionable answer than "this file was edited" — the
        #    reader already knows they edited it. It also catches a *correctly
        #    sealed* malformed lock, which a buggy writer produces and no
        #    integrity check can see.
        lock = cls._from_body(apply_migrations(body))

        # 3. Integrity against the bytes AS WRITTEN, before migration. The seal
        #    covers what is on disk; hashing the migrated body would compare the
        #    recorded digest against a document this build invented, and every
        #    1.0.0 file would then fail as tampered.
        recorded = envelope["content_hash"]
        actual = hash_body(body)
        if recorded != actual:
            raise LockHashMismatch(
                f"content hash mismatch: file records {recorded}, body hashes "
                f"to {actual}. The lock was edited after it was written — "
                "recover it from version control rather than resealing it, "
                "because resealing preserves whatever the edit did to the "
                "frozen test set."
            )
        return lock

    @classmethod
    def _from_body(cls, body: dict) -> SplitLock:
        required = ("assignments", "generation", "lock_version", "ratios", "seed")
        missing = sorted(set(required) - set(body))
        if missing:
            raise LockCorrupted(f"lock body is missing field(s) {missing}")

        raw_assignments = body["assignments"]
        if not isinstance(raw_assignments, dict):
            raise LockCorrupted("assignments must be an object of media_id -> split")
        assignments: dict[str, Split] = {}
        for media_id, value in raw_assignments.items():
            try:
                assignments[media_id] = Split(value)
            except ValueError:
                raise LockCorrupted(
                    f"{media_id!r} is assigned to {value!r}, which is not one of "
                    f"{[s.value for s in Split]}"
                ) from None

        raw_ratios = body["ratios"]
        if not isinstance(raw_ratios, dict):
            raise LockCorrupted("ratios must be an object of split -> fraction")
        try:
            ratios = {Split(k): float(v) for k, v in raw_ratios.items()}
        except (ValueError, TypeError) as exc:
            raise LockCorrupted(f"ratios are malformed: {exc}") from exc

        scope = body.get("subject_scope", "dataset")
        if scope not in ("dataset", "global"):
            raise LockCorrupted(
                f"subject_scope {scope!r} must be 'dataset' or 'global'"
            )

        generation = body["generation"]
        if not isinstance(generation, int) or isinstance(generation, bool):
            raise LockCorrupted(f"generation must be an integer, got {generation!r}")

        quarantined = tuple(body.get("quarantined") or ())
        if not all(isinstance(q, str) for q in quarantined):
            raise LockCorrupted("quarantined must be a list of media ids")

        digests: dict[str, dict[str, str]] = {}
        for name in ("identity_digests", "content_digests"):
            raw = body.get(name) or {}
            if not isinstance(raw, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in raw.items()
            ):
                raise LockCorrupted(f"{name} must be an object of media_id -> digest")
            digests[name] = dict(raw)

        try:
            lock = cls(
                assignments=assignments,
                seed=str(body["seed"]),
                ratios=ratios,
                subject_scope=scope,  # type: ignore[arg-type]
                generation=generation,
                parent_hash=body.get("parent_hash"),
                quarantined=quarantined,
                lock_version=str(body["lock_version"]),
                identity_digests=digests["identity_digests"],
                content_digests=digests["content_digests"],
            )
        except LockVersionMismatch:
            raise
        except ValueError as exc:
            raise LockCorrupted(str(exc)) from exc

        # The file states its sealing explicitly; this build derives it. A
        # disagreement means the file claims a seal it does not carry, which is
        # the one lie a sealed format must not accept.
        claimed = body.get("sealing_version", SEALING_UNSEALED)
        if claimed != lock.sealing_version:
            raise LockCorrupted(
                f"lock claims sealing_version {claimed!r} but carries "
                f"{lock.sealing_version!r} "
                f"({len(lock.identity_digests)} identity digest(s) for "
                f"{len(lock.assignments)} assignment(s))"
            )
        return lock

    @classmethod
    def load(cls, path: Path) -> SplitLock:
        path = Path(path)
        if not path.exists():
            raise LockCorrupted(f"no lock at {path}")
        return cls.from_json(path.read_text(encoding="utf-8"))


def _canonical(body: dict) -> bytes:
    """The exact bytes the content hash covers.

    Compact separators, sorted keys, ASCII-escaped — the same discipline as
    ``contracts.version.fingerprint``, for the same reason: two processes must
    agree on the bytes without agreeing on a formatting preference.
    """
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def hash_body(body: dict) -> str:
    """The content hash of a body **as given**, at whatever version it is.

    Version-agnostic on purpose: it hashes the document in front of it. That is
    what makes it usable on a 1.0.0 body during load, before migration, which is
    the only point at which the recorded digest and the bytes it was computed
    over are the same document.
    """
    return "sha256:" + hashlib.sha256(_canonical(body)).hexdigest()


def _check_readable_version(body: dict) -> None:
    """Refuse an unreadable version before anything else touches the body."""
    version = body.get("lock_version")
    if not isinstance(version, str):
        raise LockCorrupted(
            f"lock body has no usable lock_version (found {version!r})"
        )
    if version not in SUPPORTED_LOCK_VERSIONS and version not in MIGRATIONS:
        raise LockVersionMismatch(
            f"lock version {version!r} cannot be read by this build "
            f"(current {LOCK_VERSION}, supported {list(SUPPORTED_LOCK_VERSIONS)}, "
            f"migrations available from {sorted(MIGRATIONS)}). A lock from a "
            "newer build must be read by that build; an older one needs a "
            "migration registered here."
        )


# --------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------

Migration = Callable[[dict], dict]

def _migrate_1_0_0_to_1_1_0(body: dict) -> dict:
    """Add identity sealing to a lock that predates it — as *absent*, not empty.

    The temptation is to compute the digests here. It must be resisted: a
    migration has only the file, not the corpus, so any digests it invented
    would attest to an index it never saw. Worse, they would attest to it
    *retroactively* — the resulting lock would report a clean identity audit for
    a corpus that may already have been relabelled.

    So the upgrade is honest and lossy: the lock becomes 1.1.0 and stays
    ``unsealed``. :func:`extend_lock` seals it on the next growth, against the
    corpus as it stands then, and :func:`verify_lock` says plainly that it
    cannot answer the identity question until that happens.
    """
    out = dict(body)
    out.setdefault("identity_digests", {})
    out.setdefault("content_digests", {})
    out["sealing_version"] = SEALING_UNSEALED
    out["lock_version"] = "1.1.0"
    return out


MIGRATIONS: dict[str, Migration] = {"1.0.0": _migrate_1_0_0_to_1_1_0}
"""``from_version -> upgrade to the next version``.

A migration takes a body at its own version and returns one at the next, setting
``lock_version`` as it goes; :func:`apply_migrations` chains them.

A migration must never change an assignment. It may add fields, rename them, or
fill defaults. If an upgrade genuinely requires reassignment, that is not a
migration — it is a new lock, and it should be created as generation 0 with a
new file name so both are on disk and the discontinuity is visible.
"""


def apply_migrations(body: dict) -> dict:
    """Bring a lock body up to :data:`LOCK_VERSION`, or refuse.

    Refuses rather than best-efforts an unknown version. A lock is the thing
    that decides which items a model has never seen; reading one under a guess
    about its layout is how a test set quietly becomes a training set.
    """
    version = body.get("lock_version")
    if not isinstance(version, str):
        raise LockCorrupted(
            f"lock body has no usable lock_version (found {version!r})"
        )
    seen: set[str] = set()
    current = dict(body)
    while current["lock_version"] != LOCK_VERSION:
        version = current["lock_version"]
        if version in seen:
            raise LockVersionMismatch(
                f"migration cycle detected at lock version {version!r}"
            )
        seen.add(version)
        migration = MIGRATIONS.get(version)
        if migration is None:
            raise LockVersionMismatch(
                f"lock version {version!r} cannot be read by this build "
                f"(current {LOCK_VERSION}, supported {list(SUPPORTED_LOCK_VERSIONS)}, "
                f"migrations available from {sorted(MIGRATIONS)}). A lock from a "
                "newer build must be read by that build; an older one needs a "
                "migration registered here."
            )
        current = migration(current)
        if not isinstance(current, dict) or "lock_version" not in current:
            raise LockCorrupted(
                f"the migration from {version!r} did not return a versioned body"
            )
    return current


# --------------------------------------------------------------------------
# Creating and growing a lock
# --------------------------------------------------------------------------


def create_lock(
    items: list[MediaItem],
    *,
    seed: str = "dfp-v1",
    ratios: dict[Split, float] | None = None,
    subject_scope: SubjectScope = "dataset",
) -> SplitLock:
    """Split ``items`` once and freeze the result.

    Call this exactly once per benchmark, commit the file, and use
    :func:`extend_lock` from then on. Regenerating a lock over a grown corpus
    produces a *different* split — that is not a bug, it is the drift this
    module exists to prevent, and it is why regeneration is not the update path.
    """
    from evaluation.dataset import IdentitySplitter

    assignment = IdentitySplitter(
        ratios=ratios, seed=seed, subject_scope=subject_scope
    ).split(items)
    identity, content = seal(items, set(assignment.by_media))
    return SplitLock(
        assignments=dict(assignment.by_media),
        seed=seed,
        ratios=dict(assignment.ratios),
        subject_scope=subject_scope,
        generation=0,
        parent_hash=None,
        identity_digests=identity,
        content_digests=content,
    )


def extend_lock(
    lock: SplitLock,
    items: list[MediaItem],
    *,
    on_conflict: ConflictPolicy = "error",
) -> SplitLock:
    """Assign items the lock has not seen, without moving anything it has.

    ``items`` is the **whole** corpus, old and new together, because identity is
    a property of the corpus and not of the increment: a new file is linked to
    the frozen ones through sources and subjects that only exist in the full
    list. Passing just the new items would compute components over a fragment
    and place a sibling of a frozen item as though it were unrelated.

    Placement, in order:

    1. A component containing frozen members inherits their split. Not the
       emptiest fold — the identity's own fold, which is the only assignment
       that keeps the split leak-free.
    2. A component with frozen members in *two* splits is a conflict; see
       ``on_conflict`` and the module docstring.
    3. A wholly new component is placed greedily into whichever split is
       furthest below quota, counting everything already frozen. Ratios
       therefore converge over successive generations rather than being
       recomputed — the early folds are what they are.
    """
    known = {i.media_id for i in items}
    dropped = sorted(set(lock.assignments) - known)
    if dropped:
        raise SplitLockError(
            f"{len(dropped)} locked media are missing from the corpus "
            f"(e.g. {dropped[:5]}). A lock covers a corpus that only grows; "
            "removing items invalidates every comparison against previous runs "
            "and needs a new lock, not an extension."
        )

    # Check the seal BEFORE renewing it. This function re-seals over whatever
    # corpus it is handed, so without this an extension would launder a
    # relabelled index into a valid lock attesting to the new identity — and the
    # laundered lock would then pass every subsequent audit.
    drifted = detect_drift(lock, items)
    if drifted:
        raise LockIdentityDrift(
            f"{len(drifted)} sealed item(s) changed under the lock:\n"
            + "\n".join(f"  - {d.describe()}" for d in drifted[:10])
            + (f"\n  … and {len(drifted) - 10} more" if len(drifted) > 10 else "")
            + "\n\nExtending would re-seal these and the change would become "
            "invisible. Restore the index, or create a new lock and accept that "
            "results before and after are not comparable."
        )

    components = build_components(items, subject_scope=lock.subject_scope)
    frozen = lock.assignments

    inherited: dict[str, Split] = {}
    fresh: list[IdentityComponent] = []
    conflicts: list[MergeConflict] = []

    for component in components:
        splits = {frozen[m] for m in component.media_ids if m in frozen}
        if len(splits) > 1:
            conflicts.append(
                MergeConflict(
                    component_key=component.key,
                    splits=tuple(sorted(splits, key=lambda s: s.value)),
                    frozen_media_ids=tuple(
                        m for m in component.media_ids if m in frozen
                    ),
                    new_media_ids=tuple(
                        m for m in component.media_ids if m not in frozen
                    ),
                )
            )
            continue
        if splits:
            target = splits.pop()
            for media_id in component.media_ids:
                inherited[media_id] = target
        else:
            fresh.append(component)

    if conflicts and on_conflict == "error":
        raise LockMergeConflict(conflicts)

    quarantined = set(lock.quarantined)
    if conflicts:
        # Quarantine: frozen members keep their splits — the freeze wins — and
        # the new items that bridged them are set aside by name. The lock now
        # knowingly contains a spanning identity; verify_lock reports it on
        # every run, which is the price of not moving frozen items.
        for conflict in conflicts:
            for media_id in conflict.frozen_media_ids:
                inherited[media_id] = frozen[media_id]
            quarantined.update(conflict.new_media_ids)

    # Anything previously quarantined that is now placeable stops being
    # quarantined: the identity metadata may have been corrected, which is the
    # intended way out of a conflict.
    quarantined -= set(inherited)

    placed = {s: 0 for s in lock.ratios}
    for split in inherited.values():
        placed[split] = placed.get(split, 0) + 1

    n_total = len(inherited) + sum(c.size for c in fresh)
    quota = {s: r * n_total for s, r in lock.ratios.items()}
    ordered = sorted(fresh, key=lambda c: (-c.size, tiebreak_key(lock.seed, c.key)))
    for component in ordered:
        target = min(lock.ratios, key=lambda s: (placed[s] - quota[s], s.value))
        for media_id in component.media_ids:
            inherited[media_id] = target
        placed[target] += component.size

    for media_id, split in frozen.items():
        if inherited.get(media_id) != split:
            # Unreachable by construction; asserted because the one bug this
            # module must never have is a silent reassignment.
            raise SplitLockError(
                f"internal error: {media_id!r} would move from {split.value} to "
                f"{inherited.get(media_id)}; refusing to write the lock"
            )

    # Re-seal over the grown corpus. Safe only because detect_drift ran above:
    # every previously sealed item is known to be unchanged, so the new digests
    # extend the old seal rather than replacing it. An unsealed (migrated 1.0.0)
    # lock is sealed here for the first time, against the corpus as it stands.
    identity, content = seal(items, set(inherited))

    return SplitLock(
        assignments=inherited,
        seed=lock.seed,
        ratios=dict(lock.ratios),
        subject_scope=lock.subject_scope,
        generation=lock.generation + 1,
        parent_hash=lock.content_hash(),
        quarantined=tuple(sorted(quarantined)),
        identity_digests=identity,
        content_digests=content,
    )


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def detect_drift(lock: SplitLock, items: list[MediaItem]) -> list[MetadataDrift]:
    """Sealed items whose identity or content metadata no longer matches.

    Returns nothing for an unsealed lock — not because it is clean, but because
    there is nothing to compare against. Callers that need to tell those two
    apart read ``lock.is_sealed``; :class:`LockAudit` does exactly that rather
    than letting "no drift found" stand in for "no drift possible".
    """
    if not lock.is_sealed:
        return []
    drift: list[MetadataDrift] = []
    for item in items:
        sealed_identity = lock.identity_digests.get(item.media_id)
        if sealed_identity is None:
            continue  # not covered by this lock; growth, not drift
        actual = identity_digest(item)
        if actual != sealed_identity:
            drift.append(
                MetadataDrift(item.media_id, "identity", sealed_identity, actual)
            )
        sealed_content = lock.content_digests.get(item.media_id)
        if sealed_content is not None:
            actual_content = content_digest(item)
            if actual_content != sealed_content:
                drift.append(
                    MetadataDrift(
                        item.media_id, "content", sealed_content, actual_content
                    )
                )
    drift.sort(key=lambda d: (d.kind, d.media_id))
    return drift


@dataclass(frozen=True)
class LockAudit:
    """What a lock and a corpus disagree about, if anything.

    **Assignment integrity and identity integrity are separate verdicts.** A
    lock can be byte-perfect, cover the corpus exactly, and still be sealing a
    split whose meaning changed underneath it — that is the whole of R4. So
    ``drift`` is reported independently of ``leakage`` and ``conflicts``, and
    ``sealed`` is reported independently of both, because "no drift detected"
    and "no drift detectable" are different states and only one of them is
    reassuring.
    """

    unassigned: tuple[str, ...]
    """In the corpus, not in the lock. Run ``extend_lock``."""
    absent: tuple[str, ...]
    """In the lock, not in the corpus. The corpus shrank; comparisons break."""
    quarantined_present: tuple[str, ...]
    """Set aside by a previous conflict and still in the corpus."""
    leakage: LeakageReport
    conflicts: tuple[MergeConflict, ...]
    drift: tuple[MetadataDrift, ...] = ()
    sealed: bool = True
    """False for a lock that carries no digests — every 1.0.0 file after
    migration. Its assignments verify; its identity cannot be checked at all."""

    @property
    def identity_drift(self) -> tuple[MetadataDrift, ...]:
        return tuple(d for d in self.drift if d.kind == "identity")

    @property
    def content_drift(self) -> tuple[MetadataDrift, ...]:
        return tuple(d for d in self.drift if d.kind == "content")

    @property
    def assignments_intact(self) -> bool:
        """The 1.0.0 question: does the lock still cover this corpus soundly?"""
        return not (
            self.unassigned
            or self.absent
            or self.quarantined_present
            or self.conflicts
            or self.leakage.findings
        )

    @property
    def identity_intact(self) -> bool:
        """The 1.1.0 question. **False when unsealed**, which is not pedantry:
        an unsealed lock offers no evidence either way, and reporting that as
        intact is precisely the claim R4 found the format making."""
        return self.sealed and not self.drift

    @property
    def clean(self) -> bool:
        return self.assignments_intact and self.identity_intact

    def describe(self) -> str:
        if self.clean:
            return (
                "lock is clean: covers the corpus exactly, no leakage, "
                "identity seal intact"
            )
        lines: list[str] = []
        if not self.sealed:
            lines.append(
                "  this lock carries no identity seal (pre-1.1.0); its "
                "assignments verify but nothing behind them can be checked — "
                "run extend_lock to seal it against the current corpus"
            )
        for d in self.identity_drift[:5]:
            lines.append(f"  IDENTITY  {d.describe()}")
        for d in self.content_drift[:5]:
            lines.append(f"  content   {d.describe()}")
        if self.absent:
            lines.append(
                f"  {len(self.absent)} locked media are gone from the corpus "
                f"(e.g. {', '.join(self.absent[:3])})"
            )
        if self.unassigned:
            lines.append(
                f"  {len(self.unassigned)} corpus media are not in the lock "
                f"(e.g. {', '.join(self.unassigned[:3])}); run extend_lock"
            )
        if self.quarantined_present:
            lines.append(
                f"  {len(self.quarantined_present)} media are quarantined and "
                f"unusable (e.g. {', '.join(self.quarantined_present[:3])})"
            )
        for conflict in self.conflicts[:5]:
            lines.append(f"  conflict: {conflict.describe()}")
        if self.leakage.findings:
            lines.append("  " + self.leakage.describe().replace("\n", "\n  "))
        return "lock audit:\n" + "\n".join(lines)


def verify_lock(lock: SplitLock, items: list[MediaItem]) -> LockAudit:
    """Check a lock against the corpus it is supposed to cover.

    Cheap enough to run on every evaluation, and that is the intent: the lock's
    guarantees hold for as long as somebody checks them, and a lock nobody
    verifies is a file, not a control.
    """
    corpus_ids = {i.media_id for i in items}
    locked_ids = set(lock.assignments)

    covered = [i for i in items if i.media_id in locked_ids]
    leakage = (
        detect_leakage(covered, lock.assignments, subject_scope=lock.subject_scope)
        if covered
        else LeakageReport()
    )

    conflicts: list[MergeConflict] = []
    for component in build_components(items, subject_scope=lock.subject_scope):
        splits = {
            lock.assignments[m] for m in component.media_ids if m in lock.assignments
        }
        if len(splits) > 1:
            conflicts.append(
                MergeConflict(
                    component_key=component.key,
                    splits=tuple(sorted(splits, key=lambda s: s.value)),
                    frozen_media_ids=tuple(
                        m for m in component.media_ids if m in lock.assignments
                    ),
                    new_media_ids=tuple(
                        m for m in component.media_ids if m not in lock.assignments
                    ),
                )
            )

    return LockAudit(
        unassigned=tuple(sorted(corpus_ids - locked_ids - set(lock.quarantined))),
        absent=tuple(sorted(locked_ids - corpus_ids)),
        quarantined_present=tuple(sorted(set(lock.quarantined) & corpus_ids)),
        leakage=leakage,
        conflicts=tuple(conflicts),
        drift=tuple(detect_drift(lock, items)),
        sealed=lock.is_sealed,
    )


@dataclass(frozen=True)
class RegenerationCheck:
    """Whether re-deriving the split from scratch would reproduce the lock."""

    identical: bool
    moved: dict[str, tuple[Split, Split]] = field(default_factory=dict)

    def describe(self) -> str:
        if self.identical:
            return "regeneration reproduces the lock exactly"
        sample = list(self.moved.items())[:5]
        detail = "; ".join(
            f"{m}: {was.value} -> {now.value}" for m, (was, now) in sample
        )
        return (
            f"regeneration would move {len(self.moved)} item(s): {detail}. "
            "This is expected once the corpus has grown and is exactly why the "
            "lock exists; it is a problem only if the corpus has not changed."
        )


def check_regeneration(lock: SplitLock, items: list[MediaItem]) -> RegenerationCheck:
    """Compare the frozen split against what a fresh split would produce.

    Useful in two very different situations and it does not try to tell them
    apart, because only the caller can. On an unchanged corpus a difference
    means the splitter's behaviour changed under the lock — a real regression.
    On a grown corpus a difference is the normal, expected consequence of growth
    and is the evidence that freezing was necessary in the first place.
    """
    regenerated = create_lock(
        items, seed=lock.seed, ratios=dict(lock.ratios), subject_scope=lock.subject_scope
    )
    moved = {
        media_id: (split, regenerated.assignments[media_id])
        for media_id, split in lock.assignments.items()
        if media_id in regenerated.assignments
        and regenerated.assignments[media_id] is not split
    }
    return RegenerationCheck(identical=not moved, moved=moved)
