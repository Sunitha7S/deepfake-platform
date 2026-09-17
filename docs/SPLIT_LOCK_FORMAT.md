# Split lock file format, version 1.0.0

The lock is the artefact that makes a benchmark number comparable with the one
before it. It records which media belong to which split, and it is the authority
on that question — the splitter is not, because the splitter's answer changes as
the corpus grows.

Implementation: [`evaluation/split_lock.py`](../evaluation/split_lock.py).
Tests: [`tests/test_split_lock.py`](../tests/test_split_lock.py).

## Why the file exists

`IdentitySplitter` is deterministic: the same items and the same seed produce
the same split, every time, on every machine. That is not the same property as
stable. It places whole identity components largest-first into whichever fold is
furthest below quota, so **adding items moves items that were already placed**.

`test_growth_moves_items_when_the_split_is_regenerated` asserts exactly this,
against a 20-actor corpus grown to 40. It is the premise of the whole module,
and it is written as a test so that it stays true or the module gets deleted.

Nothing about that reassignment is loud. No exception, no warning, no failing
test. The only symptom is that this quarter's AUC is not comparable with last
quarter's, and nothing on disk says so.

## File layout

```json
{
  "format": "dfp-split-lock",
  "content_hash": "sha256:9f2c…",
  "body": {
    "assignments": { "v0_0_real_c23": "train", "v0_0_fake0_c23": "train" },
    "corpus_fingerprint": "1f4a9c2b",
    "generation": 3,
    "lock_version": "1.0.0",
    "parent_hash": "sha256:41ab…",
    "quarantined": [],
    "ratios": { "test": 0.15, "train": 0.7, "val": 0.15 },
    "seed": "dfp-v1",
    "subject_scope": "dataset"
  }
}
```

| Field | Meaning |
| --- | --- |
| `format` | Constant `dfp-split-lock`. Present so an unrelated JSON file is rejected as *not a lock*, rather than as a corrupt one. |
| `content_hash` | Full SHA-256 of the canonical body, `sha256:`-prefixed. Full digest rather than the 8-character `contracts.version.fingerprint`, because this is an integrity seal on a file people hand-edit, not a change detector that has to stay readable in a log line. |
| `assignments` | `media_id → split`. The frozen set. |
| `corpus_fingerprint` | Short hash of *membership only*, ignoring the split. Two locks over the same corpus share it even when their assignments differ. |
| `generation` | 0 for a created lock, +1 per `extend_lock`. |
| `lock_version` | Version of *this format*. See “Versioning”. |
| `parent_hash` | `content_hash` of the lock this one extends; `null` at generation 0. Chains the history. |
| `quarantined` | Media set aside by a merge conflict. Unassigned and unusable; see “The conflict the lock cannot solve”. |
| `ratios`, `seed`, `subject_scope` | The splitter configuration, carried so an extension reproduces the original placement rule rather than a default one. |

### Canonicalisation

The hash covers `json.dumps(body, sort_keys=True, separators=(",", ":"),
ensure_ascii=True)`. The file on disk is written with `indent=2` because it is
committed and read by people; pretty-printing therefore cannot change the hash,
which `test_pretty_printing_does_not_change_the_hash` pins.

### There is no timestamp, deliberately

A `created_at` field would make two locks built from the same corpus differ byte
for byte. That destroys the cheapest reproducibility check there is —
regenerate, diff, expect silence — and makes
`test_identical_corpus_produces_an_identical_lock` unwritable. Provenance is
carried by `generation` and `parent_hash`, which chain without varying for the
same inputs. The timestamp belongs in the commit.

`test_the_lock_carries_no_timestamp` enforces this against future additions.

## Guarantees

1. **Frozen assignments never move.** Growth may add; it may never reassign.
   `extend_lock` re-checks every frozen entry against its output before
   returning and raises rather than writing a lock that moved one.
2. **New items inherit their identity component's split.** A newly encoded copy
   of footage already in test goes to test, not to the emptiest fold. Placing it
   by quota would put one source on both sides of the evaluation.
3. **Wholly new components are placed greedily**, counting everything already
   frozen toward the quota, so ratios converge over generations rather than
   being recomputed.
4. **The file is self-verifying**: truncation, a hand-edited assignment, an
   unknown split name and a foreign JSON document each raise a distinct error.

## The conflict the lock cannot solve

A new item can *bridge* two frozen components. An actor recorded only in `v3`
turns up in `v9`; `v3` is in train and `v9` is in test; union-find now says all
of it is one identity. The two guarantees then contradict each other — honouring
the freeze leaves one identity in two splits, and removing the leakage moves
frozen items.

There is no resolution inside a file format, so the lock does not invent one.

| `on_conflict` | Behaviour |
| --- | --- |
| `"error"` (default) | Raise `LockMergeConflict`, naming the component, the splits and the bridging media. The decision is about the corpus, and it is the caller's. |
| `"quarantine"` | Frozen items keep their splits; the bridging items are set aside by name in `quarantined` and left unassigned. `verify_lock` then reports the resulting spanning identity on **every** run, so the compromise cannot be forgotten. |

Quarantine is an escape hatch, not a fix. The honest description is: the freeze
wins, the leak is accepted, and the cost stays visible. A quarantined item is
released automatically once its identity metadata stops bridging — which is the
intended way out.

## Workflow

```python
from evaluation.split_lock import create_lock, extend_lock, verify_lock, SplitLock

# Once, ever:
lock = create_lock(items, seed="dfp-v1")
lock.save(Path("benchmarks/ffpp.lock.json"))

# Every evaluation:
lock = SplitLock.load(Path("benchmarks/ffpp.lock.json"))
audit = verify_lock(lock, items)
assert audit.clean, audit.describe()

# When the corpus grows:
lock = extend_lock(lock, items)      # items = the WHOLE corpus, old and new
lock.save(Path("benchmarks/ffpp.lock.json"))
```

`extend_lock` takes the whole corpus rather than the increment because identity
is a property of the corpus: a new file is linked to the frozen ones through
sources and subjects that only exist in the full list. Passing only the new items
would compute components over a fragment and place a sibling of a frozen item as
though it were unrelated.

Commit the lock. Review its diff like code — a diff that changes an existing
assignment is a bug in this module, and a diff that adds a hundred lines is the
corpus growing.

## Versioning

`lock_version` is the version of the file format and is **independent of
`CONTRACT_VERSION`**. The lock is an offline artefact that no serving component
reads; coupling it to the frozen API contract would mean either bumping a public
contract to add a lock field, or never evolving the lock at all.

Reading is strict. An unrecognised version raises `LockVersionMismatch` and the
file is not read — never parsed on a guess. A lock decides which items a model
has never seen; reading one under an assumption about its layout is how a test
set quietly becomes a training set.

The version check runs **before** the integrity check, so a lock that is both
too new and hash-mismatched reports the version problem. That ordering is pinned
by `test_a_version_mismatch_is_reported_as_such_and_not_as_a_hash_failure`,
because sending a reader to chase a corrupted file when the real problem is a
stale build wastes the expensive kind of afternoon.

## Migration strategy

`MIGRATIONS` maps `from_version → upgrade function`. It is **empty at 1.0.0 and
that is the point**: the mechanism ships before it is needed, because the first
time a format has to change is exactly the moment when nobody wants to also be
designing the upgrade path. `apply_migrations` chains them until the body
reaches `LOCK_VERSION`, and refuses on a cycle or a gap.

**A migration may add fields, rename them, or fill defaults. A migration may
never change an assignment.** That rule is what makes an upgrade safe to run
unattended, and `test_a_registered_migration_is_applied_and_the_lock_loads`
asserts it by comparing assignments across a fabricated 0.9.0 → 1.0.0 upgrade.

To add a version:

1. Bump `LOCK_VERSION`, append the old value to `SUPPORTED_LOCK_VERSIONS`.
2. Register `MIGRATIONS["<old>"] = upgrade`, where `upgrade` returns a body with
   `lock_version` set to the next version.
3. Extend `SplitLock._from_body` to read the new field, with a default that
   reproduces the old behaviour.
4. Add a test in the shape of the existing migration test: migrate a body from
   the old version and assert the assignments are unchanged.

If an upgrade genuinely requires reassignment, **it is not a migration**. It is a
new lock: create it at generation 0 under a new filename, so both files are on
disk and the discontinuity in the benchmark history is visible to whoever reads
the numbers on either side of it.

### Adopting the lock on a corpus that already has a split

There is no migration path from “a split that lives in a script”, because there
is nothing to migrate from. Build the lock from the assignment you are already
using — `SplitLock(assignments=..., seed=..., ratios=...)` accepts one directly —
rather than calling `create_lock`, which would produce a *different* split and
silently reset the benchmark baseline. Then run `verify_lock` and read the
report: an inherited split that was never identity-aware will have leakage in it,
and freezing a leaky split preserves the leak forever.
