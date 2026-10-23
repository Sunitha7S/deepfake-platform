# R16 + R26 — Specification Resolution and Implementation Plan

**Type:** design/specification resolution. No code in this task.
**Follows:** `docs/R16_R26_DESIGN.md`, whose gate (§6) was APPROVE WITH
CHANGES on two named open decisions. **Supersedes** `docs/R16_R26_DESIGN.md`
wherever this document's re-derivation contradicts it (flagged explicitly at
each point) — `R16_R26_DESIGN.md` remains the record of the original
re-derivation (§1–§3, the tier split, the OpenAPI-blindness finding); this
document is authoritative for the *scope actually being built*.
**Method:** every claim re-derived from current source, per instruction —
`R16_R26_DESIGN.md`'s own conclusions are treated as hypotheses to verify,
not premises, exactly as that document treated `docs/MULTI_FACE_SEMANTICS.md`
and `docs/CROP_GEOMETRY.md`.

**Baseline re-confirmed, live, before any decision below:** `git status`
— no drift since `R16_R26_DESIGN.md` was written (same uncommitted
OR1-b/R30 tree, `HEAD = 8676fdd`). `CONTRACT_VERSION = "1.0.0"`
(`contracts/enums.py:16`), `RUN_SCHEMA_VERSION = "1.2.0"`
(`evaluation/benchmark.py:108`), `CALIBRATION_WITNESS_VERSION = "1.3.0"`
(`evaluation/calibration_protocol.py:74`), `LOCK_VERSION = "1.1.0"`
(`evaluation/split_lock.py:81`), `STUB_ENGINE_VERSION` ends `.d.c9d4025c`
(`backend/config.py:41`) — all read live this session, not copied from the
prior document.

---

## Decision 1 — R26 Tier 2: option (a) (test-only guard) or option (b) (loosen `BoundingBox`)?

### Re-derivation

Re-read `inference/postprocess.py:675-685` and `contracts/common.py:86-90`
in full again this session — unchanged since `R16_R26_DESIGN.md`:

```python
class BoundingBox(Base):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)

def face_track(box_xywh, *, p_fake, frames_seen=1, status=None):
    x, y, w, h = box_xywh
    return FaceTrack(
        track_id=0,
        bbox=BoundingBox(x=max(0, x), y=max(0, y), width=max(1, w), height=max(1, h)),
        ...
    )
```

**The claim that Haar cannot produce a negative-origin box was re-verified
against a source this session had not yet checked**, not merely re-asserted
from `R16_R26_DESIGN.md`: `docs/CROP_GEOMETRY.md` §3.2 (`"Haar cannot detect
a face that is partly outside the frame, so the box is always fully
inside"`) and §8's SCRFD-migration-impact table (`"negative box coords
clamped by face_track | unreachable (Haar) | ACTIVE (regressed boxes) under
SCRFD"`) — a claim that document backs with a **50,000-random-box brute
force** (§3.2, "0 mismatches") from an earlier, independent batch (Batch
15). This is a stronger evidentiary basis than `R16_R26_DESIGN.md`'s own
citation of `cv2.CascadeClassifier.detectMultiScale`'s general contract; it
is re-used here rather than re-derived from scratch a second time, per this
project's own convention of citing a prior batch's measurement rather than
re-running an identical experiment (e.g. how Batch 20 cites Batch 8's E4/E6
measurements without re-running them).

**Re-derivation of blast radius for option (b)**, repeated and confirmed
unchanged: `grep -rn "BoundingBox(" --include="*.py" .` → one production
call site (`inference/postprocess.py:681`), six `contracts/fixtures.py`
literals, four test-file literals (`tests/stub_handlers.py` ×3,
`tests/test_faces_consistency.py`, `tests/test_video_face_semantics.py`,
`tests/video_face_semantics_study.py`) — every one already non-negative.
Removing `ge=0` **cannot invalidate any existing valid data**; it can only
newly-accept values that were previously rejected, none of which any
current code path produces or consumes.

**Re-derivation of the two options' actual difference:** option (a) detects
a future violation (fails a test the day something producing a negative box
lands); option (b) prevents `face_track()` from *silently misreporting* a
real detector's true output once one exists (today, `max(0, x) == x` always,
so (b) changes zero currently-observable behavior either). Neither is a
"fix" in the sense of changing today's published data — both are
forward-looking. **This is the same structure as R31_SPEC_RESOLUTION.md's
Decision 2** (D1 ships now, zero marginal cost; D2 needs its own
migration/collateral surface) even though here neither option has a version
cost — the axis that differs is *scope of commitment*, not migration cost.

### Recommendation

**Option (a) ships in this batch. Option (b) is deferred, not decided
here.**

- Option (a) is a pure test addition — no contract file changes, no
  behavioral change, nothing for a reviewer to approve beyond "is this test
  correct," which is an engineering question with a source-verifiable
  answer (§ above).
- Option (b) is a genuine, if currently invisible, contract-shape decision
  (§Decision-1 re-derivation: it does not move this repo's OpenAPI drift
  check, confirmed in `R16_R26_DESIGN.md` and re-confirmed here, but it is
  still a real constraint loosening a schema consumer outside this repo
  could observe) — `R16_R26_DESIGN.md`'s own gate already flagged this as a
  product/team call, and nothing re-derived this session changes that
  classification. Deciding it inside a "cheap, zero-decision-needed" batch
  would misrepresent the batch's own risk profile to a reviewer skimming
  the diff.
- This keeps the batch's decision surface at **zero** open questions for a
  reviewer, matching `R31_SPEC_RESOLUTION.md`'s stated goal for its own D1/D2
  split ("a single, minimal, one-version-axis batch").

**If the team later wants option (b):** it is a one-line contract change
(`x: int = Field(ge=0)` → `x: int`, same for `y`) plus deleting
`face_track()`'s now-unnecessary `max(0, ...)` calls, with the blast radius
already fully enumerated above. It should be scoped as its own follow-up
once a real detector (SCRFD) exists to decide whether the published value
should be the raw regressed box or something else (e.g. intersected with
the frame) — a decision this document declines to make blind, matching
`docs/CROP_GEOMETRY.md`'s own recommendation not to design geometry
decisions ahead of the detector that will exercise them.

### Affected files

**Option (a), ships now:** one new test file (§ Final Plan, §3).
**Option (b), deferred, not touched:** `contracts/common.py`
(`BoundingBox.x`/`.y`), `inference/postprocess.py` (`face_track`).

### Impact

| Axis | Option (a), this batch | Option (b), deferred |
|---|---|---|
| Migration | None. | None (§ blast-radius re-derivation — no valid data invalidated). |
| Version | None on any axis. | None on any axis (re-verified, §Decision-1 of `R16_R26_DESIGN.md`, empirically re-tested there). |
| OpenAPI | None. | None (re-confirmed — a removed `minimum` is invisible to `_type_of()`). |
| Cache / `.f.` / `.d.` | None. | None — `BoundingBox` is not read by any identity-computing function. |
| Deployment | None. | None. |

### Does this change the approved `R16_R26_DESIGN.md` scope?

**No — it resolves the one item that document's gate left open (item 1),
narrowing to the side that document already recommended** ("this
re-derivation recommends (a) for the October window"). This document
converts that recommendation into a decision.

---

## Decision 2 — Implement R16 and R26 together, or as separate batches?

### Re-derivation

Re-checked `docs/PROJECT_ROADMAP.md:219` this session: **Phase 4 already
groups them** — *"Phase 4 — Product decisions, cheap paths | R16 (cheap),
R26 (cheap) | Two Low-severity findings closed via contract-prose caveats,
no pixel/version movement."* This is a pre-existing roadmap grouping, not
an assumption `R16_R26_DESIGN.md` introduced.

**Re-derivation of shared surface, repeated with fresh eyes:** R16's cheap
path touches `contracts/evidence.py` only; R26 Tier 1 touches
`contracts/common.py` only; R26 Tier 2 (option a) adds one new, self-
contained test file. **Zero file overlap.** The three pieces are
independently revertable, independently testable, and — per Decision 1's
own impact table and `R16_R26_DESIGN.md`'s corrected OpenAPI finding —
**share an identical impact profile**: no version axis moves, no OpenAPI
artifact changes, no cache/deploy consequence, for any of the three.

**Re-derivation of this project's own bundling convention**, checked against
`tests/run_all.py`'s `SUITES` list (53 entries) rather than assumed: batch
titles routinely bundle multiple findings sharing one theme — `"S8-B23
video selection identity (R29 + N4)"`, `"S8-B26 image decode identity (R30 +
D8/D9)"`. A single-finding batch is the exception in this history, not the
rule, and every multi-finding batch bundled findings that shared a version
axis or a review theme — exactly R16+R26's relationship here (same "Phase
4, cheap/documentation, zero version movement" theme).

### Recommendation

**One batch.** Bundle R16 cheap path + R26 Tier 1 + R26 Tier 2 (option a)
as a single implementation batch. Rationale, re-derived rather than
defaulted:

1. Zero file overlap means bundling adds no merge risk.
2. Identical impact profile (every axis: none) means there is no version
   axis to keep separate — the usual reason this project *splits* batches
   (e.g. R31_SPEC_RESOLUTION's D1/D2 split, driven by D2 needing its own
   `RUN_SCHEMA_VERSION` axis) does not apply here, since nothing in this
   batch needs its own axis.
3. Matches the pre-existing `docs/PROJECT_ROADMAP.md` Phase 4 grouping,
   which this document did not invent.
4. Matches this project's own dominant batch-authoring pattern.

**Named as Batch 27** (continuing the sequence: OR1-b = 25, R30-bundle =
26) — **re-verify this number at implementation time**, not from this
document, per the same concurrent-session caution `docs/R30_BUNDLE_DESIGN.md`
§7/§8 already established for this exact repository. `tests/run_all.py`'s
`SUITES` length must be re-counted live before registering a new entry.

### Affected files

None beyond what Decision 1 and the Final Plan (below) already enumerate —
this decision is about grouping, not new surface.

### Does this change the approved `R16_R26_DESIGN.md` scope?

**No — it resolves the second item that document's gate left open (item 2),
choosing bundling, which that document already offered as its stated
default.**

---

## Summary of the two resolutions

| # | Decision | Resolution |
|---|---|---|
| 1 | R26 Tier 2 — (a) vs (b) | **(a) ships now** — a pure regression test, zero contract change, zero decision surface. **(b) deferred**, not decided here — a real (if currently invisible) contract-shape choice that should wait for SCRFD to inform what the published value should become. |
| 2 | Bundle or split | **One batch (Batch 27, number to be re-verified at implementation time)** — R16 cheap path + R26 Tier 1 + R26 Tier 2(a), zero file overlap, identical (none) impact profile on every version axis, matches the pre-existing `docs/PROJECT_ROADMAP.md` Phase 4 grouping and this project's dominant batch-authoring pattern. |

Neither resolution reopens architecture or contradicts
`docs/R16_R26_DESIGN.md`'s core findings (the tier split, the closed-
prerequisites finding for R16, the OpenAPI-blindness correction). Both
**narrow** what was left open, in the safer direction — the same shape
`R31_SPEC_RESOLUTION.md`'s own three decisions took.

---

## Answers to the four scoping questions, restated precisely

- **Which portions of R16 can be safely closed now?** The cheap/
  documentation path only — extending the single-subject caveat already
  present in `FaceTrack`'s docstring to `EvidenceBundle`'s `verdict`/
  `p_fake` field descriptions (Option 1 of `docs/MULTI_FACE_SEMANTICS.md`
  §12, confirmed still partially unimplemented, §ex below).
- **Which portions remain blocked by SCRFD/R13?** Option 2 (score and
  publish every passing face) and anything beyond it (Options 3/4).
  Unchanged from `R16_R26_DESIGN.md` — R13 (310× amplification) is
  unaffected by anything re-derived this session; `inference/preprocess.py`
  still runs Haar (re-confirmed, no SCRFD import anywhere, this session).
- **Which portions of R26 can be closed immediately?** Tier 1 (the
  `FaceTrack` docstring caveat about `bbox` vs. the scored region) and Tier
  2 option (a) (the regression test proving the clamp divergence is
  currently unreachable). Tier 2 option (b) and Tier 3 are not closed now
  (Decision 1, Decision-implicit re-confirmation of `R16_R26_DESIGN.md`
  §2.2).
- **Should R16 and R26 be implemented together or split?** Together, one
  batch (Decision 2).

---

# Final Implementation Plan

## 1. Scope, restated precisely

**In scope (Batch 27, pending number re-verification):**

1. `contracts/evidence.py` — `verdict` and `p_fake` field descriptions gain
   the single-subject caveat.
2. `contracts/common.py` — `FaceTrack`'s docstring gains the bbox-vs-scored-
   region caveat.
3. A new test file proving the Haar-backed `face_track()` clamp is
   currently a structural no-op (R26 Tier 2, option a).
4. Documentation updates: `REPRODUCIBILITY_AUDIT.md` (R16/R26 rows),
   `docs/PROJECT_ROADMAP.md` (Phase 4 marked done).

**Explicitly out of scope, not touched:** R16 Option 2 (per-face scoring),
R26 Tier 2 option (b) (loosening `BoundingBox`), R26 Tier 3 (publishing the
scored region), R23-border-clip, R28, anything SCRFD-scoped.

## 2. Production files changed

| File | Change |
|---|---|
| `contracts/evidence.py` | `verdict: Verdict` (currently a bare annotation, no `Field()`) becomes `verdict: Verdict = Field(..., description="...")`, keeping it required (`...` is pydantic's explicit "no default" marker — behaviorally identical to today, metadata-only addition). `p_fake`'s existing `Field(default=None, ge=0.0, le=1.0, description=(...))` gains one clause appended to its description tuple. **Exact proposed text** (subject to copy-editing, not re-litigation of substance): <br>`verdict` — *"The three-state result for the media's single published face — the largest one detected (`face_order_key` selects it; see `FaceTrack`'s own note). A multi-subject photograph's verdict describes one person in it; the platform does not currently score or publish evidence for any other face present (R16, open)."* <br>`p_fake` appended clause — *" Describes the same single selected face `verdict` does, not an aggregate over every face in the media."* |
| `contracts/common.py` | `FaceTrack`'s class docstring gains one new bullet, in the same list as the existing R16/R18 bullets. **Exact proposed text:** *"**`bbox` is the detector's box, not the region the model scored.** `crop_face` expands it by the selection policy's margin, then clips it to the image — a different, and usually larger, window than what is published here. No field on this contract carries that window; recovering it requires re-deriving it from `bbox` and the shipped `FaceSelectionPolicy`, which a client cannot do from the wire alone. That gap is R26, open."* |

**Not changed:** `inference/preprocess.py`, `inference/image_pipeline.py`,
`inference/postprocess.py` (function bodies — `face_track()`'s clamp logic
is proved-not-fixed in this batch, per Decision 1), `backend/config.py`,
`inference/decode_env.py`, any `.f.`/`.d.`-affecting code, `contracts/
common.py`'s `BoundingBox` class itself (constraint unchanged, per
Decision 1).

## 3. Test files changed

| File | Change |
|---|---|
| New: `tests/test_face_geometry_publication.py` | R26 Tier 2, option (a). See §5 fail-before/pass-after matrix for exact test names and assertions. |
| `tests/test_schema.py` | No assertion currently pins `verdict`'s or `p_fake`'s description *content* (confirmed by search this session — only `test_field_descriptions_survive_into_the_schema` checks `p_fake`'s description, and only for the substring `"full precision"`, which the proposed edit does not remove). **No change required**, but re-run to confirm. |
| `tests/test_openapi.py` | No source change. Re-run to confirm the fingerprint/signature are **byte-identical** before and after (not merely "no drift failure" — the stronger claim, re-verified empirically this session for both a `Field(description=...)` change and, separately, for a `minimum`-constraint removal, neither of which this batch performs, but both of which were tested as a superset check on the signature's blindness). |
| `tests/test_faces_consistency.py` | No change expected — confirmed by re-reading: its assertions are about `track_id` uniqueness and the headline/face-score tie (R21's invariants), not about `bbox` or field descriptions. Re-run for confirmation only. |

**Not changed:** every other suite — this batch touches no identity,
version, cache, or scoring code path.

## 4. Documents changed (on implementation, not in this task)

| Doc | Change |
|---|---|
| `docs/R16_R26_SPEC_RESOLUTION.md` | this document |
| `REPRODUCIBILITY_AUDIT.md` | R16 row: Open → **Closed (documentation half)**, full-form entry matching the B20/B23/B26 row style, citing the deferred Option 2 explicitly as "not closed, gated on SCRFD" rather than silently dropped. R26 row: Open → **Closed (Tier 1 + Tier 2-option-a)**, citing Tier 2-option-(b) and Tier 3 as explicitly deferred. New "Still open after Batch 27" trailer section, following the established per-batch pattern. |
| `docs/PROJECT_ROADMAP.md` | Phase 4 row marked done; §1 table R16/R26 rows updated to reflect the tiered closure (not a blanket "Closed," since real sub-scope remains open — matching how R30's own roadmap update distinguished D8's partial closure). |
| `docs/MULTI_FACE_SEMANTICS.md`, `docs/CROP_GEOMETRY.md` | **Not edited** — remain the historical record of the original findings, per this session's established convention (`R31_SPEC_RESOLUTION.md`'s own §4 precedent: design documents are not retroactively rewritten by their own resolution). A superseding blockquote note, matching the pattern already used at the top of `docs/DECODE_IDENTITY_COMPLETENESS.md`, is the right shape if the team wants a forward pointer — optional, not required for the gate below. |

## 5. Fail-before / pass-after matrix (R26 Tier 2, option a)

| # | Test | Pre-change | Post-change |
|---|---|---|---|
| FB1 | `test_a_regressed_box_would_be_silently_misreported` (a permanent demonstration, not a true "fail-before" — this defect is not being fixed, only guarded; matches the "measured and rejected" / R28-style pattern of pinning a known, deliberately-unfixed gap) | N/A — this test is new either way | Constructs a synthetic `box_xywh = (-20, 5, 100, 100)` (simulating a hypothetical regressed detection), passes it to `face_track()`, and asserts `result.bbox.x == 0 != -20` — the exact, named divergence, pinned so a future silent "fix" of the clamp without updating this test is caught, and so a reader has a concrete example rather than a prose description. |
| PA1 | `test_haar_backed_detection_never_produces_a_negative_origin_box` | — | Runs `detect_faces()` (the real, Haar-backed function, not a double) against the existing fixture image set already used elsewhere in the suite (e.g. the `_face_tile()`-style fixtures in `tests/test_crop_identity.py`, reused rather than duplicated) across a representative set of positions including near every frame edge; asserts every returned `FaceBox.x >= 0 and FaceBox.y >= 0`. Reinforces, on this box, the `docs/CROP_GEOMETRY.md` §3.2 50,000-box brute-force finding — a live re-measurement, not merely a citation. |
| PA2 | `test_face_track_clamp_is_a_no_op_for_every_box_detect_faces_can_produce` | — | Feeds every `FaceBox` produced by PA1's detection runs through `face_track()` and asserts the published `bbox.x/.y` equal the input `box.x/.y` exactly (i.e., `max(0, x) == x` held for all of them) — the structural "no-op" claim asserted directly, not inferred from PA1 alone. |
| PA3 | `test_the_clamp_gap_is_tracked_in_the_reproducibility_audit` | — | `"R26" in REPRODUCIBILITY_AUDIT.md`'s text and the row's status string does not claim the clamp itself is fixed — a doc-linkage tripwire, matching the existing pattern (`test_r29_is_tracked_in_the_reproducibility_audit`, `docs/VIDEO_SELECTION_IDENTITY.md`'s neighbourhood) of at least one test per batch tying code to the audit trail. |

**Explicitly not tested (out of scope, correctly):** any claim about SCRFD's
actual box format — none exists to test against; PA1/PA2 are scoped to
"what Haar can produce today," per Decision 1's own reasoning.

## 6. Non-movement matrix

| Quantity | Must remain | Why |
|---|---|---|
| `decode_identity()` | `c9d4025c` | This batch touches no `inference/decode_env.py` code. |
| `STUB_ENGINE_VERSION` | `...f.e629642c.d.c9d4025c` unchanged | Neither `FaceSelectionPolicy` nor any decode-affecting code is touched. |
| `model_identity_of(STUB_ENGINE_VERSION)` | unchanged | Same reason — `contracts/evidence.py`/`contracts/common.py` are wire-contract files, not identity-computing files. |
| `CONTRACT_VERSION` | `1.0.0` | The platform's own Stable/Evolving rule (`contracts/evidence.py:9-19`) gates *meaning* changes to Stable fields; `verdict`'s and `p_fake`'s meaning — "the largest detected face's result" — is not changing, only becoming stated. Re-confirmed, not assumed: neither field's validation behavior, type, or default changes. |
| `RUN_SCHEMA_VERSION` | `1.2.0` | No `BenchmarkRun` field touched. |
| `CALIBRATION_WITNESS_VERSION` | `1.3.0` | No `CalibrationWitness`/`ModelSpec.pre_calibration_identity`-adjacent code touched — re-confirmed relevant this session specifically because R31 made this a live dependency for any future `.f.`-moving batch (`R16_R26_DESIGN.md` §1); this batch is not `.f.`-moving, so the dependency is inert here. |
| `LOCK_VERSION` (split lock) | `1.1.0` | Wholly unrelated subsystem; no reference to `BoundingBox`/`FaceTrack`/`EvidenceBundle` anywhere in `evaluation/split_lock.py` (confirmed by grep this session). |
| `backend/openapi.snapshot.json` (`fingerprint`/`signature`) | **byte-identical** | Re-confirmed empirically this session (Decision 1; `R16_R26_DESIGN.md`'s executive-summary correction): `_component_signature`/`_type_of` (`backend/openapi.py:209-251`) capture only `required` + `{field: type}`, no `description`, no `minimum`/`ge`. Neither field-description edits nor (the deferred) constraint removal reach the signature. |
| `schema_out/*.json` | not a repo artifact (gitignored, on-demand) | `git ls-files schema_out/` → empty, confirmed this session. No "movement" to speak of. |
| `fixtures_out/*.json` | unaffected | Fixture *values* (not field metadata) are what populate these; a description edit changes no fixture's data. |

## 7. Migration impact analysis

**None, on every axis.** No cache generation (no `.d.`/`.f.` move — §6). No
deploy coordination (no `STUB_ENGINE_VERSION` move — OR1/OR1-b's fail-closed
gate is not triggered by anything this batch touches). No client migration
(`CONTRACT_VERSION` unmoved, and the Evolving-tier `faces[]`/`FaceTrack`
field this batch's Tier 1 note documents was already Evolving before this
batch — the note describes existing behavior, it does not change what
`faces[]` contains). No stored-artifact migration: re-confirmed this
session using the same method `R31_SPEC_RESOLUTION.md` §9 used — no
committed `EvidenceBundle`/`CalibrationWitness`/`BenchmarkRun` payloads
exist anywhere in this repository outside test fixtures and
`fixtures_out/*.json` (which carry `model.version = "1.0.0-dev"` and no
witness), so there is nothing to migrate regardless of what this batch
touches.

**Rollback:** trivial — revert the two docstring/description edits and
delete the new test file. No literal, no cache entry, no version constant
to un-move, because none moved.

## 8. Risk analysis

| Risk | Severity | Mitigation |
|---|---|---|
| **Description text is imprecise or overclaims a capability the pipeline doesn't have** (e.g. implying `faces[]` already carries multiple scored faces) | Low | Exact proposed text (§2) was drafted specifically to avoid this — it states what is published today (one face, the largest) and explicitly says the platform "does not currently score or publish evidence for any other face present," matching `FaceTrack`'s own existing, already-shipped caveat style rather than inventing new claims. |
| **PA1's fixture set is not representative enough to make the "Haar never produces a negative box" claim convincing on its own** | Low | This batch's claim rests primarily on `docs/CROP_GEOMETRY.md`'s own prior 50,000-box brute force (Decision 1), not on PA1 alone — PA1 is corroborating, not the sole evidence. If the team wants a stronger PA1, it can be scaled up at implementation time without changing this plan's shape. |
| **FB1's synthetic-box test could be read as "the bug is fixed" by a future skimmer** | Low | FB1's own name and docstring must state plainly that this is a **permanent demonstration of an accepted, deferred gap**, not a fail-before/pass-after pair that closes — matching the project's existing convention for `R28`'s `interpolation_rule` (named, pinned, explicitly not fixed). |
| **Concurrent-session drift** — this repository has a demonstrated history of two sessions editing it at once (`R30_BUNDLE_DESIGN.md` §7, directly observed with OR1-b) | Medium, procedural | Re-run `git status` and re-count `tests/run_all.py`'s `SUITES` immediately before implementation, exactly as `R30_BUNDLE_DESIGN.md`'s own §8 step 1 specified — do not trust this document's batch number or file-line references without a fresh check. |
| **Bundling risk (Decision 2)** — combining three independent pieces into one batch could obscure which piece caused a failure | Low | The three pieces are file-disjoint (§Decision 2 re-derivation) — a test failure will localize to exactly one file, so bundling costs nothing in debuggability despite sharing one commit/PR. |

**No risk rises above Low/Medium-procedural.** This is consistent with
`R16_R26_DESIGN.md`'s own risk assessment (cheap path: None; Tier
2-option-a: Low) and is not weakened by anything re-derived this session.

## 9. Collateral test and documentation updates — exhaustive list

Re-derived by search this session, not copied from `R16_R26_DESIGN.md`:

- **No `.d.`/`.f.`/`STUB_ENGINE_VERSION`/`CONTRACT_VERSION`/
  `RUN_SCHEMA_VERSION`/`CALIBRATION_WITNESS_VERSION`/`LOCK_VERSION` literal
  anywhere needs updating** — none of these move (§6).
- **No OpenAPI snapshot/signature regeneration** — confirmed unnecessary,
  not merely unperformed (§6, empirically re-tested).
- **`REPRODUCIBILITY_AUDIT.md`**: R16 row, R26 row, new "Still open after
  Batch 27" trailer (§4).
- **`docs/PROJECT_ROADMAP.md`**: Phase 4 row, §1 table R16/R26 rows (§4).
- **No test file outside the one new file (§3) requires a source change** —
  re-confirmed by search for any test asserting `verdict`'s or `p_fake`'s
  description content, any test asserting `FaceTrack`'s docstring content,
  and any test asserting `BoundingBox`'s field count or constraint set:
  zero hits beyond `test_field_descriptions_survive_into_the_schema`
  (unaffected, §3).

---

## 10. Gate

## **APPROVE FOR IMPLEMENTATION**

Both specification decisions from `docs/R16_R26_DESIGN.md` §6 are resolved,
each re-derived from source rather than accepted from the prior document:

1. **R26 Tier 2 — option (a) vs (b).** Option (a) ships now (a pure,
   zero-decision regression test). Option (b) is deferred, explicitly, not
   decided here — a genuine contract-shape choice that should wait for
   SCRFD.
2. **Bundle or split.** One batch — zero file overlap, identical (none)
   impact profile on every axis, matches the pre-existing roadmap grouping
   and this project's dominant multi-finding-per-batch convention.

No new blocking issues were discovered in re-deriving `R16_R26_DESIGN.md`'s
conclusions from source a second time. The scope is **unchanged in
substance** from that document's own recommendation (§4 there) — this
resolution adds precision (exact proposed text, exact test names, an
exhaustive collateral search) rather than expanding or narrowing what gets
built. Zero version-axis movement, zero cache generation, zero deploy
coordination, zero stored-artifact migration, zero OpenAPI artifact change —
verified, not assumed, on every axis in §6.

Implementation may proceed directly against §2–§9 of this plan, subject
only to the procedural caution in §8's concurrent-session risk row
(re-verify batch number and line references live, immediately before
writing code).
