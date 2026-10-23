# R16 + R26 — Design (informing Phase 4)

**Date:** 2026-09-18. **Type:** design only — no code, no tests, no docs
changed. **Method:** complete source re-derivation. Every claim below was
independently re-checked against the current working tree (`git status`
read first; `HEAD` = `8676fdd`, same uncommitted OR1-b/R30 tree as the
verified baseline — no drift since Batch 26) rather than assumed from
`docs/PROJECT_ROADMAP.md`, `REPRODUCIBILITY_AUDIT.md`, or any prior design
doc. Those documents are **inputs to be verified, not premises.**

**Verified baseline going in** (re-confirmed live, not carried over):
53 suites, `STUB_ENGINE_VERSION` ends `.d.c9d4025c`, `CONTRACT_VERSION`
1.0.0, `RUN_SCHEMA_VERSION` 1.2.0, `CALIBRATION_WITNESS_VERSION` 1.3.0,
`LOCK_VERSION` 1.1.0 — all read live from `contracts/enums.py`,
`evaluation/benchmark.py`, `evaluation/calibration_protocol.py`,
`evaluation/split_lock.py`. No SCRFD (`inference/preprocess.py:detect_faces`
still calls a Haar `CascadeClassifier`, confirmed by grep — no `scrfd`
import anywhere in production code, only prose references to the future
migration).

---

## 1. Executive summary

**Both findings are real, both remain open, and neither has been touched by
N5/OR1-b/R30/R31** — those four batches are decode-identity and
calibration-witness plumbing with zero code-path overlap with face
detection, crop geometry, or contract publishing (verified by grep: no
reference to `FaceTrack`, `BoundingBox`, `crop_face`, or `EvidenceBundle` in
any of `inference/decode_env.py`, `backend/worker_main.py`,
`evaluation/calibration_protocol.py`'s R31 diff).

**R16's prerequisites are already satisfied** — a finding this re-derivation
adds, not carried over. The original design doc
(`docs/MULTI_FACE_SEMANTICS.md` §12, "Option 0") named R17 (selection policy
outside every identity) and R21 (`faces[]` has no consistency invariant) as
blocking preconditions for any R16 implementation. Both are independently
confirmed closed in current source: R17's identity coverage is `.f.`
(`selection_identity`, extended twice more since — R25/B22, R29/B23) and
R21's two validators (`_faces_are_uniquely_identified`,
`_headline_is_the_score_of_a_published_face`) are live in
`contracts/evidence.py` today. **This does not change the recommendation**
(the *engineering* path is still gated on SCRFD, for an unrelated reason —
R13's detector instability, not R17/R21) but it does mean the "Option 0"
framing in the 2026-08-era doc is stale and should not be re-litigated as if
still pending.

**R26 splits into three tiers, not two.** The roadmap's current
"documentation half (S) / full fix (L, blocked)" framing collapses a
provably-safe correctness fix into the same bucket as the SCRFD-blocked
geometry redesign. Re-reading `inference/postprocess.py:679-685` directly
shows `face_track()` clamps a detector box's x/y to `max(0, x)` before
constructing the published `BoundingBox` (`ge=0` constrained) — a *different*
computation from `crop_face`'s own clamp (`x0 = max(0, cx - side//2)`,
centred on the margin-expanded window, not the raw box). Under Haar this is
unreachable (`cv2.CascadeClassifier.detectMultiScale` cannot return a
negative-origin box — structurally guaranteed, not merely observed) — the
same "provably output-identical under the shipped detector" property that
let Batch 14 close R19 *before* SCRFD. That means **one real piece of R26 can
be closed now, safely, independent of SCRFD** — narrower than the roadmap's
current binary framing suggests.

**Both findings can close a meaningful cheap/documentation slice before end
of October with zero version churn.** Neither can close *fully* before
SCRFD exists — that conclusion is unchanged and independently re-confirmed.

**Correction made during this re-derivation, stated up front because it
changes a risk claim made repeatedly in prior docs:** the 2026-08-era
`MULTI_FACE_SEMANTICS.md` states Option 1 (R16's cheap path) *"does move the
OpenAPI snapshot."* **This is not true of the current drift-detection
mechanism** — verified by reading `backend/openapi.py`'s
`_component_signature`/`_type_of` (lines 209-251) directly and confirming
empirically: the committed `backend/openapi.snapshot.json` stores only a
`fingerprint` of a `structural_signature` — `required` + `{field: type}` per
component, with **no `description` field captured at any level, model or
field.** A live test, `tests/test_openapi.py::test_signature_ignores_
prose_but_not_structure`, already proves exactly this property for a
model-docstring change; this re-derivation confirms it also holds for a
per-`Field(description=...)` change and for removing a `minimum`/`ge`
constraint (tested directly: two synthetic schemas differing only in
`BoundingBox.x`'s `minimum` produce identical signatures and fingerprints).
**Net effect: every change recommended in §4 below is safer than the
original doc believed** — none of them move the checked-in OpenAPI artifact
at all, not even the "expected, description-only" drift the original doc
predicted. This does not change *what* to build, only removes a migration
step (regenerating/reviewing an OpenAPI snapshot diff) that turns out to be
unnecessary.

**R23/R28 should remain deferred.** Unchanged — `crop_face` and `resize`'s
`or`-predicate are byte-identical to every prior description; the reasons
they were deferred (no real model to validate a pixel change against) are
unaffected by anything landed since.

**One roadmap item has quietly changed in scope, not obsoleted: R31 adds a
new dependency for any future `.f.`-moving batch.** `ModelSpec.
pre_calibration_identity` (weights + gate + **selection**) is now inside
`CalibrationWitness`'s digest for schema 1.3.0 — confirmed by reading
`inference/models.py:313-328` directly. Any R16/R26 work that ever touches
`FaceSelectionPolicy` (neither of the cheap-path recommendations below does)
would, for the first time, also need to reason about witness compatibility —
low practical risk today (zero stored witnesses exist, confirmed by the same
search R31 itself performed) but a new fact, not previously true when
`docs/MULTI_FACE_SEMANTICS.md`/`docs/CROP_GEOMETRY.md` were written.

---

## 2. Re-derived findings

### 2.1 R16 — one face's verdict is published as the media's

**Current status (re-derived, not quoted):** Open. `contracts/common.py:99-102`
— `FaceTrack`'s own docstring already states the policy honestly: *"Images
yield exactly one track — the largest detected face, whose verdict is
published as the whole image's. That is a policy the product chose for
single-subject media, not a property of images; it is R16 and it is open."*
This is a finding this re-derivation surfaces that the roadmap's row does
not mention: **the `FaceTrack` docstring has already had Option 1's
treatment applied to it** (at some point between the original Batch 10
finding and now — not attributable to a specific batch from the audit
trail, but present and correct today). **What has *not* been done**: the
top-level `verdict`/`p_fake` field descriptions in `contracts/evidence.py`
(lines 45-55) carry no equivalent caveat — `p_fake`'s description is
*"Calibrated probability of manipulation. Null iff..."*, silent on *whose*
face. A client reading `EvidenceBundle`'s own most-referenced fields still
cannot discover the single-subject assumption without finding `FaceTrack`'s
docstring first. Option 1 is **partially, not fully, implemented.**

**Original concern:** `analyze_image` scores `faces[0]` and publishes one
`FaceTrack`; a multi-subject photograph's verdict describes one person in
it, undocumented at the point a consumer would actually read it.

**Does the concern still exist?** Yes, in the narrowed form above — the
policy is now documented in one place (`FaceTrack`) but not the other
(`EvidenceBundle`'s own Stable-tier fields). Verified live:
`contracts/evidence.py:45-58` re-read in full; no mention of face selection,
single-subject, or `faces[0]` anywhere in the `verdict`/`p_fake`/`interval`
field descriptions.

**Cheap / documentation closure path:** Finish Option 1. Add one clause to
`p_fake`'s and `verdict`'s `Field(description=...)` (or a class-level
docstring note on `EvidenceBundle`, consistent with how `FaceTrack` did it)
stating the prediction describes the largest detected face per
`face_order_key`. Zero behaviour change. **Does NOT move the checked-in
OpenAPI artifact** — corrected from `docs/MULTI_FACE_SEMANTICS.md` §12's
claim that Option 1 "does move the OpenAPI snapshot" (see the executive
summary's correction note: `backend/openapi.snapshot.json`'s structural
signature captures no description field at any level, verified by reading
`backend/openapi.py:241-251` and confirmed empirically). `test_openapi.py`'s
drift check therefore **passes unchanged**, not "trips by design." Separately,
`tests/test_schema.py::test_field_descriptions_survive_into_the_schema`
operates on the full, uncompressed pydantic schema (`sch.schema_for(...)`,
a different code path from the OpenAPI signature) and would need its
asserted substring updated if the new caveat text replaces rather than
appends to the existing description.

**Full engineering closure path (Option 2 — publish per-face evidence,
recommended by the original design, re-confirmed still the right shape):**
Score every face passing the floors; publish one `FaceTrack` per face with a
real per-face `p_fake`/`status`; keep `verdict`/`p_fake`/`interval` meaning
exactly what they mean today (the largest face's). **Contract impact
re-verified as none requiring a version bump**: `faces[]` is listed
Evolving-tier in `contracts/evidence.py`'s own stability-tier docstring
(lines 15-19, read directly), and `FaceTrack` gaining no new fields under
Option 2 (only more instances) means even the Evolving-tier "may reshape"
clause isn't invoked. **Not recommended before SCRFD, and this is
independently re-confirmed, not carried over**: `inference/preprocess.py`
still runs the Haar cascade (verified — no SCRFD import anywhere), and R13
(310× amplification, 25.7% of decisions, still fully open, unchanged since
last measured) means per-face scores published under Option 2 today would
be — in the original doc's own words, re-verified as still accurate —
*"mostly a report on the placeholder's eyesight."* Additionally: Option 2
needs a new per-file face-cap setting (confirmed absent — grepped
`backend/config.py` and `inference/*.py` for `face_cap`/`max_faces`, zero
hits) that interacts with the lease heartbeat `tests/test_lease_safety.py`
guards — real new plumbing, not a trivial addition.

**Files affected (cheap path):** `contracts/evidence.py` (2 field
descriptions) only. `backend/openapi.snapshot.json` is **not** touched (its
structural signature is description-blind, corrected above); `schema_out/`
is a gitignored, on-demand-generated directory (`git ls-files schema_out/`
returns nothing — confirmed this session), not a repo artifact that
"changes."

**Files affected (full path, not recommended now):** `inference/
image_pipeline.py` (score every passing face, not just `faces[0]`),
`backend/config.py` (new face-cap setting), `inference/postprocess.py`
(`face_track()` called per face), `tests/test_lease_safety.py` (cap
interaction), `contracts/evidence.py` (no field change, but the invariant
tests around R21 need a genuine multi-face fixture, currently absent —
confirmed by grep, every `FaceTrack` fixture in `contracts/fixtures.py`
constructs `faces=[...]` with exactly one entry).

**Version impact (cheap path):** `CONTRACT_VERSION` unchanged (1.0.0 —
description-only, not a Stable-field meaning change under the platform's own
rule, since the *meaning* — "the largest face's score" — is not changing,
only becoming stated). `backend/openapi.snapshot.json`'s `fingerprint`/
`signature` **do not move** (corrected above — `tests/test_openapi.py::
test_signature_ignores_prose_but_not_structure` already proves the
signature is description-blind for a model docstring; this re-derivation
confirms the same for a `Field(description=...)` change). `.f.`/`.d.`/
`+c.`/`.g.` all unmoved — no code path touched. `RUN_SCHEMA_VERSION`/
`CALIBRATION_WITNESS_VERSION`/`LOCK_VERSION` unmoved.

**Version impact (full path, informational only, not recommended now):** No
`CONTRACT_VERSION` bump (Evolving-tier). `.f.` **would** move only if a new
selection-affecting knob (e.g. the face cap, if it ever gated *which* face is
`faces[0]`) were added — the current Option 2 scope explicitly keeps the
headline computation untouched, so **no identity move is required by Option
2 as scoped**. If ever implemented, this is the first `.f.`-moving-adjacent
change since R31 landed `pre_calibration_identity`, so `CalibrationWitness`
compatibility would need a fresh check at that time (§1) — though as scoped,
`.f.` does not move, so this is a non-issue *unless* scope grows.

**Migration impact:** Cheap path — none; no OpenAPI artifact changes, no
cache generation, no deploy coordination. Full path — a latency/cost-model
change (N model forwards instead of 1), not a cache-identity migration.

**Test impact (cheap path):** `tests/test_openapi.py` should be **re-run to
confirm** (not to regenerate) that the fingerprint/signature and drift check
stay green — no source change to that test or its snapshot expected.
`tests/test_schema.py`
(`test_field_descriptions_survive_into_the_schema`-style check extended, if
the project wants a tripwire naming the exact clause — optional), no
existing test breaks (verified: no test currently asserts the *content* of
`p_fake`'s or `verdict`'s description string).

**Risk assessment:** Cheap path — **None**. Pure prose, mechanically
regenerated snapshot, no behaviour path touched. Full path — **Medium**
(new compute cost model, new setting, new test surface) but explicitly
**not the risk that should gate scheduling** — the *data quality* risk
(per-face scores from an unstable placeholder detector) is what the original
design correctly identified as the real reason to wait for SCRFD, re-confirmed
unchanged today.

**Estimated effort:** Cheap path — **XS**, well under a day. Full path — **M**
(the original doc's own estimate, unchanged), and should not be scheduled
independent of the SCRFD program.

**Recommendation:** Close the cheap path now (finish Option 1 — extend the
existing `FaceTrack`-only treatment to `EvidenceBundle`'s own Stable
fields). Do not schedule Option 2 as a standalone batch; it belongs with the
SCRFD program per `docs/PROJECT_ROADMAP.md` Phase 6, unchanged.

---

### 2.2 R26 — the published `bbox` is not the region the model scored

**Current status (re-derived, not quoted):** Open, and — this re-derivation's
main finding — **splits into three independently-schedulable tiers**, not
the roadmap's current two.

**Original concern:** (1) `EvidenceBundle`/`FaceTrack` carries the
detector's raw box, not the expanded/clipped/resampled window the model
actually saw — a documentation/completeness gap. (2) `face_track()`'s
coordinate handling diverges from `crop_face()`'s, latent under Haar,
live the moment a detector regresses a box crossing the frame edge.

**Does the concern still exist? Verified against current source, both
halves:**

- **Documentation gap — still exists.** `contracts/common.py`'s `FaceTrack`
  (lines 93-127, read in full) has exactly five fields: `track_id`, `bbox`,
  `landmarks`, `frames_seen`, `p_fake`/`status`. No field records the crop
  window `crop_face` actually computed. Confirmed absent, not assumed.
- **Clamp divergence — still exists, and now precisely characterised.**
  `inference/postprocess.py:675-685` (`face_track()`, read in full):
  ```python
  def face_track(box_xywh, *, p_fake, frames_seen=1, status=None):
      x, y, w, h = box_xywh
      return FaceTrack(
          track_id=0,
          bbox=BoundingBox(x=max(0, x), y=max(0, y), width=max(1, w), height=max(1, h)),
          ...
      )
  ```
  vs. `inference/preprocess.py`'s `crop_face` (read in full, §3 window
  computation): `side = int(max(box.width, box.height) * margin)`,
  `cx, cy = box.x + box.width // 2, box.y + box.height // 2`,
  `x0 = max(0, cx - side // 2)`. These are **different quantities clamped at
  different reference points** — `face_track` clamps the raw box origin;
  `crop_face` clamps a margin-expanded window's origin computed from the
  box's *centre*. A box with `x = -20` (top-left corner outside the frame,
  representable by a real detector, rejected today only because Haar
  structurally cannot produce it) would be published as `x = 0` — silently
  reporting a box shifted right of where the detector actually put it —
  while `crop_face`'s window is computed from the true (unclamped) centre
  and may not even need clamping if the box is wide enough. `BoundingBox.x`/
  `.y` (`contracts/common.py:87-88`) are contract-constrained `ge=0`, which
  is *why* `face_track()` clamps at all — without it, a real detector's
  negative-origin box would raise a `pydantic.ValidationError` at
  construction, not merely publish an imprecise number.

**Tier 1 — cheap / documentation closure path.** Correct any doc claims that
overstate what `bbox` represents (checked — `BoundingBox.bbox`'s current
description, *"Representative box, in source-image coordinates,"* is
already honest; it does not claim to be the scored region, so there is no
false claim to retract here, unlike R23's pre-Batch-15 `crop_face`
docstring). The real cheap-path work is **adding the caveat that does not
exist yet**: a class-level note on `FaceTrack` (matching its existing R16
paragraph's style) stating that `bbox` is the detector's box, not the crop
the model scored, and that no field currently carries the latter.

**Tier 2 — correctness-hardening path, NOT SCRFD-blocked (this
re-derivation's central finding).** Fix the clamp divergence itself,
independent of any geometry redesign. This does not require resolving R23's
open pixel-geometry question — `crop_face` can stay exactly as it is. It
requires only that `face_track()`'s published coordinates be *provably
harmless* the way B14 made R19's fix provably harmless: either (a) prove by
construction that the clamp is a structural no-op for every box the shipped
detector can emit (a test pinning `detect_faces`'s Haar-backed output to
never contain a negative coordinate, converting a "this happens to be true
today" fact into a "this cannot silently regress" guarantee), or (b) loosen
`BoundingBox.x`/`.y`'s `ge=0` constraint and stop clamping, publishing the
detector's true box. **(b) is a contract type change**, but its blast
radius is small — `BoundingBox` is constructed in exactly one production
call site (`inference/postprocess.py:681`, confirmed by repo-wide grep) plus
six fixture literals and three test files, all non-negative already, so no
existing valid data becomes invalid. Both (a) and (b) are output-identical
under Haar (`detectMultiScale` cannot return a negative origin — a
structural property of the algorithm, not a measured negative result that
could quietly stop being true, unlike some of the platform's other
"measured, not proven" residuals). This is a genuinely different risk class
from R23/R28: those two change pixels or amplify a scoring value under the
shipped detector today; **this fixes a data-honesty problem in a code path
the shipped detector cannot currently reach at all**, so there is nothing to
validate against a real model — the fix is either correct by construction
or it is not, and that can be proven with a unit test, not measured on a
benchmark.

**Tier 3 — full engineering closure path, SCRFD-blocked (unchanged from the
roadmap).** Publish the actual scored region as a new field. This is
entangled with R23 by construction — "the scored region" is defined by
whichever crop-geometry option R23/SCRFD eventually chooses (`docs/
CROP_GEOMETRY.md` §9's options A-F), so it cannot be built ahead of that
decision without being rebuilt when it lands. Unchanged conclusion,
re-confirmed: `docs/CROP_GEOMETRY.md`'s own §9 "Recommendation" (*"Do not
change crop geometry in this batch, and do not change it before SCRFD"*)
still applies to whatever pixels-under-the-hood question a `scored_region`
field would need answered.

**Files affected (Tier 1):** `contracts/common.py` (`FaceTrack` docstring
only).

**Files affected (Tier 2):** `inference/postprocess.py` (`face_track()`,
if option (b)), `contracts/common.py` (`BoundingBox.x`/`.y` constraint, if
option (b)), `tests/test_faces_consistency.py` or a new
`tests/test_face_geometry_publication.py`-style suite (the regression
proof), `tests/crop_geometry_study.py` (already names this exact divergence
at line 287 — extend, don't duplicate).

**Files affected (Tier 3, not recommended now):** `contracts/common.py`
(new `FaceTrack` field), `inference/preprocess.py` (`crop_face`, whichever
option R23 chooses), `inference/postprocess.py`, the full R23/R28 test and
doc surface (`docs/CROP_GEOMETRY.md`, `docs/CROP_IDENTITY.md`, `tests/
test_crop_geometry.py`, `tests/test_crop_identity.py`) — this is R23's
implementation, not a standalone R26 batch.

**Version impact (Tier 1):** None. Prose only. **Correction to an assumption
tested and rejected this session:** a class docstring *does* reach the raw
pydantic schema (`FaceTrack.model_json_schema()`'s top-level `description`
key — verified directly by generating it) — it is not, as first assumed,
invisible to schema generation. It still does not move
`backend/openapi.snapshot.json`, but for the reason established in the
executive summary's correction note: that file's structural signature
strips every description, model-level and field-level alike, not because
docstrings are special-cased out of generation.

**Version impact (Tier 2):** If option (a) (test-only): none — no source
behaviour change. If option (b) (loosen the constraint): **no
`CONTRACT_VERSION` bump** — `BoundingBox`'s `ge=0` is a *narrowing*
constraint being *removed*, which is a strictly more permissive change
(every currently-valid bundle stays valid); the platform's own Stable/
Evolving rule (`contracts/evidence.py:9-19`) gates *meaning* changes to
Stable fields, and `faces[]`/`FaceTrack` is Evolving-tier regardless.
**Tested directly this session, contradicting the initial assumption**: the
OpenAPI signature does **not** move either — `_type_of()`
(`backend/openapi.py:209-238`) reads only a node's `type`/`$ref`/`enum`/
combinator shape, never `minimum`/`maximum`/`exclusiveMinimum`; two
synthetic component schemas differing only in `x`'s `minimum` produced
identical `structural_signature()` output and identical `fingerprint()`
hashes when run against the live code this session. `test_openapi.py`
requires **no change** for option (b) either. `.f.`/`.d.` unmoved — no
identity-affecting code touched.

**Version impact (Tier 3):** Deferred with R23 — same profile as R23's own
entry in `docs/PROJECT_ROADMAP.md` §2 (`.f.` moves once, weights/geometry
test suite, no real model to validate the pixel change against yet).

**Migration impact:** Tier 1 — none. Tier 2 — none (no cache generation, no
`.d.`/`.f.` move, no deploy coordination, no OpenAPI artifact change under
either option). Tier 3 — rides R23's migration profile entirely.

**Test impact:** Tier 1 — none broken, one doc-only addition. Tier 2 — new
regression test(s) proving the Haar-cascade output space excludes negative
origins (a structural proof, not a sample-based measurement); if option (b),
`tests/test_faces_consistency.py` and any fixture asserting `BoundingBox`
rejects negative input needs updating (grepped — none currently do; the
`ge=0` constraint has no test explicitly exercising its rejection path,
confirmed by searching `tests/` for `BoundingBox` + `ValidationError` /
`pytest.raises`-style patterns — none found, so no test needs to flip).
Tier 3 — full R23 test surface.

**Risk assessment:** Tier 1 — **None**. Tier 2 — **Low** — the change is
either a pure test addition (option a) or a constraint loosening with a
small, fully-enumerated blast radius (option b); both are provably
output-identical under the only detector that exists. Tier 3 — **Medium**,
unchanged from R23's own entry (moves pixels, no real model to validate
against).

**Estimated effort:** Tier 1 — **XS**. Tier 2 — **S** (a half-day to a day —
mostly the regression-proof test; option (b)'s contract edit is one line).
Tier 3 — **L**, rides SCRFD, unchanged.

**Recommendation:** Close Tiers 1 and 2 now, together — they are cheap,
genuinely safe, and Tier 2 closes a real (if currently unreachable) data-
honesty gap before SCRFD makes it reachable, matching the platform's own
stated philosophy for R19/R20/R21 in `docs/MULTI_FACE_SEMANTICS.md` §13
(*"fixing them after they have started producing wrong ... means fixing them
after they have started"* — doing it before costs an afternoon and makes it
unreachable by construction). Do not attempt Tier 3 before SCRFD/R23.

---

## 3. Cross-cutting determinations

### 3.1 Can R16 and R26 close before end of October without version churn?

**Yes, for the cheap/hardening tiers identified above — and this is a
stronger "yes" than the roadmap's current framing implies**, because Tier 2
of R26 was not previously distinguished from Tier 3. Concretely, before end
of October, all with **zero `CONTRACT_VERSION`, `.f.`, `.d.`, `RUN_SCHEMA_
VERSION`, `CALIBRATION_WITNESS_VERSION`, or `LOCK_VERSION` movement**:

- R16: finish Option 1 (extend the existing `FaceTrack`-only documentation
  to `EvidenceBundle`'s Stable fields). Neither the OpenAPI fingerprint nor
  signature moves (corrected in the executive summary — the checked-in
  artifact is description-blind).
- R26 Tier 1: add the missing `FaceTrack` caveat about `bbox` vs. the scored
  region. Same — no OpenAPI artifact movement.
- R26 Tier 2, option (a) (recommended over (b) for a first pass — strictly
  additive, zero contract risk): a structural regression test proving the
  Haar detector cannot emit what `face_track()`'s clamp defends against,
  converting a currently-true fact into a permanently-guarded one. Option
  (b) (loosen `BoundingBox`) is available as a follow-up if the team wants
  the contract to state the truth outright rather than merely guard it — it
  does **not** move this project's OpenAPI drift check either (tested
  directly), but it is still a genuine, real constraint change (any
  external client-SDK generator reading the full schema, not just this
  repo's compact signature, would see `x`/`y` stop excluding negative
  values) and should be a deliberate choice for that reason, not because it
  trips a test that, in fact, it does not.

**No**, for Option 2 (R16 full) or Tier 3 (R26 full) — both are correctly
gated on SCRFD, which does not exist and is not in scope for end-of-October.

### 3.2 Should R23/R28 remain deferred?

**Yes, unchanged, re-confirmed from current source rather than assumed.**
`inference/preprocess.py`'s `crop_face` (window/clip logic) and `resize`
(the `INTER_AREA if h>size or w>size` predicate, R28) are byte-identical in
shape to every prior description — read in full this session, no batch
between B22 and now has touched either function (confirmed: N5/OR1-b/R30/R31
are decode-identity, dependency-lock, worker-composition and calibration-
witness work respectively; none imports or references `inference.preprocess`
at all except R30's `tests/test_image_decode_identity.py`'s own structural
check that `decode_env.py`'s functions are never called from the pixel path
— which is a statement about `decode_env`, not about `crop_face`/`resize`).
The reason to defer — no real model exists to validate a pixel-moving change
against, so any fix ships "on structural argument alone," the exact
pattern that triggered R30's own adversarial-review cycle — is unaffected by
anything landed since. **No new information changes this call.**

### 3.3 Has any roadmap item become obsolete because of N5, OR1-b, R30, or R31?

**No finding is obsoleted.** Re-checked against the current
`docs/PROJECT_ROADMAP.md` §1 table, row by row, for any claim whose
precondition N5/OR1-b/R30/R31 might have changed:

- **N8** — already closed (`docs/B24_R2_DESIGN.md` §4, written back in the
  OR1-b pass). Not newly obsoleted here; already accounted for.
- **R32** (`batch_size` neutrality, backend-dependent) — still tracked,
  conditional on ONNX/TensorRT/attention-pooling, none of which R30 or R31
  touched (both are stub-backend/decode-environment work). Unaffected.
- **R34** (OR1 non-rolling-deployable without coordination) — OR1-b *doubled*
  the escape-hatch surface (env var + `Settings.skip_version_alignment`)
  but did not change the underlying constraint; R34 stays "tracked,
  operational," unaffected in substance.
- **N6/N7** (fit-optimiser config / `np.quantile`) — R31's own re-derivation
  already corrected these (N7 found not applicable, N6 confirmed real but
  zero live variability) — already reflected in the current audit doc, not
  a new finding here.

**One item's *scope* changed without becoming obsolete** — already stated in
§1: R31's `pre_calibration_identity` adds a new (currently dormant, zero
live witnesses) dependency for any future `.f.`-moving batch, including a
hypothetical future R16/R26 engineering closure. This is forward-looking
information, not a correction to a stale claim.

---

## 4. Recommended path

1. **R16 cheap path** — extend `EvidenceBundle`'s `verdict`/`p_fake` field
   descriptions with the single-subject caveat `FaceTrack` already states.
2. **R26 Tier 1** — add the missing `bbox`-vs-scored-region caveat to
   `FaceTrack`'s docstring.
3. **R26 Tier 2, option (a)** — a structural regression test proving Haar's
   output space cannot trigger `face_track()`'s clamp, so the current
   safety is guarded rather than incidental.
4. Bundle 1-3 as one small batch — no version movement, no cache generation,
   no OpenAPI artifact change at all (corrected from the original "snapshot
   regeneration expected" assumption) — do not bundle with anything
   SCRFD-scoped.
5. **Explicitly do not schedule**: R16 Option 2, R26 Tier 3/option (b),
   R23-border-clip + R28, or anything else gated on SCRFD. These remain
   Phase 6+ per `docs/PROJECT_ROADMAP.md`, unchanged by this re-derivation.
6. **Product decision still needed, not an engineering one**: whether R26's
   option (b) (loosen `BoundingBox` and stop clamping, vs. option (a)'s
   test-only guard) is worth doing now or should wait — not because it
   trips this repo's OpenAPI drift check (it does not, verified), but
   because it is a genuine constraint change any external schema consumer
   would see. This re-derivation recommends (a) for the October window as
   the strictly additive, zero-decision-required option, leaving (b) for
   whenever the team is ready to state the contract's true constraint
   deliberately.

---

## 5. Exact file list for implementation (if approved)

| File | Change | Tier |
|---|---|---|
| `contracts/evidence.py` | `verdict`/`p_fake` field descriptions gain the single-subject caveat | R16 cheap |
| `contracts/common.py` | `FaceTrack` docstring gains the bbox-vs-scored-region caveat | R26 Tier 1 |
| New test (suggested location: `tests/crop_geometry_study.py` extension + a new assertion in `tests/test_faces_consistency.py` or a new `tests/test_face_geometry_publication.py`) | Structural proof that Haar's `detectMultiScale` cannot emit a negative-origin box, pinning `face_track()`'s clamp as provably-unreachable rather than incidentally-safe | R26 Tier 2 (option a) |
| `tests/test_openapi.py` | No source change; re-run to confirm the fingerprint/signature stay exactly as they are (no drift at all expected, not merely "expected, description-only" drift) | R16 cheap verification |
| `tests/test_schema.py` | No source change; re-run to confirm `test_field_descriptions_survive_into_the_schema`-class coverage still passes | R16 cheap verification |
| `REPRODUCIBILITY_AUDIT.md` | R16/R26 rows updated on implementation (not in this design task) | doc, post-implementation |
| `docs/PROJECT_ROADMAP.md` | Phase 4 marked done on implementation (not in this design task) | doc, post-implementation |

**Not touched by this batch, listed to be explicit about scope discipline**:
`inference/preprocess.py`, `inference/image_pipeline.py`,
`inference/postprocess.py` (function bodies — only a docstring/comment
context, no logic change), `backend/config.py`, `inference/decode_env.py`,
any `.f.`/`.d.`-affecting code, any R23/R28/R13/R18/SCRFD-scoped file.

---

## 6. Gate decision

## **APPROVE WITH CHANGES**

The cheap/hardening path (§4, items 1-4) is technically ready and
low-risk enough to implement largely as specified, but two decisions in
this document are **product/team calls, not engineering ones**, and
implementation should not begin until they are made explicit rather than
defaulted:

1. **R26 Tier 2 — option (a) vs (b).** This design recommends (a) (test-only
   guard) for the October window. If the team instead wants the contract to
   state the true constraint (option b — loosen `BoundingBox.x`/`.y`), that
   is equally safe and — corrected from this document's own earlier draft —
   does **not** move this repo's OpenAPI drift check either; it should still
   be chosen consciously rather than folded into "the cheap batch" silently,
   because it is a real constraint change for any external schema consumer,
   not because a test would fail.
2. **Whether to bundle the R16 and R26 cheap-path work into one batch or
   two.** Both are independent (no shared file, no shared code path, and —
   corrected from this document's own earlier draft — no shared OpenAPI
   artifact regeneration either, since neither change touches it), so they
   could ship as separate, smaller reviews if the team prefers finer-grained
   history over this design's default of bundling them.

Neither of these is a blocking defect — both are within the "approve with
changes" latitude the project's own design-review convention uses (per
`docs/B24_R2_DESIGN.md`'s own gate pattern: technically sound,
implementation-ready, with named open decisions rather than open
defects).

**Stop here — no implementation until explicit approval, per instruction.**
