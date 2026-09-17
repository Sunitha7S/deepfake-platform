# Headline and face semantics — architecture review of R20 and R22 (Batch 18)

> **Identity note (Batch 22).** §5's migration table cites the pinned constant
> `stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387`. Batch 22 (R25) moved it to
> `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9` (crop-geometry knobs into
> the identity — unrelated to R20/R22). `_EXPECTED_EFFECTIVE` and
> `test_correcting_the_description_moves_no_runtime_identity` now pin the new
> value. **Batch 23 (R29) moved the `.f.` segment again** to the composite
> `e629642c` (model identity `…f.e629642c`), folding in the video
> frame-selection / pooling policy; `_EXPECTED_EFFECTIVE` pins that. See
> `docs/VIDEO_SELECTION_IDENTITY.md`. R20/R22 remain prose-only and moved nothing
> themselves.

> **Outcome.**
> **R20 — CLOSED.** `QualityReport.face_px`'s description was a four-word sentence
> that named no metric for "largest" and mentioned neither the `0` sentinel nor
> the video `null`. Batch 14 (R19) had already removed the *semantic* half of the
> finding by collapsing ranking, gating and reporting onto one quantity
> (`FaceBox.size_px`); what was left is prose. This batch corrects the prose. The
> correction changes one `Field(description=...)` string and the four generated
> schema files that embed it — **no field, constraint, behaviour, identity
> string, contract version, schema version, calibration witness or cache key
> moves**, proved by rendering the schema with and without the change and
> diffing (§4).
>
> **R22 — already CLOSED, independently, by Batch 17** (`docs/BRANCH_B_SEMANTICS.md`),
> by a concurrent session, while this review was in progress. This document does
> not redo that work; §2 answers the brief's R22 questions by pointing at the
> Batch 17 evidence and **re-verifying its load-bearing invariants from scratch**
> (`tests/test_headline_and_face_semantics.py` §6). It also records and repairs a
> defect the Batch 17 suite shipped with (§5).

Proof and named doubles: `tests/test_headline_and_face_semantics.py` (28 tests).
The finding was re-derived from code and tests, not assumed.

**State reviewed.** 45 suites, `CONTRACT_VERSION` 1.0.0, `RUN_SCHEMA_VERSION`
1.2.0, `CALIBRATION_WITNESS_VERSION` 1.2.0. R12, R17, R19, R21, R22 closed; R18's
documentation half closed. Open before this batch: R10, R13, R16, R18 capability
half, R20, R23–R26, R6 `frame_count` residual, R9 design note.

---

## 0. The collision, stated plainly

The brief for this batch was "Batch 17 — R20 and R22 together". While it was being
worked, a second session — two sessions have edited this repo concurrently
throughout S8 — completed **Batch 17, closing R22**, and committed
`docs/BRANCH_B_SEMANTICS.md`, `tests/test_branch_b_semantics.py`,
`tests/branch_b_semantics_study.py`, a `REPRODUCIBILITY_AUDIT.md` Batch 17
section, and prose edits to `contracts/enums.py`, `contracts/branch.py`,
`README.md`, `contracts/fixtures.py` and `tests/stub_handlers.py`.

This review therefore lands as **Batch 18** and is **R20-primary**. Batch 17's
R22 closure is not reopened, not rewritten, and not duplicated. Its reasoning is
sound and its tests pass (after §5's repair). The "R20 and R22 together" analysis
the brief asked for still happens — it is §1–§3 below — it simply concludes that
R22 was resolved in parallel and that the two findings never depended on each
other.

---

## 1. R20 — `QualityReport.face_px`

### A. What does `face_px` claim to represent today?

`contracts/common.py`, before this batch:

```python
face_px: int | None = Field(default=None, ge=0, description="Short edge of the largest face.")
```

The claim is the description string: **"Short edge of the largest face."** Type
`int | None`, constraint `ge=0`, default `None`. Tier: it sits inside
`QualityReport`, which is a **required** field of `EvidenceBundle`; `face_px`
itself is optional within it. It is not in the Stable-tier subset the frontend
pins against (that is `verdict`, `p_fake`, `interval`, `threshold`,
`branches[].{id,status,logit,p}`).

### B. What value is actually published?

`inference/image_pipeline.py`, `analyze_image`:

```python
quality = QualityFacts(
    width=width, height=height,
    face_px=faces[0].size_px if faces else 0,   # image path
    ...
)
```

`inference/video_pipeline.py`, `_analyze`:

```python
quality = QualityFacts(width=meta.width, height=meta.height, face_px=None)   # video path
```

`QualityFacts.to_report()` copies the value straight into `QualityReport`.
Tracing `faces[0]`:

* `faces` is ordered by `inference.preprocess.face_order_key`, whose tuple is
  `(-box.size_px, -box.area, box.y, box.x, box.width, box.height)` — a **total**
  order (R14) whose **leading component is `-size_px`** (R19, Batch 14).
* `FaceBox.size_px` is `min(width, height)`, and `FaceBox.short_edge` is the same
  expression. They are equal by construction; the two names exist so a future
  migration can change *which fact the platform uses* without touching *the fact
  about the rectangle* (`FACE_SIZE_METRIC = "short-edge"`).

So the published value is, exactly:

| Input | `face_px` | In words |
| --- | --- | --- |
| image, ≥1 detected face | `faces[0].size_px` | the short edge of the face whose short edge is the largest (area breaks ties) |
| image, no detected face | `0` | a **sentinel**, always accompanied by `gate_passed=False`, `gate_reason=NO_FACE_DETECTED` |
| video (any) | `None` | the video path runs no face stage (R18) |

Because ranking and reporting now read the *same* quantity, the first row is also
"**the largest face size present**" — pinned since Batch 14 by
`test_face_size_policy.py::test_the_published_face_px_is_the_largest_face_size_present`
(`face_px == max(THIN.size_px, GOOD.size_px) == 200`), and re-pinned here from the
reporting angle by `test_the_published_face_px_is_the_largest_short_edge_present`.

### C. Is the discrepancy documentation-only or semantic?

**Documentation-only, today.** It was **semantic before Batch 14** and the record
must not pretend otherwise:

> `docs/MULTI_FACE_SEMANTICS.md` §10 (the Batch 11 review) states: *"The value is
> `faces[0].short_edge`, the short edge of the largest-**area** face. Stage 2
> shows the two differ (100 vs 190 on the same pair of boxes)."*

That was true when written: pre-Batch-14 `face_order_key` led with `-box.area`,
so the report took the short edge of a box chosen by area, and the two diverge
for any box more than ~2 px from square. **Batch 14 made ranking lead with
`-size_px`**, which is the quantity the report takes, so the divergence is gone.

`test_the_batch_11_reading_of_the_value_is_refuted_by_execution` runs it: on the
witness pair `THIN(900×50)`, `GOOD(200×200)`, the Batch 11 computation
(`legacy_value`) returns `50`; the live pipeline publishes `200`. The Batch 11
statement about the *value* is now false, and this is the executed refutation
required by the "preserve and refute" rule — the §10 text stays, annotated.

What remains is genuinely only prose:

1. **"largest" names no metric.** A reader cannot tell from the sentence whether
   "largest" means area or short edge — and the honest answer, that the platform
   has exactly one face-size metric and it is the short edge, is exactly the
   thing worth saying. `test_the_bare_word_largest_is_ambiguous_before_the_fix`
   asserts the legacy string mentions none of `size_px` / `min(` / `area`.
2. **The `0` sentinel is undocumented.** A `0` return is not "a 0-pixel face"
   (impossible); it means "no face". `test_a_faceless_image_publishes_face_px_
   zero_and_a_failed_gate` shows it only ever co-occurs with a failed gate.
3. **The video `null` is undocumented on the field.** `FaceTrack`'s class
   docstring mentions it; `face_px`'s own description does not.

### D. Can the description be corrected without changing meaning?

Yes. The value is unchanged; only the sentence describing it changes. Shipped:

```
"Short edge — min(width, height), in source pixels — of the largest detected
 face, where face size is FaceBox.size_px (equal to the short edge): the one
 metric selection, the gate and this figure all read (see
 docs/FACE_SIZE_POLICY.md, R20). 0 when an image has no detected face; null on
 every video bundle, whose path runs no face stage (R18)."
```

This is a strict superset of the old sentence's information. It still says "short
edge" and "largest ... face"; it adds the metric, the units, the sentinel, the
null and a pointer. `test_the_corrected_description_is_true_of_every_published_
value` checks the sentence holds for each of the three rows in B.

### E. Does any fixture, test, validator, report, OpenAPI artifact or frontend assumption depend on the current wording?

| Consumer | Depends on the wording? | Evidence |
| --- | --- | --- |
| `contracts/common.py` | it **is** the source | — |
| `backend/openapi.snapshot.json` | **no** — prose is stripped from the structural signature (`backend/openapi.py` `PROSE_KEYS = {"description", "summary", "title", "examples", "example", ...}`) | `test_the_description_is_not_in_the_openapi_structural_signature`; the snapshot stores `"face_px": "integer\|null"` — a type, not a sentence |
| `schema_out/*.json` (4 files) | **carries the string**, but **nothing guards it** — `--check` only validates example payloads; `write_all` tests use a tmpdir and check existence, not content (Batch 13 recorded the same about `schema_out` staleness) | regenerated this batch anyway (`python -m contracts.schema`); `test_schema_out_is_regenerable_and_carries_the_current_prose` |
| `test_schema.py::test_field_descriptions_survive_into_the_schema` | **no** — it asserts `p_fake`'s description contains "full precision"; it never mentions `face_px` | read the test |
| `test_face_size_policy.py::test_the_published_face_px_is_the_largest_face_size_present` | **no** — pins the *value* (`== 200`), not the wording | read the test |
| fixtures (`contracts/fixtures.py`) | **no** — they carry integer `face_px` values (412, 298, 0, 41, 96…), never the prose | `test_no_fixture_or_example_payload_changes` |
| every `EvidenceBundle` validator | **no** — none reads `face_px` at all; `_gate` tests `faces[0].size_px` | `test_the_gate_does_not_read_face_px`, `test_nothing_computes_from_face_px` (AST: `face_px` never an operand of a compare or a binop anywhere in `inference/` + `contracts/`) |
| evaluation / benchmark / calibration | **no** — `face_px`, `QualityReport`, `QualityFacts` appear nowhere under `evaluation/` | `test_no_evaluation_artifact_reads_quality_or_face_px` |
| frontend | no frontend in this repo; `schema_out/` feeds an out-of-repo TypeScript generator that turns `description` into a JSDoc comment — cosmetic, not a type | `docs/schema.py` header |
| `docs/MULTI_FACE_SEMANTICS.md`, `docs/FACE_SIZE_POLICY.md`, `docs/FACE_SELECTION_IDENTITY.md` | quote or paraphrase the old string | handled: `MULTI_FACE_SEMANTICS.md` gets a superseding annotation; the Batch-12/14 docs are left as written (true when written) and refuted here |

Nothing that a client can break against depends on the wording.

---

## 2. R22 — branch B (the brief's question F), answered against Batch 17

Batch 17 closed R22. Full trace, proofs, quantification and R16 analysis:
[`docs/BRANCH_B_SEMANTICS.md`](BRANCH_B_SEMANTICS.md). The brief's sub-questions,
answered from that evidence and re-verified in
`tests/test_headline_and_face_semantics.py` §6:

* **What does Branch B claim to score?** Before Batch 17: *"DINOv2-B/14 frozen +
  probe, over the whole image"* (`enums.py` comment, `BRANCH_DESCRIPTIONS[B]`,
  `README.md`) — against branch A's *"aligned face crop"*.
* **What pixels are actually scored?** `ModelBackend.score(batch)` takes **one**
  tensor and returns one logit vector per branch; both pipelines call it once
  with one argument (`test_r22_the_backend_interface_is_still_single_tensor`
  re-confirms). So every branch scores the same tensor: the aligned face crop
  (image path) or the whole resized frame (video path). Branch B's "whole image"
  claim is violated in the image path; branch **A's** "face crop" claim is
  violated in the **video** path.
* **Wording or a contract-level semantic promise?** Wording. `BranchResult` has
  no `description` field; `BRANCH_DESCRIPTIONS` has no functional consumer
  (AST-checked); the markers `DINOv2` / `over the whole image` / `aligned face
  crop` appear in **no** schema and **no** OpenAPI document
  (`test_r22_the_whole_image_claim_reaches_no_wire_field`). Zero wire fields
  carry a *wrong value*; four (`branches[B].{logit,p}`, `disagreement`,
  `interval`) carry an *ambiguous meaning* under the two readings.
* **Which future R16 options get harder if the wording stays?** None get *harder*;
  the ambiguity makes the R16 fusion shape undecided. `fuse` /
  `manipulation_evidence` take one `dict[BranchId, BranchOutcome]`. B crop-scoped
  (what the code does) → R16 Option 2 = call `fuse()` once per face, **no
  signature change**. B whole-image → one image-level B pooled with each
  face-level A → a new `fuse` signature and a restructured renormalised mean.
  Batch 17 corrected the wording to "crop-scoped", so R16 is the simpler shape.
* **What wording is truthful under the current implementation?** "Branch B scores
  the same prepared tensor as branch A — the aligned face crop (image) or the
  whole resized frame (video); a genuinely whole-image branch B is the
  frozen-architecture target and is unbuilt." That is what Batch 17 shipped.

**Independent re-verification (this batch).** `test_r22_the_backend_interface_is_
still_single_tensor`, `test_r22_the_whole_image_claim_reaches_no_wire_field` and
`test_r22_and_r20_share_the_same_unmoved_identity` re-derive Batch 17's P1, P5 and
P7 from the committed artifacts. All pass.

---

## 3. G — dependency analysis

### Which future findings depend on R20?

**None block on it.** R20 is a leaf.

* **R16** (one face's verdict published as the media's) has a *soft* dependency:
  when R16 populates `faces[]` with more than one scored face, "the largest
  detected face" stops being a complete account of which face `face_px`
  describes. The corrected wording ("the face selection chose", tied to
  `size_px`) stays accurate under R16 — the platform still selects one headline
  face — but an R16 design that reports per-face sizes would add a field, not
  redefine this one. Recorded, not blocking.
* **R25** (crop-geometry knobs outside every identity) and the **SCRFD
  migration** do **not** re-arm R20. The reason R20 was ever "correctness-critical
  at SCRFD" (`MULTI_FACE_SEMANTICS.md` §10) was that free-aspect boxes would let
  the area-ranked report and the short-edge gate disagree. Batch 14's
  `FACE_SIZE_METRIC` unification removed that mechanism *structurally* — ranking,
  gating and reporting read one property, and SCRFD boxes go through the same
  property. `test_the_two_readings_agree_for_haar_shaped_boxes` shows the old
  divergence needed area-vs-short-edge; `size_px` ranking has no such split at
  any aspect ratio.

### Which depend on R22?

Per `BRANCH_B_SEMANTICS.md` §9: **R22 does not block R16, is not blocked by R16,
and closed independently.** No other finding references R22 as a prerequisite.
The residual capability question ("build a dual-input backend so B scores the
whole image") is a frozen-architecture item for the model stage; it gates
nothing.

### Does leaving either open create reproducibility risk?

**No, for both.**

* `face_px` is a **reporting-only** field. It is in no identity string
  (`face_selection_fingerprint` covers the *ordering rule* and the eight
  selection knobs, not the published figure), feeds no decision
  (`test_nothing_computes_from_face_px`), and is not in any cache key. A wrong
  *sentence* is a documentation-quality risk; it cannot make two runs disagree.
* Batch 17's §5 established the same for R22: 0 fingerprints, 0 cache impact.

The one real risk of leaving R20 open was *narrative*: the audit's own index and
`MULTI_FACE_SEMANTICS.md` §10 carried a value claim that Batch 14 had silently
falsified, so a reader trusting the docs would mis-describe the field. That is
fixed by closing it and annotating §10.

---

## 4. H — migration analysis, with proof

The change: `contracts/common.py` `face_px` `Field(description=...)` string, plus
`python -m contracts.schema` to re-render the four `schema_out/*.json` that embed
it.

**Proof method.** Render the schema bundle with the *old* one-line description and
with the *new* one, into two directories, and diff every file. Result
(`tests/` scratch, reproducible):

```
CreateUploadResponse.schema.json : 2 changed lines  (face_px.description, - / +)
EvidenceBundle.schema.json       : 2 changed lines  (face_px.description, - / +)
contracts.bundle.json            : 4 changed lines  (face_px.description ×2, - / +)
AnalyzeRequest / JobSpec / JobStatus / ProgressEvent / HealthResponse
  / AnalyzeAcceptedResponse / ErrorResponse / CreateUploadRequest / index.json : 0 changed lines
```

Every changed line is `face_px`'s `description`. Nothing else in any generated
artifact moves.

| Would the correction require… | Answer | Proof |
| --- | --- | --- |
| **`CONTRACT_VERSION` bump?** | **No** | The MAJOR-bump rule (`contracts/evidence.py:10`) is "requires a client migration". No field is added, removed or retyped; `int \| None` and `ge=0` are unchanged (`test_face_px_field_shape_is_unchanged`); every existing payload validates and round-trips byte-identically (`test_no_fixture_or_example_payload_changes`, `verify_examples() == []`). A `description` is annotation, not contract surface. `test_contract_version_is_unchanged`. |
| **`RUN_SCHEMA_VERSION` bump?** | **No** | `RUN_SCHEMA_VERSION` (`evaluation/benchmark.py`, 1.2.0) governs `BenchmarkRun` / evaluation JSONL. `face_px` / `QualityReport` / `QualityFacts` appear **nowhere** under `evaluation/` (`test_no_evaluation_artifact_reads_quality_or_face_px`). `test_evaluation_versions_are_unchanged`. |
| **`CALIBRATION_WITNESS_VERSION` bump?** | **No** | The witness digest covers the calibration-set identity, `n_samples`, threshold and band (`evaluation/calibration_protocol.py`, 1.2.0). `QualityReport` prose feeds none of it; the digest does not hash contract descriptions. Same test. |
| **`STUB_ENGINE_VERSION` bump?** | **No** | `effective_version = weights + calibration + gate + face-selection + decode` identities. A `Field` description is in none of the five. `test_correcting_the_description_moves_no_runtime_identity` asserts `ImagePipeline(...).model_identity == "stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387"` — the constant Batch 12/14/16/17 pinned — and that `model_version` still composes `<identity>.d.<decode>`. |
| **Cache invalidation?** | **No** | The content-cache key and job-idempotency key are `sha256(bytes) + model_version (= effective_version + decode identity)`. The description is in neither. `effective_version` is unmoved (row above), so no key changes and nothing stale is served. |
| **`schema_out/` regeneration?** | Yes — mechanical, **not** a migration | `$id` still reads `…/schema/1.0.0/…`; the bundle's `contract_version` is still `1.0.0`. Four files gain a longer `description` string for one field. No test compares committed `schema_out` to a fresh render, so this was strictly optional; done for tidiness and because Batch 17 explicitly declined to touch them. |
| **`backend/openapi.snapshot.json` regeneration?** | **No** | Prose is stripped from the structural signature (`PROSE_KEYS`). `test_the_description_is_not_in_the_openapi_structural_signature` proves neither the old nor the new string is in the signature; `test_the_committed_openapi_snapshot_does_not_carry_the_prose` proves the committed file never carried it. `test_openapi` passes untouched. |

**No data on disk changes meaning or shape.** Stored `EvidenceBundle`s,
benchmark JSONL history, calibration witnesses and split locks are all
untouched, and no identity parser gains or loses an era boundary.

---

## 5. Newly discovered — a defect in the Batch 17 test suite (recorded separately)

Per the brief ("record a more fundamental finding separately rather than folding
it in"): running the full suite at the Batch 17 baseline is **not green**.

```
=== S8-B17 branch B semantics ===
  FAIL  test_the_whole_image_claim_is_not_on_the_wire
        AssertionError: schema_out/EvidenceBundle.schema.json carries the claim
FAILED - 1 of 45 suites had failures (1150 tests total)
```

**Cause.** The assertion greps for the bare substring `"whole image"` in
`schema_out/EvidenceBundle.schema.json`. That matches `FaceTrack`'s **R16**
docstring — *"its verdict is published as the whole image's"* (`common.py`) —
which is unrelated to branch B, is serialized as the `faces` field description,
and was already present. Batch 17's own study (`branch_b_semantics_study.py` §1)
noted *"'whole image' does appear once in the live OpenAPI doc — in `FaceTrack`'s
R16 prose … unrelated"* — the test did not act on that note.

**Classification.** A **test defect** (over-broad match) in a shipped batch, not
a defect in the platform. Severity low: the branch-B claim genuinely reaches no
wire field; the test simply failed to express that.

**Repair (this batch, explicitly not silent).** `test_the_whole_image_claim_is_
not_on_the_wire` now asserts the specific markers `"over the whole image"` and
`"DINOv2"` are absent — the phrasing Batch 17's P5 and its study already use.
This matches the test's stated intent. Batch 17's reasoning and its other 17
tests are unchanged.

This is the sole change to a prior batch's files, it is called out here and in
the audit, and it makes the suite green.

---

## 6. Decision

| Finding | Classification | Justification |
| --- | --- | --- |
| **R20** | **CLOSED** | Pure documentation defect after Batch 14. The description now names the metric (`size_px` = `min(w,h)`), the `0` sentinel and the video `null`, and points at `FACE_SIZE_POLICY.md`. The correction is proven to change **only** that one string across every source and generated artifact (§4), to move no identity/version/cache key, and to be true of every value the engine publishes. The behaviour it once mis-described is already pinned by `test_the_published_face_px_is_the_largest_face_size_present` (B14) and re-pinned here. No residual: SCRFD does not re-arm it (the area-vs-short-edge mechanism is structurally gone), and R16 would add a field rather than redefine this one. 28 executable tests. |
| **R22** | **CLOSED (by Batch 17), re-verified here** | `docs/BRANCH_B_SEMANTICS.md`; 18 tests. This batch independently re-derives its interface constraint, its "no wire surface" claim and its unmoved identity string (§2, `tests/…§6`), and repairs the one red test its suite shipped with (§5). Not reopened. |
| **B18-1** (new) | **OPEN follow-up — recorded, not a finding number** | `test_branch_b_semantics.py` shipped a red assertion at the Batch 17 baseline (over-broad `"whole image"` match). Repaired this batch; recorded so the history shows the suite was briefly not-green. |

### Distinguishing the defect classes, for the record

* **Reproducibility defect:** neither R20 nor R22 is one. `face_px` and the branch
  descriptions are in zero identity strings and zero cache keys.
* **Correctness defect:** neither, *as they stand today*. R20 *was* a latent
  correctness defect pre-Batch-14 (report ≠ gated quantity); Batch 14 closed that
  half. R22's residual — a published *example* (`image_no_face`) depicting an
  unreachable verdict path — is Batch 17's territory and is annotated there.
* **Semantic defect:** R20 was semantic pre-Batch-14 and is not now. R22's
  `disagreement` / `interval` carry a meaning that is ambiguous under the two
  branch-B readings — Batch 17 resolved the ambiguity in prose.
* **Documentation defect:** what both R20 and R22 are today. Both closed by
  making the prose match the code, with executable pins so the code cannot drift
  back under the prose.

---

## 7. Guarantees gained / NOT gained

**Gained**

1. `QualityReport.face_px`'s description is true of every value the engine
   publishes — image-with-face, faceless-image, video — and names the one metric
   behind "largest".
2. The pre-Batch-14 reading of the value (`MULTI_FACE_SEMANTICS.md` §10) is
   refuted by execution, not just superseded by a newer paragraph.
3. `face_px` is pinned as reporting-only: `_gate` reads `size_px`, and nothing in
   `inference/` or `contracts/` uses `face_px` as an operand. A future edit that
   makes a decision depend on the published figure trips a test.
4. The "documentation change ⇒ zero artifact change" claim is executable: render
   with and without, diff, one field's description.
5. R22's Batch 17 closure has an independent second measurement of its three
   load-bearing invariants.
6. The suite is green again.

**NOT gained**

1. **R20's `0` vs `None` split is documented, not harmonised.** A faceless image
   publishes `face_px=0`; a video publishes `face_px=None`; both mean "no face
   measured". Making the image path emit `None` too would change a published
   value — behaviour, out of scope for a documentation batch. Recorded as a
   follow-up.
2. **R16 is untouched.** When `faces[]` holds more than one scored face, `face_px`
   describing "the selected face" is still a one-face view of a multi-face input.
3. **`schema_out/` is still unguarded against staleness.** This batch regenerated
   it; nothing stops it drifting again. Batch 13 recorded the same; not this
   batch's brief.
4. **Branch B still scores the face crop** (Batch 17's guarantee-not-gained #1),
   and `_abstain` still hard-codes its abstention reason as `QUALITY_BELOW_FLOOR`
   on a `NO_FACE_DETECTED` gate (Batch 17 §10 follow-up). Neither is R20's.
5. **No new fingerprint.** Correcting prose protects no future policy or geometry
   change; R25 still wants to land with the geometry change it guards.
