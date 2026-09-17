# Branch B semantics — architecture review for R22 (Batch 17)

> **Identity note (Batch 22).** This doc pins `effective_version` at
> `stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387` as the "nothing moved"
> constant. Batch 22 (R25, unrelated to branch B) moved it to
> `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9`; `test_no_identity_string_moves`
> now pins the new value. **Batch 23 (R29) moved the `.f.` segment again** to the
> composite `e629642c` (model identity `…f.e629642c`);
> `tests/test_branch_b_semantics.py` `_EXPECTED_EFFECTIVE` pins that; see
> `docs/VIDEO_SELECTION_IDENTITY.md`. R22's conclusion — no wire field, no behaviour change —
> is unaffected.

> **Outcome: R22 is real, and it is a contract clarification.** The claim that
> branch B scores "the whole image" is documentation-as-data with no wire
> surface, and the code has already made the decision R22 asks about: branch B
> scores the same prepared tensor as branch A — the aligned face crop in the
> image path, the whole resized frame in the video path. This batch corrects the
> claim to say so, annotates the one fixture that depicts the unbuilt
> whole-image target, and pins the current behaviour. **No field, schema,
> fingerprint or `CONTRACT_VERSION` moves.**

Re-runnable: `python tests/branch_b_semantics_study.py`. Proof and doubles:
`tests/test_branch_b_semantics.py` (18 tests). The finding was re-derived from
code and tests; it was not assumed.

**State reviewed.** 45 suites (44 + this batch), `CONTRACT_VERSION` 1.0.0, R21
closed (B16), R16 open, R18's capability half open.

---

## 1. Trace

### 1.1 The claim, and every place it lives

| # | Site | Form | Reaches a client? |
|---|---|---|---|
| 1 | `contracts/enums.py` — `BranchId.B` comment | `# semantic — DINOv2-B/14 frozen + probe, whole image` | no — a source comment |
| 2 | `contracts/branch.py` — `BRANCH_DESCRIPTIONS[BranchId.B]` | `"DINOv2-B/14 ... over the whole image."` | **no — see 1.3** |
| 3 | `README.md` — frozen-architecture table | `Branch B — semantic | DINOv2-B/14 frozen + probe, whole image` | no — project docs |

The contrasting claim for branch A — *"aligned face crop"* — lives in the same
three places (`enums.py` has no A-scope comment; `BRANCH_DESCRIPTIONS[A]` and the
README row do).

### 1.2 Producers of a branch-B score

| Producer | File | What it consumes |
|---|---|---|
| `StubBackend.score(batch)` | `inference/models.py:348` | the **single** `(N,H,W,3)` tensor it is handed. `p_b = clip(p_a·0.85 + 0.05)` — B is a fixed function of A's score **of the same tensor** |
| test doubles implementing `ModelBackend` | `test_inference.py` (`Drifting`, `Exploding`), `test_face_size_policy.py` / `face_size_policy_study.py` (`RegionOracleBackend` — "logit from the crop's own mean luminance", **for both A and B**), scripted backends in `test_inference_determinism.py`, `test_run_identity.py` | one tensor, per-branch out |
| hand-built `BranchOutcome` / `BranchResult` | `contracts/fixtures.py`, `tests/stub_handlers.py`, ~15 test files | arbitrary logits, no input model |

The tensor handed to `score()` is:

* **image path** — `_crop_and_score` (`inference/image_pipeline.py:187`):
  `prepare_image(crop_face(image, faces[0], policy), size)` → **the aligned face crop**;
* **video path** — `_score_batch` (`inference/video_pipeline.py:276`):
  `resize(whole_frame, input_size)` → **the whole frame**, squashed onto a square (R24).

### 1.3 The interface — the architectural constraint

`inference/models.py::ModelBackend.score(self, batch: np.ndarray) -> dict[BranchId, np.ndarray]`
— *"`(N, H, W, 3) float32` in, one logit array of shape `(N,)` per branch."*

**One tensor in.** Measured (`study` §4): `score` takes exactly one array
parameter; `image_pipeline.py` and `video_pipeline.py` each call `backend.score(...)`
with exactly one positional argument. **There is no parameter through which branch
B could receive the whole image while branch A receives the crop.** The
"whole image vs face crop" distinction is not merely unimplemented — it is
unrepresentable without changing this interface, and no real backend exists.

### 1.4 Consumers of a branch-B score

| Consumer | File | Uses B for |
|---|---|---|
| `manipulation_evidence` → `fuse` → `calibrated_probability` → `decide` | `inference/postprocess.py:164-285` | the fused `p_fake` and `verdict`. B ∈ `MANIPULATION_BRANCHES`, weighted-pooled with A |
| `disagreement_of` | `postprocess.py:234` | `max(p) − min(p)` over scored {A, B} → `disagreement` |
| `confidence_interval` | `postprocess.py:256` | `interval`, widened by `disagreement` |
| `_branch_results` | `postprocess.py:293` | the published `BranchResult` row: `name="Semantic"`, `logit`, `p=σ(logit)`, `weight`, `version` |
| `_fuse_batch`, `pool_at` | `video_pipeline.py:285`, `postprocess.py` | per-frame fuse, then pool B over the same top-k frames as A |

### 1.5 Contract fields carrying or derived from branch B

| Field | Tier | Relation to B |
|---|---|---|
| `branches[]` row `id=B`: `id, status, logit, p` | **Stable** | carries B directly (4 subfields) |
| `branches[]` row `id=B`: `name, weight, abstain_reason, version` | Evolving / unlisted | carries B directly (4 subfields) |
| `disagreement` | Evolving | `f(A, B)` |
| `interval` | Stable | `f(p_fake, disagreement)` → `f(…, B)` |
| `p_fake`, `verdict` | Stable | fused from {A, B} + P |
| `model.branch_versions["B"]` | Evolving | B's version string |
| `/health` `branches[]` `id=B`: `id, loaded, version` | `HealthResponse` | 3 subfields |

### 1.6 Validators touching branch B

`BranchResult._status_consistency`, `BranchResult._provenance_is_one_sided` (P
only), `EvidenceBundle._branches_unique`, `EvidenceBundle._verdict_follows_threshold`,
`ModelSpec.__post_init__` (*"at least one of A or B must carry positive weight"* —
B **can** be the sole manipulation branch). **None asserts anything about what
input branch B scored.** Count: **0**.

### 1.7 Fixtures depicting branch-B semantics

| Fixture | Branch B | Notable |
|---|---|---|
| `image_manipulated`, `image_authentic`, `image_authentic_signed` | scored, weight 0.24, beside A | B same input as A — consistent |
| `video_manipulated`, `video_authentic` | scored, weight 0.24 | whole-frame — consistent with the claim |
| `image_insufficient_evidence`, `video_insufficient_evidence` | abstained | — |
| **`image_no_face`** | **scored, weight 0.94, verdict from B alone, `gate_passed=True`, `faces=[]`** | **depicts a face-independent whole-image B — the contradiction** |

`image_no_face` is a **published OpenAPI example** (`components/schemas/EvidenceBundle/examples`,
verified by `test_openapi::test_every_documented_example_validates_against_contracts`).
The 0.94 reweighting appears **only** here, via `fixtures._branch(B, weight=0.94)`.

`tests/stub_handlers.py::_branches(face_present=False)` reweights B to 0.94 and
keeps it scored on a faceless input — **dead code**: every call site passes
`face_present=True`.

### 1.8 Tests depending on a branch-B interpretation

| Test | Interpretation it encodes | Changed? |
|---|---|---|
| `test_contracts::test_no_face_image_abstains_on_branch_a_only` | **B = whole-image / face-independent** (via `image_no_face`) | no — now reads as "the target shape" |
| `test_inference::test_no_face_path_abstains` | **B = face-gated** (real pipeline) | no |
| `test_inference::test_low_quality_path_abstains_before_the_model_runs` | B gated | no |
| `test_face_size_policy` / `face_size_policy_study` (`RegionOracleBackend`, both A & B from the crop) | **B = crop-scoped** | no — consistent |
| `test_video_branches::test_branch_b_matches_the_backend_scores_it_pools`, `::test_every_branch_pools_over_the_same_frames` | B = whole-frame (video), same frames as A | no — consistent |
| `test_frame_extraction::test_scores_are_identical_to_the_legacy_extraction` | B = whole-frame (video) | no |
| `test_postprocess::test_fusion_renormalises_over_scored_branches` | incidental (`fuse` arithmetic) | no |

**Zero existing tests change.** The two rows in bold are the contradiction, and
both pass today.

### 1.9 Dependency graph

```
   frozen-architecture claim:  "Branch B = whole image"  (enums comment · BRANCH_DESCRIPTIONS · README)
        │  0 wire fields · 0 validators · 0 fingerprints · not in the OpenAPI structural signature
        ▼
   ModelBackend.score(ONE tensor) → {A, B, P logits}          ← the constraint:
        │                                                        B cannot be given a
        │ image path                    │ video path             different tensor than A
        ▼                               ▼
   prepare_image(crop_face(faces[0]))   resize(whole frame)  ──→ R24 squash
        │                               │
   B scores the FACE CROP          B scores the WHOLE FRAME
   (claim violated)                (claim satisfied — but A's
        │                           "face crop" claim violated ← R22-mirror, tied to R18)
        ▼                               ▼
   _gate: no face ⇒ NO_FACE_DETECTED ⇒ _abstain(ALL branches, gate_passed=False)
        │                                        └─ B's reason hard-coded QUALITY_BELOW_FLOOR (false; §10 note)
        ▼
   manipulation_evidence({A,B}) · fuse · disagreement_of(A,B) · _branch_results
        ▼
   branches[B].{id,status,logit,p}=Stable · disagreement=Evolving · interval=Stable
   p_fake · verdict=Stable · model.branch_versions[B]=Evolving · /health

   CONTRADICTS:
     fixtures.image_no_face  (B scored @0.94 · gate_passed · verdict=manipulated · faces=[])
       → published OpenAPI example
       → test_contracts::test_no_face_image_abstains_on_branch_a_only   ── PASSES
     vs
     test_inference::test_no_face_path_abstains  (all abstain · gate_passed=False)  ── PASSES

   R16 (per-face scoring): the fusion shape depends on the answer
     B crop-scoped   → fuse() runs N times, unchanged        ← what the code implements
     B whole-image   → 1 image-level B pooled with N face-level A → fuse() restructured
```

---

## 2. Executed proofs

All from `tests/test_branch_b_semantics.py` unless noted. Doubles:
`TensorCaptureBackend` (records every tensor `score()` receives),
`ScriptedDetector` (places a face in a corner so crop ≠ whole image),
`LegacyBranchBClaim` (the descriptions as of Batch 16, verbatim).

### P1 — the interface cannot express the claim
`test_the_model_backend_interface_cannot_split_the_input`. `ModelBackend.score`
has one array parameter; `image_pipeline.py` and `video_pipeline.py` call
`backend.score(...)` with one argument each. **A single tensor produces every
branch.**

### P2 — image path: branch B scores the face crop, byte for byte
`test_branch_b_scores_the_face_crop_in_the_image_path`. A 900×700 image with a
face box at (60, 60, 180×180); the crop and the whole image are provably
different pixels:

```
tensor branch B was scored from : (1, 384, 384, 3)  digest 7a123e16d9b4
prepare_image(crop_face(...))   : (1, 384, 384, 3)  digest 7a123e16d9b4   ← match
prepare_image(whole image)      : (1, 384, 384, 3)  digest b25f545ffd15   ← not

branch B saw the CROP        : True
branch B saw the WHOLE IMAGE : False
```

`test_branch_a_and_b_come_from_one_backend_call`: one `score()` call produced
both A and B.

### P3 — video path: branch A scores the whole frame (R22's mirror)
`test_branch_a_scores_the_whole_frame_in_the_video_path`:

```
tensor branch A/B scored from (frame 0) : digest 4bdaaa7196be
resize(whole frame 0, 384)             : digest 4bdaaa7196be   ← match
```

In the image path branch B's "whole image" claim is violated; in the video path
branch A's "aligned face crop" claim is violated. One single-tensor interface,
two mismatches, opposite branches. (VIDEO_FACE_SEMANTICS §9.2 already recorded
the video-A half; this batch is the first to measure it.)

### P4 — the shipped contradiction, both states pass today
`test_the_fixture_and_the_pipeline_disagree_about_a_faceless_image` — the
**fail-before evidence, executed**:

| | `fixtures.image_no_face` (published example) | real `ImagePipeline`, faceless PNG |
|---|---|---|
| `branch(B).status` | **SCORED** (logit 1.669, weight 0.94) | **ABSTAINED** (`quality_below_floor`) |
| `quality.gate_passed` | **True** | **False** (`no_face_detected`) |
| `verdict` | **MANIPULATED** | **INSUFFICIENT_EVIDENCE** |

Both `test_contracts::test_no_face_image_abstains_on_branch_a_only` and
`test_inference::test_no_face_path_abstains` pass. They encode opposite semantics
for branch B on a faceless image.

`test_the_pipeline_never_scores_branch_b_without_a_face`: on a faceless PNG and a
40×40 PNG, every branch abstains — the load-bearing current behaviour a future
whole-image B would change.

### P5 — nothing on the wire carries the claim
`test_the_whole_image_claim_is_not_on_the_wire`: `BranchResult` has no
`description` field; the generated schema files and the OpenAPI snapshot contain
none of `DINOv2` / `over the whole image` / `aligned face crop`.
`test_branch_descriptions_have_no_functional_consumer` (AST): nothing reads
`BRANCH_DESCRIPTIONS`. (`study` §1: "whole image" *does* appear once in the live
OpenAPI doc — in `FaceTrack`'s R16 prose, *"the whole image's verdict"* —
unrelated.)

### P6 — the clarification, fail-before / pass-after
Five tests failed before the wording change and pass after:
`test_the_branch_b_description_states_what_it_scores`,
`test_the_branch_a_description_notes_the_video_mismatch`,
`test_the_enums_comment_no_longer_states_whole_image_as_a_bare_fact`,
`test_the_readme_branch_table_is_corrected`,
`test_the_image_no_face_fixture_is_annotated_as_a_target`.

### P7 — nothing moved
`test_no_identity_string_moves`:
`effective_version == stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387`,
unchanged. `test_every_shipped_fixture_still_validates`,
`test_image_no_face_is_still_a_published_example_and_validates`.

---

## 3. Quantification

| Quantity | Count | Notes |
|---|---|---|
| Claim sites in source | **3** | `enums.py` comment, `BRANCH_DESCRIPTIONS[B]`, `README.md` |
| Claim sites on the wire | **0** | `BranchResult` has no description field; `BRANCH_DESCRIPTIONS` unused |
| Wire fields carrying branch B directly | **12** | `branches[B]`: 8 subfields · `model.branch_versions["B"]`: 1 · `/health branches[B]`: 3 |
| Wire fields derived from branch B | **4** | `disagreement`, `interval`, `p_fake`, `verdict` |
| Wire fields that carry a *wrong value* under any interpretation | **0** | the number `score()` produces is the number, regardless of what it means |
| Wire fields whose *meaning* is ambiguous under the two readings | **4** | `branches[B].{logit,p}` (semantic score of *what*), `disagreement`, `interval` |
| Validators asserting branch B's input scope | **0** | none exist |
| Fixtures with branch B scored | **6** | of 8 EvidenceBundle fixtures |
| Fixtures depicting the contradiction | **1** | `image_no_face` — also a published example |
| Fixture helpers touched | **1** | `fixtures._branch` (the 0.94 weight) |
| Existing tests implicitly depending on an interpretation | **~8** | §1.8 |
| Existing tests that need changing | **0** | — |
| New pinning tests | **18** | `tests/test_branch_b_semantics.py` |
| Source files whose prose changed | **4** | `contracts/enums.py`, `contracts/branch.py`, `README.md`, `contracts/fixtures.py` (+ a `stub_handlers.py` comment) |
| Behaviour changes | **0** | — |
| Migration steps | **0** | — |

---

## 4. Contract analysis

**`CONTRACT_VERSION` 1.0.0, unchanged.** The clarification changes prose in
three source locations and one fixture docstring. None is contract surface:

* `BranchResult`'s field set is unchanged (`id, name, status, logit, p, weight,
  abstain_reason, version`); Stable-tier for branch rows is `{id, status, logit,
  p}` and none changes shape or the meaning of its *value*.
* `BRANCH_DESCRIPTIONS` is not serialized and has no consumer (P5) — it is a
  documentation constant exported for a future UI.
* The `enums.py` and `README.md` text is not machine-read.
* `image_no_face`'s Python docstring is not part of its serialized payload; the
  example bytes are byte-identical.

**OpenAPI:** the structural signature strips `description`/`example`/`examples`
(`backend/openapi.py:49`), so even if a description had changed the drift guard
would not fire. `test_openapi` passes with no snapshot regeneration.

**Would any *other* interpretation need a bump?**

| Interpretation | Bump? | Why |
|---|---|---|
| **Clarify to "crop-scoped" (this batch)** | **no** | prose only; matches the code |
| Build a genuinely whole-image branch B | **no** | `branches[B].logit`/`p` values move like any model change; `calibration_fingerprint` moves (a rescore); no field changes *shape*. `p`/`logit` still mean "the semantic branch's score" |
| Redefine `disagreement` to mean "crop-vs-whole-image spread" | **no** | `disagreement` is Evolving-tier |
| Make branch B carry a verdict on a faceless image *in the pipeline* | **no** by shape; **yes** in spirit | it would make the pipeline produce the `image_no_face` shape — a new capability, calibrated against nothing today |

No interpretation of R22 requires a MAJOR bump. The clarification requires no
bump at all.

---

## 5. Identity / fingerprint analysis

`BRANCH_DESCRIPTIONS`, the `enums.py` comment and the README feed **none** of:

* `weights_identity` — `version` + per-branch versions + branch weights;
* `calibration_fingerprint` — temperature, `fitted_on`, `n_samples`, threshold, band;
* `gate_fingerprint` — the quality-gate caps;
* `face_selection_fingerprint` — the eight face-selection knobs;
* `effective_version` / the runtime version / the decode identity;
* `BenchmarkRun.effective_model_version` / `decode_identity`;
* `CalibrationWitness`;
* the OpenAPI structural signature.

`test_no_identity_string_moves` pins `effective_version` at
`stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387`. **No cache generation, no
idempotency-key change, no benchmark or calibration re-digest.**

Benchmark and calibration identity are unaffected: a benchmark run records
`effective_model_version`, which does not move; a calibration witness records the
fitted quantities, which do not move.

---

## 6. Migration analysis

**None.** No data on disk changes meaning or shape:

* stored `EvidenceBundle`s — unaffected (no field changed);
* the 8 published examples — byte-identical payloads;
* `schema_out/*.json` — regenerable, unchanged (and, as Batch 13 noted, not
  guarded against staleness anyway; this batch does not touch them);
* `backend/openapi.snapshot.json` — unchanged;
* benchmark JSONL history, calibration witnesses, split locks — untouched.

`parse_model_version` and every identity parser stay total on all eras because
no era boundary is created.

---

## 7. Guarantees gained

1. **The branch-B and branch-A input-scope claims match the code.** A reader of
   `enums.py`, `branch.py` or the README learns that every branch scores one
   tensor — the face crop (image) or the whole frame (video) — and that
   whole-image B is an unbuilt target, without running the pipeline.
2. **The `image_no_face` contradiction is labelled.** The fixture and its test
   now say they depict the whole-image-B target, not shipped behaviour — the
   same treatment Batch 13 gave the video face-stage fixtures.
3. **The current pipeline behaviour is pinned.** `test_the_pipeline_never_scores_
   branch_b_without_a_face` and `test_the_fixture_and_the_pipeline_disagree_about_
   a_faceless_image` fail loudly if a future change lets branch B score a
   faceless image without the accompanying calibration and contract work.
4. **The interface constraint is recorded and tested.** `ModelBackend.score`'s
   single-input shape is pinned; a dual-input backend cannot land without a test
   noticing.
5. **`BRANCH_DESCRIPTIONS` is pinned dead.** If it gains a functional consumer,
   `test_branch_descriptions_have_no_functional_consumer` fails and the "no
   contract change" basis is re-opened.
6. **R22's video mirror is measured.** Branch A scoring whole frames in the
   video path is now a test, not a paragraph.

---

## 8. Guarantees NOT gained

1. **Branch B still scores the face crop.** This batch makes the claim honest;
   it does not build the whole-image branch. Scene-level context a crop cannot
   show is still invisible to the platform.
2. **The `image_no_face` verdict path is still unreachable from the pipeline.** A
   faceless image still abstains; the fixture still depicts something the engine
   cannot produce. Annotated, not removed — a contract must not make its own
   roadmap unrepresentable.
3. **`disagreement` and `interval` still mean "A vs B about the same crop".**
   The more informative quantity — spatial-on-crop vs semantic-on-whole-image
   disagreement — does not exist.
4. **`_abstain` still hard-codes branch B's abstention reason as
   `QUALITY_BELOW_FLOOR`** on a `NO_FACE_DETECTED` gate. The image quality was
   fine; there was no crop. §10 records this as a follow-up — it is a published
   `abstain_reason` value and changing it is behaviour, out of scope for a
   clarification batch.
5. **Branch A's video mismatch is not fixed.** It is tied to R18 (no video face
   stage) and R24 (frame squash), and the fix is the video face stage.
6. **No structural validator forbids the contradiction.** A validator that
   refused `gate_passed=True` with branch A `NO_FACE_DETECTED` would reject the
   fixture and the target shape. The contradiction is held apart by tests and
   prose, not by an invariant — the same posture `ORDERING_RULE` has had since
   Batch 12.
7. **R16, R13, R20, R23–R26 untouched.**

---

## 9. R16 interaction

`fuse` and `manipulation_evidence` take **one** `dict[BranchId, BranchOutcome]` —
one A, one B, one P. Two readings, two R16 shapes (`study` §7,
`test_fuse_takes_a_single_branch_set`):

* **B is crop-scoped** (what the code does) → R16 Option 2 = call `fuse()` once
  per scored face, with that face's A, B and the shared P. **No signature
  change.** Demonstrated: `fuse()` runs unchanged for `a_logit ∈ {3.0, −1.0}`.
* **B is whole-image** → one image-level B must be pooled with each face-level A.
  `manipulation_evidence`'s renormalised weighted mean takes a single {A, B}
  pair, so this needs `fuse(face_a, shared_b, shared_p)` — a new signature and a
  restructured mean.

Therefore:

| Question | Answer |
|---|---|
| **Does R22 block R16?** | **No.** R16 is implementable under either reading. |
| **Is R22 blocked by R16?** | **No.** |
| **Can R22 be closed independently?** | **Yes** — this batch, with zero R16 work. |
| **Should R22 remain an open design decision?** | **The ambiguity: no** — the code decided (crop-scoped), the docs were wrong, this batch fixes the docs. **The residual capability question** — "build a genuinely whole-image branch B" — **is a real decision**, now *explicit* (named in the corrected descriptions as the frozen-architecture target) rather than latent, and it gates nothing. It belongs with real weights and a dual-input backend. |

The code having already chosen crop-scoped is the *fortunate* outcome for R16: it
is the simpler shape, and R16 Option 2's plumbing (per-face `fuse`) needs no
fusion restructuring.

---

## 10. Final recommendation

**Close R22.** It is a **contract clarification**, not a deferred design
decision and not a defect that produces a wrong number.

Classification:

| Dimension | Verdict |
|---|---|
| Documentation-only | **primarily yes** — 3 source claim sites corrected |
| Contract clarification | **yes** — the claim now matches the code; no bump |
| Correctness-critical | **no** — the engine produces no wrong value; a published *example* depicts an unreachable path (R18's shape), annotated not rewritten |
| Reproducibility-critical | **no** — 0 fingerprints, 0 cache impact |
| Requires SCRFD | **no** for the clarification; a real whole-image B needs a video face stage but not SCRFD specifically |
| Requires trained weights | **no** for the clarification; **yes** for a real whole-image B to be meaningful |
| Requires a MAJOR bump | **no** under any interpretation |

**Shipped this batch:** the wording corrections (§4), the fixture annotation, the
`stub_handlers` comment, and `tests/test_branch_b_semantics.py` (18 tests)
pinning the measurement, the contradiction, the interface constraint and the
corrected claims.

**Recorded, not shipped:**

1. **`_abstain` gives branch B a false reason.** `ImagePipeline._abstain` and
   `VideoPipeline._abstain` hard-code `BranchId.B: QUALITY_BELOW_FLOOR`
   regardless of the actual gate reason. On a `NO_FACE_DETECTED` gate this is
   false — the image quality passed. A one-line follow-up (propagate the gate
   reason, or `BRANCH_NOT_APPLICABLE`) that changes a published `abstain_reason`
   value; deliberately out of scope for a no-behaviour-change batch.
2. **The residual capability decision.** Whether to build a dual-input backend so
   branch B scores the whole image (and branch A scores an aligned crop on
   video) is a frozen-architecture item for the model stage. It is now explicit
   in the branch descriptions. It does not block R16 or anything else.
3. **Branch A's video mismatch** — the mirror of this finding — is tied to R18
   and R24 and closes with the video face stage.
