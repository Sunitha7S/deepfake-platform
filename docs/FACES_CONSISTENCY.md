# faces[] consistency — architecture report and Batch 16 (R21)

> **Prelude — the architecture review the brief asked for.** Sections 1–6 are the
> report: the dependency graph of every open finding, which one to take next and
> why, what is blocked on SCRFD, what is blocked on trained weights, and what can
> still be closed with contract and reproducibility work alone. Sections 7–14 are
> Batch 16, which implements the report's recommendation: **R21**.

**State at report.** 43 suites, 1116 tests, zero failures. `CONTRACT_VERSION`
1.0.0. OpenAPI aligned, version alignment passes. Closed: R1–R5, R7, R8,
R11, R12, R14, R15, R17, R18 (false-claim half), R19. Open: R6, R9, R10, R13,
R16, R20, R21, R22, R23, R24, R25, R26.

---

## 1. Dependency graph of the open findings

Edges are *"must land before"* or *"is only reachable once"*. `W` = needs trained
weights, `S` = needs the SCRFD migration, `V` = needs a contract MAJOR bump, `$`
= moves a fingerprint / dumps the result cache.

```
                        ┌─────────────────────────── trained weights ───────────────────────────┐
                        │                                                                       │
   R10  (GPU/training) ─┘   R13  (crop sensitivity, 310×) ──┐                                    │
        [W, no training loop]        [W: real faces] [S]    │                                    │
                                                            ▼                                    ▼
   R6  (decode residual) ── cross-OS/build agreement ──> [needs a 2nd machine, not code]   R16 Option 3
        frame_count not validated ──────────────────────> closable now (narrow)            (aggregate verdict)
                                                                                           [W, V, S, $]
   R9  (quarantine) ── by design, reported every run ────> design note only, no defect

                    ┌──────────────── SCRFD returns free-aspect NMS boxes ────────────────┐
                    │                                                                     │
   R17 (CLOSED B12) │   R19 (CLOSED B14, armed check)   R23 occupancy-collapse half  [S]  │
        │           │   R23 aspect-distortion half ── live today, no output-identical fix ┤ [$]
        ▼           │   R24 video whole-frame squash ── live today, fix = video face stage │ [S]
   ┌─── R21 ────┐   │   R25 geometry knobs outside identity ── land WITH the geometry fix  ┘ [$]
   │ uniqueness │   │   R26 published bbox ≠ scored region ── carry the crop window
   │ headline   │   │
   └─────┬──────┘   │
         │          │
         ▼          ▼
   R16 image path (Option 2: per-face evidence)  ── implementable pre-SCRFD, but the
         │                                            per-face data would be a report on
         ▼                                            Haar's eyesight (R13). Plumbing now,
   R22  (branch B whole-image vs crop) ── decide on paper; shapes R16's fusion.
   R20  (face_px prose) ── documentation only; defect condition removed by B14.
```

**Reading the graph.** Everything on the right of the SCRFD line is gated by a
detector that returns square boxes today. Everything above the trained-weights
line cannot be *measured* without a model, and R10 additionally has no training
loop to instrument. The left column — **R21 → R16(image) → R22/R20** — is the
only chain that runs entirely on contract and reproducibility work, and R21 is
its root because it is a *prerequisite*: any R16 change populates `faces[]`, and
`faces[]` currently has no invariant to populate it against.

---

## 2. Which finding next, and why: R21

Scored against the brief's five criteria:

| Finding | ↓ repro risk | provable w/ tests | no weights | no SCRFD | no contract-version change |
|---|---|---|---|---|---|
| **R21** | **yes** | **yes** | **yes** | **yes** | **yes** |
| R20 | no (doc only) | trivially | yes | yes | yes |
| R22 | no (a decision) | no (paper) | yes | yes | yes |
| R6 (frame_count) | yes | partly — test is itself decoder-dependent | yes | yes | yes |
| R9 | already reported every run | n/a | yes | yes | yes |
| R26 | low (contract clarity) | partly | yes | yes | maybe (Evolving field) |
| R13 | yes | no — deferred w/ acceptance criterion | **no** | **no** | yes |
| R16 | correctness, not repro | partly | Option 3 **no** | effectively **no** | Option 3 **no (MAJOR)** |
| R23 / R24 / R25 | yes | **no output-identical option exists** | can't quantify | R23occ/R24 **no** | R25 dumps cache `$` |
| R10 | yes | **no** | **no** | yes | yes |

R21 is the only open finding that satisfies all five. It also:

* **removes a representable-inconsistency class** on the same footing as
  `_verdict_follows_threshold`, `_branches_unique`, `_top_frames_name_a_published_track`
  and the `ThresholdInfo` band derivation — a bundle whose headline contradicts
  its own evidence — which is squarely reproducibility/provenance work;
* **unblocks the largest open contract finding, R16.** The multi-face review
  (`docs/MULTI_FACE_SEMANTICS.md` §12, Option 0) records R21 as *"a prerequisite,
  not a choice"* and *"free now, an audit of what already validated later"*;
* is **provably output-identical** — it adds two `model_validator`s that reject
  only bundles nothing in the platform produces, proved against every shipped
  fixture, the stub handlers and the real pipeline.

R20 is bundled-adjacent (same `faces[]` / quality area, same *"now"* bucket in
the review) but it is a one-line prose correction whose defect condition B14
already removed; it is left as its own change. R6's `frame_count` residual is
real but the only way to exercise it is to feed a decoder a container that lies
about its frame count, and how a decoder reacts to that is the very
build-dependence R6 is about — the test would be as non-portable as the bug.

---

## 3. Which findings depend on SCRFD

| Finding | Why SCRFD |
|---|---|
| **R13** | It is a property of the Haar cascade's instability; the fix is *"re-measure against SCRFD on real faces, crop stability under ±4 LSB an acceptance criterion"* (B8, B10). |
| **R23** (occupancy-collapse half) | Latent until boxes have free aspect ratio; square Haar boxes cannot trigger it. |
| **R24** | Live today, but the fix is a video face stage — per-frame detect + track + crop — which is SCRFD work by definition (B15). |
| **R25** | Not SCRFD-*gated*, but deliberately sequenced to land in the same batch as the geometry change it protects, which is SCRFD (B15 §10). |
| **R16** (image path, meaningfully) | Option 2 is implementable against Haar, but Haar finds neither face on a two-face canvas (`multi_face_study` stage 0) and R13 measures it at 310× amplification on 25.7% of decisions, so the per-face data would describe the placeholder's eyesight. Option 3 additionally needs real per-face score distributions. |
| R19, R20 | *Armed* by SCRFD (both closed / prose-corrected pre-emptively for exactly this reason). |

R21 depends on none of them: the uniqueness rule is detector-agnostic, and the
headline rule holds under both today's single-face pipeline and an R16 per-face
design.

---

## 4. Which findings are blocked by lack of real model weights

| Finding | What weights would unblock |
|---|---|
| **R10** | Everything — cuDNN autotuning, TF32, non-deterministic scatter/atomics, AMP loss scaling. There is no training loop to instrument, so this is doubly blocked. |
| **R13** | The *magnitude* on real faces. The 310× is a smooth-surrogate measurement; the direction is not in doubt, the size on photographs is. |
| **R16** (Option 3) | The distribution of per-face scores on real group media and of N in real traffic — without which an aggregation rule is calibrated against nothing, and the max-over-N FPR multiplication (§12 of the review: ~5× at N=5) cannot be traded off. |
| **R23 / R24** | *How much* the geometry distortion moves a verdict. `StubBackend` establishes *that* the model sees different pixels; the 13.0% stretch and 1.778× squash are facts about the tensor, not about a score. |
| R6 | Nothing directly — R6 is about the decoder, not the model — but its serving-path impact estimate uses a surrogate scorer for the same reason. |

R21 needs no weights: it is a statement about the shape of the result object,
not about any number a model produces.

---

## 5. Which findings can still be closed purely through architecture and reproducibility work

* **R21 — fully.** This batch.
* **R20 — fully**, as a prose correction (no drift: the OpenAPI structural
  signature ignores field descriptions, `test_signature_ignores_prose_but_not_structure`).
  Its own change.
* **R22 — as a decision**, recorded on paper: is branch B whole-image or
  face-crop? It changes the shape of any per-face fusion. No executable guarantee
  attaches to it until R16 is built.
* **R9 — as a design note.** It is already reported by `verify_lock` on every
  run and is Low only because it cannot happen silently. Nothing to prove; the
  finding is a standing acknowledgement, not an open defect.
* **R6 (`frame_count` sub-residual) — narrowly.** A cross-check of the declared
  frame count against the decodable frame count would catch a lying container
  before `sample_frame_indices` drifts 60% of the sampled indices. Deferred here
  because the test is decoder-dependent in the same way the bug is.
* **R26 — partially.** The `face_track` coordinate clamp divergence can be
  characterised and pinned now; carrying the actual crop window in the bundle is
  a (Evolving-tier, no-bump) field addition that is cheapest with R23.

Cannot be closed without weights or SCRFD: **R10, R13, R23 (both halves' *fix*),
R24, R16 (the scoring policy), R25 (the fingerprint is worthless without the
geometry change it guards)**.

---

## 6. What Batch 16 guarantees, and what it does not — summary

**Gained.** A published `EvidenceBundle` cannot carry two faces with the same
`track_id`, and a decided `p_fake` cannot be a number that no published face
scored. Both hold by hand, from the factory, and on the way back from storage
and the wire. R16's implementation now has an invariant to build against.

**Not gained.** One face's verdict is still published as a group photograph's
(R16). The non-selected faces' evidence is still never computed. The video path
still publishes no faces (R18). The headline rule is vacuous where the platform
legitimately publishes a per-face score with no headline (in-band abstention) or
a headline with no face (video). No fingerprint moves and no cache is dumped —
which is the point, but it means nothing here protects a *future* geometry or
policy change (R25).

---

## 7. R21, traced

`EvidenceBundle` carried, before this batch:

```
_verdict_and_probability_agree      verdict  <-> p_fake / abstain_reason
_interval_brackets_estimate         interval <-> p_fake
_branches_unique                    branches[].id unique
_video_block_matches_media_kind     media.kind <-> video
_top_frames_name_a_published_track  video.top_frames[].track_id -> faces[]   (B13)
_gate_failure_implies_abstention    quality.gate_passed -> verdict
_verdict_follows_threshold          verdict  <-> p_fake vs threshold band
```

Seven invariants; **`faces[]` appears in exactly one of them, and only as the
target of a referential-integrity check from `video.top_frames`.** Nothing
constrains `faces[]` on its own terms. Measured directly against the model,
re-validated through JSON the way storage and the wire do
(`multi_face_study.py` §7 / `docs/MULTI_FACE_SEMANTICS.md` §7):

```
two faces, both track_id=0, contradictory p_fake -> validates: 2 [0, 0] [0.9, 0.1]
bundle.p_fake: 0.9134827362  verdict: manipulated  agrees with either face?: False
```

Two distinct defects:

1. **Uniqueness.** `branches[]` has had `_branches_unique` since Batch 0;
   `faces[]` has the same shape (an identified list a client indexes and
   `top_frames` resolves against) and no such rule. Two entries at one
   `track_id` cannot be told apart, and `_top_frames_name_a_published_track`
   silently deduplicates the set it checks against.
2. **Headline consistency.** A decided `p_fake` matching no published face is
   the exact class of drift the band-derivation validator and
   `_verdict_follows_threshold` exist to make unrepresentable — left open on the
   one field a per-face design will populate.

Harmless today because the pipeline publishes at most one face — and, per the
review, *"building the feature before the invariant means shipping the
representable-inconsistency the rest of this contract works hard to eliminate."*

---

## 8. The fix

Two `model_validator(mode="after")` methods on `EvidenceBundle`, placed
immediately after `_top_frames_name_a_published_track` because that validator now
defers to them by name.

**`_faces_are_uniquely_identified`** — `len(set(track_ids)) == len(track_ids)`,
the `_branches_unique` rule verbatim on the other list.

**`_headline_is_the_score_of_a_published_face`** — *if* `p_fake` is not `None`
*and* at least one published face carries a `p_fake`, then `p_fake` must equal
one of those face scores exactly. Deliberately narrow:

* **vacuous on abstention** (`p_fake is None`) — an abstention publishes no
  headline probability to contradict, and the image path legitimately publishes
  a per-face `p_fake` on an *in-band* abstention (bundle `p_fake` `None`, face's
  not): branch-score visibility, not drift;
* **vacuous when no published face scored** — the video path publishes
  `faces=[]` with a decided verdict today (R18); an abstained-only face list gets
  the same treatment, and telling that shape apart from the video path's
  legitimate `faces=[]` is R18/R16 work, recorded as a guarantee not gained;
* **"a scored face", not "faces[0]"** — so an R16 design that designates the
  second-listed face as the headline is representable.

Why exact equality is right and not brittle: under today's pipeline
`faces[0].p_fake` is `calibrated_probability(fuse(outcomes, weights))` — the same
call, on the same inputs, as the headline — so they are bit-identical, asserted
by `test_the_single_face_pipeline_shape_validates`. Under R16 Option 2 the
headline is *defined* as one published face's score (largest, or max), which is
literally in the list. Only an *averaging* aggregate would fail the check — and
that is a deliberate meaning change to a Stable field that should trip an
invariant rather than pass unnoticed, exactly the tripwire posture Batch 9 took
with `test_benchmark_runs_still_carry_no_decode_identity`.

---

## 9. Fail-before proof

`tests/test_faces_consistency.py`, using the named double **`LegacyEvidenceContract`**
— the tuple of `EvidenceBundle` model-validators that existed as of Batch 15,
looked up by name on the live class, minus the two this batch adds.
`accepts(bundle)` runs exactly those. `test_the_legacy_contract_double_is_complete`
asserts `live validators == pre-B16 ∪ {the two new}`, so the double cannot fall
behind a rename.

Executed, against a bundle built on `fixtures.image_manipulated()` with only
`faces[]` mutated (`model_construct`, so every nested object stays a real
validated instance and only the bundle-level checks are skipped):

```
duplicate track_id (two id=0, p_fake 0.913 / 0.1):
    LegacyEvidenceContract.accepts  -> True    (satisfied the entire prior contract)
    EvidenceBundle.model_validate   -> ValueError: duplicate track_id in faces[]: [0, 0]

headline matches no face (verdict manipulated @ 0.913, faces 0.80 / 0.10):
    LegacyEvidenceContract.accepts  -> True
    EvidenceBundle.model_validate   -> ValueError: p_fake=0.9134827362 is the score
                                       of no published face ([0.8, 0.1])
```

Six pairs total: the two above, plus `test_the_legacy_contract_accepts_every_shipped_fixture`
(the double is faithful), `test_three_faces_with_a_repeated_id_are_refused`,
`test_the_rule_is_vacuous_when_no_published_face_scored`, and the quantification.

---

## 10. Quantification

`test_quantify_before_and_after`. For a decided bundle publishing one scored
face, sweeping that face's score across `{0, 0.05, …, 1.0} ∪ {headline}`:

```
                             prior contract      shipped contract
face score accepted, of 22        22                    1
                                  ^                     ^
                                  |                     the one value equal to the headline
                                  a gap of the full [0, 1] probability range
                                  against a headline it was meant to describe

duplicate track_id, pile of 8 identical faces:
    prior contract   accepts
    shipped contract refuses
```

The prior contract admitted the published face's score anywhere in `[0, 1]`
regardless of the headline, and admitted an unbounded number of faces sharing
one id. The shipped contract admits exactly the coherent bundle.

---

## 11. Contract, identity and migration impact

* **`CONTRACT_VERSION` 1.0.0, unchanged.** No field added, removed or retyped.
  `faces[]` is Evolving-tier and already unconstrained (`no maxItems`); the two
  validators reject only bundles nothing produces.
* **No OpenAPI snapshot change.** `model_validator`s do not appear in the JSON
  Schema, so `structural_signature` is byte-identical — the same reason Batch 13
  added `_top_frames_name_a_published_track` with no regeneration.
* **No fingerprint moves.** `STUB_ENGINE_VERSION`, `effective_model_version`,
  `gate_fingerprint`, `calibration_fingerprint`, the decode and face-selection
  identities are all untouched. **No cache generation, no idempotency-key
  change.**
* **No schema-version bumps.** `RUN_SCHEMA_VERSION` 1.2.0 and
  `CALIBRATION_WITNESS_VERSION` 1.2.0 unchanged; no stored benchmark row or
  witness re-digests.
* **One existing test fixture reconciled.**
  `test_video_face_semantics.py::test_a_resolving_track_id_is_still_accepted`
  hand-built a video bundle whose published face carried `p_fake=0.9` beside a
  computed headline of `0.9796…` — itself an instance of the inconsistency R21
  is about, which no test noticed (the same shape as Batch 13's *"no test
  noticed"*). Its face score is now taken from the bundle's own headline; the
  test's intent — a resolving, scored track is accepted — is unchanged.
  `video_face_semantics_study.py` case C got the same one-line reconciliation.
* **Migrations: none.** No data on disk changes meaning or shape.
* **Fingerprint changes: none.**

---

## 12. Guarantees gained

1. **No two published `FaceTrack`s share a `track_id`** — by hand, from the
   factory, on deserialisation. Parity with `_branches_unique`.
2. **A decided headline `p_fake` is the score of a face the bundle publishes** —
   whenever any published face scored. A bundle whose headline contradicts its
   own evidence is unrepresentable.
3. **`_top_frames_name_a_published_track` now resolves against a de-duplicated,
   unique set by construction**, closing the gap where two id-0 faces made its
   check ambiguous.
4. **R16's implementation has an invariant to build against.** A per-face design
   that publishes N tracks cannot ship a bundle whose headline names none of
   them.
5. **A future averaging aggregate trips a loud failure** rather than silently
   redefining a Stable field.

---

## 13. Guarantees NOT gained

1. **R16 is untouched.** One face's verdict is still published as the media's;
   the invariant makes a self-contradicting bundle impossible, not a group
   photograph's verdict correct.
2. **Non-selected faces are still not scored, abstained, or counted.**
3. **The headline rule is vacuous when no published face scored** — a decided
   bundle with `faces=[]` (video, R18) or with only abstained faces is accepted.
   Distinguishing the incoherent second case from the legitimate first needs a
   `face_stage` signal the contract does not carry (R18).
4. **The video path still publishes no faces** and its `top_frames` still name no
   track. Unchanged from Batch 13.
5. **No `faces[]` ↔ `disagreement` / `interval` relation.** The review's point
   that an unexamined second face does not widen the interval (§8) stands.
6. **No fingerprint, so nothing here protects a future policy or geometry
   change** — R25 is still open and still wants to land with the geometry change
   it guards.
7. **R20, R22, R6, R9, R10, R13, R23, R24, R25, R26 untouched.**
8. **`bbox` is still the detector's box, not the scored region** (R26).

---

## 14. Re-running

```bash
python tests/test_faces_consistency.py     # 16 tests, standalone
python tests/run_all.py                     # full suite
```

The study numbers this batch builds on are in `tests/multi_face_study.py`
(unchanged) and `docs/MULTI_FACE_SEMANTICS.md` §7.
