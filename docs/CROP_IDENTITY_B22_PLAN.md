# Batch 22 implementation plan — R25, revision 2 (B22-R2)

> Supersedes the implied plan in `docs/CROP_IDENTITY.md` §9 and incorporates
> every required change from `docs/CROP_IDENTITY_B22_REVIEW.md` (verdict: APPROVE
> WITH CHANGES). **No code in this document.** This is design correction only.
>
> Lineage: B15 opened R25 · B21 architecture (`CROP_IDENTITY.md`) · B22 review
> (`CROP_IDENTITY_B22_REVIEW.md`) · **B22-R2 (this document)**.

**Tree state.** 48 suites / 1217 tests / 0 failures.
`STUB_ENGINE_VERSION = stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387.d.a97ccb68`
(`backend/config.py:41`). `default_face_selection().identity == "19559387"`.

---

## 0. Scope of B22-R2, restated after the review

B22 closes **R25 only** — the three preprocessing knobs enter the model identity.
It is **output-identical** (no shipped pixel changes), forces **one cache
generation**, and needs **no benchmark recalibration** and **no contract /
`RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` / OpenAPI / `schema_out`
change**.

**R28 (the `resize` `or`-predicate pixel defect) is explicitly deferred** — see
§5, *Why R28 Is Deferred*. B22 names the interpolation rule with its current
shipped behaviour and protects that behaviour with a re-derivation test; it does
**not** change the predicate.

Closure strengths differ by knob and this is now stated up front (§4):

| Knob | Home | Closure mechanism | Strength |
|---|---|---|---|
| 1 `input_size` | weights identity | **functionally threaded** — `prepare_image`/`extract_frames` read `self._spec.input_size` and it now enters the hash | **strong** |
| 2 interpolation rule | face-selection identity | **label pinned by a re-derivation test** — `resize` does not read the policy | weaker |
| 3 `crop_clip_rule` | face-selection identity | **label pinned by a re-derivation test** — one value today, `crop_face` does not branch on it yet | weaker |

---

# A. Revised Batch 22 design document

## A.1 Knob 1 — `input_size` → weights identity

**Change.** `ModelSpec.weights_identity` (`inference/models.py:256-271`) folds
`input_size` into the hash. **Chosen method:** add `"input_size"` to the existing
`weights_part` fingerprint dict:

```
weights_part = fingerprint({**{b.value: w for b, w in self.branch_weights.items()},
                            "input_size": self.input_size})
```

not a new `-i<fp>` suffix.

**Rationale for the method choice (ADR).** The review flagged the `-i` suffix as
"the slightly higher-risk choice" because it adds a format element to a string
that four functions parse. Folding into `weights_part` moves the existing
`7f45fada` hash to a new value with **zero format change** — `weights_identity`
stays `f"{version}{branch_part}-w{weights_part}"`, `parse_model_version` is
completely untouched, and there is no new `-i` token for a future reader to
mis-split. The cost is operator legibility (an operator diffing two versions sees
`-w7f45fada` → `-wXXXXXXXX` and cannot tell *which* input moved), which is
acceptable: `input_size` changing without the checkpoint changing is a
misconfiguration, and `verify_version_alignment` names the mismatch at startup
regardless of legibility. Recorded in the `weights_identity` docstring.

**Why weights and not a new segment or `.f.`** — unchanged from B21 §2: a
checkpoint has an input resolution; running the wrong one is a weights
misconfiguration; `weights_identity` is where a weights misconfiguration should
surface.

**Functional thread (why knob 1 is a strong closure).** `input_size` is read at
`image_pipeline.py:98,198` and `video_pipeline.py:141,176` — `prepare_image(crop,
size=self._spec.input_size)` and `extract_frames(target_size=self._spec.input_size)`.
The value that drives the tensor shape and the value that drives the hash are the
**same attribute of the same object**. A deployment cannot change the resize
target without changing the identity.

## A.2 Knob 2 — `interpolation_rule` → `FaceSelectionPolicy`

**Change.** Add a field to `FaceSelectionPolicy` (`inference/preprocess.py:433`):

```
interpolation_rule: str
```

no default (the class forbids defaults — `test_the_policy_has_no_defaults`).
Add it to `.identity`'s dict (`preprocess.py:510`), `.to_dict` (`:523`),
`.differences`'s field tuple (`:540`), and set it in `default_face_selection()`
(`:569`) — the **sole constructor**.

**Value — names the current shipped `or` behaviour, verbatim:**

```
interpolation_rule = "area-if-either-axis-exceeds-target-else-linear"
```

This describes `resize`'s actual branch `cv2.INTER_AREA if (h > size or w > size)
else cv2.INTER_LINEAR` (`preprocess.py:237`) — **including** the `or` that R28
objects to. B22 does not fix the predicate; it records what the predicate does.

**Closure strength — weaker than knob 1, stated in the field docstring.**
`resize` is a bare module function; it does not receive the policy. So
`interpolation_rule` is a *label*, not a threaded parameter. The label is held to
the behaviour by a **re-derivation test** (§C, `test_the_interpolation_rule_name_matches_resize`)
in the exact pattern `ORDERING_RULE` (`preprocess.py:586` / `test_face_selection_identity.py:330`)
and `crop_shape` already use. A code edit to `resize` that bypasses the label
re-creates R25 one level down; the re-derivation test is the only thing that
catches it. This limitation is documented in:

- the `interpolation_rule` field docstring (`preprocess.py`),
- §4 of this plan (design rationale),
- the B22 section of `REPRODUCIBILITY_AUDIT.md`.

## A.3 Knob 3 — `crop_clip_rule` → `FaceSelectionPolicy`

**Change.** Add a field:

```
crop_clip_rule: str
```

no default. Add to `.identity`, `.to_dict`, `.differences`, set in
`default_face_selection()`.

**Value — names the current shipped behaviour:**

```
crop_clip_rule = "intersect-image"
```

`crop_face` (`preprocess.py:708-719`) computes a square window then does
`x1 = min(w, x0 + side)` / `y1 = min(h, y0 + side)` and slices — the window is
**intersected** with the image, which is why a border-flush box yields a
non-square crop (R23). `"intersect-image"` records that.

**Closure strength — weaker than knob 1, stated in the field docstring.**
`crop_face` *does* take `policy`, so `crop_clip_rule` *could* be functionally
threaded — but with a single value it will not branch on anything until R23
introduces `"pad-<fill>"`. Today it is a label pinned by a re-derivation test
(§C, `test_the_clip_rule_name_matches_crop_face`). When R23 lands, the value
changes to the new fill mode and `.f.` moves a second time — the one accepted
double-spend, scoped to this one field (B21 §11, unchanged).

## A.4 Composition, migration, cache — unchanged from B21, re-confirmed by the review

- `effective_version` = `compose_model_version(weights_identity,
  calibration_fingerprint, gate_fingerprint, face_selection_fingerprint)` —
  `models.py:274`. Knob 1 moves `weights`; knobs 2+3 move `face_selection`.
- Projected: `stub-0-w<NEW>+c.739c3fec.g.ae232cba.f.<NEW>`. `calibration` and
  `gate` fingerprints unchanged. `parse_model_version` round-trips (no format
  change at all now — see A.1). `model_identity_of` recovers it from the runtime
  string.
- **Migration:** `STUB_ENGINE_VERSION` moves; one cache generation, image + video
  namespaces, one-time full miss per file, namespaces provably disjoint, no stale
  hit. **No** `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` /
  `CALIBRATION_WITNESS_VERSION` / OpenAPI / `schema_out`. Back-fill refused
  (B9/B11).
- **Benchmark:** `regression_gate` (`benchmark.py:783`) does not short-circuit on
  `effective_model_version` inequality (only `dataset_fingerprint` and
  `decode_identity` do). A pre/post-B22 pair still compares numerically and
  correctly (numbers identical under the shipped detector). `BenchmarkReport.validate()`
  refuses a mismatched pair — the intended tightening. No `max_*` change.
- **Witness:** no digest change. `input_size` protection rides on
  `effective_model_version` moving. The incidental-binding residual (geometry
  bound only via temperature drift; converged temperatures collide) is real,
  accurately scoped, and deferred to a future `CALIBRATION_WITNESS_VERSION`
  batch — **not** B22.

---

# B. Updated implementation checklist

**Complete and internally consistent. Every item below was verified against the
current tree in the B22 review.**

## B.1 Production code

| # | File | Symbol | Edit (described) | Why |
|---|---|---|---|---|
| P1 | `inference/models.py` | `ModelSpec.weights_identity` (`:256-271`) | add `"input_size": self.input_size` to the `weights_part` fingerprint dict; extend docstring with the ADR (A.1) | knob 1 |
| P2 | `inference/preprocess.py` | `FaceSelectionPolicy` (`:433`) | add `interpolation_rule: str` and `crop_clip_rule: str`, no defaults, with docstrings stating the label-vs-threaded limitation (A.2, A.3) | knobs 2, 3 |
| P3 | `inference/preprocess.py` | `FaceSelectionPolicy.identity` (`:510-521`) | add both keys to the fingerprint dict | knobs 2, 3 |
| P4 | `inference/preprocess.py` | `FaceSelectionPolicy.to_dict` (`:523-535`) | add both keys | forensic record parity |
| P5 | `inference/preprocess.py` | `FaceSelectionPolicy.differences` (`:537-553`) | add both field names to the iterated tuple | field-by-field diff parity |
| P6 | `inference/preprocess.py` | `default_face_selection()` (`:569`) — **sole constructor** | set `interpolation_rule="area-if-either-axis-exceeds-target-else-linear"`, `crop_clip_rule="intersect-image"` | the one place shipped values live |
| P7 | `backend/config.py` | `STUB_ENGINE_VERSION` (`:41`) | regenerate from what the engines report after P1–P6; add a B22 paragraph to the docstring (the file's own instruction, `:107-109`) | read key = write key |

**No other production file changes.** Confirmed: `grep input_size contracts/` = 0;
`weights_identity` has zero sub-parsers; `FaceSelectionPolicy` has one
constructor; `resize`/`crop_face` signatures unchanged (label closures).

## B.2 Test code — pins to update (every one verified in the review)

| # | File:line | Current | Action | Class |
|---|---|---|---|---|
| T1 | `tests/test_crop_geometry.py:457` | `test_input_size_is_in_no_identity_string` — asserts `_spec(384).effective_version == _spec(224).effective_version` | **genuine fail-before**: keep as-is through the red run, then rewrite to assert `!=` and rename → `test_input_size_moves_the_weights_identity` | fail-before → replacement |
| T2 | `tests/test_crop_geometry.py:467` | `test_the_interpolation_policy_is_in_no_identity_string` — no identity-equality assertion; `{"interpolation","interp","resize"}` substring check + `np.array_equal(shipped, area)` | **REPLACE** (does not fail after the fix). New: `test_the_interpolation_rule_moves_the_face_selection_identity` (§C) | replacement |
| T3 | `tests/test_crop_geometry.py:492` | `test_the_clipping_rule_is_in_no_identity_string` — asserts only `clipped.shape != padded.shape` and `prepare_image` inequality; **no identity assertion** | **REPLACE** (stays green forever after the fix). New: `test_the_clip_rule_moves_the_face_selection_identity` (§C) | replacement |
| T4 | `tests/test_crop_geometry.py:514` | `test_two_deployments_can_agree_on_every_identity_and_differ` — asserts `a.effective_version == b.effective_version` (a,b = `_spec(384),_spec(224)`) | **genuine fail-before**: rewrite to assert the two now get different identities → `test_two_deployments_that_differ_on_geometry_now_get_different_identities` | fail-before → replacement |
| T5 | `tests/test_crop_geometry.py:630` | `test_batch_15_changed_no_geometry` — `POLICY.identity == "19559387"`, `model_identity_of(STUB_ENGINE_VERSION) == "…f.19559387"` | rewrite → `test_batch_22_moved_the_identity_not_the_pixels`: assert new `.f.`, new `STUB_ENGINE_VERSION`, **and byte-identical `crop_face`/`resize` output** vs `LegacyClipToBounds` / current `resize` | collateral → replacement |
| T6 | `tests/test_face_selection_identity.py:290-311` | `test_every_policy_field_moves_the_identity` — `alternatives` dict (8 keys) + **hard** `assert names == set(alternatives)` at `:301` | add `interpolation_rule` and `crop_clip_rule` to `alternatives` with distinct values (e.g. `"linear-always"`, `"pad-replicate"`); the loop then also proves both move the identity | **genuine fail-before** (fails at `:301` the instant a field is added) + pass-after |
| T7 | `tests/test_face_selection_identity.py:314` | `test_the_policy_has_no_defaults` — iterates `fields()` | no code change; **verifies** P2 gave the new fields no defaults | pass-after (guard) |
| T8 | `tests/test_branch_b_semantics.py:456` | `_EXPECTED_EFFECTIVE = "stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387"` (used `:463` vs `eng.model_identity`) | update the literal to the new model identity | collateral pin |
| T9 | `tests/test_headline_and_face_semantics.py:92` | `_EXPECTED_EFFECTIVE = "…f.19559387"` (used `:445`, `:447`, `:549`) | update the literal | collateral pin |
| T10 | `tests/test_decode_identity_completeness.py:90` | `MODEL_ID = "…f.19559387"` (compared to live `model_identity_of(STUB_ENGINE_VERSION)` at `:342`; hand-built fixtures at `:154`, `:244-245`) | update `MODEL_ID`; the hand-built fixtures follow for consistency | collateral pin |
| T11 | `tests/test_model_version.py:275` | `test_configured_model_version_matches_the_engines` | no code change; **self-heals** once P7 lands; fails loudly until then (the designed signal) | pass-after (guard) |
| T12 | `tests/run_all.py:68` (SUITES list ends `:116`) | 48 suites | register `("S8-B22 crop identity", test_crop_identity)` → **49 suites** | new suite |

**Not touched (verified):** `test_run_identity.py` (`MODEL` is a hand-built
constant `compose_model_version(weights="stub-0-w7f45fada", …)`, self-consistent);
`test_evaluation_identity.py` (`_bench` uses a hand-set `MODEL_ID` with no `.f.`);
`test_inference.py:323,328` (relative comparisons, both sides move together);
`PRE_BATCH_12` literal (historical, no `-i`, no `.f.` — still parses).

## B.3 New files

| # | File | Content |
|---|---|---|
| N1 | `tests/test_crop_identity.py` | the pass-after suite (§C, "pass-after" rows). Registered in `run_all.py` (T12). |
| N2 | `tests/crop_identity_study.py` | **exists (B21)**. Update §4 (projected identity) to the real post-fix values once P1–P7 land; update §5 wording (method is now `weights_part` fold, not `-i` suffix). Keep §1–3, §6. |

## B.4 Study narratives that become false after B22 (must fix — not asserted, but must not lie)

| # | File | What goes stale | Action |
|---|---|---|---|
| S1 | `tests/crop_geometry_study.py:614-622` | the stage printing "1. ModelSpec.input_size 384 → … / 224 → … / identical: True" — **inverts** | rewrite the stage to show the identities now differ, or annotate "superseded by B22" |
| S2 | `tests/face_selection_study.py` (`:251`, `:455`, `:511`) | prints `SHIPPED.identity` / `face_selection_identity_of(STUB_ENGINE_VERSION)` — value changes | regenerate any doc numbers it feeds; no logic change |
| S3 | `tests/multi_face_study.py:556-563`, `tests/face_size_policy_study.py:648-663` | print `policy.identity` / `effective_version` | same — regenerate quoted numbers |

## B.5 Documentation

| # | File | Action |
|---|---|---|
| D1 | `backend/config.py:42-110` docstring | add a B22 paragraph: the `.f.` moved (R25 knob 2+3), the weights hash moved (R25 knob 1), one cache generation, no contract/schema/witness change |
| D2 | `docs/CROP_IDENTITY.md` | append an "Outcome (B22)" section with the final identity values and a pointer to this plan |
| D3 | `docs/CROP_GEOMETRY.md` | §6 (R25) and §10 — R25 is now CLOSED for knobs 1+2, and knob 3's *field* is closed (value pending R23). One-line "superseded by B22" where `19559387` is cited as current |
| D4 | `docs/FACE_SELECTION_IDENTITY.md` | the `.f.` fingerprint list grows by two fields; `19559387` → new value; superseding note |
| D5 | `docs/FACE_SIZE_POLICY.md`, `docs/HEADLINE_AND_FACE_SEMANTICS.md`, `docs/BRANCH_B_SEMANTICS.md`, `docs/DECODE_IDENTITY_COMPLETENESS.md`, `docs/FRAME_COUNT_REPRODUCIBILITY.md`, `docs/MULTI_FACE_SEMANTICS.md` | one-line "superseded by B22 — model identity is now `<new>`; historical value correct-for-then" wherever `19559387` or the full identity is cited as *current* (B20's method for the `.d.` move) |
| D6 | `REPRODUCIBILITY_AUDIT.md` | add the Batch 22 section (§ "Audit update" below); mark R25 CLOSED, R28 OPEN (new, deferred) |
| D7 | every historical `docs/*` that cites `c20a064a`/`a97ccb68`/`19559387` as a *measurement* | leave unchanged — those are correct-for-then (the B20 rule) |

## B.6 Regenerating documented identity values

After P1–P7 are in and green, regenerate and record in one place (`docs/CROP_IDENTITY.md`
Outcome section):

- `default_face_selection().identity` (new `.f.`)
- `ModelSpec(...).weights_identity` (new `-w` hash)
- `STUB_IMAGE.with_gate(...).with_face_selection(...).effective_version` (new model identity)
- `compose_runtime_version(model=…, decode=decode_identity())` (new `STUB_ENGINE_VERSION`)
- `fingerprint({"input_size": 384})` component (for the record)

Then propagate to T7–T10 pins and D1–D6 docs. **One regeneration pass, one commit
of derived values**, so the tree is never half-updated.

---

# C. Updated fail-before / pass-after matrix

Three explicit classes, per the review.

## C.1 Genuine fail-before tests (RED before implementation, by an assertion that the fix actually violates)

| Test | File | The assertion that fails | Fails because |
|---|---|---|---|
| `test_input_size_is_in_no_identity_string` | `test_crop_geometry.py:459` | `_spec(384).effective_version == _spec(224).effective_version` | P1 makes the two `weights_identity` differ |
| `test_two_deployments_can_agree_on_every_identity_and_differ` | `test_crop_geometry.py:522` | `a.effective_version == b.effective_version` (a,b = `_spec(384),_spec(224)`) | same |
| `test_every_policy_field_moves_the_identity` | `test_face_selection_identity.py:301` | `assert names == set(alternatives)` — `names` is `{f.name for f in fields(FaceSelectionPolicy)}` | P2 adds two field names not in the hand-written `alternatives` dict |
| `test_batch_15_changed_no_geometry` | `test_crop_geometry.py:638,648` | `POLICY.identity == "19559387"` and `model_identity_of(STUB_ENGINE_VERSION) == "…f.19559387"` | P3 moves `.f.`; P1 moves weights; P7 moves the literal |
| `test_configured_model_version_matches_the_engines` | `test_model_version.py:275` | `settings.model_version == engine_versions()[...]` | P1–P6 move the engine value; RED until P7 regenerates the literal |
| `_EXPECTED_EFFECTIVE` checks | `test_branch_b_semantics.py:463`, `test_headline_and_face_semantics.py:445,447` | `eng.model_identity == _EXPECTED_EFFECTIVE` | literal is the old model identity |
| `MODEL_ID` check | `test_decode_identity_completeness.py:342` | `model_identity_of(STUB_ENGINE_VERSION) == MODEL_ID` | literal is the old model identity |

**Run order:** land P1–P6 → run the suite → capture the RED list → confirm it
matches this table exactly (no surprises) → then P7 + T1–T12 + N1 → GREEN.

## C.2 Replacement tests (the old test cannot be salvaged by inverting an assertion; it is deleted and a positive test written)

| Old test (deleted) | File | Why it cannot be inverted | New test (positive) |
|---|---|---|---|
| `test_the_interpolation_policy_is_in_no_identity_string` | `test_crop_geometry.py:467` | its only identity-gap signal is `not {"interpolation","interp","resize"} & set(POLICY.to_dict())` — a substring check that stays green with a field named `interpolation_rule`; and `assert np.array_equal(shipped, area)` is about `resize` behaviour, not identity | `test_the_interpolation_rule_moves_the_face_selection_identity` (C.3) |
| `test_the_clipping_rule_is_in_no_identity_string` | `test_crop_geometry.py:492` | contains **no identity assertion at all** — only `clipped.shape != padded.shape` and `prepare_image(clipped) != prepare_image(padded)`, both unchanged by P1–P7 | `test_the_clip_rule_moves_the_face_selection_identity` (C.3) |
| `test_input_size_is_in_no_identity_string` | `test_crop_geometry.py:457` | *is* a genuine fail-before (C.1) but its name/intent invert; keep the construction, flip the assertion, rename | `test_input_size_moves_the_weights_identity` (C.3) |
| `test_two_deployments_can_agree_on_every_identity_and_differ` | `test_crop_geometry.py:514` | intent inverts | `test_two_deployments_that_differ_on_geometry_now_get_different_identities` (C.3) |
| `test_batch_15_changed_no_geometry` | `test_crop_geometry.py:630` | premise ("investigation batch, nothing moved") is gone | `test_batch_22_moved_the_identity_not_the_pixels` (C.3) |

## C.3 Pass-after tests (GREEN only after implementation; live in `tests/test_crop_identity.py` unless noted)

| Test | Asserts |
|---|---|
| `test_input_size_moves_the_weights_identity` *(replaces T1)* | `ModelSpec(input_size=384).weights_identity != ModelSpec(input_size=224).weights_identity`; and `…effective_version` differs; and the *only* difference in `parse_model_version` is the `weights` segment |
| `test_the_interpolation_rule_moves_the_face_selection_identity` *(replaces T2)* | `replace(SHIPPED, interpolation_rule="linear-always").identity != SHIPPED.identity` — two `FaceSelectionPolicy` differing **only** in the knob → different `.identity` |
| `test_the_clip_rule_moves_the_face_selection_identity` *(replaces T3)* | `replace(SHIPPED, crop_clip_rule="pad-replicate").identity != SHIPPED.identity` |
| `test_two_deployments_that_differ_on_geometry_now_get_different_identities` *(replaces T4)* | `_spec(384).effective_version != _spec(224).effective_version`; `_spec` policies differing only in `interpolation_rule` / `crop_clip_rule` → different `effective_version` |
| `test_the_interpolation_rule_name_matches_resize` | re-derives `resize`'s branch: for a grid of `(h,w,size)` including the mixed case `h>size xor w>size`, `cv2.INTER_AREA if (h>size or w>size) else cv2.INTER_LINEAR` is exactly what the name `"area-if-either-axis-exceeds-target-else-linear"` describes; fails if `resize` or the name drifts apart |
| `test_the_clip_rule_name_matches_crop_face` | re-derives: `crop_face`'s window is `image[y0:min(h,y0+side), x0:min(w,x0+side)]` — an intersection; the name `"intersect-image"` is held to that; fails if `crop_face` starts padding without the name changing |
| `test_every_policy_field_moves_the_identity` *(T6, extended in place)* | `names == set(alternatives)` with 10 keys; the loop proves each of the 10 (incl. the 2 new) moves `.identity` |
| `test_the_policy_has_no_defaults` *(T7, unchanged)* | the 2 new fields have `default is MISSING and default_factory is MISSING` |
| `test_the_shipped_pixels_did_not_move` | `crop_face(img, box, policy=default_face_selection())` byte-equals `LegacyClipToBounds.crop(img, box)` across the B15 position sweep; `resize(x, 384)` byte-equals the pre-B22 `resize` on a grid of shapes — **the output-identical guarantee, asserted** |
| `test_the_migration_moved_only_what_it_should` | `CONTRACT_VERSION`, `RUN_SCHEMA_VERSION`, `CALIBRATION_WITNESS_VERSION`, the OpenAPI structural signature, `schema_out/*` byte-hashes — all unchanged; `STUB_ENGINE_VERSION` and `default_face_selection().identity` — changed exactly once vs the recorded pre-B22 values |
| `test_parse_model_version_is_total_across_all_eras` | pre-B9 (`w+c.x`), B9–B13 (`…f.y.d.z` with `.f.` from B12), B14–B20 (`…f.19559387.d.a97ccb68`), B22 (`…-w<new>+c.….f.<new>.d.…`) all `parse` and `compose()` round-trip |
| `test_old_artifacts_still_load` | a stored `EvidenceBundle` JSON with the old `model.version`, a `CalibrationWitness` with the old string, a `BenchmarkRun` row with the old `effective_model_version` — all `model_validate` / `from_dict` without error |
| `test_alignment_fails_on_an_input_size_mismatch` | `verify_version_alignment` raises when the configured literal's weights segment ≠ the engine's (engine built with a different `input_size`) |
| `test_the_benchmark_gate_still_compares_across_the_boundary` | `regression_gate(pre_b22_run, post_b22_run)` does **not** short-circuit (different `effective_model_version` is a note, not a failure); `BenchmarkReport.validate()` **does** refuse the mismatched pair |
| `test_batch_22_moved_the_identity_not_the_pixels` *(replaces T5, lives in `test_crop_geometry.py`)* | new `.f.`, new `STUB_ENGINE_VERSION`; `crop_face`/`resize` byte-identical; `CONTRACT_VERSION == "1.0.0"` |

---

# D. Updated risk register

### Implementation risks

| Risk | Sev | Mitigation |
|---|---|---|
| **knobs 2 & 3 are label closures — a `resize`/`crop_face` edit that bypasses the label re-creates R25 one level down** | **medium** | mandatory re-derivation tests `test_the_interpolation_rule_name_matches_resize` / `test_the_clip_rule_name_matches_crop_face`; limitation documented in field docstrings + §4 + audit; this is the *same* mechanism `ordering`/`crop_shape` already rely on |
| interpolation-rule name bakes the `or` (R28) into the identity's named contract | medium | **accepted and explicit** (§5). The name is honest about current behaviour; R28's fix will move `.f.` again, recorded as a known future cost |
| new `FaceSelectionPolicy` fields given defaults for convenience | low | `test_the_policy_has_no_defaults` fails on this (T7) |
| `weights_part` fold vs `-i` suffix decided inconsistently with future gate/face work | low | ADR in the `weights_identity` docstring (A.1); the fold is the zero-format-change choice |
| half-updated tree (some pins/docs regenerated, some not) | medium | B.6 mandates one regeneration pass → one commit of all derived values |

### Migration risks

| Risk | Sev | Mitigation |
|---|---|---|
| `STUB_ENGINE_VERSION` not regenerated → silent cache split (read key ≠ write key) | **high if missed** | `test_configured_model_version_matches_the_engines` fails the build loudly (C.1); the literal's docstring says "paste in the version it reports" |
| benchmark history spanning the boundary | low | `regression_gate` treats pre/post as different models — a note, not a refusal; numbers identical under the shipped detector; `read_runs` version-tolerant; `test_the_benchmark_gate_still_compares_across_the_boundary` pins it |
| back-fill temptation | medium | explicitly refused (B9/B11); restated in D6 |
| a deployment pinning `DFP_MODEL_VERSION` to the old string | expected | fails alignment at startup by design (B9/B20); release note |

### Backward-compatibility risks

| Risk | Sev | Mitigation |
|---|---|---|
| historical version fixtures (`PRE_BATCH_12`, `test_run_identity.MODEL`) stop parsing | **none** — verified: no `-i`, and A.1's fold adds no format token | `test_parse_model_version_is_total_across_all_eras` |
| stored `EvidenceBundle` / `CalibrationWitness` / split-lock / `BenchmarkRun` JSON | none — opaque strings, own-era verification | `test_old_artifacts_still_load` |
| `RUN_SCHEMA_VERSION` bump sneaks in via a `BenchmarkRun` field | none — no `BenchmarkRun` field added; `effective_model_version` already carries `.f.` | `test_the_migration_moved_only_what_it_should` |

### Testing risks

| Risk | Sev | Mitigation |
|---|---|---|
| **relying on B21's "invert the assertion" for T2/T3** — they do not fail after the fix | **resolved** | C.2 deletes both and writes positive `.identity`-inequality tests (C.3) |
| `test_every_policy_field_moves_the_identity:301` not on B21's list | resolved | T6, C.1 — it *is* a genuine fail-before; extend the `alternatives` dict |
| a study narrative silently goes stale and a later batch quotes it | medium | B.4 lists every affected study; B.6 forces regeneration |
| full-suite runtime (`test_evaluation_operating` 5–15 min) | low | known, unrelated; budget the run; not a per-test timeout |
| the RED run surfaces a pin this plan missed | low→resolved | C.1 "Run order" step: capture RED list, confirm it matches C.1 exactly before proceeding |

### Operational risks

| Risk | Sev | Mitigation |
|---|---|---|
| one-time full cache miss on deploy | expected, bounded | precedented B9/B14/B20; communicate the compute spike |
| mixed fleet during rollout (old `.f.19559387` workers + new) | low | safe direction (false miss, not false hit); alignment check stops a mismatched API/worker pair |

---

# 5. Why R28 Is Deferred

**R28 (candidate finding).** `inference/preprocess.py:237` —
`interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR`.
The `or` selects `INTER_AREA` for the **whole** resize whenever **either** axis
exceeds the target, so a crop that is wider than `input_size` but shorter (e.g.
500×300 → 384) is area-filtered on the axis it is **enlarging** (300 → 384) — a
low-pass applied to an upsample, in a function whose docstring picks `INTER_AREA`
because "forensic signal lives in high-frequency texture". Measured (study §3):
mean |AREA − LINEAR| on the enlarged axis = 0.040/255.

**Why it is not fixed in B22:**

1. **Fixing the predicate changes shipped pixels.** `or → and` (or a
   single-scale-factor rule) alters the output for every resize where exactly one
   axis crosses `input_size`. That is a real, content-dependent pixel change →
   a benchmark boundary that cannot be cleanly bounded → it **breaks B22's
   headline guarantees**: output-identical, no recalibration, one clean cache
   generation attributable to identity completion alone.

2. **B22's value is the clean closure.** R25 is worth doing precisely because it
   is a pure identity-completeness fix with the B14/B20 migration profile.
   Bundling a pixel change forfeits that and makes the migration analysis
   (§A.4, D) partly wrong.

3. **No urgency.** R28's impact is small and content-dependent, **rare under
   Haar** (square-ish crops seldom straddle `input_size`) and **near-zero on the
   video path** (whole frames are almost always both-axes-larger than 384). There
   is no trained model and no benchmark corpus today, so nothing is being
   actively corrupted while it waits.

4. **It rides a future `.f.`-moving batch at zero marginal cache cost.** R23 /
   SCRFD, R16, or any batch that already regenerates `.f.` can carry R28's
   predicate fix and its re-derivation-test update together. The interpolation
   rule's *name* changes then (`"area-if-either-axis-exceeds-target-else-linear"`
   → the corrected rule) and `.f.` moves once more — the same accepted
   single double-spend that knob 3 carries for R23.

**What B22 does instead:**

- Names the interpolation rule `"area-if-either-axis-exceeds-target-else-linear"`
  — an **accurate** description of the current `or` behaviour, not the
  docstring's misleading "when shrinking".
- Protects that behaviour with `test_the_interpolation_rule_name_matches_resize`
  (C.3), which re-derives `resize`'s branch — including the mixed case — and
  fails if code and name drift.
- Records R28 as an **open finding with its own ID** in `REPRODUCIBILITY_AUDIT.md`
  (not folded into R25), with the deferral rationale above and the activation
  note "fix with the next batch that moves `.f.`".

**R28 remains OPEN after B22.** B22 closes R25 only.

---

# 6. Audit update (for `REPRODUCIBILITY_AUDIT.md`)

> # Batch 22 — R25 closed (crop-geometry identity completion)
>
> **Output-identical. One cache generation. No contract / `RUN_SCHEMA_VERSION` /
> `CALIBRATION_WITNESS_VERSION` / OpenAPI / `schema_out` change.** Plan and
> proof: `docs/CROP_IDENTITY_B22_PLAN.md`. Study: `tests/crop_identity_study.py`.
> Proof suite: `tests/test_crop_identity.py` (49 suites now).
>
> R25's three knobs now enter the model identity: `ModelSpec.input_size` folds
> into `weights_identity` (functionally threaded — `prepare_image` /
> `extract_frames` read the same attribute); `interpolation_rule` and
> `crop_clip_rule` become `FaceSelectionPolicy` fields (label closures pinned by
> re-derivation tests, the `ORDERING_RULE` pattern — weaker than knob 1, stated
> in the field docstrings). `default_face_selection()` sets
> `interpolation_rule="area-if-either-axis-exceeds-target-else-linear"` (the
> current shipped `or` behaviour, verbatim) and `crop_clip_rule="intersect-image"`.
>
> `STUB_ENGINE_VERSION` `.f.` `19559387` → `<new>`, weights `-w7f45fada` →
> `-w<new>`. `model_identity_of` / `parse_model_version` unchanged (the
> `input_size` fold adds no format token). Benchmark: `regression_gate` compares
> a pre/post-B22 pair numerically (identical numbers under the shipped detector);
> `BenchmarkReport.validate` refuses the mismatched pair. Witness: no digest
> change; geometry still bound only incidentally via temperature drift — recorded
> residual, deferred to a future `CALIBRATION_WITNESS_VERSION` batch.
>
> **New finding R28 (OPEN, deferred):** `resize`'s `h > size or w > size`
> predicate area-filters an axis it enlarges (500×300 → 384: `INTER_AREA` on the
> 300→384 axis; 0.040/255). Fixing the predicate moves shipped pixels, so it is
> **not** in B22 — deferred to the next batch that moves `.f.`, which absorbs its
> cache generation. B22 names the rule with the current behaviour and pins it.
>
> **Still open after Batch 22:** R6-a, R6-b, R9, R10, R13, R16, R23, R24, R26,
> **R28** (new), D8–D10. **R25 CLOSED.**

---

# E. Is the reviewer's verdict fully satisfied?

**Yes — all five required changes are incorporated, and the verdict conditions
are met.**

| Reviewer requirement | Where addressed | Satisfied? |
|---|---|---|
| **1. Fix the fail-before strategy** — replace (not invert) `test_the_clipping_rule_is_in_no_identity_string` and `test_the_interpolation_policy_is_in_no_identity_string`; rewrite both as positive two-policy `.identity`-inequality tests; distinguish genuine fail-before / replacement / pass-after | §C.1 (7 genuine fail-before, incl. the previously-missed `:301`), §C.2 (both named tests **deleted**, not inverted, with the reason each cannot be salvaged), §C.3 (`test_the_interpolation_rule_moves_the_face_selection_identity`, `test_the_clip_rule_moves_the_face_selection_identity` — each asserts `replace(SHIPPED, knob=…).identity != SHIPPED.identity`) | **yes** |
| **2. Correct the pin inventory** — add `_EXPECTED_EFFECTIVE` (×2), `MODEL_ID`, `test_every_policy_field_moves_the_identity` field-set assertion to every checklist / impact / migration / touched-file table | §B.2 rows T6, T8, T9, T10; §C.1 rows; §B.6 propagation; internally consistent (every pin appears in the checklist **and** the fail-before matrix **and** the regeneration pass) | **yes** |
| **3. Clarify closure strength** — state `input_size` is functionally threaded; `interpolation_rule` / `crop_clip_rule` are label closures via re-derivation tests; knobs 2&3 are weaker; document in design rationale **and** field docs | §0 table, §A.1 "Functional thread", §A.2 / §A.3 "Closure strength" paragraphs (explicitly "weaker than knob 1", "documented in the field docstring"), §4 risk register row, §6 audit text | **yes** |
| **4. Split R28 (option C)** — B22 closes R25 only, output-identical, no-recalibration, one-cache-generation; name the interpolation rule per the current `or`; add re-derivation protection; defer the predicate fix; dedicated **"Why R28 Is Deferred"** section | §0 scope, §A.2 (name = `"area-if-either-axis-exceeds-target-else-linear"`, verbatim `or`), §C.3 `test_the_interpolation_rule_name_matches_resize`, **§5 "Why R28 Is Deferred"** (4 numbered reasons + what B22 does instead + "R28 remains OPEN"), §6 audit records R28 as its own open finding | **yes** |
| **5. Regeneration and documentation** — regenerate `STUB_ENGINE_VERSION`; update its docstring; fix false study narratives; add superseding notes; regenerate documented identity values | §B.1 P7, §B.4 (S1–S3, S1 is the `crop_geometry_study.py` stage that inverts), §B.5 (D1–D7), §B.6 (one regeneration pass, one commit) | **yes** |

**Residual reviewer notes carried forward, not re-litigated:**

- The witness incidental-binding residual is accurate and stays deferred to a
  future `CALIBRATION_WITNESS_VERSION` batch (reviewer §6 agreed).
- The `weights_part`-fold vs `-i`-suffix decision: B22-R2 chooses the fold (zero
  format change), which is *more* conservative than the reviewer's baseline
  assumption and removes the "format element" risk the reviewer raised.

**Verdict status:** the review's "APPROVE WITH CHANGES" becomes **APPROVE** once
this plan is executed as written. The five required changes are the entire delta;
no new risk is introduced by the revisions (the `weights_part` fold *reduces*
risk relative to B21).

**Still no code.** Next step on approval: execute §B in order — P1–P6, capture
and verify the RED list against §C.1, then P7 + T1–T12 + N1–N2 + S1–S3 + D1–D7 +
B.6 regeneration, single commit of derived values, full-suite green at 49 suites.
