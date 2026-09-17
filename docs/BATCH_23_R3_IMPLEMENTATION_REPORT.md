# Batch 23 (R3) — Implementation Report

**Design executed:** `docs/BATCH_23_R3_DESIGN.md` (verdict YES / unconditional).
**Constraint honoured:** design only — no architecture change, no parser change,
no `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` /
`LOCK_VERSION` bump, no R28 / R30 / R31 work, output-identical guarantees held.

---

## 0. Starting state (important)

The **B23 core** (R29 `VideoSelectionPolicy` + `selection_identity`, N4, the OR1
worker gate + API format check, `.f.` → composite `e629642c`, the two new test
suites) was **already implemented and green** in the tree before this session
(`REPRODUCIBILITY_AUDIT.md` B23 section, `docs/VIDEO_SELECTION_IDENTITY.md`,
`tests/test_video_selection_identity.py`, `tests/test_startup_alignment.py`).

This session implemented the **R3 delta** — the twelve gate-completion items
`BATCH_23_R3_DESIGN.md` folds in on top of that core. The R3 delta **moves no
identity, no wire schema and no version constant**, so the "regenerate exactly
once" step was a no-op verification (see §5).

---

## 1. Files modified

### Production (7)

| File | Change |
|---|---|
| `contracts/version.py` | **+`selection_identity_of(version)`** and **+`is_selection_pinned(version)`** — canonical honest-named readers, bodies identical to the pre-B23 functions (`parse_model_version(v).face_selection` / `FACE_SELECTION_MARKER in v`). `face_selection_identity_of` / `is_face_selection_pinned` re-pointed to `return selection_identity_of(v)` / `return is_selection_pinned(v)` — permanent aliases; `face_selection_identity_of` opens with the "SINCE BATCH 23 … NEVER compare to `default_face_selection().identity`" banner. Module docstring `face` paragraph + readable-format diagram (`weights calibration gate selection decode`) + the "traded away without decoding anything" note + `FACE_SELECTION_MARKER` docstring + `ModelVersionParts` docstring + `compose_model_version` param doc + two error strings all reworded to "selection (face + video frame selection / pooling)". **No parser change** — `parse_model_version`, `_MODEL_MARKERS`, `ModelVersionParts` fields, `compose_model_version` signature untouched. |
| `inference/models.py` | **+`ModelSpec.with_selection(selection_fingerprint)`** — delegates to `with_face_selection`. `with_face_selection` kept as permanent alias target; both docstrings + the `face_selection_fingerprint` field docstring reworded to name the composite. |
| `inference/video_pipeline.py` | spec binding switched `.with_face_selection(...)` → `.with_selection(...)`. |
| `inference/image_pipeline.py` | spec binding switched to `.with_selection(...)`; the "CARRIED, NOT CONSUMED" block comment now states the carry is **PERMANENT** / the image path can **NEVER** sample frames (the asymmetry with `VideoPipeline`'s temporary carry of `FaceSelectionPolicy`). |
| `inference/preprocess.py` | `selection_identity` docstring gains the non-injectivity paragraph (hash-of-two-hashes, 32-bit change detector, no collision risk beyond the two component fingerprints); the "pass the result to `ModelSpec.with_face_selection`" line updated to `.with_selection`. `face_selection_identity()` docstring already B23-aware — untouched. |
| `backend/config.py` | **`skip_version_alignment` Setting removed** (§1.8) — replaced by a comment explaining the env-only worker cannot read it (dead config), the hatch is `$DFP_SKIP_VERSION_ALIGNMENT`, and a wired field returns in B24/OR1-b. `backend/config.py` B23 change is now literal + docstring only. |
| `inference/postprocess.py` | `build_video_analysis` block comment gains one line: the `sampling` / `pooling` kwarg defaults are held equal to `default_video_selection()` by the R35 tripwire. |

### Tests (3)

| File | Change |
|---|---|
| `tests/test_video_selection_identity.py` | imports: `+inspect`, `+FACE_SELECTION_MARKER`, `+face_selection_identity_of`, `+is_face_selection_pinned`, `+is_selection_pinned`, `+selection_identity_of`, `+postprocess as post`, `+ModelSpec`. **+8 tests** (§2). PA15 folded into the existing `test_the_image_pipeline_carries_the_video_segment_without_consuming_it` (asserts `"PERMANENT"` and `"NEVER"` in `image_pipeline.py` source). |
| `tests/test_startup_alignment.py` | **+SA10** `test_api_lifespan_refuses_a_marker_bearing_but_unparseable_model_version` (`"stub-0+c.739c3fec.g.ae232cba.f..d.a97ccb68"` — has `+c.` and `.d.`, `parse_model_version` raises on the empty `.f.` → the `except ValueError` branch aborts `lifespan`). |
| `tests/test_evaluation_identity.py` | `MODEL_ID` (line 75) docstring gains "frozen pre-B22 model-only fixture … do not update on identity moves" (§4.4). |

### Docs (7)

| File | Change |
|---|---|
| `docs/BRANCH_B_SEMANTICS.md`, `docs/CROP_GEOMETRY.md`, `docs/FACE_SIZE_POLICY.md`, `docs/HEADLINE_AND_FACE_SEMANTICS.md`, `docs/CROP_IDENTITY.md` | one-line B23 pointer added to each B22-era prelude blockquote ("Batch 23 (R29) moved the `.f.` segment again to the composite `e629642c`"). `CROP_IDENTITY.md` §16 body also updated to name `selection_identity_of` alongside the alias. (§4.3) |
| `docs/VIDEO_SELECTION_IDENTITY.md` | §7 rewritten — no `Settings.skip_version_alignment`, env-only hatch, B24/OR1-b named. §3 gains the non-injectivity sentence + a paragraph on the honest-named aliases. §8 gains the **version-constant sweep** result, the **fixture** `model.version=="1.0.0-dev"` fact, a **Rollback** subsection and a **Rollout (R34)** subsection with the deploy-runbook line and the "hatch ⇒ B20/B22 semantics" corollary. |
| `REPRODUCIBILITY_AUDIT.md` | summary table gains **R32 / R33 / R34 / R35** rows (R33 + R35 CLOSED-in-B23-R3, R32 + R34 tracked). B23 section: OR1 row + `skip_version_alignment` line rewritten (env-only, B24/OR1-b); "New findings recorded" + "Still open" lists updated; **B24 candidates** list added (OR1-b, R33 rename, R35 required-kwargs, R30, R31). |

**No changes to** `fixtures_out/*`, `schema_out/*`, `backend/openapi.snapshot.json`
(N4 already applied in the B23 core; R3 delta does not touch the wire).

---

## 2. Exact code changes — the new tests

| ID | Test (in `test_video_selection_identity.py` unless noted) | Asserts |
|---|---|---|
| PA12 | `test_the_face_policy_identity_is_unchanged` | `default_face_selection().identity == "251d22e9"` and `!= "e629642c"` |
| PA13 | `test_image_and_video_engines_agree_when_only_one_policy_is_overridden` | with `face_selection=X` (X = `replace(default, crop_margin=1.4)`), `ImagePipeline(...).model_version == VideoPipeline(...).model_version` and `!= STUB_ENGINE_VERSION` (both default the video policy identically) |
| PA14 (R35) | `test_build_video_analysis_defaults_match_the_shipped_policy` | `inspect.signature(build_video_analysis).parameters["sampling"|"pooling"].default == default_video_selection().sampling_rule | .pooling_rule` |
| PA10 / R33 | `test_selection_identity_of_returns_the_composite_and_is_the_honest_name` | `selection_identity_of(STUB) == selection_identity(...) == "e629642c" != default_face_selection().identity`; `selection_identity_of ≡ face_selection_identity_of`; `is_selection_pinned ≡ is_face_selection_pinned ≡ True`; composite is marker-free |
| R33 | `test_the_f_marker_and_public_readers_document_both_policies` | module `__doc__`, `selection_identity_of.__doc__`, `face_selection_identity_of.__doc__` name the video half; `FACE_SELECTION_MARKER` **source** docstring names it (the constant is a bare `str`, no runtime `.__doc__`); the alias banner contains "never" + "default_face_selection().identity" |
| R33 | `test_model_spec_with_selection_is_the_honest_alias` | `with_selection(fp) ≡ with_face_selection(fp)`; both docstrings mention "batch 23" |
| MG5 | `test_n4_default_is_uniform_in_the_generated_schema` | `schema_out/EvidenceBundle.schema.json` contains `"default": "uniform"` and **not** `"uniform+adaptive"` |
| §1.8 | `test_settings_has_no_skip_version_alignment_field` | `"skip_version_alignment" not in Settings.model_fields` |
| SA10 | `test_startup_alignment.py::test_api_lifespan_refuses_a_marker_bearing_but_unparseable_model_version` | `lifespan` raises `RuntimeError`, message contains the string and "parse" |

---

## 3. Regenerated identity values (old → new)

**None.** The R3 delta adds functions, aliases, docstrings, one comment, tests and
docs, and removes one Settings field — none is an input to any `fingerprint()`.

| Quantity | Value | Moved? |
|---|---|---|
| `default_face_selection().identity` | `251d22e9` | no |
| `default_video_selection().identity` | `db8e6a0d` | no |
| `selection_identity(face, video)` — the `.f.` value | `e629642c` | no |
| model identity | `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c` | no |
| `STUB_ENGINE_VERSION` | `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.a97ccb68` | no |
| OpenAPI structural signature | `fingerprint(structural_signature(build_schema())) == snapshot.fingerprint` | **match** (verified) — `Settings` is not on the wire |
| `schema_out/*.json` | regen to `/tmp` → `diff` vs committed → no diff | no |

---

## 4. Fail-before evidence

Captured before any edit (`python` probe against the pre-R3 tree):

```
selection_identity_of:            ImportError (does not exist)
is_selection_pinned:              ImportError (does not exist)
ModelSpec.with_selection(...):    AttributeError (does not exist)
Settings.skip_version_alignment:  present = True   (design §1.8: remove)
selection_identity docstring:     "injective" absent
```

The 10 new/changed test functions did not exist (or, for PA15's host test, did
not assert `"PERMANENT"`/`"NEVER"`) — they could not run against the pre-R3 tree.

---

## 5. Pass-after evidence

```
=== S8-B23 video selection identity (R29 + N4) ==============
28/28 passed        (was 20/20 — +8: PA12, PA13, PA14, PA10/R33, R33 marker-doc,
                     R33 with_selection, MG5, §1.8 Settings)
=== S8-B23 startup alignment (OR1) ==========================
9/9 passed          (was 8/8 — +1: SA10)
```

Targeted re-run of every R3 test in isolation — all PASS. Independent invariant
spot-checks (`.f.` unchanged, alias equivalence, 4 parser fields, `_MODEL_MARKERS`
unchanged, round-trip, pre-B12 → `None`, `with_selection` delegates, R35 defaults
== policy, Settings field gone, `backend.app` source free of `cv2`/`numpy`/
`inference`) — all PASS.

OpenAPI: `structural_signature` fingerprint matches the committed snapshot.
`python -m contracts.schema` regen → zero diff vs `schema_out/`.

---

## 6. Full test results

```
python tests/run_all.py   →   OK - 51 suites, 1262 tests passed        (exit 0)
```

| | Suites | Tests | Failures |
|---|---|---|---|
| Baseline (pre-R3, isolation) | 51 | 1253 | 0 |
| Baseline (pre-R3, one full `run_all.py`) | 51 | 1253 | **3 suites / 5 tests** — flakes, see §8 |
| **After R3 (full `run_all.py`)** | **51** | **1262** | **0** |

+9 tests = +8 `test_video_selection_identity` + 1 `test_startup_alignment`.
The three suites that flaked in the pre-R3 full run (B18 / B19 / B20) all pass in
the post-R3 full run **and** in isolation (28/28, 17/17, 22/22), re-verified twice.

---

## 7. Deviations from design

The R3 design is executed as written. Four points where a literal instruction
referenced something not present in the tree, adapted to meet the same intent —
none changes behaviour or scope:

1. **MG5 schema target.** §6.2 names `schema_out/VideoAnalysis.schema.json`. That
   file does not exist — `schema_out/` is bundle-organised. MG5 pins
   `schema_out/EvidenceBundle.schema.json`, which carries the nested
   `VideoAnalysis.sampling` node with `"default": "uniform"`. Same guarantee
   (N4 cannot be silently reverted by a `contracts.schema` regen).

2. **CHANGELOG.** §6.1b references "the B23 CHANGELOG entry". The repo has no
   CHANGELOG file. The R34 rollout/rollback guidance and the deploy-runbook line
   were placed in `docs/VIDEO_SELECTION_IDENTITY.md` (new "Rollback" + "Rollout
   (R34)" sections) and the `REPRODUCIBILITY_AUDIT.md` B23 section instead.

3. **`FACE_SELECTION_MARKER` docstring tripwire.** §2.3 suggested asserting
   `FACE_SELECTION_MARKER.__doc__`. `FACE_SELECTION_MARKER` is a bare `str`
   constant, so `.__doc__` returns `str.__doc__`, not the trailing module-level
   string. `test_the_f_marker_and_public_readers_document_both_policies` reads
   the `contracts/version.py` **source** for the marker's docstring block, and
   checks the real `.__doc__` of the module and the two reader functions. Same
   intent; the design itself allowed "or the module docstring".

4. **Collateral call sites kept on the alias.** `test_face_size_policy.py:611`
   and the in-loop assert at `test_face_selection_identity.py:563` still call
   `face_selection_identity_of` (a permanent, exactly-equivalent alias). §3.5
   explicitly says "a pure literal-bump site keeps the old alias to minimise
   churn"; those suites are green via the alias and the B12 suite legitimately
   uses the face-specific name for its pre-B23-era tests. All **new** code — the
   pipelines (`.with_selection`), the new tests, the docs — uses the honest
   names.

**Behavioural / scope deviations: None.**

---

## 8. Unexpected findings

1. **B23 core already implemented.** As recorded in §0. This session's work is
   the R3 delta; the identity move and N4 regen were done by the prior B23
   implementation and are correctly at their B23 values. The "regenerate exactly
   once" process step reduced to verification (nothing to regenerate).

2. **Three flaky suite failures in the pre-R3 full run** — `S8-B18` (`test_the_gate_does_not_read_face_px`),
   `S8-B19` frame_count reproducibility (`test_duration_s_is_derived_from_frame_count`,
   `test_the_amplifier_measurement_is_portable_where_a_real_trigger_is_not`,
   `test_the_trusting_probe_double_matches_probe_video`), `S8-B20`
   (`test_batch_20_did_not_touch_the_frame_count_amplifier_or_validation`). All
   concern `frame_count` / `duration_s` / `probe_video` / the gate metric —
   **none of which B23 (R29/N4/OR1) or the R3 delta touches**. All five pass in
   isolation and in the post-R3 full run. The production code they exercise
   (`contracts/common.py`, `inference/decode_env.py`) has not changed since
   2026-09-08. Root cause: machine contention while `test_evaluation_operating`
   (the 5–15 min CPU-bound suite, flagged in project memory) runs concurrently,
   skewing `cv2` `CAP_PROP_FRAME_COUNT` reads on the synthetic `VideoWriter`
   fixtures those tests assert on. **Not a regression, not attributable to this
   batch.** If a concurrent session is mid-flight on R6/frame-count, its own
   test updates are already consistent (the suites pass) — no collision with B23.

3. **`schema_out/VideoAnalysis.schema.json` and a CHANGELOG do not exist** — see
   §7 items 1–2.

---

## 9. Final verdict

## **IMPLEMENTATION COMPLETE**

- All twelve `B23_R3_DESIGN.md` items implemented as written (four literal-target
  adaptations in §7, zero behavioural deviations).
- `python tests/run_all.py` → **51 suites, 1262 tests, 0 failures, exit 0**.
- No identity moved, no regeneration required, OpenAPI structural signature
  unchanged, no `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` /
  `CALIBRATION_WITNESS_VERSION` / `LOCK_VERSION` bump.
- Non-goals preserved: no parser change (verified — 4 fields, `_MODEL_MARKERS`
  unchanged, round-trips), no R28 / R30 / R31 work, output-identical guarantees
  intact (`test_the_shipped_video_bundle_did_not_move` green).
- R33 and R35 closed in-batch; R32 and R34 entered as tracked; B24 candidates
  (OR1-b, the full R33 rename, R35 required-kwargs, R30, R31) recorded.
