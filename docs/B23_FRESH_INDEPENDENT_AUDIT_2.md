# Batch 23 — Fresh Independent Adversarial Audit (second pass)

**Object:** Batch 23 as it stands in the current working tree — the B23 core plus
the B23-R3 delta (`docs/BATCH_23_R3_DESIGN.md`,
`docs/BATCH_23_R3_IMPLEMENTATION_REPORT.md`).
**Method:** every load-bearing conclusion re-derived from source and live
execution. The implementation report and the prior
`docs/B23_R3_INDEPENDENT_AUDIT.md` are both treated as unverified claim sets.
`git` has no commits; "before" is reconstructed from era tests, the R2 design and
the R2 gate, and behavioural probes.
**Date:** 2026-09-09.

---

## A. Independent recomputation

| Quantity | Recomputed | Method |
|---|---|---|
| `default_face_selection().identity` | `251d22e9` | live |
| `default_video_selection().identity` | `db8e6a0d` | live; `fingerprint({n_frames:48, min_usable_frames:8, pool_divisor:8, sampling_rule:"uniform", pooling_rule:"topk_mean"})` recomputed independently → `db8e6a0d` ✓ |
| **`.f.` composite** | **`e629642c`** | `fingerprint({"face":"251d22e9","video":"db8e6a0d"})` recomputed independently → `e629642c` ✓; `selection_identity(default_face_selection(), default_video_selection())` → `e629642c` ✓; `selection_identity_of(STUB_ENGINE_VERSION)` → `e629642c` ✓ |
| model identity | `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c` | live; `ImagePipeline`, `VideoPipeline` and `STUB_ENGINE_VERSION` all equal ✓ |
| `STUB_ENGINE_VERSION` | `…f.e629642c.d.a97ccb68` | `backend/config.py:41` — **unchanged** ✓ |
| `_MODEL_MARKERS` | `('+c.', '.g.', '.f.', '.d.')` | **unchanged**; no fifth marker ✓ |
| `ModelVersionParts` fields | `weights, calibration, gate, face_selection` (4) | **unchanged**; no `.v.` field ✓ |
| `FINGERPRINT_LENGTH` / `FINGERPRINT_DECIMALS` | `8` / `6` | unchanged ✓ |
| `parse_model_version` | partition-based; strips `.d.` via `model_identity_of`, then `.f.`, then `.g.`; raises on missing `+c.` and on empty `.f.` | round-trip `compose() == model_identity_of(STUB)` ✓; era: pre-B12 → `face_selection=None`, `is_selection_pinned=False` ✓; no-gate era parses ✓; bare `stub-0` → `ValueError` ✓ |
| `is_composed` / `is_environment_pinned` | `True` on STUB, `False` on `stub-0` / on a `.d.`-less string | live ✓ |
| `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` / `LOCK_VERSION` | `1.0.0` / `1.2.0` / `1.2.0` / `1.1.0` | **unchanged** ✓ |
| OpenAPI | `tests/test_openapi.py` 41/41; snapshot fingerprint + signature match; "no contract drift" | ✓ |
| `schema_out/*` regen | `python -m contracts.schema --out <tmp>` → **zero diff** vs committed (12 files) | ✓ |
| `fixtures_out/*` regen | `python -m contracts.fixtures --out <tmp>` → **zero diff**; `video_*.json` `sampling` = `"uniform"`, `model.version` = `"1.0.0-dev"` (synthetic) | ✓ |
| dependency graph (AST walk of every `.py`) | `contracts→∅`, `inference→{contracts}`, `workers→{contracts,inference,jobqueue,storage}`, `backend→{contracts,jobqueue,storage,workers}`, `evaluation→{contracts,inference}` | **`workers` does not import `backend`**; DFS cycle check → **none** ✓ |
| Batch-0 boundary | `import backend.app` pulls **none** of `cv2` / `numpy` / `torch` / `inference` / `onnxruntime` (subprocess `sys.modules` diff) | ✓ |
| full suite | `python tests/run_all.py` → **OK — 51 suites, 1262 tests passed, exit 0** | independent run, this audit; no partial counts, no `FAIL` |

**Identity drift: none. Parser drift: none. Marker / era drift: none.
Version-constant drift: none. OpenAPI / schema / contract structural drift: none.
Startup-alignment: no forbidden edge. Output: byte-identical for the shipped
policy (`test_the_shipped_video_bundle_did_not_move` green; `p_fake` pin holds).**

---

## B. Verified claims (report / prior audit statements that hold)

1. `.f.` = `e629642c`, recomputed from component fingerprints, not read from a pin.
2. No parser change — control flow of `parse_model_version`, `compose_model_version`,
   `model_identity_of`, `parse_runtime_version` unchanged; only 2 error-string
   edits in `compose_model_version` (no test asserts version-error text). The
   composite is marker-free `[0-9a-f]{8}`, so `compose_model_version`'s
   `any(m in fp for m in _MODEL_MARKERS)` guard passes and no new era exists.
3. R33 aliases are canonical + exact: `selection_identity_of` body is
   `return parse_model_version(version).face_selection`; `is_selection_pinned`
   body is `return FACE_SELECTION_MARKER in version`; `face_selection_identity_of`
   → `return selection_identity_of(version)`; `is_face_selection_pinned` →
   `return is_selection_pinned(version)`; `ModelSpec.with_selection` →
   `return self.with_face_selection(...)`. Equivalence verified over composed /
   pre-B12 / no-gate / bare strings. The `face_selection_identity_of` docstring
   carries the "SINCE BATCH 23 … NEVER compare to `default_face_selection().identity`"
   banner verbatim.
4. OR1 worker gate (`workers.runner.verify_engine_alignment_at_startup`) is
   env-only, called from `run_with_shutdown` before the loop, fail-closed;
   `configured_model_version` parameter present (default `None`) so B24/OR1-b is
   pure additive wiring. Imports are function-local (`os`, `contracts.version`,
   `workers.engine_registry`).
5. OR1 API half (`backend/app.py` `lifespan`) is stdlib-only: `is_composed` /
   `is_environment_pinned` / `parse_model_version` round-trip, with an
   `except ValueError` branch. Pulls nothing forbidden.
6. SA10 (`test_api_lifespan_refuses_a_marker_bearing_but_unparseable_model_version`)
   genuinely reaches the `except ValueError` branch: `"stub-0+c.739c3fec.g.ae232cba.f..d.a97ccb68"`
   passes `is_composed` + `is_environment_pinned`, then `parse_model_version`
   raises on the empty `.f.`.
7. `Settings.skip_version_alignment` removed; `Settings.model_config` has
   `extra='ignore'`, so a still-set `$DFP_SKIP_VERSION_ALIGNMENT` does not crash
   `Settings()`. `env_prefix='DFP_'` still maps `DFP_MODEL_VERSION`.
8. R35 tripwire (`test_build_video_analysis_defaults_match_the_shipped_policy`)
   catches drift of `build_video_analysis`'s kwarg defaults against
   `default_video_selection().sampling_rule` / `.pooling_rule`. Both production
   call sites pass explicit policy-sourced values.
9. Collateral pins correct: `_EXPECTED_EFFECTIVE` (`test_branch_b_semantics.py`,
   `test_headline_and_face_semantics.py`) and `MODEL_ID`
   (`test_decode_identity_completeness.py`) = `…f.e629642c`;
   `test_crop_geometry.py:578` `POLICY.identity == "251d22e9"` (face policy,
   unchanged); `test_crop_identity.py` `B22_*` / `B23_*` constants correct;
   `test_evaluation_identity.py:75` `MODEL_ID` (pre-B22, no `.f.`) carries the
   "frozen … do not update on identity moves" comment.
10. Test-count delta = **+9** (`test_video_selection_identity` 28 vs 20,
    `test_startup_alignment` 9 vs 8). Independent run of both suites: 37 passed.
11. Fail-before evidence genuine: before the R3 edits,
    `from contracts.version import selection_identity_of` → `ImportError`;
    `ModelSpec(...).with_selection` → `AttributeError`;
    `"skip_version_alignment" in Settings.model_fields` → `True`.
12. Full suite 51/1262/0 with no flakes on this run (the pre-R3 baseline flakes
    described in report §8 are in `frame_count` / `duration_s` / `probe_video`
    suites B23 does not touch; confirmed benign).

---

## C. Partially correct claims

- **C1 — "R33 documentation complete."** Substantially true (module docstring,
  `.f.` diagram, `FACE_SELECTION_MARKER` docstring, `ModelVersionParts` docstring,
  `compose_model_version` param/error strings, both reader docstrings,
  `inference/models.py` docstrings, five doc-prelude pointers all present and
  B23-aware). **But** `backend/config.py` retains two face-only descriptions the
  report's §7 deviations list does not mention (see D3, E1).

- **C2 — MG5.** `test_n4_default_is_uniform_in_the_generated_schema` asserts
  `'"default": "uniform"' in <whole EvidenceBundle.schema.json text>`. Unambiguous
  *today* (exactly one such default, at
  `$defs/VideoAnalysis/properties/sampling`), but not scoped to that node and
  **does not check `pooling`**. It blocks a silent revert of *this* N4 change; it
  is not a general default-consistency guard. Report §7.1 admits the file-target
  substitution but not the scope weakness.

- **C3 — R35 / PA14 scope.** PA14's docstring says it stops the default becoming a
  stale second source of truth "for the `VideoAnalysis` wire value". It holds only
  `inference/postprocess.build_video_analysis`'s kwarg default. The **authoritative**
  wire default — what `VideoAnalysis(...)` serialises when the field is omitted,
  and what the OpenAPI/JSON-schema publishes — is `contracts/video.py:92,101`
  `Field(default="uniform" / "topk_mean")`, tied to `default_video_selection()`
  by **no** test (`sampling` only loosely via MG5's whole-file substring;
  `pooling` not at all). Every real server response overrides both, so blast
  radius is schema + fixture staleness only. (This is the prior audit's E1/R36 —
  re-confirmed.)

---

## D. False claims

> **Post-audit cleanup batch (2026-09-09): D1, D2, D3 RESOLVED; R36 (E1) and C2
> RESOLVED; see `## Cleanup batch resolution` at the end of this file.**

### D1 — "The `VideoSelectionPolicy` docstring names the [R32 batch_size-neutrality] precondition"

**Asserted in:** `REPRODUCIBILITY_AUDIT.md:128` (R32 summary row, a shipped
artifact) and the `REPRODUCIBILITY_AUDIT.md` B23-R2/R3-findings narrative
(§ "docstring precondition + `test_batch_size_is_output_neutral` tripwire").
(The `BATCH_23_R3_DESIGN.md` §6.3 reference in an earlier draft of this line was
imprecise — R3 §6.3 is the R35 row; the false claim lived only in the shipped
`REPRODUCIBILITY_AUDIT.md`.)

**Live inspection** of `inference/preprocess.py:657-662`
(`VideoSelectionPolicy.__doc__`) and `docs/VIDEO_SELECTION_IDENTITY.md` §2
("`batch_size` is excluded, and that is proven"): both state **only** the
elementwise-widen reason —

> "The uint8→float32 widening is elementwise and `topk_indices` / `pool_at` run
> on the full `scores` array after every batch, so only progress-event fractions
> depend on it."

Neither names the **full precondition** that `BATCH_23_R2_DESIGN.md` §R32
required the docstring to carry: (a) `batch_clips` yields non-overlapping slices
**with no padding**, (b) the backend `score()` is **row-independent**,
(c) `scores` reconstruction is **order/value-preserving** — nor the
**re-opening trigger** (a static-shape ONNX/TensorRT backend, or a cross-frame
`pooling_rule="attention"` backend). The claim is false in two shipped documents.

### D2 — PA11 (the R32 docstring-completeness tripwire) was dropped between R2 and R3 and recorded as shipped

`BATCH_23_R2_DESIGN.md` §5.2 (line 680) specifies **PA11** as a full test-matrix
row: `test_the_batch_size_exclusion_note_is_complete` — "`VideoSelectionPolicy`
docstring names the full `batch_size`-neutrality precondition and the
static-shape/cross-frame trigger". `BATCH_23_R2_DESIGN.md` line 902 names
"the three tripwires (PA10, PA11, SA8) that would fail if the docs drift" as a
core property, and `B23_R2_FINAL_GATE.md` (lines 216, 503-505) endorses
"PA1-PA11 … comprehensive … the two new R32/R33 tripwires" — the gate expected
PA11 to ship.

`BATCH_23_R3_DESIGN.md` §6.2 ("Test-matrix additions") lists only PA12, PA13,
PA14, PA15, SA10, MG5. **PA11 appears nowhere in the R3 design, the
implementation report, or the test tree** (`grep -rn "PA11\|exclusion_note" tests/ docs/BATCH_23_R3_*`
→ nothing). R3 §0 states "Everything in R2 that the gate review marked *sound*
carries forward unchanged." The implementation report §7 "Deviations from design"
does not mention it.

Net effect: the R32 docstring requirement now has **neither the docstring text
(D1) nor the tripwire**, while `REPRODUCIBILITY_AUDIT.md` records the docstring
note as done and `test_batch_size_is_output_neutral` (a *behavioural* neutrality
check, not a docstring check) is offered as "the tripwire". The prior
`B23_R3_INDEPENDENT_AUDIT.md` found D1 but characterised PA11's absence as "a
non-blocking recommendation, so its absence … is not itself a defect" — that
understates it: R2 made PA11 a required matrix row and the gate relied on it.

**Severity: Low** — audit-trail accuracy + a dropped doc-drift guard. The
underlying code behaviour (R32) is not a defect today: `batch_size` *is*
output-neutral on the stub path, held by `test_batch_size_is_output_neutral`.

### D3 — `backend/config.py` B23 paragraph says the pipelines "pass its result to `.with_face_selection`"

`backend/config.py` (`STUB_ENGINE_VERSION` docstring, B23 paragraph, ~line 52):

> "Both pipelines now call `inference.preprocess.selection_identity` and pass its
> result to `.with_face_selection`"

The R3 delta switched both pipelines to `.with_selection(...)`
(`inference/video_pipeline.py:129`, `inference/image_pipeline.py:75`). A
doc/code mismatch **introduced by this batch** in a docstring whose whole subject
is the `.f.` binding, missed by the report's §7. Functionally harmless (exact
alias). (Prior audit E2 — re-confirmed; the B12 paragraph ~line 89 "names the
face-selection policy" present-tense is the softer C1 residual.)

**Severity: Very Low.**

---

## E. New findings (beyond `B23_R3_INDEPENDENT_AUDIT.md`)

### E1 — R36 (prior audit) still open, and `pooling` has *no* guard at any layer

Re-confirmed and sharpened: of the four label-carrying sites —
`inference/preprocess.py` (authority), `inference/postprocess.py` kwarg default
(PA14), `contracts/video.py` `Field(default=)`, `contracts/fixtures.py` literal —
only #2 is tied to the policy. If `default_video_selection().pooling_rule` moves
to `"attention"`, PA14 fails on #2, and `test_the_pooling_rule_and_divisor_match_the_pipeline:271`
(`assert SHIPPED.pooling_rule == "topk_mean"`) fails, so a developer *is*
alerted — but nothing updates or fails on the **published schema default** or the
**fixtures**, which is exactly the N4 defect class, for pooling. Track as **R36**
or extend PA14 to `VideoAnalysis.model_fields["sampling"|"pooling"].default`.

### E2 — implementation report §7 "Deviations" omits the SA10 test-string change

Design §6.2 SA10 gives the example string `"stub-0+c."`. That string fails
`is_environment_pinned` (no `.d.`) and **never reaches** the `except ValueError`
branch the design says it tests. The implementer correctly substituted
`"stub-0+c.739c3fec.g.ae232cba.f..d.a97ccb68"` (which does reach it). The
substitution is an **improvement**, but it is an undocumented deviation from the
design's literal instruction — the §7 list should carry it alongside the MG5 and
CHANGELOG adaptations. **Very Low / informational.**

### E3 — `is_selection_pinned` docstring deviates from the design-mandated wording

Design §3.1 mandates the docstring "The honest-named analogue of
`is_face_selection_pinned`." The shipped docstring instead reads "The exact
analogue of `is_environment_pinned` and `is_composed`." Harmless; the tripwire
`test_the_f_marker_and_public_readers_document_both_policies` does not check this
function's docstring. **Cosmetic.**

### E4 — environment noise in the full run

`libopenh264 … error -22 / Failed to initialize VideoWriter` and
`libpng error: IDA[00]: invalid chunk type` appear on stderr during the full
suite. Pre-existing OpenCV/codec-environment artefacts (the libpng line is a
deliberate corrupt-PNG decode test); every dependent suite passes. **Not a B23
finding.**

### E5 — weak / redundant R3 tests (prior audit E4 — re-confirmed)

- `test_the_face_policy_identity_is_unchanged` (PA12) duplicates the assert at
  `test_video_selection_identity.py:343`.
- The alias-equivalence asserts
  (`selection_identity_of(v) == face_selection_identity_of(v)`,
  `with_selection(fp) == with_face_selection(fp)`) are tautologies-by-construction
  — a one-line delegating wrapper delegates.
- `_names_video()` is a substring test: it returns `True` for
  `"nothing to do with video"`; same for PA15's `"PERMANENT" in src and "NEVER" in src`.
  A doc edit that dropped the B23 content but left the word "video"/"NEVER"
  elsewhere would pass.

None demonstrates a false green; they reduce the value of the R33/R35 tripwire
suite. **Very Low.**

### E6 — pre-existing OR1 docstring imprecision (prior audit E5 — re-confirmed)

`verify_engine_alignment_at_startup.__doc__`: "The composition root passes
`get_settings().model_version`" — present tense; no such composition root exists
until B24/OR1-b. From the B23 core, not the R3 delta. **Cosmetic.**

---

## F. Risk ranking

Axes: Correctness / Reproducibility / Migration / Cache / Bug-likelihood (1 low–5 high).

| Finding | Corr | Repro | Migr | Cache | Bug-lik | Overall |
|---|---|---|---|---|---|---|
| **D1 + D2** — R32 docstring claim false in 2 shipped docs; PA11 dropped and unrecorded | 1 | 2 | 1 | 0 | 2 | **Low** — audit-trail + a dropped doc-drift guard; R32 behaviour is fine |
| **C3 / E1 (R36)** — R35 tripwire guards `build_video_analysis`, not `contracts/video.py`; `pooling` un-tied in schema + fixtures | 1 | 2 | 1 | 1 | 2 | **Low** — N4-class staleness, schema/fixtures only |
| **C2** — MG5 whole-file substring, `pooling` unchecked | 1 | 1 | 1 | 0 | 2 | **Very Low** |
| **D3 / E5** — `config.py` `.with_face_selection` stale; weak R3 tests | 1 | 1 | 1 | 0 | 1 | **Very Low** — introduced/left by this batch |
| **E2 / E3 / E6 / E4** — deviation-list gap, docstring wording, env noise | 1 | 1 | 1 | 0 | 1 | **Cosmetic / informational** |

Nothing above **Low**. No correctness, cache-key, identity, parser, era, or
dependency-graph risk found.

---

## G. Required changes

All are documentation / test-strengthening; none is architecture-affecting.

1. **D1 + D2 — reconcile R32.** Either
   (a) extend the `VideoSelectionPolicy` docstring **and**
   `docs/VIDEO_SELECTION_IDENTITY.md` §2 with the three-part precondition
   (`batch_clips` no-pad · row-independent `score()` · order/value-preserving
   `scores` reconstruction) and the ONNX/TensorRT/attention re-opening trigger —
   the wording in `BATCH_23_R2_DESIGN.md` §R32 — **and restore PA11**
   (`test_the_batch_size_exclusion_note_is_complete`, a string check on
   `VideoSelectionPolicy.__doc__`); or
   (b) correct `REPRODUCIBILITY_AUDIT.md:128` and `BATCH_23_R3_DESIGN.md` §6.3
   to state the docstring note **and PA11 are a B24 candidate, not shipped**.
   Option (a) is ~4 lines of prose + one 3-line test.

2. **C3 / E1 (R36) — close or track the remaining label copies.** Either extend
   PA14 to also assert
   `VideoAnalysis.model_fields["sampling"].default == default_video_selection().sampling_rule`
   and the same for `pooling` (and ideally `contracts/fixtures.py`), or enter
   **R36** in `REPRODUCIBILITY_AUDIT.md` with the honest note that
   `contracts/video.py` + the fixture literals are a free-floating copy (server
   always overrides; blast radius = schema/fixture staleness) that a labels
   change must hand-update.

3. **D3 — fix `backend/config.py`:** in the B23 paragraph, `.with_face_selection`
   → `.with_selection` (or "`.with_selection` (alias `.with_face_selection`)").

4. *(optional)* **C2 / E5 — strengthen MG5** to parse the JSON and assert
   `$defs/VideoAnalysis/properties/{sampling,pooling}/default`; drop the
   tautological alias-equivalence asserts or replace them with a multi-era
   behavioural check on `selection_identity_of`.

5. *(optional)* **E2 / E3 / E6** — add the SA10 string change to the report's §7
   deviations list; align the `is_selection_pinned` docstring with the design
   wording; make `verify_engine_alignment_at_startup`'s "composition root passes…"
   future-tense referencing B24/OR1-b.

---

## H. Verdict

## **APPROVE WITH CHANGES**

Every load-bearing property is independently verified: `.f.` is `e629642c`
(recomputed from `fingerprint({"face":"251d22e9","video":"db8e6a0d"})`, not read
from a pin), `STUB_ENGINE_VERSION` and the parser are byte-for-byte unaffected,
`_MODEL_MARKERS` / `ModelVersionParts` / era handling / round-trips are unchanged,
no new dependency edge exists (`workers` does not import `backend`; the graph is
acyclic), the Batch-0 boundary holds (`import backend.app` pulls no
cv2/numpy/inference), the OR1 env hatch and the fail-closed worker/API split
behave as designed, the `Settings.skip_version_alignment` removal is safe under
`extra='ignore'`, no contract / schema / witness / OpenAPI version or structure
moves, `schema_out` and `fixtures_out` regenerate with zero diff, and the shipped
video bundle is bit-identical. Full suite 51/1262/0, no flakes.

The R33 alias strategy is correct — canonical readers with exact permanent
aliases, no parser surface added.

The required changes are **one false claim compounded by a silently dropped
tripwire** (D1 + D2 — two shipped documents say the `VideoSelectionPolicy`
docstring names the R32 precondition; it does not, and PA11, the R2/gate tripwire
that would have enforced it, was removed in the R2→R3 transition with no
rationale and no deviation note), **one incompletely-closed finding** (C3 / R36 —
R35's tripwire guards the less-authoritative of the runtime label copies and
ignores `pooling` at the schema and fixture layers), and **one doc/code mismatch
this batch introduced** (D3). None touches the architecture, the identity, the
parser, the cache key, or the dependency graph. With §G items 1–3 addressed,
B23 is sound.

---

## Cleanup batch resolution (2026-09-09)

A post-audit cleanup batch implemented §G items 1–4. **No identity /
`STUB_ENGINE_VERSION` / parser / schema / OpenAPI / version-constant movement**
(re-verified: `.f.` = `e629642c`, `_MODEL_MARKERS` = `('+c.','.g.','.f.','.d.')`,
4 parser fields, `schema_out` + `fixtures_out` regen zero diff, `test_openapi.py`
41/41). Full suite: **51 suites / 1264 tests / 0 failures** (+2 vs the audit's
1262: PA11 restored, R36 new; MG5 rewritten in place).

| Audit finding | Resolution |
|---|---|
| **D1 + D2** | Path (a). `VideoSelectionPolicy.__doc__` and `docs/VIDEO_SELECTION_IDENTITY.md` §2 extended with the full R32 precondition — (a) `batch_clips` no-pad, (b) row-independent `score()`, (c) order/value-preserving `scores` reconstruction — and the ONNX/TensorRT / `pooling_rule="attention"` re-opening trigger. **PA11 restored** as `test_the_batch_size_exclusion_note_is_complete`. `REPRODUCIBILITY_AUDIT.md` R32 row + narrative corrected (the claim is now true and tripwired). |
| **D3** | `backend/config.py` B23 paragraph: `.with_face_selection` → `.with_selection` (permanent alias `.with_face_selection`). |
| **R36 (E1)** | **Not tied at source** — `contracts` is a leaf package (`contracts → ∅`); `contracts/video.py` importing `inference.preprocess.default_video_selection` would create a `contracts ↔ inference` cycle the DAG test forbids. Closed instead by tripwire `test_the_video_analysis_wire_defaults_match_the_shipped_policy`: holds `VideoAnalysis.model_fields["sampling"|"pooling"].default` **and** the three `fixtures_out/video_*.json` values equal `default_video_selection()`. Entered as **R36 CLOSED** in `REPRODUCIBILITY_AUDIT.md` with the leaf-package rationale. |
| **C2 (MG5)** | Rewritten from `'"default": "uniform"' in <whole file>` to a parsed-JSON scoped check of `$defs/VideoAnalysis/properties/{sampling,pooling}/default`, both tied to `default_video_selection()`. Now also guards `pooling`. |
| **E5** (`_names_video`) | Tightened from `"video" in low or "frame selection" in low` to `"frame selection" in low and "pooling" in low` — the specific B23 concept, not the bare word. |
| **E2 / E3 / E6** (optional, cosmetic) | Not changed — informational only (SA10 string is an *improvement* over the design; `is_selection_pinned` / `verify_engine_alignment_at_startup` docstrings are harmless). Left for a future pass. |
