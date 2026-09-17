# B23 (as implemented) — Fresh Independent Adversarial Audit

> **Superseded in part.** A second independent pass
> (`docs/B23_FRESH_INDEPENDENT_AUDIT_2.md`, 2026-09-09) re-confirmed this audit
> and sharpened D1 into D1+D2 (the PA11 tripwire was dropped, not merely a
> non-blocking rec). A **post-audit cleanup batch** then resolved D1, D2, the
> `config.py:53` mismatch (E2 here), the R35/`contracts/video.py` scope gap
> (E1/R36), and strengthened MG5. See that file's `## Cleanup batch resolution`.
> The "Required changes" §G below is **done** except the optional cosmetic items.

**Object:** Batch 23 as it stands in the current working tree — the B23 core plus
the B23-R3 delta (`docs/BATCH_23_R3_DESIGN.md`,
`docs/BATCH_23_R3_IMPLEMENTATION_REPORT.md`).
**Method:** every conclusion re-derived from source and live execution. The
implementation report is treated as an unverified claim set. `git` has no
commits, so "before" is reconstructed from behaviour, era tests and string-
stripped AST fingerprints rather than a diff.

---

## A. Independent recomputation

| Quantity | Recomputed value | Method |
|---|---|---|
| `default_face_selection().identity` | `251d22e9` | live |
| `default_video_selection().identity` | `db8e6a0d` | live; `fingerprint({n_frames:48, min_usable_frames:8, pool_divisor:8, sampling_rule:"uniform", pooling_rule:"topk_mean"})` recomputed by hand → `db8e6a0d` ✓ |
| **`.f.` (composite)** | **`e629642c`** | `fingerprint({"face":"251d22e9","video":"db8e6a0d"})` recomputed by hand → `e629642c` ✓; `selection_identity_of(STUB_ENGINE_VERSION)` → `e629642c` ✓ |
| model identity | `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c` | live, all three of `ImagePipeline`/`VideoPipeline`/`STUB_ENGINE_VERSION` equal ✓ |
| `STUB_ENGINE_VERSION` | `…f.e629642c.d.a97ccb68` | `backend/config.py:41` — **unchanged** ✓ |
| `_MODEL_MARKERS` | `('+c.', '.g.', '.f.', '.d.')` | **unchanged** ✓ |
| `ModelVersionParts` fields | `weights, calibration, gate, face_selection` (4) | **unchanged** — no fifth field, no `.v.` ✓ |
| `FINGERPRINT_LENGTH` / `FINGERPRINT_DECIMALS` | `8` / `6` | unchanged ✓ |
| `parse_model_version` round-trip | `compose() == model_identity_of(STUB)`; `compose_runtime_version(model, decode="deadbeef")` round-trips | ✓ |
| era handling | pre-B12 → `face_selection=None`, `is_selection_pinned=False`; pre-B9 → `decode=None`; no-gate era parses | ✓ |
| parser fn logic (string-stripped AST hash) | `parse_model_version`, `parse_runtime_version`, `model_identity_of`, `fingerprint`, `_canonical`, `quantize`, `is_composed`, `is_environment_pinned` — bodies contain no non-string changes | `compose_model_version` has 2 error-string edits only (string constants); no test asserts version error text (grep) ✓ |
| `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` / `LOCK_VERSION` | `1.0.0` / `1.2.0` / `1.2.0` / `1.1.0` | **unchanged** ✓ |
| OpenAPI structural signature fingerprint | `f9610c4df51cd431eb69991fc4012fe7c03a81382e859ef76aa55d9ae3d0c92a` | matches committed snapshot ✓; `'default'` and `'skip_version'` absent from the signature ✓ |
| `schema_out/*` regeneration | zero diff vs committed | `python -m contracts.schema --out <tmp>` → identical ✓ |
| shipped video bundle `p_fake` | `0.9236678556413206` | live ×3, deterministic; verdict `manipulated`; `sampling="uniform"`, `pooling="topk_mean"` — output-identical for the shipped policy ✓ |

**Identity drift: none. Parser drift: none. Marker/era drift: none. Version-constant
drift: none. OpenAPI structural drift: none.**

---

## B. Verified claims

1. **`.f.` remains `e629642c`; `STUB_ENGINE_VERSION` unchanged; engines aligned.**
   Recomputed from component fingerprints, not read from a pin. (§A)

2. **No parser change.** `_MODEL_MARKERS`, `ModelVersionParts`,
   `RuntimeVersionParts`, `parse_model_version`, `compose_model_version` control
   flow, `model_identity_of`, `parse_runtime_version` all structurally identical.
   The composite is marker-free `[0-9a-f]{8}` so `compose_model_version`'s
   `any(m in face_selection for m in _MODEL_MARKERS)` guard passes and no new era
   is introduced. Round-trips and pre-B9/B12 era parsing verified live.

3. **R33 aliases are canonical + exact.** `selection_identity_of` →
   `parse_model_version(v).face_selection` (real body); `is_selection_pinned` →
   `FACE_SELECTION_MARKER in v` (real body). `face_selection_identity_of` →
   `return selection_identity_of(v)`; `is_face_selection_pinned` →
   `return is_selection_pinned(v)`; `ModelSpec.with_selection` →
   `return self.with_face_selection(fp)`. Equivalence verified over 6 version-string
   shapes (composed / pre-B12 / no-gate / bare / evaluation-style). The
   `face_selection_identity_of` docstring carries the "SINCE BATCH 23 … NEVER
   compare to `default_face_selection().identity`" banner verbatim.

4. **OR1 introduces no forbidden dependency edge.** Reconstructed the import
   graph by AST-walking every `.py` in each package:
   `contracts→∅`, `storage→∅`, `jobqueue→∅`, `inference→{contracts}`,
   `workers→{contracts, inference, jobqueue, storage}`,
   `backend→{contracts, jobqueue, storage, workers}`,
   `evaluation→{contracts, inference}`.
   **`workers` does not import `backend`.** `workers/runner.py→{contracts, jobqueue}`.
   Full DFS cycle check: **none**. `verify_engine_alignment_at_startup` imports
   only `os` (stdlib), `contracts.version`, `workers.engine_registry` (all
   function-local).

5. **Batch-0 boundary intact.** Importing `backend.app` / `backend.config` and
   running the OR1 `lifespan` format check via `TestClient` pulls **none** of
   `cv2` / `numpy` / `torch` / `inference` / `onnxruntime`. `backend/` source has
   zero imports of any of them at any scope.

6. **Env-var hatch matches the design exactly.** Live-tested:
   `DFP_SKIP_VERSION_ALIGNMENT ∈ {true, 1, YES, on, " true "}` → skip + `WARNING`
   every call; `∈ {false, 0}` → gate runs; wrong `DFP_MODEL_VERSION` + no skip →
   `RuntimeError` (fail-closed); real literal → passes; nothing set → fallback
   (engines agree + composed + env-pinned) passes.

7. **`Settings.skip_version_alignment` removal is safe.** `Settings.model_config`
   has `extra='ignore'`; live test — constructing `Settings()` with
   `DFP_SKIP_VERSION_ALIGNMENT` still set in the environment (as the hatch
   instructs) does **not** raise. No test enumerates `Settings.model_fields`.
   `env_prefix='DFP_'` still maps `DFP_MODEL_VERSION → Settings.model_version`.

8. **R35 tripwire genuinely protects against drift.** `test_build_video_analysis_
   defaults_match_the_shipped_policy` reads `inspect.signature(...).parameters[…]
   .default` and compares to `default_video_selection()`. Drift simulation
   (monkeypatch `sampling_rule="uniform-stride"`) → the test raises `AssertionError`.
   Both production call sites (`video_pipeline.py:283`, `:362`) pass explicit
   `sampling=self._video_selection.sampling_rule` / `pooling=…pooling_rule`.

9. **No contract / schema / OpenAPI version movement.** (§A) The `Settings` field
   removal is invisible to the OpenAPI signature (`'skip_version'` absent).

10. **Output-identical for the shipped policy.** `p_fake` bit-identical across 3
    runs and equal to the pinned `SHIPPED_P_FAKE`; `verdict`, `model.version`,
    `video.sampling`, `video.pooling` all as expected.

11. **Collateral pins correct.** `_EXPECTED_EFFECTIVE` (`test_branch_b_semantics.py`,
    `test_headline_and_face_semantics.py`) and `MODEL_ID`
    (`test_decode_identity_completeness.py`) = `…f.e629642c`. `POLICY.identity ==
    "251d22e9"` (`test_crop_geometry.py:578`) unchanged (face policy). `B22_*` vs
    `B23_*` constants in `test_crop_identity.py` correct.

12. **Test count delta = +9** (`test_video_selection_identity` 28 vs 20,
    `test_startup_alignment` 9 vs 8) — matches the report.

13. **Fail-before evidence was genuine.** Re-derived: before the R3 edits,
    `from contracts.version import selection_identity_of` raises `ImportError`;
    `ModelSpec(...).with_selection` raises `AttributeError`;
    `"skip_version_alignment" in Settings.model_fields` was `True`. These are
    real prior states, not staged.

14. **B23 studies run clean** (`video_selection_study`, `crop_identity_study`,
    `face_selection_study`, `frame_count_reproducibility_study`).

---

## C. Partially correct claims

### C1. "R33 documentation complete" (report §1, §7; design §3.6)

Substantially true — `contracts/version.py` module docstring, `.f.` diagram,
`FACE_SELECTION_MARKER` docstring, `ModelVersionParts` docstring,
`compose_model_version` param doc + error strings, both reader-function
docstrings, `inference/models.py` `face_selection_fingerprint` /
`with_selection` / `with_face_selection` docstrings, and five doc-prelude
pointers are all present and B23-aware.

**But three residual face-only descriptions remain**, none flagged by the report:

- **`backend/config.py:53`** — "*Both pipelines now call
  `inference.preprocess.selection_identity` and pass its result to
  `.with_face_selection`*". **Stale after the R3 delta** — the pipelines now call
  `.with_selection(...)` (`video_pipeline.py:129`, `image_pipeline.py:73`). A
  doc/code mismatch **introduced by this batch** and not caught. Functionally
  harmless (alias), factually wrong in a prominent docstring whose whole subject
  is the `.f.` binding.
- **`backend/config.py:89`** — "*The `.f.` segment itself is from S8 Batch 12 and
  **names** the face-selection policy*" — present-tense "names" in the B12
  changelog paragraph. The B23 paragraph 44 lines above corrects it, so a full
  reader is not misled; a reader who stops at the bottom paragraph is.
- **`docs/FACE_SELECTION_IDENTITY.md:260`** — a decisions table row "Dedicated
  face-selection segment in the model identity … shipped". Covered by the doc's
  B23 prelude (`:12–21`), left per the historical-changelog convention.

### C2. R35 tripwire scope (report §2 PA14 docstring; design §5.2)

PA14's docstring says it stops the default "becoming a stale second source of
truth **for the `VideoAnalysis` wire value**". The tripwire holds only
`build_video_analysis`'s kwarg default. The **actual `VideoAnalysis` wire
default** — what a `VideoAnalysis(...)` produces when `sampling`/`pooling` is
omitted — is `contracts/video.py:92,101` `Field(default="uniform" / "topk_mean")`,
which **no test ties to `default_video_selection()`**. Every real server response
passes `sampling`/`pooling` explicitly, so `build_video_analysis`'s default is
*never* the wire value on a real response; `contracts/video.py`'s is (on an
omitted field, and in the OpenAPI doc). PA14 guards the less-authoritative of the
two copies. (See E1.)

### C3. MG5 (`test_n4_default_is_uniform_in_the_generated_schema`)

The assertion `'"default": "uniform"' in schema` is a whole-file substring match
against `EvidenceBundle.schema.json`, not scoped to
`$defs/VideoAnalysis/properties/sampling`. It happens to be unambiguous **today**
(exactly one such default in the bundle, at the right path — verified), but a
future field defaulting to `"uniform"` anywhere would make it vacuous. It also
does **not** check `pooling`. It does prevent a silent revert of *this specific*
`"uniform+adaptive"` → `"uniform"` change; it is not a general default-consistency
guard.

### C4. "Regenerate identities/literals exactly once" (report §0, §3)

Correctly reported as a **no-op** — the R3 delta moves no identity. Verified: `.f.`
unchanged, snapshot fingerprint unchanged, `schema_out` regen zero diff. The
report is accurate that nothing needed regeneration.

---

## D. False claims

### D1. "The `VideoSelectionPolicy` docstring names the [batch_size-neutrality] precondition"

**Asserted in:** `REPRODUCIBILITY_AUDIT.md` R32 row (shipped artifact),
`BATCH_23_R3_DESIGN.md` §6.3 R32 row, and implied by the implementation report's
"R32 … entered as tracked".

**Live inspection of `inference.preprocess.VideoSelectionPolicy.__doc__`:** the
`batch_size` paragraph reads only —

> "`batch_size` is **deliberately excluded** — it is output-neutral. The
> uint8→float32 widening is elementwise and `topk_indices` / `pool_at` run on the
> full `scores` array after every batch, so only progress-event fractions depend
> on it."

This is exactly the **incomplete reasoning** the R2 gate review (§4.1, §6, R32)
called out: it does **not** state the full precondition (`batch_clips` never
pads · backend `score()` is row-independent · `scores` reconstruction preserves
order and values) and does **not** name the re-opening trigger (a static-shape
ONNX/TensorRT backend, or a cross-frame `pooling_rule="attention"` backend). The
gate review's fix for R32 was a **non-blocking recommendation**, so its absence
from the code is not itself a defect — but three shipped documents state as
present fact that the docstring carries content it does not carry. That is a
false claim in the audit trail.

---

## E. New findings

### E1 — R36: the R35 tripwire guards the wrong label copy; `contracts/video.py` defaults and `pooling` are un-tied

**Severity: Low.** **Status: OPEN (candidate).**

The shipped video-selection labels are typed as *values* in three code
locations plus one fixture:

| # | Location | Tied to `default_video_selection()`? |
|---|---|---|
| 1 | `inference/preprocess.py:775-776` — `default_video_selection().sampling_rule` / `.pooling_rule` | — (the authority) |
| 2 | `inference/postprocess.py:649-650` — `build_video_analysis(sampling=…, pooling=…)` kwarg defaults | **yes** — PA14 (R35) |
| 3 | `contracts/video.py:92,101` — `VideoAnalysis.sampling` / `.pooling` `Field(default=…)` | **no** — `sampling` only loosely, via MG5's whole-schema substring; `pooling` not at all |
| 4 | `contracts/fixtures.py:515-516` — synthetic fixture literals | **no** |

Copy #3 is the authoritative wire default (what an omitted field serialises to,
and what the OpenAPI/JSON-schema publishes). The R35 finding's own words — "a
stale second source of truth for the `VideoAnalysis` wire value" — describe #3,
not #2. If `default_video_selection().pooling_rule` is changed to `"attention"`
(the future the field docstring itself anticipates), PA14 catches the #2 drift
but nothing catches #3 or #4: `VideoAnalysis.pooling`'s published default stays
`"topk_mean"`, the OpenAPI doc and any fixture omitting `pooling` go stale — the
**exact N4 defect class**, for pooling, un-guarded. Impact is confined to the
published schema and fixtures (real responses always override), so Low — the same
blast radius N4 had.

**Evidence:** `grep '"topk_mean"' contracts/ inference/` → 4 value sites;
`grep` of `tests/` for any assertion tying `VideoAnalysis.model_fields["pooling"]`
or `contracts/video.py`'s defaults to the policy → none. MG5 checks `sampling`
only, via `'"default": "uniform"' in <entire bundle schema text>`.

### E2 — `backend/config.py:53` doc/code mismatch introduced by the R3 delta

**Severity: Very Low.** The `STUB_ENGINE_VERSION` docstring names `.with_face_selection`
where the pipelines now call `.with_selection`. Introduced by this batch, missed
by the report's own §7 deviations list. Source: `backend/config.py:53` vs
`inference/video_pipeline.py:129` + `inference/image_pipeline.py:73`.

### E3 — `FACE_SELECTION_MARKER` docstring self-tension

**Severity: cosmetic.** `contracts/version.py:144-148` argues "`.s.` was rejected
because … two things called 'selection' in one identity vocabulary is how an
auditor reads the wrong one", while B23-R3 adds `selection_identity_of`,
`is_selection_pinned`, `with_selection`, `selection_identity` to that vocabulary
alongside the pre-existing `CalibrationWitness.selection_witness`. Not a
contradiction (the marker letter is unchanged; the names are unambiguous in
context), but the paragraph now reads oddly.

### E4 — weak / redundant tests in the R3 addition

**Severity: cosmetic (no false green risk demonstrated).**
- `test_the_face_policy_identity_is_unchanged` (PA12) — its load-bearing assert
  (`default_face_selection().identity == "251d22e9"`) already runs at
  `test_video_selection_identity.py:343` inside
  `test_batch_23_moved_only_the_selection_identity`; its second assert
  (`251d22e9 != e629642c`) is trivially true. Redundant.
- The alias-equivalence asserts in
  `test_selection_identity_of_returns_the_composite_and_is_the_honest_name`
  (`selection_identity_of(v) == face_selection_identity_of(v)`) and
  `test_model_spec_with_selection_is_the_honest_alias`
  (`with_selection(fp) == with_face_selection(fp)`) are **tautologies by
  construction** — they verify that a one-line delegating wrapper delegates.
  Harmless, low value.
- `_names_video()` in `test_the_f_marker_and_public_readers_document_both_policies`
  is a substring check: it returns `True` for
  `"this segment has nothing to do with video"`. A careless doc edit that dropped
  the B23 content but left the word "video" elsewhere would pass. Same for PA15's
  `"PERMANENT" in src and "NEVER" in src` on `image_pipeline.py` source.
- `test_selection_identity_of_returns_the_composite…` overlaps ~50% with the
  pre-existing `test_the_selection_identity_is_the_composite`.

### E5 — pre-existing (not R3): OR1 gate docstring names a caller that does not exist

`verify_engine_alignment_at_startup.__doc__`: "*The composition root passes
`get_settings().model_version`*". No such composition root exists;
`run_with_shutdown` calls the gate with no argument. R3 §1.12 names the follow-up
(B24/OR1-b) but did not flag this present-tense docstring imprecision. From the
B23 core, not the R3 delta.

### E6 — baseline flakiness confirmed benign

The pre-R3 full `run_all.py` had 3 failing suites (S8-B18 / B19 / B20, all
`frame_count` / `duration_s` / `probe_video` / gate-metric). Re-verified: all
pass in isolation (28/28, 17/17, 22/22) and the production code they exercise
(`contracts/common.py`, `inference/decode_env.py`) has not changed since
2026-09-08. Contention flakes on `cv2.CAP_PROP_FRAME_COUNT` for synthetic-video
fixtures while `test_evaluation_operating` runs. **Not attributable to B23** —
the report's characterisation is correct.

---

## F. Risk ranking

Five axes (Correctness / Reproducibility / Migration / Cache / Bug-likelihood),
1 low – 5 high.

| Finding | Corr | Repro | Migr | Cache | Bug-lik | Overall |
|---|---|---|---|---|---|---|
| **D1** — R32 docstring claim is false in 3 shipped docs | 1 | 2 | 1 | 0 | 2 | **Low** — audit-trail accuracy defect; the underlying code behaviour is fine |
| **E1 / R36** — R35 tripwire guards `build_video_analysis` not `contracts/video.py`; `pooling` un-tied | 1 | 2 | 1 | 1 | 2 | **Low** — recurs N4-class staleness in schema/fixtures only; real responses override |
| **E2** — `config.py:53` `.with_face_selection` stale | 1 | 1 | 1 | 0 | 1 | **Very Low** — introduced by this batch |
| **E4** — weak/redundant R3 tests | 1 | 1 | 1 | 0 | 2 | **Very Low** — no demonstrated false-green; reduces the value of the R33/R35 tripwire suite |
| **E3 / E5** — docstring tensions | 1 | 1 | 1 | 0 | 1 | **Cosmetic** |
| **E6** — baseline flakes | 1 | 1 | 1 | 0 | 1 | **Not a B23 finding** |

Nothing rises above Low. No correctness, cache-key, identity, parser, or
dependency-graph risk was found.

---

## G. Required changes

All are documentation / test-strengthening; none is architecture-affecting.

1. **D1 — reconcile the R32 claim with reality.** Either (a) extend the
   `VideoSelectionPolicy` `batch_size` paragraph to state the full precondition
   (`batch_clips` no-pad · row-independent `score()` · order/value-preserving
   `scores` reconstruction) and the ONNX/TensorRT/attention re-opening trigger —
   the wording the R2 gate review supplied — or (b) correct the
   `REPRODUCIBILITY_AUDIT.md` R32 row and `BATCH_23_R3_DESIGN.md` §6.3 to say the
   docstring note is a **B24 candidate**, not done. Option (a) is ~3 lines and
   closes the gate review's non-blocking rec too.

2. **E1 / R36 — close or honestly track the remaining label copies.** Either
   extend PA14 to also assert
   `VideoAnalysis.model_fields["sampling"].default == default_video_selection().sampling_rule`
   and the same for `pooling` (and, ideally, `contracts/fixtures.py`), or enter
   **R36** in `REPRODUCIBILITY_AUDIT.md` with the honest note that
   `contracts/video.py` and the fixture literals remain a free-floating copy
   whose blast radius is schema/fixture staleness only (server always overrides),
   and that a labels change to `default_video_selection()` must hand-update them.

3. **E2 — fix `backend/config.py:53`:** `.with_face_selection` → `.with_selection`
   (or "`.with_selection` (alias `.with_face_selection`)").

4. **E4 (optional) — strengthen MG5** to parse the JSON and assert
   `$defs/VideoAnalysis/properties/sampling/default == "uniform"` and
   `…/pooling/default == "topk_mean"`, and drop the tautological alias-equivalence
   asserts or replace them with a behavioural check
   (`selection_identity_of` returns `parse_model_version(v).face_selection` for a
   hand-built multi-era string set).

5. **E3 / E5 (optional)** — soften the `FACE_SELECTION_MARKER` "`.s.` was
   rejected" paragraph; correct the `verify_engine_alignment_at_startup`
   docstring's "The composition root passes…" to future tense referencing
   B24/OR1-b.

---

## H. Verdict

## **APPROVE WITH CHANGES**

The load-bearing properties are all independently verified: `.f.` is `e629642c`
(recomputed, not read), `STUB_ENGINE_VERSION` and the parser are byte-for-byte
unaffected, `_MODEL_MARKERS` / `ModelVersionParts` / era handling / round-trips
are unchanged, no new dependency edge exists (`workers` does not import
`backend`; the graph is acyclic), the Batch-0 boundary holds (the API pulls no
cv2/numpy/inference), the OR1 env hatch behaves exactly as designed, the
`Settings` field removal is safe under `extra='ignore'`, no contract / schema /
witness / OpenAPI version or structure moves, `schema_out` is consistent, and the
shipped video bundle is bit-identical.

The R33 alias strategy is correct — canonical readers with exact permanent
aliases, no parser surface added.

The required changes are one **false claim to reconcile** (D1 — three shipped
documents say the `VideoSelectionPolicy` docstring names the R32 precondition; it
does not), one **incompletely-closed finding** (E1/R36 — R35's tripwire guards
the less-authoritative of two runtime label copies and ignores `pooling`), one
**doc/code mismatch this batch introduced** (E2), and some **weak tripwires**
(E4). None touches the architecture, the identity, the parser, the cache key, or
the dependency graph. With items 1–3 in §G addressed, B23 is sound.

### Live full-suite result at audit time

Independent re-run during this audit (`python tests/run_all.py`):

```
OK - 51 suites, 1262 tests passed
```

Every suite clean (no partial counts, no `FAIL` lines). The three suites that
flaked in the pre-R3 baseline (S8-B18 `28/28`, S8-B19 `17/17`, S8-B20 `22/22`)
all pass here and in isolation — confirming E6. The `libpng error: IDA[00]:
invalid chunk type` line is the expected stderr of a deliberate corrupt-PNG
decode test, not a failure.
