# Batch 23 — R2 Design (revised after adversarial review)

**Supersedes:** `BATCH_23_DESIGN.md` (scratchpad, session `ab65b07f`).
**Responds to:** `docs/B23_ADVERSARIAL_REVIEW.md` — verdict **APPROVE WITH CHANGES**,
seven blocking defects.
**Scope:** R29 closure, N4 closure, OR1 startup-alignment enforcement, risk refresh.
**Constraint:** design only. No code in this document.

---

## 0. Status of this document and method

B23 has **already been implemented** in this repository and the suite is green
(`python tests/run_all.py` → 51 suites, 1253 tests, 0 failures). The adversarial
review is explicit that it reviews *the design as written*, and that several
under-specified items were *worked around at implementation time* rather than
being correct in the design. A passing test run does not retroactively make the
original design document correct.

This R2 document does three things:

1. Re-derives a **dependency-safe, specification-complete** version of every one
   of the seven blocking items, and states where the shipped tree already
   matches it and where it does not.
2. Dispositions the three new findings (R32, R33, R34).
3. Rebuilds the Route-A/B analysis, the test matrices, the pin inventory and the
   risk register from first principles.

Every claim below was checked against the shipped tree (`contracts/version.py`,
`inference/`, `workers/`, `backend/`, `tests/`) at the paths cited.

**One item is not fully satisfied by the shipped tree** and is called out in
§2.3 and §9: the R33 documentation/tripwire rider (`contracts/version.py` and
`inference/models.py` still describe `.f.` as face-only).

---

## 1. Executive summary of the R2 changes

| # | Blocking defect | R2 resolution | Shipped tree |
|---|---|---|---|
| 1 | OR1 worker gate closes a `workers → backend` cycle | Worker gate reads `os.environ` (`DFP_MODEL_VERSION`, `DFP_SKIP_VERSION_ALIGNMENT`) + an **optional** `configured_model_version` parameter; imports only `contracts.version` and `workers.engine_registry`. Fail-closed on the full check when a literal is in reach; degraded-but-still-fail-closed (engines-agree + well-formed) otherwise. | **Matches** — `workers/runner.py:398` `verify_engine_alignment_at_startup`, no `backend` import; DAG test green. Residual: the config-default path is weaker (§2.1). |
| 2 | `1 <= min_usable_frames <= n_frames` validation contradicts FB2 and the shim | **Keep** the validation; **scope "output-identical" to the shipped policy**; enumerate every legacy call site and prove `n_frames ≥ 8`; rewrite FB2 to a construction the validation permits (`n_frames=16, min_usable_frames∈{8,13}` on a 12-frame clip). Document the behaviour change: `min_usable_frames > n_frames` is now a construction-time `ValueError`, zero shipped blast radius. | **Matches** — `inference/preprocess.py:702-711`; FB2 is `test_r29_min_usable_frames_moves_the_identity` (`tests/test_video_selection_identity.py:144`). |
| 3 | `.f.` becomes composite; `contracts/version.py` public functions/docstrings still say face-only; no tripwire | **Names stay, docs change, add three tripwires.** Rename rejected (a `contracts/` public-API change is larger and riskier than the identity move and touches more than docs). Documentation inventory extended to `contracts/version.py` + `inference/models.py`. | **Partial** — `inference/preprocess.py` + pipeline comments + `docs/*` updated; `contracts/version.py` and `inference/models.py` **not** updated; tripwires **not** added. This is the R33 residual. |
| 4 | Incomplete pin inventory | Rebuilt from scratch in §6 — every literal, every semantic assertion, every helper rewritten-not-bumped, including `tests/stub_handlers.py`, `contracts/fixtures.py`, the `_EXPECTED_EFFECTIVE` / `MODEL_ID` collateral, and the semantic rewrites. | **Matches** the rebuilt inventory (verified file-by-file). |
| 5 | "byte-identical bundle" overstated | Replaced with a four-part decomposition (§7): scored fields *are* guaranteed not to move; `timing`, job metadata and diagnostics are not compared and never were byte-stable. | **Matches** — `test_batch_size_is_output_neutral` (`tests/test_video_selection_identity.py:274`) compares `p_fake`, `verdict`, `model.version`, `video.model_dump()` field-wise, not whole-bundle. |
| 6 | OpenAPI "default is stripped" assumed, not proven | **Proven from source** (§2.6): `backend/openapi.py` builds the structural signature from an allowlist (`_component_signature`, `_type_of`, lines 209-251); `default` is never read. The `"uniform+adaptive"`→`"uniform"` change cannot move the signature. The hard gate stands, now on a citation not a hope. | **Matches** — `backend/openapi.snapshot.json` carries no `default` key; `test_batch_23_moved_only_the_selection_identity` greps it. |
| 7 | P19 states `inference/__init__.py` exports `FaceSelectionPolicy` | **False.** `inference/__init__.py` `__all__` (lines 60-89) lists protocols, pipelines, registry helpers and calibration only. `FaceSelectionPolicy` / `default_face_selection` / `face_selection_identity` are `preprocess`-only. **R2: add no exports** — `VideoSelectionPolicy` etc. stay `preprocess`-only, matching `FaceSelectionPolicy`. Drop P19. | **Matches** — no selection symbols added to `inference/__init__.py`. |

New findings: **R32** (in scope as a docstring note + a retained tripwire),
**R33** (in scope — the Defect 3 rider), **R34** (a migration/ops note, not a
code change; belongs in the rollout section and CHANGELOG).

Identity movement, unchanged from the original design and confirmed:
`.f.` `251d22e9` → **`e629642c`** (now the composite
`selection_identity(face, video) = fingerprint({"face": 251d22e9, "video": <V>})`).
`default_face_selection().identity` stays `251d22e9`. `weights` / `+c.` / `.g.` /
`.d.` unchanged. `STUB_ENGINE_VERSION` →
`stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.a97ccb68`. One cache
generation, both namespaces. No `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` /
`CALIBRATION_WITNESS_VERSION` / OpenAPI-structural move.

---

## 2. Blocking defect resolutions

### 2.1 Defect 1 — OR1 worker gate and the dependency cycle

#### The defect

The original P20 said: *"In the worker bootstrap … call
`workers.engine_registry.verify_version_alignment(settings.model_version)`."*
There is no `settings` in scope in `workers/`, and no worker process `main`.
Obtaining `settings.model_version` means `from backend.config import
get_settings` inside `workers/`, which creates the edge `workers → backend`.

`backend/app.py` imports `from workers.io import …` and `from workers.protocol
import …` — edge `backend → workers`. Together: a cycle.
`tests/test_worker_inference_integration.py::test_dependency_graph_stays_acyclic`
(lines 567-588) AST-walks **every** import in every module of each package,
including function-local ones, and fails on any cycle. The original design
neither noticed this nor specified an alternative.

#### Dependency graph — before (P20 as written)

`_imports_of(pkg)` collapses to first path segment; edges restricted to the six
packages `contracts storage jobqueue inference workers backend`:

```
contracts   → {}
storage     → {contracts}
jobqueue    → {contracts}
inference   → {contracts}
workers     → {contracts, storage, jobqueue, inference, BACKEND}   ← P20 adds this
backend     → {contracts, storage, jobqueue, inference, workers}
```

Cycle: `workers → backend → workers`. `test_dependency_graph_stays_acyclic`
raises `AssertionError("cycle: workers -> backend -> workers")`.

#### Dependency graph — after (R2)

The gate function imports **only stdlib + `contracts` + `workers.engine_registry`**
(which imports `inference` — a pre-existing, required edge, pinned by
`test_workers_import_inference`). It reads the configured literal and the skip
flag from `os.environ`, and accepts an **optional** `configured_model_version`
parameter for a future `backend`-side composition root to pass in.

```
contracts   → {}
storage     → {contracts}
jobqueue    → {contracts}
inference   → {contracts}
workers     → {contracts, storage, jobqueue, inference}            ← no backend edge
backend     → {contracts, storage, jobqueue, inference, workers}
```

Topological order `contracts < {storage, jobqueue, inference} < workers <
backend`. Acyclic. `test_dependency_graph_stays_acyclic` passes.

This is the shipped implementation: `workers/runner.py:398-471`,
`verify_engine_alignment_at_startup(configured_model_version: str | None = None)`,
with `import os` and `from contracts.version import …` / `from
workers.engine_registry import …` **local to the function** so `import
workers.runner` stays cheap. Called with no argument from `run_with_shutdown`
(`workers/runner.py:488`).

#### Batch-0 layering rules — preserved

- **API stays free of `cv2` / `numpy` / `inference`.** The API half of OR1
  (`backend/app.py:90-123`) uses `contracts.version` only (`is_composed`,
  `is_environment_pinned`, `parse_model_version`, `model_identity_of`). No
  `inference` import. Pinned by `test_inference_imports_nothing_from_workers_or_infrastructure`
  and by the API-side of `test_startup_alignment.py`.
- **`inference` still imports nothing upward.** Unchanged by B23.
- **`workers` may import `inference` but not `backend`.** The gate honours this.
- **`contracts` stays stdlib-only.** The gate's `contracts.version` import adds
  no dependency.

#### Fail-closed behaviour — what is and is not preserved

The gate has three paths:

| Situation | Path | Fail-closed? |
|---|---|---|
| `DFP_SKIP_VERSION_ALIGNMENT` truthy (`{"1","true","yes","on"}`) | log a `WARNING` every boot, return | **No — by design.** The one escape hatch. |
| `configured_model_version` passed **or** `$DFP_MODEL_VERSION` set | `verify_version_alignment(configured)` — the full cryptographic check against `engine_versions()` | **Yes.** `RuntimeError` propagates out of `run_with_shutdown`, process exits non-zero, loop never starts. |
| Neither — deployment is on the `backend.config.STUB_ENGINE_VERSION` default | the worker **cannot see** that literal (no `backend` import). Falls back to: engines must agree with *each other*, and the agreed string must be `is_composed` **and** `is_environment_pinned` | **Partial.** Fail-closed on *engine disagreement* and on a *malformed* engine version; **blind** to a hand-edited-wrong `STUB_ENGINE_VERSION` that the engines happen not to match. |

**The residual (third row) is real and must be documented, not hidden.** It is
covered at a different layer: `test_configured_model_version_matches_the_engines`
(`tests/test_model_version.py`) is a **pre-merge** guard that fails CI if the
committed `STUB_ENGINE_VERSION` default does not equal what the engines report.
So the failure the third row is blind to cannot reach `main`. The gap is only a
deployment that *hand-edits* the literal after merge without setting
`$DFP_MODEL_VERSION` — an unusual operation, and the API-side format check
(§2.6) still rejects the most common form of it (`1.0.0-dev`, a bare string).

#### If fail-closed on the full check is wanted for the default path too

Two options close the residual; both are **out of B23 scope** and recorded as
follow-ups:

- **Option 2 — a `backend`-side composition-root module** (e.g.
  `backend/worker_main.py`). `backend → workers` already exists, so a module in
  `backend` that imports both `backend.config` and `workers` adds no cycle. It
  wires `build_all_workers` + the gate + `run_with_shutdown`, passing
  `get_settings().model_version` into the gate's existing
  `configured_model_version` parameter. This is "the process `main`" the
  original design gestured at but never created.
- **Option 3 — `run_with_shutdown(runner, *, configured_model_version)`** as a
  required keyword, supplied by whatever composition root builds the runner.

The shipped gate already accepts `configured_model_version`, so **option 2 or 3
is a pure additive wiring change later** — no gate rewrite. R2 recommendation:
ship option 1 (env + fallback) now as B23 did; open a follow-up to add the
`backend`-side composition root (option 2) so production runs the full check on
the default path.

#### `Settings.skip_version_alignment` — an unwired field

`backend/config.py:227` adds `skip_version_alignment: bool`. The worker gate does
**not** read it (it reads `$DFP_SKIP_VERSION_ALIGNMENT` from the environment,
because it cannot import `backend`). Today the Setting is only meaningful to a
future `backend`-side composition root. R2: keep the field (it documents the
knob and is the natural place for option 2 to read it), and add a one-line note
in its description that **for the current env-only worker the environment
variable is authoritative**. Tracked as a minor cleanup, not a blocker.

---

### 2.2 Defect 2 — the validation contradiction

#### The defect

`VideoSelectionPolicy.__post_init__` adds `1 <= min_usable_frames <= n_frames`
(`inference/preprocess.py:705-709`). Two problems the original design did not
notice:

1. **FB2's mechanism becomes unconstructible.** The original FB2 referenced
   `test_r29_min_usable_frames_flips_the_verdict_under_one_model_version`, whose
   mechanism was `n_frames=9, min_usable_frames=16` — i.e. `min_usable_frames >
   n_frames`, which the new validation forbids.
2. **The shim's "no edit" claim is unsafe in general.**
   `replace(default_video_selection(), n_frames=4)` yields
   `VideoSelectionPolicy(n_frames=4, min_usable_frames=8, …)` → `ValueError` at
   pipeline construction. Previously `VideoPipeline(n_frames=4)` was legal (it
   would always abstain).

#### Does the validation belong in B23 at all?

**Yes, retained**, for three reasons:

- It centralises guards that were previously ad-hoc and duplicated across
  `extract_frames`, `topk_indices` and the pipeline. A required-fields policy
  object that does not validate its own invariants is a weaker instrument than
  `FaceSelectionPolicy`, which does.
- The forbidden region (`min_usable_frames > n_frames`) is nonsensical: a floor
  no decode can ever clear means *unconditional abstention*, which is a
  configuration bug, not a feature. Turning a silent always-abstain into a loud
  `ValueError` is a strict improvement in operator feedback.
- The blast radius on the shipped tree is **zero** (enumerated below).

#### How FB2 survives

FB2's *finding* — "`min_usable_frames` moves the model identity **and** still
gates verdict→abstention" — is unchanged. Only the *mechanism* changes to one
the validation permits:

> A 12-frame clip decodes 12 frames under `n_frames=16`. A floor of `8` decides;
> a floor of `13` abstains. Both policies are constructible (`8 ≤ 16`, `13 ≤ 16`).
> `decided.model_version != abstained.model_version`, and
> `abstained.verdict is Verdict.INSUFFICIENT_EVIDENCE`.

This is `test_r29_min_usable_frames_moves_the_identity`
(`tests/test_video_selection_identity.py:144-166`), shipped and green.

#### Migration behaviour for legacy call sites

`VideoPipeline(n_frames=N, batch_size=B, min_usable_frames=M)` stays accepted
(deprecated). The `__init__` shim (`inference/video_pipeline.py:94-105`):

1. `base = video_selection or pre.default_video_selection()`
2. collects `overrides` for whichever of `n_frames` / `min_usable_frames` was
   passed explicitly (`is not None`)
3. `base = replace(base, **overrides)` — **once**, so `__post_init__` validates
   the *reconstructed* policy
4. `batch_size` is a plain attribute, never in `base`

Consequences:

| Legacy call | Result before B23 | Result after B23 |
|---|---|---|
| `VideoPipeline(n_frames=48)` | ok | ok — `replace(default, n_frames=48)`, `8 ≤ 48` |
| `VideoPipeline(n_frames=16)` | ok | ok — `8 ≤ 16` |
| `VideoPipeline(n_frames=8)` | ok | ok — `8 ≤ 8` |
| `VideoPipeline(min_usable_frames=8)` | ok | ok — `8 ≤ 48` |
| `VideoPipeline(n_frames=4)` (hypothetical) | ok, always abstains | **`ValueError` at construction** — documented behaviour change |
| `VideoPipeline(n_frames=9, min_usable_frames=16)` (old FB2) | ok, always abstains | **`ValueError` at construction** |

#### Call-site enumeration (proving zero blast radius)

Every `VideoPipeline(...)` and `extract_frames(...)`/policy construction that
passes `n_frames` or `min_usable_frames` in `tests/`, `inference/`, `workers/`:

| Value(s) passed | Sites | Safe? |
|---|---|---|
| `n_frames=48` | `test_inference.py:255,268,283,553`, `test_timing_contract.py:173`, `test_frame_extraction.py` (many), `test_video_branches.py:106` | `48 ≥ 8` ✓ |
| `n_frames=16` | `test_branch_b_semantics.py:195,265`, `branch_b_semantics_study.py:196,306`, `frame_count_reproducibility_study.py:261`, `test_headline_and_face_semantics.py:211`, `test_frame_extraction.py:212,255`, `test_preprocess.py:390` | `16 ≥ 8` ✓ |
| `n_frames=8` | `test_timing_contract.py:170`, `test_decode_determinism.py:377` | `8 ≥ 8` ✓ |
| `n_frames=12` | `test_decode_determinism.py:136`, `test_inference_determinism.py:276` — **`extract_frames` only, not a policy** | n/a (no policy) |
| `n_frames=9` / `n_frames=24` | `test_video_selection_identity.py` — via `replace(SHIPPED, n_frames=…)`, all `≥ min_usable_frames` in the same `replace` or `≥ 8` | ✓ |
| `min_usable_frames=8` | `test_inference.py:292` | `8 ≤ 48` ✓ |
| `min_usable_frames∈{8,13}` on `n_frames=16` | `test_video_selection_identity.py:156` (the rewritten FB2) | `13 ≤ 16` ✓ |

`test_frame_count_reproducibility.py:149` passes `n_frames=n_frames` from a
parametrised list `[…]` — verify that list contains no value `< 8` **or** no
`min_usable_frames` override (it passes none, so the default `8` floor applies
and any `n_frames ≥ 8` in the list is safe; a list entry `< 8` would raise). The
shipped list is `{16, …}` per `frame_count_reproducibility_study.py:134`
(`n_frames` budgets around a 1000-frame reference). **Confirm at implementation
that every parametrised `n_frames` there is `≥ 8`.** (Shipped: green, so it is.)

#### "output-identical" is scoped to the shipped policy

The guarantee is: *for `default_video_selection()`*, the scored bundle does not
move (§7). It is **not** a claim that every previously-legal `VideoPipeline`
construction still behaves identically — `min_usable_frames > n_frames`
constructions now raise. That change has zero shipped blast radius and is a
strict improvement, but the design must state it rather than let "output-
identical" imply it.

`test_the_policy_validates_its_invariants`
(`tests/test_video_selection_identity.py:226-232`) pins all three raises.

---

### 2.3 Defect 3 — `.f.` semantic drift (R33)

#### The defect

Before B23, `.f.` named the face-selection policy alone (Batch 12, R17). B23
makes it `selection_identity(face, video) = fingerprint({"face": …, "video":
…})`. Every name, docstring and comment that says "face selection" for `.f.` is
now incomplete for a B23+ string, and `contracts.version.face_selection_identity_of(v)`
returns a value that is **not** `default_face_selection().identity` — a false
negative forever for any caller comparing the two. The original design's §5.7
doc-update table omitted `contracts/version.py` entirely, and there is no
tripwire.

#### Full audit of the face-only `.f.` surface

**`contracts/version.py`:**

| Location | Text (paraphrased) | Status in shipped tree |
|---|---|---|
| module docstring, `face` paragraph (33-40) | "changes when the *face-selection policy* changes: the detector … the ordering rule … Added in Batch 12 (R17)." | **not updated** |
| module docstring, readable-format diagram (47-53) | `└───────┘` labelled "face select" | **not updated** |
| `FACE_SELECTION_MARKER` docstring (119-130) | "Separator for the face-selection segment." | **not updated** |
| `ModelVersionParts` docstring (239-245) | "`face_selection` is `None` for a version written before Batch 12 … makes no claim about which face was chosen or how it was cropped" | **not updated** |
| `ModelVersionParts.face_selection` field name (250) | — | **name kept** (see decision) |
| `compose_model_version(face_selection=…)` param + error strings (261-310) | "invalid face-selection fingerprint" | **not updated** |
| `parse_model_version` inline comment (422-424) | "Face selection is stripped first because it is the *last* segment" | **not updated** |
| `face_selection_identity_of` docstring (446-452) | "The face-selection fingerprint" | **not updated** |
| `is_face_selection_pinned` docstring (455-463) | "Whether `version` names a face-selection policy" | **not updated** |

**`inference/models.py`:**

| Location | Text | Status |
|---|---|---|
| `ModelSpec.face_selection_fingerprint` field + docstring (150-164) | "Fingerprint of the face-selection policy this model is served behind." | **not updated** |
| `ModelSpec.with_face_selection()` method + docstring (318-327) | "Return this spec bound to a face-selection policy." | **not updated** |

**`inference/preprocess.py`:** `face_selection_identity()` docstring (625-633)
**updated** — carries an explicit "Face policy only. Since Batch 23 the `.f.`
segment … is the *combined* selection identity" note. `selection_identity()`
(785-804) is new and correct.

**`inference/image_pipeline.py` / `video_pipeline.py`:** block comments
**updated** to describe carry-vs-consume and the composite.

**Tests:** the `face_selection_identity_of(STUB) == pre.face_selection_identity()`
assertions were **semantically rewritten** to `== pre.selection_identity(
default_face_selection(), default_video_selection())` — e.g.
`test_face_size_policy.py:588-612` (`test_the_engine_version_literal_was_updated`).

**Docs:** `docs/FACE_SELECTION_IDENTITY.md`, `docs/CROP_IDENTITY.md` §15,
`docs/RUN_IDENTITY.md`, `docs/VIDEO_FACE_SEMANTICS.md`, `docs/VIDEO_SELECTION_IDENTITY.md`
**updated**.

#### Decision: names stay, docs change, add tripwires

**Rename rejected.** Renaming `face_selection_identity_of` → `selection_identity_of`
(with a deprecation alias) and `ModelSpec.with_face_selection` →
`with_selection` is a `contracts/` and `inference/` public-API change. It ripples
to ~15 call sites across `tests/` and both study trees, forces an alias to be
carried and later removed, and touches strictly more than documentation. The
original design's hard constraint was "do not touch `contracts/version.py`'s
parser"; a rename is adjacent to that and higher-risk than the identity move
itself. The functions' *behaviour* is still correct and total — they return the
`.f.` segment; only the segment's *meaning* broadened.

**Docs change (required).** Add to the §5.7 documentation inventory:
`contracts/version.py` (all nine rows above) and `inference/models.py` (both
rows). Each gets an emphatic banner, not a footnote — e.g. the
`face_selection_identity_of` docstring must open with **"SINCE BATCH 23 (R29)
THIS RETURNS THE COMPOSITE SELECTION IDENTITY (face + video frame selection),
NOT a `FaceSelectionPolicy.identity`. Compare it to
`selection_identity(default_face_selection(), default_video_selection())`, never
to `default_face_selection().identity`."**

**Tripwires (required — three new tests in `tests/test_video_selection_identity.py`):**

| Test | Assertion |
|---|---|
| `test_the_f_marker_documentation_names_both_policies` | `FACE_SELECTION_MARKER.__doc__` **and** the `contracts/version.py` module docstring contain "video" / "frame selection" (case-insensitive) — fails if the docs drift back to face-only. |
| `test_face_selection_identity_of_returns_the_composite` | `face_selection_identity_of(STUB_ENGINE_VERSION) == selection_identity(default_face_selection(), default_video_selection())` **and** `!= default_face_selection().identity`. (Extends the shipped `test_the_selection_identity_is_the_composite`.) |
| `test_model_spec_with_face_selection_docstring_is_current` | `ModelSpec.with_face_selection.__doc__` acknowledges the composite. |

**Follow-up (not a blocker):** a future batch may add `selection_identity_of` as
a thin, documented alias of `face_selection_identity_of`, and
`ModelSpec.with_selection` aliasing `with_face_selection`, so new call sites use
the honest name. Recorded in the risk register under R33.

#### R33 disposition

**In scope for B23** as the doc + tripwire rider. The shipped tree has done the
`inference/preprocess.py` + `docs/` half; the `contracts/version.py` +
`inference/models.py` + tripwire half is **outstanding** and is the one item of
this R2 design not yet satisfied (§9).

---

### 2.4 Defect 4 — see §6 (rebuilt pin inventory)

### 2.5 Defect 5 — see §7 (output-identical, decomposed)

### 2.6 Defect 6 — the OpenAPI assumption, proven from source

#### The defect

The original design made "`python -m backend.openapi --update-snapshot`
produces a **zero** structural diff" a hard gate, while *assuming* the
prose-stripped structural signature drops `default`. If the signature treated
`default` as structural, the N4 `"uniform+adaptive"` → `"uniform"` change would
move it and the gate would trip.

#### Proof from `backend/openapi.py`

The structural signature is built by an **allowlist**, not by stripping a
denylist:

- `structural_signature(schema)` (lines 278-297) returns exactly four keys:
  `openapi`, `contract_version`, `paths`, `components`.
- `components[name]` = `_component_signature(body)` (lines 241-251), which builds
  a dict from exactly: `enum` (if present, and returns early); `required`
  (sorted list); `properties` (each value = `_type_of(node)`);
  `additionalProperties` (only if it is literally `False`). **`default` is never
  read.**
- `_type_of(node)` (lines 209-238) inspects, in order: `$ref`, `enum`, `const`,
  `anyOf`/`oneOf`/`allOf`, `type` (`array` → `items`, `object` →
  `additionalProperties`, else the type string), `format`. **`default` is never
  read.**

For `VideoAnalysis.sampling` (`contracts/video.py:91`, `sampling: str =
Field(default=…)`), `_type_of` returns `"string"` **before and after** the N4
change. `_component_signature` records `properties["sampling"] = "string"`
unchanged. `required` is unchanged (`sampling` was always optional). Therefore
the structural signature is **byte-identical** across the N4 change.

The `_PROSE_KEYS` denylist (line 48: `description`, `summary`, `title`,
`examples`, `example`, `x-mock-mode`) is used elsewhere (raw-schema cleanup) and
is not even the mechanism here — the allowlist construction makes `default`
structurally invisible regardless.

#### Updated OpenAPI gate

The hard gate stands, now on a citation:

> After P18 (`contracts/video.py` default `"uniform+adaptive"` → `"uniform"`),
> `python -m backend.openapi --update-snapshot` **must** report zero structural
> diff, because `structural_signature` reads neither `default` nor
> `description`. If it reports *any* diff, something structural moved that P18
> did not intend — **halt and diagnose**; do not commit the new snapshot.

Shipped: `backend/openapi.snapshot.json` contains no `default` key anywhere in
its `signature` block, and `test_batch_23_moved_only_the_selection_identity`
(`tests/test_video_selection_identity.py:313-324`) greps the snapshot for the
new `.f.` value and `wacdd9240` and asserts absence.

`schema_out/VideoAnalysis.schema.json` **may** gain a `"default": "uniform"` key
— that is a cosmetic JSON-Schema artefact, not a structural break, and
`test_batch_23_moved_only_the_selection_identity` asserts `"stub-0"` does not
appear in any `schema_out/*.json` rather than pinning their exact content.

### 2.7 Defect 7 — the export claim

#### The defect

P19: *"Export `VideoSelectionPolicy` … if the package re-exports the face
equivalents (it exports `FaceSelectionPolicy` — match it)."* — the premise is
factually wrong.

#### The fact

`inference/__init__.py` `__all__` (lines 60-89) contains: 8 protocol/error
names, 2 pipelines (`ImagePipeline`, `VideoPipeline`), 8 registry helpers, 8
calibration helpers. It does **not** contain `FaceSelectionPolicy`,
`default_face_selection`, `face_selection_identity`, `MediaCaps`, or any
`preprocess` symbol. The module docstring is explicit: *"Preprocessing and
bundle construction are reachable through their modules but deliberately absent
from this namespace."*

#### R2 resolution

**Drop P19. Add no exports.** `VideoSelectionPolicy`, `default_video_selection`,
`video_selection_identity`, `selection_identity` are reached via
`inference.preprocess`, exactly as `FaceSelectionPolicy` is. Every consumer in
the shipped tree already does this (`inference/video_pipeline.py:52` `from
inference import preprocess as pre`; tests `from inference import preprocess as
pre`). Confirmed: `inference/__init__.py` was not modified by B23.

---

## 3. New findings — disposition

### R32 — `batch_size` output-neutrality is backend-dependent

**Mechanism.** `VideoSelectionPolicy` permanently excludes `batch_size`.
Neutrality holds only while **all three** of:

1. `batch_clips` (`inference/preprocess.py:1184`) yields contiguous
   non-overlapping slices with **no padding**;
2. the model backend's `score()` is **row-independent** — `StubBackend.score`
   (`inference/models.py:368`) iterates `for i in range(N)`;
3. `per_frame` reconstruction preserves order and values —
   `topk_indices` / `pool_at` run on the full `scores` array after every batch
   (`inference/video_pipeline.py`, `inference/postprocess.py:537-575`).

(2) and (3) are properties of code B23 does **not** own. A backend with a static
batch dimension (ONNX / TensorRT — a roadmapped migration, currently a standing
constraint) would pad the final batch, and the pad-row count depends on
`batch_size` → `batch_size` becomes output-affecting → it must move into the
policy → another `.f.` move + cache generation. A backend using eval-mode
batch-norm or cross-frame temporal attention (the design's own future
`pooling_rule = "attention"`) breaks row-independence.

**Severity (five axis):** Correctness 1 · Reproducibility 2 · Migration 2 ·
Cache-gen 3 (a future re-inclusion is one generation) · Bug-likelihood 2.
Overall **Low–Medium**.

**Belongs in B23?** As documentation only — **yes**. As a code change — **no**
(there is no defect today; the stub path is genuinely neutral).

**Decision & justification.** The `VideoSelectionPolicy` "what is NOT here and
why" docstring (`inference/preprocess.py:657-662`) states the elementwise-widen
reason. R2 requires it to additionally name the **full precondition** and the
**trigger that re-opens the exclusion**: *"Neutral only while (a) `batch_clips`
does not pad, (b) the backend's `score()` is row-independent, and (c) `scores`
reconstruction is order/value preserving. A static-shape backend (ONNX/TensorRT)
or a cross-frame backend (temporal attention) breaks (a) or (b) and forces
`batch_size` into this policy."* `test_batch_size_is_output_neutral` stays as
the tripwire that fails when the regression lands. Enter R32 in
`REPRODUCIBILITY_AUDIT.md` as **tracked, not closed**.

### R33 — `contracts.version` names/docs `.f.` as face-only

**Mechanism.** Covered in §2.3. `face_selection_identity_of()` and
`is_face_selection_pinned()` are public `contracts` functions returning /
asserting on a value that is no longer a face-selection identity for B23+
strings; their docstrings, the marker docstring, the module docstring and
`ModelSpec.with_face_selection` all still say "face selection".

**Severity (five axis):** Correctness 2 (a caller comparing to
`default_face_selection().identity` gets a false negative — but no shipped
caller does; the tests were rewritten) · Reproducibility 1 · Migration 3 (a
later rename touches ~15 sites) · Cache-gen 0 · Bug-likelihood 3 (a future
developer or external log parser is genuinely likely to misuse the name).
Overall **Low**, trending Medium on bug-likelihood.

**Belongs in B23?** **Yes** — as the Defect 3 doc + tripwire rider. The
behaviour is correct; the contract's self-description is not, and B22 shipped
the identical gap for `interpolation_rule` with no tripwire, which this batch
should not repeat.

**Decision & justification.** §2.3: names stay, docs change (inventory extended
to `contracts/version.py` + `inference/models.py`), three tripwires added. A
rename is a disproportionate response carried as a non-blocking follow-up. The
shipped tree has done half; the other half is the §9 residual.

### R34 — fail-closed OR1 + `STUB_ENGINE_VERSION` move ⇒ B23 is not rolling-deployable

**Mechanism.** OR1's worker gate is fail-closed. During a rolling deploy the
configured literal (env or `backend.config`) is either the old value or the new
value at any instant:

- **old literal + a new worker** (new engines report the new `.f.`) ⇒ the new
  worker's gate fails closed ⇒ it cannot start;
- **new literal + an old worker** ⇒ the old worker's gate fails closed.

B20 and B22 also moved `STUB_ENGINE_VERSION`, but without a fail-closed gate, so
the window was "brief cache misses". B23 converts that window into "workers
refuse to start" unless the restart is coordinated or `DFP_SKIP_VERSION_ALIGNMENT`
is held on for the duration.

**Severity (five axis):** Correctness 1 (fail-loud is *safer* than fail-silent)
· Reproducibility 1 · Migration 4 (every future `STUB_ENGINE_VERSION`-moving
batch now inherits this) · Cache-gen 0 · Bug-likelihood 2. Overall **Low as a
defect, Medium as an operational constraint**.

**Belongs in B23?** As a **rollout-guidance + CHANGELOG note** — yes. As a code
change — **no**: this is OR1 working as intended (fail loud on misalignment).

**Decision & justification.** §6.5 of this design and the B23 CHANGELOG entry
must state plainly: *"B23 moves `STUB_ENGINE_VERSION`. Because OR1's worker gate
is fail-closed, a pod-by-pod rolling update will have workers refusing to start
during the window where the configured literal and the engine build disagree.
Deploy B23 with a coordinated worker restart, or set
`DFP_SKIP_VERSION_ALIGNMENT=true` for the duration of the window and unset it
after."* Enter R34 in `REPRODUCIBILITY_AUDIT.md` as a **candidate / tracked**
operational constraint, not a finding against the system.

---

## 4. Route A vs Route B — re-scored

The review argues Route A (`selection_identity(...)` folded into `.f.`) was
oversold relative to Route B (a new `.v.` segment). Re-scored from first
principles.

### 4.1 Route A — `.f.` composite

`selection_identity(face, video) = fingerprint({"face": face.identity, "video":
video.identity})`; both pipelines pass the result to
`ModelSpec.with_face_selection(...)`.

| Dimension | Score | Detail |
|---|---|---|
| **Parser complexity** | **0 new surface** | `.f.` stays one opaque 8-hex token. `parse_model_version` (`contracts/version.py:402-443`) unchanged — it already peels `.f.` as a single `partition`. `_MODEL_MARKERS` unchanged. `ModelVersionParts` unchanged (4 fields). `compose_model_version` unchanged. **No new parser era.** `test_parse_model_version_is_total_across_the_b23_string` adds one string to an existing list, no new shape. |
| **Operator visibility** | **degraded** | An operator seeing `.f.` move in a log line **cannot tell** whether the face policy or the video policy changed. Recovering that needs both policy objects + `FaceSelectionPolicy.differences()` / `VideoSelectionPolicy.differences()` — not available from a log line. This **directly contradicts** the `contracts/version.py` module docstring's stated goal (lines 47-53): *"readable on purpose. An operator reading a log line should be able to see which component moved without decoding anything."* |
| **Migration burden** | **1 cache generation + ~8 collateral files** | One `.f.` value change ⇒ one full miss per file per namespace (both). Collateral literal pins: `backend/config.py`, `test_crop_geometry.py`, `test_crop_identity.py`, `test_branch_b_semantics.py`, `test_headline_and_face_semantics.py`, `test_decode_identity_completeness.py`. Semantic rewrites: `test_face_size_policy.py`, `test_face_selection_identity.py`. N4 fixture regen: `contracts/fixtures.py`, `tests/stub_handlers.py`, `fixtures_out/video_*.json`. (Full list in §6.) |
| **Future extensibility** | **adequate** | When SCRFD gives video a face stage, `selection_identity` already carries both policies for both pipelines; the video path starts *consuming* the face half it already *carries*. A third selection dimension becomes a third key in the `fingerprint({...})` dict — no format change. But the composite is a hash-of-hashes and is not injective over `(face, video)` (acceptable — same as every `fingerprint()`; §7). |

### 4.2 Route B — new `.v.` segment

A `DECODE_MARKER`-style `.v.` marker, `ModelSpec.with_video_selection()`, a fifth
`ModelVersionParts` field, `face_selection=None`-style pre-B23 era logic.

| Dimension | Score | Detail |
|---|---|---|
| **Parser complexity** | **+5 permanent surfaces** | (1) `.v.` in `_MODEL_MARKERS`; (2) fifth field `video_selection: str \| None` in `ModelVersionParts`; (3) a new peel step in `parse_model_version` (peel `.v.` — decide: inside or outside the model identity, before or after `.f.`); (4) `compose_model_version` grows a `video_selection` param + its marker-collision guard; (5) a permanent "pre-B23 era" case in `test_parse_model_version_is_total_across_all_eras` that lives **forever**. Every `ModelVersionParts.compose()` fixture across the tree grows a field. |
| **Operator visibility** | **preserved** | `stub-0-w…+c.…g.…f.<face>.v.<video>.d.<decode>` — an operator sees `.v.` move and knows it is frame selection / pooling, not face selection. Honours the module docstring's promise. |
| **Migration burden** | **strictly greater than A** | Same 1 cache generation and same collateral pins as A, **plus** the parser-era tax, **plus** `compose()` fixture churn everywhere, **plus** a new `video_selection_identity_of` helper and its tests, **plus** every place that constructs or asserts on `ModelVersionParts`. |
| **Future extensibility** | **cleaner separation, but** | A third selection dimension is naturally a fourth segment — clean. But the identity vocabulary's organising principle is *"one opaque fingerprint per lifecycle-distinct concern"* (`contracts/version.py:20-46`): weights (retrain), calibration (dist shift), gate (floors), face (shipped code + asset), decode (environment). Face selection and frame selection **share a lifecycle** — both are shipped code, both move together under the SCRFD migration — so they are arguably *one* concern, which is the case for A. |

### 4.3 Does the recommendation change?

**No — Route A stays, but conditionally.**

Route A is the right call **only if** the design also commits to the Defect 3 /
R33 work: (i) redocument the `.f.` marker, the module docstring, and the two
`*_face_selection_*` functions + `ModelSpec.with_face_selection` to name the
composite, and (ii) add the three tripwires. Without (i)+(ii), Route A leaves
`contracts/version.py` self-contradictory and silently degrades the log-line
legibility the module docstring explicitly promises — and at that point the
balance tips toward Route B.

The original design's §3.2 dismissed the visibility cost as "same as B22's
`input_size` → `-w`". That comparison is wrong: `input_size` is a
*sub-fingerprint inside `-w`* that an operator never reads individually; `.f.`
vs `.v.` is *top-level log-line legibility*, which is the specific thing the
module docstring promises. R2 §3.2 must acknowledge this honestly rather than
wave it away, and R33's fixes are what make Route A defensible.

The parser-era tax on Route B is real and permanent and is the genuinely strong
argument for A. On balance: **A, plus R33 as a required rider.**

---

## 5. Test matrices — rebuilt from first principles

Reconstructed, not patched. Six matrices: fail-before, pass-after, collateral,
migration assertions, parser assertions, startup-alignment assertions. Where a
row exists in the shipped `tests/test_video_selection_identity.py` or
`tests/test_startup_alignment.py`, the shipped name is given.

### 5.1 Fail-before (must fail on the pre-B23 tree, pass after)

The **one true fail-before** is against the pre-B23 tree, where
`tests/test_video_identity_gap.py` asserted the *equality* that was the gap. New
invariants that reference `VideoSelectionPolicy` cannot "fail" pre-B23 (the
class does not exist — they are import errors, not wrong assertions); they are
labelled **new invariant**.

| # | Assertion | Pre-B23 state | Post-B23 state | Shipped test |
|---|---|---|---|---|
| FB1 | two `n_frames` ⇒ different `model_version`, `model_identity`, `effective_version`, and different `p_fake` | `dense.model_version == sparse.model_version` (the gap, asserted by `test_video_identity_gap`) | all four `!=` | `test_r29_n_frames_moves_the_model_version` |
| FB2 | `min_usable_frames` floor moves the identity **and** still gates verdict→abstention; 12-frame clip, `n_frames=16`, floor 8 decides / floor 13 abstains | equality asserted; old mechanism `n_frames=9,muf=16` | `decided.model_version != abstained.model_version`; `abstained.verdict is INSUFFICIENT_EVIDENCE` | `test_r29_min_usable_frames_moves_the_identity` |
| FB3 | `pool_divisor` bound by the policy: two divisors ⇒ two identities and two pooled means | `topk_indices(scores)` ignores any policy (`k = arr.size // 8`) | `k = max(1, n_scores // pool_divisor)` from the policy; identities and means differ | `test_r29_pool_divisor_is_bound_by_the_policy` |
| FB4 | `min_usable_frames` is *in* `VideoSelectionPolicy.identity` (was: reaches no identity) | reached nothing | `replace(SHIPPED, min_usable_frames=9).identity != SHIPPED.identity` | `test_r29_min_usable_frames_is_in_the_video_selection_identity` |
| FB5 | **new invariant** — every one of the 5 policy fields moves `.identity`; `{field names} == {variation keys}` guard | n/a (no class) | each `replace(SHIPPED, field=alt).identity != SHIPPED.identity`; `differences()[0]` names the field | `test_every_policy_field_moves_the_identity` |
| FB6 | `test_configured_model_version_matches_the_engines` | passes (literal == engines, pre-B23 value) | fails until `STUB_ENGINE_VERSION` is regenerated, then passes | `tests/test_model_version.py` (dynamic) |
| FB7 | worker boot refuses a stale literal, before the loop, naming the engine value | no startup call exists | `verify_engine_alignment_at_startup(stale)` raises `RuntimeError` "does not match the engines" | `test_worker_boot_refuses_a_stale_literal` |
| FB8 | API `lifespan` refuses `"1.0.0-dev"` and a `.d.`-less version | `lifespan` had no check | `RuntimeError` mentioning `"1.0.0-dev"` / `"decode"` | `test_api_lifespan_refuses_an_unparseable_model_version`, `..._with_no_decode_segment` |

### 5.2 Pass-after (invariants that must hold post-B23)

| # | Invariant | Shipped test |
|---|---|---|
| PA1 | shipped policy through a fixed clip: `model.version == B23_STUB_ENGINE_VERSION`, `video.sampling == "uniform"`, `video.pooling == "topk_mean"`, `p_fake == SHIPPED_P_FAKE` (a pinned float) | `test_the_shipped_video_bundle_did_not_move` |
| PA2 | `sampling_rule` matches `sample_frame_indices` — re-implement the stride formula, compare index lists over `{(1000,48),(1000,9),(50,48),(48,48),(7,8),(1,4)}` (hits all three branches: `fc>n` stride, `fc<=n` passthrough, empty) | `test_the_sampling_rule_name_matches_sample_frame_indices` |
| PA3 | `pooling_rule` + `pool_divisor` match the pipeline; `bundle.video.pooling == policy.pooling_rule`; `len(topk_indices(scores, k)) == k` for `k = max(1, n_scores // pool_divisor)` | `test_the_pooling_rule_and_divisor_match_the_pipeline` |
| PA4 | `batch_size` output-neutral: two batch sizes ⇒ same `p_fake`, `verdict`, `model.version`, and `video.model_dump()` (field-wise, **not** whole bundle — timing excluded) | `test_batch_size_is_output_neutral` |
| PA5 | `VideoSelectionPolicy` has no defaults (iterate `fields()`, assert `default is MISSING and default_factory is MISSING`) | `test_the_video_selection_policy_has_no_defaults` |
| PA6 | the policy validates its invariants — `(0,0,1,…)`, `(8,9,1,…)`, `(8,4,0,…)` each raise `ValueError` | `test_the_policy_validates_its_invariants` |
| PA7 | image and video engines report the **same** `model_version`, equal to `STUB_ENGINE_VERSION` | `test_image_and_video_engines_report_the_same_version` |
| PA8 | image pipeline **carries** `VideoSelectionPolicy` without consuming it: `image_pipeline.py` source contains no `sample_frame_indices` / `topk_indices`; yet `ImagePipeline(video_selection=replace(SHIPPED, n_frames=9)).model_version != ImagePipeline().model_version` | `test_the_image_pipeline_carries_the_video_segment_without_consuming_it` |
| PA9 | the legacy scalar shim keeps the identity honest: `VideoPipeline(n_frames=9).model_version == VideoPipeline(video_selection=replace(SHIPPED,n_frames=9)).model_version != VideoPipeline().model_version` | `test_the_legacy_scalar_shim_keeps_the_identity_honest` |
| PA10 | **R33 tripwire (new)** — `face_selection_identity_of(STUB) == selection_identity(default_face_selection(), default_video_selection()) != default_face_selection().identity` **and** the `.f.` marker + module docstrings name both policies | extend `test_the_selection_identity_is_the_composite` + new `test_the_f_marker_documentation_names_both_policies` |
| PA11 | **R32 tripwire** — `VideoSelectionPolicy` docstring names the full `batch_size`-neutrality precondition and the static-shape/cross-frame trigger | new `test_the_batch_size_exclusion_note_is_complete` (string check on `VideoSelectionPolicy.__doc__`) |

### 5.3 Collateral (tests that pass today, must be updated for the `.f.` move)

| Test | What moves | Kind |
|---|---|---|
| `test_crop_geometry.py:578` | `POLICY.identity == "251d22e9"` | **unchanged** — the *face* policy identity did not move |
| `test_crop_geometry.py:582-583` | `model_identity_of(STUB_ENGINE_VERSION) == "…f.e629642c"` | literal bump `251d22e9` → `e629642c` |
| `test_crop_identity.py:52-63` | `B22_POLICY_IDENTITY` (`251d22e9`, kept), + new `B23_SELECTION_IDENTITY` / `B23_MODEL_IDENTITY` / `B23_STUB_ENGINE_VERSION` constants | new constants; model-identity/engine literals move |
| `test_crop_identity.py:266-267` | grep-guard adds `"e629642c"` alongside `"251d22e9"` / `"wacdd9240"` "not in snapshot" | literal add |
| `test_crop_identity.py:281-291` | `model_identity_of(STUB) == B23_MODEL_IDENTITY`; `parts.face_selection == B23_SELECTION_IDENTITY` | literal bump |
| `test_crop_identity.py:301-302` | parser-era list gains `B23_STUB_ENGINE_VERSION` | append, **no new shape** |
| `test_branch_b_semantics.py:462` | `_EXPECTED_EFFECTIVE = "…f.e629642c"` | literal bump |
| `test_headline_and_face_semantics.py:97,450,452,558` | `_EXPECTED_EFFECTIVE` (×1 def, 3 refs) | literal bump |
| `test_decode_identity_completeness.py:96` | `MODEL_ID = "…f.e629642c"` | literal bump |
| `test_face_size_policy.py:588-612` | `face_selection_identity_of(STUB) == combined` where `combined = selection_identity(default_face_selection(), default_video_selection())` | **semantic rewrite** (was `== face_selection_identity()`) |
| `test_face_selection_identity.py` (two-pipeline loop ~`:560`; `:650`) | expected literal + the `== pre.face_selection_identity()` in-loop assert → `== pre.selection_identity(...)` | **semantic rewrite** + literal |
| `test_evaluation_identity.py:75` | `MODEL_ID = "stub-0-w7f45fada+c.739c3fec.g.ae232cba"` (pre-B22 weights, **no `.f.`**) | **verify** — this looks like a deliberately frozen historical model-only string for evaluation-identity tests; confirm it is not meant to track the live literal before touching it |
| `tests/video_selection_study.py` | prints live `video_selection_identity()` + composite `.f.` | new study, not a pin |
| `tests/face_selection_study.py:511-515` | prints `face_selection_identity_of(STUB)` and notes `face_selection_identity()` stays face-only | doc/print only |

### 5.4 Migration assertions

| # | Assertion | Shipped test |
|---|---|---|
| MG1 | pre-B22 stored bundle still loads: `payload["model"]["version"] = PRE_B22_MODEL_IDENTITY + ".d.c20a064a"` round-trips | `test_crop_identity.py:316-320` |
| MG2 | **N4 — stored bundle with `sampling: "uniform+adaptive"` still validates** (the value already sitting in old bundles/fixtures) — a `VideoAnalysis` constructed with the stale string must not raise (it is a plain `str` field) | **new** — add to `test_old_artifacts_still_load` (this was review gap M1; add it) |
| MG3 | benchmark cross-boundary: a pre-B23 run and a post-B23 run of an unchanged pipeline read as "different models" (`.f.` moved) — the B22 pattern; extend it | extend `test_crop_identity.py` `test_the_benchmark_gate_still_compares_across_the_boundary` for a `(PRE=B22_MODEL_IDENTITY, POST=B23_MODEL_IDENTITY)` pair (review gap §7.4) |
| MG4 | idempotency-key regeneration: `{sha256}:{model_version}` for a job submitted before and after B23 differ ⇒ one re-run, no dedup against the pre-B23 self; an in-flight job re-leased after deploy re-runs once | assert in `test_video_selection_identity.py` or a service test; document in §6.3 |

### 5.5 Parser assertions

| # | Assertion | Shipped test |
|---|---|---|
| PR1 | `.f.` is still one fingerprint — no new era: `parse_model_version(B23_STUB_ENGINE_VERSION).compose() == B23_MODEL_IDENTITY`; `compose_runtime_version(model=…, decode="deadbeef")` round-trips | `test_parse_model_version_is_total_across_the_b23_string` |
| PR2 | `parse_model_version(STUB).weights == "stub-0-wacdd9240"`, `.calibration == "739c3fec"`, `.gate == "ae232cba"`, `.face_selection == "e629642c"`, `STUB.endswith(".d.a97ccb68")` | `test_batch_23_moved_only_the_selection_identity` |
| PR3 | the composite is a valid `.f.` payload: `fingerprint(...)` output is 8 hex chars, marker-free, so `compose_model_version`'s `any(m in face_selection for m in _MODEL_MARKERS)` guard (`contracts/version.py:304`) passes | implied by PR1; add an explicit assert `all(m not in combined for m in _MODEL_MARKERS)` |
| PR4 | `test_crop_identity.py` era list (`:295-303`) stays total across pre-B12, pre-B22, B22, B23 strings — one appended entry, **no branch** | `test_crop_identity.py` era test |

### 5.6 Startup-alignment assertions

| # | Assertion | Shipped test |
|---|---|---|
| SA1 | worker gate accepts the shipped configuration — via explicit arg and via `$DFP_MODEL_VERSION` | `test_the_worker_gate_accepts_the_shipped_configuration` |
| SA2 | worker gate refuses a stale literal, fail-closed, naming the engine value | `test_worker_boot_refuses_a_stale_literal` |
| SA3 | `run_with_shutdown` runs the gate **before** touching the runner (an `ExplodingRunner.run` that must never be reached) | `test_run_with_shutdown_runs_the_gate_before_the_loop` |
| SA4 | `DFP_SKIP_VERSION_ALIGNMENT` truthy ⇒ no raise + a `WARNING` log every boot | `test_skip_flag_bypasses_with_a_warning` |
| SA5 | gate refuses engines that disagree with **each other** even with no literal in reach (the config-default fallback path) | `test_the_gate_refuses_engines_that_disagree_with_each_other` |
| SA6 | API `lifespan` accepts the shipped literal (`/health` → 200) | `test_api_lifespan_accepts_the_shipped_literal` |
| SA7 | API `lifespan` refuses a non-composed string and a `.d.`-less string | `test_api_lifespan_refuses_an_unparseable_model_version`, `..._with_no_decode_segment` |
| SA8 | **partial engine mismatch (review gap M3)** — video engine reports a different `.f.` than the image engine ⇒ `verify_version_alignment` / the gate names the video engine | **new** — `test_the_gate_names_the_mismatched_engine` (mock `engine_versions` to `{"image": <shipped>, "video": <stale .f.>}`) |
| SA9 | alignment rejects a face-only `.f.` after B23: `verify_version_alignment(compose(model=B23_MODEL_IDENTITY.replace(".f.e629642c",".f.251d22e9"), decode=…))` raises | `test_alignment_accepts_the_shipped_configuration_and_rejects_a_stale_one` |

---

## 6. Pin inventory — rebuilt from scratch

### 6.1 Every touched literal

`.f.` `251d22e9` → `e629642c`. `STUB_ENGINE_VERSION`
`stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.a97ccb68`. Model identity
`stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c`.

| File:line | Literal | Change |
|---|---|---|
| `backend/config.py:41` | `STUB_ENGINE_VERSION` | `…f.251d22e9.d.a97ccb68` → `…f.e629642c.d.a97ccb68` |
| `backend/config.py` docstring | the "S8 Batch NN" history block | add a B23 paragraph (`.f.` moved, no pixel changed) |
| `tests/test_crop_geometry.py:582-583` | `model_identity_of(STUB)` expected | `…f.251d22e9` → `…f.e629642c` |
| `tests/test_crop_geometry.py:578` | `POLICY.identity == "251d22e9"` | **unchanged** — face identity |
| `tests/test_crop_identity.py:52-54` | `B22_POLICY_IDENTITY` / `B22_MODEL_IDENTITY` / `B22_STUB_ENGINE_VERSION` | `B22_POLICY_IDENTITY` kept (`251d22e9`); the two model/engine literals stay as *B22 historical* constants |
| `tests/test_crop_identity.py:61-63` | `B23_SELECTION_IDENTITY = "e629642c"`, `B23_MODEL_IDENTITY`, `B23_STUB_ENGINE_VERSION` | **new** constants |
| `tests/test_crop_identity.py:266-267` | snapshot grep-guard | add `"e629642c"` |
| `tests/test_crop_identity.py:282-283,290` | `model_identity_of(STUB) == B23_MODEL_IDENTITY`; `parts.face_selection == B23_SELECTION_IDENTITY` | point at B23 constants |
| `tests/test_crop_identity.py:301-302` | era list | append `B23_STUB_ENGINE_VERSION` |
| `tests/test_branch_b_semantics.py:462` | `_EXPECTED_EFFECTIVE` | `…f.251d22e9` → `…f.e629642c` |
| `tests/test_headline_and_face_semantics.py:97` (+ refs `:450,:452,:558`) | `_EXPECTED_EFFECTIVE` | `…f.251d22e9` → `…f.e629642c` |
| `tests/test_decode_identity_completeness.py:96` | `MODEL_ID` | `…f.251d22e9` → `…f.e629642c` |
| `contracts/video.py:92` | `sampling` `Field(default="uniform+adaptive")` | → `"uniform"` (+ description reworded to reference `VideoSelectionPolicy.sampling_rule`) |
| `contracts/fixtures.py:515` | `sampling="uniform+adaptive"` | → `"uniform"` |
| `tests/stub_handlers.py:433` | `VideoAnalysis(sampling="uniform+adaptive")` | → `"uniform"` — **review gap §7.3, was missing from the original inventory** |
| `fixtures_out/video_authentic.json:88` | `"sampling": "uniform+adaptive"` | regenerate → `"uniform"` |
| `fixtures_out/video_insufficient_evidence.json:71` | same | regenerate → `"uniform"` |
| `fixtures_out/video_manipulated.json:88` | same | regenerate → `"uniform"` |
| `tests/test_video_selection_identity.py:66-68` | `B23_SELECTION_IDENTITY` / `B23_MODEL_IDENTITY` / `B23_STUB_ENGINE_VERSION` + `SHIPPED_P_FAKE` | the new suite's own pins |
| `tests/test_evaluation_identity.py:75` | `MODEL_ID = "stub-0-w7f45fada+c.739c3fec.g.ae232cba"` | **verify only** — appears to be a frozen pre-B22 model-only fixture; do not bump without confirming it tracks the live literal |
| `docs/*` (CROP_IDENTITY, FACE_SELECTION_IDENTITY, RUN_IDENTITY, VIDEO_FACE_SEMANTICS, VIDEO_SELECTION_IDENTITY, REPRODUCIBILITY_AUDIT) | prose references to `…f.251d22e9` / "the `.f.` segment" | update / add superseding markers |

**Enumerate before implementing:**
`grep -rn "251d22e9\|f\.251d22e9\|_EXPECTED_EFFECTIVE\|MODEL_ID\s*=\|face_selection_identity_of(STUB\|uniform+adaptive" tests/ backend/ contracts/ docs/ fixtures_out/`.

### 6.2 Every semantic assertion (rewrite, not literal-bump)

| File | Assertion before | Assertion after |
|---|---|---|
| `tests/test_face_size_policy.py:588-612` (`test_the_engine_version_literal_was_updated`) | `face_selection_identity_of(STUB) == pre.face_selection_identity()` | `combined = pre.selection_identity(pre.default_face_selection(), pre.default_video_selection())`; `STUB.startswith(spec.with_face_selection(combined).effective_version)`; `face_selection_identity_of(STUB) == combined` |
| `tests/test_face_selection_identity.py` (in-loop, ~`:559`) | `face_selection_identity_of(pipeline.model_version) == SHIPPED.identity` | `== pre.selection_identity(default_face_selection(), default_video_selection())` |
| `tests/test_face_selection_identity.py` `test_every_policy_field_moves_the_identity` | asserts `names == set(alternatives)` for `FaceSelectionPolicy` | **unchanged** — face policy fields did not change; the *video* analogue is a separate test in the new suite |
| `docs/FACE_SELECTION_IDENTITY.md` — "the `.f.` segment names the face-selection policy" | — | "names the *selection* policy: face selection (image + video) plus frame selection and pooling (video), combined by `selection_identity()`" |
| **R33 (outstanding)** `contracts/version.py` module docstring `face` paragraph; `FACE_SELECTION_MARKER` docstring; `ModelVersionParts` docstring; `face_selection_identity_of` / `is_face_selection_pinned` docstrings; `inference/models.py` `ModelSpec.face_selection_fingerprint` + `with_face_selection` docstrings | "face-selection" | "selection (face + video frame selection)" with an emphatic since-B23 banner on the two public functions |

### 6.3 Helpers that must be rewritten, not bumped

| Helper | Location | Why a rewrite |
|---|---|---|
| `topk_indices(scores, k=None)` | `inference/postprocess.py:537-560` | **signature preserved**, but the *contract changes*: `k` is no longer "defaults to `arr.size // 8`, callers rarely pass it" — the pipeline now **always** passes `k = max(1, len(scores) // pool_divisor)` explicitly (`inference/video_pipeline.py`). The `arr.size // 8` fallback stays only for study/direct callers. Docstring rewritten (`:545-547`). Not a bump — a semantic change to who owns `k`. |
| `build_video_analysis(...)` | `inference/postprocess.py:643-660` | `sampling` / `pooling` change from hard-coded (`"uniform"` at `:649`, `"topk_mean"` at `:650`) to **sourced from the caller** (`VideoSelectionPolicy` via the pipeline). Kept as defaulted kwargs (not required) so study callers still work, but the real bundle path always passes them. Body + comment rewritten. |
| `VideoPipeline.__init__` | `inference/video_pipeline.py:69-132` | new `video_selection` param; the legacy `n_frames`/`batch_size`/`min_usable_frames` scalars become a `replace()`-based shim that reconstructs the policy *before* identity; `.with_face_selection(...)` now receives `selection_identity(self._selection, self._video_selection)`. Not a bump — new resolution logic. |
| `ImagePipeline.__init__` | `inference/image_pipeline.py:35-65` | new `video_selection` param (carried, not consumed); `.with_face_selection(...)` receives the composite. |
| `face_selection_identity()` | `inference/preprocess.py:625-633` | docstring rewritten to say "face policy only; the `.f.` segment is now the composite `selection_identity`". Function body unchanged. |
| `verify_engine_alignment_at_startup(configured=None)` | `workers/runner.py:398-471` | **new** helper — the OR1 worker gate. Not derived from anything; env-only + optional arg, local imports (§2.1). |
| `lifespan` (API) | `backend/app.py:66-128` | new stdlib-only `model_version` format check block (`:90-123`). |
| `sample_frame_indices(frame_count, n)` | `inference/preprocess.py:1051-1063` | **unchanged** — but it is the behaviour `sampling_rule="uniform"` is held to; PA2 re-derives it. |

### 6.4 Findings from the adversarial review folded into the inventory

- `tests/stub_handlers.py:433` — the stale `"uniform+adaptive"` site (review §7.3) — **added** (§6.1).
- `test_evaluation_identity.py:75` `MODEL_ID` — **added as verify-only** (§6.1).
- The `face_selection_identity_of(...) == SHIPPED.identity` in-loop assert in
  `test_face_selection_identity.py` (~`:559`) — **added as a semantic rewrite**,
  not a literal bump (review §13 #4).
- `build_video_analysis` in `test_postprocess.py` — if any test constructs it
  directly with positional `sampling`/`pooling`, it needs updating; the shipped
  route kept defaulted kwargs so **verify** rather than assume.
- `contracts/version.py` + `inference/models.py` docstrings — **added** (R33).

---

## 7. Output-identical — decomposed

The original design's "output byte-identical" / "byte-identical bundle" is
**replaced** by a four-part statement. What B23 guarantees does **not** move, for
`default_video_selection()` through a fixed clip:

| Category | Fields | Guarantee |
|---|---|---|
| **Scored output** | `p_fake`, `verdict`, `frame_scores` / `video.frame_scores`, `video.segments`, `video.top_frames`, `video.frames_scored`, `video.sampling`, `video.pooling` | **Guaranteed identical** — every knob names current behaviour verbatim; `sample_frame_indices`, `topk_indices`(k), `pool_at` all receive the same values they used pre-B23. Pinned by `test_the_shipped_video_bundle_did_not_move` (`p_fake == SHIPPED_P_FAKE`, a hard float). |
| **Version metadata** | `model.version` | **Guaranteed to move** — `.f.` `251d22e9` → `e629642c`. This is the point of the batch. |
| **Wire metadata (N4)** | `video.sampling` | Moves in **fixtures / OpenAPI doc only**: `"uniform+adaptive"` → `"uniform"`. The *server* always set `"uniform"` in the real bundle already (`build_video_analysis` hard-coded it), so no real response changes — only the stale default in the schema and the three `fixtures_out/video_*.json`. |
| **Timing** | `timing` (phase timings via `PhaseTimer` / `timer.to_timing()`) | **Not guaranteed and never was.** Phase timings depend on wall-clock noise and on how work is chunked (`batch_size`). `test_batch_size_is_output_neutral` compares `video.model_dump()` **field-wise excluding timing**, and `p_fake` / `verdict` / `model.version` directly — not whole-bundle equality. |
| **Job metadata** | `job_id`, `submitted_at`, and anything derived per-request | **Not comparable across runs** by construction. |
| **Diagnostics** | progress-event fractions/counts, log lines | **Depend on `batch_size`** (`test_progress_scales_with_the_batch_size` asserts exactly this) — deliberately outside the identity, deliberately not part of "output-identical". |

**Exact statement for §8.5 / §9:** *B23 is guaranteed not to move any scored
field of the video bundle for the shipped policy. `model.version` moves by
design. `video.sampling` moves only in fixtures and the OpenAPI document (the
live server was already emitting `"uniform"`). `timing`, job metadata and
progress diagnostics are not part of the guarantee and were never byte-stable
across runs.*

---

## 8. Risk register — rebuilt

Ten findings, ranked on the five axes the original design used: **Correctness /
Reproducibility / Migration cost / Cache-gen cost / Bug-likelihood** (1 low – 5
high). Ordered by aggregate severity.

| Finding | Corr | Repro | Migr | Cache | Bug-lik | Status & note |
|---|---|---|---|---|---|---|
| **R13** face-crop instability (310× amplification, 25.7% of decisions) | 5 | 4 | 4 | 4 | 3 | **Open**, blocked on SCRFD; acceptance criterion written (`REPRODUCIBILITY_AUDIT.md` R13 section). The largest open correctness defect. Not B23. |
| **R24** video whole-frame squash (1.778× on 16:9, every video bundle) | 4 | 3 | 3 | 3 | 4 | **Open**, live on 100% of video output; fix belongs with the video face stage (R18). Not B23 — pixel-moving. |
| **R23** crop border distortion (13% stretch, live today with square Haar boxes) | 4 | 2 | 3 | 3 | 4 | **Open** (correctness halves); documentation half closed B15. Blocked on SCRFD; pixel-moving. Not B23. |
| **R29** video selection identity gap | 3 | 5 | 2 | 3 | 4 | **Closed in B23.** Highest reproducibility score of anything actionable now; the next dev to wire a `Settings` frame-budget knob would have tripped it. |
| **R34** fail-closed OR1 makes `STUB_ENGINE_VERSION` moves non-rolling-deployable | 1 | 1 | 4 | 0 | 2 | **Tracked (candidate).** OR1 working as intended (fail loud). New operational constraint on every future `.f.`/`.d.`/`-w` batch. Rollout note + CHANGELOG, not code. |
| **R31** witness digest ⊄ model identity (`.f.`, `input_size`, gate) | 3 | 4 | 3 | 1 | 2 | **Open — recorded B23A.** A temperature fitted for one model applies to another if the fits quantize-converge. Needs `CALIBRATION_WITNESS_VERSION` 1.3.0 — its own batch; no real model exists to calibrate yet. |
| **R30** image decode libs (libjpeg-turbo / libpng / libwebp) outside `decode_identity` | 3 | 4 | 2 | 4 | 2 | **Open — recorded B23A.** Image analogue of R27; only `cv2.__version__` proxies. Needs a `decode_identity` **value** extension = one cache generation; pair with a D8–D10 cleanup batch. |
| **R28** `resize` `or`-predicate area-filters an enlarged axis (0.040/255) | 2 | 2 | 2 | 3 | 2 | **Open — deferred (B22).** Named verbatim in `FaceSelectionPolicy.interpolation_rule` + pinned. Its fix **moves shipped pixels**, so it stays out of B23 (would break §7's scored-output guarantee); rides the SCRFD / R23 batch, which already moves pixels. |
| **R33** `contracts.version` names/docs `.f.` as face-only; no tripwire | 2 | 1 | 3 | 0 | 3 | **In scope for B23** as a doc + tripwire rider; `contracts/version.py` + `inference/models.py` half **outstanding** (§9). A rename is a disproportionate follow-up. |
| **R32** `batch_size` neutrality is backend-dependent | 1 | 2 | 2 | 3 | 2 | **Tracked (docstring note in B23).** Neutral on the stub path; a static-shape (ONNX/TensorRT) or cross-frame backend re-opens it. `test_batch_size_is_output_neutral` is the tripwire. |

### 8.1 Ranking commentary

- **R13 / R24 / R23** dominate on correctness and are all blocked on the same
  thing (SCRFD + a real video face stage). B23 does not touch them and must say
  so (§8.2 of the original design; retained).
- **R29** is the batch: it is the single highest-reproducibility finding that is
  *actionable without a version-constant bump or a pixel change*.
- **R30 / R31** are the next reproducibility tier and each needs a deliberate
  version bump (a decode-identity value extension; `CALIBRATION_WITNESS_VERSION`
  1.3.0). They get their own batches.
- **R28** is the one finding that *could* ride a `.f.`-moving batch but must not
  — it moves pixels.
- **R32 / R33 / R34** are the three new findings; none is a code defect today,
  all three are documentation / tripwire / operational-note work, and R33's
  `contracts/version.py` half is the only piece of B23's design contract not yet
  in the tree.

---

## 9. Can B23 Be Implemented Now?

### **YES — WITH DESIGN CHANGES.**

The architecture is sound and was never in doubt: a required-fields
`VideoSelectionPolicy`, folded into `.f.` via `selection_identity(face, video)`,
`batch_size` excluded, one cache generation, no contract / schema / witness
bump, scored output unchanged for the shipped policy. The seven blocking items
are **all specification fixes, none an architecture change**, and this document
resolves every one:

| Defect | Resolved by | In the shipped tree? |
|---|---|---|
| 1 — worker cycle | env-only gate + optional arg; graphs in §2.1; residual documented | **Yes** |
| 2 — validation vs shim | keep validation, scope the claim, enumerate call sites, rewrite FB2 | **Yes** |
| 3 — `.f.` drift / R33 | names stay + docs change + 3 tripwires; inventory extended | **Partial — see below** |
| 4 — pin inventory | rebuilt in §6 (adds `stub_handlers.py`, the semantic rewrites, the `contracts` docstrings) | **Yes** (bar the R33 rows) |
| 5 — output-identical | four-part decomposition in §7 | **Yes** |
| 6 — OpenAPI | proven from `backend/openapi.py:209-251` | **Yes** |
| 7 — export claim | P19 dropped; no exports added | **Yes** |

### The one residual before B23's design contract is fully met

**R33 documentation + tripwires (Defect 3, second half).** The shipped tree
updated `inference/preprocess.py`, the pipeline comments and the `docs/*` files,
but **not**:

1. `contracts/version.py` — the module docstring `face` paragraph, the
   `FACE_SELECTION_MARKER` docstring, the `ModelVersionParts` docstring, and the
   `face_selection_identity_of` / `is_face_selection_pinned` docstrings still
   describe a face-only segment;
2. `inference/models.py` — `ModelSpec.face_selection_fingerprint` and
   `with_face_selection` docstrings likewise;
3. the three tripwires (PA10, PA11, SA8) that would fail if the docs drift
   further or a partial engine mismatch went unnamed.

This is **documentation and test-only work with no identity, version, wire or
pixel impact**. It can land as a small B23 follow-up commit. Until it does,
`contracts/version.py` is internally self-contradictory and Route A's
log-line-legibility cost (§4.3) is un-mitigated.

### Recommendation

Treat B23 as **implemented and green**, with a **required follow-up** carrying
the R33 doc + tripwire rider (items 1–3 above), and **two tracked follow-ups**:
R30 (image decode identity — own cache generation), R31
(`CALIBRATION_WITNESS_VERSION` 1.3.0). Enter R32 (docstring note — fold into the
R33 follow-up), R33, R34 in `REPRODUCIBILITY_AUDIT.md` as tracked-not-closed.
Add the R34 rolling-deploy warning to the B23 CHANGELOG entry.

**YES — WITH DESIGN CHANGES (this document; R33 rider outstanding).**
