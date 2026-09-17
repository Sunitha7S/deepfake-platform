# Batch 23 — R3 Design (final, post-gate)

**Supersedes:** `BATCH_23_R2_DESIGN.md`.
**Closes:** every item in `B23_R2_FINAL_GATE.md` §7 (12 items) plus the
follow-ups on Defects 2, 3, 4 and finding R35.
**Prior verdict:** `B23_R2_FINAL_GATE.md` → **APPROVE WITH CHANGES**, no
architecture change required, no implementation authorised.
**Goal of R3:** fold every remaining item into the design so it reaches an
**unconditional APPROVE**. No "confirm during implementation" language survives.

---

## 0. Relationship to R2

R3 is a **delta** on R2. Everything in R2 that the gate review marked *sound*
carries forward unchanged: the architecture, the Route-A decision, the identity
movement (`.f.` `251d22e9` → `e629642c`, composite), the one-cache-generation
count, the parser strategy (no new era), the fail-before / pass-after /
collateral / parser / startup matrices, and §6.1–§6.3 of the pin inventory.

R3 changes or adds only what §1–§6 below specify. Where a section number is
given (e.g. "R2 §2.3"), R3 replaces that section's text with the text here.

Every claim was re-verified against the pre-B23 tree at the cited paths.

---

## 1. Resolution of the 12 gate items

| # | Gate item (quoted) | Kind | R3 §ref |
|---|---|---|---|
| 1 | "Replace Defect-2's 'confirm at implementation' with the verified fact" | doc fix | §2 |
| 2 | "Commit to `selection_identity_of` / `ModelSpec.with_selection` aliases **in B23**; strengthen PA10 to assert the docstring banner; soften the `contracts/version.py:47-53` legibility promise" | spec decision | §3, §6.4, §6.2 |
| 3 | "Fold in the version-constant sweep + result (`effective_model_version: str` unconstrained ⇒ no bump)" | doc fix | §4.1 |
| 4 | "State the fixture `model.version == "1.0.0-dev"` fact; confirm N4 regen touches only `sampling`" | doc fix | §4.2 |
| 5 | "Enumerate and classify every `251d22e9` doc occurrence" | doc fix | §4.3 |
| 6 | "Restore a consolidated **§6.4 Rollback** subsection" | spec restore | §6.1 |
| 7 | "Add the R34 corollary (hatch ⇒ B20/B22 semantics)" | doc fix | §6.1 |
| 8 | "Resolve `Settings.skip_version_alignment` — delete it or bind it to the future composition root" | spec decision | §1.8 |
| 9 | "State the `selection_identity` non-injectivity sentence; fix the §4.1→§7 dangling reference" | doc fix | §1.9 |
| 10 | "Add missing tests: PA12, symmetric-defaulting, SA10, MG5, the 'permanent' carry-comment assertion; add the API-format-check blast-radius enumeration" | spec addition | §6.2 |
| 11 | "Add R35 (tripwire on `build_video_analysis` defaults) to the plan and the risk register" | spec addition | §5, §6.3 |
| 12 | "Name option 2 (backend composition root) as a scheduled follow-up, not 'a future … may'" | spec decision | §1.12 |

Items 1–7, 10, 11 are resolved in the sections referenced. Items 8, 9, 12 are
short and resolved inline here.

### 1.8 `Settings.skip_version_alignment` — **removed from B23**

**Issue (quoted):** *"`backend/config.py:227` defines the field; the worker gate
reads `$DFP_SKIP_VERSION_ALIGNMENT` from the environment because it cannot import
`backend`. … A `Settings` field that nothing reads is a hazard."*

**Resolution.** B23 does **not** add `Settings.skip_version_alignment`. The
worker gate cannot read it (Batch-0 layering forbids `workers → backend`), so it
would be dead config: an operator setting `skip_version_alignment=true` in
`Settings` on today's env-only worker gets no effect. The **only** escape hatch
B23 ships is the environment variable `DFP_SKIP_VERSION_ALIGNMENT`, which
`workers.runner.verify_engine_alignment_at_startup` reads directly from
`os.environ` (truthy set `{"1","true","yes","on"}`, `WARNING` logged every boot).

When the option-2 follow-up lands (§1.12), it introduces
`Settings.skip_version_alignment` **wired** — the `backend`-side composition root
reads it and passes it into the gate. Adding it there, with a consumer, is the
right time.

**Affected section:** R2 §2.1 "`Settings.skip_version_alignment` — an unwired
field" paragraph is **deleted**. R2 §2.1 P21 (which added the field) is
**removed** from the deliverables. The `backend/config.py` change list for B23
is now: `STUB_ENGINE_VERSION` literal + docstring only.

**New-risk check:** removing an unshipped field introduces nothing. The env hatch
is fully specified and tested (SA4).

### 1.9 `selection_identity` non-injectivity — the sentence, and the fixed reference

**Issue (quoted):** *"B23-R2 §4.1 mentions it but points to '§7' for the
argument, and **§7 does not contain it** — a dangling reference and an unstated
assumption."*

**Resolution.** R2 §4.1's parenthetical "(acceptable … §7)" is replaced by this
sentence, which also goes into `docs/VIDEO_SELECTION_IDENTITY.md` and the
`selection_identity` docstring:

> `selection_identity(face, video) = fingerprint({"face": face.identity,
> "video": video.identity})` is a hash of two hashes and is **not injective**
> over the `(face, video)` pair — two distinct policy pairs could in principle
> collide on the 8-hex `.f.` value. This is acceptable for exactly the reason
> every `fingerprint()` in the platform is (`contracts/version.py:107-115`): it
> is a 32-bit **change detector** across the handful of policies a deployment
> ever holds, not a security boundary, and nothing here defends against a
> deliberately constructed collision. The composition adds no collision risk
> beyond what `FaceSelectionPolicy.identity` and `VideoSelectionPolicy.identity`
> already carry individually.

**Affected sections:** R2 §4.1 (Route A "Future extensibility" row) and R2 §7
gain this sentence; the "(acceptable … §7)" dangling pointer is deleted.

### 1.12 Option 2 (backend composition root) — a **scheduled follow-up**

**Issue (quoted):** *"B23-R2 §2.3: 'a future batch **may** add …'. For a *final*
design this should be a decision, not a maybe."* (The gate applied the same
point to option 2 in §7 item 12.)

**Resolution.** The `backend`-side composition-root module is **Batch 24, item
OR1-b** (named, not "may"):

> **B24 / OR1-b.** Add `backend/worker_main.py` — the process entry point that
> imports `backend.config` and `workers`, builds the workers via
> `workers.registry.build_all_workers`, and calls
> `run_with_shutdown(runner, configured_model_version=get_settings().model_version,
> skip_version_alignment=get_settings().skip_version_alignment)`. `backend →
> workers` already exists (`backend/app.py:35-36`), so this module adds no
> cycle. It closes the Defect-1 residual: the full cryptographic alignment check
> then runs on the config-default path too, not only when `$DFP_MODEL_VERSION`
> is set. It re-introduces `Settings.skip_version_alignment` (§1.8) with a real
> consumer.

B23 ships the gate with the `configured_model_version` parameter already present
(default `None`), so B24/OR1-b is a **pure additive wiring change** — no gate
rewrite. Until B24, production **must** set `DFP_MODEL_VERSION` in the
environment (12-factor); this is now a line in the B23 deploy runbook (§6.1).

**Affected section:** R2 §2.1 "If fail-closed on the full check is wanted …"
subsection — "option 2 … out of B23 scope and recorded as follow-ups" becomes
"option 2 is **B24/OR1-b**, specified above". R2 §9 "two tracked follow-ups"
becomes "three": R30, R31, **OR1-b**.

---

## 2. Defect 2 follow-up — the validation call-site proof, stated in full

**Issue (quoted, R2 §2.2):** *"`test_frame_count_reproducibility.py:149` passes
`n_frames=n_frames` from a parametrised list … **Confirm at implementation that
every parametrised `n_frames` there is ≥ 8.**"*

**This sentence is deleted.** It is replaced by the following, which is the
complete design statement — no verification is deferred:

### 2.1 The `VideoSelectionPolicy.__post_init__` invariant

`1 <= min_usable_frames <= n_frames`, `n_frames >= 1`, `pool_divisor >= 1`.
Retained in B23 (rationale: R2 §2.2 — it centralises guards previously duplicated
across `extract_frames` / `topk_indices` / the pipeline, and
`min_usable_frames > n_frames` can only ever mean unconditional abstention, so
converting a silent effect into a loud `ValueError` is a strict improvement in
operator feedback).

### 2.2 Every construction path that reaches the invariant, enumerated

Verified against the pre-B23 tree. There are **no** parametrised `n_frames`
lists that reach a policy; the only list of "budgets" is in
`frame_count_reproducibility_study.py:134`, a **print-only study** that calls
`extract_frames` (not a policy) — the study is unaffected.

| Construction | Value(s) | Invariant check | Result |
|---|---|---|---|
| `VideoPipeline()` (shipped) | `default_video_selection()` = `(48, 8, 8, "uniform", "topk_mean")` | `1 ≤ 8 ≤ 48` | ok |
| `VideoPipeline(n_frames=48)` — `test_inference.py:255,268,283,553`, `test_timing_contract.py:173`, `test_video_branches.py:106` | shim → `replace(default, n_frames=48)` = `(48, 8, 8, …)` | `1 ≤ 8 ≤ 48` | ok |
| `VideoPipeline(n_frames=16)` — `test_branch_b_semantics.py:195`, `branch_b_semantics_study.py:196`, `frame_count_reproducibility_study.py:261`, `test_headline_and_face_semantics.py:211` | `replace(default, n_frames=16)` = `(16, 8, 8, …)` | `1 ≤ 8 ≤ 16` | ok |
| `VideoPipeline(n_frames=8)` — `test_timing_contract.py:170` | `(8, 8, 8, …)` | `1 ≤ 8 ≤ 8` | ok |
| `VideoPipeline(min_usable_frames=8)` — `test_inference.py:292` | `replace(default, min_usable_frames=8)` = `(48, 8, 8, …)` | `1 ≤ 8 ≤ 48` | ok |
| `_run_video(data)` — `test_frame_count_reproducibility.py:306, 385, 445` | `def _run_video(data, *, n_frames: int = 16)` — **all three callers use the default**, no override → `VideoPipeline(model_name="v", registry=reg, n_frames=16)` → `(16, 8, 8, …)` | `1 ≤ 8 ≤ 16` | ok |
| `VideoPipeline(batch_size=B)` — `test_frame_extraction.py:432`, `test_inference.py:553` | `batch_size` is not a policy field; `base` = default | `1 ≤ 8 ≤ 48` | ok |
| `test_video_selection_identity.py` direct `replace(SHIPPED, …)` — `n_frames ∈ {9, 24, 48}`, `min_usable_frames ∈ {4, 8, 9, 13}`, `pool_divisor ∈ {4, 8}` | one field varied at a time; `min_usable_frames=13` only ever paired with `n_frames=16` (rewritten FB2) | every pair satisfies `1 ≤ muf ≤ n_frames` | ok |
| `test_video_selection_identity.py::test_the_policy_validates_its_invariants` — `VideoSelectionPolicy(0,0,1,…)`, `(8,9,1,…)`, `(8,4,0,…)` | deliberately invalid | each raises `ValueError` | ok — this is the pin |

`extract_frames(data, n_frames=N)` sites (`test_decode_determinism.py:377`
`n_frames=8`, `:136` `n_frames=12`, `test_preprocess.py:390` `n_frames=16`,
`test_frame_extraction.py` many, etc.) do **not** construct a
`VideoSelectionPolicy` — `extract_frames` takes a bare `n_frames` argument
(`inference/preprocess.py:1099`) and is unchanged by B23. They are listed here
only to confirm they are out of scope.

### 2.3 The behaviour change, scoped

`min_usable_frames > n_frames` (constructible before B23; always abstained) is
now a construction-time `ValueError`. **No shipped or test call site produces
this combination** (table above). "Output-identical" is scoped to
`default_video_selection()` through a fixed input (R2 §7). The behaviour change
has **zero blast radius** and is a strict improvement; it is pinned by
`test_the_policy_validates_its_invariants`.

### 2.4 FB2, rewritten

Original FB2 mechanism (`n_frames=9, min_usable_frames=16`) is unconstructible
under §2.1. Rewritten: a 12-frame clip, `n_frames=16` (decodes 12), floor `8`
decides / floor `13` abstains — both `≤ 16`, both constructible. Same finding
(`min_usable_frames` moves the model identity **and** still gates
verdict→abstention). This is `test_r29_min_usable_frames_moves_the_identity`.

**Affected section:** R2 §2.2 — the "confirm at implementation" sentence and the
`test_frame_count_reproducibility.py` row are replaced by §2.2's table above.

---

## 3. Defect 3 follow-up (R33) — the honest-name aliases

**Gate conclusion (quoted):** *"the audit is complete, the tripwires are good,
but Route A should introduce a proper alias."*

### 3.1 What B23 adds to `contracts/version.py` — **no parser change**

Two new public names, each a thin wrapper over the existing implementation. The
parser (`parse_model_version`, `_MODEL_MARKERS`, `ModelVersionParts`,
`compose_model_version`) is **untouched** — these call it, exactly as the
existing functions do.

```python
def selection_identity_of(version: str) -> str | None:
    """The selection-policy fingerprint carried in the ``.f.`` segment, or
    ``None`` if the version claims none.

    Since Batch 23 (R29) the ``.f.`` segment names the *combined* selection
    policy: face selection for the image path, and face selection plus frame
    selection and pooling for the video path. This is the honest name for what
    ``.f.`` carries. Total on any composed version, pre- or post-Batch-23.

    For a version written before Batch 23 the value genuinely *is* the
    face-selection identity; ``face_selection_identity_of`` is the historical
    alias and returns the same value.
    """
    return parse_model_version(version).face_selection   # identical body


def is_selection_pinned(version: str) -> bool:
    """Whether ``version`` names a selection policy (face, or face + frame
    selection for video). The honest-named analogue of
    :func:`is_face_selection_pinned`."""
    return FACE_SELECTION_MARKER in version              # identical body
```

### 3.2 What happens to the existing names — kept, re-pointed, banner-documented

| Existing name | Disposition | New docstring (design-mandated wording) |
|---|---|---|
| `face_selection_identity_of(version)` | **kept as a permanent alias.** Body becomes `return selection_identity_of(version)`. | Opens with: **"SINCE BATCH 23 (R29) THIS RETURNS THE COMBINED SELECTION IDENTITY (face selection + video frame selection + pooling), NOT a `FaceSelectionPolicy.identity`. For a Batch-23+ version, compare the result to `selection_identity(default_face_selection(), default_video_selection())`, NEVER to `default_face_selection().identity`. For a pre-Batch-23 version the value is the face-selection identity and the two comparisons coincide. New code should call `selection_identity_of`."** |
| `is_face_selection_pinned(version)` | **kept as a permanent alias.** Body becomes `return is_selection_pinned(version)`. | "Alias of `is_selection_pinned`. Named for the pre-Batch-23 era when `.f.` carried only the face policy; both names test for the presence of the `.f.` segment." |
| `ModelVersionParts.face_selection` (field) | **name kept** (internal dataclass field, not a top-level public function). | Class docstring updated: "`face_selection` carries the combined selection fingerprint — face selection, plus frame selection and pooling for video (Batch 23). `None` for a version written before Batch 12." |
| `compose_model_version(face_selection=…)` (param) | **name kept** (keyword, high blast radius to rename). | Error strings and the param doc gain "(the combined selection fingerprint since Batch 23)". |
| `FACE_SELECTION_MARKER` (`.f.`) | **constant name kept.** | Docstring updated per §6.4. |
| module docstring `face` paragraph + format diagram (`:33-53`) | text updated per §6.4. | — |

### 3.3 What B23 adds to `inference/models.py` — one alias

```python
def with_selection(self, selection_fingerprint: str) -> "ModelSpec":
    """Return this spec bound to a selection-policy fingerprint. The honest-named
    analogue of :meth:`with_face_selection`; since Batch 23 the pipelines pass
    ``selection_identity(face, video)`` here."""
    return self.with_face_selection(selection_fingerprint)   # delegates
```

`with_face_selection` is kept as a permanent alias; its docstring gains the
"since Batch 23 this receives the combined selection identity" note.
`ModelSpec.face_selection_fingerprint` (field) name is kept; docstring updated.

The pipelines (`inference/image_pipeline.py`, `inference/video_pipeline.py`)
call `.with_selection(pre.selection_identity(self._selection,
self._video_selection))` — the honest name at the new call sites.

### 3.4 Relationship, stated explicitly

- `selection_identity_of(v)` **≡** `face_selection_identity_of(v)` for **every**
  `v` — same return value, always. The difference is only the name and the
  docstring's framing.
- `is_selection_pinned(v)` **≡** `is_face_selection_pinned(v)` for every `v`.
- `ModelSpec.with_selection(fp)` **≡** `ModelSpec.with_face_selection(fp)`.
- `selection_identity(face, video)` (in `inference.preprocess`) **produces** the
  value; `selection_identity_of(version)` **reads it back** from a composed
  string. `selection_identity_of(engine.model_version) ==
  selection_identity(default_face_selection(), default_video_selection())` for
  the shipped configuration — this is the tripwire (§6.2, PA10).

### 3.5 Migration and deprecation posture

- **No runtime `DeprecationWarning`.** The old names are not *wrong* — for a
  pre-B23 version string `face_selection_identity_of` is exactly correct, and
  for a B23+ string it returns the right value under a now-broad name. A warning
  would be noise on every call.
- **Soft deprecation:** the docstrings state the preference ("new code should
  call `selection_identity_of`"); the two old names are retained **indefinitely**
  with no removal milestone (removing them is a bigger, later change than B23
  and is not scheduled).
- **New call sites** — the pipelines, the B23 test suite, `docs/*` — use the
  honest names.
- **Existing call sites** (`test_face_size_policy.py`,
  `test_face_selection_identity.py`, studies) are updated to the honest names
  **only where the assertion is already being rewritten** for the composite
  (R2 §6.2); a pure literal-bump site keeps the old alias to minimise churn.

### 3.6 Documentation inventory (R2 §5.7 / §6, extended)

Add to the doc-update list: `contracts/version.py` (module docstring `face`
paragraph and format diagram `:33-53`; `FACE_SELECTION_MARKER` docstring `:119`;
`ModelVersionParts` docstring `:239`; `face_selection_identity_of` `:446` and
`is_face_selection_pinned` `:455` docstrings; the two new functions' docstrings;
the `compose_model_version` param doc and error strings `:266,286,304`) and
`inference/models.py` (`ModelSpec.face_selection_fingerprint` `:150`;
`with_face_selection` `:318`; the new `with_selection`).

**Affected sections:** R2 §2.3 "Decision: names stay, docs change, add tripwires"
— the "Follow-up (not a blocker)" paragraph is replaced by §3 above (the aliases
are **in B23**, not deferred). R2 §4.3 (Route A "conditionally") is updated per
§6.4.

---

## 4. Defect 4 follow-up — the exhaustive pin inventory

R2 §6.1–§6.3 carry forward. R3 adds §4.1 (version-constant sweep), §4.2 (fixture
treatment), §4.3 (documentation-literal classification), completing the
inventory.

### 4.1 Version-constant sweep — done, not assumed

Every `_VERSION`-style constant in the tree, and whether it governs a structure
that carries `model_version` / `effective_model_version` / `.f.` under a
**format, length or regex constraint**:

| Constant | Location | Governs | Constrains a version-carrying string? |
|---|---|---|---|
| `CONTRACT_VERSION = "1.0.0"` | `contracts/enums.py:16` | `EvidenceBundle.schema_version` default; OpenAPI `x-contract-version` | **No.** `ModelInfo.version` is `Field(description=…)` only — no `pattern` / `min_length` / `max_length` / validator (`contracts/common.py:494`). Verified: `grep "pattern=\|min_length\|max_length\|regex" contracts/*.py` → no hit on any version/model field. |
| `MODEL_VERSION = "1.0.0-dev"` | `contracts/fixtures.py:62` | the **synthetic** spec that renders `fixtures_out/*` | It *is* a model_version, but synthetic; it is **not** `STUB_ENGINE_VERSION` and B23's `.f.` move does not touch it (see §4.2). |
| `RUN_SCHEMA_VERSION = "1.2.0"` | `evaluation/benchmark.py:108` | `BenchmarkRun.schema_version` | **No.** `BenchmarkRun.effective_model_version: str` — bare, no `Field`, no validator (`evaluation/benchmark.py:157`). Its `.d.` decode half is a separate field by design (`:215-241`). |
| `CALIBRATION_WITNESS_VERSION = "1.2.0"` | `evaluation/calibration_protocol.py:74` | the witness digest schema | **No.** The witness digest membership **excludes** `effective_model_version` / `input_size` / gate / `.f.` — that omission is finding **R31**, tracked, not closed. A `.f.` value change moves no stored witness digest. |
| `SCHEMA_VERSIONS_WITH_DECODE = frozenset({"1.2.0"})` | `evaluation/calibration_protocol.py:107` | which witness schema versions carry a decode key | **No.** A set-membership test, not a format constraint on any version string. |
| `LOCK_VERSION = "1.1.0"` / `SUPPORTED_LOCK_VERSIONS = ("1.1.0",)` | `evaluation/split_lock.py:81,93` | the split-lock file format | **No.** `grep "model_version\|effective_model\|model_identity\|face_selection\|\.f\." evaluation/split_lock.py` → **zero hits**. The split lock carries no model identity of any kind. |

**Conclusion (design statement, not a to-do):** every carrier of a model version
in this platform — `ModelInfo.version`, `BenchmarkRun.effective_model_version`,
`JobSpec.model_version`, `Settings.model_version` — is an **unconstrained
`str`**. No version constant governs the *shape* of any of them. B23 therefore
bumps **no** version constant: not `CONTRACT_VERSION`, not `RUN_SCHEMA_VERSION`,
not `CALIBRATION_WITNESS_VERSION`, not `LOCK_VERSION`. This is established by
inspection above, not by assertion.

**Affected section:** R2 §6 gains this as **§6.5 "Version-constant sweep"**; R2
§8.5 ("Expected schema / OpenAPI / contract impact") gains the sentence "and no
`_VERSION` constant governs a model-version-carrying structure — swept in §6.5".

### 4.2 Fixture pin treatment

**Fact (corrects R2's inherited imprecision):** `fixtures_out/video_*.json` **do**
carry a `model.version` — verified `fixtures_out/video_authentic.json:293` →
`"model": {"version": "1.0.0-dev"}`. It comes from `contracts/fixtures.py:62`
`MODEL_VERSION = "1.0.0-dev"`, the synthetic spec value, **not**
`backend.config.STUB_ENGINE_VERSION`.

**Consequence for B23:**

| Fixture element | B23 effect |
|---|---|
| `fixtures_out/video_*.json` → `model.version` (`"1.0.0-dev"`) | **unchanged.** B23 moves `STUB_ENGINE_VERSION` (the real deployed literal); the fixtures use a synthetic spec whose version is `"1.0.0-dev"` and stays that way. |
| `fixtures_out/video_authentic.json:88`, `video_insufficient_evidence.json:71`, `video_manipulated.json:88` → `"sampling"` | `"uniform+adaptive"` → `"uniform"` (N4). This is the **only** field the fixture regen changes. |
| `contracts/fixtures.py:515` — `VideoAnalysis(sampling="uniform+adaptive")` | → `sampling="uniform"` (source of the regen). |
| `tests/stub_handlers.py:433` — `VideoAnalysis(sampling="uniform+adaptive")` | → `sampling="uniform"` (R2 §6.1, kept). |

**Regeneration statement (no deferral):** `python -m contracts.fixtures` after
the N4 source edits produces a diff confined to the `sampling` value in the three
`video_*.json` files. `model.version` in every fixture is untouched. If the diff
touches anything else, halt — a synthetic spec picked up a real literal, which is
a bug.

**Affected section:** R2 §5.5 regeneration table's `fixtures_out/*.json` row gains
"only `sampling` changes; `model.version` stays `"1.0.0-dev"` (synthetic) —
confirmed, not to be re-confirmed at implementation".

### 4.3 Documentation-literal classification — every `251d22e9` occurrence

`grep -rn "251d22e9" docs/ REPRODUCIBILITY_AUDIT.md` (excluding the B23 design /
review / gate documents themselves), each classified:

| Location | What `251d22e9` denotes there | Action |
|---|---|---|
| `docs/CROP_IDENTITY.md:196, 244, 245, 555, 587` | the **face-policy identity** (`default_face_selection().identity`) — **unchanged by B23** | **leave** — still literally correct |
| `docs/CROP_IDENTITY.md:13` | B22 prelude blockquote: "shipped identity `…f.251d22e9`" (model identity) | **add a one-line B23 pointer** to the prelude: "→ Batch 23 (R29) moved `.f.` to the composite `e629642c`; see §16 / `docs/VIDEO_SELECTION_IDENTITY.md`" |
| `docs/CROP_IDENTITY.md:557-558`, `REPRODUCIBILITY_AUDIT.md:3170-3173` | B22's own before/after table (`19559387` → `251d22e9`) | **leave** — B22 history, correct-for-then (B22-audit convention) |
| `docs/FACE_SELECTION_IDENTITY.md:8` | B22 prelude, face identity + weights fold | **leave** — face identity; `:17` already carries the B23 composite note |
| `docs/FACE_SELECTION_IDENTITY.md:17` | already updated: "the `.f.` fingerprint is now the composite `e629642c`" | **done** |
| `docs/VIDEO_SELECTION_IDENTITY.md:7, 140, 142, 144, 145` | B23 doc's own before/after table; `:140` face identity unchanged | **done** — this is the B23 report |
| `REPRODUCIBILITY_AUDIT.md:115` (R25 row), `:3170-3173` | B22 closure record | **leave** — B22 history |
| `REPRODUCIBILITY_AUDIT.md:119` (R29 row), `:3313`, `:3407-3412` | B23 section — `:3407` face identity unchanged, `:3409-3412` composite before/after | **done** — B23 section |
| `docs/BRANCH_B_SEMANTICS.md:6` | prelude blockquote: "`…f.251d22e9`; `test_no_identity_string_moves`" (model identity) | **add the one-line B23 pointer** to the prelude |
| `docs/CROP_GEOMETRY.md:9` | prelude blockquote: "`…f.251d22e9`, one cache generation" (model identity) | **add the one-line B23 pointer** |
| `docs/FACE_SIZE_POLICY.md:6, 8` | prelude blockquote: "`.f.` → `251d22e9`, weights … → `wacdd9240` … `…f.251d22e9`" (model identity) | **add the one-line B23 pointer** |
| `docs/HEADLINE_AND_FACE_SEMANTICS.md:5` | prelude blockquote: "`…f.251d22e9` (crop-geometry knobs into …)" (model identity) | **add the one-line B23 pointer** |

**Design statement:** exactly **five** docs (`BRANCH_B_SEMANTICS.md`,
`CROP_GEOMETRY.md`, `FACE_SIZE_POLICY.md`, `HEADLINE_AND_FACE_SEMANTICS.md`,
`CROP_IDENTITY.md`) carry a B22-era prelude blockquote whose model-identity
literal (`…f.251d22e9`) B23 supersedes; each gets a single-line forward pointer
to the B23 composite `e629642c`. Every other `251d22e9` in the docs is either the
face-policy identity (unchanged, correct) or a B22 before/after history entry
(left per convention). No prose is rewritten; no B22 record is altered.

### 4.4 Remaining B22 identity references — nothing else moves

`grep -rn "19559387\|w7f45fada\|5589e93e\|-iecf7148f"` (pre-B22 / rejected-B21
literals): all occurrences are in B21/B22 before/after tables and
superseded-marker blockquotes (`docs/CROP_IDENTITY.md`,
`docs/CROP_GEOMETRY.md`, `REPRODUCIBILITY_AUDIT.md` B21/B22 sections). **None is
a live pin.** B23 does not touch them. `test_evaluation_identity.py:75`
`MODEL_ID = "stub-0-w7f45fada+c.739c3fec.g.ae232cba"` — pre-B22 weights, **no
`.f.` segment** — a deliberately frozen model-only fixture for evaluation-identity
tests (which assert eval identity uses model-only, no selection segment).
**Design statement:** do **not** bump it; it is not tracking the live literal.
Add a comment at that line: "frozen pre-B22 model-only fixture — not the shipped
`STUB_ENGINE_VERSION`; do not update on identity moves."

**Affected section:** R2 §6.1's final `docs/*` row is replaced by §4.3's table;
R2 §6.4 gains §4.1 + §4.4.

---

## 5. R35 — `build_video_analysis` hardcoded defaults

**Finding (from `B23_R2_FINAL_GATE.md` §6):** `inference/postprocess.py:649-650`
— `build_video_analysis(*, …, sampling: str = "uniform", pooling: str =
"topk_mean")`. The wire value is *supposed* to be sourced from
`VideoSelectionPolicy` (R29/N4), but the function keeps its own hardcoded
defaults. If `default_video_selection().sampling_rule` / `.pooling_rule` ever
change, this fallback becomes a **stale second source of truth** for
`VideoAnalysis.sampling` / `.pooling`, and no test holds the two equal.

### 5.1 Options considered

| Option | Assessment |
|---|---|
| **Make the kwargs required** (`sampling`, `pooling` positional-or-keyword, no default) | Cleanest — one source of truth. But forces migrating every direct `build_video_analysis` caller in the test tree (`test_postprocess.py` and any study), and the pipeline is the only production caller. More churn than B23 warrants for a Low finding. |
| **Add a re-derivation assertion** (a test recomputes the pooled mean / sampled indices and checks the published `sampling`/`pooling` names match) | This is already PA2 / PA3 (the label re-derivation tests) — they hold the *names* to the *behaviour*. They do **not** hold the `build_video_analysis` *default* to `default_video_selection()`. |
| **Add a tripwire** — one test asserting the function's defaults equal `default_video_selection().sampling_rule` / `.pooling_rule` | 2-line test, catches the exact drift, no call-site churn. |
| **Defer** | The two-sources situation is *created by B23* (before B23 there was one hardcoded site; B23 adds `default_video_selection()` as the intended authority). Deferring ships the hazard. |
| **Classify as advisory** | `sampling` / `pooling` are on the wire and in the `.f.` identity input — not advisory-only like the N3 heuristics. |

### 5.2 Decision: **add the tripwire, in B23**

> **PA14** (new, `tests/test_video_selection_identity.py`):
> `test_build_video_analysis_defaults_match_the_shipped_policy` —
> ```
> import inspect
> sig = inspect.signature(post.build_video_analysis)
> assert sig.parameters["sampling"].default == pre.default_video_selection().sampling_rule
> assert sig.parameters["pooling"].default == pre.default_video_selection().pooling_rule
> ```
> Rationale: the pipeline always passes explicit values sourced from the policy,
> so `build_video_analysis`'s defaults are exercised only by study / test
> callers and can rot undetected. This test fails the moment
> `default_video_selection()` and the function default disagree, forcing them
> back into agreement. Making the kwargs required is the eventual cleanup
> (recorded as a B24 candidate); the tripwire is the proportionate B23 fix.

Also: `inference/postprocess.py:651-654` block comment (which already says "The
defaults name the shipped values for study / test callers") gains one line:
"held equal to `default_video_selection()` by
`test_build_video_analysis_defaults_match_the_shipped_policy` (R35)."

### 5.3 Risk-register entry

**R35 — `build_video_analysis` retains hardcoded `sampling` / `pooling` kwarg
defaults duplicating `default_video_selection()`.**
Severity (five-axis): Correctness 1 · Reproducibility 2 · Migration 1 ·
Cache-gen 1 · Bug-likelihood 2 → **Low**.
Status: **Closed in B23** by tripwire PA14; the required-kwarg cleanup is a B24
candidate. Enter in `REPRODUCIBILITY_AUDIT.md` B23 section.

**Affected sections:** R2 §6.3 (helpers table — `build_video_analysis` row gains
"PA14 holds its defaults to `default_video_selection()`"); R2 §8 risk register
gains the R35 row (§6.3 below).

---

## 6. Amended sections carried into the design

### 6.1 §6.4 Rollback (restored, updated for the fail-closed gate)

**Code rollback** (revert `backend/config.py` `STUB_ENGINE_VERSION` to
`stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9.d.a97ccb68`, revert the
`inference/` + `contracts/` changes):

- Reverted engines recompute the face-only `.f.` (`251d22e9`); `engine_versions()`
  reports `…f.251d22e9…`; the worker gate and `verify_version_alignment` pass
  against the reverted literal. **Clean.**
- The N4 `contracts/video.py` default revert (`"uniform"` → `"uniform+adaptive"`)
  is **inert** — no code reads the `Field(default=)` (the pipeline always passes
  `sampling` explicitly, and after a full revert `build_video_analysis` is back
  to its own `"uniform"` default). Fixtures regenerate back.
- `selection_identity_of` / `is_selection_pinned` / `ModelSpec.with_selection`
  disappear with the revert; no external caller depends on them within the B23
  window (they are brand-new).

**Orphaned cache entries.** Bundles written under `…f.e629642c…` during the B23
window become unreachable after rollback (their key matches no reverted engine).
**Storage cost, not correctness.** They expire under `result_ttl_s` (default 7
days). Identical to the B20 and B22 rollback profile.

**Idempotency keys.** A job whose pre-rollback attempt was keyed
`{sha256}:{…e629642c…}` is not deduplicated against a post-rollback
re-submission keyed `{sha256}:{…251d22e9…}` — one re-run, one-time cost, same as
B20/B22.

**Partial rollback** (revert code but keep the new literal, or vice versa): the
worker gate **refuses to start** (engines report X, configured literal Y). This
is the gate working as designed — it converts a silent broken cache into a loud
startup failure. Resolve by making code and literal consistent (either both B22
or both B23).

**Re-applying B23 after a rollback** restores `…f.e629642c…`; cache entries
orphaned by the rollback become reachable again if still within TTL. No
corruption in either direction.

**Affected section:** new R2 §6.4 (the R2 draft folded this into R34 / §2.1;
R3 restores it as its own subsection).

### 6.1b R34 corollary — the hatch reduces OR1 to B20/B22 semantics

R2 §3 (R34) is amended to add:

> Holding `DFP_SKIP_VERSION_ALIGNMENT=true` on every worker for the duration of
> the rolling-deploy window **bypasses the gate**, which reduces OR1's deploy
> behaviour to **exactly** B20/B22: each worker starts against whichever literal
> it is configured with, the API serves reads under its literal, and whichever
> generation the API's literal does not match produces false cache **misses**
> (re-runs) — never wrong verdicts. OR1 is therefore **not worse** than B20/B22
> during the window; it is strictly better outside it, because the misalignment
> B20/B22 left silent is now a startup failure. The hatch's every-boot `WARNING`
> log is the audit trace that it was used. The B23 CHANGELOG entry states this
> and the two safe rollout paths: (a) a coordinated worker restart at cutover,
> or (b) the hatch held on across the window and removed after.

**Deploy runbook line (B23):** "Set `DFP_MODEL_VERSION` to the new
`STUB_ENGINE_VERSION` in the worker and API environment before rollout (until
B24/OR1-b wires the config-default path into the full check). For a pod-by-pod
rolling update, either coordinate the worker restart at cutover or set
`DFP_SKIP_VERSION_ALIGNMENT=true` for the window."

### 6.2 Test-matrix additions

Added to R2 §5:

| ID | Matrix | Assertion |
|---|---|---|
| **PA12** | pass-after | `pre.default_face_selection().identity == "251d22e9"` — a standalone, named assertion inside `test_batch_23_moved_only_the_selection_identity` that the **face** policy did not move (R2 §6.1 noted it; R3 makes it a designed guard, per gate M2). |
| **PA13** | pass-after | Symmetric defaulting: for a non-default face policy `X`, `ImagePipeline(face_selection=X).model_version == VideoPipeline(face_selection=X).model_version`. Proves both pipelines default the *other* policy identically, so one configured literal still satisfies both engines when only one policy is overridden (gate §9.4 / prior review §9.4). |
| **PA15** | pass-after | The `ImagePipeline` "carries `VideoSelectionPolicy`" block-comment pin (`test_the_image_pipeline_carries_the_video_segment_without_consuming_it`) additionally asserts the comment contains **"permanent"** / **"never"** — the image path will *never* sample frames, distinguishing it from `VideoPipeline`'s temporary carry of `FaceSelectionPolicy` (which SCRFD will make live). (gate §5 / prior review §9.3) |
| **SA10** | startup-alignment | API `lifespan` rejects a **marker-bearing but unparseable** `model_version` (e.g. `"stub-0+c."` — `parse_model_version` raises `ValueError`): the `except ValueError` branch of the format check aborts startup with a message naming the string. (gate §5 / prior review §10.5) |
| **MG5** | migration | `schema_out/VideoAnalysis.schema.json`'s `sampling` default is `"uniform"`, not `"uniform+adaptive"` — so a future `python -m contracts.schema` regen cannot silently revert N4. (gate §2 Defect 6) |
| **PA14** | pass-after | `build_video_analysis` defaults equal `default_video_selection().sampling_rule` / `.pooling_rule` (R35, §5.2). |

**API-format-check blast-radius enumeration** (design statement, not a test —
gate §5 / prior review M5). Every construction of the API in the test tree
(`grep -rln "create_app(\|TestClient(" tests/` → exactly three files):

| File | How it builds the app | `lifespan` format-check exposure |
|---|---|---|
| `tests/test_startup_alignment.py` | passes `model_version="1.0.0-dev"` and a `.d.`-less string into `Settings` and **expects the `RuntimeError`** | intentional — this is the test of the check (SA6–SA10) |
| `tests/test_backend.py` | builds with default `Settings`; `:209` does `app.state.settings.model_version = "stub-1"` **after construction** | **safe** — the mutation is post-`lifespan`; `lifespan` already ran and passed against the default `STUB_ENGINE_VERSION`. No re-entry. (If this file is ever refactored to pass `model_version="stub-1"` into `Settings(...)` *before* `create_app`, `lifespan` will reject it — noted for the implementer.) |
| `tests/test_openapi.py` | `build_schema()` → `create_app(Settings(api_prefix="", docs_enabled=True, stub_scoring=True, log_json=False, log_level="CRITICAL"))`, **no `model_version` override** → the valid default | **safe** — default is composed + environment-pinned; and `test_openapi.py` calls `app.openapi()` without entering a `TestClient` lifespan context |

**Design statement:** no existing test passes a non-composed `model_version` into
`Settings` before `create_app` and expects `lifespan` to succeed. The API format
check introduces **no** collateral test breakage. The only file that needs the
check to *fail* is `test_startup_alignment.py`, which is written for it.

### 6.3 Risk register — R35 added

R2 §8's table gains one row (ranked among the ten; it sorts last):

| Finding | Corr | Repro | Migr | Cache | Bug-lik | Status & note |
|---|---|---|---|---|---|---|
| **R35** — `build_video_analysis` keeps hardcoded `sampling`/`pooling` kwarg defaults duplicating `default_video_selection()` | 1 | 2 | 1 | 1 | 2 | **Closed in B23** by tripwire PA14 (`test_build_video_analysis_defaults_match_the_shipped_policy`). Required-kwarg cleanup is a B24 candidate. The lowest-severity finding in the register. |

R2 §8.1 commentary gains: "R35 is a two-sources-of-truth hazard *created by* B23
(the batch that makes `default_video_selection()` the authority); it is closed
in-batch by the cheapest sufficient instrument, a default-equality tripwire."

### 6.4 `contracts/version.py:47-53` — the amended legibility text

R2 §2.3 listed this as "relabel the diagram." R3 specifies the full replacement
text (the *promise*, not just the label):

> The format is readable on purpose. An operator reading a log line can see which
> **segment** moved without decoding anything:
>
>     stub-0+c.1f4a9c2b.g.7e30d5a1.f.4b1d9e07.d.6f2a91cc
>     └────┘  └───────┘  └───────┘  └───────┘  └───────┘
>     weights calibration  gate    selection   decode
>
> The ``.f.`` **selection** segment aggregates every selection policy: face
> selection for the image path, and face selection **plus frame selection and
> pooling** for the video path (Batch 23, R29). When ``.f.`` moves, *which*
> policy changed is recovered from ``FaceSelectionPolicy.differences()`` /
> ``VideoSelectionPolicy.differences()`` or from
> ``verify_version_alignment``'s per-engine detail — not from the log line
> alone. This is the one place the "without decoding anything" property is
> deliberately traded away, to keep ``.f.`` a single opaque fingerprint and the
> parser (``parse_model_version``, ``_MODEL_MARKERS``) unchanged. The
> alternative — a separate ``.v.`` segment — was rejected because it adds a
> permanent parser era for a legibility gain that ``differences()`` already
> provides off the log line.

The `FACE_SELECTION_MARKER` docstring (`:119-130`) gets the parallel edit: "Named
`.f.` for the face-selection policy it carried when introduced (Batch 12, R17);
since Batch 23 (R29) it carries the combined selection identity — face selection,
plus frame selection and pooling for video. Still one opaque fingerprint; the
parser is unchanged."

### 6.5 Route A — the honest framing (R2 §4.3 amended)

R2 §4.3 said "R33's fixes are what make Route A defensible." Replace with:

> Route A permanently trades the log-line legibility of the selection
> sub-components (face vs frame-selection) for parser simplicity and a single
> `STUB_ENGINE_VERSION` literal. R33's work does not *recover* that legibility —
> an operator still cannot read face-vs-video from `.f.` alone — it makes the
> trade **explicit and tripwired** (§6.4, PA10) rather than silent, and the
> `selection_identity_of` / `is_selection_pinned` / `ModelSpec.with_selection`
> aliases (§3) give all new code the honest name. With the §6.4 doc edit and the
> §3 aliases in B23, `contracts/version.py` is internally consistent and Route A
> is the correct call: Route B's permanent parser-era tax outweighs a legibility
> property that `differences()` already provides off the log line, and face
> selection and frame selection genuinely share a lifecycle (shipped code, moved
> together by the SCRFD migration).

The recommendation is **unchanged: Route A.** No architecture change.

---

## 7. Final implementation readiness — fresh evaluation

### Prior answer

R2 §9: **"YES — WITH DESIGN CHANGES (this document; R33 rider outstanding)."**
The gate review: **APPROVE WITH CHANGES**, 12 specification-completion items,
none architecture-affecting.

### What changed in R3

All 12 gate items are now resolved as concrete design statements:

| # | Was | Now |
|---|---|---|
| 1 | "confirm at implementation" for the validation call sites | §2.2 — every path enumerated; `_run_video` defaults to `n_frames=16`, all three callers use the default, every path satisfies `1 ≤ min_usable_frames ≤ n_frames`. No deferral. |
| 2 | alias "a future batch may add" | §3 — `selection_identity_of`, `is_selection_pinned`, `ModelSpec.with_selection` are **in B23**, thin aliases, no parser change; relationship and deprecation posture defined. |
| 3 | version-constant bump asserted | §4.1 — swept; every model-version carrier is unconstrained `str`; **no** constant moves. |
| 4 | fixture `model.version` unstated / mis-stated | §4.2 — fixtures carry `"1.0.0-dev"` (synthetic); regen changes only `sampling`. |
| 5 | `251d22e9` docs "open grep" | §4.3 — every occurrence classified; exactly five preludes get a one-line B23 pointer, the rest are face-identity (correct) or B22 history (left). |
| 6 | rollback folded into R34 | §6.1 — restored as its own subsection, updated for the fail-closed gate. |
| 7 | R34 read scarier than warranted | §6.1b — the hatch reduces OR1 to B20/B22 semantics; stated, with the two safe rollout paths. |
| 8 | `Settings.skip_version_alignment` dead config | §1.8 — **removed from B23**; re-added wired in B24/OR1-b. |
| 9 | dangling §4.1→§7 reference | §1.9 — the non-injectivity sentence is written; the pointer is deleted. |
| 10 | five missing test rows + no blast-radius enumeration | §6.2 — PA12, PA13, PA14, PA15, SA10, MG5 specified; API-format-check blast radius enumerated (three files, none breaks). |
| 11 | R35 not in the plan | §5 + §6.3 — dispositioned (tripwire PA14), in the risk register. |
| 12 | option 2 "a future … may" | §1.12 — **B24/OR1-b**, named and specified; B23 ships the parameter so it is pure additive wiring. |

No new defect was introduced by these resolutions. Each is documentation, an
enumeration, a thin alias, or a one-line tripwire. Re-checked:

- §1.8 (removing an unshipped field) — subtracts nothing that anything used.
- §3 (aliases) — `selection_identity_of` has the **identical body** to
  `face_selection_identity_of` (`parse_model_version(v).face_selection`); no
  parser surface added; verified `parse_model_version` peels `.f.` as one
  `partition` (`contracts/version.py:426-432`) and a `fingerprint()` output is
  marker-free `[0-9a-f]{8}` (`:234`).
- §5.2 (PA14) — an `inspect.signature` read; touches nothing.
- §6.4 (doc text) — prose only; the parser is not mentioned except to say it is
  unchanged.

### Architecture-affecting items remaining

**None.** §1.8, §1.12 and §3 are decisions, and every decision keeps the
architecture identical:
- the aliases are additive to `contracts.version` / `inference.models` and share
  the existing implementation;
- removing `Settings.skip_version_alignment` changes no structure (nothing read
  it);
- B24/OR1-b is a new `backend`-side module importing an already-legal package
  pair (`backend → workers`, `backend → backend.config`).

The identity movement, the cache-generation count (1), the parser strategy (no
new era), the Route-A decision, the OR1 worker/API split, and the "no
`CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` bump"
conclusion are all **locked and independently verified**.

### Can implementation begin?

## **YES**

B23-R3 is the final, implementation-ready design. Every blocking finding from
`B23_ADVERSARIAL_REVIEW.md` and every specification-completion item from
`B23_R2_FINAL_GATE.md` is resolved as a concrete design statement. There are no
`"confirm during implementation"` placeholders, no deferred verifications, no
dangling references, no dead config, and no architecture-affecting open
questions. The one new finding surfaced by the gate (R35) is closed in-batch by
tripwire PA14.

The implementer's scope is fixed:

- **Production:** `inference/preprocess.py` (`VideoSelectionPolicy`,
  `default_video_selection`, `video_selection_identity`, `selection_identity`);
  `inference/video_pipeline.py` (consume the policy, legacy-scalar shim,
  composite `.f.` via `.with_selection`); `inference/image_pipeline.py` (carry
  the policy, composite `.f.`); `inference/postprocess.py` (`build_video_analysis`
  sourced labels, `topk_indices` explicit `k`); `inference/models.py`
  (`with_selection` alias + docstrings); `contracts/version.py`
  (`selection_identity_of` + `is_selection_pinned` + docstrings/text — **no
  parser change**); `contracts/video.py` (`sampling` default → `"uniform"`);
  `backend/config.py` (`STUB_ENGINE_VERSION` literal + docstring — **no new
  Setting**); `backend/app.py` (`lifespan` stdlib format check);
  `workers/runner.py` (`verify_engine_alignment_at_startup`, env-only, called
  from `run_with_shutdown`).
- **Tests:** `tests/test_video_selection_identity.py` (new — FB1–FB5, PA1–PA15,
  PR1–PR4, SA9, MG2–MG4); `tests/test_startup_alignment.py` (new — SA1–SA8,
  SA10); the collateral literal / semantic-rewrite updates in R2 §6.1–§6.2 and
  R3 §4; `tests/run_all.py` header count.
- **Generated:** `fixtures_out/video_*.json` (`sampling` only),
  `schema_out/*.json` (cosmetic `default`), `backend/openapi.snapshot.json`
  (**zero structural diff** — hard gate, proven in R2 §2.6).
- **Docs:** the R2 §5.7 list, extended per R3 §3.6 (`contracts/version.py`,
  `inference/models.py`) and R3 §4.3 (five prelude pointers); new
  `docs/VIDEO_SELECTION_IDENTITY.md`; `REPRODUCIBILITY_AUDIT.md` B23 section with
  R29 / N4 / OR1 closed and R30 / R31 / R32 / R33 / R34 / R35 tracked.
- **Follow-ups (B24):** OR1-b (`backend/worker_main.py` + wired
  `Settings.skip_version_alignment`); R35 required-kwarg cleanup (candidate);
  R30 (image decode identity); R31 (`CALIBRATION_WITNESS_VERSION` 1.3.0).

**Verdict: YES. B23-R3 is approved for implementation.**
