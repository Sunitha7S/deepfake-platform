# B23-R2 — Final Pre-Implementation Gate Review

**Object of review:** `BATCH_23_R2_DESIGN.md` (the revised design).
**Not under review:** `BATCH_23_DESIGN.md` (original) — evaluated only where R2
inherits a claim from it.
**Prior review:** `B23_ADVERSARIAL_REVIEW.md` — its seven blocking findings are
treated as hypotheses and independently re-checked below.
**Question:** does B23-R2 resolve every blocking finding, introduce no new
defect, and stand as an implementation-ready specification?

### Scope and method note

The prior review discloses that a B23 implementation exists in this repository.
Per the gate constraints, this review **does not use that implementation's test
results as evidence**. It evaluates `BATCH_23_R2_DESIGN.md` **as a written
specification**, and independently re-verifies its checkable claims against the
**B22-era, pre-B23 code** that is legitimately stable and inspectable:
`contracts/version.py`, `backend/openapi.py`, `inference/__init__.py`,
`inference/postprocess.py`, `backend/app.py`, `backend/config.py`,
`tests/test_worker_inference_integration.py`, and the `VideoPipeline` /
`extract_frames` call sites. Every criticism below cites a specific B23-R2 claim
and the file/line it was checked against.

---

## 1. Executive Verdict

### **APPROVE WITH CHANGES**

B23-R2 resolves the **architecture** questions definitively and correctly. All
seven blocking findings from the prior review are addressed at the design level;
five are fully resolved, two are resolved-with-a-documented-residual. The
Route-A recommendation survives scrutiny. No architecture-affecting defect was
introduced by the revisions.

It is **not yet an implementation-ready specification** because it:

- **defers one of its own safety verifications to implementation time** (Defect
  2, the `test_frame_count_reproducibility.py` call site) when the value is
  knowable now;
- resolves Defect 3 by keeping a **permanently misleading public function name**
  (`contracts.version.face_selection_identity_of`) with a tripwire that guards
  only doc-drift, not misuse, and defers the honest-name alias to "a future
  batch may";
- has an **incomplete pin/scope sweep** (no version-constant enumeration; an
  inherited-and-uncorrected fixture `model.version` statement; several
  `251d22e9`-bearing docs unclassified);
- **omits a consolidated rollback subsection** that the original design carried;
- contains a **dangling internal cross-reference** (§4.1 → §7) for the
  `selection_identity` non-injectivity point the prior review asked to be folded
  in, which is then never actually stated;
- leaves **`Settings.skip_version_alignment` as dead config** (the worker cannot
  read it) without resolving it;
- introduces one **new low-severity reproducibility finding** (R35 below):
  `build_video_analysis` retains hardcoded `sampling` / `pooling` kwarg defaults
  that duplicate `default_video_selection()` with no tripwire holding them equal.

None of these is a REJECT-level defect. All are specification completions. With
the changes in §7 folded in, B23-R2 becomes the final design.

---

## 2. Resolution Audit

### Defect 1 — OR1 worker gate closes a `workers → backend` dependency cycle

**Status: RESOLVED, with a documented residual on the config-default path.**

**Hypothesis re-check.** The prior review's claim is real. Verified:
`backend/app.py:35-36` — `from workers.io import StorageResultSink`,
`from workers.protocol import ResultSink` — establishes edge `backend → workers`.
`tests/test_worker_inference_integration.py:534-548` `_imports_of` AST-walks
**every** `Import` / `ImportFrom` node, function-local included;
`test_dependency_graph_stays_acyclic` (`:567-588`) restricts to the six-package
set and raises on any cycle. Any `from backend.config import …` inside `workers/`
would add `workers → backend` and the test would raise
`cycle: workers -> backend -> workers`. The original P20 required exactly that
import.

**Resolution verification.** B23-R2 §2.1 specifies the gate
(`verify_engine_alignment_at_startup`) reads `os.environ` for `DFP_MODEL_VERSION`
and `DFP_SKIP_VERSION_ALIGNMENT`, takes an **optional** `configured_model_version`
parameter, and imports only `os` (stdlib), `contracts.version`, and
`workers.engine_registry`. The before/after graphs in §2.1 are correct:

- `workers.engine_registry` already imports `inference` at module scope
  (verified: `workers/engine_registry.py` top — `from inference import
  ImagePipeline, VideoPipeline, default_registry`), and `workers → inference` is
  an existing, required edge (`test_workers_import_inference`).
- The gate adds **no new package edge**. `workers → {contracts, storage,
  jobqueue, inference}` is unchanged; `backend → workers` is unchanged;
  topological order `contracts < {storage, jobqueue, inference} < workers <
  backend` holds. Acyclic.

Batch-0 layering is preserved: the API half (`backend/app.py` `lifespan`) uses
`contracts.version` only — no `inference`, no cv2/numpy — consistent with the
`contracts/version.py` module docstring's "WHY THIS LIVES IN contracts"
rationale (`contracts/version.py:8-16`).

**Residual (correctly disclosed by B23-R2 §2.1).** The full cryptographic check
(`verify_version_alignment(configured)`) runs only when a literal is in reach —
i.e. `$DFP_MODEL_VERSION` is set or the parameter is passed. `run_with_shutdown`
calls the gate with **no argument**, and B23-R2 admits "no composition root does
yet" pass one. On a deployment that relies on `backend.config.STUB_ENGINE_VERSION`
with no env override, the gate degrades to "engines agree with each other + the
agreed string is composed and environment-pinned." B23-R2 routes the closure of
that gap to `test_configured_model_version_matches_the_engines` as a **pre-merge**
guard, and to a future `backend`-side composition root (option 2/3) as an
additive wiring change.

**Assessment of the residual.** Acceptable. The pre-merge guard makes the blind
spot unreachable on `main`; the only exposure is a *post-merge hand-edit* of the
literal without an env var, which is unusual and still partly caught by the API
format check. But B23-R2 should:
- state in the deploy runbook that production **must** set `DFP_MODEL_VERSION`
  (12-factor), which makes the full check the norm, not the exception;
- pick option 2 as a **named, scheduled follow-up**, not "a future … may".

**New risks introduced.**
1. **`Settings.skip_version_alignment` is dead config.** `backend/config.py:227`
   defines the field; the worker gate reads `$DFP_SKIP_VERSION_ALIGNMENT` from
   the environment because it cannot import `backend`. B23-R2 §2.1 flags this
   ("currently unwired for the worker") but resolves it only with "keep the field
   … add a one-line note." A `Settings` field that nothing reads is a hazard —
   an operator sets it, expects an effect, gets none. **Change required:** either
   delete the field or bind it explicitly to the future composition root and say
   so in its description.
2. **Two escape-hatch knobs for one behaviour** (`$DFP_SKIP_VERSION_ALIGNMENT`
   and `Settings.skip_version_alignment`) with no statement of precedence. Folds
   into (1).

No architecture-affecting risk.

---

### Defect 2 — the `1 <= min_usable_frames <= n_frames` validation vs the legacy shim

**Status: RESOLVED IN PRINCIPLE, but the design defers its own safety proof.**

**Hypothesis re-check.** Real. B23-R2 §2.2 keeps the `__post_init__` validation.
The original FB2 mechanism (`n_frames=9, min_usable_frames=16`) is unconstructible
under it, and `replace(default_video_selection(), n_frames=4)` →
`VideoSelectionPolicy(4, 8, …)` raises. Both are genuine consequences.

**Resolution verification.** B23-R2 §2.2:
- **Scopes "output-identical" to the shipped policy** — correct and necessary.
- **Documents the behaviour change** (`min_usable_frames > n_frames` is now a
  construction-time `ValueError`, previously a silent always-abstain) — honest.
- **Rewrites FB2** to `n_frames=16, min_usable_frames ∈ {8,13}` on a 12-frame
  clip — a construction the validation permits (`13 ≤ 16`), same finding.
- **Enumerates call sites.** The table lists `n_frames ∈ {8, 16, 48}` and
  `min_usable_frames = 8`, all safe against the default floor of 8.

**The defect in the resolution.** B23-R2 §2.2 contains this sentence:

> "`test_frame_count_reproducibility.py:149` passes `n_frames=n_frames` from a
> parametrised list … **Confirm at implementation that every parametrised
> `n_frames` there is ≥ 8.**"

A design that cannot verify its own zero-blast-radius claim is not
implementation-ready — it hands the implementer a judgment call that decides
whether the batch is safe. **This value is knowable now.** Verified:
`tests/test_frame_count_reproducibility.py:146` — `def _run_video(data, *,
n_frames: int = 16)`; its three call sites (`:306`, `:385`, `:445`) all invoke
`_run_video(data)` with **no override**, so `n_frames = 16 ≥ 8`. There is no
parametrised list of budgets in that file (the "budgets" list at `:134` is in
`frame_count_reproducibility_study.py`, a print-only study). **Change required:**
replace the "confirm at implementation" sentence with the verified fact.

**Second-order effects.** The validation runs on **every** `VideoSelectionPolicy`
construction, including every `replace()` in the shim and every direct
construction in `test_video_selection_identity.py`. Walked each:
`replace(SHIPPED, n_frames=9|24|48)` → floor stays 8, all `≥ 8` ✓;
`replace(SHIPPED, min_usable_frames=4)` → `(48, 4, 8)`, `1 ≤ 4 ≤ 48` ✓;
`test_every_policy_field_moves_the_identity` varies one field at a time, never
combining a low `n_frames` with a high floor ✓. No hidden interaction.

**Alternative not fully argued.** B23-R2 §2.2 rejects "drop the `<= n_frames`
upper bound" (option c) as "loses the centralisation benefit." That is a thin
justification for accepting a behaviour change in a batch whose mandate is
"output-identical." The design should state the stronger case for keeping the
strict validation: `min_usable_frames > n_frames` is a configuration that can
*only* mean unconditional abstention, so converting it from a silent effect to a
loud error is a strict improvement in operator feedback and the blast radius is
provably zero (once the sentence above is fixed). With that argument added, the
choice is sound.

**New risks introduced.** None beyond the deferred verification (a documentation
defect, not a system defect).

---

### Defect 3 — `.f.` semantic drift (R33)

**Status: PARTIALLY RESOLVED. The specification is complete; the chosen
resolution knowingly ships a misleading public name.**

**Hypothesis re-check.** Real and independently confirmed. `contracts/version.py`
still describes `.f.` as face-only throughout: the module docstring `face`
paragraph (`:33-40`), the readable-format promise and diagram (`:47-53`, segment
labelled "face select"), `FACE_SELECTION_MARKER` docstring (`:119-130`),
`ModelVersionParts` docstring (`:239-245`), `face_selection_identity_of`
docstring (`:446-452`, "The face-selection fingerprint"),
`is_face_selection_pinned` docstring (`:455-463`). `inference/models.py`:
`ModelSpec.face_selection_fingerprint` (`:150-164`), `with_face_selection`
(`:318-327`). After B23, `face_selection_identity_of(v)` returns
`fingerprint({"face": …, "video": …})` — not a `FaceSelectionPolicy.identity`.

**Resolution verification.** B23-R2 §2.3:
- **Audit is complete** — the nine `contracts/version.py` rows and two
  `inference/models.py` rows match what I found independently.
- **Doc-update inventory is extended** to both modules (the prior review's §5.7
  omission is fixed).
- **Three tripwires specified** — PA10 (`.f.` marker + module docstrings name
  both policies; `face_selection_identity_of(STUB) == selection_identity(...)`
  ≠ `default_face_selection().identity`), PA11 (R32 docstring note), SA8
  (partial engine mismatch).
- **Rename rejected** — ~15 call sites, deprecation-alias burden, adjacency to
  the "don't touch the parser" constraint.

**Why this is only partial.**

1. **The tripwire guards the wrong thing.** PA10 as specified checks that
   docstrings *contain* "video" / "frame selection". It does **not** prevent the
   actual defect: a caller invoking `face_selection_identity_of(v)` and comparing
   the result to `default_face_selection().identity`. The misleading name
   persists; only doc-drift is fenced.

2. **The honest-name alias is deferred with a hedge.** B23-R2 §2.3: "a future
   batch **may** add `selection_identity_of` as a thin, documented alias." For a
   *final* design this should be a decision, not a maybe. Adding
   `selection_identity_of` (aliasing `face_selection_identity_of`) and
   `ModelSpec.with_selection` (aliasing `with_face_selection`) is ~6 lines + two
   asserts, is **not** a parser change, and lets every new call site use the
   correct name immediately while the old names remain for back-compat. The prior
   review's objection — "a `contracts/` public function whose name and docstring
   lie about its return value is a defect the design ships knowingly" — is not
   answered by a doc banner alone.

3. **The legibility *promise* is not addressed, only the label.** B23-R2 §2.3
   lists `contracts/version.py:47-53` as "relabel the diagram." But those lines
   contain a *promise*: "An operator reading a log line should be able to see
   which component moved without decoding anything." Route A makes that false for
   the face-vs-video distinction (§3 below). The design must **soften the promise
   text**, not just relabel "face select" → "selection".

**Change required:** commit to the `selection_identity_of` / `with_selection`
aliases **in B23**; strengthen PA10 to assert the two public functions' docstrings
carry an explicit "never compare to `default_face_selection().identity`" banner;
edit the promise at `contracts/version.py:47-53` to carve out the aggregated
selection segment.

**New risks introduced.** None — the composite behaviour is correct; the risk is
entirely about the contract's self-description, which is what R33 tracks.

---

### Defect 4 — the pin inventory

**Status: SUBSTANTIALLY RESOLVED. Three enumeration gaps remain.**

**Hypothesis re-check.** The prior review's specific misses were real:
`tests/stub_handlers.py:433`, the `_EXPECTED_EFFECTIVE` / `MODEL_ID` collateral,
the semantic-vs-literal distinction for `face_selection_identity_of` asserts,
`contracts/version.py` docstrings.

**Resolution verification.** B23-R2 §6 rebuilds the inventory. Independently
checked against the tree:
- `tests/stub_handlers.py:433` — **folded in** (§6.1). ✓
- `_EXPECTED_EFFECTIVE` (`test_branch_b_semantics.py:462`,
  `test_headline_and_face_semantics.py:97` + refs), `MODEL_ID`
  (`test_decode_identity_completeness.py:96`) — **folded in** (§6.1). ✓
- `contracts/video.py:92`, `contracts/fixtures.py:515`,
  `fixtures_out/video_*.json` (3 files) — **folded in**. ✓
- The semantic rewrites (`test_face_size_policy.py`,
  `test_face_selection_identity.py`) are correctly separated from literal bumps
  (§6.2). ✓
- `test_evaluation_identity.py:75` `MODEL_ID = "stub-0-w7f45fada+c.…g.…"` —
  flagged **"verify only"** (§6.1). Independently: this string has pre-B22
  weights (`w7f45fada`) and **no `.f.` segment** — it is a deliberately frozen
  model-only fixture for evaluation-identity tests, not a live pin. B23-R2's
  "verify only" is the right call; it should also state the conclusion (frozen
  historical fixture, do not bump).
- Helpers-rewritten-not-bumped (§6.3): `topk_indices`, `build_video_analysis`,
  `VideoPipeline.__init__`, `ImagePipeline.__init__`, `face_selection_identity()`,
  the two startup helpers, `sample_frame_indices` (unchanged). Complete and
  correct in kind.

**The three remaining gaps.**

1. **No version-constant sweep.** The prior review's §8.1 asked for a "grep
   `_VERSION\s*=` across the tree and confirm none governs a structure that
   carries `model_version` / `effective_model_version` / `.f.`" step. B23-R2 does
   not include it. Verified independently:
   `evaluation/benchmark.py:157` — `effective_model_version: str` (unconstrained,
   no `pattern` / `constr` / length bound); `evaluation/split_lock.py`
   `LOCK_VERSION` governs the split lock, which carries no model version;
   `evaluation/calibration_protocol.py` `CALIBRATION_WITNESS_VERSION` +
   `SCHEMA_VERSIONS_WITH_DECODE` — the witness digest excludes
   `effective_model_version` (this is R31). **Conclusion: no constant needs to
   move** — but B23-R2 asserts this without the check. **Change required:** fold
   in the sweep and its result, per the prior review's §8.2 ("and there is no
   length / format / regex constraint on it").

2. **Inherited-and-uncorrected fixture statement.** The original design §5.5
   said "No `model.version` in fixtures (they use a synthetic spec) — confirm."
   That is **wrong**: verified `fixtures_out/video_authentic.json:293` —
   `"model": {"version": "1.0.0-dev"}`. The *conclusion* still holds (`1.0.0-dev`
   is synthetic, not `STUB_ENGINE_VERSION`, so the `.f.` move does not touch it),
   but B23-R2 §6.1 lists only the `sampling` value change for the fixtures and
   never states the fixture `model.version` fact. **Change required:** state
   that `fixtures_out/*` carry `model.version == "1.0.0-dev"` (synthetic) and
   that `python -m contracts.fixtures` regen changes only `sampling`.

3. **`251d22e9`-bearing docs not fully classified.** B23-R2 §6.1's last row
   lists six `docs/*` files to update. Independently, `251d22e9` also appears in
   `docs/BRANCH_B_SEMANTICS.md`, `docs/CROP_GEOMETRY.md`,
   `docs/FACE_SIZE_POLICY.md`, `docs/HEADLINE_AND_FACE_SEMANTICS.md`. Each
   reference is either the *face policy identity* (`251d22e9`, still correct) or
   a *B22-era model identity* (`…f.251d22e9`, now historical-correct-for-then per
   the B22 audit convention). **Change required:** enumerate all `251d22e9`
   occurrences and label each "face identity — unchanged", "B22 history — leave",
   or "needs superseding marker". Do not leave it as an open grep.

**New risks introduced.** None. The gaps are incompleteness, not error.

---

### Defect 5 — the "output-identical" claim

**Status: RESOLVED.**

**Hypothesis re-check.** Real — the original "byte-identical bundle" ignored
`timing` and job metadata.

**Resolution verification.** B23-R2 §7 replaces it with a six-category
decomposition (scored output / version metadata / wire metadata / timing / job
metadata / diagnostics) and an exact statement. Independently checked:
- "the server always set `"uniform"` in the real bundle already
  (`build_video_analysis` hard-coded it)" — verified: `inference/postprocess.py:649`
  — `sampling: str = "uniform"`. The `"uniform+adaptive"` lived only in the
  `VideoAnalysis` `Field(default=)`, `contracts/fixtures.py`,
  `tests/stub_handlers.py` and `fixtures_out/*`. So no real response body moves —
  correct.
- "`video.frames_scored` … guaranteed identical" — for the shipped policy,
  `n_frames` is unchanged, so the decoded/scored count is unchanged. Correct.
- "`timing` … not guaranteed and never was … depends on `batch_size`" — matches
  `test_progress_scales_with_the_batch_size`'s premise. Correct.

**Observation (not a blocking change).** §5.2 PA1 / §7 specify
`test_the_shipped_video_bundle_did_not_move` asserts `p_fake == SHIPPED_P_FAKE`
as a **hard 16-digit float equality**. This follows existing house style (the
identity fingerprints are exact, and `test_inference_determinism.py` already
relies on exact video reproduction), so it is acceptable — but it is inherently
platform/BLAS-fragile (`pool_at` is an `np.mean`). The design should note this
and consider `abs(bundle.p_fake - SHIPPED_P_FAKE) < 1e-9` as the assertion form,
consistent with the `FINGERPRINT_DECIMALS = 6` rationale in
`contracts/version.py:143-156`. Non-blocking.

**New risks introduced.** None.

---

### Defect 6 — the OpenAPI "default is stripped" assumption

**Status: RESOLVED. Independently confirmed from source.**

**Hypothesis re-check.** Real — the original design made "zero structural diff" a
hard gate while assuming, not proving, that `default` is not structural.

**Resolution verification.** B23-R2 §2.6 proves it from `backend/openapi.py`.
Independently walked:
- `structural_signature` (`backend/openapi.py:278-297`) returns exactly
  `{openapi, contract_version, paths, components}`.
- `_component_signature` (`:241-251`) builds from exactly `enum`, `required`,
  `properties` (each via `_type_of`), and `additionalProperties` **only when it
  is literally `False`**. `default` is never read.
- `_type_of` (`:209-238`) inspects `$ref`, `enum`, `const`,
  `anyOf`/`oneOf`/`allOf`, `type` (→ `array`/`object` handling), `format`.
  `default` is never read.
- For `VideoAnalysis.sampling` (`contracts/video.py:91`, `sampling: str =
  Field(default=…)`), `_type_of` returns `"string"` before and after the N4
  change; `required` is unchanged (`sampling` was always optional). The
  structural signature is byte-identical across the change.

The `_PROSE_KEYS` denylist (`:48`) is not even the operative mechanism — the
allowlist construction makes `default` invisible regardless. B23-R2's proof is
correct and the hard gate now rests on a citation. ✓

**Residual (correctly disclosed).** `schema_out/VideoAnalysis.schema.json` may
gain a `"default": "uniform"` key; B23-R2 calls it "cosmetic" and the matrix
asserts `"stub-0" not in` each schema_out file rather than pinning content.
Acceptable. A minor addition worth making: one assertion that the schema_out
`VideoAnalysis` default is `"uniform"` (not `"uniform+adaptive"`), so a future
regen cannot silently revert N4. Non-blocking.

**New risks introduced.** None.

---

### Defect 7 — the export claim (P19)

**Status: RESOLVED.**

**Hypothesis re-check.** Real. Verified: `inference/__init__.py` `__all__`
(`:60-89`) lists protocols/errors, `ImagePipeline`, `VideoPipeline`, registry
helpers, calibration helpers. It does **not** contain `FaceSelectionPolicy`,
`default_face_selection`, or `face_selection_identity`. The module docstring is
explicit: "Preprocessing and bundle construction are … deliberately absent from
this namespace."

**Resolution verification.** B23-R2 §2.7 / §1 row 7: drop P19, add no exports;
`VideoSelectionPolicy` etc. stay `preprocess`-only, reached via `from inference
import preprocess as pre`. Consistent with how `FaceSelectionPolicy` is reached
today (`inference/video_pipeline.py:52`). Correct.

**Second-order effects.** None. No consumer needs the symbol at package scope;
the pipelines and tests already use `pre.`.

**New risks introduced.** None.

---

## 3. Architecture Review

### Route A (`.f.` composite via `selection_identity(face, video)`)

Re-checked against `contracts/version.py`:
- `parse_model_version` (`:402-443`) peels `.f.` as a **single** `partition`
  after stripping `.d.` and partitioning `+c.`. A `fingerprint()` output is 8
  hex chars — `hashlib.sha256(...).hexdigest()[:8]` (`:234`) — so it is
  marker-free and `compose_model_version`'s guard
  `any(m in face_selection for m in _MODEL_MARKERS)` (`:304`) passes. **No new
  parser branch, no new era.** Confirmed.
- One `.f.` value change ⇒ one `model_version` change ⇒ one full cache miss per
  file per namespace. The B20 `.d.` precedent (video-only change regenerating the
  image namespace) is exact.

**Cost, confirmed real:** an operator reading `.f.` move in a log line cannot
tell face-policy from video-policy. `contracts/version.py:47-49` promises the
opposite ("see which component moved without decoding anything"). B23-R2 §4.3
acknowledges this and its §3.2-was-wrong critique of the original ("`input_size`
in `-w` is a sub-fingerprint an operator never reads; `.f.` vs `.v.` is
top-level log-line legibility") is correct.

### Route B (`.v.` segment)

Re-checked: a new marker in `_MODEL_MARKERS` (`:141`), a fifth
`ModelVersionParts` field (`:247-250`), a new peel step in `parse_model_version`,
a `compose_model_version` parameter, and — the genuinely permanent cost — a
"pre-B23 era" case that `test_parse_model_version_is_total_across_all_eras` and
every `.compose()` fixture must carry forever. B23-R2 §4.2's accounting is
accurate.

### Is B23-R2's recommendation (keep Route A, conditionally) justified?

**Yes.** The parser-era tax on Route B is real, permanent, and heavier than
Route A's cost. The platform's organising principle — one fingerprint per
lifecycle-distinct concern (`contracts/version.py:20-46`) — genuinely groups
face selection and frame selection: both are shipped code, both move under the
same SCRFD migration. Route A is the correct call.

**But the "conditionally" is doing more work than B23-R2 admits.** §4.3 says
"R33's fixes are what make Route A defensible." That overstates: R33's fix
(docstrings + a doc-drift tripwire) makes the legibility loss *documented*, not
*recovered* — an operator still cannot read face-vs-video from `.f.`. The honest
framing, which B23-R2 should adopt: *Route A permanently trades log-line
legibility of the selection sub-components for parser simplicity; R33's work
makes that trade explicit and tripwired rather than silent, and the
`selection_identity_of` alias (Defect 3 change) gives new code the honest name.*
With that framing and the Defect 3 changes, Route A stands.

**No architecture change is required.** The recommendation is sound.

---

## 4. Migration Review

| Item | B23-R2 claim | Verification | Verdict |
|---|---|---|---|
| **Cache generation count** | Exactly 1, spanning both image and video namespaces (§1, §4.1, §8) | Only `.f.` moves; `-w`/`+c.`/`.g.`/`.d.` unchanged ⇒ one `model_version` value change ⇒ one full miss per file per namespace. B20 precedent exact. | **Sound** |
| **Parser compatibility** | `.f.` stays one 8-hex token; no new era; `parse_model_version` untouched (§4.1, §5.5 PR1-PR4) | `contracts/version.py:402-443` peels `.f.` as one `partition`; `fingerprint` output is marker-free `[0-9a-f]{8}`; the marker-collision guard (`:304`) passes. PR3's proposed explicit assert `all(m not in combined for m in _MODEL_MARKERS)` is a good addition. | **Sound** |
| **Literal regeneration plan** | §6.1 table + "enumerate before implementing" grep | Complete for code literals. **Gaps:** version-constant sweep absent (§2, Defect 4 gap 1); fixture `model.version` unstated (gap 2); some docs unclassified (gap 3). | **Needs the three §2/Defect-4 changes** |
| **Rollback behaviour** | Discussed only inside R34 (§3) and §2.1 | The original design had a dedicated §6.4 (code rollback reverts the literal → engines report the old value → alignment passes → clean; orphaned cache entries under the new `.f.` expire under `result_ttl_s`; partial rollback is caught by the gate; the `contracts/video.py` default rollback is inert because no code reads the default). **B23-R2 drops this consolidation.** | **Change required — restore a §6.4 Rollback subsection** |
| **Mixed-deployment behaviour** | R34 (§3): fail-closed gate + `STUB_ENGINE_VERSION` move ⇒ workers refuse to start during the rolling window unless restarts are coordinated or `DFP_SKIP_VERSION_ALIGNMENT` is held on | Correct as far as it goes. **Missing the reassuring corollary:** holding the hatch on for the window reduces OR1 to exactly B20/B22 semantics (brief cache misses, no worse), because the hatch bypasses the gate and the underlying behaviour is identical to a pre-OR1 literal bump. R34 currently reads scarier than the mechanism warrants. | **Change required — state the corollary** |
| **Startup-alignment behaviour** | §2.1, §5.6 SA1-SA9 | Worker gate: fail-closed on the full check (literal in reach), degraded-but-fail-closed otherwise; API: stdlib format check; `run_with_shutdown` runs the gate before the loop. Verified the API path uses `contracts.version` only (`backend/app.py:90-123` region per §2.6). | **Sound** (residual per Defect 1) |

---

## 5. Test Review

### Fail-before coverage — **sufficient, taxonomy handled**

FB1-FB5 cover the three functional knobs (`n_frames`, `min_usable_frames`,
`pool_divisor`) plus the field-completeness guard; FB6 the alignment literal;
FB7-FB8 startup. B23-R2 §5.1's preamble correctly identifies that the *one true
fail-before* is the pre-B23 `test_video_identity_gap.py` version-equality
assertion and labels `VideoSelectionPolicy`-referencing rows "new invariant"
(this answers the prior review's M4). No gap.

### Pass-after coverage — **one missing row**

PA1-PA11 are comprehensive (shipped-bundle pin, both label re-derivations,
`batch_size` neutrality, no-defaults, validation, engine equality,
carries-not-consumes, shim honesty, the two new R32/R33 tripwires).

**Missing:** an explicit **PA12 — `default_face_selection().identity ==
"251d22e9"`** as a standalone "the face policy did not move" assertion. The prior
review's M2 asked for exactly this inside "moved only what it should". B23-R2
§6.1 notes it as an unchanged pin but never lists it as a matrix row. Add it.

### Collateral assertions — **sufficient**

§5.3 lists the `_EXPECTED_EFFECTIVE` / `MODEL_ID` / crop-identity constant
updates and the two semantic rewrites, plus the "verify only"
`test_evaluation_identity.py:75`. Complete in kind.

### Startup-alignment tests — **sufficient, with two small additions**

SA1-SA9 cover accept / reject-stale / before-loop / skip-hatch /
engines-disagree / API-accept / API-reject-non-composed / API-reject-no-decode /
partial-mismatch (SA8, the prior review's M3). **Add:**
- a row that the API format check **rejects a marker-bearing but unparseable**
  string (the prior review's §10.5 — the `parse_model_version` raises case);
- an **API-format-check blast-radius enumeration** parallel to §2.2's call-site
  table: every `create_app(` / `TestClient(` construction with a non-default
  `model_version` now aborts `lifespan` (prior review's M5). This is an
  implementation-safety enumeration, not a test — but it belongs in the design.

### Parser tests — **sufficient**

PR1-PR4 cover totality across all eras (append-only, no branch), the marker-free
guard, and the round-trip. Adequate.

### Missing tests summary

| ID | Test | Origin |
|---|---|---|
| PA12 | `default_face_selection().identity == "251d22e9"` explicit | prior review M2 |
| — | symmetric-defaulting: `ImagePipeline(face_selection=X).model_version == VideoPipeline(face_selection=X).model_version` | prior review §9.4 |
| SA10 | API format check rejects a marker-bearing unparseable `model_version` | prior review §10.5 |
| MG5 | `schema_out/VideoAnalysis.schema.json` default is `"uniform"` (N4 not silently reverted) | this review §2 Defect 6 |
| — (spec, not test) | API-format-check blast radius: enumerate `create_app`/`TestClient` non-default `model_version` sites | prior review M5 |

Also fold in the prior review's §9.3: the `ImagePipeline` block comment pinning
test must assert the carry is **permanent** (the image path will never sample
frames), not "until a future stage".

---

## 6. Reproducibility Review

### New identity blind spots

**One new finding — R35 (Low).** B23-R2 §6.3 states `build_video_analysis`
"kept as defaulted kwargs (not required) so study callers still work." Verified:
`inference/postprocess.py:649-650` — `sampling: str = "uniform"`, `pooling: str
= "topk_mean"`. The wire value is *supposed* to come from
`VideoSelectionPolicy` (R29/N4), but the function retains its own hardcoded
defaults naming the current shipped strings. If `default_video_selection()`'s
`sampling_rule` / `pooling_rule` ever change, this fallback becomes a **stale
second source of truth** for `VideoAnalysis.sampling` / `.pooling`, and no test
holds the two equal. The prior review's required-change #4 anticipated this
("if P16 takes the required-kwarg route"); B23-R2 chose the defaulted route
without a compensating guard.

- **Mechanism:** two independent definitions of the shipped `sampling` / `pooling`
  strings — `default_video_selection()` and `build_video_analysis`'s kwarg
  defaults — with the pipeline always passing the former, so the latter is
  exercised only by study/test callers and can rot undetected.
- **Severity (five-axis):** Correctness 1 · Reproducibility 2 · Migration 1 ·
  Cache-gen 1 · Bug-likelihood 2. **Low.**
- **Disposition:** in scope for B23 as a one-line fix — either make the kwargs
  required (and migrate the ~3 study callers), or add a tripwire
  `build_video_analysis.__defaults__` (or a call with no `sampling`/`pooling`)
  equals `default_video_selection().sampling_rule` / `.pooling_rule`. Recommend
  the tripwire. Enter R35 in `REPRODUCIBILITY_AUDIT.md`.

`batch_size` staying out of the policy is **not** a new blind spot — it was
always out; B23 makes the exclusion explicit and R32 tracks the backend-dependent
precondition. `pool_divisor` moving *into* the identity is a net closure.

### New label closures

`sampling_rule` and `pooling_rule` are new label closures. They sit inside the
established pattern (`ORDERING_RULE`, `interpolation_rule`) and are *stronger* on
one axis (the label is also the published wire value) — B23-R2 §1 and the prior
review §4.2 both note they are *weaker* on specificity (`"uniform"` names a
family; the re-derivation test `PA2` carries the precision). Adequately
dispositioned. No new finding.

### New undocumented assumptions

- The `sample_frame_indices` branch-coverage intent (`fc > n` stride / `fc ≤ n`
  passthrough / empty) is asserted by PA2's pair list *by construction* — B23-R2
  PA2 now states the intent explicitly (prior review M6). ✓
- The `selection_identity` **non-injectivity** over `(face, video)`: prior review
  §9.1 asked for "one sentence." B23-R2 §4.1 mentions it but points to "§7" for
  the argument, and **§7 does not contain it** — a dangling reference and an
  unstated assumption. **Change required:** state the sentence (a hash of two
  hashes is not injective over the pair; acceptable for the same reason every
  `fingerprint()` is a 32-bit change-detector, not a security boundary —
  `contracts/version.py:107-115`).

### New cache-key risks

Idempotency keys (`{sha256}:{model_version}`) regenerate in lockstep with the
content-cache key — B23-R2 §5.4 MG4 folds in the prior review's §9.2 point
(in-flight job re-leased after deploy re-runs once). No new risk.

### New deployment risks

R34 (fail-closed gate makes `STUB_ENGINE_VERSION` moves non-rolling-deployable
without coordination) is the one genuinely new operational constraint relative
to B20/B22. B23-R2 §3 dispositions it as a rollout-guidance + CHANGELOG item, not
a code change — correct (it is OR1 working as intended). Needs the "hatch reduces
to B20/B22 semantics" corollary (§4 above).

### New Rxx findings

| Finding | Severity | Status | In B23? |
|---|---|---|---|
| **R35** — `build_video_analysis` retains hardcoded `sampling`/`pooling` kwarg defaults duplicating `default_video_selection()`; no tripwire holds them equal | Low | New (this review) | Yes — one-line tripwire |

No other new Rxx is warranted. `Settings.skip_version_alignment` dead config
(§2 Defect 1) is a code-hygiene change, not a reproducibility finding — track it
as a note in the B23 section, not an Rxx.

---

## 7. Implementation Readiness

### Can implementation begin immediately?

**No — but the blockers are a short, mechanical specification pass, not design
work.** The architecture, the identity movement, the cache-generation count, the
parser strategy, the Route-A decision, and the OR1 shape are all **locked and
verified**. What is missing is specification completeness that an implementer
would otherwise have to invent.

### Exact unresolved items

| # | Item | Section | Kind |
|---|---|---|---|
| 1 | Replace Defect-2's "confirm at implementation" with the verified fact (`test_frame_count_reproducibility.py` — all callers use `n_frames=16`) | §2 Defect 2 | doc fix |
| 2 | Commit to `selection_identity_of` / `ModelSpec.with_selection` aliases **in B23**; strengthen PA10 to assert the docstring banner; soften the `contracts/version.py:47-53` legibility promise | §2 Defect 3, §3 | spec decision |
| 3 | Fold in the version-constant sweep + result (`effective_model_version: str` unconstrained ⇒ no bump) | §2 Defect 4 gap 1 | doc fix |
| 4 | State the fixture `model.version == "1.0.0-dev"` fact; confirm N4 regen touches only `sampling` | §2 Defect 4 gap 2 | doc fix |
| 5 | Enumerate and classify every `251d22e9` doc occurrence | §2 Defect 4 gap 3 | doc fix |
| 6 | Restore a consolidated **§6.4 Rollback** subsection | §4 | spec restore |
| 7 | Add the R34 corollary (hatch ⇒ B20/B22 semantics) | §4 | doc fix |
| 8 | Resolve `Settings.skip_version_alignment` — delete it or bind it to the future composition root | §2 Defect 1 | spec decision |
| 9 | State the `selection_identity` non-injectivity sentence; fix the §4.1→§7 dangling reference | §6 | doc fix |
| 10 | Add missing tests: PA12, symmetric-defaulting, SA10, MG5, the "permanent" carry-comment assertion; add the API-format-check blast-radius enumeration | §5 | spec addition |
| 11 | Add R35 (tripwire on `build_video_analysis` defaults) to the plan and the risk register | §6 | spec addition |
| 12 | Name option 2 (backend composition root) as a scheduled follow-up, not "a future … may" | §2 Defect 1 | spec decision |

### Are any of them architecture-affecting?

**No.** Items 2, 8 and 12 are *decisions* the design must make, but every
available answer keeps the architecture identical:
- the alias (item 2) is additive to `contracts.version` and does not touch the
  parser;
- `Settings.skip_version_alignment` (item 8) is a config field with no consumer —
  removing or binding it changes nothing structural;
- option 2 (item 12) is a new `backend`-side module that imports an already-legal
  pair of packages.

The other nine items are documentation, enumeration, or one-line tripwires.

---

## 8. Final Verdict

### **APPROVE WITH CHANGES**

B23-R2 successfully resolves every blocking finding from `B23_ADVERSARIAL_REVIEW.md`
at the design level:

- **Defect 1** — resolved; the `workers → backend` cycle is genuinely avoided
  (env-only gate, verified against the DAG test), with a documented and
  CI-guarded residual on the config-default path.
- **Defect 2** — resolved in principle; one self-verification is deferred that
  is knowable now and is closed in this review (all callers use `n_frames=16`).
- **Defect 3** — partially resolved; the specification is complete but the chosen
  "names stay, docs change" leaves a permanently misleading public name whose
  tripwire guards only doc-drift. Needs the `selection_identity_of` alias
  committed in B23.
- **Defect 4** — substantially resolved; three enumeration gaps remain
  (version-constant sweep, fixture `model.version`, doc classification).
- **Defect 5** — resolved; the four-part decomposition is correct and the
  "server already emits `uniform`" claim is verified.
- **Defect 6** — resolved; the "`default` is not structural" claim is proven from
  `backend/openapi.py:209-251` and independently confirmed.
- **Defect 7** — resolved; P19 dropped, `inference/__init__.py` confirmed not to
  export the face symbols.

The revisions introduce **no architecture-affecting defect** and **one new
low-severity reproducibility finding** (R35 — `build_video_analysis` default
duplication). The Route-A recommendation is re-verified and stands, with a more
honest framing of its legibility cost required.

B23-R2 is **not yet the final implementation-ready design** — it defers one
safety proof, ships one knowingly-misleading public name, drops the rollback
consolidation, carries a dangling reference and dead config, and is missing five
test rows and three enumerations. All twelve items in §7 are documentation,
enumeration, one-line tripwires, or decisions with no structural consequence.

**Fold the §7 items into a BATCH_23_R3 (or an R2 revision) and B23 is cleared
for implementation.** No further architecture review is required.
