# B23 Adversarial Review — `BATCH_23_DESIGN.md`

**Reviewer role:** independent principal engineer / adversarial reviewer.
**Object of review:** the design in `BATCH_23_DESIGN.md` (R29 + N4 + OR1). Design
only — no code was written for this review.
**Method:** re-derive from first principles, then attempt to falsify every claim
against the shipped B22 tree (`contracts/version.py`, `inference/`, `workers/`,
`backend/`, `tests/`).

> **Disclosure.** B23 has since been implemented in this repository. This review
> is of the **design as written**; where implementation corroborated a design
> gap I say so explicitly. The implementation's passing test run does **not**
> retroactively make an under-specified design item correct — several items below
> were *worked around* at implementation time precisely because the design did
> not specify them.

---

## 1. Executive Verdict

**APPROVE WITH CHANGES.**

The core architecture is sound. R29 is a real gap (verified: two `VideoPipeline`
instances differing on `n_frames` / `min_usable_frames` / pooling depth produce a
byte-identical `model_version` — the whole result-cache key). The B12
`FaceSelectionPolicy` template is the right instrument, applied a fourth time.
Folding the video-selection fingerprint into `.f.` via
`selection_identity(face, video)` genuinely keeps `contracts/version.py`'s parser
untouched and the single `STUB_ENGINE_VERSION` literal intact, and it produces
exactly one cache generation. Output is byte-identical for the shipped policy's
*scored* fields. `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` /
`CALIBRATION_WITNESS_VERSION` genuinely do not need to move.

But the design has **seven blocking defects** (§13), the most serious of which is
that **OR1's worker-side gate (P20) is not implementable as written** — obtaining
`settings.model_version` inside `workers/` closes a `workers → backend`
dependency cycle that `test_dependency_graph_stays_acyclic` forbids, and the
design neither notices this nor specifies an alternative. The remaining six are
an unspecified interaction between the new `min_usable_frames <= n_frames`
validation and the legacy-scalar shim, an incomplete pin inventory, an
overstated "byte-identical bundle" claim, an unverified load-bearing assumption
about the OpenAPI signature-stripper, a factually wrong statement about the
current export surface (P19), and a scheduled-but-omitted set of
`contracts/version.py` docstring updates that leave that module self-contradictory.

None of the seven changes the architecture. With them folded in, APPROVE.

---

## 2. Independent Re-Derivation

**The defect.** The result cache is keyed on `(sha256, model_version)`
(`workers/io.py` `content_key_for`). `model_version` =
`compose_runtime_version(model=effective_version, decode=decode_identity())`.
`effective_version` = weights `+c.` calibration `.g.` gate `.f.` face-selection.
None of those five segments move when the **video** path changes how many frames
it samples (`VideoPipeline.n_frames`, default 48), what its abstention floor is
(`min_usable_frames`, default 8 — a verdict→abstention gate, the `min_face_px` /
R19 shape, and *not* a `MediaCaps` field so it cannot reach `gate_fingerprint`),
or how deep the top-k pool runs (`topk_indices`'s `k`, the literal `arr.size // 8`
in `inference/postprocess.py:548`). The video path has no face stage (R18), so
the *face* selection policy that `.f.` already names does not touch it. Result:
two deployments (or one deployment across a config edit) share a cache key while
sampling a different frame set and able to publish a different `p_fake` — or, via
`min_usable_frames`, flip `manipulated` → `insufficient_evidence`. This is the
video analogue of R17 (face selection) and R25 (crop geometry).

**Why a policy object, not loose scalars.** `VideoPipeline.__init__` takes
`n_frames`, `batch_size`, `min_usable_frames` as bare scalar defaults — the exact
pre-B12 `FaceSelectionPolicy` situation. B12's fix — one frozen dataclass, every
field required, one factory — has been validated by R17/R19/R25. Correct to reuse.

**Why fold into `.f.` and not add `.v.`.** A `.v.` marker means a new entry in
`_MODEL_MARKERS`, a fifth `ModelVersionParts` field, a new peel-from-right step
in `parse_model_version`, and a permanent "pre-B23 era" every parser test must
carry. Folding `selection_identity(face, video) = fingerprint({"face": …,
"video": …})` into `.f.` keeps `.f.` as one opaque 8-hex fingerprint — parser,
markers, eras all unchanged — and keeps one `STUB_ENGINE_VERSION` literal because
both pipelines report the same string. The cost is that `.f.` no longer names
just the face policy (§8, §11 below). This mirrors B22 folding `input_size` into
`-w` rather than adding a `-i` token.

**Functional vs label closures.** `n_frames`, `min_usable_frames`, `pool_divisor`
are *functional* — the pipeline reads them from the policy and passes them to
`sample_frame_indices` / the gate / `topk_indices`, so the value driving the
behaviour and the value in the fingerprint are one field. `sampling_rule`
(`"uniform"`) and `pooling_rule` (`"topk_mean"`) are *label* closures — bare
module functions do not receive the policy — held to the code by re-derivation
tests, and additionally published as `VideoAnalysis.sampling` / `.pooling`.

**`batch_size` excluded.** `batch_clips` (`preprocess.py:1184`) yields contiguous
non-overlapping slices with no padding; `_fuse_batch` is per-frame;
`StubBackend.score` (`models.py:368`) iterates `for i in range(N)` — row
independent; `topk_indices` / `pool_at` run on the full `per_frame` array after
every batch. So the reconstructed `scores` array is byte-identical across batch
sizes and only progress-event fractions (and phase timings) depend on it.

**N4.** `contracts/video.py:92` declares `sampling: str = Field(
default="uniform+adaptive")` and the fixtures emit it, while
`inference/postprocess.py:649` hard-codes `sampling="uniform"` in the real
bundle. `sample_frame_indices` is pure uniform stride — nothing adaptive. Make
the live value authoritative; source it from `VideoSelectionPolicy.sampling_rule`;
correct the stale default and the fixtures. A `Field(default=)` value change.

**OR1.** `workers.engine_registry.verify_version_alignment` has existed since B9
and is called only from tests. A worker whose engines report a `model_version`
different from the API's configured literal has a *silently* disabled result
cache. Worker boot should run the real check fail-closed; the API process cannot
(it must stay free of cv2/numpy — `contracts/version.py` module docstring "WHY
THIS LIVES IN contracts") so it runs a stdlib-only format check.

This re-derivation reproduces the design's conclusions. The disagreements are in
the details below.

---

## 3. Verified Correct

| Design claim | Verification |
|---|---|
| R29 is real and measurable | `topk_indices(scores)` (`postprocess.py:548`) has no policy input; `n_frames`/`min_usable_frames` are `VideoPipeline.__init__` defaults reaching no identity. Confirmed. |
| `.f.` stays one opaque fingerprint; `parse_model_version` unchanged, no new era (§3.2, PA4) | `parse_model_version` (`version.py:402`) strips `.d.`, partitions `+c.`, then peels `.f.` as a single token. `e629642c` (or any `fingerprint()` output) is 8 hex chars, marker-free — `compose_model_version`'s `any(m in face_selection for m in _MODEL_MARKERS)` guard passes. Round-trips. |
| Single `STUB_ENGINE_VERSION` literal preserved (§3.2 route A) | `ImagePipeline` and `VideoPipeline` both compute `selection_identity(default_face_selection(), default_video_selection())` → identical `.f.` → identical `effective_version` → identical `model_version`. `verify_version_alignment` is satisfiable by one string. |
| Exactly one cache generation (§6.3) | Only the `.f.` segment moves; `-w`/`+c.`/`.g.`/`.d.` unchanged. One `model_version` change ⇒ one full miss per file per namespace. |
| Both content-cache namespaces regenerate; that is a false miss, safe direction (§3.2) | The image namespace keys on `model_version` which now carries a video-only segment. B20 `.d.` precedent (video-only demuxer change regenerated the image namespace) is exact and correctly cited. |
| No `CALIBRATION_WITNESS_VERSION` bump (§6.2) | `CalibrationWitness.compute_witness` takes `schema_version` as a parameter and its digest membership excludes `effective_model_version` / `.f.` (this is R31, memory-recorded). A `.f.` value change moves no stored witness digest; old witnesses still verify. Same as B20/B22. |
| No `CONTRACT_VERSION` bump (§6.2) | `VideoAnalysis` keeps every field, type and `required` entry. `sampling` was already optional (`default=` present since inception). The server *always* populates `sampling`, so the default is never serialized on a real response — the change is invisible on the wire. |
| `batch_size` is output-neutral **for the stub path** (§3.4) | `batch_clips` does not pad; `StubBackend.score` is row-independent; `scores` reconstruction is order- and value-preserving. Verified against `models.py:368` and `preprocess.py:1184`. (Caveat: see R32, §12.) |
| `pool_divisor` produces byte-identical output for the shipped config (§8.2) | Old: `topk_indices(scores)` → `k = min(max(1, None or arr.size // 8), arr.size)` = `arr.size // 8` (with `arr.size >= min_usable_frames = 8` ⇒ `k >= 1`). New: pipeline passes `k = max(1, len(scores) // 8)`; `topk_indices` then does `min(max(1, k or …), arr.size)` = `k`. Identical, including the `arr.size < 8` edge (both floor to 1). |
| API format check catches the historically-real failure (§3.6) | `is_composed("1.0.0-dev")` → `"+c." in "1.0.0-dev"` → False → rejected. `is_environment_pinned` similarly rejects a model-only string. |
| API format check respects the Batch-0 boundary (§4.9) | Uses only `contracts.version` (stdlib). No `inference`, no cv2/numpy in the API process. |
| `default_face_selection().identity` unchanged (§6.1) | B23 adds no field to `FaceSelectionPolicy`; its `.identity` dict is untouched. |

---

## 4. Partially Correct

### 4.1 "Output byte-identical" / "byte-identical bundle" (§3.4, §5.2 PA1, §8.2)

**Overstated.** The video bundle carries `timing` (`EvidenceBundle` … via
`timer.to_timing()`), and phase timings depend on how the inference work is
chunked — i.e. on `batch_size`, and on wall-clock noise between any two runs. Two
`batch_size` values do **not** produce a byte-identical bundle; they produce a
bundle identical in every *scored* field (`p_fake`, `verdict`, `frame_scores`,
`segments`, `top_frames`, `video.sampling`, `video.pooling`, `model.version`) and
different in `timing` and job metadata (`job_id`, `submitted_at`). PA1 as written
("before/after … produces an identical bundle **except** `model.version` and
`video.sampling`") will fail on `timing` unless timing is masked. The correct
invariant is: *identical scored output across batch sizes; identical full bundle
for identical inputs, policy and batch size.* Implementation had to compare
`a.video.model_dump() == b.video.model_dump()` and `a.p_fake == b.p_fake`
field-wise rather than whole-bundle equality.

### 4.2 "The video labels are *strictly stronger* closures than B22's `interpolation_rule`" (§3.4)

**Half true.** They are stronger on one axis (the label value *is* the published
wire value, so "wire = identity input" is true by construction) and **weaker** on
another (specificity). `interpolation_rule =
"area-if-either-axis-exceeds-target-else-linear"` names the exact predicate;
`sampling_rule = "uniform"` names a *family* — the design itself chooses the
imprecise name deliberately (§3.5) and relies on the re-derivation test for
precision. B22's `interpolation_rule` *also* has a re-derivation test. Net: the
video labels are stronger by "wire=input", equal by "held by re-derivation test",
weaker by "the name pins the behaviour". "Strictly stronger" is not defensible;
say "stronger on the wire-binding axis".

### 4.3 `contracts/version.py` "untouched" (§1, §3.2, §8.5)

**True for parsing logic, false for documentation.** The design forbids modifying
the parser (correct) but does not schedule updates to the module's own prose,
which after B23 is wrong:

- `version.py:33-40` — the `face` segment description: *"changes when the
  face-selection policy changes: the detector and its asset, its detection
  parameters, the ordering rule … Added in Batch 12 (R17)."* — now incomplete
  (also frame selection + pooling, for video).
- `version.py:119-130` — `FACE_SELECTION_MARKER` docstring: *"Separator for the
  face-selection segment."* — now the *selection* segment.
- `version.py:446-452` — `face_selection_identity_of` docstring: *"The
  face-selection fingerprint"* — now the composite.
- `version.py:455-463` — `is_face_selection_pinned` docstring.

§5.7's doc-update table lists `REPRODUCIBILITY_AUDIT.md`,
`docs/VIDEO_FACE_SEMANTICS.md`, `docs/CROP_IDENTITY.md`,
`docs/FACE_SELECTION_IDENTITY.md`, `docs/RUN_IDENTITY.md`, `backend/config.py`
docstring — **not** `contracts/version.py`. Updating a docstring is not "parsing
logic". This is a scheduled-doc omission, not a hard prohibition. (See also
§5.2 / R33.)

### 4.4 "`test_batch_size_is_output_neutral` … extends the existing check at `test_inference.py:255-268`" (§3.4)

**Miscitation.** `test_inference.py:255-268` is
`test_progress_fires_once_per_batch_never_per_frame` /
`test_progress_scales_with_the_batch_size` — those assert *progress-event
counts*, not bundle equality. There is no existing output-neutrality test at that
location to extend. `test_inference.py:553` constructs
`VideoPipeline(n_frames=48, batch_size=1)` but for a different purpose. The new
test is genuinely new, not an extension.

### 4.5 "Reachable via a code fork today … same reachability class as R25's label closures and R28" (§2, R29 row; §7.1)

**True but incomplete.** `n_frames` / `min_usable_frames` are also reachable from
**test and study call sites** that pass them explicitly (`test_inference.py`,
`test_timing_contract.py`, `test_frame_extraction.py`,
`test_branch_b_semantics.py`, `test_headline_and_face_semantics.py`,
`test_frame_count_reproducibility.py`, and 3 studies). The design's §5.4 shim
discussion knows this; the R29 characterisation ("`__init__` defaults with a
single shipped value") undersells that these knobs are *already varied in the
suite*, which is why the shim (not strict mode) is the right call — and why the
new `__post_init__` validation is a live hazard (§5.1).

---

## 5. Incorrect Claims

### 5.1 P2 validation `1 <= min_usable_frames <= n_frames` contradicts the design's own FB2 scenario and the shim's safety claim

`§4.1 P2` adds `__post_init__` validation `1 <= min_usable_frames <= n_frames`.
`§5.1 FB2` references the current test
`test_r29_min_usable_frames_flips_the_verdict_under_one_model_version`, whose
mechanism (in `tests/test_video_identity_gap.py`) is
`_analyse(n_frames=9, min_usable_frames=16)` — i.e. **`min_usable_frames >
n_frames`**, a combination P2 makes *unconstructible*. The design says "rewrite to
`..._moves_the_identity`" without noticing the rewritten test cannot use the same
mechanism (a 9-frame sample can never fail a 16-frame floor once the policy
refuses to be built). Implementation had to switch to a 12-frame clip with
`n_frames=16, min_usable_frames=13`.

Worse, `§5.4` claims the shim "keeps ~10 study/test call sites working with **no
edit**" and reconstructs the policy "via `replace()`". But
`replace(default_video_selection(), n_frames=4)` (a caller passing only a small
`n_frames`) yields `VideoSelectionPolicy(n_frames=4, min_usable_frames=8, …)`
→ `__post_init__` raises `ValueError` at **pipeline construction time**.
Previously `VideoPipeline(n_frames=4)` was legal (it would always abstain). The
design does not:
- enumerate the call sites to prove every `n_frames` passed is `>= 8`, or
- specify that the shim must lower `min_usable_frames` to `min(min_usable_frames,
  n_frames)` when only `n_frames` is passed, or
- scope "output-identical" to the shipped policy (a previously-legal policy
  `min_usable > n_frames` is now a `ValueError` — a behaviour change, however
  benign).

**This is a correctness gap in the design.** (Implementation happened to be safe
because every call site passes `n_frames ∈ {16, 48}`, but the design never
established that.)

### 5.2 `.f.` composite makes `contracts.version.face_selection_identity_of()` return a value that is not a `FaceSelectionPolicy.identity` — a permanent semantic defect in a public function, unaddressed

After B23, for any B23+ version string,
`face_selection_identity_of(v) == fingerprint({"face": <face>, "video":
<video>})` — **not** `<face>`. Any external tool, log parser, benchmark
comparator, or future developer that calls `face_selection_identity_of(v)` and
compares it to `default_face_selection().identity` gets a false negative
*forever*. `is_face_selection_pinned(v)` returns `True` but now means
"selection-pinned". The design acknowledges this only as an *implementation-time*
test failure ("`FB8` … enumerated in the pin inventory; semantic rewrite"). It
never proposes:
- renaming `face_selection_identity_of` → `selection_identity_of` (with a
  deprecation alias), or
- at minimum, redocumenting both functions.

The design's §9 required-change #2 is specifically about the *test* assertions,
not the *production* function's contract. A `contracts/` public function whose
name and docstring lie about its return value is a defect the design ships
knowingly. (Filed as R33, §12.)

### 5.3 P19 states a false fact about the current codebase

`§4.7 P19`: *"Export `VideoSelectionPolicy`, `default_video_selection`,
`video_selection_identity`, `selection_identity` **if the package re-exports the
face equivalents (it exports `FaceSelectionPolicy` — match it)**."*

`inference/__init__.py` **does not export `FaceSelectionPolicy`** — its `__all__`
lists protocols, pipelines, registry helpers and calibration only.
`FaceSelectionPolicy`, `default_face_selection`, `face_selection_identity` are all
reached via `inference.preprocess`. P19's premise is factually wrong. A reviewer
following it literally either adds exports with no precedent or has to detect the
error. The design must not contain unverified claims about the current export
surface. (Correct action: drop P19, or restate it as "do **not** add exports —
match `FaceSelectionPolicy`, which is preprocess-only.")

### 5.4 P20's `verify_version_alignment(settings.model_version)` is not implementable without a dependency cycle

`§4.8 P20`: *"In the worker bootstrap (… `workers/runner.py` `run_with_shutdown`
entry, or the process `main` …): before the first `runner.run(...)`, call
`workers.engine_registry.verify_version_alignment(settings.model_version)`."*

There is no `settings` in scope in `workers/runner.py`, and there is **no worker
process `main`** in the repo (no `__main__` under `workers/`; `build_all_workers`
in `workers/registry.py` is the closest thing). Obtaining `settings.model_version`
requires `from backend.config import get_settings` (or `Settings`) inside
`workers/`. That creates edge `workers → backend`. `backend/app.py` already has
`from workers.io import …` and `from workers.protocol import …` ⇒ edge `backend →
workers`. **Cycle.** `tests/test_worker_inference_integration.py::
test_dependency_graph_stays_acyclic` (AST-walks *all* imports, including
function-local) fails with `cycle: workers -> backend -> workers`.

The same problem hits P21's `settings.skip_version_alignment` if the worker reads
it via `Settings`.

The design neither notices this nor picks a mechanism. Viable options the design
must choose among and specify:
1. Worker reads `os.environ.get("DFP_MODEL_VERSION")` directly (stdlib) and the
   escape hatch via `os.environ.get("DFP_SKIP_VERSION_ALIGNMENT")`; when
   `DFP_MODEL_VERSION` is unset it cannot see the `backend.config` default and
   must fall back to a weaker check (engines agree with each other + are
   well-formed), relying on the existing build-time
   `test_configured_model_version_matches_the_engines` for the default.
2. A new composition-root module *in `backend`* (which may import both) that
   wires `build_all_workers` + the gate + `run_with_shutdown`. This is the "the
   process `main`" the design gestures at but does not create.
3. `run_with_shutdown(runner, *, configured_model_version, …)` — the caller
   supplies it.

Each has a different residual-gap profile. The design must pick one and state the
residual. **As written, OR1's worker half is under-specified to the point of
being un-implementable.** (Implementation took option 1 + a fallback; this is a
deviation from the design's literal text and the design should own it.)

### 5.5 §3.4's stated reason for `batch_size` neutrality is incomplete

*"`_score_batch` widens per batch and `astype` is elementwise"* is necessary but
not sufficient. Neutrality *also* requires: `batch_clips` never pads (verified:
`preprocess.py:1184`), the backend `score()` is row-independent (verified for the
stub: `models.py:368`), and `per_frame` reconstruction preserves order and values
(it does). The design should state the full precondition, because two of the
three parts are properties of code B23 does not own and a future backend can
violate (R32).

---

## 6. Missing Tests

| # | Gap | Why it matters |
|---|---|---|
| M1 | No pass-after that a **stored video bundle with `sampling: "uniform+adaptive"` still validates**. PA10 (§5.2) extends `test_old_artifacts_still_load` only with a pre-B23 **`model.version`** string. | N4 changes a value that is already sitting in stored bundles and fixtures. The migration surface is `sampling`, not `model.version`, and it is untested. |
| M2 | No assertion in PA7 (`test_batch_23_moved_only_the_selection_identity`) that `default_face_selection().identity` **did not move**. §5.2 lists the three version constants, the snapshot/`schema_out` greps, and `parse_model_version(...)` `.calibration`/`.gate`/`.weights` — not the face policy identity. | "Moved only what it should" should pin what *didn't* move. (Covered incidentally by FB7's retained `POLICY.identity == "251d22e9"` in `test_crop_geometry.py`, but PA7 is the designated guard.) |
| M3 | No OR1 test for the **partial engine mismatch** (image aligned, video stale, or vice-versa). FB10/FB11 are whole-process. | `verify_version_alignment` builds a per-engine `mismatched` dict; the video-selection-specific failure mode (video engine reports a different `.f.` because it consumed a different `VideoSelectionPolicy`) is exactly what OR1 exists to catch and is unexercised. |
| M4 | No fail-before that is expressible **against the current tree**. §5.1 FB3/FB5/FB6 cannot "fail" today — `VideoSelectionPolicy` does not exist, so they are compile errors, not wrong assertions. | The genuine fail-before is the *existing* `test_video_identity_gap.py` asserting `dense.model_version == sparse.model_version`; the design should say so explicitly and label FB3/FB5/FB6 as "new invariant" not "fail-before". Taxonomy, but it weakens the "prove the batch changed what it claims" argument (§5.1 preamble). |
| M5 | No test that the **API format check does not regress an existing app-construction test**. §5 assumes it only rejects `"1.0.0-dev"`; it does not enumerate which tests enter `lifespan` with a non-default `model_version`. | A single test that builds `create_app` with a bad `model_version` inside a `TestClient` context and *expects* success would now break silently at review time. (Implementation verified only `test_backend.py` enters `lifespan`, and it mutates `model_version` post-lifespan — but the design did not.) |
| M6 | No re-derivation coverage statement for `sample_frame_indices`' **`frame_count <= n` branch** and its **empty branch**. §3.4 prose mentions only the stride formula. | PA2's pair list `{(1000,48),(1000,9),(50,48),(48,48),(7,8)}` *does* hit all three branches — but by accident of pair choice, not by design intent. State the intent so a future edit to the pair list does not silently drop branch coverage. |
| M7 | No tripwire that `.f.`'s **documentation** matches its **meaning** (the R33 surface). B22 had the same gap for `interpolation_rule`. | Nothing fails if `contracts/version.py`'s `.f.` docstrings drift further from reality. |

---

## 7. Migration Risks

### 7.1 OR1 fail-closed gate + `STUB_ENGINE_VERSION` move ⇒ B23 is not rolling-deployable; §6.5's narrative is internally inconsistent

`§6.5` says both:
- *"Rolling deploy: mixed old/new workers each validate against their own
  configured literal."*
- (implicitly, throughout) there is **one** `Settings.model_version` /
  `STUB_ENGINE_VERSION`.

These cannot both be true. At any instant during a rolling deploy the configured
literal (from env or `backend.config`) is either the old value or the new value:
- Old literal + a **new** worker (new engines report the new `.f.`) ⇒ the new
  worker's OR1 gate **fails closed** ⇒ it cannot start.
- New literal + an **old** worker ⇒ the old worker's gate fails closed.

So OR1 converts a `STUB_ENGINE_VERSION` bump (which B20 and B22 also had, but
without a fail-closed gate) from "brief window of cache misses" into "brief window
of workers refusing to start". The design's "order of operations" (§6.5: land
code → workers boot → API boots) assumes an atomic deploy. In a
pod-by-pod rolling update of a real cluster, B23 needs a **coordinated restart**
or the `DFP_SKIP_VERSION_ALIGNMENT` hatch held on for the duration of the window.
The design must state this plainly in §6.5 and the CHANGELOG note it promises.

### 7.2 `min_usable_frames > n_frames` policies become un-constructible (see §5.1)

A behaviour change for any (hypothetical) caller. Low blast radius, but "output-
identical" must be scoped to the shipped policy.

### 7.3 `stub_handlers.py` is a stale `"uniform+adaptive"` site not in the pin inventory

`tests/stub_handlers.py:433` constructs `VideoAnalysis(sampling="uniform+adaptive")`.
§5.3's pin table lists `contracts/fixtures.py:515` and `fixtures_out/video_*.json`
but not this. N4's stated goal is "make the live value authoritative"
*everywhere*; a stale test stub contradicts that and nothing in the design catches
it. (Implementation found and corrected it.)

### 7.4 Benchmark cross-boundary behaviour for a pre/post-B23 pair is unspecified

`BenchmarkRun.effective_model_version` = model identity, whose `.f.` moves. A
benchmark comparing a pre-B23 run and a post-B23 run of an *unchanged* video
pipeline will read as "different models" (`regression_gate` /
`BenchmarkReport.validate`). This is the same situation B22 handled
(`test_crop_identity.py::test_the_benchmark_gate_still_compares_across_the_boundary`)
but §6.1's migration table and §5's test plan do not mention extending that test
for B23. Add the row.

---

## 8. Versioning Risks

### 8.1 The design does not exhaustively enumerate version constants

`§6.2` has four rows (`CONTRACT_VERSION`, `RUN_SCHEMA_VERSION`,
`CALIBRATION_WITNESS_VERSION`, OpenAPI signature). The repo also has (at least) a
split-lock format version, an evaluation-protocol version, and possibly others.
The design should include a "grep `_VERSION\s*=` across the tree and confirm none
governs a structure that carries `model_version` / `effective_model_version` /
`.f.`" step. (Belief: none do — `effective_model_version` is an unconstrained
`str` everywhere — but the design asserts rather than checks.)

### 8.2 `RUN_SCHEMA_VERSION` no-bump rests on `effective_model_version` being unconstrained `str`

`§6.2`: *"`effective_model_version` already carries `.f.`"* — true, but the
no-bump argument needs "and there is no length / format / regex constraint on it
in any `RUN_SCHEMA_VERSION`-governed model". State it.

### 8.3 The OpenAPI "zero structural diff" hard gate rests on an unverified assumption about the signature-stripper

`§5.5` / `§8.5` / `§9` #4 make "`--update-snapshot` produces a zero structural
diff" a hard gate, but the design *assumes* the prose-stripped structural
signature drops `default`. If the stripper treats `default` as structural, the
`"uniform+adaptive"` → `"uniform"` change moves the signature and the gate trips.
The design should *state* (from `backend/openapi.py`'s signature logic) that
`default` is stripped, not hope it. (Implementation confirmed: the snapshot's
`signature` block carries no `default` key, so the diff is genuinely zero — but
that is a fact to cite, not assume.)

### 8.4 `CONTRACT_VERSION` no-bump: cite the policy

`§6.2` reasons "the `sampling` default value is not structural". The stronger,
citable argument: the server *always* sets `sampling` in `build_video_analysis`,
so `Field(default=)` is dead code on every real response — the wire is
unaffected. Only the OpenAPI doc and fixtures show the old string. If the
project's contract-version policy is "wire-observable behaviour", no bump; if it
is "any change a client could notice, including the published schema's `default`",
that is a (weak) argument *for* a patch bump the design should rebut explicitly.

---

## 9. Cache / Identity Risks

### 9.1 `selection_identity` collision domain is not discussed

`.f.` was `sha256(canonical(face_policy_dict))[:8]`. It is now
`sha256(canonical({"face": <8hex>, "video": <8hex>}))[:8]` — a hash of a hash.
The 32-bit change-detector semantics (`version.py:107-115`) still apply and the
practical collision risk is unchanged (a fleet holds a handful of policies), but
the design should note that `selection_identity` is **not injective** over the
`(face, video)` pair and that this is acceptable for the same reason every other
`fingerprint()` is. One sentence.

### 9.2 Idempotency-key regeneration is correctly identified but its blast radius is understated

`§6.3`: *"a job submitted before and after B23 is not deduplicated against its
pre-B23 self."* Correct. Worth adding: an **in-flight** job leased under the old
key and re-enqueued after deploy will re-run once. Same one-time cost as B20/B22;
just say so.

### 9.3 The image pipeline now carries *two* "carried, not consumed" policies

Post-B23: `VideoPipeline` carries `FaceSelectionPolicy` (not consumed until
SCRFD) *and* consumes `VideoSelectionPolicy`. `ImagePipeline` consumes
`FaceSelectionPolicy` *and* carries `VideoSelectionPolicy` (never consumed — the
image path will never sample frames). `§3.3` mentions the symmetry but not that
the "carried, not consumed" surface doubles and that the image side's
`VideoSelectionPolicy` carry is *permanent*, not "until a future stage". The
block comment (§4.4 P15) and its pinning test should say "permanent".

### 9.4 No identity-consistency check for a caller that passes only one of the two policies

`§4.3 P8` / `§4.4 P13` both default the *other* policy to its factory. So
`ImagePipeline(face_selection=X)` and `VideoPipeline(face_selection=X)` both
compute `selection_identity(X, default_video_selection())` and agree. Good — but
no design test asserts this symmetric-defaulting property. Add:
`ImagePipeline(face_selection=X).model_version ==
VideoPipeline(face_selection=X).model_version`.

---

## 10. Startup Alignment Review

### 10.1 Batch-0 boundary — API side: **correct**

`§4.9 P22` uses `contracts.version` only. `verify_version_alignment` itself is
unusable from `backend` (it does `from inference import ImagePipeline,
VideoPipeline` — cv2/numpy). The design correctly identifies this (§3.6) and
correctly restricts the API to a format check. No boundary violation.

### 10.2 Batch-0 boundary — worker side: **the design breaks the layering it depends on** (see §5.4 / R-cycle)

`workers → backend` is a cycle. The design's own §3.6 argument ("`verify_version_
alignment` imports `from inference import …` which pulls cv2 and numpy into the
calling process") is about the *API*; it does not notice that the *worker* also
needs `backend.config` for `settings`. This is the single blocking architectural
defect.

### 10.3 Fail-closed vs warn-and-continue — **correct call, but §6.5's rollout guidance is wrong** (see §7.1)

Fail-closed is the right default (a warn-and-continue gate preserves the exact
silent-broken-cache state the gate exists to end). But the design must reconcile
fail-closed with rolling deploys.

### 10.4 The escape hatch is under-specified

`§4.8 P20`: `if settings.skip_version_alignment: log.warning(...); return`. And
`§4.8`: `DFP_SKIP_VERSION_ALIGNMENT`. Given §5.4, the worker cannot read
`settings.skip_version_alignment` without the cycle — so the hatch must be an
`os.environ` read. The design should also specify the truthy-value parsing
(`"1"/"true"/"yes"`), and that the warning fires **every** boot (not once), which
§6.5 asserts but P20 does not implement.

### 10.5 API check completeness

`§4.9 P22` checks `is_composed`, `is_environment_pinned`, and
`parse_model_version(mv).compose() == model_identity_of(mv)`. The round-trip is a
good addition. One more cheap check worth adding: reject if `parse_model_version`
raises (a malformed-but-marker-bearing string). The design's phrasing ("survive
`parse_model_version(...)`") implies it; make it explicit.

---

## 11. Alternative Architectures Considered

### 11.1 Route A (`.f.` composite) vs Route B (`.v.` segment) — the design's verdict is defensible but oversold

**Design's case for A (§3.2):** `contracts/version.py` untouched; single literal;
no new parser era; B22 precedent (`input_size` → `-w`).

**The case for B the design under-weights:**
- `contracts/version.py`'s **module docstring** (`version.py:47-52`) states the
  format is *"readable on purpose. An operator reading a log line should be able
  to see which component moved without decoding anything."* Route A **directly
  degrades this stated design goal**: an operator seeing `.f.` move in a log line
  cannot tell whether the face policy or the video policy changed. `differences()`
  requires holding both policy objects — not available from a log line. Route B
  preserves the goal. The design dismisses this as "legibility cost, same as
  B22" — but B22's `input_size`-in-`-w` is a *sub-fingerprint an operator never
  reads individually*; `.f.` vs `.v.` is *top-level log-line legibility*, which
  is the thing the module docstring explicitly promises.
- Route A makes `face_selection_identity_of` / `is_face_selection_pinned`
  permanently misnamed (§5.2 / R33). Route B keeps them honest and adds
  `video_selection_identity_of`.

**The case for A that is genuinely strong:**
- Route B's "new pre-B23 era" is a real, permanent tax: every pre-B23 string must
  parse with `video_selection=None`, the peel-from-right logic gains a branch,
  and `test_parse_model_version_is_total_across_all_eras` gains a case that lives
  forever. Route A adds *zero* parser surface.
- Route B needs `compose_model_version` / `ModelVersionParts` / every
  `compose()`-using fixture to grow a field.

**Reviewer's position:** A is acceptable **only if** the design also commits to
(i) redocumenting the `.f.` marker and the two `*_face_selection_*` functions in
`contracts/version.py`, and (ii) ideally adding a `selection_identity_of` name
(alias `face_selection_identity_of` for one release). Without (i) the design
leaves `contracts/version.py` self-contradictory, which tips the balance back
toward B. This is required-change #3.

### 11.2 Route C (per-kind `image_model_version` / `video_model_version`) — correctly rejected

Two literals, two `verify_version_alignment` mappings, touches `/health`, the
idempotency key, every fixture. The design's rejection (§3.2) is sound and
`inference/video_pipeline.py:90-95`'s existing comment already names this "a
serving change".

### 11.3 Not considered: keep `.f.` face-only, add `min_usable_frames` to `gate_fingerprint`

`min_usable_frames` is a verdict→abstention gate — arguably it *belongs* in the
gate fingerprint alongside `min_face_px`. The design puts all four video knobs in
one policy for symmetry with `FaceSelectionPolicy`, which is reasonable, but the
design never explains why the abstention floor goes in the *selection* identity
rather than the *gate* identity. `MediaCaps` is "deployment configuration passed
into a pipeline" and `min_usable_frames` is currently a `VideoPipeline.__init__`
arg, not a `MediaCaps` field — so making it a `MediaCaps`/gate field would be a
larger change. Fine to keep in the policy, but §3 should state the choice.

---

## 12. New Findings

### R32 — `batch_size` output-neutrality is backend-dependent and has a roadmapped trigger

**Severity:** reproducibility, **Low–Medium**. **Status:** OPEN (would be, if B23
ships as designed).

`§3.4` permanently excludes `batch_size` from `VideoSelectionPolicy`. Neutrality
holds **only** while (a) `batch_clips` never pads (`preprocess.py:1184` — true
today), (b) the model backend `score()` is row-independent (`models.py:368` —
true for `StubBackend`), and (c) `per_frame` reconstruction is order/value
preserving (true today). A future backend with a **static batch dimension**
(ONNX / TensorRT — a roadmapped migration, currently a standing constraint
"no ONNX migration") would force padding of the final batch, and the number of
padding rows depends on `batch_size` → `batch_size` becomes output-affecting →
it must move into the policy → another `.f.` move + cache generation. Likewise a
backend using batch-norm in eval mode or cross-frame temporal attention
(the design's own `pooling_rule = "attention"` future) breaks row-independence.

**The design should:** (1) state the full precondition for `batch_size`
neutrality; (2) add a one-line note in `VideoSelectionPolicy`'s "what is NOT here
and why" that a static-shape or cross-frame backend re-opens this; (3) keep
`test_batch_size_is_output_neutral` as the tripwire that catches the regression
when it happens.

### R33 — `contracts.version` describes and names `.f.` as face-only while B23 makes it a composite; no tripwire

**Severity:** documentation + API-contract, **Low**. **Status:** OPEN.

`face_selection_identity_of()` and `is_face_selection_pinned()`
(`version.py:446`, `:455`) — public `contracts` functions — return / assert on a
value that is no longer a face-selection identity for B23+ strings. Their
docstrings, the `FACE_SELECTION_MARKER` docstring (`version.py:119`), and the
module docstring's `face` paragraph (`version.py:33`) all still say "face
selection". §5.7's doc-update table omits `contracts/version.py`. Nothing fails
if the drift widens.

**The design should:** add `contracts/version.py` to §5.7; either rename
`face_selection_identity_of` → `selection_identity_of` (deprecation alias) or
redocument it; add a tripwire in the B23 suite asserting the `.f.` marker
docstring names both policies.

### R34 (candidate — may be a migration note rather than a finding) — fail-closed OR1 + `STUB_ENGINE_VERSION` move makes B23 non-rolling-deployable

Covered in §7.1. Recording it as a candidate so it is tracked: every future
`STUB_ENGINE_VERSION`-moving batch now inherits "workers refuse to start during
the deploy window unless restarts are coordinated or the hatch is set". This is
arguably OR1 *working as intended* (fail loud), but it is a new operational
constraint that did not exist for B20/B22 and the design does not flag it.

*(No new finding is warranted for the `.f.` collision domain (§9.1), the
benchmark cross-boundary note (§7.4), or the missing version-constant enumeration
(§8.1) — those are design-completeness gaps, not defects in the system.)*

---

## 13. Required Changes Before Approval

1. **OR1 worker gate — resolve the `workers → backend` cycle (§5.4, §10.2).**
   Specify exactly how `workers` obtains the configured `model_version` and the
   skip flag without importing `backend`: pick option 1 (env-only, with the
   weaker fallback stated), option 2 (a new `backend`-side composition-root
   `main`), or option 3 (parameter to `run_with_shutdown`). State the residual
   gap of the chosen option and confirm `test_dependency_graph_stays_acyclic`
   still passes.

2. **P2 validation × shim interaction (§5.1, §7.2).** Either specify that the
   shim lowers `min_usable_frames` to `min(min_usable_frames, n_frames)` when
   only `n_frames` is passed, or enumerate every call site and confirm
   `n_frames >= 8`. Rewrite FB2 to use a construction P2 permits (the current
   `min_usable_frames=16, n_frames=9` mechanism is now illegal). Scope
   "output-identical" (§8.2) to the shipped policy.

3. **`contracts/version.py` documentation (§4.3, §5.2, §11.1, R33).** Add
   `contracts/version.py` to §5.7's doc-update list. Update the `.f.` marker
   docstring, the module docstring's `face` paragraph, and the
   `face_selection_identity_of` / `is_face_selection_pinned` docstrings to name
   the composite. Decide whether to rename `face_selection_identity_of` →
   `selection_identity_of` (recommended, with a one-release alias).

4. **Pin inventory completeness (§5.3, §7.3, M-series).** Add to §5.3:
   `tests/stub_handlers.py:433`; the `build_video_analysis` call in
   `test_postprocess.py` (if P16 takes the required-kwarg route);
   `test_face_selection_identity.py`'s in-loop `face_selection_identity_of(...)
   == SHIPPED.identity` assertion (line ~559) as an explicit *semantic* rewrite,
   not "update the literal".

5. **"Byte-identical bundle" claim (§4.1).** Restate PA1 and §3.4 as "identical
   scored output; `timing` and job metadata excluded". Specify how
   `test_batch_size_is_output_neutral` masks `timing` / `job_id` /
   `submitted_at`.

6. **OpenAPI hard gate assumption (§8.3).** Before making "zero structural diff"
   a hard gate, cite `backend/openapi.py`'s signature logic to confirm `default`
   is stripped as prose. If it is not, the `contracts/video.py` change moves the
   signature and the gate must account for it.

7. **P19 factual error (§5.3).** `inference/__init__.py` does not export
   `FaceSelectionPolicy`. Drop P19 or restate it as "add no exports —
   `VideoSelectionPolicy` is `preprocess`-only, matching `FaceSelectionPolicy`".

---

## 14. Recommended Non-Blocking Improvements

- **N4 stored-bundle load test (M1).** Add a video bundle with
  `sampling="uniform+adaptive"` to `test_old_artifacts_still_load`.
- **PA7 completeness (M2).** Assert `default_face_selection().identity` is
  unchanged inside `test_batch_23_moved_only_the_selection_identity`.
- **OR1 partial-mismatch test (M3).** One test where the video engine reports a
  different `.f.` than the image engine and `verify_version_alignment` names the
  video engine.
- **§6.5 rollout guidance (§7.1, R34).** State that B23 needs a coordinated
  worker restart or the `DFP_SKIP_VERSION_ALIGNMENT` hatch during the deploy
  window; add it to the CHANGELOG note.
- **`batch_size` precondition note (R32).** In `VideoSelectionPolicy`'s "what is
  NOT here" docstring, name the static-shape / cross-frame backend as the trigger
  that re-opens the exclusion.
- **Fail-before taxonomy (M4).** Label FB3/FB5/FB6 "new invariant"; identify the
  existing `test_video_identity_gap.py` version-equality assertions as the
  genuine fail-before.
- **`selection_identity` collision sentence (§9.1).** One line: not injective
  over `(face, video)`, acceptable for the same reason as every `fingerprint()`.
- **Benchmark cross-boundary (§7.4).** Extend
  `test_the_benchmark_gate_still_compares_across_the_boundary` for a pre/post-B23
  pair.
- **Version-constant sweep (§8.1).** Add a "grep `_VERSION =`, confirm none
  governs a `model_version`-carrying structure" step to §6.2.
- **`.v.` acknowledgement (§11.1).** In §3.2, acknowledge that Route A degrades
  the log-line legibility the `contracts/version.py` module docstring promises,
  and that this is the real reason (i) above is required.

---

## 15. Final Verdict

**APPROVE WITH CHANGES.**

The design correctly identifies a real reproducibility gap (R29), applies a
pattern validated three times (R17/R19/R25), and reaches the right high-level
decisions: a required-fields policy object, folded into `.f.` rather than a new
segment, `batch_size` excluded, one cache generation, no contract/schema/witness
bump, output-identical for the shipped policy. N4 is a clean bundled fix. OR1 is
overdue and the API/worker split is the right shape.

It is **not** approvable as written because:

- OR1's worker gate (P20) closes a dependency cycle the design does not notice
  and does not resolve (required-change #1);
- the new `min_usable_frames <= n_frames` validation silently breaks the design's
  own referenced fail-before scenario and the shim's "no edit" claim
  (required-change #2);
- `.f.` becoming a composite leaves `contracts/version.py`'s public functions and
  docstrings describing a face-only segment, with no scheduled fix and no
  tripwire (required-change #3, R33);
- the pin inventory, the "byte-identical bundle" claim, the OpenAPI hard-gate
  assumption, and P19's statement about the export surface each have a concrete
  defect (required-changes #4–#7).

All seven are specification fixes, not architecture changes. With them folded in,
the design is sound and the batch should proceed. Two new findings (R32
`batch_size` backend-dependence, R33 `.f.` naming/doc drift) and one candidate
(R34 non-rolling-deployable) should be entered in `REPRODUCIBILITY_AUDIT.md` as
tracked-not-closed.
