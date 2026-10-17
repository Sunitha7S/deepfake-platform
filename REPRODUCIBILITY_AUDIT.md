# Reproducibility audit — S8 Batch 3

> **Status after S8 Batch 11.** R1, R2, R3 (B4), R7, R4, R5 (B5), the
> calibration half of R5 (B6), and **R8 + R11 (B7)** are **closed**. **R6 is
> measured and partially mitigated (B8)**, which opened **R12** and **R13**;
> **R12 is now fully closed** — cache, deployment and forensic identity in B9,
> and **benchmark and calibration identity in B11**, which took the
> `RUN_SCHEMA_VERSION` and `CALIBRATION_WITNESS_VERSION` bumps B9 could not.
> **Batch 10
> answered R13's question — yes, identical media can produce different verdicts
> inside one decode environment — by finding and closing R14 (face selection
> decided by detector output order) and R15 (the face cascade shared across
> threads, which also leaked detections between concurrent jobs).** R13 itself is
> now quantified at a 310× amplification factor and remains deferred to the SCRFD
> migration, with an acceptance criterion.
>
> **Batch 11's architecture review of multi-face semantics
> ([`docs/MULTI_FACE_SEMANTICS.md`](docs/MULTI_FACE_SEMANTICS.md)) raised six new
> findings, R17–R22.** **Batch 12 closed R17** — the face-selection policy and
> crop margin were outside every identity string, and a crop margin of 1.2
> against the shipped 1.3 was measured publishing a different verdict under a
> byte-identical version. **Batch 13 partially closed R18** — the video path has
> no face stage, four artifacts said it did, and every published video bundle
> carried three references to a face track that did not exist; the false claims
> are closed and the capability gap is pinned by a test rather than papered over.
> **Batch 14 closed R19** — face selection ranked on area while the gate it fed
> tested short edge, so the platform could refuse a file containing a scorable
> face; the two metrics are now one, and the batch also corrected the record's
> reason for R19 being latent (Haar does *not* return square boxes — it returns
> boxes within one pixel of square, and the defect needs three) and opened
> **R23**. **Batch 15 investigated R23 and changed no geometry** — it found the
> finding conflated two defects with different triggers (border clipping, which
> is **live today with square boxes**, and occupancy collapse, which is
> SCRFD-armed), corrected two false claims in the codebase, and opened **R24**
> (the video path squashes whole frames onto a square, 1.778× on 16:9, live on
> every video bundle), **R25** (three geometry knobs outside every identity
> string) and **R26** (the published box is not the scored region).
> **Batch 16 closed R21** — `faces[]` had no uniqueness or consistency invariant,
> so a bundle carrying two faces at one `track_id` with contradictory scores, or
> a decided headline `p_fake` equal to no published face's score, was a valid
> `EvidenceBundle`; two `model_validator`s close both, output-identical, no
> fingerprint move, and R16's implementation now has an invariant to build
> against. **Batch 17 closed R22** — the "branch B scores the whole image" claim
> was documentation-as-data with no wire surface, and the code had already
> decided: branch B scores the same prepared tensor as branch A (the face crop
> for images, the whole frame for video), measured byte-for-byte. The claim is
> corrected, the one fixture depicting an unbuilt whole-image B (`image_no_face`)
> is annotated as a target, and the contradiction is pinned — 0 fields, 0
> validators, 0 fingerprints, no bump. **Batch 18 closed R20** — `face_px`'s
> description named no metric for "largest" (area? short edge?) and mentioned
> neither the `0`-means-no-face sentinel nor the video `null`; B14 had already
> removed the semantic half by unifying the ranking and reporting metric, so B18
> corrects the prose and proves — by rendering the schema with and without the
> change — that one `Field` description string moves and nothing else. **Batch 19
> reviewed R6's `frame_count` residual** — `MediaInfo.frame_count` is a
> libavformat quantity, trusted verbatim, feeding uniform frame sampling so a
> one-step miscount moves ≥50% of the scored frames; B19 splits it, pins the
> amplifier, and promotes the reproducibility half to **R27** (the decode
> identity omits the demuxer — executed: one identity, two frame counts, two
> `p_fake`s, one cache key). **Batch 20 closed R27** — re-derived the video
> decode-path component list from first principles, found libavcodec captured but
> **libavformat + libswscale + libavutil** not, and grew `decode_identity`'s one
> FFmpeg field to every FFmpeg library the video path links. 4/4 protections
> blind → 0/4, executed. One cache generation (`.d.` `c20a064a` → `a97ccb68`); no
> contract/schema/OpenAPI/witness-version change; residuals D8–D10 recorded.
> R9, R10, R13, R16, R23–R26 and R18's capability half remain open.
> See the resolution sections at the end for what was done and what each fix does
> *not* guarantee. The body below is preserved as the Batch 3 findings record —
> it is the evidence the fixes were built against, and rewriting it would erase
> the reasoning.

**Scope.** Everything that determines a reported number: the corpus index, the
split, the calibration, the threshold, the metrics, and the version strings that
claim to identify all of it. Serving-path correctness is out of scope except
where it feeds a benchmark.

**State at audit.** 28 suites, 729 tests, zero failures. `CONTRACT_VERSION`
1.0.0, unchanged. No weights, no datasets, no GPU — which bounds what this audit
can claim, and the bound is stated wherever it applies.

**Method.** Read the code and the tests, then attempt each drift mechanism
against the running system. Findings marked **measured** were reproduced; the
rest are read from the code and marked accordingly.

---

## Summary

| # | Finding | Class | Severity | Status |
| --- | --- | --- | --- | --- |
| R1 | Calibration set *ordering* changes `effective_model_version` | calibration + version | High | **Closed in B4** — measured |
| R2 | A slice that disappears between runs is never gated | benchmark | High | **Closed in B4** |
| R3 | `BenchmarkRun` records no split-lock hash or protocol | benchmark | Medium | **Closed in B4** |
| R4 | The lock seals the assignment, not the identity metadata behind it | split | Medium | **Closed in B5** — measured |
| R5 | Nothing binds a calibration run to its protocol role | calibration | Medium | **Closed** — benchmarks B5, calibration B6 — measured |
| R6 | Decode nondeterminism (ffmpeg/OpenCV build) is unaddressed | determinism | Medium | **Measured in B8, `frame_count` residual characterised in B19.** Pixel-drift half: partially mitigated (B8), decoder now in the cache key (B9/B11). **`frame_count` residual → PARTIALLY CLOSED:** the amplifier (uniform sampling, ≥50% of scored frames move on a ~one-step miscount) is measured and pinned; its serving-path fix is a video-face-stage acceptance criterion; its reproducibility half is split out as **R27** |
| R7 | No dependency pinning; `numpy` is not even declared | determinism | Medium | **Closed in B5** |
| R8 | `band_width = 0.20` is still a placeholder | calibration | Medium | **Closed in B7** — measured |
| R9 | Quarantined items make a lock knowingly leaky | split | Low | By design, reported every run |
| R10 | GPU/training nondeterminism entirely ahead of us | determinism | Deferred | No models yet |
| R11 | Fitted threshold and configured band width can be jointly unrepresentable | calibration | Medium | **Closed in B7** — structurally unrepresentable |
| R12 | The decoder is outside every identity string | determinism + version | Medium | **Closed** — cache/deployment/forensic (B9), benchmark + calibration (B11) — measured |
| R13 | Face-crop instability amplifies decoder drift | determinism | **Medium–High** | **Open** — quantified in B10 at **310× amplification, 25.7% of decisions**; deferred to SCRFD with an acceptance criterion |
| R14 | Face selection decided by detector output order | determinism | Medium | **Closed in B10** — measured |
| R15 | The face cascade was shared across threads | determinism + correctness | **High** | **Closed in B10** — measured; leaked detections between concurrent jobs |
| R16 | One face's verdict is published as the media's | contract | Low | **Open** — by design, needs a deliberate decision |
| R17 | Face-selection policy and crop margin are outside every identity string | determinism + version | **Medium–High** | **Closed in B12** — measured; 3 of 3 verdicts reachable by crop margin alone |
| R18 | The video path has no face stage, and the contract says it does | correctness + contract | Medium | **Partially closed in B13** — dangling `track_id` and every false claim closed; the faceless-clip verdict stays open and is pinned by a test |
| R19 | The gate ranks faces on area and tests short edge | correctness | Medium | **Closed in B14** — one metric now ranks, gates and reports; activation threshold measured at 3 px against Haar's 1 px |
| R20 | `QualityReport.face_px` does not describe the value it carries | documentation | Low | **Closed in B18** — semantic half removed by B14; B18 corrects the prose (names `size_px`, the `0` sentinel and the video `null`), proven to change one `Field` description string and nothing else — 0 fields, 0 validators, 0 fingerprints, 0 version bumps |
| R21 | `faces[]` has no uniqueness or consistency invariant | contract | Low | **Closed in B16** — measured; two validators, output-identical, prior contract accepted the published face score across all of [0, 1] |
| R22 | Branch B is declared whole-image and scored on the face crop | documentation + contract | Low | **Closed in B17** — contract clarification; measured (B scores the crop byte-for-byte, `image_no_face` depicts an unreachable path); 0 wire fields, 0 validators, 0 fingerprints; no bump |
| R23 | Crop geometry: clipping distorts, and the long edge collapses occupancy | correctness + documentation | **Medium** | **Characterised in B15; documentation half CLOSED, correctness halves open.** The distortion is **live today** with square boxes — 13.0% stretch at a border — and is *not* SCRFD-gated; only the occupancy half is |
| R24 | The video path resizes whole frames onto a square | correctness | **Medium** | **Open** — live on every video bundle, 1.778× squash on 16:9; pinned by a test; fix belongs with the video face stage |
| R25 | Crop geometry inputs outside every identity string | determinism + version | **Medium** | **Closed in B22.** `input_size` folded into `weights_identity` (functionally threaded); `interpolation_rule` + `crop_clip_rule` added to `FaceSelectionPolicy` (label closures pinned by re-derivation tests). Model identity `stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387` → `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9`. **Output-identical**, one cache generation, no `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` / OpenAPI change — executed. Opened **R28**. Witness incidental-binding residual recorded |
| R26 | The published `bbox` is not the region the model scored | contract | Low | **Open** — and `face_track`'s coordinate clamp will diverge from `crop_face` under a detector emitting boxes across the frame edge |
| R28 | `resize`'s interpolation predicate area-filters an axis it is enlarging | correctness | Low | **Open — deferred (B22).** `INTER_AREA if h > size or w > size` low-passes an upsampled axis when a crop straddles `input_size` (500×300 → 384: 0.040/255). Fixing the predicate moves shipped pixels, so it is not in B22; B22 named the current behaviour in `FaceSelectionPolicy.interpolation_rule` and pinned it. Rides the next `.f.`-moving batch (R23/SCRFD or R16) |
| R27 | The decode identity omits the demuxer; `frame_count` is a demuxer output | determinism + version | Medium | **Closed in B20.** `decode_identity`'s FFmpeg component grew from libavcodec alone to every FFmpeg library on the video path (avcodec + **avformat** + **swscale** + avutil; avresample excluded — audio only). All four protections (cache key, deployment identity, benchmark identity, calibration identity) now separate a demuxer/scaler difference — **executed, 4/4 blind → 0/4**. `STUB_ENGINE_VERSION` `.d.` `c20a064a` → `a97ccb68` (one cache generation); no `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` / OpenAPI change. Residuals D8–D10 (swscale SIMD, wheel patch, same-version different build) recorded at the time, not closed by this batch — D8/D9 were the *image*-path analogues and closed in Batch 26 (R30); D10 (a video- and image-path residual alike) stays open, see the R30 row |
| R29 | Video frame-selection and pooling parameters are outside every identity string | determinism + version | **Medium** | **Closed in B23.** New `VideoSelectionPolicy` (`inference/preprocess.py`) — `n_frames`, `min_usable_frames`, `pool_divisor` (functional; the pipeline reads them) plus `sampling_rule` / `pooling_rule` (label closures, also the published `VideoAnalysis.sampling` / `.pooling` values). `selection_identity(face, video) = fingerprint({"face": …, "video": …})` feeds the `.f.` segment from **both** pipelines, so the single `STUB_ENGINE_VERSION` literal and `contracts/version.py` are untouched. Model identity `…f.251d22e9` → `…f.e629642c` (`default_face_selection().identity` unchanged at `251d22e9`). `batch_size` excluded, proven output-neutral. **Output byte-identical**, one cache generation spanning both media namespaces (B20 precedent), no `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` / OpenAPI-structural move — executed. See the Batch 23 section and `docs/VIDEO_SELECTION_IDENTITY.md` |
| N4 | `VideoAnalysis.sampling` default `"uniform+adaptive"` is stale — nothing adaptive runs | documentation + contract | Low | **Closed in B23.** `sample_frame_indices` is pure uniform stride; the live bundle already emitted `"uniform"` while the contract default, `contracts/fixtures.py` and `fixtures_out/video_*.json` said `"uniform+adaptive"`. Corrected to `"uniform"` everywhere; the wire value is now sourced from `VideoSelectionPolicy.sampling_rule`. `Field(default=)` value change only — no structural / OpenAPI-signature move |
| OR1 | `verify_version_alignment` was defined (B9) and never called at startup | determinism + operability | Medium | **Closed in B23.** Worker boot runs the real fail-closed check (`workers.runner.verify_engine_alignment_at_startup`, called from `run_with_shutdown` before the loop; `DFP_SKIP_VERSION_ALIGNMENT` is a logged escape hatch). API boot runs a stdlib-only format check in `backend.app.lifespan` (`is_composed` / `is_environment_pinned` / round-trip, including the `except ValueError` branch for a marker-bearing unparseable string — SA10) — the API process may not import `inference`. `workers` may not import `backend` (DAG), so the worker gate takes the configured literal as an argument or `$DFP_MODEL_VERSION` and the **only** escape hatch is the env var `DFP_SKIP_VERSION_ALIGNMENT` (no `Settings` field — it would be dead config; B23-R3 §1.8). `test_configured_model_version_matches_the_engines` stays the pre-merge guard for the default; the `backend`-side composition root that wires the config-default path into the full check (and re-introduces a wired `Settings.skip_version_alignment`) is **B24/OR1-b**. **OR1-b closed in Batch 25** — `backend/worker_main.py` (new) calls the gate once, directly, with both `configured_model_version=settings.model_version` and `skip_version_alignment=settings.skip_version_alignment`; `run_with_shutdown` gained a `configured_model_version` parameter (backward-compatible default `None`) but stays single-runner-shaped — the composition root does not call it, since `workers.registry.build_all_workers` returns two runners and two `run_with_shutdown` calls would silently overwrite one worker's `add_signal_handler` registration for the same signal. See `docs/OR1B_DESIGN.md`, `docs/OR1B_SPEC_RESOLUTION.md`. No `.d.` / `.f.` / `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` / `LOCK_VERSION` movement |
| R30 | Image decode libraries (libjpeg-turbo / libpng / libwebp) are outside `decode_identity` | determinism + version | Medium | **Closed in Batch 26.** New `image_codec_version` (libjpeg-turbo / libpng / libwebp, captured verbatim) and `image_codec_simd` (digest-only — performance, not pixels) fields on `DecodeEnvironment` (`inference/decode_env.py`), via a new sibling function `_image_codec_facts()` that leaves `_build_facts()` and its other call sites untouched. **Structural-completeness fix, not a response to a measured divergence** — a cross-build check (OpenCV vs. Pillow) found every accepted MIME type (JPEG/PNG/WebP) decoding bit-for-bit identically across a major libjpeg-turbo version gap, a different libpng/libwebp, and stock zlib vs. zlib-ng; the identity grows anyway on the same over-sensitive-by-design principle as every other segment (false miss costs compute, false hit costs a wrong verdict). **D9 (`cv2.__version__` drops the wheel's 4th digit) folds in and closes with it** — `image_codec_version` directly supersedes the blind spot D9 described. **D8 (SIMD path not in identity) is only PARTIALLY closed** — its image half (libjpeg-turbo SIMD) is now in `image_codec_simd`, a `cpu_dispatch`-class digest-only fact; its **video half (libswscale SIMD) stays open**, exactly as `docs/B24_ADVERSARIAL_REVIEW.md` / `docs/B24_R2_DESIGN.md` scoped it — no measured occurrence, out of this batch's scope, unrelated code path. zlib is recorded nowhere (`inflate()` is RFC-1951-deterministic, measured identical). `STUB_ENGINE_VERSION` `.d.` `a97ccb68` → `c9d4025c` (one cache generation, both media namespaces — N8). No `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` (already 1.3.0 post-R31, independently re-verified unaffected) / `LOCK_VERSION` / OpenAPI change. New finding **D12** (parser silently omits a codec whose build-info line lacks a `(ver ...)` suffix — no known occurrence, the C2 ratification). **D10** (same version string, different build) stays open — no fix exists, would need a binary hash. See `docs/R30_BUNDLE_DESIGN.md`, `docs/DECODE_IDENTITY_COMPLETENESS.md` |
| R31 | `CalibrationWitness` digest omits `effective_model_version` / `input_size` / gate / `.f.` | determinism + version | Medium | **Closed.** New `ModelSpec.pre_calibration_identity` (`inference/models.py`) — weights + gate + selection, composed the same way as `effective_version` minus the calibration segment, computed before calibration is fitted so there is no circularity with `calibration_fingerprint` (which deliberately excludes the witness). Threaded through `fit_calibration`/`CalibrationWitness`/`verify_calibration_witness` (`evaluation/calibration_protocol.py`) as `pre_calibration_identity`, inside the digest for schema `1.3.0` (`SCHEMA_VERSIONS_WITH_PRE_CALIBRATION`), exact structural twin of the Batch 11 `decode_identity` addition. **Live-demonstrated, not merely argued** — `docs/R31_DESIGN.md` §2 fits two configurations differing only in `pre_calibration_identity` against byte-identical logits and shows the pre-fix witness collides (`d7ed75b9` == `d7ed75b9`) while a genuinely different fit separates (`04c3d5c7`). `CALIBRATION_WITNESS_VERSION` `1.2.0` → `1.3.0`; **zero stored witnesses existed anywhere to migrate** (confirmed by search). Companion fix: `BenchmarkReport.validate()` (`evaluation/benchmark.py`) now passes the *already-existing* `decode_identity` into its *already-existing* `verify_calibration_witness()` call, closing a second, independently-found gap (the field was recorded but never cross-checked). A second cross-check (model config, at the `BenchmarkRun` level) was scoped and **deliberately deferred** — it would need its own new `BenchmarkRun` field and its own `RUN_SCHEMA_VERSION` bump, which `docs/R31_SPEC_RESOLUTION.md` Decision 2 found unsafe to derive from `effective_model_version` (an opaque, often-unparseable label in this codebase's own benchmark fixtures) without that separate schema change. **No `.d.` / `.f.` / weights / `STUB_ENGINE_VERSION` / `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / split-lock `LOCK_VERSION` movement.** See `docs/R31_DESIGN.md`, `docs/R31_SPEC_RESOLUTION.md` |
| N3 | Advisory heuristic constants outside every identity | documentation | Low | **Open — deliberately.** `confidence_interval`'s `0.03` / `0.5` (`postprocess.py`), `_blur_score`'s `/100.0`, `resolution_class` edges. Advisory-only fields; B23 declined the optional re-derivation tripwires (no scope expansion). Add tripwires when one of these is next edited |
| N5 | No dependency lockfile; opencv / numpy are version ranges | determinism | Low | **Closed in B24.** New `uv.lock` (committed) resolves and hash-pins the full dependency tree; `[tool.uv].override-dependencies` in `pyproject.toml` pins `opencv-python==4.8.1.78` / `numpy==1.26.4` — the exact versions every shipped identity literal (`.d.`=`a97ccb68`, `.f.`=`e629642c`) was computed against — so a first `uv lock` cannot silently resolve a newer in-range build and move `.d.` under an "identity-neutral" banner (`docs/B24_N5_DESIGN.md` §4.1, the G1 gate). `[project.dependencies]` ranges are untouched; the lock constrains *resolution*, not the declared intent. **Zero identity movement** — `.d.` / `.f.` / `STUB_ENGINE_VERSION` / `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` / the split-lock `LOCK_VERSION` all verified unchanged (new tests in `tests/test_dependencies.py`). New `.python-version` (pinned `3.12`) and `.github/workflows/tests.yml` (unexercised — no git remote exists yet) added alongside the lock; `numpy==1.26.4` has no installable wheel past CPython 3.12, which is what makes the interpreter pin load-bearing rather than cosmetic. See `docs/B24_N5_DESIGN.md`, `docs/B24_R2_DESIGN.md` §7.2, `docs/B24_ADVERSARIAL_REVIEW.md`. R30 (image codec identity) and the D8–D10 residuals remain open, deferred to a "decode-completeness pass 2" batch now that N5 makes any future `.d.` literal reproducible across machines. Building the environment from `pyproject.toml` alone (rather than an ad hoc machine) also surfaced **N9** (undeclared `httpx2`), fixed in the same batch |
| N6 / N7 | Fit-optimiser config and `np.quantile` method are unrecorded | determinism | Low | **Open, corrected during R31's re-derivation (`docs/R31_SPEC_RESOLUTION.md` Decision 3) — not folded into R31.** **N6** (`fit_temperature`'s `lo`/`hi`/`iterations`) is real — these values feed the fitted `temperature`, which is in the digest — but has **zero live variability today**: `fit_calibration` is the only caller anywhere in the repository and never overrides the defaults, so there is nothing for two configurations to converge from yet. Defer until a future caller actually parameterizes the fit. **N7, as stated, does not apply to `CalibrationWitness` at all** — traced to source: the witness's actual threshold path (`fit_calibration`'s default `margin=DERIVE` → `evaluation.band.derive_operating_point` → `evaluation.thresholds.at_target_fpr`) contains **no `np.quantile` call anywhere**; it is an exact discrete scan over observed scores. The `np.quantile` calls that exist (`inference.calibration.threshold_at_fpr`, an unused parallel/exploratory function per its own module's docstring; `evaluation.band.py`'s explicitly-not-shipped "Approach C"; two `bootstrap_threshold_ci` diagnostics) are all outside `fit_calibration`'s call graph. This row should not be read as "the same class of gap as R31, just deferred" — N6 is that; N7 is a different, unconnected observation |
| N8 | Video-shaped `decode_identity` applied to the image path | documentation | Low | **Closed — accepted architectural cost (decided in `docs/B24_R2_DESIGN.md` §4, written back in this pass).** `workers/engine_registry.py:80-88` unconditionally builds both `ImagePipeline` and `VideoPipeline` — there is no single-media-kind deployment — so the one `decode_identity()` is necessarily a union over both decode paths; a library change in either path regenerates both content-cache namespaces, a bounded false-miss, never a wrong verdict. This is the single-literal identity trade already documented in `docs/RUN_IDENTITY.md` §3 and paid by B20 (`.d.`) and B23 (`.f.`). Its only alternative — per-media-kind identities — was evaluated and rejected in `docs/VIDEO_SELECTION_IDENTITY.md` §3. **No further action.** A future decode-completeness batch (R30) enlarges this union symmetrically and does not change the disposition. **Confirmed in Batch 26:** R30's `.d.` move regenerated both the image and video content-cache namespaces, exactly as predicted; disposition unchanged |
| N9 | `httpx2` (required by `fastapi.testclient.TestClient`, used in `test_backend.py` / `test_startup_alignment.py`) was undeclared anywhere | reproducibility + test infrastructure | Low | **Discovered and closed in B24 (N5).** Every test run before B24 passed only because the ad hoc development machine happened to have `httpx2` (or the deprecated `httpx`) installed for unrelated reasons — invisible until N5's isolated `uv`-managed environment was built from `pyproject.toml` alone, at which point `uv run python tests/run_all.py` failed at import (`starlette.testclient` raises `RuntimeError` with neither package present). **Root cause:** `tests/import_audit.py`'s AST scan (and therefore the R7 suite, `tests/test_dependencies.py`) covers only the seven shipping packages (`RUNTIME_PACKAGES`) — it does not scan `tests/` itself, so a test-only import is structurally invisible to the dependency audit regardless of whether it is declared. **Fix:** `httpx2` added to `[project.optional-dependencies].dev` (the non-deprecated name `starlette.testclient` prefers over `httpx`, verified by reading `starlette/testclient.py`'s import fallback directly, not assumed). **Not fixed:** the `import_audit.py` blind spot itself (`tests/` remains unscanned) — out of scope for this batch; a future batch extending the audit to test-only imports would need to decide whether `tests/` becomes a scanned tree or stays exempt by design. Zero identity movement — this is a `dev`-extra-only, test-infrastructure change |
| R32 | `batch_size` output-neutrality is backend-dependent | determinism + version | Low | **Tracked (B23-R2 gate).** Neutral only while `batch_clips` does not pad, the backend `score()` is row-independent, and `scores` reconstruction preserves order/values — all true for `StubBackend`. A static-shape backend (ONNX / TensorRT) or a cross-frame backend (temporal attention — the `pooling_rule="attention"` future) re-opens the exclusion and forces `batch_size` into `VideoSelectionPolicy`. The `VideoSelectionPolicy` docstring names the **full** precondition — (a) `batch_clips` no-pad, (b) row-independent `score()`, (c) order/value-preserving reconstruction — and the ONNX/TensorRT/attention re-opening trigger, held by `test_the_batch_size_exclusion_note_is_complete` (PA11); `test_batch_size_is_output_neutral` is the behavioural tripwire. PA11 was specified by the B23-R2 design and gate, dropped in the R2→R3 transition, and restored by the post-audit cleanup batch (`docs/B23_FRESH_INDEPENDENT_AUDIT_2.md` D1/D2) |
| R33 | `contracts.version` named/documented `.f.` as face-only while B23 makes it a composite | documentation + API-contract | Low | **Closed in B23-R3.** Aliases `selection_identity_of` / `is_selection_pinned` (`contracts/version.py`) and `ModelSpec.with_selection` added — thin wrappers, identical bodies, **no parser change**. Pre-B23 names kept as permanent aliases with an explicit "never compare to `default_face_selection().identity`" banner. Module docstring, `.f.` marker docstring and both reader docstrings updated to name the video half; `inference/models.py` docstrings likewise. Tripwires: `test_the_f_marker_and_public_readers_document_both_policies`, `test_selection_identity_of_returns_the_composite_and_is_the_honest_name`, `test_model_spec_with_selection_is_the_honest_alias`. A full rename is a B24 candidate |
| R34 | Fail-closed OR1 + `STUB_ENGINE_VERSION` move ⇒ B23 is not rolling-deployable without coordination | operability | Low | **Tracked (B23-R2 gate) — OR1 working as intended (fail loud).** During a pod-by-pod rolling update a worker whose configured literal disagrees with its engine build refuses to start. Two safe paths: a coordinated worker restart at cutover, or `DFP_SKIP_VERSION_ALIGNMENT=true` held on for the window (which reduces OR1 to B20/B22 semantics — brief cache misses, no worse). Runbook line added to `docs/VIDEO_SELECTION_IDENTITY.md`. A new operational constraint on every future `STUB_ENGINE_VERSION`-moving batch |
| R35 | `build_video_analysis` keeps hardcoded `sampling` / `pooling` kwarg defaults duplicating `default_video_selection()` | determinism + version | Low | **Closed in B23-R3** by tripwire `test_build_video_analysis_defaults_match_the_shipped_policy` (an `inspect.signature` default-equality check). The pipeline always passes explicit policy-sourced values, so the defaults are exercised only by study / test callers and could rot undetected into a stale second source of truth for the `VideoAnalysis` wire value. Making the kwargs required is a B24 candidate |
| R36 | `VideoAnalysis.sampling` / `.pooling` `Field(default=)` and the fixture literals are a free-floating second copy of `default_video_selection()`'s labels | determinism + documentation | Low | **Closed in the post-audit cleanup batch** (`docs/B23_FRESH_INDEPENDENT_AUDIT_2.md` E1) by tripwire `test_the_video_analysis_wire_defaults_match_the_shipped_policy`. **Source-level tying is architecturally forbidden**: `contracts` is a leaf package (`contracts → ∅`), so `contracts/video.py` importing `inference.preprocess.default_video_selection` would create a `contracts ↔ inference` cycle the DAG test forbids. The labels therefore stay hand-copied literals; the tripwire holds `VideoAnalysis.model_fields["sampling"|"pooling"].default` **and** the three generated `fixtures_out/video_*.json` values equal `default_video_selection()`. This is the authoritative wire default (what an omitted field serialises to, what the OpenAPI/JSON-schema publishes) — R35's PA14 guards only `build_video_analysis`'s kwarg default. MG5 additionally pins the scoped `$defs/VideoAnalysis/properties/{sampling,pooling}/default` in the generated schema. Blast radius was schema + fixture staleness only (server responses always override); the N4 defect class, previously un-guarded for `pooling` |

R17–R22 were raised by the Batch 11 architecture review of multi-face semantics,
not by the Batch 3 audit; R23 was found by the Batch 14 trace and R24–R26 by the
Batch 15 geometry trace. They are listed here so one table remains the index of
every open finding. Their evidence is in
[`docs/MULTI_FACE_SEMANTICS.md`](docs/MULTI_FACE_SEMANTICS.md),
[`docs/FACE_SIZE_POLICY.md`](docs/FACE_SIZE_POLICY.md) and
[`docs/CROP_GEOMETRY.md`](docs/CROP_GEOMETRY.md).

---

## 1. What can still cause benchmark drift?

### R1 — Reordering the calibration set changes the model's identity (High, measured)

`fit_temperature` is a fixed 200-iteration ternary search with no randomness, so
it is deterministic for a given array. It is **not** invariant to the *order* of
that array: numpy's pairwise summation inside the NLL is not associative in
floating point, and the ternary-search comparisons flip near convergence.

Measured on 5 000 synthetic samples, same data, shuffled:

```
T unshuffled : 2.1937994756748207
T shuffled   : 2.193799452081738     (Δ ≈ 2.4e-8)
fingerprint  : 5f2e328e  →  f9c4358e
```

`contracts.version._canonical` puts floats through `repr` on purpose — so that
`1.0000000000000002` and `1.0` are different calibrations — which means a 2.4e-8
difference produces a *completely different* calibration fingerprint, and
therefore a completely different `effective_model_version`.

Consequences, in order of how annoying they are to diagnose:

- The result cache misses everything after a re-run that changed nothing.
- Two runs of the same calibration on the same data report as two models, so
  `regression_gate`'s "same effective version" note never fires when it should.
- A version string that claims to identify a model does not reproduce.

It is *not* a correctness problem: a temperature differing at the eighth decimal
moves no verdict.

**Fix (recommended for Batch 4).** Round the temperature to a fixed precision
before it enters the fingerprint — 9 significant figures is far below anything
that changes a decision and far above float noise — and/or sort the calibration
set canonically by `media_id` before fitting. Do both: the rounding fixes the
fingerprint, the sort fixes the temperature itself.

### R2 — A slice that disappears between runs is never gated (High, read)

`regression_gate` builds `by_name` from the **incumbent's** slices and then
iterates the **challenger's**:

```python
by_name = {s.name: s for s in incumbent.slices}
for s in challenger.slices:
    before = by_name.get(s.name)
    if before is None:
        notes.append(f"slice {s.name!r} is new and has no baseline")
        continue
```

A slice in the incumbent but absent from the challenger is never examined. It
does not fail, and it does not even produce a note.

Two ordinary things make a slice disappear:

- `run_benchmark` silently `continue`s past any slice with fewer than two
  classes present. Acquire a batch of authentic-only low-resolution media and
  the `low` resolution slice becomes single-class and vanishes.
- A dimension is renamed or re-bucketed, so `"heavy"` becomes `"c40"`.

Either way the two-scope gate quietly becomes a one-scope gate, and the model
that collapsed on heavy compression ships because the check that would have
caught it did not run. This is the same failure shape as the four Batch 0
invariants: the test and the code configured from the same place.

**Fix.** Fail the gate on a slice that had a baseline and is now missing, unless
the caller explicitly retires it. Report skipped single-class slices in
`BenchmarkRun` rather than dropping them silently.

### R3 — A run does not record which split or protocol produced it (Medium, read)

`BenchmarkRun` carries `dataset_fingerprint` and `effective_model_version`, and
`REQUIRED_REPORT_FIELDS` now also demands `split_lock_hash` and `protocol` — but
nothing populates them, because `BenchmarkRun` has no such fields.

The fingerprint covers the *evaluated items*, so a changed split does change it
and `regression_gate` will refuse the comparison. That is the safe direction.
What is missing is the diagnosis: the gate says "different datasets, not
comparable" and cannot say *why*, when the answer — the lock advanced a
generation — is one field away.

**Fix.** Add `split_lock_hash`, `lock_generation` and `protocol` to
`BenchmarkRun` with defaults, and populate them from the lock. Deliberately not
done in this batch: it changes a Batch 2 dataclass and its JSONL history format,
which wants its own migration note.

### R4 — The lock seals the assignment, not the metadata that justified it (Medium)

`content_hash` covers the assignments. It does not cover `source_id` or
`subject_id`, which are what made those assignments leak-free. Edit an index so
two actors merge into one identity and the lock file is still perfectly valid and
perfectly sealed.

`verify_lock` catches it — it rebuilds components from the live corpus on every
call and reports both the spanning identity and the merge conflict. So the
mitigation exists; what does not exist is anything forcing it to run.

**Fix.** Have `create_lock` record an identity fingerprint over
`(media_id, source_key, subject_key)` and have `load` compare it when items are
available, so a silent re-label is caught at load rather than at whatever point
someone remembers to audit.

### R8 — `band_width = 0.20` is still a placeholder (Medium, known)

Unchanged since Batch 1 and repeated here because it is the largest single
unjustified number in the system. It was picked, not derived. `select_band_width`
and the risk–coverage sweep exist to derive it and have never been run against
real scores. Measured on synthetic data, widening 0.02 → 0.20 costs 13 points of
coverage and cuts selective risk 2.6× — a trade nobody has actually chosen.

Also unchanged: `target_fpr` is an upper bound, not the achieved rate, because
`band_hi` is the decision boundary rather than `value`.

---

## 2. What can still cause split drift?

**Closed by this batch.** Regenerating instead of extending was the entire
mechanism, and it is now structurally prevented: `extend_lock` re-checks every
frozen assignment against its output and raises rather than writing a lock that
moved one. `test_growth_moves_items_when_the_split_is_regenerated` asserts the
old failure still exists (20 actors grown to 40 does move items), and
`test_the_lock_does_not_move_them` asserts the lock stops it.

Remaining paths, all narrow:

- **R4 above** — identity metadata edited underneath a sealed lock.
- **Not running `verify_lock`.** Every guarantee here is conditional on somebody
  checking. A lock nobody verifies is a file, not a control.
- **R9 — quarantine.** `on_conflict="quarantine"` deliberately produces a lock
  that contains a spanning identity: the freeze wins and the leak is accepted.
  This is reported by `verify_lock` on every run and is Low severity only because
  it cannot happen silently — the default is to refuse.
- **Adopting the lock over an existing leaky split.** Freezing a leaky split
  preserves the leak forever. Documented in the format spec; not detectable
  automatically, because the lock cannot know the split it was handed was not
  identity-aware.
- **Corpus shrinkage** is refused outright by `extend_lock` and reported by
  `verify_lock.absent`.

---

## 3. What can still cause calibration drift?

- **R1**, above — the ordering sensitivity. This is the live one.
- **R5 — nothing binds a calibration run to its protocol role.** `select()` will
  return the right items for `DatasetRole.CALIBRATION`, and no code path is
  obliged to call it. A future training script that fits a temperature on the
  validation split — the exact mistake `validate_protocol` exists to warn about —
  would pass every test in this repository, because the protocol object is not
  wired into anything that runs.
- **Recalibration is versioned but not gated.** The calibration fingerprint
  correctly includes threshold and band (fixed in Batch 1), so a recalibration
  changes `effective_model_version` and invalidates the cache. Nothing requires
  the recalibration to have been *validated* first — `holdout_temperature_check`
  is available and optional.
- **ECE remains binned and discontinuous.** Mitigated by quantile binning and by
  reporting NLL and Brier alongside; the guidance that NLL and Brier win when
  they disagree with ECE is in the module docstring and in nobody's checklist.

---

## 4. What remains non-deterministic?

**Deterministic and verified.** Split assignment (seeded, `hashlib` not `hash()`,
so `PYTHONHASHSEED` cannot touch it). Lock serialisation, byte for byte —
`test_identical_corpus_produces_an_identical_lock` and
`test_corpus_ordering_does_not_change_the_lock`. SVG rendering, byte for byte,
snapshot-pinned. `StubPredictor` scores. Every bootstrap (`default_rng(seed)`).
Frame index selection. Temperature fitting for a fixed input array.

**Not deterministic:**

- **R6 — media decode.** OpenCV/ffmpeg builds do not agree on the pixels of a
  given frame. Different SIMD paths, different libjpeg, different H.264 decoder
  behaviour on damaged streams. The pipeline walks the file once with
  `grab()`/`retrieve()`, which is correct and fast, and does not make the decoder
  version-independent. A different container build will produce different scores
  for the same file, and no version string in this system records which decoder
  produced a result. **Untestable here** — there is no corpus, and the two video
  fixtures in the suite are deliberately malformed (the `moov atom not found`
  lines in the test output are expected).
- **R7 — no dependency pinning.** `pyproject.toml` declares only
  `pydantic>=2.9,<3`. `numpy` is imported throughout `evaluation` and `inference`
  and is **not declared at all**; neither are `opencv-python`, `torch` or
  `onnxruntime`. `np.quantile` sits directly under `threshold_at_fpr`, so a
  change to its default interpolation would move every operating point. Nothing
  would flag it.
- **R10 — training.** Entirely ahead of us: cuDNN autotuning, TF32 on Ampere,
  non-deterministic scatter/atomics, dataloader worker ordering, AMP loss
  scaling. None of it is addressed because there is no training loop yet. Budget
  for it explicitly rather than discovering it during the first real run.
- **`datetime.now(UTC)`** in `inference/models.py`, `postprocess.py` and
  `timing.py`. Correct and intended — these are wall-clock facts about a
  particular execution, not inputs to a score — but they mean a serialised
  `EvidenceBundle` never round-trips byte-identically, and any future snapshot
  test over a bundle must exclude them.

---

## 5. What remains unversioned?

| Thing | Versioned? | Consequence |
| --- | --- | --- |
| Weights, calibration, gate caps | Yes — `effective_model_version` | — |
| Threshold and band | Yes — inside the calibration fingerprint (Batch 1) | — |
| Contract schema | Yes — `CONTRACT_VERSION` 1.0.0 | — |
| Split lock format | Yes — `lock_version`, `generation`, `parent_hash` | — |
| Split lock ↔ benchmark run | **No** | R3 |
| Protocol ↔ benchmark run | **No** | R3 |
| Corpus identity metadata | **No** | R4 |
| Evaluation harness itself | **No** | A change to `roc_auc` or to `operating_point` moves every historical number in the JSONL history with nothing recording that the ruler changed. |
| numpy / OpenCV / ffmpeg / torch | **No** | R6, R7 |
| Slice definitions | **No** | Re-bucketing `resolution` renames every slice; R2 then hides the consequence. |

The two that matter most are the harness version and the decode stack. Both are
invisible in a result and both silently rewrite history: the first changes what
the numbers mean, the second changes what the numbers are.

---

## Recommended next batch

In priority order, and deliberately small:

1. **Fix R1.** Round the temperature before fingerprinting and sort the
   calibration set. Half a day, removes a class of phantom model versions.
2. **Fix R2.** Fail the gate on a vanished slice; record skipped single-class
   slices on the run. This is the finding most likely to let a real regression
   ship.
3. **Fix R3.** Add `split_lock_hash`, `lock_generation` and `protocol` to
   `BenchmarkRun`, with a migration note for the JSONL history, so
   `REQUIRED_REPORT_FIELDS` can actually be satisfied from a run object.
4. **Fix R4.** Identity fingerprint in the lock, compared at load.
5. **Pin the dependency stack** and add `numpy` to `pyproject.toml`, which it has
   needed since S6.
6. **Wire the protocol in.** Whatever runs a calibration should obtain its items
   through `select(protocol, DatasetRole.CALIBRATION, …)` and nowhere else,
   otherwise the validator guards a document rather than a system.

Deferred until there is data or a GPU: R6, R8, R10. R8 in particular cannot be
closed by more code — the band width wants a risk–coverage sweep on real scores,
and until then 0.20 remains a placeholder that everyone should keep calling a
placeholder.

---

# Batch 4 resolutions

Three findings closed. Each entry says what was done, what it guarantees, and —
more usefully — what it does not.

## R1 — Reproducible calibration fingerprints

**A near-conflict, and why the fix went somewhere else.** The Batch 3 write-up
pointed at `contracts.version._canonical` and its use of `repr(float)`. Changing
it was rejected on two grounds. It is pinned by
`test_fingerprint_distinguishes_neighbouring_floats` with a documented rationale
— a fingerprint is a change *detector*, and one that rounds has a blind spot.
And it is shared: `_canonical` also hashes exact values — branch weights, media
ids, gate caps — where rounding is pure loss, so the change would have moved
every fingerprint in the platform to fix one of them.

The rounding therefore happens at the call site, on fitted quantities only, and
the real fix is upstream of the fingerprint entirely.

**Layer 1 — the cause.** `negative_log_likelihood` now sums with `math.fsum`
instead of `np.mean`. Floating-point addition is not associative and numpy's
pairwise summation visits terms in array order, so the same calibration set in a
different order produced a slightly different NLL; `fit_temperature` compares two
NLLs at each of 200 ternary-search steps, and near convergence those are equal to
within that noise, so the comparison flips and the bracket lands elsewhere.
`math.fsum` is exactly rounded — the correctly rounded sum regardless of order —
so the fitted temperature is now **bit-identical** under permutation.

This is not a trade of accuracy for determinism; the exactly rounded sum is also
the more accurate one. It costs about 20× a vectorised mean (measured: 400
evaluations over 100 000 samples, 0.02 s → 2.9 s), paid once per offline fit.
`negative_log_likelihood` is not on the serving path — the pipeline calls
`temperature_scale` and never this — so no request pays for it.

**Layer 2 — defence in depth.** `contracts.version.quantize` rounds fitted floats
to `FINGERPRINT_DECIMALS = 6` before they enter `calibration_fingerprint`:
temperature, threshold, both band edges, target FPR. A 1e-6 difference in any of
them moves `p_fake` by well under 1e-6, four orders of magnitude below the
resolution of any decision this system makes.

**What this does not guarantee.** Quantisation alone is *not* a stability
mechanism, and treating it as one is the trap: two values astride a rounding
boundary — `0.6199999996` and `0.6200000004` — still round apart. It narrows the
window in which noise matters; it does not close it. The actual guarantee comes
from layer 1, and it is stated as bit-equality in
`test_the_fitted_temperature_is_bit_identical_under_reordering`. Layer 2 covers
what layer 1 cannot: a different numpy build, float32 inputs, a threshold
re-derived from a resampled quantile.

**No cache churn.** The shipped defaults — 1.0, 0.62, 0.52, 0.72, 0.02 — all
round to themselves, so `calibration_fingerprint` is still `dc6fcc65` and
`STUB_ENGINE_VERSION` needed no update. Pinned by
`test_the_shipped_defaults_are_unaffected_by_quantisation`, because had any
default landed on something like `0.5199999999999999`, quantisation would have
silently invalidated every cached result in production.

## R2 — Regression gate slice completeness

`regression_gate` now compares the **union** of both runs' slice names, and
`run_benchmark` records the slices it could not measure rather than dropping
them: `empty_slices` (matched no evaluated items) and `single_class_slices`
(matched items, all one label, so no AUC is defined).

A slice with a baseline that is missing from the challenger is a **failure**, not
a note, and the failure names the cause when the run recorded one — "it is now
single-class" is a different conversation from "it was renamed". The gate cannot
know whether a slice was retired deliberately, and the safe reading of "the check
I relied on last time is gone" is that the release waits for a human.

A slice unmeasured in *both* runs is a note, not a failure: nothing regressed,
but it is a hole in the gate and a report that stays silent about it reads as
full coverage.

**What this does not guarantee.** The gate still only sees slices the caller
passed in `slice_by`. A dimension nobody sliced on is not gated, and nothing here
detects that — `slice_counts` is the tool for that question and it is not wired
into the gate.

## R3 — Benchmark reproducibility metadata

`BenchmarkRun` gained `protocol`, `split_lock_hash`, `lock_generation`,
`calibration_fingerprint` and `schema_version`. All default to `None`, which
reads as *not recorded* and never as a value; `lock_generation` in particular
must not default to `0`, which would be a lie that validates.

`run_benchmark` accepts a `SplitLock` or a bare hash, and an `EvaluationProtocol`
or a bare name — duck-typed, so a caller holding only a hash need not rebuild a
lock object.

**Reports assemble from run artifacts alone.** `BenchmarkReport` pairs the
primary-benchmark run with the in-domain run, because `REQUIRED_REPORT_FIELDS`
demands `primary_benchmark` and `in_domain` **together** and no single run holds
both. `to_payload()` satisfies every required field; `validate()` additionally
refuses two reports that would pass a field check and still say nothing: a pair
of runs from different models, and a pair from the same corpus.

Threshold-dependent numbers come from the primary run only. Under this
platform's protocol the in-domain split *is* the calibration set, so its
coverage and selective rates are best-case by construction; its AUC is untouched
by that, which is why its contribution to a report is a ranking number and
nothing else.

**Migration.** `RUN_SCHEMA_VERSION` is `1.1.0`; the Batch 2 shape is `1.0.0`.
`BenchmarkRun.from_dict` reads either, filling missing metadata with `None` and
missing slice accounting with empty — which is what a 1.0.0 row actually says:
not that there were no unmeasured slices, but that nobody looked. `read_runs`
reads a history as typed runs; `read_history` keeps its dict return for
inspecting a file that holds several schema versions. A row from a *newer*
schema is refused rather than partially read.

## Still open

R4 through R10 are unchanged. The two worth restating:

- **R7** — `numpy` is still not declared in `pyproject.toml`, and nothing pins
  OpenCV, ffmpeg or torch. `np.quantile` sits directly under `threshold_at_fpr`.
  This is the cheapest remaining fix in the audit.
- **R5** — the protocol validates a document, not a system. Nothing obliges a
  calibration run to obtain its items through
  `select(protocol, DatasetRole.CALIBRATION, …)`. R3 makes a run *record* its
  protocol; it does not make the protocol *bind*.

---

# Batch 5 resolutions

Three findings closed: R7, R4, R5. As before, each entry says what was done and
what it does not guarantee.

## R7 — Dependency declaration

**What was wrong.** `pyproject.toml` declared `pydantic` and nothing else.
Derived from the AST rather than from memory, five runtime dependencies were
undeclared. Every test passed throughout, because the machine they ran on had
all five installed for unrelated reasons — a hand-maintained list fails silently
and only for the next person.

### Dependency matrix

Generated by `python tests/import_audit.py`, asserted by `tests/test_dependencies.py`.

| Distribution | Imported by | Files | Class | Was declared? |
| --- | --- | --- | --- | --- |
| `pydantic` | backend, contracts, jobqueue, storage, workers | 20 | runtime | yes |
| `numpy` | evaluation, inference | 12 | runtime | **no** |
| `fastapi` | backend | 9 | runtime | **no** |
| `opencv-python` (`cv2`) | inference | 2 | runtime | **no** |
| `starlette` | backend | 1 | runtime | **no** |
| `pydantic-settings` | backend | 1 | runtime | **no** |
| `torch` | inference | 1 (lazy) | optional `[torch]` | **no** |
| `onnxruntime` | inference | 1 (lazy) | optional `[onnx]` | **no** |
| `psutil` | tests | 1 | test-only `[dev]` | **no** |
| `pytest` | — | 0 | test-only `[dev]` | yes (deliberate orphan) |

`boto3` and `redis` are **not** imported anywhere — `storage/s3.py` and
`jobqueue/redis.py` are interface-shaped backends with no client — and are
correctly absent. Neither is installed, which confirms it.

**The three classes are not bookkeeping.** `torch` and `onnxruntime` are
imported inside `ModelRegistry._build_torch_backend` and `_build_onnx_backend`,
deliberately: a CPU deployment serving ONNX must not be made to install a
multi-gigabyte CUDA wheel, and the API process must not import torch at all.
Declaring either as a runtime dependency would undo that design decision in a
config file, so `test_the_model_backends_are_optional_not_runtime` asserts they
stay lazy — read from the AST, not from a comment claiming they are guarded.

**Two checks, in both directions.** An import with no declaration fails; a
declaration with no import fails too. Orphan declarations are how a lock file
accumulates a transitive tree that outlives the code that needed it. `pytest` is
the one legitimate orphan — every suite is plain-assert with a `__main__` runner
precisely so the tests run without it, and it is not installed here — so the
orphan check covers `[project].dependencies` only, and a test pins why.

**A packaging bug found while doing this.** `[tool.setuptools] packages` said
`["contracts"]`, correct at S0 and quietly wrong from S1 onward: an install
would have shipped the contracts and none of the code that uses them. Tests run
from the source tree, so nothing ever exercised it. Now lists all seven.

**What this does not guarantee.** Declaring a dependency is not pinning one.
`numpy` is capped below 2.0 — the 2.x line changed scalar promotion and repr,
and `_canonical` fingerprints float `repr` on purpose — but nothing here
produces a lock file, and OpenCV's and ffmpeg's *build* still varies per machine.
That is R6 and remains open.

## R4 — Split-lock identity sealing

### What determines split membership, traced

`build_components` unions `media_id` with `source_key()`
(`dataset_id::src::source_id`) and `subject_key(scope)`
(`dataset_id::subj::subject_id`). Nothing else enters the union-find. So
membership is decided by `media_id`, `dataset_id`, `source_id`, `subject_id` and
`subject_scope` — and a 1.0.0 lock sealed only the first of those, via
`assignments`.

### What a 1.0.0 lock could and could not see — corrected

The Batch 3 finding said `verify_lock` "catches it". Measured, that is **too
generous**, and the correction is the interesting part:

| Tamper | Components | Pre-R4 verdict | Now |
| --- | --- | --- | --- |
| subject merge, **across** splits | 6 → 5 | caught (leakage) | caught, both verdicts |
| subject merge, **same** split | 6 → 5 | **clean** | identity drift, named |
| subject **split** (`p5` → `p99`) | 6 → 6 | **clean** | identity drift, named |
| source relabel | 6 → 6 | **clean** | identity drift, named |
| compression re-bucket | 6 → 6 | **clean** | content drift, named |

So the pre-R4 detection was **incidental**: it fired when the merged identities
happened to straddle folds and was silent when they did not. Whether a
relabelled corpus was caught depended on where the greedy splitter had put it.

### The fix

Two per-item digest maps in the lock body — `identity_digests` over
`IDENTITY_FIELDS` and `content_digests` over `CONTENT_FIELDS` — and the
separation is load-bearing. An identity failure means the split may leak; a
content failure means the split is sound and the numbers mean something else.
`LockAudit` reports `assignments_intact` and `identity_intact` as independent
verdicts because they send the reader to different problems.

Per-item rather than one aggregate: an aggregate answers only "something
changed", and a corpus-wide hash moves when items are *added*, so it could never
distinguish growth from tampering.

**Re-sealing must not launder a change.** `extend_lock` recomputes digests over
whatever corpus it is handed, so without a check it would turn a relabelled
index into a valid lock attesting to the new identity — and the laundered lock
would pass every later audit. It now verifies the existing seal *before*
renewing it and refuses on drift.

**"Unsealed" is a third state, not a clean one.** `identity_intact` is **false**
for a lock carrying no digests. "No drift detected" and "no drift detectable"
are different, and reporting the second as the first is exactly the claim this
finding was about.

### Migration notes

- `LOCK_VERSION` 1.0.0 → **1.1.0**; `sealing_version` is `unsealed` or
  `identity-v1`, derived from whether digests are present and cross-checked
  against the value in the file, so a lock cannot claim a seal it does not carry.
- **Existing 1.0.0 files load unchanged.** `MIGRATIONS["1.0.0"]` adds empty
  digest maps and `sealing_version: "unsealed"`. It does **not** compute
  digests: a migration has only the file, not the corpus, so any digests it
  invented would attest to an index it never saw — retroactively, and for a
  corpus that may already have been relabelled.
- **The way out of unsealed is `extend_lock`**, which seals against the corpus
  as it stands then. Until that happens `verify_lock` says plainly that it
  cannot answer the identity question.
- **Load order changed** and it matters for diagnosis: version → structure →
  integrity. Integrity is checked against the body **as written**, before
  migration — the seal covers what is on disk, and hashing a migrated body would
  compare the recorded digest against a document this build invented, failing
  every 1.0.0 file as tampered. Structure is checked before integrity because
  "assignments contains 'holdout'" is more actionable than "this file was
  edited", and because a *correctly sealed* malformed lock (a buggy writer) is
  invisible to an integrity check.
- **A migrated lock's `content_hash` differs from the file's**, by design: the
  body genuinely changed. Re-save it. `parent_hash` chains history, not files,
  and nothing verifies the chain.
- File size roughly doubles once digests are present. For FF++ plus Celeb-DF
  (~12 500 items) that is a few hundred KB.

**What this does not guarantee.** The seal is only as good as the audit that
runs it. `verify_lock` still has to be *called*; nothing forces it, and R4's
original complaint on that point stands. It also cannot detect a change made
before the first seal.

## R5 — Protocol enforcement

### Entry points, traced

Of every public function in `evaluation/`, exactly one takes items *and*
produces a reportable artifact: `run_benchmark`. Every calibration and threshold
entry point — `fit_and_report`, `evaluate_temperature`,
`holdout_temperature_check`, `at_target_fpr`, `select_band_width` — takes bare
`logits`/`scores`/`labels` arrays. That bounds what enforcement can reach and is
stated below rather than papered over.

### The defect

`run_benchmark(predictor, items, protocol=...)` took the two as unrelated
arguments. A run could score the **training set** and record
`protocol="ffpp-train/celebdf-primary/v1"` beside it; the report, the history
and the gate would all treat that as a cross-dataset benchmark.
`test_the_legacy_path_would_label_the_training_set_a_benchmark` reproduces the
old signature and asserts the false claim succeeds.

### The fix, and its honest limit

`select()` now returns a `RoleSelection` carrying the protocol, the role, the
exact media ids, the lock hash, and a `witness` digest binding the four.
`run_benchmark` records a protocol **only** from one of these; passing
`protocol=` beside a plain list is refused, as is passing both.

It is **not** unforgeable, and claiming otherwise would be the same species of
false assurance the finding is about. A caller inside the process can compute a
witness and construct what they like. What it gives:

- **Accidental false claims are impossible.** `RoleSelection.__post_init__`
  rejects a selection whose witness does not match its own fields, so a false
  claim has to be deliberate rather than a habit of passing `protocol=`.
- **Deliberate ones are detectable from artifacts.** `verify_run_selection`
  re-derives the selection from the protocol, the corpus index and the lock and
  compares witnesses. It needs nothing that was in the original process, which
  is the situation a months-later audit is actually in. A test builds a
  self-consistent forgery — training items attested as the benchmark role — and
  asserts it is accepted in-process and caught by verification.
- **An invalid protocol selects nothing.** Validation runs at the point of use,
  not only where a protocol is written down: a protocol that benchmarks its own
  training corpus is not a document problem but a selection that must never
  happen.
- **`select()` binds the lock into the run**, so a compliant run cannot forget
  to record which frozen split it used.

### Backward compatibility

Historical rows load unchanged: `selection_witness` is `None` on a pre-Batch-5
run, and `verify_run_selection` reports **"no selection witness"** rather than
"verified" — no evidence is not the same as evidence.

**An API break, deliberately.** Ten Batch 4 tests used the bypass path and now
fail by design; they were migrated to `select()`. Any caller passing
`protocol=` with a list must do the same.

**What this does not guarantee.** Calibration is still unenforced. Temperature
and threshold fitting take arrays, so nothing stops a temperature being fitted
on the benchmark corpus and the resulting fingerprint being recorded on a
compliant benchmark run. Binding that would mean threading selections through
the array-shaped calibration API, which is a larger change than R5 asked for.
**This is the largest remaining hole in protocol enforcement and should be the
first candidate if a Batch 6 revisits R5.**

## Still open

R6 (decode nondeterminism — ffmpeg/OpenCV build; untestable without data),
R8 (`band_width = 0.20` is still a placeholder and still the largest unjustified
number in the system), R9 (quarantined items make a lock knowingly leaky, by
design and reported every run), R10 (GPU/training nondeterminism, no models yet).

---

# Batch 6 resolutions — calibration protocol enforcement

Closes the calibration half of R5, which Batch 5 left open and named as the
largest remaining hole. Records one new finding, **R11**.

## Threat model

**The adversary is not a malicious engineer.** It is an ordinary one, six months
from now, running a script somebody else wrote, against a corpus that has grown
twice since. Every path below is reachable by accident, and none of them leaves
a mark in the numbers.

What an attacker — accidental or otherwise — could do before this batch:

| Capability | Before | After |
| --- | --- | --- |
| Fit a temperature on the benchmark corpus | yes, silently | **refused** — `fit_calibration` rejects any role but `calibration` |
| Choose a threshold on the corpus being reported | yes, silently | **refused** at fit; caught again at report |
| Claim `fitted_on="ffpp-test"` for a Celeb-DF fit | yes — free text, checked by nothing | still writable, but `witness is None` and no report will validate |
| Pass benchmark logits with a calibration selection | yes | refused when lengths differ; **still possible** at equal length (below) |
| Edit a threshold in a stored calibration | yes | detected — digest mismatch |
| Relabel a stored calibration's role to a benchmark | yes | refused at construction, even if re-sealed |
| Report a number without saying what it was calibrated on | yes | refused — `calibration_witness` is a required report field |

**What is assumed trusted.** The corpus index (R4 seals it against later change,
but not against being wrong when first sealed), the split lock file on disk, and
the process doing the fit. This is provenance and reproducibility machinery, not
a security boundary: nothing here defends against someone with commit access who
is trying.

## The defect, measured

`fit_temperature` and `at_target_fpr` take bare arrays. Where those arrays came
from was not an argument, was not recorded, and could not be recovered. The only
account of provenance was `CalibrationSpec.fitted_on` — free text, written by
the same caller that did the fit.

`legacy_calibrate` in `tests/test_calibration_witness.py` reproduces that path.
Fitting the operating point on the benchmark corpus and reporting it there:

```
laundered  T=1.3465  threshold 0.8058   on benchmark: FPR 0.0500  TPR 0.4037
honest     T=1.2611  threshold 0.8429   on benchmark: FPR 0.0338  TPR 0.3525
```

The in-sample threshold spends the 5% false-positive budget **exactly** — it was
chosen against those very negatives — and buys **5.1 points of detection** over
an honestly transferred one at a nominally identical budget. That is the point
worth holding onto: laundering does not merely lose traceability, it *flatters
the result*, and it flatters it in the direction everyone is hoping for.

## The fix

A `CalibrationWitness` in the same spirit as `RoleSelection`'s: protocol, role,
corpus fingerprint, sample count, every fitted output, the selection witness of
the corpus consumed, the split lock hash, and a digest binding them.
`fit_calibration(selection, logits, labels)` is the only thing that produces
one, and it takes a `RoleSelection` rather than items.

**The role check is the structural part.** `fit_calibration` refuses any role but
`CALIBRATION` — not a warning, not a flag on the output, a raise. That is what
makes benchmark-data-as-calibration-data unrepresentable rather than merely
detectable, and it is the one guarantee here that does not depend on anybody
running a verifier.

**The timestamp is recorded but not digested.** `created_at` is real provenance
and worth having; putting it in the digest would make "identical inputs produce
identical witnesses" unwritable as a test, and that property is the cheapest
reproducibility check there is. Same reasoning as the split lock carrying no
timestamp at all — with the difference that here the timestamp is genuinely
wanted, so it sits beside the digest instead of being omitted.

**Fitted floats are quantised first**, exactly as `calibration_fingerprint` does.
Without it the witness would inherit the R1 defect one level up: two runs of the
same fit would produce two witnesses and a verifier would report tampering where
there was float noise. `test_reordering_the_calibration_set_does_not_move_the_witness`
pins it.

**The witness is deliberately NOT in `calibration_fingerprint`.** That
fingerprint is half the result cache key and answers one question: would this
deployment produce the same number? Provenance changes no probability, so two
deployments with identical T, threshold and band are cache-equivalent whether or
not one can prove where it came from. Putting the witness in would fragment the
cache on a distinction the cache does not care about — and would move
`STUB_ENGINE_VERSION` for a field that is `None` on every shipped configuration.
`dc6fcc65` is unchanged and pinned by a test.

**Reports consume metadata, they do not trust claims.** `run_benchmark` derives
`calibration_fingerprint` *from* the witness and refuses a witness beside a
contradicting fingerprint. `BenchmarkReport.validate()` verifies the witness,
checks it against the report's protocol, checks its lock against the benchmark's
lock, and runs `assert_not_benchmark_corpus` — the report-level backstop for the
case the role check cannot see, where a protocol's calibration and benchmark
roles name the same corpus.

`REQUIRED_REPORT_FIELDS` gained `selection_witness` and `calibration_witness`, so
a report that cannot show it followed the protocol is not a complete report.

## Guarantees gained

- A calibration fitted on a benchmark or training selection **cannot be
  produced**. Structural, no verifier required.
- A calibration's corpus is **recoverable from the artifact** — role, fingerprint
  and sample count — with no access to the process that produced it.
- Editing any digested field of a stored witness is **detected at load**.
- Relabelling a witness's role to a benchmark is **refused even if re-sealed**,
  because the role invariant is checked in the constructor, not only in the digest.
- Compliance **cannot be asserted by omission**: a report with no calibration
  witness fails validation rather than passing quietly.
- Identical inputs produce **byte-identical witnesses**, across reordering and
  across time.
- Existing history rows and specs without witnesses **still load**, and verify as
  "no witness" rather than "verified".

## Guarantees NOT gained

- **Equal-length logits from the wrong corpus.** `fit_calibration` checks that
  the arrays match the selection's *length*, which catches the realistic mix-up
  because the calibration and benchmark splits differ in size. It cannot check
  that the numbers were produced by running the model on those items — nothing
  in this package runs the model. Closing it means the scoring step producing a
  witness of its own, which is a larger change than this batch.
- **Not unforgeable.** As with `RoleSelection`: a caller can call
  `compute_witness` and construct anything. What is bought is that accidental
  false claims are impossible and deliberate ones are detectable from artifacts.
- **The corpus index is trusted at first seal.** R4 seals it against later
  change; nothing establishes it was right to begin with.
- **`fitted_on` is still free text.** It is now decorative rather than load-
  bearing, and left in place because removing it would change a dataclass the
  serving path constructs. Read `witness`, not `fitted_on`.
- **Nothing forces verification to run.** `BenchmarkReport.validate()` has to be
  called. Same standing complaint as R4's.

## New finding: R11 — fitted threshold and configured band can be jointly unrepresentable

Discovered while building this batch, and recorded rather than fixed, per the
brief.

The threshold is **fitted** (`at_target_fpr` on the calibration split) and the
band width is **configured** (the shipped placeholder, 0.20). Nothing reconciles
them, and their product can be impossible: `ThresholdInfo.centred` correctly
refuses a band that runs outside [0, 1].

Observed immediately on a realistic fixture — a 2% budget over ~22 negatives put
the threshold at 0.9734, where no 0.20-wide symmetric band fits:

```
ValueError: a band of width 0.2 around 0.9734218951277995 runs outside [0, 1]
```

This is not a bug in `ThresholdInfo` — the refusal is the Batch 1 invariant
working exactly as designed, and failing at calibration time is far better than
failing at the first request. Three observations:

1. A tight FPR budget on a small calibration split drives the threshold toward
   1.0 by construction: at 2% with 22 negatives the threshold *is* the largest
   negative score. This is the same regime `MIN_NEGATIVES_FOR_FPR = 500` already
   warns about, now producing a hard failure instead of a warning.
2. The two knobs are set in different places by different people at different
   times, and nothing surfaces their interaction until a fit is attempted.
3. `evaluation.risk.select_band_width` exists to derive the width from a
   risk–coverage sweep and has still never been run against real scores — which
   is **R8**. R11 is the mechanical consequence of R8 going unaddressed.

Mitigated only by an error message that names both knobs and points at
`select_band_width`. The proper fix belongs with R8, on real data.

## Still open after Batch 6

R6 (decode nondeterminism), R8 (`band_width = 0.20` placeholder — and now also
the root of R11), R9 (quarantine, by design), R10 (GPU/training), R11 (above).

**R8 and R11 were closed in Batch 7; see the section below.**

---

# Batch 7 resolutions — R8 and R11

Full derivation, proofs and the six-approach comparison:
[`docs/THRESHOLD_BAND_DESIGN.md`](docs/THRESHOLD_BAND_DESIGN.md). Measurements
re-runnable with `python tests/band_study.py`.

## R11 — closed structurally, not handled

The band is now symmetric in **log-odds** rather than in probability:

    band_lo = expit(logit(value) - margin)
    band_hi = expit(logit(value) + margin)

For every threshold strictly inside (0, 1) and every finite positive margin,
`0 < band_lo < value < band_hi < 1` — by monotonicity of `logit`/`expit` and the
range of `expit`. There is no feasibility constraint linking the two, so a
jointly unrepresentable pair no longer exists. The audit's own case,
`(0.9734218951277995, ln 1.5)`, now yields `[0.960656, 0.982123)`.

**Why probability symmetry had to go.** A band family that brackets its
threshold, is symmetric in probability, and has a width whose meaning does not
depend on the threshold cannot be total: symmetry forces `band_lo = t - w/2`, so
any `t < w/2` puts the lower edge below zero. Three properties, and one has to
give. Giving up bracketing is fatal; giving up width-independence produces a knob
with no fixed meaning (approaches C and E' in the report); giving up probability
symmetry relocates the symmetry to the scale where it can hold everywhere. The
Batch 1 invariant is not weakened — the validator is still an *equality* on a
single derived parameter, and it still refuses every legacy shape, plus one more:
the probability-symmetric band that shipped before.

**Why clipping would have been worse than the bug.** Measured, at the R11
threshold: a clipped 0.20 band is `[0.8944, 1.0000)`, and since the manipulated
region is `p >= band_hi`, **no input is ever called manipulated again** — TPR
0.0000, decided-fake 0, every published field still a well-formed number.
Clipping is constructible on 100% of (0, 1) and *useful* on 80% — exactly the
region where nothing was wrong. It converts a loud failure into a silent one.

**What the failure became.** The threshold at 0.9734 was never really a band
problem; it was an extreme quantile of 22 negatives. `derive_operating_point` now
produces a band there *and* warns that the threshold will move on resampling,
naming `bootstrap_threshold_ci`. The diagnosis moved to the thing that was
actually wrong.

**What still refuses.** Three things, none of them a joint impossibility: a
threshold at exactly 0 or 1 (a saturated fit, named as one); a margin wide enough
to push an edge within `BAND_EDGE_FLOOR = 1e-6` of an endpoint (the region beyond
would be unreachable — approach B's failure, refused rather than shipped); and a
margin below float64 resolution at that threshold (an arithmetic limit, reported
as one). Measured feasibility for the shipped parameter: the old width was
infeasible on **20%** of (0, 1); the new margin on about **1.5e-6** either side.

## R8 — closed by moving the derivation into the calibration pass

Two parts, and the second is the one that matters.

**The parameter can now have a single correct value.** A fixed probability width
is not a fixed amount of evidence: 0.20 is worth 0.41 nats at a threshold of 0.5,
2.50 at 0.89, and diverges at 0.90. *No* value of `band_width` was right at two
different thresholds, which is why R8 could not be closed on its own terms. A
log-odds margin is constant evidence — "abstain unless the odds are at least
`e^margin` past the threshold odds".

**The knob is gone from the caller's hands.** `derive_operating_point` takes the
calibrated scores and the product constraints and returns threshold *and* margin
from one pass over one dataset; `fit_calibration(margin=DERIVE)` is the default
and routes through it. There is no longer any argument anywhere that sets a band
without the threshold in hand — which is the structural closure of R11 as well.

**It is a function of the data.** Measured across score spreads sigma = 1.0 to
3.5 at a fixed selective-FPR ceiling, the derived margin runs 0.99, 1.60, 2.36,
3.36 — monotonically, a factor of 3.4 from a procedure containing no constant.
Note what drives it: the heaviness of the negative tail above the threshold, i.e.
the *spread*, not the separation between class means. A fixture varying
separation alone barely moves the margin — found while writing the tests, and the
fixture was corrected rather than the assertion weakened.

**`max_selective_fpr` has no default, deliberately.** It is the product decision
the band implements, and a default would be a new unjustified constant standing
where the old one stood. It must also be *tighter* than the bare threshold
achieves or it buys nothing: widening the band only lowers FPR, so a looser
ceiling is met by the narrowest candidate and selects nothing. That case is
reported as `binding=False` with a warning rather than returned as a confident
1e-4 — a distinction the study caught on the R11 fixture itself.

## Contract impact

`CONTRACT_VERSION` **1.0.0, unchanged**. `EvidenceBundle` unchanged.
`ThresholdInfo`'s four fields unchanged in name, type, bound and required-ness —
the published JSON Schema is byte-identical and the OpenAPI drift guard passes
without a snapshot change. `band_width` survives as a derived, published property.

What changed is one internal invariant (the centring scale) and the constructor
name: `centred(value, band_width, ...)` becomes `at_margin(value, margin, ...)`.
The rename is the point — the new signature rejects every old call site loudly
rather than reinterpreting its argument silently.

**The shipped operating point moved, and moving it was correct.** [0.5200,
0.7200) becomes [0.5210, 0.7099), so `calibration_fingerprint` went `dc6fcc65` to
`739c3fec` and `STUB_ENGINE_VERSION` with it. That is Batch 1's invariant 6
working: the operating point is in the fingerprint precisely so that moving the
decision boundary invalidates the result cache. A geometry change that left the
fingerprint alone would have served old verdicts under the new rule.

Side effect worth recording: the band edges are computed now, not typed, so they
no longer round to themselves under `quantize`. The Batch 4 pin was narrowed to
the configured defaults and a new test pins what the cache actually needs — that
the edges do not sit astride a rounding boundary, checked by perturbing the
margin by 1e-12 and requiring the fingerprint to survive.

`CALIBRATION_WITNESS_VERSION` 1.0.0 becomes **1.1.0**; `band_width` becomes
`margin`. A 1.0.0 payload is **refused with a message naming the change**, not
migrated: converting the old number requires deciding which edge of the old band
to preserve, and picking one would invent an operating point that was never
calibrated.

## What Batch 7 does *not* guarantee

- **The margin is still not derived from real scores.** The mechanism is closed
  and exercised; every number above is synthetic. `ln(1.5)` remains a labelled
  placeholder until there are calibrated scores from a real model. Gated on
  datasets and a GPU.
- **The determinism floor has never been measured.** It defaults to 0.0, which
  means "no numerical guarantee is claimed" — honest, not safe. The missing
  measurement is the run-to-run spread of `p_fake` per device class.
- **A non-binding derivation is reported, not prevented, by default.** Release
  paths should pass `require_binding=True`.
- **`min_coverage = 0.85` is now the largest unjustified constant left** in the
  operating-point path. It is a rejection threshold rather than a fitted value,
  so it is a smaller version of R8 rather than a repeat — but it is the same
  species.
- **Nothing forces verification to run.** `binding`, the warnings list and
  `verify_calibration_witness` all have to be read by somebody. Unchanged
  standing complaint from R4 and R5.

## Still open after Batch 7

R6 (decode nondeterminism), R9 (quarantine, by design), R10 (GPU/training).

---

# Batch 8 resolutions — R6 measured, R12 and R13 opened

Full threat model, thirteen experiments and the guarantee table:
[`docs/DECODE_DETERMINISM.md`](docs/DECODE_DETERMINISM.md). Re-runnable with
`python tests/decode_study.py`; guarantees pinned by
`tests/test_decode_determinism.py`.

**No contract change.** `CONTRACT_VERSION` 1.0.0, `EvidenceBundle`,
`ThresholdInfo`, OpenAPI and the model contracts are untouched, and the OpenAPI
drift guard passes without a snapshot change.

## R6 — measured, and two of its three mechanisms were wrong

R6 said: "OpenCV/ffmpeg builds do not agree on the pixels of a given frame.
Different SIMD paths, different libjpeg, different H.264 decoder behaviour on
damaged streams." It was marked *open, untestable without data*.

It was testable. This OpenCV build registers two video backends — FFMPEG and
MSMF — which are separate decoder implementations, so comparing them is a real
cross-decoder experiment rather than a simulation of one.

**The conclusion is correct and was understated. Two of the three named
mechanisms do not reproduce.**

- **Different SIMD paths — not reproduced.** `resize`, `to_rgb` and `imdecode`
  are byte-identical under `OPENCV_CPU_DISABLE=AVX2,AVX,FP16,SSE4_2,SSE4_1` in a
  subprocess (the dispatch really is disabled at load), and under in-process
  `setUseOptimized(False)` and `setNumThreads(1)`.
- **Different libjpeg — not reproduced.** OpenCV's libjpeg-turbo and Pillow's
  decode the same JPEG bit-identically at q88, q40, progressive, and PNG.
- **Decoder disagreement — reproduced emphatically, and it needs no rebuild.**
  FFMPEG vs MSMF on the same file: **zero identical frames**, every codec tested
  (avc1, mp4v, XVID, MJPG). Photographic content mean |d| = 2.22/255, max 15,
  85% of pixels changed; noise mean |d| = 12.60, max 202, 96% changed. MJPG is
  in that list, so this is not an inter-frame prediction artefact.

**The cause is systematic, not noise.** A least-squares fit from MSMF to FFMPEG
gives a near-identity 3x3 with a per-channel bias near **-1.2** — a levels/range
interpretation difference — and a residual of mean 2.01 that a colour matrix
cannot explain, so chroma upsampling differs too. Nothing this codebase does to
its own arithmetic will make two decoders agree. The only available controls are
*pin one* and *record which*.

Both negative results are pinned by tests rather than dropped, so that if a
future build does diverge, a test says so instead of the finding quietly
becoming true in the dark.

**Mitigation shipped.** New `inference/decode_env.py`: the backend is resolved
by name (`PREFERRED_VIDEO_BACKENDS`, overridable by `DFP_VIDEO_BACKEND`) instead
of being inherited from OpenCV's priority table, and `open_video` is now the
single `VideoCapture` construction site in the package — `probe_video` and
`extract_frames` both route through it, so probe and decode cannot resolve
differently. `decode_environment()` captures OpenCV version, resolved backend,
how it was resolved, avcodec version, OS, machine and CPU baseline/dispatch, and
digests them.

**Reclassified**: from *open, untestable* to **measured; partially mitigated**.
The residual is R12.

## R12 — the decoder is outside every identity string (Medium, measured)

The finding R6 pointed at without naming. Same bytes, same code, same model,
two backends:

```
FFMPEG  p_fake=0.9039545661  model_version=stub-0-w7f45fada+c.739c3fec.g.ae232cba
MSMF    p_fake=0.8914745894  model_version=stub-0-w7f45fada+c.739c3fec.g.ae232cba
```

Identical version string, different pixels, different score. Before Batch 8 the
backend was chosen at runtime by OpenCV's priority table, which
`OPENCV_VIDEOIO_PRIORITY_<BACKEND>` overrides — so **an environment variable
could change a published verdict with every version field byte-identical.**

**Impact: reproducibility, not correctness.** Neither decoder is wrong; they
disagree. But the result cache is keyed on `effective_model_version`, so a
cached verdict decoded by one backend is served to a request that would have
used the other, and a benchmark comparison across two workers with different
decoders reports a model regression that is a decoder change.

**Measured size.** Per-frame |dp| ~= 0.005 for a smooth Lipschitz scorer
(correlation 0.997); against the shipped band [0.5210, 0.7099) over Beta(2,2)
scores that is **1.31% of decisions changed**, 1.63% at the corpus-level shift.
The stub — which hashes frame bytes, so any pixel change decorrelates it
(measured per-frame correlation **-0.18**) — gives |dp| ~= 0.026 and 7.2%. The
surrogate number is the honest estimate; the stub number is the upper bound and
is reported as such.

**Second-order, and worth naming separately:** a decoder change moves the
*fitted threshold* (0.17546 -> 0.16977, measured). The threshold **is** in
`calibration_fingerprint`. So the version moves if and only if somebody refits —
for the wrong reason (a refit) and never for the right one (the decoder).

**Severity Medium.** Below R1, which broke the cache for every result; above R3,
because it is silent, survives every existing check, and is triggered by an
environment variable rather than a deliberate act.

**Remediation, in order of cost.** (1) Shipped: name the backend, capture the
environment. (2) *Proposed, not implemented:* record `decode_fingerprint()` on
`BenchmarkRun` and have `regression_gate` refuse a comparison across differing
digests, exactly as it already refuses one across differing dataset
fingerprints. That needs `RUN_SCHEMA_VERSION` 1.1.0 -> 1.2.0 with a default so
old rows still load — precedented by Batch 4 — and is left for a batch whose
brief permits a schema change. (3) Folding the digest into
`effective_model_version` is **refused**, for a structural reason: Batch 0
invariant 3 keeps `backend` free of `inference`, the API process computes that
string, and `cv2.getBuildInformation()` is not available there. A fingerprint
only the worker can compute cannot be part of a key the API must compute. The
consequence — the result cache can serve a verdict produced by a different
decoder — is R12's residual risk and is not closed by this batch.

## R13 — face-crop instability amplifies decoder drift (Low-Medium, measured)

A perturbation *smaller than* the measured decoder delta changes the Haar
detection box in 29 of 30 trials at +/-2 LSB, with shifts up to 8px, and loses
the face outright in most trials — which routes a scored input to
`NO_FACE_DETECTED` instead. A moved crop feeds the model different pixels
entirely, so E3's pixel drift is amplified rather than passed through.

**Affects both reproducibility and correctness.**

**Severity Low-Medium, and the range is the honest part**: the fixture is a
synthetic face near this cascade's detection boundary, so the measured *loss*
rate overstates what a real photograph would do. The direction is not in doubt.

**Remediation: none in this batch, deliberately.** The frozen architecture
replaces Haar with SCRFD, and tuning a detector scheduled for deletion is wasted
work. What should happen instead: re-run the experiment against SCRFD on real
faces when it lands, and make crop stability under +/-4 LSB an acceptance
criterion for that migration rather than a discovery afterwards.

## What Batch 8 eliminated, and what it only made visible

**Eliminated** (the mechanism cannot occur, structurally): run-to-run variation
within one build; frame reordering (one sequential pass, monotone index);
floating-point variation in `normalize`/`astype` (elementwise, no reduction, so
no summation order); keyframe seeking on the shipped path; timestamp semantics
(read after `grab()`); and backend selection by environment.

That last one is worth a sentence. `CAP_PROP_POS_MSEC` read after
`set(POS_FRAMES, i)` and before `read()` is **not a timestamp** — measured, it
reports 0.280 where the presentation time is 280.0, a factor of a thousand. The
shipped path reads it after `grab()` and is correct; Batch 8 pins the hazard as
a double so the ordering stays load-bearing rather than incidental.

**Detectable, not eliminated**: which decoder ran. A fingerprint cannot make two
libavcodec builds agree; it ensures that when they disagree the two results are
distinguishable rather than silently pooled. Same guarantee the split lock makes
about corpora, and worth exactly as much — it works only if something compares
it, and nothing yet does (that is R12's remediation 2).

**Unresolved**: container-metadata divergence (not reproduced here — both
backends agree on every intact container and both refuse every truncated one —
but `sample_frame_indices` is uniform over the declared count, so a one-frame
miscount moves **29 of 48** sampled indices, 60% of the analysed clip, and
nothing validates `frame_count`); cross-OS and cross-build agreement, which one
machine cannot test; and the decoder's absence from the cache key.

## Still open after Batch 8

R9 (quarantine, by design), R10 (GPU/training), R12 (above, partially
mitigated), R13 (above, deferred to the SCRFD migration).

---

# Batch 9 resolutions — R12 partially closed, R13 unchanged

Architecture, threat model, migration analysis and measured impact:
[`docs/RUN_IDENTITY.md`](docs/RUN_IDENTITY.md). Proof and doubles:
`tests/test_run_identity.py`.

**No contract change.** `CONTRACT_VERSION` 1.0.0; `EvidenceBundle`,
`ThresholdInfo`, OpenAPI, serving routes and the model contracts untouched. The
OpenAPI drift guard passes with no snapshot change, and
`test_the_evidence_bundle_schema_is_unchanged` asserts `ModelInfo`'s field set
directly. One configured literal moved, explicitly (below).

## The correction Batch 8 needs

Batch 8 recorded a structural reason the decode fingerprint could not reach the
API's cache key: `backend` must stay free of `inference` (Batch 0 invariant 3),
`cv2.getBuildInformation()` is unavailable in the API process, and therefore a
fingerprint only the worker can compute cannot be part of a key the API must
build.

**Every premise is true and the conclusion does not follow.** The API does not
compute `Settings.model_version` either — it reads a *configured literal* that
`workers.engine_registry.verify_version_alignment` checks against what the
engines actually report, failing startup on a mismatch. That mechanism was built
in Batch 0 for this exact shape of problem, and Batch 8 described it two
paragraphs before concluding it could not be used. Batch 9 does what Batch 8
called impossible, by the machinery Batch 8 had already named.

## Requirement 4, answered: yes, and it is proved

The question was whether a cache entry could return a result produced under a
different decoder. `tests/test_run_identity.py` answers it by execution, against
`InMemoryResultSink` — the production key derivation, not a restatement of it —
using `LegacyModelIdentityKey`, the pre-Batch-9 key function kept verbatim
beside the new one, and the two decode identities Batch 8 actually measured
(`c20a064a` FFMPEG, `a90156c7` MSMF):

```
legacy key:  store[key(sha, UNDER_FFMPEG)]; get(key(sha, UNDER_MSMF)) -> the FFMPEG bundle
runtime key: same sequence                                             -> None
real sink:   find_by_content(sha, UNDER_MSMF) -> None
             find_by_content(sha, MODEL)      -> None   (namespaces do not overlap)
             find_by_content(sha, UNDER_FFMPEG) -> hit  (same environment still hits)
```

## The scheme: two identities, one vocabulary

Five questions were being answered by one string. They are now separated:

    model identity      weights + calibration + gate        "which model is this?"
    runtime identity    model identity + decode segment     "would this deployment
                                                             produce the same number?"

Format keeps them readable and keeps the model identity a **prefix**, so
stripping the environment is one `partition` rather than a reassembly:

    stub-0-w7f45fada+c.739c3fec.g.ae232cba.d.c20a064a
    └──────────── model identity ──────────┘└ decode ┘

`contracts/version.py` gains `DECODE_MARKER`, `compose_runtime_version`,
`parse_runtime_version`, `model_identity_of`, `decode_identity_of` and
`is_environment_pinned` — stdlib only, so the API can use them. Runtime identity
now flows to `ModelInfo.version`, the content cache key, the idempotency key,
`/health` and the alignment check. **Benchmark identity deliberately still takes
the model identity**, so "did the model get worse" is not answered yes by a
worker rebuild.

**Two decode digests, and the split is measurement-driven.**
`DecodeEnvironment.identity` (cache key) covers OpenCV version, video backend,
libavcodec version, OS and machine. `DecodeEnvironment.digest` (forensic) adds
CPU baseline and dispatch. The CPU fields are **excluded from the cache key on
the strength of Batch 8's E6**, which measured `resize`/`to_rgb`/`imdecode` as
byte-identical under `OPENCV_CPU_DISABLE` — so a mixed-hardware fleet keeps one
cache namespace instead of halving its hit rate for no correctness gain. OS and
machine are *included* on the conservative side, because a cache's two error
modes are not symmetric: a false miss costs compute, a false hit returns a wrong
verdict.

**No sentinel for "unpinned".** `compose_runtime_version(decode=None)` returns
the model identity unchanged; a placeholder would make "makes no claim" and
"pinned to nothing" the same string, and the whole scheme rests on telling them
apart. Every pre-Batch-9 version therefore still parses, with `decode=None`, and
needs no migration to be readable.

## The literal moved, and it is meant to

`STUB_ENGINE_VERSION` is now `stub-0-w7f45fada+c.739c3fec.g.ae232cba.d.c20a064a`
— **environment-specific by design**. A worker on another OS or OpenCV build
reports a different digest and `verify_version_alignment` fails at startup
rather than letting the two share a cache namespace. That is the intended
behaviour, it is stated in the literal's own docstring, and
`test_alignment_fails_when_the_configured_decoder_is_stale` pins it.

**Cache migration**: every pre-Batch-9 entry becomes unreachable — a one-time
full miss, not corruption. No stale entry is served under a new name (the
namespaces provably do not overlap), old objects stay readable by job id, and
the cost is one re-analysis per distinct file. Back-filling old entries with the
current decode identity was rejected: it would assert that historical results
were produced by *this* environment, which is the unfounded claim the batch
exists to remove.

## Guarantees gained

1. A cached result **cannot cross decode environments** — proved, not argued.
2. A decoder change is **visible in the deployment identity** (`/health`,
   alignment, logs).
3. A worker whose decode environment differs from the configured one **fails at
   startup**.
4. Model and environment changes are **distinguishable from the version string
   alone**, in both directions, and both are asserted.
5. A stored bundle **names the decode environment that produced it** — forensic
   reconstruction from the artifact, with no schema change.
6. The cache **does not fragment on CPU dispatch**, on the strength of a
   measurement.

## Guarantees NOT gained

1. **Two decoders still disagree.** This makes them distinguishable, not equal.
   Same shape as the split lock's guarantee, and worth the same.
2. **Benchmark comparisons remain environment-blind.** `BenchmarkRun` has no
   decode field, so `regression_gate` cannot refuse a cross-decoder comparison
   the way it refuses a cross-dataset one, and a decoder change can still read
   as a model regression. Needs `RUN_SCHEMA_VERSION` 1.1.0 → 1.2.0, which this
   brief forbids. Asserted by `test_benchmark_runs_still_carry_no_decode_
   identity` so it cannot quietly be forgotten.
3. **Calibration artifacts remain environment-blind.** A temperature fitted
   under one decoder can be applied under another with no complaint. Same
   schema constraint.
4. **A same-backend hardware decode path is not distinguished.** If OpenCV ever
   routes through NVDEC or VideoToolbox without changing the backend name, the
   decode identity does not move. No such path exists on this box to test.
5. **Cross-OS agreement is unmeasured.** `platform_system` is in the key on the
   conservative side and may over-fragment; one machine cannot settle it.
6. **Nothing forces the forensic record to be read.** Standing since R4.

## R12 and R13, explicitly

**R12 — PARTIALLY CLOSED.** Cache identity, deployment identity, forensic
identity: closed. Model identity: closed by design, the decoder is excluded on
purpose and has its own segment. **Benchmark identity and calibration identity:
open**, both blocked on the same schema change.

**R13 — OPEN, unchanged.** Untouched deliberately: it is a property of the
placeholder Haar detector that the frozen architecture replaces with SCRFD, and
run identity cannot mitigate it — a moved crop produces a different answer
*within* one environment, so no key can separate it. Remediation stands:
re-measure against SCRFD on real faces, and make crop stability under ±4 LSB an
acceptance criterion for that migration.

## Still open after Batch 9

R9 (quarantine, by design), R10 (GPU/training), R12 (benchmark and calibration
halves), R13 (deferred to SCRFD).

---

# Batch 10 resolutions — R13 measured, R14 and R15 opened and closed

Threat model, twelve stages, the seven required answers and the guarantee tables:
[`docs/INFERENCE_DETERMINISM.md`](docs/INFERENCE_DETERMINISM.md). Re-runnable
with `python tests/inference_determinism_study.py`; guarantees pinned by
`tests/test_inference_determinism.py` (24 tests).

**No contract change.** `CONTRACT_VERSION` 1.0.0; `EvidenceBundle`,
`ThresholdInfo`, OpenAPI, serving routes and the model contracts untouched, and
the drift guard passes without a snapshot change. No fingerprint moves —
`STUB_ENGINE_VERSION` is unchanged, because both fixes change which *answer* is
produced under concurrency and which face wins a *tie*, not the identity of any
configuration.

## The question, and why the previous instrument was unavailable

Batch 8 proved two decode environments disagree; Batch 9 made them unable to
share a cache. R13 asked what was left: **can identical media produce different
verdicts inside one decode environment?**

Every earlier determinism batch answered with the same move — make the two
states distinguishable, then let a fingerprint refuse to pool them. That move
does not exist here. Two answers produced ninety milliseconds apart by one
process share one OpenCV build, one backend, one model version, one `sha256` and
one decode identity. There is nothing to hash, the content cache would serve
whichever landed first, and `test_runtime_identity_cannot_detect_the_cascade_
race` asserts it. **Prevention was the only control**, which is why both new
findings were fixed in the batch that found them rather than recorded for a
later one.

## The methodological point

Stage 10 of the study runs the whole pipeline in six subprocesses across thread
counts and `PYTHONHASHSEED` values and gets one byte-identical bundle. It is
**blind to both findings** — every launch is single-threaded and every launch
feeds the detector one input order. The obvious determinism test passes on a
pipeline with a live data race in it, which is the reason this batch measured
eleven other things.

## R14 — face selection was decided by detector output order (Medium, measured)

`sorted(boxes, key=lambda b: -b.area)` is a partial key, and Python's sort is
stable, so two equal-area boxes resolved to the cascade's emission order.
`faces[0]` chooses the crop the whole verdict is computed from:

```
A first -> crop 6c9a2dfa3527, p_fake 0.4830  ->  authentic
B first -> crop f04cb2ef504e, p_fake 0.5637  ->  insufficient evidence
```

**Order dependence, not nondeterminism** — a thousand runs agree a thousand
times. Nothing specifies the order, so nothing stops it moving, and SCRFD's box
order falls out of NMS over float scores and is an implementation detail by
construction. The defect would have arrived with the migration as an unexplained
diff.

**Closed** by `face_order_key` = `(-area, y, x, width, height)`, a total order on
box geometry alone. Width and height sit after the origin so 100×200 and 200×100
— same area, same origin — cannot tie either.

## R15 — the face cascade was shared across threads (High, measured)

The finding R13 was really asking about, and it was live.

`cv2.CascadeClassifier` holds the feature evaluator and its scaled-image buffers
as instance state. `preprocess` kept one in a module global, and **both pipelines
reach `detect_faces` through `asyncio.to_thread`** — so it was shared across pool
threads in the shipped configuration, not in a constructed worst case. Concurrent
calls overwrite each other's working set and return a well-formed wrong answer;
nothing raises, nothing logs.

**Same file, eight threads, through the real pipeline:**

```
sequential, 40 runs   :  40 x  authentic  p_fake=0.032932877375
concurrent, 160 runs  : 136 x  authentic  p_fake=0.032932877375
                         24 x  insufficient_evidence / no_face_detected
```

15% of concurrent analyses published a different verdict from every sequential
run of the same bytes.

**Two files at once — and this is the severe half.** Four threads per file, 160
calls each, three recorded runs:

```
file a : correct 34-57, lost 10-16, other-box  87-110 (75-107 of them b's box)
file b : correct 51-72, lost 11-19, other-box  74-98  (69-88  of them a's box)
```

**Roughly 60% of detections returned the other job's bounding box.** The pipeline
then crops *this* job's pixels at coordinates found in *another* job's image and
publishes the score as this job's verdict. Worse than an unstable number, because
nothing about it looks wrong: the box is well-formed, in frame, plausibly sized,
and no consumer can tell it was located in the wrong file. A third mode was seen
during development — an out-of-range read inside OpenCV's own vectors,
`cv2.error: invalid vector<T> subscript`, killing the job — and both the study and
the tests count a raise as an outcome rather than dropping the thread.

**Severity High, above every previous determinism finding.** R12 was
reproducibility: two decoders disagreed and both answers were honest. R15
produces an answer that is simply wrong, with nothing recording that it happened.

**Closed** with `threading.local()`. A thread cannot contend with itself, so the
mechanism is gone rather than narrowed. A lock is equally correct (0/320) and was
rejected on cost — measured 27–38 s against 5.7 s at 8-way concurrency, because
it serialises the most expensive step of every image job. The per-thread cascade
costs a one-off 45–73 ms load and 0.41 MB per pool thread, and the pool is
bounded and reused. It is also marginally *faster* than the racy code it replaces
(5.7 s vs 6.2 s): contended shared mutable state was not buying speed either.

## R13 — measured, quantified, still deferred

Batch 8 recorded R13 as crop instability amplifying decoder drift and deferred it
to SCRFD. Still deferred; no longer unquantified.

A ±3 LSB perturbation — **smaller than Batch 8's measured cross-decoder delta of
2.22/255** — moved the detection box by up to 7 px of position and 9 px of size
across 11 noise realisations, and lost the face outright in 2 of them. Under the
smooth surrogate scorer, holding pixels fixed and applying each observed box:

```
crop held fixed, ±3 LSB noise only :  p spread 0.000293
crop free to move, same noise      :  p spread 0.091022     AMPLIFICATION 310x
```

The mechanism is box *size*, not position: translations of ±1 to ±8 px inside a
218 px crop survive `INTER_AREA` almost intact (|dp| 0.00016–0.00083), while a
5 px size change moves |dp| to 0.091 — a factor of 110. Changing box size changes
the resample ratio, which rewrites exactly the high-frequency texture a forensic
detector keys on; sliding the same content two pixels does not.

Against the shipped band over Beta(2,2) scores:

| \|dp\| | source | decisions changed |
| --- | --- | --- |
| 0.0050 | R12 cross-decoder drift | 1.37% |
| **0.0910** | **R13 crop movement** | **25.71%** |

**R13 is ~19× R12's decision impact.** It is not a footnote on R12; it is the
larger of the two and the one still open. What this batch adds is a gate for the
migration rather than a paragraph:

> **SCRFD acceptance criterion.** Under ±4 LSB uniform noise on real photographs,
> across ≥ 200 images: no face lost, and the box short edge moving ≤ 1%.
> Amplification below 10×, from the measured 310×.

## New finding: R16 — one face's verdict is published as the media's (Low, by design)

`analyze_image` scores `faces[0]` and publishes a single `FaceTrack`, so a
two-person photograph reports the larger face's verdict for the whole image.
Correct for the single-subject case the product targets, wrong for group media.
Not a determinism defect — R14's fix made the choice reproducible — but recorded
because that fix makes the policy explicit for the first time. Either publish
per-face verdicts or state the single-subject assumption in the contract; it is
currently neither.

## Measured and rejected

- **cv2 thread count.** `setNumThreads(0/1/2/4/8)` and the optimisation toggle
  change no detection box. Pinned, so a future build that does diverge fails
  loudly.
- **Cross-process and hash-seed variation.** Six subprocesses under varying
  `PYTHONHASHSEED` and thread counts produce one byte-identical bundle. The
  pipeline holds several sets and dicts (`pending = set(wanted)`, the per-branch
  dicts) and none is iterated order-dependently.
- **The interpolation switch.** `resize` picks `INTER_AREA` above the model input
  size and `INTER_LINEAR` at or below it, so a 1 px crop-size change switches
  kernel. A genuine discontinuity, and measured at 0.128/255 against the
  1.73/255 the content change itself contributes — ~13× too small to matter.
  Reported as a negative result rather than fixed: picking one kernel would move
  every score to correct nothing.
- **The video path is untouched by both findings.** It runs no face detection at
  all, and frame extraction, ranking and pooling measured deterministic across
  repeats, thread counts and concurrency.

## Guarantees NOT gained

1. **R13 is not fixed** and remains the largest open determinism finding.
2. **Multiprocess workers are unmeasured.** Every number here is one process.
3. **The real detector is not this detector.** SCRFD's runtime carries its own
   thread-safety contract and inherits R15's requirement rather than retiring it.
4. **Determinism is not stability** — one changed pixel still moves everything,
   by design, and a test pins that so this cannot be read as robustness.
5. **No fingerprint detects any of this, and none can.** Only the tests will
   notice if sharing returns.
6. **Nothing forces the study to be re-run.** Standing complaint since R4.

## Still open after Batch 10

R9 (quarantine, by design), R10 (GPU/training), R12 (benchmark and calibration
halves, blocked on a schema change), R13 (quantified at 310×, deferred to SCRFD
with an acceptance criterion), R16 (by design, needs a deliberate decision).

---

# Batch 11 resolutions — R12 fully closed

Architecture review, the ten answers, execution traces and the migration
analysis: [`docs/EVALUATION_IDENTITY.md`](docs/EVALUATION_IDENTITY.md). Proof and
doubles: `tests/test_evaluation_identity.py` (25 tests).

**No contract change.** `CONTRACT_VERSION` 1.0.0; `EvidenceBundle`,
`ThresholdInfo`, OpenAPI, serving routes and every serving fingerprint
untouched. Two *evaluation-side* schemas moved, and both are on their own
version axes by design.

## The review came first, and it changed the shape of the answer

The brief said not to assume the gap should be closed. Measuring it split the
finding in two, and the halves are not equally bad.

**The benchmark half is a real defect, and the obvious worry was the wrong one.**
The fear was that decoder drift would swamp the AUC gate. It does not: with an
i.i.d. per-item perturbation at Batch 8's measured magnitude, |ΔAUC| peaks at
0.000035 — **0.7% of the gate's allowance** — because AUC is a rank statistic
averaged over ~n²/4 pairs and noise averages out. No skill drop between 0.02 and
0.05 was masked through AUC.

The damage is elsewhere, and Batch 8 pointed at it: the decoder difference is a
*systematic levels shift*, not noise. A systematic shift is close to
rank-preserving, so AUC barely moves — while the threshold is a fixed number and
a shifted distribution crosses it:

```
    bias       dAUC   dCoverage       dFPR  %FPR budget   gate
   0.005  +0.000000     +0.0000    +0.0012       24.6%   PASS
   0.010  +0.000000     -0.0010    +0.0035       69.3%   PASS
   0.020  -0.000019     -0.0005    +0.0086      172.6%   FAIL
```

Two harms, in both directions. At Batch 8's measured magnitude a decoder change
**silently spends a quarter of the false-positive regression budget**, so a real
regression consuming the remaining three quarters ships. At four times that
magnitude the gate **fails and blames the model**, in its own words: *"selective
FPR rose 0.0086; detection was bought with false accusations"* — for a model that
never changed.

That is the worst available combination: the metric that survives a decoder
change is the one the gate leans on hardest, and the metric that moves is the one
it charges to the model.

**The calibration half is weaker than it looks.** A decoder change moves the
logits, so it moves the fitted temperature, so it moves the witness digest —
detection already worked, by accident of what the digest covers. The hole is
convergence: two fits landing on the same values after `quantize` to six decimals
produce **byte-identical witnesses for two environments**, measured. And
`verify_calibration_witness` had no parameter through which an auditor could ask
the question at all.

## The migration was the hard part, and a naive bump would have broken artifacts

`CalibrationWitness.compute_witness` has always hashed the schema version. So
bumping `CALIBRATION_WITNESS_VERSION` changes the digest of *unchanged fields*.
Executed against the real class with the constant set to 1.2.0:

```
stored 1.1.0 witness digest : 092895e2
digest under schema 1.2.0   : 54adb0e2
  reload FAILED: ProtocolViolation
    calibration witness '092895e2' does not match its own fields
    (expected '54adb0e2'); this witness was assembled or edited by hand
```

Every stored witness becomes **unreadable** — `__post_init__` raises, so
`from_dict` cannot return one — and the error accuses the operator of hand-editing
a file that is fine.

**The fix is to make the digest rule versioned rather than replaced.**
`compute_witness` takes `schema_version` as a parameter and verification
dispatches on the payload's *own* version, so a 1.1.0 witness is checked under the
1.1.0 rule and continues to validate. `decode_identity` enters the digest only for
schemas in `SCHEMA_VERSIONS_WITH_DECODE` — adding the key unconditionally with a
`None` value would still have moved every 1.1.0 digest. Old artifacts stay
readable **and cryptographically verifiable**, which is a migration rather than an
amnesty.

**Back-fill is impossible and is refused**, for the reason Batch 9 refused it for
cache entries: writing this environment into a historical artifact asserts a fact
nobody observed. Historical rows and witnesses keep `None` forever.

## What shipped

`RUN_SCHEMA_VERSION` **1.1.0 → 1.2.0** (`BenchmarkRun.decode_identity`);
`CALIBRATION_WITNESS_VERSION` **1.1.0 → 1.2.0** (`CalibrationWitness.decode_
identity`, inside the digest).

`regression_gate` now **refuses** a cross-decoder comparison on the same footing
as a cross-dataset one, short-circuiting for the same reason — every later number
inherits the contamination. Three states, not two, which is R4's "unsealed is a
third state" applied here:

| incumbent | challenger | outcome |
| --- | --- | --- |
| recorded `X` | recorded `X` | silent |
| recorded `X` | recorded `Y` | **failure**, comparison refused |
| `None` | anything | **note** — the question cannot be answered |

`verify_calibration_witness(..., decode_identity=...)` lets an auditor ask;
`BenchmarkReport.validate()` cross-checks the calibration's environment against
the benchmark's and refuses a report pairing two environments.

**A decode field, not a runtime identity.** The run keeps the *model* identity in
`effective_model_version` — Batch 9's decision, unchanged — and the runtime
identity stays recoverable by `compose_runtime_version`. The decomposed pair
carries strictly more information than the composed string, and folding the
decoder into the existing field would have made every model-versus-model
comparison a decoder comparison too.

**Passed in, never detected.** `run_benchmark` does not call `decode_identity()`
itself, for the reason it does not compute `effective_model_version`: the harness
process need not be the process that produced the scores, and recording this
one's environment would be a false claim.

**Batch 9's tripwire fired and was honoured.**
`test_benchmark_runs_still_carry_no_decode_identity` asserted the field's absence
and said it "should be replaced by one that gates on it". It has been, inverted
rather than deleted, so the field disappearing again reopens R12 loudly.

## Guarantees NOT gained

1. **Two decoders still disagree.** Distinguishable, not equal — Batch 9's limit,
   unchanged.
2. **Nothing forces a caller to pass `decode_identity`.** It is optional because
   it cannot honestly be inferred; an omitting run gets a note, not a guarantee.
3. **Back-fill is impossible.** Historical artifacts stay `None`.
4. **The magnitudes are synthetic** — a model of Batch 8's measured levels shift,
   not a measurement on a real corpus. Still no dataset, still no trained model.
5. **`decode_identity` excludes the CPU fields** on one build's measurement.
6. **Equal-length logits from the wrong corpus** remains open, unchanged from
   Batch 6.

## Still open after Batch 11

R9 (quarantine, by design), R10 (GPU/training), **R13** (quantified at 310×,
deferred to SCRFD with an acceptance criterion — the largest open finding), R16
(by design, needs a deliberate decision). **R12 is closed.**

---

# Batch 12 resolutions — R17 closed

Architecture review, the seven answers, the sensitivity measurements and the
migration analysis:
[`docs/FACE_SELECTION_IDENTITY.md`](docs/FACE_SELECTION_IDENTITY.md). Proof and
doubles: `tests/test_face_selection_identity.py` (32 tests). Re-runnable study:
`tests/face_selection_study.py`.

**No contract change.** `CONTRACT_VERSION` 1.0.0; `EvidenceBundle`,
`ThresholdInfo`, `ModelInfo`, OpenAPI and every serving route untouched. Unlike
Batch 11, **no evaluation schema moved either** — `RUN_SCHEMA_VERSION` stays
1.2.0 and `CALIBRATION_WITNESS_VERSION` is unchanged, so no stored benchmark row
and no stored witness re-digests.

## Eight of ten selection variables were invisible

The review enumerated every variable that decides which face is scored and what
pixels the model sees. Two were in the gate fingerprint. **Eight were not**, and
five of those eight were hard-coded literals or default arguments — values nobody
thinks of as configuration precisely because they never appear in a config file:

```
scaleFactor 1.1          literal inside detect_faces
minNeighbors 5           literal inside detect_faces
minSize 32               detect_faces(min_px=32) default argument
crop margin 1.3          crop_face(margin=1.3) default argument
ordering rule            face_order_key, a module-level function
detector + its cascade   an OpenCV data file on disk
```

**Two floors, not one, and only one was covered.** `detect_min_px = 32` decides
which boxes *exist*; `MediaCaps.min_face_px = 64` decides which is large enough
to score. A box between them is detected, becomes `faces[0]`, and is then gated
as `FACE_TOO_SMALL`. The pair jointly decides the abstention and the fingerprint
covered half of it, which looked complete.

## The crop margin reaches all three verdicts

Measured on one image, one model, one decoder, one knob varied:

```
crop_margin 1.0  crop=567a77aac7  p=0.889291  manipulated
crop_margin 1.2  crop=aaa2f1db36  p=0.659734  insufficient_evidence
crop_margin 1.3  crop=379c10e869  p=0.013562  authentic              <- shipped
```

Every identity string in the platform agreed across all three — weights,
calibration, gate, decode, model identity, runtime identity, benchmark identity.
The detector parameters fail differently and worse: `scaleFactor` 1.05, 1.2 or
1.3, and `minNeighbors` 8 or 12, **lose the face outright**, replacing a verdict
with `INSUFFICIENT_EVIDENCE / no_face_detected`. That is R15's failure shape — a
lost face, not a moved score — reached by configuration rather than by a race.

Executed against `InMemoryResultSink` with `SelectionBlindVersion`, the
pre-Batch-12 composition kept verbatim as a double: a deployment cropping at 1.2
is served the `authentic` verdict computed at 1.3, when its own crop abstains.

## What shipped

A fourth segment of the **model** identity, `.f.`:

```
stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.d662e22d.d.c20a064a
                                       └───────┘
                                     face selection
```

**The structural half matters more than the hash.** `FaceSelectionPolicy` is a
frozen dataclass with **no defaults on any field**, and `detect_faces` /
`crop_face` take the policy object rather than loose `min_px` / `margin`
keywords. Behaviour and fingerprint are read off one object, so a caller cannot
change one without changing the other — a stronger guarantee than hashing
constants some other code happens to use. Hidden defaults were the defect, so a
policy cannot be constructed without stating every knob.

**The model identity, not the runtime identity.** Face selection is code this
deployment ships; the decoder is environment it inherits. A selection change
genuinely *is* a model change for "did the model get worse", so it belongs in
`effective_version` — where the cache key, the idempotency key, `/health`,
`verify_version_alignment` and `BenchmarkRun.effective_model_version` all inherit
it with no new plumbing.

**Not folded into the gate fingerprint**, though that was the smaller diff. The
gate decides *whether* an input is scored; this decides *which face* and *what
pixels*. Folding them makes "we raised the face floor" and "we replaced the
detector" the same event — the conflation Batch 9 rejected for the decoder — and
the SCRFD migration moves exactly one of the two.

**The ordering rule is named, not hashed.** Bytecode moves on comment edits and
would dump the cache for nothing. `test_the_ordering_rule_name_matches_the_
ordering_rule` re-derives the order the name describes and compares it against
`face_order_key`, so the name is held to the behaviour by a test.

**Behaviour did not move.** `LegacyHardcodedSelection` is the pre-Batch-12
`detect_faces` and `crop_face` with their literals baked in, kept verbatim;
boxes and crop bytes are byte-identical to it. Only the version string changed.

## Guarantees NOT gained

1. **Identity does not make two policies agree.** Distinguishable, not equal —
   the same limit Batch 9 has about decoders, asserted rather than assumed.
2. **R16 is untouched.** One face's verdict is still published as the media's.
   Batch 10 made the choice reproducible; Batch 12 made it auditable; neither
   made it right.
3. **R13 is untouched.** Crop instability under ±4 LSB is a property of the
   placeholder detector, and no fingerprint separates two runs that share one.
4. **R18–R22 are untouched.**
5. **The video engine carries the segment without consuming it** — it detects no
   faces at all, so no policy can move a video verdict today. It carries it
   because `Settings.model_version` is one string keying the cache for both media
   kinds and every engine must report it; an image-only segment would fail
   startup or force a per-kind configured version, which is a serving change.
   The cost is a false miss on the video namespace — compute, not correctness.
6. **The bundle names the policy by fingerprint only.** The eight fields are not
   published; reconstructing them needs the deployment that produced the result.
   The same limitation the gate fingerprint has had since Batch 0.

## Still open after Batch 12

R9 (quarantine, by design), R10 (GPU/training), **R13** (310×, deferred to SCRFD
with an acceptance criterion — still the largest open finding), R16 (by design,
needs a deliberate decision), **R18** (video has no face stage), **R19** and
**R20** (latent, armed by the SCRFD migration), **R21** (prerequisite for any R16
fix), **R22** (documentation, architectural for R16). **R12 and R17 are closed.**

Sequencing note carried forward from Batch 11's review: R19 and R20 are
unreachable today only because Haar returns square boxes, and both become live
the moment SCRFD returns free-aspect NMS boxes. Closing them *before* that
migration costs an afternoon; closing them after means closing them once they
have started producing wrong abstentions.

---

# Batch 13 resolutions — R18 partially closed

The trace, the five-way disagreement, the alternatives and the migration
argument: [`docs/VIDEO_FACE_SEMANTICS.md`](docs/VIDEO_FACE_SEMANTICS.md). Proof
and doubles: `tests/test_video_face_semantics.py` (19 tests). Re-runnable study:
`tests/video_face_semantics_study.py`.

**No contract change.** `CONTRACT_VERSION` 1.0.0; no field added, removed or
retyped; no OpenAPI signature change and no schema snapshot regenerated. One
published value changed and one invariant was added, both argued explicitly.

## Five artifacts describe the system and the implementation is the odd one out

Batch 11's review found that `VideoPipeline` never calls `detect_faces`. Batch 13
asked whether anything else in the platform agrees, and found a fifth voice the
review had not counted — **the shipped fixtures**:

| Artifact | Detects? | Tracks? | `frames_scored` means |
|---|---|---|---|
| Contract prose (pre-B13) | implied | implied | "where a face was usable" |
| **Fixtures** | **yes** | **yes** | 46 scored, `FaceTrack.frames_seen=46` |
| **Test double** (`stub_handlers`) | **yes** | **yes** | = frames sampled |
| **Implementation** | **no** | **no** | frames that decoded |
| Tests (pre-B13) | silent | silent | only monotonicity asserted |

`video_manipulated` publishes a tracked face and `top_frames[].track_id=0` that
resolves to it. The fixtures are internally coherent and describe a face stage.
So R18 is not "a docstring is stale" — the specification artifacts describe a
capability the implementation does not have, and no test noticed.

## Only one of those disagreements was a code defect

Separating them is the whole of what this batch could act on without building a
face stage:

- **fixtures and doubles depicting a target** — intent, not a defect. Annotated,
  not rewritten: rewriting them would erase the design the contracts-first
  approach exists to carry.
- **prose describing an absent stage** — documentation defect. Corrected.
- **`top_frames` naming a track nothing publishes** — **correctness defect.**
  `top_frames()` hard-coded `track_id=0` while `faces` was empty, so *every*
  video bundle carried three references to a face that did not exist.

Executed against `LegacyTopFrameTrackId`, the pre-Batch-13 helper kept verbatim:

```
A: yesterday's bundle (legacy ids, faces=[])   REFUSED
   video.top_frames name track(s) [0] but faces[] publishes []
B: today's bundle (null ids, faces=[])         ACCEPTED
C: the post-SCRFD shape the fixtures depict    ACCEPTED
```

A is the fail-before evidence, executed: that exact bundle validated before this
batch. C shows the invariant forbids *dangling*, not *referencing* — a real face
stage still validates. Six proof tests failed before the fix and pass after.

## What shipped

**Correctness.** `top_frames()` emits `track_id=None` — already the field's
default and its declared type — with a `track_id` parameter so a future face
stage can pass a real id. And `EvidenceBundle._top_frames_name_a_published_track`
makes the dangling state unrepresentable, by hand, from a factory and on
deserialisation. Scoped narrowly to referential integrity: it says nothing about
whether `faces[]` is internally consistent, which is **R21** and untouched.

**Documentation.** `frames_scored`'s description; `TopFrame.track_id`'s null
semantics; `VideoAnalysis` and `FaceTrack` docstrings; the dead `DETECTING`/
`TRACKING` entries in `STAGE_WEIGHTS_VIDEO`; and the first paragraph of
`video_pipeline.py`, which now states the absence of a face stage up front.

**Why no version bump.** No field changed shape; `video.*` is Evolving-tier;
`track_id` was already `int | None` defaulting to `None`; and the corrected prose
describes the value the code always produced. **Why no snapshot regeneration:**
the OpenAPI structural signature ignores prose by design —
`test_signature_ignores_prose_but_not_structure` establishes it — so description
corrections produce no drift.

**Rejected: an explicit `face_stage` flag.** There is a real hole — `faces == []`
means "found none" on an image and "never looked" on a video — but it is
recoverable from `media.kind`, and a field carrying one bit that is constant
today and flips meaning at SCRFD is not the minimum correction. **Also rejected:
a validator forbidding `faces` on video.** It would reject the fixtures and the
doubles and forbid the frozen architecture's own roadmap; a contract must not
make its own future unrepresentable.

## Guarantees NOT gained

1. **A faceless clip still receives a decided verdict.** Measured on pure noise:
   `manipulated` at p_fake **0.979**, `gate_passed=True`. The image path abstains
   with `NO_FACE_DETECTED` on equivalent content. This is R18's capability half,
   it needs a video face stage, and that is SCRFD work —
   `test_a_faceless_clip_still_receives_a_decided_verdict` pins it and fails
   loudly if it changes.
2. **Video verdicts are still whole-frame scores** through a branch A the
   architecture declares to consume an aligned face crop.
3. **`faces == []` remains ambiguous across media kinds.**
4. **Fixtures and doubles still depict a face stage** — annotated, not rewritten.
5. **`JobStage.TRACKING` is still emitted by nothing.**
6. **R16, R19, R21 and R22 untouched**, per brief.

## Discovered in passing, not fixed

**`fixtures_out/*.json` are stale and nothing guards them.** Every committed
bundle carries the pre-Batch-7 probability-symmetric band (`[0.52, 0.72]`) against
the shipped log-odds band `[0.5210, 0.7099]`, so none of them validates against
the current contract. They feed the frontend's TypeScript generator; `test_schema`
exercises the generator against in-memory examples and never checks that the
committed output is current. Pre-existing since Batch 7, unrelated to R18,
recorded so it is not found a third time.

## Still open after Batch 13

R9 (quarantine, by design), R10 (GPU/training), **R13** (310×, deferred to SCRFD
with an acceptance criterion — still the largest open finding), R16 (by design,
needs a deliberate decision), **R18's capability half** (a faceless clip decides;
needs a video face stage), **R19** and **R20** (latent, armed by the SCRFD
migration), **R21** (prerequisite for any R16 fix), **R22** (documentation,
architectural for R16). **R12 and R17 are closed.**

The sequencing note stands and now covers three findings: R19 and R20 are
unreachable today only because Haar returns square boxes, and R18's remaining
half needs the same migration. All three are cheapest to close as part of the
SCRFD work rather than before or after it.

> **Corrected by Batch 14.** Haar does not return square boxes; it returns boxes
> within one pixel of square, measured over 1 025 of them. R19 is latent because
> a witness needs a box **three** pixels from square — a two-pixel margin held by
> an OpenCV rounding path this platform does not pin, not by a structural
> guarantee. And R19 turned out to be cheapest to close *before* the migration
> rather than with it, for the reason in the section below.

---

# Batch 14 resolutions — R19 closed, R23 opened

The trace, the theorem, the three-pixel threshold, the six alternatives and the
migration argument: [`docs/FACE_SIZE_POLICY.md`](docs/FACE_SIZE_POLICY.md).
Proof and doubles: `tests/test_face_size_policy.py` (27 tests). Re-runnable
study: `tests/face_size_policy_study.py`.

**No contract change.** `CONTRACT_VERSION` 1.0.0; no field added, removed or
retyped; no OpenAPI signature change and no schema snapshot regenerated. **No
published number changes under the shipped detector** — proved exhaustively, not
asserted. One identity segment moved, deliberately.

## One box, three metrics, and two of them decided the same thing

The Batch 11 review recorded R19 as *"the gate reads the largest-area face and
tests its short edge"*. The trace found a third metric the review had not
counted, and a name collision beside it:

```
ranking       face_order_key      -area                     <- R19
face gate     _gate               short_edge vs min_face_px <- R19
report        quality.face_px     short_edge
crop side     crop_face           max(width, height)        <- R23, new
image gate    _gate               image short edge vs min_short_edge
detect floor  detectMultiScale    minSize, a per-axis short-edge floor
```

`min_short_edge` and `min_face_px` are consecutive lines of one `_gate`, both
equal 64, and measure different objects — the image and the face. Nothing in
either name says which.

## The disagreement has one direction, and it is provable which

The brief asked for four executable cases. **One of them is empty, and proving
that is what identifies the defect.**

- *area-largest scorable but short-edge-largest not* — **impossible.** The gate
  is `short_edge >= floor` and the short-edge-largest box maximises exactly the
  gated quantity, so if it fails then every box fails. 200 000 random box sets,
  **0 witnesses**. This holds under the legacy ordering too: it is a fact about
  the gate, not about the fix.
- *short-edge-largest scorable but area-largest not* — **this is R19**, and it is
  the only direction that was ever a defect.

Through the real `analyze_image` with a scripted detector, boxes `900x50` and
`200x200`:

```
before  insufficient_evidence / face_too_small   0 crops scored   face_px=50
        faces[] = [(900, 50, face_too_small)]    the 200px face absent entirely
after   manipulated p_fake 0.952574              1 crop scored    face_px=200
        faces[] = [(200, 200, scored)]
```

## Latent contradiction, not two independent policies

> **Gate completeness.** With ranking metric `r` and gate metric `g`, the state
> *"`faces[0]` fails the gate and some other face would pass"* is reachable
> **iff** `r` and `g` are not monotonically related.

Area and short edge are not. So the pairing admits a published `FACE_TOO_SMALL`
with `gate_passed=false` for a file containing a face that clears every floor —
a false statement about the file, from a gate whose stated purpose is to route
*low-information* input to abstention. Two defensible questions; a composition
that answers neither.

## The activation threshold is three pixels, and so is the fix's

A witness needs `se1 < F <= se2` and `area1 > area2`, hence
`(F-1)(F-1+d) > F^2`, hence **`d > 2 + 1/(F-1)`** — `d >= 3` at every floor.
Brute-forced at floors 16/32/64/128: no witness at `d = 2`, a witness at `d = 3`.
The minimal one is `66x63` (area 4158) outranking `64x64` (area 4096).

The **same** threshold governs the two orderings: `area-desc` and
`short-edge-desc` disagree iff `se1 > se2` and `area1 < area2`, which also needs
`d >= 3`. So changing the ranking metric is a no-op exactly where the defect
cannot fire and takes effect exactly where it can — 30 000 box pairs with
`|w-h| <= 2` enumerated, **0 disagreements**.

That coincidence is the whole scheduling argument, and it is why R19 was closed
before the SCRFD migration rather than with it.

## The record's reason for R19 being latent was wrong

```
cascade window size (24, 24)          square
1 025 Haar boxes measured             max |w - h| = 1, non-square 0.13-0.37%
```

**Haar does not return square boxes.** Per-axis rounding in OpenCV's scale
pipeline, and `groupRectangles` averaging a cluster, each land width and height
on different integers. The conclusion the review drew is still correct — `d <= 1`
against a threshold of 3 — but the safety margin is **two pixels of an OpenCV
implementation detail**, covered by no dependency pin and asserted by no test
until this batch. `test_the_shipped_detector_emits_boxes_within_the_safe_margin`
now holds it and says what to do if it ever fails.

SCRFD exceeds the threshold roughly eightfold on a 100 px face, so the migration
arms it immediately — but so would a profile cascade, a re-trained Haar asset, or
an OpenCV change to its rounding. The activation condition belongs in the record
as a number rather than as a detector name.

## What shipped

**One metric.** `FACE_SIZE_METRIC = "short-edge"` and `FaceBox.size_px`, read by
the ranking, the gate and `quality.face_px`. `size_px` equals `short_edge` and is
not redundant with it: `short_edge` is a fact about a rectangle, `size_px` is the
platform's choice of which fact to use.

**One ordering rule.** `area-desc,...` becomes
`short-edge-desc,area-desc,...`. Area survives as the secondary key, where it
does real work: among boxes that resolve equally well, prefer the one covering
more of the image. Still a *name* rather than a bytecode hash, still held to the
behaviour by a test — now on a box set that separates the two components.

**Short edge rather than area**, because the short edge is what decides whether
facial detail is resolvable — a `900x50` box has a large area and no vertical
detail — and because `QualityReport.face_px` already documented itself as a short
edge.

**Rejected: rank by scoreability.** It also scores zero false abstentions, but
the ordering would then depend on `MediaCaps.min_face_px` — a deployment knob in
a *different* fingerprint — so two fingerprints would jointly determine which
face is scored, undoing Batch 12's deliberate separation of *whether* an input is
scored from *which* face is. It also leaves two face-size metrics in the system.
**Rejected: documentation alone.** The published `FACE_TOO_SMALL` is a false
statement, not an unusual policy, and the bundle does not carry the discarded
boxes for a client to reason about.

## Migration impact

```
.f. segment      d662e22d -> 19559387
effective_ver    stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387
string length    49 -> 49 characters
gate fingerprint ae232cba, unchanged -- the floor did not move, only the metric
```

One-time full cache miss on both media kinds; namespaces do not overlap, so
nothing stale is served. Idempotency keys re-derive per submission. Benchmark
rows stay readable at `RUN_SCHEMA_VERSION` 1.2.0, and a pre- and a post-B14 run
now compare as different models — *conservative rather than correct here*, since
under Haar the two produce identical numbers; the cost is one refused comparison
across the boundary. Calibration witnesses, split locks, stored bundles, the
OpenAPI snapshot and `CONTRACT_VERSION` are all untouched.
`parse_model_version` stays total on all three eras.

## Guarantees NOT gained

1. **R16 is untouched.** One face's verdict is still published as the media's —
   the *right* face now, still one face. Asserted, including that a manipulated
   smaller face stays invisible to the published verdict.
2. **R20 is not formally closed.** Its defect condition is gone — `face_px`'s
   description is now true of the value it carries — but the prose correction is
   its own change and R19 was the brief.
3. **R21 is untouched.** `faces[]` still has no consistency invariant; asserted.
4. **The gate's *metric* is in no fingerprint.** `gate_fingerprint` covers the
   floor, not the quantity compared against it. Editing `_gate` to test area
   would move no identity string; only
   `test_the_ranking_and_the_gate_read_the_same_quantity` stands in the way. Same
   shape of guarantee `ORDERING_RULE` has had since Batch 12 — a name held to
   behaviour by a test rather than by a hash — and recorded rather than fixed
   with a new policy field, which would double-record what `ordering` names and
   still not bind the gate.
5. **The two-pixel margin is measured, not guaranteed.** No dependency pin covers
   OpenCV's rounding path.
6. **R13, R18's capability half, R22 untouched.** `size_px` appears zero times in
   `video_pipeline.py`.

## New finding: R23 — the crop is scaled by the long edge, and clipped

Found while tracing the third metric, recorded rather than fixed, per the brief.

```
box                side    crop shape   square?   face fraction
200x200@(100,100)   260      260x260      True         59.2%
400x100@(100,100)   520      520x520      True         14.8%
900x50@(100,100)   1170     1170x600     False          6.4%
```

`crop_face` computes `side = max(width, height) * margin`. For a free-aspect box
the face occupies a vanishing fraction of what the model is shown, and a clipped
crop is not square — so `prepare_image` resizes a rectangle to a square, the
aspect distortion `crop_face`'s own docstring says it exists to avoid. Both are
unreachable while boxes stay within a pixel of square and both are armed by the
same migration that arms R19. Pinned by
`test_the_crop_is_still_scaled_by_the_long_edge`, which fails and asks to be
replaced if the geometry changes.

## Still open after Batch 14

R9 (quarantine, by design), R10 (GPU/training), **R13** (310×, deferred to SCRFD
with an acceptance criterion — still the largest open finding), R16 (by design,
needs a deliberate decision), **R18's capability half** (a faceless clip decides;
needs a video face stage), **R20** (prose; defect condition removed), **R21**
(prerequisite for any R16 fix), **R22** (documentation, architectural for R16),
**R23** (latent, armed by SCRFD). **R12, R17 and R19 are closed.**

The sequencing note is now shorter than it was. R19 is done and did not need the
migration; R20 is a prose change that can go any time; R23 and R18's remaining
half are genuine detector work. What Batch 14 adds to the note is a method: state
a latent finding's **activation condition as a number**, then check the shipped
component against it. R19's number was three pixels and the shipped detector sat
one pixel away — which is what turned "defer to SCRFD" into "close it this
afternoon".

---

# Batch 15 resolutions — R23 characterised; R24, R25, R26 opened

The full trace, the assumption table, the distortion measurements, the six
alternatives and the identity analysis:
[`docs/CROP_GEOMETRY.md`](docs/CROP_GEOMETRY.md). Proof and doubles:
`tests/test_crop_geometry.py` (23 tests). Re-runnable study:
`tests/crop_geometry_study.py`.

**An investigation batch. No geometry changed** — `crop_margin` 1.3,
`crop_shape` `square-longest-edge`, face-selection identity `19559387`,
`STUB_ENGINE_VERSION` and `CONTRACT_VERSION` all unmoved, asserted by
`test_batch_15_changed_no_geometry`. Two documentation corrections shipped and
are named below rather than made quietly.

## R23 is all three categories, and it was two defects wearing one number

Batch 14 recorded R23 as *"`crop_face` scales the crop by the box's long edge and
clips it"*, latent, armed by SCRFD. Measured, it separates:

```
occupancy collapse  <- free aspect      -> LATENT, armed by SCRFD
aspect distortion   <- border clipping  -> LIVE TODAY, needs no free aspect
```

**The distortion half is live and is not SCRFD-gated.** A 200×200 box — exactly
the shape Haar emits — flush with the right edge of a 480×480 image yields a
230×260 crop, which `prepare_image` then stretches onto the square model input:

| Position (200×200 box, 480×480 image) | Crop | Anisotropy | Face off-centre |
|---|---|---|---|
| centred | 260×260 | 1.0000 | (0, 0) |
| flush **left** | 260×260 | 1.0000 | **(−30, 0)** |
| flush **right** | **230×260** | **1.1304 → 13.0% stretch** | (15, 0) |
| bottom-right corner | 230×230 | 1.0000 | face arrives **28% larger** |

Three distinct failures, not one, and **the clamp at 0 and the truncate at *w*
are not each other's mirror**: the left border keeps the crop square and slides
the face off centre; the right border keeps the face centred and destroys the
shape; the corner does neither and changes the scale. Measured on content as
well as shape — a circle in the crop region reaches the model as an ellipse.

**Free aspect, by contrast, distorts nothing.** With room to spare the window is
square, so the crop is square, at every aspect ratio including 900×50. What free
aspect destroys is occupancy: 59.2% of the model's input is face at aspect 1.0,
44.4% at 0.75 (ordinary for SCRFD), 3.3% at 900×50.

## The activation condition, closed form

Truncation on the right occurs **iff**

```
gap < ceil(side/2) - ceil(s/2)        gap = W - (box right edge)
```

— about **15% of the box's own width** at margin 1.3, equivalently any face
wider than 77% of the frame. **0 mismatches over 70 000 brute-forced cases.**
That is ordinary framing, not a pathology.

## The correction, proved rather than promised

`LegacyClipToBounds` is the shipped `crop_face` verbatim, pinned to the real
function on pixels as well as shape. `PadToSquareCrop` computes the same window
and **fills** it instead of cutting it down: anisotropy 1.0 at every swept
position, face centred at every border, and demonstrably different pixels into
the model.

> The shipped code computes a window and then **intersects** it with the image,
> so the bounds enter the crop's **shape**. The correction computes the same
> window and then **fills** it, so the bounds enter only its **content**. Shape
> is what the square resize is sensitive to.

Asserted directly — the legacy crop's shape changes as the image width varies
across 480/520/560/600 with the box fixed; the corrected crop's does not.

## Reproducibility: yes, and it is R17's shape again — R25

Two deployments agreeing on weights, calibration, gate, face-selection **and**
decode identity can hand the model different pixels, through three knobs Batch 12
did not reach:

```
1. ModelSpec.input_size      384 vs 224 -> byte-identical effective_version
2. the interpolation rule    INTER_AREA vs INTER_LINEAR, mean |d| 5.402/255
3. the clipping/centring rule  clip vs pad, both honestly "square-longest-edge"
```

Knob 2 has a second problem: the rule is `h > size or w > size`, so a crop wider
than the input but *shorter* than it is area-filtered on the axis being
**enlarged** — a low-pass filter on an upsample, in a module whose docstring says
`INTER_AREA` is chosen because forensic signal lives in high-frequency texture.

Knob 3 is the `ORDERING_RULE` problem again: `crop_shape` names how `side` is
derived and says nothing about what happens where the window leaves the image, so
a name in an identity string is only as good as the test holding it to the
behaviour — and here there is none.

**Where each belongs** (§10 of the report, argued with Batch 9's and Batch 12's
criteria): the clipping rule and any fill mode in the **face-selection identity**,
which already owns `crop_margin` and `crop_shape`; `input_size` in the **weights
identity**, because a checkpoint has an input size; the interpolation *choice*
with the rest of the shipped preprocessing, its *implementation* already
correctly in the decode identity. **Not implemented** — closing R25 dumps the
cache, and spending that migration before the geometry change it exists to
protect spends it twice. Land it with the change.

## R24 — the largest geometry defect is not in the crop path

`extract_frames(target_size=input_size)` resizes each **whole frame** onto a
square through the same per-axis-independent `resize`:

```
1920x1080 -> 384x384   anisotropy 1.7778
a CIRCLE of radius 400 reaches the model as a 160 x 284 ELLIPSE
through a letterbox double:                      160 x 160
```

Live on **every video bundle the platform has ever published**, ~14× the image
path's worst case, into a branch A the architecture declares to consume an
aligned face crop. Not R23 and not R18. Pinned, paired with `LetterboxResize`,
not fixed — a video face stage replaces this code path entirely, so fixing it in
isolation is work SCRFD throws away.

## R26 — the bundle publishes a box the model never saw

`face_track` records the detector's box; the region actually scored is expanded,
possibly translated, possibly truncated and always resampled after that box is
taken. Nothing in `EvidenceBundle` carries the crop window. `face_track` also
clamps `x, y >= 0` where `crop_face` does not — unreachable under Haar, active
the moment a detector regresses a box across the frame edge.

## Why nothing was fixed, and why that differs from Batch 14

Batch 14 closed R19 *before* SCRFD because the fix was provably output-identical
under the shipped detector — 30 000 box pairs, zero divergence — so it cost one
cache generation and no benchmark discontinuity. **No crop-geometry option has
that property.** Every one of the six alternatives changes the pixels of some
real input today, and none can be validated against a real model because there is
not one.

What Batch 15 buys instead is that **activation stops being silent.** Before it,
every geometry assertion in the suite was a byte comparison or a status-quo
tripwire: nothing asserted that the resize is isotropic, nothing asserted
occupancy, nothing exercised a border with a non-corner box, and the one test
named for crop shape — `test_crop_face_is_square_and_clipped_to_bounds` —
asserted `crop.size > 0` on the single corner fixture where squareness holds by
coincidence. Move its own box off the corner and it is 16×52, a 3.25× stretch.

## Documentation corrections shipped

1. **`crop_face`'s docstring** no longer claims to avoid the aspect distortion
   that stretches facial geometry. It states the three border behaviours, the
   closed-form truncation condition and the occupancy consequence.
2. **`test_crop_face_is_square_and_clipped_to_bounds`** →
   `test_crop_face_stays_inside_the_image_and_is_never_empty`, with the reason in
   its docstring.

Both are pinned against the legacy text recorded verbatim, and both failed before
the correction.

## Guarantees NOT gained

1. **No geometry is fixed.** Distortion, mis-framing, corner scale change and
   occupancy collapse all still happen on every affected input.
2. **R24 is not fixed.** Video is still scored on squashed frames.
3. **R25 is not closed.** All three knobs still sit outside every identity.
4. **R26 is not closed.**
5. **No measurement with a real model.** Every score here is `StubBackend`, which
   hashes its input — *that* the model sees different pixels is established; *how
   much it matters* needs weights. The 13% stretch is a fact about the tensor.
6. **The fill mode is unchosen** — replicate, reflect, constant or blurred is a
   modelling question.
7. **R13, R16, R18's capability half, R20, R21, R22** untouched.

## Still open after Batch 15

R9 (quarantine, by design), R10 (GPU/training), **R13** (310×, deferred to SCRFD
with an acceptance criterion), R16 (by design), **R18's capability half**,
**R20** (prose; defect condition removed by B14), **R21**, **R22**, **R23's
correctness halves**, **R24**, **R25**, **R26**. **R12, R17 and R19 are closed.**

**The SCRFD migration now has a defensible baseline.** Five things that would
have activated silently under free-aspect boxes each have a test that fails when
they move, and the sequencing is explicit: R23's geometry and R25's fingerprint
land together, with SCRFD, because the fingerprint exists to protect the
geometry change and neither is worth a cache generation alone.

---

# Batch 16 resolutions — R21 closed

The dependency graph of every open finding, the next-finding analysis the brief
asked for, and the full trace, fail-before proof and quantification:
[`docs/FACES_CONSISTENCY.md`](docs/FACES_CONSISTENCY.md). Proof and the named
double: `tests/test_faces_consistency.py` (16 tests).

**No contract change.** `CONTRACT_VERSION` 1.0.0; no field added, removed or
retyped; no OpenAPI signature change and no schema snapshot regenerated. **No
fingerprint moves** — `STUB_ENGINE_VERSION`, `effective_model_version`,
`gate_fingerprint`, `calibration_fingerprint`, the decode and face-selection
identities, `RUN_SCHEMA_VERSION` and `CALIBRATION_WITNESS_VERSION` are all
untouched. **No cache generation, no migration.**

## The architecture review came first

The brief required a report before any code. It is §1–6 of the doc. In short:

- **Dependency graph.** The open findings split into a left chain that runs on
  contract/reproducibility work alone — **R21 → R16(image path) → R22/R20** — and
  everything else, which is gated on the SCRFD migration (R13, R23's
  occupancy half, R24, R25), on trained weights (R10, R16 Option 3, the
  *magnitudes* of R13/R23/R24), or on a second machine (R6's cross-OS residual).
- **Next finding: R21.** It is the only open finding that satisfies all five of
  the brief's criteria — reduces reproducibility risk, provable with executable
  tests, no weights, no SCRFD, no contract-version change — *and* it is the
  recorded prerequisite for R16 (`docs/MULTI_FACE_SEMANTICS.md` §12, Option 0:
  *"a prerequisite, not a choice"*).
- **SCRFD-blocked:** R13, R23 (occupancy), R24, R25 (sequenced with it), R16 in
  any meaningful form. **Weights-blocked:** R10 (also no training loop), R16
  Option 3, the real-photograph magnitudes of R13/R23/R24. **Closable now with
  architecture work:** R21 (this batch), R20 (prose, its own change), R22 (a
  paper decision), R9 (a design note — already reported every run), R6's
  `frame_count` sub-residual (narrowly; the test is as decoder-dependent as the
  bug), R26 (partially).

## The defect, measured

`EvidenceBundle` carried seven `model_validator`s. `faces[]` appeared in exactly
one — as the *target* of `_top_frames_name_a_published_track`'s referential check
(Batch 13). Nothing constrained `faces[]` itself. Re-validated through JSON the
way storage and the wire do:

```
two faces, both track_id=0, contradictory p_fake -> validates: 2 [0, 0] [0.9, 0.1]
bundle.p_fake 0.9134827362  verdict manipulated   agrees with either face? False
```

Two defects under one number:

1. **Uniqueness.** `branches[]` has had `_branches_unique` since Batch 0;
   `faces[]` is the same shape — an identified list a client indexes and
   `top_frames` resolves against — with no such rule.
2. **Headline consistency.** A decided `p_fake` matching no published face is the
   class of drift `_verdict_follows_threshold` and the `ThresholdInfo` band
   derivation exist to make unrepresentable, left open on the one field a
   per-face R16 design will populate.

## Fail-before, executed

Named double **`LegacyEvidenceContract`** — the `EvidenceBundle` validators as of
Batch 15, looked up by name on the live class, minus the two this batch adds;
`test_the_legacy_contract_double_is_complete` fails if that set drifts. Against a
bundle built on `image_manipulated()` with only `faces[]` mutated:

```
duplicate track_id (id 0 twice, p_fake 0.913 / 0.1):
    LegacyEvidenceContract.accepts -> True     (satisfied the entire prior contract)
    EvidenceBundle.model_validate  -> ValueError: duplicate track_id in faces[]: [0, 0]

headline matches no face (manipulated @ 0.913, faces 0.80 / 0.10):
    LegacyEvidenceContract.accepts -> True
    EvidenceBundle.model_validate  -> ValueError: p_fake=0.9134827362 is the score
                                      of no published face ([0.8, 0.1])
```

## Quantified

For a decided bundle publishing one scored face, sweeping that face's score over
`{0, 0.05, …, 1.0} ∪ {headline}` (22 values):

```
                        prior contract    shipped contract
face score accepted            22                1        (only the value == headline)
duplicate-id pile of 8      accepted          refused
```

The prior contract admitted the published face's score anywhere in `[0, 1]`
against a headline it was meant to describe, and an unbounded pile of faces at
one id.

## What shipped

Two `model_validator(mode="after")` on `EvidenceBundle`, after
`_top_frames_name_a_published_track` (which now defers to them by name):

- **`_faces_are_uniquely_identified`** — the `_branches_unique` rule on
  `track_id`.
- **`_headline_is_the_score_of_a_published_face`** — if `p_fake` is not `None`
  and any published face scored, `p_fake` must equal one of those scores exactly.
  Vacuous on abstention (the in-band case legitimately publishes a per-face
  `p_fake` with a `None` headline) and vacuous when no published face scored (the
  video path's `faces=[]`). "A scored face", not "faces[0]", so an R16 design
  that designates a different face as the headline is representable; an
  *averaging* aggregate is not, and is meant to trip this rather than pass
  unnoticed.

One existing test fixture reconciled:
`test_video_face_semantics.py::test_a_resolving_track_id_is_still_accepted`
hand-built a video bundle whose published face carried `p_fake=0.9` beside a
computed headline of `0.9797` — itself an instance of the inconsistency R21 is
about, which no test had noticed. Its face score is now the bundle's own
headline; the test's intent is unchanged. `video_face_semantics_study.py` case C
got the same one-line reconciliation.

## Guarantees gained

1. No two published `FaceTrack`s share a `track_id` — by hand, from a factory,
   on deserialisation.
2. A decided headline `p_fake` is the score of a face the bundle publishes,
   whenever any published face scored.
3. `_top_frames_name_a_published_track` resolves against a unique set by
   construction.
4. R16's implementation has an invariant to build against.
5. A future averaging aggregate over faces trips a loud failure rather than
   silently redefining a Stable field.

## Guarantees NOT gained

1. **R16 is untouched.** A self-contradicting bundle is now impossible; a group
   photograph's one-face verdict is still not correct.
2. **Non-selected faces are still not scored, abstained or counted.**
3. **The headline rule is vacuous when no published face scored** — a decided
   bundle with only abstained faces is accepted, because telling that shape from
   the video path's legitimate `faces=[]` needs an R18 `face_stage` signal.
4. **The video path still publishes no faces** and names no track.
5. **No `faces[]` ↔ `disagreement` / `interval` relation** — an unexamined second
   face still does not widen the interval.
6. **No fingerprint moves, so nothing here protects a future policy or geometry
   change.** R25 still wants to land with the geometry change it guards.
7. **R20, R22, R6, R9, R10, R13, R23, R24, R25, R26 untouched.**

## Still open after Batch 16

R9 (quarantine, by design), R10 (GPU/training), **R13** (310×, deferred to SCRFD
with an acceptance criterion), R16 (by design; image-path plumbing is now
unblocked), **R18's capability half**, **R20** (prose; defect condition removed
by B14), **R22** (branch B whole-image vs crop — a paper decision),
**R23's correctness halves**, **R24**, **R25**, **R26**. **R12, R17, R19 and
R21 are closed.**

---

# Batch 17 resolutions — R22 closed

Full trace, the dependency graph, the executed proofs, the quantification and the
R16 analysis: [`docs/BRANCH_B_SEMANTICS.md`](docs/BRANCH_B_SEMANTICS.md). Proof
and named doubles: `tests/test_branch_b_semantics.py` (18 tests). Re-runnable
study: `tests/branch_b_semantics_study.py`.

**No contract change.** `CONTRACT_VERSION` 1.0.0; no field added, removed or
retyped; no OpenAPI signature change and no schema snapshot regenerated. **No
fingerprint moves** — `effective_version`
`stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387`, `calibration_fingerprint`,
`gate_fingerprint`, the face-selection and decode identities all unmoved,
asserted by `test_no_identity_string_moves`. **No behaviour change, no
migration.**

## The finding, re-derived (not assumed)

`enums.py`, `BRANCH_DESCRIPTIONS` and `README.md` declared branch B as
*"DINOv2-B/14 ... whole image"*, against branch A's *"aligned face crop"*.
Measured:

- **`ModelBackend.score` takes one tensor** and returns one logit vector per
  branch; `image_pipeline.py` and `video_pipeline.py` each call it once, with one
  argument. There is no interface through which branch B could receive the whole
  image while branch A receives the crop.
- **Image path, byte-for-byte** (`TensorCaptureBackend` + a corner-face image):
  the tensor branch B is scored from is `prepare_image(crop_face(...))` — digest
  `7a123e16d9b4`, matching the crop, **not** the whole image (`b25f545ffd15`).
- **Video path**: branch A is scored from `resize(whole frame)` — digest match.
  So in the image path B's claim is violated; in the video path **A's** "face
  crop" claim is violated. One single-tensor interface, two mismatches, opposite
  branches. (VIDEO_FACE_SEMANTICS §9.2 recorded the video half; this is the first
  measurement.)
- **Nothing on the wire carries the claim.** `BranchResult` has no `description`
  field; `BRANCH_DESCRIPTIONS` has no functional consumer (AST-checked); the
  markers `DINOv2` / `over the whole image` / `aligned face crop` appear in no
  schema and no OpenAPI document.

## The shipped contradiction, executed

`contracts/fixtures.py::image_no_face` — a **published OpenAPI example** — and the
real pipeline give opposite answers for a faceless image, and both are checked in
by passing tests:

| | `image_no_face` | real `ImagePipeline` |
| --- | --- | --- |
| `branch(B).status` | **SCORED**, weight 0.94 | **ABSTAINED** |
| `quality.gate_passed` | **True** | **False** |
| `verdict` | **MANIPULATED** | **INSUFFICIENT_EVIDENCE** |

`test_contracts::test_no_face_image_abstains_on_branch_a_only` and
`test_inference::test_no_face_path_abstains` both pass — opposite semantics for
branch B on a faceless image.

## What shipped

A **contract clarification**, the shape Batch 13 used for R18's false-claim half:

- **`BRANCH_DESCRIPTIONS[A]` and `[B]`, the `enums.py` comment, the `README.md`
  branch table** now state what the code does — branch B scores the same prepared
  tensor as branch A — and name whole-image scoring as the frozen-architecture
  target it is. `LegacyBranchBClaim` holds the pre-Batch-17 strings verbatim.
- **`image_no_face`** is annotated as depicting that target rather than current
  behaviour, with the old docstring preserved in the double. The example bytes
  are byte-identical; only the Python docstring changed.
- **`tests/stub_handlers.py::_branches(face_present=False)`** — dead code that
  reweights B to 0.94 on a faceless input — is labelled as the same target shape.
- **18 pinning tests**: the interface constraint, the byte-for-byte crop scoping
  (image) and whole-frame scoping (video A), the contradiction, the corrected
  claims (5 fail-before / pass-after), and `test_the_pipeline_never_scores_
  branch_b_without_a_face` so a future whole-image B cannot land silently.

**Zero existing tests changed.**

## Quantification

| | |
| --- | --- |
| Claim sites in source / on the wire | **3 / 0** |
| Wire fields carrying branch B directly / derived from it | **12 / 4** |
| Wire fields with a *wrong value* under any reading | **0** |
| Validators asserting branch B's input scope | **0** |
| Fixtures with B scored / depicting the contradiction | **6 / 1** |
| Existing tests touching a B interpretation / needing a change | **~8 / 0** |
| Source files whose prose changed / behaviour changes / migration steps | **4 / 0 / 0** |

## R16 interaction

`fuse` and `manipulation_evidence` take one `dict[BranchId, BranchOutcome]`.

- **B crop-scoped** (what the code does) → R16 Option 2 = call `fuse()` once per
  face, unchanged. Demonstrated.
- **B whole-image** → one image-level B pooled with each face-level A → a new
  `fuse` signature and a restructured renormalised mean.

**R22 does not block R16, is not blocked by R16, and is closed independently
here.** The code having chosen crop-scoped is the simpler shape for R16. The
residual — *build a genuinely whole-image branch B* — is a real capability
decision, now **explicit** in the branch descriptions rather than latent, and it
gates nothing; it belongs with real weights and a dual-input backend.

## Guarantees gained

1. The branch-A and branch-B input-scope claims match the code.
2. The `image_no_face` contradiction is labelled as a target, not shipped
   behaviour.
3. The current pipeline behaviour (full abstention on a faceless image) is
   pinned; a whole-image B cannot land without a test firing.
4. `ModelBackend.score`'s single-input shape is pinned.
5. `BRANCH_DESCRIPTIONS` is pinned dead — a functional consumer reopens the "no
   contract change" basis.
6. R22's video mirror (branch A on whole frames) is measured, not just described.

## Guarantees NOT gained

1. **Branch B still scores the face crop.** The claim is honest now; the
   whole-image branch is not built.
2. **`image_no_face`'s verdict path is still unreachable from the pipeline.**
3. **`disagreement` / `interval` still mean "A vs B about the same crop".**
4. **`_abstain` still hard-codes branch B's abstention reason as
   `QUALITY_BELOW_FLOOR`** on a `NO_FACE_DETECTED` gate — false, and recorded as a
   follow-up (it changes a published `abstain_reason` value).
5. **Branch A's video mismatch is not fixed** — tied to R18 and R24.
6. **No structural validator forbids the contradiction** — it would reject the
   fixture and the target shape. Held apart by tests and prose.
7. **R16, R13, R20, R23–R26 untouched.**

## Still open after Batch 17

R9 (quarantine, by design), R10 (GPU/training), **R13** (310×, deferred to SCRFD
with an acceptance criterion), R16 (by design; image-path plumbing unblocked by
B16), **R18's capability half**, **R20** (prose; defect condition removed by
B14), **R23's correctness halves**, **R24**, **R25**, **R26**. **R12, R17, R19,
R21 and R22 are closed.**

The paper decision the multi-face review (`docs/MULTI_FACE_SEMANTICS.md` §12–13)
asked for on R22 is made: branch B is crop-scoped today, whole-image is the
roadmap, and R16 needs no fusion restructuring.

---

# Batch 18 resolutions — R20 closed

Full architecture review of R20 **and** R22, the dependency and migration
analysis, and the executed proofs:
[`docs/HEADLINE_AND_FACE_SEMANTICS.md`](docs/HEADLINE_AND_FACE_SEMANTICS.md).
Proof and named double: `tests/test_headline_and_face_semantics.py` (28 tests).

**No contract change.** `CONTRACT_VERSION` 1.0.0; no field added, removed or
retyped; `int | None` and `ge=0` on `face_px` unchanged. **No fingerprint
moves** — `effective_version`
`stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387`, `calibration_fingerprint`,
`gate_fingerprint`, the face-selection and decode identities, `RUN_SCHEMA_VERSION`
(1.2.0) and `CALIBRATION_WITNESS_VERSION` (1.2.0) all untouched, each asserted by
a test. **No behaviour change, no cache generation, no migration.**

## The collision — this is Batch 18, not Batch 17

The brief was "Batch 17 — R20 and R22 together". A concurrent session completed
**Batch 17, closing R22**, while this review was in progress (see
`docs/BRANCH_B_SEMANTICS.md` and the Batch 17 section above). This batch is
therefore **Batch 18** and **R20-primary**. Batch 17's R22 closure is not
reopened, rewritten or duplicated; `docs/HEADLINE_AND_FACE_SEMANTICS.md` §2
answers the brief's R22 questions against Batch 17's evidence and §6 of the test
suite re-verifies its three load-bearing invariants (single-tensor interface, no
wire surface, unmoved identity) independently. All pass.

## The finding, re-derived (not assumed)

`contracts/common.py` declared `face_px` as *"Short edge of the largest face."*
The value published (`inference/image_pipeline.py`) is:

- **image, ≥1 face:** `faces[0].size_px` — `faces[0]` is the box that leads
  `face_order_key` on `-size_px` (R19/B14), and `FaceBox.size_px == min(w,h) ==
  short_edge`. So: the short edge of the face with the largest short edge, area
  breaking ties — equivalently the largest face size present.
- **image, no face:** `0` — a sentinel, always with `gate_passed=False`,
  `gate_reason=NO_FACE_DETECTED`.
- **video:** `None` — `video_pipeline.py` builds `QualityFacts(..., face_px=None)`
  unconditionally; the video path has no face stage (R18).

Three prose gaps: "largest" names no metric; the `0` sentinel is undocumented;
the video `null` is undocumented on the field.

**Documentation-only today — and the record is corrected, not rewritten.** The
Batch 11 review (`docs/MULTI_FACE_SEMANTICS.md` §10) recorded the value as *"the
short edge of the largest-**area** face"* and predicted R20 would become
"correctness-critical at SCRFD". That was true pre-Batch-14, when `face_order_key`
led with `-area` while the report took the short edge. **Batch 14 (R19) made
ranking lead with `-size_px`**, the quantity the report takes, which removed the
divergence *structurally* — SCRFD's free-aspect boxes go through the same one
metric. `test_the_batch_11_reading_of_the_value_is_refuted_by_execution` runs it:
on `THIN(900×50)` / `GOOD(200×200)` the Batch 11 computation returns `50`, the
live pipeline publishes `200`. §10 of `MULTI_FACE_SEMANTICS.md` now carries a
`>`-quoted superseding note; its original text is preserved.

## What shipped

- **`contracts/common.py`** — `face_px`'s `Field(description=...)` rewritten to
  name `size_px` / `min(width, height)`, the `0` sentinel, the video `null`, and
  a pointer to `docs/FACE_SIZE_POLICY.md`. Nothing else in the field changed.
- **`schema_out/*.json`** — regenerated (`python -m contracts.schema`). Rendering
  the bundle with the old and the new description and diffing every file: the
  **only** changed lines are `face_px`'s `description` (2 in
  `EvidenceBundle.schema.json`, 2 in `CreateUploadResponse.schema.json`, 4 in
  `contracts.bundle.json`, 0 everywhere else). No test guards these files against
  staleness (`--check` validates only example payloads); regenerating is tidy,
  not required.
- **`docs/MULTI_FACE_SEMANTICS.md`** — superseding annotation on the §10 R20
  entry.
- **`tests/test_headline_and_face_semantics.py`** — 28 tests: the published
  values in all three cases, the executed refutation of the Batch 11 reading, the
  Haar-shaped-box agreement (why it was never a live defect), 4 fail-before /
  pass-after prose checks, the "nothing computes from `face_px`" AST sweep, and
  the full migration proof (identity, contract/schema/witness versions, OpenAPI
  signature, cache key), plus 3 R22 re-verification tests.
- **`tests/run_all.py`** — registers `S8-B18 headline + face semantics`.

**Zero existing tests changed for R20.** One test **repaired** — see below.

## Discovered in passing — a red test in the Batch 17 suite

The full suite at the Batch 17 baseline was **not green**:
`test_branch_b_semantics.py::test_the_whole_image_claim_is_not_on_the_wire`
failed. It greps `schema_out/EvidenceBundle.schema.json` for the bare substring
`"whole image"`, which matches `FaceTrack`'s **R16** docstring (*"published as
the whole image's"*) — unrelated to branch B, and flagged as such in Batch 17's
own study. **Classification: a test defect (over-broad match) in a shipped
batch**, not a platform defect — the branch-B claim genuinely reaches no wire
field. Repaired here (explicitly, not silently): the assertion now checks the
specific markers `"over the whole image"` and `"DINOv2"`, the phrasing Batch 17's
P5 already uses. This is the only change to a prior batch's files.

## Migration analysis — proved

| | Bump? | Proof |
| --- | --- | --- |
| `CONTRACT_VERSION` | no | field shape/constraints unchanged; every payload round-trips byte-identically; a description is not a client migration |
| `RUN_SCHEMA_VERSION` | no | `face_px` / `QualityReport` / `QualityFacts` appear nowhere under `evaluation/` |
| `CALIBRATION_WITNESS_VERSION` | no | witness digest covers calibration set + threshold + band, not contract prose |
| `STUB_ENGINE_VERSION` | no | `effective_version` = weights+calibration+gate+face-selection+decode; a description is in none; asserted at the pinned constant |
| cache invalidation | no | cache key = `sha256(bytes) + effective_version + decode identity`; unmoved |

## Guarantees gained

1. `face_px`'s description is true of every value the engine publishes and names
   the metric behind "largest".
2. The pre-Batch-14 value claim is refuted by execution, not just superseded.
3. `face_px` is pinned reporting-only — a future decision that reads the
   published figure trips a test.
4. "Documentation change ⇒ zero artifact change" is executable (render, diff).
5. R22's Batch 17 closure has an independent second measurement.
6. The suite is green.

## Guarantees NOT gained

1. **The `0` (image) vs `None` (video) split is documented, not harmonised** —
   making the image path emit `None` would change a published value. Follow-up.
2. **R16 untouched** — with multiple scored faces, `face_px` is still a one-face
   view.
3. **`schema_out/` is still unguarded against staleness** (Batch 13's note).
4. **No new fingerprint** — correcting prose protects no future policy or
   geometry change.
5. **Branch B still scores the crop**, and `_abstain` still gives it a false
   reason (Batch 17 §10 follow-up). Not R20's.
6. **R6, R9, R10, R13, R16, R18 capability half, R23–R26 untouched.**

## Still open after Batch 18

R9 (quarantine, by design), R10 (GPU/training), **R13** (310×, deferred to SCRFD
with an acceptance criterion), R16 (by design; image-path plumbing unblocked by
B16), **R18's capability half**, **R23's correctness halves**, **R24**, **R25**,
**R26**, R6's `frame_count` sub-residual. **R12, R17, R19, R20, R21 and R22 are
closed.**

Decision, per finding:

- **R20 — CLOSED.** Pure documentation defect after B14; correction proven
  inert beyond one string; behaviour pinned by B14 and re-pinned here.
- **R22 — CLOSED (Batch 17), re-verified.** Not reopened.
- **B18 test-repair — done.** Recorded so the history shows the suite was
  briefly not-green after Batch 17.

---

# Batch 19 resolutions — R6's `frame_count` residual reviewed; R27 opened

Full threat model, dependency trace, artifact flow, the executed fail-before, the
migration analysis and the ten-question answer:
[`docs/FRAME_COUNT_REPRODUCIBILITY.md`](docs/FRAME_COUNT_REPRODUCIBILITY.md).
Re-runnable study: `tests/frame_count_reproducibility_study.py` (6 experiments).
Proof and named doubles: `tests/test_frame_count_reproducibility.py` (16 tests).

**No behaviour change. No contract change.** `CONTRACT_VERSION` 1.0.0,
`RUN_SCHEMA_VERSION` 1.2.0, `CALIBRATION_WITNESS_VERSION` 1.2.0 all unmoved. **No
fingerprint moves** — `STUB_ENGINE_VERSION`
`stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387.d.c20a064a`,
`effective_model_version`, `decode_identity` (`c20a064a`), `gate_fingerprint`,
`calibration_fingerprint`, the face-selection identity — every one asserted
unchanged by a test. **No cache generation, no migration, no `schema_out` or
OpenAPI regeneration.**

## What `frame_count` is, measured

`MediaInfo.frame_count` = `int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)` from
`probe_video`, taken **verbatim, unvalidated**. `probe_video` then derives
`duration_s = frame_count / effective_fps`, so **one wrong count reaches two wire
fields**. The value's only score-affecting consumer is
`sample_frame_indices(meta.frame_count, n_frames)`, which chooses which frames are
decoded. `CAP_PROP_FRAME_COUNT` is produced by **libavformat** (the demuxer), not
libavcodec.

## The decomposition (per the brief — R6 splits)

| Piece | Class | This batch |
| --- | --- | --- |
| **R6-a** — uniform sampling over the declared count: a miscount of ~one sampling step (`frame_count/n` frames) moves ≥50% of the scored frames | design limitation | **measured and pinned**; fix (timestamp-fraction sampling) is a serving-path change → video-face-stage acceptance criterion |
| **R6-b** — nothing cross-checks `frame_count`; an under-count silently drops the tail of a clip from the verdict | design limitation shading into correctness | **characterised**; fix is a serving-path change → same criterion |
| **R27** *(new)* — the identity that gates the result cache (B9) and the benchmark comparison (B11) omits libavformat | **reproducibility defect** | **opened**, migration analysed, tripwired |

## R27 — measured, not argued

`getBuildInformation()` on this build: `avcodec: YES (58.134.100)` and
`avformat: YES (58.76.100)` — **separate libraries, different versions**.
`decode_env._build_facts()` parses only the avcodec line (`_AVCODEC_RE`).
`avformat` never enters `DecodeEnvironment.identity`, which is
`fingerprint({opencv, backend, avcodec, system, machine})`.

**Fail-before, executed** (`test_two_demuxers_disagreeing_by_one_frame_share_a_
cache_key`, `study` S5). `MiscountingDemuxer` — the real `probe_video` with
`frame_count → frame_count − 1`, the exact E2 witness — vs the real probe, same
bytes, real `analyze_video`:

```
true demuxer : frame_count=200  p_fake=0.9739  verdict=manipulated
-1  demuxer  : frame_count=199  p_fake=0.9546  verdict=manipulated
sampled frame indices moved : 11/16
decode_identity()   : c20a064a / c20a064a   (identical)
bundle.model.version: identical
content cache key   : identical
```

A cache populated by the first deployment serves its verdict to the second, which
scored a different set of frames. The 0.019 `p_fake` delta is the **stub upper
bound** (the stub hashes frame bytes; DECODE_DETERMINISM E9); the honest per-frame
estimate is the surrogate's `|Δp| ≈ 0.005`, ~1.3–1.6% of decisions (E10).

**Not reproduced: the trigger.** `study` S3 tried trailing garbage, truncation, a
flipped `mdat` byte, and both backends; every intact container counted exactly,
every damaged one refused — consistent with E1. A real divergence needs true VFR
media (this box's `VideoWriter` only makes CFR) and a second libavformat build.
The finding is real by construction (the identity demonstrably omits the library
that demonstrably produces the count), not by a captured occurrence — stated
plainly.

## Preserve-and-refute

`docs/DECODE_DETERMINISM.md` §3 recorded this as one row: *"Container metadata
divergence — Unresolved. E1 not reproduced, E2 shows 60% amplification if it ever
happens. No validation of `frame_count` exists."* That was **correct but
understated**: it filed a structural gap in the cache/benchmark identity as a
footnote about a missing validator. The row is preserved; a pointer to
`FRAME_COUNT_REPRODUCIBILITY.md` is added. Batch 8's E1/E2 numbers stand and are
re-measured here (S2, S3).

## Answers to the brief's questions 5–8

- **Q5 — can two deployments publish different `frame_count` under identical
  current identities?** **Yes** — if they differ only in libavformat. Executed.
- **Q6 — does Batch 9 runtime identity protect against it?** **Partially** —
  covers a backend / OpenCV / avcodec / OS change, not a libavformat-only change,
  and `decode_identity` is passed in, never detected.
- **Q7 — does Batch 11 benchmark identity protect against it?** **Partially, and
  less than it looks** — `regression_gate` refuses only on *differing*
  `decode_identity`, which omits the demuxer; and the evaluation package never
  decodes, so it cannot observe `frame_count` at all.
- **Q8 — does any cache / benchmark / calibration / forensic workflow consume
  `frame_count`?** **No — none consumes the value.** It reaches the wire
  (`frame_count`, `duration_s`) and `sample_frame_indices`, and nothing else.
  Its reproducibility impact is entirely indirect, through the frames it selects
  into `p_fake` — and `p_fake`'s cross-deployment guard is the `.d.` cache-key
  segment, which is exactly where R27 bites.

## Migration analysis

**This batch: zero.** No source behaviour changed; the table in
`FRAME_COUNT_REPRODUCIBILITY.md` proves each version constant and the cache
generation are untouched.

**The deferred R27 fix** (fold libavformat + libswscale into
`DecodeEnvironment.identity`): **no** `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` /
`CALIBRATION_WITNESS_VERSION` / OpenAPI / `schema_out` change; **yes**
`STUB_ENGINE_VERSION` `.d.` segment moves and a one-time cache-namespace miss per
file (precedented exactly by Batch 9 §6.2 and Batch 11). It is a *cache
generation* batch, not a schema batch, and belongs with the next batch that
regenerates the `.d.` segment or with the video face stage.

## Why not close R27 now

Zero measured occurrences on this build; the change forces a global cache
generation and ripples through ~6 live identity assertions plus several
historical identity docs whose `c20a064a` is a measurement that must not be
rewritten. Batch 8 deferred R12's remediation 2 for the same reason ("a batch
whose brief permits it") → Batch 11 did it. Same posture here.

## Guarantees gained

1. `frame_count`'s definition, its single score-affecting consumer, its two-wire
   reach, and its absence from every key are traced and pinned.
2. The amplifier is measured with its honest shape (severe when the miscount ≳
   one step; mild otherwise) and pinned as a lower bound.
3. **R27 is a named finding**, measured (`avcodec ≠ avformat`) and tripwired —
   `test_the_decode_identity_still_omits_the_demuxer` fails the moment the fix
   lands.
4. The cross-deployment hazard is executed: one identity, two counts, two
   `p_fake`s, one cache key.
5. The amplifier measurement is portable (`sample_frame_indices` is pure
   arithmetic); a real trigger is not, and the doc says which is which.

## Guarantees NOT gained

1. **Two demuxers still disagree** on an estimated count; nothing here makes them
   agree or detectable.
2. **`frame_count` is still trusted verbatim**; an under-count still silently
   drops a clip's tail.
3. **Sampling is still uniform over the declared count** — the robust
   timestamp-fraction sampler is deferred to the video face stage (§9 of the doc,
   a carry-forward acceptance criterion).
4. **The trigger is still unreproduced** on this build.
5. **No cache generation**, so a verdict already cached under a miscounting
   demuxer is not retroactively separated.
6. **R6-a/R6-b's serving-path fix, R10, R13, R16, R18 capability half, R23–R26
   untouched.**

## Still open after Batch 19

R9 (quarantine, by design), R10 (GPU/training), **R13** (310×, SCRFD acceptance
criterion), R16 (by design), **R18's capability half**, **R23's correctness
halves**, **R24**, **R25**, **R26**, **R27** (decode identity omits the demuxer —
migration analysed, deferred to a cache-generation batch), **R6-a/R6-b** (the
`frame_count` sampling/validation fix — a video-face-stage acceptance criterion).
**R12, R17, R19, R20, R21, R22 are closed; R6 is PARTIALLY CLOSED.**

Decision, per finding:

- **R6 — PARTIALLY CLOSED.** The `frame_count` residual is decomposed: R6-a
  (uniform-sampling amplification) and R6-b (no validation) are measured, bounded
  and pinned, with their serving-path fix made an explicit video-face-stage
  acceptance criterion; the reproducibility half is promoted to R27. Exact
  reason: every in-scope fix is either a serving-path change that moves every
  video verdict (timestamp sampling, a new abstention) or a global cache
  generation (R27's identity fix), so the truthful closure now is
  characterisation + pins + a scoped recommendation, not a code change.
- **R27 — OPEN.** New finding, measured and tripwired; migration analysed;
  deferred to a cache-generation batch or the video face stage.
- **No other new findings.** The `_AVCODEC_RE` fallback-to-constant and the
  `decode_identity`-is-passed-in-never-detected observations are recorded inside
  R27 as aggravating factors, not as separate findings.

> **Closed by Batch 20.** Batch 19 deferred R27's fix; Batch 20's brief permitted
> the cache generation, so it was done. Batch 19's characterisation
> (libavformat produces `frame_count`, the identity omitted it, the trigger is
> unreproduced on one build) all stood; Batch 20 added that **libswscale and
> libavutil** were also omitted and folded all three in. See
> `docs/DECODE_IDENTITY_COMPLETENESS.md`.

---

# Batch 20 resolutions — R27 closed

First-principles re-derivation of the video decode-path component list, the
four-protection analysis, the migration and the executed fail-before:
[`docs/DECODE_IDENTITY_COMPLETENESS.md`](docs/DECODE_IDENTITY_COMPLETENESS.md).
Study: `tests/decode_identity_completeness_study.py` (8 experiments). Proof and
named doubles: `tests/test_decode_identity_completeness.py` (22 tests).

**One cache generation. No contract, schema, OpenAPI or witness-version change.**
`CONTRACT_VERSION` 1.0.0, `RUN_SCHEMA_VERSION` 1.2.0, `CALIBRATION_WITNESS_VERSION`
1.2.0 all unmoved. `STUB_ENGINE_VERSION` `.d.` segment
`c20a064a` → `a97ccb68`; `model_identity_of(STUB_ENGINE_VERSION)` unchanged at
`stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387`, asserted.

## The finding, re-derived (not assumed)

The Batch 19 report recommended "libavformat (and possibly libswscale)" and told
this batch not to assume it. Traced from `analyze_video` → `extract_frames` →
`backend.score`, the video path links **four** FFmpeg libraries and
`getBuildInformation()` versions each separately — measured on this build:

```
avcodec:   YES (58.134.100)   <- decodes the frame        (in decode_identity)
avformat:  YES (58.76.100)    <- demuxer: CAP_PROP_FRAME_COUNT / FPS / POS_MSEC   (MISSING)
avutil:    YES (56.70.100)    <- pixfmt + rational ts math                        (MISSING)
swscale:   YES (5.9.100)      <- YUV->BGR in retrieve() (B8 E4 measured this)     (MISSING)
avresample: YES (4.0.0)       <- audio resampler, not on the video path           (excluded)
```

`decode_env._build_facts()` parsed only the `avcodec:` line. So R27's demuxer
point was one of three: **libavformat, libswscale and libavutil were all
omitted.**

## The hole across all four protections — executed

`study` E4, two `DecodeEnvironment`s identical except `avformat 58.76.100` →
`59.16.100` (a routine distro-vs-prebuilt difference):

| Protection | Before B20 | After B20 |
| --- | --- | --- |
| content cache key (`sha256` + runtime version) | **blind** | separates |
| deployment identity (`/health`, `verify_version_alignment`) | **blind** | separates |
| benchmark identity (`regression_gate` refuses) | **blind** | separates |
| calibration identity (`CalibrationWitness` digest) | **blind** | separates |

**4/4 blind → 0/4.** `test_a_libavformat_difference_now_separates_all_four_
protections`, `test_the_benchmark_gate_refuses_across_a_demuxer_difference`,
`test_the_calibration_witness_binds_the_finer_identity`. The
`LegacyAvcodecOnlyIdentity` double reproduces the pre-B20 `c20a064a` on this
build and shows all four were blind.

**Answering the brief's Q6 directly: yes — the cache and benchmark protections
were incomplete**, for a class of environment difference (FFmpeg demuxer/scaler)
broader than the CPU-field gap Batch 11 §13.6 already named.

## What shipped — the smallest expansion

`_build_facts()` now parses `avcodec / avformat / avutil / swscale` into the one
existing `ffmpeg_version` string field (`"avcodec X / avformat Y / avutil Z /
swscale W"`). No new `DecodeEnvironment` field — `to_dict`, `differences`,
`describe` and every field-list test are untouched. `DecodeEnvironment.identity`
digests the same key it always did; only the value grew. `avresample` is
excluded — `_VIDEO_FFMPEG_LIBS` names the four, and `study` E6 confirms no audio
decode API in `inference/`.

Live pins updated: `backend/config.py` (literal + docstring),
`tests/test_crop_geometry.py` (now asserts `model_identity_of(...)` — the model
half Batch 15 was actually responsible for — plus `.d.` presence, with a comment
that Batch 20 legitimately grew the decode segment).

## Preserve-and-refute

- **Batch 9 `RUN_IDENTITY.md`** described `ffmpeg_version` as "the libavcodec
  version" — accurate, but the scope was an oversight, not a measured decision
  like the CPU-field exclusion. Refuted here: libavformat/libswscale are on the
  path, measurably.
- **Batch 19's tripwire** `test_the_decode_identity_still_omits_the_demuxer`
  fired on this change and was **rewritten** to
  `test_the_decode_identity_now_covers_the_demuxer_batch_20`, asserting the new
  state; the Batch 19 framing is preserved in its docstring. The Batch 19
  fail-before `test_two_demuxers_disagreeing_by_one_frame_share_a_cache_key` was
  split: `MiscountingDemuxer` monkeypatches `probe_video`'s *return value*
  without touching the build, so it now models the R27 **residual** (D10 —
  same version string, different count), while a new companion test shows a real
  demuxer *version* change moves the cache key.

## Migration

| | Bump? | |
| --- | --- | --- |
| `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` / OpenAPI / `schema_out` | **no** | `ModelInfo.version` stays `str`; `BenchmarkRun.decode_identity` and `CalibrationWitness.decode_identity` are already `str \| None` and store the string given, not a recomputation — an old witness written with `"c20a064a"` still verifies |
| `STUB_ENGINE_VERSION` `.d.` | **yes** | `c20a064a` → `a97ccb68` |
| cache generation | **yes** | one-time full miss per file, no data loss — Batch 9 §6.2 |
| benchmark incompatibility handling | **none** | a run just before the fix (`"c20a064a"`) and one just after (`"a97ccb68"`) do not compare — `regression_gate` refuses (two recorded, different); self-healing. Pre-Batch-11 rows (`None`) still get a note. Executed: `test_an_old_benchmark_row_still_loads_and_a_cross_boundary_comparison_refuses` |

Historical identity docs keep `c20a064a` (a measurement true when written).
`docs/HEADLINE_AND_FACE_SEMANTICS.md` and `docs/FRAME_COUNT_REPRODUCIBILITY.md`,
which cited it as *current*, get a one-line "superseded by B20" note.

## Guarantees gained

1. A cached verdict cannot cross a libavformat / libswscale / libavutil
   difference — every FFmpeg library the video path links.
2. `regression_gate` refuses a cross-FFmpeg-library benchmark comparison.
3. A `CalibrationWitness` binds all four FFmpeg library versions.
4. The demuxer that decides which frames are scored (R27) is inside every
   identity that matters.
5. Batch 19's tripwire fired and was honoured.

## Guarantees NOT gained (residuals, stated)

- **D8** — libswscale's SIMD path is not in the identity; B8 E6 measured
  OpenCV's own kernels CPU-independent, not swscale's. The CPU-field exclusion
  is inherited on that assumption; `test_the_cpu_fields_are_still_excluded_from_
  the_identity` pins the current behaviour.
- **D9** — `cv2.__version__` (`4.8.1`) drops the opencv-python wheel's 4th
  component (`.78`). Capturing the FFmpeg libs directly moots this for the
  decode path; OpenCV's own imgproc changing between wheel patches with no FFmpeg
  change is the uncovered sliver (low-risk per E6).
- **D10** — two FFmpeg builds with identical version strings but different
  compilation still share an identity. A version string cannot see this; a hash
  of the linked binaries could, and is out of scope.
- **R6-a / R6-b** (uniform frame sampling; no `frame_count` validation) —
  untouched; still the video-face-stage acceptance criterion.
- **No back-fill.** Historical artifacts keep their coarse `.d.` value.

## Still open after Batch 20

R9 (by design), R10 (GPU/training), **R13** (310×, SCRFD acceptance criterion),
R16 (by design), **R18's capability half**, **R23's correctness halves**,
**R24**, **R25**, **R26**, **R6-a/R6-b** (video-face-stage criterion), and R27's
residuals **D8–D10** (finer than a version string reaches; no measured
occurrence). **R12, R17, R19, R20, R21, R22, R27 are closed; R6 is PARTIALLY
CLOSED.**

Decision:

- **R27 — CLOSED**, with the residual list attached, the same way R12 was closed
  with its "two decoders still disagree" caveat. Exact reason: `decode_identity`
  now covers every FFmpeg library on the video decode path; all four protections
  separate a demuxer/scaler difference (executed); the fix is a cache generation
  with no schema or contract change. libavformat alone was **not** sufficient —
  libswscale is on the pixel path (E4) and libavutil is the shared base.
- **D8–D10 — recorded, pinned where possible, not closed.**

---

# Batch 22 resolutions — R25 closed (crop-geometry identity completion)

Plan, adversarial review and proof strategy:
[`docs/CROP_IDENTITY.md`](docs/CROP_IDENTITY.md) §15,
[`docs/CROP_IDENTITY_B22_PLAN.md`](docs/CROP_IDENTITY_B22_PLAN.md),
[`docs/CROP_IDENTITY_B22_REVIEW.md`](docs/CROP_IDENTITY_B22_REVIEW.md). Study:
`tests/crop_identity_study.py`. Proof: `tests/test_crop_identity.py` (12 tests) —
now **49 suites**.

**Output-identical. One cache generation. No `CONTRACT_VERSION` /
`RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` / OpenAPI / `schema_out`
change** — each asserted by `test_the_migration_moved_only_what_it_should` and
`test_the_shipped_pixels_did_not_move`.

## The three knobs, and where each went

Batch 15 recorded R25 as three preprocessing inputs that decide the model's
pixels while sitting outside every identity string. Batch 22 put them in:

| Knob | Home | Mechanism | Closure strength |
| --- | --- | --- | --- |
| `ModelSpec.input_size` | **weights identity** | folded into the `-w` fingerprint dict (`{**branch_weights, "input_size": ...}`), **not** a new `-i` token — `weights_identity` format byte-unchanged and `parse_model_version` untouched | **strong** — `prepare_image` / `extract_frames` read exactly this attribute, so the value that sizes the tensor and the value in the hash are the same field |
| interpolation rule | **face-selection identity** | new `FaceSelectionPolicy.interpolation_rule` = `"area-if-either-axis-exceeds-target-else-linear"` — the **current `or` behaviour, verbatim** | **label** — `resize` does not read the policy; held to its branch by `test_the_interpolation_rule_name_matches_resize` (the `ORDERING_RULE` pattern) |
| clip / centre rule | **face-selection identity** | new `FaceSelectionPolicy.crop_clip_rule` = `"intersect-image"` | **label** — held to `crop_face`'s slice by `test_the_clip_rule_name_matches_crop_face`; value becomes `pad-<fill>` when R23 lands, moving `.f.` once more (the accepted single double-spend) |

Both new fields have no default (`test_the_policy_has_no_defaults` iterates
`fields()`); `default_face_selection()` is the sole constructor. The field
docstrings state the label-vs-threaded limitation.

## Identity migration

| | pre-B22 | B22 |
| --- | --- | --- |
| `.f.` (`default_face_selection().identity`) | `19559387` | `251d22e9` |
| weights sub-identity | `stub-0-w7f45fada` | `stub-0-wacdd9240` |
| model identity | `stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387` | `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9` |
| `STUB_ENGINE_VERSION` | `...f.19559387.d.a97ccb68` | `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9.d.a97ccb68` |
| `calibration` / `gate` / `.d.` | `739c3fec` / `ae232cba` / `a97ccb68` | **unchanged** |

One cache generation (image + video namespaces), one-time full miss per file, no
data loss — precedented by B9/B14/B20. `regression_gate` does not short-circuit
on `effective_model_version` inequality (only `dataset_fingerprint` and
`decode_identity` do), so a pre/post-B22 benchmark pair still compares
numerically — and the numbers are identical under the shipped detector.
`BenchmarkReport.validate()` refuses a mismatched pair. Back-fill refused.
`verify_version_alignment` fails a worker whose `input_size` differs from the
configured literal. Old stored bundles / witnesses / benchmark rows load
unchanged (opaque strings, own-era verification); `parse_model_version` is total
across all six eras.

## The adversarial review, and what it changed

The B22 review (`docs/CROP_IDENTITY_B22_REVIEW.md`, verdict APPROVE WITH CHANGES)
found and this batch fixed:

1. **Two of the four Batch-15 tripwires do not fail after the R25 fix.**
   `test_the_clipping_rule_is_in_no_identity_string` had no identity assertion at
   all; `test_the_interpolation_policy_is_in_no_identity_string` used a fragile
   substring check that a field named `interpolation_rule` does not trip. Both
   were **replaced** (not inverted) with positive two-policy `.identity`-inequality
   tests in `tests/test_crop_identity.py`. The genuine fail-befores were
   `test_input_size_is_in_no_identity_string` and
   `test_two_deployments_can_agree_on_every_identity_and_differ` (both assert an
   identity equality the fix breaks) plus `test_every_policy_field_moves_the_identity`'s
   `names == set(alternatives)` assertion.
2. **The pin inventory undercounted.** `_EXPECTED_EFFECTIVE` in
   `test_branch_b_semantics.py` and `test_headline_and_face_semantics.py`,
   `MODEL_ID` in `test_decode_identity_completeness.py`, and the field-set
   assertion in `test_face_selection_identity.py` were added and updated.
3. **Closure strength is now stated** — knob 1 is functionally threaded, knobs 2
   and 3 are label closures; weaker, documented in the field docstrings and here.
4. **R28 split out** (option C). B22 names the interpolation rule with the current
   `or` behaviour and pins it; the predicate fix is deferred.
5. **Regeneration** — `STUB_ENGINE_VERSION` + docstring, `crop_geometry_study.py`
   §7 superseding banner, `crop_identity_study.py` §4/§5, superseding notes on
   `CROP_GEOMETRY.md`, `FACE_SELECTION_IDENTITY.md`, `BRANCH_B_SEMANTICS.md`,
   `HEADLINE_AND_FACE_SEMANTICS.md`, `FACE_SIZE_POLICY.md`.

## New finding: R28 — `resize`'s interpolation predicate area-filters an enlarged axis

`inference/preprocess.py` — `interp = cv2.INTER_AREA if (h > size or w > size)
else cv2.INTER_LINEAR`. The `or` selects `INTER_AREA` for the whole resize
whenever **either** axis exceeds the target, so a crop wider than `input_size`
but shorter (500x300 -> 384) is area-filtered on the axis it is **enlarging** —
a low-pass on an upsample, in a function whose docstring picks `INTER_AREA`
because forensic signal is high-frequency. Measured: 0.040/255 on the enlarged
axis (`crop_identity_study.py` §3).

**OPEN, DEFERRED.** Fixing the predicate moves shipped pixels -> a benchmark
boundary -> it would forfeit B22's output-identical / no-recalibration guarantee.
Impact is small, content-dependent, rare under Haar, near-zero on the video path,
and there is no trained model or benchmark corpus today. It rides the next batch
that already moves `.f.` (R23/SCRFD or R16), which absorbs its cache generation
and its `interpolation_rule` string change. Recorded here as its own finding, not
folded into R25.

## Recorded residual (not closed)

The `CalibrationWitness` binds preprocessing geometry only **incidentally** — via
the fitted temperature drifting when `input_size` changes the logits — exactly
the shape B11 described for the decoder half of R12. Two fits that
`quantize`-converge to the same temperature under two `input_size`s produce
byte-identical witnesses. First-class binding needs `effective_model_version` in
the digest -> a `CALIBRATION_WITNESS_VERSION` bump -> its own batch.

**Superseded by R31 (Batch 24+).** First-class binding shipped:
`ModelSpec.pre_calibration_identity` (weights + gate + selection, excluding
calibration) is now inside the digest for `CALIBRATION_WITNESS_VERSION` 1.3.0.
See the R31 row above and `docs/R31_DESIGN.md`/`docs/R31_SPEC_RESOLUTION.md`.
This section is left otherwise unedited as the B22-era record.

## Still open after Batch 22

R6-a, R6-b (video-face-stage criterion), R9 (by design), R10 (GPU/training),
**R13** (310x, SCRFD acceptance criterion), R16 (by design; image-path plumbing
unblocked by B16), **R23's correctness halves**, **R24**, **R26**, **R28** (new,
deferred), and R27's residuals **D8-D10**. **R12, R17, R19, R20, R21, R22, R25,
R27 are closed; R6 is PARTIALLY CLOSED.**

Decision:

- **R25 — CLOSED.** All three **image** crop-geometry knobs are in the model
  identity; `input_size` functionally in `weights_identity`, the interpolation and
  clip rules as named label closures in the face-selection identity, each pinned
  by a re-derivation test. Output-identical, one cache generation, no schema /
  contract / witness / OpenAPI change — executed and asserted. **R25 closes the
  image crop path only** — the video path's frame-selection and pooling
  parameters are a separate gap, subsequently opened as **R29** by the Batch 23A
  audit (see the Batch 23 section).
- **R28 — OPEN**, new finding, deferred to the next `.f.`-moving batch. Not folded
  into R25.

---

# Batch 23 resolutions — B22 re-audit (B23A); R29 opened

Batch 23A was a fresh adversarial source-level audit of the shipped B22 tree (not
the plan, not the B22 report). **Verdict: APPROVE WITH CHANGES.** Every B22 claim
was re-derived from source and held: `input_size` genuinely moves
`weights_identity` (measured, 4 sizes → 4 distinct `-w`), `interpolation_rule` and
`crop_clip_rule` genuinely participate in `FaceSelectionPolicy.identity`, the
single `FaceSelectionPolicy` constructor and single `effective_version`
composition site leave no bypass, `STUB_ENGINE_VERSION` matches both engines,
every version helper is total and mutually consistent, the `CalibrationWitness`
digest contains no model-identity input (so stored witnesses still verify), and
`regression_gate` still compares a pre/post-B22 pair numerically. No falsifiable
defect in B22's committed code. B23 (this batch) implements the two blocking
items only.

## L3 — the crop-identity study printed a false conclusion

`tests/crop_identity_study.py` closed every run with
`"R25 confirmed from shipped code: 3 knobs, 0 identity coverage."` — B22 gave all
three knobs identity coverage, and the same run already printed the `R25 IS
CLOSED` banner and the shipped post-B22 identities in §4. **Fixed:** the final
summary now states the closure (3 knobs → 1 in `weights_identity`, 2 label
closures in `FaceSelectionPolicy.identity`), the `§1` knob-1 block no longer
prints `<- IDENTICAL` next to two now-different `weights_identity` values, the
`§1` knob-3 note records that `crop_clip_rule` now separates the two border
behaviours, and `§6` points at R29. The study changes no measurement and no
shipped code.

## R29 — video frame-selection and pooling parameters are outside every identity

**CLOSED in B23 (implementation). Opened by B23A.**

### The gap (as B23A found it)

R25 closed the **image** crop path: the pixels between `faces[0]` and the model
tensor. The **video** path never had a face stage (R18) — it samples frames and
resizes them whole — and the parameters that decide *which* frames the model sees
and *how* they become a verdict reached no identity string:

| Parameter | Where | In an identity? |
| --- | --- | --- |
| `VideoPipeline.n_frames` (`video_pipeline.py:74`, default `DEFAULT_FRAMES = 48`) | how many frames `sample_frame_indices` returns | **no** |
| `VideoPipeline.min_usable_frames` (`video_pipeline.py:76`, default `8`) | the `TOO_FEW_FRAMES` abstention floor (`video_pipeline.py:181`) — a gate, the exact shape of `min_face_px` / R19 | **no** — and not a `MediaCaps` field, so it cannot reach `gate_fingerprint` |
| `topk_indices` `k` (`postprocess.py:548`, default `arr.size // 8`) | pooling depth | **no** |
| `sample_frame_indices`' rule (`preprocess.py:874`), published as `VideoAnalysis.sampling = "uniform"` | frame-selection algorithm | **no** — a label bound to nothing |

**Measured end-to-end** (`tests/video_selection_study.py`, one synthetic clip
whose middle third is high-frequency; the pre-B23 tree gave all rows a
byte-identical `model_version` `…f.251d22e9.d.a97ccb68` — the whole result-cache
key):

```
n_frames= 48  min_usable=  8   verdict=manipulated             p_fake=0.9237
n_frames=  9  min_usable=  8   verdict=manipulated             p_fake=0.9181
pool_divisor 8 -> 4                                             p_fake moves
min_usable 13 on a 12-frame clip  verdict=insufficient_evidence p_fake=None
```

**This is not a reopening of R25.** R25 was scoped (B15) to the three crop-geometry
knobs and that enumeration is complete for the image crop path — B23A found no
fourth image-crop knob outside the identity. R29 is the video analogue of R17
(face selection) and R25 (crop geometry).

**Reachability (pre-B23):** code fork only — `n_frames` / `min_usable_frames` /
`batch_size` were `VideoPipeline.__init__` defaults with a single shipped value,
not wired to `Settings`. Same reachability class as R25's own label closures and
R28, both of which B22 judged worth binding.

### The fix (B23 implementation)

A `VideoSelectionPolicy` frozen dataclass (`inference/preprocess.py`), every field
required, one factory `default_video_selection()` — the B12 `FaceSelectionPolicy`
template, applied a fourth time (after R17, R19, R25):

| Field | Closure | Threaded to | Held by |
| --- | --- | --- | --- |
| `n_frames` (48) | **functional** | `sample_frame_indices` via `VideoPipeline` | it *is* the parameter |
| `min_usable_frames` (8) | **functional** | the `TOO_FEW_FRAMES` gate (`video_pipeline.py`) | the pipeline reads it from the policy |
| `pool_divisor` (8) | **functional** | `k = max(1, n // divisor)`, passed explicitly to `topk_indices` | the pipeline computes `k` from the policy |
| `sampling_rule` (`"uniform"`) | **label** | — (also `VideoAnalysis.sampling`) | `test_the_sampling_rule_name_matches_sample_frame_indices` (re-derivation) |
| `pooling_rule` (`"topk_mean"`) | **label** | — (also `VideoAnalysis.pooling`) | `test_the_pooling_rule_and_divisor_match_the_pipeline` |
| `batch_size` | **excluded** | — | `test_batch_size_is_output_neutral` — two batch sizes → byte-identical bundle |

`selection_identity(face, video) = fingerprint({"face": face.identity, "video":
video.identity})` is the value both pipelines pass to
`ModelSpec.with_face_selection(...)`. `contracts/version.py` is untouched — `.f.`
is still one opaque 8-hex fingerprint — and there is one `STUB_ENGINE_VERSION`
literal, not two. The **image** pipeline carries `VideoSelectionPolicy` without
consuming it, the exact mirror of `VideoPipeline` carrying `FaceSelectionPolicy`,
so `ImagePipeline().model_version == VideoPipeline().model_version` and worker
alignment holds.

The label closures are **stronger than B22's `interpolation_rule`**: B22's label
is not on the wire at all, held only by a re-derivation test; here the label is
*also* the published `VideoAnalysis` value, so "the wire value equals the identity
input" is true by construction.

**Legacy-scalar shim:** `VideoPipeline(n_frames=…, batch_size=…,
min_usable_frames=…)` still works (deprecated). When a scalar is passed it is
reconstructed into the policy via `replace()` *before* the identity is computed,
so the identity is always honest and ~10 study/test call sites need no edit.
`VideoSelectionPolicy.__post_init__` centralises `n_frames >= 1`,
`1 <= min_usable_frames <= n_frames`, `pool_divisor >= 1` — a mild tightening of
the constructor (a `min_usable_frames > n_frames` combination now raises; no
shipped configuration or test used it).

### N4 — the stale `"uniform+adaptive"` default

`contracts/video.py` declared `sampling: str = Field(default="uniform+adaptive")`
and the fixtures emitted it, while the live bundle hard-coded `"uniform"` and
nothing adaptive runs. B23 makes the live value authoritative: `sampling_rule =
"uniform"`, sourced from the policy, and the contract default / `contracts/fixtures.py`
/ `fixtures_out/video_*.json` / `tests/stub_handlers.py` corrected to match. A
`Field(default=)` value change — the prose-stripped OpenAPI structural signature
does not see it (verified: `python -m backend.openapi --update-snapshot` produced
a **zero** diff).

### OR1 — the startup alignment gate

`workers.engine_registry.verify_version_alignment` existed since B9 and was called
only from tests. B23 makes it a deploy-time gate:

- **Worker:** `workers.runner.verify_engine_alignment_at_startup(configured)` runs
  from `run_with_shutdown` before the loop, fail-closed. `DFP_SKIP_VERSION_ALIGNMENT`
  is a logged escape hatch. `workers` may not import `backend` (the dependency
  DAG), so the configured literal arrives as an argument or `$DFP_MODEL_VERSION`;
  with neither, the gate still refuses to start on the two failures visible from
  there (engines that disagree with each other; a malformed version), and
  `test_configured_model_version_matches_the_engines` remains the pre-merge guard
  for the `backend.config` default.
- **API:** `backend.app.lifespan` gains a stdlib-only format check on
  `settings.model_version` (`is_composed` + `is_environment_pinned` + a
  `parse_model_version(...).compose()` round-trip). The API process must stay free
  of cv2 / numpy, so it cannot run the full engine comparison; it rejects the
  historically-real failure (`"1.0.0-dev"`, a bare weights string).
- The **only** worker escape hatch is the environment variable
  `DFP_SKIP_VERSION_ALIGNMENT` (truthy `{"1","true","yes","on"}`, logged `WARNING`
  every boot), read directly from `os.environ`. **No `Settings.skip_version_alignment`
  field** (B23-R3 §1.8): the env-only worker cannot read it, so it would be dead
  config. A wired field returns in **B24/OR1-b** with `backend/worker_main.py`.
- **R33 (B23-R3):** `contracts.version.selection_identity_of` /
  `is_selection_pinned` and `ModelSpec.with_selection` are the honest-named
  readers/binder for the composite `.f.` segment. The pre-B23 names
  (`face_selection_identity_of`, `is_face_selection_pinned`, `with_face_selection`)
  are permanent aliases with identical bodies and a "never compare to
  `default_face_selection().identity`" banner. No parser change. The
  `contracts/version.py` module + `.f.` marker + reader docstrings and the
  `inference/models.py` docstrings name the video half.
- **R35 (B23-R3):** `build_video_analysis`'s `sampling` / `pooling` kwarg defaults
  are held equal to `default_video_selection()` by
  `test_build_video_analysis_defaults_match_the_shipped_policy`.
- **R32 / R36 (post-audit cleanup batch, `docs/B23_FRESH_INDEPENDENT_AUDIT_2.md`):**
  the `VideoSelectionPolicy` docstring now names the full `batch_size`-neutrality
  precondition — (a) `batch_clips` no-pad, (b) row-independent `score()`,
  (c) order/value-preserving reconstruction — and the ONNX/TensorRT/attention
  re-opening trigger, held by the restored `test_the_batch_size_exclusion_note_is_complete`
  (PA11). `VideoAnalysis.sampling` / `.pooling` `Field(default=)` and the three
  `fixtures_out/video_*.json` values are held equal to `default_video_selection()`
  by `test_the_video_analysis_wire_defaults_match_the_shipped_policy` (R36) —
  source-level tying is forbidden (`contracts` is a leaf package). MG5 is now a
  scoped `$defs/VideoAnalysis/properties/{sampling,pooling}/default` JSON check.

### Identity / version migration

| Quantity | Before | After |
| --- | --- | --- |
| `default_face_selection().identity` | `251d22e9` | `251d22e9` — **unchanged** |
| `default_video_selection().identity` | — (new) | `db8e6a0d` |
| `selection_identity(...)` — the `.f.` value | `251d22e9` (face only) | `e629642c` (composite) |
| `weights_identity` / `+c.` / `.g.` / `.d.` | `stub-0-wacdd9240` / `739c3fec` / `ae232cba` / `a97ccb68` | **unchanged** |
| model identity | `…+c.739c3fec.g.ae232cba.f.251d22e9` | `…+c.739c3fec.g.ae232cba.f.e629642c` |
| `STUB_ENGINE_VERSION` | `…f.251d22e9.d.a97ccb68` | `…f.e629642c.d.a97ccb68` |
| `VideoAnalysis.sampling` (wire) | `"uniform"` (code) / `"uniform+adaptive"` (fixtures) | `"uniform"` everywhere |

**One cache generation**, spanning both the image and video content-cache
namespaces (the single-literal architecture cost — the B20 `.d.` precedent, where
a video-only demuxer change regenerated the image namespace). Idempotency keys
regenerate in lockstep. No `CONTRACT_VERSION` (1.0.0) / `RUN_SCHEMA_VERSION`
(1.2.0) / `CALIBRATION_WITNESS_VERSION` (1.2.0) / OpenAPI-structural / `schema_out`
structural move. `schema_out/EvidenceBundle.schema.json` gains a cosmetic
`"default": "uniform"` on `VideoAnalysis.sampling` — expected, not a structural
break.

**Tripwire → closed-state suite:** `tests/test_video_identity_gap.py` (asserted
the gap) is replaced by `tests/test_video_selection_identity.py` (asserts the gap
is closed) — the B19 → B20 `..._still_omits_the_demuxer` →
`..._now_covers_the_demuxer` pattern. New `tests/test_startup_alignment.py` covers
OR1. `tests/run_all.py` now registers 51 suites.

### R28 — stays out of B23

B23 moves the `.f.` segment, which is R28's stated trigger. It is kept out
regardless: R28's fix *moves shipped pixels* (it changes `resize`'s interpolation
choice for crop-straddles-input cases), which would forfeit B23's
output-byte-identical guarantee. R28 rides the SCRFD / R23 batch, which is already
moving pixels. `FaceSelectionPolicy.interpolation_rule` still names the current
`or` behaviour verbatim and its re-derivation test still pins it.

### New findings recorded (not closed)

**R30** (image decode libs outside `decode_identity` — the image analogue of R27;
needs a decode-id value extension = own cache generation), **R31**
(`CalibrationWitness` digest omits the model identity — needs
`CALIBRATION_WITNESS_VERSION` 1.3.0), **N3** (advisory heuristic constants; B23
declined the optional tripwires — no scope expansion), **N5** (no lockfile;
opencv / numpy are ranges), **N6 / N7** (fit-optimiser config and `np.quantile`
method unrecorded), **N8** (video-shaped decode-id on the image path). Each is a
summary-table row above.

**B23-R2 gate / B23-R3 findings.** **R32** (`batch_size` neutrality is
backend-dependent — tracked; the `VideoSelectionPolicy` docstring now names the
full (a)/(b)/(c) precondition + the ONNX/TensorRT/attention trigger, held by
`test_the_batch_size_exclusion_note_is_complete` (PA11) — restored by the
post-audit cleanup batch after being dropped in the R2→R3 transition — plus the
`test_batch_size_is_output_neutral` behavioural tripwire), **R33** (`.f.`
naming/doc drift — **closed in B23-R3** by the `selection_identity_of` /
`is_selection_pinned` / `ModelSpec.with_selection` aliases + docstring updates +
three tripwires, no parser change), **R34** (fail-closed OR1 makes
`STUB_ENGINE_VERSION` moves non-rolling-deployable without coordination —
tracked; OR1 working as intended, runbook line added), **R35**
(`build_video_analysis` hardcoded defaults — **closed in B23-R3** by a
default-equality tripwire), **R36** (`VideoAnalysis` `Field(default=)` + fixture
literals are a free-floating copy of the policy labels — **closed in the
post-audit cleanup batch** by `test_the_video_analysis_wire_defaults_match_the_shipped_policy`;
source-level tying is forbidden by the `contracts` leaf-package rule).
Each is a summary-table row above.

## Still open after Batch 23

R6-a, R6-b, R9 (by design), R10, **R13** (310×, SCRFD acceptance criterion), R16,
**R23's correctness halves**, **R24**, **R26**, **R28** (deferred), **R30**,
**R31** (new), **R32** (backend-dependent, tracked), **R34** (operational
constraint, tracked), and R27's residuals **D8–D10**, plus **N3 / N5 / N6 / N7 /
N8**. **R12, R17, R19, R20, R21, R22, R25, R27, R29, N4, OR1, R33, R35, R36 are
closed; R6 is PARTIALLY CLOSED.**

**Post-audit cleanup batch** (`docs/B23_FRESH_INDEPENDENT_AUDIT_2.md`,
2026-09-09): D1/D2 — restored PA11 (`test_the_batch_size_exclusion_note_is_complete`)
and extended the `VideoSelectionPolicy` docstring + `docs/VIDEO_SELECTION_IDENTITY.md`
§2 with the full R32 (a)/(b)/(c) precondition and the re-opening trigger.
D3 — `backend/config.py` B23 paragraph corrected `.with_face_selection` →
`.with_selection`. R36 — closed by tripwire. MG5 strengthened from a whole-file
substring to a scoped `$defs/VideoAnalysis/properties/{sampling,pooling}/default`
JSON check. No identity / `STUB_ENGINE_VERSION` / parser / schema / OpenAPI /
version-constant movement (verified).

**B24 candidates:** OR1-b (`backend/worker_main.py` composition root + wired
`Settings.skip_version_alignment`); a full rename of `face_selection_identity_of`
→ `selection_identity_of` at call sites (R33); making `build_video_analysis`'s
`sampling` / `pooling` required kwargs (R35); R30; R31.

## Still open after Batch 24 (N5) and R31

**N5 closed** (dependency lockfile). **N9 closed** (undeclared `httpx2`, found
building N5's isolated environment). **R31 closed** (`CalibrationWitness.
pre_calibration_identity`; `CALIBRATION_WITNESS_VERSION` 1.2.0 → 1.3.0). **N8
closed** in this pass — see the corrected row above.

R6-a, R6-b, R9 (by design), R10, **R13** (310×, SCRFD acceptance criterion),
R16, **R18** (partially closed — capability half open), **R23's correctness
halves**, **R24**, **R26**, **R28** (deferred), **R30** (+ residuals D8–D10,
now safely sequenced behind N5's reproducible lock), **R32** (tracked), **R34**
(tracked), **N3** (deferred), **N6 / N7** (open — N6 real but zero live
variability today; N7 resolved as not applicable to `CalibrationWitness`, no
action needed on either half). **R12, R17, R19, R20, R21, R22, R25, R27, R29,
N4, N5, N8, N9, OR1, R31, R33, R35, R36 are closed; R6 and R18 are PARTIALLY
CLOSED.**

**B26 candidates (as of Batch 24/25):** R30-bundle (design complete, see
`docs/B24_R2_DESIGN.md` §3/§5/§6) — this was Batch 26, now done; see below.

## Still open after Batch 26 (R30 + D8/D9)

**R30 closed** (image decode identity completeness — `image_codec_version` /
`image_codec_simd` on `DecodeEnvironment`; `.d.` `a97ccb68` → `c9d4025c`).
**D9 closed** (folded into R30 — `image_codec_version` directly supersedes
the blind spot). **D8 PARTIALLY closed** — the image half (libjpeg-turbo
SIMD) is now a digest-only field; the **video half (libswscale SIMD) stays
open**, unaddressed, out of scope (Batch 8 E6 never measured it, and this
batch touched only the image codec surface). New finding **D12** (parser
silently omits a codec whose build-info line lacks a `(ver ...)` suffix — no
known occurrence, C2 ratification). See the R30 row above and
`docs/R30_BUNDLE_DESIGN.md` / `docs/DECODE_IDENTITY_COMPLETENESS.md`.

R6-a, R6-b, R9 (by design), R10, **R13** (310×, SCRFD acceptance criterion),
R16, **R18** (partially closed — capability half open), **R23's correctness
halves**, **R24**, **R26**, **R28** (deferred), **D8** (video/libswscale half
only — image half closed), **D10** (open — no fix exists, needs a binary
hash), **D12** (new, recorded not fixed), **R32** (tracked), **R34**
(tracked), **N3** (deferred), **N6 / N7** (open — N6 real but zero live
variability today; N7 not applicable). **R12, R17, R19, R20, R21, R22, R25,
R27, R29, R30, D9, N4, N5, N8, N9, OR1, OR1-b, R31, R33, R35, R36 are closed;
R6 and R18 are PARTIALLY CLOSED; D8 is PARTIALLY CLOSED.**
