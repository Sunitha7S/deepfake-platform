# Repository-wide reproducibility audit — post-B23 · OPEN-finding ranking + B24 recommendation

**Date:** 2026-09-09. **Type:** design recommendation only — no code changes.
**Sources re-read:** `REPRODUCIBILITY_AUDIT.md` (full), every row marked OPEN /
tracked / deferred, the B23 chain (`BATCH_23_R3_DESIGN.md`,
`BATCH_23_R3_IMPLEMENTATION_REPORT.md`, `B23_R3_INDEPENDENT_AUDIT.md`,
`B23_FRESH_INDEPENDENT_AUDIT_2.md`), `docs/DECODE_IDENTITY_COMPLETENESS.md`,
`docs/VIDEO_SELECTION_IDENTITY.md`, `docs/CROP_GEOMETRY.md` /
`docs/CROP_IDENTITY.md`, `inference/decode_env.py`, `inference/preprocess.py`,
`contracts/api.py`, `contracts/version.py`, `workers/runner.py`,
`backend/app.py` / `backend/config.py`.

**Standing constraints that gate this analysis** (project memory / every batch
report): no frontend, no Redis, no training, no ConvNeXt, **no SCRFD
integration**, **no ONNX migration**, and no change to `CONTRACT_VERSION`,
`EvidenceBundle`, `ThresholdInfo`, or the OpenAPI snapshot structure.

---

## 1. Live verification of state

Re-derived from source and execution (not read from the audit):

| Quantity | Value | Note |
|---|---|---|
| `STUB_ENGINE_VERSION` | `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.a97ccb68` | unchanged since B23 + cleanup |
| model identity | `…+c.739c3fec.g.ae232cba.f.e629642c` | `.f.` composite `e629642c` = `fingerprint({face:251d22e9, video:db8e6a0d})` |
| `.d.` decode identity | `a97ccb68` | set B20; `DecodeEnvironment.identity` over `{opencv, backend, ffmpeg, system, machine}` |
| `CONTRACT` / `RUN_SCHEMA` / `WITNESS` / `LOCK` | `1.0.0` / `1.2.0` / `1.2.0` / `1.1.0` | unchanged |
| full suite | 51 suites / 1264 tests / 0 failures | post-cleanup |
| `cv2.getBuildInformation()` "Media I/O" | `JPEG: libjpeg-turbo 2.1.3-62` (SIMD: NO), `PNG: 1.6.37`, `WEBP: enc 0x020f`, `ZLib: 1.2.13` | **none of these is in any identity string** — this is R30 |
| `ALLOWED_IMAGE_MIME` | `("image/jpeg", "image/png", "image/webp")` | bounds R30's codec set exactly |
| image decode | `cv2.imdecode(buffer, IMREAD_COLOR)` (`preprocess.py:211`); `.d.` = `decode_identity()` (`image_pipeline.py:106`) | the image path digests a decode identity whose codec component is `cv2.__version__` only |

---

## 2. Obsolescence check — did B22 / B23 retire any OPEN finding?

**No OPEN finding became obsolete.** B22 closed R25; B23 + R2/R3 + cleanup closed
R29, N4, OR1, R33, R35, R36. Those batches *created* R28, R30, R31, R32, R34,
N3, N5–N8. The pre-existing open set — **R13, R16, R18-capability, R23-correctness,
R24, R26, R6-a/R6-b, R9, R10** — is untouched by B22/B23 (identity work is
orthogonal to the pixel/geometry defects).

Two deferral conditions **drifted** and should be corrected in the audit trail
(not a batch — a one-line edit each):

- **R28** is recorded as riding "the next `.f.`-moving batch." B23 moved `.f.`
  (`251d22e9` → `e629642c`) and *deliberately excluded R28* because R28 moves
  shipped **pixels** and B23 was output-identical. The trigger is therefore
  "the next **pixel-moving** batch (SCRFD / R23 / R16)", not "the next `.f.`
  batch." `REPRODUCIBILITY_AUDIT.md:117` already says "Rides the next
  `.f.`-moving batch" — reword to pixel-moving.
- **D9** (`cv2.__version__` drops the wheel's 4th digit) is "mostly mooted by
  capturing the FFmpeg libs directly." B24/R30 (below) captures the **image**
  codec libs directly too, mooting D9 for the image path as well — it should be
  downgraded to "residual, both decode paths capture their libraries directly"
  once B24 lands.

---

## 3. Ranked table — all OPEN findings

Axes scored 1 (low) – 5 (high). **Repro** = reproducibility impact ·
**Cache** = cache-corruption risk (false hit → wrong verdict served) ·
**Deploy** = deployment risk introduced by the *fix* · **Migr** = migration cost
of the fix · **Cx** = implementation complexity. **Actionable** = can be a batch
*now*, within standing constraints.

| Rank | Finding | Repro | Cache | Deploy | Migr | Cx | Actionable now? | Disposition |
|---|---|---|---|---|---|---|---|---|
| — | **R13** — face-crop instability amplifies decoder drift (310×, 25.7% of decisions) | **5** | 3 | 2 | 5 | 5 | ❌ **blocked: SCRFD** | Highest severity in the register. Cannot be a batch — the fix *is* the SCRFD migration, a standing constraint. Acceptance criterion already written (±4 LSB, ≤1% short-edge move, <10× amplification). |
| **1** | **R30** — image decode libraries (libjpeg-turbo / libpng / libwebp) outside `decode_identity` | **4** | **4** | 3 | **2** | 3 | ✅ **yes** | **← B24. See §4.** Live false-hit exposure on the **primary** modality; the exact R27 defect for images; B20 gives a proven migration template. |
| 2 | **R31** — `CalibrationWitness` digest omits `effective_model_version` / `input_size` / gate / `.f.` | 3 | 3 | 3 | 4 | 3 | ✅ yes, but low live exposure | B25 candidate. A temperature fitted for model A verifies for model B if the fits `quantize`-converge. Needs `CALIBRATION_WITNESS_VERSION` 1.3.0 (every stored witness digest changes). **No real model exists to calibrate** → no live exposure; fold N6/N7 in; best run alongside real-model onboarding. |
| 3 | **OR1-b** — `backend/worker_main.py` composition root; full alignment check on the config-default path | **1** | 2 | 3 | **1** | **1** | ✅ yes, **scheduled** | Cheapest open item; explicitly deferred *to B24* by `BATCH_23_R3_DESIGN.md` §1.12. Near-zero reproducibility value (moves no identity); it converts a silent cache-disable on the default path into a loud startup failure. Recommend as a **rider on B24 or a standalone B24.5** — see §4.6. |
| 4 | **N5** — no dependency lockfile; `opencv` / `numpy` are version ranges | 3 | 2 | 3 | 2 | 2 | ✅ yes | `numpy` is in **no** identity; a resolver bump can move `resize` / `normalize` arithmetic silently. Build hygiene. Natural **rider on B24** (same theme: pin the decode/compute environment). |
| 5 | **R24** — video path resizes whole frames onto a square (1.778× squash on 16:9, every video bundle) | 3 | 1 | 2 | 4 | 4 | ❌ **blocked: video face stage** | Deterministic distortion → a correctness/modelling defect, not a reproducibility one (no cache divergence). A video face stage replaces the code path; fixing in isolation is thrown away. Pinned by `LetterboxResize` test. |
| 6 | **R23 correctness halves** — border clipping (13% stretch, live with square Haar boxes) + occupancy collapse (SCRFD-armed) | 2 | 1 | 2 | 4 | 4 | ❌ **blocked: SCRFD** | Deterministic; correctness not reproducibility. Every fix option moves pixels today and cannot be validated without a real model. Documented + tripwired in B15. |
| 7 | **R6-a / R6-b** — uniform frame sampling on an unvalidated `frame_count`; no decodable-count cross-check | 3 | 2 | 1 | 5 | 5 | ❌ **blocked: video face stage** | The remaining half of R6. Fix = timestamp-fraction sampling + a decodable-count cross-check = **video-face-stage acceptance criteria** (`docs/FRAME_COUNT_REPRODUCIBILITY.md` §9). `frame_count` is in no cache key directly; impact is indirect via sampled frames. |
| 8 | **R28** — `resize`'s `INTER_AREA if h>size or w>size` low-passes an axis it enlarges (0.040/255) | 2 | 2 | 2 | 2 | 2 | ⚠️ rides a **pixel-moving** batch | Fix moves shipped pixels → forfeits output-identical. Small, content-dependent, rare under Haar, ~0 on video. Named verbatim in `FaceSelectionPolicy.interpolation_rule` + re-derivation test. Rides SCRFD / R23 / R16. |
| 9 | **R18 capability half** — a faceless clip still receives a *decided* verdict | 2 | 1 | 1 | 4 | 4 | ⚠️ needs a video face stage | To distinguish a legitimately faceless clip from the image path's incoherent `faces=[]` needs a `face_stage` signal (also blocks tightening the R21 headline rule). Pinned by a test in B13. |
| 10 | **R16** — one face's verdict is published as the media's | 1 | 1 | 1 | 2 *(prose)* / 5 *(per-face)* | 2 / 5 | ⚠️ product decision | Not a determinism defect (R14 made the choice reproducible). Per-face verdicts touch `EvidenceBundle` (**standing constraint**). The cheap path — state the single-subject assumption in the `EvidenceBundle` / `FaceTrack` description string (R20/R22 pattern) — is a valid small batch but Low value. |
| 11 | **R26** — the published `bbox` is not the region the model scored; `face_track` clamps where `crop_face` does not | 1 | 1 | 1 | 3 | 3 | ⚠️ partly SCRFD | Forensic/contract. The clamp divergence is unreachable under Haar, active the moment a detector regresses a box across the frame edge (SCRFD). Adding a crop-window field touches `EvidenceBundle` (constraint). |
| 12 | **R32** — `batch_size` output-neutrality is backend-dependent | 2 | 2 | 1 | 3 | 3 | ❌ re-opens with ONNX / attention | **Not a defect today.** Re-opens only with a static-shape backend (ONNX — constraint) or `pooling_rule="attention"`. Fully documented + double-tripwired post-cleanup (`test_the_batch_size_exclusion_note_is_complete` + `test_batch_size_is_output_neutral`). Tracked, no action. |
| 13 | **R34** — fail-closed OR1 + a `STUB_ENGINE_VERSION` move ⇒ not rolling-deployable without coordination | 1 | 1 | 3 | 1 | 1 | tracked (mitigated) | OR1 working as intended (fail loud). Mitigation is operational: coordinated worker restart at cutover, or `DFP_SKIP_VERSION_ALIGNMENT=true` for the window. Runbook line shipped. **Applies to B24** (which moves the `.d.` literal). |
| 14 | **D8** — libswscale SIMD path not in the identity | 2 | 2 | 2 | 2 | 3 | ✅ pair with B24 | Inherited from B8-E6's CPU-independence result on OpenCV's own kernels, not swscale's. No measured occurrence. Fold the analogous image question (libjpeg-turbo SIMD: currently `NO`) into B24. |
| 15 | **D9** — `cv2.__version__` drops the opencv-python wheel's 4th component | 2 | 2 | 2 | 2 | 2 | ✅ mooted further by B24 | Already "mostly mooted" by B20. B24 capturing image codec libs directly moots it for the image path too. Downgrade after B24. |
| 16 | **D10** — two builds, identical version strings, different compilation | 2 | 2 | 1 | 3 | 4 | ❌ undetectable by a version string | A hash of the linked binaries could see it; out of scope. Record only. |
| 17 | **N6 / N7** — fit-optimiser config + `np.quantile` method unrecorded | 2 | 1 | 1 | 3 | 2 | ✅ pair with R31 | In no witness or identity. Fold into the R31 witness-hardening batch. |
| 18 | **N8** — video-shaped `decode_identity` on the image path | 1 | 1 | 1 | 2 | 2 | ✅ **resolved by B24** | The image `.d.` carries an FFmpeg component meaningless for `imdecode`. B24 gives the image path its own codec component; a design sub-decision (§4.3) either narrows or closes N8. |
| 19 | **N3** — advisory heuristic constants outside every identity | 1 | 1 | 1 | 1 | 1 | defer | `confidence_interval` `0.03`/`0.5`, `_blur_score` `/100.0`, `resolution_class` edges. Advisory-only fields — deliberately out. Add tripwires when one is next edited. |
| 20 | **R9** — quarantined items make a lock knowingly leaky | 1 | 1 | 1 | — | — | by design | Reported every run. Accepted, not open work. |
| 21 | **R10** — GPU / training nondeterminism | — | — | — | 5 | 5 | ❌ no models yet | Deferred until there is a training pipeline. |

### Ranking rationale

Reproducibility impact dominates the ordering (per the brief), then
cache-corruption risk, then actionability. **R13 is the single most severe open
finding** but is structurally un-batchable — its fix is the forbidden SCRFD
migration. Among *actionable* findings, **R30 is the only one with live
cache-corruption exposure** (a false cache hit that serves a wrong verdict),
and it sits on the primary modality with the cheapest migration. Everything
below R30 is either blocked by a standing constraint, not a defect today,
build-hygiene, or a documentation/forensic clarification.

---

## 4. Recommended next batch — **B24: image decode-identity completeness (R30)**

### 4.1 Why R30 comes before every other actionable finding

1. **It is the only OPEN finding with live cache-corruption exposure that a
   batch can close.** Two hosts whose opencv-python wheels bundle different
   libjpeg-turbo / libpng / libwebp builds decode the same JPEG/PNG/WebP to
   different pixels, compute a different `p_fake` (the stub hashes pixels, so
   this is *demonstrable now* — the same argument that closed R27/B20 and
   R12/B9), and **share a cache key** — a false hit serving a wrong verdict.
   This is R27, verbatim, for the image path.
2. **It is on the primary modality.** B20 closed the video analogue; the image
   path — the platform's main use case — was left with `cv2.__version__` as the
   only proxy for three separate codec libraries.
3. **The migration is proven and cheap.** B20 did exactly this shape for FFmpeg:
   grow one `DecodeEnvironment` field to name every linked library, one `.d.`
   cache generation, **no** `CONTRACT` / `RUN_SCHEMA` / `WITNESS` / OpenAPI bump
   (witness digests are computed over the *stored* decode string, not
   recomputed; old `BenchmarkRun` rows load; cross-boundary comparison correctly
   refuses). `tests/test_decode_identity_completeness.py` is a line-for-line
   template.
4. **It unblocks nothing else, but it retires N8 and further moots D9**, and it
   is the natural host for the N5 (lockfile) and D8 (image SIMD) riders — one
   "pin the decode environment" batch.
5. **Every higher-severity finding is blocked.** R13 / R23 / R24 / R6 need
   SCRFD or a video face stage (standing constraints). R31 has no live exposure
   (no model to calibrate) and a heavier migration (witness version bump).
   OR1-b has ~zero reproducibility value.

### 4.2 Exact scope

**In:**

- `inference/decode_env.py`:
  - Parse the **"Media I/O"** block of `cv2.getBuildInformation()` for the
    codecs behind `ALLOWED_IMAGE_MIME` — **`JPEG` (libjpeg-turbo version + its
    `SIMD Support:` flag), `PNG` (libpng version), `WEBP` (version), `ZLib`
    (zlib version — backs PNG's inflate)**. Define `_IMAGE_CODEC_LIBS` parallel
    to `_VIDEO_FFMPEG_LIBS`, with a regex over the Media I/O section.
  - Add one field `image_codec_version: str | None` to `DecodeEnvironment`
    (parallel to `ffmpeg_version`; `None` in the API process, where
    `getBuildInformation` is unavailable — the existing guard already handles
    this). Update `to_dict`, `differences`, `describe`, `_FIELDS` list, and the
    field-list test.
  - Add `"image_codec": self.image_codec_version` to the `identity` fingerprint
    dict → **`DecodeEnvironment.identity` (`.d.`) regenerates.** Do **not** add
    it to `digest` separately (it is already covered transitively).
  - `_build_facts()` returns the extra parsed string alongside the FFmpeg one.
- `backend/config.py`: `STUB_ENGINE_VERSION` `.d.` literal `a97ccb68` →
  `<new>`; extend the docstring with a B24 paragraph (B20/B23 pattern).
- Docs: `docs/DECODE_IDENTITY_COMPLETENESS.md` gains an image-path section;
  `REPRODUCIBILITY_AUDIT.md` gains a **Batch 24** section, R30 → CLOSED,
  N8 → narrowed/closed (§4.3), D8/D9 dispositioned, summary-table rows.
- Study: extend `tests/decode_identity_completeness_study.py` (or a new
  `tests/image_decode_identity_study.py`) with the before/after membership
  table and the "one identity, two libjpeg-turbo builds, two `p_fake`s"
  demonstration.
- Tests: new suite or extension `S8-B24 image decode identity completeness`,
  mirroring every test in the B20 suite (§4.7); update the `.d.a97ccb68` literal
  pins (§4.5).

**Out (explicitly):**

- No SCRFD, no ONNX, no video face stage, no geometry change (R13/R23/R24/R28
  untouched).
- No `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` /
  `LOCK_VERSION` bump.
- No OpenAPI or `schema_out` structural change.
- No model-identity change (`.f.` stays `e629642c`; weights / `+c.` / `.g.`
  unchanged).
- R31 (witness membership) stays open — a decode change does not need it (B20
  precedent).
- OR1-b and N5 are riders to *decide on*, not core scope (§4.6).

### 4.3 The N8 design sub-decision (must be settled in the B24 design, not deferred)

The image path currently digests `ffmpeg_version` (meaningless for `imdecode`).
Two options:

- **(a) One identity string, both components** — the image `.d.` carries the
  FFmpeg libs *and* the new image-codec libs; an FFmpeg-only change still
  fragments the image cache (a **false miss** — compute, not correctness —
  which `docs/RUN_IDENTITY.md` §3 already rules acceptable, and which B20/B23
  already accepted for the reverse direction). Smallest change; matches B20.
  **N8 → "narrowed, not closed."**
- **(b) Media-kind-parameterised identity** — `decode_identity(kind)` so the
  image `.d.` omits FFmpeg and the video `.d.` omits image codecs. Properly
  closes N8; larger surface (`compose_runtime_version` call sites, both
  pipelines, `BenchmarkRun.decode_identity`). One extra cache split (image and
  video `.d.` diverge permanently — already effectively true after this batch).

**Recommendation: (a)** for B24 — proportionate, B20-consistent — and record N8
as narrowed. (b) is a future cleanup if the cache-fragmentation of (a) ever
measures as material (it will not on a single-host deployment).

### 4.4 Required version / identity movement

| Quantity | Before | After |
|---|---|---|
| `DecodeEnvironment.identity` (`.d.`) | `a97ccb68` | `<new>` — **regenerates** (recompute; the fingerprint dict gains `image_codec`) |
| `STUB_ENGINE_VERSION` | `…f.e629642c.d.a97ccb68` | `…f.e629642c.d.<new>` |
| `model_identity_of(STUB_ENGINE_VERSION)` | `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c` | **UNCHANGED** (decode is not part of model identity — B20 design) |
| `.f.` composite / weights / `+c.` / `.g.` | — | **UNCHANGED** |
| `CONTRACT` / `RUN_SCHEMA` / `WITNESS` / `LOCK` | `1.0.0` / `1.2.0` / `1.2.0` / `1.1.0` | **UNCHANGED** |
| OpenAPI snapshot / `schema_out` | — | **UNCHANGED** (decode identity is a runtime string, not on the wire; `ModelInfo.version` is an unconstrained `str`) |

**Cache generations: one.** It spans **both** the image and video content-cache
namespaces — the single-`STUB_ENGINE_VERSION`-literal architecture cost, exactly
as B20's `.d.` move and B23's `.f.` move regenerated both. Idempotency keys
regenerate in lockstep.

### 4.5 Migration cost — **Low** (identical profile to B20)

- **Orphaned cache entries.** Bundles under `…d.a97ccb68` become unreachable;
  they expire under `result_ttl_s` (default 7 days). Storage cost, not
  correctness.
- **Idempotency.** A job whose pre-B24 attempt was keyed on the old runtime
  version is not deduplicated against a post-B24 resubmission → one re-run,
  one-time.
- **`BenchmarkRun` rows** written under the old decode identity still load;
  cross-boundary comparison refuses (correct — this is the point).
- **`CalibrationWitness`.** Stored witnesses verify against the stored decode
  string; unaffected (no `CALIBRATION_WITNESS_VERSION` change).
- **Deploy (R34 applies).** A `STUB_ENGINE_VERSION` move is not rolling-
  deployable without coordination: a coordinated worker restart at cutover, or
  `DFP_SKIP_VERSION_ALIGNMENT=true` held for the rollout window (reduces to
  B20/B22/B23 semantics — brief false misses, never wrong verdicts). Set
  `$DFP_MODEL_VERSION` to the new literal in the worker and API environment
  before rollout. This is already the standing runbook line.
- **No back-fill.** Historical artifacts keep their coarse `.d.` value (B20
  precedent).

### 4.6 The two riders — decide in the B24 design

- **OR1-b** (`backend/worker_main.py` + re-added wired
  `Settings.skip_version_alignment`, passed into
  `run_with_shutdown(configured_model_version=…, skip_version_alignment=…)`).
  Explicitly scheduled *to B24* by `BATCH_23_R3_DESIGN.md` §1.12. `backend →
  workers` and `backend → backend.config` already exist, so no cycle. **Value:**
  the full cryptographic alignment check runs on the config-default path (today
  only the env-var path gets it; the default path gets engine-agreement + format
  only). B24 moves the `.d.` literal, so shipping OR1-b in the same batch means
  the *next* deploy has the stronger gate. **Cost:** ~1 new module, 1 re-added
  Setting, 1 call-site change, ~4 tests; no identity movement.
  **Recommendation: include OR1-b in B24** — it is cheap, thematically adjacent
  (deploy-time identity checking), and B24 is already a `STUB_ENGINE_VERSION`
  move so the deploy-coordination cost is paid once.
- **N5** (dependency lockfile — `uv.lock` / `pip-tools`, pinning `opencv-python`
  and `numpy` exactly). Same theme (pin the decode/compute environment).
  **Recommendation: include N5 in B24** as a build-hygiene rider — it needs no
  identity movement and closes the "a `numpy` resolver bump moves `resize`
  arithmetic with no identity change" gap that the decode identity cannot see.

If the team prefers strict one-finding-per-batch discipline, run OR1-b as
**B24** (it is the smallest, is already scheduled, and needs no cache
generation) and R30 as **B25**. On the brief's stated axis order
(reproducibility first), **R30 is the higher-value batch** and should lead.

### 4.7 Expected fail-before / pass-after tests

**Fail-before** (assert the OPEN state against the pre-B24 tree):

| Test | Pre-B24 behaviour |
|---|---|
| `test_the_decode_identity_omits_the_image_codecs` (new tripwire, B19→B20 style) | **passes** — `decode_identity()`'s preimage does not contain the libjpeg-turbo / libpng / libwebp versions; becomes `…_now_covers_the_image_codecs` after |
| probe: two `getBuildInformation` Media I/O blocks differing only in `JPEG: libjpeg-turbo 2.1.3-62` vs `3.0.1` | **same** `decode_identity()` today → wrong-cache-hit demonstration |
| `DecodeEnvironment` field-list test | `image_codec_version` **absent** |
| `decode_identity()` == `a97ccb68` | true before; `!=` after |
| `STUB_ENGINE_VERSION.endswith(".d.a97ccb68")` (`test_decode_identity_completeness.py:341`, `test_video_selection_identity.py:341`, `test_crop_identity.py:291`) | true before; updated to `<new>` after |

**Pass-after** (mirror the B20 suite):

| Test | Assertion |
|---|---|
| `test_decode_identity_now_covers_the_image_codecs` | the identity preimage contains each of libjpeg-turbo / libpng / libwebp / zlib versions |
| `test_the_image_codec_libs_list_is_exactly_the_codecs_behind_ALLOWED_IMAGE_MIME` | `_IMAGE_CODEC_LIBS` == the JPEG/PNG/WEBP(/ZLib) set — no TIFF/JP2/EXR (not in the MIME allowlist) |
| `test_a_libjpeg_turbo_difference_now_moves_the_identity` + same for libpng, libwebp | two `DecodeEnvironment`s differing only in one codec version → different `.identity` |
| `test_the_four_protections_now_separate_an_image_codec_difference` | cache key · deployment id · benchmark id · calibration id each go blind → separating (4/4, executed) |
| `test_the_decode_segment_moved_and_the_literal_was_updated` | `.d.` == `<new>`; `_build_facts()` reflects the image libs |
| `test_the_model_identity_did_not_move_in_batch_24` | `model_identity_of(STUB_ENGINE_VERSION)` unchanged (`…f.e629642c`) |
| `test_no_contract_schema_or_witness_version_bump` | `1.0.0` / `1.2.0` / `1.2.0` / `1.1.0` unchanged |
| `test_the_openapi_snapshot_does_not_carry_the_decode_segment` | snapshot structural signature unchanged; `<new>` absent |
| `test_an_old_benchmark_row_still_loads_and_a_cross_boundary_comparison_refuses` | B20 pattern, for the new `.d.` |
| `test_an_old_calibration_witness_still_verifies_after_the_change` | witness computed over the stored decode string, unaffected |
| `test_the_image_path_no_longer_reports_a_meaningless_ffmpeg_component` **or** `test_the_image_identity_carries_both_and_that_is_the_accepted_false_miss` | whichever of N8 option (b)/(a) the design picks |
| `test_batch_24_did_not_touch_the_video_ffmpeg_libs` | `_VIDEO_FFMPEG_LIBS` unchanged; video path regression guard |
| full `tests/run_all.py` | green, suite count +1 (or +0 if extending the B20 suite) |

**Affected files (complete list):**

- `inference/decode_env.py`
- `backend/config.py`
- `tests/test_decode_identity_completeness.py` (extend, or add `tests/test_image_decode_identity.py`)
- `tests/test_crop_identity.py` (`B23_STUB_ENGINE_VERSION` const, `:291` `.d.` literal, the runtime-version list `:299-300`)
- `tests/test_video_selection_identity.py` (`B23_STUB_ENGINE_VERSION` `:77`, `:341` `.endswith`, `:413` `decode="a97ccb68"`)
- `tests/test_crop_geometry.py` (`:585` `.d.` presence — no literal, likely no change; verify)
- `tests/test_startup_alignment.py` (`:76,98,125,211` stale-fixture `.d.a97ccb68` — update for consistency; these are deliberately-wrong-`.f.` fixtures so `.d.` is incidental)
- `tests/test_evaluation_identity.py` (decode constants `UNDER_FFMPEG` / `UNDER_MSMF` are named doubles — verify no `.d.a97ccb68` assertion)
- `tests/test_run_identity.py` (runtime-version round-trip pins — verify)
- `tests/decode_identity_completeness_study.py` (extend) or new study
- `tests/run_all.py` (suite registration + count, if a new suite)
- `docs/DECODE_IDENTITY_COMPLETENESS.md`
- `REPRODUCIBILITY_AUDIT.md`
- *(if OR1-b rider)* `backend/worker_main.py` (new), `backend/config.py`
  (`skip_version_alignment` Setting re-added, wired), `workers/runner.py`
  (call-site already accepts the param), `tests/test_startup_alignment.py`,
  `tests/test_backend.py`
- *(if N5 rider)* `pyproject.toml` / new lockfile, `tests/test_dependencies.py`

---

## 5. One-line summary

**B24 = R30 (image decode-identity completeness), riders OR1-b + N5.** It is the
only OPEN finding that is simultaneously (a) a live cache-corruption risk,
(b) on the primary modality, (c) actionable within the no-SCRFD / no-ONNX
standing constraints, and (d) backed by a proven one-`.d.`-cache-generation
migration template (B20). Every higher-severity finding (R13, R23, R24, R6) is
blocked by the SCRFD / video-face-stage constraint; R31 has no live exposure and
a heavier migration; OR1-b and N5 are cheap enough to ride along.
