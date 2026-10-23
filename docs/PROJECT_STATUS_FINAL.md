# Final Project-State Audit

**Date:** 2026-09-18. **Method:** every claim below was re-derived this
session directly from source — `REPRODUCIBILITY_AUDIT.md` (3,601 lines, read
in full: header/scope, the summary table, and every "Still open after Batch
N" trailer through Batch 27), `docs/PROJECT_ROADMAP.md` (231 lines, read in
full), and every completed-batch design/resolution document in `docs/`
(`R23_R28_DESIGN.md`, `R16_R26_DESIGN.md`, `R16_R26_SPEC_RESOLUTION.md`,
`R30_BUNDLE_DESIGN.md`, `OR1B_DESIGN.md`, `OR1B_SPEC_RESOLUTION.md`, plus the
R31/N5/B24 chain already narrated in the audit) — cross-checked against the
actual working tree (`git status`, `git diff --stat`, targeted `git diff` on
the changed production/test files, and a full run of `tests/run_all.py`).
**No source document was accepted at face value**; §7 records the specific
places where a document's own claim did not match what the working tree
actually contains.

**This document does not modify, and was not permitted to modify, any other
file. No production code was written to produce it.**

> **State since this was written (2026-10-23, commit `e18eb1a`):** all the
> work this document calls uncommitted is now committed (18 commits on
> `master`; the repository still has no remote, so the CI workflow has still
> never run). SCRFD Phases 1-3 landed and R13 was measured against SCRFD:
> **no-go**, so the Haar cascade stays the default detector
> (`docs/SCRFD_R13_MEASUREMENT_EVIDENCE.md`). The serving path gained a SQLite
> job queue, per-queue retry policies, the `GET /events/{job_id}` stream and
> per-client rate limiting (single host only). Redis and S3 are still
> skeletons. The rest of this document is unchanged, as a record of its date.

---

## 0. Ground truth established this session

- **Repo:** `C:\Users\sunit\Downloads\Deepfake-Detection\deepfake-platform`.
  Last commit: `8676fdd` ("R31: bind calibration witness to model
  configuration"). Everything described as "Batch 25" (OR1-b), "Batch 26"
  (R30-bundle), "Batch 27" (R16+R26 cheap paths), and the R23/R28
  reachability tests is **uncommitted working-tree state** — real, present,
  and verified below, but not yet in git history. Two sessions have a
  documented history of editing this repository concurrently
  (`R30_BUNDLE_DESIGN.md` §7, `OR1B_SPEC_RESOLUTION.md` §8) — treat file
  mtimes, not narrative "Batch N" labels, as the timeline of record.
- **Scope boundary, unchanged and load-bearing:** the audit's own header
  states *"Serving-path correctness is out of scope except where it feeds a
  benchmark."* Everything this project has built is reproducibility and
  identity-fingerprinting plumbing around a **stub backend**
  (`StubBackend` — deterministic, hash-derived scores). There is no trained
  model, no SCRFD (still a Haar cascade), no video face stage, no frontend,
  and no exercised deployment (the CI workflow has never run — no git remote
  existed until recently).
- **Suite size, confirmed live:** `tests/run_all.py` currently lists **54
  suites** (counted this session; `R23_R28_DESIGN.md` independently counted
  the same number the same day). The most recently *documented* full-suite
  result in the tree is `docs/R30_BUNDLE_DESIGN.md`'s **52 suites, 1292
  tests, exit 0**, taken before the Batch 27 (R16+R26) and R23/R28
  reachability files were added.
- **This session's own full run** (`python tests/run_all.py` against the
  current, uncommitted working tree, via the project's `.venv`): started
  clean, **441+ tests passed with zero failures** through every suite
  checked so far, including contracts/OpenAPI/schema drift, decode
  determinism, video branch provenance, and job-lease safety. The evaluation
  suite (`test_evaluation_operating`) is long-running by design (a
  documented ~5–15 minute bootstrap-resample cost, not a hang); this
  document's production-content claims do not depend on that suite's
  specific timing, only on the counts already confirmed. See the addendum
  at the end of this document once the run completes, if you are reading
  this before that line was added.
- **Every "Stop here — no implementation until explicit approval" gate
  found in a design/resolution document in `docs/` is stale.** See §7 — in
  every one of the three cases checked (R30-bundle, OR1-b, R23/R28
  reachability), the code and tests the document proposed already exist,
  uncommitted, in the working tree, contradicting the document's own closing
  sentence. This is a real, current, self-reinforcing pattern in this
  project's process, not an isolated slip.

---

## 1. Completed findings

Verified either by reading the changed source/test files directly, or by
the file already being unchanged since a genuinely closed prior batch. Each
row states the fix's actual scope, not just the ID's label — several
findings below are compound and only *part* is in this list; the
partially-closed remainder is in §2.

| ID | What closed | Verified by |
|---|---|---|
| R1–R5, R7, R8, R11, R12, R14, R15 | Calibration/version/decode findings from Batches 4–11 | Audit narrative + summary table, unchanged since those batches; no later batch touched them |
| R17 | Face-selection policy + crop margin folded into identity (B12) | Summary table; `FaceSelectionPolicy` present, unchanged surface since |
| R19 | Face gate ranks and reports on the same metric (B14) | Summary table |
| R20 | `face_px` description corrected (B18) | Summary table |
| R21 | `faces[]` uniqueness/consistency validators (B16) | Summary table |
| R22 | Branch B "whole image" documentation corrected (B17) | Summary table |
| R25 | Crop-geometry knobs (`input_size`, interpolation rule, clip rule) folded into identity (B22) | Read `inference/preprocess.py:499-528` this session — `FaceSelectionPolicy` carries `interpolation_rule` and `crop_clip_rule` live |
| R27 | Decode identity covers the full FFmpeg library set for video (B20) | Summary table, unchanged since |
| R29, N4 | Video frame-selection/pooling folded into identity; stale `"uniform+adaptive"` default corrected (B23) | Summary table |
| OR1 | `verify_version_alignment` called at real startup (worker + API) (B23) | Summary table |
| **OR1-b** | `backend/worker_main.py` composition root wires the config-default path into the full alignment check | **Verified live**: `backend/worker_main.py` exists (4,977 bytes, new file), `workers/runner.py` and `backend/config.py` show matching diffs |
| **R30** | Image decode libraries (libjpeg-turbo/libpng/libwebp) joined `decode_identity`; `.d.` moved `a97ccb68`→`c9d4025c` | **Verified live**: `git diff --stat` shows `inference/decode_env.py` (+102/-…) and `backend/config.py` changed; new `tests/test_image_decode_identity.py` present |
| D9 | `cv2.__version__`'s dropped 4th digit — folded into R30 | Same diff as R30 |
| **N5, N9** | Dependency lockfile (`uv.lock`, committed at `8676fdd`); undeclared `httpx2` fixed | Confirmed at `HEAD` — already committed, not just working-tree |
| **R31** | `CalibrationWitness` digest now includes `pre_calibration_identity` (weights+gate+selection); `CALIBRATION_WITNESS_VERSION` 1.2.0→1.3.0 | Confirmed at `HEAD` (`8676fdd` is this commit) |
| R33, R35, R36 | Naming-honesty aliases, `build_video_analysis` kwarg-default tripwire, `VideoAnalysis` wire-default tripwire | Summary table, unchanged since B23/cleanup batch |
| N7 | "`np.quantile` unrecorded" — traced to source and found **not applicable** to `CalibrationWitness`'s actual call graph | Summary table; this is a closed non-issue, not deferred work |
| **R28 — reachability half only** | A regression suite proving R28's mixed-case interpolation defect is reachable through the real, shipped `crop_face()` (not only a scripted array) | **Verified live**, contradicting `R23_R28_DESIGN.md`'s own "no implementation" gate: `tests/test_crop_identity.py` diff (+88 lines) contains all three tests the design doc specified — `test_a_real_crop_produces_the_mixed_case_resize_predicate_objects_to`, `test_the_mixed_case_crop_gets_area_interpolation_on_its_upsampled_axis`, `test_r28_is_documented_as_live_not_hypothetical` — read in full this session. See §7 |

**Not in this list, on purpose:** R16, R18, R23, R26, D8 — each has a closed
part and an open part; listing the ID as "closed" would misstate the
remainder. See §2.

---

## 2. Partially completed findings

| ID | Closed part | Still open part | Verified by |
|---|---|---|---|
| **R6** | `frame_count` amplifier measured and pinned (B19); reproducibility half split out and closed as R27 | Serving-path fix (correct sampling so a demuxer miscount doesn't move ≥50% of scored frames) — this is a **video-face-stage acceptance criterion**, tracked as **R6-a/R6-b** | `docs/FRAME_COUNT_REPRODUCIBILITY.md`, audit row R6 |
| **R16** | Documentation half: `EvidenceBundle.verdict`/`.p_fake` now state the single-face-published policy | Option 2 — publishing per-face evidence for every detected face, not just the largest | `contracts/evidence.py` diff (+14/-2), read this session; audit row R16 |
| **R18** | False claims removed; dangling `track_id` fixed (B13) | Capability gap: a faceless video clip still receives a decided (not abstained) verdict | Audit row R18, unchanged since B13 |
| **R23** | Documentation half fully closed (B15) — `crop_face`'s docstring states all three border behaviours and the occupancy consequence, read in full this session, nothing left to document | Correctness: border-clip distortion, left/top mis-framing, corner scale error (all live today) and occupancy collapse (latent, needs a free-aspect detector) — the **pixel fixes** are all deferred | `inference/preprocess.py:909-963`, read in full this session; `docs/R23_R28_DESIGN.md` §2 (independent re-derivation, same code, same conclusion) |
| **R26** | Tier 1 (docstring states `bbox` is the detector's box, not the scored region) + Tier 2 option (a) (a permanent regression proving the clamp is currently a structural no-op under Haar) | Tier 2 option (b) (loosening `BoundingBox`'s `ge=0` constraint) and Tier 3 (publishing the actually-scored region) | `contracts/common.py` diff (+7 lines), read this session; `tests/test_face_geometry_publication.py` (new file, present) |
| **D8** | Image half (libjpeg-turbo SIMD) — now a digest-only field in `image_codec_simd`, part of R30 | Video half (libswscale SIMD) — no measured occurrence, out of R30's scope, unaddressed | Audit row D8/R30 |
| **R28 — fix half** | (reachability closed, §1) | The interpolation-predicate fix itself (choosing `INTER_AREA`/`INTER_LINEAR` per axis instead of once for both) — proven live-reachable but not fixed | `inference/preprocess.py:229-238` (`resize`), read in full this session, byte-identical to every prior description |

---

## 3. Remaining open findings, with blocker classification

Per the requested five-way blocker taxonomy. Where a finding's actual
blocker does not cleanly match one of the five, that is stated rather than
forced.

| ID | Finding | Blocker | Why (re-derived, not quoted) |
|---|---|---|---|
| **R13** | Face-crop instability, 310× amplification, 25.7% of decisions | **SCRFD** | The fix *is* replacing the Haar cascade — there is no serving-path change that closes this without a new detector. Acceptance criterion already written in the audit. |
| **R16 (Option 2)** | Per-face published evidence | **SCRFD** | Publishing scores for faces beyond the largest is only meaningful once per-face detection is stable; today it would, in `docs/MULTI_FACE_SEMANTICS.md`'s own words, be "mostly a report on the placeholder's eyesight." |
| **R18 (capability half)** | Faceless video clip scored as decided instead of abstained | **video-face-stage** | Needs a real per-frame face-presence signal in the video path, which does not exist — the video path currently scores whole frames, not faces. |
| **R23 (occupancy collapse)** | Long-edge-derived crop window collapses occupancy for free-aspect boxes | **SCRFD** | Structurally unreachable under Haar — re-verified this session: Haar's cascade returns boxes with `\|w−h\| ≤ 1`; a free-aspect box cannot occur without a different detector. |
| **R23 (border-clip, left/top mis-framing, corner-scale — the fix)** | Live pixel distortion at crop borders | **real-model** | Reachable and measured *today*, independent of SCRFD — the reason it's not fixed is that every candidate fix (`docs/CROP_GEOMETRY.md` §9, Options A–F) changes real pixels with no trained model and no benchmark corpus to say whether the change produces a *better* verdict, only a *different* one. |
| **R24** | Video path squashes whole frames onto a square (1.778× on 16:9) | **video-face-stage** | The fix "belongs with" per-frame face detection in video, which doesn't exist yet — squashing whole frames is a symptom of scoring whole frames. |
| **R26 (Tier 2-b, Tier 3)** | Loosening `BoundingBox`'s constraint; publishing the actually-scored region | **SCRFD** | Both are described in `docs/R16_R26_SPEC_RESOLUTION.md` Decision 1 as genuine contract-shape decisions that should wait until a real detector's box format is known — deciding now would be guessing. |
| **R28 (the predicate fix)** | `resize`'s `INTER_AREA`/`INTER_LINEAR` choice applied per-call instead of per-axis | **real-model** | Same reasoning as R23's border-clip fix, and bundled with it in every design document — proven live-reachable this session (§1), but fixing it moves real pixels with nothing to validate the change against. |
| **R6-a / R6-b** | `frame_count`'s serving-path fix (timestamp-fraction sampling + decodable-count cross-check) | **video-face-stage** | Explicitly scoped as a video-face-stage acceptance criterion in `docs/FRAME_COUNT_REPRODUCIBILITY.md` §9. |
| **R10** | GPU/training nondeterminism | **real-model** (and, once a training program exists, **dataset/benchmark availability**) | No models, no training pipeline exists at all yet — this finding cannot even be scoped until Phase 8 starts. |
| **R32** | `batch_size` output-neutrality is backend-dependent | **real-model** | Only re-opens if a static-shape backend (ONNX/TensorRT) or a cross-frame backend (temporal attention pooling) is adopted — none exist; `StubBackend` satisfies the exclusion today. |
| **R34** | Fail-closed OR1 + a `STUB_ENGINE_VERSION` move ⇒ not rolling-deployable without coordination | **production deployment** | This is an operational runbook constraint (a coordinated worker restart, or a temporary env-var escape hatch), not a code defect — it only matters once there is a real deployment to roll. |
| **D10** | Same version string, different build, produces different decode output | **none of the five — an inherent limitation** | No fix exists without a binary hash of the decoder library itself; this is recorded as a permanent residual, not something schedulable against any of the five blockers. |
| **D12** | Codec-build-info parser silently omits a codec whose version line lacks a `(ver ...)` suffix | **none of the five — dormant, no known trigger** | No real-world occurrence has been observed; recorded defensively (the "C2 ratification"), not fixed speculatively. |
| **N3** | Advisory heuristic constants (`confidence_interval`'s 0.03/0.5, `_blur_score`'s /100.0, `resolution_class` edges) outside every identity | **none of the five — deliberate, low-priority deferral** | These are advisory-only fields; the project's own decision is "add a tripwire when one is next edited," not "schedule a fix." |
| **N6** | Fit-optimiser config (`lo`/`hi`/`iterations`) unrecorded | **real-model** (specifically: a second caller of `fit_calibration` that overrides the defaults) | Zero live variability today — `fit_calibration` has exactly one caller, and it never overrides these values, so there is nothing yet for two configurations to diverge from. |
| **R9** | Split-lock quarantine is knowingly leaky | **not a defect — by design** | Reported every run; not blocked on anything because it is not scheduled to close. |

**Dataset/benchmark availability**, called out separately because it did not
map cleanly to any single row above: it is the blocker for the *entire*
Phase 8 (real model) program (§4), and it is a **second, independent**
blocker — not merely "no model" — for validating any of the real-model-gated
pixel fixes above (R23 border-clip, R28), because even a trained model
cannot tell you whether a geometry change *helped* without a benchmark
corpus with known-correct labels to score against. This project has neither
today (`REPRODUCIBILITY_AUDIT.md:78`, `docs/CROP_IDENTITY_B22_PLAN.md:385`,
`docs/EVALUATION_IDENTITY.md:418`, `docs/THRESHOLD_BAND_DESIGN.md:433`, all
independently confirmed by grep this session, all still true).

---

## 4. What is realistically completable before end of October

**Everything currently actionable within this project's standing
constraints is already done or reduced to a trivial commit-and-doc-sync
step.** This is a stronger conclusion than `docs/PROJECT_ROADMAP.md`'s own
"4–5 remaining batches" estimate (written before Batches 25–27 and the R23/R28
reachability work landed) — re-derived, not carried over:

- Batches/phases already executed in the working tree, verified against
  source in §1: OR1-b, R30-bundle, R16+R26 cheap paths, R28 reachability.
- What remains before this line of work is fully closed **procedurally** is
  not engineering — it is:
  1. **Committing the uncommitted working tree** (everything since `8676fdd`)
     as one or more reviewed commits.
  2. **A doc-sync pass** on `REPRODUCIBILITY_AUDIT.md`: the per-batch
     narrative trailer stops at "Still open after Batch 27" (line 3571) —
     there is no matching trailer section for the R23/R28 reachability work,
     even though the summary table's R28 row (line 117) already reflects it.
     This is the same class of gap the project already named and fixed once
     before, for N8 (`docs/PROJECT_ROADMAP.md` §5).
  3. **Fixing the stale "Stop here" gate language** in `R30_BUNDLE_DESIGN.md`,
     `OR1B_SPEC_RESOLUTION.md`, and `R23_R28_DESIGN.md` (§7) so a future
     reader doesn't mistake implemented work for pending work.
  4. Optionally, deciding whether to spend a `.f.`-moving batch on the R23
     border-clip + R28 predicate fix *without* a real model to validate
     against (every design document recommends **not** doing this — shipping
     "on structural argument alone" is exactly the pattern that already
     forced two adversarial-review cycles for the much smaller, non-pixel
     R30 change).
- **All of the above is achievable well before end of October** — none of it
  is engineering in the sense of new code; items 1–3 are process/paperwork,
  and item 4 is a decision, not a task, with a documented recommendation to
  defer it.

**Nothing else that is genuinely "audit work" remains schedulable as a
batch.** N3, N6, R32, R34, D10, D12, R9 are each either by-design,
deliberately deferred pending a trigger that hasn't occurred, or an
inherent limitation with no fix — none of them are "next batch" candidates,
they are permanently-tracked rows.

---

## 5. What requires a new engineering program, not additional batches

Each of these is named in `docs/PROJECT_ROADMAP.md` §7 as its own phase, and
this session's re-derivation found nothing that shrinks any of them to
batch scale:

1. **SCRFD integration** (re-opens R13, R23's occupancy half, R28 if still
   deferred, parts of R26). Not a finding — a new detector dependency that
   invalidates a large fraction of the current geometry/identity test
   suite's assumptions (Haar's square-box guarantee is load-bearing in
   several tests read this session).
2. **Video face stage** (closes R18's capability half, R24, R6-a/R6-b). A new
   architecture component — per-frame face detection in video — that does
   not exist in any form today.
3. **Real model** (re-validates R13's amplification factor, the
   calibration-witness findings, and every severity number in this audit —
   all of it was measured against a hash-derived stub). Requires dataset
   acquisition/licensing and training infrastructure that this project has
   never touched.
4. **Production surface** (frontend, an exercised deployment — Redis and
   real storage under load, the CI workflow's actual first run — and
   monitoring). None of this exists in any form; the CI workflow file exists
   but has never executed.

**Honest sizing, unchanged from the roadmap's own §3b/§3c because nothing
found this session weakens it:** none of these four programs is estimable
in engineering days from what is in this repository — each depends on facts
not established here (GPU/dataset access, whether SCRFD is pretrained or
trained in-house, team size, infrastructure budget).

---

## 6. Where audit-driven work ends and product-development work begins

This is the single sharpest line this session's re-derivation can draw, and
it is a **process** line, not a **finding-count** line:

> **Audit-driven work is everything closeable by reading code, writing a
> design doc, running an adversarial review, and shipping a
> reproducibility/identity/documentation fix that changes zero or one
> version-axis literal — without needing anything this project doesn't
> already have.** Every batch from B4 through the R23/R28 reachability tests
> fits this shape, including the three most recent ones verified live in
> §1.
>
> **Product-development work is everything that requires an asset, a
> component, or an infrastructure element this project does not currently
> possess at all** — a real detector (SCRFD), a real architectural stage
> (video face detection), a real trained model, a real dataset/benchmark
> corpus, or a real deployment target. These are not on the audit's own
> critical path *because the audit's method cannot produce them* — reading
> code and writing tests does not create a dataset or train a model.

Concretely: **the line falls exactly where §4 ends and §5 begins.** Every
open finding in §3 whose blocker is SCRFD, video-face-stage, real-model, or
dataset/benchmark-availability is, definitionally, on the far side of that
line — re-scoping it as "the next batch" would misrepresent what kind of
work it actually is, which is precisely the failure mode
`docs/PROJECT_ROADMAP.md` §0 already named ("two of the four goals... are
not the same size of problem"). This session's re-derivation confirms that
framing rather than revising it — the only correction this audit makes to
the standing picture is procedural (§7), not substantive.

---

## 7. Process finding: this audit's own documents are running ahead of their own gates

Found by cross-checking, not assumed. In **three separate** completed-batch
design/resolution documents, the document's own final line says
implementation has not happened yet — and in all three cases, it already
has, in the same working tree, as of this session:

| Document | Document's own closing line | What is actually in the working tree |
|---|---|---|
| `docs/R30_BUNDLE_DESIGN.md` | *"Stop here — no implementation until explicit approval, per instruction."* | `inference/decode_env.py` (+102/-… lines) and `backend/config.py` (+52/-… lines) already changed; `tests/test_image_decode_identity.py` already exists |
| `docs/OR1B_SPEC_RESOLUTION.md` | *"Stopping here per instruction — no code has been written or edited. Waiting for approval to proceed to implementation."* | `backend/worker_main.py` already exists (new file); `tests/test_worker_main.py` already exists |
| `docs/R23_R28_DESIGN.md` | *"Stop here — no implementation until explicit approval, per instruction."* | All three of its own recommended tests (`test_a_real_crop_produces_the_mixed_case_resize_predicate_objects_to`, `test_the_mixed_case_crop_gets_area_interpolation_on_its_upsampled_axis`, `test_r28_is_documented_as_live_not_hypothetical`) already exist in `tests/test_crop_identity.py`, read in full this session |

This is not evidence that the work is wrong — every one of the three was
independently verified in §1 against the actual diff and, for the ones this
session's test run reached, is passing. It is evidence that **the
design-doc "gate" text is boilerplate that stops getting updated once
approval is given and code is written**, three times in a row across three
different findings. Left uncorrected, a future reader (human or another
session) re-reading these documents cold — exactly the instruction this
audit itself was given — would wrongly conclude real, shipped work is still
pending, and could re-propose or re-review it. Recommended (not
performed, per this task's constraints): the next session that touches any
of these three documents should update its closing section to state what
actually shipped, the same correction `docs/PROJECT_ROADMAP.md` §5 already
made once for N8's stale audit-table row.

---

## 8. Test-suite verification (this session, live)

`python tests/run_all.py` was run against the current, uncommitted working
tree via the project's own `.venv`. As of this document being written: **441
tests passed across every suite reached so far (contracts, OpenAPI drift,
decode determinism and identity, crop identity, video branch provenance,
job-lease safety, and more), zero failures observed.** The run had not yet
reached the long-running `test_evaluation_operating` suite (a documented
~5–15 minute bootstrap-resample cost, not a defect) at the time this
document was finalized.

**Addendum — final result:** the run completed after this document was
drafted. **`OK - 54 suites, 1327 tests passed`, exit code 0.** This is
higher than any suite/test count recorded in any document in `docs/`
(the highest documented figure, in `docs/R30_BUNDLE_DESIGN.md`, was 52
suites / 1292 tests, taken before the Batch 27 and R23/R28-reachability
test files existed) — consistent with, and stronger confirmation of, this
document's §1 finding that work beyond what any single document claims is
already present and green in the working tree. One stderr line,
`libpng error: IDA[00]: invalid chunk type`, appeared during the run; this
is C-library stderr noise from a deliberate malformed-PNG decode-rejection
test (`test_malformed_video_is_rejected`-class coverage), not a Python test
failure — the process exit code and the harness's own `OK`/pass-count line
are the authoritative result, and both are clean.
