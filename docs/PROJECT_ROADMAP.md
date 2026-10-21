# Project-State Audit and Execution Roadmap

**Date:** 2026-09-17. **Basis:** `REPRODUCIBILITY_AUDIT.md` (current, post-R31)
re-read in full; every "open" claim below is the live table state, not a
recollection. **Type:** planning only — no code in this task, per instruction.

---

## 0. The one fact that has to be stated before any numbers

This audit has a **scope boundary stated in its own header** (line 74):
*"Serving-path correctness is out of scope except where it feeds a
benchmark."* Everything this project has built across 24+ batches is
**reproducibility and identity-fingerprinting plumbing around a stub
backend** (`StubBackend` — deterministic, hash-derived scores; `backend:
"stub"` is the only backend `ModelSpec` ships with in this repo). There is:

- **no trained model** (`docs/B24_R2_DESIGN.md` and every calibration-witness
  finding says so explicitly — "no real model exists to calibrate yet"),
- **no SCRFD** (still a Haar cascade — R13's fix "*is* the SCRFD migration,"
  a standing constraint, not a task),
- **no video face stage** (R18/R24/R6-a/R6-b all say their fix "belongs with"
  one that doesn't exist),
- **no frontend** (standing constraint, stated in every candidate-analysis
  document),
- **no exercised deployment** (`.github/workflows/tests.yml` is explicitly
  "unexercised — no git remote existed" until this session's first commit;
  `jobqueue/redis.py` exists but Redis is a standing "no" constraint — unused
  code, not a tested path).

This means **two of the four goals in the request are not the same size of
problem as the other two.** "All audit findings closed" and "production-ready
release-ready" are not points on one line — they are two different projects.
Section 3 below gives both a real answer instead of one number that would
misrepresent the other.

---

## 1. Prioritized table — every finding not fully closed

Read directly from the current `REPRODUCIBILITY_AUDIT.md` summary table
(lines 91–133) and the "Still open after Batch 23" trailer (line 3498,
itself one batch stale — see §5).

| ID | Title | Current status | Effort | Risk | Files likely affected | Depends on |
|---|---|---|---|---|---|---|
| **R30** | Image decode libs (libjpeg-turbo/libpng/libwebp) outside `decode_identity` | **Closed in Batch 26.** `image_codec_version`/`image_codec_simd` on `DecodeEnvironment`; `.d.` `a97ccb68` → `c9d4025c`. See `docs/R30_BUNDLE_DESIGN.md`, `docs/DECODE_IDENTITY_COMPLETENESS.md` | **M** | **Low** — as designed; C2 ratified (parser stays defensive, residual recorded as D12) | `inference/decode_env.py`, `backend/config.py`, new `tests/test_image_decode_identity.py`, 7 collateral test files (6 designed + 1 found live: `tests/test_dependencies.py`, `tests/test_worker_main.py`), `docs/DECODE_IDENTITY_COMPLETENESS.md` | N5 (done) |
| D9 | `cv2.__version__` drops the wheel's 4th digit | **Closed** — folded into R30; `image_codec_version` supersedes the blind spot | (in R30) | (in R30) | (in R30) | R30, N5 |
| D8 | libjpeg-turbo SIMD not in identity | **PARTIALLY closed** — image (libjpeg-turbo) half folded into R30 as a digest-only field; **video (libswscale) half stays open**, unaddressed, out of R30's scope (documented dual scope, `B24_ADVERSARIAL_REVIEW.md`/`B24_R2_DESIGN.md`) | (image half: in R30) | Low | (image half: in R30) | video half: unscheduled |
| D10 | Same version string, different build | Record-only, no fix exists (would need a binary hash) | — | — | none | — |
| D12 | Parser silently omits a codec whose build-info line lacks a `(ver ...)` suffix | **New (Batch 26).** No known real-world occurrence; the C2 ratification — recorded rather than speculatively fixed | — | Low | `inference/decode_env.py` | none |
| **OR1-b** | `backend/worker_main.py` composition root; config-default path gets the full alignment check | Open, scheduled since `BATCH_23_R3_DESIGN.md` §1.12 | **S** | **Low** | new `backend/worker_main.py`, `backend/config.py` (re-add `Settings.skip_version_alignment`), `workers/runner.py` call site | none |
| **N8** | `decode_identity` is video-shaped, applied to the image path too | Table says "Open" but **the disposition was already decided and never written back**: `docs/B24_R2_DESIGN.md` §4 concluded "CLOSED — accepted architectural cost" with exact replacement text | **XS** | **None** | `REPRODUCIBILITY_AUDIT.md` only (one row) | none — this is a **doc-sync gap, not open work** |
| N3 | Advisory heuristic constants outside every identity | Open, deliberately — "add tripwires when one is next edited" | — | — | n/a until triggered | none |
| N6 | Fit-optimiser config (`lo`/`hi`/`iterations`) unrecorded | Open, real, **zero live variability today** (only one caller, never overridden) — correctly deferred per R31's re-derivation | — | — | n/a until a caller parameterizes the fit | a future caller |
| N7 | "`np.quantile` method unrecorded" | **Row already corrected during R31** — traced to source and found not to apply to `CalibrationWitness`'s call graph at all | closed (as "not applicable") | — | — | — |
| R32 | `batch_size` neutrality is backend-dependent | Tracked, conditional — re-opens only with ONNX/TensorRT/attention-pooling, none of which exist | — | — | n/a until triggered | ONNX/attention (not planned) |
| R34 | Fail-closed OR1 + a `STUB_ENGINE_VERSION` move ⇒ not rolling-deployable without coordination | Tracked, operational — a runbook line, not a defect | — | — | none (docs only, done) | applies to any future `.d./.f.`-moving batch |
| R9 | Split-lock quarantine is knowingly leaky | By design, reported every run — not a defect | — | — | none | — |
| **R13** | Face-crop instability, 310× amplification, 25.7% of decisions | **Open — architecture-blocked.** Fix *is* the SCRFD migration (standing constraint). Acceptance criterion already written. | **L, out of this project's current shape** | **High** (real correctness exposure once a real model exists) | `inference/preprocess.py`, detector integration, likely a new module | **SCRFD integration** (not a batch — a project phase) |
| R16 | One face's verdict published as the media's | **Closed for the documentation half, in Batch 27.** `EvidenceBundle.verdict`/`.p_fake` now state the single-face policy. Option 2 (per-face scoring) remains open, SCRFD-gated | (done) / **L** (Option 2, SCRFD-gated) | (done) / architecture-blocked | `contracts/evidence.py` (done) | Option 2: SCRFD |
| R18 | Video path has no face stage (capability half) | Partially closed (false claims fixed in B13); the faceless-clip-scored-as-decided gap remains | **L, architecture-blocked** | High | new video face-stage module | **video face stage** (project phase) |
| R23 | Crop geometry: border-clipping distortion + occupancy collapse | Documentation half closed. **Split:** border-clipping is live today, **not** SCRFD-gated; occupancy collapse **is** SCRFD-gated | Border-clip half: **M** / Occupancy half: **L, blocked** | Medium (moves pixels, no real model to validate against) | `inference/preprocess.py` (`crop_face`), rides same batch as R28 | Border-clip half: none. Occupancy half: SCRFD |
| R24 | Video path squashes whole frames onto a square, 1.778× on 16:9 | Open — "fix belongs with the video face stage" | **L, blocked** | High | video pipeline | video face stage |
| R26 | Published `bbox` ≠ the region actually scored | **Closed for Tier 1 + Tier 2 (option a), in Batch 27.** `FaceTrack`'s docstring states the gap; `tests/test_face_geometry_publication.py` proves the coordinate clamp is a structural no-op under Haar. Tier 2-option-(b) and Tier 3 remain open, deferred pending SCRFD | (done) / **L** (Tier 3, SCRFD-gated) | (done) / architecture-blocked | `contracts/common.py` (`FaceTrack`, done), new test file (done) | Tier 2-b/Tier 3: SCRFD |
| R28 | `resize`'s interpolation predicate area-filters an enlarging axis | **Reachability closed** (`docs/R23_R28_DESIGN.md`, regression suite in `tests/test_crop_identity.py`) — proven live via real `crop_face()` output, not only a scripted array. **Fix still Open — deferred**, "rides the next `.f.`-moving batch (R23/SCRFD or R16)" | Reachability: done (XS) / Fix: **S** (bundled with R23 border-clip) | Reachability: none / Fix: Low | `inference/preprocess.py` (`resize`) — fix only | R23 border-clip batch, or SCRFD, or R16 |
| R10 | GPU/training nondeterminism | Deferred — "no models yet" | **L, architecture-blocked** | Unknown until a model exists | training pipeline (doesn't exist) | a training loop (project phase) |
| R6-a / R6-b | `frame_count` residual — video-face-stage acceptance criteria | Open, blocked on the same video face stage as R18/R24 | **L, blocked** | High | same as R18/R24 | video face stage |

**Cosmetic, zero-priority, explicitly optional** (named in the audit's own
"B24 candidates" trailer, not required for any closure): a full rename of
`face_selection_identity_of` → `selection_identity_of` at call sites (R33 is
already closed via aliases — this is a style nice-to-have); making
`build_video_analysis`'s `sampling`/`pooling` kwargs required (R35 is already
closed via a tripwire — this is defense-in-depth, not a gap).

---

## 2. Findings that could expand scope or create migration/versioning work

Ranked by how likely a "small fix" turns into "another design-review cycle,"
based on this project's own demonstrated pattern (every `.d.`/`.f.`/witness
change so far has needed a design doc, an adversarial review, and a
resolution doc before implementation):

| Finding | Expansion risk | Why |
|---|---|---|
| **R30-bundle** | **Done (Batch 26).** Survived two adversarial passes before implementation (`B24_ADVERSARIAL_REVIEW.md` reversed two of the original design's three ratifications); C2 (parser robustness) was ratified rather than triggering a third review cycle — the parser kept its existing defensive shape and the one residual case was recorded as a new finding (D12) instead of speculatively fixed. | `.d.` moved once (`a97ccb68` → `c9d4025c`), one cache generation, executed. |
| **R16 (if the "real" path is chosen instead of the cheap one)** | **High.** Per-face verdicts touch `EvidenceBundle`, a **frozen, standing-constraint contract**. This alone would need its own design doc, likely a `CONTRACT_VERSION` MAJOR bump, and client-migration planning — a different scale of work than anything done so far. | `CONTRACT_VERSION` MAJOR, `EvidenceBundle` schema, every downstream consumer. |
| **R23 border-clip + R28 bundle** | **Medium-High.** No trained model exists to validate the pixel change against (both findings' own text says so) — the "fix" would be shipped on structural argument alone, exactly the pattern that triggered a full adversarial-review cycle for R30. Expect the same treatment demanded here before merge. | `.f.` moves once; weights/geometry test suite. |
| **R26 (full fix)** | **Low-Medium if scoped to the cheap documentation half; High if scoped to the real coordinate fix** — the real fix depends on SCRFD's actual box format, unknowable until that integration exists. | Contract prose only (cheap) vs. `FaceTrack`/`EvidenceBundle` fields (real). |
| **Anything that eventually consumes `pre_calibration_identity` for real (D2, deferred from R31)** | **Medium.** Explicitly scoped out in `R31_SPEC_RESOLUTION.md` Decision 2 *because* it needs its own new `BenchmarkRun` field and its own `RUN_SCHEMA_VERSION` bump — this is pre-identified, not a surprise, but it is real future work with its own version axis. | `evaluation/benchmark.py`, `RUN_SCHEMA_VERSION`, 7+ collateral test files (already enumerated in the R31 docs). |
| **SCRFD integration itself** | **Very High — this is not "a finding," it's a project phase.** It simultaneously re-opens R13, R23's occupancy half, R28 (if not already closed via the bundle above), parts of R26, and interacts with R18/R24's video face stage. Every identity/geometry test written against Haar-cascade behavior (a large fraction of the 1285-test suite) would need re-deriving, not just re-running. | Touches `inference/preprocess.py` at its core, `weights_identity`, likely a new `.f.` generation, possibly a new detector-asset identity segment not yet designed. |

---

## 3. Critical path — three different bars, answered honestly

### 3a. "All *currently actionable* audit findings closed" (within standing constraints)

This is the bar this project's process can actually move on its own. Critical
path:

```
N8 doc-sync (no dependency)
   │
OR1-b (no dependency)              ──┐
   │                                  ├──▶ "actionable-now" findings closed
R30-bundle (needs N5, done)         ──┘
   │
[optional] R16 cheap path, R26 cheap path  (independent, no dependency)
   │
[optional] R23-border-clip + R28 bundle    (independent, no dependency, but no real model to validate against)
```

None of these block each other — they can run in any order or in parallel
batches. **Estimate: 4–5 remaining batches.**

### 3b. "All audit findings closed" (literal, unconditional)

**Not reachable without lifting the standing constraints.** R13, R18, R24,
R6-a/R6-b, R10, and R23's occupancy half are each blocked on one of: SCRFD
integration, a video face stage, or a real training pipeline — none of which
exist, and none of which this project's batches (design → adversarial review
→ resolution → implementation, each a few files) are shaped to produce. This
is not a scheduling problem, it's a scope problem: closing these needs a
different *kind* of work (ML engineering, model integration, dataset
licensing) than the reproducibility-plumbing work done in every batch to
date. **Not estimable in engineering days from this conversation's
evidence** — it depends on facts not established here (GPU/dataset access,
whether SCRFD is adopted pretrained or trained in-house, team size).

### 3c. "Production-ready" / "Release-ready"

**A larger bar than 3b, not a smaller one.** Beyond every architecture-blocked
finding in 3b, this needs things the standing constraints say don't exist at
all yet: a frontend, an exercised deployment (Redis, real storage backend
under load, the CI workflow actually run once — it has literally never
executed), monitoring, and — the part with the widest blast radius — **once a
real model exists, most of this audit's severity numbers need
re-measurement.** R13's 310× amplification, the crop-margin verdict flips,
the calibration-witness convergence risk — all of it was measured or argued
against a **hash-derived stub backend**. A real model's actual sensitivity to
these axes could be smaller or larger; the audit cannot currently say which
for the one thing that will actually ship.

**Honest answer to "can this be done by end of October": no, not if
"production-ready" means an actual deployed, real-model-serving system.**
That is realistically a multi-month program (model/dataset work alone is
typically weeks-to-months even before integration), not six weeks of batches
in this project's current style. **If "done by end of October" means 3a — all
currently-actionable reproducibility findings closed, the codebase's own
identity/versioning story fully consistent and documented — that is
realistic and probably achievable well before end of October.**

---

## 4. Estimates

| | 3a (actionable-now) | 3b (all findings, literal) | 3c (production/release) |
|---|---|---|---|
| Remaining batches | **4–5** | Not estimable — depends on SCRFD/model/video-face-stage scoping, each of which is its own multi-batch (or multi-*project*) effort | Strictly larger than 3b |
| Remaining engineering days | **~10–15** (S/XS batches ≈1–2 days each with this project's design-review discipline; the R30-bundle M batch ≈3–5 days on its own) | Unknown; SCRFD integration alone is commonly weeks even as a "just integrate a pretrained model" task, before the resulting re-validation of R13/R18/R23/R24/R26 | Unknown, and additive to 3b: + frontend, + deployment, + monitoring, + real-model re-audit |
| Remaining testing effort | **Proportional, ~35–40% of the above** (this codebase's own convention: every batch adds a full fail-before/pass-after/non-movement matrix, updates 5–10 collateral test files, and runs the full suite twice — ad hoc + `uv`-locked) — so roughly **4–6 of the 10–15 days are test/verification work**, not separable from implementation in this project's process | Every existing geometry/identity test written against the Haar cascade or the stub backend would need re-deriving against real detections/scores — this is not incremental test-writing, it's a second pass over a large fraction of the 1285-test suite | Same as 3b, plus integration/load/deployment testing that doesn't exist as a category yet |

---

## 5. Documentation hygiene found during this audit (zero engineering risk, worth doing alongside 3a)

- **N8's summary-table row is stale** (§1) — the disposition was decided in
  `docs/B24_R2_DESIGN.md` §4 and never written back to
  `REPRODUCIBILITY_AUDIT.md`. One-line fix.
- **The per-batch narrative sections stop at "Still open after Batch 23"**
  (line 3498) — B24 (N5) and R31 both landed without a matching "Still open
  after B24" / "...after R31" section the way every prior batch got one. The
  summary table (lines 91–133) *is* current and correct; only the narrative
  trailer is behind. Cosmetic, but worth closing in the same pass as N8 so
  the next reader isn't misled by the last narrative section being two
  batches stale.

---

## 6. Recommended execution order

1. **N8 doc-sync** — zero risk, zero dependency, clears a false "open" signal. **Done** (closed in the OR1-b audit pass).
2. **Doc-hygiene pass** (§5) — bundle with N8 since both are audit-doc-only. **Done.**
3. **OR1-b** — independent, cheap, closes a real (if narrow) live gap; no
   reason to sequence it behind anything. **Done (Batch 25).**
4. **R30-bundle** (R30 + D9, + D8's image half) — the largest remaining
   actionable item; done once N5's lock made its own `.d.` literal
   reproducible. **Done (Batch 26).** C2 (parser robustness) was ratified
   in-place rather than triggering a further design cycle; D8's video half
   and D10 remain open, out of scope, per `docs/R30_BUNDLE_DESIGN.md`.
5. **Decide R16 and R26's cheap vs. real paths explicitly.** **Done (Batch
   27).** Both closed for their cheap/documentation-plus-hardening halves,
   batched together as one sitting per `docs/R16_R26_SPEC_RESOLUTION.md`
   Decision 2. R16 Option 2 and R26 Tier 2-option-(b)/Tier 3 move into the
   SCRFD-scoped program in §3b/§3c, per that document's Decision 1 and
   `docs/R16_R26_DESIGN.md` §2 — not scheduled as "the next batch."
6. **R23-border-clip + R28**, only if a decision is made to spend a
   `.f.`-moving batch on a pixel change with no real model to validate it
   against — reasonable to defer until closer to when SCRFD or a real
   benchmark corpus exists, at which point it can be validated for real
   instead of shipped on structural argument alone (the same tension R30
   itself went through).
7. **Everything else (R13, R18, R24, R6-a/R6-b, R10, R23-occupancy, R26-real,
   R16-real)** is not "next" — it is a separate program (§3b/§3c) that needs
   its own scoping conversation (SCRFD adoption strategy, model/dataset plan,
   frontend/deployment plan) before it can be given batch numbers at all.

---

## 7. Final roadmap

| Phase | Findings included | Expected outcome | Risk |
|---|---|---|---|
| **Phase 1 — Doc sync** ✅ | N8, narrative-trailer hygiene | Audit document fully consistent with actual decided state; no reader misled by a stale "open" | None |
| **Phase 2 — Cheap wins** ✅ (Batch 25) | OR1-b | Config-default worker path gets the full fail-closed alignment check | Low |
| **Phase 3 — Decode completeness** ✅ (Batch 26) | R30, D9, D8 (image half) | Image decode path reaches the same identity completeness the video path has had since B20; `decode-completeness` story closed for both media kinds — **D8's video (libswscale) half and D10 stay open**, out of scope | Low-Medium — one ratification (C2) resolved in-place, no separate review cycle needed |
| **Phase 4 — Product decisions, cheap paths** ✅ (Batch 27) | R16 (cheap), R26 (cheap) | Two Low-severity findings closed via contract-prose caveats + one regression test, no pixel/version movement. **R16 Option 2 and R26 Tier 2-b/Tier 3 remain open**, SCRFD-gated | Low — as executed, zero version-axis movement, zero OpenAPI artifact change |
| **Phase 5 — Geometry cleanup (optional, timing-sensitive)** | R23 border-clip, R28 | **Re-derived (`docs/R23_R28_DESIGN.md`) — the pixel fix remains deferred, confirmed not merely assumed.** R28's reachability gap closed via a zero-pixel-risk regression suite (done). The `.f.`-moving fix for R23 border-clip + R28 stays undone, blocked on real-model work, not SCRFD | Reachability sub-item: done, Low. Fix: Medium — no real model to validate against yet; consider deferring until Phase 6 makes validation possible |
| **— end of what this project's current process can close on its own —** | | **This is the realistic "done by end of October" line.** | |
| **Phase 6 — SCRFD integration** (a program, not a batch) | Re-opens/closes R13, R23 occupancy, R28 (if deferred from Phase 5), parts of R26 | A modern face detector replaces the Haar cascade; every geometry/identity finding gated on it becomes closeable for the first time | High — new dependency, new identity segment, large re-validation surface |
| **Phase 7 — Video face stage** (a program) | R18 capability half, R24, R6-a/R6-b | Video path gets real per-frame face detection instead of whole-frame scoring | High — new architecture, likely its own multi-batch design process |
| **Phase 8 — Real model** (a program) | Re-validates R13's amplification factor, the calibration-witness findings, and effectively re-runs this entire audit's severity claims against real behavior instead of a stub | The audit's numbers become claims about the shipped system instead of about a hash-derived placeholder | High — dataset/GPU/training-time dependent, not schedulable from this conversation |
| **Phase 9 — Production surface** (a program) | Frontend, deployment (Redis, real storage under load), monitoring, first real CI run | An actual deployable, releasable product | High — none of this exists yet in any form |

**Phases 1–5 are realistically completable before end of October.** Phases
6–9 are a different project and should be scoped and resourced separately —
presenting a single end-of-October date across all nine phases would not be
an honest estimate.
