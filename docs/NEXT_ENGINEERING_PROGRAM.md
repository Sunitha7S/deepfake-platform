# Next Engineering Program — Planning Document

**Date:** 2026-09-18. **Basis:** `docs/PROJECT_STATUS_FINAL.md`, re-read in
full for this document. **Type:** planning only — no code, no file other
than this one touched.

**Starting premise, taken as given per instruction:** the audit/remediation
program is complete. Every finding `PROJECT_STATUS_FINAL.md` §1 lists as
closed, and every closed *half* of a compound finding in §2, is out of scope
below — this document does not re-litigate R1–R31 and friends. What follows
is only the work `PROJECT_STATUS_FINAL.md` §3 and §5 already identified as
blocked on something this project does not have: a modern face detector, a
trained model, a video-native face stage, or a real deployment.

---

## 0. Before any program below starts

Not a program — a zero-engineering precondition, carried forward from
`PROJECT_STATUS_FINAL.md` §4 items 1–3, restated here only because starting
a new multi-week program on top of an uncommitted, doc-stale working tree
would make that program harder to review later:

1. Commit the working tree (everything since `6eb1457`).
2. Sync `REPRODUCIBILITY_AUDIT.md`'s narrative trailer past "Still open
   after Batch 27."
3. Correct the three stale "stop, awaiting approval" gate lines
   (`PROJECT_STATUS_FINAL.md` §7).

None of the four programs below need anything else from the audit era. Each
section explains, for its own scope, why.

---

## 1. Program: SCRFD Integration

**Objective.** Replace the Haar cascade (`cv2.CascadeClassifier`,
`inference/preprocess.py`) with a modern detector (SCRFD, or an equivalent
free-aspect-box detector) across both the image and video pipelines, so
that findings whose blocker is literally "Haar cannot do this" become
closeable for the first time.

**What this closes, from `PROJECT_STATUS_FINAL.md` §3:** R13 (310×
amplification — re-measurable, likely reducible, against a stable detector),
R23's occupancy-collapse half (needs a free-aspect box), and it removes the
detector-side half of the blocker for R26 Tier 2-b/Tier 3 and R16 Option 2
(both still need their own contract-shape decisions afterward — SCRFD
removes the *technical* blocker, not the decision). R28 and R23's
border-clip half stay blocked on real-model work even after SCRFD lands
(§3's own distinction: those are pixel-fix questions, not detector
questions) — bundling them into this program anyway is cheap once the
detector-swap batch is already moving `.f.` and touching
`inference/preprocess.py`.

**Prerequisites:**
- A concrete SCRFD asset decision: pretrained ONNX weights from a known,
  licensable source, and a runtime (`onnxruntime` is the natural choice —
  no GPU-training dependency, CPU inference is viable for a detector this
  small).
- A read of `docs/CROP_GEOMETRY.md` §9 and `docs/R23_R28_DESIGN.md` before
  writing any code — both already contain the geometry-option analysis
  (Options A–F) this program will need.
- The §0 housekeeping done first, so the new work's own diff is legible
  against a clean baseline.

**Estimated complexity: Large.** Not a batch — the audit's own repeated
finding is that Haar's `|w−h| ≤ 1` square-box guarantee is load-bearing in a
large fraction of the 1327-test suite (`tests/test_crop_geometry.py`,
`tests/test_crop_identity.py`, `tests/test_face_selection_identity.py`, and
others). Swapping the detector doesn't just add code, it invalidates the
*assumptions* several dozen existing tests were written against, each of
which needs re-deriving, not just re-running.

**Dependencies.** None on the other three programs listed here — this is
the one program that can start immediately with nothing else in place.
Sequencing note: doing this **before** Video Face Stage is strongly
preferred (a video face stage built on Haar would need redoing once a real
detector lands) and **before** attempting any R23/R28 pixel fix (fixing
crop geometry against a detector about to be replaced wastes the fix).

**Expected files/modules affected:**
- `inference/preprocess.py` — `detect_faces`, `crop_face`, `FaceSelectionPolicy` (new detector-identity field), possibly `resize`/interpolation if R28 is bundled in.
- `inference/models.py` — `ModelSpec`/`weights_identity` gains a detector-asset fingerprint segment (new weights file, new hash).
- `contracts/common.py` / `contracts/evidence.py` — only if R26's `BoundingBox` constraint changes (a genuine contract-shape decision, not automatic).
- New dependency in `pyproject.toml` + `uv.lock` (an ONNX runtime and the SCRFD weights asset — licensing and storage need a decision, not assumed here).
- Test surface: `tests/test_crop_geometry.py`, `tests/test_crop_identity.py`, `tests/test_face_selection_identity.py`, `tests/test_face_geometry_publication.py` — large re-derivation, not incremental addition.
- Docs: `docs/CROP_GEOMETRY.md`, `docs/CROP_IDENTITY.md`, `docs/FACE_SELECTION_IDENTITY.md` all need a new implementation section, matching the project's own convention for every prior identity-moving batch.

**Expected version-axis impact:**

| Axis | Expected impact |
|---|---|
| `CONTRACT_VERSION` | **Unchanged**, unless R26's `BoundingBox.ge=0` constraint is loosened as part of this program (a real possibility now that a real detector's box format is knowable) — if so, a MINOR-or-larger bump, decided at design time, not assumed here. |
| `RUN_SCHEMA_VERSION` | **Likely unchanged.** No new `BenchmarkRun` field is implied by a detector swap alone. |
| `CALIBRATION_WITNESS_VERSION` | **Possible bump.** `pre_calibration_identity` (R31) already includes the face-selection fingerprint; whether a new detector-identity segment folds cleanly into the existing digest scope, or needs a schema bump the way R31 itself needed one, is a design-time question this program must answer explicitly — flagged, not resolved, here. |
| `.d.` | **Unchanged.** Decode identity concerns media decoding, not face detection — orthogonal code paths (confirmed by the dependency analysis in `docs/R23_R28_DESIGN.md` §5). |
| `.f.` | **Moves — the headline identity impact of this program.** A new detector-identity segment (and, if R28/R23-border-clip are bundled in, a changed `interpolation_rule`/`crop_clip_rule` value) regenerates `FaceSelectionPolicy.identity`, which is the single largest identity move since B22/B23. One or more cache generations across both media namespaces (the N8-accepted architectural cost, re-exercised at larger scale). |

---

## 2. Program: Real Model Integration

**Objective.** Replace `StubBackend`'s hash-derived scores with a real,
trained (or fine-tuned pretrained) deepfake classifier, so that every
severity number in this audit — R13's amplification factor, the
calibration-witness convergence risk, every "measured against a stub, not
validated against reality" caveat repeated throughout `docs/` — becomes a
claim about the shipped system instead of about a placeholder.

**What this closes, from `PROJECT_STATUS_FINAL.md` §3 and §5:** unblocks
R23's border-clip fix and R28's predicate fix (both are blocked on "no real
model to say whether a pixel change helps," not on a missing feature), R10
(GPU/training nondeterminism — literally cannot be scoped before a training
pipeline exists), R32 (only re-opens with a real, static-shape or
cross-frame backend), and N6 (only becomes live once a second caller of
`fit_calibration` exists with a reason to override its defaults). It also
retires the project's single largest standing caveat — every result today
is provably about `StubBackend`, not about anything that could ship.

**Prerequisites (the largest set of any program here):**
- **Dataset access and licensing** — confirmed absent today
  (`REPRODUCIBILITY_AUDIT.md:78`, `docs/EVALUATION_IDENTITY.md:418`,
  `docs/THRESHOLD_BAND_DESIGN.md:433`, all re-confirmed in
  `PROJECT_STATUS_FINAL.md` §3). A benchmark corpus (e.g. FaceForensics++,
  Celeb-DF, DFDC) with usable licensing terms must be selected and acquired
  before any model work starts.
- **Compute** — GPU access for either training or, at minimum, a
  credible fine-tuning pass; not established anywhere in this repository.
- **An architecture/sourcing decision** — train from scratch vs. adopt and
  fine-tune an existing pretrained deepfake-detection model. This is itself
  a design decision with real tradeoffs, not a default.
- Ideally, SCRFD landed first (Program 1) — calibrating a real model against
  Haar-cropped faces bakes R13's instability into the training data itself.

**Estimated complexity: Very Large — a program, not a batch, by this
project's own repeated characterization** (`docs/PROJECT_ROADMAP.md` §3b/§3c,
re-confirmed in `PROJECT_STATUS_FINAL.md` §5). Dataset/training work of this
kind is commonly weeks-to-months even in the "adopt a pretrained model"
shortcut case, before any of this codebase's own integration work begins.

**Dependencies.** Benefits substantially from SCRFD (Program 1) landing
first, for the reason above. Does not require Video Face Stage (Program 3)
— a real model can score the video path's existing whole-frame pooling
without per-frame face detection, exactly as `StubBackend` does today.

**Expected files/modules affected:**
- New `inference/models/` backend implementation (a real `ModelBackend`, alongside — not necessarily replacing — `StubBackend`, which several tests may still want as a fast deterministic fixture).
- `inference/models.py` — `ModelSpec`, `weights_identity` for a real weights file (a genuine binary asset, not a synthetic hash).
- `evaluation/calibration_protocol.py`, `evaluation/benchmark.py` — first real calibration fit, first real `BenchmarkRun` against real data.
- `backend/config.py` — a real `model_version` literal.
- Large re-validation surface: every test built on `StubBackend`'s deterministic, hash-derived scoring (`tests/test_evaluation_*`, `tests/test_crop_identity.py`'s output-identical assertions, the calibration-witness suite) needs re-deriving against real, non-deterministic-until-fixed model behavior.
- New `docs/` design chain (this project's own convention — design → adversarial review → resolution → implementation — will almost certainly apply here at least once, likely more given the stakes).

**Expected version-axis impact:**

| Axis | Expected impact |
|---|---|
| `CONTRACT_VERSION` | **Likely unchanged** for the wire shape itself — a real backend doesn't need new `EvidenceBundle` fields to function. A bump is possible only if new metadata (e.g. model card fields) is added to the public contract, a decision, not a given. |
| `RUN_SCHEMA_VERSION` | **Likely bump.** The first real `BenchmarkRun` rows plausibly need new fields (real dataset/split identifiers, GPU/driver provenance feeding R10) that the current schema, built entirely around synthetic fixtures, doesn't carry. |
| `CALIBRATION_WITNESS_VERSION` | **Likely bump, and the highest-stakes one in this document.** Every `R31`/`N6` discussion in the audit explicitly deferred its "what happens with a real model" question to this exact program. Today, zero calibration witnesses exist anywhere (confirmed by search in R31 and Batch 26) — this program creates the first ones, which is precisely the scenario the digest was designed, but never exercised, for. |
| `.d.` | **Likely unchanged** in principle — decode identity is about media decoding, independent of the scoring model — **unless** R10 (GPU/training nondeterminism) forces a new identity segment for inference-time GPU/driver provenance, which is plausible but not yet designed. |
| `.f.` | **Moves — the single largest identity move in this project's history.** A real `weights_identity` replaces the synthetic stub hash entirely; every downstream cache, benchmark, and calibration identity that folds in `.f.` regenerates. This is a full identity-tree regeneration, not an incremental segment addition like every prior batch. |

---

## 3. Program: Video Face Stage

**Objective.** Give the video pipeline real per-frame (or per-clip) face
detection instead of scoring whole resized frames, so the video path's
face semantics match the image path's.

**What this closes, from `PROJECT_STATUS_FINAL.md` §3:** R18's capability
half (a faceless clip currently receives a decided verdict instead of an
abstention), R24 (whole-frame squash onto a square, 1.778× on 16:9 —
described everywhere as a symptom of scoring whole frames, not faces), and
R6-a/R6-b (the `frame_count` reproducibility residual's serving-path fix is
explicitly scoped as a video-face-stage acceptance criterion in
`docs/FRAME_COUNT_REPRODUCIBILITY.md` §9).

**Prerequisites:**
- A working, stable face detector already in the pipeline. Not a hard
  technical requirement (a video face stage could, in principle, run Haar
  frame-by-frame), but doing this work against a detector already scheduled
  for replacement (Program 1) is the kind of throwaway effort this
  project's own process (§0's precedent: R28 named-but-not-fixed rather than
  shipped on a soon-to-move target) explicitly avoids.
- A design decision on selection semantics: which frames count as
  "face-present," how a per-frame face score aggregates into the existing
  branch/pooling architecture (`VideoSelectionPolicy`, already versioned
  since B23), and what an all-faceless clip's `EvidenceBundle` should
  contain (abstain entirely? a new capability flag?).

**Estimated complexity: Large.** A new architectural component — this
project has never had per-frame face detection in the video path in any
form — not an extension of existing code. Expect a new selection-policy
identity segment (in the same versioned style as `VideoSelectionPolicy`
from B23) and a genuine `EvidenceBundle`/`FaceTrack` semantics question for
the faceless-clip case (R18), which the audit already flagged (R21's
Batch-16 validators) as needing a real signal that doesn't exist yet.

**Dependencies.** Strongly sequenced after SCRFD (Program 1) for the reason
above. Independent of Real Model (Program 2) — this program can proceed
against `StubBackend` exactly as every other identity-plumbing batch has.

**Expected files/modules affected:**
- `inference/preprocess.py` — the video pipeline's frame-extraction/face-detection integration (new code, not a modification of an existing function the way most prior batches were).
- A likely new module for video-specific per-frame face selection (parallel to how `FaceSelectionPolicy` exists for images today).
- `contracts/video.py` (`VideoAnalysis` — possibly new fields describing face coverage), `contracts/evidence.py` (`FaceTrack`'s per-frame semantics, R18's abstention question).
- `docs/VIDEO_FACE_SEMANTICS.md`, `docs/FRAME_COUNT_REPRODUCIBILITY.md` (closing R6-a/R6-b's acceptance criteria), `docs/VIDEO_SELECTION_IDENTITY.md` (a new selection-policy generation).
- Test surface: a large new suite for the face-stage itself, plus re-derivation of every existing video-path test built on whole-frame scoring (`tests/test_video_selection_identity.py`, `tests/test_frame_count_reproducibility.py`-class coverage).

**Expected version-axis impact:**

| Axis | Expected impact |
|---|---|
| `CONTRACT_VERSION` | **Likely bump (at least MINOR).** New or changed wire fields to describe per-frame face presence/coverage are the whole point of this program — unlike Programs 1 and 2, this one is hard to scope without touching the public contract. |
| `RUN_SCHEMA_VERSION` | **Possibly unchanged**, possibly a minor addition if `BenchmarkRun` needs to record per-clip face-coverage statistics for evaluation purposes. |
| `CALIBRATION_WITNESS_VERSION` | **Possibly unchanged**, unless the video selection identity change is judged to materially affect calibration inputs the way R31's `pre_calibration_identity` was designed to catch — a design-time question, not assumed here. |
| `.d.` | **Unchanged.** Orthogonal to media decoding. |
| `.f.` | **Moves.** A new `VideoSelectionPolicy`-class identity segment for face-aware frame selection, following the exact precedent B23's `selection_identity(face, video)` composite already established — this program is architecturally the most "on-pattern" of the four for how the identity system would absorb it. |

---

## 4. Program: Production Deployment

**Objective.** Exercise the system as an actual deployable product for the
first time: a real storage backend under load, `jobqueue/redis.py` (present
but currently unused — a standing "no" constraint on every batch to date,
not a tested path), the `.github/workflows/tests.yml` CI workflow's first
real execution, monitoring, and — separately, and optionally — a frontend
(a standing "no" constraint throughout the project's history).

**What this closes, from `PROJECT_STATUS_FINAL.md` §3/§5:** R34 (the
rolling-deployment coordination constraint only matters once there is a
real deployment to roll) and the general "no exercised deployment" gap
named in every phase-boundary discussion. It does not close any numbered
finding directly — it is infrastructure, not a fix.

**Prerequisites:**
- **Soft dependency on Program 2 (Real Model).** Deploying a system whose
  backend is `StubBackend` has limited product value — the "production
  surface" is only worth building around something worth serving. The
  CI-exercise and Redis-integration slices, however, do **not** strictly
  need a real model and could be pulled forward independently if the goal
  is specifically "prove the deployment mechanics work," decoupled from
  "prove the model is good."
- A decision on hosting/infra (none exists today — no Dockerfile, no
  cloud target, confirmed absent by every design doc's "repo context"
  section since B24).

**Estimated complexity: Large, but the most bounded "Large" of the four** —
this is systems/DevOps work, not ML research. Once started, its sub-parts
(CI exercise, Redis integration, storage-under-load testing, monitoring,
frontend) are each independently schedulable and don't carry the
"invalidates existing test assumptions" cost the other three programs do.

**Dependencies.** Soft-dependent on Program 2 for product value; the
CI/Redis/infra slice is independent of all three other programs and could,
in principle, run in parallel with any of them.

**Expected files/modules affected:**
- `.github/workflows/tests.yml` (first real run, and likely revision once it actually executes against real CI infrastructure rather than being validated only by reading it).
- `jobqueue/redis.py` and its call sites (currently dead code from a testing perspective).
- Storage backend configuration (`storage/`) under real load, not just unit-level exercise.
- A new frontend (does not exist in any form) — a separate, large sub-scope of this program on its own.
- New monitoring/observability code — does not exist today.

**Expected version-axis impact:**

| Axis | Expected impact |
|---|---|
| `CONTRACT_VERSION` | **Unchanged**, unless a frontend requires new API endpoints/fields — orthogonal to deployment mechanics themselves. |
| `RUN_SCHEMA_VERSION` | **Unchanged.** |
| `CALIBRATION_WITNESS_VERSION` | **Unchanged.** |
| `.d.` | **Unchanged.** |
| `.f.` | **Unchanged.** This is the one program of the four fully orthogonal to the identity/fingerprinting axis — nothing about deployment infrastructure touches decode, weights, or selection identity. |

---

## 5. Cross-program summary

| Program | Closes (from §3) | Complexity | Hard dependency | `.f.` moves? | `.d.` moves? | Contract/Schema/Witness bump likely? |
|---|---|---|---|---|---|---|
| **1. SCRFD integration** | R13, R23-occupancy, (R28 + R23-border-clip if bundled), unblocks R26/R16 decisions | Large | None | **Yes** | No | Witness: maybe. Contract: only if R26 decided. Schema: no. |
| **2. Real model integration** | Unblocks R23-border-clip/R28 (validation), R10, R32, N6; retires the stub-only caveat entirely | Very Large | Benefits from #1 | **Yes (full regen)** | Maybe (if R10 needs it) | Schema: likely. Witness: likely (first-ever real witnesses). Contract: maybe. |
| **3. Video face stage** | R18 (capability half), R24, R6-a/R6-b | Large | Should follow #1 | Yes | No | Contract: likely. Schema/Witness: maybe. |
| **4. Production deployment** | R34, "no exercised deployment" gap | Large (bounded) | Soft dependency on #2 for value | No | No | None expected. |

**Sequencing implied by the dependency column, not otherwise stated
anywhere in this document:** 1 → (3 and/or 2 in either order) → 4. Program 1
is the only one with zero dependencies and the only one that unblocks parts
of two other programs' scope (R26/R16 decisions, and de-risking Program 3).

---

## 6. Recommendation for a single developer relocating to Germany at the end of October, with limited time afterward

**Recommended next program: SCRFD Integration (Program 1), scoped as a
vertical slice if full scope doesn't fit the remaining calendar time.**

Reasoning, weighed against the other three:

- **It is the only program with zero hard dependencies and no acquisition
  lead time.** Programs 2 and 4 need something external — a licensed
  dataset and compute, or an infrastructure/hosting decision — before real
  work can even start. Program 1 needs a pretrained ONNX asset, which is a
  same-day decision, not a multi-week acquisition.
- **It closes the single largest quantified defect in the entire audit**
  (R13's 310× amplification, 25.7% of decisions) plus three to four adjacent
  findings, for one bounded scope of work. No other program has this
  ratio of audit-impact to prerequisite cost.
- **It is time-boxable in a way the others are not.** Even an incomplete
  slice — detector swapped, identity regenerated, R13 re-measured against
  the new detector — is a complete, demonstrable result on its own; it does
  not require finishing the full geometry-fix reproof to be worth having
  done. Program 2 (Real Model) has the opposite property: a half-finished
  training run or an unlicensed, incomplete dataset pull is worse than not
  starting, and multi-month timelines routinely slip past a hard relocation
  date. Program 3 depends on Program 1 landing first, so starting it now
  risks throwaway work. Program 4 has the least payoff without Program 2
  behind it, and infra/hosting decisions are exactly the kind of
  low-reversibility choice that's hard to make well under relocation
  pressure and then live with from a new country.
- **It produces the clearest, most self-contained portfolio narrative for
  a Junior AI Engineer / AI Werkstudent application**: "replaced a
  classical Haar-cascade detector with a modern ONNX-based one inside an
  already-versioned, identity-fingerprinted reproducibility system, and
  re-validated the affected test surface" is a complete, concrete story
  even if told about a partial implementation — unlike an unfinished
  dataset-acquisition-and-training effort, which reads as unfinished at any
  stopping point.

**Explicitly not recommended as the next program, given the same
constraint:**
- **Real Model Integration** — highest long-term value, but the
  prerequisite (dataset licensing + compute) has no guaranteed lead time,
  and this project's own documents independently size it at
  weeks-to-months even in the shortcut case. Starting it now risks landing
  mid-relocation in the worst possible state: partially acquired data,
  no working model, and no clean stopping point.
- **Video Face Stage** — not wrong to want, but sequenced behind Program 1
  by this document's own dependency analysis; starting it first risks
  rework once SCRFD lands.
- **Production Deployment** — lowest ROI without a real model behind it,
  and its infra/hosting decisions are exactly the kind of thing better made
  with time to live with the consequences, not right before a move.

**If even the full scope of Program 1 does not fit the remaining time:**
scope it down to a single vertical slice — the image pipeline only, one
detector, one identity regeneration, R13 re-measured — and explicitly defer
the video-path half and the full geometry-option (Options A–F) reproof to
whoever picks this project up next. A clearly-scoped, honestly-labeled
partial result is consistent with this project's own dominant convention
(every batch states precisely what it did *not* guarantee) and is a better
outcome, both technically and for a portfolio, than an ambitiously-scoped
program abandoned mid-flight.
