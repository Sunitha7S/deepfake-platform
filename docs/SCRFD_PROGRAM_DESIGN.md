# SCRFD Integration Program — Design

**Date:** 2026-09-18. **Type:** design only — no code. **Basis:**
`docs/PROJECT_STATUS_FINAL.md` and `docs/NEXT_ENGINEERING_PROGRAM.md`,
re-read in full for this document, plus a fresh source re-derivation this
session of every module the program touches — not summarized from either
prior document. Where this document's conclusions differ in precision from
`NEXT_ENGINEERING_PROGRAM.md` §1, that is because reading the actual code
this time surfaced detail a planning-level pass could not have — noted
inline, not silently.

---

## 1. Findings currently marked SCRFD-dependent (re-confirmed from source)

Read from `PROJECT_STATUS_FINAL.md` §3 and cross-checked this session
against `REPRODUCIBILITY_AUDIT.md`'s own rows (unchanged since that document
was written — no concurrent-session drift detected, `git status` re-checked
before this document was started):

| ID | What SCRFD unblocks | Still needs a separate decision after SCRFD lands? |
|---|---|---|
| **R13** | 310× crop-instability amplification (`REPRODUCIBILITY_AUDIT.md:1484-1518`) — the fix *is* a stable detector | No — closable by measurement alone, against the acceptance criterion in §6 below |
| **R23 (occupancy collapse half)** | Free-aspect boxes are structurally unreachable under Haar (`|w−h| ≤ 1`, re-verified live this session in `inference/preprocess.py:879-906`); SCRFD's boxes are free-aspect | No, but the *fix* (which crop-geometry option) is a separate decision — §4.3 |
| **R28** | Bundled with R23's border-clip fix in every existing design document | No — same real-model blocker as R23's border-clip half, not resolved by SCRFD alone (see §7's honest scope note) |
| **R26 (Tier 2-b, Tier 3)** | "Wait for a real detector's box format" was the explicit reason for deferring, per `docs/R16_R26_SPEC_RESOLUTION.md` Decision 1 | Yes — a genuine contract-shape decision, §4.5 |
| **R16 (Option 2)** | Per-face scoring is only meaningful once detection is stable (`docs/MULTI_FACE_SEMANTICS.md`) | Yes — a product decision, out of this program's scope (§7) |

**Not moved by this program, confirmed by re-reading each:** R18 (video
capability half), R24, R6-a/R6-b — all gated on **video-face-stage**, a
separate program per `docs/NEXT_ENGINEERING_PROGRAM.md` §3, and this
program's own source re-derivation confirms why: the video pipeline never
calls `detect_faces` or `crop_face` today (§3.6) — there is no face-aware
code path in video for SCRFD to replace yet.

---

## 2. Re-derived current architecture

### 2.1 Detector layer

`inference/preprocess.py` has **no detector abstraction**. `detect_faces`
(`:879-906`) calls `_face_cascade().detectMultiScale(...)` inline — a
module function hard-wired to `cv2.CascadeClassifier`, not a pluggable
component. This is asymmetric with the *scoring* layer: `inference/models.py`
already has `ModelBackend` (`Protocol`, `:380-393`) with a registry-driven
`_build_backend` dispatch (`stub`/`torch`/`onnx`) — the exact shape a
detector abstraction needs and does not yet have. **This asymmetry is the
first concrete architecture fact this program has to resolve**, not a
cosmetic observation: swapping the detector today means editing
`detect_faces`'s body, not selecting a different implementation.

`FaceSelectionPolicy` (`:432-591`) already anticipates the swap at the
*data* level — `detector: str` and `detector_asset: str` fields exist, and
three docstrings say so explicitly (`:462-463` *"which is what the SCRFD
migration will do"*, `:365-369`, `:888-895`). But the **other seven fields
are Haar-shaped, not detector-agnostic**: `scale_factor` and `min_neighbors`
are `detectMultiScale` pyramid/merge parameters with no SCRFD equivalent
(SCRFD is a single-pass anchor/NMS detector — its native knobs are a
confidence threshold and an NMS IoU threshold, not a scale pyramid).
`detect_min_px` (`:478-486`, the `minSize` argument) has a natural SCRFD
analogue (a post-detection size filter) but is not literally the same
parameter. **`FaceSelectionPolicy` as it exists today cannot represent an
SCRFD configuration without either repurposing fields to mean something
they don't say, or extending the dataclass** — a real design decision, not
assumed away here (§4.2).

`cascade_digest()` (`:388-411`) hashes the Haar XML file **that ships
inside the `opencv-python` wheel** — zero acquisition cost, zero licensing
question, zero storage decision. **SCRFD's weights do not ship with any
dependency this project already has.** This is a new prerequisite category
this program introduces that no prior batch had to solve: asset acquisition,
licensing, and storage/vendoring, not just a hash function (§3.1).

`face_order_key` (`:831-876`) is a documented **total order** over
`FaceBox` geometry, explicitly justified in its own docstring as
future-proofing against "the frozen architecture replaces this detector
with SCRFD, whose box order comes out of NMS over float scores and is an
implementation detail by construction" (`:867-870`). **This is already done
— no change needed for SCRFD to use it correctly**, confirmed by reading
the function: every key component (`size_px`, `area`, `y`, `x`, `width`,
`height`) is a property of the box, not of detector-output order.

### 2.2 Face-track / contract layer

`contracts/common.py`: `BoundingBox` (`:86-90`) has `x: int = Field(ge=0)`,
`y: int = Field(ge=0)` — the R26 Tier-2-b constraint. `FaceTrack`
(`:93-134`) **already has a `landmarks` field** (`:121-123`, "5-point,
source coordinates. Empty if unavailable.") — read this session in
`inference/postprocess.py:675-685` (`face_track()`): the constructor never
passes landmarks, so this field is **live in the contract, dead in
production** — a genuine zero-contract-cost win the moment a detector that
produces 5-point landmarks (SCRFD does, natively) is wired in. Also
confirmed this session: `face_track()`'s `bbox=BoundingBox(x=max(0, x),
y=max(0, y), ...)` clamp is a structural no-op under Haar (Batch 27's own
finding, `tests/test_face_geometry_publication.py`) — **this clamp becomes
live, not cosmetic, the moment a detector can emit an edge-crossing box**,
which SCRFD can.

### 2.3 Crop pipeline

`crop_face` (`:909-964`) and `resize` (`:229-238`) are unchanged since
Batch 22 — re-verified byte-identical to every description in
`docs/CROP_GEOMETRY.md` and `docs/R23_R28_DESIGN.md`. **This program does
not need to re-derive the crop-geometry analysis** — `docs/CROP_GEOMETRY.md`
§7–§10 already did the work, in more depth than this document could add,
and its "Phase 6 — SCRFD migration impact" table (its own §8) is the
authoritative before/after picture:

| Behaviour | Today (Haar) | Under SCRFD |
|---|---|---|
| crop truncation at a border | live (square boxes) | live, unchanged |
| occupancy collapse from free aspect | latent (aspect ≈ 1) | **active immediately** |
| crop mostly background | no — 59% face | **active** — 44% at aspect 0.75 |
| negative box coords clamped by `face_track` | unreachable | **active** (regressed boxes) |
| video whole-frame squash (R24) | live | unchanged — no face stage |

That document's own recommendation (§9, re-confirmed by this session's
independent read, not merely cited): **Option E (letterbox — compute the
square window, pad what falls outside rather than clipping) is the
recommended fix, arriving *with* SCRFD; Option F (an explicit
affine-preserving transform, using SCRFD's 5-point landmarks) is the actual
destination but needs landmarks to exist first** — which is why Option E,
not F, is this program's crop-geometry scope (§4.3), with F named as future
work once Option E and real landmarks have both shipped.

**Explicit scope decision for this program, stated because
`docs/CROP_GEOMETRY.md` itself declined to make it:** the detector swap and
the geometry fix are **two separable identity moves**, not one. This
program's default recommendation (§8) is to land the detector swap **first,
with crop geometry unchanged (Option A retained)**, then geometry as a
follow-on phase — the same one-change-at-a-time discipline every prior
batch in this project used (R25 folded `input_size` before the geometry
itself moved; R17 isolated the margin before the ordering rule).

### 2.4 Weights/model identity, and where a detector asset would live

`ModelSpec.weights_identity` (`inference/models.py:258-294`) folds
`input_size` into the `-w` fingerprint dict — **not** the detector. The
detector's identity lives entirely in `FaceSelectionPolicy.identity`
(`.f.`), via `detector` + `detector_asset`. This is architecturally correct
and needs no change: `weights_identity` is about the *scoring* checkpoint,
`FaceSelectionPolicy.identity` is about *what the model is shown* — the
same owner/lifecycle split `docs/CROP_GEOMETRY.md` §10 already used to
place crop geometry in `.f.` rather than a new segment. An SCRFD asset
(ONNX weights) is a **detector** asset, so it belongs in
`FaceSelectionPolicy.detector_asset`, computed the same way
`cascade_digest()` computes today's value — a `sha256` of the weights file,
truncated to the platform's standard fingerprint length.

### 2.5 Benchmark pipeline

`evaluation/benchmark.py` already has the machinery this program needs for
validation, unexercised until now: `PairedComparison` / `paired_bootstrap`
(`:536-630`) and `regression_gate` (`:783-841`) compare an incumbent and a
challenger `BenchmarkRun` by `effective_model_version`. **No new
infrastructure is needed to A/B a Haar-selection run against an
SCRFD-selection run** — both are just different `FaceSelectionPolicy`
instances feeding the same harness. The honest limit, re-derived this
session, not assumed: **every existing `run_benchmark` call still scores
with `StubBackend`**, a hash-derived, deliberately discontinuous function of
its input bytes. `StubBackend` is unsuitable for measuring R13's
acceptance criterion (§6) — a hash avalanches on any single-bit input
change, so it would report near-100% amplification for *any* crop
perturbation, trivially, telling you nothing about the detector. **The
tool this program actually needs already exists for a different reason**:
`tests/decode_study.py:129`'s `surrogate_logit` — "a smooth, Lipschitz
function of the pixels (gradient energy)" — is the exact apparatus Batch 8
and Batch 10 used to produce R13's original 310× figure
(`tests/inference_determinism_study.py:648-650`). This program's validation
strategy (§6) reuses it rather than inventing a new one.

### 2.6 Calibration implications

`ModelSpec.pre_calibration_identity` (`inference/models.py:312-339`, R31)
is `fingerprint({weights, gate, face_selection})` — it reads
`face_selection_fingerprint` **automatically**. A detector swap that moves
`.f.` therefore moves `pre_calibration_identity` **for free, with no code
change to R31's machinery**. `CalibrationWitness` (schema 1.3.0) already
includes `pre_calibration_identity` in its digest. **Confirmed this session,
same search R31 and Batch 26 both performed: zero calibration witnesses
exist anywhere in this repository.** This is the single most consequential
fact for this program's version-impact analysis (§5): **there is nothing
to migrate.** A witness-schema bump is not needed — the schema already
covers the field this program's identity move feeds — only the *digest
value* changes, and there are no stored digests to invalidate.

### 2.7 Video pipeline — confirmed out of scope

`inference/video_pipeline.py:78-125` — re-read this session: `VideoPipeline`
accepts and **carries** a `FaceSelectionPolicy` (folds it into the combined
`.f.` via `selection_identity(face, video)`, per B23/R29) but **never calls
`detect_faces` or `crop_face`**. The video path scores whole resized
frames. A detector swap therefore has **zero behavioral effect on video**
today — it only moves the video cache namespace along with the image one
(the N8-accepted architectural cost, unchanged). Building real video face
detection is Video Face Stage's scope, not this program's, confirmed by
re-reading the actual call graph rather than assuming the boundary from
`docs/NEXT_ENGINEERING_PROGRAM.md`'s higher-level description.

---

## 3. Architecture plan

### 3.1 Asset acquisition (a decision, not a default)

Unlike every identity-moving batch to date, this program needs an external
binary asset with its own licensing and distribution question. Recommended
shape, to be confirmed at implementation time, not assumed here: a
pretrained SCRFD ONNX export (the InsightFace project's SCRFD models are
the standard source; exact model size/variant — e.g. the 500KB "2.5G" vs
the larger "10G" — is a latency/accuracy tradeoff to make explicitly, not
silently). `onnxruntime` as the inference runtime (CPU-only is viable — the
detector is small relative to any future scoring model, and this project
has no GPU dependency anywhere today). Both the dependency and the asset
need to go through the **same pinning discipline N5 already established**
for `opencv-python`/`numpy` (`pyproject.toml` `[tool.uv].override-dependencies`,
`uv.lock`) — a second, unpinned "just resolve whatever's on PyPI today"
mistake here would recreate N5's original defect for a new dependency.

### 3.2 Detector abstraction

Introduce a `FaceDetector` `Protocol`, structurally parallel to
`ModelBackend`:

```
class FaceDetector(Protocol):
    def detect(self, image: np.ndarray, *, policy: FaceSelectionPolicy) -> list[FaceBox]: ...
```

`HaarDetector` wraps today's `_face_cascade()`/`detectMultiScale` call
verbatim — a pure extraction, zero behavior change, provable
output-identical the same way every label-closure batch (B18, B22) proved
its own refactor inert. `SCRFDDetector` is the new implementation.
`detect_faces` becomes a thin dispatcher reading `policy.detector` (a
registry keyed by the same string the identity already carries — no new
concept, reusing `detector` the way `ModelSpec.backend` already selects
among `stub`/`torch`/`onnx`).

### 3.3 `FaceSelectionPolicy` extension

**Decision point, not resolved here — flagged as the one open design
question this document defers to implementation-time review**, matching
this project's own convention (`docs/R23_R28_DESIGN.md` §4.4 style — name
the decision, don't pre-empt it in a planning pass): either (a) add
SCRFD-specific fields (`confidence_threshold`, `nms_iou_threshold`) as
**optional-with-a-sentinel** alongside the existing Haar fields, so one
dataclass still represents both detectors (Haar-only fields read as inert
under SCRFD and vice versa, each named and tested that way — the same
"label present, not read" pattern `interpolation_rule` already uses), or
(b) split into a `HaarSelectionPolicy`/`SCRFDSelectionPolicy` pair unified
by a common base carrying only the detector-agnostic fields (`ordering`,
`crop_margin`, `crop_shape`, `interpolation_rule`, `crop_clip_rule`).
**Recommendation, stated as a recommendation and not a foreclosed
decision:** (a) is cheaper and keeps `identity`/`to_dict`/`differences`
untouched in shape (same pattern `weights_identity` used for `input_size` —
extend the dict, don't change the format); (b) is architecturally cleaner
but ripples through every call site that currently type-hints
`FaceSelectionPolicy` and every test literal that constructs one. Both
keep `.f.` a single opaque fingerprint either way — this decision is about
maintainability, not about anything the identity/parser layer (§2.4,
`contracts/version.py`) needs to know about.

### 3.4 Box and landmark model

`FaceBox` gains nothing structurally required — SCRFD's confidence score
and 5-point landmarks are **detection-time facts**, not properties of a
box's geometry, so they do not belong on `FaceBox` (which `face_order_key`
already treats as a pure-geometry total order, §2.1). They belong on
whatever `detect_faces`/`SCRFDDetector.detect` returns *alongside* the
ordered `list[FaceBox]` — a decision for implementation time on whether
that is a parallel list, a small paired dataclass, or (least invasive) a
detector-internal cache keyed by box identity, threaded only as far as
`inference/postprocess.py:face_track()` needs it to populate the already-
existing `FaceTrack.landmarks` field. **No contract change required** —
confirmed this session, `landmarks` has shipped in the wire schema since
before this program, unpopulated.

### 3.5 Crop geometry — deferred by default, scoped if bundled

Per §2.3: this program's default scope leaves `crop_face`/`resize`
unchanged (Option A). If a later decision bundles Option E (letterbox), the
implementation is already designed in `docs/CROP_GEOMETRY.md` §9 — this
document does not re-design it, only notes the sequencing choice (§8).

---

## 4. Migration plan

1. **No coexistence/rollout problem exists to solve.** Re-confirmed this
   session: `docs/PROJECT_STATUS_FINAL.md` §0 and every `docs/` design
   document agree there is no exercised deployment (`.github/workflows/
   tests.yml` has never run against real CI, no Dockerfile, no hosting
   target). A "rolling update" concern like R34's is **not yet a live
   constraint for this program** — it becomes one only once Production
   Deployment (a separate, later program per `docs/
   NEXT_ENGINEERING_PROGRAM.md` §4) exists. This program's migration is
   therefore a **local identity move**, not an operational rollout.
2. **`HaarDetector` is kept, not deleted, after cutover.** Mirrors this
   project's own precedent for `StubBackend` (kept as a fast, deterministic
   fixture even once a real backend exists, per `docs/
   NEXT_ENGINEERING_PROGRAM.md` §2). Keeping it is what makes rollback (§9)
   a one-line change rather than a re-implementation.
3. **One cache generation at cutover** (§5's Phase 4), both media
   namespaces — the N8-accepted architectural cost this project has paid
   four times already (B20's `.d.`, B22/B23/B26's `.f.`/`.d.` moves). Not a
   new kind of cost, a repeat of a well-understood one.
4. **Zero stored-artifact migration** — re-confirmed by the same "search
   for any committed `EvidenceBundle`/`CalibrationWitness`/`BenchmarkRun`
   payload" every recent design document has performed (§2.6): none exist
   outside test fixtures.

---

## 5. Implementation phases

Each phase states its own files, test impact, and version-axis impact.
Phases 0–3 are the **quickest vertical slice** (§8); Phase 4 is the
**minimum viable integration**; Phases 0–6 are the **full target state**
this program owns (video face detection is explicitly Phase 7-and-beyond,
i.e., a different program).

### Phase 0 — Asset & dependency decision (no production code)

- **Files:** `pyproject.toml` (+SCRFD/onnxruntime dependency declaration),
  `uv.lock` (regenerated), a vendored or fetched SCRFD ONNX asset + its
  storage location (a decision: committed binary vs. fetched at build/deploy
  time — the latter matching how the Haar cascade is *not* vendored, since
  it ships inside `opencv-python` already).
- **Test impact:** a `tests/test_dependencies.py`-style addition (N5
  precedent) proving the pin resolves to the exact version the asset hash
  in Phase 2 was computed against — the same G1 gate `docs/B24_N5_DESIGN.md`
  used for `opencv-python`/`numpy`.
- **Version impact:** none. No production code path reads the new
  dependency yet.
- **`.d.` impact:** none — SCRFD is not a decode-environment fact.
- **`.f.` impact:** none — nothing constructs an SCRFD-backed policy yet.

### Phase 1 — Detector abstraction, zero behavior change

- **Files:** `inference/preprocess.py` (`FaceDetector` Protocol,
  `HaarDetector` wrapping the existing `_face_cascade()` call verbatim,
  `detect_faces` becomes a dispatcher). No other production file changes —
  confirmed by this session's read that `detect_faces` has exactly one
  production call site per pipeline (`inference/image_pipeline.py:296`'s
  `_detect_and_measure`; the video pipeline does not call it at all, §2.7).
- **Test impact:** new output-identical proof (`HaarDetector.detect(...) ==
  detect_faces(...)` byte-for-byte on the existing fixture corpus) — the
  same "prove the refactor is inert" pattern every label-closure batch
  used. Zero existing test should need a code change, only possibly an
  import path.
- **Version impact:** **none.** `FaceSelectionPolicy.detector` still reads
  `"opencv-haar-frontalface-default"`; `identity` is unchanged. `CONTRACT_
  VERSION`/`RUN_SCHEMA_VERSION`/`CALIBRATION_WITNESS_VERSION` all unmoved.
- **`.d.` impact:** none. **`.f.` impact:** none.

### Phase 2 — SCRFD implementation, wired but not defaulted

- **Files:** `inference/preprocess.py` (`SCRFDDetector`, the
  `FaceSelectionPolicy` extension per §3.3's decision, a new
  `scrfd_asset_digest()`-equivalent of `cascade_digest()`), a new
  `tests/scrfd_detector_study.py` (the project's established convention —
  a re-runnable study script beside the design doc, matching
  `crop_geometry_study.py`/`decode_study.py`). `default_face_selection()`
  **is not changed** — the shipped default stays Haar.
- **Test impact:** new unit coverage for `SCRFDDetector` in isolation
  (box format, confidence filtering, landmark passthrough); no existing
  test changes, because nothing production-facing has moved yet.
- **Version impact:** **none in production.** A test-constructed
  SCRFD-backed `FaceSelectionPolicy` has its own `identity`, but nothing
  ships it — `default_face_selection()`, `STUB_ENGINE_VERSION`, and every
  pipeline's `model_version` are byte-identical to Phase 1.
- **`.d.` / `.f.` impact:** none in the shipped configuration.

### Phase 3 — Validation against the R13 acceptance criterion

- **Files:** none production — this phase is measurement, using Phase 2's
  isolated `SCRFDDetector` plus the existing `surrogate_logit`
  (`tests/decode_study.py:129`) apparatus. Output: a report (a new
  `docs/`-style measurement document, the project's own convention for
  every prior quantified finding).
- **Test impact:** none required to merge, though the measurement itself
  is expected to become a permanent regression pin once Phase 4 cuts over
  (the same "measured, then pinned" pattern R13's own original 310×
  figure used).
- **Version impact:** none — still nothing shipped.
- **Gate:** this phase's result is what decides whether Phase 4 proceeds
  at all (§6, §9 "abort" branch).

### Phase 4 — Cutover: `default_face_selection()` returns SCRFD

**This is the "minimum viable SCRFD integration" (§8).**

- **Files:** `inference/preprocess.py` (`default_face_selection()` factory
  changes, `crop_face`/`resize` **unchanged** per §2.3's scope decision),
  `backend/config.py` (`STUB_ENGINE_VERSION` literal), `inference/
  postprocess.py` (`face_track()` gains a `landmarks` parameter and passes
  it through — a genuine, low-cost capability win, §3.4), `contracts/
  common.py` **only if** §4.5's box-constraint question (below) is decided
  to move in this phase rather than deferred.
- **Test impact — the largest of any phase, matching this project's own
  precedent for every detector-adjacent change.** Every file this session
  confirmed references Haar's square-box guarantee needs re-deriving, not
  just re-running: `tests/test_crop_geometry.py`, `tests/
  test_crop_identity.py`, `tests/test_face_selection_identity.py`, `tests/
  test_face_geometry_publication.py`, `tests/test_headline_and_face_
  semantics.py`, `tests/test_inference.py`, `tests/
  test_inference_determinism.py`, `tests/test_lease_safety.py` (re-checked
  this session — 8 production test files, plus the 6 study scripts that
  are re-run, not modified: `tests/crop_geometry_study.py`, `tests/
  crop_identity_study.py`, `tests/decode_study.py`, `tests/face_selection_
  study.py`, `tests/face_size_policy_study.py`, `tests/inference_
  determinism_study.py`, `tests/multi_face_study.py`, `tests/video_face_
  semantics_study.py`). Collateral version-literal pins: every hardcoded
  `stub-0-w...+c...g...f...d...` string across the test suite (the same
  "collateral pins" bookkeeping every prior `.f.`-moving batch has done).
- **Version impact:**
  - `CONTRACT_VERSION` — **unchanged**, unless §4.5's box-constraint
    question is decided to move here.
  - `RUN_SCHEMA_VERSION` — **unchanged.**
  - `CALIBRATION_WITNESS_VERSION` — **unchanged as a schema number**
    (already 1.3.0, already covers `pre_calibration_identity` generically,
    §2.6). The **digest value** any future witness computes changes, but
    there is nothing stored to migrate.
  - `.d.` — **unchanged.** Detection is not a decode-environment fact.
  - `.f.` — **moves.** `FaceSelectionPolicy.identity` regenerates
    (`detector`, `detector_asset`, and whichever new fields §3.3 adds all
    feed it); `STUB_ENGINE_VERSION` regenerates; one cache generation,
    both media namespaces (N8).

### Phase 5 — R26 box-constraint decision (optional, can ride Phase 4 or follow it)

- **Files:** `contracts/common.py` (`BoundingBox.x`/`.y`'s `ge=0`, if
  loosened), `contracts/common.py`/`inference/postprocess.py`
  (`face_track`'s clamp, if removed rather than merely proven live).
- **Test impact:** `tests/test_face_geometry_publication.py`'s existing
  "clamp is a structural no-op" tests need re-deriving into "clamp is now
  load-bearing" tests, or replacement tests if the clamp itself changes.
- **Version impact:** `CONTRACT_VERSION` — **likely bump (at least MINOR)**
  if `BoundingBox`'s constraint changes; **unchanged** if the decision is
  instead "keep clamping, document that a regressed box is now silently
  reported as edge-flush" (the conservative option, and the one `docs/
  R16_R26_SPEC_RESOLUTION.md` Decision 1 leaned toward deferring rather
  than deciding blind). `RUN_SCHEMA_VERSION`/`CALIBRATION_WITNESS_VERSION`
  — unchanged either way. `.d.` — unchanged. `.f.` — unchanged (this is a
  contract-shape question, not a selection-policy one).

### Phase 6 — Crop-geometry follow-on: Option E (letterbox), bundled with R28's fix

- **Files:** exactly as `docs/CROP_GEOMETRY.md` §9 Option E and `docs/
  R23_R28_DESIGN.md` §2.5/§3.5 already specify: `inference/preprocess.py`
  (`crop_face`'s window-filling logic, `resize`'s interpolation predicate),
  `FaceSelectionPolicy.crop_clip_rule`'s value (changes from
  `"intersect-image"` to the chosen fill mode — "the one accepted
  double-spend" `docs/CROP_IDENTITY.md` §10 already flagged),
  `interpolation_rule`'s value, the full `tests/test_crop_geometry.py`/
  `tests/test_crop_identity.py` surface again, `docs/CROP_GEOMETRY.md` (a
  new implementation section, the project's standing convention).
- **Test impact:** large, second pass over the same files Phase 4 already
  touched — this is *why* §8 recommends sequencing this after Phase 4
  rather than bundling it in, so a test failure localizes to one identity
  move at a time.
- **Version impact:** `.f.` **moves a second time** (a real value change
  this time, not a label closure — `docs/R23_R28_DESIGN.md` §2.6 already
  named this exact cost). `CALIBRATION_WITNESS_VERSION` — per §2.6's
  reasoning, still no schema bump, still nothing stored to migrate (unless
  Phase 4/5/Real-Model-program work has since created real witnesses, in
  which case this phase's design must be revisited against that — flagged,
  not assumed away). `CONTRACT_VERSION`/`RUN_SCHEMA_VERSION`/`.d.` —
  unchanged, per `docs/CROP_GEOMETRY.md` §10's own analysis, re-confirmed
  unchanged by nothing in this session's re-read contradicting it.

---

## 6. Benchmarking strategy

**What can be measured, honestly, without a real model or a real dataset
(neither exists, `docs/PROJECT_STATUS_FINAL.md` §3):**

1. **R13's acceptance criterion, exactly as written in
   `REPRODUCIBILITY_AUDIT.md:1516-1518`:** *"Under ±4 LSB uniform noise on
   real photographs, across ≥ 200 images: no face lost, and the box short
   edge moving ≤ 1%. Amplification below 10×, from the measured 310×."*
   This is answerable **today**, using Phase 2's isolated `SCRFDDetector`
   and the existing `surrogate_logit` apparatus (§2.5) — no real model
   needed, because the criterion is about detection *stability*, which the
   surrogate was built to measure independent of what scores the crop.
   This is this program's primary, quantitative go/no-go gate.
2. **Structural regression via the existing `paired_bootstrap`/
   `regression_gate` machinery** (§2.5) — comparing a Haar-selection
   `BenchmarkRun` against an SCRFD-selection `BenchmarkRun`, both still
   scored by `StubBackend`, over the project's existing synthetic fixture
   corpus. This validates **plumbing**, not detection quality: that the
   identity move actually separates the two configurations in the cache,
   that the harness runs end to end with a different `FaceSelectionPolicy`,
   that nothing crashes on a free-aspect box. It does **not** and cannot
   answer "did detection get better," because `StubBackend`'s hash-based
   scores carry no signal about face quality.
3. **Detection-quality proxies on a small, honestly-labeled validation
   set** (bounded manual effort — tens, not thousands, of images) —
   miss rate, false-positive rate, box-tightness, landmark plausibility,
   Haar vs. SCRFD side by side. This is the only way to get any signal on
   "does SCRFD actually find faces Haar misses" before a real scoring
   model exists to ask the question that actually matters.
4. **Latency/throughput**, since SCRFD's cost profile (a full CNN forward
   pass, even a small one) differs from Haar's cascade — worth measuring
   before cutover so Phase 4 does not silently change the pipeline's
   `preprocess_ms` budget (`contracts/common.py`'s `Timing` contract, §2.2
   of this document's source read) without anyone noticing.

**What cannot be measured, stated as plainly as `docs/CROP_GEOMETRY.md` and
every real-model-blocked finding already states it:** whether SCRFD
improves the *deepfake-detection* verdict itself. That question needs a
real scoring model and a real benchmark corpus — Real Model Integration
(`docs/NEXT_ENGINEERING_PROGRAM.md` §2), not this program.

---

## 7. Validation strategy

1. **Phase 1's output-identical proof** — the refactor introduces zero risk
   by construction, validated the same way every prior label-closure batch
   validated itself (byte-for-byte comparison against the pre-refactor
   function on the existing fixture corpus).
2. **Phase 3's R13 acceptance-criterion measurement** is the substantive
   validation gate for the whole program — §9 states the abort condition
   explicitly.
3. **Phase 4's full-suite re-run**, exactly as every prior `.f.`-moving
   batch has done (`python tests/run_all.py`, currently 54 suites / 1327
   tests per `docs/PROJECT_STATUS_FINAL.md` §8) — expected to show real,
   not incidental, test failures in the 8 files named in Phase 4, each of
   which needs re-deriving (not just updating a literal) because the
   underlying assumption (Haar's near-square box guarantee) is what
   changed.
4. **Honest scope boundary, stated once and not repeated per phase:** this
   program validates that the detector swap is *correctly implemented and
   correctly identified* (the reproducibility/plumbing question this
   project's entire process is built for). It does **not** and cannot
   validate that the resulting system is a *better deepfake detector* —
   that validation needs Real Model Integration, exactly as
   `docs/NEXT_ENGINEERING_PROGRAM.md` §5's cross-program table already
   flagged this program as a soft dependency for that one.

---

## 8. Quickest vertical slice / minimum viable integration / full target state

| Scope | Phases | What it delivers | Realistic for end of October + limited time after? |
|---|---|---|---|
| **Quickest vertical slice** | 0–3 | The architecture gap closed (`FaceDetector` abstraction), SCRFD wired and testable in isolation, and a **real, defensible, quantitative answer** to "does SCRFD satisfy the R13 acceptance criterion" — without moving any production identity or touching the 8-file test-re-derivation surface. **This alone is a complete, demonstrable, portfolio-worthy result**, per `docs/NEXT_ENGINEERING_PROGRAM.md` §6's own reasoning about partial-but-honest results being preferable to abandoned ambitious ones. | **Yes** — no external acquisition lead time beyond the asset decision (§3.1), no large test-re-derivation surface, a clean stopping point either way the measurement comes out. |
| **Minimum viable SCRFD integration** | 0–4 | R13 actually closed (not just measured), R23's occupancy half technically reachable (though still not fixed — see below), landmarks flowing into the contract. The real `.f.` move, the real 8-file test re-derivation, the real cache generation. | **Plausible**, contingent on Phase 3's measurement passing the R13 gate (§9) and on how much of the 8-file re-derivation surface fits in the remaining time — this is the phase most exposed to schedule risk. |
| **Full target state (this program's own scope, not the whole roadmap)** | 0–6 | Everything above, plus Phase 5's box-constraint decision and Phase 6's Option-E geometry fix (bundled with R28) — the complete SCRFD migration `docs/CROP_GEOMETRY.md` originally scoped. | **Not realistic for this window** — Phase 6 alone is a second full pass over the same large test surface Phase 4 touches, and `docs/CROP_GEOMETRY.md`'s own Option-E analysis flags real re-analysis cost, not just code. Consistent with `docs/NEXT_ENGINEERING_PROGRAM.md` §1's own "Large" complexity rating for this program. |

**Explicitly out of this program's full target state, regardless of time
available:** video-path face detection (Video Face Stage, a separate
program this one only unblocks), R16 Option 2 and R26 Tier 3 (product
decisions this program surfaces but does not make, §1), and anything
requiring a real scoring model (§6, §7).

---

## 9. Rollback strategy

1. **Before Phase 4 (i.e., anywhere in the vertical slice, §8):** nothing
   to roll back — no production identity has moved. Discarding Phase 0–3
   work is a normal revert of unmerged branches/commits, no different from
   any other in-progress feature.
2. **At Phase 4's cutover:** because `HaarDetector` and a Haar-backed
   `FaceSelectionPolicy` factory are kept, not deleted (§4.2), reverting
   `default_face_selection()`'s return value is a **one-line change** that
   restores the exact prior `.f.` value and therefore the exact prior
   `STUB_ENGINE_VERSION` — the same "revert is trivial because nothing was
   deleted" property `docs/R16_R26_SPEC_RESOLUTION.md` §7 already
   documented for a smaller change. No stored-artifact migration exists to
   undo in either direction (§2.6, §4.4).
3. **Abort condition, stated explicitly rather than left implicit:** if
   Phase 3's measurement does **not** clear the R13 acceptance criterion
   (amplification not below 10×, or any face lost under the specified
   noise budget, or box short-edge movement above 1%), **Phase 4 does not
   proceed.** The correct action is not "ship it anyway and re-measure
   later" — this project's own standing discipline (`docs/R30_BUNDLE_DESIGN.md`,
   `docs/R23_R28_DESIGN.md`) is to not move an identity on structural
   argument alone once a measurement is available and contradicts it. A
   failed Phase 3 is itself a complete, useful, reportable result (a
   specific SCRFD configuration measured and found not to clear a
   pre-registered bar), not a wasted effort — and it leaves the codebase
   exactly where §8's vertical slice already left it: architecture ready,
   nothing shipped, easy to hand off or revisit with a different SCRFD
   variant (§3.1's model-size choice).
4. **After Phase 6 (full target state), if ever reached:** the same
   one-line-revert property holds for the detector swap; the geometry fix
   (Option E) is a second, independent revert (a different commit, by
   design — §4.3/§8's sequencing recommendation), consistent with this
   project's "each `.f.`-moving change is independently reviewable and
   revertible" convention (`docs/R30_BUNDLE_DESIGN.md` §8, item 12).

---

## 10. Summary table — version impact by phase

| Phase | `CONTRACT_VERSION` | `RUN_SCHEMA_VERSION` | `CALIBRATION_WITNESS_VERSION` | `.d.` | `.f.` |
|---|---|---|---|---|---|
| 0 — asset/dependency | none | none | none | none | none |
| 1 — abstraction refactor | none | none | none | none | none |
| 2 — SCRFD wired, not defaulted | none | none | none | none | none |
| 3 — validation | none | none | none | none | none |
| **4 — cutover (MVI)** | none | none | schema unchanged; digest value moves, nothing stored to migrate | none | **moves — one cache generation** |
| 5 — box-constraint decision | possible bump (decision-dependent) | none | none | none | none |
| 6 — geometry follow-on | none | none | schema unchanged; digest value moves again if any witness exists by then | none | **moves again — second cache generation** |

---

## 11. Gate

**Design status: ready for Phase 0 to begin (asset/dependency decision),
pending approval.** No blocking architectural defect was found in
re-deriving this program from source — the codebase's own design already
anticipated a detector swap at the data level (`FaceSelectionPolicy.
detector`/`.detector_asset`, `FaceTrack.landmarks`) more thoroughly than a
planning-only pass could have known without reading it. The one genuine
open design question this document surfaces rather than resolves is §3.3
(how `FaceSelectionPolicy` represents SCRFD-specific parameters) — narrow,
bounded, and appropriately left to implementation-time review rather than
pre-empted here.

**No code was written to produce this document, per instruction.**
