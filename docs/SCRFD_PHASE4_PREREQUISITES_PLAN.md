# SCRFD Integration Program — Phase 4 Prerequisites Plan

**Date:** 2026-09-19. **Type:** planning only — no code, no file other than
this one touched. **Basis:** a fresh re-derivation this session of `docs/
SCRFD_PHASE4_CUTOVER_DESIGN.md`, `docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md`,
`docs/SCRFD_ASSET_AUDIT.md`, `docs/SCRFD_PHASE2_SPEC_RESOLUTION.md`, `docs/
NEXT_ENGINEERING_PROGRAM.md`, `docs/PROJECT_STATUS_FINAL.md`,
`REPRODUCIBILITY_AUDIT.md`, and `docs/PROJECT_ROADMAP.md` — re-read in full,
plus live re-verification of `git status`, the five version constants,
`docs/SCRFD_ASSET.md`'s own recorded status, and `python tests/run_all.py`'s
current count. No conclusion below is inherited from a prior document
without being checked against current source first.

**Correction this document makes, stated up front rather than folded in
silently:** `docs/NEXT_ENGINEERING_PROGRAM.md` and `docs/
PROJECT_STATUS_FINAL.md` are both dated **2026-09-18**, before SCRFD Phase 1
(detector abstraction), Phase 2 (SCRFD implementation + wiring), and the
Phase 3 measurement-methodology correction were implemented. Both documents
describe SCRFD integration as a **not-yet-started** program with "zero hard
dependencies" and a same-day asset decision. That framing is now **stale**:
the engineering is largely done (§1 below); what remains is not
engineering at all — it is exactly the two external-asset acquisitions
those documents' own prerequisite lists named, now sharpened into concrete,
checkable steps by this document.

**No code was written to produce this document.**

### Outcome (2026-10-17, later the same day)

P1–P4 completed: licence accepted for local non-commercial evaluation, detector
verified, corpus acquired and verified, R13 measured twice. **R13 does not
clear — no-go for cutover.** See `docs/SCRFD_R13_MEASUREMENT_EVIDENCE.md`. The
table below is the state before the measurement.

### Status update (2026-10-17)

| # | Status |
|---|---|
| P1 licence | **Awaiting a human decision.** Terms re-verified live: unchanged, non-commercial research only. |
| P2 asset | Blocked on P1. **Easier than planned:** GitHub now publishes SHA-256 digests for every model-zoo asset, and `cheetah_l` ships a per-file hash manifest (`docs/SCRFD_ASSET.md`, "Re-verified 2026-10-17"). |
| P3 corpus | **Tooling done** (`tests/scrfd_corpus.py` + 9 tests: stratified selection, flat layout, SHA-256 manifest). WIDER FACE verified CC BY-NC-ND 4.0; download awaits approval. |
| P4 measurement | Blocked on P1–P3. **De-risked:** `SCRFDDetector` decoded dynamic-input exports wrongly on real photo sizes (fixed, 3 tests); the study now records asset/corpus digests, thresholds, seed and versions, with an optional JSON evidence file. |
| P5 test isolation | **Done** — implemented as explicit Haar pinning plus a tripwire, not a synthetic default asset (`docs/SCRFD_PHASE4_CUTOVER_DESIGN.md`, "Corrections from implementation"). |

---

## 1. Re-derived current state (verified live this session)

### 1.1 What is actually built (re-confirmed, not assumed)

- **`FaceDetector` Protocol, `HaarDetector`, `SCRFDDetector`** —
  `inference/preprocess.py`. `SCRFDDetector` is a complete, tested,
  verbatim port of the SCRFD authors' own reference anchor-decode/NMS
  logic, with a per-thread `onnxruntime.InferenceSession` (mirroring the
  R15 fix) and a hardcoded `CPUExecutionProvider`. Confirmed unchanged
  since `docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` §1.3 (re-read this session).
- **`detect_faces` dispatch** — live, `if`/`elif`/`raise` on
  `policy.detector`. `default_face_selection()` still returns
  `detector="opencv-haar-frontalface-default"` — **the shipped pipeline
  runs Haar, unchanged, today.**
- **`tests/scrfd_detector_study.py`** — implements the corrected, pooled,
  multi-realisation, cluster-bootstrap R13 measurement methodology (this
  session's own prior work, per `docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md`).
  `tests/test_scrfd_measurement_methodology.py` (17 tests, `run_all.py`
  suite `S8-B29`) hermetically proves the aggregation formulas are correct
  on synthetic data.
- **Re-verified live, this session:** `git status` unchanged since `docs/
  SCRFD_PHASE4_CUTOVER_DESIGN.md` was written (no concurrent-session
  drift). `python tests/run_all.py` → **57 suites, 1365 tests, 0
  failures.** Version constants unchanged: `CONTRACT_VERSION="1.0.0"`,
  `RUN_SCHEMA_VERSION="1.2.0"`, `CALIBRATION_WITNESS_VERSION="1.3.0"`,
  `LOCK_VERSION="1.1.0"`, `STUB_ENGINE_VERSION="stub-0-wacdd9240+
  c.739c3fec.g.ae232cba.f.f8244ad0.d.c9d4025c"` (`.f.`=`f8244ad0`,
  `.d.`=`c9d4025c`).

### 1.2 What is explicitly *not* built or done

- `default_face_selection()` does not return SCRFD — no cutover has
  happened (Phase 4 is design-only, per the cutover document's own gate).
- **No SCRFD asset has been fetched.** `docs/SCRFD_ASSET.md`, re-read
  verbatim this session: *"Status: Phase 0 (asset acquisition) has not
  been executed."* Both SHA-256 fields still read `<NOT YET RECORDED>`
  (grep-confirmed live this session, lines 74/84).
- **No license acceptance has been recorded** for the non-commercial
  research-only restriction (`docs/SCRFD_ASSET_AUDIT.md` §2.2,
  `docs/SCRFD_PHASE2_SPEC_RESOLUTION.md` §12's precondition — unmet).
- **No evaluation photo corpus has been acquired** (`docs/
  SCRFD_PHASE3_MEASUREMENT_PLAN.md` §3.3 — an open, unresolved human
  decision, unchanged since that document was written).
- **`tests/scrfd_detector_study.py` has never produced a real number.**
  Re-run live this session with neither env var set: prints both as
  missing, exits 0, no verdict — the same confirmed behavior as every
  prior check of this fact.
- **`REPRODUCIBILITY_AUDIT.md`'s R13 row is still `Open`** (line 103,
  re-read verbatim this session): *"quantified in B10 at 310× amplification,
  25.7% of decisions; deferred to SCRFD with an acceptance criterion."*

### 1.3 Reconciling `docs/NEXT_ENGINEERING_PROGRAM.md`'s prerequisite list against current reality

That document (§1, "Program: SCRFD Integration") named one prerequisite:
*"A concrete SCRFD asset decision... a same-day decision, not a multi-week
acquisition"* (§6). Two things have changed since:

1. **The engineering the document assumed still needed doing is done** —
   `FaceDetector`/`SCRFDDetector`/dispatch/`FaceSelectionPolicy` extension
   (its own §1 "Expected files/modules affected" list) all exist and are
   green.
2. **The "same-day decision" framing undersold the asset question**, per
   `docs/SCRFD_ASSET_AUDIT.md`'s own later, primary-source-verified finding
   (§2 of that document): the license is real, non-commercial-research-only,
   applies to every variant of every asset family with no exception, and
   its *acceptance* is same-day, but the audit's own gate (§8) is explicit
   that acceptance must be a conscious, recorded human act — "same-day" was
   never "automatic" or "free of a decision point."

**Net correction: the remaining prerequisite is narrower than either prior
document scoped it (no more architecture to build) and more concrete than
either scoped it (a checklist, not "a decision," §2 below) — not larger in
kind, but more precisely specified.**

---

## 2. The exact minimum prerequisites before Phase 4 cutover can be approved

Derived directly from where `docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` §2/§13
and `docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md` §11 both independently stopped,
re-verified against current source (§1.2) rather than re-copied:

| # | Prerequisite | Status today | Blocks |
|---|---|---|---|
| **P1** | A human reads and consciously accepts (or rejects) the non-commercial-research-only license for the exact environment(s) the asset is fetched into | **Not done** | P2 |
| **P2** | The SCRFD asset is fetched, its detector member's filename confirmed, both SHA-256 values (zip + extracted member) recorded in `docs/SCRFD_ASSET.md` | **Not done** (blocked on P1) | P4 |
| **P3** | A real, ≥ 200 (recommended 300) real-photograph evaluation corpus is acquired, with its own license/provenance recorded | **Not done** | P4 |
| **P4** | `tests/scrfd_detector_study.py` is run for real, at least twice with different `$DFP_SCRFD_STUDY_SEED` values, against the asset and corpus above, and **clears R13's acceptance criterion** (amplification gated on the 95% bootstrap CI's upper bound, per the corrected methodology) | **Not attempted** (blocked on P1–P3) | Phase 4 cutover approval |
| **P5** | A synthetic default-fixture mitigation exists so the full test suite does not require the licensed asset to run (`docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` §4.6/§7.4) | **Not built** | Phase 4 cutover *implementation* (not the measurement itself — independent of P1–P4, §7 below) |

**P1–P4 are the measurement gate. P5 is an implementation-readiness item
that has no dependency on P1–P4 and can proceed in parallel (§7).** Nothing
else blocks Phase 4 — the architecture, dispatch, and detector code are
already done (§1.1) and need no further design work to be *usable*; they
need real inputs to be *validated*.

---

## A. Asset acquisition

### A.1 License requirements

Three independent primary sources (`model_zoo/README.md`'s banner,
`python-package/docs/model_zoo.md`, `python-package/README.md`), all fetched
and quoted verbatim in `docs/SCRFD_ASSET_AUDIT.md` §2.2, agree word-for-word:
**"The pretrained models... are available for non-commercial research
purposes only, including both automatically and manually downloaded
models."** This applies to **every** SCRFD variant in **both** upstream
asset families (Family A's raw `.pth`/converted `.onnx`, Family B's
`buffalo_*`/`antelopev2` packs) — there is no permissively-licensed SCRFD
checkpoint to substitute instead (`docs/SCRFD_ASSET_AUDIT.md` §5.3: *"The
risk is structural to using any official InsightFace-distributed SCRFD
weight at all"*). The **code** (Apache-2.0 for `detection/scrfd/`'s
training/export tooling; MIT for the `insightface` Python library) is
separately, permissively licensed — only the **weights** carry the
restriction.

### A.2 Acceptance workflow

A concrete, ordered procedure, narrower than any prior document's framing:

1. A named human reads `docs/SCRFD_ASSET.md` §2 (the excerpted license
   text) and `docs/SCRFD_ASSET_AUDIT.md` §2 in full (the primary-source
   quotes and the redistribution analysis, §2.3).
2. That human confirms this project's own current distribution posture —
   unpublished, no commercial intent, portfolio/research use
   (`docs/SCRFD_ASSET_AUDIT.md` §2.5, re-verified live this session: no
   `LICENSE` file, no license statement in `README.md`, no git remote) —
   is still accurate for **every environment** the asset will be fetched
   into (a personal dev machine is a materially different case from a
   shared or commercial one; §A.6 below).
3. The acceptance is **recorded**, not merely performed silently — a dated
   entry (accepting party, date, environments covered) added to `docs/
   SCRFD_ASSET.md` at the time of fetch. This document does not add that
   entry itself (no code/file changes beyond this plan, per instruction);
   it specifies the requirement so whoever executes Phase 0 does not skip
   the recording step the way `docs/SCRFD_ASSET.md`'s own placeholder text
   already anticipates.
4. **Only after** steps 1–3: proceed to A.3's fetch.

**This is a same-day human task, not an engineering task** — nothing about
it can be shortened or parallelized by more design work, and nothing about
it requires code.

### A.3 SHA-256 recording procedure

Exact commands, matching `docs/SCRFD_ASSET.md` §3's own documented
procedure (Decision 3, `docs/SCRFD_PHASE2_SPEC_RESOLUTION.md`) — trust-on-
first-use, since GitHub's Releases API publishes no independently-verifiable
checksum for this specific release (`docs/SCRFD_PHASE2_SPEC_RESOLUTION.md`
Decision 3's own live-checked finding, unchanged):

```bash
# 1. Fetch (Decision 2 - Family B's official GitHub Release, buffalo_l pack,
#    the SCRFD_10G_KPS-bundling variant Decision 1 ratified)
curl -L -o buffalo_l.zip \
  https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip

# 2. Confirm size matches the recorded expectation (288.62 MB, docs/SCRFD_ASSET_AUDIT.md 1.3)
ls -lh buffalo_l.zip

# 3. Record the zip's own SHA-256
sha256sum buffalo_l.zip
#   -> paste into docs/SCRFD_ASSET.md section 3 step 2, replacing "<NOT YET RECORDED>"

# 4. Unzip and CONFIRM the detector member's actual filename - "det_10g.onnx"
#    is community convention, unverified against an actual unzipped archive
#    (docs/SCRFD_ASSET_AUDIT.md section 7 item 2)
unzip -l buffalo_l.zip
unzip buffalo_l.zip -d buffalo_l_extracted/

# 5. Record the EXTRACTED MEMBER's own SHA-256 - this is what
#    scrfd_asset_digest() actually hashes, not the zip
sha256sum buffalo_l_extracted/<confirmed-detector-filename>
#   -> paste into docs/SCRFD_ASSET.md section 3 step 4, replacing "<NOT YET RECORDED>"

# 6. Verify the opset (docs/SCRFD_ASSET_AUDIT.md section 4.3, unverified item)
python -c "import onnx; m = onnx.load('buffalo_l_extracted/<file>'); print(m.opset_import)"

# 7. Place the extracted file where $DFP_SCRFD_MODEL_PATH will point, and
#    discard the zip and every other extracted member (recognition/
#    alignment/attribute models this platform has no use for)
```

Both hashes (zip **and** extracted member) must be recorded — `docs/
SCRFD_ASSET.md` §3 already specifies this distinction, unchanged; a fetch
that records only the zip's hash cannot detect corruption introduced during
extraction.

### A.4 Storage location

**Never committed** — `.gitignore:19`'s pre-existing `*.onnx` rule already
established this convention independent of licensing, and `docs/
SCRFD_ASSET_AUDIT.md` §2.3 supplies the licensing reason it turns out to be
required anyway (committing would make this repository a second
distributor of a restrictively-licensed artifact — a redistribution act
this audit cannot clear). **Never re-hosted** on this project's own release
infrastructure (same reasoning, `docs/SCRFD_ASSET_AUDIT.md` §3's "Release
artifact" strategy, rejected). The extracted file lives at whatever local
path `$DFP_SCRFD_MODEL_PATH` names — a personal dev machine's filesystem
today; a production storage location is explicitly **not** designed here
(`docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` §4.5 — deferred to a Production
Deployment program, since no Dockerfile/hosting target exists to design
against).

### A.5 Reproducibility requirements

The trust-on-first-use chain (Decision 3) means: **the recorded SHA-256 in
`docs/SCRFD_ASSET.md`, not an independent upstream authority, is this
project's own anchor for "the same asset."** Any measurement result citing
R13's status must cite the recorded digest alongside it — a result computed
against an unrecorded or differently-hashed file is not comparable to one
computed against the recorded asset, the same discipline this project
applies to every other identity segment (`.d.`/`.f.`). Every environment
that re-uses this asset (a second developer's machine, a future CI runner)
must independently verify its local copy's `scrfd_asset_digest()` output
against the recorded value before treating a measurement as reproduced —
`_scrfd_detector_for` already performs this check mechanically (`inference/
preprocess.py`, re-confirmed in `docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` §1.6),
so this is enforced by existing code, not merely a documented convention.

### A.6 Commercial-use implications

**Explicit and structural, not a detail:** this specific SCRFD asset (any
variant, either family) cannot be used in a commercially-distributed
version of this platform without either (a) substituting a differently,
permissively-licensed face detector, or (b) obtaining direct clearance from
InsightFace/the original authors (`docs/SCRFD_ASSET_AUDIT.md` §5.3). This
constraint:

- Does **not** block the current research/portfolio use (A.2 step 2).
- **Is not resolved by variant choice** — `SCRFD_2.5G_KPS` carries the
  identical restriction as `SCRFD_10G_KPS` (§A.1).
- **Must be re-read, not re-assumed**, the moment this project's own
  distribution intent changes — a standing, unowned, unscheduled risk this
  document restates (matching `docs/SCRFD_ASSET_AUDIT.md` §5.5) rather than
  resolves, because no automated mitigation exists for a licensing-terms
  question.

---

## B. Evaluation corpus

### B.1 Minimum corpus size

**Floor: N ≥ 200 images** — the acceptance criterion's own wording
(`REPRODUCIBILITY_AUDIT.md:1516-1518`), not negotiable. **K ≥ 11
independent noise realisations per image** — matching the original 310×
measurement's own realisation count, required by the corrected methodology
(`docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md` §3.1/§4.1) for the pooled
amplification statistic to be well-defined and comparable to "310×" at all
(re-confirmed this session via `tests/test_scrfd_measurement_methodology.py`
's own fail-before/pass-after pair: a single-realisation-per-image corpus
produces an undefined, `nan` amplification ratio). **Recommended: N = 300**
(50% headroom, so a handful of unusable files doesn't silently drop the
corpus below the 200-image floor) — at 300×11×2(fixed+free) ≈ 6,600
detector calls, a few-minutes run given SCRFD's own measured per-image CPU
latency (single-digit-to-low-teens ms, `docs/SCRFD_ASSET_AUDIT.md` §1.2),
not an infrastructure problem.

### B.2 Diversity requirements

Tied to R13's own measured mechanism (`docs/INFERENCE_DETERMINISM.md`
§5.2: amplification is driven by **box-size changes near a detection
boundary**, not position), restated from `docs/
SCRFD_PHASE3_MEASUREMENT_PLAN.md` §3.2 because it is load-bearing, not
generic:

| Dimension | Requirement |
|---|---|
| Face-size range | From just-above the detection-size floor up to large, frame-filling faces |
| Lighting | Both low-light and well-lit represented |
| Compression/resolution | A realistic spread of JPEG quality and source resolutions, not one canonical setting |
| Pose/angle | Frontal and moderately off-axis both represented |
| Single vs. multiple faces | Both represented (exercises `face_order_key`'s ranking under noise) |

**Explicitly out of scope, restated:** demographic/skin-tone
representativeness is a real, separate concern for this platform's
*overall* fairness, but it is not what R13's own criterion measures (a
*stability*, not *accuracy-parity*, question) — conflating the two would
overclaim what a passing R13 gate certifies.

### B.3 Acquisition strategy

**Named as a candidate, not endorsed, per this project's own established
posture on human-decision points (matching `docs/SCRFD_ASSET.md`'s
treatment of the model asset):** a permissively-licensed public
face-detection benchmark's own validation split — **WIDER FACE** is the
concrete, cite-able candidate, since it is the same corpus SCRFD's own
paper benchmarks against (`docs/SCRFD_ASSET_AUDIT.md` §1.2's own table
headers name Easy/Medium/Hard splits, which are WIDER FACE's own
difficulty tiers) and would satisfy §B.2's diversity requirements by
construction. **Its own license must be read and recorded before use** —
this document does not perform that verification (it was not fetched or
read live this session, unlike the SCRFD asset's own license, which was
primary-source-verified in `docs/SCRFD_ASSET_AUDIT.md`). Any alternative
corpus is equally acceptable provided its license is read and recorded with
the same discipline.

### B.4 Storage strategy

**Not committed to the repository** — the same binary-asset hygiene
`.gitignore`'s `*.onnx` rule already applies to the model asset applies by
extension to a multi-hundred-image corpus: a large, licensed, external
asset does not belong in git history. Referenced via
`$DFP_SCRFD_STUDY_PHOTOS` (already the script's own env var, unchanged).
**If the corpus's own license restricts redistribution** (as WIDER FACE's
research-use terms likely do, pending B.3's verification), do not re-host
it anywhere this project controls — keep a local copy and document its
source, exactly as the model asset's own A.4 treatment specifies.

### B.5 Reproducibility requirements — a new recommendation this document adds

**Not specified by any prior SCRFD document; added here by direct analogy
to Decision 3's own TOFU precedent for the model asset, since the same gap
exists for the corpus and no document has closed it.** A corpus with no
recorded identity cannot be verified as "the same corpus" across two runs
or two machines — recommend recording, alongside `docs/SCRFD_ASSET.md`'s
existing model-asset hashes, a **corpus manifest**: the sorted list of
filenames plus each file's own SHA-256 (or a single hash over that sorted
list, the cheaper option), the corpus's source/version/split identifier,
and the acquisition date. This is the same discipline the model asset
already gets, applied to the second external input the measurement depends
on — without it, "R13 was measured against a 300-image corpus" is not a
reproducible claim, only "a" 300-image corpus existed once.

### B.6 Privacy/compliance considerations

**Not addressed by any prior SCRFD document; a real, distinct
consideration this document surfaces because the corpus is photographs of
real people's faces, a data category this project has not previously
handled (`StubBackend`'s entire history has used synthetic fixtures).**
Concrete recommendations, none of which block starting acquisition but all
of which should inform the source chosen (§B.3):

- Prefer a corpus with documented consent/ethics provenance (an academic
  benchmark's own release terms, e.g. WIDER FACE's stated research-use
  license) over ad hoc or scraped photographs of identifiable private
  individuals.
- The measurement itself does not perform recognition, matching, or
  identity storage — it measures **detection box stability under pixel
  noise**, a strictly narrower use than the corpus's own possible original
  purpose (which may include recognition benchmarking) — but "narrower use
  than the license permits" does not substitute for reading the license.
- If the corpus is ever regenerated, re-hosted, or shared alongside a
  published result (a portfolio write-up, for instance), the corpus's own
  redistribution terms — not merely this project's own preferences — govern
  whether that is permitted, exactly as §B.4 already states for storage.
- No EU/GDPR-specific analysis is performed here — this is a research/
  portfolio project with no user-facing deployment collecting personal
  data from real users; the corpus is an **input to an offline measurement
  script**, not personal data this platform processes or stores about
  anyone the platform serves. This distinction should be re-checked if the
  corpus or its results are ever incorporated into anything shipped to
  real users.

---

## C. R13 measurement execution

### C.1 Exact commands

```bash
# Single run, default seed (0)
export DFP_SCRFD_MODEL_PATH=/path/to/buffalo_l_extracted/det_10g.onnx   # section A.3
export DFP_SCRFD_STUDY_PHOTOS=/path/to/photo_corpus                     # section B
python tests/scrfd_detector_study.py

# Two-seed cross-check, per docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md section 5.4 -
# mandatory before treating any result as Phase-4-gating evidence
DFP_SCRFD_STUDY_SEED=0 python tests/scrfd_detector_study.py | tee run_seed0.txt
DFP_SCRFD_STUDY_SEED=1 python tests/scrfd_detector_study.py | tee run_seed1.txt
```

### C.2 Required inputs

1. `$DFP_SCRFD_MODEL_PATH` — the extracted, digest-recorded SCRFD asset
   (§A).
2. `$DFP_SCRFD_STUDY_PHOTOS` — the acquired, manifest-recorded corpus (§B).
3. **The exact `confidence_threshold`/`nms_iou_threshold` values Phase 4
   intends to ship.** `tests/scrfd_detector_study.py`'s own `_policy_for()`
   currently hardcodes `confidence_threshold=0.5, nms_iou_threshold=0.4` —
   these are Phase 2's own testing convention, **not yet a ratified
   production choice** (`docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` §4.2's own
   caveat, restated because it is load-bearing here specifically): if
   Phase 4 ships different values, the measurement must be re-run against
   those values, not merely re-cited from a run against different ones.
4. `$DFP_SCRFD_STUDY_SEED` (optional, default `0`) — set explicitly for
   the second cross-check run.

### C.3 Expected outputs

`report()`'s printed structure (unchanged from this session's own prior
implementation of the corrected methodology):

- Images measured, realisations per image, total trials (N×K).
- Clean detection failures (tracked separately from noise-induced losses).
- Faces lost under noise, out of total trials.
- The rule-of-three 95% upper bound on the true per-trial loss rate.
- Max short-edge move percentage, with its 95% bootstrap CI.
- The amplification point estimate, with its 95% bootstrap CI.
- Four `[PASS]`/`[FAIL]` lines against the acceptance criterion.
- A final `CLEARS`/`DOES NOT CLEAR` verdict line.

### C.4 Evidence package structure

**Recommended, not created by this document** (a future implementation
session's deliverable, once P1–P3 are met): a single `docs/
SCRFD_R13_MEASUREMENT_EVIDENCE.md` containing, at minimum:

1. Date, operator, and the asset's recorded SHA-256 (§A.3) and corpus
   manifest identity (§B.5) actually used.
2. The full `report()` output for **both** seed runs (§C.1), verbatim.
3. An explicit cross-seed consistency check: both runs' amplification CIs
   overlap substantially, and both reach the same PASS/FAIL verdict on
   every check — a documented, honest comparison, not merely "ran twice."
4. The final pass/fail determination, with the amplification check's basis
   stated explicitly (CI upper bound, not point estimate — §C.6).
5. A sign-off line naming who reviewed the evidence before it was treated
   as a go/no-go input to Phase 4.

This mirrors the structure of `docs/SCRFD_ASSET.md` for the asset itself
(a recorded-once, human-reviewed evidence file) and gives a future reader
— including a future session picking this project back up — a single place
to verify the measurement actually happened, rather than having to
re-derive it from raw script output that may not have been archived.

### C.5 Acceptance criteria

Verbatim, `REPRODUCIBILITY_AUDIT.md:1516-1518` / `docs/
INFERENCE_DETERMINISM.md:292-298`:

> Under ±4 LSB uniform noise on real photographs, across ≥ 200 images: no
> face lost, and the box short edge moving ≤ 1%. Amplification below 10×,
> from the measured 310×.

### C.6 Confidence interval interpretation

Per `docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md` §4.4, re-confirmed by this
session's own hermetic tests of the aggregation logic:

| Statistic | What the CI means here | Gate basis |
|---|---|---|
| Amplification | 95% cluster-bootstrap CI (resampling whole images, not individual trials — realisations within one image are correlated) | **The CI's upper bound must be < 10×** — a point estimate of 9.8× with a CI of (6.1, 14.3) does not clear the bar; this is a deliberate strengthening of the literal criterion text, matching this project's own precedent (`evaluation/thresholds.bootstrap_threshold_ci`) for a less consequential decision |
| Max short-edge move | Same cluster-bootstrap CI, reported as a diagnostic | **Gate on the observed point value** (≤ 1%), not the CI — a max is driven by its worst observation; loosening a worst-case ceiling using a CI would weaken exactly the guarantee this sub-criterion exists to give |
| Face-loss rate | No CI (a zero-tolerance bound); the rule-of-three 95% upper bound on the true per-trial loss rate is reported alongside the observed zero | **Gate on zero observed losses**, with the rule-of-three bound reported so "zero across N×K trials" has a stated confidence level, not left implicit |

### C.7 Pass/fail decision framework

Condensed from `docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md` §8's own go/no-go
diagram, re-verified consistent with this session's implementation:

```
Both assets present (section A, section B)?
  NO  -> report missing inputs, exit 0, no verdict (current, correct behavior)
  YES -> run the study script, >= 2 seeds (section C.1)
           |
           v
     All three checks clear (section C.5), amplification gated on CI upper bound (section C.6)?
       YES -> gather the other three evidence categories section D names
              (plumbing regression, quality proxies, latency) before
              approving cutover - R13 alone is necessary, not sufficient
       NO  -> does it fail ONLY on the amplification CI upper bound,
              marginally, with face-loss and short-edge checks clean?
                YES -> consider re-measuring against SCRFD_2.5G_KPS
                       (the named fallback, Decision 1) - if that also
                       fails, full no-go
                NO  -> full no-go; Phase 4 does not proceed
```

**No branch allows "ship anyway" on a failed measurement** — restated
because it is this project's own standing, repeatedly-cited rule (`docs/
SCRFD_PROGRAM_DESIGN.md` §9.3, `docs/SCRFD_PHASE2_SPEC_RESOLUTION.md` §11,
`docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` §13, and this document, four
independent citations of the same rule, not four different rules).

---

## D. Cutover readiness checklist

### D.1 Objective checklist

Each item independently checkable, no exceptions:

1. [ ] P1 — License read and consciously accepted, recorded in `docs/
   SCRFD_ASSET.md` (§A.2).
2. [ ] P2 — Asset fetched; both SHA-256 values recorded; detector member's
   filename confirmed (§A.3).
3. [ ] P3 — Corpus acquired; its own license/provenance and manifest
   recorded (§B.3, §B.5).
4. [ ] P4 — `tests/scrfd_detector_study.py` run twice (different seeds)
   against the shipped `confidence_threshold`/`nms_iou_threshold` values;
   **all three checks clear**, amplification gated on the CI upper bound;
   evidence package produced (§C.4).
5. [ ] P5 — Synthetic default-fixture mitigation implemented and tested,
   so `python tests/run_all.py` remains runnable with no external asset
   present (`docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` §4.6/§7.4) — independent
   of P1–P4, §7 below.
6. [ ] `verify_scrfd_asset_at_startup()` implemented and tested (`docs/
   SCRFD_PHASE4_CUTOVER_DESIGN.md` §4.7).
7. [ ] The widened collateral test-content re-derivation list (`docs/
   SCRFD_PHASE4_CUTOVER_DESIGN.md` §7.3, 10+ files) is complete.
8. [ ] `python tests/run_all.py` is green, full suite, after cutover code
   lands.
9. [ ] Every version-literal collateral pin (`.f.`, `STUB_ENGINE_VERSION`)
   is updated and swept (`grep -rn "f8244ad0"` returns nothing outside
   historical narrative text).
10. [ ] The other three evidence categories `docs/SCRFD_PROGRAM_DESIGN.md`
    §6 named (structural/plumbing regression via `paired_bootstrap`/
    `regression_gate`; detection-quality proxies on a small labeled set;
    latency/throughput) are gathered — R13 alone is necessary but not
    sufficient for full cutover approval, restated from that document's
    own §7.

### D.2 Go/no-go criteria

**Go**, only if items 1–10 above all hold. **No-go**, unconditionally, if
item 4 (P4) fails — restated because it is the one item every SCRFD
document treats as non-negotiable (§C.7).

### D.3 Evidence required for approval

The single evidence package (§C.4) for R13, plus a completed, dated copy
of §D.1's checklist with each item's own supporting artifact named (the
`docs/SCRFD_ASSET.md` entry for items 1–2, the corpus manifest for item 3,
the evidence package for item 4, the relevant test file/PR for items 5–9,
and a second, separate evidence note for item 10's three categories). No
single document currently bundles all of this — this checklist is the
bundling mechanism this plan recommends, not a new document this plan
creates.

---

## 3. What can be completed now, without external assets

Purely engineering, zero dependency on P1–P3:

- **P5 — the synthetic default-fixture mitigation** (`docs/
  SCRFD_PHASE4_CUTOVER_DESIGN.md` §4.6/§7.4). Uses the same
  hand-built-ONNX-graph technique `tests/test_scrfd_detector.py` already
  demonstrates — no licensed weights needed.
- **`verify_scrfd_asset_at_startup()`** and its tests (checklist item 6) —
  can be written and tested against the synthetic fixture; its production
  behavior (raising on a missing/wrong real asset) is testable with a
  deliberately-wrong path, no real asset required.
- **The evidence-package template** (§C.4) and the checklist itself (§D) —
  documentation scaffolding, ready to fill in once P1–P4 produce real
  content.
- **`docs/NEXT_ENGINEERING_PROGRAM.md` §0's own housekeeping items**
  (committing the working tree, syncing `REPRODUCIBILITY_AUDIT.md`'s
  narrative trailer, fixing stale "Stop here" gate language in three
  documents — all named in `docs/PROJECT_STATUS_FINAL.md` §7/§4) — entirely
  unrelated to SCRFD's own asset gate, but still open, and zero-dependency.
- **The widened collateral test-content re-derivation** (checklist item
  7) — much of this work is re-deriving assertions against whatever the
  **synthetic** default fixture returns (P5), not the real asset, so it
  can proceed once P5 exists, in parallel with P1–P4.

## 4. What requires external assets (cannot be engineered around)

- **P1/P2 — the real, licensed SCRFD ONNX asset.** No amount of code or
  design substitutes for a human accepting a license and downloading a
  specific file from a specific upstream source.
- **P3 — the real, ≥ 200-image photo corpus**, with its own license read.
- **P4's actual numeric result** — the amplification, face-loss, and
  short-edge figures do not exist without P1–P3's real inputs; the
  *methodology* that will compute them is done (§1.1), but methodology is
  not evidence.
- **Checklist item 10's quality-proxy/latency evidence** — needs the real
  asset (P1/P2) at minimum; the labeled validation set for quality proxies
  is a further, separate small acquisition (`docs/SCRFD_PROGRAM_DESIGN.md`
  §6 point 3), not designed further here.

## 5. What remains blocked regardless of engineering effort

- **The measurement's actual outcome (pass or fail).** This document, or
  any amount of further design work, cannot predict or influence whether
  the real SCRFD asset clears R13's criterion against a real corpus — that
  is an empirical fact about the asset and the corpus, not a property of
  this project's code.
- **The licensing decision itself (P1).** A judgment call about this
  project's own goals (research/portfolio use today vs. any future
  commercial intent) — more engineering cannot resolve a legal/business
  question; only a human decision can (§A.2, §A.6).
- **The eventual production asset-provisioning mechanism.** No Dockerfile,
  no CI execution, no hosting target exists (`docs/
  SCRFD_PHASE4_CUTOVER_DESIGN.md` §4.5, re-confirmed unchanged) — this is
  Production Deployment program territory (`docs/
  NEXT_ENGINEERING_PROGRAM.md` §4), genuinely a different program with its
  own prerequisites, not something this plan's scope can pull forward.
- **Corpus acquisition lead time**, if the chosen source (§B.3) turns out
  to have access friction (registration, request forms, restricted
  distribution) — unknowable until a specific source is actually pursued,
  the same honest "unbounded" framing every prior SCRFD document has given
  its own acquisition-lead-time items.

---

## 6. Version-impact re-evaluation: does *completing the prerequisites themselves* move anything?

**Answered directly, re-derived rather than assumed: no. Completing P1–P5
moves zero version axes.** This is not a restatement of `docs/
SCRFD_PHASE4_CUTOVER_DESIGN.md` §5 (which analyzes the **cutover
implementation's** impact, a later, separate event) — it is a fresh
question about the **prerequisite-completion work itself**, and the answer
follows from a different, narrower argument:

| Axis | Current value | Moved by completing P1–P5? |
|---|---|---|
| `CONTRACT_VERSION` | `"1.0.0"` | **No.** Fetching an asset, acquiring a corpus, and running a script are not code changes to any wire type. |
| `RUN_SCHEMA_VERSION` | `"1.2.0"` | **No.** `tests/scrfd_detector_study.py` never constructs or stores a `BenchmarkRun`. |
| `CALIBRATION_WITNESS_VERSION` | `"1.3.0"` | **No.** Nothing in P1–P5 fits or stores a `CalibrationWitness`. |
| `LOCK_VERSION` | `"1.1.0"` | **No — structurally unrelated**, unchanged finding repeated in every SCRFD document because it remains true: no code path connects this work to `evaluation/split_lock.py`. |
| `.d.` | `c9d4025c` | **No.** None of P1–P5 touches `inference/decode_env.py` or any decode-environment fact. |
| `.f.` | `f8244ad0` | **No — this is the one answer worth deriving carefully, not just asserting.** `tests/scrfd_detector_study.py`'s `_policy_for()` constructs its own **local** `FaceSelectionPolicy` via `dataclasses.replace(pre.default_face_selection(), detector="scrfd-10g-kps", ...)` — confirmed by reading the function this session — which **never calls `default_face_selection.cache_clear()`** and never mutates the shipped, cached policy. `default_face_selection()` itself, and therefore the shipped `.f.` value, is untouched by running the study script against any asset or corpus, real or synthetic. **`.f.` only moves at the actual Phase 4 cutover** (`default_face_selection()`'s own code changing, per `docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` §5) — a distinct, later, separately-gated event this plan's scope does not include. |
| `STUB_ENGINE_VERSION` | `"...f.f8244ad0.d.c9d4025c"` | **No**, for the same reason as `.f.` — it is derived from `default_face_selection()`'s identity, which P1–P5 never touch. |

**This matches, and extends by direct analogy, `docs/
SCRFD_PHASE3_MEASUREMENT_PLAN.md` §9's own conclusion** ("Phase 3 is pure
measurement... a measurement script cannot move a version literal it never
writes to") — that document analyzed running the study script; this
document additionally confirms that *acquiring the inputs* to run it (P1–P3)
and *building the test-infrastructure mitigation* (P5) are equally
inert on every version axis, for the same structural reason: none of it is
a change to `inference/preprocess.py`'s `default_face_selection()` or any
other identity-emitting function.

---

## 7. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Licensing risk is treated as resolved by variant choice | **None available — structural** (§A.1, §A.6) | State plainly, as this document and `docs/SCRFD_ASSET_AUDIT.md` §5.3 both do: no SCRFD variant avoids it |
| The asset is fetched into a shared/commercial/uncontrolled environment before P1's acceptance is actually recorded | High if it happens | §A.2's ordered workflow — acceptance before fetch, every time, re-checked per new environment |
| No independent upstream checksum exists for the model asset (TOFU only) | Medium, accepted residual | Named explicitly, not silently assumed away (§A.5, unchanged from Decision 3) |
| The evaluation corpus's own license is not read before use | Medium, non-technical | §B.3's explicit gate — same discipline as the model asset |
| The corpus under-samples the size/lighting/pose regimes R13's mechanism depends on | Medium — could produce a misleadingly clean or bad result | §B.2's diversity table, tied to the measured mechanism, not generic |
| The corpus has no recorded manifest, so a later re-run cannot confirm it used "the same" corpus | Medium, newly identified (§B.5) | This document's own added recommendation — record a manifest, mirroring the asset's own TOFU discipline |
| A single-seed measurement is treated as definitive | Low once §C.1's two-seed check is followed | Explicit in the exact commands given here |
| The measurement is run against placeholder `confidence_threshold`/`nms_iou_threshold` values, then different values ship in Phase 4 | **High if it happens** — the shipped configuration would be unmeasured | §C.2 item 3's explicit caveat, restated from `docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` §4.2 |
| No evidence package is archived, so a future session cannot verify the measurement happened at all | Medium, process risk (this project's own documented history, `docs/PROJECT_STATUS_FINAL.md` §7, shows gate-language and record-keeping drift has happened before) | §C.4/§D.3's evidence-package requirement |
| P5 (synthetic-fixture mitigation) is skipped or deferred until cutover implementation begins | High — the entire suite goes red the moment `default_face_selection()` flips, per `docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` §3 | Named here as independently startable now (§3), not contingent on P1–P4 |
| N10 (unpinned `onnxruntime` version, `docs/SCRFD_PHASE2_DESIGN.md` §10 item 7) affects cross-machine reproducibility of the measurement itself | Low today, real once the measurement is treated as authoritative across more than one machine | Unresolved by this plan; flagged as it was flagged before — should be closed before or alongside P4 if the measurement is run on more than one machine |
| Corpus acquisition has unknown lead time (registration, access requests) | Unbounded, non-engineering | Named honestly rather than estimated away (§5) |

---

## 8. Effort estimate

| Item | Estimate | Basis |
|---|---|---|
| P1 — license read + accept + record | Hours (human, same-day) | `docs/SCRFD_ASSET_AUDIT.md` §2's text is already excerpted in `docs/SCRFD_ASSET.md` — no new research needed, only a decision |
| P2 — fetch + hash + verify | 0.5 day | `docs/SCRFD_ASSET.md` §3's procedure is already fully specified (§A.3 above) |
| P3 — corpus acquisition | **Hours to a day if WIDER FACE's validation split clears licensing (§B.3); otherwise unbounded** | Depends entirely on which source is actually pursued — not estimable further from this repository's own evidence |
| P4 — run measurement, 2 seeds | Minutes of compute once P1–P3 are met (≈ 6,600 detector calls at 300×11×2, `docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md` §3.1's own latency estimate) + 0.5 day to write the evidence package (§C.4) | |
| P5 — synthetic-fixture mitigation | 1–2 days | `docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` §12's own estimate, unchanged, since this plan adds no new scope to it |
| `verify_scrfd_asset_at_startup` + tests | 0.5 day | Same estimate as the cutover design document |
| Collateral test re-derivation (10+ files) | 3–6 days | Unchanged from `docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` §12 |
| **Total, engineering-only (P5 + checklist items 6–9)** | **~5–9 days**, startable immediately, zero external-asset dependency | |
| **Total, including P1–P4** | **~1–2 engineering days + P3's unbounded acquisition lead time** | The only genuinely unbounded item is corpus acquisition; everything else is now a bounded, checklist-driven task |

---

## 9. Dependency graph

```
P1 (license accept)
    |
    v
P2 (fetch + hash)  --------\
                             \
P3 (corpus acquire) ---------+---> P4 (real measurement, 2 seeds)
                                        |
                                        v
                              Evidence package (C.4)
                                        |
                                        v
                          Checklist item 10 (plumbing/quality/latency)
                                        |
                                        v
                          Full cutover readiness (D.1, all items)
                                        |
                                        v
                          Phase 4 implementation gate

P5 (synthetic-fixture mitigation)  ---- independent, zero dependency on
    |                                    P1-P4 - can start today
    v
verify_scrfd_asset_at_startup + tests -- depends only on P5 existing
    |
    v
Collateral test re-derivation (7.3 list) -- depends on P5's fixture design
                                             being settled, not on P1-P4

Housekeeping (NEXT_ENGINEERING_PROGRAM section 0) -- fully independent of
                                                       everything above
```

**Critical path is P1 → P2 → P4** (P3 runs in parallel with P1/P2 and only
needs to complete before P4). **P5's branch is the only track that can
finish entirely before P1 even starts**, and doing so de-risks the eventual
cutover regardless of when the licensing/corpus track clears.

---

## 10. Implementation gate recommendation

**Do not begin Phase 4 cutover implementation.** P1–P4 are all unmet
today (§1.2), and per this document's own §C.7/§D.2, a missing measurement
is treated identically to a failed one — no branch of the decision
framework permits proceeding without it.

**Do begin, immediately, with no further approval gate needed beyond this
document:**

1. **P5** — the synthetic default-fixture mitigation (`docs/
   SCRFD_PHASE4_CUTOVER_DESIGN.md` §4.6/§7.4) — zero dependency on
   licensing or corpus acquisition, de-risks the largest failure mode
   (F3, the full-suite blast radius) identified in the cutover design,
   and is pure engineering start-to-finish.
2. **`docs/NEXT_ENGINEERING_PROGRAM.md` §0's housekeeping** — committing
   the working tree, doc-sync, stale gate-language fixes — fully
   independent, already overdue per `docs/PROJECT_STATUS_FINAL.md` §7.

**The next human decision point, and the one this document cannot make
for anyone, is P1** — reading and accepting (or explicitly declining) the
license. That single decision unblocks P2, and by extension P4 and the
whole measurement gate. **P3 (corpus) should start in parallel with P1/P2**,
not after — it has its own, unrelated lead time and no reason to be
sequenced behind the asset decision.

**No code was written to produce this document. Stop here.**
