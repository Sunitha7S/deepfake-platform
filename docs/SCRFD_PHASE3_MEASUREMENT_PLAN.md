# SCRFD Integration Program, Phase 3 — Measurement Plan

**Type:** measurement/statistical planning only — no code. **Basis:** a
fresh re-derivation this session of `docs/SCRFD_PROGRAM_DESIGN.md`, `docs/
SCRFD_PHASE2_DESIGN.md`, `docs/SCRFD_PHASE2_SPEC_RESOLUTION.md`, `docs/
SCRFD_ASSET.md`, `docs/PROJECT_ROADMAP.md`, `REPRODUCIBILITY_AUDIT.md`, and
`tests/scrfd_detector_study.py` — `git status` re-checked first, unchanged
since Phase 2 landed. This document also reads `docs/
INFERENCE_DETERMINISM.md` §5/§5A, the design document behind R13's original
310× figure, which none of the four SCRFD documents had cited beyond its
headline number — reading it surfaces a real, load-bearing methodology gap
in the shipped `scrfd_detector_study.py` (§2, below), found by re-deriving
rather than assuming that document's own citation of it was complete.

**No code was written to produce this document, per instruction.**

---

## 1. The R13 acceptance criterion, re-derived from source

Verbatim, `REPRODUCIBILITY_AUDIT.md:1516-1518`:

> **SCRFD acceptance criterion.** Under ±4 LSB uniform noise on real
> photographs, across ≥ 200 images: no face lost, and the box short edge
> moving ≤ 1%. Amplification below 10×, from the measured 310×.

A second, slightly more operational phrasing of the same criterion exists
at its point of origin, `docs/INFERENCE_DETERMINISM.md:292-298`:

> Under ±4 LSB uniform noise on real photographs, across ≥ 200 images: no
> face lost, and the box's short edge must not move by more than 1%.
> Re-run `inference_determinism_study.py` stage 3 against SCRFD and require
> the amplification factor below 10× (from the measured 310×).

The second phrasing is the operative one for **how** to measure, not just
**what** the bar is — it names the *procedure* (re-run the same stage
against SCRFD), not only the three numbers. §2 below is what re-reading
that procedure against the actually-shipped `scrfd_detector_study.py`
found.

---

## 2. Critical finding: the shipped study script does not compute the criterion's own statistic

### 2.1 What the original 310× figure actually measured

`docs/INFERENCE_DETERMINISM.md` §5.1–§5.2, §5A(c), re-read in full this
session:

- **One synthetic fixture, ±3 LSB noise, 11 independent noise
  realisations** (`:239`, `:255-262`) — *not* 200+ different images with
  one noise draw each.
- **Amplification is a ratio of two spreads computed within that one
  fixture, across those same 11 realisations**, formalised at `:341-347`:

  ```
  A = spread(p | crop re-detected) / spread(p | crop fixed)
    = 0.091022 / 0.000293 = 310.6
  ```

  *"Both spreads are computed on the same pixels with the same scorer, so
  everything but the crop position is held fixed"* (`:349`). The
  **numerator** is `p`'s spread across the 11 realisations when each
  realisation's *own* re-detected box crops it; the **denominator** is `p`'s
  spread across the same 11 realisations when every crop is held at the
  *pre-noise* box. Same images (11 noisy copies of one fixture), same
  scorer, the only thing that differs between numerator and denominator is
  which box position selects the pixels `surrogate_logit` sees.
- `docs/SCRFD_PHASE2_DESIGN.md` §2.8 step 3 already states this correctly
  as the instruction for Phase 3's tool to follow: *"compute the spread
  exactly as the original 310× measurement did"* — the design intent was
  right; §2.2 below shows the implementation did not follow it.

### 2.2 What `tests/scrfd_detector_study.py` actually computes (re-read live, this session)

`measure()` (`:122-176`): for each of the **N different real photographs**,
draws **one** noise realisation, computes one fixed-crop score and one
free-crop score, and after the loop:

```python
fixed_spread = float(np.std(fixed_crop_logits))   # std across N different images
free_spread = float(np.std(free_crop_logits))     # std across N different images
amplification = free_spread / fixed_spread
```

This is the ratio of two spreads **across the corpus's own content
diversity** (one draw per image), not the ratio of two spreads **induced
by noise on repeated realisations of the same content**. The two are not
the same statistic, and they are not comparable to "310×" for a structural
reason: different real photographs have genuinely different baseline
`surrogate_logit` values (different subjects, lighting, framing) — a `std`
taken across 200 different photographs is dominated by *which photograph it
is*, not by *how much a ±4 LSB perturbation moved that photograph's own
score*. The original 310× figure isolates the second quantity by holding
content fixed and varying only the noise draw; the shipped script's
`np.std(fixed_crop_logits)`/`np.std(free_crop_logits)` conflates it with
cross-photograph diversity.

**This is not a cosmetic difference.** A corpus with more visual diversity
would inflate *both* spreads somewhat, but there is no guarantee the ratio
tracks noise-induced instability at all once cross-image variance
dominates — the resulting number could clear 10× for a detector that is
actually unstable, or fail to clear it for one that is actually stable,
depending on how diverse the 200 photographs happen to be. The number this
script currently prints is not evidence about R13 either way.

### 2.3 Consequence for this plan

**`tests/scrfd_detector_study.py` needs a methodology amendment before its
output can be trusted as the Phase 4 gate.** This plan does not make that
amendment (no code in this task) — it specifies exactly what the
corrected procedure must compute (§4) so that amendment is a mechanical
implementation of an already-fully-specified statistic, not a new design
decision at implementation time. Everything else the script does today —
input resolution (`$DFP_SCRFD_MODEL_PATH`/`$DFP_SCRFD_STUDY_PHOTOS`),
graceful no-verdict exit when inputs are missing, per-trial face-loss and
box-movement bookkeeping, the honest "this validates stability, not
quality" framing — is correct and unaffected; only the amplification
statistic's aggregation needs correcting.

---

## 3. Minimum dataset requirements

### 3.1 Sample size

**Floor: N ≥ 200 images — not negotiable, it is the criterion's own
wording** (§1). This plan additionally specifies **K ≥ 11 independent noise
realisations per image** (§4.1) — matching the original measurement's own
realisation count exactly, so a direct, apples-to-apples comparison to
"310×" is possible, not merely a differently-shaped number that happens to
also be called "amplification."

**Why K matters as much as N, statistically:** the "no face lost" criterion
is a zero-tolerance bound. Observing zero losses across only N=200
single-draw trials gives a rule-of-three 95% upper confidence bound on the
true per-trial loss rate of about 3/200 ≈ **1.5%** — a real, and fairly
loose, ceiling on what "no face lost" actually certifies. With N=200 images
× K=11 realisations = **2,200 trials**, the same rule gives 3/2200 ≈
**0.14%** — an order of magnitude tighter, for the same 200-image floor the
criterion already requires. K is not optional padding; it is what makes
"no face lost across ≥200 images" mean something closer to what it sounds
like it means, and it is simultaneously required to compute the
amplification statistic correctly at all (§2).

**Recommended target, stated as a recommendation, not a new floor:** N=300
(50% headroom above the floor, cheap given SCRFD's own measured per-image
CPU latency — single-digit-to-low-teens milliseconds per `docs/
SCRFD_ASSET_AUDIT.md` §1.2, so 300×11×2 ≈ 6,600 detector calls is a
few-minutes run, not an infrastructure problem) × K=11, chosen so a
transcription error or a handful of unusable files does not silently drop
the corpus below the 200-image floor the criterion actually requires.

### 3.2 Image diversity requirements

Tied to R13's own measured mechanism, not a generic "representativeness"
checklist invented for this document: `docs/INFERENCE_DETERMINISM.md`
§5.2's own finding is that the amplification is driven by **box *size*
changes near a detection boundary**, not position — *"a 5 px size change
moves \|dp\| to 0.091, a factor of 110... box size changes the resample
ratio, which rewrites exactly the high-frequency texture a forensic
detector keys on"* (`:264-270`). The corpus must therefore actually exercise
that mechanism, not merely be "real photographs" in name:

| Dimension | Requirement | Why (tied to the mechanism, not generic) |
|---|---|---|
| Face-size range | Include faces from just-above `detect_min_px`-equivalent size up to large, frame-filling faces | Box-size sensitivity to noise is a boundary phenomenon (§5.2); a corpus of only large, easy faces under-samples exactly the regime R13 is about |
| Lighting | Low-light and well-lit both represented | Detection confidence margins narrow in low light, which is where a detector is more likely to sit near its own decision boundary and re-detect a differently-sized box under noise |
| Compression/resolution | A realistic spread of JPEG quality levels and source resolutions, not one canonical quality setting | ±4 LSB noise interacts with existing compression artefacts; a corpus of only pristine, uncompressed images tests a condition few real deployments see |
| Pose/angle | Frontal and moderately off-axis faces both represented | SCRFD's claimed advantage over Haar is partly non-frontal robustness (`docs/SCRFD_ASSET_AUDIT.md` §1.2); excluding non-frontal faces would make the measurement easier than the deployment it is meant to gate |
| Single vs. multiple faces per image | Both represented | `detect_faces` publishes `faces[0]` under `face_order_key`'s total order (unchanged by SCRFD, `docs/SCRFD_PHASE2_DESIGN.md` §1.1) — a multi-face image exercises the ranking step under noise, a single-face image does not |

**Explicitly out of this document's scope, flagged rather than silently
folded in:** demographic/skin-tone representativeness is a real, adjacent
concern for a deepfake-detection platform's *overall* fairness, but it is
not what R13's own acceptance criterion measures (a *stability* question,
not an *accuracy-parity* question) — conflating the two would let this
document overclaim what a passing R13 gate certifies. A fairness
evaluation, if this project wants one, is a separate measurement with its
own design, not a rider on this one.

### 3.3 Licensing and provenance — cross-referenced, not re-solved here

`docs/SCRFD_PHASE2_DESIGN.md` §2.8 point 1 already named the photo
corpus's own licensing/provenance question as *"a second, smaller version
of §2.4's asset question"* — unresolved there and unresolved here. This
plan's dataset requirements (§3.1–§3.2) describe what the corpus must
*contain*; **where it comes from, and under what license, is a human
decision this document does not make**, exactly the same posture `docs/
SCRFD_ASSET.md` already takes for the model asset itself. A candidate,
named only as an option and not endorsed: a permissively-licensed public
face-detection benchmark's own validation split (e.g., WidefFace's — the
same corpus the SCRFD paper itself benchmarks against, per `docs/
SCRFD_ASSET_AUDIT.md` §1.2's own table headers) would satisfy §3.2's
diversity requirements by construction, since it is what the accuracy
numbers already cited were measured against — but its own license must be
read before use, the same discipline `docs/SCRFD_ASSET.md` §2 already
applied to the model weights.

---

## 4. Statistical methodology (corrected)

### 4.1 Per-image, per-realisation procedure

For each image `i` in the corpus (`i = 1..N`):

1. Detect on the clean image → `box_0(i)`. If empty, exclude image `i`
   from the amplification statistic (it contributes nothing to either
   spread) but **do not exclude it from the face-loss/box-movement
   accounting** — a clean image the shipped detector cannot find a face in
   at all is itself a relevant, reportable fact, tracked separately (§5).
2. For `k = 1..K` (K ≥ 11, §3.1): draw an **independent** ±4 LSB uniform
   noise realisation and apply it to image `i` → `noisy(i,k)`.
3. Detect on `noisy(i,k)` → `box_k(i)`, or **lost** if empty.
4. **Fixed-crop score:** `surrogate_logit(crop(noisy(i,k), box_0(i)))` —
   the pre-noise box, applied to the noisy pixels. Always computable
   (does not depend on whether realisation `k` lost the face).
5. **Free-crop score:** `surrogate_logit(crop(noisy(i,k), box_k(i)))` — the
   *this-realisation's-own* re-detected box. Undefined when realisation
   `k` lost the face; excluded from the free-crop pool for that trial only
   (§4.2), not imputed.

### 4.2 Aggregation — pooled, not per-image-averaged

Pool every `(i, k)` trial across the **entire corpus** into two arrays:

```
fixed_scores  = [ surrogate_logit(crop(noisy(i,k), box_0(i)))  for all (i,k) with box_0(i) present ]
free_scores   = [ surrogate_logit(crop(noisy(i,k), box_k(i)))  for all (i,k) with box_k(i) present (not lost) ]

amplification = std(free_scores) / std(fixed_scores)
```

**Why pooling, not per-image amplification then averaging:** the original
measurement's own ratio (§2.1) is a ratio of two *marginal* spreads over
one set of realisations, not a per-image ratio subsequently combined — the
faithful generalisation to many images is the same ratio taken over the
union of every image's realisations, not a second layer of averaging that
the original statistic never had. Pooling also keeps the comparison valid:
both `fixed_scores` and `free_scores` are drawn from **the same set of
`(i,k)` noisy pixels**, differing only in which box selects them — exactly
the "everything but the crop position is held fixed" property `docs/
INFERENCE_DETERMINISM.md:349` states for the single-fixture case, preserved
symmetrically at corpus scale because cross-image content variance enters
both pools identically.

### 4.3 Uncertainty — bootstrap confidence interval, reusing this project's own idiom

A single pooled point estimate is exactly the kind of number this
project's own established practice does not ship a threshold-adjacent
decision on without a confidence interval — `evaluation/thresholds.py`'s
`bootstrap_threshold_ci` (`:303-332`) is the precedent: percentile CI over
resamples, `resamples=2000`, `alpha=0.05`, an explicit `seed`. Applied here
with one adaptation the precedent itself does not need: **resample images,
not individual trials** (a cluster bootstrap) — realisations within one
image are correlated (same underlying content), so resampling trials
independently would understate the true uncertainty by treating correlated
observations as independent. Procedure:

1. Resample `N` images **with replacement** from the corpus (each
   resampled image brings its full set of `K` realisations with it, not a
   re-drawn subset).
2. Recompute the pooled `amplification` statistic (§4.2) on the resampled
   set.
3. Repeat 2,000 times (matching `bootstrap_threshold_ci`'s own default).
4. Report `(point_estimate, lo, hi)` at the 2.5th/97.5th percentiles,
   exactly as `bootstrap_threshold_ci` returns `(threshold, lo, hi)`.

The same cluster-bootstrap CI applies to the max-short-edge-movement
statistic (§4.4) for the same correlation reason.

### 4.4 What "clears the criterion" means, stated precisely

| Criterion (verbatim, §1) | Point statistic | Recommended pass condition |
|---|---|---|
| "no face lost" | Trial-level loss count over N×K trials | **Zero losses observed**, and (§3.1) N×K large enough that the rule-of-three upper bound is itself reported alongside the zero, not left implicit |
| "box short edge moving ≤ 1%" | `max` over all N×K trials of `\|short_edge_after − short_edge_before\| / short_edge_before` | **Point-estimate max ≤ 1%.** A single-trial max is not usefully bootstrapped for a *tighter* bound (a max is driven by its worst observation, not reduced by resampling) — report the bootstrap CI on the max anyway (§4.3) as a stability-of-the-worst-case diagnostic, but gate on the observed point value, since loosening a worst-case ceiling using a CI would weaken exactly the guarantee this sub-criterion exists to give |
| "amplification below 10×, from the measured 310×" | Pooled ratio (§4.2) | **Recommended, stated as a strengthening of the literal text, not a rewrite of it:** the **upper bound of the 95% bootstrap CI** (§4.3) must be < 10×, not merely the point estimate. A point estimate of 9.8× with a CI of (6.1, 14.3) is not a number a real cutover decision should treat as "cleared" — the literal criterion text does not say this explicitly, and this plan does not purport to amend `REPRODUCIBILITY_AUDIT.md`'s own wording, but a go/no-go framework (§9) that ignores estimation uncertainty on its own primary gate would be a weaker gate than this project's own precedent (`bootstrap_threshold_ci`) already sets for a less consequential decision (a single threshold, not an architecture cutover) |

---

## 5. Execution procedure for `tests/scrfd_detector_study.py`

### 5.1 Prerequisite — the methodology amendment (§2, §4) must land first

**This plan does not implement it.** Running the script exactly as
currently written and reporting its output as "Phase 3's measurement"
would produce a number that is not the statistic the criterion asks for
(§2.2) — that would be a worse outcome than not measuring at all, because
it would look authoritative without being comparable to "310×." The
concrete, scoped amendment needed, precisely enough that implementing it
is mechanical:

- `measure()` gains a `k_realisations: int = 11` parameter.
- The inner loop draws `k_realisations` independent noise draws per image
  (§4.1), not one.
- `fixed_crop_logits`/`free_crop_logits` accumulate every `(i, k)` trial,
  pooled (§4.2), not one value per image.
- A new `bootstrap_amplification_ci(...)` function, structurally parallel
  to `bootstrap_threshold_ci` (§4.3), resampling images with replacement.
- `report()` prints the point estimate **and** the CI, and gates on the
  CI's upper bound for the amplification check (§4.4).

This is named here as a precise specification so a future implementation
session does not have to re-derive §2/§4 from scratch — it can implement
directly against this section.

### 5.2 Inputs (unchanged from Phase 2's design, re-confirmed live this session)

| Variable | Meaning | Status |
|---|---|---|
| `DFP_SCRFD_MODEL_PATH` | Path to the fetched SCRFD ONNX asset | **Not set** — Phase 0 has not run (`docs/SCRFD_ASSET.md` §"Status") |
| `DFP_SCRFD_STUDY_PHOTOS` | Directory of the photo corpus (§3) | **Not set** — no corpus acquired |

Confirmed by re-running `python tests/scrfd_detector_study.py` this
session with neither set: it prints both as missing and exits 0, exactly
as `main()` (`:212-225`) specifies — no verdict is fabricated.

### 5.3 Exact invocation, once both inputs exist

```bash
export DFP_SCRFD_MODEL_PATH=/path/to/det_10g.onnx   # docs/SCRFD_ASSET.md §1/§3
export DFP_SCRFD_STUDY_PHOTOS=/path/to/photo_corpus  # §3 of this document
python tests/scrfd_detector_study.py
```

Not part of `tests/run_all.py` (unchanged design decision, `docs/
SCRFD_PHASE2_DESIGN.md` §2.7 — a study script, not a regression suite) and
should stay that way: it needs inputs `run_all.py`'s own environment does
not and should not assume exist.

### 5.4 Reproducibility of the run itself

`measure()`'s `rng = np.random.default_rng(seed)` (`:127`, `seed: int = 0`
default) makes one invocation deterministic. For the actual Phase 4-gating
measurement, run it **at least twice with different seeds** (e.g., `seed=0`
and `seed=1`, a `seed` parameter §5.1's amendment should expose on the
command line or via a third environment variable) and confirm both runs'
CIs (§4.3) are consistent with each other — a documented, cheap, honest
check that the bootstrap CI itself is not an artefact of one particular
noise-seed's draw sequence, distinct from the CI's own within-run
uncertainty accounting.

---

## 6. Fail-before / pass-after criteria

Phase 3 has no "before" code state to diff against in the usual sense (it
changes no production file) — the applicable framing is **input-present
vs. input-absent**, and **uncorrected-methodology vs. corrected-methodology**:

| # | Check | Before (today) | After (Phase 3 executed correctly) |
|---|---|---|---|
| 1 | `scrfd_detector_study.py` run with no inputs set | Exits 0, reports both missing, no verdict (confirmed live, §5.2) | Unchanged — this remains the correct behaviour whenever inputs are absent |
| 2 | `scrfd_detector_study.py`'s amplification statistic | Computed across-image, single-draw (§2.2) — **not comparable to 310×** | Computed pooled, within-image-realisation, multi-draw (§4.2) — **comparable to 310×** |
| 3 | Amplification uncertainty | None reported — a bare point estimate | 95% bootstrap CI reported alongside the point estimate (§4.3) |
| 4 | Face-loss accounting | Binary count over N single-draw trials | Count over N×K trials, with the implied rule-of-three bound stated (§3.1) |
| 5 | The four `report()` checks (`>= 200 images`, `no face lost`, `short edge <= 1%`, `amplification < 10x`) | Exist, gate on point values only | Unchanged in name; the amplification check gates on the CI upper bound (§4.4), not the point estimate |
| 6 | A real run against a real asset and a real corpus | Has never happened (no asset, no corpus) | Produces a genuine go/no-go input (§9) — the actual "pass" this whole plan exists to make trustworthy |

---

## 7. Evidence required before approving Phase 4 cutover

R13's own criterion (§1) is the **primary, mandatory gate** — but `docs/
SCRFD_PROGRAM_DESIGN.md` §6 already scoped three further evidence
categories as part of this program's own benchmarking strategy, restated
here as what Phase 4 approval actually requires, not narrowed to R13 alone
just because this document's central task is R13's measurement:

1. **R13's acceptance criterion, corrected methodology, clears (§4.4).**
   Mandatory. Failing this is an unconditional no-go (§9) regardless of
   the other three.
2. **Structural/plumbing regression** via the existing `paired_bootstrap`/
   `regression_gate` machinery (`evaluation/benchmark.py`, `docs/
   SCRFD_PROGRAM_DESIGN.md` §6 point 2) — a Haar-selection `BenchmarkRun`
   against an SCRFD-selection one, both still `StubBackend`-scored,
   confirming the identity move actually separates the two configurations
   in the cache and nothing crashes on a free-aspect box. This is not
   R13's tool and is not built by this plan — flagged as a required,
   separate, already-scoped piece of evidence.
3. **Detection-quality proxies on a small, honestly-labelled validation
   set** (`docs/SCRFD_PROGRAM_DESIGN.md` §6 point 3) — miss rate,
   false-positive rate, box-tightness, Haar vs. SCRFD side by side, tens
   not thousands of images. R13 measures *stability*, not *quality*; this
   is the only evidence in this program that speaks to whether SCRFD is
   actually a better detector, however bounded its scope (`docs/
   SCRFD_PHASE2_DESIGN.md` §2.8's own honest limit: neither this nor R13
   can say anything about the deepfake *verdict*, which needs a real
   scoring model this project does not have).
4. **Latency/throughput measurement** (`docs/SCRFD_PROGRAM_DESIGN.md` §6
   point 4) — SCRFD's forward-pass cost against Haar's cascade, so Phase 4
   does not silently move `contracts/common.py`'s `Timing.preprocess_ms`
   budget unnoticed. Directly informs the `SCRFD_10G_KPS` vs.
   `SCRFD_2.5G_KPS` choice `docs/SCRFD_PHASE2_SPEC_RESOLUTION.md`
   Decision 1 left open pending exactly this measurement.

**What no amount of this evidence can certify, stated once rather than
repeated per item:** whether SCRFD improves the deepfake-detection verdict
itself — that needs Real Model Integration (`docs/
SCRFD_PROGRAM_DESIGN.md` §6, `docs/NEXT_ENGINEERING_PROGRAM.md` §2), a
separate, later program.

---

## 8. Go/no-go decision framework

```
                        Fetch asset (Phase 0) + corpus (§3)
                        both present?
                             │
                    ┌────────┴────────┐
                   NO                YES
                    │                 │
            Report inputs        Run study script (§5.1's
            missing, exit 0      corrected methodology),
            (current, correct    >= 2 seeds (§5.4)
            behaviour)                │
                                       ▼
                          R13 criterion clears (§4.4,
                          all three checks, amplification
                          gated on CI upper bound)?
                                       │
                      ┌────────────────┴────────────────┐
                     NO                                YES
                      │                                  │
        Does it fail ONLY on latency-adjacent      Gather §7 items 2-4
        concerns (amplification CI upper bound      (plumbing regression,
        marginally over 10x) with face-loss and     quality proxies,
        short-edge checks clean?                    latency)
                      │                                  │
           ┌──────────┴──────────┐              All four items support
          NO                    YES              cutover?
           │                     │                       │
    Full no-go.          Consider re-measuring    ┌───────┴───────┐
    Phase 4 does         SCRFD_2.5G_KPS (the      YES             NO
    NOT proceed          named fallback,           │               │
    (docs/               Decision 1). If it    Phase 4        Phase 4 does
    SCRFD_PROGRAM_       also fails, full       APPROVED       NOT proceed —
    DESIGN.md §9.3's     no-go.                 for cutover    name which
    abort condition,                                            item(s) failed,
    unchanged)                                                  re-measure only
                                                                  that item, not
                                                                  a full re-run
```

**No branch in this framework allows "ship anyway"** on a failed R13
measurement — restated from `docs/SCRFD_PHASE2_DESIGN.md` §9's own
abort condition because it applies unchanged here: *"A failed measurement
is a complete, reportable result... not a reason to lower the bar."* The
only branch this plan adds beyond that prior document is the
**variant-fallback** path (`SCRFD_2.5G_KPS`), and only for a specific,
narrow failure shape (amplification marginal, the other two R13 checks
clean) — a face-loss or short-edge failure is a stability defect a smaller,
*less* accurate model is not expected to fix, so that path does not apply
to those failure modes; the framework routes them straight to full no-go.

---

## 9. Version-impact axes — re-verified for Phase 3 itself

Phase 3 is **pure measurement**: it runs `tests/scrfd_detector_study.py`
(a study script, never imported by `tests/run_all.py` or any production
module) against `inference/preprocess.py`'s **already-shipped** Phase 2
code, with `default_face_selection()` untouched throughout. Re-confirmed
this session, not assumed from Phase 2's own conclusion:

| Axis | Current value | Phase 3 impact |
|---|---|---|
| `CONTRACT_VERSION` | `contracts/enums.py:16`, `"1.0.0"` | **None.** No wire-shape code path is touched by running a study script. |
| `RUN_SCHEMA_VERSION` | `evaluation/benchmark.py:108`, `"1.2.0"` | **None**, for R13's own measurement (§1-§6). §7 item 2's `paired_bootstrap`/`regression_gate` evidence *uses* `BenchmarkRun` but does not extend its schema — same "used, not changed" finding `docs/SCRFD_PHASE2_DESIGN.md` §2.8 already made for Phase 2. |
| `CALIBRATION_WITNESS_VERSION` | `evaluation/calibration_protocol.py:74`, `"1.3.0"` | **None.** Nothing in this measurement fits or stores a calibration witness. |
| `LOCK_VERSION` | `evaluation/split_lock.py:81`, `"1.1.0"` | **None — structurally unrelated**, unchanged finding from every prior SCRFD document (dataset split-lock format, no code path connects it to detection). |
| `.d.` | feeds `backend/config.py:41`'s `.d.c9d4025c` segment | **None.** A study-script run reads the decode environment but writes nothing to it; `inference/decode_env.py` is not in this plan's file list because there isn't one. |
| `.f.` | feeds `backend/config.py:41`'s `.f.f8244ad0` segment (current value, post-Phase-2) | **None.** `_policy_for()` (`scrfd_detector_study.py:102-111`) constructs its own **local** `FaceSelectionPolicy` via `replace(pre.default_face_selection(), ...)` — this never calls `default_face_selection.cache_clear()` or otherwise mutates the cached shipped policy; `default_face_selection()` itself, and therefore the shipped `.f.`, is untouched by anything this document specifies. |

**Net: zero identity or schema axes move as a result of Phase 3, by
construction — a measurement script cannot move a version literal it never
writes to.** This is the expected, trivial answer for a phase whose entire
scope is running an existing, already-audited tool against real inputs;
stated formally here because the task asked for it re-verified, not because
re-deriving it surfaced anything Phase 2's own documents had wrong.

---

## 10. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Phase 3 is executed using the **uncorrected** `scrfd_detector_study.py` (§2), producing a number that looks like "amplification" but is not comparable to 310× | **High** — this is the central finding of this document | §5.1 states the amendment as a blocking prerequisite, specified precisely enough (§4.1-§4.3) to implement without further design work |
| The photo corpus is acquired without checking its own license/provenance (§3.3) | Medium, non-technical | Named explicitly, not silently assumed resolved by this document |
| The bootstrap CI (§4.3) is computed by resampling individual trials rather than whole images, understating uncertainty | Medium, subtle | §4.3 specifies the cluster-bootstrap explicitly and states why plain resampling would be wrong |
| A single-seed run is treated as definitive | Low once §5.4 is followed | §5.4's two-seed cross-check |
| The corpus under-samples the size/lighting/pose regimes R13's own mechanism depends on (§3.2), producing a misleadingly clean or misleadingly bad result | Medium | §3.2's diversity table is tied to the measured mechanism, not generic |
| Phase 4 is approved on R13 alone, without §7's other three evidence categories | Medium, procedural | §7/§9 make all four explicit requirements, not an optional checklist |

---

## 11. Gate

**Plan complete. Not ready to execute yet — two prerequisites, both already
named by prior documents and re-confirmed still open this session, block
it:** Phase 0's asset fetch (`docs/SCRFD_ASSET.md`, licensing-gated) and
the photo corpus (§3.3, also licensing-gated). **A third prerequisite this
document itself adds:** `tests/scrfd_detector_study.py`'s amplification
statistic must be corrected per §4.1-§4.3 before any run of it is reported
as evidence — running it as currently written and citing the result would
produce a number that is not the statistic the acceptance criterion asks
for, which is a worse outcome than not measuring yet.

**No code was written to produce this document. Stop here.**
