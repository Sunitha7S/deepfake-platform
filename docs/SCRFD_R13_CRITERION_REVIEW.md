# R13 Acceptance Criterion — Review After the First Real Measurement

**Status:** review only. The current gate is unchanged and the measured
result stands: **NO-GO, Haar stays the shipped default**
(`docs/SCRFD_R13_MEASUREMENT_EVIDENCE.md`). Every figure below is computed
from that run's per-trial evidence (two seeds, 300 images × 11 noise
realisations each). Nothing here is applied to the gate without an explicit
decision, and nothing here may be confirmed on the same 300 images (§5).

## 1. Amplification: pooled versus within-image

**What the gate computes.** `std(free) / std(fixed)` over every trial pooled
across all images. Each image has its own baseline score, so the pooled
spread is `between-image variance + within-image (noise) variance`, and the
between-image term enters numerator and denominator identically.

**What happened.** Between-image variance was **100.0% / 99.9%** of the fixed
/ free pooled spread, so the ratio read **1.00× (CI 1.00–1.00)** in both runs —
a pass that carries no information about the detector. A test in
`tests/test_scrfd_measurement_methodology.py` reproduces this: a synthetic
corpus with a true 50× within-image amplification reads ~1× pooled.

**Why within-image is the faithful statistic.** The original 310× was measured
on one fixture, where between-image variance is zero by construction. The
within-image statistic (pooled within-group standard deviation, free over
fixed) removes that term and equals the pooled one when there is a single
image. Measured: **26.7× (CI 14.2–38.8) and 25.6× (CI 12.9–38.4)**.

**A second, separate effect.** In ~1% of trials noise changes *which* face
ranks first (a selection instability: the verdict would describe another
person). Excluding those trials, within-image amplification is **10.5× /
11.6×** — localisation jitter alone.

**Proposed alternative.** Gate on the within-image statistic's 95% CI upper
bound (< 10×), and report the top-face switch rate as its own gated or
tracked figure rather than letting it hide inside amplification. On the
current evidence this would still fail (upper bound ~38×; ~11× point estimate
without switches).

## 2. Box movement and integer pixels

**What the gate computes.** `max` over all trials of
`|Δ short edge| / short edge ≤ 1%`.

**The issue.** Boxes are integer pixels. For any face with short edge under
100 px, a single-pixel change already exceeds 1%, so the criterion as written
requires *zero* movement on small faces. In the 32–64 px band, 18–20% of trials
move exactly 1 px; in the ≥ 256 px band, 40%.

**But quantisation is not the whole story.** Alternatives, evaluated on the
same evidence (seed 0 / seed 1):

| Rule | Result | Outcome |
|---|---|---|
| Current: max ≤ 1% | max 27.3% / 25.4% | fail |
| Pixel-aware: every trial ≤ max(1 px, 1%) | 4.22% / 3.81% of trials violate it; max 18 px | fail |
| Tail: 95th percentile ≤ 2% | p95 1.96% / 1.96% | pass, barely |
| Tail: 99th percentile ≤ 2% | p99 5.7% / 5.9% | fail |

**Proposed alternative.** Replace the bare maximum with a quantisation-aware,
pre-registered tail rule — e.g. "the fraction of trials moving more than
max(1 px, 1%) has a 95% upper bound below *p*%", choosing *p* before the next
measurement. A worst-case maximum over thousands of trials is dominated by
single outliers and cannot be satisfied by any detector with integer output;
a bounded tail rate can be stated, measured and confirmed. The outliers
(up to 18 px) should be inspected before *p* is chosen.

## 3. Face loss near the confidence threshold

**What the gate computes.** Zero faces lost across all trials.

**What happened.** All 5 losses (2 + 3 of 6,402 trials) are one 84×90 face
whose detector score lies in [0.50, 0.52), just above the shipped
`confidence_threshold` of 0.5. Any thresholded detector has faces near its
threshold; a zero-tolerance rule fails whenever the corpus contains one.

**Exact 95% upper bounds on the per-trial loss rate** (Clopper–Pearson):
0.20% (seed 0), 0.24% (seed 1), 0.16% pooled.

**Proposed alternatives** (pick before re-measuring):
- **A rate bound instead of zero:** "95% upper bound on per-trial loss below
  *q*%" (the plan already reports the rule-of-three bound alongside zero).
- **A margin rule:** count a loss only for faces whose clean score is at
  least threshold + δ. This needs `SCRFDDetector` to expose detection scores,
  which it currently does not; faces inside the margin are then reported as a
  separate near-threshold rate.
- **A different threshold** is a configuration change, not a criterion
  change, and must be confirmed on a fresh corpus (§5).

## 4. What would and would not pass today

Even with all three alternatives adopted, the current configuration would not
clear: within-image amplification is above 10× with or without top-face
switches, and ~4% of trials move more than max(1 px, 1%). The revised
criteria would make the gate *meaningful*, not easier to pass on this
evidence.

## 5. Validation protocol for any revision

1. **The 300-image corpus (manifest `c5907efe…`) is now development data.**
   Anything decided after looking at it — a new statistic, threshold, tail
   level *p*, loss rate *q*, margin δ — must not be confirmed on it.
2. **Pre-register** the revised criterion and configuration in writing
   before measuring.
3. **Confirm on a fresh, disjoint corpus:** e.g. 300 images selected by
   `tests/scrfd_corpus.py` from WIDER FACE validation images *not* in the
   development manifest (2,520 qualifying images remain), or from the WIDER
   training split. Record its manifest digest before running.
4. Run two seeds and archive both JSON evidence files, as in the first run.

## 6. Licence scope (unchanged, unresolved)

The model package grants "non-commercial" use without defining it; WIDER FACE
is CC BY-NC-ND 4.0. Local evaluation is within both. Whether a public
portfolio used in a job search counts as non-commercial is **not** settled by
either licence text and is not claimed here. Weights, images, crops and
noised variants are not committed or redistributed; only aggregate figures
are published.
