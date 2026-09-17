# Threshold and abstention band: design, derivation, and operating policy

**Status:** implemented, S8 Batch 7. Closes audit findings **R8** and **R11**.
**Contract impact:** `CONTRACT_VERSION` unchanged at 1.0.0. `EvidenceBundle`
unchanged. `ThresholdInfo` fields unchanged. One *invariant* changed — the band
is now centred in log-odds rather than in probability — and the shipped
operating point moved, which moved `STUB_ENGINE_VERSION`.

Every number in this document comes from `tests/band_study.py`, which is
runnable:

```bash
python tests/band_study.py
```

---

## 1. The two findings

**R8** (open since Batch 1). `band_width = 0.20` was picked, not derived. It was
the largest single unjustified constant in the system. `evaluation.risk`
contained the machinery to derive it and that machinery had never been run
against real scores.

**R11** (recorded in Batch 6). The threshold is *fitted* — `at_target_fpr` on the
calibration split — and the band width was *configured*, independently. Their
product can be unrepresentable. Observed immediately on a realistic fixture: a 2%
false-positive budget over 22 negatives put the threshold at 0.9734, where no
0.20-wide symmetric band fits inside [0, 1], and calibration raised:

```
ValueError: a band of width 0.2 around 0.9734218951277995 runs outside [0, 1]
```

The audit recorded R11 as "the mechanical consequence of R8 going unaddressed."
That is right, and it understates the connection. R8 is not merely unfinished
work; **the quantity R8 asked us to derive could not have a single correct value**,
because a probability width means a different amount of evidence at every
threshold. R11 is what that looks like when the threshold moves far enough.

---

## 2. Why no fixed probability width can work

### 2.1 The impossibility

> **Claim.** Let a band family be a map `B : (t, w) ↦ (lo, hi)` required to
> satisfy
>
> 1. **bracketing** — `0 ≤ lo < t < hi ≤ 1`;
> 2. **probability symmetry** — `t − lo = hi − t`;
> 3. **width independence** — `hi − lo = f(w)` for some `f` not depending on `t`
>    (this is what "a configured width" means: the knob has a fixed meaning).
>
> Then `B` is not total on `(0, 1) × W` for any `W` containing a `w` with
> `f(w) > 0`.

*Proof.* Fix such a `w` and let `c = f(w) > 0`. By (2) and (3),
`t − lo = hi − t = c/2`, so `lo = t − c/2`. For any `t < c/2`, `lo < 0`, which
violates (1). ∎

So the pre-Batch-7 design was not merely badly parameterised: for a fixed width
it was *undefined* on an interval of thresholds of length `c` — 20% of the unit
interval at `c = 0.20`. Measured on a 1e-5 grid:

```
A  fixed width 0.20                constructible on 0.80002 of (0, 1)
```

Every design that is total must therefore give up (1), (2), or (3). There are
only three doors, and choosing one is the design decision:

| give up | what you get | what it costs |
|---|---|---|
| (1) bracketing | nothing useful — a band that does not straddle the threshold is not an abstention band | fatal |
| (2) probability symmetry | the shipped design: symmetry relocated to log-odds | the Batch 1 validator changes scale |
| (3) width independence | E′ proportional, or C quantile: the width becomes a function of `t` or of the corpus | the knob loses a fixed meaning |

### 2.2 Why clipping is worse than the bug

Clipping — approach B — drops (2) rather than raising. It is total, and it
destroys the thing the band is for. At the R11 threshold:

```
B  fixed width 0.20, clipped   [0.8944, 1.0000)  cov 0.341  FPR 0.0000  TPR 0.0000  decided-fake 0
```

`band_hi` clamps to exactly 1.0. The manipulated region is `p ≥ band_hi`, which
no calibrated score ever satisfies, so **the detector's positive class becomes
unreachable**. TPR is 0.0000, not by coincidence but by construction. Every
published field still validates as a well-formed number, and the failure is
invisible to any check that does not know what the fields mean.

Measured over the unit interval, clipping is constructible everywhere and
*useful* on exactly the region where it was never needed:

```
B  clipped                         constructible on 1.00000 of (0, 1)
B  clipped, both regions reachable                  0.80000 of (0, 1)
```

This is the single strongest argument in the batch: the obvious fix converts a
loud failure into a silent one with identical coverage.

---

## 3. The chosen design: a log-odds margin

### 3.1 Definition

For a threshold `t ∈ (0, 1)` and a **margin** `δ > 0` in log-odds:

```
λ      = logit(t) = ln(t / (1 − t))
band_lo = expit(λ − δ)
band_hi = expit(λ + δ)
```

Equivalently, in odds:

```
odds(band_hi) = e^δ · odds(t)
odds(band_lo) = e^−δ · odds(t)
```

which is the sentence an operator reads: *abstain unless the odds are at least
`e^δ` times past the threshold odds.* The shipped placeholder is `δ = ln(1.5)`,
i.e. a factor of 1.5.

### 3.2 Totality

> **Claim.** For every `t ∈ (0, 1)` and every finite `δ > 0`,
> `0 < band_lo < t < band_hi < 1`.

*Proof.* `logit : (0,1) → ℝ` is a bijection and strictly increasing, and `expit`
is its inverse, also strictly increasing with range `(0, 1)`. Since
`λ − δ < λ < λ + δ`, monotonicity gives `expit(λ−δ) < t < expit(λ+δ)`, and the
range of `expit` gives the outer strict inequalities. ∎

There is no feasibility constraint linking `t` and `δ`. **R11 is not handled; it
is unrepresentable.** The pair that failed in the audit now succeeds with no
special case:

```
threshold 0.9734218951277995, margin ln(1.5)  →  [0.960656, 0.982123)
```

### 3.3 What survives from Batch 1

The Batch 1 invariant was *the band is derived, never configured* — enforced by a
model validator rejecting any `(value, band_lo, band_hi)` triple not centred on
its own threshold. That check is unweakened. It is still an **equality on a
single derived parameter**; only the scale changed:

```python
above = logit(band_hi) − logit(value)
below = logit(value) − logit(band_lo)
reject unless |above − below| ≤ LOG_ODDS_BAND_TOLERANCE · max(above, below, 1)
```

Every legacy shape is still refused — `band_hi == value`, the dead zone, the
inverted band, the off-centre band — and one new shape is refused as well: the
probability-symmetric band that shipped before. `ThresholdInfo(value=0.62,
band_lo=0.52, band_hi=0.72)` now raises, and the message names the change.

### 3.4 The two numerical limits

Both are measured, not assumed.

**`LOG_ODDS_BAND_TOLERANCE = 1e-8`, relative.** Validation must invert `expit`,
and that inversion is ill-conditioned near the endpoints: a one-ULP error in an
edge becomes an error of about `2.2e-16 / (1 − band_hi)` in its log-odds. Over
16 134 `(t, δ)` pairs spanning the admissible domain the worst round-trip
asymmetry is **2.54e-10** relative to the margin. The tolerance is that with a
factor of 40 in hand, and still seven orders of magnitude below any usable
margin. `test_the_symmetry_tolerance_holds_across_the_whole_admissible_domain`
re-measures it and fails in *both* directions — too loose is also a defect.

The tolerance is relative rather than absolute because a margin is unbounded
above; an absolute tolerance would be vacuous for a wide band and unsatisfiable
for the conditioning cases above.

**`BAND_EDGE_FLOOR = 1e-6`.** An edge must stay this far from 0 and 1. Two
independent reasons that agree on the number: an edge within 1e-6 of an endpoint
makes the region beyond it unreachable (approach B's failure, arrived at from the
other side), and the log-odds round trip degrades below it — at 1e-9 the worst
case is 30× worse.

This floor is what bounds the usable margin: `max_margin(t) = 13.8155 − |logit t|`.
It binds only absurdly close to the endpoints:

```
threshold 1e-06        max margin  -0.0000   shipped margin fits: False
threshold 1.5e-06      max margin   0.4055   shipped margin fits: True
threshold 0.5          max margin  13.8155   shipped margin fits: True
threshold 0.9999985    max margin   0.4055   shipped margin fits: True
threshold 0.999999     max margin   0.0000   shipped margin fits: False
```

Compared honestly: the old width was infeasible on **20%** of the unit interval,
including every threshold a tight budget produces. The new margin is infeasible
within about **1.5e-6** of an endpoint, where the operating point is degenerate
for reasons having nothing to do with the band. The failure did not move; it
shrank by five orders of magnitude and landed somewhere no fit reaches.

### 3.5 Why the parameter now *means* something

A fixed probability width of 0.20, measured in evidence:

```
threshold   band                    evidence below   evidence above
0.15        [0.0500, 0.2500)          1.2098           0.6360
0.30        [0.2000, 0.4000)          0.5390           0.4418
0.50        [0.4000, 0.6000)          0.4055           0.4055
0.70        [0.6000, 0.8000)          0.4418           0.5390
0.85        [0.7500, 0.9500)          0.6360           1.2098
0.89        [0.7900, 0.9900)          0.7658           2.5044
0.90        [0.8000, 1.0000)          0.8109              inf
```

The same setting is worth 0.41 nats at 0.5 and 2.50 nats at 0.89, and diverges at
0.90. **There is no value of `band_width` that is right at two different
thresholds**, which is why R8 could not be closed on its own terms.

A fixed margin of ln(1.5), measured in probability:

```
threshold   band                    width below      width above
0.15        [0.1053, 0.2093)          0.0447           0.0593
0.30        [0.2222, 0.3913)          0.0778           0.0913
0.50        [0.4000, 0.6000)          0.1000           0.1000
0.70        [0.6087, 0.7778)          0.0913           0.0778
0.85        [0.7907, 0.8947)          0.0593           0.0447
0.90        [0.8571, 0.9310)          0.0429           0.0310
0.99        [0.9851, 0.9933)          0.0049           0.0033
```

Constant in evidence, varying in probability, and the variation is in the correct
direction: less probability where there is less room. The wider half in
probability always faces 0.5.

---

## 4. The six candidates, evaluated

`D` is not a geometry. It is the *selection procedure* — sweep, constrain,
choose — and needs one of the others to parameterise. It is applied over `E`,
which is what the shipped policy does.

### 4.1 Summary

| | design | total? | contract-expressible? | parameter means | verdict |
|---|---|---|---|---|---|
| A | fixed width, symmetric in probability | **no** — fails on 20% of (0,1) | yes | nothing fixed (§3.5) | **rejected**: the bug |
| B | A, clipped into [0,1] | yes | **no** (not centred) | nothing fixed | **rejected**: silently deletes a verdict class |
| C | quantile band at target abstention | yes | **no** (not centred) | abstention rate | **rejected**: a function of the corpus, not the threshold |
| D | risk–coverage derived | n/a | n/a | n/a | **adopted as the policy**, over E |
| E | log-odds margin | **yes** | yes (validator rescaled) | odds ratio past the threshold | **adopted as the geometry** |
| E′ | proportional: fraction of widest fitting band | yes | **yes, unchanged** | "fraction of the way to certainty" | runner-up; lost on meaning |
| F | Wilson interval on the threshold | yes | yes | threshold sampling error | **adopted as a floor to report**, not a policy |

### 4.2 Per-approach detail

Measured at three regimes. Full output in `tests/band_study.py`.

**A — fixed width, symmetric in probability.**
*Guarantees:* symmetric in probability; constant width.
*Failure modes:* does not exist for `t < w/2` or `t > 1 − w/2`; the parameter's
meaning drifts with the threshold; refuses at calibration time with a message
about bands when the real problem is the fit.
*Near 0 and 1:* undefined.
*Effect:* at a central threshold, coverage 0.893, FPR 0.1643, TPR 0.9739. At the
saturated threshold, **no band exists**.

**B — clipped.**
*Guarantees:* totality, and nothing else.
*Failure modes:* asymmetric, so the width no longer means what it says; and when
the upper edge clamps, the manipulated region is empty.
*Near 1:* `TPR = 0.0000, decided-fake 0` — measured, at both the synthetic
saturated regime and the audit's own fixture.
*Verdict:* strictly worse than A, because A fails loudly.

**C — quantile band.**
*Guarantees:* totality; direct control of coverage (`cov 0.800` by construction
at every regime).
*Failure modes:* the edges are a function of the *evaluation distribution*, not
of the threshold. Two corpora give two bands for one threshold, so a stored
operating point is no longer reproducible from its own published fields, and the
coverage guarantee does not transfer under domain shift. Unstable on small
samples. Not centred, so not contract-expressible.
*Effect:* buys real selective risk (0.0631 vs A's 0.0918 at the central regime)
at 9 points of coverage — but it is buying it with a knob that does not survive a
corpus change.

**D — risk–coverage derived.** The procedure, §5. Not a geometry.

**E — log-odds margin.** §3. Total, contract-expressible, constant in evidence.
At the central regime it tracks A closely (coverage 0.904 vs 0.893, TPR 0.9718 vs
0.9739) — the two agree exactly at `t = 0.5` and diverge slowly. At the saturated
regime it produces a working operating point where A produces none.

**E′ — proportional.** `width = ρ · 2 · min(t, 1−t)`. The serious competitor: it
is total *and* probability-symmetric, so it would have preserved the Batch 1
validator verbatim and moved no fingerprint. Rejected on two grounds:

1. `ρ` is "how far toward certainty", which is not a quantity anyone can reason
   about or defend in a review. It converts an unjustified absolute number into
   an unjustified relative one. Measured, the same `ρ` varies in evidence by more
   than 1.5× across `t ∈ {0.5, 0.9, 0.99}`.
2. `min(t, 1−t)` has a kink at `t = 0.5` — continuous, but not differentiable,
   so the width's response to a moving threshold flips sign discontinuously.

Worth recording that E′ and E agree to first order near an endpoint: as `t → 1`,
E gives width `≈ 2(1−t)·sinh δ` and E′ gives `2(1−t)·ρ`, so `ρ = sinh δ`. They
differ in how they *split* the width, not in how much there is. Pinned by a test
to within 2%.

**F — Wilson interval.** Margin from the threshold's own sampling error,
`z·√(t(1−t)/n)` converted through the same both-sides rule as the determinism
floor.
*Guarantees:* "abstain wherever the threshold's own uncertainty could have put
the boundary" — an appealing statement.
*Failure modes:* it is a statement about the **calibration sample size and
nothing else**. The scores never enter. A model that got worse overnight is
handed the same band as long as the holdout is the same size, and the band
shrinks as the holdout grows. Measured: `n` 200 → 800 halves the margin (ratio
2.05). It is silent about selective risk, which is what the band is actually for.
*Verdict:* a useful *floor* to report alongside a derived margin — if the derived
margin is below the Wilson margin, the band is narrower than the threshold's own
uncertainty and the operating point is not meaningful. Not the policy.

---

## 5. The operating policy

### 5.1 The procedure

`evaluation.band.derive_operating_point(y, p, ...)`, in order:

1. **Threshold** from the false-positive budget: `at_target_fpr`, the lowest
   threshold whose FPR stays inside `target_fpr`, so the budget is *spent* rather
   than merely respected.
2. **Admissibility.** The fit must have produced a real probability strictly
   inside (0, 1). `at_target_fpr` returns `max(score) + 1` as a sentinel when
   nothing meets the budget, and a saturated detector can return an endpoint.
   Both stop here, reported as problems with the *fit* — which is what they are.
3. **Determinism floor → margin.** `margin_for_probability_floor(t, f)` takes the
   larger of the two sides, so the floor is met on the weaker one:
   `δ ≥ max(logit(t+f) − logit(t), logit(t) − logit(t−f))`.
4. **Sweep** a geometric grid from that floor to `max_margin(t)`. Geometric
   because the margin is a log quantity: equal ratios are the equal steps.
5. **Constrain**: keep candidates whose *selective* FPR is within
   `max_selective_fpr` and whose coverage is at least `min_coverage`.
6. **Choose the narrowest survivor.** Narrowest is most coverage; the band is a
   cost, and the rule is to pay the least of it that meets the constraints.
7. **Refuse** if there is no survivor, returning a census of why each candidate
   was rejected.

`fit_calibration(margin=DERIVE)` — the default — routes through this. **There is
no longer any argument, anywhere, that sets the band without the threshold in
hand.** That is the structural closure of R11: not a check, an absence.

### 5.2 `max_selective_fpr` has no default, and why

It is the product decision the band exists to implement, and R8 was the
consequence of nobody having made it. A default would be a new unjustified
constant standing exactly where the old one stood.

It must also be **tighter than the bare threshold already achieves**, or it buys
nothing. Widening the band can only lower the false-positive rate, because the
decision boundary is `band_hi` and widening moves it away from the negatives. So
a ceiling at or above `target_fpr` is satisfied by the narrowest candidate on the
grid and selects nothing.

That case is not an error — "this model does not need a band at this operating
point" is a legitimate finding. It is reported as `binding=False` with a warning
naming the cause, and `fit_calibration(require_binding=True)` refuses it. An
unremarked margin of 1e-4 presented as a derived answer would be the worst
outcome available, and it is the one the study caught on the R11 fixture:

```
D  derived, sFPR<=0.0100 (m=0.0001)  [0.9944, 0.9944)  cov 0.988  <- no constraint binds
```

### 5.3 That the derivation is a function of the data

The margin responds to the score distribution, monotonically, by a factor of 3.4
across a realistic range of score spreads — from a procedure with no constant in
it:

| score spread σ | derived margin | threshold | coverage | selective FPR |
|---|---|---|---|---|
| 1.0 | 0.9907 | 0.4186 | 0.836 | 0.0081 |
| 1.6 | 1.5962 | 0.6591 | 0.608 | 0.0082 |
| 2.5 | 2.3628 | 0.8960 | 0.558 | 0.0093 |
| 3.5 | 3.3596 | 0.9785 | 0.575 | 0.0087 |

Note the last row: a threshold of 0.9785 — the R11 regime — with a derived margin
of 3.36 and no complaint. Note also *what* drives it. The margin is set by how
heavy the negative tail is above the threshold, which is a property of the
**spread**, not of the separation between class means. A fixture that varies
separation alone leaves the margin nearly unchanged; that was found while writing
the tests, and the fixture was corrected rather than the assertion weakened.

### 5.4 The determinism floor

The band's second job: numerical jitter in `p_fake` must never flip a published
verdict. The floor is stated in probability, because that is where jitter is
measured. Converted:

```
jitter    t=0.50    t=0.62    t=0.90    t=0.97    t=0.99
0.0001     0.0004    0.0004    0.0011    0.0034    0.0102
0.001      0.0040    0.0042    0.0112    0.0349    0.1064
0.005      0.0200    0.0213    0.0568    0.1875    0.6982
0.01       0.0400    0.0427    0.1164    0.4157         -
0.05       0.2007    0.2186    0.7472         -         -
```

`-` marks the one joint impossibility that remains, and it is not the band's
fault: a floor of 0.05 either side of a threshold sitting 0.03 from an endpoint
is unachievable at any width in any geometry, because probability runs out. The
refusal says so and names the knob that has to move — usually acquiring
negatives, since a tight budget on a small sample is what drove the threshold
there.

`select_margin` applies the floor to `min(value − band_lo, band_hi − value)`, not
to `band_width`. Those were the same test when the band was probability-symmetric
and they are not now; the guarantee has to hold on the *weaker* side.

### 5.5 Recommended settings

Nothing here is measured on real scores — there are still no datasets and no
weights — so these are the settings to *run the derivation with*, not derived
values.

| knob | recommendation | why |
|---|---|---|
| `target_fpr` | 0.02, and only with ≥ 500 negatives | unchanged; `MIN_NEGATIVES_FOR_FPR` |
| `max_selective_fpr` | strictly below the achieved FPR — start at `target_fpr / 2` | otherwise nothing binds (§5.2) |
| `min_coverage` | a product decision; 0.85 is a starting point, not a finding | the abstention rate the product can live with |
| `determinism_floor` | 10 × the measured run-to-run variation in `p_fake` on the target device class | **not yet measured — see §7** |
| `margin` | `DERIVE` | the default; a float is for reproducing a previously calibrated point |

The shipped placeholder, for a deployment with no calibration yet, is
`DEFAULT_LOG_ODDS_MARGIN = ln(1.5)`. It is still a placeholder and still labelled
one, but it now has three properties the old constant did not: it is stated in a
unit that means something, it is valid at every threshold, and it was chosen so
the shipped operating point moved as little as possible — [0.5200, 0.7200) →
[0.5210, 0.7099) at the shipped threshold of 0.62.

---

## 6. Contract impact

### 6.1 Can `ThresholdInfo` express the chosen design?

**The fields can; the old validator could not.**

| | before | after |
|---|---|---|
| `CONTRACT_VERSION` | 1.0.0 | 1.0.0 |
| `EvidenceBundle` schema | — | unchanged |
| `ThresholdInfo` fields | `value`, `target_fpr`, `band_lo`, `band_hi` | unchanged, same types, same `[0,1]` bounds, same required-ness |
| published JSON Schema | — | **byte-identical** |
| centring invariant | probability | **log-odds** |
| constructor | `centred(value, band_width, target_fpr)` | `at_margin(value, margin, target_fpr)` |

The four published floats still satisfy `band_lo < value < band_hi`, all still in
[0, 1]. The OpenAPI drift guard passes without a snapshot change, and the
`band_width` property is retained as a derived, published quantity — it is what
the risk–coverage reports read.

So the contract change is exactly one thing: **the internal invariant moved
scale**, and §2.1 proves that was unavoidable given totality plus a
derived-not-configured band. Per the batch brief's constraint 7, this is the
minimum change that makes the design representable.

The constructor rename is deliberate rather than incidental. `at_margin` takes a
keyword the old call sites do not pass, so every one of the 45 of them failed
loudly and had to be revisited. A same-named constructor with a re-interpreted
argument would have changed the meaning of every call site silently.

### 6.2 What moved, and why moving was correct

`ModelSpec().threshold` moved from [0.5200, 0.7200) to [0.5210, 0.7099).
Verdicts for scores in the two slivers between those edges change. Therefore:

- `calibration_fingerprint`: `dc6fcc65` → **`739c3fec`**
- `STUB_ENGINE_VERSION`: `stub-0-w7f45fada+c.dc6fcc65.g.ae232cba` →
  **`stub-0-w7f45fada+c.739c3fec.g.ae232cba`**

This is Batch 1's invariant 6 working. The threshold and band are in the
calibration fingerprint precisely so that moving the decision boundary
invalidates the result cache; a geometry change that left the fingerprint alone
would have served verdicts decided under the old rule. The literals are re-pinned
with the reason recorded in the test that pins them.

One consequence worth naming: the band edges are now *computed*
(`expit(logit(0.62) ± ln 1.5)`) rather than typed, so they no longer round to
themselves under `quantize`. The Batch 4 test that pinned "the shipped defaults
are unaffected by quantisation" was narrowed to the configured defaults, and a
new test pins what actually matters for the cache — that the edges do not sit
astride a rounding boundary, checked by perturbing the margin by 1e-12 and
requiring the fingerprint to survive.

### 6.3 Calibration witness

`CALIBRATION_WITNESS_VERSION` 1.0.0 → **1.1.0**. `band_width` became `margin`.
The two are not interconvertible without deciding which edge of the old band to
preserve, and picking one would invent an operating point that was never
calibrated — so a 1.0.0 payload is **refused with a message naming the change**,
not migrated. This version axis is independent of `CONTRACT_VERSION` by design.

---

## 7. Remaining open risks

**The margin is still not derived from real scores.** R8's *mechanism* is closed
— the derivation exists, is the default path, and is exercised — but every number
in §5.3 is synthetic. Until there are calibrated scores from a real model on a
real calibration split, the shipped `ln(1.5)` is a placeholder. This is gated on
datasets and a GPU, both explicitly out of scope for this batch.

**The determinism floor has never been measured.** `determinism_floor` defaults
to 0.0, which means "no numerical guarantee is being claimed" — honest, and not
the same as safe. The measurement needed is the run-to-run spread of `p_fake` for
one input across the target device classes. Until it exists, §5.4's table is a
conversion tool with no input.

**A non-binding derivation is reported, not prevented, by default.**
`require_binding=False` is the default on `fit_calibration` because a
temperature-only refit is a legitimate use. A release path should pass
`require_binding=True`.

**`min_coverage = 0.85` is still a number nobody chose.** It is a rejection
threshold rather than a fitted value, so it is a smaller version of R8 rather than
a repeat of it — but it is the same species, and it is now the largest
unjustified constant left in the operating-point path.

**The quantile design's objection applies weakly to the derived margin too.** A
margin derived from one calibration split is a function of that split's score
distribution. It transfers better than approach C's edges — it is at least a
function of the *threshold* as well — but it is not distribution-free, and it
should be re-derived when the calibration corpus changes. The calibration witness
makes that auditable; nothing forces it.

**Nothing forces verification to run.** The standing complaint from R4 and R5,
unchanged: `binding`, the warnings list, and `verify_calibration_witness` all
have to be read by somebody.

---

## 8. Where things live

| what | where |
|---|---|
| geometry, constructor, validator, limits | `contracts/common.py` — `ThresholdInfo`, `logit`, `expit`, `DEFAULT_LOG_ODDS_MARGIN`, `LOG_ODDS_BAND_TOLERANCE`, `BAND_EDGE_FLOOR` |
| derivation policy, alternatives, floor conversion | `evaluation/band.py` |
| margin sweep, selective rates, selection | `evaluation/risk.py` |
| the calibration entry point | `evaluation/calibration_protocol.py` — `fit_calibration(margin=DERIVE)` |
| helper, kept coupled to the contract | `inference/calibration.py` — `abstention_band` |
| geometry + R11 tests, with named doubles | `tests/test_threshold_band.py` |
| derivation tests, by requirement | `tests/test_band_derivation.py` |
| the six-approach measurement | `tests/band_study.py` |
