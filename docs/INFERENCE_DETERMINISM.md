# Inference determinism — S8 Batch 10 (R13)

> **The question.** Batch 8 proved two decode environments produce different
> verdicts for the same file. Batch 9 made those environments unable to share a
> cache. R13 was what remained: **can identical media produce different verdicts
> inside ONE decode environment?**
>
> **The answer is yes, by two mechanisms, and neither is findable by running the
> pipeline twice and comparing.** One is order dependence (R14); one is a genuine
> data race that also contaminates results across concurrent jobs (R15). Both are
> fixed. R13 as originally written — face-crop instability amplifying decoder
> drift — is now quantified rather than closed, and the amplification factor is
> **310×**.

Measurements re-runnable with `python tests/inference_determinism_study.py`
(twelve stages). Guarantees pinned by `tests/test_inference_determinism.py`
(24 tests).

**No contract change.** `CONTRACT_VERSION` 1.0.0. `EvidenceBundle`,
`ThresholdInfo`, the OpenAPI snapshot, the model contracts and every fingerprint
in the platform are untouched — the fixes change *which* answer is produced under
concurrency and *which* face wins a tie, not the identity of any configuration.
`STUB_ENGINE_VERSION` does not move.

---

## 1. Threat model

**The adversary is a load spike.** Not a malicious engineer, not a rebuilt
container, not a corrupted file. Every mechanism below is reachable by a worker
doing exactly what it was designed to do, at a moment when two requests overlap.

Prior batches could all fall back on the same instrument: make the two states
*distinguishable*, then let a fingerprint refuse to pool them. R12 got a decode
segment in the runtime version; R4 got per-item digests in the split lock. **That
instrument is unavailable here**, and the reason is worth stating precisely:

| | R12 (Batch 8/9) | R14 / R15 (this batch) |
| --- | --- | --- |
| Divergent results come from | two environments | **one process** |
| OpenCV build | differs | identical |
| Video backend | differs | identical |
| Model version | identical | identical |
| Decode identity | **differs** | identical |
| Detectable by a key? | yes — Batch 9 did it | **no, structurally** |
| Available control | pin, record, refuse to pool | **prevention only** |

Two answers produced ninety milliseconds apart by one process share every byte of
every identity string this platform can compute. There is nothing to hash. This
is why both findings were fixed rather than recorded — the "make it visible"
option that closed R12 does not exist here, and
`test_runtime_identity_cannot_detect_the_cascade_race` asserts it does not.

**What is assumed trusted.** The decoded pixels (that is R6/R12's territory), the
model weights, and the fact that a bounded, reused thread pool is what
`asyncio.to_thread` provides. This is reproducibility machinery, not a security
boundary.

**Scope.** Everything between decoded bytes and a published `EvidenceBundle`:
face detection, face selection, crop alignment, resize, frame extraction, frame
selection, pooling, fusion, calibration, thresholding, abstention. Decode itself
is out of scope — Batch 8 owns it.

---

## 2. What was measured, stage by stage

Twelve stages. Eight are clean and are reported anyway, because a study that
lists only what broke leaves the reader unable to tell a measured guarantee from
an untested assumption.

| # | Stage | Probe | Result |
| --- | --- | --- | --- |
| 1 | Face detection | 20 repeats; cv2 thread counts 0/1/2/4/8; optimisation toggle | deterministic |
| 2 | Face selection | all 6 permutations of 3 equal-area boxes | **DIVERGENT — R14** |
| 3 | Crop alignment | repeated crop + prepare | deterministic |
| 4 | Frame extraction | 4 repeats, indices + timestamps + pixels | deterministic |
| 5 | Frame ranking | all-tie curves, 6 repeats | deterministic |
| 6 | Pooling | forward vs reversed index order | deterministic *because* indices arrive sorted |
| 7 | Video aggregation | repeated segment + top-frame derivation | deterministic |
| 8 | Thresholding | 501-point grid, 4 repeats | deterministic (pure function) |
| 9 | Abstention | gate cliffs are integer comparisons | deterministic |
| 10 | End to end | 6 subprocesses × {thread counts, `PYTHONHASHSEED`} | deterministic |
| 11 | **Concurrent detection** | 8 threads on one image; 4+4 threads on two files | **DIVERGENT — R15** |
| 12 | Verdict impact | population flip rates for each measured `\|dp\|` | — |

Stage 10 is the one most likely to be mistaken for a complete answer. It runs the
whole pipeline in six separate processes under varying thread counts and hash
seeds and gets one byte-identical bundle — and it is **blind to both findings**,
because every launch is single-threaded and every launch feeds the detector the
same input order. That is the central methodological point of this batch: *the
obvious determinism test passes on a pipeline with a live data race in it.*

---

## 3. R14 — face selection was decided by detector output order

`detect_faces` returned `sorted(boxes, key=lambda b: -b.area)`. Python's sort is
stable, so two boxes of equal area came out in whatever order the cascade emitted
them — and `faces[0]` chooses the crop the entire verdict is computed from.

Measured on a 400×800 fixture with two 200×200 regions:

```
A first -> crop 6c9a2dfa3527, p_fake 0.4830  ->  authentic
B first -> crop f04cb2ef504e, p_fake 0.5637  ->  insufficient evidence
```

Same image, same model, same runtime identity, different verdict, decided by an
ordering nothing specified.

**This is order dependence, not nondeterminism**, and the distinction is the
whole point. Run the pipeline a thousand times and it agrees a thousand times,
because `detectMultiScale` is deterministic on this build. Nothing *specifies*
that order, so nothing stops it moving — and the frozen architecture replaces
Haar with SCRFD, whose box order falls out of NMS over float scores and is an
implementation detail by construction. The defect would have arrived with the
migration, silently, as a diff in numbers nobody could explain.

### The fix

`face_order_key` — `(-area, y, x, width, height)` — a **total** order on the box
geometry alone. Every component is a property of the input rather than of the
detector, so `faces[0]` became a function of the image.

Width and height are in the key after the origin so the order is genuinely total:
100×200 and 200×100 share both area and origin and differ only in aspect, and an
area-plus-origin key would still have tied them.

**Exact-area ties are rare with Haar and fired on no shipped fixture**, which is
why this was never observed in production and is not evidence it could not be.
`minSize` quantisation makes equal areas reachable, and SCRFD makes emission
order arbitrary by design.

---

## 4. R15 — one `CascadeClassifier`, every thread

This is the finding R13 was actually asking about, and it is live.

`cv2.CascadeClassifier` carries mutable per-detection state: the feature
evaluator and its scaled-image buffers hang off the classifier, not off the call.
`inference/preprocess.py` held one in a module global. **Both pipelines reach
`detect_faces` through `asyncio.to_thread`**, so a process-wide instance was
shared across pool threads in the shipped configuration — not in a worst case
constructed for this study.

Two threads inside `detectMultiScale` on one instance overwrite each other's
working set. Nothing raises a `ValueError`, nothing logs. The caller receives a
well-formed, plausible, wrong answer.

### 4.1 Same file, eight threads

Recorded across four runs of stage 11, 320 calls each:

```
one process-wide cascade (pre-R15)      wrong 6-19/320  (1.9-5.9%)   ~6.2 s
one process-wide cascade behind a lock  wrong    0/320  (0.00%)     ~27-38 s
one cascade per thread (SHIPPED)        wrong    0/320  (0.00%)      ~5.7 s
```

Every wrong answer in the first row was a **lost face** — an empty box list,
which routes a scored image to `NO_FACE_DETECTED`.

### 4.2 End to end, through the real pipeline

The claim a user of this platform actually has. One PNG, real `ImagePipeline`,
real event loop:

```
sequential, 40 runs   :  40 x  authentic  p_fake=0.032932877375
concurrent, 160 runs  : 136 x  authentic  p_fake=0.032932877375
                         24 x  insufficient_evidence / no_face_detected
```

**15% of concurrent analyses of one file published a different verdict from
every sequential run of the same bytes**, inside one process, one decode
identity, one model version. After the fix: 160/160 identical.

### 4.3 Two files at once — contamination, not just noise

This is the severe half, and same-file concurrency understates it by an order of
magnitude. Four threads on file **a**, four on file **b**, 160 calls each, three
recorded runs:

```
file a : correct 34-57,  lost 10-16,  other-box  87-110  (75-107 of them b's box)
file b : correct 51-72,  lost 11-19,  other-box  74-98   (69-88  of them a's box)
```

**Roughly 60% of detections returned the *other job's* bounding box.** The
pipeline then crops *this* job's pixels at coordinates located in *another* job's
image and publishes the resulting score as this job's verdict.

That is qualitatively worse than an unreproducible number, because it is not
visibly wrong. The box is well-formed, inside the frame, of a plausible size.
Nothing downstream — not the bundle, not the cache, not an auditor reading the
stored `FaceTrack` — can tell it was located in the wrong file.

A third failure mode was observed during development and did not recur in the
three recorded runs: the shared cascade reads out of range inside OpenCV's own
vectors and raises `cv2.error: invalid vector<T> subscript`, killing the analysis.
The study and the tests classify a raise as an outcome rather than letting the
thread die, so the rate is measured rather than dropped from the denominator.

### 4.4 The fix, and the alternative that was rejected

`threading.local()`. A thread cannot contend with itself, so the mechanism is
**gone rather than narrowed** — this is prevention, which as §1 established is
the only available control.

A lock around `detectMultiScale` is equally correct (measured 0/320) and was
rejected on cost. It serialises the single most expensive step of every
concurrent image job — 24 ms per call on this box — and measured **27–38 s
against 5.7 s**, a 5–6× wall-clock penalty at 8-way concurrency. The
`threading.local` pays instead a one-off 45–73 ms cascade load and 0.41 MB
resident per pool thread; the pool is bounded and reused, so that is paid once
per thread for the life of the process rather than once per job.

Worth noting the shipped fix is also marginally **faster than the broken code it
replaces** (5.7 s vs 6.2 s) — contended access to shared mutable state was not
buying speed either.

The lazy global was additionally unguarded, so two threads could each construct a
cascade and one would be silently discarded. That is gone with the global.

---

## 5. R13 as written — crop instability, quantified

Batch 8 recorded R13 as "face-crop instability amplifies decoder drift", measured
it at ±2 LSB on a fixture at the cascade's detection boundary, and deferred it to
the SCRFD migration. It is still deferred. It is no longer unquantified.

### 5.1 The perturbation is smaller than the one it amplifies

Batch 8 measured the cross-decoder pixel delta at mean |d| = 2.22/255 on
photographic content. So a ±3 LSB perturbation is *smaller than a decoder
change*. Applied to the detection fixture across 11 noise realisations:

```
detected in 9 of 11 realisations; the face was lost outright in 2
boxes ranged x 95-102, y 73-78, w 164-173   (7 px of position, 9 px of size)
```

### 5.2 Propagation to `p_fake`

Two scorers, following Batch 8's convention. The **stub** hashes pixel bytes, so
any change fully decorrelates it — a correct upper bound and a useless estimate.
The **surrogate** is a smooth Lipschitz function of the pixels (gradient energy on
the normalised crop); it responds proportionally to a small shift instead of
hashing it, and it is what the honest numbers below use. Neither is a deepfake
detector.

Holding the pixels fixed and applying each observed box:

```
crop held fixed, ±3 LSB pixel noise only :  p spread 0.000293
crop free to move under the same noise   :  p spread 0.091022

                              AMPLIFICATION = 310x
```

That is the R13 claim, measured: the detector converts a 0.0003 perturbation into
a 0.091 one. The mechanism is specifically **box size**, not box position —
translations of ±1 to ±8 px inside a 218 px crop survive `INTER_AREA` almost
intact (|dp| between 0.00016 and 0.00083), while a 5 px change in box *size*
moves |dp| to 0.091, a factor of 110. Changing the box size changes the resample
ratio, which rewrites exactly the high-frequency texture a forensic detector keys
on; sliding the same content two pixels does not.

### 5.3 What it costs in decisions

Against the shipped band [0.521008, 0.709924) over a Beta(2,2) score population:

| \|dp\| | source | decisions changed |
| --- | --- | --- |
| 0.0003 | fixed crop, ±3 LSB pixel noise (surrogate) | 0.08% |
| 0.0050 | R12 cross-decoder drift (surrogate, Batch 8) | 1.37% |
| **0.0910** | **R13 crop movement under ±3 LSB (surrogate)** | **25.71%** |

**R13 is roughly 19× the decision impact of R12**, the finding that justified two
whole batches. That reframes the deferral: R13 is not a footnote on R12, it is the
larger of the two, and it is the one still open.

### 5.4 Why it is still not fixed

Unchanged from Batch 8, and still correct: Haar is a placeholder the frozen
architecture replaces with SCRFD, and tuning a detector scheduled for deletion is
wasted work. What this batch adds is a **number to hold the migration to**.

**Acceptance criterion for the SCRFD migration**, stated so it is a gate rather
than a discovery afterwards:

> Under ±4 LSB uniform noise on real photographs, across ≥ 200 images: no face
> lost, and the box's short edge must not move by more than 1%. Re-run
> `inference_determinism_study.py` stage 3 against SCRFD and require the
> amplification factor below 10× (from the measured 310×).

---

## 5A. Derivations

Three places where the numbers above rest on an argument rather than on a
measurement.

**(a) `face_order_key` is a total order.** Let
`k(b) = (-wh, y, x, w, h)` for `b = (x, y, w, h)`. If `k(b₁) = k(b₂)` then the
last four components give `y₁=y₂, x₁=x₂, w₁=w₂, h₁=h₂`, hence `b₁ = b₂`. So `k`
is injective on distinct boxes and the induced order is total — no pair can tie,
and therefore no comparison can fall through to the stable sort's positional
tiebreak. The leading `-wh` is redundant for injectivity and is what makes the
order *useful*: it preserves "largest first", the rule the fix had to complete
rather than replace. `test_the_ordering_is_a_total_order` checks the same claim
by exhaustion over the cases an area-plus-origin key would miss.

**(b) A shift and an abstention are not the same event.** For a score population
with density `f` and band `[lo, hi)`, a uniform shift `δ > 0` changes a decision
exactly when the score crosses an edge, so

    P(decision changes) = ∫ over [lo−δ, lo) ∪ [hi−δ, hi) of f
                        ≈ δ · (f(lo) + f(hi))     for small δ

— proportional to `δ`, and confined to two thin strips. That is the row-by-row
table in §5.3, and it is why R12's 0.005 costs only 1.37%.

R15 is **not** of this form. A lost face does not move `p`; it deletes it and
substitutes `INSUFFICIENT_EVIDENCE`. The decision changes for every input whose
verdict was not already `INSUFFICIENT_EVIDENCE`:

    P(decision changes) = 1 − P(lo ≤ p < hi)      = 0.736 for Beta(2,2)

against the shipped band [0.521008, 0.709924), measured over 400 000 samples. So
a *lost* detection changes the decision three-quarters of the time it happens,
where R12's 0.005 shift changes it 1.37% of the time — two orders of magnitude
apart per occurrence. The measured 15% concurrent divergence is 15% of
*analyses* producing a wrong decision outright, not 15% of a band being crossed
— the two figures are not comparable and §5.3's table deliberately does not list
R15 among them.

**(c) Amplification.** Defined as the ratio of two spreads of `p` under the same
±3 LSB pixel perturbation: the numerator lets the detector re-run (crop free to
move), the denominator holds the box fixed at its unperturbed value.

    A = spread(p | crop re-detected) / spread(p | crop fixed)
      = 0.091022 / 0.000293
      = 310.6

Both spreads are computed on the same pixels with the same scorer, so everything
except the crop cancels; `A` isolates the detector's contribution. `A = 1` would
mean the detector passes noise through, `A < 1` that it suppresses it. Measured
`A ≈ 310` means the detector is the dominant source of score variance under any
upstream perturbation at this magnitude — including R12's, which is *larger* than
the ±3 LSB used here.

## 6. Answers to the seven questions

1. **Can identical input produce different `p_fake` values?**
   **Yes.** Under concurrency, measured at 15% of analyses of one file (R15).
   Single-threaded and in a fixed input order, no — stages 1 and 3–10 are clean
   across repeats, thread counts, hash seeds and process boundaries.

2. **Earliest stage where divergence appears?**
   **Face detection**, the first stage after decode. Both findings live there.
   Everything downstream is a faithful propagation of a wrong box.

3. **Noise, implementation choice, or nondeterminism?**
   All three, and they are separable. R15 is **true nondeterminism** (a data
   race). R14 is an **implementation choice** never written down (stable sort on
   a partial key). R13 is **amplification** — deterministic given exact bytes,
   and a 310× multiplier on any upstream noise, including R12's.

4. **Can verdict flips occur?** **Yes**, all three ways. R15: `AUTHENTIC` →
   `INSUFFICIENT_EVIDENCE` at 15%. R14: `authentic` → `insufficient evidence` on
   a tie. R13: 25.7% of decisions at the measured `|dp|`.

5. **Can abstention flips occur?** **Yes**, and abstention is R15's *only*
   observed same-file failure — the face is lost, not moved, so the score is
   replaced by `NO_FACE_DETECTED` rather than shifted. R13's ±3 LSB lost the face
   in 2 of 11 realisations.

6. **Can cache identity detect it?** **No, structurally** — §1. All divergent
   results share one process, one decode identity, one model version, one
   `sha256`. The content cache key is byte-identical across every outcome above,
   so it would serve whichever landed first to every later request. Asserted by
   `test_runtime_identity_cannot_detect_the_cascade_race`.

7. **Reproducibility-critical or measurement noise?**
   **Critical, and R15 is worse than a reproducibility problem** — a foreign
   bounding box is a correctness and isolation failure, not a stability one.
   R13's 25.7% is reproducibility-critical. R14 is latent today and becomes
   critical at the SCRFD migration.

---

## 7. Guarantees gained

1. **Concurrent analyses of one file publish one verdict** — proved by execution
   against the real pipeline on the real event loop, not argued.
2. **A job's detection cannot come from another job's media.** The contamination
   channel is removed structurally; a thread cannot contend with itself.
3. **Face selection is a function of the image**, not of the detector's emission
   order, under a total order that cannot tie.
4. **Eight pipeline stages are measured deterministic** — across repeats, cv2
   thread counts, optimisation toggles, `PYTHONHASHSEED`, and process boundaries
   — rather than assumed so.
5. **The order-dependence and concurrency hazards are pinned by named doubles.**
   `LegacyAreaOnlyOrder` and `LegacySharedCascade` are the pre-batch code
   verbatim; each test demonstrates the defect before proving the fix, and fails
   loudly if the double stops reproducing it.
6. **R13 has a number and an acceptance criterion**, so the SCRFD migration
   inherits a gate rather than a paragraph.

## 8. Guarantees NOT gained

1. **R13 is not fixed.** The 310× amplification is intrinsic to the placeholder
   detector and stands until SCRFD lands. It is the largest open determinism
   finding in the platform — 19× R12's decision impact.
2. **Multiprocess workers are untested.** Every measurement here is one process.
   A pre-forked or `multiprocessing` worker gives each process its own cascade
   and should be unaffected, but "should be" is not a measurement.
3. **The real detector is not this detector.** Every concurrency number is Haar's.
   SCRFD's runtime has its own thread-safety contract, and §4's requirement
   transfers to it rather than retiring with Haar.
4. **Determinism is not stability.** One changed pixel still moves everything, by
   design; `test_an_upstream_pixel_change_still_moves_everything` pins that so
   this report cannot be read as a robustness claim.
5. **No fingerprint detects any of this**, and none can — §1. If a future change
   reintroduces sharing, only the tests will say so.
6. **The "largest face wins" policy is untouched.** On a group photograph the
   platform still publishes one face's verdict as the media's. R14 made the
   choice reproducible, not correct. Recorded as **R16**.
7. **The blur estimate and the interpolation switch were checked and cleared, on
   this build only.** The `INTER_AREA`/`INTER_LINEAR` switch at the model input
   size is a genuine discontinuity — a 1 px change in crop size changes kernel —
   but measured, it contributes 0.128/255 against the 1.73/255 the content change
   itself contributes, so it is ~13× too small to matter. A negative result, and
   one build's.

---

## 9. New findings

**R14 — face selection decided by detector output order.** Medium. **Closed in
Batch 10** (§3), measured.

**R15 — the face cascade was shared across threads.** **High** — it is a live
data race that also leaks detections between concurrent jobs, and no identity in
the platform can see it. **Closed in Batch 10** (§4), measured.

R15 is rated above every previous determinism finding. R12 was reproducibility:
two decoders disagreed, both answers honest. R15 produces an answer that is
simply wrong — a crop located in a different file — with nothing anywhere
recording that it happened.

**R16 — one face's verdict is published as the media's.** Low, **open, by
design**. `analyze_image` scores `faces[0]` and publishes a single `FaceTrack`, so
a two-person photograph reports the larger face's verdict for the whole image.
Correct for the single-subject case the product targets and wrong for group
media; not a determinism defect (R14 made it reproducible) but recorded here
because R14's fix makes the policy explicit for the first time.

---

## 10. Recommended next

1. **Hold the SCRFD migration to §5.4's acceptance criterion.** It is the only
   thing that closes R13, and it is now a number rather than an intention.
2. **Measure R13 on real photographs.** Every number in §5 is a synthetic face
   sitting near this cascade's decision boundary — which the fixture constraint
   in `shifted_face_image` demonstrates independently. The direction is not in
   doubt; the magnitude is.
3. **Decide R16 deliberately** — either publish per-face verdicts or state the
   single-subject assumption in the contract. It is currently neither.
4. **Re-run this study under a multiprocess worker** before one ships.
