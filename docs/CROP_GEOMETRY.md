# Crop geometry: from the detector's box to the model's input tensor

> **Superseded in part by Batch 22 (R25 CLOSED).** §6 and §10 below recorded R25
> as open — three geometry knobs (`input_size`, the interpolation rule, the clip
> rule) outside every identity string. Batch 22 closed it: `input_size` folded
> into `weights_identity`, `interpolation_rule` and `crop_clip_rule` added to
> `FaceSelectionPolicy`. Model identity moved
> `stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387` →
> `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9`, one cache generation, no
> pixels moved. R28 (the `resize` `or` predicate, noted in §5.2 of the study)
> is deferred. R23/R24/R26 geometry defects remain open. See
> `docs/CROP_IDENTITY.md` and `docs/CROP_IDENTITY_B22_PLAN.md`. `test_batch_15_
> changed_no_geometry` is now `test_batch_22_moved_the_identity_not_the_pixels`.
> **Batch 23 (R29) moved the `.f.` segment again** to the composite `e629642c`
> (model identity `…f.e629642c`) by folding in the video frame-selection /
> pooling policy; no pixels moved. See `docs/VIDEO_SELECTION_IDENTITY.md`.

**Status:** S8 Batch 15. **An investigation batch — no geometry changed.**
R23 is characterised, split into its two independent halves, and pinned; two
documentation defects are corrected; **three new findings are recorded, R24, R25
and R26**. R13, R16, R18's capability half, R20, R21, R22 unchanged.
**Contract impact:** none. `CONTRACT_VERSION` 1.0.0; no field, no schema, no
OpenAPI signature, no version string moved — asserted by
`test_batch_15_changed_no_geometry`.
**Suite:** 43 suites, 1116 tests, zero failures.

---

## 0. The answer, up front

The brief asked whether R23 is a correctness defect, a reproducibility defect, a
documentation defect, or a combination. **It is all three, and the finding as
recorded conflates two defects with different triggers.**

| | What | Trigger | Reachable today |
|---|---|---|---|
| **Correctness, live** | a clipped crop is stretched onto the square model input | **border proximity** | **yes — with square Haar boxes** |
| **Correctness, latent** | occupancy collapse: the crop is mostly not-face | **free aspect** | no — armed by SCRFD |
| **Reproducibility** | three geometry knobs move every scored pixel and are in no identity | always | **yes** (R25) |
| **Documentation** | the docstring claims distortion is avoided; a test is named for a property it never checks | always | **yes** |

The single most load-bearing correction: **R23's distortion has nothing to do
with free aspect.** It comes from clipping, and a perfectly square 200×200 box
flush with the right edge of a 480×480 image is stretched **13.0%** today. The
free-aspect half of R23 is real but is a *different* defect — it destroys
occupancy, not shape — and only that half is SCRFD-gated.

And the largest geometry defect in the platform is not in `crop_face` at all: the
video path resizes **whole frames** onto a square, squashing 16:9 content by
**1.778×** on every video bundle ever published (§7, R24).

---

## 1. Phase 1 — the geometry flow

Every site where face geometry is created, transformed, clipped, resized,
normalised, fingerprinted, cached, benchmarked or published.

```
IMAGE PATH
  bytes
   └─ decode_image                       preprocess.py:191   (H,W,3) uint8 RGB
       └─ detect_faces                   preprocess.py:634   CREATED: list[FaceBox], ints
           └─ face_order_key             preprocess.py:625   ORDERED: -size_px,-area,y,x,w,h
               └─ faces[0]
                   ├─ _gate              image_pipeline:227  size_px vs min_face_px
                   ├─ crop_face          preprocess.py:664   EXPANDED  side = long_edge * margin
                   │                                         TRANSLATED x0 = max(0, cx - side//2)
                   │                                         CLIPPED    x1 = min(w, x0 + side)
                   │   └─ prepare_image  preprocess.py:694   RESIZED   cv2.resize(crop,(S,S))
                   │       └─ resize     preprocess.py:229   per-axis independent scale
                   │           └─ normalize (off by default) NORMALISED  astype(float32)
                   │               └─ backend.score          the model never sees the source
                   └─ face_track         postprocess.py:663  PUBLISHED BoundingBox(max(0,x),…)
                       └─ EvidenceBundle.faces[0].bbox

VIDEO PATH
  bytes
   └─ extract_frames(target_size=input_size)   preprocess.py:918
       └─ resize(WHOLE FRAME, S)               preprocess.py:229
           └─ backend.score
  No detection. No crop. No box. §7.

FINGERPRINTED            crop_margin, crop_shape        → .f. segment (B12)
NOT FINGERPRINTED        input_size, interpolation rule, clipping rule   → R25
CACHED ON                effective_version = weights + calibration + gate + .f.
BENCHMARKED ON           the same string, via BenchmarkRun.effective_model_version
PUBLISHED                the detector's box — never the crop window   → R26
```

Two structural facts fall out of the diagram and neither is written down
anywhere else:

**The model never sees the source image, and nothing records what it did see.**
Between `faces[0]` and `backend.score` the region is expanded, possibly
translated, possibly truncated and always resampled. `EvidenceBundle` carries the
box from *before* all four steps.

**The video path shares `resize` with the image path but nothing else.** It
inherits the per-axis-independent square resize without inheriting the square
crop that makes it harmless.

---

## 2. Phase 2 — every geometry assumption

| Assumption | Encoded in | Documented | Tested | Fingerprinted |
|---|---|---|---|---|
| detector boxes are square-ish | implicit in `crop_face` + the gate | no | B14 only (the 2 px margin) | no |
| **the crop is square** | `crop_face` docstring | **yes — and it was false** | no | no |
| **aspect distortion is avoided** | `crop_face` docstring | **yes — and it was false** | no | no |
| margin expands the **long** edge | `crop_face` body | since B14 (`long_edge`) | no | **yes** — `crop_margin` |
| crop shape name | `policy.crop_shape` | yes | no | **yes** — `crop_shape` |
| **clipping / centring rule** | `crop_face` body | no | no | **no** |
| **padding: none, ever** | `crop_face` body | no | no | **no** |
| model input is square | `prepare_image`, `ModelSpec.input_size` | partly | no | **no** |
| **resize is per-axis independent** | `preprocess.resize` | no | no | **no** |
| interpolation choice | `preprocess.resize` | partly (why AREA) | no | **no** |
| **video frames may be squashed** | `extract_frames` | no | no | **no** |
| published `bbox` == scored region | `face_track` | no | no | n/a |

### 2.1 The two documented falsehoods

`crop_face`'s docstring said, until this batch:

> Square because the model takes a square input, and cropping square here rather
> than resizing a rectangle later **avoids the aspect distortion that stretches
> facial geometry** — a distortion a detector will happily learn as a feature of
> one dataset.

Every clause is right about the *intent* and the last one is wrong about the
*implementation*: `prepare_image` resizes a rectangle to a square whenever the
crop was clipped, which is exactly the distortion the sentence promises to
prevent. Corrected in this batch — prose only, no behaviour — and pinned by
`test_the_crop_docstring_no_longer_claims_distortion_is_avoided`, which holds the
corrected text against the legacy sentence recorded verbatim.

### 2.2 The test named for a property it never checks

`tests/test_preprocess.py::test_crop_face_is_square_and_clipped_to_bounds`
asserted `crop.size > 0` and that the crop fits the image. The word *square*
appeared in the name and in none of the assertions — and its fixture,
`FaceBox(90, 90, 40, 40)` in a 100×100 image, sits in the **one corner where both
axes clip by the same amount**, so the crop is square by coincidence:

```
FaceBox(90, 90, 40, 40) -> 16x16   square (the fixture)
FaceBox(90, 40, 40, 40) -> 16x52   NOT square, 3.25x stretch
```

The same box, moved off the corner, produces the largest anisotropy in this
report. Renamed to
`test_crop_face_stays_inside_the_image_and_is_never_empty`, with the reason in
its docstring.

---

## 3. Phase 3 — does distortion actually occur? Measured

### 3.1 Square boxes — the only shape the shipped detector emits

200×200 box swept across a 480×480 image; `side = 260`.

| Position | Crop | Anisotropy | Stretch | Occupancy | Face off-centre |
|---|---|---|---|---|---|
| centred | 260×260 | 1.0000 | 0.0% | 59.2% | (0, 0) |
| flush **left** | 260×260 | 1.0000 | 0.0% | 59.2% | **(−30, 0)** |
| flush **top** | 260×260 | 1.0000 | 0.0% | 59.2% | **(0, −30)** |
| flush **right** | **230×260** | **1.1304** | **13.0%** | 66.9% | (15, 0) |
| flush **bottom** | **260×230** | **1.1304** | **13.0%** | 66.9% | (0, 15) |
| bottom-right corner | 230×230 | 1.0000 | 0.0% | **75.6%** | (15, 15) |
| 20 px from right | 250×260 | 1.0400 | 4.0% | 61.5% | (5, 0) |
| 40 px from right | 260×260 | 1.0000 | 0.0% | 59.2% | (0, 0) |
| face fills the frame | 480×480 | 1.0000 | 0.0% | 69.4% | (0, 0) |

**Three distinct failures, not one.**

1. **Right/bottom border → anisotropic distortion.** `x1` truncates, the crop
   stops being square, `prepare_image` stretches it. 13.0% at the border, 4.0%
   twenty pixels away.
2. **Left/top border → framing error.** `x0` clamps to 0, the crop keeps its full
   width, and the face slides 30 px off centre. Shape survives; framing does not.
   **The clamp and the truncate are not each other's mirror**, and nothing
   documents either.
3. **Corner → scale error.** Both axes clip equally, so the crop is square again
   — but it is 230×230 rather than 260×260, so the face arrives at the model
   **28% larger** than it would from the centre of the frame. Distortion-free and
   still not the same input.

Measured on content rather than on shape alone: a circle drawn in the crop region
of a right-border case reaches the model as an **ellipse** of exactly the crop's
anisotropy (`test_the_square_resize_turns_a_circle_into_an_ellipse`).

### 3.2 The exact activation condition

Truncation on the right occurs **iff**

```
gap  <  ceil(side/2) - ceil(s/2)          where gap = W - (box right edge)
```

— about **15% of the box's own width** at the shipped margin 1.3. Brute-forced
over 50 000 random square boxes in the study and 20 000 in the suite: **0
mismatches**.

In words: *a face whose box edge is within 15% of its own width of the frame
border is scored on a stretched crop.* Equivalently, any face wider than
`1/margin` = 77% of the frame. Both are ordinary photographs.

Haar cannot detect a face that is partly outside the frame, so the box is always
fully inside — but that bounds the gap at zero, not below the threshold. The
condition is reachable; what is *not* reachable is a synthetic demonstration,
because Haar loses the fixture the moment the canvas is trimmed (§5, R13 again).

### 3.3 Free aspect — no distortion at all

With room to spare, every aspect ratio yields a square crop:

| Box | Crop | Anisotropy | Occupancy |
|---|---|---|---|
| square 200×200 | 260×260 | 1.0000 | **59.2%** |
| near-square 200×203 | 263×263 | 1.0000 | 58.7% |
| portrait 150×200 (SCRFD-ish) | 260×260 | 1.0000 | **44.4%** |
| landscape 200×150 | 260×260 | 1.0000 | 44.4% |
| extreme 900×50 | 1170×1170 | 1.0000 | **3.3%** |

**Free aspect produces no distortion.** The window is square by construction. What
it destroys is *occupancy* — how much of the model's field of view is face.

So the two halves of R23 separate cleanly:

```
occupancy collapse  <- free aspect       -> LATENT, armed by SCRFD
aspect distortion   <- border clipping   -> LIVE TODAY, needs no free aspect
```

### 3.4 Both at once

| Case | Crop | Anisotropy | Occupancy |
|---|---|---|---|
| 900×50 face, room to spare | 1170×1170 | 1.0000 | 3.3% |
| 900×50 face at a border | 1085×600 | **1.8083** | 6.9% |

An 80.8% stretch — the two defects compose rather than cancel.

---

## 4. Phase 4 — the fail-before proof

Two named doubles, both executable.

`LegacyClipToBounds` is the shipped `crop_face`, verbatim, pinned to the real
function on the whole position sweep by
`test_the_legacy_double_matches_the_shipped_crop` — pixels as well as shape, so
it cannot silently drift.

`PadToSquareCrop` computes the same window and **fills** it by edge replication
where it leaves the image, instead of cutting it down.

| Position | Legacy crop | Aniso | Corrected crop | Aniso |
|---|---|---|---|---|
| centred | 260×260 | 1.0000 | 260×260 | 1.0000 |
| flush left | 260×260 | 1.0000 | 260×260 | 1.0000 |
| flush right | **230×260** | **1.1304** | 260×260 | 1.0000 |
| flush bottom | **260×230** | **1.1304** | 260×260 | 1.0000 |
| bottom-right | 230×230 | 1.0000 | 260×260 | 1.0000 |

**Why the correction eliminates the failure**, and this is the part worth stating
precisely rather than demonstrating and moving on:

> The shipped code computes a window and then **intersects** it with the image,
> so the image bounds enter the crop's **shape**. The corrected code computes the
> same window and then **fills** it, so the bounds enter only its **content**.
> Shape is exactly what `prepare_image`'s square resize is sensitive to.

Asserted directly, not as prose: `test_why_the_correction_works_bounds_change_
content_not_shape` varies the image width across 480/520/560/600 with the box
fixed and shows the legacy crop shape changing while the corrected one does not.

The correction also restores the second invariant — the face is centred in every
corrected crop, including at the left and top borders where the shipped code
slides it 30 px — and it demonstrably changes the pixels reaching the model, so
it is a real correction rather than a no-op.

**`PadToSquareCrop` is not shipped.** Edge replication is one fill among several;
which one belongs in a production crop is a modelling question this batch does not
answer (§9, option E). The double exists to prove the failure is eliminable and to
show what eliminates it.

---

## 5. Phase 5 — why the image scenarios are scripted

| Canvas | Detection | Crop | Aniso |
|---|---|---|---|
| 480×480, the fixture | 302×302 @ (88, 98) | 392×392 | 1.0000 |
| 460×480, trimmed 20 px | **no detection** | — | — |
| 440×480, trimmed 40 px | **no detection** | — | — |
| 400×480, trimmed 80 px | **no detection** | — | — |

R13 again: Haar sees the synthetic face only when it is alone and centred on a
480×480 canvas. So the border cases are built with a scripted detector — using
boxes that are **square**, exactly what Haar emits. That is the argument rather
than a limitation: the defect needs no free-aspect detector, and the scripted
boxes are chosen to make that impossible to miss.

---

## 6. Phase 6 — reproducibility. **Yes, and it is R17's shape**

> *Can two deployments with identical model, runtime, face-selection and
> calibration identity produce different verdicts solely because crop geometry
> changed?*

**Yes, through any of three knobs.** Batch 12 fingerprinted `crop_margin` and
`crop_shape` and closed R17; it did not reach these.

### Knob 1 — `ModelSpec.input_size`

```
input_size 384 -> stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387
input_size 224 -> stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387
identical: True        model input: (1,384,384,3) vs (1,224,224,3)
```

The resize target is the model's entire field of view, and it is in **no**
fingerprint. `weights_identity` covers `version`, `branch_versions` and
`branch_weights`; `input_size` is a sibling field that never enters it.

### Knob 2 — the interpolation rule in `resize`

`INTER_AREA` when `h > size or w > size`, else `INTER_LINEAR`. Neither the rule
nor the choice is fingerprinted. Measured: on a 500×300 crop to 384 the two
interpolations differ by **mean 5.402/255**.

The rule has a second problem. It is an `or`, so a crop **wider** than the input
but **shorter** than it is area-filtered on the axis being *enlarged* — a
low-pass filter applied to an upsample, in a module whose docstring says
`INTER_AREA` is chosen because *"forensic signal lives in high-frequency
texture"*. Recorded, not fixed.

### Knob 3 — the clipping / centring rule

`crop_shape = "square-longest-edge"` names how `side` is **derived**. It says
nothing about what happens where the window leaves the image. Clipping and
padding are both honestly describable by that string, they hand the model
different pixels, and they produce the identical face-selection fingerprint
`19559387`.

This is the same failure mode Batch 12 accepted for `ORDERING_RULE` and Batch 14
re-recorded for the gate metric: **a name in an identity string is only as good as
the test holding it to the behaviour**, and here there is no such test.

### Verdict

> **[B22 — R25 CLOSED. See prelude.]** The text below records the *open* state as
> of B15. B22 folded `input_size` into `weights_identity` and added
> `interpolation_rule` / `crop_clip_rule` to `FaceSelectionPolicy`; the three
> knobs are now in the model identity. Read `docs/CROP_IDENTITY.md` §15.

**R25 — crop geometry inputs outside every identity string.** One finding, three
knobs, exactly R17's structure: verdict-affecting geometry living in a dataclass
field, a module-level function and an unnamed branch policy — values nobody
thinks of as configuration because they never appear in a config file. The
missing distinction is in the **model** identity.

---

## 7. R24 — the video path, and the largest geometry defect in the platform

`extract_frames(..., target_size=self._spec.input_size)` resizes each **whole
frame** to `size × size` through the same per-axis-independent `resize`.

| Source | Model input | Anisotropy | Effect |
|---|---|---|---|
| 1920×1080 (16:9) | 384×384 | **1.7778** | squashed 1.778× |
| 1280×720 (16:9) | 384×384 | **1.7778** | squashed 1.778× |
| 640×480 (4:3) | 384×384 | 1.3333 | squashed 1.333× |
| 1080×1920 (9:16) | 384×384 | 1.7778 | squashed 1.778× |
| 512×512 (1:1) | 384×384 | 1.0000 | unchanged |

Measured on content:

```
a CIRCLE of radius 400 in a 1920x1080 frame reaches the model as
a 160 x 284 ELLIPSE -- axis ratio 0.5634, not 1.0
through LetterboxResize (one scale factor, pad):  160 x 160, ratio 1.0000
```

This is **live on every video bundle the platform has ever published**, it is
1.778× on the commonest source aspect, and it is roughly fourteen times the
image path's worst case. It reaches a branch A the architecture declares to
consume an *aligned face crop*.

It is not R23 — R23 is about `crop_face` — and it is not R18, which is about the
absent face stage. A video face stage would replace this code path entirely, so
fixing it in isolation is work that SCRFD throws away. **Recorded as R24, pinned
by `test_the_video_path_resizes_whole_frames_to_a_square`, paired with
`LetterboxResize`, and not fixed.**

---

## 8. Phase 6 — SCRFD migration impact

| Behaviour | Today (Haar) | Under SCRFD |
|---|---|---|
| crop truncation at a border | **LIVE** (square boxes) | live, unchanged |
| anisotropic stretch from truncation | **LIVE** | live, larger |
| face off-centre at left/top border | **LIVE** | live, unchanged |
| occupancy collapse from free aspect | latent (aspect ≈ 1) | **ACTIVE immediately** |
| crop mostly background | no — 59% face | **ACTIVE** — 44% at aspect 0.75 |
| negative box coords clamped by `face_track` | unreachable | **ACTIVE** (regressed boxes) |
| published `bbox` ≠ scored region | LIVE | live, and now also clamped |
| video whole-frame squash (R24) | LIVE | unchanged — no face stage |

Occupancy at the aspect ratios a real detector produces:

| w/h | Box | Crop | Occupancy |
|---|---|---|---|
| 1.00 | 200×200 | 260×260 | 59.2% |
| 0.90 | 180×200 | 260×260 | 53.3% |
| 0.80 | 160×200 | 260×260 | 47.3% |
| **0.75** | 150×200 | 260×260 | **44.4%** |
| 0.65 | 130×200 | 260×260 | 38.5% |

A 0.75-aspect box — ordinary for SCRFD — puts a quarter less face in front of the
model than a square one, at the same margin and the same fingerprint.

### 8.1 Which existing tests would notice? **None**

Every geometry assertion in the suite is a byte comparison or a status-quo
tripwire:

| Suite | What it asserts about geometry |
|---|---|
| `test_preprocess` | `crop.size > 0`, crop fits the image. Was named for squareness; never checked it. |
| `test_face_size_policy` | Batch 14's R23 tripwire — **pins** today's behaviour, *including* that a clipped crop is not square. |
| `test_face_selection_identity` | crop **bytes**, to prove the margin reaches the model. |
| `test_inference_determinism` | crop bytes again, for repeat-invariance. |

Nothing asserts that the resize is isotropic. Nothing asserts occupancy. Nothing
exercises a border position with a non-corner box. **Activation under SCRFD would
be silent** — which is the strongest argument in this report for landing the
geometry tests now, and it is what Batch 15 does.

---

## 9. Phase 7 — alternatives

Assessed against the shipped 384×384 square-input model contract. "Benchmark
continuity" means: can a pre-change and a post-change run be compared?

### A — current: longest-edge square window, clipped

- **Correctness:** distorts at right/bottom borders (13% at 200 px), mis-frames at
  left/top, changes scale at corners, collapses occupancy on free aspect.
- **Reproducibility:** margin and shape name fingerprinted; the clipping rule is
  not (R25).
- **Benchmark continuity:** the baseline.
- **Migration cost:** zero.
- **Contract impact:** none.
- **Verdict:** the status quo, and defensible only while boxes are square *and*
  centred. Neither survives SCRFD.

### B — short-edge square window

`side = short_edge * margin`. Fixes occupancy for free-aspect boxes by cropping
tighter.

- **Correctness:** raises occupancy (a 150×200 box goes from 44.4% to ~59%) but
  **cuts the long axis of the face off** — a portrait box loses forehead and chin,
  which is where several manipulation artefacts live. Border clipping unchanged.
- **Reproducibility:** one `crop_shape` string change; identity moves cleanly.
- **Benchmark continuity:** broken — every crop changes on non-square boxes; under
  Haar, nothing changes (|w−h| ≤ 1).
- **Migration cost:** low.
- **Contract impact:** none.
- **Verdict:** trades a framing defect for a truncation defect. Rejected as a
  standalone; it is the wrong axis to standardise on.

### C — the detector box only, no margin

- **Correctness:** removes the occupancy question entirely and destroys context
  the model needs; Batch 12 measured margin 1.0 vs 1.3 moving one image from
  `authentic` 0.0136 to `manipulated` 0.8893. Still distorts, because a non-square
  box resized to a square input *is* the distortion.
- **Reproducibility:** `crop_margin` becomes decorative; identity would need care.
- **Benchmark continuity:** broken everywhere.
- **Migration cost:** low.
- **Verdict:** rejected. It makes the distortion universal rather than
  border-conditional.

### D — fixed-margin rectangle, no squaring

Crop `box * margin` and let the rectangle through.

- **Correctness:** maximal occupancy, and **guarantees** the distortion for every
  non-square box, since `prepare_image` squares whatever it gets.
- **Reproducibility:** clean.
- **Benchmark continuity:** broken.
- **Verdict:** rejected — it is option A's failure made unconditional. Only viable
  paired with F.

### E — letterbox / pad to square (`PadToSquareCrop`, `LetterboxResize`)

Compute the square window, fill what falls outside; and for video, scale by one
factor and pad the short axis.

- **Correctness:** **anisotropy 1.0 everywhere, by construction**, and the face is
  centred at every border. Measured in §4 and §7. Does not fix occupancy on
  free-aspect boxes — that is orthogonal, and belongs with the margin.
- **Reproducibility:** needs the fill mode in the fingerprint, which is the same
  kind of field `crop_shape` already is.
- **Benchmark continuity:** broken only for crops that clip today; under Haar with
  centred subjects, most results are unchanged — but *which* results change is
  content-dependent, so a clean comparison across the boundary is not available.
- **Migration cost:** low in code, real in re-analysis (one cache generation).
- **Contract impact:** none.
- **Verdict:** **the correction this report recommends**, and the fill mode
  (replicate / reflect / constant / blurred) is a modelling choice that wants
  real weights to decide. A constant fill introduces a hard synthetic edge, which
  is precisely the high-frequency signal a manipulation detector reads.

### F — affine-preserving resize into the model input

Keep an explicit similarity transform from source to input; letterbox is its
degenerate case. The general form is what SCRFD's 5-point landmark alignment
needs anyway.

- **Correctness:** strictly the most correct — one transform, invertible, and it
  is what "aligned face crop" in the architecture actually means.
- **Reproducibility:** the transform's parameters become the fingerprint, which is
  cleaner than naming a shape.
- **Benchmark continuity:** broken everywhere; this is a different preprocessing
  pipeline.
- **Migration cost:** **high**, and it is SCRFD's cost — landmarks do not exist
  yet (`FaceTrack.landmarks` is an empty list on every published bundle).
- **Contract impact:** none required; `landmarks` is already in the contract and
  already empty.
- **Verdict:** **the right destination**, and it arrives with SCRFD. Doing E now
  and F later means doing the work twice; doing F now means doing it without
  landmarks, which is doing it blind.

### Recommendation

**Do not change crop geometry in this batch, and do not change it before SCRFD.**

The reasoning is the mirror of Batch 14's, and the difference is worth being
explicit about. R19's fix was provably output-identical under the shipped
detector — 30 000 box pairs, zero divergence — so it could land with no
behavioural risk and no benchmark discontinuity. **No crop-geometry option has
that property.** Every one of A–F changes the pixels of some real input today,
which means a cache generation and a benchmark boundary, and none of them can be
validated against a real model because there is not one.

What *is* worth doing now is exactly what this batch does: **make activation
loud.** The tests landed here fail if the geometry moves, if the video squash
changes, if the identity coverage changes, or if the documentation drifts back.
That converts a silent SCRFD activation into a noisy one, at zero behavioural
cost.

---

## 10. Phase 8 — where does crop geometry belong in the identity vocabulary?

Assessed with Batch 9's and Batch 12's criteria: *is this code the deployment
ships, or environment it inherits?* and *would an operator diffing two versions
want to see this separately?*

| Candidate | Argument | Verdict |
|---|---|---|
| **Face-selection identity** (`.f.`) | Crop geometry already lives here — `crop_margin` and `crop_shape` are two of the eight fields. The clipping rule and the fill mode are the same *kind* of thing: shipped code deciding **what pixels the model sees**. Adding them is an extension of a segment that already owns the question, not a new concept. | **yes — for the clipping rule and any fill mode** |
| **Runtime identity** (`.d.`) | This is the decode environment: OpenCV version, video backend, OS. The *interpolation implementation* legitimately lives here (an OpenCV change to `INTER_AREA` moves `cv2.__version__`, which is in the decode identity). But the **choice** of `INTER_AREA` is code this deployment ships, and Batch 9's whole distinction is code-vs-environment. | **no for the choice, already-yes for the implementation** |
| **Benchmark identity** | Benchmark identity is deliberately the *model* identity — `BenchmarkRun.effective_model_version` — so anything in `.f.` is inherited for free. A separate benchmark-only field would cover regression gates and miss the cache, which is the live defect. Batch 12 rejected this exact option for the same reason. | **no — inherited, not separate** |
| **A new identity segment** | A `.c.` crop segment would split geometry across two segments (`crop_margin` in `.f.`, the clipping rule in `.c.`) for one decision. Batch 12's argument against folding selection into the gate was that the two have different owners and lifecycles; here they have the *same* owner and the *same* lifecycle — SCRFD moves both together. | **no** |
| **Nowhere** | Defensible only if geometry could not move a verdict. §3 and §6 measure that it does. | **no** |
| **`input_size`** | The odd one out: it belongs with the **weights**, not with selection. A checkpoint has an input size; changing it without changing the checkpoint is a misconfiguration, and `weights_identity` is where a misconfiguration of the weights should show up. | **weights identity** |

**Conclusion.** Crop geometry belongs in the **face-selection identity** for the
clipping rule and any fill mode; `input_size` belongs in the **weights
identity**; the interpolation *choice* belongs in the face-selection identity
alongside the rest of the shipped preprocessing, while its *implementation*
already sits correctly in the decode identity.

**Not implemented in this batch.** Closing R25 moves `effective_version` and dumps
the cache, and doing that for a geometry change that has not happened yet spends
the migration twice — once now for the fingerprint and again at SCRFD for the
geometry. The defensible sequence is: **land the fingerprint extension in the same
batch as the geometry change it is meant to protect.** That is what Batch 12 did
for R17 and what Batch 14 did for the ordering rule, and it is recorded here as
the recommendation rather than executed.

> **[B22 — superseded.]** B22 closed R25 ahead of the geometry change, judging the
> reproducibility gap (three verdict-movers under one cache key) worth one cache
> generation now. The interpolation and clip rules were named as *label closures*
> (verbatim current behaviour) so no pixel moved; their values are revised once
> more when R23/SCRFD lands — the accepted single double-spend. See
> `docs/CROP_IDENTITY.md` §15.

---

## 11. What Batch 15 changed

**Behaviour: nothing.** Asserted by `test_batch_15_changed_no_geometry` —
`crop_margin` 1.3, `crop_shape` `square-longest-edge`, face-selection identity
`19559387`, `STUB_ENGINE_VERSION` and `CONTRACT_VERSION` all unmoved.

**Documentation: two corrections.**

1. `crop_face`'s docstring no longer claims to avoid aspect distortion. It states
   the three border behaviours, the exact truncation condition, and the occupancy
   consequence, and points at this report.
2. `test_crop_face_is_square_and_clipped_to_bounds` →
   `test_crop_face_stays_inside_the_image_and_is_never_empty`, with its docstring
   recording why the old name was false and why its fixture never caught it.

**Tests: 23 new**, including the two doubles, the exact truncation condition, the
before/after pairing, the R24 and R25 pins, and the R26 limits.

---

## 12. Guarantees gained

1. **The geometry is measured**, end to end, with numbers rather than a docstring.
2. **R23's two halves are separated**, with their triggers and reachability
   established independently.
3. **The distortion is proved live** — square boxes, ordinary framing — rather
   than assumed latent.
4. **The activation condition is a closed form**, brute-forced over 70 000 cases.
5. **The failure is proved eliminable**, with a double that says why.
6. **Silent SCRFD activation is now impossible** for anything in §8: each row has
   a test that fails when it moves.
7. **The identity gap is named and located** — three knobs, one finding, one
   recommended home each.
8. **Two false claims are gone** from the codebase.

## 13. Guarantees NOT gained

1. **No geometry is fixed.** Distortion, mis-framing, corner scale and occupancy
   collapse all still happen. `PadToSquareCrop` and `LetterboxResize` are doubles.
2. **R24 is not fixed.** Every video bundle is still scored on squashed frames.
3. **R25 is not closed.** All three knobs still sit outside every identity string.
   *(B22: closed — `input_size` → `weights_identity`, `interpolation_rule` +
   `crop_clip_rule` → `FaceSelectionPolicy`. See prelude and `CROP_IDENTITY.md` §15.)*
4. **R26 is not closed.** The bundle still publishes the detector's box rather
   than the scored region, and `face_track`'s clamp will diverge from `crop_face`
   the moment a detector emits a box crossing the frame edge.
5. **No measurement with a real model.** Every score here comes from
   `StubBackend`, which hashes its input — so *that* the model sees different
   pixels is established and *how much it matters* is not. The 13% stretch is a
   fact about the tensor; its effect on a verdict needs weights.
6. **The fill mode is unchosen.** §9 option E's replicate/reflect/constant
   question wants a real model.
7. **R13, R16, R18's capability half, R20, R21, R22** untouched.

---

## 14. Measured impact

| Measurement | Value |
|---|---|
| Geometry sites traced | 11, across 4 modules |
| Geometry assumptions enumerated | 12 — **2 documented and false**, 0 tested, 2 fingerprinted |
| Worst image-path anisotropy, square box | **1.1304** (13.0% stretch) |
| Worst image-path anisotropy, existing test's own box | **3.25** (16×52) |
| Worst combined (free aspect + border) | **1.8083** |
| Truncation condition mismatches, 70 000 cases | **0** |
| Occupancy, aspect 1.0 → 0.75 → 900×50 | 59.2% → 44.4% → 3.3% |
| Corner scale error | face arrives **28% larger** than from frame centre |
| **Video anisotropy, 16:9** | **1.7778**, every video bundle |
| Circle → ellipse, video path | radius 400 → **160 × 284** |
| Unfingerprinted geometry knobs | **3** (`input_size`, interpolation, clipping rule) |
| Interpolation difference, mixed-case crop | mean **5.402 / 255** |
| Existing tests that would catch SCRFD activation | **0** |
| Behaviour changes this batch | **0** |
| Documentation corrections | 2 |
| Contract / schema / OpenAPI / version changes | 0 |
| Suite | 42 → **43 suites**, 1093 → **1116 tests**, zero failures |

---

## 15. R23 status, and the new findings

**R23 — crop geometry: CHARACTERISED, SPLIT, PINNED. Not fixed.**

| Half | Class | Status |
|---|---|---|
| Border clipping → anisotropic distortion | **correctness, LIVE** | Open. Measured, closed-form condition, pinned, correction proved. Fix with SCRFD (§9). |
| Left/top clamp → face off centre | **correctness, LIVE** | Open. Same fix. |
| Corner → scale error | **correctness, LIVE** | Open. Same fix. |
| Free aspect → occupancy collapse | **correctness, latent** | Open. Armed by SCRFD. |
| Docstring claimed distortion avoided | **documentation** | **CLOSED.** |
| Test named for an unchecked property | **documentation** | **CLOSED.** |

**R24 — the video path resizes whole frames onto a square.** Correctness, live on
every video bundle, 1.778× on 16:9. Pinned; fix belongs with the video face stage.

**R25 — crop geometry inputs outside every identity string.** Reproducibility,
live. Three knobs. §10 says where each belongs; land it with the geometry change
it protects.

**R26 — the published `bbox` is not the scored region**, and `face_track`'s
coordinate clamp will diverge from `crop_face` under a detector that emits boxes
crossing the frame edge. Contract/correctness, latent.

---

## 16. Re-running this

```bash
python tests/crop_geometry_study.py    # every number above
python tests/test_crop_geometry.py     # 23 proof tests
python tests/run_all.py                # 43 suites, 1116 tests
```

## 17. Where things live

| what | where |
|---|---|
| The crop, the resize, the policy | `inference/preprocess.py` |
| The video resize call | `inference/preprocess.py` (`extract_frames`), `inference/video_pipeline.py` |
| The published box | `inference/postprocess.py` (`face_track`) |
| `input_size` and the identity composition | `inference/models.py` |
| Proof, doubles, and the asserted limits | `tests/test_crop_geometry.py` (23 tests) |
| The measurements this rests on | `tests/crop_geometry_study.py` |
| The batch that opened R23 | `docs/FACE_SIZE_POLICY.md` §9 |
| The identity precedents | `docs/RUN_IDENTITY.md`, `docs/FACE_SELECTION_IDENTITY.md` |
