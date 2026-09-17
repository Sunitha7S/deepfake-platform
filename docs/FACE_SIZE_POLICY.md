# Face size: one box, three metrics, and the two that were allowed to disagree

> **Identity note (Batch 22).** The migration tables below record the Batch-14
> `.f.` move `d662e22d` → `19559387` and the model identity
> `stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387` — correct-for-then. Batch 22
> (R25) moved both again: `.f.` → `251d22e9`, weights `w7f45fada` → `wacdd9240`
> (folding in `input_size`), model identity
> `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9`. **Batch 23 (R29) moved the
> `.f.` segment again** to the composite `e629642c` (model identity
> `…f.e629642c`) — the *face* policy identity is still `251d22e9`, but the `.f.`
> segment now also carries the video frame-selection / pooling policy.
> `FACE_SIZE_METRIC` / `size_px` / the ordering rule are unchanged. See
> `docs/CROP_IDENTITY.md` and `docs/VIDEO_SELECTION_IDENTITY.md`.

**Status:** S8 Batch 14. **R19 closed** — measured, proved reachable only beyond
an exact threshold, fixed structurally, and pinned by named doubles. R13, R16,
R20, R21, R22 and R18's capability half unchanged (open; this batch touches none
of them and says so in §10). **One new finding recorded, R23** (§9).
**Contract impact:** none. `CONTRACT_VERSION` 1.0.0; `EvidenceBundle`,
`QualityReport`, `FaceTrack`, the OpenAPI snapshot and every serving route
unchanged. One configured literal moved, deliberately (§8).
**Suite:** 42 suites, 1093 tests, zero failures.

---

## 0. What R19 said, and what turned out to be true

The Batch 11 review recorded R19 as:

> the gate reads the largest-area face and tests its short edge —
> **correctness-critical, latent.** *Not reachable today*, and it matters why:
> Haar's frontal cascade returns square boxes, so area order and short-edge
> order coincide and the divergence cannot occur.

Three of those claims survive this batch and one does not.

| Claim | Verdict |
|---|---|
| Selection ranks on area, the gate tests short edge | **True.** §2 |
| The pairing admits a false `FACE_TOO_SMALL` | **True, and it is a defect rather than a plural policy.** §4 |
| Not reachable today | **True — but for a different reason, with a margin two pixels wide.** §5 |
| *Because Haar returns square boxes* | **False.** Haar returns boxes within **one pixel** of square, measured. §5 |

The corrected reason matters more than the correction. R19 is latent because a
witness needs a box **three pixels** from square and the shipped detector emits
boxes one pixel from square — a two-pixel margin resting on per-axis rounding
inside OpenCV, which this platform neither controls nor pins. It is not latent
because of any structural guarantee.

**Method.** Every claim below is a line of source with a file and line number, an
exhaustive enumeration, or a measurement from
`python tests/face_size_policy_study.py`. Three named doubles stand in for what
the platform does not have, and §1.1 says what each replaces.

---

## 1. Phase 1 — every face-size concept, traced

Nine names, in the brief's order, with what each one actually is.

| # | Name | What it is | Where it lives | Read by |
|---|---|---|---|---|
| 1 | `area` | `width * height` | `FaceBox.area` | ranking **tiebreak** (was: ranking) |
| 2 | `width` | detector output | `FaceBox.width` | ordering totality tiebreak; crop centre |
| 3 | `height` | detector output | `FaceBox.height` | ordering totality tiebreak; crop centre |
| 4 | **short edge** | `min(width, height)` | `FaceBox.short_edge` | **the face-size metric** |
| 5 | **long edge** | `max(width, height)` | `FaceBox.long_edge` | **crop side length** — R23 |
| 6 | `face_px` | published quality figure | `QualityReport.face_px`, `QualityFacts.face_px` | reporting only; nothing computes from it |
| 7 | `min_face_px` | gate floor, 64 | `MediaCaps` | the face gate |
| 8 | `min_short_edge` | gate floor, 64 | `MediaCaps` | the **image** resolution gate |
| 9 | `detect_min_px` | detection floor, 32 | `FaceSelectionPolicy` | `minSize` — a per-axis, i.e. short-edge, floor |
| 10 | `size_px` | **new in B14** — the one metric | `FaceBox.size_px`, named by `FACE_SIZE_METRIC` | ranking, gating and reporting |

Three of these are worth pulling out.

**Two floors share the word "short edge" and measure different objects.**
`min_short_edge = 64` is the *image's* short edge; `min_face_px = 64` is the
*face's*. They are consecutive lines of the same `_gate` function, they hold the
same value, and nothing in either name says which subject it takes. They are not
interchangeable and never have been.

**`detect_min_px` is already short-edge semantics.** `minSize=(32, 32)` rejects
any window narrower than 32 **or** shorter than 32, so the detection floor was
always a floor on the short edge. It is the gate floor's sibling, at half the
value, in a different fingerprint — deliberately, per Batch 12. What R19 was
about is that the *ranking* between them was neither.

**`long_edge` did not have a name before this batch.** `crop_face` computed
`int(max(box.width, box.height) * policy.crop_margin)` inline. It is a third
metric applied to the same box in the same analysis, it is not mentioned in
R19's write-up, and it has its own latent consequence — §9, R23.

### 1.1 The doubles, and why each exists

| Double | Replaces | Why |
|---|---|---|
| `LegacyAreaOrder` | nothing — it **is** the pre-Batch-14 `face_order_key` and the gate it fed, verbatim | So R19 is demonstrated rather than described. Every claim below is a pair: the double abstains and the shipped code does not. `test_the_legacy_double_still_reproduces_r19` fails if it ever stops reproducing the defect. |
| `ScriptedDetector` | `preprocess.detect_faces` | Haar cannot find two faces on one canvas (`multi_face_study` stage 0). The list is passed through the **shipped** ordering, so only the detection is scripted. |
| `RegionOracleBackend` | real weights | Gives a scenario a ground truth: paint one region dark and one bright, read back which the platform published. `StubBackend` hashes pixels and cannot. |

Everything between them is shipped code, reached through the real
`analyze_image` on a real event loop.

---

## 2. Phase 2 — which decision reads which metric

Measured, stage 2. **One box is measured three different ways on its way through
one analysis.**

| Decision | Metric | Threshold | Site |
|---|---|---|---|
| detector floor (`minSize`) | short edge, per axis | `policy.detect_min_px` = 32 | `preprocess.detect_faces` |
| image resolution gate | **image** short edge | `caps.min_short_edge` = 64 | `image_pipeline._gate` |
| **face selection / ranking** | **area** → **`size_px`** | — | `preprocess.face_order_key` |
| **face size gate** | **short edge** | `caps.min_face_px` = 64 | `image_pipeline._gate` |
| published `quality.face_px` | short edge | — | `image_pipeline.analyze_image` |
| **crop side length** | **long edge** | `policy.crop_margin` = 1.3 | `preprocess.crop_face` |

Read off the code rather than from prose — `face_order_key(FaceBox(0,0,200,100))`
returned `(-20000, 0, 0, 200, 100)` before this batch, whose leading component is
`-area`.

Under square boxes all three metrics coincide, which is why nothing has ever
noticed.

---

## 3. Phase 3 — the four disagreement cases, and the one that is empty

The brief asked for executable examples of four things. Three exist. **The
second is provably impossible, and proving that is the most useful result in this
section**, because it is what tells you which direction of the disagreement was
ever a defect.

### (a) area-largest ≠ short-edge-largest — constructible

```
400x100@(100,100) area=40000  short=100  long=400
190x190@(700,100) area=36100  short=190  long=190

area-largest       : 400x100
short-edge-largest : 190x190
```

### (b) area-largest scorable, short-edge-largest not — **provably empty**

The gate is `short_edge >= floor`. The short-edge-largest box **maximises exactly
the gated quantity**. So if it fails the floor, every box fails — the
area-largest one included. There is no witness, for any floor, under any
ordering, and this was true before Batch 14 as well: it is a fact about the gate,
not about the fix.

Measured anyway, because a one-line proof of emptiness deserves a search that
finds nothing: **200 000 random box sets, 0 witnesses.**
`test_the_opposite_disagreement_is_impossible` runs 50 000 of them in the suite.

### (c) short-edge-largest scorable, area-largest not — **this is R19**

```
900x50@(100,100)  area=45000  short= 50   below the 64px floor
200x200@(100,300) area=40000  short=200   clears it

legacy policy selects 900x50 -> scored=False
```

Through the real `analyze_image`, before and after:

| | verdict | p_fake | crops scored | `faces[]` | `quality.face_px` |
|---|---|---|---|---|---|
| **before** | `insufficient_evidence` | — | **0** | `(900, 50, face_too_small)` | **50** |
| **after** | `manipulated` | 0.952574 | **1** | `(200, 200, scored)` | **200** |

A face 200 px across, clearing every floor, was never scored and never published.
The file abstained because a wider, flatter box outranked it on area and then
failed a test on short edge.

### (d) ranking and gating disagree — constructible, and common once aspect is free

200 000 random box sets, free aspect, edges 1–400:

```
area-first != shipped-first        : 21598 (10.8%)
legacy abstains on a scorable file :  1425 (0.7%)
```

---

## 4. Phase 4 — a latent contradiction, not two independent policies

**The claim to defeat.** "Rank by area" (which face is the subject?) and "gate by
short edge" (is there enough facial detail?) are two different questions with two
defensible answers. Two metrics for two questions is not a contradiction.

That reading fails, and it fails on the composition rather than on either policy.

### 4.1 The gate-completeness theorem

> Let `r` be the ranking metric and `g` the gate metric. The state
>
> > `faces[0]` fails the gate **and** some other face would pass
>
> is reachable **iff** `r` and `g` are not monotonically related.

*Proof.* If `r = m ∘ g` for monotone `m`, then `r(b₁) > r(b₂) ⇒ g(b₁) ≥ g(b₂)`,
so `g(b₁) < floor` forces `g(b₂) < floor` for every `b₂`. No witness exists.
Conversely, non-monotonicity supplies a pair with `r(b₁) > r(b₂)` and
`g(b₁) < g(b₂)`, and any floor strictly between the two `g` values is a witness. ∎

Area and short edge are **not** monotonically related:

```
area(900x50)=45000 > area(200x200)=40000   but short= 50 < short=200
area(400x100)=40000 > area(190x190)=36100  but short=100 < short=190
```

### 4.2 Why that makes it a defect and not a plural policy

The state the pairing admits is not "a defensible policy chose a face I would
not have chosen". It is the platform publishing `FACE_TOO_SMALL` and
`gate_passed=false` for a file containing a face that clears every floor — a
**false statement about the file**, from a gate whose stated purpose
(`image_pipeline.py`, first paragraph) is to route *low-information* input to
honest abstention. A file containing a 200 px face is not low-information.

Note also which artifacts inherit the false claim: `quality.face_px` reports 50
for a file whose largest face measures 200, and `faces[]` publishes the thin box
with `status=face_too_small` while the scorable face is absent from the bundle
entirely. The error is not confined to the verdict.

So: **a latent contradiction.** Two policies would be fine if either could be
right on its own terms; the composition makes a claim that is false regardless of
which metric one prefers.

### 4.3 The exact activation threshold: three pixels

The interesting question is not *whether* the metrics can disagree but *how far
from square* a box must be. Write `d` for the largest `|width − height|` a
detector can emit and `F` for `min_face_px`. A witness needs `se₁ < F ≤ se₂` and
`area₁ > area₂`, so at most `(F−1)(F−1+d)` must exceed at least `F²`:

```
(F-1)(F-1+d) > F^2   <=>   d > 2 + 1/(F-1)
```

Since `d` is an integer, **`d ≥ 3` at every floor**:

```
floor  32 -> d > 2.0323 -> d >= 3
floor  64 -> d > 2.0159 -> d >= 3
floor 128 -> d > 2.0079 -> d >= 3
floor 512 -> d > 2.0020 -> d >= 3
```

Brute-forced rather than trusted to the algebra
(`test_the_activation_threshold_is_three_pixels`, floors 16/32/64/128: no witness
at `d = 2`, a witness at `d = 3`). The minimal witness at the shipped floor:

```
66x63 (area 4158, short 63)  beats  64x64 (area 4096, short 64)  on area
required aspect deviation: |w-h| = 3 px, aspect ratio 1.0476
```

**The same threshold governs the two orderings.** `area-desc` and
`short-edge-desc` disagree iff `se₁ > se₂` and `area₁ < area₂`, which needs
`d > 2 + 1/se₂`, i.e. `d ≥ 3` again. Therefore:

```
d <= 2 : the two orderings are identical  AND  R19 is unreachable
d >= 3 : the two orderings differ         AND  R19 is reachable
```

**Changing the ranking metric is a no-op exactly where the defect cannot fire and
takes effect exactly where it can.** That coincidence is what makes this fix
shippable before SCRFD rather than with it, and it is verified exhaustively
rather than argued: every box pair with `|w−h| ≤ 2` over edges 30–90 —
30 000 pairs — orders identically under both keys, and
`test_the_two_orderings_agree_below_the_threshold` enumerates 10 000+ more in the
suite.

---

## 5. Phase 5 — is R19 reachable today?

**No, and the reason in the record was wrong.**

```
cascade window size: (24, 24)   (square)

273 raw candidates (grouping disabled), noise images:
  non-square      : 1 (0.37%)
  |w-h| observed  : [0, 1]
  worst aspect    : 1.0179   example (56, 57)
```

Across a wider sweep while preparing this batch — 1 025 boxes over raw
candidates, four `minNeighbors` settings and both noise and synthetic-face
content — the maximum `|w − h|` observed was **1**, and every non-square box was
off by exactly one pixel.

**Haar does not return square boxes.** The cascade's *window* is square; the
boxes are not contractually so. Per-axis rounding in OpenCV's scale pipeline, and
`groupRectangles`' averaging of a cluster, can each land width and height on
different integers. The Batch 11 review's stated reason is false.

The conclusion it supported is nevertheless correct, by §4.3: a witness needs
`d ≥ 3` and the shipped detector gives `d ≤ 1`.

**But the margin is two pixels, and nothing in this platform holds it.** It is a
property of an OpenCV rounding path that no dependency pin covers (R7 declares
`opencv-python`; it does not pin its rounding), that no test asserted before this
batch, and that a patch release is free to change. `test_the_shipped_detector_
emits_boxes_within_the_safe_margin` now asserts it and says what to do if it ever
fails: re-run the argument, do not patch around it.

**SCRFD arms it immediately and by a wide margin.** SCRFD regresses box corners
as free floats; a 100 px face at the aspect ratios face detectors actually
produce (w/h ≈ 0.75–0.8) has `|w − h| ≈ 25 px` — eight times the threshold.

---

## 6. Phase 6 — alternatives evaluated

Measured on 50 000 random box sets, free aspect, edges 1–400, floor 64. A "false
abstention" is a file that contained a face clearing every floor and was
published as `FACE_TOO_SMALL` anyway.

| Option | False abstentions | Verdict |
|---|---|---|
| **Rank by area, gate by short edge** (legacy) | 385 / 48 039 — **0.80%** | the defect |
| **Rank by short edge** | **0** | **shipped** |
| **Rank by scoreability, then area** | **0** | rejected — §6.1 |
| Rank by area, then scoreability | — | the same thing; "scoreability first" is the only reading with content, since exact-area ties are what the later components already break |
| **Unified face-size metric** | **0** | **shipped — this is the structural form of "rank by short edge"** |
| Rank by long edge | 7 674 / 48 039 — **15.97%** | rejected; included to show the metric choice is not arbitrary |
| **Explicit dual-policy documentation** | 385 / 48 039 | rejected — §6.2 |

### 6.1 Why not "rank by scoreability"

Filtering to scorable faces before ranking also scores zero, and it preserves
area ordering among the faces that survive. It was rejected on three grounds, in
increasing order of weight:

1. **It leaves two metrics in the system.** `quality.face_px` would still report
   the short edge of a box chosen by area, so R20's defect condition survives and
   the crop/gate/rank triple is still three different questions.
2. **The ordering stops being a function of geometry.** It would depend on
   `MediaCaps.min_face_px`, a **deployment** knob living in a **different**
   fingerprint. Two fingerprints would then jointly determine which face is
   scored — and Batch 12 separated those two on purpose: *the gate decides
   whether an input is scored; selection decides which face and what pixels.*
3. **`ORDERING_RULE` could no longer name the rule.** The name in the identity
   string would be true only relative to a caps value the identity string does
   not contain.

### 6.2 Why not documentation alone

Documenting "we rank on area and gate on short edge" makes the behaviour
discoverable and leaves it wrong. The published `FACE_TOO_SMALL` is a false
statement about the file, not an unusual policy, and a client cannot act on the
documentation anyway — the bundle does not carry the boxes that were discarded.
Documentation was the right answer for R16, where the behaviour is a defensible
product choice for single-subject media. It is not the right answer here.

### 6.3 Why short edge rather than area

Two reasons, and the first is forensic rather than aesthetic.

- **The short edge is what decides whether facial detail is resolvable.** A
  900×50 box has a large area and no vertical detail. Branch A consumes an
  aligned face crop; what limits it is the smaller dimension.
- **`QualityReport.face_px` already documents itself as a short edge** — *"Short
  edge of the largest face"*. Under area ranking that sentence was false whenever
  the two disagreed (R20). Under short-edge ranking it is true. R20 remains
  formally open per this batch's brief, but its defect condition is gone: `face_px`
  now reports the short edge of the largest face by the platform's only face-size
  metric.

Area survives as the **secondary** key, where it does real work: among boxes that
resolve equally well, prefer the one covering more of the image.

---

## 7. Phase 7 — what shipped

### 7.1 One metric, named

```python
FACE_SIZE_METRIC = "short-edge"      # inference/preprocess.py

class FaceBox:
    @property
    def size_px(self) -> int:        # THE face size, for every decision that asks
        return min(self.width, self.height)
```

`size_px` and `short_edge` are equal and are not redundant: `short_edge` is a
fact about a rectangle, `size_px` is the platform's *choice* of which fact to
use. A future migration changes the second without touching the first.

Three call sites, one quantity:

| Site | Before | After |
|---|---|---|
| `face_order_key` leading component | `-box.area` | `-box.size_px` |
| `ImagePipeline._gate` | `faces[0].short_edge < min_face_px` | `faces[0].size_px < min_face_px` |
| `quality.face_px` | `faces[0].short_edge` | `faces[0].size_px` |

The gate and report edits are pure renames — `size_px == short_edge` — and change
no behaviour on their own. The ordering edit is the fix.

### 7.2 The ordering rule

```
before: area-desc,y-asc,x-asc,width-asc,height-asc
after : short-edge-desc,area-desc,y-asc,x-asc,width-asc,height-asc
```

Still a **name**, not a hash of the function, for the reason Batch 12 gave:
bytecode moves with every Python release and every comment edit, which would dump
the result cache on changes that cannot move a box. The name is held to the
behaviour by `test_the_ordering_rule_name_matches_the_ordering_rule`, which
re-derives the order the name describes — now on a box set that separates the
short-edge component from the area component.

**R14 survives.** The key is still a total order and still a function of the
boxes alone; `test_the_ordering_is_still_a_total_order` checks the two pairs that
would break it, including `100x100` against `100x400`, which a short-edge-only
key would tie.

### 7.3 What makes the defect unrepresentable rather than unlikely

`faces[0]` maximises `size_px`; the gate tests `size_px`. So `faces[0]` failing
the floor implies every box fails it — asserted over random geometry at floors
1, 16, 32, 64, 128 and 399, not only the shipped one
(`test_ranking_on_the_gated_quantity_makes_the_defect_unrepresentable`).

The one thing standing between the platform and R19 recurring is
`test_the_ranking_and_the_gate_read_the_same_quantity`, which asserts both that
`face_order_key(b)[0] == -b.size_px` and that the gate's comparison is the one
written in `image_pipeline.py`. That is a test, not a fingerprint, and §10.4 says
so.

---

## 8. Phase 8 — migration impact

### 8.1 What moves

```
ordering rule    before : area-desc,y-asc,x-asc,width-asc,height-asc
                 after  : short-edge-desc,area-desc,y-asc,x-asc,width-asc,height-asc
face-selection   before : d662e22d
                 after  : 19559387
effective_ver    before : stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.d662e22d
                 after  : stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387
string length    49 -> 49 characters
gate fingerprint ae232cba -> ae232cba   (unchanged)
```

`FaceSelectionPolicy.differences` reports exactly one changed field, and it is
`ordering` — asserted, so "one deliberate move" is checked rather than claimed.

**The gate fingerprint does not move, and that is worth stating** because a reader
would assume it did: this is a change to the gate's *honesty*, but what changed
is the metric, and the metric lives with the selection policy where
`ORDERING_RULE` names it. The floor is unchanged at 64.

### 8.2 Artifact by artifact

| Artifact | Impact |
|---|---|
| **Result cache** | One-time full miss, both media kinds. Namespaces do not overlap, so nothing stale is served; old objects stay readable by job id and only the content pointer is orphaned. Cost is one re-analysis per distinct file. Same shape as Batch 9's and Batch 12's. |
| **Job idempotency keys** | `sha256:model_version`, computed per submission. In-flight duplicates re-enqueue once. No stored state. |
| **Face-selection identity** | `d662e22d` → `19559387`. `parse_model_version` is unchanged and total on both eras. |
| **Benchmark reproducibility** | Readable; `RUN_SCHEMA_VERSION` stays `1.2.0`. A pre- and a post-Batch-14 run now compare as **different models**. Under the shipped detector that is *conservative rather than correct* — the two produce identical numbers — and the cost is one refused comparison across the boundary. `RUN_IDENTITY` §3 already settles which way a cache key breaks a tie, and this is the same tie. |
| **Calibration witnesses** | **Untouched.** `CALIBRATION_WITNESS_VERSION` unchanged; no stored witness re-digests. A selection change is not a calibration — Batch 12 §5, and asserted. |
| **Stored bundles** | Readable and unchanged in shape. `ModelInfo.version` is a `str` of the same length. |
| **Split locks** | Untouched. |
| **OpenAPI snapshot** | Unchanged. No field added, removed or retyped; the drift guard passes with no snapshot edit. |
| **`CONTRACT_VERSION`** | **Unchanged at 1.0.0.** |
| **Video** | Carries the moved segment without consuming it, exactly as Batch 12 §5.2 argued. A false miss on the video cache namespace — compute, not correctness. |

### 8.3 Backward compatibility

`parse_model_version` is total on all three eras and
`test_old_version_strings_still_parse` pins the two older ones verbatim:

```
parse('...g.ae232cba.f.d662e22d.d.c20a064a')  face_selection = d662e22d   pinned
parse('...g.ae232cba.d.c20a064a')             face_selection = None       not pinned
```

A pre-Batch-12 string still means *this version makes no claim about which face
it chose*; a Batch-12 string means *it chose by area*. Neither is an error and
neither needs migrating to be readable.

---

## 9. Discovered while tracing, not fixed — R23

**`crop_face` scales the crop by the box's long edge, and clips it.**

```
crop_shape = 'square-longest-edge', crop_margin = 1.3
side = max(width, height) * margin

box                side    crop shape   square?   face fraction
200x200@(100,100)   260      260x260      True         59.2%
260x200@(100,100)   338      338x338      True         45.5%
400x100@(100,100)   520      520x520      True         14.8%
900x50@(100,100)   1170     1170x600     False          6.4%
```

Two consequences, neither of them R19:

1. **A free-aspect box is cropped at its long edge**, so the fraction of the
   model's input that is face collapses as aspect diverges — 59% for a square
   box, 6% for a 900×50 one. The gate approves a face on its short edge and the
   crop is then scaled by its long edge: the third metric, on the same box, in
   the same analysis.
2. **A clipped crop is not square**, and `prepare_image` then resizes a rectangle
   to a square — the aspect distortion `crop_face`'s own docstring says it exists
   to avoid.

Both are unreachable under a detector whose boxes are within a pixel of square,
both are armed by the same migration that arms R19, and both are outside this
batch's brief. **Recorded as R23 and pinned** by
`test_the_crop_is_still_scaled_by_the_long_edge`, which fails and asks to be
replaced if the geometry changes — the tripwire pattern Batch 9 established and
Batches 11 and 13 honoured.

---

## 10. Guarantees gained

1. **A file containing a scorable face can no longer be published as
   `FACE_TOO_SMALL`.** Structural, not statistical: `faces[0]` maximises the
   gated quantity, so its failure implies every box's failure. Proved over
   random geometry at six floors.
2. **The published `quality.face_px` is the largest face size present**, not the
   short edge of a box chosen by a different measure.
3. **The published `faces[]` box is the face the gate judged.**
4. **The platform has one face-size metric**, named in `FACE_SIZE_METRIC`, read
   by ranking, gating and reporting, and held to its name by a test.
5. **No published number changes under the shipped detector.** Proved
   exhaustively for every box pair within two pixels of square, and the
   two-pixel margin is now itself asserted against the real cascade.
6. **The change is visible in the deployment identity** — cache key, idempotency
   key, `/health`, alignment check, benchmark comparison — because the ordering
   rule is in the `.f.` segment.
7. **R14 survives.** The ordering is still total and still a function of the
   boxes alone.
8. **The legacy composition is kept executable**, so the defect can be
   re-demonstrated rather than remembered.

## 11. Guarantees NOT gained

1. **R16 is untouched.** One face's verdict is still published as the media's —
   the *right* face now, still one face. `test_r16_is_untouched` pins it,
   including that the manipulated smaller face in a two-face image remains
   invisible to the published verdict.
2. **R20 is not formally closed.** Its defect condition is gone —
   `QualityReport.face_px`'s description is now true of the value it carries —
   but correcting the prose is its own change with its own OpenAPI consideration,
   and this batch's brief was R19 only.
3. **R21 is untouched.** `faces[]` still has no uniqueness or consistency
   invariant; asserted, so the closure cannot be read as wider than it is.
4. **The gate's *metric* is in no fingerprint.** `MediaCaps.gate_fingerprint`
   covers the floor, not the quantity compared against it. Editing `_gate` to
   test `area` would move no identity string anywhere in the platform — only
   `test_the_ranking_and_the_gate_read_the_same_quantity` stands in the way.
   That is the same shape of guarantee `ORDERING_RULE` has had since Batch 12: a
   name held to behaviour by a test rather than by a hash. Recording it rather
   than adding a field, because a separate `face_size_metric` field would
   double-record what `ordering` already names and would still not bind the gate.
5. **R23 is recorded, not fixed.** §9.
6. **R13 is untouched.** Crop instability under ±4 LSB is a property of the
   placeholder detector.
7. **R18's capability half is untouched.** A faceless clip still receives a
   decided verdict; `size_px` appears zero times in `video_pipeline.py`.
8. **SCRFD is not implemented**, and nothing here makes room for it beyond a
   `detector` field that will change value.
9. **The two-pixel margin is measured, not guaranteed.** No dependency pin covers
   OpenCV's rounding path. The assertion is a tripwire, not a contract.

---

## 12. Measured impact

| Measurement | Value |
|---|---|
| Face-size concepts traced | 10 |
| Metrics applied to one box in one analysis | **3** (area, short edge, long edge) → **2** (size, long edge — R23) |
| Decisions reading the face-size metric | 3 (rank, gate, report), now all the same quantity |
| Disagreement cases the brief asked for | 4 asked, **3 constructible, 1 provably empty** |
| Witnesses for the empty case, 200 000 random sets | **0** |
| Activation threshold | **`|w − h| ≥ 3` px**, at every floor |
| Minimal witness | `66x63` beats `64x64` on area; aspect 1.0476 |
| Haar boxes measured | 1 025; max `|w − h|` = **1**; non-square 0.13–0.37% |
| Safety margin before Batch 14 | **2 pixels**, held by nothing |
| False abstentions, legacy, 48 039 live sets | 385 (**0.80%**) |
| False abstentions, shipped | **0** |
| False abstentions, rank-by-long-edge | 7 674 (15.97%) |
| Ordering divergence, free aspect | 10.8% of box sets |
| Ordering divergence, `|w − h| ≤ 2` | **0**, exhaustively |
| Published numbers changed under the shipped detector | **0** |
| `.f.` segment | `d662e22d` → `19559387` |
| Version string length | 49 → 49 characters |
| Gate fingerprint | unchanged, `ae232cba` |
| Contract / schema / OpenAPI changes | 0 |
| `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` bumps | 0 |
| Proof tests failing before the fix | **9** |
| Suite | 41 → **42 suites**, 1066 → **1093 tests**, zero failures |

The nine that failed before the fix and pass after:

```
FAIL test_a_scorable_face_is_never_gated_away_by_another_one
FAIL test_ranking_on_the_gated_quantity_makes_the_defect_unrepresentable
FAIL test_the_activation_threshold_is_three_pixels
FAIL test_the_face_selection_identity_moved
FAIL test_the_gate_metric_is_in_no_fingerprint
FAIL test_the_published_face_px_is_the_largest_face_size_present
FAIL test_the_ranking_and_the_gate_read_the_same_quantity
FAIL test_the_shipped_pipeline_scores_the_file_the_legacy_policy_refused
FAIL test_the_two_orderings_diverge_at_the_threshold
```

---

## 13. Phase 9 — the brief's three questions, answered

**Is R19 a real correctness defect today?**
**No, and it was closer than the record said.** It is a real defect in the
composition — §4 shows the pairing admits a published false statement — but no
input can reach it under the shipped detector, because a witness needs a box
three pixels from square and Haar emits boxes one pixel from square. The margin
is two pixels of OpenCV rounding behaviour, not a structural guarantee, and the
Batch 11 record's reason ("Haar returns square boxes") is factually wrong.

**Is R19 only activated by SCRFD?**
It is activated by **any detector whose boxes can deviate three pixels from
square**, which is a much weaker condition than "free aspect" and is met by every
regression-based detector. SCRFD exceeds it by roughly eightfold on a 100 px
face. The activation condition belongs in the record as a number, not as a
detector name — a profile cascade, a re-trained Haar asset, or an OpenCV change
to its rounding would each arm it without any migration at all.

**Fix now, or with SCRFD?**
**Now**, and the argument is §4.3 rather than a preference. The two orderings
diverge at exactly the deviation that makes R19 reachable, so the fix is provably
output-identical today and provably effective tomorrow. Deferring it means
either (a) shipping it with SCRFD, when it can no longer be proved neutral and
will be entangled with a migration that changes every verdict, or (b) shipping
SCRFD first and fixing R19 after it has begun producing wrong abstentions. The
cost paid now is one cache generation and one refused benchmark comparison across
the boundary. The Batch 11 review's own sequencing note said an afternoon; this
is that afternoon.

---

## 14. Re-running this

```bash
python tests/face_size_policy_study.py     # every number above, ~40 s
python tests/test_face_size_policy.py      # 27 proof tests
python tests/run_all.py                    # 42 suites, 1093 tests
```

The study is standalone and read-only, takes no arguments, and prints every
number in this document with its stage number. It is not in `run_all.py`'s
`SUITES`; `test_face_size_policy` is.

---

## 15. Where things live

| what | where |
|---|---|
| The metric, its name and the ordering rule | `inference/preprocess.py` |
| The gate that reads it | `inference/image_pipeline.py` |
| The moved literal and its warning | `backend/config.py` |
| Proof, doubles, and the asserted limits | `tests/test_face_size_policy.py` (27 tests) |
| The measurements this rests on | `tests/face_size_policy_study.py` |
| The review that raised R19 | `docs/MULTI_FACE_SEMANTICS.md` §10 |
| The identity segment this moves | `docs/FACE_SELECTION_IDENTITY.md` |
| The precedent for pinning an open limit | `docs/RUN_IDENTITY.md` §7 |
