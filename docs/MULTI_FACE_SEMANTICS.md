# Multi-face semantics — architecture review for R16

> **Review only. Nothing was implemented.** The one file added is
> `tests/multi_face_study.py`, a standalone read-only study in the same pattern
> as `decode_study.py`, `band_study.py` and `inference_determinism_study.py`. It
> is not in `run_all.py`'s `SUITES`; the suite is unchanged at **39 suites, 1015
> tests, zero failures**, re-run after the study landed.

**Scope.** What the platform does when a file contains more than one face:
where the face is chosen, which one wins, whether that is deterministic, what is
published, and whether any contract says so. Read from the code, not from the
prior reports; the reports are cited only where they already recorded something
this review confirms or corrects.

**Method.** Every claim below is either a line of source with a file and line
number, or a measurement from `python tests/multi_face_study.py`. Two named
doubles stand in for the two ends the platform does not have yet, and §0 says
exactly what each one replaces.

**State reviewed.** `CONTRACT_VERSION` 1.0.0, R12 closed, R13 open (deferred to
SCRFD), R16 open.

---

## 0. Why this study uses doubles

Haar cannot see the multi-face fixtures. Measured, stage 0:

```
two full-size tiles      -> 0 box(es)  []
one full, one half       -> 0 box(es)  []
one full, one quarter    -> 0 box(es)  []
```

The same synthetic face that `inference_determinism_study.py` detects reliably
on its own is found **neither time** once a second tile shares the canvas. That
is R13's fixture fragility reappearing exactly as it did in `shifted_face_image`,
and it is a fact about the placeholder detector's eyesight, not about
multi-face semantics. Building the scenarios out of real Haar detections would
therefore measure the wrong thing.

So two named doubles:

| Double | Replaces | Why |
|---|---|---|
| `ScriptedDetector` | `preprocess.detect_faces` | Returns a fixed `list[FaceBox]`, passed through the **shipped** `face_order_key`. The pipeline's entire contract with the detector is that list; only the detection is scripted, the ordering under test is real. SCRFD will return boxes of exactly this shape from NMS. |
| `RegionOracleBackend` | real weights | Logit from the crop's own mean luminance: dark ⇒ authentic (−3.0), bright ⇒ manipulated (+3.0), both branches. This is what gives a scenario a *ground truth* — the study paints one region dark and one bright and reads back which one the platform published. The shipped `StubBackend` hashes pixels and cannot. |

Everything between them is shipped code, reached through the real
`analyze_image` on a real event loop: `_gate`, `_abstain`, `_crop_and_score`,
`crop_face`, `prepare_image`, `fuse`, `calibrated_probability`, `decide`,
`build_image_bundle`, and every `EvidenceBundle` validator.

---

## 1. Exactly where face selection occurs

It is not a function. It is **five subscripts of one list in one file**:

| Site | Line | What it decides |
|---|---|---|
| [`image_pipeline.py:132`](../inference/image_pipeline.py) | `face_px=faces[0].short_edge if faces else 0` | the published quality figure |
| [`image_pipeline.py:144`](../inference/image_pipeline.py) | `self._crop_and_score, image, faces[0]` | **the pixels the model sees** |
| [`image_pipeline.py:151`](../inference/image_pipeline.py) | `face = faces[0]` | the published bounding box |
| [`image_pipeline.py:197`](../inference/image_pipeline.py) | `if faces[0].short_edge < self._caps.min_face_px` | **whether the file is scored at all** |
| [`image_pipeline.py:220`](../inference/image_pipeline.py) | `f = faces[0]` | the box published on the abstention path |

The ordering those subscripts depend on is set once, at
[`preprocess.py:372`](../inference/preprocess.py) —
`return sorted(boxes, key=face_order_key)`.

`inference/video_pipeline.py` has **zero** selection sites. Measured, stage 7:

```
'detect_faces' in video_pipeline.py: 0 occurrence(s)
'faces'        in video_pipeline.py: 0 occurrence(s)
'FaceTrack'    in video_pipeline.py: 0 occurrence(s)
'crop_face'    in video_pipeline.py: 0 occurrence(s)
'face_track'   in video_pipeline.py: 0 occurrence(s)
'TRACKING'     in video_pipeline.py: 0 occurrence(s)
```

There is no selection in the video path because there is no detection in the
video path. §9.

---

## 2. faces[0], largest, or highest-confidence?

**`faces[0]`, where the ordering is largest-*area* first** —
[`preprocess.py:357`](../inference/preprocess.py):

```python
return (-box.area, box.y, box.x, box.width, box.height)
```

Two things follow that the codebase's own prose does not currently say.

**Highest-confidence is not representable.** `FaceBox.__dataclass_fields__` is
`['x', 'y', 'width', 'height']`. There is no score field. A confidence-ordered
policy cannot be expressed in the shipped type at all — adding one is part of
the SCRFD migration, not a tweak.

**"Largest" is ambiguous and the two readings disagree.** Measured, stage 2:

```
wide   FaceBox(x=100, y=100, width=400, height=100)  area=40000  short_edge=100
square FaceBox(x=700, y=100, width=190, height=190)  area=36100  short_edge=190

faces[0] by face_order_key = wide      (area wins)
largest short edge         = square
```

These are different faces, and the distinction is load-bearing because the
*gate* tests `short_edge` while the *selection* ranks on `area`. See §10, R19.

---

## 3. Is the ordering deterministic?

**Yes, and provably total.** Measured, stage 3:

```
three equal-area boxes, all 6 emission orders
  shipped face_order_key                    : 1 distinct faces[0]
  area only, stable sort (pre-Batch-10, R14): 3 distinct faces[0]

4000 random boxes -> 4000 distinct keys, 4000 distinct boxes
```

R14 is closed and stays closed. The `LegacyAreaOnlyOrder` double still
reproduces the defect, so the fix is demonstrated rather than assumed.

**This is a statement about reproducibility, not about correctness.** The
policy being reproduced deterministically is still "the biggest face speaks for
the file". R14 made the answer stable; it did not make it right, and the two are
independent — which is why R16 survives R14's closure.

---

## 4. What verdict is published

All eight rows measured through the real `analyze_image`, stage 4. The two
faces carry *identical* pixel content across the first two rows; only their
sizes are swapped.

| Scenario | Faces | Crops scored | Published verdict | p_fake | `faces[]` |
|---|---|---|---|---|---|
| authentic **big** + manipulated small | 2 | **1** | `authentic` | 0.047426 | 1 entry |
| manipulated **big** + authentic small | 2 | **1** | `manipulated` | 0.952574 | 1 entry |
| two manipulated | 2 | **1** | `manipulated` | 0.952574 | 1 entry |
| two authentic | 2 | **1** | `authentic` | 0.047426 | 1 entry |
| no face | 0 | 0 | `insufficient_evidence` / `no_face_detected` | — | 0 entries |
| one tiny face (40 px, floor 64) | 1 | 0 | `insufficient_evidence` / `face_too_small` | — | 1 entry, `status=face_too_small` |
| big manipulated + tiny | 2 | **1** | `manipulated` | 0.952574 | 1 entry |
| tiny listed first + big authentic | 2 | **1** | `authentic` | 0.047426 | 1 entry |

Reading them as the task asked:

**One authentic face + one manipulated face.** The *bigger* one's verdict is
published as the file's, with no indication that a second face existed. Rows 1
and 2 are the same evidence with the sizes exchanged and they publish opposite
verdicts. A manipulated face in a photograph is invisible to this system
whenever someone standing closer to the camera is authentic.

**Multiple manipulated faces.** `manipulated`, at the largest face's
probability — correct by coincidence, not by construction. The other faces'
evidence is never computed, so the confidence is that of one face and is
published as the file's.

**Multiple authentic faces.** `authentic`, same mechanism, same coincidence.

**No face plus one tiny detected face.** Two distinct paths, and the difference
matters. Zero detections ⇒ `no_face_detected` and an **empty** `faces[]`. One
tiny detection ⇒ `face_too_small`, and a `FaceTrack` *is* published carrying
`status=face_too_small` and `p_fake=None` — the `FaceTrack._score_xor_abstain`
validator working as designed. But when a tiny face accompanies a scorable one
(rows 7 and 8), the tiny face is not published as an abstained track: it is
absent from the bundle entirely. Non-selected faces are not abstained, not
counted, not recorded.

Note rows 7 and 8 together: the tiny face is listed *first* by the detector in
row 8 and the answer does not change. That is R14's fix doing its job, and it
isolates R16 cleanly — the defect is the policy, not the ordering.

---

## 5. Do FaceTrack semantics already imply per-face scoring?

**The type does. The docstring explicitly denies it. The pipeline does not
implement it.**

The shape is per-face and always has been —
[`common.py:83`](../contracts/common.py):

```python
class FaceTrack(Base):
    """One identity across the analysed frames. Images yield exactly one track."""
    track_id: int = Field(ge=0)
    ...
    p_fake: float | None = Field(default=None, ge=0.0, le=1.0)
    status: AbstainReason | None = ...
```

A per-face `p_fake`, a per-face abstention `status`, and a `_score_xor_abstain`
validator enforcing that a track either scored or declined. That is the complete
vocabulary for per-face verdicts. It is populated with a **copy of the
image-level number**. Measured, stage 6, on a two-face image:

```
bundle.p_fake   = 0.9525741268224334
faces[0].p_fake = 0.9525741268224334
identical       = True
```

`face_track()` at [`postprocess.py:646`](../inference/postprocess.py) hard-codes
`track_id=0`, so the identifier space is not merely unused — it is pinned to a
single value at the only construction site.

And the docstring — *"Images yield exactly one track"* — is the single-subject
assumption stated in the contract source. It is the one place in the platform
where it *is* written down, and it is written as a fact about the data rather
than as a policy the product chose. §10, R16.

---

## 6. Image-level or face-level? What the API contracts say

**Every published prediction is image-level. Nothing in the API describes a
face-level prediction, and nothing forbids one.**

| Field | Scope as published | Tier |
|---|---|---|
| `verdict`, `p_fake`, `interval` | image | **Stable** |
| `branches[].{logit,p}` | image | **Stable** |
| `quality.face_px`, `quality.gate_passed`, `gate_reason` | image | Stable (`gate_passed`) |
| `disagreement` | image | Evolving |
| `faces[]` | **per-face shape, image-level content** | Evolving |
| `video.*` | image (clip) | Evolving |
| `provenance` | file — C2PA and EXIF are container-level by nature | Evolving |

`README.md` and `contracts/evidence.py` agree on the tiers, and the split is the
single most consequential fact for the options in §12:

- **`faces[]` is Evolving.** Publishing N tracks with real per-face scores
  requires **no `CONTRACT_VERSION` bump**. Clients are already required to
  tolerate it.
- **`p_fake` and `verdict` are Stable**, and the rule is *"Bump MAJOR when a
  stable-tier field changes shape **or meaning**"*. Redefining the headline
  number as an aggregate over faces is a meaning change and **is** a MAJOR bump.

The published JSON Schema confirms `faces[]` is unconstrained:

```json
"faces": {"items": {"$ref": "#/$defs/FaceTrack"}, "title": "Faces", "type": "array"}
```

No `maxItems`. So the wire format already admits the multi-face case, and
`test_openapi`'s drift check would not fire on it.

---

## 7. Does EvidenceBundle assume a single face?

**Structurally, no. Semantically, yes, and unguarded.**

`EvidenceBundle` carries seven `model_validator`s. Every invariant that binds
the verdict binds it to `p_fake`, `threshold`, `quality.gate_passed` and the
branch rows. **Not one of them mentions `faces[]`.** There is a
`_branches_unique` and there is no `_faces_unique`.

Measured directly against the model, re-validated through JSON the way storage
and the wire do:

```
two faces, both track_id=0, contradictory p_fake -> validates: 2 [0, 0] [0.9, 0.1]
bundle.p_fake: 0.9134827362  verdict: manipulated
agrees with either face?: False
```

A bundle claiming `manipulated` at 0.913, carrying two faces scored 0.9 and 0.1,
both with the same `track_id`, and a headline figure matching neither, is a
**valid** `EvidenceBundle` today. This is the exact class of drift the
`ThresholdInfo` band-derivation validator and `_verdict_follows_threshold` were
written to make unrepresentable, left open on the one field R16 would populate.
§10, R21.

---

## 8. Are disagreement, uncertainty and provenance face-scoped or image-scoped?

All three are **image-scoped in publication**; two are **face-scoped in origin**,
and one cannot be face-scoped at all.

**Disagreement** — `disagreement_of` at
[`postprocess.py:234`](../inference/postprocess.py) is the spread between
branches A and B, both of which scored *the same single crop*. It measures
branch disagreement about one face and is published as the file's. It cannot
currently express "two faces in this photograph disagree", which is the more
alarming quantity of the two and the one a group photograph actually raises.

**Uncertainty** — `interval = confidence_interval(p, disagreement)` inherits
that scope exactly. Measured, stage 6: `(0.9226, 0.9826)` on a two-face image
where the second face's evidence was never computed. The interval widens for
branch disagreement about one face and does not widen at all for the existence
of an unexamined face. **The one source of uncertainty that multi-face media
actually introduces is the one the interval is blind to.**

**Provenance** — genuinely file-scoped and correctly so. C2PA manifests and EXIF
describe a container, not a region. This is an architectural constraint on any
per-face design: branch P's authenticity credit is one number that must apply to
every face in the file, so per-face fusion means subtracting the *same* credit
from each face's evidence, not scoring P per face. `fuse()`'s one-sided
subtraction already has the right shape for that; it just has to be applied N
times.

**Quality** — `quality.face_px` is face-scoped content (`faces[0].short_edge`)
published inside an image-scoped block, and its field description says something
different again. §10, R20.

---

## 9. Video with multiple faces across frames

**The video pipeline never detects a face.** It resizes whole frames to the
model input and scores them — `extract_frames(..., target_size=self._spec.input_size)`
at [`video_pipeline.py:128`](../inference/video_pipeline.py), then
`_score_batch` straight onto the resized frame.

Measured, stage 7, on a 30-frame clip containing two faces per frame:

```
verdict=manipulated  p_fake=0.716587
faces[]              : []
quality.face_px      : None
quality.gate_reason  : None
frames_sampled       : 30
frames_scored        : 30    (contract: 'Sampled frames where a face was usable')
top_frames track_ids : [0, 0, 0]
```

Four consequences, all of them measured rather than inferred:

1. **Multi-face video is not mis-attributed; it is un-attributed.** Every face in
   the frame, plus the background, is averaged into one number by the backbone.
   There is no "which face" question because there is no face stage.
2. **The face gates are unreachable for video.** `NO_FACE_DETECTED` and
   `FACE_TOO_SMALL` cannot be produced by `VideoPipeline._abstain`, which emits
   only `TOO_FEW_FRAMES`. A clip containing **no face at all** receives a decided
   verdict — above, a synthetic clip was published as `manipulated` at p_fake
   0.717 with `gate_passed=True` and `gate_reason=None`. The image path would
   have abstained on the same content.
3. **`frames_scored` does not mean what the contract says it means.** The field
   description is *"Sampled frames where a face was usable"*; the value is
   `len(frame_scores)`, i.e. frames that decoded. 30 = 30 here because nothing
   was ever checked for a face.
4. **`top_frames[].track_id = 0` is a dangling reference.** `top_frames()` at
   [`postprocess.py:617`](../inference/postprocess.py) hard-codes `track_id=0`
   into every entry, and `faces[]` is empty for every video bundle. A client
   resolving that id against the bundle finds nothing.

The progress vocabulary records a face stage that does not exist.
`STAGE_WEIGHTS_VIDEO` budgets `DETECTING` at 30 and `TRACKING` at 40; the video
pipeline's only `progress()` call passes `JobStage.SCORING`
([`video_pipeline.py:181`](../inference/video_pipeline.py)), and `TRACKING` is
emitted by nothing anywhere in the platform.

**This is the more important half of R16 and the prior reports do not cover it.**
R16 as written in `INFERENCE_DETERMINISM.md` §9 describes the image path only.
The video path's problem is not that it picks the wrong face — it is that
per-face semantics do not exist there at all, while the contract that video
bundles are validated against says they do.

---

## 10. Findings

Existing finding restated with evidence, plus six new ones. Classification per
the task's four categories.

### R16 — one face's verdict is published as the media's

**policy/semantics** (image path) — *open, confirmed and now measured*

`analyze_image` scores `faces[0]` and publishes one `FaceTrack`. Rows 1–2 of §4
are the demonstration: identical face content, sizes swapped, opposite published
verdicts. Non-selected faces are absent from the bundle rather than abstained.

It is **not** correctness-critical *as specified*, because nothing specifies it —
that is precisely the finding. The product targets single-subject media, for
which the behaviour is correct; the contract nowhere restricts input to
single-subject media, and `FaceTrack`'s docstring asserts *"Images yield exactly
one track"* as though it were a property of images rather than a choice the
pipeline made. A consumer reading the published schema has no way to learn that a
group photograph's verdict describes one person in it.

Upgrade to correctness-critical the moment either happens: the product accepts
group media as in-scope, or a real detector starts finding the second face
reliably (SCRFD will).

### R17 — the face-selection policy and crop margin are outside every identity string

**reproducibility-critical** — *new*

Measured, stage 8:

```
effective_version      : v0-w7f45fada+c.739c3fec.g.ae232cba
min_face_px 64 -> 96   : v0-w7f45fada+c.739c3fec.g.143504db   <- gate moves it
face_order_key         : a module-level function, in no fingerprint
crop_face margin       : default 1.3, a default argument, in no fingerprint
```

`MediaCaps.gate_fingerprint` exists for a reason its own docstring states well:
*"Raising `min_face_px` turns a set of verdicts into abstentions without touching
a single weight, so a cache that cannot see the caps will happily serve the
verdicts the new floor was meant to withdraw."*

Both knobs that decide **which face** and **what crop** have exactly that
property and neither is covered. Changing `face_order_key` from area to short
edge, or `crop_face`'s margin from 1.3 to 1.4, moves `p_fake` on every affected
file while `effective_version` stays byte-identical — so the result cache and the
job idempotency key both go on serving results computed under the old policy.

**This is the finding that gates R16's implementation.** Any R16 fix *is* a
change to face-selection policy. Shipping one without closing R17 first means the
cache serves pre-change verdicts under the post-change version string, which is
the failure R12 and the gate fingerprint were both built to prevent.

### R18 — the video path has no face stage, and the contract says it does

**correctness-critical** (the gate) + **documentation-only** (`frames_scored`) — *new*

§9. A faceless clip receives a decided verdict where the image path abstains; the
`frames_scored` field description is false as shipped; `top_frames[].track_id`
references a track that is never published.

The gate half is correctness-critical on its own terms and independently of R16:
the platform's stated design is that low-information input routes to honest
abstention *before* the model votes, and for video that gate does not exist.

### R19 — the gate reads the largest-area face and tests its short edge

**correctness-critical, latent** — *new*

Measured, stage 5:

```
thin FaceBox(x=100, y=100, width=900, height=50)  area=45000  short_edge=50   (below the 64 px floor)
good FaceBox(x=100, y=300, width=200, height=200) area=40000  short_edge=200  (clears it)
face_order_key picks: thin

verdict=insufficient_evidence  reason=face_too_small
crops scored: 0
faces[] published: [(900, 50, 'face_too_small')]
```

A face 200 px across, passing every floor, was never scored and never published.
The file abstained because a wider, flatter box outranked it on area and failed a
test on short edge. Selection ranks on one metric; the gate tests another.

**Not reachable today**, and it matters why: Haar's frontal cascade returns
square boxes, so area order and short-edge order coincide and the divergence
cannot occur. **SCRFD returns NMS boxes with free aspect ratio.** The migration
is what arms this.

### R20 — `QualityReport.face_px` does not describe the value it carries

**documentation-only today, correctness-critical at SCRFD** — *new*

[`common.py:61`](../contracts/common.py) — *"Short edge of the largest face."*
The value is `faces[0].short_edge`, the short edge of the largest-**area** face.
Stage 2 shows the two differ (100 vs 190 on the same pair of boxes). Same square-
box reasoning as R19: benign under Haar, wrong under SCRFD. It is separately
worth noting the field is `None` for every video bundle, which the description
also does not mention.

> **Superseded by Batch 14, closed in Batch 18.** The claim above that the value
> is *"the short edge of the largest-**area** face"* stopped being true when
> Batch 14 (R19) made `face_order_key` lead with `-size_px` — the same quantity
> the report takes. The area-vs-short-edge divergence, and hence the
> "correctness-critical at SCRFD" prognosis, is structurally gone.
> `tests/test_headline_and_face_semantics.py::test_the_batch_11_reading_of_the_
> value_is_refuted_by_execution` runs the refutation. What was left — an
> unqualified "largest", an undocumented `0` sentinel and an undocumented video
> `null` — was pure prose and is corrected in Batch 18.
> See [`docs/HEADLINE_AND_FACE_SEMANTICS.md`](HEADLINE_AND_FACE_SEMANTICS.md).

### R21 — `faces[]` has no consistency invariant

**policy/semantics**, and a **prerequisite** for any R16 implementation — *new*

§7. Duplicate `track_id`s validate. A headline `p_fake` agreeing with no face
validates. `EvidenceBundle` has a `_branches_unique` and no `_faces_unique`, and
no validator relating `faces[]` to `verdict`/`p_fake` at all.

Harmless while `faces[]` never holds more than one entry. The moment R16 is
implemented, this is the invariant preventing a bundle whose headline
contradicts its own evidence — and building the feature before the invariant
means shipping the representable-inconsistency the rest of this contract works
hard to eliminate.

### R22 — branch B is declared whole-image and scored on the face crop

**documentation-only today, architectural for R16** — *new*

`enums.py:47` and `README.md` both declare branch B as *"DINOv2-B/14 frozen +
probe, **whole image**"*, against branch A's *"aligned face crop"*.
`_crop_and_score` passes **one crop** to `backend.score()` and takes every branch
from that one call, so branch B is de facto face-scoped.

No consequence today — the stub returns a deterministic function of whatever it
is handed, and no real backend exists. It is listed because it decides the shape
of any per-face design: if branch B is genuinely whole-image, then a per-face
fusion has one image-level B and one image-level P shared across N face-level
A's, and `manipulation_evidence`'s renormalised weighted mean has to be
restructured for that. Discovering it during implementation is more expensive
than deciding it now.

### Summary

| # | Finding | Class | Reachable today | Armed by SCRFD |
|---|---|---|---|---|
| R16 | one face's verdict published as the media's | policy/semantics | yes | escalates |
| R17 | selection policy + crop margin outside every identity | **reproducibility-critical** | yes | — |
| R18 | video has no face stage; contract says it does | **correctness-critical** + documentation | yes | — |
| R19 | gate ranks on area, tests short edge | **correctness-critical** (latent) | no | **yes** |
| R20 | `face_px` description ≠ value | documentation-only | description only | **yes** |
| R21 | `faces[]` has no consistency invariant | policy/semantics | harmless at N=1 | — |
| R22 | branch B declared whole-image, scored on crop | documentation-only | no | — |

---

## 11. What this review does not establish

1. **No real-detector measurement.** Every multi-face number above comes from
   `ScriptedDetector`. What SCRFD actually returns on a group photograph — how
   many boxes, what aspect ratios, what confidence spread — is unmeasured, and
   R19's reachability estimate depends on it.
2. **No real-model measurement.** `RegionOracleBackend` is a luminance rule. The
   *distribution* of per-face scores on real group media, which is what any
   file-level aggregation policy must be calibrated against, does not exist yet.
3. **No estimate of how common multi-face media is** in the intended traffic. The
   severity of R16 is proportional to that number and this review has no data on
   it. It is the single most decision-relevant thing missing.
4. **The video finding is measured on one synthetic clip** encoded by `mp4v`.
   The structural claims (no `detect_faces` call, empty `faces[]`, unreachable
   gates) are from source and hold universally; the specific `p_fake` is one
   clip's.

---

## 12. Implementation options, ranked

Ranked by risk, contract impact and migration cost together. Options 0 and 1 are
not alternatives to the others — they are prerequisites, and are listed first
because they are what makes options 2–4 shippable at all.

### Option 0 — Close R17 and R21 first (prerequisite, not a choice)

Fold the selection policy and crop geometry into an identity segment (naturally
an extension of `gate_fingerprint`, which already owns "what changes which
inputs get scored"), and add the `faces[]` invariants to `EvidenceBundle`.

- **Risk:** low. Additive; `effective_version` moves once, deliberately, exactly
  as it did when the gate fingerprint landed.
- **Contract impact:** none. Fingerprints are opaque; validators only reject
  bundles nothing currently produces.
- **Migration cost:** one cache generation, plus the `STUB_ENGINE_VERSION`
  update the repo has done before.
- **Why first:** without it, every option below silently serves pre-change
  verdicts from cache. This is not optional under any of the alternatives.

### Option 1 — Document the single-subject assumption; change no behaviour

State it where a client reads it: the `FaceTrack` docstring stops asserting
*"Images yield exactly one track"* as a fact and states it as the policy it is;
`p_fake` and `verdict` field descriptions say the prediction describes the
largest detected face; `QualityReport.face_px` is corrected (R20);
`VideoAnalysis.frames_scored` is corrected (R18).

- **Risk:** none. No code path changes.
- **Contract impact:** descriptions only. **This does move the OpenAPI snapshot**
  and `test_openapi`'s drift check, which is the intended effect — the assumption
  becomes visible to every client.
- **Migration cost:** zero.
- **Honest limitation:** it converts R16 from an undocumented choice into a
  documented one. It does not make a group photograph's verdict correct, and it
  does nothing for R18's video gate.

### Option 2 — Publish per-face evidence; keep the headline as it is (**recommended**)

Score every face that passes the floors. Publish one `FaceTrack` per face with a
real per-face `p_fake` and per-face `status`. Keep `verdict`, `p_fake` and
`interval` defined exactly as today — the largest face's — and say so.

- **Risk:** medium. N model forwards per image instead of one, so the latency and
  cost model changes; needs a per-file face cap and it interacts with the lease
  heartbeat that `test_lease_safety` guards.
- **Contract impact:** **none requiring a version bump.** `faces[]` is
  Evolving-tier, has no `maxItems`, and clients are already required to tolerate
  additions. The Stable fields keep their current meaning exactly.
- **Migration cost:** low on the wire, real in compute. Needs R21's invariants
  (Option 0) to be honest.
- **Why recommended:** it is the only option that stops *discarding evidence*
  without redefining a Stable field. A UI can then show "3 faces analysed, one
  scored 0.94" while the headline number keeps the meaning every stored bundle
  already has. It also converts R16 from a hidden policy into a visible one, and
  makes the eventual Option 3 a pure aggregation change over data that is already
  being computed and published.

### Option 3 — Redefine the file verdict as an aggregate over faces

Publish `p_fake` as max, or as "manipulated if any face is above `band_hi`".

- **Risk:** **high, and not primarily engineering risk.** Max-over-N multiplies
  the false positive rate. At the shipped `target_fpr` of 0.02 and independent
  faces, file-level FPR runs 0.020 / 0.040 / 0.059 / 0.078 / 0.096 for N = 1…5 —
  a five-person photograph at roughly **five times** the calibrated rate. The
  band helps (the boundary is `band_hi` = 0.7099, not `value` = 0.62, so achieved
  FPR is bounded *below* target) but the multiplication still applies, and the
  platform exists to avoid exactly this false positive: calling a real
  photograph of real people fake.
- **Contract impact:** **`CONTRACT_VERSION` MAJOR bump.** `p_fake` is
  Stable-tier and this changes its meaning; `evidence.py`'s own rule requires the
  bump and a client migration.
- **Migration cost:** highest. Bump, client migration, cache generation, and a
  re-derivation of the operating point — the existing threshold was fitted for a
  per-face decision and cannot be reused for a per-file one.
- **Blocked on data.** Choosing the aggregation rule requires the distribution of
  per-face scores on real group media and the distribution of N in real traffic.
  Neither exists. Doing this before SCRFD and real weights means picking a policy
  and calibrating it against nothing.

### Option 4 — Add a face stage to the video pipeline

Detect, track and crop per frame; make `NO_FACE_DETECTED` / `FACE_TOO_SMALL`
reachable for video; give `frames_scored` its documented meaning; make
`top_frames[].track_id` resolve.

- **Risk:** high. It is the largest change here — detection on every sampled
  frame, plus tracking, which the platform has no implementation of and which
  `JobStage.TRACKING` reserves a slot for. It also changes every video verdict
  the platform currently produces.
- **Contract impact:** none by shape. `VideoAnalysis` and `faces[]` are both
  Evolving; `frames_scored` starts meaning what it says.
- **Migration cost:** high. All video results invalidate.
- **Sequencing:** this is SCRFD work by definition. Running Haar on 48 frames
  inherits R13's 310× amplification 48 times over, and R15's threading hazard
  transfers with it.

---

## 13. Should R16 be fixed before or after SCRFD?

**Split it. The contract and identity work goes before SCRFD; the scoring policy
goes after. Neither half is sound on its own schedule.**

**Before SCRFD — Options 0 and 1, and they should not wait.**

- **R17 must land before any face-selection change ships, whenever that is.**
  It is independent of the detector, it is cheap, and it is the precondition for
  every other option. Shipping a selection change without it repeats R12 on a new
  knob.
- **R19 and R20 are armed by the migration itself.** Both are unreachable today
  purely because Haar returns square boxes and both become live the moment SCRFD
  returns free-aspect NMS boxes. Fixing them *after* SCRFD means fixing them
  after they have started producing wrong abstentions. Doing it before costs an
  afternoon and makes them unreachable by construction.
- **R21 costs nothing now and is a prerequisite later.** Adding validators to a
  field that currently holds at most one entry is free; adding them after
  multi-face bundles exist means auditing what already validated.
- **R18's documentation half** is a description correction with no code risk.
- **Option 1 is the cheapest honest thing available.** R16's own prior record
  says the choice is *"currently neither"* documented nor fixed. Documenting it
  is a day's work and removes the worst property of the finding — that a client
  cannot discover it.

**After SCRFD — Options 3 and 4, necessarily.**

- **Option 3 cannot be calibrated before SCRFD.** The FPR arithmetic in §12
  needs the distribution of N and of per-face scores on real group media. Neither
  is obtainable from the placeholder detector, and a threshold fitted for a
  per-face decision cannot be reused for a per-file one. Picking an aggregation
  rule now means picking it blind and then refitting.
- **Option 4 is SCRFD work by definition.** Per-frame Haar detection inherits
  R13's 310× amplification on every one of 48 frames and drags R15's threading
  requirement along with it.
- **A confidence-ordered or landmark-quality-ordered selection policy is not
  representable today.** `FaceBox` has four integer fields and no score. The
  type gains one with SCRFD; ordering policy should be decided against the real
  thing, not retrofitted twice.

**Option 2 sits on the boundary, and that is the real scheduling decision.**
Publishing per-face evidence is implementable now against Haar and would stop the
platform discarding evidence. But under Haar it would publish per-face scores
from a detector that R13 measures as unstable at 310× amplification on 25.7% of
decisions, and stage 0 shows finds neither face on a two-face canvas — so the
per-face data published would be mostly a report on the placeholder's eyesight.

**Recommendation:** build Option 2's *plumbing* alongside SCRFD, land Options 0
and 1 now. Concretely, and in order:

1. **Now:** R17 (identity), R21 (validators), R20 + R18-doc (descriptions),
   Option 1 (state the single-subject assumption in the contract). No behaviour
   changes; one deliberate `effective_version` move.
2. **Now, separately:** decide R22 — is branch B whole-image or crop? It shapes
   the fusion restructuring and costs nothing to settle on paper.
3. **With SCRFD:** R19 (unify the selection and gate metrics), Option 2 (per-face
   scoring and publication), Option 4 (the video face stage). All three are
   detector work and all three are cheaper done together than in sequence.
4. **After SCRFD has real scores on real group media:** Option 3, if the traffic
   data justifies the FPR cost. Not before.

The one thing that should **not** happen is R16 staying as it is through the
SCRFD migration. R19 and R20 are latent today and armed by that migration
specifically; going through it without closing them converts two documentation
notes into two live defects.

---

## 14. Re-running this

```bash
python tests/multi_face_study.py
```

Standalone, read-only, no arguments, ~1.7 s. Prints every number in this document
with its stage number. Not in `run_all.py`'s `SUITES` — the suite is unchanged at
39 suites, 1015 tests.
