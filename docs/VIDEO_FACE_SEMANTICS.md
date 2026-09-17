# Video face semantics: the stage that was described but never built

**Status:** S8 Batch 13. **R18 partially closed** — its correctness half is
closed and its documentation half is closed; **the capability half is
deliberately not closed and is now pinned by a test.** R13, R16, R19–R22
unchanged (open; this batch touches none of them).
**Contract impact:** none. `CONTRACT_VERSION` 1.0.0. No field added, removed or
retyped; no OpenAPI signature change; no schema snapshot regenerated. One
published value changed and one invariant was added — both argued in §6.
**Suite:** 41 suites, 1066 tests, zero failures.

> **Batch 23 (R29) update.** The video path's frame-selection and pooling
> parameters — `n_frames`, `min_usable_frames`, the top-k pooling depth, and the
> `"uniform"` / `"topk_mean"` rule names — are now fingerprinted, via
> `VideoSelectionPolicy` folded into the model identity's `.f.` segment. That is
> reproducibility, not a face stage: the video *face* stage is still R18 / SCRFD
> work, and `VideoAnalysis` still publishes `faces=[]` and a null
> `quality.face_px`. N4 also corrected the stale `VideoAnalysis.sampling` default
> `"uniform+adaptive"` → `"uniform"`. See `docs/VIDEO_SELECTION_IDENTITY.md`.

---

## 1. Phase 1 — every video-path use of a face-related field

Traced exhaustively. Measured against a live analysis of a clip containing no
face anywhere (`tests/video_face_semantics_study.py` stage 1).

| Field | Video value | What it means |
|---|---|---|
| `faces[]` | `[]` | never populated — no detector is called |
| `FaceTrack` | none constructed | `video_pipeline.py` references the type **0** times |
| `track_id` (on `faces`) | n/a | no track exists to carry one |
| `top_frames[].track_id` | was `[0, 0, 0]`, now `[None, None, None]` | **the one code-level incoherence — fixed** |
| `frames_scored` | `30` of `30` sampled | frames that decoded; never a face count |
| `quality.face_px` | `None` | nothing measured a face |
| `quality.gate_passed` | `True`, always | no face gate exists |
| `quality.gate_reason` | `None` | only `TOO_FEW_FRAMES` is reachable |
| `NO_FACE_DETECTED` | unreachable | `VideoPipeline._abstain` cannot emit it |
| `FACE_TOO_SMALL` | unreachable | same |

Plus the progress vocabulary: `STAGE_WEIGHTS_VIDEO` budgets `DETECTING` at 30 and
`TRACKING` at 40, and the pipeline's only `progress()` call passes
`JobStage.SCORING`. `TRACKING` is emitted by nothing anywhere in the platform.

---

## 2. Phase 2 — five artifacts, and the implementation is the odd one out

The brief asks whether contract, implementation, documentation and tests agree.
They do not, and the survey turned up a fifth voice the Batch 11 review had not
counted: **the shipped fixtures**.

| Artifact | Detects? | Tracks? | `frames_scored` means |
|---|---|---|---|
| Contract prose (pre-B13) | implied | implied | *"where a face was usable"* |
| **Fixtures** (`contracts/fixtures.py`) | **yes** | **yes** | 46 scored, `FaceTrack.frames_seen=46` |
| **Test double** (`tests/stub_handlers.py`) | **yes** | **yes** | = frames sampled |
| **Implementation** | **no** | **no** | 30 frames that decoded |
| Tests (pre-B13) | silent | silent | only monotonicity asserted |

`video_manipulated` publishes a `FaceTrack` with `frames_seen=46` — a face
*tracked across frames* — and `top_frames[].track_id=0` that **resolves** to it.
The fixtures are internally coherent and describe a system with a face stage.
`stub_handlers` does the same. Both are what the frontend and the worker tests
build against.

**This sharpens R18 considerably.** It is not "a docstring is stale". It is that
the specification artifacts describe a face stage, the implementation has none,
and nothing in the test suite noticed.

### 2.1 Three kinds of disagreement, and only one is a defect

Separating these is the whole of what Batch 13 could act on:

| Disagreement | Kind | Action |
|---|---|---|
| Fixtures and doubles depict a face stage | **intent** — they are the post-SCRFD target the frontend builds against | annotate; do **not** rewrite |
| Prose describes a stage that is absent | **documentation defect** | correct the prose |
| `top_frames` names a track nothing publishes | **correctness defect** | fix the value, and make the state unrepresentable |

Rewriting the fixtures to match today's output was considered and rejected: it
would erase the design intent the whole contracts-first approach exists to carry,
and the fixtures are not wrong about the target — they are silent about the
present. Annotating them costs nothing and loses nothing.

---

## 3. Phase 3 — the correctness defect, reproduced

Every video bundle the platform published carried three references to a track
that did not exist. `top_frames()` hard-coded `track_id=0`; `faces` was empty.

Executed against `LegacyTopFrameTrackId` — the pre-Batch-13 `top_frames()`,
verbatim (study stage 3):

```
legacy top_frames  -> track_ids [0, 0, 0]
shipped top_frames -> track_ids [None, None, None]

A: yesterday's bundle (legacy ids, faces=[])
   REFUSED   video.top_frames name track(s) [0] but faces[] publishes [];
             a client resolving those ids finds nothing.
B: today's bundle (null ids, faces=[])
   ACCEPTED  track_ids=[None, None, None]
C: the post-SCRFD shape the fixtures depict (ids that resolve)
   ACCEPTED  track_ids=[0, 0, 0] resolve to faces[0]
```

**A is the fail-before evidence, executed rather than asserted:** that exact
bundle validated before this batch and cannot be constructed now. C is the shape
a real face stage produces, and it still validates — the invariant forbids
*dangling*, not *referencing*.

Six proof tests failed before the fix and pass after:

```
FAIL  test_a_dangling_track_id_is_unrepresentable
FAIL  test_the_frames_scored_description_no_longer_claims_a_face_stage
FAIL  test_the_openapi_signature_is_unchanged_by_the_prose_edits
FAIL  test_the_shipped_top_frames_helper_names_no_track
FAIL  test_the_track_id_description_explains_what_null_means
FAIL  test_video_top_frames_do_not_name_a_track_that_does_not_exist
13/19 passed        ->   19/19 passed
```

---

## 4. Phase 4 — downstream consumers

**Nothing computes anything from any of these fields.** That is why the
correction is cheap.

| Field | Role | Read by any computation? | Consumer |
|---|---|---|---|
| `frames_scored` | published only | **no** | `test_timing_contract` asserts it grows with `n_frames` |
| `top_frames[].track_id` | published only | **no** | nothing in-repo resolves it; a UI would |
| `faces[]` | published only | now read by the new referential validator | — |
| `quality.face_px` | published only | **no** | null on every video bundle |
| `STAGE_WEIGHTS_VIDEO` | stage → percent | yes, for progress | `DETECTING`/`TRACKING` entries are dead but harmless |

The blast radius of changing `track_id` from `0` to `null` is the published
payload only. No in-repo consumer resolved it, and a consumer that did was
resolving it to nothing.

---

## 5. Phase 5 — alternatives evaluated

| Option | What it fixes | What it leaves | Verdict |
|---|---|---|---|
| **Document current behaviour only** | the prose defects | the dangling reference — a real incoherence in every published video bundle | insufficient |
| **Remove face semantics from video outputs** (drop `top_frames.track_id`, forbid `faces` on video) | everything, bluntly | it bakes today's limitation into the contract permanently and would forbid the post-SCRFD stage; also breaks the fixtures and the doubles | rejected |
| **Introduce explicit "no face stage" semantics** (a new `VideoAnalysis.face_stage: bool`) | makes "nothing looked" vs "nothing found" machine-readable | a new field for a value that is constant `False` today and would be constant `True` after SCRFD — a schema change carrying one bit of information already implied by `media.kind` | rejected as not minimal; revisit if a client actually needs the distinction before SCRFD |
| **Placeholders/invariants without adding detection** | the dangling reference, structurally; the prose | the capability gap, which no contract change can close | **shipped** |
| **Defer all behavioural change until SCRFD** | nothing | every published video bundle keeps carrying three references to a nonexistent track for the duration | rejected — the defect is live and the fix is four lines |

### 5.1 On the rejected "explicit no-face-stage flag"

There is a genuine semantic hole: `faces == []` on an **image** means *the
detector found none* (and the bundle abstains with `NO_FACE_DETECTED`); on a
**video** it means *nothing ever looked* (and the bundle decides normally). A
consumer cannot tell those apart from `faces` alone.

It is recoverable today — `media.kind == video` implies no face stage — and that
rule is now stated in `VideoAnalysis`, `FaceTrack` and the pipeline docstring.
Encoding it as a field would add schema surface for one bit that is currently a
constant, and the honest version of that field flips meaning the moment SCRFD
lands. Deferred, with the reasoning recorded rather than the option forgotten.

Enforcing the rule with a validator (`kind == video` ⇒ `faces` must be empty) was
also considered and **rejected**: it would reject the shipped fixtures and the
stub handlers, and it would forbid at the contract level the very stage the
frozen architecture calls for. A contract must not make its own roadmap
unrepresentable.

---

## 6. Phase 6 — the minimum correction, and why each piece is justified

### 6.1 Correctness (behaviour)

**One published value changed.** `top_frames()` emits `track_id=None` instead of
`0`. `None` is the honest value — no face was attributed to this frame — and it
was already the field's default and its declared type (`int | None`). A
`track_id` parameter was added so a future face stage can pass a real id without
touching the function.

**One invariant added.** `EvidenceBundle._top_frames_name_a_published_track`:
every non-null `video.top_frames[].track_id` must appear in `faces[].track_id`.

Justification for the invariant rather than the value fix alone: fixing the call
site leaves the next caller free to reintroduce the state. This is the move
`_verdict_follows_threshold` already makes for the verdict — the incoherent
bundle becomes unrepresentable by hand, from a factory, and on the way back in
from storage or the wire.

**Scope, stated because it is easy to overreach here.** The invariant checks
referential integrity from `top_frames` into `faces` and **nothing else**. It
says nothing about whether `faces[]` ids are unique or whether the headline
verdict agrees with any of them. That is **R21**, it is a separate finding, and
this batch does not touch it.

### 6.2 Documentation (prose only)

- `VideoAnalysis.frames_scored` — description corrected from *"Sampled frames
  where a face was usable"* to what it counts.
- `VideoAnalysis` class docstring — states plainly that the video path runs no
  face detection, and that the fixtures depict the target rather than the
  present.
- `TopFrame.track_id` — gains a description explaining that null means no face
  was attributed.
- `FaceTrack` — docstring now says images yield one track *by policy* (R16) and
  **video yields none at all** (R18).
- `STAGE_WEIGHTS_VIDEO` — comment recording that `DETECTING` and `TRACKING` are
  budgeted and emitted by nothing, and that they are kept because they are the
  post-SCRFD plan.
- `video_pipeline.py` module docstring — the three consequences, up front.

### 6.3 Why this is not a `CONTRACT_VERSION` bump

Four independent reasons, each checkable:

1. **No field changed shape.** `VideoAnalysis` and `TopFrame` have exactly the
   field sets they had (`test_the_video_analysis_field_set_is_unchanged`).
2. **`video.*` is Evolving-tier**, per `contracts/evidence.py` and the README.
   Even a reshape would not require a MAJOR bump; this is less than a reshape.
3. **`track_id` was already `int | None` defaulting to `None`.** Emitting the
   default is not a type or nullability change, and a client that already handled
   the declared type handles it.
4. **The corrected prose describes the value the code always produced.**
   `frames_scored` did not change; the sentence describing it did. A description
   that was wrong becoming right is not a meaning change to the field.

### 6.4 Why no OpenAPI snapshot regeneration

`backend/openapi.py`'s structural signature **ignores prose by design** —
`test_signature_ignores_prose_but_not_structure` establishes it, mutating
`info.description` and a schema description and asserting the fingerprint is
unchanged. So the description corrections produce no drift, and `test_openapi`
passes with no snapshot edit. Asserted for this batch by
`test_the_openapi_signature_is_unchanged_by_the_prose_edits`.

---

## 7. Phase 7 — correctness, documentation, and future SCRFD work, separated

### Correctness fixes — done in this batch

1. `top_frames()` no longer names a track that does not exist.
2. A dangling `top_frames[].track_id` is unrepresentable in an `EvidenceBundle`.

### Documentation fixes — done in this batch

3. `frames_scored` description corrected.
4. `TopFrame.track_id` null semantics documented.
5. `VideoAnalysis` and `FaceTrack` docstrings state what video does and does not do.
6. `STAGE_WEIGHTS_VIDEO`'s dead entries annotated.
7. `video_pipeline.py` states the absence of a face stage in its first paragraph.
8. Fixtures annotated as depicting the target rather than the present.

### Future SCRFD work — explicitly NOT done

9. **Face detection on video frames.** A Haar-based stage would inherit R13's
   310× crop instability on every one of 48 frames rather than once, and R15's
   thread-safety requirement transfers to whatever replaces it.
10. **Face tracking across frames.** The platform has no tracker;
    `JobStage.TRACKING` reserves the slot.
11. **A face gate for video**, so a faceless clip abstains as an image would.
    This is the correctness half of R18 that survives this batch, and it cannot
    be closed without 9.
12. **Per-face scoring and aggregation** for video — R16's territory, out of
    scope by brief.

---

## 8. Guarantees gained

1. **No published bundle names a face it does not publish.** Proved against the
   real pipeline and enforced structurally.
2. **The refusal is total** — hand-built, factory-built and deserialised bundles
   are all checked, so a stored artifact cannot reintroduce the state.
3. **The contract still admits a real face stage.** The post-SCRFD shape the
   fixtures depict validates unchanged.
4. **`frames_scored` means what it says**, and what it always measured.
5. **A reader of `video_pipeline.py`, `VideoAnalysis`, `FaceTrack` or
   `TopFrame.track_id` learns there is no face stage** without grepping.
6. **The fixtures no longer imply a capability silently** — they are labelled as
   the target.
7. **Behaviour is otherwise unchanged**: same verdicts, same `p_fake`, same
   `frames_scored` values, same determinism.

## 9. Guarantees NOT gained

1. **A faceless clip still receives a decided verdict.** Measured on pure noise:
   `manipulated` at p_fake **0.979**, `gate_passed=True`, `gate_reason=None`.
   The image path abstains with `NO_FACE_DETECTED` on equivalent content. This is
   R18's capability half and it is SCRFD work.
   `test_a_faceless_clip_still_receives_a_decided_verdict` pins it, and fails
   loudly if it ever changes — the same tripwire pattern Batch 9 used and Batch
   11 honoured.
2. **Video verdicts are still whole-frame scores**, not face scores, through a
   branch A the architecture declares to consume an aligned face crop. That
   mismatch is R22's shape applied to video and is untouched.
3. **`faces == []` is still ambiguous across media kinds** — "found none" versus
   "never looked" — recoverable only via `media.kind`. §5.1.
4. **The fixtures and `stub_handlers` still depict a face stage.** Annotated, not
   rewritten, deliberately.
5. **`JobStage.TRACKING` is still emitted by nothing.**
6. **R16, R19, R21 and R22 are untouched**, per brief.

---

## 10. Discovered in passing, not fixed

**`fixtures_out/*.json` are stale and no test guards them.** Every committed
bundle carries the pre-Batch-7 probability-symmetric band (`band_lo: 0.52`,
`band_hi: 0.72`) against the shipped log-odds band `[0.5210, 0.7099]`, so **none
of them validates against the current contract**:

```
fixtures_out/image_authentic.json   band=[0.52, 0.72]
current shipped band                band=[0.5210, 0.7099]
```

These files feed the frontend's TypeScript generator. `test_schema` exercises the
*generator* against in-memory examples; nothing asserts that the committed output
is current. This is a pre-existing condition dating to Batch 7, unrelated to R18,
and out of scope for this batch — recorded here so it is not discovered a third
time. Regenerating them is `python -m contracts.schema` plus the fixture writer,
and it wants its own change with its own review.

---

## 11. Measured impact

| Measurement | Value |
|---|---|
| Artifacts describing a video face stage | 4 (contract prose, fixtures, stub handlers, progress weights) |
| Artifacts implementing one | **0** |
| Dangling track references per video bundle | 3 → **0** |
| Video bundles affected | all of them |
| Downstream computations reading the changed field | **0** |
| Proof tests failing before the fix | **6** |
| Faceless-clip verdict (unchanged, pinned) | `manipulated`, p_fake 0.979 |
| Fields added or removed | 0 |
| OpenAPI signature change | none — prose is excluded by design |
| Schema snapshot regeneration | not required |
| `CONTRACT_VERSION` | unchanged at 1.0.0 |
| Suite | 40 → **41 suites**, 1047 → **1066 tests**, zero failures |

---

## 12. R18 status

**R18 — the video path has no face stage, and the contract says it does:
PARTIALLY CLOSED.**

| Half | Status |
|---|---|
| Dangling `top_frames[].track_id` | **Closed.** Value fixed and the state made unrepresentable. |
| `frames_scored` description | **Closed.** |
| Contract prose implying a face stage | **Closed.** `VideoAnalysis`, `FaceTrack`, `TopFrame`, the pipeline docstring. |
| Progress vocabulary claiming unrun stages | **Closed as documentation.** Weights retained as the post-SCRFD plan. |
| Fixtures/doubles depicting a face stage | **Closed as documentation.** Annotated as the target, not rewritten. |
| **A faceless clip receives a decided verdict** | **OPEN.** Requires a video face stage. SCRFD work. Pinned by a test. |

---

## 13. Where things live

| what | where |
|---|---|
| The referential invariant | `contracts/evidence.py` |
| The corrected helper | `inference.postprocess.top_frames` |
| The corrected prose | `contracts/video.py`, `contracts/common.py`, `contracts/job.py`, `inference/video_pipeline.py` |
| Proof, doubles, and the asserted limits | `tests/test_video_face_semantics.py` (19 tests) |
| The trace and the fail-before evidence | `tests/video_face_semantics_study.py` |
| The review that raised R18 | `docs/MULTI_FACE_SEMANTICS.md` §9 |
| The precedent for pinning an open limit | `docs/RUN_IDENTITY.md` §7 |
