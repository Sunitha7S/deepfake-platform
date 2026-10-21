# SCRFD Integration Program, Phase 4 — Cutover Design

**Date:** 2026-09-19. **Type:** design only — no code, no file modifications
other than this document. **Basis:** a fresh source re-derivation this
session of `docs/SCRFD_PROGRAM_DESIGN.md`, `docs/SCRFD_PHASE2_DESIGN.md`,
`docs/SCRFD_PHASE2_SPEC_RESOLUTION.md`, `docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md`,
`REPRODUCIBILITY_AUDIT.md`, `docs/PROJECT_ROADMAP.md`, and the current
`inference/`, `contracts/`, `backend/`, `tests/` source — read live this
session, not accepted from any prior document's own conclusions. Three
things this document found that no prior SCRFD document states plainly are
flagged inline, not folded in silently: (1) Phase 2 already shipped a fully
working `SCRFDDetector` and dispatcher — "Phase 4" here is a **cutover**, not
a build; (2) Phase 3's measurement **methodology** is implemented and
hermetically tested, but has **never been run against real inputs** — Phase
0's asset fetch and the photo-corpus acquisition are both still unexecuted,
so R13 has **no real evidence**, pass or fail; (3) switching
`default_face_selection()` to SCRFD would make **every test that constructs
`ImagePipeline()` with no override** — not only the 8 files prior documents
named — fail closed the moment `$DFP_SCRFD_MODEL_PATH` is unset, which is
every dev machine and any CI run today.

**No code was written to produce this document.**

### Corrections from implementation (2026-10-17) — read these before §4

1. **P5 was implemented as explicit Haar pinning, not a synthetic default
   asset (§4.6 superseded).** A constant-output synthetic ONNX graph "finds"
   a face in a blank image, which breaks every no-face/abstention test, so it
   cannot stand in for Haar. Instead, every test that runs detection on the
   Haar-tuned fixtures now names `tests/detector_fixtures.haar_selection()`
   (output-identical today — same identity, proven), and
   `tests/test_detector_pinning.py` fails if any test runs detection through
   the implicit default. Its `SHIPPED_DEFAULT_DETECTION` allowlist (3 entries)
   is the exact list of tests the cutover must give the real asset or
   rewrite, together with the one suite that drives the production image
   engine (`workers.image.ImageHandler` → `get_image_engine()`):
   `test_worker_inference_integration.py` (grep-verified as the only user of
   the real handler). Construct-only identity tests follow the default by
   design and need only the constant-digest rule in item 2.
2. **§4.2's `detector_asset=scrfd_asset_digest(_scrfd_model_path())` is
   wrong.** It makes the published identity depend on whether a file exists:
   a machine without the asset would report `.f.` over `"unreadable"`, so no
   single `STUB_ENGINE_VERSION` literal could pass `verify_version_alignment`
   everywhere. The SCRFD default must name the recorded digest as a
   **constant** and let `_scrfd_detector_for` verify the file against it at
   detection time. `test_the_engines_report_a_version_without_a_detector_asset`
   now holds this. (The §4.2 sketch also misstates `min_neighbors`: the shipped
   value is 5, not 3.)
3. **`SCRFDDetector` could not decode a dynamic-input export on real photo
   sizes** (fixed in `inference/preprocess.py`). SCRFD heads emit `ceil(H/s)`
   cells while the decoder's anchor grid is `H // s`; at an image's own size
   this raised `IndexError` (or misplaced boxes) for any side not a multiple
   of 32. Dynamic exports are now letterboxed into the upstream reference's
   fixed 640×640, held by three new tests in `tests/test_scrfd_detector.py`.
   Recommendation for the cutover: fold this input size into
   `FaceSelectionPolicy` in the same `.f.` move, rather than leaving a
   verdict-affecting constant outside the identity.

---

## 1. Re-derived current state (verified live this session, not assumed)

### 1.1 `FaceDetector` Protocol — unchanged since Phase 1

`inference/preprocess.py:912-930`, `@runtime_checkable`:

```python
class FaceDetector(Protocol):
    def detect(self, image: np.ndarray, *, policy: "FaceSelectionPolicy") -> list[FaceBox]: ...
```

One method, geometry in, geometry out, no ordering guarantee (`face_order_key`
sorts once, outside every detector, at `detect_faces`'s dispatch level,
`:1277`). Confirmed unchanged from Phase 1/2's own description.

### 1.2 `HaarDetector` — unchanged, still the shipped default's detector

`:933-961`. Wraps `_face_cascade().detectMultiScale(...)` verbatim, reads
`scale_factor`/`min_neighbors`/`detect_min_px` from the policy. One shared
module-level instance (`_haar_detector`, `:964`). No changes since Phase 1.

### 1.3 `SCRFDDetector` — already fully implemented, not a Phase 4 deliverable

**Correction to how prior documents frame Phase 4:** `docs/
SCRFD_PROGRAM_DESIGN.md` §5 describes Phase 4 as needing to build the
detector. It does not — `SCRFDDetector` (`:971-1166`) is complete and green
today:

- Verbatim port of the SCRFD authors' own `detection/scrfd/tools/scrfd.py`
  reference (anchor-center generation, `distance2bbox`, greedy-IoU NMS with
  the authors' own `+1` pixel-inclusive area convention) — primary-source
  read, not reimplemented from a paper (`docs/SCRFD_ASSET_AUDIT.md` §1.5).
- One `onnxruntime.InferenceSession` per thread (`threading.local`, `:1028`),
  mirroring `_cascade_local`'s R15 fix exactly — verified live by
  `tests/test_scrfd_detector.py::test_the_onnx_session_is_not_shared_unsafely`
  (20 concurrent calls × 8 threads, zero cross-contamination).
- `providers=["CPUExecutionProvider"]` hardcoded unconditionally (`:1036`);
  `onnxruntime.get_available_providers` is never called — verified by
  `test_cpu_execution_provider_is_hardcoded_not_selected`
  (`inspect.getsource` grep, not a behavioral proxy).
- Raises `ValueError` if `policy.confidence_threshold`/`policy.
  nms_iou_threshold` is `None` (`:1044-1051`) — refuses to guess a
  Haar-shaped policy's missing thresholds rather than defaulting silently.
- Reads only the first `2 * len(_FEAT_STRIDES)` output names (`:1061`) —
  works unmodified against a 6-output (box-only) or 9-output (`_KPS`,
  landmark-emitting) SCRFD export, but **never requests or decodes
  landmarks** (`:988-995`) — `FaceBox` stays pure geometry.

**Nothing in this design changes `SCRFDDetector` itself.** Phase 4's job is
making it reachable from the shipped default, not building it.

### 1.4 `FaceSelectionPolicy` — 12 fields, all identity-bearing

`:432-618`. Full current field list, confirmed by reading `identity`
(`:562-577`), `to_dict` (`:579-595`), and `differences` (`:597-617`) — all
three enumerate the same 12 keys, unconditionally:

| Field | Detector | Added |
|---|---|---|
| `detector` | both | original (R17) |
| `detector_asset` | both | original (R17) |
| `scale_factor` | Haar only | original (R17) |
| `min_neighbors` | Haar only | original (R17) |
| `detect_min_px` | both (gate uses it too) | original (R17) |
| `ordering` | both (label) | Batch 14 |
| `crop_margin` | both | original (R17) |
| `crop_shape` | both | original (R17) |
| `interpolation_rule` | both (label, R25/R28) | Batch 22 |
| `crop_clip_rule` | both (label, R23) | Batch 22 |
| `confidence_threshold` | SCRFD only, `None` under Haar | **SCRFD Phase 2** |
| `nms_iou_threshold` | SCRFD only, `None` under Haar | **SCRFD Phase 2** |

`FaceSelectionPolicy` has exactly **one constructor call site** in the
repository (`default_face_selection()`, `:621-663`) — every test builds a
policy via `dataclasses.replace(default_face_selection(), ...)`.
`test_every_policy_field_moves_the_identity` (`tests/
test_face_selection_identity.py`) asserts the field set by hard equality, so
a future field addition is loud, not silent — re-confirmed live, unchanged
from `docs/SCRFD_PHASE2_DESIGN.md` §1.2's finding.

### 1.5 `detect_faces` — dispatch already live, production branch unchanged

`:1231-1277`:

```python
policy = policy or default_face_selection()
if policy.detector == "opencv-haar-frontalface-default":
    detector = _haar_detector
elif policy.detector.startswith("scrfd-"):
    detector = _scrfd_detector_for(policy.detector_asset)
else:
    raise UnreadableMedia(f"Unknown face detector {policy.detector!r}.")
boxes = detector.detect(image, policy=policy)
return sorted(boxes, key=face_order_key)
```

`default_face_selection()` (`:621-663`) still returns
`detector="opencv-haar-frontalface-default"`, `confidence_threshold=None`,
`nms_iou_threshold=None` — **the shipped branch is byte-for-byte what Phase 1
shipped.** The `scrfd-` branch is reachable only from a policy a test or
study script constructs explicitly. **Phase 4's entire architectural change,
narrowed to its smallest true form, is: make `default_face_selection()`
return an SCRFD-shaped policy instead of a Haar-shaped one.** Everything
downstream of that one function already dispatches correctly — this is the
direct payoff of Phase 1/2's abstraction work.

### 1.6 Asset resolution — `scrfd_asset_digest` / `_scrfd_detector_for`

`:1169-1228`. `scrfd_asset_digest(path)` — `sha256(bytes)[:16]`, `@cache`
(unbounded, keyed by path, unlike `cascade_digest`'s `@lru_cache(maxsize=1)`
because the path is configurable). `_scrfd_detector_for(detector_asset)` —
`@cache` keyed by the **digest**, reads `$DFP_SCRFD_MODEL_PATH` directly via
`os.environ.get(...)` (never through `backend.config.Settings`, preserving
the `inference`→`backend` DAG-acyclicity rule), and **raises
`UnreadableMedia` if the env var is unset or the on-disk file's digest
doesn't match the policy's `detector_asset`** (`:1212-1227`). This fail-loud
behavior is deliberate and correct for a test/study context — §3.5 below is
why it becomes a production hazard the moment `default_face_selection()`
depends on it.

### 1.7 Contract layer — unchanged since before the SCRFD program

`contracts/common.py:86-90`, `BoundingBox.x`/`.y` still `ge=0` (R26 Tier
2-option-b, not decided). `FaceTrack.landmarks` (`:121-123`) still exists,
still unpopulated — `inference/postprocess.py:675-685`'s `face_track()`
takes no landmarks parameter, confirmed by reading it this session, not
assumed from the spec-resolution document's own claim.

### 1.8 Crop geometry — Option A, unchanged

`crop_face`/`resize` (`inference/preprocess.py:909-964`-region, by line
numbers in the pre-SCRFD file — re-verified this session to be
byte-identical to Batch 22's shipped behavior). R23's occupancy-collapse
half and R28's `resize` predicate fix are both still deferred — this
document does not change that; §6.5 restates the sequencing decision rather
than re-deriving it.

### 1.9 SCRFD study tooling — methodology implemented, **never run for real**

`tests/scrfd_detector_study.py` implements the corrected, pooled,
multi-realisation, bootstrap-CI methodology specified in `docs/
SCRFD_PHASE3_MEASUREMENT_PLAN.md` (`measure()` takes `k_realisations=11`,
pools every `(image, realisation)` trial, `bootstrap_amplification_ci`/
`bootstrap_max_short_edge_ci` cluster-resample by image, `report()` gates on
the CI's upper bound). `tests/test_scrfd_measurement_methodology.py` (17
tests, registered in `run_all.py` as `S8-B29`) hermetically proves the
aggregation formulas behave as specified — including a fail-before/pass-after
pair showing the pre-correction, one-realisation-per-image formula is
undefined (`nan`) on a single-fixture corpus, where the pooled formula is
well-defined.

**This is methodology validation only.** Running `python tests/
scrfd_detector_study.py` today, live, re-confirmed this session:

```
DFP_SCRFD_MODEL_PATH   : (not set / file missing)
DFP_SCRFD_STUDY_PHOTOS : (not set / directory missing)
```

Exits 0, no verdict. `docs/SCRFD_ASSET.md` (re-read this session) states its
own status in its first line: **"Phase 0 (asset acquisition) has not been
executed"** — no license acceptance recorded, no SHA-256 recorded for
either the release zip or the extracted detector file, no real photo corpus
acquired. **There is no gap between "the tool is ready" and "the tool has
been run" that this document can close by writing more code or more design —
it is a licensing-gated human step (`docs/SCRFD_ASSET.md` §2) and a
photo-corpus acquisition step (`docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md`
§3.3), neither of which is an engineering task.**

### 1.10 Version constants — re-verified live this session

| Constant | Value | Location |
|---|---|---|
| `CONTRACT_VERSION` | `"1.0.0"` | `contracts/enums.py:16` |
| `RUN_SCHEMA_VERSION` | `"1.2.0"` | `evaluation/benchmark.py:108` |
| `CALIBRATION_WITNESS_VERSION` | `"1.3.0"` | `evaluation/calibration_protocol.py:74` |
| `LOCK_VERSION` | `"1.1.0"` | `evaluation/split_lock.py:81` |
| `STUB_ENGINE_VERSION` | `"stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.f8244ad0.d.c9d4025c"` | `backend/config.py:41` |

`.f.` = `f8244ad0` (moved once already, by Phase 2's dataclass extension —
`e629642c` → `f8244ad0`, per `docs/SCRFD_PHASE2_SPEC_RESOLUTION.md` Decision
7). `.d.` = `c9d4025c` (last moved by Batch 26/R30, unrelated to SCRFD).

### 1.11 Test counts — re-verified live this session

`python tests/run_all.py`: **57 suites, 1365 tests, 0 failures** (confirmed
via a full run at the start of this session's research; `git status`
unchanged before and after, so no concurrent-session drift affects this
count). This is 1 suite / 17 tests ahead of the count `docs/
SCRFD_PHASE2_SPEC_RESOLUTION.md` describes, reflecting this session's own
prior addition of `S8-B29` (the Phase 3 methodology regression suite) —
consistent, not drifted.

### 1.12 REPRODUCIBILITY_AUDIT.md — R13's row, unchanged

`REPRODUCIBILITY_AUDIT.md:103`, re-read verbatim this session:

> R13 | Face-crop instability amplifies decoder drift | determinism |
> Medium–High | **Open** — quantified in B10 at 310× amplification, 25.7% of
> decisions; deferred to SCRFD with an acceptance criterion

Still **Open**. Nothing in this program has closed it, because nothing has
measured SCRFD against the criterion for real (§1.9, §2).

---

## 2. R13 re-evaluated against the newly implemented measurement methodology

**Answered directly, as instructed, not deferred:** R13 **cannot be
re-evaluated** — there is no measurement to evaluate. The methodology fix
(this session's own prior work, `docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md`)
corrected a **statistical defect** in how the amplification ratio would be
computed; it did not and could not produce a **number**, because computing
one requires the two inputs named in §1.9 that do not exist:

1. A real, fetched SCRFD ONNX asset (`docs/SCRFD_ASSET.md` §Status: not
   fetched — Phase 0's licensing-acceptance step has not run).
2. A real corpus of ≥ 200 real photographs meeting `docs/
   SCRFD_PHASE3_MEASUREMENT_PLAN.md` §3.2's diversity requirements (face-size
   range, lighting, compression, pose, single/multi-face) — not acquired.

**What exists today:** a corrected, hermetically-tested tool that will
produce a trustworthy number **the moment both inputs exist**, and 17 tests
proving the tool's arithmetic is right on synthetic data. **What does not
exist:** any evidence, in either direction, about whether the actual SCRFD
asset actually clears the acceptance criterion. This is not "R13 is
probably fine" or "R13 is probably still a problem" — it is genuinely
unmeasured, and this document treats that as the load-bearing fact it is,
per this project's own standing discipline (`docs/SCRFD_PROGRAM_DESIGN.md`
§9.3, `docs/SCRFD_PHASE2_SPEC_RESOLUTION.md` §11): **a missing measurement
is not evidence for proceeding, and this project does not ship an identity
move — let alone a default-detector cutover — on structural argument alone
once a measurement is the specified gate.**

**Evidence required before Phase 4 can proceed, restated as a checklist
rather than prose so it is independently verifiable:**

1. A human reads and consciously accepts `docs/SCRFD_ASSET.md` §2's
   non-commercial-research-only license for the exact environment(s) this
   is fetched into (`docs/SCRFD_PHASE2_SPEC_RESOLUTION.md` §12's
   precondition, still unmet).
2. `buffalo_l.zip` is fetched, the detection member extracted and its
   filename confirmed (still unverified against `det_10g.onnx`), both
   SHA-256 values recorded in `docs/SCRFD_ASSET.md` (Decision 3's TOFU
   chain).
3. A ≥ 200 (recommended 300) real-photograph corpus is acquired, its own
   license/provenance recorded (`docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md`
   §3.3 — still an open, human decision).
4. `DFP_SCRFD_MODEL_PATH`/`DFP_SCRFD_STUDY_PHOTOS` point at both, and
   `python tests/scrfd_detector_study.py` is run **at least twice with
   different `$DFP_SCRFD_STUDY_SEED` values** (§5.4 of the measurement
   plan), confirming both runs' CIs are consistent.
5. The result clears **all three** checks, with the amplification check
   gated on the **95% bootstrap CI's upper bound**, not the point estimate
   (`docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md` §4.4): zero face losses across
   every trial, max short-edge movement ≤ 1%, and the CI upper bound < 10×.

**None of items 1–5 can be produced by this document, by more design work,
or by writing more code.** This is exactly the prerequisite gap instruction
7 asks this document to identify explicitly rather than paper over — see
§10's gate.

---

## 3. New finding this session: the full-suite blast radius of the cutover

No prior SCRFD document traces this precisely; it follows directly from
reading `_scrfd_detector_for`'s fail-loud behavior (§1.6) against every
production call site of `default_face_selection()`.

### 3.1 The mechanism

`inference/image_pipeline.py:51`: `self._selection = face_selection or pre.
default_face_selection()`. Any test that constructs `ImagePipeline()` with
**no** `face_selection=` override — which is the common, idiomatic pattern
across this codebase's test suite — gets whatever `default_face_selection()`
currently returns. Today that is Haar, which is entirely self-contained (the
cascade XML ships inside the `opencv-python` wheel — zero external asset,
zero environment dependency). **The moment `default_face_selection()`
returns an SCRFD-shaped policy, every one of those call sites routes through
`_scrfd_detector_for`, which raises `UnreadableMedia` if
`$DFP_SCRFD_MODEL_PATH` is unset** (§1.6) — true on every developer machine
and any CI run today, since no asset is fetched (§1.9) and CI has never
executed in the first place (`docs/PROJECT_ROADMAP.md` §0).

### 3.2 Exact scope, grep-verified this session

`grep -rl "ImagePipeline()" tests/*.py` (bare construction, no override):

| File | Call sites | In `docs/SCRFD_PROGRAM_DESIGN.md` §5 Phase 4's list? |
|---|---|---|
| `tests/test_inference.py` | 13 | Yes |
| `tests/test_timing_contract.py` | 5 | **No — not previously named** |
| `tests/test_inference_determinism.py` | 2 | Yes |
| `tests/test_branch_b_semantics.py` | 2 | **No — not previously named** |
| `tests/test_lease_safety.py` | 1 | Yes |
| `tests/branch_b_semantics_study.py` (study, not in `run_all.py`) | 1 | No |
| `tests/face_selection_study.py` (study) | 1 | No |
| `tests/inference_determinism_study.py` (study) | 1 | No |

25 call sites across 5 production suites plus 3 study scripts — **wider than
the "8 files" figure every prior SCRFD document has repeated**, because that
figure was derived from which files' *assertions* reference Haar-specific
geometry (a narrower question) rather than which files' `ImagePipeline()`
construction would newly require an external asset to run **at all**. This
is not a contradiction of the prior documents — the 8-file list is still
correct for "files whose test *content* needs re-deriving because a
near-square-box assumption breaks." This is a **different, additional**
list: files that would **fail to construct a pipeline at all**, for a purely
mechanical reason (a missing environment variable), regardless of whether
their assertions say anything about box shape.

### 3.3 Why this matters for Phase 4's design, not just its risk register

A cutover design that only asks "does `default_face_selection()` return
SCRFD" without also asking "can a bare `ImagePipeline()` still be
constructed in every environment this suite runs in" ships a suite that is
**red everywhere `$DFP_SCRFD_MODEL_PATH` is unset** — which, per §1.9, is
everywhere today. This is not a corollary of the identity move; it is an
independent architectural requirement Phase 4 must solve **before** or
**as part of** flipping the default, not something the full-suite re-run
will merely reveal as expected churn. §6.3 below proposes the mitigation.

---

## 4. Phase 4 architecture

### 4.1 What does *not* need designing (already shipped, Phase 2)

- The `FaceDetector` Protocol, `HaarDetector`, `SCRFDDetector` (§1.1–§1.3).
- The dispatch mechanism in `detect_faces` (§1.5).
- `FaceSelectionPolicy`'s SCRFD-carrying fields (§1.4).
- `CPUExecutionProvider` pinning (§1.3, Decision 5 — done, tested).
- The asset-digest/override mechanism (`scrfd_asset_digest`,
  `_scrfd_detector_for`, `$DFP_SCRFD_MODEL_PATH`) (§1.6, Decisions 3/4 —
  done, tested).

**Phase 4's actual scope is four things**, each new relative to Phase 2
because Phase 2 deliberately kept SCRFD out of every shipped path (§1.5):

1. Flip `default_face_selection()` to return an SCRFD-shaped policy.
2. Give the test/dev/CI environment a way to construct a default
   `ImagePipeline()` without the licensed 288 MB asset (§3, new finding).
3. Add a production-facing, fail-closed startup check that the configured
   SCRFD asset actually resolves — mirroring OR1/OR1-b's pattern for model
   version, not inventing a new one.
4. Decide, explicitly, whether `FaceTrack.landmarks` gets populated in this
   phase (cheap, contract-neutral) and whether R26 Tier 2-b's box-constraint
   question rides along (a real `CONTRACT_VERSION` question, separable).

### 4.2 `default_face_selection()` cutover

```python
@lru_cache(maxsize=1)
def default_face_selection() -> FaceSelectionPolicy:
    return FaceSelectionPolicy(
        detector="scrfd-10g-kps",                       # was "opencv-haar-frontalface-default"
        detector_asset=scrfd_asset_digest(_scrfd_model_path()),  # was cascade_digest()
        scale_factor=1.1,                                 # unchanged, Haar-only field, now inert
        min_neighbors=3,                                   # unchanged, Haar-only field, now inert
        detect_min_px=32,                                  # unchanged — still the size gate
        ordering=ORDERING_RULE,                            # unchanged
        crop_margin=1.3,                                   # unchanged
        crop_shape="square-longest-edge",                  # unchanged (Option A retained, §1.8)
        interpolation_rule="area-if-either-axis-exceeds-target-else-linear",  # unchanged
        crop_clip_rule="intersect-image",                  # unchanged
        confidence_threshold=0.5,                          # NEW real value — was None
        nms_iou_threshold=0.4,                              # NEW real value — was None
    )
```

**Open, load-bearing sub-question this document does not pre-empt:** the
exact `confidence_threshold`/`nms_iou_threshold` shipped values must be
whatever Phase 3's *real* measurement (§2) was run against — `0.5`/`0.4`
above are the values Phase 2's own tests already use as a working
convention, not a ratified production choice. **Changing either value after
Phase 3's measurement is complete is itself a `.f.`-moving change** (§5),
so the measurement must run against the values Phase 4 intends to ship, not
a placeholder later swapped in without re-measuring.

**`scale_factor`/`min_neighbors` staying non-`None` under an SCRFD-selecting
policy is intentional, not an oversight** — `SCRFDDetector.detect()` never
reads them (§1.3), and `FaceSelectionPolicy`'s "no defaults" rule (§1.4)
requires every field to have an explicit value regardless of which detector
is selected. This is the same "label present, not read" pattern
`interpolation_rule` already established for a different pair of fields.

`_scrfd_model_path()` — a new, small helper, **not** a `FaceSelectionPolicy`
field (Decision 4 stands unchanged) — resolves the asset location for the
*shipped* configuration. This is the one genuinely new composition-root
question Phase 2 deferred:

```python
def _scrfd_model_path() -> str:
    """Where the shipped default's SCRFD asset lives. Mirrors
    workers/runner.py's `$DFP_MODEL_VERSION` pattern — an env var read
    directly, never through backend.config.Settings (preserves the
    inference -> backend DAG-acyclicity rule, unchanged from Decision 4)."""
    path = os.environ.get("DFP_SCRFD_MODEL_PATH")
    if not path:
        raise UnreadableMedia(
            "SCRFD is the configured default detector but $DFP_SCRFD_MODEL_PATH "
            "is not set.",
            detail="See docs/SCRFD_ASSET.md for the fetch and configuration steps.",
        )
    return path
```

This makes `default_face_selection()` — today a pure, argument-free function
— **read an environment variable**. That is a real behavioral change to a
function every test in the suite calls, stated plainly rather than
minimized: `default_face_selection()` is `@lru_cache(maxsize=1)`, so the
env var is read **once per process**, but which value gets cached now
depends on process-start-time environment state, a property it did not have
before. §4.4 names the mitigation (a startup check that fails before any
request is served, so a missing/wrong asset is a boot-time failure, not a
per-request one masked by the cache).

### 4.3 Model loading strategy

Unchanged from Phase 2 (§1.3): lazy `import onnxruntime` inside
`SCRFDDetector._session()`, one `InferenceSession` per thread via
`threading.local`, constructed on first use per thread. **Phase 4 adds
nothing here** — the loading strategy was already designed for production
use, not merely for isolated unit tests; Phase 2's own docstring already
states the R15 parallel explicitly (§1.3).

### 4.4 ONNX Runtime provider pinning

Unchanged, already resolved and tested (§1.3, Decision 5): hardcoded
`providers=["CPUExecutionProvider"]`, `onnxruntime.get_available_providers`
never called, a permanent regression test already in place
(`test_cpu_execution_provider_is_hardcoded_not_selected`). **Nothing to
design here for Phase 4.**

### 4.5 Asset management — the one genuinely new production concern

Three distinct environments now need the asset, each with a different
answer:

| Environment | Needs the real, licensed 288 MB asset? | Mechanism |
|---|---|---|
| **Production / any real deployment** | **Yes** — this is the whole point of the cutover | `$DFP_SCRFD_MODEL_PATH` set by whatever deploys the service (container image bake-in, mounted volume, or fetched at container-build time) — **not designed further here**, because `docs/SCRFD_PROGRAM_DESIGN.md` §4 point 1 (re-confirmed unchanged this session: no Dockerfile, no exercised CI, no hosting target) means there is no concrete deployment surface to wire this into yet. This document names the requirement; a Production Deployment program (`docs/PROJECT_ROADMAP.md` Phase 9) owns the actual mechanism. |
| **CI** (`.github/workflows/tests.yml`, unexercised) | **No, if §4.6's mitigation ships** | The synthetic-fixture strategy below |
| **Developer machine running `pytest`/`run_all.py`** | **No, if §4.6's mitigation ships** | Same |

**Recommendation, stated as a recommendation, not pre-empted:** the real
asset is a **production-only** artifact, fetched once per deployment target
via the Decision-2/3 procedure (`docs/SCRFD_ASSET.md`), never committed
(`.gitignore`'s pre-existing `*.onnx` rule stands). Test and CI environments
never need it, by design (§4.6) — this keeps Phase 4 from silently making
"can I run the test suite" contingent on "have I accepted a non-commercial-
research-only license and downloaded 288 MB," which would be a materially
worse contributor experience than this project has had at any prior phase.

### 4.6 Mitigation for §3's finding: a synthetic default-fixture asset for tests

**New design contribution of this document**, filling the gap §3 found.
`tests/test_scrfd_detector.py` already builds small, synthetic, ONNX
graphs by hand (`_build_model_bytes`, a `Constant`-plus-broadcast-`Add`
graph matching SCRFD's real 6-output shape, `score_8/16/32`+`bbox_8/16/32`)
for its own isolated unit tests — real `onnxruntime.InferenceSession`
inference, zero licensed weights. The same technique, generalized into a
small, session-scoped fixture, is the mechanical fix:

1. A new helper (e.g. `tests/scrfd_fixture.py`, or a function inside
   `tests/test_scrfd_detector.py` promoted to module scope so it can be
   imported) writes one small, deterministic, hand-verified synthetic
   `.onnx` file to a temp path once per test-process run.
2. `tests/run_all.py`'s `main()` sets `$DFP_SCRFD_MODEL_PATH` to that path
   **before** importing any suite module, if the variable is not already
   set by the caller (so a developer or CI who *does* have a real asset and
   wants to test against it is not overridden).
3. Every `ImagePipeline()`-with-no-override call site (§3.2) now resolves
   `default_face_selection()` successfully against the synthetic fixture —
   the same way it resolves against Haar's self-contained cascade today.

**What this mitigation does and does not achieve, stated honestly:** it
restores "the test suite runs without an external asset," which is
necessary for this project's whole reproducibility-plumbing discipline to
keep working (a suite that cannot run without a licensed binary is not
verifiable by an outside reviewer, and this project's own history — every
`.f.`/`.d.` move to date — depends on `python tests/run_all.py` being
runnable by anyone who clones the repo). It does **not** mean the 8-file
(now: §3.2's wider list) test-content re-derivation is unnecessary — a
synthetic SCRFD fixture still returns **free-aspect boxes** on some inputs,
still has a different confidence/NMS behavior than Haar's cascade, and every
assertion built on Haar's near-square-box guarantee still needs re-deriving
against whatever the *synthetic* fixture actually returns, which is a
**second, separate design decision** (what does the synthetic default
fixture detect, and on what images) that this document flags but does not
resolve — it depends on which of the currently-Haar-shaped fixture images
(`_face_tile()` and friends) the synthetic default is built to agree with,
a question that touches every file in §3.2's list and cannot be answered
generically here.

**pytest is declared but not required** (`tests/test_dependencies.py::
test_pytest_is_declared_but_deliberately_not_imported`) — this mitigation
is scoped to `tests/run_all.py`'s own bespoke runner, matching this
project's existing test-infrastructure convention, not a `conftest.py`
(which presumes pytest is the execution path; it is declared for the dev
extra but this codebase's actual CI-equivalent runner is `run_all.py`,
confirmed at `tests/run_all.py:3-4`'s own docstring).

### 4.7 Startup fail-closed check — new, mirrors OR1/OR1-b exactly

`workers/engine_registry.verify_version_alignment` + `workers/runner.
verify_engine_alignment_at_startup` (called from `backend/worker_main.py`)
already solve the general shape of this problem for `STUB_ENGINE_VERSION`:
fail before the service starts serving, not on the first request. Phase 4
needs the same pattern for the SCRFD asset specifically, because
`_scrfd_detector_for`'s existing check (§1.6) only fires **on first use**,
inside a request handler, which is `verify_engine_alignment_at_startup`'s
whole reason for existing one layer up (OR1's original gap).

**Recommended shape, mirroring the existing call site exactly (no new
pattern invented):**

```python
def verify_scrfd_asset_at_startup(policy: FaceSelectionPolicy | None = None) -> None:
    """Resolves the shipped default's SCRFD asset once, at boot, so a
    missing/wrong asset is a startup failure (loud, before traffic is
    served) rather than the first request's failure (per-request,
    discovered by whichever caller happens to hit it first)."""
    policy = policy or default_face_selection()
    if not policy.detector.startswith("scrfd-"):
        return  # Haar-shaped default (rollback state, §7) — nothing to verify
    _scrfd_detector_for(policy.detector_asset)  # raises UnreadableMedia if wrong/missing
```

Called from `backend/worker_main.py`'s composition root, alongside the
existing `verify_engine_alignment_at_startup(...)` call — same file, same
pattern, no new composition-root wiring style introduced. **This is new
code Phase 4 must write; it does not exist today** (confirmed — `grep -rn
"verify_scrfd" .` outside this document returns nothing).

### 4.8 Landmarks — cheap, contract-neutral, optional within Phase 4

`FaceTrack.landmarks` already exists in the wire schema, unpopulated
(§1.7). `SCRFDDetector` currently reads only a 6-or-9-output graph's first
`2*fmc` outputs, never the `kps_*` ones (§1.3) — populating landmarks
requires: (a) reading the additional `kps_8/16/32` outputs when present
(the `_KPS` variant, which Decision 1 already ratified as the fetched
variant), (b) decoding them the same way `distance2bbox` decodes box
offsets, (c) threading them from `SCRFDDetector.detect()` through to
whatever calls `inference.postprocess.face_track()`, which needs a new
optional parameter. **No `CONTRACT_VERSION` change** — the field already
exists; this only starts writing to it. **Recommendation:** in scope for
Phase 4 if time allows, but separable — nothing else in this design depends
on landmarks being populated, and deferring it costs nothing but the
capability itself.

### 4.9 R26 Tier 2-b — explicitly *not* bundled by default

Loosening `BoundingBox.x`/`.y`'s `ge=0` constraint (so an edge-crossing
SCRFD box can be published un-clamped) is a genuine `CONTRACT_VERSION`
question (`docs/R16_R26_SPEC_RESOLUTION.md` Decision 1 already deferred it
pending a real detector). **Recommendation, restated from `docs/
SCRFD_PROGRAM_DESIGN.md` §5 Phase 5 and not re-litigated:** keep this a
**separate, optional Phase 5**, not bundled into Phase 4's cutover — a
`CONTRACT_VERSION` bump changes the wire contract every downstream consumer
depends on, and bundling it with a detector swap makes a test failure after
Phase 4 ambiguous about which of two identity moves caused it, violating
this project's one-identity-move-at-a-time discipline (R17, R25, R29 all
cited previously for the same reason).

---

## 5. Version-impact axes — re-verified for the actual Phase 4 scope above

| Axis | Current value (§1.10) | Phase 4 impact |
|---|---|---|
| `CONTRACT_VERSION` | `"1.0.0"` | **None**, for the scope in §4.1–§4.8. `FaceTrack.landmarks` (§4.8) is additive-only to an already-existing, already-optional field — no shape change. **Would move (MINOR at least) only if §4.9's R26 Tier 2-b is bundled in** — explicitly not recommended (§4.9). |
| `RUN_SCHEMA_VERSION` | `"1.2.0"` | **None.** `BenchmarkRun`'s shape is untouched; Phase 4 only *uses* `paired_bootstrap`/`regression_gate` (unexercised so far — no A/B run has actually been executed, a gap named in §8's risk register), it does not extend them. |
| `CALIBRATION_WITNESS_VERSION` | `"1.3.0"` | **None, schema-wise** — unchanged, already generic over `pre_calibration_identity` (R31), which reads `face_selection_fingerprint` automatically. The **digest value** any future witness computes moves once `.f.` moves (below); **zero stored witnesses exist anywhere in this repository** (re-confirmed this session, `grep -rn "CalibrationWitness(" --include=*.json` and the same search every prior SCRFD document has run), so there is nothing to migrate. |
| `LOCK_VERSION` | `"1.1.0"` | **None — structurally unrelated**, unchanged finding from every prior SCRFD document; no code path connects `FaceSelectionPolicy`/`FaceDetector` to `evaluation/split_lock.py` (re-grepped this session, zero references either direction). |
| `.d.` | `c9d4025c` | **None.** Detection is not a decode-environment fact; nothing in §4's scope reads or writes `inference/decode_env.py`. |
| `.f.` | `f8244ad0` | **Moves a second time.** `default_face_selection()`'s **values** change (`detector`, `detector_asset`, `confidence_threshold`, `nms_iou_threshold` all move from their Haar/`None` values to real SCRFD ones) — this is a **behavioral**, not merely structural, move, unlike Phase 2's `None`-valued dict-shape move. One cache generation, both media namespaces (the N8-accepted architectural cost, paid five times already: B20/B22/B23/B26/Phase-2). |
| `STUB_ENGINE_VERSION` | `"...+c.739c3fec.g.ae232cba.f.f8244ad0.d.c9d4025c"` | New `.f.` segment (value TBD — computed from whatever `confidence_threshold`/`nms_iou_threshold`/`detector_asset` Phase 4 actually ships, §4.2); `.d.c9d4025c` **unchanged**. |

**Net: exactly one axis moves (`.f.`, second time, a real value change) —
same as every prior finding's own analysis, re-confirmed rather than
assumed. `CONTRACT_VERSION` stays put unless §4.9 is explicitly opted into,
which this document recommends against for this phase.**

---

## 6. Failure modes

| # | Failure | Trigger | Detection | Mitigation |
|---|---|---|---|---|
| F1 | `$DFP_SCRFD_MODEL_PATH` unset in production | Deployment misconfiguration | `verify_scrfd_asset_at_startup` (§4.7) raises before serving traffic | Fail-closed by design — the service does not start |
| F2 | `$DFP_SCRFD_MODEL_PATH` set but points at a stale/different asset | Re-deploy without updating the path, or a shared-filesystem race | `_scrfd_detector_for`'s digest check (§1.6) raises `UnreadableMedia`, caught by F1's startup call | Same as F1 — a stale asset is indistinguishable from a missing one at this check |
| F3 | Every test/dev/CI run breaks the moment the cutover ships, absent §4.6 | The mechanism in §3 | Immediate, total — the entire suite goes red on the first `ImagePipeline()` call | §4.6's synthetic-fixture mitigation — **this is a blocking prerequisite for shipping Phase 4 at all**, not an optional nicety |
| F4 | The asset license changes on a future upstream release | Out of this project's control | No automated detection (`docs/SCRFD_ASSET_AUDIT.md` §7 item 3, unchanged) | Manual periodic re-read, not scheduled — a named, accepted residual |
| F5 | A real deployment's `onnxruntime` build resolves a different execution-provider set than dev/CI | Environment drift | Not detected today — `.d.` does not cover ONNX providers (N10, `docs/SCRFD_PHASE2_DESIGN.md` §10 item 7, still open) | Out of Phase 4's scope; §8 lists it as a pre-existing, unclosed finding this phase does not worsen (hardcoded provider, §4.4, already prevents the *behavioral* half of this risk — only the *identity-completeness* half remains open) |
| F6 | Two concurrent requests race on `_scrfd_detector_for`'s `@cache` during the first call after boot | Cold-start concurrency | Untested — `@cache`/`@lru_cache` are documented thread-safe for the *cache lookup itself* in CPython, but the *constructed* `SCRFDDetector.__init__` does no I/O (only stores `model_path`, §1.3), so this is low-risk by construction; not separately verified this session | Recommend a dedicated concurrency test for `_scrfd_detector_for` itself (distinct from `SCRFDDetector.detect()`'s own, already-tested, per-thread-session concurrency) — named as a Phase 4 test-file addition, §7 |
| F7 | Phase 3's real measurement (§2) fails the criterion after Phase 4's code is otherwise ready | The actual gate | `tests/scrfd_detector_study.py`'s `report()` | **Full stop, no partial ship** — §9's abort condition, unchanged from every prior SCRFD document |

---

## 7. File list (exact, for the scope in §4.1–§4.8, excluding §4.9)

### 7.1 Production files

| File | Change |
|---|---|
| `inference/preprocess.py` | `default_face_selection()` cutover (§4.2); new `_scrfd_model_path()` helper; new `verify_scrfd_asset_at_startup()` (§4.7) — or this may live in `backend/worker_main.py` directly if kept out of `inference/` to preserve the module boundary `docs/SCRFD_PHASE2_SPEC_RESOLUTION.md` Decision 4 established; **if §4.8 is in scope**, `SCRFDDetector.detect()` gains landmark decoding |
| `backend/worker_main.py` | `verify_scrfd_asset_at_startup(...)` call alongside the existing `verify_engine_alignment_at_startup(...)` (§4.7) |
| `backend/config.py` | `STUB_ENGINE_VERSION` literal regenerated (§5) |
| `inference/postprocess.py` | **Only if §4.8 is in scope** — `face_track()` gains a `landmarks` parameter |

**Not changed:** `inference/image_pipeline.py` (calls `detect_faces`/
`default_face_selection()` by name, unaffected by what they return —
confirmed unchanged reasoning from Phase 2); `inference/video_pipeline.py`
(never calls `detect_faces`, re-confirmed unchanged, §1 of `docs/
SCRFD_PROGRAM_DESIGN.md` §2.7); `contracts/*` (no shape change unless §4.9
is separately opted into); `evaluation/benchmark.py` (used, not changed).

### 7.2 Test infrastructure files

| File | Change |
|---|---|
| `tests/run_all.py` | §4.6's synthetic-fixture bootstrap in `main()`, before any suite import runs detection for real |
| A new small module (e.g. `tests/scrfd_fixture.py`) | The synthetic default-asset builder (§4.6), factored out of `test_scrfd_detector.py`'s existing `_build_model_bytes` so both files can use it without duplication |

### 7.3 Test files needing content re-derivation (not just literal updates)

Per `docs/SCRFD_PROGRAM_DESIGN.md` §5 Phase 4's original list, **widened by
§3.2's finding**:

| File | Why it needs re-deriving, not just re-pinning |
|---|---|
| `tests/test_crop_geometry.py` | Assertions assume Haar's near-square-box guarantee |
| `tests/test_crop_identity.py` | Same, plus `.f.` literal pins |
| `tests/test_face_selection_identity.py` | `alternatives` dict, `.f.` literal pins |
| `tests/test_face_geometry_publication.py` | The "clamp is a structural no-op under Haar" claim becomes false the moment the synthetic/real SCRFD fixture can emit an edge-crossing box |
| `tests/test_headline_and_face_semantics.py` | `.f.`/model-id literal pins |
| `tests/test_inference.py` | 13 bare `ImagePipeline()` sites now resolve against the synthetic fixture (§4.6), not Haar — any assertion keyed to Haar's specific detection behavior needs re-deriving |
| `tests/test_inference_determinism.py` | Same reasoning, 2 sites |
| `tests/test_lease_safety.py` | Same reasoning, 1 site |
| `tests/test_timing_contract.py` | **Newly identified this session** — 5 bare `ImagePipeline()` sites, not in any prior document's list |
| `tests/test_branch_b_semantics.py` | **Newly identified this session** — 2 bare `ImagePipeline()` sites |
| `tests/test_dependencies.py` | `STUB_ENGINE_VERSION`/model-identity pins (`test_stub_engine_version_is_unchanged`, `test_model_identity_is_unchanged` — both currently assert the *pre*-cutover literal and must be updated to the *post*-cutover one, or they become the loud, correct failure this project's process expects) |
| `tests/test_run_identity.py`, `tests/test_video_selection_identity.py`, `tests/test_startup_alignment.py`, `tests/test_decode_identity_completeness.py`, `tests/test_image_decode_identity.py` | Collateral `STUB_ENGINE_VERSION`/model-id literal pins (the standard sweep every `.f.`-moving batch performs, `grep -rn "f8244ad0"`) |

**Study scripts re-run, not modified** (matching every prior phase's
convention): `tests/crop_geometry_study.py`, `tests/crop_identity_study.py`,
`tests/decode_study.py`, `tests/face_selection_study.py`, `tests/
face_size_policy_study.py`, `tests/inference_determinism_study.py`, `tests/
multi_face_study.py`, `tests/video_face_semantics_study.py`, `tests/
branch_b_semantics_study.py`.

### 7.4 New test files

| File | Purpose |
|---|---|
| `tests/test_scrfd_cutover.py` (new) | `default_face_selection().detector.startswith("scrfd-")` after cutover; `verify_scrfd_asset_at_startup` raises on a missing/mismatched asset and passes on a correct one; F6's concurrency check on `_scrfd_detector_for` itself (distinct from `SCRFDDetector.detect()`'s already-tested concurrency) |
| `tests/test_scrfd_fixture_bootstrap.py` (new, or folded into the above) | Proves every bare `ImagePipeline()` construction succeeds without `$DFP_SCRFD_MODEL_PATH` set in the ambient environment, i.e. §4.6's mitigation actually works — this is the single most important new test in Phase 4, since its absence is F3 |

### 7.5 Documentation files

| File | Change |
|---|---|
| `docs/SCRFD_PHASE4_CUTOVER_DESIGN.md` | this document |
| `REPRODUCIBILITY_AUDIT.md` | R13's row updated to **Closed** only after §2's real measurement clears (not by this document, not speculatively) |
| `docs/SCRFD_ASSET.md` | §3's checksums filled in once Phase 0 actually runs (a human step, not this document's) |

---

## 8. Fail-before / pass-after test matrix

| # | Check | Before Phase 4 | After Phase 4 (contingent on §2's gate clearing) |
|---|---|---|---|
| 1 | `python tests/run_all.py` | 57 suites, 1365 tests, 0 failures (§1.11) | Same suite count (no new production suite unless §4.9), test count net of §7.3/§7.4's additions and removals, **0 failures** |
| 2 | `default_face_selection().detector` | `"opencv-haar-frontalface-default"` | `"scrfd-10g-kps"` (or whichever variant §2's measurement validated) |
| 3 | Bare `ImagePipeline()` construction, no env var set | Succeeds (Haar, self-contained) | **Must still succeed** (§4.6's synthetic fixture) — this is the regression Phase 4 must not introduce |
| 4 | `verify_scrfd_asset_at_startup()` with `$DFP_SCRFD_MODEL_PATH` unset and the default SCRFD-shaped | N/A (function does not exist) | Raises `UnreadableMedia` before any request is served |
| 5 | `verify_scrfd_asset_at_startup()` with the correct asset present | N/A | Passes silently |
| 6 | A face near an image border, real/synthetic SCRFD detection | Haar's boxes are ≤ 1 px from square (measured) | Free-aspect boxes reachable — the R23-occupancy mechanism becomes live (still not fixed, per §1.8 — this test proves reachability, not a fix) |
| 7 | `STUB_ENGINE_VERSION` | `.f.f8244ad0` | New `.f.` segment; `.d.c9d4025c` unchanged |
| 8 | R13's acceptance criterion (`tests/scrfd_detector_study.py`) | Not measurable (§1.9/§2) | **Must have already cleared, before this phase starts** — this is a precondition, not a Phase-4-produced artifact |
| 9 | `FaceTrack.landmarks` on a real analysis | `[]`, always | **Only if §4.8 is in scope** — non-empty 5-point list |

---

## 9. Migration analysis

- **Stored `FaceSelectionPolicy`/`CalibrationWitness`/`BenchmarkRun`
  artifacts:** none exist anywhere in this repository (re-confirmed this
  session, same search every prior SCRFD document has run) — **no migration
  required for any stored value.**
- **`fixtures_out/*.json`/`schema_out/*.json`/`backend/openapi.snapshot.
  json`:** `FaceSelectionPolicy` is not a wire type; unaffected unless §4.9
  (R26 Tier 2-b) is separately opted into, in which case `contracts/
  common.py`'s JSON Schema output changes and these regenerate.
- **The SCRFD asset itself:** a first acquisition (§2 items 1–2), not a
  migration in this codebase's sense.
- **Test-suite dependency migration:** the real, new migration cost this
  phase introduces that no prior `.f.`-moving batch has had — every test
  environment (dev machines, CI-if-it-ever-runs) needs §4.6's synthetic
  fixture wired in before this phase can land without breaking the suite
  wholesale. This is a **one-time infrastructure cost**, paid once, not a
  recurring migration.
- **Conclusion:** zero stored-artifact migration (matching every prior
  `.f.`-moving batch); one new, one-time test-infrastructure cost (§4.6,
  §7.2) that is new **in kind**, not merely in scale, relative to any prior
  batch — every previous `.f.` move only had to update literal pins, never
  had to solve "how does the test suite run at all."

---

## 10. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| **Phase 4 ships without §4.6's synthetic-fixture mitigation** | **Critical** — the entire test suite goes red immediately, in every environment | §4.6/§7.4 make this a named, first-class deliverable, not an afterthought discovered during implementation |
| **R13's real measurement (§2) is skipped or its result is ignored** | **Critical** — this is the one no-go condition every SCRFD document has stated and restated | §9 below restates the abort condition explicitly; this document adds no new authority to override it |
| §5's `.f.` move ships without its full collateral-literal sweep (widened list, §7.3) | High if it happens, mechanical to catch | Full-suite re-run is the net, as always — a missed literal is a loud failure |
| The synthetic default-fixture's detection behavior does not meaningfully exercise what the 8 (now wider) re-derived tests need to assert | Medium — could produce tests that pass vacuously against a fixture engineered to agree with old Haar-shaped expectations, rather than genuinely re-deriving them | Named explicitly in §4.6 as a separate, unresolved design question — flagged, not silently assumed solved by "the fixture exists" |
| `_scrfd_detector_for`'s `@cache` under cold-start concurrency (F6) | Low, unverified | A dedicated test, §7.4 |
| Production asset provisioning mechanism does not exist yet (no Dockerfile, no CI, no hosting target) | Medium, but out of this program's scope by design (`docs/SCRFD_PROGRAM_DESIGN.md` §4 point 1, re-confirmed unchanged) | Named in §4.5 as a Production Deployment program dependency, not solved here |
| Licensing terms change on a future upstream release (F4) | Low today | No automated mitigation; periodic manual re-read, unscheduled |
| `confidence_threshold`/`nms_iou_threshold` shipped in §4.2 are placeholder values not actually validated by §2's real measurement | **High if it happens** — would mean Phase 4 ships a configuration nobody measured | §4.2's own explicit caveat: the measurement must run against the values Phase 4 intends to ship, not a value swapped in afterward |
| N10 (unpinned `onnxruntime` version, `docs/SCRFD_PHASE2_DESIGN.md` §10 item 7) remains open | Low today, real once Phase 4 ships a real `.f.`-bearing default | Not this phase's scope to close, but should be resolved before or alongside Phase 4 given it now feeds a shipped identity, not merely a test-time one |

---

## 11. Rollback strategy

Unchanged in kind from every prior SCRFD document's analysis, re-confirmed
against this document's actual scope:

1. **Before this phase (anywhere through Phase 2/3):** nothing to roll
   back — no production identity depends on SCRFD yet.
2. **At this phase's cutover:** because `HaarDetector` and a Haar-backed
   `default_face_selection()` are the **only** implementation that existed
   until this phase, reverting `default_face_selection()`'s return value
   (§4.2) to its Phase-1/2 literal is a **one-line, single-commit revert**
   that restores the exact prior `.f.` value and `STUB_ENGINE_VERSION`. No
   stored-artifact migration exists to undo in either direction (§9).
3. **`verify_scrfd_asset_at_startup` (§4.7) degrades gracefully on
   rollback** — its own first line (`if not policy.detector.startswith
   ("scrfd-"): return`) means a reverted, Haar-shaped default makes the
   check a no-op automatically, not a second thing to revert.
4. **§4.6's synthetic-fixture bootstrap is harmless to leave in place after
   a rollback** — it only sets an env var that a Haar-shaped default never
   reads, so there is no reason to revert `tests/run_all.py`'s change even
   if `default_face_selection()` itself is rolled back.
5. **Abort condition, restated because it applies unchanged:** if §2's real
   measurement does not clear R13's acceptance criterion, this entire
   phase does not proceed — not "ships with a caveat," not "ships pending
   re-measurement." A failed or (as today) **absent** measurement is
   identical in effect: no cutover.

---

## 12. Effort estimate

| Work item | Estimate | Basis |
|---|---|---|
| §2's prerequisites (asset fetch, license acceptance, photo corpus, real measurement) | **Unbounded from this document — licensing/acquisition lead time, not engineering time** | Unchanged from every prior document's own honest framing; this document adds no new information that shortens it |
| §4.2 `default_face_selection()` cutover + `_scrfd_model_path()` | 0.5 day | Small, mechanical, once §2 names the real threshold values to ship |
| §4.6 synthetic-fixture bootstrap for tests | 1–2 days | The fixture-design sub-question (does it meaningfully exercise what re-derived tests need) is the real unknown, not the plumbing |
| §4.7 startup fail-closed check | 0.5 day | Direct copy of the OR1/OR1-b pattern, already proven in this codebase |
| §4.8 landmarks (optional) | 1 day | Decode logic mirrors `distance2bbox`'s existing shape; `face_track()` parameter threading is mechanical |
| §7.3's widened test re-derivation (10 files) | **3–6 days** | Each file needs its Haar-specific assertions re-derived against actual SCRFD/synthetic-fixture behavior, not merely re-run — consistent with `docs/SCRFD_PROGRAM_DESIGN.md` §7's own estimate for the original 8-file list, scaled up for the 2 newly-found files |
| §7.4 new test files | 1 day | Structurally similar to existing suites |
| Full-suite re-run + collateral literal sweep | 0.5–1 day | Mechanical once content re-derivation (above) is done |
| **Total, engineering only, once §2's gate clears** | **~7–11 days** | Excludes §2's unbounded licensing/acquisition lead time, which this document cannot estimate any better than `docs/SCRFD_PHASE2_DESIGN.md` §8 already could not |

---

## 13. Stop/go criteria

**Go**, only if **all** of the following hold, each independently
checkable — restated as a checklist so no future reader has to re-derive
which items are actually blocking:

1. `docs/SCRFD_ASSET.md`'s license-acceptance step (§2 item 1) has
   happened, recorded, for the exact environment(s) the asset is fetched
   into.
2. `docs/SCRFD_ASSET.md`'s SHA-256 fields (§2 item 2) are filled in — the
   asset has actually been fetched and verified.
3. A real, licensed/provenance-recorded photo corpus (§2 item 3) exists.
4. `python tests/scrfd_detector_study.py` has been run for real, at least
   twice with different seeds (§2 item 4), and **both runs clear R13's
   acceptance criterion**, amplification gated on the bootstrap CI's upper
   bound (§2 item 5).
5. §4.6's synthetic-fixture mitigation is designed in enough detail to know
   it will not silently make re-derived tests vacuous (the risk register's
   "Medium" item) — this document flags the question but does not resolve
   it; resolving it is part of implementation-time work, not a precondition
   to *starting* implementation the way items 1–4 are.

**No-go / abort — unchanged from every prior SCRFD document, restated
because it applies without modification:** if item 4 fails, this phase does
not proceed. **Today, items 1–4 are all unmet** (§1.9, §2) — not failed,
**unattempted**. This document's own gate (§14) reflects that state
honestly rather than treating "the tool is ready" as equivalent to "the gate
is cleared."

---

## 14. Gate

**Design complete for the engineering scope (§4–§7). NOT approved for
implementation — three prerequisites named in §2 and restated in §13 are
still missing, and none of them can be closed by more design work or more
code:**

1. **No SCRFD asset has been fetched.** Phase 0's licensing-acceptance step
   (`docs/SCRFD_PHASE2_SPEC_RESOLUTION.md` §12) has not happened.
2. **No real photo corpus exists** for the R13 measurement (`docs/
   SCRFD_PHASE3_MEASUREMENT_PLAN.md` §3.3).
3. **R13 has therefore never been measured for real.** The methodology is
   correct and tested (this session's own prior work); the *evidence* it
   would produce does not exist. `REPRODUCIBILITY_AUDIT.md`'s R13 row is,
   correctly, still `Open`.

This document's own architecture and file-list work (§4–§9) is not
contingent on those three items — it can be reviewed, revised, or approved
in principle independently of when Phase 0's fetch happens. **But no
production code implementing this design should be written until §13's
checklist items 1–4 are independently confirmed**, per this project's own
standing rule (`docs/SCRFD_PROGRAM_DESIGN.md` §9.3, restated by every
subsequent SCRFD document without exception): a missing measurement is not
license to proceed, and this project does not cut over a shipped default on
structural argument alone once a measurement is the specified gate.

**No code was written to produce this document. Stop here.**
