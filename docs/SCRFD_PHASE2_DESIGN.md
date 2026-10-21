# SCRFD Program — Phase 2: SCRFD Implementation, Wired But Not Defaulted

**Date:** 2026-09-18. **Type:** design only — no code. **Basis:** a fresh
source re-derivation this session, done *after* confirming Batch 28 / Phase 1
is implemented and green (`git status`, `inference/preprocess.py`,
`tests/test_face_detector_abstraction.py`, `tests/run_all.py` all read live,
not assumed from `docs/SCRFD_PROGRAM_DESIGN.md`'s own Phase 2 section, which
is five paragraphs and not implementation-grade — the same gap
`docs/SCRFD_PHASE1_ABSTRACTION_VERIFICATION.md` found and closed for Phase 1).
Three of that document's Phase 2 claims are **confirmed**, two are
**sharpened** (directionally right, not precise enough to implement from),
and **three new constraints** it did not mention are found below. Where this
document narrows an option `docs/SCRFD_PROGRAM_DESIGN.md` §3.3 explicitly
left open, that narrowing is justified from source, not asserted.

**No code was written to produce this document, per instruction.**

---

## 0. Confirmed starting state (re-derived, not assumed)

- `python tests/run_all.py` baseline, live: **55 suites, 1335 tests** —
  `tests/run_all.py:130` registers `("S8-B28 face detector abstraction
  (SCRFD Phase 1)", test_face_detector_abstraction)`, and
  `tests/test_face_detector_abstraction.py` has exactly **8** `test_*`
  functions (`grep -c "^def test_"`), which reconciles the prior confirmed
  figure — `docs/PROJECT_STATUS_FINAL.md` §8's **54 suites, 1327 tests**,
  written *before* Batch 28 — as 54+1 suites, 1327+8 tests. No drift; that
  document is simply one batch stale, and is treated as such rather than
  as ground truth.
- `inference/preprocess.py` has, live: `FaceDetector` (`@runtime_checkable
  Protocol`, `:879-897`), `HaarDetector` (`:900-928`), a module-level shared
  `_haar_detector = HaarDetector()` (`:931`), and `detect_faces` (`:938-970`)
  hard-wired to it — `policy = policy or default_face_selection(); boxes =
  _haar_detector.detect(image, policy=policy); return sorted(boxes,
  key=face_order_key)`. No `SCRFDDetector`, no branch on `policy.detector`
  anywhere in the file (confirmed by reading `:938-970` in full — the
  docstring says so explicitly at `:956-966`).
- `git status`: working tree has the Phase 1 changes plus several
  design-only docs, none touching production identity (`REPRODUCIBILITY_
  AUDIT.md`, `backend/config.py`, `contracts/common.py`, `contracts/
  evidence.py`, `inference/decode_env.py`, `inference/preprocess.py`,
  various tests — all consistent with the R16/R23/R26/R30/R31/OR1-b work
  the untracked design docs describe, none of it SCRFD). No concurrent-
  session drift on any file this program touches.
- `STUB_ENGINE_VERSION` (`backend/config.py:41`), live:
  `"stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.c9d4025c"`. This is
  this document's own pinned baseline — Phase 2's version-impact claims
  (§4) are checked against this exact string, not a remembered one.

---

## 1. Re-derived architecture

### 1.1 The abstraction Phase 2 builds on

`FaceDetector` (`inference/preprocess.py:879-897`) is the structural twin of
`ModelBackend` (`inference/models.py:380-393`):

```python
@runtime_checkable
class FaceDetector(Protocol):
    def detect(self, image: np.ndarray, *, policy: "FaceSelectionPolicy") -> list[FaceBox]: ...
```

`HaarDetector.detect()` (`:918-928`) reads exactly three `policy` fields
(`scale_factor`, `min_neighbors`, `detect_min_px`) and returns *unordered*
boxes — `face_order_key` sorting happens once, outside every detector, at
`detect_faces`'s dispatch level (`:970`). This is the load-bearing structural
fact for Phase 2: **a detector returns raw geometry only; ordering, gating
and cropping are detector-agnostic and already correctly factored out**
(confirmed by re-reading `face_order_key`, `:831-876`, which takes a
`FaceBox` and nothing else — no detector-specific state reaches it).

### 1.2 `FaceSelectionPolicy.detector` / `.detector_asset` — confirmed pure labels, one construction site

`detect_faces` never inspects `policy.detector` or `policy.detector_asset`
in control flow (re-confirmed, `:938-970`) — they feed
`FaceSelectionPolicy.identity` only (`:542-555`), exactly as
`docs/SCRFD_PHASE1_ABSTRACTION_VERIFICATION.md` §1.5 found for Phase 1.

**New constraint Phase 2 must reckon with, not mentioned in
`docs/SCRFD_PROGRAM_DESIGN.md` §3.3:** `FaceSelectionPolicy` has **exactly
one constructor call site in the entire repository** —
`grep -rn "FaceSelectionPolicy("` finds only `inference/preprocess.py`
itself (`default_face_selection()`, `:611-622`). Every test and study
constructs a policy via `dataclasses.replace(default_face_selection(), ...)`
or `default_face_selection()` directly, never a bare literal. This matters
because two hard tests gate any field addition:

- `tests/test_face_selection_identity.py:283-314`,
  `test_every_policy_field_moves_the_identity` — asserts
  `{f.name for f in fields(FaceSelectionPolicy)} == set(alternatives)`
  (a **hard equality**, not subset) and that varying every field moves
  `.identity`. Adding a field without updating this test's `alternatives`
  dict fails immediately and loudly — which is the point (it is designed to
  catch exactly this), but it means **this test file is unconditionally in
  Phase 2's file list** the moment `FaceSelectionPolicy` gains a field.
- `tests/test_face_selection_identity.py:317-330`,
  `test_the_policy_has_no_defaults` — asserts every dataclass field has
  `default is MISSING and default_factory is MISSING`. A new field cannot
  have a Python-level default.

**Sharpened conclusion, correcting `docs/SCRFD_PROGRAM_DESIGN.md` §3.3's
framing:** that document called option (a) ("optional-with-a-sentinel
fields") cheap *because* it keeps `identity`/`to_dict`/`differences`
unchanged in shape. That is true, but it understated the real cost, which
is not "ripples through every call site" (§3.3's fear for option (b)) —
because there is only **one** call site — it is instead: *every field,
including a Haar-inert SCRFD one, must have an explicit, no-default value at
that one call site, and `test_every_policy_field_moves_the_identity`'s
`alternatives` dict must gain an entry for it.* That is a **two-file, not a
wide, change** (`inference/preprocess.py` + `tests/
test_face_selection_identity.py`), which makes option (a) considerably
cheaper than either the design document or a first read of the "no
defaults" rule suggests. This is the single most consequential
re-derivation finding of this document.

### 1.3 New constraint: the sentinel value's *type* is not free — a NaN sentinel is a latent equality footgun

`contracts/version.py:227-243`, `_canonical` — floats go through `repr()`
before hashing, specifically so two float-different values fingerprint
differently. `repr(float("nan"))` is `"nan"`, a deterministic string, so a
NaN sentinel for an inert `confidence_threshold`/`nms_iou_threshold` field
**would** fingerprint identically across calls — the identity machinery is
fine with it.

**But `FaceSelectionPolicy` is `@dataclass(frozen=True)` with no custom
`__eq__`** (`:432`), so its auto-generated equality compares fields with
`==`, and `float("nan") == float("nan")` is `False` in Python. Two
independently-constructed Haar policies (e.g. from two calls to
`default_face_selection()`, or a policy round-tripped through
`dataclasses.replace`) would compare unequal on that field alone —
`default_face_selection` is `@lru_cache(maxsize=1)` (`:594`) so this is not
reachable *today* through that one function, but the risk resurfaces the
moment any code path other than the cached factory constructs an
equivalent Haar policy, or a future test asserts policy equality directly
rather than `.identity` equality. **Nothing in the current suite compares
`FaceSelectionPolicy` instances with `==`** (grep-verified — every
comparison found is `parts.face_selection == "<hash>"` or `.identity ==`),
so this is latent, not live — but it is exactly the kind of "measured, not
assumed" trap this project's own history (R14, R17, R19) warns against
introducing silently.

**Recommendation:** use `None` as the Haar-inert sentinel, not `NaN`.
`_canonical(None)` (`:241`) returns `None` (→ JSON `null`), which is already
handled and equality-safe (`None == None` is `True`). Type the new fields
`float | None`.

---

## 2. Minimum viable Phase 2 vertical slice

### 2.1 `FaceSelectionPolicy` extension — resolving §3.3

Per §1.2/§1.3: **adopt option (a)**, concretely:

```python
confidence_threshold: float | None
"""SCRFD's own detection-score floor. None under Haar, which has no
equivalent knob — detectMultiScale's minNeighbors already lives in
min_neighbors."""

nms_iou_threshold: float | None
"""SCRFD's NMS IoU floor. None under Haar — detectMultiScale performs
its own internal grouping with no separate IoU parameter to expose."""
```

`default_face_selection()` (`:611-622`) gains two lines:
`confidence_threshold=None, nms_iou_threshold=None`. `identity`, `to_dict`,
`differences` each gain the two keys (mechanical, same pattern every
existing field already uses). `tests/test_face_selection_identity.py`'s
`alternatives` dict (`:290-302`) gains two entries with **non-`None`**
alternative values (e.g. `"confidence_threshold": 0.5,
"nms_iou_threshold": 0.4`) so the "every field moves the identity" property
is actually exercised for the new fields, not vacuously true because `None`
was compared to `None`.

**Version impact of this step alone: zero.** `default_face_selection()`'s
two new fields are both `None`, which — per `_canonical` — serialises
identically every time; `.identity` therefore reads a fingerprint over a
superset of fields whose new members are constant `None` across before and
after... **except this is false and must be stated honestly**: adding *any*
key to the dict `identity` fingerprints (`:542-555`) changes what
`json.dumps` serialises, which changes the SHA-256 input, which **changes
`.identity`'s value** even though every new field is `None`. `fingerprint()`
has no notion of "this key is new but inert" — it hashes whatever mapping
it is given. **This means Phase 2, if it extends `FaceSelectionPolicy` at
all, cannot avoid moving `.f.`, contradicting `docs/SCRFD_PROGRAM_DESIGN.md`
§5 Phase 2's claim of "none" `.f.` impact.** This is the most important
correction this document makes to the program design, detailed next.

### 2.2 The program design's Phase 2 version-impact claim does not survive contact with `fingerprint()` — resolved by sequencing, not by avoidance

`docs/SCRFD_PROGRAM_DESIGN.md` §5 Phase 2 says "`.f.` impact: none in the
shipped configuration," reasoning that `default_face_selection()` is
unchanged. That reasoning is right about *values* and wrong about *shape*:
`FaceSelectionPolicy.identity` is a fingerprint over a **dict literal
written in source** (`:542-555`), not over "whatever fields happen to
differ from before." Adding `"confidence_threshold": None,
"nms_iou_threshold": None` to that dict changes the dict, which changes
every fingerprint computed from it — including the shipped Haar policy's.

Two ways to resolve this, stated as a decision rather than pre-empted:

1. **(Recommended.) Accept the one-line `.f.` move as part of Phase 2,
   scoped and reasoned exactly like every prior label-closure batch (B18's
   `face_px` prose, B22's `interpolation_rule`/`crop_clip_rule`).** This is
   a **structural** move (two new, universally-`None`-valued keys), not a
   **behavioural** one — no pixel this platform produces changes, which is
   the same distinction `docs/CROP_IDENTITY.md` already uses to justify a
   `.f.` move without a benchmark boundary. One cache generation, both
   media namespaces (the N8-accepted cost, paid four times already:
   B20/B22/B23/B26). Collateral pins: `STUB_ENGINE_VERSION` literal
   (`backend/config.py:41`), and every hardcoded `.f.`/`STUB_ENGINE_VERSION`
   string across the test suite — the same bookkeeping every prior
   `.f.`-moving batch has done, template `tests/test_crop_identity.py`'s own
   batch-22 diff.
2. **(Rejected.) Compute `SCRFDDetector`'s test policy from a *second*,
   parallel dataclass or a dict that bypasses `FaceSelectionPolicy.identity`
   entirely**, so nothing shipped-adjacent changes shape. Rejected because
   it recreates exactly the "two literals" architecture
   `docs/BATCH_23_R3_DESIGN.md`/B23 already rejected for video selection —
   a second identity-bearing structure outside the one the platform
   composes `STUB_ENGINE_VERSION` from is a standing hazard, not a
   convenience, and it defeats §2.1's whole point (one dataclass represents
   both detectors).

**This document recommends option 1 and revises the version-impact table
in §4 accordingly** — `docs/SCRFD_PROGRAM_DESIGN.md`'s Phase 2 row is wrong
on this one point; everywhere else it independently checks out.

**Honest mitigation, stated plainly:** because the move is `None`-valued
and mechanical, it is safe to fold into the *same* cache generation Phase
4's real cutover already pays — i.e., a project could choose to defer §2.1
literally until Phase 4, keeping Phase 2 truly zero-impact by testing
`SCRFDDetector` against a **locally-constructed** policy object that is
never round-tripped through `default_face_selection()` or asserted against
`STUB_ENGINE_VERSION`. That is this document's **actual recommendation**
for the vertical slice (§2.6): extend the dataclass in Phase 2 (needed so
`SCRFDDetector.detect()` has real fields to read), pay the one `.f.` move
in Phase 2 rather than deferring it, specifically *because* deferring it
would mean Phase 4 has to both extend the dataclass **and** cut over the
default in the same identity move, which is exactly the kind of two-things-
at-once change this project's "one identity move at a time" discipline
(R17, R25, R29) exists to avoid.

### 2.3 `SCRFDDetector` implementation

```python
class SCRFDDetector:
    """ONNX-backed SCRFD. Lazy-imports onnxruntime — see §2.5."""

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._session: "onnxruntime.InferenceSession | None" = None  # noqa: F821

    def _session_for_thread(self):
        # Mirrors _cascade_local (R15): one InferenceSession per thread.
        # onnxruntime's own docs mark a session's Run() thread-affinity as
        # undefined across concurrent callers of one instance; this project
        # already paid for exactly this class of bug once (R15) and must not
        # re-derive it empirically for a second detector.
        ...

    def detect(
        self, image: np.ndarray, *, policy: FaceSelectionPolicy
    ) -> list[FaceBox]:
        import onnxruntime  # lazy — same pattern as models.py:585
        session = self._session_for_thread()
        # preprocess -> forward pass -> decode anchors -> NMS
        # policy.confidence_threshold / policy.nms_iou_threshold gate here
        ...
```

**Thread-safety is not optional scope creep — it is a direct consequence of
R15's precedent.** `_cascade_local` (`inference/preprocess.py:327-359`) exists
because a shared `cv2.CascadeClassifier` measurably returned another
thread's detection ~2.8–15% of the time under concurrency
(`_cascade_local`'s own docstring cites the exact figures). `onnxruntime.
InferenceSession` is a comparable piece of mutable native state reached
through `asyncio.to_thread` by the same two pipelines. **Whether it is safe
to share is a claim to verify, not assume** — `onnxruntime`'s own
documentation states `Run()` calls on one session are safe to make
concurrently from multiple threads (session state for a single inference is
local to the call), which is a *different* guarantee than Haar's
`CascadeClassifier` had, but this document does not take that claim on
faith: **a `test_the_onnx_session_is_not_shared_unsafely`-style measurement
(the same shape as `tests/inference_determinism_study.py`'s original R15
study) is mandatory before Phase 2 is considered done**, not merely
before Phase 4. This is a new constraint neither `docs/
SCRFD_PROGRAM_DESIGN.md` nor a naive reading of "wired but not defaulted"
would surface, because "not defaulted" sounds like "not yet a concurrency
concern" — it is not: Phase 2's own unit tests exercise `SCRFDDetector`
under `asyncio.to_thread` the moment they reuse `_detect_and_measure`'s
call shape, and Phase 3's benchmark plumbing run does so for real.

### 2.4 Asset acquisition strategy

`.gitignore:19` already contains `*.onnx` — **the project has already
decided, before this program, that ONNX assets are not vendored into the
repository**, the same convention the Haar cascade breaks only because it
ships *inside* the `opencv-python` wheel (`cascade_digest()`'s own
docstring, `:390-397`, makes this exact distinction). This resolves
`docs/SCRFD_PROGRAM_DESIGN.md` §3.1's "committed binary vs. fetched"
question in favour of **fetched, not committed** — not a fresh decision,
a pre-existing one this document is the first to notice.

Recommended shape: a documented download step (URL + expected SHA-256,
recorded in a new `docs/SCRFD_ASSET.md` or this document's own §7) that
places the `.onnx` file at a path `SCRFDDetector` reads, analogous to
`CASCADE_PATH` (`inference/preprocess.py`, resolved via `cv2.data.
haarcascades`). **`detector_asset` is computed the same way
`cascade_digest()` computes today's value** — `sha256(file bytes)[:16]` —
via a new `scrfd_asset_digest(path) -> str` free function, `@lru_cache`'d
identically. Model-size choice (InsightFace's "2.5G" ≈ low-single-digit MB,
"10G" ≈ mid-teens MB — the suffix names GFLOPs, not file size, a distinction
`docs/SCRFD_PROGRAM_DESIGN.md` §3.1's "500KB" figure understates and this
document corrects) is a latency/accuracy tradeoff for Phase 3's measurement
to inform, not to pre-decide.

**Licensing — flagged as an unresolved, must-verify-before-Phase-0 blocker,
not asserted.** This session attempted to verify SCRFD's current pretrained-
weight license via web search and the search tool returned only backend
errors — **this fact could not be confirmed live in this session**, and is
reported as a gap rather than papered over with a remembered answer.
InsightFace's own repository has historically drawn a line between
permissively-licensed *code* and separately-licensed *pretrained weights*
(some packs marked non-commercial-research-only); which license attaches
to whichever specific SCRFD checkpoint Phase 0 selects **must be read from
that checkpoint's own release page at implementation time**, not inherited
from this document or from general knowledge about the project. This is a
harder blocker than a dependency version pin: an unpinned dependency
version is a reproducibility risk; an unverified model license is a
distribution-legality risk, and this program should not proceed past
Phase 0 without a human explicitly reading and recording the license text
for the exact file being fetched.

### 2.5 ONNX runtime dependency strategy — cheaper than the program design assumed

**New finding, correcting `docs/SCRFD_PROGRAM_DESIGN.md` §3.1's framing of
"a new external dependency":** `onnxruntime` is **already** a declared
optional dependency — `pyproject.toml:42`, `onnx = ["onnxruntime>=1.18"]`
— added for `inference/models.py`'s `_build_onnx_backend` (`:579-591`),
which lazily imports it inside the method body exactly as `_build_
torch_backend` does for `torch`. `inference/protocol.py`'s own package
docstring (`:4-5`) already names `onnxruntime` as one of the package's
sanctioned dependencies, alongside `numpy`, `torch`, `cv2`.

**`tests/test_dependencies.py` already generically covers a second
`onnxruntime` import site, with zero test-file changes needed:**
`test_the_model_backends_are_optional_not_runtime` (`:154-164`) and
`test_the_optional_imports_are_genuinely_lazy` (`:167-174`) both iterate
`MATRIX.optional["onnxruntime"]`, and `MATRIX` (`build_matrix()`,
`tests/import_audit.py`) is an AST scan of the whole `RUNTIME_PACKAGES`
tree (`inference` is one of the seven, `:56`), not a hardcoded reference to
`models.py`. **`SCRFDDetector.detect()`'s lazy `import onnxruntime`
(§2.3) is automatically picked up and automatically required to be lazy by
this existing suite the moment it exists — no new dependency-audit test is
needed**, provided the import is written inside a method body, never at
`inference/preprocess.py` module scope. Getting this wrong (a module-scope
`import onnxruntime` at the top of `preprocess.py`) fails
`test_the_optional_imports_are_genuinely_lazy` immediately and loudly — the
regression net already exists.

**Consequence: `pyproject.toml` needs no edit for Phase 2.** No new
`[project.optional-dependencies]` entry, no `uv.lock` regeneration, no
`[tool.uv].override-dependencies` addition — `onnxruntime`'s own version is
not currently pinned there (only `opencv-python`/`numpy` are, per N5's
scope: the two packages that feed a `fingerprint()` call), and Phase 2 does
not change that, because nothing Phase 2 ships reads `onnxruntime.
__version__` in any identity. Whether it *should* be pinned once SCRFD
starts feeding real reproducibility measurements is a genuine open question
this document surfaces rather than resolves — see §7's candidate finding.

### 2.6 Detector dispatch design

`detect_faces`'s own current docstring (`:956-966`) states the condition
precisely: *"Dispatch-by-`policy.detector` arrives with the SCRFD
implementation, which is what will need a second implementation to dispatch
to."* Phase 2 is that arrival — `SCRFDDetector` is the second
implementation Phase 1 was written in anticipation of. Recommended shape,
a small registry mirroring `ModelRegistry._build_backend`'s
if/elif-by-string dispatch (`inference/models.py:545-552`), not a dict (the
project's own precedent for this exact shape uses `if`/`raise`, not a
dict-of-constructors, so an unknown value gets a named, catchable error
rather than a `KeyError`):

```python
def detect_faces(image, *, policy=None):
    policy = policy or default_face_selection()
    if policy.detector == "opencv-haar-frontalface-default":
        detector = _haar_detector
    elif policy.detector == "scrfd-<variant>":  # exact string TBD at Phase 0/2
        detector = _scrfd_detector_for(policy.detector_asset)
    else:
        raise UnreadableMedia(f"Unknown face detector {policy.detector!r}.")
    boxes = detector.detect(image, policy=policy)
    return sorted(boxes, key=face_order_key)
```

**This changes `detect_faces`'s control flow but not its identity
inputs or its behaviour for the shipped policy** — `default_face_selection()`
still returns `detector="opencv-haar-frontalface-default"`, so the `if`
branch taken in production is unchanged, byte-for-byte, from Phase 1.
`test_detect_faces_stays_monkeypatchable_at_module_level`
(`tests/test_face_detector_abstraction.py:171-194`) is unaffected by
construction — the monkey-patch pattern replaces the whole function, not a
branch inside it.

**One real risk this dispatch introduces, absent in Phase 1:** an
`_scrfd_detector_for(asset)` factory that constructs (and should cache,
`@lru_cache`-style, mirroring `default_face_selection()`'s own caching) a
`SCRFDDetector` bound to a specific asset path/digest. Getting the cache key
wrong (e.g. keying on `policy.detector` alone, ignoring `detector_asset`)
would let two different SCRFD checkpoints silently share one loaded
session — the exact shape of bug this project's whole `.f.`/`.d.`
discipline exists to prevent, one level below the identity string rather
than in it. Flagged as a required test (§2.8, test F).

### 2.7 Test strategy

New suite `tests/test_scrfd_detector.py`, structurally parallel to
`tests/test_face_detector_abstraction.py`:

| # | Test | Assertion |
|---|---|---|
| A | `test_scrfd_detector_satisfies_the_face_detector_protocol` | `isinstance(SCRFDDetector(path), FaceDetector)` |
| B | `test_scrfd_detector_finds_a_face_on_the_shipped_fixture` | Non-empty `list[FaceBox]` on the same synthetic face tile Phase 1's suite uses (`_face_tile()`), or a real photograph if the synthetic ellipse-fixture does not trigger a CNN detector (flagged as an open question — Haar and a real CNN detector do not necessarily agree on what counts as a face-like fixture; this may need a small **new**, real or more photographic test fixture, which Haar's suite never needed) |
| C | `test_scrfd_detector_respects_confidence_threshold` | Raising `policy.confidence_threshold` past the fixture's true score empties the result — the SCRFD analogue of Phase 1's test C (`scale_factor` at 1.05 losing the face) |
| D | `test_scrfd_boxes_are_free_aspect` | At least one detection has `width != height` on an asymmetric fixture — the structural property `docs/SCRFD_PROGRAM_DESIGN.md` §1/`docs/CROP_GEOMETRY.md` §8 both predict and neither can test without a real SCRFD instance |
| E | `test_the_onnx_session_is_not_shared_unsafely` | §2.3's mandatory concurrency check — N concurrent `detect()` calls across threads against known-different images do not cross-contaminate results (same shape as `tests/inference_determinism_study.py`'s original R15 stage 11) |
| F | `test_two_different_assets_are_never_conflated_by_the_dispatch_cache` | §2.6's risk — two `SCRFDDetector` instances built from different asset paths/digests are distinct objects, not silently shared |
| G | `test_scrfd_asset_digest_matches_a_known_file` | `scrfd_asset_digest()` (§2.4) is a pure content hash, pinned against a fixture-sized dummy `.onnx` file the same way `cascade_digest()`'s own tests pin the Haar XML |
| H | `test_default_face_selection_is_still_haar` | `default_face_selection().detector == "opencv-haar-frontalface-default"` — the one-line proof that Phase 2 did not silently become Phase 4 |

Plus, per this project's own convention (`crop_geometry_study.py`,
`decode_study.py`, `face_selection_study.py`): a new **`tests/
scrfd_detector_study.py`**, not part of `run_all.py`, holding Phase 3's R13
measurement (§2.8) as a re-runnable script rather than a one-off.

**Existing-suite impact of §2.2's `.f.` move:** every collateral literal
`docs/SCRFD_PROGRAM_DESIGN.md` §5 Phase 4 already enumerated
(`STUB_ENGINE_VERSION` and the hardcoded version strings across
`test_crop_identity.py`/`test_headline_and_face_semantics.py`/etc.) is
pulled **forward into Phase 2** by this document's §2.2 recommendation,
since the `.f.` move happens here now, not at Phase 4. This is a real,
material widening of Phase 2's blast radius relative to `docs/
SCRFD_PROGRAM_DESIGN.md`'s original "zero version impact" framing — stated
plainly rather than minimized, because it is the direct, unavoidable
consequence of §2.2's finding, not a choice this document is free to
undo.

### 2.8 Measurement strategy for R13

The acceptance criterion, quoted exactly
(`REPRODUCIBILITY_AUDIT.md:1516-1518`):

> Under ±4 LSB uniform noise on real photographs, across ≥ 200 images: no
> face lost, and the box short edge moving ≤ 1%. Amplification below 10×,
> from the measured 310×.

Reuse `tests/decode_study.py:129`'s `surrogate_logit` — "a smooth,
Lipschitz function of the pixels (gradient energy)" — the same apparatus
that produced the original 310× figure
(`tests/inference_determinism_study.py:648-650`). Procedure, run from
`tests/scrfd_detector_study.py`:

1. Load ≥ 200 real photographs (**a genuine, separate acquisition
   question from §2.4's model asset** — synthetic ellipse fixtures are
   known-adequate for Haar's coarse cascade but were never validated
   against what "real photographs" means for R13's own 310× measurement,
   which itself ran on real images per Batch 8/10's own methodology).
   Licensing/provenance of this corpus is a second, smaller version of
   §2.4's asset question and should be resolved the same way: a documented
   source, not an assumed one.
2. For each image: detect with `SCRFDDetector`, apply ±4 LSB uniform noise,
   re-detect, record box movement (short-edge %) and whether the face was
   lost.
3. Hold the *pre-noise* crop fixed, score with `surrogate_logit`; hold the
   crop *free to move* with the noise-perturbed box, score again; compute
   the spread exactly as the original 310× measurement did.
4. Report amplification, face-loss rate, box short-edge movement — against
   the criterion's three numbers, not a paraphrase of them.

**This validates detection stability, not detection quality** — restated
from `docs/SCRFD_PROGRAM_DESIGN.md` §7's own honest scope boundary, which
this document independently re-confirms rather than merely repeats:
`StubBackend` (`inference/models.py:396+`) is a hash-derived, deliberately
discontinuous function of input bytes, unsuitable for any question about
whether SCRFD finds faces Haar misses. `evaluation/benchmark.py`'s
`paired_bootstrap` (`:568`) / `regression_gate` (`:783`), which key on
`effective_model_version` (confirmed at `:816`), validate **plumbing** —
that an SCRFD-selection `BenchmarkRun` and a Haar-selection one separate
correctly in the cache and the harness runs end to end — not quality, for
the same reason.

---

## 3. Explicit file/dependency/risk inventory

### 3.1 New files

| File | Purpose |
|---|---|
| `tests/test_scrfd_detector.py` | §2.7's suite (A–H) |
| `tests/scrfd_detector_study.py` | §2.8's re-runnable R13 measurement, project convention |
| `docs/SCRFD_ASSET.md` (or a §7 addendum here) | Recorded download URL, SHA-256, and **the verified license text** for the exact checkpoint chosen — must exist before Phase 0 is considered closed |

### 3.2 Modified files

| File | Change |
|---|---|
| `inference/preprocess.py` | `SCRFDDetector` (§2.3), `scrfd_asset_digest()` (§2.4), `FaceSelectionPolicy` +2 fields (§2.1), `detect_faces` dispatch (§2.6) |
| `tests/test_face_selection_identity.py` | `alternatives` dict +2 entries (§2.1) — **mandatory**, the hard equality assert in `test_every_policy_field_moves_the_identity` fails otherwise |
| `backend/config.py` | `STUB_ENGINE_VERSION` literal regenerated (§2.2) |
| `tests/run_all.py` | +1 import, +1 `SUITES` row, alphabetically placed (Phase 1's own precedent, `:32`/`:130`) |
| Every file carrying a hardcoded `.f.`/`STUB_ENGINE_VERSION` string | Collateral pin update — enumerate at implementation time via the same `grep -rn <old-.f.-value>` sweep every prior `.f.`-moving batch has run |

**Not modified, confirmed by re-reading why:** `inference/image_pipeline.py`
(calls `detect_faces` by name, unaffected by an internal dispatch change,
§1.1); `inference/video_pipeline.py` (never calls `detect_faces`, §2.7 of
`docs/SCRFD_PROGRAM_DESIGN.md`, re-confirmed unaffected); `contracts/*`
(no wire-shape change — `FaceTrack.landmarks` already exists, unpopulated,
and Phase 2 populates nothing in production since nothing is shipped);
`evaluation/benchmark.py` (used, not changed, §2.8).

### 3.3 Dependency additions

**None**, per §2.5. `onnxruntime>=1.18` is already declared
(`pyproject.toml:42`, the `onnx` extra); no `pyproject.toml` edit,
no `uv.lock` regeneration, no new `[tool.uv].override-dependencies` entry.
The SCRFD `.onnx` asset is a fetched artefact, not a Python dependency, per
`.gitignore:19`'s pre-existing convention (§2.4).

### 3.4 Asset licensing/distribution implications

See §2.4. **Open, unverified, blocking** — this session's attempt to check
SCRFD's current pretrained-weight license via web search failed at the
tool layer (backend error, not a "no results" answer), so this document
states the requirement rather than a remembered license string. Recorded
as a Phase-0 gate: no asset is fetched into any environment (dev, CI, or
otherwise) until its specific license is read and recorded in
`docs/SCRFD_ASSET.md`.

### 3.5 Risks and rollback

| Risk | Severity | Mitigation |
|---|---|---|
| §2.2's `.f.` move ships without its full collateral-pin sweep (a literal missed) | High if it happens, mechanical to catch | Full-suite re-run is the net, exactly as every prior `.f.`-moving batch used it — a missed literal is a loud test failure, not a silent one |
| `SCRFDDetector`'s ONNX session shares unsafely across threads (R15 recurrence, §2.3) | High if it happened, structurally checked | Test E (§2.7) is mandatory, not deferred to Phase 3/4 |
| Asset license is wrong/unverified and gets fetched into a shared environment anyway | High, non-technical | §3.4's explicit gate — no fetch before a human reads and records the license |
| Dispatch cache conflates two SCRFD assets under one `policy.detector` string (§2.6) | Medium | Test F (§2.7) |
| The synthetic ellipse fixture that reliably triggers Haar does not trigger SCRFD, forcing a second fixture strategy mid-implementation | Medium, schedule risk not correctness risk | Named now (§2.7 test B) rather than discovered during implementation |
| `onnxruntime`'s own version is unpinned and a future minor release changes ONNX op numerics enough to move Phase 3's measurement | Low for Phase 2 (nothing shipped), real for Phase 3/4 reproducibility | Flagged as a candidate finding, §7 — not blocking Phase 2, which ships no production identity |

**Rollback:** identical in kind to Phase 1's (`docs/
SCRFD_PHASE1_ABSTRACTION_VERIFICATION.md` didn't need one; Phase 4's design,
`docs/SCRFD_PROGRAM_DESIGN.md` §9, already covers the general case).
Specific to Phase 2: reverting `detect_faces`'s dispatch to Phase 1's
hard-wired call, and reverting `FaceSelectionPolicy`'s two new fields
(and therefore `.f.`), is a **single commit revert** — nothing downstream
of `.f.` (no `CalibrationWitness`, no `BenchmarkRun`) is stored anywhere in
this repository outside test fixtures (re-confirmed this session, same
search `docs/SCRFD_PROGRAM_DESIGN.md` §2.6 performed), so there is no
migration to undo in either direction.

---

## 4. Version-impact axes, re-verified against Phase 2's actual scope

| Axis | File / value | Phase 2 impact |
|---|---|---|
| `CONTRACT_VERSION` | `contracts/enums.py:16`, `"1.0.0"` | **None.** No wire-shape change; `FaceTrack.landmarks` pre-exists and stays unpopulated (nothing calls `face_track()` with landmarks in Phase 2). |
| `RUN_SCHEMA_VERSION` | `evaluation/benchmark.py:108`, `"1.2.0"` | **None.** Nothing about `BenchmarkRun`'s shape changes; Phase 2 only *uses* the existing `paired_bootstrap`/`regression_gate` machinery (§2.8), it does not extend it. |
| `CALIBRATION_WITNESS_VERSION` | `evaluation/calibration_protocol.py:74`, `"1.3.0"` | **None, schema-wise.** Already generic over `pre_calibration_identity` (R31, `inference/models.py:312-339`), which reads `face_selection_fingerprint` automatically — §2.2's `.f.` move would change a *future* witness's digest **value**, but zero stored witnesses exist anywhere in this repository (re-confirmed this session, same search R31/B24/`docs/SCRFD_PROGRAM_DESIGN.md` §2.6 all independently performed) — nothing to migrate. |
| `LOCK_VERSION` | `evaluation/split_lock.py:81`, `"1.1.0"` | **None — and structurally cannot be, by construction.** This is the dataset split-lock *format* version (`SEALING_V1`/`SEALING_UNSEALED`, split assignment sealing), an entirely separate axis from model/detector identity by explicit design (`:82-91`'s own docstring: "independent of `CONTRACT_VERSION`... an offline artefact no serving component reads"). No code path connects `FaceSelectionPolicy`, `FaceDetector`, or anything in `inference/` to `evaluation/split_lock.py` — grep-confirmed zero references in either direction. Included in this table only because the task asked for it re-verified, and the honest answer is that it was never a candidate axis for this program. |
| `.d.` (decode identity) | `inference/decode_env.py`, feeds `backend/config.py:41`'s `.d.c9d4025c` segment | **None.** Detection is not a decode-environment fact (§2.3 of `docs/SCRFD_PROGRAM_DESIGN.md`, re-confirmed: nothing in `inference/decode_env.py` reads anything detector-related). |
| `.f.` (face/selection identity) | `FaceSelectionPolicy.identity`, feeds `backend/config.py:41`'s `.f.e629642c` segment via `ModelSpec.face_selection_fingerprint` | **Moves — corrected from `docs/SCRFD_PROGRAM_DESIGN.md`'s "none."** Per §2.2: extending the dataclass with two universally-`None` fields still changes the dict `fingerprint()` hashes, which changes the value, regardless of `default_face_selection()`'s Haar values being otherwise unchanged. One cache generation, both media namespaces (N8-precedent). `pre_calibration_identity` (R31) moves for free with it — zero stored witnesses to migrate (per the `CALIBRATION_WITNESS_VERSION` row above). |

**Net correction to `docs/SCRFD_PROGRAM_DESIGN.md` §10's summary table:**
its Phase 2 row should read `.f.` **moves** (a structural, `None`-valued
move), not "none" — every other cell in that row is confirmed correct.

---

## 5. Can Phase 2 be completed without moving any identity/version constant? Answered from source: **no, not if `FaceSelectionPolicy` is extended — and it must be, for `SCRFDDetector` to have real fields to read.**

This is a direct, provable answer, not a restatement of the question:

1. `SCRFDDetector.detect()` needs, at minimum, a confidence floor and an
   NMS IoU floor to be configurable and fingerprinted — the same reasoning
   R17 already established for Haar's `scale_factor`/`min_neighbors`
   (`inference/preprocess.py:436-446`'s own docstring: a verdict-affecting
   knob that is not in the identity is precisely the defect this project's
   whole `FaceSelectionPolicy` architecture exists to prevent). Building
   `SCRFDDetector` with hard-coded thresholds instead of policy fields
   would **reintroduce R17 for a second detector**, which no version of
   this program's design (this document, `docs/SCRFD_PROGRAM_DESIGN.md`,
   or `docs/SCRFD_PHASE1_ABSTRACTION_VERIFICATION.md`) considers acceptable.
2. `FaceSelectionPolicy.identity` is a `fingerprint()` over a source-literal
   dict (`:542-555`) — proven in §2.1/§2.2 from `_canonical`'s and
   `fingerprint()`'s actual code that adding a key to that dict changes the
   hash for *every* input, including the unchanged Haar one.
3. Therefore: **the only way to avoid moving `.f.` in Phase 2 is to not
   extend `FaceSelectionPolicy` at all** — which means `SCRFDDetector`
   cannot be configured or fingerprinted through the policy object, which
   violates (1) and this project's own R17 precedent.

**Conclusion: Phase 2 cannot both (a) give `SCRFDDetector` real,
fingerprinted configuration and (b) leave every version constant
unmoved.** It *can* leave `CONTRACT_VERSION`, `RUN_SCHEMA_VERSION`,
`CALIBRATION_WITNESS_VERSION`, `LOCK_VERSION` and `.d.` unmoved — those five
are independently proven untouched in §4, each from a specific structural
fact (no wire-shape change, no schema extension, no stored artefact, no
detector-decode coupling, no split-lock coupling respectively) — but `.f.`
moving is not a Phase 2 implementation mistake to avoid; it is the
mechanical, unavoidable consequence of giving a second detector a real,
policy-carried configuration, exactly as Batch 12 (R17) made `.f.` move
once already for the same reason on the Haar side.

---

## 6. Architecture diagram

```
                         FaceSelectionPolicy (extended, §2.1)
                         ├─ detector: str                    ─┐
                         ├─ detector_asset: str                │  feeds
                         ├─ scale_factor, min_neighbors,        │  .identity
                         │  detect_min_px          (Haar-only)  │  (.f. segment,
                         ├─ confidence_threshold,               │   §2.2/§4)
                         │  nms_iou_threshold  (SCRFD-only,      │
                         │  None under Haar, §2.1/§1.3)         │
                         ├─ ordering, crop_margin, crop_shape,  │
                         │  interpolation_rule, crop_clip_rule ─┘
                         │  (detector-agnostic, unchanged)
                                 │
                                 │ passed to
                                 ▼
                    detect_faces(image, *, policy)   (§2.6, dispatcher)
                                 │
              ┌──────────────────┴───────────────────┐
              │ policy.detector ==                    │ policy.detector ==
              │ "opencv-haar-frontalface-default"      │ "scrfd-<variant>"
              ▼                                        ▼
        _haar_detector                       _scrfd_detector_for(asset)   (§2.6,
        (HaarDetector, Phase 1,                (SCRFDDetector, §2.3 —      cached by
         unchanged)                            lazy onnxruntime import,   asset digest,
              │                                 per-thread session, §2.3)  §2.6 risk F)
              │  list[FaceBox]                         │  list[FaceBox]
              │  (unordered)                           │  (unordered, free-aspect)
              └──────────────────┬───────────────────┘
                                 ▼
                    sorted(boxes, key=face_order_key)   (detector-agnostic
                                 │                        total order, §1.1 —
                                 │                        unchanged by this phase)
                                 ▼
                    (existing) crop_face / resize / model / postprocess
                    — completely unaffected by this phase, per §3.2's
                      "not modified" list; SCRFD's boxes only reach these
                      functions in Phase 2's own isolated tests, never in
                      a shipped pipeline (default_face_selection() stays
                      Haar throughout Phase 2, test H, §2.7)

     Phase 2 file boundary — nothing production-facing outside
     inference/preprocess.py + the one collateral literal in
     backend/config.py (§2.2/§3.2) is touched.
```

---

## 7. Fail-before / pass-after matrix

Same framing as Phase 1's own plan
(`docs/SCRFD_PHASE1_ABSTRACTION_VERIFICATION.md` §5): this is not a bug fix,
so there is no traditional fail-before/pass-after *bug* pair. The
applicable shape is "prove the addition is exactly what it claims to be —
a second, real, tested detector, wired but never selected in production."

| Stage | Before Phase 2 | After Phase 2 |
|---|---|---|
| `python tests/run_all.py` | 55 suites, 1335 tests (§0's confirmed baseline) | 56 suites (+`test_scrfd_detector.py`), 1335 + 8-ish new tests, **0 failures** |
| `isinstance(SCRFDDetector(path), FaceDetector)` | `SCRFDDetector` does not exist | `True` (test A) |
| `default_face_selection().detector` | `"opencv-haar-frontalface-default"` | **Unchanged** — `"opencv-haar-frontalface-default"` (test H) — this is the single most important pass-after assertion in the whole phase |
| `default_face_selection().identity` | `"251d22e9"` (`tests/test_face_detector_abstraction.py:52`'s pinned baseline) | **A new value** — per §4/§5, this *should* move (two new `None`-valued keys), and the new literal must be captured and pinned the same way `_BASELINE_FACE_SELECTION_IDENTITY` was for Phase 1 |
| `STUB_ENGINE_VERSION` | `"...+c.739c3fec.g.ae232cba.f.e629642c.d.c9d4025c"` | New `.f.` segment, `.d.` unchanged (`c9d4025c` persists) |
| A `SCRFDDetector` instance run against a real photograph, ±4 LSB noise | No R13 measurement possible (no second detector exists) | A concrete amplification figure, face-loss rate, box-movement percentage — the go/no-go input to Phase 4 (§2.8) |
| `SCRFDDetector.detect()` called concurrently from N threads on different images | N/A | Zero cross-contamination (test E) — the R15 precedent, checked rather than assumed |

---

## 8. Effort estimate

| Work item | Estimate | Basis |
|---|---|---|
| §2.1 dataclass extension + collateral test-literal updates | 0.5–1 day | Two fields, one constructor site, one test file's `alternatives` dict — genuinely small once §1.2's "one call site" fact is known |
| §2.3 `SCRFDDetector` + thread-safety verification | 2–4 days | The ONNX pre/post-processing (anchor decode, NMS) is the real unknown; the R15-style concurrency check is a half-day addition once the detector itself works |
| §2.4 asset acquisition + license verification | 0.5 day engineering, **unbounded until a human confirms the license** | The download/hash/digest mechanics mirror `cascade_digest()` closely; the license check is a blocking dependency on a fact this document could not verify this session |
| §2.6 dispatch + cache-safety test | 0.5 day | Small, mechanical, parallel to `ModelRegistry._build_backend` |
| §2.7 test suite (A–H) | 1–2 days | Test B's fixture question (§2.7) is the main open variable — may need a small photographic (not purely synthetic) fixture, unprecedented in this codebase's face-detection tests |
| §2.8 R13 measurement + real-photo corpus acquisition | 1–2 days engineering + corpus-acquisition lead time (untracked, depends on source) | `surrogate_logit` machinery already exists; the ≥200-real-photograph requirement is new acquisition work this program has not needed before (every prior determinism study used synthetic fixtures) |
| **Total, Phase 2 alone** | **~5–10 engineering days**, plus unbounded licensing lead time | Consistent with `docs/SCRFD_PROGRAM_DESIGN.md` §8's "quickest vertical slice... realistic" framing for Phases 0–3 combined |

---

## 9. Explicit stop/go criteria for proceeding to SCRFD production cutover (Phase 4)

**Go**, only if **all** of the following hold, each independently checkable:

1. Phase 2's full suite is green with **zero changes to any file outside
   this document's §3.2 list** (the "widened beyond scope" check every
   prior phase in this program has used).
2. `default_face_selection().detector` is still
   `"opencv-haar-frontalface-default"` at the end of Phase 2 (test H) —
   i.e., nothing was accidentally defaulted early.
3. §2.3's concurrency test (E) passes — an unsafe ONNX session sharing bug
   is not something to discover after Phase 4 ships it into
   `asyncio.to_thread`.
4. §2.4's license is **read and recorded**, not assumed, for the exact
   checkpoint used in Phase 3's measurement.
5. **Phase 3's R13 measurement clears the acceptance criterion exactly as
   written** (§2.8, quoted verbatim from `REPRODUCIBILITY_AUDIT.md:1516-
   1518`): no face lost across ≥200 real photographs, box short-edge
   movement ≤1%, amplification below 10× (from the measured 310×).

**No-go / abort, restated from `docs/SCRFD_PROGRAM_DESIGN.md` §9.3 because
it applies unchanged:** if (5) fails, Phase 4 does not proceed on structural
argument alone. A failed measurement is a complete, reportable result — a
specific SCRFD configuration measured and found not to clear a
pre-registered bar — not a reason to lower the bar or ship anyway. This
document's own scope (Phase 2) is unaffected by that outcome either way:
`SCRFDDetector` remains wired, tested, and available for a different
checkpoint/variant to be measured against the same criterion (§3.1's
model-size choice becomes the next thing to vary, not the architecture).

---

## 10. Hidden blockers, stale assumptions, and design drift — search results

Performed as an explicit pass over every artefact this program has produced
since Batch 28, not folded silently into the sections above:

1. **Found and corrected in this document:** `docs/SCRFD_PROGRAM_DESIGN.md`
   §5 Phase 2's claim of zero `.f.` impact does not survive reading
   `fingerprint()`'s actual implementation (§2.2/§4/§5). This is the
   substantive drift this pass exists to catch.
2. **Found and corrected:** §3.1's "onnxruntime is a new dependency"
   framing — it is already declared, already lazily imported elsewhere,
   already covered by a generic dependency-audit test (§2.5).
3. **Found and corrected:** §3.3's "option (a) is cheap because it doesn't
   ripple through call sites" reasoning was right in conclusion, wrong in
   its stated reason — there is exactly one call site regardless of which
   option is chosen; the real cost driver is the "no defaults" rule
   interacting with that one site (§1.2).
4. **Found, not previously stated anywhere in this program's documents:**
   the NaN-sentinel equality footgun (§1.3) — latent, not live, but a real
   trap for whoever writes the dataclass extension without reading this
   document.
5. **Checked and found to be *not* drift, despite surface appearance:**
   `docs/SCRFD_PROGRAM_DESIGN.md` §2.3 cites `docs/CROP_GEOMETRY.md`'s
   "Option E" by a name that document does not literally use (its heading
   is `### E — letterbox...`, `docs/CROP_GEOMETRY.md:504`) — a labelling
   informality, not a substantive disagreement; that section's own
   "Recommendation" (`:544-560`) is consistent with both documents'
   description of it. Noted so a future reader searching for the literal
   string "Option E" in `CROP_GEOMETRY.md` does not conclude the citation
   is fabricated.
6. **Checked and found current:** `docs/PROJECT_STATUS_FINAL.md`'s 54/1327
   figure is one batch stale (pre-dates Batch 28) but was never treated as
   authoritative by either this document or `docs/SCRFD_PROGRAM_DESIGN.md`
   — both re-derive the live figure (§0). Flagged so it is corrected in
   that document the next time it is touched, not silently left
   inconsistent.
7. **New candidate finding, not yet an audit row — tentatively N10** (next
   free slot; `REPRODUCIBILITY_AUDIT.md`'s highest closed/open rows are
   R36/N9): **`onnxruntime`'s own version is unpinned** in
   `[tool.uv].override-dependencies` (`pyproject.toml:96-105`, which pins
   only `opencv-python`/`numpy` — the two packages N5 identified as
   feeding a `fingerprint()` call). Once Phase 3/4 makes ONNX inference
   output feed a real measurement or a shipped `.f.` value, a future
   `onnxruntime` minor release changing op-level numerics could move
   `SCRFDDetector`'s output non-reproducibly across environments, the same
   class of gap N5 closed for `opencv-python`/`numpy`. **Not a Phase 2
   blocker** — nothing in Phase 2 ships or stores a reproducibility-
   sensitive value derived from `onnxruntime`'s numerics — but it should
   be resolved before Phase 3's measurement is treated as authoritative
   across more than one machine, and definitely before Phase 4.
8. **Not found:** no evidence of a second, competing SCRFD design document,
   no evidence of a partially-implemented `SCRFDDetector` anywhere in the
   working tree (`grep -rn "SCRFDDetector"` outside this document and
   `docs/SCRFD_PROGRAM_DESIGN.md`'s own code-block example returns
   nothing), no evidence of concurrent-session drift on any file this
   phase touches (`git status` re-checked at the start and end of this
   session's research).

---

## 11. Gate

**Design complete. Ready for implementation, pending approval and pending
§3.4's licensing verification specifically** (the one item this document
could not close from source alone). Every other open question `docs/
SCRFD_PROGRAM_DESIGN.md` deferred to "implementation-time review" (§3.3) is
resolved here with a concrete recommendation, grounded in source facts that
document did not have (§1.2's single-constructor-site finding, §1.3's
NaN-equality risk, §2.2's `fingerprint()`-shape argument). The one
correction this document makes to the program design that is **not**
merely a sharpening — §2.2's `.f.` move — is stated as a correction, not
smoothed over, because the alternative (pretending Phase 2 is truly
zero-impact) would be exactly the kind of unflagged numbering/scope drift
`docs/SCRFD_PHASE1_ABSTRACTION_VERIFICATION.md` §0 already warned this
program against repeating.

**No code was written to produce this document. Stop here — waiting for
approval to proceed to implementation of `inference/preprocess.py`'s
`SCRFDDetector`/dispatch/dataclass extension, the new test files, and the
`tests/run_all.py` registration exactly as scoped in §3.**
