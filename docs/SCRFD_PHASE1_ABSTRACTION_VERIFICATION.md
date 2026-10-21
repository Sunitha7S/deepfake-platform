# SCRFD Program — Detector Abstraction Layer: Design Verification & Implementation Readiness

**Date:** 2026-09-18. **Type:** verification + planning only — no code. **Basis:**
a fresh source re-derivation this session (`git status` re-checked first —
no concurrent-session drift on any file this task touches since
`docs/SCRFD_PROGRAM_DESIGN.md` was written), not an assumption that that
document is correct.

## 0. Naming correction — read this before the rest

The request asks to implement **"Phase 0"** of `docs/SCRFD_PROGRAM_DESIGN.md`,
and its own requirements 2–7 ("start with the detector abstraction layer
only," "no SCRFD model execution yet," "no identity changes") describe
exactly what that design document calls **Phase 1 ("Detector abstraction,
zero behavior change")**. That document's own **Phase 0** is "Asset &
dependency decision (no production code)" — a licensing/dependency choice
with zero lines of code, which cannot be what "implement" and "detector
abstraction layer" are asking for.

**This is treated as the numbering being off, not the scope being off** —
the six numbered requirements are unambiguous and this document proceeds
against them. Everything below is Phase 1 of `docs/SCRFD_PROGRAM_DESIGN.md`.
Flagging this explicitly rather than silently renumbering is itself the
point of "do not assume the design document is correct" — a design
document's own phase labels are exactly the kind of claim that should be
checked, not inherited, and this project has already paid once for
unflagged numbering drift (`docs/PROJECT_STATUS_FINAL.md` §7's finding
about stale gates applies to phase numbers too).

---

## 1. Design verification report

Every claim in `docs/SCRFD_PROGRAM_DESIGN.md` §2.1/§3.2 that bears on this
phase was re-checked against current source this session. Three claims are
**confirmed**, one is **sharpened** (the design document was directionally
right but not precise enough to implement from directly), and one **new
constraint** was found that the design document did not mention at all.

### 1.1 Confirmed: `detect_faces` has exactly one production call site

`grep`-verified this session across the whole repository (excluding
`.venv`): `inference/image_pipeline.py:296` (`_detect_and_measure`) is the
only production call. `inference/video_pipeline.py` calls it zero times —
re-confirmed by the same two tests that already pin this
(`tests/test_face_selection_identity.py:645`,
`tests/test_face_size_policy.py:677`, both asserting
`"detect_faces" not in <video_pipeline.py source text>`). Both read
`inference/video_pipeline.py` specifically — **this phase does not touch
that file, so both assertions are unaffected by construction.**

### 1.2 Confirmed: `inference/preprocess.py` is not re-exported through `inference/__init__.py`

Read in full this session. Its own docstring states the policy explicitly:
*"Preprocessing and bundle construction are reachable through their
modules but deliberately absent from this namespace."* `FaceDetector` and
`HaarDetector` follow the same rule — **defined in `inference/preprocess.py`
only, not added to `inference/__init__.py`'s `__all__`.** This is a
constraint the design document did not state and this verification adds.

### 1.3 Sharpened: "becomes a dispatcher" (design doc §5, Phase 1) is too vague to implement from

The design document's own Phase 1 description says `detect_faces` "becomes
a dispatcher" without saying what it dispatches *on*. Read literally, that
invites building policy-driven selection (checking `policy.detector`)
**now** — which would be new behavior (a new decision: what happens on a
mismatch?) and would pre-empt Phase 2's actual scope (`SCRFDDetector`,
registry-by-string). **Correction, adopted for this implementation:**
Phase 1's `detect_faces` is **hard-wired** to a single `HaarDetector`
instance — a pure `extract method` refactor, not a routing decision. This
matches requirement 3 ("no production behavior changes yet") more strictly
than the design document's own wording did.

### 1.4 New constraint the design document did not mention: three module-level objects must stay untouched, in place, by name

Grep-verified this session, not in `docs/SCRFD_PROGRAM_DESIGN.md`:

- `_cascade_local` (`inference/preprocess.py:327`) — a module-level
  `threading.local`, the R15 fix. **Directly asserted by
  `tests/test_inference_determinism.py:753`**
  (`assert isinstance(pre._cascade_local, _threading.local)`). Must remain
  a module-level object, not become per-`HaarDetector`-instance state — an
  instance created fresh per call (as a stateless wrapper) would not
  regress thread-safety *only if* it keeps reading this same shared
  `threading.local` through `_face_cascade()`, not a new one.
- `_face_cascade()` (`:362-385`) — **directly called and its per-thread
  `id()` compared by `tests/test_inference_determinism.py:766`**, and
  independently called by two test/study files that reimplement detection
  for comparison (`tests/face_selection_study.py:109`,
  `tests/test_face_selection_identity.py:130`). Must stay a module-level
  function; `HaarDetector.detect()` must call it, not inline its own
  cascade-loading logic.
- `cascade_digest()` (`:388-411`) — read by `default_face_selection()`
  (`:613`) and directly asserted stable by
  `tests/test_face_selection_identity.py:370-373`. Must stay untouched;
  `HaarDetector` must not introduce a second, competing asset-hash path.

**Net effect on the plan:** `HaarDetector.detect()` is an **extraction**,
not a **reimplementation** — its body must be the current `detect_faces`
box-detection logic verbatim (`grey = cv2.cvtColor(...)`;
`_face_cascade().detectMultiScale(...)`; `FaceBox(int(x), ...)`
construction), calling the same free functions it calls today. `detect_faces`
itself keeps the `face_order_key` sort **outside** the detector (it is a
detector-independent total order over box geometry per its own docstring,
`:831-876` — belongs at the dispatch level so a future `SCRFDDetector`
does not have to reimplement it).

### 1.5 Confirmed: `detector`/`detector_asset` are, today, pure identity labels — not functionally read

`detect_faces` never inspects `policy.detector` or `policy.detector_asset`
in its control flow — confirmed by reading the full function body
(`:879-906`). They feed `FaceSelectionPolicy.identity` only. This matters
for §1.3's correction: since nothing today branches on `policy.detector`,
introducing branching now (rather than in Phase 2, when a second
implementation actually exists to branch to) would be new behavior, not a
refactor.

---

## 2. Exact file list

| File | Change | Why only this file |
|---|---|---|
| `inference/preprocess.py` | **Only production change.** Add `FaceDetector` `Protocol` (mirroring `ModelBackend`, `inference/models.py:380-393`, same `@runtime_checkable` style). Add `HaarDetector` class whose `detect()` method is the extracted, verbatim body of today's `detect_faces` (minus the final sort). Rewrite `detect_faces` to: resolve the policy default, construct/call `HaarDetector`, sort with the existing `face_order_key`. `_face_cascade`, `_cascade_local`, `cascade_digest`, `FaceBox`, `face_order_key`, `FaceSelectionPolicy` — **all untouched**, confirmed no reason to touch any of them in §1.4/§1.5. | The only place `detect_faces` and every Haar-specific internal live. §1.1/§1.2 confirm nothing else needs a change for this phase's scope. |
| New: `tests/test_face_detector_abstraction.py` | New suite proving the refactor is inert (§4). | This project's own convention — a new, named test file per finding/change, not additions scattered into unrelated files. |
| `tests/run_all.py` | `+1` import line, `+1` `SUITES` tuple row, alphabetically placed per the existing convention (confirmed this session — the file's own import block is alphabetically ordered). | Every new suite in this project's history registers here; confirmed by reading the file's current structure this session. |

**Explicitly not touched, and why, stated per §1's findings rather than
asserted:**

- `inference/image_pipeline.py` — calls `pre.detect_faces(image,
  policy=policy)` by attribute lookup at call time (`from inference import
  preprocess as pre`, confirmed this session); the name, module location,
  and signature of `detect_faces` are all unchanged, so this file needs no
  edit.
- `inference/video_pipeline.py` — never calls `detect_faces` (§1.1);
  zero reason to touch it, and touching it would falsify two existing
  tests' premise (§1.1).
- `inference/models.py`, `contracts/*`, `backend/config.py`,
  `evaluation/*` — nothing in this phase moves `.f.`, `.d.`, or any
  contract/schema field (requirements 4–5), so none of the files that
  carry those identities need any change. Verified by re-reading which
  fields feed each: `FaceSelectionPolicy.identity` reads `detector` and
  `detector_asset` as *values*, and this phase changes neither value —
  only *how* Haar's detection is invoked internally, which the identity
  does not observe.
- Every existing test file that references `detect_faces`/`crop_face`
  (§1.1's grep found ~20, including 5 that monkey-patch
  `pre.detect_faces` directly: `tests/branch_b_semantics_study.py`,
  `tests/face_size_policy_study.py`, `tests/multi_face_study.py`,
  `tests/test_branch_b_semantics.py`,
  `tests/test_headline_and_face_semantics.py`) — **none need editing**,
  because `pre.detect_faces = ScriptedDetector(...)` reassigns the module
  attribute directly, and that mechanism is unaffected by what the
  original function's body looks like internally. This is the single
  most load-bearing backward-compatibility fact this verification
  establishes: **the monkey-patch pattern only survives if `detect_faces`
  stays a plain, directly reassignable module-level function** — which
  §1.3's "hard-wired, not a registry" correction guarantees, and a
  design that instead resolved a detector through some cached
  singleton/registry object at import time could have silently broken.

---

## 3. Dependency analysis

- **No new external dependency.** `Protocol` and `runtime_checkable` are
  already imported elsewhere in this codebase (`inference/models.py:28`,
  stdlib `typing`) — this phase adds the same stdlib import to
  `inference/preprocess.py`, nothing new to `pyproject.toml` or `uv.lock`.
- **No new internal dependency edges.** `inference/preprocess.py` already
  depends on `cv2`/`numpy`/`contracts.version`; nothing added references
  anything outside what the module already imports. No other module gains
  a new import of `inference.preprocess` — the public surface
  (`detect_faces`, `FaceBox`, `FaceSelectionPolicy`, `default_face_
  selection`, `cascade_digest`, `face_order_key`) is unchanged in name,
  signature, and module location.
- **No dependency on Phase 0.** This phase needs no SCRFD asset, no
  `onnxruntime`, no new pin — it can proceed independent of whether the
  design document's own Phase 0 (asset/licensing decision) has happened.
  Re-confirmed against requirement 6 ("no SCRFD model execution yet"): the
  `FaceDetector` Protocol this phase introduces has exactly one
  implementation (`HaarDetector`); nothing imports or references SCRFD,
  ONNX, or any new binary asset anywhere in this phase's file list.
- **Test-suite dependency graph** — the new suite depends only on what
  `tests/test_preprocess.py`/`tests/test_inference_determinism.py` already
  depend on (the existing synthetic face fixtures); no new fixture asset
  is introduced.

---

## 4. Risk analysis

| Risk | Severity | Mitigation, grounded in §1's findings |
|---|---|---|
| Monkey-patch tests silently stop being effective (a test believes it replaced detection but production code routes around the patched name) | **High if it happened, but structurally prevented** | §1.3/§2's design keeps `detect_faces` a plain module-level function that resolves `HaarDetector` internally on each call — `pre.detect_faces = X` still fully replaces the entry point every caller uses, exactly as today. The new suite (§5, test D) asserts this directly rather than trusting the design. |
| Thread-safety regression (R15 reintroduced) by accidentally giving `HaarDetector` its own cascade state instead of reusing `_cascade_local` | **High if it happened** | §1.4 identifies the exact three objects (`_cascade_local`, `_face_cascade`, `cascade_digest`) that must stay untouched and module-level; `HaarDetector.detect()` is specified as calling `_face_cascade()`, not reimplementing it. The existing `tests/test_inference_determinism.py` thread-safety suite (unmodified) is part of the regression net (§5). |
| Accidental identity movement (`.f.` moves when it shouldn't, violating requirement 4) | **High if it happened, low likelihood given scope** | Nothing in this phase's file list touches `FaceSelectionPolicy`, `default_face_selection()`'s field values, or anything `identity` reads (§2's "explicitly not touched" list, §1.5). A pinned literal test (§5, test C) makes any accidental drift a loud failure rather than a silent one. |
| Ordering behavior changes because `face_order_key`'s sort ends up applied twice, applied inside `HaarDetector` *and* at the dispatcher, or dropped entirely | Medium | §1.4 specifies the sort stays **outside** the detector, at the `detect_faces` dispatch level, exactly once — matching today's single sort call. The equivalence test (§5, test B) would catch either a missing or a duplicated sort as an ordering mismatch. |
| Scope creep — a `SCRFDDetector` stub or an `onnxruntime` import sneaks in "since we're already touching this file" | Medium, procedural | Explicitly out of scope per requirement 6; this document's file list (§2) names `HaarDetector` as the only implementation this phase adds. Flagged here so review catches it if it happens, not assumed away. |
| `Protocol`/`@runtime_checkable` overhead or an `isinstance` check somewhere getting expensive in a hot path | Low | `detect_faces` is already called once per image analysis through `asyncio.to_thread` (`inference/image_pipeline.py:156-159`); a `Protocol` conformance check, if used at all in this phase, is at most a one-time construction-site check, not per-call. |
| A future reader assumes `detector`/`detector_asset` are now functionally read because a `Protocol` exists | Low | §1.5's finding — they remain pure labels in this phase — should be stated in the new code's docstring (implementation detail, not part of this verification document, but flagged here so the eventual PR description doesn't overclaim what Phase 1 does). |

**No risk above requires a design change** — every mitigation is already
implied by §1's re-derivation; this table exists to make each one an
explicit, checkable claim rather than an assumption carried in prose.

---

## 5. Fail-before / pass-after test plan

**Framing, stated once rather than per test:** this phase is a pure
extraction refactor with an intended zero-behavior-change outcome — there
is no pre-existing bug being fixed, so there is no traditional
fail-before/pass-after *bug* pair. The applicable shape is the one this
project's own label-closure batches (B18, B22) and R28's reachability
suite already established: **prove the refactor is inert**, and let the
**existing, unmodified test suite be the regression net**.

### 5.1 Baseline (before any edit)

Run `python tests/run_all.py` against the current working tree and record
the result as the baseline to diff against. Per `docs/
PROJECT_STATUS_FINAL.md` §8's most recent confirmed run: **54 suites, 1327
tests, exit 0.** This phase's own baseline run should reproduce that
figure before any file in §2 is touched — if it does not, that is a
pre-existing-drift signal unrelated to this phase and must be resolved
first, not folded into this phase's diff.

### 5.2 New suite: `tests/test_face_detector_abstraction.py`

| # | Test | Assertion | What it would catch |
|---|---|---|---|
| A | `test_haar_detector_satisfies_the_face_detector_protocol` | `HaarDetector` structurally conforms to `FaceDetector` (an `isinstance` check if `@runtime_checkable`, or a `hasattr`/signature check otherwise) | The abstraction was declared but not actually implemented against |
| B | `test_haar_detector_and_detect_faces_agree_on_the_shipped_fixture_corpus` | For every fixture image already used by `tests/test_preprocess.py`/`tests/test_inference_determinism.py`, `sorted(HaarDetector().detect(image, policy=policy), key=face_order_key) == detect_faces(image, policy=policy)`, box-for-box | A behavior change introduced anywhere in the extraction — box coordinates, ordering, or count |
| C | `test_default_face_selection_identity_is_unmoved` | `default_face_selection().identity == "<the literal value read live immediately before this phase's edit>"`, and `cascade_digest()` still returns the same value | Requirement 4 (no `.f.` movement) violated silently |
| D | `test_detect_faces_stays_monkeypatchable_at_module_level` | `original = pre.detect_faces; pre.detect_faces = lambda *a, **k: []; ...; pre.detect_faces = original` still fully intercepts a call made through `ImagePipeline`'s `_detect_and_measure` path (or a minimal direct-call equivalent) | The exact regression the risk table's top row names — the abstraction accidentally breaking the ~5 test/study files that already rely on this pattern |
| E | `test_face_cascade_and_cascade_local_are_unchanged_objects` | `pre._face_cascade` and `pre.cascade_digest` are still the same functions (`is`-compared to a reference captured before the refactor, or asserted by source inspection that `HaarDetector.detect` calls them rather than duplicating their logic) | A reimplementation instead of an extraction — §1.4's core constraint |

### 5.3 Registration and full-suite verification (after the edit)

1. Register the new suite in `tests/run_all.py` (§2).
2. Run `python tests/run_all.py` again. **Expected: 55 suites** (54 + this
   phase's one new suite), **same 1327-plus-the-new-suite's-own-test-count
   tests, 0 failures**, and — the specific, named check for this phase —
   **zero existing test file changed**, so any literal (`STUB_ENGINE_VERSION`,
   `default_face_selection().identity`, or any of the ~20 files identified
   in §1.1/§2 that reference `detect_faces`) that needed editing to stay
   green is itself evidence this phase moved more than intended.
3. **Pass condition for this phase as a whole:** the diff between the
   before (§5.1) and after (§5.3) full-suite runs is exactly "+1 suite,
   its own new tests, 0 changes anywhere else." Anything wider than that
   is a scope violation of requirements 3–5, not a detail to fix in place.

---

## 6. Explicit non-goals, restated from the request rather than assumed

- No `SCRFDDetector`, no `onnxruntime` import, no ONNX asset — confirmed
  absent from the file list (§2) and the dependency analysis (§3).
- No change to `FaceSelectionPolicy`'s fields, `identity`, `to_dict`, or
  `differences` — confirmed untouched (§2, §5 test C).
- No `CONTRACT_VERSION`/`RUN_SCHEMA_VERSION`/`CALIBRATION_WITNESS_VERSION`
  change — none of the files that carry those version numbers appear in
  this phase's file list (§2).
- No `.d.` change — decode identity is untouched by construction (nothing
  in `inference/decode_env.py` is in scope).
- Full backward compatibility with the existing Haar pipeline — this is
  what §5's entire test plan exists to prove, not merely assert.

---

## 7. Gate

**Verification complete. Ready for implementation, pending approval.** No
blocking defect was found in re-deriving this phase from source; one
imprecision in the design document was sharpened (§1.3) and one constraint
it omitted was found and folded into the plan (§1.4) — both before any
code was written, which is what this verification pass was for.

**Stop here — no code written, per instruction. Waiting for approval to
proceed to implementation of `inference/preprocess.py`, the new test file,
and the `tests/run_all.py` registration exactly as scoped in §2.**
