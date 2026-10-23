# R23 + R28 — Design (Phase 5)

**Date:** 2026-09-18. **Type:** design only — no code. **Method:** complete
source re-derivation. Every claim below was independently re-checked against
the current working tree (`git status` read first; `HEAD = 8676fdd`, same
uncommitted OR1-b/R30/Batch-27 tree, zero drift since Batch 27) rather than
assumed from `docs/CROP_GEOMETRY.md`, `docs/CROP_IDENTITY.md`,
`REPRODUCIBILITY_AUDIT.md`, or `docs/PROJECT_ROADMAP.md`. Those documents
are **inputs to be verified, not premises**, per instruction.

**Baseline, re-confirmed live:** 54 suites (`tests/run_all.py` `SUITES`,
counted this session), `STUB_ENGINE_VERSION` ends
`...f.e629642c.d.c9d4025c` (`backend/config.py:41`), `CONTRACT_VERSION =
1.0.0`, `RUN_SCHEMA_VERSION = 1.2.0`, `CALIBRATION_WITNESS_VERSION = 1.3.0`,
`LOCK_VERSION = 1.1.0` (all read live). No SCRFD (`inference/preprocess.py`
still calls a Haar `CascadeClassifier`, confirmed by grep — no `scrfd`
import anywhere in production code).

**Scope, exactly as asked:** R23 (crop-geometry distortion/occupancy) and
R28 (`resize`'s interpolation predicate). R24 (video whole-frame squash) and
R26 (published-bbox gap, already partially closed in Batch 27) are adjacent
findings mentioned only where they bound R23/R28's scope — **not
re-derived here**, per the user's explicit scope.

---

## 1. Executive summary

**Neither R23 nor R28 can be pixel-fixed now, and that conclusion is
unchanged from `docs/CROP_GEOMETRY.md`'s original Batch 15 recommendation —
re-derived, not carried over.** The reason has not changed: no real model
exists (`StubBackend` only, hash-derived scores), so any pixel-moving
geometry change ships "on structural argument alone," the same pattern that
forced a full adversarial-review cycle for R30 before it was allowed to
touch even a *non-pixel* identity field. A geometry fix touches pixels
directly.

**Correction to the standing framing, found by re-deriving rather than
quoting:** R23 and R28 are usually described as split cleanly along
"SCRFD-gated vs. not." That split is real for R23's **occupancy** half
(literally unreachable without a free-aspect detector — Haar's boxes
satisfy `|w−h| ≤ 1`, confirmed unchanged this session) but it
**mischaracterizes** R23's **border-clip** half and all of R28: both are
reachable *today*, confirmed live, yet neither should be fixed now — not
because of SCRFD, but because **no real model exists to validate a pixel
change against**. The user's own three-way category ("blocked by SCRFD,
video-face-stage, or real-model work") names this correctly as a *third*,
distinct blocking reason, and the roadmap's current two-way framing
("Border-clip half: none [dependency] / Occupancy half: SCRFD") is
technically accurate about *reachability* but invites the reader to think
the border-clip half has no blocker at all, which is false — its blocker is
real-model work, restated precisely in §4.

**New finding this session, empirically verified, not previously
established in any prior document:** R28's defect is **not** a purely
scripted/hypothetical scenario. A genuine `inference.preprocess.crop_face()`
call — not a scripted double — with a realistic large, near-border face box
(350×350 detected face, margin 1.3, positioned near a 1200×1200 image's
right edge) produces a **455×252** crop: one axis (455) correctly exceeds
`input_size` and gets `INTER_AREA`, but that same interpolation choice is
then applied to the *other* axis (252), which is being **upsampled** —
R28's exact defect, reproduced from first principles this session (§3.2),
not read off `docs/CROP_GEOMETRY.md`'s own scripted 500×300 random-noise
example. This does not change the "don't fix it now" recommendation, but it
does mean **a cheap, zero-pixel-risk closure is available now**: a
regression test proving the defect's real-world reachability, the same
"measured and rejected" pattern R28's own `interpolation_rule` label
already uses, and the same shape as Batch 27's R26 Tier 2.

**No part of R23 or R28 is obsoleted** by N5, OR1-b, R30, R31, or Batch 27
(R16+R26) — none of those five batches touches `inference/preprocess.py`,
`FaceSelectionPolicy`, or any crop-geometry code path (confirmed by grep,
§5). **One new, forward-looking dependency exists that did not exist when
`docs/CROP_GEOMETRY.md` was written**: R31's `ModelSpec.
pre_calibration_identity` (weights + gate + **selection**) is now inside
`CalibrationWitness`'s digest for schema 1.3.0 — any future `.f.`-moving
R23/R28 fix would, for the first time, need to consider witness
compatibility (currently zero live witnesses exist, so zero practical risk
today, but a new fact to carry forward).

---

## 2. Re-derived: R23 — crop geometry distortion and occupancy collapse

### 2.1 Current status, re-read from source, not from the audit row

`REPRODUCIBILITY_AUDIT.md:113` (live-read): *"Characterised in B15;
documentation half CLOSED, correctness halves open."* Re-verified against
`inference/preprocess.py:909-963` (`crop_face`, read in full this session):
byte-identical in shape to every description in `docs/CROP_GEOMETRY.md` —

```python
h, w = image.shape[:2]
side = int(max(box.width, box.height) * policy.crop_margin)
cx, cy = box.x + box.width // 2, box.y + box.height // 2
x0 = max(0, cx - side // 2)
y0 = max(0, cy - side // 2)
x1 = min(w, x0 + side)
y1 = min(h, y0 + side)
crop = image[y0:y1, x0:x1]
```

No batch between B22 and now has touched this function — confirmed by
checking what N5/OR1-b/R30/R31/Batch-27 actually changed (§5): none of
their diffs reference `crop_face`, `FaceSelectionPolicy`, or
`inference/preprocess.py` at all.

### 2.2 Original concern, and whether each half still exists

**Half 1 — border-clip distortion (live today, no detector dependency).**
The window is *intersected* with the image (not padded), so a box near the
right/bottom edge yields a truncated, non-square crop that `prepare_image`'s
square resize then stretches. **Re-verified this session, live**, using the
exact shipped `crop_face`:

```
FaceBox(280, 140, 200, 200) in a 480×480 image, margin 1.3
  -> side = 260 (square, unclipped)
  -> crop = image[10:270, 280:480]  = 200 columns wide, 260 tall
  -> 200x260, anisotropy 1.30 (matches docs/CROP_GEOMETRY.md's 13% figure family)
```

Still exists. Nothing about the computation changed.

**Half 2 — left/top mis-framing (live today).** `x0`/`y0` clamp to 0, so the
crop keeps its full square `side` but the face slides off-centre rather than
distorting. Still exists — same code path, unchanged.

**Half 3 — corner scale error (live today).** Both axes clip simultaneously
at a corner, so the crop shrinks on both dimensions but stays square — same
*shape* as an unclipped crop, different *scale* (measured previously: face
arrives 28% larger than from frame centre). Still exists.

**Half 4 — occupancy collapse (latent, SCRFD-gated).** `side` is derived
from the crop's **long** edge (`max(box.width, box.height)`), so a
free-aspect box produces a crop that is mostly not-face. **Re-verified
unreachable under Haar this session**: `inference/preprocess.py:879-906`
(`detect_faces`) constructs `FaceBox` from `cv2.CascadeClassifier.
detectMultiScale`'s raw output, and Haar's frontal-face cascade returns
square-ish boxes (`|w−h| ≤ 1`, the same structural guarantee re-verified for
R26 Tier 2 in Batch 27 via `docs/CROP_GEOMETRY.md`'s own 50,000-box brute
force) — there is no live path to a free-aspect box today. Still latent,
still SCRFD-armed, unchanged.

**R25 (identity coverage of the three geometry knobs) is a dependency, not
part of R23 itself, and is independently confirmed CLOSED** (Batch 22,
`REPRODUCIBILITY_AUDIT.md:115`, re-verified: `FaceSelectionPolicy` carries
`interpolation_rule` and `crop_clip_rule` today, both present in
`inference/preprocess.py:499-528`, read this session).

### 2.3 Cheap / documentation closure path

**Already fully closed (Batch 15).** Re-verified: `crop_face`'s docstring
(read in full, §2.1's excerpt above continues into the docstring text) no
longer claims to avoid aspect distortion — it states the three border
behaviours, the exact truncation condition, and the occupancy consequence,
matching `docs/CROP_GEOMETRY.md` §2.1's description of the correction
verbatim. **Nothing remains to document.** This is a material difference
from R16/R26 (Batch 27), where the documentation-caveat path was found only
*partially* done — here it is fully done, and re-deriving it a second time
confirms rather than corrects the audit row.

### 2.4 Full engineering closure path

Per `docs/CROP_GEOMETRY.md` §9 (options A–F, re-read in full this session,
not summarized from memory): **Option E (letterbox — compute the square
window, fill what falls outside rather than cutting it down) is the
recommended fix**, with **Option F (affine-preserving resize, SCRFD's actual
landmark-alignment need) as the eventual destination** once landmarks exist.
Options B/C/D are independently rejected on correctness grounds (trades one
defect for a worse one), not merely deferred.

**Why not now, re-derived rather than quoted:** every option changes real
pixels for real inputs today (§9's own per-option "Benchmark continuity"
row — re-read, unchanged: *"broken only for crops that clip today... but
*which* results change is content-dependent, so a clean comparison across
the boundary is not available"*), and there is no real model to measure
whether the fixed pixels produce a *better* verdict, only that they produce
a *different* one. This is exactly the reasoning `docs/R30_BUNDLE_DESIGN.md`
§7's risk register used for a much smaller, non-pixel change (R30's
identity-only fix still needed two adversarial-review cycles before
shipping) — a pixel-moving geometry fix is a strictly larger version of the
same risk, unmitigated by anything landed since B15.

### 2.5 Files affected (informational — not touched by this design, and not recommended for this batch)

`inference/preprocess.py` (`crop_face`, whichever option is eventually
chosen), `inference/models.py` or `FaceSelectionPolicy` (`crop_clip_rule`'s
value moves from `"intersect-image"` to the chosen fill mode's name — the
"one accepted double-spend" `docs/CROP_GEOMETRY.md` §10's superseding note
already flagged), the full `tests/test_crop_geometry.py` /
`tests/test_crop_identity.py` surface, `docs/CROP_GEOMETRY.md` (a new
implementation section, matching the pattern every prior batch used).

### 2.6 Version / migration impact (informational — for the eventual fix, not this batch)

`.f.` moves once (`FaceSelectionPolicy.identity` changes because
`crop_clip_rule`'s value changes — a real, not label-only, change this
time). One cache generation, both media namespaces (N8's accepted
architectural cost, re-exercised). `STUB_ENGINE_VERSION` moves. **New
consideration since R31 (§1):** `ModelSpec.pre_calibration_identity`
(weights + gate + **selection**) would move too, which — for the first time
in this codebase's history — puts a `.f.`-moving batch inside
`CalibrationWitness`'s digest scope for schema 1.3.0. Zero stored witnesses
exist (confirmed by the same search R31 and Batch 26 both performed), so
this is zero practical risk today, but the eventual design doc for eventual
implementation must state it rather than silently inherit R31-era baseline
knowledge. No `CONTRACT_VERSION`/`RUN_SCHEMA_VERSION`/`LOCK_VERSION`/OpenAPI
change (crop geometry is a pixel/identity concern, not a wire-contract
one — re-confirmed, `contracts/common.py`/`contracts/evidence.py` do not
reference `crop_face` or `FaceSelectionPolicy`).

---

## 3. Re-derived: R28 — `resize`'s interpolation predicate

### 3.1 Current status, re-read from source

`REPRODUCIBILITY_AUDIT.md:117` (live-read): *"Open — deferred (B22)...
Fixing the predicate moves shipped pixels, so it is not in B22."*
`inference/preprocess.py:229-238` (`resize`, read in full this session):

```python
def resize(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interp)
```

Byte-identical to every prior description. `FaceSelectionPolicy.
interpolation_rule` (`inference/preprocess.py:499-514`, read in full) still
names this exact `or` behaviour verbatim
(`"area-if-either-axis-exceeds-target-else-linear"`) and is held to it by
`tests/test_crop_identity.py::test_the_interpolation_rule_name_matches_resize`
(re-read, confirmed still present and still testing the named-vs-actual
correspondence, not the reachability question — see §3.3).

### 3.2 Original concern, and whether it still exists — with a new, empirically-verified nuance

**The concern**: a single `INTER_AREA`/`INTER_LINEAR` choice is applied to
*both* axes of a `cv2.resize` call from one `or`-combined predicate. When a
crop's two dimensions straddle `size` (one axis > `size`, the other <
`size`) — which the current predicate can produce, since it only checks "is
*either* axis over the limit" — the axis being **enlarged** (upsampled) gets
`INTER_AREA`, an area-averaging low-pass method designed for *shrinking*,
in a module whose own docstring states the reason for choosing `INTER_AREA`
at all is that *"forensic signal lives in high-frequency texture."*

**Still exists — confirmed live, not merely named.** Every prior document
(`docs/CROP_GEOMETRY.md` §6 Knob 2, `tests/crop_geometry_study.py:639-651`)
demonstrates this only against a **scripted** 500×300 random-noise array
fed directly to `resize()` — never derived from a real `crop_face()` call,
and no prior document establishes whether the shipped pipeline can actually
*produce* a crop of that shape.

**Re-derived and verified empirically this session, from first principles,
using the real (unscripted) `crop_face()`:**

```python
policy = pre.default_face_selection()      # shipped margin = 1.3
img = np.full((1200, 1200, 3), 128, np.uint8)
box = FaceBox(x=1000, y=400, width=350, height=350)   # a large, near-border face
crop = pre.crop_face(img, box, policy=policy)
# -> crop.shape[:2] == (455, 252)
```

`side = int(350 * 1.3) = 455` (> `input_size` 384, correctly triggers
`INTER_AREA` on that axis); the right-border clip truncates the *other*
axis to 252 (< 384, being **upsampled**, incorrectly inheriting
`INTER_AREA`). **This is the mixed-case shape, produced by the real,
shipped `crop_face()`, for a plausible real-world input** (a close-up or
high-resolution portrait — no floor or ceiling in `FaceSelectionPolicy`
prevents a detector, Haar or future SCRFD, from emitting a box this large
near a border). **R28 is therefore live and reachable today**, through the
same border-clipping mechanism R23 already established as live, not a
finding that waits on SCRFD or a hypothetical scenario. This is a
correction to how every prior document frames the finding's evidentiary
basis (scripted vs. real), not a correction to the finding's existence.

### 3.3 Cheap / documentation closure path — expanded from "already done" to "one gap remains"

`FaceSelectionPolicy.interpolation_rule`'s label is already accurate
(§3.1). **What is missing, found by re-deriving rather than assuming
"nothing more to document"**: no test anywhere in the suite proves the
scenario in §3.2 is reachable through genuine pipeline code — every
existing R28-adjacent test (`test_the_interpolation_rule_name_matches_resize`,
`docs/CROP_GEOMETRY.md`'s own study) exercises `resize()` in isolation
against a scripted array, or exercises `crop_face()` for shape/anisotropy
without checking whether the resulting shape ever crosses `input_size`
asymmetrically. **A new test closing this gap is a pure regression-pinning
addition — zero pixel movement, zero identity movement, zero contract
impact** — the same shape as Batch 27's R26 Tier 2 (option a): it does not
fix the predicate, it proves the predicate's current behaviour fires on
real output, so a future "quiet" change to `crop_face`'s margin or to
`resize`'s predicate cannot silently make the scenario unreachable (or
newly reachable in a different way) without a named test noticing.

### 3.4 Full engineering closure path

Change the predicate so an axis being enlarged never receives `INTER_AREA`
— e.g., choose interpolation **per axis** (two single-axis resizes instead
of one `cv2.resize` call, or an `and`-style predicate combined with a
per-axis method selection). **Not achievable without moving pixels**: §3.2
proves the current, "wrong" behaviour is genuinely exercised by the shipped
pipeline today, so any predicate change is a real behavioural change on
live inputs, not a no-op tidy-up. Rides the same `.f.`-moving batch as
R23's border-clip fix — `FaceSelectionPolicy.interpolation_rule`'s *value*
(not just whether it's present) would need to change, exactly as
`docs/CROP_GEOMETRY.md`'s own superseding note already anticipated ("R28's
fix will change this string and move the `.f.` segment again, and rides the
next batch that already moves it").

### 3.5 Files affected (informational — not touched by this design, not recommended for this batch)

`inference/preprocess.py` (`resize`, and `FaceSelectionPolicy.
interpolation_rule`'s value), the same `tests/test_crop_identity.py` /
`tests/test_crop_geometry.py` surface as R23's eventual fix (bundled, per
§4.3), `docs/CROP_GEOMETRY.md`.

### 3.6 Version / migration impact (informational — for the eventual fix)

Identical profile to R23's eventual fix (§2.6) — same `.f.` segment, same
batch, same cache-generation cost, same new R31-era witness consideration.
Not a separate cost if bundled with R23 (§4.3); a second, redundant
`.f.`-moving batch if done alone.

---

## 4. Cross-cutting determinations

### 4.1 Which parts remain applicable

All of them. Nothing in R23 or R28 has been resolved, superseded, or made
moot by any batch landed since `docs/CROP_GEOMETRY.md` was written — the
underlying code (`crop_face`, `resize`, `FaceSelectionPolicy`) is
byte-identical, and the underlying constraint (no real model) is unchanged.

### 4.2 Which parts are already obsolete

**None.** Checked explicitly, not assumed: R25 (the identity-coverage
prerequisite) is closed, but that was already true before this session and
is a *dependency* resolving, not R23/R28 themselves becoming obsolete — the
geometry defects and the interpolation defect are exactly as open as
`docs/CROP_GEOMETRY.md` left them.

### 4.3 Which parts are blocked by SCRFD, video-face-stage, or real-model work

| Part | Blocked by | Why |
|---|---|---|
| R23 occupancy collapse (free-aspect boxes) | **SCRFD** | Haar structurally cannot emit a free-aspect box (`\|w−h\| ≤ 1`, re-verified). Literally unreachable without a new detector. |
| R23 border-clip distortion (the fix, Option E/F) | **Real-model work** | Reachable and live *today*; the **fix** is what's blocked — every option changes real pixels with no real model to say whether the change is an improvement. Not a detector question. |
| R23 left/top mis-framing, corner scale (the fix) | **Real-model work** | Same reasoning — same code path, same fix (Option E/F), same blocker. |
| R28 (the fix) | **Real-model work** | §3.2's empirical finding: the defect is live and reachable today (via `crop_face`'s border-clip mechanism, same root cause as R23), and its fix moves pixels on that same live path. |
| R24 (video squash) | **Video-face-stage** | Out of this document's scope (user's explicit R23/R28 framing) — named here only as the boundary: `docs/CROP_GEOMETRY.md` §7 ties R24's fix to a video face stage that does not exist, unrelated to R23/R28's own blockers. |

**None of R23's or R28's *pixel* fixes are closeable now.** Both are
blocked by the same underlying reason — no real model — which is a
different, and in this codebase's own history a *harder*, blocker than
"needs a new dependency" (SCRFD is a scoped integration; "a real, trained
model with a validated benchmark" is a multi-week-to-month program per
`docs/PROJECT_ROADMAP.md` §3c, already established and unaffected by
anything re-derived here).

### 4.4 Which parts can be closed now

Two, both zero-pixel-risk regression-test additions, both newly identified
or newly justified by this session's re-derivation rather than carried
over:

1. **A test proving R28's mixed-case defect is reachable through real
   `crop_face()` output** (§3.3) — the gap this session's empirical
   measurement (§3.2) found. Not previously proposed in any document.
2. **Nothing further for R23** — its documentation half is genuinely
   complete (§2.3), and its existing test suite (`tests/test_crop_geometry.py`,
   23 tests from Batch 15 alone plus Batch 22 additions) already covers the
   border-clip/mis-framing/corner-scale/occupancy split exhaustively
   (70,000-case brute force, re-cited not re-run — re-running it was judged
   unnecessary this session since the code under test is byte-identical to
   when it was last run, the same reasoning Batch 20 used to cite Batch 8's
   measurements rather than re-executing them).

This is a smaller "closeable now" set than Batch 27's R16+R26 (which had a
completion gap on R16's side and a three-tier split on R26's side) —
**R23/R28 are, correctly, mostly-closed-already-or-blocked**, not a
similarly-sized cheap batch.

---

## 5. Dependency analysis

Re-derived by checking what each of the five most recent batches actually
touched, not assumed unrelated:

| Batch | Touches `inference/preprocess.py` / `FaceSelectionPolicy` / crop geometry? | Basis |
|---|---|---|
| N5 (dependency lock) | No | `pyproject.toml`, `uv.lock`, `.python-version` only |
| OR1-b (worker composition root) | No | `backend/worker_main.py`, `workers/runner.py`, `backend/config.py`'s `Settings` class |
| R30 (image decode identity) | No | `inference/decode_env.py` only — confirmed in `docs/R30_BUNDLE_DESIGN.md` §1.4 that this module has zero code-path overlap with `inference/preprocess.py`'s pixel functions |
| R31 (calibration witness) | No direct touch, **but adds a dependency** | `inference/models.py` (`pre_calibration_identity`), `evaluation/calibration_protocol.py` — reads `FaceSelectionPolicy`'s fingerprint (via `.f.`) but does not modify anything R23/R28 would touch. The dependency is one-directional: a future `.f.`-moving R23/R28 batch must consider witness compatibility; R23/R28 in their current (open, unfixed) state impose nothing on R31. |
| Batch 27 (R16+R26) | No | `contracts/evidence.py`, `contracts/common.py`, one new test file — confirmed zero reference to `crop_face`/`resize`/`FaceSelectionPolicy` in that batch's diff |

**Conclusion: R23/R28 have exactly one live dependency relationship with
anything landed since `docs/CROP_GEOMETRY.md`** — R31's witness-digest
scope, a consideration for the *eventual* fix, not a blocker on the
regression tests recommended now (§4.4), which touch no identity-affecting
code.

**R23/R28's own dependency, unchanged:** both gated on "a real model exists
to validate a pixel change against" for their fixes; R23's occupancy half
additionally gated on SCRFD specifically.

---

## 6. Exact file list (for the two closeable-now regression tests)

| File | Change |
|---|---|
| New: `tests/test_crop_geometry_reachability.py` (or an addition to the existing `tests/test_crop_identity.py`, §8 below discusses the choice) | R28's real-`crop_face()`-reachability test (§3.3, §7 test plan) |
| `tests/run_all.py` | `+` one import, `+` one `SUITES` row, if a new file is chosen |
| `REPRODUCIBILITY_AUDIT.md` | R28 row: append the reachability finding (does **not** change "Open — deferred," only strengthens the evidence cited) |
| `docs/PROJECT_ROADMAP.md` | Phase 5 row: note that the reachability gap is closed, pixel fix still deferred |

**Not touched, listed to be explicit about scope discipline (matches the
user's explicit "do not write code" for *this* task, and would remain
untouched even at implementation time per §4.3/§4.4):**
`inference/preprocess.py`, `FaceSelectionPolicy`'s field values,
`backend/config.py`'s `STUB_ENGINE_VERSION` literal, any `.f.`/`.d.`
identity code, `contracts/*`, anything SCRFD- or video-face-stage-scoped.

---

## 7. Fail-before / pass-after test plan (for the one recommended regression test)

Matches the "permanent demonstration, not a true fail-before pair" pattern
Batch 27 used for R26 Tier 2 — this defect is being *proven*, not fixed, so
there is no "pass-after" in the sense of behaviour changing.

| # | Test | Assertion |
|---|---|---|
| D1 | `test_a_real_crop_produces_the_mixed_case_resize_predicate_objects_to` | Using the real `pre.crop_face()` (not a scripted double) with a large, near-border `FaceBox` under the shipped `default_face_selection()` policy, assert the resulting crop's two dimensions straddle `input_size` (one `>`, one `<`) — reproducing §3.2's measurement as a named, permanent, executable proof rather than a one-off calculation in this design document. |
| D2 | `test_the_mixed_case_crop_gets_area_interpolation_on_its_upsampled_axis` | Feed D1's crop through the real `pre.resize()` and confirm it takes the `INTER_AREA` branch (per `FaceSelectionPolicy.interpolation_rule`'s own named predicate) — i.e., that the *label* and the *live, reachable behaviour* agree, closing the gap `test_the_interpolation_rule_name_matches_resize` (an isolated, scripted-array test) does not: this one is fed the pipeline's own real output. |
| D3 | `test_the_defect_is_documented_as_live_not_hypothetical` | Doc-linkage check (the `test_r29_is_tracked_in_the_reproducibility_audit` / Batch 27's `test_the_clamp_gap_is_tracked_in_the_reproducibility_audit` pattern): `REPRODUCIBILITY_AUDIT.md`'s R28 row, after the doc update in §6, states the defect is reachable via real pipeline output, not only via a scripted example — so a future reader cannot mistake this for a purely theoretical finding. |

**No FB (fail-before) tests in the traditional sense** — nothing is being
fixed, so there is no pre-change/post-change behavioural pair. This mirrors
R28's own existing `interpolation_rule` treatment (named and pinned, not
fixed) and Batch 27's R26 Tier 2 precedent exactly.

**Explicitly not tested, correctly:** any claim about what SCRFD's actual
box sizes/positions will be (no detector exists to measure), and any claim
about whether the mixed-case interpolation choice measurably harms a
verdict (no real model exists to score with).

---

## 8. Realistic effort estimate

| Item | Effort | Basis |
|---|---|---|
| The one recommended regression suite (§4.4, §7) | **XS** — well under a day | Three small tests, no production code change, no collateral pins (confirmed: no version constant, no identity, no contract field is touched — §2.6/§3.6 apply only to the *eventual* fix, not this addition) |
| Everything else in R23/R28 (the actual pixel fixes) | **Not estimable as "the next batch"** — re-confirmed, not newly discovered | `docs/PROJECT_ROADMAP.md` §4's own estimate ("R23 border-clip + R28 bundle... Medium-High... expect the same [adversarial-review] treatment") stands; this document adds no new information that would make the fix schedulable, only sharper information about *why* it still is not (§4.3's real-model-work framing, more precise than "gated on SCRFD or a decision") |

**Total for what this document recommends actually doing now: XS, under a
day, zero version churn, zero migration.** This is a materially smaller
scope than Phase 5's own roadmap entry implied ("Medium... one more
`.f.`-moving batch") — that estimate was for the *fix*, which this
re-derivation confirms should still not be scheduled; the only thing this
document recommends implementing now is smaller and cheaper than the
roadmap's own Phase 5 sizing suggested.

---

## 9. Recommendation

1. **Implement the one regression suite** (§4.4, §6, §7) — proves R28's
   defect is live and real-pipeline-reachable, not scripted-only. Zero
   pixel risk, zero version churn.
2. **Do not implement any R23 or R28 pixel fix.** Both remain blocked —
   R23's occupancy half by SCRFD, R23's border-clip half and all of R28 by
   the absence of a real model to validate against. This is a re-derived
   confirmation of the standing recommendation, not a new finding requiring
   further design work.
3. **When the fix is eventually scheduled** (with SCRFD, or once a real
   benchmark corpus exists — whichever comes first, per
   `docs/PROJECT_ROADMAP.md` §6/§3c), bundle R23's border-clip fix and
   R28's predicate fix into one `.f.`-moving batch, per §2.6/§3.6/§3.4's
   shared-cost reasoning — unchanged from the existing roadmap's own
   bundling recommendation, re-confirmed rather than revised.
4. **Carry forward the R31-era witness consideration** (§2.6) into
   whichever future design document scopes the actual fix — it did not
   exist when `docs/CROP_GEOMETRY.md` was written and should not be
   silently inherited as "already covered."

---

## 10. Gate

**Design status: the regression-test addition (§4.4/§6/§7) is ready for
implementation, pending approval, per instruction — no code was written in
this task.** No blocking defects were found in re-deriving R23/R28's status
from source a second time. The standing "do not fix before a real model
exists" recommendation is **re-confirmed, not merely repeated** — checked
against current source, current test coverage, and every batch landed since
the recommendation was written, none of which weakens it. The one
substantive addition this re-derivation makes to the record is empirical
(§3.2's real, not scripted, reachability proof for R28), and the one
substantive precision correction is classificatory (§4.3's three-way
blocked-by split, replacing the two-way "SCRFD vs. none" framing that
under-described R23's border-clip half and R28).

**Stop here — no implementation until explicit approval, per instruction.**

**Status (2026-10-17): the zero-pixel-risk half is implemented.** R28's
reachability is pinned by `tests/test_crop_identity.py`
(`test_a_real_crop_produces_the_mixed_case_resize_predicate_objects_to` and two
companions, including `test_r28_is_documented_as_live_not_hypothetical`). The
pixel-moving fixes for R23's border clip and R28's predicate remain deferred,
as this document recommends. The gate line above is the design-time record.
