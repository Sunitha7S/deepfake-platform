# SCRFD Phase 2 — Specification Resolution and Implementation Plan

**Type:** design/specification resolution. No code in this task.
**Follows:** `docs/SCRFD_PROGRAM_DESIGN.md` (the program), `docs/
SCRFD_PHASE2_DESIGN.md` (Phase 2's own re-derivation, which itself corrected
the program document's `.f.`-impact claim), `docs/SCRFD_ASSET_AUDIT.md` (the
asset/license audit). **Supersedes** `docs/SCRFD_PHASE2_DESIGN.md` §2.1/§2.2
and §3.3's open question wherever this document's re-derivation narrows or
corrects them, flagged explicitly at each point — that document remains the
authoritative record of *how the `.f.`-movement finding was first derived*;
this document is authoritative for *what actually ships*. **Method:** every
open decision re-derived from current source this session (`git status`
re-checked first — unchanged since the asset audit, no concurrent-session
drift), not accepted from the prior three documents' own conclusions,
matching the standard `docs/R31_SPEC_RESOLUTION.md` already set for a
resolution document in this project. Two new primary-source/source-code
facts were found this session that none of the three prior documents had
(Decision 3, Decision 4) — reported as new, not folded in silently.

---

## Decision 1 — SCRFD variant selection

### Re-derivation

`docs/SCRFD_ASSET_AUDIT.md` §5.1 recommended `SCRFD_10G_KPS` (fallback
`SCRFD_2.5G_KPS`) on two grounds: landmark support (needed for `docs/
CROP_GEOMETRY.md`'s Option F destination) and unmeasured-but-plausible
latency headroom. Re-checked against this session's own reasoning, not
re-derived from new upstream facts (none were needed — the audit's own
primary-source table, §1.2 of that document, is not in question): the
10G-vs-2.5G choice is explicitly **latency-contingent** on a measurement
(Phase 3's R13 acceptance-criterion run, `docs/SCRFD_PHASE2_DESIGN.md`
§2.8) that has not happened yet. Treating `SCRFD_10G_KPS` as a **final**
choice today would overstate what is actually known.

### Resolution

**Ratified, with the conditionality preserved explicitly rather than
dropped:** `SCRFD_10G_KPS` is the **default asset fetched and tested
against for Phase 2's own vertical slice** (unit tests, §2.7 of `docs/
SCRFD_PHASE2_DESIGN.md`, and Phase 3's R13 measurement) — not a Phase 4
production commitment. Phase 3's own measurement is the actual gate for
whether 10G or 2.5G ships as `default_face_selection()`'s eventual SCRFD
default; nothing in this resolution pre-empts that. This matters concretely
for the file list (§Implementation Plan, below): the fetch script and
tests target one specific asset, and re-running Phase 3 against a second
variant later is a parameter change to an existing script, not new
infrastructure.

---

## Decision 2 — Asset distribution strategy

### Re-derivation

`docs/SCRFD_ASSET_AUDIT.md` §3/§5.2 recommended fetching from Family B's
official GitHub Release (`github.com/deepinsight/insightface/releases/tag/
model-zoo`) rather than Family A's OneDrive/GDrive-hosted `.pth` checkpoints,
reasoning that Family B needs no mmdetection/mmcv/PyTorch conversion
toolchain. Re-verified this session against the actual release asset list
(`api.github.com/repos/deepinsight/insightface/releases/tags/model-zoo`,
fetched live): `buffalo_l.zip` (288.62 MB, the only "Auto"-downloadable
pack) bundles a `SCRFD-10GF` detector, matching Decision 1's chosen variant.
No standalone, single-file `SCRFD-10GF`-only ONNX asset exists on this
release (the smallest pack containing a 10G detector is the full
`buffalo_l.zip`, not `buffalo_sc.zip`'s 500MF-only pack) — the fetch step
must download the multi-model archive and extract one file, discarding the
rest, exactly as the asset audit already flagged (§1.3) but did not
size precisely for the *specific* variant Decision 1 now ratifies.

### Resolution

**Ratified.** Fetch `buffalo_l.zip` from the official GitHub Release,
extract the detection-model member (community-convention name `det_10g.onnx`
— still **unverified** against an actually-unzipped archive this session,
per `docs/SCRFD_ASSET_AUDIT.md` §7 item 2, and this document does not
resolve that verification gap either; it is a concrete, cheap, first-step
check for whoever implements Phase 0), and discard the recognition/
alignment/attribute members the platform has no use for. This is a
one-time, documented, human-run (or scripted-but-reviewed) step — not a
runtime dependency of the shipped pipeline in any phase this program has
designed so far (Phase 2 ships nothing to production, per `docs/
SCRFD_PHASE2_DESIGN.md` §2.7 test H).

---

## Decision 3 — SHA verification strategy

### Re-derivation — a new primary-source check this session, not in any prior document

Neither `docs/SCRFD_PHASE2_DESIGN.md` nor `docs/SCRFD_ASSET_AUDIT.md` checked
whether GitHub's Releases API publishes an independently-verifiable
checksum for the chosen asset. Checked live this session:
`api.github.com/repos/deepinsight/insightface/releases/tags/model-zoo`'s
asset object for `buffalo_l.zip` (and `buffalo_sc.zip`, cross-checked) —
`content_type`, `state`, `size` are present; **no `digest` field is
present**. GitHub does populate a `digest` (`sha256:...`) field for release
assets uploaded through certain newer upload paths, but this release's
assets predate that and carry none. **There is no upstream-published
checksum to verify against for this specific asset.**

This is a materially different situation from `cascade_digest()`'s own
precedent (`inference/preprocess.py:388-411`): the Haar cascade's bytes ship
inside `opencv-python==4.8.1.78`, a wheel whose own hash is transitively
pinned by `pyproject.toml`'s `[tool.uv].override-dependencies` and `uv.lock`
(N5's own guarantee) — an independent, cryptographically-anchored chain from
PyPI down to the bytes `cascade_digest()` hashes. **No equivalent chain
exists for the SCRFD asset.** This is a real, narrower guarantee than the
project's existing asset-verification story, and this document states that
plainly rather than reusing `cascade_digest()`'s language ("content hash of
the [asset] this build actually loads") as if it carried the same assurance.

### Resolution

**Trust-on-first-use (TOFU), explicitly, not "verified against upstream":**

1. On first fetch, a human downloads `buffalo_l.zip` from the immutable,
   numeric release-asset URL (`browser_download_url`, tied to the asset's
   own id — `542728263` for `buffalo_sc.zip` in this session's fetch,
   re-derive the `buffalo_l.zip` id at implementation time), inspects it is
   the expected size (288.62 MB, this session's measurement) and content
   (a valid zip containing the expected detector file), and **records its
   own SHA-256 in `docs/SCRFD_ASSET.md`** (the file `docs/
   SCRFD_PHASE2_DESIGN.md` §3.1 already named as a placeholder) alongside
   the extracted detector file's own SHA-256 — both hashes, not just one,
   since `scrfd_asset_digest()` (per `docs/SCRFD_PHASE2_DESIGN.md` §2.4)
   hashes the **extracted** file, not the zip.
2. `scrfd_asset_digest()` — a `sha256(file bytes)[:16]` free function,
   `@lru_cache`'d, mirroring `cascade_digest()`'s exact shape
   (`inference/preprocess.py:388-411`) — verifies the **locally present**
   file matches what was recorded at step 1 was fetched. It **detects drift
   from what this project itself recorded**, not from an independent
   upstream authority.
3. This asymmetry (protects against re-fetch corruption/tampering *after*
   the fact, does not protect against a compromised or already-swapped
   asset *at* first fetch) is recorded as an explicit, named residual risk
   (§Risk register) — not silently accepted the way an unstated assumption
   would be.

---

## Decision 4 — User-provided model override behavior

### Re-derivation — a new architectural check this session

Neither prior document traced where a configurable model path would live
in this codebase's existing dependency-direction rules. Checked live this
session: `grep -rn "^from backend\|^import backend" inference/` and the
reverse (`backend` importing `inference`) — **zero edges either direction**
at module scope. `CASCADE_PATH` (`inference/preprocess.py:325`,
`cv2.data.haarcascades + "haarcascade_frontalface_default.xml"`) is a
**plain module-level constant**, not read from any `Settings`/config object
— because the Haar asset ships inside the `opencv-python` wheel, this
project has never before needed a *configurable* external model path, so
there is no existing `FaceSelectionPolicy`-field or `Settings`-field
precedent to copy verbatim.

The closest real precedent is OR1/OR1-b's own solved version of exactly
this class of problem: `workers/runner.py:457`, `configured =
configured_model_version or os.environ.get("DFP_MODEL_VERSION")` — an
environment variable read directly at the point of need, specifically
**because** `workers` cannot import `backend` (the same DAG-acyclicity rule
that keeps `inference` and `backend` unconnected applies here too,
`workers/runner.py:414`, *"``workers`` may not import [``backend``]"*).
`ModelSpec.weights_uri` (`inference/models.py`, read at `:561-563`,
`:580-583`) is the second, complementary precedent: a file **location**
lives on the spec but is **excluded** from `weights_identity`'s fingerprint
dict (`:288-293`, only `branch_weights`/`input_size` are fingerprinted) —
only the loaded content's derived facts are identity-bearing, never the
path used to find it.

### Resolution

**Both precedents combined, neither contradicted:**

1. **The model path is not a `FaceSelectionPolicy` field.** It never
   reaches `.identity` — mirroring `weights_uri`'s exclusion, and avoiding
   inventing a new "location vs. content" ambiguity this codebase has
   already resolved once. Only `detector_asset` (the content digest, per
   Decision 3) is identity-bearing, exactly as `detector_asset` already is
   for Haar.
2. **Override mechanism: an environment variable, `DFP_SCRFD_MODEL_PATH`**,
   read via `os.environ.get(...)` at whichever call site constructs
   `SCRFDDetector`/an SCRFD-backed `FaceSelectionPolicy` — mirroring
   `$DFP_MODEL_VERSION`/`$DFP_SKIP_VERSION_ALIGNMENT`'s naming convention
   and read pattern exactly. **Not** a `backend.config.Settings` field read
   by `inference/preprocess.py` — that would create the import edge this
   codebase's dependency-graph test (`test_dependency_graph_stays_acyclic`)
   exists to forbid.
3. **Scope narrowing this document contributes, not present in `docs/
   SCRFD_PHASE2_DESIGN.md`:** because Phase 2 ships nothing to production
   (`default_face_selection()` stays Haar throughout, per that document's
   own test H), **the override mechanism needs no production
   composition-root change in Phase 2.** `tests/scrfd_detector_study.py`
   (§2.8 of `docs/SCRFD_PHASE2_DESIGN.md`) reads `DFP_SCRFD_MODEL_PATH`
   directly, falling back to the default fetched location (Decision 2), and
   that is sufficient for Phase 2's entire scope. Wiring the same
   environment variable into `backend/app.py`'s or `workers/worker_main.py`'s
   composition root is **Phase 4 work**, deferred along with the rest of
   the real cutover — stated explicitly so it is not silently assumed into
   Phase 2's file list.

---

## Decision 5 — `CPUExecutionProvider` pinning requirement

### Re-derivation

`docs/SCRFD_ASSET_AUDIT.md` §4.2 established the risk (an unpinned
`onnxruntime` provider selection is a latent, R15-shaped determinism
hazard) and recommended passing `providers=["CPUExecutionProvider"]`
explicitly, but left one question open: *"Whether the chosen provider
itself belongs in `FaceSelectionPolicy.identity`... is a design question
for Phase 2's own implementation, not this audit."* This document resolves
it, applying Batch 9's own stated criterion for what belongs in an
identity segment (`docs/CROP_GEOMETRY.md` §10, restated in `docs/
SCRFD_PHASE2_DESIGN.md`'s own architecture discussion): *"is this code the
deployment ships, or environment it inherits? Would an operator diffing
two versions want to see this separately?"*

Applied here: this project has no GPU dependency anywhere (`docs/
SCRFD_PROGRAM_DESIGN.md` §3.1, re-confirmed unchanged), pins the plain
`onnxruntime` package, never `onnxruntime-gpu` (`pyproject.toml:42`), and
this resolution's own design (below) hardcodes exactly one provider,
unconditionally, in source — it is never selected by inspecting the
environment (`onnxruntime.get_available_providers()` is never called for
this purpose). A single, unconditional, source-level constant is not a
*configuration* an operator could vary or diff between deployments — it is
implementation, the same category `HaarDetector`'s own internal use of
`cv2.COLOR_RGB2GRAY` (`inference/preprocess.py:921`) already sits in,
which nothing treats as policy.

### Resolution

**`SCRFDDetector` hardcodes `providers=["CPUExecutionProvider"]` in its
`onnxruntime.InferenceSession(...)` construction, unconditionally. This is
not a `FaceSelectionPolicy` field and has zero identity impact — it never
reaches `.f.`.** `onnxruntime.get_available_providers()` must not be called
in this code path at all (its result must never influence which provider is
selected); a test asserting this (§Test files, below) is mandatory, not
optional, given how easy "just use the default" is to fall back to
accidentally.

**Related, explicitly deferred, not part of Phase 2:** *whether the
installed `onnxruntime` build's own available-providers set is an
environment-completeness fact belonging in `.d.`* (the R30/R27 pattern,
applied to a new library) is a real, adjacent question this resolution
does not answer — it extends `docs/SCRFD_PHASE2_DESIGN.md` §10 item 7's
candidate finding (N10, unpinned `onnxruntime` version) rather than
resolving it, and is out of scope because Phase 2 ships no `.d.`-reachable
code path (§Impact axes, below).

---

## Decision 6 — Detector configuration fields added to `FaceSelectionPolicy`

### Re-derivation

`docs/SCRFD_PHASE2_DESIGN.md` §2.1 proposed exactly two new fields,
`confidence_threshold: float | None` and `nms_iou_threshold: float | None`,
with `None` (not `NaN`) as the Haar-inert sentinel — already justified from
`contracts/version.py`'s `_canonical`/dataclass-equality semantics in that
document (§1.3). Re-checked this session against Decision 4's finding: the
model **path** is confirmed **not** a third field (it was never proposed as
one, but this document makes the exclusion explicit and reasoned, closing
a gap the prior document left implicit). Re-checked against Decision 5: the
execution **provider** is confirmed **not** a fourth field, for the same
reason. **The field count is therefore exactly two, narrower than a naive
reading of "SCRFD needs configuration" might produce**, and this narrowing
is this document's own contribution.

### Resolution

**Ratified at exactly two fields:**

```python
confidence_threshold: float | None
"""SCRFD's own detection-score floor. None under Haar."""

nms_iou_threshold: float | None
"""SCRFD's NMS IoU floor. None under Haar."""
```

`default_face_selection()` (`inference/preprocess.py:611-622`) gains
`confidence_threshold=None, nms_iou_threshold=None`. `identity`, `to_dict`,
`differences` each gain both keys unconditionally (§Decision 7 proves this
is the only sound way to add them). `tests/
test_face_selection_identity.py`'s `alternatives` dict (`:290-302`) gains
two entries with real, non-`None` values.

---

## Decision 7 — Identity/fingerprint implications: is `.f.` movement unavoidable?

This is the objective this document treats with the most scrutiny, per the
task's own explicit instruction to determine this rather than restate `docs/
SCRFD_PHASE2_DESIGN.md` §2.2/§5's prior conclusion ("yes, unavoidable").
Four concrete alternative designs were constructed and checked against
source this session — not merely asserted — to see whether any of them
actually preserves `.f.`'s literal value while still giving `SCRFDDetector`
real, usable configuration.

### The mathematical fact underneath the question

`contracts/version.py:246-257`, `fingerprint()`: `json.dumps(_canonical(dict
(parts)), sort_keys=True, ...)` then SHA-256. **Any change to the *set of
keys* in the dict passed to `fingerprint()` changes the JSON string for
every possible set of values that dict can hold**, including the exact
values the shipped Haar policy already uses — this is a property of
JSON serialization plus hashing, not an assumption. `FaceSelectionPolicy.
identity` (`:542-555`) passes a **source-literal dict** to `fingerprint()`.
Therefore: **if and only if a new key is added to that dict literal,
`.identity`'s value changes for every existing `FaceSelectionPolicy`
instance, unconditionally.** This is the fact every alternative below has
to either accept or find a way around.

### Alternative 1 — add the fields, but exclude them from the `identity` dict

Technically preserves `.f.`'s value exactly (the dict literal is
unchanged). **Rejected:** `FaceSelectionPolicy.identity`'s own docstring
(`:531-541`) states its design invariant in its own words: *"unlike the
decode identity, nothing here is excluded on the strength of a
measurement: each of these was either measured to change a verdict, or is
structurally certain to once the detector is replaced."* `confidence_
threshold`/`nms_iou_threshold` are exactly the second category (structurally
certain to affect which faces are detected and therefore which crop the
model sees) — excluding them from the fingerprint is **R17, reintroduced
for a second detector**: a verdict-affecting knob the cache key, benchmark
comparison, and calibration witness cannot see. R17 is the single most
foundational fix in this codebase's own history (`inference/
preprocess.py:436-446`); reintroducing its exact shape to save one version
bump is not a trade this project's own precedent supports anywhere.

### Alternative 2 — repurpose existing Haar-shaped fields for SCRFD's meaning

E.g., let `scale_factor` mean "confidence threshold" when `detector` names
an SCRFD variant. Preserves `.f.`'s value (no new keys). **Rejected:**
`docs/SCRFD_PROGRAM_DESIGN.md` §2.1 already named and implicitly declined
this ("repurposing fields to mean something they don't say"); this
document makes the rejection explicit and grounds it in a second,
concrete precedent — `docs/R31_SPEC_RESOLUTION.md` Decision 1 rejected two
candidate field names (`model_config_identity`, `model_identity`) for
exactly this class of hazard: a name that means one thing in one context
and a different thing in another silently corrupts `to_dict()`'s own
stated purpose (`inference/preprocess.py:558`, *"Forensic record"*) — a
forensic record whose field names lie about what they hold is worse than
one that is simply longer.

### Alternative 3 — encode SCRFD's parameters into the `detector_asset` string

Preserves `.f.`'s value for the shipped Haar policy (the string stays
`cascade_digest()`'s output, unconditionally). **Rejected:** `detector_
asset`'s own docstring (`:465-467`) defines it as *"Content digest of the
detector's weights or cascade"* — a pure hash, verified byte-for-byte
against a downloaded file (Decision 3's whole verification chain depends on
this). Smuggling `|conf=0.5|nms=0.4` into that string breaks the hash
comparison itself (the string would no longer equal `scrfd_asset_digest()`'s
output), conflates two concerns this codebase's architecture keeps
deliberately separate everywhere else (`detector` = algorithm identity,
`detector_asset` = content digest — the same two-fields-not-one split
`ModelSpec.weights_identity`/`ModelSpec.calibration_fingerprint` use for an
analogous reason), and still requires the dataclass to expose *typed*
values for `SCRFDDetector.detect()` to read — it does not actually
eliminate the field-count problem, only hides it inside a string that a
future reader has to know to re-parse.

### Alternative 4 — a second, parallel identity segment outside `.f.`

Preserves `.f.`'s value completely (`FaceSelectionPolicy` itself untouched).
**Rejected, on direct precedent:** this is the exact "two identities
instead of one" architecture Batch 23 (R29) already considered and
rejected for video selection — `docs/SCRFD_PHASE2_DESIGN.md` §2.2 already
named this as "option 2, rejected" citing the same B23/R29 precedent; this
document re-confirms the rejection stands rather than re-litigating a
closed decision, and notes the precedent is now doubly established (B23
and this resolution both declining a second segment for the same reason:
one composed `.f.` is the single source of truth `compose_model_version`/
`parse_model_version` are built around, and a second, parallel
identity-bearing structure recreates exactly the two-literals hazard OR1's
own worker/backend split had to work around for an unrelated reason).

### Alternative 5 (new this session, not considered by any prior document) — conditional/sparse inclusion

Include `confidence_threshold`/`nms_iou_threshold` in the `identity` dict
**only when non-`None`** (mirroring `CalibrationWitness.compute_witness`'s
own `SCHEMA_VERSIONS_WITH_DECODE`-gated conditional inclusion,
`evaluation/calibration_protocol.py`). This would preserve `.f.`'s exact
value for every existing Haar policy (the dict literal, evaluated at
`None`, contributes no new key) **while still giving a real, non-`None`
SCRFD configuration a complete fingerprint** — the closest any alternative
comes to actually resolving the tension rather than trading it away.

**Rejected, on a different and more subtle ground than 1–4:** the witness's
own conditional inclusion is gated on **`schema_version`** — an explicit,
version-pinned discriminator, itself part of the identity contract, checked
by dedicated tests (`test_a_1_2_0_payload_cannot_smuggle_a_pre_calibration_
identity`-style guards, `docs/R31_SPEC_RESOLUTION.md` §Decision-1). A
`None`-check discriminator is not equivalent: it makes `identity`'s own
dict-construction **conditional on an instance's field values**, a strictly
more complex property than every field in this class currently has
(`:542-555`'s dict is presently a flat, unconditional literal — confirmed
by re-reading it this session). This directly weakens `test_every_policy_
field_moves_the_identity`'s own stated guarantee (`tests/
test_face_selection_identity.py:283-289`, *"No field is decoration... a
field added later without being put into `identity` fails here instead of
silently joining the set of things nothing records"*) from an
**unconditional** property (every field, every time) to a **conditional**
one (every field, except when its value happens to be the sentinel) — and
a `None`-check is exactly the kind of small, easy-to-get-subtly-wrong
condition (an `==` typo, a future refactor that changes the sentinel, a
field whose "inert" value is legitimately `0.0` rather than absent) that
this project's own history (R14's stable-sort tie, R19's metric mismatch,
R25's `or`-predicate, all *small* logic errors with large, silent
consequences) treats as a real, not theoretical, risk category. **The
one-time cost this resolution accepts (a single `.f.` move, one cache
generation, a bounded, already-precedented event, paid four times before —
B20/B22/B23/B26) is smaller than the ongoing cost Alternative 5 would add
(a permanently more complex, conditionally-testable identity property, for
the life of this codebase).**

### Determination

1. **Is `.f.` movement unavoidable?** **Yes** — provably, from
   `fingerprint()`'s own mechanics, given that `FaceSelectionPolicy` must
   gain real, typed, non-decorative configuration fields for `SCRFDDetector`
   to be usable without reintroducing R17.
2. **Can any alternative design preserve `.f.`?** **Yes, mechanically —
   five distinct methods exist and were checked, and at least one
   (Alternative 5) achieves it without literally reintroducing R17's exact
   shape.** This document does not claim no alternative exists; it claims
   every alternative found trades away something else load-bearing.
3. **Would preserving `.f.` create hidden configuration risk?** **Yes, in
   every case, and the specific risk differs by alternative:** Alternatives
   1 (a verdict-affecting knob outside every cache key/benchmark/witness —
   R17's exact defect), 2 (a forensic record whose field names lie about
   their contents), 3 (two conflated concerns, breaking asset-hash
   verification), and 4 (a second identity-bearing structure, the "two
   literals" hazard this project has already rejected once). Alternative 5
   is the most serious candidate and creates the most subtle risk: a
   conditional identity-construction path with a strictly weaker,
   value-dependent version of this class's current unconditional
   completeness guarantee.

**Ratified: `.f.` moves in Phase 2, exactly as `docs/SCRFD_PHASE2_DESIGN.md`
§2.2/§5 concluded.** This resolution reaches the same answer via a wider,
independently-constructed search (five alternatives, one new to this
session) rather than by trusting the prior document's narrower one — the
"re-derive, do not assume" instruction is satisfied by the search having
been genuinely performed, not by the destination changing.

---

## Summary of the seven resolutions

| # | Decision | Resolution |
|---|---|---|
| 1 | SCRFD variant | `SCRFD_10G_KPS` — ratified as Phase 2/3's working default, explicitly contingent on Phase 3's own latency measurement for Phase 4's final choice |
| 2 | Asset distribution | Fetch `buffalo_l.zip` from the official GitHub Release (`model-zoo` tag), extract the detector member, discard the rest — never vendor, never re-host |
| 3 | SHA verification | Trust-on-first-use: record both the release asset's own SHA-256 and the extracted detector file's SHA-256 in `docs/SCRFD_ASSET.md` at first fetch; `scrfd_asset_digest()` detects drift from that recorded value, not from an independent upstream checksum (none exists — new finding) |
| 4 | User-provided override | `DFP_SCRFD_MODEL_PATH` env var, read directly (mirroring `$DFP_MODEL_VERSION`); the model path is never a `FaceSelectionPolicy` field (mirrors `ModelSpec.weights_uri`'s exclusion from `weights_identity`); no production composition-root wiring needed in Phase 2 — test-file-only |
| 5 | `CPUExecutionProvider` pinning | Hardcoded, unconditional, in `SCRFDDetector`'s session construction; **not** a `FaceSelectionPolicy` field; zero identity impact |
| 6 | New `FaceSelectionPolicy` fields | Exactly two: `confidence_threshold: float \| None`, `nms_iou_threshold: float \| None`, both `None` under Haar |
| 7 | `.f.` movement | **Unavoidable, ratified.** Five alternatives checked and rejected, each for a distinct, source-grounded reason; the one-time `.f.` move is cheaper than any of them over this codebase's lifetime |

None of the seven resolutions reopens `docs/SCRFD_PROGRAM_DESIGN.md`'s
architecture or contradicts `docs/SCRFD_ASSET_AUDIT.md`'s licensing
findings. Three of the seven (2, 4, 5) **narrow** what the prior documents
left open toward less production-file surface, not more.

---

# Final Implementation Plan

## 1. Scope, restated precisely

**In scope for Phase 2 implementation** (per this resolution and `docs/
SCRFD_PHASE2_DESIGN.md` §2, now narrowed by Decisions 4/5/6 above):

1. `inference/preprocess.py`: `FaceSelectionPolicy` +2 fields (Decision 6);
   `identity`/`to_dict`/`differences` each +2 keys, unconditional; `default_
   face_selection()` +2 `None` values; `SCRFDDetector` class (lazy
   `onnxruntime` import, hardcoded `CPUExecutionProvider`, per-thread
   session per the R15 precedent, anchor decode + NMS per §1.5 of `docs/
   SCRFD_ASSET_AUDIT.md`); `scrfd_asset_digest()` free function, mirroring
   `cascade_digest()`; `detect_faces` dispatch on `policy.detector`
   (`docs/SCRFD_PHASE2_DESIGN.md` §2.6's `if`/`elif`/`raise` shape, not a
   dict registry).
2. `backend/config.py`: `STUB_ENGINE_VERSION` literal regenerated (the `.f.`
   move, Decision 7).
3. `tests/test_face_selection_identity.py`: `alternatives` dict +2 entries
   (mandatory — the hard equality assert fails otherwise, per `docs/
   SCRFD_PHASE2_DESIGN.md` §1.2).
4. `tests/run_all.py`: +1 import, +1 `SUITES` row.
5. New test/study files (§3, below).
6. `docs/SCRFD_ASSET.md`: the fetch URL, both recorded SHA-256 values
   (Decision 3), and the verified license text (already fully populated by
   `docs/SCRFD_ASSET_AUDIT.md` §2 — this file's licensing section can be a
   direct excerpt, not new research).

**Explicitly out of scope for Phase 2** (deferred to Phase 4, restated so
no implementer assumes it belongs here): `default_face_selection()`
returning an SCRFD policy; any `backend/app.py`/`workers/worker_main.py`
composition-root wiring of `DFP_SCRFD_MODEL_PATH`; `contracts/common.py`'s
`landmarks` field actually being populated (`inference/postprocess.py:
face_track()` gains no parameter in Phase 2); `CONTRACT_VERSION`/`RUN_
SCHEMA_VERSION`/`CALIBRATION_WITNESS_VERSION`/`LOCK_VERSION`/`.d.` moves
(none — §Impact axes, below); crop-geometry changes (`docs/CROP_GEOMETRY.md`
Option E, a separate, later phase per `docs/SCRFD_PROGRAM_DESIGN.md` §5
Phase 6).

## 2. Production files changed

| File | Change |
|---|---|
| `inference/preprocess.py` | `FaceSelectionPolicy` +2 fields; `identity`/`to_dict`/`differences` +2 keys each; `default_face_selection()` +2 `None` args; new `SCRFDDetector` class; new `scrfd_asset_digest()`; `detect_faces` gains the `if policy.detector == ... elif ... else raise` dispatch (§Decision structure per `docs/SCRFD_PHASE2_DESIGN.md` §2.6, unchanged by this resolution) |
| `backend/config.py` | `STUB_ENGINE_VERSION` literal, regenerated once the new `.f.` value is computed |

**Not changed** (confirmed, restated from `docs/SCRFD_PHASE2_DESIGN.md` §3.2
and re-confirmed by Decisions 4/5 narrowing scope further): `inference/
image_pipeline.py`, `inference/video_pipeline.py`, `inference/postprocess.py`,
`contracts/*`, `evaluation/benchmark.py`, `pyproject.toml`, `uv.lock`,
`backend/app.py`, `workers/worker_main.py`, `workers/runner.py`.

## 3. Test files changed / added

| File | Change |
|---|---|
| `tests/test_face_selection_identity.py` | `alternatives` dict (`:290-302`) +2 entries with real values |
| `tests/test_scrfd_detector.py` (new) | Tests A–H per `docs/SCRFD_PHASE2_DESIGN.md` §2.7, **plus** two new tests this resolution adds: **Test I** — `test_cpu_execution_provider_is_hardcoded_not_selected` (asserts `onnxruntime.get_available_providers` is never called by `SCRFDDetector`'s construction path — Decision 5 — via `inspect.getsource` grep for the literal, the same technique `test_haar_detector_calls_the_shared_face_cascade_not_its_own` already uses for an analogous claim); **Test J** — `test_model_path_is_not_a_face_selection_policy_field` (`"model_path" not in {f.name for f in fields(FaceSelectionPolicy)}` and `"path" not in {f.name for f in fields(FaceSelectionPolicy)}` — Decision 4, a structural pin against the field-count creep this resolution deliberately avoided) |
| `tests/scrfd_detector_study.py` (new) | Phase 3's R13 measurement (`docs/SCRFD_PHASE2_DESIGN.md` §2.8); reads `DFP_SCRFD_MODEL_PATH` via `os.environ.get(...)`, falling back to the documented default fetch location (Decision 4) |
| `tests/run_all.py` | +1 import (`test_scrfd_detector`), +1 `SUITES` row, alphabetically placed |

**Not changed:** every other collateral `STUB_ENGINE_VERSION`/`.f.`-literal
pin is a **required** update (§6, below) but not a *new* test — existing
files, literal values only.

## 4. Documentation files changed / added

| File | Change |
|---|---|
| `docs/SCRFD_PHASE2_SPEC_RESOLUTION.md` | this document |
| `docs/SCRFD_ASSET.md` (new) | Fetch URL (`buffalo_l.zip`'s release-asset URL), both recorded SHA-256 values (Decision 3), the extracted member's community-convention name flagged **unverified** until Phase 0 confirms it, the verified non-commercial-research-only license text excerpted from `docs/SCRFD_ASSET_AUDIT.md` §2.2 |
| `REPRODUCIBILITY_AUDIT.md` | **Not changed by Phase 2** — R13 remains "Open," since Phase 2 only wires the detector; it is measured (Phase 3) and potentially closed (Phase 4), not this phase |
| `docs/PROJECT_ROADMAP.md` | **Not changed by this document** — that document is dated 2026-09-17, one day before the SCRFD program began, and already correctly scopes SCRFD integration as "Phase 6... a program, not a batch" (its own §7); nothing in Phase 2 contradicts that framing, so no correction is owed to it the way `docs/PROJECT_STATUS_FINAL.md` needed one for its stale test count |

## 5. Expected collateral updates

Exhaustive, re-derived by the same method `docs/R31_SPEC_RESOLUTION.md`
used (grep the current literal, list every hit):

- Every hardcoded `STUB_ENGINE_VERSION` string and every hardcoded
  `.f.e629642c` (or its containing full literal) across the test suite —
  the same collateral sweep `docs/SCRFD_PHASE2_DESIGN.md` §3.2 already
  named as "enumerate at implementation time via the same `grep -rn
  <old-.f.-value>` sweep every prior `.f.`-moving batch has run." This
  document does not re-run that grep speculatively (the new value does not
  exist until implementation computes it), but the **method** is fixed by
  this resolution: `grep -rn "e629642c"` across `tests/` and `backend/`,
  update every hit to the new value.
- `default_face_selection().identity`'s own pinned baseline
  (`tests/test_face_detector_abstraction.py:52`, `_BASELINE_FACE_SELECTION_
  IDENTITY = "251d22e9"`) — this Phase 1 test's own baseline **does not
  move**, since it captures the value immediately before Phase 1's edit,
  not the current value; **no change needed here**, stated to prevent a
  future implementer from "fixing" a test that is not broken.

## 6. Fail-before / pass-after matrix

| # | Check | Before Phase 2 | After Phase 2 |
|---|---|---|---|
| 1 | `python tests/run_all.py` | 55 suites, 1335 tests (confirmed live, unchanged since `docs/SCRFD_PHASE2_DESIGN.md` §0) | 56 suites, 1335 + (8 from Phase 1's own count already included) + `test_scrfd_detector.py`'s test count (A–J, 10 tests), **0 failures** |
| 2 | `isinstance(SCRFDDetector(path), FaceDetector)` | Does not exist | `True` |
| 3 | `default_face_selection().detector` | `"opencv-haar-frontalface-default"` | **Unchanged** |
| 4 | `default_face_selection().identity` | `"251d22e9"` | **A new value** (two new `None`-valued keys, Decision 7) — captured and pinned at implementation time |
| 5 | `STUB_ENGINE_VERSION` | `"...+c.739c3fec.g.ae232cba.f.e629642c.d.c9d4025c"` | New `.f.` segment; `.d.c9d4025c` **unchanged** |
| 6 | `onnxruntime.get_available_providers` called during `SCRFDDetector` construction? | N/A | **No** — Test I asserts this directly |
| 7 | `{f.name for f in fields(FaceSelectionPolicy)}` contains a path/location field? | N/A | **No** — Test J asserts this directly |
| 8 | Concurrent `SCRFDDetector.detect()` calls across threads, different images | N/A | Zero cross-contamination (Test E, `docs/SCRFD_PHASE2_DESIGN.md` §2.7) |
| 9 | `scrfd_asset_digest(path)` on the recorded-good file vs. a corrupted copy | N/A | Matches recorded value / detects mismatch (Decision 3's TOFU chain, Test G) |
| 10 | A real photograph, ±4 LSB noise, `SCRFDDetector` before/after | Not measurable (no second detector exists) | A concrete amplification/face-loss/box-movement figure — Phase 3's own gate, not a pass/fail here |

## 7. Impact axes — re-verified

| Axis | Current value | Moved by this resolution? |
|---|---|---|
| `CONTRACT_VERSION` | `contracts/enums.py:16`, `"1.0.0"` | **No.** No wire-shape change — confirmed again this session; nothing in Decisions 1–6 touches `contracts/`. |
| `RUN_SCHEMA_VERSION` | `evaluation/benchmark.py:108`, `"1.2.0"` | **No.** `BenchmarkRun` untouched. |
| `CALIBRATION_WITNESS_VERSION` | `evaluation/calibration_protocol.py:74`, `"1.3.0"` | **No, schema-wise** — already generic over `pre_calibration_identity` (R31), which reads `face_selection_fingerprint` automatically; the **digest value** a future witness computes will move once `.f.` moves, but zero stored witnesses exist (re-confirmed, same search performed in every prior SCRFD document) — nothing to migrate. |
| `LOCK_VERSION` | `evaluation/split_lock.py:81`, `"1.1.0"` | **No — structurally unrelated**, re-confirmed by the same reasoning `docs/SCRFD_PHASE2_DESIGN.md` §4 and `docs/SCRFD_ASSET_AUDIT.md` §6 both already gave; this document found nothing to revise it. |
| `.d.` | feeds `backend/config.py:41`'s `.d.c9d4025c` segment | **No.** Detection is not a decode-environment fact; Decision 5's provider-pinning is a hardcoded implementation choice, not an environment fact this phase records anywhere (the adjacent, deferred question — should the *available* provider set feed `.d.` someday — is explicitly not this phase's scope, per Decision 5). |
| `.f.` | feeds `backend/config.py:41`'s `.f.e629642c` segment | **Yes — ratified, Decision 7.** One cache generation, both media namespaces (N8 precedent), zero stored artifacts to migrate. |

**Net: exactly one axis moves, for a reason now checked five independent
ways (Decision 7) rather than one.**

---

## 8. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| §Decision 7's `.f.` move ships without its full collateral-literal sweep | High if it happens, mechanical to catch | Full-suite re-run is the net (§6 row 1); a missed literal is a loud failure |
| `SCRFDDetector`'s ONNX session shares unsafely across threads (R15 recurrence) | High if it happened, structurally checked | Test E, mandatory before Phase 2 is done, not deferred to Phase 3/4 |
| `onnxruntime`'s default provider-selection creeps back in via a future refactor that calls `get_available_providers()` "just to check" | Medium, ongoing | Test I is a permanent regression pin, not a one-time check |
| Family B's `buffalo_l.zip` internal member name is not actually `det_10g.onnx` (unverified, §Decision 2) | Low-Medium, schedule risk | A five-minute unzip-and-inspect step, named explicitly as the first concrete action of implementation, not assumed |
| **No independent upstream checksum exists for the fetched asset** (Decision 3, new finding) | Medium, non-mechanical | TOFU is the accepted, explicitly-named residual — the mitigation *is* naming it, not eliminating it; a human verifies the file's provenance (source URL, size, zip validity) at first fetch, once |
| A future implementer adds a `model_path`-like field to `FaceSelectionPolicy` "for convenience," reopening Decision 4/6's narrowing | Low, structurally checked | Test J is a permanent regression pin |
| Upstream license terms (`docs/SCRFD_ASSET_AUDIT.md` §2.2) change on a future model-zoo release | Low today, unbounded if unmonitored | No automated mitigation exists (stated honestly in the asset audit, §7 item 3, unchanged by this resolution) — a periodic manual re-read is recommended, not scheduled |

## 9. Rollback strategy

Identical in kind to `docs/SCRFD_PHASE2_DESIGN.md` §3.5's rollback, narrowed
by this document's own scope reduction: reverting `FaceSelectionPolicy`'s
two new fields (and therefore `.f.`) and `detect_faces`'s dispatch branch is
a **single commit revert** — nothing downstream of `.f.` is stored anywhere
in this repository (re-confirmed, §Impact axes above). Because Decision 4
keeps the model-path override entirely inside test files and Decision 5's
provider pin is a hardcoded, non-configurable constant, **there is no
composition-root wiring to revert at all** — a strictly smaller rollback
surface than `docs/SCRFD_PHASE2_DESIGN.md` §3.5 originally described, since
that document had not yet narrowed Decisions 4/5 out of production files.

## 10. Migration analysis

- **Stored `FaceSelectionPolicy`/`CalibrationWitness`/`BenchmarkRun`
  artifacts:** none exist anywhere in this repository (re-confirmed by the
  same search every prior SCRFD/R31 document has performed) — **no
  migration required for any stored value.**
- **`fixtures_out/*.json` / `schema_out/*.json` / `backend/openapi.snapshot.
  json`:** `FaceSelectionPolicy` is not a wire type; unaffected, confirmed
  by the same reasoning `docs/SCRFD_PHASE2_DESIGN.md` §3.2 already gave.
- **The SCRFD asset itself:** not a migration in this codebase's sense (no
  existing value is being changed) — it is a **first acquisition**, covered
  fully by Decisions 2/3.
- **Conclusion, matching every prior `.f.`-moving batch's own migration
  finding (B20/B22/B23/B26):** zero stored artifacts require migration; the
  entire migration cost is one cache generation, both media namespaces,
  paid at Phase 2's completion (not Phase 4 — Decision 7 establishes the
  move happens here).

## 11. Acceptance criteria for implementation completion

Implementation of Phase 2 is complete when **all** of the following hold,
each independently checkable, no exceptions:

1. `python tests/run_all.py` is green with the exact suite/test delta in
   §6 row 1 and **zero changes to any file outside §2–§4's list.**
2. `default_face_selection().detector` is still
   `"opencv-haar-frontalface-default"` (§6 row 3).
3. Tests A–J of `tests/test_scrfd_detector.py` all pass, **including** the
   two this resolution added (I, J) — not merely the eight `docs/
   SCRFD_PHASE2_DESIGN.md` §2.7 originally specified.
4. `docs/SCRFD_ASSET.md` exists and contains: the fetch URL, both recorded
   SHA-256 values, the license text, and an explicit **unverified** flag on
   the extracted member's file name until it is actually confirmed (§Risk
   register).
5. `DFP_SCRFD_MODEL_PATH` is read only by test/study files — `grep -rn
   "DFP_SCRFD_MODEL_PATH"` outside `tests/` returns nothing.
6. `grep -n "get_available_providers"` inside `inference/preprocess.py`
   returns nothing.
7. The collateral literal sweep (§5) is complete — `grep -rn "e629642c"`
   (the pre-Phase-2 value) inside `tests/`/`backend/` returns nothing after
   implementation.
8. `REPRODUCIBILITY_AUDIT.md`'s R13 row is **not** marked closed (Phase 2
   measures nothing yet — that is Phase 3).

## 12. Should implementation proceed?

**Yes — APPROVE FOR IMPLEMENTATION**, with one explicit precondition
carried forward unchanged from `docs/SCRFD_ASSET_AUDIT.md` §8 and not
weakened by anything in this document: **a human must read and
consciously accept `docs/SCRFD_ASSET_AUDIT.md` §2.2's non-commercial-
research-only restriction before Decision 2's fetch step runs against any
machine or environment this project does not fully control.** Every
engineering decision this document was asked to resolve (Decisions 1–7)
is resolved, each re-derived from source rather than inherited, with two
new findings (Decision 3's missing upstream checksum, Decision 4's DAG-
respecting override mechanism) that narrow Phase 2's production-file
surface rather than widen it. No blocking architectural defect was found
in re-deriving any of the three source documents. Scope is **narrower**
than `docs/SCRFD_PHASE2_DESIGN.md`'s own estimate (Decisions 4 and 5 remove
what would otherwise have been composition-root changes), not wider.

**No code was written to produce this document. Stop here.**
