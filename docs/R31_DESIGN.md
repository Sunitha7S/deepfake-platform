# R31 — Design: Calibration Witness Model-Config Binding

**Type:** design document only. No code in this task.
**Author's method:** every claim below re-derived from source at the cited path,
and where marked *(live)*, executed on the implementation box
(Windows / `opencv-python==4.8.1.78` / the B24-locked `uv` environment). Prior
narratives (`REPRODUCIBILITY_AUDIT.md`'s R31 row, the B21/B22 "incidental
binding" write-ups, `docs/CROP_IDENTITY.md` §"Recorded residual",
`docs/CROP_IDENTITY_B22_REVIEW.md` §6, `docs/B24_R2_DESIGN.md` §7's re-rank)
were **not** trusted as inputs to this design — each claim they make is
re-derived independently below, and where they differ from what source shows,
the difference is called out explicitly.

---

## 1. What R31 claims, and what is actually true — re-derived

### 1.1 The claim as inherited

`REPRODUCIBILITY_AUDIT.md:123`: *"`CalibrationWitness` digest omits
`effective_model_version` / `input_size` / gate / `.f.`. A temperature fitted
for one model applies to another if the two fits quantize-converge. Needs
`CALIBRATION_WITNESS_VERSION` 1.3.0 (its own batch); no real model exists to
calibrate yet."* Traced to a B21/B22 residual (`docs/CROP_IDENTITY.md` §"Recorded
residual (not closed)", `docs/CROP_IDENTITY_B22_REVIEW.md` §6) that was
narrower: it named only `input_size`, via "different pixels → different logits
→ different fitted temperature → different `quantize(temperature)` → different
digest" — i.e. the claim was that binding was *incidental* (worked by accident
of the fitted numbers moving), not that it was *absent*. B23A generalised
"`input_size`" to "`effective_model_version` / gate / `.f.`" without a fresh
re-derivation on record.

### 1.2 `CalibrationWitness.compute_witness` — the complete, current preimage *(live)*

`evaluation/calibration_protocol.py:211-277`. Executed:

```
inspect.signature(CalibrationWitness.compute_witness).parameters ==
['protocol_name', 'role', 'corpus_fingerprint', 'n_samples', 'temperature',
 'nll_before', 'nll_after', 'threshold', 'margin', 'target_fpr',
 'selection_witness', 'split_lock_hash', 'decode_identity', 'schema_version']
```

The digest (`:250-277`) is:

```python
parts = {
    "protocol": protocol_name, "role": role,
    "corpus_fingerprint": corpus_fingerprint, "n_samples": n_samples,
    "temperature": quantize(temperature), "nll_before": quantize(nll_before),
    "nll_after": quantize(nll_after),
    "threshold": quantize(threshold) if threshold is not None else None,
    "margin": quantize(margin) if margin is not None else None,
    "target_fpr": target_fpr,          # NOT quantized — see §1.6
    "selection_witness": selection_witness, "split_lock_hash": split_lock_hash,
    "schema": schema_version,
}
if schema_version in SCHEMA_VERSIONS_WITH_DECODE:  # {"1.2.0"}
    parts["decode_identity"] = decode_identity
```

**There is no parameter, and therefore no digest key, naming the weights, the
gate, or the face/video selection policy that produced the logits being fit.**
The generalisation from "B21 named `input_size`" to "B23A names
`effective_model_version` / gate / `.f.`" is **confirmed correct** — the
omission is total, not partial: every component of the model identity
(weights, gate, selection) is equally absent, for the identical reason
`input_size` is (`input_size` only reaches the witness *through* the fitted
numbers, exactly like every other model-identity component; nothing about it
is special-cased).

### 1.3 Why this is not circular — `calibration_fingerprint` deliberately excludes the witness

`inference/models.py:76-84` (`CalibrationSpec.witness` docstring):
*"**Deliberately NOT part of `calibration_fingerprint`.** That fingerprint is
half of the result cache key and answers one question: 'would this deployment
produce the same number?' ... two deployments with identical T, threshold and
band are cache-equivalent whether or not one of them can prove where it came
from."*

`ModelSpec.calibration_fingerprint` (`:208-256`) reads only
`self.calibration.{temperature,fitted_on,n_samples}` and
`self.threshold.{value,band_lo,band_hi,target_fpr}` — never
`self.calibration.witness`. So `effective_version` (`:297-310`, `+c.` segment)
has **no dependency on the witness**, and a witness that recorded
`weights_identity` / `gate_fingerprint` / `face_selection_fingerprint` at fit
time would introduce **no cycle**: those three are properties of the `ModelSpec`
that produced the logits being fit, computed *before* calibration is fitted, and
already excluded from what feeds calibration's own contribution to
`effective_version`. This confirms a fix is architecturally free to add a
model-identity field without restructuring `contracts/version.py` or
`inference/models.py`'s existing exclusion rule.

### 1.4 The four identity chains, traced end to end

| Chain | Computed by | Feeds | Independent of the witness? |
|---|---|---|---|
| **weights** | `ModelSpec.weights_identity` (`models.py:259-294`) — version + branch weights + `input_size` (folded in since B22/R25) | `effective_version`'s prefix | Yes — witness never reads it |
| **gate** | `MediaCaps.gate_fingerprint` (`preprocess.py:71-90`) — the quality-gate caps | `effective_version`'s `.g.` segment via `ModelSpec.with_gate` | Yes |
| **selection (`.f.`)** | `inference.preprocess.selection_identity(face, video)` — face selection, plus frame selection/pooling for video (B23/R29) | `effective_version`'s `.f.` segment via `ModelSpec.with_selection` | Yes |
| **decode (`.d.`)** | `inference.decode_env.decode_identity()` | Runtime version's `.d.` segment; **also** a witness digest key since 1.2.0 (`SCHEMA_VERSIONS_WITH_DECODE`) | **No** — this is the one axis Batch 11 already closed |

None of weights, gate, or selection is threaded into the witness by any
mechanism, direct or indirect. Decode is the sole precedent for "an
environment axis that reaches the digest as a caller-supplied opaque string,"
and it is the template §3 reuses.

### 1.5 `selection_witness` vs `.f.` — a naming collision that must not recur

`CalibrationWitness.selection_witness` (`calibration_protocol.py:149`) is the
**dataset-selection** witness — `RoleSelection.witness` from
`evaluation.protocol.select()` — naming *which media items* were used, not
*which face/frames were cropped from them*. `contracts/version.py:141-148`
already documents this exact collision and why `.f.` is not called `.s.`:
*"selection is already taken in this codebase by dataset selection
(`CalibrationWitness.selection_witness`)."* Any new field this design adds must
not be named anything containing "selection" for the same reason.

### 1.6 A pre-existing, unrelated finding surfaced in passing

`target_fpr` enters the digest **unquantized** (`:260` — no `quantize()` call,
unlike `threshold`/`margin`/`temperature`/`nll_before`/`nll_after`). `target_fpr`
is normally a configured constant (`0.02`, `0.05`) rather than a fitted
quantity, so this is usually harmless — but `fit_calibration` accepts it as a
caller-supplied `float` with no rounding, so two calls differing at the 1e-15
level (e.g. `0.02` computed two different ways) would produce different
digests where every *fitted* quantity is identical. **Not part of R31** (it is
not a model-identity omission) and **not fixed by this design** — recorded here
because it was found during this re-derivation and would otherwise be lost;
disposition in §9 (N10).

---

## 2. Is the gap real? — live demonstration, not assumed

`docs/CROP_IDENTITY_B22_REVIEW.md` called the collision "incidental" and
supported it with reasoning, not a run. This design re-derives it live, using
the actual `fit_calibration` / `CalibrationWitness` code (not a synthetic
stand-in), on a real `RoleSelection` built through `evaluation.protocol.select`
exactly as the R7/R5 test suites do it.

### 2.1 Method

Corpus: 3200 synthetic `MediaItem`s (`ffpp`/`celebdf`, matching
`tests/test_calibration_witness.py`'s fixture), split-locked, a CALIBRATION
selection of 237 items resolved through `evaluation.protocol.select`. Three
"models'" logits over the *same* selection:

- **Model A** — baseline logits, `margin=1.4` separation, seed 0.
- **Model B** — a model whose weights/gate/selection differ from A's, but whose
  change happens to touch **zero** items in this calibration selection (the
  realistic case: a `min_face_px` raise, or a face-selection tie-break change,
  that this particular 237-item split never exercises). Modelled as **byte-identical
  logits** to A — this is not a contrived edge case, it is the direct
  consequence of fitting on a *fixed, already-vetted* calibration split while
  changing something that only affects a subset of possible inputs.
- **Model C** — a model whose logits differ from A's by noise at
  `~3e-9` absolute magnitude (`np.random.default_rng(1).normal(0, 1e-9, ...)`) —
  representative of the float32/float64 or sub-pixel-interpolation-scale
  differences a real weights/selection change could plausibly produce on a
  corpus it does affect only slightly.
- **Model D** (negative control) — logits with a **measurably different**
  separation (`margin=1.6`), to confirm the witness does separate calibrations
  that actually differ.

### 2.2 Result *(live, this session)*

```
Structural check — compute_witness's full parameter list:
['protocol_name', 'role', 'corpus_fingerprint', 'n_samples', 'temperature',
 'nll_before', 'nll_after', 'threshold', 'margin', 'target_fpr',
 'selection_witness', 'split_lock_hash', 'decode_identity', 'schema_version']
(no weights / gate / face_selection / effective_model_version parameter exists)

Scenario A (calibration-invisible model change — byte-identical logits):
  model A witness: d7ed75b9
  model B witness: d7ed75b9
  COLLISION: True

Scenario B (sub-float-noise perturbation, max abs delta 2.87e-09):
  model A witness: d7ed75b9
  model C witness: d7ed75b9
  model A temperature (unquantized): 1.1465133663773062
  model C temperature (unquantized): 1.1465133757805317
  COLLISION: True

Negative control (materially different fit, margin 1.4 vs 1.6):
  model A witness: d7ed75b9
  model D witness: 04c3d5c7
  COLLISION: False
```

### 2.3 What this proves

- **The omission is not merely theoretical or "incidental-by-luck."** Two
  `CalibrationWitness` digests are byte-identical for two different models
  whenever those models' logits over the calibration corpus are close enough
  (Scenario B: ~1e-9 absolute logit noise, five orders of magnitude below
  anything `FINGERPRINT_DECIMALS`/`quantize` is calibrated to treat as
  meaningful — `contracts/version.py:172-178`'s own stated design point is that
  1e-6 noise should round away; **model-identity information is rounding away
  right alongside it**, which is not the same claim and was not previously
  measured).
- **The realistic vector is Scenario A, not Scenario B.** A gate or selection
  change that happens not to touch the calibration split's items produces
  **exactly identical** logits, not merely close ones — no float convergence
  needed at all. Given a calibration split is, by design, small and fixed
  (`MIN_WITNESSED_SAMPLES = 200`, `evaluation/calibration_protocol.py:119`) while
  production traffic is unbounded, "a model change this calibration split
  cannot see" is not a rare coincidence — it is what a narrow gate/selection
  tightening looks like on a fixed 200-row set most of the time.
- **The witness does separate genuinely different fits** (negative control) —
  this is not "the witness never distinguishes anything," it is specifically
  blind to the one axis (model config) it never receives as input.

---

## 3. Is this exploitable today? — production-wiring audit

### 3.1 No serving-time consumer exists *(live, confirmed by search)*

```
grep -rln "witness|CalibrationWitness|calibration_protocol" backend/ workers/
→ backend/config.py only (two comment mentions of "witness" in docstrings about
  what does NOT bump; zero imports, zero calls)

grep -rn "\.calibration_spec\(\)|CalibrationSpec\(" --include="*.py" .
→ inference/models.py:307 (the method definition itself)
→ tests/test_calibration_witness.py:344 (a test)
  — no other call site anywhere in the repository
```

**No production code path builds a `ModelSpec` from a `CalibrationWitness`.**
`CalibrationSpec` defaults to `temperature=1.0, n_samples=0` (identity, no
fit — `models.py:55-64`), which is what every shipped `STUB_ENGINE_VERSION`
carries. `CalibrationWitness.calibration_spec()` (the one bridge method that
exists) is exercised only in a test. This matches
`REPRODUCIBILITY_AUDIT.md`'s own recurring caveat — "no real model exists to
calibrate yet" — and is now confirmed by direct search rather than repeated
from the audit doc.

### 3.2 But there IS a live, exercised consumer — `evaluation/benchmark.py`

This is the correction this re-derivation makes to the inherited framing.
"No production model" is not the same claim as "no exposure today" — R31's
mechanism is exercised on **every benchmark report this codebase can already
produce**:

- `run_benchmark` (`benchmark.py:346-524`) accepts a `calibration=` argument
  (a `CalibrationWitness` or its dict) **and separately** an
  `effective_model_version=` string, "passed in ... because it is the
  *deployment's* identity — weights, calibration, threshold and gate composed —
  and only the caller holds all four" (`:361-363`, quoted verbatim). These are
  **two independent arguments** (`:458-476` derives `calibration_fingerprint`
  from the witness but never touches `effective_model_version`) — nothing
  requires or checks that they describe the same model.
- `BenchmarkReport.validate()` (`:943-990`) checks that `primary` and
  `in_domain` runs share one `effective_model_version` (`:950-960`), and calls
  `verify_calibration_witness(self.primary.calibration, protocol=self.protocol)`
  (`:975-977`) — **with no `decode_identity=` and no model-identity argument at
  all**, even though `verify_calibration_witness` already *supports* a
  `decode_identity=` cross-check (`calibration_protocol.py:748-760`) that this,
  the one production call site, does not use.
- `regression_gate` (`:783-829`) cross-checks `effective_model_version` and
  `decode_identity` **between two benchmark runs** (`incumbent`/`challenger`) —
  the "benchmark identity" protection in the four-protections table (§4). It
  never reaches into either run's *own* `calibration` witness to ask whether
  that witness's fitted numbers are consistent with that run's own
  `effective_model_version`.
- `to_payload()` (`:897-941`) publishes `decode_identity` and
  `calibration_decode_identity` **side by side**, with the comment
  *"Published so a reader can see it, not demanded so a reader can trust
  it"* (`:922-925`) — an explicit, on-record acknowledgment that even the one
  axis (decode) the witness *does* record is not asserted to match, only
  displayed for a human to notice.

**Conclusion: R31's blast radius today is the evaluation/reporting path, not
serving.** A benchmark report can pair a witness fitted under one model
configuration with a caller-declared `effective_model_version` for a different
one, and nothing in the report's own `validate()` would catch it — the
`effective_model_version` equality check only guards against pairing two
*different* runs' declared versions, not against a witness silently
misrepresenting the model it was actually fitted for.

### 3.3 The "four protections," applied to R31 specifically

Using the exact framework `test_decode_identity_completeness.py:249-257`
defines for R27/R30 (`_four_protections_separate`), re-derived for a
model-config difference rather than a decode difference:

| Protection | Mechanism | Blind to a model-config difference? |
|---|---|---|
| **Content cache key** | `settings.model_version` = `effective_version` + `.d.` | **No** — `.g.`/`.f.`/weights already separate it; cache keys are unaffected by R31 |
| **Deployment identity** (`/health`, `verify_version_alignment`) | Same `effective_version` | **No** — same reason |
| **Benchmark identity** (`regression_gate`) | `BenchmarkRun.effective_model_version` equality (`:816-821`) | **No** — compares the *declared* strings directly, not derived from the witness |
| **Calibration identity** (`CalibrationWitness` digest) | `compute_witness`'s preimage | **Yes — this is the one blind protection** |

This is a materially **narrower** severity profile than R27 (all four blind)
or R30-as-originally-framed (all four blind). Three of the four protections
never route through the witness at all and are completely unaffected by this
finding. The exposure is specifically: **the calibration-provenance claim
itself can misrepresent which model it corroborates**, which matters exactly
when something reads the witness expecting it to answer that question — today,
that is a human reading a benchmark report's `to_payload()`, and in the future,
whatever code eventually wires `calibration_spec()` into a serving `ModelSpec`.

---

## 4. Architecture constraints

Restated from the standing constraints (`docs/B24_CANDIDATE_ANALYSIS.md`,
`docs/B24_R2_DESIGN.md`, unchanged): no SCRFD, no ONNX, no training, no
frontend, no Redis, no `CONTRACT_VERSION` / `EvidenceBundle` / `ThresholdInfo`
/ OpenAPI-snapshot change. Specific to this design:

- **No circularity.** §1.3 confirms the fix's proposed field is computed
  *before* calibration and is not read by `calibration_fingerprint`; nothing
  here can create a dependency cycle in `effective_version`'s composition.
- **`CALIBRATION_WITNESS_VERSION` is on its own axis**, independent of
  `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / model identity
  (`calibration_protocol.py:74-81`, restated from the module's own docstring).
  A bump here moves none of those three.
- **The schema-version-as-parameter migration pattern already exists and is
  proven** (`compute_witness`'s `schema_version` parameter, the 1.1.0→1.2.0
  precedent, `test_an_old_calibration_witness_still_verifies_after_the_change`).
  Any new schema version must follow it, not invent a second mechanism.
- **`SCHEMA_VERSIONS_WITH_DECODE` is a named frozenset, not a `>=` comparison**
  (`:107-114`, the module's own stated reason: "a future schema that *removes*
  the field would silently start covering it again under a `>=` test"). A
  parallel `SCHEMA_VERSIONS_WITH_MODEL_CONFIG` must follow the same shape.
- **No name containing "selection"** for the new field (§1.5).

---

## 5. Candidate designs

### Option A — one `model_config_identity` field, parallel to `decode_identity`

A new optional parameter on `fit_calibration()` / `CalibrationWitness.compute_witness()`
/ the dataclass, computed by the **caller** (whoever ran inference to produce
the logits being fit) as:

```python
fingerprint({
    "weights": spec.weights_identity,
    "gate": spec.gate_fingerprint,
    "face_selection": spec.face_selection_fingerprint,
})
```

— i.e. `effective_version` computed with `calibration=""` conceptually
subtracted out, but expressed as its own fingerprint rather than by string
surgery on `effective_version` (which would require parsing a version string
inside `evaluation`, the wrong direction of dependency). Not detected by
`fit_calibration` itself, for the identical reason `decode_identity` is not
detected (`calibration_protocol.py:495-500`): the function is handed arrays and
cannot know which process, or which `ModelSpec`, produced them.

| Axis | Assessment |
|---|---|
| **Closes the gap** | Yes — two models differing in weights, gate, or selection now differ in `model_config_identity` regardless of whether their logits converge on the calibration corpus. |
| **Circularity** | None (§1.3, §4). |
| **Migration cost** | Identical profile to Batch 11's `decode_identity` addition: one new optional field, gated by a new `SCHEMA_VERSIONS_WITH_MODEL_CONFIG = frozenset({"1.3.0"})`, `CALIBRATION_WITNESS_VERSION` `1.2.0` → `1.3.0`. **Zero stored witnesses exist to migrate** (§3.1) — cheaper than Batch 11's own migration, which had to guarantee old witnesses still verify with none actually in production either. |
| **API surface** | `verify_calibration_witness` gains a `model_config_identity=` keyword, exact parallel to its existing `decode_identity=` keyword (`calibration_protocol.py:646,748-760`) — same cross-check shape, same "recorded but not supplied is a note, mismatch is a refusal" pattern. |
| **Naming** | `model_config_identity` avoids "selection" (§1.5) and avoids "model identity" (ambiguous with `effective_version`, which *includes* calibration). |
| **Granularity** | One opaque fingerprint — consistent with `.f.`'s own precedent (`contracts/version.py:61-71`: a single composite is preferred over a new parser era for a legibility gain `differences()`-style reporting already provides off other objects). |

### Option B — three separate digest keys (`weights_id`, `gate_id`, `selection_id`)

Same coverage as A, but the witness digest carries three keys instead of one
composite.

| Axis | Assessment |
|---|---|
| **Coverage** | Identical to A. |
| **Legibility** | A mismatch could name *which* of the three moved without decoding anything — the "Option D" pattern `docs/B24_DESIGN.md` §2 considered for the image/video decode split. |
| **Cost** | Three new optional constructor parameters instead of one; three new `None`-handling branches in `compute_witness`; a wider `to_dict()`/`from_dict()` diff. |
| **Precedent verdict** | `docs/B24_R2_DESIGN.md` §3 rejected the equivalent finer-grained option (composite `.d.`) for the *decode* axis with: "the legibility gain is real but not load-bearing... there is no deployment yet." **The identical argument applies here, more strongly** — §3.1 shows there is not merely no deployment, there is no code path that would ever read three separate keys apart; a human reading `to_dict()` already sees `weights_identity`/`gate_fingerprint`/`face_selection_fingerprint` on the `ModelSpec` side if they want to triage, so the witness doesn't need to re-expose them. |

### Option C — a provenance-only field outside the digest (like `created_at`)

Record the caller's model config as metadata beside the witness, not inside
its digest.

**Reject.** This is a regression to the exact defect the witness type was
built to eliminate. `CalibrationSpec.fitted_on` was originally free text
"written by the same caller that did the fit... nothing has ever checked this
string against anything" (`models.py:56-63`) — the module docstring's own
opening motivation (`calibration_protocol.py:1-20`) is that an unverified,
digest-external provenance string is *laundering*, not evidence. Doing the
same thing for the model axis while having just fixed it for the corpus axis
would be incoherent within this codebase's own stated design philosophy.

### Option D — harden `verify_calibration_witness`'s call sites only, no digest change

Wire `run_benchmark`/`BenchmarkReport.validate()` to pass `decode_identity=`
(already supported, currently unused — §3.2) and require a caller-supplied
model-config string to `verify_calibration_witness`, without adding anything
to the digest itself.

**Reject as sufficient on its own; adopt as a companion, §8.4.** Wiring the
*existing* `decode_identity=` check into `benchmark.py:975` is a real,
independent gap (§3.2) worth closing regardless of this design, and costs
nothing (no schema change). But without a digest field to compare *against*,
"harden the caller" has no ground truth: `verify_calibration_witness` would
need a `model_config_identity=` argument to compare the caller's value to, and
there is nothing in the witness to compare it *to* unless Option A also lands.
Option D alone leaves the witness itself just as blind — it only makes *one*
caller marginally more careful.

### 5.1 Recommendation — **Option A**, with Option D wired in as a companion fix

Option A is the direct, minimal-diff analogue of the Batch 11 `decode_identity`
precedent — same shape, same migration mechanism, cheaper migration (zero
stored witnesses today vs. Batch 11's live-but-empty risk). Option D's
`benchmark.py:975` wiring should land in the same batch: it is a one-line
change (`decode_identity=self.primary.decode_identity`,
`model_config_identity=` derived from `self.primary.effective_model_version`'s
weights/gate/selection prefix) that closes the one concretely-exercised gap
identified in §3.2, at zero additional schema cost.

---

## 6. Version-bump analysis

| Quantity | Verdict | Reason |
|---|---|---|
| **`CALIBRATION_WITNESS_VERSION`** | **MOVES** — `1.2.0` → `1.3.0` | `compute_witness`'s preimage gains a key, exactly the 1.1.0→1.2.0 precedent. *(live, this session)*: fingerprinting an identical field set under `schema="1.2.0"` (no `model_config_identity` key) vs `schema="1.3.0"` + `model_config_identity=None` gives `10625ce9` vs `bfe2d0d1` — **confirmed different**, for the same reason the module docstring already states for 1.1.0→1.2.0 (`092895e2`→`54adb0e2`, `calibration_protocol.py:236-237`): the schema string itself is a digest input. |
| **`SCHEMA_VERSIONS_WITH_MODEL_CONFIG`** | **NEW** — `frozenset({"1.3.0"})` | Parallel to `SCHEMA_VERSIONS_WITH_DECODE`; a 1.2.0 payload cannot carry `model_config_identity` in its digest (mirrors the existing `ProtocolViolation` at `:265-276` for `decode_identity` on a pre-1.2.0 schema). |
| **`.d.` / `.f.` / `weights_identity` / `STUB_ENGINE_VERSION`** | **do not move** | Nothing in `inference/decode_env.py`, `inference/preprocess.py`'s selection identity, or `inference/models.py`'s `weights_identity`/`effective_version` reads the witness (§1.3 confirms the exclusion is pre-existing and unaffected). |
| **`model_identity_of(STUB_ENGINE_VERSION)`** | does not move | Same reason. |
| **`CONTRACT_VERSION`** | does not move | `ModelInfo.version` is an unconstrained `str`; the witness is not a wire contract type. |
| **`RUN_SCHEMA_VERSION`** | does not move | `BenchmarkRun.calibration: dict \| None` stores whatever payload it is given (`benchmark.py:200`); the *schema* (which fields `BenchmarkRun` has) is unchanged — only the *contents* of the nested calibration dict gain a key, exactly as `decode_identity`'s own Batch-11 addition did not require a `RUN_SCHEMA_VERSION` bump. |
| **`LOCK_VERSION`** (split-lock schema) | does not move | Unrelated artifact (`evaluation/split_lock.py:81`); this design does not touch it. |
| **OpenAPI signature / `schema_out` / `fixtures_out`** | do not move | The witness is never a wire type. *(live)* `grep -rln "CalibrationWitness\|calibration_protocol" contracts/ schema_out/ backend/openapi.snapshot.json` → one hit, `contracts/version.py:146` — the `FACE_SELECTION_MARKER` docstring's prose mention of `CalibrationWitness.selection_witness` in its naming-collision explanation (§1.5), not a code dependency (no import, no call). `schema_out/` and `backend/openapi.snapshot.json` → no hits. |

---

## 7. Migration analysis

### 7.1 Stored-witness compatibility — **trivial, because none exist**

`compute_witness` already dispatches on the **payload's own** `schema_version`
(`:181-196`, `:699-700`) — the exact mechanism that let 1.1.0 witnesses survive
the 1.2.0 bump. A witness written under `1.2.0` (no `model_config_identity`
key in its digest) continues to verify under the `1.2.0` rule after this
design ships; nothing recomputes it under `1.3.0`. **§3.1 additionally
confirms there is no stored witness anywhere in this repository or any
deployment it describes** — this migration protects a case that, unlike
Batch 11's, is not merely *safe*, it is currently *vacuous*. That is a reason
to do this now rather than a reason to skip it: every day without it is a day
closer to a real witness existing that *would* need the old-schema-still-verifies
guarantee to be exercised for real.

### 7.2 Benchmark / report compatibility

`BenchmarkRun.calibration: dict | None` stores whatever `CalibrationWitness.to_dict()`
produces; a `1.2.0`-schema calibration dict loaded by report code built after
this design still parses (the new field is additive and optional-with-`None`
default, following the identical `decode_identity` precedent at
`from_dict`'s `:363-366`). `to_payload()` gains one new key
(`calibration_model_config_identity`, or similar) exposed for a human reader —
no `REQUIRED_REPORT_FIELDS` change (matching how `decode_identity` was
"published, not demanded," `:922-925`).

### 7.3 Rollback

Symmetric to every prior schema bump: revert `calibration_protocol.py`
(drop the field, the frozenset, `CALIBRATION_WITNESS_VERSION` back to `1.2.0`)
and `benchmark.py`'s `verify_calibration_witness` call-site wiring (§5.1). No
stored artifact is orphaned (§7.1 — none exist); no cache generation is spent
(§6); no coordinated-deploy runbook line is needed, because
`CALIBRATION_WITNESS_VERSION` gates no `verify_version_alignment` check and no
`STUB_ENGINE_VERSION` segment.

### 7.4 Deploy coordination — **none required**

Unlike every `.d.`/`.f.` batch (B20/B22/B23/B24), this bump touches no
runtime-version segment `verify_version_alignment` or the worker-boot gate
inspects. R34 (fail-closed OR1 makes a `STUB_ENGINE_VERSION` move
non-rolling-deployable) **does not apply** — there is no rolling-deploy risk
here at all.

---

## 8. Cache / benchmark / witness impact

- **Result cache:** unaffected. The content cache and idempotency keys
  (`backend/service.py:98,227`) key on `settings.model_version`, which never
  reads `CalibrationSpec.witness` (§1.3). Zero cache generations.
- **Benchmark rows:** additive field on the nested `calibration` dict; existing
  rows (none in this repo, but the mechanism is general) load unchanged
  (§7.2).
- **Witness compatibility:** every existing (in-repo test-fixture) witness
  continues to verify under its recorded schema (§7.1).
- **Companion fix impact (§5.1, Option D):** wiring `decode_identity=` and a
  new `model_config_identity=` into `BenchmarkReport.validate()`'s call to
  `verify_calibration_witness` (`benchmark.py:975`) turns two currently-silent
  gaps (§3.2) into refusals. This is a **behavior change** for any report whose
  primary run's declared `decode_identity`/model config does not match its
  attached witness's recorded values — today, no such report exists in this
  repository (`tests/test_calibration_witness.py`'s fixtures always fit and
  record consistently), so this is expected to be a no-op on the existing test
  suite and a real new protection going forward.

---

## 9. Test plan

Template: the Batch 11 `decode_identity` addition to this exact file
(`SCHEMA_VERSIONS_WITH_DECODE`, the `ProtocolViolation` guard, the
`from_dict` optional-field handling) is the proven precedent; this plan mirrors
it field-for-field. New tests extend `tests/test_calibration_witness.py`
(the R5-calibration suite) and add a companion suite section for `benchmark.py`
wiring.

### 9.1 Fail-before matrix (real failures against the pre-R31 tree)

| # | Test | Pre-R31 behaviour |
|---|---|---|
| FB1 | `test_a_model_config_difference_does_not_move_the_witness` (behavioural, mirrors §2's live study) | **passes today** (i.e. the bug reproduces): two `fit_calibration` calls with byte-identical logits but different (hypothetical) model config produce the same `witness.witness` — this is the "fail" the design closes |
| FB2 | `test_a_sub_float_noise_perturbation_still_collides` | **passes today**: `~1e-9`-scale logit perturbation still yields an identical witness (Scenario B, §2.2) |
| FB3 | `CalibrationWitness.compute_witness(..., model_config_identity=...)` | **fails** — `TypeError: unexpected keyword argument` (parameter does not exist) |
| FB4 | `test_model_config_identity_is_recoverable_for_forensics` (`"model_config_identity" in witness.to_dict()`) | **fails** — key absent |

### 9.2 Pass-after matrix (real protections after R31)

| # | Test | Assertion |
|---|---|---|
| PA1 | `test_a_model_config_difference_now_moves_the_witness` | Re-run §2's Scenario A/B with `model_config_identity` supplied and differing between "model A" and "model B/C" → witnesses differ, closing FB1/FB2 |
| PA2 | `test_the_calibration_identity_protection_now_separates_a_model_config_difference` | Using the `_four_protections_separate`-style helper (§3.3), confirm **calibration_identity** now separates while **content_cache_key** / **deployment_identity** / **benchmark_identity** are correctly reported as *already* separating (they never depended on the witness) — i.e. this closes the *one* blind protection without claiming to have touched the other three |
| PA3 | `test_model_config_identity_is_recoverable_for_forensics` | `"model_config_identity" in witness.to_dict()` |
| PA4 | `test_a_1_2_0_payload_cannot_smuggle_a_model_config_identity` | Mirrors `:265-276`'s existing `ProtocolViolation` test for `decode_identity` under a pre-1.2.0 schema — `compute_witness(schema_version="1.2.0", model_config_identity="x", ...)` raises |
| PA5 | `test_verify_calibration_witness_cross_checks_model_config_identity` | Mirrors the existing `decode_identity` cross-check tests (`calibration_protocol.py:748-760`): `verify_calibration_witness(w, model_config_identity=X)` reports a problem when `X` differs from `witness.model_config_identity`, and a distinct problem when the witness records `None` and `X` is supplied |
| PA6 | `test_the_benchmark_report_now_wires_the_existing_decode_check` (Option D) | `BenchmarkReport.validate()` calls `verify_calibration_witness` with `decode_identity=self.primary.decode_identity` and `model_config_identity=` derived from `self.primary.effective_model_version`; a synthetic report with a mismatched witness now fails `validate()` where it silently passed before |

### 9.3 Non-movement matrix

| # | Test | Assertion |
|---|---|---|
| NM1 | `test_decode_identity_is_unchanged` | `decode_identity() == "a97ccb68"` (unaffected — this design touches no decode code) |
| NM2 | `test_stub_engine_version_is_unchanged` | `STUB_ENGINE_VERSION` literal unchanged |
| NM3 | `test_model_identity_is_unchanged` | `model_identity_of(STUB_ENGINE_VERSION)` unchanged |
| NM4 | `test_no_contract_run_schema_or_lock_version_bump` | `CONTRACT_VERSION == "1.0.0"`, `RUN_SCHEMA_VERSION == "1.2.0"`, `LOCK_VERSION == "1.1.0"` — **`CALIBRATION_WITNESS_VERSION` deliberately excluded from this list**, since it is the one constant this design *does* move (mirrors how B20's non-movement matrix excluded `.d.` itself) |
| NM5 | `test_an_old_calibration_witness_still_verifies_after_the_change` | Direct analogue of `test_decode_identity_completeness.py:384-392`: a `schema_version="1.2.0"` witness with no `model_config_identity` round-trips through `to_dict()`/`from_dict()` and its digest still matches |
| NM6 | `test_the_calibration_fingerprint_still_excludes_the_witness` | `ModelSpec.calibration_fingerprint` unchanged by this design (`inspect.getsource` byte-identical to a pinned string) — confirms §1.3's no-circularity claim stays true, not just at design time |
| NM7 | `test_no_production_serving_path_reads_calibration_witness` | The `grep`-based structural check from §3.1, turned into a standing tripwire: `grep -rln "CalibrationWitness" backend/ workers/` finds only comment references, so this design's companion fix does not accidentally wire a serving-time dependency into `backend`/`workers` |
| NM8 | `test_full_suite` | `python tests/run_all.py` → suite count 51 → 51 (no new suite; extends `test_calibration_witness.py` and `test_evaluation_benchmark.py`), test count increases by the count of §9.1/§9.2's new tests, 0 failures |

### 9.4 Collateral matrix — pin updates (mechanical, no logic change)

**Six files assert `CALIBRATION_WITNESS_VERSION == "1.2.0"` as a non-movement
pin and must be updated to `"1.3.0"`** *(live, confirmed by search — this list
is exhaustive, not illustrative)*:

| File:line | Old | New |
|---|---|---|
| `tests/test_crop_identity.py:259` | `== "1.2.0"` | `== "1.3.0"` |
| `tests/test_decode_identity_completeness.py:359` | `== "1.2.0"` | `== "1.3.0"` |
| `tests/test_dependencies.py:407` | `== "1.2.0"` | `== "1.3.0"` (added in B24/N5 — this design's collateral list must include the batch that most recently touched this constant) |
| `tests/test_evaluation_identity.py:475` | `a.schema_version == CALIBRATION_WITNESS_VERSION == "1.2.0"` | `== "1.3.0"` |
| `tests/test_headline_and_face_semantics.py:462` | `== "1.2.0"` | `== "1.3.0"` |
| `tests/test_video_selection_identity.py:325` | `== "1.2.0"` | `== "1.3.0"` |

`tests/test_calibration_witness.py`'s own uses (`:278`, and the module's fit
helpers) are **not collateral** — they read the constant rather than pin a
literal, so they update automatically.

### 9.5 Study

New `tests/calibration_witness_binding_study.py` (print-only, no assertions,
matching `decode_identity_completeness_study.py`'s template): the §2 collision
demonstration, kept re-runnable so a future reviewer does not have to
reconstruct the scratch script this design was verified with.

---

## 10. Re-rank versus all currently open findings

Axes as `docs/B24_R2_DESIGN.md` §7: **Repro** / **Cache** (corruption risk) /
**Deploy** (risk from the fix) / **Migr** / **Cx**. **Act** = actionable now
within standing constraints.

| Finding | Repro | Cache | Deploy | Migr | Cx | Act | Note (post-re-derivation) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| **R13** — face-crop instability (310×, measured) | 5 | 4 | 2 | 5 | 5 | ❌ SCRFD | unaffected; still most severe, still un-batchable |
| **R31** — calibration witness omits model config | **3** | **0** | **0** | **1** | 2 | ✅ | **This design.** Confirmed real (§2, live), confirmed exploitable **only** in the evaluation/reporting path (§3.2), confirmed to leave 3 of 4 protections untouched (§3.3). Cache=0 (§8 — no cache-key involvement at all, unlike every `.d.`/`.f.` batch). Deploy=0 (§7.4 — no rolling-deploy coordination). Migr=1 (§7.1 — zero stored witnesses to migrate, the cheapest version bump in this codebase's history). |
| **OR1-b** — `backend/worker_main.py` composition root | 1 | 2 | 3 | 1 | 1 | ✅ | Scheduled since `BATCH_23_R3_DESIGN.md` §1.12. Protects an **already-active** code path (the config-default worker-boot alignment check) — see §10.1. |
| **N9** *(new, B24)* — undeclared test-only `httpx2`... | — | — | — | — | — | closed | unaffected by this design |
| **R30-bundle** (R30 + D8 + D9 + D10) | 2 | 2 | 3 | 2 | 3 | ✅, low value | Unaffected; deferred behind N5 per `B24_R2_DESIGN.md` §7. |
| **N8** | 1 | 1 | 1 | — | — | closed | Unaffected — closed as accepted architectural cost (`B24_R2_DESIGN.md` §4). |
| N6 / N7 | 2 | 1 | 1 | 3 | 2 | ✅ with R31 | **Fold into this batch or the one after it** — both are witness/fit-provenance gaps of the identical shape (fit-optimiser config, `np.quantile` method — unrecorded fitted-procedure detail), and `CALIBRATION_WITNESS_VERSION` is already moving in this design. Folding them in avoids a *second* 1.3.0→1.4.0 bump for the same class of gap. See §10.2. |
| R24, R23, R18, R26, R16, R32, R34, N3 | — | — | — | — | — | unchanged | No interaction with this design; dispositions stand as in `B24_R2_DESIGN.md` §7. |

### 10.1 Is R31 (as scoped here) still the correct next batch? — **Materially, yes; nominally, it ties with OR1-b**

The re-derivation **downgrades** R31's severity relative to how it was
inherited (§3.3: only 1 of 4 protections is blind, not "all four" the way
R27/R30 were originally framed) but **upgrades** its immediacy (§3.2: unlike
R30, which turned out to have *no* measured exploitation vector, R31's gap is
exercised on every benchmark report this codebase can already produce today —
it is not gated on a future model existing). Net effect: R31 remains
actionable and cheap, but the honest comparison is against **OR1-b**, not
against R30 or R13.

| | R31 (this design) | OR1-b |
|---|---|---|
| Protects an active code path today? | The evaluation/reporting path (§3.2) — active in the sense that report code exists and runs, but no real report has yet been produced with a mismatched witness | The config-default worker-boot alignment check — active in the sense that every worker boot already runs *some* version of this check, and OR1-b closes the one path (`$DFP_MODEL_VERSION` unset, config default used) that gets the weaker check |
| Cache generation | 0 | 0 |
| Deploy coordination | none | none |
| Migration cost | Lowest possible (§7.1 — zero stored artifacts) | Low (additive wiring, `BATCH_23_R3_DESIGN.md` §1.12 already scoped it) |
| Cost of waiting | Low now, **rises** once a real witness/report exists in the wild (§7.1's "vacuous today, not vacuous forever" argument) | Low and roughly flat — the gap does not get harder to close later |

**Neither dominates the other on the evidence.** Both were already
near-tied in `docs/B24_R2_DESIGN.md` §7 and `docs/B24_N5_DESIGN.md` §7
("R31 #1 next... or, if the team wants another zero-risk batch first, OR1-b").
This design does not manufacture a tiebreaker that is not there. **Recommendation:
do both — they are disjoint files (`evaluation/calibration_protocol.py` +
`evaluation/benchmark.py` here; a new `backend/worker_main.py` plus a
re-added `Settings.skip_version_alignment` for OR1-b), touch no common code,
and neither blocks the other.** If forced to sequence one first: R31's "cost of
waiting rises" argument (§7.1) is the tiebreaker this design adds that prior
documents did not have — do R31 first, while its migration is still free.

### 10.2 N6/N7 folding — a scope decision for the reviewer, not assumed here

N6 (fit-optimiser configuration unrecorded) and N7 (`np.quantile` method
unrecorded) are, on inspection, the same *class* of gap as R31 — a fitted
*procedure* detail that could change a stored numeric output's meaning without
changing the digest — but they are not re-derived in this document (out of
this design's stated scope, which is R31 specifically) and **are not
included in §5–§9's scope**. §10's table flags them as a candidate fold-in for
whoever schedules the implementation, not as work this design has already
specified. If folded in, they would ride the same `CALIBRATION_WITNESS_VERSION`
1.3.0 bump this design already spends — but the gate below (§11) is scored
against R31 alone.

---

## 11. Final gate

## **APPROVE WITH CHANGES**

The finding is confirmed real (not merely inherited — §2, live), the
architecture is confirmed free of circularity (§1.3), the migration is
confirmed to be the cheapest version bump available in this codebase's history
(§7.1 — zero stored artifacts), and the design (Option A + the Option D
companion wiring) is a direct, minimal-diff analogue of an already-proven
precedent (Batch 11's `decode_identity`). This is not a rediscovery of R30's
outcome — unlike R30, the collision is demonstrated live in this document
(§2.2), not merely argued to be theoretically possible.

The changes required before implementation are specification-only:

### Blocking defect 1 — name the new field and its frozenset explicitly *(specification-only)*

§5's `model_config_identity` name and `SCHEMA_VERSIONS_WITH_MODEL_CONFIG`
frozenset are this design's recommendation, not yet ratified by a second
reviewer. A reviewer must explicitly accept the name (§1.5 constrains it away
from anything containing "selection") before implementation, exactly as
B24_DESIGN.md's SIMD/zlib naming decisions required ratification. *No
architectural impact either way — it is a field name and a set literal.*

### Blocking defect 2 — decide whether the Option D companion wiring ships in the same batch

§5.1 recommends wiring `benchmark.py:975`'s `verify_calibration_witness` call
with both `decode_identity=` and the new `model_config_identity=` in the same
batch, since it is the one place §3.2 shows the gap is concretely exercised.
A reviewer may instead choose to ship the schema change alone and defer the
call-site wiring — both are coherent; the design's default is "ship together"
because they touch the same file family and the wiring is what makes §9.2's
PA6 meaningful rather than aspirational.

### Blocking defect 3 — confirm the N6/N7 fold-in decision (§10.2) before implementation, not during it

Folding N6/N7 into the same `CALIBRATION_WITNESS_VERSION` bump is cheap *if*
decided now; deciding it mid-implementation risks a second undisclosed schema
change. This design takes no position beyond flagging it — a reviewer should
say yes or no before the collateral matrix (§9.4) is finalized, since folding
them in would add fields to the same digest this design is already touching.

### Nature of the changes

All three are **specification-only** — naming and scope-boundary decisions.
None touches the dependency graph, the model-identity architecture, the cache
key, the contract, or the migration profile, all of which are locked and
independently re-derived above. With §11.1–§11.3 ratified, R31 (as scoped:
Option A + Option D wiring, N6/N7 excluded unless a reviewer says otherwise)
is implementation-ready.
