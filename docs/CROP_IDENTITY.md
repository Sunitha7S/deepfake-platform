# Crop-geometry identity — architecture report and Batch 21 (R25)

> **Prelude.** This is the architecture-and-proof-strategy deliverable the Batch 21
> brief asked for. **No identity composition is changed by this document.** The
> executable study (`tests/crop_identity_study.py`) is read-only; the proof-test
> plan in §9 is a specification, not a passing suite, because its pass-after half
> requires the implementation this document defers.
>
> **SUPERSEDED BY §15 (Batch 22) — do not implement from §1–§14.** The
> implementation revised two decisions in this plan: `input_size` was folded into
> the existing `-w` fingerprint dict, **not** added as a new `-i<ifp>` token, and
> the shipped identity is
> `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9` — **not** the projected
> `stub-0-w7f45fada-iecf7148f+c.739c3fec.g.ae232cba.f.5589e93e` that §3, §5, §7
> and §10 still name. Wherever the body says `-i<ifp>`, `-iecf7148f` or
> `.f.5589e93e`, §15 is authoritative. The projections are left unedited only as
> a record of what was considered.
>
> **Batch 23 (R29)** moved the `.f.` segment again — `251d22e9` → the composite
> `e629642c` (model identity `…f.e629642c`) — by folding the video
> frame-selection / pooling policy into it via `selection_identity()`. The
> *face* policy identity is still `251d22e9`. §16 and
> `docs/VIDEO_SELECTION_IDENTITY.md` are authoritative for B23.

**State at report.** 48 suites, 1217 tests, 0 failures. `CONTRACT_VERSION` 1.0.0,
`RUN_SCHEMA_VERSION` 1.2.0, `CALIBRATION_WITNESS_VERSION` 1.2.0.
`STUB_ENGINE_VERSION = stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387.d.a97ccb68`.
Closed through B20: R1–R5, R7, R8, R11, R12, R14, R15, R17, R18 (false-claim
half), R19, R20, R21, R22, R27. Open: R6-a, R6-b, R9, R10, R13, R16, R23, R24,
**R25**, R26, residuals D8–D10.

---

## 0. The answer, up front

**R25 exists exactly as Batch 15 documented it**, verified against shipped code
this batch (§1). Three preprocessing knobs decide what pixels the model scores
and none of them appears in any identity string:

| Knob | Where it lives | Consumed by | In an identity? |
|---|---|---|---|
| 1 | `ModelSpec.input_size` (`inference/models.py:137`) | `prepare_image(size=…)`, `extract_frames(target_size=…)` | **no** |
| 2 | the interpolation rule in `resize` (`inference/preprocess.py:237`) | image crop **and** every video frame | **no** |
| 3 | the clip/centre rule in `crop_face` (`inference/preprocess.py:708–719`) | image crop | **no** — `crop_shape` names only how `side` is derived |

**Recommendation: close knobs 1 and 2 now; carry knob 3 with them.** Knobs 1 and
2 have no coupling to the SCRFD geometry change and are the R12/R27 pattern
exactly — a score-moving input outside the cache key. Knob 3's *value* will
change when R23 lands, but the field is structural (a name held to behaviour by a
test) and folding it in now costs one extra cache generation on that one field —
the same trade B20 accepted for R27 against the unscheduled video face stage.

**Migration profile:** one cache generation. `STUB_ENGINE_VERSION` moves.
**No** `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` /
OpenAPI / `schema_out` change. **No** benchmark recalibration — shipped pixels are
byte-identical, so no verdict moves (§5, §7).

---

## 1. Fresh derivation from shipped code (question 1)

Traced every identity-composition site. Verified files: `inference/models.py`,
`inference/preprocess.py`, `inference/image_pipeline.py`,
`inference/video_pipeline.py`, `contracts/version.py`, `backend/config.py`,
`workers/io.py`, `evaluation/benchmark.py`, `evaluation/calibration_protocol.py`.

### 1.1 The identity graph as shipped

```
ModelSpec.effective_version                        models.py:274
  = compose_model_version(                         contracts/version.py:261
      weights      = weights_identity              models.py:256  -> {version, branch_versions, branch_weights}
      calibration  = calibration_fingerprint       models.py:205  -> {temperature, fitted_on, n_samples, threshold, band_lo, band_hi, target_fpr}
      gate         = MediaCaps.gate_fingerprint    preprocess.py:71  -> {max_bytes, max_pixels, max_duration_s, max_frames, min_short_edge, min_face_px}
      face_selection = FaceSelectionPolicy.identity preprocess.py:510 -> {detector, detector_asset, scale_factor, min_neighbors, detect_min_px, ordering, crop_margin, crop_shape}
  )

pipeline.model_version                             image_pipeline.py:86 / video_pipeline.py:135
  = compose_runtime_version(model=effective_version, decode=decode_identity())

content cache key                                  workers/io.py:93
  = "results/by-content/{sha256}/{model_version}.json"          <- the runtime version, verbatim
API short-circuit                                  backend/service.py:98
  = sink.find_by_content(body.sha256, settings.model_version)   <- STUB_ENGINE_VERSION literal
idempotency key                                    backend/service.py:227
  = f"{sha256}:{model_version}"
```

### 1.2 Where the three knobs enter — and do not

- **`input_size`** is a bare `ModelSpec` field. `weights_identity`
  (`models.py:256–271`) hashes `version`, an optional `branch_versions` digest,
  and `branch_weights` — **not `input_size`**. It flows unhashed into
  `ImagePipeline._crop_and_score` → `prepare_image(crop, size=self._spec.input_size)`
  (`image_pipeline.py:198`) and `VideoPipeline` →
  `extract_frames(..., target_size=self._spec.input_size)` (`video_pipeline.py:176`).

- **the interpolation rule.** `resize` (`preprocess.py:229–238`):
  `interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR`.
  Neither the rule string nor the `INTER_AREA` choice is a field of
  `FaceSelectionPolicy`; `FaceSelectionPolicy.identity` (`preprocess.py:510–521`)
  lists eight keys and none is an interpolation. Shared by the image crop
  (`prepare_image` → `resize`) and every video frame (`extract_frames` →
  `resize(rgb, target_size)`, `preprocess.py:946`).

- **the clip/centre rule.** `crop_face` (`preprocess.py:708–719`) computes a
  square `side` then `x1 = min(w, x0 + side)` / `y1 = min(h, y0 + side)` and
  slices — the window is **intersected** with the image. `crop_shape =
  "square-longest-edge"` (`preprocess.py:577`) is a `FaceSelectionPolicy` field
  and is in the identity, but its docstring (`preprocess.py:496`) defines it as
  `max(width, height) * margin, centred on the box` — it says nothing about the
  intersect step. A pad-to-square implementation is equally "square-longest-edge".

### 1.3 Executed confirmation

`tests/crop_identity_study.py` §1, on this build:

```
effective_version : stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387
runtime version   : stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387.d.a97ccb68

knob 1  input_size 384 vs 224   tensor (1,384,384,3) vs (1,224,224,3)
        stub p_fake 0.423886 vs 0.660171   |dp|=0.236285
        weights_identity  stub-0-w7f45fada  ==  stub-0-w7f45fada     effective_version identical: True
knob 2  INTER_AREA vs INTER_LINEAR (500x300 -> 384)
        mean |pixel delta| 0.069/255   stub |dp|=0.074700
        an interpolation key is in FaceSelectionPolicy.identity: False
knob 3  intersect-image vs pad-to-square (right-border box)
        clipped crop (260, 230)  anisotropy 1.1304   padded (260, 260)
        stub |dp|=0.204434   crop_shape == 'square-longest-edge' for both
```

The four Batch-15 tripwires (`test_input_size_is_in_no_identity_string`,
`test_the_interpolation_policy_is_in_no_identity_string`,
`test_the_clipping_rule_is_in_no_identity_string`,
`test_two_deployments_can_agree_on_every_identity_and_differ`) are **green today**
— they assert the hole is open. They are the fail-before baseline (§9).

**Answer to question 1: yes, R25 exists exactly as documented. Nothing since
Batch 15 has touched these paths.**

---

## 2. Which identity should own `input_size` (question 2)

**The weights identity.** Argued with Batch 9's / Batch 12's test — *is this code
the deployment ships, or environment it inherits?* — plus a checkpoint-coupling
test:

- A trained checkpoint has an input resolution baked into its first layer.
  Running a 224-native checkpoint at `input_size=384`, or vice-versa, is a
  **misconfiguration of the weights**, and `weights_identity` is where a weights
  misconfiguration should surface (`models.py:256` already exists to carry
  "which checkpoint, composed with its branch structure").
- It is **not** face-selection: `input_size` is not a property of *which face* or
  *how it is cropped*; it is the model's whole field of view. Folding it into
  `.f.` would put a weights concern in the selection segment.
- It is **not** its own segment: `compose_model_version` already has four
  fingerprints and the format comment (`version.py:47–52`) treats readability as
  load-bearing. A fifth segment for one integer that changes with the checkpoint
  is noise; appending it to the weights sub-identity (`stub-0-w<wfp>-i<ifp>`)
  keeps it inside the segment that already owns the checkpoint.

**Mechanism:** extend `ModelSpec.weights_identity` to append
`-i{fingerprint({"input_size": self.input_size})}`. Projected on this build
(study §4): `stub-0-w7f45fada` → `stub-0-w7f45fada-iecf7148f`.

**Downstream:** `verify_version_alignment` (`workers/engine_registry.py`) already
fails startup when the configured `model_version` ≠ what engines report, so a
deployment that changes `input_size` without regenerating the literal fails to
start — the decode-identity mechanism from B9, reused.

---

## 3. Which identity should own the interpolation policy (question 3)

**The face-selection identity, for the *choice*; the decode identity already
owns the *implementation*.** This is Batch 15 §10's split, re-derived and
confirmed:

- `cv2`'s implementation of `INTER_AREA` is pinned by `cv2.__version__`, which is
  already inside `DecodeEnvironment.identity` (`inference/decode_env.py`). An
  OpenCV upgrade that changes the kernel moves `.d.` today. **Unchanged.**
- The *choice* to call `INTER_AREA` on shrink and `INTER_LINEAR` otherwise, and
  the `h > size or w > size` predicate, are **shipped code** — the same category
  as `crop_margin` and the ordering rule, both already in `.f.`. They belong in
  `FaceSelectionPolicy` as a named field, e.g.
  `interpolation_rule = "area-if-either-axis-shrinks-else-linear"`.
- It is **not** a new segment: `FaceSelectionPolicy` already owns "what pixels
  the model sees" (its own class docstring, `preprocess.py:434`), same owner,
  same lifecycle (SCRFD moves the whole policy at once).

**Mechanism:** add `interpolation_rule: str` to `FaceSelectionPolicy` (no
default — the class forbids defaults, B12), include it in `.identity`,
`.to_dict`, `.differences`, and hold it to `resize`'s branch by a re-derivation
test (the `ORDERING_RULE` pattern, `preprocess.py:586`). Projected `.f.` on this
build (study §4, with knob 3): `19559387` → `5589e93e`.

> **Superseded by §15.** Shipped value is
> `interpolation_rule = "area-if-either-axis-exceeds-target-else-linear"` (names
> the current `or` behaviour verbatim; R28 deferred) and the shipped `.f.` is
> `251d22e9`, not `5589e93e` — the projection above assumed a different label and
> the `-i` design.

**R28 (candidate finding, recorded not folded).** `resize`'s predicate is
`h > size or w > size`: a crop wider than `size` but shorter (500×300 → 384)
selects `INTER_AREA` for **both** axes, low-passing the 300→384 axis it is
*enlarging* — in a function whose docstring picks `INTER_AREA` because forensic
signal is high-frequency. Study §3 confirms it on this build (mean |AREA −
LINEAR| on the enlarged axis = 0.040/255). This is a *correctness* smell in the
rule, distinct from R25's "not fingerprinted". The R25 batch should fix the
predicate to `h > size and w > size` **or** switch to a single-factor rule while
it is naming the rule anyway — but it is its own finding with its own
fail-before/after, not part of R25.

---

## 4. Clip behaviour: face-selection identity or a separate geometry identity? (question 4)

**Face-selection identity.** A separate geometry segment was considered and
rejected on Batch 12's own argument against folding selection into the gate:

| Option | Argument | Verdict |
|---|---|---|
| **`.f.` (face-selection)** | `crop_margin` and `crop_shape` are already here; the clip rule is the same *kind* of thing — shipped code deciding what pixels the model sees — with the same owner and the same lifecycle (SCRFD moves margin, shape, clip and fill together). | **yes** |
| new `.c.` geometry segment | Splits one decision — "how is the crop produced" — across `crop_margin`/`crop_shape` in `.f.` and the clip rule in `.c.`. Batch 12 rejected the gate/selection split for having *different* owners; here the owners are identical, so a split is pure fragmentation. | no |
| decode identity | The clip rule is not environment; it is shipped code. | no |
| nowhere | Study §1: `|dp|` 0.20 (stub UB) on an ordinary right-border box. It moves verdicts. | no |

**Mechanism:** add `crop_clip_rule: str` to `FaceSelectionPolicy`, value
`"intersect-image"` today. When R23 lands it becomes `"pad-replicate"` (or
whichever fill wins) and `.f.` moves a second time — the accepted double-spend,
scoped to this one field. Held to `crop_face`'s behaviour by a re-derivation
test.

**This is the only part of R25 with SCRFD coupling.** Knobs 1 and 2 are closed
permanently by Batch 21; knob 3's field is closed now and its *value* is revised
once by the SCRFD batch.

---

## 5. Migrations required (question 5)

| Axis | Change? | Proof / reason |
|---|---|---|
| `CONTRACT_VERSION` | **no** | `ModelInfo.version` stays `str`; no `EvidenceBundle` field added/retyped; every stored bundle round-trips byte-identically; a version-string format extension is not a client migration (B9/B12/B14 precedent). |
| `RUN_SCHEMA_VERSION` | **no** | `BenchmarkRun.effective_model_version` already exists and already carries the `.f.` segment (`evaluation/benchmark.py:157`). **No new field** — contrast R12/R27, which added `BenchmarkRun.decode_identity` and needed 1.1.0→1.2.0. |
| `CALIBRATION_WITNESS_VERSION` | **no** | `compute_witness` digests protocol/role/corpus/n/temperature/nll/threshold/margin/target_fpr/selection_witness/split_lock_hash/schema (+decode for 1.2.0) — **not** `effective_model_version`, `.f.`, or `input_size` (`calibration_protocol.py:250–266`). Nothing re-digests; the schema-version-as-parameter dispatch (B11) is untouched. |
| OpenAPI / `schema_out` | **no** | No field or shape change; the structural signature ignores prose (`test_signature_ignores_prose_but_not_structure`). |
| `STUB_ENGINE_VERSION` (`backend/config.py:41`) | **yes** | ~~`-i<ifp>` added to the weights sub-identity, `.f.` moves `19559387`→`5589e93e`. Projected model identity: `stub-0-w7f45fada-iecf7148f+c.739c3fec.g.ae232cba.f.5589e93e`.~~ **Superseded by §15:** `input_size` folded into the existing `-w` dict (no `-i` token, format unchanged); shipped identity `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9`, literal `…f.251d22e9.d.a97ccb68`. `parse_model_version` round-trips it unchanged. |
| Live version pins | **yes** | ~~`test_batch_15_changed_no_geometry`~~ became `test_crop_geometry.py::test_batch_22_moved_the_identity_not_the_pixels` (asserts `POLICY.identity == "251d22e9"`); `_EXPECTED_EFFECTIVE` in `test_branch_b_semantics.py` / `test_headline_and_face_semantics.py`, `MODEL_ID` in `test_decode_identity_completeness.py`, `test_every_policy_field_moves_the_identity` — all updated in B22. New suite `tests/test_crop_identity.py`. |
| Stored bundles / witnesses / split locks | **readable** | strings are stored verbatim and verified under their own era's rule; `parse_model_version` gains one era and stays total (asserted). |
| Back-fill | **refused** | B9/B11 precedent: writing this geometry into a historical artifact asserts a fact nobody observed. Historical rows keep the coarser identity forever. |

---

## 6. Cache generation (question 6)

**Exactly one, image + video namespaces, one-time full miss per file.**

- `effective_version` moves → `pipeline.model_version` moves → the content-cache
  key `results/by-content/{sha256}/{model_version}.json` moves. Every prior entry
  becomes unreachable by content; still addressable by job id. No stale entry is
  served under a new name — `compose_model_version`'s marker guards
  (`version.py:280–299`) make the namespaces provably disjoint, and
  `test_run_identity.py`'s real-sink proof for the decode segment applies
  unchanged in shape.
- Both `stub-image` and `stub-video` specs carry `input_size=384`
  (`models.py:533–534`) and both pipelines compose the same `.f.`, so **both**
  namespaces generate once. The video namespace generating for a knob it
  half-shares (1 and 2, not 3) is compute cost, not correctness — same as B12's
  `.f.` on the video engine.
- Idempotency keys (`{sha256}:{model_version}`) re-derive per submission;
  in-flight jobs are unaffected.

**Cost:** one re-analysis per distinct previously-cached file. No data loss.
Precedented exactly by B9 §6.2, B14 and B20.

---

## 7. Benchmark protections that change (question 7)

`regression_gate` (`evaluation/benchmark.py:783`) compares
`effective_model_version` and `decode_identity` **separately**:

- **Today:** two runs that differ only in `input_size` / interpolation / clip
  rule have **equal** `effective_model_version`. The gate does not refuse — it
  adds a *note* (`benchmark.py:816–821`): *"both runs report the same effective
  model version; if the models really differ, something that changes a score is
  missing from the version composition."* That note is R25 speaking at the
  benchmark layer, and it is advisory only.
- **After:** the two runs carry different `effective_model_version` →
  `regression_gate` short-circuits with *"runs were measured on different
  datasets"*-shape refusal? **No** — `effective_model_version` inequality does
  **not** short-circuit the gate (only `dataset_fingerprint` and `decode_identity`
  do, `benchmark.py:803–814`). It removes the "same version" note and lets the
  AUC/FPR comparison run. So a pre-B21 vs post-B21 benchmark pair **still
  compares numerically** — and correctly, because under the shipped detector the
  numbers are identical (§5). The improvement is: a genuine `input_size` mistake
  between two runs now shows as two model identities instead of hiding behind
  one.
- `BenchmarkReport.validate()` (`benchmark.py:951`) refuses a report pairing two
  runs with different `effective_model_version` — so a report accidentally
  pairing a 224 run with a 384 run is now refused where it was silently accepted.

**No `max_*` tolerance changes. No recalibration.** The shipped operating point
does not move because the shipped pixels do not move.

---

## 8. Witness protections that change (question 8)

`CalibrationWitness` (`evaluation/calibration_protocol.py`):

- **Today:** a temperature fitted under `input_size=384` and applied under
  `input_size=224` produces a witness that *happens* to differ, because the
  logits differ, so `quantize(temperature)` differs, so the digest differs —
  **detection by accident of what the digest covers**, identical in shape to the
  decoder half of R12 pre-B11 (`REPRODUCIBILITY_AUDIT.md` B11 "the calibration
  half is weaker than it looks"). Two fits that converge to the same `quantize`d
  temperature under two `input_size`s produce **byte-identical witnesses**.
- **After (recommended):** no change to the witness digest. The witness already
  binds `selection_witness` (the corpus) and, at 1.2.0, `decode_identity`. It
  does **not** bind `effective_model_version` and adding it would be a
  `CALIBRATION_WITNESS_VERSION` bump the brief forbids and B21 does not need.
  The `input_size` protection rides on `effective_model_version` moving, which
  the cache key and `BenchmarkReport.validate` both check.
- **Recorded residual (D-class):** the calibration witness has no first-class
  binding to preprocessing geometry, only the incidental logit-drift detection.
  Closing that means threading `effective_model_version` into the witness digest
  — a witness-version bump, its own batch, sequence it with any future witness
  work. Sibling of the "equal-length logits from the wrong corpus" residual
  (B6).

---

## 9. What can be closed immediately (question 9)

| Part | Close now? | Why |
|---|---|---|
| **knob 1 — `input_size` → weights identity** | **yes** | Zero SCRFD coupling. Pure R12/R27 pattern. `verify_version_alignment` catches the misconfig at startup. |
| **knob 2 — interpolation rule → `.f.`** | **yes** | Zero SCRFD coupling (B15 §10 itself places it "alongside the rest of the shipped preprocessing"). |
| **knob 3 — clip rule → `.f.` (`"intersect-image"`)** | **yes, as a structural field** | The field is closed now; its *value* is revised once by the SCRFD batch. One extra cache generation on that field — the B20/R27 trade. |
| R28 — the `or` predicate | **fix or explicitly pin in the same batch** | The batch is already naming the rule; leaving a known mis-filter unnamed beside it is the B14 "two-pixel margin" mistake. |
| the calibration-witness geometry binding | **no** | Needs a `CALIBRATION_WITNESS_VERSION` bump; own batch. Recorded as a residual. |

**All of R25 is closable in one batch** with a one-cache-generation migration and
no contract/schema/witness bump. The only deferred piece is knob 3's *final
value*, which the SCRFD batch sets.

### 9.1 Proof-test plan — `tests/test_crop_identity.py` (spec, ~20 tests)

**Named doubles.**
`LegacyGeometryBlindIdentity` — `ModelSpec.weights_identity` +
`FaceSelectionPolicy.identity` as of B20, verbatim; a test pins it to
`stub-0-w7f45fada` / `19559387` on this build and fails if the double drifts.

**Fail-before (flip the four B15 tripwires in `test_crop_geometry.py`):**

| B15 tripwire (asserts hole open) | B21 replacement (asserts hole closed) |
|---|---|
| `test_input_size_is_in_no_identity_string` | `test_input_size_is_in_the_weights_identity` |
| `test_the_interpolation_policy_is_in_no_identity_string` | `test_the_interpolation_rule_is_in_the_face_selection_identity` |
| `test_the_clipping_rule_is_in_no_identity_string` | `test_the_clip_rule_is_in_the_face_selection_identity` |
| `test_two_deployments_can_agree_on_every_identity_and_differ` | `test_two_deployments_that_differ_on_geometry_now_get_different_identities` |

Each B15 tripwire already contains the fail-before assertion inline (e.g.
`test_input_size_is_in_no_identity_string` builds two specs and asserts the
identities are equal). B21 keeps the construction, inverts the assertion, and
adds the `|dp|` measurement from the study as context.

**Pass-after.**
- each knob perturbed → `effective_version` moves; shipped values → model
  identity prefix moves **exactly once** vs the recorded B20 string.
- `FaceSelectionPolicy` still rejects construction without every field (B12
  invariant) — new fields required too.
- `interpolation_rule` name re-derived against `resize`'s branch (ORDERING_RULE
  pattern); `crop_clip_rule` name re-derived against `crop_face`'s slice.
- `verify_version_alignment` fails startup on an `input_size` mismatch.
- `regression_gate`: a pre/post-B21 pair carries two model identities; the
  numeric comparison still runs; `BenchmarkReport.validate` refuses the pair.
- old `CalibrationWitness` / `BenchmarkRun` JSON still loads (strings stored, not
  recomputed).
- `parse_model_version` total on pre-B9 / B9–B13 / B14–B20 / B21 eras.
- `CONTRACT_VERSION`, `RUN_SCHEMA_VERSION`, `CALIBRATION_WITNESS_VERSION`,
  OpenAPI snapshot, `schema_out/*` each asserted unchanged (render-and-diff, the
  B18 method).
- **shipped pixels byte-identical**: `crop_face` and `resize` on the shipped
  config produce output identical to `LegacyClipToBounds` / the current `resize`
  — no verdict moves.
- R28: `test_the_resize_rule_no_longer_area_filters_an_enlarged_axis` (if fixed)
  or `test_the_resize_or_predicate_is_pinned_with_its_consequence` (if only
  named).

**`test_batch_15_changed_no_geometry` (`test_crop_geometry.py:630`)** asserts
`POLICY.identity == "19559387"` and `model_identity_of(STUB_ENGINE_VERSION) ==
"…f.19559387"`. Both break. Rewrite to `test_batch_21_moved_the_identity_not_the_
pixels`: assert the new `.f.`, the new `STUB_ENGINE_VERSION`, **and** that the
shipped crop/resize output is byte-unchanged.

### 9.2 Executable study — `tests/crop_identity_study.py` (delivered, runs today)

Six sections, all read-only, all executed on this build:
1. the hole — identical identity, different tensor, per knob
2. magnitude — stub `|dp|` and decisions-changed vs the shipped band (upper bound)
3. the `or` bug (R28)
4. projected post-fix identity — `fingerprint` with the extra keys; round-trip
5. shipped pixels unchanged
6. video path shares knobs 1 and 2

---

## 10. Residual if only part of R25 is fixed (question 10)

| Scenario | Residual |
|---|---|
| **Knobs 1+2 only, knob 3 deferred to SCRFD** | A clip-vs-pad build difference still shares an identity (study §1 knob 3: stub `|dp|` 0.20 on a right-border box). This is **R23's live distortion half** wearing R25's clothes — the reproducibility complement to a correctness defect that is itself unfixable output-identically today. Recorded, tripwire kept (`test_the_clip_rule_is_in_no_identity_string` stays). |
| **Knob 3 field added now, value revised at SCRFD** | One extra `.f.` cache generation when R23 lands. No correctness residual. **Recommended.** |
| **All three, R28 left unnamed** | The `resize` predicate still low-passes an enlarged axis; a future maintainer "tidying" the rule moves every non-square-crop score with nothing pinning the consequence. The B14 two-pixel-margin mistake repeated. |
| **All three + R28, witness binding deferred** | The calibration witness detects an `input_size`-induced logit shift only incidentally; a converged-temperature collision produces byte-identical witnesses across two `input_size`s. Sibling of B6's equal-length-wrong-corpus residual. Needs a witness-version batch. |
| **D8–D10 unchanged** | Out of R25's scope — those are decode-identity granularity residuals from B20. A same-`cv2.__version__` OpenCV rebuild that changes `INTER_AREA`'s kernel is D9's sibling and still shares an identity after R25. Recorded. |

---

## 11. Preserve-and-refute

**Batch 15 §10 / Batch 16 §5 recorded:** *"Closing R25 moves `effective_version`
and dumps the cache, and doing that for a geometry change that has not happened
yet spends the migration twice — land the fingerprint extension in the same batch
as the geometry change it is meant to protect."*

**Preserved:** the mechanism is exactly right — closing R25 is one cache
generation, and R23's geometry fix is another.

**Refuted, in part:**

1. The claim treats R25 as one thing. It is three knobs with different SCRFD
   coupling (§2–4, study §1/§4/§6). `input_size` and the interpolation *choice*
   are never touched by the SCRFD geometry change — B15 §10 says so itself when
   it places them in the weights and preprocessing identities respectively. For
   those two, the "spend it twice" cost is **zero**: they move once, at B21, and
   never again.
2. The "spend it twice" objection is verbatim the objection **Batch 19** raised
   against closing R27 (*"forces a global cache generation … deferred to a batch
   whose brief permits it"*). **Batch 20** overrode it because the reproducibility
   hole was live and the guarded migration (video face stage) had no schedule.
   The SCRFD migration has no schedule either (architecture FROZEN). Same posture,
   same override.
3. Even for knob 3, the premium is one extra cache generation **on one field**,
   not wasted correctness work: the field is structural (a name held to
   behaviour), and R23 changes only its string value inside the same `.f.`
   segment.

**Net:** B15's sequencing recommendation stands for knob 3's *value* and is
withdrawn for knobs 1, 2 and knob 3's *field*. This is a refinement of the B15
finding, not a reversal — B15 correctly identified all three knobs and their
homes; it over-bundled the migration decision.

**Not reopened:** R23 (still characterised, still no output-identical fix, still
SCRFD for the fix), R19/R17 (the `.f.` segment they built is the one B21
extends), R27 (the `.d.` segment is untouched).

---

## 12. Audit update section (for `REPRODUCIBILITY_AUDIT.md`)

> # Batch 21 — R25 architecture and proof strategy (no implementation)
>
> **Deliverable batch, no code changed.** `CONTRACT_VERSION` 1.0.0,
> `RUN_SCHEMA_VERSION` 1.2.0, `CALIBRATION_WITNESS_VERSION` 1.2.0,
> `STUB_ENGINE_VERSION` all unmoved. Full trace, executed study and proof plan:
> [`docs/CROP_IDENTITY.md`](docs/CROP_IDENTITY.md). Read-only study:
> `tests/crop_identity_study.py` (6 sections, runs on this build).
>
> **R25 re-derived from shipped code and confirmed exactly as Batch 15 recorded
> it** — `input_size` is outside `weights_identity` (`models.py:256`), the
> interpolation rule is outside `FaceSelectionPolicy.identity`
> (`preprocess.py:510`), the clip rule is described by neither `crop_shape` nor
> any test. The four Batch-15 tripwires are green (they assert the hole is open).
> Executed: two deployments agreeing on every identity string
> (`stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387.d.a97ccb68`) hand the model
> different tensors and, on the stub upper bound, different verdicts — |dp| 0.24
> (input_size 384→224), 0.07 (interpolation), 0.20 (clip rule at a right border).
>
> **Decision: R25 is fully closable in one batch**, one cache generation, no
> contract / schema / witness bump. `input_size` → weights identity
> (`-i<fp>` suffix); interpolation rule and clip rule → new `FaceSelectionPolicy`
> fields. Projected identity `stub-0-w7f45fada-iecf7148f+c.739c3fec.g.ae232cba.f.5589e93e`
> — `parse_model_version` round-trips, `model_identity_of` recovers it,
> calibration and gate fingerprints unchanged. Shipped pixels byte-identical → no
> verdict moves → **no benchmark recalibration** (contrast R23).
>
> **Preserve-and-refute:** Batch 15's "land it with SCRFD" holds for the clip
> rule's *value* only; knobs 1 and 2 and the clip-rule *field* have no SCRFD
> coupling and are the R12/R27 pattern. The "spend the cache generation twice"
> objection is the one Batch 19 raised against R27 and Batch 20 overrode; same
> posture applies.
>
> **New finding R28 (candidate, minor):** `preprocess.resize`'s predicate
> `h > size or w > size` selects `INTER_AREA` for both axes when only one
> shrinks, low-passing an axis it is enlarging (measured 0.040/255 on the
> enlarged axis). Correctness smell in the rule, distinct from R25's fingerprint
> gap. Fix the predicate or pin it with its consequence in the R25 batch.
>
> **Recorded residual:** the calibration witness binds preprocessing geometry
> only incidentally (via logit drift). First-class binding needs a
> `CALIBRATION_WITNESS_VERSION` bump; sequence with future witness work.
>
> **Still open after Batch 21:** R6-a, R6-b, R9, R10, R13, R16, R23, R24,
> **R25 (architecture done, implementation is Batch 22)**, R26, R28, D8–D10.

---

## 13. Re-running this

```bash
python tests/crop_identity_study.py    # the six sections above
python tests/test_crop_geometry.py     # the four R25 tripwires (green == hole open)
grep -rn 19559387 tests/ backend/      # every live pin the implementation must update
```

## 14. Where things live

| what | where |
|---|---|
| `input_size`, `weights_identity`, `effective_version` | `inference/models.py:124`, `256`, `274` |
| `resize` and its interpolation rule | `inference/preprocess.py:229` |
| `crop_face` and the clip rule | `inference/preprocess.py:664` |
| `FaceSelectionPolicy` and `.identity` | `inference/preprocess.py:433`, `499` |
| version composition and parsing | `contracts/version.py:261`, `402` |
| the content cache key | `workers/io.py:85`, `93` |
| `regression_gate` | `evaluation/benchmark.py:783` |
| `CalibrationWitness.compute_witness` | `evaluation/calibration_protocol.py:212` |
| the four fail-before tripwires | `tests/test_crop_geometry.py:445–528` |
| the batch-15 no-geometry pin to rewrite | `tests/test_crop_geometry.py:630` |
| the version literal to bump | `backend/config.py:41` |

---

## 15. Outcome — Batch 22 (R25 CLOSED)

Implemented per `docs/CROP_IDENTITY_B22_PLAN.md` (B22-R2). **Output-identical, one
cache generation, no contract / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION`
/ OpenAPI / `schema_out` change.**

**Production changes.**

| Knob | Change | Closure strength |
|---|---|---|
| 1 `input_size` | folded into `ModelSpec.weights_identity`'s `-w` fingerprint dict (not a new `-i` token — `parse_model_version` untouched, no format change) | **strong** — `prepare_image`/`extract_frames` read the same attribute |
| 2 interpolation rule | `FaceSelectionPolicy.interpolation_rule = "area-if-either-axis-exceeds-target-else-linear"` (names the current `or` behaviour verbatim) | label — pinned by `test_the_interpolation_rule_name_matches_resize` |
| 3 clip rule | `FaceSelectionPolicy.crop_clip_rule = "intersect-image"` | label — pinned by `test_the_clip_rule_name_matches_crop_face` |

Both new fields have **no default** (`test_the_policy_has_no_defaults`); set only
in `default_face_selection()`. Added to `.identity`, `.to_dict`, `.differences`.

**Regenerated identity values.**

| | pre-B22 | B22 |
|---|---|---|
| `default_face_selection().identity` (`.f.`) | `19559387` | `251d22e9` |
| `ModelSpec(...).weights_identity` | `stub-0-w7f45fada` | `stub-0-wacdd9240` |
| model identity (`effective_version`) | `stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387` | `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9` |
| `STUB_ENGINE_VERSION` | `…f.19559387.d.a97ccb68` | `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.251d22e9.d.a97ccb68` |
| `fingerprint({"input_size": 384})` component | — | `ecf7148f` |
| `calibration` / `gate` / `.d.` segments | `739c3fec` / `ae232cba` / `a97ccb68` | **unchanged** |

**Tests.** New suite `tests/test_crop_identity.py` (12 tests) holds the positive
identity-movement tests (replacing the two Batch-15 tripwires that did not fail
after the fix), the two re-derivation tests, `test_the_shipped_pixels_did_not_move`,
the migration proof, era-totality of `parse_model_version`, old-artifact load,
alignment-on-mismatch, and the benchmark-boundary behaviour. `tests/run_all.py`
registers it as **S8-B22 → 49 suites**. `test_batch_15_changed_no_geometry` →
`test_batch_22_moved_the_identity_not_the_pixels`. `_EXPECTED_EFFECTIVE` /
`MODEL_ID` collateral pins updated in `test_branch_b_semantics.py`,
`test_headline_and_face_semantics.py`, `test_decode_identity_completeness.py`.
`test_every_policy_field_moves_the_identity` extended with the two new fields.

**R28 — NOT closed, deferred.** `resize`'s `h > size or w > size` predicate
area-filters an axis it enlarges. Fixing it moves shipped pixels, so it is a
separate finding for the next batch that already moves `.f.` (R23/SCRFD or R16).
B22 recorded the behaviour in `interpolation_rule` and pinned it. **Batch 23 moved
`.f.` but kept R28 out** — a pixel change would forfeit B23's output-identical
guarantee; R28 rides the SCRFD / R23 batch, which is already moving pixels.

## 16. Batch 23 (R29) — the `.f.` segment became a composite

Batch 23 extended the `.f.` segment to carry the **video** selection policy as
well: `selection_identity(face, video) = fingerprint({"face": face.identity,
"video": video.identity})`, computed by both pipelines. The honest-named reader
`selection_identity_of` (B23) — and its permanent alias `face_selection_identity_of`
— therefore returns this *composite* for a video-era (B23+) string, not the face
policy alone. `face_selection_identity()` in `inference.preprocess` stays
face-only for callers that want just that. `default_face_selection().identity` is unchanged (`251d22e9`); the
`.f.` fingerprint moved `251d22e9` → `e629642c` and the model identity /
`STUB_ENGINE_VERSION` moved with it. `contracts/version.py` is untouched — `.f.`
is still one 8-hex fingerprint. See `docs/VIDEO_SELECTION_IDENTITY.md`.
