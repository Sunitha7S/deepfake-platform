# R31 — Specification Resolution and Implementation Plan

**Type:** design/specification resolution. No code in this task.
**Follows:** `docs/R31_DESIGN.md` §11, which gated APPROVE WITH CHANGES on three
named decisions. **Supersedes** `docs/R31_DESIGN.md` §5.1 and §8 wherever this
document's re-derivation contradicts them (flagged explicitly at each point) —
`R31_DESIGN.md` remains the authoritative record of the *finding* (§1–§4, live
collision study) and of the *rejected* options (§5 Options B/C); this document
is authoritative for the *scope actually being built*.
**Method:** every claim re-derived from source, per the instruction to not
trust the prior document's own conclusions on the three open decisions.

---

## Decision 1 — Field name for the witness-carried model-config identity

### Re-derivation

`grep -rn "model_config\|config_identity\|model_identity\b" --include="*.py" .`
(live) surfaces two disqualifying facts `R31_DESIGN.md` did not check:

1. **`model_config` is a reserved Pydantic attribute name, in active use
   throughout this codebase**: `backend/config.py:151`
   (`model_config = SettingsConfigDict(...)`), `backend/deps.py:38`,
   `contracts/common.py:23`, `jobqueue/registry.py:51`, `jobqueue/types.py:27`,
   `storage/registry.py:31`, `storage/types.py:24`, `workers/types.py:23` (all
   `model_config = ConfigDict(...)`). `R31_DESIGN.md`'s proposed
   `model_config_identity` is not a literal Python name collision (it lives on
   a plain `@dataclass`, not a `BaseModel`), but it shares Pydantic's own
   reserved prefix for an unrelated concept (serialization/validation
   configuration) in a codebase where that prefix already has one meaning
   everywhere else. A grep for "model_config" would return eight false leads
   before reaching this one.
2. **`model_identity` already means something else, and something broader.**
   `inference/image_pipeline.py:80-88` (and the identical property in
   `video_pipeline.py:135-143`): `model_identity` = `self._spec.effective_version`
   — *"Weights + calibration + gate. **Which model this is.**"* This is
   `effective_version`, which **includes** calibration. The field this design
   needs is explicitly the opposite: weights + gate + selection, **excluding**
   calibration (§1.3 of `R31_DESIGN.md` — that is precisely what avoids the
   circularity). Naming the new field anything containing `model_identity`
   would put two things meaning different, almost-opposite scopes one
   underscore apart.

Neither hazard was checked before `R31_DESIGN.md` proposed the name — this is
exactly the kind of thing "re-derive, do not trust prior conclusions" is for.

### Recommendation

- **New `contracts.version`-style vocabulary is not needed for this field**
  (see Decision 2 — the parsing-based derivation `R31_DESIGN.md` §5.1 sketched
  turns out to be unsafe and is dropped). The field is populated **only** by a
  caller holding a live `ModelSpec`, exactly like `decode_identity`.
- **New `ModelSpec` property, `inference/models.py`, beside `effective_version`:**

  ```python
  @property
  def pre_calibration_identity(self) -> str:
      """Weights + gate + selection. Everything effective_version composes
      *except* calibration — the model configuration a calibration fit is
      fitted against, computed before that fit exists.
      """
      return fingerprint({
          "weights": self.weights_identity,
          "gate": self.gate_fingerprint,
          "face_selection": self.face_selection_fingerprint,
      })
  ```

- **Witness field / `compute_witness` parameter / `fit_calibration` parameter:**
  `pre_calibration_identity: str | None = None` — no `_of` suffix, matching
  `decode_identity`'s own naming (the `_of` suffix is this codebase's reserved
  pattern for an *extractor* function — `model_identity_of`, `decode_identity_of`,
  `selection_identity_of` — not for a stored value).
- **New frozenset:** `SCHEMA_VERSIONS_WITH_PRE_CALIBRATION = frozenset({"1.3.0"})`,
  exact structural twin of `SCHEMA_VERSIONS_WITH_DECODE`.

`pre_calibration_identity` collides with nothing (`grep -rn "pre_calibration"
--include="*.py" .` → no hits, live) and is self-describing: a reader who
knows `effective_version`'s composition (weights `+c.` calibration `.g.` gate
`.f.` selection) can read "pre-calibration" as "everything before the `+c.`
segment is folded in," which is literally true of the composition order in
`contracts/version.py:57` (`weights` comes first, `calibration` second).

### Affected files

`inference/models.py` (new property only — no existing property changes),
`evaluation/calibration_protocol.py` (parameter/field names, addressed fully
under the final plan, §3).

### Impact

| Axis | Impact |
|---|---|
| Migration | None — a naming choice has no migration cost of its own. |
| Version | None on its own; the field's *existence* is what bumps `CALIBRATION_WITNESS_VERSION` (Decision 3 / §2 below), not its name. |
| Cache | None — `ModelSpec.calibration_fingerprint` does not read this property (confirmed in `R31_DESIGN.md` §1.3; unaffected by naming). |
| Benchmark | None directly (Decision 2 covers whether `benchmark.py` consumes it at all). |
| Deployment | None. |

### Does this change the approved R31 scope?

**No.** This resolves an unratified detail (`R31_DESIGN.md` §11, "Blocking
defect 1") without touching architecture, migration profile, or the digest's
membership logic — only the field's name and where it is computed.

---

## Decision 2 — Does the `benchmark.py` wiring ship in the same batch?

### Re-derivation

`R31_DESIGN.md` §5.1/§8 recommended deriving the cross-check value "from
`self.primary.effective_model_version`'s weights/gate/selection prefix" inside
`BenchmarkReport.validate()`. Re-checking this against source shows **it does
not work**, for a reason not previously checked:

1. **`effective_model_version` is never parsed anywhere in `evaluation/benchmark.py`.**
   `grep -n "parse_model_version\|model_identity_of\|selection_identity_of"
   evaluation/benchmark.py` (live) → one comment mention (`:220`, about a
   *different* function avoiding re-implementing the parser), zero calls.
   `effective_model_version` is treated throughout this file as an **opaque
   caller-chosen label**, compared for equality only (`regression_gate`,
   `validate()`'s primary/in_domain check) and never decomposed.
2. **It is frequently not a parseable composed version at all** (live,
   `grep -n 'effective_model_version=' tests/test_calibration_witness.py
   tests/test_evaluation_benchmark.py`): fixtures use bare labels —
   `"v1"`, `"w-1"`, `"w-2"`, `"w-9"`, `"w"`, `"same-version"` — with no `+c.`
   segment. `contracts.version.parse_model_version` **raises** `ValueError` on
   any of these (`contracts/version.py:445-449`: *"not a composed model
   version (no `+c.` segment)"*). Deriving `pre_calibration_identity` by
   parsing `effective_model_version` inside `validate()` would **crash on the
   majority of this repository's own existing benchmark fixtures.**
3. **Adding a genuinely new field to `BenchmarkRun` (the alternative — pass
   `pre_calibration_identity` in as its own argument, exactly parallel to
   `decode_identity`) has a cost `R31_DESIGN.md` did not account for.**
   `evaluation/benchmark.py:108-123`, `RUN_SCHEMA_VERSION`'s own docstring,
   stated as a general rule and demonstrated by the `decode_identity` addition
   itself: *"Why a bump for an optional field. Nothing forces
   `decode_identity`, so an old row loads unchanged. The version still has to
   move because `from_dict` refuses unknown fields."* **Any new `BenchmarkRun`
   field requires its own `RUN_SCHEMA_VERSION` bump** (1.2.0 → 1.3.0),
   independent of and in addition to `CALIBRATION_WITNESS_VERSION`'s bump.
   `grep -rln 'RUN_SCHEMA_VERSION == "1.2.0"'` (live) finds **seven** more
   pinning test files (`test_crop_identity.py`, `test_decode_identity_
   completeness.py`, `test_dependencies.py`, `test_evaluation_identity.py`,
   `test_headline_and_face_semantics.py`, `test_run_identity.py`,
   `test_video_selection_identity.py`) that would need updating **on top of**
   the six `CALIBRATION_WITNESS_VERSION` pins §3.4 already lists.
4. **"The benchmark.py wiring" is actually two separable pieces**, conflated
   in `R31_DESIGN.md` §5.1:
   - **(D1)** Pass the **already-existing** `self.primary.decode_identity`
     into the **already-existing** `decode_identity=` parameter of
     `verify_calibration_witness` at `benchmark.py:975` (currently unused —
     `R31_DESIGN.md` §3.2). Zero new fields, zero version bumps — `BenchmarkRun.
     decode_identity` has existed since Batch 11.
   - **(D2)** Add a **new** `pre_calibration_identity` cross-check, which
     needs a **new** `BenchmarkRun` field (point 3, above) since it cannot be
     derived from the existing `effective_model_version` string (points 1–2).

### Is D1 safe to ship now?

Checked against every existing `.validate()`-exercising fixture
(`tests/test_calibration_witness.py:597-739`, `compliant_report()` and every
sibling test): **none of them passes `decode_identity=` to either
`fit_calibration` or `run_benchmark`.** Every witness and every
`BenchmarkRun` in the current test suite has `decode_identity=None`.
`verify_calibration_witness`'s cross-check (`calibration_protocol.py:748`) is
guarded by `if decode_identity is not None:` — passing
`decode_identity=self.primary.decode_identity` (which is `None` in every
existing fixture) makes the guard **not fire**, identically to today. **D1 is
provably a no-op against the entire existing test suite** and requires no
fixture changes to ship safely.

### Is D2 safe to ship now?

No, not without additional work `R31_DESIGN.md` did not scope: it needs (a) a
new `BenchmarkRun.pre_calibration_identity` field, (b) a `RUN_SCHEMA_VERSION`
bump and its own seven-file collateral list, (c) `to_dict`/`from_dict`/
`to_payload` updates, and (d) every fixture in `test_calibration_witness.py`
that currently asserts `not report.validate()` updated to supply a matching
`pre_calibration_identity` at both fit time and run-record time, or the new
check (once wired unconditionally) reports "cannot be confirmed" and breaks
`test_a_compliant_report_validates`. This is a second, independent
design-and-migration surface, not a one-line addition.

### Recommendation

**Ship D1 in this batch. Defer D2.**

- D1 closes the one *concretely exercised* gap identified in
  `R31_DESIGN.md` §3.2 (decode identity published but not cross-checked) at
  zero marginal schema cost, since the field and the parameter both already
  exist.
- D2 requires its own `RUN_SCHEMA_VERSION` migration, its own collateral
  matrix, and fixture rework with no real consumer yet to validate the values
  against (the identical "no real model to calibrate" caveat that already
  discounts R31's core urgency, stacked on a cost R31's core fix does not
  have). Bundling it now would turn a single, minimal, one-version-axis batch
  into a two-version-axis one for a protection nothing yet exercises.
  `docs/R31_DESIGN.md` §10.1 already established that R31's advantage over
  competing batches (OR1-b) is specifically its *minimal* migration cost
  (§7.1) — D2 would spend that advantage.

### Affected files

**D1 (ships now):** `evaluation/benchmark.py` (one line inside
`BenchmarkReport.validate()`, at the existing `verify_calibration_witness`
call).

**D2 (deferred, not touched in this batch):** would need
`evaluation/benchmark.py` (`BenchmarkRun` dataclass, `run_benchmark`,
`to_dict`/`from_dict`/`to_payload`, `RUN_SCHEMA_VERSION`), plus the seven
collateral test files above, plus `tests/test_calibration_witness.py` fixture
rework. None of these files are touched by this batch.

### Impact (of shipping D1 now, D2 deferred)

| Axis | D1 (this batch) | D2 (deferred) |
|---|---|---|
| Migration | None — no field added, no schema moved. | Would need its own migration analysis (a future document). |
| Version | None — `RUN_SCHEMA_VERSION` untouched. | Would move `RUN_SCHEMA_VERSION` 1.2.0 → 1.3.0. |
| Cache | None. | None (a `BenchmarkRun` field is never a cache key). |
| Benchmark | A report whose primary run and calibration witness disagree on decode identity now fails `validate()` where it silently passed before — a **new, real protection**, exercised the moment two mismatched artifacts are ever paired. | N/A — not shipped. |
| Deployment | None. | None. |

### Does this change the approved R31 scope?

**Yes, narrows it, in the safer direction.** `R31_DESIGN.md`'s gate (§11,
"Blocking defect 2") left "ship together or defer" open with a stated default
of "ship together." This resolution **reverses that default** based on a
concrete, source-verified reason (points 1–3 above) the original design did
not have. The core R31 fix (Option A, the witness digest change) is
**unchanged and unaffected** — this narrows only the companion wiring.

---

## Decision 3 — Do N6/N7 fold into R31?

### Re-derivation

`REPRODUCIBILITY_AUDIT.md:126`: *"N6/N7 — Fit-optimiser config and `np.quantile`
method are unrecorded... not in any witness or identity."* Re-derived against
the actual call graph the witness's digest depends on, rather than accepted as
stated:

**N6 — `fit_temperature`'s optimiser configuration.**
`inference/calibration.py:126-132`: `fit_temperature(logits, labels, *,
lo=0.05, hi=20.0, iterations=200)`. `evaluation/calibration_protocol.py:533`:
`fit_calibration` calls it as `fit_temperature(z, y)` — **no override, ever**.
`grep -rn "fit_temperature(" --include="*.py" .` (live) shows this is the
**only** call site anywhere in the repository (outside its own test). N6 is
real *in principle* — these values do feed the fitted `temperature`, which
*is* in the digest — but **there is currently zero variability for it to
converge from**: every fit in this codebase uses the identical
`(0.05, 20.0, 200)` triple, so there is no pair of "two different optimiser
configs" to collide, unlike R31's model-config axis, which varies on every
weights/gate/selection-changing batch (R25, R29, every future gate tightening).

**N7 — `np.quantile` method.** `grep -rn "np.quantile" --include="*.py" .`
(live) finds six call sites: `evaluation/band.py:462,467`,
`evaluation/benchmark.py:612`, `evaluation/calibration.py:146`,
`evaluation/thresholds.py:331`, `inference/calibration.py:258`. Tracing which
of these feeds a value that reaches `CalibrationWitness.compute_witness`:

- `inference/calibration.py:258` is inside `threshold_at_fpr` — confirmed by
  `evaluation/thresholds.py:21`'s own docstring (*"`inference.calibration.
  threshold_at_fpr` takes the `1 - target_fpr` quantile"*) to be a **different,
  parallel, exploratory function**. `grep -rn "threshold_at_fpr" --include=
  "*.py" .` (live) shows it is called only from its own test
  (`tests/test_calibration.py`) — **never** from `fit_calibration` or
  `derive_operating_point`.
- The witness's actual threshold path (`fit_calibration`'s default
  `margin=DERIVE` → `evaluation/band.py:250-330` `derive_operating_point` →
  `evaluation/thresholds.py:158-186` `at_target_fpr`) uses **no `np.quantile`
  call at all** — `at_target_fpr` does an exact discrete scan over
  `candidate_thresholds(s)`, an observed-score enumeration, not an
  interpolated quantile.
- `evaluation/band.py:462,467`'s quantile use is inside a function whose own
  docstring (`:441-452`) states *"provided for the comparison in the design
  report and is not the shipped policy"* — confirmed not on the
  `fit_calibration` path.
- `evaluation/benchmark.py:612` and `evaluation/thresholds.py:331` are both
  `bootstrap_threshold_ci`-family confidence-interval diagnostics.
  `grep -n "bootstrap_threshold_ci" evaluation/calibration_protocol.py` (live)
  → no hits. Not called by `fit_calibration`.
- `evaluation/calibration.py:146` is ECE/reliability-diagram binning, an
  unrelated diagnostic, not called by `fit_calibration` either.

**N7, as stated, does not describe anything on the path
`CalibrationWitness.compute_witness` depends on.** The one function the
witness's `threshold`/`margin` actually come from (`at_target_fpr`) contains
no quantile call of any kind. Folding "the `np.quantile` method" into the
witness digest would add a digest key for a quantity that has **no causal
path** to any fitted number the witness records.

### Recommendation

**Neither N6 nor N7 folds into R31, for two different reasons:**

- **N6** is the same *shape* of gap as R31 (an unrecorded fitting-procedure
  detail that reaches the digest only through its fitted output), but has
  **zero live variability today** — folding it in now would add a field whose
  value is a compile-time constant everywhere in the current codebase,
  providing no real protection until some future batch actually parameterizes
  `fit_temperature`'s tolerance or range. That parameterization, if it ever
  happens, is the right time to design N6's binding — together with whatever
  new caller needs to vary it, so the field's semantics match a real,
  exercised knob rather than a speculative one.
- **N7** does not belong in `CalibrationWitness`'s scope **at all** — it names
  a function (`threshold_at_fpr`) that is not in the witness's call graph.
  Folding it in would be adding a digest key with no dependency to detect. If
  N7 is a real finding, it is a finding about `inference.calibration.
  threshold_at_fpr` as an unused-but-exported alternate API, not about the
  calibration witness.

Both `REPRODUCIBILITY_AUDIT.md` rows should be corrected to reflect this — see
the plan's documentation updates (§3.7).

### Affected files

None — this decision results in **no additional production code change**.
Documentation-only: `REPRODUCIBILITY_AUDIT.md`'s N6/N7 row gets a corrective
note (§3.7).

### Impact

| Axis | Impact |
|---|---|
| Migration | None — no digest change from this decision. |
| Version | `CALIBRATION_WITNESS_VERSION` moves for R31's own reason (§2) regardless of this decision; N6/N7 add no further keys to that move. |
| Cache | None. |
| Benchmark | None. |
| Deployment | None. |

### Does this change the approved R31 scope?

**No — it keeps R31's scope exactly as narrow as `R31_DESIGN.md` specified**
(one new field, one new frozenset). It closes the one item `R31_DESIGN.md`
§10.2 left open ("a scope decision for the reviewer, not assumed here") by
resolving it to "no" for both, on independently-derived, distinct grounds.

---

## Summary of the three resolutions

| # | Decision | Resolution |
|---|---|---|
| 1 | Field name | `ModelSpec.pre_calibration_identity` (new property, `inference/models.py`); witness field/parameter `pre_calibration_identity: str | None`; frozenset `SCHEMA_VERSIONS_WITH_PRE_CALIBRATION` |
| 2 | `benchmark.py` wiring | **D1 ships now** (wire the existing, already-present `decode_identity` cross-check into `validate()` — zero schema cost). **D2 deferred** (a new `pre_calibration_identity` cross-check needs its own `BenchmarkRun` field and its own `RUN_SCHEMA_VERSION` bump — out of scope for this batch). |
| 3 | N6/N7 | **Neither folds in.** N6: real gap, zero live variability today, defer until something parameterizes `fit_temperature`. N7: does not apply to `CalibrationWitness` at all — the witness's threshold path uses no `np.quantile` call; `REPRODUCIBILITY_AUDIT.md`'s row needs a corrective note, not a fix. |

None of the three resolutions reopens architecture, and none contradicts
`R31_DESIGN.md`'s core finding (§1–§4) or its rejected options (§5 B/C). Two
of the three (Decision 2, Decision 3) **narrow** what was left open toward
less scope, not more — the opposite of scope creep.

---

# Final Implementation Plan

## 1. Scope, restated precisely

**In scope:**
1. `CalibrationWitness`/`compute_witness`/`fit_calibration`/
   `verify_calibration_witness` gain `pre_calibration_identity`
   (`evaluation/calibration_protocol.py`).
2. `ModelSpec` gains a `pre_calibration_identity` property
   (`inference/models.py`).
3. `BenchmarkReport.validate()` passes the existing
   `self.primary.decode_identity` into its existing
   `verify_calibration_witness()` call (`evaluation/benchmark.py`, D1 only).
4. `CALIBRATION_WITNESS_VERSION` `1.2.0` → `1.3.0`.
5. Documentation and collateral test-literal updates.

**Explicitly out of scope (deferred, not touched):** D2 (new `BenchmarkRun`
field + `RUN_SCHEMA_VERSION` bump), N6, N7, R30-bundle, OR1-b, any decode/`.f.`/
weights identity work.

## 2. Production files changed

| File | Change |
|---|---|
| `evaluation/calibration_protocol.py` | `+pre_calibration_identity: str \| None = None` field on `CalibrationWitness`; `+SCHEMA_VERSIONS_WITH_PRE_CALIBRATION = frozenset({"1.3.0"})`; `compute_witness` gains the `pre_calibration_identity` parameter and a digest branch (`if schema_version in SCHEMA_VERSIONS_WITH_PRE_CALIBRATION: parts["pre_calibration_identity"] = pre_calibration_identity`, mirroring `decode_identity`'s exact shape at `:265-276`) plus the matching `ProtocolViolation` guard for a pre-1.3.0 schema carrying the field; `CALIBRATION_WITNESS_VERSION = "1.3.0"` with an updated docstring (a **1.2.0 → 1.3.0** entry appended, matching the existing 1.0.0→1.1.0 / 1.1.0→1.2.0 entries' format); `fit_calibration` gains a `pre_calibration_identity: str \| None = None` parameter, threaded to both `compute_witness` and the `CalibrationWitness(...)` constructor call; `verify_calibration_witness` gains a `pre_calibration_identity: str \| None = None` keyword and a cross-check block structurally identical to the existing `decode_identity` block (`:748-760`); `to_dict()` gains the key; `from_dict()` adds `"pre_calibration_identity"` to its `optional` set and gains the equivalent of the existing decode-identity-vs-schema guard (`:370-380`). |
| `inference/models.py` | `+ModelSpec.pre_calibration_identity` property (new, beside `effective_version`). No existing property changed. |
| `evaluation/benchmark.py` | One line: `BenchmarkReport.validate()`'s existing call to `verify_calibration_witness(self.primary.calibration, protocol=self.protocol)` (`:975-977`) becomes `verify_calibration_witness(self.primary.calibration, protocol=self.protocol, decode_identity=self.primary.decode_identity)`. |

**Not changed:** `contracts/version.py`, `contracts/enums.py`,
`inference/decode_env.py`, `inference/preprocess.py`, `backend/*`, `workers/*`,
`evaluation/thresholds.py`, `evaluation/band.py`, `inference/calibration.py`
(no code here — see the N6/N7 disposition above).

## 3. Test files changed

| File | Change |
|---|---|
| `tests/test_calibration_witness.py` | New tests per §5 below (fail-before/pass-after for the new field; the cross-check tests). Existing fixtures (`compliant_report()` etc.) **unchanged** — confirmed safe in Decision 2's re-derivation (none currently pass `decode_identity`, so D1's wiring is a no-op against them). |
| `tests/test_evaluation_benchmark.py` | One new test confirming D1's wiring (`validate()` now passes `decode_identity` through) does not regress any existing benchmark-report test; existing tests unchanged (none of them exercise a calibration witness — confirmed, §Decision-2 re-derivation, the only `calibration=` reference in this file is a bare `CalibrationSpec`, not a `CalibrationWitness`). |
| `tests/test_crop_identity.py:259` | `CALIBRATION_WITNESS_VERSION == "1.2.0"` → `"1.3.0"` |
| `tests/test_decode_identity_completeness.py:359` | same |
| `tests/test_dependencies.py:407` | same |
| `tests/test_evaluation_identity.py:475` | `a.schema_version == CALIBRATION_WITNESS_VERSION == "1.2.0"` → `"1.3.0"` |
| `tests/test_headline_and_face_semantics.py:462` | `CALIBRATION_WITNESS_VERSION == "1.2.0"` → `"1.3.0"` |
| `tests/test_video_selection_identity.py:325` | same |

**Not changed:** `RUN_SCHEMA_VERSION` pins (7 files, listed in Decision 2) —
unaffected because D2 is deferred and D1 touches no `BenchmarkRun` field.

## 4. Documents changed

| Doc | Change |
|---|---|
| `docs/R31_SPEC_RESOLUTION.md` | this document |
| `REPRODUCIBILITY_AUDIT.md` | R31 row → Closed in [batch], summarizing the fix and cross-referencing this document and `R31_DESIGN.md`; **N6/N7 row gets a corrective note** (Decision 3 — N7 does not apply to `CalibrationWitness`'s call graph; N6 is real but has zero live variability today, deferred pending a future caller that parameterizes `fit_temperature`) — **not** marked closed, just corrected; a new "Still open" note that D2 (benchmark-run-level cross-check) is a named, deliberately deferred follow-up, not a forgotten one. |
| `docs/R31_DESIGN.md` | **Not edited** — remains the historical record of the original finding and rejected options, per this session's established convention that design documents are not retroactively rewritten by their own resolution. |

## 5. Fixtures, snapshots, calibration artifacts — explicit verification

- **Stored `CalibrationWitness` instances:** `grep -rln "\"witness\":\|'witness':" fixtures_out/ schema_out/ 2>&1` and a repository-wide search for any committed `.json` calibration artifact → **none exist** (re-confirming `R31_DESIGN.md` §3.1/§7.1's finding; this repository has no deployment and no real model, so no witness has ever been persisted outside a test's in-memory fixture). **No migration is required for any stored witness because none are stored.**
- **`fixtures_out/*.json`:** `contracts/fixtures.py` synthetic bundles carry `model.version = "1.0.0-dev"` and no calibration witness payload — unaffected, confirmed by the same reasoning every prior `.d.`/`.f.` batch used (B22, B23, B24).
- **`schema_out/*.json` / `backend/openapi.snapshot.json`:** `CalibrationWitness` is not a wire type (re-confirmed, `R31_DESIGN.md` §6, corrected in this session to note the one prose-only hit in `contracts/version.py:146`). No regen, no diff.
- **Benchmark history (`BenchmarkRun` JSONL rows):** none exist in this repository (no evaluation run has been persisted to disk in-repo). D1's wiring does not touch `BenchmarkRun`'s shape at all, so this is moot regardless.

**Conclusion: zero stored artifacts of any kind require migration.** This
remains, as `R31_DESIGN.md` §7.1 stated, the cheapest version bump available
in this codebase's history — confirmed again after narrowing scope to D1-only.

## 6. Collateral required by `CALIBRATION_WITNESS_VERSION` 1.2.0 → 1.3.0

Exhaustive, re-confirmed by search (§3, above): **six test-literal pins**
(`test_crop_identity.py:259`, `test_decode_identity_completeness.py:359`,
`test_dependencies.py:407`, `test_evaluation_identity.py:475`,
`test_headline_and_face_semantics.py:462`, `test_video_selection_identity.py:325`).
No production file pins the literal outside `evaluation/calibration_protocol.py`
itself (the constant's definition). No `.d.`/`.f.`/`STUB_ENGINE_VERSION`
collateral — this axis is independent (§Decision-1 impact table, and
`calibration_protocol.py:74-81`'s own stated independence).

## 7. Fail-before / pass-after matrix

| # | Test | Pre-change | Post-change |
|---|---|---|---|
| FB1 | `test_a_model_config_difference_does_not_move_the_witness` (the `R31_DESIGN.md` §2 live study, promoted to a real test) | **passes** (bug reproduces): two fits with identical logits but different `pre_calibration_identity` produce the same witness | — |
| FB2 | `CalibrationWitness.compute_witness(..., pre_calibration_identity=...)` | **fails** — `TypeError: unexpected keyword argument` | — |
| FB3 | `test_pre_calibration_identity_is_recoverable_for_forensics` (`"pre_calibration_identity" in witness.to_dict()`) | **fails** — key absent | — |
| PA1 | `test_a_model_config_difference_now_moves_the_witness` | — | Re-run FB1 with `pre_calibration_identity` supplied and differing → witnesses differ |
| PA2 | `test_pre_calibration_identity_is_recoverable_for_forensics` | — | `"pre_calibration_identity" in witness.to_dict()` |
| PA3 | `test_a_1_2_0_payload_cannot_smuggle_a_pre_calibration_identity` | — | `compute_witness(schema_version="1.2.0", pre_calibration_identity="x", ...)` raises `ProtocolViolation`, mirroring the existing `decode_identity` guard |
| PA4 | `test_verify_calibration_witness_cross_checks_pre_calibration_identity` | — | Mismatch → problem reported; witness records `None` and a value is supplied → a distinct "cannot be confirmed" problem, mirroring the existing `decode_identity` cross-check tests |
| PA5 | `test_model_spec_exposes_a_pre_calibration_identity` | — | `ModelSpec(...).pre_calibration_identity` is a stable 8-hex fingerprint; two specs differing only in `gate_fingerprint` or `face_selection_fingerprint` produce different values; two differing only in `calibration` produce the **same** value (the no-circularity property, asserted positively) |
| PA6 | `test_the_report_now_cross_checks_decode_identity` (D1) | `BenchmarkReport.validate()` never passes `decode_identity` — a mismatched witness/run pair silently validates | A synthetic report whose primary run's `decode_identity` disagrees with its witness's now fails `validate()` |
| PA7 | `test_an_old_calibration_witness_still_verifies_after_the_change` | — | A `schema_version="1.2.0"` witness with no `pre_calibration_identity` round-trips through `to_dict()`/`from_dict()`; digest still matches (direct analogue of `test_an_old_calibration_witness_still_verifies_after_the_change` for the `1.1.0`→`1.2.0` transition) |

## 8. Non-movement matrix

| Quantity | Must remain | Why |
|---|---|---|
| `decode_identity()` | `a97ccb68` | This design touches no decode code. |
| `STUB_ENGINE_VERSION` | unchanged literal | `effective_version` never reads `CalibrationSpec.witness` or `ModelSpec.pre_calibration_identity` (confirmed, no new read added to `calibration_fingerprint`/`effective_version`). |
| `model_identity_of(STUB_ENGINE_VERSION)` | unchanged | Same reason. |
| `ModelSpec.calibration_fingerprint` source | byte-identical | Positive test (PA5's negative-control half) — the new property must not be read by it. |
| `CONTRACT_VERSION` | `1.0.0` | Unrelated artifact. |
| `RUN_SCHEMA_VERSION` | `1.2.0` | **D2 deferred** — no `BenchmarkRun` field added, so this axis does not move in this batch (this is the direct, load-bearing consequence of Decision 2). |
| Split-lock `LOCK_VERSION` | `1.1.0` | Unrelated artifact, untouched. |
| OpenAPI signature / `schema_out` / `fixtures_out` | unchanged | Witness is not a wire type (§5). |
| `.f.` / weights_identity / `.g.` | unchanged | None of these read the witness; the witness reads *them* (one-directional, by construction — `ModelSpec.pre_calibration_identity` reads `weights_identity`/`gate_fingerprint`/`face_selection_fingerprint`, never the reverse). |

## 9. Stored-artifact migration — explicit verification (restated for the record)

- Stored `CalibrationWitness` payloads: **none exist** → no migration.
- `BenchmarkRun` history rows: **none exist**, and unaffected regardless (D2 deferred) → no migration.
- `fixtures_out/*`, `schema_out/*`, `backend/openapi.snapshot.json`: not wire-adjacent to the witness → no migration, no regen diff expected.
- Any calibration artifact outside this repository: out of this document's knowledge — the design's guarantee (schema-version-as-parameter, §Decision-1/§2) is that any such artifact, if one exists, continues to verify unchanged under its recorded `1.2.0` schema, per §7's PA7.

---

## 10. Gate

## **APPROVE FOR IMPLEMENTATION**

All three specification decisions from `R31_DESIGN.md` §11 are resolved, each
re-derived from source rather than accepted from the prior document or the
audit trail:

1. **Field name** — `pre_calibration_identity`, computed as a new `ModelSpec`
   property; avoids both hazards found in re-derivation (the Pydantic
   `model_config` prefix collision, the pre-existing broader `model_identity`
   meaning).
2. **`benchmark.py` wiring** — split into D1 (ships now, zero cost, proven
   safe against every existing fixture) and D2 (deferred, would require its
   own `RUN_SCHEMA_VERSION` bump and is out of scope).
3. **N6/N7** — neither folds in; N6 deferred for lack of live variability, N7
   found not to apply to `CalibrationWitness` at all.

No new blocking issues were discovered. Scope is **narrower** than
`R31_DESIGN.md`'s original sketch (D2 excluded), not wider. Zero stored
artifacts require migration (§9). The version-bump surface is exactly one
axis (`CALIBRATION_WITNESS_VERSION`), with a six-file collateral list, and no
cache, deployment, or contract impact.

Implementation may proceed directly against §2–§8 of this plan.
