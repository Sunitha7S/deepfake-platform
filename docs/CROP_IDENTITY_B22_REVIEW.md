# Batch 22 implementation-readiness review of R25 (adversarial)

> Pre-implementation review of `docs/CROP_IDENTITY.md` (Batch 21). No code
> written, no files patched. Every claim below re-derived from the current
> tree, not from Batch 21's conclusions. Verdict in §10.

**Tree state verified.** 48 suites in `tests/run_all.py:68` (lines 69–116).
`STUB_ENGINE_VERSION = stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387.d.a97ccb68`
(`backend/config.py:41`). `POLICY.identity == "19559387"`
(`tests/test_crop_geometry.py:638`, green).

---

## 1. Re-derivation of every proposed modification

### 1.1 `weights_identity` — verified

`inference/models.py:256–271`, current body:

```python
per_branch = {b.value: self.branch_versions.get(b, self.version) for b in BranchId}
if all(v == self.version for v in per_branch.values()):
    branch_part = ""
else:
    branch_part = f"-b{fingerprint(per_branch)}"
weights_part = fingerprint({b.value: w for b, w in self.branch_weights.items()})
return f"{self.version}{branch_part}-w{weights_part}"
```

Shipped output: `stub-0` + `""` + `-w7f45fada` = `stub-0-w7f45fada`. Confirmed by
`tests/test_face_selection_identity.py:462`, `tests/test_run_identity.py:274`.

- **One producer, zero sub-parsers.** `grep -rn '"-w"|"-b"|split("-|partition("-'`
  over `inference/ contracts/ workers/ evaluation/ backend/` returns only
  `models.py:256`. `parse_model_version` (`contracts/version.py:421`) does
  `weights, _, rest = version.partition(CALIBRATION_MARKER)` — the entire
  pre-`+c.` chunk is opaque. **Appending `-i<fp>` is structurally safe.**
- `compose_model_version` guards (`version.py:280–299`): `weights` must not
  contain `.d.`, `.f.`, `+c.`, `.g.`. `-iecf7148f` contains none. ✓
- **Falsification attempt — real-checkpoint names.** A trained spec's `version`
  may be `convnextv2t-sbi-1.0` (hyphens). `weights_identity` becomes
  `convnextv2t-sbi-1.0-w<fp>-i<fp>`. Still opaque to `parse_model_version`. No
  break.
- **Alternative not taken by B21, noted:** fold `input_size` into the existing
  `weights_part` dict — `fingerprint({**branch_weights, "input_size": …})` —
  which moves `7f45fada` to a new 8-hex value with **zero format change**. B21's
  `-i` suffix is more operator-legible (matches the `-b` precedent) but adds a
  format element. Either is defensible; the suffix is the slightly higher-risk
  choice and should be a conscious one.

### 1.2 `interpolation_rule` → `FaceSelectionPolicy` — verified, with a caveat

`inference/preprocess.py:229–238`:

```python
def resize(image, size):
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interp)
```

- `resize` is a **bare module function**. It does **not** receive the policy.
  Neither `prepare_image` (`preprocess.py:722`) nor `extract_frames`
  (`preprocess.py:946`) pass one.
- **Caveat B21 understates.** B21 §3 says "hold it to `resize`'s branch by a
  re-derivation test (the ORDERING_RULE pattern)". That means `interpolation_rule`
  would be a **label with no functional coupling** — `resize` never reads it. A
  deployment that edits `resize` to `INTER_CUBIC` without touching the policy
  string gets the **same `.f.` identity** — the exact R25 defect, one level
  down. This is the identical situation `crop_shape` and `ordering` are already
  in, so it is *consistent*, but B21 should state plainly that knob 2's closure
  is "a name pinned by a re-derivation test", strictly weaker than knob 1
  (`input_size` is functionally read by `prepare_image`/`extract_frames`).
- **The name must encode the actual `or` behaviour**, not the docstring's
  "shrinking" (see R28, §9).

### 1.3 `crop_clip_rule` → `FaceSelectionPolicy` — verified, same caveat

`inference/preprocess.py:708–719`. `crop_face` **does** take `policy`, so
`crop_clip_rule` *could* be functionally read. With only one value
(`"intersect-image"`) it will not branch on anything until R23, so it too is a
label-pinned-by-a-test today. Acceptable; state it.

### 1.4 Identity-composition paths — verified end to end

```
_crop_and_score            image_pipeline.py:197-198   crop_face(policy) -> prepare_image(size=self._spec.input_size)
model_version              image_pipeline.py:86        compose_runtime_version(effective_version, decode_identity())
                           video_pipeline.py:135        (identical)
effective_version          models.py:274-287           compose_model_version(weights_identity, calibration_fingerprint, gate_fingerprint, face_selection_fingerprint)
content cache key          workers/io.py:85,93,164      "results/by-content/{sha256}/{bundle.model.version}.json"
API read key               backend/service.py:98        find_by_content(sha, settings.model_version)   [STUB_ENGINE_VERSION]
idempotency key            backend/service.py:227       f"{sha}:{model_version}"
alignment                  workers/engine_registry.py:86-115  engine_versions() -> pipeline.model_version, compared to configured literal
```

All confirmed. `ModelInfo.version = runtime_version or spec.effective_version`
(`postprocess.py:372,391`). `BranchResult.version = spec.version` (bare, line
307/318/342) — **unaffected** (the fix changes `weights_identity`, not
`version`).

### 1.5 Projected identity (recomputed this review, matches B21 §4)

```
weights_identity  stub-0-w7f45fada          -> stub-0-w7f45fada-iecf7148f
face-selection .f. 19559387                  -> 5589e93e   (with interp+clip keys)
effective_version stub-0-w7f45fada+c.739c3fec.g.ae232cba.f.19559387
              ->  stub-0-w7f45fada-iecf7148f+c.739c3fec.g.ae232cba.f.5589e93e
parse round-trips: True   model_identity_of(runtime)==new_ev: True
calibration/gate fingerprints: unchanged
```

`fingerprint({"input_size": 384})` = `ecf7148f` on this build (deterministic,
stdlib sha256). The exact `.f.` value depends on the two new key names/values the
implementer picks — `5589e93e` is illustrative.

---

## 2. Per-change impact tables

### input_size -> weights identity

| | |
|---|---|
| Files touched | `inference/models.py` (`weights_identity`), `backend/config.py:41` (regen literal) |
| Symbols touched | `ModelSpec.weights_identity` |
| **Fail-before (genuinely fail after the fix)** | `test_crop_geometry.py::test_input_size_is_in_no_identity_string` (asserts `_spec(384).effective_version == _spec(224).effective_version`), `::test_two_deployments_can_agree_on_every_identity_and_differ` (line 522, same equality) |
| **Collateral breaks (must update)** | `test_crop_geometry.py::test_batch_15_changed_no_geometry` (:638,:648); `test_model_version.py::test_configured_model_version_matches_the_engines` (self-heals on literal regen); `test_branch_b_semantics.py::_EXPECTED_EFFECTIVE`; `test_headline_and_face_semantics.py::_EXPECTED_EFFECTIVE`; `test_decode_identity_completeness.py::MODEL_ID` (:90,:342) |
| Must pass after | new `test_input_size_is_in_the_weights_identity` (two specs differ → identities differ); `parse_model_version(new_ev)` round-trips; `verify_version_alignment` fails on an `input_size` mismatch; every pre-existing spec built with `input_size=384` keeps its identity |
| Not touched | `contracts/*` (no schema surface — `grep input_size contracts/` = 0), OpenAPI, `schema_out/` |

### interpolation_rule -> FaceSelectionPolicy.identity

| | |
|---|---|
| Files touched | `inference/preprocess.py` (`FaceSelectionPolicy` dataclass, `.identity`, `.to_dict`, `.differences`, `default_face_selection`) |
| Symbols touched | `FaceSelectionPolicy` (new field), `default_face_selection()` (only constructor — `preprocess.py:569`) |
| **Fail-before** | `test_face_selection_identity.py::test_every_policy_field_moves_the_identity` (**hard** `assert names == set(alternatives)` at :301 — fails the instant a field is added); `test_crop_geometry.py::test_the_interpolation_policy_is_in_no_identity_string` **only if** the field name contains the substring `interpolation`/`interp`/`resize` (see §9 — this tripwire is fragile) |
| **Does NOT fail after the fix (B21 gap)** | if the field is named e.g. `interp_rule` the substring check at `test_crop_geometry.py:485-489` stays green and keeps asserting "knob 2 is open" |
| Collateral breaks | `.f.`-value pins in §1.4's list; `test_batch_15_changed_no_geometry` |
| Must pass after | `test_the_interpolation_rule_is_in_the_face_selection_identity` (rewritten — two policies differing only in the rule → different `.identity`); `test_every_policy_field_moves_the_identity` (extended with the new field+alternative); `test_the_policy_has_no_defaults` (the new field must have none); a re-derivation test binding the name to `resize`'s branch |

### crop_clip_rule -> FaceSelectionPolicy.identity

| | |
|---|---|
| Files touched | `inference/preprocess.py` (same set) |
| Symbols touched | `FaceSelectionPolicy` (new field), `default_face_selection()` |
| **Fail-before** | `test_face_selection_identity.py::test_every_policy_field_moves_the_identity` (:301 hard set-equality) |
| **Does NOT fail after the fix (B21 gap)** | `test_crop_geometry.py::test_the_clipping_rule_is_in_no_identity_string` has **no identity assertion at all** — it asserts `clipped.shape != padded.shape` and `prepare_image(clipped) != prepare_image(padded)`, both unchanged by the fix. It stays green forever. B21 §9.1's "flip the assertion" is impossible here — it must be **replaced**. |
| Collateral breaks | `.f.`-value pins; `test_batch_15_changed_no_geometry` |
| Must pass after | `test_the_clip_rule_is_in_the_face_selection_identity` (**new** — two policies differing only in `crop_clip_rule` → different `.identity`); the extended field-set test; no-defaults test; a re-derivation test binding `"intersect-image"` to `crop_face`'s slice |

---

## 3. Hidden consumers (searched, this review)

| Symbol | Consumers found | Missed by B21? |
|---|---|---|
| `weights_identity` | producer `models.py:271`; reader `models.py:283` (`effective_version`). **No sub-parser anywhere.** | no |
| `FaceSelectionPolicy.identity` | `image_pipeline.py:58`, `video_pipeline.py:107` (`.with_face_selection`); `face_selection_identity()` `preprocess.py:583`; `.to_dict` `preprocess.py:534`; tests | no |
| `FaceSelectionPolicy` **constructor** | **exactly one:** `preprocess.py:569` (`default_face_selection`). All other sites use `default_face_selection()` or `dataclasses.replace(SHIPPED, …)`. | B21 did not state the count; it is favourable — one edit |
| `compose_model_version` | `models.py:282`; `contracts/version.py:253` (`ModelVersionParts.compose`); tests (`test_run_identity.py:143`, `test_face_selection_identity.py:471`, `test_evaluation_identity.py`, `test_decode_identity_completeness.py:244`) — all hand-built fixtures, self-consistent | no |
| `parse_model_version` | `version.py:446,455` (`face_selection_identity_of`, `is_face_selection_pinned`); `test_*` | no |
| `model_identity_of` | `benchmark.py` (via `_bench`), `engine_registry`? (no), `image_pipeline.model_identity`/`video_pipeline.model_identity` return `effective_version` directly; `test_run_identity.py:97,255,281,351`, `test_face_selection_identity.py:503,557,672`, `test_evaluation_identity.py:354`, `test_decode_identity_completeness.py:342`, `test_crop_geometry.py:648` | no |
| `STUB_ENGINE_VERSION` | `backend/config.py:41,183` (default of `Settings.model_version`); imported by `test_model_version.py`, `test_run_identity.py`, `test_face_selection_identity.py`, `test_decode_identity_completeness.py`, `test_crop_geometry.py`, `test_headline_and_face_semantics.py` (via `_EXPECTED_EFFECTIVE` literal, not import) | **B21 said "5 files"; actual is 6 test files + `config.py` + the `_EXPECTED_EFFECTIVE` string literals that are not `grep 19559387` hits in one case.** See §7. |

**No hidden production consumer missed.** The gap is in the **test-pin
inventory** (§7) and the **fragility of two fail-before tests** (§9).

---

## 4. Migration claims — re-evaluated, attempts to disprove

| Claim | Verdict | Evidence / disproof attempt |
|---|---|---|
| `CONTRACT_VERSION` unchanged | **holds** | `grep -rn input_size contracts/` = 0; `ModelInfo.version` is `str`; no `EvidenceBundle` field added; `contracts/examples.py`/`fixtures.py` carry no computed version. Disproof attempt: none found. |
| `RUN_SCHEMA_VERSION` unchanged | **holds** | `BenchmarkRun.effective_model_version` already exists (`benchmark.py:157`) and already carries `.f.`. No new `BenchmarkRun` field. Disproof attempt: searched for a pinned `BenchmarkRun` JSON snapshot with a fixed `effective_model_version` — `test_evaluation_identity.py`/`test_decode_identity_completeness.py` use **hand-set** `MODEL_ID` fixtures, not the live value; they self-heal by updating the constant, no schema change. |
| `CALIBRATION_WITNESS_VERSION` unchanged | **holds** | `compute_witness` parts (`calibration_protocol.py:250–266`): protocol, role, `corpus_fingerprint`, n, temperature, nll×2, threshold, margin, target_fpr, selection_witness, split_lock_hash, schema, [decode]. `corpus_fingerprint = dataset_fingerprint(selection.as_list())` (`:593`) — **media-id list, not logit values**. No geometry, no `effective_model_version`. No re-digest. |
| OpenAPI unchanged | **holds** | `grep w7f45fada backend/openapi.snapshot.json` = 0; structural signature ignores prose; no field/shape change. |
| `schema_out` unchanged | **holds** | `grep "stub-0" schema_out/*` = 0. |
| stored artifacts readable | **holds** | strings stored verbatim; `parse_model_version` gains one era, stays total (verify with a pre-B9 / B9–13 / B14–20 / B22 round-trip test). `PRE_BATCH_12` literal (`test_face_selection_identity.py:87`) still parses (no `-i`, no `.f.`) — that is correct, it is a historical string. |
| benchmark compatibility unchanged | **holds, with nuance B21 already stated** | `regression_gate` (`benchmark.py:783`) short-circuits only on `dataset_fingerprint` (:803) and `decode_identity` (:811). `effective_model_version` inequality is a **note only** (:816), not a refusal. So a pre-B22 vs post-B22 pair still compares numerically and correctly (numbers identical under the shipped detector). `BenchmarkReport.validate()` (`:951`) **does** refuse a mismatched pair — which is the intended tightening. No `max_*` tolerance change. |

**No migration claim falsified.**

---

## 5. Cache-generation analysis — re-evaluated

**One generation, two namespaces (image + video), confirmed.**

- Primary store: `results/by-content/{sha256}/{model_version}.json` pointer
  (`workers/io.py:93`, written `:164`). Key moves → one-time full miss per file.
- **Secondary derived keys searched:**
  - idempotency: `jobqueue/memory.py:52` `self._by_idempotency: dict[(queue, key)]`
    — **in-memory only**, ephemeral job dedup, no persistence. Redis path
    (`jobqueue/redis.py`) is unimplemented (`_TODO`). Benign: in-flight keys stop
    matching, new submissions get a new job. No stale hit.
  - `/health`: reports the version, is not a store.
  - benchmark history JSONL (`append_history` `benchmark.py:843`): rows carry
    `effective_model_version` as **data**, not a filename key; `read_runs`
    (`:860`) is version-tolerant. No regeneration.
  - calibration witnesses / split locks: strings stored verbatim, verified under
    their own rule. No regeneration.
- **No hidden third store.** `grep -rn "by-content|content_key|_by_"` over
  `storage/ workers/ jobqueue/` surfaces only the two above.

**Conclusion: B21's "one cache generation" is accurate.** The idempotency dict is
a second derived key but not a persistent namespace.

---

## 6. Witness behaviour — re-evaluated

- **Does any witness/calibration component already bind the geometry?**
  Only **incidentally**: `input_size` → different pixels → different logits →
  different fitted `temperature` → different `quantize(temperature)` → different
  digest. Identical mechanism to the decoder half of R12 pre-B11
  (`REPRODUCIBILITY_AUDIT.md`, B11 "weaker than it looks").
- `corpus_fingerprint` binds the **media-id list only** (`:593`), not the
  arrays.
- Two fits that `quantize`-converge to the same temperature under two
  `input_size`s produce **byte-identical witnesses** — the collision B21's
  residual names.
- **B21's residual is accurate and correctly scoped.** First-class binding needs
  `effective_model_version` in the digest → a `CALIBRATION_WITNESS_VERSION` bump
  → its own batch. Do not attempt in B22.

---

## 7. Complete implementation checklist

### Production code

| File | Why | Edit (described, not written) | Test impact |
|---|---|---|---|
| `inference/models.py` | knob 1 | `weights_identity`: fold `input_size` into the hash (either `-i<fp>` suffix or add `"input_size"` to `weights_part`'s dict). Pick one deliberately (§1.1). | `test_model_version.py` relative-comparison tests stay green; `test_crop_geometry` knob-1 tripwires flip |
| `inference/preprocess.py` | knobs 2, 3 | Add `interpolation_rule: str` and `crop_clip_rule: str` to `FaceSelectionPolicy` (no defaults). Add both to `.identity`'s dict, `.to_dict`, `.differences`'s tuple. Set both in `default_face_selection()` (`:569`) — the sole constructor. Name `interpolation_rule` to describe the **actual `or` behaviour** (e.g. `"area-if-either-axis-exceeds-target-else-linear"`). | `test_face_selection_identity.py::test_every_policy_field_moves_the_identity` **fails at :301** until its `alternatives` dict gains the two keys; `test_the_policy_has_no_defaults` enforces no-default |
| `backend/config.py` | literal regen | Replace `STUB_ENGINE_VERSION` (`:41`) with the value the engines now report; update its docstring with a B22 paragraph (the file's own instruction, `:107-109`). | `test_model_version.py::test_configured_model_version_matches_the_engines` and every `verify_version_alignment(STUB_ENGINE_VERSION)` call self-heal |

### Test code — pins to update (verified this review)

| File:line | Current | Action |
|---|---|---|
| `tests/test_crop_geometry.py:457` | `test_input_size_is_in_no_identity_string` (asserts `==`) | rewrite → assert `!=`, rename |
| `tests/test_crop_geometry.py:467` | `test_the_interpolation_policy_is_in_no_identity_string` | **rewrite** — its `{"interpolation","interp","resize"}` substring check is fragile (§9); replace with a two-policy `.identity` inequality |
| `tests/test_crop_geometry.py:492` | `test_the_clipping_rule_is_in_no_identity_string` | **replace** — no identity assertion exists to flip; write a two-policy `.identity` inequality |
| `tests/test_crop_geometry.py:514` | `test_two_deployments_can_agree_on_every_identity_and_differ` | rewrite → they now get different identities |
| `tests/test_crop_geometry.py:630` | `test_batch_15_changed_no_geometry` (`POLICY.identity=="19559387"`, `model_identity_of(STUB_ENGINE_VERSION)=="…f.19559387"`) | rewrite → `test_batch_22_moved_the_identity_not_the_pixels`: assert new `.f.`, new `STUB_ENGINE_VERSION`, **and byte-identical crop/resize output** vs `LegacyClipToBounds` / current `resize` |
| `tests/test_face_selection_identity.py:290-311` | `test_every_policy_field_moves_the_identity` — `alternatives` dict + `names == set(alternatives)` | add `interpolation_rule` and `crop_clip_rule` with alternative values |
| `tests/test_branch_b_semantics.py:456` | `_EXPECTED_EFFECTIVE = "…f.19559387"` | update to new model identity |
| `tests/test_headline_and_face_semantics.py:92` (+ :445,:447,:549) | `_EXPECTED_EFFECTIVE` | update |
| `tests/test_decode_identity_completeness.py:90` (+ :342) | `MODEL_ID = "…f.19559387"` | update (lines 154/244/245 are hand-built fixtures — update for consistency, not correctness) |
| `tests/run_all.py:116` | 48 suites | register `("S8-B22 crop identity", test_crop_identity)` → 49 |

### New files

| File | Content |
|---|---|
| `tests/test_crop_identity.py` | the pass-after suite (§9 of `CROP_IDENTITY.md`, corrected per §9 below) |
| `tests/crop_identity_study.py` | **already exists** (B21). Update its §4 projection to the real post-fix values; keep §1–3, §5–6. |

### Docs — superseding notes (B20 precedent)

`docs/CROP_GEOMETRY.md`, `docs/FACE_SELECTION_IDENTITY.md`,
`docs/FACE_SIZE_POLICY.md`, `docs/HEADLINE_AND_FACE_SEMANTICS.md`,
`docs/BRANCH_B_SEMANTICS.md`, `docs/DECODE_IDENTITY_COMPLETENESS.md`,
`docs/FRAME_COUNT_REPRODUCIBILITY.md`, `docs/MULTI_FACE_SEMANTICS.md` — any that
cite `19559387` or the full model identity **as current** get a one-line
"superseded by B22" annotation. `docs/CROP_IDENTITY.md` gets the outcome.
`REPRODUCIBILITY_AUDIT.md` gets the Batch 22 section.

### Studies with stale narrative (not asserted, but must not lie)

`tests/crop_geometry_study.py:614-622` (stage that prints "input_size 384 vs 224
→ identical identity") — **its conclusion inverts**. Update or annotate.
`tests/face_selection_study.py`, `multi_face_study.py`, `face_size_policy_study.py`
print `POLICY.identity` / `effective_version` — regenerate any doc numbers they
feed.

---

## 8. Risk register

### Implementation risks

| Risk | Severity | Mitigation |
|---|---|---|
| `interpolation_rule` / `crop_clip_rule` are **labels `resize`/`crop_face` never read** — a code edit that bypasses the label re-creates R25 one level down | **medium** | mandatory re-derivation tests (ORDERING_RULE pattern) binding each name to the branch it describes; state the limitation in the field docstrings |
| the interpolation-rule name encodes the **buggy `or`** (R28), baking a known defect into the identity's named contract | medium | either fix R28's predicate in B22 (moves pixels — §9) or name it honestly (`"…either-axis…"`) and accept a second `.f.` move when R28 lands |
| implementer picks `-i` suffix vs `weights_part` fold inconsistently with future `gate`/`face` work | low | one-line ADR in the models.py docstring |
| new `FaceSelectionPolicy` fields given defaults "for ergonomics" | low | `test_the_policy_has_no_defaults` already fails on this |

### Migration risks

| Risk | Severity | Mitigation |
|---|---|---|
| `STUB_ENGINE_VERSION` literal not regenerated → silent cache split (read key ≠ write key) | **high if missed** | `test_configured_model_version_matches_the_engines` fails the build loudly; the literal's own docstring says "paste in the version it reports" |
| a benchmark history file spanning the boundary: pre-B22 rows vs post-B22 rows | low | `regression_gate` treats them as different models — a note, not a refusal; numbers identical under the shipped detector; `read_runs` is version-tolerant |
| back-fill temptation (write the new identity onto old artifacts) | medium | explicitly refused, B9/B11 precedent — restate in the batch |

### Backward-compatibility risks

| Risk | Severity | Mitigation |
|---|---|---|
| `PRE_BATCH_12` / hand-built version fixtures stop parsing | **none** — verified they carry no `-i` and are historical constants | `parse_model_version` unchanged; add a 4-era totality test |
| stored `EvidenceBundle` / `CalibrationWitness` / split-lock JSON | none — opaque strings, verified under own-era rules | round-trip test |
| a deployment pinning `DFP_MODEL_VERSION` to the old string | expected — it fails alignment at startup, which is the design (same as B9/B20) | release note |

### Testing risks

| Risk | Severity | Mitigation |
|---|---|---|
| **B21's "flip the 4 tripwires" is wrong for 2 of them** — `test_the_clipping_rule_is_in_no_identity_string` has no identity assertion and stays green; `test_the_interpolation_policy_is_in_no_identity_string` only breaks via a fragile substring match or via the unrelated R28 fix | **high** — this is the main falsification | treat both as **replacements**, not inversions; the pass-after suite must contain a positive two-policy `.identity`-inequality test for each knob |
| `test_every_policy_field_moves_the_identity:301` hard-asserts the field set — a genuine fail-before for knobs 2&3, but B21 never listed it | medium | add to checklist (done, §7) |
| full suite runtime — `test_evaluation_operating` takes 5–15 min (memory) | low | known; not R25-related; budget the run |
| a study's printed narrative silently goes stale and a later batch quotes it | medium | §7 lists every affected study |

### Operational risks

| Risk | Severity | Mitigation |
|---|---|---|
| one-time full cache miss on deploy — every previously-analysed file re-runs | expected, bounded | precedented B9/B14/B20; communicate the compute spike |
| mixed-fleet during rollout: old workers (`…f.19559387`) and new (`…f.5589e93e`) write to different namespaces | low — safe direction (false miss, not false hit); alignment check stops a mismatched API/worker pair | standard rolling-deploy note |

---

## 9. R25 alone, R25+R28, or split R28?

**Facts about R28 (`preprocess.resize` predicate `h > size or w > size`):**

- Real: study §3 confirms a 500×300→384 resize picks `INTER_AREA` and applies it
  to the 300→384 axis it is *enlarging*; mean |AREA−LINEAR| on that axis =
  0.040/255. Low impact, content-dependent, **rare under Haar** (square-ish
  crops rarely straddle `input_size`), **near-zero on the video path** (whole
  frames are almost always both-axes-larger than 384).
- **Fixing R28's predicate changes shipped pixels** for the straddling class →
  a (small, hard-to-bound) benchmark boundary. That directly contradicts B21's
  headline selling point for R25: *output-identical, no recalibration*.
- **Not fixing R28 but naming the interpolation rule** means the `.f.` identity's
  named contract describes the `or` behaviour; R28's later predicate fix moves
  `.f.` a second time (a knob-2 double-spend, symmetric with knob 3's).

**Recommendation: C — split R28's *pixel fix* into its own batch; B22 names the
interpolation rule with its current behaviour.**

Reasoning:
1. B22's value proposition is a clean, output-identical, one-cache-generation
   identity completion. Bundling a pixel change forfeits that and makes the
   migration analysis (§4–5) partly wrong.
2. R28's impact is small and there is **no real model or benchmark today** — so
   there is no urgency; it can ride the next `.f.`-moving batch (R23/SCRFD, R16)
   at zero marginal cache cost.
3. The constraint "R28 must not be folded into R25" is honoured either way, but C
   keeps them cleanly separate: B22 closes R25 (identity), a later batch closes
   R28 (behaviour).
4. B22 **must still** name `interpolation_rule` accurately (encode the `or`) and
   pin it with a re-derivation test — otherwise knob 2 is not actually closed.

If the team prefers B (fix R28 in B22): acceptable **only** if the batch
explicitly re-labels itself "not output-identical — a bounded class of
straddling-input_size crops changes", measures the decision impact on the
surrogate, and drops the "no recalibration" claim to "no recalibration required
(no fitted threshold exists yet)".

---

## 10. Verdict

### APPROVE WITH CHANGES

Batch 21's architecture is **sound and implementation-ready** on every load-bearing
claim: the three knobs are outside every identity (re-verified from source), the
homes are correct, `weights_identity` has no sub-parser, the migration touches no
contract/schema/witness version, the cache generation is genuinely one, and the
witness residual is accurately scoped. No migration claim could be falsified.

**Required changes before implementation:**

1. **Fix the fail-before strategy.** `test_the_clipping_rule_is_in_no_identity_string`
   and `test_the_interpolation_policy_is_in_no_identity_string` do **not** fail
   after the R25 fix (one has no identity assertion; the other uses a fragile
   substring match). Treat both as **replacements**. The pass-after suite must
   contain, per knob, a positive test: two `FaceSelectionPolicy` / `ModelSpec`
   instances differing only in that knob → different `.identity` /
   `.effective_version`. Do not rely on B21 §9.1's "invert the assertion".

2. **Add the missed pins to the checklist** (done in §7): `_EXPECTED_EFFECTIVE`
   in `test_branch_b_semantics.py` and `test_headline_and_face_semantics.py`,
   `MODEL_ID` in `test_decode_identity_completeness.py`,
   `test_every_policy_field_moves_the_identity`'s hard field-set assertion. B21's
   "5 files" undercounts.

3. **State knob 2/3's closure strength honestly.** They are names pinned by
   re-derivation tests, not functionally-threaded parameters — weaker than knob
   1. Put this in the field docstrings and the batch report.

4. **Split R28** (option C). B22 names the interpolation rule with its actual
   `or` behaviour and pins it; R28's predicate fix (which moves pixels) is a
   separate later batch. This preserves B21's output-identical / no-recalibration
   guarantee.

5. **Regenerate `STUB_ENGINE_VERSION`** and update its docstring; do not skip the
   study-narrative and doc superseding notes in §7.

With these five changes the batch is a closure-quality, output-identical,
one-cache-generation completion of R25 with no contract, schema, or witness
version movement.
