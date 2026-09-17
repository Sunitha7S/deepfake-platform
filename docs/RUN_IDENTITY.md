# Run identity: separating what the model is from where it ran

**Status:** S8 Batch 9. **R12 partially closed** (cache and deployment identity
closed; benchmark identity open). **R13 unchanged — open**, deferred to the
SCRFD migration.
**Contract impact:** none. `CONTRACT_VERSION` 1.0.0, `EvidenceBundle`,
`ThresholdInfo`, OpenAPI, serving routes and every model contract unchanged. One
configured literal moved, deliberately and loudly (§6).

---

## 1. Phase 1 — where identity is produced and consumed

Traced exhaustively. Every site that generates or reads an identity string:

### 1.1 Producers

| Site | Produces | Composed from |
|---|---|---|
| `ModelSpec.weights_identity` | weights identity | `version` + per-branch versions + branch weights |
| `ModelSpec.calibration_fingerprint` | calibration fingerprint | temperature, `fitted_on`, `n_samples`, threshold, band edges, `target_fpr` — all quantised |
| `MediaCaps.gate_fingerprint` | gate fingerprint | the six caps |
| `ModelSpec.effective_version` | **model identity** | `compose_model_version(weights, calibration, gate)` |
| `DecodeEnvironment.identity` | **decode identity** *(new)* | OpenCV version, video backend, libavcodec version, OS, machine |
| `DecodeEnvironment.digest` | forensic digest *(refined)* | decode identity + CPU baseline + CPU dispatch |
| `{Image,Video}Pipeline.model_version` | **runtime identity** *(new)* | `compose_runtime_version(model_identity, decode_identity)` |
| `evaluation.benchmark.dataset_fingerprint` | corpus identity | the evaluated items |
| `CalibrationWitness.witness` | calibration provenance | protocol, role, corpus, fitted values |
| `SplitLock.content_hash` | split identity | frozen assignments + identity digests |

### 1.2 Consumers

| Site | Reads | Was reading | Now reads |
|---|---|---|---|
| `_model_info` → `ModelInfo.version` | published version | model identity | **runtime identity** |
| `StorageResultSink.content_key_for` | cache key | `bundle.model.version` | unchanged code, **now a runtime identity** |
| `StorageResultSink.find_by_content` | cache read | `settings.model_version` | unchanged code, **now a runtime identity** |
| `AnalysisService.create_upload` | upload short-circuit | `settings.model_version` | as above |
| `AnalysisService` job spec | idempotency key | `sha256:model_version` | as above |
| `backend/routes/health.py` | deployment identity | `settings.model_version` | as above |
| `workers.engine_registry.engine_versions` | alignment check | `pipeline.model_version` | as above |
| `verify_version_alignment` | startup gate | configured vs engines | as above |
| `run_benchmark(effective_model_version=…)` | benchmark identity | caller-supplied | **unchanged — model identity** |
| `regression_gate` | comparability | dataset fingerprint; notes on equal model version | **unchanged** |
| `REQUIRED_REPORT_FIELDS` | report completeness | `effective_model_version` | **unchanged** |

The last three rows are the open half of R12 and are discussed in §7.

### 1.3 The chain, end to end

```
weights ─┐
calib ───┼─> effective_version ─┐                       (model identity)
gate ────┘                      ├─> runtime_version ──> ModelInfo.version
OpenCV/backend/ffmpeg/OS/arch ──┘   (decode identity)         │
                                                              ├─> content cache key
                                                              ├─> idempotency key
                                                              ├─> /health
                                                              └─> alignment check
```

---

## 2. Phase 2 — threat model

| Threat | Mechanism | Reaches the pixels? | Reaches the identity? |
|---|---|---|---|
| **Decoder backend change** | FFMPEG ↔ MSMF | **Yes — measured, zero identical frames** | **Yes, now** — `video_backend` |
| **OpenCV build difference** | different libs, kernels, defaults | Yes, structurally | **Yes, now** — `opencv_version` |
| **ffmpeg version difference** | libavcodec behaviour | Yes, structurally | **Yes, now** — `ffmpeg_version` |
| **Environment variable override** | `OPENCV_VIDEOIO_PRIORITY_*` | **Yes — measured** | **Neutralised** — the backend is named, not inherited (Batch 8); and if a fallback occurs, the resolved backend is what gets digested |
| **OS-specific backend selection** | MSMF on Windows, V4L/GStreamer on Linux | Yes, structurally | **Yes, now** — `platform_system` + the resolved backend |
| **Future hardware-specific decode paths** | NVDEC, VideoToolbox, QSV/INTEL_MFX | Yes, certainly | **Partly** — `platform_machine` and the backend name cover a *backend* switch; a same-backend hardware-accelerated path that OpenCV does not rename would **not** be distinguished. Open, §7 |
| CPU/SIMD dispatch | AVX2 vs SSE2 kernels | **No — measured identical** (Batch 8 E6) | Deliberately excluded from the cache key; recorded forensically |
| Python version | — | No | Recorded, excluded |

Two entries deserve emphasis because they invert the usual assumption:

**The environment-variable threat did not need a different build.** That is what
made R12 realistic rather than theoretical: one machine, one build, one binary,
and `OPENCV_VIDEOIO_PRIORITY_MSMF=9999` changes every decoded frame.

**CPU dispatch is excluded on the strength of a measurement, not a hunch.** It
is the only threat here that was tested and found *not* to change a pixel, and
excluding it is what keeps a mixed-hardware fleet on one cache namespace.

---

## 3. Phase 3 — which artifact needs which identity

Five distinct questions were being answered by one string. Separating them is
the whole design.

| Identity | Question | Composition | Consumers |
|---|---|---|---|
| **Model identity** | "Which model is this?" | weights + calibration + gate | benchmark runs, regression gates, reports, `ModelSpec.effective_version` |
| **Cache identity** | "Would this deployment produce the same number?" | model + decode | content cache key, idempotency key |
| **Deployment identity** | "What is this process running?" | model + decode | `/health`, alignment check, logs |
| **Benchmark identity** | "Is this comparison meaningful?" | model + decode + dataset + split lock + protocol | `regression_gate` — **decode half not implemented**, §7 |
| **Reproducibility identity** | "Can this result be reconstructed?" | everything above + forensic decode fields | the study, run records, audits |

The rule that decides membership, and it is not the same rule for each:

- A **cache** key must be conservative. A false miss costs compute; a false hit
  returns a wrong verdict. The two errors are not symmetric, so the tie breaks
  toward missing — which is why OS and machine are in the decode identity even
  though a cross-OS difference could not be measured from one machine.
- A **benchmark** identity must be *exactly* right in both directions. Too
  coarse and a decoder change reads as a model regression; too fine and every
  worker rebuild refuses a legitimate comparison.
- A **model** identity must exclude the environment entirely, or "did the model
  get worse" cannot be asked.

---

## 4. Phase 4 — proof that the cache could cross environments

Not argued — executed. `tests/test_run_identity.py`, three tests.

**`test_the_legacy_key_serves_one_decoders_result_to_another`** builds two real
bundles for the same bytes and the same model under the two decode identities
Batch 8 actually measured (`c20a064a` for FFMPEG, `a90156c7` for MSMF), stores
the first under `LegacyModelIdentityKey` — the pre-Batch-9 key function, kept
verbatim beside the new one — and looks up the second:

```
hit is ffmpeg          -> True
hit.p_fake != msmf.p_fake -> True
```

An MSMF deployment is served the answer FFMPEG produced.

**`test_the_real_sink_isolates_the_two_environments`** repeats it against
`InMemoryResultSink`, which derives its key from `bundle.model.version` — the
production key derivation, not a restatement:

```
find_by_content(sha, UNDER_FFMPEG)  -> hit    (same environment)
find_by_content(sha, UNDER_MSMF)    -> None   (was: the FFMPEG bundle)
find_by_content(sha, MODEL)         -> None   (bare model identity: separate namespace)
```

**Answer: yes, and it is now prevented.** With the caveat in §7 that prevention
is per-namespace isolation, not agreement.

---

## 5. Phase 5 — the scheme

### 5.1 Format

```
stub-0-w7f45fada+c.739c3fec.g.ae232cba.d.c20a064a
└──────────────┘  └───────┘  └───────┘  └───────┘
     weights      calibration   gate      decode
└──────────── model identity ──────────┘
└──────────────── runtime identity ───────────────┘
```

The decode segment is **last** so that the model identity is a *prefix* of the
runtime identity. `model_identity_of` is then one `partition`, and a runtime
version reads as a model version with something appended.

> **Later batches added a `.f.` segment between gate and decode** (B12, R17 — the
> face-selection policy). **Batch 23 (R29) made `.f.` a composite**:
> `selection_identity(face, video) = fingerprint({"face": …, "video": …})`,
> combining face selection with the video frame-selection and pooling policy,
> computed by both pipelines. It is still one opaque 8-hex fingerprint —
> `contracts/version.py`'s parser and the prefix property are unchanged. The
> decode identity is untouched. See `docs/VIDEO_SELECTION_IDENTITY.md`.

### 5.2 API, all in `contracts/version.py` (stdlib only)

```python
compose_runtime_version(model=..., decode=...) -> str
parse_runtime_version(version)   -> RuntimeVersionParts(model, decode)
model_identity_of(version)       -> str          # idempotent
decode_identity_of(version)      -> str | None
is_environment_pinned(version)   -> bool
```

`parse_model_version` strips the decode segment first, so every existing caller
keeps working on a runtime version without knowing the segment exists.

### 5.3 How each requirement is met

**Preserves existing contracts.** `ModelInfo.version` is `str` and stays `str`;
only its value gained a segment. No field added, removed or retyped anywhere.
The OpenAPI drift guard passes with no snapshot change, and
`test_the_evidence_bundle_schema_is_unchanged` asserts the field set directly.

**Avoids unnecessary cache fragmentation.** The decode identity is *strictly
coarser* than the forensic digest: CPU baseline and dispatch are excluded on the
strength of Batch 8's measurement that they do not change a pixel, as are
`backend_source` and the Python version. Measured fragmentation impact in §8.

**Allows forensic reconstruction.** Every field is recoverable —
`DecodeEnvironment.to_dict()` carries all nine plus both digests, and the
published bundle carries the identity segment, so a stored result names the
decode environment that produced it. `test_the_forensic_digest_is_strictly_finer
_than_the_cache_identity` asserts every field survives.

**Distinguishes model changes from environment changes.** A decoder change moves
the runtime version and leaves the model identity byte-identical; a model change
moves both. Both directions are asserted. An operator diffing two version strings
sees which segment moved without decoding anything.

### 5.4 What was deliberately not done

**The decode identity is not in `calibration_fingerprint`.** That fingerprint
identifies a *calibration*. Folding the decoder in would make "the temperature
changed" and "the worker was rebuilt" the same event — precisely the conflation
this batch exists to undo.

**`ModelSpec` did not gain a decode field.** Model identity must be computable
without an environment, or `effective_version` stops meaning "which model". The
pipeline composes the runtime version; the spec stays pure.

**No sentinel for "unpinned".** `compose_runtime_version(decode=None)` returns
the model identity unchanged. A placeholder like `.d.none` would make "makes no
claim" and "pinned to nothing" the same string, and the scheme rests on telling
them apart.

---

## 6. Migration analysis

### 6.1 What moves

| Artifact | Before | After |
|---|---|---|
| `STUB_ENGINE_VERSION` | `stub-0-w7f45fada+c.739c3fec.g.ae232cba` | `…ae232cba.d.c20a064a` |
| `ModelInfo.version` in new bundles | model identity | runtime identity |
| `pipeline.model_version` | model identity | runtime identity |
| `pipeline.model_identity` | — | new; the old value |
| `DecodeEnvironment.digest` | one digest | now the *forensic* one; `identity` is the cache one |

**This is an explicit version bump, not a silent one.** The literal in
`backend/config.py` carries a docstring stating that it is environment-specific,
that a worker on another OS or OpenCV build will report a different digest, and
that `verify_version_alignment` will fail startup on the mismatch — which is the
intended behaviour, not a bug to route around.

### 6.2 Cache migration

Existing cache entries are keyed on the bare model identity. After this change
lookups use the runtime identity, so **every pre-Batch-9 entry becomes
unreachable**. This is a one-time full cache miss, not corruption:

- No stale entry is served under a new name — the namespaces do not overlap, and
  `test_the_real_sink_isolates_the_two_environments` asserts the bare model
  identity misses a runtime-versioned entry.
- Old objects remain readable by job id; only the content pointer is orphaned.
- Cost is one re-analysis per distinct file. No data loss, no manual step.

The alternative — back-filling old entries with the current decode identity —
was rejected: it would assert that historical results were produced by *this*
environment, which is exactly the unfounded claim the batch removes.

### 6.3 Backward compatibility

`parse_runtime_version` is total on any non-empty string and returns
`decode=None` for a pre-Batch-9 version. Nothing needs migrating to be readable;
an old string simply makes no environment claim, and `is_environment_pinned`
reports that. `parse_model_version` accepts both forms.

### 6.4 Multi-environment deployments

A fleet with genuinely different decode environments now maintains one cache
namespace per environment. That is correct and it is the cost: hit rate falls in
proportion to the number of distinct environments. The fix is to make the fleet
homogeneous — pin one OpenCV build and one backend — not to widen the key.

---

## 7. Guarantees gained, and not gained

### Gained

1. **A cached result cannot cross decode environments.** Proved against the real
   sink with a named double for the old key (§4).
2. **A decoder change is visible in the deployment identity.** `/health` and the
   alignment check both carry it.
3. **A worker whose decode environment differs from the configured one fails at
   startup**, instead of silently sharing a cache namespace.
4. **Model and environment changes are distinguishable** from the version string
   alone, in both directions.
5. **A stored bundle names the decode environment that produced it** — forensic
   reconstruction from the artifact, with no schema change.
6. **The cache does not fragment on CPU dispatch**, on the strength of a
   measurement rather than a guess.

### Not gained

1. **Two decoders still disagree.** Nothing here makes them agree; it makes them
   distinguishable. Same shape of guarantee as the split lock's.
2. **Benchmark comparisons are still environment-blind.** `BenchmarkRun` has no
   decode field, so `regression_gate` cannot refuse a cross-decoder comparison
   the way it refuses a cross-dataset one. A decoder change can still be read as
   a model regression. Implementing this needs `RUN_SCHEMA_VERSION` 1.1.0 →
   1.2.0, which this brief forbids. Asserted by
   `test_benchmark_runs_still_carry_no_decode_identity` so it cannot be
   forgotten.
3. **Calibration artifacts are environment-blind.** `CalibrationWitness` records
   the corpus and the protocol but not the decoder, so a temperature fitted
   under one decoder can be applied under another with no complaint. Same schema
   constraint.
4. **A same-backend hardware decode path is not distinguished.** If OpenCV
   routes through NVDEC or VideoToolbox without changing the backend name, the
   decode identity does not move. Untestable here — no such path on this box.
5. **Cross-OS agreement is unmeasured.** `platform_system` is in the key on the
   conservative side, which may over-fragment. One machine cannot settle it.
6. **Nothing forces the forensic record to be read.** Standing complaint since
   R4.

---

## 8. Measured impact

| Measurement | Value |
|---|---|
| Version string length | 41 → 50 characters |
| Cache namespaces per model, homogeneous fleet | 1 → 1 (unchanged) |
| Cache namespaces per model, mixed decoders | 1 → one per environment (**intended**) |
| Cache namespaces on a mixed-CPU fleet | 1 → **1** (CPU excluded, measured safe) |
| Pre-Batch-9 cache entries invalidated | all — one-time, no data loss |
| Cross-environment false hits before | possible; proved (§4) |
| Cross-environment false hits after | 0 |
| `p_fake` divergence the key now separates | 0.0125 measured end to end (Batch 8 E11) |
| Decisions that divergence would flip | 1.31% (Batch 8 E10, surrogate scorer) |
| Contract changes | 0 |
| Schema changes | 0 |

---

## 9. R12 and R13 status

**R12 — the decoder is outside every identity string: PARTIALLY CLOSED.**

| Half | Status |
|---|---|
| Cache identity | **Closed.** Proved against the real sink with a named double. |
| Deployment identity | **Closed.** `/health`, alignment, logs. |
| Model identity | **Closed by design** — the decoder is deliberately excluded and now has its own segment. |
| Forensic identity | **Closed.** The bundle carries it. |
| **Benchmark identity** | **Open.** Needs a `RUN_SCHEMA_VERSION` bump this brief forbids. |
| **Calibration identity** | **Open.** Same constraint. |

Batch 8 recorded a structural reason the decoder could not reach the API's cache
key. **That reason was wrong**, and the disproof was already in the codebase:
`Settings.model_version` is not computed by the API either — it is a configured
literal held to the engines by `verify_version_alignment`. Batch 9 does what
Batch 8 called impossible, by the mechanism Batch 8 had itself described.

**R13 — face-crop instability amplifies decoder drift: OPEN, unchanged.**

Nothing in this batch touches it, deliberately. It is a property of the
placeholder Haar detector, which the frozen architecture replaces with SCRFD;
tuning a detector scheduled for deletion is wasted work. Run identity does not
mitigate it — a moved crop produces a different answer within *one* environment,
so no key can separate it. The remediation remains: re-measure against SCRFD on
real faces, and make crop stability under ±4 LSB an acceptance criterion for
that migration.

---

## 10. Where things live

| what | where |
|---|---|
| Composition, parsing, the two identities | `contracts/version.py` |
| Decode identity vs forensic digest | `inference/decode_env.py` |
| Runtime version composition | `{Image,Video}Pipeline.model_version` |
| Published version | `inference/postprocess._model_info` |
| The configured literal and its warning | `backend/config.py` |
| Proof, doubles, and the asserted limits | `tests/test_run_identity.py` |
| The measurements this rests on | `tests/decode_study.py`, `docs/DECODE_DETERMINISM.md` |
