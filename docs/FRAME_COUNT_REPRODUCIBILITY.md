# frame_count reproducibility — architecture review of the R6 residual (Batch 19)

> **Batch 20 closed R27.** The recommendation in §8/§10 below — "fold libavformat
> (and possibly libswscale) into `decode_identity`, deferred to a
> cache-generation batch" — was carried out in Batch 20, which re-derived the
> component list and found **libavformat, libswscale and libavutil** all omitted.
> `STUB_ENGINE_VERSION`'s `.d.` segment is now `a97ccb68`, not `c20a064a`; the
> `c20a064a` values in §5 are the pre-Batch-20 measurement, kept as written. R6-a
> and R6-b (below) are unaffected and still open.
> See [`docs/DECODE_IDENTITY_COMPLETENESS.md`](DECODE_IDENTITY_COMPLETENESS.md).

> **Outcome.**
> **R6 (the `frame_count` residual) → PARTIALLY CLOSED.** The part that is a
> *design property* — a decoder-reported integer, trusted without validation,
> feeding uniform frame sampling so that a one-frame error moves most of the
> scored clip — is now measured, bounded and pinned. The part that is a
> *reproducibility gap* is split out as its own finding:
> **R27 → OPEN — the decode identity (`decode_identity`, the Batch 9 cache-key
> segment and the Batch 11 `BenchmarkRun` field) covers libavcodec but not
> libavformat, and `CAP_PROP_FRAME_COUNT` is a libavformat quantity.** Two
> deployments that share a `decode_identity` can disagree on `frame_count`, score
> different frames, publish different `p_fake`, and collide on one content-cache
> key — **executed** in §5.
>
> **Nothing shipped as a behaviour change.** `probe_video` still trusts the
> decoder; `decode_identity` still omits the demuxer; no fingerprint, schema
> version, contract version or cache generation moved. The closure is the
> characterisation, the study, the pins, and a precise migration recommendation
> for R27 (§8) — the same posture Batch 15 took on crop geometry and Batch 8 took
> when it deferred R12's remediation 2 to "a batch whose brief permits it".

Re-runnable: `python tests/frame_count_reproducibility_study.py` (6 experiments).
Proof and named doubles: `tests/test_frame_count_reproducibility.py` (16 tests).
This review was re-derived from code and execution; it was not assumed.

**State reviewed.** 46 suites, `CONTRACT_VERSION` 1.0.0, `RUN_SCHEMA_VERSION`
1.2.0, `CALIBRATION_WITNESS_VERSION` 1.2.0. R6 was last touched in Batch 8
(`docs/DECODE_DETERMINISM.md`), which measured the amplifier (E2) but not the
trigger and left the residual "Unresolved". Batch 9 (`RUN_IDENTITY.md`) and Batch
11 (`EVALUATION_IDENTITY.md`) closed the decoder-in-identity finding R12; this
review asks whether they reach `frame_count` and finds they do not, entirely.

---

## The ten questions, answered

### 1. What exactly is `frame_count` today?

`MediaInfo.frame_count: int | None = Field(default=None, ge=1)` — a published wire
field, required for `kind=video`, null for images. Its value is set once, in
`inference/preprocess.probe_video`:

```python
frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
...
return VideoMeta(effective_fps, frames, duration, width, height)
```

`CAP_PROP_FRAME_COUNT` is the container's frame count **as reported by the video
backend's demuxer** — for MP4/MOV that is FFmpeg's `mov` demuxer in libavformat,
which returns the stream's `nb_frames` when the container carries one and an
estimate (`duration × avg_frame_rate`) when it does not. It is taken **verbatim,
with no sanity check** (`test_frame_count_is_the_raw_cap_prop_value`).

`probe_video` then computes `duration = frames / effective_fps`, so **`frame_count`
also determines `MediaInfo.duration_s`** — one wrong count reaches two wire fields
(`test_duration_s_is_derived_from_frame_count`).

Not to be confused with `SuspectSegment.frame_count` (the length of a contiguous
above-threshold run) or `VideoAnalysis.frames_scored` / `frames_sampled` (counts
of frames that actually decoded). Those are computed from the decoded array, not
from the metadata (`test_suspect_segment_frame_count_is_a_different_field`).

### 2. Which implementation path computes it?

```
VideoPipeline.analyze_video
  └─ extract_frames(data, n_frames, caps, target_size)          preprocess.py:874
       ├─ probe_video(path, caps)                                preprocess.py:788
       │    └─ open_video(path)  ── decode_env.resolve_video_backend() → FFMPEG
       │         capture.get(CAP_PROP_FRAME_COUNT)   ◄── libavformat
       ├─ wanted = sample_frame_indices(meta.frame_count, n_frames)   preprocess.py:830
       │    step = frame_count / n ;  [min(frame_count-1, int(i*step)) for i in range(n)]
       └─ one grab()/retrieve() pass, decoding only the wanted indices
  └─ build_video_bundle(frame_count=meta.frame_count, duration_s=meta.duration_s, …)
```

`probe_video` and `extract_frames` both open the capture through
`decode_env.open_video`, the single `VideoCapture` construction site (Batch 8), so
probe and decode cannot resolve to different backends.

### 3. Which decoder/backend decisions affect it?

| Decision | Affects `frame_count`? | In `decode_identity`? |
| --- | --- | --- |
| Video backend (FFMPEG vs MSMF vs GSTREAMER) | **yes** — different demuxers | **yes** (`video_backend`) |
| OpenCV version | yes — bundles a different FFmpeg | **yes** (`opencv_version`) |
| **libavformat version** (the demuxer) | **yes — this is the library that reads the count** | **NO — only libavcodec is parsed** |
| libavcodec version (the frame decoder) | no — decoding frames does not change the count | yes (`ffmpeg_version`) |
| OS / architecture | possibly (different FFmpeg build) | yes (`platform_system`, `platform_machine`) |
| `DFP_VIDEO_BACKEND` / priority-table override | yes (picks the demuxer) | neutralised — backend is named, Batch 8 |
| The container itself (VFR, damaged index, `nb_frames=0`) | **yes — this is where a demuxer's estimate diverges** | n/a — per-file, not per-deployment |

Measured on this build (`frame_count_reproducibility_study.py` S4):
`avcodec: YES (58.134.100)` and `avformat: YES (58.76.100)` are **separate
libraries with different versions**, and `decode_env._build_facts()` extracts
only the avcodec line via `_AVCODEC_RE`. `avformat` never enters
`DecodeEnvironment` (`test_the_decode_identity_still_omits_the_demuxer`).

### 4. Reproducibility defect, correctness defect, documentation defect, or design limitation?

**All four have a piece; the decomposition is the point.**

| Piece | Class | Status |
| --- | --- | --- |
| **R6-a** — `sample_frame_indices` is uniform over the declared count, so a miscount of ~one sampling step (`frame_count/n` frames) moves most scored frames | **design limitation** (a timestamp-fraction sampler would be robust; an index-fraction one is not) | measured, pinned; the fix is a serving-path change (§9) |
| **R6-b** — `frame_count` is trusted verbatim; nothing cross-checks it against the decodable frame count or `duration × fps` | **design limitation** shading into **correctness** (an undercount silently drops the tail of the clip from the verdict) | characterised; the fix is a serving-path change (§9) |
| **R27** — the identity that gates the result cache (Batch 9) and the benchmark comparison (Batch 11) omits libavformat, the library that produces `frame_count` | **reproducibility defect** | **OPEN**, migration analysed (§8) |
| the DECODE_DETERMINISM §3 line "no validation of `frame_count` exists" as the *whole* record of it | **documentation defect** (understated a structural gap as a footnote) | fixed by this document; the DECODE_DETERMINISM text is preserved and annotated |

It is **not** a correctness defect in the R15 sense (no answer computed from the
wrong *file*) and **not** non-deterministic within one deployment: `probe_video`
is a pure function of the bytes and the build (`extract_frames` control, Batch 8
E12).

### 5. Can two deployments publish different `frame_count` under identical current identities?

**Yes.** "Identical current identities" = same `effective_model_version` **and**
same `decode_identity()` (the `.d.` segment) ⇒ same `model_version` ⇒ same
content-cache key `by-content/{sha256}/{model_version}.json`.

`decode_identity()` = `fingerprint({opencv, backend, ffmpeg(=avcodec), system,
machine})`. Two OpenCV builds with the same reported avcodec version, same
backend name, same OS/arch, but a **different libavformat** — an ordinary
distro-vs-prebuilt difference — produce the **same** `decode_identity` and can
disagree on `CAP_PROP_FRAME_COUNT` for any container whose count the demuxer
estimates (VFR media, `nb_frames=0`, a partially rebuilt index).

**Executed** (`study` S5, `test_two_demuxers_disagreeing_by_one_frame_share_a_
cache_key`) with `MiscountingDemuxer`, a double that shaves one frame:

```
true demuxer : frame_count=200  p_fake=0.9739  verdict=manipulated
-1  demuxer  : frame_count=199  p_fake=0.9546  verdict=manipulated
sampled frame indices moved : 11/16
decode_identity()           : c20a064a / c20a064a   (identical)
bundle.model.version        : identical
content cache key           : identical
```

A cache populated by the first deployment serves its verdict to the second, which
would have scored a different set of frames. The `p_fake` delta here (0.019) is
the **stub upper bound** (the stub hashes frame bytes; DECODE_DETERMINISM E9);
the honest per-frame estimate for a real model is the surrogate's `|Δp| ≈ 0.005`,
and the decision-flip rate against the shipped band is E10's 1.3–1.6%.

**What this batch could not do:** reproduce the *trigger* on one build. S3
attempted trailing garbage, truncation, a flipped `mdat` byte, and both backends;
every intact container was counted exactly and every damaged one was refused —
consistent with E1. The divergence needs media this box cannot author (true VFR)
and a second libavformat, which is exactly the build-dependence R6 is about.

### 6. Does Batch 9 runtime identity already protect against the defect?

**Partially.** It protects against a `frame_count` divergence caused by anything
the decode identity captures — a backend switch, an OpenCV upgrade, an
**avcodec** version change, an OS/arch move — because each of those changes the
`.d.` segment and splits the cache namespace. It does **not** protect against a
divergence caused by **libavformat** alone (R27), and:

- `decode_identity` is **passed in, never detected** (`RUN_IDENTITY.md` §6.1
  analogue) — a worker whose configured `.d.` does not match its actual FFmpeg
  build fails `verify_version_alignment` at startup only if the *composed* string
  differs, which it will not if only avformat moved;
- `_AVCODEC_RE` falls back to `"present, version unreported"` if OpenCV's build
  string format changes, which would collapse even a real avcodec difference to a
  constant.

### 7. Does Batch 11 benchmark identity already protect against it?

**Partially, and less than it looks.** `regression_gate` refuses a cross-decoder
comparison **iff the caller passes differing `decode_identity` values**. Since
`decode_identity` omits libavformat, two benchmark runs that differ only in
demuxer carry the **same** `decode_identity`, the gate does **not** refuse, and a
`frame_count`-driven score shift reads as a model delta — the exact failure
`EVALUATION_IDENTITY.md` §3 describes, one level deeper. Additionally:

- the evaluation package **never decodes** (`EVALUATION_IDENTITY.md` §1) —
  `frame_count` is computed entirely inside the caller's `predictor`, so the
  harness cannot even observe it;
- `decode_identity` on `BenchmarkRun` is **optional**; a run that omits it gets a
  "note", not a guarantee.

### 8. Does any cache, benchmark, calibration, or forensic workflow consume `frame_count`?

**No — none consumes the value.** Traced (`study` S6,
`test_frame_count_reaches_the_wire_but_not_the_cache_or_the_harness`):

| Workflow | Reads `frame_count`? |
| --- | --- |
| Content cache key (`workers/io.py::content_key_for`) | **no** — `sha256 + model_version` only, computed by the API before any decode |
| Job idempotency key | **no** — `sha256:model_version` |
| `BenchmarkRun` / `regression_gate` / `dataset_fingerprint` | **no** — `evaluation/` contains zero references |
| `CalibrationWitness` | **no** |
| `SplitLock` | **no** |
| `DecodeEnvironment` / `decode_fingerprint` | **no** — no per-file fields |
| **The wire** (`MediaInfo.frame_count`, and `MediaInfo.duration_s` derived from it) | **yes** |
| **`sample_frame_indices`** → which frames are decoded → logits → `p_fake` → `verdict` → `FrameScore.frame_idx` values | **yes — the only score-affecting consumer** |

So `frame_count`'s reproducibility impact is entirely **indirect**: through the
frames it selects, into `p_fake`. And `p_fake`'s cross-deployment protection is
the `.d.` cache-key segment — which is where R27 bites.

### 9. Can the issue be closed without SCRFD / trained weights / contract changes / schema changes?

| Sub-finding | Closeable within those four constraints? |
| --- | --- |
| **R6-a** (uniform sampling amplifies a miscount) | **No.** The robust fix is timestamp-fraction sampling (`sample at duration·i/n`, seek to the nearest decoded frame), which changes *which frames every video is scored on* — a serving-path change that moves every video `p_fake` and requires recalibration. No contract/schema change, but not "smallest" and not free. |
| **R6-b** (no validation of `frame_count`) | **Partially.** A cheap post-hoc check — "did the grab loop reach EOF near the declared count?" — is additive and needs no contract field, but acting on a failure means either a new log-only signal (weak) or a new abstention path (a serving-path behaviour change: files that pass today would newly abstain). |
| **R27** (decode identity omits the demuxer) | **Yes.** Adding libavformat (and arguably libswscale) to `DecodeEnvironment.identity` is an `inference/decode_env.py`-only change. It touches **no** contract, **no** `RUN_SCHEMA_VERSION`, **no** `CALIBRATION_WITNESS_VERSION`, **no** OpenAPI. It **does** move the `.d.` segment of `STUB_ENGINE_VERSION` and forces a one-time cache-namespace miss for every deployment — a *cache generation*, precedented exactly by Batch 9 and Batch 11. |

### 10. If not, what is the smallest truthful closure?

**This batch: characterise, split, and pin — change no behaviour.** Specifically:

1. This document + the study + 16 pins convert R6's `frame_count` residual from a
   one-line "Unresolved" footnote into a bounded, decomposed finding with an
   executed fail-before.
2. **R27 is recorded as a distinct finding** with the migration fully analysed
   (§8 below) and an explicit recommendation: **fold libavformat into
   `decode_identity` in the batch that next legitimately regenerates the `.d.`
   segment, or with the video-face-stage / SCRFD work that rebuilds frame
   handling wholesale.** Shipping it here would force a global cache generation
   for a residual with **zero measured occurrences** on this build, and would
   ripple through ~6 live identity assertions and several historical identity
   docs — disproportionate, and against the "don't rewrite history" rule.
3. **R6-a / R6-b's real fix (timestamp-fraction sampling + a decodable-count
   cross-check) is made an explicit acceptance criterion for the video face
   stage** (§9), the same way Batch 8 made crop stability under ±4 LSB an
   acceptance criterion for the SCRFD migration rather than a discovery
   afterwards.

---

## Threat model

| # | Entry point | Mechanism | Reaches `p_fake`? | Reaches an identity string? |
| --- | --- | --- | --- | --- |
| F1 | libavformat estimates `nb_frames` differently across builds | VFR / `nb_frames=0` / damaged index | **yes** — via `sample_frame_indices` | **no** — R27 |
| F2 | a different backend's demuxer counts differently | FFMPEG ↔ MSMF ↔ GSTREAMER | yes | **yes** — `video_backend` |
| F3 | an OpenCV upgrade bundles a different FFmpeg | version bump | yes | **yes** — `opencv_version` |
| F4 | `frame_count` is an over-count | tail indices never reached; `grab()` EOFs early | yes — shorter curve, or `TOO_FEW_FRAMES` | n/a |
| F5 | `frame_count` is an under-count | tail of the clip silently never sampled | **yes — silently** | n/a |
| F6 | container lies (`mvhd` duration ≠ sample table) | authored maliciously or by a broken muxer | yes | n/a |
| F7 | run-to-run variation within one build | — | **no** — `probe_video` is pure (E12) | — |

F1 and F5 are the reproducibility/correctness core. F2–F3 are already covered by
Batch 9. F7 does not occur.

## Dependency trace

```
libavformat (demuxer)
   │  CAP_PROP_FRAME_COUNT
   ▼
probe_video ──► VideoMeta.frame_count ─┬─► MediaInfo.frame_count      (wire, Stable)
   │                                    ├─► duration = frames/fps ──► MediaInfo.duration_s (wire, Stable)
   │                                    └─► sample_frame_indices(frame_count, n_frames)
   │                                             │  uniform: step = frame_count/n
   │                                             ▼
   │                                        wanted indices ──► extract_frames grab/retrieve
   │                                             ▼
   │                                        frames (N,H,W,3) ──► backend.score ──► per-frame logits
   │                                             ▼
   │                                        topk_indices · pool_at ──► fuse ──► p_fake ──► verdict
   │                                             ▼
   │                                        FrameScore.frame_idx values (wire)
   │
   └─ NOT ──► content cache key · idempotency key · effective_model_version
             · decode_identity · gate_fingerprint · calibration_fingerprint
             · face_selection identity · BenchmarkRun · CalibrationWitness · SplitLock
```

`decode_identity` sits **beside** this trace, not on it: it keys the cache that
stores the `p_fake` at the bottom, and it is computed from
`fingerprint({opencv, backend, avcodec, system, machine})` — the demuxer at the
top of the trace is absent from it.

## Artifact flow

| Artifact | Carries `frame_count` (or a function of it)? | Guards a cross-deployment divergence? |
| --- | --- | --- |
| `EvidenceBundle.media.frame_count` | the raw value | no — published, never compared |
| `EvidenceBundle.media.duration_s` | `frame_count / fps` | no |
| `EvidenceBundle.video.frame_scores[].frame_idx` | the *sampled indices* (a function of `frame_count`) | no — published, never compared |
| `EvidenceBundle.video.frames_sampled` | `len(sample_frame_indices(...))` | no |
| `EvidenceBundle.p_fake` / `verdict` | downstream of the sampled frames | **partially** — the cache key's `.d.` segment, which omits the demuxer (R27) |
| content-cache pointer `by-content/{sha}/{model_version}.json` | no | this is where the divergence would need to be caught, and is not |
| `BenchmarkRun.decode_identity` | no | omits the demuxer (R27), and the harness never decodes anyway |
| `DecodeEnvironment.digest` (forensic) | no per-file fields | records the environment, not the count |

The published bundle **already contains** enough to detect a divergence after the
fact (`frame_count`, `duration_s`, every `frame_idx`) — two bundles for one file
under two demuxers are distinguishable by eye. What is missing is (a) anything
that *compares* them and (b) anything that stops the cache serving one for the
other.

## Fail-before proof

`tests/test_frame_count_reproducibility.py`, executed:

- **`test_two_demuxers_disagreeing_by_one_frame_share_a_cache_key`** — the
  headline fail-before. `MiscountingDemuxer` (real `probe_video`, `frame_count −
  1`) vs the real probe, same bytes, real `analyze_video` both times:
  `frame_scores` indices differ, `p_fake` differs, `(sha256, model.version)` is
  **identical**. The residual, executed.
- **`test_frame_count_is_in_no_identity_string`** — the same run shows
  `decode_identity()` and `bundle.model.version` byte-identical across the two
  counts, and greps `decode_env.py` / `version.py` for `frame_count` (absent).
- **`test_the_decode_identity_still_omits_the_demuxer`** — the R27 tripwire:
  `getBuildInformation()` lists `avformat` as a distinct line; `_build_facts()`
  returns the avcodec version; `DecodeEnvironment.to_dict()` contains no
  `avformat`. Fails if a future batch adds it — forcing this document's R27
  status to be updated.
- **`test_a_one_frame_miscount_moves_most_sampled_indices_in_the_E2_regime`** and
  **`test_the_amplifier_scales_with_the_miscount_relative_to_the_sampling_step`**
  — the amplifier, re-measured and pinned as a lower bound, with the honest shape
  (severe when the miscount ≳ one sampling step; mild for a 1-frame error on a
  long clip with a small budget).
- **`test_the_amplifier_measurement_is_portable_where_a_real_trigger_is_not`** —
  `sample_frame_indices` calls only `{list, range, int, min, len}`; this suite is
  reproducible anywhere, unlike a real cross-demuxer divergence.

Fail-before for R27's *fix* is not feasible here (it needs a second libavformat);
the tripwire test is the substitute — it will fail the moment the fix lands.

## Migration analysis

### This batch

**Zero.** No source behaviour changed. `contracts/*`, `inference/*` (except the
new test/study), `backend/*`, `evaluation/*`, `schema_out/*`,
`backend/openapi.snapshot.json` untouched.

| | Moves? | Why |
| --- | --- | --- |
| `CONTRACT_VERSION` | no | no field added/removed/retyped |
| `RUN_SCHEMA_VERSION` | no | `BenchmarkRun` untouched |
| `CALIBRATION_WITNESS_VERSION` | no | witness untouched |
| `STUB_ENGINE_VERSION` | no | no fingerprint input changed |
| `effective_model_version` / `decode_identity` / `gate_fingerprint` / `calibration_fingerprint` / face-selection identity | no | none reads `frame_count` |
| cache generation | no | `model_version` unchanged |
| `schema_out/` / OpenAPI snapshot | no | no schema change |

### The recommended R27 fix (deferred), when it lands

Adding `avformat` (and `swscale`) to `DecodeEnvironment.identity`:

| | Moves? | Detail |
| --- | --- | --- |
| `CONTRACT_VERSION` | **no** | `ModelInfo.version` is still `str`; only its value gains entropy — exactly Batch 9's precedent |
| `RUN_SCHEMA_VERSION` | **no** | `BenchmarkRun.decode_identity` is already `str \| None`; old rows keep their old digest, new rows get the finer one; `from_dict` unaffected |
| `CALIBRATION_WITNESS_VERSION` | **no** | `CalibrationWitness.decode_identity` is already a field; `compute_witness` dispatches on the payload's own `schema_version` (Batch 11), so a witness stored under the old digest still verifies — **provided the digest rule is not re-pinned to a constant** (the Batch 11 hazard) |
| `STUB_ENGINE_VERSION` | **yes** | `…f.19559387.d.c20a064a` → `…f.19559387.d.<new>`; update `backend/config.py` + the live assertions in `test_crop_geometry.py`, `test_decode_determinism.py`, `test_run_identity.py`, `face_selection_study.py` |
| cache generation | **yes** | one-time full miss per file; no data loss (results stay addressable by job id); precedented by Batch 9 §6.2 |
| historical docs (`RUN_IDENTITY.md`, `FACE_SELECTION_IDENTITY.md`, `THRESHOLD_BAND_DESIGN.md`, …) | **no** | `c20a064a` in those is a measurement that was true when written — annotate forward-looking claims, do not rewrite |
| OpenAPI / `schema_out` | **no** | — |

This is a **cache-generation batch**, not a schema batch. It is the correct home
for the fix and it is out of scope here for the same reason Batch 8 deferred
R12's remediation 2.

## Guarantees gained

1. `frame_count`'s definition, its single score-affecting consumer, and its
   two-wire-field reach are traced and pinned.
2. The amplifier is measured with its honest shape and pinned as a lower bound,
   with the note that *this* measurement is portable and a real trigger is not.
3. **R27 is a named finding**, not a clause inside R6 — the decode identity's
   omission of libavformat is measured (`avcodec 58.134.100` ≠ `avformat
   58.76.100`) and tripwired.
4. The cross-deployment hazard is **executed**: one identity, two counts, two
   `p_fake`s, one cache key.
5. `frame_count` is proven absent from every cache/benchmark/calibration/forensic
   key, so the blast radius is exactly "the sampled frames, into `p_fake`".
6. A future validator or a demuxer-in-identity change trips a test.

## Guarantees NOT gained

1. **Two demuxers still disagree** on an estimated count. Nothing here makes them
   agree; R27's fix would make them *detectable*, the Batch 9/11 guarantee shape.
2. **`frame_count` is still trusted verbatim.** An under-count still silently
   drops the tail of a clip from the verdict.
3. **Frame sampling is still uniform over the declared count.** A robust
   timestamp-fraction sampler is a serving-path change deferred to the video face
   stage.
4. **The trigger is still unreproduced** on this build — as it was in Batch 8.
   The residual is real by construction (S4/S5), not by a captured occurrence.
5. **`decode_identity` is still passed in, never detected**, and `_AVCODEC_RE`
   can still silently coarsen to `"present, version unreported"`.
6. **No cache generation**, so nothing here retroactively separates a verdict
   already cached under a demuxer that miscounted.

## Exact status recommendation

- **R6 (`frame_count` residual) → PARTIALLY CLOSED.** Its design-property half
  (R6-a, R6-b) is characterised, bounded and pinned; its fix is a serving-path
  change scoped to the video face stage with an explicit acceptance criterion
  (§9). Its reproducibility half is promoted to **R27**.
- **R27 (decode identity omits the demuxer) → OPEN.** Migration analysed;
  recommended to land in a cache-generation batch or with the video face stage.
  Tripwired.
- **DECODE_DETERMINISM §3 "Container metadata divergence — Unresolved" row** —
  preserved; a pointer to this document is the only edit.

---

## §9. Acceptance criteria for the video face stage (carry-forward)

When the video path gains SCRFD frame handling, it must also:

1. **Sample by timestamp fraction, not index fraction** — pick target times
   `duration · i / n` and decode the nearest available frame, so a ±1 (or ±k
   small) error in the declared count barely moves the scored moments. Re-run the
   amplifier study against the new sampler; the ≥50%-of-indices-move property
   must be gone.
2. **Cross-check the declared count** against the frames actually decodable
   (the single pass already knows where EOF fell), and route a gross mismatch
   (declared ≫ decodable, or the grab loop ending far before the last wanted
   index) to an explicit abstention rather than a silently-shortened curve.
3. **Fold libavformat into `decode_identity`** as part of the same cache
   generation (§8), so a demuxer that counts differently is at least detectable.

Recorded here so it is a criterion for that migration, not a discovery after it.
