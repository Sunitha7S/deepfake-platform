# Decode identity completeness — resolving R27 (Batch 20)

> **Batch 23 (R29) note:** the model-identity `.f.` segment became a composite
> (face + video selection policy) — the `.d.` decode identity this document
> covers is **unchanged**. B23 also recorded **R30** (image decode libraries —
> libjpeg-turbo / libpng / libwebp — outside `decode_identity`, the image
> analogue of R27; only `cv2.__version__` proxies them) and **N8** (the
> video-shaped decode-id string applied to the image path). Both need a
> `decode_identity` value extension = their own batch. See
> `docs/VIDEO_SELECTION_IDENTITY.md`.

> **Outcome: R27 closed for the version half; one residual and one carry-forward
> remain, both stated.** `decode_identity()` fingerprinted, for FFmpeg, only the
> **libavcodec** version. The video decode path links four FFmpeg libraries and
> `getBuildInformation()` versions each separately (measured: avcodec 58.134.100,
> **avformat 58.76.100**, avutil 56.70.100, **swscale 5.9.100**). libavformat
> produces `CAP_PROP_FRAME_COUNT` and so decides which frames are scored (R27,
> Batch 19); libswscale converts the decoded YUV frame to BGR in `retrieve()`
> (Batch 8 E4 measured a difference there). All four of the cache key, the
> deployment identity, the benchmark identity and the calibration identity were
> blind to a difference in any of those three.
>
> **This batch grows the FFmpeg component of `decode_identity` from libavcodec
> alone to every FFmpeg library on the video path** — the smallest expansion that
> closes the hole. `STUB_ENGINE_VERSION`'s `.d.` segment moves `c20a064a` →
> `a97ccb68` (a one-time cache generation, precedented by Batch 9). **No
> `CONTRACT_VERSION`, `RUN_SCHEMA_VERSION`, `CALIBRATION_WITNESS_VERSION` or
> OpenAPI change.**

Re-runnable: `python tests/decode_identity_completeness_study.py` (8 experiments).
Proof and named doubles: `tests/test_decode_identity_completeness.py` (22 tests).
Batch 19's tripwire `test_the_decode_identity_still_omits_the_demuxer` fired on
this change and was rewritten (`test_the_decode_identity_now_covers_the_demuxer_
batch_20`) — preserve-and-refute.

**State reviewed.** 47 suites, `CONTRACT_VERSION` 1.0.0, `RUN_SCHEMA_VERSION`
1.2.0, `CALIBRATION_WITNESS_VERSION` 1.2.0. R27 opened in Batch 19
(`FRAME_COUNT_REPRODUCIBILITY.md`); the Batch 19 report recommended "libavformat
(and possibly libswscale)" but told this batch not to assume it. It is re-derived
below.

---

## Preserve-and-refute

**Batch 8 (`DECODE_DETERMINISM.md` §1.1, T1)** listed video metadata as a threat
and marked it "not observed to diverge (E1)". True on one build; it understated
the *structural* gap — E1 tested whether two backends agree, not whether the
identity would catch it if they didn't.

**Batch 9 (`RUN_IDENTITY.md` §1.1, §5.3)** defined `DecodeEnvironment.identity`
with `ffmpeg_version` described as "**the libavcodec version**". That description
was accurate and the scope was deliberate — but "the FFmpeg that matters for
decoding" is more than libavcodec, and Batch 9 did not say why the other
libraries were out (they were not considered). **Refuted here:** libavformat and
libswscale are on the video path, measurably, and were omitted by oversight
rather than by a measured decision like the one that (correctly) excluded the CPU
fields.

**Batch 11 (`EVALUATION_IDENTITY.md` §5)** carried the same `decode_identity()`
onto `BenchmarkRun` and `CalibrationWitness` — "OpenCV version, backend,
libavcodec, OS, machine". The benchmark and calibration protections inherited the
libavcodec-only scope, so §13 gap 6 ("`decode_identity` excludes the CPU fields …
two genuinely different-decoding machines could share an identity") had a
sibling it did not name: two machines with different FFmpeg **demuxers**.

**Batch 19 (`FRAME_COUNT_REPRODUCIBILITY.md`)** opened R27 and recommended
"fold libavformat (+swscale) into `decode_identity` … deferred to a
cache-generation batch". This batch is that batch, and the recommendation is
confirmed with one refinement: **avutil** joins them (shared base, one more
parsed line) and **avresample** is explicitly excluded (audio only).

---

## The ten questions

### 1. Every component that can alter the video decode

Traced from `VideoPipeline.analyze_video` → `extract_frames` → `backend.score`
(`study` E1):

| Component | Can alter | Represented in `decode_identity` (post-B20) |
| --- | --- | --- |
| `temp_media_file` (spool bytes) | nothing — byte-exact | n/a |
| `resolve_video_backend` → `VideoCapture(path, FFMPEG)` | which demuxer + decoder | **yes** — `video_backend` |
| **libavformat** (demuxer) | `frame_count`, `fps`, `POS_MSEC`, packet order, seeking, container rotation/SAR | **yes (new)** — `ffmpeg_version` |
| **libavcodec** (frame decoder) | decoded YUV pixels, coded W/H | yes — `ffmpeg_version` |
| **libswscale** (`sws_scale` in `retrieve()`) | the BGR pixels returned; chroma upsampling, range | **yes (new)** — `ffmpeg_version` |
| **libavutil** (pixfmt descriptors, rational ts math) | timestamp rounding, pixfmt handling | **yes (new)** — `ffmpeg_version` |
| OpenCV `cap_ffmpeg` wrapper | `POS_MSEC` derivation, EOF, retrieve behaviour | yes — `opencv_version` |
| `sample_frame_indices(frame_count, n)` | which indices are scored | not an env component — pure arithmetic (its *input*, `frame_count`, is libavformat's) |
| `cv2.cvtColor` BGR→RGB | channel order | yes — `opencv_version`; B8 E6 measured CPU-independent |
| `cv2.resize` (`INTER_AREA`/`INTER_LINEAR`) | the final square frame | yes — `opencv_version`; B8 E6 measured CPU-independent |
| Intel IPP (resize/cvtColor kernels on x86) | resize/cvt output | via `opencv_version` (wheel-pinned); **not independently measured** — residual |
| `np.ndarray.astype(float32)` | widening | deterministic, elementwise |
| OS / architecture | which prebuilt FFmpeg, which kernels | yes — `platform_system` / `platform_machine` |
| CPU baseline / dispatch (SIMD) | opencv kernels (B8 E6: not a pixel); **swscale kernels — not measured** | `digest` only, **not** `identity` — see §"CPU fields" |
| `DFP_VIDEO_BACKEND` / `OPENCV_VIDEOIO_PRIORITY_*` | backend choice | neutralised — backend is named (B8) |
| run-to-run within one process | — | none — `probe_video`/`extract_frames` are pure (B8 E12) |

### 2. Which were represented before this batch

`opencv_version`, `video_backend`, **libavcodec** (as `ffmpeg_version`),
`platform_system`, `platform_machine`. Plus, in the forensic `digest` only:
`cpu_baseline`, `cpu_dispatch`, `backend_source`, `python_version`.

### 3. Which were missing

**libavformat, libswscale, libavutil.** (avresample is present in the build but
not on the video path — §6.) Also, coarsely: the opencv-python **wheel patch
version** (`cv2.__version__` reports `4.8.1`, the wheel is `4.8.1.78` — §"wheel
gap").

### 4. For each missing component — the omission class

| Missing | Correctness | Reproducibility | Benchmark | Cache | Documentation |
| --- | --- | --- | --- | --- | --- |
| **libavformat** | no (no decoder is "wrong") | **yes** — different `frame_count` → different frames scored → different `p_fake` under one identity (executed, Batch 19: 11/16 sampled indices moved) | **yes** — `regression_gate` would not refuse | **yes** — a cached verdict served across | — |
| **libswscale** | no | **yes** — B8 E4 measured the YUV→RGB conversion stage moving pixels ("chroma upsampling differs too"); a swscale version bump could do the same within one backend | **yes** | **yes** | — |
| **libavutil** | no | **marginal** — rational-arithmetic changes to `POS_MSEC` rounding would move `FrameScore.t_ms` / segment boundaries, not the verdict; pixfmt handling is shared with the above | benchmark-only / forensic | low | included at zero cost |
| wheel patch version | no | **residual** — only if OpenCV's own imgproc changes between wheel patches with no FFmpeg change; B8 E6 makes that low-risk | low | low | documented, not closed |

None is correctness-critical in the R15 sense (an answer computed from the wrong
*file*). All of libavformat/libswscale are **reproducibility-critical** and reach
**all four** protections.

### 5. Is libavformat sufficient? libswscale?

**libavformat is necessary but not sufficient.** It closes R27 as demonstrated
(frame selection). **libswscale is on the video pixel path** — OpenCV's
`CvCapture_FFMPEG::retrieveFrame` calls `sws_scale` to convert the decoded frame
(YUV420P for H.264/mp4v) to BGR24, and B8 E4 already measured that conversion
stage as a source of pixel difference (the non-affine residual it attributes to
chroma upsampling). **libavutil** is the shared base (pixfmt descriptors, the
rational math behind timestamps) and moves on the same FFmpeg release; it is one
more parsed line. The correct unit is **"the FFmpeg libraries OpenCV links for
Video I/O"**, not "the one we happened to demonstrate".

### 6. avresample

**Excluded.** `libavresample` is FFmpeg's *audio* sample-rate converter,
deprecated in FFmpeg 4 and removed in FFmpeg 5. This codebase decodes video
frames only — `study` E6 confirms no `swr_*` / `CAP_PROP_AUDIO` / audio-stream
API anywhere in `inference/`. Including it would fragment the cache on a library
the video path never calls.

### 7. Can two deployments disagree on sampled frames while publishing identical decode identity / cache keys / benchmark identity / calibration identity?

**Before this batch: yes to all four.** `study` E4, executed with two
`DecodeEnvironment`s identical except `avformat 58.76.100` → `59.16.100`:

```
BEFORE Batch 20 (decode_identity = avcodec only):
  BLIND      content cache key
  BLIND      deployment identity (/health, alignment)
  BLIND      benchmark identity (regression_gate refuses)
  BLIND      calibration identity (witness digest differs)
protections blind: 4/4

AFTER Batch 20:
  SEPARATED  content cache key
  SEPARATED  deployment identity
  SEPARATED  benchmark identity
  SEPARATED  calibration identity
protections blind: 0/4
```

`test_a_libavformat_difference_now_separates_all_four_protections` and
`test_the_benchmark_gate_refuses_across_a_demuxer_difference` pin it. **Yes: the
cache and benchmark protections were incomplete, for a class of environment
difference broader than the CPU-field gap Batch 11 §13.6 already named.**

### 8. Smallest identity expansion that closes the hole

Grow the one FFmpeg field from `"<avcodec version>"` to
`"avcodec X / avformat Y / avutil Z / swscale W"` — one string, one field, one
place in the digest. No new `DecodeEnvironment` field, so `to_dict`,
`differences`, `describe` and every test that iterates the field list are
unchanged. `_build_facts()` parses four `getBuildInformation` lines instead of
one; the fallbacks (avcodec-only, "present, versions unreported", `None`) are
kept.

### 9. Migration

| Question | Answer | Why |
| --- | --- | --- |
| Contract change | **no** | `ModelInfo.version` is still `str`; only its value gained entropy — Batch 9's precedent exactly |
| Schema regeneration | **no** | no schema field changed |
| OpenAPI change | **no** | the version string is a value, stripped from the structural signature; `test_the_openapi_snapshot_does_not_carry_the_decode_segment` |
| `RUN_SCHEMA_VERSION` bump | **no** | `BenchmarkRun.decode_identity` is already `str \| None`; old rows keep their coarse string, new rows get the finer one, `from_dict` is unaffected |
| `CALIBRATION_WITNESS_VERSION` bump | **no** | the witness digest is over the `decode_identity` **string it stored**, not a recomputation — a witness written with `"c20a064a"` still verifies (`test_an_old_calibration_witness_still_verifies_after_the_change`). Batch 11 bumped because it *added the field to the digest*; the field is already there |
| Cache generation | **yes** | `STUB_ENGINE_VERSION` `.d.` segment `c20a064a` → `a97ccb68`; one-time full cache miss per file, no data loss (results stay addressable by job id) — Batch 9 §6.2 |
| Benchmark incompatibility handling | **none required** | a run made just before the fix carries `decode_identity="c20a064a"`; one just after carries `"a97ccb68"`; `regression_gate` refuses to compare them (two recorded, different → refusal) — correct, and self-healing once the fleet is on the new scheme. `decode_identity=None` (pre-Batch-11 rows) still → a note. `test_an_old_benchmark_row_still_loads_and_a_cross_boundary_comparison_refuses` |

Live pins updated: `backend/config.py` (the literal + docstring),
`tests/test_crop_geometry.py` (asserted the model-identity prefix instead of the
full runtime string — Batch 15's guarantee was about the model half, not the
decode segment). Historical identity docs keep `c20a064a` — a measurement true
when written; forward-looking claims are annotated.

### 10. Guarantees gained and not gained — below.

---

## Threat model

| # | Difference between two deployments | Reaches the sampled frames / pixels? | Caught, post-B20? |
| --- | --- | --- | --- |
| D1 | video backend (FFMPEG ↔ MSMF ↔ GSTREAMER) | yes | yes — `video_backend` (B9) |
| D2 | OpenCV version (`cv2.__version__`) | yes | yes — `opencv_version` (B9) |
| D3 | **libavformat version** | yes — `frame_count` → frame selection | **yes (B20)** |
| D4 | **libswscale version** | yes — YUV→BGR conversion | **yes (B20)** |
| D5 | libavutil version | timestamps, pixfmt | yes (B20) |
| D6 | libavcodec version | yes — decoded pixels | yes (B9) |
| D7 | OS / arch | yes | yes — `platform_*` (B9) |
| D8 | CPU SIMD baseline/dispatch | opencv: measured no (E6); **swscale: not measured** | **no — digest only** (residual) |
| D9 | opencv-python wheel patch (`4.8.1.78` vs `4.8.1.86`), same FFmpeg | OpenCV imgproc only | **no** (residual, low-risk) |
| D10 | same FFmpeg version strings, different compilation | possibly | **no** — a version string cannot see it (residual) |
| D11 | `frame_count` estimated differently by one demuxer for a VFR file | yes | run-to-run stable; cross-build → D3 catches the version, D10 is the residual |

D3–D5 are what this batch closes. D8–D10 are residuals, stated.

## Dependency trace

```
getBuildInformation()  "Video I/O -> FFMPEG"
   avcodec 58.134.100   avformat 58.76.100   avutil 56.70.100   swscale 5.9.100
        │                    │                    │                  │
        └──────────── _build_facts() ── ffmpeg_version (one string) ─┘
                              │
   opencv_version ────────────┤
   video_backend ─────────────┤
   platform_system/machine ───┤
                              ▼
                DecodeEnvironment.identity  = fingerprint({opencv, backend, ffmpeg, system, machine})
                              │
        ┌─────────────────────┼───────────────────────────┬──────────────────────┐
        ▼                     ▼                           ▼                      ▼
 compose_runtime_version  BenchmarkRun.decode_identity  CalibrationWitness   /health · verify_
   model + .d.<identity>     (regression_gate)            .decode_identity     version_alignment
        │                                                   (in the digest)
        ▼
 content cache key  by-content/{sha256}/{model_version}.json
```

## Artifact flow

| Artifact | Carried the coarse identity | Carries the finer one | Migration |
| --- | --- | --- | --- |
| `STUB_ENGINE_VERSION` (config literal) | `.d.c20a064a` | `.d.a97ccb68` | edit the literal (done) |
| `ModelInfo.version` in **new** bundles | model + `.d.c20a064a` | model + `.d.a97ccb68` | automatic |
| content-cache pointers | keyed on the old runtime version | keyed on the new one | one-time full miss, no data loss |
| **stored** `EvidenceBundle`s | keep their old `.d.` value | — | readable by job id; content pointer re-created on next analysis |
| `BenchmarkRun.decode_identity` (history JSONL) | old rows: `"c20a064a"` or `None` | new rows: `"a97ccb68"` | old rows load; cross-boundary comparison refuses (correct) |
| `CalibrationWitness.decode_identity` | old: `"c20a064a"` (in its digest) | new: `"a97ccb68"` | old witnesses verify against their stored string; no re-digest |
| `backend/openapi.snapshot.json` | never carried a `.d.` value | — | untouched |
| `schema_out/*.json` | never carried it | — | untouched |

## Fail-before proof

`tests/test_decode_identity_completeness.py`:

- **`test_a_libavformat_difference_now_separates_all_four_protections`** — the
  headline. Two `DecodeEnvironment`s differing only in the avformat version:
  post-B20 all four protections separate; the `LegacyAvcodecOnlyIdentity` double
  shows all four were blind.
- **`test_the_benchmark_gate_refuses_across_a_demuxer_difference`** —
  `regression_gate` on two real `BenchmarkRun`s carrying the finer identities:
  refuses, with a "decode" failure.
- **`test_the_calibration_witness_binds_the_finer_identity`** — two witnesses,
  two demuxer identities, two digests.
- **`test_a_libswscale_difference_now_moves_the_identity`** /
  **`_libavutil_`** — the other two libraries.
- **`test_the_legacy_double_reproduces_the_pre_batch_20_identity`** — the double
  equals `c20a064a` on this build (the Batch 9/19 constant).
- **`test_an_old_benchmark_row_still_loads_and_a_cross_boundary_comparison_
  refuses`** / **`test_an_old_calibration_witness_still_verifies_after_the_
  change`** — the migration, executed.

Fail-before for a *real* library swap is not feasible (one FFmpeg on this box);
the `replace()`-on-`DecodeEnvironment` doubles are the substitute, and the
legacy-double comparison proves they exercise the actual gap.

## CPU fields — Batch 9's decision, preserved and slightly qualified

`cpu_baseline` / `cpu_dispatch` stay in the **forensic digest only**, not the
identity. Batch 8 E6 measured `resize` / `cvtColor` / `imdecode` byte-identical
under `OPENCV_CPU_DISABLE`, so keying the cache on SIMD would halve a mixed-fleet
hit rate for no correctness. **Qualification (new):** E6 tested OpenCV's own
kernels, not **libswscale's** SIMD path, which `sws_scale` selects at runtime.
A swscale build with a different SIMD default *could* convert YUV→BGR differently
on two CPUs sharing a swscale version. Not measured, not closed — recorded as
residual D8, and `test_the_cpu_fields_are_still_excluded_from_the_identity` pins
the current behaviour so a future change is loud.

## Migration analysis (summary)

| Constant | Before | After |
| --- | --- | --- |
| `STUB_ENGINE_VERSION` `.d.` | `c20a064a` | `a97ccb68` |
| `CONTRACT_VERSION` | 1.0.0 | 1.0.0 |
| `RUN_SCHEMA_VERSION` | 1.2.0 | 1.2.0 |
| `CALIBRATION_WITNESS_VERSION` | 1.2.0 | 1.2.0 |
| `LOCK_VERSION` | untouched | untouched |
| cache generation | — | **one**, per-file miss, no data loss |
| `DecodeEnvironment` field set | 9 fields | 9 fields (`ffmpeg_version` value broadened) |

`parse_runtime_version` / `parse_model_version` stay total on both eras — the
`.d.` segment is still one 8-char digest, so no parser sees a new shape.

## Guarantees gained

1. A cached verdict cannot cross a **libavformat, libswscale or libavutil**
   difference — the three FFmpeg libraries the video path links besides
   libavcodec.
2. `regression_gate` refuses a benchmark comparison across any of those, on the
   same footing as a cross-backend one.
3. A `CalibrationWitness` binds all four FFmpeg library versions.
4. `/health` and `verify_version_alignment` carry them; a worker on a different
   FFmpeg build fails startup rather than sharing a cache namespace.
5. The demuxer that decides which frames are scored (R27) is inside every
   identity that matters.
6. Batch 19's tripwire fired and was honoured — the finding did not quietly
   become true in the dark.

## Guarantees NOT gained

1. **Two FFmpeg builds reporting identical version strings but compiled
   differently** (a distro patch, a different `--enable-*` set, a changed chroma
   default) still share an identity (residual D10). A version string cannot see
   this; a content hash of the linked `.so`/`.dll` could, and is out of scope.
2. **libswscale's SIMD path** is not in the identity (residual D8) — E6 did not
   measure it, and the CPU-field exclusion is inherited on that assumption.
3. **The opencv-python wheel patch version** is dropped by `cv2.__version__`
   (residual D9). Capturing the FFmpeg libs directly moots it for the decode
   path; OpenCV's own imgproc changing between wheel patches is the uncovered
   sliver, rated low-risk by E6.
4. **R6-a (uniform frame sampling) and R6-b (no `frame_count` validation)** are
   untouched — B20 is an identity change only. Their fix is still the
   video-face-stage acceptance criterion (`FRAME_COUNT_REPRODUCIBILITY.md` §9).
5. **The trigger** — a real demuxer that miscounts — is still unreproduced on
   this build. The finding is closed by construction (the identity now covers the
   library), not by a captured occurrence.
6. **No back-fill.** Historical bundles, runs and witnesses keep their coarse
   `.d.` value; `None` stays `None`. Writing the new digest onto an old artifact
   would assert it was produced under this environment, which nobody observed
   (Batch 9 / Batch 11's rule).

## Exact status recommendation

- **R27 → CLOSED for the version half.** `decode_identity` now covers every
  FFmpeg library on the video path; all four protections separate a demuxer /
  scaler difference; migration is a cache generation with no schema or contract
  change. Executed, not argued.
- **Residuals D8 (swscale SIMD), D9 (wheel patch), D10 (same-version different
  build)** — recorded, pinned where possible, not closed. They are a strictly
  finer grain than a version string reaches and none has a measured occurrence.
- **R6-a / R6-b** — unchanged; still the video-face-stage carry-forward.

R27 can be marked **closed** in the audit with the residual list attached, the
same way R12 was closed with its "two decoders still disagree" caveat.
