# Decode determinism: threat model, measurement, and what is guaranteed

**Status:** S8 Batch 8. Closes audit finding **R6** as *measured and partially
mitigated*. Opens **R12** and **R13**.
**Contract impact:** none. `CONTRACT_VERSION` 1.0.0, `EvidenceBundle`,
`ThresholdInfo`, OpenAPI and every model contract unchanged.

Every number below comes from `tests/decode_study.py`, which is runnable:

```bash
python tests/decode_study.py
```

The guarantees are pinned by `tests/test_decode_determinism.py` (20 tests).

---

## 0. The answer

> **Can the same media, model version, calibration, threshold and code produce
> different verdicts solely because decoding changed?**

**Yes. Measured, on one machine, with one OpenCV build, without recompiling
anything.**

The two video backends this build registers — FFMPEG and MSMF — decode the same
file to **zero identical frames**. On photographic content the mean absolute
difference is 2.2/255 with 85% of pixels changed; on noise it is 12.6/255 with
96% changed. Running the real pipeline over both:

```
FFMPEG  p_fake=0.9039545661  model_version=stub-0-w7f45fada+c.739c3fec.g.ae232cba
MSMF    p_fake=0.8914745894  model_version=stub-0-w7f45fada+c.739c3fec.g.ae232cba
                             ^^^^^^^^^^^^^^^^^^^ identical
```

Which backend runs is decided at *runtime* by OpenCV's priority table, which the
environment variable `OPENCV_VIDEOIO_PRIORITY_<BACKEND>` overrides. So before
Batch 8: **an environment variable could change a published verdict, and every
version string in the response would be byte-identical.**

How often that changes a *decision* rather than a score: with the measured
per-frame shift of |Δp| ≈ 0.005 for a smooth scorer, **1.31% of decisions**
drawn from a broad score distribution cross a boundary of the shipped abstention
band. At the stub's measured |Δp| ≈ 0.026, 7.2%.

---

## 1. Phase 1 — threat model

### 1.1 Where nondeterminism can enter

| # | Entry point | Where in the code | Reachable? |
|---|---|---|---|
| T1 | **Video metadata** — `fps`, `frame_count` | `probe_video` reads `CAP_PROP_*` | Yes; **not observed to diverge** (E1) |
| T2 | **Frame selection** | `sample_frame_indices(meta.frame_count, n)` | Yes, and it is the biggest amplifier (E2) |
| T3 | **Frame ordering** | sequential `grab()`; no reordering exists | No — one pass, monotone index |
| T4 | **Keyframe seeking** | not used by the shipped path; `legacy_extract` double | Yes for the legacy path (E5) |
| T5 | **Frame timestamps** | `_position_ms` → `CAP_PROP_POS_MSEC` | Yes; semantics differ by call site (E5) |
| T6 | **Decoder pixel output** | `capture.retrieve()` | **Yes, and it is real** (E3, E4) |
| T7 | **Colour conversion** | `to_rgb` → `cv2.cvtColor` | Not observed (E6) |
| T8 | **Resize** | `resize` → `INTER_AREA`/`INTER_LINEAR` | Not observed (E6) |
| T9 | **Image decode** | `decode_image` → `cv2.imdecode` | Not observed (E7) |
| T10 | **Floating-point variation** | `normalize`, `astype` | No — elementwise, no reduction |
| T11 | **Face detection** | `detect_faces` → Haar cascade | Yes — amplifies T6 into a moved crop (E8) |
| T12 | **Backend selection** | `cv2.VideoCapture(path)` with no `apiPreference` | **Yes — via an environment variable** (E11) |

T12 was not in R6's original statement and is the one that makes the rest
practically reachable: R6 assumed a *different build* was required. It is not.

### 1.2 Which outputs are affected

Tracing each threat forward through the pipeline:

| Output | T1/T2 metadata | T5 timestamps | T6 pixels | T11 crop | T12 backend |
|---|---|---|---|---|---|
| **Frame selection** (`indices`) | **direct** | no | no | no | via T1 |
| **`FrameScore.t_ms`** | via fps | **direct** | no | no | via T5 |
| **Branch logits** | via frames scored | no | **direct** | **direct** | **direct** |
| **`p_fake` / verdict** | **direct** | no | **direct** | **direct** | **direct** |
| **Calibration** (temperature) | direct | no | direct | direct | direct |
| **Threshold fitting** | direct | no | **direct** (E10: Δ=0.0057) | direct | direct |
| **Benchmark metrics** | direct | no | rank-dependent (E10) | direct | direct |
| **`effective_model_version`** | **no** | **no** | **no** | **no** | **no** |

The last row is the finding. Every threat reaches the verdict; none reaches the
identity string that is supposed to answer "would this deployment produce the
same number?".

There is a second-order effect worth stating separately: a decoder change moves
the **fitted threshold** (measured, E10: 0.17546 → 0.16977). The threshold *is*
in `calibration_fingerprint`. So the version moves if and only if somebody
refits — which means the version changes for the *wrong reason* (a refit) and
never for the right one (the decoder). Both directions are wrong: refit and the
cache invalidates on a decoder change it cannot name; do not refit and the cache
serves verdicts decoded by a different decoder.

---

## 2. Phase 2 — measurement

Thirteen experiments. Each states a hypothesis and the failure mode expected if
it holds, *before* measuring. Four did not reproduce and are reported as
negative results.

### E1 — Container metadata across decoders — **not reproduced**

*Hypothesis:* FFMPEG and MSMF disagree on `fps`/`frame_count`.
*Expected failure:* `sample_frame_indices` selects a different clip.

```
smooth/avc1    FFMPEG=(25.0, 60, 256, 256)  MSMF=(25.0, 60, 256, 256)  agree
smooth/mp4v    FFMPEG=(25.0, 60, 256, 256)  MSMF=(25.0, 60, 256, 256)  agree
noise/mp4v     FFMPEG=(25.0, 60, 240, 240)  MSMF=(25.0, 60, 240, 240)  agree
noise/MJPG     FFMPEG=(25.0, 60, 240, 240)  MSMF=(25.0, 60, 240, 240)  agree

Truncated containers:  95% / 80% / 50% of bytes -> both backends refuse
```

Both agree on every intact container tested, and both refuse every truncated
one. **Not reproduced on this box**; the fixtures are all constant-frame-rate
files written by OpenCV itself, which is the easy case. Variable-frame-rate
media and containers with a damaged index are the untested regime.

### E2 — Metadata amplification — **measured**

*Hypothesis:* a one-frame error in `frame_count` shifts most sampled indices.

```
reference frame_count=60 -> 48 indices
  frame_count= 58: 38/48 sampled indices differ (79%)
  frame_count= 59: 29/48 sampled indices differ (60%)
  frame_count= 61: 18/48 sampled indices differ (38%)
  frame_count= 62: 33/48 sampled indices differ (69%)
  frame_count= 65: 42/48 sampled indices differ (88%)
  frame_count= 70: 44/48 sampled indices differ (92%)
```

The trigger is unconfirmed here; the amplification is not. If any decoder ever
miscounts by **one frame**, 60% of the analysed frames change — a different clip
in every meaningful sense, from a metadata field nothing validates.

### E3 — Pixel agreement across decoders — **measured, the central result**

```
smooth/avc1    n= 60  identical=  0  max|d|= 15  mean|d|= 2.2236  differing=85.46%
smooth/mp4v    n= 60  identical=  0  max|d|= 19  mean|d|= 2.4040  differing=86.60%
noise/mp4v     n= 60  identical=  0  max|d|=202  mean|d|=12.6007  differing=96.09%
noise/XVID     n= 60  identical=  0  max|d|=202  mean|d|=12.6007  differing=96.09%
noise/MJPG     n= 60  identical=  0  max|d|=119  mean|d|=11.6144  differing=95.95%
```

Zero identical frames, in every codec tested, including MJPG — where each frame
is an independent JPEG, so this is not an inter-frame prediction artefact.

### E4 — Cause: systematic, not noise — **measured**

```
per-channel mean signed delta (R,G,B): -0.9996 -1.4659 -1.0015
least-squares MSMF->FFMPEG:
    +0.99936  -0.00024  +0.00031
    +0.00239  +0.99879  +0.00266
    -0.00006  +0.00033  +0.99863
    -1.21382  -1.32313  -1.20470     <- bias
residual after the affine fit: mean|r|=2.0113  max|r|=13.98
```

A near-identity matrix with a bias near −1.2 per channel: most of the difference
is a **levels/range interpretation** (studio-range vs full-range luma). The
residual is not zero, so **chroma upsampling differs too**.

This matters for remediation. It is not float noise, it will not shrink with
better arithmetic, and no amount of care inside this codebase will make two
decoders agree. The only available controls are *pin one* and *record which*.

### E5 — Seek strategies — **measured**

```
pixel mismatches, sequential vs seek: 0/7
  frame   7: sequential=  280.000 ms   after-seek=    0.280 ms   delta=  279.720
  frame  15: sequential=  600.000 ms   after-seek=    0.600 ms   delta=  599.400
  frame  55: sequential= 2200.000 ms   after-seek=    2.200 ms   delta= 2197.800
```

Pixels agree on this all-keyframe clip. **Timestamps do not, by three orders of
magnitude.** `CAP_PROP_POS_MSEC` read after `set(POS_FRAMES, i)` and before
`read()` reports the seek target in the backend's own units, not a presentation
time. The shipped path reads it after `grab()` and is correct — E5 is the
measurement that says *why* that ordering is load-bearing, and it is now pinned
by a test.

### E6 — CPU dispatch — **not reproduced**

*Hypothesis:* `resize`/`cvtColor`/`imdecode` differ across SIMD paths.

```
default (AVX2)                                    resize=c56fb139058c7418
OPENCV_CPU_DISABLE=AVX2,AVX,FP16,SSE4_2,SSE4_1    resize=c56fb139058c7418
```

Byte-identical, in a subprocess with the dispatch actually disabled at load
time, and also identical under in-process `setUseOptimized(False)` and
`setNumThreads(1)`. **R6's "different SIMD paths" hypothesis does not reproduce
on this build.** Pinned by a test so a future divergence is noticed.

### E7 — Independent image decoders — **not reproduced**

```
noise 720p   jpg q88 / q40 / progressive / png   identical=True  max|d|=0
smooth 480p  jpg q88 / q40 / progressive / png   identical=True  max|d|=0
```

OpenCV's libjpeg-turbo and Pillow's produce bit-identical output, progressive
JPEG included. **R6's "different libjpeg" hypothesis does not reproduce.** The
image decode path is materially safer than the video one, and that asymmetry was
not previously known.

### E8 — Face detection as an amplifier — **measured, with a caveat**

```
baseline box: x=88 y=98 w=302 h=302
  +/-1 LSB: box changed 27/30, lost 26/30, max |dx|=6 |dy|=6 |dw|=13
  +/-2 LSB: box changed 29/30, lost 27/30, max |dx|=4 |dy|=2 |dw|=5
  +/-4 LSB: box changed 29/30, lost 21/30, max |dx|=8 |dy|=6 |dw|=13
```

A perturbation *smaller than* the measured decoder delta changes the crop almost
always, and loses the face outright most of the time — which routes the input to
`NO_FACE_DETECTED` instead of a score.

**Caveat, stated plainly:** the fixture is a synthetic face sitting near this
cascade's detection boundary, so the *loss* rate overstates what a real
photograph would do. The result that survives the caveat is the direction and
the ordering: the crop is not stable under decoder-scale perturbation, and the
frozen architecture replaces this detector with SCRFD anyway. Re-measure on real
faces when SCRFD lands; that is R13.

### E9 — Does it reach the logits? — **measured**

```
surrogate (smooth)   logit mean|d|=0.043243  p mean|d|=0.006432  correlation= 0.997110
stub (hash)          logit mean|d|=2.054822  p mean|d|=0.343484  correlation=-0.179601
```

Two scorers, deliberately. The **stub hashes the frame bytes**, so any pixel
change produces an uncorrelated logit — correlation −0.18 means the per-frame
curve is destroyed outright. That is the correct upper bound and a useless
estimate.

The **surrogate** is a smooth Lipschitz function of the pixels (gradient energy
on the normalised frame). It is not a deepfake detector; it stands in for a CNN
in the one respect that matters — it responds proportionally rather than
hashing. Its |Δp| ≈ 0.005 is the honest estimate for what a real model would do.

### E10 — Verdicts, thresholds, benchmark metrics — **measured**

```
clips=24
logit    mean|d|=0.040027  max|d|=0.043346
p        mean|d|=0.005752  max|d|=0.006450
ranking  identical order: False   inverted pairs: 2/276
ROC AUC  FFMPEG=1.000000  MSMF=1.000000
fitted threshold  FFMPEG=0.17546139  MSMF=0.16976583  delta=0.00569556
```

AUC is rank-only, so it moves only when the ranking moves — 2 of 276 pairs
inverted here, and this fixture is separable enough that AUC stays at 1.0 and is
therefore uninformative. **Inverted pairs is the sensitive statistic and the one
to watch**; AUC on a harder corpus would move with it.

The **fitted threshold** is a quantile of the scores themselves and moves with
any shift at all. It moved by 0.0057 — and it is the number that reaches
`calibration_fingerprint`.

Decisions flipped, against the shipped band `[0.5210, 0.7099)`, scores ~
Beta(2,2), 200 000 draws:

| \|Δp\| | decisions changed | source |
|---|---|---|
| 0.0047 | **1.310%** | measured, surrogate per frame (E9) |
| 0.0058 | **1.626%** | measured, surrogate on the corpus (E10) |
| 0.0260 | 7.215% | measured, stub end-to-end |
| 0.0500 | 13.990% | reference |

### E11 — End to end — **measured**

```
FFMPEG  decode_digest=6d534da9  verdict=manipulated  p_fake=0.9039545661
MSMF    decode_digest=a90156c7  verdict=manipulated  p_fake=0.8914745894

frame indices identical : True
timestamps identical    : True
decoded pixels identical: False
p_fake |d|              : 0.0124799768
effective_model_version : *** IDENTICAL ***
decode digest           : differs (detectable)
```

Different pixels, different score, identical model version. The verdict did not
flip *on this clip* — its `p_fake` is 0.19 from the nearest band edge. E10 gives
the rate at which it would.

### E12 — Control — **measured**

```
extract_frames x5 -> 1 distinct result: DETERMINISTIC
decode_image  x5 -> 1 distinct result: DETERMINISTIC
resize / normalize x5 -> 1 distinct result each
```

One build decodes repeatably. Without this control nothing above would be
interpretable.

---

## 3. Phase 3 — guarantees

| Source of drift | Class | Evidence |
|---|---|---|
| Run-to-run within one process/build | **Eliminated** | E12; `test_decoding_is_repeatable_within_one_build` |
| Frame ordering | **Eliminated** | One sequential pass, monotone index; no reorder path exists |
| Float variation in `normalize`/`astype` | **Eliminated** | Elementwise, no reduction; `test_normalisation_is_elementwise_and_therefore_order_free` |
| Keyframe seeking in the shipped path | **Eliminated** | Not used since Batch 1; `test_extraction_performs_no_seeks` |
| Timestamp semantics | **Eliminated** | Read after `grab()`; hazard pinned by `test_the_position_after_a_seek_is_not_a_timestamp` |
| Backend chosen by environment | **Eliminated** | `DFP_VIDEO_BACKEND` + `PREFERRED_VIDEO_BACKENDS`; `open_video` is the only capture site |
| Which decoder ran | **Detectable** | `decode_environment().digest`; E11 shows the digest separating two runs the version string cannot |
| Decoder pixel output | **Tolerated** | E3/E4 — a levels and chroma disagreement between independent implementations. Unfixable from here; pinning + recording is the whole of the mitigation |
| Face-crop instability | **Tolerated** | E8, and the detector is a placeholder due for replacement |
| CPU/SIMD dispatch | **Tolerated (not reproduced)** | E6; pinned by a test so a future divergence surfaces |
| Image decoder differences | **Tolerated (not reproduced)** | E7; pinned by a test |
| Container metadata divergence | **Unresolved** | E1 not reproduced, E2 shows 60% amplification if it ever happens. No validation of `frame_count` exists |

> **Batch 19 followed up on the `frame_count` row.** This entry was correct but
> understated: `CAP_PROP_FRAME_COUNT` is a **libavformat** quantity, and
> `decode_identity` (the Batch 9 cache-key segment and the Batch 11 `BenchmarkRun`
> field) captures **libavcodec** only — measured, `avcodec 58.134.100` vs
> `avformat 58.76.100` on this build. So two deployments sharing a `decode_identity`
> can disagree on `frame_count`, score different frames, and collide on one cache
> key — executed. Split out as **R27**; the amplifier is re-measured and pinned.
> See [`FRAME_COUNT_REPRODUCIBILITY.md`](FRAME_COUNT_REPRODUCIBILITY.md).
>
> **Batch 20 closed R27.** `decode_identity`'s FFmpeg component grew from
> libavcodec alone to every FFmpeg library on the video path — **avcodec +
> avformat + avutil + swscale** (avresample excluded, audio only). libavformat
> was not the whole gap: libswscale performs the YUV→BGR conversion in
> `retrieve()` — the stage E4 above measured — and was also omitted. All four
> protections (cache key, deployment identity, benchmark identity, calibration
> identity) now separate a demuxer/scaler difference, 4/4 blind → 0/4, executed.
> One cache generation, `.d.` `c20a064a` → `a97ccb68`.
> See [`DECODE_IDENTITY_COMPLETENESS.md`](DECODE_IDENTITY_COMPLETENESS.md).
| Cross-OS / cross-build agreement | **Unresolved** | Untestable from one machine. The digest makes a difference *visible*; it cannot make it *not happen* |
| Decoder identity in the cache key | **Unresolved** | E11: `effective_model_version` excludes it, by necessity (§5) |

### What "Eliminated" means here

That the mechanism cannot occur, not that it was fixed and hoped about. Each
elimination is structural: there is one capture site, so the backend cannot vary
between probe and decode; `normalize` performs no reduction, so there is no
summation order; the shipped extractor performs no seeks, and a test counts
them.

### What "Detectable" does not mean

A fingerprint cannot make two libavcodec builds agree on a pixel. It ensures
that when they disagree, the two results are **distinguishable rather than
silently pooled**. That is the same guarantee the split lock makes about
corpora, and it is worth exactly as much: it works only if something compares
it.

---

## 4. Phase 4 — findings

Two new findings. Both are supported by measurement; neither is speculative.
R6 itself is reclassified rather than left open.

### R6 — Decode nondeterminism — **measured; partially mitigated**

Original text: "OpenCV/ffmpeg builds do not agree on the pixels of a given
frame. Different SIMD paths, different libjpeg, different H.264 decoder
behaviour on damaged streams."

What Batch 8 found: the *conclusion* is correct and understated; two of the
three *mechanisms* named are wrong.

- **Different SIMD paths** — not reproduced (E6).
- **Different libjpeg** — not reproduced (E7).
- **Decoder disagreement** — reproduced emphatically (E3), and it does not
  require a different build at all. Two backends in one build, selected by an
  environment variable, suffice.

Mitigation shipped: the backend is named rather than inherited, and the decode
environment is captured and fingerprinted. Residual risk: everything in the
"Tolerated" and "Unresolved" rows above.

### R12 — The decoder is outside every identity string (Medium, measured)

**Reproduction.** `python tests/decode_study.py`, experiment E11. Or:

```bash
DFP_VIDEO_BACKEND=FFMPEG python -c "..."   # p_fake=0.9039545661
DFP_VIDEO_BACKEND=MSMF   python -c "..."   # p_fake=0.8914745894
```

Both print `model_version=stub-0-w7f45fada+c.739c3fec.g.ae232cba`.

**Impact.** Affects **reproducibility**, not correctness — neither decoder is
wrong, they disagree. The result cache is keyed on `effective_model_version`, so
a cached verdict decoded by one backend is served to a request that would have
been decoded by the other. A benchmark comparison across two workers with
different decoders reports a model regression that is a decoder change. Measured
size: 1.3–1.6% of decisions on a broad score distribution (E10).

**Severity: Medium.** Below R1 (which broke the cache for *every* result) and
above R3, because it is silent, it survives every existing check, and it is
triggered by an environment variable rather than by a deliberate act.

**Proposed remediation, in order of cost:**

1. *Shipped in Batch 8.* Name the backend (`PREFERRED_VIDEO_BACKENDS`,
   `DFP_VIDEO_BACKEND`) so it stops depending on OpenCV's priority table, and
   capture `decode_environment()` so a difference is at least expressible.
2. *Proposed, not implemented.* Record `decode_fingerprint()` on `BenchmarkRun`
   and make `regression_gate` refuse a comparison across differing digests, the
   way it already refuses one across differing dataset fingerprints. This needs
   a `RUN_SCHEMA_VERSION` bump (1.1.0 → 1.2.0) with a default so old rows still
   load — precedented by Batch 4 — and it is deliberately left for a batch whose
   brief permits a schema change.
3. *Proposed, and argued against below (§5).* Fold the digest into
   `effective_model_version`.

### R13 — Face-crop instability amplifies decoder drift (Low–Medium, measured)

**Reproduction.** `python tests/decode_study.py`, experiment E8.

**Impact.** Affects **both** reproducibility and correctness. A perturbation
smaller than the measured decoder delta moves the detected box by up to 8px and
frequently loses the face, which converts a scored input into a
`NO_FACE_DETECTED` abstention. A moved crop feeds the model different pixels
than it would otherwise have seen, so the pixel-level drift of E3 is amplified
by an unbounded factor rather than passed through.

**Severity: Low–Medium**, and the range is the honest part. The fixture is a
synthetic face near this cascade's detection boundary, so the measured *loss*
rate overstates a real photograph. The *direction* is not in doubt.

**Proposed remediation.** None in this batch, deliberately. The frozen
architecture replaces Haar with SCRFD, and tuning a detector that is scheduled
for deletion is wasted work. What should happen: re-run E8 against SCRFD on real
faces when it lands, and treat crop stability under ±4 LSB as an acceptance
criterion for that migration rather than discovering it afterwards.

---

## 5. Why the decode digest is not in `effective_model_version`

> **Superseded by Batch 9. The reasoning below is wrong and is kept because it
> is the reasoning the next batch had to refute.** See
> [`RUN_IDENTITY.md`](RUN_IDENTITY.md). The decoder now reaches the cache key by
> a separate *runtime identity* — `effective_model_version` still excludes it,
> deliberately, so that model changes and environment changes stay
> distinguishable. What was wrong was the claim that it could not reach the API
> at all.

It is the obvious remediation and the question deserves a direct answer, because
the decoder is unambiguously relevant to "would this deployment produce the same
number?" — which is exactly what that string is for.

It is refused for a structural reason, not a philosophical one. Batch 0
invariant 3: **`backend` must stay free of `inference`.** The API process
computes `effective_model_version` to build cache keys and to answer `/health`;
it does not import `inference`, and importing it would pull cv2 and numpy into
the API for the sake of one string. `cv2.getBuildInformation()` is not available
there and cannot be made available without undoing that invariant.

A fingerprint that only the worker can compute cannot be part of a key the API
must compute. So the digest lives where the comparison actually happens —
evaluation artifacts and run records — and the limitation is written down and
tested (`test_the_effective_model_version_does_not_cover_the_decoder`) rather
than assumed away.

The honest consequence: **the result cache can serve a verdict produced by a
different decoder.** That is R12's residual risk and it is not closed by this
batch.

**Where the argument fails** (Batch 9): every premise is true and the conclusion
does not follow. The API does not compute `Settings.model_version` either — it
reads a configured literal that `verify_version_alignment` holds to what the
engines report, failing startup on a mismatch. That mechanism was built in
Batch 0 for this exact shape of problem, and is described two paragraphs above
its own dismissal.

---

## 6. Where things live

| what | where |
|---|---|
| Backend resolution, environment capture, fingerprint | `inference/decode_env.py` |
| The single video capture site | `inference/decode_env.open_video`, used by `probe_video` and `extract_frames` |
| The thirteen experiments | `tests/decode_study.py` |
| The pinned guarantees and negative results | `tests/test_decode_determinism.py` |
| Seek/memory doubles from Batch 1 | `tests/test_frame_extraction.py` |
