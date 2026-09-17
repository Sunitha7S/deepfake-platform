# Batch 24 — Design: Image Decode Identity Completeness (R30)

**Type:** design document only. No code in this task.
**Author's method:** every claim below re-derived from source at the cited path
and, where marked *(live)*, executed on the implementation box
(Windows / opencv-python 4.8.1). Prior batch narratives
(`B24_CANDIDATE_ANALYSIS.md`, the B20 report, `DECODE_IDENTITY_COMPLETENESS.md`)
were **not** trusted; each was re-verified.

---

## 1. R30, re-derived from source

### 1.1 Current `DecodeEnvironment.identity` *(live)*

`inference/decode_env.py:181-234`. `identity` is:

```python
return fingerprint({
    "opencv":  self.opencv_version,     # cv2.__version__
    "backend": self.video_backend,      # "FFMPEG" | "MSMF" | ...
    "ffmpeg":  self.ffmpeg_version,     # "avcodec X / avformat Y / avutil Z / swscale W"
    "system":  self.platform_system,    # platform.system()
    "machine": self.platform_machine,   # platform.machine()
})
```

Executed on this box:

```
identity (.d.) = a97ccb68
ffmpeg_version = "avcodec 58.134.100 / avformat 58.76.100 / avutil 56.70.100 / swscale 5.9.100"
opencv         = 4.8.1
backend        = FFMPEG (preferred)
system/machine = Windows / AMD64
```

`fingerprint({opencv:"4.8.1", backend:"FFMPEG", ffmpeg:"avcodec 58.134.100 / …",
system:"Windows", machine:"AMD64"})` → `a97ccb68`, and
`backend/config.py:41` `STUB_ENGINE_VERSION` ends `.d.a97ccb68` *(live —
`STUB_ENGINE_VERSION.split(".d.")[-1] == "a97ccb68"`)*.

The five keys are the **whole** membership. `cpu_baseline` / `cpu_dispatch` /
`backend_source` / `python_version` are on `DecodeEnvironment` but **not** in
`identity` — they are in `digest` (`:236-258`) only. Reason on record
(`:211-224`): B8-E6 ran `resize` / `to_rgb` / `imdecode` under
`OPENCV_CPU_DISABLE` in a subprocess and got byte-identical output, so keying a
mixed-hardware fleet on CPU flags costs hit rate for no correctness.

### 1.2 Current `.d.` fingerprint construction

- `contracts/version.py:365-384` `compose_runtime_version(*, model, decode)` →
  `f"{model}{DECODE_MARKER}{decode}"` where `DECODE_MARKER = ".d."`
  (`:155`). `decode` must be non-empty and must not itself contain `.d.`
  (`:382`). It is otherwise **an opaque string** — no length, charset or format
  check.
- `decode_identity()` (`decode_env.py:400-408`) returns
  `decode_environment().identity` — a bare `fingerprint(...)` output, i.e.
  `[0-9a-f]{8}` *(live: `FINGERPRINT_LENGTH == 8`)*, **marker-free**.
- Both pipelines compose it identically:
  `inference/image_pipeline.py:105-107` and `inference/video_pipeline.py:159-160`
  → `compose_runtime_version(model=self._spec.effective_version,
  decode=decode_identity())`. **`decode_identity()` takes no arguments and is
  media-kind-agnostic** — the image and video engines call the same function and
  get the same string.
- `parse_runtime_version` (`:387-403`) splits on the first `.d.` by
  `str.partition`; a version with no `.d.` parses to `decode=None` and that is
  not an error. **The parser never inspects the fingerprint value.**

*(live)* `fingerprint` is order-independent and a new key changes the hash:
`fingerprint({a:1}) != fingerprint({a:1, z:None})`.

### 1.3 Current image decode coverage — **the gap**

`inference/preprocess.py:191-217` `decode_image(data)` →
`cv2.imdecode(buffer, cv2.IMREAD_COLOR)` (`:211`). `imdecode` dispatches by the
byte signature to the **bundled codec libraries**:

| Accepted MIME (`contracts/api.py:33` `ALLOWED_IMAGE_MIME`) | Decoder |
|---|---|
| `image/jpeg` | **libjpeg-turbo** |
| `image/png` | **libpng** (+ **zlib** for the DEFLATE stream) |
| `image/webp` | **libwebp** |

**None of these libraries is in `DecodeEnvironment.identity`.** The only proxy
is `opencv_version` (`cv2.__version__`), which:

1. is one string for the whole build — it moves when OpenCV moves, not when a
   bundled codec is upgraded in isolation;
2. *(live, `test_decode_identity_completeness.py:409-421`)* **drops the
   opencv-python wheel's 4th component** (`cv2.__version__ == "4.8.1"`, the wheel
   is `4.8.1.78`) — so two wheels that rebuilt only the codec bundle can report
   the same `cv2.__version__`.

`imdecode`'s codec libraries are the image analogue of the FFmpeg libraries R27
found missing for video. This is **R30**.

### 1.4 Current video decode coverage — for contrast

`decode_env.py:325` `_VIDEO_FFMPEG_LIBS = ("avcodec", "avformat", "avutil",
"swscale")`, parsed by `_AVLIB_RE` (`:326-328`) from the `Video I/O → FFMPEG`
block of `getBuildInformation()`, joined into the one `ffmpeg_version` string
(`_build_facts`, `:332-371`), and that string is a key in `identity`. R27/B20
closed the video path. `avresample` is excluded (audio; `test_…:220-232` proves
no audio API is used). **B24 does not touch any of this.**

### 1.5 Exact role of `cv2.getBuildInformation()` *(live)*

A single free-text diagnostic string. OpenCV exposes **no structured API** for
codec versions (`_build_facts` docstring `:341-344` says so, and it is still
true). The relevant block:

```
Media I/O:
    ZLib:                        build (ver 1.2.13)
    JPEG:                        build-libjpeg-turbo (ver 2.1.3-62)
      SIMD Support Request:      YES
      SIMD Support:              NO
    WEBP:                        build (ver encoder: 0x020f)
    PNG:                         build (ver 1.6.37)
    TIFF:                        build (ver 42 - 4.2.0)
    JPEG 2000:                   build (ver 2.5.0)
    OpenEXR:                     build (ver 2.3.0)
    ...
```

Format facts that the parser must survive:

- Each codec line is `^\s*<NAME>:\s+<impl> (ver <version>)$`. `<impl>` is
  `build` (OpenCV built its own copy), `build-libjpeg-turbo` (named
  implementation), or a system path.
- **WebP** reports an ABI number, not a semver: `(ver encoder: 0x020f)` here;
  other builds report `(ver decoder: 0x0209, encoder: 0x020f)`. The parser must
  capture the whole parenthetical verbatim, not assume `x.y.z`.
- **libjpeg-turbo SIMD** is two extra indented lines under `JPEG:` —
  `SIMD Support Request:` and `SIMD Support:` — each `YES`/`NO`. On this box the
  request is `YES` and the actual is `NO` (no NASM at build time).
- The block header is `Media I/O:` on its own; on some builds it is
  `Media I/O: ` with a trailing space (seen here). Whitespace-tolerant matching
  required.
- `getBuildInformation()` can raise or be unavailable (already handled:
  `_build_facts` returns a marker string, never raises — `:346-349`). Same
  contract for the new parser.

### 1.6 Do JPEG / PNG / WebP versions participate in identity today? — **No** *(live)*

`grep 'JPEG\|libjpeg\|PNG\|libpng\|WEBP\|libwebp' inference/decode_env.py` → the
only hit is the class docstring's discussion of the *video* path. The `identity`
preimage is the five keys in §1.1; none is a codec version.
`grep -rln 'a97ccb68\|decode_identity' fixtures_out/ schema_out/
backend/openapi.snapshot.json` → **nothing** — the decode identity is not on any
wire.

### 1.7 Does zlib participate in identity today? — **No**

Not parsed, not stored, not in any fingerprint. (Design decision on whether it
*should* be added: §2.6.)

### 1.8 Does libjpeg-turbo SIMD configuration participate today? — **No**

Not parsed. B8-E6's CPU-independence result **does not cover it**:
`OPENCV_CPU_DISABLE` disables *OpenCV's* runtime dispatch, not the compile-time
`WITH_SIMD` state of the *bundled libjpeg-turbo* (its SIMD IDCT/upsampling is
baked into the static `.lib` at wheel-build time). So "SIMD Support: YES vs NO"
is an un-measured axis. (Design decision: §2.6.)

### 1.9 Evidence chain, condensed

```
imdecode (preprocess.py:211)  ──uses──▶  libjpeg-turbo / libpng+zlib / libwebp
        │                                        │
        │                          getBuildInformation() "Media I/O" reports each,
        │                          separately, with a version cv2.__version__ hides
        ▼
image_pipeline.model_version  ──=──▶  compose_runtime_version(model, decode=decode_identity())
        │                                                              │
        │                                     decode_identity() = fingerprint({opencv,
        │                                     backend, ffmpeg, system, machine})
        ▼                                                              │
Settings.model_version  ──keys──▶  content cache (service.py:98),      │
        idempotency key (service.py:227), /health, verify_version_alignment
                                                                       ▼
   ⇒  two hosts, different bundled libjpeg-turbo, identical decode_identity,
      identical cache key  ⇒  a wrong verdict served from cache (false hit).
```

R30 is confirmed present, unmitigated, on the primary modality.

---

## 2. Candidate solutions

### Option A — one `image_codec_version` field on `DecodeEnvironment`, folded into `identity`

Parallel to `ffmpeg_version`. A new frozen-dataclass field holding
`"jpeg <impl> (ver …) / jpeg-simd <YES|NO> / png … / webp … / zlib …"`, parsed
by a new `_image_codec_facts()` (a **new** function — `_build_facts()`'s
signature and its five call sites stay byte-identical), added as one key
`"image_codec"` to the `identity` fingerprint dict.

| Axis | Assessment |
|---|---|
| **Reproducibility** | Closes R30 on the same guarantee as R27: two builds that disagree on a bundled image codec now produce different `.d.`, so a cache/benchmark/witness cannot silently pool them. |
| **Cache-key correctness** | `identity` gains one key → `.d.` value moves once → **one cache generation** (both namespaces — §4, §5). No structural change to the key. |
| **Migration cost** | Identical profile to B20: one `.d.` regeneration, `result_ttl_s` expiry of orphans, no back-fill, no witness/contract bump. **Low.** |
| **Parser impact** | **Zero.** `decode_identity()` still returns a marker-free `[0-9a-f]{8}`; `compose_runtime_version` / `parse_runtime_version` / `_MODEL_MARKERS` never inspect the value. |
| **Deploy impact** | A `STUB_ENGINE_VERSION` move → the R34 rolling-deploy constraint applies (coordinated restart, or `DFP_SKIP_VERSION_ALIGNMENT` for the window). Same as B20/B22/B23. |
| **Future extensibility** | A sixth `identity` key is cheap; the `_image_codec_facts()` regex is the only new parsing surface. Adding TIFF/AVIF later = extend `_IMAGE_CODEC_LIBS`. |
| **Interaction with N8** | Does **not** close N8 (§3): the image path still carries the FFmpeg component and now the video path also carries the image component — symmetric over-fragmentation. N8 → **narrowed**. |

### Option B — extend the existing codec fingerprint structure (reuse `ffmpeg_version`)

Rename `ffmpeg_version` → `codec_versions` (or append image libs into the same
string / same `_VIDEO_FFMPEG_LIBS`-style list).

| Axis | Assessment |
|---|---|
| **Reproducibility** | Same coverage as A. |
| **Cache-key correctness** | Same one-generation move. |
| **Migration cost** | **Higher than A.** Renaming `ffmpeg_version` touches `to_dict`, `differences`, `describe`, `test_run_identity.py:427`, `test_decode_identity_completeness.py` (`_build_facts()[0]`, `mismatched_ffmpeg`, `LegacyAvcodecOnlyIdentity.avcodec_of` which splits on `" / "` and `startswith("avcodec ")`), `test_frame_count_reproducibility.py:355,409-420`, two studies. The B20 suite's demuxer double parses the field by string surgery — conflating image codecs into it makes `LegacyAvcodecOnlyIdentity` fragile. |
| **Parser impact** | Zero (value only), same as A. |
| **Deploy impact** | Same as A. |
| **Future extensibility** | **Worse.** One field conflating two independent decode paths means `differences()` can no longer say "the *demuxer* changed" vs "the *JPEG* library changed" without re-parsing; the field name stops describing its contents. |
| **Interaction with N8** | Same as A (does not close it) but muddier — the union is now invisible in the schema. |

**Reject B.** It trades A's one clean new field for a rename with a wide
collateral radius and a less legible result, for no reproducibility gain.

### Option C — separate image and video decode identities (`decode_identity(kind)`)

`decode_identity("image")` over `{opencv, image_codec, system, machine}`;
`decode_identity("video")` over `{opencv, backend, ffmpeg, system, machine}`.

| Axis | Assessment |
|---|---|
| **Reproducibility** | Same coverage as A, **plus** it closes N8 (each path's `.d.` carries only its own libraries). |
| **Cache-key correctness** | **Breaks the single-literal architecture.** `workers.engine_registry.engine_versions()` (`:86-87`) returns `{"image": …, "video": …}` and `verify_version_alignment(configured)` (`:92-115`) raises unless **both** equal the one configured `Settings.model_version`. If the two `.d.` differ, the two `model_version` strings differ, and there is no single literal for `backend/config.py` to hold or for `service.py:98,227` to key the cache and idempotency on. |
| **Migration cost** | **High.** Two configured literals; per-kind `verify_version_alignment`; per-kind cache key and idempotency key; `/health` reports two; the OR1 worker gate and API format check both fork. This is precisely the *"per-kind configured version"* route **B23 evaluated and rejected** (`docs/VIDEO_SELECTION_IDENTITY.md` §3: "two literals … `verify_version_alignment` requires one string for both engines; touches `/health`, the idempotency key, every fixture"). |
| **Parser impact** | `decode_identity` gains a required argument → every call site changes (2 pipelines, `engine_registry`, ~6 studies/tests). |
| **Deploy impact** | Larger — two literals to keep aligned per deploy. |
| **Future extensibility** | Cleanest *end state* for N8, but only reachable through an architecture change. |
| **Interaction with N8** | Closes N8 — at the cost of the constraint "no architecture changes". |

**Reject C for B24.** It is an architecture change (the single-literal identity
is load-bearing across `engine_registry`, `config`, `service`, `/health`, OR1),
which the standing constraints forbid and which B23 already weighed and declined.
Record it as the *only* path that closes N8, gated on a future decision to split
the identity architecture (same tier as R13/SCRFD: blocked by constraint, not by
merit).

### Option D — composite `.d.`, B23-`.f.` style *(discovered during re-derivation)*

Keep `decode_identity()` argument-free and single-valued, but structure it as
`fingerprint({"video": <video decode fp>, "image": <image decode fp>})` — the
exact pattern B23 used to fold the video selection policy into `.f.` without a
new marker (`inference/preprocess.py:785` `selection_identity`).

| Axis | Assessment |
|---|---|
| **Reproducibility** | Identical to A — the composite moves whenever either half moves. |
| **Cache-key correctness** | One value, one literal — single-literal architecture preserved. One generation. |
| **Migration cost** | Slightly above A: two sub-fingerprint helpers + the composite, vs A's one field + one key. |
| **Parser impact** | Zero (still a marker-free 8-hex). |
| **Deploy impact** | Same as A. |
| **Future extensibility** | **Best of the in-constraint options.** `DecodeEnvironment.differences()` can report "image codec half moved" vs "video FFmpeg half moved" precisely, and a future Option-C split has a clean seam (the two halves already exist). |
| **Interaction with N8** | Same as A — does **not** close N8 (the image path's `.d.` still moves on a video-lib change because the composite includes both), but it makes the over-fragmentation *legible*, which A does not. |

D is A with the preimage organised as a hash-of-two-hashes. It costs ~10 lines
more and buys legibility + a future seam.

### 2.5 Recommendation — **Option A**, with the D seam noted

B20's stated principle was *"one string field, one place in the digest — the
minimal expansion"* and it has aged well. **Option A** is the faithful image-path
analogue: one `image_codec_version` field, one `"image_codec"` key in `identity`,
one new private parser, **zero** change to `_build_facts` or its callers, zero
parser-module change, one cache generation.

Option D's legibility gain is real but not load-bearing for B24 — `differences()`
granularity matters when there is an operator triaging a mismatch, and there is
no deployment yet. **Record D as the recommended shape *if and when* Option C
(the architecture split) is ever scheduled** — implement D first as the
refactor-safe intermediate. For B24: A.

### 2.6 Two value-level decisions the design makes (flagged for the gate — §8)

- **libjpeg-turbo SIMD flag → IN the identity.** `SIMD Support: YES|NO` is a
  compile-time property of the bundled codec that B8-E6 never measured (§1.8).
  libjpeg-turbo's SIMD path and its C path are *intended* to match for the
  default `JDCT_ISLOW` IDCT but have historically differed by ±1 LSB on some
  upsampling paths, and "SIMD vs no-SIMD wheel" is a real cross-build axis
  (ARM/x86, NASM-present/absent). The decode-identity philosophy is explicit
  (`decode_env.py:206-209`): *for a cache key a false miss costs compute and a
  false hit costs a wrong verdict — the tie breaks toward missing.* Include it.
  Cost: a fleet mixing SIMD and non-SIMD libjpeg-turbo wheels gets a false miss
  (rare; homogeneous fleets are the norm).
- **zlib → IN, flagged weakest.** PNG decode is `inflate`, which RFC 1951 makes
  bit-exact across zlib versions for any valid stream — so zlib is the *least*
  likely of the four to move a pixel. libpng (filtering, gamma, bit-depth
  scaling, interlace) is the real PNG risk. Include zlib for completeness and
  tie-toward-missing, but the design states it is defence, not a measured need.

Both are **specification-only** decisions — no architecture rides on either.

---

## 3. N8 — re-derived disposition

**N8 as recorded** (`REPRODUCIBILITY_AUDIT.md:127`): *"Video-shaped
`decode_identity` applied to the image path — the image path reports a
`decode_identity` string whose FFmpeg-library component is meaningless for it."*

**Re-derivation:**

- `decode_identity()` is one global function; both pipelines call it (§1.2).
- For the image path today, `ffmpeg_version` (avcodec/avformat/avutil/swscale)
  and `video_backend` (FFMPEG/MSMF) are in the `.d.` preimage, and `imdecode`
  touches neither. Result: the **image cache namespace regenerates on an FFmpeg
  upgrade** even though no image pixel changed — a false miss (compute, not
  correctness; `docs/RUN_IDENTITY.md` §3 rules this the safe direction).
- **Under Option A** the situation becomes *symmetric*: the `.d.` also gains the
  image codec libraries, so the **video** namespace now regenerates on a
  libjpeg-turbo upgrade even though no video pixel changed. N8's specific
  complaint (a path carrying an irrelevant decode component) is **not removed** —
  it is doubled and made symmetric.
- The **only** way to give the image path a `.d.` containing *only* what
  `imdecode` + OpenCV touch is Option C — a per-kind identity — which is an
  architecture change (§2, Option C) that the standing constraints forbid and
  that B23 already weighed and rejected.

**Disposition: N8 is NARROWED, not closed, not split, not promoted.**

- **Not closed** — Option A leaves each media path carrying the other's decode
  libraries. Claiming closure would be false.
- **Not split** — "split" implies an actionable decomposition into
  independently-closable parts. There is none: the residue is a single fact
  (the decode identity is a *union* over both decode paths) with a single fix
  (Option C) that is out of scope.
- **Not promoted** — the impact is a bounded **false-miss compute cost**, never
  a wrong verdict. Severity stays **Low**. Promoting it would misrepresent it.
- **Narrowed** — restate N8 as: *"The single `decode_identity()` is a union over
  the image and video decode environments; each media path's `.d.` therefore
  over-fragments by the other path's libraries. This is a bounded false-miss
  cost that the single-`STUB_ENGINE_VERSION`-literal architecture requires;
  removing it needs a per-media-kind identity (the rejected B23 'two literals'
  route). Symmetric after B24."* Keep it as a tracked Low finding whose closure
  is gated on the identity-architecture decision, not on any B24-sc*able work.

B24's design **must** state this explicitly in the `REPRODUCIBILITY_AUDIT.md`
N8 row and in `DECODE_IDENTITY_COMPLETENESS.md` — the prior `B24_CANDIDATE_
ANALYSIS.md` line "N8 → narrowed or closed (design sub-decision)" is resolved
here to **narrowed, with the restatement above**.

---

## 4. Movement analysis

### 4.1 Production files that change (2)

| File | Change | Why |
|---|---|---|
| `inference/decode_env.py` | `+_IMAGE_CODEC_LIBS = ("JPEG", "PNG", "WEBP", "ZLib")`; `+_MEDIA_IO_RE` (+ a small SIMD regex); `+_image_codec_facts() -> str \| None` (**new** function — `_build_facts` untouched); `DecodeEnvironment` `+image_codec_version: str \| None` field; `decode_environment()` sets it; `identity` fingerprint dict `+"image_codec": self.image_codec_version`; `to_dict` `+"image_codec_version"`; `differences()` field tuple `+"image_codec_version"`; `describe()` `+` one clause; class + `identity` docstrings updated. | Close R30 (§2, Option A). |
| `backend/config.py` | `STUB_ENGINE_VERSION` `.d.` literal `a97ccb68` → `<new>` (computed on the implementation box); `+` a B24 paragraph in the docstring, B20/B23 pattern. | The single configured literal must equal what the engines report (`verify_version_alignment`). |

**Not changed (verified):** `contracts/version.py` (parser is value-agnostic —
§1.2), `inference/image_pipeline.py` / `inference/video_pipeline.py` (call
`decode_identity()` unchanged), `workers/engine_registry.py`,
`workers/runner.py`, `backend/app.py`, `backend/service.py`,
`evaluation/benchmark.py`, `evaluation/calibration_protocol.py`,
`inference/preprocess.py`, `contracts/api.py`, `contracts/video.py`,
`contracts/enums.py`, `evaluation/split_lock.py`.

### 4.2 Test files that change

| File | Change | Kind |
|---|---|---|
| `tests/test_image_decode_identity.py` | **NEW** — the S8-B24 suite (§6). | new |
| `tests/run_all.py` | register `test_image_decode_identity` (`import` + tuple row); suite count 51 → 52. | new |
| `tests/test_decode_identity_completeness.py` (B20 suite) | `:341` `.endswith(".d.a97ccb68")` → `.d.<new>`; `:364` `"a97ccb68" not in snap` → `"<new>" not in snap` too. B20's subject (video FFmpeg libs) is otherwise untouched — add `test_batch_24_did_not_touch_the_video_ffmpeg_libs` here **or** in the new suite. | collateral |
| `tests/test_run_identity.py` | `:427` forensic field list — `+"image_codec_version"`. | collateral |
| `tests/test_crop_identity.py` | `:63` `B23_STUB_ENGINE_VERSION` → new `.d.`; `:291` `.endswith(".d.a97ccb68")` → new; `:299-300` runtime-version history list — `+` a B24 row `…f.e629642c.d.<new>`. `:54` `B22_STUB_ENGINE_VERSION` is **history — leave**. | collateral |
| `tests/test_video_selection_identity.py` | `:77` `B23_STUB_ENGINE_VERSION` → new `.d.`; `:341` `.endswith(".d.a97ccb68")` → new; `:413` `decode="a97ccb68"` → `decode=decode_identity()` (it is incidental to a `.f.`-mismatch test; make it live). | collateral |
| `tests/test_startup_alignment.py` | `:76,98,125` stale fixtures `…f.00000000.d.a97ccb68` and `:211` `…f..d.a97ccb68` — the `.f.` is the deliberately-wrong part; update `.d.` to `<new>` for consistency (optional but recommended — a reader should not see a stale `.d.` next to a stale `.f.`). | collateral (cosmetic) |
| `tests/test_decode_determinism.py` | `test_a_decode_environment_round_trips_through_its_dict` (`:524`) exercises the new field via `to_dict`↔constructor symmetry — **no edit needed**, but it is a non-movement check to run. Optionally `+` a positive `test_the_image_codec_string_is_captured`. | verify / optional |

**Verified no change:** `tests/test_evaluation_identity.py` and
`tests/test_run_identity.py` decode constants (`FFMPEG_DECODE="c20a064a"`,
`UNDER_FFMPEG`/`UNDER_MSMF`) are **synthetic doubles**, not the live literal.
`tests/test_frame_count_reproducibility.py` — all `_build_facts()` /
`ffmpeg_version` references are the FFmpeg half; `_build_facts` is unchanged;
`:316` `id_true == id_off` (decode_identity must not move on a frame_count
change) still holds. `tests/test_crop_geometry.py:585` asserts `".d." in
STUB_ENGINE_VERSION` (presence, no literal). `tests/test_inference_determinism.py`
decode_identity references are stability-within-a-process checks.

### 4.3 Documents that change

| Doc | Change |
|---|---|
| `docs/B24_DESIGN.md` | this file |
| `docs/DECODE_IDENTITY_COMPLETENESS.md` | `+` an "Image decode path (Batch 24, R30)" section, parallel to the FFmpeg one: what `imdecode` links, what `getBuildInformation` reports, the codec set = `ALLOWED_IMAGE_MIME` + zlib, the SIMD decision, the N8 narrowing. |
| `REPRODUCIBILITY_AUDIT.md` | `+` a `# Batch 24 resolutions — R30 closed` section; R30 summary row → **Closed in B24**; N8 row → the narrowed restatement (§3); D8 row → "video swscale half open; image libjpeg-turbo SIMD half **in the identity** since B24"; D9 row → "both decode paths now capture their libraries directly; residual is OpenCV imgproc between wheel patches only"; `Still open after Batch 24` list; B24 → move off the B24-candidates list. |
| `docs/B24_CANDIDATE_ANALYSIS.md` | `+` a one-line "superseded by `B24_DESIGN.md`; N8 resolved to *narrowed*; OR1-b + N5 riders re-evaluated in §5 there" pointer. |
| `docs/RUN_IDENTITY.md` | if it enumerates the `.d.` preimage keys, `+"image_codec"`. (Verify — it may only describe the marker, not the membership.) |

### 4.4 Fingerprints / versions — moves vs does not move

| Quantity | Verdict | Reason |
|---|---|---|
| **`.d.` (decode identity)** | **MOVES** — `a97ccb68` → `<new>` (illustrative on this build: `0ecdca05`; the real value is computed on the implementation box, as every `.d.`/`.f.` literal has been) | `identity`'s fingerprint preimage gains the `"image_codec"` key. |
| **`.f.` (selection composite)** | **does not move** — stays `e629642c` | `.f.` = `selection_identity(face, video)`; B24 touches no selection policy. |
| **`weights_identity`** | **does not move** — stays `stub-0-wacdd9240` | no weights, `input_size`, or `-w` dict change. |
| **`model_identity_of(STUB_ENGINE_VERSION)`** | **does not move** — stays `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c` | `model_identity_of` strips the `.d.` segment (`contracts/version.py:406-413`); the decoder is **deliberately not part of the model identity** (`test_the_effective_model_version_does_not_cover_the_decoder`, `docs/RUN_IDENTITY.md`). |
| **`STUB_ENGINE_VERSION`** | **MOVES** — `…f.e629642c.d.a97ccb68` → `…f.e629642c.d.<new>` | it is `compose_runtime_version(model_identity, decode_identity())`; only the `.d.` tail changes. |
| **`CONTRACT_VERSION`** | **does not move** — `1.0.0` | `ModelInfo.version` is an unconstrained `str` (`contracts/common.py`); no wire field, validator, `pattern`, `min_length` governs a version string (swept in B23-R3 §4.1). |
| **`RUN_SCHEMA_VERSION`** | **does not move** — `1.2.0` | `BenchmarkRun.decode_identity` is a `str \| None` that stores whatever value it is passed; the *schema* (which fields exist) is unchanged. B20 precedent exact. |
| **`CALIBRATION_WITNESS_VERSION`** | **does not move** — `1.2.0` | The witness digest covers `decode_identity` **as a stored string value** (schema 1.2.0, `SCHEMA_VERSIONS_WITH_DECODE`), verified over the stored string, not a recompute. A value change invalidates no stored witness. B20 precedent exact (`test_an_old_calibration_witness_still_verifies_after_the_change`). |
| **`LOCK_VERSION`** | **does not move** — `1.1.0` | `grep 'model_version\|decode\|\.d\.' evaluation/split_lock.py` → nothing. The split lock carries no decode or model identity of any kind. |
| **OpenAPI structural signature** | **does not move** | The decode identity is a runtime-version *value*, never a schema field. `test_the_openapi_snapshot_does_not_carry_the_decode_segment` (`.d.` and the literal both absent from the snapshot). Verified: `grep` of `backend/openapi.snapshot.json` for `a97ccb68` / `.d.` / `decode_identity` → nothing. |
| **`schema_out/*`** | **does not move** | `grep` of `schema_out/` for the decode literal / `.d.` / `decode_identity` → nothing. `python -m contracts.schema` regen is decode-independent. |
| **`fixtures_out/*`** | **does not move** | `fixtures_out/*.json` carry `model.version = "1.0.0-dev"` (synthetic `contracts/fixtures.py:62`, no `.d.`). `grep` of `fixtures_out/` for the decode literal → nothing. |

### 4.5 One value that *is* build-dependent

`<new>` is not fixed by this design — it is `decode_identity()` evaluated on the
box that ships B24, exactly as `a97ccb68` (B20) and `c20a064a` (B9) were. The
design fixes the **preimage** (`{opencv, backend, ffmpeg, system, machine,
image_codec}` with `image_codec` = the §2.6 string); the implementation report
records the concrete `<new>` and every collateral pin is set to it in the same
commit. `test_the_decode_segment_moved_and_the_literal_was_updated` asserts
`STUB_ENGINE_VERSION.endswith(f".d.{decode_identity()}")` — self-checking against
drift.

---

## 5. Migration analysis (re-derived from source)

### 5.1 Cache generation count — **one**

`backend/service.py:98` keys the content cache on
`self.settings.model_version`; `:227` keys the idempotency key on
`f"{sha256}:{self.settings.model_version}"`. `settings.model_version` is the one
configured literal (`STUB_ENGINE_VERSION` or `$DFP_MODEL_VERSION`), which carries
the `.d.` tail. When `.d.` moves, that one string moves once → **one cache
generation**.

### 5.2 Cache namespace impact — **both**

There is **one** `STUB_ENGINE_VERSION` for image and video engines
(`workers/engine_registry.py:86-87` build both pipelines' `model_version` and
`verify_version_alignment` requires **both** to equal the configured literal).
So the single `.d.` move regenerates **both the image and the video content-cache
namespaces**, even though the finding is image-only. This is the
single-literal-architecture cost, identical to B20's `.d.` move (which
regenerated the *image* namespace though images use no FFmpeg) and B23's `.f.`
move (which regenerated the *video* namespace though `.f.`'s video half is
image-carried-not-consumed). Idempotency keys regenerate in lockstep.

### 5.3 Benchmark loading compatibility — **no work required**

`evaluation/benchmark.py`: `BenchmarkRun.decode_identity: str | None = None`
(`:222`), `schema_version: str = RUN_SCHEMA_VERSION` (`:180`). A row written
before B24 keeps its `decode_identity` string (`"a97ccb68"` or `None` or a
synthetic); `from_dict` reads it unchanged; `RUN_SCHEMA_VERSION` is unbumped so
the `schema_version` gate (`:323`) does not trip. A cross-B24 comparison
(`regression_gate` → `_decode_comparability`, `:728-776`) sees two different
non-null decode identities and **refuses** — which is correct: the two runs were
scored under environments the identity scheme now distinguishes. `None` vs `None`
stays a *note*, never a pass (`:395-401` B20 test, unchanged). **B20 precedent
exact.**

### 5.4 Witness compatibility — **no work required, no version bump**

`evaluation/calibration_protocol.py`: `CALIBRATION_WITNESS_VERSION = "1.2.0"`
already carries `decode_identity` *inside* the digest (`:92`,
`SCHEMA_VERSIONS_WITH_DECODE = {"1.2.0"}`). `CalibrationWitness.__post_init__`
recomputes the digest **from the stored fields** (`:181-197`) — including the
stored `decode_identity` string — and compares to the stored `witness`. Changing
what `decode_identity()` *returns* does not touch any stored witness's stored
string, so every stored witness still verifies. A witness fitted after B24
records `<new>` and its digest covers `<new>`. `verify_calibration_witness(w,
decode_identity=X)` refuses when `X` ≠ the witness's recorded identity
(`test_evaluation_identity.py:551-564`) — correct. **No
`CALIBRATION_WITNESS_VERSION` bump** (B20 precedent:
`test_an_old_calibration_witness_still_verifies_after_the_change`).

R31 (the witness digest omits `effective_model_version` / gate / `.f.`) is
**untouched and irrelevant here** — B24 changes a field the digest *does* cover.

### 5.5 Parser compatibility — **total, no change**

`parse_runtime_version` / `compose_runtime_version` / `model_identity_of` /
`decode_identity_of` / `is_environment_pinned` operate by `str.partition(".d.")`
and never inspect the fingerprint value (§1.2). `<new>` is a marker-free
`[0-9a-f]{8}` exactly as `a97ccb68` was. A pre-B9 version with no `.d.` still
parses to `decode=None`. **Zero parser-module change; `_MODEL_MARKERS`
untouched.**

### 5.6 Rollback procedure

Symmetric to B20/B23:

1. Revert `inference/decode_env.py` (drop the field + parser + `"image_codec"`
   key) and `backend/config.py` (`.d.` literal back to `a97ccb68`).
2. Reverted engines recompute `decode_identity()` → `a97ccb68`;
   `engine_versions()` reports `…d.a97ccb68`; the worker gate and
   `verify_version_alignment` pass against the reverted literal. Clean.
3. **Orphaned cache entries:** bundles written under `…d.<new>` during the B24
   window become unreachable; they expire under `result_ttl_s` (default 7 days).
   Storage cost, not correctness.
4. **Idempotency:** a job whose pre-rollback attempt was keyed `{sha}:{…<new>}`
   is not deduplicated against a post-rollback resubmission keyed
   `{sha}:{…a97ccb68}` — one re-run, one-time.
5. **Partial rollback** (code reverted, literal not, or vice versa): the worker
   gate **refuses to start** (`verify_engine_alignment_at_startup` — engines
   report X, configured literal Y). This is the gate working as designed. Fix by
   making code and literal consistent.
6. **Benchmark/witness rows** written under `<new>` keep their string; a
   post-rollback comparison against an `a97ccb68` row refuses (correct).
7. Re-applying B24 restores `…d.<new>`; TTL-orphaned entries become reachable
   again if still within TTL. No corruption either direction.

### 5.7 Rolling-deploy implications (R34 applies)

B24 moves `STUB_ENGINE_VERSION`, so a pod-by-pod rolling update hits R34: a
worker whose configured literal disagrees with its just-built engines **refuses
to start** (fail-closed OR1). Two safe paths, from the B23 runbook:

- a coordinated worker restart at cutover; or
- `DFP_SKIP_VERSION_ALIGNMENT=true` held on every worker for the rollout window
  (reduces OR1 to B20/B22/B23 semantics — brief false cache **misses**, never
  wrong verdicts — and the every-boot `WARNING` is the audit trace).

Set `$DFP_MODEL_VERSION` to the new `STUB_ENGINE_VERSION` in the worker and API
environment before rollout. This is already the standing deploy-runbook line
(`docs/VIDEO_SELECTION_IDENTITY.md` §8 "Rollout (R34)").

### 5.8 Interaction with OR1-b

**Technically independent** — disjoint files (`inference/decode_env.py` +
`backend/config.py` vs a new `backend/worker_main.py` + a re-added
`Settings.skip_version_alignment` + the `run_with_shutdown` call site), disjoint
tests. B24 does not need OR1-b and OR1-b does not need B24.

**Thematically adjacent** — B24 forces a `STUB_ENGINE_VERSION` move; OR1-b makes
the *full* cryptographic `verify_version_alignment` run on the config-default
path (today only the `$DFP_MODEL_VERSION` path gets it; the default path gets
engine-agreement + format only). Without OR1-b, a worker on the config default
with a stale-but-well-formed `.d.` literal after B24 starts anyway and produces
silent false misses until noticed. With OR1-b it fails loudly.

**Recommendation:** keep B24's *core* scope to R30 (one finding, one batch). Ship
OR1-b as an explicitly delimited **§B24.1 rider** the implementer may include —
it needs no identity movement and is pure additive wiring, and the deploy that
follows B24 is the first to benefit. If the team enforces strict
one-finding-per-batch, OR1-b is B25 and B24 ships R30 alone. **N5** (lockfile) is
the same call — a build-hygiene rider, no identity movement, thematically "pin
the decode/compute environment"; include or defer to B25.

---

## 6. Test plan

All four matrices mirror the B20 suite (`tests/test_decode_identity_completeness.py`),
which is the proven template. New suite: `tests/test_image_decode_identity.py`
(`S8-B24 image decode identity completeness`). Named doubles:
`LegacyOpencvProxyIdentity` (reproduces the pre-B24 identity — the current five
keys, no `image_codec`) and `mismatched_image_codec(name, old, new)` (two
`DecodeEnvironment`s differing only in one codec's version string — since the
real libraries cannot be swapped on one box).

### 6.1 Fail-before matrix (real failures expected against the pre-B24 tree)

| # | Test | Pre-B24 behaviour |
|---|---|---|
| FB1 | `test_the_decode_identity_omits_the_image_codecs` (tripwire, B19→B20 style — kept and inverted after) | **passes** — the `identity` preimage (`inspect.getsource(DecodeEnvironment.identity.fget)`) contains no `image_codec`; `_image_codec_facts` does not exist |
| FB2 | `test_a_libjpeg_turbo_difference_moves_the_decode_identity` | **fails** — `mismatched_image_codec("JPEG", "2.1.3-62", "3.0.1")` → `env.identity == other.identity` (the codec is not in the preimage) |
| FB3 | same for `PNG` (`1.6.37` → `1.6.40`), `WEBP`, `ZLib` | **fails** identically |
| FB4 | `test_the_libjpeg_turbo_simd_flag_moves_the_identity` | **fails** — `SIMD Support: NO` vs `YES` produces one identity |
| FB5 | `test_a_libjpeg_difference_separates_all_four_protections` (headline) | **fails** — content-cache key, deployment identity, benchmark identity, calibration identity are all blind to a libjpeg-turbo change |
| FB6 | `test_the_benchmark_gate_refuses_across_an_image_codec_difference` | **fails** — `regression_gate` compares two runs whose only difference is the JPEG library and **passes** them |
| FB7 | `DecodeEnvironment(...)` with an `image_codec_version` kwarg | **fails** — `TypeError: unexpected keyword argument` (field does not exist) |
| FB8 | `test_image_codec_version_is_recoverable_for_forensics` (`"image_codec_version" in env.to_dict()`) | **fails** — key absent |

### 6.2 Pass-after matrix (real protections after B24)

| # | Test | Assertion |
|---|---|---|
| PA1 | `test_decode_identity_now_covers_the_image_codecs` | `_image_codec_facts()` names each of libjpeg-turbo / libpng / libwebp / zlib; `"image_codec"` is a key of the `identity` preimage; `env.image_codec_version` is that string |
| PA2 | `test_the_image_codec_libs_list_is_exactly_the_codecs_behind_ALLOWED_IMAGE_MIME` | `_IMAGE_CODEC_LIBS` ↔ `{JPEG, PNG, WEBP}` + `ZLib`; **no** TIFF / JPEG 2000 / OpenEXR (not in `contracts/api.py:33`) |
| PA3 | `test_a_libjpeg_turbo_difference_now_moves_the_identity` / same PNG / WebP / zlib | `mismatched_image_codec(...)` → `other.identity != env.identity`; and `LegacyOpencvProxyIdentity.identity_of(env) == …(other)` (the pre-B24 identity was blind — proves the fixture is real) |
| PA4 | `test_the_libjpeg_turbo_simd_flag_is_in_the_identity` | flipping the parsed `jpeg-simd` token → different `.identity` (§2.6 decision, pinned) |
| PA5 | `test_an_image_codec_difference_now_separates_all_four_protections` (headline) | `_four_protections_separate(env.identity, other.identity)` all true; the `LegacyOpencvProxyIdentity` computation all false |
| PA6 | `test_the_benchmark_gate_refuses_across_an_image_codec_difference` | `regression_gate(_bench(a), _bench(b))` not passed; a failure mentions "decode" |
| PA7 | `test_the_calibration_witness_binds_the_finer_identity` | `_witness(decode_identity=a).witness != _witness(decode_identity=b).witness` |
| PA8 | `test_the_decode_segment_moved_and_the_literal_was_updated` | `STUB_ENGINE_VERSION.endswith(f".d.{decode_identity()}")`; `== ".d.<new>"`; `verify_version_alignment(STUB_ENGINE_VERSION)` does not raise |
| PA9 | `test_the_image_codec_string_format_survives_a_webp_abi_version` | the parser captures `(ver encoder: 0x020f)` **and** `(ver decoder: 0x0209, encoder: 0x020f)` verbatim (regression on §1.5) |
| PA10 | `test_the_parser_is_defensive` | a `getBuildInformation()` with the Media I/O block absent → `image_codec_version is None`, no raise; `decode_identity()` still returns an 8-hex |

### 6.3 Non-movement matrix (proving nothing else moved)

| # | Test | Assertion |
|---|---|---|
| NM1 | `test_the_model_identity_did_not_move_in_batch_24` | `model_identity_of(STUB_ENGINE_VERSION) == "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c"` |
| NM2 | `test_no_contract_schema_or_witness_or_lock_version_bump` | `CONTRACT_VERSION=="1.0.0"`, `RUN_SCHEMA_VERSION=="1.2.0"`, `CALIBRATION_WITNESS_VERSION=="1.2.0"`, `LOCK_VERSION=="1.1.0"` |
| NM3 | `test_the_openapi_snapshot_does_not_carry_the_decode_segment` | `.d.` absent; `a97ccb68` absent; `<new>` absent; structural signature == committed |
| NM4 | `test_schema_out_regen_is_decode_independent` | `python -m contracts.schema --out <tmp>` → zero diff vs `schema_out/` |
| NM5 | `test_fixtures_out_regen_is_decode_independent` | `python -m contracts.fixtures --out <tmp>` → zero diff; `model.version` stays `"1.0.0-dev"` |
| NM6 | `test_batch_24_did_not_touch_the_video_ffmpeg_libs` | `_VIDEO_FFMPEG_LIBS == ("avcodec","avformat","avutil","swscale")`; `_build_facts()` signature and body unchanged (`inspect.getsource`); `env.ffmpeg_version` still the four-lib string |
| NM7 | `test_the_cpu_fields_are_still_excluded_from_the_identity` | `replace(env, cpu_baseline=…, cpu_dispatch=…)` → same `.identity`, different `.digest` |
| NM8 | `test_decode_identity_does_not_move_on_a_frame_count_change` | (in `test_frame_count_reproducibility.py:307-316`, unchanged) still green |
| NM9 | `test_the_effective_model_version_does_not_cover_the_decoder` | (`test_decode_determinism.py:484`) still green — `image_codec_version` not in the model identity |
| NM10 | `test_an_old_benchmark_row_still_loads_and_a_cross_boundary_comparison_refuses` | B20 pattern for `<new>` |
| NM11 | `test_an_old_calibration_witness_still_verifies_after_the_change` | stored `decode_identity="a97ccb68"` witness reloads and its digest matches |
| NM12 | `test_a_decode_environment_round_trips_through_its_dict` (`test_decode_determinism.py:524`) | still green with the new field (constructor ↔ `to_dict` symmetry) |
| NM13 | `test_full_suite` | `python tests/run_all.py` → 52 suites, `1264 + N` tests, 0 failures |

### 6.4 Collateral matrix (pin updates — mechanical, no logic change)

| File:line | Old | New |
|---|---|---|
| `backend/config.py:41` | `…f.e629642c.d.a97ccb68` | `…f.e629642c.d.<new>` |
| `tests/test_decode_identity_completeness.py:341` | `.endswith(".d.a97ccb68")` | `.endswith(".d.<new>")` |
| `tests/test_decode_identity_completeness.py:364` | `"a97ccb68" not in snap` | `+ "<new>" not in snap` |
| `tests/test_crop_identity.py:63` | `B23_STUB_ENGINE_VERSION = "…d.a97ccb68"` | `…d.<new>` |
| `tests/test_crop_identity.py:291` | `.endswith(".d.a97ccb68")` | `.endswith(".d.<new>")` |
| `tests/test_crop_identity.py:299-300` | history list ends B20-B23 | `+` `"…f.e629642c.d.<new>",  # B24` |
| `tests/test_video_selection_identity.py:77` | `B23_STUB_ENGINE_VERSION = "…d.a97ccb68"` | `…d.<new>` |
| `tests/test_video_selection_identity.py:341` | `.endswith(".d.a97ccb68")` | `.endswith(".d.<new>")` |
| `tests/test_video_selection_identity.py:413` | `decode="a97ccb68"` | `decode=decode_identity()` |
| `tests/test_startup_alignment.py:76,98,125,211` | `…d.a97ccb68` (incidental to a stale-`.f.` test) | `…d.<new>` (cosmetic consistency) |
| `tests/test_run_identity.py:427` | forensic field tuple | `+ "image_codec_version"` |
| `tests/run_all.py` | 51 suites | `+ import test_image_decode_identity`, `+` tuple row, 52 |

**History pins left alone (verified):** `test_crop_identity.py:54`
(`B22_STUB_ENGINE_VERSION`), `test_crop_identity.py:299` `…d.c20a064a` (B14-B19)
and `…d.a97ccb68` (B20-B23 — now also a history row),
`test_evaluation_identity.py` / `test_run_identity.py` synthetic decode
constants, `test_decode_identity_completeness.py` `LegacyAvcodecOnlyIdentity` /
`c20a064a` references (B20's own history).

### 6.5 Study

Extend `tests/decode_identity_completeness_study.py` (or new
`tests/image_decode_identity_study.py`): the before/after `identity` membership
table for the image path; the `getBuildInformation` Media I/O parse; the
"one identity, two libjpeg-turbo builds, two `p_fake`s under `StubBackend`"
demonstration (the stub hashes pixels, so a codec-string difference → a different
score → a stale cache hit is a wrong verdict — the R27 argument, for images).
Print-only, no assertions.

---

## 7. Re-ranked open findings (five axes, 1 = low, 5 = high)

Same axes as `B24_CANDIDATE_ANALYSIS.md`, re-scored after the §1–§5
re-derivation. **Actionable** = can be a batch now within the standing
constraints (no SCRFD, no ONNX, no architecture change, no
`CONTRACT`/`EvidenceBundle`/`ThresholdInfo`/OpenAPI-snapshot change).

| Finding | Repro | Cache-corr | Deploy | Migr | Cx | Actionable | Note (post-re-derivation) |
|---|---|---|---|---|---|---|---|
| **R13** face-crop instability (310×, 25.7%) | **5** | 3 | 2 | 5 | 5 | ❌ SCRFD | still the most severe; still un-batchable (the fix *is* SCRFD) |
| **R30** | **4** | **4** | 3 | **2** | 3 | ✅ | **B24.** Re-derivation confirmed: live gap, primary modality, parser-free, one-generation migration, B20 template. |
| **R31** witness digest omits model identity | 3 | 3 | 3 | 4 | 3 | ✅ (no live exposure) | B25. Needs `CALIBRATION_WITNESS_VERSION` 1.3.0 (every stored witness digest changes). No model to calibrate. Fold **N6/N7**. |
| **R24** whole-frame square resize (every video bundle) | 3 | 1 | 2 | 4 | 4 | ❌ video face stage | deterministic distortion — correctness, not reproducibility; no cache divergence |
| **R23** correctness halves | 2 | 1 | 2 | 4 | 4 | ❌ SCRFD | deterministic; every fix moves pixels, unvalidatable without a model |
| **R6-a/R6-b** *(not in the list but adjacent)* | 3 | 2 | 1 | 5 | 5 | ❌ video face stage | timestamp-fraction sampling + decodable-count cross-check = video-face-stage criteria |
| **R28** resize interp predicate area-filters an enlarged axis | 2 | 2 | 2 | 2 | 2 | ⚠️ rides a **pixel-moving** batch | trigger drifted (B23 moved `.f.` and skipped it — it moves *pixels*, not `.f.`); correct condition = SCRFD / R23 / R16 |
| **R18** capability half (faceless clip gets a decided verdict) | 2 | 1 | 1 | 4 | 4 | ⚠️ needs a `face_stage` signal | also blocks the R21 headline-rule tightening |
| **R16** one face's verdict published as the media's | 1 | 1 | 1 | 2 (prose) / 5 (per-face) | 2 / 5 | ⚠️ product decision | per-face verdicts touch `EvidenceBundle` (constraint); prose-only path is Low value |
| **R26** published bbox ≠ scored region | 1 | 1 | 1 | 3 | 3 | ⚠️ partly SCRFD | crop-window field touches `EvidenceBundle`; clamp divergence is SCRFD-armed |
| **R32** `batch_size` neutrality backend-dependent | 2 | 2 | 1 | 3 | 3 | ❌ re-opens with ONNX / attention | not a defect today; double-tripwired post-B23-cleanup |
| **R34** OR1 not rolling-deployable without coordination | 1 | 1 | 3 | 1 | 1 | tracked (mitigated) | **applies to B24** (which moves the literal); runbook line shipped |
| **D8** SIMD path not in the identity | 2 | 2 | 2 | 2 | 3 | ✅ **partly closed by B24** | B24 puts **libjpeg-turbo SIMD** in the identity (§2.6). The **libswscale** (video) SIMD half stays open, no measured occurrence. |
| **D9** `cv2.__version__` drops the wheel's 4th digit | 2 | 2 | 2 | 2 | 2 | ✅ **further mooted by B24** | after B24 *both* decode paths capture their libraries directly; residual = OpenCV imgproc between wheel patches only (B8-E6 low-risk) |
| **N5** no dependency lockfile | 3 | 2 | 3 | 2 | 2 | ✅ | `numpy` in no identity; build hygiene; **B24 rider or B25** |
| **N6 / N7** fit-optimiser config / `np.quantile` method unrecorded | 2 | 1 | 1 | 3 | 2 | ✅ pair with R31 | fold into the R31 witness-hardening batch |
| **N8** video-shaped decode identity on the image path | 1 | 1 | 1 | 2 (Option C) | 4 | ⚠️ **narrowed by B24** (§3) | after B24: symmetric union; closure needs the Option-C architecture split (rejected in B23) — tracked, Low, gated on an architecture decision |
| **D10** same version string, different build | 2 | 2 | 1 | 3 | 4 | ❌ un-seeable by a version string | a hash of linked binaries could; out of scope; record only |
| **N3** advisory heuristic constants | 1 | 1 | 1 | 1 | 1 | defer | add tripwires when one is next edited |

### 7.1 Is B24 still the correct next batch? — **Yes.**

The re-derivation did not dislodge R30. It **strengthened** the case in two ways
(D8's image half and D9 are folded in at zero extra cost) and **added one honest
caveat**: unlike R27, no image-codec divergence is *measured* on the test box
(`test_opencv_and_pillow_decode_jpeg_identically` passes bit-for-bit). R30's
severity therefore rests on the **structural** argument (three separate bundled
libraries, only `cv2.__version__` as proxy, and documented cross-wheel variance:
opencv-python 4.8.x ships libjpeg-turbo 2.1.3 while 4.9.x moved to 3.0.x) rather
than a local reproduction — the same evidentiary basis on which B20 stated D8/D10
as residuals, but here on the **primary modality** and about **whole libraries**
rather than a SIMD sub-path. Every higher-severity finding (R13, R23, R24, R6) is
blocked by SCRFD / the video face stage; R31 has no live exposure and a heavier
migration; OR1-b and N5 carry no reproducibility value. **R30 is the highest-value
actionable batch.**

---

## 8. Final gate

## **APPROVE WITH CHANGES**

The architecture and the movement/migration analysis are **locked and
independently verified**: Option A adds one field and one fingerprint key,
touches no parser, moves only `.d.` and `STUB_ENGINE_VERSION`, bumps no version
constant, is invisible to OpenAPI / `schema_out` / `fixtures_out`, and carries
the exact B20 one-cache-generation migration profile. N8 is resolved (narrowed,
with a written restatement). The test plan is the proven B20 template.

The changes required before implementation are **all specification-only** — no
architectural rework:

### Blocking defect 1 — ratify the libjpeg-turbo SIMD-flag inclusion *(specification-only)*

§2.6 puts `SIMD Support: YES|NO` **in** the cache-key identity on a
tie-toward-missing argument, and notes B8-E6 does **not** clear it. This
fragments the cache across a fleet mixing SIMD and non-SIMD libjpeg-turbo wheels.
**Correction:** the design must state the decision as *made* (IN) with the B8-E6
scope limit spelled out, and the B24 implementation report must record the
one-line study result (does flipping the parsed flag actually move a decoded
JPEG's bytes on the target build? — if the build has SIMD off, this is
un-measurable locally and the report says so, matching B20's D8/D10 honesty).
Reviewer must explicitly accept "SIMD flag in the cache key" or direct it to be
recorded as a residual like D8. *No code-structure impact either way — it is one
token in the parsed string.*

### Blocking defect 2 — ratify the zlib inclusion and the codec set *(specification-only)*

§2.6 includes zlib while noting PNG decode (`inflate`) is spec-deterministic
across zlib versions — so zlib is the weakest member and arguably noise in the
key. **Correction:** the design must either (a) drop zlib and state that libpng
alone carries the PNG risk, or (b) keep it with the explicit "defence, not a
measured need" caveat. The design as written chooses (b); a reviewer should
confirm. Same ratification for excluding TIFF / JPEG 2000 / OpenEXR — correct
*only while* `ALLOWED_IMAGE_MIME` stays `{jpeg, png, webp}`; the design must add
`test_the_image_codec_libs_list_is_exactly_the_codecs_behind_ALLOWED_IMAGE_MIME`
(PA2) so a future MIME-allowlist widening trips a red test.

### Blocking defect 3 — pin the parse strings against the deployment build *(specification-only)*

The `getBuildInformation()` "Media I/O" format was verified on **one** build
(Windows / opencv-python 4.8.1: `JPEG: build-libjpeg-turbo (ver 2.1.3-62)`,
`WEBP: (ver encoder: 0x020f)`). Linux `opencv-python-headless`, conda-forge, and
distro builds format these lines differently (system libjpeg-turbo shows a path;
WebP may show `decoder:` + `encoder:`; some builds omit the SIMD lines entirely).
**Correction:** the design must specify the parser as *"capture the entire
text after `<NAME>:` to end-of-line, verbatim; for `JPEG` also capture the two
`SIMD Support*` lines if present; join B20-style; `None` if the block or a line
is absent"* — i.e. **do not parse the version into components**, store the raw
line. The B24 implementation runs `_image_codec_facts()` on the actual target
image and pins the resulting `<new>` and the collateral literals in one commit.
This is exactly how B20 handled the FFmpeg lines (`_AVLIB_RE` captures
`([^)]+)` verbatim).

### Nature of the changes

All three are **specification-only**. None touches the dependency graph, the
identity architecture, the parser, the cache-key structure, the contract, or the
migration profile. With §8.1–§8.3 ratified in the design text (a reviewer
accepting the design's own recommendations converts this to **APPROVE**), B24 is
implementation-ready.

### If the reviewer rejects the "measured vs structural" basis (§7.1)

A reviewer may hold that R30, lacking a local reproduction (unlike R27's measured
frame-count divergence), should be **recorded as a residual** (D-series) and the
cache generation spent later, bundled with an image-codec change that *is*
observed. That is a coherent position. The design's counter, for the record: the
image path is the primary modality, the omission is *whole libraries* not a
sub-path, cross-wheel codec-version variance is a documented fact (not a
hypothesis), and the guarantee sought is R27's exact "detectable, not identical"
— none of which requires a local divergence to be worth one cache generation.
The gate stands at **APPROVE WITH CHANGES**, not REJECT.
