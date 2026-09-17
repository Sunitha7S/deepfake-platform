# Batch 24 — R2 Design (revised, post-adversarial-review)

**Supersedes:** `docs/B24_DESIGN.md`.
**Inputs:** `docs/B24_DESIGN.md`, `docs/B24_ADVERSARIAL_REVIEW.md` (the review is
treated as **hypotheses to verify**, not as findings).
**Type:** design only. No code, no implementation.
**Method:** every claim re-derived from the current repository or executed live
on the implementation box (Windows / `opencv-python==4.8.1.78` / `Pillow==12.2.0`).

**Headline outcome:** the adversarial review's four blocking findings (S1–S4)
are **all confirmed**, three of them *strengthened* by new measurements. The
corrected R30 identity design is specified in full (§3, §5, §6) and is
technically implementation-ready — **but R30 is not the right B24.** The correct
B24 is **N5 (a dependency lockfile)**, which the re-derivation shows is not
hygiene but *load-bearing for every identity literal in the platform*. R30's
identity work is demoted to a D-series residual, to be bundled with D8/D9/D10 in
a later batch, gated on N5.

---

## 1. Resolution audit — S1–S4 re-derived from source

### S1 — "R30's risk/priority is overstated" → **CONFIRMED, and stronger than the review stated**

**Review's claim:** for the declared `opencv-python` wheel deployment, R30's
residual over the status quo is D9-class (a 4th-digit wheel rebuild that changes
a bundled codec); R27-class only for system-linked OpenCV, which `pyproject.toml`
does not pin.

**Re-derivation:**

1. `pyproject.toml:19-33` declares `"opencv-python>=4.8,<5"` — the PyPI wheel,
   range-pinned. No `opencv-python-headless`, no system package.
2. `cv2.getBuildInformation()` "Media I/O" *(live)*: `ZLib: build (ver 1.2.13)`,
   `JPEG: build-libjpeg-turbo (ver 2.1.3-62)`, `WEBP: build (ver encoder:
   0x020f)`, `PNG: build (ver 1.6.37)` — **every codec is `build`**, i.e. bundled
   into the wheel and frozen to the wheel version. FFmpeg is `YES (prebuilt
   binaries)` — also bundled.
3. `cv2.__version__ == "4.8.1"` while the wheel is `4.8.1.78`
   (`importlib.metadata.version("opencv-python")`, live) — the 4th component is
   dropped. This is **D9** (`test_decode_identity_completeness.py:409-421`).
4. A bundled-codec change in opencv-python normally ships with a `cv2.__version__`
   minor/major bump (`opencv_version` — already a key in `identity` — catches it).
   The residual is exactly the case where a 4th-digit rebuild changes a codec:
   **D9's scope.**

**New evidence the review did not have** — direct cross-build decode comparison
*(live)*:

| Format | OpenCV stack | Independent stack (Pillow 12.2.0) | Result |
|---|---|---|---|
| **JPEG** (Q88 / Q40 / Q92-progressive, random noise) | libjpeg-turbo **2.1.3-62**, SIMD **NO** | libjpeg-turbo **3.1.4.1**, SIMD (Pillow wheels ship it) | **bit-identical** (`test_opencv_and_pillow_decode_jpeg_identically`, passes) |
| **PNG** (3 shapes, random noise) | libpng **1.6.37** + zlib **1.2.13** | different libpng + **zlib-ng 1.3.1** (a different DEFLATE implementation) | **bit-identical** (measured for this design) |
| **WebP** (2 shapes, Q100) | libwebp **0x020f** | Pillow's libwebp | **bit-identical**, maxdiff 0 (measured for this design) |

Two fully independent codec stacks — spanning a **major libjpeg-turbo version
gap (2.1.3 → 3.1.4)**, different libpng, different libwebp, and **stock zlib vs
zlib-ng** — decode the accepted formats bit-for-bit identically. R30's premise
("different codec library builds disagree on a pixel") is **not merely
unmeasured; it is measured false** for every accepted MIME type across the
largest version gaps available on this box.

**Verdict: S1 CONFIRMED.** R30 is not R27-class. It is not even D9-class in
*impact* (D9 presumes the codec change would matter; the evidence says it does
not). It is **D8/D10-class**: a structural completeness gap — "finer than a
version string reaches, no measured occurrence"
(`REPRODUCIBILITY_AUDIT.md:3104-3116`, the exact language B20 used for its own
residuals).

### S2 — "SIMD flag belongs in `digest`, not `identity`" → **CONFIRMED**

**Re-derivation:**

- `decode_env.py:211-221` — the `identity` membership rule: *"OUT, because
  including them would fragment the cache on things measured **not** to change a
  pixel"* — applied to `cpu_baseline` / `cpu_dispatch` (in `digest` only).
- `test_decode_determinism.py:1-34` docstring, point 3 (a **standing repo
  finding**): *"SIMD dispatch does not change `resize`/`cvtColor`/`imdecode`
  output"* — pinned so a future divergence "says so rather than the finding
  quietly becoming true again in the dark."
- libjpeg-turbo's C and SIMD IDCT implementations use the same `JDCT_ISLOW`
  (accurate integer) algorithm — the method OpenCV's JPEG loader selects — and
  the project's design goal and regression suite hold them bit-identical. Fast
  IDCT (`JDCT_IFAST`) can differ; OpenCV does not use it.
- The JPEG cross-build test above spans SIMD **NO** (OpenCV) vs SIMD **on**
  (Pillow) and agrees.

**Nuance the review slightly overstated:** `test_cpu_dispatch_does_not_change_
pixel_operations` (`test_decode_determinism.py:391`) toggles `OPENCV_CPU_DISABLE`,
which is *OpenCV's* dispatch, **not** libjpeg-turbo's internal SIMD (that needs
`JSIMD_FORCENONE`). So the *direct* SIMD-on/off test the docstring implies does
not exist — but the Pillow cross-check (SIMD-off OpenCV vs SIMD-on Pillow, three
JPEGs, identical) covers the same ground.

**Verdict: S2 CONFIRMED.** The SIMD flag is a `cpu_dispatch`-class fact:
performance for OpenCV's decode path, recorded in `digest`, pinned by a test,
promoted to `identity` only if a divergence is ever observed. The B24_DESIGN §2.6
decision to put it in the cache key is a specification defect.

### S3 — "zlib belongs OUT of `identity`" → **CONFIRMED**

**Re-derivation:**

- PNG decode: libpng calls zlib `inflate()` to decompress IDAT, then libpng does
  unfiltering / bit-depth / gamma / palette work. `inflate()` output is fully
  determined by the compressed stream (RFC 1951 §3; RFC 1950 for the zlib
  wrapper) — every conformant implementation produces the same decompressed
  bytes.
- **Measured for this design:** OpenCV (zlib 1.2.13) vs Pillow (**zlib-ng
  1.3.1**, an independent DEFLATE implementation) → PNG decode bit-identical
  across 3 shapes. If a stock-zlib vs zlib-ng difference produces identical
  pixels, a stock-zlib version bump certainly does.
- Including zlib in `identity` fragments the cache on a provably pixel-neutral
  axis — the exact thing `decode_env.py:211` says not to do.

**Verdict: S3 CONFIRMED.** zlib is **excluded from `identity`**. It may be
recorded in `digest` for diagnostics (it is cheap and an operator triaging a PNG
discrepancy would want it), matching the `backend_source` / `python_version`
treatment.

### S4 — "N8 is mislabelled as 'narrowed'" → **CONFIRMED**

**Re-derivation:**

- `workers/engine_registry.py:80-88` `engine_versions()` **unconditionally**
  constructs `ImagePipeline(model_name=IMAGE_MODEL)` **and**
  `VideoPipeline(model_name=VIDEO_MODEL)`. `verify_version_alignment` (`:91-119`)
  raises unless the one configured literal equals **both**. `backend/config.py`
  holds **one** `STUB_ENGINE_VERSION`; `backend/service.py:98,227` keys the
  content cache and idempotency key on the **one** `settings.model_version`.
- **There is no image-only or video-only deployment.** Every process holds both
  engines and one identity string covering both.
- N8 is therefore not "the image path carries a video component"; it is *"the one
  `decode_identity()` is a union over two library-disjoint decode paths, so any
  library change in either path regenerates the whole (image + video) content
  cache."*
- Any R30 identity expansion (Option A of B24_DESIGN) **enlarges** this union
  symmetrically — a libjpeg-turbo bump would then also regenerate the video
  cache. The B24_DESIGN §3 text admits this ("doubled and made symmetric"), which
  is not a "narrowing".
- This is the **single-literal architectural trade**, already documented in
  `docs/RUN_IDENTITY.md` §3 ("a cache key breaks ties toward missing"), already
  paid by **B20** (its `.d.` FFmpeg move regenerated the image namespace) and
  **B23** (its `.f.` video-selection move regenerated the image namespace). Its
  only removal is per-kind identities — an architecture change (B24_DESIGN
  Option C, correctly rejected; the "two literals" route B23 weighed and
  declined in `VIDEO_SELECTION_IDENTITY.md` §3).

**Verdict: S4 CONFIRMED.** N8 disposition corrected in §4.

### Summary

| Finding | Verdict | Change vs review |
|---|---|---|
| S1 (risk overstated) | **CONFIRMED** | strengthened — R30 is D8/D10-class, not D9-class; cross-build decode is *measured identical*, not merely unmeasured |
| S2 (SIMD → digest) | **CONFIRMED** | one nuance corrected (the direct SIMD test does not exist; the Pillow cross-check covers it) |
| S3 (zlib → out) | **CONFIRMED** | strengthened — zlib-ng cross-check measured identical |
| S4 (N8 mislabelled) | **CONFIRMED** | — |

---

## 2. Re-rank R30

### 2.1 Class

**R30 is D8/D10-class** — a structural gap in the identity's coverage with **no
demonstrated pixel impact** for the platform's accepted inputs, on the
deployment the repository declares.

- **Not R27-class.** R27 had a measured, large-magnitude divergence (a one-frame
  demuxer miscount moved 11/16 sampled indices — `test_decode_identity_
  completeness.py:260-277`) with a clear mechanism (uniform sampling on
  `CAP_PROP_FRAME_COUNT`). R30 has none of that.
- **Not D9-class in impact.** D9 (`cv2.__version__` drops the 4th digit) is a
  real blind spot, but its harm requires the hidden change to matter. §1's
  cross-build measurements say a codec change — even a *major* libjpeg-turbo
  version gap — does not change the decoded pixels for JPEG/PNG/WebP 8-bit RGB.

### 2.2 Per deployment shape

| Deployment | R30 exposure | Reason |
|---|---|---|
| **`opencv-python` wheel** (declared) | ~none | all codecs bundled + frozen to `cv2.__version__`; a codec change ships with a version bump `opencv_version` catches; cross-build decode measured identical anyway |
| **`opencv-python-headless` wheel** | ~none | same 3rd-party libs as `opencv-python` for a given version |
| **system-linked OpenCV** (conda-forge, distro apt/dnf, homebrew) — **not pinned** | low, structural | `cv2.__version__` / FFmpeg unchanged on an OS `libjpeg-turbo` update; pixel impact still unmeasured and, per §1, unlikely for accepted formats |
| **wheel rebuild, same 3-digit version, new 4th digit** | D9-narrow | requires the rebuild to bump a bundled codec; opencv-python patch rebuilds rarely do |
| **fourth-digit rebuild that *does* bump a codec** | the one real gap | still gated by "does the codec version actually change a pixel" — measured no in §1 |
| **containerized** (Docker `pip install` at build time) | inherits the range-drift of whatever it resolves | **fixed by N5, not R30** — two images built months apart resolve different `opencv-python`; without a lock the `.d.` literal in `backend/config.py` may not match the image's engines and `test_the_configured_literal_matches_the_engines` fails |
| **CI environments** | **broken today without a lock** | there is no CI config and no lockfile (`ls .github/`, `find -name '*.lock'` → nothing); a CI box that resolves a newer in-range `opencv-python` fails `verify_version_alignment` against the hardcoded `a97ccb68` (which was computed on a `4.8.1` box — `test_decode_identity_completeness.py:194` even guards on `opencv_version == "4.8.1"`) |

### 2.3 Can R30 justify a dedicated batch? — **No.**

- Its live reproducibility exposure for the declared deployment is **effectively
  zero** (§1, measured).
- Closing it costs a `STUB_ENGINE_VERSION` move → a cache generation → the R34
  coordinated-deploy dance — a real, fixed cost for a gain that is structural
  only.
- The identity expansion *worsens* N8 (§1/S4) and *increases* the number of
  build-specific inputs to the `.d.` literal, which makes the **N5 gap worse**
  (more ways for a fresh `pip install` to produce a literal that doesn't match).

**R30 should be bundled with D8/D9/D10** in a future "decode-environment
completeness (pass 2)" batch — one cache generation for the whole cluster —
**and that batch should land after N5**, because a lockfile is what makes any
`.d.` literal reliably reproducible in the first place. The corrected identity
design for that future batch is §3, §5, §6 of this document.

---

## 3. Identity membership — re-evaluated, with the final matrix

For **each** candidate, the three-way question: does it *affect pixels*,
*affect performance*, or *affect diagnostics only*?

### libjpeg-turbo version

- **Affects pixels?** In principle, for pathological inputs (arithmetic-coded
  JPEG, 12-bit, CMYK) — none of which `ALLOWED_IMAGE_MIME` accepts. For 8-bit
  baseline/progressive RGB JPEG: **measured no** across 2.1.3 → 3.1.4 (§1).
- **Affects performance?** Yes (SIMD, buffer strategies) — but that is the SIMD
  flag, separate.
- **Affects diagnostics?** Yes — an operator triaging a JPEG discrepancy wants it.
- **Include in `identity`?** **Yes, weakly** — it is the coarsest reliable
  signal, and "the JPEG decoder library changed" is the honest thing to key a
  *change-detector* on even when the change is usually pixel-neutral. The
  identity is deliberately over-sensitive (`decode_env.py:206-209`: false miss
  costs compute, false hit costs a wrong verdict). This is the one place that
  argument still applies after §1.

### libjpeg-turbo SIMD flag

- **Affects pixels?** **No** for `JDCT_ISLOW` (OpenCV's choice). libjpeg-turbo
  holds C and SIMD IDCT/upsampling bit-identical for this method. Cross-build
  evidence: SIMD-off OpenCV vs SIMD-on Pillow, three JPEGs, identical (§1).
- **Affects performance?** **Yes** — this is its entire purpose.
- **Affects diagnostics?** Yes.
- **Include in `identity`?** **No.** `digest` only — the exact `cpu_dispatch`
  treatment. Pinned by a test so a future divergence is loud.

### libpng version

- **Affects pixels?** In principle, for gamma / palette / 16-bit / interlace
  edge behaviour that changed across libpng releases — but OpenCV's PNG loader
  disables gamma and forces 8-bit, narrowing the surface. For 8-bit RGB PNG:
  **measured no** (OpenCV libpng 1.6.37 vs Pillow's libpng, 3 shapes — §1).
- **Affects performance?** Marginally.
- **Affects diagnostics?** Yes.
- **Include in `identity`?** **Yes, weakly** — same reasoning as libjpeg-turbo:
  coarsest reliable signal, over-sensitive by design. libpng (not zlib) carries
  whatever PNG pixel risk exists.

### zlib version

- **Affects pixels?** **No.** `inflate()` is RFC-1951-deterministic for any
  valid stream. Measured: stock zlib 1.2.13 vs zlib-ng 1.3.1 → identical PNG
  (§1).
- **Affects performance?** Yes (zlib-ng exists purely for speed).
- **Affects diagnostics?** Minor.
- **Include in `identity`?** **No.** Optionally `digest`. Including it violates
  the identity's own membership rule.

### libwebp version

- **Affects pixels?** Lossy WebP decode is spec-defined; lossless is exact.
  Measured: OpenCV libwebp vs Pillow's, 2 shapes, maxdiff 0 (§1).
- **Affects diagnostics?** Yes.
- **Include in `identity`?** **Yes, weakly** — consistency with libjpeg-turbo /
  libpng; it is a codec on the accepted path.

### 3.1 Final inclusion matrix

| Candidate | `identity` (cache key) | `digest` (forensic) | Class | Rationale |
|---|:---:|:---:|---|---|
| **libjpeg-turbo version** | **IN** | (via identity) | affects pixels *in principle*; measured neutral for accepted MIME | coarsest reliable signal; change-detector is deliberately over-sensitive |
| **libpng version** | **IN** | (via identity) | same | carries the PNG pixel risk (not zlib) |
| **libwebp version** | **IN** | (via identity) | same | codec on the accepted path; consistency |
| **libjpeg-turbo SIMD flag** | **OUT** | **IN** | affects performance only (`JDCT_ISLOW` bit-exact) | `cpu_dispatch` precedent; test-pinned |
| **zlib version** | **OUT** | optional | affects performance only (`inflate` RFC-deterministic) | violates the `identity` membership rule if included |
| TIFF / JPEG 2000 / OpenEXR | OUT | OUT | unreachable | not in `ALLOWED_IMAGE_MIME` — guarded by a test |

`_IMAGE_CODEC_LIBS = ("JPEG", "PNG", "WEBP")` — three version strings, captured
verbatim from the `getBuildInformation()` "Media I/O" block (B24_DESIGN §8.3
stands: raw-line capture, no component parsing). The SIMD line, when present,
goes to a separate `digest`-only field.

---

## 4. N8 — final disposition

**N8 becomes: CLOSED — accepted architectural cost.**

- **Not narrowed.** §1/S4: an R30 expansion enlarges the union; nothing narrows.
- **Not promoted.** It is a false-*miss* (compute cost), never a wrong verdict.
  Severity Low, and lowered further by §1 (a codec bump usually changes no
  pixel, so the "false miss" is often also a *correct* miss avoided — the cache
  regenerates unnecessarily but serves nothing wrong).
- **Not split.** No decomposition into independently-closable parts exists.
- **Not left "blocked".** "Blocked" (the R13 status) implies a known fix waiting
  on a constraint. N8's only fix (per-kind identities) is not merely
  constraint-blocked — it was *evaluated and rejected on its merits* by B23
  (`VIDEO_SELECTION_IDENTITY.md` §3: two literals, touches `/health`, the
  idempotency key, every fixture). A rejected design is not a pending fix.

**Rationale for CLOSED-accepted-cost:** N8 describes the deliberate,
documented consequence of the single-`STUB_ENGINE_VERSION`-literal architecture —
one identity string for two engines means a change in either decode path
regenerates both cache namespaces. This is stated in `docs/RUN_IDENTITY.md` §3,
was consciously paid by B20 and B23, and is the same category as **R9**
(quarantine — "by design, reported every run"). The `REPRODUCIBILITY_AUDIT.md`
N8 row should read:

> **N8 — CLOSED (accepted architectural cost).** The single `decode_identity()`
> is a union over the image and video decode environments (both engines always
> exist — `workers/engine_registry.py:80-88`), so a library change in either
> path regenerates both content-cache namespaces — a bounded false-miss, never a
> wrong verdict. This is the single-literal identity trade documented in
> `docs/RUN_IDENTITY.md` §3 and paid by B20 (`.d.`) and B23 (`.f.`). Its only
> alternative — per-media-kind identities — was evaluated and rejected in
> `docs/VIDEO_SELECTION_IDENTITY.md` §3. No further action; folding image codecs
> into the identity (a future decode-completeness batch) enlarges the union
> symmetrically and does not change this disposition.

---

## 5. Migration analysis — rebuilt from source (for the deferred R30 identity batch)

*(This is the migration profile for the future "decode-environment completeness
pass 2" batch, if R30's identity work is done. N5's migration is §7.3 — it is
trivial by comparison.)*

Every line re-derived; prior conclusions not reused.

### 5.1 Cache generations — **one**

`backend/service.py:98` keys the content cache on `settings.model_version`;
`:227` keys idempotency on `f"{sha256}:{settings.model_version}"`.
`settings.model_version` is the one configured literal carrying `.d.<D>`. Adding
`libjpeg-turbo` / `libpng` / `libwebp` keys to `DecodeEnvironment.identity`'s
`fingerprint()` dict changes `<D>` once → **one cache generation**.

### 5.2 Cache namespace impact — **both**

One literal, two engines (`engine_registry.py:86-87`), so the single `.d.` move
regenerates **image and video** content-cache namespaces and their idempotency
keys in lockstep. Identical to B20's `.d.` move and B23's `.f.` move. This is
N8's cost (§4), now symmetric.

### 5.3 Deploy coordination — **R34 applies**

A `STUB_ENGINE_VERSION` move → a pod-by-pod rolling update hits a worker whose
configured literal disagrees with its freshly-built engines → **fail-closed**
(OR1). Safe paths (from the B23 runbook, `VIDEO_SELECTION_IDENTITY.md` §8):
coordinated worker restart at cutover, or `DFP_SKIP_VERSION_ALIGNMENT=true` for
the window (reduces to B20/B22/B23 semantics — brief false misses, never wrong
verdicts). Set `$DFP_MODEL_VERSION` to the new literal in worker + API env.

**Second-order effect the review flagged (§5, D2):** two benchmark runs on
byte-identical pixels, one pre- and one post-batch, become **incomparable**
(`_decode_comparability`, `benchmark.py:728-776`, refuses two different non-null
decode identities). This is a *conservative false incomparability*, not a
correctness gain — the runs *are* comparable, the gate cannot know it. Accepted
cost, B9/B20 precedent; the audit doc must say "false incomparability", not
"correct".

### 5.4 Startup alignment — **no change to the mechanism**

`backend/app.py` lifespan format check (`is_composed` + `is_environment_pinned` +
`parse_model_version(mv).compose() == model_identity_of(mv)` round-trip): passes
for `…f.e629642c.d.<new>` — a clean hex suffix, still composed, still
environment-pinned. `workers/runner.py` `verify_engine_alignment_at_startup` →
`verify_version_alignment(configured)`: matches iff the collateral literal is
updated in the same commit; otherwise the worker fails closed (correct).
`_MODEL_MARKERS` / `parse_runtime_version` never inspect the fingerprint value —
**zero parser impact** (`contracts/version.py:365-403`, re-verified).

### 5.5 Rollback

Revert `inference/decode_env.py` (drop the 3 keys + the field) and
`backend/config.py` (`.d.` literal → `a97ccb68`). Reverted engines recompute
`a97ccb68`; the gate passes. Orphaned `…d.<new>` cache entries expire under
`result_ttl_s` (default 7 days). Partial rollback (code XOR literal) → worker
refuses to start (correct). Benchmark/witness rows written under `<new>` keep
their string; a post-rollback comparison against an `a97ccb68` row refuses
(correct). No corruption in either direction.

### 5.6 Benchmark compatibility — **no work**

`BenchmarkRun.decode_identity: str | None = None` (`benchmark.py:222`);
`schema_version: str = RUN_SCHEMA_VERSION` (`:180`), **unbumped**. A pre-batch
row keeps its `decode_identity`; `from_dict` reads it unchanged; the
`schema_version` gate (`:323`) does not trip. Cross-batch comparison refuses
(§5.3). `None` vs `None` stays a note, never a pass (`:395-401`). B20 precedent
exact.

### 5.7 Witness compatibility — **no `CALIBRATION_WITNESS_VERSION` bump**

`compute_witness` (`calibration_protocol.py:211-278`) hashes `decode_identity` as
a **raw string value** for schema `1.2.0` (`SCHEMA_VERSIONS_WITH_DECODE`);
`__post_init__` recomputes from the *stored* string and dispatches on the
payload's own `schema_version` (`:230-243`). Changing what `decode_identity()`
*returns* touches no stored witness's stored string → every stored witness still
verifies. A new witness records `<new>`. R31 (digest omits model id) is
orthogonal — the decode field *is* in the digest. B20 precedent exact
(`test_an_old_calibration_witness_still_verifies_after_the_change`).

### 5.8 No movement

| Quantity | Verdict | Reason |
|---|---|---|
| `.d.` | **moves** | `identity` fingerprint dict gains 3 keys |
| `STUB_ENGINE_VERSION` | **moves** (`.d.` tail only) | `compose_runtime_version(model_identity, decode_identity())` |
| `.f.` / `weights_identity` / `+c.` / `.g.` | do not move | no selection / weights / calibration / gate change |
| `model_identity_of(STUB_ENGINE_VERSION)` | does not move | `model_identity_of` strips `.d.` (`version.py:406-413`); decoder is deliberately not part of the model identity |
| `CONTRACT_VERSION` (1.0.0) | does not move | `ModelInfo.version` unconstrained `str`; no wire field |
| `RUN_SCHEMA_VERSION` (1.2.0) | does not move | `BenchmarkRun.decode_identity` stores a value; schema unchanged |
| `CALIBRATION_WITNESS_VERSION` (1.2.0) | does not move | digest over the stored string; §5.7 |
| `LOCK_VERSION` (1.1.0) | does not move | split lock carries no decode/model identity (`grep` → nothing) |
| OpenAPI signature / `schema_out` / `fixtures_out` | do not move | `grep -rln` for the decode literal / `.d.` / `decode_identity` → nothing in any of them; fixtures carry `model.version="1.0.0-dev"` |
| `DecodeEnvironment.digest` (`4aded179`) | moves (with `identity`) | expected — digest is strictly finer; **no literal pins it** (verified) |

---

## 6. Test plan — rebuilt (for the deferred R30 identity batch)

Template: the B20 suite (`tests/test_decode_identity_completeness.py`). New
suite: `tests/test_image_decode_identity.py` (`S8-B?? image decode identity`).
Named doubles: `LegacyOpencvProxyIdentity` (reproduces the pre-batch identity —
the current five keys) and `mismatched_image_codec(name, old, new)` (two
`DecodeEnvironment`s differing only in one codec version string).

### 6.1 Output-identical proof (the B22/B23-style protection the review demanded)

| ID | Test | Assertion |
|---|---|---|
| OI1 | `test_the_shipped_image_bundle_did_not_move` | a real JPEG + a real PNG through `ImagePipeline`: `p_fake`, `verdict`, every `faces[]` score **byte-identical** to a pinned constant; only `bundle.model.version` moved (its `.d.` tail). |
| OI2 | `test_the_shipped_video_bundle_did_not_move` | a real clip through `VideoPipeline`: `p_fake` / `verdict` byte-identical to the pinned `SHIPPED_P_FAKE`; only `.d.` moved. (Reuses the `test_video_selection_identity.py` fixture pattern.) |
| OI3 | `test_the_decoded_pixels_are_byte_identical_across_the_batch` | `hashlib.sha256(decode_image(JPEG).tobytes())` == pinned; same for PNG, WebP. The batch adds a fingerprint key, touches no pixel. |

### 6.2 Non-movement proof — the video decode path is untouched

| ID | Test | Assertion |
|---|---|---|
| NM1 | `test_the_video_ffmpeg_libs_list_is_unchanged` | `_VIDEO_FFMPEG_LIBS == ("avcodec","avformat","avutil","swscale")`; `inspect.getsource(_build_facts)` byte-identical to a pinned string (the 3-tuple contract and its 5 callers are untouched). |
| NM2 | `test_the_ffmpeg_component_of_the_identity_is_unchanged` | `LegacyOpencvProxyIdentity.identity_of(decode_environment())` still contains the same `ffmpeg` sub-value; a `mismatched_ffmpeg("avformat", …)` still moves the identity (B20's protection intact). |
| NM3 | `test_frame_count_behaviour_is_unchanged` | `sample_frame_indices(60, 48)` unchanged; `decode_identity()` does **not** move on a `frame_count` change (`test_frame_count_reproducibility.py:307-316` still green); no edit to `probe_video`. |
| NM4 | `test_the_four_video_protections_still_separate_a_demuxer_difference` | re-run B20's headline (`test_a_libavformat_difference_now_separates_all_four_protections`) against the new identity — all four still fire. |
| NM5 | `test_the_model_identity_did_not_move` | `model_identity_of(STUB_ENGINE_VERSION) == "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c"`. |
| NM6 | `test_no_contract_schema_witness_or_lock_bump` | `1.0.0 / 1.2.0 / 1.2.0 / 1.1.0`. |
| NM7 | `test_openapi_schema_out_fixtures_out_are_decode_independent` | snapshot signature unchanged; `.d.` / `a97ccb68` / `<new>` absent from the snapshot; `python -m contracts.schema` and `-m contracts.fixtures` regen → zero diff. |
| NM8 | `test_the_measurement_guard_still_passes` | `test_opencv_and_pillow_decode_jpeg_identically` still runs and passes (the "R6 image half under measurement" tripwire — the batch's whole premise is that this *could* fail one day). Add PNG + WebP analogues. |
| NM9 | `test_the_simd_flag_and_zlib_are_in_the_digest_not_the_identity` | `replace(env, image_codec_simd="on")` → same `.identity`, different `.digest`; same for a zlib field if recorded. Mirrors `test_the_cpu_fields_are_still_excluded_from_the_identity`. |
| NM10 | `test_a_decode_environment_round_trips_through_its_dict` | (`test_decode_determinism.py:524`) still green — the new field flows through `to_dict` ↔ constructor. |

### 6.3 Fail-before matrix (real failures against the pre-batch tree)

| ID | Test | Pre-batch behaviour |
|---|---|---|
| FB1 | `test_a_libjpeg_turbo_version_difference_moves_the_decode_identity` | **fails** — `mismatched_image_codec("JPEG", "2.1.3-62", "3.0.1")` → `env.identity == other.identity` (libjpeg-turbo is not in the preimage). Behavioural, not a source grep. |
| FB2 | same for libpng, libwebp | **fails** identically |
| FB3 | `test_an_image_codec_difference_separates_all_four_protections` | **fails** — content-cache key, deployment identity, benchmark identity, calibration identity all blind to a libjpeg-turbo change |
| FB4 | `test_the_benchmark_gate_refuses_across_an_image_codec_difference` | **fails** — `regression_gate` passes two runs whose only difference is the JPEG library |
| FB5 | `DecodeEnvironment(..., image_codec_version=...)` | **fails** — `TypeError: unexpected keyword argument` |
| FB6 | `test_image_codec_version_is_recoverable_for_forensics` | **fails** — `"image_codec_version" not in env.to_dict()` |

### 6.4 Pass-after matrix (real protections)

| ID | Test | Assertion |
|---|---|---|
| PA1 | `test_decode_identity_now_covers_the_image_codecs` | `_image_codec_facts()` names libjpeg-turbo / libpng / libwebp; `"jpeg"`/`"png"`/`"webp"` keys in the `identity` preimage; `env.image_codec_version` is that string |
| PA2 | `test_the_image_codec_libs_are_exactly_the_codecs_behind_ALLOWED_IMAGE_MIME` | `_IMAGE_CODEC_LIBS == ("JPEG","PNG","WEBP")`; **no** TIFF / JP2 / EXR; if `ALLOWED_IMAGE_MIME` ever widens, this fails |
| PA3 | `test_a_<codec>_difference_now_moves_the_identity` ×3 | `mismatched_image_codec` → `other.identity != env.identity`, and `LegacyOpencvProxyIdentity.identity_of` blind (proves the fixture is real) |
| PA4 | `test_the_four_protections_separate_an_image_codec_difference` | all four fire post-batch; all four blind under `LegacyOpencvProxyIdentity` |
| PA5 | `test_the_benchmark_and_witness_bind_the_finer_identity` | `regression_gate` refuses; `_witness(decode_identity=a).witness != _witness(decode_identity=b).witness` |
| PA6 | `test_the_decode_segment_moved_and_the_literal_was_updated` | `STUB_ENGINE_VERSION.endswith(f".d.{decode_identity()}")`; `verify_version_alignment(STUB_ENGINE_VERSION)` does not raise |
| PA7 | `test_the_parser_is_defensive` | via a pure `_parse_image_codecs(info: str)` helper the test drives directly (no `lru_cache` obstruction): empty / absent "Media I/O" block → `None`, no raise; a WebP `(ver decoder: 0x0209, encoder: 0x020f)` line → captured verbatim |
| PA8 | `test_old_benchmark_rows_and_witnesses_still_load` | B20 pattern for `<new>` |

### 6.5 Collateral matrix (pins requiring updates)

| File:line | Old | New |
|---|---|---|
| `backend/config.py:41` | `…f.e629642c.d.a97ccb68` | `…f.e629642c.d.<new>` |
| `tests/test_decode_identity_completeness.py:341` | `.endswith(".d.a97ccb68")` | `.endswith(".d.<new>")` |
| `tests/test_decode_identity_completeness.py:364` | `"a97ccb68" not in snap` | `+ "<new>" not in snap` |
| `tests/test_crop_identity.py:63` | `B23_STUB_ENGINE_VERSION = "…d.a97ccb68"` | `…d.<new>` |
| `tests/test_crop_identity.py:291` | `.endswith(".d.a97ccb68")` | `.endswith(".d.<new>")` |
| `tests/test_crop_identity.py:299-300` | history list | `+` a row `"…f.e629642c.d.<new>"` |
| `tests/test_video_selection_identity.py:77` | `B23_STUB_ENGINE_VERSION` | `…d.<new>` |
| `tests/test_video_selection_identity.py:341` | `.endswith(".d.a97ccb68")` | `.endswith(".d.<new>")` |
| `tests/test_video_selection_identity.py:413` | `decode="a97ccb68"` | `decode=decode_identity()` |
| `tests/test_startup_alignment.py:76,98,125,211` | `…d.a97ccb68` (incidental — the `.f.` is the stale part) | `…d.<new>` (cosmetic consistency) |
| `tests/test_run_identity.py:427` | forensic field tuple | `+ "image_codec_version"` (and the SIMD/zlib digest fields if recorded) |
| `tests/run_all.py` | N suites | `+ import test_image_decode_identity`, `+` tuple row, N+1 |

**History pins left alone (verified):** `test_crop_identity.py:54`
(`B22_STUB_ENGINE_VERSION`), the `…d.c20a064a` / `…d.a97ccb68` history rows,
`test_evaluation_identity.py` / `test_run_identity.py` synthetic decode constants
(`c20a064a`, `UNDER_FFMPEG`, `UNDER_MSMF`).

---

## 7. Open-finding re-rank — and the correct next batch

Axes 1–5 (5 = worst): **Repro** / **Cache** (corruption risk) / **Deploy**
(risk from the fix) / **Migr** / **Cx**. **Act** = can be a batch now within
standing constraints (no SCRFD, no ONNX, no architecture change, no
`CONTRACT`/`EvidenceBundle`/`ThresholdInfo`/OpenAPI-snapshot change).

| Finding | Repro | Cache | Deploy | Migr | Cx | Act | Rank / disposition |
|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| **N5** — no dependency lockfile | **3** | 2 | 3 | **1** | 2 | ✅ | **#1 — the next batch (B24).** Re-derivation (§2.2) shows it is *load-bearing*: there is no lockfile and no CI config; every identity literal (`.d.` especially) is valid only for one resolved dependency set; `test_the_configured_literal_matches_the_engines` fails on any box that resolves a different in-range `opencv-python`. Zero cache generation, no `.d.` move, no deploy dance. Also closes the `numpy`-in-no-identity gap (`pyproject.toml:20-24`). |
| **R13** — face-crop instability (310×, 25.7%, measured) | 5 | 4 | 2 | 5 | 5 | ❌ SCRFD | most severe; un-batchable (fix = SCRFD, a standing constraint) |
| **R31** — witness digest omits model identity / gate / `.f.` | 3 | 3 | 3 | 3 | 3 | ✅ | **#2.** Real gap; `CALIBRATION_WITNESS_VERSION` 1.3.0 migration path exists (schema-version-as-parameter). No model to calibrate yet → B25/B26 with real-model onboarding. Fold **N6/N7**. |
| **OR1-b** — `backend/worker_main.py` composition root | 1 | 2 | 3 | 1 | 1 | ✅ | **#3.** Scheduled in `BATCH_23_R3_DESIGN.md` §1.12; zero cache generation; closes the config-default alignment gap. |
| **R30** — image codec libs outside `decode_identity` | **2** | **2** | 3 | 2 | 3 | ✅ | **Demoted to D-series.** §1/§2: structural gap, **measured pixel-neutral** across libjpeg-turbo 2.1.3↔3.1.4 / libpng / libwebp / zlib↔zlib-ng for accepted MIME. Bundle with D8/D9/D10 in a "decode-completeness pass 2" batch, **after N5**. Corrected identity design = §3; migration = §5; tests = §6. |
| **D9** — `cv2.__version__` drops the wheel's 4th digit | 2 | 2 | 2 | 2 | 2 | ✅ (with R30 or N5) | N5 (exact pin) neutralises it cheaply; otherwise bundle with R30. |
| **D8** — SIMD path not in identity | 2 | 2 | 2 | 2 | 3 | ✅ (with R30) | If the pass-2 batch lands: libjpeg-turbo SIMD → `digest` (§3); swscale (video) SIMD stays a recorded residual. |
| **D10** — same version string, different build | 2 | 2 | 1 | 4 | 5 | ❌ | needs a binary hash; record only. |
| **N8** — one decode identity is a union over both paths | 1 | 1 | 1 | 5 | 4 | — | **CLOSED — accepted architectural cost** (§4). |
| **R32** — `batch_size` neutrality backend-dependent | 2 | 2 | 1 | 3 | 3 | ❌ | not a defect today; double-tripwired post-B23-cleanup |
| **R34** — OR1 non-rolling-deployable without coordination | 1 | 1 | 3 | 1 | 1 | tracked | mitigated (runbook); applies to any `.d.`/`.f.` batch |
| **R24** — whole-frame square resize (every video bundle) | 2 | 1 | 2 | 4 | 4 | ❌ video face stage | deterministic distortion → correctness, not reproducibility |
| **R28** — `resize` predicate area-filters an enlarged axis | 2 | 2 | 2 | 2 | 2 | ⚠️ rides a pixel-moving batch | trigger = SCRFD / R23 / R16 |
| **R23** — correctness halves (13% border stretch, live) | 2 | 1 | 2 | 4 | 4 | ❌ SCRFD | deterministic; every fix moves pixels |
| **R18** — faceless clip gets a decided verdict | 1 | 1 | 1 | 4 | 4 | ⚠️ needs `face_stage` | also blocks the R21 headline-rule tightening |
| **R26** — published bbox ≠ scored region | 1 | 1 | 1 | 3 | 3 | ⚠️ partly SCRFD; touches `EvidenceBundle` | forensic |
| **R16** — one face's verdict published as the media's | 1 | 1 | 1 | 2 / 5 | 2 / 5 | ⚠️ product decision | prose path Low value; per-face touches `EvidenceBundle` |
| **N6 / N7** — fit-optimiser config / `np.quantile` method unrecorded | 2 | 1 | 1 | 3 | 2 | ✅ with R31 | fold into the witness-hardening batch |
| **N3** — advisory heuristic constants | 1 | 1 | 1 | 1 | 1 | defer | add tripwires when one is next edited |

### 7.1 Does B24 remain R30? — **No.**

**B24 = N5 (dependency lockfile).** Rationale:

1. §1/§2 (measured): R30 has **no demonstrable live reproducibility exposure**
   for the declared deployment. It does not clear the bar for a cache generation.
2. N5 is **foundational** for every identity literal in the platform — including
   any future R30/D-series work. R30-on-top-of-no-lockfile makes the fragility
   worse (more build-specific inputs to `.d.`).
3. N5 costs **zero cache generation**, no `.d.` move, no R34 deploy coordination.
4. N5 also fixes a *currently-broken* property: a fresh `pip install` on a box
   that resolves a different in-range `opencv-python` fails
   `test_the_configured_literal_matches_the_engines` — the test suite is not
   reproducible today without matching this box's exact resolution.

### 7.2 The correct next batch — N5, scoped

- **Add a lockfile** pinning the full resolved dependency tree — exact versions
  and hashes — for the runtime set (`opencv-python`, `numpy`, `pydantic`,
  `fastapi`, `starlette`, `pydantic-settings`) and, separately, the `dev` set.
  Tooling choice (`uv lock` / `pip-compile` / `pip freeze` + hashes) is the one
  open decision — recommend `uv` (fast, hash-pinned, `uv.lock` is standard) but
  `pip-compile --generate-hashes` is equivalent for this purpose.
- **A test** (`tests/test_dependencies.py` extension) asserting the lockfile
  exists, is consistent with `pyproject.toml`'s ranges, and — the load-bearing
  part — that `cv2.__version__` and the resolved `opencv-python` match what the
  lockfile pins, so a drift is a red test not a silent `.d.` mismatch.
- **A CI entry point** (the repo has none) that installs from the lockfile and
  runs `tests/run_all.py`.
- **No identity movement.** `.d.` / `.f.` / `STUB_ENGINE_VERSION` /
  `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` /
  `LOCK_VERSION` / OpenAPI / `schema_out` / `fixtures_out` — **none moves.** The
  lockfile pins what is *already installed*; the shipped `a97ccb68` was computed
  against this resolution and stays valid.
- **Migration:** none. No cache generation. A future dependency bump becomes a
  reviewed lockfile change that, if it moves `cv2.__version__`, is caught by the
  new test and handled as its own `.d.`-moving batch.
- **Deploy:** strictly *safer* — removes the "two hosts resolved different
  versions" failure mode.

### 7.3 After N5

The "decode-environment completeness pass 2" batch (R30 + D8 + D9 + D10-residual)
becomes a clean B25/B26 candidate: one cache generation for the whole cluster,
the corrected identity design in §3, the migration in §5, the tests in §6. With
N5 in place, its `.d.` literal is reproducible and its value is a reviewed
artifact rather than a build accident.

---

## 8. Final gate

## **APPROVE WITH CHANGES**

- The R30 identity design in this document (§3 matrix, §4 N8, §5 migration, §6
  tests) is **technically sound and implementation-ready** — S2/S3/S4 resolved,
  the SIMD/zlib membership corrected, N8 reclassified, output-identical and
  non-movement proofs added.
- **But R30 must not be implemented as B24.** S1 is confirmed and strengthened:
  R30 is a structural residual with no measured pixel impact, and spending a
  cache generation + an R34 deploy on it is not justified. **B24 is N5.**

### Remaining defects

| # | Class | Severity | Evidence | Impact | Required correction |
|---|---|---|---|---|---|
| C1 | **specification** | **High (blocking for a B24=N5 start)** | §7.2 — N5's tooling choice (`uv` vs `pip-compile`) and the exact shape of the lockfile-vs-engines consistency test are not yet specified. | Cannot begin N5 implementation without deciding the lock tool and the test's assertion. | A short **`B24_N5_DESIGN.md`** (≈1 page): tool choice, lockfile location, the `test_dependencies.py` assertions, the CI entry point. This is the only design work outstanding. |
| C2 | specification | Medium (blocking for the pass-2 batch, **not** for B24) | §3 — `_parse_image_codecs` must be a pure function (not `lru_cache`d) so PA7 is testable; the `getBuildInformation()` "Media I/O" format varies across wheels/distros (WebP ABI vs semver, absent SIMD lines, system paths). | Pass-2 batch parser breaks on non-Windows / non-wheel builds; PA7 un-implementable. | Carried in §3/§6.4 of this doc; ratify when the pass-2 batch is scheduled. Not a B24 concern. |
| C3 | documentation | Low | §5.3 — the audit-trail wording for the cross-batch benchmark refusal. | An operator reads "correct" where the truth is "conservative false incomparability". | Use "false incomparability (accepted cost, B9/B20 precedent)" in `REPRODUCIBILITY_AUDIT.md` when the pass-2 batch lands. |
| C4 | documentation | Low | §4 — `REPRODUCIBILITY_AUDIT.md` N8 row still says "narrowed"/"tracked". | N8 masquerades as pending work. | Rewrite the N8 row per §4 ("CLOSED — accepted architectural cost"). Can be done now, independent of any batch. |

### May implementation begin after B24-R2?

**Not directly.** One more, small design iteration is required — **`B24_N5_DESIGN.md`**
— to settle the lockfile tooling and the consistency-test shape (C1). That
document is ≈1 page and has no open architectural questions. Once it lands,
**N5 implementation may begin as B24.**

The R30 identity work in this document needs **no further design iteration** —
it is ready to be picked up as the "decode-completeness pass 2" batch (B25/B26)
after N5, subject only to ratifying C2 at that time.
