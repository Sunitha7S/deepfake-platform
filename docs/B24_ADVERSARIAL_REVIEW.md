# B24 (R30) — Fresh Independent Adversarial Review

**Object:** `docs/B24_DESIGN.md`, treated as untrusted. Every claim re-derived
from the current repository; the design is not quoted as evidence for anything.
**Date:** 2026-09-09. **Scope:** review only — no implementation, no code.

**One-line finding:** the design's *technical* core is sound (Option A, the
migration profile, the Option-C architecture rejection, the parser-free claim
all hold), but its *risk framing and priority* are overstated for the deployment
the repository actually declares, and two of its three ratification decisions
(SIMD, zlib) are wrong on the merits. **Verdict: APPROVE WITH CHANGES.**

---

## 1. Independent re-derivation (source only)

### 1.1 `DecodeEnvironment` *(inference/decode_env.py:160-307, live)*

`@dataclass(frozen=True)`, nine required fields, no defaults:
`opencv_version, video_backend, backend_source, ffmpeg_version, platform_system,
platform_machine, python_version, cpu_baseline, cpu_dispatch`.

- `identity` (`:181-234`) = `fingerprint({opencv, backend, ffmpeg, system,
  machine})` — **five keys**. Live: `a97ccb68`.
- `digest` (`:236-258`) = `fingerprint({identity, cpu_baseline, cpu_dispatch})`.
  Live: `4aded179`. **Nothing pins `4aded179`** (`grep -rn 4aded179` → nothing) —
  the digest may move freely.
- `to_dict` (`:260-273`) emits all nine fields + `identity` + `digest`.
- Only **two** `DecodeEnvironment(...)` constructions exist repo-wide
  (`grep -rn 'DecodeEnvironment('`): `decode_env.py:378` (production) and
  `test_decode_determinism.py:529` (round-trip from `to_dict`, so a new field
  flows through automatically). `replace(env, …)` sites are unaffected by a new
  field.

### 1.2 `decode_identity()` and runtime-version composition

- `decode_identity()` (`decode_env.py:400-408`) = `decode_environment().identity`.
  **Argument-free, media-kind-agnostic.** `decode_fingerprint()` = `.digest`.
- `contracts/version.py`: `DECODE_MARKER = ".d."` (`:155`);
  `_MODEL_MARKERS = ('+c.', '.g.', '.f.', '.d.')` (`:164`).
- `compose_runtime_version(*, model, decode)` (`:365-384`) →
  `f"{model}{DECODE_MARKER}{decode}"`; raises only if `model` is empty, already
  contains `.d.`, or `decode` is falsy / contains `.d.`. **It never inspects the
  value's shape.**
- `parse_runtime_version` (`:387-403`) = `str.partition(".d.")`; no `.d.` →
  `decode=None` (not an error). `model_identity_of` (`:406-413`) =
  `parse_runtime_version(v).model` — strips `.d.`.
- Live: `fingerprint()` output is `[0-9a-f]{8}` (`FINGERPRINT_LENGTH == 8`),
  order-independent, and a new dict key changes the hash.

### 1.3 `STUB_ENGINE_VERSION` construction *(backend/config.py:41, live)*

Literal string
`stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.a97ccb68`. Not composed at
runtime — a hand-maintained literal that `workers.engine_registry.
verify_version_alignment` (`:91-119`) holds to what **both** engines report
(`engine_versions()` `:80-88` builds `ImagePipeline` **and** `VideoPipeline`
unconditionally). `.d.a97ccb68` is `decode_identity()` on the build box; live:
`STUB_ENGINE_VERSION.split(".d.")[-1] == "a97ccb68"`.

### 1.4 Image pipeline decode path

`inference/image_pipeline.py:105-107` `model_version` =
`compose_runtime_version(model=self._spec.effective_version,
decode=decode_identity())`. The decode itself:
`inference/preprocess.py:191-217` `decode_image` → `cv2.imdecode(buffer,
cv2.IMREAD_COLOR)` (`:211`) → `to_rgb` (`cv2.cvtColor`, OpenCV's own code).
`imdecode` dispatches by byte signature to the **bundled** codec:
libjpeg-turbo (JPEG, incl. its own YCbCr→RGB), libpng + zlib (PNG),
libwebp (WebP). No structured version API — see §1.7.

### 1.5 Video pipeline decode path

`inference/video_pipeline.py:159-160` — identical composition, **same
`decode_identity()`**. The decode: `preprocess.py:1042,1170` `open_video(path)`
→ `cv2.VideoCapture` → `grab()`/`retrieve()` (`:1188`) — FFmpeg backend
(libavformat demux, libavcodec decode, libswscale YUV→BGR). **No `imdecode`, no
libjpeg/libpng/libwebp** anywhere on the video path (`grep imencode|imwrite
inference/ workers/` → none; no JPEG-thumbnail encode path exists).

The image and video decode paths are **library-disjoint**. The one
`decode_identity()` string spans both.

### 1.6 `ALLOWED_IMAGE_MIME` *(contracts/api.py:33, live)*

`("image/jpeg", "image/png", "image/webp")`. Enforced at `:63`. So the codecs
`imdecode` can be asked to run are exactly libjpeg-turbo, libpng(+zlib),
libwebp. TIFF / JPEG 2000 / OpenEXR are in the build but **unreachable**.

### 1.7 Current "Media I/O" extraction logic — **there is none**

`grep -n 'Media I/O\|JPEG\|libjpeg\|PNG\|libpng\|WEBP\|libwebp\|ZLib'
inference/decode_env.py` → the only hits are the class docstring's discussion of
the **video** path. **No code parses the image-codec block.** `cv2.__version__`
is the sole proxy, and `test_the_wheel_patch_version_gap_is_a_known_residual`
(`test_decode_identity_completeness.py:409-421`, live) confirms
`cv2.__version__ == "4.8.1"` while the wheel is `4.8.1.78` — the 4th component
is dropped.

### 1.8 Current FFmpeg extraction logic

`decode_env.py:310-371`. `_VIDEO_FFMPEG_LIBS = ("avcodec", "avformat", "avutil",
"swscale")` (`:325`); `_AVLIB_RE = r"^\s*(avcodec|avformat|avutil|swscale):\s*
YES\s*\(([^)]+)\)"` with `re.M` (`:326-328`) — captures the parenthetical
**verbatim**. `_build_facts()` (`:332-371`) returns `(ffmpeg_string, cpu_baseline,
cpu_dispatch)` — a **3-tuple**; all five call sites do `x, _, _ = _build_facts()`
(`decode_env.py:376`, `test_decode_identity_completeness.py:204,222,241`,
`test_frame_count_reproducibility.py:355`, plus two studies). Live: the ffmpeg
string is `"avcodec 58.134.100 / avformat 58.76.100 / avutil 56.70.100 /
swscale 5.9.100"`.

### 1.9 `getBuildInformation()` "Media I/O" block *(live)*

```
Media I/O:
    ZLib:                        build (ver 1.2.13)
    JPEG:                        build-libjpeg-turbo (ver 2.1.3-62)
      SIMD Support Request:      YES
      SIMD Support:              NO
    WEBP:                        build (ver encoder: 0x020f)
    PNG:                         build (ver 1.6.37)
```

**Every codec on this build is `build` / `build-libjpeg-turbo` — bundled, not
system-linked.** FFmpeg is `YES (prebuilt binaries)` — also bundled. This matters
for §2.

---

## 2. Recomputed risk — is R30 real, or theoretical?

### 2.1 The exact chain

**Cache key** (`backend/service.py:98` `find_by_content(sha256,
settings.model_version)`; `:227` `idempotency_key = f"{sha256}:{model_version}"`)
= the configured literal = `…f.e629642c.d.<D>`, where
`<D> = fingerprint({opencv: cv2.__version__, backend: <resolved>, ffmpeg:
<4-lib string>, system: platform.system(), machine: platform.machine()})`.

**Two hosts share the key** ⇔ identical `cv2.__version__`, resolved video
backend, FFmpeg 4-lib string, OS, arch.

**Two hosts decode a JPEG differently** ⇔ different libjpeg-turbo
version / build / SIMD (or libpng / libwebp for the other formats).

**Both conditions at once — is it reachable?**

| Deployment | Reachable? | Why |
|---|---|---|
| Two hosts on the **same** `opencv-python==X.Y.Z.NNN` wheel | **No** | The wheel bundles one frozen libjpeg-turbo/libpng/libwebp; identical byte-for-byte. |
| `opencv-python` `4.8.x` vs `4.9.x`/`4.10.x` (the design's "cross-wheel variance" example — libjpeg-turbo 2.1.3 → 3.0.x) | **No — already caught** | `cv2.__version__` differs (`"4.8.1"` vs `"4.9.0"`) → `opencv` key differs → **different `.d.` today**. The design's own supporting example is covered by an existing key. |
| Two `opencv-python` wheels `4.8.1.78` vs a hypothetical `4.8.1.79` that rebuilt a bundled codec but kept the prebuilt FFmpeg | **Yes, narrowly** | `cv2.__version__` both `"4.8.1"`, FFmpeg string unchanged → same `.d.`, different codec. **This is exactly D9** ("`cv2.__version__` drops the 4th component"). opencv-python 4th-digit rebuilds are for CI/manylinux/Python-support and rarely touch 3rd-party libs. |
| conda-forge / distro / homebrew OpenCV, **system-linked** libjpeg-turbo, `apt upgrade libjpeg-turbo` between two hosts | **Yes** | `cv2.__version__`, FFmpeg, OS, arch all unchanged; system libjpeg-turbo changed → same key, different decode. **But `pyproject.toml` pins `opencv-python` (the wheel), not a system build.** |

### 2.2 Verdict on the risk

- **Reproducibility-affecting:** *conditionally.* For the **declared** deployment
  (`opencv-python>=4.8,<5`, all codecs bundled per §1.9), R30's residual over the
  status quo is the **D9 case only** — a 4th-digit wheel rebuild that changes a
  bundled codec. That is a real but narrow gap, materially **less severe than
  R27's**, which was closed on a *measured* frame-count divergence
  (`test_decode_identity_completeness.py:260-277`). R30 has **no measured
  divergence**, and `test_opencv_and_pillow_decode_jpeg_identically`
  (`test_decode_determinism.py:447-476`, live: **passes**, incl. progressive
  JPEG) is standing counter-evidence — `pyproject.toml`'s own comment calls this
  test "how R6's 'different libjpeg builds disagree' hypothesis is kept under
  measurement rather than assumed."
- **Cache-corrupting:** *yes in principle, D9-tier in practice* for the wheel;
  R27-tier only for an unpinned system-linked build the repo does not specify.
- **Deployment-relevant:** *yes* — closing it moves `STUB_ENGINE_VERSION`, which
  triggers the R34 rolling-deploy constraint (coordinated restart or the skip
  hatch). That cost is **paid regardless of how narrow the gain is**.

**The design's framing of R30 as "the image analogue of R27" with "live
cache-corruption exposure" and "the single highest-value actionable finding" is
overstated for the deployment the repository declares.** It is a D9-class
hardening dressed as a flagship batch.

### 2.3 The cheaper root-cause fix the design under-weights: N5

A dependency **lockfile** pinning `opencv-python==4.8.1.78` (and numpy) exactly:

- removes the 4th-digit-rebuild vector entirely (a codec change now requires a
  reviewed lockfile bump);
- costs **no cache generation, no `.d.` move, no R34 deploy dance**;
- also closes the `numpy`-in-no-identity gap (the `pyproject.toml` comment:
  *"a promotion change there moves cache keys without moving a single verdict"* —
  numpy is capped `<2` but floats within `1.26,<2`);
- is the meta-problem that R30, D9, and D10 are all symptoms of.

With a lockfile in place, R30's remaining value (system-linked deployments) is
addressable later, bundled with D8/D9/D10, spending **one** cache generation on
the whole decode-environment cluster instead of one on R30 alone.

---

## 3. The proposed identity expansion — challenged

### 3.A `image_codec_version` field (Option A)

| Check | Finding |
|---|---|
| **Completeness** | libjpeg-turbo + libpng + libwebp version strings are the right set for `ALLOWED_IMAGE_MIME`. **PA2** (`_IMAGE_CODEC_LIBS` ↔ the MIME allowlist) is a genuine guard against a future MIME widening. OK. |
| **Maintainability** | A separate `_image_codec_facts()` (leaving `_build_facts`'s 3-tuple and its five callers byte-identical) is the correct minimal-blast choice. OK. |
| **Collision resistance** | Same `fingerprint()` (32-bit) as everything else — a change *detector*, not a boundary. Consistent with the platform. OK. |
| **Future codec additions** | Bounded by `ALLOWED_IMAGE_MIME`; adding AVIF later = extend `_IMAGE_CODEC_LIBS` + re-generate. OK. |
| **Format fragility** | **Real.** §1.9 shows WebP reports an ABI number (`encoder: 0x020f`), some builds `decoder: 0x0209, encoder: 0x020f`; system-linked builds show a path; the SIMD lines are absent on builds without libjpeg-turbo. The design's §8.3 correctly requires "capture the raw line verbatim, do not parse into components." **This correction stands.** |

**Option A itself is acceptable.** The design's claim that Option D
(composite `.d.`) offers "operational visibility Option A lacks" is **wrong**:
`DecodeEnvironment.differences()` (`:284-307`) already reports field-by-field, so
under Option A a mismatch already says "`image_codec_version` changed" vs
"`ffmpeg_version` changed". D's composite adds nothing to `differences()` and
nobody decodes the opaque `.d.` value. D is a wash, not an improvement — the
design over-credits it.

### 3.B SIMD inclusion — **the design is wrong**

The design (§2.6) puts `SIMD Support: YES|NO` **in `identity`** (the cache key),
on a "tie toward missing" argument, noting B8-E6 does not cover it.

Re-derivation:

- **Does SIMD belong in identity?** The `identity` docstring's own membership
  rule (`decode_env.py:211-221`): *"OUT, because including them would fragment
  the cache on things measured **not** to change a pixel"* — and it applies that
  rule to `cpu_baseline` / `cpu_dispatch`, which B8-E6 measured pixel-neutral,
  keeping them in `digest` only.
- **Is SIMD already covered elsewhere?** Partially — it is the direct analogue
  of `cpu_dispatch`, which is in `digest`. The design does **not** note this
  parallel.
- **Can SIMD alter decoded pixels?** libjpeg-turbo's SIMD IDCT/upsampling is
  **designed and tested to be bit-identical** to its C path for the default
  accurate integer IDCT (`JDCT_ISLOW`), which is what OpenCV's JPEG loader uses.
  The design's "±1 LSB on some upsampling paths" was true of libjpeg-turbo
  **1.x**; the platform runs **2.1.3** (§1.9). Fast IDCT (`JDCT_IFAST`) can
  differ, but OpenCV does not select it.
- **Or merely performance?** For OpenCV's usage: **performance only**, by
  libjpeg-turbo's own guarantee.

**Correction (specification):** the SIMD flag goes in **`digest`** (forensic),
not `identity` (cache key) — exactly the `cpu_dispatch` treatment: recorded,
pinned by a test, promoted to `identity` only if a divergence is ever observed.
Putting it in the cache key fragments a fleet mixing SIMD-on and SIMD-off
libjpeg-turbo wheels (Linux wheels ship SIMD-on; this Windows box is SIMD-off)
for a difference libjpeg-turbo guarantees does not exist. The design's §8.1
("ratify SIMD in the cache key") is itself the defect.

### 3.C zlib inclusion — **the design is wrong**

The design (§2.6) puts zlib **in `identity`**, "flagged weakest".

Re-derivation:

- **Does zlib affect decoded pixels?** No. PNG decode is `inflate`, which RFC
  1951 / RFC 1950 make **bit-exact for any conformant zlib implementation** on
  any valid stream. zlib does DEFLATE decompression only; it never touches a
  pixel value. libpng does the filtering/unfiltering, gamma, bit-depth scaling,
  interlace — **libpng is the PNG risk, not zlib.**
- **Is zlib version drift observable?** In `getBuildInformation` yes; in decoded
  pixels no.
- **Does zlib belong in identity?** **No** — it is precisely a thing "measured
  [by the RFC] not to change a pixel", which the `identity` docstring says to
  keep OUT. Including it violates the membership rule the design claims to
  follow.

**Correction (specification):** **exclude zlib from `identity`.** Optionally
record it in `digest` for forensics. Capture libpng's version (that carries the
PNG pixel risk).

### 3.D Net of §3

Of the design's three ratification asks (§8.1–§8.3): **§8.3 (verbatim parse) is
correct; §8.1 (SIMD in cache key) and §8.2 (zlib in cache key) are wrong** and
must be reversed. The identity should carry exactly **libjpeg-turbo version,
libpng version, libwebp version** — the three that can actually differ across
builds *and* move a decoded pixel.

---

## 4. N8 disposition — re-derived

The design **narrows** N8. Re-derivation against the current architecture:

### 4.1 There are no single-modality deployments

`workers/engine_registry.py:80-88` `engine_versions()` **hardcodes both**
`ImagePipeline(...)` and `VideoPipeline(...)`; `verify_version_alignment`
(`:91-119`) requires the one configured literal to equal **both**. There is no
supported "image-only" or "video-only" deployment — every process holds both
engines and one `model_version`. So the design's (and the brief's) "image-only /
video-only" framing for N8 is **moot in the current architecture**: N8 is not
"the image path carries a video component", it is "**the one decode identity is
a union over two library-disjoint decode paths**, so any library change in
either path regenerates the whole (image+video) content cache."

### 4.2 What Option A does to N8

Option A adds the image-codec libraries to the same union. After B24:

- a libjpeg-turbo bump regenerates the **video** cache too (video uses no
  libjpeg) — a *new* false-miss class;
- an FFmpeg bump still regenerates the **image** cache (unchanged from today).

So Option A makes the union **symmetric** and **absolutely larger**. "Narrowed"
is the wrong verb — the design's own §3 admits "it is doubled and made
symmetric." That is not a narrowing.

### 4.3 Correct disposition

N8 is **the single-literal / one-identity-for-both-engines architectural cost**,
already documented and accepted in `docs/RUN_IDENTITY.md` §3 ("a cache key breaks
ties toward missing"), already paid by **B20** (its `.d.` FFmpeg move regenerated
the image namespace) and **B23** (its `.f.` video-selection move regenerated the
image namespace). It is the same class as **R9** (quarantine — leaky by design,
reported every run) and **R32** (a documented conditional cost, tripwired, no
action): a finding that is *working as designed within a deliberate
architecture*, whose only "fix" (Option C — per-kind identities) is an
architecture change the standing constraints forbid and that B23 explicitly
weighed and rejected.

**Recommendation — one of:**

- **Close N8** as "accepted architectural cost", cross-referenced to
  `RUN_IDENTITY.md` §3 and the B20/B23 precedents, the way R9 is closed
  ("by design, reported every run"); **or**
- keep N8 **tracked-but-blocked** like R13 — open, but its only resolution
  (Option C) is constraint-forbidden, so it carries no B24-or-later action item.

Either is more honest than "narrowed", which implies incremental progress toward
a closure that is architecturally unreachable. **Not promote** (false-miss, not
wrong verdict — Low). **Not split** (no decomposition exists).

---

## 5. Migration audit — second-order effects

| Area | Design claim | Re-derivation | Hidden effect? |
|---|---|---|---|
| **Parser** | zero change | **Confirmed.** `decode_identity()` returns `[0-9a-f]{8}`; `compose_runtime_version` / `parse_runtime_version` / `_MODEL_MARKERS` never inspect the value; the new `image_codec_version` string is hashed, never emitted. | none |
| **Benchmark** | rows load unchanged; `RUN_SCHEMA_VERSION` unbumped; cross-B24 comparison "refuses (correct)" | `BenchmarkRun.decode_identity: str \| None` stores the passed value; schema unchanged; `_decode_comparability` (`benchmark.py:728-776`) refuses two different non-null identities. **BUT** the refusal is a **false incomparability** — two runs scored on byte-identical pixels (same libjpeg-turbo; B24 only added a fingerprint key) become "not comparable". The design calls this "correct"; it is *over-conservative*, an accepted cost, not a correctness gain. Same as B9/B20. | **minor framing defect** (§9) — no functional harm |
| **Witness** | no `CALIBRATION_WITNESS_VERSION` bump | **Confirmed.** `compute_witness` (`calibration_protocol.py:211-278`) hashes `decode_identity` as a **raw string value** for schema `1.2.0` (`SCHEMA_VERSIONS_WITH_DECODE`); `__post_init__` recomputes from the *stored* string; verification dispatches on the payload's own `schema_version` (`:230-243`). Changing what `decode_identity()` *returns* touches no stored witness. R31 (digest omits model id) is orthogonal — B24 changes a field the digest **does** cover. | none |
| **Startup alignment** | passes with the updated literal | **Confirmed.** `backend/app.py` lifespan format check (`is_composed` + `is_environment_pinned` + round-trip) passes for `…d.<new>` (clean hex suffix). `workers/runner.py` `verify_engine_alignment_at_startup` → `verify_version_alignment` matches iff the collateral literal is updated in the same commit; otherwise the worker **fails closed** (correct). | none |
| **Rollback** | B20 profile | **Confirmed.** Revert `decode_env.py` + the literal → engines recompute `a97ccb68` → gate passes. Orphaned `…d.<new>` entries expire under `result_ttl_s`. Partial rollback → worker refuses to start. | none |
| **Rolling deploy** | R34 applies | **Confirmed and it is a real cost** — see §2.2. A pod-by-pod update needs a coordinated worker restart or `DFP_SKIP_VERSION_ALIGNMENT=true` for the window. **This cost is fixed regardless of how narrow R30's gain is** — the argument for spending it on R30 *alone* rather than on the D8/D9/D10/N5 cluster is weak (§2.3). | **priority defect** (§9) |
| **Digest / forensic record** | (not addressed) | `DecodeEnvironment.digest` also moves (it includes `identity`). **Nothing persists the digest in production** (`grep` — only `decode_identity()` reaches `BenchmarkRun`/`CalibrationWitness`, and it is passed in by the harness). No migration concern, but the design should state it. | **minor doc gap** |
| **`fixtures_out` / `schema_out` / OpenAPI** | do not move | **Confirmed** — `grep -rln 'a97ccb68\|decode_identity\|\.d\.' fixtures_out/ schema_out/ backend/openapi.snapshot.json` → nothing; fixtures carry `model.version = "1.0.0-dev"`. | none |

---

## 6. Dependency & architecture audit

| Check | Finding |
|---|---|
| **New dependency edges** | None. `_image_codec_facts()` parses `cv2.getBuildInformation()` — `cv2` is already imported at module scope in `decode_env.py`. `backend/config.py` change is a literal. |
| **Import-cycle risk** | None. No new imports in either file. |
| **Startup-order risk** | None. `decode_environment()` is called lazily (per job / at `engine_versions()` time), `_build_facts` / `resolve_video_backend` are `lru_cache`d; a parallel `_image_codec_facts` `lru_cache` is process-safe. |
| **Worker/API asymmetry** | None introduced. `getBuildInformation()` is worker-only (the API is cv2-free); the API reads the configured literal. Identical to `_build_facts` today. Verified: `import backend.app` pulls no `cv2` — B24 adds no module-scope import to `decode_env`, so that holds. |
| **OR1-b interaction** | Disjoint files (`decode_env.py` + `config.py` vs a new `backend/worker_main.py` + a re-added Setting + `run_with_shutdown` call site). B24 does not need OR1-b. B24 *does* strengthen the case for shipping OR1-b soon: after B24's literal move, a config-default worker with a stale literal produces silent false misses until OR1-b makes the full check run on that path. Not a blocker; a scheduling note. |
| **Option C (rejected by the design)** | **The rejection is correct.** Per-kind `decode_identity(kind)` gives two `model_version` strings, which breaks `engine_versions()` / `verify_version_alignment` / the single `STUB_ENGINE_VERSION` / the single `service.py` cache key / `/health`. It is an architecture change and is the "two literals" route B23 already rejected (`VIDEO_SELECTION_IDENTITY.md` §3). |

**No architecture defects.** Option A stays entirely inside the current
architecture: one identity, one literal, one cache key, value-only movement.

---

## 7. Test-plan audit

The design's four matrices mirror the B20 suite — a good template. Gaps found:

| # | Category | Defect | Recommended addition |
|---|---|---|---|
| T1 | fail-before | **FB1** (`test_the_decode_identity_omits_the_image_codecs`) is specified as a **source grep** (`inspect.getsource(identity.fget)` contains no `image_codec`). Weak — passes on an unrelated variable name, breaks on a comment. B20's real fail-before was **behavioural** (`mismatched_ffmpeg` → identities equal before). | Make FB1 behavioural: `mismatched_image_codec("JPEG", …)` → `env.identity == other.identity` on the pre-B24 tree. |
| T2 | pass-after | **PA10** ("the parser is defensive" — Media I/O block absent → `image_codec_version is None`) is un-specified as to *how*. If `_image_codec_facts` is `lru_cache`d (like `_build_facts`), monkeypatching `getBuildInformation` after first call is a no-op. | Specify a pure parse helper `_parse_image_codecs(info: str)` that the test drives directly with a synthetic/empty block; `_image_codec_facts` calls it. |
| T3 | **non-movement** | **Missing the output-identical proof.** B22 (`test_the_shipped_pixels_did_not_move`) and B23 (`test_the_shipped_video_bundle_did_not_move`) both pin a real bundle's `p_fake` byte-identical across the batch. B24 has no such test in any matrix. B24 *is* pixel-neutral (adds a dict key), so this is a tripwire against someone making `image_codec_version` a functional read. | Add `test_the_shipped_image_bundle_did_not_move` **and** `…_video_…`: a real image + real video bundle, `p_fake` and `verdict` unchanged, only `model.version` moves. |
| T4 | non-movement | **Missing** an analogue of `test_batch_20_did_not_touch_the_frame_count_amplifier_or_validation` (`test_decode_identity_completeness.py:434`). | `test_batch_24_did_not_touch_the_frame_count_or_the_video_ffmpeg_libs`. |
| T5 | non-movement | The **digest** (`4aded179`) moving is not asserted-as-expected anywhere, and no test confirms nothing pins it. | `test_the_forensic_digest_moved_with_the_identity_and_nothing_pins_its_literal`. |
| T6 | vacuous risk | **PA3** ("a libjpeg-turbo difference now moves the identity") is real only if the design keeps codec **versions** in identity. If §3.B/§3.C corrections land (SIMD→digest, zlib→out), the PA-matrix rows for SIMD and zlib must move to **digest** tests (`replace(env, image_codec_simd=…)` → same `.identity`, different `.digest`) — mirroring `test_the_cpu_fields_are_still_excluded_from_the_identity`. | Re-scope PA4 (SIMD) and the zlib row to digest-exclusion tests. |
| T7 | coverage | No test asserts the **measurement status** is unchanged — i.e. `test_opencv_and_pillow_decode_jpeg_identically` still runs and still passes (it is the "R6 image half under measurement" guard, and B24's whole premise is that this *could* fail one day). | Cite it explicitly in the B24 suite docstring and add it to the non-movement matrix. |
| T8 | collateral | The collateral list omits `tests/decode_study.py` and `tests/decode_identity_completeness_study.py` / `tests/crop_identity_study.py` (studies that compose runtime versions live — verify they still run, no literal pins). | Add to the "verify" column. |

The collateral literal-pin list (§6.4 of the design) is otherwise accurate —
cross-checked every `a97ccb68` occurrence: `backend/config.py:41`,
`test_decode_identity_completeness.py:341,364`, `test_crop_identity.py:63,291`
(+ history at `:54,299`), `test_video_selection_identity.py:77,341,413`,
`test_startup_alignment.py:76,98,125,211`, `test_run_identity.py:427` (field
list). Synthetic decode constants in `test_evaluation_identity.py` /
`test_run_identity.py` (`c20a064a`, `UNDER_FFMPEG`, `UNDER_MSMF`) are **not**
live pins — correctly left alone.

---

## 8. Independent open-finding re-rank

Axes 1–5 (5 = worst). **Repro** / **Cache** (cache-corruption risk) / **Deploy**
(risk introduced by the fix) / **Migr** / **Cx**. Not reusing the design's
numbers.

| Finding | Repro | Cache | Deploy | Migr | Cx | Actionable now | My rank / note |
|---|---|---|---|---|---|---|---|
| **R13** face-crop instability (310×, 25.7% — measured) | 5 | 4 | 2 | 5 | 5 | ❌ SCRFD | Most severe; un-batchable (fix = SCRFD, a standing constraint). |
| **N5** no lockfile (`opencv-python`/`numpy` ranges; `numpy` in **no** identity) | 3 | 2 | 3 | 2 | 2 | ✅ | **#1 actionable.** Cheapest; no cache generation; root cause of R30/D9/D10 for the wheel path; closes the numpy gap the `pyproject` comment flags. |
| **R31** witness digest omits `effective_model_version`/gate/`.f.`/`input_size` | 3 | 3 | 3 | 3 | 3 | ✅ (no live exposure) | **#2.** Real gap; `CALIBRATION_WITNESS_VERSION` 1.3.0 migration path already exists (schema-version-as-parameter). No model to calibrate yet → B25 with real-model onboarding. Fold N6/N7. |
| **OR1-b** *(scheduled; not in the brief list but adjacent)* | 1 | 2 | 3 | 1 | 1 | ✅ | **#3.** Zero cache generation; closes the config-default alignment gap; B24's literal move makes it timely. |
| **R30** image codec libs outside `decode_identity` | 2 (wheel) / 4 (system-linked) | 2 / 4 | 3 | 2 | 3 | ✅ | **#4.** D9-tier for the declared wheel deployment (§2); no measured divergence; forces an R34 deploy. Worth doing **bundled with D8/D9/D10** (one cache generation for the cluster), after N5. |
| **D9** `cv2.__version__` drops the wheel's 4th digit | 2 | 2 | 2 | 2 | 2 | ✅ (with R30 or N5) | Entangled with R30 — the D9 case *is* R30's realistic vector for the wheel. N5 (exact pin) neutralises it more cheaply. |
| **D8** SIMD path not in identity | 2 | 2 | 2 | 2 | 3 | ✅ (with R30) | If R30 lands, capture libjpeg-turbo SIMD **in `digest`** (§3.B), not identity. The swscale (video) half stays a recorded residual. |
| **D10** same version string, different build | 2 | 2 | 1 | 4 | 5 | ❌ | Un-seeable by a version string; needs a binary hash. Record only. |
| **R32** `batch_size` neutrality backend-dependent | 2 | 2 | 1 | 3 | 3 | ❌ (ONNX/attention) | Not a defect today; double-tripwired post-B23-cleanup. |
| **R34** OR1 non-rolling-deployable without coordination | 1 | 1 | 3 | 1 | 1 | tracked | Mitigated (runbook). **Applies to any `.d.`/`.f.` batch incl. B24.** |
| **N8** one decode identity is a union over both decode paths | 1 | 1 | 1 | 5 (Option C) | 4 | ⚠️ blocked | §4: close as accepted architectural cost (like R9) or track-blocked (like R13). Not "narrow". |
| **R24** whole-frame square resize (every video bundle) | 2 | 1 | 2 | 4 | 4 | ❌ video face stage | Deterministic distortion → correctness, not reproducibility. |
| **R28** `resize` predicate area-filters an enlarged axis (0.040/255) | 2 | 2 | 2 | 2 | 2 | ⚠️ rides a pixel-moving batch | Trigger = SCRFD / R23 / R16 (not "next `.f.`" — B23 moved `.f.` and skipped it). |
| **R23** correctness halves (13% border stretch, live) | 2 | 1 | 2 | 4 | 4 | ❌ SCRFD | Deterministic; every fix moves pixels, unvalidatable without a model. |
| **R18** capability half (faceless clip → decided verdict) | 1 | 1 | 1 | 4 | 4 | ⚠️ needs a `face_stage` signal | Also blocks the R21 headline-rule tightening. |
| **R26** published bbox ≠ scored region | 1 | 1 | 1 | 3 | 3 | ⚠️ partly SCRFD; touches `EvidenceBundle` | Forensic. |
| **R16** one face's verdict published as the media's | 1 | 1 | 1 | 2 (prose) / 5 (per-face) | 2 / 5 | ⚠️ product decision; per-face touches `EvidenceBundle` | Prose path is Low value. |
| **N6 / N7** fit-optimiser config / `np.quantile` method unrecorded | 2 | 1 | 1 | 3 | 2 | ✅ with R31 | Fold into the witness-hardening batch. |
| **N3** advisory heuristic constants | 1 | 1 | 1 | 1 | 1 | defer | Add tripwires when one is next edited. |

### 8.1 Is B24 (R30, exactly as designed) still the right next batch? — **No, not exactly.**

R30 is **not** the highest-value actionable finding once the deployment model is
taken into account. **N5 (lockfile)** is cheaper, needs no cache generation,
attacks the root cause, and de-risks R30 to the point where R30 is best done
later as part of a **decode-environment completeness pass 2** (R30 + D8 + D9 +
D10 residual, one cache generation for the cluster). **OR1-b** (already
scheduled) is also higher-value-per-cost than R30 alone.

The design's *technical work is not wasted* — Option A, the migration profile,
and the Option-C rejection are all correct and carry forward to that later
batch. What is wrong is the **priority claim** and two **membership decisions**.

---

## 9. Verdict and defects

## **APPROVE WITH CHANGES**

Not REJECT: the finding is real, the architecture analysis is sound, the
migration profile is correct, and the technical design (Option A, parser-free,
no version bumps) carries forward. Not APPROVE: the risk framing is overstated
for the declared deployment, two membership decisions are wrong on the merits,
N8's disposition is mislabelled, and the test plan is missing its
output-identical proof.

### Architecture defects

**None.** Option A is entirely in-architecture; Option C is correctly rejected;
no new edges, no cycle, no worker/API asymmetry, parser untouched.

### Specification defects

| # | Severity | Defect | Evidence | Impact | Required correction |
|---|---|---|---|---|---|
| S1 | **High** | R30's risk/priority is overstated. For `opencv-python>=4.8,<5` (all codecs bundled, §1.9), a codec change ships with a `cv2.__version__` bump that `opencv_version` already catches; the residual is the **D9** 4th-digit case. No measured divergence; `test_opencv_and_pillow_decode_jpeg_identically` passes. | §2.1 table; §1.9; `test_decode_determinism.py:447`; `pyproject.toml:26` | A cache generation + R34 deploy spent on a D9-tier gain, ranked as a flagship. | Re-derive R30's severity per deployment provenance; **either** defer R30 behind **N5** (lockfile — no cache generation, root cause), **or** rescope B24 as "decode-environment completeness pass 2" bundling R30 + D8 + D9 + D10-residual to spend the one cache generation on the cluster. Correct §2 and §7.1. |
| S2 | **High** | SIMD flag placed in `identity` (cache key). | §3.B; `decode_env.py:211-221` (membership rule); libjpeg-turbo 2.x SIMD/C bit-exactness for `JDCT_ISLOW`; `cpu_dispatch` precedent (in `digest` only). | Fragments a mixed-SIMD-wheel fleet for a difference libjpeg-turbo guarantees does not exist. | Put the SIMD flag in **`digest`**, not `identity` — the `cpu_dispatch` treatment. Reverse design §8.1. |
| S3 | **Medium** | zlib placed in `identity`. | §3.C; RFC 1951/1950 bit-exactness of `inflate`; the identity's own "OUT if it cannot change a pixel" rule. | Cache fragmentation on a provably pixel-neutral axis. | **Exclude zlib** from `identity` (optionally record in `digest`). Capture **libpng** version — that carries the PNG risk. Reverse design §8.2. |
| S4 | **Medium** | N8 called "narrowed". | §4; `engine_registry.py:80-88` (both engines always); the union is *doubled and symmetrised*, not narrowed (the design's own §3 admits this). | An architecturally-unreachable finding kept "open" as if incremental progress is possible. | Reclassify N8: **close as accepted architectural cost** (cross-ref `RUN_IDENTITY.md` §3, B20/B23 precedent, like R9) **or** track-blocked like R13. Not "narrow", not "promote", not "split". |
| S5 | **Low** | §8.3 parser instruction is correct but the design still shows a component-parsed `image_codec` string in §2.6. | §1.9 (WebP ABI, system-path variants, absent SIMD lines). | Parser breakage on non-Windows / non-wheel builds. | Keep §8.3; make §2.6's example a *verbatim-line* capture, and pin `<new>` + all collateral literals on the target build in one commit. |

### Documentation defects

| # | Severity | Defect | Correction |
|---|---|---|---|
| D1 | Low | §2 Option D credited with "operational visibility Option A lacks". `differences()` (`decode_env.py:284-307`) already reports field-by-field under A. | State D and A are equivalent for `differences()`; D's only edge is a future Option-C seam. |
| D2 | Low | §5 "cross-B24 benchmark comparison refuses (correct)". Two runs on byte-identical pixels become incomparable — a **false** refusal (over-conservative), an accepted cost, not a correctness gain. | Reword to "refuses — a conservative false incomparability, the accepted cost of value-versioning the identity (B9/B20 precedent)". |
| D3 | Low | The `digest` (`4aded179`) also moves; the design does not mention it or confirm nothing pins it. | Add a sentence: digest moves with identity; no literal pins it (`grep` verified); no migration impact (not persisted in production). |

### Test defects

| # | Severity | Defect | Correction |
|---|---|---|---|
| T-A | **Medium** | No output-identical proof (B22/B23 shipped one each). | Add `test_the_shipped_image_bundle_did_not_move` + `…_video_…` — real bundles, `p_fake`/`verdict` unchanged, only `model.version` moves. |
| T-B | Medium | FB1 is a source grep, not behavioural. | Recast as `mismatched_image_codec(...)` → identities equal on the pre-B24 tree. |
| T-C | Medium | PA10 ("parser defensive") un-implementable against an `lru_cache`d parser. | Specify a pure `_parse_image_codecs(info)` helper the test drives directly. |
| T-D | Low | No `test_batch_24_did_not_touch_the_frame_count_or_video_ffmpeg_libs`. | Add it (B20 precedent). |
| T-E | Low | If S2/S3 land, the SIMD and zlib PA rows must become **digest-exclusion** tests (mirror `test_the_cpu_fields_are_still_excluded_from_the_identity`). | Re-scope those rows. |
| T-F | Low | `test_opencv_and_pillow_decode_jpeg_identically` (the measurement guard) not cited in the plan. | Add to the non-movement matrix + suite docstring. |

### If the reviewer disagrees with S1

A reviewer who accepts R30 as R27-tier (e.g. because production may use a
system-linked OpenCV the `pyproject` does not pin) would still need S2, S3, S4
and the test additions. In that case the verdict remains **APPROVE WITH
CHANGES**, with S1 downgraded to a documentation clarification ("R30's severity
is deployment-provenance-conditional; state which provenance B24 assumes").
