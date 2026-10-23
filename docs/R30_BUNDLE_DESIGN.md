# R30 Bundle — Design (Phase 3 / Batch 26)

**Date:** 2026-09-17. **Type:** design only — no code. **Method:** complete
source re-derivation. Every claim below was re-checked against the current
working tree (`git status` read first; `HEAD` = `8676fdd` "R31: bind
calibration witness to model configuration", plus uncommitted OR1-b changes
— see §0) rather than assumed from `docs/B24_DESIGN.md`,
`docs/B24_ADVERSARIAL_REVIEW.md`, or `docs/B24_R2_DESIGN.md`. Those three
documents are **inputs to be verified, not premises** — §1 records what was
re-checked and what, if anything, had drifted.

**Scope:** R30 (image decode libraries outside `decode_identity`) bundled
with its recorded residuals D8 (SIMD not in identity), D9 (`cv2.__version__`
drops the wheel's 4th digit), D10 (same version string, different build) —
the "decode-environment completeness pass 2" batch named in
`docs/B24_R2_DESIGN.md` §7.3 and scheduled as Phase 3 in
`docs/PROJECT_ROADMAP.md`.

---

## 0. Repository state at the moment of this design

Read directly, not recalled:

- `git log --oneline`: one commit, `8676fdd` — R31. `git remote -v`: none.
- **Uncommitted in the working tree:** OR1-b (Batch 25) — modified
  `backend/config.py`, `workers/runner.py`, `REPRODUCIBILITY_AUDIT.md`,
  `docs/VIDEO_SELECTION_IDENTITY.md`, `tests/run_all.py`,
  `tests/test_startup_alignment.py`, `tests/test_video_selection_identity.py`;
  new `backend/worker_main.py`, `docs/OR1B_DESIGN.md`,
  `docs/OR1B_SPEC_RESOLUTION.md`, `docs/PROJECT_ROADMAP.md`,
  `tests/test_worker_main.py`. This bundle's diff is written **against that
  working tree**, i.e. it assumes OR1-b lands (as a commit) before or
  alongside this batch. §1.3 confirms OR1-b is orthogonal to R30 and creates
  no line-number or logic conflict.
- `tests/run_all.py` `SUITES` currently lists **52 entries** (verified by
  counting the literal list, lines 73–124 — an initial grep-based count in
  this session undercounted to 51 by missing the `("D-3 result sink race",
  ...)` row, whose label does not start with `"S`; corrected here), through
  `("S8-B25 worker composition root (OR1-b)", test_worker_main)`. This batch
  is therefore implementation-**Batch 26** (continuing the audit's own
  batch-numbering sequence, which is not 1:1 with suite count — several
  suites predate the "S8-B*" scheme), registering test suite **53**.
- Full-suite re-run (`python tests/run_all.py`, this session, current tree):
  **completed — `OK - 52 suites, 1292 tests passed`, exit 0.** Confirmed
  green baseline, not assumed; this includes `test_evaluation_operating`,
  the suite documented to take 5–15 minutes.
- `STUB_ENGINE_VERSION` (`backend/config.py:41`, live-read):
  `"stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.a97ccb68"` — matches
  every prior document's assumption exactly. **Unmoved by N5, R31, or OR1-b.**
- `CALIBRATION_WITNESS_VERSION` (`evaluation/calibration_protocol.py:74`,
  live-read): `"1.3.0"` (moved by R31 from 1.2.0). `uv.lock` and
  `pyproject.toml`'s `[tool.uv]` override block (live-read) exist and pin
  `opencv-python==4.8.1.78` / `numpy==1.26.4` (N5, closed).

---

## 1. Re-verification of R30/D8/D9/D10 after N5, R31, OR1-b

The user's instruction was to validate these findings **remain applicable**,
not to re-litigate their content from scratch where the prior re-derivation
(`docs/B24_R2_DESIGN.md` §1, itself a from-source re-derivation, not a
recollection) is still standing. Each premise was independently re-checked
against the current tree.

### 1.1 Does N5 change anything R30 depends on?

**No, and it is a precondition, not a risk.** `docs/B24_R2_DESIGN.md` §7.3
required N5 to land first so a future `.d.` move is computed against a
reproducible dependency resolution rather than an ad hoc machine's. Live-read
confirmation: `uv.lock` exists (committed under the R31 commit's tree —
`git ls-files uv.lock` finds it), `pyproject.toml:96-101` pins
`opencv-python==4.8.1.78` / `numpy==1.26.4` — the **exact** versions the
shipped `a97ccb68` was computed against (`docs/B24_N5_DESIGN.md` G1 gate).
**Precondition satisfied.**

### 1.2 Does R31 change anything R30 depends on?

**No — independently re-verified from the current
`evaluation/calibration_protocol.py`, not assumed from
`docs/R31_DESIGN.md`.** Read live (`calibration_protocol.py:280-349`):
`compute_witness`'s digest treats `decode_identity` as an **opaque string
parameter**, gated by `schema_version in SCHEMA_VERSIONS_WITH_DECODE`
(`{"1.2.0", "1.3.0"}`, confirmed both present after R31's edit). Nothing about
*how* `decode_identity()` computes its return value is inspected by this
function — it only hashes whatever string is handed to it. R30 changes what
`decode_identity()` returns; it does not touch this call's shape. **The
witness-compatibility conclusion of `docs/B24_R2_DESIGN.md` §5.7 (no
`CALIBRATION_WITNESS_VERSION` bump needed for R30) holds under 1.3.0 exactly
as it held under 1.2.0** — re-derived independently here, not carried over
by assumption. (R31 also added `pre_calibration_identity` on a disjoint
frozenset, `SCHEMA_VERSIONS_WITH_PRE_CALIBRATION = {"1.3.0"}` — a parallel,
non-interacting axis.)

### 1.3 Does OR1-b change anything R30 depends on?

**No overlap in code; one collateral file gained new pins to account for.**
OR1-b's diff (live-read via `git diff`) touches `backend/config.py` only in
the `Settings` class body (adds `skip_version_alignment: bool`, well below
the `STUB_ENGINE_VERSION` literal at line 41) and `workers/runner.py`'s
`verify_engine_alignment_at_startup` / `run_with_shutdown` signatures — none
of `inference/decode_env.py`, `backend/service.py`, or the `.d.` literal
itself. **No line-number or logic conflict with this batch's production
changes.**

**Collateral drift found and corrected here (this is exactly the kind of
gap a memory-based re-derivation would have missed):** OR1-b's changes to
`tests/test_startup_alignment.py` added **82 lines**, including three
*additional* occurrences of the `.d.a97ccb68` literal beyond the four
`docs/B24_R2_DESIGN.md` §6.5 enumerated. Live grep, current tree:

```
tests/test_startup_alignment.py:76    stale = "...f.00000000.d.a97ccb68"
tests/test_startup_alignment.py:98    stale = "...f.00000000.d.a97ccb68"
tests/test_startup_alignment.py:125   stale = "...f.00000000.d.a97ccb68"
tests/test_startup_alignment.py:211   malformed = "...f..d.a97ccb68"
tests/test_startup_alignment.py:238   stale = "...f.00000000.d.a97ccb68"   ← new (OR1-b)
tests/test_startup_alignment.py:255   stale = "...f.00000000.d.a97ccb68"   ← new (OR1-b)
tests/test_startup_alignment.py:276   agreeing_but_wrong = "...f.11111111.d.a97ccb68"  ← new (OR1-b)
```

§4.2 below carries all seven, not the four the pre-OR1-b design doc knew
about. Every other collateral-matrix line number in
`docs/B24_R2_DESIGN.md` §6.5 was independently re-verified this session
(`tests/test_decode_identity_completeness.py:341,364`;
`tests/test_crop_identity.py:63,291,299-300`;
`tests/test_video_selection_identity.py:77,341,413`;
`tests/test_run_identity.py:427-429`) and **matched exactly** — zero drift
on those files.

### 1.4 Do the underlying technical findings (S1–S4 of `docs/B24_ADVERSARIAL_REVIEW.md`, re-confirmed in `docs/B24_R2_DESIGN.md` §1) still hold?

Re-checked live, this session, independent of the old measurement log:

- `pyproject.toml:19` still declares `"opencv-python>=4.8,<5"` (range) with
  `[tool.uv].override-dependencies` pinning the resolved version — the wheel
  deployment premise (§1's S1) is unchanged.
- `cv2.getBuildInformation()` **read live on this box, this session**:

  ```
  ZLib:      build (ver 1.2.13)
  JPEG:      build-libjpeg-turbo (ver 2.1.3-62)     SIMD Support: NO
  WEBP:      build (ver encoder: 0x020f)
  PNG:       build (ver 1.6.37)
  ```

  — every codec still reports `build` (bundled into the wheel), matching
  §1's premise exactly. `cv2.__version__ == "4.8.1"` while the installed
  wheel is `4.8.1.78` (D9) — still true, still unparsed by `opencv_version`.
- `inference/decode_env.py` (re-read in full this session, §2 below) is
  **structurally unchanged** since the R2 design was written — same
  `DecodeEnvironment` dataclass, same `identity`/`digest` property split,
  same `_build_facts()` shape, same `_VIDEO_FFMPEG_LIBS` tuple. No batch
  between B23 and R31 touched this module.
- `ALLOWED_IMAGE_MIME` (`contracts/api.py:33`, live-read):
  `("image/jpeg", "image/png", "image/webp")` — unchanged; the PA2 test
  premise ("`_IMAGE_CODEC_LIBS` is exactly the codecs behind
  `ALLOWED_IMAGE_MIME`") still holds with no TIFF/JPEG2000/OpenEXR to worry
  about.
- The prior cross-build measurement (OpenCV vs. Pillow decoding JPEG/PNG/WebP
  bit-identically across a libjpeg-turbo major-version gap and stock-zlib vs.
  zlib-ng) was **not** re-run this session — it is an external-library
  behavioural fact, independent of this repository's own commits, and
  nothing in N5/R31/OR1-b could have changed it. Re-running it is optional
  defense-in-depth, not required to validate applicability.

**Conclusion: R30/D8/D9/D10 remain fully applicable, unmodified in
substance, after N5, R31, and OR1-b.** The only real update required is the
three extra `test_startup_alignment.py` pins (§1.3, carried into §4.2/§4.4).
The identity design (§3 of `docs/B24_R2_DESIGN.md`), the migration profile
(§5), and the test plan (§6) are re-adopted below with file/line references
re-verified against the current tree, and the one item that document itself
left open (C2 — parser robustness) is resolved in §6.

---

## 2. Design — identity membership (re-adopted, re-verified)

No change from `docs/B24_R2_DESIGN.md` §3's matrix; restated with the
current-tree verification behind it (§1.4):

| Candidate | `identity` (cache key) | `digest` (forensic) | Why |
|---|:---:|:---:|---|
| libjpeg-turbo version | **IN** | via identity | affects pixels *in principle* (12-bit/CMYK/arithmetic, none accepted); measured neutral for accepted 8-bit RGB JPEG across a major version gap; coarsest reliable signal, identity is deliberately over-sensitive by design (`decode_env.py:206-209`) |
| libpng version | **IN** | via identity | same reasoning; carries the PNG pixel risk (not zlib) |
| libwebp version | **IN** | via identity | same reasoning; codec on the accepted path |
| libjpeg-turbo SIMD flag | **OUT** | **IN** | performance only — `JDCT_ISLOW` (OpenCV's method) is bit-exact across C/SIMD; `cpu_dispatch` precedent |
| zlib version | **OUT** | optional | performance only — `inflate()` is RFC-1951-deterministic |
| TIFF / JPEG2000 / OpenEXR | OUT | OUT | unreachable — outside `ALLOWED_IMAGE_MIME`, guarded by PA2 |

`_IMAGE_CODEC_LIBS = ("JPEG", "PNG", "WEBP")`.

### 2.1 Parser — verified against this box's live build string, this session

```python
_IMAGE_CODEC_RE = re.compile(
    r"^\s*(" + "|".join(_IMAGE_CODEC_LIBS) + r"):\s*\S+\s*\(ver\s+([^)]+)\)", re.M
)
_SIMD_SUPPORT_RE = re.compile(r"^\s*SIMD Support:\s*(\S+)", re.M)
```

Run live against `cv2.getBuildInformation()` on this box, this session:

```
{'JPEG': '2.1.3-62', 'WEBP': 'encoder: 0x020f', 'PNG': '1.6.37'}
simd: NO
```

Values captured **verbatim** (`docs/B24_DESIGN.md` §8.3 stands: no component
parsing — a WebP `"encoder: 0x020f"` or a hypothetical
`"decoder: 0x0209, encoder: 0x020f"` is stored as-is, never split). This
matches the design's `_parse_image_codecs`/`PA7` contract in
`docs/B24_R2_DESIGN.md` §6.4 exactly, now confirmed against a real string
rather than a hypothetical one.

### 2.2 C2 — ratified here (the one item `docs/B24_R2_DESIGN.md` §8 left open "for when the pass-2 batch is scheduled")

**Decision: keep the regex-based parser, defensive by construction, no
change in shape.** Rationale, re-derived rather than deferred again:

- The parser must not raise on an unrecognized `getBuildInformation()` layout
  — a missing diagnostic must not stop a decode (the same principle
  `_build_facts()` already follows: `except Exception: return None, "unknown",
  "unknown"`). `_parse_image_codecs` follows the identical contract: any
  library absent from `found` is simply missing from the joined string; a
  `getBuildInformation()` that raises is caught the same way `_build_facts`
  already catches it.
- Cross-platform/cross-wheel format drift (WEBP `ver decoder: 0x…, encoder:
  0x…` vs. this box's encoder-only string; a system-linked build's differently
  indented "Media I/O" block) is handled by capturing **everything inside the
  parentheses verbatim** — the regex does not assume a fixed token count or
  order inside `(...)`, only that a `(ver ...)` group exists after the label.
  This is the same defensive posture `_AVLIB_RE`/`_build_facts` already use
  for the FFmpeg block, which has shipped across two prior batches (B19/B20)
  without a parser failure being reported.
- **Residual risk, stated honestly (this is what C2 was really asking for):**
  a `getBuildInformation()` whose "Media I/O" section omits the `(ver ...)`
  parenthesis entirely for a library that *is* present (a hypothetical
  non-standard OpenCV build) would silently omit that library from
  `image_codec_version` rather than raising — identical in kind to the
  existing fallback ladder in `_build_facts()` (`elif _AVCODEC_RE.search(info)`
  → `elif _FFMPEG_RE.search(info)` → `else: ffmpeg = None`). **No new fallback
  tier is added for image codecs** because, unlike the video path (R27's
  actual regression — a demuxer silently absent from a *pre-Batch-20* string),
  there is no historical evidence of an OpenCV build reporting `JPEG:`/`PNG:`/
  `WEBP:` without the `(ver ...)` suffix; PA7 pins the empty/absent case to
  `None`, not a partial-format case, which would need a real counter-example
  to design against rather than a speculative one. This is a **documented,
  test-pinned limitation**, not an unexamined gap — recorded as a new
  residual **D12** in the audit doc (§7 below) rather than silently accepted.

---

## 3. Production code changes

### 3.1 `inference/decode_env.py`

New sibling function, **not** a modification of `_build_facts()` — confirmed
necessary, not merely preferred, by a live grep this session:
`_build_facts` has **5 real call sites outside its own definition**
(`decode_environment()` in this file; `tests/decode_identity_completeness_study.py`;
`tests/frame_count_reproducibility_study.py`;
`tests/test_decode_identity_completeness.py` ×3;
`tests/test_frame_count_reproducibility.py` ×2, **one of which
(`test_the_decode_identity_now_covers_the_demuxer_batch_20`, line 363) asserts
`"_VIDEO_FFMPEG_LIBS" in inspect.getsource(_build_facts)`**). A sibling
function leaves every one of those five untouched by construction; editing
`_build_facts()`'s body would require re-verifying that assertion and
touching four test files instead of adding one.

```python
_IMAGE_CODEC_LIBS: tuple[str, ...] = ("JPEG", "PNG", "WEBP")
_IMAGE_CODEC_RE = re.compile(
    r"^\s*(" + "|".join(_IMAGE_CODEC_LIBS) + r"):\s*\S+\s*\(ver\s+([^)]+)\)", re.M
)
_SIMD_SUPPORT_RE = re.compile(r"^\s*SIMD Support:\s*(\S+)", re.M)


def _parse_image_codecs(info: str) -> tuple[str | None, str]:
    """``(image_codec_version, image_codec_simd)`` from a build-info string.

    Pure function (not ``lru_cache``d — PA7 drives it directly with a
    synthetic string). Verbatim capture, no component parsing — see R30 §2.1.
    """
    found = {lib: ver.strip() for lib, ver in _IMAGE_CODEC_RE.findall(info)}
    codecs = (
        " / ".join(f"{lib} {found[lib]}" for lib in _IMAGE_CODEC_LIBS if lib in found)
        or None
    )
    simd_match = _SIMD_SUPPORT_RE.search(info)
    simd = simd_match.group(1).strip() if simd_match else "unknown"
    return codecs, simd


@lru_cache(maxsize=1)
def _image_codec_facts() -> tuple[str | None, str]:
    """Cached wrapper around :func:`_parse_image_codecs`, the ``_build_facts``
    pattern (R30). Separate cache from ``_build_facts`` — same defensive
    try/except shape."""
    try:
        info = cv2.getBuildInformation()
    except Exception:  # pragma: no cover - no build info available
        return None, "unknown"
    return _parse_image_codecs(info)
```

`DecodeEnvironment` gains two fields:

```python
image_codec_version: str | None
image_codec_simd: str
```

`identity` property: add `"jpeg_png_webp": self.image_codec_version` to the
`fingerprint()` dict (§2 — `image_codec_simd` is **not** added here).
`digest` property: add `"image_codec_simd": self.image_codec_simd` alongside
the existing `cpu_baseline`/`cpu_dispatch` fields (digest is strictly finer
than identity, per the existing docstring contract). `to_dict()` and
`describe()` gain the two fields; `differences()` gains both field names in
its comparison tuple. `decode_environment()` calls `_image_codec_facts()`
alongside the existing `_build_facts()` call and threads the two new
constructor arguments through.

**No change to:** `_build_facts`, `_VIDEO_FFMPEG_LIBS`, `_AVLIB_RE`,
`resolve_video_backend`, `open_video`, `decode_fingerprint`,
`decode_identity` (the function — its *return value* changes because
`identity` now includes the new key, but its body is unchanged).

### 3.2 `backend/config.py`

One line: `STUB_ENGINE_VERSION`'s `.d.a97ccb68` tail → `.d.<new>`, where
`<new>` is `decode_identity()` recomputed on the implementation box **after**
§3.1 lands (must be measured live at implementation time, exactly as every
prior `.d.`/`.f.` batch did — never hand-computed). This is the **only**
production line outside `inference/decode_env.py` this batch touches. It
does not conflict with OR1-b's uncommitted edit to this file (§1.3 — that
edit is ~180 lines below, in the `Settings` class body).

---

## 4. Exact files affected

### 4.1 Production

| File | Change |
|---|---|
| `inference/decode_env.py` | + `_IMAGE_CODEC_LIBS`, `_IMAGE_CODEC_RE`, `_SIMD_SUPPORT_RE`, `_parse_image_codecs`, `_image_codec_facts`; `DecodeEnvironment` + 2 fields; `identity`/`digest`/`to_dict`/`describe`/`differences` updated; `decode_environment()` calls the new facts function |
| `backend/config.py` | `STUB_ENGINE_VERSION` `.d.` tail literal only (line 41 today) |

### 4.2 Test — new

| File | Purpose |
|---|---|
| `tests/test_image_decode_identity.py` | New suite, `S8-B26 image decode identity (R30 + D8/D9)`. Output-identical proof (OI1–OI3), non-movement proof (NM1–NM10), fail-before matrix (FB1–FB6), pass-after matrix (PA1–PA8, +PA9 for `image_codec_simd` digest-only membership), parser-defensiveness test (the C2 resolution, §2.2) |

### 4.3 Test — collateral pins to update

| File:line | Old | New |
|---|---|---|
| `tests/test_decode_identity_completeness.py:341` | `.endswith(".d.a97ccb68")` | `.endswith(".d.<new>")` |
| `tests/test_decode_identity_completeness.py:364` | `"a97ccb68" not in snap` | `+ "<new>" not in snap` |
| `tests/test_crop_identity.py:63` | `B23_STUB_ENGINE_VERSION = "...d.a97ccb68"` | `...d.<new>` |
| `tests/test_crop_identity.py:291` | `.endswith(".d.a97ccb68")` | `.endswith(".d.<new>")` |
| `tests/test_crop_identity.py:299-300` | history list | `+` row `"...f.e629642c.d.<new>"` |
| `tests/test_video_selection_identity.py:77` | `B23_STUB_ENGINE_VERSION` | `...d.<new>` |
| `tests/test_video_selection_identity.py:341` | `.endswith(".d.a97ccb68")` | `.endswith(".d.<new>")` |
| `tests/test_video_selection_identity.py:413` | `decode="a97ccb68"` | `decode=decode_identity()` |
| `tests/test_startup_alignment.py:76,98,125,238,255` | `...f.00000000.d.a97ccb68` | `...d.<new>` (cosmetic — the `.f.00000000` mismatch is the point of these fixtures, not the `.d.` tail) |
| `tests/test_startup_alignment.py:211` | `...f..d.a97ccb68` | `...f..d.<new>` |
| `tests/test_startup_alignment.py:276` | `...f.11111111.d.a97ccb68` | `...d.<new>` |
| `tests/test_run_identity.py:427-429` | forensic field tuple | `+ "image_codec_version", "image_codec_simd"` |
| `tests/run_all.py` | 52 suites | `+ import test_image_decode_identity`, `+` SUITES row (`"S8-B26 image decode identity (R30 + D8/D9)"`), 53 |

**History pins confirmed left alone (re-verified live this session, not
assumed):** `tests/test_crop_identity.py:54` (`B22_STUB_ENGINE_VERSION`, a
`…f.251d22e9…` history row, predates this axis entirely);
`tests/test_evaluation_identity.py` / `tests/test_run_identity.py` synthetic
decode constants (`c20a064a`, `UNDER_FFMPEG`, `UNDER_MSMF`) — none reference
`a97ccb68` and are unaffected by a `.d.` move.

### 4.4 Documentation

| File | Change |
|---|---|
| `docs/DECODE_IDENTITY_COMPLETENESS.md` | New section (parallel to the existing R27/video-path narrative) documenting the image-codec extension: what's in/out (§2 matrix), the parser (§2.1), the C2 resolution (§2.2), the new fields |
| `REPRODUCIBILITY_AUDIT.md` | R30 row: Open → Closed, full rewrite matching the B20/B23 row style; D9 row: Open → Closed (folded into R30); **D8 row: PARTIALLY closed** — image (libjpeg-turbo SIMD) half only, video (libswscale SIMD) half stays open (documented dual scope, `docs/B24_ADVERSARIAL_REVIEW.md`/`docs/B24_R2_DESIGN.md`: "swscale (video) SIMD stays a recorded residual"); D10 row: unchanged (record-only, no fix exists — a binary hash, not a version string, is needed); N8 row already reads "CLOSED — accepted architectural cost" (landed with OR1-b's uncommitted audit edit, §1.3) with a note that this batch enlarges the union symmetrically as predicted, not a new disposition; new **"Still open after Batch 26"** trailer section; new **D12** residual row (§2.2 — not D11, already taken locally by `docs/DECODE_IDENTITY_COMPLETENESS.md`'s own threat-model table for an unrelated VFR `frame_count` finding) |
| `docs/PROJECT_ROADMAP.md` | Phase 3 row: mark done; §1 table R30/D9 rows: Open → Closed; D8 row: PARTIALLY closed (image half only); D10 row: unchanged (open, no fix exists) |

---

## 5. Version-impact analysis

| Quantity | Moves? | Reason |
|---|---|---|
| `DecodeEnvironment.identity` (`.d.` preimage) | **Yes** | fingerprint dict gains one key (`jpeg_png_webp`) |
| `STUB_ENGINE_VERSION` `.d.` tail | **Yes** (`a97ccb68` → `<new>`, measured live at implementation time) | `compose_runtime_version(model_identity, decode_identity())` |
| `.f.` / `weights_identity` / `+c.` / `.g.` | No | no selection/weights/calibration/gate change |
| `model_identity_of(STUB_ENGINE_VERSION)` | No | strips `.d.` (`contracts/version.py`); decoder deliberately excluded from model identity |
| `CONTRACT_VERSION` (1.0.0) | No | `ModelInfo.version` is an unconstrained `str`; no wire field carries the decode identity |
| `RUN_SCHEMA_VERSION` (1.2.0) | No | `BenchmarkRun.decode_identity` stores whatever string it is given; schema shape unchanged |
| `CALIBRATION_WITNESS_VERSION` (**1.3.0**, current — R31 already moved it from 1.2.0) | No | §1.2 — digest hashes `decode_identity` as an opaque string under both `{1.2.0, 1.3.0}`; nothing inspects how the string was produced |
| `LOCK_VERSION` (split lock) | No | carries no decode/model identity (re-grepped this session: zero references) |
| OpenAPI signature / `schema_out` / `fixtures_out` | No | no wire field carries `.d.` or any codec-version string; `fixtures_out` fixtures carry `model.version="1.0.0-dev"` |
| `DecodeEnvironment.digest` | Moves (with `identity`) | expected — strictly finer than `identity`; no literal pins it |
| `pyproject.toml` / `uv.lock` (N5) | No | R30 adds no new third-party dependency — `cv2.getBuildInformation()` is already imported; nothing new to lock |

**Net: exactly one version literal moves (`.d.`), one cache generation, zero
contract/schema/witness/lock version bumps.** This is the B20/B23 profile,
re-confirmed rather than assumed.

---

## 6. Migration-impact analysis

### 6.1 Cache generations — one

`backend/service.py:98,226-227` (live-read, unchanged by N5/R31/OR1-b) keys
the content cache and idempotency key on `settings.model_version`, the one
literal carrying `.d.<D>`. One `.d.` move → one cache generation.

### 6.2 Cache namespace impact — both (N8, accepted cost, re-confirmed)

`workers/engine_registry.py:80-88` (live-read this session, unchanged)
unconditionally builds both `ImagePipeline` and `VideoPipeline` under the one
`STUB_ENGINE_VERSION` literal. This batch's `.d.` move regenerates **both**
content-cache namespaces in lockstep — the disposition already written into
the audit as "N8 — CLOSED, accepted architectural cost" (landed via OR1-b's
uncommitted audit edit, §1.3) explicitly anticipates this batch as the next
event that enlarges the union symmetrically. No new decision required here;
this batch is the predicted case, not a surprise.

### 6.3 Deploy coordination — R34 applies, unchanged mechanism

A `STUB_ENGINE_VERSION` move requires coordinated rollout (OR1's fail-closed
worker gate — now doubly enforced after OR1-b, §1.3: both the bare-worker
`$DFP_SKIP_VERSION_ALIGNMENT` path and the composition-root
`Settings.skip_version_alignment` path exist). Safe paths, unchanged from the
B23/B20 runbook: coordinated worker restart at cutover, or
`DFP_SKIP_VERSION_ALIGNMENT=true` (or `Settings.skip_version_alignment=true`
through `backend/worker_main.py`) for the rollout window — reduces to brief
false cache misses, never a wrong verdict. Set `$DFP_MODEL_VERSION` (or the
composition root's configured literal) to `<new>` in worker + API env at
cutover.

**Second-order effect (unchanged from `docs/B24_R2_DESIGN.md` §5.3):** two
benchmark runs straddling this batch become incomparable
(`evaluation/benchmark.py`'s `_decode_comparability` refuses two different
non-null decode identities) — a conservative false incomparability, not a
correctness gain. `REPRODUCIBILITY_AUDIT.md` must say exactly that, not
"correct" (the doc-hygiene point C3 from the original review, folded into
§4.4's rewrite instructions above).

### 6.4 Startup alignment — no change to the mechanism, re-verified against the OR1-b tree

`backend/app.py`'s lifespan format check
(`is_composed`/`is_environment_pinned`/round-trip) passes for
`…f.e629642c.d.<new>` — a clean hex suffix. `workers/runner.py`'s
`verify_engine_alignment_at_startup` (now, post-OR1-b, taking an optional
`skip_version_alignment` kwarg in addition to the env var, §1.3) still
compares against `verify_version_alignment(configured)` — matches iff the
collateral `backend/config.py` literal is updated in the same commit;
otherwise fails closed (correct, and now checked through two call paths
instead of one). `contracts/version.py`'s `_MODEL_MARKERS`/
`parse_runtime_version` never inspect the fingerprint's *value* — zero parser
impact, re-confirmed by reading the current parser this session (unchanged
since B23-R3).

### 6.5 Rollback

Revert `inference/decode_env.py` (drop the 2 fields + the 2 new module-level
helpers) and `backend/config.py` (`.d.` literal → `a97ccb68`). Reverted
engines recompute `a97ccb68`; the startup gate passes. Orphaned `…d.<new>`
cache entries expire under `result_ttl_s` (default 7 days, unchanged).
Partial rollback (code reverted, literal not, or vice versa) → worker refuses
to start (OR1, correct — now doubly enforced post-OR1-b). Benchmark/witness
rows written under `<new>` keep their string; a post-rollback comparison
against an `a97ccb68` row refuses (correct, `_decode_comparability`). No
corruption path in either direction.

### 6.6 Benchmark compatibility — no schema work

`BenchmarkRun.decode_identity: str | None` stores whatever string it is
given; `RUN_SCHEMA_VERSION` (1.2.0) does not move. A pre-batch row keeps its
recorded `a97ccb68`; `from_dict` reads it unchanged.

### 6.7 Witness compatibility — no `CALIBRATION_WITNESS_VERSION` bump

Independently re-verified in §1.2 against the **current** (post-R31, 1.3.0)
`compute_witness` — not carried over from the pre-R31 design doc. Every
stored witness (**none exist yet**, confirmed by the same "zero stored
witnesses" search R31's own design already performed and this session did
not need to repeat, since no witness-producing code has run outside test
fixtures) still verifies; a new witness records `<new>`.

---

## 7. Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| **Pixel-impact risk** — a codec version divergence actually does change a decoded pixel somewhere R30 doesn't measure | **Low** — §1.4 re-confirms the cross-build measurement premise is unchanged; the identity is deliberately over-sensitive (false miss costs compute, false hit costs a wrong verdict — the tie already breaks the safe way) | none needed beyond the existing design; this is the entire point of including the codec versions in `identity` even though no pixel divergence is expected |
| **N8 enlargement** — this batch symmetrically enlarges the single-literal union (image codec change now also regenerates the video cache namespace) | **Low, accepted, pre-decided** (§6.2) | already the documented disposition of N8; no new mitigation required, only confirmation that this batch is the anticipated trigger |
| **Deploy-coordination risk** — a `.d.` move without coordinated rollout fails workers closed mid-rollout | **Low** — OR1 (+OR1-b, now landed) already fail-closed by design; this is intended behavior, not a defect, and the runbook (R34) is unchanged | coordinated restart or the (now doubly-available) skip-alignment escape hatch for the cutover window |
| **Collateral-pin miss** — a `.d.a97ccb68` occurrence outside the enumerated list is missed, leaving a stale literal that either silently fails to update (cosmetic drift) or breaks a test | **Medium** — this design already found 3 such misses relative to the pre-OR1-b document (§1.3); a second grep must be run **at implementation time**, not trusted from this document, since the working tree will have changed again by then (OR1-b's own commit, plus whatever else lands first) | run `grep -rn "a97ccb68" --include="*.py"` (and doc equivalents) as the **first implementation step**, before writing `<new>`, and diff it against §4.3's table rather than assuming the table is exhaustive |
| **Parser fragility on a non-Windows/non-wheel build (C2)** | **Low-Medium, documented not fixed** (§2.2) | ratified as an accepted, test-pinned residual (new finding **D12**) rather than solved speculatively; revisit only if a real counter-example build is encountered |
| **Concurrent-session hazard** — this repo has a demonstrated history (memory: `orgchart-concurrent-sessions`-style hazard, and directly observed here: OR1-b work sitting uncommitted while this design was written) of two sessions editing the tree at once | **Medium** — directly relevant, not hypothetical, in *this* repository right now | re-read `git status` and `tests/run_all.py`'s `SUITES` length immediately before implementation begins, exactly as this design's own §0 did; do not assume Batch 26 numbering is still correct without re-checking |
| **Full-suite regression** — a full run was needed to establish the baseline before trusting this design | **None — resolved.** Full run completed during this design task: `OK - 52 suites, 1292 tests passed`, exit 0 | none needed; re-run once more immediately before implementation begins (§8 step 1) to catch any drift between this design and the moment coding starts |

---

## 8. Recommended implementation order

1. **Re-confirm the green baseline immediately before coding.** §0 already
   established `OK - 52 suites, 1292 tests passed` for the tree as it stood
   at design time; re-run `python tests/run_all.py` once more at the start
   of implementation to catch drift from anything landing in between
   (this repo has a demonstrated concurrent-editing hazard — §7). If any
   failure appears, check first whether it matches the known
   `frame_count`/machine-contention flakes documented in memory and
   `REPRODUCIBILITY_AUDIT.md` before treating it as a real regression.
2. **Re-run the collateral grep** (`a97ccb68`, plus `_build_facts`,
   `DecodeEnvironment(`, `decode_identity()`) against the *then-current* tree
   — not this document's snapshot — to catch any further drift from OR1-b
   landing or any other concurrent edit (§7's collateral-pin risk).
3. **Implement §3.1** (`inference/decode_env.py`): new module-level
   constants, `_parse_image_codecs`, `_image_codec_facts`, `DecodeEnvironment`
   field additions, property updates. Run the file's own module in isolation
   first (no other file depends on the new fields yet).
4. **Write the FB (fail-before) tests first** against the pre-change state if
   still reachable via `git stash`, or write them to assert the *intended*
   post-change behavior and confirm they fail against `HEAD` before applying
   step 3 — whichever ordering this session's tooling makes easier; the
   B20/B22/B23 convention is fail-before-then-pass-after as a **recorded
   matrix**, not necessarily as literal test-first authorship order.
5. **Write the new suite** `tests/test_image_decode_identity.py` (§4.2) —
   OI/NM/FB/PA matrix, parser test (C2 resolution).
6. **Measure `<new>`** live on the implementation box: import
   `inference.decode_env`, call `decode_identity()`, record the resulting
   hex. Never hand-compute.
7. **Update `backend/config.py`** (§3.2) with `<new>`.
8. **Update the collateral pins** (§4.3) — every row, plus whatever the
   step-2 re-grep found beyond this document's table.
9. **Register the new suite** in `tests/run_all.py` (§4.3 last row).
10. **Run the full suite.** Confirm the exact shape predicted in §5/§6: one
    `.d.` move, `.f.`/weights/gate/`+c.`/`.g.` unmoved,
    `CONTRACT_VERSION`/`RUN_SCHEMA_VERSION`/`CALIBRATION_WITNESS_VERSION`/
    `LOCK_VERSION`/OpenAPI all unmoved, output-byte-identical on the OI1–OI3
    fixtures.
11. **Update documentation** (§4.4): `docs/DECODE_IDENTITY_COMPLETENESS.md`
    new section, `REPRODUCIBILITY_AUDIT.md` R30/D8/D9/N8-confirmation/D12
    rows + new "Still open after Batch 26" trailer, `docs/PROJECT_ROADMAP.md`
    Phase 3 marked done.
12. **Do not commit OR1-b and this batch as one commit** unless the user
    directs otherwise — they are independently reviewable, independently
    revertible changes (§6.5's rollback story is written per-batch); OR1-b
    should land first (it is already implemented and green) so this batch's
    "one commit, `8676fdd`" baseline in §0 stays accurate for whoever reads
    this document next.

---

## 9. Gate

**Design status: ready for implementation**, pending only the user's approval
to write code (per instruction, no code was written in this task). No
blocking defects were found in re-deriving R30/D8/D9/D10's applicability
after N5/R31/OR1-b. One prior open item (C2) is ratified in §2.2 rather than
carried forward again. One new, honestly-scoped residual (**D12** — parser
silently omits a codec whose `getBuildInformation()` line lacks a
`(ver ...)` suffix, no known real-world occurrence) is recorded rather than
speculatively fixed.

**Stop here — no implementation until explicit approval, per instruction.**

**Status (2026-10-17): implemented as Batch 26.** `image_codec_version` /
`image_codec_simd` landed in `inference/decode_env.py`; `.d.` moved
`a97ccb68` → `c9d4025c`; `tests/test_image_decode_identity.py` (28 tests)
covers it and is green in `tests/run_all.py`. The gate line above is kept as
the record of when the design was approved, not as pending work.
