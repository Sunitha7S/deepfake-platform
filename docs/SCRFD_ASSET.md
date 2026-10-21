# SCRFD Asset — Fetch, Verification, and License Record

**Status: Phase 0 (asset acquisition) has not been executed.** This
document records the *procedure* — decided and ratified in `docs/
SCRFD_ASSET_AUDIT.md` and `docs/SCRFD_PHASE2_SPEC_RESOLUTION.md` — for
whoever runs it. **No asset has been downloaded into this repository or
any environment it controls as part of implementing Phase 2** (`docs/
SCRFD_PHASE2_SPEC_RESOLUTION.md`'s own Gate, §12, is explicit that a human
must read and consciously accept the license below *before* any fetch runs
against an uncontrolled machine — that acceptance has not happened in this
session, so the fetch has deliberately not been performed either).
`SCRFDDetector`, `scrfd_asset_digest()`, and the `DFP_SCRFD_MODEL_PATH`
override all work against *any* correctly-shaped asset placed at the
configured path — nothing in Phase 2's code depends on this document's
hash fields being filled in yet.

---

## 1. What to fetch

**Source (Decision 2, `docs/SCRFD_PHASE2_SPEC_RESOLUTION.md`):**
`buffalo_l.zip` from InsightFace's official GitHub Release —

```
https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip
```

Measured (live, `docs/SCRFD_ASSET_AUDIT.md` §1.3) at **288.62 MB**, bundling
an `SCRFD-10GF` detector (Decision 1's ratified variant, `SCRFD_10G_KPS`)
alongside recognition/alignment/attribute models this platform has no use
for. **Extract only the detection member** — community convention names it
`det_10g.onnx`; **this name is unverified against an actually-unzipped
archive** (`docs/SCRFD_ASSET_AUDIT.md` §7 item 2) and must be confirmed,
not assumed, the first time this procedure is actually run. Discard every
other file in the archive.

## 2. License — read this before fetching anything

Verified from three independent primary sources this program fetched
directly (`docs/SCRFD_ASSET_AUDIT.md` §2.2), quoted verbatim:

> **ALL models are available for non-commercial research purposes only.**
> — `model_zoo/README.md`, `deepinsight/insightface`

> The InsightFace Python library code is released under the MIT License
> and can be used for academic and commercial purposes. **The pretrained
> models provided with this library are available for non-commercial
> research purposes only, including both automatically and manually
> downloaded models.**
> — `python-package/docs/model_zoo.md`

This applies to `buffalo_l.zip` and the `SCRFD-10GF` detector inside it,
without exception or carve-out. **This repository has no license file and
no commercial-distribution intent today** (`docs/SCRFD_ASSET_AUDIT.md`
§2.5) — squarely inside the grant above — but this must be re-read, not
re-assumed, the moment that changes. Do not fetch this asset into any
shared, commercial, or otherwise uncontrolled environment without
independently re-confirming this text is still current for whatever
release is fetched.

## 3. Fetch and verification procedure (trust-on-first-use, Decision 3)

There is **no independently-published checksum** for this asset — GitHub's
Releases API does not carry a `digest` field for it (checked live,
`docs/SCRFD_ASSET_AUDIT.md` §2 "Decision 3" derivation). Verification is
therefore trust-on-first-use, not verification against an upstream
authority:

1. Download `buffalo_l.zip` from the URL in §1. Confirm its size is
   288.62 MB (or record the actual size if the upstream release has since
   changed — re-verify §1's numbers were not silently superseded).
2. Compute and record its SHA-256:
   ```
   buffalo_l.zip sha256: <NOT YET RECORDED — run `sha256sum buffalo_l.zip`
   at fetch time and paste the result here>
   ```
3. Unzip and confirm the detection member's actual file name (§1's
   `det_10g.onnx` is unverified). Extract it to the path
   `$DFP_SCRFD_MODEL_PATH` will point at.
4. Compute and record the **extracted member's** own SHA-256 — this is
   what `scrfd_asset_digest()` (`inference/preprocess.py`) actually hashes,
   not the zip:
   ```
   <detection member file name> sha256: <NOT YET RECORDED>
   ```
5. Set `DFP_SCRFD_MODEL_PATH` to the extracted file's path. Every
   subsequent `scrfd_asset_digest($DFP_SCRFD_MODEL_PATH)` call detects
   drift from *this recorded value*, not from an independent upstream
   source — a residual, named risk (`docs/SCRFD_PHASE2_SPEC_RESOLUTION.md`
   §Risk register), not a gap this procedure can close further.

## 4. Configuration — how the platform finds the asset

`DFP_SCRFD_MODEL_PATH` (an environment variable, mirroring the existing
`$DFP_MODEL_VERSION` pattern in `workers/runner.py` — `docs/
SCRFD_PHASE2_SPEC_RESOLUTION.md` Decision 4) — read directly by
`inference.preprocess._scrfd_detector_for` the moment a policy names an
SCRFD detector. **Not read anywhere in `backend/` or `workers/`
composition roots in Phase 2** — nothing production-facing constructs an
SCRFD-selecting policy yet (`default_face_selection()` stays Haar). Wiring
this into a real deployment's startup path is Phase 4 work.

A user or operator who wants to supply a **different**, differently
licensed model may point `DFP_SCRFD_MODEL_PATH` at it directly — nothing
in `SCRFDDetector` or `FaceSelectionPolicy` assumes this specific asset;
only `detector_asset`'s digest (computed from whatever file is actually
there) reaches any identity string. This mirrors InsightFace's own
documented escape hatch (`python-package/docs/model_zoo.md`, "Use your own
licensed model").

## 5. Variant

`SCRFD_10G_KPS` (Decision 1) — chosen for its 5-point landmark output
(needed for `docs/CROP_GEOMETRY.md`'s Option F destination) and its
accuracy tier, contingent on Phase 3's own latency measurement possibly
preferring `SCRFD_2.5G_KPS` instead. Re-running Phase 3 against the
smaller variant is a parameter change to `tests/scrfd_detector_study.py`,
not new infrastructure.

## 6. Open items

- §1's extracted-member file name is unverified.
- §3's checksums are not yet recorded — this procedure has not been run.
- No independent upstream checksum exists to cross-verify against (§3) —
  a standing, accepted residual risk, not something a future fetch can fix.
