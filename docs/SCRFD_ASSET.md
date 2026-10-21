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

### Re-verified 2026-10-17 (supersedes §3's "no upstream checksum" statement)

Fetched live from `api.github.com/repos/deepinsight/insightface/releases/tags/model-zoo`
and the three license files on `raw.githubusercontent.com`:

- **License unchanged:** all three primary sources still read
  *non-commercial research purposes only* for every pretrained model,
  automatic or manual download. Library code remains MIT.
- **GitHub now publishes a SHA-256 for every release asset.** The
  trust-on-first-use limitation recorded below (§3, and
  `docs/SCRFD_PHASE2_SPEC_RESOLUTION.md` Decision 3) no longer applies to
  the archive itself:

  | Asset | Size | Upstream digest |
  |---|---|---|
  | `buffalo_l.zip` | 288.62 MB | `sha256:80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f` |
  | `cheetah_l.zip` | 211.54 MB | `sha256:df81a05801731af706836384124a35ea02430f27edd08023532e1bee21cabc8d` |

  A fetched archive must match its digest before anything is extracted.
- **A stronger alternative exists.** Upstream documents that `cheetah_l`'s
  face-detection ONNX is identical to `buffalo_l`'s, and V2 packages such as
  `cheetah_l` ship a `manifest.json` declaring each model file's SHA-256
  plus a signed `MODEL.LICENSE`. Fetching `cheetah_l.zip` instead gives an
  upstream-declared hash for the *extracted detector* too, not only the
  archive — closing the remaining trust-on-first-use gap. Same license.
- **Not a face detector:** `scrfd_person_2.5g.onnx` on the same release is
  a person detector. Do not substitute it.

### License acceptance record

```
Accepted by: project owner      Date: 2026-10-17
Environments: personal dev machine only; weights stored outside the repository
Use: local non-commercial research / portfolio evaluation — no commercial use,
     no redistribution of weights
Package fetched: cheetah_l.zip (official model-zoo GitHub release)
Package licence: MODEL.LICENSE cheetah_l-public-v1, grant "non-commercial"
                 (signed; signature not verifiable — no published key)
Detector member: det_10g.onnx
  sha256 5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91
  = the value in the package's upstream manifest.json (verified)
Archive digest: verified 2026-10-21 — cheetah_l.zip, 211,538,148 bytes,
  sha256 df81a05801731af706836384124a35ea02430f27edd08023532e1bee21cabc8d
  = GitHub's published release-asset digest. The det_10g.onnx, manifest.json
  and MODEL.LICENSE extracted from the verified archive are byte-identical to
  the copies used for the 2026-10-17 measurement (which had been read from a
  partial download and CRC32-checked).
```

**Unresolved licence-scope question:** the grant says "non-commercial" without
defining it. Publishing aggregate measurement results in a public portfolio is
consistent with it, but whether a portfolio used in a job search counts as
non-commercial is not stated by the licence text and is not resolved here.

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
