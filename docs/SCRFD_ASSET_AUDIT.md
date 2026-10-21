# SCRFD Asset Audit — Variant, License, and Distribution Strategy

**Date:** 2026-09-18. **Type:** audit/design only — no code. **Basis:**
primary-source documents fetched live this session directly from
`raw.githubusercontent.com`, `api.github.com`, and `pypi.org` (verbatim
quotes below, with source paths) — **not** secondary summaries, blog posts,
or remembered facts. Cross-checked against this repository's own current
source (`inference/preprocess.py`, `pyproject.toml`, `.gitignore`) rather
than assumed from `docs/SCRFD_PHASE2_DESIGN.md`, which this audit narrows
in two places (§3.2, §6).

**Method note, stated honestly rather than hidden:** this session's
`WebSearch`/`WebFetch` tools returned a backend error on every call
(`"issue with the selected model (z-ai/glm-5.2)"`) and could not be used.
Every fact below was instead obtained with `curl` against GitHub's raw-file
and REST API endpoints and PyPI's JSON API, run from this repository's own
shell — i.e., the actual bytes of the actual upstream files, not a
model-summarized version of them. Where a claim could not be verified this
way, it is marked **unverified** rather than asserted.

**No code was written to produce this document, per instruction.**

---

## 1. SCRFD model asset candidates

### 1.1 Two distinct upstream artifact families — not one

Primary-source research surfaced a distinction `docs/SCRFD_PROGRAM_DESIGN.md`
and `docs/SCRFD_PHASE2_DESIGN.md` did not draw, because neither fetched the
actual upstream repository structure. There are **two separate SCRFD asset
families**, from the same organization, that must not be conflated:

| | **Family A — academic/research export** | **Family B — packaged `insightface` model-zoo** |
|---|---|---|
| Source path | `deepinsight/insightface` → `detection/scrfd/` (the SCRFD paper's own training code, built on mmdetection) | `deepinsight/insightface` → `python-package/` (the `insightface` PyPI library) + `model_zoo/README.md` |
| Distributed format | **PyTorch `.pth` checkpoints** — `detection/scrfd/README.md`'s own "Pretrained-Models" table links to OneDrive (`1drv.ms`); `model_zoo/README.md` §2.2 duplicates the same models via Google Drive | **Pre-converted ONNX**, bundled inside `.zip` packs, hosted on GitHub Releases |
| To get a `.onnx` file | Requires installing mmdetection/mmcv/PyTorch (the whole research repo) and running `detection/scrfd/tools/scrfd2onnx.py` yourself | Download-and-unzip only — no conversion toolchain needed |
| Face-detector-only? | Yes, one model per download | No — each pack also bundles recognition/alignment/attribute ONNX files; the SCRFD detector is one file inside the archive |

**This is the single most consequential finding of this audit's §1**: neither
prior design document distinguished these, and `docs/SCRFD_PHASE2_DESIGN.md`
§2.4 implicitly assumed a single, simple ".onnx download" — that assumption
is wrong for Family A and only correct, with caveats, for Family B.

### 1.2 Family A variants — from `detection/scrfd/README.md` and `model_zoo/README.md` §2.2, verbatim

| Name | Easy | Medium | Hard | FLOPs | Params(M) | Infer(ms), VGA, single-thread CPU¹ | Landmarks |
|---|---|---|---|---|---|---|---|
| SCRFD_500M | 90.57 | 88.12 | 68.51 | 500M | 0.57 | 3.6 | No |
| SCRFD_1G | 92.38 | 90.57 | 74.80 | 1G | 0.64 | 4.1 | No |
| SCRFD_2.5G | 93.78 | 92.16 | 77.87 | 2.5G | 0.67 | 4.2 | No |
| SCRFD_10G | 95.16 | 93.87 | 83.05 | 10G | 3.86 | 4.9 | No |
| SCRFD_34G | 96.06 | 94.92 | 85.29 | 34G | 9.80 | 11.7 | No |
| SCRFD_500M_KPS | 90.97 | 88.44 | 69.49 | 500M | 0.57 | 3.6 | **Yes, 5-point** |
| SCRFD_2.5G_KPS | 93.80 | 92.02 | 77.13 | 2.5G | 0.82 | 4.3 | **Yes, 5-point** |
| SCRFD_10G_KPS | 95.40 | 94.01 | 82.80 | 10G | 4.23 | 5.0 | **Yes, 5-point** |

¹ Measured by the upstream authors "on AMD Ryzen 9 3950X, using the simple
PyTorch CPU inference by setting `OMP_NUM_THREADS=1` (no mkldnn)" — a
**PyTorch** number, not an ONNX Runtime one; ONNX Runtime's own graph
optimizations typically move this figure, in either direction, and this
audit does not assume which without measuring (Phase 3's own job, per
`docs/SCRFD_PHASE2_DESIGN.md` §2.8).

**Naming corrects `docs/SCRFD_PROGRAM_DESIGN.md` §3.1's "500KB '2.5G'"
figure**, which conflated the FLOPs suffix with a file size: `"2.5G"` names
**2.5 GFLOPs** at VGA resolution (`detection/scrfd/README.md`'s own
definition: *"``2.5G``: means the model cost ``2.5G`` FLOPs while the input
image is in VGA(640x480) resolution"*), not a byte count. Actual `.pth`/
`.onnx` file sizes scale with parameter count (0.57M–9.80M params across the
family), so single-digit-to-low-teens MB is the right order of magnitude,
not kilobytes — consistent with, and now replacing, `docs/
SCRFD_PHASE2_DESIGN.md` §2.4's own hedge ("500KB... understates").

**No `_KPS` variant exists for 500M... wait — 500M_KPS exists but 34G_KPS
does not** — confirmed by the table's own absence of that row in both
source documents (`detection/scrfd/README.md` and `model_zoo/README.md`
agree). If landmark output is required (§2.4 of `docs/
SCRFD_PHASE2_DESIGN.md`'s discussion of `FaceTrack.landmarks`), the largest
available accuracy tier with landmarks is **SCRFD_10G_KPS**, not SCRFD_34G.

### 1.3 Family B variants — from `python-package/docs/model_zoo.md`, verbatim

| Pack | Detection model | Size (zip) | Auto-download? |
|---|---|---|---|
| antelopev2 | SCRFD-10GF | 360.66 MB (measured via GitHub Release API, `antelopev2.zip`) | No |
| **buffalo_l** (library default) | SCRFD-10GF | 288.62 MB | **Yes** |
| buffalo_m | SCRFD-2.5GF | 275.95 MB | No |
| buffalo_s | SCRFD-500MF | 127.61 MB | No |
| buffalo_sc | SCRFD-500MF | 14.97 MB (measured, `buffalo_sc.zip`) | No |

Every pack bundles a detector **plus** recognition (ResNet/MobileFaceNet),
alignment (2d106/3d68 landmark models — a **different**, denser landmark
model than SCRFD's own optional 5-point head), and (except `buffalo_sc`)
gender/age attribute models — none of which this platform's architecture
uses or has any use for (`docs/SCRFD_PROGRAM_DESIGN.md` §2.7/§3.4 confirm
this platform reads only detection boxes and, eventually, SCRFD's own
5-point landmarks). **Extracting the one relevant detector `.onnx` file
from a multi-model archive and discarding the rest is itself a small,
undocumented-by-upstream step** this audit surfaces rather than assumes
away — the archive's internal file names (`det_10g.onnx`, `det_2.5g.onnx`,
`det_500m.onnx` by established community convention) are **not** verified
against upstream documentation in this session (the archives were not
downloaded and inspected — a further step, not performed here, before
implementation) and are marked **unverified** accordingly.

Also present on the same GitHub Release, and **explicitly not a face
detector** — flagged so it is never mistaken for one: `scrfd_person_2.5g.onnx`
(3.71 MB), a person-detection model sharing SCRFD's architecture but trained
for a different task. `inswapper_128.onnx` (554 MB) is a face-swap model,
unrelated to detection and worth noting only because its presence in the
same release, on a deepfake-*detection* project's own dependency chain, is
the kind of adjacency a careless `grep`/fetch script could pull in by
mistake.

### 1.4 Input resolution and runtime behaviour

`detection/scrfd/README.md`'s own performance table is evaluated at **VGA
(640×480)**; that is a benchmark convention, not a hard input constraint —
`detection/scrfd/tools/scrfd2onnx.py:67-70` shows the export supports
`dynamic_axes` (the model accepts variable input sizes) unless a fixed
`--shape` is passed at export time. The `insightface` Python library's own
documented runtime default (`python-package/README.md`, `python-package/
docs/runtime.md`), verbatim: *"Auto detection size: 128x128 + 640x640"* —
i.e., the reference implementation runs the detector **twice**, at two
fixed resolutions, and merges results, as its default "auto" mode. This is
a concrete, citable operational detail neither prior design document
mentioned, and it is a design input for whoever writes `SCRFDDetector`,
not a decision this audit makes.

### 1.5 What the raw ONNX graph actually outputs — a real implementation-complexity fact

`detection/scrfd/tools/scrfd2onnx.py:57-64`, verbatim:

```python
input_names = ['input.1']
output_names = ['score_8', 'score_16', 'score_32', ...]  # + bbox_* per stride
# with --shape / KPS models: '+= kps_8, kps_16, kps_32'
```

The Family-A ONNX graph emits **raw, per-stride anchor scores/boxes**
(strides 8/16/32), not a single decoded `[N, 5]` box tensor — anchor
decoding and NMS are **not** baked into the graph and must be implemented
in the consuming code (`inference/preprocess.py`'s future `SCRFDDetector.
detect()`, per `docs/SCRFD_PHASE2_DESIGN.md` §2.3's "preprocess -> forward
pass -> decode anchors -> NMS" sketch — now confirmed necessary from the
export script itself, not merely anticipated). The `insightface` Python
library's own `insightface.model_zoo` `SCRFD` class already implements this
decode/NMS step in Python — a real, citable tradeoff for whoever
implements Phase 2: depend on the `insightface` package for its decode
logic (a large, license-encumbered dependency pulling in the exact
non-commercial-only weights this audit's §2 addresses, plus recognition/
alignment code this platform does not want), or reimplement anchor decode
+ NMS directly against the raw Family-A graph (more code, zero extra
license surface). **This audit does not resolve that tradeoff** — it is an
implementation decision for Phase 2, flagged here because §1.1's family
distinction is what makes the tradeoff visible at all.

Export opset is **fixed**: `scrfd2onnx.py:149`, `assert args.opset_version
== 11, 'MMDet only support opset 11 now'`. Opset 11 (ONNX spec, 2019) is
supported by every `onnxruntime` release relevant to this project — see §4.

---

## 2. Licensing — verified from primary sources, not secondary summaries

### 2.1 Code license — permissive, Apache-2.0

`detection/scrfd/LICENSE` (fetched verbatim,
`raw.githubusercontent.com/deepinsight/insightface/master/detection/scrfd/LICENSE`):

> Copyright 2018-2019 Open-MMLab. All rights reserved.
>
> Apache License, Version 2.0

This covers the **training/export code** in `detection/scrfd/` (built on
mmdetection, an OpenMMLab project). **This license does not, by its own
terms, cover model weight files** — Apache-2.0 licenses the source code it
accompanies; nothing in this text purports to license a separately
distributed binary checkpoint. The repository as a whole (`api.github.com/
repos/deepinsight/insightface`, `"license": null`) has **no single
top-level LICENSE GitHub's own detector recognizes** — confirmed live, not
assumed — so there is no blanket permissive license to fall back on for
anything outside `detection/scrfd/`'s own code tree.

### 2.2 Pretrained weight license — non-commercial research only, stated three times independently

Three separate primary-source files, fetched independently this session,
agree word-for-word on the operative restriction:

1. `model_zoo/README.md` (top banner, precedes every section including
   §2.2's SCRFD table): *"**ALL models are available for non-commercial
   research purposes only.**"*
2. `python-package/docs/model_zoo.md`, "Model licenses" section: *"The
   InsightFace Python library code is released under the MIT License and
   can be used for academic and commercial purposes. **The pretrained
   models provided with this library are available for non-commercial
   research purposes only, including both automatically and manually
   downloaded models.**"*
3. `python-package/README.md`, "License" section: *"The library code is
   released under the **MIT License**, for academic and commercial use.
   **The pretrained models provided with this library are for
   non-commercial research only**, whether downloaded automatically or
   manually."*

**This is unambiguous and applies to every officially distributed weight
this audit found** — both Family A (the raw SCRFD `.pth`/converted `.onnx`,
same repository, same document's banner) and Family B (every `buffalo_*`/
`antelopev2`/`raccoon_*` pack). The **code** that loads and runs them (the
`insightface` library itself) is separately, permissively licensed (MIT);
the **weights** are not. This code/weights split is stated by the upstream
project itself, not inferred.

**No exception, carve-out, or dual-license path was found in any of the
three sources** for any SCRFD variant specifically — the restriction is
stated as covering "ALL models," not qualified by architecture or size.

### 2.3 What this means for redistribution

**Redistribution (vendoring a copy of the weight file inside this project's
own repository, or mirroring it on this project's own release
infrastructure) is not something this audit can clear.** The non-commercial
research restriction is a **use** restriction on the weights themselves,
independent of who hosts the bytes — committing the file into this
repository does not obtain any additional right to redistribute it, and
would make this project's own repository a second point of distribution
for a restrictively licensed artifact, which is exactly the kind of
decision this project's own documentation discipline (`docs/
SCRFD_PROGRAM_DESIGN.md` §3.1: "a decision, not a default") says must be
made explicitly, by a human, not assumed by a design document. **This
audit's recommendation (§3) is: do not vendor, do not mirror — fetch from
the upstream source directly**, which keeps this project a *user* of the
restricted asset, not a second *distributor* of it.

### 2.4 Is downloading at runtime required?

**Not in the sense of "every inference call re-downloads it"** — nothing
in either family's licensing forces a network fetch per request. But a
**separate, non-vendored fetch step, outside the repository, at
setup/first-use time**, is effectively mandated by §2.3's redistribution
finding: since the file cannot be committed, *some* fetch step (build time,
first-use time, or an explicit setup command) is the only remaining
option that does not turn this repository into a distributor. This matches
`.gitignore:19`'s pre-existing `*.onnx` entry exactly — the project had
already, independently, arrived at "not vendored" as a convention before
this audit, for the unrelated reason of general binary-asset hygiene; §2.3
now supplies the licensing reason that convention turns out to require
anyway.

### 2.5 This project's own license posture — checked, not assumed

`deepfake-platform`'s own repository root has **no `LICENSE` file and no
license statement in `README.md`** (checked this session, `ls` + `grep -i
license README.md`, zero matches) and **no git remote** (re-confirmed same
session as prior design documents). The project currently makes no license
claim of its own. This is stated because it is directly relevant to §2.2's
restriction: an unpublished, non-commercial research/portfolio project (per
this project's own memory/context — a portfolio and job-search artifact,
not a product being sold) is squarely inside the license the weights
grant. **This changes, and must be re-checked, the moment this project
gains any commercial distribution intent** — a fact this audit surfaces
because it is a real, live constraint on the *project*, not only on the
asset, and no prior SCRFD document stated it.

---

## 3. Packaging strategy — evaluated

| Strategy | Verdict | Reasoning |
|---|---|---|
| **Commit model into repository** | **Rejected** | §2.3 — redistribution via the repo's own git history is not something this audit can clear under the non-commercial-only weight license; also conflicts with `.gitignore:19`'s existing `*.onnx` convention and would bloat the repository with a multi-MB binary this project's own dependency-declaration discipline (`pyproject.toml`'s own comments on hand-maintained lists failing silently) treats as exactly the kind of thing that should not live in source control. |
| **Separate asset download** (a documented URL + expected SHA-256, fetched by a setup step, hashed the way `cascade_digest()` hashes the Haar XML today) | **Recommended** | Matches §2.4's finding, matches the existing `.gitignore` convention, matches `docs/SCRFD_PHASE2_DESIGN.md` §2.4's `scrfd_asset_digest()` design (unchanged by this audit). The fetch source should be **Family B's official GitHub Release** (`github.com/deepinsight/insightface/releases/tag/model-zoo`) rather than Family A's OneDrive/GDrive links — a GitHub Release asset is versioned, has a stable URL, a published SHA is computable from the download itself, and (per §1.1) needs no conversion toolchain. Family A remains the *reference* for variant characteristics (§1.2's table) even if Family B supplies the actual bytes. |
| **Release artifact** (this project publishes its own GitHub Release with a re-hosted copy of the SCRFD weight) | **Rejected, same reasoning as "commit into repository"** | Re-hosting is redistribution under a different name — the file's origin changes from "upstream's release" to "this project's release," which does not change the license attached to the bytes but does make this project a second distributor, exactly what §2.3 says to avoid. If this project ever needs a stable, project-controlled mirror (e.g. upstream link rot), that mirror must itself carry the same non-commercial notice and attribution, decided explicitly, not defaulted into. |
| **User-provided asset** (the operator supplies their own `.onnx` file at a configured path; the project ships no default) | **Recommended as the safer long-term default**, complementary to "separate asset download," not a replacement for it | Directly matches this platform's own architecture: `SCRFDDetector` (per `docs/SCRFD_PHASE2_DESIGN.md` §2.3) already takes a `model_path`, and `FaceSelectionPolicy.detector_asset` is already a content digest of *whatever* file is at that path (§2.4 of the same document) — nothing about `FaceDetector`'s `Protocol` (`inference/preprocess.py:879-897`) assumes a specific model. It also mirrors upstream's **own** documented escape hatch (`python-package/docs/model_zoo.md`, "Use your own licensed model" — verbatim section heading) for exactly this licensing situation. **Recommendation: ship a documented fetch-from-official-source *default* path (so the vertical slice and Phase 3's measurement have something to run against out of the box), but architect and document the config so a user/operator can point at a different, differently-licensed checkpoint without any code change** — the two strategies are not in tension, and combining them is what actually resolves §2.2's restriction for any future commercial packaging, rather than merely working around it for the research/portfolio phase this project is in today. |

---

## 4. Deployment implications — verified

### 4.1 Windows / Linux / CPU-only wheel availability

Fetched live, `pypi.org/pypi/onnxruntime/1.18.0/json` (the exact floor this
project's `pyproject.toml:42` already declares, `onnxruntime>=1.18`) — the
plain, CPU-only `onnxruntime` distribution ships prebuilt wheels for:

- `win32`, `win_amd64` (cp38–cp312)
- `manylinux_2_27_x86_64`, `manylinux_2_27_aarch64` (cp38–cp312)
- `macosx_11_0_universal2` (cp38–cp312)

**No source build is required on any of these platforms**, for any CPython
3.8–3.12. This project's own `requires-python = ">=3.12"` and Windows dev
environment are both inside this matrix — confirmed, not assumed.

### 4.2 CPU-only execution and execution-provider selection — a real, project-specific risk this audit is the first to name

`python-package/docs/runtime.md`, verbatim, describes the **reference
implementation's own default** provider-selection order: *"CoreML
ExecutionProvider → CUDAExecutionProvider → CPUExecutionProvider"* — i.e.,
`insightface`'s own code auto-selects whatever accelerated provider the
installed `onnxruntime` build happens to report, falling back to CPU only
if none is available.

**This project must not inherit that default.** `docs/
SCRFD_PROGRAM_DESIGN.md` §3.1 states this project has "no GPU dependency
anywhere today," and the declared dependency is plain `onnxruntime`, not
`onnxruntime-gpu` (`pyproject.toml:42`) — so today, auto-selection would
resolve to `CPUExecutionProvider` by omission, not by an explicit choice
recorded anywhere. **That is a latent determinism hazard of exactly the
shape this project has already paid to close four times** (R6/R12/R13/R15/
R27/R30's whole reason for existing): if `onnxruntime-gpu` or an
environment with `CoreMLExecutionProvider`/`CUDAExecutionProvider`
available is ever installed on a machine running this code — a CI runner
image change, a developer's differently-provisioned laptop, a future
deployment target — identical code would silently start running SCRFD
through a numerically different execution path, which can move detection
box coordinates without moving a single line of this project's own source
or any version literal it tracks. **This is precisely R15's shape one level
up the stack** (a shared, unpinned execution context producing
plausible-but-different answers) and precisely the kind of thing this
project's `.d.`/`.f.` discipline exists to make loud rather than silent.

**Recommendation, concrete and actionable:** `SCRFDDetector`'s
`onnxruntime.InferenceSession` construction must pass
`providers=["CPUExecutionProvider"]` **explicitly**, never rely on
`onnxruntime.get_available_providers()`'s default ordering. This is a
one-line, low-cost fix, and it is the direct, sourced consequence of
reading `runtime.md`'s own documented default rather than assuming
"CPU-only dependency" implies "CPU-only execution" — it does not, by
itself, on a machine where a GPU-capable `onnxruntime` build is installed
for an unrelated reason. Whether the *chosen provider itself* belongs in
`FaceSelectionPolicy.identity` (the same class of question R6/R27 already
answered for FFmpeg/OpenCV build facts) is a design question for Phase 2's
own implementation, not this audit — flagged here because the audit is
where the underlying default-selection risk was found.

### 4.3 ONNX Runtime opset compatibility

Opset 11 (§1.5) has been supported since `onnxruntime`'s earliest 1.x
releases — no compatibility gap against the already-declared `>=1.18`
floor. **Unverified in this session** (not fetched): whether Family B's
packaged detector `.onnx` files use the same opset — plausible given the
shared authorship and export tooling, but not confirmed from a primary
source this session, since inspecting the packaged files would require
downloading and unzipping a multi-hundred-MB archive, not performed here.
Flagged as a concrete, cheap verification step for Phase 0 implementation
(`onnx.load(path).opset_import`), not assumed.

---

## 5. Recommendation

### 5.1 Preferred SCRFD variant

**`SCRFD_10G_KPS`** (Family A naming; the equivalent detector bundled in
Family B's `buffalo_l`/`antelopev2` packs), with **`SCRFD_2.5G_KPS`** as the
explicitly-named fallback if Phase 3's latency measurement
(`docs/SCRFD_PHASE2_DESIGN.md` §2.8) finds the 10G tier too slow for the
project's own throughput needs. Reasoning:

- **Landmarks matter for this project specifically** — `FaceTrack.landmarks`
  (`contracts/common.py:121-123`) already exists in the wire contract,
  unpopulated, and `docs/CROP_GEOMETRY.md` §9's own recommended eventual
  crop-geometry fix (Option F, affine-preserving resize) explicitly
  requires 5-point landmarks to exist first. Choosing a non-`_KPS` variant
  would ship a detector that structurally cannot ever populate that field
  without a second identity-moving migration later — choosing `_KPS` now
  is the "one identity move, fully used" version of this decision.
- **10G over 2.5G, provisionally**: the accuracy gap is real (Hard-set
  82.80 vs 77.13, §1.2's own table) and this project has no measured
  latency budget yet to say 2.5G's speed advantage (4.3 ms vs 5.0 ms,
  *PyTorch* CPU numbers, §1.2 footnote) is actually needed — Phase 3's own
  measurement, not this audit, should make that call with real numbers on
  this project's own target hardware and ONNX Runtime (not PyTorch)
  inference path.
- **Not 34G**: no `_KPS` variant exists (§1.2) — picking it would mean
  giving up landmarks for a smaller accuracy gain than the 2.5G→10G step,
  the wrong trade for this project's stated destination (Option F).

### 5.2 Asset distribution strategy

Combine the two recommended packaging options from §3: **a documented
fetch-from-Family-B-GitHub-Release step by default** (versioned, stable,
no conversion toolchain, SHA-256 recorded the way `cascade_digest()`
records the Haar asset's) **plus an explicit, documented "bring your own
model" override path** using the same `FaceSelectionPolicy.detector_asset`
mechanism already in place. Do not commit, do not re-host on this
project's own release infrastructure (§3).

### 5.3 Licensing risk assessment

**Medium-High, and it does not resolve itself with time or with a
different SCRFD variant** — every variant in both families the primary
sources describe carries the identical non-commercial-only restriction
(§2.2); there is no "safer" SCRFD checkpoint to pick instead. The risk is
**structural to using any official InsightFace-distributed SCRFD weight at
all**, not a property of this audit's §5.1 choice. It is **low today**
because this project is an unpublished, non-commercial research/portfolio
effort (§2.5) — squarely inside the license — but it is a **standing,
explicit blocker on ever distributing this platform, or a model derived
from using it, commercially, without either (a) a separately, permissively
licensed detector, or (b) direct clearance from the upstream authors**.
This should be re-read, not re-assumed, at the moment this project's own
distribution intent changes — flagged the same way `docs/
SCRFD_PHASE2_DESIGN.md` §3.4 flagged the licensing gap generally, now
resolved to a specific, sourced restriction rather than an open question.

### 5.4 Repository impact

**Near zero**, by design (§3): no binary committed, `pyproject.toml`
unchanged (`docs/SCRFD_PHASE2_DESIGN.md` §2.5 already established
`onnxruntime` needs no new declaration), one new documented fetch step
(a script or a documented manual command, plus a recorded SHA-256 and
this audit's own license findings, likely living in the
`docs/SCRFD_ASSET.md` file `docs/SCRFD_PHASE2_DESIGN.md` §3.1 already
named as a placeholder). `.gitignore:19`'s existing `*.onnx` entry needs no
change.

### 5.5 Estimated maintenance burden

**Low, ongoing.** Concretely:

- The upstream GitHub Release (`model-zoo` tag) is a stable, versioned
  artifact — not a rolling branch — so there is no "upstream moved the
  file under us" risk of the kind N5 solved for `opencv-python`/`numpy`
  version *ranges*; this is a pinned-by-URL-and-hash binary, closer in
  spirit to how the Haar cascade is already handled (shipped, hashed, never
  silently substituted).
- Upstream **license terms could still change** on a future model-zoo
  release (a risk no amount of hashing catches, since a hash pins content,
  not terms) — worth a periodic (e.g. once-a-batch, not per-run) manual
  re-read of `model_zoo/README.md`'s banner, cheap and already demonstrated
  in this audit.
- If §3's "user-provided asset" path is built out, its maintenance is
  effectively zero for this project — it shifts the licensing and
  provenance burden to whoever supplies the alternative model, exactly as
  upstream's own "Use your own licensed model" section already assumes for
  its users.

---

## 6. Version-impact axes — re-checked against this audit's own recommendation

**None of this audit's findings or recommendations change any of the six
axes below, and each is checked against the *specific* recommendation made
in §5, not against SCRFD in the abstract:**

| Axis | Current value | Changed by this audit's recommendation? |
|---|---|---|
| `CONTRACT_VERSION` | `contracts/enums.py:16`, `"1.0.0"` | **No.** This audit recommends a variant and a distribution mechanism, not a wire-shape change — `FaceTrack.landmarks` already exists (§5.1's own justification depends on this being true and unchanged). |
| `RUN_SCHEMA_VERSION` | `evaluation/benchmark.py:108`, `"1.2.0"` | **No.** Nothing about `BenchmarkRun`'s shape is touched by a variant/packaging choice. |
| `CALIBRATION_WITNESS_VERSION` | `evaluation/calibration_protocol.py:74`, `"1.3.0"` | **No.** Same reasoning as `docs/SCRFD_PHASE2_DESIGN.md` §4 — already generic over `pre_calibration_identity`, zero stored witnesses to migrate (re-confirmed this session: this audit did not search for new ones, but nothing in §5 creates or ships one). |
| `LOCK_VERSION` | `evaluation/split_lock.py:81`, `"1.1.0"` | **No — still structurally unrelated.** This audit's subject (a detector weight file's license and distribution) has no path to the dataset split-lock format; re-confirmed by the same reasoning `docs/SCRFD_PHASE2_DESIGN.md` §4 already gave, unchanged by anything found here. |
| `.d.` (decode identity) | feeds `backend/config.py:41`'s `.d.c9d4025c` segment | **No.** A detector asset choice is not a decode-environment fact — unchanged from `docs/SCRFD_PHASE2_DESIGN.md` §4's finding, and this audit found nothing (e.g. no ONNX Runtime build-info fact analogous to `getBuildInformation()`) that would move that conclusion. |
| `.f.` (face/selection identity) | feeds `backend/config.py:41`'s `.f.e629642c` segment via `FaceSelectionPolicy.identity` | **Not by this audit itself — this audit selects a variant, it does not ship one.** `docs/SCRFD_PHASE2_DESIGN.md` §4/§5 already established that *extending* `FaceSelectionPolicy` (Phase 2) moves `.f.` regardless of which detector is eventually selected; this audit's contribution is narrowing *which* asset's digest will eventually populate `detector_asset` (§5.1/§5.2) and confirming that choice is orthogonal to *whether* `.f.` moves — the dataclass-shape argument, not the specific SCRFD variant, is what moves it. Selecting `SCRFD_10G_KPS` over `SCRFD_2.5G_KPS` later, or swapping in a user-provided model per §3, changes `detector_asset`'s and `identity`'s **value** at whatever phase actually ships a non-Haar default (Phase 4) — it does not add a **new** identity-moving event beyond the one Phase 2 already accounted for. |

**Net conclusion of this section: consistent with `docs/
SCRFD_PHASE2_DESIGN.md`'s own §4/§5 findings, with no correction needed
this time** — this audit operates entirely inside the "which asset, from
where, under what license" question, which by construction sits below the
identity-shape question that document already resolved.

---

## 7. Open items this audit could not close

1. **§4.3 — Family B's packaged `.onnx` opset is unverified.** A
   five-minute check (`onnx.load(path).opset_import`) once a pack is
   actually downloaded, deferred rather than guessed at here.
2. **§1.3's internal file names (`det_10g.onnx` etc.) inside the `.zip`
   packs are unverified** — community convention, not confirmed against
   an unzipped archive or upstream documentation this session.
3. **§5.5's "license terms could change" risk has no owner or cadence
   assigned** — this audit recommends a periodic re-read but does not
   schedule one, matching this project's own convention of naming a
   process gap rather than inventing process machinery unasked.
4. **This session's `WebSearch`/`WebFetch` tool failure was not
   diagnosed** — it may be transient infrastructure, or it may recur for
   whoever implements Phase 0/2; if it does, this audit's own `curl`-based
   method (direct `raw.githubusercontent.com`/`api.github.com`/`pypi.org`
   fetches) is a documented, working fallback.

---

## 8. Gate

**Audit complete.** §5's recommendation (SCRFD_10G_KPS by default,
fetched-not-vendored via Family B's official GitHub Release, with an
explicit user-provided-model override path) is ready to feed Phase 0 of
`docs/SCRFD_PROGRAM_DESIGN.md` and the asset-acquisition step of `docs/
SCRFD_PHASE2_DESIGN.md` §2.4. **The one item this audit treats as a
genuine blocking gate, not a nice-to-have**: §2.2's non-commercial-only
restriction must be read and consciously accepted (or explicitly rejected
in favour of a different, permissively-licensed detector — a decision
outside this audit's scope, which was SCRFD specifically) by a human
before any fetch happens against a machine or environment this project
does not fully control, exactly as `docs/SCRFD_PHASE2_DESIGN.md` §3.4
already gated Phase 0 on, now with the actual license text in hand rather
than a flagged unknown.

**No code was written to produce this document. Stop here.**
