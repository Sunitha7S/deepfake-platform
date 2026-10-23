# AI Model Development Plan

**Date:** 2026-10-23, revised 2026-10-24. **Repository state audited:** `master`
at `638b6c9` (locked CI 64 suites / 1,486 tests, local and GitHub Actions).
**Status of this document:** approved in direction (image-only DINOv2-probe
MVP), not yet implemented. No training has been run, no dataset or model
weights have been downloaded, no GPU time has been bought, and no production
code, contract, detector setting, acceptance criterion or version axis has been
changed to produce it.

Everything in §1 was established by reading the code, tests and artefacts named
beside it. Dataset and weight terms were read from their primary sources on
2026-10-24 and are recorded, with open questions, in
[`DATA_AND_WEIGHT_LICENSES.md`](DATA_AND_WEIGHT_LICENSES.md). Anything else
that depends on a fact outside this repository (cloud prices, literature
numbers, package licences) is marked **verify**.

---

## 1. Verified current state

### 1.1 Summary

The platform is a well-instrumented **serving and evaluation harness with no
model in it**. The only scoring backend that runs is `StubBackend`
(`inference/models.py`), which turns a SHA-256 of the input pixels into logits.
It is deterministic and documented as "not a model". There is no training code,
no dataset loader for any real corpus, no checkpoint, and no measurement of
deepfake-detection quality anywhere in the repository. The evaluation,
calibration and split tooling is substantial and tested, but it has only ever
been run on synthetic items and stub scores.

### 1.2 Item-by-item audit

Classification key: **Verified** = implemented and exercised by tests;
**Partial** = some of it exists and is tested, but the part that matters for a
real model is missing; **Design-only** = described in docs or README, no code;
**Absent** = neither code nor design artefact.

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Model architectures and implementations | **Design-only** (real models); stub **Verified** | README "Frozen architecture" names ConvNeXt V2-T + SRM/Bayar (branch A), DINOv2-B/14 + probe (branch B) and a logistic-regression fusion. No class, module or weight file implements any of them. `inference/models.py` holds `StubBackend` (hash → logit) and `_build_torch_backend` / `_build_onnx_backend`, which **raise `ModelUnavailable` on purpose** rather than fall back. Tests: `tests/test_inference.py::test_torch_backend_refuses_rather_than_falling_back`, `::test_onnx_backend_refuses_rather_than_falling_back`, `::test_stub_backend_scores_without_torch`. `DEFAULT_BRANCH_WEIGHTS` (0.72/0.24/0.04) are documented as placeholders. |
| 2 | Training and fine-tuning code | **Absent** | No training loop, optimiser, loss, `DataLoader`, `state_dict` or checkpoint I/O anywhere outside `.venv` (searched `import torch`, `timm`, `optimizer`, `backward(`, `state_dict`, `DataLoader`, `def train`, `fine-tun`, `epoch`). `torch` appears only as a lazy availability check in `inference/models.py`. `docs/PROJECT_ROADMAP.md` row R10: "training pipeline (doesn't exist)". |
| 3 | Dataset loaders, preprocessing, augmentation | **Partial** | *Serving* preprocessing is real and heavily tested: decode to RGB (`inference/preprocess.py` `cvtColor … BGR2RGB`), Haar detection, `FaceSelectionPolicy` with crop margin 1.3, `prepare_image` (square resize, `INTER_AREA` when shrinking, **un-normalised float32 0–255**; `normalize` with ImageNet mean/std exists but is unused by the stub). Corpus *identity* model exists: `evaluation/dataset.py` (`MediaItem`, `Compression.from_ffpp`, `IdentitySplitter`). **No loader reads FaceForensics++, Celeb-DF, DFDC or any other corpus from disk. No augmentation code exists.** |
| 4 | Train / validation / test splits | **Partial** (tooling verified; no real split) | Leakage-proof splitting by union-find over source and subject (`evaluation/dataset.py`), frozen split lock files with content hashes and migration (`evaluation/split_lock.py`, `docs/SPLIT_LOCK_FORMAT.md`), role protocol `FFPP_CELEBDF` (`evaluation/protocol.py`: FF++ train / FF++ val / FF++ test as calibration / Celeb-DF primary / DFDC secondary). Tests: `tests/test_evaluation_dataset.py` (21), `tests/test_split_lock.py` (40), `tests/test_evaluation_protocol.py` (40). **No lock file for any real corpus exists in the repository.** The existing protocol spends FF++ test on calibration; §3.2 replaces it. |
| 5 | Checkpoints and trained artefacts | **Absent** | No `*.pt`, `*.pth`, `*.onnx`, `*.safetensors`, `*.ckpt`, `*.bin` or `*.npz` outside `.venv`. `STUB_IMAGE` / `STUB_VIDEO` (`inference/models.py`) have `weights_uri=None`. The only real weights the code knows about are the optional SCRFD face-detector asset, which is not in the repo (`docs/SCRFD_ASSET.md`). |
| 6 | Experiment tracking and reproducibility | **Partial** | Strong *serving-side* identity: `ModelSpec.effective_version` composes weights, calibration, gate, selection and decode environment (`backend/config.py` `STUB_ENGINE_VERSION`), `uv.lock` with `[tool.uv] override-dependencies` pinning numpy/OpenCV, `BenchmarkRun` records with dataset fingerprint and history (`evaluation/benchmark.py`), calibration witnesses (`evaluation/calibration_protocol.py`). **No training-run record format, no experiment store, no seed or GPU/driver provenance.** R10 (GPU nondeterminism) is explicitly deferred (`docs/PROJECT_ROADMAP.md`). |
| 7 | Metrics, baselines, error analysis | **Partial** | Metrics verified against hand computation with explicit tie handling: ROC-AUC, PR/AP, EER, confusion matrix, P/R/F1 (`inference/metrics.py`, `evaluation/metrics.py`; `tests/test_evaluation_metrics.py`). Paired bootstrap and two-scope (overall + worst-slice) regression gate (`evaluation/benchmark.py`; `tests/test_evaluation_benchmark.py`, `tests/test_benchmark_reproducibility.py`). Risk–coverage and abstention breakdown (`evaluation/risk.py`). SVG report plots (`evaluation/svg.py`). **The only "baseline" is `StubPredictor`. No metric has ever been computed on real data.** |
| 8 | Threshold selection and calibration | **Partial** (tooling verified; shipped values are placeholders) | Temperature fitting and NLL (`inference/calibration.py`), ECE/MCE/Brier/reliability (`evaluation/calibration.py`), four threshold rules with bootstrap CI (`evaluation/thresholds.py`), joint threshold+band derivation (`evaluation/band.py`), witness-bound fitting that refuses benchmark corpora (`evaluation/calibration_protocol.py`). Tests: `tests/test_calibration*.py` (95 tests), `tests/test_threshold_band.py` (31), `tests/test_evaluation_operating.py` (28). **Shipped values are not fitted:** temperature 1.0 with `n_samples=0` (`CalibrationSpec.is_identity`), threshold 0.62 / target FPR 0.02 as `ModelSpec` defaults; `docs/NEXT_ENGINEERING_PROGRAM.md` records zero calibration witnesses in existence. |
| 9 | Model integration into inference | **Partial** (seam verified; no real backend) | The seam is `ModelBackend.score(batch: (N,H,W,3) float32) -> {BranchId: (N,) logits}` (`inference/models.py`). Pipelines call it at `inference/image_pipeline.py` `_crop_and_score` and `inference/video_pipeline.py` `_score_batch`. Engines are hard-wired to `"stub-image"` / `"stub-video"` (`workers/engine_registry.py`), and `verify_version_alignment` requires **both** engines to report the single configured `model_version`. Worker integration: `tests/test_worker_inference_integration.py`. |
| 10 | Image and video inference behaviour | **Verified as plumbing; meaningless as detection** | Image: decode → Haar → selection policy → crop (margin 1.3) → resize to `input_size` 384 → stub → fusion → temperature → three-state verdict (`inference/postprocess.py`). Video: 48 uniformly sampled frames (`default_video_selection`, `sample_frame_indices`), **whole frame, no face stage**, top-k mean pooling (k = n/8), hysteresis segments. Determinism, identity and abstention semantics are tested extensively (`tests/test_inference_determinism.py`, `tests/test_video_*`, `tests/test_headline_and_face_semantics.py`). No output says anything about authenticity. |
| 11 | GPU, memory and latency | **Partial** (measurement plumbing only) | Timing contract `inference/timing.py` (`preprocess_ms` / `inference_ms` split; Haar measured at ~1.27 s of a 1.43 s job on a 900×900 image). Peak-RSS measurement for frame extraction in `tests/test_frame_extraction.py` (one 4K job previously grew RSS by ~2.5 GB before the fix it guards). **No model latency, memory or throughput has ever been measured; no GPU requirement is established anywhere** (`docs/NEXT_ENGINEERING_PROGRAM.md` §2 prerequisites). |
| 12 | Dataset and model licences | **Partial** | Repo code MIT (`LICENSE`). SCRFD weights (non-commercial, not redistributable) and WIDER FACE (CC BY-NC-ND 4.0) audited in `docs/SCRFD_ASSET_AUDIT.md`. FF++, Celeb-DF v2 and DINOv2 terms were read on 2026-10-24 and recorded in [`DATA_AND_WEIGHT_LICENSES.md`](DATA_AND_WEIGHT_LICENSES.md); several questions there are unresolved. Package licences for the training stack have not been checked. |

Side finding: `settings.allowed_image_mime` and `settings.allowed_video_mime`
(`backend/config.py`) are declared but never enforced; upload MIME checks read
the constants in `contracts/api.py` directly.

### 1.3 Facts that constrain the plan

1. **The SCRFD NO-GO is a face-detector stability result, not a deepfake-quality
   result.** `docs/SCRFD_R13_MEASUREMENT_EVIDENCE.md` measures box jitter and
   score amplification under noise, using the stub as the scorer. It says
   nothing about whether any model detects manipulation. Haar stays the
   default, and the R13 criteria stay as they are.
2. **Train/serve skew is the main integration risk.** Serving crops with Haar
   at margin 1.3, resizes to a square without padding, and passes un-normalised
   RGB. A model trained on RetinaFace/MTCNN-aligned crops (the usual published
   setup) would be scored on different pixels in production. The training data
   pipeline must therefore call `inference.preprocess` itself (§3.3).
3. **Video is scored on whole frames.** A face-crop classifier scores whole
   frames out of distribution. The first real model is therefore image-only,
   and video is rejected rather than scored (§5.3).
4. **One `model_version` covers both engines.** `verify_version_alignment`
   compares a single configured string against both engines. §5.3 restricts
   the check to enabled engines.
5. **The protocol's Celeb-DF target (AUC 0.90–0.93, `evaluation/protocol.py`)
   is not a realistic first-model gate.** Published FF++-trained baselines
   score much lower on Celeb-DF (§4.2). The note is left unchanged in v1; §4
   pre-registers realistic gates instead.
6. **The existing protocol cannot express an untouched in-domain test.**
   `FFPP_CELEBDF` assigns calibration to FF++ test, and with only three split
   labels (`train`/`val`/`test`) the validation rules leave no room for a
   separate calibration set. §3.2 is the fix.
7. **Dataset access probably needs an academic affiliation.** The FF++ request
   form asks for a lab or affiliation and a supervisor's name and email;
   Celeb-DF asks for an academic email address. Applications must describe the
   applicant and the project accurately.

---

## 2. Recommended model and dataset strategy

### 2.1 Dataset options

Terms marked **checked** were read from the primary source on 2026-10-24; the
quotes, sources and open questions are in
[`DATA_AND_WEIGHT_LICENSES.md`](DATA_AND_WEIGHT_LICENSES.md).

| Corpus | Role | Access and terms | Size | Fit |
|---|---|---|---|---|
| **FaceForensics++ (c23)** | Train, validation, calibration, in-domain test | **Checked:** request form with affiliation and supervisor; "non-commercial research and educational purposes"; sharing only with people bound by the same terms; silent on trained models | **Checked:** ~10 GB for FF++ c23 videos (official dataset README); raw ~500 GB, not needed | Standard training corpus; official 360/70/70 **pair** splits (counted from the official split files); 4 classic methods allow leave-one-method-out tests |
| **Celeb-DF v2** | External benchmark only | **Checked:** request form asking for an academic email; "non-commercial research purposes only"; must not reproduce, publish or distribute "any derived data" | Not stated in the README | Standard cross-dataset benchmark: 590 celebrity + 300 YouTube real videos, 5,639 synthesized, official 518-video test list |
| DFDC (full) | Secondary benchmark | Not checked | ~470 GB (verify) | Deferred |
| DeeperForensics-1.0, WildDeepfake | Later robustness | Not checked | — | Optional, later |
| Third-party mirrors of FF++ / Celeb-DF | — | Unverifiable provenance | — | **Do not use** |

**Recommendation:** FF++ c23 for fitting, selection, calibration and the
in-domain test; Celeb-DF v2 as the single external benchmark. DFDC stays
deferred.

**What may be published.** Until the authors confirm otherwise in writing:

- no frames, crops, features or trained weights are committed or published;
- no public hosted demo serves an FF++-trained model;
- nothing derived from Celeb-DF beyond aggregate metrics is published, and
  whether aggregate metrics count as "derived data" is an open question.

**Personal data.** These corpora depict real people, so faces are personal
data under UK GDPR. Keep them on local, access-controlled storage outside the
repository, and delete them when the licence or the project ends.
Error-analysis galleries stay local; only aggregate numbers go into `docs/`.

**If access is refused.** Email the dataset authors describing the project
accurately, or research openly licensed alternatives as a separate task. No
alternative corpus has been evaluated yet.

### 2.2 Model options

| Option | What it is | Weights licence | Compute | Generalisation evidence | Contract fit | Interview value |
|---|---|---|---|---|---|---|
| **A. Frozen foundation backbone + linear probe** (DINOv2-B/14, or CLIP ViT-B/16) | Extract features once, then fit logistic regression | DINOv2 ViT-B/14: **checked**, Apache-2.0 for code and weights. CLIP: verify | Feature extraction only; the probe trains in seconds on CPU | Frozen CLIP-feature probes (Ojha et al., CVPR 2023, "UnivFD") generalise well to unseen *generators*; evidence for face-swap video at c23 is weaker (verify) | Matches the planned **branch B**. Single `ModelBackend` | High: probing foundation models, cheap ablations, deterministic training |
| **B. Fine-tuned CNN** (EfficientNet-B0/B4 from `timm`, ImageNet init) | End-to-end fine-tune on face crops | Verify | Single GPU, a few GPU-hours for B0, ~10–20 for B4 | The classic FF++ baseline family. In-domain is strong; cross-dataset drops sharply | Fits **branch A** (without SRM/Bayar) | High: a real training loop with augmentation, early stopping, multi-seed |
| **C. Original branch A**: ConvNeXt V2-T + SRM/Bayar residual channels | Modified stem (6+ channels) | Official pretrained weights believed CC-BY-NC 4.0 (verify) | Most expensive | Plausible but unmeasured | Fits branch A | High, but a research bet, not an MVP |
| **D. Original fusion**: logistic regression with interaction terms | Second-stage model | — | Trivial | Needs two working branches and a fusion-fitting split | `fuse()` in `inference/postprocess.py` would change | Medium. **Gap:** no protocol role exists for fitting fusion; it must be designed before fusion is fitted |
| **E. Adopt a third-party pretrained deepfake detector** | Download someone else's weights | Often unclear | None | Theirs, not yours | Fits | **Low** |

**Recommendation:**

1. **MVP model = Option A with DINOv2-B/14**, trained as a linear probe on
   Haar crops produced by the platform's own preprocessing.
2. **Checkpoint:** Hugging Face `facebook/dinov2-base` at revision
   `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, file `model.safetensors`
   (346,345,912 bytes, SHA-256
   `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`).
   Safetensors avoids executing pickled code, and pinning by revision and
   SHA-256 avoids `torch.hub`, which runs code from the repository's current
   head. The XRay-DINO and Cell-DINO weights in the same GitHub repository have
   different, non-commercial licences and must not be used.
3. **Comparison baselines, required in the same report:** (i) the stub
   (chance, as a sanity floor); (ii) a CLIP ViT-B/16 probe (same pipeline,
   different backbone, licence to verify); (iii) a deliberately weak low-level
   baseline (logistic regression on colour/JPEG statistics).
4. **Flagship extension = Option B**, then a designed fusion (Option D).
   Option C stays a documented research target, off the critical path.

The backbone is chosen on **FF++ validation only**, before FF++ test or
Celeb-DF is ever scored.

---

## 3. Training and evaluation pipeline

### 3.1 Where the code lives

A new top-level `training/` directory that is **not** added to
`[tool.setuptools] packages`. That keeps it out of every shipping package and
out of `tests/import_audit.py`'s serving dependency audit. It may import
`inference.preprocess`, `evaluation.*` and `contracts`. Nothing may import it.
Its dependencies go in a new optional extra (`train = [torch, transformers,
scikit-learn]`, licences to verify). That requires a `uv lock` refresh, so it is
a reviewed, identity-neutral change: the `override-dependencies` pins on numpy
and OpenCV must stay, and the change must be verified as identity-neutral
before commit.

### 3.2 Split, calibration and test design

The FF++ official splits are **pairs** of source videos: 360 train / 70
validation / 70 test pairs (720/140/140 sources), counted from
`dataset/splits/{train,val,test}.json` in the official repository. Each pair
has 2 real videos and, for the 4 classic methods, 8 manipulated videos (fakes
are named `<target>_<source>`). Every item from a pair gets the pair id as its
`source_id`, so a pair can never straddle splits.

| Partition | Content | Used for | Never used for |
|---|---|---|---|
| FF++ train-fit | 290 official-train pairs | Fitting the probe; choosing its regularisation by 5-fold cross-validation grouped by pair | Anything else |
| FF++ validation | 70 official-validation pairs | Choosing the backbone and the pooling rule. Consumed by the end, never reported as performance | Calibration, reporting |
| **FF++ calibration** | **70 pairs carved from official train** (seeded, pair-grouped, written to the split lock) | Temperature, threshold and abstention band only | Training, selection, reporting |
| **FF++ in-domain test** | 70 official-test pairs | Scored **once**, after pre-registration: AUC and thresholded metrics at the frozen operating point | Anything before that single run |
| **Celeb-DF v2** (external) | All of it, plus a sub-report on the official 518-video test list | Scored **once**, after pre-registration | Selection, calibration, thresholds, iteration |

Calibration is carved from train rather than validation so that the FF++
validation and test partitions stay identical to the official ones, which keeps
FF++ numbers comparable with published work. The cost is 19% less training data,
which is acceptable for a linear probe.

**Conflict with the existing protocol.** `FFPP_CELEBDF`
(`ffpp-train/celebdf-primary/v1`) assigns calibration to FF++ test, so FF++
test is spent and cannot be described as untouched. With three split labels the
rules in `validate_protocol` (a VALIDATION role is required; CALIBRATION may not
use the training split) leave no way to hold selection, calibration and an
untouched test apart. The smallest safe change, in evaluation code only:

1. Add a split label `calibration` to `evaluation.dataset.Split`. The split-lock
   format moves from 1.1.0 to 1.2.0 with a no-op migration for existing files.
   This is an evaluation-artefact version, not the model-version axis.
2. Add a dataset role `IN_DOMAIN_TEST`. It must use the training corpus, share
   no split with any fitting role, and is not a benchmark role. The report's
   `in_domain` run is selected from it.
3. Add a protocol constant `FFPP_CELEBDF_V2`
   (`ffpp-train/celebdf-primary/v2`). v1 stays unchanged because existing tests
   pin it; its docstring gains a pointer to v2.

Code that iterates over every split label must keep all existing suites
passing: `evaluation/dataset.py:673,731`, `evaluation/split_lock.py:352,386,534`,
`evaluation/protocol.py:282`.

### 3.3 Stages

Each stage writes one artefact with a SHA-256, and each stage records the hashes
of its inputs.

1. **Manifest** (`training/manifest.py`). Walk a local corpus directory and emit
   one row per video: `dataset_id`, pair id as `source_id`, `media_id`,
   `method`, `compression`, `label`, file SHA-256, byte size. Output is
   `manifest-<corpus>.jsonl` plus its digest. Rows map one-to-one onto
   `evaluation.dataset.MediaItem`.
2. **Split.** Build the five partitions in §3.2, freeze them with
   `evaluation.split_lock.create_lock`, and run `detect_leakage` to prove no
   pair straddles partitions. All later stages read items only through
   `FFPP_CELEBDF_V2.select()`.
3. **Frames and crops.** Choose `k` frames per video with the serving rule
   (`sample_frame_indices`, uniform): fixed `k` for training (e.g. 32) and the
   serving 48 for evaluation. Detect and crop with **the same `HaarDetector` +
   `FaceSelectionPolicy` + `crop_face` + `prepare_image` the image pipeline
   uses**. Record per-video Haar miss rates **by class and method**, because
   unequal miss rates are a selection bias the benchmark must report. A test
   asserts that a crop produced by the training path is byte-identical to the
   one `ImagePipeline` scores, for a fixture image.
4. **Features.** Run the frozen backbone once in eval mode at input size 224
   and cache features with a manifest. Record the backbone SHA-256,
   torch/CUDA/cuDNN versions and the device (the provenance R10 needs).
   Augmentation does not apply to a frozen-feature probe; for Option B, JPEG,
   resize, blur and colour-jitter augmentation is applied to training crops
   only.
5. **Fit.** Logistic regression (L2, C chosen by 5-fold cross-validation
   *within FF++ train-fit*, grouped by pair). Option A's solver is
   deterministic given features; anything stochastic uses 3 seeds.
6. **Select.** Backbone and pooling choice are selected on FF++ validation
   (video-level AUC) only. The selection, and everything rejected, is written
   to the run record.
7. **Calibrate.** `evaluation.calibration_protocol.fit_calibration` on the
   **FF++ calibration** partition only. This produces the first real
   `CalibrationWitness`. The threshold comes from `at_target_fpr` (2%) and the
   band from `derive_operating_point`.
8. **Pre-register.** Commit `docs/PREREGISTRATION.md` (§4) before step 9.
9. **Evaluate, once.** Score the FF++ in-domain test and Celeb-DF v2 through
   `run_benchmark`, each run recording the pre-registration commit hash.
   Report Celeb-DF on the full corpus and on the official 518-video list. Any
   later model is a new run compared with `paired_bootstrap`, never a quiet
   re-score.
10. **Export.** Backbone plus probe exported to ONNX as one graph.
    Normalisation happens inside the graph or inside `score`, because
    `prepare_image` passes raw 0–255. Check parity against torch on cached
    validation crops.

### 3.4 Run record (experiment tracking)

One JSON file per run in `experiments/runs/<run_id>.json`, committed (no data
inside). It contains: git SHA, `uv.lock` SHA-256, manifest digests, split-lock
digest, protocol name, backbone revision and SHA-256, every hyperparameter,
seeds, hardware and driver provenance, wall-clock time, per-partition metrics
with bootstrap CIs, the calibration witness digest, the pre-registration commit
(for evaluation runs), and the checkpoint and ONNX SHA-256. Benchmark rows go
through the existing `append_history`. No external tracking service is needed.

### 3.5 Artefact storage

Weights stay **out of git**. Default: a local `artifacts/` directory
(gitignored), addressed by SHA-256 in `ModelSpec.weights_uri`. Publishing waits
on the open licence questions.

---

## 4. Acceptance criteria

All thresholds below are **pre-registered**. Before anything touches the FF++
in-domain test or Celeb-DF, `docs/PREREGISTRATION.md` is committed with every
gate in this section, and each evaluation run records that commit hash.
Criteria cannot be edited after results are seen; a change is a dated addendum
appended below the original, which stays visible. Every Celeb-DF evaluation is
appended to the benchmark history, so repeated scoring is visible rather than
silent. A model that misses a performance gate is reported as missing it.

### 4.1 Training completion and artefact reproducibility

| Criterion | Measure | Why |
|---|---|---|
| R-1 Re-running a probe fit from the same cached features gives the same weights | Max absolute coefficient difference ≤ 1e-6 | The probe solver is deterministic. Anything else is a pipeline bug |
| R-2 Re-extracting features on the same hardware class reproduces validation AUC | \|ΔAUC\| ≤ 0.002 | GPU kernels are not bit-exact (R10). This bounds the effect instead of pretending it is zero |
| R-3 Seed variance is reported (Option B) | Mean ± SD over 3 seeds; validation-AUC SD ≤ 0.01 | A single-seed number is not a result |
| R-4 Every artefact is traceable | Run record present; the hashes in the record match the files; serving refuses a weights file whose SHA-256 differs from `ModelSpec` | Turns "which model produced this verdict?" into a lookup |

### 4.2 Held-out performance and generalisation

Leakage control: the partitions in §3.2, frozen in a split lock and checked
with `detect_leakage`. The FF++ in-domain test and Celeb-DF contribute nothing
to any fitted quantity. All metrics are **video-level** with 95% bootstrap CIs
(resampling videos, not frames).

| Criterion | Gate | Why this number |
|---|---|---|
| G-1 Pipeline sanity: FF++ in-domain test AUC | ≥ 0.85 | FF++ in-domain detection is well established in the literature. Failing this almost certainly means a data or preprocessing bug, not a scientific finding |
| G-2 **External: Celeb-DF v2 AUC** (official test list) | Point estimate ≥ 0.70 **and** 95% CI lower bound > 0.60 | The Celeb-DF paper reports FF++-trained Xception-c23 at about 0.65 frame-level (verify). 0.70 means "at or above the classic baseline", not state of the art |
| G-3 Beats the low-level baseline on Celeb-DF | `paired_bootstrap` win probability ≥ 0.95 | Shows the backbone carries signal beyond trivial statistics |
| G-4 Leave-one-method-out on FF++ (diagnostic) | Reported for all 4 methods; no gate | The honest description of unseen-method generalisation |
| G-5 Worst slice reported | Per method, compression and resolution bucket, with Haar miss rate | The existing two-scope gate exists for exactly this |

### 4.3 Precision, recall, F1 and ROC-AUC

ROC-AUC and AP are reported everywhere. Precision, recall and F1 are reported
**only at the operating point fixed on the calibration partition**, never at a
threshold chosen on an evaluation set (`at_max_f1` on Celeb-DF would be
leakage). The report gives the achieved FPR on the FF++ in-domain test and on
Celeb-DF real videos next to the 2% target. **Disclosure rule O-1:** if the
achieved Celeb-DF FPR exceeds 10%, the demo must show the caveat.

### 4.4 Calibration and threshold selection

The calibration partition holds about 140 real videos. At a 2% target FPR that
is roughly 3 expected false positives, so the threshold is poorly determined
and its confidence interval will be wide. The 2% target (the shipped design
value) is kept, and the interval is reported from a bootstrap that resamples
whole videos. If the achieved-FPR interval is wider than ±2 percentage points,
the operating point is reported as statistically under-determined; the target
is not changed to hide it.

| Criterion | Gate | Why |
|---|---|---|
| C-1 Temperature and threshold fitted only on the calibration partition | A `CalibrationWitness` exists and `verify_calibration_witness` passes | Uses the guard the repo already built |
| C-2 Calibration quality on data that did not fit it | ECE (15 bins) ≤ 0.05 on the FF++ in-domain test | The in-domain test is untouched, so this is an honest out-of-sample check |
| C-3 External calibration reported | Celeb-DF ECE and Brier score with reliability diagram; no gate | Shift is expected; hiding it would be dishonest |
| C-4 Threshold uncertainty | `bootstrap_threshold_ci`, resampling by video | A threshold is an estimate |

### 4.5 Latency, memory, throughput

Measured on named hardware (CPU model, RAM, OS) with the `Timing` contract,
p50/p95 over ≥ 100 runs after warmup. These are budgets for a CPU-only demo
machine, not universal constants.

| Criterion | Gate (CPU, ONNX Runtime) |
|---|---|
| L-1 Image `inference_ms` p95 (one crop) | ≤ 400 ms |
| L-2 Worker RSS increase after model load | ≤ 1.5 GB |
| L-3 Cold `warmup` | ≤ 30 s; reported |
| L-4 Throughput (images/min, one worker) | Reported; no gate |

Haar time is reported separately in `preprocess_ms` and is not counted against
L-1. Video latency is out of scope while video is disabled.

### 4.6 Failure handling and model-version identity

| Criterion | Test |
|---|---|
| F-1 Missing, truncated or hash-mismatched weights → `ModelUnavailable` at warmup; the worker refuses to start; no stub fallback | New tests mirroring `test_torch_backend_refuses_rather_than_falling_back` |
| F-2 `effective_version` changes when weights, temperature, threshold or input size change | Extend `tests/test_model_version.py` |
| F-3 `verify_version_alignment` passes with the real model configured, and fails on a stale `DFP_MODEL_VERSION` | `tests/test_startup_alignment.py` |
| F-4 ONNX/torch parity | Max \|Δlogit\| ≤ 1e-3 on 500 validation crops |
| F-5 No input yields `NaN`/`inf`, and the contract invariants hold | Property test over corrupt or extreme inputs through `build_image_bundle` |
| F-6 Video is never scored while a real image model is configured | `tests/test_video_disabled.py` (§5.3) |

### 4.7 Integration and end-to-end

| Criterion | Test |
|---|---|
| E-1 Upload → analyse → bundle with the real backend through the API and SQLite queue | Opt-in suite (skipped when weights are absent, and says why), plus one CI-able run with a tiny licence-clean ONNX graph (the SCRFD suite precedent) |
| E-2 Golden outputs | 10–20 licence-clean images (own photos or CC0): logits pinned to ±1e-3 for the exact version |
| E-3 Existing suites stay green | `StubBackend` remains the default test fixture, and all existing suites keep passing |

---

## 5. Preserving the platform: the smallest safe interface

### 5.1 What does not change

API routes and response shapes, `EvidenceBundle` and `CONTRACT_VERSION`, the
OpenAPI snapshot, the SQLite queue, the worker, retry and lease logic, the rate
limiter, Haar as the default detector, and the R13 criteria. The SCRFD NO-GO
stands as a detector-stability result.

### 5.2 The model seam

A real model needs exactly these changes, all to be made **after approval**:

1. **`inference/models.py`**: implement `OnnxBackend.score` behind
   `_build_onnx_backend`. Input `(N,H,W,3)` float32 0–255 RGB; normalise
   internally; return `{BranchId.B: logits}`. Verify the SHA-256 of
   `weights_uri` on load. `StubBackend` stays as is.
2. **A `ModelSpec`** for the trained model: `backend="onnx"`, `input_size=224`,
   `branch_weights={B: 1.0}` (A absent until it exists; `manipulation_evidence`
   already skips unscored branches, which a test must confirm), and the fitted
   `CalibrationSpec` with its witness and threshold.
3. **`workers/engine_registry.py`**: take the model names from settings
   instead of the `IMAGE_MODEL`/`VIDEO_MODEL` constants.
4. **`backend/config.py`**: the new `model_version` literal. This moves the
   weights identity, which is an intended identity move that needs its own
   approval (`docs/NEXT_ENGINEERING_PROGRAM.md` §2 version-axis table).

### 5.3 Image-only behaviour: video is rejected, never scored

No contract change and no new enum value.

- **New setting `DFP_VIDEO_ANALYSIS_ENABLED`.** Default `true`, so the
  stub-only development configuration and all existing suites behave exactly as
  today.
- **Fail-closed startup.** The API and worker refuse to start if a non-stub
  image model is configured while video analysis is enabled with the stub video
  model. A stub can never produce video scores beside a real model.
- **When disabled, `POST /uploads` with `media_kind=video` returns 415
  `unsupported_media_type`** before any cache lookup. The code and status
  already exist and are already declared for this route, so the OpenAPI
  snapshot does not change. The message is overridden through `ApiError.of`'s
  `message` argument, because the default message names MP4 as supported.
- **Worker defence in depth.** A video job already queued when video is
  disabled fails as a permanent error without any scoring call. The recorded
  job error is pinned by a test during implementation.
- **Version check.** `verify_version_alignment` checks only enabled engines.
- **Tests (`tests/test_video_disabled.py`):**
  - the upload returns 415 with the overridden message;
  - a spy confirms the backend's `score` is called 0 times for video;
  - the startup refusal fires for the forbidden combination;
  - the OpenAPI snapshot is unchanged;
  - the default configuration behaves exactly as before.

Returning an `insufficient_evidence` bundle for video was rejected: it would
need an invented `VideoAnalysis` block and probably a new abstain reason (a
contract change), and it would mislabel "unsupported" as "not enough evidence".

---

## 6. Compute and budget assumptions

The figures below are **targets and estimates, not guarantees**; cloud prices
are **verify**. Assumes about 160k training crops (FF++ c23, 32 frames × ~5,000
videos) and ~35k evaluation crops.

| Work | Hardware | Estimate |
|---|---|---|
| Smoke test S0 | Local CPU | Under 15 min; £0 |
| Frame extraction + Haar crops | Local CPU | 4–10 h wall clock, unattended |
| DINOv2-B/14 / CLIP ViT-B features | Free notebook GPU, or a rented consumer GPU (~£0.25–0.60/h) | 1–3 GPU-h per backbone; CPU overnight is a fallback |
| Probe fits and calibration | CPU | Minutes |
| EfficientNet-B4 fine-tune, 3 seeds (M6) | One 24 GB GPU | 30–60 GPU-h |
| Storage | Local disk | ~50–80 GB including crops and features |

**Budget target:** MVP ≤ £20 of compute (possibly £0 on free tiers); the full
flagship program ≤ £100. **No GPU rental, paid service or full-dataset download
happens without a separate approval for that step.**

### 6.1 Smoke tests

**S0: before any dataset access.** Needs approval for the downloads listed.

- **Data:** about 10 of your own photos and 2 short phone clips (only people
  who consent), plus synthetic frames with no face to exercise the Haar-miss
  path. Nothing from FF++ or Celeb-DF.
- **Checks, in order:**
  1. Decoding and frame sampling run through the serving code.
  2. Haar crops are byte-identical to what `ImagePipeline` produces.
  3. Haar miss rates are counted.
  4. Crops are resized to 224 and normalised.
  5. DINOv2 features are (N, 768), finite, identical across 2 runs, and the
     weight SHA-256 matches the pin.
  6. A logistic-regression fit on toy labels (originals vs a deterministic
     synthetic degradation) gives identical coefficients on re-run.
  7. A run-record JSON is written with all hashes.
  8. `FFPP_CELEBDF_V2.select()` refuses Celeb-DF and in-domain-test items.
- **Toy labels only test the mechanics; they say nothing about detection.**
- **Downloads:** CPU torch, transformers and scikit-learn wheels (about 1 GB,
  estimated) and the pinned 346 MB DINOv2 safetensors.
- **Cost:** £0, local CPU.

**S1: after FF++ access is granted.** 5 train-fit pairs, 2 validation pairs and
2 calibration pairs; never in-domain-test pairs or Celeb-DF. Full extraction
follows only after S1 passes and is approved.

---

## 7. Milestones, dependencies and estimates

Estimates are in **focused working days** for one engineer, assuming this
repository's testing discipline is kept. Calendar time depends on weekly hours
and on dataset approval.

| Milestone | Content | Depends on | Estimate |
|---|---|---|---|
| **M0a** Licence register and access requests | Keep `DATA_AND_WEIGHT_LICENSES.md` current; submit accurate FF++ and Celeb-DF requests; email authors with the open questions | — | 1 day + **unknown wait** (approval is not guaranteed without an academic affiliation) |
| **M0b** Protocol v2 | `Split.CALIBRATION`, `IN_DOMAIN_TEST`, `FFPP_CELEBDF_V2`, split-lock 1.2.0 migration, tests | Approval | 2–3 |
| **M0c** Video gating | §5.3 setting, rejection, fail-closed startup, tests | Approval | 1–2 |
| **M0d** Smoke test S0 | `training/` scaffold, `train` extra, S0 checks | Approval of downloads | 2–3 |
| **M1** Data pipeline | Manifest, split lock, leakage check; frame/crop extraction through `inference.preprocess`; Haar miss-rate report; S1 | M0a–M0d + FF++ access | 4–6 |
| **M2** Baselines + MVP probe | Feature cache; DINOv2 + CLIP probes; low-level baseline; validation selection; run records; R-1/R-2 | M1 | 4–6 |
| **M3** Calibration, pre-registration, evaluation | First real witness; operating point; `PREREGISTRATION.md`; one in-domain test run and one Celeb-DF run; slices; leave-one-method-out; local error analysis; results write-up | M2 + Celeb-DF access | 4–6 |
| **M4** Serving integration | ONNX export + parity; `OnnxBackend`; spec, version move (separate approval); F-1…F-6, E-1…E-3; L-1…L-4 | M3 | 4–6 |
| **M5** Demo + model card | Docker Compose (API + worker + SQLite); minimal single-page image upload UI showing verdict, calibrated probability, interval and abstention; model card with data, metrics and limitations | M4 | 5–8 |
| **MVP total** | M0a–M5 | | **~27–40 days** (+ access wait) |
| **M6** Flagship extension | EfficientNet-B4 branch A, 3 seeds; designed fusion split + fitted fusion; Video Face Stage, after which video is re-enabled; robustness (JPEG, resize, blur) curves | M5 | **20–30 days** |

---

## 8. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Dataset access refused (affiliation required) | Blocks M1 onward | Accurate application; email the authors; research openly licensed alternatives as a separate task |
| Licence questions stay unresolved | No public weights or hosted demo | Local Docker demo and a recorded walkthrough; publish only aggregate metrics |
| Cross-dataset AUC below gate G-2 | Weaker headline | Pre-registered; report honestly with failure analysis. Do not tune on Celeb-DF |
| Calibration partition too small for a 2% FPR target | Wide threshold interval | Reported as under-determined (§4.4); the target is not moved after the fact |
| Haar misses faces unevenly by class | Biased evaluation | Report miss rates by class and method, and treat them as a slice |
| Train/serve preprocessing drift | Silent accuracy loss | Byte-identity test between the training crop path and `ImagePipeline` |
| Split-lock format change breaks existing artefacts | Unreadable locks | No-op migration; existing suites must pass unchanged |
| Backbone licences (ConvNeXt V2, CLIP) | Blocks options B–C | Verify each before use; DINOv2 is checked |
| GPU nondeterminism (R10) | Irreproducible numbers | Bounded by R-2; provenance in the run record |
| Scope creep into research | MVP never ships | M6 starts only after M5 |

---

## 9. Practical route to a demo

1. **Local (MVP):** `docker compose up` runs the API, one worker and SQLite.
   The weights file is mounted from a local path and checked by SHA-256 at
   warmup. A static single-page UI posts images to the existing API and renders
   the `EvidenceBundle`: verdict, calibrated `p_fake`, interval, abstention
   reason, timing and model version. Video uploads receive the 415 response in
   §5.3.
2. **Hosted:** only after the dataset authors confirm in writing that serving a
   model trained on their data publicly is permitted.
3. **Always:** a model card in `docs/` and a results page with the reliability
   diagram, ROC curves and slice table generated by `evaluation/svg.py`,
   containing aggregate numbers only.

---

## 10. Open decisions

| Decision | Status |
|---|---|
| Applicant affiliation for the dataset requests | Open; owner: project author |
| Target FPR | 2% kept as pre-registered (§4.4) |
| Stub video in the stub-only development configuration | Unchanged (default `DFP_VIDEO_ANALYSIS_ENABLED=true`); revisit if desired |
| Split-lock format bump to 1.2.0 | Approved as part of M0b |
| Model-version move for the trained model | Requires its own approval at M4 |
