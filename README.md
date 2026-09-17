# Deepfake Platform

Image and video deepfake detection. Web application, no terminal workflow.

**Status: S0 complete.** The contracts package is built and tested. No models, no
backend, no frontend yet — by design.

## Why contracts first

Every layer negotiates through `EvidenceBundle`. Defining it before anything that
produces or consumes it means the frontend, backend and workers can all be built
in parallel against fixtures, while the ML path is still blocked on dataset
licences and GPU rental. Eleven of the twenty-one build steps need neither.

`contracts/fixtures.py` emits realistic, contract-valid bundles covering every
verdict, both media kinds and every branch status — with no torch and no weights.

## Frozen architecture

| | |
|---|---|
| Branch A — spatial | ConvNeXt V2-T, RGB + SRM/Bayar residual channels. Target: the aligned face crop. Today scores the crop (image) or the whole frame (video, no face stage). |
| Branch B — semantic | DINOv2-B/14 frozen + probe. Target: the whole image. Today scores the same tensor as branch A — the face crop (image) or the whole frame (video). See R22. |
| Branch P — provenance | C2PA + EXIF. Not learned. **One-sided.** |
| Fusion | Logistic regression with interaction terms, temperature-calibrated |
| Verdict | Three states: `authentic` / `manipulated` / `insufficient_evidence` |
| Upload | Direct to object storage via presigned URL; bytes never touch the API |

There is no Branch C. The frequency branch was folded into Branch A as input
channels at architecture review.

### The one-sided provenance rule

A valid signature is evidence of authenticity. **The absence of provenance data
is evidence of nothing.**

A fitted fusion would otherwise learn "no metadata implies fake" — a real
correlation in every training corpus and a false one in production, where
platforms strip metadata from nearly everything. Left alone it penalises ordinary
photographs of real people, which is exactly the false positive the design exists
to avoid.

This is enforced structurally rather than by convention: `BranchResult` rejects a
negative logit for branch P, and `ProvenanceReport.authenticity_credit` is bounded
to `[0, 1]`. A negative contribution is unrepresentable.

## Layout

```
contracts/
├── enums.py        closed vocabularies — every cross-boundary string
├── common.py       MediaInfo, QualityReport, FaceTrack, Artifact, Threshold
├── branch.py       BranchResult + the one-sided guard for branch P
├── provenance.py   C2PA / EXIF reports, authenticity credit
├── video.py        frame curve, suspect segments, top frames
├── evidence.py     EvidenceBundle — the keystone
├── job.py          JobSpec, JobStatus, progress stages
├── errors.py       error taxonomy, HTTP mapping, user-facing messages
├── api.py          request/response schemas for every route
└── fixtures.py     deterministic fixtures — the reason S1 can start now
tests/
└── test_contracts.py
```

## Setup

Install [uv](https://docs.astral.sh/uv/) once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`)

Then, from the repo root:

```bash
uv sync --extra dev
```

This creates `.venv/` and installs the exact dependency versions pinned in
`uv.lock` — including `opencv-python==4.8.1.78` and `numpy==1.26.4`, the
versions every shipped runtime-identity literal (`.d.`, `STUB_ENGINE_VERSION`)
was computed against (see `docs/B24_N5_DESIGN.md`). `.python-version` pins the
interpreter to `3.12`: `numpy==1.26.4` has no installable wheel on newer
CPython releases, so `uv sync` will try to build it from source and fail
without a C toolchain if this file is missing or changed.

Run the suite with `uv run python tests/run_all.py`, or activate `.venv` and
run `python tests/run_all.py` directly.

**Windows note:** if `uv sync` fails installing a package with a hardlink
error (`os error 396`), the repo is likely under a cloud-synced folder
(OneDrive and similar). Set `UV_LINK_MODE=copy` and retry.

## Running

```bash
python tests/test_contracts.py
```

Runs standalone (47 tests, no pytest needed) and under `pytest` unchanged.

Generate fixtures for the frontend:

```bash
python -m contracts.fixtures --out fixtures_out
```

Print one:

```bash
python -m contracts.fixtures --stdout video_manipulated
```

## Requirements

Python 3.11+ and pydantic 2.9+. Nothing else. No torch, no numpy, no I/O.

## Contract stability

Fields split into two tiers, documented in `contracts/evidence.py`:

- **Stable** — `verdict`, `p_fake`, `interval`, `threshold`, `media`,
  `branches[].{id,status,logit,p}`, `quality.gate_passed`, `model.version`.
  Breaking one bumps `CONTRACT_VERSION` MAJOR and requires a client migration.
- **Evolving** — `video.*`, `provenance.findings`, `artifacts[]`, `faces[]`,
  `timing`, `disagreement`. May gain or reshape fields within a MAJOR version;
  clients must tolerate additions.

Without that split every field is frozen by default, which is how a contract
stops being an asset and becomes a constraint.

## Next

S1 opens two parallel tracks — backend and frontend against fixtures on one side,
preprocessing and dataset splitting on the other. Neither needs a dataset or a
GPU.
