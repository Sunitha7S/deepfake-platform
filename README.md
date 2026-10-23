# Deepfake Platform

Image and video deepfake detection. Web application, no terminal workflow.

**Status:** the API, job queue, object storage, worker and inference pipeline are
built and run end to end (see [Running the platform locally](#running-the-platform-locally)).
Scoring is a **deterministic stub**, not a trained model — a verdict reflects the
file's hash, not its authenticity. There is no frontend yet.

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

## Running the platform locally

The API and the worker are **separate processes**. They meet through two shared
places: the job queue and the storage root. Both processes read `DFP_*`
environment variables, or a `.env` file in the directory they are started from.

**1. Point both processes at a shared queue.** Create `.env` in the repo root
(it is gitignored):

```
DFP_QUEUE_KIND=sqlite
```

Everything else can stay at its default, which both processes then share:

| Setting | Default | What lives there |
|---|---|---|
| `DFP_QUEUE_PATH` | `./var/queue.sqlite3` | The job queue: one SQLite file, plus its `-wal`/`-shm` side files |
| `DFP_STORAGE_ROOT` | `./var/storage` | Uploaded files, upload sessions, results |

Relative paths resolve against **each process's own working directory**. Start
both from the repo root, or set both variables to absolute paths. To confirm
the processes agree, compare the `queue_backend` field in the API's
`dependencies checked` log line with the one in the worker's `worker starting`
line: they must name the same file. `var/` is gitignored; delete it to reset all
local state.

**2. Start the API.** uvicorn is not a declared dependency, so `--with` adds it
to a temporary overlay without touching `uv.lock`:

```bash
uv run --locked --with uvicorn uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000
```

**3. Start the worker**, in a second terminal, from the same directory:

```bash
uv run --locked python -m backend.worker_main
```

Open `http://127.0.0.1:8000/docs`. A submission is `POST /uploads` → `PUT` the
bytes to the returned `upload.url` → `POST /analyze` → poll `GET /jobs/{job_id}`
→ `GET /results/{job_id}`.

Constraints worth knowing:

- **`DFP_QUEUE_KIND=memory` is the default and only works inside one process**
  (the test suite drives API and workers that way). A worker started on it
  could never receive a job, so `backend.worker_main` refuses to boot on it, and
  `DFP_ENVIRONMENT=prod` rejects it at configuration time.
- **The SQLite queue is single-host.** Its locking is only as reliable as the
  filesystem's: local disk, never NFS or SMB. The multi-host backend is meant to
  be Redis, and `jobqueue/redis.py` is still a skeleton that raises
  `NotImplementedError` — it is not a working option today.
- Job claims are atomic across processes (`BEGIN IMMEDIATE` transactions), a
  worker that dies mid-job has its lease expire and the job retried with backoff,
  and a job is dead-lettered after its attempt budget or on a permanent error
  such as an undecodable file. Budgets are per queue (`QUEUE_DEFAULTS` in
  `jobqueue/registry.py`): image jobs get 3 attempts with jittered backoff from
  2 s up to 30 s; video jobs, which cost far more to rerun, get 2 attempts with
  backoff from 10 s up to 120 s. See `jobqueue/sqlite.py`.

## Running the test suites

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
