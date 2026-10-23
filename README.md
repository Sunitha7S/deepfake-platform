# Deepfake Platform

Image and video deepfake detection. Web application, no terminal workflow.

**Status:** the API, job queue, object storage, worker and inference pipeline are
built and run end to end (see [Running the platform locally](#running-the-platform-locally)).
Scoring is a **deterministic stub**, not a trained model — a verdict reflects the
file's hash, not its authenticity. There is no frontend yet. Only the local
storage backend and the in-memory and SQLite queues work: `storage/s3.py` and
`jobqueue/redis.py` are skeletons that raise `NotImplementedError`. The face
detector is OpenCV's Haar cascade; an optional SCRFD detector is wired in but
not the default (see `docs/SCRFD_R13_MEASUREMENT_EVIDENCE.md`).

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
| Branch P — provenance | C2PA + EXIF. Not learned. **One-sided.** Target: no C2PA or EXIF reader exists yet; every bundle carries an empty provenance report. |
| Fusion | Logistic regression with interaction terms, temperature-calibrated. Target: today a weighted pool of branch log-odds minus the provenance credit, at the identity temperature (`inference/postprocess.py`). |
| Verdict | Three states: `authentic` / `manipulated` / `insufficient_evidence` |
| Upload | Direct to object storage via presigned URL; bytes never touch the API. Target: the only working backend is local storage, whose presigned URL is the API's own `PUT /uploads/{upload_id}`. |

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
contracts/    EvidenceBundle and every cross-boundary schema, fixtures, version strings
backend/      FastAPI app, routes, settings, rate limiter, worker entry point (worker_main)
jobqueue/     queue protocol; memory and SQLite backends, Redis skeleton
storage/      storage protocol and URL signing; local backend, S3 skeleton
workers/      composition root: queued job → bytes from storage → EvidenceBundle
inference/    decode, face detection, preprocessing, stub model backend, fusion
evaluation/   offline metrics, calibration, benchmarking (never imported by serving)
tests/        every suite, run together by tests/run_all.py
docs/         per-batch design records; REPRODUCIBILITY_AUDIT.md is the finding index
```

## Setup

Install [uv](https://docs.astral.sh/uv/) once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`)

Then, from the repo root:

```bash
uv sync --locked --extra dev --extra onnx
```

`--extra onnx` installs `onnxruntime`, which the SCRFD detector suite needs
even though the shipped default detector does not. This is the same command the
CI workflow runs.

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
DFP_RATE_LIMIT_KIND=sqlite
```

The second line matters only when you run more than one API process; it is
required with `DFP_ENVIRONMENT=prod`. Everything else can stay at its default,
which the processes then share:

| Setting | Default | What lives there |
|---|---|---|
| `DFP_QUEUE_PATH` | `./var/queue.sqlite3` | The job queue: one SQLite file, plus its `-wal`/`-shm` side files |
| `DFP_STORAGE_ROOT` | `./var/storage` | Uploaded files, upload sessions, results |
| `DFP_RATE_LIMIT_PATH` | `./var/ratelimit.sqlite3` | Per-client request counts, shared by every API process |

Relative paths resolve against **each process's own working directory**. Start
both from the repo root, or set both variables to absolute paths. To confirm
the processes agree, compare the `queue_backend` field in the API's
`dependencies checked` log line with the one in the worker's `worker starting`
line: they must name the same file. `var/` is gitignored; delete it to reset all
local state.

**2. Start the API.** uvicorn is not a declared dependency, so `--with` adds it
to a temporary overlay without touching `uv.lock`:

```bash
uv run --locked --with uvicorn uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000 --no-proxy-headers
```

`--no-proxy-headers` is not optional. By default uvicorn trusts
`X-Forwarded-For` from 127.0.0.1 and rewrites the client address with it before
the app sees the request, so a local caller could pick a new rate-limit
identity per request (measured: five spoofed requests all passed a limit of
three without the flag; with it, the fourth was refused).

**3. Start the worker**, in a second terminal, from the same directory:

```bash
uv run --locked python -m backend.worker_main
```

Open `http://127.0.0.1:8000/docs`. A submission is `POST /uploads` → `PUT` the
bytes to the returned `upload.url` → `POST /analyze` → follow the returned
`events_url` (`GET /events/{job_id}`, Server-Sent Events) or poll
`GET /jobs/{job_id}` → `GET /results/{job_id}`. The stream sends the same
`JobStatus` the poll returns: a `state` frame first, `done` or `error` last, and
a `heartbeat` during silence. The server re-derives the status every
`DFP_EVENTS_POLL_INTERVAL_S` (1 s) and sends a heartbeat after
`DFP_EVENTS_HEARTBEAT_S` (15 s) without a frame. It closes after its final
frame, or after `DFP_EVENTS_MAX_DURATION_S` (600 s) if the job never finishes.
A stream that ends without `done` or `error` (that limit, the job being purged,
a dropped connection) says nothing about the job: reconnect, which starts again
from a fresh `state` frame (there is no `Last-Event-ID` resume), or poll.
Progress is as coarse as the poll's (state changes, not per-frame percentages).

Constraints worth knowing:

- **`DFP_QUEUE_KIND=memory` is the default and only works inside one process**
  (the test suite drives API and workers that way). A worker started on it
  could never receive a job, so `backend.worker_main` refuses to boot on it, and
  `DFP_ENVIRONMENT=prod` rejects it at configuration time.
- **The SQLite queue is single-host.** Its locking is only as reliable as the
  filesystem's: local disk, never NFS or SMB. The multi-host backend is meant to
  be Redis, and `jobqueue/redis.py` is still a skeleton that raises
  `NotImplementedError` — it is not a working option today.
  `DFP_ENVIRONMENT=prod` refuses it at startup, and likewise refuses
  `DFP_STORAGE_KIND=s3`, whose backend is also a skeleton.
- Job claims are atomic across processes (`BEGIN IMMEDIATE` transactions), a
  worker that dies mid-job has its lease expire and the job retried with backoff,
  and a job is dead-lettered after its attempt budget or on a permanent error
  such as an undecodable file. Budgets are per queue (`QUEUE_DEFAULTS` in
  `jobqueue/registry.py`): image jobs get 3 attempts with jittered backoff from
  2 s up to 30 s; video jobs, which cost far more to rerun, get 2 attempts with
  backoff from 10 s up to 120 s. See `jobqueue/sqlite.py`.
- **Rate limiting covers `POST /uploads` and `POST /analyze`**, which share
  one budget of `DFP_RATE_LIMIT_PER_MINUTE` (60) requests per client per fixed
  one-minute window. A full submission therefore costs two. Polling, the event
  stream, results, `/health` and the byte `PUT` are not limited. Over the limit
  the answer is `429 rate_limited` (`error.retryable: true`) with
  `Retry-After` set to the seconds left in the window (1–60, also in
  `error.retry_after_s`). Across a window boundary
  a client can send up to twice the limit in quick succession.
- **A client is its direct peer address.** `X-Forwarded-For`, `Forwarded` and
  `X-Real-IP` are ignored, because no trusted proxy is configured and a client
  can set those headers to anything. Behind a reverse proxy every caller
  therefore shares the proxy's one budget.
- **`DFP_RATE_LIMIT_KIND=memory` (the default) counts per process**, so N API
  processes allow N times the limit; `sqlite` shares one count on one host
  (atomic `BEGIN IMMEDIATE` upserts, local disk only). **Limiting across
  hosts is not supported.** If the counter file cannot be reached, limited
  requests fail with `500 internal_error` rather than pass unchecked.

## Running the test suites

```bash
uv run python tests/run_all.py
```

Runs every suite (64 suites, 1,486 tests at the time of writing). It takes
15–20 minutes, most of it `test_evaluation_operating`'s bootstrap resampling;
that suite is slow, not hung. Each suite also runs on its own, for example:

```bash
uv run python tests/test_contracts.py
```

It needs no pytest, and runs under `pytest` unchanged.

Generate fixtures for the frontend:

```bash
python -m contracts.fixtures --out fixtures_out
```

Print one:

```bash
python -m contracts.fixtures --stdout video_manipulated
```

## Requirements

Python 3.12 (pinned by `.python-version`). Runtime dependencies are declared in
`pyproject.toml` and locked in `uv.lock`: pydantic, numpy, opencv-python,
FastAPI, starlette and pydantic-settings. torch and onnxruntime are optional
extras, imported only when a model backend or the SCRFD detector asks for them.
`contracts/` alone still needs only pydantic.

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

Not built yet: a trained model, a frontend, the Redis queue, the S3 backend,
provenance extraction, and a video face stage. The CI workflow
(`.github/workflows/tests.yml`) has never run, because the repository has no
remote. Open findings are tracked in `REPRODUCIBILITY_AUDIT.md`; the remaining
programs are described in `docs/NEXT_ENGINEERING_PROGRAM.md`.
