# Batch 24 — N5 Design: Dependency Lockfile

**Type:** design only. No code, no implementation.
**Follows:** `docs/B24_R2_DESIGN.md` §7.2–8, which concluded N5 (not R30) is the
correct next batch and asked for exactly this document before implementation
starts.
**Goal restated:** reproducible dependency resolution across machines, with
**no runtime-identity movement** — `.d.`, `.f.`, `weights_identity`,
`STUB_ENGINE_VERSION`, and every version constant must be provably unchanged.

---

## 1. Re-derivation from source

### 1.1 Current dependency management

`pyproject.toml` (full, live-read):

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "deepfake-platform"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.9,<3",
    "numpy>=1.26,<2",
    "opencv-python>=4.8,<5",
    "fastapi>=0.115",
    "starlette>=0.37",
    "pydantic-settings>=2.4,<3",
]

[project.optional-dependencies]
torch = ["torch>=2.4"]
onnx = ["onnxruntime>=1.18"]
dev = ["pytest>=8", "ruff>=0.6", "psutil>=5.9", "pillow>=10"]

[tool.setuptools]
packages = ["backend", "contracts", "evaluation", "inference", "jobqueue", "storage", "workers"]

[tool.ruff]
line-length = 100
target-version = "py311"
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Every declaration is a **range**, not a pin. `[project.dependencies]` is
derived and *asserted*, not hand-maintained: `tests/import_audit.py` (`:260-262`)
reads this table with stdlib `tomllib` and cross-checks it against an AST scan
of every module-scope import (`tests/test_dependencies.py`, the R7 suite). That
machinery checks **package names and classification** (runtime / optional /
test-only) — **it asserts nothing about versions.**

### 1.2 Absence of a lockfile — confirmed

```
find . -iname "*.lock" -o -iname "requirements*.txt" -o -iname "constraints*.txt" \
       -o -iname "Pipfile*" -o -iname "poetry.lock" -o -iname "uv.lock"
→ (nothing)
```

No lockfile of any kind exists, tracked or otherwise.

### 1.3 CI dependency installation path — none exists

```
find . -iname "*.yml" -o -iname "*.yaml"   → (nothing)
ls .github/                                → (no such directory)
find . -iname "dockerfile*" -o -iname "docker-compose*"  → (nothing)
git remote -v                              → (no remotes)
git log                                    → "does not have any commits yet"
```

There is **no CI configuration, no Dockerfile, no git remote, and no commit
history** — the repository is a local, uncommitted working tree. "The CI
dependency installation path" is not merely undocumented; it does not exist to
document. Any CI artifact this design proposes is **prospective** — correct for
whenever the repo gains a remote, not something that runs today.

### 1.4 Developer installation path — undocumented

`README.md`'s only runnable instructions are:

```
python tests/test_contracts.py
python -m contracts.fixtures --out fixtures_out
```

**There is no `pip install` step anywhere in the README or `docs/`.** A new
developer has no documented way to obtain a working environment; today it works
only because whoever's machine runs the tests happens to have the right
packages — the exact "silent, machine-specific" failure mode `pyproject.toml`'s
own comment (`:9-13`) describes for the *pre-B5* undeclared-numpy defect,
recurring one layer down at the *version* level instead of the *name* level.

Separately: `README.md` is stale in an unrelated way ("Status: S0 complete" — the
repo is actually past S8 Batch 23). **Out of scope for N5**; not touched.

### 1.5 Deployment installation path — does not exist

No Dockerfile, no deployment manifest, no `requirements.txt` for a container
build. There is nothing to change here either — only something to *provide for
the first time*, which this design scopes narrowly (§3).

### 1.6 What is actually installed on the box the shipped identities were computed against

*(live)* — this is the load-bearing fact for §4:

| Package | Resolved version | Feeds an identity? |
|---|---|---|
| `opencv-python` | **4.8.1.78** (`cv2.__version__` reports `4.8.1`) | **Yes** — `DecodeEnvironment.identity`'s `opencv` key, and indirectly `ffmpeg_version` (the bundled prebuilt FFmpeg ships with this wheel version) |
| `numpy` | **1.26.4** | Indirectly — `pyproject.toml:20-24`'s own comment: numpy 2.x changed scalar `repr`, and `contracts.version._canonical` fingerprints float `repr` "on purpose" (feeds `weights_identity`/gate/calibration fingerprints wherever a float is hashed) |
| `pydantic` | 2.12.5 | No — not in any `fingerprint()` input |
| `fastapi` | 0.141.1 | No |
| `starlette` | 1.1.0 | No |
| `pydantic-settings` | 2.14.1 | No |
| `pytest` / `ruff` / `psutil` / `pillow` | 9.1.1 / 0.16.6 / 7.2.2 / 12.2.0 | No — dev/test-only |

`uv 0.11.28` is present on this box (`uv --version`); `pip-compile` is not
(`pip-compile: command not found`).

### 1.7 A terminology collision that must be named explicitly

`evaluation/split_lock.py:81` defines `LOCK_VERSION = "1.1.0"` — the schema
version of the **corpus split lock** (a dataset-membership artifact). This is
**unrelated** to the dependency lockfile this design produces. §4 verifies
`LOCK_VERSION` (the split-lock schema) is unmoved; nowhere in this document does
"lock" refer to `evaluation/split_lock.py` unless stated. The new artifact is
called **the dependency lock** or **`uv.lock`** throughout.

---

## 2. Candidate approaches

| | **A. `uv lock`** | **B. pip-tools / `pip-compile`** | **C. Poetry** | **D. `pip freeze`** |
|---|---|---|---|---|
| **Reads existing `pyproject.toml` unmodified** | **Yes** — PEP 621 `[project]` tables, no restructuring | Yes, `pip-compile pyproject.toml` supports PEP 621 as of pip-tools 6.x+ | Partial — Poetry conventionally wants `[tool.poetry.dependencies]`; PEP 621 support exists but extras/groups still favour Poetry's own tables | N/A — not a compiler, just a snapshot |
| **`import_audit.py` / `test_dependencies.py` impact** | **None** — verified live: `declared_dependencies()` (`import_audit.py:260-262`) parses `[project]` with stdlib `tomllib`; `uv lock` adds a *sibling* `uv.lock` file and touches nothing in `[project]` | None if using the PEP 621 mode | Likely **requires touching `[project.optional-dependencies]`/`[tool.poetry]`**, which `declared_dependencies()` would then mis-parse — a hidden test-breakage risk | N/A |
| **Cross-platform lock (Windows dev + Linux deploy)** | **One file** — `uv.lock` resolves and stores wheels for every relevant platform/marker in a single TOML | Needs **separate per-platform requirement files** (`pip-compile` does not natively multiplex platforms in one output) — a concrete, repo-relevant maintenance cost, since dev here is Windows and deployment (whenever it exists) is presumably Linux | `poetry.lock` is cross-platform-aware (similar to uv) | No — a `pip freeze` is single-machine, single-platform by construction |
| **Reproducibility (hash pinning)** | Yes — every entry hash-pinned | Yes, with `--generate-hashes` | Yes | No — freeze has no hashes and includes whatever happens to be on the freezing machine (test-only packages, transitive noise) — **the exact "hand-maintained list fails silently" anti-pattern `pyproject.toml`'s own comment (§1.1) already warns against, recurring at the version layer** |
| **Maintenance burden** | Low — `uv lock` regenerates in seconds; `uv lock --upgrade` for deliberate bumps | Medium — separate `requirements.in`/platform files to keep in sync | Medium-high — a second dependency-declaration surface to keep in sync with `[project]`, plus its own venv/build model | High — must be manually re-frozen, no diff-friendly provenance, drifts silently |
| **CI friendliness** | High — single static binary, `curl \| sh` or a pinned Docker layer, `uv sync --locked` is one step, very fast (no separate cache warmup typically needed) | Medium — needs `pip install pip-tools` first (its own bootstrap-reproducibility problem: what pins pip-tools?), then a second install step | Medium — needs the Poetry installer, slower first-run | N/A |
| **Developer ergonomics** | `uv sync` replaces "hope you installed the right thing"; **already installed on this box** (`uv 0.11.28`) | `pip install pip-tools && pip-compile && pip install -r requirements.txt` — three tool invocations for one outcome | `poetry install` — fine, but is a bigger tool to onboard for a project that otherwise uses plain `setuptools` + `pytest` | `pip install -r requirements.txt` — simple, but the file behind it is the problem |
| **Deployment ergonomics** | `uv sync --frozen` in a Dockerfile, or `uv export --format requirements-txt` to hand a plain-pip deployment a hash-pinned `requirements.txt` — **not locked into a `uv`-only runtime** | Native — deployment already knows `pip install -r requirements.txt` | `poetry export` gives the same fallback, at the cost of Poetry's own venv model in between | N/A |
| **Build-backend impact** | **None** — `[build-system]` stays `setuptools.build_meta` | None | Poetry conventionally pairs with `poetry-core` as the build backend — adopting Poetry "just for locking" while keeping `setuptools` is an unusual, friction-prone hybrid | N/A |
| **Lockfile stability (diff noise on trivial changes)** | Low — uv's TOML lock is structured per-package, stable ordering | Low-medium — `requirements.txt` output is a flat, less structured format but stable enough | Low | High — full re-freeze reorders/rewrites everything |

### Recommendation: **A — `uv lock`**

The deciding factors, all re-derived rather than assumed:

1. **Zero impact on `import_audit.py`/`test_dependencies.py`** — verified by
   reading the parser (`tomllib.loads` over `[project]`), not inferred from
   uv's marketing. This is the single largest practical differentiator versus
   Poetry, which risks silently breaking the R7 suite's TOML assumptions.
2. **One lockfile, many platforms** — this repository is developed on Windows
   (this box) and will deploy on Linux (the only realistic target, given
   `opencv-python` and FFmpeg prebuilt binaries). `uv.lock`'s per-platform
   marker resolution is a direct, cited need; pip-tools would require splitting
   into multiple requirement files to get the same guarantee.
3. **No new bootstrap-reproducibility problem.** pip-tools and Poetry both need
   themselves installed before they can pin anything else — a lockfile tool that
   itself has no lock is a smaller version of the exact problem N5 exists to
   solve. `uv` ships as a single static binary with no Python-environment
   dependency (installable via a pinned curl script or a pinned container
   image), which is the cleanest available answer to "what pins the pinning
   tool."
4. **Already present in this environment** (`uv 0.11.28`) — not decisive on its
   own, but a genuine zero-friction signal for the immediate next step.
5. **No `[build-system]` change** — `setuptools.build_meta` is untouched; N5
   stays a pure addition.

**D (`pip freeze`) is rejected outright** — it reproduces, at the version layer,
the exact defect `pyproject.toml`'s own comment (§1.1) records the project
already learned from once (a hand-maintained list that "fails silently and only
for the next person").

---

## 3. Exact scope

### 3.1 Production files affected

| File | Change |
|---|---|
| `pyproject.toml` | **Additive only.** No existing table is restructured. Adds, if needed, a small `[tool.uv]` section carrying `override-dependencies = ["opencv-python==4.8.1.78", "numpy==1.26.4"]` — see §4.1 for why this override is required, not optional. `[project.dependencies]`, `[project.optional-dependencies]`, `[build-system]`, `[tool.setuptools]`, `[tool.ruff]`, `[tool.pytest.ini_options]` are **byte-identical**. |
| `uv.lock` **(new)** | The generated lock: every runtime and `dev`/`torch`/`onnx` package, resolved and hash-pinned, for the platforms the project targets (Windows dev + Linux deploy, `manylinux` + `win_amd64` wheel tags). Committed to version control (it is source, not build output — same status as `pyproject.toml`). |

**Nothing in `inference/`, `backend/`, `contracts/`, `evaluation/`, `workers/`,
`jobqueue/`, `storage/` changes.** No source file is touched.

### 3.2 CI files affected

| File | Change |
|---|---|
| `.github/workflows/tests.yml` **(new)** | A minimal, prospective workflow: install `uv`, `uv sync --locked` (fails if `uv.lock` and `pyproject.toml` have drifted), `uv run python tests/run_all.py`. **Does not execute today** — no git remote exists (§1.3) — but is authored now so CI is correct the moment the repo is hosted, and so this design's "CI dependency installation path" question has a concrete, reviewable answer rather than a deferred one. |

### 3.3 Developer workflow changes

- `README.md` gains a **"Setup"** section (new content, inserted before
  "Running"): `curl -LsSf https://astral.sh/uv/install.sh | sh` (or the
  Windows equivalent) once, then `uv sync --extra dev` to create `.venv/` and
  install from `uv.lock`. Replaces the previously-undocumented "however your
  machine happens to have it" path.
- `.gitignore` needs **no change** — `.venv/` is already ignored (`:5-6`);
  `uv.lock` must **not** be ignored (it is committed source, like a lockfile
  always is) and nothing in the current `.gitignore` would exclude it.
- No change to how tests are invoked: `python tests/run_all.py` keeps working
  unchanged for anyone with an environment already satisfying `pyproject.toml`
  (a lockfile does not forbid an ad-hoc `pip install -e .`); `uv run python
  tests/run_all.py` is the new *reproducible* invocation.

### 3.4 Documentation changes

- `README.md` — the new Setup section (§3.3).
- `REPRODUCIBILITY_AUDIT.md` — N5 summary row and "Still open" list updated to
  **Closed in B24**, cross-referencing this document; no other row changes
  (§7).
- `docs/B24_N5_DESIGN.md` — this document.
- **Not touched:** `docs/RUN_IDENTITY.md`, `docs/DECODE_IDENTITY_COMPLETENESS.md`,
  `docs/VIDEO_SELECTION_IDENTITY.md`, `docs/B24_DESIGN.md`,
  `docs/B24_ADVERSARIAL_REVIEW.md`, `docs/B24_R2_DESIGN.md` — none describes
  dependency management; none needs a decode/identity correction from this
  batch, because no identity moves (§4).

### 3.5 What does NOT change — explicit list

- `[project.dependencies]` / `[project.optional-dependencies]` content or
  structure.
- `[build-system]` (`setuptools.build_meta`).
- Any file under `backend/`, `contracts/`, `evaluation/`, `inference/`,
  `jobqueue/`, `storage/`, `workers/`.
- Any test's *behavioural* assertions — only new tests are added (§6); no
  existing test is edited.
- `README.md`'s existing content (only an insertion, nothing removed or
  reworded — the S0-vs-S8 staleness is explicitly left alone, out of scope).
- The resolved version of any already-installed package on any machine that
  currently passes the suite — the lock is built to match, not to move, the
  installed set (§4.1).
- `EvidenceBundle`, `ThresholdInfo`, `CONTRACT_VERSION`, the OpenAPI snapshot
  structure — none of this batch's changes are wire-visible.

---

## 4. Identity and version analysis

### 4.1 The load-bearing constraint: the lock must reproduce, not upgrade

`uv lock` run for the first time (no pre-existing `uv.lock`) resolves each
range to the **highest version currently available on PyPI that satisfies it**
— not "whatever is installed." For `"opencv-python>=4.8,<5"`, run today, that
is almost certainly a materially newer release than the `4.8.1.78` the shipped
`a97ccb68` decode identity was computed against (§1.6). **A naive first lock
would therefore itself move `.d.`** — turning N5 into an undisclosed
`.d.`-moving batch, which directly contradicts this document's stated goal and
`docs/B24_R2_DESIGN.md`'s explicit framing of N5 as an identity-neutral batch.

**Required correction, built into scope (§3.1):** `[tool.uv].override-
dependencies` pins `opencv-python==4.8.1.78` and `numpy==1.26.4` — the two
packages that feed a fingerprint (§1.6) — to the exact versions already
installed and already reflected in the shipped literals. `uv` resolves every
other package (`pydantic`, `fastapi`, `starlette`, `pydantic-settings`, and the
`dev`/`torch`/`onnx` extras) to their latest-compatible-at-lock-time versions,
since none of them feeds any `fingerprint()` call — resolving those freely is
exactly the reproducibility win N5 exists to deliver, with no identity
consequence.

This override is not a permanent pin on the *declared range* — `[project.
dependencies]`'s `opencv-python>=4.8,<5` is untouched (§3.1), preserving the
intent ranges exist for. It is a pin on the *lock's resolution*, load-bearing
until a deliberate future decision to move it — which, per §7, is exactly the
trigger for the deferred "decode-completeness pass 2" batch
(`docs/B24_R2_DESIGN.md` §5), not this one.

### 4.2 Per-quantity verification

| Quantity | Verdict | Evidence |
|---|---|---|
| **`.d.` (decode identity)** | **UNCHANGED — `a97ccb68`** | `DecodeEnvironment.identity` (`inference/decode_env.py:181-234`) is computed at *runtime* from `cv2.__version__`, the resolved video backend, and `getBuildInformation()` — none of which this batch's files (`pyproject.toml`, `uv.lock`) are read by. The lock only constrains *which* `opencv-python` gets *installed*; §4.1's override ensures that stays `4.8.1.78`, so `cv2.__version__` stays `"4.8.1"` and every parsed FFmpeg/build string stays byte-identical. No line of `inference/` is touched. |
| **`.f.` (selection composite)** | **UNCHANGED — `e629642c`** | `selection_identity(face, video)` (`inference/preprocess.py`) depends on `FaceSelectionPolicy` / `VideoSelectionPolicy` field values, none of which are package-version-derived. Not read by this batch. |
| **`weights_identity`** | **UNCHANGED — `stub-0-wacdd9240`** | Derived from `ModelSpec` fields (`inference/models.py`), not from any installed package version. |
| **`STUB_ENGINE_VERSION`** | **UNCHANGED** — `stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.a97ccb68` | It is `compose_runtime_version(model_identity, decode_identity())` (`backend/config.py:41`); both halves are unchanged (rows above). |
| **`CONTRACT_VERSION`** | **UNCHANGED — `1.0.0`** | `contracts/enums.py` — a hand-set schema constant, not derived from any dependency. Not read or written by this batch. |
| **`RUN_SCHEMA_VERSION`** | **UNCHANGED — `1.2.0`** | `evaluation/benchmark.py` — same reasoning. |
| **`CALIBRATION_WITNESS_VERSION`** | **UNCHANGED — `1.2.0`** | `evaluation/calibration_protocol.py` — same reasoning. |
| **`LOCK_VERSION`** | **UNCHANGED — `1.1.0`** *(the split-lock schema, §1.7 — not the new dependency lock)* | `evaluation/split_lock.py:81` — a corpus-membership schema constant. This batch does not import, read, or modify `evaluation/split_lock.py`. No source-derived argument requires movement; none is proposed. |
| **`numpy` float-repr fingerprint inputs** | **UNCHANGED** — `numpy==1.26.4` pinned by the override (§4.1) | `pyproject.toml:20-24`'s own comment: a numpy 2.x promotion/repr change would move fingerprints that hash a float `repr`. Pinning `numpy` in the lock to the exact installed 1.x version this design also verified (§1.6) removes the one dependency-version axis with a **documented** fingerprint effect. |
| **OpenAPI snapshot / `schema_out` / `fixtures_out`** | **UNCHANGED** | None of these are generated from, or sensitive to, installed package *versions* (they are generated from Pydantic model *definitions* in `contracts/`, which this batch does not touch); `fastapi`/`pydantic` version drift *could* in principle change generated-schema cosmetics, which is exactly why §6 adds a regen-diff test as a non-movement proof rather than an assumption. |

**Summary: every identity and version constant in the platform is unchanged,
verified against source, with one explicit, load-bearing mechanism
(§4.1's override) that makes the "unchanged" claim true by construction rather
than by luck.**

---

## 5. Migration analysis

| Question | Answer | Reasoning |
|---|---|---|
| **Cache generations** | **Zero.** | No `.d.`/`.f.`/weights/`STUB_ENGINE_VERSION` movement (§4) → no content-cache key changes → nothing to regenerate, no namespace impact, no orphaned entries. |
| **Startup alignment impact** | **None.** | `workers.engine_registry.verify_version_alignment` and `backend.app`'s lifespan format check compare version *strings*; neither reads `pyproject.toml` or `uv.lock`. A worker built from the locked environment reports the same `engine_versions()` as one built ad hoc today (`opencv-python` pinned identically by §4.1). |
| **Benchmark impact** | **None.** | `BenchmarkRun.decode_identity` / `effective_model_version` are unaffected values; `RUN_SCHEMA_VERSION` unbumped; no stored row's meaning changes. |
| **Witness impact** | **None.** | `CalibrationWitness` digests cover `decode_identity` and calibration fields, both unchanged; `CALIBRATION_WITNESS_VERSION` unbumped. |
| **Rollback procedure** | Delete `uv.lock`, revert the `[tool.uv]` addition in `pyproject.toml`, revert `.github/workflows/tests.yml` and the README section. No running system is affected — there is nothing to "roll back" in a deployed sense, because nothing computed a new identity or wrote a new cache entry. The only observable effect of rollback is developers/CI returning to unpinned, range-resolved installs. |
| **Deployment procedure** | **N/A today** (§1.5 — no deployment artifact exists). For whenever one is authored: `uv sync --frozen` (fails closed if the environment can't match the lock exactly) in the image build step, or `uv export --format requirements-txt --no-hashes` piped into a plain `pip install -r -` for a base image without `uv`. Either path installs the *same* `opencv-python==4.8.1.78` / `numpy==1.26.4` this design verified, so a freshly built image reports the same `engine_versions()` as this box — the property N5 exists to guarantee. |

**Net migration cost: none of the kind B20/B22/B23/the-deferred-R30-batch pay.**
No coordinated rollout, no `DFP_SKIP_VERSION_ALIGNMENT` window, no `.d.`/`.f.`
runbook line. This is the property that made N5 rank #1 in
`docs/B24_R2_DESIGN.md` §7.

---

## 6. Test plan

### 6.1 Fail-before matrix (real failures against the pre-N5 tree)

| # | Test | Pre-N5 behaviour |
|---|---|---|
| FB1 | `test_a_lockfile_exists` | **fails** — no `uv.lock` in the repo (`Path("uv.lock").exists()` is `False`) |
| FB2 | `test_the_lockfile_pins_the_versions_the_shipped_identity_was_computed_against` | **fails** — same reason; the assertion cannot even run without a lock to parse |
| FB3 | `test_a_fresh_resolve_would_not_silently_move_opencv_or_numpy` | **fails today in spirit** — nothing prevents `pip install -e .` on a machine from resolving a different `opencv-python`/`numpy` than `4.8.1.78`/`1.26.4`; there is no override anywhere (`grep 'override-dependencies' pyproject.toml` → nothing) |
| FB4 | `test_the_ci_workflow_exists_and_installs_from_the_lock` | **fails** — no `.github/workflows/*.yml` exists at all |

### 6.2 Pass-after matrix (real protections after N5)

| # | Test | Assertion |
|---|---|---|
| PA1 | `test_a_lockfile_exists_and_matches_pyproject` | `uv.lock` exists; `uv lock --locked --check` (or the equivalent programmatic check) reports no drift against `pyproject.toml` |
| PA2 | `test_the_lockfile_pins_the_versions_the_shipped_identity_was_computed_against` | parses `uv.lock`'s TOML; asserts the locked `opencv-python` version `== "4.8.1.78"` and the locked `numpy` version `== "1.26.4"` — **this is the test that makes §4's "unchanged" claim self-verifying rather than asserted once and left to rot** |
| PA3 | `test_the_installed_environment_matches_the_lock` | `importlib.metadata.version("opencv-python") == "4.8.1.78"` and same for `numpy`, run as part of `tests/run_all.py` — catches a developer/CI box whose environment drifted from `uv sync` |
| PA4 | `test_cv2_version_matches_what_the_shipped_decode_identity_assumes` | `cv2.__version__ == "4.8.1"` — ties PA2/PA3 directly to the *live* `decode_identity()` input, closing the loop from "the lock says X" to "the running process actually reports X" |
| PA5 | `test_the_override_pins_are_documented` | `pyproject.toml`'s `[tool.uv]` section carries a comment referencing this design doc and `docs/B24_R2_DESIGN.md`'s deferred pass-2 batch, so a future reader finds the reason rather than an unexplained pin |
| PA6 | `test_the_ci_workflow_installs_from_the_lock_with_locked_flag` | the workflow YAML's install step uses `--locked`/`--frozen` (fails the CI job on drift) rather than a bare `uv sync` (which would silently update the lock) |

### 6.3 Non-movement matrix

| # | Test | Assertion |
|---|---|---|
| NM1 | `test_decode_identity_is_unchanged` | `decode_identity() == "a97ccb68"` |
| NM2 | `test_stub_engine_version_is_unchanged` | `STUB_ENGINE_VERSION == "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.a97ccb68"` |
| NM3 | `test_model_identity_is_unchanged` | `model_identity_of(STUB_ENGINE_VERSION) == "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c"` |
| NM4 | `test_no_contract_schema_witness_or_split_lock_version_bump` | `CONTRACT_VERSION == "1.0.0"`, `RUN_SCHEMA_VERSION == "1.2.0"`, `CALIBRATION_WITNESS_VERSION == "1.2.0"`, `evaluation.split_lock.LOCK_VERSION == "1.1.0"` — named explicitly to defeat the §1.7 terminology collision |
| NM5 | `test_import_audit_is_unaffected_by_the_lockfile` | `tests/import_audit.py`'s `declared_dependencies()` / `build_matrix()` outputs are byte-identical before and after `uv.lock` exists (proves §2's "zero `import_audit.py` impact" claim empirically, not just by inspection) |
| NM6 | `test_openapi_and_schema_out_regen_are_lock_independent` | `python -m backend.openapi --update-snapshot` and `python -m contracts.schema` produce zero diff with the locked environment active |
| NM7 | `test_no_production_source_file_changed` | (a review-time check, not a runtime test) — diff of the batch touches only `pyproject.toml`, `uv.lock`, `.github/workflows/tests.yml`, `README.md`, `REPRODUCIBILITY_AUDIT.md`, `docs/B24_N5_DESIGN.md` |
| NM8 | `test_full_suite_is_unaffected` | `python tests/run_all.py` (or `uv run python tests/run_all.py`) → same suite/test count and 0 failures, run once under the pre-existing ad hoc environment and once under `uv sync`, compared |

### 6.4 Collateral matrix

**None.** No existing literal, fixture, docstring, or test assertion anywhere in
the repository encodes a dependency version or a lockfile fact (verified: no
prior test references `opencv-python`'s wheel string, `numpy`'s version, or any
`uv`/`pip-tools` artifact). N5 is the first batch to introduce version-pinning
assertions at all — PA2–PA4/NM1–NM4 are **new** tests, not edits to existing
ones. This absence of collateral is itself notable: it is the gap N5 closes.

---

## 7. Open-finding re-rank after N5

Re-ranked against the same five axes as `docs/B24_R2_DESIGN.md` §7, now
treating N5 as closed.

| Finding | Repro | Cache | Deploy | Migr | Cx | Actionable | Disposition after N5 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| **R13** — face-crop instability (310×, measured) | 5 | 4 | 2 | 5 | 5 | ❌ SCRFD | unaffected; still the most severe, still un-batchable |
| **R31** — witness digest omits model identity/gate/`.f.` | 3 | 3 | 3 | 3 | 3 | ✅ | **#1 next.** N5 does not touch calibration; still needs `CALIBRATION_WITNESS_VERSION` 1.3.0, still gated on a real model existing to calibrate. Ranking unchanged by N5 — it was already B24_R2's #2, and N5's closure removes nothing that blocked it. |
| **OR1-b** — `backend/worker_main.py` composition root | 1 | 2 | 3 | 1 | 1 | ✅ | **#2.** Still scheduled, still zero cache generation. N5 slightly *sharpens* its case: with dependency drift now closed off, a mismatched `configured_model_version` is more likely to come from a config/deploy error than a silent environment drift — exactly the class of error OR1-b's stronger config-default check catches. |
| **R30-bundle** (R30 + D8 + D9 + D10-residual) | 2 | 2 | 3 | 2 | 3 | ✅, but low value (`B24_R2_DESIGN.md` §1/§2 — measured pixel-neutral for accepted MIME) | **#3, and now safely sequenced.** This was the entire reason N5 had to come first: any `.d.`-moving batch's literal is only as reproducible as the environment it was computed in. With `opencv-python`/`numpy` pinned in `uv.lock`, the pass-2 batch's own `.d.` regeneration is now itself reproducible across machines — N5 removes exactly the failure mode that made rushing R30 risky. |
| **D9** — `cv2.__version__` drops the wheel's 4th digit | 2 | 2 | 2 | 2 | 2 | mostly closed | N5's override pins the **full** wheel version intent (`4.8.1.78`) even though `identity` still only reads `cv2.__version__` (`"4.8.1"`) — the *lock* now records the exact wheel PA2 checks, so a drift is caught by PA2/PA3 before it ever reaches `decode_identity()`. The residual (identity itself still can't distinguish `.78` from a hypothetical `.79`) is unchanged and stays with the pass-2 batch. |
| **N6 / N7** — fit-optimiser config / `np.quantile` method unrecorded | 2 | 1 | 1 | 3 | 2 | ✅ with R31 | unaffected; folds into R31's batch |
| **R32 / R34 / N8 / N3 / R24 / R23 / R28 / R18 / R26 / R16 / D10** | — | — | — | — | — | unchanged | none of these findings involve dependency versions; N5 has no interaction with any of them. Dispositions from `docs/B24_R2_DESIGN.md` §7 stand without modification (N8 closed-accepted-cost; the rest tracked/blocked as previously derived). |

### 7.1 What is the batch after N5?

**R31** (witness-completeness), gated on a real model existing to calibrate —
or, if the team wants another zero-risk, zero-cache-generation batch first,
**OR1-b**. The R30-bundle (pass-2 decode completeness) is next in line after
either of those, now that N5 makes its migration reproducible across machines.
No finding is reprioritized *above* these three by N5 landing — N5's effect is
entirely to de-risk what comes after it, not to promote anything new.

---

## 8. Final gate

## **APPROVE WITH CHANGES**

The lockfile approach, scope, and non-movement analysis are sound and
source-verified. Two items must be resolved before implementation, both
narrowly scoped:

| # | Severity | Defect | Evidence | Impact | Required correction |
|---|---|---|---|---|---|
| G1 | **High (blocking)** | The override pins (§4.1) are **specified but not yet executed and verified**. A first `uv lock` run, *without* confirming the override actually lands `opencv-python==4.8.1.78`/`numpy==1.26.4` (rather than, say, uv reporting the override is unsatisfiable against `>=4.8,<5` for some reason not yet observed, e.g. that exact wheel being pulled from PyPI), would be the one way this batch could silently move `.d.` — the failure mode §4.1 exists to prevent. | §4.1; no `uv.lock` exists yet to inspect. | If unverified, N5 could ship a `.d.`-moving regression under an "identity-neutral" banner. | At implementation time: run `uv lock`, then run PA2–PA4 *before* committing the lock or touching any other file. If they fail, the override syntax or PyPI availability needs adjustment — this is a one-command verification, not a design gap, but it must happen before merge, not be assumed. |
| G2 | Medium (non-blocking for N5 itself) | The CI workflow (§3.2) is authored against a repository with **no git remote and no commit history** (§1.3) — it cannot be validated by actually running it. | `git remote -v` / `git log` (§1.3). | The workflow file could contain an untested error (wrong `uv` install action version, wrong Python version pin) that only surfaces once the repo is finally pushed and Actions runs for the first time. | Author it conservatively (pin `actions/checkout@vN`, `astral-sh/setup-uv@vN`, `requires-python`'s `3.12`) and flag it in the workflow's own header comment as "unexercised — verify on first push" rather than presenting it as tested. This does not block N5's core deliverable (the lockfile), only the CI artifact's confidence level. |

Both are **specification/verification gaps**, not architecture defects — no
architecture, parser, cache key, or contract is touched anywhere in this
design, and none of the five re-derivations in §4 depends on anything other
than "the override successfully pins the two version-bearing packages," which
G1 makes an explicit, checked step rather than an assumption.

**Implementation may begin after this document, subject to G1 being satisfied
as the first concrete step of implementation** (lock, then verify PA2–PA4,
*then* proceed) rather than as a follow-up test written after the fact. **No
further design iteration is required** — there is no remaining architectural or
scope question, only the one execution-order discipline in G1.
