# OR1-b — Design: Worker Composition Root, Full Alignment Check on the Config-Default Path

**Type:** design document only. No code in this task.
**Scope:** OR1-b exclusively, per instruction. Does **not** touch, fold in, or
depend on R30, D8, D9, D10, R16, R23, R26, or R28.
**Method:** every claim re-derived from source at the cited path. The prior
scoping (`docs/BATCH_23_R3_DESIGN.md` §1.12, repeated verbatim in
`docs/B24_CANDIDATE_ANALYSIS.md`, `docs/B24_DESIGN.md` §5.8,
`docs/B24_ADVERSARIAL_REVIEW.md`, `docs/B24_R2_DESIGN.md`,
`docs/B24_N5_DESIGN.md`, `docs/R31_DESIGN.md` §10.1) is treated as a
**hypothesis to verify, not a spec to implement** — and it does not fully
survive the re-derivation. Two real gaps below were not previously identified.

---

## 1. Root cause, re-derived

### 1.1 What OR1 (Batch 23) actually shipped *(live)*

`workers/runner.py:398-471`, `verify_engine_alignment_at_startup`:

```python
def verify_engine_alignment_at_startup(configured_model_version: str | None = None) -> None:
```

Three paths, in order:
1. `DFP_SKIP_VERSION_ALIGNMENT` truthy → log a warning, return (no check at all).
2. `configured_model_version` given, or `$DFP_MODEL_VERSION` set → run the full
   `workers.engine_registry.verify_version_alignment(configured)` — the real
   cryptographic comparison against what the engines report.
3. **Neither given** → fall back to the *weak* check: `engine_versions()`,
   assert the two engines (`image`, `video`) agree **with each other**, and
   that the agreed value is `is_composed` + `is_environment_pinned`. This
   **never compares against `backend.config.STUB_ENGINE_VERSION` at all** —
   it cannot, because `workers` may not import `backend` (the DAG rule this
   whole design works around).

`workers/runner.py:474-509`, `run_with_shutdown`:

```python
async def run_with_shutdown(
    runner: WorkerRunner, *, max_jobs: int | None = None, idle_timeout_s: float | None = None
) -> WorkerReport:
    ...
    verify_engine_alignment_at_startup()
```

**Confirmed live: `run_with_shutdown` calls the gate with zero arguments.**
There is no `configured_model_version` parameter on `run_with_shutdown` at
all — the design chain's own summary of what it does
(`docs/BATCH_23_R3_DESIGN.md:111`: *"calls
`run_with_shutdown(runner, configured_model_version=get_settings().model_version, ...)`"*)
describes a call signature `run_with_shutdown` **does not have**. This is the
first gap (§4).

### 1.2 The actual gap OR1 left open

Every deployment that does **not** set `$DFP_MODEL_VERSION` (the documented
12-factor requirement, `docs/BATCH_23_R3_DESIGN.md:121-122`: *"Until B24,
production **must** set `DFP_MODEL_VERSION`"*) gets path 3 — the weak check.
Path 3 catches two failures (engines disagree with each other; the agreed
value is malformed) but is **structurally blind** to the one failure that
matters most for the config-default path: **both engines agree with each
other but disagree with `backend.config.STUB_ENGINE_VERSION`** — the literal
`Settings.model_version` defaults to and the literal the API keys its result
cache on. In that state the worker boots clean, logs an "alignment: engines
agree" info line, and **silently writes/reads the result cache under a key
the API never produces or looks up** — every job re-runs, nothing errors,
exactly the OR1 failure mode restated one layer down.

`test_configured_model_version_matches_the_engines` (a **build-time** test,
not a runtime guard) is the only thing holding this closed today — confirmed
still the only protection by `backend/config.py`'s own comment (`:227-234`):
*"the worker startup alignment gate (OR1) has no Settings field... A wired
`skip_version_alignment` field returns in B24/OR1-b together with the
`backend/worker_main.py` composition root that reads it."*

---

## 2. Current behavior vs. desired behavior

| | Current | Desired |
|---|---|---|
| `$DFP_MODEL_VERSION` set | Full cryptographic check, fail-closed | Unchanged |
| `$DFP_MODEL_VERSION` unset, engines agree with each other, agree with `STUB_ENGINE_VERSION` | Weak check passes (coincidentally correct) | Full check passes (correctly, not coincidentally) |
| `$DFP_MODEL_VERSION` unset, engines agree with each other, **disagree with `STUB_ENGINE_VERSION`** | **Weak check passes — silent cache-key mismatch, the exact OR1 failure mode** | **Fails closed, names the mismatch** |
| `$DFP_MODEL_VERSION` unset, engines disagree with each other | Fails closed (already caught) | Unchanged |
| `DFP_SKIP_VERSION_ALIGNMENT` set | No check at all, warning logged | Unchanged (escape hatch preserved) |

Desired behavior is achieved by giving the **composition root** — a process
that legitimately may import both `backend.config` (for the configured
literal) and `workers` (to run the engines) — the job neither package can do
alone: pass `backend.config.get_settings().model_version` into the gate as
`configured_model_version`, converting path 3 into path 2 for every
deployment, whether or not `$DFP_MODEL_VERSION` is set.

---

## 3. Two gaps the prior design did not resolve

### 3.1 Gap A — `run_with_shutdown` has no parameter to receive the literal

Confirmed (§1.1): `run_with_shutdown(runner, *, max_jobs=None, idle_timeout_s=None)`
calls `verify_engine_alignment_at_startup()` with no arguments. The prior
design's claim that OR1-b is *"a pure additive wiring change — no gate
rewrite"* (`docs/BATCH_23_R3_DESIGN.md:119-120`) is **only true of
`verify_engine_alignment_at_startup` itself** (which does have the parameter,
unused by its one caller) — it is **not true of `run_with_shutdown`**, which
needs a real signature change to forward anything. This is small (one new
keyword parameter, one line in its body) but it is not zero, and it was not
previously stated as a required change anywhere in the design chain.

### 3.2 Gap B — `build_all_workers` returns *two* runners; `run_with_shutdown` takes *one*

`workers/registry.py:85-92`:
```python
def build_all_workers(*, queue: QueueBackend, storage: StorageBackend) -> dict[str, WorkerRunner]:
    """One worker per queue. What a supervisor process starts."""
    return {queue_for(kind): build_worker(kind, queue=queue, storage=storage)
            for kind in (MediaKind.IMAGE, MediaKind.VIDEO)}
```

The prior design's pseudocode — *"builds the workers via
`workers.registry.build_all_workers`, and calls
`run_with_shutdown(runner, ...)`"* — uses **singular** `runner` against a
function that returns **two**. `build_all_workers`'s own docstring says it is
*"what a supervisor process starts"* — i.e. `backend/worker_main.py` is
specified to be exactly that supervisor, and nothing in any design document
says how it starts two.

**This is not cosmetic.** `run_with_shutdown`'s shutdown mechanism
(`:492-509`) is:
```python
loop.add_signal_handler(sig, runner.request_stop)
```
`asyncio.add_signal_handler` **replaces** any previously registered handler
for that signal on the same event loop. If `worker_main.py` naively calls
`run_with_shutdown` twice — once per runner, via `asyncio.gather` — **the
second call's `add_signal_handler` silently overwrites the first's.** One of
the two workers loses its graceful-drain-on-SIGTERM/SIGINT behavior entirely,
which is exactly the property `run_with_shutdown` exists to provide. Verified:
`grep` finds no existing test or code path anywhere in the repository that
runs two `WorkerRunner`s concurrently through `run_with_shutdown` — this is
new ground, not a previously-exercised pattern with a known-safe answer.

**Both gaps must be resolved by this design, not deferred**, because
`backend/worker_main.py` cannot be written correctly without deciding them.

---

## 4. Candidate designs for the two gaps

### 4.1 Gap A — how `configured_model_version` reaches the gate

**Option A1 (recommended): add a keyword parameter to `run_with_shutdown`.**
```python
async def run_with_shutdown(
    runner: WorkerRunner, *, max_jobs=None, idle_timeout_s=None,
    configured_model_version: str | None = None,
) -> WorkerReport:
    ...
    verify_engine_alignment_at_startup(configured_model_version)
```
Backward-compatible: the one existing call site
(`tests/test_startup_alignment.py:101`, `run_with_shutdown(ExplodingRunner())`)
keeps its current behavior exactly (falls through to `$DFP_MODEL_VERSION`,
then the weak check) since the new parameter defaults to `None`. Minimal
diff, no test rewrite required for the existing suite.

**Option A2 (rejected): call `verify_engine_alignment_at_startup` directly
from `worker_main.py`, before `run_with_shutdown`, and leave
`run_with_shutdown` unchanged.** Rejected because `run_with_shutdown` **always**
calls the gate itself (`:488`), unconditionally — `test_run_with_shutdown_
runs_the_gate_before_the_loop` locks this in as part of the function's
contract. Calling the gate a second time (once explicitly, once inside
`run_with_shutdown`) is redundant but not incorrect (the check is pure and
side-effect-free besides logging/raising) — but it is a worse design than A1
for no benefit, and doubles a (cheap but real) startup computation per
worker.

### 4.2 Gap B — how one process runs two `WorkerRunner`s with correct shutdown

**Option B1 (rejected): `asyncio.gather(run_with_shutdown(image, ...), run_with_shutdown(video, ...))`.**
Rejected outright — §3.2 demonstrates this silently drops one worker's signal
handler. This is not a hypothetical; it is how `asyncio.add_signal_handler`
is documented to behave, and it would ship a startup-gate feature with a
graceful-shutdown regression as its side effect.

**Option B2 (recommended): `worker_main.py` owns its own multi-runner
shutdown orchestration; `run_with_shutdown` is not called for the two-worker
path at all.**

```python
# backend/worker_main.py (new)
async def _run_all(runners: dict[str, WorkerRunner], configured_model_version, skip_version_alignment):
    verify_engine_alignment_at_startup(
        configured_model_version, skip_version_alignment=skip_version_alignment,
    )
    loop = asyncio.get_running_loop()
    installed = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda: [r.request_stop() for r in runners.values()])
            installed.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            pass
    try:
        return await asyncio.gather(*(r.run() for r in runners.values()))
    finally:
        for sig in installed:
            with contextlib.suppress(...):
                loop.remove_signal_handler(sig)
```

This is the *same* pattern `run_with_shutdown` already uses (one handler per
signal, `request_stop()` fan-out, `finally`-guarded teardown) — generalized
from one runner to N, so it carries no new risk pattern, only a wider
fan-out. `run_with_shutdown` itself is **untouched in its multi-runner
behavior** — it remains exactly what it is today, correct for its one
existing use (a single-runner caller, e.g. a test or a hypothetical
single-media-kind deployment), and Option A1's new parameter is still needed
on it for that single-runner case, but `worker_main.py`'s two-runner path
calls `verify_engine_alignment_at_startup` directly (not through
`run_with_shutdown`) and implements its own gather-based loop.

**Option B3 (rejected): modify `run_with_shutdown` to accept a
`dict[str, WorkerRunner]` instead of one `WorkerRunner`, becoming the
multi-runner orchestrator itself.** Rejected because it changes
`run_with_shutdown`'s existing, tested, single-purpose contract (its name,
docstring, and every existing test describe it as running *a* worker) for
the sake of a caller (`worker_main.py`) that can trivially implement the
20-line fan-out itself without touching a function three other things
(implicitly, via the test suite) already depend on staying single-runner-shaped.

### 4.3 `skip_version_alignment` — parameter or env-var-only?

`Settings.skip_version_alignment`, once re-added with `env_prefix="DFP_"`
(`backend/config.py:151-156`), automatically maps to the **same**
`DFP_SKIP_VERSION_ALIGNMENT` environment variable `workers.runner` already
reads directly (`:435`). Since `worker_main.py` runs in the same process and
therefore the same `os.environ` that `Settings` parsed, **the env-var path
alone already synchronizes the two** — re-adding the field is not strictly
required to change `verify_engine_alignment_at_startup`'s runtime behavior at
all.

**Option C1 (recommended, matches the `configured_model_version` precedent):**
add `skip_version_alignment: bool = False` as an explicit parameter to
`verify_engine_alignment_at_startup`, checked with `or` alongside the
existing env-var read:
```python
if skip_version_alignment or os.environ.get("DFP_SKIP_VERSION_ALIGNMENT", "").strip().lower() in _TRUE:
```
`worker_main.py` passes `settings.skip_version_alignment` explicitly. This
mirrors why `configured_model_version` is a parameter rather than
env-var-only (the composition root is the one place that legitimately holds
the typed `Settings` value and should not have to round-trip it through
`os.environ` to communicate with its own subprocess call), makes the
skip path unit-testable without mutating `os.environ`, and gives an
operator two independent ways to trigger it (a `.env`-backed typed setting,
or a bare exported env var for a `workers`-only deployment with no `backend`
composition root at all) — which is a real, still-supported deployment shape
per `docs/BATCH_23_R3_DESIGN.md`'s own framing of the env var as the hatch
for exactly that case.

**Option C2 (rejected as insufficient alone): re-add the `Settings` field for
typed visibility only (e.g. surfaced in `/health` or config dumps), pass
nothing new into `verify_engine_alignment_at_startup`.** This technically
"works" (the shared env var already covers the runtime behavior) but leaves
`Settings.skip_version_alignment` as a field whose only effect is
documentation — a plausible-looking config knob that does nothing if a
caller sets it programmatically (e.g. `Settings(skip_version_alignment=True)`
in a test) without also exporting the env var. That is a footgun the
explicit-parameter option (C1) does not have.

### 4.4 Recommendation

**A1 + B2 + C1.** Together: one new keyword parameter on `run_with_shutdown`
(`configured_model_version`), one new keyword parameter on
`verify_engine_alignment_at_startup` (`skip_version_alignment`, `or`-ed with
the existing env check), a re-added `Settings.skip_version_alignment` field,
and a new `backend/worker_main.py` that calls the gate once directly and
drives both runners with its own signal fan-out modeled on
`run_with_shutdown`'s existing pattern.

---

## 5. Exact files affected

| File | Change | Kind |
|---|---|---|
| `workers/runner.py` | `verify_engine_alignment_at_startup` gains `skip_version_alignment: bool = False`, checked alongside the env var (§4.3 C1). `run_with_shutdown` gains `configured_model_version: str | None = None`, forwarded to the gate call (§4.1 A1). Docstrings updated to name OR1-b. | production |
| `backend/config.py` | Re-add `skip_version_alignment: bool = Field(default=False, description=...)` to `Settings`; remove/replace the B23-R3 §1.8 comment block (`:227-234`) explaining the field's *absence* with one explaining its *presence* and the composition root that reads it. | production |
| `backend/worker_main.py` (**new**) | The composition root: builds queue/storage via existing `jobqueue.registry.build_queue` / `storage.registry.build_storage` (already used by `backend/app.py`, confirmed no new dependency), workers via `workers.registry.build_all_workers`, reads `get_settings()`, calls `verify_engine_alignment_at_startup(settings.model_version, skip_version_alignment=settings.skip_version_alignment)` once, then runs both workers to completion with its own signal fan-out (§4.2 B2). | production, new file |
| `tests/test_startup_alignment.py` | New tests for: `run_with_shutdown`'s new parameter (old zero-arg call still passes, confirming no regression); `verify_engine_alignment_at_startup`'s new `skip_version_alignment` parameter path; the config-default-path closure itself (the headline fix). | test |
| `tests/test_workers.py` or a new `tests/test_worker_main.py` | Tests for `backend/worker_main.py`: both workers start, both receive `request_stop()` on a simulated signal (or via direct call, since signal delivery itself is OS-level and awkward to test — see §9), the gate runs exactly once before either loop starts, a stale configured literal refuses to start either worker. | test, likely new file |
| `docs/VIDEO_SELECTION_IDENTITY.md` | §7/§8 already anticipate this batch by name (`:160`, `:239`) — update the "until B24/OR1-b" language to "as of OR1-b" and confirm the deploy-runbook line (`$DFP_MODEL_VERSION` required until OR1-b) is now conditional/superseded. | doc |
| `REPRODUCIBILITY_AUDIT.md` | OR1 row (`:121`) gets a pointer to OR1-b's closure; the "Still open after Batch 24 (N5) and R31" trailer (just added in Phase 1) drops OR1-b from the B25+ candidates list once closed. | doc |
| `docs/OR1B_DESIGN.md` | this document | doc |

**Not changed:** `contracts/version.py`, `inference/*`, `evaluation/*`,
`backend/app.py` (the API-side format check is already done, OR1's other
half — untouched), `workers/engine_registry.py` (its `verify_version_alignment`
and `engine_versions` are consumed, not modified).

---

## 6. Version impacts

| Constant | Verdict | Reason |
|---|---|---|
| `CONTRACT_VERSION` | **No move** | Not a wire contract change; `Settings` is not part of any OpenAPI-visible schema field touched here. |
| `RUN_SCHEMA_VERSION` | **No move** | `BenchmarkRun` is untouched — OR1-b has nothing to do with benchmark recording. |
| `CALIBRATION_WITNESS_VERSION` | **No move** | `CalibrationWitness` is untouched. |
| `LOCK_VERSION` (split-lock) | **No move** | `evaluation/split_lock.py` is untouched. |
| `.d.` (decode identity) | **No move** | `decode_identity()` is read (via `engine_versions()`), never computed differently. |
| `.f.` (selection identity) | **No move** | Not read or written by anything in this design. |
| `STUB_ENGINE_VERSION` literal | **No move** | The literal itself is unchanged; this batch only changes *whether it is checked* on the config-default path, never *what it is*. |
| `effective_version` / `model_identity_of` semantics | **No move** | `contracts/version.py` is not touched at all (confirmed by scope — no file in §5 is in that module). |

**OR1-b triggers zero identity or contract version movement of any kind.**
This is a pure operational/startup-safety change.

---

## 7. Migration impacts

- **No stored artifact of any kind is affected** (no witness, no benchmark
  row, no fixture, no cache entry format changes).
- **Backward compatibility of `run_with_shutdown`:** the new
  `configured_model_version` parameter defaults to `None`, preserving
  exactly today's behavior for any caller that does not pass it — verified
  against the one existing call site (§4.1).
- **Backward compatibility of `verify_engine_alignment_at_startup`:** the new
  `skip_version_alignment` parameter defaults to `False`, so every existing
  call (`verify_engine_alignment_at_startup()`,
  `verify_engine_alignment_at_startup(STUB_ENGINE_VERSION)`,
  `verify_engine_alignment_at_startup(stale)`) is unaffected.
- **Deploy-time behavior change (real, not a migration in the versioned-artifact
  sense):** a deployment that today runs workers via `backend/worker_main.py`
  (once it exists) **without** `$DFP_MODEL_VERSION` set, and whose engines
  happen to disagree with `Settings.model_version`, will now **fail to start**
  where it previously started silently-miscached. This is the fix working as
  designed (R34's precedent: "OR1 working as intended — fail loud"), but it
  is the one operationally-visible change this batch makes. Rollback is
  trivial (revert the file/parameters; no state to unwind).
- **Deploy-runbook update required:** `docs/VIDEO_SELECTION_IDENTITY.md`'s
  line stating `$DFP_MODEL_VERSION` **must** be set in production
  (`docs/BATCH_23_R3_DESIGN.md:121-122`) becomes conditional on which entry
  point is used — still required for anyone invoking `workers.runner`
  directly (a bare worker, no `backend` composition root), no longer strictly
  required through `backend/worker_main.py` (which now supplies the
  equivalent literal from `Settings` itself).

---

## 8. Test impacts

### 8.1 Existing tests — verified unaffected

- `tests/test_startup_alignment.py::test_the_worker_gate_accepts_the_shipped_configuration`,
  `::test_worker_boot_refuses_a_stale_literal`,
  `::test_the_gate_refuses_engines_that_disagree_with_each_other` — call
  `verify_engine_alignment_at_startup` with 0 or 1 positional args; the new
  `skip_version_alignment` keyword defaults to `False`, unaffected.
- `::test_run_with_shutdown_runs_the_gate_before_the_loop` — calls
  `run_with_shutdown(ExplodingRunner())`, no new kwarg; the new
  `configured_model_version` keyword defaults to `None`, unaffected — this
  test's behavior (env-var-driven, gate runs before the loop) is unchanged.
- `::test_skip_flag_bypasses_with_a_warning` — calls
  `verify_engine_alignment_at_startup()` under `DFP_SKIP_VERSION_ALIGNMENT=true`
  in the environment; unaffected, since the new parameter is `or`-ed with,
  not substituted for, the env check.
- `tests/test_workers.py::test_registry_builds_one_worker_per_queue` — unaffected,
  `build_all_workers` itself is not touched.
- `tests/test_backend.py`, `backend/app.py`'s lifespan tests — unaffected;
  `backend/app.py` is not touched (the API-side format check is a separate,
  already-closed half of OR1).

### 8.2 Collateral search performed

- `grep -rn "run_with_shutdown(" --include="*.py" .` → exactly one call site
  (the test above) — confirmed exhaustive, no other caller to update.
- `grep -rn "verify_engine_alignment_at_startup(" --include="*.py" .` → confirmed
  below in §8.3, all call sites in the test file already enumerated.
- `grep -rn "build_all_workers" --include="*.py" .` → one call site
  (`tests/test_workers.py:682`), unaffected (does not run the workers, only
  builds and inspects them).
- `grep -rn "skip_version_alignment" --include="*.py" .` → currently only the
  B23-R3 comment in `backend/config.py` (no live field, no other reference) —
  confirms nothing else needs updating for the field's return.
- `grep -rn "Settings(" tests/test_startup_alignment.py` → the `_settings()`
  helper (`:164-172`) constructs `Settings` with explicit overrides for
  unrelated fields (`log_json`, `log_level`, `storage_root`); adding a new
  `Settings` field with a default requires no change to this helper.
- `grep -rn "worker_main\|backend/worker_main" --include="*.py" .` → zero
  existing references outside the docs already cited — confirms this is
  genuinely a new file with no prior partial implementation anywhere.

### 8.3 New tests required

| Area | Test |
|---|---|
| `run_with_shutdown` signature | `test_run_with_shutdown_forwards_the_configured_literal` — pass a stale `configured_model_version=` explicitly (no env var), confirm it refuses exactly as the env-var path does |
| `verify_engine_alignment_at_startup` signature | `test_skip_version_alignment_parameter_bypasses_without_the_env_var` — pass `skip_version_alignment=True` with `DFP_SKIP_VERSION_ALIGNMENT` unset, confirm it logs the same warning and returns |
| The headline fix | `test_the_config_default_path_now_gets_the_full_check` — monkeypatch `engine_versions` to agree with each other but disagree with a supplied `configured_model_version`, confirm `run_with_shutdown`/`verify_engine_alignment_at_startup` now refuses where the old zero-argument call would have passed (this is the fail-before/pass-after headline, §9) |
| `Settings` field | `test_skip_version_alignment_defaults_false_and_is_env_backed` — `Settings().skip_version_alignment is False`; `Settings(_env_file=None)` under `DFP_SKIP_VERSION_ALIGNMENT=true` in the environment → `True` |
| `backend/worker_main.py` | `test_worker_main_gates_once_before_either_loop_starts` — a stale `Settings.model_version` refuses before either worker's `run()` is ever called (mirroring `ExplodingRunner`'s pattern) |
| `backend/worker_main.py` — Gap B | `test_worker_main_stop_signal_reaches_both_workers` — call the registered signal-handler callback directly (not an actual OS signal, for portability — Windows has no `add_signal_handler`, already handled by the existing `try/except (NotImplementedError, RuntimeError, ValueError)` pattern) and confirm **both** runners' `request_stop()` fire, closing Gap B explicitly rather than leaving it unverified |

---

## 9. Fail-before / pass-after matrix

| # | Test | Pre-OR1-b | Post-OR1-b |
|---|---|---|---|
| FB1 | `run_with_shutdown(runner, configured_model_version=X)` | **fails** — `TypeError: unexpected keyword argument` | — |
| FB2 | `verify_engine_alignment_at_startup(X, skip_version_alignment=True)` | **fails** — `TypeError: unexpected keyword argument` | — |
| FB3 | `Settings().skip_version_alignment` | **fails** — `AttributeError`, field does not exist | — |
| FB4 | The headline: two engines agreeing with each other but **not** with the configured default, boot via the config-default path (no `$DFP_MODEL_VERSION`, no explicit literal passed — i.e. today's only available call shape) | **passes silently** — the weak check only compares the engines to each other; **this is the live gap** | — |
| PA1 | `run_with_shutdown(runner, configured_model_version=X)` | — | Accepted; forwards to the gate exactly as a bare `verify_engine_alignment_at_startup(X)` call would |
| PA2 | `verify_engine_alignment_at_startup(X, skip_version_alignment=True)` | — | Returns with a warning logged, no `RuntimeError`, matching the existing env-var path's observable behavior |
| PA3 | `Settings().skip_version_alignment` | — | `False` by default; `True` under `DFP_SKIP_VERSION_ALIGNMENT=true` |
| PA4 | The headline, repeated through `backend/worker_main.py`'s composition (engines agree with each other, disagree with `settings.model_version`, no `$DFP_MODEL_VERSION` override) | — | **Refuses to start**, `RuntimeError` naming the configured value and what the engines report — FB4 closed |
| PA5 | Gap B: a simulated stop signal while both workers are running | — | Both `WorkerRunner.request_stop()` calls observed to fire (not just the second-registered one) |

---

## 10. Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Gap A (missing parameter) shipped as "already there" without noticing, per the inherited design text | **Was Medium, now closed by this design** | §4.1 makes the parameter addition explicit and scoped |
| Gap B (signal-handler clobbering) shipped silently, causing one worker to lose graceful shutdown in production | **Was High (silent, hard to detect — only visible under a real SIGTERM during in-flight work), now closed by this design** | §4.2 Option B2 avoids the double-`run_with_shutdown` pattern entirely; §8.3 adds an explicit test for it |
| `backend/worker_main.py` becomes a second, divergent source of queue/storage wiring versus `backend/app.py` | Low | Reuse the exact same `jobqueue.registry.build_queue` / `storage.registry.build_storage` calls `backend/app.py` already makes (`backend/app.py:32,34`) — no new wiring logic invented |
| A future reviewer assumes "OR1-b is trivial" (as every prior document said) and rubber-stamps without re-checking Gaps A/B | Medium | This document exists specifically to correct that inherited assumption |
| Deploy-time behavior change surprises an operator relying on today's silent config-default pass | Low | This is the intended fix (fail loud beats silent cache corruption), and only affects deployments not already setting `$DFP_MODEL_VERSION`, which the existing runbook already says to do |

**No architecture risk.** `backend → workers` already exists as an import
edge (`backend/app.py:35-36`); adding `backend/worker_main.py` introduces no
new package-level dependency and no cycle (`workers` still never imports
`backend`, unaffected by this design).

---

## 11. Independence and dependency analysis

**Can OR1-b be completed independently? Yes.** Every file in §5 is disjoint
from R30/D8/D9/D10 (`inference/decode_env.py`, `backend/config.py`'s `.d.`
literal — OR1-b touches a *different* part of `backend/config.py`, the
`Settings` class, not `STUB_ENGINE_VERSION`'s value), from R16/R23/R26/R28
(`inference/preprocess.py`, `contracts/*` — none touched here), and from
R31 (`evaluation/calibration_protocol.py`, `inference/models.py` — none
touched here).

**Does it depend on R30 or any deferred finding? No.**
`verify_engine_alignment_at_startup`/`engine_versions()` are **value-agnostic**
— they compare whatever `effective_version`/`decode_identity()` currently
produce, without caring what feeds those computations. R30 landing later
(moving `.d.`) would not require touching anything in this design; OR1-b
landing first does not constrain R30's later `.d.` move either. Confirmed
independent in both directions.

**Can it trigger `CONTRACT_VERSION`, `RUN_SCHEMA_VERSION`,
`CALIBRATION_WITNESS_VERSION`, `LOCK_VERSION`, `.d.`, or `.f.`? No — none of
these move, and nothing in this design's scope has a code path that could
move them** (§6, exhaustive).

---

## 12. Open decisions for the reviewer (per this project's own ratification convention)

1. **Confirm A1 + B2 + C1** (§4.4) as the resolution to Gaps A and B, since no
   prior document specified either — this is new design work, not a
   rubber-stamp of an existing plan.
2. **Confirm the two-runner test strategy** (§8.3, calling the registered
   signal-handler callback directly rather than sending a real OS signal) is
   acceptable — real signal delivery in a test is flaky/platform-dependent
   (Windows has no `add_signal_handler` at all, per the existing
   `except (NotImplementedError, ...)` guard), so this is consistent with
   how `run_with_shutdown` itself is already tested (`test_run_with_shutdown_
   runs_the_gate_before_the_loop` never sends a real signal either).
3. **Confirm the deploy-runbook wording change** (§7) is acceptable scope for
   this batch, or should be split into a documentation-only follow-up.

None of these are architectural — all are specification-only, matching this
project's own bar for "ready to implement" (the same bar R31 and the B24
chain were held to).

---

## 13. Final gate

## **APPROVE WITH CHANGES**

The finding is real, independent, and triggers zero identity/version/contract
movement (§6, §11). Two genuine implementation gaps in the inherited scoping
were found and resolved by this design (§3, §4) — not present in any prior
document, and not discoverable without re-deriving from the actual current
`workers/runner.py`/`workers/registry.py` source rather than trusting the
"pure additive wiring" description. Ratify §12's three items; no
architectural rework needed. Implementation may begin once ratified.
