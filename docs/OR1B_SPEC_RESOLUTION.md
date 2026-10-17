# OR1-b — Specification Resolution and Final Implementation Plan

**Type:** design/specification resolution. No code in this task.
**Follows:** `docs/OR1B_DESIGN.md` §12, which left three items open for
ratification. Each is re-derived from source below, not accepted from the
design document's own recommendation.
**Scope:** OR1-b only — no file or finding outside `docs/OR1B_DESIGN.md` §5
is touched by anything in this document.

---

## Resolution 1 — A1 + B2 + C1 (Gaps A and B, and the `skip_version_alignment` mechanism)

### Re-derivation

Re-read live, this pass, to check the design's own claims rather than trust
them:

- `workers/runner.py:474-488` — confirmed again: `run_with_shutdown(runner,
  *, max_jobs=None, idle_timeout_s=None)` calls
  `verify_engine_alignment_at_startup()` with **zero arguments**. No
  parameter exists to receive a configured literal. (Gap A stands.)
- `workers/registry.py:85-92` — confirmed again: `build_all_workers` returns
  `dict[str, WorkerRunner]`, two entries (`IMAGE_QUEUE`, `VIDEO_QUEUE`). No
  function anywhere in the repository takes this dict and drives both
  runners concurrently. (Gap B stands — this is genuinely unbuilt, not
  merely undocumented.)
- `workers/runner.py:92-94` — `WorkerRunner.request_stop`:
  ```python
  def request_stop(self) -> None:
      """Signal shutdown. Safe from a signal handler; returns immediately."""
      self._stop.set()
  ```
  Confirmed: a bare `asyncio.Event.set()`. **Cannot raise, cannot block,
  cannot partially fail.** This is the load-bearing fact for the fan-out
  safety proof in §2 below — it was asserted in the design document but not
  previously traced to this exact implementation.
- `workers/runner.py:492-498` (inside `run_with_shutdown`, for comparison):
  ```python
  for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
      if sig is None:
          continue
      try:
          loop.add_signal_handler(sig, runner.request_stop)
          installed.append(sig)
      except (NotImplementedError, RuntimeError, ValueError):
          pass
  ```
  One `add_signal_handler` call **per signal**, each bound to **one
  runner's** `request_stop`. This is the exact shape that breaks under
  Option B1 (call this twice, once per runner, and the second call's
  `add_signal_handler(sig, ...)` — same `sig`, different callback —
  overwrites the first, per CPython's `asyncio.unix_events` implementation,
  which stores exactly one handler per signal number in a dict keyed by
  `sig`).

### Recommended resolution

**A1 + B2 + C1, exactly as `OR1B_DESIGN.md` §4.4 proposed, ratified as-is.**
No changes to the prior recommendation survive this re-check — the
re-derivation confirms rather than revises it. Restated precisely for the
implementation plan (§4 below):

- **A1:** `run_with_shutdown` gains `configured_model_version: str | None = None`,
  forwarded verbatim to `verify_engine_alignment_at_startup`.
- **B2:** `backend/worker_main.py` does **not** call `run_with_shutdown` for
  its two-runner path. It calls `verify_engine_alignment_at_startup` once,
  directly, then registers **one** `add_signal_handler` per signal, each
  bound to a single fan-out closure (not one registration per runner):
  ```python
  def _stop_all() -> None:
      for r in runners.values():
          r.request_stop()

  for sig in (...):
      try:
          loop.add_signal_handler(sig, _stop_all)
      ...
  ```
  This is the precise mechanism that makes Gap B structurally unable to
  recur — see §2's proof.
- **C1:** `verify_engine_alignment_at_startup` gains
  `skip_version_alignment: bool = False`, checked with `or` against the
  existing `os.environ.get("DFP_SKIP_VERSION_ALIGNMENT", ...)` read — neither
  replaces the other.

### Why correct

`_stop_all` is a single, named, testable closure. Registering **it** (once
per signal) rather than registering each runner's `request_stop` (once per
runner, per signal) means there is exactly one entry in asyncio's internal
signal-handler table per signal number, so there is nothing for a second
registration to overwrite — the failure mode isn't mitigated, it's
structurally absent from this shape.

### Why alternatives were rejected

- **Option B1** (`asyncio.gather(run_with_shutdown(image, ...), run_with_shutdown(video, ...))`):
  re-confirmed unsafe — each `run_with_shutdown` call independently runs the
  registration loop in §"Re-derivation" above, each binding a *different*
  runner's `request_stop` to the *same* signal number. Rejected, unchanged
  from `OR1B_DESIGN.md`.
- **Option B3** (make `run_with_shutdown` itself accept a dict of runners):
  re-confirmed as unnecessary churn — it would change a function's contract
  that three things (the existing test, its docstring, its name) already
  describe as single-runner-shaped, for a caller (`worker_main.py`) that can
  implement the fan-out in the four extra lines shown above. Rejected,
  unchanged.
- **Option C2** (typed `Settings` field only, no new parameter): re-confirmed
  as a footgun — `Settings(skip_version_alignment=True)` constructed
  programmatically (e.g. in a future test or a non-`.env` config source)
  would silently do nothing unless the operator *also* exports the
  environment variable, since nothing would read the field's value. Rejected,
  unchanged.

### Scope impact

**None beyond what `OR1B_DESIGN.md` §5 already lists.** This resolution
selects among options already enumerated there; it adds no new file and
removes none.

### Startup / shutdown / deployment semantics impact

- **Startup:** the gate runs exactly once per process boot (not once per
  runner), before either runner's loop starts — matching
  `verify_engine_alignment_at_startup`'s own whole-process scope
  (`engine_versions()` already returns both engines in one call, confirmed
  in `OR1B_DESIGN.md` §1.1).
- **Shutdown:** both runners now correctly receive `request_stop()` on
  SIGTERM/SIGINT — a **new, correct** behavior; there was no working two-runner
  shutdown path before this design (Gap B), so this is not a regression risk,
  it is the first time this path exists at all.
- **Deployment:** no change beyond what OR1-b's headline fix already
  entails (§3 of `OR1B_DESIGN.md`) — this resolution is about *how* the
  composition root is built, not *what* it decides.

---

## Resolution 2 — signal-handler-callback test strategy

### Re-derivation

`workers/runner.py:479-481` (docstring): *"Signal handlers are installed
only when a running loop supports them, so this stays importable on
platforms and in test harnesses where they do not — notably Windows, where
`add_signal_handler` is unavailable."* Confirmed live on this exact box: this
project's own implementation box is Windows (`docs/B24_N5_DESIGN.md` §1.6,
and every `uv`/`python` command in this session's history has run under
`C:\...`). `test_run_with_shutdown_runs_the_gate_before_the_loop`
(`tests/test_startup_alignment.py:88-105`), the only existing test that
exercises `run_with_shutdown`'s signal-handling code path at all, **never
sends a real OS signal** — it tests the gate-runs-before-the-loop property by
forcing the gate to raise, not by testing signal delivery.

**No existing test in this repository sends a real SIGTERM/SIGINT to verify
`add_signal_handler` registration.** This is not an oversight specific to
OR1-b — it's this project's standing approach to code that is a documented
no-op on its own development platform.

### Recommended resolution

**Confirmed: call the fan-out closure directly, not via a real signal.**
Structure `backend/worker_main.py` so `_stop_all` (or equivalent) is a
module-level or otherwise directly-importable function — not an anonymous
lambda closed only inside `_run_all`'s local scope — specifically so a test
can do:

```python
runners = {"a": FakeRunner(), "b": FakeRunner()}
_stop_all = _make_stop_all(runners)  # or however it's exposed
_stop_all()
assert runners["a"].stop_requested and runners["b"].stop_requested
```

This tests the actually-load-bearing logic (does the callback reach every
runner) without depending on OS signal delivery, `add_signal_handler`'s
platform availability, or process-level signal semantics — none of which
OR1-b is trying to test (those are asyncio's and the OS's responsibility, not
this batch's).

### Why correct

It isolates the one new piece of logic this batch actually introduces (the
fan-out) from infrastructure (`add_signal_handler`, real signal delivery)
that is already exercised — or already known to be a documented no-op — by
`run_with_shutdown`'s existing tests and docstring. Testing at the right
seam.

### Why alternatives were rejected

- **Send a real `os.kill(os.getpid(), signal.SIGTERM)` in the test:**
  rejected — would be a no-op on this project's own development platform
  (Windows), giving a false sense of coverage: the test would pass whether
  or not the fan-out logic was even correct, because `add_signal_handler`
  never registers anything to receive the signal.
- **Skip signal testing entirely, test only that `verify_engine_alignment_at_startup`
  is called once:** rejected as insufficient — this would leave Gap B's
  actual fix (the fan-out closure) with zero direct test coverage, exactly
  the gap that let Gap B go unnoticed through five prior design documents.

### Scope impact

None — already implied by `OR1B_DESIGN.md` §8.3's PA test description; this
resolution only fixes the exact shape of `worker_main.py`'s internals needed
to make that test possible (the fan-out must be a nameable, callable unit,
not an inline anonymous lambda).

### Startup / shutdown / deployment semantics impact

None — this is a test-authoring decision, not a behavior change.

---

## Resolution 3 — deploy-runbook wording update, in-batch or split off

### Re-derivation

`docs/VIDEO_SELECTION_IDENTITY.md:153-166` and `:238-240`, re-read in full
this pass:

- `:158-162`: *"There is **no** `Settings.skip_version_alignment` field...
  A wired field returns in **B24/OR1-b**... Until B24, the configured
  literal reaches the gate only as `$DFP_MODEL_VERSION`..."*
- `:238-240`: *"Deploy runbook: set `DFP_MODEL_VERSION`... before rollout
  (until B24/OR1-b wires the config-default path into the full check)."*

Both are **forward references to this exact batch**, written by the batch
that came before it, in the same file that already documents everything
else about this gate. This is the identical shape as the N8 gap Phase 1
just fixed: a decision/change whose completion was anticipated by name, in
a specific document, and left unwritten until the batch that fulfills it
actually lands.

### Recommended resolution

**In-batch. Not split off.** Update both passages in the same commit as the
code change:
- `:158-162` → past tense: the field exists, `backend/worker_main.py` exists
  and passes `get_settings().model_version` into the gate.
- `:238-240` → the `$DFP_MODEL_VERSION` requirement is no longer necessary
  when deploying through `backend/worker_main.py` (which now supplies the
  equivalent from `Settings` directly), but remains necessary for a bare
  `workers`-only deployment with no `backend` composition root — state both
  cases rather than deleting the line.

### Why correct

The whole reason Phase 1 of this project's roadmap exists is that
"decided but not written back" is a recurring, real defect class in this
repository (N8, and the B22 witness-residual note both needed a superseding
fix after the fact). Fixing OR1-b's own forward-reference in the same batch
that resolves it costs two paragraphs and prevents a fourth instance of the
exact same defect class from being created on purpose, by the same process
that just spent a phase cleaning up three prior instances of it.

### Why the alternative was rejected

**Split into a documentation-only follow-up:** rejected. The cost of doing
it now (two paragraphs, zero code, zero risk, same file already being
mentally loaded for review) is lower than the cost of tracking a fourth
"remember to close this pointer" item across batches, and there is no
scope, risk, or review reason to prefer deferral — unlike R30 or R31's
actual code changes, this has no version, cache, or migration surface to
reason about separately.

### Scope impact

Adds `docs/VIDEO_SELECTION_IDENTITY.md` to the file list — already present
in `OR1B_DESIGN.md` §5's table; this resolution confirms it stays in-scope
rather than being cut.

### Startup / shutdown / deployment semantics impact

None — documentation only.

---

## Verification block (explicitly requested)

### The fan-out approach cannot lose SIGTERM or SIGINT for either worker

**Proven, not merely argued:** the design registers `add_signal_handler(sig,
_stop_all)` **exactly once per signal type**, where `_stop_all` is a single
closure that iterates every runner and calls `request_stop()` on each.
There is no second `add_signal_handler` call for the same signal to
overwrite the first — the Option-B1 failure mode requires *two*
registrations for the same signal number, and this design performs *one*.
Within that one registration's callback, the fan-out loop cannot partially
complete: `request_stop()` is `self._stop.set()` (`workers/runner.py:94`,
re-confirmed above), a synchronous operation on an `asyncio.Event` that
cannot raise. A `for` loop over two such calls cannot fail on the first and
skip the second.

### The design preserves all current behavior when version alignment is not deliberately overridden

Traced for the default, healthy case: `Settings.model_version` defaults to
`STUB_ENGINE_VERSION` (`backend/config.py:217-225`); a correctly built
deployment's engines report exactly that value, guaranteed at build/test
time by the existing, unaffected `test_configured_model_version_matches_the_engines`.
Before this batch, that case reached the *weak* check (engines agree with
each other) and passed. After this batch, `backend/worker_main.py` passes
`configured_model_version=settings.model_version` (non-`None`), so it now
reaches the *full* `verify_version_alignment` check — which passes for
exactly the same reason the weak check did (the values agree), because
nothing about *what the values are* has changed, only *which check runs*.
**Observable outcome for a healthy default deployment is identical: the
worker starts.** The only observable difference is in the failure case this
batch exists to close (§1.2 of `OR1B_DESIGN.md`) — which is the intended
change, not a regression. `DFP_SKIP_VERSION_ALIGNMENT` and
`$DFP_MODEL_VERSION` env-var paths are untouched (both still read exactly as
before — confirmed, `os.environ.get` calls in `workers/runner.py` are
unmodified by this design, only `or`-ed with / supplemented by the new
parameters, never replaced).

### No version constant movement is required

Re-confirmed against `OR1B_DESIGN.md` §6, and independently re-checked this
pass: nothing in §4's resolved design reads or writes
`contracts/version.py`, `evaluation/calibration_protocol.py`,
`evaluation/benchmark.py`, or `evaluation/split_lock.py`. `CONTRACT_VERSION`,
`RUN_SCHEMA_VERSION`, `CALIBRATION_WITNESS_VERSION`, `LOCK_VERSION`, `.d.`,
`.f.`, and `STUB_ENGINE_VERSION`'s literal value are untouched by every file
in the final plan below.

### No migration path is required

No stored artifact (witness, benchmark row, fixture, cache entry, lock file)
has a shape this design changes. `run_with_shutdown` and
`verify_engine_alignment_at_startup`'s new parameters both default to values
that reproduce today's exact behavior for every existing caller — confirmed
against the single existing call site of each (`tests/test_startup_alignment.py`,
enumerated in `OR1B_DESIGN.md` §8.1). There is nothing to migrate because
nothing persisted changes shape.

---

## Final implementation plan

### Files, in the order they should be written

1. **`workers/runner.py`**
   - `verify_engine_alignment_at_startup(configured_model_version: str | None = None, *, skip_version_alignment: bool = False) -> None` — one new keyword-only parameter; the existing `if os.environ.get(...) in _TRUE:` line becomes `if skip_version_alignment or os.environ.get(...) in _TRUE:`.
   - `run_with_shutdown(runner, *, max_jobs=None, idle_timeout_s=None, configured_model_version: str | None = None) -> WorkerReport` — one new keyword-only parameter, forwarded to `verify_engine_alignment_at_startup(configured_model_version)`.
   - Docstring updates on both, naming OR1-b and cross-referencing `backend/worker_main.py`.

2. **`backend/config.py`**
   - Re-add `skip_version_alignment: bool = Field(default=False, description="...")` to `Settings`, replacing the B23-R3 §1.8 comment block (`:227-234`) that explained the field's absence.

3. **`backend/worker_main.py`** (new)
   - Build `queue`/`storage` via the same `jobqueue.registry.build_queue` / `storage.registry.build_storage` calls `backend/app.py` already makes.
   - `runners = workers.registry.build_all_workers(queue=queue, storage=storage)`.
   - `settings = backend.config.get_settings()`.
   - Call `workers.runner.verify_engine_alignment_at_startup(settings.model_version, skip_version_alignment=settings.skip_version_alignment)` once, directly — not through `run_with_shutdown`.
   - Define `_stop_all()` as a nameable function closing over `runners`, calling `request_stop()` on each.
   - Register `loop.add_signal_handler(sig, _stop_all)` once per signal (`SIGTERM`, `SIGINT`), inside the same `try/except (NotImplementedError, RuntimeError, ValueError)` guard `run_with_shutdown` uses.
   - `await asyncio.gather(*(r.run() for r in runners.values()))`, `finally`-guarded `remove_signal_handler` cleanup, mirroring `run_with_shutdown`'s existing structure.

4. **`tests/test_startup_alignment.py`**
   - `test_run_with_shutdown_forwards_the_configured_literal` (PA1)
   - `test_skip_version_alignment_parameter_bypasses_without_the_env_var` (PA2)
   - `test_the_config_default_path_now_gets_the_full_check` (PA4, the headline fail-before/pass-after)
   - `test_skip_version_alignment_defaults_false_and_is_env_backed` (PA3)

5. **New `tests/test_worker_main.py`**
   - `test_worker_main_gates_once_before_either_loop_starts`
   - `test_worker_main_stop_signal_reaches_both_workers` (calls the fan-out function directly, per Resolution 2)

6. **`docs/VIDEO_SELECTION_IDENTITY.md`**
   - `:153-166` and `:238-240` updated past-tense, per Resolution 3.

7. **`REPRODUCIBILITY_AUDIT.md`**
   - OR1 row (`:121`) gains a pointer to OR1-b's closure.
   - The "Still open after Batch 24 (N5) and R31" trailer (added in Phase 1) drops OR1-b from its "B25+ candidates" list.

8. **`docs/OR1B_DESIGN.md` / `docs/OR1B_SPEC_RESOLUTION.md`** — left as the historical design record, not edited.

### Version / migration / test summary

| | Verdict |
|---|---|
| `CONTRACT_VERSION` / `RUN_SCHEMA_VERSION` / `CALIBRATION_WITNESS_VERSION` / `LOCK_VERSION` | No movement |
| `.d.` / `.f.` / `STUB_ENGINE_VERSION` literal | No movement |
| Stored-artifact migration | None required |
| Backward compatibility | Both new parameters default to values reproducing current behavior for the one existing caller of each function |
| New test count | 6 (4 in `test_startup_alignment.py`, 2 in a new `test_worker_main.py`) |
| Full suite impact | Additive only — no existing test's assertion changes |

### Gate

## **APPROVE FOR IMPLEMENTATION**

All three `OR1B_DESIGN.md` §12 items are resolved by re-derivation, not by
re-stating the design document's own preference. All four explicitly
requested verifications hold, proven against source rather than asserted.
No blocking items remain.

Stopping here per instruction — no code has been written or edited. Waiting
for approval to proceed to implementation.

**Status (2026-10-17): implemented as Batch 25.** `backend/worker_main.py` is
the composition root described above; `tests/test_worker_main.py` covers it and
is green in `tests/run_all.py`. The paragraph above records the design-time
gate, not pending work.
