"""OR1 — ``verify_version_alignment`` was defined in Batch 9 and never called.

**The gap.** ``workers.engine_registry.verify_version_alignment`` compares the
API's configured ``model_version`` against what the engines actually report, and
raises on a mismatch — a mismatch that silently disables the result cache (write
key and read key differ, every lookup misses, every job re-runs, nothing errors).
Until Batch 23 it was only ever invoked from the test suite. Batch 22 widened the
exposure by folding ``input_size`` into the identity's dependency set.

Batch 23 (OR1) makes it a **deploy-time gate**:

* **Worker** boot runs the real check, fail-closed, via
  ``workers.runner.verify_engine_alignment_at_startup`` (called from
  ``run_with_shutdown`` before the loop). ``DFP_SKIP_VERSION_ALIGNMENT`` is a
  logged escape hatch.
* **API** boot runs a stdlib-only *format* check in ``backend.app.lifespan`` —
  the API process may not import ``inference`` (no cv2 / numpy), so it cannot run
  the full engine comparison, but it can reject a version that is not composed
  and environment-pinned (the ``"1.0.0-dev"`` failure that actually occurred).

    python tests/test_startup_alignment.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app import create_app  # noqa: E402
from backend.config import STUB_ENGINE_VERSION, Settings  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from workers.engine_registry import engine_versions  # noqa: E402
from workers.runner import verify_engine_alignment_at_startup  # noqa: E402


@contextmanager
def _env(**over):
    saved = {k: os.environ.get(k) for k in over}
    try:
        for k, v in over.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------------
# Worker startup gate
# --------------------------------------------------------------------------


def test_the_worker_gate_accepts_the_shipped_configuration():
    """The real literal against the real engines — the deploy that should work."""
    with _env(DFP_MODEL_VERSION=None, DFP_SKIP_VERSION_ALIGNMENT=None):
        verify_engine_alignment_at_startup(STUB_ENGINE_VERSION)
        # and via the env override path
    with _env(DFP_MODEL_VERSION=STUB_ENGINE_VERSION, DFP_SKIP_VERSION_ALIGNMENT=None):
        verify_engine_alignment_at_startup()


def test_worker_boot_refuses_a_stale_literal():
    """A worker whose configured version differs from its engines fails closed,
    before the loop, naming the value the engines report."""
    stale = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.00000000.d.a97ccb68"
    assert stale != STUB_ENGINE_VERSION
    with _env(DFP_MODEL_VERSION=None, DFP_SKIP_VERSION_ALIGNMENT=None):
        try:
            verify_engine_alignment_at_startup(stale)
        except RuntimeError as exc:
            assert "does not match the engines" in str(exc)
            assert engine_versions()["image"] in str(exc)
            return
    raise AssertionError("a stale configured literal passed the worker startup gate")


def test_run_with_shutdown_runs_the_gate_before_the_loop():
    """``run_with_shutdown`` calls the gate before it ever touches the runner."""
    from workers.runner import run_with_shutdown

    class ExplodingRunner:
        def request_stop(self) -> None: ...

        async def run(self, **_kw):  # pragma: no cover - must never be reached
            raise AssertionError("the loop started despite a version mismatch")

    stale = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.00000000.d.a97ccb68"
    with _env(DFP_MODEL_VERSION=stale, DFP_SKIP_VERSION_ALIGNMENT=None):
        try:
            asyncio.run(run_with_shutdown(ExplodingRunner()))
        except RuntimeError as exc:
            assert "does not match the engines" in str(exc)
            return
    raise AssertionError("run_with_shutdown started the loop with a stale literal")


def test_skip_flag_bypasses_with_a_warning(capfd=None):
    """``DFP_SKIP_VERSION_ALIGNMENT`` starts the worker anyway and logs a warning
    every boot — the one escape hatch, deliberately noisy."""
    import logging

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("workers.runner")
    handler = _Capture()
    logger.addHandler(handler)
    prior_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        stale = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.00000000.d.a97ccb68"
        with _env(DFP_MODEL_VERSION=stale, DFP_SKIP_VERSION_ALIGNMENT="true"):
            verify_engine_alignment_at_startup()  # must NOT raise
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)

    assert any(
        r.levelno == logging.WARNING and "DFP_SKIP_VERSION_ALIGNMENT" in r.getMessage()
        for r in records
    ), "the skip path did not log a warning"


def test_the_gate_refuses_engines_that_disagree_with_each_other():
    """Even with no configured literal in reach, a worker whose two engines
    report different versions fails closed."""
    import workers.engine_registry as er

    def _split() -> dict[str, str]:
        return {"image": "stub-0-a+c.1.d.x", "video": "stub-0-b+c.2.d.y"}

    saved = er.engine_versions
    er.engine_versions = _split
    try:
        with _env(DFP_MODEL_VERSION=None, DFP_SKIP_VERSION_ALIGNMENT=None):
            verify_engine_alignment_at_startup()
    except RuntimeError as exc:
        assert "do not agree" in str(exc)
        return
    finally:
        er.engine_versions = saved
    raise AssertionError("mismatched engine versions passed the gate")


# --------------------------------------------------------------------------
# API startup format check
# --------------------------------------------------------------------------


def _settings(**over) -> Settings:
    import tempfile

    return Settings(
        log_json=False,
        log_level="CRITICAL",
        storage_root=Path(tempfile.mkdtemp(prefix="dfp-or1-")),
        **over,
    )


def test_api_lifespan_accepts_the_shipped_literal():
    with TestClient(create_app(_settings())) as client:
        assert client.get("/health").status_code == 200


def test_api_lifespan_refuses_an_unparseable_model_version():
    """The historically-real failure: a bare, uncomposed string."""
    try:
        with TestClient(create_app(_settings(model_version="1.0.0-dev"))):
            pass
    except RuntimeError as exc:
        assert "1.0.0-dev" in str(exc)
        return
    raise AssertionError("lifespan accepted a non-composed model_version")


def test_api_lifespan_refuses_a_model_version_with_no_decode_segment():
    """Composed but not environment-pinned — the cache cannot tell two decoders
    apart, which Batch 8 measured to be a real source of wrong answers."""
    model_only = STUB_ENGINE_VERSION.split(".d.")[0]
    assert ".d." not in model_only
    try:
        with TestClient(create_app(_settings(model_version=model_only))):
            pass
    except RuntimeError as exc:
        assert "decode" in str(exc).lower()
        return
    raise AssertionError("lifespan accepted a model_version with no .d. segment")


def test_api_lifespan_refuses_a_marker_bearing_but_unparseable_model_version():
    """SA10 — a string that *has* the ``+c.`` and ``.d.`` markers (so it passes
    ``is_composed`` / ``is_environment_pinned``) but that ``parse_model_version``
    raises on: a ``.f.`` marker with no fingerprint after it. The ``except
    ValueError`` branch of the format check must abort startup and name the
    string."""
    malformed = "stub-0+c.739c3fec.g.ae232cba.f..d.a97ccb68"
    try:
        with TestClient(create_app(_settings(model_version=malformed))):
            pass
    except RuntimeError as exc:
        assert malformed in str(exc)
        assert "parse" in str(exc).lower()
        return
    raise AssertionError("lifespan accepted a marker-bearing unparseable model_version")


# --------------------------------------------------------------------------
# OR1-b — the composition-root parameters
# --------------------------------------------------------------------------


def test_run_with_shutdown_forwards_the_configured_literal():
    """PA1. ``run_with_shutdown``'s new ``configured_model_version`` keyword
    reaches the gate exactly as a bare call would — no env var needed."""
    from workers.runner import run_with_shutdown

    class ExplodingRunner:
        def request_stop(self) -> None: ...

        async def run(self, **_kw):  # pragma: no cover - must never be reached
            raise AssertionError("the loop started despite a version mismatch")

    stale = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.00000000.d.a97ccb68"
    with _env(DFP_MODEL_VERSION=None, DFP_SKIP_VERSION_ALIGNMENT=None):
        try:
            asyncio.run(
                run_with_shutdown(ExplodingRunner(), configured_model_version=stale)
            )
        except RuntimeError as exc:
            assert "does not match the engines" in str(exc)
            return
    raise AssertionError(
        "run_with_shutdown did not forward configured_model_version to the gate"
    )


def test_skip_version_alignment_parameter_bypasses_without_the_env_var():
    """PA2. The new keyword is an independent path to the same escape hatch —
    it must not require $DFP_SKIP_VERSION_ALIGNMENT to also be set."""
    stale = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.00000000.d.a97ccb68"
    with _env(DFP_MODEL_VERSION=None, DFP_SKIP_VERSION_ALIGNMENT=None):
        verify_engine_alignment_at_startup(stale, skip_version_alignment=True)  # must NOT raise


def test_skip_version_alignment_defaults_false_and_is_env_backed():
    """PA3. ``Settings.skip_version_alignment`` — re-added in OR1-b — defaults
    to False and is reachable through the same $DFP_SKIP_VERSION_ALIGNMENT
    variable the env-only path already uses."""
    with _env(DFP_SKIP_VERSION_ALIGNMENT=None):
        assert _settings().skip_version_alignment is False
    with _env(DFP_SKIP_VERSION_ALIGNMENT="true"):
        assert _settings().skip_version_alignment is True


def test_the_config_default_path_now_gets_the_full_check():
    """PA4 — the headline. Two engines that agree with each other but not
    with the configured default used to pass (the weak check only compares
    them to each other); passing the literal explicitly now catches it."""
    import workers.engine_registry as er

    agreeing_but_wrong = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.11111111.d.a97ccb68"

    def _agree() -> dict[str, str]:
        return {"image": agreeing_but_wrong, "video": agreeing_but_wrong}

    saved = er.engine_versions
    er.engine_versions = _agree
    try:
        with _env(DFP_MODEL_VERSION=None, DFP_SKIP_VERSION_ALIGNMENT=None):
            # Old shape: no configured literal in reach -> weak check only ->
            # the two engines agree with each other -> passes.
            verify_engine_alignment_at_startup()
            # OR1-b: the composition root passes the configured literal
            # explicitly -> the full check runs -> the agreed value does not
            # match it -> refuses.
            try:
                verify_engine_alignment_at_startup(STUB_ENGINE_VERSION)
            except RuntimeError as exc:
                assert "does not match the engines" in str(exc)
                return
            raise AssertionError(
                "the config-default path is still blind to a value both "
                "engines agree on but that disagrees with the configured default"
            )
    finally:
        er.engine_versions = saved


# --------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------


def _run() -> int:
    tests = sorted((n, o) for n, o in globals().items() if n.startswith("test_") and callable(o))
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
