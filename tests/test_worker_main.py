"""OR1-b — ``backend.worker_main``, the worker composition root.

Two things only this module can prove, because they need two runners at
once: (1) the startup gate runs before *either* loop starts, and (2) a
shutdown signal reaches *every* runner, not just the last one registered.
(2) is the point of this whole file — ``docs/OR1B_DESIGN.md`` §3.2 traces a
concrete way a naive implementation loses one worker's graceful shutdown
(registering ``add_signal_handler`` once per runner, for the same signal,
silently overwrites the earlier registration). ``backend.worker_main``
avoids that shape entirely by registering each signal exactly once, bound to
a fan-out function — tested directly here, not through real signal delivery
(unavailable on Windows in the first place; see ``workers.runner.
run_with_shutdown``'s own docstring).

    python tests/test_worker_main.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import STUB_ENGINE_VERSION  # noqa: E402
from backend.worker_main import _run_all, _stop_all  # noqa: E402


class FakeRunner:
    """A ``WorkerRunner`` stand-in: records whether it was run or stopped."""

    def __init__(self, result: str = "ok") -> None:
        self.ran = False
        self.stop_requested = False
        self._result = result

    def request_stop(self) -> None:
        self.stop_requested = True

    async def run(self, **_kw):
        self.ran = True
        return self._result


class ExplodingRunner:
    """A ``WorkerRunner`` stand-in whose ``run`` must never be reached."""

    def request_stop(self) -> None: ...

    async def run(self, **_kw):
        raise AssertionError("the loop started despite a version mismatch")


def test_worker_main_gates_once_before_either_loop_starts():
    """The gate runs before ``asyncio.gather`` ever awaits a runner — a stale
    configured literal must stop both queues, not just one."""
    stale = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.00000000.d.c9d4025c"  # R30, Batch 26
    runners = {"image": ExplodingRunner(), "video": ExplodingRunner()}
    try:
        asyncio.run(_run_all(runners, configured_model_version=stale))
    except RuntimeError as exc:
        assert "does not match the engines" in str(exc)
        return
    raise AssertionError("a stale configured literal did not stop the supervisor")


def test_worker_main_runs_both_runners_when_aligned():
    """The positive path: a correctly configured literal starts every runner
    and their results are collected under their own queue names."""
    runners = {"image": FakeRunner("img-report"), "video": FakeRunner("vid-report")}
    reports = asyncio.run(
        _run_all(runners, configured_model_version=STUB_ENGINE_VERSION)
    )
    assert runners["image"].ran and runners["video"].ran
    assert reports == {"image": "img-report", "video": "vid-report"}


def test_worker_main_stop_signal_reaches_both_workers():
    """The fan-out itself, tested directly — this is what makes losing one
    worker's shutdown handling structurally impossible rather than merely
    unlikely: there is exactly one registration per signal, bound to this
    function, which visits every runner."""
    runners = {"image": FakeRunner(), "video": FakeRunner(), "third": FakeRunner()}
    _stop_all(runners)
    assert all(r.stop_requested for r in runners.values())


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
