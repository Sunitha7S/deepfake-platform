"""The worker process entry point. **OR1-b (Batch 25).**

``backend`` is the only package allowed to import both ``backend.config`` (for
the configured literal) and ``workers`` (to build and run the engines) — the
composition root the config-default startup gate needed since Batch 23. See
``workers.runner.verify_engine_alignment_at_startup`` and
``docs/OR1B_DESIGN.md`` / ``docs/OR1B_SPEC_RESOLUTION.md`` for the full design.

``workers.registry.build_all_workers`` returns one ``WorkerRunner`` per queue
(image, video) — "what a supervisor process starts." This module is that
supervisor. It does **not** call ``workers.runner.run_with_shutdown``, which
stays single-runner-shaped: calling it once per runner would register
``add_signal_handler`` twice for the same signal, and the second registration
silently replaces the first, dropping one worker's graceful shutdown. Instead
this module calls the fail-closed gate once, directly, then drives every
runner with its own signal fan-out.

    python -m backend.worker_main
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from backend.config import get_settings
from backend.logging import configure_logging
from jobqueue.registry import SHARED_KINDS, build_queue
from storage.registry import build_storage
from workers.registry import build_all_workers
from workers.runner import WorkerReport, WorkerRunner, verify_engine_alignment_at_startup

log = logging.getLogger(__name__)


def _stop_all(runners: dict[str, WorkerRunner]) -> None:
    """Fan out a shutdown request to every runner. Safe from a signal handler.

    A standalone function, not a closure, specifically so a test can call it
    directly with fake runners and confirm every one of them is signalled —
    without depending on ``asyncio.add_signal_handler`` (unavailable on
    Windows, per ``workers.runner.run_with_shutdown``'s own docstring) or on
    real OS signal delivery, which this module does not need its own test
    coverage of.
    """
    for runner in runners.values():
        runner.request_stop()


async def _run_all(
    runners: dict[str, WorkerRunner],
    *,
    configured_model_version: str | None,
    skip_version_alignment: bool = False,
) -> dict[str, WorkerReport]:
    """Gate once, then run every runner to completion, draining on shutdown.

    The gate is called **once** for the whole process, not once per runner:
    ``verify_engine_alignment_at_startup`` already compares every engine
    (``workers.engine_registry.engine_versions()`` returns both image and
    video in one call), so one call covers both queues. Calling it per-runner
    would be redundant, not incorrect — but there is no reason to pay it
    twice.

    Exactly one ``add_signal_handler`` registration per signal, each bound to
    :func:`_stop_all` (which itself fans out to every runner) — never one
    registration per runner. This is what makes losing a runner's shutdown
    handling structurally impossible rather than merely unlikely: there is
    only one handler to register for each signal, so there is nothing for a
    second registration to overwrite.
    """
    verify_engine_alignment_at_startup(
        configured_model_version, skip_version_alignment=skip_version_alignment
    )

    loop = asyncio.get_running_loop()
    installed: list[int] = []
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _stop_all, runners)
            installed.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            # Windows, or a non-main thread. A caller can still call
            # _stop_all(runners) directly, so this is a reduction in
            # convenience rather than in correctness.
            pass
    try:
        names = list(runners)
        reports = await asyncio.gather(*(runners[name].run() for name in names))
        return dict(zip(names, reports, strict=True))
    finally:
        for sig in installed:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(sig)


def require_shared_queue(queue_kind: str) -> None:
    """Refuse to run a worker on a queue no other process can reach.

    This module is a separate process from the API by construction. With
    ``queue_kind='memory'`` the API enqueues into its own dictionary and this
    process polls an empty one of its own: every submission stays ``queued``
    forever, and nothing anywhere reports an error. Failing at boot, with the
    setting that fixes it, is the only point where that is visible.
    """
    if queue_kind not in SHARED_KINDS:
        raise RuntimeError(
            f"queue_kind={queue_kind!r} cannot be shared with the API process, so "
            "this worker would never receive a job. Set DFP_QUEUE_KIND=sqlite (and "
            "the same DFP_QUEUE_PATH for the API and the worker) for one host."
        )


async def main() -> dict[str, WorkerReport]:
    """Build the real queue, storage and workers, then run them.

    Mirrors ``backend/app.py``'s own ``build_storage(settings.storage_config())``
    / ``build_queue(settings.queue_config())`` calls exactly — the same
    settings-to-config mapping, so a worker and the API process agree on which
    backend they are pointed at.
    """
    settings = get_settings()
    # The API configures logging in create_app(); without the same call here
    # the worker process logged nothing at all below WARNING.
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    require_shared_queue(settings.queue_kind)
    storage = build_storage(settings.storage_config())
    queue = build_queue(settings.queue_config())
    # Same field the API logs at boot: two processes pointed at different queue
    # files is otherwise invisible — both start cleanly and never meet.
    log.info("worker starting", extra={"queue_backend": repr(queue)})
    runners = build_all_workers(queue=queue, storage=storage)
    return await _run_all(
        runners,
        configured_model_version=settings.model_version,
        skip_version_alignment=settings.skip_version_alignment,
    )


if __name__ == "__main__":
    asyncio.run(main())
