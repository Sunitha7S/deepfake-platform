"""The worker loop.

Owns everything stateful so handlers do not have to: leasing, media fetch, the
heartbeat, result persistence, ack/nack, and shutdown.

Three properties are worth reading the code for.

**Lease ownership is enforced on every renewal.** The heartbeat presents its
``lease_id``; the queue rejects it if the job has been re-leased. On rejection
the runner raises ``LeaseLost``, cancels the handler, and then does *neither* an
ack nor a nack — the job belongs to someone else and both calls would damage
their work.

**The result is looked up before the job runs.** A worker that persisted a
bundle and died before acking would otherwise redo minutes of GPU work to
produce a result already on disk. Finding one means acking immediately.

**Shutdown drains rather than drops.** On stop, the loop stops leasing, gives
in-flight work a deadline, and nacks anything still running as retryable so it
returns to the queue at once instead of waiting out its lease.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from datetime import UTC, datetime

from contracts.enums import JobStage
from contracts.evidence import EvidenceBundle
from contracts.job import JobSpec
from jobqueue.base import DEFAULT_LEASE_SECONDS, QueueBackend
from jobqueue.errors import JobNotLeased, LeaseNotHeld
from jobqueue.types import LeasedJob, NackOutcome
from workers.protocol import JobHandler, MediaFetcher, ResultSink
from workers.types import (
    ExecutionOutcome,
    ExecutionReport,
    JobRequest,
    LeaseLost,
    PermanentError,
    ShutdownReason,
    WorkerReport,
    classify,
)

log = logging.getLogger(__name__)

HEARTBEAT_FRACTION = 1 / 3
"""Renew at a third of the lease.

Two missed renewals must still leave room for a third before expiry; renewing at
half would let a single slow call lose the lease.
"""


class WorkerRunner:
    """Runs one handler against one queue until asked to stop."""

    def __init__(
        self,
        *,
        queue: QueueBackend,
        queue_name: str,
        handler: JobHandler,
        fetcher: MediaFetcher,
        sink: ResultSink,
        worker_id: str | None = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        poll_interval_s: float = 0.5,
        drain_timeout_s: float = 30.0,
    ) -> None:
        self.queue = queue
        self.queue_name = queue_name
        self.handler = handler
        self.fetcher = fetcher
        self.sink = sink
        self.worker_id = worker_id or f"{handler.name}-{secrets.token_hex(4)}"
        self.lease_seconds = lease_seconds
        self.poll_interval_s = poll_interval_s
        self.drain_timeout_s = drain_timeout_s

        self._stop = asyncio.Event()
        self._draining = False
        self._current: str | None = None

    # ---- lifecycle -------------------------------------------------------

    def request_stop(self) -> None:
        """Signal shutdown. Safe from a signal handler; returns immediately."""
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    async def run(
        self, *, max_jobs: int | None = None, idle_timeout_s: float | None = None
    ) -> WorkerReport:
        """Loop until stopped, or until ``max_jobs`` or ``idle_timeout_s``.

        The bounds exist for tests and for one-shot batch runs; a long-lived
        worker passes neither.
        """
        started_at = datetime.now(UTC)
        counts = {k: 0 for k in ("leased", "completed", "already_done", "retried",
                                 "dead_lettered", "lease_lost", "abandoned")}
        reason = ShutdownReason.STOP_REQUESTED
        idle_since: float | None = None

        while not self._stop.is_set():
            if max_jobs is not None and counts["leased"] >= max_jobs:
                reason = ShutdownReason.JOB_LIMIT
                break

            leased = await self.queue.lease(
                self.queue_name,
                worker_id=self.worker_id,
                count=1,
                lease_seconds=self.lease_seconds,
            )
            if not leased:
                now = time.monotonic()
                idle_since = idle_since if idle_since is not None else now
                if idle_timeout_s is not None and now - idle_since >= idle_timeout_s:
                    reason = ShutdownReason.IDLE_LIMIT
                    break
                # Wait on the stop event rather than sleeping: a shutdown
                # request during an idle poll should be acted on at once, not
                # after the interval elapses.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval_s)
                continue

            idle_since = None
            counts["leased"] += 1
            report = await self.process(leased[0])
            key = {
                ExecutionOutcome.COMPLETED: "completed",
                ExecutionOutcome.ALREADY_DONE: "already_done",
                ExecutionOutcome.RETRY_SCHEDULED: "retried",
                ExecutionOutcome.DEAD_LETTERED: "dead_lettered",
                ExecutionOutcome.LEASE_LOST: "lease_lost",
                ExecutionOutcome.ABANDONED_SHUTDOWN: "abandoned",
            }[report.outcome]
            counts[key] += 1

        return WorkerReport(
            worker_id=self.worker_id,
            queue=self.queue_name,
            reason=reason,
            started_at=started_at,
            stopped_at=datetime.now(UTC),
            **counts,
        )

    async def _drain_deadline(self) -> None:
        """Resolve once shutdown has been requested *and* the grace period has passed.

        In-flight work is given ``drain_timeout_s`` to finish on its own, because
        throwing away a nearly-complete video analysis to save a few seconds of
        deploy time is a poor trade. Past that, the job is handed back.
        """
        await self._stop.wait()
        self._draining = True
        await asyncio.sleep(self.drain_timeout_s)

    # ---- one job ---------------------------------------------------------

    async def process(self, leased: LeasedJob) -> ExecutionReport:
        """Execute a single leased job to a terminal outcome."""
        started = time.perf_counter()
        job_id = leased.job_id
        lease_id = leased.lease.lease_id
        self._current = job_id
        heartbeats = 0

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        try:
            spec = JobSpec.model_validate(leased.payload)
        except ValueError as exc:
            # A payload that does not parse will not parse on retry either.
            await self._nack(job_id, lease_id, f"malformed job payload: {exc}", retryable=False)
            return ExecutionReport(
                job_id=job_id,
                outcome=ExecutionOutcome.DEAD_LETTERED,
                attempts=leased.envelope.attempts + 1,
                duration_ms=elapsed_ms(),
                error="malformed job payload",
            )

        # Two distinct identifiers are in play and confusing them silently
        # disables crash recovery: `job_id` is the *queue's* handle (``j_…``),
        # used for ack, nack and lease operations, while the sink is keyed by
        # the *contract's* ``JobSpec.job_id`` (a UUID), which is what the API
        # and the user see. Looking a result up by the queue's handle finds
        # nothing, every time, and the only symptom is work being redone.
        result_id = str(spec.job_id)

        # Crash recovery: a previous attempt may have persisted a result and
        # died before acking.
        existing = await self.sink.find(result_id)
        if existing is not None:
            await self._ack(job_id, lease_id)
            return ExecutionReport(
                job_id=job_id,
                outcome=ExecutionOutcome.ALREADY_DONE,
                attempts=leased.envelope.attempts + 1,
                duration_ms=elapsed_ms(),
                verdict=existing.verdict.value,
            )

        heartbeat_failed = asyncio.Event()
        beats = {"n": 0}

        async def heartbeat() -> None:
            interval = max(0.05, self.lease_seconds * HEARTBEAT_FRACTION)
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.queue.extend_lease(
                        job_id, lease_id, lease_seconds=self.lease_seconds
                    )
                    beats["n"] += 1
                except (LeaseNotHeld, JobNotLeased):
                    # The job was reclaimed. Stop renewing and tell the main
                    # path; it must not ack or nack from here.
                    heartbeat_failed.set()
                    return

        async def progress(fraction: float, *, stage: JobStage | None = None) -> None:
            """What a handler calls. Renews the lease opportunistically."""
            if heartbeat_failed.is_set():
                raise LeaseLost(job_id, lease_id)
            try:
                await self.queue.extend_lease(
                    job_id, lease_id, lease_seconds=self.lease_seconds
                )
                beats["n"] += 1
            except (LeaseNotHeld, JobNotLeased) as exc:
                heartbeat_failed.set()
                raise LeaseLost(job_id, lease_id) from exc

        beat_task = asyncio.create_task(heartbeat())
        try:
            media = await self.fetcher.fetch(spec)
            request = JobRequest(spec=spec, media=media)

            work = asyncio.create_task(self.handler.handle(request, progress))
            lost = asyncio.create_task(heartbeat_failed.wait())
            drain = asyncio.create_task(self._drain_deadline())
            done, _ = await asyncio.wait(
                {work, lost, drain}, return_when=asyncio.FIRST_COMPLETED
            )

            if work not in done:
                work.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await work
                if drain in done:
                    # Shutdown outlasted the drain window. Return the job
                    # explicitly rather than letting the lease lapse: a nack
                    # makes it available after a short backoff, where lease
                    # expiry would strand it for the full lease duration while
                    # the replacement worker sits idle.
                    lost.cancel()
                    outcome = await self._nack(
                        job_id,
                        lease_id,
                        "worker shut down before the job finished",
                        retryable=True,
                    )
                    return ExecutionReport(
                        job_id=job_id,
                        outcome=ExecutionOutcome.ABANDONED_SHUTDOWN
                        if outcome is not ExecutionOutcome.LEASE_LOST
                        else outcome,
                        attempts=leased.envelope.attempts + 1,
                        duration_ms=elapsed_ms(),
                        heartbeats=beats["n"],
                        error="abandoned at shutdown",
                    )
                # The lease went before the work finished.
                raise LeaseLost(job_id, lease_id)

            lost.cancel()
            drain.cancel()
            bundle: EvidenceBundle = work.result()

        except LeaseLost:
            heartbeats = beats["n"]
            return ExecutionReport(
                job_id=job_id,
                outcome=ExecutionOutcome.LEASE_LOST,
                attempts=leased.envelope.attempts + 1,
                duration_ms=elapsed_ms(),
                heartbeats=heartbeats,
                error="lease lost mid-execution; job abandoned to its new owner",
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - the runner classifies everything
            heartbeats = beats["n"]
            retryable = classify(exc)
            detail = getattr(exc, "detail", None) or str(exc)
            outcome = await self._nack(job_id, lease_id, detail, retryable=retryable)
            return ExecutionReport(
                job_id=job_id,
                outcome=outcome,
                attempts=leased.envelope.attempts + 1,
                duration_ms=elapsed_ms(),
                heartbeats=heartbeats,
                error=f"{type(exc).__name__}: {detail}",
            )
        finally:
            beat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beat_task
            self._current = None

        heartbeats = beats["n"]

        # Persist first, then ack. The other order loses the result if the
        # process dies between them, and the job would be marked done with
        # nothing to show.
        try:
            stored, already = await self.sink.publish(bundle)
        except BaseException as exc:  # noqa: BLE001
            retryable = classify(exc)
            detail = getattr(exc, "detail", None) or str(exc)
            outcome = await self._nack(job_id, lease_id, detail, retryable=retryable)
            return ExecutionReport(
                job_id=job_id,
                outcome=outcome,
                attempts=leased.envelope.attempts + 1,
                duration_ms=elapsed_ms(),
                heartbeats=heartbeats,
                error=f"persist failed: {detail}",
            )

        try:
            await self._ack(job_id, lease_id)
        except LeaseLost:
            # The result is stored; another worker will find it and ack. Losing
            # the lease at this point costs nothing.
            return ExecutionReport(
                job_id=job_id,
                outcome=ExecutionOutcome.LEASE_LOST,
                attempts=leased.envelope.attempts + 1,
                duration_ms=elapsed_ms(),
                heartbeats=heartbeats,
                verdict=stored.verdict.value,
            )

        return ExecutionReport(
            job_id=job_id,
            outcome=ExecutionOutcome.ALREADY_DONE if already else ExecutionOutcome.COMPLETED,
            attempts=leased.envelope.attempts + 1,
            duration_ms=elapsed_ms(),
            heartbeats=heartbeats,
            verdict=stored.verdict.value,
        )

    # ---- queue interaction ----------------------------------------------

    async def _ack(self, job_id: str, lease_id: str) -> None:
        try:
            await self.queue.ack(job_id, lease_id)
        except (LeaseNotHeld, JobNotLeased) as exc:
            raise LeaseLost(job_id, lease_id) from exc

    async def _nack(
        self, job_id: str, lease_id: str, error: str, *, retryable: bool
    ) -> ExecutionOutcome:
        try:
            result = await self.queue.nack(
                job_id, lease_id, error=error, retryable=retryable
            )
        except (LeaseNotHeld, JobNotLeased):
            # Losing the lease during the nack is not an error: the job is
            # already back with someone else, which is what a nack wanted.
            return ExecutionOutcome.LEASE_LOST
        return (
            ExecutionOutcome.RETRY_SCHEDULED
            if result.outcome is NackOutcome.RETRY_SCHEDULED
            else ExecutionOutcome.DEAD_LETTERED
        )


_TRUE = {"1", "true", "yes", "on"}


def verify_engine_alignment_at_startup(configured_model_version: str | None = None) -> None:
    """Fail a worker closed if its engines do not report the configured version.

    **OR1 (Batch 23).** ``workers.engine_registry.verify_version_alignment`` has
    existed since Batch 9 but was only ever called from the test suite. A worker
    whose engines report a different ``model_version`` than the API's configured
    literal has a *silently* disabled result cache — the write key and the read
    key differ, every lookup misses, every job re-runs, and nothing errors. This
    turns that into a startup failure.

    ``configured_model_version`` is the string the API keys its cache on. The
    composition root passes ``get_settings().model_version``; when nothing is
    passed this falls back to ``$DFP_MODEL_VERSION``. ``workers`` may not import
    ``backend`` — that would close a dependency cycle the DAG test forbids — so
    the *default* configured literal (``backend.config.STUB_ENGINE_VERSION``,
    used only when ``$DFP_MODEL_VERSION`` is unset) is not visible here; the
    build-time guard ``test_configured_model_version_matches_the_engines`` covers
    that case, and this function still fails closed on the two failures it *can*
    see without the literal: engines that disagree with each other, and a
    configured string that is malformed.

    Fail-closed: on mismatch the ``RuntimeError`` (which already names the
    expected string) propagates out of ``run_with_shutdown`` and the process
    exits non-zero without starting the loop.

    Escape hatch: ``DFP_SKIP_VERSION_ALIGNMENT=true`` logs a warning every boot
    and returns. It exists for the one case where an operator must deploy now and
    regenerate the literal after.

    Imports are local so ``import workers.runner`` stays cheap and free of cv2 /
    numpy for callers that only need ``WorkerRunner``.
    """
    import os

    from contracts.version import is_composed, is_environment_pinned
    from workers.engine_registry import engine_versions, verify_version_alignment

    if os.environ.get("DFP_SKIP_VERSION_ALIGNMENT", "").strip().lower() in _TRUE:
        log.warning(
            "version alignment check skipped by DFP_SKIP_VERSION_ALIGNMENT; the "
            "result cache is silently disabled if the configured model_version "
            "is wrong"
        )
        return

    configured = configured_model_version or os.environ.get("DFP_MODEL_VERSION")
    if configured is not None:
        verify_version_alignment(configured)
        return

    # No explicit literal and no $DFP_MODEL_VERSION override: the API is on the
    # backend.config default, which the pre-merge test pins to the engines. Still
    # refuse to start on the failures visible from here.
    versions = engine_versions()
    unique = set(versions.values())
    if len(unique) != 1:
        detail = ", ".join(f"{n} reports {v!r}" for n, v in sorted(versions.items()))
        raise RuntimeError(
            f"the engines do not agree on a model_version ({detail}); the API "
            "holds one configured string for both media kinds and its cache "
            "would be disabled for whichever engine it does not match"
        )
    reported = unique.pop()
    if not (is_composed(reported) and is_environment_pinned(reported)):
        raise RuntimeError(
            f"the engines report a malformed model_version {reported!r}: it must "
            "be a composed, environment-pinned version"
        )
    log.info(
        "version alignment: engines agree on %r; no DFP_MODEL_VERSION override, "
        "the API default is pinned to this by test_configured_model_version_"
        "matches_the_engines",
        reported,
    )


async def run_with_shutdown(
    runner: WorkerRunner, *, max_jobs: int | None = None, idle_timeout_s: float | None = None
) -> WorkerReport:
    """Run a worker, draining in-flight work on SIGINT or SIGTERM.

    Signal handlers are installed only when a running loop supports them, so
    this stays importable on platforms and in test harnesses where they do not
    — notably Windows, where ``add_signal_handler`` is unavailable.

    Before the loop starts, :func:`verify_engine_alignment_at_startup` runs the
    fail-closed OR1 gate.
    """
    import signal

    verify_engine_alignment_at_startup()


    loop = asyncio.get_running_loop()
    installed: list[int] = []
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, runner.request_stop)
            installed.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            # Windows, or a non-main thread. The caller can still call
            # request_stop() directly, so this is a reduction in convenience
            # rather than in correctness.
            pass
    try:
        return await runner.run(max_jobs=max_jobs, idle_timeout_s=idle_timeout_s)
    finally:
        for sig in installed:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(sig)
