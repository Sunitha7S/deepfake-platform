"""Worker-side protocols.

Three seams. None of them exists for dependency purity — the runner imports
``storage`` and ``jobqueue`` concretely and is meant to. They exist because each
has more than one real implementation: a handler is image or video, a fetcher is
storage or an in-test stub, a sink writes to storage now and may write to a
database later.

``JobHandler`` is the important one. It is the whole of your requirement 5: a
handler receives a request and returns a bundle. It cannot see the queue, and it
reports liveness by calling ``progress`` rather than by renewing anything — the
runner turns that into a heartbeat. A handler that could call ``extend_lease``
would be a handler that knows what a lease is.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contracts.enums import JobStage
from contracts.evidence import EvidenceBundle
from contracts.job import JobSpec
from workers.types import JobRequest


class ProgressReporter(Protocol):
    """How a handler says it is still alive and how far along it is.

    Called by the handler; implemented by the runner. The runner uses it both to
    publish progress and to decide the lease needs renewing, which is why a long
    handler that never calls it will eventually have its lease reclaimed — that
    is the intended pressure, not a defect.
    """

    async def __call__(self, fraction: float, *, stage: JobStage | None = None) -> None: ...


@runtime_checkable
class JobHandler(Protocol):
    """Pure analysis. Payload in, result out.

    Raise ``PermanentError`` for input that will never succeed and
    ``TransientError`` for a condition worth retrying. Return normally — with a
    bundle whose verdict is ``insufficient_evidence`` — for input that is simply
    unjudgeable. That last case is a *success*, and conflating it with failure is
    how a product ends up showing an error for a photograph of a landscape.
    """

    media_kind: str
    name: str

    async def handle(self, request: JobRequest, progress: ProgressReporter) -> EvidenceBundle:
        """Analyse and return. Must not mutate ``request``."""
        ...


@runtime_checkable
class MediaFetcher(Protocol):
    """Resolves a job spec to the bytes it refers to."""

    async def fetch(self, spec: JobSpec) -> bytes:
        """Raise ``PermanentError`` when the object is absent or corrupt,
        ``TransientError`` when the store is merely unreachable.

        The distinction is the difference between dead-lettering a job whose
        file was deleted and dead-lettering every job during a storage outage.
        """
        ...


@runtime_checkable
class ResultSink(Protocol):
    """Durable home for a finished bundle.

    Must be **idempotent**: at-least-once delivery means a job can genuinely run
    twice, so publishing the same job's result again returns the result already
    stored rather than overwriting it. First writer wins.
    """

    async def put_if_absent(self, job_id: str, bundle: EvidenceBundle) -> bool:
        """Commit ``bundle`` only if nothing is stored for ``job_id`` yet.

        Returns True when this caller's write won, False when a result already
        existed. **Must be atomic**: a check followed by a write leaves a window
        in which two workers both see the key free and both write, and the
        second one wins — the exact inversion of first-writer-wins.

        Losing is an ordinary outcome, never an exception. At-least-once
        delivery means duplicate execution is expected, and forcing callers to
        catch an exception for the common case would make the normal path look
        like the error path.

        A committed result is immutable. ``missing → present`` is the only legal
        transition; ``present → different present`` must be impossible.
        """
        ...

    async def publish(self, bundle: EvidenceBundle) -> tuple[EvidenceBundle, bool]:
        """Return ``(stored_bundle, was_already_present)``.

        Convenience over ``put_if_absent`` for callers that want the winning
        bundle back rather than a boolean. Implementations must build it *on*
        ``put_if_absent`` rather than beside it, or the atomicity guarantee only
        holds for whichever entry point a caller happens to use.
        """
        ...

    async def find(self, job_id: str) -> EvidenceBundle | None:
        """Existing result for a job, if any.

        The runner calls this *before* executing. A worker that persisted a
        result and then died before acking would otherwise redo minutes of GPU
        work to produce a bundle that is already on disk.
        """
        ...

    async def find_by_content(
        self, sha256: str, model_version: str
    ) -> EvidenceBundle | None:
        """Existing result for this exact content under this exact model.

        Powers the upload short-circuit: a file already analysed is answered
        without being uploaded at all. Both halves of the key matter — a new
        model must not serve a verdict computed by the old one under its own
        version number.
        """
        ...
