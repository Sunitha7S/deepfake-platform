"""Application service: the API's use cases, composed over the real platform.

Routes are thin because the interesting decisions live here — and because a use
case that can be tested without an HTTP client is a use case that gets tested.

The design point worth stating: **``JobStatus`` is derived, never stored.**
S1 kept its own map of job statuses, which was fine while responses were
fixtures. With a real queue it would be a fourth copy of state that nothing
keeps in sync, and the failure is silent and specific — the queue dead-letters a
job, the stored status still reads ``running``, and a user polls a spinner
forever. So status is computed on read from two authorities:

* the queue envelope, which owns delivery state;
* the result sink, which owns whether an answer exists.

Everything else this class touches is owned elsewhere too: bytes and upload
sessions by ``storage``, results by the sink. This module owns no state at all,
which is why it needs no cleanup, no migration and no reconciliation job.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from contracts.api import (
    AnalyzeAcceptedResponse,
    CreateUploadRequest,
    CreateUploadResponse,
    PresignedUpload,
)
from contracts.enums import ErrorCode, JobStage, JobState, MediaKind
from contracts.evidence import EvidenceBundle
from contracts.job import AnalysisOptions, JobProgress, JobSpec, JobStatus
from jobqueue.base import QueueBackend
from jobqueue.errors import JobNotFound
from jobqueue.registry import IMAGE_QUEUE, VIDEO_QUEUE
from jobqueue.types import JobEnvelope, JobState as QueueState
from jobqueue.types import Priority
from storage.base import StorageBackend, UploadReceiver
from storage.errors import (
    ChecksumMismatch,
    SignatureExpired,
    SignatureInvalid,
    SizeMismatch,
    StorageError,
    UploadExpired,
    UploadNotFound,
)
from storage.types import NewUpload, UploadState
from workers.protocol import ResultSink

from backend.config import Settings
from backend.errors import AppError
from backend.logging import get_logger, job_context

log = get_logger(__name__)

# How the queue's delivery states project onto the application's job states.
# Kept as a table so the mapping is one readable thing rather than a chain of
# conditionals spread through the routes.
_STATE_MAP: dict[QueueState, JobState] = {
    QueueState.READY: JobState.QUEUED,
    QueueState.DELAYED: JobState.QUEUED,
    QueueState.LEASED: JobState.RUNNING,
    QueueState.DONE: JobState.SUCCEEDED,
    QueueState.DEAD: JobState.FAILED,
}

_STAGE_FOR: dict[JobState, tuple[JobStage, int]] = {
    JobState.QUEUED: (JobStage.VALIDATING, 0),
    JobState.RUNNING: (JobStage.SCORING, 50),
    JobState.SUCCEEDED: (JobStage.FINALIZING, 100),
}


class AnalysisService:
    """Every use case the API exposes, over real storage, queue and sink."""

    def __init__(
        self,
        *,
        settings: Settings,
        storage: StorageBackend,
        queue: QueueBackend,
        sink: ResultSink,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.queue = queue
        self.sink = sink

    # ---- uploads ---------------------------------------------------------

    async def create_upload(self, body: CreateUploadRequest) -> CreateUploadResponse:
        """Reserve storage, or answer immediately from the result cache."""
        cached = await self.sink.find_by_content(body.sha256, self.settings.model_version)
        if cached is not None:
            log.info(
                "upload short-circuited by cache",
                extra={"sha256": body.sha256, "job_id": str(cached.job_id)},
            )
            return CreateUploadResponse(
                upload_id=f"u_cached_{secrets.token_urlsafe(8)}",
                cache_hit=True,
                cached_result=cached,
            )

        session, signed = await self.storage.create_upload(
            NewUpload(
                sha256=body.sha256,
                size_bytes=body.size_bytes,
                content_type=body.content_type,
                prefix="uploads",
                ttl_seconds=self.settings.upload_url_ttl_s,
            )
        )
        log.info(
            "upload negotiated",
            extra={
                "upload_id": session.upload_id,
                "sha256": body.sha256,
                "media_kind": body.media_kind.value,
                "size_bytes": body.size_bytes,
            },
        )
        # Where the client should PUT depends on who can receive bytes.
        #
        # An object store issues a URL pointing at itself, and the browser
        # uploads directly — which is the entire reason for presigning. A local
        # filesystem has no such endpoint, so this process must expose one, and
        # the client is pointed at our own receiver. Either way the client sees
        # one field and does not care which it got.
        if isinstance(self.storage, UploadReceiver):
            url = (
                f"{self.settings.api_prefix}/uploads/{session.upload_id}"
                f"?signature={signed.signature}"
            )
        else:
            url = signed.url

        return CreateUploadResponse(
            upload_id=session.upload_id,
            cache_hit=False,
            upload=PresignedUpload(
                url=url,
                method=signed.method,
                headers={"Content-Type": body.content_type},
                expires_at=signed.expires_at,
            ),
        )

    async def receive_bytes(
        self, upload_id: str, data: bytes, *, signature: str | None
    ) -> None:
        """Server half of the signed URL. Local deployments only.

        An object store receives bytes through its own presigned endpoint and
        never routes them through this process; asking it to would discard the
        entire benefit of presigning.
        """
        if not isinstance(self.storage, UploadReceiver):
            raise AppError(
                ErrorCode.INVALID_REQUEST,
                message=(
                    "This deployment uploads directly to object storage. PUT the "
                    "file to the URL returned by POST /uploads instead."
                ),
                detail=f"{type(self.storage).__name__} is not an UploadReceiver",
            )
        try:
            await self.storage.accept_upload(upload_id, data, signature=signature)
            await self.storage.complete_upload(upload_id)
        except UploadNotFound as exc:
            raise AppError(ErrorCode.UPLOAD_NOT_FOUND) from exc
        except (UploadExpired, SignatureExpired) as exc:
            raise AppError(ErrorCode.UPLOAD_EXPIRED) from exc
        except SignatureInvalid as exc:
            raise AppError(
                ErrorCode.INVALID_REQUEST,
                message="That upload link is not valid.",
                detail="signature verification failed",
            ) from exc
        except ChecksumMismatch as exc:
            raise AppError(ErrorCode.CHECKSUM_MISMATCH, detail=str(exc)) from exc
        except SizeMismatch as exc:
            raise AppError(
                ErrorCode.INVALID_REQUEST,
                message="The uploaded file is a different size than declared.",
                detail=str(exc),
            ) from exc

    # ---- analysis --------------------------------------------------------

    async def submit(self, upload_id: str, options: AnalysisOptions) -> AnalyzeAcceptedResponse:
        """Turn a completed upload into a queued job."""
        try:
            session = await self.storage.get_session(upload_id)
        except UploadNotFound as exc:
            raise AppError(ErrorCode.UPLOAD_NOT_FOUND) from exc
        except StorageError as exc:
            raise AppError(ErrorCode.INTERNAL_ERROR, detail=str(exc)) from exc

        if session.state is UploadState.CREATED:
            raise AppError(
                ErrorCode.UPLOAD_INCOMPLETE,
                message="The file has not been uploaded yet.",
            )
        if not session.state.has_bytes:
            raise AppError(
                ErrorCode.UPLOAD_INCOMPLETE,
                message="The upload has not finished verifying yet.",
                detail=f"session is {session.state.value}",
            )

        media_kind = _media_kind_for(session.content_type)
        job_id = uuid4()
        spec = JobSpec(
            job_id=job_id,
            media_kind=media_kind,
            storage_key=session.key,
            sha256=session.sha256,
            size_bytes=session.size_bytes,
            declared_mime=session.content_type,
            model_version=self.settings.model_version,
            idempotency_key=f"{session.sha256}:{self.settings.model_version}",
            options=options,
            submitted_at=datetime.now(UTC),
        )

        queue_name = IMAGE_QUEUE if media_kind is MediaKind.IMAGE else VIDEO_QUEUE
        envelope = await self.queue.enqueue(
            queue_name,
            spec.model_dump(mode="json"),
            priority=Priority.NORMAL,
            idempotency_key=spec.idempotency_key,
            job_id=str(job_id),
        )

        # Idempotency may have returned an existing job. The caller must be
        # given the id that actually exists, not the one it was about to get.
        effective_id = envelope.job_id
        deduped = effective_id != str(job_id)
        if deduped:
            log.info(
                "submission deduplicated",
                extra={"requested": str(job_id), "existing": effective_id},
            )

        stats = await self.queue.stats(queue_name)
        return AnalyzeAcceptedResponse(
            job_id=UUID(effective_id),
            state=JobState.QUEUED,
            status_url=self.settings.status_url(effective_id),
            events_url=self.settings.events_url(effective_id),
            result_url=self.settings.result_url(effective_id),
            queue_position=stats.ready,
            estimated_wait_s=None,
        )

    # ---- reads -----------------------------------------------------------

    async def job_status(self, job_id: UUID) -> JobStatus:
        """Derive status from the queue and the result store.

        Order matters. The result store is consulted when the queue has no
        envelope, because a queue may purge a completed job while its result
        lives on — reporting 404 for a job whose answer is on disk would be a
        lie told by the wrong authority.
        """
        key = str(job_id)
        with job_context(job_id):
            envelope = await self._envelope(key)
            result = await self.sink.find(key)

            if envelope is None:
                if result is None:
                    raise AppError(ErrorCode.JOB_NOT_FOUND, job_id=job_id)
                return _succeeded_status(job_id, result, self.settings.result_url(key))

            return _derive_status(job_id, envelope, result, self.settings.result_url(key))

    async def result(self, job_id: UUID) -> EvidenceBundle:
        """Return the bundle, or explain precisely why there isn't one."""
        key = str(job_id)
        with job_context(job_id):
            bundle = await self.sink.find(key)
            if bundle is not None:
                log.info("result served", extra={"verdict": bundle.verdict.value})
                return bundle

            envelope = await self._envelope(key)
            if envelope is None:
                raise AppError(ErrorCode.JOB_NOT_FOUND, job_id=job_id)

            if envelope.state is QueueState.DEAD:
                # Surface the job's own failure. A generic 404 would send a user
                # to retry a file that can never succeed.
                raise AppError(
                    ErrorCode.DECODE_FAILED,
                    message=(
                        "This file could not be analysed. "
                        + (envelope.last_error or "The job failed.")
                    ),
                    detail=envelope.last_error,
                    job_id=job_id,
                )
            if envelope.state is QueueState.DONE:
                # Completed with no stored result: the result expired.
                raise AppError(ErrorCode.JOB_NOT_FOUND, job_id=job_id)

            raise AppError(ErrorCode.RESULT_NOT_READY, job_id=job_id, retry_after_s=2)

    async def _envelope(self, job_id: str) -> JobEnvelope | None:
        try:
            return await self.queue.get(job_id)
        except JobNotFound:
            return None

    # ---- health ----------------------------------------------------------

    async def health(self) -> tuple[bool, dict[str, int], list[str]]:
        """(ready, queue depths, problems). Never raises."""
        problems: list[str] = []

        storage_health = await self.storage.health()
        if not storage_health.ok:
            problems.append(f"storage: {storage_health.detail or 'unavailable'}")

        queue_health = await self.queue.health()
        if not queue_health.ok:
            problems.append(f"queue: {queue_health.detail or 'unavailable'}")

        depths: dict[str, int] = {}
        for name in (IMAGE_QUEUE, VIDEO_QUEUE):
            try:
                depths[name] = (await self.queue.stats(name)).backlog
            except Exception as exc:  # noqa: BLE001 - health never raises
                depths[name] = 0
                problems.append(f"queue {name}: {type(exc).__name__}")

        return not problems, depths, problems


# --------------------------------------------------------------------------
# Derivation helpers
# --------------------------------------------------------------------------


def _media_kind_for(content_type: str) -> MediaKind:
    return MediaKind.VIDEO if content_type.startswith("video/") else MediaKind.IMAGE


def _succeeded_status(job_id: UUID, result: EvidenceBundle, result_url: str) -> JobStatus:
    return JobStatus(
        job_id=job_id,
        state=JobState.SUCCEEDED,
        progress=JobProgress(stage=JobStage.FINALIZING, percent=100),
        submitted_at=result.created_at,
        started_at=result.created_at,
        finished_at=result.created_at + timedelta(milliseconds=result.timing.total_ms),
        attempt=1,
        result_url=result_url,
    )


def _derive_status(
    job_id: UUID,
    envelope: JobEnvelope,
    result: EvidenceBundle | None,
    result_url: str,
) -> JobStatus:
    state = _STATE_MAP[envelope.state]

    # A persisted result outranks EVERY queue state, dead-lettering included.
    #
    # The envelope answers "has this delivery been acknowledged", not "does an
    # answer exist". The runner persists and then acks, so a stored bundle means
    # the analysis finished; publish() is first-writer-wins over deterministic
    # handlers, so whatever is stored for a job is that job's result.
    #
    # DEAD is reachable with a result present and is not a hypothetical: lease
    # expiry counts as an attempt, so a job with max_attempts=1 that persists
    # and then dies before acking is reaped straight to the dead-letter queue
    # with a perfectly good bundle on disk. Excluding FAILED here would report
    # "failed" from /jobs while /results returned 200 — contradictory answers to
    # the same question, one request apart.
    #
    # The dead-letter entry stays in the DLQ regardless. A crash-after-persist
    # pattern is worth an operator's attention even when users are served
    # correctly; the two facts answer different questions.
    if result is not None:
        state = JobState.SUCCEEDED

    error = None
    if state is JobState.FAILED:
        from contracts.errors import ApiError

        error = ApiError.of(
            ErrorCode.DECODE_FAILED,
            message="This file could not be analysed.",
            detail=envelope.last_error,
            job_id=job_id,
        )

    terminal = state in (JobState.SUCCEEDED, JobState.FAILED)
    stage, percent = _STAGE_FOR.get(state, (JobStage.VALIDATING, 0))

    return JobStatus(
        job_id=job_id,
        state=state,
        # Progress is coarse by design. Fine-grained progress needs workers to
        # publish it on a channel the API can read, which arrives with SSE; a
        # percentage invented here would be a number that means nothing.
        progress=JobProgress(stage=stage, percent=percent),
        submitted_at=envelope.enqueued_at,
        started_at=envelope.updated_at if state is not JobState.QUEUED else None,
        finished_at=envelope.updated_at if terminal else None,
        attempt=max(1, envelope.attempts),
        error=error,
        # Advertise the result URL only when a result was actually found. A job
        # can reach DONE with its result already expired, and handing out a link
        # that 404s is worse than omitting it: the client follows it, fails, and
        # has no way to tell "gone" from "broken".
        result_url=result_url if result is not None else None,
    )
