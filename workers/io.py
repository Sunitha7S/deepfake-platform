"""Concrete ``MediaFetcher`` and ``ResultSink`` over ``StorageBackend``.

Both translate storage-domain exceptions into the worker's retryable /
permanent split. That translation is the point of these classes: ``ObjectNotFound``
means the file is gone and no amount of retrying will bring it back, while a
``StorageUnavailable`` means the store is down and the job should wait. Getting
that backwards dead-letters every job during an outage.
"""

from __future__ import annotations

import asyncio
import json

from contracts.evidence import EvidenceBundle
from contracts.job import JobSpec
from storage.base import StorageBackend
from storage.errors import ObjectNotFound, StorageError, StorageUnavailable
from workers.types import PermanentError, TransientError

RESULTS_PREFIX = "results"


class StorageMediaFetcher:
    """Reads the uploaded bytes a job refers to, and verifies them."""

    def __init__(self, storage: StorageBackend, *, verify_checksum: bool = True) -> None:
        self._storage = storage
        self._verify = verify_checksum

    async def fetch(self, spec: JobSpec) -> bytes:
        try:
            obj = await self._storage.get_object(spec.storage_key)
        except ObjectNotFound as exc:
            raise PermanentError(
                "The uploaded file is no longer available.",
                detail=f"object {spec.storage_key!r} not found",
            ) from exc
        except StorageUnavailable as exc:
            raise TransientError("Storage is unreachable.", detail=str(exc)) from exc
        except StorageError as exc:
            # An unclassified storage error is treated as transient, matching
            # the runner's default. A key that is malformed rather than missing
            # would surface here, and retrying it three times is cheap.
            raise TransientError("Storage read failed.", detail=str(exc)) from exc

        if self._verify and obj.metadata.sha256 is not None:
            if obj.metadata.sha256 != spec.sha256:
                # The bytes are not what the job says they are. Re-reading them
                # will not change that.
                raise PermanentError(
                    "The stored file does not match the checksum recorded for this job.",
                    detail=f"expected {spec.sha256[:12]}…, stored {obj.metadata.sha256[:12]}…",
                )

        if len(obj.data) != spec.size_bytes:
            raise PermanentError(
                "The stored file is a different size than recorded for this job.",
                detail=f"expected {spec.size_bytes} bytes, stored {len(obj.data)}",
            )
        return obj.data


class StorageResultSink:
    """Persists bundles as JSON objects, keyed by job id.

    Idempotent by construction: ``publish`` checks for an existing object first
    and returns it untouched if one is there. First writer wins.

    The check-then-write is not atomic, so two workers racing on the same job
    could both observe an empty key. That window is narrow and the consequence
    is benign — analysis is a pure function of the bytes and the model version,
    so the two bundles differ only in ids and timings. Closing it properly needs
    a conditional put (S3 ``If-None-Match``), which is noted for the S3 backend
    rather than faked here.
    """

    def __init__(self, storage: StorageBackend, *, prefix: str = RESULTS_PREFIX) -> None:
        self._storage = storage
        self._prefix = prefix

    def key_for(self, job_id: str) -> str:
        return f"{self._prefix}/{job_id}.json"

    def content_key_for(self, sha256: str, model_version: str) -> str:
        """Where the content→job pointer lives.

        The upload short-circuit needs to answer "has this exact content already
        been analysed by this model", but results are addressed by job id. A
        small pointer object keeps that lookup in storage rather than requiring
        a database purely for one index.
        """
        return f"{self._prefix}/by-content/{sha256}/{model_version}.json"

    async def find_by_content(
        self, sha256: str, model_version: str
    ) -> EvidenceBundle | None:
        """Resolve the pointer, then the bundle. Returns None if either is absent."""
        try:
            pointer = await self._storage.get_object(
                self.content_key_for(sha256, model_version)
            )
        except ObjectNotFound:
            return None
        except StorageError as exc:
            raise TransientError("Could not read the result index.", detail=str(exc)) from exc

        try:
            job_id = json.loads(pointer.data)["job_id"]
        except (ValueError, KeyError, TypeError):
            # A corrupt pointer is a cache miss, not a failure: the authoritative
            # result is still addressable by job id, and re-analysing costs
            # compute rather than correctness.
            return None
        return await self.find(job_id)

    async def find(self, job_id: str) -> EvidenceBundle | None:
        try:
            obj = await self._storage.get_object(self.key_for(job_id))
        except ObjectNotFound:
            return None
        except StorageError as exc:
            raise TransientError("Could not read existing results.", detail=str(exc)) from exc

        try:
            return EvidenceBundle.model_validate_json(obj.data)
        except ValueError as exc:
            # A stored result that no longer parses means the contract moved
            # under it. Treating it as absent would overwrite it; surfacing it
            # as permanent gets a human to look.
            raise PermanentError(
                "A stored result exists but could not be read.",
                detail=f"{type(exc).__name__} parsing {self.key_for(job_id)}",
            ) from exc

    async def put_if_absent(self, job_id: str, bundle: EvidenceBundle) -> bool:
        """Commit atomically. True if this caller won the key.

        No read precedes the write. Asking ``find`` first and then writing is
        precisely the race this replaces: two workers both observe the key free
        and both write, and the second wins. The storage backend's conditional
        create decides it instead, in one operation.
        """
        payload = bundle.model_dump_json().encode("utf-8")
        try:
            won = await self._storage.put_object_if_absent(
                self.key_for(job_id), payload, content_type="application/json"
            )
        except StorageError as exc:
            raise TransientError("Could not persist the result.", detail=str(exc)) from exc

        if not won:
            return False

        # The pointer follows the bundle, and only for the winner. Written
        # if-absent too: a loser overwriting it would repoint the content cache
        # at a job whose bundle is not the authoritative one.
        #
        # Order matters. A pointer to a result that does not exist yet makes the
        # upload cache serve a 404; a result with no pointer merely misses the
        # cache, which costs compute and nothing else.
        try:
            await self._storage.put_object_if_absent(
                self.content_key_for(bundle.media.sha256, bundle.model.version),
                json.dumps({"job_id": job_id}).encode("utf-8"),
                content_type="application/json",
            )
        except StorageError:
            # The result is committed and authoritative; only the cache index
            # failed. Raising here would nack a job whose work is safely stored,
            # sending it round again to lose its own race.
            pass
        return True

    async def publish(self, bundle: EvidenceBundle) -> tuple[EvidenceBundle, bool]:
        job_id = str(bundle.job_id)
        if await self.put_if_absent(job_id, bundle):
            return bundle, False

        existing = await self.find(job_id)
        if existing is None:
            # Lost the race, then found nothing — the winner's object was
            # deleted between the two calls. Treat this caller's bundle as the
            # result rather than reporting a result that does not exist.
            return bundle, True
        return existing, True


class InMemoryResultSink:
    """Non-durable sink for tests and local development.

    Shares the idempotency contract with ``StorageResultSink`` so the same
    conformance expectations apply to both.
    """

    def __init__(self) -> None:
        self._results: dict[str, EvidenceBundle] = {}
        self._by_content: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()

    async def find(self, job_id: str) -> EvidenceBundle | None:
        return self._results.get(job_id)

    async def find_by_content(
        self, sha256: str, model_version: str
    ) -> EvidenceBundle | None:
        job_id = self._by_content.get((sha256, model_version))
        return self._results.get(job_id) if job_id else None

    async def put_if_absent(self, job_id: str, bundle: EvidenceBundle) -> bool:
        """Commit under a lock.

        A dict assignment after a dict read is atomic today only because no
        ``await`` separates them — a property that a later maintainer adding one
        line would silently destroy. The lock states the requirement instead of
        depending on it.
        """
        async with self._lock:
            if job_id in self._results:
                return False
            self._results[job_id] = bundle
            self._by_content.setdefault(
                (bundle.media.sha256, bundle.model.version), job_id
            )
            return True

    async def publish(self, bundle: EvidenceBundle) -> tuple[EvidenceBundle, bool]:
        job_id = str(bundle.job_id)
        if await self.put_if_absent(job_id, bundle):
            return bundle, False
        return self._results.get(job_id, bundle), True

    def __len__(self) -> int:
        return len(self._results)
