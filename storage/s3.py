"""S3-compatible object storage. Skeleton.

Targets AWS S3, MinIO, Cloudflare R2, Backblaze B2 — anything speaking the S3
API. Satisfies ``StorageBackend`` structurally today and raises
``NotImplementedError`` from every operation, each with the specific call that
will fill it in.

**It deliberately does not implement ``UploadReceiver``.** Bytes travel from the
browser to the object store through the presigned PUT and never enter this
process; routing them through it would discard the entire reason for presigning.
A caller needing in-process ingestion detects the absence structurally rather
than catching an exception:

    isinstance(backend, UploadReceiver)  # False here, True for local

That is the whole argument for splitting the protocols, made concrete: this
class does not have to lie about what it can do.

Session persistence is the open design question for S2 proper. S3 has no session
concept, so the sidecars the local backend keeps on disk must live somewhere
else — most likely the same Redis that S3 of the roadmap introduces for the
queue. The constructor takes a ``session_store`` for that reason; wiring it is
part of finishing this class, not of the skeleton.
"""

from __future__ import annotations

from typing import Any

from storage.base import BaseStorageBackend
from storage.signing import SignedUrl, UrlSigner
from storage.types import (
    NewUpload,
    ObjectMetadata,
    StorageHealth,
    StoredObject,
    UploadSession,
    UploadState,
)

_TODO = "S3CompatibleStorageBackend is a skeleton; "


class S3CompatibleStorageBackend(BaseStorageBackend):
    """``StorageBackend`` over the S3 API. Not yet implemented."""

    name = "s3"

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region: str = "us-east-1",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        signer: UrlSigner | None = None,
        session_store: Any = None,
        client: Any = None,
    ) -> None:
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.region = region
        # Credentials are held only to pass to a client factory. They are never
        # logged and never appear in a repr — a config object that prints its
        # own secret key is how credentials reach a bug tracker.
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._signer = signer
        self._session_store = session_store
        self._client = client

    def __repr__(self) -> str:
        return (
            f"S3CompatibleStorageBackend(bucket={self.bucket!r}, "
            f"endpoint_url={self.endpoint_url!r}, region={self.region!r})"
        )

    # ---- upload lifecycle ------------------------------------------------

    async def create_upload(self, request: NewUpload) -> tuple[UploadSession, SignedUrl]:
        raise NotImplementedError(
            _TODO + "create_upload will call generate_presigned_url('put_object', "
            "Params={'Bucket': self.bucket, 'Key': key, 'ContentType': "
            "request.content_type}, ExpiresIn=request.ttl_seconds) and persist "
            "the session to self._session_store."
        )

    async def complete_upload(self, upload_id: str) -> UploadSession:
        raise NotImplementedError(
            _TODO + "complete_upload will head_object() the staging key to read "
            "ContentLength and ETag, verify them against the session, then "
            "copy_object() to the final key and delete_object() the staging one. "
            "Note ETag is only an MD5 for single-part uploads, so checksum "
            "verification should request ChecksumSHA256 rather than trusting ETag."
        )

    async def transition(self, upload_id: str, target: UploadState) -> UploadSession:
        raise NotImplementedError(
            _TODO + "transition will read-modify-write the session in "
            "self._session_store, using check_transition() from BaseStorageBackend "
            "so the state machine stays identical to the local backend's."
        )

    async def get_session(self, upload_id: str) -> UploadSession:
        raise NotImplementedError(
            _TODO + "get_session will read from self._session_store. S3 itself has "
            "no session concept; this is why the constructor takes one."
        )

    # ---- objects ---------------------------------------------------------

    async def put_object(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> ObjectMetadata:
        raise NotImplementedError(
            _TODO + "put_object will call put_object(Bucket=..., Key=key, Body=data, "
            "ContentType=content_type, ChecksumSHA256=...). For first-writer-wins "
            "semantics use IfNoneMatch='*', which S3 now supports and which closes "
            "the check-then-write race the local backend documents."
        )

    async def put_object_if_absent(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> bool:
        raise NotImplementedError(
            _TODO + "put_object_if_absent will call put_object(..., IfNoneMatch='*'), "
            "which S3 rejects with a 412 PreconditionFailed when the key already "
            "exists — translate that to False and let anything else propagate. "
            "This is a genuine conditional write on the server, so unlike the "
            "local backend it needs no temporary object and holds across "
            "processes and regions. On stores without IfNoneMatch support, a "
            "conditional-put emulation via a DynamoDB or Redis lock is required; "
            "a read-then-write must not be substituted, because that reinstates "
            "exactly the race this method exists to close."
        )

    async def get_object(self, key: str) -> StoredObject:
        raise NotImplementedError(
            _TODO + "get_object will call get_object() and read Body. Large "
            "objects should stream rather than materialise; see the note in "
            "types.StoredObject."
        )

    async def stat_object(self, key: str) -> ObjectMetadata:
        raise NotImplementedError(_TODO + "stat_object will call head_object().")

    async def object_exists(self, key: str) -> bool:
        raise NotImplementedError(
            _TODO + "object_exists will call head_object() and translate a 404 "
            "ClientError to False. Any other ClientError must propagate — "
            "swallowing a 403 here would report a permissions failure as 'absent'."
        )

    async def delete_object(self, key: str) -> bool:
        raise NotImplementedError(
            _TODO + "delete_object will call delete_object(). S3 returns success "
            "for an absent key, so returning a truthful bool needs a head_object() "
            "first — matching the local backend's contract."
        )

    # ---- operational -----------------------------------------------------

    async def health(self) -> StorageHealth:
        """Never raises, even as a skeleton.

        Returning an unhealthy result rather than raising ``NotImplementedError``
        is the correct behaviour and not a shortcut: a health check exists to
        report condition, and a backend that cannot serve is exactly what
        ``ok=False`` means. Wiring this class in prematurely therefore fails
        readiness instead of crashing the process.
        """
        return StorageHealth(
            ok=False,
            backend=self.name,
            writable=False,
            detail="S3CompatibleStorageBackend is not implemented",
        )
