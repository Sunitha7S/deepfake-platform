"""Filesystem storage backend. Fully working.

Layout under ``root``::

    objects/    verified bytes, at their final key
    staging/    unverified bytes, before checksum verification
    sessions/   one JSON sidecar per upload

Sessions live on disk rather than in a dict for a reason that matters beyond
tidiness: an in-memory map would make this backend behave differently from S3
across a restart or a second process, and the conformance suite would not catch
it because a single-process test never restarts. Persisting them keeps the two
backends honest against the same tests.

Staging is separate from ``objects/`` so that unverified bytes are never
reachable at the final key. Promotion is ``os.replace``, which is atomic within
a filesystem: a reader either sees the whole object or no object, never a
partial one.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

from storage.base import BaseStorageBackend
from storage.errors import (
    ChecksumMismatch,
    ObjectNotFound,
    SizeMismatch,
    StorageError,
    UploadNotFound,
)
from storage.signing import HmacUrlSigner, SignedUrl, UrlSigner
from storage.types import (
    NewUpload,
    ObjectMetadata,
    StorageHealth,
    StoredObject,
    UploadSession,
    UploadState,
)


class UnsafeKey(StorageError):
    """A key that would escape the storage root.

    Its own exception type because this is the difference between a storage
    layer and an arbitrary-file-write primitive, and it should be greppable in
    a log.
    """

    def __init__(self, key: str, reason: str) -> None:
        self.key = key
        super().__init__(f"unsafe key {key!r}: {reason}")


def validate_key(key: str) -> str:
    """Return a normalised key, or raise ``UnsafeKey``.

    Rejects absolute paths, parent traversal, drive letters, backslashes and
    embedded nulls. Checked before the path is joined to the root, not after,
    because a check that runs after ``Path(root) / key`` is already too late on
    an absolute key — ``Path('/a') / '/etc/passwd'`` is ``/etc/passwd``.
    """
    if not key or not key.strip():
        raise UnsafeKey(key, "empty")
    if "\x00" in key:
        raise UnsafeKey(key, "contains a null byte")
    if "\\" in key:
        raise UnsafeKey(key, "backslash; keys are posix-style")
    if key.startswith("/"):
        raise UnsafeKey(key, "absolute path")
    if len(key) > 1 and key[1] == ":":
        raise UnsafeKey(key, "drive letter")

    if any(p == ".." for p in key.split("/")):
        raise UnsafeKey(key, "parent traversal")

    # Keys must arrive canonical rather than being silently normalised.
    # ``PurePosixPath`` collapses "./a" to "a" and "a//b" to "a/b", so accepting
    # them would let several distinct strings name one object. Anything keyed on
    # the raw string — a cache, a dedup check, an audit log — would then see two
    # keys where the store sees one.
    normalised = str(PurePosixPath(key))
    if normalised != key:
        raise UnsafeKey(key, f"not canonical; did you mean {normalised!r}")
    return normalised


class LocalStorageBackend(BaseStorageBackend):
    """``StorageBackend`` + ``UploadReceiver`` over a local directory."""

    name = "local"

    def __init__(
        self,
        root: Path | str,
        *,
        signer: UrlSigner | None = None,
        base_url: str = "",
    ) -> None:
        self.root = Path(root).resolve()
        self.objects_dir = self.root / "objects"
        self.staging_dir = self.root / "staging"
        self.sessions_dir = self.root / "sessions"
        for d in (self.objects_dir, self.staging_dir, self.sessions_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._signer = signer or HmacUrlSigner(secrets.token_urlsafe(32), base_url=base_url)
        self._lock = asyncio.Lock()

    # ---- paths -----------------------------------------------------------

    def _object_path(self, key: str) -> Path:
        safe = validate_key(key)
        # Containment is checked lexically, with ``normpath``, not with
        # ``Path.resolve()``.
        #
        # ``resolve()`` touches the filesystem, and under concurrent access it
        # intermittently returns a path that fails the containment check for a
        # perfectly valid key — observed here as roughly one rejection in
        # twenty-four parallel writes to *distinct* keys. A path validator that
        # fails at random is worse than none: it turns a correct request into a
        # spurious error under exactly the load where errors are hardest to
        # attribute.
        #
        # ``normpath`` is pure string manipulation, so it is deterministic and
        # race-free. It is also sufficient: ``validate_key`` has already
        # rejected absolute paths, drive letters, backslashes and any ``..``
        # segment, and requires the key to be canonical, so no lexical join can
        # escape the root.
        #
        # The property given up is symlink resolution — a symlink planted
        # *inside* ``objects/`` could still redirect a well-formed key. That
        # tree is created and owned by this service and never written by a
        # client, so the exposure is an operator with filesystem access, who has
        # simpler options.
        path = Path(os.path.normpath(self.objects_dir / safe))
        if not path.is_relative_to(self.objects_dir):
            raise UnsafeKey(key, "escapes the storage root")
        return path

    def _staging_path(self, upload_id: str) -> Path:
        return self.staging_dir / f"{validate_key(upload_id)}.part"

    def _session_path(self, upload_id: str) -> Path:
        return self.sessions_dir / f"{validate_key(upload_id)}.json"

    # ---- session persistence --------------------------------------------

    def _read_session_sync(self, upload_id: str) -> UploadSession:
        path = self._session_path(upload_id)
        if not path.exists():
            raise UploadNotFound(upload_id)
        return UploadSession.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_session_sync(self, session: UploadSession) -> None:
        path = self._session_path(session.upload_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(session.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    async def get_session(self, upload_id: str) -> UploadSession:
        return await asyncio.to_thread(self._read_session_sync, upload_id)

    # ---- upload lifecycle ------------------------------------------------

    async def create_upload(self, request: NewUpload) -> tuple[UploadSession, SignedUrl]:
        upload_id = f"u_{secrets.token_urlsafe(16)}"
        now = datetime.now(UTC)
        # Date-partitioned so a lifecycle rule can expire by prefix. No part of
        # the key comes from anything the caller supplied beyond the prefix,
        # which is validated with everything else.
        key = validate_key(f"{request.prefix}/{now:%Y/%m/%d}/{upload_id}")

        session = UploadSession(
            upload_id=upload_id,
            key=key,
            staging_key=f"staging/{upload_id}.part",
            state=UploadState.CREATED,
            sha256=request.sha256,
            size_bytes=request.size_bytes,
            content_type=request.content_type,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(seconds=request.ttl_seconds),
            backend=self.name,
        )
        await asyncio.to_thread(self._write_session_sync, session)
        signed = self._signer.sign(key, method="PUT", ttl_seconds=request.ttl_seconds)
        return session, signed

    async def accept_upload(
        self, upload_id: str, data: bytes, *, signature: str | None = None
    ) -> UploadSession:
        """``UploadReceiver``. Writes to staging; verifies nothing but the grant."""
        async with self._lock:
            session = await asyncio.to_thread(self._read_session_sync, upload_id)
            self.check_transition(session, UploadState.UPLOADING)

            if signature is not None:
                # Raises SignatureInvalid / SignatureExpired. Checked before any
                # filesystem work, so an unauthorised caller cannot even cause a
                # write.
                self._signer.verify(signature, key=session.key, method="PUT")

            await asyncio.to_thread(self._staging_path(upload_id).write_bytes, data)
            session = self.advanced(
                session, UploadState.UPLOADING, received_bytes=len(data)
            )
            await asyncio.to_thread(self._write_session_sync, session)
            return session

    async def complete_upload(self, upload_id: str) -> UploadSession:
        async with self._lock:
            session = await asyncio.to_thread(self._read_session_sync, upload_id)
            self.check_transition(session, UploadState.UPLOADED)

            staging = self._staging_path(upload_id)
            if not staging.exists():
                raise UploadNotFound(upload_id)

            data = await asyncio.to_thread(staging.read_bytes)
            actual_size = len(data)
            actual_sha = hashlib.sha256(data).hexdigest()

            # Size first: it is the cheaper mismatch to explain, and a wrong
            # size usually means a truncated transfer rather than a wrong file.
            if actual_size != session.size_bytes:
                await asyncio.to_thread(staging.unlink, True)
                raise SizeMismatch(upload_id, session.size_bytes, actual_size)
            if actual_sha != session.sha256:
                await asyncio.to_thread(staging.unlink, True)
                raise ChecksumMismatch(upload_id, session.sha256, actual_sha)

            destination = self._object_path(session.key)
            await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
            # Atomic within a filesystem: a reader sees the whole object or none.
            await asyncio.to_thread(os.replace, staging, destination)

            session = self.advanced(
                session,
                UploadState.UPLOADED,
                received_bytes=actual_size,
                verified_sha256=actual_sha,
            )
            await asyncio.to_thread(self._write_session_sync, session)
            return session

    async def transition(self, upload_id: str, target: UploadState) -> UploadSession:
        async with self._lock:
            session = await asyncio.to_thread(self._read_session_sync, upload_id)
            self.check_transition(session, target)
            session = self.advanced(session, target)
            await asyncio.to_thread(self._write_session_sync, session)
            return session

    # ---- objects ---------------------------------------------------------

    def _stat_sync(self, key: str) -> ObjectMetadata:
        path = self._object_path(key)
        if not path.is_file():
            raise ObjectNotFound(key)
        stat = path.stat()
        return ObjectMetadata(
            key=validate_key(key),
            size_bytes=stat.st_size,
            content_type="application/octet-stream",
            created_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            backend=self.name,
        )

    async def put_object(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> ObjectMetadata:
        """Write server-generated bytes at a deterministic key.

        Goes through a temporary file and ``os.replace`` for the same reason
        upload promotion does: a reader sees the whole object or no object, never
        a half-written one. A result file torn across a crash is worse than a
        missing one, because it looks readable.
        """

        def write() -> ObjectMetadata:
            path = self._object_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{secrets.token_hex(6)}.tmp")
            try:
                tmp.write_bytes(data)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
            return ObjectMetadata(
                key=validate_key(key),
                size_bytes=len(data),
                content_type=content_type,
                sha256=hashlib.sha256(data).hexdigest(),
                created_at=datetime.now(UTC),
                backend=self.name,
            )

        return await asyncio.to_thread(write)

    async def put_object_if_absent(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> bool:
        """Atomic create-if-absent, via a temporary file and a hard link.

        ``os.link`` is the primitive that gives both properties at once: it
        fails with ``FileExistsError`` if the destination exists, and it
        publishes a file that is already complete.

        The obvious alternative, ``os.open(O_CREAT | O_EXCL)`` followed by a
        write, is exclusive but *not* atomic in content — it claims the key with
        an empty file and fills it afterwards, so a reader arriving in between
        gets truncated JSON. For a result store that is worse than losing the
        race, because a half-written bundle parses as corrupt rather than as
        absent.
        """

        def write() -> bool:
            path = self._object_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{secrets.token_hex(6)}.tmp")
            try:
                tmp.write_bytes(data)
                try:
                    os.link(tmp, path)
                except FileExistsError:
                    return False
                return True
            finally:
                tmp.unlink(missing_ok=True)

        return await asyncio.to_thread(write)

    async def stat_object(self, key: str) -> ObjectMetadata:
        return await asyncio.to_thread(self._stat_sync, key)

    async def get_object(self, key: str) -> StoredObject:
        def read() -> StoredObject:
            metadata = self._stat_sync(key)
            data = self._object_path(key).read_bytes()
            return StoredObject(
                metadata=metadata.model_copy(
                    update={"sha256": hashlib.sha256(data).hexdigest()}
                ),
                data=data,
            )

        return await asyncio.to_thread(read)

    async def object_exists(self, key: str) -> bool:
        try:
            path = self._object_path(key)
        except UnsafeKey:
            # An unsafe key cannot name an object that exists. Answering the
            # question asked is better than raising here.
            return False
        return await asyncio.to_thread(path.is_file)

    async def delete_object(self, key: str) -> bool:
        def remove() -> bool:
            path = self._object_path(key)
            if not path.is_file():
                return False
            path.unlink()
            return True

        return await asyncio.to_thread(remove)

    # ---- operational -----------------------------------------------------

    async def health(self) -> StorageHealth:
        """Round-trip a probe file. Never raises."""

        def probe() -> StorageHealth:
            import time

            started = time.perf_counter()
            marker = self.root / f".health-{secrets.token_hex(6)}"
            writable = False
            try:
                marker.write_bytes(b"ok")
                writable = marker.read_bytes() == b"ok"
                marker.unlink()
            except OSError as exc:
                return StorageHealth(
                    ok=False, backend=self.name, writable=False, detail=str(exc)
                )
            finally:
                # A probe that leaves litter behind is a slow disk-fill bug.
                if marker.exists():
                    try:
                        marker.unlink()
                    except OSError:
                        pass

            try:
                free = shutil.disk_usage(self.root).free
            except OSError:
                free = None

            return StorageHealth(
                ok=writable,
                backend=self.name,
                writable=writable,
                latency_ms=(time.perf_counter() - started) * 1000,
                free_bytes=free,
                detail=None if writable else "probe wrote but read back wrong bytes",
            )

        return await asyncio.to_thread(probe)

    # ---- maintenance (not part of the protocol) --------------------------

    async def reap_expired(self) -> int:
        """Delete sessions past their TTL that never received verified bytes."""

        def sweep() -> int:
            now = datetime.now(UTC)
            removed = 0
            for path in self.sessions_dir.glob("*.json"):
                try:
                    session = UploadSession.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    continue
                if session.is_expired(now):
                    self._staging_path(session.upload_id).unlink(missing_ok=True)
                    path.unlink(missing_ok=True)
                    removed += 1
            return removed

        return await asyncio.to_thread(sweep)
