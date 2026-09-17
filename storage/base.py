"""Storage protocols and the lifecycle rules every backend shares.

Two protocols, deliberately separate.

``StorageBackend`` owns durable objects and the upload lifecycle. Eight
operations: the six specified, plus ``transition`` (``processing`` and
``archived`` are reachable by no other operation, and one table-validated move
beats two near-identical methods) and ``get_session`` (state must be readable,
or callers cache it and drift).

``UploadReceiver`` is the server half of a signed URL, and it is a separate,
optional protocol rather than a ninth method on ``StorageBackend``. An object
store receives bytes directly through the presigned PUT — this process never
sees them — so requiring S3 to declare ``accept_upload`` would force it to
implement a method it can never perform. That is a refused bequest, and the
interface-segregation objection to it is sound: a caller reading the protocol
would reasonably conclude every backend can ingest bytes, and be wrong.

Splitting them raises one fair question — who owns session state, if two
objects touch the same upload? ``StorageBackend`` does, exclusively.
``UploadReceiver`` only moves bytes into staging and reports the resulting
session; every state decision stays on the one object that persists it. A local
backend implements both protocols and is its own receiver; an object store
implements only the first.

Transition validation lives in ``BaseStorageBackend`` rather than in each
implementation, so a new backend cannot invent its own state machine by
accident.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from storage.errors import InvalidTransition, UploadExpired
from storage.signing import SignedUrl
from storage.types import (
    NewUpload,
    ObjectMetadata,
    StorageHealth,
    StoredObject,
    UploadSession,
    UploadState,
)


@runtime_checkable
class StorageBackend(Protocol):
    """Everything the platform needs from durable storage.

    Every method is async. The local implementation gains nothing from it today,
    but S3 and any network store are async, and a synchronous protocol now would
    force every call site to change when one arrives — the rework this seam
    exists to prevent.
    """

    name: str

    # ---- upload lifecycle ------------------------------------------------

    async def create_upload(self, request: NewUpload) -> tuple[UploadSession, SignedUrl]:
        """Reserve a key and issue a time-limited grant. Writes no bytes."""
        ...

    async def complete_upload(self, upload_id: str) -> UploadSession:
        """Verify size and checksum, then promote staging to the final key.

        ``uploading`` → ``uploaded``. Verification happens here and not earlier
        because only here are all the bytes present. A backend that trusts the
        declared checksum is storing something it cannot vouch for.
        """
        ...

    async def transition(self, upload_id: str, target: UploadState) -> UploadSession:
        """Advance the lifecycle. Rejects any move outside ``TRANSITIONS``."""
        ...

    async def get_session(self, upload_id: str) -> UploadSession:
        """Raises ``UploadNotFound``."""
        ...

    # ---- objects ---------------------------------------------------------

    async def put_object(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> ObjectMetadata:
        """Store bytes the *server* generated, at a caller-chosen key.

        Distinct from the upload lifecycle, which exists to receive bytes from
        an untrusted client at a key the backend chooses. Results, heat maps and
        spectra are produced here and must land at a deterministic key so they
        can be found again by job id — an upload session would assign an opaque
        one and defeat that.

        Overwrites an existing key. Callers needing first-writer-wins check
        ``object_exists`` first and accept the race, or use a backend offering a
        conditional put.
        """
        ...

    async def put_object_if_absent(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> bool:
        """Store bytes only if the key is free. True if this caller wrote them.

        The atomic counterpart to ``put_object``. A check-then-write built from
        ``object_exists`` followed by ``put_object`` has a window between the two
        in which a second writer can also observe the key as free, and both then
        write — with the *later* one winning, which is the opposite of what a
        first-writer-wins contract promises.

        Returning ``False`` is an ordinary outcome, not an error: losing a race
        to store an identical result is not a failure, and raising would force
        every caller to catch an exception for the common case.
        """
        ...

    async def get_object(self, key: str) -> StoredObject:
        """Raises ``ObjectNotFound``."""
        ...

    async def object_exists(self, key: str) -> bool:
        """Never raises for a missing object; that is the question being asked."""
        ...

    async def delete_object(self, key: str) -> bool:
        """True if something was deleted, False if it was already gone.

        Idempotent: deleting an absent object is a success, because the caller's
        intent — that the object not exist — is satisfied either way.
        """
        ...

    async def stat_object(self, key: str) -> ObjectMetadata:
        """Metadata without transferring the bytes. Raises ``ObjectNotFound``."""
        ...

    # ---- operational -----------------------------------------------------

    async def health(self) -> StorageHealth:
        """Probe the store. Must never raise.

        An unreachable store is reported as ``ok=False`` with a reason. A health
        check that throws converts a degraded dependency into a 500 and hides
        which dependency it was.
        """
        ...


@runtime_checkable
class UploadReceiver(Protocol):
    """Backends that ingest bytes in-process.

    Optional and separate: implement it when the client cannot write to the
    store directly. A local filesystem deployment must; an object store must
    not, because its presigned URL already does the job and routing bytes
    through this process would throw away the entire benefit.

    Callers detect support structurally::

        if isinstance(backend, UploadReceiver):
            await backend.accept_upload(upload_id, data, signature=sig)
        else:
            ...  # client PUTs to the signed URL itself
    """

    name: str

    async def accept_upload(
        self, upload_id: str, data: bytes, *, signature: str | None = None
    ) -> UploadSession:
        """Write bytes to staging and report the session. ``created`` → ``uploading``.

        Verifies the signature before touching the filesystem. Does not verify
        the checksum — that is ``complete_upload``'s job, and only it knows all
        the bytes have arrived.
        """
        ...


class BaseStorageBackend:
    """Lifecycle rules shared by every backend.

    Not an abstract base in the inheritance-heavy sense — implementations
    satisfy ``StorageBackend`` structurally and could ignore this entirely. It
    exists so the state machine is enforced in one place instead of being
    re-derived, slightly differently, in each backend.
    """

    name: str = "base"

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC)

    def check_transition(self, session: UploadSession, target: UploadState) -> None:
        """Raise unless this move is legal and the session is still usable."""
        if session.is_expired(self.now()):
            raise UploadExpired(session.upload_id)
        if session.state is target:
            # Idempotent re-entry is not an error, but it is not a transition
            # either — callers that retry a network hiccup should not be
            # punished for it.
            return
        if not session.state.can_transition_to(target):
            raise InvalidTransition(session.upload_id, session.state.value, target.value)

    def advanced(self, session: UploadSession, target: UploadState, **fields) -> UploadSession:
        """Return a copy moved to ``target``, with ``updated_at`` refreshed."""
        return session.model_copy(
            update={"state": target, "updated_at": self.now(), **fields}
        )
