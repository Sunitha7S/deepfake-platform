"""Storage value types and the upload lifecycle.

This package depends on the standard library and pydantic. Nothing else — no
FastAPI, no workers, no models, and deliberately not even ``contracts``.

The omission of ``contracts`` is a design choice, not an oversight. A storage
backend moves opaque bytes; whether those bytes are an image or a video is the
caller's concern. Importing ``MediaKind`` here would couple the storage
subsystem to a detection product it has no reason to know about, and would stop
it being reusable or testable on its own.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UploadState(StrEnum):
    """The upload lifecycle.

    Strictly linear. Each state means something an operator can act on:

    ``created``    a slot has been reserved and a signed URL issued; no bytes yet.
    ``uploading``  bytes are arriving or have arrived, unverified.
    ``uploaded``   bytes are verified against the declared size and checksum,
                   and promoted to their final key. Safe to read.
    ``processing`` claimed by a consumer. Prevents a second consumer claiming it.
    ``archived``   retained but cold; no longer served on the hot path. Terminal.

    Retries do not add a state: a consumer retrying its work stays in
    ``processing``, and an abandoned upload is reaped by ``expires_at`` rather
    than moved to a state of its own. Both keep the machine to the five states
    that carry distinct meaning.
    """

    CREATED = "created"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    ARCHIVED = "archived"

    @property
    def is_terminal(self) -> bool:
        return self is UploadState.ARCHIVED

    @property
    def has_bytes(self) -> bool:
        """Whether an object is expected to exist for this session."""
        return self in (
            UploadState.UPLOADED,
            UploadState.PROCESSING,
            UploadState.ARCHIVED,
        )

    def can_transition_to(self, target: UploadState) -> bool:
        return target in TRANSITIONS[self]


TRANSITIONS: dict[UploadState, frozenset[UploadState]] = {
    UploadState.CREATED: frozenset({UploadState.UPLOADING}),
    UploadState.UPLOADING: frozenset({UploadState.UPLOADED}),
    UploadState.UPLOADED: frozenset({UploadState.PROCESSING}),
    UploadState.PROCESSING: frozenset({UploadState.ARCHIVED}),
    UploadState.ARCHIVED: frozenset(),
}
"""The whole state machine, in one readable place.

Kept as data rather than scattered ``if`` statements so that the legal paths can
be read at a glance and asserted in a test, instead of being an emergent
property of code in three files.
"""


class NewUpload(Base):
    """A request to reserve storage. Carries no bytes."""

    sha256: str = Field(pattern=SHA256_PATTERN, description="Checksum the caller expects.")
    size_bytes: int = Field(ge=1, description="Declared size. Verified on completion.")
    content_type: str = Field(description="Declared MIME type. Stored as object metadata.")
    prefix: str = Field(
        default="uploads",
        description="Key namespace. Lifecycle rules are usually written against this.",
    )
    ttl_seconds: int = Field(
        default=900, ge=30, le=86_400, description="How long the signed URL stays valid."
    )


class ObjectMetadata(Base):
    """What the backend knows about a stored object without reading it."""

    key: str
    size_bytes: int = Field(ge=0)
    content_type: str
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    created_at: datetime
    backend: str


class StoredObject(Base):
    """An object and its metadata.

    Bytes are returned whole. Streaming is a later concern, and pretending
    otherwise now would complicate every call site for a benefit nothing in the
    system currently needs.
    """

    metadata: ObjectMetadata
    data: bytes

    @property
    def key(self) -> str:
        return self.metadata.key

    def __len__(self) -> int:
        return len(self.data)


class UploadSession(Base):
    """The record of one upload, from reservation to archive."""

    upload_id: str
    key: str = Field(description="Final object key. Stable from creation.")
    staging_key: str = Field(description="Where unverified bytes land first.")
    state: UploadState
    sha256: str = Field(pattern=SHA256_PATTERN, description="Declared by the caller.")
    size_bytes: int = Field(ge=1, description="Declared by the caller.")
    content_type: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime = Field(description="After this, an incomplete upload is reapable.")
    received_bytes: int | None = Field(
        default=None, ge=0, description="Actual bytes written. Set once they arrive."
    )
    verified_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
        description="Computed over the received bytes. Set on completion.",
    )
    backend: str

    def is_expired(self, at: datetime) -> bool:
        """Only an incomplete upload can expire. Verified bytes are not garbage."""
        return not self.state.has_bytes and at >= self.expires_at


class StorageHealth(Base):
    """Backing-store condition, in a shape that survives changing the backend.

    Mirrors ``QueueHealth`` deliberately: an operator reading a health page
    should not have to learn two vocabularies for the same question.
    """

    ok: bool
    backend: str
    latency_ms: float | None = Field(default=None, ge=0.0)
    writable: bool = Field(default=False, description="A read-only store is degraded, not healthy.")
    detail: str | None = Field(default=None, description="Why it is unhealthy. Operator-facing.")
    free_bytes: int | None = Field(default=None, ge=0)
