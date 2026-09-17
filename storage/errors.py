"""Storage-domain exceptions.

These carry no HTTP status and no ``ErrorCode``. That mapping belongs to the
layer that speaks HTTP, and putting it here would invert the dependency the
whole package is arranged to avoid — storage would then need ``contracts``, and
through it a vocabulary defined by an API it should be able to outlive.

The backend layer will map these to contract errors when it adopts this package
at S3. Until then they are a complete, self-describing taxonomy.
"""

from __future__ import annotations


class StorageError(Exception):
    """Base for everything this package raises."""


class ObjectNotFound(StorageError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"no object at key {key!r}")


class UploadNotFound(StorageError):
    def __init__(self, upload_id: str) -> None:
        self.upload_id = upload_id
        super().__init__(f"no upload session {upload_id!r}")


class UploadExpired(StorageError):
    def __init__(self, upload_id: str) -> None:
        self.upload_id = upload_id
        super().__init__(f"upload session {upload_id!r} has expired")


class InvalidTransition(StorageError):
    """An illegal move through the upload lifecycle.

    Raised rather than silently corrected: a caller trying to archive an upload
    that never received bytes has a bug, and quietly permitting it would hide
    the bug behind an object that does not exist.
    """

    def __init__(self, upload_id: str, current: str, target: str) -> None:
        self.upload_id = upload_id
        self.current = current
        self.target = target
        super().__init__(
            f"upload {upload_id!r} cannot move from {current!r} to {target!r}"
        )


class ChecksumMismatch(StorageError):
    """Received bytes do not hash to the declared checksum.

    Always fatal for that upload. The bytes are not what the caller said they
    were, and no retry of the same request changes that.
    """

    def __init__(self, upload_id: str, declared: str, actual: str) -> None:
        self.upload_id = upload_id
        self.declared = declared
        self.actual = actual
        super().__init__(
            f"upload {upload_id!r}: declared sha256 {declared[:12]}… "
            f"but received {actual[:12]}…"
        )


class SizeMismatch(StorageError):
    def __init__(self, upload_id: str, declared: int, actual: int) -> None:
        self.upload_id = upload_id
        self.declared = declared
        self.actual = actual
        super().__init__(
            f"upload {upload_id!r}: declared {declared} bytes but received {actual}"
        )


class SignatureInvalid(StorageError):
    """The signature does not verify. Deliberately says nothing about why.

    Distinguishing "wrong key" from "tampered payload" in the message would hand
    an attacker a probing oracle.
    """

    def __init__(self, detail: str = "signature verification failed") -> None:
        super().__init__(detail)


class SignatureExpired(StorageError):
    """Separate from ``SignatureInvalid`` because the remedy differs: an expired
    URL means request a new one, not that something is wrong."""

    def __init__(self) -> None:
        super().__init__("signed URL has expired")


class UnsupportedOperation(StorageError):
    """The backend cannot perform this operation by design.

    Raised by object-store backends for ``accept_upload``: bytes go straight to
    the store via the presigned URL and never pass through this process.
    """

    def __init__(self, backend: str, operation: str) -> None:
        self.backend = backend
        self.operation = operation
        super().__init__(f"{backend} does not support {operation}()")


class StorageUnavailable(StorageError):
    """The backing store cannot be reached or written to."""

    def __init__(self, backend: str, detail: str) -> None:
        self.backend = backend
        self.detail = detail
        super().__init__(f"{backend} unavailable: {detail}")
