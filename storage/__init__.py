"""Storage subsystem.

Independent by construction: this package imports the standard library and
pydantic, and nothing else in the repository. No FastAPI, no workers, no models,
and not ``contracts`` either — a backend that moves opaque bytes has no business
knowing what a deepfake is.

The consequence worth having is that the whole upload lifecycle can be
exercised, and every backend swapped, with no web framework in the process.

Usage::

    from storage import LocalStorageBackend, NewUpload, HmacUrlSigner

    backend = LocalStorageBackend(root, signer=HmacUrlSigner(secret))
    session, url = await backend.create_upload(NewUpload(...))
    await backend.accept_upload(session.upload_id, data, signature=url.signature)
    await backend.complete_upload(session.upload_id)
    obj = await backend.get_object(session.key)
"""

from storage.base import BaseStorageBackend, StorageBackend, UploadReceiver
from storage.errors import (
    ChecksumMismatch,
    InvalidTransition,
    ObjectNotFound,
    SignatureExpired,
    SignatureInvalid,
    SizeMismatch,
    StorageError,
    StorageUnavailable,
    UnsupportedOperation,
    UploadExpired,
    UploadNotFound,
)
from storage.signing import (
    HmacUrlSigner,
    NullUrlSigner,
    SignedClaims,
    SignedUrl,
    UrlSigner,
)
from storage.types import (
    TRANSITIONS,
    NewUpload,
    ObjectMetadata,
    StorageHealth,
    StoredObject,
    UploadSession,
    UploadState,
)

__all__ = [
    "TRANSITIONS",
    "BaseStorageBackend",
    "ChecksumMismatch",
    "HmacUrlSigner",
    "InvalidTransition",
    "NewUpload",
    "NullUrlSigner",
    "ObjectMetadata",
    "ObjectNotFound",
    "SignatureExpired",
    "SignatureInvalid",
    "SignedClaims",
    "SignedUrl",
    "SizeMismatch",
    "StorageBackend",
    "UploadReceiver",
    "StorageError",
    "StorageHealth",
    "StorageUnavailable",
    "StoredObject",
    "UnsupportedOperation",
    "UploadExpired",
    "UploadNotFound",
    "UploadSession",
    "UploadState",
    "UrlSigner",
]
