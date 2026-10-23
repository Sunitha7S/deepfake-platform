"""Backend selection.

The only module that names a concrete storage class. Everything else depends on
``StorageBackend``, so switching from a filesystem to an object store is a
config change here and nowhere else.

``StorageConfig`` is a plain pydantic model rather than a settings object: this
package must not read the environment or know that a ``DFP_`` prefix exists.
The application layer maps its own settings into this and passes it in, which
keeps the dependency pointing the right way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from storage.base import StorageBackend
from storage.local import LocalStorageBackend
from storage.s3 import S3CompatibleStorageBackend
from storage.signing import HmacUrlSigner, UrlSigner

BackendKind = Literal["local", "s3"]

UNIMPLEMENTED_KINDS: frozenset[str] = frozenset({"s3"})
"""Backends that are still skeletons raising ``NotImplementedError``.

``build_storage`` constructs them without complaint, so a deployment pointed at
one starts cleanly and fails on its first request. ``backend.config`` refuses
them in production. Remove a kind from this set in the change that implements
it; ``tests/test_unimplemented_backends.py`` fails until you do."""


class StorageConfig(BaseModel):
    """How to build a backend. Carries no environment knowledge."""

    model_config = ConfigDict(extra="forbid")

    kind: BackendKind = "local"
    signing_secret: str = Field(
        min_length=16,
        description=(
            "HMAC key for signed URLs. Minimum length enforced here as well as "
            "in the signer, so a weak secret fails at config parse rather than "
            "at first use."
        ),
    )
    public_base_url: str = Field(
        default="", description="Origin prepended to signed URLs. No trailing path."
    )

    # local
    root: Path | None = Field(default=None, description="Required when kind is 'local'.")

    # s3
    bucket: str | None = None
    endpoint_url: str | None = Field(
        default=None, description="Set for MinIO, R2 or B2; leave unset for AWS."
    )
    region: str = "us-east-1"
    access_key_id: str | None = None
    secret_access_key: str | None = None

    @model_validator(mode="after")
    def _required_fields_per_kind(self) -> StorageConfig:
        if self.kind == "local" and self.root is None:
            raise ValueError("root is required when kind='local'")
        if self.kind == "s3" and not self.bucket:
            raise ValueError("bucket is required when kind='s3'")
        return self

    def __repr__(self) -> str:
        """Never render the secrets. A config that prints its own signing key is
        how credentials end up pasted into a bug report."""
        return (
            f"StorageConfig(kind={self.kind!r}, root={self.root!r}, "
            f"bucket={self.bucket!r}, endpoint_url={self.endpoint_url!r})"
        )

    __str__ = __repr__


def build_signer(config: StorageConfig) -> UrlSigner:
    return HmacUrlSigner(config.signing_secret, base_url=config.public_base_url)


def build_storage(config: StorageConfig) -> StorageBackend:
    """Return a backend. Typed as the protocol, never as a concrete class."""
    signer = build_signer(config)

    if config.kind == "local":
        assert config.root is not None  # guaranteed by the validator
        return LocalStorageBackend(
            config.root, signer=signer, base_url=config.public_base_url
        )

    return S3CompatibleStorageBackend(
        bucket=config.bucket or "",
        endpoint_url=config.endpoint_url,
        region=config.region,
        access_key_id=config.access_key_id,
        secret_access_key=config.secret_access_key,
        signer=signer,
    )
