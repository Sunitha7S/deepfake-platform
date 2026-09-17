"""Signed URL abstraction.

An upload URL has to be safe to hand to a browser: it authorises exactly one
key, one method, and only for a while. S3 and its clones already do this;
a local deployment has to implement it. ``UrlSigner`` is the seam so callers
cannot tell which they are talking to.

``HmacUrlSigner`` is the local implementation. It is a real HMAC over the
authorised claims, not a decorative token — a local development deployment that
accepts any URL is a directory-traversal bug waiting for someone to point a
fuzzer at it, and the cost of doing it properly is thirty lines.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from pydantic import Field

from storage.errors import SignatureExpired, SignatureInvalid
from storage.types import Base


class SignedUrl(Base):
    """A time-limited authorisation to perform one operation on one key."""

    url: str
    method: str = Field(default="PUT", description="HTTP method the signature authorises.")
    key: str
    headers: dict[str, str] = Field(
        default_factory=dict, description="Must be sent verbatim by the client."
    )
    expires_at: datetime
    signature: str = Field(description="Opaque to the client; verified by the backend.")

    def is_expired(self, at: datetime | None = None) -> bool:
        return (at or datetime.now(UTC)) >= self.expires_at


class SignedClaims(Base):
    """What a signature attests to. Encoded into the token itself."""

    key: str
    method: str
    expires_at_epoch: int

    @property
    def expires_at(self) -> datetime:
        return datetime.fromtimestamp(self.expires_at_epoch, tz=UTC)


@runtime_checkable
class UrlSigner(Protocol):
    """Issues and verifies time-limited operation grants."""

    def sign(self, key: str, *, method: str = "PUT", ttl_seconds: int = 900) -> SignedUrl:
        """Authorise ``method`` on ``key`` for ``ttl_seconds``."""
        ...

    def verify(self, signature: str, *, key: str, method: str = "PUT") -> SignedClaims:
        """Raise ``SignatureInvalid`` or ``SignatureExpired``; else return claims."""
        ...


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class HmacUrlSigner:
    """HMAC-SHA256 signer for backends that must police their own uploads.

    The token is ``<payload>.<mac>``: claims travel with the signature, so
    verification needs no server-side session lookup and stays correct across a
    restart or a second process.
    """

    ALGORITHM = "sha256"

    def __init__(self, secret: str, *, base_url: str = "") -> None:
        if len(secret) < 16:
            # A short secret makes the whole mechanism theatre. Fail at
            # construction, where it is obvious, rather than shipping a signer
            # that looks like it is protecting something.
            raise ValueError("signing secret must be at least 16 characters")
        self._secret = secret.encode("utf-8")
        self._base_url = base_url.rstrip("/")

    def _mac(self, payload: bytes) -> str:
        return _b64e(hmac.new(self._secret, payload, hashlib.sha256).digest())

    def sign(self, key: str, *, method: str = "PUT", ttl_seconds: int = 900) -> SignedUrl:
        expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
        claims = SignedClaims(
            key=key,
            method=method.upper(),
            expires_at_epoch=int(expires_at.timestamp()),
        )
        payload = json.dumps(claims.model_dump(), sort_keys=True, separators=(",", ":")).encode()
        token = f"{_b64e(payload)}.{self._mac(payload)}"
        return SignedUrl(
            url=f"{self._base_url}/{key}?signature={token}" if self._base_url else key,
            method=claims.method,
            key=key,
            expires_at=expires_at,
            signature=token,
        )

    def verify(self, signature: str, *, key: str, method: str = "PUT") -> SignedClaims:
        try:
            encoded_payload, mac = signature.split(".", 1)
            payload = _b64d(encoded_payload)
        except (ValueError, TypeError) as exc:
            raise SignatureInvalid() from exc

        # Constant-time comparison. A timing-variable check on a MAC is a
        # textbook oracle, and `==` on strings short-circuits.
        if not hmac.compare_digest(self._mac(payload), mac):
            raise SignatureInvalid()

        try:
            claims = SignedClaims.model_validate_json(payload)
        except Exception as exc:  # noqa: BLE001 - malformed payload is invalid
            raise SignatureInvalid() from exc

        # Only checked after the MAC verifies: reading unauthenticated claims to
        # decide anything, including whether they expired, trusts data an
        # attacker controls.
        if claims.key != key or claims.method != method.upper():
            raise SignatureInvalid()
        if time.time() >= claims.expires_at_epoch:
            raise SignatureExpired()
        return claims


class NullUrlSigner:
    """Issues unsigned grants that always verify. Tests only.

    Never wire this into anything reachable from a network. It exists so a test
    can exercise upload flow without carrying a secret, and its name is
    deliberately alarming.
    """

    def sign(self, key: str, *, method: str = "PUT", ttl_seconds: int = 900) -> SignedUrl:
        return SignedUrl(
            url=key,
            method=method.upper(),
            key=key,
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl_seconds),
            signature="unsigned",
        )

    def verify(self, signature: str, *, key: str, method: str = "PUT") -> SignedClaims:
        return SignedClaims(
            key=key,
            method=method.upper(),
            expires_at_epoch=int((datetime.now(UTC) + timedelta(seconds=900)).timestamp()),
        )
