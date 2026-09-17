"""Reusable ``StorageBackend`` conformance suite.

Not a test module — a suite any backend can be run through. A future
``RedisSessionS3Backend``, a GCS backend, an encrypted-at-rest wrapper: each
imports this and proves it honours the same contract, rather than growing its
own tests that quietly assert weaker guarantees.

Usage::

    from tests.storage_conformance import run_conformance

    failures = asyncio.run(run_conformance(lambda: MyBackend(...), receiver=True))
    assert failures == []

Each check builds a **fresh backend from the factory**, so ordering between
checks cannot matter and a failure in one cannot cascade. The factory therefore
has to hand back an isolated store every call — for the filesystem backend that
means a fresh temporary directory.

``receiver=False`` skips the ingestion checks for backends that do not implement
``UploadReceiver`` (an object store, where the client PUTs directly). Everything
else still applies.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from typing import Any

from storage.base import StorageBackend, UploadReceiver
from storage.errors import (
    ChecksumMismatch,
    InvalidTransition,
    ObjectNotFound,
    SignatureInvalid,
    SizeMismatch,
    StorageError,
    UploadNotFound,
)
from storage.types import NewUpload, StorageHealth, UploadState

BackendFactory = Callable[[], Any]

PAYLOAD = b"deepfake-platform storage conformance payload \x00\xff binary safe"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _new_upload(**over) -> NewUpload:
    body = {
        "sha256": PAYLOAD_SHA,
        "size_bytes": len(PAYLOAD),
        "content_type": "application/octet-stream",
        "prefix": "uploads",
        "ttl_seconds": 900,
    }
    body.update(over)
    return NewUpload(**body)


async def _uploaded(backend: Any) -> Any:
    """Drive a backend to ``uploaded`` and return the session."""
    session, signed = await backend.create_upload(_new_upload())
    await backend.accept_upload(session.upload_id, PAYLOAD, signature=signed.signature)
    return await backend.complete_upload(session.upload_id)


# --------------------------------------------------------------------------
# Structural
# --------------------------------------------------------------------------


async def check_satisfies_protocol(make: BackendFactory) -> None:
    backend = make()
    assert isinstance(backend, StorageBackend), (
        f"{type(backend).__name__} does not satisfy StorageBackend"
    )
    assert isinstance(backend.name, str) and backend.name, "backend must have a name"


async def check_health_never_raises(make: BackendFactory) -> None:
    health = await make().health()
    assert isinstance(health, StorageHealth)
    assert isinstance(health.ok, bool)
    assert health.backend, "health must name its backend"


async def check_health_reports_writable(make: BackendFactory) -> None:
    health = await make().health()
    assert health.ok is True, f"expected a healthy backend, got: {health.detail}"
    assert health.writable is True, "a read-only store is degraded, not healthy"


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


async def check_create_upload_starts_in_created(make: BackendFactory) -> None:
    backend = make()
    session, signed = await backend.create_upload(_new_upload())
    assert session.state is UploadState.CREATED
    assert session.received_bytes is None, "no bytes have arrived yet"
    assert session.verified_sha256 is None
    assert signed.key == session.key, "the grant must authorise the session's key"
    assert signed.expires_at == session.expires_at or signed.expires_at is not None


async def check_session_is_readable_back(make: BackendFactory) -> None:
    backend = make()
    session, _ = await backend.create_upload(_new_upload())
    assert (await backend.get_session(session.upload_id)).upload_id == session.upload_id


async def check_unknown_session_raises(make: BackendFactory) -> None:
    try:
        await make().get_session("u_does_not_exist")
    except UploadNotFound:
        return
    raise AssertionError("expected UploadNotFound")


async def check_accept_moves_to_uploading(make: BackendFactory) -> None:
    backend = make()
    session, signed = await backend.create_upload(_new_upload())
    after = await backend.accept_upload(
        session.upload_id, PAYLOAD, signature=signed.signature
    )
    assert after.state is UploadState.UPLOADING
    assert after.received_bytes == len(PAYLOAD)
    assert after.verified_sha256 is None, "verification belongs to complete_upload"


async def check_complete_verifies_and_promotes(make: BackendFactory) -> None:
    backend = make()
    session = await _uploaded(backend)
    assert session.state is UploadState.UPLOADED
    assert session.verified_sha256 == PAYLOAD_SHA
    assert await backend.object_exists(session.key) is True


async def check_object_absent_until_completed(make: BackendFactory) -> None:
    """Unverified bytes must never be readable at the final key."""
    backend = make()
    session, signed = await backend.create_upload(_new_upload())
    assert await backend.object_exists(session.key) is False
    await backend.accept_upload(session.upload_id, PAYLOAD, signature=signed.signature)
    assert await backend.object_exists(session.key) is False, (
        "staged, unverified bytes were reachable at the final key"
    )
    await backend.complete_upload(session.upload_id)
    assert await backend.object_exists(session.key) is True


async def check_full_lifecycle_to_archived(make: BackendFactory) -> None:
    backend = make()
    session = await _uploaded(backend)
    session = await backend.transition(session.upload_id, UploadState.PROCESSING)
    assert session.state is UploadState.PROCESSING
    session = await backend.transition(session.upload_id, UploadState.ARCHIVED)
    assert session.state is UploadState.ARCHIVED
    assert session.state.is_terminal


async def check_illegal_transition_rejected(make: BackendFactory) -> None:
    backend = make()
    session, _ = await backend.create_upload(_new_upload())
    for target in (UploadState.UPLOADED, UploadState.PROCESSING, UploadState.ARCHIVED):
        try:
            await backend.transition(session.upload_id, target)
        except InvalidTransition:
            continue
        raise AssertionError(f"created -> {target.value} should be rejected")


async def check_transition_is_idempotent(make: BackendFactory) -> None:
    """Re-entering the current state is not an error; a retried call is not a bug."""
    backend = make()
    session = await _uploaded(backend)
    again = await backend.transition(session.upload_id, UploadState.UPLOADED)
    assert again.state is UploadState.UPLOADED


async def check_checksum_mismatch_rejected(make: BackendFactory) -> None:
    backend = make()
    session, signed = await backend.create_upload(
        _new_upload(sha256="0" * 64, size_bytes=len(PAYLOAD))
    )
    await backend.accept_upload(session.upload_id, PAYLOAD, signature=signed.signature)
    try:
        await backend.complete_upload(session.upload_id)
    except ChecksumMismatch:
        assert await backend.object_exists(session.key) is False, (
            "bytes that failed verification must not be promoted"
        )
        return
    raise AssertionError("expected ChecksumMismatch")


async def check_size_mismatch_rejected(make: BackendFactory) -> None:
    backend = make()
    session, signed = await backend.create_upload(_new_upload(size_bytes=len(PAYLOAD) + 99))
    await backend.accept_upload(session.upload_id, PAYLOAD, signature=signed.signature)
    try:
        await backend.complete_upload(session.upload_id)
    except SizeMismatch:
        assert await backend.object_exists(session.key) is False
        return
    raise AssertionError("expected SizeMismatch")


async def check_invalid_signature_rejected(make: BackendFactory) -> None:
    backend = make()
    session, _ = await backend.create_upload(_new_upload())
    try:
        await backend.accept_upload(session.upload_id, PAYLOAD, signature="forged.token")
    except SignatureInvalid:
        assert (await backend.get_session(session.upload_id)).state is UploadState.CREATED, (
            "a rejected signature must not advance the session"
        )
        return
    raise AssertionError("expected SignatureInvalid")


# --------------------------------------------------------------------------
# Objects
# --------------------------------------------------------------------------


async def check_round_trip_is_byte_exact(make: BackendFactory) -> None:
    backend = make()
    session = await _uploaded(backend)
    obj = await backend.get_object(session.key)
    assert obj.data == PAYLOAD, "round trip altered the bytes"
    assert obj.metadata.size_bytes == len(PAYLOAD)
    assert obj.metadata.sha256 == PAYLOAD_SHA
    assert len(obj) == len(PAYLOAD)


async def check_stat_does_not_require_reading(make: BackendFactory) -> None:
    backend = make()
    session = await _uploaded(backend)
    meta = await backend.stat_object(session.key)
    assert meta.key == session.key
    assert meta.size_bytes == len(PAYLOAD)
    assert meta.backend == backend.name


async def check_put_object_if_absent_is_exclusive(make: BackendFactory) -> None:
    """First writer wins; the second reports False and changes nothing."""
    backend = make()
    assert await backend.put_object_if_absent("conformance/excl", b"first") is True
    assert await backend.put_object_if_absent("conformance/excl", b"second") is False
    obj = await backend.get_object("conformance/excl")
    assert obj.data == b"first", "a losing write overwrote the winner"


async def check_put_object_if_absent_is_atomic_under_concurrency(
    make: BackendFactory,
) -> None:
    """Exactly one of many concurrent writers may claim a key.

    A check-then-write implementation passes the sequential case and fails here,
    which is the whole reason this check exists.
    """
    import asyncio

    backend = make()
    outcomes = await asyncio.gather(
        *(
            backend.put_object_if_absent("conformance/race", str(i).encode())
            for i in range(16)
        )
    )
    assert sum(outcomes) == 1, f"{sum(outcomes)} of 16 writers claimed the key"
    obj = await backend.get_object("conformance/race")
    assert obj.data == str(outcomes.index(True)).encode()


async def check_concurrent_writes_to_distinct_keys_all_succeed(
    make: BackendFactory,
) -> None:
    """Exclusivity is per key, not a global lock — and path handling must stay
    correct under load rather than rejecting valid keys at random."""
    import asyncio

    backend = make()
    outcomes = await asyncio.gather(
        *(backend.put_object_if_absent(f"conformance/many/{i}", b"x") for i in range(24))
    )
    assert all(outcomes), f"only {sum(outcomes)} of 24 distinct keys were written"


async def check_put_object_overwrites_but_if_absent_does_not(
    make: BackendFactory,
) -> None:
    """The two writes differ deliberately; a backend implementing one in terms
    of the other has lost a guarantee."""
    backend = make()
    await backend.put_object("conformance/both", b"one")
    await backend.put_object("conformance/both", b"two")
    assert (await backend.get_object("conformance/both")).data == b"two"
    assert await backend.put_object_if_absent("conformance/both", b"three") is False
    assert (await backend.get_object("conformance/both")).data == b"two"


async def check_missing_object_raises(make: BackendFactory) -> None:
    backend = make()
    try:
        await backend.get_object("uploads/nothing/here")
    except ObjectNotFound:
        return
    raise AssertionError("expected ObjectNotFound")


async def check_exists_is_false_for_missing(make: BackendFactory) -> None:
    assert await make().object_exists("uploads/nothing/here") is False


async def check_delete_is_idempotent(make: BackendFactory) -> None:
    backend = make()
    session = await _uploaded(backend)
    assert await backend.delete_object(session.key) is True
    assert await backend.delete_object(session.key) is False, (
        "deleting an absent object reports False, not an error"
    )
    assert await backend.object_exists(session.key) is False


async def check_deleted_object_is_unreadable(make: BackendFactory) -> None:
    backend = make()
    session = await _uploaded(backend)
    await backend.delete_object(session.key)
    try:
        await backend.get_object(session.key)
    except ObjectNotFound:
        return
    raise AssertionError("a deleted object was still readable")


async def check_hostile_keys_rejected(make: BackendFactory) -> None:
    """A storage layer that accepts these is an arbitrary-file primitive."""
    backend = make()
    hostile = [
        "../../../etc/passwd",
        "uploads/../../../etc/passwd",
        "/etc/passwd",
        "C:\\Windows\\System32\\config\\SAM",
        "uploads/..\\..\\secret",
        "",
        "uploads/\x00truncated",
    ]
    for key in hostile:
        assert await backend.object_exists(key) is False, f"{key!r} reported as existing"
        try:
            await backend.get_object(key)
        except StorageError:
            continue
        raise AssertionError(f"hostile key accepted by get_object: {key!r}")


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

CORE_CHECKS: list[Callable[[BackendFactory], Awaitable[None]]] = [
    check_satisfies_protocol,
    check_health_never_raises,
    check_health_reports_writable,
    check_create_upload_starts_in_created,
    check_session_is_readable_back,
    check_unknown_session_raises,
    check_missing_object_raises,
    check_exists_is_false_for_missing,
    check_hostile_keys_rejected,
    check_put_object_if_absent_is_exclusive,
    check_put_object_if_absent_is_atomic_under_concurrency,
    check_concurrent_writes_to_distinct_keys_all_succeed,
    check_put_object_overwrites_but_if_absent_does_not,
]

RECEIVER_CHECKS: list[Callable[[BackendFactory], Awaitable[None]]] = [
    check_accept_moves_to_uploading,
    check_complete_verifies_and_promotes,
    check_object_absent_until_completed,
    check_full_lifecycle_to_archived,
    check_illegal_transition_rejected,
    check_transition_is_idempotent,
    check_checksum_mismatch_rejected,
    check_size_mismatch_rejected,
    check_invalid_signature_rejected,
    check_round_trip_is_byte_exact,
    check_stat_does_not_require_reading,
    check_delete_is_idempotent,
    check_deleted_object_is_unreadable,
]


def checks_for(*, receiver: bool) -> list[Callable[[BackendFactory], Awaitable[None]]]:
    return CORE_CHECKS + (RECEIVER_CHECKS if receiver else [])


async def run_conformance(make: BackendFactory, *, receiver: bool = True) -> list[str]:
    """Run the suite. Returns a list of failure descriptions; empty means conformant."""
    probe = make()
    if receiver and not isinstance(probe, UploadReceiver):
        return [
            f"{type(probe).__name__} was declared a receiver but does not "
            "satisfy UploadReceiver"
        ]

    failures: list[str] = []
    for check in checks_for(receiver=receiver):
        try:
            await check(make)
        except Exception as exc:  # noqa: BLE001 - a suite reports everything
            failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
    return failures
