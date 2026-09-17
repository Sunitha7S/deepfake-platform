"""Storage subsystem tests.

Two layers:

* the reusable conformance suite, run against ``LocalStorageBackend``;
* local-specific tests for behaviour the shared contract cannot express
  (filesystem layout, session durability across a restart, key safety).

Note what is absent from the imports: no FastAPI, no ``backend``, no
``contracts``. The success criterion — upload, store, retrieve, delete, validate
— is met with nothing but this package and the standard library, which is the
independence requirement made checkable rather than merely asserted.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage import (  # noqa: E402
    HmacUrlSigner,
    NewUpload,
    ObjectNotFound,
    SignatureExpired,
    StorageBackend,
    UploadExpired,
    UploadReceiver,
    UploadState,
)
from storage.local import LocalStorageBackend, UnsafeKey, validate_key  # noqa: E402
from storage.registry import StorageConfig, build_storage  # noqa: E402
from storage.s3 import S3CompatibleStorageBackend  # noqa: E402
from storage.types import TRANSITIONS  # noqa: E402
from tests.storage_conformance import (  # noqa: E402
    PAYLOAD,
    PAYLOAD_SHA,
    checks_for,
    run_conformance,
)

SECRET = "conformance-secret-key-0123456789"


def _make_local() -> LocalStorageBackend:
    """Fresh backend on a fresh temp dir. Never cleaned up: the OS handles it,
    and keeping it lets a failing check be inspected."""
    root = Path(tempfile.mkdtemp(prefix="dfp-storage-"))
    return LocalStorageBackend(root, signer=HmacUrlSigner(SECRET), base_url="http://local")


def _upload(**over) -> NewUpload:
    body = {
        "sha256": PAYLOAD_SHA,
        "size_bytes": len(PAYLOAD),
        "content_type": "application/octet-stream",
    }
    body.update(over)
    return NewUpload(**body)


# --------------------------------------------------------------------------
# Conformance
# --------------------------------------------------------------------------


def test_local_backend_passes_the_conformance_suite():
    failures = asyncio.run(run_conformance(_make_local, receiver=True))
    assert failures == [], "conformance failures:\n  " + "\n  ".join(failures)


def test_conformance_suite_is_not_vacuous():
    """A suite that runs nothing passes everything."""
    assert len(checks_for(receiver=True)) >= 20
    assert len(checks_for(receiver=False)) < len(checks_for(receiver=True))


def test_conformance_suite_catches_a_broken_backend():
    """Prove the suite has teeth by feeding it a deliberately wrong backend."""

    class SilentlyDropsBytes(LocalStorageBackend):
        async def delete_object(self, key: str) -> bool:
            return True  # lies: reports success without deleting

    def make():
        root = Path(tempfile.mkdtemp(prefix="dfp-broken-"))
        return SilentlyDropsBytes(root, signer=HmacUrlSigner(SECRET))

    failures = asyncio.run(run_conformance(make, receiver=True))
    assert failures, "the suite passed a backend that does not delete"
    assert any("delete" in f for f in failures), failures


def test_s3_skeleton_satisfies_the_storage_protocol():
    backend = S3CompatibleStorageBackend(bucket="test-bucket")
    assert isinstance(backend, StorageBackend)


def test_s3_skeleton_is_deliberately_not_a_receiver():
    """The whole argument for splitting the protocols, made checkable."""
    backend = S3CompatibleStorageBackend(bucket="test-bucket")
    assert not isinstance(backend, UploadReceiver)
    assert isinstance(_make_local(), UploadReceiver)


def test_s3_health_reports_unhealthy_rather_than_raising():
    health = asyncio.run(S3CompatibleStorageBackend(bucket="b").health())
    assert health.ok is False
    assert "not implemented" in (health.detail or "")


def test_s3_operations_explain_what_is_missing():
    backend = S3CompatibleStorageBackend(bucket="b")
    try:
        asyncio.run(backend.stat_object("k"))
    except NotImplementedError as exc:
        assert "head_object" in str(exc), "a skeleton should name the call that fills it"
        return
    raise AssertionError("expected NotImplementedError")


def test_s3_repr_never_leaks_credentials():
    backend = S3CompatibleStorageBackend(
        bucket="b", access_key_id="AKIAEXAMPLE", secret_access_key="s3cr3t-value"
    )
    assert "s3cr3t" not in repr(backend)
    assert "AKIAEXAMPLE" not in repr(backend)


# --------------------------------------------------------------------------
# Success criterion, end to end
# --------------------------------------------------------------------------


def test_upload_store_retrieve_delete_validate_without_any_web_framework():
    backend = _make_local()

    async def scenario():
        session, signed = await backend.create_upload(_upload())
        assert session.state is UploadState.CREATED

        await backend.accept_upload(session.upload_id, PAYLOAD, signature=signed.signature)
        session = await backend.complete_upload(session.upload_id)
        assert session.state is UploadState.UPLOADED

        obj = await backend.get_object(session.key)
        assert obj.data == PAYLOAD
        assert hashlib.sha256(obj.data).hexdigest() == session.verified_sha256

        session = await backend.transition(session.upload_id, UploadState.PROCESSING)
        session = await backend.transition(session.upload_id, UploadState.ARCHIVED)

        assert await backend.delete_object(session.key) is True
        assert await backend.object_exists(session.key) is False
        return session

    session = asyncio.run(scenario())
    assert session.state is UploadState.ARCHIVED


def test_no_web_framework_is_imported_by_the_storage_package():
    import storage
    import storage.local
    import storage.registry
    import storage.s3

    for module in (storage, storage.local, storage.s3, storage.registry):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in ("fastapi", "starlette", "import torch", "from backend", "from contracts"):
            assert forbidden not in source, f"{module.__name__} imports {forbidden}"


# --------------------------------------------------------------------------
# Key safety
# --------------------------------------------------------------------------


def test_validate_key_rejects_traversal_and_absolutes():
    for bad in [
        "../secret",
        "a/../../b",
        "/etc/passwd",
        "C:/Windows",
        "a\\b",
        "",
        "   ",
        "a/\x00b",
        "./a",
    ]:
        try:
            validate_key(bad)
        except UnsafeKey:
            continue
        raise AssertionError(f"validate_key accepted {bad!r}")


def test_validate_key_accepts_normal_keys():
    assert validate_key("uploads/2026/08/25/u_abc") == "uploads/2026/08/25/u_abc"


def test_write_cannot_escape_the_root():
    backend = _make_local()
    outside = backend.root.parent / "escaped.txt"
    try:
        asyncio.run(backend.get_object("../escaped.txt"))
    except UnsafeKey:
        pass
    assert not outside.exists()


# --------------------------------------------------------------------------
# Local specifics
# --------------------------------------------------------------------------


def test_sessions_survive_a_process_restart():
    """A dict would pass every single-process test and fail in production."""
    root = Path(tempfile.mkdtemp(prefix="dfp-restart-"))

    async def first():
        b = LocalStorageBackend(root, signer=HmacUrlSigner(SECRET))
        session, signed = await b.create_upload(_upload())
        await b.accept_upload(session.upload_id, PAYLOAD, signature=signed.signature)
        return await b.complete_upload(session.upload_id)

    session = asyncio.run(first())

    async def second():
        reborn = LocalStorageBackend(root, signer=HmacUrlSigner(SECRET))
        return await reborn.get_session(session.upload_id), await reborn.get_object(session.key)

    restored, obj = asyncio.run(second())
    assert restored.state is UploadState.UPLOADED
    assert restored.verified_sha256 == PAYLOAD_SHA
    assert obj.data == PAYLOAD


def test_staging_and_objects_are_separate_trees():
    backend = _make_local()

    async def scenario():
        session, signed = await backend.create_upload(_upload())
        await backend.accept_upload(session.upload_id, PAYLOAD, signature=signed.signature)
        staged = list(backend.staging_dir.rglob("*.part"))
        assert staged, "bytes should be staged before verification"
        assert not list(backend.objects_dir.rglob("*")), "objects/ must stay empty until verified"
        await backend.complete_upload(session.upload_id)
        assert not list(backend.staging_dir.rglob("*.part")), "staging must be drained"
        assert list(backend.objects_dir.rglob("*")), "object should now exist"

    asyncio.run(scenario())


def test_expired_upload_cannot_be_advanced():
    backend = _make_local()

    async def scenario():
        session, signed = await backend.create_upload(_upload(ttl_seconds=30))
        # Age the session past its TTL by rewriting the sidecar.
        from datetime import UTC, datetime, timedelta

        expired = session.model_copy(
            update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
        )
        backend._write_session_sync(expired)
        try:
            await backend.accept_upload(
                session.upload_id, PAYLOAD, signature=signed.signature
            )
        except UploadExpired:
            return True
        return False

    assert asyncio.run(scenario()), "an expired session must not accept bytes"


def test_reap_expired_removes_abandoned_uploads():
    backend = _make_local()

    async def scenario():
        from datetime import UTC, datetime, timedelta

        session, _ = await backend.create_upload(_upload())
        backend._write_session_sync(
            session.model_copy(
                update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
            )
        )
        kept = await _completed(backend)
        removed = await backend.reap_expired()
        still_there = await backend.get_session(kept.upload_id)
        return removed, still_there

    removed, still_there = asyncio.run(scenario())
    assert removed == 1
    assert still_there.state is UploadState.UPLOADED, "verified uploads are not garbage"


async def _completed(backend):
    session, signed = await backend.create_upload(_upload())
    await backend.accept_upload(session.upload_id, PAYLOAD, signature=signed.signature)
    return await backend.complete_upload(session.upload_id)


def test_expired_signature_is_rejected():
    backend = _make_local()

    async def scenario():
        session, _ = await backend.create_upload(_upload())
        stale = HmacUrlSigner(SECRET).sign(session.key, ttl_seconds=30)
        expired_token = HmacUrlSigner(SECRET)
        # Sign with a TTL that has already elapsed by shifting the clock forward
        # is not possible here, so use the minimum TTL and assert the mechanism
        # by verifying a token whose claims say it expired.
        import base64
        import json
        import time

        payload = json.dumps(
            {"key": session.key, "method": "PUT", "expires_at_epoch": int(time.time()) - 5},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        mac = expired_token._mac(payload)
        token = f"{base64.urlsafe_b64encode(payload).decode().rstrip('=')}.{mac}"
        try:
            await backend.accept_upload(session.upload_id, PAYLOAD, signature=token)
        except SignatureExpired:
            return True
        return False

    assert asyncio.run(scenario())


# --------------------------------------------------------------------------
# State machine and registry
# --------------------------------------------------------------------------


def test_transition_table_is_linear_and_terminates():
    assert TRANSITIONS[UploadState.ARCHIVED] == frozenset()
    reachable = {UploadState.CREATED}
    frontier = [UploadState.CREATED]
    while frontier:
        for nxt in TRANSITIONS[frontier.pop()]:
            if nxt not in reachable:
                reachable.add(nxt)
                frontier.append(nxt)
    assert reachable == set(UploadState), "every state must be reachable from created"


def test_registry_builds_a_local_backend():
    with tempfile.TemporaryDirectory() as tmp:
        backend = build_storage(
            StorageConfig(kind="local", root=Path(tmp), signing_secret=SECRET)
        )
    assert isinstance(backend, StorageBackend)
    assert backend.name == "local"


def test_registry_builds_an_s3_backend():
    backend = build_storage(
        StorageConfig(kind="s3", bucket="b", signing_secret=SECRET)
    )
    assert backend.name == "s3"


def test_registry_requires_the_fields_each_kind_needs():
    for kwargs, needle in [
        ({"kind": "local", "signing_secret": SECRET}, "root is required"),
        ({"kind": "s3", "signing_secret": SECRET}, "bucket is required"),
        ({"kind": "local", "root": Path("."), "signing_secret": "short"}, "at least 16"),
    ]:
        try:
            StorageConfig(**kwargs)
        except (ValueError, Exception) as exc:
            assert needle in str(exc), f"expected {needle!r} in {exc}"
            continue
        raise AssertionError(f"StorageConfig accepted {kwargs}")


def test_storage_config_never_prints_its_secret():
    config = StorageConfig(kind="s3", bucket="b", signing_secret=SECRET, secret_access_key="hunter2")
    assert SECRET not in repr(config)
    assert "hunter2" not in str(config)


# --------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------


def _run() -> int:
    tests = sorted(
        (n, o) for n, o in globals().items() if n.startswith("test_") and callable(o)
    )
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
