"""Production refuses the Redis queue and S3 storage skeletons at startup.

``jobqueue/redis.py`` and ``storage/s3.py`` are skeletons: they construct
without error and raise ``NotImplementedError`` from their first operation.
Before this suite, ``DFP_ENVIRONMENT=prod`` accepted either. The API booted and
logged itself started; with Redis it even answered ``POST /uploads`` with 200,
and with S3 the first request raised. A misconfigured deployment looked healthy
until it was handed real work.

Layers:

* the settings guard: each skeleton is refused in production, with a message
  that names the setting that fixes it;
* the real entry points: ``create_app()`` and ``python -m backend.worker_main``,
  started as child processes from the environment alone, exit with that
  message instead of booting;
* the supported configuration (SQLite queue, local storage, SQLite rate
  limiter) still validates and serves ``/health``;
* the skeletons stay selectable outside production, where their own tests and
  their eventual implementation need to build them;
* the ``UNIMPLEMENTED_KINDS`` sets match which backends actually raise
  ``NotImplementedError``, so implementing one forces the guard to be lifted
  deliberately rather than leaving it to refuse a working backend.

    python tests/test_unimplemented_backends.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import get_args

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import create_app  # noqa: E402
from backend.config import Settings  # noqa: E402
from jobqueue import registry as queue_registry  # noqa: E402
from storage import registry as storage_registry  # noqa: E402
from storage.errors import StorageError  # noqa: E402

SECRET = "a-real-secret-for-this-test-only"
REDIS_URL = "redis://localhost:6379/0"

REDIS_REFUSAL = "queue_kind='redis' is not implemented yet"
S3_REFUSAL = "storage_kind='s3' is not implemented yet"


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="dfp-unimpl-"))


def _prod(tmp: Path, **overrides: object) -> Settings:
    """A production configuration that is valid apart from ``overrides``."""
    fields: dict[str, object] = dict(
        environment="prod",
        signing_secret=SECRET,
        queue_kind="sqlite",
        queue_path=tmp / "queue.sqlite3",
        rate_limit_kind="sqlite",
        rate_limit_path=tmp / "ratelimit.sqlite3",
        storage_root=tmp / "storage",
    )
    fields.update(overrides)
    return Settings(**fields)


def _refused(build, expected: str) -> str:
    try:
        build()
    except ValueError as exc:
        message = str(exc)
        assert expected in message, f"refused, but not for the right reason: {message}"
        return message
    raise AssertionError(f"production accepted a skeleton backend ({expected!r})")


# --------------------------------------------------------------------------
# The settings guard
# --------------------------------------------------------------------------


def test_production_refuses_the_redis_queue_skeleton():
    message = _refused(
        lambda: _prod(_tmp(), queue_kind="redis", redis_url=REDIS_URL), REDIS_REFUSAL
    )
    assert "DFP_QUEUE_KIND=sqlite" in message, "the error must name the fix"
    assert "NotImplementedError" in message, "the error must say why"


def test_production_refuses_the_s3_storage_skeleton():
    message = _refused(
        lambda: _prod(_tmp(), storage_kind="s3", storage_bucket="bucket"), S3_REFUSAL
    )
    assert "DFP_STORAGE_KIND=local" in message, "the error must name the fix"
    assert "NotImplementedError" in message, "the error must say why"


def test_s3_without_a_bucket_is_refused_as_unimplemented_not_as_incomplete():
    """Asking for a bucket would send an operator to fix the wrong thing."""
    message = _refused(lambda: _prod(_tmp(), storage_kind="s3"), S3_REFUSAL)
    assert "bucket" not in message


def test_both_skeletons_together_are_refused():
    _refused(
        lambda: _prod(
            _tmp(),
            storage_kind="s3",
            storage_bucket="bucket",
            queue_kind="redis",
            redis_url=REDIS_URL,
        ),
        "is not implemented yet",
    )


def test_the_existing_production_guards_still_fire_first_or_alongside():
    """The new guard must not displace the secret, memory-queue or limiter checks."""
    _refused(lambda: _prod(_tmp(), signing_secret="dev-only-insecure-xx"), "signing_secret")
    _refused(lambda: _prod(_tmp(), queue_kind="memory"), "queue_kind='memory'")
    _refused(lambda: _prod(_tmp(), rate_limit_kind="memory"), "rate_limit_kind='memory'")


# --------------------------------------------------------------------------
# Supported configurations
# --------------------------------------------------------------------------


def test_the_supported_production_configuration_validates_and_serves_health():
    tmp = _tmp()
    settings = _prod(tmp)
    assert (settings.queue_kind, settings.storage_kind, settings.rate_limit_kind) == (
        "sqlite",
        "local",
        "sqlite",
    )
    with TestClient(create_app(settings=settings)) as client:
        response = client.get("/health")
    assert response.status_code == 200, response.text


def test_the_skeletons_stay_selectable_outside_production():
    """Their own tests, and whoever implements them, still need to build them."""
    for environment in ("dev", "staging"):
        queue = Settings(environment=environment, queue_kind="redis", redis_url=REDIS_URL)
        assert queue.queue_config().kind == "redis"
        storage = Settings(environment=environment, storage_kind="s3", storage_bucket="b")
        assert storage.storage_config().kind == "s3"


# --------------------------------------------------------------------------
# The real entry points, started from the environment alone
# --------------------------------------------------------------------------


def _env(tmp: Path, **dfp: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("DFP_")}
    env.update(
        PYTHONPATH=str(ROOT),
        DFP_ENVIRONMENT="prod",
        DFP_SIGNING_SECRET=SECRET,
        DFP_QUEUE_KIND="sqlite",
        DFP_QUEUE_PATH=str(tmp / "queue.sqlite3"),
        DFP_RATE_LIMIT_KIND="sqlite",
        DFP_RATE_LIMIT_PATH=str(tmp / "ratelimit.sqlite3"),
        DFP_STORAGE_ROOT=str(tmp / "storage"),
    )
    env.update(dfp)
    return env


def _start(args: list[str], **dfp: str) -> subprocess.CompletedProcess[str]:
    # cwd is an empty directory so no stray .env file can change the outcome.
    tmp = _tmp()
    return subprocess.run(
        [sys.executable, *args],
        cwd=tmp,
        env=_env(tmp, **dfp),
        capture_output=True,
        text=True,
        timeout=120,
    )


SKELETON_ENVS = {
    REDIS_REFUSAL: dict(DFP_QUEUE_KIND="redis", DFP_REDIS_URL=REDIS_URL),
    S3_REFUSAL: dict(DFP_STORAGE_KIND="s3", DFP_STORAGE_BUCKET="bucket"),
}


def test_the_api_process_refuses_to_start_on_a_skeleton():
    for expected, dfp in SKELETON_ENVS.items():
        proc = _start(["-c", "from backend.app import create_app; create_app()"], **dfp)
        assert proc.returncode != 0, f"the API started on a skeleton: {dfp}"
        assert expected in proc.stderr, proc.stderr[-2000:]


def test_the_worker_process_refuses_to_start_on_a_skeleton():
    for expected, dfp in SKELETON_ENVS.items():
        proc = _start(["-m", "backend.worker_main"], **dfp)
        assert proc.returncode != 0, f"the worker started on a skeleton: {dfp}"
        assert expected in proc.stderr, proc.stderr[-2000:]


def test_the_api_process_starts_on_the_supported_configuration():
    """The control for the two tests above: same harness, nothing refused."""
    proc = _start(["-c", "from backend.app import create_app; create_app()"])
    assert proc.returncode == 0, proc.stderr[-2000:]


# --------------------------------------------------------------------------
# The guard's sets match the code
# --------------------------------------------------------------------------


def _queue_is_a_skeleton(kind: str) -> bool:
    tmp = _tmp()
    config = queue_registry.QueueConfig(
        kind=kind,
        url=REDIS_URL if kind == "redis" else None,
        path=tmp / "queue.sqlite3" if kind == "sqlite" else None,
    )
    queue = queue_registry.build_queue(config)
    try:
        asyncio.run(queue.enqueue(queue_registry.IMAGE_QUEUE, {"probe": True}))
    except NotImplementedError:
        return True
    return False


def _storage_is_a_skeleton(kind: str) -> bool:
    tmp = _tmp()
    config = storage_registry.StorageConfig(
        kind=kind,
        signing_secret=SECRET,
        root=tmp / "storage" if kind == "local" else None,
        bucket="bucket" if kind == "s3" else None,
    )
    storage = storage_registry.build_storage(config)
    try:
        asyncio.run(storage.get_object("probe/missing"))
    except NotImplementedError:
        return True
    except StorageError:
        return False  # a real backend: the object simply is not there
    return False


def test_the_unimplemented_queue_kinds_are_exactly_the_skeletons():
    kinds = get_args(queue_registry.BackendKind)
    skeletons = {k for k in kinds if _queue_is_a_skeleton(k)}
    assert skeletons == set(queue_registry.UNIMPLEMENTED_KINDS), (
        f"queue skeletons {sorted(skeletons)} != UNIMPLEMENTED_KINDS "
        f"{sorted(queue_registry.UNIMPLEMENTED_KINDS)}; update the set in the "
        "change that implements (or stubs) a backend"
    )


def test_the_unimplemented_storage_kinds_are_exactly_the_skeletons():
    kinds = get_args(storage_registry.BackendKind)
    skeletons = {k for k in kinds if _storage_is_a_skeleton(k)}
    assert skeletons == set(storage_registry.UNIMPLEMENTED_KINDS), (
        f"storage skeletons {sorted(skeletons)} != UNIMPLEMENTED_KINDS "
        f"{sorted(storage_registry.UNIMPLEMENTED_KINDS)}; update the set in the "
        "change that implements (or stubs) a backend"
    )


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
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:600]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
