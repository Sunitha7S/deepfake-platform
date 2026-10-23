"""Rate limiting on ``POST /uploads`` and ``POST /analyze``.

``rate_limit_per_minute`` existed since S1 and nothing enforced it. These tests
hold the limiter to the decisions it was built on:

* **Client identification.** An anonymous client is its direct peer address.
  ``X-Forwarded-For``, ``Forwarded`` and ``X-Real-IP`` are never trusted — no
  proxy is configured, so they are whatever the client sends.
* **Boundaries.** Exactly ``limit`` requests per window pass; the next is a 429
  ``rate_limited`` envelope with ``Retry-After`` equal to ``retry_after_s``.
  Both write routes share one budget; reads are not limited.
* **Reset.** Fixed one-minute windows: a new window is a new budget, and
  ``Retry-After`` counts down to it.
* **Concurrency.** The SQLite counter admits exactly ``limit`` requests under
  threads, under separate OS processes, and across two app instances sharing
  one file. The in-memory counter is per process, by design.

Clocks are injected, so nothing here waits for a real minute.

    python tests/test_rate_limit.py
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from starlette.requests import Request  # noqa: E402

from backend.app import create_app  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.openapi import build_schema  # noqa: E402
from backend.ratelimit import (  # noqa: E402
    WINDOW_S,
    InMemoryRateLimiter,
    SQLiteRateLimiter,
    build_rate_limiter,
    client_key,
)
from contracts.enums import ErrorCode  # noqa: E402
from tests.test_backend import IMAGE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
T0 = 1_800_000_000.0  # a window boundary: 1.8e9 is a multiple of 60
assert T0 % WINDOW_S == 0


class Clock:
    def __init__(self, at: float = T0) -> None:
        self.now = at

    def __call__(self) -> float:
        return self.now


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="dfp-ratelimit-"))


def _settings(**over) -> Settings:
    return Settings(log_json=False, log_level="CRITICAL", storage_root=_tmp() / "storage", **over)


@contextmanager
def api(limiter=None, *, peer: str = "203.0.113.7", **over):
    """An app with ``limiter`` and a client connecting from ``peer``."""
    app = create_app(_settings(**over), rate_limiter=limiter)
    with TestClient(app, client=(peer, 50000)) as client:
        yield client, app


def _body() -> dict:
    return {"sha256": hashlib.sha256(IMAGE).hexdigest(), "size_bytes": len(IMAGE),
            "content_type": "image/jpeg", "media_kind": "image"}


def negotiate(client, **headers):
    return client.post("/uploads", json=_body(), headers=headers)


def _request(peer: tuple[str, int] | None, headers: dict[str, str] | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "POST", "path": "/", "headers": raw,
                    "client": peer, "query_string": b""})


def _assert_limited(r) -> None:
    assert r.status_code == 429, r.text
    err = r.json()["error"]
    assert err["code"] == ErrorCode.RATE_LIMITED.value
    assert 1 <= err["retry_after_s"] <= WINDOW_S
    assert r.headers["Retry-After"] == str(err["retry_after_s"])


# --------------------------------------------------------------------------
# Client identification
# --------------------------------------------------------------------------


def test_an_anonymous_client_is_its_direct_peer_address():
    assert client_key(_request(("198.51.100.4", 1)), "anonymous:anonymous", anonymous=True) == (
        "ip:198.51.100.4"
    )


def test_forwarding_headers_never_change_the_key():
    spoofed = {
        "X-Forwarded-For": "1.1.1.1, 2.2.2.2",
        "Forwarded": "for=3.3.3.3;proto=https",
        "X-Real-IP": "4.4.4.4",
        "X-Client-IP": "5.5.5.5",
    }
    key = client_key(_request(("198.51.100.4", 1), spoofed), "anonymous:anonymous", anonymous=True)
    assert key == "ip:198.51.100.4"


def test_an_identified_principal_keys_on_itself_not_its_address():
    assert client_key(_request(("198.51.100.4", 1)), "user:u_42", anonymous=False) == "user:u_42"


def test_a_request_with_no_reported_peer_shares_the_unknown_budget():
    assert client_key(_request(None), "anonymous:anonymous", anonymous=True) == "ip:unknown"


def test_different_peers_get_separate_budgets():
    limiter = InMemoryRateLimiter(2, clock=Clock())
    app = create_app(_settings(), rate_limiter=limiter)
    with TestClient(app, client=("10.0.0.1", 1)) as a, TestClient(app, client=("10.0.0.2", 1)) as b:
        assert [negotiate(a).status_code for _ in range(2)] == [200, 200]
        _assert_limited(negotiate(a))
        assert negotiate(b).status_code == 200, "a's spent budget was charged to b"


def test_rotating_spoofed_forwarding_headers_does_not_buy_a_fresh_budget():
    """The attack the decision rules out: a new X-Forwarded-For per request."""
    with api(InMemoryRateLimiter(3, clock=Clock())) as (c, _):
        codes = [
            negotiate(c, **{"X-Forwarded-For": f"9.9.9.{i}", "X-Real-IP": f"8.8.8.{i}",
                            "Forwarded": f"for=7.7.7.{i}"}).status_code
            for i in range(5)
        ]
    assert codes == [200, 200, 200, 429, 429], codes


def test_a_spoofed_header_naming_another_client_does_not_spend_its_budget():
    limiter = InMemoryRateLimiter(2, clock=Clock())
    app = create_app(_settings(), rate_limiter=limiter)
    with TestClient(app, client=("10.0.0.1", 1)) as attacker, \
            TestClient(app, client=("10.0.0.2", 1)) as victim:
        for _ in range(4):
            negotiate(attacker, **{"X-Forwarded-For": "10.0.0.2"})
        assert [negotiate(victim).status_code for _ in range(2)] == [200, 200]


# --------------------------------------------------------------------------
# Boundaries
# --------------------------------------------------------------------------


def test_exactly_the_limit_passes_and_the_next_request_is_a_429_envelope():
    with api(InMemoryRateLimiter(3, clock=Clock(T0 + 15))) as (c, _):
        codes = [negotiate(c).status_code for _ in range(3)]
        limited = negotiate(c)
    assert codes == [200, 200, 200]
    _assert_limited(limited)
    assert limited.json()["error"]["retry_after_s"] == 45
    assert limited.headers["Retry-After"] == "45"


def test_the_configured_limit_is_the_one_enforced():
    with api(rate_limit_per_minute=2) as (c, app):
        assert app.state.rate_limiter.limit == 2
        assert [negotiate(c).status_code for _ in range(3)] == [200, 200, 429]


def test_uploads_and_analyze_share_one_budget():
    with api(InMemoryRateLimiter(2, clock=Clock())) as (c, _):
        negotiated = negotiate(c).json()
        assert c.put(negotiated["upload"]["url"], content=IMAGE).status_code == 204
        assert c.post("/analyze", json={"upload_id": negotiated["upload_id"]}).status_code == 202
        _assert_limited(c.post("/analyze", json={"upload_id": negotiated["upload_id"]}))
        _assert_limited(negotiate(c))


def test_reads_and_the_byte_upload_are_not_limited():
    with api(InMemoryRateLimiter(1, clock=Clock())) as (c, _):
        negotiated = negotiate(c).json()
        _assert_limited(negotiate(c))
        assert c.put(negotiated["upload"]["url"], content=IMAGE).status_code == 204
        unknown = "99999999-9999-9999-9999-999999999999"
        for _ in range(5):
            assert c.get(f"/jobs/{unknown}").status_code == 404
            assert c.get(f"/events/{unknown}").status_code == 404
            assert c.get(f"/results/{unknown}").status_code == 404
            assert c.get("/health").status_code in (200, 503)


def test_a_malformed_request_still_counts_and_is_refused_before_validation():
    with api(InMemoryRateLimiter(2, clock=Clock())) as (c, _):
        assert c.post("/uploads", json={"nonsense": True}).status_code == 400
        assert negotiate(c).status_code == 200
        _assert_limited(c.post("/uploads", json={"nonsense": True}))


def test_the_limiter_is_off_the_path_for_other_routes_entirely():
    """A read must not even touch the counter: a broken limiter cannot take
    polling down with it."""

    class Broken:
        name, limit = "broken", 1

        async def hit(self, key):
            raise RuntimeError("limiter store unreachable")

    app = create_app(_settings(), rate_limiter=Broken())
    # As a real client sees it: the server's 500, not the exception re-raised.
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.get("/jobs/99999999-9999-9999-9999-999999999999").status_code == 404
        r = negotiate(c)
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error", (
        "a limiter outage must fail loudly, not silently admit"
    )


# --------------------------------------------------------------------------
# Reset
# --------------------------------------------------------------------------


def _reset_scenario(make):
    clock = Clock(T0 + 50)
    limiter = make(clock)

    async def go():
        assert [(await limiter.hit("k")).allowed for _ in range(3)] == [True, True, False]
        refused = await limiter.hit("k")
        assert refused.retry_after_s == 10
        clock.now = T0 + 59.5
        assert (await limiter.hit("k")).retry_after_s == 1, "never tell a client to wait 0 s"
        clock.now = T0 + WINDOW_S  # the boundary itself opens the new window
        fresh = await limiter.hit("k")
        assert fresh.allowed and fresh.count == 1 and fresh.retry_after_s == WINDOW_S
        assert (await limiter.hit("k")).allowed
        assert not (await limiter.hit("k")).allowed

    asyncio.run(go())


def test_a_new_window_is_a_new_budget_in_memory():
    _reset_scenario(lambda clock: InMemoryRateLimiter(2, clock=clock))


def test_a_new_window_is_a_new_budget_in_sqlite():
    path = _tmp() / "rl.sqlite3"
    _reset_scenario(lambda clock: SQLiteRateLimiter(path, 2, clock=clock))


def test_waiting_retry_after_seconds_always_succeeds_over_http():
    clock = Clock(T0 + 23.4)
    with api(InMemoryRateLimiter(1, clock=clock)) as (c, _):
        negotiate(c)
        wait = int(negotiate(c).headers["Retry-After"])
        clock.now += wait
        assert negotiate(c).status_code == 200


def test_spent_windows_are_pruned_from_the_sqlite_file():
    path = _tmp() / "rl.sqlite3"
    clock = Clock()
    limiter = SQLiteRateLimiter(path, 5, clock=clock)

    async def go():
        for minute in range(4):
            clock.now = T0 + minute * WINDOW_S
            for key in ("a", "b", "c"):
                await limiter.hit(key)

    asyncio.run(go())
    with sqlite3.connect(path) as conn:
        windows = {w for (w,) in conn.execute("SELECT window_start FROM rate_hits")}
    assert windows == {int(T0) + 3 * WINDOW_S}, windows


# --------------------------------------------------------------------------
# Concurrency and sharing
# --------------------------------------------------------------------------


def test_concurrent_threads_admit_exactly_the_limit():
    path = _tmp() / "rl.sqlite3"
    limiter = SQLiteRateLimiter(path, 20, clock=Clock())
    results, barrier = [], threading.Barrier(16)
    lock = threading.Lock()

    def worker():
        barrier.wait()
        for _ in range(5):
            d = limiter._hit("shared")
            with lock:
                results.append(d)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 80
    assert sum(d.allowed for d in results) == 20
    assert sorted(d.count for d in results) == list(range(1, 81)), "a count was lost or repeated"


_CHILD = r"""
import sys
sys.path.insert(0, sys.argv[1])
from backend.ratelimit import SQLiteRateLimiter
limiter = SQLiteRateLimiter(sys.argv[2], 30, clock=lambda: float(sys.argv[3]))
print(sum(limiter._hit("shared").allowed for _ in range(25)))
"""


def test_separate_processes_sharing_one_file_admit_exactly_the_limit():
    """What the SQLite backend is for: several API processes, one budget."""
    path = _tmp() / "rl.sqlite3"
    procs = [
        subprocess.Popen([sys.executable, "-c", _CHILD, str(ROOT), str(path), str(T0)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(4)
    ]
    outs = [p.communicate(timeout=120) for p in procs]
    for p, (_, err) in zip(procs, outs, strict=True):
        assert p.returncode == 0, err
    admitted = [int(out.strip()) for out, _ in outs]
    assert sum(admitted) == 30, f"4 processes x 25 hits, limit 30: admitted {admitted}"
    with sqlite3.connect(path) as conn:
        (count,) = conn.execute("SELECT count FROM rate_hits").fetchone()
    assert count == 100, "every hit was counted exactly once"


def test_two_app_instances_on_one_file_share_a_budget():
    path = _tmp() / "rl.sqlite3"
    clock = Clock()
    with api(SQLiteRateLimiter(path, 3, clock=clock)) as (one, _), \
            api(SQLiteRateLimiter(path, 3, clock=clock)) as (two, _):
        codes = [negotiate(c).status_code for c in (one, two, one, two)]
    assert codes == [200, 200, 200, 429], codes


def test_the_in_memory_counter_is_per_process_as_documented():
    clock = Clock()
    with api(InMemoryRateLimiter(1, clock=clock)) as (one, _), \
            api(InMemoryRateLimiter(1, clock=clock)) as (two, _):
        assert negotiate(one).status_code == 200
        assert negotiate(two).status_code == 200, "memory counters are not shared"


# --------------------------------------------------------------------------
# Configuration and contract
# --------------------------------------------------------------------------


def test_settings_select_the_limiter():
    path = _tmp() / "rl.sqlite3"
    app = create_app(_settings(rate_limit_kind="sqlite", rate_limit_path=path,
                               rate_limit_per_minute=7))
    limiter = app.state.rate_limiter
    assert isinstance(limiter, SQLiteRateLimiter) and limiter.limit == 7
    assert limiter.path == path.absolute()
    assert isinstance(create_app(_settings()).state.rate_limiter, InMemoryRateLimiter)
    try:
        build_rate_limiter("sqlite", limit=1, path=None)
    except ValueError as exc:
        assert "rate_limit_path" in str(exc)
    else:
        raise AssertionError("sqlite limiter built without a path")


def test_production_refuses_a_per_process_counter():
    secret = "a-real-secret-for-this-test-only"
    try:
        Settings(environment="prod", signing_secret=secret, queue_kind="sqlite")
    except ValueError as exc:
        assert "rate_limit_kind='memory'" in str(exc)
    else:
        raise AssertionError("production accepted a limit multiplied by its process count")
    Settings(environment="prod", signing_secret=secret, queue_kind="sqlite",
             rate_limit_kind="sqlite")


def test_429_and_retry_after_are_documented_on_exactly_the_limited_routes():
    schema = build_schema()
    component = schema["components"]["responses"]["RateLimited"]
    header = component["headers"]["Retry-After"]
    assert header["schema"]["type"] == "integer"
    assert header["example"] == 60
    example = component["content"]["application/json"]["examples"]["rate_limited"]["value"]
    assert example["error"]["retry_after_s"] == 60
    limited = sorted(
        f"{method.upper()} {path}"
        for path, methods in schema["paths"].items()
        for method, op in methods.items()
        if "429" in op.get("responses", {})
    )
    assert limited == ["POST /analyze", "POST /uploads"], limited


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
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:400]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
