"""The model version chain: one authority, and a cache that can see it.

Three things independently change what this platform will answer — weights,
calibration, and the quality gate — and all three must reach the cache key. The
chain:

    ModelSpec.version          weights identity
    ModelSpec.branch_weights   fusion coefficients
    CalibrationSpec            temperature and its provenance
    MediaCaps                  quality floors
            │
            └─> ModelSpec.effective_version      the ONE authority
                        │
                        ├─> ModelInfo.version                 (bundle)
                        ├─> results/by-content/{sha}/{ver}    (cache WRITE key)
                        ├─> Settings.model_version            (cache READ key)
                        └─> JobSpec.idempotency_key

The failure this suite exists to prevent is quiet by nature. When the read key
and the write key disagree nothing raises: the cache simply never hits, every
upload re-uploads, every job re-runs, and the only signal is the bill. When
calibration is missing from the key, it is worse than quiet — the cache hits and
serves probabilities computed under a temperature that has since been replaced.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import STUB_ENGINE_VERSION, Settings  # noqa: E402
from contracts.common import Timing  # noqa: E402
from contracts.enums import BranchId, MediaKind  # noqa: E402
from contracts.version import (  # noqa: E402
    compose_model_version,
    fingerprint,
    is_composed,
    parse_model_version,
)
from inference.models import CalibrationSpec, ModelSpec  # noqa: E402
from inference.postprocess import (  # noqa: E402
    BranchOutcome,
    QualityFacts,
    build_image_bundle,
)
from inference.preprocess import DEFAULT_CAPS, MediaCaps  # noqa: E402
from workers.engine_registry import engine_versions, verify_version_alignment  # noqa: E402
from workers.io import InMemoryResultSink  # noqa: E402

from contracts.job import JobSpec  # noqa: E402

BASE = ModelSpec(name="m", version="w-1", backend="stub", gate_fingerprint="g0")
TIMING = Timing(queued_ms=0, preprocess_ms=1, inference_ms=1, total_ms=2)
SHA = "c" * 64


def _bundle(spec: ModelSpec):
    return build_image_bundle(
        job=JobSpec(
            job_id=uuid4(),
            media_kind=MediaKind.IMAGE,
            storage_key="uploads/x",
            sha256=SHA,
            size_bytes=1024,
            declared_mime="image/jpeg",
            model_version=spec.effective_version,
            idempotency_key=f"{SHA}:{spec.effective_version}",
            submitted_at=datetime.now(UTC),
        ),
        model=spec,
        size_bytes=1024,
        width=1920,
        height=1080,
        quality=QualityFacts(width=1920, height=1080, face_px=400),
        outcomes={
            BranchId.A: BranchOutcome(logit=2.0),
            BranchId.B: BranchOutcome(logit=1.5),
            BranchId.P: BranchOutcome(logit=0.0),
        },
        timing=TIMING,
    )


def _cache_identity(spec: ModelSpec) -> str:
    """What the sink would key this model's results under."""
    return _bundle(spec).model.version


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def test_composed_version_round_trips():
    v = compose_model_version(weights="w-1", calibration="aaaa1111", gate="bbbb2222")
    parts = parse_model_version(v)
    assert parts.weights == "w-1"
    assert parts.calibration == "aaaa1111"
    assert parts.gate == "bbbb2222"
    assert parts.compose() == v


def test_composed_version_round_trips_without_a_gate():
    v = compose_model_version(weights="w-1", calibration="aaaa1111")
    parts = parse_model_version(v)
    assert parts.gate is None
    assert parts.compose() == v


def test_a_bare_version_is_rejected_rather_than_guessed_at():
    """An unparseable version means somebody wrote one by hand."""
    try:
        parse_model_version("1.0.0-dev")
    except ValueError as exc:
        assert "not a composed model version" in str(exc)
        return
    raise AssertionError("a bare version string parsed as composed")


def test_weights_identity_may_not_smuggle_a_marker():
    try:
        compose_model_version(weights="w+c.deadbeef", calibration="aaaa1111")
    except ValueError:
        return
    raise AssertionError("a weights string containing a marker was accepted")


def test_fingerprint_is_stable_across_processes():
    """No ``hash()``: PYTHONHASHSEED would give a new cache key per restart.

    Run under two different hash seeds, in fresh interpreters, because that is
    the only way to catch a fingerprint that depends on Python's randomised
    string hashing. In-process the two would agree with each other and with a
    broken implementation.
    """
    import os
    import subprocess

    root = str(Path(__file__).resolve().parent.parent)
    script = (
        f"import sys; sys.path.insert(0, r'{root}');"
        "from contracts.version import fingerprint;"
        "print(fingerprint({'t': 1.5, 'n': 200, 'name': 'cdf-v2'}))"
    )
    seen = set()
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        )
        assert out.returncode == 0, out.stderr
        seen.add(out.stdout.strip())
    assert len(seen) == 1, f"fingerprint varied with PYTHONHASHSEED: {seen}"
    assert seen.pop() == fingerprint({"t": 1.5, "n": 200, "name": "cdf-v2"})


def test_fingerprint_ignores_key_order():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_fingerprint_distinguishes_neighbouring_floats():
    """1.0 and 1.0000000000000002 are different calibrations."""
    assert fingerprint({"t": 1.0}) != fingerprint({"t": 1.0000000000000002})


# --------------------------------------------------------------------------
# Cache identity moves when the model moves
# --------------------------------------------------------------------------


def test_changing_calibration_changes_cache_identity():
    """The headline requirement, and the one the old chain failed.

    Calibration is stored outside the weights on purpose, which is exactly what
    made it invisible: recalibrating shipped a small JSON, the version string
    never moved, and the cache went on serving probabilities computed under the
    temperature that had just been replaced.
    """
    recalibrated = replace(
        BASE, calibration=CalibrationSpec(temperature=1.7, fitted_on="cdf-v2", n_samples=2000)
    )
    assert _cache_identity(BASE) != _cache_identity(recalibrated)
    assert parse_model_version(_cache_identity(BASE)).weights == parse_model_version(
        _cache_identity(recalibrated)
    ).weights, "only the calibration segment should have moved"


def test_two_temperatures_of_one_point_zero_are_still_distinguishable():
    """Uncalibrated is not the same model as 'fitted, and found to need T=1.0'."""
    uncalibrated = replace(BASE, calibration=CalibrationSpec())
    fitted = replace(
        BASE, calibration=CalibrationSpec(temperature=1.0, fitted_on="cdf-v2", n_samples=2000)
    )
    assert uncalibrated.calibration.temperature == fitted.calibration.temperature
    assert _cache_identity(uncalibrated) != _cache_identity(fitted)


def test_changing_weights_changes_cache_identity():
    assert _cache_identity(BASE) != _cache_identity(replace(BASE, version="w-2"))


def test_changing_a_single_branch_changes_cache_identity():
    """Retraining branch A alone must invalidate, and must be visible as such."""
    retrained = replace(BASE, branch_versions={BranchId.A: "a-2"})
    assert _cache_identity(BASE) != _cache_identity(retrained)
    bundle = _bundle(retrained)
    assert bundle.model.branch_versions["A"] == "a-2"
    assert bundle.model.branch_versions["B"] == "w-1"


def test_changing_fusion_coefficients_changes_cache_identity():
    """No retraining, no recalibration — and every fused logit moves."""
    reweighted = replace(
        BASE, branch_weights={BranchId.A: 0.5, BranchId.B: 0.5, BranchId.P: 0.04}
    )
    assert _cache_identity(BASE) != _cache_identity(reweighted)


def test_changing_the_quality_gate_changes_cache_identity():
    """Raising the face floor withdraws verdicts without touching a weight."""
    stricter = MediaCaps(min_face_px=128)
    assert stricter.gate_fingerprint != DEFAULT_CAPS.gate_fingerprint
    assert _cache_identity(BASE) != _cache_identity(
        replace(BASE, gate_fingerprint=stricter.gate_fingerprint)
    )


def test_unchanged_model_assets_preserve_cache_identity():
    """The other half of the requirement: no spurious invalidation.

    A version that changed on every process start would be just as broken as one
    that never changed — every deploy would dump the cache.
    """
    identical = ModelSpec(name="m", version="w-1", backend="stub", gate_fingerprint="g0")
    assert _cache_identity(BASE) == _cache_identity(identical)
    assert BASE.effective_version == identical.effective_version
    # And stable across repeated evaluation.
    assert len({BASE.effective_version for _ in range(5)}) == 1


def test_a_rebuilt_spec_with_the_same_assets_hits_the_cache():
    """End to end through the sink, which is where it actually matters."""

    async def scenario():
        sink = InMemoryResultSink()
        first = _bundle(BASE)
        await sink.put_if_absent(str(first.job_id), first)

        # A worker restarts; the spec is reconstructed from the same config.
        rebuilt = ModelSpec(name="m", version="w-1", backend="stub", gate_fingerprint="g0")
        hit = await sink.find_by_content(SHA, rebuilt.effective_version)
        assert hit is not None, "an identically configured model missed its own cache"
        assert hit.job_id == first.job_id

        # Recalibrate: the same content must now miss.
        recal = replace(BASE, calibration=CalibrationSpec(temperature=2.0, n_samples=500))
        assert await sink.find_by_content(SHA, recal.effective_version) is None, (
            "a recalibrated model served a result computed under the old temperature"
        )

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# The two sides of the cache key agree
# --------------------------------------------------------------------------


def test_configured_model_version_matches_the_engines():
    """The read key and the write key must be the same string.

    Before Batch 0 they were not: the API defaulted to ``1.0.0-dev`` while the
    engines emitted ``stub-0``, so the sink wrote its content pointer under one
    key and the upload short-circuit read another. Every lookup missed. The
    suite passed throughout, because every fixture pinned both sides to the same
    literal and the default configuration was never exercised — which is why
    this test uses ``Settings()`` with no overrides at all.
    """
    settings = Settings()
    versions = engine_versions()
    for name, version in versions.items():
        assert settings.model_version == version, (
            f"the {name} engine reports {version!r} but the API is configured "
            f"with {settings.model_version!r}; the result cache reads one and "
            f"writes the other. Update backend.config.STUB_ENGINE_VERSION."
        )


def test_alignment_check_accepts_the_shipped_configuration():
    verify_version_alignment(Settings().model_version)


def test_alignment_check_rejects_a_drifted_configuration():
    try:
        verify_version_alignment("1.0.0-dev")
    except RuntimeError as exc:
        assert "does not match the engines" in str(exc)
        assert "silently disables it" in str(exc)
        return
    raise AssertionError("a drifted model_version passed the alignment check")


def test_the_shipped_version_is_composed_not_bare():
    """A bare version cannot express a calibration change, so it cannot key one."""
    assert is_composed(STUB_ENGINE_VERSION)
    parts = parse_model_version(STUB_ENGINE_VERSION)
    assert parts.calibration and parts.gate


def test_the_stub_version_still_announces_itself_as_a_stub():
    """Nothing downstream may mistake a stub score for a measurement."""
    assert "stub" in STUB_ENGINE_VERSION
    for version in engine_versions().values():
        assert "stub" in version


def test_branch_versions_are_no_longer_three_copies_of_one_string():
    bundle = _bundle(replace(BASE, branch_versions={BranchId.A: "a-2", BranchId.B: "b-9"}))
    assert set(bundle.model.branch_versions) == {"A", "B", "P"}
    assert len(set(bundle.model.branch_versions.values())) == 3


def _run() -> int:
    tests = sorted((n, o) for n, o in globals().items() if n.startswith("test_") and callable(o))
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
