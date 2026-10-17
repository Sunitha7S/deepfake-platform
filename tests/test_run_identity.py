"""R12 — a cache keyed on model identity returns another decoder's results.

THE DEFECT, AS A SENTENCE
-------------------------
The result cache is keyed on ``(sha256, model_version)``. Until Batch 9
``model_version`` was ``effective_model_version`` — weights, calibration, gate —
and Batch 8 measured that the *decoder* changes ``p_fake`` enough to flip 1.3%
of decisions while leaving every one of those three byte-identical.

So the cache could store a bundle produced under FFMPEG and serve it to a
deployment running MSMF, and nothing anywhere would record that it had. This
file proves it, against the real cache implementation, using a named double for
the old key.

WHY A DOUBLE RATHER THAN A GIT CHECKOUT
---------------------------------------
``LegacyModelIdentityKey`` is not a description of the old behaviour; it is the
old key function, three lines, kept next to the new one. Every claim below is a
pair: the double does the wrong thing and the shipped code does not, measured
against the same sink, the same bundles and the same asserted facts. A test that
only exercises the fix cannot tell you the fix was needed.

WHAT IS AND IS NOT CLAIMED
--------------------------
Keying the cache on the decode environment stops a *cross-environment* hit. It
does not make two decoders agree, and it does not make a benchmark comparison
environment-aware — ``BenchmarkRun`` carries no decode field, and adding one is
a schema change this batch's brief forbids. Both limits are asserted here rather
than left to the report, because an untested limit drifts into an assumed
guarantee.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.common import ModelInfo, Timing  # noqa: E402
from contracts.enums import BranchId, MediaKind  # noqa: E402
from contracts.job import JobSpec  # noqa: E402
from contracts.version import (  # noqa: E402
    DECODE_MARKER,
    compose_model_version,
    compose_runtime_version,
    decode_identity_of,
    is_environment_pinned,
    model_identity_of,
    parse_model_version,
    parse_runtime_version,
)
from inference.decode_env import decode_environment, decode_identity  # noqa: E402
from inference.models import ModelSpec  # noqa: E402
from inference.postprocess import (  # noqa: E402
    BranchOutcome,
    QualityFacts,
    build_image_bundle,
)
from workers.io import InMemoryResultSink  # noqa: E402

TIMING = Timing(queued_ms=0, preprocess_ms=1, inference_ms=1, total_ms=2)
SHA_A = "a" * 64
SHA_B = "b" * 64

FFMPEG_DECODE = "c20a064a"
MSMF_DECODE = "a90156c7"
"""Two decode identities from Batch 8's measurement, used verbatim.

Not invented: these are the digests the study produced for FFMPEG and MSMF on
this machine, so the fixture and the finding are the same case.
"""


# --------------------------------------------------------------------------
# The named doubles
# --------------------------------------------------------------------------


class LegacyModelIdentityKey:
    """The pre-Batch-9 cache key: content plus **model** identity.

    Verbatim in the only respect that matters — it derives the key from a
    version string that carries no decode segment, so two decoders share a
    namespace.
    """

    name = "model identity only (pre-Batch-9)"

    @staticmethod
    def key(sha256: str, version: str) -> tuple[str, str]:
        return (sha256, model_identity_of(version))


class RuntimeIdentityKey:
    """The shipped key: content plus **runtime** identity."""

    name = "runtime identity (Batch 9)"

    @staticmethod
    def key(sha256: str, version: str) -> tuple[str, str]:
        return (sha256, version)


def _job(sha256: str = SHA_A, version: str = "v") -> JobSpec:
    return JobSpec(
        job_id=UUID(int=int(sha256[0], 16) + 1),
        media_kind=MediaKind.IMAGE,
        storage_key="uploads/x",
        sha256=sha256,
        size_bytes=1024,
        declared_mime="image/jpeg",
        model_version=version,
        idempotency_key=f"{sha256}:{version}",
        submitted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _bundle(*, runtime_version: str, sha256: str = SHA_A, logit: float = 4.0):
    """A real bundle, published under ``runtime_version``."""
    return build_image_bundle(
        job=_job(sha256, runtime_version),
        model=ModelSpec(name="t", version="stub-0", backend="stub"),
        size_bytes=1024,
        width=1920,
        height=1080,
        quality=QualityFacts(width=1920, height=1080, face_px=400),
        outcomes={
            BranchId.A: BranchOutcome(logit=logit),
            BranchId.B: BranchOutcome(logit=logit - 0.5),
            BranchId.P: BranchOutcome(logit=0.0),
        },
        timing=TIMING,
        runtime_version=runtime_version,
    )


MODEL = compose_model_version(weights="stub-0-w7f45fada", calibration="739c3fec", gate="ae232cba")
UNDER_FFMPEG = compose_runtime_version(model=MODEL, decode=FFMPEG_DECODE)
UNDER_MSMF = compose_runtime_version(model=MODEL, decode=MSMF_DECODE)


# --------------------------------------------------------------------------
# 1. The proof: can a cache entry cross decode environments?
# --------------------------------------------------------------------------


def test_the_legacy_key_serves_one_decoders_result_to_another():
    """Requirement 4, proved rather than argued.

    Two bundles, same bytes, same model, different decoders and different
    scores. Under the legacy key they collide, so the second deployment reads
    the first's answer.
    """
    ffmpeg = _bundle(runtime_version=UNDER_FFMPEG, logit=4.0)
    msmf = _bundle(runtime_version=UNDER_MSMF, logit=2.0)
    assert ffmpeg.p_fake != msmf.p_fake, "the fixture must actually disagree"

    store: dict[tuple[str, str], object] = {}
    store[LegacyModelIdentityKey.key(SHA_A, UNDER_FFMPEG)] = ffmpeg
    hit = store.get(LegacyModelIdentityKey.key(SHA_A, UNDER_MSMF))

    assert hit is ffmpeg, (
        "the legacy double no longer collides; it can no longer demonstrate R12"
    )
    assert hit.p_fake != msmf.p_fake, (
        f"an MSMF deployment was served p_fake={hit.p_fake}, which FFMPEG "
        f"produced, when its own decoder gives {msmf.p_fake}"
    )


def test_the_runtime_key_does_not():
    ffmpeg = _bundle(runtime_version=UNDER_FFMPEG, logit=4.0)
    store: dict[tuple[str, str], object] = {
        RuntimeIdentityKey.key(SHA_A, UNDER_FFMPEG): ffmpeg
    }
    assert store.get(RuntimeIdentityKey.key(SHA_A, UNDER_MSMF)) is None
    assert store.get(RuntimeIdentityKey.key(SHA_A, UNDER_FFMPEG)) is ffmpeg


def test_the_real_sink_isolates_the_two_environments():
    """The same proof against ``InMemoryResultSink``, not a dict.

    The sink keys ``_by_content`` on ``bundle.model.version``, so this exercises
    the actual production key derivation rather than a restatement of it.
    """

    async def scenario():
        sink = InMemoryResultSink()
        ffmpeg = _bundle(runtime_version=UNDER_FFMPEG, logit=4.0)
        await sink.put_if_absent(str(ffmpeg.job_id), ffmpeg)

        same = await sink.find_by_content(SHA_A, UNDER_FFMPEG)
        cross = await sink.find_by_content(SHA_A, UNDER_MSMF)
        legacy = await sink.find_by_content(SHA_A, MODEL)
        return same, cross, legacy

    same, cross, legacy = asyncio.run(scenario())
    assert same is not None, "a same-environment lookup must still hit"
    assert cross is None, (
        "a bundle decoded under FFMPEG was served to an MSMF lookup"
    )
    assert legacy is None, (
        "a lookup by bare model identity hit a runtime-versioned entry; the two "
        "namespaces must not overlap"
    )


def test_a_recalibration_still_misses_as_it_did_before():
    """Batch 9 must not have bought environment isolation by losing the old one."""

    async def scenario():
        sink = InMemoryResultSink()
        bundle = _bundle(runtime_version=UNDER_FFMPEG)
        await sink.put_if_absent(str(bundle.job_id), bundle)
        recalibrated = compose_runtime_version(
            model=compose_model_version(
                weights="stub-0-w7f45fada", calibration="deadbeef", gate="ae232cba"
            ),
            decode=FFMPEG_DECODE,
        )
        return await sink.find_by_content(SHA_A, recalibrated)

    assert asyncio.run(scenario()) is None


def test_different_content_still_misses():
    async def scenario():
        sink = InMemoryResultSink()
        bundle = _bundle(runtime_version=UNDER_FFMPEG, sha256=SHA_A)
        await sink.put_if_absent(str(bundle.job_id), bundle)
        return await sink.find_by_content(SHA_B, UNDER_FFMPEG)

    assert asyncio.run(scenario()) is None


# --------------------------------------------------------------------------
# 2. Model identity and runtime identity are distinguishable
# --------------------------------------------------------------------------


def test_a_decoder_change_moves_the_runtime_version_and_not_the_model_version():
    """Requirement 5's fourth criterion, as a property.

    An operator looking at two versions must be able to say *which* changed. The
    segmented format is what makes that a substring comparison rather than an
    investigation.
    """
    assert UNDER_FFMPEG != UNDER_MSMF
    assert model_identity_of(UNDER_FFMPEG) == model_identity_of(UNDER_MSMF) == MODEL
    assert decode_identity_of(UNDER_FFMPEG) != decode_identity_of(UNDER_MSMF)
    # And the model half really is a prefix, so "strip the environment" is one
    # partition rather than a reassembly.
    assert UNDER_FFMPEG.startswith(MODEL + DECODE_MARKER)


def test_a_model_change_moves_both():
    other_model = compose_model_version(
        weights="stub-1-wffffffff", calibration="739c3fec", gate="ae232cba"
    )
    other = compose_runtime_version(model=other_model, decode=FFMPEG_DECODE)
    assert model_identity_of(other) != model_identity_of(UNDER_FFMPEG)
    assert other != UNDER_FFMPEG
    assert decode_identity_of(other) == decode_identity_of(UNDER_FFMPEG)


def test_the_model_components_still_parse_out_of_a_runtime_version():
    parts = parse_model_version(UNDER_FFMPEG)
    assert parts.weights == "stub-0-w7f45fada"
    assert parts.calibration == "739c3fec"
    assert parts.gate == "ae232cba"
    assert parts.compose() == MODEL


def test_model_identity_of_is_idempotent():
    assert model_identity_of(MODEL) == MODEL
    assert model_identity_of(model_identity_of(UNDER_FFMPEG)) == MODEL


# --------------------------------------------------------------------------
# 3. An unpinned version is representable and distinguishable
# --------------------------------------------------------------------------


def test_a_version_with_no_decode_segment_is_readable_and_says_so():
    """Migration: every version written before Batch 9 still parses.

    ``decode=None`` is not an error. It means "this deployment makes no claim
    about its decode environment", which is exactly what a pre-Batch-9 string
    means, and is distinguishable from a claim.
    """
    parts = parse_runtime_version(MODEL)
    assert parts.model == MODEL
    assert parts.decode is None
    assert not parts.is_environment_pinned
    assert not is_environment_pinned(MODEL)
    assert is_environment_pinned(UNDER_FFMPEG)


def test_composing_without_a_decode_returns_the_model_identity_unchanged():
    """No sentinel. A placeholder would make 'unpinned' and 'pinned to nothing'
    the same string, and the whole scheme rests on telling them apart."""
    assert compose_runtime_version(model=MODEL, decode=None) == MODEL


def test_double_composition_is_refused():
    try:
        compose_runtime_version(model=UNDER_FFMPEG, decode=MSMF_DECODE)
    except ValueError as exc:
        assert "already carries a decode segment" in str(exc)
        return
    raise AssertionError("a version with two decode segments was composed")


def test_a_weights_version_may_not_contain_the_decode_marker():
    try:
        compose_model_version(
            weights=f"stub{DECODE_MARKER}0", calibration="abcd1234"
        )
    except ValueError as exc:
        assert DECODE_MARKER in str(exc)
        return
    raise AssertionError("a weights version carrying the decode marker was accepted")


def test_a_trailing_marker_with_no_fingerprint_is_refused():
    try:
        parse_runtime_version(MODEL + DECODE_MARKER)
    except ValueError as exc:
        assert "no fingerprint" in str(exc)
        return
    raise AssertionError("a dangling decode marker parsed")


# --------------------------------------------------------------------------
# 4. The live pipeline publishes what the cache reads
# --------------------------------------------------------------------------


def test_the_pipelines_publish_a_runtime_version():
    from inference import ImagePipeline, VideoPipeline

    for pipeline in (ImagePipeline(model_name="stub-image"), VideoPipeline(model_name="stub-video")):
        version = pipeline.model_version
        assert is_environment_pinned(version), f"{version} carries no decode segment"
        assert model_identity_of(version) == pipeline.model_identity
        assert decode_identity_of(version) == decode_identity()


def test_the_bundle_carries_the_same_string_the_cache_keys_on():
    """The write key and the read key must be the same string.

    Batch 0's defect was that they were not: the API wrote under one version and
    read under another, so the cache never hit at all. This asserts the property
    from the bundle's side.
    """
    from inference import ImagePipeline

    pipeline = ImagePipeline(model_name="stub-image")
    bundle = _bundle(runtime_version=pipeline.model_version)
    assert bundle.model.version == pipeline.model_version
    assert is_environment_pinned(bundle.model.version)


def test_the_configured_literal_matches_the_engines():
    """``verify_version_alignment``, exercised on the value that actually ships.

    This is the mechanism Batch 8 wrongly concluded could not carry a decode
    fingerprint to the API: the API never computes the version, it reads a
    configured literal that this check holds to what the engines report.
    """
    from backend.config import STUB_ENGINE_VERSION
    from workers.engine_registry import verify_version_alignment

    assert is_environment_pinned(STUB_ENGINE_VERSION)
    verify_version_alignment(STUB_ENGINE_VERSION)


def test_alignment_fails_when_the_configured_decoder_is_stale():
    """The failure mode the literal is supposed to produce on a foreign machine.

    A worker whose decode environment differs from the configured one must fail
    at startup rather than share a cache namespace with it.
    """
    from backend.config import STUB_ENGINE_VERSION
    from workers.engine_registry import verify_version_alignment

    stale = compose_runtime_version(
        model=model_identity_of(STUB_ENGINE_VERSION), decode="00000000"
    )
    try:
        verify_version_alignment(stale)
    except RuntimeError as exc:
        assert "does not match the engines" in str(exc)
        return
    raise AssertionError("a stale decode identity passed the alignment check")


# --------------------------------------------------------------------------
# 5. The limits, asserted so they cannot become assumed guarantees
# --------------------------------------------------------------------------


def test_the_cache_identity_ignores_cpu_fields():
    """Deliberate: Batch 8 measured CPU dispatch not to change a decoded pixel.

    Two workers differing only in AVX support must share a cache namespace, or a
    mixed fleet halves its hit rate for no correctness gain.
    """
    env = decode_environment()
    other = replace(env, cpu_baseline="SSE SSE2", cpu_dispatch="")
    assert other.identity == env.identity, "a CPU difference fragmented the cache"
    assert other.digest != env.digest, "the forensic digest must still see it"


def test_the_forensic_digest_is_strictly_finer_than_the_cache_identity():
    env = decode_environment()
    assert env.identity != env.digest
    payload = env.to_dict()
    assert payload["identity"] == env.identity
    assert payload["digest"] == env.digest
    for field in ("opencv_version", "video_backend", "ffmpeg_version",
                  "platform_system", "platform_machine", "cpu_baseline",
                  "cpu_dispatch", "backend_source", "python_version",
                  "image_codec_version", "image_codec_simd"):
        assert field in payload, f"{field} is not recoverable for forensics"


def test_the_backend_changes_the_cache_identity():
    env = decode_environment()
    other = replace(env, video_backend=env.video_backend + "-OTHER")
    assert other.identity != env.identity
    assert other.digest != env.digest


def test_benchmark_runs_carry_a_decode_identity():
    """**Batch 11 replaced this test, exactly as its own message asked.**

    Written in Batch 9, it asserted the *absence* of a decode field on
    ``BenchmarkRun`` and said: "R12's benchmark half may now be closable and this
    test should be replaced by one that gates on it." Batch 11 closed it and did
    so. The tripwire worked — it is kept here, inverted, rather than deleted,
    because the same field disappearing again would silently reopen R12.

    The gate behaviour it protects lives in ``tests/test_evaluation_identity.py``.
    """
    from evaluation.benchmark import RUN_SCHEMA_VERSION, BenchmarkRun

    fields = BenchmarkRun.__dataclass_fields__
    assert "decode_identity" in fields, (
        "BenchmarkRun lost its decode identity; R12's benchmark half is reopened "
        "and a cross-decoder comparison can pass again"
    )
    assert fields["decode_identity"].default is None, (
        "the default must be None — a run that does not know its decoder must "
        "say so rather than claim one"
    )
    assert RUN_SCHEMA_VERSION == "1.2.0"


def test_the_model_identity_still_excludes_the_decoder():
    """The other half of Batch 9's decision, and it did **not** change.

    A run records the *model* identity so that "did the model get worse" is not
    answered yes by a worker rebuild. Batch 11 added a separate field rather than
    widening this one; if the decoder ever leaks in here, every model-versus-model
    comparison silently becomes a decoder comparison too.
    """
    from evaluation.benchmark import BenchmarkRun

    fields = BenchmarkRun.__dataclass_fields__
    assert "effective_model_version" in fields
    assert model_identity_of(UNDER_FFMPEG) == MODEL
    assert DECODE_MARKER not in MODEL, (
        "the model identity has grown a decode segment; a benchmark comparison "
        "would now answer 'the model changed' for a worker rebuild"
    )


def test_the_evidence_bundle_schema_is_unchanged():
    """No schema change: the version string got longer, the contract did not."""
    from contracts.evidence import EvidenceBundle

    model_fields = set(ModelInfo.model_fields)
    assert model_fields == {"version", "branch_versions", "calibration_temp", "git_sha"}
    assert ModelInfo.model_fields["version"].annotation is str
    assert "model" in EvidenceBundle.model_fields


def test_a_bundle_built_without_a_runtime_version_is_unpinned_not_wrong():
    """The default path for callers outside a pipeline.

    It publishes the model identity, which makes no claim about the decoder —
    and ``is_environment_pinned`` reports exactly that, rather than the bundle
    quietly asserting an environment it never observed.
    """
    spec = ModelSpec(name="t", version="stub-0", backend="stub")
    bundle = build_image_bundle(
        job=_job(),
        model=spec,
        size_bytes=1024,
        width=640,
        height=480,
        quality=QualityFacts(width=640, height=480, face_px=200),
        outcomes={
            BranchId.A: BranchOutcome(logit=1.0),
            BranchId.B: BranchOutcome(logit=0.5),
            BranchId.P: BranchOutcome(logit=0.0),
        },
        timing=TIMING,
    )
    assert bundle.model.version == spec.effective_version
    assert not is_environment_pinned(bundle.model.version)


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
