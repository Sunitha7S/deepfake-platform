"""R12's benchmark and calibration halves — the gap, and the migration.

WHAT BATCH 9 LEFT OPEN
----------------------
Batch 9 put the decode environment into the *serving* runtime identity, so a
cached verdict cannot cross decoders. It deliberately left ``BenchmarkRun`` and
``CalibrationWitness`` alone, because closing those needed a schema bump its
brief forbade, and it left ``test_benchmark_runs_still_carry_no_decode_identity``
behind as a tripwire naming exactly this batch's work.

THE TWO HALVES ARE NOT EQUALLY BAD, AND THE TESTS SAY SO
--------------------------------------------------------
**Benchmark (the real defect).** ``regression_gate`` refused a comparison across
datasets and had no field for the decoder, so two runs under two decoders
compared cleanly. Measured on a synthetic model of Batch 8's levels shift: a
decoder change moves AUC by ~0.0% and selective FPR by 24.6% of the gate's
allowance — so it is invisible to the check the gate leans on and charged to the
model by the check that moves.

**Calibration (weaker, with one blind spot).** A decoder change moves the
logits, so it moves the fitted temperature, so it moves the witness digest —
detection mostly worked by accident. The hole is convergence: two fits landing on
the same values after ``quantize`` produce byte-identical witnesses for two
environments. ``test_two_environments_could_share_one_witness`` is that case.

THE MIGRATION IS THE INTERESTING PART
-------------------------------------
``compute_witness`` has always hashed the schema version, so bumping
``CALIBRATION_WITNESS_VERSION`` changes the digest of *unchanged fields* and
makes every stored witness raise on load. ``LegacySchemaPinnedDigest`` reproduces
the pre-batch function verbatim and the tests below prove both halves: that the
naive bump breaks a stored artifact, and that dispatching the digest on the
payload's own schema version keeps it readable **and** verifiable.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from contracts.common import DEFAULT_LOG_ODDS_MARGIN, ThresholdInfo  # noqa: E402
from contracts.version import compose_runtime_version, fingerprint, quantize  # noqa: E402
from evaluation.benchmark import (  # noqa: E402
    RUN_SCHEMA_VERSION,
    BenchmarkRun,
    StubPredictor,
    append_history,
    read_runs,
    regression_gate,
    run_benchmark,
)
from evaluation.calibration_protocol import (  # noqa: E402
    CALIBRATION_WITNESS_VERSION,
    SCHEMA_VERSIONS_WITH_DECODE,
    CalibrationWitness,
    verify_calibration_witness,
)
from evaluation.dataset import MediaItem  # noqa: E402
from evaluation.protocol import ProtocolViolation  # noqa: E402

TH = ThresholdInfo.at_margin(value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02)
TMP = Path(tempfile.mkdtemp(prefix="dfp-eval-identity-"))

# The two decode identities Batch 8 actually measured on this machine.
UNDER_FFMPEG = "c20a064a"
UNDER_MSMF = "a90156c7"

MODEL_ID = "stub-0-w7f45fada+c.739c3fec.g.ae232cba"
"""A *model* identity — no ``.d.`` segment. What a run records, by design.

Frozen pre-B22 model-only fixture (pre-B22 weights ``w7f45fada``, no ``.f.``
segment): it is **not** the shipped ``STUB_ENGINE_VERSION`` and does not track
it. Do not update it on identity moves (B22 weights / ``.f.``, B23 composite
``.f.``); these tests assert that evaluation identity uses the model-only form."""


# --------------------------------------------------------------------------
# Named doubles
# --------------------------------------------------------------------------


class LegacyDecodeBlindRun:
    """``run_benchmark`` as it stood before Batch 11: no decode field at all.

    Reproduced by stripping the field from a real run rather than by
    reimplementing the function, so the double cannot drift away from what the
    code actually did.
    """

    name = "decode-blind BenchmarkRun (pre-Batch-11)"

    @staticmethod
    def strip(run: BenchmarkRun) -> BenchmarkRun:
        payload = json.loads(run.to_json())
        payload.pop("decode_identity", None)
        payload["schema_version"] = "1.1.0"
        return BenchmarkRun.from_dict(payload)


class LegacySchemaPinnedDigest:
    """``compute_witness`` as it stood: the schema version hard-coded.

    Verbatim, including the ordering of the dict, because the digest is over a
    canonicalised form and any difference here would make the double prove
    something other than the thing it exists to prove.
    """

    name = "schema-pinned witness digest (pre-Batch-11)"

    @staticmethod
    def compute(*, schema: str, **f) -> str:
        return fingerprint(
            {
                "protocol": f["protocol_name"],
                "role": f["role"],
                "corpus_fingerprint": f["corpus_fingerprint"],
                "n_samples": f["n_samples"],
                "temperature": quantize(f["temperature"]),
                "nll_before": quantize(f["nll_before"]),
                "nll_after": quantize(f["nll_after"]),
                "threshold": None if f["threshold"] is None else quantize(f["threshold"]),
                "margin": None if f["margin"] is None else quantize(f["margin"]),
                "target_fpr": None
                if f["target_fpr"] is None
                else quantize(f["target_fpr"]),
                "selection_witness": f["selection_witness"],
                "split_lock_hash": f["split_lock_hash"],
                "schema": schema,
            }
        )


class DecoderSensitivePredictor:
    """A predictor whose scores depend on the decode environment.

    ``StubPredictor`` hashes the media id and never decodes anything, so it
    cannot exhibit the defect at all. A real model decodes the media; Batch 8
    measured the cross-decoder difference as a systematic levels shift. This
    applies one, which is what makes the AUC/FPR asymmetry below reproducible.
    """

    def __init__(self, decode_identity: str, *, skill: float = 0.30, bias: float = 0.0,
                 version: str = "stub-0") -> None:
        self.name = "decoder-sensitive"
        self.version = version
        self.decode_identity = decode_identity
        self.bias = bias
        self._inner = StubPredictor(version=version, skill=skill)

    def predict(self, items: list[MediaItem]) -> np.ndarray:
        return np.clip(self._inner.predict(items) + self.bias, 1e-9, 1.0 - 1e-9)


def _items(n: int = 2000) -> list[MediaItem]:
    return [
        MediaItem(media_id=f"m{i:05d}", label=i % 2, dataset_id="d",
                  source_id=f"s{i // 4}", subject_id=f"p{i // 8}")
        for i in range(n)
    ]


ITEMS = _items()


def _bench(decode: str | None, *, bias: float = 0.0, version: str = MODEL_ID) -> BenchmarkRun:
    return run_benchmark(
        DecoderSensitivePredictor(decode or "none", bias=bias),
        ITEMS,
        threshold=TH,
        effective_model_version=version,
        decode_identity=decode,
    )


def _witness_fields(**overrides) -> dict:
    base = dict(
        protocol_name="ffpp-train/celebdf-primary/v1",
        role="calibration",
        corpus_fingerprint="cafe1234",
        n_samples=400,
        temperature=1.234567,
        nll_before=0.69,
        nll_after=0.61,
        threshold=0.62,
        margin=DEFAULT_LOG_ODDS_MARGIN,
        target_fpr=0.02,
        selection_witness="sel12345",
        split_lock_hash="lock1234",
    )
    base.update(overrides)
    return base


def _witness(**overrides) -> CalibrationWitness:
    fields = _witness_fields(**overrides)
    return CalibrationWitness(
        **fields, witness=CalibrationWitness.compute_witness(**fields)
    )


# --------------------------------------------------------------------------
# 1. The benchmark defect, and the fix
# --------------------------------------------------------------------------


def test_the_legacy_run_lets_a_cross_decoder_comparison_pass():
    """The defect at Batch 8's *measured* magnitude, where it is silent.

    The bias here is 0.005 — the cross-decoder |dp| Batch 8 measured for a smooth
    scorer — not the 0.020 used by the test below. That distinction is the whole
    finding: at the realistic magnitude the gate does not fail, it *passes while
    spending a fifth of its own false-positive budget* on something it cannot
    see. A genuine regression consuming the remaining four fifths then ships.

    The gate's one note points the reader at the opposite conclusion — that the
    model version composition is incomplete — when what changed was the
    environment.
    """
    a = LegacyDecodeBlindRun.strip(_bench(UNDER_FFMPEG, bias=0.0))
    b = LegacyDecodeBlindRun.strip(_bench(UNDER_MSMF, bias=0.005))
    assert a.decode_identity is None and b.decode_identity is None
    assert a.dataset_fingerprint == b.dataset_fingerprint

    gate = regression_gate(a, b)
    assert gate.passed, (
        "the legacy double no longer passes a cross-decoder comparison; it can "
        f"no longer demonstrate R12's benchmark half. failures={gate.failures}"
    )
    assert not any("decode" in f for f in gate.failures)

    # It passed, and it was not free: measure what the decoder quietly spent.
    spent = (b.selective_fpr - a.selective_fpr) / 0.005  # the gate's allowance
    assert spent > 0.15, (
        f"the decoder consumed only {spent:.1%} of the FPR budget; the fixture "
        "no longer reproduces the silent-cost half of the finding"
    )
    print(f"        (gate PASSED; the decoder alone spent {spent:.1%} of the "
          "false-positive regression budget)")

    # And the shipped gate refuses the same pair outright.
    guarded = regression_gate(_bench(UNDER_FFMPEG, bias=0.0), _bench(UNDER_MSMF, bias=0.005))
    assert not guarded.passed
    assert any("different decode environments" in f for f in guarded.failures)


def test_the_gate_refuses_a_cross_decoder_comparison():
    """The fix. A refusal, on the same footing as a cross-dataset one."""
    a = _bench(UNDER_FFMPEG)
    b = _bench(UNDER_MSMF)
    gate = regression_gate(a, b)
    assert not gate.passed
    assert any("different decode environments" in f for f in gate.failures), gate.failures
    assert any(UNDER_FFMPEG in f and UNDER_MSMF in f for f in gate.failures)


def test_the_gate_is_silent_when_both_runs_share_an_environment():
    """The fix must not refuse the comparison everybody actually makes."""
    a = _bench(UNDER_FFMPEG)
    b = _bench(UNDER_FFMPEG, version="stub-1+c.739c3fec.g.ae232cba")
    gate = regression_gate(a, b)
    assert not any("decode" in f for f in gate.failures), gate.failures
    assert not any("decode identity" in n for n in gate.notes), gate.notes


def test_an_unrecorded_decode_identity_is_a_note_and_never_a_pass():
    """Three states, not two. No evidence is not evidence of sameness.

    The same principle R4 established for an unsealed split lock: reporting
    "cannot answer" as "answered yes" is the failure the field exists to remove,
    and a historical row must not be laundered into a guarantee it never made.
    """
    recorded = _bench(UNDER_FFMPEG)
    unrecorded = _bench(None)
    assert unrecorded.decode_identity is None

    for incumbent, challenger, expect in (
        (unrecorded, recorded, "the incumbent does not record"),
        (recorded, unrecorded, "the challenger does not record"),
        (unrecorded, unrecorded, "neither run records"),
    ):
        gate = regression_gate(incumbent, challenger)
        assert not any("decode environments" in f for f in gate.failures), (
            "an unrecorded identity must not be reported as a difference"
        )
        assert any(expect in n for n in gate.notes), (
            f"expected a note saying {expect!r}, got {gate.notes}"
        )


def test_a_decoder_change_alone_is_charged_to_the_model_without_the_field():
    """Why this is a defect and not bookkeeping. Measured, not asserted.

    A systematic decoder shift leaves AUC essentially untouched — it is close to
    rank-preserving — and moves the threshold-dependent metrics, which is the
    worst combination available: the metric that survives is the one the gate
    leans on, and the metric that moves is the one it blames the model for.
    """
    base = LegacyDecodeBlindRun.strip(_bench(UNDER_FFMPEG, bias=0.0))
    shifted = LegacyDecodeBlindRun.strip(_bench(UNDER_MSMF, bias=0.02))

    auc_move = abs(shifted.roc_auc - base.roc_auc)
    fpr_move = abs(shifted.selective_fpr - base.selective_fpr)
    assert auc_move < 0.005, f"AUC moved {auc_move:.6f}; the asymmetry has gone"
    assert fpr_move > 0.005, (
        f"selective FPR moved only {fpr_move:.6f}; the fixture no longer "
        "exercises the threshold-dependent half"
    )

    gate = regression_gate(base, shifted)
    assert not gate.passed
    assert any("false accusations" in f for f in gate.failures), (
        "the legacy gate should blame the model for a decoder change"
    )
    print(f"        (dAUC={auc_move:.6f}, dFPR={fpr_move:.6f}, "
          f"gate blames the model)")

    # And with the field present the same pair is refused for the right reason.
    guarded = regression_gate(_bench(UNDER_FFMPEG, bias=0.0), _bench(UNDER_MSMF, bias=0.02))
    assert any("different decode environments" in f for f in guarded.failures)


# --------------------------------------------------------------------------
# 2. The decomposition: a decode field, not a runtime identity
# --------------------------------------------------------------------------


def test_the_run_records_the_model_identity_not_the_runtime_identity():
    """Batch 9's decision, preserved. A worker rebuild is not a model change."""
    run = _bench(UNDER_FFMPEG)
    assert run.effective_model_version == MODEL_ID
    assert ".d." not in run.effective_model_version, (
        "the decoder has leaked into the model identity; every model-versus-"
        "model comparison is now also a decoder comparison"
    )


def test_the_runtime_identity_is_recoverable_from_the_pair():
    """Why a separate field beats storing the composed string.

    The decomposed pair carries strictly more information: the runtime identity
    can always be built from it, and the model identity cannot be recovered from
    the composed string without agreeing on a parse.
    """
    run = _bench(UNDER_FFMPEG)
    runtime = compose_runtime_version(
        model=run.effective_model_version, decode=run.decode_identity
    )
    assert runtime == f"{MODEL_ID}.d.{UNDER_FFMPEG}"

    from contracts.version import decode_identity_of, model_identity_of

    assert model_identity_of(runtime) == run.effective_model_version
    assert decode_identity_of(runtime) == run.decode_identity


def test_run_benchmark_does_not_detect_the_environment_itself():
    """It must be told, not guess. A confident wrong answer beats no answer only
    in systems that do not have to be audited.

    The harness process need not be the process that produced the scores —
    offline scoring, cached logits, a report assembled on another machine — so
    recording this process's environment would be a false claim.
    """
    run = run_benchmark(
        StubPredictor(), ITEMS, threshold=TH, effective_model_version=MODEL_ID
    )
    assert run.decode_identity is None, (
        "run_benchmark inferred a decode identity nobody gave it; the value "
        "would attest to this process rather than to the scoring one"
    )


# --------------------------------------------------------------------------
# 3. Benchmark history: the schema bump reads old rows
# --------------------------------------------------------------------------


def test_the_schema_version_moved_and_says_why():
    assert RUN_SCHEMA_VERSION == "1.2.0"
    assert _bench(UNDER_FFMPEG).schema_version == "1.2.0"


def test_a_pre_batch_history_row_still_loads():
    """No migration needed, and none performed. Absent means not recorded."""
    legacy = LegacyDecodeBlindRun.strip(_bench(UNDER_FFMPEG))
    payload = json.loads(legacy.to_json())
    payload.pop("decode_identity", None)
    assert "decode_identity" not in payload

    rebuilt = BenchmarkRun.from_dict(payload)
    assert rebuilt.decode_identity is None
    assert rebuilt.schema_version == "1.1.0"
    assert rebuilt.roc_auc == legacy.roc_auc


def test_a_mixed_schema_history_file_reads_end_to_end():
    path = TMP / "history.jsonl"
    old = LegacyDecodeBlindRun.strip(_bench(UNDER_FFMPEG))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        payload = json.loads(old.to_json())
        payload.pop("decode_identity", None)
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    append_history(_bench(UNDER_MSMF), path)

    runs = read_runs(path)
    assert [r.schema_version for r in runs] == ["1.1.0", "1.2.0"]
    assert [r.decode_identity for r in runs] == [None, UNDER_MSMF]


def test_the_decode_identity_is_not_back_filled():
    """Refused for the reason Batch 9 refused it for cache entries.

    Writing this environment into a historical row would assert that the run
    happened here, which nobody observed. ``None`` forever is the honest state.
    """
    legacy = LegacyDecodeBlindRun.strip(_bench(UNDER_FFMPEG))
    assert legacy.decode_identity is None
    assert "(unrecorded)" in legacy.describe()


# --------------------------------------------------------------------------
# 4. The calibration migration — the part that would have broken artifacts
# --------------------------------------------------------------------------


def test_the_naive_bump_would_have_made_every_stored_witness_unreadable():
    """The hazard, executed against the double rather than described.

    ``compute_witness`` hashes the schema version, so changing the constant
    changes the digest of unchanged fields. A build that did only that would
    recompute a stored witness's digest, find a mismatch, and raise — with a
    message accusing the operator of hand-editing a file that is fine.
    """
    fields = _witness_fields()
    stored_digest = LegacySchemaPinnedDigest.compute(schema="1.1.0", **fields)
    naive_digest = LegacySchemaPinnedDigest.compute(schema="1.2.0", **fields)
    assert stored_digest != naive_digest, (
        "the schema version no longer affects the digest; this migration is "
        "unnecessary and the versioned-digest machinery should be removed"
    )
    print(f"        (1.1.0 -> {stored_digest}, 1.2.0 -> {naive_digest})")


def test_a_stored_1_1_0_witness_still_loads_and_still_verifies():
    """The migration. Old artifacts stay verifiable, not merely tolerated."""
    fields = _witness_fields()
    stored = {
        **fields,
        "witness": LegacySchemaPinnedDigest.compute(schema="1.1.0", **fields),
        "schema_version": "1.1.0",
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    witness = CalibrationWitness.from_dict(stored)
    assert witness.schema_version == "1.1.0"
    assert witness.decode_identity is None
    assert witness.witness == stored["witness"]
    assert verify_calibration_witness(witness) == [], (
        "a 1.1.0 witness must verify under the 1.1.0 rule; verifying an old "
        "artifact against a new rule reports tampering where there is none"
    )


def test_a_new_witness_binds_its_decode_identity():
    a = _witness(decode_identity=UNDER_FFMPEG)
    b = _witness(decode_identity=UNDER_MSMF)
    assert a.schema_version == CALIBRATION_WITNESS_VERSION == "1.3.0"
    assert a.witness != b.witness, "the decode identity is not inside the digest"
    assert verify_calibration_witness(a) == []


def test_editing_the_decode_identity_of_a_stored_witness_is_detected():
    """Inside the digest, so it cannot be edited in place — the point of putting
    it there rather than beside it like ``created_at``."""
    payload = _witness(decode_identity=UNDER_FFMPEG).to_dict()
    payload["decode_identity"] = UNDER_MSMF
    try:
        CalibrationWitness.from_dict(payload)
    except ProtocolViolation as exc:
        assert "does not match its own fields" in str(exc)
    else:
        raise AssertionError("an edited decode identity was accepted")


def test_a_decode_identity_cannot_ride_on_a_schema_that_does_not_cover_it():
    """Recording an environment outside the digest is not evidence of it."""
    assert SCHEMA_VERSIONS_WITH_DECODE == frozenset({"1.2.0", "1.3.0"})
    try:
        CalibrationWitness.compute_witness(
            **_witness_fields(), decode_identity=UNDER_FFMPEG, schema_version="1.1.0"
        )
    except ProtocolViolation as exc:
        assert "no place for a decode identity" in str(exc)
    else:
        raise AssertionError("a 1.1.0 digest silently dropped the decode identity")

    payload = _witness(decode_identity=UNDER_FFMPEG).to_dict()
    payload["schema_version"] = "1.1.0"
    try:
        CalibrationWitness.from_dict(payload)
    except ValueError as exc:
        assert "does not cover it in the digest" in str(exc)
    else:
        raise AssertionError("a mislabelled payload was accepted")


def test_two_environments_could_share_one_witness_before_this_batch():
    """The calibration blind spot, and why the half was closed rather than
    deferred despite being weaker than the benchmark one.

    A decoder change usually moves the fitted temperature, so it usually moves
    the digest — detection worked by accident of what the digest covered. It
    fails exactly when two fits converge, and ``quantize`` to six decimals makes
    that reachable: two fits differing at the seventh decimal collapse to one.
    """
    fields = _witness_fields(temperature=1.234567)
    old_ff = LegacySchemaPinnedDigest.compute(schema="1.1.0", **fields)
    old_ms = LegacySchemaPinnedDigest.compute(schema="1.1.0", **fields)
    assert old_ff == old_ms, "the legacy double no longer shows the blind spot"

    new_ff = _witness(decode_identity=UNDER_FFMPEG).witness
    new_ms = _witness(decode_identity=UNDER_MSMF).witness
    assert new_ff != new_ms, "converged fits still share a witness"
    print(f"        (legacy: {old_ff} == {old_ms}; now: {new_ff} != {new_ms})")


def test_quantisation_really_does_merge_neighbouring_temperatures():
    """The premise of the test above, checked rather than assumed."""
    a = _witness(temperature=1.2345671, decode_identity=UNDER_FFMPEG)
    b = _witness(temperature=1.2345674, decode_identity=UNDER_FFMPEG)
    assert a.witness == b.witness, (
        "temperatures differing at the 7th decimal now produce different "
        "witnesses; the convergence blind spot is narrower than described"
    )


# --------------------------------------------------------------------------
# 5. The verifier can finally be asked the question
# --------------------------------------------------------------------------


def test_a_verifier_can_check_the_calibration_environment():
    witness = _witness(decode_identity=UNDER_FFMPEG)
    assert verify_calibration_witness(witness, decode_identity=UNDER_FFMPEG) == []

    problems = verify_calibration_witness(witness, decode_identity=UNDER_MSMF)
    assert any("fitted under decode environment" in p for p in problems), problems


def test_an_unrecorded_calibration_environment_is_reported_as_unanswerable():
    """Not silently passed. The same three-state rule as the gate's."""
    witness = _witness()
    assert witness.decode_identity is None
    assert verify_calibration_witness(witness) == []

    problems = verify_calibration_witness(witness, decode_identity=UNDER_FFMPEG)
    assert any("records no decode environment" in p for p in problems), problems


# --------------------------------------------------------------------------
# 6. Report-level cross-checks
# --------------------------------------------------------------------------


def _report(primary_decode, in_domain_decode, calibration_decode):
    from evaluation.benchmark import BenchmarkReport

    calibration = _witness(decode_identity=calibration_decode).to_dict()
    primary = run_benchmark(
        DecoderSensitivePredictor(primary_decode or "none"),
        ITEMS,
        threshold=TH,
        effective_model_version=MODEL_ID,
        decode_identity=primary_decode,
        calibration=calibration,
    )
    in_domain = run_benchmark(
        DecoderSensitivePredictor(in_domain_decode or "none"),
        _items(1600),
        threshold=TH,
        effective_model_version=MODEL_ID,
        decode_identity=in_domain_decode,
    )
    return BenchmarkReport(primary=primary, in_domain=in_domain, protocol="p/v1")


def test_a_report_refuses_a_calibration_fitted_under_another_decoder():
    problems = _report(UNDER_FFMPEG, UNDER_FFMPEG, UNDER_MSMF).validate()
    assert any(
        "fitted on logits from a different decoder's pixels" in p for p in problems
    ), problems


def test_a_report_refuses_two_runs_from_two_environments():
    problems = _report(UNDER_FFMPEG, UNDER_MSMF, UNDER_FFMPEG).validate()
    assert any("not two deployments" in p for p in problems), problems


def test_a_consistent_report_raises_neither_complaint():
    problems = _report(UNDER_FFMPEG, UNDER_FFMPEG, UNDER_FFMPEG).validate()
    assert not any("decoder" in p or "deployments" in p for p in problems), problems


def test_the_decode_identity_is_published_but_not_required():
    """Deliberately outside ``REQUIRED_REPORT_FIELDS``.

    ``check_report`` treats ``None`` as missing, so requiring it would make every
    historical report incomplete overnight for a field nothing can back-fill.
    Published so a reader can see it; not demanded so a reader can trust it.
    """
    from evaluation.protocol import REQUIRED_REPORT_FIELDS

    assert "decode_identity" not in REQUIRED_REPORT_FIELDS
    payload = _report(UNDER_FFMPEG, UNDER_FFMPEG, UNDER_FFMPEG).to_payload()
    assert payload["decode_identity"] == UNDER_FFMPEG
    assert payload["calibration_decode_identity"] == UNDER_FFMPEG


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


_run = _run  # noqa: PLW0127 - keep the fixture helper name; runner uses _run_tests


if __name__ == "__main__":
    raise SystemExit(_run())
