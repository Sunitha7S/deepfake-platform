"""R5-calibration: a calibration must prove which corpus it was fitted on.

THE THREAT MODEL, STATED PLAINLY
--------------------------------
The adversary is not a malicious engineer. It is an ordinary one, six months
from now, running a script somebody else wrote, on a corpus that has grown twice
since. Every defect this suite covers is reachable by accident and invisible in
the numbers afterwards.

The specific laundering path: ``fit_temperature`` and ``at_target_fpr`` take
bare arrays. Where those arrays came from is not an argument, is not recorded,
and cannot be recovered. The only account of provenance was
``CalibrationSpec.fitted_on`` — free text, written by the same caller that did
the fit. So a threshold chosen on Celeb-DF could be stamped
``fitted_on="ffpp-test"`` and carried into a Celeb-DF benchmark report that
passed every check in the system.

``legacy_calibrate`` below is that path, written out.
``test_the_legacy_path_launders_a_benchmark_fitted_threshold`` builds the false
claim with it and asserts it succeeds — and asserts, with numbers, that the
laundered threshold *flatters the result*, because "it is untraceable" is a
weaker argument than "it is untraceable and it inflates the number".

WHAT IS PROVED HERE
-------------------
Each test states a property, not a code path. In order: the legacy path
launders; the compliant path cannot be asked to; compliance cannot be asserted
without evidence; witnesses are reproducible and survive serialisation;
tampering is detected; and a report carries enough for a months-later
reconstruction with no other context.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.common import ThresholdInfo  # noqa: E402
from evaluation.benchmark import (  # noqa: E402
    BenchmarkReport,
    StubPredictor,
    dataset_fingerprint,
    run_benchmark,
)
from evaluation.calibration_protocol import (  # noqa: E402
    CALIBRATION_WITNESS_VERSION,
    MIN_WITNESSED_SAMPLES,
    CalibrationWitness,
    assert_not_benchmark_corpus,
    fit_calibration,
    verify_calibration_witness,
)
from evaluation.dataset import MediaItem  # noqa: E402
from evaluation.protocol import (  # noqa: E402
    FFPP_CELEBDF,
    REQUIRED_REPORT_FIELDS,
    DatasetRole,
    ProtocolViolation,
    select,
)
from evaluation.split_lock import create_lock  # noqa: E402
from evaluation.thresholds import at_target_fpr  # noqa: E402
from inference.calibration import fit_temperature, temperature_scale  # noqa: E402
from inference.models import CalibrationSpec, ModelSpec  # noqa: E402

BENCH_THRESHOLD = ThresholdInfo.at_margin(value=0.5, margin=0.2007, target_fpr=0.02)


# --------------------------------------------------------------------------
# A corpus big enough that the real sample floor applies
# --------------------------------------------------------------------------


def corpus(n_per_dataset: int = 1600) -> list[MediaItem]:
    """Sized so the ffpp CALIBRATION split clears MIN_WITNESSED_SAMPLES.

    The calibration role is the ffpp *test* split, 15% of the ffpp items, so
    1600 ffpp items give roughly 240 — deliberately above the 200 floor rather
    than comfortably above it, because a fixture that clears a limit by 10x
    stops testing the limit.
    """
    items: list[MediaItem] = []
    for dataset in ("ffpp", "celebdf"):
        for i in range(n_per_dataset):
            items.append(
                MediaItem(
                    f"{dataset}_m{i}", f"{dataset}_s{i}", dataset, i % 2,
                    subject_id=f"{dataset}_p{i}",
                )
            )
    return items


def setup():
    items = corpus()
    return items, create_lock(items)


def pick(items, lock, role, protocol=FFPP_CELEBDF):
    return select(protocol, role, items, lock.assignments, lock_hash=lock.content_hash())


def scores_for(selection, *, margin: float = 1.4, seed: int = 0):
    """Deterministic logits lined up with a selection, both classes present."""
    y = np.array([i.label for i in selection], dtype=np.float64)
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, y.size) + margin * (2.0 * y - 1.0)
    return z, y


FIT = {"target_fpr": 0.05, "margin": 0.2007}
"""Budget and margin used throughout.

A 5% budget rather than the shipped 2% because this fixture's calibration split
is small and a tighter budget puts the threshold at the largest negative score,
which makes the *threshold* an unstable subject for tests about witnesses. The
margin is passed explicitly rather than derived so these tests exercise one
thing; ``margin=DERIVE`` has its own suite in test_band_derivation.py.

Note what is no longer true: before Batch 7 this fixture could not use the
shipped band at all, because no 0.20-wide band fits around a threshold of 0.97
(R11). It fits now, at any threshold."""


# --------------------------------------------------------------------------
# 1. The legacy path launders
# --------------------------------------------------------------------------


def legacy_calibrate(logits, labels, *, fitted_on: str, target_fpr: float = 0.05):
    """The pre-Batch-6 path: bare arrays in, a free-text claim out.

    Exactly what was available before this batch — ``fit_temperature`` plus
    ``at_target_fpr`` plus a ``CalibrationSpec`` whose ``fitted_on`` says
    whatever the caller types.
    """
    t = fit_temperature(logits, labels)
    probabilities = np.asarray(temperature_scale(logits, t), dtype=np.float64)
    choice = at_target_fpr(labels, probabilities, target_fpr=target_fpr)
    spec = CalibrationSpec(
        temperature=t, fitted_on=fitted_on, n_samples=int(np.size(labels))
    )
    return spec, choice


def test_the_legacy_path_launders_a_benchmark_fitted_threshold():
    """The premise. A threshold fitted on the benchmark, labelled otherwise.

    Asserted with numbers: the laundered threshold does not merely lack
    provenance, it *flatters* the result. Fitting the operating point on the
    corpus it is then measured on spends the false-positive budget exactly, so
    the reported FPR is the best that corpus can produce.
    """
    items, lock = setup()
    benchmark = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    calibration = pick(items, lock, DatasetRole.CALIBRATION)

    z_bench, y_bench = scores_for(benchmark)
    z_cal, y_cal = scores_for(calibration, seed=1)

    # Fitted on the benchmark, stamped as the calibration split. Nothing objects.
    laundered, on_bench = legacy_calibrate(z_bench, y_bench, fitted_on="ffpp/calibration")
    assert laundered.fitted_on == "ffpp/calibration"
    assert laundered.witness is None
    assert not laundered.is_protocol_compliant

    honest, on_calibration = legacy_calibrate(
        z_cal, y_cal, fitted_on="ffpp/calibration"
    )
    assert honest.temperature != laundered.temperature, (
        "fixture: the two fits must differ, or there is nothing to launder"
    )

    # Now measure both operating points on the benchmark corpus — the number
    # that would actually be reported.
    from inference.metrics import confusion_matrix

    def on_benchmark(spec, choice):
        p = np.asarray(temperature_scale(z_bench, spec.temperature), dtype=np.float64)
        return confusion_matrix(y_bench, p, threshold=choice.value)

    cheated = on_benchmark(laundered, on_bench)
    transferred = on_benchmark(honest, on_calibration)

    # The in-sample threshold spends the false-positive budget *exactly*,
    # because it was chosen against these very negatives. The honestly
    # transferred one lands wherever it lands.
    assert abs(cheated.fpr - 0.05) < 1e-9, (
        f"the in-sample fit should spend the budget exactly, got {cheated.fpr}"
    )
    assert transferred.fpr < cheated.fpr, (
        "fixture: the transferred threshold should not also land on the budget"
    )

    # And that is the flattery, in the number anyone would quote: more
    # detections at a nominally identical budget. Measured 0.4037 vs 0.3525.
    assert cheated.tpr > transferred.tpr + 0.04, (
        f"laundering bought only {cheated.tpr - transferred.tpr:.4f} TPR; the "
        "fixture no longer demonstrates why this matters"
    )
    # Neither spec can be told apart by anything the system records.
    assert laundered.witness is honest.witness is None


def test_the_legacy_spec_cannot_be_distinguished_by_its_own_fields():
    """Two specs, one honest and one laundered, identical where it counts."""
    items, lock = setup()
    z, y = scores_for(pick(items, lock, DatasetRole.CALIBRATION))
    honest, _ = legacy_calibrate(z, y, fitted_on="ffpp/calibration")
    lie, _ = legacy_calibrate(z, y, fitted_on="celebdf/primary_benchmark")
    assert honest.temperature == lie.temperature
    assert honest.fitted_on != lie.fitted_on, "only the free text differs"
    assert honest.witness is lie.witness is None, (
        "neither carries evidence; the string is the only account of provenance"
    )


# --------------------------------------------------------------------------
# 2. The compliant path, and what it refuses
# --------------------------------------------------------------------------


def test_fitting_on_a_benchmark_selection_is_refused():
    """The structural guarantee, and the one that needs no verifier to run."""
    items, lock = setup()
    benchmark = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    z, y = scores_for(benchmark)
    try:
        fit_calibration(benchmark, z, y, **FIT)
    except ProtocolViolation as exc:
        assert "primary_benchmark" in str(exc)
        assert "not a prediction of anything" in str(exc)
        return
    raise AssertionError("a calibration was fitted on the benchmark corpus")


def test_fitting_on_the_training_selection_is_refused_too():
    items, lock = setup()
    train = pick(items, lock, DatasetRole.TRAIN)
    z, y = scores_for(train)
    try:
        fit_calibration(train, z, y, **FIT)
    except ProtocolViolation:
        return
    raise AssertionError("a calibration was fitted on the training split")


def test_bare_arrays_cannot_produce_a_witness():
    """The bare path stays available; it simply cannot make a claim."""
    items, lock = setup()
    z, y = scores_for(pick(items, lock, DatasetRole.CALIBRATION))
    try:
        fit_calibration([1, 2, 3], z, y, **FIT)
    except ProtocolViolation as exc:
        assert "requires a RoleSelection" in str(exc)
        return
    raise AssertionError("a witness was produced from a bare item list")


def test_the_compliant_path_produces_a_usable_witness():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    result = fit_calibration(selection, z, y, **FIT)
    w = result.witness

    assert w.protocol_name == FFPP_CELEBDF.name
    assert w.role == DatasetRole.CALIBRATION.value
    assert w.n_samples == len(selection) >= MIN_WITNESSED_SAMPLES
    assert w.corpus_fingerprint == dataset_fingerprint(selection.as_list())
    assert w.selection_witness == selection.witness
    assert w.split_lock_hash == lock.content_hash()
    assert w.created_at, "a witness must record when it was generated"
    assert w.schema_version == CALIBRATION_WITNESS_VERSION
    # The published operating point rebuilds through the contract's own
    # constructor, so the centred-band invariant holds on this path too.
    info = w.threshold_info()
    assert info.band_lo < info.value < info.band_hi
    assert not verify_calibration_witness(w), verify_calibration_witness(w)


def test_logits_that_do_not_match_the_selection_are_refused():
    """The check that stops benchmark logits reaching a calibration witness."""
    items, lock = setup()
    calibration = pick(items, lock, DatasetRole.CALIBRATION)
    benchmark = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    z, y = scores_for(benchmark)
    assert len(benchmark) != len(calibration), "fixture: the two must differ in size"
    try:
        fit_calibration(calibration, z, y, **FIT)
    except ProtocolViolation as exc:
        assert "do not correspond to the corpus" in str(exc)
        return
    raise AssertionError("benchmark logits were witnessed as a calibration")


def test_a_fit_below_the_sample_floor_is_refused():
    """There is no such thing as a compliant fit on 40 samples."""
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    try:
        fit_calibration(selection, z, y, min_samples=len(selection) + 1, **FIT)
    except ProtocolViolation as exc:
        assert "attest to noise" in str(exc)
        return
    raise AssertionError("a witness was issued for a fit below the floor")


def test_a_temperature_only_fit_records_no_threshold():
    """Recalibrating without moving the operating point is a real use."""
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    w = fit_calibration(selection, z, y, margin=None).witness
    assert w.threshold is None and w.margin is None and w.target_fpr is None
    assert w.temperature > 0
    try:
        w.threshold_info()
    except ValueError as exc:
        assert "records no threshold" in str(exc)
        return
    raise AssertionError("a threshold was invented for a temperature-only witness")


# --------------------------------------------------------------------------
# 3. Compliance cannot be asserted without evidence
# --------------------------------------------------------------------------


def test_a_spec_without_a_witness_is_not_compliant_however_it_is_labelled():
    lie = CalibrationSpec(temperature=1.3, fitted_on="ffpp/calibration", n_samples=5000)
    assert not lie.is_protocol_compliant


def test_a_spec_built_from_a_witness_is_compliant():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    spec = fit_calibration(selection, z, y, **FIT).witness.calibration_spec()
    assert spec.is_protocol_compliant
    assert spec.fitted_on == f"{FFPP_CELEBDF.name}/calibration"


def test_verification_of_a_missing_witness_says_no_evidence_not_verified():
    problems = verify_calibration_witness(None)
    assert problems and "no calibration witness" in problems[0]


def test_the_witness_is_not_in_the_calibration_fingerprint():
    """Provenance does not change a probability, so it must not fragment the cache.

    Pinned because the opposite is a tempting and wrong instinct: the fingerprint
    is half the result cache key and answers "would this deployment produce the
    same number?". Two deployments with identical T, threshold and band are
    cache-equivalent whether or not one can prove where it came from.
    """
    base = ModelSpec(name="m", version="w-1", gate_fingerprint="g0")
    witnessed = ModelSpec(
        name="m", version="w-1", gate_fingerprint="g0",
        calibration=CalibrationSpec(temperature=1.0, witness="abcd1234"),
    )
    assert base.calibration_fingerprint == witnessed.calibration_fingerprint
    assert base.effective_version == witnessed.effective_version


def test_the_shipped_stub_version_matches_its_configuration():
    """The pin, and what moved it.

    Batch 6 asserted ``dc6fcc65`` here to show that *adding the witness field*
    invalidated no cached result — provenance changes no probability, so it must
    not fragment the cache.

    Batch 7 moved it to ``739c3fec``, deliberately and for the opposite reason.
    The band geometry changed, so the shipped operating point moved from
    [0.5200, 0.7200) to [0.5210, 0.7099) and scores in the slivers between those
    edges now get a different verdict. A geometry change that left the
    fingerprint alone would have served the old verdicts under the new rule —
    exactly the Batch 1 defect the operating point was put into the fingerprint
    to prevent. The literal moving here is the invariant working.

    What must still hold is the *witness* half of the Batch 6 claim, and it is
    checked by the test above rather than by this literal.
    """
    from backend.config import STUB_ENGINE_VERSION

    spec = ModelSpec(name="stub", version="stub-0")
    assert spec.calibration_fingerprint == "739c3fec"
    assert STUB_ENGINE_VERSION.startswith(spec.effective_version)


# --------------------------------------------------------------------------
# 4. Reproducibility, serialisation, tampering
# --------------------------------------------------------------------------


def test_identical_inputs_produce_identical_witnesses():
    """Refit, compare, expect silence — the cheapest reproducibility check."""
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    a = fit_calibration(selection, z, y, **FIT).witness
    b = fit_calibration(selection, z, y, **FIT).witness
    assert a.witness == b.witness
    assert a.temperature == b.temperature


def test_the_timestamp_is_recorded_but_does_not_move_the_digest():
    """Both requirements at once: provenance kept, determinism kept."""
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    a = fit_calibration(selection, z, y, created_at="2026-01-01T00:00:00+00:00", **FIT)
    b = fit_calibration(selection, z, y, created_at="2099-12-31T23:59:59+00:00", **FIT)
    assert a.witness.created_at != b.witness.created_at
    assert a.witness.witness == b.witness.witness, (
        "the clock moved the witness digest; identical inputs no longer produce "
        "identical witnesses"
    )


def test_reordering_the_calibration_set_does_not_move_the_witness():
    """The R1 property, inherited one level up.

    Without quantisation and the fsum fit, a reordered calibration set would
    produce a different temperature at ~1e-8 and therefore a different witness,
    and a verifier would report tampering where there was float noise.
    """
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    order = np.random.default_rng(5).permutation(z.size)

    straight = fit_calibration(selection, z, y, **FIT).witness
    # Reorder the arrays only; the selection (and so the corpus fingerprint,
    # which is over a sorted id list) is unchanged.
    shuffled = fit_calibration(selection, z[order], y[order], **FIT).witness
    assert straight.temperature == shuffled.temperature
    assert straight.witness == shuffled.witness


def test_a_witness_survives_serialisation():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    original = fit_calibration(selection, z, y, **FIT).witness
    restored = CalibrationWitness.from_dict(original.to_dict())
    assert restored == original
    assert not verify_calibration_witness(restored)


def test_a_witness_survives_a_json_round_trip():
    import json

    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    original = fit_calibration(selection, z, y, **FIT).witness
    restored = CalibrationWitness.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.witness == original.witness


def test_an_edited_threshold_is_detected():
    """The realistic tamper: adjust a number in a JSON file, leave the rest."""
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    payload = fit_calibration(selection, z, y, **FIT).witness.to_dict()
    payload["threshold"] = 0.5
    try:
        CalibrationWitness.from_dict(payload)
    except ProtocolViolation as exc:
        assert "does not match its own fields" in str(exc)
        return
    raise AssertionError("an edited threshold passed the witness check")


def test_an_edited_sample_count_is_detected():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    payload = fit_calibration(selection, z, y, **FIT).witness.to_dict()
    payload["n_samples"] = 99999
    try:
        CalibrationWitness.from_dict(payload)
    except ProtocolViolation:
        return
    raise AssertionError("an inflated sample count passed the witness check")


def test_a_witness_relabelled_to_a_benchmark_role_is_refused():
    """Editing the role is the laundering move, so it is refused at construction."""
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    payload = fit_calibration(selection, z, y, **FIT).witness.to_dict()
    payload["role"] = DatasetRole.PRIMARY_BENCHMARK.value
    payload["witness"] = CalibrationWitness.compute_witness(
        protocol_name=payload["protocol_name"], role=payload["role"],
        corpus_fingerprint=payload["corpus_fingerprint"], n_samples=payload["n_samples"],
        temperature=payload["temperature"], nll_before=payload["nll_before"],
        nll_after=payload["nll_after"], threshold=payload["threshold"],
        margin=payload["margin"], target_fpr=payload["target_fpr"],
        selection_witness=payload["selection_witness"],
        split_lock_hash=payload["split_lock_hash"],
    )
    try:
        CalibrationWitness.from_dict(payload)
    except ProtocolViolation as exc:
        assert "only 'calibration'" in str(exc)
        return
    raise AssertionError("a re-sealed benchmark-role witness was accepted")


def test_an_unknown_field_is_refused():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    payload = fit_calibration(selection, z, y, **FIT).witness.to_dict()
    payload["from_the_future"] = 1
    try:
        CalibrationWitness.from_dict(payload)
    except ValueError as exc:
        assert "from_the_future" in str(exc)
        return
    raise AssertionError("a witness from a newer schema was read anyway")


# --------------------------------------------------------------------------
# 5. Witness / corpus mismatch
# --------------------------------------------------------------------------


def test_verification_detects_a_corpus_mismatch():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    w = fit_calibration(selection, z, y, **FIT).witness

    assert not verify_calibration_witness(w, items=selection.as_list())
    other = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK).as_list()
    problems = verify_calibration_witness(w, items=other)
    assert any("corpus fingerprint mismatch" in p for p in problems), problems


def test_verification_detects_a_protocol_mismatch():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    w = fit_calibration(selection, z, y, **FIT).witness
    problems = verify_calibration_witness(w, protocol="somebody-elses/v3")
    assert any("was fitted under protocol" in p for p in problems), problems


def test_verification_detects_a_selection_mismatch():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    w = fit_calibration(selection, z, y, **FIT).witness
    problems = verify_calibration_witness(
        w, selection=pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    )
    assert any("does not correspond to this selection" in p for p in problems), problems
    assert any("no calibration may be fitted on" in p for p in problems), problems


def test_verification_detects_a_lock_mismatch():
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    w = fit_calibration(selection, z, y, **FIT).witness
    problems = verify_calibration_witness(w, split_lock_hash="sha256:" + "0" * 64)
    assert any("was resolved under lock" in p for p in problems), problems


def test_benchmark_data_cannot_masquerade_as_calibration_data():
    """The report-level backstop for the case the role check cannot see."""
    items, lock = setup()
    selection = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(selection)
    w = fit_calibration(selection, z, y, **FIT).witness

    assert not assert_not_benchmark_corpus(w, "some-other-corpus")
    problems = assert_not_benchmark_corpus(w, w.corpus_fingerprint)
    assert any("fitted on the very items being benchmarked" in p for p in problems)


# --------------------------------------------------------------------------
# 6. A report reconstructable months later
# --------------------------------------------------------------------------


def compliant_report():
    items, lock = setup()
    calibration = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(calibration)
    witness = fit_calibration(calibration, z, y, **FIT).witness

    primary_sel = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    hard = [i for i in primary_sel if int(i.media_id.rsplit("m", 1)[1]) % 4 < 2]
    shared = {"split_lock": lock, "calibration": witness}
    primary = run_benchmark(
        StubPredictor(version="v1"), primary_sel, threshold=BENCH_THRESHOLD,
        effective_model_version="v1", slice_by={"hard": hard}, **shared,
    )
    in_domain = run_benchmark(
        StubPredictor(version="v1"), calibration, threshold=BENCH_THRESHOLD,
        effective_model_version="v1", **shared,
    )
    report = BenchmarkReport(
        primary=primary, in_domain=in_domain, protocol=FFPP_CELEBDF.name
    )
    return items, lock, witness, report


def test_a_compliant_report_validates():
    _, _, _, report = compliant_report()
    assert not report.validate(), report.describe()


def test_the_report_carries_every_required_field_including_both_witnesses():
    _, _, witness, report = compliant_report()
    payload = report.to_payload()
    missing = [f for f in REQUIRED_REPORT_FIELDS if payload.get(f) in (None, "")]
    assert not missing, f"missing {missing}"
    assert payload["calibration_witness"] == witness.witness
    assert payload["selection_witness"] == report.primary.selection_witness


def test_the_report_alone_answers_the_audit_questions():
    """Months later, with nothing but the payload: what was this calibrated on?"""
    _, _, witness, report = compliant_report()
    payload = report.to_payload()

    assert payload["protocol"] == FFPP_CELEBDF.name
    assert payload["calibration_role"] == DatasetRole.CALIBRATION.value
    assert payload["calibration_corpus_fingerprint"] == witness.corpus_fingerprint
    assert payload["calibration_n_samples"] == witness.n_samples
    assert payload["split_lock_hash"]
    # And the question that matters most: was it calibrated on what it reports?
    assert payload["calibration_corpus_fingerprint"] != payload["dataset_fingerprint"]


def test_a_report_with_no_calibration_evidence_is_refused():
    """Compliance cannot be asserted by omission."""
    items, lock = setup()
    primary_sel = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    calibration = pick(items, lock, DatasetRole.CALIBRATION)
    runs = [
        run_benchmark(
            StubPredictor(version="v1"), sel, threshold=BENCH_THRESHOLD,
            effective_model_version="v1", split_lock=lock,
        )
        for sel in (primary_sel, calibration)
    ]
    report = BenchmarkReport(
        primary=runs[0], in_domain=runs[1], protocol=FFPP_CELEBDF.name
    )
    problems = report.validate()
    assert any("no calibration witness" in p for p in problems), report.describe()


def test_a_report_whose_calibration_is_the_benchmark_corpus_is_refused():
    """The masquerade, caught at report level even when the run looks fine."""
    items, lock = setup()
    calibration = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(calibration)
    witness = fit_calibration(calibration, z, y, **FIT).witness

    # Benchmark the *calibration* corpus: the fingerprints now coincide.
    run = run_benchmark(
        StubPredictor(version="v1"), calibration, threshold=BENCH_THRESHOLD,
        effective_model_version="v1", split_lock=lock, calibration=witness,
    )
    other = run_benchmark(
        StubPredictor(version="v1"), pick(items, lock, DatasetRole.PRIMARY_BENCHMARK),
        threshold=BENCH_THRESHOLD, effective_model_version="v1",
        split_lock=lock, calibration=witness,
    )
    report = BenchmarkReport(primary=run, in_domain=other, protocol=FFPP_CELEBDF.name)
    problems = report.validate()
    assert any("the very items being benchmarked" in p for p in problems), (
        report.describe()
    )


def test_a_run_refuses_a_witness_beside_a_contradicting_fingerprint():
    items, lock = setup()
    calibration = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(calibration)
    witness = fit_calibration(calibration, z, y, **FIT).witness
    try:
        run_benchmark(
            StubPredictor(version="v1"),
            pick(items, lock, DatasetRole.PRIMARY_BENCHMARK),
            threshold=BENCH_THRESHOLD, effective_model_version="v1",
            calibration=witness, calibration_fingerprint="something-else",
        )
    except ProtocolViolation as exc:
        assert "one of the two is wrong" in str(exc)
        return
    raise AssertionError("a run recorded a fingerprint its witness contradicts")


def test_a_report_detects_a_calibration_from_a_different_split_lock():
    """Same protocol, different frozen corpus — not the same evaluation."""
    items, lock = setup()
    calibration = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(calibration)
    witness = fit_calibration(
        calibration, z, y, split_lock_hash="sha256:" + "1" * 64, **FIT
    ).witness
    primary = run_benchmark(
        StubPredictor(version="v1"), pick(items, lock, DatasetRole.PRIMARY_BENCHMARK),
        threshold=BENCH_THRESHOLD, effective_model_version="v1",
        split_lock=lock, calibration=witness,
    )
    in_domain = run_benchmark(
        StubPredictor(version="v1"), calibration, threshold=BENCH_THRESHOLD,
        effective_model_version="v1", split_lock=lock, calibration=witness,
    )
    problems = BenchmarkReport(
        primary=primary, in_domain=in_domain, protocol=FFPP_CELEBDF.name
    ).validate()
    assert any("not the same frozen corpus" in p for p in problems), problems


def test_a_run_survives_a_history_round_trip_with_its_calibration():
    import json

    from evaluation.benchmark import BenchmarkRun

    _, _, witness, report = compliant_report()
    restored = BenchmarkRun.from_dict(json.loads(report.primary.to_json()))
    assert restored.calibration["witness"] == witness.witness
    assert restored == report.primary


def test_a_pre_batch6_history_row_still_loads():
    """Backward compatibility: enforcement is forward-looking."""
    import json

    from evaluation.benchmark import BenchmarkRun

    _, _, _, report = compliant_report()
    row = json.loads(report.primary.to_json())
    del row["calibration"]
    restored = BenchmarkRun.from_dict(row)
    assert restored.calibration is None
    assert restored.roc_auc == report.primary.roc_auc


# --------------------------------------------------------------------------
# 7. R31 — pre_calibration_identity binds the witness to a model configuration
# --------------------------------------------------------------------------
#
# Before this batch, nothing about the model's weights, gate or selection
# policy reached CalibrationWitness's digest at all — only the corpus and
# (since Batch 11) the decode environment did. Two models differing in gate
# or selection but scoring a fixed calibration corpus identically produced
# byte-identical witnesses. See docs/R31_DESIGN.md §2 for the live collision
# study this closes.


def test_a_model_config_difference_now_moves_the_witness():
    """The headline. Same corpus, same logits, different model configuration."""
    items, lock = setup()
    calibration = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(calibration)

    a = fit_calibration(
        calibration, z, y, **FIT, pre_calibration_identity="cfg-a"
    ).witness
    b = fit_calibration(
        calibration, z, y, **FIT, pre_calibration_identity="cfg-b"
    ).witness
    assert a.witness != b.witness, (
        "identical logits fitted under two different model configurations "
        "produced the same witness"
    )

    # And the converse: two calls agreeing on both logits and configuration
    # still agree with each other (this is not a use-created-different digest
    # for any two calls — it is specifically config-sensitive).
    c = fit_calibration(
        calibration, z, y, **FIT, pre_calibration_identity="cfg-a"
    ).witness
    assert a.witness == c.witness


def test_pre_calibration_identity_is_recoverable_for_forensics():
    items, lock = setup()
    calibration = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(calibration)
    witness = fit_calibration(
        calibration, z, y, **FIT, pre_calibration_identity="cfg-a"
    ).witness
    assert witness.pre_calibration_identity == "cfg-a"
    assert witness.to_dict()["pre_calibration_identity"] == "cfg-a"


def test_omitting_pre_calibration_identity_records_none():
    """The honest default — no config named is not the same as "matches"."""
    items, lock = setup()
    calibration = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(calibration)
    witness = fit_calibration(calibration, z, y, **FIT).witness
    assert witness.pre_calibration_identity is None


def test_a_1_2_0_payload_cannot_smuggle_a_pre_calibration_identity():
    """Mirrors the existing decode_identity guard: a pre-1.3.0 schema has no
    digest slot for this field, so silently carrying it would be unprotected."""
    try:
        CalibrationWitness.compute_witness(
            protocol_name="p", role=DatasetRole.CALIBRATION.value,
            corpus_fingerprint="cf", n_samples=200, temperature=1.0,
            nll_before=0.5, nll_after=0.4, threshold=None, margin=None,
            target_fpr=None, selection_witness="sw", split_lock_hash="sl",
            pre_calibration_identity="cfg-a", schema_version="1.2.0",
        )
    except ProtocolViolation as exc:
        assert "pre-calibration identity" in str(exc)
        return
    raise AssertionError("a 1.2.0 schema accepted a pre_calibration_identity")


def test_verify_calibration_witness_cross_checks_pre_calibration_identity():
    items, lock = setup()
    calibration = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(calibration)

    recorded = fit_calibration(
        calibration, z, y, **FIT, pre_calibration_identity="cfg-a"
    ).witness
    unrecorded = fit_calibration(calibration, z, y, **FIT).witness

    assert not verify_calibration_witness(
        recorded, pre_calibration_identity="cfg-a"
    )
    mismatch = verify_calibration_witness(recorded, pre_calibration_identity="cfg-b")
    assert any("cfg-a" in p and "cfg-b" in p for p in mismatch), mismatch

    unanswerable = verify_calibration_witness(
        unrecorded, pre_calibration_identity="cfg-a"
    )
    assert any("no model configuration" in p for p in unanswerable), unanswerable


def test_model_spec_exposes_a_pre_calibration_identity():
    """Positive + negative control for the no-circularity property
    (docs/R31_DESIGN.md §1.3): gate/selection move it, calibration does not.
    """
    from dataclasses import replace

    base = ModelSpec(name="m", version="v1")
    gated = base.with_gate("g1")
    selected = base.with_selection("s1")
    recalibrated = replace(base, calibration=CalibrationSpec(temperature=2.0, n_samples=500))

    assert gated.pre_calibration_identity != base.pre_calibration_identity
    assert selected.pre_calibration_identity != base.pre_calibration_identity
    assert recalibrated.pre_calibration_identity == base.pre_calibration_identity
    assert recalibrated.effective_version != base.effective_version


def test_an_old_calibration_witness_still_verifies_after_the_change():
    """Direct analogue of the 1.1.0 -> 1.2.0 precedent: a 1.2.0 witness with
    no pre_calibration_identity round-trips and its digest still matches,
    because compute_witness dispatches on the payload's own schema_version.
    """
    items, lock = setup()
    calibration = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(calibration)
    result = fit_calibration(calibration, z, y, **FIT, decode_identity="c20a064a")
    legacy_digest = CalibrationWitness.compute_witness(
        protocol_name=result.witness.protocol_name,
        role=result.witness.role,
        corpus_fingerprint=result.witness.corpus_fingerprint,
        n_samples=result.witness.n_samples,
        temperature=result.witness.temperature,
        nll_before=result.witness.nll_before,
        nll_after=result.witness.nll_after,
        threshold=result.witness.threshold,
        margin=result.witness.margin,
        target_fpr=result.witness.target_fpr,
        selection_witness=result.witness.selection_witness,
        split_lock_hash=result.witness.split_lock_hash,
        decode_identity="c20a064a",
        schema_version="1.2.0",
    )
    legacy = CalibrationWitness(
        protocol_name=result.witness.protocol_name,
        role=result.witness.role,
        corpus_fingerprint=result.witness.corpus_fingerprint,
        n_samples=result.witness.n_samples,
        temperature=result.witness.temperature,
        nll_before=result.witness.nll_before,
        nll_after=result.witness.nll_after,
        threshold=result.witness.threshold,
        margin=result.witness.margin,
        target_fpr=result.witness.target_fpr,
        selection_witness=result.witness.selection_witness,
        split_lock_hash=result.witness.split_lock_hash,
        witness=legacy_digest,
        decode_identity="c20a064a",
        schema_version="1.2.0",
    )
    payload = legacy.to_dict()
    reloaded = CalibrationWitness.from_dict(payload)  # __post_init__ raises on mismatch
    assert reloaded.pre_calibration_identity is None
    assert reloaded.witness == legacy_digest


def test_the_report_now_cross_checks_decode_identity():
    """R31 D1: BenchmarkReport.validate() now wires the existing
    decode_identity cross-check that was previously computed but unused.
    """
    items, lock = setup()
    calibration = pick(items, lock, DatasetRole.CALIBRATION)
    z, y = scores_for(calibration)
    witness = fit_calibration(
        calibration, z, y, **FIT, decode_identity="env-a"
    ).witness

    primary_sel = pick(items, lock, DatasetRole.PRIMARY_BENCHMARK)
    shared = {"split_lock": lock, "calibration": witness}
    primary = run_benchmark(
        StubPredictor(version="v1"), primary_sel, threshold=BENCH_THRESHOLD,
        effective_model_version="v1", decode_identity="env-b", **shared,
    )
    in_domain = run_benchmark(
        StubPredictor(version="v1"), calibration, threshold=BENCH_THRESHOLD,
        effective_model_version="v1", decode_identity="env-b", **shared,
    )
    report = BenchmarkReport(
        primary=primary, in_domain=in_domain, protocol=FFPP_CELEBDF.name
    )
    problems = report.validate()
    assert any("env-a" in p and "env-b" in p for p in problems), report.describe()


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
