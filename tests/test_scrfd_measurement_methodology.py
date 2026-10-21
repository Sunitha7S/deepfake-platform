"""SCRFD Integration Program, Phase 3 — regression tests for the measurement
methodology correction. See ``docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md``.

Tests the pure aggregation/statistics functions in ``scrfd_detector_study.py``
against hand-constructed ``image_trials`` fixtures — no real photographs, no
SCRFD asset, no ``onnxruntime``. The methodology bug this batch fixes (plan
Sec2) lives entirely in how per-trial scores are aggregated into a spread
ratio, not in detection or scoring itself, so the aggregation functions are
tested directly and hermetically, independent of Phase 0's asset fetch and
Sec3's photo-corpus acquisition (both still unresolved, per the plan's own
Sec11 gate).

FAIL-BEFORE / PASS-AFTER
-------------------------
``_shipped_before_amplification`` below reproduces the formula
``scrfd_detector_study.py`` shipped with *before* this batch (one noise
realisation per image, ``std`` taken across images — plan Sec2.2). It is
reproduced here, inline, ONLY to prove that formula cannot even evaluate the
single-fixture case the original 310x figure came from (``docs/
INFERENCE_DETERMINISM.md`` Sec5.1-Sec5.2: one fixture, 11 realisations). It
is not reintroduced into ``scrfd_detector_study.py`` itself — the corrected
script only ever computes the pooled, multi-realisation statistic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import scrfd_detector_study as study  # noqa: E402

# --------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------


def _trial(*, lost: bool = False, move_pct: float | None = 0.0, fixed=None, free=None) -> dict:
    if lost:
        return {"lost": True, "short_edge_move_pct": None, "fixed_logit": None, "free_logit": None}
    return {"lost": False, "short_edge_move_pct": move_pct, "fixed_logit": fixed, "free_logit": free}


def _image(trials: list[dict], *, box0_present: bool = True, clean_detection_failure: bool = False) -> dict:
    return {
        "box0_present": box0_present,
        "clean_detection_failure": clean_detection_failure,
        "trials": trials,
    }


def _shipped_before_amplification(image_trials: list[dict]) -> float:
    """Reproduces the formula ``scrfd_detector_study.py`` shipped with
    before this batch (plan Sec2.2, quoted):

        fixed_spread = np.std(fixed_crop_logits)   # std across N images
        free_spread = np.std(free_crop_logits)     # std across N images
        amplification = free_spread / fixed_spread

    i.e. exactly one realisation (index 0) per image, aggregated across
    images. Reproduced here only to prove it does not compute the
    criterion's own statistic; ``scrfd_detector_study.py`` itself no longer
    contains this code path."""
    fixed = [
        img["trials"][0]["fixed_logit"]
        for img in image_trials
        if img["trials"] and img["trials"][0]["fixed_logit"] is not None
    ]
    free = [
        img["trials"][0]["free_logit"]
        for img in image_trials
        if img["trials"] and img["trials"][0]["free_logit"] is not None
    ]
    fixed_spread = float(np.std(fixed)) if fixed else 0.0
    free_spread = float(np.std(free)) if free else 0.0
    return (free_spread / fixed_spread) if fixed_spread > 0 else float("nan")


def _single_fixture_trials(k: int) -> list[dict]:
    """One image, ``k`` realisations — matching the original 310x
    measurement's own shape (docs/INFERENCE_DETERMINISM.md Sec5.1: one
    fixture, 11 realisations). ``fixed_logit`` barely moves (small spread);
    ``free_logit`` moves a lot (large spread) — the R13 mechanism itself
    (Sec5.2 of that doc), values chosen only so the true ratio is knowably
    > 1, not to match 310x exactly."""
    rng = np.random.default_rng(42)
    fixed_vals = rng.normal(loc=0.5, scale=0.001, size=k)
    free_vals = rng.normal(loc=0.5, scale=0.05, size=k)
    trials = [_trial(fixed=float(f), free=float(fr)) for f, fr in zip(fixed_vals, free_vals)]
    return [_image(trials)]


# --------------------------------------------------------------------------
# A — fail-before / pass-after: a single fixture, the original measurement's
#     own shape (one image, 11 realisations)
# --------------------------------------------------------------------------


def test_old_methodology_cannot_evaluate_a_single_fixture_at_all():
    """Plan Sec2.1: the original 310x figure came from ONE fixture and 11
    realisations. The pre-Phase-3 shipped formula draws exactly one
    realisation per image, so on a single-image corpus it always reduces to
    ``std`` of a 1-element list for both pools -> 0.0/0.0 -> nan. It cannot
    reproduce anything comparable to the original measurement's own shape,
    even in the ideal case of zero cross-image content variance (a
    single-image corpus has none, by construction)."""
    image_trials = _single_fixture_trials(k=11)
    old_amplification = _shipped_before_amplification(image_trials)
    assert np.isnan(old_amplification), (
        "expected the shipped-before formula to be undefined (nan) on a "
        f"single-fixture corpus, got {old_amplification}"
    )


def test_new_methodology_computes_a_well_defined_amplification_on_the_same_fixture():
    """The corrected, pooled multi-realisation formula (plan Sec4.1-Sec4.2)
    computes a real, well-defined ratio on exactly the same single-fixture
    data the old formula could not evaluate at all — because it draws
    ``k_realisations`` (>= 11) per image instead of one, matching the
    original measurement's own procedure (one fixture, many realisations)
    rather than conflating draws with images."""
    image_trials = _single_fixture_trials(k=11)
    fixed_scores, free_scores = study.pool_amplification_scores(image_trials)
    assert len(fixed_scores) == len(free_scores) == 11
    amplification = study.amplification_ratio(fixed_scores, free_scores)
    assert not np.isnan(amplification)
    assert amplification > 1.0, "free-crop spread should exceed fixed-crop spread by construction"


def test_new_methodology_matches_a_hand_computed_ratio():
    """Sanity: pool_amplification_scores + amplification_ratio reproduce
    exactly ``np.std(free) / np.std(fixed)`` over the pooled trials — the
    same formula shape as the original measurement (docs/
    INFERENCE_DETERMINISM.md Sec5A(c)), just applied to pooled ``(i, k)``
    trials instead of one fixture's 11 realisations."""
    image_trials = _single_fixture_trials(k=11)
    fixed_scores, free_scores = study.pool_amplification_scores(image_trials)
    expected = np.std(free_scores) / np.std(fixed_scores)
    got = study.amplification_ratio(fixed_scores, free_scores)
    assert abs(got - expected) < 1e-12


# --------------------------------------------------------------------------
# B — pooling across a multi-image corpus is the union of every image's
#     realisations, not a second layer of per-image averaging (plan Sec4.2)
# --------------------------------------------------------------------------


def test_pooling_is_the_union_not_a_per_image_average():
    """Two images with different realisation counts and different
    per-image variances: the pooled statistic must equal ``std`` over the
    concatenation of every ``(i, k)`` trial, not the average of two
    per-image ratios (plan Sec4.2's own stated rejection of 'a second layer
    of averaging')."""
    rng = np.random.default_rng(7)
    img_a_fixed = rng.normal(0.5, 0.001, size=11)
    img_a_free = rng.normal(0.5, 0.02, size=11)
    img_b_fixed = rng.normal(0.3, 0.001, size=20)
    img_b_free = rng.normal(0.3, 0.08, size=20)
    image_trials = [
        _image([_trial(fixed=float(f), free=float(fr)) for f, fr in zip(img_a_fixed, img_a_free)]),
        _image([_trial(fixed=float(f), free=float(fr)) for f, fr in zip(img_b_fixed, img_b_free)]),
    ]
    fixed_scores, free_scores = study.pool_amplification_scores(image_trials)
    assert len(fixed_scores) == 31 and len(free_scores) == 31
    pooled = study.amplification_ratio(fixed_scores, free_scores)
    per_image_average = (
        study.amplification_ratio(list(img_a_fixed), list(img_a_free))
        + study.amplification_ratio(list(img_b_fixed), list(img_b_free))
    ) / 2.0
    assert abs(pooled - per_image_average) > 1e-6, (
        "pooling collapsed to the per-image average it is supposed to differ from"
    )


# --------------------------------------------------------------------------
# C — face-loss accounting: N x K trials, rule-of-three bound, clean-image
#     detection failures tracked separately (plan Sec3.1, Sec4.1 step 1)
# --------------------------------------------------------------------------


def test_face_loss_is_counted_over_every_trial_not_every_image():
    image_trials = [
        _image([_trial(fixed=0.5, free=0.5) for _ in range(11)]),
        _image([_trial(lost=True)] + [_trial(fixed=0.5, free=0.5) for _ in range(10)]),
    ]
    summary = study.face_loss_summary(image_trials)
    assert summary["total_trials"] == 22
    assert summary["lost_trials"] == 1
    assert summary["clean_detection_failures"] == 0


def test_clean_detection_failure_is_not_counted_as_a_noise_induced_loss():
    """Plan Sec4.1 step 1: an image whose clean pixels never got a box_0 is
    a reportable fact, but a different one from a face lost under noise —
    it must not inflate ``lost_trials``."""
    image_trials = [
        _image([], box0_present=False, clean_detection_failure=True),
        _image([_trial(fixed=0.5, free=0.5) for _ in range(11)]),
    ]
    summary = study.face_loss_summary(image_trials)
    assert summary["clean_detection_failures"] == 1
    assert summary["lost_trials"] == 0
    assert summary["total_trials"] == 11


def test_rule_of_three_bound_tightens_with_more_trials():
    """Plan Sec3.1: N=200 images x K=1 gives ~1.5%; N=200 x K=11 gives
    ~0.14% — an order of magnitude tighter for the same 200-image floor,
    with zero observed losses in both cases."""
    few = study.face_loss_summary([_image([_trial(fixed=0.5, free=0.5)]) for _ in range(200)])
    many = study.face_loss_summary(
        [_image([_trial(fixed=0.5, free=0.5) for _ in range(11)]) for _ in range(200)]
    )
    assert few["total_trials"] == 200
    assert many["total_trials"] == 2200
    assert few["rule_of_three_upper_bound_pct"] > many["rule_of_three_upper_bound_pct"] * 5
    assert abs(few["rule_of_three_upper_bound_pct"] - 1.5) < 0.2
    assert abs(many["rule_of_three_upper_bound_pct"] - 0.136) < 0.02


# --------------------------------------------------------------------------
# D — bootstrap CI: cluster by image (plan Sec4.3)
# --------------------------------------------------------------------------


def test_bootstrap_ci_is_well_formed_and_reproducible():
    image_trials = [
        _image([_trial(fixed=0.5 + 0.0001 * i, free=0.5 + 0.01 * i) for i in range(11)])
        for _ in range(20)
    ]
    point1, lo1, hi1 = study.bootstrap_amplification_ci(image_trials, resamples=200, seed=0)
    point2, lo2, hi2 = study.bootstrap_amplification_ci(image_trials, resamples=200, seed=0)
    assert (point1, lo1, hi1) == (point2, lo2, hi2), "same seed must reproduce the same CI"
    assert lo1 <= hi1
    assert point1 > 0


def test_bootstrap_ci_uses_a_different_seed_for_a_different_draw_sequence():
    """Plan Sec5.4's two-seed cross-check presumes different seeds actually
    draw different resamples — this pins that the seed is wired through.
    Images must actually differ from each other, or every resample (any
    seed) is statistically identical regardless of which images it drew."""
    rng = np.random.default_rng(3)
    image_trials = [
        _image(
            [
                _trial(fixed=float(f), free=float(fr))
                for f, fr in zip(
                    rng.normal(0.5, 0.001, size=11), rng.normal(0.5, 0.01 + 0.01 * img_idx, size=11)
                )
            ]
        )
        for img_idx in range(30)
    ]
    _, lo_a, hi_a = study.bootstrap_amplification_ci(image_trials, resamples=200, seed=0)
    _, lo_b, hi_b = study.bootstrap_amplification_ci(image_trials, resamples=200, seed=1)
    assert (lo_a, hi_a) != (lo_b, hi_b)


def test_max_short_edge_bootstrap_ci_is_also_cluster_based():
    image_trials = [
        _image([_trial(fixed=0.5, free=0.5, move_pct=0.1 * i) for i in range(11)])
        for _ in range(15)
    ]
    point, lo, hi = study.bootstrap_max_short_edge_ci(image_trials, resamples=200, seed=0)
    assert lo <= hi
    assert point == study.max_short_edge_move_pct(image_trials)


# --------------------------------------------------------------------------
# E — report() gates the amplification check on the CI's upper bound, not
#     the point estimate (plan Sec4.4)
# --------------------------------------------------------------------------


def _fake_result(**overrides) -> dict:
    base = dict(
        asset_digest="0123456789abcdef",
        corpus_digest="f" * 64,
        confidence_threshold=0.5,
        nms_iou_threshold=0.4,
        seed=0,
        environment={},
        latency_ms={"scrfd": {"calls": 0}, "haar_baseline": {"calls": 0}},
        n_images=250,
        k_realisations=11,
        total_trials=2750,
        clean_detection_failures=0,
        faces_lost=0,
        rule_of_three_upper_bound_pct=0.11,
        max_short_edge_move_pct=0.5,
        max_short_edge_move_ci=(0.3, 0.7),
        fixed_crop_spread=0.001,
        free_crop_spread=0.005,
        amplification=8.0,
        amplification_ci=(6.0, 8.5),
    )
    base.update(overrides)
    return base


def test_report_fails_the_amplification_check_when_only_the_ci_upper_bound_crosses_10x():
    """Plan Sec4.4: 'a point estimate of 9.8x with a CI of (6.1, 14.3) is
    not a number a real cutover decision should treat as cleared' — a point
    estimate of 8.0 alone would pass 8.0 < 10, but a CI upper bound of 11.0
    must fail the check."""
    result = _fake_result(amplification=8.0, amplification_ci=(6.0, 11.0))
    assert study.report(result) is False


def test_report_passes_the_amplification_check_when_the_ci_upper_bound_also_clears_10x():
    result = _fake_result(amplification=8.0, amplification_ci=(6.0, 9.5))
    assert study.report(result) is True


def test_report_still_fails_on_any_face_loss_regardless_of_amplification():
    result = _fake_result(faces_lost=1)
    assert study.report(result) is False


def test_report_still_fails_below_the_200_image_floor():
    result = _fake_result(n_images=150)
    assert study.report(result) is False


# --------------------------------------------------------------------------
# F — measure()'s k_realisations default matches the original measurement's
#     own realisation count (plan Sec3.1)
# --------------------------------------------------------------------------


def test_measure_defaults_to_11_realisations():
    import inspect

    sig = inspect.signature(study.measure)
    assert sig.parameters["k_realisations"].default == 11


# --------------------------------------------------------------------------
# G — scrfd_detector_study.py stays outside tests/run_all.py's SUITES
#     (plan Sec5.3 — a study script, not a regression suite)
# --------------------------------------------------------------------------


def test_the_study_script_is_still_not_a_run_all_suite():
    import run_all

    suite_modules = [m.__name__ for _, m in run_all.SUITES]
    assert "scrfd_detector_study" not in suite_modules


# --------------------------------------------------------------------------
# H — version-impact axes untouched by this batch (plan Sec9)
# --------------------------------------------------------------------------


def test_no_identity_or_schema_axis_moved():
    from backend.config import STUB_ENGINE_VERSION
    from contracts.enums import CONTRACT_VERSION
    from evaluation.benchmark import RUN_SCHEMA_VERSION
    from evaluation.calibration_protocol import CALIBRATION_WITNESS_VERSION
    from evaluation.split_lock import LOCK_VERSION

    assert CONTRACT_VERSION == "1.0.0"
    assert RUN_SCHEMA_VERSION == "1.2.0"
    assert CALIBRATION_WITNESS_VERSION == "1.3.0"
    assert LOCK_VERSION == "1.1.0"
    assert STUB_ENGINE_VERSION == "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.f8244ad0.d.c9d4025c"


# --------------------------------------------------------------------------
# K — the study scores the face the shipped pipeline would score
# --------------------------------------------------------------------------


class _ScriptedDetector:
    """Returns pre-scripted boxes per call, in the (unordered) sequence given —
    the FaceDetector contract promises no order; SCRFD emits score order."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def detect(self, image, *, policy):
        boxes = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return list(boxes)


def _tile(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (120, 160, 3), dtype=np.uint8)


def test_the_study_orders_boxes_like_detect_faces():
    """``detect_faces`` sorts with ``face_order_key`` (largest first); the study
    must track that same face, not whatever the detector happened to emit
    first. Here a small face is emitted first on every call, and the large face
    — the one the pipeline scores — never moves."""
    from inference import preprocess as pre

    small, large = pre.FaceBox(5, 5, 20, 20), pre.FaceBox(60, 30, 50, 50)
    det = _ScriptedDetector([[small, large]] * 4)
    trials = study._collect_image_trials([_tile()], det, None, np.random.default_rng(0), 3)
    assert trials[0]["box0"] == [60, 30, 50, 50]
    assert all(t["short_edge_move_pct"] == 0.0 for t in trials[0]["trials"])


def test_a_top_face_switch_is_flagged_and_the_same_face_is_still_measured():
    """Noise makes the second face slightly larger, so the pipeline's top face
    switches identity. The documented statistic records that as a large box
    move (a real selection instability); the diagnostic flags the switch and
    measures the *original* face's own movement separately."""
    from inference import preprocess as pre

    a0, b0 = pre.FaceBox(10, 10, 50, 50), pre.FaceBox(90, 10, 48, 48)
    a1, b1 = pre.FaceBox(10, 10, 50, 49), pre.FaceBox(90, 10, 52, 52)
    det = _ScriptedDetector([[a0, b0], [a1, b1]])
    t = study._collect_image_trials([_tile()], det, None, np.random.default_rng(0), 1)[0]["trials"][0]
    assert t["top_switched"] is True and t["iou_top"] == 0.0
    assert abs(t["short_edge_move_pct"] - 4.0) < 1e-9  # 50 -> 52, the other face
    assert abs(t["matched_move_pct"] - 2.0) < 1e-9  # 50 -> 49, the same face
    assert t["matched_iou"] > 0.9


def test_box_stability_summary_separates_switches_from_jitter():
    from inference import preprocess as pre

    a0, b0 = pre.FaceBox(10, 10, 50, 50), pre.FaceBox(90, 10, 48, 48)
    a1, b1 = pre.FaceBox(10, 10, 50, 49), pre.FaceBox(90, 10, 52, 52)
    det = _ScriptedDetector([[a0, b0], [a0, b0], [a1, b1], [], [a0, b0]])
    trials = study._collect_image_trials([_tile()], det, None, np.random.default_rng(0), 4,
                                         names=["x.jpg"])
    s = study.box_stability_summary(trials)
    assert s["trials"] == 3 and s["lost"] == [["x.jpg", 2]]
    assert abs(s["top_face_switch_rate"] - 1 / 3) < 1e-9
    assert abs(s["matched_max_move_pct"] - 2.0) < 1e-9
    assert abs(s["top_frac_over_1pct"] - 1 / 3) < 1e-9


# --------------------------------------------------------------------------
# J — the pooled statistic's blind spot, and the within-image diagnostic
# --------------------------------------------------------------------------


def _offset_corpus(n_images: int, k: int, *, spread: float, fixed_sd: float, free_sd: float):
    """Images with very different baseline scores (``spread``) and a known
    noise response inside each image: true within-image amplification is
    ``free_sd / fixed_sd``."""
    rng = np.random.default_rng(11)
    out = []
    for _ in range(n_images):
        c = float(rng.normal(0.0, spread))
        fixed = rng.normal(c, fixed_sd, size=k)
        free = rng.normal(c, free_sd, size=k)
        out.append(_image([_trial(fixed=float(f), free=float(fr)) for f, fr in zip(fixed, free)]))
    return out


def test_the_pooled_statistic_is_blind_when_images_differ_more_than_noise_moves_them():
    """The documented statistic pools trials across images, so between-image
    score variance enters both spreads and drives the ratio toward 1 however
    unstable the detector is. Here the true within-image amplification is 50x
    and the pooled ratio still reads about 1x — which would pass R13."""
    trials = _offset_corpus(60, 11, spread=5.0, fixed_sd=0.001, free_sd=0.05)
    pooled = study.amplification_ratio(*study.pool_amplification_scores(trials))
    within = study.within_image_amplification(trials)
    assert pooled < 1.1, pooled
    assert 40.0 < within < 60.0, within


def test_the_between_image_share_exposes_the_blind_spot():
    trials = _offset_corpus(60, 11, spread=5.0, fixed_sd=0.001, free_sd=0.05)
    share = study.between_image_variance_share(trials)
    assert share["fixed"] > 0.999 and share["free"] > 0.99, share


def test_within_and_pooled_agree_when_there_is_one_image():
    """With a single fixture — the original 310x set-up — there is no
    between-image variance and the two statistics are the same number."""
    trials = _single_fixture_trials(k=11)
    pooled = study.amplification_ratio(*study.pool_amplification_scores(trials))
    assert abs(study.within_image_amplification(trials) - pooled) < 1e-9


def test_within_image_amplification_pairs_trials_and_skips_lost_ones():
    trials = [
        _image([_trial(fixed=0.0, free=0.0), _trial(fixed=0.002, free=0.2), _trial(lost=True)]),
        _image([_trial(fixed=5.0, free=5.0), _trial(fixed=5.002, free=5.2)]),
    ]
    assert abs(study.within_image_amplification(trials) - 100.0) < 1e-6


def test_within_image_amplification_has_a_cluster_bootstrap_ci():
    trials = _offset_corpus(30, 11, spread=1.0, fixed_sd=0.01, free_sd=0.03)
    point, lo, hi = study.bootstrap_within_image_amplification_ci(trials, resamples=200, seed=0)
    assert lo <= point <= hi and 2.0 < point < 4.5, (point, lo, hi)


# --------------------------------------------------------------------------
# I — end to end: measure() runs, records provenance, and a box that never
#     moves gives amplification exactly 1 (plumbing, not R13 evidence)
# --------------------------------------------------------------------------


def _synthetic_inputs(n_images: int) -> tuple[str, str]:
    """A constant-output SCRFD-shaped model (its box never moves, whatever the
    noise) and a folder of generated photographs. Proves the measurement path
    executes; says nothing about any real detector."""
    import tempfile

    import cv2

    from test_scrfd_detector import _write_fixture

    model = Path(tempfile.mkdtemp(prefix="dfp-r13-model-")) / "constant.onnx"
    _write_fixture(str(model), [(8, 18, 0.95, [1.0, 1.0, 1.0, 1.0])])
    photos = Path(tempfile.mkdtemp(prefix="dfp-r13-photos-"))
    for i in range(n_images):
        img = np.random.default_rng(i).integers(0, 256, (96, 96, 3), dtype=np.uint8)
        assert cv2.imwrite(str(photos / f"p{i:03d}.png"), img)
    return str(model), str(photos)


def test_measure_runs_end_to_end_and_records_provenance():
    from scrfd_corpus import manifest_digest

    from inference import preprocess as pre

    model, photos = _synthetic_inputs(4)
    result = study.measure(model, photos, seed=3, k_realisations=5)
    assert result["asset_digest"] == pre.scrfd_asset_digest(model)
    assert result["corpus_digest"] == manifest_digest(Path(photos))
    assert (result["confidence_threshold"], result["nms_iou_threshold"]) == (0.5, 0.4)
    assert result["seed"] == 3
    assert {"python", "numpy", "opencv", "onnxruntime"} <= set(result["environment"])
    assert result["n_images"] == 4 and result["total_trials"] == 20
    assert result["faces_lost"] == 0 and result["clean_detection_failures"] == 0
    # one clean + K noisy detections per image are timed; the warm-up is not
    assert result["latency_ms"]["scrfd"]["calls"] == 4 * (1 + 5)
    assert result["latency_ms"]["haar_baseline"]["calls"] == 4
    assert result["latency_ms"]["scrfd"]["median"] > 0


def test_a_box_that_never_moves_amplifies_by_exactly_one():
    """``A = spread(free) / spread(fixed)``: when the re-detected box equals the
    pre-noise box on every realisation, the two crops are the same pixels and
    the ratio is 1 — the floor a perfectly stable detector reaches."""
    model, photos = _synthetic_inputs(4)
    result = study.measure(model, photos, seed=0, k_realisations=5)
    assert result["max_short_edge_move_pct"] == 0.0
    assert result["amplification"] == 1.0
    assert result["amplification_ci"] == (1.0, 1.0)
    assert result["within_image_amplification"] == 1.0
    assert result["within_image_amplification_ci"] == (1.0, 1.0)


def test_the_evidence_record_is_written_when_requested():
    import io
    import json
    import os
    import tempfile
    from contextlib import redirect_stdout

    model, photos = _synthetic_inputs(3)
    out = Path(tempfile.mkdtemp(prefix="dfp-r13-evidence-")) / "run.json"
    saved = {k: os.environ.get(k) for k in ("DFP_SCRFD_MODEL_PATH", "DFP_SCRFD_STUDY_PHOTOS",
                                             "DFP_SCRFD_STUDY_JSON", "DFP_SCRFD_STUDY_SEED")}
    os.environ.update(DFP_SCRFD_MODEL_PATH=model, DFP_SCRFD_STUDY_PHOTOS=photos,
                      DFP_SCRFD_STUDY_JSON=str(out), DFP_SCRFD_STUDY_SEED="1")
    try:
        with redirect_stdout(io.StringIO()):
            assert study.main() == 0
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["seed"] == 1
    assert record["clears_criterion"] is False, "three images is below the 200-image floor"
    assert record["asset_digest"] and record["corpus_digest"]


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
