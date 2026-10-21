"""SCRFD Integration Program, Phase 3 — the R13 acceptance-criterion
measurement. Standalone, re-runnable, not part of ``tests/run_all.py``
(same convention as ``decode_study.py``/``crop_geometry_study.py``).

    python tests/scrfd_detector_study.py

THE QUESTION
------------
``REPRODUCIBILITY_AUDIT.md:1516-1518``, quoted exactly:

    SCRFD acceptance criterion. Under +/-4 LSB uniform noise on real
    photographs, across >= 200 images: no face lost, and the box short
    edge moving <= 1%. Amplification below 10x, from the measured 310x.

This is Phase 2's own vertical-slice deliverable per ``docs/
SCRFD_PHASE2_DESIGN.md`` Sec2.8/Sec8: the tool that answers the question, not
the answer itself. Running it for real against >=200 real photographs and
the actual fetched SCRFD asset is Phase 3's job (``docs/
SCRFD_PROGRAM_DESIGN.md`` Phase 3), not this batch's — this script reports
honestly, and exits without a verdict, when either input is missing rather
than fabricating a result.

METHODOLOGY (corrected per ``docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md``)
-----------------------------------------------------------------------
The amplification statistic is a *pooled, within-image, multi-realisation*
ratio-of-spreads (Sec4.1/Sec4.2 of the plan): for each image, ``k_realisations``
(>= 11, matching the original 310x measurement's own realisation count,
``docs/INFERENCE_DETERMINISM.md`` Sec5.1) independent noise draws are taken, and
every ``(image, realisation)`` trial's fixed-crop and free-crop scores are
pooled across the whole corpus before the ratio is computed — not one draw
per image with the ratio taken across images (that conflates cross-image
content diversity with noise-induced instability; ``docs/
SCRFD_PHASE3_MEASUREMENT_PLAN.md`` Sec2 is the full derivation of why the
prior, one-realisation-per-image version of this script did not compute the
criterion's own statistic). A cluster bootstrap (resampling whole images,
not individual trials, Sec4.3) gives a 95% CI on both the amplification
ratio and the max short-edge movement; the amplification check gates on the
CI's *upper bound*, not the point estimate (Sec4.4).

WHY THE SURROGATE, NOT STUBBACKEND
-----------------------------------
``StubBackend`` hashes crop bytes, so it would report near-100%
amplification for *any* crop perturbation — telling you nothing about the
detector. ``surrogate_logit`` (``tests/decode_study.py``) is "a smooth,
Lipschitz function of the pixels" — the same apparatus Batch 8/10 used to
produce R13's original 310x figure
(``tests/inference_determinism_study.py``). Reused here, not reinvented,
per ``docs/SCRFD_PHASE2_DESIGN.md`` Sec2.5/Sec2.8.

INPUTS THIS SCRIPT NEEDS AND DOES NOT SUPPLY
----------------------------------------------
* ``$DFP_SCRFD_MODEL_PATH`` — the fetched SCRFD asset (``docs/
  SCRFD_ASSET.md``; not fetched by this program, a licensing-gated human
  step).
* ``$DFP_SCRFD_STUDY_PHOTOS`` — a directory of >= 200 real photographs
  (JPEG/PNG), a second, smaller acquisition question
  (``docs/SCRFD_PHASE2_DESIGN.md`` Sec2.8 point 1) this script also does not
  resolve.
* ``$DFP_SCRFD_STUDY_SEED`` (optional, default ``0``) — the noise-generator
  seed. ``docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md`` Sec5.4 recommends running
  at least twice with different seeds and confirming both runs' CIs are
  consistent before treating a result as Phase-4-gating evidence.
* ``$DFP_SCRFD_STUDY_JSON`` (optional) — a path; the full result, including
  the asset digest, corpus manifest digest, thresholds, seed and library
  versions, is written there as the machine-readable half of the evidence
  record (``docs/SCRFD_PHASE4_PREREQUISITES_PLAN.md`` §C.4).

Build the photo directory with ``tests/scrfd_corpus.py`` so it carries a
``MANIFEST.json`` whose ``corpus_digest`` matches what this script records.

Neither of the first two is committed to this repository. Running this
script without both set prints what is missing and exits 0 (a missing input
is not a test failure) rather than exercising a code path with no real data
behind it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from decode_study import surrogate_logit  # noqa: E402
from scrfd_corpus import manifest_digest  # noqa: E402

from inference import preprocess as pre  # noqa: E402

ACCEPTANCE_NOISE_LSB = 4
ACCEPTANCE_MIN_IMAGES = 200
ACCEPTANCE_MAX_BOX_MOVEMENT_PCT = 1.0
ACCEPTANCE_MAX_AMPLIFICATION = 10.0
MEASURED_BASELINE_AMPLIFICATION = 310.0

# docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md Sec3.1/Sec4.1: matches the original
# measurement's own realisation count, so the pooled ratio is an
# apples-to-apples comparison to "310x", not merely a differently-shaped
# number that also happens to be called "amplification".
DEFAULT_K_REALISATIONS = 11

# docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md Sec4.3: reuses
# evaluation/thresholds.py's bootstrap_threshold_ci defaults exactly.
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_BOOTSTRAP_ALPHA = 0.05


def head(title: str) -> None:
    print(f"\n{'=' * 74}")
    print(title)
    print(f"{'=' * 74}")


def _resolve_inputs() -> tuple[str | None, str | None]:
    model_path = os.environ.get("DFP_SCRFD_MODEL_PATH")
    if model_path and not Path(model_path).is_file():
        model_path = None
    photo_dir = os.environ.get("DFP_SCRFD_STUDY_PHOTOS")
    if photo_dir and not Path(photo_dir).is_dir():
        photo_dir = None
    return model_path, photo_dir


def _load_photos(photo_dir: str) -> list[np.ndarray]:
    import cv2

    paths = sorted(
        p
        for p in Path(photo_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    images = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return images


def _policy_for(model_path: str) -> pre.FaceSelectionPolicy:
    from dataclasses import replace

    return replace(
        pre.default_face_selection(),
        detector="scrfd-10g-kps",
        detector_asset=pre.scrfd_asset_digest(model_path),
        confidence_threshold=0.5,
        nms_iou_threshold=0.4,
    )


def _crop_for_box(image: np.ndarray, box) -> np.ndarray:
    x0, y0 = max(0, box.x), max(0, box.y)
    x1, y1 = min(image.shape[1], box.x + box.width), min(image.shape[0], box.y + box.height)
    if x1 <= x0 or y1 <= y0:
        return image
    return image[y0:y1, x0:x1]


def _collect_image_trials(
    images: list[np.ndarray],
    detector,
    policy: pre.FaceSelectionPolicy,
    rng: np.random.Generator,
    k_realisations: int,
) -> list[dict]:
    """Runs the acceptance-criterion procedure (docs/
    SCRFD_PHASE3_MEASUREMENT_PLAN.md Sec4.1) once per ``(image, realisation)``
    pair. Returns one dict per input image, each carrying its own list of
    per-realisation trial dicts — kept grouped by image (not flattened) so
    callers can cluster-bootstrap by image (Sec4.3) rather than by trial.

    Each image dict: ``{"box0_present": bool, "clean_detection_failure":
    bool, "trials": [{"lost": bool, "short_edge_move_pct": float | None,
    "fixed_logit": float | None, "free_logit": float | None}, ...]}``.
    """
    image_trials: list[dict] = []
    for image in images:
        boxes_before = detector.detect(image, policy=policy)
        if not boxes_before:
            # Sec4.1 step 1: excluded from the amplification statistic, but
            # tracked as its own reportable fact — a clean image the shipped
            # detector cannot find a face in at all — not conflated with a
            # noise-induced loss.
            image_trials.append(
                {"box0_present": False, "clean_detection_failure": True, "trials": []}
            )
            continue
        box = boxes_before[0]
        short_edge_before = min(box.width, box.height)

        trials: list[dict] = []
        for _ in range(k_realisations):
            noise = rng.integers(
                -ACCEPTANCE_NOISE_LSB, ACCEPTANCE_NOISE_LSB + 1, size=image.shape, dtype=np.int16
            )
            noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            boxes_after = detector.detect(noisy, policy=policy)
            if not boxes_after:
                trials.append(
                    {
                        "lost": True,
                        "short_edge_move_pct": None,
                        "fixed_logit": None,
                        "free_logit": None,
                    }
                )
                continue
            moved_box = boxes_after[0]
            short_edge_after = min(moved_box.width, moved_box.height)
            move_pct = (
                100.0 * abs(short_edge_after - short_edge_before) / short_edge_before
                if short_edge_before > 0
                else None
            )
            trials.append(
                {
                    "lost": False,
                    "short_edge_move_pct": move_pct,
                    "fixed_logit": surrogate_logit(_crop_for_box(noisy, box)),
                    "free_logit": surrogate_logit(_crop_for_box(noisy, moved_box)),
                }
            )
        image_trials.append(
            {"box0_present": True, "clean_detection_failure": False, "trials": trials}
        )
    return image_trials


def pool_amplification_scores(image_trials: list[dict]) -> tuple[list[float], list[float]]:
    """``(fixed_scores, free_scores)``, pooled over every ``(i, k)`` trial
    that has a computable score (docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md
    Sec4.2). A realisation that lost the face contributes to neither pool for
    that trial only — not imputed."""
    fixed_scores: list[float] = []
    free_scores: list[float] = []
    for img in image_trials:
        for trial in img["trials"]:
            if trial["fixed_logit"] is not None:
                fixed_scores.append(trial["fixed_logit"])
            if trial["free_logit"] is not None:
                free_scores.append(trial["free_logit"])
    return fixed_scores, free_scores


def amplification_ratio(fixed_scores: list[float], free_scores: list[float]) -> float:
    """``std(free) / std(fixed)`` — the same ratio-of-spreads formula the
    original 310x measurement used (docs/INFERENCE_DETERMINISM.md Sec5A(c)),
    applied to a pooled multi-image, multi-realisation sample instead of one
    fixture's 11 realisations."""
    fixed_spread = float(np.std(fixed_scores)) if fixed_scores else 0.0
    free_spread = float(np.std(free_scores)) if free_scores else 0.0
    return (free_spread / fixed_spread) if fixed_spread > 0 else float("nan")


def max_short_edge_move_pct(image_trials: list[dict]) -> float:
    """``max`` over every ``(i, k)`` trial of the short-edge movement percent
    (docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md Sec4.4)."""
    moves = [
        trial["short_edge_move_pct"]
        for img in image_trials
        for trial in img["trials"]
        if trial["short_edge_move_pct"] is not None
    ]
    return max(moves) if moves else 0.0


def _cluster_bootstrap_ci(
    image_trials: list[dict],
    statistic_fn,
    *,
    resamples: int,
    alpha: float,
    seed: int,
) -> tuple[float, float, float]:
    """Resamples whole images with replacement (each resampled image brings
    its full set of realisations, not a re-drawn subset), recomputes
    ``statistic_fn`` on the resample, and returns ``(point, lo, hi)`` at the
    ``alpha/2`` and ``1 - alpha/2`` percentiles — the same shape
    ``evaluation/thresholds.bootstrap_threshold_ci`` returns, adapted to
    resample images rather than individual observations because
    realisations within one image are correlated (docs/
    SCRFD_PHASE3_MEASUREMENT_PLAN.md Sec4.3)."""
    point = statistic_fn(image_trials)
    rng = np.random.default_rng(seed)
    n = len(image_trials)
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        resampled = [image_trials[j] for j in idx]
        draws[i] = statistic_fn(resampled)
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(point), float(lo), float(hi)


def bootstrap_amplification_ci(
    image_trials: list[dict],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(point, lo, hi)`` cluster-bootstrap CI on the pooled amplification
    ratio (docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md Sec4.3)."""

    def statistic(trials: list[dict]) -> float:
        return amplification_ratio(*pool_amplification_scores(trials))

    return _cluster_bootstrap_ci(image_trials, statistic, resamples=resamples, alpha=alpha, seed=seed)


def bootstrap_max_short_edge_ci(
    image_trials: list[dict],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(point, lo, hi)`` cluster-bootstrap CI on the max short-edge-move
    statistic — "the same cluster-bootstrap CI applies to the max-short-edge-
    movement statistic for the same correlation reason" (docs/
    SCRFD_PHASE3_MEASUREMENT_PLAN.md Sec4.3)."""
    return _cluster_bootstrap_ci(
        image_trials, max_short_edge_move_pct, resamples=resamples, alpha=alpha, seed=seed
    )


def face_loss_summary(image_trials: list[dict]) -> dict:
    """Trial-level loss accounting over N x K trials, the clean-image
    detection-failure count tracked separately (Sec4.1 step 1), and the
    rule-of-three 95% upper bound on the true per-trial loss rate (Sec3.1) —
    "reported alongside the zero, not left implicit"."""
    total_trials = sum(len(img["trials"]) for img in image_trials)
    lost_trials = sum(1 for img in image_trials for trial in img["trials"] if trial["lost"])
    clean_detection_failures = sum(1 for img in image_trials if img["clean_detection_failure"])
    rule_of_three_upper_bound_pct = (300.0 / total_trials) if total_trials else float("nan")
    return {
        "total_trials": total_trials,
        "lost_trials": lost_trials,
        "clean_detection_failures": clean_detection_failures,
        "rule_of_three_upper_bound_pct": rule_of_three_upper_bound_pct,
    }


def _environment() -> dict:
    import platform

    import cv2
    import onnxruntime

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "onnxruntime": onnxruntime.__version__,
    }


def measure(
    model_path: str,
    photo_dir: str,
    *,
    seed: int = 0,
    k_realisations: int = DEFAULT_K_REALISATIONS,
) -> dict:
    """Runs the acceptance-criterion procedure (docs/
    SCRFD_PHASE3_MEASUREMENT_PLAN.md Sec4.1-Sec4.3) and returns the measured
    figures, pooled across every image and realisation and bootstrapped for
    a 95% CI. Does not itself decide pass/fail against a hardcoded number
    some caller cannot see — ``report`` does that, against the criterion
    quoted verbatim above."""
    rng = np.random.default_rng(seed)
    detector = pre.SCRFDDetector(model_path)
    policy = _policy_for(model_path)
    images = _load_photos(photo_dir)

    image_trials = _collect_image_trials(images, detector, policy, rng, k_realisations)

    fixed_scores, free_scores = pool_amplification_scores(image_trials)
    amp_point, amp_lo, amp_hi = bootstrap_amplification_ci(image_trials, seed=seed)
    edge_point, edge_lo, edge_hi = bootstrap_max_short_edge_ci(image_trials, seed=seed)
    losses = face_loss_summary(image_trials)

    return {
        # Provenance: which asset, which corpus, which thresholds, which stack.
        # A result without these is not comparable to any other result
        # (docs/SCRFD_PHASE4_PREREQUISITES_PLAN.md §A.5, §B.5, §C.2, §C.4).
        "asset_digest": pre.scrfd_asset_digest(model_path),
        "corpus_digest": manifest_digest(Path(photo_dir)),
        "confidence_threshold": policy.confidence_threshold,
        "nms_iou_threshold": policy.nms_iou_threshold,
        "seed": seed,
        "environment": _environment(),
        "n_images": len(images),
        "k_realisations": k_realisations,
        "total_trials": losses["total_trials"],
        "clean_detection_failures": losses["clean_detection_failures"],
        "faces_lost": losses["lost_trials"],
        "rule_of_three_upper_bound_pct": losses["rule_of_three_upper_bound_pct"],
        "max_short_edge_move_pct": edge_point,
        "max_short_edge_move_ci": (edge_lo, edge_hi),
        "fixed_crop_spread": float(np.std(fixed_scores)) if fixed_scores else 0.0,
        "free_crop_spread": float(np.std(free_scores)) if free_scores else 0.0,
        "amplification": amp_point,
        "amplification_ci": (amp_lo, amp_hi),
    }


def report(result: dict) -> bool:
    """Prints the measurement against the acceptance criterion, verbatim.
    Returns whether it clears the bar — the go/no-go input to Phase 4
    (docs/SCRFD_PHASE2_DESIGN.md Sec9's abort condition), not itself a
    decision to proceed."""
    head("R13 acceptance criterion")
    print(
        "  'Under +/-4 LSB uniform noise on real photographs, across >= 200 "
        "images: no face lost, and the box short edge moving <= 1%. "
        "Amplification below 10x, from the measured 310x.'"
    )
    amp_lo, amp_hi = result["amplification_ci"]
    edge_lo, edge_hi = result["max_short_edge_move_ci"]
    print(f"\n  asset digest               : {result['asset_digest']}")
    print(f"  corpus digest              : {result['corpus_digest']}")
    print(
        f"  thresholds (conf / nms)    : {result['confidence_threshold']} / "
        f"{result['nms_iou_threshold']}"
    )
    print(f"  seed                       : {result['seed']}")
    print(f"\n  images measured           : {result['n_images']}")
    print(f"  realisations per image     : {result['k_realisations']}")
    print(f"  total trials (N x K)       : {result['total_trials']}")
    print(
        f"  clean detection failures   : {result['clean_detection_failures']} "
        "(box_0 not found on clean pixels; excluded from amplification, tracked separately)"
    )
    print(f"  faces lost under noise     : {result['faces_lost']} of {result['total_trials']} trials")
    print(
        f"  rule-of-three upper bound  : {result['rule_of_three_upper_bound_pct']:.3f}% "
        "(95% bound on the true per-trial loss rate, given the observed losses)"
    )
    print(
        f"  max short-edge move        : {result['max_short_edge_move_pct']:.3f}%  "
        f"(95% bootstrap CI [{edge_lo:.3f}%, {edge_hi:.3f}%])"
    )
    print(
        f"  amplification              : {result['amplification']:.2f}x "
        f"(was {MEASURED_BASELINE_AMPLIFICATION}x)"
    )
    print(f"    95% bootstrap CI          : [{amp_lo:.2f}x, {amp_hi:.2f}x]")

    checks = {
        f">= {ACCEPTANCE_MIN_IMAGES} images": result["n_images"] >= ACCEPTANCE_MIN_IMAGES,
        "no face lost": result["faces_lost"] == 0,
        "short edge move <= 1%": (
            result["max_short_edge_move_pct"] <= ACCEPTANCE_MAX_BOX_MOVEMENT_PCT
        ),
        # docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md Sec4.4: gates on the CI's
        # upper bound, a strengthening of the literal criterion text — a
        # point estimate under 10x with an upper CI bound over it is not a
        # number a cutover decision should treat as cleared.
        "amplification CI upper bound < 10x": amp_hi < ACCEPTANCE_MAX_AMPLIFICATION,
    }
    print()
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return all(checks.values())


def main() -> int:
    model_path, photo_dir = _resolve_inputs()
    head("SCRFD Phase 3 study — input check")
    print(f"  DFP_SCRFD_MODEL_PATH   : {model_path or '(not set / file missing)'}")
    print(f"  DFP_SCRFD_STUDY_PHOTOS : {photo_dir or '(not set / directory missing)'}")

    if not model_path or not photo_dir:
        print(
            "\n  Missing input(s) above. This is Phase 2's tool, not Phase 3's "
            "run — see docs/SCRFD_ASSET.md for the fetch procedure and "
            "docs/SCRFD_PHASE2_DESIGN.md Sec2.8 for the photo-corpus question. "
            "Exiting without a verdict (not a failure)."
        )
        return 0

    seed = int(os.environ.get("DFP_SCRFD_STUDY_SEED", "0"))
    result = measure(model_path, photo_dir, seed=seed)
    cleared = report(result)
    evidence_path = os.environ.get("DFP_SCRFD_STUDY_JSON")
    if evidence_path:
        import json

        Path(evidence_path).write_text(
            json.dumps({**result, "clears_criterion": cleared}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    head("Result")
    print(f"  seed used: {seed}")
    if evidence_path:
        print(f"  machine-readable record: {evidence_path}")
    print(
        "  Per docs/SCRFD_PHASE3_MEASUREMENT_PLAN.md Sec5.4: run at least twice "
        "with different $DFP_SCRFD_STUDY_SEED values and confirm both runs' "
        "CIs are consistent before treating this as Phase-4-gating evidence."
    )
    print(
        "  CLEARS the acceptance criterion"
        if cleared
        else "  DOES NOT CLEAR the acceptance criterion"
    )
    print(
        "  Per docs/SCRFD_PHASE2_DESIGN.md Sec9: if this does not clear, Phase 4 "
        "does not proceed on structural argument alone — a failed measurement "
        "is a complete, reportable result, not a reason to lower the bar."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
