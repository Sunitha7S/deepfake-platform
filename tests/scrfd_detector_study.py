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
import time
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
    return [image for _, image in _load_named_photos(photo_dir)]


def _load_named_photos(photo_dir: str) -> list[tuple[str, np.ndarray]]:
    import cv2

    paths = sorted(
        p
        for p in Path(photo_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    out = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        out.append((p.name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return out


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


def _ordered(boxes) -> list:
    """The order ``detect_faces`` imposes: ``face_order_key``, largest first.
    ``FaceDetector.detect`` promises no order (SCRFD emits confidence order), so
    without this the study would track the most *confident* face rather than
    the face the shipped pipeline scores."""
    return sorted(boxes, key=pre.face_order_key)


def _iou(a, b) -> float:
    ix = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    iy = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union else 0.0


def _move_pct(before, after) -> float | None:
    edge = min(before.width, before.height)
    return 100.0 * abs(min(after.width, after.height) - edge) / edge if edge > 0 else None


def _collect_image_trials(
    images: list[np.ndarray],
    detector,
    policy: pre.FaceSelectionPolicy,
    rng: np.random.Generator,
    k_realisations: int,
    *,
    names: list[str] | None = None,
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
    for idx, image in enumerate(images):
        name = names[idx] if names else f"#{idx}"
        boxes_before = _ordered(detector.detect(image, policy=policy))
        if not boxes_before:
            # Sec4.1 step 1: excluded from the amplification statistic, but
            # tracked as its own reportable fact — a clean image the shipped
            # detector cannot find a face in at all — not conflated with a
            # noise-induced loss.
            image_trials.append(
                {"name": name, "box0_present": False, "clean_detection_failure": True,
                 "trials": []}
            )
            continue
        box = boxes_before[0]

        trials: list[dict] = []
        for _ in range(k_realisations):
            noise = rng.integers(
                -ACCEPTANCE_NOISE_LSB, ACCEPTANCE_NOISE_LSB + 1, size=image.shape, dtype=np.int16
            )
            noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            boxes_after = _ordered(detector.detect(noisy, policy=policy))
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
            # Diagnostics: did the pipeline's top face stay the same face, and
            # how far did the *original* face's own box move?
            matched = max(boxes_after, key=lambda b: _iou(box, b))
            iou_top = _iou(box, moved_box)
            trials.append(
                {
                    "lost": False,
                    "short_edge_move_pct": _move_pct(box, moved_box),
                    "fixed_logit": surrogate_logit(_crop_for_box(noisy, box)),
                    "free_logit": surrogate_logit(_crop_for_box(noisy, moved_box)),
                    "box": [moved_box.x, moved_box.y, moved_box.width, moved_box.height],
                    "iou_top": iou_top,
                    "top_switched": iou_top < 0.5,
                    "matched_iou": _iou(box, matched),
                    "matched_move_pct": _move_pct(box, matched),
                }
            )
        image_trials.append(
            {
                "name": name,
                "box0_present": True,
                "clean_detection_failure": False,
                "box0": [box.x, box.y, box.width, box.height],
                "faces_clean": len(boxes_before),
                "trials": trials,
            }
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


def _paired_groups(image_trials: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per image, the (fixed, free) scores of realisations where both exist —
    the same noisy pixels under two crops."""
    groups = []
    for img in image_trials:
        pairs = [
            (t["fixed_logit"], t["free_logit"])
            for t in img["trials"]
            if t["fixed_logit"] is not None and t["free_logit"] is not None
        ]
        if len(pairs) >= 2:
            arr = np.asarray(pairs, dtype=np.float64)
            groups.append((arr[:, 0], arr[:, 1]))
    return groups


def within_image_amplification(image_trials: list[dict]) -> float:
    """Diagnostic, not the documented gate: the noise-induced spread around each
    image's *own* mean, pooled across images (a one-way ANOVA within-group
    standard deviation), free crop over fixed crop.

    The documented statistic (:func:`amplification_ratio` over
    :func:`pool_amplification_scores`) pools raw scores across images, so
    between-image score variance adds to both spreads and pulls the ratio toward
    1 regardless of detector stability. The original 310x was measured on one
    fixture, where that term is zero; this statistic removes it, so it is the
    like-for-like generalisation. With one image the two coincide."""
    ss_fixed = ss_free = 0.0
    for fixed, free in _paired_groups(image_trials):
        ss_fixed += float(((fixed - fixed.mean()) ** 2).sum())
        ss_free += float(((free - free.mean()) ** 2).sum())
    return float(np.sqrt(ss_free / ss_fixed)) if ss_fixed > 0 else float("nan")


def between_image_variance_share(image_trials: list[dict]) -> dict:
    """Fraction of each pooled spread that comes from images differing from one
    another rather than from noise. Near 1 means the documented pooled ratio
    cannot see detector instability on this corpus."""
    out = {}
    groups = _paired_groups(image_trials)
    for idx, name in ((0, "fixed"), (1, "free")):
        if not groups:
            out[name] = float("nan")
            continue
        allv = np.concatenate([g[idx] for g in groups])
        total = float(((allv - allv.mean()) ** 2).sum())
        within = sum(float(((g[idx] - g[idx].mean()) ** 2).sum()) for g in groups)
        out[name] = 1.0 - within / total if total > 0 else float("nan")
    return out


def bootstrap_within_image_amplification_ci(
    image_trials: list[dict],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Same cluster bootstrap as the gate statistic, over images."""
    return _cluster_bootstrap_ci(
        image_trials, within_image_amplification, resamples=resamples, alpha=alpha, seed=seed
    )


def box_stability_summary(image_trials: list[dict]) -> dict:
    """Diagnostic breakdown of the box checks, not the documented gate.

    Separates three different failures the gate statistics fold together: the
    face lost outright; the pipeline's top-ranked face *switching identity*
    (a selection instability — the verdict would describe another person);
    and the *same* face's box jittering (a localisation instability — R13's
    original mechanism). "Same face" means best-IoU >= 0.5 to the pre-noise
    box."""
    done = [(img, k, t) for img in image_trials for k, t in enumerate(img["trials"]) if not t["lost"]]
    lost = [[img["name"], k] for img in image_trials for k, t in enumerate(img["trials"]) if t["lost"]]
    top = np.array([t["short_edge_move_pct"] for _, _, t in done if t["short_edge_move_pct"] is not None])
    same = [t for _, _, t in done if t["matched_iou"] >= 0.5]
    matched = np.array([t["matched_move_pct"] for t in same if t["matched_move_pct"] is not None])

    def pct(arr, q):
        return float(np.percentile(arr, q)) if arr.size else 0.0

    return {
        "trials": len(done),
        "lost": lost,
        "clean_detection_failures": [img["name"] for img in image_trials if img["clean_detection_failure"]],
        "top_face_switch_rate": (sum(t["top_switched"] for _, _, t in done) / len(done)) if done else 0.0,
        "switched": [[img["name"], k] for img, k, t in done if t["top_switched"]],
        "original_face_missing_rate": (1 - len(same) / len(done)) if done else 0.0,
        "top_frac_over_1pct": float(np.mean(top > 1.0)) if top.size else 0.0,
        "top_move_pct": {"median": pct(top, 50), "p95": pct(top, 95), "p99": pct(top, 99),
                         "max": float(top.max()) if top.size else 0.0},
        "matched_frac_over_1pct": float(np.mean(matched > 1.0)) if matched.size else 0.0,
        "matched_move_pct": {"median": pct(matched, 50), "p95": pct(matched, 95),
                             "p99": pct(matched, 99)},
        "matched_max_move_pct": float(matched.max()) if matched.size else 0.0,
    }


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


class _TimedDetector:
    """Wall-clock per ``detect`` call, otherwise transparent."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.ms: list[float] = []

    def detect(self, image: np.ndarray, *, policy):
        start = time.perf_counter()
        boxes = self.inner.detect(image, policy=policy)
        self.ms.append((time.perf_counter() - start) * 1000.0)
        return boxes


def _haar_latency_ms(images: list[np.ndarray]) -> list[float]:
    """The shipped Haar detector on the same clean images, for comparison."""
    haar, policy = pre.HaarDetector(), pre.default_face_selection()
    if images:
        haar.detect(images[0], policy=policy)  # cascade load is not latency
    out = []
    for image in images:
        start = time.perf_counter()
        haar.detect(image, policy=policy)
        out.append((time.perf_counter() - start) * 1000.0)
    return out


def _latency_summary(ms: list[float]) -> dict:
    if not ms:
        return {"calls": 0}
    arr = np.asarray(ms)
    return {
        "calls": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
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
    policy = _policy_for(model_path)
    named = _load_named_photos(photo_dir)
    names, images = [n for n, _ in named], [i for _, i in named]
    detector = _TimedDetector(pre.SCRFDDetector(model_path))
    if images:
        detector.inner.detect(images[0], policy=policy)  # session creation is not latency

    image_trials = _collect_image_trials(
        images, detector, policy, rng, k_realisations, names=names
    )
    haar_ms = _haar_latency_ms(images)

    fixed_scores, free_scores = pool_amplification_scores(image_trials)
    amp_point, amp_lo, amp_hi = bootstrap_amplification_ci(image_trials, seed=seed)
    edge_point, edge_lo, edge_hi = bootstrap_max_short_edge_ci(image_trials, seed=seed)
    within_point, within_lo, within_hi = bootstrap_within_image_amplification_ci(
        image_trials, seed=seed
    )
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
        # docs/SCRFD_PROGRAM_DESIGN.md §6 point 4 / prerequisites plan §D.1
        # item 10: SCRFD's forward-pass cost against Haar's cascade, same
        # machine, same images. Not part of the R13 pass/fail.
        "latency_ms": {
            "scrfd": _latency_summary(detector.ms),
            "haar_baseline": _latency_summary(haar_ms),
        },
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
        # Diagnostics, not the documented gate (see within_image_amplification).
        "within_image_amplification": within_point,
        "within_image_amplification_ci": (within_lo, within_hi),
        "between_image_variance_share": between_image_variance_share(image_trials),
        "box_stability": box_stability_summary(image_trials),
        # The per-trial record, so every figure above can be re-derived and
        # every failure traced to an image. Written to the JSON evidence only.
        "image_trials": image_trials,
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
    if "within_image_amplification" in result:
        w_lo, w_hi = result["within_image_amplification_ci"]
        share = result["between_image_variance_share"]
        print(
            f"  diagnostic: within-image   : {result['within_image_amplification']:.2f}x "
            f"(95% CI [{w_lo:.2f}x, {w_hi:.2f}x]) — between-image share of the pooled "
            f"spread: fixed {share['fixed']:.3f}, free {share['free']:.3f}"
        )
    if "box_stability" in result:
        bs = result["box_stability"]
        print(
            f"  diagnostic: top-face switch: {100 * bs['top_face_switch_rate']:.2f}% of trials "
            f"({len(bs['switched'])}); original face missing {100 * bs['original_face_missing_rate']:.2f}%"
        )
        print(
            f"  diagnostic: same-face move : median {bs['matched_move_pct']['median']:.2f}%, "
            f"p99 {bs['matched_move_pct']['p99']:.2f}%, max {bs['matched_max_move_pct']:.2f}%, "
            f"{100 * bs['matched_frac_over_1pct']:.1f}% of trials over 1%"
        )
    for name, stats in result["latency_ms"].items():
        if stats.get("calls"):
            print(
                f"  latency {name:<18} : median {stats['median']:.1f} ms, "
                f"p95 {stats['p95']:.1f} ms over {stats['calls']} calls"
            )

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
