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

Neither is committed to this repository. Running this script without both
set prints what is missing and exits 0 (a missing input is not a test
failure) rather than exercising a code path with no real data behind it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from decode_study import surrogate_logit  # noqa: E402

from inference import preprocess as pre  # noqa: E402

ACCEPTANCE_NOISE_LSB = 4
ACCEPTANCE_MIN_IMAGES = 200
ACCEPTANCE_MAX_BOX_MOVEMENT_PCT = 1.0
ACCEPTANCE_MAX_AMPLIFICATION = 10.0
MEASURED_BASELINE_AMPLIFICATION = 310.0


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


def measure(model_path: str, photo_dir: str, *, seed: int = 0) -> dict:
    """Runs the acceptance-criterion procedure (docs/SCRFD_PHASE2_DESIGN.md
    Sec2.8) and returns the measured figures. Does not itself decide pass/fail
    against a hardcoded number some caller cannot see — ``report`` does that,
    against the criterion quoted verbatim above."""
    rng = np.random.default_rng(seed)
    detector = pre.SCRFDDetector(model_path)
    policy = _policy_for(model_path)
    images = _load_photos(photo_dir)

    faces_lost = 0
    short_edge_moves_pct: list[float] = []
    fixed_crop_logits: list[float] = []
    free_crop_logits: list[float] = []

    for image in images:
        boxes_before = detector.detect(image, policy=policy)
        if not boxes_before:
            continue
        box = boxes_before[0]

        noise = rng.integers(
            -ACCEPTANCE_NOISE_LSB, ACCEPTANCE_NOISE_LSB + 1, size=image.shape, dtype=np.int16
        )
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        boxes_after = detector.detect(noisy, policy=policy)
        if not boxes_after:
            faces_lost += 1
            continue
        moved_box = boxes_after[0]

        short_edge_before = min(box.width, box.height)
        short_edge_after = min(moved_box.width, moved_box.height)
        if short_edge_before > 0:
            short_edge_moves_pct.append(
                100.0 * abs(short_edge_after - short_edge_before) / short_edge_before
            )

        fixed_crop_logits.append(surrogate_logit(_crop_for_box(noisy, box)))
        free_crop_logits.append(surrogate_logit(_crop_for_box(noisy, moved_box)))

    n = len(images)
    fixed_spread = float(np.std(fixed_crop_logits)) if fixed_crop_logits else 0.0
    free_spread = float(np.std(free_crop_logits)) if free_crop_logits else 0.0
    amplification = (free_spread / fixed_spread) if fixed_spread > 0 else float("nan")

    return {
        "n_images": n,
        "faces_lost": faces_lost,
        "max_short_edge_move_pct": max(short_edge_moves_pct) if short_edge_moves_pct else 0.0,
        "fixed_crop_spread": fixed_spread,
        "free_crop_spread": free_spread,
        "amplification": amplification,
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
    print(f"\n  images measured        : {result['n_images']}")
    print(f"  faces lost             : {result['faces_lost']}")
    print(f"  max short-edge move    : {result['max_short_edge_move_pct']:.3f}%")
    print(
        f"  amplification          : {result['amplification']:.2f}x "
        f"(was {MEASURED_BASELINE_AMPLIFICATION}x)"
    )

    checks = {
        f">= {ACCEPTANCE_MIN_IMAGES} images": result["n_images"] >= ACCEPTANCE_MIN_IMAGES,
        "no face lost": result["faces_lost"] == 0,
        "short edge move <= 1%": (
            result["max_short_edge_move_pct"] <= ACCEPTANCE_MAX_BOX_MOVEMENT_PCT
        ),
        "amplification < 10x": result["amplification"] < ACCEPTANCE_MAX_AMPLIFICATION,
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

    result = measure(model_path, photo_dir)
    cleared = report(result)
    head("Result")
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
