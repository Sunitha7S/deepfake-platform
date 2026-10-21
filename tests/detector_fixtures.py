"""The Haar detector configuration this suite's synthetic face fixtures need.

Every synthetic face in the suite — the ellipse tiles behind ``_face_image``,
``_face_tile``, ``face_image(seed)`` — was tuned until the Haar cascade finds
it. A CNN detector does not see them, and a synthetic ONNX graph with constant
outputs "finds" a face in a blank image, so neither can stand in for Haar in a
test whose assertions were derived from what Haar returns.

So a test whose subject is pipeline mechanics (gating, timing, the event loop,
bundle shape) or Haar's own behaviour (R14 ordering, R15 per-thread cascades)
passes :func:`haar_selection` explicitly instead of letting
``default_face_selection()`` pick the detector. While the shipped default *is*
Haar this is output-identical — ``tests/test_detector_pinning.py`` proves the
two policies have the same identity — and when the shipped detector changes
these tests keep testing what they were written to test.

Tests whose subject is the *shipped* identity (``.f.``, ``STUB_ENGINE_VERSION``,
engine alignment) deliberately keep following the default and are expected to
move with it. ``tests/test_detector_pinning.py`` holds the line between the two.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import preprocess as pre  # noqa: E402

HAAR_DETECTOR = "opencv-haar-frontalface-default"


def haar_selection() -> pre.FaceSelectionPolicy:
    """The shipped crop/ordering policy with Haar's detector fields pinned.

    Detector fields are stated, not inherited, so a change to the shipped
    detector cannot reach a test through here. Crop, ordering and
    interpolation fields still come from the shipped default — the pipeline
    behaviour downstream of detection is what these tests exercise.
    """
    return replace(
        pre.default_face_selection(),
        detector=HAAR_DETECTOR,
        detector_asset=pre.cascade_digest(),
        scale_factor=1.1,
        min_neighbors=5,
        detect_min_px=32,
        confidence_threshold=None,
        nms_iou_threshold=None,
    )
