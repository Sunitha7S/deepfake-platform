"""Inference layer.

Analyses image and video bytes and returns contract-valid ``EvidenceBundle``
objects. Depends on ``contracts``, numpy, cv2 and (lazily) torch or
onnxruntime. It imports nothing from ``backend``, ``workers``, ``storage`` or
FastAPI, which is what lets it be tested — and replaced — on its own.

**Torch is never imported at module scope.** On this machine that import costs
15.5 seconds, and a package paying it on import would charge every test run,
worker boot and CI job for a library most of them never touch. It is imported
inside ``models.load_model`` and only for backends that need it; the stub
backend never touches it at all.

A worker swaps stub scoring for real analysis with one line and no orchestration
changes::

    engine = ImagePipeline()
    bundle = await engine.analyze_image(spec, media, progress)

Exports are the public surface only: protocols, pipelines, registry helpers and
calibration. Preprocessing and bundle construction are reachable through their
modules but deliberately absent from this namespace — importing
``build_image_bundle`` at a call site would put bundle construction outside the
one module that enforces its invariants.
"""

from inference.calibration import (
    CalibrationFit,
    abstention_band,
    fit_and_validate,
    fit_temperature,
    inverse_sigmoid,
    sigmoid,
    temperature_scale,
    threshold_at_fpr,
)
from inference.image_pipeline import ImagePipeline
from inference.models import (
    CalibrationSpec,
    LoadedModel,
    ModelRegistry,
    ModelSpec,
    default_registry,
    get_loaded_models,
    load_model,
    unload_model,
)
from inference.protocol import (
    ImageInferenceEngine,
    InferenceEngine,
    InferenceError,
    ModelUnavailable,
    ProgressFn,
    UnreadableMedia,
    UnsupportedMedia,
    VideoInferenceEngine,
)
from inference.video_pipeline import VideoPipeline

__all__ = [
    # protocols and errors
    "ImageInferenceEngine",
    "InferenceEngine",
    "InferenceError",
    "ModelUnavailable",
    "ProgressFn",
    "UnreadableMedia",
    "UnsupportedMedia",
    "VideoInferenceEngine",
    # pipelines
    "ImagePipeline",
    "VideoPipeline",
    # registry
    "CalibrationSpec",
    "LoadedModel",
    "ModelRegistry",
    "ModelSpec",
    "default_registry",
    "get_loaded_models",
    "load_model",
    "unload_model",
    # calibration
    "CalibrationFit",
    "abstention_band",
    "fit_and_validate",
    "fit_temperature",
    "inverse_sigmoid",
    "sigmoid",
    "temperature_scale",
    "threshold_at_fpr",
]
