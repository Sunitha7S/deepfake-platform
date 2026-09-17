"""Process-wide inference engines: lazy, singleton, thread safe.

An engine holds loaded weights. Constructing one per job would reload them per
job, which on a real backend is seconds of latency and gigabytes of churn for
work that is identical every time. So engines are created once and reused for
the process lifetime.

**Lazy, not eager.** Import of this module must stay cheap: a worker that only
handles images should not load a video model, and a test that never analyses
anything should load neither. Construction happens on first request.

**Double-checked locking**, because workers are multi-threaded at the edges —
``asyncio.to_thread`` for blocking decode work means two threads can reach
``get_image_engine`` simultaneously. Without the lock they would each build an
engine and one would silently win, leaving the other's weights resident and
unreferenced.

A ``threading.Lock`` rather than an asyncio one: weight loading is blocking CPU
and disk work, and an asyncio lock held across a thread boundary protects
nothing.
"""

from __future__ import annotations

import threading

from inference import ImagePipeline, VideoPipeline, default_registry

_lock = threading.Lock()
_image: ImagePipeline | None = None
_video: VideoPipeline | None = None

IMAGE_MODEL = "stub-image"
VIDEO_MODEL = "stub-video"


def get_image_engine() -> ImagePipeline:
    """The process's image engine. Built on first call, reused thereafter."""
    global _image
    if _image is None:  # fast path, no lock
        with _lock:
            if _image is None:  # re-check under the lock
                _image = ImagePipeline(model_name=IMAGE_MODEL)
    return _image


def get_video_engine() -> VideoPipeline:
    """The process's video engine. Built on first call, reused thereafter."""
    global _video
    if _video is None:
        with _lock:
            if _video is None:
                _video = VideoPipeline(model_name=VIDEO_MODEL)
    return _video


def shutdown_engines() -> int:
    """Drop the engines and release every loaded model. Returns models unloaded.

    Synchronous on purpose. It is called from process shutdown and from test
    teardown, neither of which reliably has a running event loop — an async
    version would be unusable in exactly the places it is needed.
    """
    global _image, _video
    with _lock:
        _image = None
        _video = None
        registry = default_registry()
        loaded = list(registry.get_loaded_models())
        for name in loaded:
            registry.unload_model(name)
        return len(loaded)


def loaded_engines() -> dict[str, bool]:
    """Which engines currently exist. Diagnostics and tests."""
    return {"image": _image is not None, "video": _video is not None}


def engine_versions() -> dict[str, str]:
    """Effective version of each engine, without loading any weights.

    Reads specs rather than models, so calling it at startup costs nothing.
    """
    return {
        "image": ImagePipeline(model_name=IMAGE_MODEL).model_version,
        "video": VideoPipeline(model_name=VIDEO_MODEL).model_version,
    }


def verify_version_alignment(configured: str) -> None:
    """Raise unless the API's configured ``model_version`` matches the engines.

    The API and the engines each hold the model version, and they cannot import
    each other: the API process is deliberately free of cv2 and numpy, and
    ``inference`` is deliberately free of everything else. That leaves one
    duplicated string, and a duplicated string that nothing compares is a
    duplicated string that will drift.

    The drift is not loud when it happens. The result cache writes its
    content pointer under ``bundle.model.version`` and reads it under the API's
    ``settings.model_version``; when those differ, every lookup misses, every
    upload re-uploads, and every job re-runs. Nothing errors and no metric moves
    except cost. This function is what turns that into a startup failure.

    Call it from worker boot and from the test suite. It is cheap — specs only,
    no weights.
    """
    versions = engine_versions()
    mismatched = {name: v for name, v in versions.items() if v != configured}
    if not mismatched:
        return
    detail = ", ".join(f"{name} reports {v!r}" for name, v in sorted(mismatched.items()))
    raise RuntimeError(
        f"configured model_version {configured!r} does not match the engines: {detail}. "
        "The result cache keys on this string from both sides, so a mismatch "
        "silently disables it. Set DFP_MODEL_VERSION (and backend.config."
        "STUB_ENGINE_VERSION) to the value the engines report."
    )
