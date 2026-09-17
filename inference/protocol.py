"""Inference protocols and the error taxonomy.

This package analyses media. It knows nothing about queues, storage, HTTP or
job orchestration, and its dependency list says so: ``contracts``, numpy, torch,
cv2, onnxruntime, stdlib. Nothing else.

Two boundary decisions carry that:

**Engines receive bytes, never a storage key.** A key would drag ``storage`` in.
Callers resolve the bytes; inference analyses them. The only shared vocabulary
is ``contracts`` — a ``JobSpec`` in, an ``EvidenceBundle`` out.

**Errors are inference's own.** Importing ``workers.types.PermanentError`` would
invert the dependency and make this package unusable outside a worker. Instead
it raises its own, and the worker maps them onto its retryable/permanent split —
which is where that judgement belongs, since only the caller knows whether a
retry is affordable.

The protocols are split rather than combined so that an image-only deployment
implements two of the three instead of raising ``NotImplementedError`` from a
method it can never perform.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contracts.enums import JobStage
from contracts.evidence import EvidenceBundle
from contracts.job import JobSpec


class InferenceError(Exception):
    """Base for everything this package raises."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        self.detail = detail
        super().__init__(message)


class UnreadableMedia(InferenceError):
    """The bytes could not be decoded as the declared media type.

    Permanent for that input: the same bytes will not decode next time. A caller
    that retries this is burning capacity on a certainty.
    """


class UnsupportedMedia(InferenceError):
    """Decodable, but not something this engine analyses — a 12-hour video, a
    1×1 image, a codec outside the allowlist."""


class ModelUnavailable(InferenceError):
    """Weights are missing, corrupt, or could not be loaded.

    Transient from the caller's view: the deployment is broken, not the input,
    and a retry after a fix succeeds without the user doing anything.
    """


class ProgressFn(Protocol):
    """How a long analysis reports liveness.

    Signature matches ``workers.ProgressReporter`` exactly so a worker's
    callable satisfies both without either package importing the other. That is
    the point of structural typing and the reason this is a Protocol rather than
    a shared base class.
    """

    async def __call__(self, fraction: float, *, stage: JobStage | None = None) -> None: ...


@runtime_checkable
class InferenceEngine(Protocol):
    """Lifecycle common to every engine."""

    model_version: str

    async def warmup(self) -> None:
        """Load weights and run a throwaway inference.

        Called at worker boot so the first real request does not pay for lazy
        initialisation. A cold first request on a video is the difference
        between a 6-second job and a 25-second one.
        """
        ...

    async def close(self) -> None:
        """Release weights. Safe to call twice."""
        ...


@runtime_checkable
class ImageInferenceEngine(Protocol):
    """Analyses a single image."""

    model_version: str

    async def analyze_image(
        self,
        spec: JobSpec,
        media: bytes,
        progress: ProgressFn | None = None,
    ) -> EvidenceBundle:
        """Return a contract-valid bundle for ``media``.

        Raises ``UnreadableMedia`` for bytes that will never decode and
        ``ModelUnavailable`` when the deployment cannot serve.

        Returns — never raises — for input that is merely unjudgeable: no face,
        a face below the size floor, quality under the gate. Those produce a
        bundle whose verdict is ``insufficient_evidence``, which is a successful
        analysis and must not be modelled as failure anywhere downstream.
        """
        ...


@runtime_checkable
class VideoInferenceEngine(Protocol):
    """Analyses a video clip."""

    model_version: str

    async def analyze_video(
        self,
        spec: JobSpec,
        media: bytes,
        progress: ProgressFn | None = None,
    ) -> EvidenceBundle:
        """As ``analyze_image``, plus a per-frame curve and suspect segments.

        ``progress`` is called on frame *batches*, never per frame. A 48-frame
        clip reporting 48 times turns a liveness signal into queue traffic, and
        on a Redis-backed lease that is 48 round trips for something that needs
        to fire a handful of times.
        """
        ...
