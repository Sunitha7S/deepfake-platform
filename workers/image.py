"""Image handler. Delegates analysis to the inference layer.

Deliberately thin. Everything this used to do — decoding, gating, scoring,
fusion, bundle construction — now lives in ``inference``, where it is tested
against 149 of its own tests and can be replaced with real weights without a
worker ever noticing.

What remains here is the two things a handler is actually for: satisfying the
``JobHandler`` protocol, and translating one layer's error vocabulary into the
other's. There is no scoring logic in this module, and a regression test asserts
that none returns.

Progress is passed straight through. The inference ``ProgressFn`` signature
matches ``workers.ProgressReporter`` exactly, so forwarding is identity rather
than a wrapper — a wrapper is where extra or dropped events come from, and the
batching guarantee the video pipeline makes only holds if nothing sits in
between.
"""

from __future__ import annotations

from contracts.enums import MediaKind
from contracts.evidence import EvidenceBundle
from workers.engine_registry import get_image_engine
from workers.inference_adapter import translated
from workers.protocol import ProgressReporter
from workers.types import JobRequest, PermanentError


class ImageHandler:
    """``JobHandler`` for images, backed by ``inference.ImagePipeline``."""

    media_kind = MediaKind.IMAGE.value
    name = "image"

    async def handle(
        self, request: JobRequest, progress: ProgressReporter
    ) -> EvidenceBundle:
        if request.spec.media_kind is not MediaKind.IMAGE:
            raise PermanentError(
                "This handler analyses images; the job declares "
                f"{request.spec.media_kind.value}."
            )
        with translated():
            return await get_image_engine().analyze_image(
                request.spec, request.media, progress
            )
