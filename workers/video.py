"""Video handler. Delegates analysis to the inference layer.

Thin for the same reasons as the image handler, with one property worth naming:
**progress is forwarded unmodified**. The video pipeline reports once per batch
of eight frames — six calls for a forty-eight-frame clip — and that guarantee is
the pipeline's, not this module's. Wrapping the callback here to add a stage
label, rescale a fraction or emit a completion event would break it silently,
and the only symptom would be a lease renewed more often than necessary on every
video job.
"""

from __future__ import annotations

from contracts.enums import MediaKind
from contracts.evidence import EvidenceBundle
from workers.engine_registry import get_video_engine
from workers.inference_adapter import translated
from workers.protocol import ProgressReporter
from workers.types import JobRequest, PermanentError


class VideoHandler:
    """``JobHandler`` for video, backed by ``inference.VideoPipeline``."""

    media_kind = MediaKind.VIDEO.value
    name = "video"

    async def handle(
        self, request: JobRequest, progress: ProgressReporter
    ) -> EvidenceBundle:
        if request.spec.media_kind is not MediaKind.VIDEO:
            raise PermanentError(
                "This handler analyses video; the job declares "
                f"{request.spec.media_kind.value}."
            )
        with translated():
            return await get_video_engine().analyze_video(
                request.spec, request.media, progress
            )
