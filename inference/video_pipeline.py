"""Video pipeline: bytes → frames → batches → inference → pooling → bundle.

**THERE IS NO FACE STAGE HERE, AND THAT IS R18.** This module samples frames,
resizes them whole to the model input and scores them. It does not detect, crop,
align, track or attribute faces, and it never has. Three consequences a reader
must not have to discover by grepping:

* Every bundle this produces carries ``faces=[]``, a null ``quality.face_px``,
  and null ``top_frames[].track_id``. Empty here means *nothing looked*, not
  *nothing was found* — the distinction the image path's ``NO_FACE_DETECTED``
  makes and this path cannot.
* ``NO_FACE_DETECTED`` and ``FACE_TOO_SMALL`` are unreachable from
  :meth:`VideoPipeline._abstain`, which emits only ``TOO_FEW_FRAMES``. **A clip
  containing no face at all receives a decided verdict** — measured on pure
  noise: ``manipulated`` at p_fake 0.979 with ``gate_passed=True``, where the
  image path would have abstained on the same content.
* ``JobStage.DETECTING`` and ``JobStage.TRACKING`` are budgeted in
  ``STAGE_WEIGHTS_VIDEO`` and emitted by nothing.

Batch 13 corrected the *claims* — the dangling ``track_id``, the ``frames_scored``
description, this docstring. It deliberately did not add a face stage: that is
the SCRFD migration, and building a Haar-based one here would inherit R13's 310×
crop instability on every one of 48 frames. See ``docs/VIDEO_FACE_SEMANTICS.md``.

Progress fires **once per completed batch and never per frame**. Forty-eight
frames at a batch of eight is six calls. Reporting per frame would turn a
liveness signal into queue traffic — on a Redis-backed lease that is
forty-eight round trips for something that needs to fire a handful of times,
and the extra forty-two carry no information a consumer can act on.

Pooling is top-k mean, the v1 baseline from the frozen architecture. A plain
mean over every frame dilutes a six-frame manipulation inside a forty-eight
frame clip into nothing, which is the single failure the video path exists to
avoid.

Deterministic throughout: uniform frame sampling, content-derived scores, no
randomness. The same bytes always produce the same bundle.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np

from contracts.enums import AbstainReason, BranchId, JobStage, MediaKind
from contracts.evidence import EvidenceBundle
from contracts.job import JobSpec
from contracts.video import FrameScore
from inference import postprocess as post
from inference import preprocess as pre
from inference.calibration import sigmoid
from contracts.version import compose_runtime_version
from inference.decode_env import decode_identity
from inference.models import ModelRegistry, ModelSpec, default_registry
from inference.postprocess import BranchOutcome, QualityFacts
from inference.protocol import ProgressFn, UnsupportedMedia
from inference.timing import INFERENCE, PREPROCESS, PhaseTimer

DEFAULT_FRAMES = 48
PROGRESS_BATCH = 8
MIN_USABLE_FRAMES = 8


class VideoPipeline:
    """``VideoInferenceEngine`` over a registered model."""

    def __init__(
        self,
        *,
        model_name: str = "stub-video",
        registry: ModelRegistry | None = None,
        caps: pre.MediaCaps = pre.DEFAULT_CAPS,
        n_frames: int | None = None,
        batch_size: int = PROGRESS_BATCH,
        min_usable_frames: int | None = None,
        face_selection: pre.FaceSelectionPolicy | None = None,
        video_selection: pre.VideoSelectionPolicy | None = None,
    ) -> None:
        self._registry = registry or default_registry()
        self._model_name = model_name
        self._caps = caps
        # R29 — the frame-selection and pooling knobs are a required-fields policy
        # object whose fingerprint enters the model identity, exactly as R17 did
        # for face selection and R25 for crop geometry.
        #
        # ``n_frames`` / ``batch_size`` / ``min_usable_frames`` remain accepted as
        # deprecated scalars (slated for removal). When one is passed explicitly
        # it is reconstructed into the policy via ``replace()`` *before* the
        # identity is computed, so the identity always reflects the values
        # actually used. ``batch_size`` is not a policy field — it is
        # output-neutral (see ``VideoSelectionPolicy``).
        base = video_selection or pre.default_video_selection()
        overrides: dict[str, int] = {}
        if n_frames is not None:
            overrides["n_frames"] = n_frames
        if min_usable_frames is not None:
            overrides["min_usable_frames"] = min_usable_frames
        if overrides:
            base = replace(base, **overrides)
        self._video_selection = base
        self._n_frames = base.n_frames
        self._batch_size = batch_size
        self._min_usable = base.min_usable_frames
        # CARRIES ``FaceSelectionPolicy``, CONSUMES ``VideoSelectionPolicy``.
        #
        # This pipeline detects no faces at all (R18): it scores whole resized
        # frames, so the *face* selection policy cannot move a single video
        # verdict today. It is folded into the version anyway, for one reason
        # that is about the platform rather than about this pipeline: ``Settings.
        # model_version`` is a single string that the API uses to key the result
        # cache for *both* media kinds, and ``verify_version_alignment`` requires
        # every engine to report it. An image-only segment would either fail
        # startup or force a per-kind configured version, which is a serving
        # change. When R18 gives video a face stage this segment already carries
        # the face policy it will consume.
        #
        # The video pipeline *does* now consume its own ``VideoSelectionPolicy``
        # (n_frames, min_usable_frames, pool_divisor), and ``selection_identity``
        # folds both policies into the one ``.f.`` segment.
        # ``test_the_video_pipeline_carries_the_segment_without_consuming_it``
        # pins the face half; ``test_video_selection_identity.py`` pins the video
        # half.
        self._selection = face_selection or pre.default_face_selection()
        self._spec: ModelSpec = (
            self._registry.get_spec(model_name)
            .with_gate(caps.gate_fingerprint)
            .with_selection(
                pre.selection_identity(self._selection, self._video_selection)
            )
        )

    @property
    def model_identity(self) -> str:
        """Weights + calibration + gate. **Which model this is.**

        The string a benchmark comparison uses: "did the model get worse" must
        not be answered yes because a worker was rebuilt with a different
        libavcodec.
        """
        return self._spec.effective_version

    @property
    def model_version(self) -> str:
        """Model identity **plus the decode environment**. The cache key.

        Named ``model_version`` because that is what every consumer already
        calls it — ``engine_versions``, ``verify_version_alignment``, the
        content cache, ``/health`` — and every one of them wants this, the
        broader question: *would this deployment produce the same number?*
        Batch 8 measured that the decoder alone can change the answer, so a
        version that omits it cannot key a cache.

        :attr:`model_identity` is the narrower string for callers that genuinely
        mean the model, and ``contracts.version.model_identity_of`` recovers it
        from this one.
        """
        return compose_runtime_version(
            model=self._spec.effective_version, decode=decode_identity()
        )

    async def warmup(self) -> None:
        loaded = self._registry.load_model(self._model_name)
        probe = np.zeros((1, self._spec.input_size, self._spec.input_size, 3), np.float32)
        loaded.backend.score(probe)

    async def close(self) -> None:
        self._registry.unload_model(self._model_name)

    async def analyze_video(
        self,
        spec: JobSpec,
        media: bytes,
        progress: ProgressFn | None = None,
    ) -> EvidenceBundle:
        timer = PhaseTimer(spec.submitted_at)
        if spec.media_kind is not MediaKind.VIDEO:
            raise UnsupportedMedia(
                f"This pipeline analyses video; the job declares {spec.media_kind.value}."
            )

        # Raises UnreadableMedia for an undecodable container, UnsupportedMedia
        # for one that exceeds the caps. Caps are applied to the container's own
        # metadata before a single frame is decoded.
        # Container probe and frame decode are blocking cv2 work — the single
        # longest uninterrupted stretch in the whole video path.
        #
        # ``target_size`` makes the decoder resize as it goes and write into one
        # pre-allocated buffer. Extracting at source resolution and resizing
        # afterwards held 48 full-resolution frames plus a stacked copy of them:
        # 2.4 GB peak on a 4K clip whose useful output is 21 MB.
        with timer.phase(PREPROCESS):
            meta, frames, indices, stamps = await asyncio.to_thread(
                pre.extract_frames,
                media,
                mime=spec.declared_mime,
                n_frames=self._n_frames,
                caps=self._caps,
                target_size=self._spec.input_size,
            )

        quality = QualityFacts(width=meta.width, height=meta.height, face_px=None)

        if frames.shape[0] < self._min_usable:
            # Too little of the clip survived decoding to say anything. A verdict
            # from three frames of a damaged file is a guess wearing a number.
            return self._abstain(
                spec,
                AbstainReason.TOO_FEW_FRAMES,
                meta,
                quality,
                frames_sampled=len(indices) or self._n_frames,
                frames_scored=int(frames.shape[0]),
                size_bytes=len(media),
                timer=timer,
            )

        with timer.phase(PREPROCESS):
            loaded = await asyncio.to_thread(self._registry.load_model, self._model_name)

        # Per-branch curves, kept apart. The previous implementation fused each
        # frame, pooled the fused curve, and then *re-derived* branches A, B and
        # P from that single pooled number — branch B as `pooled * 0.85 + 0.05`.
        # The backend had already produced a real branch-B score for every
        # frame; it was computed, discarded, and replaced by an invention. The
        # published `disagreement` and the interval derived from it were
        # therefore a deterministic function of branch A and told a user nothing.
        per_branch: dict[BranchId, list[float]] = {}
        per_frame: list[float] = []
        n_batches = max(1, (frames.shape[0] + self._batch_size - 1) // self._batch_size)
        for i, batch in enumerate(pre.batch_clips(frames, self._batch_size)):
            # Off the event loop: a model forward blocks, and the worker's lease
            # heartbeat is a task on that loop.
            #
            # The uint8 → float32 widening happens per batch rather than once
            # over the whole clip. Widening up front quadrupled the resident
            # frame data for no benefit: the backend only ever sees one batch.
            with timer.phase(INFERENCE):
                logits = await asyncio.to_thread(self._score_batch, loaded, batch)
            for branch, values in logits.items():
                per_branch.setdefault(branch, []).extend(float(v) for v in values)
            per_frame.extend(self._fuse_batch(logits, batch.shape[0]))
            if progress is not None:
                # One call per completed batch. Nothing else reports progress —
                # that is what makes 48 frames fire exactly 6 times.
                await progress((i + 1) / n_batches, stage=JobStage.SCORING)

        scores = np.asarray(per_frame, dtype=np.float64)
        # Both fields describe the *same* retained frame: its source index and
        # the container's own timestamp for it. Previously ``t_ms`` was
        # ``i * duration / len(survivors)`` — the survivors re-spread evenly
        # across the whole clip — so whenever a frame failed to decode the
        # curve's timestamps silently stopped matching its frame indices, and
        # every suspect-segment boundary pointed at the wrong moment.
        frame_scores = [
            FrameScore(
                frame_idx=int(indices[i]),
                t_ms=int(stamps[i]),
                p=float(min(1.0, max(0.0, sigmoid(scores[i])))),
            )
            for i in range(len(scores))
        ]

        # One frame selection, shared by every branch. Ranking on the fused
        # curve and then pooling each branch over those same frames keeps the
        # branches describing the same moments of the clip, so a disagreement
        # between them is a disagreement about the media rather than about which
        # frames each happened to look at.
        # ``k`` comes from the policy's ``pool_divisor`` (R29), passed explicitly
        # rather than left to ``topk_indices``' ``arr.size // 8`` fallback, so the
        # value driving the pool and the value in the identity are one field.
        # ``max(1, …)`` matches the fallback's own floor; with the shipped
        # divisor 8 and ``min_usable_frames`` 8 this is byte-identical to the
        # pre-Batch-23 call.
        k = max(1, len(scores) // self._video_selection.pool_divisor)
        selected = post.topk_indices(scores, k=k)
        outcomes = {
            branch: BranchOutcome(logit=post.pool_at(values, selected))
            for branch, values in per_branch.items()
        }

        analysis = post.build_video_analysis(
            frames_sampled=len(indices),
            frames_scored=len(frame_scores),
            frame_scores=frame_scores,
            threshold=self._spec.threshold,
            sampling=self._video_selection.sampling_rule,
            pooling=self._video_selection.pooling_rule,
        )
        return post.build_video_bundle(
            job=spec,
            model=self._spec,
            size_bytes=len(media),
            width=meta.width,
            height=meta.height,
            duration_s=meta.duration_s,
            frame_count=meta.frame_count,
            fps=meta.fps,
            quality=quality,
            outcomes=outcomes,
            video=analysis,
            timing=timer.to_timing(),
            runtime_version=self.model_version,
        )

    # ---- helpers ---------------------------------------------------------

    def _score_batch(self, loaded, batch: np.ndarray) -> dict[BranchId, np.ndarray]:
        """Widen one batch to float32 and score it. Runs in a worker thread.

        The widening is per batch by design — see the call site. Values are
        identical to widening the whole clip at once, because ``astype`` is
        elementwise.
        """
        return loaded.backend.score(batch.astype(np.float32))

    def _fuse_batch(
        self, logits: dict[BranchId, np.ndarray], n: int
    ) -> list[float]:
        """One fused logit per frame in the batch."""
        out: list[float] = []
        for i in range(n):
            outcomes = {
                branch: BranchOutcome(logit=float(values[i]))
                for branch, values in logits.items()
            }
            fused = post.fuse(outcomes, self._spec.branch_weights)
            out.append(0.0 if fused is None else fused)
        return out

    def _abstain(
        self,
        spec: JobSpec,
        reason: AbstainReason,
        meta: pre.VideoMeta,
        quality: QualityFacts,
        *,
        frames_sampled: int,
        frames_scored: int,
        size_bytes: int,
        timer: PhaseTimer,
    ) -> EvidenceBundle:
        quality.gate_passed = False
        quality.gate_reason = reason
        outcomes = {
            BranchId.A: BranchOutcome(abstain_reason=reason),
            BranchId.B: BranchOutcome(abstain_reason=AbstainReason.QUALITY_BELOW_FLOOR),
            BranchId.P: BranchOutcome(abstain_reason=AbstainReason.NO_PROVENANCE_DATA),
        }
        return post.build_video_bundle(
            job=spec,
            model=self._spec,
            size_bytes=size_bytes,
            width=meta.width,
            height=meta.height,
            duration_s=meta.duration_s,
            frame_count=meta.frame_count,
            fps=meta.fps,
            quality=quality,
            outcomes=outcomes,
            video=post.build_video_analysis(
                frames_sampled=max(1, frames_sampled),
                frames_scored=frames_scored,
                frame_scores=[],
                threshold=self._spec.threshold,
                sampling=self._video_selection.sampling_rule,
                pooling=self._video_selection.pooling_rule,
            ),
            timing=timer.to_timing(),
            forced_abstention=reason,
            runtime_version=self.model_version,
        )
