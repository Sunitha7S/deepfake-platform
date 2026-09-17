"""Image pipeline: bytes → preprocess → inference → calibration → bundle.

Gates run **before** the model, in cost order: decode, then face detection, then
the size and resolution floors. A face 40 pixels across carries no forensic
detail, and scoring it produces a confident number about nothing — the gate
turns that into an honest "insufficient evidence" instead.

No randomness anywhere. Scores derive from content, so the same bytes always
produce the same bundle: crash recovery compares results across attempts, and a
pipeline that drifted would make those comparisons meaningless.
"""

from __future__ import annotations

import asyncio

import numpy as np

from contracts.enums import AbstainReason, BranchId, JobStage, MediaKind
from contracts.evidence import EvidenceBundle
from contracts.job import JobSpec
from inference import preprocess as pre
from inference import postprocess as post
from contracts.version import compose_runtime_version
from inference.decode_env import decode_identity
from inference.models import ModelRegistry, ModelSpec, default_registry
from inference.protocol import ProgressFn, UnsupportedMedia
from inference.timing import INFERENCE, PREPROCESS, PhaseTimer
from inference.postprocess import BranchOutcome, QualityFacts


class ImagePipeline:
    """``ImageInferenceEngine`` over a registered model."""

    def __init__(
        self,
        *,
        model_name: str = "stub-image",
        registry: ModelRegistry | None = None,
        caps: pre.MediaCaps = pre.DEFAULT_CAPS,
        face_selection: pre.FaceSelectionPolicy | None = None,
        video_selection: pre.VideoSelectionPolicy | None = None,
    ) -> None:
        self._registry = registry or default_registry()
        self._model_name = model_name
        self._caps = caps
        # One policy object drives both the detection and the identity. Passing
        # it explicitly to detect_faces and crop_face — rather than letting them
        # fall back to their own default — is what makes the fingerprint a
        # statement about this pipeline rather than about the module.
        self._selection = face_selection or pre.default_face_selection()
        # CARRIED, NOT CONSUMED — and the carry is PERMANENT. The image path
        # samples no frames and pools nothing, so VideoSelectionPolicy can NEVER
        # move an image verdict; there is no future stage that will make it live
        # here. (This is the asymmetry with VideoPipeline's carry of
        # FaceSelectionPolicy, which SCRFD *will* make live.) It is folded into
        # the identity anyway because ``Settings.model_version`` is one string
        # keying the cache for both media kinds and ``verify_version_alignment``
        # requires every engine to report it — an image string that omitted the
        # video segment would fail worker startup. ``selection_identity``
        # combines the two into the one ``.f.`` segment (Batch 23, R29). The cost
        # is that the image cache namespace regenerates when a video-only policy
        # changes: a false miss, the safe direction — the B20 ``.d.`` precedent.
        # ``test_the_image_pipeline_carries_the_video_segment_without_consuming_it``
        # pins both halves.
        self._video_selection = video_selection or pre.default_video_selection()
        # The served spec carries the gate and the selection policy it is served
        # behind, so the effective version moves when a quality floor moves and
        # when the detector, its parameters, the ordering rule or the crop
        # geometry move (R17/R25) — and now when the video frame-selection or
        # pooling policy moves (R29).
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
        """Load weights and run one throwaway inference.

        A cold first request otherwise pays the whole lazy-initialisation cost,
        which on a real backend is the difference between a fast job and one
        that looks broken.
        """
        loaded = self._registry.load_model(self._model_name)
        probe = np.zeros((1, self._spec.input_size, self._spec.input_size, 3), np.float32)
        loaded.backend.score(probe)

    async def close(self) -> None:
        self._registry.unload_model(self._model_name)

    async def analyze_image(
        self,
        spec: JobSpec,
        media: bytes,
        progress: ProgressFn | None = None,
    ) -> EvidenceBundle:
        timer = PhaseTimer(spec.submitted_at)
        if spec.media_kind is not MediaKind.IMAGE:
            raise UnsupportedMedia(
                f"This pipeline analyses images; the job declares {spec.media_kind.value}."
            )

        # Raises UnreadableMedia / UnsupportedMedia — malformed input is the
        # caller's problem to classify, not something to paper over with a
        # verdict.
        #
        # Off the event loop. cv2 decode, Haar detection and a model forward are
        # CPU-bound and yield to nothing, so running them inline holds the loop
        # for the whole analysis — and the worker's lease heartbeat is a task on
        # that same loop. A blocking analysis longer than the renewal interval
        # does not merely delay the heartbeat, it prevents it, and the worker
        # loses a lease it is actively working. See tests/test_lease_safety.py.
        with timer.phase(PREPROCESS):
            image = await asyncio.to_thread(pre.decode_image, media, caps=self._caps)
        height, width = image.shape[:2]

        if progress is not None:
            await progress(0.4, stage=JobStage.DETECTING)

        # Detection is preprocessing and is timed as such. Measuring
        # ``preprocess_ms`` before this line and ``inference_ms`` after it left
        # detection — the most expensive step on the abstention path — inside
        # neither, and 89% of a gate-refused job went unaccounted for.
        with timer.phase(PREPROCESS):
            faces, blur = await asyncio.to_thread(
                _detect_and_measure, image, self._selection
            )
        quality = QualityFacts(
            width=width,
            height=height,
            # ``size_px``, not ``short_edge``, and not ``area``: one metric ranks,
            # gates and reports, so the published figure describes the face the
            # gate judged and the ordering chose. R19.
            face_px=faces[0].size_px if faces else 0,
            blur_score=blur,
        )

        gate = self._gate(faces, quality)
        if gate is not None:
            return self._abstain(spec, gate, quality, faces, len(media), timer)

        if progress is not None:
            await progress(0.75, stage=JobStage.SCORING)

        with timer.phase(INFERENCE):
            logits = await asyncio.to_thread(self._crop_and_score, image, faces[0])

        outcomes = {
            branch: BranchOutcome(logit=float(values[0]))
            for branch, values in logits.items()
        }

        face = faces[0]
        return post.build_image_bundle(
            job=spec,
            model=self._spec,
            size_bytes=len(media),
            width=width,
            height=height,
            quality=quality,
            outcomes=outcomes,
            faces=[
                post.face_track(
                    (face.x, face.y, face.width, face.height),
                    p_fake=_face_probability(outcomes, self._spec),
                )
            ],
            timing=timer.to_timing(),
            runtime_version=self.model_version,
        )

    # ---- blocking sections -----------------------------------------------

    def _crop_and_score(
        self, image: np.ndarray, face: pre.FaceBox
    ) -> dict[BranchId, np.ndarray]:
        """Crop, prepare and score. Runs in a worker thread.

        Weight loading is inside deliberately: it is disk and CPU work, it is
        the single slowest thing a cold engine does, and it is guarded by a
        ``threading.RLock`` rather than an asyncio one precisely so it can be
        called from here.
        """
        crop = pre.crop_face(image, face, policy=self._selection)
        batch = pre.prepare_image(crop, size=self._spec.input_size)
        loaded = self._registry.load_model(self._model_name)
        return loaded.backend.score(batch)

    # ---- gates -----------------------------------------------------------

    def _gate(
        self, faces: list[pre.FaceBox], quality: QualityFacts
    ) -> AbstainReason | None:
        """Reasons to decline before spending a model call. Cheapest first.

        **The face floor tests ``faces[0].size_px``, the same quantity
        :func:`~inference.preprocess.face_order_key` ranks on.** That identity is
        R19's fix and it is what makes this gate honest: ``faces[0]`` maximises
        ``size_px``, so ``FACE_TOO_SMALL`` here means *no face in the file
        clears the floor*, not merely *the one we happened to pick does not*.
        Until Batch 14 the ordering ranked on area while this line tested the
        short edge, and the two disagree for any box more than two pixels from
        square — so a file containing a 200 px face could be refused because a
        900x50 box outranked it. ``docs/FACE_SIZE_POLICY.md``.

        The first floor is a different quantity that happens to share a name:
        ``min_short_edge`` is the *image's* short edge, ``min_face_px`` the
        *face's*. Both default to 64 and they are not the same check.
        """
        if min(quality.width, quality.height) < self._caps.min_short_edge:
            return AbstainReason.QUALITY_BELOW_FLOOR
        if not faces:
            return AbstainReason.NO_FACE_DETECTED
        if faces[0].size_px < self._caps.min_face_px:
            return AbstainReason.FACE_TOO_SMALL
        return None

    def _abstain(
        self,
        spec: JobSpec,
        reason: AbstainReason,
        quality: QualityFacts,
        faces: list[pre.FaceBox],
        size_bytes: int,
        timer: PhaseTimer,
    ) -> EvidenceBundle:
        """A gate refusal. A successful analysis whose answer is 'not enough'."""
        quality.gate_passed = False
        quality.gate_reason = reason
        outcomes = {
            BranchId.A: BranchOutcome(abstain_reason=reason),
            BranchId.B: BranchOutcome(abstain_reason=AbstainReason.QUALITY_BELOW_FLOOR),
            BranchId.P: BranchOutcome(abstain_reason=AbstainReason.NO_PROVENANCE_DATA),
        }
        tracks = []
        if faces:
            f = faces[0]
            tracks = [
                post.face_track((f.x, f.y, f.width, f.height), p_fake=None, status=reason)
            ]
        return post.build_image_bundle(
            job=spec,
            model=self._spec,
            size_bytes=size_bytes,
            width=quality.width,
            height=quality.height,
            quality=quality,
            outcomes=outcomes,
            faces=tracks,
            timing=timer.to_timing(),
            forced_abstention=reason,
            runtime_version=self.model_version,
        )


def _detect_and_measure(
    image: np.ndarray, policy: pre.FaceSelectionPolicy
) -> tuple[list[pre.FaceBox], float]:
    """Detection and the blur estimate, together, in one thread hop.

    Two hops would cost two context switches for work that is always needed
    together and always cheap relative to the model call.
    """
    return pre.detect_faces(image, policy=policy), _blur_score(image)


def _blur_score(image: np.ndarray) -> float:
    """Variance of the Laplacian, inverted and bounded to [0,1].

    Low variance means few sharp edges, which means blur. A rough proxy, used
    only for reporting — the resolution and face-size floors do the gating,
    because they are measured rather than estimated.
    """
    import cv2

    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    variance = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    return float(min(1.0, 1.0 / (1.0 + variance / 100.0)))


def _face_probability(
    outcomes: dict[BranchId, BranchOutcome], spec: ModelSpec
) -> float | None:
    fused = post.fuse(outcomes, spec.branch_weights)
    return None if fused is None else post.calibrated_probability(fused, spec)
