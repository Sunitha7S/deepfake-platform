"""Model registry: metadata, lazy weight loading, lifecycle.

**Nothing here imports torch at module scope.** On this machine ``import torch``
costs 15.5 seconds — measured, not estimated. A package that pays that at import
time makes every test run, every worker boot and every CI job wait for a library
most of them never use. Torch is imported inside ``load_model`` and only for
backends that need it; the stub backend never touches it at all, which is why
the whole test suite runs in milliseconds.

**Metadata is separate from weights.** ``ModelSpec`` carries the version,
threshold, branch weights and a reference to calibration — all readable without
loading anything. That is what lets ``/health`` report a model version, and the
registry answer ``get_loaded_models()``, without a gigabyte resident.

**Calibration is separate again.** ``CalibrationSpec`` is its own object, so
recalibrating ships a small JSON file rather than a re-exported checkpoint. The
two have genuinely different lifecycles: weights change when you retrain,
calibration changes whenever the validation distribution moves.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from contracts.common import DEFAULT_LOG_ODDS_MARGIN, ThresholdInfo
from contracts.enums import BranchId
from contracts.version import compose_model_version, fingerprint, quantize
from inference.protocol import ModelUnavailable

BackendKind = Literal["stub", "torch", "onnx"]


# --------------------------------------------------------------------------
# Specifications
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationSpec:
    """Temperature and the evidence for it. Stored apart from weights.

    ``fitted_on`` and ``n_samples`` are not bookkeeping: a temperature fitted on
    400 images from one dataset should not be applied silently to a different
    distribution, and recording where it came from is what makes that
    reviewable.
    """

    temperature: float = 1.0
    fitted_on: str | None = None
    """Free text, and that is the point of ``witness`` below.

    Nothing has ever checked this string against anything. It is written by the
    same caller that did the fit, so it records an intention, not a fact — a
    temperature fitted on the benchmark corpus can say ``"ffpp-test"`` here and
    no part of the system disagrees.
    """
    n_samples: int = 0
    nll_before: float | None = None
    nll_after: float | None = None
    witness: str | None = None
    """Digest of the :class:`~evaluation.calibration_protocol.CalibrationWitness`.

    ``None`` means this calibration was fitted on bare arrays: still perfectly
    usable, and not claiming otherwise. A non-``None`` value can only have come
    from ``fit_calibration``, which requires a CALIBRATION-role selection — so
    this field is the difference between a claim and evidence, and
    :attr:`is_protocol_compliant` is how a caller tells them apart.

    **Deliberately NOT part of ``calibration_fingerprint``.** That fingerprint
    is half of the result cache key and answers one question: "would this
    deployment produce the same number?". Provenance does not change a single
    probability, so two deployments with identical T, threshold and band are
    cache-equivalent whether or not one of them can prove where it came from.
    Putting the witness in would fragment the cache on a distinction the cache
    does not care about — and would move ``STUB_ENGINE_VERSION`` for a field
    that is ``None`` on every shipped configuration.
    """

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    @property
    def is_protocol_compliant(self) -> bool:
        """Whether this calibration can show which corpus it was fitted on."""
        return self.witness is not None

    @property
    def is_identity(self) -> bool:
        """True when no real calibration has been applied.

        The honest default. An uncalibrated model should behave exactly as it
        did rather than slightly differently, so T defaults to 1.0 and callers
        can tell that apart from a fitted 1.0-ish value by checking
        ``n_samples``.
        """
        return self.temperature == 1.0 and self.n_samples == 0


DEFAULT_BRANCH_WEIGHTS: dict[BranchId, float] = {
    BranchId.A: 0.72,
    BranchId.B: 0.24,
    BranchId.P: 0.04,
}
"""Placeholder weights, pending the fitted fusion.

A and B are pooled against each other and so only their *ratio* matters — 0.72
and 0.24 mean "A carries three times B's vote", not "A carries 72% of the
verdict". P is not in that pool at all: it is the coefficient on a one-sided
authenticity credit, in logits per unit of credit, and 0.04 is deliberately
small because nothing has yet measured what a valid C2PA signature is worth.
Fitting all three belongs to the calibration stage, not here.
"""


@dataclass(frozen=True)
class ModelSpec:
    """Everything about a model that is knowable without loading it.

    ``version`` identifies the **weights only**. It is deliberately not the
    string anything else should use: the value that identifies *what will
    happen to a score* is :attr:`effective_version`, which folds in calibration
    and the quality gate. Publishing ``version`` as the model identity is what
    let a recalibration reuse results computed under the previous temperature.
    """

    name: str
    version: str
    backend: BackendKind = "stub"
    input_size: int = 384
    weights_uri: str | None = None
    branch_versions: dict[BranchId, str] = field(default_factory=dict)
    """Per-branch weights identity. Falls back to ``version`` for any branch
    left unset, so a single-checkpoint model needs no ceremony — but a
    deployment that retrains only branch A can say so, and the bundle records
    it. Three copies of one string cannot express that and so never did."""
    gate_fingerprint: str | None = None
    """Fingerprint of the quality-gate caps this model is served behind.

    Set by the pipeline, which is the only object that holds both the spec and
    the caps. Kept as a plain string here so ``models`` stays free of the cv2
    import that ``preprocess`` carries."""
    face_selection_fingerprint: str | None = None
    """Fingerprint of the selection policy this model is served behind.

    **R17, Batch 12.** Set by the pipeline for the same reason and by the same
    mechanism as :attr:`gate_fingerprint` — the pipeline is the only object
    holding both the spec and the policy — and kept as a plain string so this
    module stays free of cv2. The field name is kept for compatibility; since
    Batch 23 (R29) the pipelines set it to the *combined* selection fingerprint
    (``inference.preprocess.selection_identity`` — face selection, plus frame
    selection and pooling for video).

    Why it is a *separate* field from the gate rather than folded into it: the
    gate decides which inputs get a score at all, and this decides which
    face/frames get scored and what pixels the model is shown. Folding them
    would make "we raised the face floor" and "we replaced the detector" the
    same event, and the SCRFD migration moves exactly one of the two.

    ``None`` means the spec makes no claim about selection — the honest
    value for a spec constructed outside a pipeline, and what every version
    written before Batch 12 parses back to."""
    threshold: ThresholdInfo = field(
        default_factory=lambda: ThresholdInfo.at_margin(
            value=0.62, margin=DEFAULT_LOG_ODDS_MARGIN, target_fpr=0.02
        )
    )
    branch_weights: dict[BranchId, float] = field(
        default_factory=lambda: dict(DEFAULT_BRANCH_WEIGHTS)
    )
    calibration: CalibrationSpec = field(default_factory=CalibrationSpec)

    def __post_init__(self) -> None:
        if BranchId.P in self.branch_weights and self.branch_weights[BranchId.P] < 0:
            # The one-sided provenance rule, enforced at the weight as well as
            # at the logit. A negative weight on a non-negative logit produces
            # negative evidence, which is the shortcut the whole rule exists to
            # prevent — absence of provenance must contribute exactly zero.
            raise ValueError("branch P weight must be non-negative")
        total = sum(self.branch_weights.values())
        if total <= 0:
            raise ValueError("branch weights must sum to something positive")
        # At least one branch must be able to produce manipulation evidence.
        # Provenance modulates that evidence and cannot substitute for it, so a
        # spec weighted entirely on P would abstain on every input it ever saw
        # while looking perfectly well configured. Refusing at construction is
        # the only place that is cheap to notice.
        if not any(self.branch_weights.get(b, 0.0) > 0 for b in (BranchId.A, BranchId.B)):
            raise ValueError(
                "at least one manipulation branch (A or B) must carry positive weight; "
                "branch P alone can never produce a verdict"
            )

    @property
    def is_stub(self) -> bool:
        return self.backend == "stub"

    # ---- identity --------------------------------------------------------

    @property
    def calibration_fingerprint(self) -> str:
        """Short hash of everything about the calibration that moves a verdict.

        The temperature is the part that changes probabilities, but ``fitted_on``
        and ``n_samples`` are in here too: two deployments running T = 1.0 for
        genuinely different reasons — one uncalibrated, one fitted and found to
        need no adjustment — are not interchangeable, and a cache that treats
        them as the same model cannot later tell which produced a stored result.

        The **operating point is in here as well**, and it was the gap Batch 0
        left open. Weights, fusion coefficients and the quality gate all reached
        the composite version; the threshold and its abstention band did not. So
        moving the decision boundary from 0.62 to 0.80 changed every verdict the
        deployment would produce while leaving its version string identical, and
        the result cache went on serving verdicts decided at the old boundary.
        Threshold and temperature belong together anyway: both are outputs of
        the same fit on the same calibration holdout.

        EVERY FLOAT HERE IS QUANTISED FIRST
        -----------------------------------
        These are the platform's only *fitted* quantities, and fitted quantities
        carry float noise that configured ones do not. A temperature refitted on
        the same calibration set in a different order used to differ at about
        1e-8 — far below anything that moves a verdict, and more than enough to
        produce an entirely different fingerprint, because ``_canonical``
        preserves floats exactly and is right to.

        ``fit_temperature`` is now order-invariant, which fixes the cause.
        :func:`~contracts.version.quantize` is the second layer, covering the
        noise sources that fix does not: a different numpy build, float32 inputs,
        a threshold re-derived from a resampled quantile. Rounding to 1e-6 cannot
        merge two calibrations that would decide differently — that is four
        orders of magnitude below the resolution of any decision this system
        makes.

        Note the shipped defaults are unaffected: 0.62, 0.52, 0.72, 0.02 and
        1.0 all round to themselves, so ``STUB_ENGINE_VERSION`` does not move.
        """
        return fingerprint(
            {
                "temperature": quantize(self.calibration.temperature),
                "fitted_on": self.calibration.fitted_on,
                "n_samples": self.calibration.n_samples,
                "threshold": quantize(self.threshold.value),
                "band_lo": quantize(self.threshold.band_lo),
                "band_hi": quantize(self.threshold.band_hi),
                "target_fpr": quantize(self.threshold.target_fpr),
            }
        )

    @property
    def weights_identity(self) -> str:
        """The weights half of the version, including per-branch detail.

        Branch weights are in here because they are fusion coefficients: moving
        branch A from 0.72 to 0.80 changes every fused logit the model produces,
        with no retraining and no recalibration involved.

        ``input_size`` is in here as of **Batch 22 (R25)**. A trained checkpoint
        has an input resolution baked into its first layer, so serving it at a
        different ``input_size`` is a misconfiguration *of the weights*, and this
        is where a weights misconfiguration should surface — ``prepare_image``
        and ``extract_frames`` read exactly this attribute to size the model's
        field of view, so the value that drives the tensor and the value that
        drives the hash are the same. It is folded into the ``-w`` fingerprint
        rather than added as a new ``-i`` token: that keeps ``weights_identity``'s
        format ( ``{version}{-b…}-w{fp}`` ) byte-for-byte unchanged, so
        ``parse_model_version`` — which treats everything before ``+c.`` as an
        opaque blob — needs no change and no new segment marker can be
        mis-split. The cost is legibility: an operator sees ``-w7f45fada`` ->
        ``-wXXXXXXXX`` and cannot tell *which* input moved. ``verify_version_
        alignment`` names the mismatch at startup regardless.
        """
        per_branch = {b.value: self.branch_versions.get(b, self.version) for b in BranchId}
        if all(v == self.version for v in per_branch.values()):
            # Nothing branch-specific to record. Keep the common case readable
            # rather than appending a hash that never varies.
            branch_part = ""
        else:
            branch_part = f"-b{fingerprint(per_branch)}"
        weights_part = fingerprint(
            {
                **{b.value: w for b, w in self.branch_weights.items()},
                "input_size": self.input_size,
            }
        )
        return f"{self.version}{branch_part}-w{weights_part}"

    @property
    def effective_version(self) -> str:
        """**The** model identity. Weights, calibration and gate, composed.

        Everything downstream — ``ModelInfo.version``, the result cache key, the
        job idempotency key, ``/health`` — must use this and nothing else. It is
        the answer to "would this deployment produce the same number today", and
        that is the only question a cache key is asking.
        """
        return compose_model_version(
            weights=self.weights_identity,
            calibration=self.calibration_fingerprint,
            gate=self.gate_fingerprint,
            face_selection=self.face_selection_fingerprint,
        )

    @property
    def pre_calibration_identity(self) -> str:
        """Weights + gate + selection. Everything :attr:`effective_version`
        composes **except** calibration. **R31.**

        The model configuration a calibration fit is fitted *against*,
        computable before that fit exists — which is what makes it safe to
        pass into :func:`~evaluation.calibration_protocol.fit_calibration`
        without a dependency cycle: :attr:`calibration_fingerprint` never
        reads this property or anything derived from it, only
        ``self.calibration``/``self.threshold``, so calibration cannot depend
        on its own binding key.

        Not part of any cache key or the result cache — that remains
        :attr:`effective_version` alone. This exists solely so a calibration
        witness can record which model configuration produced the logits it
        was fitted on, closing R31: two models differing in weights, gate or
        selection but scoring a calibration corpus identically previously
        produced indistinguishable witnesses, because nothing about the
        model's own configuration reached the digest at all.
        """
        return fingerprint(
            {
                "weights": self.weights_identity,
                "gate": self.gate_fingerprint,
                "face_selection": self.face_selection_fingerprint,
            }
        )

    def version_of(self, branch: BranchId) -> str:
        return self.branch_versions.get(branch, self.version)

    def with_gate(self, gate_fingerprint: str) -> "ModelSpec":
        """Return this spec bound to a quality gate. Used by the pipelines."""
        if self.gate_fingerprint == gate_fingerprint:
            return self
        return replace(self, gate_fingerprint=gate_fingerprint)

    def with_selection(self, selection_fingerprint: str) -> "ModelSpec":
        """Return this spec bound to a selection-policy fingerprint. The
        honest-named analogue of :meth:`with_face_selection`; since Batch 23 (R29)
        the pipelines pass ``inference.preprocess.selection_identity(face, video)``
        here.
        """
        return self.with_face_selection(selection_fingerprint)

    def with_face_selection(self, face_selection_fingerprint: str) -> "ModelSpec":
        """Return this spec bound to a selection policy. Used by the pipelines.

        Kept as a permanent alias target of :meth:`with_selection`; since Batch 23
        (R29) it receives the combined selection identity (face selection, plus
        frame selection and pooling for video), not a ``FaceSelectionPolicy``
        fingerprint alone.

        Exact structural parallel of :meth:`with_gate`, including the identity
        short-circuit: rebinding to the same fingerprint returns ``self`` so a
        pipeline constructed twice does not churn frozen instances.
        """
        if self.face_selection_fingerprint == face_selection_fingerprint:
            return self
        return replace(self, face_selection_fingerprint=face_selection_fingerprint)


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


@runtime_checkable
class ModelBackend(Protocol):
    """Turns a batch of prepared arrays into per-branch logits.

    The only thing a backend does. Everything else — preprocessing, fusion,
    calibration, bundle construction — lives outside, so swapping torch for ONNX
    changes this object and nothing else.
    """

    def score(self, batch: np.ndarray) -> dict[BranchId, np.ndarray]:
        """``(N, H, W, 3) float32`` in, one logit array of shape ``(N,)`` per branch."""
        ...

    def close(self) -> None: ...


class StubBackend:
    """Deterministic scoring from content. No torch, no weights, no GPU.

    Derives logits from a hash of the array contents, which gives three
    properties the tests depend on: identical input always scores identically,
    different inputs score differently, and nothing needs to be downloaded or
    loaded. It is not a model and never pretends to be — ``ModelSpec.backend``
    reads ``stub`` and the version string says so.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def _unit(self, arr: np.ndarray, salt: str) -> float:
        digest = hashlib.sha256(salt.encode() + arr.tobytes()).hexdigest()
        return int(digest[:12], 16) / float(0x1000000000000)

    def score(self, batch: np.ndarray) -> dict[BranchId, np.ndarray]:
        if batch.ndim != 4 or batch.shape[-1] != 3:
            raise ValueError(f"expected (N, H, W, 3), got {batch.shape}")
        out: dict[BranchId, list[float]] = {b: [] for b in self.spec.branch_weights}
        for i in range(batch.shape[0]):
            frame = batch[i]
            p_a = self._unit(frame, "spatial")
            # Branch B correlates with A but not perfectly, so fusion and
            # disagreement have something real to work with.
            p_b = min(0.99, max(0.01, p_a * 0.85 + 0.05))
            for branch, p in ((BranchId.A, p_a), (BranchId.B, p_b)):
                if branch in out:
                    out[branch].append(float(np.log(p / (1 - p))))
            if BranchId.P in out:
                # Never negative. No provenance data means zero evidence, not
                # evidence of manipulation.
                out[BranchId.P].append(0.0)
        return {b: np.asarray(v, dtype=np.float64) for b, v in out.items()}

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass
class LoadedModel:
    """A spec plus its live backend."""

    spec: ModelSpec
    backend: ModelBackend
    loaded_at: datetime
    load_ms: int

    @property
    def version(self) -> str:
        """The effective version, not the bare weights string.

        Diagnostics that report a loaded model must report the identity the
        cache and the bundles use, or an operator comparing /health against a
        stored result is comparing two different things.
        """
        return self.spec.effective_version


class ModelRegistry:
    """Holds specs always, weights only while loaded.

    Guarded by a plain lock rather than an asyncio one: loading weights is
    blocking CPU and disk work, and it is called from ``asyncio.to_thread``.
    An asyncio lock would be held across a thread boundary it does not protect.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._loaded: dict[str, LoadedModel] = {}
        self._lock = threading.RLock()

    # ---- specs -----------------------------------------------------------

    def register(self, spec: ModelSpec) -> ModelSpec:
        with self._lock:
            self._specs[spec.name] = spec
        return spec

    def get_spec(self, name: str) -> ModelSpec:
        with self._lock:
            try:
                return self._specs[name]
            except KeyError as exc:
                known = sorted(self._specs)
                raise ModelUnavailable(
                    f"No model registered as {name!r}.",
                    detail=f"registered: {known}",
                ) from exc

    def specs(self) -> dict[str, ModelSpec]:
        with self._lock:
            return dict(self._specs)

    # ---- lifecycle -------------------------------------------------------

    def load_model(self, name: str) -> LoadedModel:
        """Load weights if not already resident. Idempotent."""
        with self._lock:
            existing = self._loaded.get(name)
            if existing is not None:
                return existing
            spec = self.get_spec(name)
            started = time.perf_counter()
            backend = self._build_backend(spec)
            loaded = LoadedModel(
                spec=spec,
                backend=backend,
                loaded_at=datetime.now(UTC),
                load_ms=int((time.perf_counter() - started) * 1000),
            )
            self._loaded[name] = loaded
            return loaded

    def unload_model(self, name: str) -> bool:
        """Release weights. True if something was unloaded."""
        with self._lock:
            loaded = self._loaded.pop(name, None)
        if loaded is None:
            return False
        try:
            loaded.backend.close()
        except Exception:  # noqa: BLE001 - unload must not raise
            # A backend that fails to close is still unloaded from our side.
            # Raising here would leave the registry and reality disagreeing.
            pass
        return True

    def get_loaded_models(self) -> dict[str, LoadedModel]:
        with self._lock:
            return dict(self._loaded)

    def is_loaded(self, name: str) -> bool:
        with self._lock:
            return name in self._loaded

    def unload_all(self) -> int:
        for name in list(self.get_loaded_models()):
            self.unload_model(name)
        return 0

    # ---- backend construction -------------------------------------------

    def _build_backend(self, spec: ModelSpec) -> ModelBackend:
        if spec.backend == "stub":
            return StubBackend(spec)
        if spec.backend == "torch":
            return self._build_torch_backend(spec)
        if spec.backend == "onnx":
            return self._build_onnx_backend(spec)
        raise ModelUnavailable(f"Unknown backend {spec.backend!r} for model {spec.name!r}.")

    def _build_torch_backend(self, spec: ModelSpec) -> ModelBackend:
        """Import torch here and nowhere else.

        Deliberately inside the function. Importing at module scope costs 15.5
        seconds on this machine and would be paid by every consumer of this
        package, including the ones that only ever use the stub.
        """
        if not spec.weights_uri:
            raise ModelUnavailable(
                f"Model {spec.name!r} declares the torch backend but has no weights_uri."
            )
        try:
            import torch  # noqa: F401  (imported for availability, used by the real loader)
        except ImportError as exc:  # pragma: no cover - torch is installed here
            raise ModelUnavailable("PyTorch is not installed.", detail=str(exc)) from exc
        raise ModelUnavailable(
            f"The torch backend is not implemented yet; {spec.name!r} cannot be served.",
            detail=(
                "Real weights arrive with the model stage. This raises rather than "
                "silently falling back to the stub, because a deployment quietly "
                "serving hash-derived scores while reporting a model version is "
                "worse than one that refuses to start."
            ),
        )

    def _build_onnx_backend(self, spec: ModelSpec) -> ModelBackend:
        if not spec.weights_uri:
            raise ModelUnavailable(
                f"Model {spec.name!r} declares the onnx backend but has no weights_uri."
            )
        try:
            import onnxruntime  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ModelUnavailable("onnxruntime is not installed.", detail=str(exc)) from exc
        raise ModelUnavailable(
            f"The ONNX backend is not implemented yet; {spec.name!r} cannot be served.",
            detail="Same reasoning as the torch backend: refuse rather than fall back.",
        )


# --------------------------------------------------------------------------
# Default registry
# --------------------------------------------------------------------------

STUB_IMAGE = ModelSpec(name="stub-image", version="stub-0", backend="stub", input_size=384)
STUB_VIDEO = ModelSpec(name="stub-video", version="stub-0", backend="stub", input_size=384)

_default = ModelRegistry()
_default.register(STUB_IMAGE)
_default.register(STUB_VIDEO)


def default_registry() -> ModelRegistry:
    """The process-wide registry.

    A module-level instance rather than a global loaded model: it holds specs
    (cheap) and only holds weights once someone asks. Tests build their own
    where isolation matters.
    """
    return _default


def load_model(name: str, *, registry: ModelRegistry | None = None) -> LoadedModel:
    return (registry or _default).load_model(name)


def unload_model(name: str, *, registry: ModelRegistry | None = None) -> bool:
    return (registry or _default).unload_model(name)


def get_loaded_models(*, registry: ModelRegistry | None = None) -> dict[str, LoadedModel]:
    return (registry or _default).get_loaded_models()
