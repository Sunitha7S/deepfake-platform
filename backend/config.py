"""Typed application settings.

Read from the environment with a ``DFP_`` prefix, or from a ``.env`` file in
development. Two rules this module exists to enforce:

* **Limits are imported, never redeclared.** The upload caps live in
  ``contracts.api`` because the contract validates against them; duplicating the
  numbers here would let the API and the schema drift apart silently.
* **URLs are built in one place.** ``api_prefix`` defaults to ``""`` so routes
  sit at ``/health``, ``/uploads``, ``/jobs/{id}``, ``/results/{id}`` as
  specified. Moving the whole API under ``/api/v1`` later is then one
  environment variable rather than an edit to every route and every
  self-referential URL in a response body.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from contracts.api import (
    ALLOWED_IMAGE_MIME,
    ALLOWED_VIDEO_MIME,
    MAX_IMAGE_BYTES,
    MAX_VIDEO_BYTES,
    MAX_VIDEO_DURATION_S,
)
from contracts.enums import CONTRACT_VERSION

if TYPE_CHECKING:  # pragma: no cover - import only for type names
    from jobqueue.registry import QueueConfig
    from storage.registry import StorageConfig

Environment = Literal["dev", "staging", "prod"]

STUB_ENGINE_VERSION = "stub-0-wacdd9240+c.739c3fec.g.ae232cba.f.e629642c.d.a97ccb68"
"""Runtime version of the shipped stub engines, copied here as a literal.

**The ``.f.`` segment moved in S8 Batch 23 (R29 + N4).** It was ``251d22e9`` —
the face-selection policy fingerprint alone. It is now ``e629642c``, a
*composite*: ``fingerprint({"face": <face policy>, "video": <video policy>})``,
where the video policy is the new ``VideoSelectionPolicy`` (``n_frames``,
``min_usable_frames``, ``pool_divisor`` — functional — plus ``sampling_rule`` /
``pooling_rule`` — label closures also published on the wire). Batch 23 closed
the video analogue of R17/R25: frame-selection and pooling knobs that reach no
identity string, so two deployments that disagree on them share a result-cache
key. Both pipelines now call ``inference.preprocess.selection_identity`` and pass
its result to ``.with_selection`` (permanent alias ``.with_face_selection``) — the single literal here and
``contracts/version.py``'s parser are preserved (``.f.`` is still one opaque
8-hex fingerprint). ``default_face_selection().identity`` is **unchanged**
(``251d22e9``). **No published number changes** — every field names the current
shipped behaviour verbatim (proven by
``tests/test_video_selection_identity.py::test_the_shipped_video_bundle_did_not_move``).
One cache generation spanning both media namespaces (the B20 ``.d.`` precedent:
a video-only change regenerates the image namespace too, a false miss). No
contract / RUN_SCHEMA / witness / OpenAPI-structural move. N4 also corrected the
stale ``VideoAnalysis.sampling`` default ``"uniform+adaptive"`` → ``"uniform"``
(nothing adaptive runs). See ``docs/VIDEO_SELECTION_IDENTITY.md``.

**Both the ``-w`` and the ``.f.`` segment moved in S8 Batch 22 (R25)** — weights
``w7f45fada`` -> ``wacdd9240`` and face-selection ``19559387`` -> ``251d22e9`` —
because the three crop-geometry knobs that decide what pixels the model sees
entered the identity. ``ModelSpec.input_size`` folded into the ``-w`` fingerprint
(a checkpoint has an input resolution; serving the wrong one is a weights
misconfiguration); ``FaceSelectionPolicy`` gained ``interpolation_rule`` and
``crop_clip_rule`` fields. **No published number changes** — all three fields
name the *current* shipped behaviour verbatim, proven byte-identical by
``tests/test_crop_identity.py::test_the_shipped_pixels_did_not_move``. One cache
generation, no contract / RUN_SCHEMA / witness / OpenAPI change. The ``or`` in
the interpolation rule (R28) is recorded, not fixed — that would move pixels.
See ``docs/CROP_IDENTITY_B22_PLAN.md`` and ``docs/CROP_IDENTITY.md``.

**The ``.f.`` segment moved in S8 Batch 14** — ``d662e22d`` to ``19559387`` —
because the ordering rule it names changed from ``area-desc,...`` to
``short-edge-desc,area-desc,...``. Ranking now maximises the quantity the face
gate tests, so a file whose largest face clears ``min_face_px`` can no longer be
refused because a flatter box outranked it on area (R19). **No published number
changes under the shipped detector**: the two orderings are provably identical
for boxes within two pixels of square, and Haar's are within one, measured. The
segment moves anyway because the policy genuinely changed and a cache that could
not see the change would serve results computed under the other one — which is
the whole reason the ``.f.`` segment exists. See ``docs/FACE_SIZE_POLICY.md``.

**The ``.f.`` segment itself is from S8 Batch 12 and names the face-selection policy**
— the detector, the digest of its cascade asset, its ``scaleFactor`` and
``minNeighbors``, the detection floor, the ordering rule that picks ``faces[0]``,
the crop margin and crop shape, and (S8 Batch 22) the interpolation rule and the
crop clipping rule. Batch 12 measured a crop margin of 1.2 against the shipped
1.3 moving one image from ``authentic`` at p_fake 0.0136 to
``insufficient_evidence`` at 0.6597, with every other segment byte-identical, so
a cache keyed without it serves a verdict computed under a different crop. See
``docs/FACE_SELECTION_IDENTITY.md`` and ``tests/test_face_selection_identity.py``.

Note the segment moves when OpenCV reships the cascade XML, because the asset is
hashed rather than assumed. That is intended: a re-trained cascade under the same
filename changes which face is found.

**This literal is environment-specific as of S8 Batch 9, and that is the point.**
The trailing ``.d.<digest>`` segment names the decode environment — OpenCV
version, video backend, the FFmpeg libraries the video path links, OS and
machine. Batch 8 measured two video backends in one OpenCV build decoding the
same file to zero identical frames, moving ``p_fake`` enough to change 1.3% of
decisions while every other version field stayed byte-identical. A cache keyed
without this segment returns results produced by a different decoder;
``tests/test_run_identity.py`` proves that against a named double.

**The ``.d.`` segment moved in S8 Batch 20** — ``c20a064a`` to ``a97ccb68`` —
because the FFmpeg-version component grew from libavcodec alone to every FFmpeg
library on the video path (libavcodec + **libavformat** + **libswscale** +
libavutil). Batch 19 measured that libavformat, which produces
``CAP_PROP_FRAME_COUNT``, was outside the identity, so two deployments sharing a
``.d.`` digest could sample different frames and publish different verdicts under
one content-cache key (R27). This is a one-time full cache miss per file, no data
loss — precedented by Batch 9. See ``docs/DECODE_IDENTITY_COMPLETENESS.md``.

The consequence, stated plainly: **the value checked in here is correct for the
environment it was generated in and will not match a different one.** A worker
on another OS or another OpenCV build reports a different digest, and
``verify_version_alignment`` fails at startup rather than letting the two share
a cache. That is the intended behaviour and not a bug to route around — set
``DFP_MODEL_VERSION`` per deployment, or regenerate this literal for the image
you actually ship. A deployment that wants one cache namespace across mixed
decoders is asking for the failure Batch 8 measured.

**This value is owned by ``inference``, not by this module.** It is duplicated
rather than imported because importing ``inference`` from the API process pulls
in cv2 and numpy for the sake of one string, and the API is deliberately free of
both — ``workers`` is the composition root that gets to see the model.

Duplication is safe only because it is enforced: ``verify_version_alignment``
compares the two and ``test_configured_model_version_matches_the_engines`` fails
the build when they drift. Before that test existed, the API defaulted to
``1.0.0-dev`` while the engines emitted ``stub-0`` — so the result cache wrote
its content pointer under one key and read it under another, and every upload
short-circuit missed. The whole suite passed, because the fixtures pinned both
sides to the same literal and the default configuration was never exercised.

If a test tells you this value is stale, paste in the version it reports rather
than relaxing the check: a drift here is a silently broken cache.
"""


class Settings(BaseSettings):
    """Everything configurable, in one typed object."""

    model_config = SettingsConfigDict(
        env_prefix="DFP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- identity -------------------------------------------------------
    app_name: str = "Deepfake Detection Platform"
    version: str = "0.1.0"
    environment: Environment = "dev"

    # ---- routing --------------------------------------------------------
    api_prefix: str = Field(
        default="",
        description="Mount point for every route. Empty = routes at the root.",
    )
    docs_enabled: bool = Field(
        default=True, description="Serve /docs and /openapi.json. Disable in prod."
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"],
        description="Vite and Next dev servers by default.",
    )

    # ---- model / stub disclosure ----------------------------------------
    stub_scoring: bool = Field(
        default=True,
        description=(
            "True while handlers use deterministic stub scoring rather than a "
            "model. The API is real from S5 onward — this flag no longer means "
            "'fixtures', it means 'the numbers are not measurements'. Surfaced "
            "in /health so nobody quotes a stub score as a result."
        ),
    )
    mock_latency_ms: int = Field(
        default=0, ge=0, le=5000, description="Artificial delay, for testing UI loading states."
    )

    # ---- storage ---------------------------------------------------------
    storage_kind: Literal["local", "s3"] = "local"
    storage_root: Path = Field(
        default=Path("./var/storage"),
        description="Filesystem root when storage_kind is 'local'.",
    )
    storage_bucket: str | None = None
    storage_endpoint_url: str | None = Field(
        default=None, description="Set for MinIO, R2 or B2; leave unset for AWS."
    )
    storage_region: str = "us-east-1"
    signing_secret: str = Field(
        default="dev-only-insecure-signing-secret-change-me",
        min_length=16,
        description=(
            "HMAC key for signed upload URLs. The default is deliberately "
            "self-describing: a secret that announces it is insecure is harder "
            "to ship by accident than a random-looking one."
        ),
    )

    # ---- queue -----------------------------------------------------------
    queue_kind: Literal["memory", "redis"] = "memory"
    redis_url: str | None = Field(default=None, description="Required when queue_kind='redis'.")
    redis_namespace: str = "dfp"

    # ---- model ----------------------------------------------------------
    model_version: str = Field(
        default=STUB_ENGINE_VERSION,
        description=(
            "The deployed model's effective version — weights, calibration and "
            "quality gate composed. Half the result cache key and half the job "
            "idempotency key. MUST equal what the engines actually report; "
            "workers.engine_registry.verify_version_alignment enforces it."
        ),
    )

    # OR1-b (Batch 25): wired up. ``backend/worker_main.py``, the composition
    # root, reads this field and passes it into
    # ``workers.runner.verify_engine_alignment_at_startup(..., skip_version_
    # alignment=...)`` directly — it may import both ``backend`` and
    # ``workers``, unlike a bare worker process, which still cannot import
    # ``backend`` (Batch-0 layering) and keeps using the environment variable
    # ``DFP_SKIP_VERSION_ALIGNMENT`` for the same effect. The two mechanisms
    # are checked with ``or`` inside that function; this field does not
    # replace the environment variable, it adds a second, typed way to reach
    # the identical escape hatch for anyone deploying through the composition
    # root.
    skip_version_alignment: bool = Field(
        default=False,
        description=(
            "Bypass the worker startup version-alignment gate (OR1), logging a "
            "warning every boot instead of failing closed. For the one case "
            "where an operator must deploy now and regenerate model_version "
            "after. Equivalent to $DFP_SKIP_VERSION_ALIGNMENT; only observed by "
            "backend/worker_main.py, not by a bare workers-only deployment."
        ),
    )

    # ---- storage --------------------------------------------------------
    storage_base_url: str = Field(
        default="https://storage.localhost",
        description=(
            "Origin for presigned upload URLs, with no path. The storage key "
            "supplies the path, so a trailing segment here would duplicate it."
        ),
    )
    upload_url_ttl_s: int = Field(default=900, ge=60, le=86_400)
    result_ttl_s: int = Field(default=604_800, ge=3600, description="How long a result is kept.")

    # ---- limits (imported, not redeclared) ------------------------------
    max_image_bytes: int = MAX_IMAGE_BYTES
    max_video_bytes: int = MAX_VIDEO_BYTES
    max_video_duration_s: int = MAX_VIDEO_DURATION_S
    allowed_image_mime: tuple[str, ...] = ALLOWED_IMAGE_MIME
    allowed_video_mime: tuple[str, ...] = ALLOWED_VIDEO_MIME

    # ---- logging --------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = Field(
        default=True, description="JSON in every environment except an interactive dev shell."
    )

    # ---- rate limiting (keys on Principal; see backend/deps.py) ---------
    rate_limit_per_minute: int = Field(default=60, ge=1)

    @field_validator("api_prefix")
    @classmethod
    def _normalise_prefix(cls, v: str) -> str:
        """Accept ``api/v1``, ``/api/v1`` or ``/api/v1/`` and store one form."""
        v = v.strip().strip("/")
        return f"/{v}" if v else ""

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        level = v.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level {v!r}")
        return level

    @model_validator(mode="after")
    def _production_requires_real_secrets(self) -> Settings:
        """Refuse to start a production deployment on development defaults.

        Signed upload URLs are the only thing stopping an arbitrary caller from
        writing to any key, so a known secret in production is not a weak
        configuration — it is no configuration. Failing at startup is the one
        moment this is cheap to fix.
        """
        if self.environment != "prod":
            return self
        if self.signing_secret.startswith("dev-only"):
            raise ValueError("signing_secret must be set in production")
        if self.storage_kind == "s3" and not self.storage_bucket:
            raise ValueError("storage_bucket is required when storage_kind='s3'")
        if self.queue_kind == "redis" and not self.redis_url:
            raise ValueError("redis_url is required when queue_kind='redis'")
        return self

    @model_validator(mode="after")
    def _queue_backend_is_configured(self) -> Settings:
        if self.queue_kind == "redis" and not self.redis_url:
            raise ValueError("redis_url is required when queue_kind='redis'")
        return self

    # ---- component configs ----------------------------------------------

    def storage_config(self) -> "StorageConfig":
        """Map application settings onto the storage package's own config.

        The mapping exists so ``storage`` never learns that a ``DFP_`` prefix
        or an environment exists. It reads a plain model; this is where the
        translation happens.
        """
        from storage.registry import StorageConfig

        return StorageConfig(
            kind=self.storage_kind,
            signing_secret=self.signing_secret,
            public_base_url=self.storage_base_url,
            root=self.storage_root if self.storage_kind == "local" else None,
            bucket=self.storage_bucket,
            endpoint_url=self.storage_endpoint_url,
            region=self.storage_region,
        )

    def queue_config(self) -> "QueueConfig":
        from jobqueue.registry import QueueConfig

        return QueueConfig(
            kind=self.queue_kind,
            url=self.redis_url,
            namespace=self.redis_namespace,
        )

    # ---- derived --------------------------------------------------------

    @property
    def contract_version(self) -> str:
        return CONTRACT_VERSION

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"

    def status_url(self, job_id: UUID | str) -> str:
        return f"{self.api_prefix}/jobs/{job_id}"

    def result_url(self, job_id: UUID | str) -> str:
        return f"{self.api_prefix}/results/{job_id}"

    def events_url(self, job_id: UUID | str) -> str:
        """Not routed until S2, but responses already advertise it."""
        return f"{self.api_prefix}/events/{job_id}"

    def max_bytes_for(self, media_kind: str) -> int:
        return self.max_image_bytes if media_kind == "image" else self.max_video_bytes


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. Overridden in tests via ``app.dependency_overrides``."""
    return Settings()
