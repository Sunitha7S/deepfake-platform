"""Queue backend selection.

The only module that names a concrete queue class, mirroring
``storage/registry.py``. Everything else depends on ``QueueBackend``, so moving
from in-process to Redis is a config change here and nowhere else.

``QueueConfig`` is a plain pydantic model, not a settings object: this package
must not read the environment or know a ``DFP_`` prefix exists. The application
maps its own settings into this and passes it in, keeping the dependency
pointing the right way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jobqueue.base import QueueBackend
from jobqueue.memory import InMemoryQueue
from jobqueue.policy import RetryPolicy
from jobqueue.redis import RedisQueue
from jobqueue.sqlite import SQLiteQueue

BackendKind = Literal["memory", "sqlite", "redis"]

SHARED_KINDS: frozenset[str] = frozenset({"sqlite", "redis"})
"""Backends a *separate* process can reach. ``memory`` is not one: the API and
a worker each build their own, and a job enqueued in one is invisible to the
other. ``backend.worker_main`` refuses to start on anything outside this set."""

IMAGE_QUEUE = "image"
VIDEO_QUEUE = "video"

QUEUE_NAMES: tuple[str, ...] = (IMAGE_QUEUE, VIDEO_QUEUE)
"""Two queues, sized independently.

A single pool lets one 60-second video hold a worker while image requests starve
behind it. The names live here rather than in the application so that a worker,
a dashboard and an autoscaler all read them from one place.
"""

QUEUE_DEFAULTS: dict[str, RetryPolicy] = {
    # Image work is fast and cheap to retry, so a short backoff gets a transient
    # blip out of the way quickly.
    IMAGE_QUEUE: RetryPolicy(max_attempts=3, base_delay_s=2.0, max_delay_s=30.0),
    # Video work is minutes of GPU time. Retrying it aggressively burns the
    # capacity healthy jobs need, so back off harder and give up sooner.
    VIDEO_QUEUE: RetryPolicy(max_attempts=2, base_delay_s=10.0, max_delay_s=120.0),
}
"""Per-queue retry policy. The asymmetry is deliberate, not an oversight."""


class QueueConfig(BaseModel):
    """How to build a queue backend. Carries no environment knowledge."""

    model_config = ConfigDict(extra="forbid")

    kind: BackendKind = "memory"
    policy: RetryPolicy = Field(default_factory=RetryPolicy)

    # redis
    url: str | None = Field(default=None, description="Required when kind is 'redis'.")
    namespace: str = "dfp"

    # sqlite
    path: Path | None = Field(default=None, description="Required when kind is 'sqlite'.")

    @model_validator(mode="after")
    def _required_fields_per_kind(self) -> QueueConfig:
        if self.kind == "redis" and not self.url:
            raise ValueError("url is required when kind='redis'")
        if self.kind == "sqlite" and self.path is None:
            raise ValueError("path is required when kind='sqlite'")
        return self

    def __repr__(self) -> str:
        """Never render the URL: it can carry a password."""
        return f"QueueConfig(kind={self.kind!r}, namespace={self.namespace!r})"

    __str__ = __repr__


def build_queue(config: QueueConfig) -> QueueBackend:
    """Return a backend. Typed as the protocol, never as a concrete class."""
    if config.kind == "memory":
        return InMemoryQueue(policy=config.policy)
    if config.kind == "sqlite":
        assert config.path is not None  # enforced by QueueConfig
        return SQLiteQueue(config.path, policy=config.policy)
    return RedisQueue(
        url=config.url or "",
        namespace=config.namespace,
        policy=config.policy,
    )


def policy_for(queue: str) -> RetryPolicy:
    """Retry policy for a named queue, falling back to the default."""
    return QUEUE_DEFAULTS.get(queue, RetryPolicy())
