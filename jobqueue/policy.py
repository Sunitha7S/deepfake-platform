"""Retry policy.

Two decisions live here, and the first is the one that matters most.

**Retryability is supplied by the caller, never inferred.** The queue cannot
know whether a failure was a GPU that ran out of memory or a video file with no
moov atom, and guessing would be worse than useless: retrying a corrupt file
produces the identical failure three times and burns capacity healthy jobs
need. So ``nack()`` takes ``retryable`` as an argument. The application layer
already owns that taxonomy — it is exactly the transient/deterministic split in
``contracts.enums`` — and this package stays ignorant of it.

**Backoff is exponential with jitter.** Without jitter, a hundred jobs failing
on the same transient outage all retry at the same instant, hit the recovering
dependency together, and fail together — a thundering herd that turns a brief
blip into a sustained one. Full jitter costs one call to ``random`` and removes
the synchronisation entirely.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from pydantic import Field, model_validator

from jobqueue.types import Base


class RetryPolicy(Base):
    """How a failed job is rescheduled, and when it is given up on."""

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Total attempts including the first. 1 means never retry.",
    )
    base_delay_s: float = Field(default=2.0, gt=0.0, description="Delay before the first retry.")
    max_delay_s: float = Field(
        default=300.0,
        gt=0.0,
        description="Ceiling. Without one, attempt 15 schedules itself for next week.",
    )
    multiplier: float = Field(default=3.0, ge=1.0)
    jitter: bool = Field(
        default=True,
        description="Full jitter. Disable only in tests that assert exact delays.",
    )

    @model_validator(mode="after")
    def _ceiling_above_floor(self) -> RetryPolicy:
        if self.max_delay_s < self.base_delay_s:
            raise ValueError("max_delay_s must be >= base_delay_s")
        return self

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> timedelta:
        """Backoff before ``attempt`` (1-based: attempt 1 is the first retry)."""
        if attempt < 1:
            raise ValueError("attempt is 1-based")
        raw = self.base_delay_s * (self.multiplier ** (attempt - 1))
        capped = min(raw, self.max_delay_s)
        if self.jitter:
            # Full jitter: uniform in [0, capped]. Decorrelates retries
            # completely, where "equal jitter" still leaves half the delay
            # synchronised across every failing job.
            capped = (rng or random).uniform(0.0, capped)
        return timedelta(seconds=capped)

    def should_retry(
        self, *, attempts: int, retryable: bool, max_attempts: int | None = None
    ) -> bool:
        """``attempts`` is the count already completed, including the one that just failed.

        ``max_attempts`` overrides the policy's own budget with the job's. A
        caller that passes ``max_attempts`` to ``enqueue`` has said something
        specific about that job — usually that it is expensive or that a second
        run would be harmful — and letting the policy default silently outrank
        it makes the parameter decorative.
        """
        if not retryable:
            return False
        return attempts < (max_attempts if max_attempts is not None else self.max_attempts)

    def next_attempt_at(
        self, now: datetime, *, attempts: int, rng: random.Random | None = None
    ) -> datetime:
        return now + self.delay_for(attempts, rng=rng)


NO_RETRY = RetryPolicy(max_attempts=1)
"""For work that must never run twice, whatever the failure."""

DEFAULT_POLICY = RetryPolicy()
"""Three attempts, ~2s then ~6s, jittered. Suits image jobs."""

VIDEO_POLICY = RetryPolicy(max_attempts=2, base_delay_s=10.0, max_delay_s=120.0)
"""Fewer, slower retries.

A video job costs GPU-minutes, so a retry is expensive and a tight loop of them
is worse than a clean failure the user can act on.
"""
