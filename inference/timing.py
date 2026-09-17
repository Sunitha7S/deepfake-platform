"""The timing contract, and the one object that implements it.

WHAT EACH FIELD MEANS
---------------------
``Timing`` is published in every bundle and read by operators sizing hardware.
Before this module the two pipelines filled it by two different conventions and
neither measured what its field names claimed:

    queued_ms      time from submission to the start of analysis
    preprocess_ms  decode, frame sampling, face detection, tensor preparation —
                   everything before the first model call
    inference_ms   model forward passes, and nothing else
    explain_ms     artifact generation (zero until explanations ship)
    total_ms       wall clock for the whole ``analyze_*`` call

THE TWO FAILURES THIS REPLACES
------------------------------
**Phases were derived by subtracting cumulative elapsed times.** The image path
computed ``preprocess_ms`` as elapsed-so-far after decoding, then
``inference_ms`` as elapsed-so-far *minus* that. Face detection ran between the
two measurements, so its cost belonged to neither: on a 900x900 image with the
Haar detector, a measured 1268 ms of a 1431 ms job — 89% of it — appeared in no
field at all, while ``total_ms`` reported the truth. The video path used the
same subtraction the other way round, so ``inference_ms`` silently absorbed
fusion and bundle construction along with the model calls.

**Each phase now times itself.** A phase is measured from its own start to its
own end and accumulated under its own name. Nothing is inferred by subtraction,
so no interval can be attributed twice and none can go missing.

WHY PHASES FLOOR AND TOTAL CEILS
--------------------------------
Rounding decides whether the published numbers can contradict each other.
Flooring each phase makes every part no larger than the interval it measured;
ceiling the total makes it no smaller. Since the true parts sum to no more than
the true total, the published parts sum to no more than the published total —
for every input, with no tolerance and no epsilon. That is what lets
``Timing`` assert the relationship instead of hoping for it.
"""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from collections.abc import Iterator
from contextlib import contextmanager

from contracts.common import Timing

PREPROCESS = "preprocess"
INFERENCE = "inference"
EXPLAIN = "explain"


class PhaseTimer:
    """Wall-clock accounting for one analysis.

    Shared by both pipelines so the contract above is implemented once. A phase
    may be entered more than once — the video path scores in batches — and the
    intervals accumulate.
    """

    __slots__ = ("_started", "_elapsed", "_queued_ms")

    def __init__(self, submitted_at=None) -> None:
        self._started = time.perf_counter()
        self._elapsed: dict[str, float] = {}
        # Sampled here, not at ``to_timing``. The queue wait ends the moment
        # analysis begins, so measuring it at the end of the job adds the whole
        # analysis to it — a 1.5 s wait on a 1.4 s job reported 2.9 s.
        self._queued_ms = 0 if submitted_at is None else queued_ms_since(submitted_at)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time a block and add it to ``name``.

        ``finally`` rather than a plain exit: a decode that raises has still
        consumed the time, and a pipeline that reports timing on its error paths
        needs that interval to be recorded rather than lost.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self._elapsed[name] = self._elapsed.get(name, 0.0) + (
                time.perf_counter() - started
            )

    def seconds(self, name: str) -> float:
        return self._elapsed.get(name, 0.0)

    def ms(self, name: str) -> int:
        """Floored. See the module docstring for why."""
        return int(self._elapsed.get(name, 0.0) * 1000.0)

    @property
    def total_ms(self) -> int:
        """Ceiled. Measured to *now*, so it includes everything so far."""
        return int(math.ceil((time.perf_counter() - self._started) * 1000.0))

    @property
    def queued_ms(self) -> int:
        """Queue wait, sampled when this timer was created."""
        return self._queued_ms

    def to_timing(self, *, queued_ms: int | None = None) -> Timing:
        """Build the published object. Call last, so ``total_ms`` covers the job."""
        return Timing(
            queued_ms=max(0, self._queued_ms if queued_ms is None else queued_ms),
            preprocess_ms=self.ms(PREPROCESS),
            inference_ms=self.ms(INFERENCE),
            explain_ms=self.ms(EXPLAIN),
            total_ms=self.total_ms,
        )


def queued_ms_since(submitted_at, *, now=None) -> int:
    """How long the job waited before analysis began.

    Clamped at zero. The submission timestamp is set by the API and read by a
    worker that may be on another host, so a small negative value means clock
    skew rather than a job that finished before it started — and a negative
    duration is not representable in the contract anyway.
    """
    current = now or datetime.now(UTC)
    return max(0, int((current - submitted_at).total_seconds() * 1000.0))
