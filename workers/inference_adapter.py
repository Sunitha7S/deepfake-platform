"""Translation between the inference layer's vocabulary and the worker's.

The inference package raises errors describing *what went wrong with the media
or the model*. The worker needs to know something different: *is this worth
retrying*. Only the worker can answer that, because retryability is a property
of the deployment's economics rather than of the error — which is exactly why
``inference`` does not import ``workers.types`` and define the answer itself.

**This module is the only place that mapping exists.** Scattering it would let
two call sites disagree about whether a corrupt file is worth a second attempt,
and the disagreement would be invisible until one of them burned a queue.

    UnreadableMedia   → PermanentError   the bytes will not decode next time
    UnsupportedMedia  → PermanentError   the file is out of scope, permanently
    ValidationError   → PermanentError   the produced bundle is malformed; a
                                         retry produces the same malformed one
    InferenceError    → PermanentError   conservative default for the base class
    ModelUnavailable  → TransientError   the deployment is broken, not the input

``ModelUnavailable`` is checked before ``InferenceError`` because it subclasses
it; the reverse order would classify every missing-weights failure as permanent
and dead-letter a queue's worth of perfectly good jobs during a bad deploy.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from pydantic import ValidationError

from inference.protocol import (
    InferenceError,
    ModelUnavailable,
    UnreadableMedia,
    UnsupportedMedia,
)
from workers.types import HandlerError, PermanentError, TransientError


def to_handler_error(exc: BaseException) -> HandlerError | None:
    """Translate an inference exception. Returns None for anything unrecognised.

    Returning None rather than guessing matters: an unrecognised exception falls
    through to the runner, which treats it as transient by design. Coercing it
    to permanent here would dead-letter real work on the first unhandled
    ``KeyError`` in a model backend.
    """
    if isinstance(exc, ModelUnavailable):
        return TransientError(
            "The analysis model is not available.",
            detail=getattr(exc, "detail", None) or str(exc),
        )
    if isinstance(exc, UnreadableMedia):
        return PermanentError(
            "That file could not be read. It may be corrupt or incompletely uploaded.",
            detail=getattr(exc, "detail", None) or str(exc),
        )
    if isinstance(exc, UnsupportedMedia):
        return PermanentError(
            str(exc),
            detail=getattr(exc, "detail", None),
        )
    if isinstance(exc, ValidationError):
        # The engine produced a bundle the contract rejects. That is a bug in
        # the engine, not a condition of the input, and re-running it produces
        # the same invalid bundle — so it is permanent even though it is ours.
        return PermanentError(
            "The analysis produced a malformed result.",
            detail=f"{exc.error_count()} contract violation(s): {str(exc).splitlines()[0]}",
        )
    if isinstance(exc, InferenceError):
        return PermanentError(
            str(exc),
            detail=getattr(exc, "detail", None),
        )
    return None


@contextlib.contextmanager
def translated() -> Iterator[None]:
    """Run a block, translating inference errors into worker errors.

    Anything unrecognised propagates untouched, so the runner's own
    classification still applies to it.
    """
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 - translation is the whole point
        mapped = to_handler_error(exc)
        if mapped is None:
            raise
        raise mapped from exc
