"""Shared FastAPI dependencies.

Two jobs:

**Hand routes the service, not its parts.** Routes depend on
``AnalysisService`` and never on a storage backend, a queue or a sink directly.
That keeps the use cases in one testable place and stops a route quietly
growing its own opinion about, say, what a missing result means.

**Make authentication a swap, not a migration.** ``get_principal`` returns an
anonymous principal today. Every route already takes it and rate limiting
already keys on it, so adding real accounts means replacing one function — not
editing every route signature later under time pressure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import Depends, Request
from pydantic import BaseModel, ConfigDict

from backend.config import Settings

if TYPE_CHECKING:  # pragma: no cover - import only for type names
    from backend.service import AnalysisService

PrincipalKind = Literal["anonymous", "user", "service"]


class Principal(BaseModel):
    """Who is making this request.

    Anonymous today. The shape already carries what authorisation will need, so
    call sites written now keep working when real identities arrive.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = "anonymous"
    kind: PrincipalKind = "anonymous"
    scopes: tuple[str, ...] = ()

    @property
    def is_anonymous(self) -> bool:
        return self.kind == "anonymous"

    @property
    def rate_limit_key(self) -> str:
        """What per-caller limits count against.

        Anonymous callers will key on client IP once a limiter exists; the
        indirection is here so that change lands in one place.
        """
        return f"{self.kind}:{self.id}"


ANONYMOUS = Principal()


async def get_principal(request: Request) -> Principal:
    """Resolve the caller.

    Returns anonymous unconditionally for now. A later version reads a bearer
    token here and raises ``AppError(ErrorCode.INVALID_REQUEST)`` on a bad one.
    It already accepts ``request`` so the signature does not change when it
    starts reading headers.
    """
    return ANONYMOUS


def get_app_settings(request: Request) -> Settings:
    """Settings from app state, not the module-level cache.

    Reading ``app.state`` rather than calling ``get_settings()`` is what lets a
    test build an app with overridden settings and have routes actually see
    them.
    """
    return request.app.state.settings


def get_service(request: Request) -> "AnalysisService":
    """The service assembled by the app factory."""
    return request.app.state.service


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
PrincipalDep = Annotated[Principal, Depends(get_principal)]
ServiceDep = Annotated["AnalysisService", Depends(get_service)]
