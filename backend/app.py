"""Application factory.

``create_app()`` rather than a module-level ``app``: tests need to build an
instance with overridden settings, and a module-level singleton reads the
environment at import time, which makes that impossible without monkeypatching.

The OpenAPI customisation at the bottom is what makes the success criterion
reachable. FastAPI derives the schema from the same pydantic models the workers
will produce, and ``_inject_examples`` attaches the real fixture payloads from
``contracts.examples`` to it. A frontend developer opening ``/docs`` therefore
sees, for every endpoint, a payload that is byte-identical to what the finished
system will return.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from backend.config import Settings, get_settings
from backend.errors import install_exception_handlers
from backend.logging import configure_logging, get_logger, request_context
from backend.service import AnalysisService
from jobqueue.base import QueueBackend
from jobqueue.registry import build_queue
from storage.base import StorageBackend
from storage.registry import build_storage
from workers.io import StorageResultSink
from workers.protocol import ResultSink
from contracts.enums import CONTRACT_VERSION
from contracts.examples import EXAMPLES

log = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

DESCRIPTION = """
Image and video deepfake detection.

**Scores are not measurements yet.** The upload, queue, worker and result
pipeline is real, but analysis uses deterministic stub scoring rather than a
model, so a verdict reflects the file's hash and nothing about its authenticity.
Check `stub_scoring` on `/health` before quoting any number.

### Three verdicts, not two

`insufficient_evidence` is a **successful** result, returned with HTTP 200 and a
null `p_fake`. It means the input carried too little information to judge — a
face too small, an image too compressed. Render it as an answer, never as a
failure.

### Errors

Every non-2xx body is an `ErrorResponse`. `error.retryable` tells you whether
repeating the request could succeed; when it is false, retrying will fail the
same way.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    log.info(
        "starting",
        extra={
            "app": settings.app_name,
            "version": settings.version,
            "environment": settings.environment,
            "contract_version": CONTRACT_VERSION,
            "stub_scoring": settings.stub_scoring,
        },
    )
    app.state.started_at = time.monotonic()

    # Report the real condition of the dependencies at boot rather than waiting
    # for the first request to discover a misconfiguration.
    ready, depths, problems = await app.state.service.health()
    log.info(
        "dependencies checked",
        extra={
            "ready": ready,
            "queue_backend": repr(app.state.queue),
            "queues": depths,
            "problems": problems or None,
        },
    )

    # OR1 (Batch 23) — a stdlib-only *format* check on the configured
    # model_version. The API process is deliberately free of cv2 and numpy, so it
    # cannot import ``inference`` to run the full engine-alignment check; that is
    # the worker's job (workers.runner.verify_engine_alignment_at_startup). What
    # the API *can* do without importing anything forbidden is reject a version
    # that is not a composed, environment-pinned string — the shape of the
    # failure that has actually occurred (``1.0.0-dev``, a bare weights string).
    from contracts.version import (
        is_composed,
        is_environment_pinned,
        model_identity_of,
        parse_model_version,
    )

    mv = settings.model_version
    problem: str | None = None
    if not is_composed(mv):
        problem = "carries no calibration segment (not a composed model version)"
    elif not is_environment_pinned(mv):
        problem = "carries no decode-environment (.d.) segment"
    else:
        try:
            if parse_model_version(mv).compose() != model_identity_of(mv):
                problem = "does not round-trip through parse_model_version"
        except ValueError as exc:
            problem = f"does not parse: {exc}"
    if problem is not None:
        raise RuntimeError(
            f"configured model_version {mv!r} {problem}. It must equal what the "
            "engines report — set DFP_MODEL_VERSION (and backend.config."
            "STUB_ENGINE_VERSION) to a composed, environment-pinned version. The "
            "worker startup gate performs the full cryptographic check."
        )
    log.info("model_version format check passed", extra={"model_version": mv})

    # A later stage loads model weights here and leaves readiness failing until
    # they are resident, so a deploy that cannot load them takes no traffic.
    yield
    log.info("shutting down")


def create_app(
    settings: Settings | None = None,
    *,
    storage: StorageBackend | None = None,
    queue: QueueBackend | None = None,
    sink: ResultSink | None = None,
) -> FastAPI:
    """Build an application.

    Every dependency is typed as its protocol and injectable. Production passes
    none of them and gets what the settings describe; a test passes an in-memory
    queue and a temporary storage root and gets the same code paths.
    """
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    storage = storage if storage is not None else build_storage(settings.storage_config())
    queue = queue if queue is not None else build_queue(settings.queue_config())
    sink = sink if sink is not None else StorageResultSink(storage)

    app.state.settings = settings
    app.state.storage = storage
    app.state.queue = queue
    app.state.sink = sink
    app.state.service = AnalysisService(
        settings=settings, storage=storage, queue=queue, sink=sink
    )
    app.state.started_at = time.monotonic()

    _install_middleware(app, settings)
    install_exception_handlers(app)
    _register_routes(app, settings)
    _customise_openapi(app, settings)
    return app


def _install_middleware(app: FastAPI, settings: Settings) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        # PUT is required by the local-dev upload receiver. Without it a browser
        # preflight rejects the upload before it is attempted, and the failure
        # looks like a network error rather than a CORS policy.
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )

    @app.middleware("http")
    async def _request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Assign a request id, log the access line, time the handler.

        The id is echoed in a header so a user reporting a problem can quote it
        and an operator can find the exact request.
        """
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        started = time.perf_counter()
        with request_context(rid):
            response = await call_next(request)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            response.headers[REQUEST_ID_HEADER] = rid
            # The path is logged, never the query string — a query string can
            # carry values a caller should not have put there.
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "route": request.url.path,
                    "status": response.status_code,
                    "duration_ms": elapsed_ms,
                },
            )
            return response


def _register_routes(app: FastAPI, settings: Settings) -> None:
    """Imported at call time, not module scope, so ``import backend.app`` stays
    cheap and no import cycle can form between routes and the factory.

    ``system_router`` is mounted without the prefix on purpose — see
    ``backend/routes/__init__.py``.
    """
    from backend.routes import api_router, system_router

    app.include_router(system_router)
    app.include_router(api_router, prefix=settings.api_prefix)


def _customise_openapi(app: FastAPI, settings: Settings) -> None:
    def openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema["info"]["x-contract-version"] = CONTRACT_VERSION
        schema["info"]["x-stub-scoring"] = settings.stub_scoring
        _inject_examples(schema)

        # Error documentation is assembled here rather than in each route: the
        # phantom 422 must go before components are hoisted, or an unreachable
        # response would be promoted into a shared component and look canonical.
        from backend.openapi import (
            promote_error_responses,
            strip_phantom_validation_errors,
        )

        strip_phantom_validation_errors(schema)
        promote_error_responses(schema)
        app.openapi_schema = schema
        return schema

    app.openapi = openapi  # type: ignore[method-assign]


def _inject_examples(schema: dict[str, Any]) -> None:
    """Attach fixture payloads to every schema component that has them.

    Examples are built as real model instances in ``contracts.examples`` and
    re-validated in CI, so a payload shown in ``/docs`` is always one the server
    would accept.
    """
    components = schema.get("components", {}).get("schemas", {})
    for model, entries in EXAMPLES.items():
        target = components.get(model.__name__)
        if target is None:
            continue
        payloads = [fn().model_dump(mode="json") for _, fn in entries]
        if payloads:
            target["examples"] = payloads


# No module-level ``app``. A singleton built at import time reads the
# environment before a test can override it, and forces route imports during
# ``import backend.app``. Run with the factory flag instead:
#
#     uvicorn backend.app:create_app --factory --reload
