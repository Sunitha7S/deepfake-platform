"""Router assembly.

Two routers, and the split is operational rather than cosmetic:

``system_router`` is mounted at the **root, never under ``api_prefix``**. A
liveness or readiness probe is configured once in a load balancer or a
deployment manifest, and having ``/health`` relocate to ``/api/v1/health``
because someone set an environment variable is exactly the kind of change that
takes an instance out of rotation at an inconvenient moment.

``api_router`` carries everything a client calls and moves with the prefix, so
the API can be versioned without touching route definitions.
"""

from fastapi import APIRouter

from backend.routes import analyze, health, jobs, results, uploads

system_router = APIRouter()
system_router.include_router(health.router)

api_router = APIRouter()
api_router.include_router(uploads.router)
api_router.include_router(analyze.router)
api_router.include_router(jobs.router)
api_router.include_router(results.router)

__all__ = ["api_router", "system_router"]
