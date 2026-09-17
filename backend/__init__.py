"""FastAPI backend.

The API runs on the real platform: uploads land in ``storage``, jobs go through
``jobqueue``, workers produce results, and every response is read back from those
same authorities. No fixtures, and no state of its own.

The one thing still standing in for the finished system is **scoring**: handlers
use deterministic stub analysis rather than a model, so a verdict reflects the
file's hash and nothing about its authenticity. ``stub_scoring`` on ``/health``
says so, and it is the only claim here that a later stage changes.

``JobStatus`` is derived per request from the queue envelope plus the existence
of a result. There is deliberately no status record: a stored copy would be a
fourth authority nothing keeps in sync, and it fails silently — the queue
dead-letters a job while the record still reads "running".
"""

from backend.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
