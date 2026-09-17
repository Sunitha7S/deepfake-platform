"""Structured logging with request correlation, and no personal data.

Two properties this module guarantees:

**Correlation.** Every log line emitted while handling a request carries the
same ``request_id``, and lines emitted while processing a job carry its
``job_id``. Both travel through ``contextvars``, so nothing has to be threaded
through call signatures.

**No personal data, by construction.** We handle photographs of people. Log
retention is therefore data retention, and a filename or an EXIF field in a log
line is personal data sitting in a system with a different retention policy from
the one that governs uploads. ``PiiFilter`` drops known-dangerous keys rather
than trusting every future call site to remember. It is a backstop, not a
licence to pass personal data to the logger.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)

REDACTED = "[redacted]"

PII_KEYS: frozenset[str] = frozenset(
    {
        "filename",
        "filename_hint",
        "file_name",
        "original_filename",
        "path",
        "storage_path",
        "exif",
        "gps",
        "camera_make",
        "camera_model",
        "email",
        "user_agent",
        "ip",
        "client_ip",
        "authorization",
        "cookie",
        "token",
        "signature",
    }
)
"""Keys never written to a log, whatever a call site passes.

``sha256`` and ``storage_key`` are deliberately absent: a content hash and an
opaque storage key are how an operator correlates a report with a job, and
neither reveals anything about the person in the image.
"""

_STANDARD = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }
)


class PiiFilter(logging.Filter):
    """Redacts personal fields from log record extras."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key in list(record.__dict__):
            if key in _STANDARD:
                continue
            if key.lower() in PII_KEYS:
                record.__dict__[key] = REDACTED
        return True


class ContextFilter(logging.Filter):
    """Attaches the current request and job ids to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.job_id = job_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line. What log aggregators want."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if getattr(record, "request_id", None):
            payload["request_id"] = record.request_id
        if getattr(record, "job_id", None):
            payload["job_id"] = record.job_id

        for key, value in record.__dict__.items():
            if key in _STANDARD or key in ("request_id", "job_id"):
                continue
            if key.startswith("_"):
                continue
            payload[key] = _safe(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Readable single-line output for an interactive dev shell."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S")
        rid = getattr(record, "request_id", None)
        tag = f" [{rid[:8]}]" if rid else ""
        base = f"{ts} {record.levelname:<7}{tag} {record.name}: {record.getMessage()}"
        extras = {
            k: _safe(v)
            for k, v in record.__dict__.items()
            if k not in _STANDARD and k not in ("request_id", "job_id") and not k.startswith("_")
        }
        if extras:
            base += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def _safe(value: Any) -> Any:
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, list | tuple):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    return str(value)


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Install handlers on the root logger. Idempotent."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else ConsoleFormatter())
    handler.addFilter(ContextFilter())
    handler.addFilter(PiiFilter())

    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn ships its own handlers; route them through ours so every line in
    # the process has the same shape and the same redaction.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def request_context(request_id: str) -> Iterator[None]:
    token = request_id_var.set(request_id)
    try:
        yield
    finally:
        request_id_var.reset(token)


@contextmanager
def job_context(job_id: str | UUID) -> Iterator[None]:
    token = job_id_var.set(str(job_id))
    try:
        yield
    finally:
        job_id_var.reset(token)
