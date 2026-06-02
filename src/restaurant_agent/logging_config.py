"""Self-hosted structured logging (no LangSmith / paid tracing).

Provides:
  * ``configure_logging`` — one-time setup of stdlib logging with either a
    human-readable text formatter or a machine-parseable JSON formatter.
  * ``get_logger`` — module-level logger accessor.
  * ``log_event`` — emit a log record with arbitrary structured fields, which
    the JSON formatter serialises into the line. This is our lightweight stand-in
    for a tracing backend: every agent step logs an ``event`` plus timing/token
    context that you can later ``grep`` or ship to a log aggregator.

Usage::

    configure_logging(level="INFO", fmt="json")
    log = get_logger(__name__)
    log_event(log, "info", "scout.search", query="pizza", results=8, latency_ms=412)
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

# Standard LogRecord attributes; anything else in ``record.__dict__`` is treated
# as a user-supplied structured field and included in JSON output.
_RESERVED = set(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime", "taskName"}

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Render each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-friendly formatter that appends structured fields as ``key=value``."""

    _BASE = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"

    def __init__(self) -> None:
        super().__init__(self._BASE, datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        if extras:
            kv = " ".join(f"{k}={v}" for k, v in extras.items())
            base = f"{base} | {kv}"
        return base


def configure_logging(
    level: str = "INFO",
    fmt: str = "text",
    log_file: str | None = None,
) -> None:
    """Configure root logging once. Safe to call multiple times (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    formatter: logging.Formatter = JsonFormatter() if fmt == "json" else TextFormatter()

    handlers: list[logging.Handler] = []
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    handlers.append(stream)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.setLevel(level.upper())
    # Clear any pre-existing handlers (e.g. from notebooks) to avoid dupes.
    root.handlers.clear()
    for handler in handlers:
        root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: str, event: str, **fields: Any) -> None:
    """Emit a structured log line.

    ``event`` is a short dotted name (e.g. ``scout.candidates``) and ``fields``
    are arbitrary structured values attached to the record.
    """
    logger.log(
        getattr(logging, level.upper(), logging.INFO),
        event,
        extra={"event": event, **fields},
    )
