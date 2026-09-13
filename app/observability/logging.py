"""Process-wide structured logging configuration.

Logs are emitted to stdout only; no Elasticsearch sink is configured here.
"""
from __future__ import annotations

import logging
import os

import structlog


def _renderer():
    if (os.getenv("ENV", "dev") or "dev").lower() in {"prod", "production"}:
        return structlog.processors.JSONRenderer()
    return structlog.dev.ConsoleRenderer()


def _level() -> int:
    name = (os.getenv("LOG_LEVEL", "INFO") or "INFO").upper()
    return logging.getLevelNamesMapping().get(name, logging.INFO)


def setup_logging() -> None:
    """Configure structlog idempotently for the current process."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            _renderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_level()),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def close_logging() -> None:
    """Compatibility hook for application shutdown."""
