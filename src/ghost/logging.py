"""Structured logging for Ghost platform.

Uses structlog for consistent, structured logging across all components.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

from ghost.config import get_config

if TYPE_CHECKING:
    from pathlib import Path


def setup_logging(log_file: str | Path | None = None) -> None:
    """Configure structured logging for the application."""
    config = get_config()

    log_level = getattr(logging, config.log_level.upper(), logging.INFO)

    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
        if not sys.stdout.isatty()
        else structlog.dev.ConsoleRenderer(colors=True),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
    )

    for logger_name in ["urllib3", "requests", "matplotlib"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)
