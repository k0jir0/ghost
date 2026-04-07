"""Unit tests for ghost.logging."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

from ghost import logging as ghost_logging


def test_setup_logging_configures_structlog_and_stdlib_handlers(
    ghost_config,
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_structlog = MagicMock()
    fake_stdout = MagicMock()
    fake_stdout.isatty.return_value = False
    configure_mock = MagicMock()
    fake_structlog.configure = configure_mock

    monkeypatch.setattr(ghost_logging, "structlog", fake_structlog)
    monkeypatch.setattr(ghost_logging.sys, "stdout", fake_stdout)

    basic_config_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        ghost_logging.logging,
        "basicConfig",
        lambda **kwargs: basic_config_calls.append(kwargs),
    )

    log_path = tmp_path / "ghost.log"
    ghost_logging.setup_logging(log_file=log_path)

    configure_mock.assert_called_once()
    assert basic_config_calls
    handlers = basic_config_calls[0]["handlers"]
    assert len(handlers) == 2
    assert isinstance(handlers[0], logging.StreamHandler)
    assert isinstance(handlers[1], logging.FileHandler)


def test_get_logger_returns_structlog_logger(monkeypatch) -> None:
    fake_logger = MagicMock()
    fake_structlog = MagicMock()
    fake_structlog.get_logger.return_value = fake_logger
    monkeypatch.setattr(ghost_logging, "structlog", fake_structlog)

    result = ghost_logging.get_logger("ghost.tests")

    assert result is fake_logger
    fake_structlog.get_logger.assert_called_once_with("ghost.tests")
