"""Unit tests for ghost.health_monitor."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ghost.config import GhostConfig
from ghost.health_monitor import HealthMonitor


class TestHealthMonitor:
    def test_healthy_snapshot_reports_cache_sizes(self, tmp_path: Path) -> None:
        cfg = GhostConfig(
            model_cache_dir=tmp_path / "models",
            data_cache_dir=tmp_path / "data",
            gpu_enabled=False,
        )
        cfg.ensure_directories()
        (cfg.model_cache_dir / "model.bin").write_bytes(b"1234")
        (cfg.data_cache_dir / "dataset.bin").write_bytes(b"12")

        monitor = HealthMonitor(config=cfg)

        with patch("ghost.health_monitor.psutil") as psutil_mock:
            psutil_mock.virtual_memory.return_value = SimpleNamespace(percent=42.0)
            snapshot = monitor.check_resources()

        assert snapshot.status == "healthy"
        assert snapshot.model_cache_size_bytes == 4
        assert snapshot.data_cache_size_bytes == 2

    def test_memory_pressure_marks_snapshot_degraded(self, tmp_path: Path) -> None:
        cfg = GhostConfig(
            model_cache_dir=tmp_path / "models",
            data_cache_dir=tmp_path / "data",
            gpu_enabled=False,
            system_memory_threshold=0.50,
        )
        cfg.ensure_directories()

        monitor = HealthMonitor(config=cfg)

        with patch("ghost.health_monitor.psutil") as psutil_mock:
            psutil_mock.virtual_memory.return_value = SimpleNamespace(percent=80.0)
            snapshot = monitor.check_resources()

        assert snapshot.status == "degraded"
        assert any(issue.code == "system-memory-high" for issue in snapshot.issues)
        assert monitor.recommended_batch_size(64, snapshot) == 32

    def test_missing_psutil_returns_warning(self, tmp_path: Path) -> None:
        cfg = GhostConfig(
            model_cache_dir=tmp_path / "models",
            data_cache_dir=tmp_path / "data",
            gpu_enabled=False,
        )
        cfg.ensure_directories()

        monitor = HealthMonitor(config=cfg)

        with patch("ghost.health_monitor.psutil", None):
            snapshot = monitor.check_resources()

        assert snapshot.status == "warning"
        assert any(issue.code == "psutil-unavailable" for issue in snapshot.issues)