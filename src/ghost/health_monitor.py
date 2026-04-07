"""Health and resource monitoring for Ghost.

Tracks system memory, optional GPU memory, and cache directory usage so the
training pipeline and MCP layer can react before the process destabilizes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any, Literal

from ghost.config import GhostConfig, get_config

try:
    import psutil
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is absent
    torch = None

HealthStatus = Literal["healthy", "warning", "degraded"]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class HealthIssue:
    """A health issue found during resource sampling."""

    severity: Literal["warning", "degraded"]
    code: str
    message: str


@dataclass
class ResourceSnapshot:
    """Point-in-time health sample for Ghost runtime resources."""

    status: HealthStatus
    checked_at: str
    system_memory_ratio: float | None
    gpu_memory_ratio: float | None
    model_cache_size_bytes: int
    data_cache_size_bytes: int
    issues: list[HealthIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the snapshot."""
        return {
            "status": self.status,
            "checked_at": self.checked_at,
            "system_memory_ratio": self.system_memory_ratio,
            "system_memory_percent": None
            if self.system_memory_ratio is None
            else round(self.system_memory_ratio * 100, 2),
            "gpu_memory_ratio": self.gpu_memory_ratio,
            "gpu_memory_percent": None
            if self.gpu_memory_ratio is None
            else round(self.gpu_memory_ratio * 100, 2),
            "model_cache_size_bytes": self.model_cache_size_bytes,
            "data_cache_size_bytes": self.data_cache_size_bytes,
            "issues": [asdict(issue) for issue in self.issues],
        }


class HealthMonitor:
    """Collect and interpret resource health for the Ghost runtime."""

    def __init__(self, config: GhostConfig | None = None):
        self.config = config or get_config()
        self._last_snapshot: ResourceSnapshot | None = None
        self._last_checked_monotonic: float | None = None

    def check_resources(self, *, force: bool = False) -> ResourceSnapshot:
        """Collect the current resource state and evaluate configured thresholds."""
        now = time.monotonic()
        if (
            not force
            and self._last_snapshot is not None
            and self._last_checked_monotonic is not None
            and now - self._last_checked_monotonic < self.config.health_check_interval
        ):
            return self._last_snapshot

        issues: list[HealthIssue] = []
        system_memory_ratio = self._get_system_memory_ratio(issues)
        gpu_memory_ratio = self._get_gpu_memory_ratio(issues)

        if (
            system_memory_ratio is not None
            and system_memory_ratio >= self.config.system_memory_threshold
        ):
            issues.append(
                HealthIssue(
                    severity="degraded",
                    code="system-memory-high",
                    message=(
                        "System memory usage "
                        f"{system_memory_ratio:.1%} exceeded the configured threshold "
                        f"{self.config.system_memory_threshold:.1%}."
                    ),
                )
            )

        if gpu_memory_ratio is not None and gpu_memory_ratio >= self.config.gpu_memory_threshold:
            issues.append(
                HealthIssue(
                    severity="degraded",
                    code="gpu-memory-high",
                    message=(
                        "GPU memory usage "
                        f"{gpu_memory_ratio:.1%} exceeded the configured threshold "
                        f"{self.config.gpu_memory_threshold:.1%}."
                    ),
                )
            )

        snapshot = ResourceSnapshot(
            status=self._derive_status(issues),
            checked_at=_utc_now_iso(),
            system_memory_ratio=system_memory_ratio,
            gpu_memory_ratio=gpu_memory_ratio,
            model_cache_size_bytes=self._directory_size(self.config.model_cache_dir),
            data_cache_size_bytes=self._directory_size(self.config.data_cache_dir),
            issues=issues,
        )
        self._last_snapshot = snapshot
        self._last_checked_monotonic = now
        return snapshot

    def recommended_batch_size(
        self,
        current_batch_size: int,
        snapshot: ResourceSnapshot | None = None,
    ) -> int:
        """Suggest a smaller batch size when memory pressure is present."""
        active_snapshot = snapshot or self.check_resources()
        has_memory_pressure = any(
            issue.code in {"system-memory-high", "gpu-memory-high"}
            for issue in active_snapshot.issues
        )
        if has_memory_pressure and current_batch_size > 1:
            return max(1, current_batch_size // 2)
        return current_batch_size

    def get_health_report(self) -> dict[str, Any]:
        """Return a tool-friendly health report with threshold metadata."""
        snapshot = self.check_resources()
        report = snapshot.to_dict()
        report["thresholds"] = {
            "system_memory": self.config.system_memory_threshold,
            "gpu_memory": self.config.gpu_memory_threshold,
            "health_check_interval": self.config.health_check_interval,
        }
        return report

    def _get_system_memory_ratio(self, issues: list[HealthIssue]) -> float | None:
        if psutil is None:
            issues.append(
                HealthIssue(
                    severity="warning",
                    code="psutil-unavailable",
                    message="psutil is not available; system memory usage cannot be sampled.",
                )
            )
            return None

        memory = psutil.virtual_memory()
        return float(memory.percent) / 100.0

    def _get_gpu_memory_ratio(self, issues: list[HealthIssue]) -> float | None:
        if not self.config.gpu_enabled or torch is None:
            return None

        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return None

        mem_get_info = getattr(cuda, "mem_get_info", None)
        if mem_get_info is None:
            issues.append(
                HealthIssue(
                    severity="warning",
                    code="gpu-metrics-unavailable",
                    message="GPU is available, but memory metrics could not be read.",
                )
            )
            return None

        free_bytes, total_bytes = mem_get_info()
        if total_bytes <= 0:
            return None
        used_bytes = total_bytes - free_bytes
        return float(used_bytes) / float(total_bytes)

    def _directory_size(self, path: Path) -> int:
        if not path.exists():
            return 0

        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    continue
        return total

    def _derive_status(self, issues: list[HealthIssue]) -> HealthStatus:
        if any(issue.severity == "degraded" for issue in issues):
            return "degraded"
        if issues:
            return "warning"
        return "healthy"