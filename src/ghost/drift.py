"""Simple production drift detection from recorded prediction events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ghost.config import GhostConfig, get_config
from ghost.metadata_store import MetadataStore
from ghost.observability import ModelObservability


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DriftReport:
    """Persisted drift summary derived from observability events."""

    report_id: str
    registry_id: str
    status: str
    sample_count: int
    baseline_input_mean: float = 0.0
    current_input_mean: float = 0.0
    mean_shift: float = 0.0
    issues: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DriftReport:
        return cls(**payload)


class DriftDetector:
    """Generate drift reports from prediction observability history."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
        observability: ModelObservability | None = None,
    ):
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )
        self.observability = observability or ModelObservability(
            config=self.config,
            metadata_store=self.metadata_store,
        )

    def get_report(
        self,
        registry_id: str,
        *,
        mean_shift_threshold: float = 0.5,
    ) -> DriftReport:
        events = [
            event
            for event in self.observability.list_events(registry_id)
            if event.success
        ]
        if not events:
            report = DriftReport(
                report_id=f"{registry_id}__drift",
                registry_id=registry_id,
                status="unknown",
                sample_count=0,
                issues=["No successful prediction events recorded yet"],
            )
            self._persist(report)
            return report

        baseline = events[0]
        current_events = events[-min(len(events), 10) :]
        current_mean = sum(event.input_mean for event in current_events) / len(
            current_events
        )
        mean_shift = abs(current_mean - baseline.input_mean)
        issues: list[str] = []
        status = "stable"
        if mean_shift > mean_shift_threshold:
            status = "warning"
            issues.append("Observed input mean shifted beyond the configured threshold")

        report = DriftReport(
            report_id=f"{registry_id}__drift",
            registry_id=registry_id,
            status=status,
            sample_count=len(events),
            baseline_input_mean=baseline.input_mean,
            current_input_mean=current_mean,
            mean_shift=mean_shift,
            issues=issues,
        )
        self._persist(report)
        return report

    def _persist(self, report: DriftReport) -> None:
        self.metadata_store.save_record(
            "drift-reports", report.report_id, report.to_dict()
        )
