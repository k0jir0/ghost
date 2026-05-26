"""Alert evaluation for Ghost observability and drift signals."""

from __future__ import annotations

from typing import Any

from ghost.config import GhostConfig, get_config
from ghost.metadata_store import MetadataStore


class AlertManager:
    """Derive alert payloads from observability and drift summaries."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )

    def evaluate(
        self,
        registry_id: str,
        *,
        observability: dict[str, Any],
        drift_report: dict[str, Any],
        latency_threshold_ms: float = 1_000.0,
        error_rate_threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        if float(observability.get("avg_latency_ms", 0.0)) > latency_threshold_ms:
            alerts.append(
                {
                    "type": "latency",
                    "message": "Average prediction latency exceeded the configured threshold",
                }
            )
        if float(observability.get("error_rate", 0.0)) > error_rate_threshold:
            alerts.append(
                {
                    "type": "errors",
                    "message": "Prediction error rate exceeded the configured threshold",
                }
            )
        if drift_report.get("status") == "warning":
            alerts.append(
                {
                    "type": "drift",
                    "message": "Prediction inputs are drifting from the baseline profile",
                }
            )

        if alerts:
            self.metadata_store.save_record(
                "alerts",
                f"{registry_id}__alerts",
                {"registry_id": registry_id, "alerts": alerts},
            )
        return alerts
