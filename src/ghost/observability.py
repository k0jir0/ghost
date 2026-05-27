"""Prediction observability and aggregate inference metrics for Ghost."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Any
from uuid import uuid4

import numpy as np

from ghost.config import GhostConfig, get_config
from ghost.metadata_store import MetadataStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PredictionEvent:
    """Persisted observability event for a prediction call."""

    event_id: str
    registry_id: str
    model_id: str
    latency_ms: float
    batch_size: int
    success: bool
    predicted_classes: list[int] = field(default_factory=list)
    input_mean: float = 0.0
    input_std: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PredictionEvent:
        return cls(**payload)


class ModelObservability:
    """Record prediction events and derive model-level observability summaries."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )

    def record_prediction(
        self,
        registry_id: str,
        model_id: str,
        *,
        latency_ms: float,
        batch_size: int,
        success: bool,
        inputs: list[Any],
        predictions: list[dict[str, Any]],
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> PredictionEvent:
        input_mean, input_std = self._input_summary(inputs)
        event = PredictionEvent(
            event_id=uuid4().hex,
            registry_id=registry_id,
            model_id=model_id,
            latency_ms=float(latency_ms),
            batch_size=int(batch_size),
            success=success,
            predicted_classes=[
                int(payload["predicted_class"])
                for payload in predictions
                if isinstance(payload, dict)
                and payload.get("predicted_class") is not None
            ],
            input_mean=input_mean,
            input_std=input_std,
            error_type=error_type,
            error_message=error_message,
        )
        self.metadata_store.save_record(
            "prediction-events",
            event.event_id,
            event.to_dict(),
        )
        return event

    def list_events(self, registry_id: str) -> list[PredictionEvent]:
        events: list[PredictionEvent] = []
        for payload in self.metadata_store.list_records("prediction-events"):
            try:
                event = PredictionEvent.from_dict(payload)
            except Exception:
                continue
            if event.registry_id != registry_id:
                continue
            events.append(event)
        return sorted(events, key=lambda event: event.created_at)

    def get_summary(self, registry_id: str) -> dict[str, Any]:
        events = self.list_events(registry_id)
        if not events:
            return {
                "registry_id": registry_id,
                "request_count": 0,
                "error_count": 0,
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "last_event_at": None,
            }

        latencies = [event.latency_ms for event in events]
        error_count = sum(1 for event in events if not event.success)
        return {
            "registry_id": registry_id,
            "request_count": len(events),
            "error_count": error_count,
            "error_rate": error_count / len(events),
            "avg_latency_ms": mean(latencies),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "last_event_at": events[-1].created_at,
            "predicted_class_counts": self._predicted_class_counts(events),
        }

    def _predicted_class_counts(self, events: list[PredictionEvent]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in events:
            for predicted_class in event.predicted_classes:
                key = str(predicted_class)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def _input_summary(self, inputs: list[Any]) -> tuple[float, float]:
        try:
            array = np.asarray(inputs, dtype=np.float32)
        except Exception:
            return 0.0, 0.0

        if array.size == 0:
            return 0.0, 0.0

        return float(array.mean()), float(array.std())
