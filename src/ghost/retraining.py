"""Retraining request management for Ghost."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ghost.config import GhostConfig, get_config
from ghost.metadata_store import MetadataStore
from ghost.model_registry import ModelRegistry
from ghost.task_queue import TaskQueueStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RetrainingRequest:
    request_id: str
    registry_id: str
    model_id: str
    reason: str
    task_id: str
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetrainingManager:
    """Queue retraining work from operational signals."""

    def __init__(
        self,
        task_queue: TaskQueueStore,
        model_registry: ModelRegistry,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.task_queue = task_queue
        self.model_registry = model_registry
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )

    def queue_retraining(self, registry_id: str, *, reason: str) -> RetrainingRequest:
        record = self.model_registry.get_model(registry_id)
        if record is None:
            raise KeyError(f"Unknown registry id: {registry_id}")

        task = self.task_queue.add_task(
            f"Retrain {record.model_id} on {record.dataset_id or 'its dataset'}",
            task_id=f"retrain-{registry_id}",
        )
        request = RetrainingRequest(
            request_id=f"{registry_id}__retrain",
            registry_id=registry_id,
            model_id=record.model_id,
            reason=reason,
            task_id=task.task_id,
        )
        self.metadata_store.save_record(
            "retraining-requests",
            request.request_id,
            request.to_dict(),
        )
        return request