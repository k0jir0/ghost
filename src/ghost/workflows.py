"""Workflow coordination for drift-triggered retraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ghost.config import GhostConfig, get_config
from ghost.metadata_store import MetadataStore
from ghost.retraining import RetrainingManager


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WorkflowRecord:
    workflow_id: str
    workflow_type: str
    registry_id: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorkflowEngine:
    """Create persisted workflow records from operational triggers."""

    def __init__(
        self,
        retraining_manager: RetrainingManager,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.retraining_manager = retraining_manager
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )

    def trigger_drift_retraining(self, registry_id: str, *, reason: str) -> WorkflowRecord:
        request = self.retraining_manager.queue_retraining(registry_id, reason=reason)
        workflow = WorkflowRecord(
            workflow_id=f"workflow__{registry_id}__drift-retraining",
            workflow_type="drift-retraining",
            registry_id=registry_id,
            status="queued",
            metadata={"retraining_request_id": request.request_id, "reason": reason},
        )
        self.metadata_store.save_record("workflows", workflow.workflow_id, workflow.to_dict())
        return workflow