"""Simple policy scheduler for retraining workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ghost.config import GhostConfig, get_config
from ghost.drift import DriftDetector
from ghost.metadata_store import MetadataStore
from ghost.workflows import WorkflowEngine, WorkflowRecord


@dataclass
class RetrainingPolicy:
    policy_id: str
    registry_id: str
    mean_shift_threshold: float = 0.5
    min_samples: int = 5
    enabled: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class WorkflowScheduler:
    """Evaluate simple drift policies and queue retraining workflows."""

    def __init__(
        self,
        workflow_engine: WorkflowEngine,
        drift_detector: DriftDetector,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.workflow_engine = workflow_engine
        self.drift_detector = drift_detector
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )

    def upsert_policy(self, policy: RetrainingPolicy) -> None:
        self.metadata_store.save_record(
            "retraining-policies", policy.policy_id, policy.to_dict()
        )

    def evaluate_policies(self) -> list[WorkflowRecord]:
        created: list[WorkflowRecord] = []
        for payload in self.metadata_store.list_records("retraining-policies"):
            policy = RetrainingPolicy(**payload)
            if not policy.enabled:
                continue
            report = self.drift_detector.get_report(
                policy.registry_id,
                mean_shift_threshold=policy.mean_shift_threshold,
            )
            if report.sample_count < policy.min_samples:
                continue
            if report.status != "warning":
                continue
            created.append(
                self.workflow_engine.trigger_drift_retraining(
                    policy.registry_id,
                    reason="Drift policy triggered retraining",
                )
            )
        return created
