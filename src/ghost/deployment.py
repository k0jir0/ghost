"""Environment-aware model deployment records and rollback controls."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from ghost.audit import AuditLogger
from ghost.config import GhostConfig, get_config
from ghost.environment import EnvironmentManager
from ghost.metadata_store import MetadataStore
from ghost.model_registry import ModelRegistry

DeploymentStatus = Literal["pending", "active", "deactivated", "rolled_back"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DeploymentRecord:
    """Persisted environment deployment state for a registry model."""

    deployment_id: str
    environment: str
    registry_id: str
    model_id: str
    status: DeploymentStatus = "pending"
    traffic_percent: int = 100
    health_status: str = "unknown"
    health_check_url: str = ""
    previous_registry_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DeploymentRecord:
        return cls(**payload)


class DeploymentManager:
    """Create, activate, and roll back environment deployment records."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
        model_registry: ModelRegistry | None = None,
        environment_manager: EnvironmentManager | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )
        self.model_registry = model_registry or ModelRegistry(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.environment_manager = environment_manager or EnvironmentManager(
            config=self.config
        )
        self.audit_logger = audit_logger or AuditLogger(
            config=self.config,
            metadata_store=self.metadata_store,
        )

    def create_deployment(
        self,
        registry_id: str,
        *,
        environment: str,
        actor: str = "system",
        traffic_percent: int = 100,
        health_check_url: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DeploymentRecord:
        if traffic_percent < 1 or traffic_percent > 100:
            raise ValueError("traffic_percent must be between 1 and 100")
        model = self.model_registry.get_model(registry_id)
        if model is None:
            raise KeyError(f"Unknown registry id: {registry_id}")
        if environment != "dev" and model.stage not in {
            "staging",
            "production",
            "archived",
        }:
            raise ValueError(
                "Only staging, production, or archived registry records can deploy outside dev"
            )

        profile = self.environment_manager.get_profile(environment)
        previous = self.current_active(environment, model.model_id)
        record = DeploymentRecord(
            deployment_id=self._deployment_id(environment, registry_id),
            environment=profile.name,
            registry_id=registry_id,
            model_id=model.model_id,
            traffic_percent=traffic_percent,
            health_check_url=health_check_url,
            previous_registry_id=previous.registry_id if previous else "",
            metadata=metadata or {},
        )
        self._save_record(record)
        self.audit_logger.record(
            "create_deployment",
            subject_type="deployment",
            subject_id=record.deployment_id,
            actor=actor,
            details={
                "environment": environment,
                "registry_id": registry_id,
                "traffic_percent": traffic_percent,
            },
        )
        return record

    def activate_deployment(
        self,
        deployment_id: str,
        *,
        actor: str = "system",
        health_status: str = "healthy",
    ) -> DeploymentRecord:
        record = self.get_deployment(deployment_id)
        if record is None:
            raise KeyError(f"Unknown deployment id: {deployment_id}")
        if health_status != "healthy":
            raise ValueError("Deployment health check must be healthy before activation")

        current = self.current_active(record.environment, record.model_id)
        if current is not None and current.deployment_id != record.deployment_id:
            current.status = "deactivated"
            current.updated_at = _utc_now_iso()
            self._save_record(current)

        record.status = "active"
        record.health_status = health_status
        record.updated_at = _utc_now_iso()
        self._save_record(record)
        self.audit_logger.record(
            "activate_deployment",
            subject_type="deployment",
            subject_id=record.deployment_id,
            actor=actor,
            details={
                "environment": record.environment,
                "registry_id": record.registry_id,
            },
        )
        return record

    def rollback(
        self,
        *,
        environment: str,
        model_id: str,
        actor: str = "system",
    ) -> DeploymentRecord:
        current = self.current_active(environment, model_id)
        if current is None:
            raise ValueError("No active deployment is available to roll back")

        target_registry_id = current.previous_registry_id or self._latest_prior_registry(
            environment,
            model_id,
            exclude_deployment_id=current.deployment_id,
        )
        if not target_registry_id:
            raise ValueError("No previous deployment is available for rollback")

        current.status = "rolled_back"
        current.updated_at = _utc_now_iso()
        self._save_record(current)

        rollback_record = self.create_deployment(
            target_registry_id,
            environment=environment,
            actor=actor,
            metadata={"rollback_of": current.deployment_id},
        )
        activated = self.activate_deployment(
            rollback_record.deployment_id,
            actor=actor,
            health_status="healthy",
        )
        self.audit_logger.record(
            "rollback_deployment",
            subject_type="deployment",
            subject_id=current.deployment_id,
            actor=actor,
            details={"rollback_deployment_id": activated.deployment_id},
        )
        return activated

    def get_deployment(self, deployment_id: str) -> DeploymentRecord | None:
        payload = self.metadata_store.load_record("deployments", deployment_id)
        if not isinstance(payload, dict):
            return None
        return DeploymentRecord.from_dict(payload)

    def list_deployments(
        self,
        *,
        environment: str | None = None,
        model_id: str | None = None,
    ) -> list[DeploymentRecord]:
        deployments: list[DeploymentRecord] = []
        for payload in self.metadata_store.list_records("deployments"):
            try:
                record = DeploymentRecord.from_dict(payload)
            except TypeError:
                continue
            if environment is not None and record.environment != environment:
                continue
            if model_id is not None and record.model_id != model_id:
                continue
            deployments.append(record)
        return sorted(deployments, key=lambda record: record.created_at)

    def current_active(
        self,
        environment: str,
        model_id: str,
    ) -> DeploymentRecord | None:
        active = [
            record
            for record in self.list_deployments(
                environment=environment,
                model_id=model_id,
            )
            if record.status == "active"
        ]
        return active[-1] if active else None

    def _latest_prior_registry(
        self,
        environment: str,
        model_id: str,
        *,
        exclude_deployment_id: str,
    ) -> str:
        for record in reversed(
            self.list_deployments(environment=environment, model_id=model_id)
        ):
            if record.deployment_id == exclude_deployment_id:
                continue
            if record.registry_id:
                return record.registry_id
        return ""

    def _deployment_id(self, environment: str, registry_id: str) -> str:
        seed = f"{environment}|{registry_id}|{_utc_now_iso()}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
        return f"{environment}__{registry_id}__{digest}"

    def _save_record(self, record: DeploymentRecord) -> None:
        self.metadata_store.save_record(
            "deployments",
            record.deployment_id,
            record.to_dict(),
        )
