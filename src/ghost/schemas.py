"""Shared control-plane schemas for persisted Ghost metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DatasetManifest:
    """Versioned dataset descriptor for future governed ingestion flows."""

    dataset_id: str
    version: str
    source_uri: str
    schema: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetManifest:
        return cls(**payload)


@dataclass
class ArtifactRecord:
    """Artifact metadata persisted independently of backend runtime objects."""

    artifact_id: str
    artifact_type: str
    uri: str
    run_id: str = ""
    model_id: str = ""
    checksum: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactRecord:
        return cls(**payload)


@dataclass
class ExperimentRunRecord:
    """Searchable experiment-tracking record derived from a training run."""

    run_id: str
    experiment_id: str
    model_id: str
    status: str
    backend: str = ""
    architecture: str = ""
    dataset_id: str = ""
    dataset_version: str = ""
    input_shape: list[int] = field(default_factory=list)
    num_classes: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    artifact_ids: list[str] = field(default_factory=list)
    code_version: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentRunRecord:
        return cls(**payload)


@dataclass
class ModelRegistryRecord:
    """Registry-stage metadata for a versioned model artifact."""

    registry_id: str
    model_id: str
    run_id: str
    artifact_id: str
    stage: str = "draft"
    backend: str = ""
    architecture: str = ""
    dataset_id: str = ""
    dataset_version: str = ""
    evaluation_id: str = ""
    evaluation_status: str = "unknown"
    aliases: list[str] = field(default_factory=list)
    approval: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelRegistryRecord:
        return cls(**payload)


@dataclass
class EvaluationRecord:
    """Persisted model-evaluation result for registry decisions."""

    evaluation_id: str
    run_id: str
    model_id: str
    status: str
    passed: bool
    thresholds: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationRecord:
        return cls(**payload)


@dataclass
class AuditEntry:
    """Persisted audit-log entry for registry and workflow operations."""

    audit_id: str
    action: str
    subject_type: str
    subject_id: str
    actor: str
    details: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AuditEntry:
        return cls(**payload)