"""Versioned model registry and promotion controls for Ghost."""

from __future__ import annotations

from typing import Literal

from ghost.audit import AuditLogger
from ghost.config import GhostConfig, get_config
from ghost.evaluation import EvaluationPolicy, ModelEvaluator
from ghost.metadata_store import MetadataStore
from ghost.run_store import RunStore
from ghost.schemas import ModelRegistryRecord

RegistryStage = Literal["draft", "staging", "production", "archived", "rejected"]


class ModelRegistry:
    """Persist model versions, approval status, and promotion aliases."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
        run_store: RunStore | None = None,
        evaluator: ModelEvaluator | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )
        self.run_store = run_store or RunStore(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.evaluator = evaluator or ModelEvaluator(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.audit_logger = audit_logger or AuditLogger(
            config=self.config,
            metadata_store=self.metadata_store,
        )

    def register_model(
        self,
        run_id: str,
        *,
        actor: str = "system",
        policy: EvaluationPolicy | None = None,
        baseline_registry_id: str | None = None,
    ) -> ModelRegistryRecord:
        candidate = self.run_store.get_run(run_id)
        if candidate is None:
            raise KeyError(f"Unknown run id: {run_id}")

        artifact = self.run_store.get_checkpoint_artifact_for_run(run_id)
        if artifact is None:
            raise ValueError("Run does not have a checkpoint artifact to register")

        baseline_run = None
        if baseline_registry_id is not None:
            baseline_record = self.get_model(baseline_registry_id)
            if baseline_record is None:
                raise KeyError(f"Unknown baseline registry id: {baseline_registry_id}")
            baseline_run = self.run_store.get_run(baseline_record.run_id)
        else:
            current_production = self.current_production(candidate.model_id)
            if current_production is not None:
                baseline_run = self.run_store.get_run(current_production.run_id)

        evaluation = self.evaluator.evaluate_candidate(
            candidate,
            baseline=baseline_run,
            policy=policy,
        )

        record = ModelRegistryRecord(
            registry_id=self._next_registry_id(candidate.model_id),
            model_id=candidate.model_id,
            run_id=run_id,
            artifact_id=artifact.artifact_id,
            stage="draft",
            backend=candidate.backend,
            architecture=candidate.architecture,
            dataset_id=candidate.dataset_id,
            dataset_version=candidate.dataset_version,
            evaluation_id=evaluation.evaluation_id,
            evaluation_status=evaluation.status,
            metrics=dict(candidate.metrics),
            metadata={
                "artifact_uri": artifact.uri,
                "artifact_checksum": artifact.checksum,
                "code_version": candidate.code_version,
                "experiment_id": candidate.experiment_id,
                "input_shape": candidate.input_shape,
                "num_classes": candidate.num_classes,
                "eligible_for_promotion": evaluation.passed,
            },
        )
        self._save_record(record)
        self.audit_logger.record(
            "register_model",
            subject_type="model_registry",
            subject_id=record.registry_id,
            actor=actor,
            details={"run_id": run_id, "evaluation_status": evaluation.status},
        )
        return record

    def list_models(
        self,
        *,
        stage: str | None = None,
        model_id: str | None = None,
    ) -> list[ModelRegistryRecord]:
        records: list[ModelRegistryRecord] = []
        for payload in self.metadata_store.list_records("model-registry"):
            try:
                record = ModelRegistryRecord.from_dict(payload)
            except Exception:
                continue
            if stage is not None and record.stage != stage:
                continue
            if model_id is not None and record.model_id != model_id:
                continue
            records.append(record)
        return sorted(records, key=lambda record: record.created_at)

    def get_model(self, registry_id: str) -> ModelRegistryRecord | None:
        payload = self.metadata_store.load_record("model-registry", registry_id)
        if not isinstance(payload, dict):
            return None
        return ModelRegistryRecord.from_dict(payload)

    def promote_model(
        self,
        registry_id: str,
        *,
        stage: RegistryStage,
        actor: str = "system",
        alias: str | None = None,
    ) -> ModelRegistryRecord:
        record = self.get_model(registry_id)
        if record is None:
            raise KeyError(f"Unknown registry id: {registry_id}")
        if record.evaluation_status != "passed":
            raise ValueError("Model has not passed evaluation gates")

        if stage == "production":
            for current in self.list_models(
                stage="production", model_id=record.model_id
            ):
                if current.registry_id == record.registry_id:
                    continue
                current.stage = "archived"
                current.aliases = [
                    item for item in current.aliases if item != "current-production"
                ]
                self._save_record(current)

        record.stage = stage
        aliases = list(record.aliases)
        if alias and alias not in aliases:
            aliases.append(alias)
        if stage == "production" and "current-production" not in aliases:
            aliases.append("current-production")
        record.aliases = aliases
        record.approval = {
            "approved_by": actor,
            "approved_at": record.updated_at,
        }
        self._save_record(record)
        self.audit_logger.record(
            "promote_model",
            subject_type="model_registry",
            subject_id=registry_id,
            actor=actor,
            details={"stage": stage, "aliases": aliases},
        )
        return record

    def reject_model(
        self,
        registry_id: str,
        *,
        reason: str,
        actor: str = "system",
    ) -> ModelRegistryRecord:
        record = self.get_model(registry_id)
        if record is None:
            raise KeyError(f"Unknown registry id: {registry_id}")

        record.stage = "rejected"
        record.metadata["rejection_reason"] = reason
        self._save_record(record)
        self.audit_logger.record(
            "reject_model",
            subject_type="model_registry",
            subject_id=registry_id,
            actor=actor,
            details={"reason": reason},
        )
        return record

    def current_production(self, model_id: str) -> ModelRegistryRecord | None:
        production_records = self.list_models(stage="production", model_id=model_id)
        return production_records[-1] if production_records else None

    def _next_registry_id(self, model_id: str) -> str:
        safe_model_id = model_id.replace("/", "-")
        version = len(self.list_models(model_id=model_id)) + 1
        return f"{safe_model_id}__v{version}"

    def _save_record(self, record: ModelRegistryRecord) -> None:
        self.metadata_store.save_record(
            "model-registry",
            record.registry_id,
            record.to_dict(),
        )
