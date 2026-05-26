"""Experiment tracking and artifact lineage for Ghost training runs."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ghost.context import ModelContext
from ghost.run_store import RunStore
from ghost.schemas import ArtifactRecord, ExperimentRunRecord

if TYPE_CHECKING:
    from ghost.orchestration import TrainingRunRecord


class ExperimentTracker:
    """Materialize searchable experiment metadata from orchestration runs."""

    def __init__(self, run_store: RunStore | None = None):
        self.run_store = run_store or RunStore()

    def record_training_run(
        self,
        record: TrainingRunRecord,
        *,
        context: ModelContext | None = None,
    ) -> ExperimentRunRecord:
        experiment_record = self._build_run_record(record, context=context)
        artifacts = self._build_artifacts(
            record,
            experiment_record=experiment_record,
            context=context,
        )
        experiment_record.artifact_ids = [
            artifact.artifact_id for artifact in artifacts
        ]
        self.run_store.upsert_run(experiment_record)
        for artifact in artifacts:
            self.run_store.upsert_artifact(artifact)
        return experiment_record

    def _build_run_record(
        self,
        record: TrainingRunRecord,
        *,
        context: ModelContext | None = None,
    ) -> ExperimentRunRecord:
        plan = record.plan.to_dict() if record.plan is not None else {}
        request = record.request.to_dict() if record.request is not None else {}
        dataset_id = record.dataset.dataset_id if record.dataset is not None else ""
        dataset_version = self._dataset_version(record, context=context)
        backend = record.plan.backend.value if record.plan is not None else ""
        architecture = record.plan.architecture if record.plan is not None else ""
        num_classes = 0
        if record.plan is not None:
            num_classes = record.plan.num_classes
        elif record.dataset is not None:
            num_classes = record.dataset.num_classes

        return ExperimentRunRecord(
            run_id=record.run_id,
            experiment_id=self._experiment_id(record),
            model_id=record.model_id,
            status=record.status,
            backend=backend,
            architecture=architecture,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            input_shape=self._input_shape(record, context=context),
            num_classes=num_classes,
            metrics=self._metrics(record),
            plan=plan,
            request=request,
            analysis=record.analysis if isinstance(record.analysis, dict) else {},
            code_version=self._resolve_code_version(),
            metadata={
                "error": record.error,
                "event_count": len(record.events),
                "health_status": (
                    record.result.health_status
                    if record.result is not None
                    else "unknown"
                ),
                "used_synthetic_data": (
                    record.result.used_synthetic_data
                    if record.result is not None
                    else False
                ),
            },
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def _build_artifacts(
        self,
        record: TrainingRunRecord,
        *,
        experiment_record: ExperimentRunRecord,
        context: ModelContext | None = None,
    ) -> list[ArtifactRecord]:
        checkpoint_path = None
        if record.result is not None and record.result.checkpoint_path is not None:
            checkpoint_path = record.result.checkpoint_path
        elif context is not None:
            checkpoint_path = context.checkpoint_path

        if checkpoint_path is None:
            return []

        artifact = ArtifactRecord(
            artifact_id=f"{record.run_id}__checkpoint",
            artifact_type="checkpoint",
            uri=str(checkpoint_path),
            run_id=record.run_id,
            model_id=record.model_id,
            checksum=self._checksum(checkpoint_path),
            metadata={
                "backend": experiment_record.backend,
                "architecture": experiment_record.architecture,
                "dataset_id": experiment_record.dataset_id,
                "dataset_version": experiment_record.dataset_version,
                "code_version": experiment_record.code_version,
                "input_shape": experiment_record.input_shape,
                "num_classes": experiment_record.num_classes,
                "metrics": experiment_record.metrics,
            },
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        return [artifact]

    def _experiment_id(
        self,
        record: TrainingRunRecord,
    ) -> str:
        task = record.request.task if record.request is not None else record.model_id
        dataset = (
            record.dataset.dataset_id if record.dataset is not None else "no-dataset"
        )
        backend = record.plan.backend.value if record.plan is not None else "unknown"
        architecture = (
            record.plan.architecture if record.plan is not None else "unknown"
        )
        raw = f"{task}-{dataset}-{backend}-{architecture}".lower()
        normalized = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
        return normalized or record.model_id

    def _dataset_version(
        self,
        record: TrainingRunRecord,
        *,
        context: ModelContext | None = None,
    ) -> str:
        if record.dataset is not None and isinstance(record.dataset.metadata, dict):
            dataset_version = record.dataset.metadata.get("dataset_version")
            if dataset_version:
                return str(dataset_version)
        if context is not None and isinstance(
            context.metadata.get("dataset_spec"), dict
        ):
            dataset_version = (
                context.metadata["dataset_spec"]
                .get("metadata", {})
                .get("dataset_version")
            )
            if dataset_version:
                return str(dataset_version)
        return "builtin-v1"

    def _input_shape(
        self,
        record: TrainingRunRecord,
        *,
        context: ModelContext | None = None,
    ) -> list[int]:
        if record.dataset is not None:
            return list(record.dataset.input_shape)
        if context is not None:
            input_shape = context.config.get("input_shape")
            if isinstance(input_shape, list):
                return [int(value) for value in input_shape]
        return []

    def _metrics(self, record: TrainingRunRecord) -> dict[str, Any]:
        if record.result is None:
            return {}

        return {
            "final_loss": record.result.final_loss,
            "final_accuracy": record.result.final_accuracy,
            "epochs_completed": record.result.epochs_completed,
            "duration_seconds": record.result.duration_seconds,
            "effective_batch_size": record.result.effective_batch_size,
            "health_status": record.result.health_status,
            "data_mode": record.result.data_mode,
        }

    def _resolve_code_version(self) -> str:
        repo_root = Path(__file__).resolve().parents[2]
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
        except Exception:
            return "unknown"

        revision = result.stdout.strip()
        return revision or "unknown"

    def _checksum(self, path: Path) -> str:
        if not path.exists() or not path.is_file():
            return ""

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()
