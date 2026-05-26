"""Persistent experiment-run and artifact storage for Ghost."""

from __future__ import annotations

from math import inf
from typing import Any

from ghost.config import GhostConfig, get_config
from ghost.metadata_store import MetadataStore
from ghost.schemas import ArtifactRecord, ExperimentRunRecord


class RunStore:
    """Persist searchable experiment runs and their artifacts."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )

    def upsert_run(self, record: ExperimentRunRecord) -> None:
        self.metadata_store.save_record(
            "experiment-runs", record.run_id, record.to_dict()
        )

    def get_run(self, run_id: str) -> ExperimentRunRecord | None:
        payload = self.metadata_store.load_record("experiment-runs", run_id)
        if not isinstance(payload, dict):
            return None
        return ExperimentRunRecord.from_dict(payload)

    def list_runs(
        self,
        *,
        experiment_id: str | None = None,
        status: str | None = None,
        dataset_id: str | None = None,
    ) -> list[ExperimentRunRecord]:
        runs: list[ExperimentRunRecord] = []
        for payload in self.metadata_store.list_records("experiment-runs"):
            try:
                record = ExperimentRunRecord.from_dict(payload)
            except Exception:
                continue
            if experiment_id is not None and record.experiment_id != experiment_id:
                continue
            if status is not None and record.status != status:
                continue
            if dataset_id is not None and record.dataset_id != dataset_id:
                continue
            runs.append(record)
        return sorted(runs, key=lambda record: record.created_at)

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        runs = [
            record for run_id in run_ids if (record := self.get_run(run_id)) is not None
        ]
        if not runs:
            return {
                "runs": [],
                "count": 0,
                "summary": {},
                "deltas": [],
            }

        baseline = runs[0]
        best_accuracy_run = self._best_run(
            runs, "final_accuracy", higher_is_better=True
        )
        lowest_loss_run = self._best_run(runs, "final_loss", higher_is_better=False)

        deltas = []
        for run in runs[1:]:
            deltas.append(
                {
                    "run_id": run.run_id,
                    "baseline_run_id": baseline.run_id,
                    "final_accuracy_delta": self._metric_delta(
                        run.metrics,
                        baseline.metrics,
                        "final_accuracy",
                    ),
                    "final_loss_delta": self._metric_delta(
                        run.metrics,
                        baseline.metrics,
                        "final_loss",
                    ),
                }
            )

        return {
            "runs": [run.to_dict() for run in runs],
            "count": len(runs),
            "summary": {
                "baseline_run_id": baseline.run_id,
                "best_accuracy_run_id": (
                    best_accuracy_run.run_id if best_accuracy_run is not None else None
                ),
                "lowest_loss_run_id": (
                    lowest_loss_run.run_id if lowest_loss_run is not None else None
                ),
            },
            "deltas": deltas,
        }

    def upsert_artifact(self, artifact: ArtifactRecord) -> None:
        self.metadata_store.save_record(
            "artifacts", artifact.artifact_id, artifact.to_dict()
        )

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        payload = self.metadata_store.load_record("artifacts", artifact_id)
        if not isinstance(payload, dict):
            return None
        return ArtifactRecord.from_dict(payload)

    def list_artifacts(
        self,
        *,
        run_id: str | None = None,
        model_id: str | None = None,
        artifact_type: str | None = None,
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        for payload in self.metadata_store.list_records("artifacts"):
            try:
                artifact = ArtifactRecord.from_dict(payload)
            except Exception:
                continue
            if run_id is not None and artifact.run_id != run_id:
                continue
            if model_id is not None and artifact.model_id != model_id:
                continue
            if artifact_type is not None and artifact.artifact_type != artifact_type:
                continue
            artifacts.append(artifact)
        return sorted(artifacts, key=lambda artifact: artifact.created_at)

    def get_checkpoint_artifact_for_run(self, run_id: str) -> ArtifactRecord | None:
        artifacts = self.list_artifacts(run_id=run_id, artifact_type="checkpoint")
        return artifacts[-1] if artifacts else None

    def _best_run(
        self,
        runs: list[ExperimentRunRecord],
        metric_name: str,
        *,
        higher_is_better: bool,
    ) -> ExperimentRunRecord | None:
        best_record: ExperimentRunRecord | None = None
        best_value = -inf if higher_is_better else inf
        for run in runs:
            value = self._metric_value(run.metrics, metric_name)
            if value is None:
                continue
            if higher_is_better and value > best_value:
                best_value = value
                best_record = run
            if not higher_is_better and value < best_value:
                best_value = value
                best_record = run
        return best_record

    def _metric_delta(
        self,
        current: dict[str, Any],
        baseline: dict[str, Any],
        metric_name: str,
    ) -> float | None:
        current_value = self._metric_value(current, metric_name)
        baseline_value = self._metric_value(baseline, metric_name)
        if current_value is None or baseline_value is None:
            return None
        return current_value - baseline_value

    def _metric_value(
        self,
        metrics: dict[str, Any],
        metric_name: str,
    ) -> float | None:
        value = metrics.get(metric_name)
        if value is None:
            return None
        return float(value)
