"""Evaluation gates for model registry decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from ghost.config import GhostConfig, get_config
from ghost.metadata_store import MetadataStore
from ghost.schemas import EvaluationRecord, ExperimentRunRecord


@dataclass
class EvaluationPolicy:
    """Thresholds used to decide whether a run is promotion-eligible."""

    min_accuracy: float | None = None
    max_loss: float | None = None
    max_accuracy_drop: float = 0.05
    max_loss_increase: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelEvaluator:
    """Evaluate candidate experiment runs against explicit thresholds."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )

    def evaluate_candidate(
        self,
        candidate: ExperimentRunRecord,
        *,
        baseline: ExperimentRunRecord | None = None,
        policy: EvaluationPolicy | None = None,
    ) -> EvaluationRecord:
        policy = policy or EvaluationPolicy()
        issues: list[str] = []
        metrics = dict(candidate.metrics)
        final_accuracy = self._metric(metrics, "final_accuracy")
        final_loss = self._metric(metrics, "final_loss")

        if policy.min_accuracy is not None:
            if final_accuracy is None or final_accuracy < policy.min_accuracy:
                issues.append("Candidate accuracy is below the minimum threshold")

        if policy.max_loss is not None:
            if final_loss is None or final_loss > policy.max_loss:
                issues.append("Candidate loss exceeds the maximum threshold")

        baseline_metrics = dict(baseline.metrics) if baseline is not None else {}
        baseline_accuracy = self._metric(baseline_metrics, "final_accuracy")
        baseline_loss = self._metric(baseline_metrics, "final_loss")

        if baseline_accuracy is not None and final_accuracy is not None:
            if final_accuracy < baseline_accuracy - policy.max_accuracy_drop:
                issues.append("Candidate accuracy regressed beyond the allowed drop")

        if baseline_loss is not None and final_loss is not None:
            if final_loss > baseline_loss + policy.max_loss_increase:
                issues.append("Candidate loss regressed beyond the allowed increase")

        evaluation = EvaluationRecord(
            evaluation_id=uuid4().hex,
            run_id=candidate.run_id,
            model_id=candidate.model_id,
            status="passed" if not issues else "failed",
            passed=not issues,
            thresholds=policy.to_dict(),
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            issues=issues,
        )
        self.metadata_store.save_record(
            "evaluations",
            evaluation.evaluation_id,
            evaluation.to_dict(),
        )
        return evaluation

    def get_evaluation(self, evaluation_id: str) -> EvaluationRecord | None:
        payload = self.metadata_store.load_record("evaluations", evaluation_id)
        if not isinstance(payload, dict):
            return None
        return EvaluationRecord.from_dict(payload)

    def _metric(self, metrics: dict[str, Any], name: str) -> float | None:
        value = metrics.get(name)
        if value is None:
            return None
        return float(value)