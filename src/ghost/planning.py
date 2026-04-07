"""Planning layer for Ghost training execution.

Turns task descriptions and recommendation payloads into concrete training plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from ghost.config import GhostConfig, get_config
from ghost.context import BackendType
from ghost.ollama_client import OllamaClient
from ghost.training import TrainingConfig

_SUPPORTED_ARCHITECTURES = {"mlp", "resnet18", "resnet50", "custom"}


@dataclass
class PlanningRequest:
    """Inputs required to build a training plan."""

    task: str
    dataset: str = ""
    recommendations: dict[str, Any] | None = None


@dataclass
class TrainingPlan:
    """Structured execution plan for a training task."""

    task: str
    backend: BackendType
    architecture: str
    num_classes: int
    batch_size: int
    learning_rate: float
    epochs: int
    dataset: str = ""
    optimizer: str | None = None
    recommendation_source: str = "defaults"
    tips: list[str] = field(default_factory=list)
    raw_recommendations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "backend": self.backend.value,
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "dataset": self.dataset,
            "optimizer": self.optimizer,
            "recommendation_source": self.recommendation_source,
            "tips": self.tips,
            "raw_recommendations": self.raw_recommendations,
        }

    def to_training_config(
        self,
        model_id: str,
        *,
        checkpoint_interval: int = 5,
    ) -> TrainingConfig:
        """Bridge the planning layer into the runtime training pipeline."""
        return TrainingConfig(
            model_id=model_id,
            backend=self.backend,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            checkpoint_interval=checkpoint_interval,
        )


class TrainingPlanner:
    """Create execution plans from tasks and optional recommendations."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        ollama_client: OllamaClient | None = None,
    ):
        self.config = config or get_config()
        self.ollama_client = ollama_client or OllamaClient()

    async def create_plan(self, request: PlanningRequest) -> TrainingPlan:
        """Build a plan from the request and any available recommendations."""
        recommendations = request.recommendations
        if recommendations is None:
            recommendations = await self.ollama_client.get_recommendation(
                task=request.task,
                dataset=request.dataset,
            )

        payload = self._extract_recommendation_payload(recommendations or {})
        recommendation_source = "ollama" if payload else "defaults"
        tips = payload.get("tips", [])
        if not isinstance(tips, list):
            tips = []

        dataset = request.dataset or self._infer_dataset(request.task)

        return TrainingPlan(
            task=request.task,
            backend=self._select_backend(request.task, payload),
            architecture=self._infer_architecture(request.task, payload),
            num_classes=self._coerce_positive_int(payload.get("num_classes"), 10, maximum=10_000),
            batch_size=self._coerce_positive_int(
                payload.get("batch_size"),
                self.config.default_batch_size,
            ),
            learning_rate=self._coerce_positive_float(
                payload.get("learning_rate"),
                self.config.default_learning_rate,
            ),
            epochs=self._coerce_positive_int(
                payload.get("epochs"),
                self.config.default_epochs,
                maximum=self.config.max_iterations,
            ),
            dataset=dataset,
            optimizer=str(payload.get("optimizer")) if payload.get("optimizer") else None,
            recommendation_source=recommendation_source,
            tips=[str(item) for item in tips],
            raw_recommendations=payload,
        )

    def _extract_recommendation_payload(
        self,
        recommendations: dict[str, Any],
    ) -> dict[str, Any]:
        payload = recommendations.get("recommendations")
        return payload if isinstance(payload, dict) else {}

    def _infer_dataset(self, task_text: str) -> str:
        match = re.search(r"\bon\s+([^,]+)", task_text, flags=re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).strip()

    def _coerce_positive_int(
        self,
        value: Any,
        default: int,
        *,
        minimum: int = 1,
        maximum: int = 4096,
    ) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default

        if parsed < minimum:
            return default
        return min(parsed, maximum)

    def _coerce_positive_float(
        self,
        value: Any,
        default: float,
        *,
        minimum: float = 1.0e-6,
        maximum: float = 10.0,
    ) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default

        if parsed < minimum:
            return default
        return min(parsed, maximum)

    def _select_backend(self, task_text: str, recommendations: dict[str, Any]) -> BackendType:
        task_lower = task_text.lower()
        if "tensorflow" in task_lower or "keras" in task_lower:
            return BackendType.TENSORFLOW

        recommended_backend = str(recommendations.get("backend", "")).lower()
        if "tensorflow" in recommended_backend or "keras" in recommended_backend:
            return BackendType.TENSORFLOW
        if "pytorch" in recommended_backend or "torch" in recommended_backend:
            return BackendType.PYTORCH

        configured_backend = self.config.get_backend()
        return (
            BackendType.PYTORCH
            if configured_backend == "pytorch"
            else BackendType.TENSORFLOW
        )

    def _infer_architecture(
        self,
        task_text: str,
        recommendations: dict[str, Any],
    ) -> str:
        recommended = str(recommendations.get("architecture", "")).lower().strip()
        if recommended in _SUPPORTED_ARCHITECTURES:
            return recommended

        task_lower = task_text.lower()
        keyword_map = {
            "resnet50": "resnet50",
            "resnet18": "resnet18",
            "mlp": "mlp",
            "bert": "custom",
            "transformer": "custom",
            "lstm": "custom",
        }
        for keyword, architecture in keyword_map.items():
            if keyword in task_lower:
                return architecture

        return "mlp"