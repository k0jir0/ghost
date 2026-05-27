"""Transport-independent MCP tool catalog for Ghost."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

_Architecture = Literal["resnet18", "resnet50", "mlp", "custom"]


class CreatePyTorchModelArgs(BaseModel):
    model_id: str
    model_name: str
    architecture: _Architecture = "mlp"
    num_classes: int = Field(default=10, ge=1, le=10_000)
    input_shape: list[int] = Field(default=[3, 224, 224])


class CreateTensorFlowModelArgs(BaseModel):
    model_id: str
    model_name: str
    architecture: _Architecture = "mlp"
    num_classes: int = Field(default=10, ge=1, le=10_000)
    input_shape: list[int] = Field(default=[224, 224, 3])


class TrainStepArgs(BaseModel):
    model_id: str
    batch_size: int = Field(default=32, ge=1, le=4096)
    learning_rate: float = Field(default=0.001, gt=0.0, le=10.0)


class EvaluateArgs(BaseModel):
    model_id: str


class SaveCheckpointArgs(BaseModel):
    model_id: str
    path: str | None = None


class LoadCheckpointArgs(BaseModel):
    model_id: str
    path: str


class GetTrainingStatusArgs(BaseModel):
    model_id: str


class GetModelRecommendationArgs(BaseModel):
    task: str
    dataset: str = ""


class GetTrainingAnalysisArgs(BaseModel):
    model_id: str


class GetRunArgs(BaseModel):
    run_id: str


class CompareRunsArgs(BaseModel):
    run_ids: list[str] = Field(min_length=2, max_length=20)


class RegisterModelArgs(BaseModel):
    run_id: str
    baseline_registry_id: str | None = None
    min_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    max_loss: float | None = Field(default=None, ge=0.0)
    max_accuracy_drop: float = Field(default=0.05, ge=0.0, le=1.0)
    max_loss_increase: float = Field(default=0.5, ge=0.0)
    actor: str = "system"


class ListRegisteredModelsArgs(BaseModel):
    stage: str | None = None
    model_id: str | None = None


class PromoteModelArgs(BaseModel):
    registry_id: str
    stage: Literal["staging", "production", "archived"]
    approved_by: str = "system"
    alias: str | None = None


class RejectModelArgs(BaseModel):
    registry_id: str
    reason: str = Field(min_length=1)
    rejected_by: str = "system"


class PredictOnlineArgs(BaseModel):
    registry_id: str
    features: list[Any] = Field(min_length=1)


class PredictBatchArgs(BaseModel):
    registry_id: str
    inputs: list[Any] = Field(min_length=1)


class GetModelObservabilityArgs(BaseModel):
    registry_id: str


class GetDriftReportArgs(BaseModel):
    registry_id: str


class GetDatasetManifestArgs(BaseModel):
    dataset_id: str
    version: str = "builtin-v1"


class GetDatasetValidationReportArgs(BaseModel):
    dataset_id: str
    version: str = "builtin-v1"


class ListModelsArgs(BaseModel):
    pass


class ListRunsArgs(BaseModel):
    pass


class ListDatasetManifestsArgs(BaseModel):
    pass


class GetSystemHealthArgs(BaseModel):
    pass


class ListTrainingTasksArgs(BaseModel):
    include_completed: bool = False


class CreateTrainingTaskArgs(BaseModel):
    text: str = Field(min_length=1)
    task_id: str | None = None


class UpdateTrainingTaskArgs(BaseModel):
    task_id: str | None = None
    match_text: str | None = Field(default=None, min_length=1)
    text: str | None = Field(default=None, min_length=1)
    completed: bool | None = None

    @model_validator(mode="after")
    def validate_update(self) -> UpdateTrainingTaskArgs:
        if self.task_id is None and self.match_text is None:
            raise ValueError("task_id or match_text is required")
        if self.text is None and self.completed is None:
            raise ValueError("text or completed must be provided")
        return self


class DeleteTrainingTaskArgs(BaseModel):
    task_id: str | None = None
    match_text: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_delete(self) -> DeleteTrainingTaskArgs:
        if self.task_id is None and self.match_text is None:
            raise ValueError("task_id or match_text is required")
        return self


@dataclass(frozen=True)
class ToolSpec:
    """Transport-independent description of an MCP tool."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler_name: str
    tags: tuple[str, ...] = ()

    def input_schema(self) -> dict[str, Any]:
        """Return JSON schema for tool input validation and discovery."""
        return self.input_model.model_json_schema()


class ToolCatalog:
    """Registry of Ghost tool specifications."""

    def __init__(self, specs: Iterable[ToolSpec]):
        self._specs = tuple(specs)
        self._by_name = {spec.name: spec for spec in self._specs}

    @classmethod
    def default(cls) -> ToolCatalog:
        return cls(
            [
                ToolSpec(
                    name="pytorch_create_model",
                    description="Create a new PyTorch model",
                    input_model=CreatePyTorchModelArgs,
                    handler_name="_handle_pytorch_create_model",
                    tags=("pytorch", "model", "create"),
                ),
                ToolSpec(
                    name="pytorch_train_step",
                    description="Execute one training step",
                    input_model=TrainStepArgs,
                    handler_name="_handle_pytorch_train_step",
                    tags=("pytorch", "training"),
                ),
                ToolSpec(
                    name="pytorch_evaluate",
                    description="Evaluate PyTorch model on dataset",
                    input_model=EvaluateArgs,
                    handler_name="_handle_pytorch_evaluate",
                    tags=("pytorch", "evaluation"),
                ),
                ToolSpec(
                    name="pytorch_save_checkpoint",
                    description="Save PyTorch model checkpoint",
                    input_model=SaveCheckpointArgs,
                    handler_name="_handle_pytorch_save_checkpoint",
                    tags=("pytorch", "checkpoint"),
                ),
                ToolSpec(
                    name="pytorch_load_checkpoint",
                    description="Load PyTorch model checkpoint",
                    input_model=LoadCheckpointArgs,
                    handler_name="_handle_pytorch_load_checkpoint",
                    tags=("pytorch", "checkpoint"),
                ),
                ToolSpec(
                    name="tensorflow_create_model",
                    description="Create a new TensorFlow/Keras model",
                    input_model=CreateTensorFlowModelArgs,
                    handler_name="_handle_tensorflow_create_model",
                    tags=("tensorflow", "model", "create"),
                ),
                ToolSpec(
                    name="tensorflow_train_step",
                    description="Execute one training step",
                    input_model=TrainStepArgs,
                    handler_name="_handle_tensorflow_train_step",
                    tags=("tensorflow", "training"),
                ),
                ToolSpec(
                    name="tensorflow_evaluate",
                    description="Evaluate TensorFlow model on dataset",
                    input_model=EvaluateArgs,
                    handler_name="_handle_tensorflow_evaluate",
                    tags=("tensorflow", "evaluation"),
                ),
                ToolSpec(
                    name="tensorflow_save_checkpoint",
                    description="Save TensorFlow model checkpoint",
                    input_model=SaveCheckpointArgs,
                    handler_name="_handle_tensorflow_save_checkpoint",
                    tags=("tensorflow", "checkpoint"),
                ),
                ToolSpec(
                    name="tensorflow_load_checkpoint",
                    description="Load TensorFlow model checkpoint",
                    input_model=LoadCheckpointArgs,
                    handler_name="_handle_tensorflow_load_checkpoint",
                    tags=("tensorflow", "checkpoint"),
                ),
                ToolSpec(
                    name="get_training_status",
                    description="Get current training status for a model",
                    input_model=GetTrainingStatusArgs,
                    handler_name="_handle_get_training_status",
                    tags=("training", "status"),
                ),
                ToolSpec(
                    name="list_models",
                    description="List all registered models",
                    input_model=ListModelsArgs,
                    handler_name="_handle_list_models",
                    tags=("models", "listing"),
                ),
                ToolSpec(
                    name="list_runs",
                    description="List persisted orchestration and experiment runs",
                    input_model=ListRunsArgs,
                    handler_name="_handle_list_runs",
                    tags=("runs", "orchestration", "listing"),
                ),
                ToolSpec(
                    name="get_run",
                    description="Get a persisted training orchestration run by id",
                    input_model=GetRunArgs,
                    handler_name="_handle_get_run",
                    tags=("runs", "orchestration"),
                ),
                ToolSpec(
                    name="compare_runs",
                    description="Compare persisted experiment runs by metrics and lineage metadata",
                    input_model=CompareRunsArgs,
                    handler_name="_handle_compare_runs",
                    tags=("runs", "comparison", "experiments"),
                ),
                ToolSpec(
                    name="register_model",
                    description="Register a checkpointed run as a versioned model candidate",
                    input_model=RegisterModelArgs,
                    handler_name="_handle_register_model",
                    tags=("registry", "models", "create"),
                ),
                ToolSpec(
                    name="list_registered_models",
                    description="List registered model versions and promotion stages",
                    input_model=ListRegisteredModelsArgs,
                    handler_name="_handle_list_registered_models",
                    tags=("registry", "models", "listing"),
                ),
                ToolSpec(
                    name="promote_model",
                    description="Promote a registered model to staging, production, or archived",
                    input_model=PromoteModelArgs,
                    handler_name="_handle_promote_model",
                    tags=("registry", "models", "promotion"),
                ),
                ToolSpec(
                    name="reject_model",
                    description="Reject a registered model candidate and record the reason",
                    input_model=RejectModelArgs,
                    handler_name="_handle_reject_model",
                    tags=("registry", "models", "rejection"),
                ),
                ToolSpec(
                    name="predict_online",
                    description="Run an online prediction against a promoted registry model",
                    input_model=PredictOnlineArgs,
                    handler_name="_handle_predict_online",
                    tags=("inference", "serving", "prediction"),
                ),
                ToolSpec(
                    name="predict_batch",
                    description="Run batch predictions against a promoted registry model",
                    input_model=PredictBatchArgs,
                    handler_name="_handle_predict_batch",
                    tags=("inference", "serving", "batch"),
                ),
                ToolSpec(
                    name="get_model_observability",
                    description="Get aggregate prediction observability for a served registry model",
                    input_model=GetModelObservabilityArgs,
                    handler_name="_handle_get_model_observability",
                    tags=("observability", "inference", "monitoring"),
                ),
                ToolSpec(
                    name="get_drift_report",
                    description="Get a drift report derived from served prediction events",
                    input_model=GetDriftReportArgs,
                    handler_name="_handle_get_drift_report",
                    tags=("drift", "monitoring", "inference"),
                ),
                ToolSpec(
                    name="list_dataset_manifests",
                    description="List persisted dataset manifests",
                    input_model=ListDatasetManifestsArgs,
                    handler_name="_handle_list_dataset_manifests",
                    tags=("datasets", "manifests", "listing"),
                ),
                ToolSpec(
                    name="get_dataset_manifest",
                    description="Get a dataset manifest by dataset id and version",
                    input_model=GetDatasetManifestArgs,
                    handler_name="_handle_get_dataset_manifest",
                    tags=("datasets", "manifests"),
                ),
                ToolSpec(
                    name="get_dataset_validation_report",
                    description="Get a persisted dataset validation report by dataset id and version",
                    input_model=GetDatasetValidationReportArgs,
                    handler_name="_handle_get_dataset_validation_report",
                    tags=("datasets", "validation"),
                ),
                ToolSpec(
                    name="list_training_tasks",
                    description="List tasks from the autonomous training queue",
                    input_model=ListTrainingTasksArgs,
                    handler_name="_handle_list_training_tasks",
                    tags=("tasks", "queue", "listing"),
                ),
                ToolSpec(
                    name="create_training_task",
                    description="Create a task in the autonomous training queue",
                    input_model=CreateTrainingTaskArgs,
                    handler_name="_handle_create_training_task",
                    tags=("tasks", "queue", "create"),
                ),
                ToolSpec(
                    name="update_training_task",
                    description="Update text or completion state for a queued training task",
                    input_model=UpdateTrainingTaskArgs,
                    handler_name="_handle_update_training_task",
                    tags=("tasks", "queue", "update"),
                ),
                ToolSpec(
                    name="delete_training_task",
                    description="Delete a queued training task",
                    input_model=DeleteTrainingTaskArgs,
                    handler_name="_handle_delete_training_task",
                    tags=("tasks", "queue", "delete"),
                ),
                ToolSpec(
                    name="get_system_health",
                    description="Inspect resource health, thresholds, and cache usage",
                    input_model=GetSystemHealthArgs,
                    handler_name="_handle_get_system_health",
                    tags=("health", "system"),
                ),
                ToolSpec(
                    name="get_model_recommendation",
                    description="Get Ollama-powered model training recommendations",
                    input_model=GetModelRecommendationArgs,
                    handler_name="_handle_get_model_recommendation",
                    tags=("ollama", "planning"),
                ),
                ToolSpec(
                    name="get_training_analysis",
                    description="Analyze a model's training history with Ollama",
                    input_model=GetTrainingAnalysisArgs,
                    handler_name="_handle_get_training_analysis",
                    tags=("ollama", "analysis", "training"),
                ),
            ]
        )

    def list_specs(self) -> tuple[ToolSpec, ...]:
        return self._specs

    def get_spec(self, name: str) -> ToolSpec | None:
        return self._by_name.get(name)

    def argument_models(self) -> dict[str, type[BaseModel]]:
        return {spec.name: spec.input_model for spec in self._specs}
