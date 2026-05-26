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


class ListModelsArgs(BaseModel):
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
