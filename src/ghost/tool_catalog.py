"""Transport-independent MCP tool catalog for Ghost."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

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
