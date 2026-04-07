"""Integration tests for the real TrainingAgent runtime handoff."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost.config import reset_config
from ghost.context import BackendType, ContextManager, ModelState


class FakePyTorchOps:
    instances: list["FakePyTorchOps"] = []

    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
        self.models: dict[str, dict[str, object]] = {}
        self.__class__.instances.append(self)

    async def create_model(
        self,
        model_id: str,
        model_name: str,
        architecture: str = "mlp",
        num_classes: int = 10,
        input_shape: list[int] | None = None,
    ) -> dict[str, object]:
        self.models[model_id] = {"architecture": architecture}
        ctx = self.context_manager.create_context(
            model_id=model_id,
            model_name=model_name,
            backend=BackendType.PYTORCH,
            architecture=architecture,
            num_classes=num_classes,
            input_shape=input_shape or [3, 224, 224],
        )
        ctx.update_state(ModelState.READY)
        self.context_manager.update_context(ctx)
        return {"status": "success", "model_id": model_id}

    async def train_step(
        self,
        model_id: str,
        batch_size: int,
        learning_rate: float,
    ) -> dict[str, object]:
        if model_id not in self.models:
            return {"status": "error", "message": "Model not found"}

        return {
            "status": "success",
            "loss": 0.25,
            "accuracy": 0.9,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "data_mode": "synthetic",
        }

    async def save_checkpoint(self, model_id: str) -> dict[str, object]:
        return {
            "status": "success",
            "path": str(self.context_manager.storage_path / f"{model_id}.pt"),
        }


class FakeTensorFlowOps(FakePyTorchOps):
    instances: list["FakeTensorFlowOps"] = []

    async def create_model(
        self,
        model_id: str,
        model_name: str,
        architecture: str = "mlp",
        num_classes: int = 10,
        input_shape: list[int] | None = None,
    ) -> dict[str, object]:
        self.models[model_id] = {"architecture": architecture}
        ctx = self.context_manager.create_context(
            model_id=model_id,
            model_name=model_name,
            backend=BackendType.TENSORFLOW,
            architecture=architecture,
            num_classes=num_classes,
            input_shape=input_shape or [224, 224, 3],
        )
        ctx.update_state(ModelState.READY)
        self.context_manager.update_context(ctx)
        return {"status": "success", "model_id": model_id}


@pytest.mark.asyncio
async def test_execute_task_reuses_the_agent_backend_instance(tmp_path: Path) -> None:
    reset_config()
    FakePyTorchOps.instances = []

    tasks_file = tmp_path / "TASKS.md"
    tasks_file.write_text("## Queue\n\n- [ ] Train MLP classifier\n")
    shared_context = ContextManager(storage_path=tmp_path / "ctx")

    ollama_client = MagicMock()
    ollama_client.get_recommendation = AsyncMock(return_value={"status": "error"})

    with (
        patch("ghost.config.GhostConfig.ensure_directories"),
        patch("agents.training_agent.ContextManager", return_value=shared_context),
        patch("agents.training_agent.PyTorchOps", FakePyTorchOps),
        patch("agents.training_agent.TensorFlowOps", FakeTensorFlowOps),
        patch("ghost.pytorch_ops.PyTorchOps", FakePyTorchOps),
        patch("ghost.tensorflow_ops.TensorFlowOps", FakeTensorFlowOps),
        patch("agents.training_agent.OllamaClient", return_value=ollama_client),
    ):
        from agents.training_agent import TrainingAgent

        agent = TrainingAgent(
            tasks_file=tasks_file,
            agent_memory=tmp_path / "AGENT.md",
        )

    task = agent.parse_tasks()[0]
    result = await agent.execute_task(task)

    assert result["success"] is True
    assert result["result"].epochs_completed > 0
    assert len(FakePyTorchOps.instances) == 1
    assert agent.parse_tasks() == []