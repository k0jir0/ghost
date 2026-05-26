"""Unit tests for ghost.inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ghost.config import GhostConfig, reset_config
from ghost.context import BackendType, ContextManager
from ghost.inference import InferenceService
from ghost.metadata_store import MetadataStore
from ghost.model_registry import ModelRegistry
from ghost.run_store import RunStore
from ghost.schemas import ArtifactRecord, ExperimentRunRecord


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(4, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]))
            self.linear.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = self.flatten(inputs)
        return self.linear(flattened)


class _FakePyTorchOps:
    def __init__(self) -> None:
        self.models: dict[str, torch.nn.Module] = {}

    async def create_model(
        self,
        model_id: str,
        model_name: str,
        architecture: str = "mlp",
        num_classes: int = 2,
        input_shape: list[int] | None = None,
    ) -> dict[str, object]:
        self.models[model_id] = _ToyModel()
        return {"status": "success", "model_id": model_id}

    async def load_checkpoint(self, model_id: str, path: str) -> dict[str, object]:
        return {"status": "success", "path": path}


@pytest.mark.asyncio
async def test_inference_service_predicts_from_promoted_model(tmp_path: Path) -> None:
    reset_config()
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    metadata_store = MetadataStore(config.data_cache_dir / "metadata")
    run_store = RunStore(config=config, metadata_store=metadata_store)
    model_registry = ModelRegistry(
        config=config,
        metadata_store=metadata_store,
        run_store=run_store,
    )
    context_manager = ContextManager(storage_path=tmp_path / "contexts")
    fake_ops = _FakePyTorchOps()

    run_store.upsert_run(
        ExperimentRunRecord(
            run_id="run-1",
            experiment_id="exp-1",
            model_id="model-1",
            status="completed",
            backend="pytorch",
            architecture="mlp",
            dataset_id="mnist",
            dataset_version="builtin-v1",
            input_shape=[1, 2, 2],
            num_classes=2,
            metrics={"final_accuracy": 0.95, "final_loss": 0.1},
        )
    )
    run_store.upsert_artifact(
        ArtifactRecord(
            artifact_id="run-1__checkpoint",
            artifact_type="checkpoint",
            uri=str(tmp_path / "models" / "model-1.pt"),
            run_id="run-1",
            model_id="model-1",
        )
    )
    record = model_registry.register_model("run-1")
    model_registry.promote_model(record.registry_id, stage="production")

    service = InferenceService(
        config=config,
        context_manager=context_manager,
        model_registry=model_registry,
        backend_ops={BackendType.PYTORCH: fake_ops},
    )

    approved = await service._approved_record(record.registry_id)
    model = await service._ensure_runtime_model(approved)
    predictions = service._prediction_payloads(
        np.array([[0.1, 0.9]], dtype=np.float32)
    )

    assert model is not None
    assert predictions[0]["predicted_class"] == 1
    assert len(predictions[0]["scores"]) == 2