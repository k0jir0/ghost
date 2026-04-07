"""Unit tests for ghost.training — TrainingPipeline with mocked backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost.context import BackendType, ContextManager, ModelState
from ghost.health_monitor import HealthIssue, ResourceSnapshot
from ghost.training import TrainingConfig, TrainingPipeline, TrainingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_success_step(loss: float = 0.5) -> dict[str, Any]:
    return {
        "status": "success",
        "loss": loss,
        "accuracy": 0.8,
        "data_mode": "synthetic",
    }


def _mock_ops(losses: list[float] | None = None) -> MagicMock:
    """Return a mock BackendOps whose train_step yields supplied loss values."""
    if losses is None:
        losses = [0.5] * 20

    ops = MagicMock()
    results = [_make_success_step(l) for l in losses]
    ops.train_step = AsyncMock(side_effect=results + [{"status": "success", "loss": losses[-1]}] * 1000)
    ops.save_checkpoint = AsyncMock(return_value={"status": "success", "path": "/tmp/ckpt.pt"})
    return ops


class StubHealthMonitor:
    def __init__(self, snapshot: ResourceSnapshot):
        self.snapshot = snapshot

    def check_resources(self) -> ResourceSnapshot:
        return self.snapshot

    def recommended_batch_size(
        self,
        current_batch_size: int,
        snapshot: ResourceSnapshot | None = None,
    ) -> int:
        active_snapshot = snapshot or self.snapshot
        has_memory_pressure = any(
            issue.code in {"system-memory-high", "gpu-memory-high"}
            for issue in active_snapshot.issues
        )
        if has_memory_pressure and current_batch_size > 1:
            return max(1, current_batch_size // 2)
        return current_batch_size


class SharedRuntimeFakePyTorchOps:
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
        runtime = context_manager.get_runtime_bucket("shared-runtime-fake-pytorch")
        self.models: dict[str, dict[str, Any]] = runtime.setdefault("models", {})

    async def create_model(
        self,
        model_id: str,
        model_name: str,
        architecture: str = "mlp",
        num_classes: int = 10,
        input_shape: list[int] | None = None,
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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

    async def save_checkpoint(self, model_id: str) -> dict[str, Any]:
        return {
            "status": "success",
            "path": str(self.context_manager.storage_path / f"{model_id}.pt"),
        }


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_default_epochs(self) -> None:
        cfg = TrainingConfig(model_id="x", backend=BackendType.PYTORCH)
        assert cfg.epochs == 10

    def test_default_batch_size(self) -> None:
        cfg = TrainingConfig(model_id="x", backend=BackendType.PYTORCH)
        assert cfg.batch_size == 32

    def test_early_stopping_patience_positive(self) -> None:
        cfg = TrainingConfig(model_id="x", backend=BackendType.PYTORCH)
        assert cfg.early_stopping_patience > 0


# ---------------------------------------------------------------------------
# TrainingPipeline
# ---------------------------------------------------------------------------

class TestTrainingPipelineContextMissing:
    """pipeline.train() returns an error result when context is absent."""

    def test_missing_context_returns_failure(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        pipeline = TrainingPipeline(context_manager=cm)

        import asyncio

        cfg = TrainingConfig(model_id="no_such_model", backend=BackendType.PYTORCH)
        result: TrainingResult = asyncio.get_event_loop().run_until_complete(
            pipeline.train(cfg)
        )
        assert result.success is False
        assert "not found" in (result.error or "").lower()


class TestTrainingPipelineLoop:
    """_run_training_loop behaves correctly with a mocked backend."""

    def _run(self, pipeline: TrainingPipeline, cfg: TrainingConfig) -> TrainingResult:
        import asyncio

        return asyncio.get_event_loop().run_until_complete(pipeline.train(cfg))

    def test_successful_run_returns_success(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        cm.create_context("m1", "Model1", BackendType.PYTORCH)

        pipeline = TrainingPipeline(context_manager=cm)
        ops = _mock_ops([0.5] * 30)

        cfg = TrainingConfig(
            model_id="m1",
            backend=BackendType.PYTORCH,
            epochs=3,
            steps_per_epoch=5,
            checkpoint_interval=2,
        )

        with patch("ghost.pytorch_ops.PyTorchOps", return_value=ops):
            result = self._run(pipeline, cfg)

        assert result.success is True
        assert result.epochs_completed == 3
        assert len(result.metrics_history) > 0
        assert result.data_mode == "synthetic"
        assert result.used_synthetic_data is True

    def test_early_stopping_triggers(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        cm.create_context("es_model", "EarlyStop", BackendType.PYTORCH)

        pipeline = TrainingPipeline(context_manager=cm)
        # Loss never improves — patience will exhaust
        ops = _mock_ops([0.9] * 100)

        cfg = TrainingConfig(
            model_id="es_model",
            backend=BackendType.PYTORCH,
            epochs=20,
            steps_per_epoch=3,
            early_stopping_patience=2,
        )

        with patch("ghost.pytorch_ops.PyTorchOps", return_value=ops):
            result = self._run(pipeline, cfg)

        # Should stop early, not run all 20 epochs
        assert result.success is True
        assert result.epochs_completed < 20

    def test_checkpoint_saved_at_interval(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        cm.create_context("ckpt_m", "CkptModel", BackendType.PYTORCH)

        pipeline = TrainingPipeline(context_manager=cm)
        ops = _mock_ops([0.5, 0.4, 0.3, 0.2, 0.1] * 10)

        cfg = TrainingConfig(
            model_id="ckpt_m",
            backend=BackendType.PYTORCH,
            epochs=4,
            steps_per_epoch=2,
            checkpoint_interval=2,
        )

        with patch("ghost.pytorch_ops.PyTorchOps", return_value=ops):
            self._run(pipeline, cfg)

        # save_checkpoint should have been called for epochs 2 and 4
        assert ops.save_checkpoint.call_count >= 1

    def test_decreasing_loss_no_early_stop(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        cm.create_context("dec_loss", "DecLoss", BackendType.PYTORCH)

        pipeline = TrainingPipeline(context_manager=cm)
        # Strictly decreasing — should never early-stop
        losses = [1.0 - i * 0.05 for i in range(50)]
        ops = _mock_ops(losses)

        cfg = TrainingConfig(
            model_id="dec_loss",
            backend=BackendType.PYTORCH,
            epochs=5,
            steps_per_epoch=4,
            early_stopping_patience=3,
        )

        with patch("ghost.pytorch_ops.PyTorchOps", return_value=ops):
            result = self._run(pipeline, cfg)

        assert result.epochs_completed == 5

    def test_stop_training_sets_stop_event(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        pipeline = TrainingPipeline(context_manager=cm)

        import asyncio

        event = asyncio.Event()
        pipeline._stop_events["test_m"] = event
        result = pipeline.stop_training("test_m")
        assert result is True
        assert event.is_set()

    def test_stop_training_missing_model(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        pipeline = TrainingPipeline(context_manager=cm)
        assert pipeline.stop_training("ghost") is False

    def test_is_training_true_when_event_exists(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        pipeline = TrainingPipeline(context_manager=cm)

        import asyncio

        pipeline._stop_events["active"] = asyncio.Event()  # not set
        assert pipeline.is_training("active") is True

    def test_is_training_false_when_no_event(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        pipeline = TrainingPipeline(context_manager=cm)
        assert pipeline.is_training("inactive") is False

    def test_degraded_health_reduces_batch_size(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        cm.create_context("pressure", "Pressure", BackendType.PYTORCH)

        snapshot = ResourceSnapshot(
            status="degraded",
            checked_at="2026-04-06T00:00:00",
            system_memory_ratio=0.95,
            gpu_memory_ratio=None,
            model_cache_size_bytes=0,
            data_cache_size_bytes=0,
            issues=[
                HealthIssue(
                    severity="degraded",
                    code="system-memory-high",
                    message="System memory usage exceeded the configured threshold.",
                )
            ],
        )

        pipeline = TrainingPipeline(
            context_manager=cm,
            health_monitor=StubHealthMonitor(snapshot),
        )
        ops = _mock_ops([0.5] * 10)

        cfg = TrainingConfig(
            model_id="pressure",
            backend=BackendType.PYTORCH,
            epochs=1,
            steps_per_epoch=1,
            batch_size=64,
        )

        with patch("ghost.pytorch_ops.PyTorchOps", return_value=ops):
            result = self._run(pipeline, cfg)

        assert result.success is True
        assert result.effective_batch_size == 32
        assert result.health_status == "degraded"
        assert ops.train_step.await_args_list[0].kwargs["batch_size"] == 32

    def test_no_successful_steps_returns_failure(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        cm.create_context("broken", "Broken", BackendType.PYTORCH)

        pipeline = TrainingPipeline(context_manager=cm)
        ops = MagicMock()
        ops.train_step = AsyncMock(
            return_value={"status": "error", "message": "Model not found"}
        )
        ops.save_checkpoint = AsyncMock(return_value={"status": "success"})

        cfg = TrainingConfig(
            model_id="broken",
            backend=BackendType.PYTORCH,
            epochs=1,
            steps_per_epoch=1,
        )

        with patch("ghost.pytorch_ops.PyTorchOps", return_value=ops):
            result = self._run(pipeline, cfg)

        assert result.success is False
        assert result.epochs_completed == 0
        assert "model not found" in (result.error or "").lower()

    def test_create_then_train_works_with_fresh_backend_instance(
        self,
        tmp_data_dir: Path,
    ) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        pipeline = TrainingPipeline(context_manager=cm)
        creator = SharedRuntimeFakePyTorchOps(cm)

        import asyncio

        asyncio.get_event_loop().run_until_complete(
            creator.create_model("shared", "Shared", architecture="mlp")
        )

        cfg = TrainingConfig(
            model_id="shared",
            backend=BackendType.PYTORCH,
            epochs=1,
            steps_per_epoch=1,
        )

        with patch("ghost.pytorch_ops.PyTorchOps", SharedRuntimeFakePyTorchOps):
            result = self._run(pipeline, cfg)

        assert result.success is True
        assert result.epochs_completed == 1
