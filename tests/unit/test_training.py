"""Unit tests for ghost.training — TrainingPipeline with mocked backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost.context import BackendType, ContextManager, ModelState
from ghost.training import TrainingConfig, TrainingPipeline, TrainingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_success_step(loss: float = 0.5) -> dict[str, Any]:
    return {"status": "success", "loss": loss, "accuracy": 0.8}


def _mock_ops(losses: list[float] | None = None) -> MagicMock:
    """Return a mock BackendOps whose train_step yields supplied loss values."""
    if losses is None:
        losses = [0.5] * 20

    ops = MagicMock()
    results = [_make_success_step(l) for l in losses]
    ops.train_step = AsyncMock(side_effect=results + [{"status": "success", "loss": losses[-1]}] * 1000)
    ops.save_checkpoint = AsyncMock(return_value={"status": "success", "path": "/tmp/ckpt.pt"})
    return ops


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

        with patch("ghost.training.PyTorchOps", return_value=ops):
            result = self._run(pipeline, cfg)

        assert result.success is True
        assert result.epochs_completed == 3
        assert len(result.metrics_history) > 0

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

        with patch("ghost.training.PyTorchOps", return_value=ops):
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

        with patch("ghost.training.PyTorchOps", return_value=ops):
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

        with patch("ghost.training.PyTorchOps", return_value=ops):
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
