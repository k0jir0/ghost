"""Unit tests for ghost.orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost.config import GhostConfig, reset_config
from ghost.context import BackendType, ContextManager, TrainingMetrics
from ghost.datasets import DatasetSpec
from ghost.orchestration import TrainingOrchestrator, TrainingRunRequest
from ghost.planning import TrainingPlan
from ghost.training import TrainingResult


@pytest.fixture()
def orchestration_runtime(tmp_path: Path):
    reset_config()
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    context_manager = ContextManager(storage_path=tmp_path / "contexts")

    planner = MagicMock()
    dataset_resolver = MagicMock()
    training_pipeline = MagicMock()
    ollama_client = MagicMock()
    pytorch_ops = MagicMock()
    tensorflow_ops = MagicMock()

    orchestrator = TrainingOrchestrator(
        config=config,
        context_manager=context_manager,
        planner=planner,
        dataset_resolver=dataset_resolver,
        training_pipeline=training_pipeline,
        ollama_client=ollama_client,
        backend_ops={
            BackendType.PYTORCH: pytorch_ops,
            BackendType.TENSORFLOW: tensorflow_ops,
        },
    )

    return {
        "orchestrator": orchestrator,
        "config": config,
        "context_manager": context_manager,
        "planner": planner,
        "dataset_resolver": dataset_resolver,
        "training_pipeline": training_pipeline,
        "ollama_client": ollama_client,
        "pytorch_ops": pytorch_ops,
        "tensorflow_ops": tensorflow_ops,
    }


def _make_plan() -> TrainingPlan:
    return TrainingPlan(
        task="Train MLP on CIFAR-10",
        backend=BackendType.PYTORCH,
        architecture="mlp",
        num_classes=10,
        batch_size=32,
        learning_rate=0.001,
        epochs=3,
        dataset="",
    )


class TestTrainingOrchestrator:
    @pytest.mark.asyncio
    async def test_execute_records_dataset_plan_and_analysis(self, orchestration_runtime) -> None:
        orchestrator = orchestration_runtime["orchestrator"]
        planner = orchestration_runtime["planner"]
        dataset_resolver = orchestration_runtime["dataset_resolver"]
        training_pipeline = orchestration_runtime["training_pipeline"]
        ollama_client = orchestration_runtime["ollama_client"]
        pytorch_ops = orchestration_runtime["pytorch_ops"]

        plan = _make_plan()
        planner.create_plan = AsyncMock(return_value=plan)
        dataset_resolver.resolve.return_value = DatasetSpec(
            dataset_id="cifar-10",
            task_type="image-classification",
            source="builtin-catalog",
            input_shape=(3, 32, 32),
            num_classes=10,
            synthetic=False,
        )
        pytorch_ops.create_model = AsyncMock(return_value={"status": "success"})
        training_pipeline.train = AsyncMock(
            return_value=TrainingResult(
                model_id="model-123",
                success=True,
                final_loss=0.12,
                epochs_completed=3,
                metrics_history=[
                    TrainingMetrics(
                        epoch=1,
                        step=1,
                        loss=0.12,
                        accuracy=0.88,
                        learning_rate=0.001,
                    )
                ],
            )
        )
        ollama_client.analyze_training_progress = AsyncMock(
            return_value={"status": "success", "analysis": {"analysis": "stable"}}
        )

        record = await orchestrator.execute(
            TrainingRunRequest(
                task="Train MLP on CIFAR-10",
                dataset_ref="cifar10",
                model_id="model-123",
            )
        )

        assert record.status == "completed"
        assert record.plan is plan
        assert record.plan.dataset == "cifar-10"
        dataset_resolver.resolve.assert_called_once_with(
            "cifar10",
            allow_synthetic=False,
        )
        pytorch_ops.create_model.assert_awaited_once()
        assert pytorch_ops.create_model.await_args.kwargs["input_shape"] == [3, 32, 32]
        training_pipeline.train.assert_awaited_once()
        ollama_client.analyze_training_progress.assert_awaited_once()
        assert any(event["stage"] == "dataset_resolved" for event in record.events)
        assert any(event["stage"] == "training_completed" for event in record.events)

    @pytest.mark.asyncio
    async def test_resume_retries_run_from_recorded_state(self, orchestration_runtime) -> None:
        orchestrator = orchestration_runtime["orchestrator"]
        planner = orchestration_runtime["planner"]
        training_pipeline = orchestration_runtime["training_pipeline"]
        ollama_client = orchestration_runtime["ollama_client"]
        pytorch_ops = orchestration_runtime["pytorch_ops"]

        plan = _make_plan()
        planner.create_plan = AsyncMock(return_value=plan)
        pytorch_ops.create_model = AsyncMock(return_value={"status": "success"})
        pytorch_ops.load_checkpoint = AsyncMock(return_value={"status": "error", "message": "no checkpoint"})
        training_pipeline.train = AsyncMock(
            side_effect=[
                TrainingResult(
                    model_id="model-resume",
                    success=False,
                    final_loss=0.7,
                    error="stopped",
                ),
                TrainingResult(
                    model_id="model-resume",
                    success=True,
                    final_loss=0.2,
                    epochs_completed=2,
                    metrics_history=[
                        TrainingMetrics(
                            epoch=1,
                            step=1,
                            loss=0.2,
                            accuracy=0.9,
                            learning_rate=0.001,
                        )
                    ],
                ),
            ]
        )
        ollama_client.analyze_training_progress = AsyncMock(
            return_value={"status": "success", "analysis": {"analysis": "recovered"}}
        )

        record = await orchestrator.execute(
            TrainingRunRequest(
                task="Train MLP",
                model_id="model-resume",
            )
        )
        assert record.status == "failed"

        resumed = await orchestrator.resume(record.run_id)

        assert resumed.status == "completed"
        assert resumed.run_id == record.run_id
        assert training_pipeline.train.await_count == 2
        assert pytorch_ops.create_model.await_count == 2
        assert any(event["stage"] == "resume_requested" for event in resumed.events)
        assert any(event["stage"] == "resume_completed" for event in resumed.events)

    @pytest.mark.asyncio
    async def test_execute_persists_run_record_for_fresh_orchestrator(self, orchestration_runtime) -> None:
        orchestrator = orchestration_runtime["orchestrator"]
        planner = orchestration_runtime["planner"]
        training_pipeline = orchestration_runtime["training_pipeline"]
        ollama_client = orchestration_runtime["ollama_client"]
        pytorch_ops = orchestration_runtime["pytorch_ops"]
        config = orchestration_runtime["config"]
        context_manager = orchestration_runtime["context_manager"]

        plan = _make_plan()
        planner.create_plan = AsyncMock(return_value=plan)
        pytorch_ops.create_model = AsyncMock(return_value={"status": "success"})
        pytorch_ops.save_checkpoint = AsyncMock(
            return_value={
                "status": "success",
                "path": str(config.model_cache_dir / "model-persisted.pt"),
            }
        )
        training_pipeline.train = AsyncMock(
            return_value=TrainingResult(
                model_id="model-persisted",
                success=True,
                final_loss=0.15,
                epochs_completed=3,
                metrics_history=[
                    TrainingMetrics(
                        epoch=1,
                        step=1,
                        loss=0.15,
                        accuracy=0.85,
                        learning_rate=0.001,
                    )
                ],
            )
        )
        ollama_client.analyze_training_progress = AsyncMock(
            return_value={"status": "success", "analysis": {"analysis": "stable"}}
        )

        record = await orchestrator.execute(
            TrainingRunRequest(
                task="Train MLP",
                model_id="model-persisted",
            )
        )

        reloaded = TrainingOrchestrator(
            config=config,
            context_manager=ContextManager(storage_path=context_manager.storage_path),
            planner=MagicMock(),
            dataset_resolver=MagicMock(),
            training_pipeline=MagicMock(),
            ollama_client=MagicMock(),
            backend_ops={
                BackendType.PYTORCH: MagicMock(),
                BackendType.TENSORFLOW: MagicMock(),
            },
        )

        restored = reloaded.get_run(record.run_id)
        experiment_run = reloaded.run_store.get_run(record.run_id)

        assert restored is not None
        assert restored.run_id == record.run_id
        assert restored.status == "completed"
        assert restored.request is not None
        assert restored.request.task == "Train MLP"
        assert restored.result is not None
        assert restored.result.metrics_history[0].loss == pytest.approx(0.15)
        assert restored.result.checkpoint_path == config.model_cache_dir / "model-persisted.pt"
        assert experiment_run is not None
        assert experiment_run.run_id == record.run_id
        assert experiment_run.metrics["final_loss"] == pytest.approx(0.15)

    @pytest.mark.asyncio
    async def test_resume_works_after_orchestrator_restart(self, orchestration_runtime) -> None:
        orchestrator = orchestration_runtime["orchestrator"]
        planner = orchestration_runtime["planner"]
        training_pipeline = orchestration_runtime["training_pipeline"]
        ollama_client = orchestration_runtime["ollama_client"]
        pytorch_ops = orchestration_runtime["pytorch_ops"]
        config = orchestration_runtime["config"]
        context_manager = orchestration_runtime["context_manager"]

        plan = _make_plan()
        planner.create_plan = AsyncMock(return_value=plan)
        pytorch_ops.create_model = AsyncMock(return_value={"status": "success"})
        pytorch_ops.load_checkpoint = AsyncMock(
            return_value={"status": "error", "message": "no checkpoint"}
        )
        training_pipeline.train = AsyncMock(
            return_value=TrainingResult(
                model_id="model-restart",
                success=False,
                final_loss=0.4,
                error="stopped",
            )
        )
        ollama_client.analyze_training_progress = AsyncMock(return_value=None)

        record = await orchestrator.execute(
            TrainingRunRequest(
                task="Train MLP",
                model_id="model-restart",
            )
        )

        resumed_pipeline = MagicMock()
        resumed_pipeline.train = AsyncMock(
            return_value=TrainingResult(
                model_id="model-restart",
                success=True,
                final_loss=0.1,
                epochs_completed=2,
                metrics_history=[
                    TrainingMetrics(
                        epoch=1,
                        step=1,
                        loss=0.1,
                        accuracy=0.92,
                        learning_rate=0.001,
                    )
                ],
            )
        )
        resumed_ollama = MagicMock()
        resumed_ollama.analyze_training_progress = AsyncMock(
            return_value={"status": "success", "analysis": {"analysis": "recovered"}}
        )
        resumed_pytorch = MagicMock()
        resumed_pytorch.create_model = AsyncMock(return_value={"status": "success"})
        resumed_pytorch.load_checkpoint = AsyncMock(
            return_value={"status": "error", "message": "no checkpoint"}
        )

        reloaded = TrainingOrchestrator(
            config=config,
            context_manager=ContextManager(storage_path=context_manager.storage_path),
            planner=MagicMock(),
            dataset_resolver=MagicMock(),
            training_pipeline=resumed_pipeline,
            ollama_client=resumed_ollama,
            backend_ops={
                BackendType.PYTORCH: resumed_pytorch,
                BackendType.TENSORFLOW: MagicMock(),
            },
        )

        resumed = await reloaded.resume(record.run_id)

        assert resumed.status == "completed"
        assert resumed.result is not None
        assert resumed.result.final_loss == pytest.approx(0.1)
        resumed_pipeline.train.assert_awaited_once()