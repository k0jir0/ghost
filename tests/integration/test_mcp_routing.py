"""Integration tests for GhostMCPServer tool routing and Pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost.context import BackendType, ContextManager, TrainingMetrics
from ghost.metadata_store import MetadataStore
from ghost.orchestration import TrainingRunRecord, TrainingRunRequest
from ghost.planning import TrainingPlan
from ghost.run_store import RunStore
from ghost.schemas import ArtifactRecord, DatasetManifest, ExperimentRunRecord
from ghost.task_queue import TaskQueueStore
from ghost.training import TrainingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_server(tmp_data_dir: Path) -> Any:
    """Build a GhostMCPServer with mocked ML backends and Ollama."""
    cm = ContextManager(storage_path=tmp_data_dir)
    task_queue = TaskQueueStore(tmp_data_dir / "TASKS.json")
    metadata_store = MetadataStore(tmp_data_dir / "metadata")

    pytorch_ops = MagicMock()
    tensorflow_ops = MagicMock()
    ollama_client = MagicMock()
    health_monitor = MagicMock()

    pytorch_ops.create_model = AsyncMock(
        return_value={"status": "success", "model_id": "m1", "num_parameters": 1000}
    )
    pytorch_ops.train_step = AsyncMock(
        return_value={"status": "success", "loss": 0.5, "step": 1}
    )
    pytorch_ops.evaluate = AsyncMock(
        return_value={"status": "success", "eval_loss": 0.4}
    )
    pytorch_ops.save_checkpoint = AsyncMock(
        return_value={"status": "success", "path": "/tmp/ckpt.pt"}
    )
    pytorch_ops.load_checkpoint = AsyncMock(
        return_value={"status": "success", "path": "/tmp/ckpt.pt"}
    )

    tensorflow_ops.create_model = AsyncMock(
        return_value={"status": "success", "model_id": "tf1"}
    )
    tensorflow_ops.train_step = AsyncMock(
        return_value={"status": "success", "loss": 0.6}
    )
    tensorflow_ops.evaluate = AsyncMock(
        return_value={"status": "success", "eval_loss": 0.5}
    )
    tensorflow_ops.save_checkpoint = AsyncMock(
        return_value={"status": "success", "path": "/tmp/tf.h5"}
    )
    tensorflow_ops.load_checkpoint = AsyncMock(
        return_value={"status": "success", "path": "/tmp/tf.h5"}
    )

    ollama_client.get_recommendation = AsyncMock(
        return_value={"status": "success", "recommendations": {"architecture": "mlp"}}
    )
    ollama_client.analyze_training_progress = AsyncMock(
        return_value={
            "status": "success",
            "analysis": {
                "status": "good",
                "analysis": "Training is converging normally.",
                "suggestions": ["Continue training"],
            },
        }
    )
    health_monitor.get_health_report.return_value = {
        "status": "healthy",
        "issues": [],
        "system_memory_ratio": 0.42,
        "gpu_memory_ratio": None,
    }

    with (
        patch("ghost.mcp_server.PyTorchOps", return_value=pytorch_ops),
        patch("ghost.mcp_server.TensorFlowOps", return_value=tensorflow_ops),
        patch("ghost.mcp_server.OllamaClient", return_value=ollama_client),
        patch("ghost.mcp_server.ContextManager", return_value=cm),
    ):
        from ghost.mcp_server import GhostMCPServer

        server = GhostMCPServer(
            context_manager=cm,
            ollama_client=ollama_client,
            health_monitor=health_monitor,
            task_queue=task_queue,
            metadata_store=metadata_store,
        )
        # Inject mocked ops directly
        server.pytorch_ops = pytorch_ops
        server.tensorflow_ops = tensorflow_ops
        server.ollama_client = ollama_client
        server.health_monitor = health_monitor
        server.task_queue = task_queue
        server.metadata_store = metadata_store

    return server, cm, pytorch_ops, tensorflow_ops, ollama_client, health_monitor


# ---------------------------------------------------------------------------
# PyTorch tool routing
# ---------------------------------------------------------------------------

class TestPyTorchToolRouting:
    @pytest.mark.asyncio
    async def test_pytorch_create_model(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        result = await server._handle_tool(
            "pytorch_create_model",
            {"model_id": "m1", "model_name": "Net", "architecture": "mlp"},
        )
        assert result.get("status") == "success"

    @pytest.mark.asyncio
    async def test_pytorch_train_step(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        result = await server._handle_tool(
            "pytorch_train_step",
            {"model_id": "m1"},
        )
        assert result.get("status") == "success"

    @pytest.mark.asyncio
    async def test_pytorch_evaluate(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        result = await server._handle_tool("pytorch_evaluate", {"model_id": "m1"})
        assert result.get("status") == "success"

    @pytest.mark.asyncio
    async def test_pytorch_save_checkpoint(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        result = await server._handle_tool(
            "pytorch_save_checkpoint", {"model_id": "m1"}
        )
        assert result.get("status") == "success"

    @pytest.mark.asyncio
    async def test_pytorch_load_checkpoint(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        result = await server._handle_tool(
            "pytorch_load_checkpoint",
            {"model_id": "m1", "path": "/tmp/ckpt.pt"},
        )
        assert result.get("status") == "success"


# ---------------------------------------------------------------------------
# TensorFlow tool routing
# ---------------------------------------------------------------------------

class TestTensorFlowToolRouting:
    @pytest.mark.asyncio
    async def test_tensorflow_create_model(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        result = await server._handle_tool(
            "tensorflow_create_model",
            {"model_id": "tf1", "model_name": "TFNet", "architecture": "mlp"},
        )
        assert result.get("status") == "success"

    @pytest.mark.asyncio
    async def test_tensorflow_train_step(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        result = await server._handle_tool(
            "tensorflow_train_step", {"model_id": "tf1"}
        )
        assert result.get("status") == "success"

    @pytest.mark.asyncio
    async def test_tensorflow_evaluate(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        result = await server._handle_tool(
            "tensorflow_evaluate", {"model_id": "tf1"}
        )
        assert result.get("status") == "success"


# ---------------------------------------------------------------------------
# Context / listing tools
# ---------------------------------------------------------------------------

class TestContextTools:
    @pytest.mark.asyncio
    async def test_list_models_empty(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        result = await server._handle_tool("list_models", {})
        assert "models" in result
        assert isinstance(result["models"], list)

    @pytest.mark.asyncio
    async def test_list_models_populated(self, tmp_data_dir: Path) -> None:
        server, cm, *_ = _make_server(tmp_data_dir)
        cm.create_context("x", "X", BackendType.PYTORCH)
        result = await server._handle_tool("list_models", {})
        assert any(m["model_id"] == "x" for m in result["models"])

    @pytest.mark.asyncio
    async def test_get_training_status_present(self, tmp_data_dir: Path) -> None:
        server, cm, *_ = _make_server(tmp_data_dir)
        cm.create_context("stat", "S", BackendType.PYTORCH)
        result = await server._handle_tool(
            "get_training_status", {"model_id": "stat"}
        )
        assert result.get("model_id") == "stat"
        assert "state" in result

    @pytest.mark.asyncio
    async def test_get_training_status_missing(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        result = await server._handle_tool(
            "get_training_status", {"model_id": "ghost"}
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_model_recommendation(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        result = await server._handle_tool(
            "get_model_recommendation",
            {"task": "image classification"},
        )
        assert result.get("status") == "success"

    @pytest.mark.asyncio
    async def test_get_system_health(self, tmp_data_dir: Path) -> None:
        server, *_, health_monitor = _make_server(tmp_data_dir)
        result = await server._handle_tool("get_system_health", {})
        assert result.get("status") == "healthy"
        health_monitor.get_health_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_training_analysis(self, tmp_data_dir: Path) -> None:
        server, cm, *_, ollama_client, _ = _make_server(tmp_data_dir)
        ctx = cm.create_context("analysis", "Analysis", BackendType.PYTORCH)
        ctx.add_metric(
            TrainingMetrics(
                epoch=1,
                step=1,
                loss=0.5,
                accuracy=0.8,
                learning_rate=0.001,
            )
        )
        cm.update_context(ctx)

        result = await server._handle_tool(
            "get_training_analysis", {"model_id": "analysis"}
        )

        assert result.get("status") == "success"
        ollama_client.analyze_training_progress.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        result = await server._handle_tool("nonexistent_tool", {})
        assert "error" in result


class TestTaskQueueTools:
    @pytest.mark.asyncio
    async def test_create_training_task_writes_to_json_queue(
        self, tmp_data_dir: Path
    ) -> None:
        server, *_ = _make_server(tmp_data_dir)

        result = await server._handle_tool(
            "create_training_task",
            {"text": "Train MLP classifier"},
        )

        assert result["status"] == "success"
        assert result["task"]["text"] == "Train MLP classifier"
        assert result["format"] == "json"

    @pytest.mark.asyncio
    async def test_list_training_tasks_returns_queue_contents(
        self, tmp_data_dir: Path
    ) -> None:
        server, *_ = _make_server(tmp_data_dir)
        await server._handle_tool("create_training_task", {"text": "Train MLP"})

        result = await server._handle_tool("list_training_tasks", {})

        assert len(result["queue"]) == 1
        assert result["queue"][0]["text"] == "Train MLP"

    @pytest.mark.asyncio
    async def test_update_training_task_can_mark_completed(
        self, tmp_data_dir: Path
    ) -> None:
        server, *_ = _make_server(tmp_data_dir)
        created = await server._handle_tool(
            "create_training_task",
            {"text": "Train MLP"},
        )

        result = await server._handle_tool(
            "update_training_task",
            {"task_id": created["task"]["task_id"], "completed": True},
        )

        assert result["status"] == "success"
        assert result["task"]["completed"] is True

    @pytest.mark.asyncio
    async def test_delete_training_task_removes_item(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        created = await server._handle_tool(
            "create_training_task",
            {"text": "Train MLP"},
        )

        result = await server._handle_tool(
            "delete_training_task",
            {"task_id": created["task"]["task_id"]},
        )

        assert result["status"] == "success"
        remaining = await server._handle_tool("list_training_tasks", {})
        assert remaining["queue"] == []


class TestRunMetadataTools:
    @pytest.mark.asyncio
    async def test_list_runs_returns_persisted_run_records(
        self, tmp_data_dir: Path
    ) -> None:
        server, *_ = _make_server(tmp_data_dir)
        record = TrainingRunRecord(
            run_id="run-1",
            model_id="model-1",
            status="completed",
            plan=TrainingPlan(
                task="Train MLP",
                backend=BackendType.PYTORCH,
                architecture="mlp",
                num_classes=10,
                batch_size=32,
                learning_rate=0.001,
                epochs=3,
            ),
            analysis=None,
            events=[],
            request=TrainingRunRequest(task="Train MLP", model_id="model-1"),
            result=TrainingResult(
                model_id="model-1",
                success=True,
                final_loss=0.1,
                epochs_completed=3,
            ),
        )
        server.metadata_store.save_record("runs", record.run_id, record.to_dict())

        result = await server._handle_tool("list_runs", {})

        assert result["count"] == 1
        assert result["runs"][0]["run_id"] == "run-1"

    @pytest.mark.asyncio
    async def test_get_run_returns_persisted_run_by_id(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        record = TrainingRunRecord(
            run_id="run-1",
            model_id="model-1",
            status="completed",
            plan=TrainingPlan(
                task="Train MLP",
                backend=BackendType.PYTORCH,
                architecture="mlp",
                num_classes=10,
                batch_size=32,
                learning_rate=0.001,
                epochs=3,
            ),
            analysis=None,
            events=[],
            request=TrainingRunRequest(task="Train MLP", model_id="model-1"),
            result=TrainingResult(
                model_id="model-1",
                success=True,
                final_loss=0.1,
                epochs_completed=3,
            ),
        )
        server.metadata_store.save_record("runs", record.run_id, record.to_dict())

        result = await server._handle_tool("get_run", {"run_id": "run-1"})

        assert result["run"]["run_id"] == "run-1"
        assert result["run"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_compare_runs_returns_metric_summary(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        run_store = RunStore(metadata_store=server.metadata_store)
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
                metrics={"final_accuracy": 0.8, "final_loss": 0.4},
            )
        )
        run_store.upsert_run(
            ExperimentRunRecord(
                run_id="run-2",
                experiment_id="exp-1",
                model_id="model-2",
                status="completed",
                backend="pytorch",
                architecture="mlp",
                dataset_id="mnist",
                dataset_version="builtin-v1",
                metrics={"final_accuracy": 0.9, "final_loss": 0.2},
            )
        )

        result = await server._handle_tool(
            "compare_runs",
            {"run_ids": ["run-1", "run-2"]},
        )

        assert result["count"] == 2
        assert result["summary"]["best_accuracy_run_id"] == "run-2"


class TestModelRegistryTools:
    @pytest.mark.asyncio
    async def test_register_and_promote_model(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        run_store = RunStore(metadata_store=server.metadata_store)
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
                input_shape=[1, 28, 28],
                num_classes=10,
                metrics={"final_accuracy": 0.9, "final_loss": 0.2},
            )
        )
        run_store.upsert_artifact(
            ArtifactRecord(
                artifact_id="run-1__checkpoint",
                artifact_type="checkpoint",
                uri=str(tmp_data_dir / "model-1.pt"),
                run_id="run-1",
                model_id="model-1",
            )
        )

        created = await server._handle_tool(
            "register_model",
            {"run_id": "run-1", "min_accuracy": 0.8, "max_loss": 0.3},
        )
        promoted = await server._handle_tool(
            "promote_model",
            {
                "registry_id": created["model"]["registry_id"],
                "stage": "production",
                "approved_by": "tester",
            },
        )

        assert created["model"]["stage"] == "draft"
        assert promoted["model"]["stage"] == "production"
        assert "current-production" in promoted["model"]["aliases"]

    @pytest.mark.asyncio
    async def test_reject_model_marks_candidate_rejected(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        run_store = RunStore(metadata_store=server.metadata_store)
        run_store.upsert_run(
            ExperimentRunRecord(
                run_id="run-2",
                experiment_id="exp-2",
                model_id="model-2",
                status="completed",
                backend="pytorch",
                architecture="mlp",
                metrics={"final_accuracy": 0.95, "final_loss": 0.1},
            )
        )
        run_store.upsert_artifact(
            ArtifactRecord(
                artifact_id="run-2__checkpoint",
                artifact_type="checkpoint",
                uri=str(tmp_data_dir / "model-2.pt"),
                run_id="run-2",
                model_id="model-2",
            )
        )
        created = await server._handle_tool(
            "register_model",
            {"run_id": "run-2", "min_accuracy": 0.8},
        )

        rejected = await server._handle_tool(
            "reject_model",
            {
                "registry_id": created["model"]["registry_id"],
                "reason": "baseline changed",
                "rejected_by": "tester",
            },
        )

        assert rejected["model"]["stage"] == "rejected"
        assert rejected["model"]["metadata"]["rejection_reason"] == "baseline changed"


class TestInferenceTools:
    @pytest.mark.asyncio
    async def test_predict_tools_route_through_inference_service(
        self, tmp_data_dir: Path
    ) -> None:
        server, *_ = _make_server(tmp_data_dir)
        server.inference_service = MagicMock()
        server.inference_service.predict_online = AsyncMock(
            return_value={
                "registry_id": "model-1__v1",
                "model_id": "model-1",
                "prediction": {"predicted_class": 1, "scores": [0.1, 0.9]},
            }
        )
        server.inference_service.predict_batch = AsyncMock(
            return_value={
                "registry_id": "model-1__v1",
                "model_id": "model-1",
                "predictions": [
                    {"predicted_class": 1, "scores": [0.1, 0.9]},
                    {"predicted_class": 0, "scores": [0.7, 0.3]},
                ],
                "count": 2,
            }
        )

        online = await server._handle_tool(
            "predict_online",
            {"registry_id": "model-1__v1", "features": [1.0, 2.0]},
        )
        batch = await server._handle_tool(
            "predict_batch",
            {"registry_id": "model-1__v1", "inputs": [[1.0, 2.0], [3.0, 4.0]]},
        )

        assert online["prediction"]["predicted_class"] == 1
        assert batch["count"] == 2


class TestObservabilityTools:
    @pytest.mark.asyncio
    async def test_observability_and_drift_tools(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)
        server.observability.record_prediction(
            "model-1__v1",
            "model-1",
            latency_ms=20.0,
            batch_size=1,
            success=True,
            inputs=[[0.0, 0.0]],
            predictions=[{"predicted_class": 1, "scores": [0.1, 0.9]}],
        )
        server.observability.record_prediction(
            "model-1__v1",
            "model-1",
            latency_ms=25.0,
            batch_size=1,
            success=True,
            inputs=[[10.0, 10.0]],
            predictions=[{"predicted_class": 1, "scores": [0.1, 0.9]}],
        )

        observability = await server._handle_tool(
            "get_model_observability",
            {"registry_id": "model-1__v1"},
        )
        drift = await server._handle_tool(
            "get_drift_report",
            {"registry_id": "model-1__v1"},
        )

        assert observability["observability"]["request_count"] == 2
        assert drift["report"]["sample_count"] == 2


class TestDatasetMetadataTools:
    @pytest.mark.asyncio
    async def test_list_dataset_manifests_returns_persisted_manifests(
        self, tmp_data_dir: Path
    ) -> None:
        server, *_ = _make_server(tmp_data_dir)
        manifest = DatasetManifest(
            dataset_id="mnist",
            version="builtin-v1",
            source_uri="builtin://mnist",
            validation_status="passed",
        )
        server.metadata_store.save_record(
            "dataset-manifests",
            "mnist@builtin-v1",
            manifest.to_dict(),
        )

        result = await server._handle_tool("list_dataset_manifests", {})

        assert result["count"] == 1
        assert result["manifests"][0]["dataset_id"] == "mnist"

    @pytest.mark.asyncio
    async def test_get_dataset_manifest_returns_manifest(
        self, tmp_data_dir: Path
    ) -> None:
        server, *_ = _make_server(tmp_data_dir)
        manifest = DatasetManifest(
            dataset_id="mnist",
            version="builtin-v1",
            source_uri="builtin://mnist",
            validation_status="passed",
        )
        server.metadata_store.save_record(
            "dataset-manifests",
            "mnist@builtin-v1",
            manifest.to_dict(),
        )

        result = await server._handle_tool(
            "get_dataset_manifest",
            {"dataset_id": "mnist", "version": "builtin-v1"},
        )

        assert result["manifest"]["dataset_id"] == "mnist"

    @pytest.mark.asyncio
    async def test_get_dataset_validation_report_returns_report(
        self, tmp_data_dir: Path
    ) -> None:
        server, *_ = _make_server(tmp_data_dir)
        server.metadata_store.save_record(
            "dataset-validation-reports",
            "mnist@builtin-v1",
            {
                "report_id": "mnist@builtin-v1",
                "dataset_id": "mnist",
                "dataset_version": "builtin-v1",
                "status": "passed",
                "issues": [],
                "stats": {"train_samples": 4},
                "created_at": "2026-05-26T00:00:00+00:00",
            },
        )

        result = await server._handle_tool(
            "get_dataset_validation_report",
            {"dataset_id": "mnist", "version": "builtin-v1"},
        )

        assert result["report"]["dataset_id"] == "mnist"
        assert result["report"]["stats"]["train_samples"] == 4


class TestCallToolResultShape:
    def test_call_tool_result_uses_structured_content(self, tmp_data_dir: Path) -> None:
        server, *_ = _make_server(tmp_data_dir)

        result = server._call_tool_result({"status": "success", "model_id": "m1"})

        assert result["structuredContent"] == {
            "status": "success",
            "model_id": "m1",
        }
        assert result["content"] == []


# ---------------------------------------------------------------------------
# Pydantic input validation (shift-left boundary enforcement)
# ---------------------------------------------------------------------------

class TestPydanticValidation:
    """Invalid arguments are rejected before reaching backend ops."""

    def _validate(self, tool_name: str, args: dict[str, Any]) -> Any:
        from ghost.mcp_server import _TOOL_ARG_MODELS
        from pydantic import ValidationError

        model_cls = _TOOL_ARG_MODELS.get(tool_name)
        assert model_cls is not None, f"Unknown tool: {tool_name}"
        try:
            model_cls.model_validate(args)
            return None
        except ValidationError as exc:
            return exc

    def test_create_model_requires_model_id(self) -> None:
        exc = self._validate(
            "pytorch_create_model",
            {"model_name": "Net", "architecture": "mlp"},
        )
        assert exc is not None

    def test_create_model_requires_model_name(self) -> None:
        exc = self._validate(
            "pytorch_create_model",
            {"model_id": "m1", "architecture": "mlp"},
        )
        assert exc is not None

    def test_create_model_rejects_invalid_architecture(self) -> None:
        exc = self._validate(
            "pytorch_create_model",
            {"model_id": "m1", "model_name": "N", "architecture": "vgg"},
        )
        assert exc is not None

    def test_create_model_rejects_zero_num_classes(self) -> None:
        exc = self._validate(
            "pytorch_create_model",
            {
                "model_id": "m1",
                "model_name": "N",
                "architecture": "mlp",
                "num_classes": 0,
            },
        )
        assert exc is not None

    def test_train_step_rejects_negative_lr(self) -> None:
        exc = self._validate(
            "pytorch_train_step",
            {"model_id": "m1", "learning_rate": -0.001},
        )
        assert exc is not None

    def test_train_step_rejects_zero_batch_size(self) -> None:
        exc = self._validate(
            "pytorch_train_step",
            {"model_id": "m1", "batch_size": 0},
        )
        assert exc is not None

    def test_load_checkpoint_requires_path(self) -> None:
        exc = self._validate("pytorch_load_checkpoint", {"model_id": "m1"})
        assert exc is not None

    def test_valid_create_model_passes(self) -> None:
        exc = self._validate(
            "pytorch_create_model",
            {"model_id": "m1", "model_name": "Net", "architecture": "resnet18"},
        )
        assert exc is None

    def test_valid_train_step_passes(self) -> None:
        exc = self._validate(
            "pytorch_train_step",
            {"model_id": "m1", "batch_size": 64, "learning_rate": 0.01},
        )
        assert exc is None

    def test_recommendation_requires_task(self) -> None:
        exc = self._validate("get_model_recommendation", {})
        assert exc is not None

    def test_training_analysis_requires_model_id(self) -> None:
        exc = self._validate("get_training_analysis", {})
        assert exc is not None

    def test_get_run_requires_run_id(self) -> None:
        exc = self._validate("get_run", {})
        assert exc is not None

    def test_compare_runs_requires_two_run_ids(self) -> None:
        exc = self._validate("compare_runs", {"run_ids": ["run-1"]})
        assert exc is not None

    def test_register_model_requires_run_id(self) -> None:
        exc = self._validate("register_model", {})
        assert exc is not None

    def test_promote_model_requires_registry_id(self) -> None:
        exc = self._validate("promote_model", {"stage": "production"})
        assert exc is not None

    def test_reject_model_requires_reason(self) -> None:
        exc = self._validate("reject_model", {"registry_id": "reg-1"})
        assert exc is not None

    def test_predict_online_requires_features(self) -> None:
        exc = self._validate("predict_online", {"registry_id": "reg-1"})
        assert exc is not None

    def test_predict_batch_requires_inputs(self) -> None:
        exc = self._validate("predict_batch", {"registry_id": "reg-1"})
        assert exc is not None

    def test_get_model_observability_requires_registry_id(self) -> None:
        exc = self._validate("get_model_observability", {})
        assert exc is not None

    def test_get_drift_report_requires_registry_id(self) -> None:
        exc = self._validate("get_drift_report", {})
        assert exc is not None

    def test_get_dataset_manifest_requires_dataset_id(self) -> None:
        exc = self._validate("get_dataset_manifest", {})
        assert exc is not None

    def test_update_training_task_requires_target(self) -> None:
        exc = self._validate("update_training_task", {"completed": True})
        assert exc is not None

    def test_update_training_task_requires_change(self) -> None:
        exc = self._validate("update_training_task", {"task_id": "abc"})
        assert exc is not None

    def test_delete_training_task_requires_target(self) -> None:
        exc = self._validate("delete_training_task", {})
        assert exc is not None

    def test_list_models_accepts_empty_args(self) -> None:
        exc = self._validate("list_models", {})
        assert exc is None

    def test_list_runs_accepts_empty_args(self) -> None:
        exc = self._validate("list_runs", {})
        assert exc is None

    def test_list_dataset_manifests_accepts_empty_args(self) -> None:
        exc = self._validate("list_dataset_manifests", {})
        assert exc is None

    def test_list_registered_models_accepts_empty_args(self) -> None:
        exc = self._validate("list_registered_models", {})
        assert exc is None

    def test_list_training_tasks_accepts_empty_args(self) -> None:
        exc = self._validate("list_training_tasks", {})
        assert exc is None

    def test_get_system_health_accepts_empty_args(self) -> None:
        exc = self._validate("get_system_health", {})
        assert exc is None
