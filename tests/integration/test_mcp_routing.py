"""Integration tests for GhostMCPServer tool routing and Pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost.context import BackendType, ContextManager, TrainingMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_server(tmp_data_dir: Path) -> Any:
    """Build a GhostMCPServer with mocked ML backends and Ollama."""
    cm = ContextManager(storage_path=tmp_data_dir)

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
        )
        # Inject mocked ops directly
        server.pytorch_ops = pytorch_ops
        server.tensorflow_ops = tensorflow_ops
        server.ollama_client = ollama_client
        server.health_monitor = health_monitor

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

    def test_list_models_accepts_empty_args(self) -> None:
        exc = self._validate("list_models", {})
        assert exc is None

    def test_get_system_health_accepts_empty_args(self) -> None:
        exc = self._validate("get_system_health", {})
        assert exc is None
