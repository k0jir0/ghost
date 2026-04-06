"""Unit tests for agents.training_agent — TrainingAgent task parsing and execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost.config import reset_config
from ghost.context import BackendType, ContextManager


# ---------------------------------------------------------------------------
# Helpers — build a TrainingAgent with all heavy deps mocked
# ---------------------------------------------------------------------------

def _make_agent(tasks_file: Path, agent_memory: Path, tmp_path: Path) -> Any:
    """Return a TrainingAgent isolated from disk and ML libraries."""
    reset_config()

    with (
        patch("ghost.config.GhostConfig.ensure_directories"),
        patch("agents.training_agent.PyTorchOps"),
        patch("agents.training_agent.TensorFlowOps"),
        patch("agents.training_agent.TrainingPipeline"),
        patch("agents.training_agent.OllamaClient"),
        patch("agents.training_agent.ContextManager",
              return_value=ContextManager(storage_path=tmp_path / "ctx")),
    ):
        from agents.training_agent import TrainingAgent

        return TrainingAgent(
            tasks_file=tasks_file,
            agent_memory=agent_memory,
        )


# ---------------------------------------------------------------------------
# Task parsing
# ---------------------------------------------------------------------------

class TestParseTasks:
    def test_returns_only_pending(self, tasks_file: Path, tmp_path: Path) -> None:
        from agents.training_agent import TrainingAgent

        agent = _make_agent(tasks_file, tmp_path / "AGENT.md", tmp_path)
        tasks = agent.parse_tasks()
        texts = [t["text"] for t in tasks]
        # "Setup environment" is already checked [x], should be excluded
        assert "Setup environment" not in texts
        assert "Train MLP on CIFAR-10" in texts

    def test_returns_all_pending_tasks(self, tasks_file: Path, tmp_path: Path) -> None:
        agent = _make_agent(tasks_file, tmp_path / "AGENT.md", tmp_path)
        tasks = agent.parse_tasks()
        assert len(tasks) == 2  # CIFAR-10 and TensorFlow benchmark

    def test_tensorflow_task_included(self, tasks_file: Path, tmp_path: Path) -> None:
        agent = _make_agent(tasks_file, tmp_path / "AGENT.md", tmp_path)
        tasks = agent.parse_tasks()
        texts = [t["text"] for t in tasks]
        assert any("TensorFlow" in t for t in texts)

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        agent = _make_agent(
            tmp_path / "nonexistent.md",
            tmp_path / "AGENT.md",
            tmp_path,
        )
        assert agent.parse_tasks() == []

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "TASKS.md"
        empty.write_text("")
        agent = _make_agent(empty, tmp_path / "AGENT.md", tmp_path)
        assert agent.parse_tasks() == []

    def test_all_completed_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "TASKS.md"
        f.write_text("## Queue\n\n- [x] Done already\n- [x] Also done\n")
        agent = _make_agent(f, tmp_path / "AGENT.md", tmp_path)
        assert agent.parse_tasks() == []

    def test_completed_flag_case_insensitive(self, tmp_path: Path) -> None:
        """- [X] (uppercase X) should also be treated as complete."""
        f = tmp_path / "TASKS.md"
        f.write_text("## Queue\n\n- [X] Done\n- [ ] Pending\n")
        agent = _make_agent(f, tmp_path / "AGENT.md", tmp_path)
        tasks = agent.parse_tasks()
        assert len(tasks) == 1
        assert tasks[0]["text"] == "Pending"


# ---------------------------------------------------------------------------
# Mark task complete
# ---------------------------------------------------------------------------

class TestMarkTaskComplete:
    def test_marks_pending_as_complete(self, tasks_file: Path, tmp_path: Path) -> None:
        agent = _make_agent(tasks_file, tmp_path / "AGENT.md", tmp_path)
        tasks = agent.parse_tasks()
        first = tasks[0]
        agent.mark_task_complete(first)
        # Re-parse — the task should now be gone from pending list
        remaining = agent.parse_tasks()
        remaining_texts = [t["text"] for t in remaining]
        assert first["text"] not in remaining_texts

    def test_mark_idempotent(self, tasks_file: Path, tmp_path: Path) -> None:
        agent = _make_agent(tasks_file, tmp_path / "AGENT.md", tmp_path)
        task = agent.parse_tasks()[0]
        agent.mark_task_complete(task)
        agent.mark_task_complete(task)  # second call should not raise
        remaining = [t["text"] for t in agent.parse_tasks()]
        assert task["text"] not in remaining

    def test_missing_file_does_not_raise(self, tmp_path: Path) -> None:
        agent = _make_agent(
            tmp_path / "gone.md",
            tmp_path / "AGENT.md",
            tmp_path,
        )
        task = {"text": "dummy", "completed": False, "raw": "- [ ] dummy"}
        agent.mark_task_complete(task)  # should not raise


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

class TestBackendDetection:
    """execute_task picks the right backend from the task description."""

    @pytest.mark.asyncio
    async def test_tensorflow_in_task_uses_tf_backend(
        self, tmp_path: Path
    ) -> None:
        tasks_file = tmp_path / "TASKS.md"
        tasks_file.write_text(
            "## Queue\n\n- [ ] Train tensorflow image classifier\n"
        )
        agent = _make_agent(tasks_file, tmp_path / "AGENT.md", tmp_path)

        # Make ops return success
        agent.pytorch_ops.create_model = AsyncMock(return_value={"status": "success"})
        agent.tensorflow_ops.create_model = AsyncMock(return_value={"status": "success"})
        agent.training_pipeline.train = AsyncMock(
            return_value=MagicMock(success=True)
        )
        agent.ollama_client.get_recommendation = AsyncMock(
            return_value={"status": "error"}
        )

        tasks = agent.parse_tasks()
        await agent.execute_task(tasks[0])

        agent.tensorflow_ops.create_model.assert_called_once()
        agent.pytorch_ops.create_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_uses_pytorch_backend(self, tmp_path: Path) -> None:
        tasks_file = tmp_path / "TASKS.md"
        tasks_file.write_text(
            "## Queue\n\n- [ ] Train MLP classifier\n"
        )
        agent = _make_agent(tasks_file, tmp_path / "AGENT.md", tmp_path)

        agent.pytorch_ops.create_model = AsyncMock(return_value={"status": "success"})
        agent.tensorflow_ops.create_model = AsyncMock(return_value={"status": "success"})
        agent.training_pipeline.train = AsyncMock(
            return_value=MagicMock(success=True)
        )
        agent.ollama_client.get_recommendation = AsyncMock(
            return_value={"status": "error"}
        )

        tasks = agent.parse_tasks()
        await agent.execute_task(tasks[0])

        agent.pytorch_ops.create_model.assert_called_once()
        agent.tensorflow_ops.create_model.assert_not_called()


# ---------------------------------------------------------------------------
# execute_task — model creation failure path
# ---------------------------------------------------------------------------

class TestExecuteTask:
    @pytest.mark.asyncio
    async def test_model_creation_failure_returns_error(self, tmp_path: Path) -> None:
        tasks_file = tmp_path / "TASKS.md"
        tasks_file.write_text("## Queue\n\n- [ ] Train something\n")
        agent = _make_agent(tasks_file, tmp_path / "AGENT.md", tmp_path)

        agent.pytorch_ops.create_model = AsyncMock(
            return_value={"status": "error", "message": "GPU OOM"}
        )
        agent.ollama_client.get_recommendation = AsyncMock(
            return_value={"status": "error"}
        )

        tasks = agent.parse_tasks()
        result = await agent.execute_task(tasks[0])

        assert result["success"] is False
        assert "GPU OOM" in result.get("error", "")
        # training should NOT have been called
        agent.training_pipeline.train.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_task_marks_complete(self, tasks_file: Path, tmp_path: Path) -> None:
        agent = _make_agent(tasks_file, tmp_path / "AGENT.md", tmp_path)

        agent.pytorch_ops.create_model = AsyncMock(return_value={"status": "success"})
        agent.tensorflow_ops.create_model = AsyncMock(return_value={"status": "success"})
        agent.training_pipeline.train = AsyncMock(
            return_value=MagicMock(success=True)
        )
        agent.ollama_client.get_recommendation = AsyncMock(
            return_value={"status": "error"}
        )

        tasks = agent.parse_tasks()
        first_text = tasks[0]["text"]
        await agent.execute_task(tasks[0])

        remaining = [t["text"] for t in agent.parse_tasks()]
        assert first_text not in remaining
