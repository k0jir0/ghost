"""Training Agent for Ghost platform.

Autonomous agent that watches TASKS.md and executes training tasks.
Similar to Hephaestus agent pattern.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any

from ghost.config import get_config
from ghost.context import ContextManager, BackendType
from ghost.pytorch_ops import PyTorchOps
from ghost.tensorflow_ops import TensorFlowOps
from ghost.training import TrainingPipeline, TrainingConfig
from ghost.ollama_client import OllamaClient
from ghost.logging import get_logger, setup_logging

logger = get_logger(__name__)

_SUPPORTED_ARCHITECTURES = {"mlp", "resnet18", "resnet50", "custom"}


@dataclass
class TrainingTaskPlan:
    """Structured execution plan for a training task."""

    task: str
    backend: BackendType
    architecture: str
    num_classes: int
    batch_size: int
    learning_rate: float
    epochs: int
    dataset: str = ""
    optimizer: str | None = None
    recommendation_source: str = "defaults"
    tips: list[str] = field(default_factory=list)
    raw_recommendations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "backend": self.backend.value,
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "dataset": self.dataset,
            "optimizer": self.optimizer,
            "recommendation_source": self.recommendation_source,
            "tips": self.tips,
            "raw_recommendations": self.raw_recommendations,
        }


class TrainingAgent:
    """Autonomous training agent following Hephaestus patterns.

    Watches TASKS.md for training tasks and executes them autonomously.
    Uses polling-based file watching (no external JS dependencies).
    """

    def __init__(
        self,
        tasks_file: str | Path = "TASKS.md",
        agent_memory: str | Path = "AGENT.md",
        config_path: str | Path = ".env",
    ):
        """Initialize training agent."""
        self.config = get_config()
        self.tasks_file = Path(tasks_file)
        self.agent_memory = Path(agent_memory)

        self.context_manager = ContextManager()
        self.pytorch_ops = PyTorchOps(self.context_manager)
        self.tensorflow_ops = TensorFlowOps(self.context_manager)
        self.training_pipeline = TrainingPipeline(
            self.context_manager,
            backend_ops={
                BackendType.PYTORCH: self.pytorch_ops,
                BackendType.TENSORFLOW: self.tensorflow_ops,
            },
        )
        self.ollama_client = OllamaClient()

        self._running = False
        self._iteration_count = 0
        self._last_task: str | None = None
        self._last_plan: TrainingTaskPlan | None = None
        self._last_analysis: dict[str, Any] | None = None
        self._last_mtime: float | None = None

        # Load agent memory
        self._load_memory()

        logger.info("agent_init", tasks_file=str(self.tasks_file))

    def _load_memory(self) -> None:
        """Load agent memory from AGENT.md."""
        if self.agent_memory.exists():
            try:
                content = self.agent_memory.read_text()
                logger.info("memory_loaded", size=len(content))
            except Exception as e:
                logger.warning("memory_load_failed", error=str(e))

    def _save_memory(self) -> None:
        """Save agent memory to AGENT.md."""
        try:
            last_plan = "None"
            if self._last_plan is not None:
                last_plan = (
                    f"- Backend: {self._last_plan.backend.value}\n"
                    f"- Architecture: {self._last_plan.architecture}\n"
                    f"- Epochs: {self._last_plan.epochs}\n"
                    f"- Batch Size: {self._last_plan.batch_size}\n"
                    f"- Learning Rate: {self._last_plan.learning_rate}\n"
                    f"- Recommendation Source: {self._last_plan.recommendation_source}"
                )

            last_analysis = "None"
            if isinstance(self._last_analysis, dict):
                status = self._last_analysis.get("status", "unknown")
                analysis_payload = self._last_analysis.get("analysis")
                if isinstance(analysis_payload, dict):
                    summary = analysis_payload.get("analysis", "No summary provided.")
                    suggestions = analysis_payload.get("suggestions", [])
                    suggestion_text = ", ".join(str(item) for item in suggestions[:3]) or "None"
                    last_analysis = (
                        f"- Status: {status}\n"
                        f"- Summary: {summary}\n"
                        f"- Suggestions: {suggestion_text}"
                    )
                else:
                    last_analysis = f"- Status: {status}"

            memory_content = f"""# Ghost Agent Memory

## Last Updated
{datetime.now(UTC).isoformat()}

## Iterations
{self._iteration_count}

## Recent Tasks
{self._last_task or "None"}

## Status
Running: {self._running}

## Configuration

- Backend: {self.config.training_backend}
- Ollama Model: {self.config.ollama_model}
- Max Iterations: {self.config.max_iterations}
- Synthetic Demo Data: {self.config.allow_synthetic_data}

## Last Training Plan

{last_plan}

## Last Training Analysis

{last_analysis}

## Notes

This agent watches TASKS.md and executes training tasks autonomously.
"""
            self.agent_memory.write_text(memory_content)
        except Exception as e:
            logger.warning("memory_save_failed", error=str(e))

    def _extract_recommendation_payload(
        self,
        recommendations: dict[str, Any],
    ) -> dict[str, Any]:
        payload = recommendations.get("recommendations")
        return payload if isinstance(payload, dict) else {}

    def _infer_dataset(self, task_text: str) -> str:
        match = re.search(r"\bon\s+([^,]+)", task_text, flags=re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).strip()

    def _coerce_positive_int(
        self,
        value: Any,
        default: int,
        *,
        minimum: int = 1,
        maximum: int = 4096,
    ) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default

        if parsed < minimum:
            return default
        return min(parsed, maximum)

    def _coerce_positive_float(
        self,
        value: Any,
        default: float,
        *,
        minimum: float = 1.0e-6,
        maximum: float = 10.0,
    ) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default

        if parsed < minimum:
            return default
        return min(parsed, maximum)

    def _select_backend(self, task_text: str, recommendations: dict[str, Any]) -> BackendType:
        task_lower = task_text.lower()
        if "tensorflow" in task_lower or "keras" in task_lower:
            return BackendType.TENSORFLOW

        recommended_backend = str(recommendations.get("backend", "")).lower()
        if "tensorflow" in recommended_backend or "keras" in recommended_backend:
            return BackendType.TENSORFLOW
        if "pytorch" in recommended_backend or "torch" in recommended_backend:
            return BackendType.PYTORCH

        configured_backend = self.config.get_backend()
        return (
            BackendType.PYTORCH
            if configured_backend == "pytorch"
            else BackendType.TENSORFLOW
        )

    def _infer_architecture(
        self,
        task_text: str,
        recommendations: dict[str, Any],
    ) -> str:
        recommended = str(recommendations.get("architecture", "")).lower().strip()
        if recommended in _SUPPORTED_ARCHITECTURES:
            return recommended

        task_lower = task_text.lower()
        keyword_map = {
            "resnet50": "resnet50",
            "resnet18": "resnet18",
            "mlp": "mlp",
            "bert": "custom",
            "transformer": "custom",
            "lstm": "custom",
        }
        for keyword, architecture in keyword_map.items():
            if keyword in task_lower:
                return architecture

        return "mlp"

    def _build_plan(
        self,
        task_text: str,
        recommendations: dict[str, Any],
    ) -> TrainingTaskPlan:
        payload = self._extract_recommendation_payload(recommendations)
        recommendation_source = "ollama" if payload else "defaults"
        tips = payload.get("tips", [])
        if not isinstance(tips, list):
            tips = []

        return TrainingTaskPlan(
            task=task_text,
            backend=self._select_backend(task_text, payload),
            architecture=self._infer_architecture(task_text, payload),
            num_classes=self._coerce_positive_int(payload.get("num_classes"), 10, maximum=10_000),
            batch_size=self._coerce_positive_int(
                payload.get("batch_size"),
                self.config.default_batch_size,
            ),
            learning_rate=self._coerce_positive_float(
                payload.get("learning_rate"),
                self.config.default_learning_rate,
            ),
            epochs=self._coerce_positive_int(
                payload.get("epochs"),
                self.config.default_epochs,
                maximum=self.config.max_iterations,
            ),
            dataset=self._infer_dataset(task_text),
            optimizer=str(payload.get("optimizer")) if payload.get("optimizer") else None,
            recommendation_source=recommendation_source,
            tips=[str(item) for item in tips],
            raw_recommendations=payload,
        )

    def parse_tasks(self) -> list[dict[str, Any]]:
        """Parse TASKS.md and extract pending task queue.

        Parses standard GitHub-flavoured markdown checkbox syntax:
          - [ ] pending task
          - [x] completed task
        """
        if not self.tasks_file.exists():
            return []

        try:
            content = self.tasks_file.read_text()
            tasks: list[dict[str, Any]] = []

            lines = content.split("\n")
            in_queue = False

            for line in lines:
                if "## Queue" in line or "## Tasks" in line:
                    in_queue = True
                    continue

                # Stop at the next section header
                if in_queue and line.startswith("## "):
                    break

                if in_queue and line.strip().startswith("- ["):
                    # Extract checkbox state and text
                    stripped = line.strip()
                    is_complete = stripped.lower().startswith("- [x]")
                    # Remove the "- [ ] " or "- [x] " prefix
                    task_text = stripped[5:].strip() if len(stripped) > 5 else ""

                    if task_text:
                        tasks.append(
                            {
                                "text": task_text,
                                "completed": is_complete,
                                "raw": stripped,
                            }
                        )

            return [t for t in tasks if not t["completed"]]
        except Exception as e:
            logger.error("task_parse_failed", error=str(e))
            return []

    def mark_task_complete(self, task: dict[str, Any]) -> None:
        """Mark a task as complete in TASKS.md by updating its checkbox."""
        if not self.tasks_file.exists():
            return

        try:
            content = self.tasks_file.read_text()
            updated = content.replace(
                task["raw"],
                task["raw"].replace("- [ ]", "- [x]").replace("- [X]", "- [x]"),
                1,
            )
            self.tasks_file.write_text(updated)
            logger.info("task_marked_complete", task=task["text"])
        except Exception as e:
            logger.warning("task_mark_failed", error=str(e))

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a single training task."""
        task_text = task["text"]
        logger.info("executing_task", task=task_text)

        try:
            # Get recommendations from Ollama
            recommendations = await self.ollama_client.get_recommendation(
                task=task_text,
            )
            plan = self._build_plan(task_text, recommendations)

            if recommendations.get("status") == "success":
                logger.info(
                    "recommendations_received",
                    recs=recommendations.get("recommendations"),
                )
            logger.info(
                "task_plan_ready",
                task=task_text,
                backend=plan.backend.value,
                architecture=plan.architecture,
                epochs=plan.epochs,
                batch_size=plan.batch_size,
                learning_rate=plan.learning_rate,
                recommendation_source=plan.recommendation_source,
            )

            # Determine backend from task description or fall back to config
            backend = plan.backend

            # Generate a timestamped model ID
            model_id = f"model_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
            model_name = task_text[:50]

            # --- Shift-left fix: create the model (context + weights) BEFORE
            # calling train().  Without this step training_pipeline.train()
            # immediately returns success=False with "Model context not found".
            ops = (
                self.pytorch_ops
                if backend == BackendType.PYTORCH
                else self.tensorflow_ops
            )
            create_result = await ops.create_model(
                model_id=model_id,
                model_name=model_name,
                architecture=plan.architecture,
                num_classes=plan.num_classes,
            )
            if create_result.get("status") != "success":
                logger.error(
                    "model_creation_failed",
                    model_id=model_id,
                    error=create_result.get("message"),
                )
                return {
                    "task": task_text,
                    "success": False,
                    "error": create_result.get("message", "model creation failed"),
                }

            ctx = self.context_manager.get_context(model_id)
            if ctx is not None:
                ctx.metadata["task_text"] = task_text
                ctx.metadata["dataset"] = plan.dataset
                ctx.metadata["training_plan"] = plan.to_dict()
                ctx.metadata["recommendations"] = recommendations
                self.context_manager.update_context(ctx)

            # Build training config from global config defaults
            config = TrainingConfig(
                model_id=model_id,
                backend=backend,
                epochs=plan.epochs,
                batch_size=plan.batch_size,
                learning_rate=plan.learning_rate,
                checkpoint_interval=self.config.checkpoint_interval,
            )

            # Execute training
            result = await self.training_pipeline.train(config)

            analysis: dict[str, Any] | None = None
            metrics_history = getattr(result, "metrics_history", None)
            if isinstance(metrics_history, list) and metrics_history:
                try:
                    analysis = await self.ollama_client.analyze_training_progress(
                        [
                            {
                                "epoch": metric.epoch,
                                "step": metric.step,
                                "loss": metric.loss,
                                "accuracy": metric.accuracy,
                                "learning_rate": metric.learning_rate,
                            }
                            for metric in metrics_history
                        ]
                    )
                    self._last_analysis = analysis

                    ctx = self.context_manager.get_context(model_id)
                    if ctx is not None and isinstance(analysis, dict):
                        ctx.metadata["training_analysis"] = analysis
                        self.context_manager.update_context(ctx)
                except Exception as exc:
                    logger.warning(
                        "training_analysis_unavailable",
                        task=task_text,
                        error=str(exc),
                    )
                    analysis = None
                    self._last_analysis = None
            else:
                self._last_analysis = None

            self._last_task = task_text
            self._last_plan = plan
            self._iteration_count += 1

            logger.info(
                "task_completed",
                task=task_text,
                success=result.success,
                iterations=self._iteration_count,
            )

            if result.success:
                self.mark_task_complete(task)

            return {
                "task": task_text,
                "success": result.success,
                "result": result,
                "plan": plan.to_dict(),
                "analysis": analysis,
            }

        except Exception as e:
            logger.error("task_failed", task=task_text, error=str(e))
            return {
                "task": task_text,
                "success": False,
                "error": str(e),
            }

    async def run_cycle(self) -> None:
        """Run one agent cycle: check for pending tasks and execute the first one."""
        tasks = self.parse_tasks()

        if not tasks:
            logger.info("no_pending_tasks")
            return

        # Check iteration limit before doing any work
        if self._iteration_count >= self.config.max_iterations:
            logger.warning("max_iterations_reached", iterations=self._iteration_count)
            return

        task = tasks[0]
        await self.execute_task(task)
        self._save_memory()

    async def watch_and_run(self) -> None:
        """Watch TASKS.md via polling and run agent cycles on changes."""
        self._running = True
        logger.info("agent_started", poll_interval_seconds=5)

        # Run immediately on start
        await self.run_cycle()

        while self._running:
            await asyncio.sleep(5)

            # Check iteration limit
            if self._iteration_count >= self.config.max_iterations:
                logger.warning(
                    "max_iterations_reached", iterations=self._iteration_count
                )
                break

            # Poll for file changes
            if self.tasks_file.exists():
                try:
                    mtime = self.tasks_file.stat().st_mtime
                    if self._last_mtime is None or self._last_mtime != mtime:
                        self._last_mtime = mtime
                        logger.info("tasks_file_changed")
                        await self.run_cycle()
                except OSError as e:
                    logger.warning("file_stat_failed", error=str(e))

    def stop(self) -> None:
        """Stop the agent."""
        self._running = False
        self._save_memory()
        logger.info("agent_stopped", total_iterations=self._iteration_count)


async def main() -> None:
    """Main entry point for training agent."""
    setup_logging()

    agent = TrainingAgent()

    try:
        await agent.watch_and_run()
    except KeyboardInterrupt:
        agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
