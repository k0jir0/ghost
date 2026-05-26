"""Training Agent for Ghost platform.

Autonomous agent that watches the configured task queue and executes training tasks.
Similar to Hephaestus agent pattern.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from ghost.config import get_config
from ghost.context import BackendType, ContextManager
from ghost.datasets import DatasetResolver, dataset_input_shape
from ghost.logging import get_logger, setup_logging
from ghost.ollama_client import OllamaClient
from ghost.planning import PlanningRequest, TrainingPlan, TrainingPlanner
from ghost.pytorch_ops import PyTorchOps
from ghost.task_queue import QueueTask, TaskQueueStore
from ghost.tensorflow_ops import TensorFlowOps
from ghost.training import TrainingPipeline

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentStateSnapshot:
    """Object-backed agent runtime state persisted between runs."""

    last_updated: str = field(default_factory=_utc_now_iso)
    iterations: int = 0
    recent_task: str | None = None
    running: bool = False
    configuration: dict[str, Any] = field(default_factory=dict)
    last_training_plan: dict[str, Any] | None = None
    last_training_analysis: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainingAgent:
    """Autonomous training agent following Hephaestus patterns.

    Watches the configured task queue for training tasks and executes them autonomously.
    Uses polling-based file watching (no external JS dependencies).
    """

    def __init__(
        self,
        tasks_file: str | Path | None = None,
        agent_memory: str | Path | None = None,
        config_path: str | Path = ".env",
    ):
        """Initialize training agent."""
        self.config = get_config()
        self.tasks_file = Path(tasks_file) if tasks_file is not None else self.config.task_queue_file
        self.agent_memory = Path(agent_memory) if agent_memory is not None else self.config.agent_state_file
        self.agent_state_file = self._resolve_agent_state_file(self.agent_memory)
        self.task_queue = TaskQueueStore(self.tasks_file)

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
        self.dataset_resolver = DatasetResolver(config=self.config)
        self.ollama_client = OllamaClient()
        self.planner = TrainingPlanner(
            config=self.config,
            ollama_client=self.ollama_client,
        )

        self._running = False
        self._iteration_count = 0
        self._last_task: str | None = None
        self._last_plan: TrainingPlan | None = None
        self._last_analysis: dict[str, Any] | None = None
        self._last_mtime: float | None = None
        self._last_watch_path: Path | None = None

        # Load agent memory
        self._load_memory()

        logger.info("agent_init", tasks_file=str(self.tasks_file))

    def _resolve_agent_state_file(self, path: Path) -> Path:
        if path.suffix.lower() == ".json":
            return path
        return path.with_suffix(".json")

    def _task_value(
        self,
        task: QueueTask | Mapping[str, Any],
        key: str,
        default: Any = None,
    ) -> Any:
        if isinstance(task, QueueTask):
            return getattr(task, key, default)
        return task.get(key, default)

    def _maybe_restore_plan(self, payload: Any) -> TrainingPlan | None:
        if not isinstance(payload, dict):
            return None

        tips = payload.get("tips", [])
        raw_recommendations = payload.get("raw_recommendations", {})

        try:
            return TrainingPlan(
                task=str(payload["task"]),
                backend=BackendType(str(payload["backend"])),
                architecture=str(payload["architecture"]),
                num_classes=int(payload["num_classes"]),
                batch_size=int(payload["batch_size"]),
                learning_rate=float(payload["learning_rate"]),
                epochs=int(payload["epochs"]),
                dataset=str(payload.get("dataset", "")),
                optimizer=(
                    str(payload["optimizer"])
                    if payload.get("optimizer") is not None
                    else None
                ),
                recommendation_source=str(
                    payload.get("recommendation_source", "defaults")
                ),
                tips=[str(item) for item in tips] if isinstance(tips, list) else [],
                raw_recommendations=(
                    raw_recommendations if isinstance(raw_recommendations, dict) else {}
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _load_memory(self) -> None:
        """Load object-backed agent state from JSON."""
        if self.agent_state_file.exists():
            try:
                payload = json.loads(self.agent_state_file.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Agent state must be a JSON object")

                self._iteration_count = int(payload.get("iterations", 0))
                recent_task = payload.get("recent_task")
                self._last_task = str(recent_task) if recent_task is not None else None
                self._running = bool(payload.get("running", False))
                self._last_plan = self._maybe_restore_plan(
                    payload.get("last_training_plan")
                )
                analysis_payload = payload.get("last_training_analysis")
                self._last_analysis = (
                    analysis_payload if isinstance(analysis_payload, dict) else None
                )
                logger.info(
                    "memory_loaded",
                    path=str(self.agent_state_file),
                    iterations=self._iteration_count,
                )
            except Exception as exc:
                logger.warning("memory_load_failed", error=str(exc))
            return

        if self.agent_memory.exists() and self.agent_memory.suffix.lower() == ".md":
            logger.info("legacy_memory_ignored", path=str(self.agent_memory))

    def _save_memory(self) -> None:
        """Save object-backed agent state to JSON."""
        try:
            snapshot = AgentStateSnapshot(
                last_updated=_utc_now_iso(),
                iterations=self._iteration_count,
                recent_task=self._last_task,
                running=self._running,
                configuration={
                    "backend": self.config.training_backend,
                    "ollama_model": self.config.ollama_model,
                    "max_iterations": self.config.max_iterations,
                    "allow_synthetic_data": self.config.allow_synthetic_data,
                },
                last_training_plan=(
                    self._last_plan.to_dict() if self._last_plan is not None else None
                ),
                last_training_analysis=self._last_analysis,
            )
            self.agent_state_file.write_text(
                json.dumps(snapshot.to_dict(), indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("memory_save_failed", error=str(exc))

    def parse_tasks(self) -> list[QueueTask]:
        """Return pending task objects from the configured queue source."""
        return self.task_queue.pending_tasks()

    def mark_task_complete(self, task: QueueTask | Mapping[str, Any]) -> None:
        """Mark a task as complete in the configured queue source."""
        updated_task = self.task_queue.complete_task(task)
        if updated_task is None:
            logger.warning("task_mark_failed", task=self._task_value(task, "text", ""))
            return
        logger.info("task_marked_complete", task=updated_task.text)

    async def execute_task(
        self,
        task: QueueTask | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Execute a single training task."""
        task_text = str(self._task_value(task, "text", ""))
        logger.info("executing_task", task=task_text)

        try:
            # Get recommendations from Ollama
            recommendations = await self.ollama_client.get_recommendation(
                task=task_text,
            )
            plan = await self.planner.create_plan(
                PlanningRequest(
                    task=task_text,
                    recommendations=recommendations,
                )
            )

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
            dataset_spec = None
            if plan.dataset:
                dataset_spec = self.dataset_resolver.resolve(
                    plan.dataset,
                    allow_synthetic=self.config.allow_synthetic_data,
                )
                if "num_classes" not in plan.raw_recommendations:
                    plan.num_classes = dataset_spec.num_classes

            # Generate a timestamped model ID
            model_id = f"model_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
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
                input_shape=list(dataset_input_shape(dataset_spec, backend))
                if dataset_spec is not None
                else None,
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
                ctx.metadata["dataset"] = dataset_spec.dataset_id if dataset_spec else plan.dataset
                if dataset_spec is not None:
                    ctx.metadata["dataset_spec"] = asdict(dataset_spec)
                ctx.metadata["training_plan"] = plan.to_dict()
                ctx.metadata["recommendations"] = recommendations
                self.context_manager.update_context(ctx)

            # Build training config from global config defaults
            config = plan.to_training_config(
                model_id=model_id,
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
        """Watch the configured queue via polling and run agent cycles on changes."""
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

            # Poll for file changes on the active queue source.
            watch_path = self.task_queue.active_path()
            if watch_path.exists():
                try:
                    mtime = watch_path.stat().st_mtime
                    if (
                        self._last_mtime is None
                        or self._last_mtime != mtime
                        or self._last_watch_path != watch_path
                    ):
                        self._last_mtime = mtime
                        self._last_watch_path = watch_path
                        logger.info("tasks_file_changed", path=str(watch_path))
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
