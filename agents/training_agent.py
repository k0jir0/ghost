"""Training Agent for Ghost platform.

Autonomous agent that watches TASKS.md and executes training tasks.
Similar to Hephaestus agent pattern.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from ghost.config import get_config
from ghost.context import ContextManager, BackendType
from ghost.pytorch_ops import PyTorchOps
from ghost.tensorflow_ops import TensorFlowOps
from ghost.training import TrainingPipeline, TrainingConfig
from ghost.ollama_client import OllamaClient
from ghost.logging import get_logger, setup_logging

logger = get_logger(__name__)


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
        self.training_pipeline = TrainingPipeline(self.context_manager)
        self.ollama_client = OllamaClient()

        self._running = False
        self._iteration_count = 0
        self._last_task: str | None = None
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
            memory_content = f"""# Ghost Agent Memory

## Last Updated
{datetime.utcnow().isoformat()}

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
- Daily Token Budget: ${self.config.daily_token_budget:.2f}

## Notes

This agent watches TASKS.md and executes training tasks autonomously.
"""
            self.agent_memory.write_text(memory_content)
        except Exception as e:
            logger.warning("memory_save_failed", error=str(e))

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

            if recommendations.get("status") == "success":
                logger.info(
                    "recommendations_received",
                    recs=recommendations.get("recommendations"),
                )

            # Determine backend from task description or fall back to config
            task_lower = task_text.lower()
            if "tensorflow" in task_lower or "keras" in task_lower:
                backend = BackendType.TENSORFLOW
            else:
                backend = BackendType.PYTORCH  # default

            # Generate a timestamped model ID
            model_id = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
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
                architecture="mlp",
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

            # Build training config from global config defaults
            config = TrainingConfig(
                model_id=model_id,
                backend=backend,
                epochs=self.config.default_epochs,
                batch_size=self.config.default_batch_size,
                learning_rate=self.config.default_learning_rate,
                checkpoint_interval=self.config.checkpoint_interval,
            )

            # Execute training
            result = await self.training_pipeline.train(config)

            self._last_task = task_text
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
