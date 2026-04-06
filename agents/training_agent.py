"""Training Agent for Ghost platform.

Autonomous agent that watches TASKS.md and executes training tasks.
Similar to Hephaestus agent pattern.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import chokidar
import gray_matter

from ghost.config import get_config
from ghost.context import ContextManager, BackendType
from ghost.training import TrainingPipeline, TrainingConfig
from ghost.ollama_client import OllamaClient
from ghost.logging import get_logger, setup_logging

logger = get_logger(__name__)


class TrainingAgent:
    """Autonomous training agent following Hephaestus patterns.
    
    Watches TASKS.md for training tasks and executes them autonomously.
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
        self.training_pipeline = TrainingPipeline(self.context_manager)
        self.ollama_client = OllamaClient()
        
        self._running = False
        self._iteration_count = 0
        self._last_task: str | None = None
        
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
"""
            self.agent_memory.write_text(memory_content)
        except Exception as e:
            logger.warning("memory_save_failed", error=str(e))

    def parse_tasks(self) -> list[dict[str, Any]]:
        """Parse TASKS.md and extract task queue."""
        if not self.tasks_file.exists():
            return []
        
        try:
            content = self.tasks_file.read_text()
            tasks = []
            
            # Simple markdown task parsing
            lines = content.split('\n')
            in_queue = False
            
            for line in lines:
                if '## Queue' in line or '## Tasks' in line:
                    in_queue = True
                    continue
                
                if in_queue and line.startswith('## '):
                    break
                
                if in_queue and line.strip().startswith('- ['):
                    task_text = line.strip()[4:].strip()
                    if task_text.startswith(']'):
                        task_text = task_text[1:].strip()
                    
                    is_complete = '[x]' in line.lower()
                    
                    tasks.append({
                        'text': task_text,
                        'completed': is_complete,
                        'raw': line.strip()
                    })
            
            return [t for t in tasks if not t['completed']]
        except Exception as e:
            logger.error("task_parse_failed", error=str(e))
            return []

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a single training task."""
        task_text = task['text']
        logger.info("executing_task", task=task_text)
        
        try:
            # Get recommendations from Ollama
            recommendations = await self.ollama_client.get_recommendation(
                task=task_text,
            )
            
            if recommendations.get('status') == 'success':
                logger.info("recommendations_received", 
                           recs=recommendations.get('recommendations'))
            
            # Determine backend from task or config
            backend = BackendType.PYTORCH if 'pytorch' in task_text.lower() else \
                     BackendType.TENSORFLOW if 'tensorflow' in task_text.lower() else \
                     BackendType.PYTORCH  # default
            
            # Extract model name
            model_id = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            model_name = task_text[:50]  # Truncate for name
            
            # Create training config
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
            
            logger.info("task_completed", 
                       task=task_text,
                       success=result.success,
                       iterations=self._iteration_count)
            
            return {
                'task': task_text,
                'success': result.success,
                'result': result,
            }
            
        except Exception as e:
            logger.error("task_failed", task=task_text, error=str(e))
            return {
                'task': task_text,
                'success': False,
                'error': str(e),
            }

    async def run_cycle(self) -> None:
        """Run one agent cycle."""
        tasks = self.parse_tasks()
        
        if not tasks:
            logger.info("no_pending_tasks")
            return
        
        # Get next task
        task = tasks[0]
        
        # Check iteration limit
        if self._iteration_count >= self.config.max_iterations:
            logger.warning("max_iterations_reached", iterations=self._iteration_count)
            return
        
        # Execute task
        result = await self.execute_task(task)
        
        # Save progress
        self._save_memory()

    async def watch_and_run(self) -> None:
        """Watch TASKS.md and run agent cycles."""
        self._running = True
        logger.info("agent_started")
        
        # Set up file watcher
        watcher = chokidar.watch(str(self.tasks_file), poll_interval=1)
        
        async def on_change(event: str) -> None:
            if event in ('modified', 'add'):
                await self.run_cycle()
        
        # Initial run
        await self.run_cycle()
        
        # Watch loop
        while self._running:
            await asyncio.sleep(5)
            
            # Periodic check
            if self.tasks_file.exists():
                mtime = self.tasks_file.stat().st_mtime
                if not hasattr(self, '_last_mtime') or self._last_mtime != mtime:
                    self._last_mtime = mtime
                    await self.run_cycle()
            
            # Check iteration limit
            if self._iteration_count >= self.config.max_iterations:
                logger.warning("max_iterations_reached", iterations=self._iteration_count)
                break

    def stop(self) -> None:
        """Stop the agent."""
        self._running = False
        self._save_memory()
        logger.info("agent_stopped")


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
