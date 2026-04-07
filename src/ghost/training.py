"""Training pipeline for Ghost platform.

Unified training interface supporting both PyTorch and TensorFlow backends.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from ghost.context import ContextManager, BackendType, ModelState, TrainingMetrics
from ghost.health_monitor import HealthMonitor
from ghost.logging import get_logger

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class BackendOps(Protocol):
    """Protocol that both PyTorchOps and TensorFlowOps satisfy."""

    async def train_step(
        self,
        model_id: str,
        batch_size: int,
        learning_rate: float,
    ) -> dict[str, Any]: ...

    async def save_checkpoint(self, model_id: str) -> dict[str, Any]: ...


@dataclass
class TrainingConfig:
    """Configuration for a single training run."""

    model_id: str
    backend: BackendType
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    steps_per_epoch: int = 10
    checkpoint_interval: int = 5
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    callbacks: list[Callable[..., Any]] = field(default_factory=list)


@dataclass
class TrainingResult:
    """Result of a completed training run."""

    model_id: str
    success: bool
    final_loss: float
    final_accuracy: float | None = None
    epochs_completed: int = 0
    checkpoint_path: Path | None = None
    duration_seconds: float = 0.0
    metrics_history: list[TrainingMetrics] = field(default_factory=list)
    effective_batch_size: int | None = None
    health_status: str = "unknown"
    health_issues: list[str] = field(default_factory=list)
    data_mode: str = "unknown"
    used_synthetic_data: bool = False
    error: str | None = None


class TrainingPipeline:
    """Unified training pipeline for PyTorch and TensorFlow."""

    def __init__(
        self,
        context_manager: ContextManager | None = None,
        health_monitor: HealthMonitor | None = None,
        backend_ops: dict[BackendType, BackendOps] | None = None,
    ):
        """Initialize training pipeline."""
        self.context_manager = context_manager or ContextManager()
        self.health_monitor = health_monitor or HealthMonitor()
        self._backend_ops = backend_ops or {}
        self.active_trainings: dict[str, asyncio.Task[Any]] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        logger.info("training_pipeline_init")

    def _get_ops(self, backend: BackendType) -> BackendOps:
        """Return a cached backend ops instance for the selected backend."""
        ops = self._backend_ops.get(backend)
        if ops is not None:
            return ops

        if backend == BackendType.PYTORCH:
            from ghost.pytorch_ops import PyTorchOps

            ops = PyTorchOps(self.context_manager)
        else:
            from ghost.tensorflow_ops import TensorFlowOps

            ops = TensorFlowOps(self.context_manager)

        self._backend_ops[backend] = ops
        return ops

    async def train(
        self,
        config: TrainingConfig,
    ) -> TrainingResult:
        """Execute a training run.

        Args:
            config: Training configuration

        Returns:
            Training result with metrics history and checkpoint info.
        """
        start_time = _utc_now()
        ctx = self.context_manager.get_context(config.model_id)

        if not ctx:
            return TrainingResult(
                model_id=config.model_id,
                success=False,
                final_loss=0.0,
                error="Model context not found. Create the model first.",
            )

        if ctx.backend != config.backend:
            return TrainingResult(
                model_id=config.model_id,
                success=False,
                final_loss=0.0,
                error=(
                    f"Model context backend is {ctx.backend.value}, but training requested "
                    f"{config.backend.value}."
                ),
            )

        ctx.update_state(ModelState.TRAINING)
        self.context_manager.update_context(ctx)

        stop_event = asyncio.Event()
        self._stop_events[config.model_id] = stop_event

        logger.info(
            "training_started",
            model_id=config.model_id,
            backend=config.backend.value,
            epochs=config.epochs,
            steps_per_epoch=config.steps_per_epoch,
        )

        try:
            ops = self._get_ops(config.backend)

            result = await self._run_training_loop(ops, config, stop_event)

            duration = (_utc_now() - start_time).total_seconds()
            result.duration_seconds = duration

            final_state = ModelState.READY if result.success else ModelState.FAILED
            ctx.update_state(final_state)
            self.context_manager.update_context(ctx)

            return result

        except Exception as e:
            logger.error("training_failed", model_id=config.model_id, error=str(e))
            ctx.update_state(ModelState.FAILED)
            self.context_manager.update_context(ctx)
            return TrainingResult(
                model_id=config.model_id,
                success=False,
                final_loss=0.0,
                error=str(e),
            )
        finally:
            self._stop_events.pop(config.model_id, None)

    async def _run_training_loop(
        self,
        ops: BackendOps,
        config: TrainingConfig,
        stop_event: asyncio.Event,
    ) -> TrainingResult:
        """Core training loop shared by both backends.

        Runs ``config.epochs`` epochs, each consisting of
        ``config.steps_per_epoch`` gradient steps. Supports early stopping
        and periodic checkpointing.

        Args:
            ops: Backend operations object (PyTorchOps or TensorFlowOps).
            config: Training configuration.
            stop_event: Asyncio event to signal early termination.

        Returns:
            TrainingResult populated with metrics history.
        """
        metrics_history: list[TrainingMetrics] = []
        best_loss = float("inf")
        patience_counter = 0
        epochs_completed = 0
        current_batch_size = config.batch_size
        health_status = "healthy"
        health_issues: list[str] = []
        latest_checkpoint_path: Path | None = None
        data_mode = "unknown"
        used_synthetic_data = False
        encountered_error: str | None = None

        for epoch in range(config.epochs):
            if stop_event.is_set():
                logger.info(
                    "training_stopped_by_request",
                    model_id=config.model_id,
                    epoch=epoch,
                )
                break

            snapshot = self.health_monitor.check_resources()
            if snapshot.status == "degraded":
                health_status = "degraded"
            elif snapshot.status == "warning" and health_status == "healthy":
                health_status = "warning"

            for issue in snapshot.issues:
                if issue.message not in health_issues:
                    health_issues.append(issue.message)

            recommended_batch_size = self.health_monitor.recommended_batch_size(
                current_batch_size,
                snapshot,
            )
            if recommended_batch_size != current_batch_size:
                logger.warning(
                    "training_batch_size_reduced",
                    model_id=config.model_id,
                    previous_batch_size=current_batch_size,
                    new_batch_size=recommended_batch_size,
                    status=snapshot.status,
                )
                current_batch_size = recommended_batch_size

            epoch_metrics: list[TrainingMetrics] = []

            for step in range(config.steps_per_epoch):
                if stop_event.is_set():
                    break

                step_result = await ops.train_step(
                    model_id=config.model_id,
                    batch_size=current_batch_size,
                    learning_rate=config.learning_rate,
                )

                if step_result.get("status") == "success":
                    step_data_mode = str(step_result.get("data_mode", data_mode))
                    data_mode = step_data_mode
                    used_synthetic_data = used_synthetic_data or step_data_mode == "synthetic"
                    metric = TrainingMetrics(
                        epoch=epoch + 1,
                        step=len(metrics_history) + 1,
                        loss=float(step_result.get("loss", 0.0)),
                        accuracy=step_result.get("accuracy"),
                        learning_rate=config.learning_rate,
                    )
                    epoch_metrics.append(metric)
                    metrics_history.append(metric)
                else:
                    encountered_error = str(
                        step_result.get("message", "Training step failed")
                    )
                    logger.warning(
                        "training_step_failed",
                        model_id=config.model_id,
                        epoch=epoch + 1,
                        step=step + 1,
                        error=encountered_error,
                    )
                    break

            if not epoch_metrics:
                if stop_event.is_set():
                    break

                if encountered_error is not None:
                    break

                continue

            epochs_completed = epoch + 1
            avg_loss = sum(m.loss for m in epoch_metrics) / len(epoch_metrics)

            logger.info(
                "epoch_complete",
                model_id=config.model_id,
                epoch=epochs_completed,
                avg_loss=round(avg_loss, 6),
            )

            # Periodic checkpointing
            if epochs_completed % config.checkpoint_interval == 0:
                checkpoint_result = await ops.save_checkpoint(config.model_id)
                if checkpoint_result.get("status") == "success":
                    checkpoint_value = checkpoint_result.get("path")
                    if checkpoint_value:
                        latest_checkpoint_path = Path(str(checkpoint_value))
                    logger.info(
                        "checkpoint_saved",
                        model_id=config.model_id,
                        epoch=epochs_completed,
                    )
                else:
                    logger.warning(
                        "checkpoint_save_failed",
                        model_id=config.model_id,
                        epoch=epochs_completed,
                        error=checkpoint_result.get("message"),
                    )

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    logger.info(
                        "early_stopping_triggered",
                        model_id=config.model_id,
                        epoch=epochs_completed,
                        best_loss=round(best_loss, 6),
                    )
                    break

            if encountered_error is not None:
                break

        # Update context with final epoch count
        ctx = self.context_manager.get_context(config.model_id)
        if ctx:
            ctx.epochs_completed = epochs_completed
            self.context_manager.update_context(ctx)

        final_accuracy: float | None = None
        if metrics_history and metrics_history[-1].accuracy is not None:
            final_accuracy = metrics_history[-1].accuracy

        if encountered_error is not None:
            return TrainingResult(
                model_id=config.model_id,
                success=False,
                final_loss=best_loss if best_loss != float("inf") else 0.0,
                final_accuracy=final_accuracy,
                epochs_completed=epochs_completed,
                checkpoint_path=latest_checkpoint_path,
                metrics_history=metrics_history,
                effective_batch_size=current_batch_size,
                health_status=health_status,
                health_issues=health_issues,
                data_mode=data_mode,
                used_synthetic_data=used_synthetic_data,
                error=encountered_error,
            )

        if not metrics_history:
            return TrainingResult(
                model_id=config.model_id,
                success=False,
                final_loss=0.0,
                final_accuracy=None,
                epochs_completed=0,
                checkpoint_path=latest_checkpoint_path,
                metrics_history=[],
                effective_batch_size=current_batch_size,
                health_status=health_status,
                health_issues=health_issues,
                data_mode=data_mode,
                used_synthetic_data=used_synthetic_data,
                error="Training produced no successful steps.",
            )

        return TrainingResult(
            model_id=config.model_id,
            success=True,
            final_loss=best_loss if best_loss != float("inf") else 0.0,
            final_accuracy=final_accuracy,
            epochs_completed=epochs_completed,
            checkpoint_path=latest_checkpoint_path,
            metrics_history=metrics_history,
            effective_batch_size=current_batch_size,
            health_status=health_status,
            health_issues=health_issues,
            data_mode=data_mode,
            used_synthetic_data=used_synthetic_data,
        )

    def stop_training(self, model_id: str) -> bool:
        """Signal a running training task to stop gracefully.

        Args:
            model_id: Model identifier.

        Returns:
            True if a stop signal was sent, False if not currently training.
        """
        stop_event = self._stop_events.get(model_id)
        if stop_event:
            stop_event.set()
            logger.info("stop_requested", model_id=model_id)
            return True
        return False

    def is_training(self, model_id: str) -> bool:
        """Return True if the given model is actively being trained."""
        event = self._stop_events.get(model_id)
        return event is not None and not event.is_set()
