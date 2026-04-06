"""Training pipeline for Ghost platform.

Unified training interface supporting both PyTorch and TensorFlow backends.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from ghost.context import ContextManager, BackendType, ModelState, TrainingMetrics
from ghost.config import get_config
from ghost.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')  # Generic model type


@dataclass
class TrainingConfig:
    """Configuration for training run."""
    model_id: str
    backend: BackendType
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    checkpoint_interval: int = 5
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    callbacks: list[Callable] = field(default_factory=list)


@dataclass
class TrainingResult:
    """Result of a training run."""
    model_id: str
    success: bool
    final_loss: float
    final_accuracy: float | None = None
    epochs_completed: int = 0
    checkpoint_path: Path | None = None
    duration_seconds: float = 0.0
    metrics_history: list[TrainingMetrics] = field(default_factory=list)
    error: str | None = None


class TrainingPipeline:
    """Unified training pipeline for PyTorch and TensorFlow."""

    def __init__(
        self,
        context_manager: ContextManager | None = None,
    ):
        """Initialize training pipeline."""
        self.context_manager = context_manager or ContextManager()
        self.active_trainings: dict[str, asyncio.Task] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        logger.info("training_pipeline_init")

    async def train(
        self,
        config: TrainingConfig,
    ) -> TrainingResult:
        """Execute a training run.
        
        Args:
            config: Training configuration
        
        Returns:
            Training result
        """
        start_time = datetime.utcnow()
        ctx = self.context_manager.get_context(config.model_id)
        
        if not ctx:
            return TrainingResult(
                model_id=config.model_id,
                success=False,
                final_loss=0.0,
                error="Model context not found",
            )
        
        ctx.update_state(ModelState.TRAINING)
        self.context_manager.update_context(ctx)
        
        stop_event = asyncio.Event()
        self._stop_events[config.model_id] = stop_event
        
        logger.info("training_started", model_id=config.model_id, epochs=config.epochs)
        
        try:
            if config.backend == BackendType.PYTORCH:
                result = await self._train_pytorch(config, stop_event)
            else:
                result = await self._train_tensorflow(config, stop_event)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            result.duration_seconds = duration
            
            if result.success:
                ctx.update_state(ModelState.READY)
            else:
                ctx.update_state(ModelState.FAILED)
            
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

    async def _train_pytorch(
        self,
        config: TrainingConfig,
        stop_event: asyncio.Event,
    ) -> TrainingResult:
        """Train using PyTorch backend."""
        from ghost.pytorch_ops import PyTorchOps
        
        pytorch_ops = PyTorchOps(self.context_manager)
        
        metrics_history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            if stop_event.is_set():
                logger.info("training_stopped", model_id=config.model_id, epoch=epoch)
                break
            
            epoch_metrics = []
            
            # Simulate training steps per epoch
            steps_per_epoch = 10
            for step in range(steps_per_epoch):
                if stop_event.is_set():
                    break
                
                result = await pytorch_ops.train_step(
                    model_id=config.model_id,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                )
                
                if result.get("status") == "success":
                    metric = TrainingMetrics(
                        epoch=epoch + 1,
                        step=len(metrics_history) + 1,
                        loss=result.get("loss", 0.0),
                        learning_rate=config.learning_rate,
                    )
                    epoch_metrics.append(metric)
                    metrics_history.append(metric)
            
            # Calculate epoch metrics
            if epoch_metrics:
                avg_loss = sum(m.loss for m in epoch_metrics) / len(epoch_metrics)
                
                # Checkpointing
                if (epoch + 1) % config.checkpoint_interval == 0:
                    await pytorch_ops.save_checkpoint(config.model_id)
                    logger.info("checkpoint_saved", model_id=config.model_id, epoch=epoch + 1)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        logger.info("early_stopping", model_id=config.model_id, epoch=epoch + 1)
                        break
        
        ctx = self.context_manager.get_context(config.model_id)
        if ctx:
            ctx.epochs_completed = len(metrics_history) // 10
        
        return TrainingResult(
            model_id=config.model_id,
            success=True,
            final_loss=best_loss if best_loss != float('inf') else 0.0,
            epochs_completed=len(metrics_history) // 10 if metrics_history else 0,
            metrics_history=metrics_history,
        )

    async def _train_tensorflow(
        self,
        config: TrainingConfig,
        stop_event: asyncio.Event,
    ) -> TrainingResult:
        """Train using TensorFlow backend."""
        from ghost.tensorflow_ops import TensorFlowOps
        
        tf_ops = TensorFlowOps(self.context_manager)
        
        metrics_history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            if stop_event.is_set():
                logger.info("training_stopped", model_id=config.model_id, epoch=epoch)
                break
            
            epoch_metrics = []
            
            # Simulate training steps per epoch
            steps_per_epoch = 10
            for step in range(steps_per_epoch):
                if stop_event.is_set():
                    break
                
                result = await tf_ops.train_step(
                    model_id=config.model_id,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                )
                
                if result.get("status") == "success":
                    metric = TrainingMetrics(
                        epoch=epoch + 1,
                        step=len(metrics_history) + 1,
                        loss=result.get("loss", 0.0),
                        accuracy=result.get("accuracy"),
                        learning_rate=config.learning_rate,
                    )
                    epoch_metrics.append(metric)
                    metrics_history.append(metric)
            
            # Calculate epoch metrics
            if epoch_metrics:
                avg_loss = sum(m.loss for m in epoch_metrics) / len(epoch_metrics)
                
                # Checkpointing
                if (epoch + 1) % config.checkpoint_interval == 0:
                    await tf_ops.save_checkpoint(config.model_id)
                    logger.info("checkpoint_saved", model_id=config.model_id, epoch=epoch + 1)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        logger.info("early_stopping", model_id=config.model_id, epoch=epoch + 1)
                        break
        
        ctx = self.context_manager.get_context(config.model_id)
        if ctx:
            ctx.epochs_completed = len(metrics_history) // 10
        
        return TrainingResult(
            model_id=config.model_id,
            success=True,
            final_loss=best_loss if best_loss != float('inf') else 0.0,
            final_accuracy=metrics_history[-1].accuracy if metrics_history else None,
            epochs_completed=len(metrics_history) // 10 if metrics_history else 0,
            metrics_history=metrics_history,
        )

    def stop_training(self, model_id: str) -> bool:
        """Request training to stop.
        
        Args:
            model_id: Model identifier
        
        Returns:
            True if stop was requested
        """
        stop_event = self._stop_events.get(model_id)
        if stop_event:
            stop_event.set()
            logger.info("stop_requested", model_id=model_id)
            return True
        return False

    def is_training(self, model_id: str) -> bool:
        """Check if model is currently training."""
        return model_id in self._stop_events and not self._stop_events[model_id].is_set()
