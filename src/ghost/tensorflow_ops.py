"""TensorFlow operations for Ghost MCP tools.

Provides MCP tools for TensorFlow/Keras model creation, training, and evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ghost.config import get_config
from ghost.context import BackendType, ContextManager, ModelState, TrainingMetrics
from ghost.logging import get_logger

logger = get_logger(__name__)

# Lazy import TensorFlow to avoid hard dependency
_tf = None


def _get_tf():
    """Lazy load TensorFlow."""
    global _tf
    if _tf is None:
        import tensorflow as tf

        _tf = tf
    return _tf


class TensorFlowOps:
    """TensorFlow operations handler for MCP tools."""

    def __init__(self, context_manager: ContextManager):
        """Initialize TensorFlow operations."""
        self.context_manager = context_manager
        self.config = get_config()
        runtime = self.context_manager.get_runtime_bucket("tensorflow")
        self.models: dict[str, Any] = runtime.setdefault("models", {})
        self._initialized = False
        logger.info("tensorflow_ops_init")

    def _ensure_synthetic_data_enabled(self, operation: str) -> None:
        """Fail closed when no real dataset pipeline exists."""
        if self.config.allow_synthetic_data:
            return

        raise RuntimeError(
            f"{operation} requires a real dataset pipeline. "
            "Set ALLOW_SYNTHETIC_DATA=true only for demo runs."
        )

    def _resolve_checkpoint_path(self, model_id: str, path: str | None = None) -> Path:
        """Resolve checkpoint paths inside the configured model cache."""
        return self.config.resolve_checkpoint_path(model_id, path, suffix=".keras")

    def _ensure_initialized(self) -> None:
        """Ensure TensorFlow is imported."""
        if not self._initialized:
            tf = _get_tf()
            self._initialized = True
            logger.info("tensorflow_init", version=tf.__version__)

    def _create_architecture(
        self, architecture: str, num_classes: int, input_shape: list[int]
    ) -> Any:
        """Create a model architecture using Keras."""
        tf = _get_tf()

        if architecture == "mlp":
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Flatten(input_shape=input_shape),
                    tf.keras.layers.Dense(256, activation="relu"),
                    tf.keras.layers.Dense(256, activation="relu"),
                    tf.keras.layers.Dense(num_classes),
                ]
            )
            return model

        if architecture == "resnet18":
            # Simplified ResNet-like model
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same")(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Residual blocks
            for filters in [64, 128, 256, 512]:
                for _ in range(2):
                    shortcut = x
                    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.ReLU()(x)
                    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Add()([x, shortcut])
                    x = tf.keras.layers.ReLU()(x)

            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(num_classes)(x)

            model = tf.keras.Model(inputs, outputs)
            return model

        if architecture == "resnet50":
            # Larger ResNet variant
            return self._create_resnet50(num_classes, input_shape)

        # Default simple CNN
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, 3, activation="relu", input_shape=input_shape
                ),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(num_classes),
            ]
        )
        return model

    def _create_resnet50(self, num_classes: int, input_shape: list[int]) -> Any:
        """Create ResNet50 model."""
        tf = _get_tf()

        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            classes=num_classes,
        )

        model = tf.keras.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(num_classes),
            ]
        )

        return model

    async def create_model(
        self,
        model_id: str,
        model_name: str,
        architecture: str = "mlp",
        num_classes: int = 10,
        input_shape: list[int] | None = None,
    ) -> dict[str, Any]:
        """Create a new TensorFlow/Keras model."""
        try:
            self._ensure_initialized()
            input_shape = input_shape or [224, 224, 3]

            model = self._create_architecture(architecture, num_classes, input_shape)
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            self.models[model_id] = model

            # Create context
            ctx = self.context_manager.create_context(
                model_id=model_id,
                model_name=model_name,
                backend=BackendType.TENSORFLOW,
                architecture=architecture,
                num_classes=num_classes,
                input_shape=input_shape,
            )
            ctx.update_state(ModelState.READY)
            self.context_manager.update_context(ctx)

            logger.info("model_created", model_id=model_id, architecture=architecture)

            return {
                "status": "success",
                "model_id": model_id,
                "architecture": architecture,
                "num_parameters": model.count_params(),
            }
        except Exception as e:
            logger.error("model_creation_failed", model_id=model_id, error=str(e))
            return {"status": "error", "message": str(e)}

    async def train_step(
        self,
        model_id: str,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> dict[str, Any]:
        """Execute one training step."""
        try:
            model = self.models.get(model_id)
            if model is None:
                return {"status": "error", "message": "Model not found"}

            self._ensure_synthetic_data_enabled("Training")

            ctx = self.context_manager.get_context(model_id)
            if ctx:
                ctx.metadata["data_mode"] = "synthetic"
                ctx.update_state(ModelState.TRAINING)
                self.context_manager.update_context(ctx)

            # Create dummy training data
            import numpy as np

            x_train = np.random.randn(batch_size, *model.input_shape[1:]).astype(
                np.float32
            )
            y_train = np.random.randint(
                0, ctx.config.get("num_classes", 10) if ctx else 10, batch_size
            )

            # Train for one step
            history = model.train_on_batch(x_train, y_train)

            # Update metrics
            if ctx:
                metric = TrainingMetrics(
                    epoch=ctx.epochs_completed,
                    step=ctx.current_step + 1,
                    loss=float(history[0])
                    if isinstance(history, (list, tuple))
                    else float(history),
                    accuracy=float(history[1])
                    if isinstance(history, (list, tuple)) and len(history) > 1
                    else None,
                    learning_rate=learning_rate,
                )
                ctx.add_metric(metric)
                ctx.update_state(ModelState.READY)
                self.context_manager.update_context(ctx)

            return {
                "status": "success",
                "model_id": model_id,
                "step": ctx.current_step if ctx else 1,
                "loss": float(history[0])
                if isinstance(history, (list, tuple))
                else float(history),
                "accuracy": float(history[1])
                if isinstance(history, (list, tuple)) and len(history) > 1
                else None,
                "data_mode": "synthetic",
            }
        except Exception as e:
            logger.error("train_step_failed", model_id=model_id, error=str(e))
            return {"status": "error", "message": str(e)}

    async def evaluate(self, model_id: str) -> dict[str, Any]:
        """Evaluate model on dataset."""
        try:
            model = self.models.get(model_id)
            if model is None:
                return {"status": "error", "message": "Model not found"}

            self._ensure_synthetic_data_enabled("Evaluation")

            ctx = self.context_manager.get_context(model_id)
            if ctx:
                ctx.metadata["data_mode"] = "synthetic"
                ctx.update_state(ModelState.EVALUATING)
                self.context_manager.update_context(ctx)

            # Create dummy evaluation data
            import numpy as np

            x_eval = np.random.randn(10, *model.input_shape[1:]).astype(np.float32)
            y_eval = np.random.randint(
                0, ctx.config.get("num_classes", 10) if ctx else 10, 10
            )

            # Evaluate
            results = model.evaluate(x_eval, y_eval, verbose=0)

            if ctx:
                ctx.update_state(ModelState.READY)
                self.context_manager.update_context(ctx)

            return {
                "status": "success",
                "model_id": model_id,
                "eval_loss": float(results[0])
                if isinstance(results, (list, tuple))
                else float(results),
                "eval_accuracy": float(results[1])
                if isinstance(results, (list, tuple)) and len(results) > 1
                else None,
                "num_samples": 10,
                "data_mode": "synthetic",
            }
        except Exception as e:
            logger.error("evaluate_failed", model_id=model_id, error=str(e))
            return {"status": "error", "message": str(e)}

    async def save_checkpoint(
        self,
        model_id: str,
        path: str | None = None,
    ) -> dict[str, Any]:
        """Save model checkpoint."""
        try:
            model = self.models.get(model_id)
            if model is None:
                return {"status": "error", "message": "Model not found"}

            save_path = self._resolve_checkpoint_path(model_id, path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            save_str = str(save_path)
            model.save(save_str)

            ctx = self.context_manager.get_context(model_id)
            if ctx:
                ctx.checkpoint_path = save_path
                ctx.update_state(ModelState.CHECKPOINTED)
                self.context_manager.update_context(ctx)

            logger.info("checkpoint_saved", model_id=model_id, path=save_str)

            return {"status": "success", "path": save_str}
        except Exception as e:
            logger.error("save_checkpoint_failed", model_id=model_id, error=str(e))
            return {"status": "error", "message": str(e)}

    async def load_checkpoint(
        self,
        model_id: str,
        path: str,
    ) -> dict[str, Any]:
        """Load model checkpoint."""
        try:
            checkpoint_path = self._resolve_checkpoint_path(model_id, path)
            if not checkpoint_path.exists():
                return {"status": "error", "message": "Checkpoint not found"}

            # Load model from SavedModel
            loaded_model = _get_tf().keras.models.load_model(str(checkpoint_path))
            self.models[model_id] = loaded_model

            ctx = self.context_manager.get_context(model_id)
            if ctx:
                ctx.checkpoint_path = checkpoint_path
                ctx.update_state(ModelState.READY)
                self.context_manager.update_context(ctx)

            logger.info(
                "checkpoint_loaded", model_id=model_id, path=str(checkpoint_path)
            )

            return {"status": "success", "path": str(checkpoint_path)}
        except Exception as e:
            logger.error("load_checkpoint_failed", model_id=model_id, error=str(e))
            return {"status": "error", "message": str(e)}
