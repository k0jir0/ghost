"""PyTorch operations for Ghost MCP tools.

Provides MCP tools for PyTorch model creation, training, and evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from ghost.config import get_config
from ghost.context import BackendType, ContextManager, ModelState, TrainingMetrics
from ghost.logging import get_logger

logger = get_logger(__name__)


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for testing."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ResNetSimple(nn.Module):
    """Simplified ResNet-like architecture."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(
        self, in_channels: int, out_channels: int, blocks: int
    ) -> nn.Sequential:
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(blocks - 1):
            layers.append(
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class PyTorchOps:
    """PyTorch operations handler for MCP tools."""

    def __init__(self, context_manager: ContextManager):
        """Initialize PyTorch operations."""
        self.context_manager = context_manager
        self.config = get_config()
        runtime = self.context_manager.get_runtime_bucket("pytorch")
        self.models: dict[str, nn.Module] = runtime.setdefault("models", {})
        self.optimizers: dict[str, optim.Optimizer] = runtime.setdefault(
            "optimizers", {}
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("pytorch_init", device=str(self._device))

    def _ensure_synthetic_data_enabled(self, operation: str) -> None:
        """Fail closed when the runtime would otherwise fabricate training data."""
        if self.config.allow_synthetic_data:
            return

        raise RuntimeError(
            f"{operation} requires a real dataset pipeline. "
            "Set ALLOW_SYNTHETIC_DATA=true only for demo runs."
        )

    def _resolve_checkpoint_path(self, model_id: str, path: str | None = None) -> Path:
        """Resolve checkpoint paths inside the configured model cache."""
        return self.config.resolve_checkpoint_path(model_id, path, suffix=".pt")

    def _create_architecture(
        self, architecture: str, num_classes: int, input_shape: list[int]
    ) -> nn.Module:
        """Create a model architecture."""
        if architecture == "mlp":
            input_size = 1
            for dim in input_shape:
                input_size *= dim
            return SimpleMLP(input_size, 256, num_classes)

        if architecture in ("resnet18", "resnet50"):
            return ResNetSimple(num_classes=num_classes)

        # Default MLP
        input_size = 1
        for dim in input_shape:
            input_size *= dim
        return SimpleMLP(input_size, 256, num_classes)

    async def create_model(
        self,
        model_id: str,
        model_name: str,
        architecture: str = "mlp",
        num_classes: int = 10,
        input_shape: list[int] | None = None,
    ) -> dict[str, Any]:
        """Create a new PyTorch model."""
        try:
            input_shape = input_shape or [3, 224, 224]
            model = self._create_architecture(architecture, num_classes, input_shape)
            model = model.to(self._device)

            self.models[model_id] = model
            self.optimizers[model_id] = optim.Adam(model.parameters(), lr=0.001)

            # Create context
            ctx = self.context_manager.create_context(
                model_id=model_id,
                model_name=model_name,
                backend=BackendType.PYTORCH,
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
                "num_parameters": sum(p.numel() for p in model.parameters()),
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

            # Simulate training step with dummy data
            model.train()

            # Create dummy batch
            if isinstance(model, ResNetSimple):
                dummy_input = torch.randn(batch_size, 3, 224, 224).to(self._device)
            else:
                dummy_input = torch.randn(batch_size, 3, 224, 224).to(self._device)

            dummy_target = torch.randint(
                0, ctx.config.get("num_classes", 10) if ctx else 10, (batch_size,)
            ).to(self._device)

            # Forward pass
            output = model(dummy_input)
            loss = nn.CrossEntropyLoss()(output, dummy_target)

            # Backward pass
            optimizer = self.optimizers.get(model_id) or optim.Adam(
                model.parameters(), lr=learning_rate
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            if ctx:
                metric = TrainingMetrics(
                    epoch=ctx.epochs_completed,
                    step=ctx.current_step + 1,
                    loss=loss.item(),
                    accuracy=None,
                    learning_rate=learning_rate,
                )
                ctx.add_metric(metric)
                ctx.update_state(ModelState.READY)
                self.context_manager.update_context(ctx)

            return {
                "status": "success",
                "model_id": model_id,
                "step": ctx.current_step if ctx else 1,
                "loss": loss.item(),
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

            model.eval()

            # Simulate evaluation
            with torch.no_grad():
                dummy_input = torch.randn(10, 3, 224, 224).to(self._device)
                output = model(dummy_input)
                loss = nn.CrossEntropyLoss()(
                    output, torch.randint(0, 10, (10,)).to(self._device)
                )

            if ctx:
                ctx.update_state(ModelState.READY)
                self.context_manager.update_context(ctx)

            return {
                "status": "success",
                "model_id": model_id,
                "eval_loss": loss.item(),
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

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_id": model_id,
                },
                save_path,
            )

            ctx = self.context_manager.get_context(model_id)
            if ctx:
                ctx.checkpoint_path = save_path
                ctx.update_state(ModelState.CHECKPOINTED)
                self.context_manager.update_context(ctx)

            logger.info("checkpoint_saved", model_id=model_id, path=str(save_path))

            return {"status": "success", "path": str(save_path)}
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

            # weights_only=True prevents arbitrary code execution from
            # malicious checkpoint files (CVE-style torch.load risk).
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self._device,
                weights_only=True,
            )

            if model_id not in self.models:
                # Need to recreate architecture first
                return {
                    "status": "error",
                    "message": "Model not found. Create model first.",
                }

            self.models[model_id].load_state_dict(checkpoint["model_state_dict"])

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
