"""Model Context Management for Ghost.

Tracks model state, training history, and provides context for MCP interactions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelState(Enum):
    """Current state of a model."""

    INITIALIZED = "initialized"
    TRAINING = "training"
    EVALUATING = "evaluating"
    CHECKPOINTED = "checkpointed"
    FAILED = "failed"
    READY = "ready"


class BackendType(Enum):
    """ML framework backend."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


@dataclass
class TrainingMetrics:
    """Training metrics snapshot."""

    epoch: int
    step: int
    loss: float
    accuracy: float | None = None
    learning_rate: float | None = None
    timestamp: str = field(default_factory=_utc_now_iso)


@dataclass
class ModelContext:
    """Context information for a model."""

    model_id: str
    model_name: str
    backend: BackendType
    state: ModelState = ModelState.INITIALIZED
    epochs_completed: int = 0
    total_epochs: int = 10
    current_step: int = 0
    metrics: list[TrainingMetrics] = field(default_factory=list)
    checkpoint_path: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def update_state(self, state: ModelState) -> None:
        """Update model state."""
        self.state = state
        self.updated_at = _utc_now_iso()

    def add_metric(self, metric: TrainingMetrics) -> None:
        """Add a training metric."""
        self.metrics.append(metric)
        self.current_step = metric.step
        self.updated_at = _utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["backend"] = self.backend.value
        data["state"] = self.state.value
        data["checkpoint_path"] = (
            str(self.checkpoint_path) if self.checkpoint_path else None
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelContext:
        """Create from dictionary."""
        payload = dict(data)

        if payload.get("checkpoint_path"):
            payload["checkpoint_path"] = Path(payload["checkpoint_path"])

        backend = payload.get("backend")
        if isinstance(backend, str):
            payload["backend"] = BackendType(backend)

        state = payload.get("state")
        if isinstance(state, str):
            payload["state"] = ModelState(state)

        metrics = payload.get("metrics", [])
        payload["metrics"] = [
            metric if isinstance(metric, TrainingMetrics) else TrainingMetrics(**metric)
            for metric in metrics
        ]

        return cls(**payload)


class ContextManager:
    """Manages model contexts across the platform."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize context manager."""
        self.storage_path = storage_path or Path("./data/contexts")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._contexts: dict[str, ModelContext] = {}
        self._runtime_buckets: dict[str, dict[str, Any]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing contexts from storage."""
        if not self.storage_path.exists():
            return

        for ctx_file in self.storage_path.glob("*.json"):
            try:
                data = json.loads(ctx_file.read_text())
                ctx = ModelContext.from_dict(data)
                self._contexts[ctx.model_id] = ctx
            except Exception as exc:
                # Log corrupt/incompatible context files so they're visible
                # in the log stream rather than silently disappearing.
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "Skipping corrupt context file %s: %s", ctx_file.name, exc
                )
                continue

    def create_context(
        self,
        model_id: str,
        model_name: str,
        backend: BackendType | str,
        **config: Any,
    ) -> ModelContext:
        """Create a new model context."""
        if isinstance(backend, str):
            backend = BackendType(backend)

        ctx = ModelContext(
            model_id=model_id,
            model_name=model_name,
            backend=backend,
            config=config,
        )
        self._contexts[model_id] = ctx
        self._save_context(ctx)
        return ctx

    def get_context(self, model_id: str) -> ModelContext | None:
        """Get a model context by ID."""
        return self._contexts.get(model_id)

    def update_context(self, ctx: ModelContext) -> None:
        """Update a model context and persist."""
        ctx.updated_at = _utc_now_iso()
        self._contexts[ctx.model_id] = ctx
        self._save_context(ctx)

    def list_contexts(self) -> list[ModelContext]:
        """List all model contexts."""
        return list(self._contexts.values())

    def get_runtime_bucket(self, name: str) -> dict[str, Any]:
        """Return an in-memory runtime bucket shared across collaborating components."""
        return self._runtime_buckets.setdefault(name, {})

    def _save_context(self, ctx: ModelContext) -> None:
        """Save context to disk."""
        ctx_file = self.storage_path / f"{ctx.model_id}.json"
        temp_file = ctx_file.with_name(f".{ctx_file.name}.{uuid4().hex}.tmp")

        try:
            temp_file.write_text(json.dumps(ctx.to_dict(), indent=2), encoding="utf-8")
            temp_file.replace(ctx_file)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def delete_context(self, model_id: str) -> bool:
        """Delete a model context."""
        if model_id in self._contexts:
            del self._contexts[model_id]
            ctx_file = self.storage_path / f"{model_id}.json"
            if ctx_file.exists():
                ctx_file.unlink()
            return True
        return False

    def get_training_history(self, model_id: str) -> list[TrainingMetrics]:
        """Get training history for a model."""
        ctx = self.get_context(model_id)
        return ctx.metrics if ctx else []
