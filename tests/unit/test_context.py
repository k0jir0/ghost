"""Unit tests for ghost.context — ModelContext, ContextManager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghost.context import (
    BackendType,
    ContextManager,
    ModelContext,
    ModelState,
    TrainingMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(model_id: str = "m1", backend: BackendType = BackendType.PYTORCH) -> ModelContext:
    return ModelContext(
        model_id=model_id,
        model_name="Test Model",
        backend=backend,
    )


# ---------------------------------------------------------------------------
# ModelContext
# ---------------------------------------------------------------------------

class TestModelContext:
    def test_default_state_is_initialized(self) -> None:
        ctx = _make_ctx()
        assert ctx.state == ModelState.INITIALIZED

    def test_update_state_changes_state_and_timestamp(self) -> None:
        ctx = _make_ctx()
        old_ts = ctx.updated_at
        ctx.update_state(ModelState.TRAINING)
        assert ctx.state == ModelState.TRAINING
        assert ctx.updated_at >= old_ts

    def test_add_metric_appends_and_updates_step(self) -> None:
        ctx = _make_ctx()
        metric = TrainingMetrics(epoch=1, step=1, loss=0.5)
        ctx.add_metric(metric)
        assert len(ctx.metrics) == 1
        assert ctx.current_step == 1

    def test_add_multiple_metrics(self) -> None:
        ctx = _make_ctx()
        for i in range(5):
            ctx.add_metric(TrainingMetrics(epoch=1, step=i + 1, loss=float(i)))
        assert len(ctx.metrics) == 5
        assert ctx.current_step == 5

    def test_to_dict_round_trips(self) -> None:
        ctx = _make_ctx()
        d = ctx.to_dict()
        assert d["model_id"] == "m1"
        assert d["backend"] == "pytorch"
        assert d["state"] == "initialized"

    def test_from_dict_restores_context(self) -> None:
        ctx = _make_ctx("restore_me")
        d = ctx.to_dict()
        restored = ModelContext.from_dict(d)
        assert restored.model_id == "restore_me"
        assert restored.state == ModelState.INITIALIZED

    def test_checkpoint_path_serialization(self, tmp_path: Path) -> None:
        ctx = _make_ctx()
        ctx.checkpoint_path = tmp_path / "model.pt"
        d = ctx.to_dict()
        assert isinstance(d["checkpoint_path"], str)
        restored = ModelContext.from_dict(d)
        assert isinstance(restored.checkpoint_path, Path)

    def test_null_checkpoint_path(self) -> None:
        ctx = _make_ctx()
        d = ctx.to_dict()
        assert d["checkpoint_path"] is None
        restored = ModelContext.from_dict(d)
        assert restored.checkpoint_path is None


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_create_and_get(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        ctx = cm.create_context("id1", "MyModel", BackendType.PYTORCH)
        assert ctx.model_id == "id1"
        fetched = cm.get_context("id1")
        assert fetched is not None
        assert fetched.model_id == "id1"

    def test_create_persists_to_disk(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        cm.create_context("disk_test", "DiskModel", "pytorch")
        assert (tmp_data_dir / "disk_test.json").exists()

    def test_get_missing_returns_none(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        assert cm.get_context("nonexistent") is None

    def test_update_persists(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        ctx = cm.create_context("upd", "Upd", BackendType.PYTORCH)
        ctx.update_state(ModelState.TRAINING)
        cm.update_context(ctx)
        # Reload from disk
        raw = json.loads((tmp_data_dir / "upd.json").read_text())
        assert raw["state"] == "training"

    def test_list_contexts_returns_all(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        cm.create_context("a", "A", BackendType.PYTORCH)
        cm.create_context("b", "B", BackendType.TENSORFLOW)
        lst = cm.list_contexts()
        ids = {c.model_id for c in lst}
        assert ids == {"a", "b"}

    def test_delete_removes_from_memory_and_disk(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        cm.create_context("del_me", "Del", BackendType.PYTORCH)
        result = cm.delete_context("del_me")
        assert result is True
        assert cm.get_context("del_me") is None
        assert not (tmp_data_dir / "del_me.json").exists()

    def test_delete_missing_returns_false(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        assert cm.delete_context("ghost") is False

    def test_load_existing_on_init(self, tmp_data_dir: Path) -> None:
        # Persist one context manually
        ctx = _make_ctx("pre_existing")
        (tmp_data_dir / "pre_existing.json").write_text(
            json.dumps(ctx.to_dict())
        )
        # New CM should load it automatically
        cm = ContextManager(storage_path=tmp_data_dir)
        assert cm.get_context("pre_existing") is not None

    def test_corrupt_context_file_is_skipped(self, tmp_data_dir: Path) -> None:
        (tmp_data_dir / "bad.json").write_text("not valid json{{")
        # Should not raise
        cm = ContextManager(storage_path=tmp_data_dir)
        assert cm.get_context("bad") is None

    def test_get_training_history_returns_metrics(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        ctx = cm.create_context("hist", "H", BackendType.PYTORCH)
        ctx.add_metric(TrainingMetrics(epoch=1, step=1, loss=0.9))
        cm.update_context(ctx)
        history = cm.get_training_history("hist")
        assert len(history) == 1
        assert history[0].loss == pytest.approx(0.9)

    def test_get_training_history_missing_model(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        assert cm.get_training_history("nope") == []

    def test_backend_string_is_coerced(self, tmp_data_dir: Path) -> None:
        cm = ContextManager(storage_path=tmp_data_dir)
        ctx = cm.create_context("str_backend", "SB", "tensorflow")
        assert ctx.backend == BackendType.TENSORFLOW
