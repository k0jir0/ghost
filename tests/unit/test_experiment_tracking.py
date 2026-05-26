"""Unit tests for ghost.experiment_tracking."""

from __future__ import annotations

from pathlib import Path

from ghost.config import GhostConfig, reset_config
from ghost.context import BackendType, ContextManager
from ghost.datasets import DatasetSpec
from ghost.experiment_tracking import ExperimentTracker
from ghost.metadata_store import MetadataStore
from ghost.orchestration import TrainingRunRecord, TrainingRunRequest
from ghost.planning import TrainingPlan
from ghost.run_store import RunStore
from ghost.training import TrainingResult


def test_experiment_tracker_records_lineage_for_checkpointed_run(tmp_path: Path) -> None:
    reset_config()
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    metadata_store = MetadataStore(config.data_cache_dir / "metadata")
    run_store = RunStore(config=config, metadata_store=metadata_store)
    tracker = ExperimentTracker(run_store=run_store)
    checkpoint_path = config.model_cache_dir / "model-1.pt"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    context_manager = ContextManager(storage_path=tmp_path / "contexts")
    ctx = context_manager.create_context(
        model_id="model-1",
        model_name="Net",
        backend=BackendType.PYTORCH,
        architecture="mlp",
        num_classes=10,
        input_shape=[1, 28, 28],
    )
    ctx.checkpoint_path = checkpoint_path
    ctx.metadata["dataset_spec"] = {
        "metadata": {"dataset_version": "builtin-v2"},
    }
    context_manager.update_context(ctx)

    training_record = TrainingRunRecord(
        run_id="run-1",
        model_id="model-1",
        status="completed",
        plan=TrainingPlan(
            task="Train MLP",
            backend=BackendType.PYTORCH,
            architecture="mlp",
            num_classes=10,
            batch_size=32,
            learning_rate=0.001,
            epochs=3,
            dataset="mnist",
        ),
        analysis={"analysis": "stable"},
        events=[],
        request=TrainingRunRequest(task="Train MLP", model_id="model-1"),
        dataset=DatasetSpec(
            dataset_id="mnist",
            task_type="image-classification",
            source="builtin-catalog",
            input_shape=(1, 28, 28),
            num_classes=10,
            synthetic=False,
            metadata={"dataset_version": "builtin-v2"},
        ),
        result=TrainingResult(
            model_id="model-1",
            success=True,
            final_loss=0.12,
            final_accuracy=0.94,
            epochs_completed=3,
            checkpoint_path=checkpoint_path,
        ),
    )

    experiment = tracker.record_training_run(training_record, context=ctx)
    artifact = run_store.get_checkpoint_artifact_for_run("run-1")

    assert experiment.dataset_version == "builtin-v2"
    assert experiment.metrics["final_accuracy"] == 0.94
    assert artifact is not None
    assert artifact.uri == str(checkpoint_path)
    assert artifact.metadata["dataset_version"] == "builtin-v2"