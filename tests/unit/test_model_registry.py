"""Unit tests for ghost.model_registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from ghost.config import GhostConfig, reset_config
from ghost.evaluation import EvaluationPolicy
from ghost.metadata_store import MetadataStore
from ghost.model_registry import ModelRegistry
from ghost.run_store import RunStore
from ghost.schemas import ArtifactRecord, ExperimentRunRecord


def _make_registry(tmp_path: Path) -> tuple[ModelRegistry, RunStore]:
    reset_config()
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    metadata_store = MetadataStore(config.data_cache_dir / "metadata")
    run_store = RunStore(config=config, metadata_store=metadata_store)
    registry = ModelRegistry(
        config=config,
        metadata_store=metadata_store,
        run_store=run_store,
    )
    return registry, run_store


def test_register_model_creates_draft_candidate(tmp_path: Path) -> None:
    registry, run_store = _make_registry(tmp_path)
    run_store.upsert_run(
        ExperimentRunRecord(
            run_id="run-1",
            experiment_id="exp-1",
            model_id="model-1",
            status="completed",
            backend="pytorch",
            architecture="mlp",
            dataset_id="mnist",
            dataset_version="builtin-v1",
            input_shape=[1, 28, 28],
            num_classes=10,
            metrics={"final_accuracy": 0.9, "final_loss": 0.2},
            code_version="abc123",
        )
    )
    run_store.upsert_artifact(
        ArtifactRecord(
            artifact_id="run-1__checkpoint",
            artifact_type="checkpoint",
            uri=str(tmp_path / "models" / "model-1.pt"),
            run_id="run-1",
            model_id="model-1",
        )
    )

    record = registry.register_model(
        "run-1",
        policy=EvaluationPolicy(min_accuracy=0.8, max_loss=0.3),
    )

    assert record.stage == "draft"
    assert record.evaluation_status == "passed"
    assert record.metadata["eligible_for_promotion"] is True


def test_promote_model_requires_passing_evaluation(tmp_path: Path) -> None:
    registry, run_store = _make_registry(tmp_path)
    run_store.upsert_run(
        ExperimentRunRecord(
            run_id="run-2",
            experiment_id="exp-2",
            model_id="model-2",
            status="completed",
            backend="pytorch",
            architecture="mlp",
            metrics={"final_accuracy": 0.5, "final_loss": 1.2},
        )
    )
    run_store.upsert_artifact(
        ArtifactRecord(
            artifact_id="run-2__checkpoint",
            artifact_type="checkpoint",
            uri=str(tmp_path / "models" / "model-2.pt"),
            run_id="run-2",
            model_id="model-2",
        )
    )
    record = registry.register_model(
        "run-2",
        policy=EvaluationPolicy(min_accuracy=0.8, max_loss=0.3),
    )

    with pytest.raises(ValueError):
        registry.promote_model(record.registry_id, stage="production")


def test_promote_model_archives_previous_production_version(tmp_path: Path) -> None:
    registry, run_store = _make_registry(tmp_path)
    for run_id, accuracy in (("run-1", 0.9), ("run-2", 0.95)):
        run_store.upsert_run(
            ExperimentRunRecord(
                run_id=run_id,
                experiment_id="exp-1",
                model_id="model-1",
                status="completed",
                backend="pytorch",
                architecture="mlp",
                metrics={"final_accuracy": accuracy, "final_loss": 0.2},
            )
        )
        run_store.upsert_artifact(
            ArtifactRecord(
                artifact_id=f"{run_id}__checkpoint",
                artifact_type="checkpoint",
                uri=str(tmp_path / "models" / f"{run_id}.pt"),
                run_id=run_id,
                model_id="model-1",
            )
        )

    first = registry.register_model("run-1", policy=EvaluationPolicy(min_accuracy=0.8))
    second = registry.register_model("run-2", policy=EvaluationPolicy(min_accuracy=0.8))
    registry.promote_model(first.registry_id, stage="production")
    promoted = registry.promote_model(second.registry_id, stage="production")
    archived = registry.get_model(first.registry_id)

    assert promoted.stage == "production"
    assert "current-production" in promoted.aliases
    assert archived is not None
    assert archived.stage == "archived"


def test_register_model_rejects_non_completed_run(tmp_path: Path) -> None:
    registry, run_store = _make_registry(tmp_path)
    run_store.upsert_run(
        ExperimentRunRecord(
            run_id="run-failed",
            experiment_id="exp-failed",
            model_id="model-failed",
            status="failed",
            backend="pytorch",
            architecture="mlp",
            metrics={"final_loss": 1.0},
        )
    )
    run_store.upsert_artifact(
        ArtifactRecord(
            artifact_id="run-failed__checkpoint",
            artifact_type="checkpoint",
            uri=str(tmp_path / "models" / "model-failed.pt"),
            run_id="run-failed",
            model_id="model-failed",
        )
    )

    with pytest.raises(ValueError, match="completed runs"):
        registry.register_model("run-failed")


def test_register_model_default_policy_fails_closed_without_metrics(
    tmp_path: Path,
) -> None:
    registry, run_store = _make_registry(tmp_path)
    run_store.upsert_run(
        ExperimentRunRecord(
            run_id="run-no-metrics",
            experiment_id="exp-no-metrics",
            model_id="model-no-metrics",
            status="completed",
            backend="pytorch",
            architecture="mlp",
            metrics={},
        )
    )
    run_store.upsert_artifact(
        ArtifactRecord(
            artifact_id="run-no-metrics__checkpoint",
            artifact_type="checkpoint",
            uri=str(tmp_path / "models" / "model-no-metrics.pt"),
            run_id="run-no-metrics",
            model_id="model-no-metrics",
        )
    )

    record = registry.register_model("run-no-metrics")

    assert record.stage == "draft"
    assert record.evaluation_status == "failed"
    assert record.metadata["eligible_for_promotion"] is False
