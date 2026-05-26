"""Unit tests for ghost.run_store."""

from __future__ import annotations

from pathlib import Path

import pytest

from ghost.config import GhostConfig, reset_config
from ghost.metadata_store import MetadataStore
from ghost.run_store import RunStore
from ghost.schemas import ArtifactRecord, ExperimentRunRecord


def test_run_store_persists_and_compares_runs(tmp_path: Path) -> None:
    reset_config()
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    store = RunStore(
        config=config,
        metadata_store=MetadataStore(config.data_cache_dir / "metadata"),
    )

    first = ExperimentRunRecord(
        run_id="run-1",
        experiment_id="exp-1",
        model_id="model-1",
        status="completed",
        backend="pytorch",
        architecture="mlp",
        dataset_id="mnist",
        dataset_version="builtin-v1",
        metrics={"final_accuracy": 0.8, "final_loss": 0.4},
    )
    second = ExperimentRunRecord(
        run_id="run-2",
        experiment_id="exp-1",
        model_id="model-2",
        status="completed",
        backend="pytorch",
        architecture="mlp",
        dataset_id="mnist",
        dataset_version="builtin-v1",
        metrics={"final_accuracy": 0.9, "final_loss": 0.2},
    )

    store.upsert_run(first)
    store.upsert_run(second)

    comparison = store.compare_runs(["run-1", "run-2"])

    assert comparison["count"] == 2
    assert comparison["summary"]["best_accuracy_run_id"] == "run-2"
    assert comparison["summary"]["lowest_loss_run_id"] == "run-2"
    assert comparison["deltas"][0]["final_accuracy_delta"] == pytest.approx(0.1)


def test_run_store_filters_artifacts(tmp_path: Path) -> None:
    reset_config()
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    store = RunStore(
        config=config,
        metadata_store=MetadataStore(config.data_cache_dir / "metadata"),
    )

    checkpoint = ArtifactRecord(
        artifact_id="run-1__checkpoint",
        artifact_type="checkpoint",
        uri=str(tmp_path / "models" / "m1.pt"),
        run_id="run-1",
        model_id="model-1",
    )
    store.upsert_artifact(checkpoint)

    artifacts = store.list_artifacts(run_id="run-1", artifact_type="checkpoint")

    assert len(artifacts) == 1
    assert artifacts[0].artifact_id == "run-1__checkpoint"