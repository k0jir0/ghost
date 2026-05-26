from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ghost.config import GhostConfig
from ghost.context import BackendType, ContextManager
from ghost.data_loading import DatasetBatchProvider, LoadedDataset, RealDatasetLoader
from ghost.datasets import DatasetSpec
from ghost.metadata_store import MetadataStore


class FakeLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def load(self, spec: DatasetSpec) -> LoadedDataset:
        self.calls.append(spec.dataset_id)
        features = np.arange(48, dtype=np.float32).reshape(6, 2, 2, 2)
        labels = np.arange(6, dtype=np.int64)
        return LoadedDataset(
            train_features=features,
            train_labels=labels,
            eval_features=features[:4],
            eval_labels=labels[:4],
        )


def test_batch_provider_uses_context_dataset_metadata(tmp_path) -> None:
    context_manager = ContextManager(storage_path=tmp_path / "contexts")
    ctx = context_manager.create_context("m1", "Model 1", BackendType.PYTORCH)
    spec = DatasetSpec(
        dataset_id="mnist",
        task_type="image-classification",
        source="builtin-catalog",
        input_shape=(1, 28, 28),
        num_classes=10,
        synthetic=False,
    )
    ctx.metadata["dataset_spec"] = asdict(spec)
    context_manager.update_context(ctx)

    loader = FakeLoader()
    provider = DatasetBatchProvider(context_manager, loader=loader)

    features, labels, resolved = provider.next_training_batch("m1", batch_size=4)

    assert resolved.dataset_id == "mnist"
    assert features.shape[0] == 4
    assert labels.shape == (4,)
    assert loader.calls == ["mnist"]


def test_batch_provider_requires_dataset_metadata(tmp_path) -> None:
    context_manager = ContextManager(storage_path=tmp_path / "contexts")
    context_manager.create_context("m1", "Model 1", BackendType.PYTORCH)
    provider = DatasetBatchProvider(context_manager, loader=FakeLoader())

    with pytest.raises(RuntimeError, match="resolved dataset"):
        provider.next_training_batch("m1", batch_size=2)


def test_real_loader_persists_dataset_manifest_and_validation_report(tmp_path, monkeypatch) -> None:
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    loader = RealDatasetLoader(config=config)
    dataset = LoadedDataset(
        train_features=np.zeros((4, 28, 28, 1), dtype=np.float32),
        train_labels=np.array([0, 1, 2, 3], dtype=np.int64),
        eval_features=np.zeros((2, 28, 28, 1), dtype=np.float32),
        eval_labels=np.array([0, 1], dtype=np.int64),
    )
    spec = DatasetSpec(
        dataset_id="mnist",
        task_type="image-classification",
        source="builtin-catalog",
        input_shape=(1, 28, 28),
        num_classes=10,
        synthetic=False,
    )
    monkeypatch.setattr(loader, "_load_mnist", lambda: dataset)

    loaded = loader.load(spec)
    store = MetadataStore(config.data_cache_dir / "metadata")
    manifests = store.list_records("dataset-manifests")
    reports = store.list_records("dataset-validation-reports")

    assert loaded is dataset
    assert manifests[0]["dataset_id"] == "mnist"
    assert manifests[0]["validation_status"] == "passed"
    assert reports[0]["dataset_id"] == "mnist"


def test_real_loader_ingests_file_backed_dataset_and_keeps_versions_separate(
    tmp_path: Path,
) -> None:
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    bundle_v1 = tmp_path / "dataset-v1.npz"
    bundle_v2 = tmp_path / "dataset-v2.npz"
    np.savez(
        bundle_v1,
        train_features=np.full((2, 2, 2, 1), 1.0, dtype=np.float32),
        train_labels=np.array([0, 1], dtype=np.int64),
        eval_features=np.full((1, 2, 2, 1), 2.0, dtype=np.float32),
        eval_labels=np.array([1], dtype=np.int64),
    )
    np.savez(
        bundle_v2,
        train_features=np.full((2, 2, 2, 1), 9.0, dtype=np.float32),
        train_labels=np.array([1, 0], dtype=np.int64),
        eval_features=np.full((1, 2, 2, 1), 8.0, dtype=np.float32),
        eval_labels=np.array([0], dtype=np.int64),
    )
    loader = RealDatasetLoader(config=config)
    spec_v1 = DatasetSpec(
        dataset_id="custom-images",
        task_type="image-classification",
        source="filesystem",
        input_shape=(1, 2, 2),
        num_classes=2,
        synthetic=False,
        metadata={
            "source_uri": bundle_v1.resolve().as_uri(),
            "dataset_version": "file-v1",
        },
    )
    spec_v2 = DatasetSpec(
        dataset_id="custom-images",
        task_type="image-classification",
        source="filesystem",
        input_shape=(1, 2, 2),
        num_classes=2,
        synthetic=False,
        metadata={
            "source_uri": bundle_v2.resolve().as_uri(),
            "dataset_version": "file-v2",
        },
    )

    loaded_v1 = loader.load(spec_v1)
    loaded_v2 = loader.load(spec_v2)
    manifests = MetadataStore(config.data_cache_dir / "metadata").list_records(
        "dataset-manifests"
    )

    assert loaded_v1.train_features[0, 0, 0, 0] == pytest.approx(1.0)
    assert loaded_v2.train_features[0, 0, 0, 0] == pytest.approx(9.0)
    assert len(manifests) == 2
    assert {manifest["version"] for manifest in manifests} == {"file-v1", "file-v2"}