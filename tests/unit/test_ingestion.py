from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ghost.config import GhostConfig
from ghost.datasets import DatasetSpec
from ghost.ingestion import DatasetIngestionService, ObjectStoreDatasetIngestor


def _write_dataset_bundle(path: Path, *, offset: float = 0.0) -> None:
    np.savez(
        path,
        train_features=np.full((2, 2, 2, 1), offset + 1.0, dtype=np.float32),
        train_labels=np.array([0, 1], dtype=np.int64),
        eval_features=np.full((1, 2, 2, 1), offset + 2.0, dtype=np.float32),
        eval_labels=np.array([1], dtype=np.int64),
    )


def test_filesystem_ingestion_returns_local_artifact(tmp_path: Path) -> None:
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    bundle_path = tmp_path / "external-dataset.npz"
    _write_dataset_bundle(bundle_path)
    spec = DatasetSpec(
        dataset_id="custom-images",
        task_type="image-classification",
        source="filesystem",
        input_shape=(1, 2, 2),
        num_classes=2,
        synthetic=False,
        metadata={
            "source_uri": bundle_path.resolve().as_uri(),
            "dataset_version": "file-v1",
        },
    )

    artifact = DatasetIngestionService(config=config).ingest(spec)

    assert artifact.local_path == bundle_path.resolve()


def test_object_store_ingestion_fetches_and_caches_artifact(tmp_path: Path) -> None:
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    source_bundle = tmp_path / "object-store-bundle.npz"
    _write_dataset_bundle(source_bundle)
    calls: list[str] = []

    def fetcher(source_uri: str, destination: Path) -> Path:
        calls.append(source_uri)
        shutil.copyfile(source_bundle, destination)
        return destination

    spec = DatasetSpec(
        dataset_id="custom-images",
        task_type="image-classification",
        source="object-store",
        input_shape=(1, 2, 2),
        num_classes=2,
        synthetic=False,
        metadata={
            "source_uri": "s3://ghost-bucket/custom-images.npz",
            "dataset_version": "s3-v1",
        },
    )
    service = DatasetIngestionService(
        config=config,
        ingestors=[ObjectStoreDatasetIngestor(config=config, object_fetcher=fetcher)],
    )

    first = service.ingest(spec)
    second = service.ingest(spec)

    assert first.local_path.exists()
    assert second.local_path == first.local_path
    assert calls == ["s3://ghost-bucket/custom-images.npz"]