from __future__ import annotations

from pathlib import Path

from ghost.config import GhostConfig
from ghost.dataset_registry import DatasetRegistry
from ghost.datasets import DatasetSpec


def test_upsert_manifest_persists_dataset_metadata(tmp_path: Path) -> None:
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    registry = DatasetRegistry(config=config)
    spec = DatasetSpec(
        dataset_id="mnist",
        task_type="image-classification",
        source="builtin-catalog",
        input_shape=(1, 28, 28),
        num_classes=10,
        synthetic=False,
    )

    manifest = registry.upsert_manifest(
        spec,
        validation_status="passed",
        metadata={"validation_report_id": "mnist@builtin-v1"},
    )
    restored = registry.get_manifest("mnist", manifest.version)

    assert restored is not None
    assert restored.dataset_id == "mnist"
    assert restored.validation_status == "passed"
    assert restored.metadata["validation_report_id"] == "mnist@builtin-v1"


def test_upsert_manifest_preserves_external_source_uri(tmp_path: Path) -> None:
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    registry = DatasetRegistry(config=config)
    spec = DatasetSpec(
        dataset_id="custom-images",
        task_type="image-classification",
        source="filesystem",
        input_shape=(1, 28, 28),
        num_classes=2,
        synthetic=False,
        metadata={
            "source_uri": "file:///tmp/custom-images.npz",
        },
    )

    manifest = registry.upsert_manifest(spec, validation_status="passed")

    assert manifest.source_uri == "file:///tmp/custom-images.npz"
    assert manifest.version.startswith("uri-")