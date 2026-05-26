from __future__ import annotations

from pathlib import Path

import numpy as np

from ghost.config import GhostConfig
from ghost.data_loading import LoadedDataset
from ghost.data_validation import DatasetValidator
from ghost.datasets import DatasetSpec


def test_validate_loaded_dataset_persists_report(tmp_path: Path) -> None:
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    validator = DatasetValidator(config=config)
    spec = DatasetSpec(
        dataset_id="mnist",
        task_type="image-classification",
        source="builtin-catalog",
        input_shape=(1, 28, 28),
        num_classes=10,
        synthetic=False,
    )
    dataset = LoadedDataset(
        train_features=np.zeros((4, 28, 28, 1), dtype=np.float32),
        train_labels=np.array([0, 1, 2, 3], dtype=np.int64),
        eval_features=np.zeros((2, 28, 28, 1), dtype=np.float32),
        eval_labels=np.array([0, 1], dtype=np.int64),
    )

    report = validator.validate_loaded_dataset(spec, dataset)
    restored = validator.get_report("mnist", "builtin-v1")

    assert report.status == "passed"
    assert restored is not None
    assert restored.stats["train_samples"] == 4