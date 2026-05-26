from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from ghost.context import BackendType, ContextManager
from ghost.data_loading import DatasetBatchProvider, LoadedDataset
from ghost.datasets import DatasetSpec


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