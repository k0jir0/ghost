"""Unit tests for ghost.datasets."""

from __future__ import annotations

from pathlib import Path

import pytest

from ghost.config import GhostConfig, reset_config
from ghost.datasets import DatasetResolver


@pytest.fixture()
def dataset_resolver(tmp_path: Path) -> DatasetResolver:
    reset_config()
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    return DatasetResolver(config=config)


class TestDatasetResolver:
    def test_resolve_accepts_aliases(self, dataset_resolver: DatasetResolver) -> None:
        spec = dataset_resolver.resolve("cifar10", allow_synthetic=False)
        assert spec.dataset_id == "cifar-10"
        assert spec.synthetic is False

    def test_resolve_rejects_synthetic_without_explicit_opt_in(
        self,
        dataset_resolver: DatasetResolver,
    ) -> None:
        with pytest.raises(ValueError, match="allow_synthetic=True"):
            dataset_resolver.resolve("synthetic-image", allow_synthetic=False)

    def test_resolve_allows_synthetic_when_opted_in(
        self,
        dataset_resolver: DatasetResolver,
    ) -> None:
        spec = dataset_resolver.resolve("synthetic-image", allow_synthetic=True)
        assert spec.dataset_id == "synthetic-image-classification"
        assert spec.synthetic is True

    def test_list_available_can_filter_synthetic(
        self,
        dataset_resolver: DatasetResolver,
    ) -> None:
        public_specs = dataset_resolver.list_available(include_synthetic=False)
        assert public_specs
        assert all(spec.synthetic is False for spec in public_specs)

    def test_unknown_dataset_raises_key_error(
        self,
        dataset_resolver: DatasetResolver,
    ) -> None:
        with pytest.raises(KeyError, match="Unknown dataset reference"):
            dataset_resolver.resolve("unknown-dataset", allow_synthetic=False)