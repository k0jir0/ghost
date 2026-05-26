"""Real dataset loading and batch provisioning for Ghost backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ghost.config import GhostConfig, get_config
from ghost.context import ContextManager
from ghost.data_validation import DatasetValidator
from ghost.dataset_registry import DatasetRegistry
from ghost.datasets import DatasetSpec
from ghost.ingestion import DatasetIngestionService
from ghost.logging import get_logger

logger = get_logger(__name__)


def _get_tensorflow():
    import tensorflow as tf

    return tf


@dataclass(frozen=True)
class LoadedDataset:
    """In-memory train/eval arrays for a resolved dataset."""

    train_features: np.ndarray
    train_labels: np.ndarray
    eval_features: np.ndarray
    eval_labels: np.ndarray


class RealDatasetLoader:
    """Load real datasets backed by the local data cache."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        dataset_registry: DatasetRegistry | None = None,
        dataset_validator: DatasetValidator | None = None,
        ingestion_service: DatasetIngestionService | None = None,
    ):
        self.config = config or get_config()
        self.dataset_registry = dataset_registry or DatasetRegistry(config=self.config)
        self.dataset_validator = dataset_validator or DatasetValidator(config=self.config)
        self.ingestion_service = ingestion_service or DatasetIngestionService(
            config=self.config
        )
        self._cache: dict[str, LoadedDataset] = {}

    def load(self, spec: DatasetSpec) -> LoadedDataset:
        """Return cached dataset arrays for a known non-synthetic dataset."""
        cache_key = self._cache_key(spec)
        cached = self._cache.get(cache_key)
        if spec.synthetic:
            raise ValueError(
                "Synthetic dataset specs do not require real dataset loading."
            )

        if cached is not None:
            self._record_dataset_metadata(spec, cached)
            return cached

        self._ensure_cache_home()
        source_uri = self.dataset_registry.source_uri_for_spec(spec)

        if source_uri.startswith("builtin://") and spec.dataset_id == "cifar-10":
            dataset = self._load_cifar10()
        elif source_uri.startswith("builtin://") and spec.dataset_id == "mnist":
            dataset = self._load_mnist()
        elif source_uri.startswith("builtin://") and spec.dataset_id == "imdb-reviews":
            dataset = self._load_imdb_reviews(spec)
        else:
            dataset = self._load_ingested_dataset(spec)

        self._cache[cache_key] = dataset
        self._record_dataset_metadata(spec, dataset)
        logger.info("dataset_loaded", dataset_id=spec.dataset_id)
        return dataset

    def _cache_key(self, spec: DatasetSpec) -> str:
        version = self.dataset_registry.version_for_spec(spec)
        return f"{spec.dataset_id}@{version}"

    def _record_dataset_metadata(
        self,
        spec: DatasetSpec,
        dataset: LoadedDataset,
    ) -> None:
        dataset_version = self.dataset_registry.version_for_spec(spec)
        report = self.dataset_validator.validate_loaded_dataset(
            spec,
            dataset,
            dataset_version=dataset_version,
        )
        self.dataset_registry.upsert_manifest(
            spec,
            validation_status=report.status,
            metadata={
                "validation_report_id": report.report_id,
                "train_samples": report.stats.get("train_samples", 0),
                "eval_samples": report.stats.get("eval_samples", 0),
            },
        )

    def _ensure_cache_home(self) -> None:
        keras_home = self.config.data_cache_dir / "keras"
        keras_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("KERAS_HOME", str(keras_home))

    def _load_cifar10(self) -> LoadedDataset:
        tf = _get_tensorflow()
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return LoadedDataset(
            train_features=x_train.astype("float32") / 255.0,
            train_labels=y_train.astype("int64").reshape(-1),
            eval_features=x_test.astype("float32") / 255.0,
            eval_labels=y_test.astype("int64").reshape(-1),
        )

    def _load_mnist(self) -> LoadedDataset:
        tf = _get_tensorflow()
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        return LoadedDataset(
            train_features=(x_train[..., np.newaxis].astype("float32") / 255.0),
            train_labels=y_train.astype("int64").reshape(-1),
            eval_features=(x_test[..., np.newaxis].astype("float32") / 255.0),
            eval_labels=y_test.astype("int64").reshape(-1),
        )

    def _load_imdb_reviews(self, spec: DatasetSpec) -> LoadedDataset:
        tf = _get_tensorflow()
        vocab_size = int(spec.metadata.get("vocab_size", 20000))
        sequence_length = int(spec.metadata.get("sequence_length", 256))
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
            num_words=vocab_size
        )
        pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
        return LoadedDataset(
            train_features=pad_sequences(
                x_train,
                maxlen=sequence_length,
                padding="post",
                truncating="post",
            ).astype("float32"),
            train_labels=np.asarray(y_train, dtype="int64"),
            eval_features=pad_sequences(
                x_test,
                maxlen=sequence_length,
                padding="post",
                truncating="post",
            ).astype("float32"),
            eval_labels=np.asarray(y_test, dtype="int64"),
        )

    def _load_ingested_dataset(self, spec: DatasetSpec) -> LoadedDataset:
        artifact = self.ingestion_service.ingest(spec)
        return self._load_npz_bundle(artifact.local_path)

    def _load_npz_bundle(self, path: Path) -> LoadedDataset:
        with np.load(path, allow_pickle=False) as bundle:
            required_keys = {
                "train_features",
                "train_labels",
                "eval_features",
                "eval_labels",
            }
            missing_keys = required_keys.difference(bundle.files)
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(
                    f"Dataset bundle is missing required arrays: {missing}"
                )

            return LoadedDataset(
                train_features=np.asarray(bundle["train_features"], dtype=np.float32),
                train_labels=np.asarray(bundle["train_labels"], dtype=np.int64),
                eval_features=np.asarray(bundle["eval_features"], dtype=np.float32),
                eval_labels=np.asarray(bundle["eval_labels"], dtype=np.int64),
            )


class DatasetBatchProvider:
    """Serve repeatable train/eval batches from resolved dataset metadata."""

    def __init__(
        self,
        context_manager: ContextManager,
        config: GhostConfig | None = None,
        loader: RealDatasetLoader | None = None,
    ):
        self.context_manager = context_manager
        self.config = config or get_config()
        self.loader = loader or RealDatasetLoader(config=self.config)
        runtime = self.context_manager.get_runtime_bucket("datasets")
        self._cursors: dict[str, dict[str, np.ndarray | int]] = runtime.setdefault(
            "cursors", {}
        )

    def next_training_batch(
        self,
        model_id: str,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, DatasetSpec]:
        return self._next_batch(model_id, "train", batch_size, shuffle=True)

    def next_eval_batch(
        self,
        model_id: str,
        batch_size: int = 128,
    ) -> tuple[np.ndarray, np.ndarray, DatasetSpec]:
        return self._next_batch(model_id, "eval", batch_size, shuffle=False)

    def _next_batch(
        self,
        model_id: str,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
    ) -> tuple[np.ndarray, np.ndarray, DatasetSpec]:
        spec = self._resolve_dataset_spec(model_id)
        dataset = self.loader.load(spec)
        features = (
            dataset.train_features if split == "train" else dataset.eval_features
        )
        labels = dataset.train_labels if split == "train" else dataset.eval_labels

        sample_count = min(batch_size, len(labels))
        if sample_count <= 0:
            raise RuntimeError(f"Dataset '{spec.dataset_id}' does not contain samples")

        state = self._get_cursor_state(model_id, split, len(labels), shuffle)
        order = state["order"]
        offset = int(state["offset"])

        if offset + sample_count <= len(order):
            indices = order[offset : offset + sample_count]
            next_offset = offset + sample_count
            if next_offset >= len(order):
                next_offset = 0
                if shuffle:
                    order = np.random.permutation(len(labels))
        else:
            head = order[offset:]
            remaining = sample_count - len(head)
            if shuffle:
                order = np.random.permutation(len(labels))
            else:
                order = np.arange(len(labels), dtype=np.int64)
            tail = order[:remaining]
            indices = np.concatenate([head, tail])
            next_offset = remaining

        state["order"] = order
        state["offset"] = next_offset
        return features[indices], labels[indices], spec

    def _get_cursor_state(
        self,
        model_id: str,
        split: str,
        dataset_size: int,
        shuffle: bool,
    ) -> dict[str, np.ndarray | int]:
        key = f"{model_id}:{split}"
        order = (
            np.random.permutation(dataset_size)
            if shuffle
            else np.arange(dataset_size, dtype=np.int64)
        )
        state = self._cursors.setdefault(key, {"offset": 0, "order": order})
        current_order = state.get("order")
        if not isinstance(current_order, np.ndarray) or len(current_order) != dataset_size:
            state["order"] = order
            state["offset"] = 0
        return state

    def _resolve_dataset_spec(self, model_id: str) -> DatasetSpec:
        ctx = self.context_manager.get_context(model_id)
        if ctx is None:
            raise RuntimeError(f"Model context not found for dataset-backed model: {model_id}")

        dataset_payload = ctx.metadata.get("dataset_spec")
        if not isinstance(dataset_payload, dict):
            raise RuntimeError(
                "Model does not have a resolved dataset. Provide dataset_ref or dataset before training."
            )

        return DatasetSpec(**dataset_payload)