"""Dataset catalog and resolution for Ghost.

This module provides a stable boundary between high-level planning and the
eventual data-loading pipeline. For now it focuses on catalog metadata,
resolution, and explicit synthetic-data controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ghost.config import GhostConfig, get_config


@dataclass
class DatasetSpec:
    """Metadata describing a dataset available to Ghost."""

    dataset_id: str
    task_type: str
    source: str
    input_shape: tuple[int, ...]
    num_classes: int
    synthetic: bool
    aliases: tuple[str, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetResolver:
    """Resolve dataset references into catalog entries.

    The resolver does not load dataset contents yet. It establishes the
    contract for dataset discovery and enforces explicit synthetic opt-in.
    """

    def __init__(
        self,
        config: GhostConfig | None = None,
        specs: list[DatasetSpec] | None = None,
    ):
        self.config = config or get_config()
        self._specs: dict[str, DatasetSpec] = {}
        self._aliases: dict[str, str] = {}

        for spec in specs or self._default_specs():
            self.register(spec)

    def register(self, spec: DatasetSpec) -> None:
        """Register a dataset spec and its aliases."""
        canonical_id = self._normalize_key(spec.dataset_id)
        self._specs[canonical_id] = spec
        self._aliases[canonical_id] = canonical_id

        for alias in spec.aliases:
            self._aliases[self._normalize_key(alias)] = canonical_id

    def resolve(self, dataset_ref: str, *, allow_synthetic: bool) -> DatasetSpec:
        """Resolve a dataset reference to a known catalog entry.

        Args:
            dataset_ref: Canonical dataset id or alias.
            allow_synthetic: Must be explicitly set when synthetic datasets are
                permitted for demo-mode execution.

        Returns:
            Matching dataset specification.

        Raises:
            KeyError: If the dataset is unknown.
            ValueError: If the dataset is synthetic and synthetic use is not
                explicitly allowed.
        """
        key = self._normalize_key(dataset_ref)
        canonical_id = self._aliases.get(key)
        if canonical_id is None:
            raise KeyError(f"Unknown dataset reference: {dataset_ref}")

        spec = self._specs[canonical_id]
        if spec.synthetic and not allow_synthetic:
            raise ValueError(
                "Synthetic dataset resolution requires allow_synthetic=True"
            )
        return spec

    def list_available(self, *, include_synthetic: bool = True) -> list[DatasetSpec]:
        """Return the known dataset catalog."""
        specs = list(self._specs.values())
        if not include_synthetic:
            specs = [spec for spec in specs if not spec.synthetic]
        return sorted(specs, key=lambda spec: spec.dataset_id)

    def _normalize_key(self, value: str) -> str:
        return value.strip().lower().replace("_", "-")

    def _default_specs(self) -> list[DatasetSpec]:
        return [
            DatasetSpec(
                dataset_id="cifar-10",
                aliases=("cifar10",),
                task_type="image-classification",
                source="builtin-catalog",
                input_shape=(3, 32, 32),
                num_classes=10,
                synthetic=False,
                description="Canonical small image classification benchmark.",
            ),
            DatasetSpec(
                dataset_id="mnist",
                task_type="image-classification",
                source="builtin-catalog",
                input_shape=(1, 28, 28),
                num_classes=10,
                synthetic=False,
                description="Handwritten digit classification benchmark.",
            ),
            DatasetSpec(
                dataset_id="imdb-reviews",
                aliases=("imdb",),
                task_type="text-classification",
                source="builtin-catalog",
                input_shape=(1,),
                num_classes=2,
                synthetic=False,
                description="Binary sentiment analysis benchmark.",
            ),
            DatasetSpec(
                dataset_id="synthetic-image-classification",
                aliases=("synthetic-image", "demo-image"),
                task_type="image-classification",
                source="synthetic",
                input_shape=(3, 224, 224),
                num_classes=10,
                synthetic=True,
                description="Synthetic image batches for demo-mode training.",
            ),
            DatasetSpec(
                dataset_id="synthetic-text-classification",
                aliases=("synthetic-text", "demo-text"),
                task_type="text-classification",
                source="synthetic",
                input_shape=(1,),
                num_classes=2,
                synthetic=True,
                description="Synthetic text batches for demo-mode training.",
            ),
        ]
