"""Persistent dataset manifest storage for Ghost."""

from __future__ import annotations

from typing import Any

from ghost.config import GhostConfig, get_config
from ghost.datasets import DatasetSpec
from ghost.metadata_store import MetadataStore
from ghost.schemas import DatasetManifest


class DatasetRegistry:
    """Store and retrieve versioned dataset manifests."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )

    def upsert_manifest(
        self,
        spec: DatasetSpec,
        *,
        validation_status: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> DatasetManifest:
        version = self.version_for_spec(spec)
        record_id = self._record_id(spec.dataset_id, version)
        existing_payload = self.metadata_store.load_record("dataset-manifests", record_id)
        existing = (
            DatasetManifest.from_dict(existing_payload)
            if isinstance(existing_payload, dict)
            else None
        )

        manifest = DatasetManifest(
            dataset_id=spec.dataset_id,
            version=version,
            source_uri=f"builtin://{spec.dataset_id}",
            schema={
                "task_type": spec.task_type,
                "input_shape": list(spec.input_shape),
                "num_classes": spec.num_classes,
            },
            splits={"train": {}, "eval": {}},
            validation_status=validation_status,
            metadata={
                "source": spec.source,
                "synthetic": spec.synthetic,
                **(spec.metadata if isinstance(spec.metadata, dict) else {}),
                **(metadata or {}),
            },
            created_at=existing.created_at if existing is not None else DatasetManifest(
                dataset_id=spec.dataset_id,
                version=version,
                source_uri=f"builtin://{spec.dataset_id}",
            ).created_at,
        )
        self.metadata_store.save_record(
            "dataset-manifests",
            record_id,
            manifest.to_dict(),
        )
        return manifest

    def get_manifest(self, dataset_id: str, version: str) -> DatasetManifest | None:
        payload = self.metadata_store.load_record(
            "dataset-manifests",
            self._record_id(dataset_id, version),
        )
        if not isinstance(payload, dict):
            return None
        return DatasetManifest.from_dict(payload)

    def list_manifests(self) -> list[DatasetManifest]:
        manifests: list[DatasetManifest] = []
        for payload in self.metadata_store.list_records("dataset-manifests"):
            try:
                manifests.append(DatasetManifest.from_dict(payload))
            except Exception:
                continue
        return manifests

    def version_for_spec(self, spec: DatasetSpec) -> str:
        metadata_version = spec.metadata.get("dataset_version") if isinstance(spec.metadata, dict) else None
        if metadata_version:
            return str(metadata_version)
        return "builtin-v1"

    def _record_id(self, dataset_id: str, version: str) -> str:
        safe_dataset = dataset_id.replace("/", "-")
        safe_version = version.replace("/", "-")
        return f"{safe_dataset}@{safe_version}"