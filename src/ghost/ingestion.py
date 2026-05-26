"""Dataset ingestion interfaces for filesystem and object-store-backed data."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse
from urllib.request import url2pathname

from ghost.config import GhostConfig, get_config
from ghost.datasets import DatasetSpec


def source_uri_for_spec(spec: DatasetSpec) -> str:
    """Return the declared source URI for an external dataset spec."""
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    source_uri = metadata.get("source_uri")
    if source_uri:
        return str(source_uri)
    raise ValueError(
        "Dataset spec does not declare a source_uri required for ingestion."
    )


@dataclass(frozen=True)
class IngestedDatasetArtifact:
    """Locally addressable artifact produced by an ingestion backend."""

    source_uri: str
    local_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetIngestor(Protocol):
    """Protocol for Ghost dataset ingestors."""

    def supports(self, source_uri: str) -> bool: ...

    def ingest(
        self,
        spec: DatasetSpec,
        source_uri: str,
    ) -> IngestedDatasetArtifact: ...


class FilesystemDatasetIngestor:
    """Resolve file-backed dataset bundles without copying them."""

    def supports(self, source_uri: str) -> bool:
        return source_uri.startswith("file://")

    def ingest(
        self,
        spec: DatasetSpec,
        source_uri: str,
    ) -> IngestedDatasetArtifact:
        path = self._path_from_uri(source_uri)
        if not path.exists():
            raise FileNotFoundError(f"Dataset artifact not found: {path}")
        return IngestedDatasetArtifact(source_uri=source_uri, local_path=path.resolve())

    def _path_from_uri(self, source_uri: str) -> Path:
        parsed = urlparse(source_uri)
        path_text = url2pathname(parsed.path)
        if parsed.netloc:
            path_text = f"//{parsed.netloc}{path_text}"
        return Path(path_text)


class ObjectStoreDatasetIngestor:
    """Stage object-store-backed dataset bundles into the local data cache."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        object_fetcher: Any | None = None,
    ):
        self.config = config or get_config()
        self.object_fetcher = object_fetcher

    def supports(self, source_uri: str) -> bool:
        return source_uri.startswith(("s3://", "minio://", "gs://"))

    def ingest(
        self,
        spec: DatasetSpec,
        source_uri: str,
    ) -> IngestedDatasetArtifact:
        destination = self._destination_path(spec, source_uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            if self.object_fetcher is None:
                raise RuntimeError(
                    "Object-store dataset ingestion requires an object_fetcher callback."
                )

            fetched = self.object_fetcher(source_uri, destination)
            fetched_path = Path(fetched) if fetched is not None else destination
            if (
                fetched_path != destination
                and fetched_path.exists()
                and not destination.exists()
            ):
                shutil.copyfile(fetched_path, destination)

        if not destination.exists():
            raise FileNotFoundError(
                f"Object-store dataset fetch did not materialize an artifact at {destination}"
            )

        return IngestedDatasetArtifact(source_uri=source_uri, local_path=destination)

    def _destination_path(self, spec: DatasetSpec, source_uri: str) -> Path:
        parsed = urlparse(source_uri)
        source_name = Path(parsed.path).name or f"{spec.dataset_id}.npz"
        version = self._version_for_spec(spec, source_uri)
        safe_dataset = spec.dataset_id.replace("/", "-")
        return (
            self.config.data_cache_dir
            / "ingested"
            / f"{safe_dataset}-{version}-{source_name}"
        )

    def _version_for_spec(self, spec: DatasetSpec, source_uri: str) -> str:
        metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
        if metadata.get("dataset_version"):
            return str(metadata["dataset_version"])
        return hashlib.sha256(source_uri.encode("utf-8")).hexdigest()[:12]


class DatasetIngestionService:
    """Select an ingestion backend for a dataset spec and stage its artifact."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        ingestors: list[DatasetIngestor] | None = None,
    ):
        self.config = config or get_config()
        self._ingestors = ingestors or [
            FilesystemDatasetIngestor(),
            ObjectStoreDatasetIngestor(config=self.config),
        ]

    def ingest(self, spec: DatasetSpec) -> IngestedDatasetArtifact:
        source_uri = source_uri_for_spec(spec)
        for ingestor in self._ingestors:
            if ingestor.supports(source_uri):
                return ingestor.ingest(spec, source_uri)
        raise KeyError(f"No dataset ingestor registered for source URI: {source_uri}")
