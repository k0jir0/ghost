"""Lightweight feature registry and point-in-time feature retrieval."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ghost.config import GhostConfig, get_config
from ghost.metadata_store import MetadataStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FeatureDefinition:
    """Versioned reusable feature definition."""

    feature_name: str
    version: str
    value_type: str
    entities: list[str] = field(default_factory=list)
    description: str = ""
    transformation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)

    @property
    def definition_id(self) -> str:
        return f"{self.feature_name}__{self.version}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["definition_id"] = self.definition_id
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureDefinition:
        clean_payload = dict(payload)
        clean_payload.pop("definition_id", None)
        return cls(**clean_payload)


@dataclass
class FeatureValue:
    """Persisted point-in-time feature value for an entity."""

    entity_id: str
    feature_name: str
    version: str
    value: Any
    event_timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)

    @property
    def value_id(self) -> str:
        key = "|".join(
            [self.entity_id, self.feature_name, self.version, self.event_timestamp]
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return f"{self.entity_id}__{self.feature_name}__{self.version}__{digest}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["value_id"] = self.value_id
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureValue:
        clean_payload = dict(payload)
        clean_payload.pop("value_id", None)
        return cls(**clean_payload)


@dataclass
class FeatureMaterializationRecord:
    """Summary of an offline-to-online feature materialization."""

    materialization_id: str
    dataset_id: str
    feature_names: list[str]
    entity_count: int
    row_count: int
    created_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureMaterializationRecord:
        return cls(**payload)


class FeatureStore:
    """Persist feature definitions and point-in-time entity feature values."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )

    def register_definition(self, definition: FeatureDefinition) -> FeatureDefinition:
        self.metadata_store.save_record(
            "feature-definitions",
            definition.definition_id,
            definition.to_dict(),
        )
        return definition

    def get_definition(
        self,
        feature_name: str,
        version: str | None = None,
    ) -> FeatureDefinition | None:
        definitions = [
            definition
            for definition in self.list_definitions()
            if definition.feature_name == feature_name
        ]
        if version is not None:
            for definition in definitions:
                if definition.version == version:
                    return definition
            return None
        return definitions[-1] if definitions else None

    def list_definitions(self) -> list[FeatureDefinition]:
        definitions: list[FeatureDefinition] = []
        for payload in self.metadata_store.list_records("feature-definitions"):
            try:
                definitions.append(FeatureDefinition.from_dict(payload))
            except TypeError:
                continue
        return sorted(
            definitions,
            key=lambda definition: (definition.feature_name, definition.created_at),
        )

    def put_online_features(
        self,
        entity_id: str,
        values: Mapping[str, Any],
        *,
        event_timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[FeatureValue]:
        timestamp = event_timestamp or _utc_now_iso()
        stored: list[FeatureValue] = []
        for feature_name, value in values.items():
            definition = self.get_definition(feature_name)
            if definition is None:
                raise KeyError(f"Unknown feature definition: {feature_name}")
            feature_value = FeatureValue(
                entity_id=entity_id,
                feature_name=definition.feature_name,
                version=definition.version,
                value=value,
                event_timestamp=timestamp,
                metadata=metadata or {},
            )
            self.metadata_store.save_record(
                "feature-values",
                feature_value.value_id,
                feature_value.to_dict(),
            )
            stored.append(feature_value)
        return stored

    def materialize_offline(
        self,
        dataset_id: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        entity_key: str = "entity_id",
        event_timestamp_key: str = "event_timestamp",
        metadata: dict[str, Any] | None = None,
    ) -> FeatureMaterializationRecord:
        definitions = {
            definition.feature_name for definition in self.list_definitions()
        }
        feature_names: set[str] = set()
        entity_ids: set[str] = set()
        row_count = 0

        for row in rows:
            row_count += 1
            entity_value = row.get(entity_key)
            if entity_value is None:
                raise ValueError(f"Offline feature row is missing {entity_key}")
            entity_id = str(entity_value)
            entity_ids.add(entity_id)
            timestamp = row.get(event_timestamp_key)
            values = {
                key: value
                for key, value in row.items()
                if key in definitions and key not in {entity_key, event_timestamp_key}
            }
            if values:
                self.put_online_features(
                    entity_id,
                    values,
                    event_timestamp=str(timestamp or _utc_now_iso()),
                    metadata=metadata,
                )
                feature_names.update(values)

        materialization_id = self._materialization_id(dataset_id, row_count)
        record = FeatureMaterializationRecord(
            materialization_id=materialization_id,
            dataset_id=dataset_id,
            feature_names=sorted(feature_names),
            entity_count=len(entity_ids),
            row_count=row_count,
            metadata=metadata or {},
        )
        self.metadata_store.save_record(
            "feature-materializations",
            record.materialization_id,
            record.to_dict(),
        )
        return record

    def get_online_features(
        self,
        entity_id: str,
        feature_names: Iterable[str] | None = None,
        *,
        as_of: str | None = None,
    ) -> dict[str, Any]:
        requested = set(feature_names) if feature_names is not None else None
        latest: dict[str, FeatureValue] = {}
        for payload in self.metadata_store.list_records("feature-values"):
            try:
                value = FeatureValue.from_dict(payload)
            except TypeError:
                continue
            if value.entity_id != entity_id:
                continue
            if requested is not None and value.feature_name not in requested:
                continue
            if as_of is not None and value.event_timestamp > as_of:
                continue
            current = latest.get(value.feature_name)
            if current is None or value.event_timestamp >= current.event_timestamp:
                latest[value.feature_name] = value
        return {
            feature_name: value.value
            for feature_name, value in sorted(latest.items(), key=lambda item: item[0])
        }

    def _materialization_id(self, dataset_id: str, row_count: int) -> str:
        seed = f"{dataset_id}|{row_count}|{_utc_now_iso()}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
        return f"{dataset_id}__features__{digest}"
