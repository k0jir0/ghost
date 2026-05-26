"""Lightweight JSON metadata persistence for Ghost control-plane records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from uuid import uuid4


class MetadataStore:
    """Persist JSON metadata records under category-specific directories."""

    def __init__(self, root_path: str | Path | None = None):
        self.root_path = Path(root_path or "./data/metadata")
        self.root_path.mkdir(parents=True, exist_ok=True)

    def save_record(
        self, category: str, record_id: str, payload: dict[str, Any]
    ) -> None:
        category_dir = self._category_dir(category)
        record_path = category_dir / f"{record_id}.json"
        temp_path = record_path.with_name(f".{record_path.name}.{uuid4().hex}.tmp")

        try:
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp_path.replace(record_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def load_record(self, category: str, record_id: str) -> dict[str, Any] | None:
        record_path = self._category_dir(category) / f"{record_id}.json"
        if not record_path.exists():
            return None
        loaded = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return None
        return cast(dict[str, Any], loaded)

    def list_records(self, category: str) -> list[dict[str, Any]]:
        category_dir = self._category_dir(category)
        records: list[dict[str, Any]] = []
        for record_path in sorted(category_dir.glob("*.json")):
            try:
                loaded = json.loads(record_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    records.append(cast(dict[str, Any], loaded))
            except Exception:
                continue
        return records

    def delete_record(self, category: str, record_id: str) -> bool:
        record_path = self._category_dir(category) / f"{record_id}.json"
        if not record_path.exists():
            return False
        record_path.unlink()
        return True

    def _category_dir(self, category: str) -> Path:
        category_dir = self.root_path / category
        category_dir.mkdir(parents=True, exist_ok=True)
        return category_dir
