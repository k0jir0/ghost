"""Metadata persistence for Ghost control-plane records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4


class MetadataStoreBackend(Protocol):
    """Backend protocol for persisted Ghost metadata records."""

    def save_record(
        self,
        category: str,
        record_id: str,
        payload: dict[str, Any],
    ) -> None: ...

    def load_record(self, category: str, record_id: str) -> dict[str, Any] | None: ...

    def list_records(self, category: str) -> list[dict[str, Any]]: ...

    def delete_record(self, category: str, record_id: str) -> bool: ...


class JsonFileMetadataBackend:
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


class SQLiteMetadataBackend:
    """Persist metadata records in a single SQLite database file."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save_record(
        self,
        category: str,
        record_id: str,
        payload: dict[str, Any],
    ) -> None:
        encoded = json.dumps(payload, indent=2, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO metadata_records (category, record_id, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(category, record_id)
                DO UPDATE SET payload = excluded.payload
                """,
                (category, record_id, encoded),
            )

    def load_record(self, category: str, record_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload
                FROM metadata_records
                WHERE category = ? AND record_id = ?
                """,
                (category, record_id),
            ).fetchone()
        if row is None:
            return None
        loaded = json.loads(str(row["payload"]))
        if not isinstance(loaded, dict):
            return None
        return cast(dict[str, Any], loaded)

    def list_records(self, category: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload
                FROM metadata_records
                WHERE category = ?
                ORDER BY record_id
                """,
                (category,),
            ).fetchall()
        for row in rows:
            try:
                loaded = json.loads(str(row["payload"]))
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                records.append(cast(dict[str, Any], loaded))
        return records

    def delete_record(self, category: str, record_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM metadata_records
                WHERE category = ? AND record_id = ?
                """,
                (category, record_id),
            )
        return cursor.rowcount > 0

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata_records (
                    category TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (category, record_id)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection


class MetadataStore:
    """Facade around pluggable metadata persistence backends."""

    def __init__(
        self,
        root_path: str | Path | None = None,
        *,
        backend: MetadataStoreBackend | None = None,
    ):
        self.root_path = Path(root_path or "./data/metadata")
        self.backend = backend or JsonFileMetadataBackend(self.root_path)

    def save_record(
        self,
        category: str,
        record_id: str,
        payload: dict[str, Any],
    ) -> None:
        self.backend.save_record(category, record_id, payload)

    def load_record(self, category: str, record_id: str) -> dict[str, Any] | None:
        return self.backend.load_record(category, record_id)

    def list_records(self, category: str) -> list[dict[str, Any]]:
        return self.backend.list_records(category)

    def delete_record(self, category: str, record_id: str) -> bool:
        return self.backend.delete_record(category, record_id)
