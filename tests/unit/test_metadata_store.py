"""Unit tests for ghost.metadata_store."""

from __future__ import annotations

from pathlib import Path

from ghost.metadata_store import MetadataStore


def test_save_and_load_record(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata")

    store.save_record("runs", "run-1", {"run_id": "run-1", "status": "queued"})

    payload = store.load_record("runs", "run-1")

    assert payload == {"run_id": "run-1", "status": "queued"}


def test_list_records_returns_saved_payloads(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata")
    store.save_record("runs", "run-1", {"run_id": "run-1"})
    store.save_record("runs", "run-2", {"run_id": "run-2"})

    payloads = store.list_records("runs")

    assert {payload["run_id"] for payload in payloads} == {"run-1", "run-2"}


def test_save_record_overwrites_existing_payload_atomically(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata")
    store.save_record("runs", "run-1", {"run_id": "run-1", "status": "queued"})
    store.save_record(
        "runs",
        "run-1",
        {"run_id": "run-1", "status": "completed"},
    )

    payload = store.load_record("runs", "run-1")

    assert payload == {"run_id": "run-1", "status": "completed"}