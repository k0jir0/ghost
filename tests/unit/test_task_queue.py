"""Unit tests for the shared task queue store."""

from __future__ import annotations

import json
from pathlib import Path

from ghost.task_queue import TaskQueueStore


def test_json_store_adds_and_lists_tasks(tmp_path: Path) -> None:
    store = TaskQueueStore(tmp_path / "TASKS.json")

    created = store.add_task("Train MLP classifier")
    tasks = store.pending_tasks()

    assert len(tasks) == 1
    assert tasks[0]["text"] == "Train MLP classifier"
    assert tasks[0]["task_id"] == created.task_id


def test_json_store_reads_legacy_markdown_when_json_missing(tmp_path: Path) -> None:
    legacy = tmp_path / "TASKS.md"
    legacy.write_text("## Queue\n\n- [ ] Train MLP\n- [x] Done\n", encoding="utf-8")
    store = TaskQueueStore(tmp_path / "TASKS.json")

    tasks = store.pending_tasks()

    assert [task["text"] for task in tasks] == ["Train MLP"]


def test_json_store_prefers_json_over_legacy_markdown(tmp_path: Path) -> None:
    legacy = tmp_path / "TASKS.md"
    legacy.write_text("## Queue\n\n- [ ] Legacy markdown task\n", encoding="utf-8")
    primary = tmp_path / "TASKS.json"
    primary.write_text(
        json.dumps(
            {
                "version": 1,
                "queue": [
                    {
                        "task_id": "primary",
                        "text": "Primary JSON task",
                        "completed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    store = TaskQueueStore(primary)

    tasks = store.pending_tasks()

    assert [task["text"] for task in tasks] == ["Primary JSON task"]


def test_json_store_write_migrates_legacy_markdown_to_json(tmp_path: Path) -> None:
    legacy = tmp_path / "TASKS.md"
    legacy.write_text("## Queue\n\n- [ ] Train MLP\n", encoding="utf-8")
    store = TaskQueueStore(tmp_path / "TASKS.json")

    task = store.pending_tasks()[0]
    updated = store.complete_task(task)

    assert updated is not None
    payload = json.loads((tmp_path / "TASKS.json").read_text(encoding="utf-8"))
    assert payload["queue"][0]["completed"] is True


def test_markdown_store_still_updates_explicit_markdown_files(tmp_path: Path) -> None:
    path = tmp_path / "TASKS.md"
    path.write_text("## Queue\n\n- [ ] Train MLP\n", encoding="utf-8")
    store = TaskQueueStore(path)

    task = store.pending_tasks()[0]
    updated = store.complete_task(task)

    assert updated is not None
    assert "- [x] Train MLP" in path.read_text(encoding="utf-8")


def test_update_task_changes_text(tmp_path: Path) -> None:
    store = TaskQueueStore(tmp_path / "TASKS.json")
    created = store.add_task("Train MLP")

    updated = store.update_task(task_id=created.task_id, text="Train ResNet")

    assert updated is not None
    assert updated.text == "Train ResNet"