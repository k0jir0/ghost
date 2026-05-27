"""Shared task queue storage for the autonomous training agent and MCP tools."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned or "task"


@dataclass(frozen=True)
class QueueTask:
    """Single task item stored in the training queue."""

    text: str
    completed: bool = False
    raw: str = ""
    task_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": self.text,
            "completed": self.completed,
        }
        if self.task_id is not None:
            payload["task_id"] = self.task_id
        if self.created_at is not None:
            payload["created_at"] = self.created_at
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> QueueTask:
        metadata = payload.get("metadata", {})
        return cls(
            text=str(payload.get("text", "")).strip(),
            completed=bool(payload.get("completed", False)),
            raw=str(payload.get("raw", "")),
            task_id=(
                str(payload["task_id"]) if payload.get("task_id") is not None else None
            ),
            created_at=(
                str(payload["created_at"])
                if payload.get("created_at") is not None
                else None
            ),
            updated_at=(
                str(payload["updated_at"])
                if payload.get("updated_at") is not None
                else None
            ),
            metadata=metadata if isinstance(metadata, dict) else {},
        )


class TaskQueueStore:
    """Load and mutate the agent task queue across JSON and legacy markdown."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._legacy_markdown_path = (
            self.path.with_suffix(".md")
            if self.path.suffix.lower() == ".json"
            else None
        )

    def exists(self) -> bool:
        return self.active_path().exists()

    def active_path(self) -> Path:
        if self.path.exists():
            return self.path
        if (
            self._legacy_markdown_path is not None
            and self._legacy_markdown_path.exists()
        ):
            return self._legacy_markdown_path
        return self.path

    def active_format(self) -> str:
        return "markdown" if self.active_path().suffix.lower() == ".md" else "json"

    def list_tasks(self, *, include_completed: bool = False) -> list[QueueTask]:
        source = self.active_path()
        if not source.exists():
            return []
        if source.suffix.lower() == ".md":
            return self._load_markdown_tasks(
                source, include_completed=include_completed
            )
        return self._load_json_tasks(source, include_completed=include_completed)

    def pending_tasks(self) -> list[QueueTask]:
        return self.list_tasks(include_completed=False)

    def add_task(
        self,
        text: str,
        *,
        task_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> QueueTask:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Task text cannot be empty")

        if self.path.suffix.lower() == ".md":
            tasks = self._load_markdown_tasks(self.path, include_completed=True)
            resolved_task_id = task_id or self._generate_markdown_task_id(
                clean_text, tasks
            )
            self._ensure_unique_markdown_task_id(resolved_task_id, tasks)
            task = QueueTask(
                text=clean_text,
                completed=False,
                task_id=resolved_task_id,
                updated_at=_utc_now_iso(),
                metadata=dict(metadata or {}),
            )
            tasks.append(task)
            self._write_markdown_tasks(tasks)
            return task

        payload = self._load_json_payload_for_write()
        queue = payload.setdefault("queue", [])
        now = _utc_now_iso()
        resolved_task_id = task_id or self._generate_json_task_id(clean_text, queue)
        self._ensure_unique_json_task_id(resolved_task_id, queue)
        task = QueueTask(
            text=clean_text,
            completed=False,
            task_id=resolved_task_id,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        queue.append(task.to_dict())
        self._write_json_payload(payload)
        return task

    def complete_task(self, task: QueueTask | Mapping[str, Any]) -> QueueTask | None:
        return self.update_task(
            task_id=self._task_value(task, "task_id"),
            match_text=self._task_value(task, "text"),
            raw=self._task_value(task, "raw"),
            completed=True,
        )

    def update_task(
        self,
        *,
        task_id: str | None = None,
        match_text: str | None = None,
        raw: str | None = None,
        text: str | None = None,
        completed: bool | None = None,
    ) -> QueueTask | None:
        if task_id is None and match_text is None and raw is None:
            raise ValueError("Task update requires task_id, match_text, or raw")
        if text is None and completed is None:
            raise ValueError("Task update requires at least one field to change")

        clean_text = text.strip() if text is not None else None
        if text is not None and not clean_text:
            raise ValueError("Task text cannot be empty")

        if self.path.suffix.lower() == ".md":
            tasks = self._load_markdown_tasks(self.path, include_completed=True)
            for index, existing in enumerate(tasks):
                if self._matches_task(
                    existing,
                    task_id=task_id,
                    match_text=match_text,
                    raw=raw,
                ):
                    updated = QueueTask(
                        text=clean_text or existing.text,
                        completed=(
                            existing.completed if completed is None else completed
                        ),
                        task_id=existing.task_id,
                        updated_at=_utc_now_iso(),
                        metadata=existing.metadata,
                    )
                    tasks[index] = updated
                    self._write_markdown_tasks(tasks)
                    return updated
            return None

        payload = self._load_json_payload_for_write()
        queue = payload.setdefault("queue", [])
        for item in queue:
            if not isinstance(item, dict):
                continue
            if self._matches_payload(
                item,
                task_id=task_id,
                match_text=match_text,
                raw=raw,
            ):
                if clean_text is not None:
                    item["text"] = clean_text
                if completed is not None:
                    item["completed"] = completed
                item["updated_at"] = _utc_now_iso()
                self._write_json_payload(payload)
                return QueueTask.from_dict(item)
        return None

    def delete_task(
        self,
        *,
        task_id: str | None = None,
        match_text: str | None = None,
        raw: str | None = None,
    ) -> QueueTask | None:
        if task_id is None and match_text is None and raw is None:
            raise ValueError("Task deletion requires task_id, match_text, or raw")

        if self.path.suffix.lower() == ".md":
            tasks = self._load_markdown_tasks(self.path, include_completed=True)
            for index, existing in enumerate(tasks):
                if self._matches_task(
                    existing,
                    task_id=task_id,
                    match_text=match_text,
                    raw=raw,
                ):
                    removed = tasks.pop(index)
                    self._write_markdown_tasks(tasks)
                    return removed
            return None

        payload = self._load_json_payload_for_write()
        queue = payload.setdefault("queue", [])
        for index, item in enumerate(queue):
            if not isinstance(item, dict):
                continue
            if self._matches_payload(
                item,
                task_id=task_id,
                match_text=match_text,
                raw=raw,
            ):
                removed = QueueTask.from_dict(item)
                queue.pop(index)
                self._write_json_payload(payload)
                return removed
        return None

    def _task_value(
        self,
        task: QueueTask | Mapping[str, Any],
        key: str,
        default: Any = None,
    ) -> Any:
        if isinstance(task, QueueTask):
            return getattr(task, key, default)
        return task.get(key, default)

    def _matches_task(
        self,
        task: QueueTask,
        *,
        task_id: str | None,
        match_text: str | None,
        raw: str | None,
    ) -> bool:
        if task_id is not None and task.task_id == task_id:
            return True
        if raw is not None and task.raw == raw:
            return True
        if match_text is not None and task.text == match_text:
            return True
        return False

    def _matches_payload(
        self,
        payload: Mapping[str, Any],
        *,
        task_id: str | None,
        match_text: str | None,
        raw: str | None,
    ) -> bool:
        payload_task_id = payload.get("task_id")
        payload_text = payload.get("text")
        payload_raw = payload.get("raw")

        if task_id is not None and payload_task_id == task_id:
            return True
        if raw is not None and payload_raw == raw:
            return True
        if match_text is not None and payload_text == match_text:
            return True
        return False

    def _load_json_tasks(
        self,
        path: Path,
        *,
        include_completed: bool,
    ) -> list[QueueTask]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []

        queue = payload.get("queue") if isinstance(payload, dict) else payload
        if not isinstance(queue, list):
            return []

        tasks: list[QueueTask] = []
        for item in queue:
            if not isinstance(item, dict):
                continue
            task = QueueTask.from_dict(item)
            if task.text and (include_completed or not task.completed):
                tasks.append(task)
        return tasks

    def _load_markdown_tasks(
        self,
        path: Path,
        *,
        include_completed: bool,
    ) -> list[QueueTask]:
        if not path.exists():
            return []

        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return []

        tasks: list[QueueTask] = []
        in_queue = False
        task_index = 0

        for line in content.splitlines():
            if "## Queue" in line or "## Tasks" in line:
                in_queue = True
                continue

            if in_queue and line.startswith("## "):
                break

            if in_queue and line.strip().startswith("- ["):
                stripped = line.strip()
                completed = stripped.lower().startswith("- [x]")
                task_text = stripped[5:].strip() if len(stripped) > 5 else ""
                if not task_text:
                    continue
                task_index += 1
                task = QueueTask(
                    text=task_text,
                    completed=completed,
                    raw=stripped,
                    task_id=f"{_slugify(task_text)}-{task_index}",
                )
                if include_completed or not completed:
                    tasks.append(task)

        return tasks

    def _load_json_payload_for_write(self) -> dict[str, Any]:
        payload: dict[str, Any]
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = cast(dict[str, Any], loaded)
                else:
                    payload = self._default_payload()
                    if isinstance(loaded, list):
                        payload["queue"] = loaded
            except Exception:
                payload = self._default_payload()
        elif (
            self._legacy_markdown_path is not None
            and self._legacy_markdown_path.exists()
        ):
            payload = self._default_payload()
            payload["queue"] = [
                task.to_dict()
                for task in self._load_markdown_tasks(
                    self._legacy_markdown_path,
                    include_completed=True,
                )
            ]
        else:
            payload = self._default_payload()

        if not isinstance(payload, dict):
            queue = payload if isinstance(payload, list) else []
            payload = self._default_payload()
            payload["queue"] = queue

        queue = payload.get("queue")
        if not isinstance(queue, list):
            payload["queue"] = []

        payload.setdefault("version", 1)
        return payload

    def _default_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "updated_at": _utc_now_iso(),
            "queue": [],
        }

    def _write_json_payload(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _utc_now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_markdown_tasks(self, tasks: list[QueueTask]) -> None:
        lines = ["## Queue", ""]
        for task in tasks:
            marker = "x" if task.completed else " "
            lines.append(f"- [{marker}] {task.text}")
        lines.append("")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(lines), encoding="utf-8")

    def _generate_json_task_id(self, text: str, queue: list[Any]) -> str:
        base = _slugify(text)
        existing = {
            str(item.get("task_id"))
            for item in queue
            if isinstance(item, dict) and item.get("task_id") is not None
        }
        candidate = base
        counter = 2
        while candidate in existing:
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    def _generate_markdown_task_id(self, text: str, tasks: list[QueueTask]) -> str:
        base = _slugify(text)
        existing = {task.task_id for task in tasks if task.task_id is not None}
        candidate = base
        counter = 2
        while candidate in existing:
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    def _ensure_unique_json_task_id(
        self,
        task_id: str,
        queue: list[Any],
    ) -> None:
        for item in queue:
            if not isinstance(item, dict):
                continue
            if item.get("task_id") == task_id:
                raise ValueError(f"Task id already exists: {task_id}")

    def _ensure_unique_markdown_task_id(
        self,
        task_id: str,
        tasks: list[QueueTask],
    ) -> None:
        for task in tasks:
            if task.task_id == task_id:
                raise ValueError(f"Task id already exists: {task_id}")
