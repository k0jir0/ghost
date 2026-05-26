"""Audit logging for control-plane state changes."""

from __future__ import annotations

from ghost.config import GhostConfig, get_config
from ghost.metadata_store import MetadataStore
from ghost.schemas import AuditEntry


class AuditLogger:
    """Persist audit entries for registry and workflow operations."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )

    def record(
        self,
        action: str,
        *,
        subject_type: str,
        subject_id: str,
        actor: str,
        details: dict[str, object] | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            audit_id=self._entry_id(subject_type, subject_id, action),
            action=action,
            subject_type=subject_type,
            subject_id=subject_id,
            actor=actor,
            details=details or {},
        )
        self.metadata_store.save_record("audit-log", entry.audit_id, entry.to_dict())
        return entry

    def list_entries(
        self,
        *,
        subject_type: str | None = None,
        subject_id: str | None = None,
    ) -> list[AuditEntry]:
        entries: list[AuditEntry] = []
        for payload in self.metadata_store.list_records("audit-log"):
            try:
                entry = AuditEntry.from_dict(payload)
            except Exception:
                continue
            if subject_type is not None and entry.subject_type != subject_type:
                continue
            if subject_id is not None and entry.subject_id != subject_id:
                continue
            entries.append(entry)
        return sorted(entries, key=lambda entry: entry.created_at)

    def _entry_id(self, subject_type: str, subject_id: str, action: str) -> str:
        safe_subject = subject_id.replace("/", "-")
        return f"{subject_type}__{safe_subject}__{action}"
