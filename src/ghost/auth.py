"""Lightweight token-based authn/authz helpers for Ghost surfaces."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ghost.config import GhostConfig, get_config
from ghost.metadata_store import MetadataStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AccessTokenRecord:
    token_id: str
    subject: str
    scopes: list[str]
    token_hash: str
    revoked: bool = False
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuthService:
    """Issue, revoke, and authorize simple bearer tokens."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )

    def issue_token(
        self, subject: str, scopes: list[str]
    ) -> tuple[str, AccessTokenRecord]:
        raw_token = secrets.token_hex(16)
        token_id = secrets.token_hex(8)
        record = AccessTokenRecord(
            token_id=token_id,
            subject=subject,
            scopes=scopes,
            token_hash=self._hash(raw_token),
        )
        self.metadata_store.save_record("auth-tokens", token_id, record.to_dict())
        return raw_token, record

    def authorize(self, token: str, scope: str) -> bool:
        token_hash = self._hash(token)
        for payload in self.metadata_store.list_records("auth-tokens"):
            record = AccessTokenRecord(**payload)
            if record.revoked:
                continue
            if record.token_hash != token_hash:
                continue
            return scope in record.scopes
        return False

    def revoke(self, token_id: str) -> bool:
        payload = self.metadata_store.load_record("auth-tokens", token_id)
        if not isinstance(payload, dict):
            return False
        record = AccessTokenRecord(**payload)
        record.revoked = True
        self.metadata_store.save_record("auth-tokens", token_id, record.to_dict())
        return True

    def _hash(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
