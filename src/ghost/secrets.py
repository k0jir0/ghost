"""Secret-provider abstractions for Ghost production integrations."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from ghost.config import GhostConfig, get_config
from ghost.metadata_store import MetadataStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SecretProvider(Protocol):
    """Protocol implemented by environment, vault, or cloud secret providers."""

    provider_name: str

    def get_secret(self, name: str) -> str | None: ...


@dataclass
class SecretReference:
    """A named secret lookup request."""

    name: str
    provider: str = ""
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EnvironmentSecretProvider:
    """Resolve secrets from process environment variables."""

    provider_name = "env"

    def get_secret(self, name: str) -> str | None:
        return os.getenv(name)


class LocalDevelopmentSecretProvider:
    """Metadata-backed secret provider for local development and tests.

    This provider intentionally makes local storage explicit. Production
    deployments should supply a provider backed by their enterprise vault,
    cloud secrets manager, or workload identity service.
    """

    provider_name = "local"

    def __init__(
        self,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )

    def set_secret(self, name: str, value: str, *, actor: str = "system") -> None:
        self.metadata_store.save_record(
            "local-secrets",
            name,
            {
                "name": name,
                "value": value,
                "updated_by": actor,
                "updated_at": _utc_now_iso(),
            },
        )

    def get_secret(self, name: str) -> str | None:
        payload = self.metadata_store.load_record("local-secrets", name)
        if not isinstance(payload, dict):
            return None
        value = payload.get("value")
        return str(value) if value is not None else None


class SecretResolver:
    """Resolve secret references through a configured provider chain."""

    def __init__(self, providers: list[SecretProvider] | None = None):
        configured = providers or [EnvironmentSecretProvider()]
        self.providers = {provider.provider_name: provider for provider in configured}

    def resolve(self, reference: SecretReference | str) -> str | None:
        secret_reference = (
            SecretReference(name=reference) if isinstance(reference, str) else reference
        )
        providers = (
            [self.providers[secret_reference.provider]]
            if secret_reference.provider
            else list(self.providers.values())
        )
        for provider in providers:
            value = provider.get_secret(secret_reference.name)
            if value is not None:
                return value
        if secret_reference.required:
            raise KeyError(f"Secret not found: {secret_reference.name}")
        return None

    def describe_providers(self) -> list[dict[str, str]]:
        return [
            {"provider": provider_name}
            for provider_name in sorted(self.providers.keys())
        ]
