"""Environment profiles for dev, staging, and production Ghost deployments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ghost.config import GhostConfig, get_config


@dataclass
class EnvironmentProfile:
    name: str
    data_dir: Path
    model_dir: Path
    metadata_dir: Path

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        return {key: str(value) for key, value in payload.items()}


class EnvironmentManager:
    """Provide isolated directory layouts for named environments."""

    def __init__(self, config: GhostConfig | None = None):
        self.config = config or get_config()

    def get_profile(self, name: str) -> EnvironmentProfile:
        base = Path("./deploy") / name
        return EnvironmentProfile(
            name=name,
            data_dir=base / "data",
            model_dir=base / "models",
            metadata_dir=base / "metadata",
        )

    def list_profiles(self) -> list[EnvironmentProfile]:
        return [self.get_profile(name) for name in ("dev", "staging", "production")]