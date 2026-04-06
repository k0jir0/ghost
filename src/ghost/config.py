"""Configuration management for Ghost platform.

Uses Pydantic settings for type-safe configuration with environment variable support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GhostConfig(BaseSettings):
    """Main configuration for the Ghost platform.
    
    Loads from environment variables with optional .env file support.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama Configuration
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    ollama_model: str = Field(
        default="llama3",
        description="Default Ollama model for assistance"
    )
    ollama_timeout: int = Field(
        default=60,
        description="Ollama request timeout in seconds"
    )

    # Training Backend
    training_backend: Literal["pytorch", "tensorflow", "auto"] = Field(
        default="auto",
        description="Default ML backend for training"
    )
    gpu_enabled: bool = Field(
        default=True,
        description="Enable GPU acceleration if available"
    )
    cuda_visible_devices: str = Field(
        default="0",
        description="CUDA visible devices"
    )

    # Paths
    model_cache_dir: Path = Field(
        default=Path("./models"),
        description="Directory for model checkpoints"
    )
    data_cache_dir: Path = Field(
        default=Path("./data"),
        description="Directory for dataset cache"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: str = Field(
        default="ghost.log",
        description="Log file path"
    )

    # Training Defaults
    default_batch_size: int = Field(
        default=32,
        description="Default batch size for training"
    )
    default_learning_rate: float = Field(
        default=0.001,
        description="Default learning rate"
    )
    default_epochs: int = Field(
        default=10,
        description="Default number of training epochs"
    )
    checkpoint_interval: int = Field(
        default=5,
        description="Save checkpoint every N epochs"
    )

    # Agent Settings
    ai_backend: Literal["ollama", "openai", "anthropic"] = Field(
        default="ollama",
        description="AI backend for agent assistance"
    )
    max_iterations: int = Field(
        default=100,
        description="Maximum agent iterations before check-in"
    )
    daily_token_budget: float = Field(
        default=10.0,
        description="Daily token budget in USD"
    )

    # Health Checks
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    gpu_memory_threshold: float = Field(
        default=0.9,
        description="GPU memory usage threshold"
    )
    system_memory_threshold: float = Field(
        default=0.85,
        description="System memory usage threshold"
    )

    @field_validator("model_cache_dir", "data_cache_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)

    def is_gpu_available(self) -> bool:
        """Check if GPU is available for training."""
        if not self.gpu_enabled:
            return False
        
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_backend(self) -> Literal["pytorch", "tensorflow"]:
        """Get the effective training backend."""
        if self.training_backend == "auto":
            try:
                import torch
                return "pytorch"
            except ImportError:
                return "tensorflow"
        return self.training_backend


# Global configuration instance
_config: GhostConfig | None = None


def get_config() -> GhostConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = GhostConfig()
        _config.ensure_directories()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
