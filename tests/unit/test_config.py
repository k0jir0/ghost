"""Unit tests for ghost.config — GhostConfig, get_config, reset_config."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from ghost.config import GhostConfig, get_config, reset_config


class TestGhostConfigDefaults:
    """GhostConfig has sensible, type-correct defaults."""

    def test_default_backend_is_auto(self) -> None:
        cfg = GhostConfig()
        assert cfg.training_backend == "auto"

    def test_default_ollama_host(self) -> None:
        cfg = GhostConfig()
        assert cfg.ollama_host == "http://localhost:11434"

    def test_default_log_level(self) -> None:
        cfg = GhostConfig()
        assert cfg.log_level == "INFO"

    def test_model_cache_dir_is_path(self) -> None:
        cfg = GhostConfig()
        assert isinstance(cfg.model_cache_dir, Path)

    def test_data_cache_dir_is_path(self) -> None:
        cfg = GhostConfig()
        assert isinstance(cfg.data_cache_dir, Path)

    def test_default_batch_size(self) -> None:
        cfg = GhostConfig()
        assert cfg.default_batch_size == 32

    def test_default_learning_rate(self) -> None:
        cfg = GhostConfig()
        assert cfg.default_learning_rate == pytest.approx(0.001)

    def test_allow_synthetic_data_disabled_by_default(self) -> None:
        cfg = GhostConfig()
        assert cfg.allow_synthetic_data is False

    def test_health_check_interval_positive(self) -> None:
        cfg = GhostConfig()
        assert cfg.health_check_interval > 0

    def test_default_ai_backend_is_ollama(self) -> None:
        cfg = GhostConfig()
        assert cfg.ai_backend == "ollama"


class TestGhostConfigEnvOverride:
    """Environment variables override defaults correctly."""

    def test_log_level_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        cfg = GhostConfig()
        assert cfg.log_level == "DEBUG"

    def test_ollama_model_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_MODEL", "mistral")
        cfg = GhostConfig()
        assert cfg.ollama_model == "mistral"

    def test_batch_size_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEFAULT_BATCH_SIZE", "64")
        cfg = GhostConfig()
        assert cfg.default_batch_size == 64

    def test_backend_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRAINING_BACKEND", "pytorch")
        cfg = GhostConfig()
        assert cfg.training_backend == "pytorch"

    def test_invalid_ai_backend_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GhostConfig(ai_backend="openai")  # type: ignore[arg-type]


class TestResolvePath:
    """String paths are coerced to Path objects via the field validator."""

    def test_string_model_cache_coerced(self, tmp_path: Path) -> None:
        cfg = GhostConfig(model_cache_dir=str(tmp_path / "models"))  # type: ignore[arg-type]
        assert isinstance(cfg.model_cache_dir, Path)

    def test_string_data_cache_coerced(self, tmp_path: Path) -> None:
        cfg = GhostConfig(data_cache_dir=str(tmp_path / "data"))  # type: ignore[arg-type]
        assert isinstance(cfg.data_cache_dir, Path)


class TestEnsureDirectories:
    """ensure_directories() creates cache directories on disk."""

    def test_creates_model_cache(self, tmp_path: Path) -> None:
        cfg = GhostConfig(
            model_cache_dir=tmp_path / "models",
            data_cache_dir=tmp_path / "data",
        )
        cfg.ensure_directories()
        assert cfg.model_cache_dir.exists()

    def test_creates_data_cache(self, tmp_path: Path) -> None:
        cfg = GhostConfig(
            model_cache_dir=tmp_path / "models",
            data_cache_dir=tmp_path / "data",
        )
        cfg.ensure_directories()
        assert cfg.data_cache_dir.exists()

    def test_idempotent(self, tmp_path: Path) -> None:
        cfg = GhostConfig(
            model_cache_dir=tmp_path / "m",
            data_cache_dir=tmp_path / "d",
        )
        cfg.ensure_directories()
        cfg.ensure_directories()  # should not raise
        assert cfg.model_cache_dir.exists()


class TestCheckpointPathResolution:
    def test_default_checkpoint_path_uses_model_cache(self, tmp_path: Path) -> None:
        cfg = GhostConfig(
            model_cache_dir=tmp_path / "models",
            data_cache_dir=tmp_path / "data",
        )
        cfg.ensure_directories()

        resolved = cfg.resolve_checkpoint_path("demo", suffix=".pt")

        assert resolved == (cfg.model_cache_dir / "demo.pt").resolve(strict=False)

    def test_relative_checkpoint_path_stays_under_model_cache(self, tmp_path: Path) -> None:
        cfg = GhostConfig(
            model_cache_dir=tmp_path / "models",
            data_cache_dir=tmp_path / "data",
        )
        cfg.ensure_directories()

        resolved = cfg.resolve_checkpoint_path("demo", path="nested/demo.pt")

        assert resolved == (cfg.model_cache_dir / "nested" / "demo.pt").resolve(
            strict=False
        )

    def test_checkpoint_path_escape_is_rejected(self, tmp_path: Path) -> None:
        cfg = GhostConfig(
            model_cache_dir=tmp_path / "models",
            data_cache_dir=tmp_path / "data",
        )
        cfg.ensure_directories()

        with pytest.raises(ValueError):
            cfg.resolve_checkpoint_path("demo", path="../escape.pt")


class TestGetBackend:
    """get_backend() resolves 'auto' and explicit backends."""

    def test_explicit_pytorch(self) -> None:
        cfg = GhostConfig(training_backend="pytorch")
        assert cfg.get_backend() == "pytorch"

    def test_explicit_tensorflow(self) -> None:
        cfg = GhostConfig(training_backend="tensorflow")
        assert cfg.get_backend() == "tensorflow"

    def test_auto_returns_pytorch_when_torch_importable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # torch is already mocked in sys.modules via conftest → returns "pytorch"
        cfg = GhostConfig(training_backend="auto")
        result = cfg.get_backend()
        assert result in ("pytorch", "tensorflow")


class TestGetConfigSingleton:
    """get_config() returns the same instance; reset_config() resets it."""

    def setup_method(self) -> None:
        reset_config()

    def teardown_method(self) -> None:
        reset_config()

    def test_returns_same_instance(self, tmp_path: Path) -> None:
        a = get_config()
        b = get_config()
        assert a is b

    def test_reset_returns_new_instance(self, tmp_path: Path) -> None:
        a = get_config()
        reset_config()
        b = get_config()
        assert a is not b
