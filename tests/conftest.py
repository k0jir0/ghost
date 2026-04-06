"""Shared pytest fixtures for Ghost tests.

All ML-heavy imports (torch, tensorflow) are mocked so the test suite runs in
seconds without a GPU, CUDA, or Ollama installation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Heavy-dependency stubs (installed before any ghost.* imports)
# ---------------------------------------------------------------------------

def _make_torch_stub() -> MagicMock:
    """Return a minimal torch stub sufficient for unit tests."""
    torch = MagicMock(name="torch")
    torch.cuda.is_available.return_value = False
    torch.device.return_value = "cpu"

    # nn sub-module
    torch.nn = MagicMock(name="torch.nn")
    torch.nn.Module = object  # base class used by SimpleMLP / ResNetSimple
    torch.nn.Linear = MagicMock()
    torch.nn.ReLU = MagicMock()
    torch.nn.Conv2d = MagicMock()
    torch.nn.BatchNorm2d = MagicMock()
    torch.nn.MaxPool2d = MagicMock()
    torch.nn.AdaptiveAvgPool2d = MagicMock()
    torch.nn.Sequential = MagicMock(return_value=MagicMock())
    torch.nn.CrossEntropyLoss = MagicMock(return_value=MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.5))))

    # optim sub-module
    torch.optim = MagicMock(name="torch.optim")
    torch.optim.Adam = MagicMock(return_value=MagicMock())

    # Tensor helpers
    torch.randn = MagicMock(return_value=MagicMock())
    torch.randint = MagicMock(return_value=MagicMock())
    torch.flatten = MagicMock(return_value=MagicMock())
    torch.save = MagicMock()
    torch.load = MagicMock(return_value={"model_state_dict": {}, "model_id": "test"})

    return torch


def _make_tensorflow_stub() -> MagicMock:
    tf = MagicMock(name="tensorflow")
    tf.keras = MagicMock()
    tf.keras.models.Sequential = MagicMock()
    tf.keras.layers.Dense = MagicMock()
    tf.keras.layers.Conv2D = MagicMock()
    tf.keras.layers.Flatten = MagicMock()
    tf.keras.optimizers.Adam = MagicMock()
    tf.keras.losses.SparseCategoricalCrossentropy = MagicMock()
    return tf


def _make_ollama_stub() -> MagicMock:
    stub = MagicMock(name="ollama")
    stub.chat.return_value = {"message": {"content": '{"architecture": "mlp"}'}}
    stub.ps.return_value = {}
    return stub


def _make_mcp_stub() -> MagicMock:
    mcp = MagicMock(name="mcp")
    mcp.server = MagicMock()
    mcp.server.Server = MagicMock()
    mcp.server.stdio = MagicMock()
    mcp.server.stdio.stdio_server = MagicMock()
    mcp.types = MagicMock()
    mcp.types.Tool = MagicMock()
    mcp.types.TextContent = MagicMock(side_effect=lambda **kw: kw)
    mcp.types.CallToolResult = MagicMock(side_effect=lambda **kw: kw)
    mcp.types.ListToolsResult = MagicMock(side_effect=lambda **kw: kw)
    return mcp


# Inject stubs before any test (or production) module imports them.
for _name, _factory in [
    ("torch", _make_torch_stub),
    ("torch.nn", lambda: _make_torch_stub().nn),
    ("torch.optim", lambda: _make_torch_stub().optim),
    ("tensorflow", _make_tensorflow_stub),
    ("ollama", _make_ollama_stub),
    ("mcp", _make_mcp_stub),
    ("mcp.server", lambda: _make_mcp_stub().server),
    ("mcp.server.stdio", lambda: _make_mcp_stub().server.stdio),
    ("mcp.types", lambda: _make_mcp_stub().types),
    ("structlog", MagicMock),
]:
    if _name not in sys.modules:
        sys.modules[_name] = _factory()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """Isolated data directory for context storage."""
    d = tmp_path / "data" / "contexts"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def tmp_models_dir(tmp_path: Path) -> Path:
    """Isolated model checkpoint directory."""
    d = tmp_path / "models"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def ghost_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return a fresh GhostConfig backed by a temp directory."""
    from ghost.config import reset_config, GhostConfig

    reset_config()
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DATA_CACHE_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    cfg = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    cfg.ensure_directories()
    return cfg


@pytest.fixture()
def context_manager(tmp_data_dir: Path) -> Any:
    """Return a ContextManager wired to a temp directory."""
    from ghost.context import ContextManager

    return ContextManager(storage_path=tmp_data_dir)


@pytest.fixture()
def tasks_file(tmp_path: Path) -> Path:
    """Write a minimal TASKS.md and return its path."""
    content = """\
# Ghost Tasks

## Queue

- [ ] Train MLP on CIFAR-10
- [x] Setup environment
- [ ] Run TensorFlow benchmark
"""
    f = tmp_path / "TASKS.md"
    f.write_text(content)
    return f


@pytest.fixture()
def agent_memory_file(tmp_path: Path) -> Path:
    return tmp_path / "AGENT.md"
