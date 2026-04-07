"""Architecture-first red tests for MCP tool catalog separation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest


def _load_module(module_name: str) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        pytest.fail(
            f"Architecture contract missing: expected module '{module_name}'."
        )


def test_tool_catalog_module_exposes_transport_independent_types() -> None:
    module = _load_module("ghost.tool_catalog")

    for symbol in ("ToolSpec", "ToolCatalog"):
        assert hasattr(module, symbol), f"ghost.tool_catalog must expose {symbol}"


def test_tool_spec_captures_schema_and_handler_metadata() -> None:
    module = _load_module("ghost.tool_catalog")
    tool_spec = getattr(module, "ToolSpec", None)
    assert tool_spec is not None, "ghost.tool_catalog must define ToolSpec"

    annotations = getattr(tool_spec, "__annotations__", {})
    expected_fields = {
        "name",
        "description",
        "input_model",
        "handler_name",
        "tags",
    }

    missing_fields = expected_fields.difference(annotations)
    assert not missing_fields, f"ToolSpec is missing fields: {sorted(missing_fields)}"


def test_default_catalog_includes_core_training_tools() -> None:
    module = _load_module("ghost.tool_catalog")
    tool_catalog = getattr(module, "ToolCatalog", None)
    assert tool_catalog is not None, "ghost.tool_catalog must define ToolCatalog"
    assert hasattr(tool_catalog, "default"), "ToolCatalog must define default()"

    catalog = tool_catalog.default()
    assert hasattr(catalog, "list_specs"), "Default tool catalog must support list_specs()"

    specs = list(catalog.list_specs())
    names = {spec.name for spec in specs}
    assert "get_training_status" in names
    assert "get_training_analysis" in names