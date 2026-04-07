"""Unit tests for ghost.tool_catalog."""

from __future__ import annotations

from ghost.tool_catalog import ToolCatalog


def test_default_catalog_contains_all_current_tools() -> None:
    catalog = ToolCatalog.default()
    names = {spec.name for spec in catalog.list_specs()}

    assert {
        "pytorch_create_model",
        "pytorch_train_step",
        "pytorch_evaluate",
        "pytorch_save_checkpoint",
        "pytorch_load_checkpoint",
        "tensorflow_create_model",
        "tensorflow_train_step",
        "tensorflow_evaluate",
        "tensorflow_save_checkpoint",
        "tensorflow_load_checkpoint",
        "get_training_status",
        "list_models",
        "get_system_health",
        "get_model_recommendation",
        "get_training_analysis",
    }.issubset(names)


def test_tensorflow_create_model_schema_uses_tensorflow_input_shape_default() -> None:
    catalog = ToolCatalog.default()
    spec = catalog.get_spec("tensorflow_create_model")

    assert spec is not None
    schema = spec.input_schema()
    assert schema["properties"]["input_shape"]["default"] == [224, 224, 3]


def test_catalog_exposes_handler_metadata() -> None:
    catalog = ToolCatalog.default()
    spec = catalog.get_spec("get_training_analysis")

    assert spec is not None
    assert spec.handler_name == "_handle_get_training_analysis"
    assert "analysis" in spec.tags