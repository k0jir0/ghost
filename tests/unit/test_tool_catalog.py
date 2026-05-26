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
        "list_runs",
        "get_run",
        "compare_runs",
        "register_model",
        "list_registered_models",
        "promote_model",
        "reject_model",
        "predict_online",
        "predict_batch",
        "get_model_observability",
        "get_drift_report",
        "list_dataset_manifests",
        "get_dataset_manifest",
        "get_dataset_validation_report",
        "list_training_tasks",
        "create_training_task",
        "update_training_task",
        "delete_training_task",
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


def test_update_training_task_schema_requires_target_and_change_fields() -> None:
    catalog = ToolCatalog.default()
    spec = catalog.get_spec("update_training_task")

    assert spec is not None
    schema = spec.input_schema()
    assert "task_id" in schema["properties"]
    assert "match_text" in schema["properties"]
    assert "completed" in schema["properties"]