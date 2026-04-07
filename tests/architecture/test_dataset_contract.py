"""Architecture-first red tests for dataset resolution in Ghost."""

from __future__ import annotations

import inspect
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


def test_dataset_module_exposes_spec_and_resolver_types() -> None:
    module = _load_module("ghost.datasets")

    for symbol in ("DatasetSpec", "DatasetResolver"):
        assert hasattr(module, symbol), f"ghost.datasets must expose {symbol}"


def test_dataset_spec_captures_real_dataset_metadata() -> None:
    module = _load_module("ghost.datasets")
    dataset_spec = getattr(module, "DatasetSpec", None)
    assert dataset_spec is not None, "ghost.datasets must define DatasetSpec"

    annotations = getattr(dataset_spec, "__annotations__", {})
    expected_fields = {
        "dataset_id",
        "task_type",
        "source",
        "input_shape",
        "num_classes",
        "synthetic",
    }

    missing_fields = expected_fields.difference(annotations)
    assert not missing_fields, f"DatasetSpec is missing fields: {sorted(missing_fields)}"


def test_dataset_resolver_requires_explicit_synthetic_opt_in() -> None:
    module = _load_module("ghost.datasets")
    resolver = getattr(module, "DatasetResolver", None)
    assert resolver is not None, "ghost.datasets must define DatasetResolver"
    assert hasattr(resolver, "resolve"), "DatasetResolver must define resolve"

    signature = inspect.signature(resolver.resolve)
    parameters = signature.parameters

    assert "dataset_ref" in parameters, "DatasetResolver.resolve must accept dataset_ref"
    assert "allow_synthetic" in parameters, (
        "DatasetResolver.resolve must require explicit allow_synthetic control."
    )


def test_dataset_resolver_can_list_supported_datasets() -> None:
    module = _load_module("ghost.datasets")
    resolver = getattr(module, "DatasetResolver", None)
    assert resolver is not None, "ghost.datasets must define DatasetResolver"
    assert hasattr(resolver, "list_available"), (
        "DatasetResolver must define list_available() so the agent and MCP layer "
        "can inspect supported datasets."
    )