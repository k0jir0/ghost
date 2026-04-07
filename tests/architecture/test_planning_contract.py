"""Architecture-first red tests for Ghost planning components.

These tests intentionally describe the desired planning layer before the
implementation exists.
"""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any

import pytest


def _load_module(module_name: str) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"Architecture contract missing: expected module '{module_name}'."
        )


def test_planning_module_exposes_core_contract_types() -> None:
    module = _load_module("ghost.planning")

    for symbol in ("PlanningRequest", "TrainingPlan", "TrainingPlanner"):
        assert hasattr(module, symbol), f"ghost.planning must expose {symbol}"


def test_training_plan_contract_contains_execution_fields() -> None:
    module = _load_module("ghost.planning")
    training_plan = getattr(module, "TrainingPlan", None)
    assert training_plan is not None, "ghost.planning must define TrainingPlan"

    annotations = getattr(training_plan, "__annotations__", {})
    expected_fields = {
        "task",
        "backend",
        "architecture",
        "num_classes",
        "batch_size",
        "learning_rate",
        "epochs",
        "dataset",
        "optimizer",
        "recommendation_source",
    }

    missing_fields = expected_fields.difference(annotations)
    assert not missing_fields, f"TrainingPlan is missing fields: {sorted(missing_fields)}"


def test_training_plan_can_bridge_into_runtime_training_config() -> None:
    module = _load_module("ghost.planning")
    training_plan = getattr(module, "TrainingPlan", None)
    assert training_plan is not None, "ghost.planning must define TrainingPlan"
    assert hasattr(training_plan, "to_training_config"), (
        "TrainingPlan must provide to_training_config(model_id) so the planning "
        "layer can hand off cleanly to the training pipeline."
    )


def test_training_planner_create_plan_is_async_and_request_driven() -> None:
    module = _load_module("ghost.planning")
    planner = getattr(module, "TrainingPlanner", None)
    assert planner is not None, "ghost.planning must define TrainingPlanner"
    assert hasattr(planner, "create_plan"), "TrainingPlanner must define create_plan"
    assert inspect.iscoroutinefunction(planner.create_plan), (
        "TrainingPlanner.create_plan should be async so recommendation providers "
        "remain non-blocking."
    )

    signature = inspect.signature(planner.create_plan)
    parameter_names = list(signature.parameters)
    assert parameter_names[:2] == ["self", "request"], (
        "TrainingPlanner.create_plan must accept a PlanningRequest to keep planning "
        "inputs explicit."
    )