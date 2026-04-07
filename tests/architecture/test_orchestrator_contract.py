"""Architecture-first red tests for Ghost training orchestration."""

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


def test_orchestration_module_exposes_run_types() -> None:
    module = _load_module("ghost.orchestration")

    for symbol in ("TrainingRunRequest", "TrainingRunRecord", "TrainingOrchestrator"):
        assert hasattr(module, symbol), f"ghost.orchestration must expose {symbol}"


def test_training_run_record_tracks_operational_state() -> None:
    module = _load_module("ghost.orchestration")
    record = getattr(module, "TrainingRunRecord", None)
    assert record is not None, "ghost.orchestration must define TrainingRunRecord"

    annotations = getattr(record, "__annotations__", {})
    expected_fields = {
        "run_id",
        "model_id",
        "status",
        "plan",
        "analysis",
        "events",
    }

    missing_fields = expected_fields.difference(annotations)
    assert not missing_fields, (
        f"TrainingRunRecord is missing fields: {sorted(missing_fields)}"
    )


def test_training_orchestrator_executes_requests_async() -> None:
    module = _load_module("ghost.orchestration")
    orchestrator = getattr(module, "TrainingOrchestrator", None)
    assert orchestrator is not None, "ghost.orchestration must define TrainingOrchestrator"
    assert hasattr(orchestrator, "execute"), "TrainingOrchestrator must define execute"
    assert inspect.iscoroutinefunction(orchestrator.execute), (
        "TrainingOrchestrator.execute should be async because it coordinates "
        "planning, training, and analysis."
    )

    signature = inspect.signature(orchestrator.execute)
    parameter_names = list(signature.parameters)
    assert parameter_names[:2] == ["self", "request"], (
        "TrainingOrchestrator.execute must take a TrainingRunRequest so "
        "orchestration inputs stay explicit."
    )


def test_training_orchestrator_supports_resume_after_interruption() -> None:
    module = _load_module("ghost.orchestration")
    orchestrator = getattr(module, "TrainingOrchestrator", None)
    assert orchestrator is not None, "ghost.orchestration must define TrainingOrchestrator"
    assert hasattr(orchestrator, "resume"), (
        "TrainingOrchestrator must define resume(run_id) so interrupted runs can "
        "be continued from recorded state."
    )