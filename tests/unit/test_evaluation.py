"""Unit tests for ghost.evaluation."""

from __future__ import annotations

from pathlib import Path

from ghost.config import GhostConfig, reset_config
from ghost.evaluation import EvaluationPolicy, ModelEvaluator
from ghost.metadata_store import MetadataStore
from ghost.schemas import ExperimentRunRecord


def test_model_evaluator_flags_regression_against_baseline(tmp_path: Path) -> None:
    reset_config()
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    evaluator = ModelEvaluator(
        config=config,
        metadata_store=MetadataStore(config.data_cache_dir / "metadata"),
    )
    baseline = ExperimentRunRecord(
        run_id="baseline",
        experiment_id="exp-1",
        model_id="model-1",
        status="completed",
        metrics={"final_accuracy": 0.9, "final_loss": 0.2},
    )
    candidate = ExperimentRunRecord(
        run_id="candidate",
        experiment_id="exp-1",
        model_id="model-1",
        status="completed",
        metrics={"final_accuracy": 0.7, "final_loss": 0.8},
    )

    evaluation = evaluator.evaluate_candidate(
        candidate,
        baseline=baseline,
        policy=EvaluationPolicy(min_accuracy=0.75, max_loss=0.7),
    )

    assert evaluation.passed is False
    assert len(evaluation.issues) == 4


def test_model_evaluator_passes_candidate_within_thresholds(tmp_path: Path) -> None:
    reset_config()
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
    )
    config.ensure_directories()
    evaluator = ModelEvaluator(
        config=config,
        metadata_store=MetadataStore(config.data_cache_dir / "metadata"),
    )
    candidate = ExperimentRunRecord(
        run_id="candidate",
        experiment_id="exp-1",
        model_id="model-1",
        status="completed",
        metrics={"final_accuracy": 0.92, "final_loss": 0.18},
    )

    evaluation = evaluator.evaluate_candidate(
        candidate,
        policy=EvaluationPolicy(min_accuracy=0.8, max_loss=0.3),
    )

    assert evaluation.passed is True
    assert evaluation.status == "passed"