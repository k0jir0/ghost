"""Unit tests for observability, drift, workflow, auth, and environments."""

from __future__ import annotations

from pathlib import Path

from ghost.auth import AuthService
from ghost.config import GhostConfig, reset_config
from ghost.drift import DriftDetector
from ghost.environment import EnvironmentManager
from ghost.metadata_store import MetadataStore
from ghost.model_registry import ModelRegistry
from ghost.observability import ModelObservability
from ghost.retraining import RetrainingManager
from ghost.run_store import RunStore
from ghost.scheduler import RetrainingPolicy, WorkflowScheduler
from ghost.schemas import ArtifactRecord, ExperimentRunRecord
from ghost.task_queue import TaskQueueStore
from ghost.workflows import WorkflowEngine


def _make_runtime(tmp_path: Path):
    reset_config()
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
        task_queue_file=tmp_path / "TASKS.json",
    )
    config.ensure_directories()
    metadata_store = MetadataStore(config.data_cache_dir / "metadata")
    run_store = RunStore(config=config, metadata_store=metadata_store)
    registry = ModelRegistry(config=config, metadata_store=metadata_store, run_store=run_store)
    task_queue = TaskQueueStore(config.task_queue_file)
    return config, metadata_store, run_store, registry, task_queue


def test_observability_and_drift_report(tmp_path: Path) -> None:
    config, metadata_store, *_ = _make_runtime(tmp_path)
    observability = ModelObservability(config=config, metadata_store=metadata_store)
    drift = DriftDetector(
        config=config,
        metadata_store=metadata_store,
        observability=observability,
    )

    observability.record_prediction(
        "reg-1",
        "model-1",
        latency_ms=12.0,
        batch_size=1,
        success=True,
        inputs=[[0.0, 0.0]],
        predictions=[{"predicted_class": 1, "scores": [0.1, 0.9]}],
    )
    observability.record_prediction(
        "reg-1",
        "model-1",
        latency_ms=18.0,
        batch_size=1,
        success=True,
        inputs=[[10.0, 10.0]],
        predictions=[{"predicted_class": 1, "scores": [0.2, 0.8]}],
    )

    summary = observability.get_summary("reg-1")
    report = drift.get_report("reg-1", mean_shift_threshold=1.0)

    assert summary["request_count"] == 2
    assert report.status == "warning"
    assert report.sample_count == 2


def test_scheduler_queues_retraining_on_drift(tmp_path: Path) -> None:
    config, metadata_store, run_store, registry, task_queue = _make_runtime(tmp_path)
    observability = ModelObservability(config=config, metadata_store=metadata_store)
    drift = DriftDetector(
        config=config,
        metadata_store=metadata_store,
        observability=observability,
    )
    retraining = RetrainingManager(
        task_queue=task_queue,
        model_registry=registry,
        config=config,
        metadata_store=metadata_store,
    )
    workflows = WorkflowEngine(
        retraining_manager=retraining,
        config=config,
        metadata_store=metadata_store,
    )
    scheduler = WorkflowScheduler(
        workflow_engine=workflows,
        drift_detector=drift,
        config=config,
        metadata_store=metadata_store,
    )

    run_store.upsert_run(
        ExperimentRunRecord(
            run_id="run-1",
            experiment_id="exp-1",
            model_id="model-1",
            status="completed",
            backend="pytorch",
            architecture="mlp",
            dataset_id="mnist",
            dataset_version="builtin-v1",
            metrics={"final_accuracy": 0.9, "final_loss": 0.2},
        )
    )
    run_store.upsert_artifact(
        ArtifactRecord(
            artifact_id="run-1__checkpoint",
            artifact_type="checkpoint",
            uri=str(tmp_path / "models" / "model-1.pt"),
            run_id="run-1",
            model_id="model-1",
        )
    )
    registered = registry.register_model("run-1")
    registry.promote_model(registered.registry_id, stage="production")

    observability.record_prediction(
        registered.registry_id,
        "model-1",
        latency_ms=10.0,
        batch_size=1,
        success=True,
        inputs=[[0.0, 0.0]],
        predictions=[{"predicted_class": 1, "scores": [0.1, 0.9]}],
    )
    observability.record_prediction(
        registered.registry_id,
        "model-1",
        latency_ms=10.0,
        batch_size=1,
        success=True,
        inputs=[[10.0, 10.0]],
        predictions=[{"predicted_class": 1, "scores": [0.1, 0.9]}],
    )
    scheduler.upsert_policy(
        RetrainingPolicy(
            policy_id="policy-1",
            registry_id=registered.registry_id,
            mean_shift_threshold=1.0,
            min_samples=2,
        )
    )

    workflows_created = scheduler.evaluate_policies()

    assert len(workflows_created) == 1
    assert task_queue.list_tasks()[0].task_id == f"retrain-{registered.registry_id}"


def test_auth_and_environment_services(tmp_path: Path) -> None:
    config, metadata_store, *_ = _make_runtime(tmp_path)
    auth = AuthService(config=config, metadata_store=metadata_store)
    token, record = auth.issue_token("tester", ["serve:predict"])
    environments = EnvironmentManager(config=config)

    assert auth.authorize(token, "serve:predict") is True
    assert auth.revoke(record.token_id) is True
    assert auth.authorize(token, "serve:predict") is False
    assert [profile.name for profile in environments.list_profiles()] == [
        "dev",
        "staging",
        "production",
    ]