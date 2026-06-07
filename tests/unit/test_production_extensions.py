"""Tests for production-facing Ghost extension points."""

from __future__ import annotations

from pathlib import Path

import pytest

from ghost.auth import AuthService
from ghost.config import GhostConfig, reset_config
from ghost.datasets import DatasetSpec
from ghost.deployment import DeploymentManager
from ghost.feature_store import FeatureDefinition, FeatureStore
from ghost.ingestion import LocalMirrorObjectStoreAdapter, ObjectStoreDatasetIngestor
from ghost.metadata_store import MetadataStore, SQLiteMetadataBackend
from ghost.model_registry import ModelRegistry
from ghost.run_store import RunStore
from ghost.schemas import ArtifactRecord, ExperimentRunRecord
from ghost.secrets import (
    EnvironmentSecretProvider,
    LocalDevelopmentSecretProvider,
    SecretReference,
    SecretResolver,
)


def _make_config(tmp_path: Path) -> GhostConfig:
    reset_config()
    config = GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
        task_queue_file=tmp_path / "TASKS.json",
        agent_state_file=tmp_path / "AGENT.json",
    )
    config.ensure_directories()
    return config


def _register_completed_model(
    run_store: RunStore,
    registry: ModelRegistry,
    tmp_path: Path,
    *,
    run_id: str,
    accuracy: float,
) -> str:
    model_path = tmp_path / "models" / f"{run_id}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("checkpoint", encoding="utf-8")
    run_store.upsert_run(
        ExperimentRunRecord(
            run_id=run_id,
            experiment_id=f"exp-{run_id}",
            model_id="risk-model",
            status="completed",
            backend="pytorch",
            architecture="mlp",
            dataset_id="risk-dataset",
            dataset_version="v1",
            metrics={"final_accuracy": accuracy, "final_loss": 0.1},
        )
    )
    run_store.upsert_artifact(
        ArtifactRecord(
            artifact_id=f"{run_id}__checkpoint",
            artifact_type="checkpoint",
            uri=str(model_path),
            run_id=run_id,
            model_id="risk-model",
        )
    )
    return registry.register_model(run_id).registry_id


def test_sqlite_metadata_backend_persists_records(tmp_path: Path) -> None:
    backend = SQLiteMetadataBackend(tmp_path / "metadata.sqlite")
    store = MetadataStore(backend=backend)

    store.save_record("runs", "run-1", {"run_id": "run-1", "status": "queued"})
    store.save_record("runs", "run-2", {"run_id": "run-2", "status": "done"})

    assert store.load_record("runs", "run-1") == {
        "run_id": "run-1",
        "status": "queued",
    }
    assert {record["run_id"] for record in store.list_records("runs")} == {
        "run-1",
        "run-2",
    }


def test_object_store_ingestion_uses_local_mirror_adapter(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    mirrored = tmp_path / "mirror" / "s3" / "ghost-bucket" / "datasets" / "risk.npz"
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    mirrored.write_text("dataset", encoding="utf-8")
    spec = DatasetSpec(
        dataset_id="risk",
        task_type="tabular-classification",
        source="external",
        input_shape=(4,),
        num_classes=2,
        synthetic=False,
        metadata={
            "source_uri": "s3://ghost-bucket/datasets/risk.npz",
            "dataset_version": "v1",
        },
    )
    ingestor = ObjectStoreDatasetIngestor(
        config=config,
        object_adapters=[LocalMirrorObjectStoreAdapter(tmp_path / "mirror")],
    )

    artifact = ingestor.ingest(spec, "s3://ghost-bucket/datasets/risk.npz")

    assert artifact.local_path.read_text(encoding="utf-8") == "dataset"
    assert artifact.local_path.is_relative_to(config.data_cache_dir)


def test_feature_store_materializes_and_reads_point_in_time_features(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    store = MetadataStore(config.data_cache_dir / "metadata")
    features = FeatureStore(config=config, metadata_store=store)
    features.register_definition(
        FeatureDefinition(
            feature_name="risk_score",
            version="v1",
            value_type="float",
            entities=["account"],
        )
    )

    materialization = features.materialize_offline(
        "risk-dataset",
        [
            {
                "entity_id": "acct-1",
                "event_timestamp": "2026-01-01T00:00:00+00:00",
                "risk_score": 0.2,
            },
            {
                "entity_id": "acct-1",
                "event_timestamp": "2026-02-01T00:00:00+00:00",
                "risk_score": 0.8,
            },
        ],
    )

    assert materialization.feature_names == ["risk_score"]
    assert features.get_online_features(
        "acct-1",
        as_of="2026-01-15T00:00:00+00:00",
    ) == {"risk_score": 0.2}
    assert features.get_online_features("acct-1") == {"risk_score": 0.8}


def test_deployment_manager_activates_and_rolls_back_models(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    metadata_store = MetadataStore(config.data_cache_dir / "metadata")
    run_store = RunStore(config=config, metadata_store=metadata_store)
    registry = ModelRegistry(
        config=config,
        metadata_store=metadata_store,
        run_store=run_store,
    )
    deployments = DeploymentManager(
        config=config,
        metadata_store=metadata_store,
        model_registry=registry,
    )
    first_registry_id = _register_completed_model(
        run_store,
        registry,
        tmp_path,
        run_id="run-1",
        accuracy=0.91,
    )
    registry.promote_model(first_registry_id, stage="production")
    first_deployment = deployments.activate_deployment(
        deployments.create_deployment(
            first_registry_id,
            environment="production",
        ).deployment_id
    )

    second_registry_id = _register_completed_model(
        run_store,
        registry,
        tmp_path,
        run_id="run-2",
        accuracy=0.93,
    )
    registry.promote_model(second_registry_id, stage="production")
    deployments.activate_deployment(
        deployments.create_deployment(
            second_registry_id,
            environment="production",
        ).deployment_id
    )

    rollback = deployments.rollback(environment="production", model_id="risk-model")

    assert first_deployment.registry_id == first_registry_id
    assert rollback.registry_id == first_registry_id
    assert deployments.current_active("production", "risk-model") == rollback


def test_secret_resolver_and_expiring_auth_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    metadata_store = MetadataStore(config.data_cache_dir / "metadata")
    local_provider = LocalDevelopmentSecretProvider(
        config=config,
        metadata_store=metadata_store,
    )
    local_provider.set_secret("GHOST_LOCAL_TOKEN", "local-secret")
    monkeypatch.setenv("GHOST_ENV_TOKEN", "env-secret")
    resolver = SecretResolver(
        providers=[EnvironmentSecretProvider(), local_provider],
    )
    auth = AuthService(config=config, metadata_store=metadata_store)

    wildcard_token, _ = auth.issue_token(
        "service-account",
        ["*"],
        roles=["inference-writer"],
    )
    expired_token, _ = auth.issue_token("analyst", ["serve:predict"], ttl_seconds=-1)

    assert resolver.resolve("GHOST_ENV_TOKEN") == "env-secret"
    assert (
        resolver.resolve(SecretReference(name="GHOST_LOCAL_TOKEN", provider="local"))
        == "local-secret"
    )
    assert auth.authorize(wildcard_token, "deploy:write") is True
    assert auth.authorize(expired_token, "serve:predict") is False
