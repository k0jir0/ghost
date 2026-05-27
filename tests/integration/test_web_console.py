"""Integration coverage for the Ghost web console surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("starlette")
from fastapi.testclient import TestClient

from ghost.config import GhostConfig, reset_config
from ghost.schemas import ArtifactRecord, ExperimentRunRecord
from ghost.serving import create_serving_app


@pytest.fixture()
def web_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, Any]:
    reset_config()
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DATA_CACHE_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TASK_QUEUE_FILE", str(tmp_path / "TASKS.json"))
    monkeypatch.setenv("AGENT_STATE_FILE", str(tmp_path / "AGENT.json"))

    GhostConfig(
        model_cache_dir=tmp_path / "models",
        data_cache_dir=tmp_path / "data",
        task_queue_file=tmp_path / "TASKS.json",
        agent_state_file=tmp_path / "AGENT.json",
    ).ensure_directories()

    client = TestClient(create_serving_app())
    return client, client.app.state.console


def _seed_completed_run(console: Any, tmp_path: Path) -> None:
    console.run_store.upsert_run(
        ExperimentRunRecord(
            run_id="run-1",
            experiment_id="exp-1",
            model_id="model-1",
            status="completed",
            backend="pytorch",
            architecture="mlp",
            dataset_id="mnist",
            dataset_version="builtin-v1",
            input_shape=[1, 2, 2],
            num_classes=2,
            metrics={"final_accuracy": 0.91, "final_loss": 0.12},
        )
    )
    console.run_store.upsert_artifact(
        ArtifactRecord(
            artifact_id="run-1__checkpoint",
            artifact_type="checkpoint",
            uri=str(tmp_path / "models" / "model-1.pt"),
            run_id="run-1",
            model_id="model-1",
        )
    )


def test_console_shell_serves_browser_app(web_client: tuple[TestClient, Any]) -> None:
    client, _console = web_client

    response = client.get("/")

    assert response.status_code == 200
    assert "Ghost Control Plane" in response.text
    assert "/console-assets/app.js" in response.text


def test_console_task_endpoints_round_trip(
    web_client: tuple[TestClient, Any],
) -> None:
    client, _console = web_client

    create_response = client.post("/api/tasks", json={"text": "Train a console model"})
    assert create_response.status_code == 200
    task_id = create_response.json()["task_id"]

    list_response = client.get("/api/tasks")
    assert list_response.status_code == 200
    assert list_response.json()["tasks"][0]["task_id"] == task_id

    update_response = client.patch(
        f"/api/tasks/{task_id}",
        json={"completed": True},
    )
    assert update_response.status_code == 200
    assert update_response.json()["completed"] is True

    delete_response = client.delete(f"/api/tasks/{task_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["task_id"] == task_id


def test_console_register_promote_and_expose_model_detail(
    web_client: tuple[TestClient, Any],
    tmp_path: Path,
) -> None:
    client, console = web_client
    _seed_completed_run(console, tmp_path)

    register_response = client.post(
        "/api/runs/run-1/register",
        json={"actor": "console-test"},
    )
    assert register_response.status_code == 200
    registry_id = register_response.json()["model"]["registry_id"]

    promote_response = client.post(
        f"/api/models/{registry_id}/promote",
        json={"stage": "production", "approved_by": "console-test"},
    )
    assert promote_response.status_code == 200
    assert promote_response.json()["model"]["stage"] == "production"

    detail_response = client.get(f"/api/models/{registry_id}")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["model"]["registry_id"] == registry_id
    assert detail_payload["model"]["evaluation_status"] == "passed"
    assert detail_payload["run"]["summary"]["run_id"] == "run-1"

    overview_response = client.get("/api/overview")
    assert overview_response.status_code == 200
    overview_payload = overview_response.json()
    assert overview_payload["production_models"][0]["registry_id"] == registry_id
