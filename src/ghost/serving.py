"""Optional HTTP serving surface for Ghost inference and the web console."""

# mypy: disable-error-code=untyped-decorator

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ghost.alerts import AlertManager
from ghost.audit import AuditLogger
from ghost.config import get_config
from ghost.context import BackendType, ContextManager
from ghost.data_validation import DatasetValidator
from ghost.dataset_registry import DatasetRegistry
from ghost.drift import DriftDetector
from ghost.environment import EnvironmentManager
from ghost.evaluation import EvaluationPolicy, ModelEvaluator
from ghost.health_monitor import HealthMonitor
from ghost.inference import InferenceService
from ghost.metadata_store import MetadataStore
from ghost.model_registry import ModelRegistry
from ghost.observability import ModelObservability
from ghost.orchestration import TrainingOrchestrator, TrainingRunRequest
from ghost.prediction_schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    OnlinePredictionRequest,
    OnlinePredictionResponse,
)
from ghost.pytorch_ops import PyTorchOps
from ghost.retraining import RetrainingManager
from ghost.run_store import RunStore
from ghost.task_queue import TaskQueueStore
from ghost.tensorflow_ops import TensorFlowOps
from ghost.training import TrainingPipeline
from ghost.web_console import WebConsoleService

_WEBUI_DIR = Path(__file__).with_name("webui")
_INDEX_PATH = _WEBUI_DIR / "index.html"


class LaunchRunPayload(BaseModel):
    task: str = Field(min_length=1)
    dataset_ref: str = ""
    dataset: str = ""
    model_name: str = ""
    allow_synthetic: bool = False
    recommendations: dict[str, Any] | None = None


class RegisterRunPayload(BaseModel):
    actor: str = "web-console"
    baseline_registry_id: str | None = None
    min_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    max_loss: float | None = Field(default=None, ge=0.0)
    max_accuracy_drop: float = Field(default=0.05, ge=0.0, le=1.0)
    max_loss_increase: float = Field(default=0.5, ge=0.0)


class PromoteModelPayload(BaseModel):
    stage: Literal["staging", "production", "archived"]
    approved_by: str = "web-console"
    alias: str | None = None


class RejectModelPayload(BaseModel):
    reason: str = Field(min_length=1)
    rejected_by: str = "web-console"


class RetrainModelPayload(BaseModel):
    reason: str = Field(min_length=1)


class PredictionPayload(BaseModel):
    features: list[Any] = Field(min_length=1)


class CreateTaskPayload(BaseModel):
    text: str = Field(min_length=1)
    task_id: str | None = None
    metadata: dict[str, Any] | None = None


class UpdateTaskPayload(BaseModel):
    text: str | None = Field(default=None, min_length=1)
    completed: bool | None = None


def _build_console_service(
    inference_service: InferenceService | None = None,
) -> WebConsoleService:
    config = get_config()
    context_manager = (
        inference_service.context_manager if inference_service is not None else None
    ) or ContextManager()
    metadata_store = MetadataStore(config.data_cache_dir / "metadata")
    run_store = RunStore(config=config, metadata_store=metadata_store)
    evaluator = ModelEvaluator(config=config, metadata_store=metadata_store)
    audit_logger = AuditLogger(config=config, metadata_store=metadata_store)
    model_registry = (
        inference_service.model_registry if inference_service is not None else None
    ) or ModelRegistry(
        config=config,
        metadata_store=metadata_store,
        run_store=run_store,
        evaluator=evaluator,
        audit_logger=audit_logger,
    )
    observability = (
        inference_service.observability if inference_service is not None else None
    ) or ModelObservability(
        config=config,
        metadata_store=metadata_store,
    )
    drift_detector = DriftDetector(
        config=config,
        metadata_store=metadata_store,
        observability=observability,
    )
    alert_manager = AlertManager(
        config=config,
        metadata_store=metadata_store,
    )
    health_monitor = HealthMonitor(config=config)
    backend_ops = (
        getattr(inference_service, "_backend_ops", None)
        if inference_service is not None
        else None
    ) or {
        BackendType.PYTORCH: PyTorchOps(context_manager),
        BackendType.TENSORFLOW: TensorFlowOps(context_manager),
    }
    training_pipeline = TrainingPipeline(
        context_manager=context_manager,
        health_monitor=health_monitor,
        backend_ops=backend_ops,
    )
    task_queue = TaskQueueStore(config.task_queue_file)
    dataset_registry = DatasetRegistry(config=config, metadata_store=metadata_store)
    dataset_validator = DatasetValidator(config=config, metadata_store=metadata_store)
    orchestrator = TrainingOrchestrator(
        config=config,
        context_manager=context_manager,
        training_pipeline=training_pipeline,
        backend_ops=backend_ops,
        metadata_store=metadata_store,
        run_store=run_store,
    )
    active_inference_service = inference_service or InferenceService(
        config=config,
        context_manager=context_manager,
        model_registry=model_registry,
        backend_ops=backend_ops,
        observability=observability,
    )
    retraining_manager = RetrainingManager(
        task_queue=task_queue,
        model_registry=model_registry,
        config=config,
        metadata_store=metadata_store,
    )
    environment_manager = EnvironmentManager(config=config)

    return WebConsoleService(
        config=config,
        context_manager=context_manager,
        metadata_store=metadata_store,
        run_store=run_store,
        model_registry=model_registry,
        evaluator=evaluator,
        observability=observability,
        drift_detector=drift_detector,
        alert_manager=alert_manager,
        task_queue=task_queue,
        dataset_registry=dataset_registry,
        dataset_validator=dataset_validator,
        health_monitor=health_monitor,
        inference_service=active_inference_service,
        orchestrator=orchestrator,
        retraining_manager=retraining_manager,
        audit_logger=audit_logger,
        environment_manager=environment_manager,
    )


def create_serving_app(inference_service: InferenceService | None = None) -> Any:
    """Create a FastAPI app when FastAPI is installed."""

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - optional dependency surface
        raise RuntimeError(
            "FastAPI is not installed. Install it to run the Ghost serving API."
        ) from exc

    console = _build_console_service(inference_service=inference_service)
    app = FastAPI(title="Ghost Control Plane", version="1.0")
    app.state.console = console

    if _WEBUI_DIR.exists():
        app.mount(
            "/console-assets",
            StaticFiles(directory=_WEBUI_DIR),
            name="console-assets",
        )

    def _shell() -> FileResponse:
        return FileResponse(_INDEX_PATH)

    def _http_error(exc: Exception) -> HTTPException:
        if isinstance(exc, KeyError):
            return HTTPException(status_code=404, detail=str(exc))
        if isinstance(exc, ValueError):
            return HTTPException(status_code=400, detail=str(exc))
        return HTTPException(status_code=500, detail=str(exc))

    @app.get("/", response_class=HTMLResponse)
    async def console_root() -> FileResponse:
        return _shell()

    @app.get("/runs", response_class=HTMLResponse)
    async def console_runs() -> FileResponse:
        return _shell()

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def console_run_detail(run_id: str) -> FileResponse:
        _ = run_id
        return _shell()

    @app.get("/registry", response_class=HTMLResponse)
    async def console_registry() -> FileResponse:
        return _shell()

    @app.get("/models/{registry_id}", response_class=HTMLResponse)
    async def console_model_detail(registry_id: str) -> FileResponse:
        _ = registry_id
        return _shell()

    @app.get("/tasks", response_class=HTMLResponse)
    async def console_tasks() -> FileResponse:
        return _shell()

    @app.get("/datasets", response_class=HTMLResponse)
    async def console_datasets() -> FileResponse:
        return _shell()

    @app.get("/playground", response_class=HTMLResponse)
    async def console_playground() -> FileResponse:
        return _shell()

    @app.get("/api/overview")
    async def api_overview() -> dict[str, Any]:
        return console.get_overview()

    @app.get("/api/agent")
    async def api_agent() -> dict[str, Any]:
        return console.get_agent_status()

    @app.get("/api/runs")
    async def api_runs() -> dict[str, Any]:
        runs = console.list_runs()
        return {"runs": runs, "count": len(runs)}

    @app.get("/api/runs/{run_id}")
    async def api_run_detail(run_id: str) -> dict[str, Any]:
        detail = console.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return detail

    @app.post("/api/runs")
    async def api_launch_run(payload: LaunchRunPayload) -> dict[str, Any]:
        try:
            return console.launch_run(
                TrainingRunRequest(
                    task=payload.task,
                    dataset_ref=payload.dataset_ref,
                    dataset=payload.dataset,
                    model_name=payload.model_name,
                    allow_synthetic=payload.allow_synthetic,
                    recommendations=payload.recommendations,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _http_error(exc) from exc

    @app.post("/api/runs/{run_id}/resume")
    async def api_resume_run(run_id: str) -> dict[str, Any]:
        try:
            return console.resume_run(run_id)
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _http_error(exc) from exc

    @app.post("/api/runs/{run_id}/cancel")
    async def api_cancel_run(run_id: str) -> dict[str, Any]:
        try:
            return console.cancel_run(run_id)
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _http_error(exc) from exc

    @app.post("/api/runs/{run_id}/register")
    async def api_register_run(
        run_id: str,
        payload: RegisterRunPayload,
    ) -> dict[str, Any]:
        try:
            return console.register_run(
                run_id,
                actor=payload.actor,
                baseline_registry_id=payload.baseline_registry_id,
                policy=EvaluationPolicy(
                    min_accuracy=payload.min_accuracy,
                    max_loss=payload.max_loss,
                    max_accuracy_drop=payload.max_accuracy_drop,
                    max_loss_increase=payload.max_loss_increase,
                ),
            )
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _http_error(exc) from exc

    @app.get("/api/models")
    async def api_models() -> dict[str, Any]:
        models = console.list_models()
        return {"models": models, "count": len(models)}

    @app.get("/api/models/{registry_id}")
    async def api_model_detail(registry_id: str) -> dict[str, Any]:
        try:
            return console.get_model_detail(registry_id)
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _http_error(exc) from exc

    @app.post("/api/models/{registry_id}/promote")
    async def api_promote_model(
        registry_id: str,
        payload: PromoteModelPayload,
    ) -> dict[str, Any]:
        try:
            return console.promote_model(
                registry_id,
                stage=payload.stage,
                actor=payload.approved_by,
                alias=payload.alias,
            )
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _http_error(exc) from exc

    @app.post("/api/models/{registry_id}/reject")
    async def api_reject_model(
        registry_id: str,
        payload: RejectModelPayload,
    ) -> dict[str, Any]:
        try:
            return console.reject_model(
                registry_id,
                reason=payload.reason,
                actor=payload.rejected_by,
            )
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _http_error(exc) from exc

    @app.post("/api/models/{registry_id}/retrain")
    async def api_retrain_model(
        registry_id: str,
        payload: RetrainModelPayload,
    ) -> dict[str, Any]:
        try:
            return console.retrain_model(registry_id, reason=payload.reason)
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _http_error(exc) from exc

    @app.post("/api/models/{registry_id}/predict")
    async def api_predict_model(
        registry_id: str,
        payload: PredictionPayload,
    ) -> dict[str, Any]:
        try:
            return await console.predict_online(registry_id, payload.features)
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _http_error(exc) from exc

    @app.get("/api/tasks")
    async def api_tasks(include_completed: bool = True) -> dict[str, Any]:
        return console.list_tasks(include_completed=include_completed)

    @app.post("/api/tasks")
    async def api_create_task(payload: CreateTaskPayload) -> dict[str, Any]:
        try:
            return console.create_task(
                text=payload.text,
                task_id=payload.task_id,
                metadata=payload.metadata,
            )
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _http_error(exc) from exc

    @app.patch("/api/tasks/{task_id}")
    async def api_update_task(
        task_id: str,
        payload: UpdateTaskPayload,
    ) -> dict[str, Any]:
        try:
            return console.update_task(
                task_id,
                text=payload.text,
                completed=payload.completed,
            )
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _http_error(exc) from exc

    @app.delete("/api/tasks/{task_id}")
    async def api_delete_task(task_id: str) -> dict[str, Any]:
        try:
            return console.delete_task(task_id)
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _http_error(exc) from exc

    @app.get("/api/datasets")
    async def api_datasets() -> dict[str, Any]:
        return console.list_datasets()

    @app.get("/api/datasets/{dataset_id}")
    async def api_dataset_detail(
        dataset_id: str,
        version: str = "builtin-v1",
    ) -> dict[str, Any]:
        try:
            return console.get_dataset_detail(dataset_id, version)
        except Exception as exc:  # pragma: no cover - exercised via tests
            raise _http_error(exc) from exc

    @app.get("/api/events")
    async def api_events() -> StreamingResponse:
        async def _stream() -> Any:
            while True:
                yield f"data: {json.dumps(console.events_snapshot())}\n\n"
                await asyncio.sleep(4)

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    async def predict_online(
        registry_id: str,
        request: OnlinePredictionRequest,
    ) -> OnlinePredictionResponse:
        payload = await console.inference_service.predict_online(
            registry_id, request.features
        )
        return OnlinePredictionResponse.model_validate(payload)

    async def predict_batch(
        registry_id: str,
        request: BatchPredictionRequest,
    ) -> BatchPredictionResponse:
        payload = await console.inference_service.predict_batch(
            registry_id, request.inputs
        )
        return BatchPredictionResponse.model_validate(payload)

    app.add_api_route(
        "/v1/models/{registry_id}:predict",
        predict_online,
        methods=["POST"],
        response_model=OnlinePredictionResponse,
    )
    app.add_api_route(
        "/v1/models/{registry_id}:predict-batch",
        predict_batch,
        methods=["POST"],
        response_model=BatchPredictionResponse,
    )

    return app
