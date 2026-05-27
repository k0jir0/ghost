"""Shared control-plane view models and actions for the Ghost web console."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from ghost.alerts import AlertManager
from ghost.audit import AuditLogger
from ghost.config import GhostConfig, get_config
from ghost.context import ContextManager, ModelContext
from ghost.data_validation import DatasetValidator
from ghost.dataset_registry import DatasetRegistry
from ghost.drift import DriftDetector
from ghost.environment import EnvironmentManager
from ghost.evaluation import EvaluationPolicy, ModelEvaluator
from ghost.health_monitor import HealthMonitor
from ghost.inference import InferenceService
from ghost.logging import get_logger
from ghost.metadata_store import MetadataStore
from ghost.model_registry import ModelRegistry, RegistryStage
from ghost.observability import ModelObservability
from ghost.orchestration import (
    TrainingOrchestrator,
    TrainingRunRecord,
    TrainingRunRequest,
)
from ghost.retraining import RetrainingManager
from ghost.run_store import RunStore
from ghost.schemas import (
    DatasetManifest,
    ExperimentRunRecord,
    ModelRegistryRecord,
)
from ghost.task_queue import TaskQueueStore

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_sort(value: str | None) -> str:
    return value or ""


class WebConsoleService:
    """Assemble Ghost browser-facing views and actions."""

    def __init__(
        self,
        *,
        config: GhostConfig | None = None,
        context_manager: ContextManager | None = None,
        metadata_store: MetadataStore | None = None,
        run_store: RunStore | None = None,
        model_registry: ModelRegistry | None = None,
        evaluator: ModelEvaluator | None = None,
        observability: ModelObservability | None = None,
        drift_detector: DriftDetector | None = None,
        alert_manager: AlertManager | None = None,
        task_queue: TaskQueueStore | None = None,
        dataset_registry: DatasetRegistry | None = None,
        dataset_validator: DatasetValidator | None = None,
        health_monitor: HealthMonitor | None = None,
        inference_service: InferenceService | None = None,
        orchestrator: TrainingOrchestrator | None = None,
        retraining_manager: RetrainingManager | None = None,
        audit_logger: AuditLogger | None = None,
        environment_manager: EnvironmentManager | None = None,
    ):
        self.config = config or get_config()
        self.context_manager = context_manager or ContextManager()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )
        self.run_store = run_store or RunStore(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.evaluator = evaluator or ModelEvaluator(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.audit_logger = audit_logger or AuditLogger(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.model_registry = model_registry or ModelRegistry(
            config=self.config,
            metadata_store=self.metadata_store,
            run_store=self.run_store,
            evaluator=self.evaluator,
            audit_logger=self.audit_logger,
        )
        self.observability = observability or ModelObservability(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.drift_detector = drift_detector or DriftDetector(
            config=self.config,
            metadata_store=self.metadata_store,
            observability=self.observability,
        )
        self.alert_manager = alert_manager or AlertManager(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.task_queue = task_queue or TaskQueueStore(self.config.task_queue_file)
        self.dataset_registry = dataset_registry or DatasetRegistry(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.dataset_validator = dataset_validator or DatasetValidator(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.health_monitor = health_monitor or HealthMonitor(config=self.config)
        self.inference_service = inference_service or InferenceService(
            config=self.config,
            context_manager=self.context_manager,
            model_registry=self.model_registry,
            observability=self.observability,
        )
        self.orchestrator = orchestrator or TrainingOrchestrator(
            config=self.config,
            context_manager=self.context_manager,
            metadata_store=self.metadata_store,
            run_store=self.run_store,
        )
        self.retraining_manager = retraining_manager or RetrainingManager(
            task_queue=self.task_queue,
            model_registry=self.model_registry,
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.environment_manager = environment_manager or EnvironmentManager(
            config=self.config
        )
        self._active_run_tasks: dict[str, asyncio.Task[TrainingRunRecord]] = {}

    def launch_run(self, request: TrainingRunRequest) -> dict[str, Any]:
        """Queue a run record and execute it in the background."""
        record = self.orchestrator.prepare_run(request)
        task = asyncio.create_task(self.orchestrator.execute_prepared(record.run_id))
        self._track_run_task(record.run_id, task)
        detail = self.get_run_detail(record.run_id)
        if detail is None:
            raise KeyError(f"Unknown run id: {record.run_id}")
        return detail

    def resume_run(self, run_id: str) -> dict[str, Any]:
        """Resume a recorded run in the background."""
        record = self.orchestrator.get_run(run_id)
        if record is None:
            raise KeyError(f"Unknown run id: {run_id}")
        if record.request is None:
            raise ValueError(f"Run {run_id} does not contain enough state to resume")
        if run_id in self._active_run_tasks:
            raise ValueError(f"Run {run_id} is already active")

        task = asyncio.create_task(self.orchestrator.resume(run_id))
        self._track_run_task(run_id, task)
        detail = self.get_run_detail(run_id)
        if detail is None:
            raise KeyError(f"Unknown run id: {run_id}")
        return detail

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        """Signal a running orchestrated model to stop."""
        detail = self.get_run_detail(run_id)
        if detail is None:
            raise KeyError(f"Unknown run id: {run_id}")

        model_id = str(detail["summary"].get("model_id", ""))
        cancelled = self.orchestrator.training_pipeline.stop_training(model_id)
        if not cancelled:
            raise ValueError(f"Run {run_id} is not currently training")
        return {
            "status": "accepted",
            "run_id": run_id,
            "model_id": model_id,
            "message": "Stop signal sent to training pipeline.",
        }

    def register_run(
        self,
        run_id: str,
        *,
        actor: str = "web-console",
        baseline_registry_id: str | None = None,
        policy: EvaluationPolicy | None = None,
    ) -> dict[str, Any]:
        record = self.model_registry.register_model(
            run_id,
            actor=actor,
            baseline_registry_id=baseline_registry_id,
            policy=policy,
        )
        return self.get_model_detail(record.registry_id)

    def promote_model(
        self,
        registry_id: str,
        *,
        stage: RegistryStage,
        actor: str = "web-console",
        alias: str | None = None,
    ) -> dict[str, Any]:
        record = self.model_registry.promote_model(
            registry_id,
            stage=stage,
            actor=actor,
            alias=alias,
        )
        return self.get_model_detail(record.registry_id)

    def reject_model(
        self,
        registry_id: str,
        *,
        reason: str,
        actor: str = "web-console",
    ) -> dict[str, Any]:
        record = self.model_registry.reject_model(
            registry_id,
            reason=reason,
            actor=actor,
        )
        return self.get_model_detail(record.registry_id)

    def retrain_model(self, registry_id: str, *, reason: str) -> dict[str, Any]:
        request = self.retraining_manager.queue_retraining(registry_id, reason=reason)
        return request.to_dict()

    async def predict_online(
        self, registry_id: str, features: list[Any]
    ) -> dict[str, Any]:
        payload = await self.inference_service.predict_online(registry_id, features)
        observability = self.observability.get_summary(registry_id)
        return {
            "prediction": payload,
            "observability": observability,
        }

    def get_overview(self) -> dict[str, Any]:
        runs = self.list_runs()
        models = self.list_models()
        tasks = self.list_tasks(include_completed=True)["tasks"]
        health = self.health_monitor.get_health_report()
        alerts = self._list_alerts()
        production_models = [
            model for model in models if model.get("stage") == "production"
        ]
        contexts = [ctx.to_dict() for ctx in self.context_manager.list_contexts()]
        recent_runs = sorted(
            runs,
            key=lambda run: _iso_sort(str(run.get("created_at"))),
            reverse=True,
        )[:6]
        active_runs = [
            run
            for run in recent_runs
            if run.get("status") in {"queued", "planned", "running", "resumed"}
            or run.get("is_training")
        ]
        manifests = self.list_datasets()["datasets"]
        workflows = self._list_workflows()

        return {
            "generated_at": _utc_now_iso(),
            "config": {
                "training_backend": self.config.training_backend,
                "gpu_enabled": self.config.gpu_enabled,
                "ollama_model": self.config.ollama_model,
                "allow_synthetic_data": self.config.allow_synthetic_data,
                "data_cache_dir": str(self.config.data_cache_dir),
                "model_cache_dir": str(self.config.model_cache_dir),
            },
            "health": health,
            "agent": self.get_agent_status(),
            "environments": [
                profile.to_dict()
                for profile in self.environment_manager.list_profiles()
            ],
            "run_counts": self._count_by_key(runs, "status"),
            "model_counts": self._count_by_key(models, "stage"),
            "task_counts": {
                "total": len(tasks),
                "pending": sum(1 for task in tasks if not task.get("completed")),
                "completed": sum(1 for task in tasks if task.get("completed")),
            },
            "dataset_count": len(manifests),
            "context_count": len(contexts),
            "recent_runs": recent_runs,
            "active_runs": active_runs,
            "production_models": production_models,
            "recent_alerts": alerts[:6],
            "recent_workflows": workflows[:6],
        }

    def events_snapshot(self) -> dict[str, Any]:
        runs = self.list_runs()
        tasks = self.list_tasks(include_completed=True)["tasks"]
        health = self.health_monitor.get_health_report()
        active_runs = [
            run
            for run in runs
            if run.get("status") in {"queued", "planned", "running", "resumed"}
            or run.get("is_training")
        ]
        return {
            "generated_at": _utc_now_iso(),
            "health": {
                "status": health.get("status"),
                "checked_at": health.get("checked_at"),
            },
            "active_runs": active_runs[:8],
            "run_counts": self._count_by_key(runs, "status"),
            "task_counts": {
                "pending": sum(1 for task in tasks if not task.get("completed")),
                "completed": sum(1 for task in tasks if task.get("completed")),
            },
        }

    def list_runs(self) -> list[dict[str, Any]]:
        run_ids = {record.run_id for record in self.orchestrator.list_runs()} | {
            record.run_id for record in self.run_store.list_runs()
        }
        summaries = [
            summary
            for run_id in run_ids
            if (summary := self._build_run_summary(run_id)) is not None
        ]
        return sorted(
            summaries,
            key=lambda run: _iso_sort(str(run.get("created_at"))),
            reverse=True,
        )

    def get_run_detail(self, run_id: str) -> dict[str, Any] | None:
        orchestration, experiment, context = self._load_run_sources(run_id)
        if orchestration is None and experiment is None:
            return None

        summary = self._build_run_summary(run_id)
        if summary is None:
            return None

        result = orchestration.result if orchestration is not None else None
        metrics_history = (
            [asdict(metric) for metric in result.metrics_history]
            if result is not None
            else self._metrics_history(context)
        )
        artifacts = [
            artifact.to_dict()
            for artifact in self.run_store.list_artifacts(run_id=run_id)
        ]
        registry_entries = [
            self._registry_record_view(record)
            for record in self.model_registry.list_models()
            if record.run_id == run_id
        ]

        return {
            "summary": summary,
            "request": (
                orchestration.request.to_dict()
                if orchestration is not None and orchestration.request is not None
                else experiment.request
                if experiment is not None
                else {}
            ),
            "plan": (
                orchestration.plan.to_dict()
                if orchestration is not None and orchestration.plan is not None
                else experiment.plan
                if experiment is not None
                else {}
            ),
            "analysis": (
                orchestration.analysis
                if orchestration is not None and orchestration.analysis is not None
                else experiment.analysis
                if experiment is not None
                else {}
            ),
            "dataset": (
                asdict(orchestration.dataset)
                if orchestration is not None and orchestration.dataset is not None
                else {
                    "dataset_id": experiment.dataset_id,
                    "dataset_version": experiment.dataset_version,
                }
                if experiment is not None
                else {}
            ),
            "events": orchestration.events if orchestration is not None else [],
            "metrics_history": metrics_history,
            "artifacts": artifacts,
            "registry_entries": registry_entries,
            "runtime_context": context.to_dict() if context is not None else None,
            "experiment_record": experiment.to_dict()
            if experiment is not None
            else None,
            "orchestration_record": (
                orchestration.to_dict() if orchestration is not None else None
            ),
        }

    def list_models(self) -> list[dict[str, Any]]:
        records = [
            self._registry_record_view(record)
            for record in self.model_registry.list_models()
        ]
        return sorted(
            records,
            key=lambda record: _iso_sort(str(record.get("created_at"))),
            reverse=True,
        )

    def get_model_detail(self, registry_id: str) -> dict[str, Any]:
        record = self.model_registry.get_model(registry_id)
        if record is None:
            raise KeyError(f"Unknown registry id: {registry_id}")

        evaluation = self.evaluator.get_evaluation(record.evaluation_id)
        observability = self.observability.get_summary(registry_id)
        drift_report = self.drift_detector.get_report(registry_id).to_dict()
        alerts = self.alert_manager.evaluate(
            registry_id,
            observability=observability,
            drift_report=drift_report,
        )
        audit_entries = [
            entry.to_dict()
            for entry in self.audit_logger.list_entries(
                subject_type="model_registry",
                subject_id=registry_id,
            )
        ]
        run_detail = self.get_run_detail(record.run_id)
        workflows = [
            payload
            for payload in self._list_category("workflows")
            if payload.get("registry_id") == registry_id
        ]
        retraining_requests = [
            payload
            for payload in self._list_category("retraining-requests")
            if payload.get("registry_id") == registry_id
        ]

        return {
            "model": self._registry_record_view(record),
            "evaluation": evaluation.to_dict() if evaluation is not None else None,
            "observability": observability,
            "drift_report": drift_report,
            "alerts": alerts,
            "audit_entries": audit_entries,
            "run": run_detail,
            "workflows": workflows,
            "retraining_requests": retraining_requests,
            "can_promote": record.evaluation_status == "passed",
            "serving_ready": record.stage in {"staging", "production"}
            and record.evaluation_status == "passed",
        }

    def list_tasks(self, *, include_completed: bool = True) -> dict[str, Any]:
        tasks = [
            task.to_dict()
            for task in self.task_queue.list_tasks(include_completed=include_completed)
        ]
        return {
            "tasks": sorted(
                tasks,
                key=lambda task: _iso_sort(str(task.get("created_at"))),
                reverse=False,
            ),
            "source_path": str(self.task_queue.active_path()),
            "source_format": self.task_queue.active_format(),
            "agent": self.get_agent_status(),
        }

    def create_task(
        self,
        *,
        text: str,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task = self.task_queue.add_task(text, task_id=task_id, metadata=metadata)
        return task.to_dict()

    def update_task(
        self,
        task_id: str,
        *,
        text: str | None = None,
        completed: bool | None = None,
    ) -> dict[str, Any]:
        task = self.task_queue.update_task(
            task_id=task_id,
            text=text,
            completed=completed,
        )
        if task is None:
            raise KeyError(f"Unknown task id: {task_id}")
        return task.to_dict()

    def delete_task(self, task_id: str) -> dict[str, Any]:
        task = self.task_queue.delete_task(task_id=task_id)
        if task is None:
            raise KeyError(f"Unknown task id: {task_id}")
        return task.to_dict()

    def list_datasets(self) -> dict[str, Any]:
        manifests = []
        for manifest in self.dataset_registry.list_manifests():
            manifests.append(self._dataset_manifest_view(manifest))
        return {
            "datasets": sorted(
                manifests,
                key=lambda item: _iso_sort(str(item.get("updated_at"))),
                reverse=True,
            )
        }

    def get_dataset_detail(self, dataset_id: str, version: str) -> dict[str, Any]:
        manifest = self.dataset_registry.get_manifest(dataset_id, version)
        if manifest is None:
            raise KeyError(f"Unknown dataset manifest: {dataset_id}@{version}")
        return self._dataset_manifest_view(manifest)

    def get_agent_status(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "running": False,
            "iterations": 0,
            "recent_task": None,
            "last_updated": None,
            "last_training_plan": None,
            "last_training_analysis": None,
        }
        state_file = self.config.agent_state_file
        if state_file.exists():
            try:
                loaded = json.loads(state_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload.update(loaded)
            except Exception as exc:
                logger.warning("agent_state_load_failed", error=str(exc))

        payload["state_file"] = str(state_file)
        payload["task_source"] = str(self.task_queue.active_path())
        return payload

    def _track_run_task(
        self,
        run_id: str,
        task: asyncio.Task[TrainingRunRecord],
    ) -> None:
        self._active_run_tasks[run_id] = task

        def _cleanup(completed_task: asyncio.Task[TrainingRunRecord]) -> None:
            self._active_run_tasks.pop(run_id, None)
            try:
                completed_task.exception()
            except asyncio.CancelledError:
                logger.warning("web_console_run_task_cancelled", run_id=run_id)
            except Exception as exc:
                logger.warning(
                    "web_console_run_task_failed",
                    run_id=run_id,
                    error=str(exc),
                )

        task.add_done_callback(_cleanup)

    def _build_run_summary(self, run_id: str) -> dict[str, Any] | None:
        orchestration, experiment, context = self._load_run_sources(run_id)
        if orchestration is None and experiment is None:
            return None

        result = orchestration.result if orchestration is not None else None
        metrics = experiment.metrics if experiment is not None else {}
        model_id = (
            orchestration.model_id
            if orchestration is not None
            else experiment.model_id
            if experiment is not None
            else context.model_id
            if context is not None
            else run_id
        )
        created_at = (
            orchestration.created_at
            if orchestration is not None
            else experiment.created_at
            if experiment is not None
            else None
        )
        updated_at = (
            orchestration.updated_at
            if orchestration is not None
            else experiment.updated_at
            if experiment is not None
            else created_at
        )
        backend = (
            orchestration.plan.backend.value
            if orchestration is not None and orchestration.plan is not None
            else experiment.backend
            if experiment is not None
            else context.backend.value
            if context is not None
            else ""
        )
        architecture = (
            orchestration.plan.architecture
            if orchestration is not None and orchestration.plan is not None
            else experiment.architecture
            if experiment is not None
            else ""
        )
        dataset_id = (
            orchestration.dataset.dataset_id
            if orchestration is not None and orchestration.dataset is not None
            else experiment.dataset_id
            if experiment is not None
            else (
                orchestration.plan.dataset
                if orchestration is not None and orchestration.plan is not None
                else ""
            )
        )
        dataset_version = (
            experiment.dataset_version if experiment is not None else "builtin-v1"
        )
        runtime_state = context.state.value if context is not None else None
        status = (
            orchestration.status
            if orchestration is not None
            else experiment.status
            if experiment is not None
            else runtime_state
        )
        final_accuracy = (
            result.final_accuracy
            if result is not None and result.final_accuracy is not None
            else _float_or_none(metrics.get("final_accuracy"))
        )
        final_loss = (
            result.final_loss
            if result is not None
            else _float_or_none(metrics.get("final_loss"))
        )
        is_training = self.orchestrator.training_pipeline.is_training(model_id)

        return {
            "run_id": run_id,
            "model_id": model_id,
            "status": status,
            "runtime_state": runtime_state,
            "backend": backend,
            "architecture": architecture,
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "created_at": created_at,
            "updated_at": updated_at,
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
            "epochs_completed": (
                result.epochs_completed
                if result is not None
                else context.epochs_completed
                if context is not None
                else None
            ),
            "health_status": result.health_status if result is not None else None,
            "data_mode": result.data_mode if result is not None else None,
            "error": orchestration.error if orchestration is not None else None,
            "has_checkpoint": (
                result.checkpoint_path is not None
                if result is not None
                else context.checkpoint_path is not None
                if context is not None
                else False
            ),
            "source": self._run_source(orchestration, experiment),
            "is_training": is_training,
            "can_resume": orchestration is not None
            and orchestration.status in {"failed", "cancelled"},
            "can_cancel": is_training,
            "can_register": status == "completed",
        }

    def _load_run_sources(
        self,
        run_id: str,
    ) -> tuple[
        TrainingRunRecord | None, ExperimentRunRecord | None, ModelContext | None
    ]:
        orchestration = self.orchestrator.get_run(run_id)
        experiment = self.run_store.get_run(run_id)
        context_model_id = None
        if orchestration is not None:
            context_model_id = orchestration.model_id
        elif experiment is not None:
            context_model_id = experiment.model_id
        context = (
            self.context_manager.get_context(context_model_id)
            if context_model_id is not None
            else None
        )
        return orchestration, experiment, context

    def _run_source(
        self,
        orchestration: TrainingRunRecord | None,
        experiment: ExperimentRunRecord | None,
    ) -> str:
        if orchestration is not None and experiment is not None:
            return "orchestration+experiment"
        if orchestration is not None:
            return "orchestration"
        return "experiment"

    def _registry_record_view(self, record: ModelRegistryRecord) -> dict[str, Any]:
        evaluation = self.evaluator.get_evaluation(record.evaluation_id)
        return {
            **record.to_dict(),
            "evaluation_issues": evaluation.issues if evaluation is not None else [],
            "eligible_for_promotion": bool(
                record.metadata.get("eligible_for_promotion", False)
            ),
            "rejection_reason": record.metadata.get("rejection_reason"),
        }

    def _dataset_manifest_view(self, manifest: DatasetManifest) -> dict[str, Any]:
        report = self.dataset_validator.get_report(
            manifest.dataset_id, manifest.version
        )
        payload = manifest.to_dict()
        payload["validation_report"] = report.to_dict() if report is not None else None
        return payload

    def _metrics_history(self, context: ModelContext | None) -> list[dict[str, Any]]:
        if context is None:
            return []
        return [asdict(metric) for metric in context.metrics]

    def _count_by_key(self, records: list[dict[str, Any]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            name = str(record.get(key) or "unknown")
            counts[name] = counts.get(name, 0) + 1
        return counts

    def _list_alerts(self) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for payload in self._list_category("alerts"):
            registry_id = str(payload.get("registry_id", ""))
            alerts = payload.get("alerts", [])
            if not isinstance(alerts, list):
                continue
            for alert in alerts:
                if not isinstance(alert, dict):
                    continue
                flattened.append(
                    {
                        "registry_id": registry_id,
                        "type": str(alert.get("type", "unknown")),
                        "message": str(alert.get("message", "")),
                    }
                )
        return flattened

    def _list_workflows(self) -> list[dict[str, Any]]:
        workflows = self._list_category("workflows")
        return sorted(
            workflows,
            key=lambda workflow: _iso_sort(str(workflow.get("created_at"))),
            reverse=True,
        )

    def _list_category(self, category: str) -> list[dict[str, Any]]:
        return self.metadata_store.list_records(category)
