"""Training orchestration for Ghost.

Coordinates planning, dataset resolution, model creation, training, and
post-run analysis as a reusable application-layer service.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ghost.config import GhostConfig, get_config
from ghost.context import BackendType, ContextManager, TrainingMetrics
from ghost.datasets import DatasetResolver, DatasetSpec, dataset_input_shape
from ghost.experiment_tracking import ExperimentTracker
from ghost.logging import get_logger
from ghost.metadata_store import MetadataStore
from ghost.ollama_client import OllamaClient
from ghost.planning import PlanningRequest, TrainingPlan, TrainingPlanner
from ghost.pytorch_ops import PyTorchOps
from ghost.run_store import RunStore
from ghost.tensorflow_ops import TensorFlowOps
from ghost.training import TrainingPipeline, TrainingResult

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrainingRunRequest:
    """Request to execute or resume a training run."""

    task: str
    dataset_ref: str = ""
    dataset: str = ""
    model_name: str = ""
    model_id: str | None = None
    allow_synthetic: bool = False
    recommendations: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingRunRequest:
        return cls(**payload)


@dataclass
class TrainingRunRecord:
    """Operational record for a training run."""

    run_id: str
    model_id: str
    status: str
    plan: TrainingPlan | None
    analysis: dict[str, Any] | None
    events: list[dict[str, Any]]
    request: TrainingRunRequest | None = None
    dataset: DatasetSpec | None = None
    result: TrainingResult | None = None
    error: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "status": self.status,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "analysis": self.analysis,
            "events": self.events,
            "request": self.request.to_dict() if self.request is not None else None,
            "dataset": asdict(self.dataset) if self.dataset is not None else None,
            "result": _training_result_to_dict(self.result),
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingRunRecord:
        return cls(
            run_id=str(payload["run_id"]),
            model_id=str(payload["model_id"]),
            status=str(payload["status"]),
            plan=_training_plan_from_dict(payload.get("plan")),
            analysis=(
                payload["analysis"]
                if isinstance(payload.get("analysis"), dict)
                else None
            ),
            events=(
                payload["events"] if isinstance(payload.get("events"), list) else []
            ),
            request=(
                TrainingRunRequest.from_dict(payload["request"])
                if isinstance(payload.get("request"), dict)
                else None
            ),
            dataset=(
                DatasetSpec(**payload["dataset"])
                if isinstance(payload.get("dataset"), dict)
                else None
            ),
            result=_training_result_from_dict(payload.get("result")),
            error=str(payload["error"]) if payload.get("error") is not None else None,
            created_at=str(payload.get("created_at", _utc_now_iso())),
            updated_at=str(payload.get("updated_at", _utc_now_iso())),
        )


def _training_plan_from_dict(payload: Any) -> TrainingPlan | None:
    if not isinstance(payload, dict):
        return None

    tips = payload.get("tips", [])
    raw_recommendations = payload.get("raw_recommendations", {})

    return TrainingPlan(
        task=str(payload["task"]),
        backend=BackendType(str(payload["backend"])),
        architecture=str(payload["architecture"]),
        num_classes=int(payload["num_classes"]),
        batch_size=int(payload["batch_size"]),
        learning_rate=float(payload["learning_rate"]),
        epochs=int(payload["epochs"]),
        dataset=str(payload.get("dataset", "")),
        optimizer=(
            str(payload["optimizer"]) if payload.get("optimizer") is not None else None
        ),
        recommendation_source=str(payload.get("recommendation_source", "defaults")),
        tips=[str(item) for item in tips] if isinstance(tips, list) else [],
        raw_recommendations=(
            raw_recommendations if isinstance(raw_recommendations, dict) else {}
        ),
    )


def _training_result_to_dict(result: TrainingResult | None) -> dict[str, Any] | None:
    if result is None:
        return None

    return {
        "model_id": result.model_id,
        "success": result.success,
        "final_loss": result.final_loss,
        "status": result.status,
        "cancelled": result.cancelled,
        "final_accuracy": result.final_accuracy,
        "epochs_completed": result.epochs_completed,
        "checkpoint_path": (
            str(result.checkpoint_path) if result.checkpoint_path is not None else None
        ),
        "duration_seconds": result.duration_seconds,
        "metrics_history": [asdict(metric) for metric in result.metrics_history],
        "effective_batch_size": result.effective_batch_size,
        "health_status": result.health_status,
        "health_issues": result.health_issues,
        "data_mode": result.data_mode,
        "used_synthetic_data": result.used_synthetic_data,
        "error": result.error,
    }


def _training_result_from_dict(payload: Any) -> TrainingResult | None:
    if not isinstance(payload, dict):
        return None

    metrics_payload = payload.get("metrics_history", [])
    metrics = [
        metric if isinstance(metric, TrainingMetrics) else TrainingMetrics(**metric)
        for metric in metrics_payload
        if isinstance(metric, (dict, TrainingMetrics))
    ]

    checkpoint_value = payload.get("checkpoint_path")
    checkpoint_path = Path(str(checkpoint_value)) if checkpoint_value else None

    return TrainingResult(
        model_id=str(payload["model_id"]),
        success=bool(payload["success"]),
        final_loss=float(payload["final_loss"]),
        status=str(payload.get("status", "")),
        cancelled=bool(payload.get("cancelled", False)),
        final_accuracy=(
            float(payload["final_accuracy"])
            if payload.get("final_accuracy") is not None
            else None
        ),
        epochs_completed=int(payload.get("epochs_completed", 0)),
        checkpoint_path=checkpoint_path,
        duration_seconds=float(payload.get("duration_seconds", 0.0)),
        metrics_history=metrics,
        effective_batch_size=(
            int(payload["effective_batch_size"])
            if payload.get("effective_batch_size") is not None
            else None
        ),
        health_status=str(payload.get("health_status", "unknown")),
        health_issues=(
            [str(item) for item in payload.get("health_issues", [])]
            if isinstance(payload.get("health_issues", []), list)
            else []
        ),
        data_mode=str(payload.get("data_mode", "unknown")),
        used_synthetic_data=bool(payload.get("used_synthetic_data", False)),
        error=str(payload["error"]) if payload.get("error") is not None else None,
    )


class TrainingOrchestrator:
    """Coordinate Ghost planning, training, and analysis."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        context_manager: ContextManager | None = None,
        planner: TrainingPlanner | None = None,
        dataset_resolver: DatasetResolver | None = None,
        training_pipeline: TrainingPipeline | None = None,
        ollama_client: OllamaClient | None = None,
        backend_ops: dict[BackendType, Any] | None = None,
        metadata_store: MetadataStore | None = None,
        run_store: RunStore | None = None,
        experiment_tracker: ExperimentTracker | None = None,
    ):
        self.config = config or get_config()
        self.context_manager = context_manager or ContextManager()
        self.ollama_client = ollama_client or OllamaClient()
        self.planner = planner or TrainingPlanner(
            config=self.config,
            ollama_client=self.ollama_client,
        )
        self.dataset_resolver = dataset_resolver or DatasetResolver(config=self.config)
        self.training_pipeline = training_pipeline or TrainingPipeline(
            self.context_manager
        )
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )
        self.run_store = run_store or RunStore(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.experiment_tracker = experiment_tracker or ExperimentTracker(
            run_store=self.run_store,
        )
        self._backend_ops = backend_ops or {
            BackendType.PYTORCH: PyTorchOps(self.context_manager),
            BackendType.TENSORFLOW: TensorFlowOps(self.context_manager),
        }

        self._records = self._load_records()
        logger.info("training_orchestrator_init")

    def prepare_run(self, request: TrainingRunRequest) -> TrainingRunRecord:
        """Create and persist a queued run record without executing it yet."""
        record = TrainingRunRecord(
            run_id=uuid4().hex,
            model_id=request.model_id or self._generate_model_id(),
            status="queued",
            plan=None,
            analysis=None,
            events=[],
            request=request,
        )
        self._record_event(record, "request_received", task=request.task)
        self._persist(record)
        return record

    async def execute(self, request: TrainingRunRequest) -> TrainingRunRecord:
        """Execute a planned training run end to end."""
        record = self.prepare_run(request)
        return await self.execute_prepared(record.run_id)

    async def execute_prepared(self, run_id: str) -> TrainingRunRecord:
        """Execute a previously prepared run record by id."""
        record = self._records.get(run_id)
        if record is None:
            raise KeyError(f"Unknown run id: {run_id}")
        if record.request is None:
            raise ValueError(f"Run {run_id} does not contain a training request")
        return await self._execute_record(record)

    async def _execute_record(self, record: TrainingRunRecord) -> TrainingRunRecord:
        """Execute a queued run record and persist lifecycle updates."""
        try:
            request = record.request
            if request is None:
                raise ValueError(
                    f"Run {record.run_id} does not contain a training request"
                )

            dataset_spec = self._resolve_dataset(request, record)
            plan = await self.planner.create_plan(
                PlanningRequest(
                    task=request.task,
                    dataset=request.dataset or request.dataset_ref,
                    recommendations=request.recommendations,
                )
            )
            if dataset_spec is not None:
                plan.dataset = dataset_spec.dataset_id
                if "num_classes" not in plan.raw_recommendations:
                    plan.num_classes = dataset_spec.num_classes

            record.plan = plan
            record.status = "planned"
            self._record_event(
                record,
                "plan_created",
                backend=plan.backend.value,
                architecture=plan.architecture,
                dataset=plan.dataset,
            )

            await self._ensure_model_ready(record, model_name=request.model_name)

            record.status = "running"
            self._record_event(record, "training_started")
            result = await self.training_pipeline.train(
                plan.to_training_config(
                    model_id=record.model_id,
                    checkpoint_interval=self.config.checkpoint_interval,
                )
            )
            record.result = result
            record.analysis = await self._analyze_result(record, result)

            if result.status == "completed":
                await self._ensure_checkpoint_artifact(record, result)
                record.status = "completed"
                record.error = None
                self._record_event(
                    record,
                    "training_completed",
                    epochs_completed=result.epochs_completed,
                )
            elif result.status == "cancelled":
                record.status = "cancelled"
                record.error = result.error
                self._record_event(
                    record,
                    "training_cancelled",
                    error=result.error or "Training cancelled",
                )
            else:
                record.status = "failed"
                record.error = result.error
                self._record_event(
                    record,
                    "training_failed",
                    error=result.error or "Training failed",
                )
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            self._record_event(record, "run_failed", error=str(exc))
            logger.error(
                "training_orchestration_failed", run_id=record.run_id, error=str(exc)
            )

        self._persist(record)
        return record

    async def resume(self, run_id: str) -> TrainingRunRecord:
        """Resume a previously recorded run.

        If the runtime model is missing, the orchestrator recreates it and tries
        to restore from the last checkpoint when available.
        """
        record = self._records.get(run_id)
        if record is None:
            raise KeyError(f"Unknown run id: {run_id}")
        if record.request is None or record.plan is None:
            raise ValueError(f"Run {run_id} does not contain enough state to resume")

        if record.status == "completed":
            return record

        self._record_event(record, "resume_requested", previous_status=record.status)

        try:
            await self._ensure_model_ready(
                record,
                model_name=record.request.model_name,
                try_restore_checkpoint=True,
            )
            record.status = "resumed"
            result = await self.training_pipeline.train(
                record.plan.to_training_config(
                    model_id=record.model_id,
                    checkpoint_interval=self.config.checkpoint_interval,
                )
            )
            record.result = result
            record.analysis = await self._analyze_result(record, result)

            if result.status == "completed":
                await self._ensure_checkpoint_artifact(record, result)
                record.status = "completed"
                record.error = None
                self._record_event(
                    record,
                    "resume_completed",
                    epochs_completed=result.epochs_completed,
                )
            elif result.status == "cancelled":
                record.status = "cancelled"
                record.error = result.error
                self._record_event(
                    record,
                    "resume_cancelled",
                    error=result.error or "Resumed training cancelled",
                )
            else:
                record.status = "failed"
                record.error = result.error
                self._record_event(
                    record,
                    "resume_failed",
                    error=result.error or "Resumed training failed",
                )
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            self._record_event(record, "resume_failed", error=str(exc))
            logger.error("training_resume_failed", run_id=run_id, error=str(exc))

        self._persist(record)
        return record

    def get_run(self, run_id: str) -> TrainingRunRecord | None:
        """Return a recorded run if present."""
        return self._records.get(run_id)

    def list_runs(self) -> list[TrainingRunRecord]:
        """Return all known runs ordered by creation time."""
        return sorted(self._records.values(), key=lambda record: record.created_at)

    def _resolve_dataset(
        self,
        request: TrainingRunRequest,
        record: TrainingRunRecord,
    ) -> DatasetSpec | None:
        dataset_reference = request.dataset_ref or request.dataset
        if not dataset_reference:
            return None

        dataset_spec = self.dataset_resolver.resolve(
            dataset_reference,
            allow_synthetic=request.allow_synthetic,
        )
        record.dataset = dataset_spec
        self._record_event(
            record,
            "dataset_resolved",
            dataset_id=dataset_spec.dataset_id,
            synthetic=dataset_spec.synthetic,
        )
        return dataset_spec

    async def _ensure_model_ready(
        self,
        record: TrainingRunRecord,
        *,
        model_name: str = "",
        try_restore_checkpoint: bool = False,
    ) -> None:
        if record.plan is None:
            raise ValueError("Cannot prepare model without a training plan")

        plan = record.plan
        ops = self._backend_ops[plan.backend]
        ctx = self.context_manager.get_context(record.model_id)
        models = getattr(ops, "models", None)
        runtime_model_missing = (
            isinstance(models, dict) and record.model_id not in models
        )

        if ctx is None or runtime_model_missing:
            create_result = await ops.create_model(
                model_id=record.model_id,
                model_name=model_name
                or record.request.model_name
                or record.request.task[:50]
                if record.request
                else record.model_id,
                architecture=plan.architecture,
                num_classes=plan.num_classes,
                input_shape=list(dataset_input_shape(record.dataset, plan.backend))
                if record.dataset is not None
                else None,
            )
            if create_result.get("status") != "success":
                raise RuntimeError(
                    create_result.get("message", "Model creation failed")
                )
            self._record_event(
                record,
                "model_ready",
                backend=plan.backend.value,
                architecture=plan.architecture,
            )
            ctx = self.context_manager.get_context(record.model_id)

        if (
            try_restore_checkpoint
            and ctx is not None
            and ctx.checkpoint_path is not None
        ):
            load_checkpoint = getattr(ops, "load_checkpoint", None)
            if callable(load_checkpoint):
                load_result = await load_checkpoint(
                    record.model_id, str(ctx.checkpoint_path)
                )
                if load_result.get("status") == "success":
                    self._record_event(
                        record,
                        "checkpoint_restored",
                        path=load_result.get("path", str(ctx.checkpoint_path)),
                    )
                else:
                    self._record_event(
                        record,
                        "checkpoint_restore_failed",
                        error=load_result.get("message", "Checkpoint restore failed"),
                    )

        if ctx is not None:
            ctx.metadata["orchestration_run_id"] = record.run_id
            ctx.metadata["training_plan"] = plan.to_dict()
            if record.dataset is not None:
                ctx.metadata["dataset_spec"] = asdict(record.dataset)
            self.context_manager.update_context(ctx)

    async def _analyze_result(
        self,
        record: TrainingRunRecord,
        result: TrainingResult,
    ) -> dict[str, Any] | None:
        metrics = result.metrics_history
        if not metrics:
            return None

        try:
            analysis = await self.ollama_client.analyze_training_progress(
                [
                    {
                        "epoch": metric.epoch,
                        "step": metric.step,
                        "loss": metric.loss,
                        "accuracy": metric.accuracy,
                        "learning_rate": metric.learning_rate,
                    }
                    for metric in metrics
                ]
            )
        except Exception as exc:
            self._record_event(record, "analysis_unavailable", error=str(exc))
            return None

        ctx = self.context_manager.get_context(record.model_id)
        if ctx is not None:
            ctx.metadata["training_analysis"] = analysis
            self.context_manager.update_context(ctx)

        self._record_event(record, "analysis_completed")
        return analysis

    async def _ensure_checkpoint_artifact(
        self,
        record: TrainingRunRecord,
        result: TrainingResult,
    ) -> None:
        if result.checkpoint_path is not None or record.plan is None:
            return

        ops = self._backend_ops[record.plan.backend]
        save_checkpoint = getattr(ops, "save_checkpoint", None)
        if save_checkpoint is None:
            return

        maybe_coro = save_checkpoint(record.model_id)
        if not inspect.isawaitable(maybe_coro):
            return

        save_result = await maybe_coro
        if save_result.get("status") != "success":
            self._record_event(
                record,
                "checkpoint_save_failed",
                error=save_result.get("message", "Checkpoint save failed"),
            )
            return

        checkpoint_value = save_result.get("path")
        if checkpoint_value:
            result.checkpoint_path = Path(str(checkpoint_value))
            self._record_event(
                record,
                "checkpoint_saved",
                path=str(result.checkpoint_path),
            )

    def _record_event(
        self,
        record: TrainingRunRecord,
        stage: str,
        **details: Any,
    ) -> None:
        record.updated_at = _utc_now_iso()
        event = {"timestamp": record.updated_at, "stage": stage}
        event.update(details)
        record.events.append(event)

    def _persist(self, record: TrainingRunRecord) -> None:
        record.updated_at = _utc_now_iso()
        self._records[record.run_id] = record
        self.metadata_store.save_record("runs", record.run_id, record.to_dict())
        try:
            self.experiment_tracker.record_training_run(
                record,
                context=self.context_manager.get_context(record.model_id),
            )
        except Exception as exc:
            logger.warning(
                "experiment_tracking_failed",
                run_id=record.run_id,
                error=str(exc),
            )

    def _load_records(self) -> dict[str, TrainingRunRecord]:
        records: dict[str, TrainingRunRecord] = {}
        for payload in self.metadata_store.list_records("runs"):
            try:
                record = TrainingRunRecord.from_dict(payload)
            except Exception as exc:
                logger.warning("run_record_load_failed", error=str(exc))
                continue
            records[record.run_id] = record
        return records

    def _generate_model_id(self) -> str:
        return f"model_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
