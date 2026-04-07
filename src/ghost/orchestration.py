"""Training orchestration for Ghost.

Coordinates planning, dataset resolution, model creation, training, and
post-run analysis as a reusable application-layer service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ghost.config import GhostConfig, get_config
from ghost.context import BackendType, ContextManager
from ghost.datasets import DatasetResolver, DatasetSpec
from ghost.logging import get_logger
from ghost.ollama_client import OllamaClient
from ghost.planning import PlanningRequest, TrainingPlan, TrainingPlanner
from ghost.pytorch_ops import PyTorchOps
from ghost.tensorflow_ops import TensorFlowOps
from ghost.training import TrainingPipeline, TrainingResult

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
    ):
        self.config = config or get_config()
        self.context_manager = context_manager or ContextManager()
        self.ollama_client = ollama_client or OllamaClient()
        self.planner = planner or TrainingPlanner(
            config=self.config,
            ollama_client=self.ollama_client,
        )
        self.dataset_resolver = dataset_resolver or DatasetResolver(config=self.config)
        self.training_pipeline = training_pipeline or TrainingPipeline(self.context_manager)
        self._backend_ops = backend_ops or {
            BackendType.PYTORCH: PyTorchOps(self.context_manager),
            BackendType.TENSORFLOW: TensorFlowOps(self.context_manager),
        }

        runtime = self.context_manager.get_runtime_bucket("orchestration")
        self._records: dict[str, TrainingRunRecord] = runtime.setdefault("runs", {})
        logger.info("training_orchestrator_init")

    async def execute(self, request: TrainingRunRequest) -> TrainingRunRecord:
        """Execute a planned training run end to end."""
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

        try:
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

            if result.success:
                record.status = "completed"
                self._record_event(
                    record,
                    "training_completed",
                    epochs_completed=result.epochs_completed,
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
            logger.error("training_orchestration_failed", run_id=record.run_id, error=str(exc))

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

            if result.success:
                record.status = "completed"
                record.error = None
                self._record_event(
                    record,
                    "resume_completed",
                    epochs_completed=result.epochs_completed,
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

    def _resolve_dataset(
        self,
        request: TrainingRunRequest,
        record: TrainingRunRecord,
    ) -> DatasetSpec | None:
        if not request.dataset_ref:
            return None

        dataset_spec = self.dataset_resolver.resolve(
            request.dataset_ref,
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
        runtime_model_missing = isinstance(models, dict) and record.model_id not in models

        if ctx is None or runtime_model_missing:
            create_result = await ops.create_model(
                model_id=record.model_id,
                model_name=model_name or record.request.model_name or record.request.task[:50] if record.request else record.model_id,
                architecture=plan.architecture,
                num_classes=plan.num_classes,
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

        if try_restore_checkpoint and ctx is not None and ctx.checkpoint_path is not None:
            load_checkpoint = getattr(ops, "load_checkpoint", None)
            if callable(load_checkpoint):
                load_result = await load_checkpoint(record.model_id, str(ctx.checkpoint_path))
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

    def _generate_model_id(self) -> str:
        return f"model_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"