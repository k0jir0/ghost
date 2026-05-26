"""MCP Server implementation for Ghost.

Provides Model Context Protocol tools for ML training and inference operations.
All tool arguments are validated with Pydantic before reaching backend ops,
so malformed inputs surface as structured errors rather than deep stack traces.
"""

from __future__ import annotations

from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    Tool,
)
from pydantic import ValidationError

from ghost.config import get_config
from ghost.context import BackendType, ContextManager
from ghost.data_validation import DatasetValidator
from ghost.dataset_registry import DatasetRegistry
from ghost.evaluation import EvaluationPolicy
from ghost.health_monitor import HealthMonitor
from ghost.inference import InferenceService
from ghost.logging import get_logger
from ghost.metadata_store import MetadataStore
from ghost.model_registry import ModelRegistry
from ghost.alerts import AlertManager
from ghost.drift import DriftDetector
from ghost.observability import ModelObservability
from ghost.ollama_client import OllamaClient
from ghost.orchestration import TrainingRunRecord
from ghost.pytorch_ops import PyTorchOps
from ghost.run_store import RunStore
from ghost.task_queue import TaskQueueStore
from ghost.tensorflow_ops import TensorFlowOps
from ghost.tool_catalog import ToolCatalog, ToolSpec

logger = get_logger(__name__)

_DEFAULT_TOOL_CATALOG = ToolCatalog.default()
# Backward-compatibility shim for validation-focused tests and callers that
# still introspect the MCP argument model registry from this module.
_TOOL_ARG_MODELS = _DEFAULT_TOOL_CATALOG.argument_models()


class GhostMCPServer:
    """MCP Server for Ghost ML platform.

    Provides tools for PyTorch and TensorFlow operations with context tracking.
    """

    def __init__(
        self,
        context_manager: ContextManager | None = None,
        ollama_client: OllamaClient | None = None,
        health_monitor: HealthMonitor | None = None,
        task_queue: TaskQueueStore | None = None,
        metadata_store: MetadataStore | None = None,
        dataset_registry: DatasetRegistry | None = None,
        dataset_validator: DatasetValidator | None = None,
    ):
        """Initialize the MCP server."""
        self.server = Server("ghost-mcp")
        self.config = get_config()
        self.context_manager = context_manager or ContextManager()
        self.ollama_client = ollama_client or OllamaClient()
        self.health_monitor = health_monitor or HealthMonitor()
        self.pytorch_ops = PyTorchOps(self.context_manager)
        self.tensorflow_ops = TensorFlowOps(self.context_manager)
        self.task_queue = task_queue or TaskQueueStore(self.config.task_queue_file)
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )
        self.run_store = RunStore(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.model_registry = ModelRegistry(
            config=self.config,
            metadata_store=self.metadata_store,
            run_store=self.run_store,
        )
        self.observability = ModelObservability(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.drift_detector = DriftDetector(
            config=self.config,
            metadata_store=self.metadata_store,
            observability=self.observability,
        )
        self.alert_manager = AlertManager(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.inference_service = InferenceService(
            config=self.config,
            context_manager=self.context_manager,
            model_registry=self.model_registry,
            backend_ops={
                BackendType.PYTORCH: self.pytorch_ops,
                BackendType.TENSORFLOW: self.tensorflow_ops,
            },
            observability=self.observability,
        )
        self.dataset_registry = dataset_registry or DatasetRegistry(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.dataset_validator = dataset_validator or DatasetValidator(
            config=self.config,
            metadata_store=self.metadata_store,
        )
        self.tool_catalog = _DEFAULT_TOOL_CATALOG

        self._register_tools()

    def _spec_to_tool(self, spec: ToolSpec) -> Tool:
        return Tool(
            name=spec.name,
            description=spec.description,
            inputSchema=spec.input_schema(),
        )

    def _call_tool_result(
        self,
        payload: dict[str, Any],
        *,
        is_error: bool = False,
    ) -> CallToolResult:
        return CallToolResult(
            content=[],
            structuredContent=payload,
            isError=is_error,
        )

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available MCP tools."""
            return ListToolsResult(
                tools=[
                    self._spec_to_tool(spec) for spec in self.tool_catalog.list_specs()
                ]
            )

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> CallToolResult:
            """Handle tool calls with upfront Pydantic validation."""
            # --- Shift-left: validate arguments before any business logic ---
            spec = self.tool_catalog.get_spec(name)
            if spec is None:
                return self._call_tool_result(
                    {"error": {"message": f"Unknown tool: {name}", "tool": name}},
                    is_error=True,
                )
            try:
                spec.input_model.model_validate(arguments or {})
            except ValidationError as exc:
                logger.warning("tool_validation_error", tool=name, errors=exc.errors())
                return self._call_tool_result(
                    {
                        "error": {
                            "message": f"Invalid arguments for '{name}'",
                            "tool": name,
                            "details": exc.errors(),
                        }
                    },
                    is_error=True,
                )

            try:
                result = await self._handle_tool(name, arguments or {})
                return self._call_tool_result(result)
            except Exception as exc:
                logger.error("tool_error", tool=name, error=str(exc))
                return self._call_tool_result(
                    {"error": {"message": str(exc), "tool": name}},
                    is_error=True,
                )

    async def _handle_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Route tool calls to appropriate handlers."""
        spec = self.tool_catalog.get_spec(name)
        if spec is None:
            return {"error": f"Unknown tool: {name}"}

        handler = getattr(self, spec.handler_name, None)
        if handler is None:
            return {"error": f"Handler not found for tool: {name}"}

        return await handler(arguments)

    async def _handle_pytorch_create_model(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.pytorch_ops.create_model(
            model_id=arguments["model_id"],
            model_name=arguments["model_name"],
            architecture=arguments.get("architecture", "mlp"),
            num_classes=arguments.get("num_classes", 10),
            input_shape=arguments.get("input_shape", [3, 224, 224]),
        )

    async def _handle_pytorch_train_step(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.pytorch_ops.train_step(
            model_id=arguments["model_id"],
            batch_size=arguments.get("batch_size", 32),
            learning_rate=arguments.get("learning_rate", 0.001),
        )

    async def _handle_pytorch_evaluate(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.pytorch_ops.evaluate(arguments["model_id"])

    async def _handle_pytorch_save_checkpoint(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.pytorch_ops.save_checkpoint(
            model_id=arguments["model_id"],
            path=arguments.get("path"),
        )

    async def _handle_pytorch_load_checkpoint(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.pytorch_ops.load_checkpoint(
            model_id=arguments["model_id"],
            path=arguments["path"],
        )

    async def _handle_tensorflow_create_model(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.tensorflow_ops.create_model(
            model_id=arguments["model_id"],
            model_name=arguments["model_name"],
            architecture=arguments.get("architecture", "mlp"),
            num_classes=arguments.get("num_classes", 10),
            input_shape=arguments.get("input_shape", [224, 224, 3]),
        )

    async def _handle_tensorflow_train_step(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.tensorflow_ops.train_step(
            model_id=arguments["model_id"],
            batch_size=arguments.get("batch_size", 32),
            learning_rate=arguments.get("learning_rate", 0.001),
        )

    async def _handle_tensorflow_evaluate(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.tensorflow_ops.evaluate(arguments["model_id"])

    async def _handle_tensorflow_save_checkpoint(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.tensorflow_ops.save_checkpoint(
            model_id=arguments["model_id"],
            path=arguments.get("path"),
        )

    async def _handle_tensorflow_load_checkpoint(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.tensorflow_ops.load_checkpoint(
            model_id=arguments["model_id"],
            path=arguments["path"],
        )

    async def _handle_get_training_status(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        ctx = self.context_manager.get_context(arguments["model_id"])
        if ctx:
            return {
                "model_id": ctx.model_id,
                "state": ctx.state.value,
                "epochs_completed": ctx.epochs_completed,
                "current_step": ctx.current_step,
                "metrics": len(ctx.metrics),
            }
        return {"error": "Model not found"}

    async def _handle_list_models(self, arguments: dict[str, Any]) -> dict[str, Any]:
        contexts = self.context_manager.list_contexts()
        return {
            "models": [
                {
                    "model_id": ctx.model_id,
                    "name": ctx.model_name,
                    "backend": ctx.backend.value,
                }
                for ctx in contexts
            ]
        }

    async def _handle_list_runs(self, arguments: dict[str, Any]) -> dict[str, Any]:
        experiment_runs = [run.to_dict() for run in self.run_store.list_runs()]
        if experiment_runs:
            return {
                "runs": experiment_runs,
                "count": len(experiment_runs),
            }

        runs: list[dict[str, Any]] = []
        for payload in self.metadata_store.list_records("runs"):
            try:
                runs.append(TrainingRunRecord.from_dict(payload).to_dict())
            except Exception as exc:
                logger.warning("run_record_load_failed", error=str(exc))
        return {
            "runs": runs,
            "count": len(runs),
        }

    async def _handle_get_run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        experiment_run = self.run_store.get_run(arguments["run_id"])
        if experiment_run is not None:
            return {
                "run": experiment_run.to_dict(),
            }

        payload = self.metadata_store.load_record("runs", arguments["run_id"])
        if payload is None:
            return {"error": "Run not found"}

        try:
            record = TrainingRunRecord.from_dict(payload)
        except Exception as exc:
            logger.warning(
                "run_record_load_failed",
                run_id=arguments["run_id"],
                error=str(exc),
            )
            return {"error": "Run record is corrupt or incompatible"}

        return {
            "run": record.to_dict(),
        }

    async def _handle_compare_runs(self, arguments: dict[str, Any]) -> dict[str, Any]:
        comparison = self.run_store.compare_runs(arguments["run_ids"])
        if comparison["count"] != len(arguments["run_ids"]):
            found_ids = {run["run_id"] for run in comparison["runs"]}
            missing = [run_id for run_id in arguments["run_ids"] if run_id not in found_ids]
            return {
                "error": "One or more runs were not found",
                "missing_run_ids": missing,
            }
        return comparison

    async def _handle_register_model(self, arguments: dict[str, Any]) -> dict[str, Any]:
        policy = EvaluationPolicy(
            min_accuracy=arguments.get("min_accuracy"),
            max_loss=arguments.get("max_loss"),
            max_accuracy_drop=arguments.get("max_accuracy_drop", 0.05),
            max_loss_increase=arguments.get("max_loss_increase", 0.5),
        )
        record = self.model_registry.register_model(
            arguments["run_id"],
            actor=arguments.get("actor", "system"),
            policy=policy,
            baseline_registry_id=arguments.get("baseline_registry_id"),
        )
        return {"model": record.to_dict()}

    async def _handle_list_registered_models(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        models = [
            record.to_dict()
            for record in self.model_registry.list_models(
                stage=arguments.get("stage"),
                model_id=arguments.get("model_id"),
            )
        ]
        return {"models": models, "count": len(models)}

    async def _handle_promote_model(self, arguments: dict[str, Any]) -> dict[str, Any]:
        record = self.model_registry.promote_model(
            arguments["registry_id"],
            stage=arguments["stage"],
            actor=arguments.get("approved_by", "system"),
            alias=arguments.get("alias"),
        )
        return {"model": record.to_dict()}

    async def _handle_reject_model(self, arguments: dict[str, Any]) -> dict[str, Any]:
        record = self.model_registry.reject_model(
            arguments["registry_id"],
            reason=arguments["reason"],
            actor=arguments.get("rejected_by", "system"),
        )
        return {"model": record.to_dict()}

    async def _handle_predict_online(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self.inference_service.predict_online(
            arguments["registry_id"],
            arguments["features"],
        )

    async def _handle_predict_batch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self.inference_service.predict_batch(
            arguments["registry_id"],
            arguments["inputs"],
        )

    async def _handle_get_model_observability(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        summary = self.observability.get_summary(arguments["registry_id"])
        drift_report = self.drift_detector.get_report(arguments["registry_id"])
        alerts = self.alert_manager.evaluate(
            arguments["registry_id"],
            observability=summary,
            drift_report=drift_report.to_dict(),
        )
        return {
            "observability": summary,
            "alerts": alerts,
        }

    async def _handle_get_drift_report(self, arguments: dict[str, Any]) -> dict[str, Any]:
        report = self.drift_detector.get_report(arguments["registry_id"])
        return {"report": report.to_dict()}

    async def _handle_list_dataset_manifests(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        manifests = [manifest.to_dict() for manifest in self.dataset_registry.list_manifests()]
        return {
            "manifests": manifests,
            "count": len(manifests),
        }

    async def _handle_get_dataset_manifest(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        manifest = self.dataset_registry.get_manifest(
            arguments["dataset_id"],
            arguments.get("version", "builtin-v1"),
        )
        if manifest is None:
            return {"error": "Dataset manifest not found"}
        return {"manifest": manifest.to_dict()}

    async def _handle_get_dataset_validation_report(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        report = self.dataset_validator.get_report(
            arguments["dataset_id"],
            arguments.get("version", "builtin-v1"),
        )
        if report is None:
            return {"error": "Dataset validation report not found"}
        return {"report": report.to_dict()}

    async def _handle_list_training_tasks(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        tasks = self.task_queue.list_tasks(
            include_completed=arguments.get("include_completed", False)
        )
        return {
            "queue": [task.to_dict() for task in tasks],
            "path": str(self.task_queue.path),
            "active_path": str(self.task_queue.active_path()),
            "format": self.task_queue.active_format(),
        }

    async def _handle_create_training_task(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        task = self.task_queue.add_task(
            arguments["text"],
            task_id=arguments.get("task_id"),
        )
        return {
            "status": "success",
            "task": task.to_dict(),
            "path": str(self.task_queue.path),
            "format": self.task_queue.active_format(),
        }

    async def _handle_update_training_task(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        task = self.task_queue.update_task(
            task_id=arguments.get("task_id"),
            match_text=arguments.get("match_text"),
            text=arguments.get("text"),
            completed=arguments.get("completed"),
        )
        if task is None:
            return {"error": "Task not found"}
        return {
            "status": "success",
            "task": task.to_dict(),
            "path": str(self.task_queue.path),
            "format": self.task_queue.active_format(),
        }

    async def _handle_delete_training_task(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        task = self.task_queue.delete_task(
            task_id=arguments.get("task_id"),
            match_text=arguments.get("match_text"),
        )
        if task is None:
            return {"error": "Task not found"}
        return {
            "status": "success",
            "task": task.to_dict(),
            "path": str(self.task_queue.path),
            "format": self.task_queue.active_format(),
        }

    async def _handle_get_system_health(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return self.health_monitor.get_health_report()

    async def _handle_get_model_recommendation(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.ollama_client.get_recommendation(
            task=arguments["task"],
            dataset=arguments.get("dataset", ""),
        )

    async def _handle_get_training_analysis(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        ctx = self.context_manager.get_context(arguments["model_id"])
        if not ctx:
            return {"error": "Model not found"}

        if not ctx.metrics:
            return {"error": "No training metrics available for model"}

        analysis = await self.ollama_client.analyze_training_progress(
            [
                {
                    "epoch": metric.epoch,
                    "step": metric.step,
                    "loss": metric.loss,
                    "accuracy": metric.accuracy,
                    "learning_rate": metric.learning_rate,
                }
                for metric in ctx.metrics
            ]
        )
        if analysis.get("status") == "success":
            ctx.metadata["training_analysis"] = analysis
            self.context_manager.update_context(ctx)
        return analysis

    async def run(self) -> None:
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main() -> None:
    """Main entry point for MCP server."""
    server = GhostMCPServer()
    await server.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
