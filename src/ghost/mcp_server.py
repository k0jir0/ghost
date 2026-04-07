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
    TextContent,
    Tool,
)
from pydantic import ValidationError

from ghost.context import ContextManager
from ghost.health_monitor import HealthMonitor
from ghost.logging import get_logger
from ghost.ollama_client import OllamaClient
from ghost.pytorch_ops import PyTorchOps
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
    ):
        """Initialize the MCP server."""
        self.server = Server("ghost-mcp")
        self.context_manager = context_manager or ContextManager()
        self.ollama_client = ollama_client or OllamaClient()
        self.health_monitor = health_monitor or HealthMonitor()
        self.pytorch_ops = PyTorchOps(self.context_manager)
        self.tensorflow_ops = TensorFlowOps(self.context_manager)
        self.tool_catalog = _DEFAULT_TOOL_CATALOG

        self._register_tools()

    def _spec_to_tool(self, spec: ToolSpec) -> Tool:
        return Tool(
            name=spec.name,
            description=spec.description,
            inputSchema=spec.input_schema(),
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
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                    isError=True,
                )
            try:
                spec.input_model.model_validate(arguments or {})
            except ValidationError as exc:
                error_msg = f"Invalid arguments for '{name}': {exc.errors()}"
                logger.warning("tool_validation_error", tool=name, errors=exc.errors())
                return CallToolResult(
                    content=[TextContent(type="text", text=error_msg)],
                    isError=True,
                )

            try:
                result = await self._handle_tool(name, arguments or {})
                return CallToolResult(
                    content=[TextContent(type="text", text=str(result))],
                    isError=False,
                )
            except Exception as e:
                logger.error("tool_error", tool=name, error=str(e))
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    isError=True,
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
