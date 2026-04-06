"""MCP Server implementation for Ghost.

Provides Model Context Protocol tools for ML training and inference operations.
All tool arguments are validated with Pydantic before reaching backend ops,
so malformed inputs surface as structured errors rather than deep stack traces.
"""

from __future__ import annotations

from typing import Any, Literal

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    ListToolsResult,
)
from pydantic import BaseModel, Field, ValidationError

from ghost.context import ContextManager, BackendType, ModelState
from ghost.health_monitor import HealthMonitor
from ghost.pytorch_ops import PyTorchOps
from ghost.tensorflow_ops import TensorFlowOps
from ghost.ollama_client import OllamaClient
from ghost.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic argument models — shift-left validation at the MCP boundary
# ---------------------------------------------------------------------------

_Architecture = Literal["resnet18", "resnet50", "mlp", "custom"]


class CreateModelArgs(BaseModel):
    model_id: str
    model_name: str
    architecture: _Architecture = "mlp"
    num_classes: int = Field(default=10, ge=1, le=10_000)
    input_shape: list[int] = Field(default=[3, 224, 224])


class TrainStepArgs(BaseModel):
    model_id: str
    batch_size: int = Field(default=32, ge=1, le=4096)
    learning_rate: float = Field(default=0.001, gt=0.0, le=10.0)


class EvaluateArgs(BaseModel):
    model_id: str


class SaveCheckpointArgs(BaseModel):
    model_id: str
    path: str | None = None


class LoadCheckpointArgs(BaseModel):
    model_id: str
    path: str


class GetTrainingStatusArgs(BaseModel):
    model_id: str


class GetModelRecommendationArgs(BaseModel):
    task: str
    dataset: str = ""


class ListModelsArgs(BaseModel):
    pass


class GetSystemHealthArgs(BaseModel):
    pass


# ---------------------------------------------------------------------------
# Tool → argument model registry
# ---------------------------------------------------------------------------

_TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "pytorch_create_model": CreateModelArgs,
    "pytorch_train_step": TrainStepArgs,
    "pytorch_evaluate": EvaluateArgs,
    "pytorch_save_checkpoint": SaveCheckpointArgs,
    "pytorch_load_checkpoint": LoadCheckpointArgs,
    "tensorflow_create_model": CreateModelArgs,
    "tensorflow_train_step": TrainStepArgs,
    "tensorflow_evaluate": EvaluateArgs,
    "tensorflow_save_checkpoint": SaveCheckpointArgs,
    "tensorflow_load_checkpoint": LoadCheckpointArgs,
    "get_training_status": GetTrainingStatusArgs,
    "list_models": ListModelsArgs,
    "get_system_health": GetSystemHealthArgs,
    "get_model_recommendation": GetModelRecommendationArgs,
}


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

        self._register_tools()

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available MCP tools."""
            return ListToolsResult(
                tools=[
                    # PyTorch Tools
                    Tool(
                        name="pytorch_create_model",
                        description="Create a new PyTorch model",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "model_id": {"type": "string"},
                                "model_name": {"type": "string"},
                                "architecture": {
                                    "type": "string",
                                    "enum": ["resnet18", "resnet50", "mlp", "custom"],
                                },
                                "num_classes": {"type": "integer", "default": 10},
                                "input_shape": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "default": [3, 224, 224],
                                },
                            },
                            "required": ["model_id", "model_name", "architecture"],
                        },
                    ),
                    Tool(
                        name="pytorch_train_step",
                        description="Execute one training step",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "model_id": {"type": "string"},
                                "batch_size": {"type": "integer", "default": 32},
                                "learning_rate": {"type": "number", "default": 0.001},
                            },
                            "required": ["model_id"],
                        },
                    ),
                    Tool(
                        name="pytorch_evaluate",
                        description="Evaluate PyTorch model on dataset",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "model_id": {"type": "string"},
                            },
                            "required": ["model_id"],
                        },
                    ),
                    Tool(
                        name="pytorch_save_checkpoint",
                        description="Save PyTorch model checkpoint",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "model_id": {"type": "string"},
                                "path": {"type": "string"},
                            },
                            "required": ["model_id"],
                        },
                    ),
                    Tool(
                        name="pytorch_load_checkpoint",
                        description="Load PyTorch model checkpoint",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "model_id": {"type": "string"},
                                "path": {"type": "string"},
                            },
                            "required": ["model_id", "path"],
                        },
                    ),
                    # TensorFlow Tools
                    Tool(
                        name="tensorflow_create_model",
                        description="Create a new TensorFlow/Keras model",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "model_id": {"type": "string"},
                                "model_name": {"type": "string"},
                                "architecture": {
                                    "type": "string",
                                    "enum": ["resnet18", "resnet50", "mlp", "custom"],
                                },
                                "num_classes": {"type": "integer", "default": 10},
                                "input_shape": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "default": [224, 224, 3],
                                },
                            },
                            "required": ["model_id", "model_name", "architecture"],
                        },
                    ),
                    Tool(
                        name="tensorflow_train_step",
                        description="Execute one training step",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "model_id": {"type": "string"},
                                "batch_size": {"type": "integer", "default": 32},
                                "learning_rate": {"type": "number", "default": 0.001},
                            },
                            "required": ["model_id"],
                        },
                    ),
                    Tool(
                        name="tensorflow_evaluate",
                        description="Evaluate TensorFlow model on dataset",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "model_id": {"type": "string"},
                            },
                            "required": ["model_id"],
                        },
                    ),
                    Tool(
                        name="tensorflow_save_checkpoint",
                        description="Save TensorFlow model checkpoint",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "model_id": {"type": "string"},
                                "path": {"type": "string"},
                            },
                            "required": ["model_id"],
                        },
                    ),
                    Tool(
                        name="tensorflow_load_checkpoint",
                        description="Load TensorFlow model checkpoint",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "model_id": {"type": "string"},
                                "path": {"type": "string"},
                            },
                            "required": ["model_id", "path"],
                        },
                    ),
                    # Training Tools
                    Tool(
                        name="get_training_status",
                        description="Get current training status for a model",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "model_id": {"type": "string"},
                            },
                            "required": ["model_id"],
                        },
                    ),
                    Tool(
                        name="list_models",
                        description="List all registered models",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="get_system_health",
                        description="Inspect resource health, thresholds, and cache usage",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="get_model_recommendation",
                        description="Get Ollama-powered model training recommendations",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"},
                                "dataset": {"type": "string"},
                            },
                            "required": ["task"],
                        },
                    ),
                ]
            )

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> CallToolResult:
            """Handle tool calls with upfront Pydantic validation."""
            # --- Shift-left: validate arguments before any business logic ---
            model_cls = _TOOL_ARG_MODELS.get(name)
            if model_cls is None:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                    isError=True,
                )
            try:
                model_cls.model_validate(arguments or {})
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

    async def _handle_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Route tool calls to appropriate handlers."""

        # PyTorch operations
        if name == "pytorch_create_model":
            return await self.pytorch_ops.create_model(
                model_id=arguments["model_id"],
                model_name=arguments["model_name"],
                architecture=arguments.get("architecture", "mlp"),
                num_classes=arguments.get("num_classes", 10),
                input_shape=arguments.get("input_shape", [3, 224, 224]),
            )

        if name == "pytorch_train_step":
            return await self.pytorch_ops.train_step(
                model_id=arguments["model_id"],
                batch_size=arguments.get("batch_size", 32),
                learning_rate=arguments.get("learning_rate", 0.001),
            )

        if name == "pytorch_evaluate":
            return await self.pytorch_ops.evaluate(arguments["model_id"])

        if name == "pytorch_save_checkpoint":
            return await self.pytorch_ops.save_checkpoint(
                model_id=arguments["model_id"],
                path=arguments.get("path"),
            )

        if name == "pytorch_load_checkpoint":
            return await self.pytorch_ops.load_checkpoint(
                model_id=arguments["model_id"],
                path=arguments["path"],
            )

        # TensorFlow operations
        if name == "tensorflow_create_model":
            return await self.tensorflow_ops.create_model(
                model_id=arguments["model_id"],
                model_name=arguments["model_name"],
                architecture=arguments.get("architecture", "mlp"),
                num_classes=arguments.get("num_classes", 10),
                input_shape=arguments.get("input_shape", [224, 224, 3]),
            )

        if name == "tensorflow_train_step":
            return await self.tensorflow_ops.train_step(
                model_id=arguments["model_id"],
                batch_size=arguments.get("batch_size", 32),
                learning_rate=arguments.get("learning_rate", 0.001),
            )

        if name == "tensorflow_evaluate":
            return await self.tensorflow_ops.evaluate(arguments["model_id"])

        if name == "tensorflow_save_checkpoint":
            return await self.tensorflow_ops.save_checkpoint(
                model_id=arguments["model_id"],
                path=arguments.get("path"),
            )

        if name == "tensorflow_load_checkpoint":
            return await self.tensorflow_ops.load_checkpoint(
                model_id=arguments["model_id"],
                path=arguments["path"],
            )

        # Training tools
        if name == "get_training_status":
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

        if name == "list_models":
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

        if name == "get_system_health":
            return self.health_monitor.get_health_report()

        if name == "get_model_recommendation":
            return await self.ollama_client.get_recommendation(
                task=arguments["task"],
                dataset=arguments.get("dataset", ""),
            )

        return {"error": f"Unknown tool: {name}"}

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
