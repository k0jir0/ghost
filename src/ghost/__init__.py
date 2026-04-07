"""Ghost - AI Model Context & Training Platform.

A platform combining PyTorch, TensorFlow, MCP, and Ollama for intelligent ML training.
"""

__version__ = "1.0.0"
__author__ = "McGill Software"

from ghost.config import GhostConfig
from ghost.context import ModelContext
from ghost.datasets import DatasetResolver, DatasetSpec
from ghost.health_monitor import HealthMonitor
from ghost.orchestration import (
    TrainingOrchestrator,
    TrainingRunRecord,
    TrainingRunRequest,
)
from ghost.planning import PlanningRequest, TrainingPlan, TrainingPlanner
from ghost.tool_catalog import ToolCatalog, ToolSpec
from ghost.training import TrainingPipeline

__all__ = [
    "DatasetResolver",
    "DatasetSpec",
    "GhostConfig",
    "HealthMonitor",
    "ModelContext",
    "TrainingOrchestrator",
    "PlanningRequest",
    "TrainingPlan",
    "TrainingPlanner",
    "TrainingRunRecord",
    "TrainingRunRequest",
    "TrainingPipeline",
    "ToolCatalog",
    "ToolSpec",
    "__version__",
]
