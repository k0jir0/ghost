"""Ghost - AI Model Context & Training Platform.

A platform combining PyTorch, TensorFlow, MCP, and Ollama for intelligent ML training.
"""

__version__ = "1.0.0"
__author__ = "McGill Software"

from ghost.config import GhostConfig
from ghost.context import ModelContext
from ghost.datasets import DatasetResolver, DatasetSpec
from ghost.deployment import DeploymentManager, DeploymentRecord
from ghost.feature_store import FeatureDefinition, FeatureStore
from ghost.health_monitor import HealthMonitor
from ghost.metadata_store import MetadataStore, SQLiteMetadataBackend
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
    "DeploymentManager",
    "DeploymentRecord",
    "FeatureDefinition",
    "FeatureStore",
    "GhostConfig",
    "HealthMonitor",
    "MetadataStore",
    "ModelContext",
    "SQLiteMetadataBackend",
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
