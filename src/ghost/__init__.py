"""Ghost - AI Model Context & Training Platform.

A platform combining PyTorch, TensorFlow, MCP, and Ollama for intelligent ML training.
"""

__version__ = "1.0.0"
__author__ = "McGill Software"

from ghost.config import GhostConfig
from ghost.context import ModelContext
from ghost.training import TrainingPipeline

__all__ = [
    "GhostConfig",
    "ModelContext",
    "TrainingPipeline",
    "__version__",
]
