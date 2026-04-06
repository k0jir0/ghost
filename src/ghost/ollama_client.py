"""Ollama client for Ghost platform.

Provides integration with local Ollama LLM for training assistance and recommendations.
"""

from __future__ import annotations

from typing import Any

import ollama

from ghost.config import get_config
from ghost.logging import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Client for Ollama local LLM integration."""

    def __init__(self, host: str | None = None, model: str | None = None):
        """Initialize Ollama client.
        
        Args:
            host: Ollama server URL (default from config)
            model: Default model name (default from config)
        """
        config = get_config()
        self.host = host or config.ollama_host
        self.model = model or config.ollama_model
        self.timeout = config.ollama_timeout
        logger.info("ollama_client_init", host=self.host, model=self.model)

    def chat(self, message: str, system: str | None = None) -> str:
        """Send a chat message to Ollama.
        
        Args:
            message: User message
            system: Optional system prompt
        
        Returns:
            LLM response text
        """
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": message})
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={"timeout": self.timeout}
            )
            
            return response["message"]["content"]
        except Exception as e:
            logger.error("ollama_chat_error", error=str(e))
            return f"Error: {str(e)}"

    async def get_recommendation(
        self,
        task: str,
        dataset: str = "",
    ) -> dict[str, Any]:
        """Get model training recommendations from Ollama.
        
        Args:
            task: Training task description
            dataset: Dataset description
        
        Returns:
            Recommendation dictionary
        """
        try:
            system_prompt = """You are an expert ML training assistant. Based on the training task and dataset, 
provide recommendations for model architecture, hyperparameters, and training strategy. 
Respond in JSON format with keys: architecture, learning_rate, batch_size, epochs, 
optimizer, and tips (array of strings)."""

            user_prompt = f"""Training Task: {task}
Dataset: {dataset if dataset else 'Not specified'}

Provide recommendations for training this model."""

            response = self.chat(user_prompt, system=system_prompt)
            
            # Try to parse as JSON
            try:
                import json
                # Extract JSON from response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                else:
                    json_str = response
                recommendations = json.loads(json_str)
            except Exception:
                # Return as text if JSON parsing fails
                recommendations = {"recommendation": response}
            
            return {
                "status": "success",
                "model": self.model,
                "recommendations": recommendations,
            }
        except Exception as e:
            logger.error("recommendation_error", error=str(e))
            return {"status": "error", "message": str(e)}

    async def analyze_training_progress(
        self,
        metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze training progress and provide suggestions.
        
        Args:
            metrics: List of training metrics
        
        Returns:
            Analysis and suggestions
        """
        try:
            metrics_summary = "\n".join([
                f"Step {m.get('step', i)}: loss={m.get('loss', 'N/A')}, "
                f"accuracy={m.get('accuracy', 'N/A')}"
                for i, m in enumerate(metrics[-10:])
            ])
            
            system_prompt = """You are an expert ML training assistant. Analyze the training progress 
and provide actionable suggestions. Respond in JSON format with keys: status 
(good/warning/concerning), analysis, and suggestions (array of strings)."""

            user_prompt = f"""Recent training metrics:
{metrics_summary}

Provide analysis and suggestions for improving training."""

            response = self.chat(user_prompt, system=system_prompt)
            
            try:
                import json
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                else:
                    json_str = response
                analysis = json.loads(json_str)
            except Exception:
                analysis = {"analysis": response}
            
            return {
                "status": "success",
                "analysis": analysis,
            }
        except Exception as e:
            logger.error("analysis_error", error=str(e))
            return {"status": "error", "message": str(e)}

    def generate_training_code(
        self,
        task: str,
        framework: str = "pytorch",
    ) -> str:
        """Generate training code snippet.
        
        Args:
            task: Training task description
            framework: ML framework (pytorch or tensorflow)
        
        Returns:
            Code snippet
        """
        system_prompt = f"""You are an expert {framework} developer. Generate a clean, well-commented 
training code snippet based on the task description. Include model creation, 
training loop, and evaluation. Use best practices."""

        user_prompt = f"""Generate {framework} training code for: {task}"""

        return self.chat(user_prompt, system=system_prompt)

    def check_connection(self) -> bool:
        """Check if Ollama server is reachable.
        
        Returns:
            True if connection successful
        """
        try:
            ollama.ps()
            return True
        except Exception as e:
            logger.warning("ollama_connection_failed", error=str(e))
            return False
