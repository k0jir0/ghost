"""Registry-backed inference service for Ghost."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from ghost.config import GhostConfig, get_config
from ghost.context import BackendType, ContextManager
from ghost.model_registry import ModelRegistry
from ghost.observability import ModelObservability
from ghost.pytorch_ops import PyTorchOps
from ghost.tensorflow_ops import TensorFlowOps


class InferenceService:
    """Load registry-approved models and serve batch or online predictions."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        context_manager: ContextManager | None = None,
        model_registry: ModelRegistry | None = None,
        backend_ops: dict[BackendType, Any] | None = None,
        observability: ModelObservability | None = None,
    ):
        self.config = config or get_config()
        self.context_manager = context_manager or ContextManager()
        self.model_registry = model_registry or ModelRegistry(config=self.config)
        self.observability = observability or ModelObservability(config=self.config)
        self._backend_ops = backend_ops or {
            BackendType.PYTORCH: PyTorchOps(self.context_manager),
            BackendType.TENSORFLOW: TensorFlowOps(self.context_manager),
        }

    async def predict_online(
        self,
        registry_id: str,
        features: list[Any],
    ) -> dict[str, Any]:
        started = perf_counter()
        record = await self._approved_record(registry_id)
        predictions = await self._predict(record, [features])
        self.observability.record_prediction(
            registry_id,
            record.model_id,
            latency_ms=(perf_counter() - started) * 1000.0,
            batch_size=1,
            success=True,
            inputs=[features],
            predictions=predictions,
        )
        return {
            "registry_id": registry_id,
            "model_id": record.model_id,
            "prediction": predictions[0],
        }

    async def predict_batch(
        self,
        registry_id: str,
        inputs: list[Any],
    ) -> dict[str, Any]:
        started = perf_counter()
        record = await self._approved_record(registry_id)
        predictions = await self._predict(record, inputs)
        self.observability.record_prediction(
            registry_id,
            record.model_id,
            latency_ms=(perf_counter() - started) * 1000.0,
            batch_size=len(inputs),
            success=True,
            inputs=inputs,
            predictions=predictions,
        )
        return {
            "registry_id": registry_id,
            "model_id": record.model_id,
            "predictions": predictions,
            "count": len(predictions),
        }

    async def _approved_record(self, registry_id: str) -> Any:
        record = self.model_registry.get_model(registry_id)
        if record is None:
            raise KeyError(f"Unknown registry id: {registry_id}")
        if record.stage not in {"staging", "production"}:
            raise ValueError("Model must be promoted before it can serve predictions")
        if record.evaluation_status != "passed":
            raise ValueError("Model did not pass evaluation and cannot be served")
        return record

    async def _predict(self, record: Any, inputs: list[Any]) -> list[dict[str, Any]]:
        model = await self._ensure_runtime_model(record)
        input_shape = [int(value) for value in record.metadata.get("input_shape", [])]
        batch = self._prepare_batch(inputs, input_shape, record.backend)
        if record.backend == BackendType.PYTORCH.value:
            return self._predict_pytorch(model, batch)
        if record.backend == BackendType.TENSORFLOW.value:
            return self._predict_tensorflow(model, batch)
        raise ValueError(f"Unsupported backend for inference: {record.backend}")

    async def _ensure_runtime_model(self, record: Any) -> Any:
        serving_model_id = self._serving_model_id(record.registry_id)
        backend = BackendType(record.backend)
        ops = self._backend_ops[backend]
        models = getattr(ops, "models", {})
        if isinstance(models, dict) and serving_model_id in models:
            return models[serving_model_id]

        create_result = await ops.create_model(
            model_id=serving_model_id,
            model_name=record.model_id,
            architecture=record.architecture,
            num_classes=int(record.metadata.get("num_classes", 1) or 1),
            input_shape=[int(value) for value in record.metadata.get("input_shape", [])],
        )
        if create_result.get("status") != "success":
            raise RuntimeError(create_result.get("message", "Model creation failed"))

        artifact_uri = str(record.metadata.get("artifact_uri", ""))
        load_result = await ops.load_checkpoint(serving_model_id, artifact_uri)
        if load_result.get("status") != "success":
            raise RuntimeError(load_result.get("message", "Checkpoint load failed"))

        model = getattr(ops, "models", {}).get(serving_model_id)
        if model is None:
            raise RuntimeError("Backend did not expose a runtime model after load")
        return model

    def _prepare_batch(
        self,
        inputs: list[Any],
        input_shape: list[int],
        backend: str,
    ) -> np.ndarray:
        samples = [self._prepare_sample(sample, input_shape, backend) for sample in inputs]
        return np.stack(samples).astype(np.float32)

    def _prepare_sample(
        self,
        sample: Any,
        input_shape: list[int],
        backend: str,
    ) -> np.ndarray:
        array = np.asarray(sample, dtype=np.float32)
        if not input_shape:
            return array

        expected_shape = tuple(input_shape)
        if array.shape == expected_shape:
            return array

        if len(input_shape) == 3:
            channels_first = (input_shape[0], input_shape[1], input_shape[2])
            channels_last = (input_shape[1], input_shape[2], input_shape[0])
            if backend == BackendType.PYTORCH.value and array.shape == channels_last:
                return np.transpose(array, (2, 0, 1))
            if backend == BackendType.TENSORFLOW.value and array.shape == channels_first:
                return np.transpose(array, (1, 2, 0))

        if array.size == int(np.prod(expected_shape)):
            return array.reshape(expected_shape)

        raise ValueError(
            f"Input shape {array.shape} does not match expected shape {expected_shape}"
        )

    def _predict_pytorch(self, model: Any, batch: np.ndarray) -> list[dict[str, Any]]:
        device = torch.device("cpu")
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            first_parameter = next(parameters(), None)
            if first_parameter is not None:
                device = first_parameter.device
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(batch).float().to(device)
            if callable(model):
                logits = model(tensor)
            elif hasattr(model, "forward"):
                logits = model.forward(tensor)
            else:
                raise TypeError("Loaded PyTorch model is not callable")
            scores = torch.softmax(logits, dim=1).cpu().numpy()
        return self._prediction_payloads(scores)

    def _predict_tensorflow(self, model: Any, batch: np.ndarray) -> list[dict[str, Any]]:
        predictions = model.predict(batch, verbose=0)
        scores = self._softmax(np.asarray(predictions, dtype=np.float32))
        return self._prediction_payloads(scores)

    def _prediction_payloads(self, scores: np.ndarray) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for row in scores:
            payloads.append(
                {
                    "predicted_class": int(np.argmax(row)),
                    "scores": [float(value) for value in row.tolist()],
                }
            )
        return payloads

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def _serving_model_id(self, registry_id: str) -> str:
        safe_registry_id = registry_id.replace("/", "-")
        return f"serve__{safe_registry_id}"