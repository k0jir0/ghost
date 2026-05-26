"""Optional HTTP serving surface for Ghost inference."""

from __future__ import annotations

from typing import Any

from ghost.inference import InferenceService
from ghost.prediction_schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    OnlinePredictionRequest,
    OnlinePredictionResponse,
)


def create_serving_app(inference_service: InferenceService | None = None) -> Any:
    """Create a FastAPI app when FastAPI is installed."""

    try:
        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover - optional dependency surface
        raise RuntimeError(
            "FastAPI is not installed. Install it to run the Ghost serving API."
        ) from exc

    service = inference_service or InferenceService()
    app = FastAPI(title="Ghost Serving API", version="1.0")

    @app.post("/v1/models/{registry_id}:predict", response_model=OnlinePredictionResponse)
    async def predict_online(
        registry_id: str,
        request: OnlinePredictionRequest,
    ) -> OnlinePredictionResponse:
        payload = await service.predict_online(registry_id, request.features)
        return OnlinePredictionResponse.model_validate(payload)

    @app.post(
        "/v1/models/{registry_id}:predict-batch",
        response_model=BatchPredictionResponse,
    )
    async def predict_batch(
        registry_id: str,
        request: BatchPredictionRequest,
    ) -> BatchPredictionResponse:
        payload = await service.predict_batch(registry_id, request.inputs)
        return BatchPredictionResponse.model_validate(payload)

    return app