"""Prediction request and response schemas for serving surfaces."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OnlinePredictionRequest(BaseModel):
    features: list[Any] = Field(min_length=1)


class BatchPredictionRequest(BaseModel):
    inputs: list[Any] = Field(min_length=1)


class PredictionPayload(BaseModel):
    predicted_class: int
    scores: list[float]


class OnlinePredictionResponse(BaseModel):
    registry_id: str
    model_id: str
    prediction: PredictionPayload


class BatchPredictionResponse(BaseModel):
    registry_id: str
    model_id: str
    predictions: list[PredictionPayload]
    count: int