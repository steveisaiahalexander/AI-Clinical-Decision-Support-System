"""Pydantic request and response schemas for the inference API."""

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symptoms: List[str] = Field(min_length=1)
    top_k: int = Field(default=5, strict=True, ge=1, le=100)


class RankedPrediction(BaseModel):
    rank: int
    disease: str
    probability: float
    percentage: float


class PredictResponse(BaseModel):
    predicted_disease: str
    predicted_probability: float
    predicted_percentage: float
    top_predictions: List[RankedPrediction]
    ensemble_weights: dict[str, float]


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symptoms: List[str] = Field(min_length=1)


class FeatureAttribution(BaseModel):
    feature: str
    shap_value: float
    direction: Literal[
        "increases_class_score",
        "decreases_class_score",
        "no_change",
    ]


class ModelAttribution(BaseModel):
    model: Literal["XGBoost", "CatBoost"]
    output_scale: Literal["raw_class_score"]
    features: List[FeatureAttribution]


class ExplainResponse(BaseModel):
    predicted_disease: str
    components: List[ModelAttribution]
    explanation_note: str
