"""Pydantic request and response schemas for the inference API."""

from typing import List, Literal, Optional

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
        "increases_probability",
        "decreases_probability",
        "no_change",
    ]


class ComponentAttribution(BaseModel):
    model: Literal["XGBoost model attribution", "CatBoost model attribution"]
    class_index: int
    weight: float
    output_space: Literal["model_probability"]
    baseline_probability: float
    predicted_probability: float
    features: List[FeatureAttribution]


class ExplainResponse(BaseModel):
    predicted_disease: str
    predicted_class_index: int
    predicted_probability: float
    baseline_probability: float
    output_space: Literal["ensemble_probability"]
    method: Literal["exact_coalition_shapley", "sampled_permutation_shapley"]
    is_exact: bool
    permutation_count: Optional[int]
    features: List[FeatureAttribution]
    component_attributions: List[ComponentAttribution]
    explanation_note: str
