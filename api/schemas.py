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


class PredictionUncertainty(BaseModel):
    top_probability: float
    top_two_margin: float
    entropy: float
    normalized_entropy: float


class PredictResponse(BaseModel):
    predicted_disease: str
    predicted_probability: float
    predicted_percentage: float
    raw_predicted_probability: float
    top_predictions: List[RankedPrediction]
    probability_status: Literal["calibrated", "raw"]
    calibration_method: Literal["temperature_scaling", "raw_ensemble"]
    calibration_temperature: float
    uncertainty: PredictionUncertainty
    decision_status: Literal["ranked_prediction", "insufficient_evidence"]
    insufficient_evidence: bool
    abstention_reasons: List[str]
    abstention_thresholds: dict[str, float]
    thresholds_clinically_validated: Literal[False]
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


class EvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: str = Field(min_length=1, max_length=120)
    context: List[str] = Field(default_factory=list, max_length=30)
    top_k: Optional[int] = Field(default=None, strict=True, ge=1, le=5)


class EvidenceSource(BaseModel):
    title: str
    organization: str
    url: str
    attribution: str
    license: str
    rights_url: str
    accessed_on: str
    last_updated: Optional[str] = None
    last_reviewed: Optional[str] = None


class EvidencePassage(BaseModel):
    text: str
    section: str
    source: EvidenceSource
    relevance_score: float


class EvidenceResponse(BaseModel):
    condition: str
    supported: bool
    retrieval_method: Literal["tfidf_cosine"]
    passages: List[EvidencePassage]
    disclaimer: str
