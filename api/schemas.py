"""Pydantic request and response schemas for the inference API."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    strategy: Optional[Literal["tfidf", "semantic", "hybrid"]] = None


class EvidenceSource(BaseModel):
    title: str
    organization: str
    url: str
    attribution: str
    license: str
    rights_url: str
    accessed_on: str
    publication_date: Optional[str] = None
    last_updated: Optional[str] = None
    last_reviewed: Optional[str] = None


class EvidencePassage(BaseModel):
    chunk_id: Optional[str] = None
    text: str
    section: str
    source: EvidenceSource
    relevance_score: float


class EvidenceResponse(BaseModel):
    condition: str
    supported: bool
    sufficient_evidence: bool = False
    status: Literal[
        "evidence_available",
        "no_sufficient_evidence",
        "unsupported_condition",
    ] = "evidence_available"
    retrieval_method: Literal[
        "tfidf_cosine",
        "semantic_cosine",
        "hybrid_weighted_rrf",
    ]
    passages: List[EvidencePassage]
    disclaimer: str


class GroundedExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symptoms: List[str] = Field(min_length=1)


class GroundedClaim(BaseModel):
    text: str
    evidence_ids: List[str] = Field(min_length=1)


class GroundedExplanation(BaseModel):
    summary: str
    claims: List[GroundedClaim] = Field(min_length=1)
    grounding_status: Literal["grounded"]


class GroundedEvidence(BaseModel):
    evidence_id: str
    passage: EvidencePassage


class GroundedExplainResponse(BaseModel):
    prediction: PredictResponse
    explanation: Optional[GroundedExplanation] = None
    grounding_status: Literal[
        "grounded",
        "abstained",
        "insufficient_evidence",
        "llm_unavailable",
        "validation_failed",
    ]
    evidence: List[GroundedEvidence]
    status_message: str
    disclaimer: str


class NextQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symptoms: List[str] = Field(min_length=1, max_length=174)
    answered_negative_symptoms: List[str] = Field(default_factory=list, max_length=174)
    skipped_symptoms: List[str] = Field(default_factory=list, max_length=174)
    top_k: int = Field(default=3, strict=True, ge=1, le=10)

    @model_validator(mode="after")
    def validate_answer_sets(self):
        groups = (
            self.symptoms,
            self.answered_negative_symptoms,
            self.skipped_symptoms,
        )
        if any(len(values) != len(set(values)) for values in groups):
            raise ValueError("Symptom identifiers must not be duplicated.")
        if (
            set(self.symptoms) & set(self.answered_negative_symptoms)
            or set(self.symptoms) & set(self.skipped_symptoms)
            or set(self.answered_negative_symptoms) & set(self.skipped_symptoms)
        ):
            raise ValueError("A symptom can have only one current questioning response.")
        return self


class QuestionPredictionStatus(BaseModel):
    predicted_disease: str
    decision_status: Literal["ranked_prediction", "insufficient_evidence"]
    insufficient_evidence: bool
    abstention_reasons: List[str]
    uncertainty: PredictionUncertainty


class NextQuestionSuggestion(BaseModel):
    feature_id: str
    label: str
    information_gain_bits: float
    expected_entropy_after_answer_bits: float
    rationale: str


class NextQuestionResponse(BaseModel):
    prediction_status: QuestionPredictionStatus
    ranking_method: Literal["expected_information_gain_bits"]
    prior_entropy_bits: float
    eligible_candidate_count: int
    useful_candidate_count: int
    status: Literal[
        "suggestions_available",
        "no_useful_candidate",
        "no_unasked_candidates",
        "no_eligible_candidates",
    ]
    status_message: str
    suggestions: List[NextQuestionSuggestion]
    disclaimer: str
