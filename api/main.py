"""FastAPI service for the saved clinical disease classification ensemble."""

from functools import lru_cache
from math import isclose
from typing import Dict

from fastapi import FastAPI, HTTPException

from api.schemas import (
    EvidenceRequest,
    EvidenceResponse,
    ExplainRequest,
    ExplainResponse,
    GroundedExplainRequest,
    GroundedExplainResponse,
    PredictRequest,
    PredictResponse,
)
from api.symptom_input import SymptomVectorizer, UnknownSymptomsError
from src.explainability.shap_explainer import EnsembleShapExplainer
from src.models.inference import ClinicalEnsemblePredictor


app = FastAPI(
    title="AI Clinical Disease Classification API",
    description=(
        "Inference-only API for the saved XGBoost/CatBoost ensemble. "
        "Predictions are for decision support and are not a medical diagnosis."
    ),
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_predictor() -> ClinicalEnsemblePredictor:
    """Load the inference artifacts once and reuse them between requests."""
    return ClinicalEnsemblePredictor()


@lru_cache(maxsize=1)
def get_symptom_vectorizer() -> SymptomVectorizer:
    """Build allowed symptoms from the model's saved feature columns."""
    return SymptomVectorizer(get_predictor().get_feature_columns())


@lru_cache(maxsize=1)
def get_shap_explainer() -> EnsembleShapExplainer:
    """Create the probability-space explainer only when requested."""
    return EnsembleShapExplainer(get_predictor())


@lru_cache(maxsize=1)
def get_evidence_retriever():
    """Build the separate evidence index only when evidence is requested."""
    from rag.retrieval import EvidenceRetriever

    return EvidenceRetriever()


@lru_cache(maxsize=1)
def get_grounded_explanation_service():
    """Configure optional local LLM synthesis independently from classifier inference."""
    from rag.grounded_explanation import GroundedExplanationService, create_generator

    return GroundedExplanationService(create_generator())


@app.get("/health", tags=["system"])
def health() -> Dict[str, str]:
    """Report service readiness after confirming model artifacts can load."""
    try:
        predictor = get_predictor()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Inference model is unavailable: {error}",
        ) from error

    return {
        "status": "healthy",
        "model": "XGBoost + CatBoost probability ensemble",
        "features": str(predictor.n_features),
        "classes": str(predictor.n_classes),
    }


@app.get("/symptoms", tags=["inference"])
def list_symptoms() -> Dict[str, object]:
    """List valid symptom names in the saved model feature order."""
    symptoms = get_symptom_vectorizer().available_symptoms()
    return {"count": len(symptoms), "symptoms": symptoms}


@app.post(
    "/evidence",
    response_model=EvidenceResponse,
    tags=["evidence"],
    summary="Retrieve attributed context for a supported model class",
)
def retrieve_evidence(request: EvidenceRequest) -> EvidenceResponse:
    """Retrieve source passages without changing or re-running the classifier."""
    try:
        result = get_evidence_retriever().retrieve(
            request.condition,
            context=request.context,
            top_k=request.top_k,
            strategy=request.strategy,
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Curated evidence retrieval is temporarily unavailable.",
        ) from error
    return EvidenceResponse(**result)


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Predict a disease class from selected symptoms",
)
def predict(request: PredictRequest) -> PredictResponse:
    """Encode selected symptoms, then call the inference-only model layer."""
    predictor = get_predictor()
    if request.top_k > predictor.n_classes:
        raise HTTPException(
            status_code=422,
            detail=f"top_k cannot exceed the {predictor.n_classes} supported classes.",
        )

    try:
        symptom_vector = get_symptom_vectorizer().encode(request.symptoms)
    except UnknownSymptomsError as error:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Request contains unknown symptoms.",
                "unknown_symptoms": error.symptoms,
            },
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    result = predictor.predict(symptom_vector, top_k=request.top_k)
    return PredictResponse(**result)


@app.post(
    "/explain",
    response_model=ExplainResponse,
    tags=["explainability"],
    summary="Attribute the ensemble prediction to selected symptoms",
)
def explain(request: ExplainRequest) -> ExplainResponse:
    """Explain the current ensemble's top class in probability space."""
    predictor = get_predictor()
    try:
        symptom_vector = get_symptom_vectorizer().encode(request.symptoms)
    except UnknownSymptomsError as error:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Request contains unknown symptoms.",
                "unknown_symptoms": error.symptoms,
            },
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    prediction = predictor.predict(symptom_vector, top_k=1)
    try:
        class_label = predictor.label_encoder.transform(
            [prediction["predicted_disease"]]
        )[0]
        explanation = get_shap_explainer().explain(symptom_vector, class_label)
        if not isclose(
            explanation["predicted_probability"],
            prediction["raw_predicted_probability"],
            rel_tol=1e-7,
            abs_tol=1e-8,
        ):
            raise ValueError("Explanation target does not match the ensemble prediction.")
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Feature attributions are temporarily unavailable for the saved models.",
        ) from error

    return ExplainResponse(
        predicted_disease=prediction["predicted_disease"],
        **explanation,
        explanation_note=(
            "Attributions are for the raw weighted-ensemble probability before calibration. "
            "Shapley values allocate the difference between the ensemble's predicted-class "
            "probability and its all-symptoms-absent reference across the selected symptoms. "
            "Component values are each model's probability attributions; their weighted sum "
            "is the ensemble attribution. These describe model behavior, not causal effects "
            "or a clinical explanation."
            if explanation["is_exact"]
            else "Sampled permutation Shapley values approximate the allocation between the "
            "ensemble's predicted-class probability and its all-symptoms-absent reference. "
            "Component values are each model's probability attributions; their weighted sum "
            "is the ensemble attribution. These describe model behavior, not causal effects "
            "or a clinical explanation."
        ),
    )


@app.post(
    "/explain-grounded",
    response_model=GroundedExplainResponse,
    tags=["evidence"],
    summary="Generate an optional source-grounded explanation after hybrid retrieval",
)
def explain_grounded(request: GroundedExplainRequest) -> GroundedExplainResponse:
    """Re-run the saved prediction, retrieve hybrid evidence, then optionally synthesize."""
    predictor = get_predictor()
    try:
        symptom_vector = get_symptom_vectorizer().encode(request.symptoms)
    except UnknownSymptomsError as error:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Request contains unknown symptoms.",
                "unknown_symptoms": error.symptoms,
            },
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    prediction = predictor.predict(symptom_vector, top_k=1)
    try:
        retrieval = get_evidence_retriever().retrieve(
            prediction["predicted_disease"],
            context=request.symptoms,
            strategy="hybrid",
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Hybrid evidence retrieval is temporarily unavailable.",
        ) from error

    try:
        explanation_result = get_grounded_explanation_service().explain(
            prediction,
            request.symptoms,
            retrieval,
        )
        return GroundedExplainResponse(
            prediction=PredictResponse(**prediction),
            **explanation_result,
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Grounded explanation is temporarily unavailable; retrieved evidence remains available.",
        ) from error
