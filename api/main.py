"""FastAPI service for the saved clinical disease classification ensemble."""

from functools import lru_cache
from typing import Dict

from fastapi import FastAPI, HTTPException

from api.schemas import PredictRequest, PredictResponse
from api.symptom_input import SymptomVectorizer, UnknownSymptomsError
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
