"""FastAPI service for the saved clinical disease classification ensemble."""

from functools import lru_cache
import json
from math import isclose
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from api.schemas import (
    EvidenceRequest,
    EvidenceResponse,
    ExplainRequest,
    ExplainResponse,
    GroundedExplainRequest,
    GroundedExplainResponse,
    NextQuestionRequest,
    NextQuestionResponse,
    PredictRequest,
    PredictResponse,
    SystemStatusResponse,
)
from api.observability import OBSERVABILITY
from api.status import probe_ollama
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@app.middleware("http")
async def record_request_metrics(request, call_next):
    started = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        OBSERVABILITY.record_request(
            getattr(route, "path", "unmatched"), status_code, perf_counter() - started
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

    return GroundedExplanationService(create_generator(), observer=OBSERVABILITY)


@lru_cache(maxsize=1)
def get_symptom_question_ranker():
    """Load train-split symptom frequencies and verify saved model ordering."""
    import json
    from pathlib import Path

    from src.models.question_ranking import SymptomQuestionRanker

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "question_ranking.json").read_text(encoding="utf-8"))
    statistics_path = root / config["statistics_path"]
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    predictor = get_predictor()
    feature_order = predictor.get_feature_columns()
    class_order = [str(label) for label in predictor.label_encoder.classes_.tolist()]
    if statistics.get("feature_order") != feature_order:
        raise ValueError("Question statistics do not match the saved model feature order.")
    if statistics.get("class_order") != class_order:
        raise ValueError("Question statistics do not match the saved model class order.")
    return SymptomQuestionRanker(
        feature_order=feature_order,
        class_order=class_order,
        conditional_yes=statistics.get("conditional_probability_yes", {}),
        constant_features=statistics.get("constant_features", []),
        min_information_gain_bits=float(config["min_information_gain_bits"]),
    )


@app.get("/health", tags=["system"])
def health() -> Dict[str, str]:
    """Report service readiness after confirming model artifacts can load."""
    try:
        predictor = get_predictor()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Inference model is unavailable.",
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
    started = perf_counter()
    try:
        result = get_evidence_retriever().retrieve(
            request.condition,
            context=request.context,
            top_k=request.top_k,
            strategy=request.strategy,
        )
    except Exception as error:
        OBSERVABILITY.record_retrieval(perf_counter() - started, failed=True)
        raise HTTPException(
            status_code=503,
            detail="Curated evidence retrieval is temporarily unavailable.",
        ) from error
    OBSERVABILITY.record_retrieval(perf_counter() - started, len(result.get("passages", [])))
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
    OBSERVABILITY.record_prediction(result)
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
    OBSERVABILITY.record_prediction(prediction)
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
    OBSERVABILITY.record_prediction(prediction)
    retrieval_started = perf_counter()
    try:
        retrieval = get_evidence_retriever().retrieve(
            prediction["predicted_disease"],
            context=request.symptoms,
            strategy="hybrid",
        )
    except Exception as error:
        OBSERVABILITY.record_retrieval(perf_counter() - retrieval_started, failed=True)
        raise HTTPException(
            status_code=503,
            detail="Hybrid evidence retrieval is temporarily unavailable.",
        ) from error

    OBSERVABILITY.record_retrieval(perf_counter() - retrieval_started, len(retrieval.get("passages", [])))
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


@app.post(
    "/next-question",
    response_model=NextQuestionResponse,
    tags=["inference"],
    summary="Rank optional symptom questions by estimated information gain",
)
def next_question(request: NextQuestionRequest) -> NextQuestionResponse:
    """Rank unanswered symptoms without changing the existing prediction workflow."""
    predictor = get_predictor()
    vectorizer = get_symptom_vectorizer()
    try:
        symptom_vector = vectorizer.encode(request.symptoms)
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

    excluded_answers = [
        *request.answered_negative_symptoms,
        *request.skipped_symptoms,
    ]
    valid_features = set(vectorizer.available_symptoms())
    unknown_answers = sorted(set(excluded_answers) - valid_features)
    if unknown_answers:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Questioning responses contain unknown symptoms.",
                "unknown_symptoms": unknown_answers,
            },
        )

    try:
        ranker = get_symptom_question_ranker()
        prediction = predictor.predict(symptom_vector, top_k=len(ranker.class_order))
        OBSERVABILITY.record_prediction(prediction)
        class_probabilities = {
            item["disease"]: float(item["probability"])
            for item in prediction["top_predictions"]
        }
        ranking = ranker.rank(
            class_probabilities,
            excluded_features=[*request.symptoms, *excluded_answers],
            top_k=request.top_k,
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Training-derived symptom question statistics are unavailable.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=503, detail="Symptom question ranking is unavailable.") from error

    return NextQuestionResponse(
        prediction_status={
            "predicted_disease": prediction["predicted_disease"],
            "decision_status": prediction["decision_status"],
            "insufficient_evidence": prediction["insufficient_evidence"],
            "abstention_reasons": prediction["abstention_reasons"],
            "uncertainty": prediction["uncertainty"],
        },
        ranking_method="expected_information_gain_bits",
        **ranking,
    )


@app.get("/metrics", tags=["system"], summary="Read process-local aggregate observability")
def metrics() -> Dict[str, Any]:
    return OBSERVABILITY.snapshot()


def _component(status: str, message: str, **details: Any) -> Dict[str, Any]:
    return {"status": status, "message": message, "details": details}


@app.get("/system-status", response_model=SystemStatusResponse, tags=["system"])
def system_status() -> SystemStatusResponse:
    components: Dict[str, Dict[str, Any]] = {}
    try:
        predictor = get_predictor()
        components["predictor"] = _component("ready", "Saved inference models are available.", features=predictor.n_features, classes=predictor.n_classes)
    except Exception:
        components["predictor"] = _component("unavailable", "Saved inference models are unavailable.")

    calibration_path = PROJECT_ROOT / "configs" / "ensemble_calibration.json"
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        components["calibration"] = _component("ready", "Calibration and decision policy are configured.", method=calibration.get("method", "temperature_scaling"))
    except Exception:
        components["calibration"] = _component("unavailable", "Calibration configuration is unavailable.")

    try:
        retriever = get_evidence_retriever()
        stats = retriever.stats
        components["evidence"] = _component("ready", "Curated evidence index is available.", documents=stats.get("documents"), chunks=stats.get("chunks"), default_strategy="hybrid")
    except Exception:
        components["evidence"] = _component("unavailable", "Curated evidence index is unavailable.")

    try:
        ranker = get_symptom_question_ranker()
        components["next_question"] = _component("ready", "Optional symptom question ranking is available.", features=len(ranker.feature_order), classes=len(ranker.class_order))
    except Exception:
        components["next_question"] = _component("unavailable", "Optional symptom question ranking is unavailable.")

    try:
        from rag.grounded_explanation import load_llm_settings
        components["llm"] = probe_ollama(load_llm_settings())
    except Exception:
        components["llm"] = _component("unavailable", "Optional local language model status is unavailable.")

    report_available = (PROJECT_ROOT / "docs" / "unified_evaluation_report.json").is_file()
    required = [components[name]["status"] for name in ("predictor", "calibration", "evidence")]
    overall = "unavailable" if "unavailable" in required else (
        "degraded" if any(item["status"] != "ready" for item in components.values()) else "ready"
    )
    return SystemStatusResponse(
        overall_status=overall,
        components=components,
        observability=OBSERVABILITY.snapshot(),
        evaluation_report_available=report_available,
        evaluation_report_url="/evaluation-report" if report_available else None,
        disclaimer="Development status and service telemetry only; not clinical monitoring.",
    )


@app.get("/evaluation-report", tags=["evaluation"], summary="Read the saved unified evaluation report")
def evaluation_report() -> Dict[str, Any]:
    report_path = PROJECT_ROOT / "docs" / "unified_evaluation_report.json"
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Evaluation report is not available.") from error
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail="Evaluation report is temporarily unavailable.") from error
