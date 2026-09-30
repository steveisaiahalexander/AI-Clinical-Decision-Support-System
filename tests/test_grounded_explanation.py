import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from api.main import explain_grounded
from api.schemas import (
    GroundedExplainRequest,
    GroundedExplainResponse,
    PredictResponse,
)
from rag.grounded_explanation import (
    EXPLANATION_DISCLAIMER,
    GenerationUnavailable,
    GroundedExplanationService,
    OllamaStructuredGenerator,
    load_llm_settings,
)
from scripts.evaluate_grounded_explanations import run_evaluation


def prediction(decision_status="ranked_prediction"):
    abstained = decision_status == "insufficient_evidence"
    return {
        "predicted_disease": "Asthma",
        "predicted_probability": 0.68,
        "predicted_percentage": 68.0,
        "raw_predicted_probability": 0.65,
        "top_predictions": [
            {"rank": 1, "disease": "Asthma", "probability": 0.68, "percentage": 68.0}
        ],
        "probability_status": "calibrated",
        "calibration_method": "temperature_scaling",
        "calibration_temperature": 1.2,
        "uncertainty": {
            "top_probability": 0.68,
            "top_two_margin": 0.2,
            "entropy": 0.8,
            "normalized_entropy": 0.75,
        },
        "decision_status": decision_status,
        "insufficient_evidence": abstained,
        "abstention_reasons": ["high_distribution_entropy"] if abstained else [],
        "abstention_thresholds": {
            "min_top_probability": 0.4,
            "min_top_two_margin": 0.05,
            "max_normalized_entropy": 0.95,
        },
        "thresholds_clinically_validated": False,
        "ensemble_weights": {"xgboost": 0.675, "catboost": 0.325},
    }


def passage(text="Asthma causes wheezing, breathlessness, chest tightness, and coughing."):
    return {
        "chunk_id": "cdc-asthma-about:0",
        "text": text,
        "section": "Symptoms",
        "source": {
            "title": "About Asthma",
            "organization": "Centers for Disease Control and Prevention",
            "url": "https://www.cdc.gov/asthma/about/index.html",
            "attribution": "Source: CDC",
            "license": "CDC public-domain content; attribution provided.",
            "rights_url": "https://www.cdc.gov/other/agencymaterials.html",
            "accessed_on": "2026-09-30",
        },
        "relevance_score": 0.61,
    }


def retrieval(passages=None, sufficient=True, supported=True):
    return {
        "condition": "Asthma",
        "supported": supported,
        "sufficient_evidence": sufficient,
        "status": "evidence_available" if sufficient else "no_sufficient_evidence",
        "retrieval_method": "hybrid_weighted_rrf",
        "passages": [passage()] if passages is None and sufficient else passages or [],
    }


class StubGenerator:
    def __init__(self, response):
        self.response = response
        self.payload = None
        self.calls = 0

    def generate(self, payload):
        self.calls += 1
        self.payload = payload
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class GroundedExplanationTests(unittest.TestCase):
    def test_valid_grounded_claim_has_stable_evidence_and_source(self):
        generator = StubGenerator(
            {"claims": [{
                "text": "Wheezing and coughing are among the symptoms described in the retrieved asthma evidence.",
                "evidence_ids": ["E1"],
            }]}
        )
        result = GroundedExplanationService(generator).explain(
            prediction(), ["symptom_wheezing", "symptom_coughing"], retrieval()
        )

        self.assertEqual(result["grounding_status"], "grounded")
        self.assertIn("[E1]", result["explanation"]["summary"])
        self.assertEqual(result["explanation"]["claims"][0]["evidence_ids"], ["E1"])
        self.assertEqual(result["evidence"][0]["evidence_id"], "E1")
        self.assertEqual(result["evidence"][0]["passage"]["source"]["title"], "About Asthma")
        self.assertEqual(generator.payload["model_output"]["predicted_class"], "Asthma")
        self.assertEqual(generator.payload["selected_symptoms"], ["wheezing", "coughing"])
        self.assertIn("uncertainty", generator.payload["model_output"])
        self.assertIn("source", generator.payload["evidence"][0])
        self.assertIn("synthetic data", result["disclaimer"])

    def test_unknown_evidence_id_rejects_all_generated_text(self):
        generator = StubGenerator(
            {"claims": [{"text": "Wheezing is described in the retrieved evidence.", "evidence_ids": ["E9"]}]}
        )
        result = GroundedExplanationService(generator).explain(prediction(), ["wheezing"], retrieval())
        self.assertEqual(result["grounding_status"], "validation_failed")
        self.assertIsNone(result["explanation"])
        self.assertEqual(result["evidence"][0]["evidence_id"], "E1")

    def test_claim_without_citation_is_rejected(self):
        generator = StubGenerator(
            {"claims": [{"text": "Wheezing is described in the retrieved evidence.", "evidence_ids": []}]}
        )
        result = GroundedExplanationService(generator).explain(prediction(), ["wheezing"], retrieval())
        self.assertEqual(result["grounding_status"], "validation_failed")
        self.assertIsNone(result["explanation"])

    def test_malformed_llm_json_fails_safely(self):
        generator = StubGenerator("{not-json")
        result = GroundedExplanationService(generator).explain(prediction(), ["wheezing"], retrieval())
        self.assertEqual(result["grounding_status"], "validation_failed")
        self.assertIsNone(result["explanation"])

    def test_diagnosis_or_treatment_language_is_rejected(self):
        generator = StubGenerator(
            {"claims": [{"text": "You should take medication for asthma.", "evidence_ids": ["E1"]}]}
        )
        result = GroundedExplanationService(generator).explain(prediction(), ["wheezing"], retrieval())
        self.assertEqual(result["grounding_status"], "validation_failed")
        self.assertIsNone(result["explanation"])

    def test_empty_or_unsupported_retrieval_never_calls_llm(self):
        generator = StubGenerator({"claims": []})
        service = GroundedExplanationService(generator)
        empty = service.explain(prediction(), ["wheezing"], retrieval(passages=[], sufficient=False))
        unsupported = service.explain(
            prediction(), ["wheezing"], retrieval(passages=[], sufficient=False, supported=False)
        )
        self.assertEqual(empty["grounding_status"], "insufficient_evidence")
        self.assertEqual(empty["status_message"], "Insufficient evidence for a grounded explanation.")
        self.assertEqual(unsupported["grounding_status"], "insufficient_evidence")
        self.assertEqual(generator.calls, 0)

    def test_model_abstention_never_calls_llm_and_preserves_retrieved_evidence(self):
        generator = StubGenerator({"claims": []})
        result = GroundedExplanationService(generator).explain(
            prediction("insufficient_evidence"), ["wheezing"], retrieval()
        )
        self.assertEqual(result["grounding_status"], "abstained")
        self.assertIn("abstained", result["status_message"])
        self.assertEqual(result["evidence"][0]["evidence_id"], "E1")
        self.assertEqual(generator.calls, 0)

    def test_empty_model_draft_returns_insufficient_evidence(self):
        generator = StubGenerator({"claims": []})
        result = GroundedExplanationService(generator).explain(prediction(), ["wheezing"], retrieval())
        self.assertEqual(result["grounding_status"], "insufficient_evidence")
        self.assertIsNone(result["explanation"])

    def test_llm_unavailable_keeps_retrieved_evidence(self):
        result = GroundedExplanationService(None).explain(prediction(), ["wheezing"], retrieval())
        self.assertEqual(result["grounding_status"], "llm_unavailable")
        self.assertIn("shown below", result["status_message"])
        self.assertEqual(result["evidence"][0]["evidence_id"], "E1")

    def test_config_is_local_by_default_and_environment_overridable(self):
        settings = load_llm_settings(environ={"RAG_LLM_MODEL": "test:model"})
        self.assertEqual(settings["model"], "test:model")
        self.assertEqual(settings["provider"], "ollama")
        self.assertFalse(settings["allow_remote"])

    def test_ollama_uses_json_schema_and_never_uses_environment_proxy(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "message": {"content": '{"claims":[{"text":"Wheezing is described in the evidence.","evidence_ids":["E1"]}]}'},
        }
        settings = {
            "base_url": "http://127.0.0.1:11434",
            "model": "qwen2.5:1.5b-instruct-q4_K_M",
            "timeout_seconds": 4,
            "allow_remote": False,
        }
        payload = {"evidence": [{"evidence_id": "E1", "text": "Wheezing."}]}
        with patch("httpx.post", return_value=response) as post:
            raw = OllamaStructuredGenerator(settings).generate(payload)

        self.assertTrue(raw.startswith("{"))
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["json"]["format"]["properties"]["claims"]["type"], "array")
        self.assertEqual(kwargs["json"]["messages"][1]["content"], '{"evidence":[{"evidence_id":"E1","text":"Wheezing."}]}')
        self.assertFalse(kwargs["trust_env"])
        self.assertIn("Use only the supplied retrieved evidence", kwargs["json"]["messages"][0]["content"])

    def test_remote_ollama_host_is_blocked_without_explicit_opt_in(self):
        generator = OllamaStructuredGenerator({
            "base_url": "http://example.com:11434",
            "model": "qwen2.5:1.5b-instruct-q4_K_M",
            "timeout_seconds": 4,
            "allow_remote": False,
        })
        with patch("httpx.post") as post:
            with self.assertRaises(GenerationUnavailable):
                generator.generate({})
        post.assert_not_called()

    def test_api_returns_prediction_and_grounded_evidence_without_changing_predict(self):
        predictor = Mock()
        predictor.predict.return_value = prediction()
        vectorizer = Mock()
        vectorizer.encode.return_value = {"symptom_wheezing": 1}
        retriever = Mock()
        retriever.retrieve.return_value = {
            "supported": True,
            "sufficient_evidence": True,
            "passages": [passage()],
        }
        generated = {
            "explanation": {
                "summary": "Wheezing is described in the retrieved evidence. [E1]",
                "claims": [{"text": "Wheezing is described in the retrieved evidence.", "evidence_ids": ["E1"]}],
                "grounding_status": "grounded",
            },
            "grounding_status": "grounded",
            "evidence": [{"evidence_id": "E1", "passage": passage()}],
            "status_message": "Citations were validated against the retrieved passages.",
            "disclaimer": EXPLANATION_DISCLAIMER,
        }
        service = Mock()
        service.explain.return_value = generated
        request = GroundedExplainRequest(symptoms=["symptom_wheezing"])
        with (
            patch("api.main.get_predictor", return_value=predictor),
            patch("api.main.get_symptom_vectorizer", return_value=vectorizer),
            patch("api.main.get_evidence_retriever", return_value=retriever),
            patch("api.main.get_grounded_explanation_service", return_value=service),
        ):
            result = explain_grounded(request)

        self.assertIsInstance(result, GroundedExplainResponse)
        self.assertIsInstance(result.prediction, PredictResponse)
        self.assertEqual(result.prediction.decision_status, "ranked_prediction")
        self.assertEqual(result.grounding_status, "grounded")
        self.assertEqual(result.evidence[0].evidence_id, "E1")
        predictor.predict.assert_called_once_with({"symptom_wheezing": 1}, top_k=1)
        retriever.retrieve.assert_called_once_with(
            "Asthma", context=["symptom_wheezing"], strategy="hybrid"
        )

    def test_api_validates_request_shape_and_unknown_symptoms(self):
        with self.assertRaises(ValidationError):
            GroundedExplainRequest(symptoms=[])
        with self.assertRaises(ValidationError):
            GroundedExplainRequest(symptoms=["symptom_wheezing"], unexpected=True)

        predictor = Mock()
        vectorizer = Mock()
        from api.symptom_input import UnknownSymptomsError
        vectorizer.encode.side_effect = UnknownSymptomsError(["unknown"])
        with (
            patch("api.main.get_predictor", return_value=predictor),
            patch("api.main.get_symptom_vectorizer", return_value=vectorizer),
        ):
            with self.assertRaises(HTTPException) as raised:
                explain_grounded(GroundedExplainRequest(symptoms=["unknown"]))
        self.assertEqual(raised.exception.status_code, 422)

    def test_manual_evaluation_reports_structure_metrics_without_classifier_samples(self):
        class StubRetriever:
            def retrieve(self, condition, context, top_k, strategy):
                if condition != "Asthma":
                    return retrieval(passages=[], sufficient=False, supported=False)
                return retrieval()

        cases = [
            {
                "id": "valid",
                "condition": "Asthma",
                "context": ["wheezing"],
                "draft": {"claims": [{
                    "text": "Wheezing is mentioned in the retrieved asthma evidence.",
                    "evidence_ids": ["E1"],
                }]},
                "expected_status": "grounded",
            },
            {
                "id": "unsupported-condition",
                "condition": "Unknown",
                "context": ["other"],
                "draft": {"claims": []},
                "expected_status": "insufficient_evidence",
            },
        ]
        report = run_evaluation(cases, StubRetriever())
        self.assertFalse(report["classifier_test_samples_used"])
        self.assertEqual(report["status_checks_passed"], 2)
        self.assertEqual(report["grounded_claim_rate"]["rate"], 1.0)
        self.assertTrue(report["evidence_gate_behavior"]["correct"])


if __name__ == "__main__":
    unittest.main()
