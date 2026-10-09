"""Opt-in local Ollama integration test for the grounded explanation API."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from api.main import app
from rag.grounded_explanation import GroundedExplanationService, create_generator
from rag.retrieval import EvidenceRetriever


@unittest.skipUnless(
    os.environ.get("RAG_RUN_LIVE_LLM_TEST") == "1",
    "Set RAG_RUN_LIVE_LLM_TEST=1 to call the configured local Ollama model.",
)
class LiveGroundedExplanationTests(unittest.TestCase):
    def test_grounded_endpoint_with_real_hybrid_retrieval_and_ollama(self):
        settings_response = httpx.get(
            "http://127.0.0.1:11434/api/tags",
            timeout=5,
            trust_env=False,
        )
        self.assertEqual(settings_response.status_code, 200)
        available_models = {
            item["name"] for item in settings_response.json().get("models", [])
        }
        configured_model = "qwen2.5:1.5b-instruct-q4_K_M"
        self.assertIn(configured_model, available_models)

        symptoms = ["symptom_wheezing", "symptom_shortness_of_breath"]
        calibration = json.loads(
            (Path(__file__).resolve().parents[1] / "configs" / "ensemble_calibration.json")
            .read_text(encoding="utf-8")
        )
        thresholds = calibration["abstention"]["thresholds"]
        prediction = {
            "predicted_disease": "Asthma",
            "predicted_probability": 0.82,
            "predicted_percentage": 82.0,
            "raw_predicted_probability": 0.79,
            "top_predictions": [
                {"rank": 1, "disease": "Asthma", "probability": 0.82, "percentage": 82.0}
            ],
            "probability_status": "calibrated",
            "calibration_method": "temperature_scaling",
            "calibration_temperature": 0.673403810833824,
            "uncertainty": {
                "top_probability": 0.82,
                "top_two_margin": 0.61,
                "entropy": 1.3,
                "normalized_entropy": 0.38,
            },
            "decision_status": "ranked_prediction",
            "insufficient_evidence": False,
            "abstention_reasons": [],
            "abstention_thresholds": thresholds,
            "thresholds_clinically_validated": False,
            "ensemble_weights": {"xgboost": 0.675, "catboost": 0.325},
        }
        self.assertGreaterEqual(
            prediction["uncertainty"]["top_probability"],
            thresholds["min_top_probability"],
        )
        self.assertGreaterEqual(
            prediction["uncertainty"]["top_two_margin"],
            thresholds["min_top_two_margin"],
        )
        self.assertLessEqual(
            prediction["uncertainty"]["normalized_entropy"],
            thresholds["max_normalized_entropy"],
        )

        class DevelopmentPredictor:
            def predict(self, _vector, top_k=1):
                self.top_k = top_k
                return prediction

        class DevelopmentVectorizer:
            def encode(self, values):
                return {value: 1 for value in values}

        retriever = EvidenceRetriever()
        predictor = DevelopmentPredictor()
        evidence = retriever.retrieve(
            "Asthma", context=symptoms, strategy="hybrid"
        )
        self.assertTrue(evidence["supported"])
        self.assertTrue(evidence["sufficient_evidence"])
        self.assertTrue(evidence["passages"])

        generator = create_generator()
        self.assertIsNotNone(generator)
        service = GroundedExplanationService(generator)
        with (
            patch("api.main.get_predictor", return_value=predictor),
            patch("api.main.get_symptom_vectorizer", return_value=DevelopmentVectorizer()),
            patch("api.main.get_evidence_retriever", return_value=retriever),
            patch("api.main.get_grounded_explanation_service", return_value=service),
            TestClient(app) as client,
        ):
            response = client.post("/explain-grounded", json={"symptoms": symptoms})

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["prediction"]["decision_status"], "ranked_prediction")
        self.assertEqual(body["grounding_status"], "grounded", body["status_message"])
        self.assertTrue(body["explanation"]["claims"])

        supplied_ids = {item["evidence_id"] for item in body["evidence"]}
        used_ids = {
            evidence_id
            for claim in body["explanation"]["claims"]
            for evidence_id in claim["evidence_ids"]
        }
        self.assertTrue(used_ids)
        self.assertLessEqual(used_ids, supplied_ids)
        self.assertTrue(
            all(
                item["passage"]["source"].get("title")
                and item["passage"]["source"].get("url")
                for item in body["evidence"]
            )
        )
        print(
            "LIVE_GROUNDED_RESULT="
            + json.dumps(
                {
                    "status": body["grounding_status"],
                    "claims": body["explanation"]["claims"],
                    "evidence": [
                        {
                            "evidence_id": item["evidence_id"],
                            "title": item["passage"]["source"]["title"],
                            "url": item["passage"]["source"]["url"],
                            "section": item["passage"]["section"],
                        }
                        for item in body["evidence"]
                    ],
                },
                ensure_ascii=False,
            )
        )

    def test_reported_abstentions_skip_generation_and_return_retrieval_result(self):
        class CallCounterGenerator:
            def __init__(self):
                self.calls = 0

            def generate(self, _payload):
                self.calls += 1
                raise AssertionError("Abstained predictions must not invoke Ollama.")

        generator = CallCounterGenerator()
        service = GroundedExplanationService(generator)
        cases = [
            (
                ["symptom_fatigue", "symptom_cough", "symptom_headache", "symptom_nausea"],
                "Type2_Diabetes",
                ["low_top_two_separation", "high_distribution_entropy"],
            ),
            (
                ["symptom_polyuria", "symptom_polydipsia", "symptom_polyphagia"],
                "Depression",
                ["low_top_two_separation", "high_distribution_entropy"],
            ),
        ]

        with (
            patch("api.main.get_grounded_explanation_service", return_value=service),
            TestClient(app) as client,
        ):
            for symptoms, expected_class, expected_reasons in cases:
                response = client.post("/explain-grounded", json={"symptoms": symptoms})
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["prediction"]["predicted_disease"], expected_class)
                self.assertEqual(body["prediction"]["decision_status"], "insufficient_evidence")
                self.assertEqual(body["prediction"]["abstention_reasons"], expected_reasons)
                self.assertEqual(body["grounding_status"], "abstained")
                self.assertIsNone(body["explanation"])
                self.assertIsInstance(body["evidence"], list)
                print(
                    "LIVE_ABSTENTION_RESULT="
                    + json.dumps(
                        {
                            "symptoms": symptoms,
                            "predicted_class": body["prediction"]["predicted_disease"],
                            "probability": body["prediction"]["predicted_probability"],
                            "top_two_margin": body["prediction"]["uncertainty"]["top_two_margin"],
                            "normalized_entropy": body["prediction"]["uncertainty"]["normalized_entropy"],
                            "reasons": body["prediction"]["abstention_reasons"],
                            "evidence_count": len(body["evidence"]),
                        }
                    )
                )

        self.assertEqual(generator.calls, 0)


if __name__ == "__main__":
    unittest.main()
