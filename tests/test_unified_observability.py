import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from api.observability import InMemoryObservability, OBSERVABILITY
from rag.grounded_explanation import GroundedExplanationService
from scripts.build_unified_evaluation_report import build_unified_report, render_markdown


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        OBSERVABILITY.reset()

    def tearDown(self):
        OBSERVABILITY.reset()

    def test_aggregates_are_bounded_and_do_not_retain_input_values(self):
        stats = InMemoryObservability()
        stats.record_request("/predict", 422, 0.012)
        stats.record_request("/unrecognized/path", 500, 0.020)
        stats.record_retrieval(0.03, passage_count=2)
        stats.record_prediction({
            "decision_status": "insufficient_evidence",
            "abstention_reasons": ["high_distribution_entropy", "private_reason"],
        })
        stats.record_llm("generated", 0.15)
        snapshot = stats.snapshot()
        serialized = str(snapshot)

        self.assertIn("/predict", snapshot["by_endpoint"])
        self.assertIn("unmatched", snapshot["by_endpoint"])
        self.assertEqual(snapshot["retrieval"]["passages_returned"], 2)
        self.assertEqual(snapshot["predictions"]["abstention_reasons"]["other"], 1)
        self.assertEqual(snapshot["llm_generation"]["outcomes"]["generated"], 1)
        self.assertNotIn("secret", serialized)
        self.assertNotIn("cough", serialized)
        self.assertNotIn("private_reason", serialized)

    def test_status_endpoint_sanitizes_components_and_keeps_optional_llm_degraded(self):
        class Predictor:
            n_features = 174
            n_classes = 30

        class Retriever:
            stats = {"documents": 30, "chunks": 90, "features": 1200}

        class Ranker:
            feature_order = list(range(174))
            class_order = list(range(30))

        with (
            patch("api.main.get_predictor", return_value=Predictor()),
            patch("api.main.get_evidence_retriever", return_value=Retriever()),
            patch("api.main.get_symptom_question_ranker", return_value=Ranker()),
            patch("api.main.probe_ollama", return_value={"status": "unavailable", "message": "Ollama is not reachable.", "details": {}}),
            TestClient(app) as client,
        ):
            response = client.get("/system-status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["overall_status"], "degraded")
        self.assertEqual(payload["components"]["llm"]["status"], "unavailable")
        self.assertEqual(payload["components"]["evidence"]["details"]["documents"], 30)
        self.assertIn("not clinical monitoring", payload["disclaimer"])

    def test_health_error_does_not_expose_internal_exception(self):
        with patch("api.main.get_predictor", side_effect=RuntimeError("private filesystem path")), TestClient(app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Inference model is unavailable.")
        self.assertNotIn("private filesystem path", response.text)

    def test_grounded_generation_outcomes_are_aggregated_without_claim_text(self):
        stats = InMemoryObservability()
        retrieval = {
            "supported": True,
            "sufficient_evidence": True,
            "passages": [{"text": "Wheezing is described.", "section": "Symptoms", "source": {}}],
        }
        prediction = {"decision_status": "ranked_prediction", "predicted_disease": "Asthma"}

        class Generator:
            def __init__(self, response):
                self.response = response

            def generate(self, payload):
                return self.response

        GroundedExplanationService(Generator({"claims": [{"text": "Evidence mentions wheezing.", "evidence_ids": ["E1"]}]}), stats).explain(prediction, [], retrieval)
        GroundedExplanationService(Generator("{bad"), stats).explain(prediction, [], retrieval)
        GroundedExplanationService(None, stats).explain(prediction, [], retrieval)
        GroundedExplanationService(Generator({"claims": []}), stats).explain({**prediction, "decision_status": "insufficient_evidence"}, [], retrieval)
        snapshot = stats.snapshot()
        self.assertEqual(snapshot["llm_generation"]["outcomes"], {"generated": 1, "unavailable": 1, "failed": 1, "skipped": 1})
        self.assertNotIn("Evidence mentions", str(snapshot))

    def test_report_consolidates_saved_artifacts_without_test_metrics_mixing(self):
        root = Path(__file__).resolve().parents[1]
        report = build_unified_report(root)
        sections = report["sections"]
        self.assertFalse(report["evaluation_integrity"]["classifier_test_samples_used_for_tuning"])
        self.assertEqual(sections["classification"]["metrics"]["accuracy"]["dataset"], "Final held-out classifier test set (n=19974)")
        self.assertEqual(sections["retrieval"]["strategies"]["hybrid"]["metrics"]["recall_at_3"]["category"], "retrieval")
        self.assertEqual(sections["retrieval"]["supported_class_coverage"]["value"], 30)
        self.assertIn("not semantic or medical truth", render_markdown(report))

    def test_metrics_and_report_endpoints_return_saved_contracts(self):
        with TestClient(app) as client:
            metrics_response = client.get("/metrics")
            report_response = client.get("/evaluation-report")
        self.assertEqual(metrics_response.status_code, 200)
        self.assertIn("by_endpoint", metrics_response.json())
        self.assertEqual(report_response.status_code, 200)
        self.assertEqual(report_response.json()["title"], "Unified Evaluation Report")


if __name__ == "__main__":
    unittest.main()
