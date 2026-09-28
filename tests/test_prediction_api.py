import unittest
from unittest.mock import Mock, patch

from api.main import predict as predict_route
from api.schemas import PredictRequest, PredictResponse


class PredictionApiTests(unittest.TestCase):
    def setUp(self):
        self.predictor = Mock()
        self.predictor.n_classes = 2
        self.predictor.predict.return_value = {
            "predicted_disease": "Condition B",
            "predicted_probability": 0.58,
            "predicted_percentage": 58.0,
            "raw_predicted_probability": 0.55,
            "top_predictions": [
                {
                    "rank": 1,
                    "disease": "Condition B",
                    "probability": 0.58,
                    "percentage": 58.0,
                },
                {
                    "rank": 2,
                    "disease": "Condition A",
                    "probability": 0.42,
                    "percentage": 42.0,
                },
            ],
            "probability_status": "calibrated",
            "calibration_method": "temperature_scaling",
            "calibration_temperature": 1.4,
            "uncertainty": {
                "top_probability": 0.58,
                "top_two_margin": 0.16,
                "entropy": 0.68,
                "normalized_entropy": 0.98,
            },
            "decision_status": "insufficient_evidence",
            "insufficient_evidence": True,
            "abstention_reasons": ["high_distribution_entropy"],
            "abstention_thresholds": {
                "min_top_probability": 0.4,
                "min_top_two_margin": 0.05,
                "max_normalized_entropy": 0.95,
            },
            "thresholds_clinically_validated": False,
            "ensemble_weights": {"xgboost": 0.675, "catboost": 0.325},
        }
        self.vectorizer = Mock()
        self.vectorizer.encode.return_value = {
            "symptom_alpha": 1,
            "symptom_beta": 0,
        }

    def test_predict_route_returns_probability_uncertainty_and_abstention(self):
        request = PredictRequest(symptoms=["symptom_alpha"], top_k=2)
        with (
            patch("api.main.get_predictor", return_value=self.predictor),
            patch("api.main.get_symptom_vectorizer", return_value=self.vectorizer),
        ):
            response = predict_route(request)

        self.assertIsInstance(response, PredictResponse)
        self.assertEqual(response.probability_status, "calibrated")
        self.assertEqual(response.decision_status, "insufficient_evidence")
        self.assertEqual(response.uncertainty.top_two_margin, 0.16)
        self.assertFalse(response.thresholds_clinically_validated)
        self.assertEqual(response.abstention_reasons, ["high_distribution_entropy"])
        self.predictor.predict.assert_called_once_with(
            {"symptom_alpha": 1, "symptom_beta": 0},
            top_k=2,
        )

    def test_predict_route_rejects_top_k_above_supported_classes(self):
        request = PredictRequest(symptoms=["symptom_alpha"], top_k=3)
        with patch("api.main.get_predictor", return_value=self.predictor):
            with self.assertRaises(Exception) as raised:
                predict_route(request)
        self.assertEqual(getattr(raised.exception, "status_code", None), 422)


if __name__ == "__main__":
    unittest.main()
