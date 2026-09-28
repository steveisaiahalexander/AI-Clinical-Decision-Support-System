import unittest

import numpy as np

from src.models.inference import ClinicalEnsemblePredictor


class FixedProbabilityModel:
    classes_ = np.asarray([0, 1])

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, frame):
        return np.repeat(self.probabilities[None, :], len(frame), axis=0)


class FixedLabelEncoder:
    def inverse_transform(self, labels):
        names = ["Condition A", "Condition B"]
        return [names[int(label)] for label in labels]


class InferenceDecisionTests(unittest.TestCase):
    def make_predictor(self, thresholds):
        predictor = ClinicalEnsemblePredictor.__new__(ClinicalEnsemblePredictor)
        predictor.feature_columns = ["symptom_alpha", "symptom_beta"]
        predictor.n_features = 2
        predictor.n_classes = 2
        predictor.classes_ = np.asarray([0, 1])
        predictor.label_encoder = FixedLabelEncoder()
        predictor.xgb_model = FixedProbabilityModel([0.45, 0.55])
        predictor.catboost_model = FixedProbabilityModel([0.45, 0.55])
        predictor._catboost_column_by_class = {0: 0, 1: 1}
        predictor.calibration_method = "temperature_scaling"
        predictor.calibration_temperature = 2.0
        predictor.probability_status = "calibrated"
        predictor.abstention_thresholds = thresholds
        return predictor

    def test_inference_returns_calibrated_probabilities_and_uncertainty(self):
        predictor = self.make_predictor(
            {
                "min_top_probability": 0.5,
                "min_top_two_margin": 0.0,
                "max_normalized_entropy": 1.0,
            }
        )
        result = predictor.predict(
            {"symptom_alpha": 1, "symptom_beta": 0},
            top_k=2,
        )
        self.assertEqual(result["probability_status"], "calibrated")
        self.assertEqual(result["decision_status"], "ranked_prediction")
        self.assertFalse(result["insufficient_evidence"])
        self.assertAlmostEqual(result["predicted_probability"], result["uncertainty"]["top_probability"])
        self.assertAlmostEqual(sum(item["probability"] for item in result["top_predictions"]), 1.0)
        self.assertFalse(result["thresholds_clinically_validated"])

    def test_inference_abstains_and_returns_machine_readable_reason(self):
        predictor = self.make_predictor(
            {
                "min_top_probability": 0.9,
                "min_top_two_margin": 0.0,
                "max_normalized_entropy": 1.0,
            }
        )
        result = predictor.predict(
            {"symptom_alpha": 1, "symptom_beta": 0},
            top_k=2,
        )
        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["decision_status"], "insufficient_evidence")
        self.assertIn("low_top_probability", result["abstention_reasons"])


if __name__ == "__main__":
    unittest.main()
