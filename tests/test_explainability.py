"""Synthetic checks for probability-space ensemble attributions."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi import HTTPException

from api.main import explain as explain_route
from api.schemas import ExplainRequest
from src.explainability.shap_explainer import EnsembleShapExplainer


class LinearProbabilityModel:
    def __init__(self, classes, feature_weights, baseline=0.5):
        self.classes_ = np.asarray(classes)
        self.feature_weights = np.asarray(feature_weights, dtype=float)
        self.baseline = baseline

    def predict_proba(self, frame):
        target_probability = self.baseline + frame.to_numpy() @ self.feature_weights
        if self.classes_[0] == 1:
            return np.column_stack((target_probability, 1.0 - target_probability))
        return np.column_stack((1.0 - target_probability, target_probability))


class FakePredictor:
    XGB_WEIGHT = 0.675
    CATBOOST_WEIGHT = 0.325

    def __init__(self, feature_columns=None, bad_xgb_shape=False):
        self.feature_columns = feature_columns or [
            "symptom_alpha",
            "symptom_beta",
            "symptom_unused",
        ]
        self.xgb_model = LinearProbabilityModel(
            [0, 1], [0.10, 0.04, 0.0], baseline=0.62
        )
        self.catboost_model = LinearProbabilityModel(
            [1, 0], [0.02, 0.25, 0.0], baseline=0.55
        )
        if bad_xgb_shape:
            self.xgb_model.predict_proba = lambda frame: np.zeros((len(frame), 1))

    def get_feature_columns(self):
        return self.feature_columns.copy()


class EnsembleShapExplainerTests(unittest.TestCase):
    def setUp(self):
        self.predictor = FakePredictor()
        self.explainer = EnsembleShapExplainer(self.predictor)
        self.symptoms = {
            "symptom_alpha": 1,
            "symptom_beta": 1,
            "symptom_unused": 0,
        }

    def test_exact_values_select_predicted_class_and_are_additive(self):
        result = self.explainer.explain(self.symptoms, class_label=1)

        self.assertEqual(result["predicted_class_index"], 1)
        self.assertEqual(result["output_space"], "ensemble_probability")
        self.assertEqual(result["method"], "exact_coalition_shapley")
        self.assertTrue(result["is_exact"])
        self.assertIsNone(result["permutation_count"])
        self.assertAlmostEqual(result["baseline_probability"], 0.59725)
        self.assertAlmostEqual(result["predicted_probability"], 0.7795)
        self.assertAlmostEqual(
            sum(item["shap_value"] for item in result["features"]),
            result["predicted_probability"] - result["baseline_probability"],
        )

        self.assertEqual(len(result["features"]), 2)
        self.assertEqual(
            [item["feature"] for item in result["features"]],
            ["symptom_beta", "symptom_alpha"],
        )
        self.assertAlmostEqual(result["features"][0]["shap_value"], 0.10825)
        self.assertAlmostEqual(result["features"][1]["shap_value"], 0.074)

        components = result["component_attributions"]
        self.assertEqual(
            [component["model"] for component in components],
            ["XGBoost model attribution", "CatBoost model attribution"],
        )
        self.assertEqual([component["class_index"] for component in components], [1, 0])
        self.assertTrue(all(component["output_space"] == "model_probability" for component in components))
        for index, ensemble_item in enumerate(result["features"]):
            feature = ensemble_item["feature"]
            xgb_value = next(
                item["shap_value"] for item in components[0]["features"]
                if item["feature"] == feature
            )
            cat_value = next(
                item["shap_value"] for item in components[1]["features"]
                if item["feature"] == feature
            )
            self.assertAlmostEqual(
                ensemble_item["shap_value"],
                self.predictor.XGB_WEIGHT * xgb_value
                + self.predictor.CATBOOST_WEIGHT * cat_value,
            )

    def test_class_zero_uses_each_models_own_class_column(self):
        result = self.explainer.explain(self.symptoms, class_label=0)
        self.assertEqual(result["predicted_class_index"], 0)
        self.assertEqual(
            [component["class_index"] for component in result["component_attributions"]],
            [0, 1],
        )
        self.assertAlmostEqual(result["predicted_probability"], 0.2205)
        self.assertTrue(all(item["shap_value"] < 0 for item in result["features"]))

    def test_large_input_uses_deterministic_sampled_permutation_attribution(self):
        feature_columns = [f"symptom_{index}" for index in range(13)]
        predictor = FakePredictor(feature_columns=feature_columns)
        predictor.xgb_model = LinearProbabilityModel(
            [0, 1], [0.005] * 5 + [0.0] * 8, baseline=0.4
        )
        predictor.catboost_model = LinearProbabilityModel(
            [1, 0], [0.003] * 5 + [0.0] * 8, baseline=0.45
        )
        symptoms = {feature: 1 for feature in feature_columns}

        result = EnsembleShapExplainer(predictor).explain(symptoms, class_label=1)
        total = sum(item["shap_value"] for item in result["features"])
        self.assertEqual(result["method"], "sampled_permutation_shapley")
        self.assertFalse(result["is_exact"])
        self.assertEqual(result["permutation_count"], 128)
        self.assertEqual(len(result["features"]), EnsembleShapExplainer.TOP_FEATURES)
        self.assertEqual(
            [len(component["features"]) for component in result["component_attributions"]],
            [13, 13],
        )
        self.assertAlmostEqual(
            total,
            result["predicted_probability"] - result["baseline_probability"],
            places=10,
        )

    def test_invalid_class_and_probability_dimensions_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "does not contain the predicted class"):
            self.explainer.explain(self.symptoms, class_label=7)

        broken = EnsembleShapExplainer(FakePredictor(bad_xgb_shape=True))
        with self.assertRaisesRegex(ValueError, "probability output shape"):
            broken.explain(self.symptoms, class_label=1)

    def test_empty_selected_features_fail_clearly(self):
        empty = {feature: 0 for feature in self.predictor.get_feature_columns()}
        with self.assertRaisesRegex(ValueError, "At least one selected symptom"):
            self.explainer.explain(empty, class_label=1)


class ExplainRouteTests(unittest.TestCase):
    def setUp(self):
        self.predictor = Mock()
        self.predictor.predict.return_value = {
            "predicted_disease": "Disease_B",
            "predicted_probability": 0.7,
        }
        self.predictor.label_encoder.transform.return_value = np.asarray([1])
        self.vectorizer = Mock()
        self.vectorizer.encode.return_value = {"symptom_alpha": 1}
        self.request = ExplainRequest(symptoms=["symptom_alpha"])

    def test_route_targets_the_class_from_the_current_ensemble_prediction(self):
        explainer = Mock()
        explainer.explain.return_value = {
            "predicted_class_index": 1,
            "predicted_probability": 0.7,
            "baseline_probability": 0.4,
            "output_space": "ensemble_probability",
            "method": "exact_coalition_shapley",
            "is_exact": True,
            "permutation_count": None,
            "features": [
                {
                    "feature": "symptom_alpha",
                    "shap_value": 0.3,
                    "direction": "increases_probability",
                }
            ],
            "component_attributions": [],
        }
        with (
            patch("api.main.get_predictor", return_value=self.predictor),
            patch("api.main.get_symptom_vectorizer", return_value=self.vectorizer),
            patch("api.main.get_shap_explainer", return_value=explainer),
        ):
            response = explain_route(self.request)

        explainer.explain.assert_called_once_with({"symptom_alpha": 1}, 1)
        self.assertEqual(response.predicted_disease, "Disease_B")
        self.assertEqual(response.predicted_class_index, 1)
        self.assertEqual(response.output_space, "ensemble_probability")

    def test_route_converts_explainer_failures_to_service_unavailable(self):
        explainer = Mock()
        explainer.explain.side_effect = ValueError("malformed model output")
        with (
            patch("api.main.get_predictor", return_value=self.predictor),
            patch("api.main.get_symptom_vectorizer", return_value=self.vectorizer),
            patch("api.main.get_shap_explainer", return_value=explainer),
        ):
            with self.assertRaises(HTTPException) as raised:
                explain_route(self.request)

        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
