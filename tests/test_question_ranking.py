import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from api.main import next_question
from api.schemas import NextQuestionRequest
from api.symptom_input import SymptomVectorizer
from src.models.question_ranking import (
    SymptomQuestionRanker,
    binary_question_information_gain,
)


class InformationGainTests(unittest.TestCase):
    def test_perfect_binary_question_reduces_one_bit_of_entropy(self):
        result = binary_question_information_gain([0.5, 0.5], [1.0, 0.0])
        self.assertAlmostEqual(result["information_gain_bits"], 1.0)
        self.assertAlmostEqual(result["expected_entropy_after_answer_bits"], 0.0)

    def test_uninformative_question_has_zero_gain(self):
        result = binary_question_information_gain([0.5, 0.5], [0.3, 0.3])
        self.assertAlmostEqual(result["information_gain_bits"], 0.0)

    def test_degenerate_distribution_and_probability_edges_are_finite(self):
        result = binary_question_information_gain([1.0, 0.0], [0.0, 1.0])
        self.assertEqual(result["information_gain_bits"], 0.0)
        self.assertEqual(result["expected_entropy_after_answer_bits"], 0.0)
        always_yes = binary_question_information_gain([0.5, 0.5], [1.0, 1.0])
        self.assertEqual(always_yes["probability_yes"], 1.0)
        self.assertEqual(always_yes["information_gain_bits"], 0.0)

    def test_ranking_is_deterministic_and_ties_follow_model_feature_order(self):
        ranker = SymptomQuestionRanker(
            ["symptom_beta", "symptom_alpha"],
            ["ClassA", "ClassB"],
            {
                "symptom_beta": [1.0, 0.0],
                "symptom_alpha": [0.0, 1.0],
            },
        )
        first = ranker.rank({"ClassA": 0.5, "ClassB": 0.5}, top_k=2)
        second = ranker.rank({"ClassA": 0.5, "ClassB": 0.5}, top_k=2)
        self.assertEqual(first, second)
        self.assertEqual(
            [item["feature_id"] for item in first["suggestions"]],
            ["symptom_beta", "symptom_alpha"],
        )

    def test_selected_negative_and_skipped_features_are_excluded(self):
        ranker = SymptomQuestionRanker(
            ["symptom_a", "symptom_b", "symptom_c"],
            ["ClassA", "ClassB"],
            {
                "symptom_a": [1.0, 0.0],
                "symptom_b": [0.0, 1.0],
                "symptom_c": [0.8, 0.2],
            },
        )
        result = ranker.rank(
            {"ClassA": 0.6, "ClassB": 0.4},
            excluded_features=["symptom_a", "symptom_b"],
        )
        self.assertEqual([item["feature_id"] for item in result["suggestions"]], ["symptom_c"])

    def test_constant_missing_or_degenerate_statistics_fail_safely(self):
        ranker = SymptomQuestionRanker(
            ["symptom_constant", "symptom_missing", "symptom_variable"],
            ["ClassA", "ClassB"],
            {
                "symptom_constant": [1.0, 1.0],
                "symptom_variable": [0.9, 0.1],
            },
            constant_features=["symptom_constant"],
        )
        degenerate = ranker.rank({"ClassA": 1.0, "ClassB": 0.0})
        self.assertEqual(degenerate["status"], "no_useful_candidate")
        self.assertEqual(degenerate["suggestions"], [])
        self.assertEqual(degenerate["eligible_candidate_count"], 1)
        missing = SymptomQuestionRanker(
            ["symptom_missing"], ["ClassA", "ClassB"], {}
        ).rank({"ClassA": 0.5, "ClassB": 0.5})
        self.assertEqual(missing["status"], "no_eligible_candidates")
        self.assertEqual(missing["suggestions"], [])

    def test_fully_selected_features_return_no_unasked_candidates(self):
        ranker = SymptomQuestionRanker(
            ["symptom_a", "symptom_b"],
            ["ClassA", "ClassB"],
            {"symptom_a": [1.0, 0.0], "symptom_b": [0.0, 1.0]},
        )
        result = ranker.rank(
            {"ClassA": 0.5, "ClassB": 0.5},
            excluded_features=["symptom_a", "symptom_b"],
        )
        self.assertEqual(result["status"], "no_unasked_candidates")
        self.assertEqual(result["eligible_candidate_count"], 0)
        self.assertEqual(result["suggestions"], [])

    def test_unknown_exclusions_and_invalid_class_probabilities_are_rejected(self):
        ranker = SymptomQuestionRanker(
            ["symptom_a"], ["ClassA", "ClassB"], {"symptom_a": [0.9, 0.1]}
        )
        with self.assertRaisesRegex(ValueError, "Unknown symptoms"):
            ranker.rank({"ClassA": 0.5, "ClassB": 0.5}, excluded_features=["not_a_symptom"])
        with self.assertRaises(ValueError):
            ranker.rank({"ClassA": 0.0, "ClassB": 0.0})


def prediction(decision_status="ranked_prediction"):
    return {
        "predicted_disease": "ClassA",
        "predicted_probability": 0.6,
        "predicted_percentage": 60.0,
        "raw_predicted_probability": 0.58,
        "top_predictions": [
            {"rank": 1, "disease": "ClassA", "probability": 0.6, "percentage": 60.0},
            {"rank": 2, "disease": "ClassB", "probability": 0.4, "percentage": 40.0},
        ],
        "probability_status": "calibrated",
        "calibration_method": "temperature_scaling",
        "calibration_temperature": 0.673,
        "uncertainty": {
            "top_probability": 0.6,
            "top_two_margin": 0.2,
            "entropy": 0.97,
            "normalized_entropy": 0.7,
        },
        "decision_status": decision_status,
        "insufficient_evidence": decision_status == "insufficient_evidence",
        "abstention_reasons": ["high_distribution_entropy"]
        if decision_status == "insufficient_evidence"
        else [],
        "abstention_thresholds": {
            "min_top_probability": 0.1,
            "min_top_two_margin": 0.1,
            "max_normalized_entropy": 0.9,
        },
        "thresholds_clinically_validated": False,
        "ensemble_weights": {"xgboost": 0.675, "catboost": 0.325},
    }


class NextQuestionApiTests(unittest.TestCase):
    def setUp(self):
        self.features = ["symptom_a", "symptom_b", "symptom_c"]
        self.vectorizer = SymptomVectorizer(self.features)
        self.predictor = Mock()
        self.predictor.predict.return_value = prediction()
        self.ranker = SymptomQuestionRanker(
            self.features,
            ["ClassA", "ClassB"],
            {
                "symptom_a": [1.0, 0.0],
                "symptom_b": [0.0, 1.0],
                "symptom_c": [0.5, 0.5],
            },
        )

    def call_api(self, request):
        with (
            patch("api.main.get_predictor", return_value=self.predictor),
            patch("api.main.get_symptom_vectorizer", return_value=self.vectorizer),
            patch("api.main.get_symptom_question_ranker", return_value=self.ranker),
        ):
            return next_question(request)

    def test_api_returns_ranked_suggestions_and_preserves_abstention_status(self):
        self.predictor.predict.return_value = prediction("insufficient_evidence")
        request = NextQuestionRequest(
            symptoms=["symptom_c"],
            answered_negative_symptoms=["symptom_a"],
            top_k=2,
        )
        response = self.call_api(request)
        self.assertTrue(response.prediction_status.insufficient_evidence)
        self.assertEqual(response.prediction_status.abstention_reasons, ["high_distribution_entropy"])
        self.assertEqual(response.status, "suggestions_available")
        self.assertEqual([item.feature_id for item in response.suggestions], ["symptom_b"])
        self.predictor.predict.assert_called_once_with(
            self.vectorizer.encode(["symptom_c"]), top_k=2
        )

    def test_empty_and_fully_selected_requests_are_safe(self):
        with self.assertRaises(ValidationError):
            NextQuestionRequest(symptoms=[])
        request = NextQuestionRequest(symptoms=self.features)
        response = self.call_api(request)
        self.assertEqual(response.status, "no_unasked_candidates")
        self.assertEqual(response.suggestions, [])

    def test_unknown_positive_or_questioning_answer_returns_422(self):
        with self.assertRaises(HTTPException) as unknown_positive:
            self.call_api(NextQuestionRequest(symptoms=["unknown_symptom"]))
        self.assertEqual(unknown_positive.exception.status_code, 422)

        with self.assertRaises(HTTPException) as unknown_negative:
            self.call_api(
                NextQuestionRequest(
                    symptoms=["symptom_a"],
                    answered_negative_symptoms=["unknown_symptom"],
                )
            )
        self.assertEqual(unknown_negative.exception.status_code, 422)

    def test_duplicate_or_conflicting_answers_are_rejected(self):
        with self.assertRaises(ValidationError):
            NextQuestionRequest(symptoms=["symptom_a", "symptom_a"])
        with self.assertRaises(ValidationError):
            NextQuestionRequest(
                symptoms=["symptom_a"], answered_negative_symptoms=["symptom_a"]
            )
        with self.assertRaises(ValidationError):
            NextQuestionRequest(symptoms=["symptom_a"], top_k=11)


if __name__ == "__main__":
    unittest.main()
