import unittest

import numpy as np

from src.models.uncertainty import abstention_decision, uncertainty_summary


class UncertaintyTests(unittest.TestCase):
    def test_margin_and_normalized_entropy(self):
        values = uncertainty_summary([0.6, 0.3, 0.1])
        self.assertAlmostEqual(values["top_probability"], 0.6)
        self.assertAlmostEqual(values["top_two_margin"], 0.3)
        self.assertGreater(values["normalized_entropy"], 0.0)
        self.assertLess(values["normalized_entropy"], 1.0)

    def test_uniform_distribution_has_maximum_normalized_entropy(self):
        values = uncertainty_summary(np.full(4, 0.25))
        self.assertAlmostEqual(values["normalized_entropy"], 1.0)
        self.assertAlmostEqual(values["top_two_margin"], 0.0)

    def test_abstains_when_any_configured_threshold_fails(self):
        summary = uncertainty_summary([0.45, 0.4, 0.15])
        thresholds = {
            "min_top_probability": 0.5,
            "min_top_two_margin": 0.1,
            "max_normalized_entropy": 0.8,
        }
        result = abstention_decision(summary, thresholds)
        self.assertTrue(result["abstained"])
        self.assertEqual(result["decision"], "insufficient_evidence")
        self.assertEqual(
            result["reasons"],
            ["low_top_probability", "low_top_two_separation", "high_distribution_entropy"],
        )

    def test_threshold_boundary_is_accepted(self):
        summary = uncertainty_summary([0.8, 0.15, 0.05])
        thresholds = {
            "min_top_probability": summary["top_probability"],
            "min_top_two_margin": summary["top_two_margin"],
            "max_normalized_entropy": summary["normalized_entropy"],
        }
        result = abstention_decision(summary, thresholds)
        self.assertFalse(result["abstained"])
        self.assertEqual(result["decision"], "ranked_prediction")
        self.assertEqual(result["reasons"], [])


if __name__ == "__main__":
    unittest.main()
