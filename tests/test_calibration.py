import unittest

import numpy as np

from src.models.calibration import (
    calibration_metrics,
    fit_temperature,
    temperature_scale,
)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.probabilities = np.asarray(
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7]],
            dtype=float,
        )
        self.labels = np.asarray([0, 1, 1, 1])

    def test_temperature_one_preserves_normalized_probabilities(self):
        scaled = temperature_scale(self.probabilities, 1.0)
        np.testing.assert_allclose(scaled, self.probabilities)

    def test_temperature_fit_reduces_calibration_sample_log_loss(self):
        temperature = fit_temperature(self.probabilities, self.labels)
        calibrated = temperature_scale(self.probabilities, temperature)
        raw_loss = calibration_metrics(self.probabilities, self.labels)["log_loss"]
        calibrated_loss = calibration_metrics(calibrated, self.labels)["log_loss"]
        self.assertGreater(temperature, 1.0)
        self.assertLess(calibrated_loss, raw_loss)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)

    def test_metrics_include_multiclass_brier_and_top_label_ece_bins(self):
        metrics = calibration_metrics(self.probabilities, self.labels, n_bins=5)
        self.assertEqual(metrics["sample_count"], 4)
        self.assertIn("multiclass_brier_score", metrics)
        self.assertIn("top_label_ece", metrics)
        self.assertEqual(metrics["ece_definition"], "equal-width top-label expected calibration error")
        self.assertEqual(sum(item["count"] for item in metrics["bins"]), 4)

    def test_invalid_temperature_and_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            temperature_scale(self.probabilities, 0.0)
        with self.assertRaises(ValueError):
            fit_temperature(self.probabilities, [0, 1])


if __name__ == "__main__":
    unittest.main()
