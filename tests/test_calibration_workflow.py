import unittest

import numpy as np

from scripts.calibrate_final_ensemble import (
    MIN_POLICY_COVERAGE,
    _partition_indices,
    _select_thresholds,
)


class CalibrationWorkflowTests(unittest.TestCase):
    def test_training_only_partitions_are_disjoint_and_reproducible(self):
        labels = np.repeat(np.arange(3), 1000)
        first = _partition_indices(labels)
        second = _partition_indices(labels)
        self.assertEqual(set(first), set(second))
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

        all_indices = np.concatenate(list(first.values()))
        self.assertEqual(len(np.unique(all_indices)), len(labels))
        self.assertEqual(set(all_indices.tolist()), set(range(len(labels))))
        self.assertAlmostEqual(len(first["model_fit"]) / len(labels), 0.60, delta=0.01)
        for name in (
            "calibration_fit",
            "method_selection",
            "threshold_selection",
            "validation_evaluation",
        ):
            self.assertAlmostEqual(len(first[name]) / len(labels), 0.10, delta=0.01)

    def test_threshold_search_obeys_minimum_validation_coverage(self):
        confidence = np.linspace(0.35, 0.95, 300)
        probabilities = np.column_stack((confidence, 1.0 - confidence))
        labels = (confidence >= 0.62).astype(int)
        thresholds, selection = _select_thresholds(probabilities, labels)
        self.assertGreaterEqual(selection["selection_coverage"], MIN_POLICY_COVERAGE)
        self.assertGreaterEqual(thresholds["min_top_probability"], 0.5)
        self.assertGreaterEqual(thresholds["min_top_two_margin"], 0.0)
        self.assertLessEqual(thresholds["max_normalized_entropy"], 1.0)


if __name__ == "__main__":
    unittest.main()
