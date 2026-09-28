"""Probability-distribution uncertainty and conservative abstention logic."""

from typing import Any, Dict, List

import numpy as np


def uncertainty_summary(probabilities: Any) -> Dict[str, float]:
    """Summarize top-class separation and normalized predictive entropy."""
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("At least two finite, non-negative class probabilities are required.")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("Class probabilities must have a positive sum.")
    values = values / total
    ordered = np.sort(values)[::-1]
    entropy = float(-np.sum(values * np.log(np.clip(values, 1e-15, 1.0))))
    return {
        "top_probability": float(ordered[0]),
        "top_two_margin": float(ordered[0] - ordered[1]),
        "entropy": entropy,
        "normalized_entropy": float(np.clip(entropy / np.log(values.size), 0.0, 1.0)),
    }


def abstention_decision(
    uncertainty: Dict[str, float],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Return insufficient evidence when any validation-set threshold fails."""
    required = (
        "min_top_probability",
        "min_top_two_margin",
        "max_normalized_entropy",
    )
    if any(key not in thresholds for key in required):
        raise ValueError("All three abstention thresholds must be provided.")
    reasons: List[str] = []
    if uncertainty["top_probability"] < thresholds["min_top_probability"]:
        reasons.append("low_top_probability")
    if uncertainty["top_two_margin"] < thresholds["min_top_two_margin"]:
        reasons.append("low_top_two_separation")
    if uncertainty["normalized_entropy"] > thresholds["max_normalized_entropy"]:
        reasons.append("high_distribution_entropy")
    return {
        "abstained": bool(reasons),
        "decision": "insufficient_evidence" if reasons else "ranked_prediction",
        "reasons": reasons,
    }
