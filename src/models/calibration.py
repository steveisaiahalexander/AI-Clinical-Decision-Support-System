"""Multiclass probability calibration and evaluation helpers."""

from typing import Any, Dict

import numpy as np
from scipy.optimize import minimize_scalar


def _validate_probabilities(probabilities: Any) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("Probabilities must be a non-empty 2D multiclass matrix.")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Probabilities must be finite and non-negative.")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("Each probability row must have a positive sum.")
    return values / totals


def temperature_scale(probabilities: Any, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling to a multiclass probability matrix."""
    values = _validate_probabilities(probabilities)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Temperature must be finite and greater than zero.")
    logits = np.log(np.clip(values, 1e-15, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: Any, labels: Any) -> float:
    """Fit one temperature by minimizing multiclass negative log likelihood."""
    values = _validate_probabilities(probabilities)
    targets = np.asarray(labels, dtype=int).reshape(-1)
    if len(targets) != len(values) or np.any(targets < 0) or np.any(targets >= values.shape[1]):
        raise ValueError("Labels must be valid class indices matching the probability rows.")
    rows = np.arange(len(targets))

    def objective(log_temperature: float) -> float:
        calibrated = temperature_scale(values, np.exp(log_temperature))
        return float(-np.mean(np.log(np.clip(calibrated[rows, targets], 1e-15, 1.0))))

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(-3.0, 3.0),
        options={"xatol": 1e-6},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature scaling optimization did not converge.")
    return float(np.exp(result.x))


def calibration_metrics(probabilities: Any, labels: Any, n_bins: int = 15) -> Dict[str, Any]:
    """Return multiclass Brier/log-loss and equal-width top-label ECE bins."""
    values = _validate_probabilities(probabilities)
    targets = np.asarray(labels, dtype=int).reshape(-1)
    if len(targets) != len(values) or np.any(targets < 0) or np.any(targets >= values.shape[1]):
        raise ValueError("Labels must be valid class indices matching the probability rows.")
    if n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")

    rows = np.arange(len(targets))
    predictions = np.argmax(values, axis=1)
    confidence = values[rows, predictions]
    correct = predictions == targets
    one_hot = np.eye(values.shape[1], dtype=float)[targets]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for index in range(n_bins):
        lower, upper = bin_edges[index], bin_edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == n_bins - 1 else confidence < upper
        )
        count = int(mask.sum())
        if count == 0:
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        ece += count / len(values) * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    return {
        "sample_count": int(len(values)),
        "multiclass_brier_score": float(np.mean(np.sum((values - one_hot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(np.clip(values[rows, targets], 1e-15, 1.0)))),
        "top_label_ece": float(ece),
        "ece_definition": "equal-width top-label expected calibration error",
        "bins": bins,
    }
