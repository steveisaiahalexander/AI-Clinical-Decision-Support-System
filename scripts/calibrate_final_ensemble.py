"""Fit and validate probability calibration using X_train only.

Temporary validation models using the locked ensemble hyperparameters are
trained on a reproducible development subset. The locked inference artifacts
are never changed. X_test and y_test are deliberately not read by this script.
"""

import argparse
import json
from pathlib import Path
import sys

import joblib
import matplotlib
import numpy as np
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.calibration import calibration_metrics, fit_temperature, temperature_scale
from src.models.uncertainty import uncertainty_summary


RANDOM_STATE = 42
XGB_WEIGHT = 0.675
CATBOOST_WEIGHT = 0.325
SPLIT_SEEDS = (42, 43, 44, 45)
ECE_BINS = 15
MIN_POLICY_COVERAGE = 0.50


def _partition_indices(labels):
    indices = np.arange(len(labels))
    fit_indices, remainder = train_test_split(
        indices,
        test_size=0.40,
        stratify=labels,
        random_state=SPLIT_SEEDS[0],
    )
    calibration_indices, remainder = train_test_split(
        remainder,
        test_size=0.75,
        stratify=labels[remainder],
        random_state=SPLIT_SEEDS[1],
    )
    selection_indices, remainder = train_test_split(
        remainder,
        test_size=2.0 / 3.0,
        stratify=labels[remainder],
        random_state=SPLIT_SEEDS[2],
    )
    policy_indices, evaluation_indices = train_test_split(
        remainder,
        test_size=0.50,
        stratify=labels[remainder],
        random_state=SPLIT_SEEDS[3],
    )
    return {
        "model_fit": fit_indices,
        "calibration_fit": calibration_indices,
        "method_selection": selection_indices,
        "threshold_selection": policy_indices,
        "validation_evaluation": evaluation_indices,
    }


def _fit_validation_models(X, y, device):
    sample_weights = compute_sample_weight(class_weight="balanced", y=y) ** 0.25
    xgb_model = XGBClassifier(
        n_estimators=900,
        learning_rate=0.075,
        max_depth=5,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        device="cuda" if device == "cuda" else "cpu",
    )
    xgb_model.fit(X, y, sample_weight=sample_weights)

    catboost_parameters = dict(
        iterations=700,
        learning_rate=0.1,
        depth=6,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        task_type="GPU" if device == "cuda" else "CPU",
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    if device == "cuda":
        catboost_parameters.update(devices="0", gpu_ram_part=0.85)
    catboost_model = CatBoostClassifier(**catboost_parameters)
    catboost_model.fit(X, y)
    return xgb_model, catboost_model


def _predict_ensemble(models, X):
    xgb_model, catboost_model = models
    xgb_classes = np.asarray(xgb_model.classes_)
    catboost_classes = np.asarray(catboost_model.classes_)
    catboost_order = [
        int(np.flatnonzero(catboost_classes == label)[0]) for label in xgb_classes
    ]
    xgb_probabilities = xgb_model.predict_proba(X)
    catboost_probabilities = catboost_model.predict_proba(X)[:, catboost_order]
    return (
        XGB_WEIGHT * xgb_probabilities
        + CATBOOST_WEIGHT * catboost_probabilities
    )


def _uncertainty_rows(probabilities):
    values = [uncertainty_summary(row) for row in probabilities]
    return {
        name: np.asarray([item[name] for item in values], dtype=float)
        for name in (
            "top_probability",
            "top_two_margin",
            "normalized_entropy",
        )
    }


def _select_thresholds(probabilities, labels):
    summary = _uncertainty_rows(probabilities)
    correct = np.argmax(probabilities, axis=1) == labels
    quantiles = np.linspace(0.0, 1.0, 11)
    confidence_candidates = np.unique(
        np.quantile(summary["top_probability"], quantiles)
    )
    margin_candidates = np.unique(
        np.quantile(summary["top_two_margin"], quantiles)
    )
    entropy_candidates = np.unique(
        np.concatenate(
            (
                np.quantile(summary["normalized_entropy"], quantiles),
                np.asarray([1.0]),
            )
        )
    )

    best = None
    for min_probability in confidence_candidates:
        for min_margin in margin_candidates:
            base_acceptance = (
                (summary["top_probability"] >= min_probability)
                & (summary["top_two_margin"] >= min_margin)
            )
            for max_entropy in entropy_candidates:
                accepted = base_acceptance & (
                    summary["normalized_entropy"] <= max_entropy
                )
                coverage = float(accepted.mean())
                if coverage < MIN_POLICY_COVERAGE:
                    continue
                selective_accuracy = float(correct[accepted].mean())
                candidate = (
                    selective_accuracy,
                    coverage,
                    float(min_probability),
                    float(min_margin),
                    float(max_entropy),
                )
                if best is None or candidate[:2] > best[:2]:
                    best = candidate

    if best is None:
        raise RuntimeError("No abstention thresholds met the minimum coverage.")
    thresholds = {
        "min_top_probability": best[2],
        "min_top_two_margin": best[3],
        "max_normalized_entropy": best[4],
    }
    return thresholds, {
        "target_minimum_coverage": MIN_POLICY_COVERAGE,
        "selection_coverage": best[1],
        "selection_accuracy_when_accepted": best[0],
    }


def _evaluate_policy(probabilities, labels, thresholds):
    summary = _uncertainty_rows(probabilities)
    accepted = (
        (summary["top_probability"] >= thresholds["min_top_probability"])
        & (summary["top_two_margin"] >= thresholds["min_top_two_margin"])
        & (
            summary["normalized_entropy"]
            <= thresholds["max_normalized_entropy"]
        )
    )
    correct = np.argmax(probabilities, axis=1) == labels
    return {
        "sample_count": int(len(labels)),
        "coverage": float(accepted.mean()),
        "abstention_rate": float(1.0 - accepted.mean()),
        "accuracy_when_accepted": (
            float(correct[accepted].mean()) if accepted.any() else None
        ),
        "accuracy_on_all_rows": float(correct.mean()),
    }


def _plot_reliability(raw_metrics, selected_metrics, path):
    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="#8b948d", linestyle="--", label="Ideal")
    for metrics, label, color in (
        (raw_metrics, "Raw ensemble", "#3d7b58"),
        (selected_metrics, "Selected output", "#b26936"),
    ):
        bins = metrics["bins"]
        axis.plot(
            [item["mean_confidence"] for item in bins],
            [item["accuracy"] for item in bins],
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean top-label probability",
        ylabel="Observed top-label accuracy",
        title="Development validation reliability",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _choose_device(requested):
    if requested != "auto":
        return requested
    try:
        if get_gpu_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-validation-refits",
        action="store_true",
        help="Explicitly allow temporary model copies fitted on X_train development data.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not args.allow_validation_refits:
        parser.error(
            "This script fits disposable validation copies. Pass "
            "--allow-validation-refits only when that is authorized."
        )

    processed = PROJECT_ROOT / "data" / "processed"
    X = joblib.load(processed / "X_train.pkl")
    y = np.asarray(joblib.load(processed / "y_train.pkl"), dtype=int).reshape(-1)
    if len(X) != len(y) or X.shape[1] != 174:
        raise ValueError("Unexpected X_train/y_train dimensions.")
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration requires at least two disease classes.")

    splits = _partition_indices(y)
    device = _choose_device(args.device)
    print(
        "Development-only calibration: "
        f"{len(splits['model_fit']):,} model-fit rows; "
        f"{len(splits['calibration_fit']):,} calibration rows; "
        f"{len(splits['method_selection']):,} method-selection rows; "
        f"{len(splits['threshold_selection']):,} threshold-selection rows; "
        f"{len(splits['validation_evaluation']):,} final validation rows."
    )
    print(f"Fitting disposable validation models on {device}...")
    models = _fit_validation_models(
        X.iloc[splits["model_fit"]],
        y[splits["model_fit"]],
        device,
    )

    probabilities = {
        name: _predict_ensemble(models, X.iloc[indices])
        for name, indices in splits.items()
        if name != "model_fit"
    }
    temperature = fit_temperature(
        probabilities["calibration_fit"],
        y[splits["calibration_fit"]],
    )
    raw_selection = calibration_metrics(
        probabilities["method_selection"],
        y[splits["method_selection"]],
        n_bins=ECE_BINS,
    )
    scaled_selection = calibration_metrics(
        temperature_scale(probabilities["method_selection"], temperature),
        y[splits["method_selection"]],
        n_bins=ECE_BINS,
    )
    use_temperature = scaled_selection["log_loss"] < raw_selection["log_loss"]
    method = "temperature_scaling" if use_temperature else "raw_ensemble"
    applied_temperature = temperature if use_temperature else 1.0

    policy_probabilities = temperature_scale(
        probabilities["threshold_selection"], applied_temperature
    )
    thresholds, threshold_selection = _select_thresholds(
        policy_probabilities,
        y[splits["threshold_selection"]],
    )
    evaluation_raw = probabilities["validation_evaluation"]
    evaluation_selected = temperature_scale(evaluation_raw, applied_temperature)
    raw_metrics = calibration_metrics(
        evaluation_raw,
        y[splits["validation_evaluation"]],
        n_bins=ECE_BINS,
    )
    selected_metrics = calibration_metrics(
        evaluation_selected,
        y[splits["validation_evaluation"]],
        n_bins=ECE_BINS,
    )

    config_dir = PROJECT_ROOT / "configs"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    config_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reliability_path = figures_dir / "calibration_reliability.png"
    _plot_reliability(raw_metrics, selected_metrics, reliability_path)

    calibration_config = {
        "calibration": {
            "method": method,
            "temperature": float(applied_temperature),
            "fitted_temperature": float(temperature),
            "selection_rule": (
                "Use scalar temperature scaling only when its log loss is lower "
                "on the independent method-selection subset."
            ),
        },
        "abstention": {
            "thresholds": thresholds,
            "selection_method": (
                "Maximize top-1 accuracy among threshold combinations retaining "
                "at least 50% coverage on the threshold-selection subset."
            ),
            "selection_results": threshold_selection,
            "validation_results": _evaluate_policy(
                evaluation_selected,
                y[splits["validation_evaluation"]],
                thresholds,
            ),
            "thresholds_clinically_validated": False,
        },
        "validation_metrics": {
            "raw": raw_metrics,
            "selected": selected_metrics,
        },
        "selection_metrics": {
            "raw": raw_selection,
            "temperature_scaled": scaled_selection,
        },
        "provenance": {
            "dataset": "data/processed/X_train.pkl and y_train.pkl only",
            "held_out_test_set_used": False,
            "training_samples": int(len(y)),
            "model_fit_samples": int(len(splits["model_fit"])),
            "calibration_fit_samples": int(len(splits["calibration_fit"])),
            "method_selection_samples": int(len(splits["method_selection"])),
            "threshold_selection_samples": int(len(splits["threshold_selection"])),
            "validation_evaluation_samples": int(len(splits["validation_evaluation"])),
            "split_seeds": list(SPLIT_SEEDS),
            "device": device,
            "temporary_models_saved": False,
            "final_models_modified": False,
            "ensemble_weights": {
                "xgboost": XGB_WEIGHT,
                "catboost": CATBOOST_WEIGHT,
            },
        },
    }

    config_path = config_dir / "ensemble_calibration.json"
    metrics_path = reports_dir / "ensemble_calibration_validation_metrics.json"
    with open(config_path, "w", encoding="utf-8") as output:
        json.dump(calibration_config, output, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as output:
        json.dump(calibration_config["validation_metrics"], output, indent=2)

    print(f"Calibration method: {method}; temperature={applied_temperature:.6f}")
    print(f"Raw validation metrics: {json.dumps(raw_metrics, separators=(',', ':'))}")
    print(f"Selected validation metrics: {json.dumps(selected_metrics, separators=(',', ':'))}")
    print(f"Validation abstention policy: {calibration_config['abstention']['validation_results']}")
    print(f"Calibration policy: {config_path}")
    print(f"Validation metrics: {metrics_path}")
    print(f"Reliability plot: {reliability_path}")


if __name__ == "__main__":
    main()
