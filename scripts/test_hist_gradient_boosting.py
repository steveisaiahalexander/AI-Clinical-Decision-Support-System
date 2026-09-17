"""
Experiment: HistGradientBoosting

Compares HistGradientBoosting against the current
Gradient Boosting configuration using ONLY the
training/development data and validation split.

The final test set is NOT used.
"""

import time
import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.core.config import Config


def evaluate_model(name, model, X_val, y_val):
    """Evaluate model on validation data."""

    start = time.time()

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)

    weighted_precision = precision_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0
    )

    macro_precision = precision_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    elapsed = time.time() - start

    results = {
        "model": name,
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "prediction_time_seconds": elapsed,
    }

    return results


def main():

    print("=" * 75)
    print("HISTGRADIENTBOOSTING EXPERIMENT")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load training data ONLY
    # ---------------------------------------------------------
    print("\n[1/5] Loading training data...")

    X_train_path = Config.PROCESSED_DATA_DIR / "X_train.pkl"
    y_train_path = Config.PROCESSED_DATA_DIR / "y_train.pkl"

    X_train = joblib.load(X_train_path)
    y_train = joblib.load(y_train_path)

    print(f"Training data: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")

    # ---------------------------------------------------------
    # 2. Create validation split
    # ---------------------------------------------------------
    print("\n[2/5] Creating validation split...")

    X_dev, X_val, y_dev, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    print(f"Development training: {X_dev.shape}")
    print(f"Validation: {X_val.shape}")

    # ---------------------------------------------------------
    # 3. Calculate balanced sample weights
    # ---------------------------------------------------------
    print("\n[3/5] Calculating class-balanced sample weights...")

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_dev
    )

    print(f"Sample weights: {len(sample_weights)}")
    print(f"Min weight: {sample_weights.min():.4f}")
    print(f"Max weight: {sample_weights.max():.4f}")

    # ---------------------------------------------------------
    # 4. Train HistGradientBoosting
    # ---------------------------------------------------------
    print("\n[4/5] Training HistGradientBoosting...")

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42
    )

    start_time = time.time()

    model.fit(
        X_dev,
        y_dev,
        sample_weight=sample_weights
    )

    training_time = time.time() - start_time

    print("\nTraining completed.")
    print(f"Training time: {training_time / 60:.2f} minutes")

    # ---------------------------------------------------------
    # 5. Validation evaluation
    # ---------------------------------------------------------
    print("\n[5/5] Evaluating on validation set...")

    results = evaluate_model(
        "HistGradientBoosting",
        model,
        X_val,
        y_val
    )

    print("\n" + "=" * 75)
    print("VALIDATION RESULTS")
    print("=" * 75)

    print(f"\nAccuracy          : {results['accuracy']:.4f} "
          f"({results['accuracy'] * 100:.2f}%)")

    print(f"Weighted Precision: "
          f"{results['weighted_precision']:.4f}")

    print(f"Weighted Recall   : "
          f"{results['weighted_recall']:.4f}")

    print(f"Weighted F1       : "
          f"{results['weighted_f1']:.4f}")

    print(f"Macro Precision   : "
          f"{results['macro_precision']:.4f}")

    print(f"Macro Recall      : "
          f"{results['macro_recall']:.4f}")

    print(f"Macro F1          : "
          f"{results['macro_f1']:.4f}")

    print("\n" + "=" * 75)
    print("CURRENT BEST VS HISTGRADIENTBOOSTING")
    print("=" * 75)

    print("\nCurrent Gradient Boosting validation:")
    print("  Accuracy : 56.56%")
    print("  Weighted F1: 57.73%")
    print("  Macro F1: 47.09%")

    print("\nHistGradientBoosting validation:")
    print(
        f"  Accuracy : {results['accuracy'] * 100:.2f}%"
    )
    print(
        f"  Weighted F1: {results['weighted_f1']:.4f}"
    )
    print(
        f"  Macro F1: {results['macro_f1']:.4f}"
    )

    # Save experiment results
    Config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = (
        Config.REPORTS_DIR /
        "hist_gradient_boosting_validation.json"
    )

    results["training_time_seconds"] = training_time

    with open(results_path, "w", encoding="utf-8") as file:
        import json
        json.dump(results, file, indent=4)

    print(f"\nResults saved to:")
    print(results_path)

    print("\nTest set was NOT used.")
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()