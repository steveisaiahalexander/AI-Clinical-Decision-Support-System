"""
XGBoost validation experiment.

IMPORTANT:
The untouched test set is NOT used.
Only the training data is split into development/validation sets.
"""

import json
import time
import joblib

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.core.config import Config


def main():

    print("=" * 75)
    print("XGBOOST VALIDATION EXPERIMENT")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load training data
    # ---------------------------------------------------------
    print("\n[1/5] Loading training data...")

    X_train_path = Config.PROCESSED_DATA_DIR / "X_train.pkl"
    y_train_path = Config.PROCESSED_DATA_DIR / "y_train.pkl"

    X_train = joblib.load(X_train_path)
    y_train = joblib.load(y_train_path)

    print(f"Training data: {X_train.shape}")
    print(f"Labels: {y_train.shape}")

    # ---------------------------------------------------------
    # 2. Validation split
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
    # 3. Balanced sample weights
    # ---------------------------------------------------------
    print("\n[3/5] Calculating class-balanced weights...")

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_dev
    )

    print(f"Sample weights: {len(sample_weights)}")
    print(f"Minimum weight: {sample_weights.min():.4f}")
    print(f"Maximum weight: {sample_weights.max():.4f}")

    # ---------------------------------------------------------
    # 4. Train XGBoost
    # ---------------------------------------------------------
    print("\n[4/5] Training XGBoost...")

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
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
    # 5. Evaluate validation set
    # ---------------------------------------------------------
    print("\n[5/5] Evaluating validation set...")

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

    print("\n" + "=" * 75)
    print("XGBOOST VALIDATION RESULTS")
    print("=" * 75)

    print(
        f"\nAccuracy          : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall   : {weighted_recall:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")
    print(f"Macro Precision   : {macro_precision:.4f}")
    print(f"Macro Recall      : {macro_recall:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")

    print("\n" + "=" * 75)
    print("COMPARISON WITH CURRENT BEST")
    print("=" * 75)

    print("\nCurrent Gradient Boosting:")
    print("  Accuracy   : 56.56%")
    print("  Weighted F1: 57.73%")
    print("  Macro F1   : 47.09%")

    print("\nXGBoost:")
    print(f"  Accuracy   : {accuracy * 100:.2f}%")
    print(f"  Weighted F1: {weighted_f1 * 100:.2f}%")
    print(f"  Macro F1   : {macro_f1 * 100:.2f}%")

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------
    Config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {
        "model": "XGBoost",
        "training_samples": int(len(X_dev)),
        "validation_samples": int(len(X_val)),
        "features": int(X_train.shape[1]),
        "accuracy": float(accuracy),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "training_time_seconds": float(training_time),
        "test_set_used": False,
        "hyperparameters": {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": 42
        }
    }

    results_path = (
        Config.REPORTS_DIR /
        "xgboost_validation.json"
    )

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nResults saved to:")
    print(results_path)

    print("\nTest set was NOT used.")
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()