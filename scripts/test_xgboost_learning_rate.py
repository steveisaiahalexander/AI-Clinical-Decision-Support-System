"""
XGBoost learning-rate experiment.

Tests learning rates for the current best XGBoost architecture.

IMPORTANT:
The untouched test set is NOT used.
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


def evaluate_model(model, X_val, y_val):

    y_pred = model.predict(X_val)

    return {
        "accuracy": accuracy_score(y_val, y_pred),

        "weighted_precision": precision_score(
            y_val,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "weighted_recall": recall_score(
            y_val,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "weighted_f1": f1_score(
            y_val,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "macro_precision": precision_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0
        ),
    }


def main():

    print("=" * 75)
    print("XGBOOST LEARNING RATE EXPERIMENT")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load training data
    # ---------------------------------------------------------
    print("\n[1/4] Loading training data...")

    X_train = joblib.load(
        Config.PROCESSED_DATA_DIR / "X_train.pkl"
    )

    y_train = joblib.load(
        Config.PROCESSED_DATA_DIR / "y_train.pkl"
    )

    print(f"Training data: {X_train.shape}")
    print(f"Labels: {y_train.shape}")

    # ---------------------------------------------------------
    # 2. Same validation split
    # ---------------------------------------------------------
    print("\n[2/4] Creating validation split...")

    X_dev, X_val, y_dev, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_dev
    )

    print(f"Development: {X_dev.shape}")
    print(f"Validation:  {X_val.shape}")

    # ---------------------------------------------------------
    # 3. Learning-rate configurations
    # ---------------------------------------------------------
    configurations = {
        "lr_0.05": 0.05,
        "lr_0.075": 0.075,
        "lr_0.10": 0.10,
    }

    results = []

    print("\n[3/4] Running learning-rate experiments...")

    for name, learning_rate in configurations.items():

        print("\n" + "-" * 75)
        print(f"Configuration: {name}")
        print("-" * 75)

        print(f"n_estimators  : 700")
        print(f"learning_rate : {learning_rate}")
        print(f"max_depth     : 6")
        print(f"subsample     : 0.8")
        print(f"colsample     : 0.8")

        model = XGBClassifier(
            n_estimators=700,
            learning_rate=learning_rate,
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

        metrics = evaluate_model(
            model,
            X_val,
            y_val
        )

        result = {
            "name": name,
            "n_estimators": 700,
            "learning_rate": learning_rate,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            **metrics,
            "training_time_seconds": training_time
        }

        results.append(result)

        print(
            f"\nAccuracy   : "
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Weighted F1: "
            f"{metrics['weighted_f1'] * 100:.2f}%"
        )

        print(
            f"Macro F1   : "
            f"{metrics['macro_f1'] * 100:.2f}%"
        )

        print(
            f"Training time: "
            f"{training_time / 60:.2f} minutes"
        )

    # ---------------------------------------------------------
    # 4. Compare results
    # ---------------------------------------------------------
    results.sort(
        key=lambda x: x["accuracy"],
        reverse=True
    )

    print("\n" + "=" * 75)
    print("LEARNING RATE RESULTS")
    print("=" * 75)

    for rank, result in enumerate(results, start=1):

        print(
            f"\n{rank}. {result['name']}"
        )

        print(
            f"   Accuracy   : "
            f"{result['accuracy'] * 100:.2f}%"
        )

        print(
            f"   Weighted F1: "
            f"{result['weighted_f1'] * 100:.2f}%"
        )

        print(
            f"   Macro F1   : "
            f"{result['macro_f1'] * 100:.2f}%"
        )

    Config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = (
        Config.REPORTS_DIR /
        "xgboost_learning_rate_results.json"
    )

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n[4/4] Results saved to:")
    print(results_path)

    print("\nTest set was NOT used.")
    print("Experiment complete.")


if __name__ == "__main__":
    main()