"""
Targeted XGBoost hyperparameter tuning.

The test set is NOT used.
All configurations are evaluated on the same validation split.
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


def evaluate(model, X_val, y_val):

    y_pred = model.predict(X_val)

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "weighted_precision": precision_score(
            y_val, y_pred,
            average="weighted",
            zero_division=0
        ),
        "weighted_recall": recall_score(
            y_val, y_pred,
            average="weighted",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_val, y_pred,
            average="weighted",
            zero_division=0
        ),
        "macro_precision": precision_score(
            y_val, y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            y_val, y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_val, y_pred,
            average="macro",
            zero_division=0
        ),
    }


def main():

    print("=" * 75)
    print("TARGETED XGBOOST HYPERPARAMETER TUNING")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------
    print("\n[1/4] Loading training data...")

    X_train = joblib.load(
        Config.PROCESSED_DATA_DIR / "X_train.pkl"
    )

    y_train = joblib.load(
        Config.PROCESSED_DATA_DIR / "y_train.pkl"
    )

    # ---------------------------------------------------------
    # 2. Validation split
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
    # 3. Configurations
    # ---------------------------------------------------------
    configurations = {

        "baseline": {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },

        "more_trees": {
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },

        "shallower": {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 4,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },

        "deeper": {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 8,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },

        "regularized": {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },

        "more_subsampling": {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
        },
    }

    results = []

    # ---------------------------------------------------------
    # 4. Train and compare
    # ---------------------------------------------------------
    print("\n[3/4] Running experiments...")

    for name, params in configurations.items():

        print("\n" + "-" * 75)
        print(f"Configuration: {name}")
        print("-" * 75)

        print(params)

        start_time = time.time()

        model = XGBClassifier(
            **params,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        )

        model.fit(
            X_dev,
            y_dev,
            sample_weight=sample_weights
        )

        training_time = time.time() - start_time

        metrics = evaluate(
            model,
            X_val,
            y_val
        )

        result = {
            "name": name,
            **params,
            **metrics,
            "training_time_seconds": training_time
        }

        results.append(result)

        print(
            f"Accuracy   : {metrics['accuracy'] * 100:.2f}%"
        )
        print(
            f"Weighted F1: {metrics['weighted_f1'] * 100:.2f}%"
        )
        print(
            f"Macro F1   : {metrics['macro_f1'] * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    results.sort(
        key=lambda x: x["accuracy"],
        reverse=True
    )

    print("\n" + "=" * 75)
    print("XGBOOST TUNING RESULTS")
    print("=" * 75)

    for i, result in enumerate(results, start=1):

        print(
            f"\n{i}. {result['name']}"
        )

        print(
            f"   Accuracy    : "
            f"{result['accuracy'] * 100:.2f}%"
        )

        print(
            f"   Weighted F1 : "
            f"{result['weighted_f1'] * 100:.2f}%"
        )

        print(
            f"   Macro F1    : "
            f"{result['macro_f1'] * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------
    print("\n[4/4] Saving tuning results...")

    Config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = (
        Config.REPORTS_DIR /
        "xgboost_tuning_results.json"
    )

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nResults saved to:")
    print(results_path)

    print("\nTest set was NOT used.")
    print("Tuning complete.")


if __name__ == "__main__":
    main()