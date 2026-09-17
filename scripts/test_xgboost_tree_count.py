"""
XGBoost tree-count experiment.

Tests whether increasing the number of estimators beyond
500 continues to improve validation performance.

The test set is NOT used.
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
    print("XGBOOST TREE COUNT EXPERIMENT")
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
    # 3. Test tree counts
    # ---------------------------------------------------------
    tree_counts = [400, 500, 600, 700]

    results = []

    print("\n[3/4] Running tree-count experiments...")

    for n_trees in tree_counts:

        print("\n" + "-" * 75)
        print(f"Testing n_estimators = {n_trees}")
        print("-" * 75)

        model = XGBClassifier(
            n_estimators=n_trees,
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

        y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)

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

        result = {
            "n_estimators": n_trees,
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "training_time_seconds": training_time
        }

        results.append(result)

        print(
            f"Accuracy   : {accuracy * 100:.2f}%"
        )

        print(
            f"Weighted F1: {weighted_f1 * 100:.2f}%"
        )

        print(
            f"Macro F1   : {macro_f1 * 100:.2f}%"
        )

        print(
            f"Training time: {training_time / 60:.2f} minutes"
        )

    # ---------------------------------------------------------
    # 4. Final comparison
    # ---------------------------------------------------------
    results.sort(
        key=lambda x: x["accuracy"],
        reverse=True
    )

    print("\n" + "=" * 75)
    print("TREE COUNT RESULTS")
    print("=" * 75)

    for rank, result in enumerate(results, start=1):

        print(
            f"\n{rank}. {result['n_estimators']} trees"
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
        "xgboost_tree_count_results.json"
    )

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n[4/4] Results saved to:")
    print(results_path)

    print("\nTest set was NOT used.")
    print("Experiment complete.")


if __name__ == "__main__":
    main()