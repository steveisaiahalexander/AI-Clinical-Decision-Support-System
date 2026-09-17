from pathlib import Path
import json
import time

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def main():

    print("=" * 75)
    print("XGBoost - CLASS WEIGHT STRENGTH EXPERIMENT")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    print("\n[1/5] Loading processed training data...")

    X_train = joblib.load(PROCESSED_DIR / "X_train.pkl")
    y_train = joblib.load(PROCESSED_DIR / "y_train.pkl")

    print(f"X shape: {X_train.shape}")
    print(f"y shape: {y_train.shape}")

    # ---------------------------------------------------------
    # 2. Same validation split
    # ---------------------------------------------------------

    print("\n[2/5] Creating validation split...")

    X_dev, X_val, y_dev, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    print(f"Development samples: {len(X_dev)}")
    print(f"Validation samples: {len(X_val)}")

    # ---------------------------------------------------------
    # 3. Calculate full balanced weights
    # ---------------------------------------------------------

    print("\n[3/5] Calculating balanced weights...")

    balanced_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_dev,
    )

    print(
        f"Full balanced weight range: "
        f"{balanced_weights.min():.4f} → "
        f"{balanced_weights.max():.4f}"
    )

    # ---------------------------------------------------------
    # Weight strengths
    # ---------------------------------------------------------

    # Weight formula:
    #
    #     weight = balanced_weight ^ alpha
    #
    # alpha = 0      -> no weighting
    # alpha = 0.5    -> square-root weighting
    # alpha = 1      -> full balanced weighting

    alphas = [0.25, 0.35, 0.65, 0.80]

    results = []

    # ---------------------------------------------------------
    # 4. Train each configuration
    # ---------------------------------------------------------

    print("\n[4/5] Running weight-strength experiments...")

    for alpha in alphas:

        print("\n" + "-" * 75)
        print(f"Testing alpha = {alpha}")
        print("-" * 75)

        weights = balanced_weights ** alpha

        print(
            f"Weight range: "
            f"{weights.min():.4f} → "
            f"{weights.max():.4f}"
        )

        model = XGBClassifier(
            n_estimators=700,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,

            objective="multi:softprob",
            eval_metric="mlogloss",

            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )

        start_time = time.time()

        model.fit(
            X_dev,
            y_dev,
            sample_weight=weights,
        )

        training_time = time.time() - start_time

        y_pred = model.predict(X_val)

        metrics = evaluate_model(y_val, y_pred)

        result = {
            "alpha": alpha,
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "training_time_minutes": round(
                training_time / 60,
                2,
            ),
            **metrics,
        }

        results.append(result)

        print(
            f"\nAccuracy:    "
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Weighted F1: "
            f"{metrics['weighted_f1']:.4f}"
        )

        print(
            f"Macro F1:    "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Training:    "
            f"{training_time / 60:.2f} minutes"
        )

    # ---------------------------------------------------------
    # 5. Results summary
    # ---------------------------------------------------------

    print("\n" + "=" * 75)
    print("WEIGHT STRENGTH RESULTS")
    print("=" * 75)

    print(
        "\nAlpha        Accuracy       Weighted F1       Macro F1"
    )
    print("-" * 65)

    for result in results:

        print(
            f"{result['alpha']:<12.2f}"
            f"{result['accuracy'] * 100:>8.2f}%"
            f"{result['weighted_f1']:>17.4f}"
            f"{result['macro_f1']:>17.4f}"
        )

    # Find best by accuracy
    best_accuracy = max(
        results,
        key=lambda x: x["accuracy"],
    )

    # Find best by weighted F1
    best_weighted_f1 = max(
        results,
        key=lambda x: x["weighted_f1"],
    )

    # Find best by macro F1
    best_macro_f1 = max(
        results,
        key=lambda x: x["macro_f1"],
    )

    print("\n" + "=" * 75)
    print("BEST RESULTS")
    print("=" * 75)

    print(
        f"\nBest Accuracy:"
        f"  alpha={best_accuracy['alpha']}"
        f"  → {best_accuracy['accuracy'] * 100:.2f}%"
    )

    print(
        f"Best Weighted F1:"
        f"  alpha={best_weighted_f1['alpha']}"
        f"  → {best_weighted_f1['weighted_f1']:.4f}"
    )

    print(
        f"Best Macro F1:"
        f"  alpha={best_macro_f1['alpha']}"
        f"  → {best_macro_f1['macro_f1']:.4f}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output = {
        "experiment": "xgboost_class_weight_strength",
        "description": (
            "Tests different powers of balanced sample weights "
            "using weight = balanced_weight ^ alpha."
        ),
        "model_configuration": {
            "n_estimators": 700,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE,
        },
        "results": results,
        "best_by_accuracy": best_accuracy,
        "best_by_weighted_f1": best_weighted_f1,
        "best_by_macro_f1": best_macro_f1,
    }

    output_file = (
        REPORT_DIR /
        "xgboost_weight_strength_results.json"
    )

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)

    print("\nResults saved to:")
    print(output_file)

    print("\n" + "=" * 75)
    print("EXPERIMENT COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()