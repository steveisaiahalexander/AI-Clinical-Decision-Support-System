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

    print("=" * 80)
    print("XGBoost V2 - TARGETED HYPERPARAMETER SEARCH")
    print("=" * 80)

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
    # 3. Moderate class weighting
    # ---------------------------------------------------------

    print("\n[3/5] Calculating alpha=0.25 sample weights...")

    balanced_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_dev,
    )

    sample_weights = balanced_weights ** 0.25

    print(
        f"Weight range: "
        f"{sample_weights.min():.4f} → "
        f"{sample_weights.max():.4f}"
    )

    # ---------------------------------------------------------
    # 4. Targeted configurations
    # ---------------------------------------------------------

    configurations = [

        {
            "name": "depth5_regular",
            "n_estimators": 900,
            "learning_rate": 0.075,
            "max_depth": 5,
            "min_child_weight": 1,
            "gamma": 0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },

        {
            "name": "depth6_more_trees",
            "n_estimators": 1000,
            "learning_rate": 0.075,
            "max_depth": 6,
            "min_child_weight": 1,
            "gamma": 0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },

        {
            "name": "depth7",
            "n_estimators": 900,
            "learning_rate": 0.075,
            "max_depth": 7,
            "min_child_weight": 1,
            "gamma": 0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },

        {
            "name": "depth8_regular",
            "n_estimators": 800,
            "learning_rate": 0.075,
            "max_depth": 8,
            "min_child_weight": 1,
            "gamma": 0.1,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },

        {
            "name": "depth6_child3",
            "n_estimators": 900,
            "learning_rate": 0.075,
            "max_depth": 6,
            "min_child_weight": 3,
            "gamma": 0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },

        {
            "name": "depth7_child3",
            "n_estimators": 900,
            "learning_rate": 0.075,
            "max_depth": 7,
            "min_child_weight": 3,
            "gamma": 0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },

        {
            "name": "full_features",
            "n_estimators": 800,
            "learning_rate": 0.075,
            "max_depth": 6,
            "min_child_weight": 1,
            "gamma": 0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },

        {
            "name": "lower_lr_deeper",
            "n_estimators": 1200,
            "learning_rate": 0.05,
            "max_depth": 7,
            "min_child_weight": 1,
            "gamma": 0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    ]

    results = []

    print("\n[4/5] Running targeted configurations...")
    print(f"Total configurations: {len(configurations)}")

    for i, config in enumerate(configurations, start=1):

        print("\n" + "=" * 80)
        print(
            f"CONFIGURATION {i}/{len(configurations)}: "
            f"{config['name']}"
        )
        print("=" * 80)

        print(
            f"Trees={config['n_estimators']}, "
            f"LR={config['learning_rate']}, "
            f"Depth={config['max_depth']}, "
            f"Child={config['min_child_weight']}, "
            f"Gamma={config['gamma']}, "
            f"Subsample={config['subsample']}, "
            f"Columns={config['colsample_bytree']}"
        )

        model = XGBClassifier(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            min_child_weight=config["min_child_weight"],
            gamma=config["gamma"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],

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
            sample_weight=sample_weights,
        )

        training_time = time.time() - start_time

        y_pred = model.predict(X_val)

        metrics = evaluate_model(y_val, y_pred)

        result = {
            **config,
            "training_time_minutes": round(
                training_time / 60,
                2,
            ),
            **metrics,
        }

        results.append(result)

        print("\nRESULT:")
        print(
            f"Accuracy:    "
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
    # 5. Final comparison
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL HYPERPARAMETER COMPARISON")
    print("=" * 80)

    sorted_results = sorted(
        results,
        key=lambda x: x["accuracy"],
        reverse=True,
    )

    print(
        "\nConfiguration              Accuracy    Weighted F1    Macro F1"
    )
    print("-" * 80)

    for result in sorted_results:

        print(
            f"{result['name']:<25}"
            f"{result['accuracy'] * 100:>8.2f}%"
            f"{result['weighted_f1']:>15.4f}"
            f"{result['macro_f1']:>13.4f}"
        )

    best_accuracy = sorted_results[0]

    best_weighted_f1 = max(
        results,
        key=lambda x: x["weighted_f1"],
    )

    best_macro_f1 = max(
        results,
        key=lambda x: x["macro_f1"],
    )

    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)

    print(
        f"\nBest Accuracy:"
        f" {best_accuracy['name']}"
        f" → {best_accuracy['accuracy'] * 100:.2f}%"
    )

    print(
        f"Best Weighted F1:"
        f" {best_weighted_f1['name']}"
        f" → {best_weighted_f1['weighted_f1']:.4f}"
    )

    print(
        f"Best Macro F1:"
        f" {best_macro_f1['name']}"
        f" → {best_macro_f1['macro_f1']:.4f}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output = {
        "experiment": "xgboost_v2_targeted_hyperparameter_search",
        "weighting": "balanced_weights^0.25",
        "validation_samples": int(len(X_val)),
        "features": int(X_train.shape[1]),
        "classes": int(len(np.unique(y_train))),
        "results": results,
        "best_by_accuracy": best_accuracy,
        "best_by_weighted_f1": best_weighted_f1,
        "best_by_macro_f1": best_macro_f1,
    }

    output_file = (
        REPORT_DIR /
        "xgboost_v2_tuning_results.json"
    )

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)

    print("\nResults saved to:")
    print(output_file)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()