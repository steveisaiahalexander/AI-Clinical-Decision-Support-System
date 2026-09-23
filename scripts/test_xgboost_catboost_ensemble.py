from pathlib import Path
import json
import time

import joblib
import numpy as np
from catboost import CatBoostClassifier
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


def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

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
    print("XGBoost + CatBoost PROBABILITY ENSEMBLE")
    print("=" * 80)

    # =========================================================
    # 1. Load data
    # =========================================================

    print("\n[1/7] Loading processed training data...")

    X_train = joblib.load(
        PROCESSED_DIR / "X_train.pkl"
    )

    y_train = joblib.load(
        PROCESSED_DIR / "y_train.pkl"
    )

    print(f"X shape: {X_train.shape}")
    print(f"y shape: {y_train.shape}")

    # =========================================================
    # 2. Same validation split
    # =========================================================

    print("\n[2/7] Creating validation split...")

    X_dev, X_val, y_dev, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    print(f"Development samples: {len(X_dev)}")
    print(f"Validation samples: {len(X_val)}")

    # =========================================================
    # 3. XGBoost
    # =========================================================

    print("\n[3/7] Training best XGBoost model...")

    balanced_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_dev,
    )

    xgb_weights = balanced_weights ** 0.25

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
        device="cuda",
    )

    xgb_start = time.time()

    xgb_model.fit(
        X_dev,
        y_dev,
        sample_weight=xgb_weights,
    )

    xgb_time = time.time() - xgb_start

    print(
        f"XGBoost training time: "
        f"{xgb_time / 60:.2f} minutes"
    )

    # =========================================================
    # 4. CatBoost
    # =========================================================

    print("\n[4/7] Training CatBoost GPU model...")

    cat_model = CatBoostClassifier(
        iterations=700,
        learning_rate=0.1,
        depth=6,

        loss_function="MultiClass",
        eval_metric="Accuracy",

        random_seed=RANDOM_STATE,

        task_type="GPU",
        devices="0",
        gpu_ram_part=0.85,

        verbose=100,
        allow_writing_files=False,
    )

    cat_start = time.time()

    cat_model.fit(
        X_dev,
        y_dev,
        eval_set=(X_val, y_val),
        early_stopping_rounds=75,
    )

    cat_time = time.time() - cat_start

    print(
        f"\nCatBoost training time: "
        f"{cat_time / 60:.2f} minutes"
    )

    # =========================================================
    # 5. Generate probabilities
    # =========================================================

    print("\n[5/7] Generating validation probabilities...")

    print("Generating XGBoost probabilities...")

    xgb_probabilities = xgb_model.predict_proba(
        X_val
    )

    print("Generating CatBoost probabilities...")

    cat_probabilities = cat_model.predict_proba(
        X_val
    )

    print(
        f"XGBoost probability shape: "
        f"{xgb_probabilities.shape}"
    )

    print(
        f"CatBoost probability shape: "
        f"{cat_probabilities.shape}"
    )

    # Sanity check
    if xgb_probabilities.shape != cat_probabilities.shape:

        raise ValueError(
            "XGBoost and CatBoost probability "
            "matrices have different shapes."
        )

    # =========================================================
    # 6. Test ensemble weights
    # =========================================================

    print("\n[6/7] Testing probability mixtures...")

    ensemble_weights = [
        1.00,
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
    ]

    results = []

    for xgb_weight in ensemble_weights:

        cat_weight = 1.0 - xgb_weight

        print("\n" + "-" * 70)

        print(
            f"XGBoost weight: {xgb_weight:.2f}"
        )

        print(
            f"CatBoost weight: {cat_weight:.2f}"
        )

        # Weighted probability combination
        combined_probabilities = (
            xgb_weight * xgb_probabilities
            +
            cat_weight * cat_probabilities
        )

        y_pred = np.argmax(
            combined_probabilities,
            axis=1,
        )

        metrics = calculate_metrics(
            y_val,
            y_pred,
        )

        result = {
            "xgboost_weight": xgb_weight,
            "catboost_weight": cat_weight,
            **metrics,
        }

        results.append(result)

        print(
            f"Accuracy: "
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Weighted F1: "
            f"{metrics['weighted_f1']:.4f}"
        )

        print(
            f"Macro F1: "
            f"{metrics['macro_f1']:.4f}"
        )

    # =========================================================
    # 7. Results
    # =========================================================

    print("\n" + "=" * 80)
    print("ENSEMBLE RESULTS")
    print("=" * 80)

    print(
        "\nXGB Weight   CatBoost Weight   Accuracy   "
        "Weighted F1   Macro F1"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result['xgboost_weight']:>10.2f}"
            f"{result['catboost_weight']:>18.2f}"
            f"{result['accuracy'] * 100:>11.2f}%"
            f"{result['weighted_f1']:>14.4f}"
            f"{result['macro_f1']:>11.4f}"
        )

    best_accuracy = max(
        results,
        key=lambda x: x["accuracy"],
    )

    best_weighted_f1 = max(
        results,
        key=lambda x: x["weighted_f1"],
    )

    best_macro_f1 = max(
        results,
        key=lambda x: x["macro_f1"],
    )

    print("\n" + "=" * 80)
    print("BEST ENSEMBLE RESULTS")
    print("=" * 80)

    print(
        f"\nBest Accuracy:"
        f" XGB={best_accuracy['xgboost_weight']:.2f},"
        f" CatBoost={best_accuracy['catboost_weight']:.2f}"
        f" -> {best_accuracy['accuracy'] * 100:.2f}%"
    )

    print(
        f"Best Weighted F1:"
        f" XGB={best_weighted_f1['xgboost_weight']:.2f},"
        f" CatBoost={best_weighted_f1['catboost_weight']:.2f}"
        f" -> {best_weighted_f1['weighted_f1']:.4f}"
    )

    print(
        f"Best Macro F1:"
        f" XGB={best_macro_f1['xgboost_weight']:.2f},"
        f" CatBoost={best_macro_f1['catboost_weight']:.2f}"
        f" -> {best_macro_f1['macro_f1']:.4f}"
    )

    # =========================================================
    # Save
    # =========================================================

    output = {
        "experiment": "xgboost_catboost_probability_ensemble",

        "validation_samples": int(len(X_val)),
        "features": int(X_train.shape[1]),
        "classes": int(len(np.unique(y_train))),

        "xgboost_configuration": {
            "n_estimators": 900,
            "learning_rate": 0.075,
            "max_depth": 5,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "class_weighting": "balanced^0.25",
            "device": "cuda",
        },

        "catboost_configuration": {
            "iterations": 700,
            "learning_rate": 0.1,
            "depth": 6,
            "class_weighting": "none",
            "device": "GPU",
        },

        "results": results,

        "best_by_accuracy": best_accuracy,

        "best_by_weighted_f1": best_weighted_f1,

        "best_by_macro_f1": best_macro_f1,

        "xgboost_training_time_minutes": round(
            xgb_time / 60,
            2,
        ),

        "catboost_training_time_minutes": round(
            cat_time / 60,
            2,
        ),
    }

    output_file = (
        REPORT_DIR /
        "xgboost_catboost_ensemble_results.json"
    )

    with open(output_file, "w") as f:
        json.dump(
            output,
            f,
            indent=4,
        )

    print("\nResults saved to:")
    print(output_file)

    print("\n" + "=" * 80)
    print("ENSEMBLE EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()