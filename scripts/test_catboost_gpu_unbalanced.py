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


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 75)
    print("CatBoost GPU - UNBALANCED VALIDATION EXPERIMENT")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    print("\n[1/5] Loading processed training data...")

    X_train = joblib.load(
        PROCESSED_DIR / "X_train.pkl"
    )

    y_train = joblib.load(
        PROCESSED_DIR / "y_train.pkl"
    )

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
    print(f"Features: {X_dev.shape[1]}")
    print(f"Classes: {len(np.unique(y_train))}")

    # ---------------------------------------------------------
    # 3. CatBoost GPU
    # ---------------------------------------------------------

    print("\n[3/5] Creating unbalanced CatBoost GPU model...")

    model = CatBoostClassifier(
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

    print("\nConfiguration:")
    print("  iterations       = 700")
    print("  learning_rate    = 0.1")
    print("  depth            = 6")
    print("  class weighting  = NONE")
    print("  task_type        = GPU")

    # ---------------------------------------------------------
    # 4. Train
    # ---------------------------------------------------------

    print("\n[4/5] Training CatBoost on GPU...")

    start_time = time.time()

    model.fit(
        X_dev,
        y_dev,
        eval_set=(X_val, y_val),
        early_stopping_rounds=75,
    )

    training_time = time.time() - start_time

    print(
        f"\nTraining completed in "
        f"{training_time / 60:.2f} minutes."
    )

    # ---------------------------------------------------------
    # 5. Evaluate
    # ---------------------------------------------------------

    print("\n[5/5] Evaluating validation set...")

    y_pred = model.predict(X_val)
    y_pred = np.asarray(y_pred).reshape(-1)

    accuracy = accuracy_score(
        y_val,
        y_pred,
    )

    weighted_precision = precision_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 75)
    print("CATBOOST UNBALANCED VALIDATION RESULTS")
    print("=" * 75)

    print(
        f"\nAccuracy:           "
        f"{accuracy:.4f} ({accuracy * 100:.2f}%)"
    )

    print(
        f"Weighted Precision: {weighted_precision:.4f}"
    )
    print(
        f"Weighted Recall:    {weighted_recall:.4f}"
    )
    print(
        f"Weighted F1:        {weighted_f1:.4f}"
    )

    print(
        f"\nMacro Precision:    {macro_precision:.4f}"
    )
    print(
        f"Macro Recall:       {macro_recall:.4f}"
    )
    print(
        f"Macro F1:           {macro_f1:.4f}"
    )

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    xgb_accuracy = 0.7175
    xgb_weighted_f1 = 0.7139
    xgb_macro_f1 = 0.5417

    print("\n" + "=" * 75)
    print("COMPARISON WITH BEST XGBOOST")
    print("=" * 75)

    print(
        "\n                         XGBoost       CatBoost"
    )
    print("-" * 60)

    print(
        f"Accuracy                 "
        f"{xgb_accuracy * 100:>7.2f}%"
        f"       {accuracy * 100:>7.2f}%"
    )

    print(
        f"Weighted F1              "
        f"{xgb_weighted_f1:>10.4f}"
        f"       {weighted_f1:>10.4f}"
    )

    print(
        f"Macro F1                 "
        f"{xgb_macro_f1:>10.4f}"
        f"       {macro_f1:>10.4f}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    results = {
        "model": "CatBoost",
        "experiment": "gpu_unbalanced_validation",

        "validation_samples": int(len(X_val)),
        "development_samples": int(len(X_dev)),
        "features": int(X_train.shape[1]),
        "classes": int(len(np.unique(y_train))),

        "iterations": 700,
        "learning_rate": 0.1,
        "depth": 6,

        "class_weighting": "none",

        "task_type": "GPU",
        "device": "0",

        "best_iteration": int(
            model.get_best_iteration()
        ),

        "training_time_minutes": round(
            training_time / 60,
            2,
        ),

        "accuracy": float(accuracy),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
    }

    output_file = (
        REPORT_DIR /
        "catboost_gpu_unbalanced_validation.json"
    )

    with open(output_file, "w") as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    print("\nResults saved to:")
    print(output_file)

    print("\n" + "=" * 75)
    print("EXPERIMENT COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()