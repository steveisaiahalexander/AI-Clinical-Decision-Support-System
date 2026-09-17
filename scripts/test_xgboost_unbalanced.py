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
from xgboost import XGBClassifier


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 70)
    print("XGBoost - UNBALANCED VALIDATION EXPERIMENT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load processed training data
    # ---------------------------------------------------------

    print("\n[1/6] Loading processed training data...")

    X_train = joblib.load(PROCESSED_DIR / "X_train.pkl")
    y_train = joblib.load(PROCESSED_DIR / "y_train.pkl")

    print(f"X shape: {X_train.shape}")
    print(f"y shape: {y_train.shape}")

    # ---------------------------------------------------------
    # 2. Create development/validation split
    # ---------------------------------------------------------

    print("\n[2/6] Creating validation split...")

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
    # 3. Define UNBALANCED XGBoost
    # ---------------------------------------------------------

    print("\n[3/6] Creating unbalanced XGBoost model...")

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

    print("\nConfiguration:")
    print("  n_estimators      = 700")
    print("  learning_rate     = 0.1")
    print("  max_depth         = 6")
    print("  min_child_weight  = 1")
    print("  subsample         = 0.8")
    print("  colsample_bytree  = 0.8")
    print("  class balancing   = NONE")

    # ---------------------------------------------------------
    # 4. Train
    # ---------------------------------------------------------

    print("\n[4/6] Training model...")
    print("This may take around 10-15 minutes.")

    start_time = time.time()

    # IMPORTANT:
    # No sample_weight is supplied here.
    model.fit(X_dev, y_dev)

    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time / 60:.2f} minutes.")

    # ---------------------------------------------------------
    # 5. Validation prediction
    # ---------------------------------------------------------

    print("\n[5/6] Evaluating on validation set...")

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)

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

    # ---------------------------------------------------------
    # 6. Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(f"\nAccuracy:           {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall:    {weighted_recall:.4f}")
    print(f"Weighted F1:        {weighted_f1:.4f}")

    print(f"\nMacro Precision:    {macro_precision:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")

    # ---------------------------------------------------------
    # Comparison with balanced XGBoost
    # ---------------------------------------------------------

    balanced_accuracy = 0.7009
    balanced_weighted_f1 = 0.7020
    balanced_macro_f1 = 0.5404

    print("\n" + "=" * 70)
    print("COMPARISON WITH BALANCED XGBOOST")
    print("=" * 70)

    print("\n                         Balanced       Unbalanced")
    print("-" * 55)
    print(
        f"Accuracy                 {balanced_accuracy:.4f}"
        f"         {accuracy:.4f}"
    )
    print(
        f"Weighted F1              {balanced_weighted_f1:.4f}"
        f"         {weighted_f1:.4f}"
    )
    print(
        f"Macro F1                 {balanced_macro_f1:.4f}"
        f"         {macro_f1:.4f}"
    )

    accuracy_change = accuracy - balanced_accuracy
    f1_change = weighted_f1 - balanced_weighted_f1

    print("\nChange:")
    print(
        f"Accuracy:    {accuracy_change:+.4f}"
        f" ({accuracy_change * 100:+.2f} percentage points)"
    )
    print(
        f"Weighted F1: {f1_change:+.4f}"
        f" ({f1_change * 100:+.2f} percentage points)"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results = {
        "model": "XGBoost",
        "experiment": "unbalanced",
        "validation_samples": int(len(X_val)),
        "features": int(X_train.shape[1]),
        "classes": int(len(np.unique(y_train))),
        "n_estimators": 700,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_balancing": False,
        "training_time_minutes": round(training_time / 60, 2),
        "accuracy": float(accuracy),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }

    output_file = REPORT_DIR / "xgboost_unbalanced_validation.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to:")
    print(output_file)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()