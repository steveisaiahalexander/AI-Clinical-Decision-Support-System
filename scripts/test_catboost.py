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
from catboost import CatBoostClassifier


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 80)
    print("CatBoost - VALIDATION EXPERIMENT")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load processed training data
    # ---------------------------------------------------------

    print("\n[1/6] Loading processed training data...")

    X_train = joblib.load(PROCESSED_DIR / "X_train.pkl")
    y_train = joblib.load(PROCESSED_DIR / "y_train.pkl")

    print(f"X shape: {X_train.shape}")
    print(f"y shape: {y_train.shape}")

    # ---------------------------------------------------------
    # 2. Same validation split used by XGBoost experiments
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
    # 3. Moderate class weighting
    # ---------------------------------------------------------

    print("\n[3/6] Calculating moderate class weights...")

    balanced_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_dev,
    )

    # Same weighting strategy that produced our best
    # XGBoost validation accuracy.
    sample_weights = balanced_weights ** 0.25

    print(
        f"Weight range: "
        f"{sample_weights.min():.4f} → "
        f"{sample_weights.max():.4f}"
    )

    # ---------------------------------------------------------
    # 4. Create CatBoost model
    # ---------------------------------------------------------

    print("\n[4/6] Creating CatBoost model...")

    model = CatBoostClassifier(
        iterations=900,
        learning_rate=0.075,
        depth=6,

        loss_function="MultiClass",
        eval_metric="Accuracy",

        # Randomized row sampling
        bootstrap_type="Bernoulli",
        subsample=0.9,

        # Random feature sampling
        rsm=0.9,

        random_seed=RANDOM_STATE,

        thread_count=-1,

        # Keep terminal output manageable
        verbose=100,

        allow_writing_files=False,
    )

    print("\nConfiguration:")
    print("  iterations        = 900")
    print("  learning_rate     = 0.075")
    print("  depth             = 6")
    print("  bootstrap         = Bernoulli")
    print("  subsample         = 0.9")
    print("  rsm               = 0.9")
    print("  class weighting   = balanced^0.25")

    # ---------------------------------------------------------
    # 5. Train
    # ---------------------------------------------------------

    print("\n[5/6] Training CatBoost...")
    print("This may take several minutes.")

    start_time = time.time()

    model.fit(
        X_dev,
        y_dev,
        sample_weight=sample_weights,
        eval_set=(X_val, y_val),
        use_best_model=False,
    )

    training_time = time.time() - start_time

    print(
        f"\nTraining completed in "
        f"{training_time / 60:.2f} minutes."
    )

    # ---------------------------------------------------------
    # 6. Evaluate
    # ---------------------------------------------------------

    print("\n[6/6] Evaluating on validation set...")

    y_pred = model.predict(X_val)

    # CatBoost can return shape (n_samples, 1)
    # depending on the version.
    y_pred = np.asarray(y_pred).reshape(-1)

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
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("CATBOOST VALIDATION RESULTS")
    print("=" * 80)

    print(
        f"\nAccuracy:           "
        f"{accuracy:.4f} ({accuracy * 100:.2f}%)"
    )

    print(
        f"Weighted Precision: "
        f"{weighted_precision:.4f}"
    )

    print(
        f"Weighted Recall:    "
        f"{weighted_recall:.4f}"
    )

    print(
        f"Weighted F1:        "
        f"{weighted_f1:.4f}"
    )

    print(
        f"\nMacro Precision:    "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall:       "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1:           "
        f"{macro_f1:.4f}"
    )

    # ---------------------------------------------------------
    # Compare with current best XGBoost
    # ---------------------------------------------------------

    xgb_accuracy = 0.7175
    xgb_weighted_f1 = 0.7139
    xgb_macro_f1 = 0.5417

    print("\n" + "=" * 80)
    print("CATBOOST vs CURRENT BEST XGBOOST")
    print("=" * 80)

    print(
        "\nMetric                  "
        "XGBoost       CatBoost"
    )
    print("-" * 65)

    print(
        f"Accuracy                "
        f"{xgb_accuracy * 100:>8.2f}%"
        f"       {accuracy * 100:>8.2f}%"
    )

    print(
        f"Weighted F1             "
        f"{xgb_weighted_f1:>8.4f}"
        f"       {weighted_f1:>8.4f}"
    )

    print(
        f"Macro F1                "
        f"{xgb_macro_f1:>8.4f}"
        f"       {macro_f1:>8.4f}"
    )

    print("\nDifference:")

    print(
        f"Accuracy: "
        f"{(accuracy - xgb_accuracy) * 100:+.2f} "
        f"percentage points"
    )

    print(
        f"Weighted F1: "
        f"{weighted_f1 - xgb_weighted_f1:+.4f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1 - xgb_macro_f1:+.4f}"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results = {
        "model": "CatBoost",
        "experiment": "catboost_validation",

        "validation_samples": int(len(X_val)),
        "features": int(X_train.shape[1]),
        "classes": int(len(np.unique(y_train))),

        "iterations": 900,
        "learning_rate": 0.075,
        "depth": 6,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.9,
        "rsm": 0.9,

        "class_weighting": "balanced^0.25",

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
        "catboost_validation.json"
    )

    with open(output_file, "w") as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    print("\nResults saved to:")
    print(output_file)

    print("\n" + "=" * 80)
    print("CATBOOST EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()