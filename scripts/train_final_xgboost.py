"""
Final XGBoost model training.

Uses the selected hyperparameters from validation experiments
and trains on the complete training dataset.

The test set is NOT loaded or used during training.
"""

import json
import time
import joblib

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.core.config import Config


def main():

    print("=" * 75)
    print("FINAL XGBOOST MODEL TRAINING")
    print("=" * 75)

    start_time = time.time()

    # ---------------------------------------------------------
    # 1. Load complete training data
    # ---------------------------------------------------------
    print("\n[1/5] Loading complete training data...")

    X_train = joblib.load(
        Config.PROCESSED_DATA_DIR / "X_train.pkl"
    )

    y_train = joblib.load(
        Config.PROCESSED_DATA_DIR / "y_train.pkl"
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # ---------------------------------------------------------
    # 2. Verify training data
    # ---------------------------------------------------------
    print("\n[2/5] Verifying training data...")

    if X_train.shape != (79892, 174):
        raise ValueError(
            f"Unexpected X_train shape: {X_train.shape}"
        )

    if len(y_train) != 79892:
        raise ValueError(
            f"Unexpected y_train length: {len(y_train)}"
        )

    print("Training samples : 79,892")
    print("Features         : 174")
    print(f"Classes          : {len(set(y_train))}")

    # ---------------------------------------------------------
    # 3. Calculate balanced sample weights
    # ---------------------------------------------------------
    print("\n[3/5] Calculating class-balanced sample weights...")

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    print(f"Minimum weight: {sample_weights.min():.4f}")
    print(f"Maximum weight: {sample_weights.max():.4f}")

    # ---------------------------------------------------------
    # 4. Train final XGBoost
    # ---------------------------------------------------------
    print("\n[4/5] Training final XGBoost model...")

    print("\nSelected hyperparameters:")
    print("  n_estimators     = 700")
    print("  learning_rate    = 0.1")
    print("  max_depth        = 6")
    print("  min_child_weight = 1")
    print("  subsample        = 0.8")
    print("  colsample_bytree = 0.8")
    print("  tree_method      = hist")
    print("  random_state     = 42")
    print("  class balancing  = balanced sample weights")

    model = XGBClassifier(
        n_estimators=700,
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

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    training_time = time.time() - start_time

    print("\nTraining completed successfully.")

    # ---------------------------------------------------------
    # 5. Save model and configuration
    # ---------------------------------------------------------
    print("\n[5/5] Saving final model...")

    Config.MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    Config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        Config.MODEL_DIR /
        "final_xgboost.pkl"
    )

    joblib.dump(model, model_path)

    configuration = {
        "model": "XGBClassifier",
        "training_samples": 79892,
        "features": 174,
        "classes": 30,
        "class_balancing": "balanced_sample_weight",
        "hyperparameters": {
            "n_estimators": 700,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist"
        },
        "validation_accuracy": 0.7009,
        "validation_weighted_f1": 0.7020,
        "validation_macro_f1": 0.5404,
        "test_set_used_during_training": False
    }

    config_path = (
        Config.REPORTS_DIR /
        "final_xgboost_configuration.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            configuration,
            file,
            indent=4
        )

    print("\n" + "=" * 75)
    print("FINAL XGBOOST TRAINING COMPLETE")
    print("=" * 75)

    print(f"\nModel saved to:")
    print(model_path)

    print(f"\nConfiguration saved to:")
    print(config_path)

    print(
        f"\nTraining time: "
        f"{training_time / 60:.2f} minutes"
    )

    print("\nFinal model:")
    print("  Samples  : 79,892")
    print("  Features : 174")
    print("  Classes  : 30")

    print("\nValidation benchmark:")
    print("  Accuracy    : 70.09%")
    print("  Weighted F1 : 70.20%")
    print("  Macro F1    : 54.04%")

    print("\nTest set was NOT used during training.")


if __name__ == "__main__":
    main()