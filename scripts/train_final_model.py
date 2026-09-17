"""
Final model training script.

Trains the selected Gradient Boosting model on the complete
training dataset using the best hyperparameters found during
validation-based tuning.

The test dataset is NOT used during training.
"""

import json
import time
import joblib


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.core.config import Config


def main():
    print("=" * 70)
    print("FINAL MODEL TRAINING")
    print("=" * 70)

    start_time = time.time()

    # ---------------------------------------------------------
    # 1. Load processed training data
    # ---------------------------------------------------------
    print("\n[1/5] Loading training data...")

    X_train_path = Config.PROCESSED_DATA_DIR / "X_train.pkl"
    y_train_path = Config.PROCESSED_DATA_DIR / "y_train.pkl"

    X_train = joblib.load(X_train_path)
    y_train = joblib.load(y_train_path)

    print(f"X_train loaded from: {X_train_path}")
    print(f"y_train loaded from: {y_train_path}")

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # ---------------------------------------------------------
    # 2. Verify training data
    # ---------------------------------------------------------
    print("\n[2/5] Verifying training data...")

    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of classes: {len(set(y_train))}")

    if X_train.shape[1] != 174:
        raise ValueError(
            f"Expected 174 features, but found {X_train.shape[1]}"
        )

    if X_train.shape[0] != 79892:
        raise ValueError(
            f"Expected 79892 training samples, "
            f"but found {X_train.shape[0]}"
        )

    # ---------------------------------------------------------
    # 3. Create class-balanced sample weights
    # ---------------------------------------------------------
    print("\n[3/5] Calculating class-balanced sample weights...")

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    print(f"Sample weights calculated: {len(sample_weights)}")
    print(f"Minimum weight: {sample_weights.min():.4f}")
    print(f"Maximum weight: {sample_weights.max():.4f}")

    # ---------------------------------------------------------
    # 4. Create and train final model
    # ---------------------------------------------------------
    print("\n[4/5] Training final Gradient Boosting model...")

    print("\nSelected configuration:")
    print("  n_estimators = 150")
    print("  learning_rate = 0.1")
    print("  max_depth = 4")
    print("  subsample = 0.8")
    print("  random_state = 42")
    print("  class balancing = balanced sample weights")

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    print("\nTraining completed successfully.")

    # ---------------------------------------------------------
    # 5. Save final model and configuration
    # ---------------------------------------------------------
    print("\n[5/5] Saving final model...")

    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Config.MODEL_DIR / "final_gradient_boosting.pkl"

    joblib.dump(model, model_path)

    # Save configuration for reproducibility
    config = {
        "model": "GradientBoostingClassifier",
        "training_samples": int(X_train.shape[0]),
        "features": int(X_train.shape[1]),
        "classes": int(len(set(y_train))),
        "class_balancing": "balanced_sample_weight",
        "hyperparameters": {
            "n_estimators": 150,
            "learning_rate": 0.1,
            "max_depth": 4,
            "subsample": 0.8,
            "random_state": 42
        },
        "test_set_used_during_training": False
    }

    config_path = (
        Config.REPORTS_DIR /
        "final_model_configuration.json"
    )

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("FINAL MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(f"\nModel saved to:")
    print(model_path)

    print(f"\nConfiguration saved to:")
    print(config_path)

    print(f"\nTraining time: {elapsed_time / 60:.2f} minutes")

    print("\nModel summary:")
    print(f"  Samples:  {X_train.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes:  {len(set(y_train))}")

    print("\nTest set was NOT used during training.")


if __name__ == "__main__":
    main()