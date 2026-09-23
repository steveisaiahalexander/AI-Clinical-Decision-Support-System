from pathlib import Path
import time

import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():

    print("=" * 70)
    print("XGBoost GPU TEST")
    print("=" * 70)

    # Load processed training data
    X = joblib.load(PROCESSED_DIR / "X_train.pkl")
    y = joblib.load(PROCESSED_DIR / "y_train.pkl")

    # Use a small subset for a quick GPU test
    X_small, _, y_small, _ = train_test_split(
        X,
        y,
        train_size=10000,
        stratify=y,
        random_state=42,
    )

    print(f"\nTest samples: {len(X_small)}")
    print(f"Features: {X_small.shape[1]}")

    # GPU-enabled XGBoost
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,

        objective="multi:softprob",
        eval_metric="mlogloss",

        random_state=42,

        tree_method="hist",
        device="cuda",
    )

    print("\nGPU configuration:")
    print("  tree_method = hist")
    print("  device      = cuda")

    print("\nStarting GPU training...")

    start = time.time()

    model.fit(
        X_small,
        y_small,
    )

    elapsed = time.time() - start

    print(f"\nTraining completed in {elapsed:.2f} seconds.")

    # Test prediction
    predictions = model.predict(X_small)

    print(f"Predictions generated: {len(predictions)}")

    print("\n" + "=" * 70)
    print("GPU TEST COMPLETE")
    print("=" * 70)

    print(
        "\nIf GPU utilization increased during training, "
        "CUDA is working correctly."
    )


if __name__ == "__main__":
    main()