"""
Final evaluation of the trained clinical decision support model.

IMPORTANT:
The test set is used ONLY here for final evaluation.
It must not be used for model selection or hyperparameter tuning.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.core.config import Config


def main():

    print("=" * 75)
    print("FINAL MODEL EVALUATION")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load final model
    # ---------------------------------------------------------
    print("\n[1/6] Loading final model...")

    model_path = Config.MODEL_DIR / "final_gradient_boosting.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Final model not found:\n{model_path}\n"
            "Run train_final_model.py first."
        )

    model = joblib.load(model_path)

    print(f"Model loaded from:")
    print(model_path)

    # ---------------------------------------------------------
    # 2. Load untouched test data
    # ---------------------------------------------------------
    print("\n[2/6] Loading test data...")

    X_test_path = Config.PROCESSED_DATA_DIR / "X_test.pkl"
    y_test_path = Config.PROCESSED_DATA_DIR / "y_test.pkl"

    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # ---------------------------------------------------------
    # 3. Verify test data
    # ---------------------------------------------------------
    print("\n[3/6] Verifying test data...")

    if X_test.shape[1] != 174:
        raise ValueError(
            f"Expected 174 test features, "
            f"but found {X_test.shape[1]}"
        )

    if X_test.shape[0] != 19974:
        raise ValueError(
            f"Expected 19974 test samples, "
            f"but found {X_test.shape[0]}"
        )

    # Verify feature order against model
    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != X_test.shape[1]:
            raise ValueError(
                f"Model expects {model.n_features_in_} features, "
                f"but test data contains {X_test.shape[1]}."
            )

    print("Test feature count verified: 174")
    print("Test sample count verified: 19,974")

    # ---------------------------------------------------------
    # 4. Generate predictions
    # ---------------------------------------------------------
    print("\n[4/6] Generating predictions...")

    y_pred = model.predict(X_test)

    print("Predictions generated successfully.")

    # ---------------------------------------------------------
    # 5. Calculate final metrics
    # ---------------------------------------------------------
    print("\n[5/6] Calculating final metrics...")

    accuracy = accuracy_score(y_test, y_pred)

    weighted_precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 75)
    print("FINAL TEST RESULTS")
    print("=" * 75)

    print(f"\nAccuracy          : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall   : {weighted_recall:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")
    print(f"Macro Precision   : {macro_precision:.4f}")
    print(f"Macro Recall      : {macro_recall:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")

    # ---------------------------------------------------------
    # 6. Detailed reports
    # ---------------------------------------------------------
    print("\n[6/6] Generating detailed reports...")

    Config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    Config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load label encoder
    encoder_path = Config.LABEL_ENCODER

    if encoder_path.exists():
        label_encoder = joblib.load(encoder_path)

        target_names = label_encoder.inverse_transform(
            np.arange(len(label_encoder.classes_))
        )

        # Only include labels actually represented in test set
        present_labels = sorted(
            set(y_test) | set(y_pred)
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=present_labels,
            target_names=[
                target_names[i] for i in present_labels
            ],
            output_dict=True,
            zero_division=0
        )

    else:
        print("Warning: label encoder not found.")
        print("Using encoded class labels instead.")

        present_labels = sorted(
            set(y_test) | set(y_pred)
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=present_labels,
            output_dict=True,
            zero_division=0
        )

    # ---------------------------------------------------------
    # Save classification report
    # ---------------------------------------------------------
    report_df = pd.DataFrame(report).transpose()

    report_path = (
        Config.REPORTS_DIR /
        "final_model_classification_report.csv"
    )

    report_df.to_csv(report_path)

    print(f"\nClassification report saved:")
    print(report_path)

    # ---------------------------------------------------------
    # Save final metrics
    # ---------------------------------------------------------
    metrics = {
        "model": "final_gradient_boosting",
        "test_samples": int(len(y_test)),
        "features": int(X_test.shape[1]),
        "accuracy": float(accuracy),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "test_set_used_only_for_final_evaluation": True
    }

    metrics_path = (
        Config.REPORTS_DIR /
        "final_model_metrics.json"
    )

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Final metrics saved:")
    print(metrics_path)

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=present_labels
    )

    cm_path = (
        Config.REPORTS_DIR /
        "final_model_confusion_matrix.csv"
    )

    pd.DataFrame(
        cm,
        index=present_labels,
        columns=present_labels
    ).to_csv(cm_path)

    print(f"Confusion matrix saved:")
    print(cm_path)

    # ---------------------------------------------------------
    # Confusion matrix visualization
    # ---------------------------------------------------------
    plt.figure(figsize=(16, 14))

    plt.imshow(cm, interpolation="nearest")

    plt.title("Final Gradient Boosting - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    plt.colorbar()

    plt.tight_layout()

    figure_path = (
        Config.FIGURES_DIR /
        "final_model_confusion_matrix.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Confusion matrix figure saved:")
    print(figure_path)

    # ---------------------------------------------------------
    # Print top and bottom diseases
    # ---------------------------------------------------------
    class_results = report_df[
        ~report_df.index.isin(
            ["accuracy", "macro avg", "weighted avg"]
        )
    ].copy()

    class_results = class_results.sort_values(
        "f1-score",
        ascending=False
    )

    print("\n" + "=" * 75)
    print("TOP 10 DISEASES BY F1-SCORE")
    print("=" * 75)

    print(
        class_results[
            ["precision", "recall", "f1-score", "support"]
        ].head(10).to_string()
    )

    print("\n" + "=" * 75)
    print("BOTTOM 10 DISEASES BY F1-SCORE")
    print("=" * 75)

    print(
        class_results[
            ["precision", "recall", "f1-score", "support"]
        ].tail(10).to_string()
    )

    print("\n" + "=" * 75)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 75)

    print("\nThe test set was used ONLY for final evaluation.")
    print("No hyperparameter tuning was performed on the test set.")


if __name__ == "__main__":
    main()