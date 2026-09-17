"""
Final evaluation of the XGBoost clinical decision-support model.

IMPORTANT:
The test set is used ONLY for this final evaluation.
It has not been used for training or hyperparameter selection.
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
    print("FINAL XGBOOST MODEL EVALUATION")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load final XGBoost model
    # ---------------------------------------------------------
    print("\n[1/7] Loading final XGBoost model...")

    model_path = (
        Config.MODEL_DIR /
        "final_xgboost.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Final XGBoost model not found:\n{model_path}"
        )

    model = joblib.load(model_path)

    print("Model loaded from:")
    print(model_path)

    # ---------------------------------------------------------
    # 2. Load untouched test data
    # ---------------------------------------------------------
    print("\n[2/7] Loading untouched test data...")

    X_test = joblib.load(
        Config.PROCESSED_DATA_DIR /
        "X_test.pkl"
    )

    y_test = joblib.load(
        Config.PROCESSED_DATA_DIR /
        "y_test.pkl"
    )

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # ---------------------------------------------------------
    # 3. Verify test data
    # ---------------------------------------------------------
    print("\n[3/7] Verifying test data...")

    if X_test.shape != (19974, 174):
        raise ValueError(
            f"Unexpected X_test shape: {X_test.shape}"
        )

    if len(y_test) != 19974:
        raise ValueError(
            f"Unexpected y_test length: {len(y_test)}"
        )

    if model.n_features_in_ != X_test.shape[1]:
        raise ValueError(
            f"Model expects {model.n_features_in_} features, "
            f"but test data contains {X_test.shape[1]}"
        )

    print("Test samples : 19,974")
    print("Test features: 174")
    print("Feature count matches model.")

    # ---------------------------------------------------------
    # 4. Generate predictions
    # ---------------------------------------------------------
    print("\n[4/7] Generating XGBoost predictions...")

    y_pred = model.predict(X_test)

    print("Predictions generated successfully.")

    # ---------------------------------------------------------
    # 5. Calculate metrics
    # ---------------------------------------------------------
    print("\n[5/7] Calculating final metrics...")

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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
    print("FINAL XGBOOST TEST RESULTS")
    print("=" * 75)

    print(
        f"\nAccuracy          : "
        f"{accuracy:.4f} ({accuracy * 100:.2f}%)"
    )

    print(
        f"Weighted Precision: "
        f"{weighted_precision:.4f}"
    )

    print(
        f"Weighted Recall   : "
        f"{weighted_recall:.4f}"
    )

    print(
        f"Weighted F1       : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Macro Precision   : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall      : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{macro_f1:.4f}"
    )

    # ---------------------------------------------------------
    # 6. Detailed classification report
    # ---------------------------------------------------------
    print("\n[6/7] Generating detailed classification report...")

    Config.REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    Config.FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    encoder_path = Config.LABEL_ENCODER

    if encoder_path.exists():

        label_encoder = joblib.load(
            encoder_path
        )

        present_labels = sorted(
            set(y_test) | set(y_pred)
        )

        target_names = [
            label_encoder.inverse_transform(
                [label]
            )[0]
            for label in present_labels
        ]

        report = classification_report(
            y_test,
            y_pred,
            labels=present_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )

    else:

        print(
            "Warning: label encoder not found."
        )

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

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_path = (
        Config.REPORTS_DIR /
        "final_xgboost_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    print("\nClassification report saved:")
    print(report_path)

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------
    metrics = {
        "model": "final_xgboost",
        "test_samples": 19974,
        "features": 174,
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
        "final_xgboost_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print("Final metrics saved:")
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
        "final_xgboost_confusion_matrix.csv"
    )

    pd.DataFrame(
        cm,
        index=target_names,
        columns=target_names
    ).to_csv(cm_path)

    print("Confusion matrix saved:")
    print(cm_path)

    # ---------------------------------------------------------
    # Confusion matrix figure
    # ---------------------------------------------------------
    plt.figure(
        figsize=(16, 14)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "Final XGBoost - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.colorbar()

    plt.tight_layout()

    figure_path = (
        Config.FIGURES_DIR /
        "final_xgboost_confusion_matrix.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion matrix figure saved:")
    print(figure_path)

    # ---------------------------------------------------------
    # Top and bottom diseases
    # ---------------------------------------------------------
    class_results = report_df[
        ~report_df.index.isin(
            [
                "accuracy",
                "macro avg",
                "weighted avg"
            ]
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
            [
                "precision",
                "recall",
                "f1-score",
                "support"
            ]
        ].head(10).to_string()
    )

    print("\n" + "=" * 75)
    print("BOTTOM 10 DISEASES BY F1-SCORE")
    print("=" * 75)

    print(
        class_results[
            [
                "precision",
                "recall",
                "f1-score",
                "support"
            ]
        ].tail(10).to_string()
    )

    # ---------------------------------------------------------
    # Final comparison
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("MODEL COMPARISON")
    print("=" * 75)

    print("\nPrevious final Gradient Boosting:")
    print("  Test Accuracy: 56.58%")

    print("\nNew XGBoost:")
    print(
        f"  Test Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nTest set was used ONLY for this final evaluation.")
    print("Final evaluation complete.")


if __name__ == "__main__":
    main()