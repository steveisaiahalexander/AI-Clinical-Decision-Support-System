import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from src.core.config import Config
from src.core.logger import logger


# =====================================================
# LOAD TEST DATA
# =====================================================

logger.info("=" * 60)
logger.info("DETAILED MODEL EVALUATION")
logger.info("=" * 60)

X_test = joblib.load(
    Config.PROCESSED_DATA_DIR / "X_test.pkl"
)

y_test = joblib.load(
    Config.PROCESSED_DATA_DIR / "y_test.pkl"
)

logger.info(
    f"X_test: {X_test.shape}"
)

logger.info(
    f"y_test: {y_test.shape}"
)


# =====================================================
# LOAD BALANCED MODEL
# =====================================================

model_path = (
    Config.MODEL_DIR /
    "balanced_gradient_boosting.pkl"
)

logger.info(
    f"Loading model: {model_path}"
)

model = joblib.load(
    model_path
)


# =====================================================
# PREDICTIONS
# =====================================================

logger.info(
    "Generating predictions..."
)

y_pred = model.predict(
    X_test
)


# =====================================================
# LOAD LABEL ENCODER
# =====================================================

encoder = joblib.load(
    Config.MODEL_DIR /
    "label_encoder.pkl"
)

class_names = encoder.classes_


# =====================================================
# CLASSIFICATION REPORT
# =====================================================

report = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)


report_df = pd.DataFrame(
    report
).transpose()


# =====================================================
# REMOVE SUMMARY ROWS
# =====================================================

class_report = report_df[
    ~report_df.index.isin([
        "accuracy",
        "macro avg",
        "weighted avg",
    ])
].copy()


# =====================================================
# DISPLAY RESULTS
# =====================================================

print()

print("=" * 90)
print("PER-DISEASE PERFORMANCE")
print("=" * 90)

print(
    class_report[
        [
            "precision",
            "recall",
            "f1-score",
            "support",
        ]
    ].to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# =====================================================
# BEST / WORST CLASSES
# =====================================================

print()

print("=" * 90)
print("TOP 10 DISEASES BY F1 SCORE")
print("=" * 90)

print(
    class_report
    .sort_values(
        "f1-score",
        ascending=False,
    )
    .head(10)[
        [
            "precision",
            "recall",
            "f1-score",
            "support",
        ]
    ]
    .to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


print()

print("=" * 90)
print("BOTTOM 10 DISEASES BY F1 SCORE")
print("=" * 90)

print(
    class_report
    .sort_values(
        "f1-score",
        ascending=True,
    )
    .head(10)[
        [
            "precision",
            "recall",
            "f1-score",
            "support",
        ]
    ]
    .to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# =====================================================
# SAVE REPORT
# =====================================================

report_path = (
    Config.REPORTS_DIR /
    "balanced_gradient_boosting_classification_report.csv"
)

class_report.to_csv(
    report_path
)

logger.info(
    f"Saved detailed report to: {report_path}"
)


# =====================================================
# CONFUSION MATRIX
# =====================================================

matrix = confusion_matrix(
    y_test,
    y_pred,
)

matrix_df = pd.DataFrame(
    matrix,
    index=class_names,
    columns=class_names,
)

matrix_path = (
    Config.REPORTS_DIR /
    "balanced_gradient_boosting_confusion_matrix.csv"
)

matrix_df.to_csv(
    matrix_path
)

logger.info(
    f"Saved confusion matrix to: {matrix_path}"
)


# =====================================================
# COMPLETION
# =====================================================

logger.info("=" * 60)
logger.info("DETAILED EVALUATION COMPLETED")
logger.info("=" * 60)