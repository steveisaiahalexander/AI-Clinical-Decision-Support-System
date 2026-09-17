import joblib
import pandas as pd

from src.core.config import Config
from src.core.logger import logger
from src.models.evaluate import ModelEvaluator


# =====================================================
# START
# =====================================================

logger.info("=" * 60)
logger.info("STARTING DETAILED MODEL EVALUATION")
logger.info("=" * 60)


# =====================================================
# LOAD TEST DATA
# =====================================================

X_test = joblib.load(
    Config.PROCESSED_DATA_DIR / "X_test.pkl"
)

y_test = joblib.load(
    Config.PROCESSED_DATA_DIR / "y_test.pkl"
)

logger.info(
    f"X_test loaded: {X_test.shape}"
)

logger.info(
    f"y_test loaded: {y_test.shape}"
)


# =====================================================
# LOAD SELECTED FEATURES
# =====================================================

selected_features = joblib.load(
    Config.MODEL_DIR / "selected_features.pkl"
)

logger.info(
    f"Selected features: {len(selected_features)}"
)


# =====================================================
# APPLY FEATURE SELECTION
# =====================================================

missing_features = [
    feature
    for feature in selected_features
    if feature not in X_test.columns
]

if missing_features:

    raise ValueError(
        "Selected features missing from X_test: "
        f"{missing_features}"
    )


X_test_selected = X_test[
    selected_features
].copy()


# =====================================================
# SAFETY CHECK
# =====================================================

if (
    X_test_selected.shape[0]
    != y_test.shape[0]
):

    raise ValueError(
        "Test samples and target samples do not match."
    )


if (
    list(X_test_selected.columns)
    != selected_features
):

    raise ValueError(
        "Test feature order does not match "
        "selected feature order."
    )


# =====================================================
# LOAD LABEL ENCODER
# =====================================================

label_encoder = joblib.load(
    Config.MODEL_DIR / "label_encoder.pkl"
)


# =====================================================
# INITIALIZE EVALUATOR
# =====================================================

evaluator = ModelEvaluator()


# =====================================================
# MODELS
# =====================================================

model_names = [

    "logistic_regression",

    "decision_tree",

    "random_forest",

    "extra_trees",

    "gradient_boosting",

]


# =====================================================
# EVALUATE ALL MODELS
# =====================================================

all_results = []


for model_name in model_names:

    logger.info("=" * 60)
    logger.info(
        f"LOADING MODEL: {model_name}"
    )
    logger.info("=" * 60)

    model_path = (
        Config.MODEL_DIR /
        f"{model_name}.pkl"
    )

    model = joblib.load(
        model_path
    )

    results = evaluator.evaluate_model(

        model=model,

        X_test=X_test_selected,

        y_test=y_test,

        model_name=model_name,

    )

    # -------------------------------------------------
    # Store summary
    # -------------------------------------------------

    all_results.append({

        "model": model_name,

        "accuracy":
            results["accuracy"],

        "weighted_precision":
            results["weighted_precision"],

        "weighted_recall":
            results["weighted_recall"],

        "weighted_f1":
            results["weighted_f1"],

        "macro_precision":
            results["macro_precision"],

        "macro_recall":
            results["macro_recall"],

        "macro_f1":
            results["macro_f1"],

    })

    # =================================================
    # SAVE PER-CLASS REPORT
    # =================================================

    class_report = pd.DataFrame(
        results["classification_report_dict"]
    ).transpose()

    # Keep only actual classes
    class_report = class_report[
        class_report.index.str.match(
            r"^\d+$"
        )
    ].copy()

    # Convert encoded class names to diseases
    class_report.index = [
        label_encoder.inverse_transform(
            [int(label)]
        )[0]
        for label in class_report.index
    ]

    class_report.index.name = "disease"

    class_report = class_report.reset_index()

    class_report_path = (
        Config.REPORTS_DIR /
        f"{model_name}_classification_report.csv"
    )

    class_report.to_csv(
        class_report_path,
        index=False,
    )

    logger.info(
        f"Saved class report: "
        f"{class_report_path}"
    )


# =====================================================
# MODEL COMPARISON
# =====================================================

comparison = pd.DataFrame(
    all_results
)


# Sort by macro F1
comparison = comparison.sort_values(
    by="macro_f1",
    ascending=False,
).reset_index(
    drop=True
)


# =====================================================
# DISPLAY COMPARISON
# =====================================================

print()

print("=" * 100)
print("DETAILED MODEL PERFORMANCE COMPARISON")
print("=" * 100)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =====================================================
# BEST MODEL — MACRO F1
# =====================================================

best_model = comparison.iloc[0]


print()

print("=" * 60)
print("BEST MODEL — MACRO F1")
print("=" * 60)

print(
    f"Model             : "
    f"{best_model['model']}"
)

print(
    f"Accuracy          : "
    f"{best_model['accuracy']:.4f}"
)

print(
    f"Weighted F1       : "
    f"{best_model['weighted_f1']:.4f}"
)

print(
    f"Macro Precision   : "
    f"{best_model['macro_precision']:.4f}"
)

print(
    f"Macro Recall      : "
    f"{best_model['macro_recall']:.4f}"
)

print(
    f"Macro F1          : "
    f"{best_model['macro_f1']:.4f}"
)


# =====================================================
# SAVE MODEL COMPARISON
# =====================================================

comparison_path = (
    Config.REPORTS_DIR /
    "detailed_model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False,
)


logger.info(
    f"Saved detailed comparison to: "
    f"{comparison_path}"
)


# =====================================================
# COMPLETION
# =====================================================

logger.info("=" * 60)
logger.info("DETAILED MODEL EVALUATION COMPLETED")
logger.info("=" * 60)