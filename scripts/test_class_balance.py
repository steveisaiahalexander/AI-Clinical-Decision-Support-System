
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.core.config import Config
from src.core.logger import logger
from src.models.model_factory import ModelFactory


# =====================================================
# LOAD PROCESSED DATA
# =====================================================

logger.info("=" * 60)
logger.info("CLASS IMBALANCE EXPERIMENT")
logger.info("=" * 60)

X_train = joblib.load(
    Config.PROCESSED_DATA_DIR / "X_train.pkl"
)

X_test = joblib.load(
    Config.PROCESSED_DATA_DIR / "X_test.pkl"
)

y_train = joblib.load(
    Config.PROCESSED_DATA_DIR / "y_train.pkl"
)

y_test = joblib.load(
    Config.PROCESSED_DATA_DIR / "y_test.pkl"
)

logger.info(
    f"X_train: {X_train.shape}"
)

logger.info(
    f"X_test : {X_test.shape}"
)

logger.info(
    f"Classes: {len(set(y_train))}"
)


# =====================================================
# CREATE BALANCED SAMPLE WEIGHTS
# =====================================================

logger.info("=" * 60)
logger.info("CREATING CLASS-BALANCED SAMPLE WEIGHTS")
logger.info("=" * 60)

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

logger.info(
    "Balanced sample weights created."
)


# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_model(
    model_name,
    model,
    use_sample_weight=False,
):
    """
    Train and evaluate one balanced model.
    """

    logger.info("=" * 60)
    logger.info(
        f"TRAINING BALANCED MODEL: {model_name}"
    )
    logger.info("=" * 60)

    if use_sample_weight:

        logger.info(
            "Using class-balanced sample weights."
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
        )

    else:

        model.fit(
            X_train,
            y_train,
        )

    logger.info(
        f"{model_name} trained successfully."
    )

    # -------------------------------------------------
    # Save balanced model for detailed analysis
    # -------------------------------------------------

    model_path = (
        Config.MODEL_DIR /
        f"{model_name}.pkl"
    )

    joblib.dump(
        model,
        model_path,
    )

    logger.info(
        f"Saved balanced model to: {model_path}"
    )

    y_pred = model.predict(
        X_test
    )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    weighted_precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    logger.info(
        f"Accuracy        : {accuracy:.4f}"
    )

    logger.info(
        f"Weighted F1     : {weighted_f1:.4f}"
    )

    logger.info(
        f"Macro Precision : {macro_precision:.4f}"
    )

    logger.info(
        f"Macro Recall    : {macro_recall:.4f}"
    )

    logger.info(
        f"Macro F1        : {macro_f1:.4f}"
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


# =====================================================
# CREATE MODELS
# =====================================================

models = ModelFactory.get_all_models()


# =====================================================
# BALANCE MODELS THAT SUPPORT class_weight
# =====================================================

models["logistic_regression"].set_params(
    class_weight="balanced"
)

models["decision_tree"].set_params(
    class_weight="balanced"
)

models["random_forest"].set_params(
    class_weight="balanced"
)

models["extra_trees"].set_params(
    class_weight="balanced"
)


# =====================================================
# RUN BALANCED MODELS
# =====================================================

results = []

'''
# -----------------------------------------------------
# Logistic Regression
# -----------------------------------------------------

results.append(
    evaluate_model(
        "balanced_logistic_regression",
        models["logistic_regression"],
    )
)


# -----------------------------------------------------
# Decision Tree
# -----------------------------------------------------

results.append(
    evaluate_model(
        "balanced_decision_tree",
        models["decision_tree"],
    )
)


# -----------------------------------------------------
# Random Forest
# -----------------------------------------------------

results.append(
    evaluate_model(
        "balanced_random_forest",
        models["random_forest"],
    )
)


# -----------------------------------------------------
# Extra Trees
# -----------------------------------------------------

results.append(
    evaluate_model(
        "balanced_extra_trees",
        models["extra_trees"],
    )
)
'''


# =====================================================
# GRADIENT BOOSTING
# =====================================================

logger.info("=" * 60)
logger.info("TRAINING BALANCED GRADIENT BOOSTING")
logger.info("=" * 60)

gradient_boosting = models["gradient_boosting"]

results.append(
    evaluate_model(
        "balanced_gradient_boosting",
        gradient_boosting,
        use_sample_weight=True,
    )
)


# =====================================================
# COMPARISON
# =====================================================

comparison = pd.DataFrame(
    results
)

comparison = comparison.sort_values(
    by="macro_f1",
    ascending=False,
).reset_index(
    drop=True
)


print()

print("=" * 100)
print("BALANCED MODEL PERFORMANCE")
print("=" * 100)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# =====================================================
# BEST BALANCED MODEL
# =====================================================

best = comparison.iloc[0]

print()

print("=" * 60)
print("BEST BALANCED MODEL")
print("=" * 60)

print(
    f"Model       : {best['model']}"
)

print(
    f"Accuracy    : {best['accuracy']:.4f}"
)

print(
    f"Weighted F1 : {best['weighted_f1']:.4f}"
)

print(
    f"Macro F1    : {best['macro_f1']:.4f}"
)

print(
    f"Macro Recall: {best['macro_recall']:.4f}"
)


# =====================================================
# SAVE RESULTS
# =====================================================

output_path = (
    Config.REPORTS_DIR /
    "class_balance_comparison.csv"
)

comparison.to_csv(
    output_path,
    index=False,
)

logger.info(
    f"Saved comparison report to: {output_path}"
)


# =====================================================
# COMPLETION
# =====================================================

logger.info("=" * 60)
logger.info("CLASS IMBALANCE EXPERIMENT COMPLETED")
logger.info("=" * 60)