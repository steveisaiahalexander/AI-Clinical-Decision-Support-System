import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.core.config import Config
from src.core.logger import logger


# =====================================================
# LOAD DATA
# =====================================================

logger.info("=" * 60)
logger.info("FEATURE COUNT EXPERIMENT")
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


# =====================================================
# LOAD SELECTED FEATURES
# =====================================================

selected_features = joblib.load(
    Config.MODEL_DIR / "selected_features.pkl"
)

X_train_50 = X_train[
    selected_features
].copy()

X_test_50 = X_test[
    selected_features
].copy()


# =====================================================
# EXPERIMENT FUNCTION
# =====================================================

def evaluate_feature_set(
    name,
    X_train_exp,
    X_test_exp,
):
    """
    Train and evaluate Gradient Boosting
    using a specific feature set.
    """

    logger.info("=" * 60)
    logger.info(
        f"TRAINING: {name}"
    )
    logger.info("=" * 60)

    logger.info(
        f"Features: {X_train_exp.shape[1]}"
    )

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )

    model.fit(
        X_train_exp,
        y_train,
    )

    y_pred = model.predict(
        X_test_exp
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
        "feature_set": name,
        "features": X_train_exp.shape[1],
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


# =====================================================
# RUN 50-FEATURE EXPERIMENT
# =====================================================

result_50 = evaluate_feature_set(
    "50_features",
    X_train_50,
    X_test_50,
)


# =====================================================
# RUN 174-FEATURE EXPERIMENT
# =====================================================

result_174 = evaluate_feature_set(
    "174_features",
    X_train,
    X_test,
)


# =====================================================
# COMPARISON
# =====================================================

comparison = pd.DataFrame([
    result_50,
    result_174,
])


comparison = comparison.sort_values(
    by="macro_f1",
    ascending=False,
).reset_index(
    drop=True
)


print()

print("=" * 100)
print("50 vs 174 FEATURE COMPARISON")
print("=" * 100)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =====================================================
# BEST FEATURE SET
# =====================================================

best = comparison.iloc[0]

print()

print("=" * 60)
print("BEST FEATURE SET")
print("=" * 60)

print(
    f"Feature Set : {best['feature_set']}"
)

print(
    f"Features    : {best['features']}"
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


# =====================================================
# SAVE EXPERIMENT
# =====================================================

output_path = (
    Config.REPORTS_DIR /
    "feature_count_comparison.csv"
)

comparison.to_csv(
    output_path,
    index=False,
)

logger.info(
    f"Saved experiment report to: "
    f"{output_path}"
)


# =====================================================
# COMPLETION
# =====================================================

logger.info("=" * 60)
logger.info("FEATURE COUNT EXPERIMENT COMPLETED")
logger.info("=" * 60)