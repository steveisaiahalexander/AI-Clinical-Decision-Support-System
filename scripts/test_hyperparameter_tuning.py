import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from src.core.config import Config
from src.core.logger import logger


# =====================================================
# LOAD TRAINING DATA
# =====================================================

logger.info("=" * 60)
logger.info("GRADIENT BOOSTING HYPERPARAMETER EXPERIMENT")
logger.info("=" * 60)

X_train = joblib.load(
    Config.PROCESSED_DATA_DIR / "X_train.pkl"
)

y_train = joblib.load(
    Config.PROCESSED_DATA_DIR / "y_train.pkl"
)

logger.info(
    f"Training data: {X_train.shape}"
)

logger.info(
    f"Training classes: {len(set(y_train))}"
)


# =====================================================
# CREATE VALIDATION SPLIT
# =====================================================

logger.info("=" * 60)
logger.info("CREATING VALIDATION SPLIT")
logger.info("=" * 60)

X_subtrain, X_validation, y_subtrain, y_validation = (
    train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train,
    )
)

logger.info(
    f"Sub-training samples : {X_subtrain.shape[0]}"
)

logger.info(
    f"Validation samples   : {X_validation.shape[0]}"
)


# =====================================================
# CREATE BALANCED SAMPLE WEIGHTS
# =====================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_subtrain,
)


# =====================================================
# HYPERPARAMETER CONFIGURATIONS
# =====================================================
'''
experiments = [

    {
        "name": "baseline",
        "n_estimators": 100,
        "learning_rate": 0.10,
        "max_depth": 3,
        "subsample": 1.0,
    },

    {
        "name": "more_trees",
        "n_estimators": 150,
        "learning_rate": 0.10,
        "max_depth": 3,
        "subsample": 1.0,
    },

    {
        "name": "slow_learning",
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 1.0,
    },

    {
        "name": "deeper_trees",
        "n_estimators": 100,
        "learning_rate": 0.10,
        "max_depth": 4,
        "subsample": 1.0,
    },

    {
        "name": "shallower_trees",
        "n_estimators": 100,
        "learning_rate": 0.10,
        "max_depth": 2,
        "subsample": 1.0,
    },

    {
        "name": "subsampled",
        "n_estimators": 150,
        "learning_rate": 0.10,
        "max_depth": 3,
        "subsample": 0.80,
    },
]

experiments = [

    {
        "name": "depth4_baseline",
        "n_estimators": 100,
        "learning_rate": 0.10,
        "max_depth": 4,
        "subsample": 1.0,
    },

    {
        "name": "depth4_subsampled",
        "n_estimators": 150,
        "learning_rate": 0.10,
        "max_depth": 4,
        "subsample": 0.8,
    },

    {
        "name": "depth5_baseline",
        "n_estimators": 100,
        "learning_rate": 0.10,
        "max_depth": 5,
        "subsample": 1.0,
    },
]
'''
experiments = [

    {
        "name": "depth4_baseline",
        "n_estimators": 100,
        "learning_rate": 0.10,
        "max_depth": 4,
        "subsample": 1.0,
    },

    {
        "name": "depth4_subsampled",
        "n_estimators": 150,
        "learning_rate": 0.10,
        "max_depth": 4,
        "subsample": 0.8,
    },

    {
        "name": "depth5_baseline",
        "n_estimators": 100,
        "learning_rate": 0.10,
        "max_depth": 5,
        "subsample": 1.0,
    },
]

# =====================================================
# RUN EXPERIMENT
# =====================================================

results = []

for config in experiments:

    logger.info("=" * 60)
    logger.info(
        f"EXPERIMENT: {config['name']}"
    )
    logger.info("=" * 60)

    logger.info(
        f"n_estimators  : {config['n_estimators']}"
    )

    logger.info(
        f"learning_rate : {config['learning_rate']}"
    )

    logger.info(
        f"max_depth     : {config['max_depth']}"
    )

    logger.info(
        f"subsample     : {config['subsample']}"
    )

    model = GradientBoostingClassifier(
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        subsample=config["subsample"],
        random_state=42,
    )

    # -------------------------------------------------
    # TRAIN
    # -------------------------------------------------

    model.fit(
        X_subtrain,
        y_subtrain,
        sample_weight=sample_weights,
    )

    # -------------------------------------------------
    # VALIDATION PREDICTIONS
    # -------------------------------------------------

    y_pred = model.predict(
        X_validation
    )

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------

    accuracy = accuracy_score(
        y_validation,
        y_pred,
    )

    weighted_f1 = f1_score(
        y_validation,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_validation,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_validation,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_validation,
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

    results.append({
        "experiment": config["name"],
        "n_estimators": config["n_estimators"],
        "learning_rate": config["learning_rate"],
        "max_depth": config["max_depth"],
        "subsample": config["subsample"],
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    })


# =====================================================
# RESULTS
# =====================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="macro_f1",
    ascending=False,
).reset_index(
    drop=True
)


print()

print("=" * 100)
print("HYPERPARAMETER EXPERIMENT RESULTS")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# =====================================================
# BEST CONFIGURATION
# =====================================================

best = results_df.iloc[0]

print()

print("=" * 60)
print("BEST VALIDATION CONFIGURATION")
print("=" * 60)

print(
    f"Experiment    : {best['experiment']}"
)

print(
    f"Estimators    : {int(best['n_estimators'])}"
)

print(
    f"Learning Rate : {best['learning_rate']}"
)

print(
    f"Max Depth     : {int(best['max_depth'])}"
)

print(
    f"Subsample     : {best['subsample']}"
)

print(
    f"Accuracy      : {best['accuracy']:.4f}"
)

print(
    f"Weighted F1   : {best['weighted_f1']:.4f}"
)

print(
    f"Macro Recall  : {best['macro_recall']:.4f}"
)

print(
    f"Macro F1      : {best['macro_f1']:.4f}"
)


# =====================================================
# SAVE RESULTS
# =====================================================

output_path = (
    Config.REPORTS_DIR /
    "gradient_boosting_hyperparameter_results.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

logger.info(
    f"Saved tuning results to: {output_path}"
)

logger.info("=" * 60)
logger.info(
    "HYPERPARAMETER EXPERIMENT COMPLETED"
)
logger.info("=" * 60)