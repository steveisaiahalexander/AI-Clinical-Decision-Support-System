import joblib

from src.core.config import Config
from src.core.logger import logger
from src.models.train import ModelTrainer


# =====================================================
# START
# =====================================================

logger.info("=" * 60)
logger.info("STARTING MODEL TRAINING PIPELINE")
logger.info("=" * 60)


# =====================================================
# LOAD TRAINING DATA
# =====================================================

logger.info("Loading processed training data...")

X_train = joblib.load(
    Config.PROCESSED_DATA_DIR / "X_train.pkl"
)

y_train = joblib.load(
    Config.PROCESSED_DATA_DIR / "y_train.pkl"
)

logger.info(
    f"X_train loaded: {X_train.shape}"
)

logger.info(
    f"y_train loaded: {y_train.shape}"
)


# =====================================================
# LOAD SELECTED FEATURES
# =====================================================

logger.info("=" * 60)
logger.info("LOADING SELECTED FEATURES")
logger.info("=" * 60)

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
    if feature not in X_train.columns
]

if missing_features:

    raise ValueError(
        "Selected features missing from X_train: "
        f"{missing_features}"
    )


X_train_selected = X_train[
    selected_features
].copy()


# =====================================================
# VERIFY TRAINING DATA
# =====================================================

logger.info("=" * 60)
logger.info("FINAL TRAINING DATA")
logger.info("=" * 60)

logger.info(
    f"Original feature count : {X_train.shape[1]}"
)

logger.info(
    f"Selected feature count : "
    f"{X_train_selected.shape[1]}"
)

logger.info(
    f"Training samples       : "
    f"{X_train_selected.shape[0]}"
)

logger.info(
    f"Target samples         : "
    f"{y_train.shape[0]}"
)


# =====================================================
# SAFETY CHECKS
# =====================================================

if X_train_selected.shape[0] != y_train.shape[0]:

    raise ValueError(
        "Training feature count and target count do not match."
    )


if list(X_train_selected.columns) != selected_features:

    raise ValueError(
        "Training feature order does not match "
        "selected feature order."
    )


# =====================================================
# TRAIN ALL MODELS
# =====================================================

logger.info("=" * 60)
logger.info("TRAINING ALL MODELS")
logger.info("=" * 60)

trainer = ModelTrainer()

trained_models = trainer.train_all_models(
    X_train_selected,
    y_train,
)


# =====================================================
# TRAINING SUMMARY
# =====================================================

logger.info("=" * 60)
logger.info("MODEL TRAINING SUMMARY")
logger.info("=" * 60)

logger.info(
    f"Total models trained: {len(trained_models)}"
)

for model_name in trained_models:

    logger.info(
        f"✓ {model_name}"
    )


# =====================================================
# COMPLETION
# =====================================================

logger.info("=" * 60)
logger.info("MODEL TRAINING PIPELINE COMPLETED")
logger.info("=" * 60)