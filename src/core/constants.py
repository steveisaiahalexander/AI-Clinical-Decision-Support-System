"""
Application Constants
=====================

Immutable constants used throughout the project.
"""

# ======================================================
# RANDOMNESS
# ======================================================

RANDOM_STATE = 42

# ======================================================
# DATASET
# ======================================================

TARGET_COLUMN = "prognosis"

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.10

# ======================================================
# TRAINING
# ======================================================

CV_FOLDS = 5

N_JOBS = -1

VERBOSE = 1

# ======================================================
# MODELS
# ======================================================

SUPPORTED_MODELS = [
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "Naive Bayes",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]

DEFAULT_MODEL = "Random Forest"

# ======================================================
# VISUALIZATION
# ======================================================

FIGURE_DPI = 300

FIGURE_SIZE = (10, 6)

# ======================================================
# REPORTING
# ======================================================

TOP_FEATURES = 20

TOP_PREDICTIONS = 5

# ======================================================
# FEATURE ENGINEERING
# ======================================================

CORRELATION_THRESHOLD = 0.95

TOP_FEATURES = 20