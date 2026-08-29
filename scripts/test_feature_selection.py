import pandas as pd
import joblib

from src.core.config import Config
from src.data.feature_engineering import FeatureEngineer


# =====================================================
# LOAD SAVED FEATURE RANKING
# =====================================================

ranking_path = (
    Config.REPORTS_DIR /
    "feature_ranking.csv"
)

print("=" * 50)
print("LOADING SAVED FEATURE RANKING")
print("=" * 50)

print(f"Path: {ranking_path}")

ranking = pd.read_csv(
    ranking_path
)

print(
    f"Loaded {len(ranking)} ranked features."
)


# =====================================================
# LOAD PROCESSED DATA
# =====================================================

print()
print("=" * 50)
print("LOADING PROCESSED DATA")
print("=" * 50)

X_train = joblib.load(
    Config.PROCESSED_DATA_DIR /
    "X_train.pkl"
)

X_test = joblib.load(
    Config.PROCESSED_DATA_DIR /
    "X_test.pkl"
)

print(
    "X_train:",
    X_train.shape
)

print(
    "X_test :",
    X_test.shape
)


# =====================================================
# CREATE FEATURE ENGINEER
# =====================================================

engineer = FeatureEngineer()


# =====================================================
# PHASE 4.1
# SELECT TOP 50 FEATURES
# =====================================================

selected_features = (
    engineer.select_top_features(
        ranking,
        n_features=50,
    )
)


# =====================================================
# PHASE 4.2
# APPLY FEATURE SELECTION
# =====================================================

X_train_selected, X_test_selected = (
    engineer.apply_feature_selection(
        X_train,
        X_test,
        selected_features,
    )
)


# =====================================================
# PHASE 4.3
# SAVE SELECTED FEATURES
# =====================================================

engineer.save_selected_features(
    selected_features
)


# =====================================================
# FINAL RESULTS
# =====================================================

print()
print("=" * 50)
print("FINAL FEATURE SELECTION")
print("=" * 50)

print(
    f"Original features : {X_train.shape[1]}"
)

print(
    f"Selected features : "
    f"{X_train_selected.shape[1]}"
)

print(
    f"X_train shape     : "
    f"{X_train_selected.shape}"
)

print(
    f"X_test shape      : "
    f"{X_test_selected.shape}"
)


# =====================================================
# DISPLAY SELECTED FEATURES
# =====================================================

print()
print("=" * 50)
print("SELECTED TOP 50 FEATURES")
print("=" * 50)

for index, feature in enumerate(
    selected_features,
    start=1,
):

    print(
        f"{index:02d}. {feature}"
    )


# =====================================================
# VERIFY TRAIN / TEST CONSISTENCY
# =====================================================

print()
print("=" * 50)
print("FEATURE CONSISTENCY CHECK")
print("=" * 50)

print(
    "Same feature count:",
    X_train_selected.shape[1]
    == X_test_selected.shape[1]
)

print(
    "Same feature order:",
    list(X_train_selected.columns)
    == list(X_test_selected.columns)
)


# =====================================================
# COMPLETION
# =====================================================

print()
print("=" * 50)
print("FEATURE SELECTION COMPLETED SUCCESSFULLY")
print("=" * 50)