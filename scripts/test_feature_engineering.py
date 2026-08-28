from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer


# =====================================================
# PREPROCESS DATA
# =====================================================

preprocessor = DataPreprocessor()

processed = preprocessor.preprocess()


# =====================================================
# FEATURE ENGINEERING
# =====================================================

engineer = FeatureEngineer()


# =====================================================
# PHASE 3.1
# REMOVE CONSTANT FEATURES
# =====================================================

X_train = engineer.remove_constant_features(
    processed["X_train"]
)

print()

print("=" * 50)
print("CONSTANT FEATURE REMOVAL")
print("=" * 50)

print(
    "Original Shape:",
    processed["X_train"].shape
)

print(
    "New Shape:",
    X_train.shape
)


# =====================================================
# PHASE 3.2
# CORRELATION ANALYSIS
# =====================================================

report = engineer.correlation_analysis(
    X_train
)

print()

print("=" * 50)
print("CORRELATION ANALYSIS")
print("=" * 50)

print(
    "Original Features:",
    processed["X_train"].shape[1]
)

print(
    "Remaining Features:",
    X_train.shape[1]
)

print()

print("Highly Correlated Features")

print(
    report.head(10).to_string(index=False)
)


# =====================================================
# PHASE 3.3
# RANDOM FOREST FEATURE IMPORTANCE
# =====================================================

importance = engineer.feature_importance_pipeline(
    X_train,
    processed["y_train"],
)

print()

print("=" * 50)
print("TOP 10 IMPORTANT FEATURES")
print("=" * 50)

print(
    importance
    .head(10)
    .to_string(index=False)
)


# =====================================================
# PHASE 3.4
# MUTUAL INFORMATION
# =====================================================

mi = engineer.compute_mutual_information(
    X_train,
    processed["y_train"],
)

print()

print("=" * 50)
print("TOP 10 MUTUAL INFORMATION FEATURES")
print("=" * 50)

print(
    mi
    .head(10)
    .to_string(index=False)
)


# =====================================================
# PHASE 3.5
# TRAIN MODEL FOR PERMUTATION IMPORTANCE
# =====================================================

model = engineer.train_feature_importance_model(
    X_train,
    processed["y_train"],
)


# =====================================================
# PHASE 3.6
# PERMUTATION IMPORTANCE
# =====================================================

permutation = (
    engineer.compute_permutation_importance(
        model,
        X_train,
        processed["y_train"],
    )
)

print()

print("=" * 50)
print("TOP 10 PERMUTATION FEATURES")
print("=" * 50)

print(
    permutation
    .head(10)
    .to_string(index=False)
)


# =====================================================
# PHASE 3.7
# COMBINE FEATURE RANKINGS
# =====================================================

ranking = engineer.combine_feature_rankings(

    importance,

    mi,

    permutation,

)

print()

print("=" * 50)
print("FINAL FEATURE RANKING")
print("=" * 50)

print(
    ranking
    .head(20)
    .to_string(index=False)
)


# =====================================================
# SAVE FINAL FEATURE RANKING
# =====================================================

engineer.save_feature_ranking(
    ranking
)


# =====================================================
# COMPLETION
# =====================================================

print()

print("=" * 50)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 50)

print(
    "Final feature count:",
    len(ranking)
)

print(
    "Top feature:",
    ranking.iloc[0]["Feature"]
)