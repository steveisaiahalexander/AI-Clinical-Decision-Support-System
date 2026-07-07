from src.data.dataset import DatasetLoader
from src.data.validation import DatasetValidator
from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer

# -------------------------------------------------------

train = DatasetLoader.load_training_dataset()

DatasetValidator.validate(train)

preprocessor = DataPreprocessor()

processed = preprocessor.preprocess(train)

engineer = FeatureEngineer()

# Phase 1
X_train = engineer.remove_constant_features(
    processed["X_train"]
)

# Phase 2
report = engineer.correlation_analysis(
    X_train
)

print("\nOriginal Shape")
print(processed["X_train"].shape)

print("\nNew Shape")
print(X_train.shape)

print("\nHighly Correlated Features")
print(report.head())