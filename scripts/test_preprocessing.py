import pandas as pd

from src.data.preprocessing import DataPreprocessor


# =====================================================
# INITIALIZE PREPROCESSOR
# =====================================================

preprocessor = DataPreprocessor()


# =====================================================
# RUN PREPROCESSING
# =====================================================

processed = preprocessor.preprocess()


# =====================================================
# DISPLAY SHAPES
# =====================================================

print()

print("=" * 50)
print("PROCESSED DATASET SHAPES")
print("=" * 50)

print(
    "X_train :",
    processed["X_train"].shape
)

print(
    "X_test  :",
    processed["X_test"].shape
)

print(
    "y_train :",
    processed["y_train"].shape
)

print(
    "y_test  :",
    processed["y_test"].shape
)


# =====================================================
# UNIQUE CLASSES
# =====================================================

print()

print("=" * 50)
print("UNIQUE CLASSES")
print("=" * 50)

print(
    "Training classes:",
    len(set(processed["y_train"]))
)

print(
    "Testing classes :",
    len(set(processed["y_test"]))
)


# =====================================================
# TRAINING CLASS DISTRIBUTION
# =====================================================

print()

print("=" * 50)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 50)

train_distribution = (
    pd.Series(processed["y_train"])
    .value_counts()
    .sort_index()
)

print(
    train_distribution.to_string()
)


# =====================================================
# TESTING CLASS DISTRIBUTION
# =====================================================

print()

print("=" * 50)
print("TESTING CLASS DISTRIBUTION")
print("=" * 50)

test_distribution = (
    pd.Series(processed["y_test"])
    .value_counts()
    .sort_index()
)

print(
    test_distribution.to_string()
)


# =====================================================
# DISTRIBUTION CHECK
# =====================================================

print()

print("=" * 50)
print("TRAIN / TEST DISTRIBUTION CHECK")
print("=" * 50)

print(
    "Training samples:",
    len(processed["y_train"])
)

print(
    "Testing samples :",
    len(processed["y_test"])
)

print(
    "Total samples   :",
    len(processed["y_train"])
    + len(processed["y_test"])
)

print(
    "Training classes:",
    len(train_distribution)
)

print(
    "Testing classes :",
    len(test_distribution)
)

if (
    len(train_distribution) == 30
    and len(test_distribution) == 30
):

    print(
        "\nAll 30 classes are present "
        "in both training and testing sets."
    )

else:

    print(
        "\nWARNING: Class distribution "
        "does not contain all 30 classes."
    )