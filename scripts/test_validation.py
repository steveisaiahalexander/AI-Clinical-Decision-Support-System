from pprint import pprint


from src.data.dataset import DatasetLoader
from src.data.validation import DatasetValidator

train = DatasetLoader.load_training_dataset()

DatasetValidator.validate(train)

class_report = DatasetValidator.analyze_class_support(
    train,
    min_samples=20,
)

modeling_data = DatasetValidator.filter_supported_classes(
    train,
    min_samples=20,
)

print("\nModeling Dataset Shape")
print(modeling_data.shape)

print("\nModeling Classes")
print(modeling_data["disease"].nunique())

# =====================================================
# CLASS SUPPORT ANALYSIS
# =====================================================

class_report = DatasetValidator.analyze_class_support(
    train,
    min_samples=20,
)


# =====================================================
# FILTER SUPPORTED CLASSES
# =====================================================

modeling_data = DatasetValidator.filter_supported_classes(
    train,
    min_samples=20,
)


# =====================================================
# SAVE CLASS SUPPORT REPORT
# =====================================================

DatasetValidator.save_class_support_report(
    class_report
)


# =====================================================
# SAVE MODELING DATASET
# =====================================================

DatasetValidator.save_modeling_dataset(
    modeling_data
)


# =====================================================
# DISPLAY RESULTS
# =====================================================

print()

print("=" * 50)
print("MODELING DATASET")
print("=" * 50)

print(
    "Shape:",
    modeling_data.shape
)

print(
    "Classes:",
    modeling_data["disease"].nunique()
)