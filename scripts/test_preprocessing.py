from src.data.dataset import DatasetLoader
from src.data.validation import DatasetValidator
from src.data.preprocessing import DataPreprocessor

train = DatasetLoader.load_training_dataset()

DatasetValidator.validate(train)

preprocessor = DataPreprocessor()

data = preprocessor.preprocess(train)

print("\nProcessed Dataset Shapes")
print("-" * 40)

print("X_train :", data["X_train"].shape)
print("X_test  :", data["X_test"].shape)
print("y_train :", data["y_train"].shape)
print("y_test  :", data["y_test"].shape)