from pprint import pprint

from src.data.dataset import DatasetLoader
from src.data.validation import DatasetValidator

train = DatasetLoader.load_training_dataset()

report = DatasetValidator.validate(train)

print()

print("Validation Report")

pprint(report)