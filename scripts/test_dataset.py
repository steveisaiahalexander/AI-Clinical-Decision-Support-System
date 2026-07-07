from src.data.dataset import DatasetLoader

train = DatasetLoader.load_training_dataset()

DatasetLoader.dataset_summary(train)

print(train.head())