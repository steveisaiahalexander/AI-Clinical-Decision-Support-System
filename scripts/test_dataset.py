from src.core.logger import logger
from src.data.dataset import DatasetLoader

train = DatasetLoader.load_training_dataset()

DatasetLoader.dataset_summary(train)

logger.info("Columns:")

for col in train.columns:
    print(col)