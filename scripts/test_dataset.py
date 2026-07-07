from src.core.logger import logger
from src.data.dataset import DatasetLoader

train = DatasetLoader.load_training_dataset()

DatasetLoader.dataset_summary(train)

logger.info("First 10 Columns:")
print(train.iloc[:, :10].head())

logger.info(f"Target Column: {train.columns[-1]}")
logger.info(f"Dataset Shape: {train.shape}")