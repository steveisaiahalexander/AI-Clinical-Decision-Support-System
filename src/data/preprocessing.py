"""
Data Preprocessing
==================

Prepares validated datasets for machine learning.
"""

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.core.config import Config
from src.core.constants import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.core.logger import logger


class DataPreprocessor:
    """
    Handles preprocessing before model training.
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()

    def split_features_target(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Split dataset into features and target.
        """

        logger.info("Splitting features and target...")

        X = dataframe.drop(columns=[TARGET_COLUMN])

        y = dataframe[TARGET_COLUMN]

        return X, y

    def encode_target(self, target):
        """
        Encode disease labels.
        """

        logger.info("Encoding target labels...")

        y = self.label_encoder.fit_transform(target)

        encoder_path = Config.MODEL_DIR / "label_encoder.pkl"

        joblib.dump(self.label_encoder, encoder_path)

        logger.info(
            f"Saved LabelEncoder to {encoder_path}"
        )

        # --------------------------------------------
        # Display class distribution
        # --------------------------------------------

        distribution = (
            pd.Series(y)
            .value_counts()
            .sort_index()
        )

        logger.info("=" * 50)
        logger.info("ENCODED CLASS DISTRIBUTION")
        logger.info("=" * 50)

        logger.info(distribution.to_string())

        return y

    def split_train_test(
        self,
        X,
        y,
    ):
        """
        Perform train/test split.
        """

        logger.info("Performing train/test split...")

        return train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y,

        )

    def preprocess(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Execute complete preprocessing pipeline.
        """

        X, y = self.split_features_target(dataframe)

        y = self.encode_target(y)

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.split_train_test(X, y)

        # --------------------------------------------
        # Save processed datasets
        # --------------------------------------------

        logger.info("Saving processed datasets...")

        joblib.dump(
            X_train,
            Config.PROCESSED_DATA_DIR / "X_train.pkl",
        )

        joblib.dump(
            X_test,
            Config.PROCESSED_DATA_DIR / "X_test.pkl",
        )

        joblib.dump(
            y_train,
            Config.PROCESSED_DATA_DIR / "y_train.pkl",
        )

        joblib.dump(
            y_test,
            Config.PROCESSED_DATA_DIR / "y_test.pkl",
        )

        logger.info(
            f"Saved processed data to {Config.PROCESSED_DATA_DIR}"
        )

        logger.info("Preprocessing completed successfully.")

        return {

            "X_train": X_train,

            "X_test": X_test,

            "y_train": y_train,

            "y_test": y_test,

            "label_encoder": self.label_encoder,

        }