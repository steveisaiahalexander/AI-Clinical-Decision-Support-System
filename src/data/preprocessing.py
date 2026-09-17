"""
Data Preprocessing
==================

Prepares the validated modeling dataset for machine learning.

The raw dataset contains 46 disease classes. The validation stage
creates a modeling dataset containing only classes with sufficient
sample support.
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

    # =========================================================
    # LOAD MODELING DATASET
    # =========================================================

    def load_modeling_dataset(self):
        """
        Load the filtered modeling dataset.

        Returns
        -------
        pd.DataFrame
            Dataset containing only sufficiently supported
            disease classes.
        """

        logger.info(
            "Loading modeling dataset..."
        )

        path = Config.MODELING_DATASET

        if not path.exists():

            raise FileNotFoundError(
                f"Modeling dataset not found: {path}. "
                f"Run test_validation.py first."
            )

        dataframe = joblib.load(path)

        logger.info(
            f"Loaded modeling dataset "
            f"({dataframe.shape[0]} rows, "
            f"{dataframe.shape[1]} columns)"
        )

        return dataframe

    # =========================================================
    # SPLIT FEATURES AND TARGET
    # =========================================================

    def split_features_target(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Split dataset into features and target.
        """

        logger.info(
            "Splitting features and target..."
        )

        if TARGET_COLUMN not in dataframe.columns:

            raise ValueError(
                f"Target column '{TARGET_COLUMN}' "
                f"not found in modeling dataset."
            )

        X = dataframe.drop(
            columns=[TARGET_COLUMN]
        )

        y = dataframe[TARGET_COLUMN]

        logger.info(
            f"Features: {X.shape[1]}"
        )

        logger.info(
            f"Samples: {X.shape[0]}"
        )

        logger.info(
            f"Classes: {y.nunique()}"
        )

        return X, y

    # =========================================================
    # ENCODE TARGET
    # =========================================================

    def encode_target(self, target):
        """
        Encode disease labels into numerical classes.
        """

        logger.info(
            "Encoding target labels..."
        )

        y = self.label_encoder.fit_transform(
            target
        )

        encoder_path = (
            Config.MODEL_DIR /
            "label_encoder.pkl"
        )

        joblib.dump(
            self.label_encoder,
            encoder_path,
        )

        logger.info(
            f"Saved LabelEncoder to "
            f"{encoder_path}"
        )

        # -----------------------------------------------------
        # Class distribution
        # -----------------------------------------------------

        distribution = (
            pd.Series(y)
            .value_counts()
            .sort_index()
        )

        logger.info("=" * 50)

        logger.info(
            "ENCODED CLASS DISTRIBUTION"
        )

        logger.info("=" * 50)

        logger.info(
            distribution.to_string()
        )

        return y

    # =========================================================
    # TRAIN / TEST SPLIT
    # =========================================================

    def split_train_test(
        self,
        X,
        y,
    ):
        """
        Perform stratified train/test split.
        """

        logger.info(
            "Performing stratified train/test split..."
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y,

        )

        logger.info(
            f"Training samples: "
            f"{len(X_train)}"
        )

        logger.info(
            f"Testing samples : "
            f"{len(X_test)}"
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
        )

    # =========================================================
    # SAVE PROCESSED DATA
    # =========================================================

    def save_processed_data(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
    ):
        """
        Save train/test datasets.
        """

        logger.info(
            "Saving processed datasets..."
        )

        joblib.dump(
            X_train,
            Config.PROCESSED_DATA_DIR /
            "X_train.pkl",
        )

        joblib.dump(
            X_test,
            Config.PROCESSED_DATA_DIR /
            "X_test.pkl",
        )

        joblib.dump(
            y_train,
            Config.PROCESSED_DATA_DIR /
            "y_train.pkl",
        )

        joblib.dump(
            y_test,
            Config.PROCESSED_DATA_DIR /
            "y_test.pkl",
        )

        logger.info(
            f"Saved processed data to "
            f"{Config.PROCESSED_DATA_DIR}"
        )

    # =========================================================
    # COMPLETE PREPROCESSING PIPELINE
    # =========================================================

    def preprocess(
        self,
        dataframe=None,
    ):
        """
        Execute the complete preprocessing pipeline.

        If dataframe is not supplied, the saved modeling
        dataset is loaded automatically.
        """

        logger.info("=" * 50)

        logger.info(
            "STARTING PREPROCESSING PIPELINE"
        )

        logger.info("=" * 50)

        # -----------------------------------------------------
        # Load modeling dataset
        # -----------------------------------------------------

        if dataframe is None:

            dataframe = (
                self.load_modeling_dataset()
            )

        # -----------------------------------------------------
        # Split features and target
        # -----------------------------------------------------

        X, y = (
            self.split_features_target(
                dataframe
            )
        )

        # -----------------------------------------------------
        # Encode target
        # -----------------------------------------------------

        y = self.encode_target(y)

        # -----------------------------------------------------
        # Train / test split
        # -----------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.split_train_test(
            X,
            y,
        )

        # -----------------------------------------------------
        # Save processed datasets
        # -----------------------------------------------------

        self.save_processed_data(
            X_train,
            X_test,
            y_train,
            y_test,
        )

        logger.info(
            "Preprocessing completed successfully."
        )

        return {

            "X_train": X_train,

            "X_test": X_test,

            "y_train": y_train,

            "y_test": y_test,

            "label_encoder":
                self.label_encoder,

        }