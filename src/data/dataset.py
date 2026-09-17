"""
Dataset Loader
==============

Provides reusable functions for loading datasets used
throughout the AI Clinical Decision Support System.
"""

from pathlib import Path
from typing import Union

import pandas as pd

from src.core.config import Config
from src.core.logger import logger
from src.core.constants import TARGET_COLUMN
from src.core.exceptions import (
    DatasetNotFoundError,
    InvalidDatasetError,
)


class DatasetLoader:
    """
    Handles loading datasets from disk.
    """

    @staticmethod
    def load_csv(
        path: Union[str, Path]
    ) -> pd.DataFrame:
        """
        Load a CSV dataset.
        """

        path = Path(path)

        logger.info(
            f"Loading dataset: {path}"
        )

        # ==================================================
        # CHECK FILE EXISTENCE
        # ==================================================

        if not path.exists():

            logger.error(
                f"Dataset not found: {path}"
            )

            raise DatasetNotFoundError(
                f"Dataset does not exist: {path}"
            )

        # ==================================================
        # READ CSV
        # ==================================================

        try:

            dataframe = pd.read_csv(path)

        except Exception as error:

            logger.exception(error)

            raise InvalidDatasetError(
                f"Unable to load dataset: {path}"
            )

        # ==================================================
        # REMOVE UNNAMED COLUMNS
        # ==================================================

        unnamed_columns = dataframe.columns[
            dataframe.columns.astype(str).str.contains(
                r"^Unnamed"
            )
        ]

        if len(unnamed_columns) > 0:

            logger.warning(
                f"Dropping unnamed columns: "
                f"{list(unnamed_columns)}"
            )

            dataframe = dataframe.drop(
                columns=unnamed_columns
            )

        # ==================================================
        # CLEAN COLUMN NAMES
        # ==================================================

        dataframe.columns = (
            dataframe.columns
            .astype(str)
            .str.strip()
        )

        # ==================================================
        # CHECK EMPTY DATASET
        # ==================================================

        if dataframe.empty:

            logger.error(
                "Dataset is empty."
            )

            raise InvalidDatasetError(
                "Dataset contains no rows."
            )

        # ==================================================
        # VERIFY TARGET COLUMN
        # ==================================================

        if TARGET_COLUMN not in dataframe.columns:

            logger.error(
                f"Target column '{TARGET_COLUMN}' "
                f"not found in dataset."
            )

            raise InvalidDatasetError(
                f"Target column '{TARGET_COLUMN}' "
                f"not found in dataset."
            )

        # ==================================================
        # VERIFY TARGET VALUES
        # ==================================================

        if dataframe[TARGET_COLUMN].isna().any():

            missing_target_count = (
                dataframe[TARGET_COLUMN]
                .isna()
                .sum()
            )

            logger.error(
                f"Target column '{TARGET_COLUMN}' "
                f"contains {missing_target_count} "
                f"missing values."
            )

            raise InvalidDatasetError(
                f"Target column '{TARGET_COLUMN}' "
                f"contains missing values."
            )

        # ==================================================
        # LOG SUCCESS
        # ==================================================

        logger.info(
            f"Loaded dataset successfully "
            f"({dataframe.shape[0]} rows, "
            f"{dataframe.shape[1]} columns)"
        )

        return dataframe

    # ======================================================
    # TRAINING DATASET
    # ======================================================

    @classmethod
    def load_training_dataset(
        cls
    ) -> pd.DataFrame:
        """
        Load the training dataset.
        """

        logger.info(
            "Loading training dataset..."
        )

        return cls.load_csv(
            Config.TRAIN_DATA
        )

    # ======================================================
    # TESTING DATASET
    # ======================================================

    @classmethod
    def load_testing_dataset(
        cls
    ) -> pd.DataFrame:
        """
        Load the testing dataset.
        """

        logger.info(
            "Loading testing dataset..."
        )

        return cls.load_csv(
            Config.TEST_DATA
        )

    # ======================================================
    # DATASET SUMMARY
    # ======================================================

    @staticmethod
    def dataset_summary(
        dataframe: pd.DataFrame
    ) -> None:
        """
        Display dataset statistics.
        """

        logger.info("=" * 50)

        logger.info(
            "DATASET SUMMARY"
        )

        logger.info("=" * 50)

        # --------------------------------------------------
        # Dataset size
        # --------------------------------------------------

        logger.info(
            f"Rows           : "
            f"{dataframe.shape[0]}"
        )

        logger.info(
            f"Columns        : "
            f"{dataframe.shape[1]}"
        )

        # --------------------------------------------------
        # Feature count
        # --------------------------------------------------

        feature_count = (
            dataframe.shape[1] - 1
        )

        logger.info(
            f"Features       : "
            f"{feature_count}"
        )

        # --------------------------------------------------
        # Missing values
        # --------------------------------------------------

        logger.info(
            f"Missing Values : "
            f"{dataframe.isna().sum().sum()}"
        )

        # --------------------------------------------------
        # Duplicate rows
        # --------------------------------------------------

        logger.info(
            f"Duplicate Rows : "
            f"{dataframe.duplicated().sum()}"
        )

        # --------------------------------------------------
        # Target column
        # --------------------------------------------------

        logger.info(
            f"Target Column  : "
            f"{TARGET_COLUMN}"
        )

        # --------------------------------------------------
        # Number of classes
        # --------------------------------------------------

        unique_classes = (
            dataframe[TARGET_COLUMN]
            .nunique()
        )

        logger.info(
            f"Unique Classes : "
            f"{unique_classes}"
        )

        # --------------------------------------------------
        # Class distribution
        # --------------------------------------------------

        logger.info("=" * 50)

        logger.info(
            "TARGET CLASS DISTRIBUTION"
        )

        logger.info("=" * 50)

        class_distribution = (
            dataframe[TARGET_COLUMN]
            .value_counts()
        )

        logger.info(
            class_distribution.to_string()
        )

        logger.info("=" * 50)   