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
from src.core.exceptions import (
    DatasetNotFoundError,
    InvalidDatasetError,
)


class DatasetLoader:
    """
    Handles loading datasets from disk.
    """

    @staticmethod
    def load_csv(path: Union[str, Path]) -> pd.DataFrame:
        """
        Load a CSV dataset.

        Parameters
        ----------
        path : str | Path
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded and cleaned dataset.

        Raises
        ------
        DatasetNotFoundError
            If the dataset file does not exist.

        InvalidDatasetError
            If the dataset cannot be loaded or is empty.
        """

        path = Path(path)

        logger.info(f"Loading dataset: {path}")

        # --------------------------------------------------
        # Check file existence
        # --------------------------------------------------

        if not path.exists():

            logger.error(f"Dataset not found: {path}")

            raise DatasetNotFoundError(
                f"Dataset does not exist: {path}"
            )

        # --------------------------------------------------
        # Read CSV
        # --------------------------------------------------

        try:

            dataframe = pd.read_csv(path)

        except Exception as error:

            logger.exception(error)

            raise InvalidDatasetError(
                f"Unable to load dataset: {path}"
            )

        # --------------------------------------------------
        # Remove automatically generated unnamed columns
        # --------------------------------------------------

        unnamed_columns = dataframe.columns[
            dataframe.columns.str.contains(r"^Unnamed")
        ]

        if len(unnamed_columns) > 0:

            logger.warning(
                f"Dropping unnamed columns: {list(unnamed_columns)}"
            )

            dataframe = dataframe.drop(columns=unnamed_columns)

        # --------------------------------------------------
        # Clean column names
        # --------------------------------------------------

        dataframe.columns = dataframe.columns.str.strip()

        # --------------------------------------------------
        # Verify dataset is not empty
        # --------------------------------------------------

        if dataframe.empty:

            logger.error("Dataset is empty.")

            raise InvalidDatasetError(
                "Dataset contains no rows."
            )

        logger.info(
            f"Loaded dataset successfully "
            f"({dataframe.shape[0]} rows, "
            f"{dataframe.shape[1]} columns)"
        )

        return dataframe

    @classmethod
    def load_training_dataset(cls) -> pd.DataFrame:
        """
        Load the training dataset.
        """

        logger.info("Loading training dataset...")

        return cls.load_csv(Config.TRAIN_DATA)

    @classmethod
    def load_testing_dataset(cls) -> pd.DataFrame:
        """
        Load the testing dataset.
        """

        logger.info("Loading testing dataset...")

        return cls.load_csv(Config.TEST_DATA)

    @staticmethod
    def dataset_summary(dataframe: pd.DataFrame) -> None:
        """
        Display dataset statistics.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataset to summarize.
        """

        logger.info("=" * 50)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 50)

        logger.info(f"Rows           : {dataframe.shape[0]}")
        logger.info(f"Columns        : {dataframe.shape[1]}")
        logger.info(f"Missing Values : {dataframe.isna().sum().sum()}")
        logger.info(f"Duplicate Rows : {dataframe.duplicated().sum()}")

        logger.info("=" * 50)