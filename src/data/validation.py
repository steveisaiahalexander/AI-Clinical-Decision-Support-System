"""
Dataset Validation
==================

Performs integrity checks on datasets before they
enter the machine learning pipeline.
"""

import pandas as pd

from src.core.constants import TARGET_COLUMN
from src.core.exceptions import (
    DataValidationError,
    MissingTargetColumnError,
)
from src.core.logger import logger


class DatasetValidator:

    @staticmethod
    def check_empty(df: pd.DataFrame):

        if df.empty:
            raise DataValidationError("Dataset is empty.")

    @staticmethod
    def check_target_column(df: pd.DataFrame):

        if TARGET_COLUMN not in df.columns:

            raise MissingTargetColumnError(
                f"Target column '{TARGET_COLUMN}' not found."
            )

    @staticmethod
    def check_missing_values(df: pd.DataFrame):

        missing = df.isna().sum().sum()

        if missing > 0:

            logger.warning(
                f"Dataset contains {missing} missing values."
            )

        return missing

    @staticmethod
    def check_duplicate_rows(df: pd.DataFrame):

        duplicates = df.duplicated().sum()

        if duplicates > 0:

            logger.warning(
                f"Dataset contains {duplicates} duplicate rows."
            )

        return duplicates

    @staticmethod
    def check_duplicate_columns(df: pd.DataFrame):

        duplicates = df.columns[df.columns.duplicated()]

        if len(duplicates):

            raise DataValidationError(
                f"Duplicate columns detected: {list(duplicates)}"
            )

    @staticmethod
    def check_binary_features(df: pd.DataFrame):

        feature_columns = df.drop(columns=[TARGET_COLUMN])

        for column in feature_columns.columns:

            unique = set(feature_columns[column].dropna().unique())

            if not unique.issubset({0, 1}):

                raise DataValidationError(
                    f"Non-binary values found in '{column}'"
                )

    @staticmethod
    def check_constant_features(df: pd.DataFrame):

        constant = []

        for column in df.columns:

            if column == TARGET_COLUMN:
                continue

            if df[column].nunique() == 1:
                constant.append(column)

        if constant:

            logger.warning(
                f"{len(constant)} constant features detected."
            )

        return constant

    @classmethod
    def validate(cls, df: pd.DataFrame):

        logger.info("=" * 50)
        logger.info("VALIDATING DATASET")
        logger.info("=" * 50)

        cls.check_empty(df)

        cls.check_target_column(df)

        missing = int(cls.check_missing_values(df))

        duplicate_rows = int(cls.check_duplicate_rows(df))

        cls.check_duplicate_columns(df)

        cls.check_binary_features(df)

        constant = cls.check_constant_features(df)

        logger.info("Validation completed successfully.")

        return {

            "rows": df.shape[0],

            "columns": df.shape[1],

            "missing_values": missing,

            "duplicate_rows": duplicate_rows,

            "constant_features": constant,

            "target_column": TARGET_COLUMN

        }