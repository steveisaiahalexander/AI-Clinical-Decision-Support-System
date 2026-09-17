"""
Dataset Validation
==================

Performs integrity checks on datasets before they
enter the machine learning pipeline.
"""
import joblib
import pandas as pd
from uvicorn import Config

from src.core.config import Config
from src.core.logger import logger

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

    @staticmethod
    def analyze_class_support(
        dataframe,
        min_samples=20,
    ):
            """
            Analyze target-class sample support.

            Classes with fewer than min_samples
            are considered rare for primary modeling.
            """

            from src.core.constants import TARGET_COLUMN

            logger.info("=" * 50)
            logger.info("CLASS SUPPORT ANALYSIS")
            logger.info("=" * 50)

            distribution = (
                dataframe[TARGET_COLUMN]
                .value_counts()
                .sort_values()
            )

            rare_classes = distribution[
                distribution < min_samples
            ]

            supported_classes = distribution[
                distribution >= min_samples
            ]

            logger.info(
                f"Minimum class support : {min_samples}"
            )

            logger.info(
                f"Total classes         : {len(distribution)}"
            )

            logger.info(
                f"Supported classes     : "
                f"{len(supported_classes)}"
            )

            logger.info(
                f"Rare classes          : "
                f"{len(rare_classes)}"
            )

            logger.info(
                f"Supported samples     : "
                f"{supported_classes.sum()}"
            )

            logger.info(
                f"Rare samples          : "
                f"{rare_classes.sum()}"
            )

            logger.info("=" * 50)
            logger.info("RARE CLASSES")
            logger.info("=" * 50)

            if rare_classes.empty:

                logger.info(
                    "No rare classes detected."
                )

            else:

                logger.info(
                    rare_classes.to_string()
                )

            return {
                "distribution": distribution,
                "rare_classes": rare_classes,
                "supported_classes": supported_classes,
                "min_samples": min_samples,
            }

    @staticmethod
    def filter_supported_classes(
        dataframe,
        min_samples=20,
    ):
        """
        Keep only classes with sufficient samples
        for primary model training.
        """

        from src.core.constants import TARGET_COLUMN

        distribution = (
            dataframe[TARGET_COLUMN]
            .value_counts()
        )

        supported_classes = distribution[
            distribution >= min_samples
        ].index

        filtered_dataframe = dataframe[
            dataframe[TARGET_COLUMN]
            .isin(supported_classes)
        ].copy()

        logger.info("=" * 50)
        logger.info("FILTERING SUPPORTED CLASSES")
        logger.info("=" * 50)

        logger.info(
            f"Original samples : "
            f"{len(dataframe)}"
        )

        logger.info(
            f"Filtered samples : "
            f"{len(filtered_dataframe)}"
        )

        logger.info(
            f"Original classes : "
            f"{dataframe[TARGET_COLUMN].nunique()}"
        )

        logger.info(
            f"Remaining classes: "
            f"{filtered_dataframe[TARGET_COLUMN].nunique()}"
        )

        return filtered_dataframe

    @staticmethod
    def save_class_support_report(
        class_report,
    ):
        """
        Save class-support information as a CSV report.
        """

        report = (
            class_report["distribution"]
            .reset_index()
        )

        report.columns = [
            "disease",
            "sample_count",
        ]

        minimum_samples = (
            class_report["min_samples"]
        )

        report["status"] = report[
            "sample_count"
        ].apply(
            lambda count:
            "supported"
            if count >= minimum_samples
            else "rare"
        )

        output_path = (
            Config.REPORTS_DIR /
            "class_support_report.csv"
        )

        report.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            f"Saved class support report to "
            f"{output_path}"
        )

        return output_path


    @staticmethod
    def save_modeling_dataset(
        dataframe,
    ):
        """
        Save the filtered modeling dataset.
        """

        output_path = (
            Config.MODELING_DATASET
        )

        joblib.dump(
            dataframe,
            output_path,
        )

        logger.info(
            f"Saved modeling dataset to "
            f"{output_path}"
        )

        return output_path  