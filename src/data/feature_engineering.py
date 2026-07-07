"""
Feature Engineering
===================

Performs feature engineering before model training.

Phase 1
-------
- Remove constant features

Phase 2
-------
- Correlation analysis
- Correlation report
- Correlation heatmap
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_selection import VarianceThreshold

from src.core.config import Config
from src.core.constants import CORRELATION_THRESHOLD
from src.core.logger import logger


class FeatureEngineer:
    """
    Performs feature engineering operations.
    """

    def __init__(self):

        self.selector = VarianceThreshold(threshold=0.0)

    # =====================================================
    # PHASE 1
    # =====================================================

    def remove_constant_features(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove zero-variance features.
        """

        logger.info("=" * 50)
        logger.info("REMOVING CONSTANT FEATURES")
        logger.info("=" * 50)

        original_features = X.shape[1]

        X_selected = self.selector.fit_transform(X)

        selected_columns = X.columns[
            self.selector.get_support()
        ]

        removed_columns = X.columns[
            ~self.selector.get_support()
        ]

        logger.info(f"Original Features : {original_features}")
        logger.info(f"Remaining Features: {len(selected_columns)}")
        logger.info(f"Removed Features  : {len(removed_columns)}")

        if len(removed_columns):

            logger.warning(
                f"Removed Features: {list(removed_columns)}"
            )

        feature_path = (
            Config.MODEL_DIR /
            "selected_features.pkl"
        )

        joblib.dump(
            list(selected_columns),
            feature_path,
        )

        logger.info(
            f"Saved selected feature list to {feature_path}"
        )

        return pd.DataFrame(
            X_selected,
            columns=selected_columns,
            index=X.index,
        )

    # =====================================================
    # PHASE 2
    # =====================================================

    def compute_correlation_matrix(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute Pearson correlation matrix.
        """

        logger.info("=" * 50)
        logger.info("COMPUTING CORRELATION MATRIX")
        logger.info("=" * 50)

        correlation_matrix = X.corr()

        output_path = (
            Config.REPORTS_DIR /
            "correlation_matrix.csv"
        )

        correlation_matrix.to_csv(output_path)

        logger.info(
            f"Saved correlation matrix to {output_path}"
        )

        return correlation_matrix

    def find_correlated_features(
        self,
        correlation_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Find highly correlated feature pairs.
        """

        logger.info(
            "Searching for highly correlated features..."
        )

        correlated_pairs = []

        columns = correlation_matrix.columns

        for i in range(len(columns)):

            for j in range(i + 1, len(columns)):

                correlation = correlation_matrix.iloc[i, j]

                if abs(correlation) >= CORRELATION_THRESHOLD:

                    correlated_pairs.append(

                        {

                            "Feature 1": columns[i],

                            "Feature 2": columns[j],

                            "Correlation": correlation,

                        }

                    )

        report = pd.DataFrame(correlated_pairs)

        report_path = (
            Config.REPORTS_DIR /
            "correlated_features.csv"
        )

        report.to_csv(
            report_path,
            index=False,
        )

        logger.info(
            f"Found {len(report)} correlated feature pairs."
        )

        logger.info(
            f"Saved report to {report_path}"
        )

        return report

    def plot_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
    ) -> None:
        """
        Generate and save correlation heatmap.
        """

        logger.info(
            "Generating correlation heatmap..."
        )

        plt.figure(figsize=(18, 16))

        plt.imshow(
            correlation_matrix,
            interpolation="nearest",
            aspect="auto",
        )

        plt.colorbar()

        plt.title("Feature Correlation Matrix")

        plt.tight_layout()

        figure_path = (
            Config.FIGURES_DIR /
            "correlation_heatmap.png"
        )

        plt.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            f"Saved heatmap to {figure_path}"
        )

    # =====================================================
    # COMPLETE PHASE 2 PIPELINE
    # =====================================================

    def correlation_analysis(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute complete correlation analysis.
        """

        correlation_matrix = self.compute_correlation_matrix(
            X
        )

        report = self.find_correlated_features(
            correlation_matrix
        )

        self.plot_correlation_heatmap(
            correlation_matrix
        )

        logger.info(
            "Correlation analysis completed successfully."
        )

        return report