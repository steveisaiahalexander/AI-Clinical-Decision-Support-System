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

import json
from datetime import datetime

import seaborn as sns
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

from src.core.config import Config
from src.core.logger import logger
from src.core.constants import (
    CORRELATION_THRESHOLD,
    N_ESTIMATORS,
    IMPORTANCE_RANDOM_STATE,
    TOP_FEATURES,
    FIGURE_WIDTH,
    FIGURE_HEIGHT,
    FIGURE_DPI,
    RANDOM_STATE,
)



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
    
# =====================================================
# PHASE 3
# RANDOM FOREST FEATURE IMPORTANCE
# =====================================================

    def train_feature_importance_model(
        self,
        X_train: pd.DataFrame,
        y_train,
    ):
        """
        Train a Random Forest model solely for
        feature importance analysis.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features

        y_train :
            Encoded labels

        Returns
        -------
        RandomForestClassifier
        """

        logger.info("=" * 50)
        logger.info("TRAINING RANDOM FOREST")
        logger.info("=" * 50)

        model = RandomForestClassifier(

            n_estimators=N_ESTIMATORS,

            random_state=IMPORTANCE_RANDOM_STATE,

            n_jobs=-1,

        )

        model.fit(
            X_train,
            y_train,
        )

        logger.info(
            "Random Forest trained successfully."
        )

        # Save model

        model_path = (

            Config.MODEL_DIR /

            "feature_importance_model.pkl"

        )

        joblib.dump(
            model,
            model_path,
        )

        logger.info(
            f"Saved Random Forest model to {model_path}"
        )

        return model
    

    def compute_feature_importance(
        self,
        model: RandomForestClassifier,
        X_train: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute feature importance using the trained
        Random Forest model.

        Parameters
        ----------
        model : RandomForestClassifier
            Trained model.

        X_train : pd.DataFrame
            Training feature matrix.

        Returns
        -------
        pd.DataFrame
            Feature importance sorted in descending order.
        """

        logger.info("=" * 50)
        logger.info("COMPUTING FEATURE IMPORTANCE")
        logger.info("=" * 50)

        importance = pd.DataFrame(
            {
                "Feature": X_train.columns,
                "Importance": model.feature_importances_,
            }
        )

        importance = importance.sort_values(
            by="Importance",
            ascending=False,
        ).reset_index(drop=True)

        logger.info(
            f"Computed importance for {len(importance)} features."
        )

        logger.info(
            f"Most important feature: {importance.iloc[0]['Feature']}"
        )

        logger.info(
            f"Importance Score: {importance.iloc[0]['Importance']:.6f}"
        )

        return importance
    
    def save_feature_importance(
        self,
        importance: pd.DataFrame,
    ) -> None:
        """
        Save feature importance reports.

        Parameters
        ----------
        importance : pd.DataFrame
            Sorted feature importance dataframe.
        """

        logger.info("=" * 50)
        logger.info("SAVING FEATURE IMPORTANCE REPORTS")
        logger.info("=" * 50)

        # -----------------------------------------
        # CSV
        # -----------------------------------------

        csv_path = (
            Config.REPORTS_DIR /
            "feature_importance.csv"
        )

        importance.to_csv(
            csv_path,
            index=False,
        )

        logger.info(
            f"Saved CSV report -> {csv_path}"
        )

        # -----------------------------------------
        # Top 20 CSV
        # -----------------------------------------

        top20_path = (
            Config.REPORTS_DIR /
            "top20_features.csv"
        )

        importance.head(TOP_FEATURES).to_csv(
            top20_path,
            index=False,
        )

        logger.info(
            f"Saved Top {TOP_FEATURES} report -> {top20_path}"
        )

        # -----------------------------------------
        # JSON
        # -----------------------------------------

        json_path = (
            Config.REPORTS_DIR /
            "feature_importance.json"
        )

        report = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_features": len(importance),
            "top_feature": importance.iloc[0]["Feature"],
            "top_importance": float(
                importance.iloc[0]["Importance"]
            ),
            "features": importance.to_dict(
                orient="records"
            ),
        }

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=4,
            )

        logger.info(
            f"Saved JSON report -> {json_path}"
        )

    def plot_feature_importance(
        self,
        importance: pd.DataFrame,
    ) -> None:
        """
        Generate feature importance plots.
        """

        logger.info("=" * 50)
        logger.info("GENERATING FEATURE IMPORTANCE PLOTS")
        logger.info("=" * 50)

        # ======================================================
        # TOP 20 FEATURES
        # ======================================================

        top = importance.head(TOP_FEATURES)

        plt.figure(
            figsize=(
                FIGURE_WIDTH,
                FIGURE_HEIGHT,
            )
        )

        sns.barplot(
            data=top,
            x="Importance",
            y="Feature",
            hue="Feature",
            palette="viridis",
            legend=False,
        )

        plt.title(
            f"Top {TOP_FEATURES} Important Features",
            fontsize=18,
            fontweight="bold",
        )

        plt.xlabel(
            "Feature Importance",
            fontsize=13,
        )

        plt.ylabel(
            "Feature",
            fontsize=13,
        )

        plt.tight_layout()

        top_path = (
            Config.FIGURES_DIR /
            "top20_features.png"
        )

        plt.savefig(
            top_path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            f"Saved {top_path}"
        )

        # ======================================================
        # ALL FEATURES
        # ======================================================

        plt.figure(
            figsize=(
                18,
                7,
            )
        )

        plt.plot(
            importance["Importance"].values,
            linewidth=2,
        )

        plt.title(
            "Feature Importance Distribution",
            fontsize=18,
            fontweight="bold",
        )

        plt.xlabel("Feature Rank")

        plt.ylabel("Importance")

        plt.grid(
            alpha=0.30,
        )

        plt.tight_layout()

        all_path = (
            Config.FIGURES_DIR /
            "feature_importance.png"
        )

        plt.savefig(
            all_path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            f"Saved {all_path}"
        )
    
    # =====================================================
    # COMPLETE FEATURE IMPORTANCE PIPELINE
    # =====================================================

    def feature_importance_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train,
    ) -> pd.DataFrame:
        """
        Execute the complete Random Forest feature
        importance pipeline.

        Steps
        -----
        1. Train Random Forest
        2. Compute feature importance
        3. Save reports
        4. Generate plots

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.

        y_train :
            Encoded target labels.

        Returns
        -------
        pd.DataFrame
            Sorted feature importance dataframe.
        """

        logger.info("=" * 50)
        logger.info("STARTING FEATURE IMPORTANCE PIPELINE")
        logger.info("=" * 50)

        # Train model
        model = self.train_feature_importance_model(
            X_train,
            y_train,
        )

        # Compute importance
        importance = self.compute_feature_importance(
            model,
            X_train,
        )

        # Save reports
        self.save_feature_importance(
            importance,
        )

        # Generate plots
        self.plot_feature_importance(
            importance,
        )

        logger.info("=" * 50)
        logger.info("FEATURE IMPORTANCE PIPELINE COMPLETED")
        logger.info("=" * 50)

        return importance
    
    # =====================================================
    # PHASE 3.2
    # MUTUAL INFORMATION
    # =====================================================

    def compute_mutual_information(
        self,
        X_train: pd.DataFrame,
        y_train,
    ) -> pd.DataFrame:
        """
        Compute Mutual Information scores for all features.
        """

        logger.info("=" * 50)
        logger.info("COMPUTING MUTUAL INFORMATION")
        logger.info("=" * 50)

        scores = mutual_info_classif(
            X_train,
            y_train,
            random_state=IMPORTANCE_RANDOM_STATE,
        )

        mi = pd.DataFrame(
            {
                "Feature": X_train.columns,
                "Mutual Information": scores,
            }
        )

        mi = mi.sort_values(
            by="Mutual Information",
            ascending=False,
        ).reset_index(drop=True)

        logger.info(
            f"Computed Mutual Information for {len(mi)} features."
        )

        logger.info(
            f"Top Feature: {mi.iloc[0]['Feature']}"
        )

        logger.info(
            f"MI Score: {mi.iloc[0]['Mutual Information']:.6f}"
        )

        return mi
    
    # =====================================================
    # PHASE 3.3
    # PERMUTATION IMPORTANCE
    # =====================================================

    def compute_permutation_importance(
        self,
        model,
        X,
        y,
        sample_size=10000,
        n_repeats=5,
    ):
        """
        Compute permutation feature importance.

        A representative sample is used to reduce memory
        consumption on large datasets.

        Parameters
        ----------
        model :
            Trained machine-learning model.

        X : pandas.DataFrame
            Feature dataset.

        y : array-like
            Target values.

        sample_size : int
            Maximum number of samples used for permutation
            importance calculation.

        n_repeats : int
            Number of permutations per feature.

        Returns
        -------
        pandas.DataFrame
            Features ranked by permutation importance.
        """

        logger.info("=" * 50)
        logger.info("COMPUTING PERMUTATION IMPORTANCE")
        logger.info("=" * 50)

        # =====================================================
        # SAMPLE DATA
        # =====================================================

        if len(X) > sample_size:

            logger.info(
                f"Dataset contains {len(X)} samples."
            )

            logger.info(
                f"Sampling {sample_size} samples "
                f"for permutation importance."
            )

            sample_indices = (
                X.sample(
                    n=sample_size,
                    random_state=RANDOM_STATE,
                ).index
            )

            X_sample = X.loc[
                sample_indices
            ]

            if hasattr(y, "loc"):

                y_sample = y.loc[
                    sample_indices
                ]

            else:

                y_sample = y[
                    X.index.get_indexer(
                        sample_indices
                    )
                ]

        else:

            X_sample = X.copy()

            y_sample = y

        logger.info(
            f"Permutation dataset shape: "
            f"{X_sample.shape}"
        )

        # =====================================================
        # COMPUTE PERMUTATION IMPORTANCE
        # =====================================================

        result = permutation_importance(

            model,

            X_sample,

            y_sample,

            n_repeats=n_repeats,

            random_state=RANDOM_STATE,

            n_jobs=1,

            scoring="accuracy",

        )

        # =====================================================
        # CREATE REPORT
        # =====================================================

        permutation = pd.DataFrame({

            "Feature":
                X_sample.columns,

            "Permutation Importance":
                result.importances_mean,

            "Importance Std":
                result.importances_std,

        })

        permutation = (
            permutation
            .sort_values(
                by="Permutation Importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        # =====================================================
        # LOG RESULTS
        # =====================================================

        logger.info(
            f"Computed permutation importance "
            f"for {len(permutation)} features."
        )

        if not permutation.empty:

            logger.info(
                f"Top Feature: "
                f"{permutation.iloc[0]['Feature']}"
            )

            logger.info(
                f"Permutation Score: "
                f"{permutation.iloc[0]['Permutation Importance']:.6f}"
            )

        return permutation

    # =====================================================
    # PHASE 3.4
    # FEATURE RANKING FUSION
    # =====================================================

    def combine_feature_rankings(
        self,
        rf_importance: pd.DataFrame,
        mi_importance: pd.DataFrame,
        permutation_importance_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine multiple feature ranking methods into a
        single consensus ranking.

        Parameters
        ----------
        rf_importance : pd.DataFrame

        mi_importance : pd.DataFrame

        permutation_importance_df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        logger.info("=" * 50)
        logger.info("COMBINING FEATURE RANKINGS")
        logger.info("=" * 50)

        # ------------------------------------------
        # Copy DataFrames
        # ------------------------------------------

        rf = rf_importance.copy()

        mi = mi_importance.copy()

        perm = permutation_importance_df.copy()

        # ------------------------------------------
        # Create rankings
        # ------------------------------------------

        rf["RF Rank"] = range(
            1,
            len(rf) + 1,
        )

        mi["MI Rank"] = range(
            1,
            len(mi) + 1,
        )

        perm["Permutation Rank"] = range(
            1,
            len(perm) + 1,
        )

        # ------------------------------------------
        # Merge
        # ------------------------------------------

        ranking = rf[
            ["Feature", "RF Rank"]
        ]

        ranking = ranking.merge(

            mi[
                ["Feature", "MI Rank"]
            ],

            on="Feature",

        )

        ranking = ranking.merge(

            perm[
                [
                    "Feature",
                    "Permutation Rank",
                ]
            ],

            on="Feature",

        )

        # ------------------------------------------
        # Average Rank
        # ------------------------------------------

        ranking["Average Rank"] = (

            ranking[
                [
                    "RF Rank",
                    "MI Rank",
                    "Permutation Rank",
                ]
            ]

            .mean(axis=1)

        )

        ranking = ranking.sort_values(

            by="Average Rank",

        ).reset_index(

            drop=True

        )

        logger.info(

            "Feature rankings combined successfully."

        )

        return ranking
    
    # =====================================================
    # SAVE COMBINED FEATURE RANKING
    # =====================================================

    def save_feature_ranking(
        self,
        ranking: pd.DataFrame,
    ) -> None:
        """
        Save the combined feature ranking.

        Parameters
        ----------
        ranking : pd.DataFrame
            Combined feature ranking dataframe.
        """

        logger.info("=" * 50)
        logger.info("SAVING FEATURE RANKING")
        logger.info("=" * 50)

        # ------------------------------------------
        # CSV
        # ------------------------------------------

        csv_path = (
            Config.REPORTS_DIR /
            "feature_ranking.csv"
        )

        ranking.to_csv(
            csv_path,
            index=False,
        )

        logger.info(
            f"Saved CSV report -> {csv_path}"
        )

        # ------------------------------------------
        # TOP 20
        # ------------------------------------------

        top20_path = (
            Config.REPORTS_DIR /
            "top20_feature_ranking.csv"
        )

        ranking.head(TOP_FEATURES).to_csv(
            top20_path,
            index=False,
        )

        logger.info(
            f"Saved Top {TOP_FEATURES} report -> {top20_path}"
        )

        # ------------------------------------------
        # JSON
        # ------------------------------------------

        json_path = (
            Config.REPORTS_DIR /
            "feature_ranking.json"
        )

        report = {

            "generated_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "total_features":
                len(ranking),

            "top_feature":
                ranking.iloc[0]["Feature"],

            "average_rank":
                float(
                    ranking.iloc[0]["Average Rank"]
                ),

            "ranking":
                ranking.to_dict(
                    orient="records"
                )

        }

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=4,
            )

        logger.info(
            f"Saved JSON report -> {json_path}"
        )

    
    # =====================================================
    # FEATURE RANKING VISUALIZATION
    # =====================================================

    def plot_feature_ranking(
        self,
        ranking: pd.DataFrame,
    ) -> None:
        """
        Plot the Top N consensus ranked features.
        """

        logger.info("=" * 50)
        logger.info("GENERATING FEATURE RANKING PLOT")
        logger.info("=" * 50)

        top = ranking.head(TOP_FEATURES).copy()

        # Reverse order so best feature appears at top
        top = top.iloc[::-1]

        top["Ranking Score"] = (
        top["Average Rank"].max()
        - top["Average Rank"]
        + 1
        )

        sns.barplot(
        data=top,
        x="Ranking Score",
        y="Feature",
        hue="Feature",
        palette="viridis_r",
        legend=False,
    )

        plt.title(
            f"Top {TOP_FEATURES} Consensus Features",
            fontsize=18,
            fontweight="bold",
        )

        plt.xlabel(
            "Average Rank (Lower is Better)",
            fontsize=13,
        )

        plt.ylabel(
            "Feature",
            fontsize=13,
        )

        plt.grid(
            axis="x",
            alpha=0.30,
        )

        plt.tight_layout()

        output_path = (
            Config.FIGURES_DIR /
            "feature_ranking.png"
        )

        plt.savefig(
            output_path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            f"Saved {output_path}"
        )
