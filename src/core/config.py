"""
Configuration Module
====================

Central configuration for the AI Clinical Decision Support System.

This module stores all project paths and application-wide settings.
No business logic should exist here.
"""

from pathlib import Path


class Config:
    """
    Central configuration class.

    Every module should import values from this class instead of
    hardcoding paths.
    """

    # ============================================================
    # PROJECT ROOT
    # ============================================================

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # ============================================================
    # DATA
    # ============================================================

    DATA_DIR = PROJECT_ROOT / "data"

    RAW_DATA_DIR = DATA_DIR / "raw"

    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    TRAIN_DATA = RAW_DATA_DIR / "Training.csv"

    TEST_DATA = RAW_DATA_DIR / "Testing.csv"

    # ============================================================
    # MODELS
    # ============================================================

    MODEL_DIR = PROJECT_ROOT / "models"

    TRAINED_MODEL = MODEL_DIR / "disease_model.pkl"

    LABEL_ENCODER = MODEL_DIR / "label_encoder.pkl"

    FEATURE_COLUMNS = MODEL_DIR / "feature_columns.pkl"

    # ============================================================
    # OUTPUTS
    # ============================================================

    OUTPUT_DIR = PROJECT_ROOT / "outputs"

    FIGURES_DIR = OUTPUT_DIR / "figures"

    REPORTS_DIR = OUTPUT_DIR / "reports"

    LOGS_DIR = OUTPUT_DIR / "logs"

    # ============================================================
    # NOTEBOOKS
    # ============================================================

    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

    # ============================================================
    # DOCUMENTATION
    # ============================================================

    DOCS_DIR = PROJECT_ROOT / "docs"

    # ============================================================
    # API
    # ============================================================

    API_DIR = PROJECT_ROOT / "api"

    # ============================================================
    # STREAMLIT
    # ============================================================

    APP_DIR = PROJECT_ROOT / "app"

    # ============================================================
    # CONFIGS
    # ============================================================

    CONFIG_DIR = PROJECT_ROOT / "configs"

    # ============================================================
    # CREATE DIRECTORIES
    # ============================================================

    @classmethod
    @classmethod
    @classmethod
    def initialize(cls):
        """
        Create runtime directories if they don't exist.
        """

        runtime_dirs = [
            cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR,
            cls.OUTPUT_DIR,
            cls.FIGURES_DIR,
            cls.REPORTS_DIR,
            cls.LOGS_DIR,
        ]

        for directory in runtime_dirs:

            print(f"Checking: {directory}")

            directory.mkdir(
                parents=True,
                exist_ok=True
            )

            print(f"✔ Created/Exists: {directory}")