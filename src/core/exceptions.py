"""
Custom Exceptions
=================

Defines project-specific exceptions used throughout the
AI Clinical Decision Support System.
"""


class ClinicalDecisionSupportError(Exception):
    """
    Base exception for the project.
    """

    def __init__(self, message: str = "An unexpected application error occurred."):
        super().__init__(message)


# ==========================================================
# DATASET EXCEPTIONS
# ==========================================================

class DatasetNotFoundError(ClinicalDecisionSupportError):
    """Raised when a dataset file cannot be located."""


class InvalidDatasetError(ClinicalDecisionSupportError):
    """Raised when a dataset is empty or corrupted."""


class SchemaMismatchError(ClinicalDecisionSupportError):
    """Raised when the dataset schema is invalid."""


class MissingTargetColumnError(ClinicalDecisionSupportError):
    """Raised when the target column is missing."""


# ==========================================================
# MODEL EXCEPTIONS
# ==========================================================

class ModelNotTrainedError(ClinicalDecisionSupportError):
    """Raised when prediction is attempted before training."""


class UnsupportedModelError(ClinicalDecisionSupportError):
    """Raised when an unsupported model is requested."""


# ==========================================================
# PREDICTION EXCEPTIONS
# ==========================================================

class PredictionError(ClinicalDecisionSupportError):
    """Raised when prediction fails."""


# ==========================================================
# EXPLAINABILITY EXCEPTIONS
# ==========================================================

class ExplainabilityError(ClinicalDecisionSupportError):
    """Raised when SHAP/LIME explanations fail."""


# ==========================================================
# CONFIGURATION EXCEPTIONS
# ==========================================================

class ConfigurationError(ClinicalDecisionSupportError):
    """Raised when application configuration is invalid."""