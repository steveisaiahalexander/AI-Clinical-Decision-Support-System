import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.core.logger import logger


class ModelEvaluator:
    """
    Handles evaluation of trained classification models.
    """

    @staticmethod
    def evaluate_model(
        model,
        X_test,
        y_test,
        model_name="model",
    ):
        """
        Evaluate a trained classification model.

        Returns
        -------
        dict
            Evaluation metrics, classification report,
            and confusion matrix.
        """

        logger.info("=" * 50)
        logger.info(
            f"EVALUATING MODEL: {model_name}"
        )
        logger.info("=" * 50)

        # =================================================
        # PREDICTIONS
        # =================================================

        y_pred = model.predict(X_test)

        # =================================================
        # WEIGHTED METRICS
        # =================================================

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        weighted_precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        weighted_recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        # =================================================
        # MACRO METRICS
        # =================================================

        macro_precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        macro_recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        # =================================================
        # CLASSIFICATION REPORT
        # =================================================

        report_dict = classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True,
        )

        report_text = classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )

        # =================================================
        # CONFUSION MATRIX
        # =================================================

        matrix = confusion_matrix(
            y_test,
            y_pred,
        )

        # =================================================
        # LOG METRICS
        # =================================================

        logger.info(
            f"Accuracy          : {accuracy:.4f}"
        )

        logger.info(
            f"Weighted Precision : "
            f"{weighted_precision:.4f}"
        )

        logger.info(
            f"Weighted Recall    : "
            f"{weighted_recall:.4f}"
        )

        logger.info(
            f"Weighted F1        : "
            f"{weighted_f1:.4f}"
        )

        logger.info(
            f"Macro Precision    : "
            f"{macro_precision:.4f}"
        )

        logger.info(
            f"Macro Recall       : "
            f"{macro_recall:.4f}"
        )

        logger.info(
            f"Macro F1           : "
            f"{macro_f1:.4f}"
        )

        logger.info(
            f"Evaluation completed: {model_name}"
        )

        # =================================================
        # RETURN RESULTS
        # =================================================

        return {

            "model": model_name,

            "accuracy": accuracy,

            "precision": weighted_precision,

            "recall": weighted_recall,

            "f1_score": weighted_f1,

            "weighted_precision":
                weighted_precision,

            "weighted_recall":
                weighted_recall,

            "weighted_f1":
                weighted_f1,

            "macro_precision":
                macro_precision,

            "macro_recall":
                macro_recall,

            "macro_f1":
                macro_f1,

            "classification_report":
                report_text,

            "classification_report_dict":
                report_dict,

            "confusion_matrix":
                matrix,
        }