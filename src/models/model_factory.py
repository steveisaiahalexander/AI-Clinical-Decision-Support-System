from typing import Dict

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class ModelFactory:
    """
    Factory responsible for creating classification models.

    All models use the same training interface:
        model.fit(X_train, y_train)
    """

    @staticmethod
    def create_logistic_regression():
        """
        Create Logistic Regression model.
        """

        return LogisticRegression(
            max_iter=2000,
            random_state=42,
            n_jobs=-1,
        )

    @staticmethod
    def create_decision_tree():
        """
        Create Decision Tree classifier.
        """

        return DecisionTreeClassifier(
            random_state=42,
        )

    @staticmethod
    def create_random_forest():
        """
        Create Random Forest classifier.
        """

        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )

    @staticmethod
    def create_extra_trees():
        """
        Create Extra Trees classifier.
        """

        return ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )

    @staticmethod
    def create_gradient_boosting():
        """
        Create Gradient Boosting classifier.
        """

        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )

    @classmethod
    def get_all_models(cls) -> Dict[str, object]:
        """
        Create and return all baseline models.

        Returns
        -------
        dict
            Dictionary containing model names and
            initialized model objects.
        """

        return {
            "logistic_regression":
                cls.create_logistic_regression(),

            "decision_tree":
                cls.create_decision_tree(),

            "random_forest":
                cls.create_random_forest(),

            "extra_trees":
                cls.create_extra_trees(),

            "gradient_boosting":
                cls.create_gradient_boosting(),
        }