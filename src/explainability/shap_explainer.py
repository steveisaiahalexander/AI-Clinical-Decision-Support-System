"""Local, component-level SHAP explanations for the saved ensemble."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from catboost import Pool


class EnsembleShapExplainer:
    """Explain each tree model's raw score for one selected class."""

    TOP_FEATURES = 5

    def __init__(self, predictor: Any):
        import shap

        self.predictor = predictor
        self._xgb_explainer = shap.TreeExplainer(
            predictor.xgb_model,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )

    def explain(
        self,
        symptoms: Dict[str, Any],
        class_label: Any,
    ) -> List[Dict[str, Any]]:
        """Return local attributions for selected symptoms, separately by model."""
        feature_columns = self.predictor.get_feature_columns()
        row = pd.DataFrame(
            [{feature: symptoms[feature] for feature in feature_columns}],
            columns=feature_columns,
        )
        selected = [
            feature for feature in feature_columns
            if float(row.iloc[0][feature]) != 0.0
        ]
        if not selected:
            raise ValueError("At least one selected symptom is required for explanation.")

        components = []
        for name, model in (
            ("XGBoost", self.predictor.xgb_model),
            ("CatBoost", self.predictor.catboost_model),
        ):
            class_index = self._class_index(model.classes_, class_label, name)
            if name == "XGBoost":
                values = self._xgb_explainer.shap_values(
                    row,
                    check_additivity=False,
                )
                feature_values = self._class_values(
                    values,
                    class_index,
                    len(feature_columns),
                    len(model.classes_),
                    name,
                )
            else:
                values = model.get_feature_importance(
                    Pool(row, feature_names=feature_columns),
                    type="ShapValues",
                    thread_count=4,
                )
                feature_values = self._catboost_class_values(
                    values,
                    class_index,
                    len(feature_columns),
                    len(model.classes_),
                )
            selected_values = [
                (feature, float(feature_values[feature_columns.index(feature)]))
                for feature in selected
            ]
            selected_values.sort(key=lambda item: abs(item[1]), reverse=True)

            components.append(
                {
                    "model": name,
                    "output_scale": "raw_class_score",
                    "features": [
                        {
                            "feature": feature,
                            "shap_value": value,
                            "direction": (
                                "increases_class_score"
                                if value > 0
                                else "decreases_class_score"
                                if value < 0
                                else "no_change"
                            ),
                        }
                        for feature, value in selected_values[: self.TOP_FEATURES]
                    ],
                }
            )
        return components

    @staticmethod
    def _class_index(classes: np.ndarray, class_label: Any, model_name: str) -> int:
        matches = np.flatnonzero(np.asarray(classes) == class_label)
        if len(matches) != 1:
            raise ValueError(f"{model_name} does not contain the predicted class.")
        return int(matches[0])

    @staticmethod
    def _class_values(
        values: Any,
        class_index: int,
        n_features: int,
        n_classes: int,
        model_name: str,
    ) -> np.ndarray:
        if isinstance(values, list):
            if len(values) != n_classes:
                raise ValueError(f"Unexpected {model_name} SHAP class count.")
            result = np.asarray(values[class_index])[0]
        else:
            array = np.asarray(values)
            if array.ndim == 3 and array.shape[0] == 1:
                if array.shape[1:] == (n_features, n_classes):
                    result = array[0, :, class_index]
                elif array.shape[1:] == (n_classes, n_features):
                    result = array[0, class_index, :]
                else:
                    raise ValueError(f"Unexpected {model_name} SHAP output shape.")
            elif array.ndim == 2 and array.shape == (1, n_features):
                result = array[0]
            else:
                raise ValueError(f"Unexpected {model_name} SHAP output shape.")

        result = np.asarray(result, dtype=float).reshape(-1)
        if result.size != n_features or not np.isfinite(result).all():
            raise ValueError(f"Invalid {model_name} SHAP values returned.")
        return result

    @staticmethod
    def _catboost_class_values(
        values: Any,
        class_index: int,
        n_features: int,
        n_classes: int,
    ) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == 3 and array.shape == (1, n_classes, n_features + 1):
            result = array[0, class_index, :-1]
        elif array.ndim == 3 and array.shape == (1, n_features + 1, n_classes):
            result = array[0, :-1, class_index]
        else:
            raise ValueError("Unexpected CatBoost SHAP output shape.")

        result = np.asarray(result, dtype=float).reshape(-1)
        if result.size != n_features or not np.isfinite(result).all():
            raise ValueError("Invalid CatBoost SHAP values returned.")
        return result
