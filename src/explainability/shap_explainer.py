"""Selected-feature Shapley attributions for the saved probability ensemble."""

from math import comb
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


class EnsembleShapExplainer:
    """Attribute the predicted-class probability across the selected symptoms.

    Coalition values use the saved models' ``predict_proba`` outputs and the
    same weights as inference. The reference is the all-zero symptom vector.
    Small symptom sets use exact Shapley enumeration; larger sets use a
    deterministic permutation approximation.
    """

    TOP_FEATURES = 5
    EXACT_FEATURE_LIMIT = 12
    PERMUTATION_COUNT = 128
    RANDOM_SEED = 2025
    ADDITIVITY_TOLERANCE = 1e-7

    def __init__(self, predictor: Any):
        self.predictor = predictor

    def explain(
        self,
        symptoms: Dict[str, Any],
        class_label: Any,
    ) -> Dict[str, Any]:
        feature_columns = self.predictor.get_feature_columns()
        values = np.asarray(
            [symptoms[feature] for feature in feature_columns], dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError("Symptom features must contain finite numeric values.")

        selected_indices = np.flatnonzero(values != 0.0).tolist()
        if not selected_indices:
            raise ValueError("At least one selected symptom is required for explanation.")

        xgb_index = self._class_index(
            self.predictor.xgb_model.classes_, class_label, "XGBoost"
        )
        catboost_index = self._class_index(
            self.predictor.catboost_model.classes_, class_label, "CatBoost"
        )
        coalition_masks, sampled_paths = self._coalitions(len(selected_indices))
        coalition_inputs = self._coalition_inputs(
            values, selected_indices, coalition_masks, feature_columns
        )

        xgb_probability = self._class_probabilities(
            self.predictor.xgb_model,
            coalition_inputs,
            xgb_index,
            len(self.predictor.xgb_model.classes_),
            "XGBoost",
        )
        catboost_probability = self._class_probabilities(
            self.predictor.catboost_model,
            coalition_inputs,
            catboost_index,
            len(self.predictor.catboost_model.classes_),
            "CatBoost",
        )

        xgb_values = self._shapley_values(
            xgb_probability, len(selected_indices), coalition_masks, sampled_paths
        )
        catboost_values = self._shapley_values(
            catboost_probability, len(selected_indices), coalition_masks, sampled_paths
        )
        xgb_weight = float(self.predictor.XGB_WEIGHT)
        catboost_weight = float(self.predictor.CATBOOST_WEIGHT)
        if not np.isclose(xgb_weight + catboost_weight, 1.0, atol=1e-12):
            raise ValueError("Ensemble attribution weights must sum to one.")

        ensemble_values = xgb_weight * xgb_values + catboost_weight * catboost_values
        baseline_position = coalition_masks.index(0)
        full_mask = (1 << len(selected_indices)) - 1
        full_position = coalition_masks.index(full_mask)
        ensemble_probabilities = (
            xgb_weight * xgb_probability + catboost_weight * catboost_probability
        )

        self._validate_additivity(
            xgb_values,
            xgb_probability,
            baseline_position,
            full_position,
            "XGBoost",
        )
        self._validate_additivity(
            catboost_values, catboost_probability, baseline_position, full_position, "CatBoost"
        )
        self._validate_additivity(
            ensemble_values,
            ensemble_probabilities,
            baseline_position,
            full_position,
            "ensemble",
        )

        selected_features = [feature_columns[index] for index in selected_indices]
        ensemble_feature_items = [
            (feature, float(ensemble_values[position]))
            for position, feature in enumerate(selected_features)
        ]
        ensemble_feature_items.sort(key=lambda item: abs(item[1]), reverse=True)

        exact = sampled_paths is None
        method = "exact_coalition_shapley" if exact else "sampled_permutation_shapley"
        result = {
            "predicted_class_index": xgb_index,
            "baseline_probability": float(ensemble_probabilities[baseline_position]),
            "predicted_probability": float(ensemble_probabilities[full_position]),
            "output_space": "ensemble_probability",
            "method": method,
            "is_exact": exact,
            "permutation_count": None if exact else len(sampled_paths),
            "features": self._feature_records(
                ensemble_feature_items, self.TOP_FEATURES
            ),
            "component_attributions": [
                self._component_record(
                    "XGBoost model attribution",
                    xgb_index,
                    xgb_weight,
                    xgb_probability,
                    xgb_values,
                    baseline_position,
                    full_position,
                    selected_features,
                ),
                self._component_record(
                    "CatBoost model attribution",
                    catboost_index,
                    catboost_weight,
                    catboost_probability,
                    catboost_values,
                    baseline_position,
                    full_position,
                    selected_features,
                ),
            ],
        }
        return result

    def _coalitions(
        self, feature_count: int
    ) -> Tuple[List[int], Sequence[Sequence[int]] | None]:
        if feature_count <= self.EXACT_FEATURE_LIMIT:
            return list(range(1 << feature_count)), None

        rng = np.random.default_rng(self.RANDOM_SEED)
        paths = [
            rng.permutation(feature_count).tolist()
            for _ in range(self.PERMUTATION_COUNT)
        ]
        masks = {0}
        for path in paths:
            mask = 0
            for feature_index in path:
                mask |= 1 << feature_index
                masks.add(mask)
        return sorted(masks), paths

    @staticmethod
    def _coalition_inputs(
        values: np.ndarray,
        selected_indices: Sequence[int],
        masks: Sequence[int],
        feature_columns: Sequence[str],
    ) -> pd.DataFrame:
        matrix = np.zeros((len(masks), len(feature_columns)), dtype=float)
        for row_index, mask in enumerate(masks):
            for selected_position, feature_index in enumerate(selected_indices):
                if mask & (1 << selected_position):
                    matrix[row_index, feature_index] = values[feature_index]
        return pd.DataFrame(matrix, columns=feature_columns)

    @staticmethod
    def _class_index(classes: Any, class_label: Any, model_name: str) -> int:
        matches = np.flatnonzero(np.asarray(classes) == class_label)
        if len(matches) != 1:
            raise ValueError(f"{model_name} does not contain the predicted class.")
        return int(matches[0])

    @staticmethod
    def _class_probabilities(
        model: Any,
        inputs: pd.DataFrame,
        class_index: int,
        class_count: int,
        model_name: str,
    ) -> np.ndarray:
        probabilities = np.asarray(model.predict_proba(inputs), dtype=float)
        expected_shape = (len(inputs), class_count)
        if probabilities.shape != expected_shape or not np.isfinite(probabilities).all():
            raise ValueError(f"Unexpected {model_name} probability output shape or values.")
        if not 0 <= class_index < class_count:
            raise ValueError(f"Invalid {model_name} predicted class index.")
        selected = probabilities[:, class_index]
        if np.any(selected < 0.0) or np.any(selected > 1.0):
            raise ValueError(f"Invalid {model_name} class probabilities returned.")
        return selected

    @staticmethod
    def _shapley_values(
        coalition_values: np.ndarray,
        feature_count: int,
        coalition_masks: Sequence[int],
        sampled_paths: Sequence[Sequence[int]] | None,
    ) -> np.ndarray:
        if sampled_paths is None:
            contributions = np.zeros(feature_count, dtype=float)
            for feature_index in range(feature_count):
                bit = 1 << feature_index
                for mask in range(1 << feature_count):
                    if mask & bit:
                        continue
                    coalition_size = mask.bit_count()
                    weight = 1.0 / (
                        feature_count * comb(feature_count - 1, coalition_size)
                    )
                    contributions[feature_index] += weight * (
                        coalition_values[mask | bit] - coalition_values[mask]
                    )
            return contributions

        contributions = np.zeros(feature_count, dtype=float)
        coalition_positions = {mask: index for index, mask in enumerate(coalition_masks)}
        for path in sampled_paths:
            previous_mask = 0
            for feature_index in path:
                current_mask = previous_mask | (1 << feature_index)
                contributions[feature_index] += (
                    coalition_values[coalition_positions[current_mask]]
                    - coalition_values[coalition_positions[previous_mask]]
                )
                previous_mask = current_mask
        return contributions / len(sampled_paths)

    @staticmethod
    def _validate_additivity(
        attributions: np.ndarray,
        coalition_values: np.ndarray,
        baseline_position: int,
        full_position: int,
        model_name: str,
    ) -> None:
        expected = coalition_values[full_position] - coalition_values[baseline_position]
        if not np.isclose(
            float(np.sum(attributions)),
            float(expected),
            atol=EnsembleShapExplainer.ADDITIVITY_TOLERANCE,
            rtol=1e-7,
        ):
            raise ValueError(
                f"{model_name} probability attributions failed additivity validation."
            )

    def _component_record(
        self,
        model_name: str,
        class_index: int,
        weight: float,
        probabilities: np.ndarray,
        attributions: np.ndarray,
        baseline_position: int,
        full_position: int,
        selected_features: Sequence[str],
    ) -> Dict[str, Any]:
        items = [
            (feature, float(attributions[position]))
            for position, feature in enumerate(selected_features)
        ]
        items.sort(key=lambda item: abs(item[1]), reverse=True)
        return {
            "model": model_name,
            "class_index": class_index,
            "weight": weight,
            "output_space": "model_probability",
            "baseline_probability": float(probabilities[baseline_position]),
            "predicted_probability": float(probabilities[full_position]),
            "features": self._feature_records(items),
        }

    @staticmethod
    def _feature_records(
        items: Sequence[Tuple[str, float]],
        limit: int | None = None,
    ) -> List[Dict[str, Any]]:
        records = []
        for feature, value in items if limit is None else items[:limit]:
            records.append(
                {
                    "feature": feature,
                    "shap_value": value,
                    "direction": (
                        "increases_probability"
                        if value > 1e-12
                        else "decreases_probability"
                        if value < -1e-12
                        else "no_change"
                    ),
                }
            )
        return records
