"""Final Ensemble Inference Pipeline.

Loads the trained XGBoost + CatBoost ensemble and classifies a complete
174-feature symptom vector. This module performs inference only; it does not
train or tune models and does not access the test dataset.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from src.models.calibration import temperature_scale
from src.models.uncertainty import abstention_decision, uncertainty_summary


class ClinicalEnsemblePredictor:
    """Inference wrapper for the final XGBoost + CatBoost ensemble."""

    XGB_WEIGHT = 0.675
    CATBOOST_WEIGHT = 0.325
    TOP_K_DEFAULT = 5

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        self.project_root = Path(project_root)
        self.models_dir = self.project_root / "models"

        self.xgb_path = self.models_dir / "final_xgboost_ensemble_component.json"
        self.catboost_path = self.models_dir / "final_catboost_ensemble_component.cbm"
        self.feature_columns_path = self.models_dir / "feature_columns.pkl"
        self.label_encoder_path = self.models_dir / "label_encoder.pkl"
        self.calibration_path = (
            self.project_root / "configs" / "ensemble_calibration.json"
        )

        self._validate_artifacts()
        self.feature_columns = list(joblib.load(self.feature_columns_path))
        self.label_encoder = joblib.load(self.label_encoder_path)
        with open(self.calibration_path, encoding="utf-8") as calibration_file:
            calibration = json.load(calibration_file)
        self.calibration_method = calibration["calibration"]["method"]
        self.calibration_temperature = float(
            calibration["calibration"]["temperature"]
        )
        self.probability_status = (
            "calibrated"
            if self.calibration_method == "temperature_scaling"
            else "raw"
        )
        self.abstention_thresholds = calibration["abstention"]["thresholds"]
        required_thresholds = {
            "min_top_probability",
            "min_top_two_margin",
            "max_normalized_entropy",
        }
        if not required_thresholds.issubset(self.abstention_thresholds):
            raise ValueError("Calibration policy is missing abstention thresholds.")
        threshold_values = [
            float(self.abstention_thresholds[key])
            for key in sorted(required_thresholds)
        ]
        if not np.isfinite(threshold_values).all():
            raise ValueError("Abstention thresholds must be finite.")
        if not 0.0 <= self.abstention_thresholds["min_top_probability"] <= 1.0:
            raise ValueError("Minimum top probability must be between zero and one.")
        if not 0.0 <= self.abstention_thresholds["min_top_two_margin"] <= 1.0:
            raise ValueError("Minimum top-two margin must be between zero and one.")
        if not 0.0 <= self.abstention_thresholds["max_normalized_entropy"] <= 1.0:
            raise ValueError("Maximum normalized entropy must be between zero and one.")
        if self.calibration_method not in {"temperature_scaling", "raw_ensemble"}:
            raise ValueError("Unsupported ensemble calibration method.")
        if not np.isfinite(self.calibration_temperature) or self.calibration_temperature <= 0.0:
            raise ValueError("Calibration temperature must be finite and positive.")
        if self.calibration_method == "raw_ensemble" and not np.isclose(
            self.calibration_temperature, 1.0
        ):
            raise ValueError("Raw ensemble output must use temperature one.")
        provenance = calibration.get("provenance", {})
        if provenance.get("held_out_test_set_used") is not False:
            raise ValueError("Calibration policy must confirm the held-out test set was not used.")
        weights = provenance.get("ensemble_weights", {})
        if weights:
            if not {"xgboost", "catboost"}.issubset(weights):
                raise ValueError("Calibration policy is missing ensemble weights.")
            if (
                not np.isclose(weights["xgboost"], self.XGB_WEIGHT)
                or not np.isclose(weights["catboost"], self.CATBOOST_WEIGHT)
            ):
                raise ValueError("Calibration policy ensemble weights do not match inference.")

        self.xgb_model = XGBClassifier()
        self.xgb_model.load_model(str(self.xgb_path))

        self.catboost_model = CatBoostClassifier()
        self.catboost_model.load_model(str(self.catboost_path))

        self.classes_ = np.asarray(self.xgb_model.classes_)
        catboost_classes = np.asarray(self.catboost_model.classes_)
        if set(self.classes_.tolist()) != set(catboost_classes.tolist()):
            raise ValueError("XGBoost and CatBoost class labels do not match.")
        self._catboost_column_by_class = {
            label: index for index, label in enumerate(catboost_classes.tolist())
        }

        self.n_features = len(self.feature_columns)
        self.n_classes = len(self.classes_)
        if self.n_features != 174:
            raise ValueError(
                f"Expected 174 saved feature columns; found {self.n_features}."
            )
        if self.n_classes != len(self.label_encoder.classes_):
            raise ValueError(
                "The model class count does not match the saved label encoder."
            )

    def _validate_artifacts(self) -> None:
        """Raise a clear error if any required model artifact is missing."""
        required_files = {
            "XGBoost model": self.xgb_path,
            "CatBoost model": self.catboost_path,
            "feature columns": self.feature_columns_path,
            "label encoder": self.label_encoder_path,
            "calibration policy": self.calibration_path,
        }
        missing = [f"{name}: {path}" for name, path in required_files.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Required inference artifacts are missing:\n\n" + "\n".join(missing)
            )

    def _prepare_input(self, symptoms: Dict[str, Any]) -> pd.DataFrame:
        """Validate and order a complete feature dictionary for prediction."""
        if not isinstance(symptoms, dict):
            raise TypeError("Input must be a dictionary mapping feature names to values.")

        expected = set(self.feature_columns)
        provided = set(symptoms)
        missing = expected - provided
        extra = provided - expected
        if missing:
            examples = sorted(missing, key=str)[:20]
            raise ValueError(
                f"Missing required symptom features ({len(missing)}). Examples: {examples}"
            )
        if extra:
            examples = sorted(extra, key=str)[:20]
            raise ValueError(
                f"Unknown symptom features ({len(extra)}). Examples: {examples}"
            )

        X = pd.DataFrame(
            [{feature: symptoms[feature] for feature in self.feature_columns}],
            columns=self.feature_columns,
        ).apply(pd.to_numeric, errors="coerce")

        if X.isna().to_numpy().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
            invalid_features = X.columns[X.isna().any()].tolist()
            raise ValueError(
                "Symptom features must contain finite numeric values. "
                f"Invalid features: {invalid_features}"
            )
        return X

    def predict(
        self,
        symptoms: Dict[str, Any],
        top_k: int = TOP_K_DEFAULT,
    ) -> Dict[str, Any]:
        """Predict the most likely disease and return the top-k alternatives."""
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        top_k = min(top_k, self.n_classes)
        X = self._prepare_input(symptoms)

        xgb_probabilities = self.xgb_model.predict_proba(X)
        catboost_probabilities = self.catboost_model.predict_proba(X)
        catboost_probabilities = catboost_probabilities[
            :, [self._catboost_column_by_class[label] for label in self.classes_.tolist()]
        ]
        raw_ensemble_probabilities = (
            self.XGB_WEIGHT * xgb_probabilities
            + self.CATBOOST_WEIGHT * catboost_probabilities
        )
        if self.calibration_method == "temperature_scaling":
            ensemble_probabilities = temperature_scale(
                raw_ensemble_probabilities, self.calibration_temperature
            )
        else:
            ensemble_probabilities = raw_ensemble_probabilities

        probabilities = ensemble_probabilities[0]
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_index = int(top_indices[0])
        uncertainty = uncertainty_summary(probabilities)
        decision = abstention_decision(uncertainty, self.abstention_thresholds)
        top_predictions: List[Dict[str, Any]] = []
        for rank, class_index in enumerate(top_indices, start=1):
            class_label = self.classes_[class_index]
            if isinstance(class_label, (int, np.integer)):
                disease_name = self.label_encoder.inverse_transform([int(class_label)])[0]
            else:
                disease_name = str(class_label)
            probability = float(probabilities[class_index])
            top_predictions.append(
                {
                    "rank": rank,
                    "disease": str(disease_name),
                    "probability": probability,
                    "percentage": round(probability * 100, 2),
                }
            )

        best_prediction = top_predictions[0]
        return {
            "predicted_disease": best_prediction["disease"],
            "predicted_probability": best_prediction["probability"],
            "predicted_percentage": best_prediction["percentage"],
            "raw_predicted_probability": float(
                raw_ensemble_probabilities[0, top_index]
            ),
            "top_predictions": top_predictions,
            "probability_status": self.probability_status,
            "calibration_method": self.calibration_method,
            "calibration_temperature": self.calibration_temperature,
            "uncertainty": uncertainty,
            "decision_status": decision["decision"],
            "insufficient_evidence": decision["abstained"],
            "abstention_reasons": decision["reasons"],
            "abstention_thresholds": self.abstention_thresholds,
            "thresholds_clinically_validated": False,
            "ensemble_weights": {
                "xgboost": self.XGB_WEIGHT,
                "catboost": self.CATBOOST_WEIGHT,
            },
        }

    def get_feature_columns(self) -> List[str]:
        """Return the saved feature order expected by the models."""
        return self.feature_columns.copy()

    def get_model_information(self) -> Dict[str, Any]:
        """Return basic information about the loaded ensemble."""
        return {
            "model_type": "XGBoost + CatBoost weighted ensemble",
            "xgboost_weight": self.XGB_WEIGHT,
            "catboost_weight": self.CATBOOST_WEIGHT,
            "features": self.n_features,
            "classes": self.n_classes,
            "diseases": self.label_encoder.classes_.tolist(),
            "calibration_method": self.calibration_method,
            "probability_status": self.probability_status,
            "thresholds_clinically_validated": False,
        }
