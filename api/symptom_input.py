"""Convert selected symptom names into the model's saved feature vector."""

from typing import Dict, Iterable, List


class UnknownSymptomsError(ValueError):
    """Raised when a request includes names absent from saved features."""

    def __init__(self, symptoms: List[str]):
        self.symptoms = symptoms
        super().__init__(f"Unknown symptoms: {', '.join(symptoms)}")


class SymptomVectorizer:
    """Map selected symptom names to a complete, ordered model input."""

    def __init__(self, feature_columns: Iterable[str]):
        self.feature_columns = list(feature_columns)
        if not self.feature_columns:
            raise ValueError("Saved feature columns cannot be empty.")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("Saved feature columns contain duplicates.")
        self._valid_features = set(self.feature_columns)

    def available_symptoms(self) -> List[str]:
        """Return allowed symptom names in the saved model feature order."""
        return self.feature_columns.copy()

    def encode(self, symptoms: List[str]) -> Dict[str, int]:
        """Build a complete 0/1 dictionary in the exact training order."""
        if not symptoms:
            raise ValueError("Select at least one symptom.")

        duplicates = sorted(
            {name for name in symptoms if symptoms.count(name) > 1}
        )
        if duplicates:
            raise ValueError(f"Duplicate symptoms: {', '.join(duplicates)}")

        unknown = sorted(set(symptoms) - self._valid_features)
        if unknown:
            raise UnknownSymptomsError(unknown)

        selected = set(symptoms)
        return {
            feature: int(feature in selected)
            for feature in self.feature_columns
        }
