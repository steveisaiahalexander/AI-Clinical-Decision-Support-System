"""Training-split information-gain ranking for optional symptom questions."""

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np


QUESTIONING_DISCLAIMER = (
    "Questions are ranked by estimated information gain from synthetic training data. "
    "This score is not clinical importance, diagnostic certainty, or a validated interview. "
    "No and Skip answers only prevent repeat questions; they are not classifier inputs."
)


def entropy_bits(probabilities: Sequence[float]) -> float:
    """Shannon entropy in bits, ignoring zero-mass classes safely."""
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("A finite, nonnegative class distribution is required.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("The class distribution must have positive mass.")
    values = values / total
    positive = values[values > 0]
    return float(-np.sum(positive * np.log2(positive)))


def binary_question_information_gain(
    class_probabilities: Sequence[float],
    symptom_yes_given_class: Sequence[float],
) -> Dict[str, float]:
    """Expected entropy reduction for observing one binary symptom response."""
    prior = np.asarray(class_probabilities, dtype=np.float64).reshape(-1).copy()
    likelihood_yes = np.asarray(symptom_yes_given_class, dtype=np.float64).reshape(-1)
    if prior.size == 0 or prior.size != likelihood_yes.size:
        raise ValueError("Class probabilities and symptom frequencies must align.")
    if (
        not np.isfinite(prior).all()
        or not np.isfinite(likelihood_yes).all()
        or np.any(prior < 0)
        or np.any(likelihood_yes < 0)
        or np.any(likelihood_yes > 1)
    ):
        raise ValueError("Probabilities and symptom frequencies must be finite and in range.")
    total = float(prior.sum())
    if total <= 0:
        raise ValueError("The class distribution must have positive mass.")
    prior = prior / total

    probability_yes = float(np.clip(np.dot(prior, likelihood_yes), 0.0, 1.0))
    probability_no = 1.0 - probability_yes
    prior_entropy = entropy_bits(prior)

    expected_posterior_entropy = 0.0
    if probability_yes > 0:
        posterior_yes = prior * likelihood_yes / probability_yes
        expected_posterior_entropy += probability_yes * entropy_bits(posterior_yes)
    if probability_no > 0:
        posterior_no = prior * (1.0 - likelihood_yes) / probability_no
        expected_posterior_entropy += probability_no * entropy_bits(posterior_no)

    gain = max(0.0, prior_entropy - expected_posterior_entropy)
    return {
        "information_gain_bits": float(gain),
        "prior_entropy_bits": float(prior_entropy),
        "expected_entropy_after_answer_bits": float(expected_posterior_entropy),
        "probability_yes": probability_yes,
    }


def display_symptom_label(feature_id: str) -> str:
    value = feature_id.removeprefix("rare_")
    value = value.removeprefix("symptom_")
    value = value.removesuffix("_rare").removesuffix("_symptom")
    return " ".join(part.capitalize() for part in value.split("_")).replace("Gi ", "GI ")


class SymptomQuestionRanker:
    """Rank unasked features from a saved, class-conditional training summary."""

    def __init__(
        self,
        feature_order: Iterable[str],
        class_order: Iterable[str],
        conditional_yes: Mapping[str, Sequence[float]],
        constant_features: Iterable[str] = (),
        min_information_gain_bits: float = 1e-6,
    ):
        self.feature_order = tuple(feature_order)
        self.class_order = tuple(class_order)
        self.conditional_yes = {
            str(feature): tuple(float(value) for value in values)
            for feature, values in conditional_yes.items()
        }
        self.constant_features: Set[str] = set(constant_features)
        self.min_information_gain_bits = float(min_information_gain_bits)
        if not self.feature_order or len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("Feature order must be non-empty and unique.")
        if not self.class_order or len(set(self.class_order)) != len(self.class_order):
            raise ValueError("Class order must be non-empty and unique.")
        if not math.isfinite(self.min_information_gain_bits) or self.min_information_gain_bits < 0:
            raise ValueError("Minimum information gain must be finite and nonnegative.")

    def rank(
        self,
        class_probabilities: Mapping[str, float],
        excluded_features: Iterable[str] = (),
        top_k: int = 3,
    ) -> Dict[str, Any]:
        if not 1 <= int(top_k) <= 10:
            raise ValueError("top_k must be between 1 and 10.")
        unknown = sorted(set(excluded_features) - set(self.feature_order))
        if unknown:
            raise ValueError(f"Unknown symptoms: {', '.join(unknown)}")

        prior = [float(class_probabilities.get(label, 0.0)) for label in self.class_order]
        prior_entropy = entropy_bits(prior)
        excluded = set(excluded_features)
        remaining_count = sum(feature not in excluded for feature in self.feature_order)
        eligible_count = 0
        scored: List[tuple[int, Dict[str, Any]]] = []

        for feature_index, feature_id in enumerate(self.feature_order):
            if feature_id in excluded or feature_id in self.constant_features:
                continue
            frequencies = self.conditional_yes.get(feature_id)
            if frequencies is None or len(frequencies) != len(self.class_order):
                continue
            if any(not math.isfinite(value) or value < 0 or value > 1 for value in frequencies):
                continue

            eligible_count += 1
            gain = binary_question_information_gain(prior, frequencies)
            if gain["information_gain_bits"] <= self.min_information_gain_bits:
                continue
            scored.append(
                (
                    feature_index,
                    {
                        "feature_id": feature_id,
                        "label": display_symptom_label(feature_id),
                        "information_gain_bits": gain["information_gain_bits"],
                        "expected_entropy_after_answer_bits": gain[
                            "expected_entropy_after_answer_bits"
                        ],
                        "rationale": (
                            "Ranked by estimated expected reduction in class entropy, "
                            "using yes/no symptom frequencies from the training split."
                        ),
                    },
                )
            )

        scored.sort(key=lambda item: (-item[1]["information_gain_bits"], item[0]))
        suggestions = [item[1] for item in scored[: int(top_k)]]
        if not remaining_count:
            status = "no_unasked_candidates"
            message = "No unasked, nonconstant symptom candidates remain."
        elif not eligible_count:
            status = "no_eligible_candidates"
            message = "No unasked symptom has complete, usable training statistics."
        elif not suggestions:
            status = "no_useful_candidate"
            message = "No unasked symptom exceeded the configured information-gain floor."
        else:
            status = "suggestions_available"
            message = "Suggestions are algorithmic prompts, not clinical recommendations."

        return {
            "suggestions": suggestions,
            "eligible_candidate_count": eligible_count,
            "useful_candidate_count": len(scored),
            "prior_entropy_bits": prior_entropy,
            "status": status,
            "status_message": message,
            "disclaimer": QUESTIONING_DISCLAIMER,
        }
