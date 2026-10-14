"""Build question-ranking frequencies from the train split only."""

import json
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def build_statistics(root: Path = ROOT) -> dict:
    config = json.loads((root / "configs" / "question_ranking.json").read_text(encoding="utf-8"))
    X_train = joblib.load(root / "data" / "processed" / "X_train.pkl")
    y_train = np.asarray(joblib.load(root / "data" / "processed" / "y_train.pkl"))
    feature_order = list(joblib.load(root / "models" / "feature_columns.pkl"))
    label_encoder = joblib.load(root / "models" / "label_encoder.pkl")
    class_order = [str(label) for label in label_encoder.classes_.tolist()]

    if list(X_train.columns) != feature_order:
        raise ValueError("X_train feature order does not match the saved inference feature order.")
    values = X_train.to_numpy(dtype=np.float64, copy=False)
    if values.ndim != 2 or values.shape[1] != len(feature_order):
        raise ValueError("X_train has an invalid feature shape.")
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("Question statistics require finite binary training features.")
    if y_train.ndim != 1 or len(y_train) != len(values):
        raise ValueError("y_train does not align with X_train.")
    if not np.issubdtype(y_train.dtype, np.integer):
        raise ValueError("y_train must contain the saved integer class indices.")

    class_count = len(class_order)
    if np.any(y_train < 0) or np.any(y_train >= class_count):
        raise ValueError("y_train contains a class index outside the saved label encoder.")
    class_counts = np.bincount(y_train, minlength=class_count).astype(np.int64)
    if np.any(class_counts == 0):
        raise ValueError("Every inference class must have training-split examples.")

    yes_counts = np.zeros((class_count, len(feature_order)), dtype=np.int64)
    for class_index in range(class_count):
        yes_counts[class_index] = values[y_train == class_index].sum(axis=0).astype(np.int64)

    alpha = float(config["smoothing_alpha"])
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("smoothing_alpha must be finite and positive.")
    probabilities = (yes_counts + alpha) / (class_counts[:, None] + 2 * alpha)
    constant_indices = np.flatnonzero(np.ptp(values, axis=0) == 0).tolist()

    return {
        "format_version": 1,
        "source": {
            "X_train": "data/processed/X_train.pkl",
            "y_train": "data/processed/y_train.pkl",
            "sample_count": int(len(y_train)),
            "held_out_test_split_read": False,
            "feature_count": len(feature_order),
        },
        "smoothing_alpha": alpha,
        "class_order": class_order,
        "class_counts": {
            class_order[index]: int(class_counts[index]) for index in range(class_count)
        },
        "feature_order": feature_order,
        "constant_features": [feature_order[index] for index in constant_indices],
        "conditional_probability_yes": {
            feature: [float(value) for value in probabilities[:, index]]
            for index, feature in enumerate(feature_order)
        },
    }


def main() -> None:
    statistics = build_statistics()
    config = json.loads((ROOT / "configs" / "question_ranking.json").read_text(encoding="utf-8"))
    output_path = ROOT / config["statistics_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "samples": statistics["source"]["sample_count"],
                "classes": len(statistics["class_order"]),
                "features": len(statistics["feature_order"]),
                "constant_features": len(statistics["constant_features"]),
                "held_out_test_split_read": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
