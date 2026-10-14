"""Evaluate next-question ranking on manual development queries only."""

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.symptom_input import SymptomVectorizer
from src.models.inference import ClinicalEnsemblePredictor
from src.models.question_ranking import SymptomQuestionRanker


def load_ranker(root: Path) -> SymptomQuestionRanker:
    config = json.loads((root / "configs" / "question_ranking.json").read_text(encoding="utf-8"))
    stats = json.loads((root / config["statistics_path"]).read_text(encoding="utf-8"))
    return SymptomQuestionRanker(
        feature_order=stats["feature_order"],
        class_order=stats["class_order"],
        conditional_yes=stats["conditional_probability_yes"],
        constant_features=stats["constant_features"],
        min_information_gain_bits=config["min_information_gain_bits"],
    )


def run_evaluation(root: Path = ROOT) -> dict:
    case_file = root / "configs" / "question_ranking_cases.json"
    cases = json.loads(case_file.read_text(encoding="utf-8"))["cases"]
    predictor = ClinicalEnsemblePredictor(project_root=root)
    vectorizer = SymptomVectorizer(predictor.get_feature_columns())
    ranker = load_ranker(root)
    config = json.loads((root / "configs" / "question_ranking.json").read_text(encoding="utf-8"))

    results = []
    eligible_counts = []
    useful_count = 0
    top_gains = []
    repeated = 0
    asked_total = 0
    for case in cases:
        symptoms = case["symptoms"]
        prediction = predictor.predict(vectorizer.encode(symptoms), top_k=len(ranker.class_order))
        distribution = {
            item["disease"]: float(item["probability"])
            for item in prediction["top_predictions"]
        }
        ranked = ranker.rank(distribution, excluded_features=symptoms, top_k=3)
        eligible_counts.append(ranked["eligible_candidate_count"])
        useful_count += int(bool(ranked["suggestions"]))
        if ranked["suggestions"]:
            top_gains.append(ranked["suggestions"][0]["information_gain_bits"])

        asked = set(symptoms)
        sequence = []
        for _ in range(int(config["evaluation_steps"])):
            next_item = ranker.rank(distribution, excluded_features=asked, top_k=1)["suggestions"]
            if not next_item:
                break
            feature_id = next_item[0]["feature_id"]
            repeated += int(feature_id in asked)
            asked.add(feature_id)
            asked_total += 1
            sequence.append(feature_id)

        results.append(
            {
                "id": case["id"],
                "decision_status": prediction["decision_status"],
                "eligible_candidate_count": ranked["eligible_candidate_count"],
                "useful_question_available": bool(ranked["suggestions"]),
                "top_questions": ranked["suggestions"],
                "unique_skip_sequence": sequence,
            }
        )

    all_features = list(ranker.feature_order)
    representative_distribution = {
        item["disease"]: float(item["probability"])
        for item in predictor.predict(
            vectorizer.encode(cases[0]["symptoms"]), top_k=len(ranker.class_order)
        )["top_predictions"]
    }
    all_selected = ranker.rank(
        representative_distribution, excluded_features=all_features, top_k=3
    )
    degenerate_distribution = {
        name: float(index == 0) for index, name in enumerate(ranker.class_order)
    }
    no_information = ranker.rank(degenerate_distribution, top_k=3)

    stats = json.loads(
        (root / "configs" / "symptom_question_statistics.json").read_text(encoding="utf-8")
    )
    return {
        "scope": "Algorithmic information-gain evaluation on hand-authored development queries; not clinical outcomes or classifier metrics.",
        "classifier_test_samples_used": False,
        "training_statistics_source": stats["source"],
        "query_count": len(cases),
        "eligible_candidates": {
            "mean": round(mean(eligible_counts), 2) if eligible_counts else 0,
            "minimum": min(eligible_counts, default=0),
            "maximum": max(eligible_counts, default=0),
        },
        "queries_with_useful_question": {
            "count": useful_count,
            "total": len(cases),
            "percent": round(100 * useful_count / max(1, len(cases)), 2),
        },
        "mean_top_question_expected_entropy_reduction_bits": round(mean(top_gains), 6)
        if top_gains
        else 0.0,
        "repeat_rate_after_skip_responses": round(repeated / max(1, asked_total), 6),
        "question_slots_simulated": asked_total,
        "edge_cases": {
            "all_features_selected": {
                "status": all_selected["status"],
                "suggestion_count": len(all_selected["suggestions"]),
            },
            "degenerate_single_class_distribution": {
                "status": no_information["status"],
                "suggestion_count": len(no_information["suggestions"]),
            },
        },
        "cases": results,
    }


def main() -> None:
    report = run_evaluation()
    output_path = ROOT / "docs" / "next_question_evaluation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
