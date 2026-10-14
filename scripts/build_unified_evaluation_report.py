"""Consolidate saved evaluation artifacts without rerunning experiments."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


SOURCES = {
    "classification": "outputs/reports/final_ensemble_metrics.json",
    "calibration_policy": "configs/ensemble_calibration.json",
    "retrieval": "docs/evidence_retrieval_evaluation.json",
    "grounded_llm": "docs/grounded_explanation_evaluation.json",
    "next_questions": "docs/next_question_evaluation.json",
}


def _read_json(root: Path, relative_path: str) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads((root / relative_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _metric(
    value: Any,
    category: str,
    dataset: str,
    method: str,
    source: str,
    unit: str = "ratio",
    note: Optional[str] = None,
) -> Dict[str, Any]:
    result = {
        "value": value,
        "availability": "available" if value is not None else "unavailable",
        "category": category,
        "dataset": dataset,
        "method": method,
        "source": source,
        "unit": unit,
    }
    if note:
        result["note"] = note
    return result


def _missing_metric(category: str, dataset: str, method: str, source: str, reason: str):
    return _metric(None, category, dataset, method, source, note=reason)


def _classification(root: Path) -> Dict[str, Any]:
    source = SOURCES["classification"]
    data = _read_json(root, source)
    dataset = "Final held-out classifier test set"
    method = "Saved final evaluation report; no fitting or tuning performed by this script"
    names = (
        "accuracy", "weighted_precision", "weighted_recall", "weighted_f1",
        "macro_precision", "macro_recall", "macro_f1",
    )
    if data is None:
        return {
            "availability": "unavailable",
            "evaluation_type": "classifier",
            "dataset": dataset,
            "method": method,
            "source": source,
            "metrics": {
                name: _missing_metric("classifier", dataset, method, source, "Saved final metrics artifact is missing or invalid.")
                for name in names
            },
            "class_wise": {},
        }

    sample_count = data.get("test_samples")
    dataset = f"Final held-out classifier test set (n={sample_count})" if sample_count else dataset
    metrics = {
        name: _metric(data.get(name), "classifier", dataset, method, source)
        for name in names
    }
    report = data.get("classification_report")
    class_wise: Dict[str, Any] = {}
    if isinstance(report, dict):
        for class_name, values in report.items():
            if class_name in {"accuracy", "macro avg", "weighted avg"} or not isinstance(values, dict):
                continue
            class_wise[class_name] = {
                name: _metric(values.get(source_name), "classifier", dataset, method, source,
                              unit="samples" if source_name == "support" else "ratio")
                for name, source_name in (
                    ("precision", "precision"), ("recall", "recall"),
                    ("f1", "f1-score"), ("support", "support"),
                )
            }
    return {
        "availability": "available",
        "evaluation_type": "classifier",
        "dataset": dataset,
        "method": method,
        "source": source,
        "test_set_used_only_for_final_evaluation": data.get("test_set_used_only_for_final_evaluation"),
        "metrics": metrics,
        "class_wise": class_wise,
    }


def _calibration(root: Path) -> Dict[str, Any]:
    source = SOURCES["calibration_policy"]
    data = _read_json(root, source)
    if data is None:
        missing = _missing_metric("classifier_calibration", "Development validation split", "Saved calibration evaluation", source, "Calibration artifact is missing or invalid.")
        return {"availability": "unavailable", "evaluation_type": "classifier_calibration", "source": source, "validation": {"raw": missing, "selected": missing}, "method_selection": {}}

    validation = data.get("validation_metrics", {})
    selection = data.get("selection_metrics", {})
    validation_source = source
    validation_dataset = f"Development validation partition (n={data.get('provenance', {}).get('validation_evaluation_samples', 'unavailable')}); temporary validation models"
    validation_method = "Multiclass Brier score, log loss, and equal-width 15-bin top-label ECE"
    metric_names = ("multiclass_brier_score", "log_loss", "top_label_ece")

    def values_for(block, dataset, method):
        return {
            name: _metric(block.get(name) if isinstance(block, dict) else None,
                          "classifier_calibration", dataset, method, validation_source)
            for name in metric_names
        }

    raw_values = values_for(validation.get("raw", {}), validation_dataset, validation_method)
    selected_values = values_for(validation.get("selected", {}), validation_dataset, validation_method)
    selection_dataset = f"Independent method-selection partition (n={data.get('provenance', {}).get('method_selection_samples', 'unavailable')})"
    selection_method = "Compare raw ensemble with temperature scaling; select only if log loss decreases"
    method_selection = {
        key: values_for(selection.get(key, {}), selection_dataset, selection_method)
        for key in ("raw", "temperature_scaled")
    }
    policy_results = data.get("abstention", {}).get("validation_results", {})
    threshold_selection = data.get("abstention", {}).get("selection_results", {})
    threshold_dataset = f"Independent threshold-selection partition (n={data.get('provenance', {}).get('threshold_selection_samples', 'unavailable')})"
    threshold_method = "Select fixed OR-threshold policy subject to minimum development coverage"
    validation_count = policy_results.get("sample_count")
    abstention_rate = policy_results.get("abstention_rate")
    abstention_count = round(validation_count * abstention_rate) if isinstance(validation_count, int) and isinstance(abstention_rate, (int, float)) else None

    def policy_metric(value, dataset, method, unit="ratio", note=None):
        return _metric(value, "uncertainty_abstention", dataset, method, source, unit=unit, note=note)

    return {
        "availability": "available",
        "evaluation_type": "classifier_calibration",
        "method": "Scalar temperature scaling selected by a separate development partition; no test-set tuning",
        "source": source,
        "selected_calibration_method": data.get("calibration", {}).get("method"),
        "temperature": _metric(data.get("calibration", {}).get("temperature"), "classifier_calibration", selection_dataset, selection_method, source),
        "validation": {"dataset": validation_dataset, "raw": raw_values, "selected": selected_values},
        "method_selection": {"dataset": selection_dataset, **method_selection},
        "abstention": {
            "dataset": f"Development validation partition (n={validation_count}); temporary validation models",
            "method": "Evaluate the locked three-condition OR policy without refitting thresholds",
            "metrics": {
                "coverage": policy_metric(policy_results.get("coverage"), f"Development validation partition (n={validation_count})", "Fixed selected thresholds", note="Estimate for temporary validation models; may differ from the full-data deployed model."),
                "abstention_rate": policy_metric(policy_results.get("abstention_rate"), f"Development validation partition (n={validation_count})", "Fixed selected thresholds", note="Estimate for temporary validation models; may differ from the full-data deployed model."),
                "abstention_count": policy_metric(abstention_count, f"Development validation partition (n={validation_count})", "Rounded from saved abstention rate × sample count", unit="samples"),
                "accuracy_when_accepted": policy_metric(policy_results.get("accuracy_when_accepted"), f"Development validation partition (n={validation_count})", "Accuracy on non-abstained rows", note="Selective performance estimate for temporary validation models."),
                "accuracy_on_all_rows": policy_metric(policy_results.get("accuracy_on_all_rows"), f"Development validation partition (n={validation_count})", "Top-1 accuracy before abstention"),
                "selection_coverage": policy_metric(threshold_selection.get("selection_coverage"), threshold_dataset, threshold_method),
                "selection_accuracy_when_accepted": policy_metric(threshold_selection.get("selection_accuracy_when_accepted"), threshold_dataset, threshold_method),
                "minimum_coverage_target": policy_metric(threshold_selection.get("target_minimum_coverage"), threshold_dataset, threshold_method),
                "abstention_counts_by_reason": _missing_metric("uncertainty_abstention", f"Development validation partition (n={validation_count})", "Per-reason count from saved evaluation predictions", source, "The saved aggregate artifact does not include reason-level counts."),
            },
            "caveat": "Development estimates from temporary models fitted on a subset of X_train; not measured on the deployed full-data model or held-out test set.",
        },
    }


def _retrieval(root: Path) -> Dict[str, Any]:
    source = SOURCES["retrieval"]
    data = _read_json(root, source)
    category = "retrieval"
    dataset = "Manually curated queries in rag/evaluation_queries.json; classifier test samples not used"
    method = "Offline relevance/source matching; strategy-specific Recall@k and MRR@3"
    if data is None:
        return {"availability": "unavailable", "evaluation_type": category, "dataset": dataset, "method": method, "source": source, "strategies": {}}
    strategies = {}
    for strategy, values in data.get("strategies", {}).items():
        metrics = {}
        for key, unit in (("recall_at_1", "ratio"), ("recall_at_3", "ratio"), ("mrr_at_3", "ratio"), ("queries_with_no_relevant_evidence", "queries")):
            metrics[key] = _metric(values.get(key), category, dataset, method, source, unit)
        source_coverage = values.get("source_coverage", {})
        metrics["source_coverage_recall"] = _metric(source_coverage.get("recall"), category, dataset, "Distinct retrieved sources / expected sources", source, note=f"{source_coverage.get('sources_retrieved', 'unavailable')} / {source_coverage.get('sources_expected', 'unavailable')} sources")
        metrics["queries_without_sufficient_evidence"] = _metric(values.get("queries_without_sufficient_evidence"), category, dataset, "Retriever evidence threshold behavior", source, "queries")
        strategies[strategy] = {"query_count": _metric(values.get("query_count"), category, dataset, method, source, "queries"), "metrics": metrics}
    return {
        "availability": "available",
        "evaluation_type": category,
        "dataset": dataset,
        "method": method,
        "source": source,
        "supported_class_coverage": _metric(
            len(json.loads((root / "rag" / "supported_conditions.json").read_text(encoding="utf-8"))),
            category,
            "Curated corpus versus the supported model classes",
            "Corpus/document coverage check",
            "rag/supported_conditions.json and retrieval tests",
            "classes",
        ),
        "unsupported_query_behavior": {
            "availability": "verified_by_fixture",
            "expected_behavior": "Unsupported class returns an explicit unsupported/no-sufficient-evidence state and no passages.",
            "dataset": "Unknown-condition fixture in tests/test_evidence_retrieval.py",
            "method": "Deterministic API behavior test",
            "source": "tests/test_evidence_retrieval.py::test_unknown_condition_returns_no_unattributed_fallback",
        },
        "strategies": strategies,
        "scope": data.get("scope"),
    }


def _grounded(root: Path) -> Dict[str, Any]:
    source = SOURCES["grounded_llm"]
    data = _read_json(root, source)
    dataset = "Nine manually curated grounded-explanation fixtures with deterministic mocked drafts"
    method = "Structural citation validation, evidence-gate status checks; not semantic truth validation"
    category = "grounded_llm_structural_mock"
    names = ("citation_validity", "citation_completeness", "grounded_claim_rate", "structurally_unsupported_claim_rate")
    if data is None:
        return {"availability": "unavailable", "evaluation_type": category, "dataset": dataset, "method": method, "source": source, "metrics": {name: _missing_metric(category, dataset, method, source, "Saved grounded evaluation artifact is missing or invalid.") for name in names}}
    metrics = {}
    for name in names:
        block = data.get(name, {})
        metrics[name] = _metric(block.get("rate"), category, dataset, method, source, note=f"{block.get('valid_references', block.get('accepted_claims_with_known_citations', block.get('claims_with_any_reference', block.get('rejected_or_uncited_fixture_claims', 'unavailable'))))} / {block.get('all_references_in_fixture_drafts', block.get('fixture_claims', 'unavailable'))}")
    gate = data.get("evidence_gate_behavior", {})
    metrics["evidence_gate_correct"] = _metric(gate.get("correct"), category, dataset, "Compare expected versus actual LLM-call gating", source, "boolean")
    live_count = data.get("live_llm_invocations")
    return {
        "availability": "available",
        "evaluation_type": category,
        "dataset": dataset,
        "method": method,
        "source": source,
        "metrics": metrics,
        "manual_case_count": _metric(data.get("manual_case_count"), category, dataset, method, source, "cases"),
        "mocked_evaluation": True,
        "live_evaluation": {"status": "not_measured", "invocations_in_saved_report": live_count, "reason": "The saved report used mocks; optional live Ollama tests are separate and not a scored quality evaluation."},
        "semantic_grounding_validation": {"status": "not_measured", "note": "Citation ID/structure acceptance does not establish factual or medical correctness."},
        "status_counts": data.get("status_counts", {}),
        "scope": data.get("scope"),
    }


def _questions(root: Path) -> Dict[str, Any]:
    source = SOURCES["next_questions"]
    data = _read_json(root, source)
    dataset = "Four manually authored development query cases using X_train/y_train-derived symptom frequencies"
    method = "Expected information-gain ranking diagnostics; simulated exclusions after No/Skip"
    category = "next_question_algorithmic_development"
    if data is None:
        return {"availability": "unavailable", "evaluation_type": category, "dataset": dataset, "method": method, "source": source, "metrics": {}}

    def metric(value, unit="ratio", note=None):
        return _metric(value, category, dataset, method, source, unit, note)

    return {
        "availability": "available",
        "evaluation_type": category,
        "dataset": dataset,
        "method": method,
        "source": source,
        "metrics": {
            "query_count": metric(data.get("query_count"), "queries"),
            "queries_with_useful_question": metric(data.get("queries_with_useful_question", {}).get("percent"), "percent"),
            "mean_top_question_expected_entropy_reduction_bits": metric(data.get("mean_top_question_expected_entropy_reduction_bits"), "bits"),
            "repeat_rate_after_skip_responses": metric(data.get("repeat_rate_after_skip_responses")),
            "question_slots_simulated": metric(data.get("question_slots_simulated"), "question_slots"),
            "mean_eligible_candidates": metric(data.get("eligible_candidates", {}).get("mean"), "candidates"),
        },
        "edge_cases": data.get("edge_cases", {}),
        "training_statistics_source": data.get("training_statistics_source"),
        "scope": data.get("scope"),
    }


def build_unified_report(root: Path) -> Dict[str, Any]:
    """Build a provenance-aware report only from existing saved artifacts."""
    root = Path(root)
    calibration = _calibration(root)
    sections = {
        "classification": _classification(root),
        "calibration": calibration,
        "uncertainty_abstention": calibration["abstention"] if calibration.get("availability") == "available" else {"availability": "unavailable", "evaluation_type": "uncertainty_abstention", "source": SOURCES["calibration_policy"]},
        "retrieval": _retrieval(root),
        "grounded_llm": _grounded(root),
        "next_question": _questions(root),
    }
    return {
        "report_version": 1,
        "title": "Unified Evaluation Report",
        "evaluation_integrity": {
            "classifier_test_samples_used_for_tuning": False,
            "classifier_retrained": False,
            "source_artifacts_reused": True,
            "note": "Metrics remain separated by evaluation dataset and type; no aggregate score is computed.",
        },
        "sections": sections,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    sections = report["sections"]

    def val(metric):
        value = metric.get("value") if isinstance(metric, dict) else None
        if value is None:
            return "unavailable"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# Unified Evaluation Report",
        "",
        "Saved evaluation results consolidated without retraining or rerunning expensive experiments. Metrics are kept separate by dataset and evaluation type; no overall score is produced.",
        "",
        "## Classification",
    ]
    classification = sections["classification"]
    lines.append(f"Dataset: {classification.get('dataset', 'unavailable')}. Type: classifier. Source: `{classification.get('source', '')}`.")
    for name, metric in classification.get("metrics", {}).items():
        lines.append(f"- {name}: {val(metric)}")
    lines.extend(["", f"Class-wise results are included in the JSON report for {len(classification.get('class_wise', {}))} classes.", "", "## Calibration"])
    calibration = sections["calibration"]
    lines.append(f"Method: {calibration.get('method', 'unavailable')}. Source: `{calibration.get('source', '')}`.")
    for phase in ("validation", "method_selection"):
        lines.append(f"### {phase.replace('_', ' ').title()}")
        phase_data = calibration.get(phase, {})
        for variant in ("raw", "selected", "temperature_scaled"):
            values = phase_data.get(variant)
            if not isinstance(values, dict):
                continue
            shown = ", ".join(f"{key}={val(metric)}" for key, metric in values.items())
            lines.append(f"- {variant}: {shown}")
    abstention = sections["uncertainty_abstention"]
    lines.extend(["", "## Uncertainty and Abstention"])
    for name, metric in abstention.get("metrics", {}).items():
        lines.append(f"- {name}: {val(metric)}")
    lines.append(f"- Caveat: {abstention.get('caveat', 'No saved estimate available.')}")
    retrieval = sections["retrieval"]
    lines.extend(["", "## Retrieval", f"Dataset: {retrieval.get('dataset', 'unavailable')}. These are retrieval metrics only."])
    for strategy, entry in retrieval.get("strategies", {}).items():
        metrics = entry.get("metrics", {})
        lines.append(f"- {strategy}: Recall@1={val(metrics.get('recall_at_1'))}, Recall@3={val(metrics.get('recall_at_3'))}, MRR@3={val(metrics.get('mrr_at_3'))}, source coverage={val(metrics.get('source_coverage_recall'))}")
    lines.append(f"- Supported corpus coverage: {val(retrieval.get('supported_class_coverage'))} classes.")
    grounded = sections["grounded_llm"]
    lines.extend(["", "## Grounded LLM", "Structural/mock fixture results only; citation validity is not semantic or medical truth validation."])
    for name, metric in grounded.get("metrics", {}).items():
        lines.append(f"- {name}: {val(metric)}")
    lines.append(f"- Live quality evaluation: {grounded.get('live_evaluation', {}).get('status', 'unavailable')}.")
    questions = sections["next_question"]
    lines.extend(["", "## Next-Question Engine", "Algorithmic development diagnostics only, not clinical outcomes."])
    for name, metric in questions.get("metrics", {}).items():
        lines.append(f"- {name}: {val(metric)}")
    lines.extend(["", "## Limitations", "- Held-out test metrics are reused from the saved final report for final reporting only; the consolidator does not read test samples or tune anything.", "- Abstention validation metrics come from temporary development models fitted on a subset of training data; per-reason counts on the deployed model are unavailable.", "- No live LLM quality score or semantic verification of medical claims is available.", "- Synthetic training data and manually curated fixtures do not establish clinical utility or safety.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    report = build_unified_report(root)
    json_path = args.json_out or root / "docs" / "unified_evaluation_report.json"
    markdown_path = args.markdown_out or root / "docs" / "unified_evaluation_report.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
