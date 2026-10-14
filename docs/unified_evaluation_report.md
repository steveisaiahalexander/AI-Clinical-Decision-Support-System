# Unified Evaluation Report

Saved evaluation results consolidated without retraining or rerunning expensive experiments. Metrics are kept separate by dataset and evaluation type; no overall score is produced.

## Classification
Dataset: Final held-out classifier test set (n=19974). Type: classifier. Source: `outputs/reports/final_ensemble_metrics.json`.
- accuracy: 0.7348
- weighted_precision: 0.7320
- weighted_recall: 0.7348
- weighted_f1: 0.7301
- macro_precision: 0.6869
- macro_recall: 0.5133
- macro_f1: 0.5479

Class-wise results are included in the JSON report for 30 classes.

## Calibration
Method: Scalar temperature scaling selected by a separate development partition; no test-set tuning. Source: `configs/ensemble_calibration.json`.
### Validation
- raw: multiclass_brier_score=0.4316, log_loss=0.9611, top_label_ece=0.1348
- selected: multiclass_brier_score=0.4045, log_loss=0.8796, top_label_ece=0.0171
### Method Selection
- raw: multiclass_brier_score=0.4306, log_loss=0.9480, top_label_ece=0.1351
- temperature_scaled: multiclass_brier_score=0.4039, log_loss=0.8630, top_label_ece=0.0109

## Uncertainty and Abstention
- coverage: 0.4876
- abstention_rate: 0.5124
- abstention_count: 4094
- accuracy_when_accepted: 0.8943
- accuracy_on_all_rows: 0.7018
- selection_coverage: 0.5001
- selection_accuracy_when_accepted: 0.8941
- minimum_coverage_target: 0.5000
- abstention_counts_by_reason: unavailable
- Caveat: Development estimates from temporary models fitted on a subset of X_train; not measured on the deployed full-data model or held-out test set.

## Retrieval
Dataset: Manually curated queries in rag/evaluation_queries.json; classifier test samples not used. These are retrieval metrics only.
- tfidf: Recall@1=0.9000, Recall@3=1.0000, MRR@3=0.9500, source coverage=1.0000
- semantic: Recall@1=1.0000, Recall@3=1.0000, MRR@3=1.0000, source coverage=1.0000
- hybrid: Recall@1=0.9667, Recall@3=1.0000, MRR@3=0.9833, source coverage=1.0000
- Supported corpus coverage: 30 classes.

## Grounded LLM
Structural/mock fixture results only; citation validity is not semantic or medical truth validation.
- citation_validity: 0.7500
- citation_completeness: 0.8000
- grounded_claim_rate: 0.6000
- structurally_unsupported_claim_rate: 0.4000
- evidence_gate_correct: True
- Live quality evaluation: not_measured.

## Next-Question Engine
Algorithmic development diagnostics only, not clinical outcomes.
- query_count: 4
- queries_with_useful_question: 100.0000
- mean_top_question_expected_entropy_reduction_bits: 0.0652
- repeat_rate_after_skip_responses: 0.0000
- question_slots_simulated: 12
- mean_eligible_candidates: 171

## Limitations
- Held-out test metrics are reused from the saved final report for final reporting only; the consolidator does not read test samples or tune anything.
- Abstention validation metrics come from temporary development models fitted on a subset of training data; per-reason counts on the deployed model are unavailable.
- No live LLM quality score or semantic verification of medical claims is available.
- Synthetic training data and manually curated fixtures do not establish clinical utility or safety.
