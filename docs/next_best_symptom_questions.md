# Next-Best Symptom Questions

## Purpose and boundaries

`POST /next-question` ranks optional, unanswered symptom prompts after an existing prediction has abstained. It does not change `/predict`, calibration, abstention thresholds, SHAP, evidence retrieval, or grounded explanation behavior. It does not diagnose, triage, or recommend treatment. The ranking is an algorithmic development aid, not a clinically validated interview.

## Training-only statistics

`scripts/build_symptom_question_statistics.py` reads only `data/processed/X_train.pkl` and `data/processed/y_train.pkl`, plus the saved feature and label ordering. It does not read the held-out classifier test split or the raw training CSV. For each symptom and class it saves the smoothed conditional frequency `P(symptom=yes | class)` and provenance identifying the train split and sample count. The checked-in statistics contain 79,892 training rows, 30 classes, and 174 symptom features. Rebuild with:

```powershell
python scripts/build_symptom_question_statistics.py
```

Smoothing and candidate thresholds are configured in `configs/question_ranking.json`.

## Ranking

For the current calibrated class probabilities `P(C)` and training-derived class-conditional frequency `P(Q=yes | C)`, the ranker computes binary-question mutual information in bits:

`IG(C; Q) = H(C) - P(yes) H(C | yes) - P(no) H(C | no)`

It sorts descending by estimated information gain, then uses saved model feature order as a deterministic tie-break. Constant features and gains at or below the configured floor are excluded. The API returns the method, prior entropy, candidate counts, status, and suggestions; the numeric gain describes the ranking algorithm only and is not medical confidence.

## API and answer handling

Request:

```json
{
  "symptoms": ["symptom_fatigue"],
  "answered_negative_symptoms": ["symptom_cough"],
  "skipped_symptoms": [],
  "top_k": 3
}
```

Positive, negative, and skipped identifiers must be unique and mutually exclusive. A Yes response adds the symptom to the positive input, after which the user can explicitly run the existing analysis again. No and Skip only exclude that candidate from future questions; neither is added to the classifier input. The endpoint returns explicit empty statuses when no usable candidates remain. The UI is shown only when the existing prediction reports `insufficient_evidence`, and users can continue without answering.

## Evaluation

`scripts/evaluate_next_questions.py` uses four manually authored development scenarios and training-derived statistics. It does not use classifier test samples. Its report is `docs/next_question_evaluation.json`.

The scenarios produced a useful question in 4/4 cases; mean estimated top-question entropy reduction was 0.065210 bits. No candidate repeated across 12 simulated No/Skip slots. The complete-selection and degenerate-posterior checks returned zero questions. These are algorithmic retrieval/ranking diagnostics only, not diagnostic, clinical, or classifier metrics. The hand-authored examples are illustrative; they do not establish clinical usefulness.
