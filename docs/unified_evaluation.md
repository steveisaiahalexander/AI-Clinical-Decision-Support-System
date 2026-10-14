# Unified Evaluation and Observability

## Evaluation report

`unified_evaluation_report.json` and `unified_evaluation_report.md` consolidate the saved classifier, calibration/abstention, retrieval, grounded-explanation, and next-question evaluation artifacts. Regenerate them with:

```powershell
.\.venv\Scripts\python.exe scripts\build_unified_evaluation_report.py
```

The builder reads saved summaries only. It does not open classifier samples, train a model, alter thresholds, or tune against the held-out classifier test set. That test set appears only for final classifier reporting. Metrics remain separated by dataset and method, with no combined score.

Classification figures are final held-out metrics from the saved ensemble evaluation. Calibration and abstention development figures use the documented training/validation partitions; the abstention estimate is from temporary validation models and may not equal deployed-model behavior. Retrieval figures use manually curated queries and are retrieval metrics, not clinical performance. Grounded-explanation figures are mocked structural/citation checks; they do not establish semantic or medical truth. Next-question figures are algorithmic diagnostics using training-derived symptom frequencies, not clinical outcomes. Missing reason-level abstention counts are reported as unavailable rather than inferred.

## Runtime status and telemetry

- `GET /system-status` reports readiness for the saved predictor, calibration settings, evidence corpus, optional question ranker, and optional local Ollama model. Failure of Ollama degrades optional generation but does not make the core predictor unavailable.
- `GET /metrics` returns the process-local aggregate counters. Counters reset when the API process restarts.
- `GET /evaluation-report` returns the saved JSON report, or a generic unavailable response if it has not been generated.
- The frontend status section is collapsed by default and fetches status only when opened.

Instrumentation stores endpoint-template counts, status codes, aggregate durations, retrieval totals, LLM outcome counts, and aggregate abstention counts/reasons. It does not retain request bodies, query strings, or symptom values. It is development observability, not clinical monitoring or durable production telemetry. Do not expose these development endpoints publicly without deployment-specific access controls.

## Verification

Run backend tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Run frontend checks from `frontend` with `npm test` and `npm run build`.
