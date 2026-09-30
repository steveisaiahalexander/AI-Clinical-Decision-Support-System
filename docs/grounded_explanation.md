# Grounded LLM Explanation

This milestone adds optional language-model synthesis after the saved classifier and hybrid evidence retrieval. It does not change `/predict`, the SHAP `/explain` route, `/evidence`, calibration, uncertainty, abstention thresholds, trained artifacts, or either retriever.

## Provider and Model

The provider is local Ollama, called through its local `/api/chat` endpoint with a JSON Schema response format. The default model is `qwen2.5:1.5b-instruct-q4_K_M`, 1.54B parameters, Q4_K_M quantization, about 986 MB, Ollama model digest prefix `65ec06548149`, and Apache-2.0 license. It is small enough for the development GPU and supports structured JSON output. Ollama supports constraining chat output with a JSON schema through the `format` field. See the [official model listing](https://ollama.com/library/qwen2.5:1.5b-instruct-q4_K_M) and [structured-output documentation](https://docs.ollama.com/capabilities/structured-outputs).

This machine currently has no Ollama runtime or generation model installed. The application therefore returns `llm_unavailable` by default while keeping retrieved evidence available. There is no paid-provider fallback and no API key. Install Ollama separately, then run `ollama run qwen2.5:1.5b-instruct-q4_K_M` to download and start the model. The first generation request may take longer while the model loads.

Install the small optional HTTP dependency with:

    python -m pip install -r rag/requirements-llm.txt

Configuration defaults live in `rag/llm_config.json`. Environment variables override them: `RAG_LLM_PROVIDER` (`ollama` or `disabled`), `RAG_LLM_BASE_URL`, `RAG_LLM_MODEL`, `RAG_LLM_TIMEOUT_SECONDS`, and `RAG_LLM_ALLOW_REMOTE`. `.env.example` lists the values, but this API does not load `.env` files automatically; pass them to the API process environment. Remote hosts are denied by default. Explicit remote opt-in requires HTTPS. The HTTP client ignores ambient proxy settings and does not send credentials.

## Request Flow

`POST /explain-grounded` accepts selected symptoms. The route independently runs the saved predictor and retrieves up to the configured evidence top-k using `hybrid` RRF. The JSON payload sent to the local model includes the top model class, probability status, uncertainty and abstention fields, selected symptoms, and retrieved passage text with title, organization, URL, section, and stable per-request IDs (`E1`, `E2`, ...).

The system prompt treats symptom/evidence JSON as data, requires supplied evidence only, disallows diagnosis, treatment, medication, dosage, and emergency guidance, and distinguishes model output from retrieved medical context. Ollama receives a Pydantic-derived JSON Schema. The model returns claim objects only; the API constructs the displayed `summary` deterministically by joining each validated claim with its citations:

    {
      "summary": "... [E1] ... [E2]",
      "claims": [
        {"text": "...", "evidence_ids": ["E1"]}
      ],
      "grounding_status": "grounded"
    }

The response also includes the full prediction (including uncertainty and abstention state), the source-attributed evidence entries keyed by evidence ID, a status message, and a safety disclaimer. The API binds evidence IDs to server-owned passages; the model cannot provide or alter source metadata.

The existing SHAP endpoint remains `POST /explain`; it is intentionally not overloaded with the new feature. `/predict` and `/evidence` remain independently usable and backward-compatible.

## Validation and Fallbacks

Application validation requires non-empty claim text, one to three IDs per claim, the `E<number>` format, and IDs present in that request's retrieved evidence. Duplicate IDs, citations embedded in free-form claim text, disallowed advice/diagnosis language, malformed JSON, unknown IDs, or missing citations reject the entire generated explanation. No claim text is returned on validation failure. This is deterministic citation/structure and phrase-policy validation, not medical entailment verification or proof of truth.

Generation is not attempted if the classifier abstains or retrieval is unsupported/insufficient. Abstention returns `grounding_status: abstained`; missing evidence returns `insufficient_evidence` and the message “Insufficient evidence for a grounded explanation.” If Ollama is stopped, the model is missing, or generation times out, the route returns `llm_unavailable` with the retrieved passages intact. These states are also shown in the React summary panel, above the existing Evidence / Clinical Context panel.

The disclaimer states that the summary describes model behavior and retrieved context, does not confirm a disease or provide diagnosis/treatment recommendations, and that the classifier uses synthetic data. Citation structure does not establish clinical efficacy.

## Evaluation

`rag/grounded_evaluation_cases.json` contains three manually curated representative symptom/evidence overlaps plus challenge fixtures for unknown IDs, missing citations, malformed output, classifier abstention, unsupported coverage, and provider unavailability. Run:

    python scripts/evaluate_grounded_explanations.py
    python -m unittest tests.test_grounded_explanation

The report in `docs/grounded_explanation_evaluation.json` uses deterministic mocked drafts. Citation validity and completeness are calculated on generation-eligible fixtures; grounded-claim and structurally-unsupported rates measure whether citations pass the validator, not whether a medical statement is true. These are not live LLM quality metrics, retrieval metrics, classifier metrics, or clinical safety/validity measures. The current environment lacks Ollama, so no actual generated output is included in the evaluation. No held-out classifier test samples are used.
