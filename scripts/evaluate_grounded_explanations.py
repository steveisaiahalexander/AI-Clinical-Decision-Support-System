"""Evaluate citation structure and evidence-gating with curated mock generations."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.grounded_explanation import GroundedExplanationService
from rag.retrieval import EvidenceRetriever


class FixtureGenerator:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, payload):
        self.calls += 1
        return self.response


def _prediction(condition: str, decision_status: str) -> Dict[str, Any]:
    return {
        "predicted_disease": condition,
        "predicted_probability": 0.62,
        "probability_status": "calibrated",
        "decision_status": decision_status,
        "insufficient_evidence": decision_status == "insufficient_evidence",
        "uncertainty": {"top_probability": 0.62, "normalized_entropy": 0.8},
        "abstention_reasons": ["fixture_abstention"]
        if decision_status == "insufficient_evidence"
        else [],
    }


def run_evaluation(cases, retriever):
    status_counts = Counter()
    candidate_claims = 0
    claims_with_citations = 0
    total_citation_references = 0
    valid_citation_references = 0
    accepted_claims = 0
    expected_sources = 0
    retrieved_expected_sources = 0
    no_generation_expected = 0
    no_generation_observed = 0
    checks_passed = 0
    results = []

    for case in cases:
        retrieval = retriever.retrieve(
            case["condition"], context=case["context"], top_k=3, strategy="hybrid"
        )
        draft = case.get("draft")
        decision_status = case.get("decision_status", "ranked_prediction")
        should_skip_generation = (
            decision_status == "insufficient_evidence"
            or not retrieval.get("sufficient_evidence")
            or case.get("provider_unavailable", False)
        )
        if isinstance(draft, dict) and not should_skip_generation:
            claims = draft.get("claims", [])
            candidate_claims += len(claims)
            evidence_ids = {f"E{index}" for index in range(1, len(retrieval["passages"]) + 1)}
            for claim in claims:
                references = claim.get("evidence_ids", [])
                claims_with_citations += int(bool(references))
                total_citation_references += len(references)
                valid_citation_references += sum(item in evidence_ids for item in references)

        expected_document = case.get("expected_document_id")
        if expected_document:
            expected_sources += 1
            retrieved_expected_sources += int(
                any(
                    passage.get("chunk_id", "").startswith(expected_document + ":")
                    for passage in retrieval["passages"]
                )
            )

        if case.get("provider_unavailable"):
            generator = None
        else:
            generator = FixtureGenerator(draft)
        result = GroundedExplanationService(generator).explain(
            _prediction(case["condition"], decision_status), case["context"], retrieval
        )
        status_counts[result["grounding_status"]] += 1
        if result["grounding_status"] == "grounded":
            accepted_claims += len(result["explanation"]["claims"])

        no_generation_expected += int(should_skip_generation)
        no_generation_observed += int(
            should_skip_generation and (generator is None or generator.calls == 0)
        )
        passed = result["grounding_status"] == case["expected_status"]
        checks_passed += int(passed)
        results.append(
            {
                "id": case["id"],
                "status": result["grounding_status"],
                "expected_status": case["expected_status"],
                "status_check_passed": passed,
                "retrieved_expected_source": (
                    any(
                        passage.get("chunk_id", "").startswith(expected_document + ":")
                        for passage in retrieval["passages"]
                    )
                    if expected_document
                    else None
                ),
                "generator_calls": generator.calls if generator is not None else 0,
            }
        )

    claim_denominator = max(1, candidate_claims)
    citation_denominator = max(1, total_citation_references)
    return {
        "scope": "Structural/citation and evidence-gating evaluation with mocked LLM drafts; not clinical, classifier, or live LLM quality metrics.",
        "classifier_test_samples_used": False,
        "live_llm_invocations": 0,
        "manual_case_count": len(cases),
        "expected_source_coverage": {
            "retrieved": retrieved_expected_sources,
            "expected": expected_sources,
            "recall": round(retrieved_expected_sources / max(1, expected_sources), 4),
        },
        "citation_validity": {
            "valid_references": valid_citation_references,
            "all_references_in_fixture_drafts": total_citation_references,
            "rate": round(valid_citation_references / citation_denominator, 4),
        },
        "citation_completeness": {
            "claims_with_any_reference": claims_with_citations,
            "fixture_claims": candidate_claims,
            "rate": round(claims_with_citations / claim_denominator, 4),
        },
        "grounded_claim_rate": {
            "accepted_claims_with_known_citations": accepted_claims,
            "fixture_claims": candidate_claims,
            "rate": round(accepted_claims / claim_denominator, 4),
        },
        "structurally_unsupported_claim_rate": {
            "rejected_or_uncited_fixture_claims": candidate_claims - accepted_claims,
            "fixture_claims": candidate_claims,
            "rate": round((candidate_claims - accepted_claims) / claim_denominator, 4),
            "meaning": "Missing or unknown citations; not a determination of medical truth.",
        },
        "evidence_gate_behavior": {
            "cases_expected_to_skip_generation": no_generation_expected,
            "cases_that_skipped_generation": no_generation_observed,
            "correct": no_generation_expected == no_generation_observed,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "case_results": results,
        "status_checks_passed": checks_passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "grounded_explanation_evaluation.json",
    )
    args = parser.parse_args()
    cases = json.loads(
        (PROJECT_ROOT / "rag" / "grounded_evaluation_cases.json").read_text(encoding="utf-8")
    )["cases"]
    report = run_evaluation(cases, EvidenceRetriever(project_root=PROJECT_ROOT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status_checks_passed"] != report["manual_case_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
