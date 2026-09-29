"""Compare TF-IDF, local semantic, and hybrid evidence retrieval."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieval import EvidenceRetriever


STRATEGIES = ("tfidf", "semantic", "hybrid")


def _matches(passage: Dict[str, Any], query: Dict[str, Any]) -> bool:
    document_id = query.get("relevant_document_id")
    if not document_id or not passage.get("chunk_id", "").startswith(document_id + ":"):
        return False
    return passage.get("section") in query.get("relevant_sections", [])


def evaluate_strategy(
    retriever: EvidenceRetriever,
    queries: List[Dict[str, Any]],
    strategy: str,
) -> Dict[str, Any]:
    positive_queries = [item for item in queries if item.get("relevant_document_id")]
    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = 0.0
    no_relevant = 0
    insufficient = 0
    expected_sources = {item["relevant_document_id"] for item in positive_queries}
    covered_sources = set()

    for query in queries:
        result = retriever.retrieve(
            query["condition"],
            context=query["context"],
            top_k=3,
            strategy=strategy,
        )
        passages = result["passages"][:3]
        insufficient += int(not result["sufficient_evidence"])
        relevant_ranks = [
            rank
            for rank, passage in enumerate(passages, start=1)
            if _matches(passage, query)
        ]
        if not relevant_ranks:
            no_relevant += 1
            continue

        first_rank = relevant_ranks[0]
        hits_at_1 += int(first_rank == 1)
        hits_at_3 += 1
        reciprocal_ranks += 1.0 / first_rank
        covered_sources.add(query["relevant_document_id"])

    denominator = max(1, len(positive_queries))
    return {
        "query_count": len(queries),
        "queries_with_relevant_targets": len(positive_queries),
        "recall_at_1": round(hits_at_1 / denominator, 4),
        "recall_at_3": round(hits_at_3 / denominator, 4),
        "mrr_at_3": round(reciprocal_ranks / denominator, 4),
        "queries_with_no_relevant_evidence": no_relevant,
        "queries_without_sufficient_evidence": insufficient,
        "source_coverage": {
            "sources_retrieved": len(covered_sources),
            "sources_expected": len(expected_sources),
            "recall": round(len(covered_sources) / max(1, len(expected_sources)), 4),
        },
    }


def run_evaluation(retriever: EvidenceRetriever, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "scope": "retrieval metrics only; not clinical or classifier metrics",
        "query_source": "manually curated rag/evaluation_queries.json",
        "classifier_test_samples_used": False,
        "strategies": {
            strategy: evaluate_strategy(retriever, queries, strategy)
            for strategy in STRATEGIES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "evidence_retrieval_evaluation.json",
    )
    args = parser.parse_args()
    query_path = PROJECT_ROOT / "rag" / "evaluation_queries.json"
    queries = json.loads(query_path.read_text(encoding="utf-8"))["queries"]
    retriever = EvidenceRetriever(project_root=PROJECT_ROOT)
    report = run_evaluation(retriever, queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
