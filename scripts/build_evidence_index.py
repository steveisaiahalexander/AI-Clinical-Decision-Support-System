"""Validate and report the local curated evidence index."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieval import EvidenceRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Build the optional local semantic embedding index too.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode chunks instead of reusing a valid semantic index.",
    )
    args = parser.parse_args()
    retriever = EvidenceRetriever(project_root=PROJECT_ROOT)
    semantic_index = (
        retriever.build_semantic_index(force=args.force) if args.semantic else None
    )
    examples = {}
    for condition, terms in (
        ("Asthma", ["wheezing", "chest tightness"]),
        ("Type2_Diabetes", ["thirst", "urination"]),
        ("Tuberculosis", ["cough", "night sweats"]),
        ("Migraine", ["headache", "light sensitivity"]),
        ("Pneumonia", ["fever", "shortness of breath"]),
    ):
        result = retriever.retrieve(condition, context=terms)
        examples[condition] = [
            {
                "section": passage["section"],
                "source": passage["source"]["title"],
                "score": round(passage["relevance_score"], 3),
            }
            for passage in result["passages"]
        ]
    print(
        json.dumps(
            {
                "index": retriever.stats,
                "embedding": "scikit-learn TF-IDF word unigram/bigram vectors",
                "semantic_index": semantic_index,
                "similarity": "cosine",
                "top_k": retriever.top_k,
                "examples": examples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
