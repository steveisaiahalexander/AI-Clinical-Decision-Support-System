"""Deterministic, local TF-IDF vector retrieval for curated source passages."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from rag.ingestion import chunk_documents, load_documents


DISCLAIMER = (
    "Retrieved passages are contextual information only. They do not provide a "
    "diagnosis or treatment recommendation, and source inclusion does not imply "
    "CDC, NIH, or NIDDK endorsement."
)


def _normalize_condition(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _search_text(chunk: Dict[str, Any]) -> str:
    condition = chunk["condition"].replace("_", " ")
    source = chunk["source"]
    return " ".join(
        (condition, source["title"], chunk["section"], chunk["text"])
    )


class EvidenceRetriever:
    """Build an in-memory sparse-vector index from the curated JSON documents."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        selected_config = Path(config_path) if config_path else self.project_root / "rag" / "config.json"
        with selected_config.open(encoding="utf-8") as config_file:
            self.config = json.load(config_file)

        documents_path = Path(self.config["documents_path"])
        if not documents_path.is_absolute():
            documents_path = self.project_root / documents_path
        self.documents = load_documents(documents_path)
        self.chunks = chunk_documents(
            self.documents,
            max_chars=int(self.config["max_chunk_chars"]),
            overlap_sentences=int(self.config["overlap_sentences"]),
        )
        if not self.chunks:
            raise ValueError("The evidence corpus did not produce any text chunks.")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            norm="l2",
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.matrix = self.vectorizer.fit_transform(
            [_search_text(chunk) for chunk in self.chunks]
        )
        self._conditions: Dict[str, str] = {}
        for document in self.documents:
            canonical = document["condition"]
            for alias in [canonical, *document.get("aliases", [])]:
                normalized = _normalize_condition(alias)
                previous = self._conditions.get(normalized)
                if previous is not None and previous != canonical:
                    raise ValueError(f"Evidence condition alias is ambiguous: {alias}.")
                self._conditions[normalized] = canonical

        self.top_k = int(self.config["top_k"])
        self.max_chunk_chars = int(self.config["max_chunk_chars"])
        self.min_relevance_score = float(self.config["min_relevance_score"])
        if not 1 <= self.top_k <= 5:
            raise ValueError("Configured evidence top_k must be between 1 and 5.")
        if not 0 <= self.min_relevance_score <= 1:
            raise ValueError("Minimum relevance score must be between zero and one.")

    @property
    def stats(self) -> Dict[str, int]:
        """Return reproducible index-build counts for tests and the build script."""
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "features": len(self.vectorizer.vocabulary_),
        }

    def retrieve(
        self,
        condition: str,
        context: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Rank passages only within the requested model class."""
        canonical = self._conditions.get(_normalize_condition(condition))
        requested = self.top_k if top_k is None else int(top_k)
        if not 1 <= requested <= 5:
            raise ValueError("top_k must be between 1 and 5.")
        if canonical is None:
            return {
                "condition": condition,
                "supported": False,
                "retrieval_method": "tfidf_cosine",
                "passages": [],
                "disclaimer": DISCLAIMER,
            }

        selected = [
            (index, chunk)
            for index, chunk in enumerate(self.chunks)
            if chunk["condition"] == canonical
        ]
        query_context = [
            re.sub(r"[_\s]+", " ", item).strip()
            for item in (context or [])
            if isinstance(item, str) and item.strip()
        ]
        query = " ".join([canonical.replace("_", " "), *query_context])
        query_vector = self.vectorizer.transform([query])
        scores = np.asarray((self.matrix @ query_vector.T).toarray()).ravel()
        ranked = sorted(
            selected,
            key=lambda item: (-float(scores[item[0]]), item[0]),
        )
        ranked = [
            (index, chunk)
            for index, chunk in ranked
            if float(scores[index]) >= self.min_relevance_score
        ]
        if not ranked:
            query_vector = self.vectorizer.transform([canonical.replace("_", " ")])
            scores = np.asarray((self.matrix @ query_vector.T).toarray()).ravel()
            ranked = sorted(
                selected,
                key=lambda item: (-float(scores[item[0]]), item[0]),
            )
            ranked = [
                (index, chunk)
                for index, chunk in ranked
                if float(scores[index]) >= self.min_relevance_score
            ]

        passages = [
            {
                "text": chunk["text"],
                "section": chunk["section"],
                "source": dict(chunk["source"]),
                "relevance_score": float(scores[index]),
            }
            for index, chunk in ranked[:requested]
        ]
        return {
            "condition": canonical,
            "supported": True,
            "retrieval_method": "tfidf_cosine",
            "passages": passages,
            "disclaimer": DISCLAIMER,
        }
