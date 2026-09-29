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
    """Keep lexical retrieval local and add lazy semantic and hybrid strategies."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config_path: Optional[Path] = None,
        embedding_encoder: Optional[Any] = None,
    ):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        selected_config = Path(config_path) if config_path else self.project_root / "rag" / "config.json"
        with selected_config.open(encoding="utf-8") as config_file:
            self.config = json.load(config_file)

        documents_path = Path(self.config["documents_path"])
        if not documents_path.is_absolute():
            documents_path = self.project_root / documents_path
        self.documents = load_documents(documents_path)
        supported_conditions_path = Path(
            self.config.get("supported_conditions_path", "rag/supported_conditions.json")
        )
        if not supported_conditions_path.is_absolute():
            supported_conditions_path = self.project_root / supported_conditions_path
        supported_conditions = set(
            json.loads(supported_conditions_path.read_text(encoding="utf-8"))
        )
        corpus_conditions = {document["condition"] for document in self.documents}
        if corpus_conditions != supported_conditions:
            missing = sorted(supported_conditions - corpus_conditions)
            extra = sorted(corpus_conditions - supported_conditions)
            raise ValueError(
                f"Evidence coverage does not match the supported model classes; missing={missing}, extra={extra}."
            )
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
        self.embedding_encoder = embedding_encoder
        self._semantic_index = None

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

    def _retrieve_tfidf(
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
                "chunk_id": chunk["chunk_id"],
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

    def _get_semantic_index(self):
        if self._semantic_index is None:
            from rag.semantic import SemanticVectorIndex

            self._semantic_index = SemanticVectorIndex(
                self.chunks,
                self.config["semantic"],
                self.project_root,
                encoder=self.embedding_encoder,
            )
        return self._semantic_index

    def build_semantic_index(self, force: bool = False) -> Dict[str, Any]:
        """Build or validate the local dense vector matrix without serving a query."""
        return self._get_semantic_index().build(force=force)

    def _query_context(self, context: Optional[List[str]]) -> List[str]:
        return [
            re.sub(r"[_\s]+", " ", item).strip()
            for item in (context or [])
            if isinstance(item, str) and item.strip()
        ]

    def _semantic_rank(
        self,
        canonical: str,
        query_context: List[str],
    ) -> List[tuple[int, float]]:
        query = " ".join(query_context) or canonical.replace("_", " ")
        scores = self._get_semantic_index().search(query)
        threshold = float(self.config["semantic"]["min_similarity"])
        selected = [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk["condition"] == canonical and float(scores[index]) >= threshold
        ]
        return [
            (index, float(scores[index]))
            for index in sorted(selected, key=lambda item: (-float(scores[item]), item))
        ]

    def _hybrid_rank(
        self,
        canonical: str,
        query_context: List[str],
    ) -> List[tuple[int, float]]:
        lexical_query = " ".join(query_context) or canonical.replace("_", " ")
        lexical_vector = self.vectorizer.transform([lexical_query])
        lexical_scores = np.asarray((self.matrix @ lexical_vector.T).toarray()).ravel()
        selected = [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk["condition"] == canonical
            and float(lexical_scores[index]) >= self.min_relevance_score
        ]
        lexical_ranked = sorted(
            selected,
            key=lambda item: (-float(lexical_scores[item]), item),
        )
        semantic_ranked = self._semantic_rank(canonical, query_context)

        settings = self.config["hybrid"]
        lexical_weight = float(settings["tfidf_weight"])
        semantic_weight = float(settings["semantic_weight"])
        weight_sum = lexical_weight + semantic_weight
        if lexical_weight < 0 or semantic_weight < 0 or weight_sum <= 0:
            raise ValueError("Hybrid weights must be nonnegative and have a positive sum.")
        lexical_weight /= weight_sum
        semantic_weight /= weight_sum
        rrf_k = int(settings["rrf_k"])
        if rrf_k < 1:
            raise ValueError("Hybrid reciprocal-rank constant must be positive.")

        fused: Dict[int, float] = {}
        for weight, ranked in (
            (lexical_weight, lexical_ranked),
            (semantic_weight, [index for index, _ in semantic_ranked]),
        ):
            for rank, index in enumerate(ranked, start=1):
                fused[index] = fused.get(index, 0.0) + weight / (rrf_k + rank)
        return sorted(fused.items(), key=lambda item: (-item[1], item[0]))

    def retrieve(
        self,
        condition: str,
        context: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve evidence by the configured strategy without calling the classifier."""
        selected_strategy = strategy or self.config.get("default_strategy", "tfidf")
        method_names = {
            "tfidf": "tfidf_cosine",
            "semantic": "semantic_cosine",
            "hybrid": "hybrid_weighted_rrf",
        }
        if selected_strategy not in method_names:
            raise ValueError("strategy must be tfidf, semantic, or hybrid.")

        requested = self.top_k if top_k is None else int(top_k)
        if not 1 <= requested <= 5:
            raise ValueError("top_k must be between 1 and 5.")
        canonical = self._conditions.get(_normalize_condition(condition))
        if canonical is None:
            return {
                "condition": condition,
                "supported": False,
                "sufficient_evidence": False,
                "status": "unsupported_condition",
                "retrieval_method": method_names[selected_strategy],
                "passages": [],
                "disclaimer": DISCLAIMER,
            }

        query_context = self._query_context(context)
        if selected_strategy == "tfidf":
            result = self._retrieve_tfidf(canonical, context=query_context, top_k=requested)
            passages = result["passages"]
        else:
            ranked = (
                self._semantic_rank(canonical, query_context)
                if selected_strategy == "semantic"
                else self._hybrid_rank(canonical, query_context)
            )
            passages = [
                {
                    "chunk_id": self.chunks[index]["chunk_id"],
                    "text": self.chunks[index]["text"],
                    "section": self.chunks[index]["section"],
                    "source": dict(self.chunks[index]["source"]),
                    "relevance_score": score,
                }
                for index, score in ranked[:requested]
            ]

        sufficient = bool(passages)
        return {
            "condition": canonical,
            "supported": True,
            "sufficient_evidence": sufficient,
            "status": "evidence_available" if sufficient else "no_sufficient_evidence",
            "retrieval_method": method_names[selected_strategy],
            "passages": passages,
            "disclaimer": DISCLAIMER,
        }
