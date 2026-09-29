"""Local sentence-embedding index for source-attributed evidence chunks."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class SemanticEmbeddingError(RuntimeError):
    """Raised when the configured local embedding runtime is unavailable."""


def _corpus_digest(chunks: List[Dict[str, Any]]) -> str:
    payload = [
        {"chunk_id": item["chunk_id"], "text": item["text"]}
        for item in chunks
    ]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class SentenceTransformerEncoder:
    """Thin CPU-capable adapter around the configured Sentence Transformers model."""

    def __init__(self, config: Dict[str, Any], project_root: Path):
        if config.get("use_system_cert_store", False):
            try:
                import truststore
            except ImportError as error:
                raise SemanticEmbeddingError(
                    "System certificate-store support requires the optional truststore dependency."
                ) from error
            truststore.inject_into_ssl()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise SemanticEmbeddingError(
                "Semantic retrieval requires the pinned sentence-transformers dependency."
            ) from error

        model_cache = Path(config["model_cache_dir"])
        if not model_cache.is_absolute():
            model_cache = project_root / model_cache
        try:
            self.model = SentenceTransformer(
                config["model_name"],
                revision=config["model_revision"],
                device=config.get("device", "cpu"),
                cache_folder=str(model_cache),
            )
        except Exception as error:
            raise SemanticEmbeddingError(
                "The configured local embedding model could not be loaded."
            ) from error
        self.batch_size = int(config.get("batch_size", 32))

    def encode(self, texts: List[str]) -> np.ndarray:
        try:
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise SemanticEmbeddingError("The local embedding model failed to encode text.") from error
        return np.asarray(vectors, dtype=np.float32)


class SemanticVectorIndex:
    """Exact cosine search over a compressed, reproducible NumPy vector matrix."""

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        config: Dict[str, Any],
        project_root: Path,
        encoder: Optional[Any] = None,
    ):
        self.chunks = chunks
        self.config = config
        self.project_root = Path(project_root)
        self.encoder = encoder
        index_path = Path(config["index_path"])
        if not index_path.is_absolute():
            index_path = self.project_root / index_path
        self.index_path = index_path
        self.manifest_path = index_path.with_suffix(".json")
        self._embeddings: Optional[np.ndarray] = None

    @property
    def model_identity(self) -> Dict[str, str]:
        return {
            "name": self.config["model_name"],
            "revision": self.config["model_revision"],
        }

    def _get_encoder(self):
        if self.encoder is None:
            self.encoder = SentenceTransformerEncoder(self.config, self.project_root)
        return self.encoder

    def _manifest(self) -> Dict[str, Any]:
        return {
            "format_version": 1,
            "model": self.model_identity,
            "corpus_sha256": _corpus_digest(self.chunks),
            "chunk_ids": [item["chunk_id"] for item in self.chunks],
        }

    def _load_existing(self, expected: Dict[str, Any]) -> bool:
        if not self.index_path.is_file() or not self.manifest_path.is_file():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest != expected:
                return False
            with np.load(self.index_path, allow_pickle=False) as archive:
                embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.chunks):
            return False
        if not np.isfinite(embeddings).all():
            return False
        self._embeddings = embeddings
        return True

    def build(self, force: bool = False) -> Dict[str, Any]:
        """Load a valid local matrix or encode and atomically replace the index."""
        manifest = self._manifest()
        if not force and self._load_existing(manifest):
            return self.stats

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        texts = [
            " ".join(
                (
                    item["condition"].replace("_", " "),
                    item["source"]["title"],
                    item["section"],
                    item["text"],
                )
            )
            for item in self.chunks
        ]
        embeddings = np.asarray(self._get_encoder().encode(texts), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.chunks):
            raise SemanticEmbeddingError("The model returned an invalid embedding matrix.")
        if not np.isfinite(embeddings).all():
            raise SemanticEmbeddingError("The model returned non-finite embeddings.")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise SemanticEmbeddingError("The model returned a zero-length embedding.")
        embeddings /= norms

        vector_temp = self.index_path.with_name(self.index_path.name + ".tmp")
        manifest_temp = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        with vector_temp.open("wb") as vector_file:
            np.savez_compressed(vector_file, embeddings=embeddings)
        manifest_temp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(vector_temp, self.index_path)
        os.replace(manifest_temp, self.manifest_path)
        self._embeddings = embeddings
        return self.stats

    @property
    def stats(self) -> Dict[str, Any]:
        if self._embeddings is None:
            raise RuntimeError("The semantic index has not been built.")
        return {
            "chunks": int(self._embeddings.shape[0]),
            "dimensions": int(self._embeddings.shape[1]),
            "model": self.model_identity,
            "index_path": str(self.index_path),
        }

    def search(self, query: str) -> np.ndarray:
        if self._embeddings is None:
            self.build()
        query_vector = np.asarray(self._get_encoder().encode([query]), dtype=np.float32)
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise SemanticEmbeddingError("The model returned an invalid query embedding.")
        if query_vector.shape[1] != self._embeddings.shape[1]:
            raise SemanticEmbeddingError("Query and corpus embedding dimensions do not match.")
        norm = float(np.linalg.norm(query_vector[0]))
        if not np.isfinite(norm) or norm == 0:
            raise SemanticEmbeddingError("The model returned an invalid query vector.")
        return self._embeddings @ (query_vector[0] / norm)
