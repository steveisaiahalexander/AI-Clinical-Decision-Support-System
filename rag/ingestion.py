"""Load curated source records and split their text into stable retrieval chunks."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List


class CorpusError(ValueError):
    """Raised when curated source documents are malformed or incomplete."""


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_long_sentence(sentence: str, max_chars: int) -> List[str]:
    words = sentence.split()
    pieces: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) > max_chars and current:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    if any(len(piece) > max_chars for piece in pieces):
        raise CorpusError("A single token exceeds the configured chunk size.")
    return pieces


def split_text(text: str, max_chars: int = 700, overlap_sentences: int = 1) -> List[str]:
    """Split text on sentence boundaries, keeping bounded sentence overlap."""
    if max_chars < 1 or overlap_sentences < 0:
        raise ValueError("Chunk size must be positive and overlap cannot be negative.")
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences: List[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(normalized):
        if len(sentence) <= max_chars:
            sentences.append(sentence)
        else:
            sentences.extend(_split_long_sentence(sentence, max_chars))

    chunks: List[str] = []
    current: List[str] = []
    for sentence in sentences:
        candidate = " ".join([*current, sentence])
        if len(candidate) > max_chars and current:
            chunks.append(" ".join(current))
            overlap = current[-overlap_sentences:] if overlap_sentences else []
            current = [*overlap, sentence]
            while len(" ".join(current)) > max_chars and overlap:
                overlap = overlap[1:]
                current = [*overlap, sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_documents(documents_path: Path) -> List[Dict[str, Any]]:
    """Read one JSON source record per file, in stable path order."""
    path = Path(documents_path)
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not paths or any(not item.is_file() for item in paths):
        raise CorpusError(f"No evidence documents found at {path}.")

    documents: List[Dict[str, Any]] = []
    seen_ids = set()
    for item in paths:
        with item.open(encoding="utf-8") as source_file:
            document = json.load(source_file)
        source = document.get("source", {})
        if not document.get("id") or not document.get("condition"):
            raise CorpusError(f"Document {item.name} is missing its id or condition.")
        required_source_fields = (
            "title",
            "organization",
            "url",
            "attribution",
            "license",
            "rights_url",
            "accessed_on",
        )
        if not all(source.get(key) for key in required_source_fields):
            raise CorpusError(f"Document {item.name} is missing source metadata.")
        if not isinstance(document.get("passages"), list) or not document["passages"]:
            raise CorpusError(f"Document {item.name} has no passages.")
        if document["id"] in seen_ids:
            raise CorpusError(f"Duplicate document id: {document['id']}.")
        seen_ids.add(document["id"])
        for passage in document["passages"]:
            if not passage.get("section") or not passage.get("text", "").strip():
                raise CorpusError(f"Document {item.name} contains an empty passage.")
        documents.append(document)
    return documents


def chunk_documents(
    documents: List[Dict[str, Any]],
    max_chars: int = 700,
    overlap_sentences: int = 1,
) -> List[Dict[str, Any]]:
    """Create chunks while retaining complete condition and source attribution."""
    chunks: List[Dict[str, Any]] = []
    for document in documents:
        for passage_number, passage in enumerate(document["passages"], start=1):
            for chunk_number, text in enumerate(
                split_text(passage["text"], max_chars, overlap_sentences),
                start=1,
            ):
                chunks.append(
                    {
                        "chunk_id": f"{document['id']}:p{passage_number}:c{chunk_number}",
                        "condition": document["condition"],
                        "aliases": document.get("aliases", []),
                        "section": passage["section"],
                        "text": text,
                        "source": dict(document["source"]),
                    }
                )
    return chunks
