import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from api.main import retrieve_evidence
from api.schemas import EvidenceRequest, EvidenceResponse
from rag.ingestion import split_text
from rag.retrieval import EvidenceRetriever
from scripts.evaluate_evidence_retrieval import run_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConceptEncoder:
    """Deterministic fixture mapping a synonym to the source concept vector."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            normalized = text.casefold()
            if "breathlessness" in normalized or "shortness of breath" in normalized:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class EvidenceRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = EvidenceRetriever(project_root=PROJECT_ROOT)

    def test_curated_documents_and_sparse_index_build(self):
        self.assertEqual(self.retriever.stats["documents"], 30)
        self.assertGreaterEqual(self.retriever.stats["chunks"], 30)
        self.assertGreater(self.retriever.stats["features"], 0)
        self.assertEqual(
            self.retriever.matrix.shape,
            (self.retriever.stats["chunks"], self.retriever.stats["features"]),
        )
        self.assertLessEqual(
            max(len(chunk["text"]) for chunk in self.retriever.chunks),
            self.retriever.max_chunk_chars,
        )
        expected = set(json.loads((PROJECT_ROOT / "rag/supported_conditions.json").read_text(encoding="utf-8")))
        self.assertEqual({document["condition"] for document in self.retriever.documents}, expected)
        self.assertEqual(len(expected), 30)

    def test_sentence_chunking_has_bounded_size_and_one_sentence_overlap(self):
        chunks = split_text(
            "First sentence. Second sentence. Third sentence. Fourth sentence.",
            max_chars=34,
            overlap_sentences=1,
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 34 for chunk in chunks))
        self.assertTrue(chunks[0].endswith("Second sentence."))
        self.assertTrue(chunks[1].startswith("Second sentence."))

    def test_source_metadata_survives_ingestion_and_retrieval(self):
        result = self.retriever.retrieve(
            "Tuberculosis",
            context=["cough", "night sweats", "weight loss"],
        )
        self.assertTrue(result["supported"])
        self.assertTrue(result["passages"])
        source = result["passages"][0]["source"]
        self.assertEqual(source["organization"], "Centers for Disease Control and Prevention")
        self.assertEqual(source["url"], "https://www.cdc.gov/tb/signs-symptoms/index.html")
        self.assertEqual(source["accessed_on"], "2026-09-30")
        self.assertTrue(source["attribution"])
        self.assertTrue(source["license"])
        self.assertTrue(result["passages"][0]["section"])
        self.assertTrue(result["sufficient_evidence"])
        self.assertEqual(result["status"], "evidence_available")

    def test_complete_provenance_is_preserved_for_every_chunk(self):
        required = {
            "title", "organization", "url", "attribution", "license",
            "rights_url", "accessed_on",
        }
        for chunk in self.retriever.chunks:
            with self.subTest(chunk=chunk["chunk_id"]):
                self.assertTrue(required.issubset(chunk["source"]))
                self.assertTrue(chunk["section"])

    def test_representative_supported_queries_return_relevant_passages(self):
        cases = [
            ("Asthma", ["wheezing", "chest tightness"], ["wheezing", "chest tightness"]),
            ("COPD", ["phlegm", "shortness of breath"], ["phlegm", "shortness of breath"]),
            ("Common_Cold", ["runny nose", "sore throat"], ["runny nose", "sore throat"]),
            ("Flu", ["fever", "body aches"], ["fever", "body aches"]),
            ("COVID19", ["loss of taste", "fatigue"], ["loss of taste", "fatigue"]),
            ("Tuberculosis", ["night sweats", "weight loss"], ["night sweats", "weight loss"]),
            ("Pneumonia", ["fever", "shortness of breath"], ["fever", "shortness of breath"]),
            ("Stroke", ["sudden weakness", "speech"], ["sudden weakness", "speech"]),
            ("Type2_Diabetes", ["thirst", "urination", "fatigue"], ["thirst", "urination"]),
            ("Chronic_Kidney_Disease", ["foamy urine", "swelling"], ["foamy urine", "swelling"]),
            ("Hypertension", ["high blood pressure", "headache"], ["hypertension", "headaches"]),
            ("Migraine", ["nausea", "sensitivity to light"], ["nausea", "light"]),
        ]
        for condition, context, expected_terms in cases:
            with self.subTest(condition=condition):
                result = self.retriever.retrieve(condition, context=context)
                self.assertTrue(result["supported"])
                self.assertGreater(len(result["passages"]), 0)
                self.assertLessEqual(len(result["passages"]), 3)
                self.assertGreater(result["passages"][0]["relevance_score"], 0.02)
                returned_text = " ".join(passage["text"] for passage in result["passages"]).casefold()
                self.assertTrue(
                    any(term.casefold() in returned_text for term in expected_terms),
                    returned_text,
                )

    def test_all_manually_curated_supported_queries_return_the_expected_source_section(self):
        query_file = PROJECT_ROOT / "rag" / "evaluation_queries.json"
        queries = json.loads(query_file.read_text(encoding="utf-8"))["queries"]
        for query in queries:
            if not query["relevant_document_id"]:
                continue
            with self.subTest(condition=query["condition"]):
                result = self.retriever.retrieve(
                    query["condition"], context=query["context"], top_k=3, strategy="tfidf"
                )
                self.assertTrue(result["sufficient_evidence"])
                expected_prefix = query["relevant_document_id"] + ":"
                self.assertTrue(
                    any(
                        passage["chunk_id"].startswith(expected_prefix)
                        and passage["section"] in query["relevant_sections"]
                        for passage in result["passages"]
                    )
                )

    def test_unknown_condition_returns_no_unattributed_fallback(self):
        result = self.retriever.retrieve("Uncurated Condition", context=["fever"])
        self.assertFalse(result["supported"])
        self.assertFalse(result["sufficient_evidence"])
        self.assertEqual(result["status"], "unsupported_condition")
        self.assertEqual(result["passages"], [])

    def test_semantic_retrieval_builds_local_index_and_matches_a_synonym(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = copy.deepcopy(self.retriever.config)
            config["semantic"]["index_path"] = str(Path(temporary) / "vectors.npz")
            config["semantic"]["model_cache_dir"] = str(Path(temporary) / "model-cache")
            config_path = Path(temporary) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            retriever = EvidenceRetriever(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                embedding_encoder=ConceptEncoder(),
            )
            result = retriever.retrieve(
                "Coronary_Artery_Disease",
                context=["breathlessness"],
                strategy="semantic",
            )
            self.assertEqual(result["retrieval_method"], "semantic_cosine")
            self.assertTrue(result["sufficient_evidence"])
            self.assertIn("shortness of breath", result["passages"][0]["text"])
            self.assertTrue((Path(temporary) / "vectors.npz").is_file())
            self.assertTrue((Path(temporary) / "vectors.json").is_file())

    def test_hybrid_retrieval_fuses_ranked_candidates_without_scores_in_ui_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = copy.deepcopy(self.retriever.config)
            config["semantic"]["index_path"] = str(Path(temporary) / "vectors.npz")
            config["semantic"]["model_cache_dir"] = str(Path(temporary) / "model-cache")
            config_path = Path(temporary) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            retriever = EvidenceRetriever(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                embedding_encoder=ConceptEncoder(),
            )
            result = retriever.retrieve(
                "Coronary_Artery_Disease",
                context=["breathlessness"],
                strategy="hybrid",
            )
            self.assertEqual(result["retrieval_method"], "hybrid_weighted_rrf")
            self.assertTrue(result["passages"])
            self.assertTrue(result["sufficient_evidence"])

    def test_semantic_retrieval_returns_explicit_no_sufficient_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = copy.deepcopy(self.retriever.config)
            config["semantic"]["index_path"] = str(Path(temporary) / "vectors.npz")
            config["semantic"]["model_cache_dir"] = str(Path(temporary) / "model-cache")
            config_path = Path(temporary) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            retriever = EvidenceRetriever(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                embedding_encoder=ConceptEncoder(),
            )
            result = retriever.retrieve(
                "Coronary_Artery_Disease",
                context=["unrelated phrase"],
                strategy="semantic",
            )
            self.assertTrue(result["supported"])
            self.assertFalse(result["sufficient_evidence"])
            self.assertEqual(result["status"], "no_sufficient_evidence")
            self.assertEqual(result["passages"], [])

    def test_small_manual_retrieval_evaluation_reports_only_retrieval_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = copy.deepcopy(self.retriever.config)
            config["semantic"]["index_path"] = str(Path(temporary) / "vectors.npz")
            config["semantic"]["model_cache_dir"] = str(Path(temporary) / "model-cache")
            config_path = Path(temporary) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            retriever = EvidenceRetriever(
                project_root=PROJECT_ROOT,
                config_path=config_path,
                embedding_encoder=ConceptEncoder(),
            )
            queries = [
                {
                    "condition": "Coronary_Artery_Disease",
                    "context": ["breathlessness"],
                    "relevant_document_id": "nhlbi-coronary-heart-disease-symptoms",
                    "relevant_sections": ["Symptoms"],
                },
                {
                    "condition": "Uncurated_Condition",
                    "context": ["unrelated phrase"],
                    "relevant_document_id": None,
                    "relevant_sections": [],
                },
            ]
            report = run_evaluation(retriever, queries)
            self.assertFalse(report["classifier_test_samples_used"])
            self.assertIn("recall_at_1", report["strategies"]["hybrid"])
            self.assertEqual(
                report["strategies"]["tfidf"]["queries_with_no_relevant_evidence"],
                1,
            )

    def test_fastapi_evidence_route_keeps_retrieval_separate(self):
        retriever = Mock()
        response_result = self.retriever.retrieve(
            "Asthma",
            context=["wheezing"],
        )
        response_result["retrieval_method"] = "semantic_cosine"
        retriever.retrieve.return_value = response_result
        with patch("api.main.get_evidence_retriever", return_value=retriever):
            response = retrieve_evidence(
                EvidenceRequest(condition="Asthma", context=["wheezing"], strategy="semantic")
            )
        self.assertIsInstance(response, EvidenceResponse)
        self.assertTrue(response.supported)
        self.assertEqual(response.retrieval_method, "semantic_cosine")
        retriever.retrieve.assert_called_once_with(
            "Asthma",
            context=["wheezing"],
            top_k=None,
            strategy="semantic",
        )

    def test_fastapi_evidence_route_keeps_legacy_request_default(self):
        retriever = Mock()
        retriever.retrieve.return_value = self.retriever.retrieve(
            "Asthma", context=["wheezing"]
        )
        with patch("api.main.get_evidence_retriever", return_value=retriever):
            response = retrieve_evidence(
                EvidenceRequest(condition="Asthma", context=["wheezing"])
            )
        self.assertEqual(response.retrieval_method, "tfidf_cosine")
        retriever.retrieve.assert_called_once_with(
            "Asthma",
            context=["wheezing"],
            top_k=None,
            strategy=None,
        )

    def test_api_returns_no_sufficient_evidence_state_without_passages(self):
        retriever = Mock()
        retriever.retrieve.return_value = {
            "condition": "Coronary_Artery_Disease",
            "supported": True,
            "sufficient_evidence": False,
            "status": "no_sufficient_evidence",
            "retrieval_method": "semantic_cosine",
            "passages": [],
            "disclaimer": self.retriever.retrieve("Asthma")["disclaimer"],
        }
        with patch("api.main.get_evidence_retriever", return_value=retriever):
            response = retrieve_evidence(
                EvidenceRequest(
                    condition="Coronary_Artery_Disease",
                    context=["unrelated phrase"],
                    strategy="semantic",
                )
            )
        self.assertFalse(response.sufficient_evidence)
        self.assertEqual(response.status, "no_sufficient_evidence")
        self.assertEqual(response.passages, [])


if __name__ == "__main__":
    unittest.main()
