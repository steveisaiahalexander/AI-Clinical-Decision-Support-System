import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from api.main import retrieve_evidence
from api.schemas import EvidenceRequest, EvidenceResponse
from rag.ingestion import split_text
from rag.retrieval import EvidenceRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EvidenceRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = EvidenceRetriever(project_root=PROJECT_ROOT)

    def test_curated_documents_and_sparse_index_build(self):
        self.assertEqual(self.retriever.stats["documents"], 12)
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

    def test_unknown_condition_returns_no_unattributed_fallback(self):
        result = self.retriever.retrieve("Uncurated Condition", context=["fever"])
        self.assertFalse(result["supported"])
        self.assertEqual(result["passages"], [])

    def test_fastapi_evidence_route_keeps_retrieval_separate(self):
        retriever = Mock()
        retriever.retrieve.return_value = self.retriever.retrieve(
            "Asthma",
            context=["wheezing"],
        )
        with patch("api.main.get_evidence_retriever", return_value=retriever):
            response = retrieve_evidence(
                EvidenceRequest(condition="Asthma", context=["wheezing"])
            )
        self.assertIsInstance(response, EvidenceResponse)
        self.assertTrue(response.supported)
        self.assertEqual(response.retrieval_method, "tfidf_cosine")
        retriever.retrieve.assert_called_once_with(
            "Asthma",
            context=["wheezing"],
            top_k=None,
        )


if __name__ == "__main__":
    unittest.main()
