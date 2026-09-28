import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from LLM import hic


class KnowledgeBaseTests(unittest.TestCase):
    def test_rejects_empty_knowledge_base_before_loading_model(self):
        with patch.object(hic, "get_embedder") as get_embedder:
            with self.assertRaisesRegex(ValueError, "empty knowledge base"):
                hic.select_similar_case({"event": "query"}, [])
        get_embedder.assert_not_called()

    def test_builds_knowledge_base_with_persisted_bge_embeddings(self):
        sample = {"series": [1.0], "news": "test event", "truth": [2.0]}
        summary = {"event": "test event"}
        embedding = [1.0] + [0.0] * 383

        class FakeEmbedder:
            def encode(self, texts, is_query=False):
                self.is_query = is_query
                return [embedding for _ in texts]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "new-directory" / "kb.json"
            with (
                patch.object(hic, "kb_file", output_path),
                patch.object(hic, "get_embedder", return_value=FakeEmbedder(), create=True),
                patch.object(hic, "summarize", return_value=summary),
                patch.object(hic, "reason", return_value={
                    "prediction": [1.5], "reasoning": "initial"
                }),
                patch.object(hic, "correct", return_value={
                    "Key Insights": "Use cautious adjustment."
                }),
            ):
                knowledge_base = hic.build_knowledge_base([sample], {}, append=False)

            self.assertTrue(output_path.exists())
            self.assertEqual(knowledge_base, [{
                "summary": summary,
                "correction": "Use cautious adjustment.",
                "embedding": embedding,
                "embedding_model": "BAAI/bge-small-en-v1.5",
            }])
            self.assertEqual(json.loads(output_path.read_text()), knowledge_base)


class LocalEmbeddingRetrievalTests(unittest.TestCase):
    @staticmethod
    def _unit_vector(index):
        vector = [0.0] * 384
        vector[index] = 1.0
        return vector

    def test_serializes_summary_deterministically(self):
        summary = {"z": 1, "a": {"b": "value"}}
        self.assertEqual(
            hic.summary_to_embedding_text(summary),
            hic.summary_to_embedding_text({"a": {"b": "value"}, "z": 1}),
        )

    def test_selects_candidate_with_highest_cosine_similarity(self):
        candidates = [
            {
                "summary": {"event": "unrelated"}, "correction": "first",
                "embedding": self._unit_vector(1),
                "embedding_model": "BAAI/bge-small-en-v1.5",
            },
            {
                "summary": {"event": "related"}, "correction": "second",
                "embedding": self._unit_vector(0),
                "embedding_model": "BAAI/bge-small-en-v1.5",
            },
        ]

        class FakeEmbedder:
            def encode(self, texts, is_query=False):
                self.is_query = is_query
                return [LocalEmbeddingRetrievalTests._unit_vector(0)]

        with (
            patch.object(hic, "get_embedder", return_value=FakeEmbedder(), create=True),
            patch.object(hic, "call_llm", side_effect=AssertionError("LLM used for retrieval")) as call_llm,
        ):
            selected, metadata = hic.select_similar_case({"event": "related"}, candidates)

        self.assertIs(selected, candidates[1])
        self.assertEqual(metadata["method"], "bge_small_cosine")
        self.assertAlmostEqual(metadata["similarity"], 1.0)
        self.assertNotIn("match_reason", metadata)
        call_llm.assert_not_called()

    def test_builds_embeddings_for_legacy_knowledge_base_entries(self):
        candidates = [
            {"summary": {"event": "one"}, "correction": "lesson one"},
            {"summary": {"event": "two"}, "correction": "lesson two"},
        ]

        class FakeEmbedder:
            def encode(self, texts, is_query=False):
                if is_query:
                    return [LocalEmbeddingRetrievalTests._unit_vector(0)]
                return [
                    LocalEmbeddingRetrievalTests._unit_vector(1),
                    LocalEmbeddingRetrievalTests._unit_vector(0),
                ]

        with (
            patch.object(hic, "get_embedder", return_value=FakeEmbedder(), create=True),
            patch.object(hic, "call_llm", side_effect=AssertionError("LLM used for retrieval")) as call_llm,
        ):
            selected, metadata = hic.select_similar_case({"event": "two"}, candidates)

        self.assertIs(selected, candidates[1])
        self.assertEqual(metadata["selected_index"], 1)
        self.assertEqual(candidates[1]["embedding_model"], "BAAI/bge-small-en-v1.5")
        call_llm.assert_not_called()

    def test_rejects_stored_vector_with_wrong_dimension(self):
        candidate = {
            "summary": {"event": "case"},
            "correction": "lesson",
            "embedding": [1.0, 0.0],
            "embedding_model": "BAAI/bge-small-en-v1.5",
        }

        class FakeEmbedder:
            def encode(self, texts, is_query=False):
                return [LocalEmbeddingRetrievalTests._unit_vector(0)]

        with (
            patch.object(hic, "get_embedder", return_value=FakeEmbedder(), create=True),
        ):
            with self.assertRaisesRegex(ValueError, "384 dimensions"):
                hic.select_similar_case({"event": "query"}, [candidate])


if __name__ == "__main__":
    unittest.main()
