"""
Integration tests for end-to-end agent pipelines.

These tests verify full pipeline connectivity with mocked external services
(LLM API, web search) but real internal processing (chunking, embedding,
retrieval).
"""

import pytest
from unittest.mock import patch, MagicMock


class TestWebSearchIntegration:
    """End-to-end tests for the web search agent."""

    def test_full_pipeline_with_mocks(self):
        """Full search pipeline should work with mocked externals."""
        from agent.web_search_agent import WebSearchAgent
        from agent.tools.search_tool import SearchResult

        mock_llm = MagicMock()
        mock_llm.generate_json.side_effect = [
            # Think phase
            {
                "thought": "Need to search for info",
                "search_queries": ["test query"],
            },
            # Synthesis phase
            {
                "answer": "Based on search results, the answer is X.",
                "confidence": "high",
                "key_facts": ["Fact 1"],
            },
        ]

        mock_search = MagicMock()
        mock_search.search_and_extract.return_value = [
            SearchResult(
                title="Result 1",
                url="https://example.com/1",
                snippet="First result snippet",
                body="Full first result content.",
            ),
        ]

        agent = WebSearchAgent(llm_client=mock_llm, search_tool=mock_search)
        response = agent.answer("test question")

        assert response.answer != ""
        assert len(response.sources) == 1
        assert response.confidence == "high"


class TestTextProcessingIntegration:
    """Integration tests for text processing pipeline."""

    def test_chunk_count_scales_with_text(self):
        """More text should produce more chunks."""
        from agent.utils.text_processing import RecursiveTextSplitter

        splitter = RecursiveTextSplitter()

        short_text = "Short text. " * 10
        long_text = "Long text about various topics. " * 500

        short_chunks = splitter.split(short_text)
        long_chunks = splitter.split(long_text)

        assert len(long_chunks) > len(short_chunks)

    def test_chunks_maintain_content(self):
        """Chunked text should preserve all original content."""
        from agent.utils.text_processing import RecursiveTextSplitter

        splitter = RecursiveTextSplitter()
        original = "AI is great. ML is powerful. NLP is fascinating. " * 100

        chunks = splitter.split(original)

        # All key terms should appear in at least one chunk
        all_text = " ".join(c.text for c in chunks)
        assert "AI" in all_text
        assert "ML" in all_text
        assert "NLP" in all_text


class TestVectorStoreIntegration:
    """Integration tests for vector store operations."""

    def test_add_search_cycle(self):
        """Full add-search cycle should work correctly."""
        import numpy as np
        from agent.retrieval.vector_store import VectorStore

        store = VectorStore(dimension=8)

        # Add some vectors
        vectors = np.random.rand(10, 8).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        texts = [f"Document {i}" for i in range(10)]
        metadata = [{"page": i} for i in range(10)]

        store.add(vectors, texts, metadata)
        assert store.size == 10

        # Search
        query = vectors[0]  # Should match Document 0
        results = store.search(query, top_k=3)

        assert len(results) == 3
        assert results[0].text == "Document 0"
        assert results[0].score > 0.99  # Almost perfect match

    def test_save_load_preserves_search(self, tmp_path):
        """Saved/loaded index should produce same search results."""
        import numpy as np
        from agent.retrieval.vector_store import VectorStore

        store = VectorStore(dimension=8)
        vectors = np.random.rand(5, 8).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        texts = [f"Doc {i}" for i in range(5)]

        store.add(vectors, texts)

        # Search before save
        query = vectors[2]
        results_before = store.search(query, top_k=2)

        # Save and load
        store.save(str(tmp_path), "test")
        new_store = VectorStore(dimension=8)
        new_store.load(str(tmp_path), "test")

        # Search after load
        results_after = new_store.search(query, top_k=2)

        assert results_before[0].text == results_after[0].text


class TestBM25Integration:
    """Integration tests for BM25 retrieval."""

    def test_keyword_matching(self):
        """BM25 should rank exact keyword matches higher."""
        from agent.retrieval.bm25_store import BM25Store

        store = BM25Store()
        docs = [
            "Python is a programming language used for machine learning",
            "Java is used in enterprise applications",
            "Python machine learning libraries include scikit-learn and TensorFlow",
        ]
        store.add(docs)

        results = store.search("Python machine learning")
        assert len(results) >= 2
        # Doc 0 and Doc 2 should rank higher than Doc 1
        result_texts = [r.text for r in results]
        assert "Java" not in result_texts[0]
