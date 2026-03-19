"""
Tests for the retrieval pipeline.

Tests cover:
- Embedding generation (mocked model)
- FAISS vector store CRUD operations
- BM25 store search and tokenization
- Hybrid retriever RRF fusion
- Cross-encoder re-ranking (mocked)
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestEmbeddingGenerator:
    """Tests for the embedding generator."""

    def test_embed_returns_correct_shape(self):
        from agent.retrieval.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            return_value=np.random.rand(3, 384).astype(np.float32)
        )
        gen._model = mock_model
        result = gen.embed(["hello", "world", "test"])
        assert result.shape == (3, 384)

    def test_embed_empty_list(self):
        from agent.retrieval.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        result = gen.embed([])
        assert len(result) == 0


class TestVectorStore:
    """Tests for the FAISS vector store."""

    def setup_method(self):
        from agent.retrieval.vector_store import VectorStore
        self.store = VectorStore(dimension=4)

    def test_initial_size_is_zero(self):
        assert self.store.size == 0

    def test_add_and_search(self):
        """Adding vectors and searching should work correctly."""
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        # Normalize for cosine sim
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        texts = ["cat", "dog", "fish"]
        self.store.add(vectors, texts)

        assert self.store.size == 3

        # Search with a query similar to "cat"
        query = np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32)
        query = query / np.linalg.norm(query)
        results = self.store.search(query, top_k=2)

        assert len(results) == 2
        assert results[0].text == "cat"  # Most similar

    def test_add_mismatched_lengths_raises(self):
        """Mismatched embeddings and texts should raise ValueError."""
        vectors = np.random.rand(3, 4).astype(np.float32)
        texts = ["a", "b"]  # Only 2 texts for 3 vectors
        with pytest.raises(ValueError, match="Mismatch"):
            self.store.add(vectors, texts)

    def test_search_empty_store(self):
        """Searching empty store should return empty list."""
        query = np.random.rand(4).astype(np.float32)
        results = self.store.search(query, top_k=5)
        assert results == []

    def test_clear(self):
        """Clear should reset the store."""
        vectors = np.random.rand(3, 4).astype(np.float32)
        self.store.add(vectors, ["a", "b", "c"])
        assert self.store.size == 3
        self.store.clear()
        assert self.store.size == 0

    def test_save_and_load(self, tmp_path):
        """Save and load should preserve data."""
        vectors = np.random.rand(5, 4).astype(np.float32)
        texts = ["a", "b", "c", "d", "e"]
        metadata = [{"idx": i} for i in range(5)]
        self.store.add(vectors, texts, metadata)

        self.store.save(str(tmp_path), "test")

        from agent.retrieval.vector_store import VectorStore
        new_store = VectorStore(dimension=4)
        new_store.load(str(tmp_path), "test")

        assert new_store.size == 5


class TestBM25Store:
    """Tests for the BM25 sparse retrieval store."""

    def setup_method(self):
        from agent.retrieval.bm25_store import BM25Store
        self.store = BM25Store()

    def test_initial_size_is_zero(self):
        assert self.store.size == 0

    def test_add_and_search(self):
        """Adding documents and searching should return ranked results."""
        docs = [
            "machine learning is a subset of artificial intelligence",
            "natural language processing deals with text",
            "computer vision processes images and videos",
        ]
        self.store.add(docs)
        assert self.store.size == 3

        results = self.store.search("artificial intelligence machine learning")
        assert len(results) > 0
        assert results[0].text == docs[0]  # Best match

    def test_search_empty_store(self):
        """Searching empty store returns empty list."""
        results = self.store.search("test query")
        assert results == []

    def test_tokenization_removes_stopwords(self):
        """Tokenization should remove common stopwords."""
        tokens = self.store._tokenize("the quick brown fox is a test")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_clear(self):
        self.store.add(["hello world"])
        self.store.clear()
        assert self.store.size == 0


class TestHybridRetriever:
    """Tests for the hybrid retriever with RRF fusion."""

    def test_reciprocal_rank_fusion(self):
        """RRF should correctly fuse results from two rankers."""
        from agent.retrieval.hybrid_retriever import HybridRetriever
        from agent.retrieval.vector_store import VectorStore, RetrievalResult
        from agent.retrieval.bm25_store import BM25Store, BM25RetrievalResult
        from agent.retrieval.embeddings import EmbeddingGenerator

        vs = VectorStore(dimension=4)
        bm25 = BM25Store()
        emb = EmbeddingGenerator()

        retriever = HybridRetriever(vs, bm25, emb)

        bm25_results = [
            BM25RetrievalResult(text="doc A - bm25 rank 1", score=5.0, index=0),
            BM25RetrievalResult(text="doc B - bm25 rank 2", score=3.0, index=1),
        ]
        dense_results = [
            RetrievalResult(text="doc B - bm25 rank 2", score=0.9, index=1),
            RetrievalResult(text="doc C - dense rank 2", score=0.7, index=2),
        ]

        fused = retriever._reciprocal_rank_fusion(bm25_results, dense_results)

        # doc B appears in both lists → highest RRF score
        assert len(fused) == 3
        assert fused[0].text.startswith("doc B")  # Should be ranked first


class TestTextProcessing:
    """Tests for text chunking utilities."""

    def test_recursive_split_basic(self):
        from agent.utils.text_processing import RecursiveTextSplitter
        splitter = RecursiveTextSplitter()

        text = "Paragraph one about AI.\n\nParagraph two about ML.\n\nParagraph three about NLP."
        chunks = splitter.split(text)
        assert len(chunks) >= 1
        assert all(isinstance(c.text, str) for c in chunks)

    def test_empty_text_returns_empty(self):
        from agent.utils.text_processing import RecursiveTextSplitter
        splitter = RecursiveTextSplitter()
        assert splitter.split("") == []
        assert splitter.split("   ") == []

    def test_chunk_metadata_exists(self):
        from agent.utils.text_processing import RecursiveTextSplitter
        splitter = RecursiveTextSplitter()

        text = "Hello world. " * 200  # Generate enough tokens
        chunks = splitter.split(text)

        for chunk in chunks:
            assert chunk.index >= 0
            assert "token_count" in chunk.metadata

    def test_text_cleaning(self):
        from agent.utils.text_processing import RecursiveTextSplitter
        cleaned = RecursiveTextSplitter._clean_text("hello\n\n\n\n\nworld  test")
        assert "\n\n\n" not in cleaned
        assert "  " not in cleaned
