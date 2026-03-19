"""
BM25 sparse retrieval store.

Implements BM25 (Okapi) ranking for keyword-based retrieval.
Complements dense vector search for hybrid retrieval (Paper #5, HYRR).

BM25 excels at exact keyword matching while dense embeddings
capture semantic similarity — combining both yields the best results.
"""

import logging
import re
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config

logger = logging.getLogger(__name__)

# Common English stopwords
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "the", "this", "but", "they",
    "have", "had", "what", "when", "where", "who", "which", "or", "not",
    "no", "can", "do", "does", "did", "been", "being", "would", "could",
    "should", "may", "might", "shall", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "same",
    "so", "than", "too", "very", "just", "because", "if", "while",
}


class BM25RetrievalResult:
    """Result from a BM25 search."""
    def __init__(self, text: str, score: float, index: int, metadata: dict = None):
        self.text = text
        self.score = score
        self.index = index
        self.metadata = metadata or {}


class BM25Store:
    """
    BM25 sparse retrieval store using rank-bm25.
    
    Tokenizes documents with simple word splitting + stopword removal.
    Returns scored document indices compatible with the VectorStore interface.
    """

    def __init__(self):
        self._bm25 = None
        self._texts: list[str] = []
        self._metadata: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []

    @property
    def size(self) -> int:
        """Number of documents in the store."""
        return len(self._texts)

    def add(self, texts: list[str], metadata: Optional[list[dict]] = None):
        """
        Add documents to the BM25 index.

        Args:
            texts: List of text documents.
            metadata: Optional metadata for each document.
        """
        self._texts.extend(texts)
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in texts])

        # Tokenize and rebuild index
        self._tokenized_corpus = [self._tokenize(text) for text in self._texts]
        self._build_index()

        logger.info("BM25 index built with %d documents", len(self._texts))

    def _build_index(self):
        """Build the BM25 index from tokenized corpus."""
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        except ImportError:
            raise ImportError(
                "rank-bm25 is required. Install with: pip install rank-bm25"
            )

    def search(self, query: str, top_k: int = 5) -> list[BM25RetrievalResult]:
        """
        Search for the most relevant documents using BM25.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of BM25RetrievalResult sorted by score (descending).
        """
        if self._bm25 is None or self.size == 0:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            idx = int(idx)
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(BM25RetrievalResult(
                text=self._texts[idx],
                score=score,
                index=idx,
                metadata=self._metadata[idx] if idx < len(self._metadata) else {},
            ))

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenize text with lowercasing and stopword removal.

        Args:
            text: Text to tokenize.

        Returns:
            List of lowercase tokens with stopwords removed.
        """
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove stopwords and very short tokens
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    def clear(self):
        """Clear all data from the store."""
        self._texts.clear()
        self._metadata.clear()
        self._tokenized_corpus.clear()
        self._bm25 = None
        logger.info("Cleared BM25 store")
