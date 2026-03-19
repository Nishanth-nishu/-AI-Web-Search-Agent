"""
Embedding generation using sentence-transformers.

Uses all-MiniLM-L6-v2 (384-dim) for fast, high-quality embeddings.
Supports batch processing and normalization for cosine similarity.

Part of the RAG pipeline (Paper #2) — converts text chunks into 
dense vector representations for semantic similarity search.
"""

import logging
from typing import Optional
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Embedding generator using sentence-transformers.
    
    Lazy-loads the model on first use to save memory when not needed.
    Supports batch processing for efficient bulk embedding.
    """

    def __init__(self, embedding_config=None):
        self.config = embedding_config or config.embedding
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config.model_name)
                logger.info("Loaded embedding model: %s", self.config.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def embed(self, texts: list[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Override default batch size.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        batch_size = batch_size or self.config.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=len(texts) > 100,
        )

        logger.info(
            "Generated %d embeddings (dim=%d)", len(embeddings), embeddings.shape[1]
        )

        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            numpy array of shape (embedding_dim,).
        """
        if not text or not text.strip():
            return np.zeros(self.config.dimension)

        embeddings = self.embed([text])
        return embeddings[0]

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.config.dimension
