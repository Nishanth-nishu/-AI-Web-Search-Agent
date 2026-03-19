"""
FAISS-based vector store for dense retrieval.

Implements:
- Index creation (IndexFlatIP for cosine similarity on normalized vectors)
- Add/search/delete operations
- Persistence (save/load to disk)
- Metadata storage alongside vectors

Part of the RAG pipeline (Paper #2) — stores document chunk embeddings
for fast similarity search during the retrieval phase.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a vector store search."""
    text: str
    score: float
    index: int
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """
    FAISS-based vector store with metadata persistence.
    
    Uses IndexFlatIP (inner product) for cosine similarity search
    on L2-normalized embeddings. Stores metadata in a parallel JSON file.
    """

    def __init__(self, dimension: int):
        """
        Initialize vector store.

        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self._index = None
        self._texts: list[str] = []
        self._metadata: list[dict] = []
        self._initialize_index()

    def _initialize_index(self):
        """Create a new FAISS index."""
        try:
            import faiss
            # Inner product on normalized vectors = cosine similarity
            self._index = faiss.IndexFlatIP(self.dimension)
            logger.info("Initialized FAISS index (dim=%d)", self.dimension)
        except ImportError:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            )

    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        return self._index.ntotal if self._index else 0

    def add(
        self,
        embeddings: np.ndarray,
        texts: list[str],
        metadata: Optional[list[dict]] = None,
    ):
        """
        Add vectors with associated texts and metadata to the store.

        Args:
            embeddings: numpy array of shape (n, dimension).
            texts: Corresponding text strings.
            metadata: Optional metadata dicts for each entry.
        """
        if len(embeddings) == 0:
            return

        if len(embeddings) != len(texts):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(texts)} texts"
            )

        # Ensure float32 for FAISS
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self._index.add(embeddings)
        self._texts.extend(texts)

        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in texts])

        logger.info("Added %d vectors (total: %d)", len(texts), self.size)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        """
        Search for the most similar vectors.

        Args:
            query_embedding: Query vector of shape (dimension,).
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult objects sorted by score (descending).
        """
        if self.size == 0:
            return []

        # Ensure correct shape and dtype
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # Search
        k = min(top_k, self.size)
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for not-found
                continue
            results.append(RetrievalResult(
                text=self._texts[idx],
                score=float(score),
                index=int(idx),
                metadata=self._metadata[idx] if idx < len(self._metadata) else {},
            ))

        return results

    def save(self, directory: str, name: str = "index"):
        """
        Save the index and metadata to disk.

        Args:
            directory: Directory to save files.
            name: Base name for the files.
        """
        import faiss

        os.makedirs(directory, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(directory, f"{name}.faiss")
        faiss.write_index(self._index, index_path)

        # Save texts and metadata
        meta_path = os.path.join(directory, f"{name}_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "texts": self._texts,
                "metadata": self._metadata,
                "dimension": self.dimension,
            }, f)

        logger.info("Saved vector store to %s (%d vectors)", directory, self.size)

    def load(self, directory: str, name: str = "index"):
        """
        Load the index and metadata from disk.

        Args:
            directory: Directory containing saved files.
            name: Base name for the files.
        """
        import faiss

        index_path = os.path.join(directory, f"{name}.faiss")
        meta_path = os.path.join(directory, f"{name}_meta.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        self._index = faiss.read_index(index_path)

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                data = json.load(f)
                self._texts = data["texts"]
                self._metadata = data.get("metadata", [{} for _ in self._texts])
                self.dimension = data.get("dimension", self.dimension)

        logger.info("Loaded vector store from %s (%d vectors)", directory, self.size)

    def clear(self):
        """Clear all data from the store."""
        self._texts.clear()
        self._metadata.clear()
        self._initialize_index()
        logger.info("Cleared vector store")
