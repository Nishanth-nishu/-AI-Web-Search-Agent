"""
Hybrid retriever with RRF fusion and cross-encoder re-ranking.

Implements a 3-stage retrieval pipeline (Papers #5, #6):
1. Dual Retrieval: BM25 (sparse) + FAISS (dense) in parallel
2. RRF Fusion: Reciprocal Rank Fusion to merge ranked lists
3. Cross-Encoder Re-ranking: Fine-grained relevance scoring

This hybrid approach combines the strengths of:
- BM25: Exact keyword matching, handles rare terms well
- Dense embeddings: Semantic similarity, handles paraphrasing
- Cross-encoder: Precise query-document relevance scoring
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config
from agent.retrieval.vector_store import VectorStore, RetrievalResult
from agent.retrieval.bm25_store import BM25Store
from agent.retrieval.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Result from hybrid retrieval with fusion score."""
    text: str
    score: float
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)


class HybridRetriever:
    """
    Hybrid retriever combining BM25, dense retrieval, and cross-encoder re-ranking.
    
    Pipeline:
    1. Query → [BM25 top-K] + [FAISS top-K] (parallel)
    2. RRF fusion → merged candidates
    3. Cross-encoder re-ranking → final top-k
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedding_generator: EmbeddingGenerator,
        retrieval_config=None,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedding_generator = embedding_generator
        self.config = retrieval_config or config.retrieval
        self._cross_encoder = None

    @property
    def cross_encoder(self):
        """Lazy-load cross-encoder model."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self.config.cross_encoder_model)
                logger.info(
                    "Loaded cross-encoder: %s", self.config.cross_encoder_model
                )
            except ImportError:
                logger.warning(
                    "sentence-transformers not available, skipping cross-encoder"
                )
        return self._cross_encoder

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranking: bool = True,
    ) -> list[HybridResult]:
        """
        Perform hybrid retrieval with optional cross-encoder re-ranking.

        Args:
            query: Search query string.
            top_k: Number of final results to return.
            use_reranking: Whether to apply cross-encoder re-ranking.

        Returns:
            List of HybridResult sorted by relevance score (descending).
        """
        top_k = top_k or self.config.rerank_top_k
        initial_k = self.config.initial_retrieval_k

        # Stage 1: Dual retrieval
        bm25_results = self.bm25_store.search(query, top_k=initial_k)
        
        query_embedding = self.embedding_generator.embed_single(query)
        dense_results = self.vector_store.search(query_embedding, top_k=initial_k)

        logger.info(
            "Stage 1 - BM25: %d results, Dense: %d results",
            len(bm25_results), len(dense_results),
        )

        # Stage 2: RRF Fusion
        fused_results = self._reciprocal_rank_fusion(
            bm25_results, dense_results
        )

        logger.info("Stage 2 - RRF fused: %d candidates", len(fused_results))

        # Stage 3: Cross-encoder re-ranking (optional)
        if use_reranking and self.cross_encoder and fused_results:
            fused_results = self._rerank(query, fused_results)
            logger.info("Stage 3 - Re-ranked: %d results", len(fused_results))

        return fused_results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list,
        dense_results: list,
    ) -> list[HybridResult]:
        """
        Merge BM25 and dense results using Reciprocal Rank Fusion.

        RRF score = Σ 1 / (k + rank_i) for each ranking

        This approach is rank-based (not score-based), which avoids the
        problem of incomparable score scales between BM25 and cosine similarity.
        
        Reference: Cormack et al., "Reciprocal Rank Fusion outperforms 
        Condorcet and individual Rank Learning Methods" (SIGIR 2009)
        """
        k = self.config.rrf_k
        text_scores: dict[str, dict] = {}

        # Add BM25 results
        for rank, result in enumerate(bm25_results):
            text_key = result.text[:200]  # Use first 200 chars as key
            if text_key not in text_scores:
                text_scores[text_key] = {
                    "text": result.text,
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "dense_rank": None,
                    "metadata": result.metadata,
                }
            text_scores[text_key]["rrf_score"] += 1.0 / (k + rank + 1)
            text_scores[text_key]["bm25_rank"] = rank + 1

        # Add dense results
        for rank, result in enumerate(dense_results):
            text_key = result.text[:200]
            if text_key not in text_scores:
                text_scores[text_key] = {
                    "text": result.text,
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "dense_rank": None,
                    "metadata": result.metadata,
                }
            text_scores[text_key]["rrf_score"] += 1.0 / (k + rank + 1)
            text_scores[text_key]["dense_rank"] = rank + 1

        # Sort by RRF score
        sorted_results = sorted(
            text_scores.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        return [
            HybridResult(
                text=item["text"],
                score=item["rrf_score"],
                bm25_rank=item["bm25_rank"],
                dense_rank=item["dense_rank"],
                metadata=item["metadata"],
            )
            for item in sorted_results
        ]

    def _rerank(
        self, query: str, candidates: list[HybridResult]
    ) -> list[HybridResult]:
        """
        Re-rank candidates using a cross-encoder model.

        The cross-encoder scores each (query, candidate) pair independently,
        providing much more accurate relevance scores than bi-encoder similarity.
        
        Reference: Paper #6 — Cross-encoder re-ranking improves retrieval
        precision by 15-30%.
        """
        if not self.cross_encoder or not candidates:
            return candidates

        # Create query-document pairs
        pairs = [(query, c.text) for c in candidates]

        try:
            scores = self.cross_encoder.predict(pairs)

            # Update scores and sort
            for candidate, score in zip(candidates, scores):
                candidate.rerank_score = float(score)
                candidate.score = float(score)  # Use rerank score as primary

            candidates.sort(key=lambda x: x.score, reverse=True)

        except Exception as e:
            logger.warning("Cross-encoder re-ranking failed: %s", e)

        return candidates

    def dense_only(self, query: str, top_k: int = 5) -> list[HybridResult]:
        """Dense-only retrieval (no BM25 or re-ranking)."""
        query_embedding = self.embedding_generator.embed_single(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return [
            HybridResult(
                text=r.text,
                score=r.score,
                dense_rank=i + 1,
                metadata=r.metadata,
            )
            for i, r in enumerate(results)
        ]

    def bm25_only(self, query: str, top_k: int = 5) -> list[HybridResult]:
        """BM25-only retrieval (no dense or re-ranking)."""
        results = self.bm25_store.search(query, top_k=top_k)
        return [
            HybridResult(
                text=r.text,
                score=r.score,
                bm25_rank=i + 1,
                metadata=r.metadata,
            )
            for i, r in enumerate(results)
        ]
