"""
Centralized configuration for the AI Agent system.
Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for the LLM provider."""
    provider: str = "groq"
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
    temperature: float = 0.3
    max_tokens: int = 4096
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class SearchConfig:
    """Configuration for web search tools."""
    max_results: int = 5
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    max_content_length: int = 8000  # Max chars to extract per page
    request_timeout: int = 10
    rate_limit_delay: float = 0.5


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    dimension: int = 384  # Dimension for all-MiniLM-L6-v2
    batch_size: int = 64
    normalize: bool = True


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50"))
    )
    separators: list = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " "]
    )


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "5"))
    )
    rerank_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_TOP_K", "3"))
    )
    initial_retrieval_k: int = 20  # Candidates before reranking
    rrf_k: int = 60  # RRF constant
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    bm25_weight: float = 0.4
    dense_weight: float = 0.6


@dataclass
class AppConfig:
    """Main application configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    vector_store_dir: str = "./vector_stores"
    log_level: str = "INFO"


# Global config instance
config = AppConfig()
