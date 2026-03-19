# AI Agent System — Web Search Agent + PDF RAG Agent

A production-grade AI agent system implementing two capabilities:
- **Challenge A**: AI Web Search Agent with ReAct reasoning
- **Challenge B**: PDF RAG Agent for document summarization & Q&A

Built from scratch without LangChain to demonstrate deep understanding of LLM integration, retrieval-augmented generation, and AI agent design.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface (main.py)                   │
├──────────────────────────┬──────────────────────────────────┤
│   Web Search Agent       │       PDF RAG Agent              │
│   (ReAct Loop)           │       (RAG Pipeline)             │
│                          │                                  │
│  ┌─────────┐             │  ┌──────────┐ ┌──────────────┐  │
│  │ Think   │             │  │ Ingest   │ │ Query/       │  │
│  │ Act     │             │  │ PDF→Chunk│ │ Summarize    │  │
│  │ Observe │             │  │ →Embed   │ │              │  │
│  │ Synth   │             │  │ →Index   │ │ Retrieve→    │  │
│  └─────────┘             │  └──────────┘ │ Rerank→LLM   │  │
│       │                  │       │       └──────────────┘  │
│  ┌─────────┐             │  ┌──────────────────────────┐   │
│  │ Search  │             │  │   Hybrid Retriever       │   │
│  │ Tool    │             │  │  BM25 + FAISS + Rerank   │   │
│  │(DDG)    │             │  └──────────────────────────┘   │
│  └─────────┘             │                                  │
├──────────────────────────┴──────────────────────────────────┤
│                    LLM Client (Groq)                        │
│              Llama 3.3 70B · Retry · JSON mode              │
└─────────────────────────────────────────────────────────────┘
```

## Research Papers Used

| # | Paper | Applied To |
|---|-------|-----------|
| 1 | **ReAct** (arXiv:2210.03629) — Reasoning + Acting | Web Search Agent loop |
| 2 | **RAG** (arXiv:2005.11401) — Retrieval-Augmented Generation | PDF RAG pipeline |
| 3 | **Self-RAG** (arXiv:2310.11511) — Adaptive retrieval | Retrieval quality |
| 4 | **FLARE** — Forward-Looking Active Retrieval | Agent retrieval |
| 5 | **HYRR** (ACL'24) — Hybrid re-ranking | Hybrid retrieval |
| 6 | **Cross-Encoder Re-ranking** | Precision improvement |
| 7 | **Recursive Chunking** (industry) | Text splitting |
| 8 | **Contextual Retrieval** (Anthropic'24) | Chunk quality |

## Setup Instructions

### Prerequisites
- Python 3.10+
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd end-edn-proj

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `groq` | LLM provider (Llama 3.3 70B) |
| `duckduckgo-search` | Web search (no API key needed) |
| `trafilatura` | Web page content extraction |
| `PyMuPDF` | PDF text extraction |
| `pdfplumber` | PDF fallback extraction |
| `sentence-transformers` | Embeddings + cross-encoder |
| `faiss-cpu` | Vector similarity search |
| `rank-bm25` | BM25 sparse retrieval |
| `tiktoken` | Token counting |
| `rich` | CLI formatting |

## How to Run

### Challenge A — Web Search Agent

```bash
python main.py search "What are the latest specs in MacBook this year?"
```

**Example Output:**
```
Question: What are the latest specs in MacBook this year?

Answer: Recent MacBook Pro models feature Apple's latest M4 family chips
for enhanced AI performance and efficiency...

Sources:
  [1] MacBook Pro - Apple
      https://apple.com/macbook-pro
  [2] MacBook Air - Apple  
      https://apple.com/macbook-air

Confidence: high
```

### Challenge B — PDF RAG Agent

```bash
# Summarize a PDF
python main.py pdf --file document.pdf --summarize

# Ask a question about a PDF
python main.py pdf --file document.pdf --query "What methodology was used?"

# Interactive mode
python main.py pdf --file document.pdf
```

**Example Output:**
```
Question: What methodology was used in the study?

Answer: The study used case studies combined with experimental evaluations
across three enterprise environments...

Relevant Pages: [2, 3]
Confidence: high
Chunks Used: 3
```

### Running Tests

```bash
pytest tests/ -v --tb=short
```

## Design Decisions & Trade-offs

### No Framework (No LangChain)
The entire system is built from scratch. Every component—ReAct loop, RAG pipeline, hybrid retrieval, RRF fusion—is implemented directly. This demonstrates deep understanding rather than framework abstraction.

### Groq (Free Tier, Llama 3.3 70B)
- **Pro**: Free, fast inference (>100 tok/s), strong open model
- **Con**: Rate limits on free tier
- **Mitigation**: Exponential backoff retry logic

### Hybrid Retrieval (BM25 + Dense + Cross-Encoder)
- **Pro**: Best-in-class accuracy, handles both keyword and semantic queries
- **Con**: Slower than single retriever
- **Mitigation**: Initial retrieval is fast; re-ranking only on top-20 candidates

### Recursive Chunking (512 tokens, 50 overlap)
- **Pro**: Robust default per NVIDIA benchmarks
- **Con**: Fixed chunk size may not fit all document types
- **Mitigation**: Configurable via `.env` file

### FAISS over Chroma/Pinecone
- **Pro**: No external service, fast, battle-tested
- **Con**: No filtering, in-memory only
- **Mitigation**: Metadata stored in JSON alongside index

## Project Structure

```
├── main.py                         # CLI entry point
├── config.py                       # Centralized configuration
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── agent/
│   ├── llm_client.py               # Groq LLM client with retry
│   ├── web_search_agent.py         # Challenge A — ReAct agent
│   ├── pdf_rag_agent.py            # Challenge B — RAG agent
│   ├── tools/
│   │   ├── search_tool.py          # DuckDuckGo + Tavily search
│   │   └── pdf_tool.py             # PDF extraction (PyMuPDF)
│   ├── retrieval/
│   │   ├── embeddings.py           # Sentence-transformers embeddings
│   │   ├── vector_store.py         # FAISS vector store
│   │   ├── bm25_store.py           # BM25 sparse retrieval
│   │   └── hybrid_retriever.py     # RRF + cross-encoder pipeline
│   └── utils/
│       └── text_processing.py      # Recursive text chunking
├── tests/
│   ├── test_tools.py               # Tool unit tests
│   ├── test_retrieval.py           # Retrieval pipeline tests
│   ├── test_web_search_agent.py    # Web agent tests
│   ├── test_pdf_rag_agent.py       # PDF agent tests
│   └── test_integration.py         # End-to-end tests
└── sample_data/
    └── sample.pdf                  # Test PDF document
```

## License

MIT
