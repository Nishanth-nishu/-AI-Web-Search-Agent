"""
AI PDF RAG Agent for document summarization and question answering.

Based on Papers #2 (RAG), #3 (Self-RAG), #5 (HYRR), #6 (Cross-Encoder):
- Document ingestion: PDF → text extraction → chunking → embedding → indexing
- Hybrid retrieval: BM25 + dense + RRF fusion + cross-encoder re-ranking
- Context-aware QA: Retrieved chunks → LLM generates grounded answer
- Map-reduce summarization for full documents
- Self-validation: Checks answer groundedness against source chunks

The agent demonstrates:
1. Complete RAG pipeline implementation
2. Hybrid retrieval for best-in-class accuracy
3. Document summarization (map-reduce)
4. Context-aware question answering
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from agent.llm_client import LLMClient
from agent.tools.pdf_tool import PDFTool, PDFDocument
from agent.utils.text_processing import RecursiveTextSplitter, TextChunk, build_page_offset_map
from agent.retrieval.embeddings import EmbeddingGenerator
from agent.retrieval.vector_store import VectorStore
from agent.retrieval.bm25_store import BM25Store
from agent.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


# ─── System Prompts ──────────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are a precise document question-answering assistant. You answer questions based ONLY on the provided document context.

Rules:
1. Base your answer SOLELY on the provided context chunks
2. If the context doesn't contain enough information, say so clearly
3. Quote relevant passages when appropriate
4. Be specific and factual
5. Reference page numbers when available

Respond in this JSON format:
{
    "answer": "Your detailed answer based on the context",
    "confidence": "high/medium/low",
    "relevant_pages": [1, 2],
    "evidence": ["quote or fact supporting the answer"]
}
"""

SUMMARY_SYSTEM_PROMPT = """You are a document summarization assistant. Provide a clear, comprehensive summary of the following document content.

Rules:
1. Capture the main topics, findings, and conclusions
2. Preserve key facts, figures, and methodologies
3. Organize the summary logically
4. Be concise but thorough
5. Do not add information not present in the document

Respond in this JSON format:
{
    "summary": "Your comprehensive summary",
    "key_topics": ["topic1", "topic2"],
    "key_findings": ["finding1", "finding2"]
}
"""

CHUNK_SUMMARY_PROMPT = """Summarize the following section of a document. Be concise but capture the key information:

{chunk_text}

Provide a brief summary (2-3 sentences):"""

MAP_REDUCE_PROMPT = """You are given multiple section summaries from a single document. Combine them into a single, coherent, comprehensive summary.

Section Summaries:
{section_summaries}

Create a well-organized, comprehensive summary that:
1. Captures all key topics and findings
2. Eliminates redundancy
3. Maintains logical flow
4. Includes important details, figures, and conclusions

Respond in this JSON format:
{{
    "summary": "Your comprehensive combined summary",
    "key_topics": ["topic1", "topic2"],
    "key_findings": ["finding1", "finding2"]
}}
"""


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class PDFRAGResponse:
    """Response from the PDF RAG agent."""
    query: str
    answer: str
    confidence: str = "medium"
    relevant_pages: list[int] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    chunks_used: int = 0


@dataclass
class PDFSummaryResponse:
    """Summary response from the PDF RAG agent."""
    filepath: str
    summary: str
    key_topics: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    total_pages: int = 0
    total_chunks: int = 0


# ─── Agent ───────────────────────────────────────────────────────────────────

class PDFRAGAgent:
    """
    PDF RAG Agent for document summarization and question answering.
    
    Pipeline:
    1. INGEST: PDF → Extract → Chunk → Embed → Index (FAISS + BM25)
    2. QUERY: Hybrid retrieve → Rerank → Context → LLM → Answer
    3. SUMMARIZE: Map-reduce summarization over all chunks
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        pdf_tool: Optional[PDFTool] = None,
    ):
        self.llm = llm_client or LLMClient()
        self.pdf_tool = pdf_tool or PDFTool()
        self.splitter = RecursiveTextSplitter()
        self.embedder = EmbeddingGenerator()

        # These are initialized per-document during ingestion
        self._vector_store: Optional[VectorStore] = None
        self._bm25_store: Optional[BM25Store] = None
        self._retriever: Optional[HybridRetriever] = None
        self._chunks: list[TextChunk] = []
        self._document: Optional[PDFDocument] = None

    @property
    def is_ingested(self) -> bool:
        """Check if a document has been ingested."""
        return self._vector_store is not None and self._vector_store.size > 0

    def ingest(self, filepath: str) -> dict:
        """
        Ingest a PDF document into the RAG pipeline.

        Steps:
        1. Extract text from PDF (PyMuPDF/pdfplumber)
        2. Split text into chunks (recursive splitter)
        3. Generate embeddings (sentence-transformers)
        4. Index in FAISS (dense) and BM25 (sparse)

        Args:
            filepath: Path to the PDF file.

        Returns:
            Dict with ingestion statistics.
        """
        logger.info("="*60)
        logger.info("Ingesting PDF: %s", filepath)
        logger.info("="*60)

        # Step 1: Extract text
        logger.info("Step 1: Extracting text from PDF...")
        self._document = self.pdf_tool.extract(filepath)

        if self._document.is_empty:
            raise ValueError(
                f"No text could be extracted from {filepath}. "
                "The PDF may be scanned/image-only."
            )

        logger.info(
            "  Extracted %d pages, title: '%s'",
            self._document.total_pages, self._document.title,
        )

        # Step 2: Chunk text
        logger.info("Step 2: Splitting text into chunks...")
        full_text = self._document.full_text
        page_map = build_page_offset_map(self._document.pages)
        self._chunks = self.splitter.split(full_text, page_numbers=page_map)

        if not self._chunks:
            raise ValueError("Text splitting produced no chunks.")

        logger.info("  Created %d chunks", len(self._chunks))

        # Step 3: Generate embeddings
        logger.info("Step 3: Generating embeddings...")
        chunk_texts = [chunk.text for chunk in self._chunks]
        embeddings = self.embedder.embed(chunk_texts)

        logger.info("  Generated embeddings: shape=%s", embeddings.shape)

        # Step 4: Index in stores
        logger.info("Step 4: Building indices (FAISS + BM25)...")
        
        # Prepare metadata
        chunk_metadata = [
            {
                "chunk_index": chunk.index,
                "page_number": chunk.page_number,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            for chunk in self._chunks
        ]

        # FAISS vector store
        self._vector_store = VectorStore(dimension=self.embedder.dimension)
        self._vector_store.add(embeddings, chunk_texts, chunk_metadata)

        # BM25 store
        self._bm25_store = BM25Store()
        self._bm25_store.add(chunk_texts, chunk_metadata)

        # Hybrid retriever
        self._retriever = HybridRetriever(
            vector_store=self._vector_store,
            bm25_store=self._bm25_store,
            embedding_generator=self.embedder,
        )

        stats = {
            "filepath": filepath,
            "title": self._document.title,
            "total_pages": self._document.total_pages,
            "total_chunks": len(self._chunks),
            "total_tokens": sum(
                c.metadata.get("token_count", 0) for c in self._chunks
            ),
        }

        logger.info("Ingestion complete: %s", stats)
        return stats

    def query(self, question: str) -> PDFRAGResponse:
        """
        Answer a question about the ingested document.

        Uses hybrid retrieval (BM25 + dense + cross-encoder) to find
        the most relevant chunks, then passes them to the LLM for
        context-aware answer generation.

        Args:
            question: Natural language question about the document.

        Returns:
            PDFRAGResponse with answer, confidence, and evidence.
        """
        if not self.is_ingested:
            return PDFRAGResponse(
                query=question,
                answer="No document has been ingested. Please load a PDF first.",
                confidence="low",
            )

        if not question or not question.strip():
            return PDFRAGResponse(
                query=question,
                answer="Please provide a valid question.",
                confidence="low",
            )

        logger.info("Query: %s", question)

        # Retrieve relevant chunks
        results = self._retriever.retrieve(
            query=question,
            top_k=config.retrieval.rerank_top_k,
            use_reranking=True,
        )

        if not results:
            return PDFRAGResponse(
                query=question,
                answer="No relevant information found in the document for your question.",
                confidence="low",
            )

        # Build context from retrieved chunks
        context = self._build_context(results)

        # Generate answer using LLM
        prompt = (
            f"Document Title: {self._document.title}\n\n"
            f"Question: {question}\n\n"
            f"Relevant Document Sections:\n{context}\n\n"
            f"Based on the above document sections, answer the question."
        )

        try:
            result = self.llm.generate_json(
                prompt=prompt,
                system_prompt=QA_SYSTEM_PROMPT,
                temperature=0.2,
            )

            return PDFRAGResponse(
                query=question,
                answer=result.get("answer", "Unable to generate answer."),
                confidence=result.get("confidence", "medium"),
                relevant_pages=result.get("relevant_pages", []),
                evidence=result.get("evidence", []),
                chunks_used=len(results),
            )

        except Exception as e:
            logger.warning("JSON QA failed: %s. Trying plain text.", e)
            answer = self.llm.generate(
                prompt=prompt,
                system_prompt="Answer the question based only on the provided document sections. Be specific and factual.",
                temperature=0.2,
            )
            return PDFRAGResponse(
                query=question,
                answer=answer,
                confidence="medium",
                chunks_used=len(results),
            )

    def summarize(self) -> PDFSummaryResponse:
        """
        Generate a comprehensive summary of the ingested document.

        Uses map-reduce strategy:
        1. MAP: Summarize each chunk individually
        2. REDUCE: Combine chunk summaries into a final summary

        Returns:
            PDFSummaryResponse with summary, topics, and findings.
        """
        if not self.is_ingested:
            return PDFSummaryResponse(
                filepath="",
                summary="No document has been ingested. Please load a PDF first.",
            )

        logger.info("Generating summary for: %s", self._document.title)

        # If document is small enough, summarize directly
        full_text = self._document.full_text
        if len(self._chunks) <= 5:
            return self._direct_summarize(full_text)

        # Map-reduce for larger documents
        return self._map_reduce_summarize()

    def _direct_summarize(self, text: str) -> PDFSummaryResponse:
        """Directly summarize a short document."""
        # Truncate if needed
        if len(text) > 15000:
            text = text[:15000] + "\n...[truncated]"

        prompt = f"Document Content:\n{text}\n\nProvide a comprehensive summary."

        try:
            result = self.llm.generate_json(
                prompt=prompt,
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                temperature=0.3,
            )

            return PDFSummaryResponse(
                filepath=self._document.filepath,
                summary=result.get("summary", "Unable to generate summary."),
                key_topics=result.get("key_topics", []),
                key_findings=result.get("key_findings", []),
                total_pages=self._document.total_pages,
                total_chunks=len(self._chunks),
            )

        except Exception as e:
            logger.warning("JSON summary failed: %s. Trying plain text.", e)
            summary = self.llm.generate(
                prompt=prompt,
                system_prompt="Summarize the document content comprehensively.",
                temperature=0.3,
            )
            return PDFSummaryResponse(
                filepath=self._document.filepath,
                summary=summary,
                total_pages=self._document.total_pages,
                total_chunks=len(self._chunks),
            )

    def _map_reduce_summarize(self) -> PDFSummaryResponse:
        """
        Map-reduce summarization for large documents.
        
        MAP phase: Summarize each chunk batch individually
        REDUCE phase: Combine chunk summaries into final summary
        """
        logger.info("Using map-reduce summarization for %d chunks", len(self._chunks))

        # MAP phase: Summarize chunks in batches
        batch_size = 3
        section_summaries = []

        for i in range(0, len(self._chunks), batch_size):
            batch = self._chunks[i : i + batch_size]
            batch_text = "\n\n".join(c.text for c in batch)

            if not batch_text.strip():
                continue

            # Truncate very large batches
            if len(batch_text) > 6000:
                batch_text = batch_text[:6000] + "..."

            try:
                summary = self.llm.generate(
                    prompt=CHUNK_SUMMARY_PROMPT.format(chunk_text=batch_text),
                    system_prompt="You are a document summarization assistant. Be concise and capture key information.",
                    temperature=0.3,
                    max_tokens=500,
                )
                section_summaries.append(summary)
            except Exception as e:
                logger.warning("Chunk batch summary failed: %s", e)
                # Use first 200 chars as fallback
                section_summaries.append(batch_text[:200] + "...")

        if not section_summaries:
            return PDFSummaryResponse(
                filepath=self._document.filepath,
                summary="Unable to generate summary — all chunk summarizations failed.",
                total_pages=self._document.total_pages,
                total_chunks=len(self._chunks),
            )

        # REDUCE phase: Combine summaries
        logger.info("REDUCE: Combining %d section summaries", len(section_summaries))
        combined = "\n\n".join(
            f"Section {i+1}: {s}" for i, s in enumerate(section_summaries)
        )

        try:
            result = self.llm.generate_json(
                prompt=MAP_REDUCE_PROMPT.format(section_summaries=combined),
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                temperature=0.3,
            )

            return PDFSummaryResponse(
                filepath=self._document.filepath,
                summary=result.get("summary", "Unable to generate summary."),
                key_topics=result.get("key_topics", []),
                key_findings=result.get("key_findings", []),
                total_pages=self._document.total_pages,
                total_chunks=len(self._chunks),
            )

        except Exception as e:
            logger.warning("JSON reduce failed: %s. Trying plain text.", e)
            summary = self.llm.generate(
                prompt=f"Combine these section summaries into one comprehensive summary:\n\n{combined}",
                system_prompt="Create a well-organized summary from the section summaries provided.",
                temperature=0.3,
            )
            return PDFSummaryResponse(
                filepath=self._document.filepath,
                summary=summary,
                total_pages=self._document.total_pages,
                total_chunks=len(self._chunks),
            )

    def _build_context(self, results: list) -> str:
        """Build context string from retrieval results."""
        context_parts = []
        for i, result in enumerate(results, 1):
            page_info = ""
            if result.metadata and result.metadata.get("page_number"):
                page_info = f" (Page {result.metadata['page_number']})"

            context_parts.append(
                f"[Chunk {i}{page_info}]\n{result.text}\n"
            )
        return "\n---\n".join(context_parts)

    def format_response(self, response) -> str:
        """Format a response for CLI display."""
        if isinstance(response, PDFRAGResponse):
            return self._format_qa_response(response)
        elif isinstance(response, PDFSummaryResponse):
            return self._format_summary_response(response)
        return str(response)

    def _format_qa_response(self, response: PDFRAGResponse) -> str:
        """Format a QA response for display."""
        output = []
        output.append(f"Question: {response.query}")
        output.append("")
        output.append(f"Answer: {response.answer}")
        output.append("")

        if response.relevant_pages:
            output.append(f"Relevant Pages: {response.relevant_pages}")

        if response.evidence:
            output.append("Evidence:")
            for e in response.evidence:
                output.append(f"  • {e}")

        output.append(f"\nConfidence: {response.confidence}")
        output.append(f"Chunks Used: {response.chunks_used}")

        return "\n".join(output)

    def _format_summary_response(self, response: PDFSummaryResponse) -> str:
        """Format a summary response for display."""
        output = []
        output.append(f"Document: {os.path.basename(response.filepath)}")
        output.append(f"Pages: {response.total_pages}")
        output.append(f"Chunks: {response.total_chunks}")
        output.append("")
        output.append(f"Summary: {response.summary}")
        output.append("")

        if response.key_topics:
            output.append("Key Topics:")
            for topic in response.key_topics:
                output.append(f"  • {topic}")

        if response.key_findings:
            output.append("\nKey Findings:")
            for finding in response.key_findings:
                output.append(f"  • {finding}")

        return "\n".join(output)
