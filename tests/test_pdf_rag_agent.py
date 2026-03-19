"""
Tests for the PDF RAG Agent.

Tests cover:
- PDF ingestion pipeline (mocked)
- Question answering with mocked retrieval
- Document summarization
- Error handling (empty PDF, no ingestion)
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from agent.pdf_rag_agent import PDFRAGAgent, PDFRAGResponse, PDFSummaryResponse
from agent.tools.pdf_tool import PDFDocument, PageContent
from agent.utils.text_processing import TextChunk


class TestPDFRAGAgent:
    """Tests for the PDF RAG agent."""

    def _mock_llm(self):
        """Create a mock LLM client."""
        mock = MagicMock()
        mock.generate_json.return_value = {
            "answer": "The study uses case studies.",
            "confidence": "high",
            "relevant_pages": [1, 2],
            "evidence": ["Case studies were used"],
        }
        mock.generate.return_value = "The study uses case studies."
        return mock

    def _mock_pdf_tool(self):
        """Create a mock PDF tool that returns a sample document."""
        mock = MagicMock()
        mock.extract.return_value = PDFDocument(
            filepath="/test/sample.pdf",
            title="Test Paper",
            author="Author",
            total_pages=3,
            pages=[
                PageContent(
                    page_number=1,
                    text="This paper presents a study on AI-driven automation. "
                         "We examine its impact on enterprise workflows.",
                ),
                PageContent(
                    page_number=2,
                    text="Methodology: We used case studies combined with "
                         "experimental evaluations across three enterprise environments. "
                         "The experiments measured productivity improvements.",
                ),
                PageContent(
                    page_number=3,
                    text="Results: Organizations saw a 30% improvement in productivity. "
                         "Conclusions: AI automation significantly enhances workflow efficiency.",
                ),
            ],
        )
        return mock

    def test_query_without_ingestion(self):
        """Should return low-confidence response if no document ingested."""
        agent = PDFRAGAgent(llm_client=self._mock_llm())
        response = agent.query("What is the methodology?")
        assert response.confidence == "low"
        assert "no document" in response.answer.lower()

    def test_summarize_without_ingestion(self):
        """Should return informative message if no document ingested."""
        agent = PDFRAGAgent(llm_client=self._mock_llm())
        response = agent.summarize()
        assert "no document" in response.summary.lower()

    def test_empty_query(self):
        """Empty queries should return low-confidence response."""
        agent = PDFRAGAgent(llm_client=self._mock_llm())
        response = agent.query("")
        assert response.confidence == "low"

    @patch("agent.pdf_rag_agent.EmbeddingGenerator")
    @patch("agent.pdf_rag_agent.VectorStore")
    @patch("agent.pdf_rag_agent.BM25Store")
    def test_ingest_pipeline(self, MockBM25, MockVS, MockEmb):
        """Ingestion should extract, chunk, embed, and index."""
        # Setup mocks
        mock_emb_instance = MagicMock()
        mock_emb_instance.embed.return_value = MagicMock(shape=(5, 384))
        mock_emb_instance.dimension = 384
        MockEmb.return_value = mock_emb_instance

        mock_vs_instance = MagicMock()
        type(mock_vs_instance).size = PropertyMock(return_value=5)
        MockVS.return_value = mock_vs_instance

        MockBM25.return_value = MagicMock()

        agent = PDFRAGAgent(
            llm_client=self._mock_llm(),
            pdf_tool=self._mock_pdf_tool(),
        )

        stats = agent.ingest("/test/sample.pdf")

        assert stats["total_pages"] == 3
        assert stats["title"] == "Test Paper"
        assert stats["total_chunks"] > 0

    def test_format_qa_response(self):
        """format_response should produce readable QA output."""
        agent = PDFRAGAgent(llm_client=self._mock_llm())
        response = PDFRAGResponse(
            query="What methodology?",
            answer="Case studies were used.",
            confidence="high",
            relevant_pages=[2],
            evidence=["Case studies combined with experimental evaluations"],
            chunks_used=3,
        )
        formatted = agent.format_response(response)
        assert "Case studies" in formatted
        assert "high" in formatted
        assert "3" in formatted

    def test_format_summary_response(self):
        """format_response should produce readable summary output."""
        agent = PDFRAGAgent(llm_client=self._mock_llm())
        response = PDFSummaryResponse(
            filepath="/test/paper.pdf",
            summary="The paper discusses AI automation.",
            key_topics=["AI", "Automation"],
            key_findings=["30% productivity improvement"],
            total_pages=10,
            total_chunks=25,
        )
        formatted = agent.format_response(response)
        assert "AI automation" in formatted
        assert "AI" in formatted
        assert "Automation" in formatted
