"""
Tests for web search agent tools.

Tests cover:
- SearchResult data model
- DuckDuckGo search (mocked)
- Content extraction (mocked)
- URL deduplication
- Empty query handling
- Error paths
"""

import pytest
from unittest.mock import patch, MagicMock
from agent.tools.search_tool import WebSearchTool, SearchResult


class TestSearchResult:
    """Tests for the SearchResult data model."""

    def test_creation(self):
        result = SearchResult(
            title="Test", url="https://example.com", snippet="test snippet"
        )
        assert result.title == "Test"
        assert result.url == "https://example.com"
        assert result.snippet == "test snippet"
        assert result.body == ""
        assert result.relevance_score == 0.0


class TestWebSearchTool:
    """Tests for the WebSearchTool."""

    def setup_method(self):
        self.tool = WebSearchTool()

    def test_empty_query_returns_empty(self):
        """Empty queries should return no results."""
        assert self.tool.search("") == []
        assert self.tool.search("   ") == []

    def test_url_normalization(self):
        """URLs should be normalized correctly."""
        assert self.tool._normalize_url("https://example.com/path/") == \
            "https://example.com/path"
        assert self.tool._normalize_url("https://example.com") == \
            "https://example.com"

    def test_deduplication(self):
        """Duplicate URLs should be removed."""
        results = [
            SearchResult(title="A", url="https://example.com/page", snippet="a"),
            SearchResult(title="B", url="https://example.com/page/", snippet="b"),
            SearchResult(title="C", url="https://other.com/page", snippet="c"),
        ]
        deduped = self.tool._deduplicate_results(results)
        assert len(deduped) == 2  # B is a dup of A (trailing slash)

    @patch("agent.tools.search_tool.WebSearchTool._search_duckduckgo")
    def test_search_returns_results(self, mock_ddg):
        """Search should return structured results."""
        mock_ddg.return_value = [
            SearchResult(
                title="MacBook Pro",
                url="https://apple.com/macbook",
                snippet="Latest M4 chip specs",
            ),
        ]
        results = self.tool.search("MacBook specs")
        assert len(results) == 1
        assert results[0].title == "MacBook Pro"

    @patch("agent.tools.search_tool.WebSearchTool._search_duckduckgo")
    def test_search_handles_ddg_failure(self, mock_ddg):
        """Should handle DuckDuckGo failures gracefully."""
        mock_ddg.return_value = []
        results = self.tool.search("test query")
        assert results == []

    @patch("agent.tools.search_tool.WebSearchTool._search_duckduckgo")
    @patch("agent.tools.search_tool.WebSearchTool._extract_page_content")
    def test_search_and_extract(self, mock_extract, mock_ddg):
        """search_and_extract should populate body field."""
        mock_ddg.return_value = [
            SearchResult(
                title="Test",
                url="https://example.com",
                snippet="snippet",
            ),
        ]
        mock_extract.return_value = "Full page content here"

        results = self.tool.search_and_extract("test query")
        assert len(results) == 1
        assert results[0].body == "Full page content here"


class TestPDFTool:
    """Tests for the PDF extraction tool."""

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing files."""
        from agent.tools.pdf_tool import PDFTool

        tool = PDFTool()
        with pytest.raises(FileNotFoundError):
            tool.extract("/nonexistent/file.pdf")

    def test_invalid_extension(self):
        """Should raise ValueError for non-PDF files."""
        from agent.tools.pdf_tool import PDFTool
        import tempfile
        import os

        tool = PDFTool()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a pdf")
            f.flush()
            try:
                with pytest.raises(ValueError, match="Not a PDF file"):
                    tool.extract(f.name)
            finally:
                os.unlink(f.name)
