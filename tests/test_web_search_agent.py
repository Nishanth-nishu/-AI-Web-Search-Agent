"""
Tests for the Web Search Agent.

Tests cover:
- Query processing and plan generation (mocked LLM)
- Search execution with mocked results
- Response synthesis
- Source attribution
- Error handling (empty results, LLM failure)
"""

import pytest
from unittest.mock import patch, MagicMock
from agent.web_search_agent import WebSearchAgent, WebSearchResponse
from agent.tools.search_tool import SearchResult


class TestWebSearchAgent:
    """Tests for the ReAct web search agent."""

    def _mock_llm(self):
        """Create a mock LLM client."""
        mock = MagicMock()
        mock.generate_json.return_value = {
            "thought": "I need to search for MacBook specs",
            "search_queries": ["MacBook Pro 2024 specs"],
        }
        mock.generate.return_value = "The MacBook Pro features M4 chips."
        return mock

    def _mock_search(self):
        """Create a mock search tool."""
        mock = MagicMock()
        mock.search_and_extract.return_value = [
            SearchResult(
                title="MacBook Pro Specs",
                url="https://apple.com/macbook-pro",
                snippet="The new MacBook Pro features M4 chip",
                body="The new MacBook Pro features the M4 chip with enhanced performance.",
            ),
            SearchResult(
                title="MacBook Air 2024",
                url="https://apple.com/macbook-air",
                snippet="MacBook Air now with M3",
                body="The MacBook Air has been updated with the M3 chip.",
            ),
        ]
        mock.search.return_value = mock.search_and_extract.return_value
        return mock

    def test_empty_query(self):
        """Empty queries should return low-confidence response."""
        agent = WebSearchAgent(llm_client=self._mock_llm())
        response = agent.answer("")
        assert response.confidence == "low"
        assert "valid question" in response.answer.lower()

    def test_answer_returns_response(self):
        """Should return a complete WebSearchResponse."""
        mock_llm = self._mock_llm()
        # Override synthesis response
        mock_llm.generate_json.side_effect = [
            # Think phase
            {"thought": "searching", "search_queries": ["MacBook specs"]},
            # Synthesis phase
            {
                "answer": "The MacBook Pro features M4 chips.",
                "confidence": "high",
                "key_facts": ["M4 chip", "Enhanced performance"],
            },
        ]

        agent = WebSearchAgent(
            llm_client=mock_llm,
            search_tool=self._mock_search(),
        )

        response = agent.answer("MacBook specs?")

        assert isinstance(response, WebSearchResponse)
        assert response.answer != ""
        assert len(response.sources) > 0
        assert response.query == "MacBook specs?"

    def test_no_results_fallback(self):
        """Should handle empty search results gracefully."""
        mock_llm = self._mock_llm()
        mock_search = MagicMock()
        mock_search.search_and_extract.return_value = []
        mock_search.search.return_value = []

        agent = WebSearchAgent(llm_client=mock_llm, search_tool=mock_search)
        response = agent.answer("something obscure")

        assert response.confidence == "low"
        assert "couldn't find" in response.answer.lower() or "no relevant" in response.answer.lower() or response.answer != ""

    def test_format_response(self):
        """format_response should produce readable output."""
        agent = WebSearchAgent(llm_client=self._mock_llm())
        response = WebSearchResponse(
            query="test?",
            answer="Test answer.",
            sources=[{"title": "Source 1", "url": "https://example.com", "snippet": "test"}],
            confidence="high",
        )
        formatted = agent.format_response(response)
        assert "Test answer." in formatted
        assert "https://example.com" in formatted
        assert "high" in formatted

    def test_think_phase_fallback(self):
        """Think phase should fall back to original query on error."""
        mock_llm = MagicMock()
        mock_llm.generate_json.side_effect = Exception("LLM Error")
        mock_llm.generate.return_value = "fallback answer"

        agent = WebSearchAgent(
            llm_client=mock_llm,
            search_tool=self._mock_search(),
        )
        # Should not crash
        response = agent.answer("test query")
        assert isinstance(response, WebSearchResponse)
