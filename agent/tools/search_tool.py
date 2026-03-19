"""
Web search tool using DuckDuckGo with optional Tavily fallback.

Implements:
- DuckDuckGo search (no API key required)
- Full page content extraction via trafilatura
- Result deduplication by URL
- Rate limiting to avoid being blocked

This is the primary "Action" tool in the ReAct agent loop (Paper #1).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured search result."""
    title: str
    url: str
    snippet: str
    body: str = ""  # Full extracted content (when available)
    relevance_score: float = 0.0


class WebSearchTool:
    """
    Web search tool with DuckDuckGo primary and optional Tavily fallback.
    
    Returns structured search results with optional full-page content extraction.
    Implements deduplication and rate limiting.
    """

    def __init__(self, search_config=None):
        self.config = search_config or config.search

    def search(self, query: str, max_results: Optional[int] = None) -> list[SearchResult]:
        """
        Search the web for the given query.

        Args:
            query: Natural language search query.
            max_results: Override default max results count.

        Returns:
            List of SearchResult objects, deduplicated by URL.
        """
        max_results = max_results or self.config.max_results

        if not query or not query.strip():
            logger.warning("Empty search query provided")
            return []

        # Try DuckDuckGo first (free, no API key)
        results = self._search_duckduckgo(query, max_results)

        # Fallback to Tavily if DuckDuckGo fails and API key is available
        if not results and self.config.tavily_api_key:
            logger.info("DuckDuckGo returned no results, trying Tavily fallback")
            results = self._search_tavily(query, max_results)

        # Deduplicate by URL
        results = self._deduplicate_results(results)

        return results[:max_results]

    def search_and_extract(
        self, query: str, max_results: Optional[int] = None
    ) -> list[SearchResult]:
        """
        Search the web and extract full page content from top results.

        Args:
            query: Natural language search query.
            max_results: Override default max results count.

        Returns:
            List of SearchResult objects with body content populated.
        """
        results = self.search(query, max_results)

        for result in results:
            try:
                body = self._extract_page_content(result.url)
                if body:
                    result.body = body
                time.sleep(self.config.rate_limit_delay)
            except Exception as e:
                logger.warning("Failed to extract content from %s: %s", result.url, e)
                result.body = result.snippet  # Fallback to snippet

        return results

    def _search_duckduckgo(self, query: str, max_results: int) -> list[SearchResult]:
        """Search using DuckDuckGo (no API key required)."""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                search_results = list(ddgs.text(
                    query,
                    max_results=max_results,
                    safesearch="moderate",
                ))

            for item in search_results:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", item.get("link", "")),
                    snippet=item.get("body", item.get("snippet", "")),
                ))

            logger.info("DuckDuckGo returned %d results for: %s", len(results), query)
            return results

        except ImportError:
            logger.error("duckduckgo-search package not installed")
            return []
        except Exception as e:
            logger.error("DuckDuckGo search failed: %s", str(e))
            return []

    def _search_tavily(self, query: str, max_results: int) -> list[SearchResult]:
        """Fallback search using Tavily API."""
        try:
            import requests

            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.config.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "advanced",
                },
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    relevance_score=item.get("score", 0.0),
                ))

            logger.info("Tavily returned %d results for: %s", len(results), query)
            return results

        except Exception as e:
            logger.error("Tavily search failed: %s", str(e))
            return []

    def _extract_page_content(self, url: str) -> str:
        """
        Extract clean text content from a web page using trafilatura.

        Falls back to basic HTML parsing if trafilatura fails.
        """
        try:
            import trafilatura

            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                content = trafilatura.extract(
                    downloaded,
                    include_links=False,
                    include_tables=True,
                    no_fallback=False,
                )
                if content:
                    # Truncate to max content length
                    return content[: self.config.max_content_length]

        except ImportError:
            logger.warning("trafilatura not installed, using snippet only")
        except Exception as e:
            logger.warning("Content extraction failed for %s: %s", url, e)

        return ""

    def _deduplicate_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate results based on normalized URL."""
        seen_urls = set()
        unique_results = []

        for result in results:
            normalized = self._normalize_url(result.url)
            if normalized and normalized not in seen_urls:
                seen_urls.add(normalized)
                unique_results.append(result)

        return unique_results

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL for deduplication."""
        try:
            parsed = urlparse(url)
            # Remove trailing slashes and fragments
            path = parsed.path.rstrip("/")
            return f"{parsed.scheme}://{parsed.netloc}{path}"
        except Exception:
            return url
