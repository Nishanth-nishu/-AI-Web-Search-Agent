"""
AI Web Search Agent using the ReAct paradigm.

Based on Paper #1 (ReAct: Synergizing Reasoning and Acting, arXiv:2210.03629):
- The agent interleaves reasoning (Thought) with actions (Search, Extract)
- Each cycle: Thought → Action → Observation → (repeat or Answer)
- Supports query reformulation for better search results
- Returns grounded answers with source attribution

The agent demonstrates:
1. LLM integration with structured prompting
2. External tool usage (web search)
3. Response synthesis from retrieved data
4. Source reference tracking
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from agent.llm_client import LLMClient
from agent.tools.search_tool import WebSearchTool, SearchResult

logger = logging.getLogger(__name__)


# ─── System Prompts ──────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are an intelligent web search agent. You follow the ReAct (Reasoning + Acting) paradigm.

For each user query, you will:
1. THINK about what information is needed
2. Decide what search queries to use
3. Analyze the search results
4. Synthesize a comprehensive answer

You MUST respond in the following JSON format:
{
    "thought": "Your reasoning about the query and what to search for",
    "search_queries": ["query1", "query2"],
    "needs_more_search": false
}

Guidelines:
- Generate 1-3 focused search queries that would best answer the user's question
- Make queries specific and varied to cover different angles
- If the original query is clear enough, a single refined query is fine
"""

SYNTHESIS_SYSTEM_PROMPT = """You are a helpful AI assistant that synthesizes information from web search results to answer questions.

You MUST:
1. Base your answer ONLY on the provided search results
2. Be specific and factual
3. Cite sources where possible
4. If the search results don't contain enough information, say so clearly
5. Do NOT make up information that isn't in the search results

Respond in this JSON format:
{
    "answer": "Your comprehensive answer based on the search results",
    "confidence": "high/medium/low",
    "key_facts": ["fact1", "fact2"]
}
"""


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class WebSearchResponse:
    """Final response from the web search agent."""
    query: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    confidence: str = "medium"
    search_queries_used: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)


# ─── Agent ───────────────────────────────────────────────────────────────────

class WebSearchAgent:
    """
    ReAct-style web search agent.
    
    Implements the Thought→Action→Observation loop from Paper #1 to:
    1. Analyze the user's query
    2. Generate targeted search queries
    3. Execute web searches and extract content
    4. Synthesize a grounded answer with source attribution
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        search_tool: Optional[WebSearchTool] = None,
    ):
        self.llm = llm_client or LLMClient()
        self.search = search_tool or WebSearchTool()

    def answer(self, query: str) -> WebSearchResponse:
        """
        Answer a natural language query using web search.

        This is the main entry point for the ReAct agent loop:
        1. Thought: Analyze query and plan search strategy
        2. Action: Execute web searches
        3. Observation: Process search results
        4. Synthesis: Generate grounded answer

        Args:
            query: The user's natural language question.

        Returns:
            WebSearchResponse with answer, sources, and metadata.
        """
        if not query or not query.strip():
            return WebSearchResponse(
                query=query,
                answer="Please provide a valid question.",
                confidence="low",
            )

        logger.info("="*60)
        logger.info("WebSearchAgent processing query: %s", query)
        logger.info("="*60)

        # ── Step 1: Thought — Analyze query and plan search strategy ──
        logger.info("Step 1: THOUGHT — Analyzing query...")
        search_plan = self._think(query)
        search_queries = search_plan.get("search_queries", [query])
        
        logger.info("  Thought: %s", search_plan.get("thought", ""))
        logger.info("  Search queries: %s", search_queries)

        # ── Step 2: Action — Execute web searches ──
        logger.info("Step 2: ACTION — Executing web searches...")
        all_results = self._act(search_queries)
        
        logger.info("  Retrieved %d total results", len(all_results))

        if not all_results:
            # Retry with the original query if reformulated queries failed
            logger.info("  No results from planned queries, retrying with original...")
            all_results = self.search.search_and_extract(query)

        if not all_results:
            return WebSearchResponse(
                query=query,
                answer="I couldn't find any relevant results for your query. Please try rephrasing your question.",
                confidence="low",
                search_queries_used=search_queries,
            )

        # ── Step 3: Observation — Process and organize results ──
        logger.info("Step 3: OBSERVATION — Processing %d results...", len(all_results))
        context = self._observe(all_results)

        # ── Step 4: Synthesis — Generate grounded answer ──
        logger.info("Step 4: SYNTHESIS — Generating answer...")
        response = self._synthesize(query, context, all_results, search_queries)
        
        logger.info("  Confidence: %s", response.confidence)
        logger.info("  Sources: %d", len(response.sources))

        return response

    def _think(self, query: str) -> dict:
        """
        THOUGHT phase: Analyze the query and plan search strategy.
        
        Uses the LLM to understand the query intent and generate
        targeted search queries.
        """
        try:
            result = self.llm.generate_json(
                prompt=f"User query: {query}\n\nAnalyze this query and generate search queries that would best find the answer.",
                system_prompt=REACT_SYSTEM_PROMPT,
                temperature=0.3,
            )
            
            # Validate response
            if "search_queries" not in result or not result["search_queries"]:
                result["search_queries"] = [query]
            
            return result

        except Exception as e:
            logger.warning("Think phase failed: %s. Using original query.", e)
            return {
                "thought": f"Using original query due to error: {e}",
                "search_queries": [query],
            }

    def _act(self, search_queries: list[str]) -> list[SearchResult]:
        """
        ACTION phase: Execute web searches for each planned query.
        
        Deduplicates results across multiple queries.
        Extracts full page content from top results.
        """
        all_results = []
        seen_urls = set()

        for sq in search_queries[:3]:  # Max 3 queries
            try:
                results = self.search.search_and_extract(sq, max_results=5)
                for result in results:
                    if result.url not in seen_urls:
                        seen_urls.add(result.url)
                        all_results.append(result)
            except Exception as e:
                logger.warning("Search failed for '%s': %s", sq, e)

        return all_results

    def _observe(self, results: list[SearchResult]) -> str:
        """
        OBSERVATION phase: Process and organize search results into context.
        
        Formats results with content, titles, and URLs for the synthesis LLM.
        """
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result.body if result.body else result.snippet
            if not content:
                continue

            # Truncate very long content
            if len(content) > 2000:
                content = content[:2000] + "..."

            context_parts.append(
                f"[Source {i}] {result.title}\n"
                f"URL: {result.url}\n"
                f"Content: {content}\n"
            )

        return "\n---\n".join(context_parts)

    def _synthesize(
        self,
        query: str,
        context: str,
        results: list[SearchResult],
        search_queries: list[str],
    ) -> WebSearchResponse:
        """
        SYNTHESIS phase: Generate a grounded answer using the LLM.
        
        Provides all search results as context and asks the LLM to
        synthesize a comprehensive, factual answer with source citations.
        """
        prompt = (
            f"User Question: {query}\n\n"
            f"Search Results:\n{context}\n\n"
            f"Based on the above search results, provide a comprehensive answer "
            f"to the user's question. Be specific and cite the sources."
        )

        try:
            result = self.llm.generate_json(
                prompt=prompt,
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                temperature=0.2,
            )

            answer = result.get("answer", "Unable to generate answer.")
            confidence = result.get("confidence", "medium")
            key_facts = result.get("key_facts", [])

        except Exception as e:
            logger.warning("JSON synthesis failed: %s. Trying plain text.", e)
            # Fallback to plain text generation
            answer = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant. Synthesize the search results to answer the question. Be factual and cite sources.",
                temperature=0.2,
            )
            confidence = "medium"
            key_facts = []

        # Build sources list
        sources = []
        for r in results:
            if r.url and (r.body or r.snippet):
                sources.append({
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet[:200] if r.snippet else "",
                })

        return WebSearchResponse(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence,
            search_queries_used=search_queries,
            key_facts=key_facts,
        )

    def format_response(self, response: WebSearchResponse) -> str:
        """Format a WebSearchResponse for display."""
        output = []
        output.append(f"Question: {response.query}")
        output.append("")
        output.append(f"Answer: {response.answer}")
        output.append("")

        if response.sources:
            output.append("Sources:")
            for i, source in enumerate(response.sources, 1):
                output.append(f"  [{i}] {source['title']}")
                output.append(f"      {source['url']}")
            output.append("")

        if response.key_facts:
            output.append("Key Facts:")
            for fact in response.key_facts:
                output.append(f"  • {fact}")
            output.append("")

        output.append(f"Confidence: {response.confidence}")

        return "\n".join(output)
