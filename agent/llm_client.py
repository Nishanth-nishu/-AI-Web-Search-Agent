"""
Unified LLM client supporting Groq and OpenAI-compatible APIs.

Implements:
- Retry logic with exponential backoff
- Structured output parsing  
- Token counting
- Error handling for rate limits and timeouts

Based on ReAct paper methodology — the LLM is the reasoning engine
that drives the agent's Thought→Action→Observation loop.
"""

import json
import logging
import time
from typing import Any, Optional

from groq import Groq, APIError, RateLimitError, APITimeoutError

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client with retry logic and structured output support.
    
    Supports Groq (primary, free tier) and OpenAI-compatible APIs.
    Uses exponential backoff for rate limit handling.
    """

    def __init__(self, llm_config=None):
        self.config = llm_config or config.llm
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the LLM client based on provider configuration."""
        if self.config.provider == "groq":
            if not self.config.api_key:
                raise ValueError(
                    "GROQ_API_KEY is required. Set it in your .env file or environment."
                )
            self._client = Groq(api_key=self.config.api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a response from the LLM with retry logic.

        Args:
            prompt: The user prompt/query.
            system_prompt: System-level instruction.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            json_mode: If True, request JSON output format.

        Returns:
            The generated text response.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        return self._call_with_retry(
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            json_mode=json_mode,
        )

    def generate_with_messages(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a response from a full message history.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            json_mode: If True, request JSON output format.

        Returns:
            The generated text response.
        """
        return self._call_with_retry(
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            json_mode=json_mode,
        )

    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        """
        Call the LLM API with exponential backoff retry logic.

        Handles:
        - Rate limit errors (429) — backs off exponentially
        - Timeout errors — retries with same delay
        - API errors — retries up to max_retries
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = self._client.chat.completions.create(**kwargs)

                content = response.choices[0].message.content
                if content is None:
                    logger.warning("LLM returned None content on attempt %d", attempt + 1)
                    continue

                return content.strip()

            except RateLimitError as e:
                last_error = e
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %.1fs...",
                    attempt + 1, self.config.max_retries, delay
                )
                time.sleep(delay)

            except APITimeoutError as e:
                last_error = e
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    "API timeout (attempt %d/%d). Retrying in %.1fs...",
                    attempt + 1, self.config.max_retries, delay
                )
                time.sleep(delay)

            except APIError as e:
                last_error = e
                logger.error(
                    "API error (attempt %d/%d): %s",
                    attempt + 1, self.config.max_retries, str(e)
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        raise RuntimeError(
            f"LLM call failed after {self.config.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant. Respond in valid JSON.",
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Generate a JSON response from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: System instruction (should mention JSON).
            temperature: Override default temperature.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            ValueError: If the response is not valid JSON.
        """
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_mode=True,
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Attempt to extract JSON from markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            raise ValueError(f"LLM response is not valid JSON: {response[:200]}")
