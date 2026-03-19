"""
Text processing utilities — chunking, cleaning, and token counting.

Implements Recursive Character Text Splitting (Paper #7):
- Hierarchical delimiter-based splitting: \\n\\n → \\n → '. ' → ' '
- Configurable chunk size (default 512 tokens) with overlap (50 tokens)
- Token-aware splitting using tiktoken
- Chunk metadata tracking (index, page number, offsets)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    text: str
    index: int
    start_char: int = 0
    end_char: int = 0
    page_number: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.text)


class RecursiveTextSplitter:
    """
    Recursive character text splitter with token-aware splitting.
    
    Based on the methodology from Paper #7 (LangChain/industry best practices):
    - Tries to split by paragraph (\\n\\n) first
    - Falls back to line breaks (\\n), then sentences ('. '), then words (' ')
    - Maintains overlap between chunks for context continuity
    """

    def __init__(self, chunking_config=None):
        cfg = chunking_config or config.chunking
        self.chunk_size = cfg.chunk_size
        self.chunk_overlap = cfg.chunk_overlap
        self.separators = cfg.separators
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load tiktoken tokenizer."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                logger.warning("tiktoken not available, using character-based counting")
                self._tokenizer = None
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback to char/4."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4

    def split(
        self,
        text: str,
        page_numbers: Optional[dict[int, int]] = None,
    ) -> list[TextChunk]:
        """
        Split text into chunks using recursive character splitting.

        Args:
            text: The full text to split.
            page_numbers: Optional mapping of character offset → page number.

        Returns:
            List of TextChunk objects with metadata.
        """
        if not text or not text.strip():
            return []

        # Clean the text
        text = self._clean_text(text)

        # Split recursively
        raw_chunks = self._recursive_split(text, self.separators)

        # Merge small chunks and apply overlap
        merged_chunks = self._merge_with_overlap(raw_chunks)

        # Create TextChunk objects with metadata
        chunks = []
        char_offset = 0
        for idx, chunk_text in enumerate(merged_chunks):
            start_char = text.find(chunk_text[:50], max(0, char_offset - 100))
            if start_char == -1:
                start_char = char_offset

            page_num = None
            if page_numbers:
                page_num = self._get_page_number(start_char, page_numbers)

            chunks.append(TextChunk(
                text=chunk_text,
                index=idx,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                page_number=page_num,
                metadata={"token_count": self.count_tokens(chunk_text)},
            ))
            char_offset = start_char + len(chunk_text)

        logger.info(
            "Split text into %d chunks (avg %d tokens/chunk)",
            len(chunks),
            sum(c.metadata.get("token_count", 0) for c in chunks) // max(len(chunks), 1),
        )

        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using hierarchical separators.

        Tries the first separator; if chunks are too large, uses the next
        separator on the oversized chunks, and so on.
        """
        if not separators:
            # Base case: just split by chunk_size characters
            return self._split_by_size(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        parts = text.split(separator)
        
        result = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if self.count_tokens(part) <= self.chunk_size:
                result.append(part)
            elif remaining_separators:
                # Chunk too large — recurse with finer separator
                sub_chunks = self._recursive_split(part, remaining_separators)
                result.extend(sub_chunks)
            else:
                # No more separators — split by size
                result.extend(self._split_by_size(part))

        return result

    def _split_by_size(self, text: str) -> list[str]:
        """Split text into fixed-size chunks when no separator works."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_tokens = 0

        for word in words:
            word_tokens = self.count_tokens(word)
            if current_tokens + word_tokens > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            current_chunk.append(word)
            current_tokens += word_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        """
        Merge small consecutive chunks and apply overlap between chunks.
        
        - Merges consecutive chunks that together are under chunk_size
        - Adds overlap from the end of previous chunk to start of next
        """
        if not chunks:
            return []

        # Phase 1: Merge small consecutive chunks
        merged = []
        current = chunks[0]

        for i in range(1, len(chunks)):
            combined = current + " " + chunks[i]
            if self.count_tokens(combined) <= self.chunk_size:
                current = combined
            else:
                merged.append(current)
                current = chunks[i]
        merged.append(current)

        # Phase 2: Apply overlap
        if self.chunk_overlap <= 0 or len(merged) <= 1:
            return merged

        overlapped = [merged[0]]
        for i in range(1, len(merged)):
            prev_text = merged[i - 1]
            prev_words = prev_text.split()
            
            # Calculate overlap word count
            overlap_words = []
            overlap_tokens = 0
            for word in reversed(prev_words):
                word_tokens = self.count_tokens(word)
                if overlap_tokens + word_tokens > self.chunk_overlap:
                    break
                overlap_words.insert(0, word)
                overlap_tokens += word_tokens

            if overlap_words:
                overlapped.append(" ".join(overlap_words) + " " + merged[i])
            else:
                overlapped.append(merged[i])

        return overlapped

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text content."""
        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)
        # Remove null bytes and other control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return text.strip()

    @staticmethod
    def _get_page_number(char_offset: int, page_map: dict[int, int]) -> int:
        """Get page number for a given character offset."""
        page_num = 1
        for offset, pnum in sorted(page_map.items()):
            if char_offset >= offset:
                page_num = pnum
            else:
                break
        return page_num


def build_page_offset_map(pages: list) -> dict[int, int]:
    """
    Build a character offset → page number mapping from PDFDocument pages.
    
    Args:
        pages: List of PageContent objects from PDFDocument.
        
    Returns:
        Dict mapping character offsets to page numbers.
    """
    page_map = {}
    current_offset = 0
    for page in pages:
        page_map[current_offset] = page.page_number
        current_offset += len(page.text) + 2  # +2 for the "\n\n" separator
    return page_map
