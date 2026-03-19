"""
PDF processing tool for text extraction and metadata parsing.

Implements:
- Text extraction with layout preservation (PyMuPDF)  
- Fallback to pdfplumber for complex layouts
- Metadata extraction (title, author, page count)
- Table detection
- OCR-ready hook for scanned documents

This tool handles the document ingestion phase of the RAG pipeline (Paper #2).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Content extracted from a single PDF page."""
    page_number: int
    text: str
    tables: list[str] = field(default_factory=list)


@dataclass 
class PDFDocument:
    """Structured representation of an extracted PDF document."""
    filepath: str
    title: str = ""
    author: str = ""
    total_pages: int = 0
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Get the full text content of the document."""
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    @property
    def is_empty(self) -> bool:
        """Check if the document has any extractable text."""
        return not any(page.text.strip() for page in self.pages)


class PDFTool:
    """
    PDF processing tool for text extraction and metadata parsing.
    
    Uses PyMuPDF (fitz) as the primary extraction engine with
    pdfplumber as fallback for complex layouts.
    """

    def extract(self, filepath: str) -> PDFDocument:
        """
        Extract text and metadata from a PDF file.

        Args:
            filepath: Path to the PDF file.

        Returns:
            PDFDocument with extracted content and metadata.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ValueError: If the file is not a valid PDF.
        """
        filepath = os.path.abspath(filepath)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found: {filepath}")

        if not filepath.lower().endswith(".pdf"):
            raise ValueError(f"Not a PDF file: {filepath}")

        # Try PyMuPDF first (faster, better layout preservation)
        doc = self._extract_with_pymupdf(filepath)

        # Fallback to pdfplumber if PyMuPDF yields empty results
        if doc.is_empty:
            logger.info("PyMuPDF returned empty text, trying pdfplumber fallback")
            doc = self._extract_with_pdfplumber(filepath)

        # If still empty, it might be a scanned document
        if doc.is_empty:
            logger.warning(
                "No text extracted from %s. It may be a scanned/image-only PDF.",
                filepath,
            )

        return doc

    def _extract_with_pymupdf(self, filepath: str) -> PDFDocument:
        """Extract text using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF

            pdf = fitz.open(filepath)
            
            # Extract metadata
            meta = pdf.metadata or {}
            doc = PDFDocument(
                filepath=filepath,
                title=meta.get("title", "") or os.path.basename(filepath),
                author=meta.get("author", ""),
                total_pages=len(pdf),
                metadata={
                    "subject": meta.get("subject", ""),
                    "creator": meta.get("creator", ""),
                    "producer": meta.get("producer", ""),
                    "creation_date": meta.get("creationDate", ""),
                },
            )

            # Extract text page by page
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                
                # Extract text with layout preservation
                text = page.get_text("text")
                
                # Extract tables if present
                tables = self._extract_tables_pymupdf(page)

                doc.pages.append(PageContent(
                    page_number=page_num + 1,
                    text=text.strip(),
                    tables=tables,
                ))

            pdf.close()
            logger.info(
                "PyMuPDF extracted %d pages from %s", doc.total_pages, filepath
            )
            return doc

        except ImportError:
            logger.error("PyMuPDF (fitz) not installed")
            return PDFDocument(filepath=filepath)
        except Exception as e:
            logger.error("PyMuPDF extraction failed: %s", str(e))
            return PDFDocument(filepath=filepath)

    def _extract_with_pdfplumber(self, filepath: str) -> PDFDocument:
        """Fallback extraction using pdfplumber."""
        try:
            import pdfplumber

            doc = PDFDocument(
                filepath=filepath,
                title=os.path.basename(filepath),
            )

            with pdfplumber.open(filepath) as pdf:
                doc.total_pages = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    
                    # Extract tables
                    tables = []
                    for table in (page.extract_tables() or []):
                        if table:
                            table_str = "\n".join(
                                " | ".join(str(cell or "") for cell in row)
                                for row in table
                            )
                            tables.append(table_str)

                    doc.pages.append(PageContent(
                        page_number=page_num + 1,
                        text=text.strip(),
                        tables=tables,
                    ))

            logger.info(
                "pdfplumber extracted %d pages from %s", doc.total_pages, filepath
            )
            return doc

        except ImportError:
            logger.error("pdfplumber not installed")
            return PDFDocument(filepath=filepath)
        except Exception as e:
            logger.error("pdfplumber extraction failed: %s", str(e))
            return PDFDocument(filepath=filepath)

    def _extract_tables_pymupdf(self, page) -> list[str]:
        """Attempt to extract tables from a PyMuPDF page."""
        tables = []
        try:
            # PyMuPDF >= 1.23.0 has built-in table detection
            if hasattr(page, "find_tables"):
                found_tables = page.find_tables()
                for table in found_tables:
                    table_data = table.extract()
                    if table_data:
                        table_str = "\n".join(
                            " | ".join(str(cell or "") for cell in row)
                            for row in table_data
                        )
                        tables.append(table_str)
        except Exception as e:
            logger.debug("Table extraction failed: %s", e)
        return tables

    def get_page_text(self, filepath: str, page_numbers: list[int]) -> str:
        """
        Extract text from specific pages.

        Args:
            filepath: Path to the PDF file.
            page_numbers: List of 1-indexed page numbers.

        Returns:
            Concatenated text from the specified pages.
        """
        doc = self.extract(filepath)
        texts = []
        for page in doc.pages:
            if page.page_number in page_numbers:
                texts.append(page.text)
        return "\n\n".join(texts)
