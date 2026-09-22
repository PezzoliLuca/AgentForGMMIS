"""High-performance PDF manuscript parser using PyMuPDF (fitz)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import pymupdf  # PyMuPDF


@dataclass
class ParsedManuscript:
    """Represents the structured content extracted from an academic PDF."""

    file_path: str
    total_pages: int
    title: str = ""
    authors: str = ""
    pages_text: Dict[int, str] = field(default_factory=dict)
    table_pages: List[int] = field(default_factory=list)
    abstract: str = ""

    def get_full_text(self, max_pages: Optional[int] = None) -> str:
        """Concatenates pages text up to max_pages."""
        pages = sorted(self.pages_text.keys())
        if max_pages:
            pages = pages[:max_pages]
        return "\n\n--- PAGE BREAK ---\n\n".join(
            f"[Page {p}]\n{self.pages_text[p]}" for p in pages
        )

    def get_table_dense_text(self) -> str:
        """Returns concatenated text from pages identified as containing evaluation tables."""
        if not self.table_pages:
            # Fallback to last half of document where results/tables usually reside
            mid = self.total_pages // 2
            pages = [p for p in sorted(self.pages_text.keys()) if p >= mid]
        else:
            pages = self.table_pages

        return "\n\n--- TABLE PAGE ---\n\n".join(
            f"[Page {p}]\n{self.pages_text[p]}" for p in pages
        )


class PDFParser:
    """Parses scientific PDF manuscripts into structured text and table locations."""

    @staticmethod
    def parse(pdf_path: Union[str, Path]) -> ParsedManuscript:
        """Extracts text, metadata, and table locations from a PDF."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        doc = pymupdf.open(str(path))
        total_pages = len(doc)
        pages_text: Dict[int, str] = {}
        table_pages: List[int] = []
        abstract_text = ""

        # Metadata from doc
        meta = doc.metadata or {}
        doc_title = meta.get("title", "").strip()
        doc_author = meta.get("author", "").strip()

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]
            text = page.get_text("text").strip()
            pages_text[page_num] = text

            # Heuristic for table detection
            text_lower = text.lower()
            if any(k in text_lower for k in ("table 1", "table 2", "table 3", "table i", "dice score", "dsc", "mean dice")):
                table_pages.append(page_num)

            # Heuristic for abstract (usually page 1 or 2)
            if page_num in (1, 2) and "abstract" in text_lower and not abstract_text:
                lines = text.split("\n")
                capturing = False
                abs_lines = []
                for line in lines:
                    if "abstract" in line.lower():
                        capturing = True
                        continue
                    if capturing:
                        if any(term in line.lower() for term in ("introduction", "1. introduction", "keywords:")):
                            break
                        abs_lines.append(line)
                if abs_lines:
                    abstract_text = " ".join(abs_lines).strip()

        # Fallback for doc_title if empty: extract first prominent line of page 1
        if not doc_title and 1 in pages_text:
            first_lines = [l.strip() for l in pages_text[1].split("\n") if len(l.strip()) > 10]
            if first_lines:
                doc_title = first_lines[0]

        doc.close()

        return ParsedManuscript(
            file_path=str(path),
            total_pages=total_pages,
            title=doc_title,
            authors=doc_author,
            pages_text=pages_text,
            table_pages=table_pages,
            abstract=abstract_text,
        )
