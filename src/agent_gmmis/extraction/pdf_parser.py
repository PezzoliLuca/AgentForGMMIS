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
    page_scores: Dict[int, int] = field(default_factory=dict)
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

        import re

        doc = pymupdf.open(str(path))
        total_pages = len(doc)
        pages_text: Dict[int, str] = {}
        page_scores: Dict[int, int] = {}
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

            # Intelligent scoring for benchmark tables and evaluation sections
            text_lower = text.lower()
            score = 0

            # 1. Table and figure mentions (e.g. Table 1, Table 4, Figure 7)
            tables = re.findall(r"\btable\s+(\d+|[ivx]+)\b", text_lower)
            figures = re.findall(r"\b(?:figure|fig\.?)\s+(\d+|[ivx]+)\b", text_lower)
            metrics = re.findall(r"\b(dice|dsc|mean dice|iou|jaccard|hd95|nsd)\b", text_lower)

            score += len(tables) * 6
            if len(metrics) > 0 and len(figures) > 0:
                # Evaluation figures reporting segmentation performance (e.g. Figure 7 in Merlin)
                score += len(figures) * 4 + len(metrics) * 6
            else:
                score += len(metrics) * 4

            # 2. Benchmark and comparison terminology
            bench = re.findall(r"\b(benchmark|comparison|baseline|sota|state-of-the-art|test set|validation set|cross-validation)\b", text_lower)
            score += len(bench) * 2

            # 3. Dense numerical reporting (e.g. 0.8420, 88.5%)
            decimals = re.findall(r"\b0\.\d{3,4}\b", text_lower)
            score += min(len(decimals), 10)

            # 4. Main manuscript priority: the primary methodology & evaluation tables reside in main body (pages <= 25)
            if page_num <= 25 and score > 0:
                score += 15

            page_scores[page_num] = score

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

        # Multi-page table continuation bonus:
        # If previous page has high evaluation density and current page has metrics, boost current page
        orig_scores = dict(page_scores)
        for p in range(2, total_pages + 1):
            if orig_scores.get(p - 1, 0) >= 30 and orig_scores.get(p, 0) > 0:
                page_scores[p] += 12

        # Extract table pages sorted by relevance score descending
        table_pages = [p for p, s in sorted(page_scores.items(), key=lambda x: x[1], reverse=True) if s > 0]

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
            page_scores=page_scores,
            abstract=abstract_text,
        )
