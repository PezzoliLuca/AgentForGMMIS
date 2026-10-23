"""Manuscript ingestion, parsing, structured extraction, and evaluation."""

from agent_gmmis.extraction.dual_coordinator import DualExtractionCoordinator
from agent_gmmis.extraction.evaluator import ExtractionEvaluator
from agent_gmmis.extraction.extractor import ExtractionAgent
from agent_gmmis.extraction.pdf_parser import ParsedManuscript, PDFParser
from agent_gmmis.extraction.schemas import (
    ExtractedPaperData,
    ExtractionThoughtProcess,
    RawExtractedScore,
)

__all__ = [
    "PDFParser",
    "ParsedManuscript",
    "ExtractionThoughtProcess",
    "RawExtractedScore",
    "ExtractedPaperData",
    "ExtractionAgent",
    "ExtractionEvaluator",
    "DualExtractionCoordinator",
]
