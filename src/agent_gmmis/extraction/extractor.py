"""ExtractionAgent: Ingests scientific manuscripts and extracts structured benchmark records."""

from pathlib import Path
from typing import Optional, Union
from agent_gmmis.extraction.pdf_parser import PDFParser, ParsedManuscript
from agent_gmmis.extraction.prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT_TEMPLATE
from agent_gmmis.extraction.schemas import ExtractedPaperData
from agent_gmmis.llm.client import LLMClient


class ExtractionAgent:
    """Autonomous agent for reading scientific manuscripts and extracting benchmark data."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.parser = PDFParser()

    def process_pdf(self, pdf_path: Union[str, Path]) -> ExtractedPaperData:
        """Parses a PDF manuscript and runs LLM structured extraction."""
        manuscript = self.parser.parse(pdf_path)
        return self.process_manuscript(manuscript)

    def process_manuscript(self, manuscript: ParsedManuscript) -> ExtractedPaperData:
        """Extracts structured benchmark information from a parsed manuscript."""
        # Compose content prioritizing key introduction (pages 1-2 for metadata, backbone, abstract)
        # and the highest-scoring evaluation table pages (up to 8 pages ranked by metrics and table density)
        top_table_pages = manuscript.table_pages[:8] if manuscript.table_pages else []
        selected_pages = sorted(set([1, 2] + top_table_pages))

        content_snippet = "\n\n--- PAGE BREAK ---\n\n".join(
            f"[Page {p}]\n{manuscript.pages_text[p]}"
            for p in selected_pages
            if p in manuscript.pages_text
        )

        prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
            title=manuscript.title,
            total_pages=manuscript.total_pages,
            abstract=manuscript.abstract or "(Extracted from document)",
            content=content_snippet,
        )

        extracted_data = self.llm.generate_structured(
            prompt=prompt,
            response_model=ExtractedPaperData,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )

        return extracted_data
