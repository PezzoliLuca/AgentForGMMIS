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
        # and highest-ranked evaluation table pages up to a safe 35,000 character budget
        pages_to_include = [p for p in (1, 2) if p in manuscript.pages_text]
        current_len = sum(len(manuscript.pages_text[p]) for p in pages_to_include)

        for p in (manuscript.table_pages or []):
            if p not in pages_to_include and p in manuscript.pages_text:
                page_len = len(manuscript.pages_text[p])
                if current_len + page_len > 35000:
                    continue
                pages_to_include.append(p)
                current_len += page_len

        selected_pages = sorted(pages_to_include)

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
