"""ExtractionAgent: Ingests scientific manuscripts and extracts structured benchmark records."""

from pathlib import Path
from typing import List, Optional, Union
from agent_gmmis.extraction.pdf_parser import PDFParser, ParsedManuscript
from agent_gmmis.extraction.prompts import (
    BLIND_DISCREPANCY_SYSTEM_PROMPT,
    BLIND_DISCREPANCY_USER_TEMPLATE,
    BLIND_GRANULARITY_SYSTEM_PROMPT,
    BLIND_GRANULARITY_USER_TEMPLATE,
    BLIND_OMISSION_SYSTEM_PROMPT,
    BLIND_OMISSION_USER_TEMPLATE,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE,
)
from agent_gmmis.extraction.schemas import ExtractedPaperData, ReinterrogationScoreResponse
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

    def _build_context_content(
        self,
        manuscript: ParsedManuscript,
        focal_keywords: Optional[List[str]] = None,
        max_chars: int = 24000,
    ) -> str:
        """Composes a relevant text snippet prioritizing focal keywords, introductory pages,

        and evaluation table pages up to max_chars budget.
        """
        if not manuscript or not getattr(manuscript, "pages_text", None):
            return ""

        pages_to_include: List[int] = []
        current_len = 0

        if focal_keywords:
            clean_kw = [kw.lower().strip() for kw in focal_keywords if kw and len(kw.strip()) > 2]
            # 1. Pages mentioning focal keywords (dataset name, proposed model)
            for page_num, page_text in sorted(manuscript.pages_text.items()):
                p_lower = page_text.lower()
                if any(kw in p_lower for kw in clean_kw):
                    if page_num not in pages_to_include:
                        p_len = len(page_text)
                        if current_len + p_len <= max_chars:
                            pages_to_include.append(page_num)
                            current_len += p_len

            # 2. Add table pages
            for page_num in (manuscript.table_pages or []):
                if page_num not in pages_to_include and page_num in manuscript.pages_text:
                    p_len = len(manuscript.pages_text[page_num])
                    if current_len + p_len <= max_chars:
                        pages_to_include.append(page_num)
                        current_len += p_len

            # 3. Add intro pages (1, 2) if space remains
            for page_num in (1, 2):
                if page_num not in pages_to_include and page_num in manuscript.pages_text:
                    p_len = len(manuscript.pages_text[page_num])
                    if current_len + p_len <= max_chars:
                        pages_to_include.append(page_num)
                        current_len += p_len
        else:
            # Baseline behavior: pages 1 and 2 first
            for page_num in (1, 2):
                if page_num in manuscript.pages_text:
                    pages_to_include.append(page_num)
                    current_len += len(manuscript.pages_text[page_num])

            for page_num in (manuscript.table_pages or []):
                if page_num not in pages_to_include and page_num in manuscript.pages_text:
                    p_len = len(manuscript.pages_text[page_num])
                    if current_len + p_len > max_chars:
                        continue
                    pages_to_include.append(page_num)
                    current_len += p_len

        selected_pages = sorted(pages_to_include)
        return "\n\n--- PAGE BREAK ---\n\n".join(
            f"[Page {p}]\n{manuscript.pages_text[p]}"
            for p in selected_pages
            if p in manuscript.pages_text
        )

    def process_manuscript(self, manuscript: ParsedManuscript) -> ExtractedPaperData:
        """Extracts structured benchmark information from a parsed manuscript."""
        content_snippet = self._build_context_content(manuscript)

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

    def reinterrogate_discrepancy(
        self,
        manuscript: ParsedManuscript,
        dataset_name: str,
        organ_or_task: str,
        previous_score: Optional[float],
        previous_prompt: Optional[str],
        proposed_model_id: str,
    ) -> ReinterrogationScoreResponse:
        """Re-interrogates the agent on a disputed benchmark using a double-blind challenge.

        Enforces epistemic independence: does NOT reveal the competitor agent's score or reasoning.
        """
        content = self._build_context_content(
            manuscript,
            focal_keywords=[dataset_name, proposed_model_id, organ_or_task],
        )
        prompt = BLIND_DISCREPANCY_USER_TEMPLATE.format(
            dataset_name=dataset_name,
            organ_or_task=organ_or_task or "Overall / Target Organ",
            previous_score=f"{previous_score:.4f}" if previous_score is not None else "None (Omitted or qualitative)",
            previous_prompt=previous_prompt or "Unspecified",
            proposed_model_id=proposed_model_id or (manuscript.title if manuscript else "Proposed Model"),
            content=content,
        )
        return self.llm.generate_structured(
            prompt=prompt,
            response_model=ReinterrogationScoreResponse,
            system_prompt=BLIND_DISCREPANCY_SYSTEM_PROMPT,
        )

    def reinterrogate_omission(
        self,
        manuscript: ParsedManuscript,
        dataset_name: str,
        proposed_model_id: str,
    ) -> ReinterrogationScoreResponse:
        """Re-interrogates the agent regarding a benchmark candidate it did not extract.

        Double-blind: informs the agent that an evaluation was flagged by peer review,
        without revealing the score extracted by the other agent.
        """
        content = self._build_context_content(
            manuscript,
            focal_keywords=[dataset_name, proposed_model_id],
        )
        prompt = BLIND_OMISSION_USER_TEMPLATE.format(
            dataset_name=dataset_name,
            proposed_model_id=proposed_model_id or (manuscript.title if manuscript else "Proposed Model"),
            content=content,
        )
        return self.llm.generate_structured(
            prompt=prompt,
            response_model=ReinterrogationScoreResponse,
            system_prompt=BLIND_OMISSION_SYSTEM_PROMPT,
        )

    def reinterrogate_granularity_macro(
        self,
        manuscript: ParsedManuscript,
        dataset_name: str,
        extracted_organs: List[str],
        proposed_model_id: str,
    ) -> ReinterrogationScoreResponse:
        """Re-interrogates the agent to extract the aggregated Overall/Mean score when only sub-structures were extracted.

        Double-blind: instructs the agent that the database requires the representative Macro-Average / Mean
        score, without revealing competitor scores.
        """
        content = self._build_context_content(
            manuscript,
            focal_keywords=[dataset_name, proposed_model_id, "Average", "Mean", "Overall"],
        )
        organs_summary = ", ".join(extracted_organs[:6]) + (f" and {len(extracted_organs) - 6} more" if len(extracted_organs) > 6 else "")
        prompt = BLIND_GRANULARITY_USER_TEMPLATE.format(
            dataset_name=dataset_name,
            organs_summary=organs_summary or "individual anatomical sub-structures",
            proposed_model_id=proposed_model_id or (manuscript.title if manuscript else "Proposed Model"),
            content=content,
        )
        return self.llm.generate_structured(
            prompt=prompt,
            response_model=ReinterrogationScoreResponse,
            system_prompt=BLIND_GRANULARITY_SYSTEM_PROMPT,
        )

