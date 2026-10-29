"""ExtractionAgent: Ingests scientific manuscripts and extracts structured benchmark records."""

from pathlib import Path
import re
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

    def get_default_context_chars(self) -> int:
        """Determines optimal character budget based on model context window."""
        if not self.llm or not hasattr(self.llm, "config"):
            return 45000
        provider = (getattr(self.llm.config, "provider", "") or "").lower()
        if provider == "gemini":
            return 80000
        elif provider in ("mufasa", "ollama", "vllm"):
            return 16000
        return 45000

    def _score_page(self, page_num: int, text: str, focal_keywords: Optional[List[str]] = None) -> float:
        """Heuristic score to rank pages by density of quantitative benchmark results."""
        score = 0.0
        text_lower = text.lower()

        # Decimal numbers (e.g. 0.852, 85.4)
        decimals = len(re.findall(r"\b\d{1,2}\.\d{1,4}\b", text))
        score += min(decimals, 60) * 2.0

        # Keywords related to segmentation metrics & comparisons
        for kw in ("dice", "dsc", "hd95", "miou", "iou", "sota", "benchmark", "comparison", "quantitative"):
            if kw in text_lower:
                score += 5.0

        # Table markers (including Markdown tables if present)
        score += text_lower.count("table ") * 4.0
        score += min(text_lower.count("|"), 100) * 0.2

        # Focal keywords (e.g. specific dataset or proposed model)
        if focal_keywords:
            for fkw in focal_keywords:
                if fkw and len(fkw.strip()) > 2 and fkw.lower().strip() in text_lower:
                    score += 30.0

        return score

    def _build_context_content(
        self,
        manuscript: ParsedManuscript,
        focal_keywords: Optional[List[str]] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """Composes a relevant text snippet prioritizing focal keywords, introductory pages,

        and evaluation table pages up to max_chars budget.
        """
        if not manuscript or not getattr(manuscript, "pages_text", None):
            return ""

        if max_chars is None:
            max_chars = self.get_default_context_chars()

        pages_to_include: List[int] = []
        current_len = 0

        # Always include Page 1 (Title, Abstract, Introduction) to anchor proposed model identification
        if 1 in manuscript.pages_text:
            p1_text = manuscript.pages_text[1]
            pages_to_include.append(1)
            current_len += len(p1_text)

        # Include Page 2 if space allows and no focal keywords are set
        if not focal_keywords and 2 in manuscript.pages_text and 2 not in pages_to_include:
            p2_text = manuscript.pages_text[2]
            if current_len + len(p2_text) < max_chars // 3:
                pages_to_include.append(2)
                current_len += len(p2_text)

        # Rank all other candidate pages by quantitative result density
        candidate_pages = [
            p for p in sorted(manuscript.pages_text.keys())
            if p not in pages_to_include
        ]

        scored_candidates = sorted(
            candidate_pages,
            key=lambda p: self._score_page(p, manuscript.pages_text[p], focal_keywords),
            reverse=True,
        )

        for p in scored_candidates:
            p_text = manuscript.pages_text[p]
            p_len = len(p_text)
            if current_len + p_len <= max_chars:
                pages_to_include.append(p)
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

        if not extracted_data.paper_title and manuscript.title:
            extracted_data.paper_title = manuscript.title
        if not extracted_data.proposed_model_id:
            extracted_data.proposed_model_id = manuscript.title or "ProposedModel"

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

