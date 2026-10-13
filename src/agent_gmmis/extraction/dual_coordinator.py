"""DualExtractionCoordinator: Executes concurrent manuscript extractions with two distinct LLM agents."""

import concurrent.futures
from pathlib import Path
from typing import Optional, Tuple, Union

from agent_gmmis.extraction.extractor import ExtractionAgent
from agent_gmmis.extraction.pdf_parser import ParsedManuscript, PDFParser
from agent_gmmis.extraction.schemas import ExtractedPaperData


class DualExtractionCoordinator:
    """
    Coordinates concurrent extraction of scientific manuscripts using two independent ExtractionAgent instances.
    Enforces epistemic independence by ensuring neither agent observes the other's extracted data.
    """

    def __init__(
        self,
        agent_a: Optional[ExtractionAgent] = None,
        agent_b: Optional[ExtractionAgent] = None,
    ):
        self.agent_a = agent_a or ExtractionAgent()
        self.agent_b = agent_b or ExtractionAgent()
        self.parser = PDFParser()

    def extract_manuscript_parallel(
        self,
        manuscript: ParsedManuscript,
    ) -> Tuple[ExtractedPaperData, ExtractedPaperData]:
        """
        Executes parallel extraction of a pre-parsed manuscript using ThreadPoolExecutor.

        Args:
            manuscript: ParsedManuscript object extracted from the PDF.

        Returns:
            Tuple of (extracted_data_a, extracted_data_b).
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(self.agent_a.process_manuscript, manuscript)
            future_b = executor.submit(self.agent_b.process_manuscript, manuscript)

            # Retrieve results; re-raise exceptions with clear origin if either fails
            try:
                extracted_a = future_a.result()
            except Exception as exc:
                raise RuntimeError(f"Agent A extraction failed: {exc}") from exc

            try:
                extracted_b = future_b.result()
            except Exception as exc:
                raise RuntimeError(f"Agent B extraction failed: {exc}") from exc

        return extracted_a, extracted_b

    def extract_pdf_parallel(
        self,
        pdf_path: Union[str, Path],
    ) -> Tuple[ExtractedPaperData, ExtractedPaperData]:
        """
        Parses a PDF manuscript once and submits it to both extraction agents concurrently.

        Args:
            pdf_path: Filesystem path to the PDF manuscript.

        Returns:
            Tuple of (extracted_data_a, extracted_data_b).
        """
        manuscript = self.parser.parse(pdf_path)
        return self.extract_manuscript_parallel(manuscript)
