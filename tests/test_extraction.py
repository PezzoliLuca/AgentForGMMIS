"""Unit and integration tests for PDF parsing, extraction schemas, and evaluation."""

from pathlib import Path
import pytest

from agent_gmmis.config import PROJECT_ROOT
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.data.models import InclusionCriteriaCheck
from agent_gmmis.extraction.evaluator import ExtractionEvaluator
from agent_gmmis.extraction.extractor import ExtractionAgent
from agent_gmmis.extraction.pdf_parser import PDFParser
from agent_gmmis.extraction.schemas import ExtractedPaperData, RawExtractedScore
from agent_gmmis.llm.client import LLMClient, LLMConfig

SAMPLE_PDF_PATH = PROJECT_ROOT / "data" / "sample_papers" / "medsam_arxiv.pdf"


class TestPDFParser:
    """Verifies PyMuPDF extraction capabilities on real or synthetic manuscripts."""

    def test_parse_manuscript(self, tmp_path):
        if SAMPLE_PDF_PATH.exists():
            target_path = SAMPLE_PDF_PATH
        else:
            import pymupdf
            target_path = tmp_path / "synthetic_sample.pdf"
            doc = pymupdf.open()
            p1 = doc.new_page()
            p1.insert_text((50, 72), "Segment Anything in Medical Images (MedSAM)\nAbstract: Medical foundation model.\n")
            p2 = doc.new_page()
            p2.insert_text((50, 72), "Table 1: Benchmark Results on BraTS and BTCV\nBraTS Dice: 85.5%")
            doc.save(target_path)
            doc.close()

        manuscript = PDFParser.parse(target_path)
        assert manuscript.total_pages > 0
        assert len(manuscript.pages_text) == manuscript.total_pages
        assert len(manuscript.table_pages) > 0, "Should identify pages containing evaluation tables"

        # Check full text generation
        full_text = manuscript.get_full_text(max_pages=3)
        assert len(full_text) > 30
        assert "Segment Anything" in full_text or "MedSAM" in full_text

        # Check table dense text
        table_text = manuscript.get_table_dense_text()
        assert len(table_text) > 10


class TestExtractionSchemas:
    """Verifies schema validation and score normalization."""

    def test_raw_score_normalization(self):
        score = RawExtractedScore(
            dataset_name="BraTS 2021",
            organ_or_task="Brain Tumor",
            dice_score=85.5,  # percentage
            table_or_section_ref="Table 1",
        )
        assert score.dice_score == 0.855

    def test_extracted_paper_data_validation(self):
        data = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            code_repository_url="https://github.com/bowang-lab/MedSAM",
            millions_of_parameters=93.7,
            resources_total_vram_gb=1600,
            screening=InclusionCriteriaCheck(
                is_3d=True,
                is_radiological=True,
                is_public_dataset=True,
                has_dice_metric=True,
            ),
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="BraTS",
                    organ_or_task="Brain Tumor",
                    dice_score=0.7563,
                    is_baseline_comparison=False,
                ),
                RawExtractedScore(
                    dataset_name="BTCV",
                    organ_or_task="Abdomen Multi-Organ",
                    dice_score=0.8030,
                    is_baseline_comparison=False,
                ),
            ],
        )
        assert data.proposed_model_id == "MedSAM"

    def test_robust_string_and_float_parsing(self):
        """Verifies that LLM string phrases like 'Not explicitly mentioned' and float VRAM parse cleanly."""
        data = ExtractedPaperData(
            paper_title="Test Model",
            proposed_model_id="TestModel",
            millions_of_parameters="Not explicitly mentioned",
            number_of_gflops="N/A",
            resources_total_vram_gb=91.767,  # float VRAM reported in FLAP-SAM/MoME
            screening=InclusionCriteriaCheck(
                is_3d=True,
                is_radiological=True,
                is_public_dataset=True,
                has_dice_metric=True,
            ),
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="BTCV",
                    organ_or_task="Overall",
                    dice_score="88.5%",
                ),
                RawExtractedScore(
                    dataset_name="BraTS",
                    organ_or_task="Overall",
                    dice_score="not reported",
                ),
            ],
        )
        assert data.millions_of_parameters is None
        assert data.number_of_gflops is None
        assert data.resources_total_vram_gb == 91.767
        assert data.extracted_scores[0].dice_score == 0.885
        assert data.extracted_scores[1].dice_score is None
        assert len(data.extracted_scores) == 2
        assert data.screening.is_included is True

    def test_chain_of_thought_and_derivation_rationale(self):
        """Verifies ExtractionThoughtProcess and derivation_rationale serialization and deserialization."""
        from agent_gmmis.extraction.schemas import ExtractionThoughtProcess

        tp = ExtractionThoughtProcess(
            proposed_model_identification="The paper proposes KnowSAM, a semi-supervised SAM distillation model.",
            primary_benchmark_tables="Table 3 reports SOTA comparison on BCSS; Table 6 reports Kvasir.",
            peak_performance_logic="For BCSS, 30% labeled data gives peak DSC 0.7269 over 1% and 10%.",
            multitask_or_macro_averages="Using the macro-average Overall row from Table 3.",
            ambiguities_or_caveats="No standard deviation reported in Table 6.",
        )
        score = RawExtractedScore(
            dataset_name="BCSS",
            organ_or_task="Overall",
            derivation_rationale="Extracted from Table 3, row KnowSAM (Ours), column 30% labeled data.",
            dice_score=0.7269,
            table_or_section_ref="Table 3",
        )
        data = ExtractedPaperData(
            thought_process=tp,
            paper_title="Learnable Prompting SAM-induced Knowledge Distillation",
            proposed_model_id="KnowSAM",
            screening=InclusionCriteriaCheck(
                is_3d=True,
                is_radiological=True,
                is_public_dataset=True,
                has_dice_metric=True,
            ),
            extracted_scores=[score],
        )

        assert data.thought_process is not None
        assert data.thought_process.proposed_model_identification.startswith("The paper proposes KnowSAM")
        assert data.extracted_scores[0].derivation_rationale.startswith("Extracted from Table 3")

        # Verify JSON round-trip serialization
        raw_json = data.model_dump_json()
        restored = ExtractedPaperData.model_validate_json(raw_json)
        assert restored.thought_process.peak_performance_logic == tp.peak_performance_logic
        assert restored.extracted_scores[0].derivation_rationale == score.derivation_rationale


class TestExtractionEvaluator:
    """Verifies benchmarking extracted predictions against database ground truth."""

    def test_ground_truth_comparison(self):
        db = DatabaseManager()
        evaluator = ExtractionEvaluator(db_manager=db)

        # Build mock extracted data with exact GT values for BraTS and BTCV on MedSAM
        extracted = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            millions_of_parameters=93.7,
            code_repository_url="https://github.com/bowang-lab/MedSAM",
            resources_total_vram_gb=1600,
            screening=InclusionCriteriaCheck(
                is_3d=True,
                is_radiological=True,
                is_public_dataset=True,
                has_dice_metric=True,
            ),
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="BraTS",
                    organ_or_task="Brain Tumor",
                    dice_score=0.7563,  # exact GT in database
                    is_baseline_comparison=False,
                ),
                RawExtractedScore(
                    dataset_name="BTCV",
                    organ_or_task="Abdomen",
                    dice_score=0.8030,  # exact GT in database
                    is_baseline_comparison=False,
                ),
            ],
        )

        report = evaluator.evaluate(extracted, ground_truth_id="MedSAM")

        assert report.model_id == "MedSAM"
        assert report.parameter_error_pct == 0.0
        assert report.code_url_match is True
        assert report.matched_datasets_count == 2
        assert report.dataset_precision == 1.0
        assert report.dice_mae is not None
        assert report.dice_mae < 1e-4  # effectively 0 error


class TestExtractionAgentMock:
    """Verifies ExtractionAgent execution in offline/mock mode."""

    def test_offline_extractor_execution(self):
        mock_client = LLMClient(config=LLMConfig(provider="mock"))
        agent = ExtractionAgent(llm_client=mock_client)

        data = agent.process_pdf(SAMPLE_PDF_PATH)
        assert isinstance(data, ExtractedPaperData)


class TestDatasetAliasResolver:
    """Verifies robust resolution of non-standard and challenge aliases."""

    def test_alias_resolution(self):
        from agent_gmmis.extraction.alias_resolver import DatasetAliasResolver

        resolver = DatasetAliasResolver()
        assert resolver.resolve("Organs in Abdomen CT") == "FLARE"
        assert resolver.resolve("Organs in Abdomen MR") == "AMOS"
        assert resolver.resolve("Abdomen CT") == "FLARE"
        assert resolver.resolve("Synapse") == "BTCV"
        assert resolver.resolve("btcv") == "BTCV"
        assert resolver.resolve("BraTS challenge") == "BraTS"
        assert resolver.resolve("ATLAS v2.0 dataset") == "ATLAS v2.0"
        assert resolver.resolve("Prost.MRI") == "PROMISE12"
        assert resolver.resolve("Fed-KiTS") == "KiTS"
        assert resolver.resolve("AMOS external test set (MRI only)") == "AMOS"
        assert resolver.resolve("CHAOS external test set (MRI only)") == "CHAOS"
        assert resolver.resolve("NonExistentDatasetXYZ") is None

    def test_evaluator_with_aliases(self):
        """Verifies evaluator matches BioSAM-2 aliases (Organs in Abdomen CT/MR)."""
        db = DatabaseManager()
        evaluator = ExtractionEvaluator(db_manager=db)

        extracted = ExtractedPaperData(
            paper_title="Biomedical SAM 2",
            proposed_model_id="Biomedical SAM-2 (BioSAM-2)",
            millions_of_parameters=38.9,
            screening=InclusionCriteriaCheck(
                is_3d=True,
                is_radiological=True,
                is_public_dataset=True,
                has_dice_metric=True,
            ),
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="Organs in Abdomen CT",
                    organ_or_task="Abdomen",
                    dice_score=0.7632,
                    is_baseline_comparison=False,
                ),
                RawExtractedScore(
                    dataset_name="Organs in Abdomen MR",
                    organ_or_task="Abdomen",
                    dice_score=0.7439,
                    is_baseline_comparison=False,
                ),
            ],
        )

        report = evaluator.evaluate(extracted, ground_truth_id="Biomedical SAM-2 (BioSAM-2)", compare_source="primary")
        assert report.matched_datasets_count == 2
        assert report.ground_truth_datasets_count == 2
        assert report.dataset_recall == 1.0
        assert report.dice_mae is not None
        assert report.dice_mae < 1e-4

