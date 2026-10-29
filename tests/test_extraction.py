"""Unit and integration tests for PDF parsing, extraction schemas, and evaluation."""

from pathlib import Path
import pytest

from agent_gmmis.config import PROJECT_ROOT
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.data.models import InclusionCriteriaCheck
from agent_gmmis.extraction.evaluator import ExtractionEvaluator
from agent_gmmis.extraction.extractor import ExtractionAgent
from agent_gmmis.extraction.pdf_parser import PDFParser, ParsedManuscript
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

    def test_dynamic_context_budget_and_table_prioritization(self):
        """Verifies that high-density result table pages are prioritized over low-density narrative pages."""
        from agent_gmmis.extraction.pdf_parser import ParsedManuscript

        # Client configured for Gemini -> 80,000 budget
        gemini_client = LLMClient(config=LLMConfig(provider="gemini", api_key="dummy"))
        agent_gemini = ExtractionAgent(llm_client=gemini_client)
        assert agent_gemini.get_default_context_chars() == 80000

        # Client configured for Mufasa -> 28,000 budget
        mufasa_client = LLMClient(config=LLMConfig(provider="mufasa", api_key="dummy"))
        agent_mufasa = ExtractionAgent(llm_client=mufasa_client)
        assert agent_mufasa.get_default_context_chars() == 28000

        # Construct a synthetic manuscript where page 10 has the primary results table
        manuscript = ParsedManuscript(
            file_path="synthetic.pdf",
            title="Synthetic Model Paper",
            total_pages=10,
            pages_text={
                1: "Introduction to Synthetic Model for 3D segmentation.",
                2: "Related Work: CNNs and Transformers have evolved over time." * 20,
                3: "Methodology overview." * 30,
                10: "Table 1. SOTA benchmark comparison. BraTS: 0.892, BTCV: 0.854, AMOS: 0.821, FLARE: 0.880, KiTS: 0.875.",
            },
            table_pages=[10],
        )

        # With a tight budget (e.g. 500 chars), Page 1 and Page 10 must be included before Page 2 or Page 3
        context = agent_mufasa._build_context_content(manuscript, max_chars=400)
        assert "[Page 1]" in context
        assert "[Page 10]" in context
        assert "Table 1. SOTA benchmark comparison" in context

    def test_mufasa_default_budget_does_not_trigger_middle_trimming(self, monkeypatch):
        """Verifies that 28000-char context budget on mufasa provider does not trigger pre-trimming in LLMClient."""
        from unittest.mock import MagicMock

        client = LLMClient(config=LLMConfig(provider="mufasa", api_key="dummy"))
        agent = ExtractionAgent(llm_client=client)
        assert agent.get_default_context_chars() == 28000

        # Construct a manuscript with ~24,000 chars of text across pages
        table_content = "Table 1. SOTA Benchmark: BraTS 0.892, BTCV 0.854, AMOS 0.821. " * 380
        manuscript = ParsedManuscript(
            file_path="synthetic_large.pdf",
            title="Synthetic Model Paper",
            total_pages=5,
            pages_text={
                1: "Introduction to Synthetic Model for 3D segmentation.",
                2: table_content,
            },
            table_pages=[2],
        )

        captured_messages = []
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"paper_title": "Synthetic", "proposed_model_id": "Synthetic", "extracted_scores": []}'))
        ]

        def fake_create(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return mock_response

        monkeypatch.setattr(client._openai_client.chat.completions, "create", fake_create)

        agent.process_manuscript(manuscript)

        assert len(captured_messages) > 0
        user_message = captured_messages[-1]["content"]
        # Must never trim the middle text where comparative tables are located
        assert "[... middle text trimmed to respect context limit ...]" not in user_message
        assert "Table 1. SOTA Benchmark" in user_message


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

    def test_evaluator_multi_organ_averaging(self):
        """Verifies evaluator computes mathematical mean across individual organs when no overall row exists."""
        db = DatabaseManager()
        evaluator = ExtractionEvaluator(db_manager=db)

        # TP-Mamba GT for BTCV is 0.848
        extracted = ExtractedPaperData(
            paper_title="Tri-Plane Mamba",
            proposed_model_id="TP-Mamba",
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="BTCV",
                    organ_or_task="Liver",
                    dice_score=0.972,
                    is_baseline_comparison=False,
                ),
                RawExtractedScore(
                    dataset_name="BTCV",
                    organ_or_task="Spleen",
                    dice_score=0.724,
                    is_baseline_comparison=False,
                ),
            ],
        )

        report = evaluator.evaluate(extracted, ground_truth_id="TP-Mamba", compare_source="primary")
        assert report.matched_datasets_count == 1
        assert report.dice_mae is not None
        # Mean of 0.972 and 0.724 is 0.848, which matches GT 0.848 exactly
        assert abs(report.dice_mae) < 1e-3


class TestPrioritaP3Robustness:
    """Verifies Priorità P3 robustness fixes: permissive validators, graceful degradation, and collision fixes."""

    def test_pydantic_permissive_is_baseline_comparison(self):
        """Verifies is_baseline_comparison validator converts None, strings and numbers into valid booleans."""
        # False cases
        for val in [None, "Not applicable", "none", "n/a", "false", "no", 0, ""]:
            s = RawExtractedScore(
                dataset_name="BraTS",
                organ_or_task="Overall",
                dice_score=0.88,
                is_baseline_comparison=val,
            )
            assert s.is_baseline_comparison is False

        # True cases
        for val in [True, "true", "yes", 1, "baseline", "competitor"]:
            s = RawExtractedScore(
                dataset_name="BraTS",
                organ_or_task="Overall",
                dice_score=0.88,
                is_baseline_comparison=val,
            )
            assert s.is_baseline_comparison is True

    def test_pydantic_filter_malformed_scores(self):
        """Verifies that incomplete score dicts missing dataset_name are dropped gracefully."""
        data = {
            "paper_title": "Robustness Test",
            "proposed_model_id": "TestModel",
            "extracted_scores": [
                {"dataset_name": "BraTS", "organ_or_task": "Overall", "dice_score": 0.89},
                {"dataset_name": "", "organ_or_task": "Empty Dataset", "dice_score": 0.50},
                {"dataset_name": None, "organ_or_task": "Null Dataset", "dice_score": 0.40},
                {"organ_or_task": "Missing Key", "dice_score": 0.30},
                {"dataset_name": "BTCV", "organ_or_task": "Overall", "dice_score": 0.82},
            ],
        }
        extracted = ExtractedPaperData.model_validate(data)
        assert len(extracted.extracted_scores) == 2
        assert [s.dataset_name for s in extracted.extracted_scores] == ["BraTS", "BTCV"]

    def test_dual_extraction_graceful_degradation_when_one_agent_fails(self):
        """Verifies that if one extraction agent fails, the coordinator gracefully degrades to single-agent mode."""
        from unittest.mock import MagicMock
        from agent_gmmis.extraction.dual_coordinator import DualExtractionCoordinator
        from agent_gmmis.extraction.pdf_parser import ParsedManuscript

        mock_agent_a = MagicMock()
        mock_agent_a.process_manuscript.return_value = ExtractedPaperData(
            paper_title="Survivor Model Paper",
            proposed_model_id="SurvivorModel",
            extracted_scores=[
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Overall", dice_score=0.91),
            ],
        )

        mock_agent_b = MagicMock()
        mock_agent_b.process_manuscript.side_effect = RuntimeError("vLLM connection refused / OOM")

        coord = DualExtractionCoordinator(agent_a=mock_agent_a, agent_b=mock_agent_b)
        manuscript = ParsedManuscript(
            file_path="synthetic.pdf",
            title="Survivor Model Paper",
            total_pages=2,
            pages_text={1: "Paper text"},
        )

        # Must NOT raise RuntimeError!
        ext_a, ext_b = coord.extract_manuscript_parallel(manuscript)

        assert ext_a.proposed_model_id == "SurvivorModel"
        assert len(ext_a.extracted_scores) == 1
        assert ext_b.proposed_model_id == "SurvivorModel"
        assert len(ext_b.extracted_scores) == 0  # Graceful dummy fallback
        assert "Single-agent fallback" in ext_b.extraction_notes

    def test_find_matching_score_index_budget_and_prompt_discrimination(self):
        """Verifies that _find_matching_score_index discriminates between data budgets and prompt types."""
        from agent_gmmis.extraction.dual_coordinator import _find_matching_score_index

        scores = [
            RawExtractedScore(
                dataset_name="BTCV",
                organ_or_task="Overall",
                dice_score=0.72,
                data_budget_or_setting="10% labeled",
                prompt_type="1 point",
            ),
            RawExtractedScore(
                dataset_name="BTCV",
                organ_or_task="Overall",
                dice_score=0.88,
                data_budget_or_setting="100% labeled",
                prompt_type="1 point",
            ),
        ]

        # Target the 100% budget setting
        idx_100 = _find_matching_score_index(
            scores,
            dataset_name="BTCV",
            organ_or_task="Overall",
            data_budget_or_setting="100% labeled",
            prompt_type="1 point",
        )
        assert idx_100 == 1

        # Target the 10% budget setting
        idx_10 = _find_matching_score_index(
            scores,
            dataset_name="BTCV",
            organ_or_task="Overall",
            data_budget_or_setting="10% labeled",
            prompt_type="1 point",
        )
        assert idx_10 == 0


class TestLLMClientRobustness:
    """Verifies robustness of LLMClient structured generation, schema inlining, and JSON healing."""

    def test_schema_inlining_and_no_defs(self, monkeypatch):
        """Verifies that generate_structured inlines schema refs and strips $defs from prompt."""
        from unittest.mock import MagicMock
        from agent_gmmis.llm.client import LLMClient, LLMConfig

        client = LLMClient(config=LLMConfig(provider="mufasa", api_key="dummy"))
        captured = []

        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(message=MagicMock(content='{"paper_title": "Test Paper", "proposed_model_id": "TestNet", "extracted_scores": []}'))
        ]

        def fake_create(**kwargs):
            captured.append(kwargs)
            return mock_resp

        monkeypatch.setattr(client._openai_client.chat.completions, "create", fake_create)

        res = client.generate_structured(prompt="Analyze this paper", response_model=ExtractedPaperData)
        assert res.paper_title == "Test Paper"
        assert len(captured) == 1
        
        # Verify prompt instructions do not contain $defs
        content = captured[0]["messages"][0]["content"]
        assert '"$defs"' not in content
        assert "CRITICAL INSTRUCTIONS FOR STRUCTURED JSON OUTPUT" in content

    def test_json_healing_with_preamble_and_unclosed_brackets(self, monkeypatch):
        """Verifies that conversational preamble and unclosed brackets are healed correctly."""
        from unittest.mock import MagicMock
        from agent_gmmis.llm.client import LLMClient, LLMConfig

        client = LLMClient(config=LLMConfig(provider="mufasa", api_key="dummy"))

        # Preamble + truncated JSON
        malformed_content = (
            "Here is the extraction you requested:\n"
            '{"paper_title": "Healed Paper", "proposed_model_id": "HealedNet", "extracted_scores": [{"dataset_name": "BTCV", "dice_score": 0.85'
        )

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=malformed_content))]

        monkeypatch.setattr(client._openai_client.chat.completions, "create", lambda **kw: mock_resp)

        res = client.generate_structured(prompt="Analyze", response_model=ExtractedPaperData)
        assert res.paper_title == "Healed Paper"
        assert res.proposed_model_id == "HealedNet"




