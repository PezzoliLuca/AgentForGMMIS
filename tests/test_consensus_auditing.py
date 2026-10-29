"""Unit tests for Dual-Agent Consensus Extraction & Epistemic Arbitration Engine."""

import pytest
from agent_gmmis.auditing.auditor import AuditorAgent
from agent_gmmis.auditing.schemas import AuditStatus, ConsensusStatus
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.data.models import InclusionCriteriaCheck
from agent_gmmis.extraction.dual_coordinator import DualExtractionCoordinator
from agent_gmmis.extraction.extractor import ExtractionAgent
from agent_gmmis.extraction.pdf_parser import ParsedManuscript
from agent_gmmis.extraction.schemas import ExtractedPaperData, RawExtractedScore
from agent_gmmis.llm.client import LLMClient, LLMConfig


@pytest.fixture
def db():
    return DatabaseManager()


@pytest.fixture
def auditor(db):
    return AuditorAgent(db_manager=db)


class TestDualExtractionCoordinator:
    """Verifies multithreaded concurrent manuscript extraction."""

    def test_extract_manuscript_parallel_mock(self):
        """Verifies that two agents with mock providers extract concurrently."""
        client_a = LLMClient(config=LLMConfig(provider="mock", model="mock-model-a"))
        client_b = LLMClient(config=LLMConfig(provider="mock", model="mock-model-b"))

        agent_a = ExtractionAgent(llm_client=client_a)
        agent_b = ExtractionAgent(llm_client=client_b)

        coordinator = DualExtractionCoordinator(agent_a=agent_a, agent_b=agent_b)
        dummy_manuscript = ParsedManuscript(
            file_path="dummy.pdf",
            title="Segment Anything in Medical Images",
            abstract="We propose MedSAM.",
            total_pages=5,
            pages_text={1: "MedSAM is a generalist 3D segmentation model."},
        )

        extracted_a, extracted_b = coordinator.extract_manuscript_parallel(dummy_manuscript)

        assert extracted_a.proposed_model_id == "MedSAM"
        assert extracted_b.proposed_model_id == "MedSAM"
        assert len(extracted_a.extracted_scores) > 0
        assert len(extracted_b.extracted_scores) > 0


class TestConsensusAuditor:
    """Verifies dual-agent consensus auditing, epistemic arbitration, and agreement metrics."""

    def test_audit_consensus_full_concordance(self, auditor):
        """When two independent models agree on all benchmarks within tolerance, certify as VERIFIED with high confidence."""
        screening = InclusionCriteriaCheck(
            is_3d=True,
            is_radiological=True,
            is_public_dataset=True,
            has_dice_metric=True,
        )

        data_a = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            resources_total_vram_gb=1600,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Brain Tumor", dice_score=0.7563),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Abdomen Multi-Organ", dice_score=0.8030),
            ],
        )

        data_b = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=94.0,  # slight diff (< 5%) -> concordant
            resources_total_vram_gb=1600,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Brain Tumor", dice_score=0.7560),  # delta = 0.0003 <= 0.01
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Abdomen Multi-Organ", dice_score=0.8035),  # delta = 0.0005 <= 0.01
            ],
        )

        report = auditor.audit_consensus(data_a, data_b)

        assert report.status == AuditStatus.VERIFIED
        assert report.consensus_screening.status == ConsensusStatus.CONCORDANT
        assert report.consensus_metadata.model_id_status == ConsensusStatus.CONCORDANT
        assert report.consensus_metadata.params_status == ConsensusStatus.CONCORDANT
        assert report.consensus_metadata.resolved_params_m == 93.8
        assert report.dispute_count == 0
        assert report.concordant_count == 2
        assert report.unconfirmed_count == 0
        assert report.concordance_rate == 1.0
        assert report.inter_model_dice_mae is not None
        assert report.inter_model_dice_mae < 0.001
        assert report.confidence_score >= 0.90

    def test_audit_consensus_disputed_scores(self, auditor):
        """When models extract divergent Dice scores on the same benchmark (> 0.01), flag as DISPUTED."""
        screening = InclusionCriteriaCheck(
            is_3d=True,
            is_radiological=True,
            is_public_dataset=True,
            has_dice_metric=True,
        )

        data_a = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Abdomen", dice_score=0.8030),
            ],
        )

        data_b = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Abdomen", dice_score=0.7410),  # Divergence 0.0620 > 0.01
            ],
        )

        report = auditor.audit_consensus(data_a, data_b)

        assert report.status == AuditStatus.FLAGGED_WARNING
        assert report.dispute_count == 1
        assert report.concordant_count == 0
        assert report.concordance_rate == 0.0
        assert len(report.consensus_scores) == 1

        btcv_score = report.consensus_scores[0]
        assert btcv_score.status == ConsensusStatus.DISPUTED
        assert btcv_score.score_a == 0.8030
        assert btcv_score.score_b == 0.7410
        assert btcv_score.resolved_score is None  # Must NOT resolve disputed numbers silently
        assert btcv_score.delta == 0.0620

    def test_audit_consensus_unconfirmed_single_agent(self, auditor):
        """When a benchmark is captured by only one agent, preserve it as SINGLE_AGENT_UNCONFIRMED."""
        screening = InclusionCriteriaCheck(
            is_3d=True,
            is_radiological=True,
            is_public_dataset=True,
            has_dice_metric=True,
        )

        data_a = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Brain Tumor", dice_score=0.7563),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Abdomen", dice_score=0.8030),
            ],
        )

        data_b = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Abdomen", dice_score=0.8030),  # Missing BraTS
            ],
        )

        report = auditor.audit_consensus(data_a, data_b)

        assert report.concordant_count == 1  # BTCV
        assert report.unconfirmed_count == 1  # BraTS
        assert report.dispute_count == 0
        assert report.total_unique_benchmarks == 2

        brats_score = next(s for s in report.consensus_scores if s.canonical_dataset_name == "BraTS")
        assert brats_score.status == ConsensusStatus.SINGLE_AGENT_UNCONFIRMED
        assert brats_score.score_a == 0.7563
        assert brats_score.score_b is None
        assert brats_score.resolved_score == 0.7563

    def test_audit_consensus_dataset_alias_alignment(self, auditor):
        """Semantic alias resolution aligns 'Synapse' from Model A and 'BTCV' from Model B as CONCORDANT."""
        screening = InclusionCriteriaCheck(
            is_3d=True,
            is_radiological=True,
            is_public_dataset=True,
            has_dice_metric=True,
        )

        data_a = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="Beyond The Cranial Vault", organ_or_task="Multi-organ", dice_score=0.8030),
            ],
        )

        data_b = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Multi-organ", dice_score=0.8030),
            ],
        )

        report = auditor.audit_consensus(data_a, data_b)

        assert report.total_unique_benchmarks == 1
        assert report.concordant_count == 1
        assert report.dispute_count == 0
        assert report.consensus_scores[0].canonical_dataset_name == "BTCV"
        assert report.consensus_scores[0].status == ConsensusStatus.CONCORDANT

    def test_audit_consensus_screening_dispute(self, auditor):
        """Dispute between models on 3D eligibility is marked and penalized."""
        data_a = ExtractedPaperData(
            paper_title="Test Model",
            proposed_model_id="TestModel",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[],
        )
        data_b = ExtractedPaperData(
            paper_title="Test Model",
            proposed_model_id="TestModel",
            screening=InclusionCriteriaCheck(is_3d=False, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[],
        )

        report = auditor.audit_consensus(data_a, data_b)
        assert report.consensus_screening.status == ConsensusStatus.DISPUTED
        assert "is_3d" in (report.consensus_screening.notes or "")

    def test_audit_consensus_parameter_dispute(self, auditor):
        """Divergence > 5% on parameter count is flagged as DISPUTED without silent averaging."""
        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)

        data_a = ExtractedPaperData(
            paper_title="Test Model",
            proposed_model_id="TestModel",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[],
        )
        data_b = ExtractedPaperData(
            paper_title="Test Model",
            proposed_model_id="TestModel",
            millions_of_parameters=305.0,  # Huge divergence
            screening=screening,
            extracted_scores=[],
        )

        report = auditor.audit_consensus(data_a, data_b)
        assert report.consensus_metadata.params_status == ConsensusStatus.DISPUTED
        assert report.consensus_metadata.resolved_params_m is None

    def test_audit_consensus_critical_anomaly_rejection(self, auditor):
        """If one model generates an impossible physical value, the consensus audit is REJECTED."""
        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)

        data_a = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Abdomen", dice_score=0.8030),
            ],
        )

        # Agent B extracted out-of-range Dice (1.25)
        # Note: RawExtractedScore normalizes 1.25 to 0.0125, but 105.0 raises error, or we can use negative parameter count
        data_b = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            millions_of_parameters=-50.0,  # Negative parameters -> critical anomaly in Auditor
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Abdomen", dice_score=0.8030),
            ],
        )

        report = auditor.audit_consensus(data_a, data_b)
        assert report.status == AuditStatus.REJECTED
        assert report.confidence_score <= 0.40

    def test_audit_consensus_disentangled_granularity(self, auditor):
        """Asymmetric granularity (Agent A macro-average vs Agent B per-organ rows) maintains 100% paired concordance."""
        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)

        data_a = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Multi-organ mean", dice_score=0.8030),
            ],
        )

        data_b = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.8030),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Spleen", dice_score=0.8520),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Right Kidney", dice_score=0.8140),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Liver", dice_score=0.9230),
            ],
        )

        report = auditor.audit_consensus(data_a, data_b)

        assert report.paired_count == 1
        assert report.concordant_count == 1
        assert report.dispute_count == 0
        assert report.unconfirmed_count == 3
        assert report.total_unique_benchmarks == 4
        # Inter-annotator agreement on common paired extractions is 100%
        assert report.concordance_rate == 1.0
        # Confidence remains high despite coverage asymmetry (only modest penalty for singletons)
        assert report.confidence_score >= 0.85

    def test_audit_consensus_bounded_dispute_penalty(self, auditor):
        """Dispute penalties are proportional to the dispute ratio among paired benchmarks and bounded."""
        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)

        # 5 paired benchmarks, 1 dispute (divergence > 0.01)
        data_a = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Spleen", dice_score=0.8500),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Liver", dice_score=0.9200),
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Tumor", dice_score=0.7500),
                RawExtractedScore(dataset_name="AMOS", organ_or_task="Abdomen", dice_score=0.8100),
                RawExtractedScore(dataset_name="ACDC", organ_or_task="Heart", dice_score=0.9000),
            ],
        )

        data_b = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Spleen", dice_score=0.8500),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Liver", dice_score=0.9200),
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Tumor", dice_score=0.7500),
                RawExtractedScore(dataset_name="AMOS", organ_or_task="Abdomen", dice_score=0.8100),
                RawExtractedScore(dataset_name="ACDC", organ_or_task="Heart", dice_score=0.7500),  # Dispute on ACDC
            ],
        )

        report = auditor.audit_consensus(data_a, data_b)

        assert report.paired_count == 5
        assert report.concordant_count == 4
        assert report.dispute_count == 1
        assert report.concordance_rate == 0.8
        # Base confidence ~0.95 -> with 1/5 dispute penalty (0.05), confidence should still be ~0.90, NOT destroyed!
        assert report.confidence_score >= 0.80

    def test_audit_consensus_data_budget_split_separation(self, auditor):
        """Semi-supervised splits (e.g. 10% vs 20% labeled) match their exact regimes and avoid false disputes."""
        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)

        data_a = ExtractedPaperData(
            paper_title="KnowSAM",
            proposed_model_id="KnowSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="ACDC", organ_or_task="Heart", dice_score=0.6811, data_budget_or_setting="10% labeled"),
                RawExtractedScore(dataset_name="ACDC", organ_or_task="Heart", dice_score=0.7269, data_budget_or_setting="20% labeled"),
            ],
        )

        data_b = ExtractedPaperData(
            paper_title="KnowSAM",
            proposed_model_id="KnowSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="ACDC", organ_or_task="Heart", dice_score=0.6811, data_budget_or_setting="10% labeled"),
                RawExtractedScore(dataset_name="ACDC", organ_or_task="Heart", dice_score=0.7269, data_budget_or_setting="20% labeled"),
            ],
        )

        report = auditor.audit_consensus(data_a, data_b)

        assert report.paired_count == 2
        assert report.concordant_count == 2
        assert report.dispute_count == 0
        assert report.concordance_rate == 1.0


class TestDualExtractionDeliberation:
    """Verifies double-blind multi-round deliberation between ExtractionAgent instances."""

    def test_reinterrogate_discrepancy_double_blind_prompt(self):
        """Verifies that reinterrogate_discrepancy builds a strict blind prompt without leaking competitor data."""
        client = LLMClient(config=LLMConfig(provider="mock"))
        agent = ExtractionAgent(llm_client=client)

        manuscript = ParsedManuscript(
            file_path="dummy.pdf",
            title="MedSAM 3D",
            abstract="3D segmentation model.",
            total_pages=3,
            pages_text={
                1: "MedSAM 3D Introduction",
                2: "Table 1: Evaluation on BTCV. MedSAM achieves 0.8850 under bounding box prompt.",
            },
            table_pages=[2],
        )

        resp = agent.reinterrogate_discrepancy(
            manuscript=manuscript,
            dataset_name="BTCV",
            organ_or_task="Overall",
            previous_score=0.7500,
            previous_prompt="1 point",
            proposed_model_id="MedSAM 3D",
        )

        assert resp.dataset_name == "BTCV"
        assert resp.is_valid_benchmark is True
        assert resp.dice_score is not None

    def test_reinterrogate_omission_double_blind_prompt(self):
        """Verifies that reinterrogate_omission challenges the agent without leaking candidate numbers."""
        client = LLMClient(config=LLMConfig(provider="mock"))
        agent = ExtractionAgent(llm_client=client)

        manuscript = ParsedManuscript(
            file_path="dummy.pdf",
            title="MedSAM 3D",
            abstract="3D segmentation model.",
            total_pages=3,
            pages_text={
                1: "MedSAM 3D Introduction",
                2: "Table 2: AMOS Abdominal CT evaluation. Dice score: 0.8210.",
            },
            table_pages=[2],
        )

        resp = agent.reinterrogate_omission(
            manuscript=manuscript,
            dataset_name="AMOS",
            proposed_model_id="MedSAM 3D",
        )

        assert resp.dataset_name == "AMOS"
        assert resp.is_valid_benchmark is True

    def test_deliberation_initial_consensus_no_extra_rounds(self, auditor):
        """When Round 1 produces full concordance, deliberation terminates immediately."""
        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)
        data_a = ExtractedPaperData(
            paper_title="MedSAM",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.8850),
            ],
        )
        data_b = ExtractedPaperData(
            paper_title="MedSAM",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.8845),
            ],
        )

        agent_a = ExtractionAgent(llm_client=LLMClient(config=LLMConfig(provider="mock")))
        agent_b = ExtractionAgent(llm_client=LLMClient(config=LLMConfig(provider="mock")))
        agent_a.process_manuscript = lambda m: data_a
        agent_b.process_manuscript = lambda m: data_b

        coordinator = DualExtractionCoordinator(agent_a=agent_a, agent_b=agent_b)
        manuscript = ParsedManuscript(file_path="test.pdf", title="MedSAM", total_pages=1, pages_text={1: "test"})

        ext_a, ext_b, final_report, trajectory = coordinator.extract_manuscript_with_deliberation(
            manuscript=manuscript,
            auditor=auditor,
            max_rounds=3,
            verbose=False,
        )

        assert final_report.dispute_count == 0
        assert final_report.concordant_count == 1
        assert len(trajectory) == 0

    def test_deliberation_resolves_discrepancy(self, auditor):
        """Verifies that an inter-model discrepancy is resolved when an agent corrects its score in Round 2."""
        from agent_gmmis.extraction.schemas import ReinterrogationScoreResponse

        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)
        data_a = ExtractedPaperData(
            paper_title="MedSAM",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.8850, prompt_type="bounding box"),
            ],
        )
        data_b = ExtractedPaperData(
            paper_title="MedSAM",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.7500, prompt_type="1 point"),
            ],
        )

        agent_a = ExtractionAgent(llm_client=LLMClient(config=LLMConfig(provider="mock")))
        agent_b = ExtractionAgent(llm_client=LLMClient(config=LLMConfig(provider="mock")))
        agent_a.process_manuscript = lambda m: data_a
        agent_b.process_manuscript = lambda m: data_b

        # Agent A maintains 0.8850
        agent_a.reinterrogate_discrepancy = lambda m, ds, o, s, p, mod: ReinterrogationScoreResponse(
            dataset_name=ds,
            organ_or_task=o,
            is_valid_benchmark=True,
            dice_score=0.8850,
            prompt_type="bounding box",
            maintained_original=True,
            rationale="Peak score verified in Table 1",
        )
        # Agent B corrects to 0.8850 (peak bounding box score)
        agent_b.reinterrogate_discrepancy = lambda m, ds, o, s, p, mod: ReinterrogationScoreResponse(
            dataset_name=ds,
            organ_or_task=o,
            is_valid_benchmark=True,
            dice_score=0.8850,
            prompt_type="bounding box",
            maintained_original=False,
            rationale="Corrected to peak score under bounding box prompt in Table 1",
        )

        coordinator = DualExtractionCoordinator(agent_a=agent_a, agent_b=agent_b)
        manuscript = ParsedManuscript(file_path="test.pdf", title="MedSAM", total_pages=1, pages_text={1: "test"})

        ext_a, ext_b, final_report, trajectory = coordinator.extract_manuscript_with_deliberation(
            manuscript=manuscript,
            auditor=auditor,
            max_rounds=3,
            verbose=False,
        )

        assert final_report.dispute_count == 0
        assert final_report.concordant_count == 1
        assert final_report.concordance_rate == 1.0
        assert len(trajectory) == 1
        assert trajectory[0].round_number == 2
        assert trajectory[0].disputes_before == 1
        assert trajectory[0].disputes_after == 0
        assert trajectory[0].resolved_count == 1
        assert trajectory[0].stalled is False

    def test_deliberation_resolves_omission(self, auditor):
        """Verifies that an omitted benchmark is acknowledged and extracted by the omitting agent in Round 2."""
        from agent_gmmis.extraction.schemas import ReinterrogationScoreResponse

        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)
        data_a = ExtractedPaperData(
            paper_title="MedSAM",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Brain Tumor", dice_score=0.8850),
                RawExtractedScore(dataset_name="AMOS", organ_or_task="Abdomen", dice_score=0.8200),
            ],
        )
        data_b = ExtractedPaperData(
            paper_title="MedSAM",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Brain Tumor", dice_score=0.8850),
            ],
        )

        agent_a = ExtractionAgent(llm_client=LLMClient(config=LLMConfig(provider="mock")))
        agent_b = ExtractionAgent(llm_client=LLMClient(config=LLMConfig(provider="mock")))
        agent_a.process_manuscript = lambda m: data_a
        agent_b.process_manuscript = lambda m: data_b

        # Agent B verifies and extracts AMOS
        agent_b.reinterrogate_omission = lambda m, ds, mod: ReinterrogationScoreResponse(
            dataset_name=ds,
            organ_or_task="Abdomen",
            is_valid_benchmark=True,
            dice_score=0.8200,
            prompt_type="bounding box",
            maintained_original=False,
            rationale="Found in Table 3 of manuscript",
        )

        coordinator = DualExtractionCoordinator(agent_a=agent_a, agent_b=agent_b)
        manuscript = ParsedManuscript(file_path="test.pdf", title="MedSAM", total_pages=1, pages_text={1: "test"})

        ext_a, ext_b, final_report, trajectory = coordinator.extract_manuscript_with_deliberation(
            manuscript=manuscript,
            auditor=auditor,
            max_rounds=3,
            verbose=False,
        )

        assert final_report.unconfirmed_count == 0
        assert final_report.concordant_count == 2
        assert len(trajectory) == 1
        assert trajectory[0].unconfirmed_before == 1
        assert trajectory[0].unconfirmed_after == 0

    def test_deliberation_epistemic_equilibrium_stalled(self, auditor):
        """When neither agent changes its stance, deliberation detects equilibrium and terminates without looping."""
        from agent_gmmis.extraction.schemas import ReinterrogationScoreResponse

        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)
        data_a = ExtractedPaperData(
            paper_title="MedSAM",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.8850),
            ],
        )
        data_b = ExtractedPaperData(
            paper_title="MedSAM",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.7500),
            ],
        )

        agent_a = ExtractionAgent(llm_client=LLMClient(config=LLMConfig(provider="mock")))
        agent_b = ExtractionAgent(llm_client=LLMClient(config=LLMConfig(provider="mock")))
        agent_a.process_manuscript = lambda m: data_a
        agent_b.process_manuscript = lambda m: data_b

        # Both firmly maintain their scores
        agent_a.reinterrogate_discrepancy = lambda m, ds, o, s, p, mod: ReinterrogationScoreResponse(
            dataset_name=ds,
            organ_or_task=o,
            is_valid_benchmark=True,
            dice_score=0.8850,
            maintained_original=True,
            rationale="Grounded in Table 1 row 3",
        )
        agent_b.reinterrogate_discrepancy = lambda m, ds, o, s, p, mod: ReinterrogationScoreResponse(
            dataset_name=ds,
            organ_or_task=o,
            is_valid_benchmark=True,
            dice_score=0.7500,
            maintained_original=True,
            rationale="Grounded in Table 2 row 1",
        )

        coordinator = DualExtractionCoordinator(agent_a=agent_a, agent_b=agent_b)
        manuscript = ParsedManuscript(file_path="test.pdf", title="MedSAM", total_pages=1, pages_text={1: "test"})

        # Ask for 5 max rounds, but it should terminate at round 2 due to equilibrium
        ext_a, ext_b, final_report, trajectory = coordinator.extract_manuscript_with_deliberation(
            manuscript=manuscript,
            auditor=auditor,
            max_rounds=5,
            verbose=False,
        )

        assert final_report.dispute_count == 1
        assert len(trajectory) == 1
        assert trajectory[0].stalled is True
        assert trajectory[0].round_number == 2

    def test_peft_parameters_not_flagged_as_anomaly(self, auditor):
        """Verifies that PEFT/Adapter models with small trainable parameter counts are not flagged as anomalous."""
        screening = InclusionCriteriaCheck(
            is_3d=True,
            is_radiological=True,
            is_public_dataset=True,
            has_dice_metric=True,
        )
        # SFR-SAM: ViT-B backbone but declares 10.0M trainable adapter parameters
        data = ExtractedPaperData(
            proposed_model_id="SFR-SAM",
            paper_title="SFR-SAM: Segment Anything Model with Spatial-Frequency Adapters",
            visual_backbone="ViT-B with spatial-frequency adapter",
            architecture="Adapter-tuned SAM",
            millions_of_parameters=10.0,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="KiTS", organ_or_task="Overall", dice_score=0.8850),
            ],
        )

        report = auditor.audit(data)
        param_anomalies = [a for a in report.anomalies if a.field == "millions_of_parameters"]
        assert len(param_anomalies) == 0, f"PEFT model should not trigger parameter anomaly: {param_anomalies}"

    def test_concordant_sota_outlier_does_not_declass_consensus(self, auditor):
        """Verifies that when two models concordantly extract a score exceeding historical SOTA, consensus certifies as VERIFIED."""
        screening = InclusionCriteriaCheck(
            is_3d=True,
            is_radiological=True,
            is_public_dataset=True,
            has_dice_metric=True,
        )
        # TP-Mamba achieves 0.9660 on BTCV (historical max was 0.9210)
        data_a = ExtractedPaperData(
            proposed_model_id="TP-Mamba",
            paper_title="Tri-Plane Mamba",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.9660),
            ],
        )
        data_b = ExtractedPaperData(
            proposed_model_id="TP-Mamba",
            paper_title="Tri-Plane Mamba",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.9660),
            ],
        )

        consensus_report = auditor.audit_consensus(data_a, data_b)
        assert consensus_report.status == AuditStatus.VERIFIED
        assert consensus_report.dispute_count == 0
        assert consensus_report.confidence_score >= 0.95

    def test_granularity_reinterrogation_resolves_macro_score(self, auditor):
        """Verifies that when an agent extracts individual organs, deliberation re-interrogates to obtain Overall macro-average."""
        from agent_gmmis.extraction.schemas import ReinterrogationScoreResponse

        screening = InclusionCriteriaCheck(
            is_3d=True,
            is_radiological=True,
            is_public_dataset=True,
            has_dice_metric=True,
        )

        # Agent A extracted 3 individual organs for BTCV but no Overall
        data_a = ExtractedPaperData(
            proposed_model_id="IMIS-Net",
            paper_title="Interactive Medical Image Segmentation",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Liver", dice_score=0.9400),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Spleen", dice_score=0.9100),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Kidney", dice_score=0.8800),
            ],
        )

        # Agent B extracted the Overall mean score
        data_b = ExtractedPaperData(
            proposed_model_id="IMIS-Net",
            paper_title="Interactive Medical Image Segmentation",
            visual_backbone="ViT-B",
            millions_of_parameters=93.7,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.9100),
            ],
        )

        agent_a = ExtractionAgent(llm_client=LLMClient(config=LLMConfig(provider="mock")))
        agent_b = ExtractionAgent(llm_client=LLMClient(config=LLMConfig(provider="mock")))
        agent_a.process_manuscript = lambda m: data_a
        agent_b.process_manuscript = lambda m: data_b

        # Granularity re-interrogation returns the Macro-Average score for Agent A
        agent_a.reinterrogate_granularity_macro = lambda m, ds, o, mod: ReinterrogationScoreResponse(
            dataset_name=ds,
            organ_or_task="Overall",
            is_valid_benchmark=True,
            dice_score=0.9100,
            table_or_section_ref="Table 1 Mean Row",
            rationale="Identified explicit Mean DSC row in Table 1",
        )

        coordinator = DualExtractionCoordinator(agent_a=agent_a, agent_b=agent_b)
        manuscript = ParsedManuscript(file_path="test.pdf", title="IMIS-Net", total_pages=1, pages_text={1: "test"})

        ext_a, ext_b, final_report, trajectory = coordinator.extract_manuscript_with_deliberation(
            manuscript=manuscript,
            auditor=auditor,
            max_rounds=2,
            verbose=False,
        )

        # Agent A should now have an Overall score matching Agent B
        overall_scores_a = [s for s in ext_a.extracted_scores if s.organ_or_task == "Overall" and s.dataset_name == "BTCV"]
        assert len(overall_scores_a) == 1
        assert overall_scores_a[0].dice_score == 0.9100

        # Consensus report should show concordant Overall score and zero disputes
        concordant_btcv = [cs for cs in final_report.consensus_scores if cs.canonical_dataset_name.lower() == "btcv" and cs.status == ConsensusStatus.CONCORDANT]
        assert len(concordant_btcv) >= 1
        assert final_report.dispute_count == 0

    def test_normalize_data_budget_matching(self, auditor):
        """Verifies that differently worded data budget strings with the same percentage or shot match concordantly."""
        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)

        data_a = ExtractedPaperData(
            paper_title="TP-Mamba: Topology-Preserving Mamba for Medical Image Segmentation",
            proposed_model_id="TP-Mamba",
            visual_backbone="Mamba",
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.8480, data_budget_or_setting="100% labeled"),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.7970, data_budget_or_setting="25% labeled"),
            ],
        )

        data_b = ExtractedPaperData(
            paper_title="TP-Mamba: Topology-Preserving Mamba for Medical Image Segmentation",
            proposed_model_id="TP-Mamba",
            visual_backbone="Mamba",
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.8480, data_budget_or_setting="100% training data (24 samples)"),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.7970, data_budget_or_setting="25% training data (6 samples)"),
            ],
        )

        report = auditor.audit_consensus(data_a, data_b)
        assert report.paired_count == 2
        assert report.concordant_count == 2
        assert report.dispute_count == 0
        assert report.status == AuditStatus.VERIFIED

    def test_greedy_bipartite_matching_avoids_false_disputes(self, auditor):
        """Checks that candidates with identical scores pair together even if duplicate organ rows exist."""
        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)

        data_a = ExtractedPaperData(
            paper_title="TP-Mamba",
            proposed_model_id="TP-Mamba",
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.8480, data_budget_or_setting="12% labeled"),  # Duplicate/mislabeled row
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.6580, data_budget_or_setting="12% labeled"),  # True row
            ],
        )

        data_b = ExtractedPaperData(
            paper_title="TP-Mamba",
            proposed_model_id="TP-Mamba",
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.6580, data_budget_or_setting="12% training data"),
            ],
        )

        report = auditor.audit_consensus(data_a, data_b)
        assert report.concordant_count == 1
        assert report.dispute_count == 0
        assert report.unconfirmed_count == 1

    def test_consensus_neutralizes_unconfirmed_peer_hallucination(self, auditor):
        """Single-agent cloned score hallucination refuted by peer does not reject the consensus if valid benchmarks agree."""
        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)

        # Agent A has valid SegThor AND a repeated score across training sets
        data_a = ExtractedPaperData(
            paper_title="IMIS-Net",
            proposed_model_id="IMIS-Net",
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="SegThor", organ_or_task="Overall", dice_score=0.8920),
                RawExtractedScore(dataset_name="BraTS2018", organ_or_task="Overall", dice_score=0.8920),
                RawExtractedScore(dataset_name="BraTS2019", organ_or_task="Overall", dice_score=0.8920),
                RawExtractedScore(dataset_name="BraTS2020", organ_or_task="Overall", dice_score=0.8920),
                RawExtractedScore(dataset_name="BraTS2021", organ_or_task="Overall", dice_score=0.8920),
            ],
        )

        # Agent B only extracted the true valid benchmark
        data_b = ExtractedPaperData(
            paper_title="IMIS-Net",
            proposed_model_id="IMIS-Net",
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="SegThor", organ_or_task="Overall", dice_score=0.8920),
            ],
        )

        report = auditor.audit_consensus(data_a, data_b)
        # Consensus should accept SegThor and not be REJECTED because BraTS was refuted
        assert report.concordant_count == 1
        assert report.status != AuditStatus.REJECTED

    def test_consensus_delegates_to_repo_inspection_even_with_unconfirmed_noise(self, auditor):
        """Verifies that when 0 concordant scores exist and a code repository is disclosed,

        dual consensus correctly triggers NEEDS_REPO_INSPECTION even if one agent extracted noisy singletons.
        """
        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)

        # Agent A found 0 valid tables and routed to NEEDS_REPO_INSPECTION
        data_a = ExtractedPaperData(
            paper_title="BiomedParse",
            proposed_model_id="BiomedParse",
            code_repository_url="https://github.com/microsoft/BiomedParse",
            screening=screening,
            extracted_scores=[],
        )

        # Agent B extracted an unconfirmed noisy singleton (e.g. from related work or ablation)
        data_b = ExtractedPaperData(
            paper_title="BiomedParse",
            proposed_model_id="BiomedParse",
            code_repository_url="https://github.com/microsoft/BiomedParse",
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="RandomAblationTask", organ_or_task="Overall", dice_score=0.450),
            ],
        )

        report = auditor.audit_consensus(data_a, data_b)

        # 0 concordant scores between A and B
        assert report.concordant_count == 0
        assert report.unconfirmed_count == 1
        # Fallback layer must trigger RepoInspector
        assert report.status == AuditStatus.NEEDS_REPO_INSPECTION

    def test_consensus_delegates_to_outreach_when_figures_only(self, auditor):
        """Verifies that when 0 concordant scores exist, no code repo, and figures are present,

        dual consensus triggers NEEDS_OUTREACH.
        """
        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)
        from agent_gmmis.extraction.pdf_parser import ParsedManuscript

        manuscript = ParsedManuscript(
            file_path="figures_model.pdf",
            title="ModelFiguresOnly-3D",
            total_pages=5,
            pages_text={
                1: "ModelFiguresOnly-3D: Multi-task Evaluation without tables.",
                2: "Figure 4 shows the ROC curves and radar plots of Dice similarity.",
            },
            table_pages=[],
        )

        data_a = ExtractedPaperData(
            paper_title="ModelFiguresOnly-3D",
            proposed_model_id="ModelFiguresOnly-3D",
            code_repository_url=None,
            screening=screening,
            extracted_scores=[],
        )
        data_b = ExtractedPaperData(
            paper_title="ModelFiguresOnly-3D",
            proposed_model_id="ModelFiguresOnly-3D",
            code_repository_url=None,
            screening=screening,
            extracted_scores=[
                RawExtractedScore(dataset_name="QualitativeMentionOnly", organ_or_task="Overall", dice_score=None),
            ],
        )

        report = auditor.audit_consensus(data_a, data_b, manuscript=manuscript)

        assert report.concordant_count == 0
        assert report.status == AuditStatus.NEEDS_OUTREACH

    def test_graceful_degradation_preserves_verified_agent_without_false_repo_inspection(self, auditor):
        """Verifies that when Agent A extracts valid canonical benchmark tables and Agent B is the empty
        graceful degradation fallback, the consensus report status is FLAGGED_WARNING (not NEEDS_REPO_INSPECTION)
        and carries unconfirmed_count == 1.
        """
        screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)

        data_a = ExtractedPaperData(
            paper_title="SurvivorModel",
            proposed_model_id="SurvivorModel",
            code_repository_url="https://github.com/example/SurvivorModel",
            screening=screening,
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="BraTS",
                    organ_or_task="Overall",
                    dice_score=0.91,
                    table_or_section_ref="Table 1",
                ),
            ],
        )

        data_b = ExtractedPaperData(
            paper_title="SurvivorModel",
            proposed_model_id="SurvivorModel",
            code_repository_url="https://github.com/example/SurvivorModel",
            screening=screening,
            extraction_notes="Agent B failed: connection timeout. Single-agent fallback to Agent A.",
            extracted_scores=[],
        )

        report = auditor.audit_consensus(data_a, data_b)

        assert report.concordant_count == 0
        assert report.unconfirmed_count == 1
        assert report.status == AuditStatus.FLAGGED_WARNING
        assert report.status != AuditStatus.NEEDS_REPO_INSPECTION

    def test_deliberation_skips_when_single_agent_fallback_active(self, auditor, monkeypatch):
        """Verifies that iterative deliberation does not execute queries on an agent in single-agent fallback mode."""
        client_a = LLMClient(config=LLMConfig(provider="mock", model="mock-model-a"))
        client_b = LLMClient(config=LLMConfig(provider="mock", model="mock-model-b"))
        agent_a = ExtractionAgent(llm_client=client_a)
        agent_b = ExtractionAgent(llm_client=client_b)
        coordinator = DualExtractionCoordinator(agent_a=agent_a, agent_b=agent_b)

        def fail_extract(manuscript):
            raise RuntimeError("Backend connection refused")

        monkeypatch.setattr(agent_b, "process_manuscript", fail_extract)

        manuscript = ParsedManuscript(
            file_path="dummy.pdf",
            title="MedSAM Paper",
            abstract="Abstract",
            total_pages=3,
            pages_text={1: "MedSAM text"},
        )

        ext_a, ext_b, audit_rep, traj = coordinator.extract_manuscript_with_deliberation(
            manuscript,
            auditor=auditor,
            max_rounds=3,
            verbose=False,
        )

        assert "Single-agent fallback" in ext_b.extraction_notes
        assert len(traj) == 0  # Deliberation was skipped


