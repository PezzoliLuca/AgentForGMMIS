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
