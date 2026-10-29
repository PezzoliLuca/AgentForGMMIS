"""Unit tests for AuditorAgent ('Il Vigile') - Phase 3.1."""

import pytest
from agent_gmmis.auditing.auditor import AuditorAgent
from agent_gmmis.auditing.schemas import AuditStatus, MissingDataDiagnosis
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.data.models import InclusionCriteriaCheck
from agent_gmmis.extraction.pdf_parser import ParsedManuscript
from agent_gmmis.extraction.schemas import ExtractedPaperData, RawExtractedScore


@pytest.fixture
def db():
    return DatabaseManager()


@pytest.fixture
def auditor(db):
    return AuditorAgent(db_manager=db)


class TestAuditorAgent:
    """Verifies all auditing, sanity-checking, outlier, and screening rules."""

    def test_audit_verified_valid_paper(self, auditor):
        """Standard valid generalist model (MedSAM) must be VERIFIED."""
        data = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            visual_backbone="ViT-B",
            architecture="SAM, Transformer with Convolutions",
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

        report = auditor.audit(data)
        assert report.status == AuditStatus.VERIFIED
        assert report.passed_sanity_checks is True
        assert report.passed_inclusion_screening is True
        assert len(report.anomalies) == 0
        assert len(report.outliers) == 0
        assert report.confidence_score >= 0.95

    def test_audit_rejected_inclusion_criteria(self, auditor):
        """Paper violating 3D or public dataset criteria must be REJECTED."""
        data = ExtractedPaperData(
            paper_title="2D Retinal Segmentation Model",
            proposed_model_id="RetinaNet2D",
            screening=InclusionCriteriaCheck(
                is_3d=False,  # 2D only
                is_radiological=False,  # Fundus
                is_public_dataset=False,  # Private clinic
                has_dice_metric=True,
            ),
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="PrivateRetina",
                    organ_or_task="Vessels",
                    dice_score=0.8200,
                    is_baseline_comparison=False,
                )
            ],
        )

        report = auditor.audit(data)
        assert report.status == AuditStatus.REJECTED
        assert report.passed_inclusion_screening is False
        assert len(report.rejection_reasons) >= 3
        assert report.confidence_score == 0.0

    def test_audit_rejected_impossible_parameters(self, auditor):
        """Negative or absurdly huge parameter counts must fail sanity checks."""
        data = ExtractedPaperData(
            paper_title="Broken Param Model",
            proposed_model_id="BrokenModel",
            millions_of_parameters=-12.5,  # Impossible negative
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
                    dice_score=0.7500,
                    is_baseline_comparison=False,
                )
            ],
        )

        report = auditor.audit(data)
        assert report.status == AuditStatus.REJECTED
        assert report.passed_sanity_checks is False
        assert any("strictly positive" in a.message for a in report.anomalies)

    def test_audit_architectural_parameter_mismatch(self, auditor):
        """ViT-B claiming only 5M parameters must raise an architectural warning."""
        data = ExtractedPaperData(
            paper_title="ViT Model with Typo in Params",
            proposed_model_id="TinyViT",
            visual_backbone="ViT-B",
            millions_of_parameters=5.2,  # Way too small for ViT-B (typical ~85-150M)
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
                    dice_score=0.7500,
                    is_baseline_comparison=False,
                )
            ],
        )

        report = auditor.audit(data)
        assert report.status == AuditStatus.FLAGGED_WARNING
        assert any("ViT-B" in a.message for a in report.anomalies)

    def test_audit_statistical_outlier_detection(self, auditor):
        """Score claiming 0.99 on Pancreas tumor must trigger an outlier warning."""
        data = ExtractedPaperData(
            paper_title="Superhuman Pancreas Model",
            proposed_model_id="SuperPancreas",
            screening=InclusionCriteriaCheck(
                is_3d=True,
                is_radiological=True,
                is_public_dataset=True,
                has_dice_metric=True,
            ),
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="MSD Pancreas Tumour",
                    organ_or_task="Pancreas Tumour",
                    dice_score=0.9850,  # Far above SOTA (~0.77)
                    is_baseline_comparison=False,
                )
            ],
        )

        report = auditor.audit(data)
        assert report.status == AuditStatus.FLAGGED_WARNING
        assert len(report.outliers) > 0
        assert any("Pancreas" in o.reason for o in report.outliers)

    def test_audit_human_survey_discrepancy_lhu_net(self, auditor):
        """Detects the human survey copy-paste discrepancy in LHU-Net Table 2 (87.83% vs 87.49%)."""
        data = ExtractedPaperData(
            paper_title="LHU-Net: A Lightweight Hybrid U-Net",
            proposed_model_id="LHU-Net",
            screening=InclusionCriteriaCheck(
                is_3d=True,
                is_radiological=True,
                is_public_dataset=True,
                has_dice_metric=True,
            ),
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="BTCV",
                    organ_or_task="Abdomen Multi-Organ Mean",
                    dice_score=0.8783,  # Actual paper Table 2 score
                    is_baseline_comparison=False,
                )
            ],
        )

        report = auditor.audit(data, benchmark_mode=True)
        # In database.sqlite, LHU-Net on BTCV is recorded as 87.49% (from baseline TransUNet column)
        assert len(report.discrepancies) > 0
        assert report.discrepancies[0].dataset_name == "BTCV"
        assert report.discrepancies[0].extracted_score == 0.8783
        assert report.discrepancies[0].survey_ground_truth == 0.8749

    def test_audit_missing_data_diagnosis(self, auditor):
        """Diagnoses papers without scalar tables: figures present vs repo available vs no data."""
        # Case A: Figures present in manuscript -> NEEDS_OUTREACH
        manuscript_with_figs = ParsedManuscript(
            file_path="mock.pdf",
            total_pages=10,
            pages_text={
                1: "Introduction to our model.",
                5: "Figure 7: Comparison of Dice score (DSC) across different modalities using radar plot.",
            },
        )
        data_figs = ExtractedPaperData(
            paper_title="Graphical Only Paper",
            proposed_model_id="GraphModel",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[],
        )
        report_figs = auditor.audit(data_figs, manuscript=manuscript_with_figs)
        assert report_figs.status == AuditStatus.NEEDS_OUTREACH
        assert report_figs.data_diagnosis == MissingDataDiagnosis.MISSING_TABLE_FIGURES_FOUND

        # Case B: No figures, but official code repository disclosed -> NEEDS_REPO_INSPECTION
        data_repo = ExtractedPaperData(
            paper_title="Repo Based Paper",
            proposed_model_id="BiomedParse",
            code_repository_url="https://github.com/microsoft/BiomedParse",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[],
        )
        report_repo = auditor.audit(data_repo)
        assert report_repo.status == AuditStatus.NEEDS_REPO_INSPECTION
        assert report_repo.data_diagnosis == MissingDataDiagnosis.CODE_REPO_AVAILABLE

        # Case C: No figures, no repo, 0 scores -> REJECTED
        data_empty = ExtractedPaperData(
            paper_title="Empty Paper",
            proposed_model_id="EmptyModel",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[],
        )
        report_empty = auditor.audit(data_empty)
        assert report_empty.status == AuditStatus.REJECTED
        assert report_empty.data_diagnosis == MissingDataDiagnosis.MISSING_TABLE_NO_DATA

    def test_audit_baseline_model_confusion(self, auditor):
        """Flag when extractor mistakes a competitor baseline (e.g. nnU-Net) as proposed model."""
        data = ExtractedPaperData(
            paper_title="MambaSeg: A State Space Architecture for 3D CT",
            proposed_model_id="nnU-Net",  # Extractor accidentally picked baseline
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Tumor", dice_score=0.8500)
            ],
        )
        report = auditor.audit(data)
        assert report.status == AuditStatus.REJECTED
        assert any("standard baseline 'nnU-Net'" in a.message for a in report.anomalies)

    def test_audit_baseline_model_acronym_not_confused(self, auditor):
        """Intelligently distinguish between true baseline confusion and legitimate acronym/abbreviation (e.g. Med-SA)."""
        data = ExtractedPaperData(
            paper_title="Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation",
            proposed_model_id="Med-SA",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.8830)
            ],
        )
        report = auditor.audit(data)
        # Must NOT be falsely rejected as a baseline confusion
        assert report.status != AuditStatus.REJECTED
        assert not any("standard baseline 'Med-SA'" in a.message for a in report.anomalies)

    def test_audit_conflicting_duplicate_scores(self, auditor):
        """Flag when model produces two contradictory scores on the same dataset/task/prompt."""
        data = ExtractedPaperData(
            paper_title="Novel Segmentation Model",
            proposed_model_id="NovelNet",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Brain Tumor", dice_score=0.8200, prompt_type="1 point"),
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Brain Tumor", dice_score=0.8950, prompt_type="1 point"),  # Conflict!
            ],
        )
        report = auditor.audit(data)
        assert report.status == AuditStatus.FLAGGED_WARNING
        assert any("Conflicting duplicate Dice scores" in a.message for a in report.anomalies)

    def test_audit_cloned_template_hallucination(self, auditor):
        """Flag when extractor outputs suspiciously identical numbers across multiple datasets."""
        data = ExtractedPaperData(
            paper_title="Hallucinated Model",
            proposed_model_id="HallucinatedNet",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Brain", dice_score=0.8500),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Abdomen", dice_score=0.8500),
                RawExtractedScore(dataset_name="AMOS", organ_or_task="Multi", dice_score=0.8500),
                RawExtractedScore(dataset_name="KiTS", organ_or_task="Kidney", dice_score=0.8500),  # 4x identical 0.8500!
            ],
        )
        report = auditor.audit(data)
        assert report.status == AuditStatus.REJECTED
        assert any("Suspiciously identical Dice score" in a.message for a in report.anomalies)

    def test_audit_single_gpu_vram_inconsistency(self, auditor):
        """Flag when paper reports 1 GPU but claims impossible VRAM (>120GB)."""
        data = ExtractedPaperData(
            paper_title="Single GPU Paper with VRAM Typo",
            proposed_model_id="SingleGpuNet",
            resources="1x NVIDIA RTX 3090",
            resources_total_vram_gb=512,  # A single 3090 has 24GB, not 512GB!
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Tumor", dice_score=0.8500)
            ],
        )
        report = auditor.audit(data)
        assert report.status == AuditStatus.FLAGGED_WARNING
        assert any("Paper reports single GPU" in a.message for a in report.anomalies)

    def test_audit_domain_knowledge_3d_override_acdc(self, auditor):
        """Auditor overrides LLM false rejection on is_3d when paper evaluates on canonical 3D benchmark (ACDC)."""
        data = ExtractedPaperData(
            paper_title="KnowSAM: Learnable Prompting SAM for Semi-supervised Medical Image Segmentation",
            proposed_model_id="KnowSAM",
            screening=InclusionCriteriaCheck(
                is_3d=False,  # LLM erroneously flagged False because of 2D slice inference
                is_radiological=True,
                is_public_dataset=True,
                has_dice_metric=True,
                notes="2D slice-by-slice evaluation on ACDC cardiac MRI",
            ),
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="ACDC",
                    organ_or_task="Myocardium",
                    dice_score=0.8956,
                    is_baseline_comparison=False,
                )
            ],
        )
        report = auditor.audit(data)
        assert report.passed_inclusion_screening is True
        assert report.status != AuditStatus.REJECTED

    def test_audit_semi_supervised_budgets_not_conflicting(self, auditor):
        """Multiple scores across differing labeled data ratios (1%, 10%, 100%) are legitimate experiments, not contradictions."""
        data = ExtractedPaperData(
            paper_title="SFR SAM: Semi-supervised 3D Segmentation",
            proposed_model_id="SFR-SAM",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="avg", data_budget_or_setting="1% labeled", dice_score=0.7009),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="avg", data_budget_or_setting="10% labeled", dice_score=0.7707),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="avg", data_budget_or_setting="100% labeled", dice_score=0.8250),
            ],
        )
        report = auditor.audit(data)
        assert not any("Conflicting duplicate Dice scores" in a.message for a in report.anomalies)
        assert report.confidence_score >= 0.85

    def test_audit_graphical_missing_scores_merlin(self, auditor):
        """Scores with dice_score=None (from figure captions) correctly trigger NEEDS_OUTREACH without crashing."""
        manuscript = ParsedManuscript(
            file_path="mock_merlin.pdf",
            total_pages=12,
            pages_text={
                6: "Figure 4: Comparative evaluation across modalities showing bar chart of Dice score (DSC).",
            },
        )
        data = ExtractedPaperData(
            paper_title="Merlin: A Generalist Foundation Model",
            proposed_model_id="Merlin",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[
                RawExtractedScore(dataset_name="AMOS", organ_or_task="Multi-organ", dice_score=None, table_or_section_ref="Figure 4"),
            ],
        )
        report = auditor.audit(data, manuscript=manuscript, repo_inspected=True)
        assert report.status == AuditStatus.NEEDS_OUTREACH
        assert report.data_diagnosis == MissingDataDiagnosis.MISSING_TABLE_FIGURES_FOUND


