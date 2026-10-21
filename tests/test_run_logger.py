"""Unit tests for RunLogger and intermediate agent extraction logging."""

import json
from pathlib import Path
import pytest

from agent_gmmis.auditing.schemas import (
    AuditAnomaly,
    AuditReport,
    AuditStatus,
    ConsensusAuditReport,
    ConsensusMetadata,
    ConsensusScore,
    ConsensusScreening,
    ConsensusStatus,
)
from agent_gmmis.data.models import InclusionCriteriaCheck
from agent_gmmis.extraction.run_logger import LoggerTee, RunLogger, archive_legacy_flat_logs, sanitize_filename
from agent_gmmis.extraction.schemas import ExtractedPaperData, RawExtractedScore


@pytest.fixture
def mock_extracted_a():
    return ExtractedPaperData(
        paper_title="Segment Anything in Medical Images",
        proposed_model_id="MedSAM",
        visual_backbone="ViT-B",
        millions_of_parameters=93.7,
        code_repository_url="https://github.com/bowang-lab/MedSAM",
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
                dice_score=0.8850,
                table_or_section_ref="Table 1",
            ),
            RawExtractedScore(
                dataset_name="BTCV",
                organ_or_task="Multi-Organ",
                dice_score=0.8120,
                table_or_section_ref="Table 2",
            ),
        ],
    )


@pytest.fixture
def mock_extracted_b():
    return ExtractedPaperData(
        paper_title="Segment Anything in Medical Images",
        proposed_model_id="MedSAM",
        visual_backbone="ViT-B",
        millions_of_parameters=93.7,
        code_repository_url="https://github.com/bowang-lab/MedSAM",
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
                dice_score=0.8845,
                table_or_section_ref="Table 1",
            ),
            RawExtractedScore(
                dataset_name="BTCV",
                organ_or_task="Multi-Organ",
                dice_score=0.8115,
                table_or_section_ref="Table 2",
            ),
        ],
    )


@pytest.fixture
def mock_consensus_report():
    report_a = AuditReport(
        model_id="MedSAM",
        status=AuditStatus.VERIFIED,
        confidence_score=0.95,
        passed_sanity_checks=True,
        passed_inclusion_screening=True,
    )
    report_b = AuditReport(
        model_id="MedSAM",
        status=AuditStatus.VERIFIED,
        confidence_score=0.95,
        passed_sanity_checks=True,
        passed_inclusion_screening=True,
    )
    return ConsensusAuditReport(
        model_id="MedSAM",
        status=AuditStatus.VERIFIED,
        confidence_score=0.98,
        concordance_rate=1.0,
        inter_model_dice_mae=0.0005,
        concordant_count=2,
        dispute_count=0,
        unconfirmed_count=0,
        paired_count=2,
        total_unique_benchmarks=2,
        audit_report_a=report_a,
        audit_report_b=report_b,
        consensus_screening=ConsensusScreening(
            status=ConsensusStatus.CONCORDANT,
            is_included=True,
            is_3d_a=True,
            is_3d_b=True,
            is_radiological_a=True,
            is_radiological_b=True,
            is_public_a=True,
            is_public_b=True,
            has_dice_a=True,
            has_dice_b=True,
        ),
        consensus_metadata=ConsensusMetadata(
            model_id_status=ConsensusStatus.CONCORDANT,
            resolved_model_id="MedSAM",
            model_id_a="MedSAM",
            model_id_b="MedSAM",
            params_status=ConsensusStatus.CONCORDANT,
            resolved_params_m=93.7,
            params_m_a=93.7,
            params_m_b=93.7,
            visual_backbone_a="ViT-B",
            visual_backbone_b="ViT-B",
            resolved_visual_backbone="ViT-B",
        ),
        consensus_scores=[
            ConsensusScore(
                dataset_name="BraTS",
                canonical_dataset_name="BraTS",
                organ_or_task="Brain Tumor",
                status=ConsensusStatus.CONCORDANT,
                score_a=0.8850,
                score_b=0.8845,
                resolved_score=0.8850,
                delta=0.0005,
            ),
            ConsensusScore(
                dataset_name="BTCV",
                canonical_dataset_name="BTCV",
                organ_or_task="Multi-Organ",
                status=ConsensusStatus.CONCORDANT,
                score_a=0.8120,
                score_b=0.8115,
                resolved_score=0.8120,
                delta=0.0005,
            ),
        ],
    )


class TestRunLogger:
    """Verifies RunLogger initialization, artifact persistence, and intermediate outputs."""

    def test_run_directory_creation(self, tmp_path):
        logger = RunLogger(base_dir=tmp_path, tag="test_run", capture_stdout=False)
        assert logger.run_dir.exists()
        assert logger.extractions_dir.exists()
        assert logger.manifest_file.exists()

        with open(logger.manifest_file, "r") as f:
            manifest = json.load(f)
        assert manifest["is_dual"] is False
        assert "paths" in manifest
        logger.close()

    def test_single_agent_logging(self, tmp_path, mock_extracted_a):
        logger = RunLogger(base_dir=tmp_path, capture_stdout=False)
        audit_rep = AuditReport(
            model_id="MedSAM",
            status=AuditStatus.VERIFIED,
            confidence_score=0.90,
            passed_sanity_checks=True,
            passed_inclusion_screening=True,
        )

        paper_dir = logger.log_paper_extraction(
            model_id="MedSAM",
            extracted_a=mock_extracted_a,
            audit_report=audit_rep,
            final_extracted=mock_extracted_a,
            elapsed_seconds=12.5,
        )

        assert paper_dir.exists()
        assert (paper_dir / "agent_a_extracted.json").exists()
        assert not (paper_dir / "agent_b_extracted.json").exists()
        assert (paper_dir / "audit_report.json").exists()
        assert (paper_dir / "final_reconciled.json").exists()
        assert (paper_dir / "paper_summary.json").exists()

        with open(paper_dir / "agent_a_extracted.json") as f:
            loaded_a = json.load(f)
        assert loaded_a["proposed_model_id"] == "MedSAM"
        assert len(loaded_a["extracted_scores"]) == 2
        logger.close()

    def test_dual_agent_intermediate_logging(
        self, tmp_path, mock_extracted_a, mock_extracted_b, mock_consensus_report
    ):
        logger = RunLogger(base_dir=tmp_path, is_dual=True, tag="dual_test", capture_stdout=False)
        paper_dir = logger.log_paper_extraction(
            model_id="MedSAM",
            extracted_a=mock_extracted_a,
            extracted_b=mock_extracted_b,
            audit_report=mock_consensus_report,
            final_extracted=mock_extracted_a,
            elapsed_seconds=25.0,
        )

        assert paper_dir.exists()
        assert (paper_dir / "agent_a_extracted.json").exists()
        assert (paper_dir / "agent_b_extracted.json").exists()
        assert (paper_dir / "consensus_matrix.json").exists()
        assert (paper_dir / "audit_report.json").exists()
        assert (paper_dir / "final_reconciled.json").exists()

        # Check content of consensus matrix
        with open(paper_dir / "consensus_matrix.json") as f:
            cons = json.load(f)
        assert cons["concordance_rate"] == 1.0
        assert cons["inter_model_dice_mae"] == 0.0005
        assert len(cons["consensus_scores"]) == 2
        assert cons["consensus_scores"][0]["score_a"] == 0.8850
        assert cons["consensus_scores"][0]["score_b"] == 0.8845

        logger.close()

    def test_save_run_results(self, tmp_path):
        logger = RunLogger(base_dir=tmp_path, tag="summary_test", capture_stdout=False)
        results = {"metadata": {"run_id": logger.run_id}, "summary": []}
        report_md = "# Benchmark Report\n\nAll tests passed."

        logger.save_run_results(results, report_md, update_latest_symlinks=False)
        assert logger.results_file.exists()
        assert logger.report_file.exists()

        with open(logger.results_file) as f:
            loaded_res = json.load(f)
        assert loaded_res["metadata"]["run_id"] == logger.run_id

        with open(logger.report_file) as f:
            loaded_md = f.read()
        assert "All tests passed" in loaded_md
        logger.close()

    def test_log_paper_extraction_with_deliberation_trajectory(self, tmp_path, mock_extracted_a, mock_extracted_b, mock_consensus_report):
        from agent_gmmis.extraction.schemas import DeliberationRoundSummary
        logger = RunLogger(base_dir=tmp_path, run_name="test_delib_logging", capture_stdout=False)

        summary_r2 = DeliberationRoundSummary(
            round_number=2,
            disputes_before=1,
            unconfirmed_before=1,
            concordant_before=2,
            concordance_rate_before=0.6667,
            inter_model_mae_before=0.0350,
            disputes_after=0,
            unconfirmed_after=0,
            concordant_after=4,
            concordance_rate_after=1.0,
            inter_model_mae_after=0.0005,
            resolved_count=2,
            stalled=False,
            details=["Agent B confirmed omitted BraTS with score 0.8850"],
        )

        paper_dir = logger.log_paper_extraction(
            model_id="MedSAM",
            extracted_a=mock_extracted_a,
            extracted_b=mock_extracted_b,
            audit_report=mock_consensus_report,
            deliberation_trajectory=[summary_r2],
        )

        traj_file = paper_dir / "deliberation_trajectory.json"
        assert traj_file.exists()
        with open(traj_file) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["round_number"] == 2
        assert data[0]["resolved_count"] == 2
        assert data[0]["concordance_rate_after"] == 1.0

        summary_file = paper_dir / "paper_summary.json"
        with open(summary_file) as f:
            summary = json.load(f)
        assert summary["artifacts"]["deliberation_trajectory"] == "deliberation_trajectory.json"
        assert summary["deliberation_rounds_count"] == 1
        logger.close()

    def test_sanitize_filename(self):
        assert sanitize_filename("MedSAM-2") == "MedSAM_2"
        assert sanitize_filename("Medical SAM 2 (MedSAM-2)") == "Medical_SAM_2_MedSAM_2"
        assert sanitize_filename("Qwen/Qwen2.5-14B") == "Qwen_Qwen2_5_14B"


class TestArchiveLegacyLogs:
    """Verifies moving legacy flat log files into archive/legacy_flat_logs/."""

    def test_archive_flat_logs(self, tmp_path):
        # Create some fake flat log files
        (tmp_path / "batch_benchmark_20261001_120000.log").write_text("log")
        (tmp_path / "batch_benchmark_report_20261001_120000.md").write_text("report")
        (tmp_path / "mufasa_bench_9999.out").write_text("slurm out")
        (tmp_path / "latest_batch_benchmark.json").write_text("{}")
        sub = tmp_path / "runs"
        sub.mkdir()

        archived = archive_legacy_flat_logs(logs_dir=tmp_path)
        assert archived == 3
        # Ensure latest_batch_benchmark was preserved in root
        assert (tmp_path / "latest_batch_benchmark.json").exists()
        # Ensure archived files moved
        archive_dir = tmp_path / "archive" / "legacy_flat_logs"
        assert (archive_dir / "batch_benchmark_20261001_120000.log").exists()
        assert (archive_dir / "batch_benchmark_report_20261001_120000.md").exists()
        assert (archive_dir / "mufasa_bench_9999.out").exists()
