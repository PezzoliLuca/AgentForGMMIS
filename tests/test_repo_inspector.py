"""Unit tests for the GitHub RepoInspector module."""

import json
from pathlib import Path
import pytest

from agent_gmmis.auditing.auditor import AuditorAgent
from agent_gmmis.auditing.repo_inspector import RepoInspector
from agent_gmmis.auditing.schemas import AuditStatus, MissingDataDiagnosis
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.data.models import InclusionCriteriaCheck
from agent_gmmis.extraction.schemas import ExtractedPaperData, RawExtractedScore


@pytest.fixture
def inspector(tmp_path):
    """RepoInspector with a dedicated temporary cache directory."""
    cache_dir = tmp_path / "cache_repos"
    return RepoInspector(cache_dir=cache_dir)


@pytest.fixture
def auditor():
    """AuditorAgent configured for testing."""
    return AuditorAgent()


class TestRepoInspector:
    """Test suite for RepoInspector URL parsing, table parsing, file ingestion, and auditor integration."""

    def test_parse_github_url(self):
        """Validates robust extraction of owner and repository name across various URL schemes."""
        valid_cases = [
            ("https://github.com/microsoft/BiomedParse", ("microsoft", "BiomedParse")),
            ("https://github.com/microsoft/BiomedParse.git", ("microsoft", "BiomedParse")),
            ("https://github.com/microsoft/BiomedParse/tree/main/docs", ("microsoft", "BiomedParse")),
            ("https://microsoft.github.io/BiomedParse/", ("microsoft", "BiomedParse")),
            ("git@github.com:microsoft/BiomedParse.git", ("microsoft", "BiomedParse")),
            ("http://github.com/PezzoliLuca/AgentForGMMIS/", ("PezzoliLuca/AgentForGMMIS".split("/"))),
        ]
        for url, expected in valid_cases:
            assert RepoInspector.parse_github_url(url) == tuple(expected)

        invalid_cases = [
            "",
            "https://google.com/test",
            "not_a_valid_url",
            "https://gitlab.com/test/repo",
        ]
        for url in invalid_cases:
            assert RepoInspector.parse_github_url(url) is None

    def test_parse_markdown_tables_long_format(self, inspector):
        """Parses markdown table with explicit Dataset and Dice columns."""
        md_content = """
# BiomedParse Evaluation Results

| Dataset | Target Task | Dice (%) | HD95 |
| :--- | :--- | :---: | :---: |
| Synapse | Abdominal Multi-organ | 88.5 | 4.2 |
| BraTS21 | Brain Tumor | 91.2% | 3.1 |
| KiTS | Kidney Tumor | 0.795 | 8.0 |
"""
        scores = inspector._parse_markdown_tables(md_content, "README.md", model_name="BiomedParse")
        assert len(scores) == 3

        # Synapse should resolve to BTCV
        btcv = next(s for s in scores if s.dataset_name == "BTCV")
        assert btcv.dice_score == 0.885
        assert btcv.organ_or_task == "Abdominal Multi-organ"

        # BraTS21 should resolve to BraTS
        brats = next(s for s in scores if s.dataset_name == "BraTS")
        assert brats.dice_score == 0.912

        # KiTS
        kits = next(s for s in scores if s.dataset_name == "KiTS")
        assert kits.dice_score == 0.795

    def test_parse_markdown_tables_wide_format(self, inspector):
        """Parses wide comparison table where columns correspond to competing models."""
        md_content = """
### Benchmark Leaderboard

| Dataset | SwinUNETR | nnU-Net | BiomedParse (Ours) |
| :--- | :---: | :---: | :---: |
| BTCV | 82.3 | 84.5 | 88.5 |
| BraTS | 89.1 | 90.2 | 91.2 |
| AMOS | 85.0 | 87.1 | 89.4 |
"""
        scores = inspector._parse_markdown_tables(md_content, "README.md", model_name="BiomedParse")
        assert len(scores) == 3

        scores_by_ds = {s.dataset_name: s.dice_score for s in scores}
        assert scores_by_ds["BTCV"] == 0.885
        assert scores_by_ds["BraTS"] == 0.912
        assert scores_by_ds["AMOS"] == 0.894

    def test_parse_json_metrics_flat_and_nested(self, inspector, tmp_path):
        """Parses structured JSON metrics files in both flat and nested dictionary structures."""
        # Flat dict
        json_flat = tmp_path / "results_flat.json"
        json_flat.write_text(json.dumps({"BTCV": 0.885, "BraTS": 91.2, "unknown_dataset": 0.50}))

        scores_flat = inspector._parse_json_file(json_flat, "results_flat.json", model_name="TestModel")
        scores_by_ds = {s.dataset_name: s.dice_score for s in scores_flat}
        assert scores_by_ds["BTCV"] == 0.885
        assert scores_by_ds["BraTS"] == 0.912

        # Nested dict with model container
        json_nested = tmp_path / "eval_nested.json"
        nested_data = {
            "BiomedParse": {
                "BTCV": {"dice": 0.885, "hd95": 4.1},
                "KiTS": {"mean_dice": 0.782},
            }
        }
        json_nested.write_text(json.dumps(nested_data))

        scores_nested = inspector._parse_json_file(json_nested, "eval_nested.json", model_name="BiomedParse")
        nested_by_ds = {s.dataset_name: s.dice_score for s in scores_nested}
        assert nested_by_ds["BTCV"] == 0.885
        assert nested_by_ds["KiTS"] == 0.782

    def test_parse_json_metrics_list_format(self, inspector, tmp_path):
        """Parses list of benchmark records in JSON."""
        json_list = tmp_path / "test_metrics.json"
        records = [
            {"dataset": "BTCV", "organ": "Multi-organ", "dice": 0.885},
            {"dataset": "BraTS 2021", "organ": "Brain", "dsc": "91.2%"},
        ]
        json_list.write_text(json.dumps(records))

        scores = inspector._parse_json_file(json_list, "test_metrics.json", model_name="TestModel")
        assert len(scores) == 2
        scores_by_ds = {s.dataset_name: s.dice_score for s in scores}
        assert scores_by_ds["BTCV"] == 0.885
        assert scores_by_ds["BraTS"] == 0.912

    def test_parse_csv_metrics(self, inspector, tmp_path):
        """Parses tabular CSV files containing benchmark evaluations."""
        csv_file = tmp_path / "eval_metrics.csv"
        csv_file.write_text("dataset,organ,dice\nBTCV,Multi-organ,0.885\nAMOS,Abdomen,0.892\n")

        scores = inspector._parse_csv_file(csv_file, "eval_metrics.csv", model_name="TestModel")
        assert len(scores) == 2
        scores_by_ds = {s.dataset_name: s.dice_score for s in scores}
        assert scores_by_ds["BTCV"] == 0.885
        assert scores_by_ds["AMOS"] == 0.892

    def test_inspect_repository_mock_dir(self, inspector, tmp_path):
        """Inspects an entire repository directory and deduplicates results across multiple files."""
        repo_dir = tmp_path / "mock_repo"
        repo_dir.mkdir()

        # Add README with table
        readme = repo_dir / "README.md"
        readme.write_text(
            """
# Proposed Model
| Dataset | Dice (%) |
| :--- | :---: |
| BTCV | 88.5 |
| BraTS | 91.2 |
"""
        )

        # Add results.json with an additional dataset
        results_json = repo_dir / "results.json"
        results_json.write_text(json.dumps({"KiTS": 0.785}))

        res = inspector.inspect_repository(str(repo_dir), model_name="BiomedParse")
        assert res.success is True
        assert len(res.extracted_scores) == 3
        datasets = {s.dataset_name for s in res.extracted_scores}
        assert datasets == {"BTCV", "BraTS", "KiTS"}
        assert "README.md" in res.inspected_files
        assert "results.json" in res.inspected_files

    def test_inspect_repository_empty(self, inspector, tmp_path):
        """Handles empty repositories gracefully without crashing."""
        empty_dir = tmp_path / "empty_repo"
        empty_dir.mkdir()
        res = inspector.inspect_repository(str(empty_dir))
        assert res.success is False
        assert len(res.extracted_scores) == 0
        assert "found no quantitative benchmark tables" in res.message

    def test_inspect_and_update_with_auditor_recovery(self, inspector, auditor, tmp_path):
        """Validates end-to-end delegation flow:
        A paper with 0 scalar scores is initially diagnosed as NEEDS_REPO_INSPECTION.
        RepoInspector recovers the scores from repository artifacts and re-auditing transitions
        the paper to VERIFIED certification!
        """
        # 1. Create mock repo
        repo_dir = tmp_path / "biomedparse_repo"
        repo_dir.mkdir()
        readme = repo_dir / "README.md"
        readme.write_text(
            """
| Dataset | Organ | Dice |
| :--- | :--- | :---: |
| BTCV | Multi-organ | 0.885 |
| BraTS | Brain | 0.912 |
| KiTS | Kidney | 0.782 |
"""
        )

        # 2. Paper extracted data initially has 0 scores
        paper_data = ExtractedPaperData(
            paper_title="BiomedParse: A Multimodal Foundation Model",
            proposed_model_id="BiomedParse",
            visual_backbone="ViT-B",
            millions_of_parameters=110.0,
            code_repository_url=str(repo_dir),
            screening=InclusionCriteriaCheck(
                is_3d=True,
                is_radiological=True,
                is_public_dataset=True,
                has_dice_metric=True,
            ),
            extracted_scores=[],
        )

        # Initial audit -> NEEDS_REPO_INSPECTION
        initial_report = auditor.audit(paper_data)
        assert initial_report.status == AuditStatus.NEEDS_REPO_INSPECTION
        assert initial_report.data_diagnosis == MissingDataDiagnosis.CODE_REPO_AVAILABLE

        # 3. Delegation step: inspect_and_update
        updated_data, reaudited_report = inspector.inspect_and_update(paper_data, auditor=auditor)

        # 4. Check results
        assert len(updated_data.extracted_scores) == 3
        assert reaudited_report is not None
        assert reaudited_report.status == AuditStatus.VERIFIED
        assert reaudited_report.passed_sanity_checks is True
        assert reaudited_report.confidence_score >= 0.95
