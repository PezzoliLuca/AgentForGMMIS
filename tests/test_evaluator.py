"""Unit tests for ExtractionEvaluator: verifies evaluation integrity, bounds, and absence of data leakage."""

import pytest
from unittest.mock import MagicMock
from agent_gmmis.extraction.evaluator import ExtractionEvaluator, ExtractionEvaluationReport
from agent_gmmis.extraction.schemas import ExtractedPaperData, RawExtractedScore
from agent_gmmis.data.models import ModelMetadata


class TestExtractionEvaluatorIntegrity:
    """Rigorous evaluation suite ensuring mathematical consistency and epistemic fairness."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        # Mock a model with 2 GT datasets
        db.get_model.return_value = ModelMetadata(
            id="TestModel",
            title="Test Paper Title",
            millions_of_parameters=100.0,
            resources_total_vram=16,
            code="https://github.com/test/repo",
        )
        db.get_scores_for_model.return_value = {
            "BraTS": {"primary": 85.0, "best": 88.0},
            "Synapse": {"primary": 80.0, "best": 85.0},
        }
        return db

    def test_recall_never_exceeds_100_percent_on_duplicate_matches(self, mock_db):
        """Verifies that multiple variants matching the same GT dataset do not inflate Recall > 100%."""
        evaluator = ExtractionEvaluator(db_manager=mock_db)

        # Extracted data has 3 entries: 1 for BraTS, and 2 separate alias variants for Synapse/BTCV
        extracted = ExtractedPaperData(
            paper_title="Test Paper",
            proposed_model_id="TestModel",
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="BraTS 2018",
                    organ_or_task="Overall",
                    dice_score=0.85,
                ),
                RawExtractedScore(
                    dataset_name="BTCV",
                    organ_or_task="Overall",
                    dice_score=0.80,
                ),
                RawExtractedScore(
                    dataset_name="Synapse multi-organ CT",
                    organ_or_task="Overall",
                    dice_score=0.81,
                ),
            ],
        )

        report = evaluator.evaluate(extracted, ground_truth_id="TestModel", compare_source="primary")

        # 2 GT datasets (BraTS, Synapse). Both matched, but Synapse matched once!
        assert report.ground_truth_datasets_count == 2
        assert report.matched_datasets_count == 2
        assert report.dataset_recall == 1.0  # Exactly 100%, NOT 150%!
        assert report.dataset_recall <= 1.0
        assert report.dataset_precision == 1.0
        # Score comparisons should have exactly 2 entries, not 3
        assert len(report.score_comparisons) == 2
        matched_datasets = [c["dataset"] for c in report.score_comparisons]
        assert len(matched_datasets) == len(set(matched_datasets))

    def test_no_oracle_data_leakage_cherry_picking(self, mock_db):
        """Verifies that candidate selection does NOT peek at true_dice to choose the closest score."""
        evaluator = ExtractionEvaluator(db_manager=mock_db)

        # Extracted has 2 scores for BraTS:
        # 1. 1-point prompt (0.75)
        # 2. Peak bounding box (0.95)
        # Note: GT primary is 0.85. The midpoint is 0.85.
        extracted = ExtractedPaperData(
            paper_title="Test Paper",
            proposed_model_id="TestModel",
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="BraTS",
                    organ_or_task="Overall",
                    prompt_type="1 point prompt",
                    dice_score=0.75,
                ),
                RawExtractedScore(
                    dataset_name="BraTS",
                    organ_or_task="Overall",
                    prompt_type="bounding box",
                    dice_score=0.95,
                ),
            ],
        )

        # In compare_source='primary', selection must prefer 1-point prompt (0.75),
        # even though 0.95 and 0.75 are equidistant from 0.85 or if one was closer.
        report_primary = evaluator.evaluate(extracted, ground_truth_id="TestModel", compare_source="primary")
        comp_primary = next(c for c in report_primary.score_comparisons if c["dataset"] == "BraTS")
        assert comp_primary["predicted"] == 0.75
        assert comp_primary["diff"] == 0.10  # |0.75 - 0.85|

        # In compare_source='best', selection must pick the peak candidate (0.95)
        report_best = evaluator.evaluate(extracted, ground_truth_id="TestModel", compare_source="best")
        comp_best = next(c for c in report_best.score_comparisons if c["dataset"] == "BraTS")
        assert comp_best["predicted"] == 0.95
        assert comp_best["diff"] == 0.07  # |0.95 - 0.88|

    def test_macro_average_computed_blindly_when_no_overall_row(self, mock_db):
        """Verifies mathematical mean is computed across individual organs without peeking at GT."""
        evaluator = ExtractionEvaluator(db_manager=mock_db)

        extracted = ExtractedPaperData(
            paper_title="Test Paper",
            proposed_model_id="TestModel",
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="Synapse",
                    organ_or_task="Liver",
                    dice_score=0.90,
                ),
                RawExtractedScore(
                    dataset_name="Synapse",
                    organ_or_task="Spleen",
                    dice_score=0.70,
                ),
            ],
        )

        report = evaluator.evaluate(extracted, ground_truth_id="TestModel", compare_source="primary")
        comp = next(c for c in report.score_comparisons if c["dataset"] in ("Synapse", "BTCV"))
        # Mean of 0.90 and 0.70 is 0.80
        assert comp["predicted"] == 0.80
        assert comp["ground_truth"] == 0.80
        assert comp["diff"] == 0.0

    def test_dual_evaluation_reports_both_primary_and_best(self, mock_db):
        """Verifies report simultaneously calculates MAE against Primary, Best, and Regime-Agnostic."""
        evaluator = ExtractionEvaluator(db_manager=mock_db)

        # BraTS GT: primary=85.0 (0.85), best=88.0 (0.88)
        # Synapse GT: primary=80.0 (0.80), best=85.0 (0.85)
        extracted = ExtractedPaperData(
            paper_title="Test Paper",
            proposed_model_id="TestModel",
            extracted_scores=[
                RawExtractedScore(
                    dataset_name="BraTS",
                    organ_or_task="Overall",
                    prompt_type="1 point",
                    dice_score=0.85,  # matches primary exactly (diff 0.0), diff best 0.03
                ),
                RawExtractedScore(
                    dataset_name="Synapse",
                    organ_or_task="Overall",
                    prompt_type="1 point",
                    dice_score=0.85,  # matches best exactly (diff 0.0), diff primary 0.05
                ),
            ],
        )

        report = evaluator.evaluate(extracted, ground_truth_id="TestModel", compare_source="primary")
        assert report.dice_mae_primary is not None
        assert report.dice_mae_best is not None
        assert report.dice_mae_regime_agnostic is not None

        # Primary diffs: BraTS |0.85-0.85|=0.0, Synapse |0.85-0.80|=0.05 -> MAE primary = 0.025
        assert abs(report.dice_mae_primary - 0.025) < 1e-4

        # Best diffs: BraTS |0.85-0.88|=0.03, Synapse |0.85-0.85|=0.0 -> MAE best = 0.015
        assert abs(report.dice_mae_best - 0.015) < 1e-4

        # Regime-agnostic: BraTS matches primary (0.0), Synapse matches best (0.0) -> MAE = 0.000
        assert abs(report.dice_mae_regime_agnostic - 0.0) < 1e-4

    def test_evaluator_and_table_format_with_null_primary_ground_truth(self, mock_db):
        """
        Verifies that when a dataset is in results_best but NULL in results_primary,
        evaluating against primary does not crash and comp_rows formatting handles None safely.
        """
        mock_db.get_scores_for_model.return_value = {
            "BraTS": {"primary": 85.0, "best": 88.0},
            "BTCV": {"primary": None, "best": 82.0},
        }
        evaluator = ExtractionEvaluator(db_manager=mock_db)

        extracted = ExtractedPaperData(
            paper_title="Test Paper",
            proposed_model_id="TestModel",
            extracted_scores=[
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Overall", dice_score=0.85),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.82),
            ],
        )

        report = evaluator.evaluate(extracted, ground_truth_id="TestModel", compare_source="primary")

        # BTCV has no primary GT, so only BraTS counts as matched on primary
        assert report.ground_truth_datasets_count == 1
        assert report.matched_datasets_count == 1
        assert report.dataset_recall == 1.0  # 1/1

        # Safe table row formatting check (same logic as in run_batch_benchmark and run_extraction)
        comp_rows = [
            [
                c["dataset"],
                c.get("prompt_type", "N/A"),
                f"{c['predicted']:.4f}" if c.get("predicted") is not None else "N/A",
                f"{c['ground_truth']:.4f}" if c.get("ground_truth") is not None else (
                    f"{c['ground_truth_best']:.4f} (best)" if c.get("ground_truth_best") is not None else "N/A"
                ),
                f"{c['diff']:.4f}" if c.get("diff") is not None else (
                    f"{c['diff_best']:.4f} (best)" if c.get("diff_best") is not None else "N/A"
                ),
            ]
            for c in report.score_comparisons
        ]
        assert len(comp_rows) == 2
        btcv_row = next(r for r in comp_rows if r[0] == "BTCV")
        assert "0.8200 (best)" in btcv_row[3]
