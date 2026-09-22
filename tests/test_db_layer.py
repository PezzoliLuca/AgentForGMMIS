"""Comprehensive unit and integration tests for the Data Layer and SQLite DAL."""

import pytest
from pydantic import ValidationError

from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.data.enums import ScoreCategory
from agent_gmmis.data.models import BenchmarkScore, DatasetMetadata, InclusionCriteriaCheck, ModelMetadata


@pytest.fixture
def db():
    """Initializes DatabaseManager connected to default data/database.sqlite."""
    return DatabaseManager()


class TestDatabaseManager:
    """Verifies SQLite data access and Pydantic validation on the GMMIS dataset."""

    def test_all_models_validation(self, db):
        """Ensures all 98 models in the database strictly adhere to the ModelMetadata schema."""
        models = db.get_all_models()
        assert len(models) == 98, f"Expected 98 models, found {len(models)}"
        assert all(isinstance(m, ModelMetadata) for m in models)

        # Check a known anchor model
        medsam = db.get_model("MedSAM")
        assert medsam is not None
        assert medsam.id == "MedSAM"
        assert "Segment anything" in medsam.title
        assert medsam.framework is not None
        assert medsam.visual_backbone is not None

    def test_all_datasets_validation(self, db):
        """Ensures all 58 datasets in the database strictly adhere to the DatasetMetadata schema."""
        datasets = db.get_all_datasets()
        assert len(datasets) == 58, f"Expected 58 datasets, found {len(datasets)}"
        assert all(isinstance(d, DatasetMetadata) for d in datasets)

        # Check known anchor datasets
        brats = db.get_dataset("BraTS")
        assert brats is not None
        assert brats.imaging_modality is not None
        assert brats.region == "Head"

        amos = db.get_dataset("AMOS")
        assert amos is not None
        assert "Abdominal" in amos.full_name or "Multi-Modality" in amos.full_name

    def test_model_scores_retrieval(self, db):
        """Verifies retrieval of primary and best scores for a model."""
        scores = db.get_scores_for_model("MedSAM")
        assert isinstance(scores, dict)
        assert len(scores) > 0, "MedSAM should have benchmark evaluations"

    def test_leaderboard_retrieval(self, db):
        """Verifies ranking and leaderboard generation for target datasets."""
        leaderboard = db.get_leaderboard_for_dataset("BraTS", category=ScoreCategory.BEST, top_k=5)
        assert len(leaderboard) > 0
        # Check that scores are monotonically non-increasing
        scores = [score for _, score in leaderboard]
        assert scores == sorted(scores, reverse=True)


class TestBenchmarkScoreValidation:
    """Verifies Pydantic validation rules and edge cases for benchmark metrics."""

    def test_valid_dice_scores(self):
        """Standard decimal Dice scores in [0.0, 1.0]."""
        score = BenchmarkScore(
            model_id="MedSAM",
            dataset_name="BraTS",
            dice_score=0.8875,
        )
        assert score.dice_score == 0.8875

    def test_percentage_dice_conversion(self):
        """Automatically converts percentage notation (e.g. 88.75% -> 0.8875)."""
        score = BenchmarkScore(
            model_id="MedSAM",
            dataset_name="BraTS",
            dice_score=88.75,
        )
        assert score.dice_score == 0.8875

    def test_invalid_dice_bounds(self):
        """Rejects impossible scores (> 100% or negative)."""
        with pytest.raises(ValidationError):
            BenchmarkScore(
                model_id="TestModel",
                dataset_name="TestDataset",
                dice_score=105.0,
            )

        with pytest.raises(ValidationError):
            BenchmarkScore(
                model_id="TestModel",
                dataset_name="TestDataset",
                dice_score=-0.1,
            )


class TestInclusionCriteria:
    """Verifies scientific screening logic according to Moglia et al."""

    def test_valid_inclusion(self):
        check = InclusionCriteriaCheck(
            is_3d=True,
            is_radiological=True,
            is_public_dataset=True,
            has_dice_metric=True,
        )
        assert check.is_included is True
        assert len(check.rejection_reasons) == 0

    def test_rejection_2d_and_private(self):
        check = InclusionCriteriaCheck(
            is_3d=False,
            is_radiological=True,
            is_public_dataset=False,
            has_dice_metric=True,
        )
        assert check.is_included is False
        reasons = check.rejection_reasons
        assert any("2D instead of 3D" in r for r in reasons)
        assert any("private" in r.lower() for r in reasons)
