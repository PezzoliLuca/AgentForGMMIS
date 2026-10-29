"""Unit and integration tests for DatabaseAgent ('Database Manager Agent')."""

import shutil
import sqlite3
import tempfile
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
from agent_gmmis.config import DEFAULT_DB_PATH
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.database.database_agent import DatabaseAgent
from agent_gmmis.database.schemas import CurationActionType


@pytest.fixture
def temp_db() -> Path:
    """Creates a temporary isolated copy of database.sqlite for testing."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / "test_database.sqlite"
    shutil.copyfile(DEFAULT_DB_PATH, temp_path)
    yield temp_path
    shutil.rmtree(temp_dir)


@pytest.fixture
def db_manager(temp_db: Path) -> DatabaseManager:
    return DatabaseManager(temp_db)


@pytest.fixture
def database_agent(db_manager: DatabaseManager) -> DatabaseAgent:
    return DatabaseAgent(db_manager, single_agent_min_conf=0.85)


def make_dummy_consensus_report(
    model_id: str = "TestModel-3D",
    status: AuditStatus = AuditStatus.VERIFIED,
    confidence: float = 0.95,
    scores: list = None,
) -> ConsensusAuditReport:
    """Helper to create synthetic ConsensusAuditReport objects."""
    screening = ConsensusScreening(
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
    )
    metadata = ConsensusMetadata(
        model_id_status=ConsensusStatus.CONCORDANT,
        resolved_model_id=model_id,
        model_id_a=model_id,
        model_id_b=model_id,
        params_status=ConsensusStatus.CONCORDANT,
    )
    dummy_report_a = AuditReport(
        model_id=model_id,
        status=status,
        passed_sanity_checks=True,
        passed_inclusion_screening=True,
        confidence_score=confidence,
    )
    dummy_report_b = AuditReport(
        model_id=model_id,
        status=status,
        passed_sanity_checks=True,
        passed_inclusion_screening=True,
        confidence_score=confidence,
    )

    return ConsensusAuditReport(
        model_id=model_id,
        status=status,
        consensus_scores=scores or [],
        consensus_screening=screening,
        consensus_metadata=metadata,
        confidence_score=confidence,
        audit_report_a=dummy_report_a,
        audit_report_b=dummy_report_b,
        concordance_rate=1.0,
    )


# ----------------------------------------------------------------------
# 1. Gatekeeping Tests
# ----------------------------------------------------------------------
def test_gatekeeping_rejected_audit(database_agent: DatabaseAgent):
    """Verifies that REJECTED audits cannot be committed."""
    report = make_dummy_consensus_report(
        model_id="RejectedModel",
        status=AuditStatus.REJECTED,
        confidence=0.10,
    )
    plan = database_agent.plan_curation(report)
    assert not plan.can_commit
    assert "rejected" in plan.rejection_reason.lower()
    assert len(plan.generated_sql) == 0


def test_gatekeeping_delegation_audit(database_agent: DatabaseAgent):
    """Verifies that audits needing delegation (outreach/repo) are blocked."""
    report = make_dummy_consensus_report(
        model_id="NeedsOutreachModel",
        status=AuditStatus.NEEDS_OUTREACH,
        confidence=0.50,
    )
    plan = database_agent.plan_curation(report)
    assert not plan.can_commit
    assert "requires delegation" in plan.rejection_reason


# ----------------------------------------------------------------------
# 2. Entity Resolution & Model Ingestion Tests
# ----------------------------------------------------------------------
def test_new_model_insertion_plan(database_agent: DatabaseAgent):
    """Verifies that a novel model generates insertion queries for models and all 5 wide tables."""
    report = make_dummy_consensus_report(model_id="NovelMedNet-2026")
    meta = {
        "paper_title": "NovelMedNet: Ultra 3D Segmentation",
        "authors_affiliations": "Politecnico di Milano",
        "code_repository_url": "https://github.com/polimi/novelmednet",
        "architecture": "Vision-Mamba",
        "visual_backbone": "Mamba-B",
        "millions_of_parameters": 45.2,
    }
    plan = database_agent.plan_curation(report, paper_metadata=meta)

    assert plan.can_commit
    assert plan.model_action == CurationActionType.INSERT_NEW_MODEL

    # Should contain INSERT for models and 5 wide tables
    sql_text = "\n".join(plan.generated_sql)
    assert 'INSERT OR IGNORE INTO "models"' in sql_text
    assert 'INSERT OR IGNORE INTO "results_primary"' in sql_text
    assert 'INSERT OR IGNORE INTO "results_best"' in sql_text
    assert 'INSERT OR IGNORE INTO "results_best_source"' in sql_text
    assert 'INSERT OR IGNORE INTO "results_primary_comments"' in sql_text
    assert 'INSERT OR IGNORE INTO "results_best_comments"' in sql_text
    assert "NovelMedNet-2026" in sql_text


def test_existing_model_metadata_update(database_agent: DatabaseAgent):
    """Verifies that an existing model updates metadata without duplicate table initializations."""
    # MedSAM already exists in survey database
    report = make_dummy_consensus_report(model_id="MedSAM")
    meta = {
        "code_repository_url": "https://github.com/bowang-lab/MedSAM-Updated",
        "millions_of_parameters": 93.7,
    }
    plan = database_agent.plan_curation(report, paper_metadata=meta)

    assert plan.can_commit
    assert plan.model_action == CurationActionType.UPDATE_MODEL_METADATA


# ----------------------------------------------------------------------
# 3. Epistemic Boundary: Private vs Public Datasets
# ----------------------------------------------------------------------
def test_private_cohort_does_not_create_columns(database_agent: DatabaseAgent):
    """Verifies that private in-house hospital cohorts NEVER trigger schema expansion."""
    private_score = ConsensusScore(
        dataset_name="San Raffaele Internal Hospital Cohort 50 scans",
        canonical_dataset_name="San Raffaele Internal Hospital Cohort 50 scans",
        status=ConsensusStatus.CONCORDANT,
        score_a=0.88,
        score_b=0.88,
        resolved_score=0.88,
        delta=0.0000,
        notes="Evaluated on 50 in-house private clinical cases",
    )
    report = make_dummy_consensus_report(
        model_id="PrivateCohortTester",
        scores=[private_score],
    )
    plan = database_agent.plan_curation(report)

    # Must NOT have schema changes
    assert len(plan.schema_changes) == 0
    sql_text = "\n".join(plan.generated_sql)
    assert "ADD COLUMN" not in sql_text
    # Should append to comments
    assert "commentOnPaper" in sql_text


def test_novel_public_benchmark_schema_evolution(database_agent: DatabaseAgent):
    """Verifies that a valid novel public benchmark triggers wide-table schema evolution."""
    novel_public_score = ConsensusScore(
        dataset_name="AbdomenAtlas-2026",
        canonical_dataset_name="AbdomenAtlas-2026",
        organ_or_task="Abdominal Organs",
        status=ConsensusStatus.CONCORDANT,
        score_a=0.895,
        score_b=0.895,
        resolved_score=0.895,
        delta=0.0000,
        notes="Public 3D Abdomen Atlas challenge benchmark",
    )
    report = make_dummy_consensus_report(
        model_id="AtlasTester",
        scores=[novel_public_score],
    )
    plan = database_agent.plan_curation(report)

    assert len(plan.schema_changes) == 1
    sc = plan.schema_changes[0]
    assert sc.dataset_name == "AbdomenAtlas-2026"
    assert len(sc.columns_added) == 5

    sql_text = "\n".join(plan.generated_sql)
    assert 'ALTER TABLE "results_primary" ADD COLUMN "AbdomenAtlas-2026" REAL' in sql_text
    assert 'ALTER TABLE "results_best" ADD COLUMN "AbdomenAtlas-2026" REAL' in sql_text
    assert 'INSERT OR IGNORE INTO datasets' in sql_text


# ----------------------------------------------------------------------
# 4. Score Curation & Consensus Policies
# ----------------------------------------------------------------------
def test_concordant_score_insertion(database_agent: DatabaseAgent):
    """Verifies that CONCORDANT scores are inserted cleanly into results_best."""
    score = ConsensusScore(
        dataset_name="BraTS",
        canonical_dataset_name="BraTS",
        status=ConsensusStatus.CONCORDANT,
        score_a=0.915,
        score_b=0.915,
        resolved_score=0.915,
        delta=0.0000,
        prompt_type="One-prompt segmentation",
        table_ref_a="Table 1",
    )
    report = make_dummy_consensus_report(model_id="BraTSTester", scores=[score])
    plan = database_agent.plan_curation(report)

    sql_text = "\n".join(plan.generated_sql)
    assert 'UPDATE "results_best" SET "BraTS" = 91.5' in sql_text
    assert 'UPDATE "results_best_comments" SET "BraTS" =' in sql_text
    assert "[Concordant Dual-Agent: Δ=0.0]" in sql_text


def test_disputed_score_blocked_from_benchmark_column(database_agent: DatabaseAgent):
    """Verifies that DISPUTED scores leave numeric column NULL and document dispute in comments."""
    score = ConsensusScore(
        dataset_name="BTCV",
        canonical_dataset_name="BTCV",
        status=ConsensusStatus.DISPUTED,
        score_a=0.820,
        score_b=0.865,
        resolved_score=None,
        delta=0.045,
        prompt_type="Interactive",
        table_ref_a="Table 2",
    )
    report = make_dummy_consensus_report(model_id="DisputeTester", scores=[score])
    plan = database_agent.plan_curation(report)

    sql_text = "\n".join(plan.generated_sql)
    # Must NOT update results_best numeric column
    assert 'UPDATE "results_best" SET "BTCV"' not in sql_text
    # Must document in comments
    assert 'UPDATE "results_best_comments" SET "BTCV" =' in sql_text
    assert "[DISPUTE BLOCKED]" in sql_text


def test_single_agent_unconfirmed_confidence_gating(database_agent: DatabaseAgent):
    """Verifies that unconfirmed single-agent scores require high confidence (>=0.85)."""
    score = ConsensusScore(
        dataset_name="AMOS",
        canonical_dataset_name="AMOS",
        status=ConsensusStatus.SINGLE_AGENT_UNCONFIRMED,
        score_a=0.88,
        score_b=None,
        resolved_score=0.88,
        notes="Extracted only by Agent A",
    )

    # Scenario A: Low confidence (0.75 < 0.85) -> Omitted from numeric column
    report_low = make_dummy_consensus_report(model_id="UnconfirmedLow", confidence=0.75, scores=[score])
    plan_low = database_agent.plan_curation(report_low)
    sql_low = "\n".join(plan_low.generated_sql)
    assert 'UPDATE "results_best" SET "AMOS"' not in sql_low
    assert "[UNCONFIRMED OMITTED" in sql_low

    # Scenario B: High confidence (0.92 >= 0.85) -> Admitted with single-agent attribution
    report_high = make_dummy_consensus_report(model_id="UnconfirmedHigh", confidence=0.92, scores=[score])
    plan_high = database_agent.plan_curation(report_high)
    sql_high = "\n".join(plan_high.generated_sql)
    assert 'UPDATE "results_best" SET "AMOS" = 88.0' in sql_high
    assert "[Unconfirmed Single-Agent: Conf 0.92]" in sql_high


# ----------------------------------------------------------------------
# 5. Monotonic Peak Improvement Policy Tests
# ----------------------------------------------------------------------
def test_monotonic_peak_improvement_rule(database_agent: DatabaseAgent, db_manager: DatabaseManager):
    """
    Verifies that:
    1. If new score > existing peak -> results_best is updated.
    2. If new score <= existing peak -> results_best is preserved.
    """
    # Let's inspect an existing model in database: e.g. 'EMedSAM' on 'BraTS' has 89.3
    existing_score = db_manager.get_score_raw("EMedSAM", "BraTS", "results_best")
    assert existing_score == 89.3

    # Scenario 1: New score is 92.5% (> 89.3%) -> Must update!
    score_higher = ConsensusScore(
        dataset_name="BraTS",
        canonical_dataset_name="BraTS",
        status=ConsensusStatus.CONCORDANT,
        score_a=0.925,
        score_b=0.925,
        resolved_score=0.925,
        delta=0.0000,
    )
    report_higher = make_dummy_consensus_report(model_id="EMedSAM", scores=[score_higher])
    plan_higher = database_agent.plan_curation(report_higher)
    sql_higher = "\n".join(plan_higher.generated_sql)
    assert 'UPDATE "results_best" SET "BraTS" = 92.5' in sql_higher
    assert "Peak updated from 89.30%" in sql_higher

    # Scenario 2: New score is 85.0% (<= 89.3%) -> Must PRESERVE existing peak!
    score_lower = ConsensusScore(
        dataset_name="BraTS",
        canonical_dataset_name="BraTS",
        status=ConsensusStatus.CONCORDANT,
        score_a=0.850,
        score_b=0.850,
        resolved_score=0.850,
        delta=0.0000,
    )
    report_lower = make_dummy_consensus_report(model_id="EMedSAM", scores=[score_lower])
    plan_lower = database_agent.plan_curation(report_lower)
    sql_lower = "\n".join(plan_lower.generated_sql)
    assert 'UPDATE "results_best" SET "BraTS" = 85' not in sql_lower
    assert "Alternative evaluation: 85.0%" in sql_lower


# ----------------------------------------------------------------------
# 6. Atomic Execution & Dry-Run Integration
# ----------------------------------------------------------------------
def test_dry_run_vs_apply_atomic_commit(database_agent: DatabaseAgent, db_manager: DatabaseManager):
    """Verifies that dry_run=True modifies zero rows, while dry_run=False commits cleanly."""
    score = ConsensusScore(
        dataset_name="BraTS",
        canonical_dataset_name="BraTS",
        status=ConsensusStatus.CONCORDANT,
        score_a=0.940,
        score_b=0.940,
        resolved_score=0.940,
        delta=0.0000,
    )
    report = make_dummy_consensus_report(model_id="AtomicTester-3D", scores=[score])

    # 1. Dry run
    res_dry = database_agent.curate(report, dry_run=True)
    assert not res_dry.committed
    assert db_manager.get_model("AtomicTester-3D") is None

    # 2. Apply commit
    res_apply = database_agent.curate(report, dry_run=False)
    assert res_apply.committed
    assert db_manager.get_model("AtomicTester-3D") is not None
    assert db_manager.get_score_raw("AtomicTester-3D", "BraTS", "results_best") == 94.0


def test_single_agent_audit_report_curation(database_agent: DatabaseAgent, db_manager: DatabaseManager):
    """Verifies that single-agent AuditReport with validated_scores is successfully curated."""
    from agent_gmmis.extraction.schemas import RawExtractedScore

    score = RawExtractedScore(
        dataset_name="BraTS",
        organ_or_task="Overall",
        dice_score=0.910,
        prompt_type="1 point",
    )
    report = AuditReport(
        model_id="SingleAgentTester-3D",
        status=AuditStatus.VERIFIED,
        passed_sanity_checks=True,
        passed_inclusion_screening=True,
        confidence_score=0.92,
        summary="Certified single-agent extraction",
        validated_scores=[score],
    )

    plan = database_agent.plan_curation(report)
    assert plan.model_id == "SingleAgentTester-3D"
    assert len(plan.score_diffs) == 1
    assert plan.score_diffs[0].dataset_name == "BraTS"
    assert plan.score_diffs[0].new_score == 91.0
    assert plan.score_diffs[0].consensus_status == "single_agent_unconfirmed"
    assert plan.can_commit is True

    # Commit and verify in SQLite
    res = database_agent.curate(report, dry_run=False)
    assert res.committed is True
    assert db_manager.get_model("SingleAgentTester-3D") is not None
    assert db_manager.get_score_raw("SingleAgentTester-3D", "BraTS", "results_best") == 91.0


# ----------------------------------------------------------------------
# 7. Priorità P2: Schema Contamination Guards & Entity Resolution
# ----------------------------------------------------------------------
def test_schema_guard_rejects_organs_modalities_and_placeholders(database_agent: DatabaseAgent):
    """Verifies that anatomical organs, imaging modalities, and placeholders never trigger ADD COLUMN."""
    # Direct validation tests
    assert not database_agent._is_valid_new_benchmark_name("Liver")
    assert not database_agent._is_valid_new_benchmark_name("Spleen")
    assert not database_agent._is_valid_new_benchmark_name("Left Kidney")
    assert not database_agent._is_valid_new_benchmark_name("Aorta")
    assert not database_agent._is_valid_new_benchmark_name("CT")
    assert not database_agent._is_valid_new_benchmark_name("MRI")
    assert not database_agent._is_valid_new_benchmark_name("FLAIR")
    assert not database_agent._is_valid_new_benchmark_name("Dataset 1")
    assert not database_agent._is_valid_new_benchmark_name("Task 2")
    assert not database_agent._is_valid_new_benchmark_name("Test Set")
    assert not database_agent._is_valid_new_benchmark_name("Ours")
    assert database_agent._is_valid_new_benchmark_name("AbdomenAtlas-2026")
    assert database_agent._is_valid_new_benchmark_name("MyoPS-2020")

    # In curation plan: non-benchmark items should be diverted to comments, not schema changes
    organ_score = ConsensusScore(
        dataset_name="Liver",
        canonical_dataset_name="Liver",
        organ_or_task="Liver",
        status=ConsensusStatus.CONCORDANT,
        score_a=0.92,
        score_b=0.92,
        resolved_score=0.92,
    )
    placeholder_score = ConsensusScore(
        dataset_name="Dataset 1",
        canonical_dataset_name="Dataset 1",
        organ_or_task="Overall",
        status=ConsensusStatus.CONCORDANT,
        score_a=0.85,
        score_b=0.85,
        resolved_score=0.85,
    )
    report = make_dummy_consensus_report(
        model_id="GuardTesterModel",
        scores=[organ_score, placeholder_score],
    )
    plan = database_agent.plan_curation(report)
    assert len(plan.schema_changes) == 0
    sql_text = "\n".join(plan.generated_sql)
    assert "ADD COLUMN" not in sql_text
    assert "Schema Guard" in sql_text


def test_schema_guard_blocks_unconfirmed_single_agent_new_columns(database_agent: DatabaseAgent):
    """Verifies that single-agent unconfirmed scores cannot trigger wide-table ADD COLUMN."""
    unconfirmed_new_score = ConsensusScore(
        dataset_name="BrandNewChallenge2026",
        canonical_dataset_name="BrandNewChallenge2026",
        organ_or_task="Overall",
        status=ConsensusStatus.SINGLE_AGENT_UNCONFIRMED,
        score_a=0.89,
        score_b=None,
        resolved_score=0.89,
    )
    report = make_dummy_consensus_report(
        model_id="SingleAgentNovelTester",
        scores=[unconfirmed_new_score],
    )
    plan = database_agent.plan_curation(report)
    assert len(plan.schema_changes) == 0
    sql_text = "\n".join(plan.generated_sql)
    assert "ADD COLUMN" not in sql_text


def test_unicode_nfkc_and_generic_model_id_resolution(database_agent: DatabaseAgent):
    """Verifies Unicode NFKC ligature normalization and generic placeholder filtering."""
    # 1. 'Diﬀ-UNet' has Unicode ligature U+FB00 ('ﬀ') -> Must resolve to existing 'Diff-UNet' in DB
    report_ligature = make_dummy_consensus_report(model_id="Diﬀ-UNet")
    plan_ligature = database_agent.plan_curation(report_ligature)
    assert plan_ligature.model_id == "Diff-UNet"
    assert plan_ligature.model_action == CurationActionType.UPDATE_MODEL_METADATA

    # 2. 'Ours' must be discarded in favor of expected_model_id or paper title
    report_generic = make_dummy_consensus_report(model_id="Ours")
    meta = {
        "expected_model_id": "Cheap Lunch SAM",
        "proposed_model_id": "Ours",
        "paper_title": "A Cheap Lunch for Medical Image Segmentation",
    }
    plan_generic = database_agent.plan_curation(report_generic, paper_metadata=meta)
    assert plan_generic.model_id != "Ours"
    assert plan_generic.model_id == "Cheap Lunch SAM"


def test_pre_update_deduplication_single_update_per_cell(database_agent: DatabaseAgent):
    """Verifies that multiple scores for the same dataset generate at most one UPDATE query with Overall peak."""
    organ1 = ConsensusScore(
        dataset_name="BTCV",
        canonical_dataset_name="BTCV",
        organ_or_task="Liver",
        status=ConsensusStatus.CONCORDANT,
        score_a=0.91,
        score_b=0.91,
        resolved_score=0.91,
    )
    organ2 = ConsensusScore(
        dataset_name="BTCV",
        canonical_dataset_name="BTCV",
        organ_or_task="Spleen",
        status=ConsensusStatus.CONCORDANT,
        score_a=0.93,
        score_b=0.93,
        resolved_score=0.93,
    )
    overall = ConsensusScore(
        dataset_name="BTCV",
        canonical_dataset_name="BTCV",
        organ_or_task="Overall",
        status=ConsensusStatus.CONCORDANT,
        score_a=0.88,
        score_b=0.88,
        resolved_score=0.88,
    )
    report = make_dummy_consensus_report(
        model_id="BTCVDedupTester",
        scores=[organ1, organ2, overall],
    )
    plan = database_agent.plan_curation(report)
    sql_text = "\n".join(plan.generated_sql)

    # Exactly one UPDATE to results_best for BTCV
    best_updates = [line for line in sql_text.splitlines() if 'UPDATE "results_best" SET "BTCV"' in line]
    assert len(best_updates) == 1
    # Overall score (88.0%) should be preferred over organ scores
    assert "88.0" in best_updates[0]

