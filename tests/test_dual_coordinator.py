"""Unit tests for DualExtractionCoordinator matching, inverted organ deliberation, and thesis tables."""

import pytest

from agent_gmmis.auditing.auditor import AuditorAgent
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.extraction.alias_resolver import DatasetAliasResolver
from agent_gmmis.extraction.dual_coordinator import (
    DualExtractionCoordinator,
    _find_matching_score_index,
)
from agent_gmmis.extraction.extractor import ExtractionAgent
from agent_gmmis.extraction.pdf_parser import ParsedManuscript
from agent_gmmis.extraction.schemas import (
    ExtractedPaperData,
    InclusionCriteriaCheck,
    RawExtractedScore,
    ReinterrogationOrganBreakdownResponse,
)
from agent_gmmis.llm.client import LLMClient, LLMConfig


def test_find_matching_score_index_canonical_organ_and_budget():
    resolver = DatasetAliasResolver()
    scores = [
        RawExtractedScore(
            dataset_name="BraTS21",
            organ_or_task="Whole Tumor (WT)",
            data_budget_or_setting="100% labeled",
            dice_score=0.92,
        ),
        RawExtractedScore(
            dataset_name="BraTS21",
            organ_or_task="Whole Tumor (WT)",
            data_budget_or_setting="25% labeled",
            dice_score=0.81,
        ),
    ]
    # "WT" with budget=None (standard 100%) should match index 0 ("100% labeled" -> None), not index 1 ("25%")
    idx_full = _find_matching_score_index(
        scores,
        dataset_name="BraTS",
        organ_or_task="WT",
        data_budget_or_setting=None,
        alias_resolver=resolver,
    )
    assert idx_full == 0

    idx_25 = _find_matching_score_index(
        scores,
        dataset_name="BraTS",
        organ_or_task="WT",
        data_budget_or_setting="25% training data",
        alias_resolver=resolver,
    )
    assert idx_25 == 1


def test_inverted_deliberation_respects_only_macro_average_reported():
    """If peer confirms only_macro_average_reported=True, deliberation exits cleanly without hallucinating organs."""
    auditor = AuditorAgent(db_manager=DatabaseManager())
    screening = InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True)

    data_a = ExtractedPaperData(
        paper_title="DualModel",
        proposed_model_id="DualModel",
        screening=screening,
        extracted_scores=[
            RawExtractedScore(dataset_name="BTCV", organ_or_task="Liver", dice_score=0.93),
            RawExtractedScore(dataset_name="BTCV", organ_or_task="Spleen", dice_score=0.90),
        ],
    )
    data_b = ExtractedPaperData(
        paper_title="DualModel",
        proposed_model_id="DualModel",
        screening=screening,
        extracted_scores=[
            RawExtractedScore(dataset_name="BTCV", organ_or_task="Overall", dice_score=0.915),
        ],
    )

    agent_a = ExtractionAgent(llm_client=LLMClient(config=LLMConfig(provider="mock")))
    agent_b = ExtractionAgent(llm_client=LLMClient(config=LLMConfig(provider="mock")))
    agent_a.process_manuscript = lambda m: data_a
    agent_b.process_manuscript = lambda m: data_b
    agent_b.reinterrogate_organ_breakdown = lambda m, ds, target_organs, mod: ReinterrogationOrganBreakdownResponse(
        dataset_name=ds,
        organ_scores=[],
        only_macro_average_reported=True,
        rationale="Paper reports only macro-average in Table 1.",
    )

    coordinator = DualExtractionCoordinator(agent_a=agent_a, agent_b=agent_b)
    manuscript = ParsedManuscript(file_path="test.pdf", title="DualModel", total_pages=1, pages_text={1: "test"})
    ext_a, ext_b, report, trajectory = coordinator.extract_manuscript_with_deliberation(
        manuscript=manuscript,
        auditor=auditor,
        max_rounds=2,
        verbose=False,
    )
    assert len(ext_b.extracted_scores) == 1
    assert report.dispute_count == 0


def test_generate_thesis_tables_markdown_and_latex(tmp_path):
    """Smoke-tests scripts/generate_thesis_tables.py across T1-T5 in both Markdown and LaTeX."""
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "generate_thesis_tables.py"
    spec = importlib.util.spec_from_file_location("generate_thesis_tables", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    build_latex_tables = mod.build_latex_tables
    build_markdown_tables = mod.build_markdown_tables

    sample_run = {
        "metadata": {
            "model_a": "qwen2.5-14b-awq",
            "model_b": "deepseek-r1-32b-awq",
            "is_dual_mode": True,
            "total_papers": 1,
        },
        "global_metrics": {
            "total_matched_benchmarks": 3,
            "total_gt_benchmarks_active": 4,
            "overall_recall": 0.75,
            "micro_mae_active": 0.012,
            "micro_mae_primary": 0.012,
            "micro_mae_best": 0.018,
            "micro_mae_regime_agnostic": 0.008,
            "coverage_adjusted_score": 0.741,
            "iaa_metrics": {
                "micro_paired_iaa": 0.90,
                "macro_paired_iaa": 0.90,
                "micro_strict_iaa": 0.85,
                "micro_exact_iaa": 0.70,
                "micro_inter_agent_mae": 0.006,
                "macro_inter_agent_mae": 0.006,
                "concordant_items_total": 9,
                "disputed_items_total": 1,
                "singleton_items_total": 1,
                "round1_macro_iaa": 0.75,
                "round1_macro_inter_mae": 0.014,
                "deliberation_iaa_delta_pp": 15.0,
                "deliberation_resolved_total": 2,
                "gt_mae_when_concordant": 0.005,
                "gt_mae_when_concordant_n": 2,
                "gt_mae_when_non_concordant": 0.025,
                "gt_mae_when_non_concordant_n": 1,
            },
        },
        "papers": [
            {
                "model_id": "SAM-Med2D",
                "audit": {"status": "verified"},
                "evaluation": {"dice_mae": 0.012},
                "consensus": {
                    "paired_count": 10,
                    "concordant_count": 9,
                    "dispute_count": 1,
                    "unconfirmed_count": 1,
                    "paired_concordance_rate": 0.90,
                    "strict_concordance_rate": 0.8182,
                    "inter_agent_mae": 0.006,
                },
            }
        ],
    }

    md = build_markdown_tables([sample_run])
    tex = build_latex_tables([sample_run])
    assert "Table T1" in md and "Table T5" in md and "90.0%" in md
    assert r"\toprule" in tex and r"\bottomrule" in tex and "SAM-Med2D" in tex
