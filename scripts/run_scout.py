#!/usr/bin/env python3
"""
Standalone CLI Runner for the Autonomous ScoutAgent (Step 4).
Supports:
  1. Living Survey Discovery (`--since 2025-06-01`)
  2. Retrospective Time-Machine Backtesting (`--backtest-2025`)
  3. Custom Agentic Search Objectives (`--objective "..."`)
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_gmmis.config import DEFAULT_DB_PATH
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.llm.client import LLMClient, LLMConfig
from agent_gmmis.scouting.deduplicator import normalize_title_key, title_token_similarity
from agent_gmmis.scouting.harvesters import _extract_arxiv_id, _normalize_date
from agent_gmmis.scouting.scout_agent import ScoutAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone ScoutAgent literature discovery & screening.")
    parser.add_argument("--since", default="2025-06-01", help="Start publication date (YYYY-MM-DD)")
    parser.add_argument("--until", default=None, help="End publication date (YYYY-MM-DD)")
    parser.add_argument("--backtest-2025", action="store_true", help="Run retrospective backtest on Jan–Jun 2025 survey models")
    parser.add_argument("--ignore-db-after", default=None, help="Mask DB models published on/after this date (YYYY-MM-DD) for custom window backtests")
    parser.add_argument("--provider", default="gemini", help="LLM provider (gemini, ollama, mufasa, mock)")
    parser.add_argument("--model", default="gemini-3.1-flash-lite", help="LLM model identifier")
    parser.add_argument("--base-url", default=None, help="Optional custom OpenAI-compatible base URL")
    parser.add_argument("--objective", default=None, help="Optional custom natural-language research goal for Agentic Query Planner")
    parser.add_argument("--max-per-query", type=int, default=25, help="Max results fetched per API query")
    parser.add_argument("--max-llm-evals", type=int, default=60, help="Max candidates evaluated by Stage 3 LLM triage")
    parser.add_argument("--no-agentic-queries", action="store_true", help="Disable LLM dynamic query planning (use only PRISMA queries)")
    parser.add_argument("--no-snowball", action="store_true", help="Disable OpenAlex citation snowballing")
    parser.add_argument("--no-lead-followup", action="store_true", help="Disable Agentic Follow-the-Lead exploration")
    parser.add_argument("--no-cache", action="store_true", help="Bypass local disk cache for API calls")
    args = parser.parse_args()

    db = DatabaseManager(DEFAULT_DB_PATH)
    llm_cfg = LLMConfig(provider=args.provider, model=args.model, base_url=args.base_url)
    llm_client = LLMClient(llm_cfg)

    if args.backtest_2025:
        start_date = "2025-01-01"
        end_date = "2025-06-30"
        ignore_after = "2025-01-01"
        print("\n⏳ [ScoutAgent Time-Machine Backtest] Masking DB models published >= 2025-01-01...")
    else:
        start_date = args.since
        end_date = args.until
        ignore_after = args.ignore_db_after
        if ignore_after:
            print(f"\n⏳ [ScoutAgent Window Backtest] Masking DB models published >= {ignore_after}...")

    scout = ScoutAgent(
        llm_client=llm_client,
        db_manager=db,
        use_cache=not args.no_cache,
        ignore_db_models_after_date=ignore_after,
    )

    print(
        f"🚀 Launching ScoutAgent campaign [{start_date} -> {end_date or 'Present'}] "
        f"using {llm_client.config.provider}/{llm_client.config.model}..."
    )

    report = scout.run_campaign(
        start_date=start_date,
        end_date=end_date,
        max_results_per_query=args.max_per_query,
        max_llm_evaluations=args.max_llm_evals,
        enable_citation_snowball=not args.no_snowball,
        enable_agentic_queries=not args.no_agentic_queries,
        enable_lead_followup=not args.no_lead_followup,
        user_objective=args.objective,
    )

    json_path, md_path = scout.save_report(report)
    cot_path = md_path.parent / "scout_cot_trace.md"
    m = report.metrics

    print("\n" + "=" * 74)
    print(f"✅ SCOUT CAMPAIGN COMPLETED ({report.run_id})")
    print("=" * 74)
    print(f"  Raw Harvested (Stage 1)       : {m.total_raw_harvested} {m.harvested_by_source}")
    print(f"  Unique After Merge (Stage 2)  : {m.unique_after_cross_source_merge}")
    print(f"  Skipped Already in DB         : {m.skipped_already_in_db}")
    print(f"  Lifecycle Upgrades Detected   : {m.detected_lifecycle_upgrades}")
    print(f"  Stage 2 Lexical Rejections    : {m.rejected_stage2_lexical}")
    print(f"  Stage 3 LLM Triage Evaluated  : {m.evaluated_by_llm_stage3}")
    print(f"  Promoted TIER_1_GENERALIST    : {m.promoted_tier1_generalist}")
    print(f"  Promoted TIER_2_BENCHMARK_SOTA: {m.promoted_tier2_benchmark_sota}")
    print(f"  Stage 4 Open-Access PDFs      : {m.pdfs_resolved_stage4}")
    print(f"  Saved Markdown Report         : {md_path}")
    print(f"  Saved Structured CoT Trace    : {cot_path}")
    print(f"  Saved JSON Report             : {json_path}")
    print("=" * 74)

    if args.backtest_2025 or args.ignore_db_after:
        s_month = (start_date or "2025-01-01")[:7]
        e_month = (end_date or "2025-12-31")[:7]
        models_in_window = [
            mod for mod in db.get_all_models()
            if s_month <= ((_normalize_date(mod.first_publication_date) or "")[:7]) <= e_month
        ]
        promoted_all = report.tier1_candidates + report.tier2_candidates
        matched_window = []
        for gt_m in models_in_window:
            gt_arxiv = _extract_arxiv_id(gt_m.first_publication_link)
            found = False
            for cand in promoted_all:
                if gt_arxiv and cand.paper.arxiv_id == gt_arxiv:
                    found = True
                    break
                if title_token_similarity(gt_m.title or "", cand.paper.title) >= 0.80:
                    found = True
                    break
                if normalize_title_key(gt_m.id) in normalize_title_key(cand.paper.title):
                    found = True
                    break
            if found:
                matched_window.append(gt_m.id)

        print(f"\n🎯 [Backtest Recall on Window Survey Models ({s_month}..{e_month})]: {len(matched_window)}/{len(models_in_window)} rediscovered")
        if models_in_window:
            print(f"   Expected Survey Models in Window: {', '.join(m.id for m in models_in_window)}")
        if matched_window:
            print(f"   Rediscovered Models             : {', '.join(matched_window)}")


if __name__ == "__main__":
    main()
