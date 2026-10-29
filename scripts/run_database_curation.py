#!/usr/bin/env python3
"""
CLI Runner for DatabaseAgent: Ingests audited benchmark runs and generates SQL.

Usage:
  python scripts/run_database_curation.py --dry-run
  python scripts/run_database_curation.py --run-dir logs/mufasa_runs/20261023_190928_mufasa_dual_local_40 --sql-out logs/curation_40.sql
  python scripts/run_database_curation.py --paper Medical_SAM_2_MedSAM_2 --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_gmmis.auditing.schemas import AuditReport, ConsensusAuditReport
from agent_gmmis.config import DEFAULT_DB_PATH
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.database.database_agent import DatabaseAgent
from agent_gmmis.database.schemas import CurationActionType, CurationPlan, CurationResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DatabaseAgent curation over batch benchmark extractions.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="logs/mufasa_runs/20261023_190928_mufasa_dual_local_40",
        help="Path to benchmark run directory containing an 'extractions' subdirectory.",
    )
    parser.add_argument(
        "--paper",
        type=str,
        default=None,
        help="Filter for a single paper folder (e.g. 'Medical_SAM_2_MedSAM_2').",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="Target SQLite database file path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Simulate curation without modifying the database (default: True).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Execute transactions and commit mutations to the SQLite database.",
    )
    parser.add_argument(
        "--sql-out",
        type=str,
        default="logs/curation_generated.sql",
        help="Path where all generated SQL statements will be exported.",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default="logs/curation_report.md",
        help="Path where the Markdown curation report will be saved.",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.85,
        help="Minimum confidence threshold for single-agent unconfirmed scores.",
    )
    return parser.parse_args()


def load_audit_report(report_path: Path) -> Optional[ConsensusAuditReport]:
    """Loads and deserializes an audit report JSON file."""
    if not report_path.exists():
        return None
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        if "consensus_scores" in data:
            return ConsensusAuditReport.model_validate(data)
        return AuditReport.model_validate(data)
    except Exception as e:
        print(f"Warning: Failed to parse audit report at {report_path}: {e}")
        return None


def load_paper_metadata(folder: Path) -> Dict[str, Any]:
    """Extracts paper metadata from agent extraction artifacts or summary."""
    meta = {"expected_model_id": folder.name}
    for candidate in ("agent_a_extracted.json", "agent_b_extracted.json", "final_reconciled.json"):
        p = folder / candidate
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for k in (
                    "paper_title",
                    "proposed_model_id",
                    "proposed_model_full_title",
                    "authors_affiliations",
                    "code_repository_url",
                    "framework",
                    "architecture",
                    "visual_backbone",
                    "millions_of_parameters",
                    "number_of_gflops",
                    "resources",
                    "resources_total_vram_gb",
                ):
                    if k in data and data[k] is not None:
                        meta[k] = data[k]
                if meta.get("paper_title"):
                    break
            except Exception:
                pass

    summary_file = folder / "paper_summary.json"
    if summary_file.exists() and "paper_title" not in meta:
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            if "title" in data:
                meta["paper_title"] = data["title"]
        except Exception:
            pass

    return meta


def main() -> None:
    args = parse_args()
    dry_run = not args.apply

    run_dir = Path(args.run_dir)
    extractions_dir = run_dir / "extractions" if (run_dir / "extractions").exists() else run_dir
    if not extractions_dir.exists():
        print(f"Error: Directory not found: {extractions_dir}")
        sys.exit(1)

    print("=" * 80)
    print("   GMMIS Autonomous Database Manager Agent (DatabaseAgent)")
    print("=" * 80)
    print(f"Target Database   : {args.db_path}")
    print(f"Input Extractions : {extractions_dir}")
    print(f"Execution Mode    : {'DRY RUN (Preview Only)' if dry_run else 'APPLY (Committing Mutations)'}")
    print(f"Single-Agent Conf : >= {args.min_conf:.2f}")
    print(f"SQL Export Output : {args.sql_out}")
    print(f"Report Output     : {args.report_out}")
    print("=" * 80)

    db_manager = DatabaseManager(args.db_path)
    agent = DatabaseAgent(db_manager, single_agent_min_conf=args.min_conf)

    paper_folders = sorted([f for f in extractions_dir.iterdir() if f.is_dir()])
    if args.paper:
        paper_folders = [f for f in paper_folders if args.paper.lower() in f.name.lower()]

    if not paper_folders:
        print(f"No paper directories found in {extractions_dir}")
        sys.exit(0)

    all_plans: List[CurationPlan] = []
    all_results: List[CurationResult] = []
    total_sql_statements: List[str] = []

    print(f"\nProcessing {len(paper_folders)} papers...\n")
    print(f"{'#':<3} | {'Model ID':<26} | {'Status':<16} | {'Conf':<6} | {'Action':<15} | {'Scores':<8} | {'SQLs':<5}")
    print("-" * 92)

    for idx, folder in enumerate(paper_folders, start=1):
        report_file = folder / "audit_report.json"
        if not report_file.exists():
            continue

        audit_report = load_audit_report(report_file)
        if not audit_report:
            continue

        meta = load_paper_metadata(folder)
        result = agent.curate(audit_report, paper_metadata=meta, dry_run=dry_run)
        plan = result.plan

        all_plans.append(plan)
        all_results.append(result)

        if plan.generated_sql:
            header = f"\n-- ====================================================================\n"
            header += f"-- Model: {plan.model_id} | Status: {plan.audit_status} (Conf: {plan.audit_confidence:.2f})\n"
            header += f"-- Action: {plan.model_action.value} | Folder: {folder.name}\n"
            header += f"-- ====================================================================\n"
            total_sql_statements.append(header)
            total_sql_statements.extend(plan.generated_sql)

        # Status formatting
        status_disp = plan.audit_status
        action_disp = plan.model_action.value
        if not plan.can_commit:
            action_disp = "BLOCKED"

        scores_summary = f"{len(plan.score_diffs)}"
        print(
            f"{idx:<3} | {plan.model_id[:26]:<26} | {status_disp[:16]:<16} | "
            f"{plan.audit_confidence * 100:.0f}%   | {action_disp[:15]:<15} | "
            f"{scores_summary:<8} | {len(plan.generated_sql):<5}"
        )

    # Export SQL file
    sql_out_path = Path(args.sql_out)
    sql_out_path.parent.mkdir(parents=True, exist_ok=True)
    full_sql_text = (
        "-- ====================================================================\n"
        "-- GMMIS Autonomous Database Curation Script\n"
        f"-- Generated by DatabaseAgent on {len(all_plans)} papers\n"
        f"-- Execution Mode: {'DRY RUN' if dry_run else 'APPLIED'}\n"
        "-- ====================================================================\n\n"
        "BEGIN TRANSACTION;\n\n"
        + "\n\n".join(total_sql_statements)
        + "\n\nCOMMIT;\n"
    )
    sql_out_path.write_text(full_sql_text, encoding="utf-8")

    # Generate Markdown Report
    report_out_path = Path(args.report_out)
    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = generate_markdown_report(all_plans, dry_run, args.sql_out)
    report_out_path.write_text(report_text, encoding="utf-8")

    print("-" * 92)
    print(f"\nProcessing Complete!")
    print(f"Total Papers Evaluated   : {len(all_plans)}")
    print(f"Total SQL Queries Built  : {len(total_sql_statements)}")
    print(f"SQL Script Exported to   : {sql_out_path}")
    print(f"Markdown Report Saved to : {report_out_path}")


def generate_markdown_report(plans: List[CurationPlan], dry_run: bool, sql_file_path: str) -> str:
    """Generates an executive markdown report of all curation decisions."""
    total_papers = len(plans)
    insert_models = sum(1 for p in plans if p.model_action == CurationActionType.INSERT_NEW_MODEL and p.can_commit)
    update_models = sum(1 for p in plans if p.model_action == CurationActionType.UPDATE_MODEL_METADATA and p.can_commit)
    blocked_models = sum(1 for p in plans if not p.can_commit)

    total_schema_changes = sum(len(p.schema_changes) for p in plans)
    novel_benchmarks = list(dict.fromkeys(c.dataset_name for p in plans for c in p.schema_changes))

    inserted_scores = sum(sum(1 for d in p.score_diffs if d.action == CurationActionType.INSERT_SCORE) for p in plans)
    updated_scores = sum(sum(1 for d in p.score_diffs if d.action == CurationActionType.UPDATE_PEAK_SCORE) for p in plans)
    preserved_scores = sum(sum(1 for d in p.score_diffs if d.action == CurationActionType.PRESERVE_EXISTING_PEAK) for p in plans)
    disputed_scores = sum(sum(1 for d in p.score_diffs if d.action == CurationActionType.SKIP_DISPUTED_SCORE) for p in plans)
    unconfirmed_scores = sum(sum(1 for d in p.score_diffs if d.action == CurationActionType.SKIP_UNCONFIRMED_SCORE) for p in plans)

    lines = [
        "# DatabaseAgent Curation Executive Report",
        "",
        f"> **Generated Mode**: {'DRY RUN (Preview Only)' if dry_run else 'APPLIED (Committed)'}  ",
        f"> **Total Papers Evaluated**: {total_papers}  ",
        f"> **SQL Script Export**: [`{sql_file_path}`]({sql_file_path})  ",
        "",
        "## 1. High-Level Curation Summary",
        "",
        "| Metric | Count | Description |",
        "| :--- | :--- | :--- |",
        f"| **New Models Inserted** | **{insert_models}** | Unindexed architectures added to `models` and initialized in wide tables |",
        f"| **Models Updated** | **{update_models}** | Existing models enriched with new publication or code metadata |",
        f"| **Papers Blocked** | **{blocked_models}** | Rejected or held in staging due to audit status |",
        f"| **Novel Benchmarks Added** | **{len(novel_benchmarks)}** | Public 3D datasets triggering wide-table schema evolution (`{', '.join(novel_benchmarks) or 'None'}`) |",
        f"| **New Benchmark Scores** | **{inserted_scores}** | Newly established baseline or peak evaluations |",
        f"| **Peak Scores Improved** | **{updated_scores}** | Monotonic improvement over existing historical best scores |",
        f"| **Peaks Preserved** | **{preserved_scores}** | Existing historical peak was higher than newly extracted score |",
        f"| **Disputes Blocked** | **{disputed_scores}** | Numeric column kept NULL to guard leaderboard purity |",
        f"| **Unconfirmed Omitted** | **{unconfirmed_scores}** | Single-agent scores below confidence threshold omitted |",
        "",
        "## 2. Paper-by-Paper Curation Log",
        "",
        "| # | Model ID | Audit Status | Conf | Action | Novel Benchmarks | Peak Scores Inserted/Updated | Disputes Blocked |",
        "| :- | :--- | :--- | :- | :--- | :--- | :- | :- |",
    ]

    for idx, p in enumerate(plans, start=1):
        novel_b = ", ".join(c.dataset_name for c in p.schema_changes) or "-"
        peak_cnt = sum(1 for d in p.score_diffs if d.action in (CurationActionType.INSERT_SCORE, CurationActionType.UPDATE_PEAK_SCORE))
        disp_cnt = sum(1 for d in p.score_diffs if d.action == CurationActionType.SKIP_DISPUTED_SCORE)
        act = p.model_action.value if p.can_commit else f"BLOCKED ({p.rejection_reason})"

        lines.append(
            f"| {idx} | **{p.model_id}** | `{p.audit_status}` | {p.audit_confidence * 100:.0f}% | "
            f"`{act}` | {novel_b} | {peak_cnt} | {disp_cnt} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
