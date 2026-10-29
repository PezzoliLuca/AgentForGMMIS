"""RunLogger: Structured run directory and intermediate agent artifact logging engine.

Manages dedicated run directories for single and dual-agent evaluations,
persisting raw intermediate outputs of each agent, consensus matrices,
audit certificates, and aggregated benchmark summaries.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Union

from agent_gmmis.config import LOGS_DIR
from agent_gmmis.extraction.schemas import ExtractedPaperData


class LoggerTee:
    """Duplicates stdout/stderr to both console and a destination log file."""

    def __init__(self, filepath: Path, original_stream=None):
        self.terminal = original_stream or sys.stdout
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(filepath, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        if not self.log_file.closed:
            self.log_file.close()


def sanitize_filename(name: str) -> str:
    """Converts a model ID or paper identifier into a clean filesystem folder name."""
    # Replace any character that is not alphanumeric or dash/underscore with an underscore
    s = re.sub(r"[^\w\s-]", "_", name)
    # Collapse any sequences of whitespace, dashes, or underscores into a single underscore
    s = re.sub(r"[-\s_]+", "_", s).strip("_")
    return s or "unnamed"


class RunLogger:
    """
    Orchestrates organized logging for benchmark and extraction runs.

    Directory Layout:
    logs/runs/<run_id>/
        ├── run_manifest.json          # Run metadata, parameters, timestamp
        ├── execution.log              # Complete console execution stream
        ├── benchmark_report.md        # Formatted Markdown report
        ├── benchmark_results.json     # Consolidated machine-readable JSON
        └── extractions/               # Intermediate agent artifacts per paper
            └── <model_id>/
                ├── agent_a_extracted.json     # Intermediate extraction by Agent A
                ├── agent_b_extracted.json     # Intermediate extraction by Agent B (dual mode)
                ├── consensus_matrix.json      # Inter-agent agreement and dispute matrix
                ├── audit_report.json          # Quality assurance audit certificate
                ├── final_reconciled.json      # Final reconciled extraction
                ├── evaluation.json            # Ground truth comparison report
                ├── outreach_draft.json        # Author email draft (if drafted)
                └── paper_summary.json         # High-level index of this paper's outputs
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        run_name: Optional[str] = None,
        tag: Optional[str] = None,
        is_dual: bool = False,
        capture_stdout: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.base_dir = Path(base_dir) if base_dir else (LOGS_DIR / "runs")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.is_dual = is_dual
        self.metadata = metadata or {}

        # Build clean run directory name
        if run_name:
            self.run_id = sanitize_filename(run_name)
        else:
            tag_part = f"_{sanitize_filename(tag)}" if tag else ("_dual" if is_dual else "_single")
            self.run_id = f"{self.timestamp}{tag_part}"

        self.run_dir = self.base_dir / self.run_id
        self.extractions_dir = self.run_dir / "extractions"
        self.extractions_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.run_dir / "execution.log"
        self.results_file = self.run_dir / "benchmark_results.json"
        self.report_file = self.run_dir / "benchmark_report.md"
        self.manifest_file = self.run_dir / "run_manifest.json"

        # Console stdout tee
        self.capture_stdout = capture_stdout
        self.tee: Optional[LoggerTee] = None
        self._original_stdout = None
        if capture_stdout:
            self._original_stdout = sys.stdout
            self.tee = LoggerTee(self.log_file, original_stream=sys.stdout)
            sys.stdout = self.tee

        # Write initial manifest
        self._write_manifest()

    def _write_manifest(self) -> None:
        """Writes run_manifest.json with execution metadata."""
        manifest = {
            "run_id": self.run_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_dual": self.is_dual,
            "paths": {
                "run_dir": str(self.run_dir),
                "execution_log": str(self.log_file),
                "benchmark_results": str(self.results_file),
                "benchmark_report": str(self.report_file),
                "extractions_dir": str(self.extractions_dir),
            },
            "metadata": self.metadata,
        }
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def log_paper_extraction(
        self,
        model_id: str,
        extracted_a: ExtractedPaperData,
        extracted_b: Optional[ExtractedPaperData] = None,
        audit_report: Optional[Any] = None,
        final_extracted: Optional[ExtractedPaperData] = None,
        eval_report: Optional[Any] = None,
        repo_inspector_result: Optional[Any] = None,
        outreach_draft: Optional[Any] = None,
        curation_result: Optional[Any] = None,
        elapsed_seconds: Optional[float] = None,
        deliberation_trajectory: Optional[List[Any]] = None,
        error: Optional[str] = None,
    ) -> Path:
        """
        Persists all intermediate and final artifacts for a single scientific paper.

        Saves:
        - agent_a_extracted.json
        - agent_b_extracted.json (if present)
        - consensus_matrix.json (if ConsensusAuditReport)
        - deliberation_trajectory.json (if multi-round deliberation occurred)
        - audit_report.json
        - final_reconciled.json
        - evaluation.json (if present)
        - outreach_draft.json and outreach_draft.md (if present)
        - repo_inspection.json (if present)
        - paper_summary.json (index)

        Returns:
            Path to the paper's dedicated extraction directory.
        """
        paper_dir = self.extractions_dir / sanitize_filename(model_id)
        paper_dir.mkdir(parents=True, exist_ok=True)

        summary_data: Dict[str, Any] = {
            "model_id": model_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": elapsed_seconds,
            "has_error": error is not None,
            "error": error,
            "artifacts": {},
        }

        # 1. Agent A intermediate extraction
        if extracted_a:
            data_a = extracted_a.model_dump()
            path_a = paper_dir / "agent_a_extracted.json"
            with open(path_a, "w", encoding="utf-8") as f:
                json.dump(data_a, f, indent=2)
            summary_data["artifacts"]["agent_a"] = "agent_a_extracted.json"
            summary_data["agent_a_scores_count"] = len(extracted_a.extracted_scores)

        # 2. Agent B intermediate extraction (if dual mode)
        if extracted_b:
            data_b = extracted_b.model_dump()
            path_b = paper_dir / "agent_b_extracted.json"
            with open(path_b, "w", encoding="utf-8") as f:
                json.dump(data_b, f, indent=2)
            summary_data["artifacts"]["agent_b"] = "agent_b_extracted.json"
            summary_data["agent_b_scores_count"] = len(extracted_b.extracted_scores)

        # 3. Consensus Matrix (if dual mode with ConsensusAuditReport)
        if audit_report and hasattr(audit_report, "consensus_scores"):
            consensus_data = {
                "model_id": getattr(audit_report, "model_id", model_id),
                "inter_model_dice_mae": getattr(audit_report, "inter_model_dice_mae", None),
                "concordance_rate": getattr(audit_report, "concordance_rate", 0.0),
                "concordant_count": getattr(audit_report, "concordant_count", 0),
                "dispute_count": getattr(audit_report, "dispute_count", 0),
                "unconfirmed_count": getattr(audit_report, "unconfirmed_count", 0),
                "paired_count": getattr(audit_report, "paired_count", 0),
                "total_unique_benchmarks": getattr(audit_report, "total_unique_benchmarks", 0),
                "screening": (
                    audit_report.consensus_screening.model_dump()
                    if hasattr(audit_report, "consensus_screening")
                    else None
                ),
                "metadata": (
                    audit_report.consensus_metadata.model_dump()
                    if hasattr(audit_report, "consensus_metadata")
                    else None
                ),
                "consensus_scores": [
                    cs.model_dump() if hasattr(cs, "model_dump") else cs
                    for cs in audit_report.consensus_scores
                ],
            }
            path_cons = paper_dir / "consensus_matrix.json"
            with open(path_cons, "w", encoding="utf-8") as f:
                json.dump(consensus_data, f, indent=2)
            summary_data["artifacts"]["consensus_matrix"] = "consensus_matrix.json"
            summary_data["concordance_rate"] = consensus_data["concordance_rate"]
            summary_data["inter_model_dice_mae"] = consensus_data["inter_model_dice_mae"]

        # 3.5 Deliberation Trajectory (if multi-round deliberation occurred)
        if deliberation_trajectory:
            traj_data = [
                d.model_dump() if hasattr(d, "model_dump") else d
                for d in deliberation_trajectory
            ]
            path_traj = paper_dir / "deliberation_trajectory.json"
            with open(path_traj, "w", encoding="utf-8") as f:
                json.dump(traj_data, f, indent=2)
            summary_data["artifacts"]["deliberation_trajectory"] = "deliberation_trajectory.json"
            summary_data["deliberation_rounds_count"] = len(deliberation_trajectory)

        # 4. Audit Report certification
        if audit_report:
            if hasattr(audit_report, "model_dump"):
                audit_dict = audit_report.model_dump()
            else:
                audit_dict = {
                    "status": getattr(audit_report.status, "value", str(audit_report.status)),
                    "confidence_score": getattr(audit_report, "confidence_score", None),
                }
            path_audit = paper_dir / "audit_report.json"
            with open(path_audit, "w", encoding="utf-8") as f:
                json.dump(audit_dict, f, indent=2)
            summary_data["artifacts"]["audit_report"] = "audit_report.json"
            summary_data["audit_status"] = getattr(audit_report.status, "value", str(audit_report.status))
            summary_data["confidence_score"] = getattr(audit_report, "confidence_score", None)

        # 5. Final Reconciled Data
        effective_final = final_extracted or extracted_a
        if effective_final:
            path_final = paper_dir / "final_reconciled.json"
            with open(path_final, "w", encoding="utf-8") as f:
                json.dump(effective_final.model_dump(), f, indent=2)
            summary_data["artifacts"]["final_reconciled"] = "final_reconciled.json"
            summary_data["final_scores_count"] = len(effective_final.extracted_scores)

        # 6. Evaluation against Ground Truth (if provided)
        if eval_report:
            eval_dict = {
                "matched_datasets_count": getattr(eval_report, "matched_datasets_count", None),
                "ground_truth_datasets_count": getattr(eval_report, "ground_truth_datasets_count", None),
                "dataset_recall": getattr(eval_report, "dataset_recall", None),
                "dice_mae": getattr(eval_report, "dice_mae", None),
                "dice_mae_primary": getattr(eval_report, "dice_mae_primary", None),
                "dice_mae_best": getattr(eval_report, "dice_mae_best", None),
                "dice_mae_regime_agnostic": getattr(eval_report, "dice_mae_regime_agnostic", None),
                "score_comparisons": getattr(eval_report, "score_comparisons", []),
            }
            path_eval = paper_dir / "evaluation.json"
            with open(path_eval, "w", encoding="utf-8") as f:
                json.dump(eval_dict, f, indent=2)
            summary_data["artifacts"]["evaluation"] = "evaluation.json"
            summary_data["dice_mae"] = eval_dict["dice_mae"]
            summary_data["dataset_recall"] = eval_dict["dataset_recall"]

        # 7. Outreach Draft (if generated)
        if outreach_draft:
            path_outreach_json = paper_dir / "outreach_draft.json"
            outreach_dict = (
                outreach_draft.model_dump()
                if hasattr(outreach_draft, "model_dump")
                else outreach_draft
            )
            with open(path_outreach_json, "w", encoding="utf-8") as f:
                json.dump(outreach_dict, f, indent=2)
            summary_data["artifacts"]["outreach_draft"] = "outreach_draft.json"

            # Markdown inquiry text for immediate reading
            if hasattr(outreach_draft, "email_body") and hasattr(outreach_draft, "email_subject"):
                path_outreach_md = paper_dir / "outreach_draft.md"
                with open(path_outreach_md, "w", encoding="utf-8") as f:
                    f.write(f"# Academic Inquiry: {outreach_draft.email_subject}\n\n")
                    f.write(f"- **Recipient**: {outreach_draft.recipient.name} ({outreach_draft.recipient.email or 'N/A'})\n")
                    f.write(f"- **Model**: {outreach_draft.model_id}\n\n")
                    f.write("## Email Text\n\n```text\n")
                    f.write(outreach_draft.email_body)
                    f.write("\n```\n")
                summary_data["artifacts"]["outreach_markdown"] = "outreach_draft.md"

        # 8. Repo Inspector Result (if triggered)
        if repo_inspector_result:
            path_repo = paper_dir / "repo_inspection.json"
            repo_dict = (
                repo_inspector_result.model_dump()
                if hasattr(repo_inspector_result, "model_dump")
                else repo_inspector_result
            )
            with open(path_repo, "w", encoding="utf-8") as f:
                json.dump(repo_dict, f, indent=2)
            summary_data["artifacts"]["repo_inspection"] = "repo_inspection.json"

        # 9. Database Curation Artifacts (if generated)
        if curation_result:
            plan = getattr(curation_result, "plan", None)
            if plan:
                path_curation_json = paper_dir / "curation_plan.json"
                plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan
                with open(path_curation_json, "w", encoding="utf-8") as f:
                    json.dump(plan_dict, f, indent=2)
                summary_data["artifacts"]["curation_plan"] = "curation_plan.json"

                # curation.sql
                if plan.generated_sql:
                    path_curation_sql = paper_dir / "curation.sql"
                    with open(path_curation_sql, "w", encoding="utf-8") as f:
                        f.write(f"-- SQL Curation Statements for {model_id}\n")
                        f.write(f"-- Action: {plan.model_action.value} | Status: {plan.audit_status}\n\n")
                        f.write("\n\n".join(plan.generated_sql) + "\n")
                    summary_data["artifacts"]["curation_sql"] = "curation.sql"

                # curation_summary.md
                if plan.summary:
                    path_curation_md = paper_dir / "curation_summary.md"
                    with open(path_curation_md, "w", encoding="utf-8") as f:
                        f.write(f"# Database Curation: {model_id}\n\n")
                        f.write(plan.summary + "\n")
                    summary_data["artifacts"]["curation_summary"] = "curation_summary.md"

                summary_data["curation"] = {
                    "action": getattr(plan.model_action, "value", str(plan.model_action)),
                    "can_commit": plan.can_commit,
                    "committed": getattr(curation_result, "committed", False),
                    "sql_count": len(plan.generated_sql),
                    "novel_benchmarks_count": len(plan.schema_changes),
                }

        # 10. Paper Summary Index
        path_summary = paper_dir / "paper_summary.json"
        with open(path_summary, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        return paper_dir

    def save_run_results(
        self,
        json_results: Dict[str, Any],
        markdown_report: str,
        database_curation_sql: Optional[str] = None,
        database_curation_report: Optional[str] = None,
        update_latest_symlinks: bool = True,
    ) -> None:
        """
        Saves aggregated benchmark results and updates convenience links in the logs root.

        Args:
            json_results: Full structured dictionary of benchmark results.
            markdown_report: Formatted Markdown report string.
            database_curation_sql: Optional complete SQL curation script for all models.
            database_curation_report: Optional Markdown report of database curation.
            update_latest_symlinks: If True, writes copies/symlinks to logs/latest_batch_benchmark.*
        """
        # Save within dedicated run directory
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2)

        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write(markdown_report)

        if database_curation_sql:
            curation_sql_file = self.run_dir / "database_curation.sql"
            with open(curation_sql_file, "w", encoding="utf-8") as f:
                f.write(database_curation_sql)

        if database_curation_report:
            curation_rep_file = self.run_dir / "database_curation_report.md"
            with open(curation_rep_file, "w", encoding="utf-8") as f:
                f.write(database_curation_report)

        # Update latest pointers in logs root for backward compatibility
        if update_latest_symlinks:
            latest_json = LOGS_DIR / "latest_batch_benchmark.json"
            latest_md = LOGS_DIR / "latest_batch_benchmark.md"
            latest_run_link = LOGS_DIR / "latest_run"

            with open(latest_json, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2)

            with open(latest_md, "w", encoding="utf-8") as f:
                f.write(markdown_report)

            if database_curation_sql:
                latest_sql = LOGS_DIR / "latest_database_curation.sql"
                with open(latest_sql, "w", encoding="utf-8") as f:
                    f.write(database_curation_sql)

            if database_curation_report:
                latest_cur_rep = LOGS_DIR / "latest_database_curation_report.md"
                with open(latest_cur_rep, "w", encoding="utf-8") as f:
                    f.write(database_curation_report)

            # Create or update latest_run symlink
            try:
                if latest_run_link.is_symlink() or latest_run_link.exists():
                    latest_run_link.unlink()
                # Relative symlink from logs/ to logs/runs/<run_id>
                rel_target = Path("runs") / self.run_id
                latest_run_link.symlink_to(rel_target, target_is_directory=True)
            except Exception:
                pass

    def close(self) -> None:
        """Flushes and closes the execution log tee."""
        if self.tee:
            sys.stdout = self._original_stdout or sys.__stdout__
            self.tee.close()
            self.tee = None


def archive_legacy_flat_logs(logs_dir: Optional[Path] = None) -> int:
    """
    Moves loose benchmark files in the root logs/ directory to logs/archive/legacy_flat_logs/.

    Preserves existing subdirectories (gemini_30, mufasa_30, runs, etc.)
    and latest_batch_benchmark.* files.

    Returns:
        Number of files archived.
    """
    target_logs = Path(logs_dir) if logs_dir else LOGS_DIR
    archive_dir = target_logs / "archive" / "legacy_flat_logs"
    archive_dir.mkdir(parents=True, exist_ok=True)

    archived_count = 0
    patterns = [
        "batch_benchmark_20*.log",
        "batch_benchmark_report_20*.md",
        "batch_benchmark_results_20*.json",
        "mufasa_bench_*.out",
        "mufasa_bench_*.err",
        "gmmis_job_*.out",
        "gmmis_job_*.err",
    ]

    for pattern in patterns:
        for file_path in target_logs.glob(pattern):
            if file_path.is_file():
                dest = archive_dir / file_path.name
                shutil.move(str(file_path), str(dest))
                archived_count += 1

    return archived_count
