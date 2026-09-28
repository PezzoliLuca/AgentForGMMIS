"""Batch benchmark runner: evaluates ExtractionAgent across multiple literature papers automatically.

Logs execution and saves comprehensive markdown/table reports to the logs/ directory.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from tabulate import tabulate

from agent_gmmis.config import DEFAULT_DB_PATH
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.extraction.evaluator import ExtractionEvaluator
from agent_gmmis.extraction.extractor import ExtractionAgent
from agent_gmmis.extraction.paper_fetcher import PaperFetcher
from agent_gmmis.llm.client import LLMClient, LLMConfig


class LoggerTee:
    """Duplicates stdout to both terminal and a log file."""

    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def main():
    parser = argparse.ArgumentParser(description="Run ExtractionAgent on multiple scientific papers automatically.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of model IDs from database.sqlite to test (e.g. 'SAM-Med3D' 'Merlin')",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of available arXiv models to test automatically if --models is not specified (default: 10)",
    )
    parser.add_argument("--provider", type=str, default=None, help="LLM provider: gemini, ollama, mufasa, or mock")
    parser.add_argument("--model", type=str, default=None, help="LLM model name (default: gemini-3.5-flash-lite)")
    parser.add_argument(
        "--compare-source",
        type=str,
        default="primary",
        choices=["primary", "best"],
        help="Ground truth source to compare against: 'primary' (paper report) or 'best' (literature best)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs",
        help="Directory where logs and markdown reports will be saved (default: logs)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"batch_benchmark_{timestamp}.log"
    report_path = out_dir / f"batch_benchmark_report_{timestamp}.md"
    latest_report_path = out_dir / "latest_batch_benchmark.md"

    # Set up dual logging to console and file
    tee = LoggerTee(log_path)
    sys.stdout = tee

    try:
        db = DatabaseManager(DEFAULT_DB_PATH)
        fetcher = PaperFetcher(db_manager=db)

        # Determine models to evaluate
        if args.models:
            target_model_ids = args.models
        else:
            all_models = db.get_all_models()
            target_model_ids = []
            for m in all_models:
                link = m.first_publication_link or m.last_publication_link or ""
                if "arxiv" in link.lower():
                    scores = db.get_scores_for_model(m.id)
                    if any(scores[d].get(args.compare_source) is not None for d in scores):
                        target_model_ids.append(m.id)
                        if len(target_model_ids) >= args.count:
                            break

        start_time_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "=" * 80)
        print("🔬 AGENT FOR GMMIS — BATCH EXTRACTION BENCHMARK")
        print(f"Timestamp:       {start_time_iso}")
        print(f"Papers to Test:  {len(target_model_ids)}")
        print(f"Target Models:   {', '.join(target_model_ids)}")
        print(f"LLM Provider:    {args.provider or 'gemini'} ({args.model or 'gemini-3.5-flash-lite'})")
        print(f"Ground Truth:    {args.compare_source.upper()} literature records")
        print(f"Log File:        {log_path}")
        print("=" * 80)

        config_kwargs = {}
        if args.provider:
            config_kwargs["provider"] = args.provider
        if args.model:
            config_kwargs["model"] = args.model
        llm_config = LLMConfig(**config_kwargs)
        llm_client = LLMClient(config=llm_config)
        agent = ExtractionAgent(llm_client=llm_client)
        evaluator = ExtractionEvaluator(db_manager=db)

        from agent_gmmis.auditing.auditor import AuditorAgent
        from agent_gmmis.extraction.pdf_parser import PDFParser

        auditor = AuditorAgent(db_manager=db)
        summary_rows = []
        detailed_reports = []

        for idx, model_id in enumerate(target_model_ids, 1):
            print(f"\n[{idx}/{len(target_model_ids)}] 🚀 Processing '{model_id}'...")
            t0 = time.time()
            try:
                pdf_path = fetcher.fetch_for_model(model_id)
                manuscript = PDFParser.parse(pdf_path)
                extracted = agent.process_manuscript(manuscript)
                audit_report = auditor.audit(extracted, manuscript=manuscript)
                report = evaluator.evaluate(extracted, ground_truth_id=model_id, compare_source=args.compare_source)
                elapsed = time.time() - t0

                dice_mae_str = f"{report.dice_mae:.4f}" if report.dice_mae is not None else "N/A"
                audit_status_badge = (
                    "✅ VERIFIED"
                    if audit_report.status.value == "verified"
                    else ("⚠️ " + audit_report.status.value.upper())
                )

                summary_rows.append([
                    model_id,
                    audit_status_badge,
                    f"{audit_report.confidence_score * 100:.0f}%",
                    f"{report.matched_datasets_count}/{report.ground_truth_datasets_count}",
                    f"{report.dataset_recall:.1%}",
                    dice_mae_str,
                    f"{elapsed:.1f}s",
                ])

                detailed_reports.append({
                    "model_id": model_id,
                    "report": report,
                    "extracted": extracted,
                    "audit_report": audit_report,
                    "elapsed": elapsed,
                    "pdf_name": pdf_path.name,
                })

                audit_notes = []
                if audit_report.discrepancies:
                    audit_notes.append(f"{len(audit_report.discrepancies)} survey typo/discrepancies")
                if audit_report.outliers:
                    audit_notes.append(f"{len(audit_report.outliers)} outliers")
                if audit_report.data_diagnosis.value != "none":
                    audit_notes.append(f"diagnosis: {audit_report.data_diagnosis.value}")
                notes_str = f" — Audit: [{', '.join(audit_notes)}]" if audit_notes else ""

                print(f"   ✅ Done in {elapsed:.1f}s — Audit: {audit_report.status.value.upper()} (conf={audit_report.confidence_score*100:.0f}%) — Scores: {len(extracted.extracted_scores)}, Matched: {report.matched_datasets_count}/{report.ground_truth_datasets_count}, Dice MAE: {dice_mae_str}{notes_str}")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"   ❌ Error processing {model_id}: {e}")
                summary_rows.append([
                    model_id,
                    "❌ ERROR",
                    "0%",
                    "0/0",
                    "0.0%",
                    "N/A",
                    f"{elapsed:.1f}s",
                ])

            # Polite pause between papers to stay well within free tier burst limits
            if idx < len(target_model_ids):
                time.sleep(3.5)

        # Print Aggregated Summary Table to console and log
        print("\n" + "=" * 80)
        print("📊 BATCH BENCHMARK & AUDIT AGGREGATED REPORT")
        print("=" * 80)
        headers = ["Model ID", "Audit Status", "Confidence", "Datasets Matched", "Recall", "Dice MAE", "Time"]
        summary_table_str = tabulate(summary_rows, headers=headers, tablefmt="github")
        print(summary_table_str)

        # Write detailed Markdown Report
        md_lines = [
            "# Batch Extraction Benchmark Report",
            "",
            f"- **Execution Date**: `{start_time_iso}`",
            f"- **LLM Provider**: `{args.provider or 'gemini'}` (`{args.model or 'gemini-3.5-flash-lite'}`)",
            f"- **Evaluation Baseline**: `{args.compare_source.upper()}` Survey Ground Truth",
            f"- **Total Papers Evaluated**: {len(target_model_ids)}",
            "",
            "## Summary Results",
            "",
            summary_table_str,
            "",
            "---",
            "",
            "## Detailed Per-Paper Comparisons",
            "",
        ]

        for item in detailed_reports:
            mid = item["model_id"]
            rep = item["report"]
            ext = item["extracted"]
            elap = item["elapsed"]
            md_lines.append(f"### Model: {mid}")
            md_lines.append(f"- **Paper Title**: {ext.paper_title}")
            md_lines.append(f"- **PDF File**: `{item['pdf_name']}` (cached)")
            md_lines.append(f"- **Visual Backbone**: {ext.visual_backbone or 'N/A'}")
            md_lines.append(f"- **Repository URL**: {ext.code_repository_url or 'N/A'}")
            md_lines.append(f"- **Execution Time**: {elap:.1f}s")
            if "audit_report" in item:
                ar = item["audit_report"]
                md_lines.append(f"- **Audit Status**: `{ar.status.value.upper()}` (Confidence: {ar.confidence_score*100:.1f}%)")
                if ar.anomalies:
                    md_lines.append(f"- **Anomalies**: {'; '.join(a.message for a in ar.anomalies)}")
                if ar.outliers:
                    md_lines.append(f"- **Outliers**: {'; '.join(o.reason or '' for o in ar.outliers)}")
                if ar.discrepancies:
                    md_lines.append(f"- **Survey Discrepancies**: {'; '.join(d.message for d in ar.discrepancies)}")
                if ar.data_diagnosis.value != "none":
                    md_lines.append(f"- **Missing Data Diagnosis**: `{ar.data_diagnosis.value}`")
            md_lines.append("")
            if rep.score_comparisons:
                md_lines.append("#### Benchmark Table Matches vs. Ground Truth:")
                comp_rows = [
                    [c["dataset"], c.get("prompt_type", "N/A"), f"{c['predicted']:.4f}", f"{c['ground_truth']:.4f}", f"{c['diff']:.4f}"]
                    for c in rep.score_comparisons
                ]
                md_lines.append(tabulate(comp_rows, headers=["Dataset", "Prompt Setting", "Extracted", "Ground Truth", "Diff (Absolute)"], tablefmt="github"))
            else:
                md_lines.append("*No direct benchmark dataset matches found in primary ground truth.*")
                if ext.extracted_scores:
                    datasets_seen = ", ".join(sorted(set(s.dataset_name for s in ext.extracted_scores))[:8])
                    md_lines.append(f"*(Extracted scores from: {datasets_seen})*")
            md_lines.append("")

        report_content = "\n".join(md_lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        with open(latest_report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print("\n" + "=" * 80)
        print(f"📄 Detailed Markdown report saved to: {report_path}")
        print(f"📄 Latest report available at:        {latest_report_path}")
        print(f"📝 Full execution log saved to:       {log_path}")
        print("=" * 80 + "\n")

    finally:
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()
