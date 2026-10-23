"""Batch benchmark runner: evaluates ExtractionAgent across multiple literature papers automatically.

Logs execution and saves comprehensive markdown/table reports to the logs/ directory.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

# Ensure src/ is on sys.path regardless of execution working directory or environment setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tabulate import tabulate

from agent_gmmis.config import DEFAULT_DB_PATH, LOGS_DIR
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.extraction.evaluator import ExtractionEvaluator
from agent_gmmis.extraction.extractor import ExtractionAgent
from agent_gmmis.extraction.paper_fetcher import PaperFetcher
from agent_gmmis.extraction.run_logger import RunLogger, archive_legacy_flat_logs
from agent_gmmis.llm.client import LLMClient, LLMConfig


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
    parser.add_argument("--provider-a", type=str, default=None, help="LLM provider for Agent A (default: --provider or gemini)")
    parser.add_argument("--model-a", type=str, default=None, help="LLM model name for Agent A (default: --model or gemini-3.5-flash-lite)")
    parser.add_argument("--provider-b", type=str, default=None, help="LLM provider for Agent B: gemini, ollama, mufasa, or mock")
    parser.add_argument("--model-b", type=str, default=None, help="LLM model name for Agent B. If specified, enables dual-agent parallel consensus mode.")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for OpenAI-compatible LLM endpoint")
    parser.add_argument("--base-url-a", type=str, default=None, help="Base URL for Agent A LLM endpoint")
    parser.add_argument("--base-url-b", type=str, default=None, help="Base URL for Agent B LLM endpoint")
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
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag for the run directory name (e.g. mufasa_dual_20)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional exact custom name for the run directory",
    )
    parser.add_argument(
        "--audit-ground-truth",
        action="store_true",
        default=False,
        help="If set, AuditorAgent cross-checks extracted values against human survey ground truth in SQLite (benchmark mode). Default: False (autonomous deploy mode).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum deliberation rounds in dual-agent mode (default: 3; set to 1 to disable re-interrogation)",
    )
    parser.add_argument(
        "--archive-legacy",
        action="store_true",
        default=False,
        help="Move older loose flat files in logs/ to logs/archive/legacy_flat_logs/",
    )
    args = parser.parse_args()

    if args.archive_legacy:
        archived_count = archive_legacy_flat_logs()
        if archived_count > 0:
            print(f"📦 Archived {archived_count} legacy flat log files to logs/archive/legacy_flat_logs/")

    provider_a = args.provider_a or args.provider or "gemini"
    model_a = args.model_a or args.model or "gemini-3.5-flash-lite"
    provider_b = args.provider_b or provider_a
    model_b = args.model_b
    is_dual = model_b is not None
    base_url_a = args.base_url_a or args.base_url
    base_url_b = args.base_url_b or args.base_url

    tag = args.tag or (f"dual_{provider_a}_{provider_b}" if is_dual else f"single_{provider_a}")
    base_runs_dir = Path(args.output_dir) if args.output_dir != "logs" else None

    run_logger = RunLogger(
        base_dir=base_runs_dir,
        run_name=args.run_name,
        tag=tag,
        is_dual=is_dual,
        capture_stdout=True,
        metadata={
            "provider_a": provider_a,
            "model_a": model_a,
            "base_url_a": base_url_a,
            "provider_b": provider_b if is_dual else None,
            "model_b": model_b if is_dual else None,
            "base_url_b": base_url_b if is_dual else None,
            "compare_source": args.compare_source,
            "audit_ground_truth": args.audit_ground_truth,
            "max_rounds": args.max_rounds,
        },
    )

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
        print(f"Run ID:          {run_logger.run_id}")
        print(f"Timestamp:       {start_time_iso}")
        print(f"Papers to Test:  {len(target_model_ids)}")
        print(f"Target Models:   {', '.join(target_model_ids)}")
        if is_dual:
            print(f"Pipeline Mode:   Dual-Agent Parallel Consensus Extraction")
            print(f"Agent A:         {provider_a} ({model_a})")
            print(f"Agent B:         {provider_b} ({model_b})")
        else:
            print(f"Pipeline Mode:   Single-Agent Extraction")
            print(f"LLM Provider:    {provider_a} ({model_a})")
        print(f"Ground Truth:    {args.compare_source.upper()} literature records")
        print(f"Run Directory:   {run_logger.run_dir}")
        print(f"Execution Log:   {run_logger.log_file}")
        print("=" * 80)

        evaluator = ExtractionEvaluator(db_manager=db)

        from agent_gmmis.auditing.auditor import AuditorAgent
        from agent_gmmis.auditing.outreach import OutreachAgent
        from agent_gmmis.auditing.repo_inspector import RepoInspector
        from agent_gmmis.auditing.schemas import AuditStatus
        from agent_gmmis.extraction.pdf_parser import PDFParser
        from agent_gmmis.extraction.schemas import ExtractedPaperData, RawExtractedScore

        auditor = AuditorAgent(db_manager=db)
        repo_inspector = RepoInspector()
        outreach_agent = OutreachAgent()

        if is_dual:
            from agent_gmmis.extraction.dual_coordinator import DualExtractionCoordinator
            client_a = LLMClient(config=LLMConfig(provider=provider_a, model=model_a, base_url=base_url_a))
            agent_a = ExtractionAgent(llm_client=client_a)
            client_b = LLMClient(config=LLMConfig(provider=provider_b, model=model_b, base_url=base_url_b))
            agent_b = ExtractionAgent(llm_client=client_b)
            coordinator = DualExtractionCoordinator(agent_a=agent_a, agent_b=agent_b)
        else:
            config_kwargs = {}
            if provider_a:
                config_kwargs["provider"] = provider_a
            if model_a:
                config_kwargs["model"] = model_a
            if base_url_a:
                config_kwargs["base_url"] = base_url_a
            llm_config = LLMConfig(**config_kwargs)
            llm_client = LLMClient(config=llm_config)
            agent = ExtractionAgent(llm_client=llm_client)

        summary_rows = []
        detailed_reports = []

        for idx, model_id in enumerate(target_model_ids, 1):
            print(f"\n[{idx}/{len(target_model_ids)}] 🚀 Processing '{model_id}'...")
            t0 = time.time()
            try:
                pdf_path = fetcher.fetch_for_model(model_id)
                manuscript = PDFParser.parse(pdf_path)

                deliberation_trajectory = []
                if is_dual:
                    if args.max_rounds > 1:
                        extracted_a, extracted_b, audit_report, deliberation_trajectory = coordinator.extract_manuscript_with_deliberation(
                            manuscript=manuscript,
                            auditor=auditor,
                            max_rounds=args.max_rounds,
                            benchmark_mode=args.audit_ground_truth,
                            verbose=True,
                        )
                    else:
                        extracted_a, extracted_b = coordinator.extract_manuscript_parallel(manuscript)
                        audit_report = auditor.audit_consensus(
                            extracted_a,
                            extracted_b,
                            manuscript=manuscript,
                            benchmark_mode=args.audit_ground_truth,
                        )
                    reconciled_scores = []
                    for cs in audit_report.consensus_scores:
                        if cs.resolved_score is not None:
                            reconciled_scores.append(
                                RawExtractedScore(
                                    dataset_name=cs.canonical_dataset_name or cs.dataset_name or "Unknown",
                                    organ_or_task=cs.organ_or_task or "Overall",
                                    dice_score=cs.resolved_score,
                                    prompt_type=cs.prompt_type,
                                    table_or_section_ref=cs.table_ref_a or cs.table_ref_b,
                                )
                            )
                    extracted = ExtractedPaperData(
                        paper_title=extracted_a.paper_title,
                        proposed_model_id=audit_report.consensus_metadata.resolved_model_id,
                        visual_backbone=audit_report.consensus_metadata.resolved_visual_backbone,
                        millions_of_parameters=audit_report.consensus_metadata.resolved_params_m,
                        code_repository_url=audit_report.consensus_metadata.code_repository_url,
                        screening=extracted_a.screening,
                        extracted_scores=reconciled_scores,
                    )
                else:
                    extracted = agent.process_manuscript(manuscript)
                    extracted_a = extracted
                    extracted_b = None
                    audit_report = auditor.audit(extracted, manuscript=manuscript, benchmark_mode=args.audit_ground_truth)

                # Delegation Layer: GitHub Repository Inspector
                repo_recovered = False
                if audit_report.status == AuditStatus.NEEDS_REPO_INSPECTION and extracted.code_repository_url:
                    print(f"   🔍 [Delegation Layer] Triggering RepoInspector for '{model_id}' ({extracted.code_repository_url})...")
                    extracted, reaudited_report = repo_inspector.inspect_and_update(extracted, auditor=auditor, manuscript=manuscript)
                    if reaudited_report:
                        audit_report = reaudited_report
                        if audit_report.status not in (AuditStatus.NEEDS_OUTREACH, AuditStatus.NEEDS_REPO_INSPECTION):
                            repo_recovered = True
                            print(f"   ✨ RepoInspector recovered scores! New Status: {audit_report.status.value.upper()} (conf={audit_report.confidence_score*100:.0f}%)")
                        else:
                            print(f"   ℹ️ Repo inspected but lacked benchmark tables. Delegating to OutreachAgent...")

                # Delegation Layer: OutreachAgent (Author Email Inquiry)
                outreach_draft = None
                if audit_report.status == AuditStatus.NEEDS_OUTREACH:
                    print(f"   ✉️ [Delegation Layer] Triggering OutreachAgent for '{model_id}'...")
                    outreach_draft = outreach_agent.handle_audit_report(extracted, manuscript, audit_report, auto_save=True)
                    if outreach_draft:
                        print(f"   📝 Generated inquiry draft for {outreach_draft.recipient.name} ({outreach_draft.recipient.email or 'no email'}): {outreach_draft.draft_filepath}")

                report = evaluator.evaluate(extracted, ground_truth_id=model_id, compare_source=args.compare_source)
                elapsed = time.time() - t0

                # Persist intermediate agent extractions and consensus matrix
                paper_dir = run_logger.log_paper_extraction(
                    model_id=model_id,
                    extracted_a=extracted_a,
                    extracted_b=extracted_b if is_dual else None,
                    audit_report=audit_report,
                    final_extracted=extracted,
                    eval_report=report,
                    repo_inspector_result=repo_inspector if repo_recovered else None,
                    outreach_draft=outreach_draft,
                    elapsed_seconds=elapsed,
                    deliberation_trajectory=deliberation_trajectory,
                )

                dice_mae_str = f"{report.dice_mae:.4f}" if report.dice_mae is not None else "N/A"
                if outreach_draft:
                    audit_status_badge = "✉️ OUTREACH DRAFTED"
                elif repo_recovered:
                    audit_status_badge = "✅ VERIFIED (REPO)"
                elif audit_report.status.value == "verified":
                    audit_status_badge = "✅ VERIFIED"
                else:
                    audit_status_badge = "⚠️ " + audit_report.status.value.upper()

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
                    "extracted_a": extracted_a,
                    "extracted_b": extracted_b if is_dual else None,
                    "audit_report": audit_report,
                    "elapsed": elapsed,
                    "pdf_name": pdf_path.name,
                    "repo_recovered": repo_recovered,
                    "outreach_draft": outreach_draft,
                    "paper_dir": paper_dir,
                })

                audit_notes = []
                if hasattr(audit_report, "consensus_scores"):
                    paired_c = getattr(audit_report, "paired_count", 0)
                    conc_c = audit_report.concordant_count
                    if paired_c > 0:
                        audit_notes.append(f"paired agreement: {conc_c}/{paired_c} ({audit_report.concordance_rate*100:.0f}%)")
                    else:
                        audit_notes.append(f"concordance: {conc_c}/{audit_report.total_unique_benchmarks}")
                    if audit_report.dispute_count > 0:
                        audit_notes.append(f"disputes: {audit_report.dispute_count}")
                    if audit_report.unconfirmed_count > 0:
                        audit_notes.append(f"singletons: {audit_report.unconfirmed_count}")
                    if audit_report.inter_model_dice_mae is not None:
                        audit_notes.append(f"inter-model MAE: {audit_report.inter_model_dice_mae:.4f}")
                if getattr(audit_report, "discrepancies", None):
                    audit_notes.append(f"{len(audit_report.discrepancies)} survey typo/discrepancies")
                if getattr(audit_report, "outliers", None):
                    audit_notes.append(f"{len(audit_report.outliers)} outliers")
                if getattr(audit_report, "data_diagnosis", None) and audit_report.data_diagnosis.value != "none":
                    audit_notes.append(f"diagnosis: {audit_report.data_diagnosis.value}")
                notes_str = f" — Audit: [{', '.join(audit_notes)}]" if audit_notes else ""

                print(f"   ✅ Done in {elapsed:.1f}s — Audit: {audit_report.status.value.upper()} (conf={audit_report.confidence_score*100:.0f}%) — Scores: {len(extracted.extracted_scores)}, Matched: {report.matched_datasets_count}/{report.ground_truth_datasets_count}, Dice MAE: {dice_mae_str}{notes_str}")
                print(f"   📂 Artifacts saved to: {paper_dir.relative_to(run_logger.run_dir)}")

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
                detailed_reports.append({
                    "model_id": model_id,
                    "error": str(e),
                    "elapsed": elapsed,
                    "repo_recovered": False,
                })

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
        auditor_mode_desc = (
            "Benchmark Ground Truth Cross-Check"
            if args.audit_ground_truth
            else "Autonomous Deploy Mode (Zero Ground Truth)"
        )
        md_lines = [
            "# Batch Extraction Benchmark Report",
            "",
            f"- **Execution Date**: `{start_time_iso}`",
            f"- **Pipeline Mode**: `{'Dual-Agent Parallel Consensus' if is_dual else 'Single-Agent Extraction'}`",
            f"- **Agent A**: `{provider_a}` (`{model_a}`)",
            f"- **Agent B**: `{(provider_b + ' (' + model_b + ')') if is_dual else 'N/A'}`",
            f"- **Auditor Mode**: `{auditor_mode_desc}`",
            f"- **Evaluation Baseline**: `{args.compare_source.upper()}` Survey Ground Truth",
            f"- **Total Papers Evaluated**: {len(target_model_ids)}",
            "",
            "## Summary Results",
            "",
            summary_table_str,
            "",
        ]

        outreach_items = [it for it in detailed_reports if it.get("outreach_draft")]
        if outreach_items:
            md_lines.extend([
                "---",
                "",
                "## ✉️ Scholarly Outreach Inquiries (Author Correspondence Drafts)",
                "",
                "The following manuscripts presented evaluation results primarily in visual or graphical format without scalar tables. The Delegation Layer automatically drafted formal academic inquiries for Human-in-the-Loop review:",
                "",
            ])
            outreach_table = [
                [
                    it["model_id"],
                    it["outreach_draft"].recipient.name,
                    it["outreach_draft"].recipient.email or "N/A",
                    ", ".join(it["outreach_draft"].target_datasets[:3]),
                    ", ".join(it["outreach_draft"].target_figures[:2]),
                    f"[{Path(it['outreach_draft'].draft_filepath).stem}.md](file://{it['outreach_draft'].draft_filepath.replace('.json', '.md')})" if it["outreach_draft"].draft_filepath else "Saved",
                ]
                for it in outreach_items
            ]
            md_lines.append(tabulate(outreach_table, headers=["Model", "Corresponding Author", "Email", "Target Datasets", "Target Figures", "Review Draft"], tablefmt="github"))
            md_lines.append("")

        md_lines.extend([
            "---",
            "",
            "## Detailed Per-Paper Comparisons",
            "",
        ])

        # Machine-readable JSON output data structure
        json_results = {
            "metadata": {
                "timestamp": start_time_iso,
                "is_dual_mode": is_dual,
                "provider_a": provider_a,
                "model_a": model_a,
                "provider_b": provider_b if is_dual else None,
                "model_b": model_b if is_dual else None,
                "auditor_mode": auditor_mode_desc,
                "compare_source": args.compare_source,
                "total_papers": len(target_model_ids),
                "run_id": run_logger.run_id,
            },
            "summary": [
                {
                    "model_id": row[0],
                    "audit_status": row[1],
                    "confidence": row[2],
                    "matched_datasets": row[3],
                    "recall": row[4],
                    "dice_mae": row[5],
                    "elapsed_time": row[6],
                }
                for row in summary_rows
            ],
            "papers": [],
        }

        for item in detailed_reports:
            mid = item["model_id"]
            if "error" in item:
                md_lines.append(f"### Model: {mid}")
                md_lines.append(f"- **Status**: ❌ ERROR")
                md_lines.append(f"- **Error Details**: `{item['error']}`")
                md_lines.append(f"- **Execution Time**: {item['elapsed']:.1f}s\n")
                json_results["papers"].append({
                    "model_id": mid,
                    "error": item["error"],
                    "elapsed_seconds": item["elapsed"],
                    "audit_status": "error",
                    "confidence_score": 0.0,
                    "repo_recovered": False,
                    "dataset_recall": 0.0,
                    "dice_mae": None,
                })
                continue

            rep = item["report"]
            ext = item["extracted"]
            elap = item["elapsed"]
            ar = item.get("audit_report")
            md_lines.append(f"### Model: {mid}")
            md_lines.append(f"- **Paper Title**: {ext.paper_title}")
            md_lines.append(f"- **PDF File**: `{item['pdf_name']}` (cached)")
            md_lines.append(f"- **Visual Backbone**: {ext.visual_backbone or 'N/A'}")
            md_lines.append(f"- **Repository URL**: {ext.code_repository_url or 'N/A'}")
            md_lines.append(f"- **Execution Time**: {elap:.1f}s")
            if "paper_dir" in item:
                md_lines.append(f"- **Intermediate Artifacts**: `{item['paper_dir'].relative_to(run_logger.run_dir)}/`")
            if ar:
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

            ext_a = item.get("extracted_a")
            ext_b = item.get("extracted_b")

            # Render Chain-of-Thought (CoT) accordion if thought_process is present
            has_cot = (
                (ext_a and getattr(ext_a, "thought_process", None))
                or (ext_b and getattr(ext_b, "thought_process", None))
                or getattr(ext, "thought_process", None)
            )
            if has_cot:
                md_lines.append("<details>")
                md_lines.append("<summary>🧠 <b>Chain-of-Thought (CoT) & Extraction Reasoning</b></summary>\n")
                if ext_a and getattr(ext_a, "thought_process", None) and ext_a.thought_process:
                    tp = ext_a.thought_process
                    md_lines.append(f"**Agent A ({provider_a}):**")
                    if tp.proposed_model_identification:
                        md_lines.append(f"- **Model Identification**: {tp.proposed_model_identification}")
                    if tp.primary_benchmark_tables:
                        md_lines.append(f"- **Primary Tables**: {tp.primary_benchmark_tables}")
                    if tp.peak_performance_logic:
                        md_lines.append(f"- **Peak Performance Logic**: {tp.peak_performance_logic}")
                    if tp.multitask_or_macro_averages:
                        md_lines.append(f"- **Macro-Averages**: {tp.multitask_or_macro_averages}")
                    if tp.ambiguities_or_caveats:
                        md_lines.append(f"- **Ambiguities / Caveats**: {tp.ambiguities_or_caveats}")
                    md_lines.append("")
                if is_dual and ext_b and getattr(ext_b, "thought_process", None) and ext_b.thought_process:
                    tp = ext_b.thought_process
                    md_lines.append(f"**Agent B ({provider_b}):**")
                    if tp.proposed_model_identification:
                        md_lines.append(f"- **Model Identification**: {tp.proposed_model_identification}")
                    if tp.primary_benchmark_tables:
                        md_lines.append(f"- **Primary Tables**: {tp.primary_benchmark_tables}")
                    if tp.peak_performance_logic:
                        md_lines.append(f"- **Peak Performance Logic**: {tp.peak_performance_logic}")
                    if tp.multitask_or_macro_averages:
                        md_lines.append(f"- **Macro-Averages**: {tp.multitask_or_macro_averages}")
                    if tp.ambiguities_or_caveats:
                        md_lines.append(f"- **Ambiguities / Caveats**: {tp.ambiguities_or_caveats}")
                    md_lines.append("")
                md_lines.append("</details>\n")

            paper_entry = {
                "model_id": mid,
                "paper_title": ext.paper_title,
                "pdf_name": item["pdf_name"],
                "visual_backbone": ext.visual_backbone,
                "code_repository_url": ext.code_repository_url,
                "elapsed_seconds": elap,
                "audit_status": ar.status.value if ar else None,
                "confidence_score": ar.confidence_score if ar else None,
                "repo_recovered": item.get("repo_recovered", False),
                "anomalies": [a.message for a in ar.anomalies] if ar else [],
                "outliers": [o.reason for o in ar.outliers] if ar else [],
                "discrepancies": [d.message for d in ar.discrepancies] if ar else [],
                "data_diagnosis": ar.data_diagnosis.value if ar else "none",
                "extracted_scores_count": len(ext.extracted_scores),
                "matched_datasets_count": rep.matched_datasets_count,
                "ground_truth_datasets_count": rep.ground_truth_datasets_count,
                "dataset_recall": rep.dataset_recall,
                "dice_mae": rep.dice_mae,
                "score_comparisons": rep.score_comparisons,
                "extractions_dir": str(item["paper_dir"].relative_to(run_logger.run_dir)) if "paper_dir" in item else None,
                "is_dual": is_dual,
                "agent_a": {
                    "model_id": ext_a.proposed_model_id,
                    "visual_backbone": ext_a.visual_backbone,
                    "millions_of_parameters": ext_a.millions_of_parameters,
                    "code_repository_url": ext_a.code_repository_url,
                    "thought_process": ext_a.thought_process.model_dump() if getattr(ext_a, "thought_process", None) else None,
                    "extracted_scores_count": len(ext_a.extracted_scores),
                    "scores": [
                        {
                            "dataset": s.dataset_name,
                            "organ": s.organ_or_task,
                            "dice": s.dice_score,
                            "prompt": s.prompt_type,
                            "table_ref": s.table_or_section_ref,
                            "is_baseline": s.is_baseline_comparison,
                            "derivation_rationale": getattr(s, "derivation_rationale", None),
                        }
                        for s in ext_a.extracted_scores
                    ],
                } if ext_a else None,
                "agent_b": {
                    "model_id": ext_b.proposed_model_id,
                    "visual_backbone": ext_b.visual_backbone,
                    "millions_of_parameters": ext_b.millions_of_parameters,
                    "code_repository_url": ext_b.code_repository_url,
                    "thought_process": ext_b.thought_process.model_dump() if getattr(ext_b, "thought_process", None) else None,
                    "extracted_scores_count": len(ext_b.extracted_scores),
                    "scores": [
                        {
                            "dataset": s.dataset_name,
                            "organ": s.organ_or_task,
                            "dice": s.dice_score,
                            "prompt": s.prompt_type,
                            "table_ref": s.table_or_section_ref,
                            "is_baseline": s.is_baseline_comparison,
                            "derivation_rationale": getattr(s, "derivation_rationale", None),
                        }
                        for s in ext_b.extracted_scores
                    ],
                } if is_dual and ext_b else None,
                "consensus": {
                    "concordance_rate": ar.concordance_rate,
                    "inter_model_dice_mae": ar.inter_model_dice_mae,
                    "concordant_count": ar.concordant_count,
                    "dispute_count": ar.dispute_count,
                    "unconfirmed_count": ar.unconfirmed_count,
                    "scores": [
                        {
                            "dataset": cs.canonical_dataset_name,
                            "organ": cs.organ_or_task,
                            "score_a": cs.score_a,
                            "score_b": cs.score_b,
                            "delta": cs.delta,
                            "status": cs.status.value,
                            "resolved_score": cs.resolved_score,
                        }
                        for cs in ar.consensus_scores
                    ],
                } if is_dual and ar and hasattr(ar, "consensus_scores") else None,
            }
            json_results["papers"].append(paper_entry)

        report_content = "\n".join(md_lines)
        run_logger.save_run_results(json_results, report_content)

        print("\n" + "=" * 80)
        print(f"📁 Run Directory:            {run_logger.run_dir}")
        print(f"📄 Detailed Markdown report: {run_logger.report_file}")
        print(f"📊 Structured JSON results:  {run_logger.results_file}")
        print(f"📂 Intermediate extractions: {run_logger.extractions_dir}")
        print(f"📝 Full execution log:       {run_logger.log_file}")
        print(f"🔗 Latest report pointer:    {LOGS_DIR / 'latest_batch_benchmark.md'}")
        print("=" * 80 + "\n")

    finally:
        run_logger.close()


if __name__ == "__main__":
    main()
