"""CLI tool to execute and evaluate the ExtractionAgent on a scientific manuscript."""

import argparse
import os
import sys
from pathlib import Path
from tabulate import tabulate

from agent_gmmis.config import DEFAULT_DB_PATH
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.extraction.evaluator import ExtractionEvaluator
from agent_gmmis.extraction.extractor import ExtractionAgent
from agent_gmmis.extraction.paper_fetcher import PaperFetcher
from agent_gmmis.llm.client import LLMClient, LLMConfig


def main():
    parser = argparse.ArgumentParser(description="Extract structured benchmark data from a medical segmentation paper.")
    parser.add_argument("pdf_path", type=str, nargs="?", default=None, help="Path to PDF manuscript (optional if --model-id or --arxiv is provided)")
    parser.add_argument("--model-id", type=str, default=None, help="Model ID from database.sqlite (automatically resolves arXiv link and downloads PDF)")
    parser.add_argument("--arxiv", type=str, default=None, help="arXiv ID or URL (e.g. 2408.00874) to download and extract automatically")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider: gemini, ollama, mufasa, or mock")
    parser.add_argument("--model", type=str, default=None, help="LLM model name (default: gemini-3.1-flash-lite)")
    parser.add_argument("--provider-a", type=str, default=None, help="LLM provider for Agent A (default: --provider or gemini)")
    parser.add_argument("--model-a", type=str, default=None, help="LLM model name for Agent A (default: --model or gemini-3.1-flash-lite)")
    parser.add_argument("--provider-b", type=str, default=None, help="LLM provider for Agent B: gemini, ollama, mufasa, or mock")
    parser.add_argument("--model-b", type=str, default=None, help="LLM model name for Agent B. If set, runs dual-agent parallel consensus mode.")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for OpenAI-compatible LLM endpoint")
    parser.add_argument("--base-url-a", type=str, default=None, help="Base URL for Agent A LLM endpoint")
    parser.add_argument("--base-url-b", type=str, default=None, help="Base URL for Agent B LLM endpoint")
    parser.add_argument("--api-key", type=str, default=None, help="API key (optional if set in environment or .env)")
    parser.add_argument("--evaluate", action="store_true", help="Compare extracted scores against database.sqlite ground truth")
    parser.add_argument(
        "--compare-source",
        type=str,
        default="primary",
        choices=["primary", "best"],
        help="Ground truth benchmark to compare against: 'primary' (paper's own report) or 'best' (subsequent literature best)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where intermediate extraction logs and artifacts will be saved (default: logs/runs/<timestamp>_single_<model_id>/)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag for the run directory name",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum deliberation rounds in dual-agent mode (default: 3; set to 1 to disable re-interrogation)",
    )
    parser.add_argument(
        "--no-save-logs",
        action="store_true",
        default=False,
        help="Disable automatic persistence of intermediate extraction logs to disk",
    )
    args = parser.parse_args()

    fetcher = PaperFetcher()
    if args.pdf_path:
        pdf_file = Path(args.pdf_path)
    elif args.model_id:
        pdf_file = fetcher.fetch_for_model(args.model_id)
    elif args.arxiv:
        pdf_file = fetcher.fetch_arxiv(args.arxiv)
    else:
        parser.error("You must provide either a pdf_path, --model-id <name>, or --arxiv <id>")

    if not pdf_file.exists():
        print(f"Error: File not found at {pdf_file}", file=sys.stderr)
        sys.exit(1)

    # Check API key configuration
    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    provider_a = args.provider_a or args.provider or os.getenv("LLM_PROVIDER", "gemini")
    model_a = args.model_a or args.model or os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
    provider_b = args.provider_b or provider_a
    model_b = args.model_b
    is_dual = model_b is not None
    base_url_a = args.base_url_a or args.base_url
    base_url_b = args.base_url_b or args.base_url

    local_providers = ("mock", "mufasa", "ollama")
    needs_key_a = provider_a not in local_providers
    needs_key_b = is_dual and (provider_b not in local_providers)
    if (needs_key_a or needs_key_b) and not api_key:
        print("\n" + "=" * 70)
        print("⚠️  MISSING API KEY")
        print("To run live extraction with an external provider, you need an API key.")
        print("1. Get a free key (0.00 €, no credit card needed) at: https://aistudio.google.com/app/apikey")
        print("2. Set it in your terminal:")
        print("   export GEMINI_API_KEY=\"AIzaSy...\"")
        print("   OR create a .env file with: GEMINI_API_KEY=\"AIzaSy...\"")
        print("   OR pass it with --api-key: python scripts/run_extraction.py paper.pdf --api-key \"AIzaSy...\"")
        print("=" * 70 + "\n")
        sys.exit(1)

    print(f"\n📄 Ingesting and parsing manuscript: {pdf_file.name}...")
    from agent_gmmis.extraction.pdf_parser import PDFParser
    manuscript = PDFParser.parse(pdf_file)

    deliberation_trajectory = []
    if is_dual:
        from agent_gmmis.auditing.auditor import AuditorAgent
        from agent_gmmis.extraction.dual_coordinator import DualExtractionCoordinator
        from agent_gmmis.extraction.schemas import ExtractedPaperData, RawExtractedScore
        db = DatabaseManager(DEFAULT_DB_PATH)
        auditor = AuditorAgent(db_manager=db)

        client_a = LLMClient(config=LLMConfig(provider=provider_a, model=model_a, api_key=api_key, base_url=base_url_a))
        agent_a = ExtractionAgent(llm_client=client_a)
        client_b = LLMClient(config=LLMConfig(provider=provider_b, model=model_b, api_key=api_key, base_url=base_url_b))
        agent_b = ExtractionAgent(llm_client=client_b)
        coordinator = DualExtractionCoordinator(agent_a=agent_a, agent_b=agent_b)
        print(f"🤖 Invoking DualExtractionCoordinator in parallel:")
        print(f"   • Agent A: {provider_a} ({model_a}) -> {base_url_a or 'default'}")
        print(f"   • Agent B: {provider_b} ({model_b}) -> {base_url_b or 'default'}")
        if args.max_rounds > 1:
            extracted_data_a, extracted_data_b, audit_report, deliberation_trajectory = coordinator.extract_manuscript_with_deliberation(
                manuscript=manuscript,
                auditor=auditor,
                max_rounds=args.max_rounds,
                benchmark_mode=bool(args.evaluate or args.model_id),
                verbose=True,
            )
        else:
            extracted_data_a, extracted_data_b = coordinator.extract_manuscript_parallel(manuscript)
            audit_report = auditor.audit_consensus(
                extracted_data_a,
                extracted_data_b,
                manuscript=manuscript,
                benchmark_mode=bool(args.evaluate or args.model_id),
            )
        extracted_data = extracted_data_a
    else:
        llm_config = LLMConfig(
            provider=provider_a,
            model=model_a,
            api_key=api_key,
            base_url=base_url_a,
        )
        llm_client = LLMClient(config=llm_config)
        agent = ExtractionAgent(llm_client=llm_client)
        print(f"🤖 Invoking ExtractionAgent with model: {llm_config.model} (provider: {llm_config.provider})...")
        extracted_data = agent.process_manuscript(manuscript)

    # Display Extracted Metadata
    print("\n" + "=" * 70)
    print("🎯 EXTRACTED MODEL METADATA")
    print("=" * 70)
    meta_table = [
        ["Model ID", extracted_data.proposed_model_id],
        ["Paper Title", extracted_data.paper_title],
        ["Affiliations", extracted_data.authors_affiliations or "N/A"],
        ["Architecture", extracted_data.architecture or "N/A"],
        ["Visual Backbone", extracted_data.visual_backbone or "N/A"],
        ["Parameters (M)", extracted_data.millions_of_parameters or "N/A"],
        ["Complexity (GFLOPs)", extracted_data.number_of_gflops or "N/A"],
        ["Hardware Resources", extracted_data.resources or "N/A"],
        ["Total VRAM (GB)", extracted_data.resources_total_vram_gb or "N/A"],
        ["Code Repository", extracted_data.code_repository_url or "N/A"],
    ]
    print(tabulate(meta_table, headers=["Field", "Extracted Value"], tablefmt="github"))

    # Display Screening
    print("\n" + "=" * 70)
    print("🔬 INCLUSION / EXCLUSION SCREENING")
    print("=" * 70)
    screening = extracted_data.screening
    screen_table = [
        ["3D Volumetric Modality", "✅ Yes" if screening.is_3d else "❌ No"],
        ["Radiological (CT/MRI/PET)", "✅ Yes" if screening.is_radiological else "❌ No"],
        ["Public Benchmark Datasets", "✅ Yes" if screening.is_public_dataset else "❌ No"],
        ["Reports Dice Metric (DSC)", "✅ Yes" if screening.has_dice_metric else "❌ No"],
        ["FINAL ELIGIBILITY", "✅ INCLUDED" if screening.is_included else f"❌ REJECTED: {', '.join(screening.rejection_reasons)}"],
    ]
    print(tabulate(screen_table, headers=["Criterion", "Status"], tablefmt="github"))

    # Display Benchmark Scores
    print("\n" + "=" * 70)
    print(f"📊 EXTRACTED BENCHMARK TABLES ({len(extracted_data.extracted_scores)} scores found)")
    print("=" * 70)
    if extracted_data.extracted_scores:
        scores_table = []
        for s in extracted_data.extracted_scores:
            scores_table.append([
                s.dataset_name,
                s.organ_or_task,
                f"{s.dice_score:.4f}" if s.dice_score is not None else "N/A",
                s.prompt_type or "N/A",
                s.table_or_section_ref or "N/A",
                "Baseline" if s.is_baseline_comparison else "Proposed",
                s.evaluated_model_name or extracted_data.proposed_model_id,
            ])
        print(tabulate(scores_table, headers=["Dataset", "Organ / Task", "Dice (DSC)", "Prompt Setting", "Table Ref", "Type", "Model"], tablefmt="github"))
    else:
        print("No numerical evaluation scores extracted.")

    # -------------------------------------------------------------
    # Step 3.1: AuditorAgent ("Il Vigile" - Quality Assurance Critic / Arbiter)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("👮 AUDITOR AGENT ('IL VIGILE') QUALITY ASSURANCE CERTIFICATION")
    print("=" * 70)

    if not is_dual:
        from agent_gmmis.auditing.auditor import AuditorAgent
        db = DatabaseManager(DEFAULT_DB_PATH)
        auditor = AuditorAgent(db_manager=db)
        audit_report = auditor.audit(
            extracted_data,
            manuscript=manuscript,
            benchmark_mode=bool(args.evaluate or args.model_id),
        )
        # Display Consensus Matrix
        print("\n" + "=" * 70)
        print("🤝 DUAL-AGENT INTER-MODEL CONSENSUS MATRIX")
        print("=" * 70)
        cons_rows = [
            [
                cs.canonical_dataset_name,
                cs.organ_or_task or "N/A",
                f"{cs.score_a:.4f}" if cs.score_a is not None else "—",
                f"{cs.score_b:.4f}" if cs.score_b is not None else "—",
                f"{cs.delta:.4f}" if cs.delta is not None else "—",
                f"{cs.resolved_score:.4f}" if cs.resolved_score is not None else "DISPUTED",
                cs.status.value.upper(),
            ]
            for cs in audit_report.consensus_scores
        ]
        print(tabulate(cons_rows, headers=["Dataset", "Organ/Task", f"Score A ({model_a})", f"Score B ({model_b})", "Delta", "Resolved", "Status"], tablefmt="github"))
        mae_disp = f"{audit_report.inter_model_dice_mae:.4f}" if audit_report.inter_model_dice_mae is not None else "N/A"
        print(f"\nInter-Model Dice MAE: {mae_disp}")
        print(f"Concordance Rate: {audit_report.concordance_rate*100:.1f}% ({audit_report.concordant_count} concordant, {audit_report.dispute_count} disputed, {audit_report.unconfirmed_count} unconfirmed / {audit_report.total_unique_benchmarks} total)")

        # Reconcile scores for downstream evaluation
        reconciled_scores = []
        for cs in audit_report.consensus_scores:
            if cs.resolved_score is not None:
                reconciled_scores.append(
                    RawExtractedScore(
                        dataset_name=cs.canonical_dataset_name,
                        organ_or_task=cs.organ_or_task or "Overall",
                        dice_score=cs.resolved_score,
                        prompt_type=cs.prompt_type,
                        table_or_section_ref=cs.table_ref_a or cs.table_ref_b,
                    )
                )
        extracted_data = ExtractedPaperData(
            paper_title=extracted_data_a.paper_title,
            proposed_model_id=audit_report.consensus_metadata.resolved_model_id,
            visual_backbone=audit_report.consensus_metadata.resolved_visual_backbone,
            millions_of_parameters=audit_report.consensus_metadata.resolved_params_m,
            code_repository_url=audit_report.consensus_metadata.code_repository_url,
            screening=extracted_data_a.screening,
            extracted_scores=reconciled_scores,
        )
    else:
        audit_report = auditor.audit(
            extracted_data,
            manuscript=manuscript,
            benchmark_mode=bool(args.evaluate or args.model_id),
        )

    status_icon = (
        "✅"
        if audit_report.status.value == "verified"
        else ("⚠️" if audit_report.status.value == "flagged_warning" else "❌")
    )
    print(f"\nAudit Status: {status_icon} {audit_report.status.value.upper()}")
    print(f"Confidence Score: {audit_report.confidence_score * 100:.1f}%")
    print(f"Passed Sanity Checks: {'✅ Yes' if audit_report.passed_sanity_checks else '❌ No'}")
    print(f"Passed Inclusion Screening: {'✅ Yes' if audit_report.passed_inclusion_screening else '❌ No'}")

    if audit_report.rejection_reasons:
        print(f"\n🚨 Rejection Reasons: {'; '.join(audit_report.rejection_reasons)}")

    if audit_report.anomalies:
        print(f"\n⚠️  Architectural / Parameter Anomalies ({len(audit_report.anomalies)}):")
        anomaly_table = [[a.field, a.severity.upper(), a.message] for a in audit_report.anomalies]
        print(tabulate(anomaly_table, headers=["Field", "Severity", "Message"], tablefmt="github"))

    if audit_report.outliers:
        print(f"\n📈 Statistical Baseline Outliers ({len(audit_report.outliers)}):")
        outlier_table = [
            [
                o.dataset_name,
                f"{o.extracted_score:.4f}",
                f"{o.historical_mean:.4f} (max={o.historical_max:.4f})",
                o.reason,
            ]
            for o in audit_report.outliers
        ]
        print(tabulate(outlier_table, headers=["Dataset", "Extracted", "Historical Benchmark", "Reason"], tablefmt="github"))

    if audit_report.discrepancies:
        print(f"\n🔍 Human Survey Discrepancies Intercepted ({len(audit_report.discrepancies)}):")
        disc_table = [
            [
                d.dataset_name,
                f"{d.extracted_score:.4f}",
                f"{d.survey_ground_truth:.4f}",
                f"{d.delta:+.4f}",
                d.message,
            ]
            for d in audit_report.discrepancies
        ]
        print(tabulate(disc_table, headers=["Dataset", "Paper Extracted", "Survey Ground Truth", "Delta", "Notice"], tablefmt="github"))

    if audit_report.data_diagnosis.value != "none":
        print(f"\n🩺 Missing Data Diagnosis: {audit_report.data_diagnosis.value}")

    # -------------------------------------------------------------
    # Step 3.2: Delegation Layer (RepoInspector)
    # -------------------------------------------------------------
    from agent_gmmis.auditing.schemas import AuditStatus
    if audit_report.status == AuditStatus.NEEDS_REPO_INSPECTION and extracted_data.code_repository_url:
        print("\n" + "=" * 70)
        print("🔍 DELEGATION LAYER — GITHUB REPO INSPECTOR")
        print("=" * 70)
        print(f"Inspecting code repository: {extracted_data.code_repository_url}")
        from agent_gmmis.auditing.repo_inspector import RepoInspector
        repo_inspector = RepoInspector()
        extracted_data, reaudited_report = repo_inspector.inspect_and_update(extracted_data, auditor=auditor)
        if reaudited_report:
            audit_report = reaudited_report
            print(f"✨ Successfully recovered scores from code repository!")
            print(f"Updated Audit Status: {audit_report.status.value.upper()} (Confidence: {audit_report.confidence_score * 100:.1f}%)")
            print(f"Total Scores Now Available: {len(extracted_data.extracted_scores)}")

    # Evaluation against Ground Truth (if requested or available)
    gt_model = db.get_model(extracted_data.proposed_model_id)

    if args.evaluate or gt_model:
        print("\n" + "=" * 70)
        print(f"⚖️  EVALUATION AGAINST DATABASE GROUND TRUTH ({args.compare_source.upper()})")
        print("=" * 70)
        if gt_model:
            evaluator = ExtractionEvaluator(db_manager=db)
            report = evaluator.evaluate(extracted_data, compare_source=args.compare_source)
            print(report.summary())
            if report.score_comparisons:
                print(f"\nDetailed Score Comparisons against {args.compare_source.capitalize()} Report:")
                comp_table = [
                    [c["dataset"], c.get("prompt_type", "N/A"), f"{c['predicted']:.4f}", f"{c['ground_truth']:.4f}", f"{c['diff']:.4f}"]
                    for c in report.score_comparisons
                ]
                print(tabulate(comp_table, headers=["Dataset", "Prompt Setting", "Extracted", f"GT ({args.compare_source.capitalize()})", "Diff (Absolute)"], tablefmt="github"))
        else:
            print(f"Notice: Model '{extracted_data.proposed_model_id}' is not yet in database.sqlite (new discovery!).")

    # -------------------------------------------------------------
    # Step 3.3: Persist Intermediate Agent Artifacts
    # -------------------------------------------------------------
    if not args.no_save_logs:
        from agent_gmmis.extraction.run_logger import RunLogger
        model_tag = extracted_data.proposed_model_id or pdf_file.stem
        tag = args.tag or f"single_{model_tag}"
        run_logger = RunLogger(
            base_dir=Path(args.output_dir) if args.output_dir else None,
            tag=tag,
            is_dual=is_dual,
            capture_stdout=False,
            metadata={
                "pdf_name": pdf_file.name,
                "provider_a": provider_a,
                "model_a": model_a,
                "provider_b": provider_b if is_dual else None,
                "model_b": model_b if is_dual else None,
                "compare_source": args.compare_source,
                "max_rounds": args.max_rounds,
            },
        )
        paper_dir = run_logger.log_paper_extraction(
            model_id=extracted_data.proposed_model_id or pdf_file.stem,
            extracted_a=extracted_data_a if is_dual else extracted_data,
            extracted_b=extracted_data_b if is_dual else None,
            audit_report=audit_report,
            final_extracted=extracted_data,
            eval_report=report if (args.evaluate or gt_model) and 'report' in locals() else None,
            repo_inspector_result=repo_inspector if 'reaudited_report' in locals() and reaudited_report else None,
            deliberation_trajectory=deliberation_trajectory,
        )
        print("\n" + "=" * 70)
        print("📁 INTERMEDIATE ARTIFACTS & LOGS PERSISTED")
        print("=" * 70)
        print(f"Directory: {paper_dir}")
        print(f"• Agent A:       {paper_dir / 'agent_a_extracted.json'}")
        if is_dual:
            print(f"• Agent B:       {paper_dir / 'agent_b_extracted.json'}")
            print(f"• Consensus:     {paper_dir / 'consensus_matrix.json'}")
            if deliberation_trajectory:
                print(f"• Deliberation:  {paper_dir / 'deliberation_trajectory.json'} ({len(deliberation_trajectory)} rounds)")
        print(f"• Audit Report:  {paper_dir / 'audit_report.json'}")
        print(f"• Reconciled:    {paper_dir / 'final_reconciled.json'}")
        if (args.evaluate or gt_model) and 'report' in locals():
            print(f"• Evaluation:    {paper_dir / 'evaluation.json'}")

    print("\n✅ Extraction and validation workflow completed.\n")


if __name__ == "__main__":
    main()
