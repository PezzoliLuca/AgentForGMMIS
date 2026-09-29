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
    parser.add_argument("--model", type=str, default=None, help="LLM model name (default: gemini-3.5-flash-lite)")
    parser.add_argument("--api-key", type=str, default=None, help="API key (optional if set in environment or .env)")
    parser.add_argument("--evaluate", action="store_true", help="Compare extracted scores against database.sqlite ground truth")
    parser.add_argument(
        "--compare-source",
        type=str,
        default="primary",
        choices=["primary", "best"],
        help="Ground truth benchmark to compare against: 'primary' (paper's own report) or 'best' (subsequent literature best)",
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
    provider = args.provider or os.getenv("LLM_PROVIDER", "gemini")

    if provider != "mock" and not api_key:
        print("\n" + "=" * 70)
        print("⚠️  MISSING API KEY")
        print("To run live extraction, you need a free Google Gemini API key.")
        print("1. Get a free key (0.00 €, no credit card needed) at: https://aistudio.google.com/app/apikey")
        print("2. Set it in your terminal:")
        print("   export GEMINI_API_KEY=\"AIzaSy...\"")
        print("   OR create a .env file with: GEMINI_API_KEY=\"AIzaSy...\"")
        print("   OR pass it with --api-key: python scripts/run_extraction.py paper.pdf --api-key \"AIzaSy...\"")
        print("=" * 70 + "\n")
        sys.exit(1)

    print(f"\n📄 Ingesting and parsing manuscript: {pdf_file.name}...")
    llm_config = LLMConfig(
        provider=provider,
        model=args.model or os.getenv("LLM_MODEL", "gemini-3.5-flash-lite"),
        api_key=api_key,
    )
    llm_client = LLMClient(config=llm_config)
    agent = ExtractionAgent(llm_client=llm_client)

    print(f"🤖 Invoking ExtractionAgent with model: {llm_config.model} (provider: {llm_config.provider})...")
    extracted_data = agent.process_pdf(pdf_file)

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
                f"{s.dice_score:.4f}",
                s.prompt_type or "N/A",
                s.table_or_section_ref or "N/A",
                "Baseline" if s.is_baseline_comparison else "Proposed",
                s.evaluated_model_name or extracted_data.proposed_model_id,
            ])
        print(tabulate(scores_table, headers=["Dataset", "Organ / Task", "Dice (DSC)", "Prompt Setting", "Table Ref", "Type", "Model"], tablefmt="github"))
    else:
        print("No numerical evaluation scores extracted.")

    # -------------------------------------------------------------
    # Step 3.1: AuditorAgent ("Il Vigile" - Quality Assurance Critic)
    # -------------------------------------------------------------
    from agent_gmmis.auditing.auditor import AuditorAgent
    from agent_gmmis.extraction.pdf_parser import PDFParser

    print("\n" + "=" * 70)
    print("👮 AUDITOR AGENT ('IL VIGILE') QUALITY ASSURANCE CERTIFICATION")
    print("=" * 70)
    db = DatabaseManager(DEFAULT_DB_PATH)
    auditor = AuditorAgent(db_manager=db)
    manuscript = PDFParser.parse(pdf_file)
    audit_report = auditor.audit(
        extracted_data,
        manuscript=manuscript,
        benchmark_mode=bool(args.ground_truth_id),
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

    print("\n✅ Extraction and validation workflow completed.\n")


if __name__ == "__main__":
    main()
