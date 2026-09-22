"""Batch benchmark runner: evaluates ExtractionAgent across multiple literature papers automatically."""

import argparse
import sys
import time
from tabulate import tabulate

from agent_gmmis.config import DEFAULT_DB_PATH
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.extraction.evaluator import ExtractionEvaluator
from agent_gmmis.extraction.extractor import ExtractionAgent
from agent_gmmis.extraction.paper_fetcher import PaperFetcher
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
        default=3,
        help="Number of available arXiv models to test automatically if --models is not specified (default: 3)",
    )
    parser.add_argument("--provider", type=str, default=None, help="LLM provider: gemini, ollama, mufasa, or mock")
    parser.add_argument("--model", type=str, default=None, help="LLM model name (default: gemini-3.5-flash-lite)")
    parser.add_argument("--compare-source", type=str, default="primary", choices=["primary", "best"])
    args = parser.parse_args()

    db = DatabaseManager(DEFAULT_DB_PATH)
    fetcher = PaperFetcher(db_manager=db)

    # Determine models to evaluate
    if args.models:
        target_model_ids = args.models
    else:
        # Find models with an arXiv link and primary scores
        all_models = db.get_all_models()
        target_model_ids = []
        for m in all_models:
            link = m.first_publication_link or m.last_publication_link or ""
            if "arxiv" in link.lower():
                scores = db.get_scores_for_model(m.model_id)
                if any(scores[d].get(args.compare_source) is not None for d in scores):
                    target_model_ids.append(m.model_id)
                    if len(target_model_ids) >= args.count:
                        break

    print("\n" + "=" * 75)
    print(f"🔬 RUNNING BATCH EXTRACTION BENCHMARK ({len(target_model_ids)} papers)")
    print(f"Models to evaluate: {', '.join(target_model_ids)}")
    print("=" * 75)

    llm_config = LLMConfig(
        provider=args.provider or "gemini",
        model=args.model or "gemini-3.5-flash-lite",
    )
    llm_client = LLMClient(config=llm_config)
    agent = ExtractionAgent(llm_client=llm_client)
    evaluator = ExtractionEvaluator(db_manager=db)

    summary_rows = []

    for idx, model_id in enumerate(target_model_ids, 1):
        print(f"\n[{idx}/{len(target_model_ids)}] Processing '{model_id}'...")
        t0 = time.time()
        try:
            pdf_path = fetcher.fetch_for_model(model_id)
            extracted = agent.process_pdf(pdf_path)
            report = evaluator.evaluate(extracted, ground_truth_id=model_id, compare_source=args.compare_source)
            elapsed = time.time() - t0

            summary_rows.append([
                model_id,
                "✅ Pass",
                f"{report.matched_datasets_count}/{report.ground_truth_datasets_count}",
                f"{report.dataset_precision:.1%}",
                f"{report.dataset_recall:.1%}",
                f"{report.dice_mae:.4f}" if report.dice_mae is not None else "N/A",
                "✅ Match" if report.code_url_match else ("❌ Mismatch" if report.code_url_match is False else "N/A"),
                f"{elapsed:.1f}s",
            ])
            print(f"  -> Extracted {len(extracted.extracted_scores)} scores in {elapsed:.1f}s. Dice MAE: {report.dice_mae}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ❌ Error on {model_id}: {e}")
            summary_rows.append([
                model_id,
                f"❌ Error",
                "-",
                "-",
                "-",
                "-",
                "-",
                f"{elapsed:.1f}s",
            ])

    print("\n" + "=" * 75)
    print("📊 BATCH BENCHMARK AGGREGATED REPORT")
    print("=" * 75)
    headers = ["Model ID", "Status", "Datasets Matched", "Precision", "Recall", "Dice MAE", "Code Match", "Time"]
    print(tabulate(summary_rows, headers=headers, tablefmt="github"))
    print("\n✅ Batch extraction benchmark completed.\n")


if __name__ == "__main__":
    main()
