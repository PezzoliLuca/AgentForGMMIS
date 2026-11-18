#!/usr/bin/env python3
"""
Standalone Calibration & Evaluation Benchmark for ScoutAgent (Stage 2 + Stage 3).
Evaluates Precision, Recall, Specificity, F1-Score, and Tier Classification Accuracy
against the 40-abstract Gold Standard (`data/benchmarks/scout_abstract_benchmark.json`).
"""

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_gmmis.config import DATA_DIR, DEFAULT_DB_PATH, LOGS_DIR
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.llm.client import LLMClient, LLMConfig
from agent_gmmis.scouting.schemas import RawPaperCandidate, ScoutTier
from agent_gmmis.scouting.scout_agent import ScoutAgent


DEFAULT_BENCHMARK_PATH = DATA_DIR / "benchmarks" / "scout_abstract_benchmark.json"


def run_evaluation(
    provider: str = "gemini",
    model: str = "gemini-3.5-flash-lite",
    base_url: str | None = None,
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    count: int | None = None,
) -> dict:
    raw_items = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if count is not None and count > 0:
        half = max(1, count // 2)
        positives = [x for x in raw_items if x["expected_included"]][:half]
        negatives = [x for x in raw_items if not x["expected_included"]][: count - len(positives)]
        raw_items = positives + negatives

    db = DatabaseManager(DEFAULT_DB_PATH)
    # Mask all existing DB models so the 20 True Positives from the survey aren't skipped as ALREADY_IN_DB!
    all_db_ids = {m.id for m in db.get_all_models()}

    llm_cfg = LLMConfig(provider=provider, model=model, base_url=base_url)
    llm_client = LLMClient(llm_cfg)
    scout = ScoutAgent(
        llm_client=llm_client,
        db_manager=db,
        masked_model_ids=all_db_ids,
        use_cache=True,
    )

    print(f"\n🔎 [ScoutAgent Gold-Standard Benchmark] Evaluating {len(raw_items)} abstracts")
    print(f"   Engine: provider='{llm_client.config.provider}', model='{llm_client.config.model}' (mock={llm_client.is_mock})\n")

    tp = fp = tn = fn = 0
    llm_tp = llm_fp = llm_tn = llm_fn = 0
    tier_correct = 0
    tier_total_tp = 0
    stage2_lexical_catches = 0
    details: list[dict] = []

    t0 = time.time()
    for idx, item in enumerate(raw_items, 1):
        cand = RawPaperCandidate(
            title=item["title"],
            abstract=item["abstract"],
            venue=item.get("venue"),
            publication_date=item.get("publication_date"),
            arxiv_id=item.get("arxiv_id"),
            doi=item.get("doi"),
        )
        expected_inc = bool(item["expected_included"])
        expected_tier = item["expected_tier"]

        # 1. Stage 2 Lexical Pre-Filter check
        passed_lex, lex_reason = scout.deduplicator.apply_lexical_prefilter(cand)
        if not passed_lex:
            stage2_lexical_catches += 1

        # 2. Always evaluate Stage 3 LLM so we can compare pure LLM capability across all 40 abstracts
        screened = scout.screen_candidate_stage3(cand)
        llm_tier = screened.final_tier.value
        llm_inc = screened.final_tier in (ScoutTier.TIER_1_GENERALIST, ScoutTier.TIER_2_BENCHMARK_SOTA)
        llm_reason = screened.rejection_reason or (screened.screening.reasoning if screened.screening else "")

        # Pure LLM confusion matrix
        if expected_inc and llm_inc:
            llm_tp += 1
        elif expected_inc and not llm_inc:
            llm_fn += 1
        elif not expected_inc and llm_inc:
            llm_fp += 1
        else:
            llm_tn += 1

        # Full Funnel (Stage 2 + Stage 3) prediction
        if not passed_lex:
            pred_inc = False
            pred_tier = ScoutTier.REJECTED.value
            reason = f"[Stage2+LLM] {lex_reason} | LLM: {llm_tier}"
        else:
            pred_inc = llm_inc
            pred_tier = llm_tier
            reason = llm_reason

        is_correct = pred_inc == expected_inc
        if expected_inc and pred_inc:
            tp += 1
            tier_total_tp += 1
            if pred_tier == expected_tier:
                tier_correct += 1
        elif expected_inc and not pred_inc:
            fn += 1
        elif not expected_inc and llm_inc and passed_lex:
            fp += 1
        else:
            tn += 1

        llm_icon = "✅" if (llm_inc == expected_inc) else "❌"
        tier_icon = "🎯" if (llm_tier == expected_tier) else "⚠️"
        s2_tag = "[S2-Lex]" if not passed_lex else "        "
        print(
            f"[{idx:02d}/{len(raw_items):02d}] {llm_icon}{tier_icon} {s2_tag} {item['id']:<28} | "
            f"Exp: {expected_tier:<21} | LLM: {llm_tier:<21}"
        )
        details.append(
            {
                "id": item["id"],
                "title": item["title"],
                "expected_included": expected_inc,
                "stage2_passed": passed_lex,
                "llm_predicted_included": llm_inc,
                "funnel_predicted_included": pred_inc,
                "expected_tier": expected_tier,
                "llm_predicted_tier": llm_tier,
                "funnel_predicted_tier": pred_tier,
                "correct_inclusion_llm": llm_inc == expected_inc,
                "correct_inclusion_funnel": is_correct,
                "correct_tier": llm_tier == expected_tier,
                "reason": reason,
            }
        )

    elapsed = time.time() - t0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(raw_items) if raw_items else 0.0
    tier_acc = tier_correct / tier_total_tp if tier_total_tp > 0 else 0.0

    llm_precision = llm_tp / (llm_tp + llm_fp) if (llm_tp + llm_fp) > 0 else 0.0
    llm_recall = llm_tp / (llm_tp + llm_fn) if (llm_tp + llm_fn) > 0 else 0.0
    llm_specificity = llm_tn / (llm_tn + llm_fp) if (llm_tn + llm_fp) > 0 else 0.0
    llm_f1 = (2 * llm_precision * llm_recall / (llm_precision + llm_recall)) if (llm_precision + llm_recall) > 0 else 0.0

    summary = {
        "total_samples": len(raw_items),
        "funnel_metrics": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "overall_accuracy": round(accuracy, 4),
        },
        "pure_llm_metrics": {
            "tp": llm_tp,
            "tn": llm_tn,
            "fp": llm_fp,
            "fn": llm_fn,
            "precision": round(llm_precision, 4),
            "recall": round(llm_recall, 4),
            "specificity": round(llm_specificity, 4),
            "f1_score": round(llm_f1, 4),
        },
        "stage2_lexical_catches": stage2_lexical_catches,
        "tier_classification_accuracy": round(tier_acc, 4),
        "elapsed_seconds": round(elapsed, 2),
        "provider": llm_client.config.provider,
        "model": llm_client.config.model,
        "details": details,
    }

    print("\n" + "=" * 76)
    print(f"📊 SCOUT AGENT 40-ABSTRACT BENCHMARK SUMMARY ({llm_client.config.provider} / {llm_client.config.model})")
    print("=" * 76)
    print(f"  [Pure Stage 3 LLM Only]  TP={llm_tp}, TN={llm_tn}, FP={llm_fp}, FN={llm_fn}")
    print(f"    • LLM Recall (Sens.)      : {llm_recall * 100:.2f}%")
    print(f"    • LLM Precision           : {llm_precision * 100:.2f}%")
    print(f"    • LLM Specificity         : {llm_specificity * 100:.2f}%")
    print(f"    • LLM F1-Score            : {llm_f1 * 100:.2f}%")
    print("-" * 76)
    print(f"  [Full Funnel S2 + S3]    TP={tp}, TN={tn}, FP={fp}, FN={fn} (S2 caught {stage2_lexical_catches}/20 TN)")
    print(f"    • Funnel Recall (Sens.)   : {recall * 100:.2f}%")
    print(f"    • Funnel Precision        : {precision * 100:.2f}%")
    print(f"    • Funnel Specificity      : {specificity * 100:.2f}%")
    print(f"    • Funnel F1-Score         : {f1 * 100:.2f}%")
    print(f"    • Tier 1 vs Tier 2 Acc.   : {tier_acc * 100:.2f}% ({tier_correct}/{tier_total_tp})")
    print(f"    • Total Runtime           : {elapsed:.2f}s ({elapsed/max(1, len(raw_items)):.2f}s/abstract)")
    print("=" * 76 + "\n")

    out_dir = LOGS_DIR / "scout_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = llm_client.config.model.replace("/", "_").replace(":", "_")
    model_file = out_dir / f"eval_40_{llm_client.config.provider}_{safe_model}.json"
    latest_file = out_dir / "latest_scout_benchmark_eval.json"
    payload_str = json.dumps(summary, indent=2, ensure_ascii=False)
    model_file.write_text(payload_str, encoding="utf-8")
    latest_file.write_text(payload_str, encoding="utf-8")
    print(f"💾 Saved evaluation report to: {model_file}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ScoutAgent on 40-abstract Gold Standard benchmark.")
    parser.add_argument("--provider", default="gemini", help="LLM provider (gemini, ollama, mufasa, mock)")
    parser.add_argument("--model", default="gemini-3.5-flash-lite", help="LLM model identifier")
    parser.add_argument("--base-url", default=None, help="Optional custom OpenAI-compatible base URL")
    parser.add_argument("--count", type=int, default=None, help="Optional subset size for quick smoke testing")
    args = parser.parse_args()

    run_evaluation(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        count=args.count,
    )


if __name__ == "__main__":
    main()

