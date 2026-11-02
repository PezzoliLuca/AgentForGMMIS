"""Benchmark Run Discovery, Deep Log Analysis, and Cross-Run Matrix Comparator.

Parses structured run artifacts, computes statistical KPIs across single/dual benchmark runs,
aggregates consensus and curation telemetry, and exports publication-ready LaTeX & Markdown.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import re
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from agent_gmmis.config import LOGS_DIR
from agent_gmmis.extraction.run_logger import sanitize_filename


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely coerces strings with %, 'N/A', or numeric types to float."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace("%", "").replace("s", "").strip()
        if not cleaned or cleaned.upper() in ("N/A", "NONE", "NULL", "-"):
            return default
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def discover_runs(
    base_dirs: Optional[List[Union[str, Path]]] = None,
) -> List[Dict[str, Any]]:
    """
    Scans project log repositories for benchmark run directories.

    Scans logs/runs/, logs/mufasa_runs/, and legacy flat run folders.
    Deduplicates identical run_ids, prioritizing runs containing complete extraction artifacts.
    """
    if base_dirs is None:
        proj_root = Path(__file__).resolve().parent.parent.parent.parent
        logs_root = proj_root / "logs"
        base_dirs = [
            logs_root / "runs",
            logs_root / "mufasa_runs",
            logs_root,
        ]

    runs_by_id: Dict[str, Dict[str, Any]] = {}

    for b_dir in base_dirs:
        b_path = Path(b_dir)
        if not b_path.exists():
            continue

        # Look for subdirectories with benchmark_results.json or run_manifest.json
        candidate_dirs = []
        for p in b_path.iterdir():
            if p.is_dir() and not p.name.startswith((".", "archive", "stage_cache")):
                candidate_dirs.append(p)

        for c_dir in candidate_dirs:
            results_file = c_dir / "benchmark_results.json"
            if not results_file.exists():
                batch_candidates = list(c_dir.glob("batch_benchmark_results_*.json"))
                results_file = batch_candidates[0] if batch_candidates else None

            manifest_file = c_dir / "run_manifest.json"

            if not results_file and not manifest_file:
                continue

            # Read metadata
            meta: Dict[str, Any] = {}
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass

            results_data: Dict[str, Any] = {}
            if results_file and results_file.exists():
                try:
                    with open(results_file, "r", encoding="utf-8") as f:
                        results_data = json.load(f)
                except Exception:
                    pass

            inner_meta = results_data.get("metadata", {})
            manifest_meta = meta.get("metadata", {})

            run_id = (
                inner_meta.get("run_id")
                or meta.get("run_id")
                or c_dir.name
            )

            timestamp = (
                inner_meta.get("timestamp")
                or meta.get("created_at")
                or ""
            )
            if not timestamp:
                m = re.match(r"^(\d{8}_\d{6})", c_dir.name)
                if m:
                    raw_ts = m.group(1)
                    timestamp = f"{raw_ts[:4]}-{raw_ts[4:6]}-{raw_ts[6:8]} {raw_ts[9:11]}:{raw_ts[11:13]}:{raw_ts[13:15]}"

            is_dual = inner_meta.get("is_dual_mode", manifest_meta.get("is_dual", meta.get("is_dual", False)))
            provider_a = inner_meta.get("provider_a") or manifest_meta.get("provider_a", "unknown")
            model_a = inner_meta.get("model_a") or manifest_meta.get("model_a", "unknown")
            provider_b = inner_meta.get("provider_b") or manifest_meta.get("provider_b")
            model_b = inner_meta.get("model_b") or manifest_meta.get("model_b")
            compare_source = inner_meta.get("compare_source") or manifest_meta.get("compare_source", "primary")

            global_metrics = results_data.get("global_metrics", {})
            total_papers = (
                global_metrics.get("total_target_papers")
                or inner_meta.get("total_papers")
                or len(results_data.get("summary", []))
            )
            elapsed_sec = global_metrics.get("total_elapsed_seconds", 0.0)
            duration_min = round(elapsed_sec / 60.0, 1) if elapsed_sec else 0.0

            extractions_dir = c_dir / "extractions"
            has_extractions = extractions_dir.exists() and any(extractions_dir.iterdir())
            num_extractions = len(list(extractions_dir.iterdir())) if has_extractions else 0

            run_entry = {
                "run_id": run_id,
                "folder_name": c_dir.name,
                "run_dir": str(c_dir.resolve()),
                "timestamp": timestamp,
                "provider_a": provider_a,
                "model_a": model_a,
                "provider_b": provider_b,
                "model_b": model_b,
                "is_dual": is_dual,
                "compare_source": compare_source,
                "total_papers": total_papers,
                "duration_min": duration_min,
                "has_extractions": has_extractions,
                "num_extractions": num_extractions,
            }

            # Deduplication: prefer run with extractions or more extractions
            if run_id not in runs_by_id:
                runs_by_id[run_id] = run_entry
            else:
                existing = runs_by_id[run_id]
                if run_entry["num_extractions"] > existing["num_extractions"]:
                    runs_by_id[run_id] = run_entry

    runs_list = list(runs_by_id.values())
    runs_list.sort(key=lambda r: (r["timestamp"] or "", r["run_id"]), reverse=True)
    return runs_list


def load_run_analysis(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Conducts comprehensive statistical and artifact analysis of a single benchmark run.

    Computes macro/median/micro Dice MAE, coverage scores, consensus concordance rates,
    audit status distribution, and curation impacts.
    """
    r_path = Path(run_dir)
    results_file = r_path / "benchmark_results.json"
    if not results_file.exists():
        candidates = list(r_path.glob("batch_benchmark_results_*.json"))
        results_file = candidates[0] if candidates else None

    results_data: Dict[str, Any] = {}
    if results_file and results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                results_data = json.load(f)
        except Exception:
            pass

    metadata = results_data.get("metadata", {})
    global_m = results_data.get("global_metrics", {})
    summary_rows = results_data.get("summary", [])

    # Per-Paper summary processing
    papers: List[Dict[str, Any]] = []
    dice_maes: List[float] = []
    recalls: List[float] = []
    confidences: List[float] = []
    elapsed_times: List[float] = []
    audit_counts: Dict[str, int] = {
        "VERIFIED": 0,
        "FLAGGED_WARNING": 0,
        "REJECTED": 0,
        "NEEDS_OUTREACH": 0,
        "NEEDS_REPO_INSPECTION": 0,
        "OTHER": 0,
    }

    for row in summary_rows:
        model_id = row.get("model_id", "Unknown")
        status_raw = str(row.get("audit_status", ""))
        clean_status = "OTHER"
        if "VERIFIED" in status_raw.upper():
            clean_status = "VERIFIED"
        elif "FLAGGED" in status_raw.upper() or "WARNING" in status_raw.upper():
            clean_status = "FLAGGED_WARNING"
        elif "OUTREACH" in status_raw.upper():
            clean_status = "NEEDS_OUTREACH"
        elif "REPO" in status_raw.upper():
            clean_status = "NEEDS_REPO_INSPECTION"
        elif "REJECT" in status_raw.upper():
            clean_status = "REJECTED"

        audit_counts[clean_status] = audit_counts.get(clean_status, 0) + 1

        mae = _safe_float(row.get("dice_mae"))
        rec = _safe_float(row.get("recall"))
        conf = _safe_float(row.get("confidence"))
        elap = _safe_float(row.get("elapsed_time"))

        if mae is not None:
            dice_maes.append(mae)
        if rec is not None:
            recalls.append(rec)
        if conf is not None:
            confidences.append(conf)
        if elap is not None:
            elapsed_times.append(elap)

        papers.append({
            "model_id": model_id,
            "audit_status": clean_status,
            "raw_status": status_raw,
            "confidence": conf if conf is not None else 0.0,
            "matched_datasets": row.get("matched_datasets", "0/0"),
            "recall": rec if rec is not None else 0.0,
            "dice_mae": mae if mae is not None else 0.0,
            "elapsed_seconds": elap if elap is not None else 0.0,
        })

    # Global KPI Calculations
    macro_mae = (
        _safe_float(global_m.get("macro_mae_active"))
        or _safe_float(global_m.get("macro_mae_primary"))
        or (statistics.mean(dice_maes) if dice_maes else 0.0)
    )
    median_mae = statistics.median(dice_maes) if dice_maes else 0.0
    micro_mae = (
        _safe_float(global_m.get("micro_mae_active"))
        or _safe_float(global_m.get("micro_mae_primary"))
        or macro_mae
    )
    overall_recall = (
        _safe_float(global_m.get("overall_recall"))
        if global_m.get("overall_recall") is not None
        else (statistics.mean(recalls) / 100.0 if recalls else 0.0)
    )
    if overall_recall is not None and overall_recall > 1.0:
        overall_recall = overall_recall / 100.0

    cov_score = (
        _safe_float(global_m.get("coverage_adjusted_score"))
        or (overall_recall * (1.0 - macro_mae) if (overall_recall and macro_mae is not None) else 0.0)
    )

    zero_error_count = sum(1 for m in dice_maes if m == 0.0)
    near_zero_count = sum(1 for m in dice_maes if m < 0.005)

    # Consensus and Deliberation Telemetry Aggregation
    extractions_dir = r_path / "extractions"
    paired_concordance_rates: List[float] = []
    total_disputes = 0
    total_unconfirmed = 0
    round_distribution: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    paper_concordance_map: Dict[str, Dict[str, Any]] = {}

    if extractions_dir.exists():
        for paper_dir in extractions_dir.iterdir():
            if not paper_dir.is_dir():
                continue

            matrix_file = paper_dir / "consensus_matrix.json"
            delib_file = paper_dir / "deliberation_trajectory.json"

            p_concordance = None
            p_disputes = 0
            p_unconf = 0

            if matrix_file.exists():
                try:
                    with open(matrix_file, "r", encoding="utf-8") as f:
                        c_data = json.load(f)
                    p_concordance = _safe_float(c_data.get("concordance_rate"))
                    if p_concordance is not None:
                        if p_concordance <= 1.0:
                            p_concordance *= 100.0
                        paired_concordance_rates.append(p_concordance)
                    p_disputes = c_data.get("dispute_count", 0)
                    p_unconf = c_data.get("unconfirmed_count", 0)
                    total_disputes += p_disputes
                    total_unconfirmed += p_unconf
                except Exception:
                    pass

            rounds_ran = 1
            if delib_file.exists():
                try:
                    with open(delib_file, "r", encoding="utf-8") as f:
                        traj = json.load(f)
                    if isinstance(traj, list) and len(traj) > 0:
                        rounds_ran = max(item.get("round_number", 1) for item in traj)
                except Exception:
                    pass

            round_distribution[rounds_ran] = round_distribution.get(rounds_ran, 0) + 1
            paper_concordance_map[paper_dir.name] = {
                "concordance_rate": p_concordance,
                "disputes": p_disputes,
                "unconfirmed": p_unconf,
                "deliberation_rounds": rounds_ran,
            }

    # Enrich per-paper records with consensus telemetry
    for p in papers:
        s_name = sanitize_filename(p["model_id"])
        c_info = paper_concordance_map.get(s_name) or paper_concordance_map.get(p["model_id"]) or {}
        p["concordance_rate"] = c_info.get("concordance_rate")
        p["disputes"] = c_info.get("disputes", 0)
        p["unconfirmed"] = c_info.get("unconfirmed", 0)
        p["deliberation_rounds"] = c_info.get("deliberation_rounds", 1)

    avg_concordance = statistics.mean(paired_concordance_rates) if paired_concordance_rates else None

    # Database Curation Telemetry Aggregation
    curation_sql_file = r_path / "database_curation.sql"
    curation_plan_file = r_path / "curation_plan.json"
    novel_columns: Set[str] = set()
    alter_table_count = 0
    curation_inserts = 0
    curation_updates = 0
    peak_gains = 0
    preserved_peaks = 0
    blocked_disputes = 0

    if curation_sql_file.exists():
        try:
            sql_text = curation_sql_file.read_text(encoding="utf-8")
            for line in sql_text.splitlines():
                if "ALTER TABLE" in line.upper() and "ADD COLUMN" in line.upper():
                    alter_table_count += 1
                    match = re.search(r"ADD\s+COLUMN\s+([a-zA-Z0-9_]+)", line, re.IGNORECASE)
                    if match:
                        novel_columns.add(match.group(1))
                elif "INSERT OR REPLACE INTO models" in line:
                    curation_inserts += 1
                elif "UPDATE models SET" in line:
                    curation_updates += 1
        except Exception:
            pass

    if curation_plan_file.exists():
        try:
            with open(curation_plan_file, "r", encoding="utf-8") as f:
                c_plan = json.load(f)
            peak_gains = c_plan.get("peak_gains_count", 0)
            preserved_peaks = c_plan.get("preserved_peaks_count", 0)
            blocked_disputes = c_plan.get("blocked_disputes_count", 0)
        except Exception:
            pass

    return {
        "run_id": metadata.get("run_id") or r_path.name,
        "run_dir": str(r_path.resolve()),
        "timestamp": metadata.get("timestamp", ""),
        "metadata": metadata,
        "kpis": {
            "overall_recall": round((overall_recall or 0.0) * 100.0, 1),
            "macro_mae": round(macro_mae, 4),
            "median_mae": round(median_mae, 4),
            "micro_mae": round(micro_mae, 4),
            "macro_mae_best": round(_safe_float(global_m.get("macro_mae_best"), 0.0) or 0.0, 4),
            "micro_mae_best": round(_safe_float(global_m.get("micro_mae_best"), 0.0) or 0.0, 4),
            "micro_mae_regime_agnostic": round(_safe_float(global_m.get("micro_mae_regime_agnostic"), 0.0) or 0.0, 4),
            "coverage_adjusted_score": round(cov_score, 4),
            "zero_error_count": zero_error_count,
            "near_zero_count": near_zero_count,
            "total_papers": len(papers),
            "total_matched_benchmarks": global_m.get("total_matched_benchmarks", 0),
            "total_gt_benchmarks": global_m.get("total_gt_benchmarks_active", 0),
            "avg_confidence": round(statistics.mean(confidences), 1) if confidences else 0.0,
            "avg_elapsed_seconds": round(statistics.mean(elapsed_times), 1) if elapsed_times else 0.0,
            "total_elapsed_seconds": round(global_m.get("total_elapsed_seconds", sum(elapsed_times)), 1),
            "audit_counts": audit_counts,
        },
        "deliberation": {
            "paired_concordance_rate": round(avg_concordance, 1) if avg_concordance is not None else None,
            "total_disputes": total_disputes,
            "total_unconfirmed": total_unconfirmed,
            "round_distribution": round_distribution,
        },
        "curation": {
            "inserted_models": curation_inserts,
            "updated_models": curation_updates,
            "peak_gains": peak_gains,
            "preserved_peaks": preserved_peaks,
            "blocked_disputes": blocked_disputes,
            "alter_table_count": alter_table_count,
            "novel_columns": sorted(list(novel_columns)),
        },
        "papers": papers,
    }


def load_paper_detail(run_dir: Union[str, Path], model_id: str) -> Dict[str, Any]:
    """
    Loads all intermediate agent artifacts for a specific paper from a benchmark run.
    """
    r_path = Path(run_dir)
    clean_id = sanitize_filename(model_id)
    target_dir = r_path / "extractions" / clean_id
    if not target_dir.exists():
        target_dir = r_path / "extractions" / model_id

    result: Dict[str, Any] = {
        "model_id": model_id,
        "found": target_dir.exists(),
        "folder": str(target_dir.resolve()) if target_dir.exists() else None,
        "agent_a_extracted": None,
        "agent_b_extracted": None,
        "consensus_matrix": None,
        "deliberation_trajectory": None,
        "audit_report": None,
        "evaluation": None,
        "curation_plan": None,
        "curation_sql": None,
        "outreach_markdown": None,
        "final_reconciled": None,
        "paper_summary": None,
    }

    if not target_dir.exists():
        return result

    file_mapping = {
        "agent_a_extracted": "agent_a_extracted.json",
        "agent_b_extracted": "agent_b_extracted.json",
        "consensus_matrix": "consensus_matrix.json",
        "deliberation_trajectory": "deliberation_trajectory.json",
        "audit_report": "audit_report.json",
        "evaluation": "evaluation.json",
        "curation_plan": "curation_plan.json",
        "final_reconciled": "final_reconciled.json",
        "paper_summary": "paper_summary.json",
    }

    for key, filename in file_mapping.items():
        fp = target_dir / filename
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    result[key] = json.load(f)
            except Exception:
                pass

    sql_fp = target_dir / "curation.sql"
    if sql_fp.exists():
        try:
            result["curation_sql"] = sql_fp.read_text(encoding="utf-8")
        except Exception:
            pass

    # Look for outreach drafts
    outreach_candidates = list(target_dir.glob("*inquiry*.md")) + list(target_dir.glob("*outreach*.md"))
    if outreach_candidates:
        try:
            result["outreach_markdown"] = outreach_candidates[0].read_text(encoding="utf-8")
        except Exception:
            pass

    return result


def compare_runs(
    run_dirs: List[Union[str, Path]],
    custom_labels: Optional[List[str]] = None,
    metric: str = "dice_mae",
) -> Dict[str, Any]:
    """
    Compares 2..N benchmark runs, assembling an aggregate KPI comparison table,
    a cross-run paper-by-paper matrix, and generating LaTeX / Markdown reports.
    """
    if not run_dirs:
        raise ValueError("At least one run directory must be provided for comparison.")

    analyses = [load_run_analysis(d) for d in run_dirs]

    labels = []
    for idx, (analysis, d) in enumerate(zip(analyses, run_dirs)):
        if custom_labels and idx < len(custom_labels) and custom_labels[idx].strip():
            labels.append(custom_labels[idx].strip())
        else:
            meta = analysis.get("metadata", {})
            m_a = (meta.get("model_a") or "").split("/")[-1]
            m_b = (meta.get("model_b") or "").split("/")[-1]
            if m_b:
                labels.append(f"{m_a} + {m_b}")
            elif m_a:
                labels.append(m_a)
            else:
                labels.append(Path(d).name)

    # 1. KPI Comparison DataFrame
    kpi_rows = [
        ("Dataset Recall (%)", "overall_recall", "%", True),
        ("Macro Dice MAE", "macro_mae", "", False),
        ("Median Dice MAE", "median_mae", "", False),
        ("Micro Dice MAE", "micro_mae", "", False),
        ("Coverage-Adjusted Score", "coverage_adjusted_score", "", True),
        ("Zero-Error Papers (MAE=0.0)", "zero_error_count", "", True),
        ("Near-Zero Papers (MAE<0.005)", "near_zero_count", "", True),
        ("Verified Papers", "verified_count", "", True),
        ("Flagged Papers", "flagged_count", "", False),
        ("Rejected / Outreach", "rejected_count", "", False),
        ("Avg Confidence (%)", "avg_confidence", "%", True),
        ("Concordance Rate (%)", "paired_concordance_rate", "%", True),
        ("Deliberation Disputes", "total_disputes", "", False),
        ("Total Target Papers", "total_papers", "", None),
        ("Avg Runtime / Paper (s)", "avg_elapsed_seconds", "s", False),
        ("Total Runtime (min)", "total_runtime_min", "min", False),
    ]

    kpi_dict: Dict[str, List[Any]] = {"Metric": [r[0] for r in kpi_rows]}

    for label, a in zip(labels, analyses):
        col_vals = []
        kpis = a["kpis"]
        delib = a["deliberation"]
        aud = kpis.get("audit_counts", {})

        for name, key, unit, higher_is_better in kpi_rows:
            val = None
            if key == "verified_count":
                val = aud.get("VERIFIED", 0)
            elif key == "flagged_count":
                val = aud.get("FLAGGED_WARNING", 0)
            elif key == "rejected_count":
                val = aud.get("REJECTED", 0) + aud.get("NEEDS_OUTREACH", 0)
            elif key == "paired_concordance_rate":
                val = delib.get("paired_concordance_rate", "N/A")
            elif key == "total_disputes":
                val = delib.get("total_disputes", 0)
            elif key == "total_runtime_min":
                val = round(kpis.get("total_elapsed_seconds", 0) / 60.0, 1)
            else:
                val = kpis.get(key, 0)

            col_vals.append(val)
        kpi_dict[label] = col_vals

    kpi_df = pd.DataFrame(kpi_dict).set_index("Metric")

    # If exactly 2 runs, compute Delta column (Run 2 - Run 1)
    if len(labels) == 2:
        col1 = labels[0]
        col2 = labels[1]
        deltas = []
        for idx, (name, key, unit, higher_is_better) in enumerate(kpi_rows):
            v1 = kpi_df.loc[name, col1]
            v2 = kpi_df.loc[name, col2]
            try:
                f1 = float(v1)
                f2 = float(v2)
                diff = f2 - f1
                sign = "+" if diff > 0 else ""
                deltas.append(f"{sign}{diff:.4f}" if abs(diff) < 1.0 else f"{sign}{diff:.1f}")
            except (ValueError, TypeError):
                deltas.append("-")
        kpi_df["Delta (2 - 1)"] = deltas

    # 2. Paper-by-Paper Cross-Run Matrix DataFrame
    all_paper_ids: Set[str] = set()
    for a in analyses:
        for p in a["papers"]:
            all_paper_ids.add(p["model_id"])

    sorted_papers = sorted(list(all_paper_ids))
    paper_matrix_dict: Dict[str, List[Any]] = {"Model ID": sorted_papers}

    for label, a in zip(labels, analyses):
        paper_lookup = {p["model_id"]: p for p in a["papers"]}
        col_vals = []
        for pid in sorted_papers:
            p_data = paper_lookup.get(pid)
            if not p_data:
                col_vals.append("N/A")
            else:
                if metric == "dice_mae":
                    col_vals.append(round(p_data.get("dice_mae", 0.0), 4))
                elif metric == "recall":
                    col_vals.append(round(p_data.get("recall", 0.0), 1))
                elif metric == "audit_status":
                    col_vals.append(p_data.get("audit_status", "UNKNOWN"))
                elif metric == "confidence":
                    col_vals.append(round(p_data.get("confidence", 0.0), 1))
                elif metric == "matched_datasets":
                    col_vals.append(p_data.get("matched_datasets", "-"))
                elif metric == "elapsed_seconds":
                    col_vals.append(round(p_data.get("elapsed_seconds", 0.0), 1))
                elif metric == "concordance_rate":
                    c_val = p_data.get("concordance_rate")
                    col_vals.append(round(c_val, 1) if c_val is not None else "-")
                elif metric == "disputes":
                    col_vals.append(p_data.get("disputes", 0))
                else:
                    col_vals.append(p_data.get(metric, "-"))
        paper_matrix_dict[label] = col_vals

    paper_df = pd.DataFrame(paper_matrix_dict).set_index("Model ID")

    # Filter mask for papers that differ across runs
    diff_mask = []
    for pid in sorted_papers:
        vals = [paper_df.loc[pid, l] for l in labels]
        valid_vals = [v for v in vals if v != "N/A" and v != "-"]
        diff_mask.append(len(set(valid_vals)) > 1)
    paper_df["Has Discrepancy"] = diff_mask

    latex_code = generate_latex_comparison_table(kpi_df, labels)
    markdown_report = generate_markdown_comparison_report(kpi_df, paper_df, labels, metric)

    return {
        "labels": labels,
        "kpi_df": kpi_df,
        "paper_df": paper_df,
        "analyses": analyses,
        "latex_table": latex_code,
        "markdown_report": markdown_report,
    }


def generate_latex_comparison_table(
    kpi_df: pd.DataFrame,
    labels: List[str],
) -> str:
    """
    Generates publication-ready booktabs LaTeX comparison table with bolding of best metrics.
    """
    num_cols = len(kpi_df.columns)
    col_spec = "l" + "r" * num_cols

    lines = [
        "% Auto-generated by Agent for GMMIS Benchmark Comparator",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Header line
    headers = ["Metric"] + [f"\\textbf{{{c}}}" for c in kpi_df.columns]
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    # Metrics where lower is better
    lower_better = {
        "Macro Dice MAE",
        "Median Dice MAE",
        "Micro Dice MAE",
        "Flagged Papers",
        "Rejected / Outreach",
        "Deliberation Disputes",
        "Avg Runtime / Paper (s)",
        "Total Runtime (min)",
    }

    for metric_name, row in kpi_df.iterrows():
        row_vals = []
        is_lower = metric_name in lower_better

        # Find best among standard runs (exclude Delta if present)
        numeric_candidates = []
        for l in labels:
            val = row[l]
            try:
                numeric_candidates.append(float(val))
            except (ValueError, TypeError):
                pass

        best_val = None
        if numeric_candidates:
            best_val = min(numeric_candidates) if is_lower else max(numeric_candidates)

        for col_name in kpi_df.columns:
            val = row[col_name]
            str_val = str(val)
            try:
                f_val = float(val)
                if col_name in labels and best_val is not None and f_val == best_val:
                    str_val = f"\\textbf{{{str_val}}}"
            except (ValueError, TypeError):
                pass
            row_vals.append(str_val)

        escaped_metric = str(metric_name).replace("%", "\\%").replace("_", "\\_")
        lines.append(f"{escaped_metric} & " + " & ".join(row_vals) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Multi-Run Benchmark Performance Comparison across Medical Image Segmentation Literature.}",
        "\\label{tab:gmmis_multi_run_comparison}",
        "\\end{table*}",
    ])

    return "\n".join(lines)


def generate_markdown_comparison_report(
    kpi_df: pd.DataFrame,
    paper_df: pd.DataFrame,
    labels: List[str],
    metric: str = "dice_mae",
) -> str:
    """Generates an executive Markdown synthesis report of the cross-run comparison."""
    md = [
        "# ⚔️ Multi-Run Benchmark Comparison & Synthesis Report",
        "",
        f"**Evaluated Model Configurations**: {', '.join(f'`{l}`' for l in labels)}  ",
        f"**Total Papers in Union**: {len(paper_df)}  ",
        f"**Primary Matrix Metric**: `{metric}`  ",
        "",
        "## 1. Aggregate KPI Comparison Matrix",
        "",
        kpi_df.to_markdown(),
        "",
        "## 2. Paper-by-Paper Cross-Run Matrix",
        "",
        paper_df.to_markdown(),
        "",
        "---",
        "*Report automatically synthesized by Agent for GMMIS Benchmark Orchestrator.*",
    ]
    return "\n".join(md)

