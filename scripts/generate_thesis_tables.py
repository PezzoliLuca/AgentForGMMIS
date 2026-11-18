#!/usr/bin/env python3
"""
Thesis Tables Generator (T1-T5) for Dual-Agent GMMIS Extraction & Auditing.

Reads one or more `benchmark_results.json` outputs produced by `scripts/run_batch_benchmark.py`
and exports publication-ready Markdown and LaTeX (booktabs) tables covering:
  - T1: Dual-Agent / Single-Agent IAA & Accuracy Comparison across model configurations
  - T2: Double-Blind Iterative Deliberation Trajectory (Round-1 vs Final IAA & Inter-Agent MAE)
  - T3: Epistemic Audit Breakdown & Ground-Truth Error Stratification (MAE_GT | Concordant vs Non-Concordant)
  - T4: Granular Per-Paper & Per-Organ Benchmark Coverage Summary
  - T5: Regime-Stratified Ground-Truth Accuracy (Active, Primary, Best, Regime-Agnostic, Recall)
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _fmt_pct(val: Optional[float], decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def _fmt_pct_tex(val: Optional[float], decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.{decimals}f}\\%"


def _fmt_float(val: Optional[float], decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _escape_tex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = str(text)
    for k, v in replacements.items():
        if k == "\\":
            continue
        out = out.replace(k, v)
    return out


def _load_runs(inputs: List[Path]) -> List[Dict[str, Any]]:
    loaded: List[Dict[str, Any]] = []
    for p in inputs:
        target = p / "benchmark_results.json" if p.is_dir() else p
        if not target.exists():
            raise FileNotFoundError(f"Benchmark JSON not found: {target}")
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_source_label"] = p.parent.name if target.name == "benchmark_results.json" and not p.is_dir() else p.name
        loaded.append(data)
    return loaded


def _run_config_label(run: Dict[str, Any]) -> str:
    meta = run.get("metadata", {})
    model_a = meta.get("model_a") or "AgentA"
    if meta.get("is_dual_mode"):
        model_b = meta.get("model_b") or "AgentB"
        return f"{model_a} + {model_b}"
    return f"{model_a} (Single)"


def _extract_paper_audit_status(paper: Dict[str, Any]) -> str:
    """Extracts audit status across flat (`audit_status`) and nested (`audit.status`) schemas."""
    status = paper.get("audit_status") or (paper.get("audit") or {}).get("status")
    return str(status) if status else "N/A"


def _extract_paper_gt_mae(paper: Dict[str, Any]) -> Optional[float]:
    """Extracts ground-truth Dice MAE across flat (`dice_mae`) and nested (`evaluation.dice_mae`) schemas."""
    if paper.get("dice_mae") is not None:
        return paper.get("dice_mae")
    return (paper.get("evaluation") or {}).get("dice_mae")


def _compute_run_iaa_fallback(run: Dict[str, Any]) -> Dict[str, Any]:
    """Computes IAA metrics from per-paper consensus entries if global iaa_metrics is absent."""
    gm = run.get("global_metrics", {})
    if gm.get("iaa_metrics"):
        return gm["iaa_metrics"]

    papers = run.get("papers", [])
    conc_papers = [p.get("consensus") for p in papers if p.get("consensus")]
    if not conc_papers:
        return {}

    tot_conc = sum(c.get("concordant_count", 0) for c in conc_papers)
    tot_disp = sum(c.get("dispute_count", 0) for c in conc_papers)
    tot_sing = sum(c.get("unconfirmed_count", 0) for c in conc_papers)
    all_paired = [
        sc for c in conc_papers for sc in c.get("scores", [])
        if sc.get("score_a") is not None and sc.get("score_b") is not None
    ]
    tot_paired = len(all_paired)
    tot_items = tot_conc + tot_disp + tot_sing
    deltas = [sc.get("delta") for sc in all_paired if sc.get("delta") is not None]
    exact_cnt = sum(1 for d in deltas if d <= 0.001)

    return {
        "micro_paired_iaa": (tot_conc / tot_paired) if tot_paired > 0 else None,
        "macro_paired_iaa": (
            sum(c.get("paired_concordance_rate", c.get("concordance_rate", 0.0)) for c in conc_papers) / len(conc_papers)
        ),
        "micro_strict_iaa": (tot_conc / tot_items) if tot_items > 0 else None,
        "macro_strict_iaa": None,
        "micro_exact_iaa": (exact_cnt / len(deltas)) if deltas else None,
        "micro_inter_agent_mae": (sum(deltas) / len(deltas)) if deltas else None,
        "concordant_items_total": tot_conc,
        "disputed_items_total": tot_disp,
        "singleton_items_total": tot_sing,
        "paired_items_total": tot_paired,
        "round1_macro_iaa": None,
        "round1_macro_inter_mae": None,
        "deliberation_iaa_delta_pp": None,
        "deliberation_resolved_total": 0,
        "gt_mae_when_concordant": None,
        "gt_mae_when_non_concordant": None,
    }


def build_markdown_tables(runs: List[Dict[str, Any]]) -> str:
    lines: List[str] = [
        "# Thesis Evaluation Tables (T1-T5)",
        "",
        "## Table T1: Architecture Comparison & Primary Inter-Agent Agreement (IAA)",
        "",
        "| Run / Architecture | Papers | Micro Paired IAA | Macro Paired IAA | Micro Strict IAA | Micro Exact IAA | Inter-Agent MAE | GT Recall | GT Micro MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        meta = run.get("metadata", {})
        gm = run.get("global_metrics", {})
        iaa = _compute_run_iaa_fallback(run)
        lines.append(
            f"| {_run_config_label(run)} | {meta.get('total_papers', len(run.get('papers', [])))} | "
            f"{_fmt_pct(iaa.get('micro_paired_iaa'))} | {_fmt_pct(iaa.get('macro_paired_iaa'))} | "
            f"{_fmt_pct(iaa.get('micro_strict_iaa'))} | {_fmt_pct(iaa.get('micro_exact_iaa'))} | "
            f"{_fmt_float(iaa.get('micro_inter_agent_mae'))} | {_fmt_pct(gm.get('overall_recall'))} | "
            f"{_fmt_float(gm.get('micro_mae_active'))} |"
        )

    lines.extend([
        "",
        "## Table T2: Double-Blind Iterative Deliberation Trajectory",
        "",
        "| Run / Architecture | Round-1 Macro IAA | Final Macro IAA | Delta IAA (pp) | Round-1 Inter-MAE | Final Inter-MAE | Items Resolved | Concordant / Disputed / Singletons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for run in runs:
        iaa = _compute_run_iaa_fallback(run)
        delta_pp = iaa.get("deliberation_iaa_delta_pp")
        delta_str = f"{delta_pp:+.1f} pp" if delta_pp is not None else "N/A"
        triplet = f"{iaa.get('concordant_items_total', 0)} / {iaa.get('disputed_items_total', 0)} / {iaa.get('singleton_items_total', 0)}"
        lines.append(
            f"| {_run_config_label(run)} | {_fmt_pct(iaa.get('round1_macro_iaa'))} | "
            f"{_fmt_pct(iaa.get('macro_paired_iaa'))} | {delta_str} | "
            f"{_fmt_float(iaa.get('round1_macro_inter_mae'))} | {_fmt_float(iaa.get('macro_inter_agent_mae'))} | "
            f"{iaa.get('deliberation_resolved_total', 0)} | {triplet} |"
        )

    lines.extend([
        "",
        "## Table T3: Consensus Audit Outcomes & Epistemic GT Stratification",
        "",
        "| Run / Architecture | Verified | Warning | Repo Inspect | Outreach | Rejected | MAE_GT (Concordant) | MAE_GT (Non-Concordant) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for run in runs:
        iaa = _compute_run_iaa_fallback(run)
        counts = Counter(
            _extract_paper_audit_status(p).lower()
            for p in run.get("papers", [])
        )
        c_n = iaa.get("gt_mae_when_concordant_n")
        nc_n = iaa.get("gt_mae_when_non_concordant_n")
        c_str = f"{_fmt_float(iaa.get('gt_mae_when_concordant'))} (n={c_n})" if iaa.get("gt_mae_when_concordant") is not None else "N/A"
        nc_str = f"{_fmt_float(iaa.get('gt_mae_when_non_concordant'))} (n={nc_n})" if iaa.get("gt_mae_when_non_concordant") is not None else "N/A"
        lines.append(
            f"| {_run_config_label(run)} | {counts.get('verified', 0)} | {counts.get('flagged_warning', 0)} | "
            f"{counts.get('needs_repo_inspection', 0)} | {counts.get('needs_outreach', 0)} | "
            f"{counts.get('rejected', 0)} | {c_str} | {nc_str} |"
        )

    lines.extend([
        "",
        "## Table T4: Per-Paper Granular Consensus & Benchmark Breakdown (Primary Run)",
        "",
        "| Model ID | Audit Status | Paired | Concordant | Disputed | Singletons | Paired IAA | Strict IAA | Inter-Agent MAE | GT MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    primary_run = runs[0]
    for p in primary_run.get("papers", []):
        cons = p.get("consensus") or {}
        status_val = _extract_paper_audit_status(p)
        gt_mae_val = _extract_paper_gt_mae(p)
        lines.append(
            f"| {p.get('model_id', 'Unknown')} | {status_val} | "
            f"{cons.get('paired_count', 0)} | {cons.get('concordant_count', 0)} | "
            f"{cons.get('dispute_count', 0)} | {cons.get('unconfirmed_count', 0)} | "
            f"{_fmt_pct(cons.get('paired_concordance_rate', cons.get('concordance_rate')))} | "
            f"{_fmt_pct(cons.get('strict_concordance_rate'))} | "
            f"{_fmt_float(cons.get('inter_agent_mae', cons.get('inter_model_dice_mae')))} | "
            f"{_fmt_float(gt_mae_val)} |"
        )

    lines.extend([
        "",
        "## Table T5: Ground-Truth Alignment by Experimental Regime",
        "",
        "| Run / Architecture | Matched / Total GT | Recall | Active MAE | Primary MAE | Best MAE | Regime-Agnostic MAE | Coverage-Adj Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for run in runs:
        gm = run.get("global_metrics", {})
        matched_str = f"{gm.get('total_matched_benchmarks', 0)} / {gm.get('total_gt_benchmarks_active', 0)}"
        lines.append(
            f"| {_run_config_label(run)} | {matched_str} | {_fmt_pct(gm.get('overall_recall'))} | "
            f"{_fmt_float(gm.get('micro_mae_active'))} | {_fmt_float(gm.get('micro_mae_primary'))} | "
            f"{_fmt_float(gm.get('micro_mae_best'))} | {_fmt_float(gm.get('micro_mae_regime_agnostic'))} | "
            f"{_fmt_float(gm.get('coverage_adjusted_score'))} |"
        )

    return "\n".join(lines) + "\n"


def build_latex_tables(runs: List[Dict[str, Any]]) -> str:
    sections: List[str] = [
        "% Auto-generated LaTeX booktabs tables (T1-T5) from benchmark_results.json",
        "% Requires \\usepackage{booktabs}",
        "",
        "% Table T1: Dual-Agent IAA & Accuracy Comparison",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Architecture Comparison and Primary Inter-Agent Agreement (IAA).}",
        r"\label{tab:t1_iaa_comparison}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Configuration & Papers & Micro IAA & Macro IAA & Strict IAA & Exact IAA & Inter-MAE & Recall & GT MAE \\",
        r"\midrule",
    ]
    for run in runs:
        meta = run.get("metadata", {})
        gm = run.get("global_metrics", {})
        iaa = _compute_run_iaa_fallback(run)
        sections.append(
            f"{_escape_tex(_run_config_label(run))} & {meta.get('total_papers', len(run.get('papers', [])))} & "
            f"{_fmt_pct_tex(iaa.get('micro_paired_iaa'))} & {_fmt_pct_tex(iaa.get('macro_paired_iaa'))} & "
            f"{_fmt_pct_tex(iaa.get('micro_strict_iaa'))} & {_fmt_pct_tex(iaa.get('micro_exact_iaa'))} & "
            f"{_fmt_float(iaa.get('micro_inter_agent_mae'))} & {_fmt_pct_tex(gm.get('overall_recall'))} & "
            f"{_fmt_float(gm.get('micro_mae_active'))} \\\\"
        )
    sections.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    sections.extend([
        "% Table T2: Deliberation Trajectory",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Double-Blind Iterative Deliberation Trajectory (Round-1 to Final).}",
        r"\label{tab:t2_deliberation_trajectory}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Configuration & R1 IAA & Final IAA & $\Delta$ IAA & R1 MAE & Final MAE & Resolved & C / D / S \\",
        r"\midrule",
    ])
    for run in runs:
        iaa = _compute_run_iaa_fallback(run)
        dpp = iaa.get("deliberation_iaa_delta_pp")
        dpp_str = f"{dpp:+.1f} pp" if dpp is not None else "N/A"
        triplet = f"{iaa.get('concordant_items_total', 0)}/{iaa.get('disputed_items_total', 0)}/{iaa.get('singleton_items_total', 0)}"
        sections.append(
            f"{_escape_tex(_run_config_label(run))} & {_fmt_pct_tex(iaa.get('round1_macro_iaa'))} & "
            f"{_fmt_pct_tex(iaa.get('macro_paired_iaa'))} & {dpp_str} & "
            f"{_fmt_float(iaa.get('round1_macro_inter_mae'))} & {_fmt_float(iaa.get('macro_inter_agent_mae'))} & "
            f"{iaa.get('deliberation_resolved_total', 0)} & {triplet} \\\\"
        )
    sections.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    sections.extend([
        "% Table T3: Epistemic Audit Outcomes & GT Stratification",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Consensus Certification Outcomes and Ground-Truth Error Stratification by Agreement.}",
        r"\label{tab:t3_epistemic_stratification}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Configuration & Verified & Warning & Repo & Outreach & Rejected & MAE (Conc.) & MAE (Non-Conc.) \\",
        r"\midrule",
    ])
    for run in runs:
        iaa = _compute_run_iaa_fallback(run)
        counts = Counter(_extract_paper_audit_status(p).lower() for p in run.get("papers", []))
        sections.append(
            f"{_escape_tex(_run_config_label(run))} & {counts.get('verified', 0)} & {counts.get('flagged_warning', 0)} & "
            f"{counts.get('needs_repo_inspection', 0)} & {counts.get('needs_outreach', 0)} & {counts.get('rejected', 0)} & "
            f"{_fmt_float(iaa.get('gt_mae_when_concordant'))} & {_fmt_float(iaa.get('gt_mae_when_non_concordant'))} \\\\"
        )
    sections.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    sections.extend([
        "% Table T4: Per-Paper Granular Consensus Coverage",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-Paper Granular Consensus and Inter-Agent Agreement Coverage.}",
        r"\label{tab:t4_per_paper_iaa}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Model ID & Status & Paired & Conc. & Disp. & Sing. & Paired IAA & Inter-MAE & GT MAE \\",
        r"\midrule",
    ])
    for p in runs[0].get("papers", []):
        cons = p.get("consensus") or {}
        status_val = _extract_paper_audit_status(p)
        gt_mae_val = _extract_paper_gt_mae(p)
        sections.append(
            f"{_escape_tex(p.get('model_id', 'Unknown'))} & {_escape_tex(status_val)} & "
            f"{cons.get('paired_count', 0)} & {cons.get('concordant_count', 0)} & "
            f"{cons.get('dispute_count', 0)} & {cons.get('unconfirmed_count', 0)} & "
            f"{_fmt_pct_tex(cons.get('paired_concordance_rate', cons.get('concordance_rate')))} & "
            f"{_fmt_float(cons.get('inter_agent_mae', cons.get('inter_model_dice_mae')))} & "
            f"{_fmt_float(gt_mae_val)} \\\\"
        )
    sections.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    sections.extend([
        "% Table T5: Regime-Stratified Ground-Truth Accuracy",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ground-Truth Alignment Across Evaluation Regimes (Primary, Best, Regime-Agnostic).}",
        r"\label{tab:t5_regime_accuracy}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Configuration & Matched/GT & Recall & Active MAE & Primary MAE & Best MAE & Min-Regime MAE & Cov-Adj Score \\",
        r"\midrule",
    ])
    for run in runs:
        gm = run.get("global_metrics", {})
        matched_str = f"{gm.get('total_matched_benchmarks', 0)}/{gm.get('total_gt_benchmarks_active', 0)}"
        sections.append(
            f"{_escape_tex(_run_config_label(run))} & {matched_str} & {_fmt_pct_tex(gm.get('overall_recall'))} & "
            f"{_fmt_float(gm.get('micro_mae_active'))} & {_fmt_float(gm.get('micro_mae_primary'))} & "
            f"{_fmt_float(gm.get('micro_mae_best'))} & {_fmt_float(gm.get('micro_mae_regime_agnostic'))} & "
            f"{_fmt_float(gm.get('coverage_adjusted_score'))} \\\\"
        )
    sections.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Thesis Tables T1-T5 from benchmark_results.json files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more benchmark_results.json paths or run directories.")
    parser.add_argument("--output-dir", default=None, help="Output directory (defaults to parent of first input).")
    parser.add_argument("--format", choices=["both", "markdown", "latex"], default="both")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    runs = _load_runs(input_paths)
    out_dir = Path(args.output_dir) if args.output_dir else (input_paths[0] if input_paths[0].is_dir() else input_paths[0].parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("both", "markdown"):
        md_content = build_markdown_tables(runs)
        md_path = out_dir / "thesis_tables.md"
        md_path.write_text(md_content, encoding="utf-8")
        print(f"Markdown thesis tables saved to: {md_path}")

    if args.format in ("both", "latex"):
        tex_content = build_latex_tables(runs)
        tex_path = out_dir / "thesis_tables.tex"
        tex_path.write_text(tex_content, encoding="utf-8")
        print(f"LaTeX thesis tables saved to: {tex_path}")


if __name__ == "__main__":
    main()

