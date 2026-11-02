import json
from pathlib import Path
import re

RUNS = [
    {
        "key": "qwen_llama",
        "name": "Dual Local: Qwen 2.5 7B + Llama 3.1 8B",
        "short_name": "Qwen 7B + Llama 8B",
        "type": "Dual Local (100% On-Premise)",
        "old_path": Path("logs/runs/20261026_200829_mufasa_dual_local_40"),
        "new_path": Path("logs/runs/20261029_124310_mufasa_dual_local_Qwen257B_MetaLlama318B_40"),
    },
    {
        "key": "qwen_mistral",
        "name": "Dual Local: Qwen 2.5 7B + Mistral 7B v0.3",
        "short_name": "Qwen 7B + Mistral 7B",
        "type": "Dual Local (100% On-Premise)",
        "old_path": Path("logs/runs/20261027_212758_mufasa_dual_local_Qwen257B_Mistral7B_40"),
        "new_path": Path("logs/runs/20261029_164230_mufasa_dual_local_Qwen257B_Mistral7B_40"),
    },
    {
        "key": "qwen14b_gemini",
        "name": "Hybrid: Qwen 2.5 14B + Gemini 3.1 Flash-Lite",
        "short_name": "Qwen 14B + Gemini",
        "type": "Hybrid (Local + Cloud API)",
        "old_path": Path("logs/runs/20261026_181522_mufasa_qwen14b_gemini_40"),
        "new_path": Path("logs/runs/20261029_202730_mufasa_qwen2_5_14b_gemini_40"),
    },
    {
        "key": "codestral_gemini",
        "name": "Hybrid: Codestral 22B + Gemini 3.1 Flash-Lite",
        "short_name": "Codestral 22B + Gemini",
        "type": "Hybrid (Local + Cloud API)",
        "old_path": Path("logs/runs/20261027_172306_mufasa_codestral_22b_gemini_40"),
        "new_path": Path("logs/runs/20261029_220511_mufasa_codestral_22b_gemini_40"),
    },
    {
        "key": "mistral_gemini",
        "name": "Hybrid: Mistral 7B v0.3 + Gemini 3.1 Flash-Lite",
        "short_name": "Mistral 7B + Gemini",
        "type": "Hybrid (Local + Cloud API)",
        "old_path": Path("logs/runs/20261027_200809_mufasa_mistral_7b_v0_3_gemini_40"),
        "new_path": Path("logs/runs/20261030_001511_mufasa_mistral_7b_v0_3_gemini_40"),
    },
]

def parse_matched(m_str):
    if not m_str or "/" not in str(m_str): return 0, 0
    p = str(m_str).split("/")
    return int(p[0]), int(p[1])

def parse_mae(m_str):
    if m_str is None or "n/a" in str(m_str).lower(): return None
    try: return float(m_str)
    except: return None

def analyze_run(run_path):
    res_path = run_path / "benchmark_results.json"
    with open(res_path) as f:
        data = json.load(f)
    
    summary = data.get("summary", [])
    papers_dict = {x["model_id"]: x for x in summary}
    
    # Deliberation analysis
    extractions_dir = run_path / "extractions"
    rounds_count = {}
    final_concordances = []
    final_inter_maes = []
    
    # Curation / Schema analysis
    sql_path = run_path / "database_curation.sql"
    alter_count = 0
    novel_cols = set()
    total_sql_lines = 0
    if sql_path.exists():
        with open(sql_path) as f:
            lines = f.readlines()
            total_sql_lines = len(lines)
            for line in lines:
                if "ALTER TABLE" in line.upper():
                    alter_count += 1
                    m = re.search(r'ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', line, re.IGNORECASE)
                    if m:
                        novel_cols.add(m.group(1))

    for p_dir in extractions_dir.iterdir():
        if not p_dir.is_dir(): continue
        traj_file = p_dir / "deliberation_trajectory.json"
        if traj_file.exists():
            with open(traj_file) as f:
                traj = json.load(f)
            rounds = len(traj)
            rounds_count[rounds] = rounds_count.get(rounds, 0) + 1
            if traj:
                last = traj[-1]
                c = last.get("concordance_rate_after")
                m = last.get("inter_model_mae_after")
                if c is not None: final_concordances.append(c)
                if m is not None: final_inter_maes.append(m)

    # Runtime analysis from execution.log or metadata
    duration_min = None
    exec_log = run_path / "execution.log"
    if exec_log.exists():
        with open(exec_log) as f:
            content = f.read()
        # Look for total elapsed or timestamps
        # Check start & end timestamps
        lines = content.strip().split("\n")
        first_line = lines[0] if lines else ""
        last_line = lines[-1] if lines else ""
        # Or look for Elapsed time pattern
        m = re.search(r'Elapsed time:\s*([0-9\.]+)\s*minutes', content)
        if m:
            duration_min = float(m.group(1))
        else:
            # Try to infer from first and last line timestamp
            m_start = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', first_line)
            m_end = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', last_line)
            if m_start and m_end:
                from datetime import datetime
                t0 = datetime.strptime(m_start.group(1), "%Y-%m-%d %H:%M:%S")
                t1 = datetime.strptime(m_end.group(1), "%Y-%m-%d %H:%M:%S")
                duration_min = (t1 - t0).total_seconds() / 60.0

    # Macro accuracy metrics
    maes = []
    matched_counts = []
    total_gt = 0
    total_matched = 0
    zero_err_count = 0
    verified_count = 0
    confidences = []

    for item in summary:
        m_cnt, gt_cnt = parse_matched(item.get("matched_datasets", "0/0"))
        total_matched += m_cnt
        total_gt += gt_cnt
        if m_cnt > 0:
            matched_counts.append(m_cnt)
            mae = parse_mae(item.get("dice_mae"))
            if mae is not None:
                maes.append(mae)
                if mae == 0.0:
                    zero_err_count += 1
        
        status = item.get("audit_status", "").replace("✅", "").replace("⚠️", "").strip().upper()
        if "VERIFIED" in status:
            verified_count += 1
            
        conf_str = item.get("confidence", "0%").replace("%", "").strip()
        try:
            confidences.append(float(conf_str))
        except:
            pass

    macro_mae = sum(maes) / len(maes) if maes else None
    import statistics
    median_mae = statistics.median(maes) if maes else None
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    avg_concordance = sum(final_concordances) / len(final_concordances) if final_concordances else None
    avg_inter_mae = sum(final_inter_maes) / len(final_inter_maes) if final_inter_maes else None

    return {
        "run_name": run_path.name,
        "total_papers": len(summary),
        "papers_with_matches": len(matched_counts),
        "total_matched_benchmarks": total_matched,
        "total_gt_benchmarks": total_gt,
        "recall_pct": (total_matched / total_gt * 100) if total_gt else 0.0,
        "macro_mae": macro_mae,
        "median_mae": median_mae,
        "zero_err_papers": zero_err_count,
        "zero_err_pct": (zero_err_count / len(matched_counts) * 100) if matched_counts else 0.0,
        "verified_papers": verified_count,
        "avg_confidence": avg_conf,
        "deliberation_rounds": rounds_count,
        "avg_concordance": avg_concordance,
        "avg_inter_mae": avg_inter_mae,
        "alter_table_count": alter_count,
        "novel_columns_count": len(novel_cols),
        "novel_columns": sorted(list(novel_cols)),
        "duration_min": duration_min,
        "sec_per_paper": (duration_min * 60.0 / len(summary)) if duration_min and len(summary) else None,
        "papers_dict": papers_dict,
    }

results = []
for r in RUNS:
    print(f"Analyzing {r['name']}...")
    old_res = analyze_run(r["old_path"])
    new_res = analyze_run(r["new_path"])
    results.append({
        "info": r,
        "old": old_res,
        "new": new_res,
    })

with open("logs/runs/mega_matrix_aggregated_raw.json", "w") as f:
    # Serialize json without paper_dict to keep readable
    summary_for_json = []
    for item in results:
        info_clean = {k: (str(v) if isinstance(v, Path) else v) for k, v in item["info"].items()}
        summary_for_json.append({
            "info": info_clean,
            "old": {k: v for k, v in item["old"].items() if k != "papers_dict"},
            "new": {k: v for k, v in item["new"].items() if k != "papers_dict"},
        })
    json.dump(summary_for_json, f, indent=2)

print("\n" + "="*120)
print(f"{'Configuration':<35} | {'Type':<10} | {'Matched':<14} | {'Recall':<8} | {'Macro MAE':<18} | {'Median MAE':<18} | {'Zero Err':<12} | {'Time (min)':<14}")
print("="*120)
for item in results:
    inf = item["info"]
    o = item["old"]
    n = item["new"]
    
    name = inf["short_name"]
    type_str = "Local" if "Local" in inf["type"] else "Hybrid"
    
    m_str = f"{o['total_matched_benchmarks']} -> {n['total_matched_benchmarks']}"
    rec_str = f"{n['recall_pct']:.1f}%"
    mae_str = f"{o['macro_mae']:.4f} -> {n['macro_mae']:.4f}"
    med_str = f"{o['median_mae']:.4f} -> {n['median_mae']:.4f}"
    zero_str = f"{o['zero_err_papers']} -> {n['zero_err_papers']}"
    time_str = f"{o['duration_min']:.1f}m -> {n['duration_min']:.1f}m" if o['duration_min'] and n['duration_min'] else "N/A"
    
    print(f"{name:<35} | {type_str:<10} | {m_str:<14} | {rec_str:<8} | {mae_str:<18} | {med_str:<18} | {zero_str:<12} | {time_str:<14}")


print("Raw analysis complete. Saved to logs/runs/mega_matrix_aggregated_raw.json")
