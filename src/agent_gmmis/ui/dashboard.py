"""Unified Streamlit Web Dashboard for Agent for GMMIS.

3-Tab Interactive Workbench:
  Tab 1: 🚀 Launch & Monitor Benchmarks (Local & SLURM Mufasa cluster launcher)
  Tab 2: 🔍 Single-Run & Paper Inspector (KPI cards, score alignment, CoT, deliberation)
  Tab 3: ⚔️ Multi-Run Comparator & Thesis Exporter (Plotly Pareto charts, Mega-Matrix, LaTeX/Markdown export)
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agent_gmmis.config import DEFAULT_DB_PATH
from agent_gmmis.data.db_manager import DatabaseManager
from agent_gmmis.extraction.run_logger import sanitize_filename
from agent_gmmis.llm.registry import (
    MODEL_CATALOG,
    ModelProfile,
    plan_mufasa_allocation,
    resolve_model_profile,
)
from agent_gmmis.ui.cluster_manager import ClusterManager
from agent_gmmis.ui.run_analyzer import (
    compare_runs,
    discover_runs,
    load_paper_detail,
    load_run_analysis,
)

# Page configuration
st.set_page_config(
    page_title="Agent for GMMIS — Benchmark Orchestrator & Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for cleaner metrics and readable code snippets
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 8px;
    }
    .status-badge {
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cluster_manager() -> ClusterManager:
    return ClusterManager()


@st.cache_data(ttl=60)
def get_cached_runs() -> List[Dict[str, Any]]:
    return discover_runs()


@st.cache_data(ttl=120)
def get_analysis_for_run(run_dir: str) -> Dict[str, Any]:
    return load_run_analysis(run_dir)


@st.cache_data(ttl=300)
def get_db_model_ids() -> List[str]:
    try:
        db = DatabaseManager(DEFAULT_DB_PATH)
        return [m.id for m in db.get_all_models()]
    except Exception:
        return []


cm = get_cluster_manager()
all_runs = get_cached_runs()
all_db_models = get_db_model_ids()

# ==============================================================================
# SIDEBAR: Cluster Health & Global Sync Actions
# ==============================================================================

with st.sidebar:
    st.markdown("## 🔬 **Agent for GMMIS**")
    st.caption("Generalist Medical Image Segmentation Agent Orchestrator")
    st.divider()

    st.markdown("### 🖥️ **Cluster Connectivity**")
    check_status = st.button("📡 Check Mufasa Status", use_container_width=True)

    if "mufasa_status" not in st.session_state or check_status:
        with st.spinner("Pinging Mufasa HPC..."):
            st.session_state["mufasa_status"] = cm.check_mufasa_status()

    m_status = st.session_state.get("mufasa_status", {"online": False})
    if m_status.get("online"):
        st.success(f"🟢 **Mufasa Online** (`{m_status.get('host')}`)")
        n_jobs = len(m_status.get("active_jobs", []))
        st.info(f"Active SLURM Jobs: **{n_jobs}**")
        with st.expander("📦 Cached HF Models on Mufasa", expanded=False):
            models_cached = m_status.get("cached_models", [])
            if models_cached:
                for mc in models_cached:
                    st.markdown(f"- `{mc}`")
            else:
                st.caption("No cached models discovered.")
    else:
        st.warning("⚪ **Mufasa Offline** (Local Mode Active)")
        if m_status.get("error"):
            st.caption(f"Error: {m_status['error']}")

    st.divider()
    st.markdown("### 🔄 **Data Sync & Actions**")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("📥 Sync Logs", help="Pulls latest runs from Mufasa to logs/mufasa_runs", use_container_width=True):
            with st.spinner("Rsyncing logs from Mufasa..."):
                ok, msg = cm.sync_logs_from_mufasa()
                if ok:
                    st.toast("Logs successfully synced from Mufasa!", icon="✅")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"Sync failed:\n{msg}")

    with col_s2:
        if st.button("📤 Push Code", help="Syncs local src/ and scripts/ to Mufasa", use_container_width=True):
            with st.spinner("Rsyncing code to Mufasa..."):
                ok, msg = cm.sync_code_to_mufasa()
                if ok:
                    st.toast("Codebase pushed to Mufasa!", icon="🚀")
                else:
                    st.error(f"Push failed:\n{msg}")

    if st.button("🔃 Refresh Runs Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(f"**Discovered Runs**: `{len(all_runs)}`")
    st.markdown("---")
    st.caption("Politecnico di Milano — M.Sc. Thesis in Computer Science & Engineering")


# ==============================================================================
# MAIN TABS INTERFACE
# ==============================================================================

tab_launch, tab_inspect, tab_compare = st.tabs([
    "🚀 Tab 1: Launch & Monitor",
    "🔍 Tab 2: Run & Paper Inspector",
    "⚔️ Tab 3: Multi-Run Comparator & Thesis Exporter",
])

# ------------------------------------------------------------------------------
# TAB 1: LAUNCH & MONITOR BENCHMARKS
# ------------------------------------------------------------------------------
with tab_launch:
    st.header("🚀 Launch & Monitor Benchmark Jobs")
    st.markdown(
        "Configure model architectures, execution topologies, and paper evaluation cohorts. "
        "Supports 100% on-premise dual execution with dynamic sequential staging on 20GB MIG slices."
    )

    col_env, col_topo = st.columns(2)
    with col_env:
        exec_env = st.radio(
            "Target Execution Environment",
            options=["Mufasa HPC (SLURM)", "Local Machine (CLI)"],
            index=0 if m_status.get("online") else 1,
            horizontal=True,
        )
    with col_topo:
        pipeline_topology = st.radio(
            "Pipeline Topology",
            options=[
                "Single-Agent Extraction",
                "Dual-Agent Hybrid (Mufasa GPU + Gemini Cloud)",
                "Dual-Agent 100% On-Premise (Mufasa GPU + Mufasa GPU)",
            ],
            index=2,
            horizontal=False,
        )

    st.markdown("---")
    st.subheader("🤖 Model Selection & VRAM Allocation")

    model_keys = list(MODEL_CATALOG.keys())
    mufasa_cached_keys = [k for k, p in MODEL_CATALOG.items() if p.provider == "mufasa"]

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**Agent A Configuration**")
        agent_a_key = st.selectbox(
            "Select Agent A Model",
            options=model_keys,
            index=model_keys.index("qwen2.5-7b-awq") if "qwen2.5-7b-awq" in model_keys else 0,
            key="agent_a_select",
        )
        prof_a = MODEL_CATALOG[agent_a_key]
        custom_a = st.checkbox("Specify Custom HuggingFace Repo ID for Agent A", value=False)
        model_a_id = st.text_input("Custom Agent A Repo ID", value=prof_a.hf_repo_or_id) if custom_a else prof_a.hf_repo_or_id

        st.caption(
            f"**Provider**: `{prof_a.provider}` | **VRAM**: `{prof_a.vram_gb:.1f} GB` | "
            f"**Context**: `{prof_a.default_max_len_single}` tokens"
            + (f" | ⚠️ *4-bit BnB quantization*" if prof_a.quantization_flag else "")
            + (f" | 🧠 *Reasoning (<think> traces)*" if prof_a.is_reasoning else "")
        )

    is_dual_selected = pipeline_topology != "Single-Agent Extraction"
    model_b_id = None
    provider_b = None

    with col_m2:
        if is_dual_selected:
            st.markdown("**Agent B Configuration**")
            if "Hybrid" in pipeline_topology:
                default_b_keys = [k for k, p in MODEL_CATALOG.items() if p.provider == "gemini"]
                b_options = default_b_keys + [k for k in model_keys if k not in default_b_keys]
                default_idx = 0
            else:
                b_options = mufasa_cached_keys
                default_idx = b_options.index("mistral-7b-awq") if "mistral-7b-awq" in b_options else 0

            agent_b_key = st.selectbox(
                "Select Agent B Model",
                options=b_options,
                index=default_idx,
                key="agent_b_select",
            )
            prof_b = MODEL_CATALOG[agent_b_key]
            custom_b = st.checkbox("Specify Custom HuggingFace Repo ID for Agent B", value=False)
            model_b_id = st.text_input("Custom Agent B Repo ID", value=prof_b.hf_repo_or_id) if custom_b else prof_b.hf_repo_or_id
            provider_b = prof_b.provider

            st.caption(
                f"**Provider**: `{prof_b.provider}` | **VRAM**: `{prof_b.vram_gb:.1f} GB` | "
                f"**Context**: `{prof_b.default_max_len_single}` tokens"
                + (f" | ⚠️ *4-bit BnB quantization*" if prof_b.quantization_flag else "")
                + (f" | 🧠 *Reasoning (<think> traces)*" if prof_b.is_reasoning else "")
            )
        else:
            st.markdown("**Agent B Configuration**")
            st.info("Agent B is disabled for Single-Agent Extraction.")

    # Live Allocation Plan Calculation
    plan = plan_mufasa_allocation(
        model_a=model_a_id,
        model_b=model_b_id if is_dual_selected else None,
        provider_a=prof_a.provider,
        provider_b=provider_b,
    )

    # Dynamic Execution Mode Banner
    if is_dual_selected:
        if plan["topology"] == "dual_simultaneous":
            st.success(
                f"⚡ **Simultaneous Dual-vLLM Execution**: Combined weight footprint ({plan['prof_a'].vram_gb + plan['prof_b'].vram_gb:.1f} GB) "
                f"fits comfortably on the 20 GB A100 MIG slice! Dedicated allocations: "
                f"**Agent A ({plan['util_a']*100:.0f}% VRAM)** + **Agent B ({plan['util_b']*100:.0f}% VRAM)**."
            )
        elif plan["topology"] == "dual_sequential":
            st.warning(
                f"🔄 **2-Stage Sequential Dual Execution**: Combined weights ({plan['prof_a'].vram_gb + plan['prof_b'].vram_gb:.1f} GB) "
                f"exceed concurrent KV-cache budget on the 20 GB MIG slice. "
                f"The orchestrator will automatically execute **Stage 1 (Agent A at 90% VRAM)** followed by "
                f"**Stage 2 (Agent B + Deliberation at 90% VRAM)** with zero VRAM OOM risk."
            )
        elif plan["topology"] == "hybrid":
            st.info(
                f"☁️ **Hybrid Cloud Execution**: Local model runs with full 90% VRAM on Mufasa GPU, "
                f"while Cloud Agent runs via native SDK API."
            )

    if plan.get("needs_bitsandbytes"):
        st.info("✨ **Dynamic 4-bit Quantization**: `bitsandbytes` will be loaded on-the-fly on Mufasa for BF16 models.")

    st.markdown("---")
    st.subheader("📑 Paper Evaluation Cohort & Protocol Settings")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        cohort_mode = st.radio("Cohort Selection", ["First N ArXiv Papers", "Custom Paper List"], horizontal=True)
        if cohort_mode == "First N ArXiv Papers":
            col_sl, col_btn = st.columns([3, 1])
            with col_sl:
                paper_count = st.slider("Number of Papers", min_value=1, max_value=98, value=40)
            with col_btn:
                st.caption("Quick Presets")
                preset_cols = st.columns(3)
                if preset_cols[0].button("3"):
                    paper_count = 3
                if preset_cols[1].button("20"):
                    paper_count = 20
                if preset_cols[2].button("40"):
                    paper_count = 40
            custom_paper_models = None
        else:
            custom_paper_models = st.multiselect("Select Models from SQLite", options=all_db_models, default=all_db_models[:5])
            paper_count = len(custom_paper_models)

    with col_p2:
        max_delib_rounds = st.slider("Max Deliberation Rounds", min_value=1, max_value=5, value=3)
        gt_source = st.selectbox("Ground Truth Benchmark Source", ["primary", "best"], index=0)
        c_audit, c_curate = st.columns(2)
        with c_audit:
            audit_gt = st.checkbox("Benchmark Mode (Cross-Check Survey GT)", value=False)
        with c_curate:
            apply_curate = st.checkbox("Apply Curation to SQLite (Live DB)", value=False)

    default_tag = f"bench_{plan['topology']}_{plan['prof_a'].key}_{paper_count}"
    run_tag_input = st.text_input("Custom Run Tag", value=default_tag)

    # Command Line Preview
    st.markdown("**Command Preview**")
    if exec_env == "Mufasa HPC (SLURM)":
        preview_cmd = (
            f"sbatch --export=ALL,MODEL_A={model_a_id}"
            + (f",MODEL_B={model_b_id}" if is_dual_selected else "")
            + f",COUNT={paper_count},MAX_ROUNDS={max_delib_rounds},COMPARE_SOURCE={gt_source},TAG={run_tag_input} "
            f"scripts/mufasa/run_unified_benchmark_mufasa.sh"
        )
    else:
        preview_cmd = (
            f"python3 scripts/run_batch_benchmark.py "
            f"--provider-a {prof_a.provider} --model-a '{model_a_id}' "
            + (f"--provider-b {provider_b} --model-b '{model_b_id}' " if is_dual_selected else "")
            + f"--count {paper_count} --max-rounds {max_delib_rounds} --compare-source {gt_source} --tag {run_tag_input} "
            + ("--audit-ground-truth " if audit_gt else "")
            + ("--apply-curation " if apply_curate else "")
            + (f"--models {' '.join(custom_paper_models)} " if custom_paper_models else "")
        )
    st.code(preview_cmd, language="bash")

    col_btn_launch, col_btn_sync = st.columns([1, 3])
    with col_btn_launch:
        if st.button("🚀 Launch Benchmark Job", type="primary", use_container_width=True):
            if exec_env == "Mufasa HPC (SLURM)":
                if not m_status.get("online"):
                    st.error("Cannot launch on Mufasa: Host is unreachable.")
                else:
                    with st.spinner("Submitting SLURM job to Mufasa..."):
                        cfg = {
                            "model_a": model_a_id,
                            "model_b": model_b_id or "",
                            "provider_a": prof_a.provider,
                            "provider_b": provider_b or "",
                            "count": paper_count,
                            "max_rounds": max_delib_rounds,
                            "compare_source": gt_source,
                            "tag": run_tag_input,
                        }
                        ok, res_job_id = cm.launch_mufasa_job(cfg, sync_first=True)
                        if ok:
                            st.success(f"🎉 Job successfully submitted! **SLURM Job ID**: `{res_job_id}`")
                            st.session_state["monitored_job_id"] = res_job_id
                        else:
                            st.error(f"Job submission failed:\n{res_job_id}")
            else:
                st.info("To execute locally, run the command shown above in your terminal.")

    st.markdown("---")
    st.subheader("📊 Live Cluster Job Queue & Log Streamer")

    active_jobs = m_status.get("active_jobs", [])
    if active_jobs:
        jobs_df = pd.DataFrame(active_jobs)
        st.dataframe(jobs_df, use_container_width=True)

        job_ids = [j["job_id"] for j in active_jobs]
        monitored_id = st.session_state.get("monitored_job_id", job_ids[0] if job_ids else "")
        selected_job = st.selectbox("Select Job to Inspect / Cancel", job_ids, index=job_ids.index(monitored_id) if monitored_id in job_ids else 0)

        col_tail, col_scancel = st.columns([3, 1])
        with col_tail:
            if st.button("📜 Fetch Live Remote Logs", use_container_width=True):
                with st.spinner("Reading remote logs from Mufasa..."):
                    logs = cm.get_mufasa_job_logs(selected_job, tail_lines=120)
                    st.text_area("Live Job Output", value=logs, height=350)
        with col_scancel:
            if st.button("🛑 Cancel Job (scancel)", type="secondary", use_container_width=True):
                with st.spinner(f"Cancelling Job {selected_job}..."):
                    ok, res = cm.cancel_mufasa_job(selected_job)
                    if ok:
                        st.warning(f"Job {selected_job} cancelled.")
                        st.rerun()
                    else:
                        st.error(f"Cancellation error:\n{res}")
    else:
        st.caption("No active jobs currently running in SLURM queue for user `lpezzoli`.")


# ------------------------------------------------------------------------------
# TAB 2: SINGLE-RUN & PAPER INSPECTOR
# ------------------------------------------------------------------------------
with tab_inspect:
    st.header("🔍 Single-Run Deep Telemetry & Paper Inspector")

    if not all_runs:
        st.warning("No benchmark runs discovered in `logs/runs` or `logs/mufasa_runs`. Run a benchmark or sync logs from Mufasa.")
    else:
        # Run Selector with metadata formatting
        run_options = {
            r["run_dir"]: f"{r['timestamp']} | {r['run_id']} ({r['total_papers']} papers)"
            for r in all_runs
        }
        selected_run_dir = st.selectbox(
            "Select Benchmark Run",
            options=list(run_options.keys()),
            format_func=lambda d: run_options[d],
            key="tab2_run_selector",
        )

        analysis = get_analysis_for_run(selected_run_dir)
        kpis = analysis["kpis"]
        delib = analysis["deliberation"]
        curate = analysis["curation"]
        meta = analysis["metadata"]

        # Header Badge Row
        st.markdown(
            f"**Run ID**: `{analysis['run_id']}` | **Models**: `{meta.get('model_a', 'N/A')}` "
            + (f" + `{meta.get('model_b')}`" if meta.get("model_b") else "")
            + f" | **Ground Truth**: `{meta.get('compare_source', 'primary').upper()}`"
            + f" | **Deliberation Mode**: `{meta.get('auditor_mode', 'Autonomous')}`"
        )

        # Global KPI Metric Cards
        st.markdown("### 📈 Global Performance KPIs")
        m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
        m_c1.metric("Dataset Recall", f"{kpis['overall_recall']:.1f}%")
        m_c2.metric("Macro Dice MAE", f"{kpis['macro_mae']:.4f}")
        m_c3.metric("Median Dice MAE", f"{kpis['median_mae']:.4f}")
        m_c4.metric("Micro Dice MAE", f"{kpis['micro_mae']:.4f}")
        m_c5.metric("Coverage Score", f"{kpis['coverage_adjusted_score']:.4f}")
        m_c6.metric("Zero-Error Papers", f"{kpis['zero_error_count']} / {kpis['total_papers']}")

        m_d1, m_d2, m_d3, m_d4, m_d5, m_d6 = st.columns(6)
        aud_c = kpis.get("audit_counts", {})
        m_d1.metric("Verified Papers", f"{aud_c.get('VERIFIED', 0)}")
        m_d2.metric("Flagged Warnings", f"{aud_c.get('FLAGGED_WARNING', 0)}")
        m_d3.metric("Rejected / Outreach", f"{aud_c.get('REJECTED', 0) + aud_c.get('NEEDS_OUTREACH', 0)}")
        concordance_disp = f"{delib['paired_concordance_rate']:.1f}%" if delib['paired_concordance_rate'] is not None else "N/A"
        m_d4.metric("Concordance Rate", concordance_disp)
        m_d5.metric("Total Disputes", f"{delib['total_disputes']}")
        m_d6.metric("Runtime", f"{kpis['total_elapsed_seconds']/60.0:.1f} min")

        st.markdown("---")

        # Papers Table with Interactive Filters
        st.subheader("📑 Evaluated Papers Overview")
        papers_data = analysis["papers"]
        papers_df = pd.DataFrame(papers_data)

        col_f1, col_f2 = st.columns([2, 2])
        with col_f1:
            status_filter = st.multiselect(
                "Filter by Audit Status",
                options=["VERIFIED", "FLAGGED_WARNING", "NEEDS_OUTREACH", "REJECTED"],
                default=["VERIFIED", "FLAGGED_WARNING", "NEEDS_OUTREACH", "REJECTED"],
            )
        with col_f2:
            search_query = st.text_input("Search Paper Model Name", "")

        filtered_df = papers_df[papers_df["audit_status"].isin(status_filter)]
        if search_query.strip():
            filtered_df = filtered_df[filtered_df["model_id"].str.contains(search_query.strip(), case=False)]

        st.dataframe(filtered_df, use_container_width=True, height=280)

        st.markdown("---")

        # Per-Paper Deep Drilldown
        st.subheader("🔬 Paper Artifact & Deliberation Drill-Down")
        paper_names = [p["model_id"] for p in papers_data]
        selected_model_id = st.selectbox("Select Paper Model to Inspect", paper_names)

        detail = load_paper_detail(selected_run_dir, selected_model_id)

        if not detail["found"]:
            st.info(f"Intermediate step artifacts not stored or archived for `{selected_model_id}`.")
        else:
            d_tab_scores, d_tab_cot, d_tab_delib, d_tab_artifacts = st.tabs([
                "📊 Score Alignment & Ground Truth",
                "🧠 Chain-of-Thought (CoT)",
                "🤝 Deliberation Trajectory",
                "📦 Audit, SQL & Outreach",
            ])

            with d_tab_scores:
                st.markdown("#### Score Alignment Table")
                consensus = detail.get("consensus_matrix")
                eval_data = detail.get("evaluation")

                if consensus and "consensus_scores" in consensus:
                    scores_list = []
                    for s in consensus.get("consensus_scores", []):
                        scores_list.append({
                            "Dataset": s.get("canonical_dataset_name") or s.get("dataset_name"),
                            "Organ/Task": s.get("organ_or_task", "Overall"),
                            "Agent A Score": s.get("score_a"),
                            "Agent B Score": s.get("score_b"),
                            "Status": s.get("status"),
                            "Resolved Score": s.get("resolved_score"),
                            "Prompt Type": s.get("prompt_type"),
                        })
                    st.dataframe(pd.DataFrame(scores_list), use_container_width=True)
                elif eval_data and "per_dataset" in eval_data:
                    st.json(eval_data.get("per_dataset"))
                else:
                    st.caption("No tabular score alignment matrix available for this paper.")

            with d_tab_cot:
                st.markdown("#### Side-by-Side Agent Chain-of-Thought Reasoning")
                c_a, c_b = st.columns(2)
                with c_a:
                    st.markdown("**Agent A Extracted CoT**")
                    data_a = detail.get("agent_a_extracted") or {}
                    st.text_area(
                        "Agent A Thought Process",
                        value=data_a.get("thought_process") or data_a.get("extraction_notes") or "No CoT recorded.",
                        height=350,
                        key=f"cot_a_{selected_model_id}",
                    )
                with c_b:
                    st.markdown("**Agent B Extracted CoT**")
                    data_b = detail.get("agent_b_extracted") or {}
                    st.text_area(
                        "Agent B Thought Process",
                        value=data_b.get("thought_process") or data_b.get("extraction_notes") or "No CoT recorded.",
                        height=350,
                        key=f"cot_b_{selected_model_id}",
                    )

            with d_tab_delib:
                st.markdown("#### Iterative Deliberation Trajectory")
                traj = detail.get("deliberation_trajectory")
                if traj and isinstance(traj, list) and len(traj) > 0:
                    for rnd in traj:
                        r_num = rnd.get("round_number", 1)
                        st.markdown(f"##### 🔄 Round {r_num}")
                        st.caption(
                            f"Disputes: {rnd.get('disputes_before')} -> {rnd.get('disputes_after')} | "
                            f"Unconfirmed: {rnd.get('unconfirmed_before')} -> {rnd.get('unconfirmed_after')} | "
                            f"Concordance: {rnd.get('concordance_rate_after', 0)*100:.1f}%"
                        )
                        details_list = rnd.get("details", [])
                        if details_list:
                            for d_line in details_list:
                                st.markdown(f"- {d_line}")
                        else:
                            st.caption("Epistemic equilibrium reached. No modifications.")
                else:
                    st.caption("Full consensus achieved in Round 1 or single-agent run (zero deliberation rounds needed).")

            with d_tab_artifacts:
                st.markdown("#### Downstream Curation, SQL & Outreach Artifacts")
                c_sql, c_outreach = st.columns(2)
                with c_sql:
                    st.markdown("**SQLite Curation SQL**")
                    st.code(detail.get("curation_sql") or "-- No SQL patch generated", language="sql")
                with c_outreach:
                    st.markdown("**Author Outreach Inquiry Draft**")
                    st.markdown(detail.get("outreach_markdown") or "*No author outreach draft required for this model.*")


# ------------------------------------------------------------------------------
# TAB 3: MULTI-RUN COMPARATOR & THESIS EXPORTER
# ------------------------------------------------------------------------------
with tab_compare:
    st.header("⚔️ Multi-Run Performance Comparator & Publication Exporter")
    st.markdown(
        "Compare 2 to N benchmark runs side-by-side. Inspect Pareto efficiency frontiers, "
        "cross-run discrepancy heatmaps, and export publication-ready LaTeX tables."
    )

    if len(all_runs) < 2:
        st.warning("At least 2 benchmark runs are required for cross-run comparison.")
    else:
        # Run Selection
        run_name_map = {r["run_dir"]: f"{r['timestamp']} | {r['run_id']}" for r in all_runs}
        all_dirs = list(run_name_map.keys())

        # Quick Preset Buttons
        st.markdown("**Quick Comparison Presets**")
        p_c1, p_c2, p_c3 = st.columns(3)
        preset_runs = None

        if p_c1.button("🏆 5-Model Benchmark Suite (40 Papers)"):
            # Discover runs matching the 5 models
            p_selected = []
            for target_substr in ["Mistral7B", "Codestral", "Qwen14B", "MetaLlama", "Qwen257B"]:
                for r in all_runs:
                    if target_substr.lower() in r["run_id"].lower() and r["total_papers"] >= 35:
                        if r["run_dir"] not in p_selected:
                            p_selected.append(r["run_dir"])
                            break
            if len(p_selected) >= 2:
                preset_runs = p_selected

        if p_c2.button("👥 Dual-Local vs. Single-Agent"):
            d_runs = [r["run_dir"] for r in all_runs if r["is_dual"] and r["total_papers"] >= 35]
            s_runs = [r["run_dir"] for r in all_runs if not r["is_dual"] and r["total_papers"] >= 35]
            if d_runs and s_runs:
                preset_runs = [d_runs[0], s_runs[0]]

        default_selection = preset_runs or all_dirs[: min(4, len(all_dirs))]

        selected_compare_dirs = st.multiselect(
            "Select Runs to Compare (2 or more)",
            options=all_dirs,
            default=default_selection,
            format_func=lambda d: run_name_map[d],
        )

        if len(selected_compare_dirs) < 2:
            st.info("Please select at least 2 runs above to generate the comparison matrix.")
        else:
            # Custom display labels
            with st.expander("🏷️ Edit Run Display Labels", expanded=False):
                custom_labels = []
                cols = st.columns(len(selected_compare_dirs))
                for idx, r_dir in enumerate(selected_compare_dirs):
                    r_info = next(r for r in all_runs if r["run_dir"] == r_dir)
                    m_a = (r_info.get("model_a") or "").split("/")[-1]
                    m_b = (r_info.get("model_b") or "").split("/")[-1]
                    sugg = f"{m_a} + {m_b}" if m_b else m_a
                    lbl = cols[idx].text_input(f"Run {idx+1} Label", value=sugg or f"Run {idx+1}", key=f"lbl_{idx}")
                    custom_labels.append(lbl)

            metric_choice = st.selectbox(
                "Matrix Metric for Paper-by-Paper Comparison",
                options=["dice_mae", "recall", "audit_status", "confidence", "matched_datasets", "elapsed_seconds", "concordance_rate", "disputes"],
                index=0,
                format_func=lambda m: {
                    "dice_mae": "Dice MAE (Lower is Better)",
                    "recall": "Dataset Recall % (Higher is Better)",
                    "audit_status": "Audit Classification Status",
                    "confidence": "Extraction Confidence Score %",
                    "matched_datasets": "Matched Benchmarks Ratio",
                    "elapsed_seconds": "Elapsed Extraction Time (s)",
                    "concordance_rate": "Inter-Model Concordance %",
                    "disputes": "Number of Deliberation Disputes",
                }.get(m, m),
            )

            filter_diff_only = st.checkbox("Show only papers with cross-run discrepancies", value=False)

            # Generate Comparison Data
            comparison = compare_runs(
                run_dirs=selected_compare_dirs,
                custom_labels=custom_labels,
                metric=metric_choice,
            )

            kpi_df = comparison["kpi_df"]
            paper_df = comparison["paper_df"]
            labels = comparison["labels"]

            # 1. Aggregate KPI Comparison Table
            st.subheader("📊 Aggregate KPI Comparison Table")
            st.dataframe(kpi_df, use_container_width=True)

            st.markdown("---")

            # 2. Plotly Visualizations
            st.subheader("📈 Interactive Comparison Visualizations")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Pareto Frontier: Runtime vs MAE
                scatter_data = []
                for label, analysis in zip(labels, comparison["analyses"]):
                    k = analysis["kpis"]
                    scatter_data.append({
                        "Model Configuration": label,
                        "Avg Runtime per Paper (s)": k["avg_elapsed_seconds"],
                        "Macro Dice MAE": k["macro_mae"],
                        "Dataset Recall (%)": k["overall_recall"],
                        "Coverage Score": k["coverage_adjusted_score"],
                    })
                sc_df = pd.DataFrame(scatter_data)
                fig_pareto = px.scatter(
                    sc_df,
                    x="Avg Runtime per Paper (s)",
                    y="Macro Dice MAE",
                    size="Coverage Score",
                    color="Model Configuration",
                    hover_name="Model Configuration",
                    text="Model Configuration",
                    title="Pareto Frontier: Inference Speed vs. Error (Macro Dice MAE)",
                )
                fig_pareto.update_traces(textposition="top center")
                fig_pareto.update_layout(yaxis=dict(autorange="reversed")) # lower MAE is top
                st.plotly_chart(fig_pareto, use_container_width=True)

            with chart_col2:
                # Audit Status Breakdown
                audit_breakdown = []
                for label, analysis in zip(labels, comparison["analyses"]):
                    aud = analysis["kpis"].get("audit_counts", {})
                    for status, count in aud.items():
                        if status in ("VERIFIED", "FLAGGED_WARNING", "REJECTED", "NEEDS_OUTREACH"):
                            audit_breakdown.append({
                                "Configuration": label,
                                "Status": status,
                                "Paper Count": count,
                            })
                ab_df = pd.DataFrame(audit_breakdown)
                fig_bar = px.bar(
                    ab_df,
                    x="Configuration",
                    y="Paper Count",
                    color="Status",
                    title="Audit Classification Distribution Across Configurations",
                    barmode="stack",
                    color_discrete_map={
                        "VERIFIED": "#2ca02c",
                        "FLAGGED_WARNING": "#ff7f0e",
                        "NEEDS_OUTREACH": "#1f77b4",
                        "REJECTED": "#d62728",
                    },
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Per-Paper Error Distribution Boxplot
            box_data = []
            for label, analysis in zip(labels, comparison["analyses"]):
                for p in analysis["papers"]:
                    box_data.append({
                        "Configuration": label,
                        "Dice MAE": p["dice_mae"],
                        "Model ID": p["model_id"],
                    })
            if box_data:
                fig_box = px.box(
                    pd.DataFrame(box_data),
                    x="Configuration",
                    y="Dice MAE",
                    points="all",
                    hover_data=["Model ID"],
                    title="Distribution of Per-Paper Dice MAE Errors",
                )
                st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("---")

            # 3. Paper-by-Paper Cross-Run Matrix
            st.subheader(f"📑 Paper-by-Paper Matrix (`{metric_choice}`)")
            disp_matrix = paper_df.copy()
            if filter_diff_only and "Has Discrepancy" in disp_matrix.columns:
                disp_matrix = disp_matrix[disp_matrix["Has Discrepancy"]]
                st.caption(f"Displaying {len(disp_matrix)} papers with discrepant `{metric_choice}` values across selected runs.")
            else:
                st.caption(f"Displaying all {len(disp_matrix)} papers in union of selected runs.")

            st.dataframe(disp_matrix, use_container_width=True)

            st.markdown("---")

            # 4. Publication-Ready Exports
            st.subheader("📄 1-Click Thesis Exports (LaTeX & Markdown)")

            exp_tab_tex, exp_tab_md, exp_tab_csv = st.tabs([
                "📑 LaTeX Booktabs (.tex)",
                "📝 Markdown Synthesis (.md)",
                "📊 CSV Export (.csv)",
            ])

            with exp_tab_tex:
                latex_code = comparison["latex_table"]
                st.code(latex_code, language="latex")
                st.download_button(
                    label="⬇️ Download LaTeX Table (.tex)",
                    data=latex_code,
                    file_name="gmmis_multi_benchmark_comparison.tex",
                    mime="text/plain",
                )

            with exp_tab_md:
                md_report = comparison["markdown_report"]
                st.text_area("Markdown Report Preview", value=md_report, height=300)
                st.download_button(
                    label="⬇️ Download Markdown Report (.md)",
                    data=md_report,
                    file_name="gmmis_multi_benchmark_comparison.md",
                    mime="text/markdown",
                )

            with exp_tab_csv:
                csv_data = kpi_df.to_csv()
                st.download_button(
                    label="⬇️ Download Aggregate KPIs (.csv)",
                    data=csv_data,
                    file_name="gmmis_kpi_comparison.csv",
                    mime="text/csv",
                )
                csv_matrix = paper_df.to_csv()
                st.download_button(
                    label="⬇️ Download Paper Matrix (.csv)",
                    data=csv_matrix,
                    file_name="gmmis_paper_matrix.csv",
                    mime="text/csv",
                )

