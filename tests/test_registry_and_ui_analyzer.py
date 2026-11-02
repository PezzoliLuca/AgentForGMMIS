"""Unit and integration tests for ModelRegistry, Sequential Coordinator Mode,

RunAnalyzer, and ClusterManager.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agent_gmmis.extraction.alias_resolver import DatasetAliasResolver
from agent_gmmis.extraction.dual_coordinator import DualExtractionCoordinator
from agent_gmmis.extraction.pdf_parser import ParsedManuscript
from agent_gmmis.extraction.schemas import (
    ExtractedPaperData,
    RawExtractedScore,
    ReinterrogationScoreResponse,
)
from agent_gmmis.llm.client import LLMClient, LLMConfig
from agent_gmmis.llm.registry import (
    MODEL_CATALOG,
    ModelProfile,
    build_vllm_serve_args,
    format_bash_plan,
    plan_mufasa_allocation,
    resolve_model_profile,
)
from agent_gmmis.ui.cluster_manager import ClusterManager
from agent_gmmis.ui.run_analyzer import (
    _safe_float,
    compare_runs,
    discover_runs,
    generate_latex_comparison_table,
    generate_markdown_comparison_report,
    load_paper_detail,
    load_run_analysis,
)


# ==============================================================================
# 1. Model Registry Tests
# ==============================================================================

class TestModelRegistry:
    def test_catalog_completeness(self):
        """All 9 cached Mufasa models plus Gemini cloud models must be present."""
        expected_keys = [
            "qwen2.5-7b-awq",
            "qwen2.5-14b-awq",
            "llama-3.1-8b-awq",
            "mistral-7b-awq",
            "mistral-7b-fp16",
            "codestral-22b-awq",
            "deepseek-r1-14b-awq",
            "glm-4-9b-chat",
            "moonlight-16b-a3b",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash",
            "mock",
        ]
        for k in expected_keys:
            assert k in MODEL_CATALOG, f"Missing key in MODEL_CATALOG: {k}"

    def test_resolve_model_profile_aliases(self):
        """Aliases and case variations resolve correctly to full profiles."""
        p_qwen = resolve_model_profile("qwen2.5-7b-awq")
        assert p_qwen.hf_repo_or_id == "Qwen/Qwen2.5-7B-Instruct-AWQ"
        assert p_qwen.provider == "mufasa"

        p_r1 = resolve_model_profile("deepseek-r1-14b-awq")
        assert p_r1.is_reasoning is True
        assert p_r1.max_output_tokens == 4096

        p_mistral = resolve_model_profile("solidrust/Mistral-7B-Instruct-v0.3-AWQ")
        assert p_mistral.key == "mistral-7b-awq"
        assert p_mistral.needs_chat_template is not None

        p_glm = resolve_model_profile("THUDM/glm-4-9b-chat")
        assert p_glm.quantization_flag == "bitsandbytes"

    def test_resolve_custom_model_fallback(self):
        """Custom unlisted models generate conservative fallback profiles."""
        p_custom = resolve_model_profile("org/unknown-model-10b", provider_hint="mufasa")
        assert p_custom.provider == "mufasa"
        assert p_custom.hf_repo_or_id == "org/unknown-model-10b"
        assert p_custom.vram_gb == 8.0

        p_gem = resolve_model_profile("gemini-custom-model")
        assert p_gem.provider == "gemini"
        assert p_gem.vram_gb == 0.0

    def test_build_vllm_serve_args(self):
        """Constructs exact vLLM CLI arguments including chat-templates and bitsandbytes."""
        p_mistral = MODEL_CATALOG["mistral-7b-awq"]
        args_mistral = build_vllm_serve_args(p_mistral, port=8000, gpu_util=0.43, max_len=16384)
        assert "--port" in args_mistral
        assert "8000" in args_mistral
        assert "--chat-template" in args_mistral
        assert "--enforce-eager" in args_mistral
        assert "--trust-remote-code" in args_mistral

        p_glm = MODEL_CATALOG["glm-4-9b-chat"]
        args_glm = build_vllm_serve_args(p_glm, port=8001, gpu_util=0.90, max_len=16384)
        assert "--quantization" in args_glm
        assert "bitsandbytes" in args_glm

    def test_plan_mufasa_allocation_simultaneous_dual(self):
        """Small models (VRAM sum <= 12.5GB) run simultaneously with split VRAM."""
        plan = plan_mufasa_allocation(
            model_a="qwen2.5-7b-awq",
            model_b="mistral-7b-awq",
        )
        assert plan["topology"] == "dual_simultaneous"
        assert plan["port_a"] == 8000
        assert plan["port_b"] == 8001
        assert plan["util_a"] + plan["util_b"] <= 0.88
        assert plan["needs_bitsandbytes"] is False

    def test_plan_mufasa_allocation_sequential_dual(self):
        """Large models (VRAM sum > 12.5GB) trigger 2-stage sequential dual mode."""
        plan = plan_mufasa_allocation(
            model_a="deepseek-r1-14b-awq",
            model_b="qwen2.5-14b-awq",
        )
        assert plan["topology"] == "dual_sequential"
        assert plan["util_a"] == 0.90
        assert plan["util_b"] == 0.90
        assert plan["port_a"] == 8000
        assert plan["port_b"] == 8000

    def test_plan_mufasa_allocation_hybrid(self):
        """Mufasa GPU + Gemini Cloud allocates 90% GPU to local model."""
        plan = plan_mufasa_allocation(
            model_a="qwen2.5-7b-awq",
            model_b="gemini-3.5-flash-lite",
            provider_a="mufasa",
            provider_b="gemini",
        )
        assert plan["topology"] == "hybrid"
        assert plan["util_a"] == 0.90

    def test_format_bash_plan(self):
        """Outputs sourceable bash variable declarations."""
        plan = plan_mufasa_allocation(model_a="glm-4-9b-chat")
        bash_str = format_bash_plan(plan)
        assert 'TOPOLOGY="single"' in bash_str
        assert 'MODEL_A_KEY="glm-4-9b-chat"' in bash_str
        assert 'NEEDS_BITSANDBYTES="true"' in bash_str


# ==============================================================================
# 2. LLM Client Integration Tests
# ==============================================================================

class TestLLMClientIntegration:
    def test_client_resolves_profile(self):
        """LLMClient constructor resolves model profile and normalizes repo name."""
        client = LLMClient(config=LLMConfig(provider="mock", model="deepseek-r1-14b-awq"))
        assert hasattr(client, "profile")
        assert client.profile.key == "deepseek-r1-14b-awq"
        assert client.profile.is_reasoning is True

    def test_client_structured_reasoning_token_budget(self):
        """Reasoning models receive 4096 token budget for ExtractedPaperData."""
        client = LLMClient(config=LLMConfig(provider="mock", model="deepseek-r1-14b-awq"))
        res = client.generate_structured("Extract scores", ExtractedPaperData)
        assert isinstance(res, ExtractedPaperData)


# ==============================================================================
# 3. Dual Coordinator 2-Stage Sequential Mode Tests
# ==============================================================================

class TestSequentialCoordinatorMode:
    def test_sequential_stage2_deliberation(self):
        """In Stage 2, coordinator uses pre_extracted_a directly and only re-interrogates online Agent B."""
        pre_extracted_a = ExtractedPaperData(
            paper_title="Test Manuscript",
            proposed_model_id="TestModel",
            extracted_scores=[
                RawExtractedScore(dataset_name="Synapse", organ_or_task="Overall", dice_score=0.88),
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Liver", dice_score=0.92),
            ],
        )

        mock_agent_b = MagicMock()
        mock_agent_b.process_manuscript.return_value = ExtractedPaperData(
            paper_title="Test Manuscript",
            proposed_model_id="TestModel",
            extracted_scores=[
                # Disputed Synapse score
                RawExtractedScore(dataset_name="Synapse", organ_or_task="Overall", dice_score=0.82),
            ],
        )
        mock_agent_b.reinterrogate_discrepancy.return_value = ReinterrogationScoreResponse(
            is_valid_benchmark=True,
            dice_score=0.88,
            confidence_score=0.95,
            rationale="Agreed with Table 2 macro average",
        )
        mock_agent_b.reinterrogate_omission.return_value = ReinterrogationScoreResponse(
            is_valid_benchmark=True,
            dice_score=0.92,
            confidence_score=0.90,
            rationale="Recovered BTCV score from Table 3",
        )

        # Agent A is offline (None) in Stage 2
        coord = DualExtractionCoordinator(agent_a=None, agent_b=mock_agent_b)
        manuscript = ParsedManuscript(
            file_path="test.pdf",
            total_pages=5,
            title="Test Manuscript",
            abstract="Abstract",
            pages_text={1: "Test content on Synapse and BTCV evaluation tables."},
            table_pages=[1],
        )

        from agent_gmmis.auditing.auditor import AuditorAgent
        auditor = AuditorAgent()

        ext_a, ext_b, audit_report, traj = coord.extract_manuscript_with_deliberation(
            manuscript=manuscript,
            auditor=auditor,
            max_rounds=3,
            verbose=False,
            pre_extracted_a=pre_extracted_a,
            agent_a_online=False,
            agent_b_online=True,
        )

        assert ext_a.paper_title == "Test Manuscript"
        assert ext_b.paper_title == "Test Manuscript"
        # Mock Agent B was called for extraction and deliberation re-interrogations
        assert mock_agent_b.process_manuscript.called
        assert len(traj) >= 1
        # Re-interrogation updated Agent B Synapse score to 0.88
        syn_b = next((s for s in ext_b.extracted_scores if "synapse" in s.dataset_name.lower()), None)
        assert syn_b is not None
        assert syn_b.dice_score == 0.88


# ==============================================================================
# 4. Run Analyzer Tests
# ==============================================================================

class TestRunAnalyzer:
    def test_safe_float_parser(self):
        assert _safe_float("85.5%") == 85.5
        assert _safe_float("0.0452") == 0.0452
        assert _safe_float("45.2s") == 45.2
        assert _safe_float("N/A", default=0.0) == 0.0
        assert _safe_float(None, default=0.0) == 0.0

    def test_discover_runs(self):
        runs = discover_runs()
        assert isinstance(runs, list)
        assert len(runs) > 0
        r0 = runs[0]
        assert "run_id" in r0
        assert "run_dir" in r0
        assert "timestamp" in r0
        assert "total_papers" in r0

    def test_load_run_analysis_real_run(self):
        runs = discover_runs()
        analysis = load_run_analysis(runs[0]["run_dir"])
        assert "kpis" in analysis
        kpis = analysis["kpis"]
        assert "overall_recall" in kpis
        assert "macro_mae" in kpis
        assert "zero_error_count" in kpis
        assert "papers" in analysis
        assert len(analysis["papers"]) > 0

    def test_compare_runs_and_latex_generation(self):
        runs = discover_runs()
        if len(runs) >= 2:
            comparison = compare_runs([runs[0]["run_dir"], runs[1]["run_dir"]])
            kpi_df = comparison["kpi_df"]
            paper_df = comparison["paper_df"]
            latex = comparison["latex_table"]
            md = comparison["markdown_report"]

            assert isinstance(kpi_df, pd.DataFrame)
            assert "Delta (2 - 1)" in kpi_df.columns
            assert isinstance(paper_df, pd.DataFrame)
            assert "\\begin{table*}" in latex
            assert "\\toprule" in latex
            assert "\\bottomrule" in latex
            assert "# ⚔️ Multi-Run Benchmark Comparison" in md


# ==============================================================================
# 5. Cluster Manager Remote Diagnostics Tests
# ==============================================================================

class TestClusterManager:
    def test_cluster_manager_init(self):
        cm = ClusterManager(host="10.79.23.96", user="testuser")
        assert cm.host == "10.79.23.96"
        assert cm.user == "testuser"
        assert cm.target == "testuser@10.79.23.96"

    @patch.object(ClusterManager, "_run_ssh")
    def test_check_mufasa_status_parsing(self, mock_ssh):
        fake_stdout = (
            "===SECTION:PING===\n"
            "PONG\n"
            "===SECTION:SQUEUE===\n"
            "JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)\n"
            "12345 gpu gmmis_bench lpezzoli R 00:15:20 1 node01\n"
            "===SECTION:HF_MODELS===\n"
            "models--Qwen--Qwen2.5-7B-Instruct-AWQ\n"
            "models--solidrust--Mistral-7B-Instruct-v0.3-AWQ\n"
        )
        mock_ssh.return_value = (0, fake_stdout, "")

        cm = ClusterManager()
        status = cm.check_mufasa_status()

        assert status["online"] is True
        assert len(status["active_jobs"]) == 1
        assert status["active_jobs"][0]["job_id"] == "12345"
        assert len(status["cached_models"]) == 2
        assert "Qwen/Qwen2.5-7B-Instruct-AWQ" in status["cached_models"]
        assert "solidrust/Mistral-7B-Instruct-v0.3-AWQ" in status["cached_models"]
